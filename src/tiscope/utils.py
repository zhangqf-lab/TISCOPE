"""
Utility functions for TISCOPE.
"""

import gc
import time
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests


def convert_to_dense(X):
    """
    Converts a sparse matrix or AnnData view to a dense NumPy array.

    Args:
        X (np.ndarray, scipy.sparse matrix, or anndata.View):
            Input data structure.

    Returns:
        np.ndarray: Dense representation of the input.

    Raises:
        TypeError: If the input type is not supported.
        ValueError: If the resulting dense matrix contains non-finite values.
    """
    if isinstance(X, np.ndarray):
        return X
    elif issparse(X) or str(type(X)).endswith("ArrayView'>"):
        X_dense = X.toarray()
    else:
        raise TypeError(f"Input format not recognized. Must be dense/sparse matrix or AnnData view. Got {type(X)}")

    if not np.isfinite(X_dense).all():
        raise ValueError("Matrix contains non-finite values (NaN or Inf).")
    return X_dense


def compute_neighbor_average(adata: AnnData, adjacency_matrix):
    """
    Computes the average feature value of neighbors for each node.

    This is done by multiplying the adjacency matrix (weights)
    with the feature matrix (X).

    Args:
        adata (AnnData): AnnData object containing the feature matrix `adata.X`.
        adjacency_matrix (scipy.sparse matrix or np.ndarray):
            Adjacency matrix where element (i, j) represents the weight
            of the edge from node i to node j.

    Returns:
        np.ndarray: A dense matrix where each row represents the
                    neighbor-averaged features for the corresponding node.
    """
    start_time = time.perf_counter()
    X_dense = convert_to_dense(adata.X)
    # Matrix multiplication: (N x N) @ (N x F) -> (N x F)
    # Each row i in the result is the sum of feature vectors of i's neighbors,
    # weighted by the adjacency matrix.
    neighbor_avg = adjacency_matrix @ X_dense
    gc.collect()  # Suggest garbage collection
    elapsed = round((time.perf_counter() - start_time) / 60, 2)
    print(f"Computed neighbor average in {elapsed} mins")
    return neighbor_avg


def overcorrection_score(emb, celltype, n_neighbors=100, n_samples=None, min_cells_per_type=5, random_state=42, weighted=False):
    """
    Improved overcorrection score that evaluates whether integration has over-mixed
    biological cell types while removing batch effects.

    Parameters:
    -----------
    emb : array-like, shape (n_cells, n_features)
        Embedding coordinates (e.g., UMAP, PCA)
    celltype : array-like, shape (n_cells,)
        Cell type labels for each cell
    n_neighbors : int, optional (default=30)
        Number of neighbors to consider for each cell
    n_samples : int, optional (default=1000)
        Number of cells to sample for estimation (use None for all cells)
    min_cells_per_type : int, optional (default=5)
        Minimum number of cells required for a cell type to be included in scoring
    random_state : int, optional (default=42)
        Random seed for reproducibility
    weighted : bool, optional (default=True)
        Whether to weight scores by cell type prevalence

    Returns:
    --------
    score : float
        Overcorrection score (higher values indicate more overcorrection)
    celltype_scores : dict
        Dictionary with scores for each cell type
    """

    # Input validation
    if len(emb) != len(celltype):
        raise ValueError("emb and celltype must have the same length")

    if n_neighbors >= len(emb):
        warnings.warn(f"n_neighbors ({n_neighbors}) is too large for dataset size ({len(emb)}). Reducing to {len(emb) - 1}")
        n_neighbors = len(emb) - 1

    # Convert to numpy arrays
    emb = np.asarray(emb)
    celltype = np.asarray(celltype)

    # Get unique cell types and their counts
    unique_types, type_counts = np.unique(celltype, return_counts=True)

    # Filter out cell types with too few cells
    valid_types = unique_types[type_counts >= min_cells_per_type]
    if len(valid_types) < 2:
        raise ValueError(f"Need at least 2 cell types with ≥ {min_cells_per_type} cells each")

    # Create a mask for valid cells
    valid_mask = np.isin(celltype, valid_types)

    if not np.all(valid_mask):
        warnings.warn(f"Ignoring {np.sum(~valid_mask)} cells from rare cell types (< {min_cells_per_type} cells)")

    # Subset to valid cells
    emb_valid = emb[valid_mask]
    celltype_valid = celltype[valid_mask]

    # Build nearest neighbors graph
    nne = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(emb_valid)), n_jobs=-1)  # Use all available cores
    nne.fit(emb_valid)
    kmatrix = nne.kneighbors_graph(emb_valid, mode="connectivity")

    # Remove self-connections
    kmatrix = kmatrix - sparse.identity(kmatrix.shape[0])

    if n_samples is not None and n_samples < len(emb_valid):
        rng = np.random.RandomState(random_state)

    # Calculate scores per cell type
    celltype_scores = {}
    for ct in valid_types:
        # Get indices of cells of this type
        ct_indices = np.where(celltype_valid == ct)[0]

        # Sample from this cell type if needed
        if n_samples is not None:
            n_sample_ct = max(1, int(n_samples * len(ct_indices) / len(emb_valid)))
            if n_sample_ct < len(ct_indices):
                ct_sample_indices = rng.choice(ct_indices, size=n_sample_ct, replace=False)
            else:
                ct_sample_indices = ct_indices
        else:
            ct_sample_indices = ct_indices

        # Calculate average same-type proportion for this cell type
        same_type_props = []
        for i in ct_sample_indices:
            # Get neighbors (excluding self)
            neighbors = kmatrix[i].nonzero()[1]

            # Calculate proportion of same-type neighbors
            same_type_count = np.sum(celltype_valid[neighbors] == celltype_valid[i])
            same_type_prop = same_type_count / len(neighbors) if len(neighbors) > 0 else 0
            same_type_props.append(same_type_prop)

        celltype_scores[ct] = np.mean(same_type_props) if same_type_props else 0

    # Calculate overall score
    if weighted:
        # Weight by cell type prevalence
        weights = [type_counts[unique_types == ct][0] for ct in valid_types]
        overall_score = 1 - np.average(list(celltype_scores.values()), weights=weights)
    else:
        # Simple average across cell types
        overall_score = 1 - np.mean(list(celltype_scores.values()))

    return overall_score


def calculate_tm_enrichment(adata, sample_type_col="Sample_type", batch_col="batch", cc_col="TM"):
    """
    Compute TM enrichment scores per sample type using AUC (Mann–Whitney U).
    Returns a dict of DataFrames keyed by Sample_type.
    """
    # Extract required columns
    obs_df = adata.obs[[sample_type_col, batch_col, cc_col]].copy()
    obs_df.columns = ["Sample_type", "batch", "cc"]

    # Cell counts per batch and TM
    count_df = obs_df.groupby(["batch", "cc"]).size().unstack(fill_value=0)

    # Proportions per batch
    sample_totals = count_df.sum(axis=1)
    prop_df = count_df.div(sample_totals, axis=0)

    # Attach sample type
    sample_conditions = obs_df[["batch", "Sample_type"]].drop_duplicates().set_index("batch")["Sample_type"]
    prop_df["Sample_type"] = sample_conditions

    results_dict = {}
    conditions = prop_df["Sample_type"].unique()
    tms = count_df.columns

    # Iterate over conditions
    for c in conditions:
        scores, auc_values, p_values, tm_names = [], [], [], []

        # Iterate over TMs
        for t in tms:
            prop_c = prop_df[prop_df["Sample_type"] == c][t]
            prop_other = prop_df[prop_df["Sample_type"] != c][t]

            # Handle edge cases
            if len(prop_c) == 0 or len(prop_other) == 0:
                auc, p_value = 0.5, 1.0
            else:
                try:
                    u_stat, p_value = mannwhitneyu(prop_c, prop_other, alternative="two-sided")
                    auc = u_stat / (len(prop_c) * len(prop_other))
                except Exception:
                    auc, p_value = 0.5, 1.0

            # Enrichment score centered at 0
            score = auc - 0.5

            tm_names.append(t)
            scores.append(score)
            auc_values.append(auc)
            p_values.append(p_value)

        # FDR correction
        _, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

        # Build result table
        result_df = (
            pd.DataFrame(
                {
                    "TM": tm_names,
                    "score": scores,
                    "AUC": auc_values,
                    "p_value": p_values,
                    "p_adjusted": p_adjusted,
                }
            )
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        results_dict[c] = result_df

    return results_dict

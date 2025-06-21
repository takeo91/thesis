"""Retrieval-style evaluation helpers.

This module provides common ranking metrics for *query vs. library* fuzzy-set
similarity experiments.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np

__all__ = [
    "rank_library_by_similarity",
    "compute_hit_at_k",
    "compute_mean_reciprocal_rank",
    "compute_retrieval_metrics",
]


def rank_library_by_similarity(sim_matrix: np.ndarray) -> np.ndarray:
    """Return indices of *library* windows sorted by descending similarity.

    Parameters
    ----------
    sim_matrix : np.ndarray, shape (n_query, n_library)
        Similarity scores where larger means more similar.

    Returns
    -------
    np.ndarray, shape (n_query, n_library)
        Each row contains column indices sorted from most to least similar.
    """
    return np.argsort(-sim_matrix, axis=1)


def _first_match_rank(sorted_indices: Sequence[int], query_label, library_labels: Sequence) -> int | None:
    """Return 1-based rank of first library item whose label matches the query.

    Returns ``None`` if no match is found.
    """
    for rank, lib_idx in enumerate(sorted_indices, 1):
        if library_labels[lib_idx] == query_label:
            return rank
    return None


def compute_hit_at_k(
    sim_matrix: np.ndarray,
    query_labels: Sequence,
    library_labels: Sequence,
    k: int = 1,
) -> float:
    """Compute Hit@k.

    Hit@k is the proportion of queries where at least one *correct* item (same
    label) is present in the top-*k* most similar library windows.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    topk_idx = rank_library_by_similarity(sim_matrix)[:, :k]
    hits = 0
    for q_idx, row in enumerate(topk_idx):
        q_label = query_labels[q_idx]
        if any(library_labels[j] == q_label for j in row):
            hits += 1
    return hits / len(query_labels)


def compute_mean_reciprocal_rank(
    sim_matrix: np.ndarray,
    query_labels: Sequence,
    library_labels: Sequence,
) -> float:
    """Compute Mean Reciprocal Rank (MRR)."""
    sorted_idx = rank_library_by_similarity(sim_matrix)
    reciprocal_ranks: List[float] = []
    for q_idx, row in enumerate(sorted_idx):
        rank = _first_match_rank(row, query_labels[q_idx], library_labels)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return float(np.mean(reciprocal_ranks))


def compute_retrieval_metrics(
    sim_matrix: np.ndarray,
    query_labels: Sequence,
    library_labels: Sequence,
    *,
    topk: int = 5,
) -> Dict[str, float]:
    """Return a dictionary with common retrieval metrics."""
    hit1 = compute_hit_at_k(sim_matrix, query_labels, library_labels, k=1)
    hitk = compute_hit_at_k(sim_matrix, query_labels, library_labels, k=topk)
    mrr = compute_mean_reciprocal_rank(sim_matrix, query_labels, library_labels)
    return {
        "hit@1": hit1,
        f"hit@{topk}": hitk,
        "mrr": mrr,
    } 
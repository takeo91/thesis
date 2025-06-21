import numpy as np
from thesis.exp.retrieval_utils import (
    rank_library_by_similarity,
    compute_hit_at_k,
    compute_mean_reciprocal_rank,
    compute_retrieval_metrics,
)


def _toy_similarity_matrix():
    # 3 queries, 4 library items, labels such that each query label appears in library
    sim = np.array([
        [0.9, 0.1, 0.2, 0.3],  # query 0 most similar to lib 0 (label 0)
        [0.2, 0.8, 0.3, 0.4],  # query 1 most similar to lib 1 (label 1)
        [0.1, 0.2, 0.7, 0.6],  # query 2 most similar to lib 2 (label 2)
    ])
    query_labels = np.array([0, 1, 2])
    library_labels = np.array([0, 1, 2, 0])
    return sim, query_labels, library_labels


def test_rank_library():
    sim, _, _ = _toy_similarity_matrix()
    ranks = rank_library_by_similarity(sim)
    # Check that top-ranked index is as expected for each query
    assert np.array_equal(ranks[:, 0], np.array([0, 1, 2]))


def test_hit_at_k():
    sim, q_labels, l_labels = _toy_similarity_matrix()
    hit1 = compute_hit_at_k(sim, q_labels, l_labels, k=1)
    hit2 = compute_hit_at_k(sim, q_labels, l_labels, k=2)
    assert hit1 == 1.0  # top-1 correct for all queries
    assert hit2 == 1.0  # still perfect


def test_mrr():
    sim, q_labels, l_labels = _toy_similarity_matrix()
    mrr = compute_mean_reciprocal_rank(sim, q_labels, l_labels)
    # All correct at rank 1 -> MRR = 1
    assert mrr == 1.0


def test_compute_retrieval_metrics():
    sim, q_labels, l_labels = _toy_similarity_matrix()
    metrics = compute_retrieval_metrics(sim, q_labels, l_labels, topk=2)
    assert metrics["hit@1"] == 1.0
    assert metrics["hit@2"] == 1.0
    assert metrics["mrr"] == 1.0 
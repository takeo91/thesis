import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from new_source.distribution import (
    similarity_jaccard,
    similarity_dice,
    similarity_overlap_coefficient,
    similarity_hamming,
    similarity_euclidean,
    similarity_chebyshev,
    similarity_cosine,
    similarity_pearson,
    similarity_matlab_M,
    similarity_matlab_S1,
    similarity_matlab_S3,
    similarity_matlab_S5,
    similarity_matlab_S4,
    similarity_matlab_S6,
    similarity_matlab_S8,
    similarity_matlab_S9,
    similarity_matlab_S10,
    similarity_matlab_S11,
    similarity_matlab_S2_W,
    similarity_matlab_S,
)

# --- Helper arrays ---
identical = np.array([0.2, 0.5, 0.8, 1.0])
complement = 1 - identical
zeros = np.zeros(4)
ones = np.ones(4)
orthogonal = np.array([1, 0, 1, 0])
orthogonal2 = np.array([0, 1, 0, 1])

# --- Parametrized tests for basic properties ---
@pytest.mark.parametrize("func", [
    similarity_jaccard,
    similarity_dice,
    similarity_overlap_coefficient,
    similarity_hamming,
    similarity_euclidean,
    similarity_chebyshev,
    similarity_cosine,
    similarity_pearson,
    similarity_matlab_M,
    similarity_matlab_S1,
    similarity_matlab_S3,
    similarity_matlab_S5,
    similarity_matlab_S4,
    similarity_matlab_S6,
    similarity_matlab_S8,
    similarity_matlab_S9,
    similarity_matlab_S10,
    similarity_matlab_S11,
    similarity_matlab_S2_W,
    similarity_matlab_S,
])
def test_identical_vectors(func):
    result = func(identical, identical)
    assert np.isclose(result, 1.0, atol=1e-8), f"{func.__name__} failed for identical vectors"

@pytest.mark.parametrize("func", [
    similarity_jaccard,
    similarity_dice,
    similarity_overlap_coefficient,
    similarity_hamming,
    similarity_euclidean,
    similarity_chebyshev,
    similarity_cosine,
    similarity_pearson,
    similarity_matlab_M,
    similarity_matlab_S1,
    similarity_matlab_S3,
    similarity_matlab_S5,
    similarity_matlab_S4,
    similarity_matlab_S6,
    similarity_matlab_S8,
    similarity_matlab_S9,
    similarity_matlab_S10,
    similarity_matlab_S11,
    similarity_matlab_S2_W,
    similarity_matlab_S,
])
def test_zeros_vs_ones(func):
    result = func(zeros, ones)
    # For most metrics, zeros vs ones should be 0 similarity
    assert np.isclose(result, 0.0, atol=1e-8), f"{func.__name__} failed for zeros vs ones"

def test_cosine_orthogonal():
    # Cosine similarity of orthogonal vectors should be 0
    assert np.isclose(similarity_cosine(orthogonal, orthogonal2), 0.0, atol=1e-8)

def test_pearson_constant():
    # Pearson similarity with a constant vector should be 0
    const = np.ones(4)
    varying = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.isclose(similarity_pearson(const, varying), 0.0, atol=1e-8)

def test_hamming_similarity():
    # Hamming similarity of identical vectors is 1, of complementary is 0
    assert np.isclose(similarity_hamming(ones, ones), 1.0, atol=1e-8)
    assert np.isclose(similarity_hamming(ones, zeros), 0.0, atol=1e-8)

def test_euclidean_similarity():
    # Euclidean similarity of identical vectors is 1, of zeros vs ones is 1/(1+2)
    assert np.isclose(similarity_euclidean(ones, ones), 1.0, atol=1e-8)
    assert np.isclose(similarity_euclidean(zeros, ones), 1.0/(1.0+2.0), atol=1e-8)

def test_chebyshev_similarity():
    # Chebyshev similarity of identical vectors is 1, of zeros vs ones is 0
    assert np.isclose(similarity_chebyshev(ones, ones), 1.0, atol=1e-8)
    assert np.isclose(similarity_chebyshev(zeros, ones), 0.0, atol=1e-8)

# --- Add more edge case tests as needed --- 
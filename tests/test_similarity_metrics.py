import numpy as np
import pytest

from thesis.similarity_metrics import (
    similarity_jaccard,
    similarity_dice,
    similarity_overlap_coefficient,
    similarity_hamming,
    similarity_euclidean,
    similarity_chebyshev,
    similarity_cosine,
    similarity_pearson,
    mean_min_over_max,
    mean_dice_coefficient,
    mean_one_minus_abs_diff,
    intersection_over_max_cardinality,
    negated_intersection_over_max_cardinality,
    negated_overlap_coefficient,
    negated_symdiff_over_min_negated_component,
    negated_symdiff_over_max_negated_component,
    product_over_min_norm_squared,
    one_minus_mean_symmetric_difference,
    one_minus_abs_diff_over_sum_cardinality,
    fuzzy_symmetric_difference,
    fuzzy_intersection,
    fuzzy_union,
    fuzzy_negation,
    fuzzy_cardinality,
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
    mean_min_over_max,
    mean_dice_coefficient,
    mean_one_minus_abs_diff,
    intersection_over_max_cardinality,
    negated_intersection_over_max_cardinality,
    negated_overlap_coefficient,
    negated_symdiff_over_min_negated_component,
    negated_symdiff_over_max_negated_component,
    product_over_min_norm_squared,
    one_minus_abs_diff_over_sum_cardinality,
])
def test_identical_vectors(func):
    """Test that similarity metrics return 1.0 for identical vectors."""
    result = func(identical, identical)
    assert np.isclose(result, 1.0, atol=1e-8), f"{func.__name__} failed for identical vectors"

# Include only metrics that are expected to return 0.0 for zeros vs ones
@pytest.mark.parametrize("func", [
    similarity_jaccard,
    similarity_dice,
    similarity_hamming,
    similarity_chebyshev,
    similarity_cosine,
    mean_min_over_max,
    mean_dice_coefficient,
    mean_one_minus_abs_diff,
    intersection_over_max_cardinality,
    negated_intersection_over_max_cardinality,
    negated_symdiff_over_max_negated_component,
    product_over_min_norm_squared,
    one_minus_mean_symmetric_difference,
    one_minus_abs_diff_over_sum_cardinality,
])
def test_zeros_vs_ones(func):
    """Test that most metrics return 0.0 for completely dissimilar vectors (zeros vs ones)."""
    result = func(zeros, ones)
    assert np.isclose(result, 0.0, atol=1e-8), f"{func.__name__} failed for zeros vs ones"

def test_overlap_coefficient_edge_cases():
    """Test special cases for overlap coefficient."""
    # Regular case
    a = np.array([0.1, 0.3, 0.5, 0.7])
    b = np.array([0.2, 0.4, 0.6, 0.8])
    
    # Calculate expected result
    intersection = fuzzy_intersection(a, b)
    min_card = min(fuzzy_cardinality(a), fuzzy_cardinality(b))
    expected = fuzzy_cardinality(intersection) / min_card
    
    assert np.isclose(similarity_overlap_coefficient(a, b), expected, atol=1e-8)
    
    # For zeros vs ones, the result is 1.0 because the intersection is all zeros
    # and min cardinality is also 0, leading to 0/0 which is handled as 1.0
    assert np.isclose(similarity_overlap_coefficient(zeros, ones), 1.0, atol=1e-8)
    
    # Two empty sets (all zeros) are considered identical with similarity 1.0
    assert np.isclose(similarity_overlap_coefficient(zeros, zeros), 1.0, atol=1e-8)

def test_pearson_edge_cases():
    """Test special cases for Pearson correlation."""
    
    # For const vectors, Pearson should return 1.0 if both are constant, else 0.0
    assert np.isclose(similarity_pearson(ones, ones), 1.0, atol=1e-8)
    assert np.isclose(similarity_pearson(zeros, zeros), 1.0, atol=1e-8)
    
    # For zeros vs ones, Pearson is 1.0 because both are constant vectors with std=0
    assert np.isclose(similarity_pearson(zeros, ones), 1.0, atol=1e-8)
    
    # Test that non-constant vectors with perfect correlation give 1.0
    v1 = np.array([1, 2, 3, 4])
    v2 = np.array([2, 4, 6, 8])  # v2 = 2*v1, perfect correlation
    assert np.isclose(similarity_pearson(v1, v2), 1.0, atol=1e-8)
    
    # Test that non-constant vectors with perfect anti-correlation give -1.0
    v3 = np.array([4, 3, 2, 1])  # Reversed v1
    assert np.isclose(similarity_pearson(v1, v3), -1.0, atol=1e-8)

def test_negated_metrics_edge_cases():
    """Test special cases for metrics involving negation."""
    # For negated_overlap_coefficient
    assert np.isclose(negated_overlap_coefficient(zeros, ones), 1.0, atol=1e-8)
    
    # For negated_symdiff_over_min_negated_component
    assert np.isclose(negated_symdiff_over_min_negated_component(zeros, ones), 1.0, atol=1e-8)
    
    # Calculate expected value for a more typical case
    a = np.array([0.1, 0.3, 0.5, 0.7])
    b = np.array([0.2, 0.4, 0.6, 0.8])
    
    # Manual calculation for negated_overlap_coefficient
    neg_a = fuzzy_negation(a)
    neg_b = fuzzy_negation(b)
    intersection = fuzzy_intersection(neg_a, neg_b)
    min_card = min(fuzzy_cardinality(neg_a), fuzzy_cardinality(neg_b))
    expected = fuzzy_cardinality(intersection) / min_card
    
    assert np.isclose(negated_overlap_coefficient(a, b), expected, atol=1e-8)

def test_euclidean_similarity_zeros_ones():
    """Test the specific expected value for Euclidean similarity between zeros and ones."""
    # For Euclidean similarity, we expect 1/(1+d) where d is Euclidean distance
    # Distance between [0,0,0,0] and [1,1,1,1] is sqrt(4) = 2
    result = similarity_euclidean(zeros, ones)
    expected = 1.0 / (1.0 + 2.0)
    assert np.isclose(result, expected, atol=1e-8)

def test_cosine_orthogonal():
    """Test that cosine similarity of orthogonal vectors is 0."""
    assert np.isclose(similarity_cosine(orthogonal, orthogonal2), 0.0, atol=1e-8)

def test_pearson_constant():
    """Test that Pearson similarity with a constant vector is 0."""
    const = np.ones(4)
    varying = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.isclose(similarity_pearson(const, varying), 0.0, atol=1e-8)

def test_hamming_similarity():
    """Test Hamming similarity properties."""
    # Identical vectors should have similarity 1.0
    assert np.isclose(similarity_hamming(ones, ones), 1.0, atol=1e-8)
    # Completely different vectors should have similarity 0.0
    assert np.isclose(similarity_hamming(ones, zeros), 0.0, atol=1e-8)
    
    # Test with partial differences
    partial = np.array([1, 1, 0, 0])
    # Expected: 1 - (sum of abs differences / length)
    # sum of abs differences = 2, length = 4, so 1 - 2/4 = 0.5
    assert np.isclose(similarity_hamming(ones, partial), 0.5, atol=1e-8)

def test_euclidean_similarity():
    """Test Euclidean similarity properties."""
    # Identical vectors should have similarity 1.0
    assert np.isclose(similarity_euclidean(ones, ones), 1.0, atol=1e-8)
    
    # Custom test vector
    v1 = np.array([0.1, 0.2, 0.3, 0.4])
    v2 = np.array([0.2, 0.3, 0.4, 0.5])
    # Distance = sqrt(sum((v1-v2)^2)) = sqrt(0.1^2 * 4) = 0.2
    # Similarity = 1/(1+0.2) = 1/1.2 ≈ 0.833
    expected = 1.0 / (1.0 + 0.2)
    assert np.isclose(similarity_euclidean(v1, v2), expected, atol=1e-8)

def test_chebyshev_similarity():
    """Test Chebyshev similarity properties."""
    # Identical vectors should have similarity 1.0
    assert np.isclose(similarity_chebyshev(ones, ones), 1.0, atol=1e-8)
    # Completely different vectors (0s vs 1s) should have similarity 0.0
    assert np.isclose(similarity_chebyshev(zeros, ones), 0.0, atol=1e-8)
    
    # Custom test with partial differences
    v1 = np.array([0.1, 0.2, 0.3, 0.4])
    v2 = np.array([0.2, 0.4, 0.6, 0.8])
    # Max difference is 0.4, so similarity is 1 - 0.4 = 0.6
    assert np.isclose(similarity_chebyshev(v1, v2), 0.6, atol=1e-8)

def test_one_minus_mean_symmetric_difference():
    """Test one_minus_mean_symmetric_difference with mathematically derived expected values."""
    # For identical vectors, the symmetric difference components are:
    # - Term1 = Intersection(identical, Negation(identical)) = min(identical, 1-identical)
    # - Term2 = Intersection(Negation(identical), identical) = min(1-identical, identical)
    # Both terms are the same, and the union is the maximum at each point
    
    # Compute expected result for identical vectors
    symm_diff = fuzzy_symmetric_difference(identical, identical)
    expected = 1.0 - np.mean(symm_diff)
    
    result = one_minus_mean_symmetric_difference(identical, identical)
    assert np.isclose(result, expected, atol=1e-8)
    
    # Verify the actual expected value is approximately 0.775
    assert np.isclose(expected, 0.775, atol=1e-3)
    
    # For identical ones, the symmetric difference is 0, so the result is 1.0
    assert np.isclose(one_minus_mean_symmetric_difference(ones, ones), 1.0, atol=1e-8)
    
    # For identical zeros, the symmetric difference is 0, so the result is 1.0
    assert np.isclose(one_minus_mean_symmetric_difference(zeros, zeros), 1.0, atol=1e-8)
    
    # Test with zeros vs ones
    # Symmetric difference will be 1 at each point, so the mean is 1, resulting in 0
    assert np.isclose(one_minus_mean_symmetric_difference(zeros, ones), 0.0, atol=1e-8)

def test_jaccard_similarity():
    """Test Jaccard similarity with mathematical verification."""
    # For custom vectors
    a = np.array([0.1, 0.3, 0.5, 0.7])
    b = np.array([0.2, 0.4, 0.6, 0.8])
    
    # Calculate components manually
    intersection = fuzzy_intersection(a, b)
    union = fuzzy_union(a, b)
    expected = fuzzy_cardinality(intersection) / fuzzy_cardinality(union)
    
    assert np.isclose(similarity_jaccard(a, b), expected, atol=1e-8)

def test_dice_similarity():
    """Test Dice similarity with mathematical verification."""
    # For custom vectors
    a = np.array([0.1, 0.3, 0.5, 0.7])
    b = np.array([0.2, 0.4, 0.6, 0.8])
    
    # Calculate components manually
    intersection = fuzzy_intersection(a, b)
    expected = 2 * fuzzy_cardinality(intersection) / (fuzzy_cardinality(a) + fuzzy_cardinality(b))
    
    assert np.isclose(similarity_dice(a, b), expected, atol=1e-8)

# --- Add more edge case tests as needed --- 
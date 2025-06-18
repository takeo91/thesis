"""
Subset of similarity metrics for faster testing

This module provides a subset of the similarity metrics from the main similarity module,
focusing on a few representative metrics from different categories for faster testing.
"""

from typing import Dict, Union, Sequence

import numpy as np

from thesis.fuzzy.similarity import (
    # Set-theoretic metrics
    similarity_jaccard,
    similarity_dice,
    similarity_overlap_coefficient,
    
    # Distance-based metrics
    similarity_hamming,
    similarity_euclidean,
    
    # Correlation-based metrics
    similarity_cosine,
    similarity_pearson,
    
    # Information-theoretic metrics
    similarity_jensen_shannon,
    
    # Custom metrics
    mean_min_over_max,
    max_intersection
)

ArrayLike = Union[Sequence[float], np.ndarray]


def calculate_subset_similarity_metrics(
    mu_s1: ArrayLike,
    mu_s2: ArrayLike,
    x_values: ArrayLike,
    *,
    normalise: bool = False,
) -> Dict[str, float]:
    """
    Calculate a subset of similarity metrics between two membership functions.
    
    Args:
        mu_s1: First membership function values
        mu_s2: Second membership function values
        x_values: Domain values
        normalise: Whether to normalize metrics to [0,1]
    
    Returns:
        Dictionary of metric names to similarity values
    """
    # Ensure inputs are numpy arrays
    mu_s1, mu_s2 = map(np.asarray, (mu_s1, mu_s2))
    x_values = np.asarray(x_values)
    
    # Calculate all metrics
    metrics = {
        # Set-theoretic metrics
        "Jaccard": similarity_jaccard(mu_s1, mu_s2),
        "Dice": similarity_dice(mu_s1, mu_s2),
        "OverlapCoefficient": similarity_overlap_coefficient(mu_s1, mu_s2),
        
        # Distance-based metrics
        "Similarity_Hamming": similarity_hamming(mu_s1, mu_s2),
        "Similarity_Euclidean": similarity_euclidean(mu_s1, mu_s2),
        
        # Correlation-based metrics
        "Cosine": similarity_cosine(mu_s1, mu_s2),
        "Pearson": similarity_pearson(mu_s1, mu_s2),
        
        # Information-theoretic metrics
        "JensenShannon": similarity_jensen_shannon(mu_s1, mu_s2),
        
        # Custom metrics
        "MeanMinOverMax": mean_min_over_max(mu_s1, mu_s2),
        "MaxIntersection": max_intersection(mu_s1, mu_s2),
    }
    
    return metrics 
"""
Fuzzy set operations and similarity metrics for thesis project.

This module provides a collection of functions for working with fuzzy sets,
including basic set operations, membership function generation, and similarity metrics.
"""

from thesis.fuzzy.operations import (
    fuzzy_intersection,
    fuzzy_union,
    fuzzy_negation,
    fuzzy_cardinality,
    fuzzy_symmetric_difference,
    safe_divide,
)

from thesis.fuzzy.similarity import (
    # Set-theoretic metrics
    similarity_jaccard,
    similarity_dice,
    similarity_overlap_coefficient,
    
    # Distance-based metrics
    similarity_hamming,
    similarity_euclidean,
    similarity_chebyshev,
    
    # Correlation-based metrics
    similarity_cosine,
    similarity_pearson,
    
    # Additional metrics (previously MATLAB-specific)
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
    
    # Distance calculations
    distance_hamming,
    distance_euclidean,
    distance_chebyshev,
    
    # Master function to calculate all metrics
    calculate_all_similarity_metrics,
)

from thesis.fuzzy.membership import (
    compute_ndg,
    compute_membership_function,
    compute_membership_function_kde,
    compute_membership_functions,
)

from thesis.fuzzy.distributions import (
    # Empirical distribution computation
    compute_empirical_distribution_kde,
    compute_empirical_distribution_counts,
    
    # Fitness evaluation metrics
    compute_error_metrics,
    compute_kl_divergence,
    compute_chi_squared_test,
    compute_information_criteria,
    time_series_cross_validation,
    
    # Data preprocessing
    normalize_data,
    standardize_data,
    
    # Main fitness calculation
    compute_fitness_metrics,
)

__all__ = [
    # Fuzzy operations
    "fuzzy_intersection", "fuzzy_union", "fuzzy_negation", 
    "fuzzy_cardinality", "fuzzy_symmetric_difference", "safe_divide",
    
    # Similarity metrics - set-theoretic
    "similarity_jaccard", "similarity_dice", "similarity_overlap_coefficient",
    
    # Similarity metrics - distance-based
    "similarity_hamming", "similarity_euclidean", "similarity_chebyshev",
    
    # Similarity metrics - correlation-based
    "similarity_cosine", "similarity_pearson",
    
    # Additional metrics
    "mean_min_over_max", "mean_dice_coefficient", "mean_one_minus_abs_diff",
    "intersection_over_max_cardinality", "negated_intersection_over_max_cardinality",
    "negated_overlap_coefficient", "negated_symdiff_over_min_negated_component",
    "negated_symdiff_over_max_negated_component", "product_over_min_norm_squared",
    "one_minus_mean_symmetric_difference", "one_minus_abs_diff_over_sum_cardinality",
    
    # Distance calculations
    "distance_hamming", "distance_euclidean", "distance_chebyshev",
    
    # Membership function generation
    "compute_ndg", "compute_membership_function", "compute_membership_function_kde",
    "compute_membership_functions",
    
    # Master function for similarity
    "calculate_all_similarity_metrics",
    
    # Distribution analysis and fitness evaluation
    "compute_empirical_distribution_kde", "compute_empirical_distribution_counts",
    "compute_error_metrics", "compute_kl_divergence", "compute_chi_squared_test",
    "compute_information_criteria", "time_series_cross_validation",
    "normalize_data", "standardize_data", "compute_fitness_metrics",
] 
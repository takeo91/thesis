"""
fuzzy_helpers.py

Provides basic fuzzy logic operations and utility functions.
"""
import numpy as np

# ==============================================================================
# Utility Functions
# ==============================================================================

def safe_divide(numerator, denominator, default=0.0):
    """
    Performs division, returning a default value if the denominator is close to zero.

    Args:
        numerator (float or np.ndarray): The numerator(s).
        denominator (float or np.ndarray): The denominator(s).
        default (float, optional): Value to return on division by zero. Defaults to 0.0.

    Returns:
        float or np.ndarray: Result of the division or the default value.
    """
    if np.isscalar(denominator):
        # Handle scalar denominator
        return numerator / denominator if np.abs(denominator) > 1e-9 else default
    else:
        # Handle array denominator
        denominator = np.asarray(denominator)
        numerator = np.asarray(numerator)
        result = np.full_like(numerator, default, dtype=np.float64)
        valid_indices = np.abs(denominator) > 1e-9
        # Ensure numerator matches shape for broadcasting or element-wise division
        if numerator.shape == denominator.shape:
             result[valid_indices] = numerator[valid_indices] / denominator[valid_indices]
        # Check for scalar numerator (shape () or size 1)
        elif numerator.shape == () or numerator.size == 1:
             scalar_num = numerator.item() if isinstance(numerator, np.ndarray) else numerator
             result[valid_indices] = scalar_num / denominator[valid_indices]
        else: # Shapes incompatible other than scalar numerator
             raise ValueError("Numerator and denominator shapes incompatible for safe_divide.")

        return result

# ==============================================================================
# Basic Fuzzy Set Operations
# ==============================================================================

def fuzzy_intersection(mu1, mu2):
    """Computes the fuzzy intersection (pointwise minimum)."""
    return np.minimum(np.asarray(mu1), np.asarray(mu2))

def fuzzy_union(mu1, mu2):
    """Computes the fuzzy union (pointwise maximum)."""
    return np.maximum(np.asarray(mu1), np.asarray(mu2))

def fuzzy_negation(mu):
    """Computes the fuzzy negation (1 - mu)."""
    return 1.0 - np.asarray(mu)

def fuzzy_cardinality(mu):
    """Computes the fuzzy cardinality (sum of membership values)."""
    return np.sum(np.asarray(mu))

def fuzzy_symmetric_difference(mu1, mu2):
    """
    Computes the fuzzy symmetric difference using standard min/max/negation operators.
    Equivalent to Union(Intersection(A, neg(B)), Intersection(neg(A), B)).
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    neg_mu1 = fuzzy_negation(mu1)
    neg_mu2 = fuzzy_negation(mu2)
    term1 = fuzzy_intersection(mu1, neg_mu2)
    term2 = fuzzy_intersection(neg_mu1, mu2)
    return fuzzy_union(term1, term2)
    # Note: An alternative often used is simply np.abs(mu1 - mu2),
    # which corresponds to the difference in Lukasiewicz logic.
    # Using the standard definition here unless MATLAB source implies absolute difference.
"""
Basic fuzzy set operations.

This module provides fundamental fuzzy logic operations and utility functions
that serve as building blocks for fuzzy similarity metrics.
"""

from __future__ import annotations

from typing import Union, Sequence

import numpy as np
from thesis.core.constants import NUMERICAL_TOLERANCE

# Type definitions
ArrayLike = Union[Sequence[float], np.ndarray]


def safe_divide(numerator: ArrayLike, denominator: ArrayLike, default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safely perform division, returning a default value if the denominator is close to zero.

    Args:
        numerator: The numerator(s).
        denominator: The denominator(s).
        default: Value to return for division by zero. Defaults to 0.0.

    Returns:
        Result of division or the default value where denominator is near zero.
    """
    if np.isscalar(denominator):
        # Handle scalar denominator
        return numerator / denominator if np.abs(denominator) > NUMERICAL_TOLERANCE else default
    else:
        # Handle array denominator
        denominator = np.asarray(denominator)
        numerator = np.asarray(numerator)
        result = np.full_like(numerator, default, dtype=np.float64)
        valid_indices = np.abs(denominator) > NUMERICAL_TOLERANCE
        
        # Ensure numerator matches shape for broadcasting or element-wise division
        if numerator.shape == denominator.shape:
            result[valid_indices] = numerator[valid_indices] / denominator[valid_indices]
        elif numerator.shape == () or numerator.size == 1:
            scalar_num = numerator.item() if isinstance(numerator, np.ndarray) else numerator
            result[valid_indices] = scalar_num / denominator[valid_indices]
        else:
            raise ValueError("Numerator and denominator shapes incompatible for safe_divide.")

        return result


def fuzzy_intersection(mu1: ArrayLike, mu2: ArrayLike) -> np.ndarray:
    """
    Compute the fuzzy intersection (pointwise minimum).
    
    Args:
        mu1: First membership function.
        mu2: Second membership function.
        
    Returns:
        Fuzzy intersection (elementwise minimum).
    """
    return np.minimum(np.asarray(mu1), np.asarray(mu2))


def fuzzy_union(mu1: ArrayLike, mu2: ArrayLike) -> np.ndarray:
    """
    Compute the fuzzy union (pointwise maximum).
    
    Args:
        mu1: First membership function.
        mu2: Second membership function.
        
    Returns:
        Fuzzy union (elementwise maximum).
    """
    return np.maximum(np.asarray(mu1), np.asarray(mu2))


def fuzzy_negation(mu: ArrayLike) -> np.ndarray:
    """
    Compute the fuzzy negation (1 - mu).
    
    Args:
        mu: Membership function.
        
    Returns:
        Fuzzy negation (1 - mu).
    """
    return 1.0 - np.asarray(mu)


def fuzzy_cardinality(mu: ArrayLike) -> float:
    """
    Compute the fuzzy cardinality (sum of membership values).
    
    Args:
        mu: Membership function.
        
    Returns:
        Fuzzy cardinality (sum of membership values).
    """
    return float(np.sum(np.asarray(mu)))


def fuzzy_symmetric_difference(mu1: ArrayLike, mu2: ArrayLike) -> np.ndarray:
    """
    Compute the fuzzy symmetric difference.
    
    Equivalent to Union(Intersection(A, neg(B)), Intersection(neg(A), B)).
    
    Args:
        mu1: First membership function.
        mu2: Second membership function.
        
    Returns:
        Fuzzy symmetric difference.
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    neg_mu1 = fuzzy_negation(mu1)
    neg_mu2 = fuzzy_negation(mu2)
    term1 = fuzzy_intersection(mu1, neg_mu2)
    term2 = fuzzy_intersection(neg_mu1, mu2)
    return fuzzy_union(term1, term2) 
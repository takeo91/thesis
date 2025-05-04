"""
preprocessing.py

Provides data preprocessing utilities like normalization and standardization.
"""

import numpy as np
from fuzzy_helpers import safe_divide  # Relative import

# ==============================================================================
# Data Preprocessing Utilities
# ==============================================================================


def normalize_data(data):
    """
    Performs Min-Max Normalization on the data to scale it to the range [0, 1].

    Args:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: Normalized data array. Returns array of 0.5s if range is zero.
    """
    data = np.asarray(data)
    if data.size == 0:
        return data  # Handle empty
    x_min = np.min(data)
    x_max = np.max(data)
    data_range = x_max - x_min
    # Handle case where all data points are the same (range is zero)
    if data_range < 1e-9:
        # Returning 0.5s for constant data.
        return np.full_like(data, 0.5, dtype=np.float64)
    normalized_data = (data - x_min) / data_range
    return normalized_data


def standardize_data(data):
    """
    Performs Z-Score Standardization (mean=0, std=1) on the data.

    Args:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: Standardized data array. Returns array of zeros if std dev is zero.
    """
    data = np.asarray(data)
    if data.size == 0:
        return data  # Handle empty
    mean = np.mean(data)
    std = np.std(data)
    # Handle case where standard deviation is zero
    standardized_data = safe_divide(data - mean, std, default=0.0)
    return standardized_data

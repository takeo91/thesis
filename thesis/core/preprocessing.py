"""
Data preprocessing utilities.

This module provides functions for data preprocessing such as
normalization and standardization of sensor data.
"""

from __future__ import annotations
from typing import Union, Sequence

import numpy as np

from thesis.fuzzy.operations import safe_divide

# Type definitions
ArrayLike = Union[Sequence[float], np.ndarray]


def normalize_data(data: ArrayLike) -> np.ndarray:
    """
    Performs Min-Max Normalization on the data to scale it to the range [0, 1].

    Args:
        data: Input data array.

    Returns:
        Normalized data array. Returns array of 0.5s if range is zero.
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


def standardize_data(data: ArrayLike) -> np.ndarray:
    """
    Performs Z-Score Standardization (mean=0, std=1) on the data.

    Args:
        data: Input data array.

    Returns:
        Standardized data array. Returns array of zeros if std dev is zero.
    """
    data = np.asarray(data)
    if data.size == 0:
        return data  # Handle empty
    mean = np.mean(data)
    std = np.std(data)
    # Handle case where standard deviation is zero
    standardized_data = safe_divide(data - mean, std, default=0.0)
    return standardized_data 
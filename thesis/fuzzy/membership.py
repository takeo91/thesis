"""
Fuzzy membership function computation.

This module provides functions for computing membership functions
from sensor data using neighbor density method.
"""

from __future__ import annotations

from typing import Tuple, Union, Sequence, Optional, Final
from numpy.typing import ArrayLike, NDArray

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import math

# Type definitions
ArrayLike = Union[Sequence[float], np.ndarray]

SQRT_2PI: Final[float] = math.sqrt(2.0 * math.pi)

def compute_ndg_streaming(
    x_values: ArrayLike,
    sensor_data: ArrayLike,
    sigma: float,
    chunk_size: int = 10_000,
    dtype: str | np.dtype = "float64",
) -> NDArray[np.floating]:
    """
    Neighbour–Density Graph (streaming, memory–bounded).

    Parameters
    ----------
    x_values
        1-D coordinates at which to evaluate the density.  Length **N**.
    sensor_data
        1-D array of observed sensor samples.  Length **M**.  Must be non-empty.
    sigma
        Positive bandwidth of the Gaussian kernel.
    chunk_size
        How many `x_values` to process per iteration.  Trade-off RAM ↔ speed.
    dtype
        Accumulator dtype.  Use float32 when RAM is very tight.

    Returns
    -------
    ndg : ndarray, shape (N,)
        Normalised kernel-density estimate at each `x_values[i]`.
    """
    x = np.asarray(x_values, dtype=dtype)
    data = np.asarray(sensor_data, dtype=dtype)

    if data.size == 0:
        raise ValueError("sensor_data must contain at least one point.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    norm = 1.0 / (SQRT_2PI * sigma * data.size)
    inv_two_sigma_sq = 0.5 / (sigma * sigma)
    out = np.empty_like(x)

    for start in range(0, x.size, chunk_size):
        end = start + chunk_size
        x_chunk = x[start:end][:, None]            # shape (c,1)
        d2 = (x_chunk - data[None, :]) ** 2        # (c,M) but *small* c
        out[start:end] = np.exp(-d2 * inv_two_sigma_sq).sum(axis=1)

    return out * norm

def compute_ndg_dense(x_values: ArrayLike, sensor_data: ArrayLike, sigma: float) -> np.ndarray:
    """
    Compute Neighbor Density Graph (NDG) values.
    
    Calculates the sum of Gaussian kernel values centered at each sensor data point,
    evaluated at the given x_values.
    
    Args:
        x_values: Points at which to calculate density.
        sensor_data: Input data points.
        sigma: Bandwidth parameter for the Gaussian kernel.
    
    Returns:
        NDG values corresponding to x_values.
    """
    x_values = np.asarray(x_values)[:, np.newaxis]  # Shape: (len(x_values), 1)
    sensor_data = np.asarray(sensor_data)[np.newaxis, :]  # Shape: (1, len(sensor_data))
    
    if sensor_data.size == 0:
        return np.zeros(x_values.shape[0])  # Handle empty data

    squared_diffs = (x_values - sensor_data) ** 2
    # Use a minimum sigma to prevent division by zero
    safe_sigma_sq = max(sigma**2, 1e-18)
    exponentials = np.exp(-squared_diffs / safe_sigma_sq)
    ndg = exponentials.sum(axis=1)
    
    return ndg


def compute_membership_function(
    sensor_data: ArrayLike, 
    x_values: Optional[ArrayLike] = None, 
    sigma: Optional[Union[float, str]] = None, 
    num_points: int = 500
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a normalized membership function using the Neighbor Density Graph method.

    Args:
        sensor_data: Input data points.
        x_values: Domain for the membership function. If None, calculated from data range.
        sigma: Bandwidth parameter. If None, uses 0.1 * data range.
               If str like 'r0.2', uses 0.2 * data range.
        num_points: Number of points for x_values if automatically calculated.

    Returns:
        Tuple containing:
        - x_values: Domain points where membership function is evaluated
        - mu_s: Normalized membership function values
        - sigma_val: The actual sigma value used
    """
    sensor_data = np.asarray(sensor_data)
    
    if sensor_data.size == 0:
        # Handle empty data gracefully
        if x_values is None:
            x_values = np.linspace(0, 1, num_points)  # Default range
        else:
            x_values = np.asarray(x_values)
        # Return zeros for mu, default x_values, and a default sigma estimate
        return x_values, np.zeros_like(x_values), 0.1

    x_min, x_max = np.min(sensor_data), np.max(sensor_data)
    data_range = x_max - x_min

    # Define x_values if not provided
    if x_values is None:
        # Handle case where data is constant (range is zero)
        center = x_min if data_range < 1e-9 else (x_min + x_max) / 2
        spread = 1.0 if data_range < 1e-9 else data_range  # Use default spread if range is zero
        x_values = np.linspace(center - spread, center + spread, num_points)
    else:
        x_values = np.asarray(x_values)

    # Determine sigma value to use
    default_sigma_ratio = 0.1
    
    if sigma is None:
        sigma_val = default_sigma_ratio * data_range if data_range > 1e-9 else default_sigma_ratio
    elif isinstance(sigma, str) and sigma.startswith("r"):
        try:
            ratio = float(sigma[1:])
            sigma_val = ratio * data_range if data_range > 1e-9 else ratio
        except ValueError:
            raise ValueError(f"Invalid sigma string format: {sigma}. Expected 'r<float>'.")
    else:
        try:
            sigma_val = float(sigma)  # Assume numeric if not string 'r...'
        except ValueError:
            raise ValueError(f"Invalid sigma value: {sigma}. Expected float, None, or 'r<float>'.")

    # Ensure sigma is positive and non-zero
    if sigma_val < 1e-9:
        sigma_val = 1e-9

    # Compute NDG and normalize to get membership function (mu)
    ndg_s = compute_ndg(x_values, sensor_data, sigma_val)
    sum_ndg = np.sum(ndg_s)
    mu_s = ndg_s / sum_ndg if sum_ndg > 1e-9 else np.zeros_like(ndg_s)

    return x_values, mu_s, sigma_val


def compute_membership_function_kde(
    sensor_data: ArrayLike, 
    x_values: Optional[ArrayLike] = None, 
    num_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a normalized membership function using Gaussian KDE.

    Args:
        sensor_data: Input data points.
        x_values: Domain for the membership function. If None, calculated from data range.
        num_points: Number of points for x_values if automatically calculated.

    Returns:
        Tuple containing:
        - x_values: Domain points where membership function is evaluated
        - mu_s: Normalized membership function values
    """
    sensor_data = np.asarray(sensor_data)
    
    if sensor_data.size < 2:  # KDE requires at least 2 points
        if x_values is None:
            # Define a default range if data is empty or constant
            x_min = np.min(sensor_data) if sensor_data.size > 0 else 0
            x_max = np.max(sensor_data) if sensor_data.size > 0 else 1
            data_range = x_max - x_min
            center = x_min if data_range < 1e-9 else (x_min + x_max) / 2
            spread = 1.0 if data_range < 1e-9 else data_range
            x_values = np.linspace(center - spread, center + spread, num_points)
        else:
            x_values = np.asarray(x_values)
        return x_values, np.zeros_like(x_values)  # Return zeros if not enough data

    # Define x_values if not provided
    if x_values is None:
        x_min, x_max = np.min(sensor_data), np.max(sensor_data)
        data_range = x_max - x_min
        center = x_min if data_range < 1e-9 else (x_min + x_max) / 2
        spread = 1.0 if data_range < 1e-9 else data_range
        x_values = np.linspace(center - spread, center + spread, num_points)
    else:
        x_values = np.asarray(x_values)

    # Compute KDE and normalize
    try:
        kde = gaussian_kde(sensor_data)
        mu_s = kde.evaluate(x_values)
        mu_s = np.clip(mu_s, 0, None)  # Ensure non-negative
        sum_mu = np.sum(mu_s)
        if sum_mu > 1e-9:
            mu_s /= sum_mu  # Normalize
        else:
            mu_s = np.zeros_like(mu_s)
    except (
        np.linalg.LinAlgError,
        ValueError,
    ):  # Handle LinAlgError or cases like all points identical
        # Fallback: return zeros if KDE fails
        mu_s = np.zeros_like(x_values)

    return x_values, mu_s


def compute_membership_functions(
    sensor_data: ArrayLike, 
    x_values: ArrayLike, 
    method: str = "nd", 
    sigma: Optional[Union[float, str]] = None
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Wrapper function to compute membership function using the specified method.

    Args:
        sensor_data: The sensor data.
        x_values: The x-values over which to compute the membership function.
        method: Method to use ('nd' or 'kde').
        sigma: Sigma for 'nd' method. Ignored for 'kde'.

    Returns:
        Tuple containing:
        - mu: The computed membership function
        - sigma_val: The sigma used (if method='nd'), else None.
    """
    sigma_val = None
    
    if method == "nd":
        x_values_calc, mu, sigma_val = compute_membership_function(
            sensor_data, x_values, sigma=sigma
        )
    elif method == "kde":
        x_values_calc, mu = compute_membership_function_kde(sensor_data, x_values)
        # Ensure output mu has same shape as input x_values even if internal calculation used different points
        if x_values_calc.shape != x_values.shape or not np.allclose(
            x_values_calc, x_values
        ):
            interp_mu = interp1d(
                x_values_calc, mu, kind="linear", bounds_error=False, fill_value=0
            )
            mu = interp_mu(x_values)
            mu = np.clip(mu, 0, None)
            sum_mu = np.sum(mu)
            if sum_mu > 1e-9:
                mu /= sum_mu
            else:
                mu = np.zeros_like(mu)
    else:
        raise ValueError("Unknown method for membership function. Use 'nd' or 'kde'.")
        
    return mu, sigma_val 
"""
Fuzzy membership function computation.

This module provides functions for computing membership functions
from sensor data using neighbor density method.
"""

from __future__ import annotations

from typing import Tuple, Union, Optional, Final
from numpy.typing import ArrayLike, NDArray

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import math

# Constants
SQRT_2PI: Final[float] = math.sqrt(2.0 * math.pi)


def compute_ndg_streaming(
    x_values: ArrayLike,
    sensor_data: ArrayLike,
    sigma: float,
    *,
    kernel_type: str = "gaussian",
    chunk_size: int = 10_000,
    dtype: str | np.dtype = "float64",
) -> NDArray[np.floating]:
    """
    Streaming Neighbour-Density Graph with selectable kernel functions.

    Parameters
    ----------
    x_values : array-like
        Points at which to evaluate the density
    sensor_data : array-like
        Data points from which to estimate the density
    sigma : float
        Bandwidth parameter, controls the smoothness of the density estimate
    kernel_type : str, default="gaussian"
        Type of kernel to use. Options:
        - "gaussian": Standard normal kernel (default)
        - "epanechnikov": Optimal kernel in terms of mean squared error
        - "triangular": Linear falloff kernel
        - "uniform": Rectangular/box kernel
        - "quartic": Biweight kernel, smoother than Epanechnikov
        - "cosine": Cosine-based kernel with smooth falloff
    chunk_size : int, default=10_000
        Size of chunks for memory-efficient processing
    dtype : str or numpy.dtype, default="float64"
        Data type for computation
    normalize : bool, default=True
        Whether to normalize by the number of data points and kernel constant
        Set to False to get raw values for custom normalization later

    Returns
    -------
    ndg : ndarray, shape (len(x_values),)
        Kernel-density estimate using the specified kernel function

    Notes
    -----
    For the Gaussian kernel, the formula is:
        Σ_j  exp(-0.5 · (x_i - x_j)² / σ²)  / (√(2π) · σ · n)
    
    Other kernels use their respective formulations with appropriate normalization.
    """
    x = np.asarray(x_values, dtype=dtype)
    data = np.asarray(sensor_data, dtype=dtype)

    # Keep legacy contract: empty input → zeros not error
    if data.size == 0:
        return np.zeros_like(x)

    if sigma <= 0:
        raise ValueError("sigma must be > 0.")
    
    # Define kernel functions and their normalization constants
    if kernel_type == "gaussian":
        def kernel_func(d2, inv_two_sigma_sq):
            return np.exp(-d2 * inv_two_sigma_sq)
        norm = 1.0 / (SQRT_2PI * sigma * data.size)
        
    elif kernel_type == "epanechnikov":
        def kernel_func(d2, inv_two_sigma_sq):
            u2 = d2 * inv_two_sigma_sq * 2  # u² = (x-x_j)²/σ²
            return 0.75 * (1 - u2) * (u2 <= 1)
        norm = 1.0 / (sigma * data.size)
        
    elif kernel_type == "triangular":
        def kernel_func(d2, inv_two_sigma_sq):
            u = np.sqrt(d2 * inv_two_sigma_sq * 2)  # u = |x-x_j|/σ
            return (1 - u) * (u <= 1)
        norm = 1.0 / (sigma * data.size)
        
    elif kernel_type == "uniform":
        def kernel_func(d2, inv_two_sigma_sq):
            u = np.sqrt(d2 * inv_two_sigma_sq * 2)  # u = |x-x_j|/σ
            return 0.5 * (u <= 1)
        norm = 1.0 / (sigma * data.size)
        
    elif kernel_type == "quartic":
        def kernel_func(d2, inv_two_sigma_sq):
            u2 = d2 * inv_two_sigma_sq * 2  # u² = (x-x_j)²/σ²
            return (15/16) * ((1 - u2)**2) * (u2 <= 1)
        norm = 1.0 / (sigma * data.size)
        
    elif kernel_type == "cosine":
        def kernel_func(d2, inv_two_sigma_sq):
            u = np.sqrt(d2 * inv_two_sigma_sq * 2)  # u = |x-x_j|/σ
            return (np.pi/4) * np.cos(np.pi*u/2) * (u <= 1)
        norm = 1.0 / (sigma * data.size)
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. " 
                        f"Supported types: gaussian, epanechnikov, triangular, " 
                        f"uniform, quartic, cosine")

    inv_two_sigma_sq = 0.5 / (sigma * sigma)
    out = np.empty_like(x)

    for start in range(0, x.size, chunk_size):
        end = min(start + chunk_size, x.size)
        x_chunk = x[start:end, None]                  # (c, 1)
        d2 = (x_chunk - data[None, :]) ** 2           # (c, m)
        out[start:end] = kernel_func(d2, inv_two_sigma_sq).sum(axis=1)

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
    norm = 1.0 / (SQRT_2PI * sigma * sensor_data.size)

    if sensor_data.size == 0:
        return np.zeros(x_values.shape[0])  # Handle empty data

    squared_diffs = (x_values - sensor_data) ** 2
    # Use a minimum sigma to prevent division by zero
    safe_sigma_sq = max(sigma**2, 1e-18)
    inv_two_sigma_sq = 0.5 / (safe_sigma_sq )
    exponentials = np.exp(-squared_diffs * inv_two_sigma_sq)
    ndg = exponentials.sum(axis=1)

    return ndg * norm


def compute_membership_function(
    sensor_data: ArrayLike, 
    x_values: Optional[ArrayLike] = None, 
    sigma: Optional[Union[float, str]] = None, 
    num_points: int = 500,
    kernel_type: str = "gaussian",
    normalization: str = "sum"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a normalized membership function using the Neighbor Density Graph method.

    Args:
        sensor_data: Input data points.
        x_values: Domain for the membership function. If None, calculated from data range.
        sigma: Bandwidth parameter. If None, uses 0.1 * data range.
               If str like 'r0.2', uses 0.2 * data range.
        num_points: Number of points for x_values if automatically calculated.
        kernel_type: Type of kernel to use (gaussian, epanechnikov, triangular, etc.).
                    See compute_ndg_streaming for all options.
        normalization: Method to normalize the membership function:
                      - "sum": Sum of all values equals 1 (traditional fuzzy membership)
                      - "integral": Integral equals 1 (probability density function)

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

    # Compute NDG
    ndg_s = compute_ndg_streaming(x_values, sensor_data, sigma_val, kernel_type=kernel_type)
    
    # Normalize based on selected method
    if normalization == "sum":
        sum_ndg = np.sum(ndg_s)
        mu_s = ndg_s / sum_ndg if sum_ndg > 1e-9 else np.zeros_like(ndg_s)
    elif normalization == "integral":
        integral = np.trapezoid(ndg_s, x=x_values)
        mu_s = ndg_s / integral if integral > 1e-9 else np.zeros_like(ndg_s)
    else:
        raise ValueError(f"Unknown normalization: {normalization}. Use 'sum' or 'integral'.")

    return x_values, mu_s, sigma_val


def compute_kde_density(x: ArrayLike, data: ArrayLike, sigma: float = None) -> NDArray[np.floating]:
    """
    Compute 1-D Gaussian KDE at points x with bandwidth sigma.
    
    Args:
        x: Points to evaluate density at
        data: Input data points
        sigma: Bandwidth parameter (in standard deviation units)
              If None, uses scipy's default bandwidth selection (Scott's rule)
        
    Returns:
        Array of density values normalized to integrate to 1.0
    """
    x = np.asarray(x)
    data = np.asarray(data)
    
    if data.size < 2:
        return np.full_like(x, 1 / max(x.size, 1))
    
    if sigma is not None:
        # Use sigma to override Scott's factor
        bw = sigma / max(np.std(data), 1e-9)
        kde = gaussian_kde(dataset=data, bw_method=bw)
    else:
        # Use default bandwidth selection
        kde = gaussian_kde(dataset=data)
        
    density = np.clip(kde(x), 1e-15, None)
    
    # Normalize to integrate to 1.0
    integral = np.trapezoid(density, x=x)
    if integral > 1e-15:
        return density / integral
    else:
        return np.full_like(x, 1 / max(x.size, 1))


def compute_membership_function_kde(
    sensor_data: ArrayLike, 
    x_values: Optional[ArrayLike] = None, 
    num_points: int = 500,
    sigma: Optional[float] = None,
    normalization: str = "integral"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a normalized membership function using Gaussian KDE.

    Args:
        sensor_data: Input data points.
        x_values: Domain for the membership function. If None, calculated from data range.
        num_points: Number of points for x_values if automatically calculated.
        sigma: Optional bandwidth parameter. If None, uses scipy's default.
        normalization: Method to normalize the membership function:
                      - "sum": Sum of all values equals 1 (traditional fuzzy membership)
                      - "integral": Integral equals 1 (probability density function)

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
        # Use compute_kde_density for the core KDE computation
        kde_result = compute_kde_density(x_values, sensor_data, sigma)
        
        # Apply the requested normalization method
        if normalization == "sum":
            sum_kde = np.sum(kde_result)
            if sum_kde > 1e-9:
                mu_s = kde_result / sum_kde
            else:
                mu_s = np.zeros_like(kde_result)
        elif normalization == "integral":
            # Density is already normalized to integrate to 1.0 by compute_kde_density
            # But we double-check to ensure numerical stability
            integral = np.trapezoid(kde_result, x=x_values)
            if integral > 1e-9:
                mu_s = kde_result / integral
            else:
                mu_s = np.zeros_like(kde_result)
        else:
            raise ValueError(f"Unknown normalization: {normalization}. Use 'sum' or 'integral'.")
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: return zeros if KDE fails
        mu_s = np.zeros_like(x_values)

    return x_values, mu_s


def compute_membership_functions(
    sensor_data: ArrayLike, 
    x_values: ArrayLike, 
    method: str = "nd", 
    sigma: Optional[Union[float, str]] = None,
    kernel_type: str = "gaussian",
    normalization: str = "integral"
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Wrapper function to compute membership function using the specified method.

    Args:
        sensor_data: The sensor data.
        x_values: The x-values over which to compute the membership function.
        method: Method to use ('nd' or 'kde').
        sigma: Sigma for 'nd' method or KDE. If relative sigma (e.g., 'r0.1') it's only applied for 'nd'. 
        kernel_type: Type of kernel to use for 'nd' method. Ignored for 'kde'.
                    Options: gaussian, epanechnikov, triangular, uniform, quartic, cosine.
        normalization: Normalization method ('sum' or 'integral') for membership function.

    Returns:
        Tuple containing:
        - mu: The computed membership function
        - sigma_val: The sigma used (if method='nd'), else None.
    """
    sigma_val = None
    
    if method == "nd":
        x_values_calc, mu, sigma_val = compute_membership_function(
            sensor_data, x_values, sigma=sigma, kernel_type=kernel_type, 
            normalization=normalization
        )
    elif method == "kde":
        # Process sigma for KDE if it's a string or None
        kde_sigma = None
        if sigma is not None and not isinstance(sigma, str):
            kde_sigma = float(sigma)
            
        x_values_calc, mu = compute_membership_function_kde(
            sensor_data, x_values, sigma=kde_sigma, normalization=normalization
        )
        
        # Ensure output mu has same shape as input x_values even if internal calculation used different points
        if x_values_calc.shape != x_values.shape or not np.allclose(
            x_values_calc, x_values
        ):
            interp_mu = interp1d(
                x_values_calc, mu, kind="linear", bounds_error=False, fill_value=0
            )
            mu = interp_mu(x_values)
            mu = np.clip(mu, 0, None)
            
            # Apply selected normalization
            if normalization == "sum":
                sum_mu = np.sum(mu)
                if sum_mu > 1e-9:
                    mu /= sum_mu
                else:
                    mu = np.zeros_like(mu)
            elif normalization == "integral":
                integral = np.trapezoid(mu, x=x_values)
                if integral > 1e-9:
                    mu /= integral
                else:
                    mu = np.zeros_like(mu)
            else:
                raise ValueError(f"Unknown normalization: {normalization}. Use 'sum' or 'integral'.")
    else:
        raise ValueError("Unknown method for membership function. Use 'nd' or 'kde'.")
        
    return mu, sigma_val 
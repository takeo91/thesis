"""
membership_functions.py

Functions for computing membership functions from sensor data using
Neighbor Density Graph (NDG) or Kernel Density Estimation (KDE) methods.
"""
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

# ==============================================================================
# Membership Function Computation Core Logic
# ==============================================================================

def compute_ndg(x_values, sensor_data, sigma):
    """
    Computes Neighbor Density Graph (NDG) values. Helper for compute_membership_function.

    Args:
        x_values (np.ndarray): Points at which to calculate density.
        sensor_data (np.ndarray): Input data points.
        sigma (float): Bandwidth parameter for the Gaussian kernel.

    Returns:
        np.ndarray: NDG values corresponding to x_values.
    """
    x_values = np.asarray(x_values)[:, np.newaxis] # Shape: (len(x_values), 1)
    sensor_data = np.asarray(sensor_data)[np.newaxis, :] # Shape: (1, len(sensor_data))
    if sensor_data.size == 0: return np.zeros(x_values.shape[0]) # Handle empty data

    squared_diffs = (x_values - sensor_data) ** 2
    # Use a minimum sigma to prevent division by zero
    safe_sigma_sq = max(sigma ** 2, 1e-18)
    exponentials = np.exp(-squared_diffs / safe_sigma_sq)
    ndg = exponentials.sum(axis=1)
    return ndg

def compute_membership_function(sensor_data, x_values=None, sigma=None, num_points=500):
    """
    Computes a normalized membership function using the Neighbor Density Graph method.

    Args:
        sensor_data (np.ndarray): Input data points.
        x_values (np.ndarray, optional): Domain for the membership function.
                                         If None, calculated from data range. Defaults to None.
        sigma (float or str, optional): Bandwidth parameter. If None, uses 0.1 * data range.
                                        If str like 'r0.2', uses 0.2 * data range. Defaults to None.
        num_points (int, optional): Number of points for x_values if automatically calculated. Defaults to 500.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Tuple containing (x_values, mu_s, sigma_val).
                                                mu_s is the normalized membership function.
                                                sigma_val is the actual sigma value used.
    """
    sensor_data = np.asarray(sensor_data)
    if sensor_data.size == 0:
        # Handle empty data gracefully
        if x_values is None: x_values = np.linspace(0, 1, num_points) # Default range
        else: x_values = np.asarray(x_values)
        # Return zeros for mu, default x_values, and a default sigma estimate (e.g., 0.1)
        return x_values, np.zeros_like(x_values), 0.1

    x_min, x_max = np.min(sensor_data), np.max(sensor_data)
    data_range = x_max - x_min

    # Define x_values if not provided
    if x_values is None:
        # Handle case where data is constant (range is zero)
        center = x_min if data_range < 1e-9 else (x_min + x_max) / 2
        spread = 1.0 if data_range < 1e-9 else data_range # Use a default spread if range is zero
        x_values = np.linspace(center - spread, center + spread, num_points)
    else:
        x_values = np.asarray(x_values)

    # Determine sigma value to use
    default_sigma_ratio = 0.1
    if sigma is None:
        sigma_val = default_sigma_ratio * data_range if data_range > 1e-9 else default_sigma_ratio
    elif isinstance(sigma, str) and sigma.startswith('r'):
        try:
            ratio = float(sigma[1:])
            sigma_val = ratio * data_range if data_range > 1e-9 else ratio
        except ValueError:
            raise ValueError(f"Invalid sigma string format: {sigma}. Expected 'r<float>'.")
    else:
        try:
            sigma_val = float(sigma) # Assume numeric if not string 'r...'
        except ValueError:
            raise ValueError(f"Invalid sigma value: {sigma}. Expected float, None, or 'r<float>'.")

    # Ensure sigma is positive and non-zero
    if sigma_val < 1e-9: sigma_val = 1e-9

    # Compute NDG and normalize to get membership function (mu)
    ndg_s = compute_ndg(x_values, sensor_data, sigma_val)
    sum_ndg = np.sum(ndg_s)
    mu_s = ndg_s / sum_ndg if sum_ndg > 1e-9 else np.zeros_like(ndg_s)

    return x_values, mu_s, sigma_val


def compute_membership_function_kde(sensor_data, x_values=None, num_points=500):
    """
    Computes a normalized membership function using Gaussian KDE.

    Args:
        sensor_data (np.ndarray): Input data points.
        x_values (np.ndarray, optional): Domain for the membership function.
                                         If None, calculated from data range. Defaults to None.
        num_points (int, optional): Number of points for x_values if automatically calculated. Defaults to 500.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing (x_values, mu_s).
                                        mu_s is the normalized membership function.
    """
    sensor_data = np.asarray(sensor_data)
    if sensor_data.size < 2: # KDE requires at least 2 points
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
        return x_values, np.zeros_like(x_values) # Return zeros if not enough data

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
        mu_s = np.clip(mu_s, 0, None) # Ensure non-negative
        sum_mu = np.sum(mu_s)
        if sum_mu > 1e-9:
             mu_s /= sum_mu # Normalize
        else:
             mu_s = np.zeros_like(mu_s)
    except (np.linalg.LinAlgError, ValueError): # Handle LinAlgError or cases like all points identical
        # Fallback: return zeros if KDE fails
        mu_s = np.zeros_like(x_values)

    return x_values, mu_s

# ==============================================================================
# Wrapper Function
# ==============================================================================

def compute_membership_functions(sensor_data, x_values, method='nd', sigma=None):
    """
    Wrapper function to compute membership function using the specified method.

    Args:
        sensor_data (np.ndarray): The sensor data.
        x_values (np.ndarray): The x-values over which to compute the membership function.
        method (str, optional): Method to use ('nd' or 'kde'). Defaults to 'nd'.
        sigma (float or str, optional): Sigma for 'nd' method. Defaults to None.

    Returns:
        tuple[np.ndarray, float or None]: Tuple containing (mu, sigma_val).
                                          mu is the computed membership function.
                                          sigma_val is the sigma used (if method='nd'), else None.
    """
    sigma_val = None
    if method == 'nd':
        x_values_calc, mu, sigma_val = compute_membership_function(sensor_data, x_values, sigma=sigma)
        # Note: compute_membership_function already ensures output aligns with input x_values if provided
    elif method == 'kde':
        x_values_calc, mu = compute_membership_function_kde(sensor_data, x_values)
        # Ensure output mu has same shape as input x_values even if internal calculation used different points
        if not np.array_equal(x_values_calc, x_values):
            interp_mu = interp1d(x_values_calc, mu, kind='linear', bounds_error=False, fill_value=0)
            mu = interp_mu(x_values)
            # Re-normalize after interpolation
            mu = np.clip(mu, 0, None)
            sum_mu = np.sum(mu)
            if sum_mu > 1e-9: mu /= sum_mu
            else: mu = np.zeros_like(mu)

    else:
        raise ValueError("Unknown method for membership function. Use 'nd' or 'kde'.")
    return mu, sigma_val

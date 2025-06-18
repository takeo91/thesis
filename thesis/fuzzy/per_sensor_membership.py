"""
Per-Sensor Membership Function Module

This module implements an alternative approach to fuzzy membership function generation
where each sensor gets its own membership function, rather than combining all sensors
into a single membership function.

This allows for more granular similarity calculations that can account for
sensor-specific characteristics and importance.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Sequence
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from thesis.fuzzy.similarity import (
    similarity_jaccard,
    similarity_dice,
    similarity_cosine,
    similarity_euclidean,
    similarity_pearson
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = Union[Sequence[float], np.ndarray]


def compute_ndg_per_sensor(
    window_data: np.ndarray,
    kernel_type: str = "gaussian",
    sigma_method: str = "std",
    n_points: int = 100,
    domain_expansion: float = 0.2
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute NDG membership function for each sensor in the window.
    
    Args:
        window_data: 2D array of shape (window_size, n_features)
        kernel_type: Type of kernel to use ('gaussian' or 'triangular')
        sigma_method: Method to compute sigma ('std', 'range', or 'iqr')
        n_points: Number of points to evaluate the membership function
        domain_expansion: Expansion factor for the domain range
        
    Returns:
        Tuple of (x_values, list of membership functions per sensor)
    """
    # Get window dimensions
    window_size, n_features = window_data.shape
    
    # Create list to store membership functions
    membership_functions = []
    
    # Process each sensor/feature separately
    for i in range(n_features):
        sensor_data = window_data[:, i]
        
        # Define domain (x-axis)
        x_min = np.min(sensor_data)
        x_max = np.max(sensor_data)
        
        # Expand domain slightly to ensure full coverage
        domain_range = x_max - x_min
        if domain_range == 0:  # Handle constant sensor values
            x_min -= 0.1
            x_max += 0.1
            domain_range = 0.2
            
        x_min -= domain_expansion * domain_range
        x_max += domain_expansion * domain_range
        
        x_values = np.linspace(x_min, x_max, n_points)
        
        # Compute sigma based on selected method
        if sigma_method == "std":
            sigma = np.std(sensor_data)
            if sigma == 0:  # Handle constant sensor values
                sigma = 0.1
        elif sigma_method == "range":
            sigma = domain_range / 10
        elif sigma_method == "iqr":
            q75, q25 = np.percentile(sensor_data, [75, 25])
            sigma = (q75 - q25) / 2
            if sigma == 0:  # Handle constant sensor values
                sigma = 0.1
        elif sigma_method == "adaptive":
            # Use 10% of the data range as sigma (same as in membership.py)
            sigma = domain_range * 0.1
        else:
            try:
                # Try to convert to float (direct sigma value)
                sigma = float(sigma_method)
            except ValueError:
                raise ValueError(f"Unknown sigma method: {sigma_method}")
        
        # Compute membership function
        mu_values = np.zeros(n_points)
        
        if kernel_type == "gaussian":
            for j, x in enumerate(x_values):
                # Sum of Gaussian kernels
                kernel_sum = 0
                for val in sensor_data:
                    kernel_sum += np.exp(-0.5 * ((x - val) / sigma) ** 2)
                
                # Normalize by window size
                mu_values[j] = kernel_sum / window_size
                
        elif kernel_type == "triangular":
            for j, x in enumerate(x_values):
                # Sum of triangular kernels
                kernel_sum = 0
                for val in sensor_data:
                    dist = abs(x - val)
                    if dist <= sigma:
                        kernel_sum += 1 - (dist / sigma)
                
                # Normalize by window size
                mu_values[j] = kernel_sum / window_size
                
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Store membership function
        membership_functions.append(mu_values)
    
    return x_values, membership_functions


def compute_similarity_per_sensor(
    mu_s1_list: List[np.ndarray],
    mu_s2_list: List[np.ndarray],
    x_values: np.ndarray,
    metric: str = "jaccard",
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute similarity between two windows based on per-sensor membership functions.
    
    Args:
        mu_s1_list: List of membership functions for first window
        mu_s2_list: List of membership functions for second window
        x_values: Domain values
        metric: Similarity metric to use
        weights: Optional weights for each sensor
        
    Returns:
        Similarity value
    """
    n_sensors = len(mu_s1_list)
    
    # Validate input
    if len(mu_s2_list) != n_sensors:
        raise ValueError("Both windows must have the same number of sensors")
    
    # Use equal weights if not provided
    if weights is None:
        weights = np.ones(n_sensors) / n_sensors
    else:
        # Normalize weights to sum to 1
        weights = np.asarray(weights) / np.sum(weights)
    
    # Select similarity function
    if metric.lower() == "jaccard":
        sim_func = similarity_jaccard
    elif metric.lower() == "dice":
        sim_func = similarity_dice
    elif metric.lower() == "cosine":
        sim_func = similarity_cosine
    elif metric.lower() == "euclidean":
        sim_func = similarity_euclidean
    elif metric.lower() == "pearson":
        sim_func = similarity_pearson
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    
    # Compute weighted similarity
    total_similarity = 0.0
    
    for i in range(n_sensors):
        mu_s1 = mu_s1_list[i]
        mu_s2 = mu_s2_list[i]
        
        # Compute similarity for this sensor
        sim = sim_func(mu_s1, mu_s2)
        
        # Add weighted similarity
        total_similarity += weights[i] * sim
    
    return total_similarity


# Define worker function outside the main function so it can be pickled
def compute_similarity_for_pair(args):
    """
    Compute similarity for a pair of windows.
    
    Args:
        args: Tuple containing (i, j, membership_i, membership_j, x_values, metric, weights)
        
    Returns:
        Tuple of (i, j, similarity)
    """
    i, j, membership_i, membership_j, x_values, metric, weights = args
    sim = compute_similarity_per_sensor(
        membership_i, membership_j, x_values,
        metric=metric, weights=weights
    )
    return i, j, sim


def compute_pairwise_similarities_per_sensor(
    windows: List[np.ndarray],
    metric: str = "jaccard",
    kernel_type: str = "gaussian",
    sigma_method: str = "std",
    weights: Optional[np.ndarray] = None,
    n_jobs: int = 1
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for a list of windows using per-sensor membership functions.
    
    Args:
        windows: List of window data arrays
        metric: Similarity metric to use
        kernel_type: Type of kernel for NDG
        sigma_method: Method to compute sigma
        weights: Optional weights for each sensor
        n_jobs: Number of parallel jobs
        
    Returns:
        Similarity matrix of shape (n_windows, n_windows)
    """
    n_windows = len(windows)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_windows, n_windows))
    np.fill_diagonal(similarity_matrix, 1.0)  # Self-similarity = 1
    
    # Precompute membership functions for all windows
    logger.info(f"Computing membership functions for {n_windows} windows...")
    all_memberships = []
    x_values = None
    
    for i, window in enumerate(windows):
        x_vals, memberships = compute_ndg_per_sensor(
            window, kernel_type=kernel_type, sigma_method=sigma_method
        )
        all_memberships.append(memberships)
        
        # Store x_values from the first window (assumed to be the same domain for all)
        if i == 0:
            x_values = x_vals
        
        if (i + 1) % 10 == 0 or i == n_windows - 1:
            logger.info(f"  Processed {i+1}/{n_windows} windows")
    
    # Compute similarities
    logger.info(f"Computing pairwise similarities using {metric} metric...")
    
    if n_jobs > 1:
        # Generate all pairs
        pairs = []
        for i in range(n_windows):
            for j in range(i+1, n_windows):
                pairs.append((i, j, all_memberships[i], all_memberships[j], 
                             x_values, metric, weights))
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(compute_similarity_for_pair, pair) for pair in pairs]
            
            completed = 0
            total = len(pairs)
            
            for future in as_completed(futures):
                try:
                    i, j, sim = future.result()
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Symmetric
                    
                    completed += 1
                    if completed % 100 == 0 or completed == total:
                        logger.info(f"  Processed {completed}/{total} pairs ({completed/total*100:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Error computing similarity: {e}")
    
    else:
        # Sequential processing
        total_pairs = n_windows * (n_windows - 1) // 2
        completed_pairs = 0
        
        for i in range(n_windows):
            for j in range(i+1, n_windows):
                sim = compute_similarity_per_sensor(
                    all_memberships[i], all_memberships[j], x_values,
                    metric=metric, weights=weights
                )
                
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric
                
                completed_pairs += 1
                if completed_pairs % 100 == 0 or completed_pairs == total_pairs:
                    logger.info(f"  Processed {completed_pairs}/{total_pairs} pairs ({completed_pairs/total_pairs*100:.1f}%)")
    
    return similarity_matrix 
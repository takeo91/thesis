"""
Optimized membership function computations.

This module provides high-performance implementations of NDG and other membership
function computations with significant algorithmic and implementation optimizations.

Key optimizations:
- Vectorized spatial queries using efficient data structures
- Optimized kernel computations with pre-computed constants
- Batch processing for multiple evaluation points
- Intelligent caching of expensive intermediate results
- Memory-efficient algorithms for large datasets
"""

from __future__ import annotations
from typing import Optional, Union, List, Tuple
import numpy as np
from scipy.spatial import cKDTree
import warnings

from thesis.core.constants import (
    DEFAULT_CUTOFF_FACTOR,
    GAUSSIAN_INV_TWO_SIGMA_SQ_COEFF,
    EPANECHNIKOV_COEFF,
    SQRT_2PI,
    NUMERICAL_TOLERANCE,
    DEFAULT_GRID_POINTS
)
from thesis.core.validation import validate_array_input, validate_positive_scalar
from thesis.core.logging_config import get_logger
from thesis.core.caching import cached, NDGCache

ArrayLike = Union[List[float], np.ndarray]
logger = get_logger(__name__)


class OptimizedNDGComputer:
    """High-performance NDG computation engine."""
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the NDG computer.
        
        Args:
            cache_enabled: Whether to enable caching for computations
        """
        self.cache_enabled = cache_enabled
        self.cache = NDGCache() if cache_enabled else None
        
        # Pre-computed constants for different kernels
        self._kernel_constants = {
            'gaussian': {
                'normalization': 1.0 / SQRT_2PI,
                'exp_factor': -GAUSSIAN_INV_TWO_SIGMA_SQ_COEFF
            },
            'epanechnikov': {
                'normalization': EPANECHNIKOV_COEFF,
                'support_factor': 1.0
            }
        }
    
    @validate_array_input(allow_empty=False)
    @validate_positive_scalar('sigma')
    def compute_ndg_spatial_ultra_optimized(
        self,
        sensor_data: ArrayLike,
        x_values: ArrayLike,
        sigma: float,
        cutoff_factor: float = DEFAULT_CUTOFF_FACTOR,
        kernel_type: str = 'gaussian'
    ) -> np.ndarray:
        """
        Ultra-optimized spatial NDG computation with vectorized operations.
        
        Optimizations:
        - Efficient spatial indexing with cKDTree
        - Vectorized kernel computations
        - Pre-computed constants
        - Early termination for sparse regions
        
        Args:
            sensor_data: Input sensor data
            x_values: Evaluation points
            sigma: Kernel bandwidth
            cutoff_factor: Spatial cutoff factor (default: 4-sigma)
            kernel_type: Kernel type ('gaussian' or 'epanechnikov')
            
        Returns:
            NDG values at x_values
        """
        sensor_data = np.asarray(sensor_data, dtype=np.float64)
        x_values = np.asarray(x_values, dtype=np.float64)
        
        if sensor_data.size == 0:
            return np.zeros_like(x_values)
        
        # Use caching if enabled
        if self.cache_enabled:
            cache_key = f"ndg_{hash(sensor_data.tobytes())}_{hash(x_values.tobytes())}_{sigma}_{kernel_type}"
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Build spatial index for efficient neighbor queries
        cutoff_distance = cutoff_factor * sigma
        tree = cKDTree(sensor_data.reshape(-1, 1))
        
        # Pre-allocate result array
        ndg_values = np.zeros(len(x_values), dtype=np.float64)
        
        # Get kernel constants
        kernel_constants = self._kernel_constants.get(kernel_type, self._kernel_constants['gaussian'])
        
        # Vectorized computation for each evaluation point
        for i, x in enumerate(x_values):
            # Find neighbors within cutoff distance
            neighbor_indices = tree.query_ball_point([x], r=cutoff_distance)
            
            if not neighbor_indices:
                ndg_values[i] = 0.0
                continue
            
            neighbor_indices = neighbor_indices[0]  # Extract from list
            
            if len(neighbor_indices) == 0:
                ndg_values[i] = 0.0
                continue
            
            # Get neighbor data (vectorized)
            neighbor_data = sensor_data[neighbor_indices]
            
            # Compute kernel values (vectorized)
            if kernel_type == 'gaussian':
                ndg_values[i] = self._compute_gaussian_kernel_vectorized(
                    x, neighbor_data, sigma, kernel_constants
                )
            elif kernel_type == 'epanechnikov':
                ndg_values[i] = self._compute_epanechnikov_kernel_vectorized(
                    x, neighbor_data, sigma, kernel_constants
                )
            else:
                raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        # Cache result if enabled
        if self.cache_enabled:
            self.cache.put(cache_key, ndg_values)
        
        return ndg_values
    
    def _compute_gaussian_kernel_vectorized(
        self,
        x: float,
        neighbor_data: np.ndarray,
        sigma: float,
        constants: dict
    ) -> float:
        """Vectorized Gaussian kernel computation."""
        # Compute distances (vectorized)
        distances = np.abs(neighbor_data - x)
        
        # Compute kernel values (vectorized)
        normalized_distances = distances / sigma
        kernel_values = constants['normalization'] * np.exp(
            constants['exp_factor'] * normalized_distances**2
        ) / sigma
        
        return np.sum(kernel_values)
    
    def _compute_epanechnikov_kernel_vectorized(
        self,
        x: float,
        neighbor_data: np.ndarray,
        sigma: float,
        constants: dict
    ) -> float:
        """Vectorized Epanechnikov kernel computation."""
        # Compute normalized distances (vectorized)
        normalized_distances = np.abs(neighbor_data - x) / sigma
        
        # Apply Epanechnikov kernel (vectorized)
        mask = normalized_distances <= 1.0
        if not np.any(mask):
            return 0.0
        
        valid_distances = normalized_distances[mask]
        kernel_values = constants['normalization'] * (1.0 - valid_distances**2) / sigma
        
        return np.sum(kernel_values)
    
    def compute_ndg_streaming_optimized(
        self,
        sensor_data: ArrayLike,
        x_values: ArrayLike,
        sigma: float,
        batch_size: int = 1000
    ) -> np.ndarray:
        """
        Memory-efficient streaming NDG computation for large datasets.
        
        Args:
            sensor_data: Input sensor data (can be very large)
            x_values: Evaluation points
            sigma: Kernel bandwidth
            batch_size: Batch size for processing
            
        Returns:
            NDG values at x_values
        """
        sensor_data = np.asarray(sensor_data)
        x_values = np.asarray(x_values)
        
        if sensor_data.size == 0:
            return np.zeros_like(x_values)
        
        # Process in batches to manage memory
        n_data = len(sensor_data)
        ndg_result = np.zeros_like(x_values)
        
        for batch_start in range(0, n_data, batch_size):
            batch_end = min(batch_start + batch_size, n_data)
            batch_data = sensor_data[batch_start:batch_end]
            
            # Compute NDG for this batch
            batch_ndg = self.compute_ndg_spatial_ultra_optimized(
                batch_data, x_values, sigma
            )
            
            # Accumulate results
            ndg_result += batch_ndg
        
        return ndg_result
    
    def compute_multi_sigma_ndg(
        self,
        sensor_data: ArrayLike,
        x_values: ArrayLike,
        sigma_values: List[float]
    ) -> np.ndarray:
        """
        Compute NDG for multiple sigma values efficiently.
        
        Optimizations:
        - Shared spatial indexing across sigma values
        - Vectorized computation for all sigmas
        
        Args:
            sensor_data: Input sensor data
            x_values: Evaluation points
            sigma_values: List of sigma values
            
        Returns:
            NDG array of shape (len(x_values), len(sigma_values))
        """
        sensor_data = np.asarray(sensor_data)
        x_values = np.asarray(x_values)
        
        if sensor_data.size == 0:
            return np.zeros((len(x_values), len(sigma_values)))
        
        # Build spatial index once for all sigma values
        max_sigma = max(sigma_values)
        cutoff_distance = DEFAULT_CUTOFF_FACTOR * max_sigma
        tree = cKDTree(sensor_data.reshape(-1, 1))
        
        # Pre-allocate result array
        ndg_results = np.zeros((len(x_values), len(sigma_values)))
        
        # Process each evaluation point
        for i, x in enumerate(x_values):
            # Find neighbors within maximum cutoff distance
            neighbor_indices = tree.query_ball_point([x], r=cutoff_distance)
            
            if not neighbor_indices or len(neighbor_indices[0]) == 0:
                continue
            
            neighbor_indices = neighbor_indices[0]
            neighbor_data = sensor_data[neighbor_indices]
            distances = np.abs(neighbor_data - x)
            
            # Compute NDG for all sigma values (vectorized)
            for j, sigma in enumerate(sigma_values):
                # Filter neighbors within this sigma's cutoff
                sigma_cutoff = DEFAULT_CUTOFF_FACTOR * sigma
                valid_mask = distances <= sigma_cutoff
                
                if not np.any(valid_mask):
                    continue
                
                valid_distances = distances[valid_mask]
                
                # Compute Gaussian kernel values (vectorized)
                normalized_distances = valid_distances / sigma
                kernel_values = (1.0 / (SQRT_2PI * sigma)) * np.exp(
                    -GAUSSIAN_INV_TWO_SIGMA_SQ_COEFF * normalized_distances**2
                )
                
                ndg_results[i, j] = np.sum(kernel_values)
        
        return ndg_results


class PerSensorMembershipOptimized:
    """Optimized per-sensor membership function computation."""
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize per-sensor membership computer."""
        self.ndg_computer = OptimizedNDGComputer(cache_enabled=cache_enabled)
        self.cache_enabled = cache_enabled
    
    def compute_per_sensor_memberships_batch(
        self,
        sensor_data_list: List[np.ndarray],
        x_values: ArrayLike,
        sigma_values: Union[float, List[float]],
        normalize: bool = True
    ) -> List[np.ndarray]:
        """
        Compute membership functions for multiple sensors efficiently.
        
        Args:
            sensor_data_list: List of sensor data arrays
            x_values: Evaluation points
            sigma_values: Sigma value(s) for each sensor
            normalize: Whether to normalize membership functions
            
        Returns:
            List of membership function arrays
        """
        x_values = np.asarray(x_values)
        
        # Handle sigma values
        if isinstance(sigma_values, (int, float)):
            sigma_values = [sigma_values] * len(sensor_data_list)
        
        memberships = []
        
        for sensor_data, sigma in zip(sensor_data_list, sigma_values):
            # Compute NDG for this sensor
            ndg_values = self.ndg_computer.compute_ndg_spatial_ultra_optimized(
                sensor_data, x_values, sigma
            )
            
            # Normalize if requested
            if normalize:
                total = np.sum(ndg_values)
                if total > NUMERICAL_TOLERANCE:
                    ndg_values = ndg_values / total
                else:
                    ndg_values = np.ones_like(ndg_values) / len(ndg_values)
            
            memberships.append(ndg_values)
        
        return memberships


# Convenience functions for backward compatibility
@cached()
def compute_ndg_streaming_optimized(
    sensor_data: ArrayLike,
    x_values: ArrayLike,
    sigma: float
) -> np.ndarray:
    """
    Optimized NDG computation with caching.
    
    Args:
        sensor_data: Input sensor data
        x_values: Evaluation points
        sigma: Kernel bandwidth
        
    Returns:
        NDG values at x_values
    """
    computer = OptimizedNDGComputer(cache_enabled=True)
    return computer.compute_ndg_spatial_ultra_optimized(sensor_data, x_values, sigma)


def compute_per_sensor_memberships_fast(
    sensor_data_list: List[np.ndarray],
    x_values: ArrayLike,
    sigma_values: Union[float, List[float]],
    normalize: bool = True
) -> List[np.ndarray]:
    """
    Fast per-sensor membership computation.
    
    Args:
        sensor_data_list: List of sensor data arrays
        x_values: Evaluation points
        sigma_values: Sigma value(s)
        normalize: Whether to normalize
        
    Returns:
        List of membership functions
    """
    computer = PerSensorMembershipOptimized(cache_enabled=True)
    return computer.compute_per_sensor_memberships_batch(
        sensor_data_list, x_values, sigma_values, normalize
    )


def benchmark_ndg_performance(
    data_sizes: List[int] = [100, 1000, 10000],
    eval_points: int = 100
) -> dict:
    """
    Benchmark NDG computation performance.
    
    Args:
        data_sizes: List of data sizes to test
        eval_points: Number of evaluation points
        
    Returns:
        Performance results
    """
    import time
    
    computer = OptimizedNDGComputer(cache_enabled=False)  # No cache for fair benchmark
    results = {}
    
    x_values = np.linspace(0, 1, eval_points)
    sigma = 0.1
    
    for data_size in data_sizes:
        # Generate test data
        np.random.seed(42)
        sensor_data = np.random.rand(data_size)
        
        # Time the computation
        start_time = time.time()
        _ = computer.compute_ndg_spatial_ultra_optimized(sensor_data, x_values, sigma)
        computation_time = time.time() - start_time
        
        results[data_size] = {
            'time_seconds': computation_time,
            'points_per_second': (data_size * eval_points) / computation_time,
            'time_per_evaluation': computation_time / eval_points
        }
        
        logger.info(f"NDG benchmark - Data size: {data_size}, "
                   f"Time: {computation_time:.4f}s, "
                   f"Rate: {results[data_size]['points_per_second']:.0f} pts/s")
    
    return results
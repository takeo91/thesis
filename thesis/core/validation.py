"""
Input validation decorators and utilities.

This module provides decorators and functions for validating inputs to
similarity metrics, membership functions, and other core operations.
All validation includes comprehensive error messages and type checking.
"""

from functools import wraps
from pathlib import Path
from typing import Any, Callable, Union
import numpy as np

from .constants import NUMERICAL_TOLERANCE, DATA_RANGE_THRESHOLD
from .exceptions import (
    ThesisError, DataValidationError, ConfigurationError, ComputationError, 
    SecurityError, validate_arrays, validate_positive_number, 
    validate_file_path, validate_membership_functions_arrays,
    format_error_message, format_validation_error
)


def validate_membership_functions(func: Callable) -> Callable:
    """
    Decorator to validate membership function inputs.
    
    Validates that:
    - Inputs are convertible to numpy arrays
    - Arrays have the same shape
    - No NaN or infinite values
    - All values are non-negative
    - Arrays are not empty
    
    Args:
        func: Function that takes mu1, mu2 as first two arguments
        
    Returns:
        Wrapped function with input validation
        
    Raises:
        DataValidationError: If validation fails
    """
    @wraps(func)
    def wrapper(mu1, mu2, *args, **kwargs):
        try:
            mu1, mu2 = np.asarray(mu1, dtype=np.float64), np.asarray(mu2, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Cannot convert inputs to numpy arrays: {e}")
        
        # Check shapes
        if mu1.shape != mu2.shape:
            raise DataValidationError(
                f"Membership functions must have same shape. "
                f"Got {mu1.shape} and {mu2.shape}"
            )
        
        # Check for empty arrays
        if mu1.size == 0:
            raise DataValidationError("Membership functions cannot be empty")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(mu1)) or np.any(np.isnan(mu2)):
            raise DataValidationError("NaN values not allowed in membership functions")
        
        if np.any(np.isinf(mu1)) or np.any(np.isinf(mu2)):
            raise DataValidationError("Infinite values not allowed in membership functions")
        
        # Check for negative values
        if np.any(mu1 < 0) or np.any(mu2 < 0):
            raise DataValidationError("Membership functions must be non-negative")
        
        return func(mu1, mu2, *args, **kwargs)
    
    return wrapper


def validate_array_input(allow_empty: bool = False, min_dims: int = 1, max_dims: int = 2) -> Callable:
    """
    Decorator to validate array inputs for the first argument.
    
    Args:
        allow_empty: Whether to allow empty arrays
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            try:
                data = np.asarray(data, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise DataValidationError(f"Cannot convert input to numpy array: {e}")
            
            # Check empty arrays
            if not allow_empty and data.size == 0:
                raise DataValidationError("Input array cannot be empty")
            
            # Check dimensions
            if data.ndim < min_dims or data.ndim > max_dims:
                raise DataValidationError(
                    f"Array must have {min_dims}-{max_dims} dimensions, got {data.ndim}"
                )
            
            # Check for NaN or infinite values
            if np.any(np.isnan(data)):
                raise DataValidationError("NaN values not allowed in input data")
            
            if np.any(np.isinf(data)):
                raise DataValidationError("Infinite values not allowed in input data")
            
            return func(data, *args, **kwargs)
        
        return wrapper
    return decorator


def validate_positive_scalar(param_name: str) -> Callable:
    """
    Decorator to validate that a named parameter is a positive scalar.
    
    Args:
        param_name: Name of the parameter to validate
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if param_name in kwargs:
                value = kwargs[param_name]
                if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                    raise DataValidationError(
                        f"Parameter '{param_name}' must be a positive finite scalar, got {value}"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_probability_array(func: Callable) -> Callable:
    """
    Decorator to validate probability distributions.
    
    Validates that:
    - Input is a valid array
    - All values are in [0, 1]
    - Values sum to approximately 1.0
    
    Args:
        func: Function with probability array as first argument
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(probs, *args, **kwargs):
        try:
            probs = np.asarray(probs, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise DataValidationError(f"Cannot convert to probability array: {e}")
        
        # Check range
        if np.any(probs < 0) or np.any(probs > 1):
            raise DataValidationError("Probability values must be in [0, 1]")
        
        # Check normalization (within tolerance)
        prob_sum = np.sum(probs)
        if abs(prob_sum - 1.0) > NUMERICAL_TOLERANCE:
            raise DataValidationError(
                f"Probability array must sum to 1.0, got {prob_sum}"
            )
        
        return func(probs, *args, **kwargs)
    
    return wrapper


def safe_path_join(base_dir: Path, filename: str) -> Path:
    """
    Safely join paths, preventing directory traversal attacks.
    
    Args:
        base_dir: Base directory path
        filename: Filename to join
        
    Returns:
        Safe combined path
        
    Raises:
        SecurityError: If path traversal is detected
    """
    base_dir = Path(base_dir).resolve()
    full_path = (base_dir / filename).resolve()
    
    # Check if the resolved path is still within the base directory
    try:
        full_path.relative_to(base_dir)
    except ValueError:
        raise SecurityError(f"Path traversal attempt detected: {filename}")
    
    return full_path


def validate_dataset_path(func: Callable) -> Callable:
    """
    Decorator to validate dataset file paths.
    
    Args:
        func: Function with file path as first argument
        
    Returns:
        Wrapped function with path validation
    """
    @wraps(func)
    def wrapper(file_path, *args, **kwargs):
        try:
            file_path = Path(file_path).resolve()
        except (OSError, ValueError) as e:
            raise DataValidationError(f"Invalid file path: {e}")
        
        if not file_path.exists():
            raise DataValidationError(f"File does not exist: {file_path}")
        
        if not file_path.is_file():
            raise DataValidationError(f"Path is not a file: {file_path}")
        
        return func(file_path, *args, **kwargs)
    
    return wrapper


def safe_column_slice(df, start: int, end: int):
    """
    Safely slice DataFrame columns with bounds checking.
    
    Args:
        df: pandas DataFrame
        start: Start column index
        end: End column index
        
    Returns:
        Column slice
        
    Raises:
        DataValidationError: If indices are out of bounds
    """
    if end > len(df.columns):
        raise DataValidationError(
            f"Column range [{start}:{end}] exceeds DataFrame width {len(df.columns)}"
        )
    
    if start < 0 or end < 0:
        raise DataValidationError(
            f"Column indices must be non-negative, got [{start}:{end}]"
        )
    
    if start >= end:
        raise DataValidationError(
            f"Start index must be less than end index, got [{start}:{end}]"
        )
    
    return df.columns[start:end]


def validate_window_config(func: Callable) -> Callable:
    """
    Decorator to validate windowing configuration parameters.
    
    Args:
        func: Function with window configuration parameters
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate window_size if present
        if 'window_size' in kwargs:
            window_size = kwargs['window_size']
            if not isinstance(window_size, int) or window_size <= 0:
                raise ConfigurationError(
                    f"window_size must be a positive integer, got {window_size}"
                )
        
        # Validate overlap_ratio if present
        if 'overlap_ratio' in kwargs:
            overlap_ratio = kwargs['overlap_ratio']
            if not (0 <= overlap_ratio < 1):
                raise ConfigurationError(
                    f"overlap_ratio must be in [0, 1), got {overlap_ratio}"
                )
        
        return func(*args, **kwargs)
    
    return wrapper
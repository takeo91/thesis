"""
Standardized exception hierarchy and error handling utilities for thesis project.

This module provides a consistent approach to error handling across all thesis modules,
with specific exception types, standardized error messages, and validation utilities.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Union
import numpy as np
from pathlib import Path


class ThesisError(Exception):
    """Base exception for all thesis-related errors.
    
    All custom exceptions in the thesis codebase should inherit from this class
    to provide a consistent exception hierarchy and enable blanket error handling
    when needed.
    """
    
    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{super().__str__()} (Context: {context_str})"
        return super().__str__()


class DataValidationError(ThesisError):
    """Raised when input data validation fails.
    
    This exception should be used for all data-related validation failures,
    including array shape mismatches, empty data, invalid ranges, etc.
    """
    pass


class ComputationError(ThesisError):
    """Raised when mathematical computation fails.
    
    This exception should be used for computational failures that are not
    due to invalid input data, such as numerical instability, convergence
    failures, or mathematical edge cases.
    """
    pass


class ConfigurationError(ThesisError):
    """Raised when configuration parameters are invalid.
    
    This exception should be used for invalid parameter combinations,
    missing required configuration, or invalid parameter values.
    """
    pass


class SecurityError(ThesisError):
    """Raised when security validation fails.
    
    This exception should be used for path traversal attempts,
    unsafe file operations, or other security-related issues.
    """
    pass


class DatasetError(ThesisError):
    """Raised when dataset operations fail.
    
    This exception should be used for dataset-specific errors such as
    missing files, corrupted data, or unsupported dataset formats.
    """
    pass


# Error message formatting utilities
def format_error_message(operation: str, cause: str, suggestion: Optional[str] = None, **context) -> str:
    """
    Format error messages consistently across the codebase.
    
    Args:
        operation: The operation that failed (e.g., "compute similarity", "load dataset")
        cause: The specific cause of the failure
        suggestion: Optional suggestion for fixing the error
        **context: Additional context information to include
        
    Returns:
        Formatted error message string
    """
    msg = f"Failed to {operation}: {cause}"
    
    if suggestion:
        msg += f". Suggestion: {suggestion}"
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        msg += f" (Context: {context_str})"
    
    return msg


def format_validation_error(parameter: str, value: Any, requirement: str, **context) -> str:
    """
    Format validation error messages consistently.
    
    Args:
        parameter: Name of the parameter that failed validation
        value: The invalid value
        requirement: Description of what the value should be
        **context: Additional context information
        
    Returns:
        Formatted validation error message
    """
    msg = f"Invalid {parameter}: got {value}, expected {requirement}"
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        msg += f" (Context: {context_str})"
    
    return msg


# Validation utilities
def validate_arrays(*arrays, min_size: int = 1, same_shape: bool = True, 
                   allow_empty: bool = False, parameter_names: Optional[Sequence[str]] = None) -> None:
    """
    Validate multiple arrays with consistent error messages.
    
    Args:
        *arrays: Arrays to validate
        min_size: Minimum required size for each array
        same_shape: Whether all arrays must have the same shape
        allow_empty: Whether to allow empty arrays
        parameter_names: Names of the parameters for error messages
        
    Raises:
        DataValidationError: If validation fails
    """
    if not arrays:
        raise DataValidationError("No arrays provided for validation")
    
    # Convert to numpy arrays and validate individually
    converted_arrays = []
    for i, arr in enumerate(arrays):
        param_name = parameter_names[i] if parameter_names and i < len(parameter_names) else f"array_{i}"
        
        try:
            arr = np.asarray(arr)
        except Exception as e:
            raise DataValidationError(
                format_error_message(
                    f"convert {param_name} to array", 
                    str(e),
                    "ensure input is array-like"
                )
            ) from e
        
        if not allow_empty and arr.size == 0:
            raise DataValidationError(
                format_validation_error(param_name, "empty array", "non-empty array")
            )
        
        if arr.size < min_size:
            raise DataValidationError(
                format_validation_error(
                    param_name, 
                    f"size {arr.size}", 
                    f"minimum size {min_size}"
                )
            )
        
        # Check for NaN/Inf values
        if np.issubdtype(arr.dtype, np.floating):
            if np.any(np.isnan(arr)):
                raise DataValidationError(
                    format_validation_error(param_name, "contains NaN", "finite values only")
                )
            if np.any(np.isinf(arr)):
                raise DataValidationError(
                    format_validation_error(param_name, "contains Inf", "finite values only")
                )
        
        converted_arrays.append(arr)
    
    # Check shape consistency if required
    if same_shape and len(converted_arrays) > 1:
        shapes = [arr.shape for arr in converted_arrays]
        if len(set(shapes)) > 1:
            param_names_str = ", ".join(parameter_names) if parameter_names else f"arrays 0-{len(shapes)-1}"
            raise DataValidationError(
                format_error_message(
                    "validate array shapes",
                    f"inconsistent shapes: {shapes}",
                    "ensure all arrays have the same shape",
                    parameters=param_names_str
                )
            )


def validate_positive_number(value: Union[int, float], parameter_name: str, 
                           allow_zero: bool = False, max_value: Optional[float] = None) -> None:
    """
    Validate that a number is positive.
    
    Args:
        value: Value to validate
        parameter_name: Name of the parameter
        allow_zero: Whether to allow zero values
        max_value: Optional maximum allowed value
        
    Raises:
        DataValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise DataValidationError(
            format_validation_error(parameter_name, type(value).__name__, "numeric type")
        )
    
    if np.isnan(value) or np.isinf(value):
        raise DataValidationError(
            format_validation_error(parameter_name, value, "finite number")
        )
    
    min_allowed = 0 if allow_zero else 0 + np.finfo(float).eps
    if value < min_allowed:
        requirement = "non-negative" if allow_zero else "positive"
        raise DataValidationError(
            format_validation_error(parameter_name, value, f"{requirement} number")
        )
    
    if max_value is not None and value > max_value:
        raise DataValidationError(
            format_validation_error(parameter_name, value, f"<= {max_value}")
        )


def validate_probability(value: Union[int, float], parameter_name: str) -> None:
    """
    Validate that a value is a valid probability (between 0 and 1).
    
    Args:
        value: Value to validate
        parameter_name: Name of the parameter
        
    Raises:
        DataValidationError: If validation fails
    """
    validate_positive_number(value, parameter_name, allow_zero=True, max_value=1.0)


def validate_file_path(path: Union[str, Path], parameter_name: str, 
                      must_exist: bool = True, must_be_file: bool = True) -> Path:
    """
    Validate file path with security checks.
    
    Args:
        path: Path to validate
        parameter_name: Name of the parameter
        must_exist: Whether the path must exist
        must_be_file: Whether the path must be a file (vs directory)
        
    Returns:
        Validated Path object
        
    Raises:
        SecurityError: If path contains security issues
        DatasetError: If path validation fails
    """
    try:
        path = Path(path)
    except Exception as e:
        raise DatasetError(
            format_error_message(f"parse {parameter_name}", str(e), "provide valid path string")
        ) from e
    
    # Security validation - prevent path traversal
    try:
        resolved_path = path.resolve()
        if ".." in str(path) or str(resolved_path) != str(path.resolve(strict=False)):
            raise SecurityError(
                format_error_message(
                    f"validate {parameter_name}",
                    "potential path traversal detected",
                    "use absolute paths without '..' components"
                )
            )
    except Exception as e:
        raise SecurityError(
            format_error_message(f"resolve {parameter_name}", str(e))
        ) from e
    
    if must_exist and not path.exists():
        raise DatasetError(
            format_validation_error(parameter_name, f"'{path}'", "existing path")
        )
    
    if must_exist and must_be_file and not path.is_file():
        raise DatasetError(
            format_validation_error(parameter_name, f"'{path}'", "existing file")
        )
    
    return path


def validate_membership_functions_arrays(mu1: Any, mu2: Any, x_values: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate membership function arrays for similarity computation.
    
    Args:
        mu1: First membership function
        mu2: Second membership function
        x_values: Domain values
        
    Returns:
        Tuple of validated numpy arrays
        
    Raises:
        DataValidationError: If validation fails
    """
    validate_arrays(mu1, mu2, x_values, same_shape=True, parameter_names=["mu1", "mu2", "x_values"])
    
    mu1_arr = np.asarray(mu1, dtype=float)
    mu2_arr = np.asarray(mu2, dtype=float)
    x_values_arr = np.asarray(x_values, dtype=float)
    
    # Validate membership function properties
    if np.any(mu1_arr < 0) or np.any(mu2_arr < 0):
        raise DataValidationError(
            "Membership functions must be non-negative. "
            "Suggestion: check your membership function computation"
        )
    
    return mu1_arr, mu2_arr, x_values_arr


# Graceful error handling utilities
def handle_computation_with_fallback(primary_func, fallback_func, *args, 
                                   fallback_on: tuple = (ComputationError, ValueError),
                                   logger=None, **kwargs):
    """
    Execute computation with graceful fallback on specified errors.
    
    Args:
        primary_func: Primary function to try
        fallback_func: Fallback function if primary fails
        *args: Arguments to pass to functions
        fallback_on: Tuple of exception types that trigger fallback
        logger: Optional logger for warnings
        **kwargs: Keyword arguments to pass to functions
        
    Returns:
        Result from primary or fallback function
    """
    try:
        return primary_func(*args, **kwargs)
    except fallback_on as e:
        if logger:
            logger.warning(f"Primary computation failed ({e}), using fallback")
        return fallback_func(*args, **kwargs)


def safe_computation_wrapper(func, default_value=None, logger=None, 
                           error_context: Optional[dict] = None):
    """
    Decorator/wrapper for safe computation that logs errors and returns defaults.
    
    Args:
        func: Function to wrap
        default_value: Value to return on error
        logger: Optional logger for error reporting
        error_context: Additional context for error messages
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if logger:
                context_str = f" (Context: {error_context})" if error_context else ""
                logger.error(f"Computation failed in {func.__name__}: {e}{context_str}")
            return default_value
    
    return wrapper
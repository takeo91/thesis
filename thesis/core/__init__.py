"""
Core utilities for thesis project.

This module provides fundamental utilities for data processing,
including parsing, preprocessing, and metadata extraction.
"""

from thesis.core.utils import (
    extract_column_names,
    extract_labels,
    extract_metadata,
    get_dataset_specific_window_sizes,
)
# Note: preprocessing imports removed to prevent circular dependency
# Import directly: from thesis.core.preprocessing import normalize_data, standardize_data 
from thesis.core.constants import *
from thesis.core.validation import (
    validate_membership_functions,
    validate_array_input,
    validate_positive_scalar,
    validate_probability_array,
    validate_dataset_path,
    validate_window_config,
    safe_path_join,
    safe_column_slice,
    DataValidationError,
    ConfigurationError,
    ComputationError,
    SecurityError,
    ThesisError,
)
from thesis.core.logging_config import (
    setup_logging,
    get_logger,
    log_experiment_start,
    log_experiment_progress,
    log_experiment_result,
    log_experiment_complete,
)

# Define what should be available when importing from this package
__all__ = [
    "extract_column_names",
    "extract_labels",
    "extract_metadata",
    # "normalize_data",  # Available via direct import
    # "standardize_data",  # Available via direct import
    "get_dataset_specific_window_sizes",
    # Constants from constants module
    "NUMERICAL_TOLERANCE",
    "STRICT_NUMERICAL_TOLERANCE",
    "DEFAULT_CUTOFF_FACTOR",
    "SQRT_2PI",
    "EPANECHNIKOV_COEFF",
    "DEFAULT_MIN_SAMPLES_PER_CLASS",
    # Validation decorators and functions
    "validate_membership_functions",
    "validate_array_input",
    "validate_positive_scalar",
    "validate_probability_array",
    "validate_dataset_path",
    "validate_window_config",
    "safe_path_join",
    "safe_column_slice",
    # Exception classes
    "DataValidationError",
    "ConfigurationError",
    "ComputationError",
    "SecurityError",
    "ThesisError",
    # Logging utilities
    "setup_logging",
    "get_logger",
    "log_experiment_start",
    "log_experiment_progress",
    "log_experiment_result",
    "log_experiment_complete",
] 
"""
Constants module for thesis project.

This module centralizes all magic numbers and hardcoded constants
used throughout the fuzzy similarity analysis system.

All constants are documented with their mathematical justification
and usage context to improve code maintainability and understanding.
"""

import math
import numpy as np
from typing import List, Final

# =============================================================================
# NUMERICAL TOLERANCES AND STABILITY
# =============================================================================

# General numerical tolerance for division by zero protection and comparisons
NUMERICAL_TOLERANCE: Final[float] = 1e-9

# Strict tolerance for probability computations and critical operations
STRICT_NUMERICAL_TOLERANCE: Final[float] = 1e-12

# Minimum value for data range to avoid constant signal issues
DATA_RANGE_TOLERANCE: Final[float] = 1e-9

# Minimum allowable sigma value for kernel computations
MIN_SIGMA_VALUE: Final[float] = 1e-9

# Threshold for normalization operations
NORMALIZATION_THRESHOLD: Final[float] = 1e-9

# Data range threshold for detecting constant signals
DATA_RANGE_THRESHOLD: Final[float] = 1e-9

# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================

# Pre-computed mathematical constants for kernel functions
SQRT_2PI: Final[float] = math.sqrt(2.0 * math.pi)  # Gaussian normalization
SQRT_HALF: Final[float] = 1.0 / math.sqrt(2.0)     # Hellinger distance coefficient

# Gaussian kernel coefficient: 0.5 / sigma^2
GAUSSIAN_INV_TWO_SIGMA_SQ_COEFF: Final[float] = 0.5

# Kernel-specific coefficients based on mathematical definitions
EPANECHNIKOV_COEFF: Final[float] = 0.75          # 3/4 for Epanechnikov kernel
QUARTIC_COEFF: Final[float] = 15.0 / 16.0        # 15/16 for quartic kernel
COSINE_COEFF: Final[float] = math.pi / 4.0       # π/4 for cosine kernel

# =============================================================================
# ALGORITHMIC PARAMETERS
# =============================================================================

# NDG spatial optimization - 4-sigma rule covers 99.99% of normal distribution
DEFAULT_CUTOFF_FACTOR: Final[float] = 4.0

# Pre-computed 4-sigma squared coefficient for optimization
FOUR_SIGMA_CUTOFF_SQUARED_COEFF: Final[float] = 16.0  # (4*sigma)^2

# Memory management for large datasets
DEFAULT_CHUNK_SIZE: Final[int] = 10_000

# Default sigma calculation as fraction of data range
DEFAULT_SIGMA_RATIO: Final[float] = 0.1

# Domain expansion factor for membership function support
DOMAIN_MARGIN_FACTOR: Final[float] = 0.1

# =============================================================================
# DEFAULT CONFIGURATION VALUES
# =============================================================================

# Windowing configuration defaults
DEFAULT_MIN_SAMPLES_PER_CLASS: Final[int] = 5
DEFAULT_WINDOW_SIZES: Final[List[int]] = [128, 256]
DEFAULT_OVERLAP_RATIOS: Final[List[float]] = [0.5, 0.7]

# Base windowing configuration for experiments
BASE_WINDOW_SIZES: Final[List[int]] = [120]
BASE_OVERLAP_RATIOS: Final[List[float]] = [0.5]

# Grid resolution for numerical integration and function evaluation
DEFAULT_GRID_POINTS: Final[int] = 100

# Default top-k value for retrieval evaluation
DEFAULT_TOPK: Final[int] = 5

# =============================================================================
# PREPROCESSING CONSTANTS
# =============================================================================

# Value assigned to constant data during normalization
CONSTANT_DATA_NORM_VALUE: Final[float] = 0.5

# =============================================================================
# SIMILARITY METRIC CONSTANTS
# =============================================================================

# Information-theoretic metrics
MI_DEFAULT_BINS_DIVISOR: Final[int] = 4  # bins = max(3, len(data) // 4)
MI_COMPUTATION_TOLERANCE: Final[float] = 1e-12
RENYI_ALPHA_KL_THRESHOLD: Final[float] = 1.0  # Alpha value for KL divergence

# Probability distribution handling
PROB_DIST_EPSILON: Final[float] = 1e-12
PROBABILITY_EPSILON: Final[float] = 1e-12

# KDE parameters
KDE_BANDWIDTH_MIN: Final[float] = 1e-12

# =============================================================================
# EXPERIMENTAL CONSTANTS
# =============================================================================

# Default sigma values for experimental evaluation
DEFAULT_SIGMA_VALUES: Final[List[float]] = [0.01, 0.1, 0.2, 0.5]
DEFAULT_RELATIVE_SIGMA_VALUES: Final[List[str]] = ["r0.01", "r0.1", "r0.2", "r0.5"]

# =============================================================================
# DATASET-SPECIFIC CONSTANTS
# =============================================================================

# Opportunity dataset structure
OPPORTUNITY_SENSOR_COLS_START: Final[int] = 1
OPPORTUNITY_SENSOR_COLS_END: Final[int] = 243

# PAMAP2 dataset structure  
PAMAP2_ACTIVITY_COL: Final[int] = 1

# Maximum fraction of missing values allowed in datasets
MISSING_VALUE_THRESHOLD: Final[float] = 0.95
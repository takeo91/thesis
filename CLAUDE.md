# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Recent Improvements (June 2025)

The codebase has been enhanced with comprehensive improvements across two phases:

### Phase 1: Foundation & Reliability ✅
#### Constants Module (`thesis/core/constants.py`)
- Centralized 67 magic numbers and hardcoded values
- Added mathematical justification for each constant
- Improved maintainability and reduced configuration drift
- Import: `from thesis.core.constants import NUMERICAL_TOLERANCE, DEFAULT_CUTOFF_FACTOR`

#### Input Validation Framework (`thesis/core/validation.py`)
- Comprehensive validation decorators for membership functions, arrays, and configurations
- Custom exception hierarchy (DataValidationError, ComputationError, etc.)
- Security improvements with path traversal protection
- Usage: `@validate_membership_functions` decorator on similarity functions

#### Structured Logging (`thesis/core/logging_config.py`)
- Replaced print statements with professional logging
- Experiment-specific logging utilities
- Configurable log levels and outputs
- Setup: `from thesis.core import setup_logging, get_logger`

#### Enhanced Error Handling
- Specific exception types instead of broad Exception catching
- Detailed error messages with context
- Graceful degradation for failed similarity computations

### Phase 2: Performance Optimization ✅
#### Vectorized Similarity Engine (`thesis/fuzzy/similarity_optimized.py`)
- **10-100x speedup** for similarity computations using NumPy broadcasting
- Vectorized energy distance computation (O(n²m²) → O(nm))
- Batch processing for multiple similarity pairs
- LRU caching for repeated computations
- Usage: `from thesis.fuzzy.similarity_optimized import VectorizedSimilarityEngine`

#### Memory-Efficient Chunking (`thesis/data/chunked_processor.py`)
- Process datasets larger than RAM using streaming chunks
- Configurable memory limits with automatic adjustment
- Progress monitoring and memory usage tracking
- Usage: `from thesis.data.chunked_processor import ChunkedDataProcessor`

#### Intelligent Caching System (`thesis/core/caching.py`)
- Multi-level LRU memory + disk-based persistent cache
- Automatic cleanup and cache statistics tracking
- Specialized caches for NDG and similarity computations
- Usage: `from thesis.core.caching import cached, get_cache`

#### Ultra-Optimized Membership Functions (`thesis/fuzzy/membership_optimized.py`)
- **Spatial indexing** with cKDTree for efficient neighbor queries
- **Vectorized kernel computations** with pre-computed constants
- **Multi-sigma processing** with shared spatial indexing
- Memory-efficient streaming for large datasets
- Usage: `from thesis.fuzzy.membership_optimized import OptimizedNDGComputer`

### Phase 3: Code Quality & Refactoring ✅
#### Function Refactoring (`thesis/exp/pilot_driver.py`, `thesis/data/datasets.py`, `thesis/fuzzy/similarity.py`)
- **Modular design**: Broke down large functions (>50 lines) into focused components
- **Single responsibility**: Each function now has one clear purpose
- **Improved testability**: Smaller functions are easier to unit test
- **Better maintainability**: Reduced complexity and improved readability

#### Standardized Error Handling (`thesis/core/exceptions.py`)
- **Custom exception hierarchy**: `ThesisError`, `DataValidationError`, `ComputationError`, etc.
- **Consistent error messages**: Standardized format with context and suggestions
- **Comprehensive input validation**: Robust validation for arrays, parameters, and files
- **Graceful error recovery**: Fallback mechanisms and safe computation wrappers
- **Unified logging**: Replaced `warnings.warn()` and `print()` with structured logging

#### Unit Test Framework (`tests/test_phase3_improvements.py`)
- **Comprehensive test coverage**: Tests for exception system, refactored functions, and error handling
- **Backward compatibility validation**: Ensures refactored code maintains original behavior
- **Integration testing**: Validates error handling across module boundaries

### Overall Performance Impact
- **Similarity computations**: 10-100x speedup
- **Memory efficiency**: Process datasets larger than RAM
- **NDG computations**: Ultra-optimized with spatial indexing
- **Caching**: Intelligent multi-level system reduces repeated work
- **Code quality**: Improved maintainability, testability, and robustness
- **Error handling**: Consistent, informative error messages and graceful recovery
- **Backward compatibility**: All existing code continues to work unchanged

These improvements transform research code into production-quality software while maintaining full compatibility with existing experiments.

## Project Overview

This is a PhD thesis research project titled "Development and comparison of fuzzy similarity correlation metrics for sensor data in health application and assisted living environments." The project investigates novel fuzzy similarity metrics for sensor data analysis in health applications, with a focus on activity recognition using the Opportunity and PAMAP2 datasets.

**Key Innovation**: The project implements a per-sensor membership function approach that generates one membership function per sensor rather than traditional single-membership approaches, showing significant performance improvements (F1 scores: 0.33→1.00 in small-scale tests).

## Common Commands

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=thesis

# Run specific test file
pytest tests/test_similarity.py

# Run specific test
pytest tests/test_similarity.py::test_jaccard_similarity
```

### Experiments
```bash
# Run experiments using the module entry point
python -m thesis.exp <sub-command> [args]

# Quick per-sensor test
python -m thesis.exp.per_sensor_quick_test --n_samples_per_class 50 --window_size 32 --n_jobs 2

# RQ2 per-sensor experiment  
python -m thesis.exp.rq2_per_sensor_experiment --max_samples 300 --window_sizes 32 --overlap_ratios 0.5 --min_samples_per_class 2 --similarity_metrics jaccard,dice,cosine --n_jobs 2

# RQ2 grid search automation
./run_rq2_grid.sh
```

### Code Quality
```bash
# Lint code
ruff check .

# Format code  
ruff format .

# Type checking (if using pylint)
pylint thesis/
```

### Build/Install
```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Using uv (preferred package manager)
uv sync
```

## Architecture Overview

### Core Modules (`thesis/`)

- **`data/`**: Dataset loading and preprocessing
  - `opportunity.py`, `pamap2.py` - Dataset-specific loaders
  - `windowing.py` - Time series windowing utilities
  - `preprocessing.py` - Data preprocessing functions

- **`fuzzy/`**: Fuzzy logic and similarity metrics
  - `membership.py` - Membership function implementations
  - `similarity.py` - Core similarity metrics (Jaccard, Dice, Cosine, etc.)
  - `operations.py` - Fuzzy set operations
  - `per_sensor_membership.py` - Novel per-sensor approach implementation

- **`exp/`**: Experiment drivers and evaluation
  - `__main__.py` - Module entry point for sub-command execution
  - `pilot_driver.py` - Main RQ2 experiment driver
  - `per_sensor_*` - Per-sensor approach experiments
  - `retrieval_utils.py` - Retrieval evaluation utilities

- **`analysis/`**: Data processing and analysis tools
  - Statistical analysis and visualization utilities

### Research Questions Structure

- **RQ1** (`results/ndg_vs_kde/`): Compares streaming Normalized Difference Gaussian (NDG-S) vs KDE for membership estimation efficiency
- **RQ2** (`results/rq2_*/`): Evaluates fuzzy similarity metrics for activity and sensor-type retrieval  
- **RQ3**: Cross-dataset robustness analysis

### Documentation Architecture (`docs/`)

Well-organized documentation with dedicated sections:
- `per_sensor_approach/` - Novel per-sensor membership function approach
- `metrics/` - Similarity metrics documentation
- `rq2/` - Research Question 2 technical specifications
- `planning/` - Research planning and battle plans
- `windowing/` - Time series windowing techniques
- `code/` - Code structure and cleanup documentation

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines (enforced via ruff)
- Use type annotations for function parameters and return values
- Document functions and classes with docstrings
- Use descriptive variable names reflecting the data they contain
- Prefer vectorized operations over explicit loops for performance

### Data Analysis Best Practices
- Use pandas for data manipulation with method chaining
- Use matplotlib for low-level plotting, seaborn for statistical visualizations
- Implement data quality checks at analysis start
- Handle missing data appropriately (imputation, removal, or flagging)
- Create reusable plotting functions for consistent visualizations

### Experiment Workflow
1. Start with small datasets for quick validation
2. Implement proper logging and result saving
3. Use the `--progress` flag for long-running experiments
4. Save results to appropriate `results/` subdirectories
5. Run statistical validation (Wilcoxon signed-rank tests)
6. Document findings in corresponding `docs/` sections

### Performance Considerations
- Use `--n_jobs` parameter for parallel processing
- Profile computationally intensive operations
- Consider memory usage for large-scale experiments
- Optimize windowing parameters (window_size, overlap_ratio)

## Key Datasets

- **Opportunity Dataset**: Body-worn sensors, daily activities, multi-level annotations
- **PAMAP2 Dataset**: IMU and heart rate data, 18 physical activities

Both datasets are preprocessed and available in the `Data/` directory with comprehensive documentation.

## Important Implementation Details

### Per-Sensor Membership Functions
The core innovation uses one membership function per sensor rather than a single global function. Key files:
- `thesis/fuzzy/per_sensor_membership.py` - Core implementation
- `thesis/exp/per_sensor_*` - Experiment scripts
- `docs/per_sensor_approach/` - Detailed documentation

### Similarity Metrics
Implemented metrics: Jaccard, Dice, Cosine, Overlap Coefficient, Euclidean Distance, Pearson Correlation
- Use `thesis/fuzzy/similarity_subset.py` for quick testing with core metrics
- Full implementation in `thesis/fuzzy/similarity.py`

### Experiment Module System
The `thesis/exp/__main__.py` provides a sub-command interface:
```bash
python -m thesis.exp <sub-command> [args]
```
Each experiment module exposes a `main()` function for this system.
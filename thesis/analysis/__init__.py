"""
Analysis utilities for thesis project.

This module provides functions for running experiments,
analyzing results, and visualizing fuzzy similarity metrics performance.
"""

from thesis.analysis.experiment import (
    generate_case_data,
    run_experiment,
    run_cases_parallel,
    plot_membership_functions,
    save_results_to_csv
)

from thesis.analysis.data_processor import (
    SensorDataProcessor,
    process_opportunity_dataset
)

# Define what should be available when importing from this package
__all__ = [
    "generate_case_data",
    "run_experiment",
    "run_cases_parallel",
    "plot_membership_functions",
    "save_results_to_csv",
    "SensorDataProcessor",
    "process_opportunity_dataset"
] 
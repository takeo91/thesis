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

# Import from data.datasets instead of local data_processor
from thesis.data.datasets import (
    SensorDataset,
    OpportunityDataset,
    PAMAP2Dataset,
    create_opportunity_dataset,
    create_pamap2_dataset
)

# Define what should be available when importing from this package
__all__ = [
    "generate_case_data",
    "run_experiment",
    "run_cases_parallel",
    "plot_membership_functions",
    "save_results_to_csv",
    "SensorDataset",
    "OpportunityDataset",
    "PAMAP2Dataset",
    "create_opportunity_dataset",
    "create_pamap2_dataset"
] 
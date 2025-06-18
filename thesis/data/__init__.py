"""
Data processing and loading package for thesis research.

This package provides functionality for:
- Dataset loading and preprocessing (Opportunity, PAMAP2, etc.)
- Time series windowing for activity classification
- Data management and utilities
"""

# Import dataset functionality
from .datasets import (
    SensorDataset,
    OpportunityDataset,
    PAMAP2Dataset,
    create_opportunity_dataset,
    create_pamap2_dataset,
)

# Import windowing functionality
from .windowing import (
    WindowConfig,
    WindowedData,
    create_sliding_windows,
    filter_windowed_data_by_class_count,
    create_multiple_window_configs,
    windowed_data_to_dataframe,
    get_windowing_summary,
    demo_windowing,
)

__all__ = [
    # Dataset classes and functions
    "SensorDataset",
    "OpportunityDataset",
    "PAMAP2Dataset",
    "create_opportunity_dataset",
    "create_pamap2_dataset",
    # Windowing classes and functions
    "WindowConfig",
    "WindowedData",
    "create_sliding_windows",
    "filter_windowed_data_by_class_count",
    "create_multiple_window_configs",
    "windowed_data_to_dataframe",
    "get_windowing_summary",
    "demo_windowing",
]

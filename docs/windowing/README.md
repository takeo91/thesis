# Time Series Windowing Documentation

This directory contains documentation for the time series windowing techniques used in the thesis project.

## Documentation Files

1. [Time Series Windowing Summary](TIMESERIES_WINDOWING_SUMMARY.md) - Summary of time series windowing techniques

## Windowing Overview

Time series windowing is a critical preprocessing step for sensor data analysis. It involves dividing continuous sensor data into fixed-size segments (windows) that can be processed individually.

## Implemented Techniques

The thesis implements the following windowing techniques:

1. **Fixed-Size Windowing**: Divides the data into windows of a fixed size
2. **Sliding Window**: Creates overlapping windows by sliding a fixed-size window over the data
3. **Activity-Based Windowing**: Creates windows based on activity boundaries

## Key Parameters

The windowing process is controlled by several parameters:

1. **Window Size**: The number of samples in each window
2. **Overlap Ratio**: The percentage of overlap between consecutive windows
3. **Label Strategy**: How to assign a label to a window (e.g., majority vote)
4. **Minimum Samples Per Class**: Minimum number of samples required for each class

## Implementation

The windowing functionality is implemented in:

1. `thesis/data/windowing.py` - Core windowing functionality
2. `thesis/data/__init__.py` - Integration with data loading

## Usage Example

```python
from thesis.data import WindowConfig, create_sliding_windows

# Create a window configuration
window_config = WindowConfig(
    window_size=64,
    overlap_ratio=0.5,
    label_strategy="majority_vote",
    min_samples_per_class=10
)

# Create windows from data and labels
windowed_data = create_sliding_windows(
    data=sensor_data,
    labels=activity_labels,
    config=window_config
)
``` 
# Code Documentation

This directory contains documentation related to the code structure, organization, and cleanup efforts for the thesis project.

## Documentation Files

1. [Code Cleanup Summary](CODE_CLEANUP_SUMMARY.md) - Summary of code cleanup and refactoring efforts

## Code Structure

The thesis codebase is organized as follows:

```
thesis/
├── data/                # Data loading and preprocessing
│   ├── __init__.py
│   ├── datasets.py      # Dataset loading functions
│   └── windowing.py     # Time series windowing
├── exp/                 # Experiments
│   ├── activity_classification.py  # Core classification functionality
│   ├── compare_label_types.py      # Label type comparison
│   ├── per_sensor_quick_test.py    # Quick test for per-sensor approach
│   ├── per_sensor_test.py          # Full test for per-sensor approach
│   ├── rq2_confusion_matrices.py   # Confusion matrix generation
│   ├── rq2_experiment.py           # Traditional approach experiment
│   ├── rq2_experiment_locomotion.py # Locomotion-focused experiment
│   ├── rq2_mini_test.py            # Simplified test
│   ├── rq2_per_sensor_experiment.py # Per-sensor approach experiment
│   ├── rq2_visualize_results.py    # Results visualization
│   └── visualize_metric_comparison.py # Metric comparison visualization
└── fuzzy/               # Fuzzy logic and similarity metrics
    ├── __init__.py
    ├── per_sensor_membership.py    # Per-sensor membership functions
    ├── similarity.py               # Core similarity metrics
    └── similarity_subset.py        # Subset of metrics for quick testing
```

## Best Practices

The codebase follows these best practices:

1. **Modular Design**: Code is organized into logical modules
2. **Type Annotations**: Functions and classes use type hints
3. **Documentation**: Functions and modules have docstrings
4. **Error Handling**: Robust error handling and informative error messages
5. **Logging**: Consistent logging throughout the codebase
6. **Testing**: Test scripts for validating functionality
7. **Configuration**: Command-line arguments for experiment configuration

## Development Workflow

The development workflow includes:

1. **Feature Implementation**: New features are implemented in dedicated modules
2. **Testing**: Quick tests are created to validate functionality
3. **Experimentation**: Full experiments are conducted to evaluate performance
4. **Visualization**: Results are visualized for analysis
5. **Documentation**: Code and results are documented 
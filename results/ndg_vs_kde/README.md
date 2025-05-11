# Experiment 1: NDG vs KDE Results

This directory contains the results from comparing the Neighbor Density Graph (NDG) method
with Kernel Density Estimation (KDE) for computing membership functions.

## Files

- `length_experiment_results.csv`: Performance comparison with varying dataset sizes
- `sigma_experiment_results.csv`: Impact of different sigma (bandwidth) values
- `signal_type_results.csv`: Comparison across different signal types
- `real_dataset_results.csv`: Results on real-world datasets
- `single_experiment.csv`: Detailed results from a single experiment run

## Plots

- `time_comparison.png`: Execution time comparison between NDG and KDE
- `memory_comparison.png`: Memory usage comparison between NDG and KDE
- `error_comparison.png`: Accuracy comparison using KL-divergence and Chi-squared metrics
- `sigma_comparison.png`: Impact of sigma parameter on results

## Summary Statistics

The `summary_statistics.json` file contains aggregated metrics across all experiments.

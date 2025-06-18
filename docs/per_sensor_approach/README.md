# Per-Sensor Membership Function Approach Documentation

This directory contains documentation for the per-sensor membership function approach, a novel method for fuzzy similarity calculation in sensor data analysis.

## Overview

The per-sensor membership function approach generates one membership function per sensor, rather than combining all sensors into a single membership function. This allows for more granular similarity calculations that can account for sensor-specific characteristics.

## Documentation Files

1. [Per-Sensor Membership Approach](PER_SENSOR_MEMBERSHIP_APPROACH.md) - Detailed explanation of the approach, motivation, and test results
2. [Implementation Summary](PER_SENSOR_IMPLEMENTATION_SUMMARY.md) - Summary of implementation accomplishments, key components, and future work

## Visualizations

The `images` directory contains visualizations comparing different similarity metrics:

1. [Performance Comparison](images/metric_performance_comparison.png) - Comparison of accuracy, balanced accuracy, and macro F1 scores
2. [Computation Time](images/metric_computation_time.png) - Comparison of computation time for different metrics
3. [Radar Comparison](images/metric_radar_comparison.png) - Radar chart comparing all performance metrics
4. [Combined Comparison](images/metric_combined_comparison.png) - Combined bar chart of all metrics

## Key Findings

- The per-sensor approach consistently outperforms the traditional single-membership approach
- Jaccard and Dice similarity metrics perform identically and excellently
- Cosine similarity performs poorly for this specific task
- The approach achieves near-perfect classification in several test scenarios

## Implementation Files

The implementation can be found in the following files:

1. `thesis/fuzzy/per_sensor_membership.py` - Core implementation
2. `thesis/exp/per_sensor_quick_test.py` - Quick test script
3. `thesis/exp/per_sensor_test.py` - Comprehensive test script
4. `thesis/exp/rq2_per_sensor_experiment.py` - RQ2 experiment integration
5. `thesis/exp/visualize_metric_comparison.py` - Visualization script 
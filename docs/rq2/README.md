# Research Question 2 (RQ2) Documentation

This directory contains documentation for Research Question 2 (RQ2) of the thesis project, which focuses on activity classification using fuzzy similarity metrics.

## Documentation Files

1. [Dataset Diagram](RQ2_DATASET_DIAGRAM.md) - Visual diagram of the RQ2 dataset structure
2. [Dataset Structure](RQ2_DATASET_STRUCTURE.txt) - Detailed description of the RQ2 dataset
3. [Detailed Plan](RQ2_DETAILED_PLAN.md) - Comprehensive plan for RQ2 experiments
4. [Next Steps](RQ2_NEXT_STEPS.md) - Upcoming tasks and next steps for RQ2 experiments
5. [Technical Specifications](RQ2_TECHNICAL_SPECIFICATIONS.md) - Technical details and specifications

## RQ2 Overview

Research Question 2 investigates how different fuzzy similarity metrics perform in activity classification tasks using sensor data. The experiments use the Opportunity dataset, which contains recordings from wearable sensors during various activities.

## Key Components

1. **Dataset**: Opportunity dataset with Locomotion and High-Level Activity labels
2. **Preprocessing**: Sliding window approach with different window sizes and overlap ratios
3. **Feature Extraction**: NDG (Normalized Data Gaussian) membership function generation
4. **Similarity Calculation**: Various similarity metrics, including traditional and per-sensor approaches
5. **Classification**: K-nearest neighbors and leave-one-out cross-validation
6. **Evaluation**: Accuracy, balanced accuracy, and macro F1 score

## Implementation Files

The implementation of the RQ2 experiments can be found in:

1. `thesis/exp/activity_classification.py` - Core classification functionality
2. `thesis/exp/rq2_experiment.py` - Traditional approach experiment
3. `thesis/exp/rq2_per_sensor_experiment.py` - Per-sensor approach experiment
4. `thesis/exp/rq2_mini_test.py` - Simplified test for quick evaluation
5. `thesis/exp/rq2_confusion_matrices.py` - Confusion matrix generation
6. `thesis/exp/rq2_visualize_results.py` - Results visualization 
# Similarity Metrics Documentation

This directory contains documentation for the similarity metrics implemented in the thesis project.

## Documentation Files

1. [Similarity Metrics Overview](SIMILARITY_METRICS.md) - Comprehensive overview of all implemented similarity metrics
2. [New Metrics Summary](NEW_METRICS_SUMMARY.md) - Summary of newly implemented similarity metrics

## Implemented Metrics

The thesis implements and compares a variety of similarity metrics for sensor data, including:

### Set-Based Metrics
- Jaccard Similarity
- Dice Coefficient
- Overlap Coefficient
- Cosine Similarity

### Statistical Metrics
- Pearson Correlation
- Spearman Correlation
- Kendall's Tau

### Distance-Based Metrics
- Euclidean Distance
- Manhattan Distance
- Chebyshev Distance
- Minkowski Distance

### Fuzzy Metrics
- Fuzzy Jaccard
- Fuzzy Dice
- Fuzzy Hamming Distance

### Per-Sensor Metrics
- Per-Sensor Jaccard
- Per-Sensor Dice
- Per-Sensor Cosine

## Implementation Files

The implementation of these metrics can be found in:

1. `thesis/fuzzy/similarity.py` - Core similarity metrics
2. `thesis/fuzzy/per_sensor_membership.py` - Per-sensor similarity metrics 
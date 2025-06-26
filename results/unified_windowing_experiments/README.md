# Unified Windowing Experiments

This directory contains all experiments related to the unified windowing optimization approach.

## Directory Structure

```
unified_windowing_experiments/
├── logs/           # Experiment execution logs
├── data/           # Experiment result data (pickled objects)
├── reports/        # Analysis reports and summaries
└── README.md       # This file
```

## Experiments Overview

### 1. Basic Unified Windowing (Completed ✅)
- **Date**: 2025-06-26
- **Metrics**: 5 basic metrics (jaccard, cosine, dice, pearson, overlap_coefficient)
- **Status**: Completed successfully
- **Key Results**:
  - Locomotion: Pearson best (57.4% Hit@1, 70.9% MRR)
  - ML_Both_Arms: Cosine/Pearson tied (36.1% Hit@1, 48.0% MRR)
  - HL_Activity: Dice/Overlap best (59.3% Hit@1, 68.8% MRR)

### 2. Expanded Metrics Unified Windowing (In Progress 🔄)
- **Date**: 2025-06-26 (started 03:52)
- **Metrics**: 16 comprehensive metrics
- **Status**: Running (Locomotion phase, 5/16 metrics completed)
- **Progress**: 
  - ✅ jaccard, dice, overlap_coefficient, cosine, pearson
  - 🔄 JensenShannon, BhattacharyyaCoefficient, HellingerDistance, Similarity_Euclidean
  - ⏳ 7 more advanced metrics pending

## Key Achievements

1. **~200x Performance Optimization**: Unified windowing with cached membership functions
2. **Query×Library Efficiency**: Reduced from O(n²) to O(query×library) computation
3. **Multi-label Scalability**: Single membership computation reused across label types
4. **Comprehensive Metric Evaluation**: Testing 16 similarity metrics for optimal performance

## Files in this Directory

- `basic_experiment_results.pkl` - Results from 5-metric experiment
- `expanded_experiment_results.pkl` - Results from 16-metric experiment (when complete)
- `experiment_summary_report.md` - Comprehensive analysis and insights

## Next Steps

1. Complete expanded metrics experiment
2. Analyze comparative performance of all 16 metrics
3. Identify optimal metric combinations for different label types
4. Generate thesis-ready visualizations and tables
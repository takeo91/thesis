# Dataset Coverage Analysis Report for RQ2

**Date**: 2025-01-20  
**Analysis**: Coverage of Opportunity and PAMAP2 datasets for RQ2 activity classification experiments

## Executive Summary

The RQ2 experiments use a targeted subset of activities from both datasets:
- **Opportunity Dataset**: Using only 4.2% of the data (2,142 out of 51,116 samples)
- **PAMAP2 Dataset**: Using 38.0% of the data (143,079 out of 376,417 samples)

This selective approach focuses on specific activities relevant to the research questions while significantly reducing computational requirements.

## Opportunity Dataset Analysis

### Label Hierarchy Overview

The Opportunity dataset contains multiple label hierarchies:

1. **Locomotion Labels** (5 classes)
   - Stand: 43.8%
   - Unknown: 26.6%
   - Sit: 14.6%
   - Walk: 12.8%
   - Lie: 2.2%

2. **High-Level Activity Labels** (6 classes)
   - Unknown: 30.5%
   - Early morning: 24.7%
   - Sandwich time: 20.0%
   - Coffee time: 11.2%
   - Cleanup: 10.0%
   - Relaxing: 3.6%

3. **ML_Both_Arms Labels** (18 classes)
   - Unknown: 90.5%
   - Drink from Cup: 2.3%
   - Various door/drawer/fridge operations: ~7%

### RQ2 Target Activities (ML_Both_Arms)

For RQ2, we focus on 7 specific activities from the ML_Both_Arms label set:

| Activity | Samples | Percentage |
|----------|---------|------------|
| Open Door 1 | 452 | 0.9% |
| Close Door 2 | 405 | 0.8% |
| Open Fridge | 384 | 0.8% |
| Open Door 2 | 329 | 0.6% |
| Close Door 1 | 271 | 0.5% |
| Close Fridge | 244 | 0.5% |
| Toggle Switch | 57 | 0.1% |
| **Total** | **2,142** | **4.2%** |

### Impact of Filtering

- **Data Reduction**: 95.8% of the Opportunity dataset is excluded
- **Focus**: Concentrates on fine-grained manipulation activities
- **Class Balance**: Reasonable distribution across target activities (57-452 samples per class)

## PAMAP2 Dataset Analysis

### Activity Distribution

The PAMAP2 dataset contains 13 activity classes:

| Activity | Samples | Percentage |
|----------|---------|------------|
| other | 126,460 | 33.6% |
| lying | 27,187 | 7.2% |
| cycling | 23,575 | 6.3% |
| ironing | 23,573 | 6.3% |
| sitting | 23,480 | 6.2% |
| vacuum_cleaning | 22,941 | 6.1% |
| walking | 22,253 | 5.9% |
| standing | 21,717 | 5.8% |
| running | 21,265 | 5.6% |
| Nordic_walking | 20,265 | 5.4% |
| ascending_stairs | 15,890 | 4.2% |
| descending_stairs | 14,899 | 4.0% |
| rope_jumping | 12,912 | 3.4% |

### RQ2 Target Activities

For RQ2, we focus on 7 basic physical activities:

| Activity | Samples | Percentage |
|----------|---------|------------|
| walking | 22,253 | 5.9% |
| running | 21,265 | 5.6% |
| cycling | 23,575 | 6.3% |
| sitting | 23,480 | 6.2% |
| standing | 21,717 | 5.8% |
| ascending_stairs | 15,890 | 4.2% |
| descending_stairs | 14,899 | 4.0% |
| **Total** | **143,079** | **38.0%** |

### Impact of Filtering

- **Data Reduction**: 62.0% of the PAMAP2 dataset is excluded
- **Focus**: Basic locomotion and posture activities
- **Class Balance**: Well-balanced across activities (14,899-23,575 samples per class)

## Comparative Analysis

### Coverage Comparison

| Dataset | Total Samples | Used Samples | Coverage | Activities Used |
|---------|---------------|--------------|----------|-----------------|
| Opportunity | 51,116 | 2,142 | 4.2% | 7 (manipulation) |
| PAMAP2 | 376,417 | 143,079 | 38.0% | 7 (locomotion) |

### Key Differences

1. **Coverage Ratio**: PAMAP2 uses 9x more data proportionally than Opportunity
2. **Activity Types**: 
   - Opportunity: Fine-grained object manipulation
   - PAMAP2: Gross motor activities
3. **Sample Distribution**:
   - Opportunity: Imbalanced (57-452 samples per activity)
   - PAMAP2: Well-balanced (14,899-23,575 samples per activity)

## Implications for RQ2 Analysis

### Advantages of Selective Filtering

1. **Computational Efficiency**: 
   - Reduces Opportunity data by 95.8%
   - Reduces PAMAP2 data by 62.0%
   - Enables faster experimentation

2. **Focused Analysis**:
   - Targets specific activity types relevant to research questions
   - Avoids noise from "Unknown" labels (90.5% in Opportunity)
   - Concentrates on well-defined activities

3. **Statistical Power**:
   - Sufficient samples per class for robust classification
   - Balanced representation across target activities
   - Adequate data for cross-validation

### Potential Limitations

1. **Generalizability**: Results may not extend to excluded activities
2. **Dataset Bias**: Different coverage ratios between datasets
3. **Activity Complexity**: Opportunity activities are more fine-grained than PAMAP2

## Recommendations

1. **For Current RQ2 Analysis**: The filtered datasets provide sufficient data for meaningful discriminative power assessment while maintaining computational efficiency.

2. **For Future Work**: Consider:
   - Including additional activity types for broader coverage
   - Analyzing the impact of "Unknown" labels on classification
   - Exploring hierarchical classification using multiple label types

3. **For Publication**: Clearly document the filtering criteria and rationale for activity selection to ensure reproducibility.

## Visualizations

See `dataset_coverage_analysis.png` for detailed visual representations of:
- Activity distributions in both datasets
- Coverage pie charts showing used vs. unused data
- Comparative bar charts highlighting target activities
- Summary statistics panel

---

*This analysis demonstrates that while we use a small fraction of the Opportunity dataset (4.2%), we capture meaningful manipulation activities. For PAMAP2, we use a more substantial portion (38.0%) focusing on fundamental physical activities. This selective approach balances computational efficiency with statistical validity for the RQ2 discriminative power assessment.* 
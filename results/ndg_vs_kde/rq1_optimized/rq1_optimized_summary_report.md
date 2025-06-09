
# RQ1 Optimized Experiment Results Summary

**Date**: 2025-06-09 20:43:35

## Optimization Details
- **NDG Implementation**: Optimized with spatial pruning + JIT compilation + parallelization
- **Kernel Type**: Epanechnikov (compact support)
- **Expected Speedup**: 10-100x over original NDG
- **Actual Average Speedup**: 12.48x

## Experiment Configuration
- **Datasets**: ['synthetic_normal', 'synthetic_bimodal', 'opportunity', 'pamap2']
- **Signal lengths**: [100, 1000, 10000, 100000] samples
- **Sigma values**: [0.1, 0.3, 0.5, 'r0.1', 'r0.3']
- **K-folds**: 5
- **Total experiments**: 400

## H1 Hypothesis Validation Results
**H1**: NDG-S is computationally more efficient than KDE

### Key Results:
- **Average speedup factor**: 12.48x
- **Median speedup factor**: 13.64x
- **Significant improvements**: 16/16 (100.0%)

### Statistical Test Results
- Total comparisons: 16
- Significant results (p < 0.05): 16

**Detailed Findings**:

- opportunity (n=100): NDG is 3.73x faster than KDE (p=0.0000, effect size=1.109)
- opportunity (n=1000): NDG is 11.12x faster than KDE (p=0.0000, effect size=1.109)
- opportunity (n=10000): NDG is 19.19x faster than KDE (p=0.0000, effect size=1.109)
- opportunity (n=100000): NDG is 19.47x faster than KDE (p=0.0000, effect size=1.109)
- pamap2 (n=100): NDG is 4.30x faster than KDE (p=0.0000, effect size=0.904)
- pamap2 (n=1000): NDG is 11.86x faster than KDE (p=0.0000, effect size=1.109)
- pamap2 (n=10000): NDG is 19.53x faster than KDE (p=0.0000, effect size=1.109)
- pamap2 (n=100000): NDG is 20.48x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_bimodal (n=100): NDG is 3.90x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_bimodal (n=1000): NDG is 9.05x faster than KDE (p=0.0006, effect size=0.688)
- synthetic_bimodal (n=10000): NDG is 15.79x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_bimodal (n=100000): NDG is 15.41x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_normal (n=100): NDG is 3.57x faster than KDE (p=0.0001, effect size=0.774)
- synthetic_normal (n=1000): NDG is 9.48x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_normal (n=10000): NDG is 15.47x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_normal (n=100000): NDG is 17.33x faster than KDE (p=0.0000, effect size=1.109)

### Final H1 Status: VALIDATED (Strong evidence)

# RQ1 Optimized Experiment Results Summary

**Date**: 2025-06-22 02:40:31

## Optimization Details
- **NDG Implementation**: Optimized with spatial pruning + JIT compilation + parallelization
- **Kernel Type**: Epanechnikov (compact support)
- **Expected Speedup**: 10-100x over original NDG
- **Actual Average Speedup**: 13.04x

## Experiment Configuration
- **Datasets**: ['synthetic_normal', 'synthetic_bimodal', 'opportunity', 'pamap2']
- **Signal lengths**: [100, 1000, 10000, 100000] samples
- **Sigma values**: [0.1, 0.3, 0.5, 'r0.1', 'r0.3']
- **K-folds**: 5
- **Total experiments**: 400

## H1 Hypothesis Validation Results
**H1**: NDG-S is computationally more efficient than KDE

### Key Results:
- **Average speedup factor**: 13.04x
- **Median speedup factor**: 13.17x
- **Significant improvements**: 16/16 (100.0%)

### Statistical Test Results
- Total comparisons: 16
- Significant results (p < 0.05): 16

**Detailed Findings**:

- opportunity (n=100): NDG is 4.20x faster than KDE (p=0.0000, effect size=1.109)
- opportunity (n=1000): NDG is 10.74x faster than KDE (p=0.0000, effect size=1.109)
- opportunity (n=10000): NDG is 18.95x faster than KDE (p=0.0000, effect size=1.109)
- opportunity (n=100000): NDG is 21.78x faster than KDE (p=0.0000, effect size=1.109)
- pamap2 (n=100): NDG is 4.58x faster than KDE (p=0.0000, effect size=0.904)
- pamap2 (n=1000): NDG is 11.22x faster than KDE (p=0.0000, effect size=1.109)
- pamap2 (n=10000): NDG is 19.95x faster than KDE (p=0.0000, effect size=1.109)
- pamap2 (n=100000): NDG is 23.61x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_bimodal (n=100): NDG is 3.91x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_bimodal (n=1000): NDG is 9.35x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_bimodal (n=10000): NDG is 15.63x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_bimodal (n=100000): NDG is 16.97x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_normal (n=100): NDG is 4.28x faster than KDE (p=0.0001, effect size=0.774)
- synthetic_normal (n=1000): NDG is 9.51x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_normal (n=10000): NDG is 15.12x faster than KDE (p=0.0000, effect size=1.109)
- synthetic_normal (n=100000): NDG is 18.81x faster than KDE (p=0.0000, effect size=1.109)

### Final H1 Status: VALIDATED (Strong evidence)
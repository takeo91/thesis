# RQ2 Technical Specifications Summary

## Key Implementation Details

| Component | Specification | Value/Method |
|-----------|---------------|--------------|
| **Similarity Metrics** | Total count | 29+ metrics |
| | New implementations | Jensen-Shannon, β-similarity (4 variants) |
| | Existing metrics | Jaccard, Dice, Cosine, Pearson, Hamming, etc. |
| **NDG Optimization** | Kernel type | Epanechnikov (10-100x faster) |
| | Sigma calculation | Adaptive: 0.1 × data_range |
| | Spatial pruning | 4-sigma cutoff |
| | Parallelization | Multi-core processing |
| **Datasets** | Primary | Opportunity, PAMAP2 |
| | Activities per dataset | 7 (minimum 100 windows each) |
| | Preprocessing | Low-pass filter, resampling |
| **Windowing** | Window sizes | 128, 256 samples |
| | Overlap ratios | 50%, 70% |
| | Label assignment | Majority vote per window |
| **Classification** | Algorithm | 1-NN with similarity matrices |
| | Cross-validation | Leave-One-Window-Out (LOWO) |
| | Primary metrics | Macro-F1, Balanced Accuracy |
| **Statistical Analysis** | Overall comparison | Friedman test |
| | Post-hoc analysis | Nemenyi test |
| | Effect sizes | Cohen's d, Cliff's delta |
| **Experimental Scale** | Total experiments | ~232 configurations |
| | Computation time | Reduced by 10-100x with optimization |
| | Expected runtime | 2-4 hours (vs 20-40 hours unoptimized) |

## Expected Performance Targets

| Metric | Target Value | Significance |
|--------|--------------|--------------|
| **Best Macro-F1** | > 0.70 | High discriminative power |
| **Statistical Significance** | p < 0.05 | Robust metric differences |
| **Cross-dataset Consistency** | Spearman ρ > 0.6 | Generalizable rankings |
| **Computation Speedup** | 10-100x | Practical feasibility |
| **Memory Efficiency** | < 16GB RAM | Standard hardware compatibility |

## File Structure and Outputs

```
results/rq2_classification/
├── rq2_experimental_results.csv          # Raw experimental data
├── rq2_statistical_analysis.csv          # Statistical test results  
├── rq2_metric_rankings.csv              # Ranked similarity metrics
├── rq2_summary_report.md                # Comprehensive analysis
├── rq2_performance_heatmap.png          # Metric performance visualization
├── rq2_statistical_significance.png     # Statistical comparison plots
├── rq2_activity_performance.png         # Per-activity results
└── rq2_confusion_matrices.png           # Classification confusion matrices
```

## Code Implementation Files

```
thesis/
├── fuzzy/
│   └── similarity.py                    # +Jensen-Shannon, +β-similarity
├── exp/
│   ├── time_series_windowing.py        # NEW: Windowing infrastructure
│   ├── membership_computation.py       # NEW: Optimized NDG wrapper
│   ├── similarity_matrix.py            # NEW: Pairwise similarity computation
│   ├── nearest_neighbor_classifier.py  # NEW: 1-NN with similarity matrices
│   ├── rq2_experiment.py              # NEW: Main experiment controller
│   ├── rq2_statistical_analysis.py    # NEW: Statistical testing
│   └── rq2_visualizations.py          # NEW: Plotting and visualization
└── notebooks/
    └── rq2_classification_analysis.ipynb # Analysis notebook
```

## Risk Assessment and Mitigation

| Risk Category | Risk | Mitigation Strategy |
|---------------|------|-------------------|
| **Computational** | Memory overflow | Chunked processing, spatial pruning |
| | Excessive runtime | Optimized NDG (10-100x speedup) |
| **Statistical** | Multiple comparisons | Bonferroni correction, FDR control |
| | Class imbalance | Balanced accuracy, stratified sampling |
| **Data Quality** | Insufficient samples | Minimum 100 windows per activity |
| | Noise sensitivity | Robust preprocessing, outlier detection |
| **Implementation** | Integration complexity | Incremental testing, modular design |
| | Reproducibility | Fixed random seeds, version control |

This technical specification provides the foundation for implementing RQ2 with clear targets, measurable outcomes, and robust risk mitigation strategies. 
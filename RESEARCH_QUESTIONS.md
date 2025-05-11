# Research Questions

## RQ1 — Membership estimation efficiency

**How does the proposed streaming Normalised Difference Gaussian (NDG-S) algorithm compare to standard Gaussian KDE in estimating membership functions for large multi-sensor time-series?**

**Hypothesis (H1):** NDG-S yields KL-divergence ≤ 5% higher than KDE (i.e., statistically indistinguishable accuracy) while using ≥ 70% less peak RAM and ≥ 2× faster wall-clock time.

**Methodology:**
- Run NDG-S vs sklearn.KernelDensity on 2 datasets (Opportunity, PAMAP2)
- Measure KL or χ² between estimate and held-out distribution; log runtime + psutil max-RSS
- Wilcoxon signed-rank on per-fold pairs

## RQ2 — Discriminative power of similarity metrics

**Which fuzzy-set similarity metrics most strongly separate activity classes when membership curves are built with NDG-S?**

**Hypothesis (H2):** Information-theoretic metrics (e.g., Jensen-Shannon, β-similarity) achieve ≥ 10 pp higher macro-F1 in a 1-NN classifier than overlap-based metrics (Jaccard, Dice).

**Methodology:**
- For each dataset, build membership curves ➜ pairwise similarity matrix
- 1-nearest-neighbour leave-one-window-out classification
- Compare macro-F1 / balanced accuracy across 10 metrics; Friedman + Nemenyi post-hoc

## RQ3 — Cross-dataset robustness

**Are the metric rankings from RQ2 consistent across datasets with different sensor modalities and sampling rates?**

**Hypothesis (H3):** The top-3 metrics in RQ2 maintain Spearman ρ ≥ 0.7 correlation in rank order across all datasets.

**Methodology:**
- Produce per-dataset rank lists of metric performance
- Compute pairwise Spearman correlations of ranks; bootstrap CI


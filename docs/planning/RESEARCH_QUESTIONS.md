# Research Questions

## RQ1 — Membership estimation efficiency

**How does the proposed streaming Normalised Difference Gaussian (NDG-S) algorithm compare to standard Gaussian KDE in estimating membership functions for large multi-sensor time-series?**

**Hypothesis (H1):** NDG-S yields KL-divergence ≤ 5% higher than KDE (i.e., statistically indistinguishable accuracy) while using ≥ 70% less peak RAM and ≥ 2× faster wall-clock time.

**Methodology:**
- Run NDG-S vs sklearn.KernelDensity on 2 datasets (Opportunity, PAMAP2)
- Measure KL or χ² between estimate and held-out distribution; log runtime + psutil max-RSS
- Wilcoxon signed-rank on per-fold pairs

## RQ2 — Similarity-based retrieval of unseen windows

**Revised Question:** *Given an unseen window of multi-sensor data, how well can fuzzy-set similarity metrics retrieve the correct activity **and** sensor-type from a reference library of labelled windows?*

**Hypothesis (H2):** When the library is class-balanced and the query set is disjoint in time, per-sensor overlap-based metrics (Jaccard, Dice, Overlap-Coefficient) will achieve **≥ 80 % top-5 retrieval accuracy** for locomotion activities **and** ≥ 75 % for sensor-type identification, outperforming magnitude-only baselines (Euclidean, Pearson) by ≥ 10 pp.

**Experimental protocol:**
1. **Windowing** – use 120- and 180-sample windows with 0.5 / 0.7 overlap.  
2. **Balancing** – cap each activity class to *M* windows (e.g., *M* = 125).  
3. **Library / Query split** – randomly assign 80 % of balanced windows to the *library* and 20 % to the *query* set, stratified by class.  
4. **Similarity computation** – compute Query × Library matrices for 25 vectorisable metrics.  
5. **Retrieval evaluation** – report top-{1,5} accuracy and mAP for:
   • activity label match  
   • sensor-type / placement match  
   (A hit requires both to match.)
6. **Statistics** – Friedman test across metrics; Nemenyi post-hoc.

## RQ3 — Cross-dataset robustness of retrieval metrics

**Revised Question:** *Do the retrieval-accuracy rankings of similarity metrics remain stable across datasets with different sampling rates and sensor modalities (Opportunity vs PAMAP2)?*

**Hypothesis (H3):** The top-3 metrics from RQ2 will maintain **Spearman rank correlation ρ ≥ 0.7** for both activity and sensor-type retrieval accuracy across datasets.

**Methodology:**  
• Repeat the RQ2 experiment on PAMAP2.  
• Produce per-dataset metric rankings (based on top-1 activity accuracy).  
• Compute pairwise Spearman ρ; bootstrap 95 % CI.


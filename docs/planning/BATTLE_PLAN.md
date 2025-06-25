# 📊 Thesis Battle Plan (v1)

## Overview of Research Questions

**RQ1 — Membership Estimation Efficiency**  
How does the proposed streaming Normalised Difference Gaussian (NDG-S) algorithm compare to Gaussian KDE for large multi-sensor time-series?

*Hypothesis H1:* NDG-S achieves KL-divergence ≤ 5 % worse than KDE while requiring ≥ 70 % less RAM and ≥ 2 × faster wall-clock time.

**Status:** ✅ **FINISHED (June 2025)** — All experiments, analysis, and documentation for RQ1 are complete. See `results/ndg_vs_kde/rq1_markdown_results/` for final results and methods.

---

**RQ2 — Similarity-Based Retrieval of Unseen Windows**  
Given an unseen window, how well can fuzzy-set similarity metrics retrieve the correct activity *and* sensor-type from a reference library?

*Hypothesis H2:* Overlap-based metrics (Jaccard, Dice, Overlap-Coeff.) reach ≥ 80 % top-5 retrieval accuracy for locomotion and ≥ 75 % for sensor-type, outperforming magnitude-based baselines by ≥ 10 pp.

---

**RQ3 — Cross-Dataset Robustness of Retrieval Metrics**  
Do metric-accuracy rankings remain stable across datasets with different rates/modalities (Opportunity vs PAMAP2)?

*Hypothesis H3:* Top-3 metrics from RQ2 maintain Spearman ρ ≥ 0.7 across datasets.

---

## What still needs to be tested?

| RQ | Gaps & TODOs | Candidate Metrics / Analyses |
|----|--------------|------------------------------|
| RQ1 | ✅ **FINISHED** (see markdown results folder) | KL-div, χ², runtime, max-RSS |
| RQ2 | • Add per-sensor *Dice* & *Jaccard* to retrieval script  <br/>• Evaluate *Overlap-Coeff.*  <br/>• Integrate *Cosine* baseline  <br/>• Sensor-placement retrieval metric | Hit@1, Hit@5, MRR, mAP |
| RQ3 | • Repeat retrieval on PAMAP2  <br/>• Compute Spearman ρ between datasets  <br/>• Friedman + Nemenyi across metrics | Spearman ρ, p-values, critical diff. |

---

## Immediate Next Steps (Today)

1. **Refocus checklist** – copy this plan into Google Doc for quick annotations.
2. **Pilot grid driver** – implement `thesis/exp/pilot_driver.py` looping over:
   * Window sizes: {4 s, 6 s}
   * Overlaps: {0.5, 0.7}
   * Metrics: {jaccard, dice, cosine}
   * Sensor sets: {single-ankle, single-torso, all-IMUs}
3. Commit + push; run small "quick" mode to verify.

---

*All seeds fixed at 42 for reproducibility.* 
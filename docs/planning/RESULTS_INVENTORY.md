# Results Inventory for Thesis

This document enumerates **all deliverables** (tables, figures, CSV exports) required to answer the three research questions and to assemble the final thesis/paper package.

---
## Legend
* **Path** – where the artefact will be written (relative to repo root)
* **Source experiment** – Python script or notebook that generates the artefact
* **Purpose** – how the artefact is used in the thesis (e.g. main chapter figure, appendix table)

---
## RQ-1  Efficiency of NDG vs KDE

| # | Artefact | Path | Source experiment | Purpose |
|---|----------|------|-------------------|---------|
| 1 | Runtime table (mean ± std) | `results/ndg_vs_kde/runtime_table.csv`  / `runtime_table.tex` | `thesis/exp/ndg_vs_kde.py` | Chapter 4 table, LaTeX import |
| 2 | Line plot: *samples vs runtime* | `results/ndg_vs_kde/ndg_vs_kde_runtime.png` | same | Visual proof of scalability |
| 3 | Bar chart: NDG speed-up factor | `results/ndg_vs_kde/speedup_bar.png` | same | Highlight performance gain |
| 4 | Stats report (Wilcoxon, effect size) | `results/ndg_vs_kde/stats_report.md` | same | Methods appendix |

---
## RQ-2  Discriminative Power (Retrieval Protocol)

### Experimental settings
* Datasets: Opportunity, PAMAP2  
* Window configs: 120×0.5 and 180×0.7  
* Library: **25 windows / class**, 3 random splits  
* Metrics: 25 vectorisable, per-sensor implementation.

| # | Artefact | Path | Source experiment | Purpose |
|---|----------|------|-------------------|---------|
| 1 | Per-metric retrieval scores (hit@1, hit@5, MRR ∓ std) | `results/rq2_classification/per_metric_scores.csv` | `thesis/exp/rq2_experiment.py` | Core quantitative result |
| 2 | Heat-map: metric × class (hit@1) | `results/rq2_classification/class_heatmap_hit1.png` | same | Visual insight per activity |
| 3 | Ranking bar chart (mean hit@1) | `results/rq2_classification/ranking_bar_hit1.png` | same | Top-metric overview |
| 4 | Critical-difference diagram | `results/rq2_classification/friedman_cd_hit1.png` | `thesis/exp/rq2_statistical_analysis.py` | Significance testing |
| 5 | Similarity computation runtime | `results/rq2_classification/similarity_runtime.csv` | `rq2_experiment.py` | Methods & scalability |
| 6 | Example confusion-style retrieval map (best metric) | `results/rq2_classification/example_conf_matrix_jaccard.png` | optional notebook | Qualitative explanation |

---
## RQ-3  Cross-Dataset Robustness

| # | Artefact | Path | Source experiment | Purpose |
|---|----------|------|-------------------|---------|
| 1 | Spearman ρ per metric (Opp ↔ PAMAP2) | `results/rq3_robustness/spearman_table.csv` | `thesis/exp/cross_dataset_robustness.py` | Quantifies robustness |
| 2 | Scatter: rank_Opp vs rank_PAMAP2 | `results/rq3_robustness/rank_scatter.png` | same | Visual correlation |
| 3 | Violin plot: rank variance | `results/rq3_robustness/rank_variance_violin.png` | same | Shows stability per metric |
| 4 | Bootstrap confidence intervals for ρ | `results/rq3_robustness/bootstrap_ci.json` | same | Statistical rigor |

---
## Shared / Supporting Figures

| Artefact | Path | Source | Purpose |
|----------|------|--------|---------|
| NDG vs KDE membership example | `results/ndg_vs_kde/ndg_membership_example.png` | small notebook | Illustrative figure (Chapter 3) |
| Retrieval workflow diagram (Mermaid) | `docs/figures/retrieval_workflow.svg` | `docs/figures/` | Method overview |
| Experiment-pipeline architecture | `docs/figures/pipeline_architecture.svg` | draw.io | Thesis methods section |

---
## Generation checklist

```
make rq1               # populates results/ndg_vs_kde/
make rq2 library_per_class=25 topk="1 5"  # populates results/rq2_classification/
make rq3               # populates results/rq3_robustness/
``` 
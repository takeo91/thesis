# RQ2 Retrieval-Style Evaluation Plan

> **Objective**: Complement the classification study (H2) with a *retrieval* perspective: given a **query** window, retrieve the most similar windows from a reference **library** and assess whether items of the same activity appear in the top of the ranked list.

The retrieval formulation provides additional evidence of discriminative power and is better aligned with real-time *search/recommendation* use-cases.

---
## 1  Pipeline Overview

```
Raw sensor data  ─►  Sliding windows  ─►  Balanced WindowedData
                                  │
                                  ├─►  train_test_split_windows  (library/query)
                                  │
                                  ├─►  NDG per-sensor membership functions (fast Epanechnikov)
                                  │
                                  ├─►  compute_per_sensor_pairwise_similarities  (Q × L)
                                  │
                                  └─►  retrieval_utils.compute_retrieval_metrics
```

Key building blocks were just implemented:
* `train_test_split_windows` – deterministic library/query split.
* `compute_per_sensor_pairwise_similarities` – now supports asymmetric Q × L matrices.
* `retrieval_utils` – Hit@k and MRR metrics.

---
## 2  Experimental Factors

| Factor | Levels | Rationale |
|--------|--------|-----------|
| Dataset | Opportunity, PAMAP2 | Same as classification study |
| Label sets | Locomotion (Opp.), Core-7 (PAMAP2) | ensure comparability |
| Window size | 120, 180 | aligns with classification |
| Overlap | 0.5, 0.7 | mirrors previous design |
| Library per class | **1, 2, 5** | explore effect of reference size |
| Similarity metric | 5 core + (optionally) all 38 | manage runtime |
| Top-k | 1, 3, 5 | standard retrieval cut-offs |

Full factorial (with core metrics) ≈ 2 (datasets) × 2 (ws) × 2 (overlap) × 3 (lib size) × 5 (metrics) × 3 (k) = **360 configs**.  Manageable with current speedups.

---
## 3  Implementation Tasks

1. **Experiment driver** (`thesis/exp/rq2_retrieval_experiment.py`)
   • Parses CLI: `--datasets`, `--window_sizes`, `--overlaps`, `--library_per_class`, `--metrics`, `--topk`, `--n_jobs`, etc.  
   • Iterates over factor grid, persists intermediate results (reuse checkpoint logic from `UnifiedRQ2Experiment`).

2. **Metrics computation**  
   • Use asymmetric similarity function with `n_jobs=-1` for production, `n_jobs=1` in notebooks.

3. **Result aggregation**  
   • Collect per-config metrics → `DataFrame` with columns `[dataset, window_size, overlap, lib_per_class, metric, hit@1, hit@k, mrr]`.

4. **Statistical analysis** (`thesis/exp/rq2_retrieval_stats.py`)  
   • Friedman + Nemenyi on MRR across configs.  
   • Effect sizes vs. classification-based rankings.

5. **Visualisation** (`thesis/exp/rq2_retrieval_visuals.py`)  
   • Heatmaps of MRR / Hit@k.  
   • Radar plot comparing classification vs retrieval rankings.

6. **Documentation**  
   • Update `docs/rq2/` README and summary tables.  
   • Add retrieval figure panels to thesis chapter.

---
## 4  Computing Resources & Runtime Estimate

Approx. timings per config (core metrics, 1 library window/class, Opp.):
* NDG membership (Q=600, L=200): **~15 s** (parallel)
* Similarity matrix (5 metrics): **~8 s**
* Metric evaluation: negligible

Total ≈ 23 s × 360 ≈ **2.3 h** on 8-core laptop.  With PAMAP2 (larger), expect ~4 h.  Fine for overnight batch.

---
## 5  Milestones & Timeline (2 days)

| Day | Deliverable |
|-----|-------------|
| **D1 – Morning** | Implement `rq2_retrieval_experiment.py`; smoke-test on tiny subset |
| D1 – Afternoon | Full run on Opportunity (core metrics); verify outputs |
| **D2 – Morning** | Extend to PAMAP2; complete visualisations |
| D2 – Afternoon | Statistical analysis + documentation; push results to `results/rq2_retrieval/` |

---
## 6  Success Criteria

1. Pipeline runs to completion on both datasets within 6 h wall-time.
2. Retrieval metrics exported for all configs; no missing values.
3. At least one metric achieves Hit@1 > 0.6 and MRR > 0.7 on both datasets.
4. Findings integrated into thesis draft with clear comparison to classification results.

---
*Last updated: {{DATE}}* 
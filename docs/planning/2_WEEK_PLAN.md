# 1-Week Execution Plan (compressed from original 2-week schedule)

**Last Updated**: 2025-06-19  
**Goal**: Finalise RQ2 retrieval-style evaluation, cross-dataset RQ3, documentation, and publication artefacts **in 7 days**.

---
## Day-by-Day Breakdown

### Day 1 – Code freeze on metric pipeline
1. Implement retrieval-style library/query split helper (`train_test_split_windows`).
2. Extend `compute_per_sensor_pairwise_similarities` with asymmetric query-vs-library mode.
3. Add retrieval metrics (hit@1, hit@k, MRR) in `thesis.exp.retrieval_utils`.
4. CLI flags: `--library_per_class`, `--topk`.

### Day 2 – Run RQ2 experiments (Opportunity & PAMAP2)
1. Balanced window generation (`max_windows_per_class`).
2. Reference/query split with three random seeds.
3. Store raw similarity matrices and retrieval metrics in `results/rq2_classification/`.

### Day 3 – RQ2 statistical analysis + visualisation
1. Aggregate metrics across splits, compute mean ± std.
2. Friedman + Nemenyi on hit@1 and MRR.
3. Produce heatmaps and ranking plots.

### Day 4 – RQ3 cross-dataset robustness
1. Train library on Opportunity, query with PAMAP2 and vice-versa.
2. Compute Spearman rank correlations of metric performance.
3. Bootstrap CIs, store in `results/rq3_robustness/`.

### Day 5 – Global synthesis
1. Combine RQ1 efficiency, RQ2 discriminative power, RQ3 robustness.
2. Identify "best overall" metric-kernel combo.
3. Draft summary tables and key figures.

### Day 6 – Documentation & reproducibility pass
1. Update `docs/` with final methodology, config examples, and CLI instructions.
2. Remove residual dead code, ensure `pytest` passes on clean install.
3. Add `summarize_windowed_data()` helper usage in all experiments.

### Day 7 – Publication packaging
1. Export high-resolution figures (PNG + PDF, 300 DPI).
2. Generate LaTeX-ready tables (CSV → `tabularx`).
3. Final pass on README and `METHODOLOGY.md`.
4. Tag git release `v1.0-thesis`.

---
## Deliverables after 7 days
* Retrieval-based RQ2 implementation with balanced windows.
* Cross-dataset robustness results (RQ3).
* Full statistical analysis notebooks & scripts.
* Clean codebase with central config and standard helpers.
* Publication-ready figures, tables, and documentation.

---
**Risk buffer**: 0.5 day built into Days 5-6 for unforeseen issues. 
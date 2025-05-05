#!/usr/bin/env bash
# Run full RQ-1 grid (2 datasets × 3 folds × 3 σ)
set -euo pipefail

mkdir -p results
OUT=results/rq1_grid.csv
echo "dataset,fold,sigma,kl_div,chi2,wall_sec,peak_rss_mb" >"$OUT"

DATASETS=(opportunity pamap2)
FOLDS=(0 1 2)
SIGMAS=(0.2 0.4 0.8)

for ds in "${DATASETS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    for s in "${SIGMAS[@]}"; do
      printf "▶ %s fold=%d σ=%s\\n" "$ds" "$fold" "$s"
      metrics=$(python -m thesis.exp ndg_vs_kde \
                 --dataset "$ds" --fold "$fold" --sigma "$s" \
                 --print-values --quiet)
      echo "$ds,$fold,$s,$metrics" >>"$OUT"
    done
  done
done

echo "✅  Grid finished → $OUT"
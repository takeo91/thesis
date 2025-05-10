#!/usr/bin/env bash
# Run full RQ-1 grid (2 datasets × N folds × 3 σ)
set -euo pipefail

mkdir -p results
OUT=results/rq1_grid.csv
echo "dataset,fold,sigma,kl_div,chi2,wall_sec,peak_rss_mb" >"$OUT"

DATASETS=(opportunity pamap2)
NUM_FOLDS=3
SIGMAS=(0.2 0.4 0.8)

for ds in "${DATASETS[@]}"; do
  for (( fold=0; fold<NUM_FOLDS; fold++ )); do
    for s in "${SIGMAS[@]}"; do
      printf "▶ %s fold=%d/%d σ=%s\\n" "$ds" "$fold" "$NUM_FOLDS" "$s"
      metrics=$(python -m thesis.exp ndg_vs_kde \
                 --dataset "$ds" --fold "$fold" --num-folds "$NUM_FOLDS" \
                 --sigma "$s" \
                 --print-values --quiet)
      echo "$ds,$fold,$s,$metrics" >>"$OUT"
    done
  done
done

echo "✅  Grid finished → $OUT"
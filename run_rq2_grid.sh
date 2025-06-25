#!/bin/bash

# Automated RQ2 grid search for pilot_driver.py
# Sweeps library_per_class and topk, all other params as lists

# Set correct data path
export THESIS_DATA="/Users/nterlemes/personal/thesis_unified/thesis/Data"

LIBRARY_PER_CLASS_LIST=(5 10 20)
TOPK_LIST=(1 3 5)

for lib in "${LIBRARY_PER_CLASS_LIST[@]}"; do
  for k in "${TOPK_LIST[@]}"; do
    OUTDIR="results/pilot_grid/rq2_full/lib${lib}_topk${k}"
    mkdir -p "$OUTDIR"
    echo "Running: library_per_class=$lib, topk=$k"
    uv run python -m thesis.exp.pilot_driver \
      --datasets opportunity pamap2 \
      --window_durations 4 6 \
      --overlaps 0.5 0.7 \
      --library_per_class $lib \
      --max_query_per_class 50 \
      --metrics jaccard dice cosine overlap_coefficient \
      --sensor_sets ankle torso hand chest back all \
      --topk $k \
      --progress \
      --output_dir "$OUTDIR"
  done
done 
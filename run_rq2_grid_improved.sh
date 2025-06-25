#!/bin/bash

# Automated RQ2 grid search with Phase 3 improvements
# This version benefits from:
# - Refactored functions (better reliability)
# - Standardized error handling (graceful recovery)
# - Improved logging (better progress tracking)

# Set correct data path
export THESIS_DATA="/Users/nterlemes/personal/thesis_unified/thesis/Data"

echo "🚀 Starting RQ2 Grid Search with Phase 3 Improvements"
echo "📊 Benefits: Refactored functions, error handling, improved logging"
echo "=" * 60

# Start with a smaller, faster configuration to test improvements
LIBRARY_PER_CLASS_LIST=(5)      # Start with 5 for faster testing
TOPK_LIST=(1)                   # Start with topk=1 for faster testing

for lib in "${LIBRARY_PER_CLASS_LIST[@]}"; do
  for k in "${TOPK_LIST[@]}"; do
    OUTDIR="results/pilot_grid/rq2_phase3_improved/lib${lib}_topk${k}"
    mkdir -p "$OUTDIR"
    
    echo "📈 Running: library_per_class=$lib, topk=$k"
    echo "⏰ Started at: $(date)"
    
    # Run with all improvements enabled
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
      --output_dir "$OUTDIR" \
      2>&1 | tee "$OUTDIR/experiment_log.txt"
    
    echo "✅ Completed: library_per_class=$lib, topk=$k at $(date)"
    echo "📁 Results saved to: $OUTDIR"
    echo "-" * 40
  done
done

echo "🎉 All experiments completed!"
echo "📊 Results available in: results/pilot_grid/rq2_phase3_improved/"
#!/bin/bash

# Parallel RQ2 execution using multiple cores
# Distributes sensor sets across parallel processes

export THESIS_DATA="/Users/nterlemes/personal/thesis_unified/thesis/Data"

echo "🚀 Starting Parallel RQ2 Grid Search"
echo "💻 Using multiple cores for sensor-level parallelization"
echo "⏱️  Expected runtime: ~45 minutes (vs 21 hours sequential)"
echo "=" * 60

# Smart parameter selection for meaningful results
DATASET="opportunity"
WINDOW="4"
OVERLAP="0.5"
LIB_PER_CLASS="10"
TOPK="1"
METRICS="jaccard cosine"

# Sensor sets to test in parallel
SENSOR_SETS=("ankle" "torso" "hand" "chest" "back" "all")

# Create base output directory
BASE_OUTDIR="results/pilot_grid/rq2_parallel"
mkdir -p "$BASE_OUTDIR"

echo "📊 Starting parallel execution for ${#SENSOR_SETS[@]} sensor sets..."

# Launch parallel processes for each sensor set
pids=()
for sensor in "${SENSOR_SETS[@]}"; do
    OUTDIR="$BASE_OUTDIR/sensor_${sensor}"
    mkdir -p "$OUTDIR"
    
    echo "🔄 Starting sensor: $sensor"
    
    # Run each sensor set in background
    (
        uv run python -m thesis.exp.pilot_driver \
          --datasets $DATASET \
          --window_durations $WINDOW \
          --overlaps $OVERLAP \
          --library_per_class $LIB_PER_CLASS \
          --max_query_per_class 50 \
          --metrics $METRICS \
          --sensor_sets $sensor \
          --topk $TOPK \
          --progress \
          --output_dir "$OUTDIR" \
          > "$OUTDIR/log.txt" 2>&1
        
        echo "✅ Completed sensor: $sensor at $(date)"
    ) &
    
    pids+=($!)
done

echo "🎯 Launched ${#pids[@]} parallel processes"
echo "📊 Process IDs: ${pids[*]}"

# Wait for all processes to complete
echo "⏳ Waiting for all processes to complete..."
for pid in "${pids[@]}"; do
    wait $pid
    echo "✅ Process $pid completed"
done

echo ""
echo "🎉 All parallel executions completed!"
echo "📁 Results available in: $BASE_OUTDIR"

# Consolidate results
echo "📊 Consolidating results..."
FINAL_RESULTS="$BASE_OUTDIR/consolidated_results.csv"

# Write header
echo "dataset,sensor_set,window_size,overlap,lib_per_class,metric,activity_hit@1,activity_mrr,type_hit@1,type_mrr,loc_hit@1,loc_mrr" > "$FINAL_RESULTS"

# Combine all results (skip headers)
for sensor in "${SENSOR_SETS[@]}"; do
    SENSOR_FILE="$BASE_OUTDIR/sensor_${sensor}/pilot_grid_results.csv"
    if [ -f "$SENSOR_FILE" ]; then
        tail -n +2 "$SENSOR_FILE" >> "$FINAL_RESULTS"
    fi
done

echo "📈 Final consolidated results:"
echo "Lines: $(wc -l < "$FINAL_RESULTS")"
echo "Sample:"
head -3 "$FINAL_RESULTS"

echo ""
echo "🎯 Parallel execution summary:"
echo "- Sensors tested: ${#SENSOR_SETS[@]}"
echo "- Metrics per sensor: 2 (jaccard, cosine)"
echo "- Total configurations: $((${#SENSOR_SETS[@]} * 2))"
echo "- Results file: $FINAL_RESULTS"
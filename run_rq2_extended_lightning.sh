#!/bin/bash

# Extended Lightning-Fast RQ2 Test with Optimized NDG
# Goal: Test multiple sensor combinations in ~30 minutes with 5x speedup

export THESIS_DATA="/Users/nterlemes/personal/thesis_unified/thesis/Data"

echo "⚡⚡⚡ Starting Extended Lightning-Fast RQ2 Test"
echo "🎯 Goal: Multiple sensor combinations in ~30 minutes"
echo "🔬 Strategy: Optimized NDG + representative sensor sets"
echo "⚡ Using Epanechnikov kernel (5x speedup confirmed)"
echo "=" * 60

# Smart parameter selection for meaningful comparison
DATASET="opportunity"
WINDOW="4"
OVERLAP="0.5"
LIB_PER_CLASS="5"        # Increased from 3 for better reliability
TOPK="1"
METRICS="jaccard cosine" # 2 most important metrics

# Representative sensor sets for comparison
SENSOR_SETS=("ankle" "torso" "hand" "all")  # 4 key combinations

echo "📊 Testing ${#SENSOR_SETS[@]} sensor sets with optimized NDG:"
echo "   Sensors: ${SENSOR_SETS[*]}"
echo "   Metrics: $METRICS"
echo "   Expected pairs per sensor: ~1,875 (75 pairs/sec = 25 seconds each)"
echo "   Total estimated time: ~25-30 minutes"
echo ""

# Create base output directory
BASE_OUTDIR="results/pilot_grid/rq2_extended_lightning"
mkdir -p "$BASE_OUTDIR"

echo "🚀 Starting extended lightning test..."
echo "⏰ Started at: $(date)"

start_time=$(date +%s)

# Test each sensor set sequentially (to avoid memory conflicts)
for i in "${!SENSOR_SETS[@]}"; do
    sensor="${SENSOR_SETS[$i]}"
    OUTDIR="$BASE_OUTDIR/sensor_${sensor}"
    mkdir -p "$OUTDIR"
    
    echo ""
    echo "📡 [$((i+1))/${#SENSOR_SETS[@]}] Testing sensor: $sensor"
    echo "⏰ Started at: $(date)"
    
    # Run with optimized parameters
    uv run python -m thesis.exp.pilot_driver \
      --datasets $DATASET \
      --window_durations $WINDOW \
      --overlaps $OVERLAP \
      --library_per_class $LIB_PER_CLASS \
      --max_query_per_class 30 \
      --metrics $METRICS \
      --sensor_sets $sensor \
      --topk $TOPK \
      --progress \
      --output_dir "$OUTDIR" \
      2>&1 | tee "$OUTDIR/log.txt"
    
    # Quick results preview
    if [ -f "$OUTDIR/pilot_grid_results.csv" ]; then
        echo "✅ Completed sensor: $sensor"
        echo "📊 Results preview:"
        tail -n +2 "$OUTDIR/pilot_grid_results.csv" | head -2
    else
        echo "❌ Failed for sensor: $sensor"
    fi
    
    echo "⏱️  Sensor $sensor completed at: $(date)"
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "🎉 Extended lightning test completed!"
echo "⏱️  Total runtime: ${duration} seconds (~$((duration/60)) minutes)"
echo "📁 Results available in: $BASE_OUTDIR"

# Consolidate results
echo ""
echo "📊 Consolidating results..."
FINAL_RESULTS="$BASE_OUTDIR/consolidated_results.csv"

# Write header
echo "dataset,sensor_set,window_size,overlap,lib_per_class,metric,activity_hit@1,activity_mrr,type_hit@1,type_mrr,loc_hit@1,loc_mrr" > "$FINAL_RESULTS"

# Combine all results (skip headers)
for sensor in "${SENSOR_SETS[@]}"; do
    SENSOR_FILE="$BASE_OUTDIR/sensor_${sensor}/pilot_grid_results.csv"
    if [ -f "$SENSOR_FILE" ]; then
        tail -n +2 "$SENSOR_FILE" >> "$FINAL_RESULTS"
        echo "✅ Added results for: $sensor"
    else
        echo "❌ No results found for: $sensor"
    fi
done

echo ""
echo "📈 Final Performance Summary:"
echo "Results file: $FINAL_RESULTS"
echo "Total results: $(wc -l < "$FINAL_RESULTS") lines"

echo ""
echo "🎯 Performance Comparison by Sensor:"
if [ -f "$FINAL_RESULTS" ]; then
    echo "Sensor Set | Activity Hit@1 | MRR"
    echo "-----------|----------------|----"
    
    # Parse results for quick comparison
    for sensor in "${SENSOR_SETS[@]}"; do
        # Get best performance for this sensor (cosine metric)
        perf=$(grep ",$sensor," "$FINAL_RESULTS" | grep ",cosine," | cut -d',' -f7 | head -1)
        mrr=$(grep ",$sensor," "$FINAL_RESULTS" | grep ",cosine," | cut -d',' -f8 | head -1)
        
        if [ ! -z "$perf" ]; then
            printf "%-10s | %-14s | %.3f\n" "$sensor" "$perf" "$mrr"
        fi
    done
fi

echo ""
echo "💡 Key Insights:"
echo "- Single sensors (ankle, torso, hand): Individual performance"
echo "- All sensors: Multi-sensor fusion performance"
echo "- Use these results to decide on full experiment scope"

echo ""
echo "🚀 Next Steps:"
echo "- If results look good: Run parallel version for full comparison" 
echo "- If specific sensors excel: Focus experiments on those"
echo "- Total runtime with 5x speedup makes full experiments feasible!"
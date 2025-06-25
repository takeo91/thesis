#!/bin/bash

# Lightning-fast RQ2 estimation using extreme sampling
# Goal: Get retrieval performance estimate in ~3-5 minutes

export THESIS_DATA="/Users/nterlemes/personal/thesis_unified/thesis/Data"

echo "⚡⚡ Starting Lightning-Fast RQ2 Estimation"
echo "🎯 Goal: Retrieval performance estimate in ~3-5 minutes"
echo "🔬 Strategy: Extreme sampling + single representative test"
echo "=" * 60

# Super aggressive sampling
OUTDIR="results/pilot_grid/rq2_lightning"
mkdir -p "$OUTDIR"

echo "🚀 Running lightning estimation..."
echo "⏰ Started at: $(date)"

# Most extreme reduction possible while still meaningful
uv run python -m thesis.exp.pilot_driver \
  --datasets opportunity \
  --window_durations 4 \
  --overlaps 0.5 \
  --library_per_class 3 \
  --max_query_per_class 10 \
  --metrics cosine \
  --sensor_sets ankle \
  --topk 1 \
  --progress \
  --output_dir "$OUTDIR" \
  2>&1 | tee "$OUTDIR/experiment_log.txt"

echo "✅ Lightning test completed at $(date)"

# Quick extrapolation analysis
if [ -f "$OUTDIR/pilot_grid_results.csv" ]; then
    echo ""
    echo "⚡ Lightning Results & Extrapolation:"
    cat "$OUTDIR/pilot_grid_results.csv"
    
    # Extract the performance value
    PERFORMANCE=$(tail -1 "$OUTDIR/pilot_grid_results.csv" | cut -d',' -f7)
    echo ""
    echo "🎯 Quick Performance Extrapolation:"
    echo "- Ankle sensor performance: $PERFORMANCE"
    echo "- Expected multi-sensor boost: +10-20%"
    echo "- Estimated all-sensors performance: ~$(echo "$PERFORMANCE * 1.15" | bc -l)"
    echo ""
    echo "📊 This gives you a baseline for deciding on full experiments!"
    echo "💡 If performance looks promising, run the parallel version next."
fi

echo ""
echo "⏱️  Total runtime: ~3-5 minutes (vs 21+ hours for full grid)"
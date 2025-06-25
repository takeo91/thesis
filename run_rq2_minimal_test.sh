#!/bin/bash

# Minimal RQ2 test to verify Phase 3 improvements
# Single configuration for fast validation (~15-20 minutes)

# Set correct data path
export THESIS_DATA="/Users/nterlemes/personal/thesis_unified/thesis/Data"

echo "🧪 Starting Minimal RQ2 Test - Phase 3 Improvements Verification"
echo "📊 Testing: 1 dataset, 1 sensor, 1 window, 1 overlap, 2 metrics"
echo "⏱️  Expected runtime: ~15-20 minutes"
echo "=" * 60

# Minimal configuration for fast testing
OUTDIR="results/pilot_grid/rq2_minimal_test/lib5_topk1"
mkdir -p "$OUTDIR"

echo "📈 Running minimal test: library_per_class=5, topk=1"
echo "⏰ Started at: $(date)"

# Run with minimal parameters for quick validation
uv run python -m thesis.exp.pilot_driver \
  --datasets opportunity \
  --window_durations 4 \
  --overlaps 0.5 \
  --library_per_class 5 \
  --max_query_per_class 50 \
  --metrics jaccard cosine \
  --sensor_sets ankle \
  --topk 1 \
  --progress \
  --output_dir "$OUTDIR" \
  2>&1 | tee "$OUTDIR/experiment_log.txt"

echo "✅ Minimal test completed at $(date)"
echo "📁 Results saved to: $OUTDIR"

# Quick summary
echo ""
echo "📊 Quick Results Summary:"
if [ -f "$OUTDIR/pilot_grid_results.csv" ]; then
    echo "Results file created successfully:"
    wc -l "$OUTDIR/pilot_grid_results.csv"
    echo ""
    echo "Sample results:"
    head -5 "$OUTDIR/pilot_grid_results.csv"
else
    echo "❌ No results file found"
fi

echo ""
echo "🎉 Phase 3 improvements verification complete!"
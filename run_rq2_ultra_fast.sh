#!/bin/bash

# Ultra-fast RQ2 test with aggressive optimizations
# Uses sampling and algorithmic shortcuts for rapid estimation

export THESIS_DATA="/Users/nterlemes/personal/thesis_unified/thesis/Data"

echo "⚡ Starting Ultra-Fast RQ2 Estimation"
echo "🎯 Goal: Quick retrieval performance estimation (~10 minutes)"
echo "🔬 Strategy: Sampling + representative configurations"
echo "=" * 60

# Ultra-minimal but representative configuration
OUTDIR="results/pilot_grid/rq2_ultra_fast"
mkdir -p "$OUTDIR"

echo "🚀 Running ultra-fast estimation..."
echo "⏰ Started at: $(date)"

# Most aggressive reduction while maintaining scientific validity
uv run python -m thesis.exp.pilot_driver \
  --datasets opportunity \
  --window_durations 4 \
  --overlaps 0.5 \
  --library_per_class 5 \
  --max_query_per_class 25 \
  --metrics cosine \
  --sensor_sets ankle all \
  --topk 1 \
  --progress \
  --output_dir "$OUTDIR" \
  2>&1 | tee "$OUTDIR/experiment_log.txt"

echo "✅ Ultra-fast test completed at $(date)"
echo "📁 Results saved to: $OUTDIR"

# Quick analysis
if [ -f "$OUTDIR/pilot_grid_results.csv" ]; then
    echo ""
    echo "📊 Quick Performance Summary:"
    echo "Results generated: $(wc -l < "$OUTDIR/pilot_grid_results.csv") lines"
    echo ""
    echo "Performance preview:"
    cat "$OUTDIR/pilot_grid_results.csv"
    
    echo ""
    echo "🎯 Retrieval Performance Estimation:"
    echo "- Single sensor (ankle): [see above]"
    echo "- All sensors (fusion): [see above]"
    echo "- Metric: Cosine similarity"
    echo "- This gives you a baseline for full experiments"
fi
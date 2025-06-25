#!/bin/bash

# Monitor RQ2 Grid Progress
echo "=== RQ2 Grid Progress Monitor ==="
echo "Monitoring grid execution at $(date)"
echo

RESULTS_DIR="/Users/nterlemes/personal/thesis_unified/thesis/results/pilot_grid/rq2_full"

while true; do
    clear
    echo "=== RQ2 Grid Progress Monitor === $(date)"
    echo
    
    # Check if main process is still running
    if pgrep -f "pilot_driver" > /dev/null; then
        echo "✅ Grid script is RUNNING"
        
        # Show current configuration being processed
        CURRENT_CONFIG=$(ps aux | grep pilot_driver | grep -v grep | head -1 | grep -o 'lib[0-9]*_topk[0-9]*' || echo "unknown")
        echo "🔄 Current configuration: $CURRENT_CONFIG"
    else
        echo "❌ Grid script is NOT RUNNING"
    fi
    
    echo
    echo "📊 Progress by configuration:"
    echo "Config          | Lines | Status"
    echo "----------------|-------|--------"
    
    for config_dir in "$RESULTS_DIR"/lib*_topk*; do
        if [ -d "$config_dir" ]; then
            config_name=$(basename "$config_dir")
            csv_file="$config_dir/pilot_grid_results.csv"
            
            if [ -f "$csv_file" ]; then
                lines=$(wc -l < "$csv_file")
                if [ "$lines" -gt 1 ]; then
                    echo "$config_name | $lines     | ✅ Progress"
                else
                    echo "$config_name | $lines     | 🔄 Started"
                fi
            else
                echo "$config_name | 0     | ⏳ Waiting"
            fi
        fi
    done
    
    echo
    echo "📈 Latest results from active configurations:"
    
    # Show last few lines from most recent file
    LATEST_FILE=$(find "$RESULTS_DIR" -name "pilot_grid_results.csv" -exec ls -t {} \; | head -1)
    if [ -n "$LATEST_FILE" ]; then
        echo "Latest from: $(basename $(dirname $LATEST_FILE))"
        tail -3 "$LATEST_FILE" 2>/dev/null || echo "No data yet"
    fi
    
    echo
    echo "Press Ctrl+C to exit monitor"
    sleep 5
done
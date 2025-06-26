#!/usr/bin/env python3
"""
Monitor current experiment and automatically start expanded metrics experiment when done.
"""

import time
import subprocess
import sys
from pathlib import Path

def is_experiment_running():
    """Check if the current basic experiment is still running."""
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True
        )
        return 'run_experiment.py' in result.stdout
    except:
        return False

def get_experiment_progress():
    """Get latest progress from experiment log."""
    try:
        log_file = Path("experiment_expanded_metrics.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Get last few lines with progress info
                progress_lines = [line for line in lines[-10:] if 'progress:' in line]
                return progress_lines[-1].strip() if progress_lines else "No progress info"
        return "Log file not found"
    except:
        return "Error reading log"

def main():
    print("🔍 Monitoring current experiment progress...")
    print("⏳ Will automatically start expanded metrics experiment when current one finishes\n")
    
    while is_experiment_running():
        progress = get_experiment_progress()
        current_time = time.strftime('%H:%M:%S')
        print(f"[{current_time}] Current: {progress}")
        time.sleep(30)  # Check every 30 seconds
    
    print(f"\n✅ Basic experiment completed at {time.strftime('%H:%M:%S')}")
    print("🚀 Starting EXPANDED METRICS experiment...\n")
    
    # Run the expanded metrics experiment
    try:
        process = subprocess.Popen(
            [sys.executable, "run_expanded_metrics_experiment.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        with open("experiment_expanded_metrics_full.log", "w") as log_file:
            for line in process.stdout:
                print(line.rstrip())
                log_file.write(line)
                log_file.flush()
        
        process.wait()
        
        if process.returncode == 0:
            print("\n🎯 EXPANDED METRICS experiment completed successfully!")
        else:
            print(f"\n❌ EXPANDED METRICS experiment failed with return code {process.returncode}")
            
    except Exception as e:
        print(f"❌ Failed to start expanded metrics experiment: {e}")

if __name__ == "__main__":
    main()
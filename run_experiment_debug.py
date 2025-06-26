#!/usr/bin/env python3
"""
Debug version of the experiment to trace the exact error location.
"""

import sys
import os
import time
import traceback
from pathlib import Path

def run_unified_experiment_debug():
    """Run the unified windowing experiment with detailed debugging."""
    try:
        print("🚀 Starting Debug Unified Windowing Experiment...")
        
        # Import after ensuring we're in the right directory
        from thesis.exp.unified_windowing_experiment import UnifiedWindowingExperiment
        from thesis.data import WindowConfig
        
        print("✅ Imports successful")
        
        # Initialize experiment
        print("\n📝 Initializing experiment...")
        experiment = UnifiedWindowingExperiment(
            window_config=WindowConfig(window_size=120, overlap_ratio=0.5),
            cache_dir="cache/unified_production"
        )
        print("✅ Experiment initialized")
        
        # Create standard windows first
        print("\n🪟 Creating standard windows...")
        window_info = experiment.create_standard_windows()
        print(f"✅ Standard windows created: {window_info}")
        
        # Run just Locomotion first to debug
        print("\n🔬 Running single label experiment for debugging...")
        try:
            label_result = experiment.run_label_type_experiment("Locomotion", ["jaccard"])
            print(f"✅ Locomotion experiment completed: {label_result}")
        except Exception as e:
            print(f"❌ Locomotion experiment failed: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Change to thesis directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = run_unified_experiment_debug()
    sys.exit(0 if success else 1)
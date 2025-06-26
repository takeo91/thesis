#!/usr/bin/env python3
"""
Run the unified windowing experiment with proper error handling and progress monitoring.
"""

import sys
import os
import time
import traceback
from pathlib import Path

def run_unified_experiment():
    """Run the unified windowing experiment with monitoring."""
    try:
        print("🚀 Starting Unified Windowing Experiment...")
        print(f"   Python: {sys.version}")
        print(f"   Working Dir: {os.getcwd()}")
        print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
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
        
        # Run the experiment
        print("\n🔬 Running multi-label experiment...")
        print("   This will take several hours due to the comprehensive analysis...")
        
        start_time = time.time()
        results = experiment.run_multi_label_experiment(
            label_types=["Locomotion", "ML_Both_Arms", "HL_Activity"],
            metrics=["jaccard", "cosine", "dice", "pearson", "overlap_coefficient"]
        )
        end_time = time.time()
        
        # Save results
        print(f"\n💾 Saving results...")
        output_dir = Path("results/unified_windowing")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(output_dir / "unified_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Print summary
        duration = end_time - start_time
        print(f"\n🎯 Unified Windowing Experiment Complete!")
        print(f"   Duration: {duration/3600:.2f} hours")
        print(f"   Standard windows: {results['window_info']['num_windows']}")
        print(f"   Results saved to: {output_dir / 'unified_results.pkl'}")
        
        # Print label type results
        for label_type in results['label_type_results']:
            label_result = results['label_type_results'][label_type]
            print(f"\n   {label_type} Results:")
            for metric, data in label_result['results'].items():
                print(f"     {metric}: Hit@1={data['hit_at_1']:.3f}, MRR={data['mrr']:.3f}")
        
        print(f"\n✅ Experiment completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Please install required dependencies:")
        print("   pip install numpy scipy pandas scikit-learn joblib tqdm matplotlib seaborn")
        return False
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Change to thesis directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = run_unified_experiment()
    sys.exit(0 if success else 1)
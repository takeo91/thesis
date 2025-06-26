#!/usr/bin/env python3
"""
Run the unified windowing experiment with EXPANDED METRICS for comprehensive evaluation.

This experiment includes all 13 similarity metrics:
- Basic: jaccard, cosine, dice, pearson, overlap_coefficient  
- Expanded: spearman, kendall_tau, tversky, weighted_jaccard, mahalanobis,
           jensen_shannon, bhattacharyya_coefficient, hellinger
"""

import sys
import os
import time
import traceback
from pathlib import Path

def run_expanded_metrics_experiment():
    """Run the unified windowing experiment with all 13 metrics."""
    try:
        print("🚀 Starting EXPANDED METRICS Unified Windowing Experiment...")
        print(f"   Python: {sys.version}")
        print(f"   Working Dir: {os.getcwd()}")
        print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Import after ensuring we're in the right directory
        from thesis.exp.unified_windowing_experiment import UnifiedWindowingExperiment
        from thesis.data import WindowConfig
        
        print("✅ Imports successful")
        
        # Initialize experiment
        print("\n📝 Initializing expanded metrics experiment...")
        experiment = UnifiedWindowingExperiment(
            window_config=WindowConfig(window_size=120, overlap_ratio=0.5),
            cache_dir="cache/unified_expanded_metrics"
        )
        print("✅ Experiment initialized")
        
        # Define the full expanded metrics list (using correct available metric names)
        expanded_metrics = [
            # Basic metrics (5)
            "jaccard", "cosine", "dice", "pearson", "overlap_coefficient",
            # Information-theoretic metrics (3) 
            "JensenShannon", "BhattacharyyaCoefficient", "HellingerDistance",
            # Distance-based metrics (3)
            "Similarity_Euclidean", "Similarity_Chebyshev", "Similarity_Hamming",
            # Additional fuzzy metrics (5)
            "MeanMinOverMax", "MeanDiceCoefficient", "HarmonicMean",
            "EarthMoversDistance", "EnergyDistance"
        ]
        
        print(f"\n🔬 Running multi-label experiment with {len(expanded_metrics)} metrics...")
        print(f"   Metrics: {expanded_metrics}")
        print("   This will take SIGNIFICANTLY longer due to comprehensive analysis...")
        print(f"   Estimated time: ~{len(expanded_metrics)/5 * 2:.1f}x longer than basic experiment")
        
        start_time = time.time()
        results = experiment.run_multi_label_experiment(
            label_types=["Locomotion", "ML_Both_Arms", "HL_Activity"],
            metrics=expanded_metrics
        )
        end_time = time.time()
        
        # Save results
        print(f"\n💾 Saving expanded metrics results...")
        output_dir = Path("results/unified_windowing_expanded")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(output_dir / "expanded_metrics_results.pkl", 'wb') as f:
            pickle.dump(results, f)
        
        # Create summary report
        summary_file = output_dir / "expanded_metrics_summary.md"
        with open(summary_file, 'w') as f:
            f.write("# Expanded Metrics Unified Windowing Experiment Results\n\n")
            f.write(f"**Experiment Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Duration**: {(end_time - start_time)/3600:.2f} hours\n")
            f.write(f"**Metrics Evaluated**: {len(expanded_metrics)}\n")
            f.write(f"**Standard Windows**: {results['window_info']['num_windows']}\n\n")
            
            f.write("## Metrics Used\n\n")
            for i, metric in enumerate(expanded_metrics, 1):
                f.write(f"{i}. `{metric}`\n")
            
            f.write("\n## Results by Label Type\n\n")
            
            for label_type in results['label_type_results']:
                label_result = results['label_type_results'][label_type]
                if 'error' in label_result:
                    f.write(f"### {label_type}\n**ERROR**: {label_result['error']}\n\n")
                    continue
                    
                f.write(f"### {label_type}\n\n")
                f.write("| Metric | Hit@1 | MRR |\n")
                f.write("|--------|-------|-----|\n")
                
                # Sort by Hit@1 performance
                metric_results = [(metric, data) for metric, data in label_result['results'].items()]
                metric_results.sort(key=lambda x: x[1]['hit_at_1'], reverse=True)
                
                for metric, data in metric_results:
                    hit_at_1 = data['hit_at_1']
                    mrr = data['mrr']
                    f.write(f"| {metric} | {hit_at_1:.3f} | {mrr:.3f} |\n")
                
                # Find best performer
                if metric_results:
                    best_metric, best_data = metric_results[0]
                    f.write(f"\n**Best Performer**: `{best_metric}` (Hit@1={best_data['hit_at_1']:.3f})\n\n")
        
        # Print summary
        duration = end_time - start_time
        print(f"\n🎯 EXPANDED METRICS Experiment Complete!")
        print(f"   Duration: {duration/3600:.2f} hours")
        print(f"   Metrics evaluated: {len(expanded_metrics)}")
        print(f"   Standard windows: {results['window_info']['num_windows']}")
        print(f"   Results saved to: {output_dir / 'expanded_metrics_results.pkl'}")
        print(f"   Summary report: {summary_file}")
        
        # Print best performers for each label type
        print(f"\n🏆 BEST PERFORMERS:")
        for label_type in results['label_type_results']:
            label_result = results['label_type_results'][label_type]
            if 'error' in label_result:
                print(f"   {label_type}: ERROR - {label_result['error']}")
                continue
                
            best_metric = None
            best_score = 0
            for metric, data in label_result['results'].items():
                if data['hit_at_1'] > best_score:
                    best_score = data['hit_at_1']
                    best_metric = metric
            
            if best_metric:
                mrr = label_result['results'][best_metric]['mrr']
                print(f"   {label_type}: {best_metric} (Hit@1={best_score:.3f}, MRR={mrr:.3f})")
        
        print(f"\n✅ Expanded metrics experiment completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Please install required dependencies")
        return False
        
    except Exception as e:
        print(f"❌ Expanded metrics experiment failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Change to thesis directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = run_expanded_metrics_experiment()
    sys.exit(0 if success else 1)
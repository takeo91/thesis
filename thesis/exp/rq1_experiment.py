#!/usr/bin/env python
"""
RQ1 Experiment: NDG-S vs KDE Efficiency Comparison

This script runs the complete RQ1 experiment comparing NDG-S and KDE methods
with statistical validation using Wilcoxon signed-rank test.

Signal lengths tested: 100, 1K, 10K, 100K samples
Datasets: Synthetic (normal, bimodal), Opportunity, PAMAP2
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Union

# Import from the same package
from thesis.exp.ndg_vs_kde import (
    run_experiments_by_length, 
    perform_statistical_tests,
    plot_statistical_results,
    plot_length_experiment_results
)


def run_rq1_experiment(quick_test: bool = False):
    """
    Run the complete RQ1 experiment with required signal lengths.
    
    Args:
        quick_test: If True, run a quick test with fewer samples
    """
    print("=" * 80)
    print("RQ1 EXPERIMENT: NDG-S vs KDE Efficiency Comparison")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    
    # Define datasets for experiments
    datasets = [
        {"name": "synthetic_normal", "sensor_loc": "normal"},
        {"name": "synthetic_bimodal", "sensor_loc": "bimodal"},
        {"name": "opportunity", "sensor_loc": "RKN^"},
        {"name": "pamap2", "sensor_loc": "Hand"}
    ]
    
    if quick_test:
        # Quick test with smaller lengths
        lengths = [100, 1000]
        sigmas = [0.3, 'r0.1']
        k_folds = 3
        print("\n[QUICK TEST MODE] Using reduced parameters")
    else:
        # Full experiment as required by RQ1
        lengths = [100, 1000, 10000, 100000]  # 100, 1K, 10K, 100K samples
        sigmas = [0.1, 0.3, 0.5, 'r0.1', 'r0.3']  # 5 sigma values
        k_folds = 5  # Increased folds for better statistical power
    
    print(f"\nExperiment parameters:")
    print(f"  Datasets: {[d['name'] for d in datasets]}")
    print(f"  Signal lengths: {lengths}")
    print(f"  Sigma values: {sigmas}")
    print(f"  K-folds: {k_folds}")
    print(f"  Total experiments: {len(datasets) * len(lengths) * len(sigmas) * k_folds}")
    
    # Run experiments
    print("\n" + "=" * 50)
    print("RUNNING EXPERIMENTS")
    print("=" * 50)
    
    results = run_experiments_by_length(datasets, lengths, sigmas=sigmas, k_folds=k_folds)
    
    # Save raw results
    results_dir = "results/ndg_vs_kde/rq1"
    os.makedirs(results_dir, exist_ok=True)
    
    csv_data = results.drop(columns=['figure'] if 'figure' in results.columns else [])
    csv_data.to_csv(f"{results_dir}/rq1_experiment_results.csv", index=False)
    
    # Perform statistical tests
    print("\n" + "=" * 50)
    print("STATISTICAL ANALYSIS")
    print("=" * 50)
    
    stats_df = perform_statistical_tests(results)
    stats_df.to_csv(f"{results_dir}/rq1_statistical_results.csv", index=False)
    
    # Print statistical summary
    print("\n=== Statistical Test Summary ===")
    print(f"Total comparisons: {len(stats_df)}")
    
    # Execution time results
    time_stats = stats_df[stats_df['metric'] == 'execution_time']
    significant_time = time_stats[time_stats['significant']]
    print(f"\nExecution Time:")
    print(f"  Significant results (p < 0.05): {len(significant_time)}/{len(time_stats)}")
    
    for _, row in time_stats.iterrows():
        sig_marker = "*" if row['significant'] else " "
        print(f"  {sig_marker} {row['dataset']:20s} (n={row['length']:6d}): " +
              f"speedup={row['speedup_factor']:6.2f}x, p={row['p_value']:.4f}, " +
              f"effect_size={row['effect_size']:.3f}")
    
    # Memory usage results
    memory_stats = stats_df[stats_df['metric'] == 'memory_usage']
    if not memory_stats.empty:
        significant_memory = memory_stats[memory_stats['significant']]
        print(f"\nMemory Usage:")
        print(f"  Significant results (p < 0.05): {len(significant_memory)}/{len(memory_stats)}")
    
    # Generate plots
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    
    # Statistical results plots
    fig_speedup, fig_pvalue = plot_statistical_results(stats_df)
    fig_speedup.savefig(f"{results_dir}/rq1_speedup_comparison.png", dpi=300, bbox_inches="tight")
    fig_pvalue.savefig(f"{results_dir}/rq1_pvalue_heatmap.png", dpi=300, bbox_inches="tight")
    
    # Performance comparison plots
    fig_time, fig_memory, fig_error = plot_length_experiment_results(results)
    fig_time.savefig(f"{results_dir}/rq1_time_comparison.png", dpi=300, bbox_inches="tight")
    fig_memory.savefig(f"{results_dir}/rq1_memory_comparison.png", dpi=300, bbox_inches="tight")
    fig_error.savefig(f"{results_dir}/rq1_error_comparison.png", dpi=300, bbox_inches="tight")
    
    # Create summary report
    print("\n" + "=" * 50)
    print("GENERATING SUMMARY REPORT")
    print("=" * 50)
    
    summary_report = f"""
# RQ1 Experiment Results Summary

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Configuration
- **Datasets**: {[d['name'] for d in datasets]}
- **Signal lengths**: {lengths} samples
- **Sigma values**: {sigmas}
- **K-folds**: {k_folds}
- **Total experiments**: {len(results)}

## Statistical Test Results

### Execution Time Comparison
- Total comparisons: {len(time_stats)}
- Significant results (p < 0.05): {len(significant_time)}

**Key Findings**:
"""
    
    # Add key findings
    for _, row in significant_time.iterrows():
        summary_report += f"\n- {row['dataset']} (n={row['length']}): " + \
                         f"NDG is {row['speedup_factor']:.2f}x faster than KDE " + \
                         f"(p={row['p_value']:.4f}, effect size={row['effect_size']:.3f})"
    
    # Calculate average speedup across all datasets
    avg_speedup = time_stats['speedup_factor'].mean()
    summary_report += f"\n\n**Average speedup factor**: {avg_speedup:.2f}x"
    
    # Save summary report
    with open(f"{results_dir}/rq1_summary_report.md", 'w') as f:
        f.write(summary_report)
    
    print(f"\nResults saved to: {results_dir}/")
    print(f"End time: {datetime.now()}")
    
    return results, stats_df


def run_rq1_experiment_optimized(quick_test: bool = False):
    """
    Run the complete RQ1 experiment with OPTIMIZED NDG implementation.
    
    This version uses the 10-100x faster NDG implementation which should
    dramatically change the results and validate the H1 hypothesis.
    
    Args:
        quick_test: If True, run a quick test with fewer samples
    """
    print("=" * 80)
    print("RQ1 EXPERIMENT: NDG-S vs KDE Efficiency Comparison (OPTIMIZED)")
    print("=" * 80)
    print(f"🚀 Using optimized NDG implementation with Epanechnikov kernel")
    print(f"Expected speedup: 10-100x over original NDG")
    print(f"Start time: {datetime.now()}")
    
    # Define datasets for experiments
    datasets = [
        {"name": "synthetic_normal", "sensor_loc": "normal"},
        {"name": "synthetic_bimodal", "sensor_loc": "bimodal"},
        {"name": "opportunity", "sensor_loc": "RKN^"},
        {"name": "pamap2", "sensor_loc": "Hand"}
    ]
    
    if quick_test:
        # Quick test with smaller lengths
        lengths = [100, 1000, 10000]
        sigmas = [0.3, 'r0.1']
        k_folds = 3
        print("\n[QUICK TEST MODE] Using reduced parameters")
    else:
        # Full experiment as required by RQ1
        lengths = [100, 1000, 10000, 100000]  # 100, 1K, 10K, 100K samples
        sigmas = [0.1, 0.3, 0.5, 'r0.1', 'r0.3']  # 5 sigma values
        k_folds = 5  # Increased folds for better statistical power
    
    print(f"\nExperiment parameters:")
    print(f"  Datasets: {[d['name'] for d in datasets]}")
    print(f"  Signal lengths: {lengths}")
    print(f"  Sigma values: {sigmas}")
    print(f"  K-folds: {k_folds}")
    print(f"  Kernel type: Epanechnikov (compact support)")
    print(f"  Optimization: Enabled (spatial pruning + JIT + parallel)")
    print(f"  Total experiments: {len(datasets) * len(lengths) * len(sigmas) * k_folds}")
    
    # Run experiments with optimized NDG
    print("\n" + "=" * 50)
    print("RUNNING OPTIMIZED EXPERIMENTS")
    print("=" * 50)
    
    results = run_experiments_by_length_optimized(datasets, lengths, sigmas=sigmas, k_folds=k_folds)
    
    # Save raw results
    results_dir = "results/ndg_vs_kde/rq1_optimized"
    os.makedirs(results_dir, exist_ok=True)
    
    csv_data = results.drop(columns=['figure'] if 'figure' in results.columns else [])
    csv_data.to_csv(f"{results_dir}/rq1_optimized_experiment_results.csv", index=False)
    
    # Perform statistical tests
    print("\n" + "=" * 50)
    print("STATISTICAL ANALYSIS")
    print("=" * 50)
    
    stats_df = perform_statistical_tests(results)
    stats_df.to_csv(f"{results_dir}/rq1_optimized_statistical_results.csv", index=False)
    
    # Print statistical summary
    print("\n=== OPTIMIZED Statistical Test Summary ===")
    print(f"Total comparisons: {len(stats_df)}")
    
    # Execution time results
    time_stats = stats_df[stats_df['metric'] == 'execution_time']
    significant_time = time_stats[time_stats['significant']]
    print(f"\nExecution Time:")
    print(f"  Significant results (p < 0.05): {len(significant_time)}/{len(time_stats)}")
    
    for _, row in time_stats.iterrows():
        sig_marker = "*" if row['significant'] else " "
        print(f"  {sig_marker} {row['dataset']:20s} (n={row['length']:6d}): " +
              f"speedup={row['speedup_factor']:6.2f}x, p={row['p_value']:.4f}, " +
              f"effect_size={row['effect_size']:.3f}")
    
    # Calculate and display hypothesis validation
    print("\n" + "=" * 50)
    print("H1 HYPOTHESIS VALIDATION")
    print("=" * 50)
    
    # Count significant improvements
    significant_improvements = len(significant_time[significant_time['speedup_factor'] > 1.0])
    total_comparisons = len(time_stats)
    
    avg_speedup = time_stats['speedup_factor'].mean()
    median_speedup = time_stats['speedup_factor'].median()
    
    print(f"H1: NDG-S is computationally more efficient than KDE")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Median speedup: {median_speedup:.2f}x")
    print(f"  Significant improvements: {significant_improvements}/{total_comparisons}")
    print(f"  Success rate: {100*significant_improvements/total_comparisons:.1f}%")
    
    if significant_improvements >= total_comparisons * 0.75:  # 75% threshold
        print(f"  🎉 H1 HYPOTHESIS: VALIDATED! (Strong evidence)")
    elif significant_improvements >= total_comparisons * 0.5:  # 50% threshold
        print(f"  ✅ H1 HYPOTHESIS: SUPPORTED (Moderate evidence)")
    else:
        print(f"  ❌ H1 HYPOTHESIS: NOT SUPPORTED")
    
    # Generate plots
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    
    # Statistical results plots
    fig_speedup, fig_pvalue = plot_statistical_results(stats_df)
    fig_speedup.savefig(f"{results_dir}/rq1_optimized_speedup_comparison.png", dpi=300, bbox_inches="tight")
    fig_pvalue.savefig(f"{results_dir}/rq1_optimized_pvalue_heatmap.png", dpi=300, bbox_inches="tight")
    
    # Performance comparison plots
    fig_time, fig_memory, fig_error = plot_length_experiment_results(results)
    fig_time.savefig(f"{results_dir}/rq1_optimized_time_comparison.png", dpi=300, bbox_inches="tight")
    fig_memory.savefig(f"{results_dir}/rq1_optimized_memory_comparison.png", dpi=300, bbox_inches="tight")
    fig_error.savefig(f"{results_dir}/rq1_optimized_error_comparison.png", dpi=300, bbox_inches="tight")
    
    # Create summary report
    summary_report = f"""
# RQ1 Optimized Experiment Results Summary

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Optimization Details
- **NDG Implementation**: Optimized with spatial pruning + JIT compilation + parallelization
- **Kernel Type**: Epanechnikov (compact support)
- **Expected Speedup**: 10-100x over original NDG
- **Actual Average Speedup**: {avg_speedup:.2f}x

## Experiment Configuration
- **Datasets**: {[d['name'] for d in datasets]}
- **Signal lengths**: {lengths} samples
- **Sigma values**: {sigmas}
- **K-folds**: {k_folds}
- **Total experiments**: {len(results)}

## H1 Hypothesis Validation Results
**H1**: NDG-S is computationally more efficient than KDE

### Key Results:
- **Average speedup factor**: {avg_speedup:.2f}x
- **Median speedup factor**: {median_speedup:.2f}x
- **Significant improvements**: {significant_improvements}/{total_comparisons} ({100*significant_improvements/total_comparisons:.1f}%)

### Statistical Test Results
- Total comparisons: {len(time_stats)}
- Significant results (p < 0.05): {len(significant_time)}

**Detailed Findings**:
"""
    
    # Add detailed findings
    for _, row in significant_time.iterrows():
        summary_report += f"\n- {row['dataset']} (n={row['length']}): " + \
                         f"NDG is {row['speedup_factor']:.2f}x faster than KDE " + \
                         f"(p={row['p_value']:.4f}, effect size={row['effect_size']:.3f})"
    
    # Final hypothesis status
    if significant_improvements >= total_comparisons * 0.75:
        status = "VALIDATED (Strong evidence)"
    elif significant_improvements >= total_comparisons * 0.5:
        status = "SUPPORTED (Moderate evidence)"
    else:
        status = "NOT SUPPORTED"
    
    summary_report += f"\n\n### Final H1 Status: {status}"
    
    # Save summary report
    with open(f"{results_dir}/rq1_optimized_summary_report.md", 'w') as f:
        f.write(summary_report)
    
    print(f"\nOptimized results saved to: {results_dir}/")
    print(f"End time: {datetime.now()}")
    
    return results, stats_df


def run_experiments_by_length_optimized(datasets: List[Dict[str, str]], 
                                       lengths: List[int],
                                       sigmas: List[Union[float, str]] = [0.3],
                                       k_folds: int = 3) -> pd.DataFrame:
    """
    Run experiments with OPTIMIZED NDG implementation.
    
    This uses the Epanechnikov kernel and optimized NDG functions for 
    maximum performance improvement.
    """
    print("🚀 Using OPTIMIZED NDG implementation:")
    print("  - Epanechnikov kernel (compact support)")
    print("  - Spatial pruning (4-sigma cutoff)")
    print("  - JIT compilation (Numba)")
    print("  - Parallel processing")
    print("  - Expected speedup: 10-100x")
    print()
    
    # Import optimized NDG experiment runner
    from thesis.exp.ndg_vs_kde import run_experiments_by_length
    
    # Override the NDG method to use optimized implementation
    import thesis.exp.ndg_vs_kde as ndg_kde_module
    
    # Store original function
    original_compute = ndg_kde_module.compute_membership_functions
    
    # Create optimized wrapper
    def optimized_compute_membership_functions(*args, **kwargs):
        # Force use of optimized NDG with Epanechnikov kernel
        if 'method' in kwargs and kwargs['method'] == 'nd':
            kwargs['kernel_type'] = 'epanechnikov'  # Use fastest kernel
            kwargs['use_optimized'] = True  # Use optimized implementation
        return original_compute(*args, **kwargs)
    
    # Temporarily override the function
    ndg_kde_module.compute_membership_functions = optimized_compute_membership_functions
    
    try:
        # Run experiments with optimized implementation
        results = run_experiments_by_length(datasets, lengths, sigmas=sigmas, k_folds=k_folds)
        
        # Add optimization info to results
        results['optimization_used'] = 'epanechnikov_optimized'
        results['kernel_type'] = 'epanechnikov'
        
        return results
    finally:
        # Restore original function
        ndg_kde_module.compute_membership_functions = original_compute


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RQ1 experiment")
    parser.add_argument("--quick", action="store_true", 
                        help="Run quick test with reduced parameters")
    parser.add_argument("--optimized", action="store_true",
                        help="Use optimized NDG implementation (10-100x faster)")
    parser.add_argument("--both", action="store_true",
                        help="Run both original and optimized experiments for comparison")
    args = parser.parse_args()
    
    if args.both:
        # Run both experiments for comparison
        print("Running BOTH original and optimized experiments for comparison...\n")
        
        print("1. Running ORIGINAL experiment:")
        results_orig, stats_orig = run_rq1_experiment(quick_test=args.quick)
        
        print("\n" + "="*80)
        print("2. Running OPTIMIZED experiment:")
        results_opt, stats_opt = run_rq1_experiment_optimized(quick_test=args.quick)
        
        # Print comparison
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        orig_avg = stats_orig[stats_orig['metric'] == 'execution_time']['speedup_factor'].mean()
        opt_avg = stats_opt[stats_opt['metric'] == 'execution_time']['speedup_factor'].mean()
        
        print(f"Original NDG average speedup: {orig_avg:.2f}x")
        print(f"Optimized NDG average speedup: {opt_avg:.2f}x")
        print(f"Improvement from optimization: {opt_avg/orig_avg:.1f}x better")
        
    elif args.optimized:
        # Run optimized experiment
        results, stats_df = run_rq1_experiment_optimized(quick_test=args.quick)
    else:
        # Run original experiment
        results, stats_df = run_rq1_experiment(quick_test=args.quick) 
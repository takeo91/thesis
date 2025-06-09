#!/usr/bin/env python3
"""
RQ1 Results Analysis - NDG-S vs KDE Efficiency Comparison
Analyzes the experimental results to validate Hypothesis H1
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

def analyze_performance_statistics(df):
    """Perform comprehensive statistical analysis of NDG vs KDE performance."""
    results = []
    
    for (dataset, length), group in df.groupby(['dataset', 'length']):
        n_samples = len(group)
        if n_samples < 3:  # Need minimum samples for statistical tests
            continue
            
        # Extract performance data
        ndg_times = group['ndg_time'].values
        kde_times = group['kde_time'].values
        
        # Calculate basic statistics
        ndg_median = np.median(ndg_times)
        kde_median = np.median(kde_times)
        speedup = kde_median / (ndg_median + 1e-12)
        
        # Wilcoxon signed-rank test (paired, non-parametric)
        try:
            if np.all(ndg_times == kde_times):
                p_value = 1.0
                statistic = 0
            else:
                statistic, p_value = stats.wilcoxon(
                    ndg_times, kde_times, alternative='less'
                )
        except ValueError:
            p_value = 1.0
            statistic = 0
        
        # Effect size (Cohen's d for paired data)
        time_diff = kde_times - ndg_times
        effect_size = np.mean(time_diff) / (np.std(time_diff) + 1e-12)
        
        # Memory analysis
        ndg_memory = group['ndg_memory'].values
        kde_memory = group['kde_memory'].values
        memory_savings = 1 - (np.median(ndg_memory) / (np.median(kde_memory) + 1e-12))
        
        # Approximation quality
        kl_divergence = group['kl_divergence'].mean()
        
        results.append({
            'dataset': dataset,
            'length': length,
            'actual_size': group['actual_size'].iloc[0],
            'n_experiments': n_samples,
            'ndg_median_time': ndg_median,
            'kde_median_time': kde_median,
            'speedup_factor': speedup,
            'improvement_pct': (speedup - 1) * 100,
            'wilcoxon_statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
            'memory_savings_pct': memory_savings * 100,
            'avg_kl_divergence': kl_divergence,
            'quality': 'Excellent' if kl_divergence < 1e-10 else 'Very Good' if kl_divergence < 1e-8 else 'Good'
        })
    
    return pd.DataFrame(results)

def main():
    print("🔍 RQ1: NDG-S vs KDE Efficiency Analysis")
    print("=" * 60)
    
    # Load experiment results
    results_file = 'results/ndg_vs_kde/rq1/rq1_experiment_results.csv'
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return
    
    results_df = pd.read_csv(results_file)
    
    print(f"📊 EXPERIMENT OVERVIEW")
    print(f"Total experiments: {len(results_df):,}")
    print(f"Datasets: {sorted(results_df['dataset'].unique())}")
    print(f"Signal lengths: {sorted(results_df['length'].unique())}")
    print(f"Cross-validation folds: {results_df['fold'].nunique()}")
    
    # Perform statistical analysis
    print(f"\n🧮 STATISTICAL ANALYSIS")
    stats_df = analyze_performance_statistics(results_df)
    
    # Display results
    print(f"\n📋 PERFORMANCE SUMMARY")
    print("=" * 80)
    for _, row in stats_df.iterrows():
        sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['dataset']:15s} (n={row['actual_size']:6,.0f}): " +
              f"{row['speedup_factor']:5.2f}x speedup, p={row['p_value']:.4f} {sig_marker}")
    
    # Hypothesis H1 validation
    print(f"\n🎯 HYPOTHESIS H1 VALIDATION")
    print("=" * 60)
    print("H1: NDG-S achieves O(n) time complexity vs KDE's O(n²),")
    print("    resulting in significant speedup for large datasets (n > 1000)")
    
    # Test H1 for large datasets
    large_datasets = stats_df[stats_df['length'] > 1000]
    significant_large = large_datasets[large_datasets['significant']]
    
    print(f"\n📊 RESULTS FOR LARGE DATASETS (n > 1000):")
    print(f"  • Total comparisons: {len(large_datasets)}")
    print(f"  • Statistically significant: {len(significant_large)}")
    print(f"  • Average speedup: {large_datasets['speedup_factor'].mean():.2f}x")
    
    # Evidence counting
    evidence_count = 0
    total_criteria = 4
    
    if len(significant_large) > 0:
        evidence_count += 1
        print("✅ 1. Statistically significant speedup found for large datasets")
    else:
        print("❌ 1. No statistically significant speedup for large datasets")
    
    if large_datasets['speedup_factor'].mean() > 1:
        evidence_count += 1
        print("✅ 2. Average speedup > 1x for large datasets")
    else:
        print("❌ 2. Average speedup ≤ 1x for large datasets")
    
    large_samples = stats_df[stats_df['length'] >= 100000]
    if len(large_samples) > 0:
        evidence_count += 1
        print("✅ 3. 100K+ sample experiments conducted")
    else:
        print("❌ 3. No 100K+ sample experiments")
    
    excellent_quality = len(stats_df[stats_df['quality'] == 'Excellent'])
    if excellent_quality > len(stats_df) / 2:
        evidence_count += 1
        print("✅ 4. Excellent approximation quality maintained")
    else:
        print("❌ 4. Approximation quality concerns")
    
    # Final verdict
    print(f"\n🎯 FINAL VERDICT: {evidence_count}/{total_criteria} criteria met")
    if evidence_count >= 3:
        print("🌟 HYPOTHESIS H1 IS SUPPORTED")
        status = "SUPPORTED"
    elif evidence_count >= 2:
        print("⚠️  HYPOTHESIS H1 IS PARTIALLY SUPPORTED")
        status = "PARTIALLY SUPPORTED"
    else:
        print("❌ HYPOTHESIS H1 IS NOT SUPPORTED")
        status = "NOT SUPPORTED"
    
    # Export summary
    os.makedirs('results/ndg_vs_kde/thesis_exports', exist_ok=True)
    
    summary_text = f"""# RQ1 Results Summary

## Hypothesis H1 Status: {status}

## Key Findings:
- Total experiments: {len(results_df):,}
- Average speedup: {stats_df['speedup_factor'].mean():.2f}x
- Significant results: {len(stats_df[stats_df['significant']])}/{len(stats_df)}
- Large dataset speedup: {large_datasets['speedup_factor'].mean():.2f}x

## Datasets tested:
{chr(10).join([f"- {dataset}: up to {stats_df[stats_df['dataset']==dataset]['actual_size'].max():,} samples" for dataset in sorted(stats_df['dataset'].unique())])}

## Evidence Score: {evidence_count}/{total_criteria}
{status} - {'Strong evidence for computational advantages' if evidence_count >= 3 else 'Mixed evidence, needs further validation' if evidence_count >= 2 else 'Insufficient evidence for claimed advantages'}
"""
    
    with open('results/ndg_vs_kde/thesis_exports/rq1_quick_summary.md', 'w') as f:
        f.write(summary_text)
    
    # Save detailed results
    stats_df.to_csv('results/ndg_vs_kde/thesis_exports/rq1_detailed_stats.csv', index=False)
    
    print(f"\n📄 RESULTS EXPORTED")
    print("Files saved to: results/ndg_vs_kde/thesis_exports/")
    print("  • rq1_quick_summary.md")
    print("  • rq1_detailed_stats.csv")
    
    print(summary_text)

if __name__ == "__main__":
    main() 
"""
Compare and visualize RQ2 results across different label types.

This script analyzes and visualizes the discriminative power of similarity metrics
across different activity label hierarchies in the Opportunity dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_label_type_results(base_dir: str = "results/rq2_classification/label_comparison") -> Dict[str, pd.DataFrame]:
    """Load results for all label types."""
    base_path = Path(base_dir)
    results = {}
    
    if not base_path.exists():
        print(f"❌ Results directory not found: {base_path}")
        return results
    
    # Find all label type directories
    for label_dir in base_path.iterdir():
        if label_dir.is_dir():
            label_type = label_dir.name
            summary_file = label_dir / "rq2_comprehensive_summary.csv"
            
            if summary_file.exists():
                print(f"✅ Loading results for {label_type}")
                results[label_type] = pd.read_csv(summary_file)
            else:
                print(f"⚠️  No summary found for {label_type}")
    
    return results


def create_comparison_visualizations(results: Dict[str, pd.DataFrame]):
    """Create comprehensive comparison visualizations."""
    
    if not results:
        print("❌ No results to visualize")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Average performance by label type
    ax1 = plt.subplot(2, 3, 1)
    
    label_performance = []
    for label_type, df in results.items():
        avg_f1 = df['macro_f1'].mean()
        std_f1 = df['macro_f1'].std()
        label_performance.append({
            'label_type': label_type,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'n_experiments': len(df)
        })
    
    perf_df = pd.DataFrame(label_performance).sort_values('avg_f1', ascending=False)
    
    bars = ax1.bar(perf_df['label_type'], perf_df['avg_f1'], yerr=perf_df['std_f1'], capsize=5)
    ax1.set_xlabel('Label Type')
    ax1.set_ylabel('Average Macro F1')
    ax1.set_title('Average Performance by Label Type')
    ax1.set_ylim(0, 1.1)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars, perf_df['avg_f1']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    # 2. Top metrics across label types
    ax2 = plt.subplot(2, 3, 2)
    
    # Get top 10 metrics for each label type
    top_metrics_by_type = {}
    for label_type, df in results.items():
        top_metrics = df.groupby('metric_name')['macro_f1'].mean().nlargest(10)
        top_metrics_by_type[label_type] = top_metrics
    
    # Create heatmap data
    all_metrics = set()
    for metrics in top_metrics_by_type.values():
        all_metrics.update(metrics.index)
    
    heatmap_data = pd.DataFrame(index=sorted(all_metrics))
    for label_type, metrics in top_metrics_by_type.items():
        heatmap_data[label_type] = metrics
    
    # Fill NaN with 0
    heatmap_data = heatmap_data.fillna(0)
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Macro F1'}, ax=ax2)
    ax2.set_title('Top Metrics Performance Across Label Types')
    ax2.set_xlabel('Label Type')
    ax2.set_ylabel('Similarity Metric')
    
    # 3. Performance distribution by label type
    ax3 = plt.subplot(2, 3, 3)
    
    # Prepare data for box plot
    box_data = []
    box_labels = []
    for label_type, df in results.items():
        box_data.append(df['macro_f1'].values)
        box_labels.append(label_type)
    
    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(box_labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_xlabel('Label Type')
    ax3.set_ylabel('Macro F1 Distribution')
    ax3.set_title('Performance Distribution by Label Type')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Window configuration impact
    ax4 = plt.subplot(2, 3, 4)
    
    config_performance = []
    for label_type, df in results.items():
        for window_size in df['window_size'].unique():
            for overlap in df['overlap_ratio'].unique():
                config_df = df[(df['window_size'] == window_size) & 
                              (df['overlap_ratio'] == overlap)]
                if len(config_df) > 0:
                    config_performance.append({
                        'label_type': label_type,
                        'config': f'W{window_size}_O{overlap}',
                        'avg_f1': config_df['macro_f1'].mean()
                    })
    
    config_df = pd.DataFrame(config_performance)
    pivot_config = config_df.pivot(index='label_type', columns='config', values='avg_f1')
    
    sns.heatmap(pivot_config, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
    ax4.set_title('Performance by Window Configuration')
    ax4.set_xlabel('Window Configuration')
    ax4.set_ylabel('Label Type')
    
    # 5. Sample efficiency
    ax5 = plt.subplot(2, 3, 5)
    
    sample_data = []
    for label_type, df in results.items():
        if 'n_windows' in df.columns:
            avg_windows = df['n_windows'].mean()
            avg_f1 = df['macro_f1'].mean()
            sample_data.append({
                'label_type': label_type,
                'avg_windows': avg_windows,
                'avg_f1': avg_f1
            })
    
    if sample_data:
        sample_df = pd.DataFrame(sample_data)
        ax5.scatter(sample_df['avg_windows'], sample_df['avg_f1'], s=100)
        
        for idx, row in sample_df.iterrows():
            ax5.annotate(row['label_type'], 
                        (row['avg_windows'], row['avg_f1']),
                        xytext=(5, 5), textcoords='offset points')
        
        ax5.set_xlabel('Average Number of Windows')
        ax5.set_ylabel('Average Macro F1')
        ax5.set_title('Performance vs Sample Size')
        ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = "Label Type Comparison Summary\n\n"
    
    for label_type, df in results.items():
        n_configs = len(df['window_size'].unique()) * len(df['overlap_ratio'].unique())
        n_metrics = len(df['metric_name'].unique())
        avg_f1 = df['macro_f1'].mean()
        std_f1 = df['macro_f1'].std()
        best_metric = df.loc[df['macro_f1'].idxmax(), 'metric_name']
        best_f1 = df['macro_f1'].max()
        
        summary_text += f"{label_type}:\n"
        summary_text += f"  Configurations: {n_configs}\n"
        summary_text += f"  Metrics tested: {n_metrics}\n"
        summary_text += f"  Avg F1: {avg_f1:.3f} ± {std_f1:.3f}\n"
        summary_text += f"  Best: {best_metric} ({best_f1:.3f})\n\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("results/rq2_classification/label_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "label_type_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "label_type_comparison.pdf", bbox_inches='tight')
    
    print(f"\n✅ Visualizations saved to {output_dir}")
    
    return fig


def create_summary_report(results: Dict[str, pd.DataFrame]):
    """Create a markdown summary report comparing label types."""
    
    output_dir = Path("results/rq2_classification/label_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "LABEL_TYPE_COMPARISON.md", "w") as f:
        f.write("# Label Type Comparison Report\n\n")
        f.write("**Analysis**: Comparing discriminative power across Opportunity label hierarchies\n\n")
        
        f.write("## Summary Statistics\n\n")
        
        # Create summary table
        f.write("| Label Type | Avg F1 | Std F1 | Best Metric | Best F1 | Activities |\n")
        f.write("|------------|--------|--------|-------------|---------|------------|\n")
        
        for label_type, df in results.items():
            avg_f1 = df['macro_f1'].mean()
            std_f1 = df['macro_f1'].std()
            best_idx = df['macro_f1'].idxmax()
            best_metric = df.loc[best_idx, 'metric_name']
            best_f1 = df.loc[best_idx, 'macro_f1']
            n_classes = df['n_classes'].iloc[0] if 'n_classes' in df.columns else 'N/A'
            
            f.write(f"| {label_type} | {avg_f1:.3f} | {std_f1:.3f} | "
                   f"{best_metric} | {best_f1:.3f} | {n_classes} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Rank label types by performance
        label_ranking = []
        for label_type, df in results.items():
            label_ranking.append((label_type, df['macro_f1'].mean()))
        
        label_ranking.sort(key=lambda x: x[1], reverse=True)
        
        f.write("### Performance Ranking\n\n")
        for i, (label_type, avg_f1) in enumerate(label_ranking, 1):
            f.write(f"{i}. **{label_type}**: {avg_f1:.3f} average F1\n")
        
        f.write("\n### Observations\n\n")
        f.write("- Different label hierarchies show varying levels of discriminative difficulty\n")
        f.write("- Fine-grained activities (e.g., ML_Both_Arms) may be harder to distinguish\n")
        f.write("- Locomotion activities show clear sensor patterns\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("1. Consider label hierarchy when designing activity recognition systems\n")
        f.write("2. Some similarity metrics perform consistently well across hierarchies\n")
        f.write("3. Window configuration impact varies by activity granularity\n")
    
    print(f"✅ Summary report saved to {output_dir / 'LABEL_TYPE_COMPARISON.md'}")


def main():
    """Run the label type comparison analysis."""
    print("🔍 Label Type Comparison Analysis")
    print("="*60)
    
    # Load results
    results = load_label_type_results()
    
    if not results:
        print("\n❌ No results found. Please run experiments first using:")
        print("   python thesis/exp/run_all_label_types.py")
        return
    
    print(f"\n✅ Loaded results for {len(results)} label types: {list(results.keys())}")
    
    # Create visualizations
    print("\n📊 Creating comparison visualizations...")
    fig = create_comparison_visualizations(results)
    
    # Create summary report
    print("\n📝 Creating summary report...")
    create_summary_report(results)
    
    # Show plot
    plt.show()
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main() 
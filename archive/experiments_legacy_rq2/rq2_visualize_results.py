"""
RQ2 Results Visualization

This script creates visualizations for the results of the RQ2 experiment,
including metric performance comparisons and confusion matrices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_metric_performance(results_dir: Path, output_dir: Optional[Path] = None):
    """
    Create visualizations of metric performance from the results.
    
    Args:
        results_dir: Directory containing the results
        output_dir: Directory to save the visualizations (defaults to results_dir)
    """
    if output_dir is None:
        output_dir = results_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summary data
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        logger.error(f"Summary file not found: {summary_path}")
        return
    
    summary_df = pd.read_csv(summary_path)
    logger.info(f"Loaded summary data with {len(summary_df)} rows")
    
    # Create performance comparison plot
    plt.figure(figsize=(12, 6))
    
    # Sort metrics by macro F1 score
    sorted_df = summary_df.sort_values('macro_f1', ascending=False)
    
    # Plot macro F1, balanced accuracy, and accuracy
    x = range(len(sorted_df))
    width = 0.25
    
    plt.bar([i - width for i in x], sorted_df['macro_f1'], width=width, 
            label='Macro F1', color='#1f77b4')
    plt.bar(x, sorted_df['balanced_accuracy'], width=width, 
            label='Balanced Accuracy', color='#ff7f0e')
    plt.bar([i + width for i in x], sorted_df['accuracy'], width=width, 
            label='Accuracy', color='#2ca02c')
    
    plt.xlabel('Similarity Metric')
    plt.ylabel('Score')
    plt.title('Performance Comparison of Similarity Metrics')
    plt.xticks(x, sorted_df['metric_name'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = output_dir / "metric_performance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved metric performance plot to {plot_path}")
    
    # Create computation time comparison plot
    plt.figure(figsize=(12, 6))
    
    # Sort metrics by computation time
    time_df = summary_df.sort_values('computation_time', ascending=False)
    
    plt.bar(range(len(time_df)), time_df['computation_time'], color='#1f77b4')
    plt.xlabel('Similarity Metric')
    plt.ylabel('Computation Time (s)')
    plt.title('Computation Time Comparison of Similarity Metrics')
    plt.xticks(range(len(time_df)), time_df['metric_name'], rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = output_dir / "computation_time.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved computation time plot to {plot_path}")
    
    # Try to load the full results to get confusion matrices
    try:
        with open(results_dir / "rq2_all_results.pkl", "rb") as f:
            all_results = pickle.load(f)
        
        # Plot confusion matrices for the top 3 metrics
        top_metrics = sorted_df['metric_name'].head(3).tolist()
        
        for result in all_results:
            window_size = result.window_config.window_size
            overlap_ratio = result.window_config.overlap_ratio
            
            # Get label mapping if available
            label_mapping = result.metadata.get('label_mapping', {})
            
            for metric_name in top_metrics:
                if metric_name not in result.predictions:
                    continue
                
                true_labels = result.windowed_data.labels
                pred_labels = result.predictions[metric_name]
                
                # Create confusion matrix
                cm = np.zeros((len(np.unique(true_labels)), len(np.unique(true_labels))), dtype=int)
                for t, p in zip(true_labels, pred_labels):
                    cm[t][p] += 1
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=[label_mapping.get(str(i), i) for i in range(cm.shape[1])],
                           yticklabels=[label_mapping.get(str(i), i) for i in range(cm.shape[0])])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {metric_name}\nWindow={window_size}, Overlap={overlap_ratio}')
                
                # Save the plot
                plot_path = output_dir / f"cm_{metric_name}_w{window_size}_o{overlap_ratio}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved confusion matrix for {metric_name} to {plot_path}")
    
    except Exception as e:
        logger.warning(f"Could not load full results for confusion matrices: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize RQ2 experiment results")
    parser.add_argument("--results_dir", type=str, default="results/rq2_mini_test",
                       help="Directory containing the results")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save the visualizations (defaults to results_dir)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    visualize_metric_performance(results_dir, output_dir) 
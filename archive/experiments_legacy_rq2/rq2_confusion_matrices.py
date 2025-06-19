"""
RQ2 Confusion Matrix Visualization

This script loads the classification results and generates confusion matrices
for the top performing metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_confusion_matrices(results_dir: Path, output_dir: Optional[Path] = None, top_n: int = 3):
    """
    Generate confusion matrices for the top N performing metrics.
    
    Args:
        results_dir: Directory containing the classification results
        output_dir: Directory to save the visualizations (defaults to results_dir)
        top_n: Number of top metrics to visualize
    """
    if output_dir is None:
        output_dir = results_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load classification results
    results_path = results_dir / "classification_results.pkl"
    if not results_path.exists():
        logger.error(f"Classification results not found: {results_path}")
        return
    
    with open(results_path, "rb") as f:
        all_results = pickle.load(f)
    
    logger.info(f"Loaded classification results with {len(all_results)} configurations")
    
    # Load summary to get top metrics
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        logger.error(f"Summary file not found: {summary_path}")
        return
    
    summary_df = pd.read_csv(summary_path)
    
    # Get top N metrics by macro F1 score
    top_metrics = summary_df.sort_values('macro_f1', ascending=False)['metric_name'].head(top_n).tolist()
    logger.info(f"Top {top_n} metrics: {', '.join(top_metrics)}")
    
    # Generate confusion matrices
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
            
            # Get unique labels
            unique_labels = sorted(np.unique(np.concatenate([true_labels, pred_labels])))
            n_labels = len(unique_labels)
            
            # Create confusion matrix
            cm = np.zeros((n_labels, n_labels), dtype=int)
            for t, p in zip(true_labels, pred_labels):
                cm[t][p] += 1
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            
            # Create labels for the confusion matrix
            if label_mapping:
                labels = [label_mapping.get(str(i), i) for i in range(n_labels)]
            else:
                labels = [str(i) for i in range(n_labels)]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {metric_name}\nWindow={window_size}, Overlap={overlap_ratio}')
            
            # Save the plot
            plot_path = output_dir / f"cm_{metric_name}_w{window_size}_o{overlap_ratio}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix for {metric_name} to {plot_path}")
            
            # Also generate normalized confusion matrix
            plt.figure(figsize=(8, 6))
            
            # Normalize by row (true labels)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
            
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Normalized Confusion Matrix - {metric_name}\nWindow={window_size}, Overlap={overlap_ratio}')
            
            # Save the plot
            plot_path = output_dir / f"cm_norm_{metric_name}_w{window_size}_o{overlap_ratio}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved normalized confusion matrix for {metric_name} to {plot_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate confusion matrices for RQ2 results")
    parser.add_argument("--results_dir", type=str, default="results/rq2_mini_test",
                       help="Directory containing the results")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save the visualizations (defaults to results_dir)")
    parser.add_argument("--top_n", type=int, default=3,
                       help="Number of top metrics to visualize")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    generate_confusion_matrices(results_dir, output_dir, args.top_n) 
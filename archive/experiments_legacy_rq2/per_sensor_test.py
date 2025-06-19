"""
Per-Sensor Membership Function Test

This script tests the per-sensor membership function approach on a small subset
of the Opportunity dataset to evaluate its performance compared to the
traditional approach of using a single membership function for all sensors.
"""

import pytest
pytest.skip("Legacy comparison script using removed APIs", allow_module_level=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import pickle

# Import thesis modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from thesis.data import (
    create_opportunity_dataset,
    WindowConfig, create_sliding_windows,
    filter_windowed_data_by_class_count
)
from thesis.fuzzy.per_sensor_membership import (
    compute_ndg_per_sensor,
    compute_similarity_per_sensor,
    compute_pairwise_similarities_per_sensor
)
from thesis.fuzzy.similarity import similarity_jaccard
from thesis.exp.activity_classification import (
    compute_window_ndg_membership,
    classify_with_similarity_matrix
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_comparison_test(
    output_dir: Path,
    n_samples: int = 500,
    window_size: int = 64,
    overlap_ratio: float = 0.5,
    n_jobs: int = 4
):
    """
    Run a comparison test between per-sensor and traditional approaches.
    
    Args:
        output_dir: Directory to save results
        n_samples: Number of samples to use
        window_size: Window size
        overlap_ratio: Window overlap ratio
        n_jobs: Number of parallel jobs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Opportunity dataset with Locomotion labels
    logger.info("Loading Opportunity dataset (Locomotion labels)...")
    dataset = create_opportunity_dataset()
    df = dataset.df
    
    # Get Locomotion labels
    idx = pd.IndexSlice
    try:
        labels = df.loc[:, idx["Label", "Locomotion", "Label", "N/A"]].values
        logger.info("Successfully loaded Locomotion labels")
    except KeyError:
        logger.error("Could not find Locomotion labels")
        return
    
    # Filter to keep only Stand, Walk, Sit (exclude Unknown and Lie for simplicity)
    target_activities = ["Stand", "Walk", "Sit"]
    
    # Convert labels to strings to ensure they are hashable
    labels = np.array([str(label[0]) if isinstance(label, np.ndarray) else str(label) for label in labels])
    
    activity_mask = np.isin(labels, target_activities)
    
    # Get sensor data (just accelerometer data for simplicity)
    sensor_mask = df.columns.get_level_values('SensorType').isin(['Accelerometer'])
    data = df.loc[:, sensor_mask].values
    
    # Apply activity mask
    data = data[activity_mask]
    labels = labels[activity_mask]
    
    # Subsample to make it faster
    if n_samples < len(data):
        # Use stratified sampling to maintain class distribution
        unique_labels = np.unique(labels)
        indices = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            n_label_samples = int(n_samples * len(label_indices) / len(labels))
            
            # Ensure at least 10 samples per class
            n_label_samples = max(n_label_samples, 10)
            
            # Random sample from this class
            if len(label_indices) > n_label_samples:
                sampled_indices = np.random.choice(
                    label_indices, size=n_label_samples, replace=False
                )
                indices.extend(sampled_indices)
        
        # Use the sampled indices
        data = data[indices]
        labels = labels[indices]
    
    # Convert labels to integers for classification
    unique_labels = np.unique(labels)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_mapping[label] for label in labels])
    
    logger.info(f"Using {len(labels)} samples, {data.shape[1]} features")
    logger.info(f"Activities: {unique_labels}")
    
    # Count samples per class
    for label, idx in label_mapping.items():
        count = np.sum(encoded_labels == idx)
        logger.info(f"  - {label}: {count} samples")
    
    # Create windows
    window_config = WindowConfig(
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        label_strategy="majority_vote",
        min_samples_per_class=5
    )
    
    windowed_data = create_sliding_windows(
        data=data,
        labels=encoded_labels,
        config=window_config
    )
    
    # Filter windows based on class count
    windowed_data = filter_windowed_data_by_class_count(
        windowed_data=windowed_data,
        min_samples_per_class=5
    )
    
    # Log windowing stats
    n_classes = len(np.unique(windowed_data.labels))
    class_counts = np.bincount(windowed_data.labels)
    min_count = np.min(class_counts[class_counts > 0])
    max_count = np.max(class_counts)
    balance_ratio = min_count / max_count if max_count > 0 else 0
    
    logger.info(f"Windows: {windowed_data.n_windows}, Classes: {n_classes}")
    logger.info(f"Balance ratio: {balance_ratio:.3f}")
    
    # Check if we have enough classes for classification
    if n_classes < 2:
        logger.error(f"Not enough classes for classification (found {n_classes}). Try increasing n_samples.")
        return
    
    # Run traditional approach (single membership function)
    logger.info("\nRunning traditional approach (single membership function)...")
    traditional_start = time.time()
    
    # Compute membership functions for all windows
    membership_functions = []
    x_values_list = []
    
    for i in range(windowed_data.n_windows):
        window_data = windowed_data.windows[i]
        x_vals, mu_vals = compute_window_ndg_membership(
            window_data,
            kernel_type="gaussian",
            sigma_method="adaptive"  # Use adaptive sigma method instead of "std"
        )
        membership_functions.append(mu_vals)
        x_values_list.append(x_vals)
    
    # Use common x_values (assume all windows have same domain structure)
    x_values_common = x_values_list[0]
    
    # Compute similarity matrix
    traditional_sim_matrix = np.zeros((windowed_data.n_windows, windowed_data.n_windows))
    np.fill_diagonal(traditional_sim_matrix, 1.0)
    
    for i in range(windowed_data.n_windows):
        for j in range(i+1, windowed_data.n_windows):
            sim = similarity_jaccard(membership_functions[i], membership_functions[j])
            traditional_sim_matrix[i, j] = sim
            traditional_sim_matrix[j, i] = sim
    
    # Perform classification
    traditional_pred, traditional_metrics = classify_with_similarity_matrix(
        similarity_matrix=traditional_sim_matrix,
        true_labels=windowed_data.labels
    )
    
    traditional_time = time.time() - traditional_start
    
    # Run per-sensor approach
    logger.info("\nRunning per-sensor approach...")
    per_sensor_start = time.time()
    
    # Compute pairwise similarities using per-sensor approach
    per_sensor_sim_matrix = compute_pairwise_similarities_per_sensor(
        windows=windowed_data.windows,
        metric="jaccard",
        kernel_type="gaussian",
        sigma_method="std",
        n_jobs=n_jobs
    )
    
    # Perform classification
    per_sensor_pred, per_sensor_metrics = classify_with_similarity_matrix(
        similarity_matrix=per_sensor_sim_matrix,
        true_labels=windowed_data.labels
    )
    
    per_sensor_time = time.time() - per_sensor_start
    
    # Compare results
    logger.info("\n=== Results Comparison ===")
    logger.info(f"Traditional approach time: {traditional_time:.2f}s")
    logger.info(f"Per-sensor approach time: {per_sensor_time:.2f}s")
    logger.info(f"Time difference: {per_sensor_time - traditional_time:.2f}s")
    
    logger.info("\nTraditional approach metrics:")
    logger.info(f"  Accuracy: {traditional_metrics['accuracy']:.4f}")
    logger.info(f"  Balanced accuracy: {traditional_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Macro F1: {traditional_metrics['macro_f1']:.4f}")
    
    logger.info("\nPer-sensor approach metrics:")
    logger.info(f"  Accuracy: {per_sensor_metrics['accuracy']:.4f}")
    logger.info(f"  Balanced accuracy: {per_sensor_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Macro F1: {per_sensor_metrics['macro_f1']:.4f}")
    
    # Save results
    results = {
        "traditional": {
            "predictions": traditional_pred,
            "metrics": traditional_metrics,
            "time": traditional_time,
            "similarity_matrix": traditional_sim_matrix
        },
        "per_sensor": {
            "predictions": per_sensor_pred,
            "metrics": per_sensor_metrics,
            "time": per_sensor_time,
            "similarity_matrix": per_sensor_sim_matrix
        },
        "metadata": {
            "n_samples": len(data),
            "n_features": data.shape[1],
            "window_size": window_size,
            "overlap_ratio": overlap_ratio,
            "n_windows": windowed_data.n_windows,
            "n_classes": n_classes,
            "class_balance": balance_ratio,
            "label_mapping": {str(v): k for k, v in label_mapping.items()},
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }
    
    with open(output_dir / "comparison_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Create confusion matrices
    create_confusion_matrices(
        true_labels=windowed_data.labels,
        traditional_pred=traditional_pred,
        per_sensor_pred=per_sensor_pred,
        label_mapping={v: k for k, v in label_mapping.items()},
        output_dir=output_dir
    )
    
    # Create similarity matrix heatmaps
    create_similarity_heatmaps(
        traditional_sim=traditional_sim_matrix,
        per_sensor_sim=per_sensor_sim_matrix,
        output_dir=output_dir
    )
    
    logger.info(f"Results saved to {output_dir}")


def create_confusion_matrices(
    true_labels: np.ndarray,
    traditional_pred: np.ndarray,
    per_sensor_pred: np.ndarray,
    label_mapping: Dict[int, str],
    output_dir: Path
):
    """Create and save confusion matrices for both approaches."""
    # Get unique labels
    unique_labels = sorted(np.unique(np.concatenate([true_labels, traditional_pred, per_sensor_pred])))
    n_labels = len(unique_labels)
    
    # Create labels for the confusion matrix
    if label_mapping:
        labels = [label_mapping.get(i, i) for i in unique_labels]
    else:
        labels = [str(i) for i in unique_labels]
    
    # Create traditional confusion matrix
    plt.figure(figsize=(8, 6))
    cm_trad = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(true_labels, traditional_pred):
        cm_trad[t][p] += 1
    
    sns.heatmap(cm_trad, annot=True, fmt='d', cmap='Blues',
               xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Traditional Approach - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "traditional_confusion_matrix.png", dpi=300, bbox_inches='tight')
    
    # Create per-sensor confusion matrix
    plt.figure(figsize=(8, 6))
    cm_ps = np.zeros((n_labels, n_labels), dtype=int)
    for t, p in zip(true_labels, per_sensor_pred):
        cm_ps[t][p] += 1
    
    sns.heatmap(cm_ps, annot=True, fmt='d', cmap='Blues',
               xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Per-Sensor Approach - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "per_sensor_confusion_matrix.png", dpi=300, bbox_inches='tight')


def create_similarity_heatmaps(
    traditional_sim: np.ndarray,
    per_sensor_sim: np.ndarray,
    output_dir: Path
):
    """Create and save similarity matrix heatmaps for both approaches."""
    # Traditional similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(traditional_sim, cmap='viridis', vmin=0, vmax=1)
    plt.title('Traditional Approach - Similarity Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "traditional_similarity_matrix.png", dpi=300, bbox_inches='tight')
    
    # Per-sensor similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(per_sensor_sim, cmap='viridis', vmin=0, vmax=1)
    plt.title('Per-Sensor Approach - Similarity Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / "per_sensor_similarity_matrix.png", dpi=300, bbox_inches='tight')
    
    # Difference matrix
    plt.figure(figsize=(10, 8))
    diff_matrix = per_sensor_sim - traditional_sim
    sns.heatmap(diff_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Difference Matrix (Per-Sensor - Traditional)')
    plt.tight_layout()
    plt.savefig(output_dir / "difference_matrix.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test per-sensor membership function approach")
    parser.add_argument("--output_dir", type=str, default="results/per_sensor_test",
                       help="Directory to save results")
    parser.add_argument("--n_samples", type=int, default=500,
                       help="Number of samples to use")
    parser.add_argument("--window_size", type=int, default=64,
                       help="Window size")
    parser.add_argument("--overlap_ratio", type=float, default=0.5,
                       help="Window overlap ratio")
    parser.add_argument("--n_jobs", type=int, default=4,
                       help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    run_comparison_test(
        output_dir=output_dir,
        n_samples=args.n_samples,
        window_size=args.window_size,
        overlap_ratio=args.overlap_ratio,
        n_jobs=args.n_jobs
    ) 
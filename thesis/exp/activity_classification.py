"""
Activity Classification Pipeline for RQ2: Discriminative Power Assessment

This module implements the complete pipeline for evaluating discriminative power
of fuzzy similarity metrics using sensor-based activity recognition.

Key Components:
- Time series windowing with configurable overlaps
- Optimized NDG membership function computation
- 38 similarity metrics evaluation
- 1-NN classification with Leave-One-Window-Out cross-validation
- Comprehensive performance evaluation and statistical analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import LeaveOneOut
import time
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import os
import multiprocessing
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import thesis modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from thesis.data import (
    WindowConfig, WindowedData, create_sliding_windows,
    filter_windowed_data_by_class_count, get_windowing_summary, balance_windows_by_class
)
# NDG membership helpers are imported lazily in the wrapper below
from thesis.fuzzy.similarity import calculate_all_similarity_metrics
# Per-sensor logic moved to rq2_experiment.py
from thesis.core.config import BaseConfig, WindowingMixin, NDGMixin


def compute_window_ndg_membership(
    window_data: np.ndarray,
    n_grid_points: int = 100,
    kernel_type: str = "epanechnikov",
    sigma_method: str = "adaptive",
    use_per_sensor: bool = False,  # retained for backward compatibility; ignored
):
    """Deprecated wrapper kept for backward compatibility.

    All logic now lives in ``thesis.fuzzy.membership.compute_ndg_window``.
    The *use_per_sensor* flag is ignored – per-sensor calculations should call
    ``compute_ndg_window_per_sensor`` directly.
    """
    from thesis.fuzzy.membership import compute_ndg_window

    return compute_ndg_window(
        window_data,
        n_grid_points=n_grid_points,
        kernel_type=kernel_type,
        sigma_method=sigma_method,
    )


@dataclass
class ClassificationConfig(BaseConfig, WindowingMixin, NDGMixin):
    """Configuration for activity-classification experiments."""

    similarity_normalization: bool = True
    chunk_size: int = 1000  # pairs per parallel batch

    # Approach toggles --------------------------------------------------------
    use_per_sensor: bool = True
    use_all_metrics: bool = False

    # Override some defaults from mix-ins for this specific use-case ---------
    window_sizes: List[int] = field(default_factory=lambda: [128, 256])
    overlap_ratios: List[float] = field(default_factory=lambda: [0.5, 0.7])
    min_samples_per_class: int = 10

    ndg_kernel_type: str = "epanechnikov"  # faster than Gaussian for NDG


@dataclass
class ClassificationResults:
    """Container for classification experiment results."""
    window_config: WindowConfig
    windowed_data: WindowedData
    similarity_matrices: Dict[str, np.ndarray]
    predictions: Dict[str, np.ndarray]
    performance_metrics: Dict[str, Dict[str, float]]
    computation_time: Dict[str, float]
    metadata: Dict[str, Any]


def compute_similarity_for_pair(args):
    """
    Compute similarity metrics for a pair of windows.
    
    Args:
        args: Tuple containing (i, j, mu_i, mu_j, x_values_common, normalize, use_per_sensor)
        
    Returns:
        Tuple of (i, j, similarities_dict)
    """
    if len(args) == 7:  # Per-sensor approach
        i, j, mu_i, mu_j, x_values_common, normalize, use_per_sensor = args
    else:  # Traditional approach
        i, j, mu_i, mu_j, x_values_common, normalize = args
        use_per_sensor = False
    
    # Only traditional approach supported here
    # Per-sensor logic is now handled directly in rq2_experiment.py
    similarities = calculate_all_similarity_metrics(
        mu_i, mu_j, x_values_common, normalise=normalize
    )
    
    return i, j, similarities


def compute_pairwise_similarities(
    windowed_data: WindowedData,
    config: ClassificationConfig
) -> Dict[str, np.ndarray]:
    """
    Compute pairwise similarity matrices for all windows using all metrics.
    Uses parallel processing for improved performance.
    
    Args:
        windowed_data: Windowed sensor data
        config: Classification configuration
    
    Returns:
        Dictionary mapping metric names to similarity matrices
    """
    n_windows = windowed_data.n_windows
    
    # Per-sensor approach is now handled directly in rq2_experiment.py
    # This function only handles traditional approach
    
    # Traditional approach (flattened windows)
    logger.info(f"🔄 Computing NDG membership functions for {n_windows} windows...")
    
    # Compute membership functions for all windows
    membership_functions = []
    x_values_list = []
    
    for i in range(n_windows):
        window_data = windowed_data.windows[i]
        x_vals, mu_vals = compute_window_ndg_membership(
            window_data,
            kernel_type=config.ndg_kernel_type,
            sigma_method=config.ndg_sigma_method,
            use_per_sensor=False  # Force traditional approach
        )
        membership_functions.append(mu_vals)
        x_values_list.append(x_vals)
    
    # Use common x_values (assume all windows have same domain structure)
    x_values_common = x_values_list[0]
    
    logger.info(f"🔄 Computing similarity matrices using 38 metrics...")
    
    # Initialize similarity matrices
    similarity_matrices = {}
    
    # Compute first pair to get metric names
    sample_similarities = calculate_all_similarity_metrics(
        membership_functions[0], membership_functions[1], x_values_common,
        normalise=config.similarity_normalization
    )
    
    metric_names = list(sample_similarities.keys())
    logger.info(f"📊 Computing {len(metric_names)} similarity metrics")
    
    # Initialize matrices
    for metric_name in metric_names:
        similarity_matrices[metric_name] = np.zeros((n_windows, n_windows))
    
    # Set diagonal to 1.0 (self-similarity)
    for metric_name in metric_names:
        np.fill_diagonal(similarity_matrices[metric_name], 1.0)
    
    # Generate all pairs for parallel computation
    pairs = []
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            pairs.append((i, j, membership_functions[i], membership_functions[j], 
                         x_values_common, config.similarity_normalization))
    
    total_pairs = len(pairs)
    logger.info(f"   Total pairs to compute: {total_pairs}")
    
    # Determine optimal chunk size
    chunk_size = min(config.chunk_size, max(1, total_pairs // (config.n_jobs * 4)))
    
    # Process in chunks to avoid memory issues with very large datasets
    completed_pairs = 0
    
    # Use ProcessPoolExecutor for parallel computation
    with ProcessPoolExecutor(max_workers=config.n_jobs) as executor:
        # Process pairs in chunks
        for chunk_start in range(0, len(pairs), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(pairs))
            chunk = pairs[chunk_start:chunk_end]
            
            # Submit all pairs in this chunk
            futures = [executor.submit(compute_similarity_for_pair, pair) for pair in chunk]
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    i, j, similarities = future.result()
                    
                    # Store in matrices (symmetric)
                    for metric_name, sim_value in similarities.items():
                        similarity_matrices[metric_name][i, j] = sim_value
                        similarity_matrices[metric_name][j, i] = sim_value
                    
                    completed_pairs += 1
                    if completed_pairs % 100 == 0 or completed_pairs == total_pairs:
                        logger.info(f"   Progress: {completed_pairs}/{total_pairs} pairs ({100*completed_pairs/total_pairs:.1f}%)")
                except Exception as e:
                    logger.error(f"   Error processing pair: {e}")
    
    return similarity_matrices


def classify_with_similarity_matrix(
    similarity_matrix: np.ndarray,
    true_labels: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Perform 1-NN classification using similarity matrix with Leave-One-Out CV.
    
    Args:
        similarity_matrix: Precomputed similarity matrix (n_windows, n_windows)
        true_labels: True activity labels for each window
    
    Returns:
        Tuple of (predictions, performance_metrics)
    """
    n_windows = len(true_labels)
    predictions = np.zeros(n_windows, dtype=true_labels.dtype)
    
    # Leave-One-Out Cross-Validation
    for test_idx in range(n_windows):
        # Get similarities to all other windows (excluding self)
        similarities = similarity_matrix[test_idx].copy()
        similarities[test_idx] = -np.inf  # Exclude self
        
        # Find most similar window (1-NN)
        nearest_neighbor_idx = np.argmax(similarities)
        predictions[test_idx] = true_labels[nearest_neighbor_idx]
    
    # Compute performance metrics
    macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    
    # Per-class F1 scores
    per_class_f1 = f1_score(true_labels, predictions, average=None, zero_division=0)
    unique_labels = np.unique(true_labels)
    
    metrics = {
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc,
        'accuracy': np.mean(predictions == true_labels)
    }
    
    # Add per-class metrics
    for i, label in enumerate(unique_labels):
        if i < len(per_class_f1):
            metrics[f'f1_class_{label}'] = per_class_f1[i]
    
    return predictions, metrics


def run_activity_classification_experiment(
    data: np.ndarray,
    labels: np.ndarray, 
    config: ClassificationConfig,
    experiment_name: str = "activity_classification"
) -> List[ClassificationResults]:
    """
    Run complete activity classification experiment with multiple window configurations.
    
    Args:
        data: Sensor data, shape (n_samples, n_features)
        labels: Activity labels, shape (n_samples,)
        config: Experiment configuration
        experiment_name: Name for the experiment
    
    Returns:
        List of ClassificationResults for each window configuration
    """
    print(f"🚀 Starting Activity Classification Experiment: {experiment_name}")
    print(f"📊 Data shape: {data.shape}, {len(np.unique(labels))} unique activities")
    
    results = []
    
    # Test all window size and overlap combinations
    for window_size in config.window_sizes:
        for overlap_ratio in config.overlap_ratios:
            
            print(f"\n⚙️ Configuration: Window={window_size}, Overlap={overlap_ratio}")
            start_time = time.time()
            
            # Create windowing configuration
            window_config = WindowConfig(
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                label_strategy="majority_vote",
                min_samples_per_class=config.min_samples_per_class
            )
            
            try:
                # Create windows
                windowed_data = create_sliding_windows(data, labels, window_config)
                windowed_data = filter_windowed_data_by_class_count(
                    windowed_data, config.min_samples_per_class
                )
                
                # Optional class balancing -------------------------------------------------
                if config.max_windows_per_class is not None:
                    balance_start_time = time.time()
                    before = windowed_data.n_windows
                    windowed_data = balance_windows_by_class(
                        windowed_data,
                        max_windows_per_class=config.max_windows_per_class,
                        random_state=42,
                    )
                    balance_time = time.time() - balance_start_time
                    print(f"   ⚖️  Balanced windows: {before} -> {windowed_data.n_windows} ({balance_time:.2f}s)")
                
                summary = get_windowing_summary(windowed_data)
                print(f"   📈 Windows: {summary['n_windows']}, Classes: {summary['n_classes']}")
                print(f"   📈 Balance ratio: {summary['class_balance_ratio']:.3f}")
                
                # Compute similarity matrices
                sim_start_time = time.time()
                similarity_matrices = compute_pairwise_similarities(windowed_data, config)
                sim_time = time.time() - sim_start_time
                
                # Perform classification for each similarity metric
                predictions = {}
                performance_metrics = {}
                
                print(f"🔄 Running 1-NN classification with {len(similarity_matrices)} metrics...")
                
                for metric_name, sim_matrix in similarity_matrices.items():
                    pred, perf = classify_with_similarity_matrix(sim_matrix, windowed_data.labels)
                    predictions[metric_name] = pred
                    performance_metrics[metric_name] = perf
                
                total_time = time.time() - start_time
                
                # Store results
                result = ClassificationResults(
                    window_config=window_config,
                    windowed_data=windowed_data,
                    similarity_matrices=similarity_matrices,
                    predictions=predictions,
                    performance_metrics=performance_metrics,
                    computation_time={
                        'total': total_time,
                        'similarity_computation': sim_time,
                        'classification': total_time - sim_time
                    },
                    metadata={
                        'experiment_name': experiment_name,
                        'config': config,
                        'summary': summary
                    }
                )
                
                results.append(result)
                
                # Print top performing metrics
                sorted_metrics = sorted(
                    performance_metrics.items(),
                    key=lambda x: x[1]['macro_f1'],
                    reverse=True
                )
                
                print(f"   ✅ Top 5 metrics by Macro F1:")
                for i, (metric_name, perf) in enumerate(sorted_metrics[:5]):
                    print(f"      {i+1}. {metric_name}: {perf['macro_f1']:.3f}")
                
                print(f"   ⏱️  Total time: {total_time:.1f}s (similarity: {sim_time:.1f}s)")
                
            except Exception as e:
                print(f"   ❌ Configuration failed: {e}")
                continue
    
    print(f"\n🎉 Experiment completed! {len(results)} configurations successful.")
    return results


def save_classification_results(
    results: List[ClassificationResults],
    output_dir: Union[str, Path]
) -> None:
    """Save classification results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    with open(output_dir / "classification_results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Create summary CSV
    summary_data = []
    for result in results:
        config = result.window_config
        for metric_name, perf in result.performance_metrics.items():
            summary_data.append({
                'window_size': config.window_size,
                'overlap_ratio': config.overlap_ratio,
                'metric_name': metric_name,
                'macro_f1': perf['macro_f1'],
                'balanced_accuracy': perf['balanced_accuracy'],
                'accuracy': perf['accuracy'],
                'n_windows': result.windowed_data.n_windows,
                'n_classes': len(np.unique(result.windowed_data.labels)),
                'computation_time': result.computation_time['total']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "classification_summary.csv", index=False)
    
    print(f"💾 Results saved to {output_dir}")


# Demo function for testing
def demo_activity_classification():
    """Demonstrate activity classification with synthetic data."""
    print("🧪 Activity Classification Demo")
    
    # Generate synthetic multi-activity sensor data
    np.random.seed(42)
    
    # Create 3 different activity patterns
    n_samples_per_activity = 500
    n_features = 3
    
    activities_data = []
    activities_labels = []
    
    # Activity 0: Walking (periodic pattern)
    t = np.linspace(0, 10*np.pi, n_samples_per_activity)
    walking_data = np.column_stack([
        2*np.sin(t) + 0.5*np.random.randn(n_samples_per_activity),
        1.5*np.cos(t) + 0.3*np.random.randn(n_samples_per_activity),
        0.8*np.sin(2*t) + 0.2*np.random.randn(n_samples_per_activity)
    ])
    activities_data.append(walking_data)
    activities_labels.extend([0] * n_samples_per_activity)
    
    # Activity 1: Running (higher frequency)
    t = np.linspace(0, 20*np.pi, n_samples_per_activity) 
    running_data = np.column_stack([
        3*np.sin(2*t) + 0.8*np.random.randn(n_samples_per_activity),
        2.5*np.cos(2*t) + 0.6*np.random.randn(n_samples_per_activity),
        1.5*np.sin(4*t) + 0.4*np.random.randn(n_samples_per_activity)
    ])
    activities_data.append(running_data)
    activities_labels.extend([1] * n_samples_per_activity)
    
    # Activity 2: Standing (low variance)
    standing_data = np.column_stack([
        1 + 0.1*np.random.randn(n_samples_per_activity),
        -0.5 + 0.1*np.random.randn(n_samples_per_activity),
        0.2 + 0.05*np.random.randn(n_samples_per_activity)
    ])
    activities_data.append(standing_data)
    activities_labels.extend([2] * n_samples_per_activity)
    
    # Combine all activities
    data = np.vstack(activities_data)
    labels = np.array(activities_labels)
    
    # Create experiment configuration
    config = ClassificationConfig(
        window_sizes=[64, 128],
        overlap_ratios=[0.5, 0.7],
        min_samples_per_class=5
    )
    
    # Run experiment
    results = run_activity_classification_experiment(
        data, labels, config, "synthetic_demo"
    )
    
    return results


if __name__ == "__main__":
    demo_activity_classification() 
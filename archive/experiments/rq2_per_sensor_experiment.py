"""
RQ2 Experiment with Per-Sensor Membership Functions

This script runs the RQ2 experiment (activity classification) using the per-sensor
membership function approach. It evaluates the performance of the approach on the
Opportunity dataset with Locomotion labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Any
import argparse
import multiprocessing
from dataclasses import dataclass, field

# Import thesis modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from thesis.data import (
    create_opportunity_dataset,
    WindowConfig, create_sliding_windows,
    filter_windowed_data_by_class_count
)
from thesis.fuzzy.per_sensor_membership import (
    compute_pairwise_similarities_per_sensor
)
from thesis.exp.activity_classification import (
    ClassificationConfig,
    ClassificationResults,
    classify_with_similarity_matrix
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PerSensorExperimentConfig:
    """Configuration for per-sensor RQ2 experiment."""
    window_sizes: List[int] = field(default_factory=lambda: [64, 128])
    overlap_ratios: List[float] = field(default_factory=lambda: [0.5, 0.7])
    min_samples_per_class: int = 10
    similarity_metrics: List[str] = field(default_factory=lambda: ["jaccard"])
    kernel_type: str = "gaussian"
    sigma_method: str = "std"
    n_jobs: int = -1  # Parallel processing (-1 = all cores)
    
    def __post_init__(self):
        # Convert n_jobs to actual number of cores
        if self.n_jobs <= 0:
            self.n_jobs = max(1, multiprocessing.cpu_count() + self.n_jobs + 1)


def run_per_sensor_experiment(
    output_dir: Path,
    config: PerSensorExperimentConfig,
    label_type: str = "Locomotion",
    max_samples: Optional[int] = None,
    random_seed: int = 42
):
    """
    Run the RQ2 experiment with per-sensor membership functions.
    
    Args:
        output_dir: Directory to save results
        config: Experiment configuration
        label_type: Type of labels to use ('Locomotion' or 'HL_Activity')
        max_samples: Maximum number of samples to use (for testing)
        random_seed: Random seed for reproducibility
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(random_seed)
    
    # Load Opportunity dataset
    logger.info(f"Loading Opportunity dataset ({label_type} labels)...")
    dataset = create_opportunity_dataset()
    df = dataset.df
    
    # Get labels
    idx = pd.IndexSlice
    try:
        labels = df.loc[:, idx["Label", label_type, "Label", "N/A"]].values
        logger.info(f"Successfully loaded {label_type} labels")
    except KeyError:
        logger.error(f"Could not find {label_type} labels")
        return
    
    # Convert labels to strings to ensure they are hashable
    labels = np.array([str(label[0]) if isinstance(label, np.ndarray) else str(label) for label in labels])
    
    # Filter out Unknown labels
    valid_mask = labels != "Unknown"
    
    # Get sensor data (just accelerometer data for simplicity and speed)
    sensor_mask = df.columns.get_level_values('SensorType').isin(['Accelerometer'])
    data = df.loc[:, sensor_mask].values
    
    # Apply valid mask
    data = data[valid_mask]
    labels = labels[valid_mask]
    
    # Subsample if needed
    if max_samples is not None and max_samples < len(data):
        # Use stratified sampling to maintain class distribution
        unique_labels = np.unique(labels)
        indices = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            n_label_samples = int(max_samples * len(label_indices) / len(labels))
            
            # Ensure at least 10 samples per class
            n_label_samples = max(n_label_samples, 10)
            
            # Random sample from this class
            if len(label_indices) > n_label_samples:
                sampled_indices = np.random.choice(
                    label_indices, size=n_label_samples, replace=False
                )
                indices.extend(sampled_indices)
            else:
                indices.extend(label_indices)
        
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
    
    # Run experiment for each window size and overlap ratio
    results = []
    
    for window_size in config.window_sizes:
        for overlap_ratio in config.overlap_ratios:
            logger.info(f"\n=== Running with window_size={window_size}, overlap_ratio={overlap_ratio} ===")
            
            # Create windows
            window_config = WindowConfig(
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                label_strategy="majority_vote",
                min_samples_per_class=config.min_samples_per_class
            )
            
            windowed_data = create_sliding_windows(
                data=data,
                labels=encoded_labels,
                config=window_config
            )
            
            # Filter windows based on class count
            windowed_data = filter_windowed_data_by_class_count(
                windowed_data=windowed_data,
                min_samples_per_class=config.min_samples_per_class
            )
            
            # Log windowing stats
            n_classes = len(np.unique(windowed_data.labels))
            class_counts = np.bincount(windowed_data.labels)
            min_count = np.min(class_counts[class_counts > 0])
            max_count = np.max(class_counts)
            balance_ratio = min_count / max_count if max_count > 0 else 0
            
            logger.info(f"Windows: {windowed_data.n_windows}, Classes: {n_classes}")
            logger.info(f"Class distribution: {class_counts}")
            logger.info(f"Balance ratio: {balance_ratio:.3f}")
            
            # Check if we have enough classes for classification
            if n_classes < 2:
                logger.error(f"Not enough classes for classification (found {n_classes}). Try increasing max_samples.")
                continue
            
            # Run per-sensor approach for each similarity metric
            similarity_matrices = {}
            predictions = {}
            performance_metrics = {}
            computation_times = {}
            
            for metric in config.similarity_metrics:
                logger.info(f"\nRunning per-sensor approach with {metric} metric...")
                start_time = time.time()
                
                # Compute pairwise similarities using per-sensor approach
                sim_matrix = compute_pairwise_similarities_per_sensor(
                    windows=windowed_data.windows,
                    metric=metric,
                    kernel_type=config.kernel_type,
                    sigma_method=config.sigma_method,
                    n_jobs=config.n_jobs
                )
                
                # Perform classification
                pred, metrics = classify_with_similarity_matrix(
                    similarity_matrix=sim_matrix,
                    true_labels=windowed_data.labels
                )
                
                computation_time = time.time() - start_time
                
                # Store results
                similarity_matrices[metric] = sim_matrix
                predictions[metric] = pred
                performance_metrics[metric] = metrics
                computation_times[metric] = computation_time
                
                logger.info(f"Computation time: {computation_time:.2f}s")
                logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
                logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
            
            # Create result object
            result = ClassificationResults(
                window_config=window_config,
                windowed_data=windowed_data,
                similarity_matrices=similarity_matrices,
                predictions=predictions,
                performance_metrics=performance_metrics,
                computation_time=computation_times,
                metadata={
                    "label_type": label_type,
                    "label_mapping": {str(v): k for k, v in label_mapping.items()},
                    "kernel_type": config.kernel_type,
                    "sigma_method": config.sigma_method,
                    "approach": "per_sensor"
                }
            )
            
            results.append(result)
    
    # Save results
    save_experiment_results(results, output_dir)
    
    # Create summary visualizations
    create_summary_visualizations(results, output_dir)
    
    logger.info(f"\nExperiment completed. Results saved to {output_dir}")


def save_experiment_results(
    results: List[ClassificationResults],
    output_dir: Path
):
    """Save experiment results to disk."""
    # Create results directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each result separately
    for i, result in enumerate(results):
        window_size = result.window_config.window_size
        overlap_ratio = result.window_config.overlap_ratio
        
        # Create a descriptive filename
        filename = f"result_ws{window_size}_ov{overlap_ratio:.1f}.pkl"
        
        with open(output_dir / filename, "wb") as f:
            pickle.dump(result, f)
    
    # Save all results in a single file
    with open(output_dir / "all_results.pkl", "wb") as f:
        pickle.dump(results, f)


def create_summary_visualizations(
    results: List[ClassificationResults],
    output_dir: Path
):
    """Create summary visualizations of experiment results."""
    # Extract metrics for plotting
    window_sizes = []
    overlap_ratios = []
    metrics = []
    computation_times = []
    
    for result in results:
        window_size = result.window_config.window_size
        overlap_ratio = result.window_config.overlap_ratio
        
        for metric_name, metric_results in result.performance_metrics.items():
            window_sizes.append(window_size)
            overlap_ratios.append(overlap_ratio)
            metrics.append({
                "metric_name": metric_name,
                "accuracy": metric_results["accuracy"],
                "balanced_accuracy": metric_results["balanced_accuracy"],
                "macro_f1": metric_results["macro_f1"]
            })
            computation_times.append(result.computation_time[metric_name])
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        "window_size": window_sizes,
        "overlap_ratio": overlap_ratios,
        "metric_name": [m["metric_name"] for m in metrics],
        "accuracy": [m["accuracy"] for m in metrics],
        "balanced_accuracy": [m["balanced_accuracy"] for m in metrics],
        "macro_f1": [m["macro_f1"] for m in metrics],
        "computation_time": computation_times
    })
    
    # Save metrics to CSV
    df.to_csv(output_dir / "metrics_summary.csv", index=False)
    
    # Plot performance metrics
    plt.figure(figsize=(12, 8))
    
    for i, metric_name in enumerate(df["metric_name"].unique()):
        metric_df = df[df["metric_name"] == metric_name]
        
        plt.subplot(1, len(df["metric_name"].unique()), i + 1)
        
        for window_size in df["window_size"].unique():
            ws_df = metric_df[metric_df["window_size"] == window_size]
            plt.plot(ws_df["overlap_ratio"], ws_df["macro_f1"], 
                    marker='o', label=f"Window Size {window_size}")
        
        plt.title(f"{metric_name} Metric")
        plt.xlabel("Overlap Ratio")
        plt.ylabel("Macro F1 Score")
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_metrics.png", dpi=300, bbox_inches='tight')
    
    # Plot computation time
    plt.figure(figsize=(10, 6))
    
    for metric_name in df["metric_name"].unique():
        metric_df = df[df["metric_name"] == metric_name]
        
        x = np.arange(len(metric_df))
        labels = [f"WS={ws}, OV={ov:.1f}" for ws, ov in 
                 zip(metric_df["window_size"], metric_df["overlap_ratio"])]
        
        plt.bar(x, metric_df["computation_time"], label=metric_name)
        plt.xticks(x, labels, rotation=45)
    
    plt.title("Computation Time")
    plt.xlabel("Configuration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "computation_time.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RQ2 experiment with per-sensor membership functions")
    parser.add_argument("--output_dir", type=str, default="results/rq2_per_sensor",
                       help="Directory to save results")
    parser.add_argument("--label_type", type=str, default="Locomotion", choices=["Locomotion", "HL_Activity"],
                       help="Type of labels to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use (for testing)")
    parser.add_argument("--window_sizes", type=str, default="64,128",
                       help="Comma-separated list of window sizes")
    parser.add_argument("--overlap_ratios", type=str, default="0.5,0.7",
                       help="Comma-separated list of overlap ratios")
    parser.add_argument("--min_samples_per_class", type=int, default=10,
                       help="Minimum samples per class")
    parser.add_argument("--similarity_metrics", type=str, default="jaccard",
                       help="Comma-separated list of similarity metrics")
    parser.add_argument("--kernel_type", type=str, default="gaussian",
                       help="Kernel type for NDG")
    parser.add_argument("--sigma_method", type=str, default="std",
                       help="Method to compute sigma for NDG")
    parser.add_argument("--n_jobs", type=int, default=-1,
                       help="Number of parallel jobs")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Parse window sizes and overlap ratios
    window_sizes = [int(ws) for ws in args.window_sizes.split(",")]
    overlap_ratios = [float(ov) for ov in args.overlap_ratios.split(",")]
    similarity_metrics = args.similarity_metrics.split(",")
    
    # Create experiment configuration
    config = PerSensorExperimentConfig(
        window_sizes=window_sizes,
        overlap_ratios=overlap_ratios,
        min_samples_per_class=args.min_samples_per_class,
        similarity_metrics=similarity_metrics,
        kernel_type=args.kernel_type,
        sigma_method=args.sigma_method,
        n_jobs=args.n_jobs
    )
    
    # Run experiment
    run_per_sensor_experiment(
        output_dir=Path(args.output_dir),
        config=config,
        label_type=args.label_type,
        max_samples=args.max_samples,
        random_seed=args.random_seed
    ) 
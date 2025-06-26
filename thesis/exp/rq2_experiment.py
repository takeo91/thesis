"""
Unified RQ2 Experiment: Discriminative Power Assessment

This module implements a unified experiment controller for Research Question 2,
evaluating the discriminative power of fuzzy similarity metrics for sensor-based
activity classification. It supports both traditional and per-sensor approaches.

Key Features:
- Unified traditional and per-sensor approaches
- Multiple datasets (Opportunity, PAMAP2)
- Configurable label types for Opportunity dataset
- Automated experiment orchestration
- Statistical analysis integration
- Progress tracking and result persistence
- Publication-ready output generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
import time
from datetime import datetime
import json
import pickle
from dataclasses import dataclass, asdict, field
import logging
import multiprocessing
import cProfile
import pstats
from io import StringIO

# Import thesis modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from thesis.data import (
    create_opportunity_dataset, create_pamap2_dataset,
    WindowConfig, create_sliding_windows,
    filter_windowed_data_by_class_count,
    balance_windows_by_class
)
from thesis.exp.activity_classification import (
    ClassificationConfig, ClassificationResults,
    run_activity_classification_experiment, save_classification_results,
    classify_with_similarity_matrix
)
# Per-sensor logic implemented directly in this file
from thesis.fuzzy.membership import compute_membership_functions, compute_ndg_window_per_sensor
from thesis.fuzzy.similarity import (
    calculate_all_similarity_metrics, 
    calculate_vectorizable_similarity_metrics,
    get_vectorizable_metrics_list,
    compute_per_sensor_pairwise_similarities,
)
from thesis.core.config import BaseConfig, WindowingMixin, NDGMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedRQ2Config(BaseConfig, WindowingMixin, NDGMixin):
    """Unified configuration for RQ2 experiments."""
    # Dataset configuration
    datasets: List[str] = field(default_factory=lambda: ["opportunity"])
    opportunity_label_type: str = "Locomotion"  # ML_Both_Arms, Locomotion, HL_Activity, etc.
    
    # Override some window defaults for RQ2 experiments
    window_sizes: List[int] = field(default_factory=lambda: [120, 180])
    overlap_ratios: List[float] = field(default_factory=lambda: [0.5, 0.7])
    min_samples_per_class: int = 2

    # NDG
    ndg_kernel_type: str = "gaussian"
    ndg_sigma_method: str = "adaptive"
    
    # Approach configuration
    use_per_sensor: bool = True  # Use per-sensor approach by default
    similarity_metrics: List[str] = field(default_factory=lambda: ["jaccard", "dice", "cosine"])
    use_all_metrics: bool = False  # If True, use all 38 metrics (traditional approach only)
    
    # Processing configuration
    n_jobs: int = -1  # Parallel processing (-1 = all cores)
    max_samples: Optional[int] = None  # For testing with smaller datasets
    use_optimized_similarity: bool = True  # Use optimized similarity computation
    
    # Output configuration
    output_dir: str = "results/rq2_unified"
    quick_test: bool = False
    enable_checkpoints: bool = True  # Enable checkpoint/resume functionality
    checkpoint_frequency: int = 1  # Save checkpoint after every N configurations
    
    # Profiling configuration
    enable_profiling: bool = False  # Enable detailed profiling
    profile_top_n: int = 20  # Number of top functions to show in profile
    
    def __post_init__(self):
        # Convert n_jobs to actual number of cores
        if self.n_jobs <= 0:
            self.n_jobs = max(1, multiprocessing.cpu_count() + self.n_jobs + 1)
        
        # Log approach and metrics configuration
        if self.use_all_metrics:
            if self.use_per_sensor:
                vectorizable_count = len(get_vectorizable_metrics_list())
                logger.info(f"🔄 Using {vectorizable_count} vectorizable metrics with per-sensor approach")
            else:
                logger.info("🔄 Using all 38 metrics with traditional approach")
        else:
            if self.use_per_sensor:
                logger.info("🔄 Using 5 core metrics with per-sensor approach")
            else:
                logger.info("🔄 Using all 38 metrics with traditional approach (default)")
        
        # Define target activities for different label types
        self.target_activities = {
            "opportunity": {
                "ML_Both_Arms": [
                    "Open Door 1", "Open Door 2", "Close Door 1", "Close Door 2",
                    "Open Fridge", "Close Fridge", "Toggle Switch"
                ],
                "Locomotion": [
                    "Stand", "Walk", "Sit", "Lie"
                ],
                "HL_Activity": [
                    "Relaxing", "Coffee time", "Early morning", "Cleanup", 
                    "Sandwich time"
                ]
            },
            "pamap2": [
                "walking", "running", "cycling", "sitting", "standing",
                "ascending_stairs", "descending_stairs"
            ]
        }


def get_metrics_for_approach(use_per_sensor: bool, use_all_metrics: bool) -> List[str]:
    """Get appropriate metrics based on approach and user preference."""
    if use_per_sensor:
        if use_all_metrics:
            # Use all 25 vectorizable metrics for maximum performance
            return get_vectorizable_metrics_list()
        else:
            # Use core subset for quick testing
            return ["Jaccard", "Dice", "OverlapCoefficient", "Cosine", "Pearson"]
    else:
        # Traditional approach supports all metrics
        if use_all_metrics:
            # Get all metrics by running a dummy calculation
            dummy_mu = np.array([0.1, 0.5, 0.8, 0.3])
            dummy_x = np.array([0, 1, 2, 3])
            all_metrics = calculate_all_similarity_metrics(dummy_mu, dummy_mu, dummy_x)
            return list(all_metrics.keys())
        else:
            # Use core subset for quick testing
            return ["Jaccard", "Dice", "OverlapCoefficient", "Cosine", "Pearson"]


def compute_per_sensor_membership_optimized(
    window_data: np.ndarray,
    kernel_type: str = "gaussian",
    sigma_method: str = "adaptive",
    n_points: int = 100
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute per-sensor membership functions using the optimized traditional approach.
    
    This function reuses the optimized membership function computation from the traditional
    approach instead of recalculating NDG from scratch for each sensor.
    
    Args:
        window_data: Window data with shape (window_size, n_sensors)
        kernel_type: Kernel type for membership function computation
        sigma_method: Sigma calculation method
        n_points: Number of points in the membership function domain
    
    Returns:
        Tuple of (x_values, list_of_membership_functions)
        where list_of_membership_functions[i] is the membership function for sensor i
    """
    # Delegate to central implementation to avoid code duplication
    return compute_ndg_window_per_sensor(
        window_data,
        n_grid_points=n_points,
        kernel_type=kernel_type,
        sigma_method=sigma_method,
    )


def compute_per_sensor_pairwise_similarities(*args, **kwargs):
    """Deprecated shim – use thesis.fuzzy.similarity.compute_per_sensor_pairwise_similarities"""
    from thesis.fuzzy.similarity import compute_per_sensor_pairwise_similarities as _impl
    return _impl(*args, **kwargs)


class UnifiedRQ2Experiment:
    """Unified controller for RQ2 discriminative power experiments."""
    
    def __init__(self, config: UnifiedRQ2Config):
        self.config = config
        self.results = {}
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file paths
        self.checkpoint_file = self.output_dir / "checkpoint.pkl"
        self.progress_file = self.output_dir / "progress.json"
        
        # Create experiment metadata
        self.metadata = {
            "experiment_id": f"rq2_unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "config": asdict(config),
            "start_time": datetime.now().isoformat(),
            "approach": "per_sensor" if config.use_per_sensor else "traditional"
        }
        
        # Check for existing checkpoint
        if config.enable_checkpoints and self.checkpoint_file.exists():
            logger.info(f"🔄 Found existing checkpoint at {self.checkpoint_file}")
            if self.load_checkpoint():
                logger.info("✅ Successfully resumed from checkpoint")
            else:
                logger.warning("❌ Failed to load checkpoint, starting fresh")
        
        logger.info(f"🚀 Unified RQ2 Experiment initialized: {self.metadata['experiment_id']}")
        logger.info(f"📊 Approach: {'Per-Sensor' if config.use_per_sensor else 'Traditional'}")
    
    def save_checkpoint(self) -> None:
        """Save current experiment state to checkpoint file."""
        if not self.config.enable_checkpoints:
            return
            
        try:
            checkpoint_data = {
                'metadata': self.metadata,
                'results': self.results,
                'config': asdict(self.config),
                'checkpoint_time': datetime.now().isoformat()
            }
            
            # Save checkpoint atomically
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            temp_file.replace(self.checkpoint_file)
            
            # Save progress summary
            progress_data = {
                'total_datasets': len(self.config.datasets),
                'completed_datasets': len([d for d, r in self.results.items() if r]),
                'total_configurations': sum(len(r) for r in self.results.values()),
                'last_update': datetime.now().isoformat()
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
            logger.info(f"💾 Checkpoint saved: {progress_data['completed_datasets']}/{progress_data['total_datasets']} datasets, {progress_data['total_configurations']} configs")
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> bool:
        """Load experiment state from checkpoint file."""
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore state
            self.metadata = checkpoint_data['metadata']
            self.results = checkpoint_data['results']
            
            # Update metadata with resume info
            self.metadata['resumed_at'] = datetime.now().isoformat()
            self.metadata['resume_count'] = self.metadata.get('resume_count', 0) + 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def get_remaining_work(self) -> List[Tuple[str, int, float]]:
        """Get list of remaining dataset/window configurations to process."""
        remaining = []
        
        for dataset_name in self.config.datasets:
            # Check if dataset is already completed
            if dataset_name in self.results and self.results[dataset_name]:
                completed_configs = set()
                for result in self.results[dataset_name]:
                    config_key = (result.window_config.window_size, result.window_config.overlap_ratio)
                    completed_configs.add(config_key)
                
                # Check for missing configurations
                for window_size in self.config.window_sizes:
                    for overlap_ratio in self.config.overlap_ratios:
                        config_key = (window_size, overlap_ratio)
                        if config_key not in completed_configs:
                            remaining.append((dataset_name, window_size, overlap_ratio))
            else:
                # Dataset not started, add all configurations
                for window_size in self.config.window_sizes:
                    for overlap_ratio in self.config.overlap_ratios:
                        remaining.append((dataset_name, window_size, overlap_ratio))
        
        return remaining
    
    def load_opportunity_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[int, str]]:
        """Load and preprocess Opportunity dataset."""
        logger.info(f"📊 Loading Opportunity dataset ({self.config.opportunity_label_type} labels)...")
        
        dataset = create_opportunity_dataset()
        df = dataset.df
        
        # Get labels
        idx = pd.IndexSlice
        label_type = self.config.opportunity_label_type
        
        try:
            labels = df.loc[:, idx["Label", label_type, "Label", "N/A"]].values
            logger.info(f"   Successfully loaded {label_type} labels")
        except KeyError:
            logger.error(f"   Could not find {label_type} labels")
            # List available label types
            label_types = df.columns[df.columns.get_level_values('SensorType') == 'Label']
            available_types = label_types.get_level_values('BodyPart').unique()
            logger.info(f"   Available label types: {list(available_types)}")
            raise
        
        # Convert labels to strings
        labels = np.array([str(label[0]) if isinstance(label, np.ndarray) else str(label) for label in labels])
        
        # Filter out Unknown labels
        valid_mask = labels != "Unknown"
        
        # Get sensor data - use accelerometer data for speed and consistency
        sensor_mask = df.columns.get_level_values('SensorType').isin(['Accelerometer'])
        data = df.loc[:, sensor_mask].values
        
        # Apply valid mask
        data = data[valid_mask]
        labels = labels[valid_mask]
        
        # Get unique activities
        unique_labels = np.unique(labels)
        
        # Create label mapping
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        encoded_labels = np.array([label_mapping[label] for label in labels])
        
        logger.info(f"   Dataset shape: {data.shape}")
        logger.info(f"   Activities: {unique_labels}")
        for label, idx in label_mapping.items():
            count = np.sum(encoded_labels == idx)
            logger.info(f"     - {label}: {count} samples")
        
        return data, encoded_labels, unique_labels.tolist(), {v: k for k, v in label_mapping.items()}
    
    def load_pamap2_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[int, str]]:
        """Load and preprocess PAMAP2 dataset."""
        logger.info("📊 Loading PAMAP2 dataset...")
        
        dataset = create_pamap2_dataset()
        df = dataset.df
        
        # Get labels
        idx = pd.IndexSlice
        try:
            labels = df.loc[:, idx["Label", "Activity", "Name", "N/A"]].values
            logger.info("   Successfully loaded activity labels")
        except KeyError:
            logger.error("   Could not find activity labels")
            # List available label types
            available_types = df.columns.get_level_values('BodyPart').unique()
            logger.info(f"   Available label types: {list(available_types)}")
            raise
        
        # Convert labels to strings and filter out 'other'
        labels = np.array([str(label[0]) if isinstance(label, np.ndarray) else str(label) for label in labels])
        
        # Filter out 'other' and NaN labels
        valid_mask = (labels != "other") & (labels != "nan") & (labels != "None")
        
        # Get sensor data - use accelerometer data for consistency
        sensor_mask = df.columns.get_level_values('SensorType').isin(['Accelerometer'])
        data = df.loc[:, sensor_mask].values
        
        # Apply valid mask
        data = data[valid_mask]
        labels = labels[valid_mask]
        
        # Filter to target activities
        target_activities = self.config.target_activities["pamap2"]
        activity_mask = np.isin(labels, target_activities)
        
        if activity_mask.any():
            data = data[activity_mask]
            labels = labels[activity_mask]
        else:
            logger.warning(f"No target activities found. Available: {np.unique(labels)}")
        
        # Get unique activities
        unique_labels = np.unique(labels)
        
        # Create label mapping
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        encoded_labels = np.array([label_mapping[label] for label in labels])
        
        logger.info(f"   Dataset shape: {data.shape}")
        logger.info(f"   Activities: {unique_labels}")
        for label, idx in label_mapping.items():
            count = np.sum(encoded_labels == idx)
            logger.info(f"     - {label}: {count} samples")
        
        return data, encoded_labels, unique_labels.tolist(), {v: k for k, v in label_mapping.items()}
    
    def subsample_data(
        self, 
        data: np.ndarray, 
        labels: np.ndarray, 
        max_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample data using temporal-aware stratified sampling to ensure class diversity."""
        if max_samples >= len(data):
            return data, labels
        
        logger.info(f"   Subsampling from {len(data)} to {max_samples} samples...")
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # For quick tests, ensure we have enough samples per class to create diverse windows
        # We need at least 100 samples per class to create meaningful windows
        min_per_class = max(100, max_samples // (n_classes * 3))  # At least 1/3 of equal distribution
        
        logger.info(f"   Target: {min_per_class} samples per class minimum")
        
        indices = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            
            if len(label_indices) >= min_per_class:
                # Use temporal sampling: take samples from different parts of the time series
                # This helps ensure we get diverse activity patterns
                n_segments = 5  # Divide each class into 5 temporal segments
                segment_size = len(label_indices) // n_segments
                samples_per_segment = min_per_class // n_segments
                
                segment_indices = []
                for i in range(n_segments):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(label_indices)
                    segment = label_indices[start_idx:end_idx]
                    
                    if len(segment) > 0:
                        n_samples = min(samples_per_segment, len(segment))
                        if n_samples > 0:
                            sampled = np.random.choice(segment, size=n_samples, replace=False)
                            segment_indices.extend(sampled)
                
                # Add any remaining samples needed
                remaining_needed = min_per_class - len(segment_indices)
                if remaining_needed > 0:
                    unused_indices = [idx for idx in label_indices if idx not in segment_indices]
                    if len(unused_indices) >= remaining_needed:
                        additional = np.random.choice(unused_indices, size=remaining_needed, replace=False)
                        segment_indices.extend(additional)
                
                indices.extend(segment_indices)
                logger.info(f"     Class {label}: sampled {len(segment_indices)} from {len(label_indices)} available")
            else:
                # Use all available samples for this class
                indices.extend(label_indices)
                logger.warning(f"     Class {label}: only {len(label_indices)} samples available (< {min_per_class})")
        
        # Sort indices to maintain temporal order (important for windowing)
        indices.sort()
        
        sampled_data = data[indices]
        sampled_labels = labels[indices]
        
        # Log final distribution
        unique_sampled, counts = np.unique(sampled_labels, return_counts=True)
        logger.info(f"   Final sample distribution:")
        for label, count in zip(unique_sampled, counts):
            logger.info(f"     Class {label}: {count} samples")
        
        return sampled_data, sampled_labels
    
    def run_windowing_experiment(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        overlap_ratio: float
    ) -> Optional[ClassificationResults]:
        """Run experiment for a specific window configuration."""
        logger.info(f"   Window size: {window_size}, Overlap: {overlap_ratio}")
        
        # Create windows
        window_config = WindowConfig(
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            label_strategy="majority_vote",
            min_samples_per_class=self.config.min_samples_per_class
        )
        
        windowed_data = self.profile_section(
            "Windowing",
            create_sliding_windows,
            data=data,
            labels=labels,
            config=window_config
        )
        
        # Filter windows based on class count
        windowed_data = self.profile_section(
            "Filtering",
            filter_windowed_data_by_class_count,
            windowed_data=windowed_data,
            min_samples_per_class=self.config.min_samples_per_class
        )

        # Optional balancing to cap max windows per class
        if self.config.max_windows_per_class is not None:
            before_windows = windowed_data.n_windows
            windowed_data = self.profile_section(
                "Balancing",
                balance_windows_by_class,
                windowed_data=windowed_data,
                max_windows_per_class=self.config.max_windows_per_class,
                random_state=42  # deterministic for reproducibility
            )
            logger.info(f"     Balanced windows: {before_windows} -> {windowed_data.n_windows}")
        
        # Log windowing stats
        n_classes = len(np.unique(windowed_data.labels))
        class_counts = np.bincount(windowed_data.labels)
        min_count = np.min(class_counts[class_counts > 0])
        max_count = np.max(class_counts)
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        logger.info(f"     Windows: {windowed_data.n_windows}, Classes: {n_classes}")
        logger.info(f"     Class distribution: {class_counts}")
        logger.info(f"     Balance ratio: {balance_ratio:.3f}")
        
        # Check if we have enough classes for classification
        if n_classes < 2:
            logger.warning(f"     Not enough classes for classification (found {n_classes})")
            return None
        
        start_time = time.time()
        
        if self.config.use_per_sensor:
            # Use optimized per-sensor approach
            if self.config.use_all_metrics:
                # Get vectorizable metrics for per-sensor approach
                metrics_to_use = get_metrics_for_approach(
                    use_per_sensor=self.config.use_per_sensor,
                    use_all_metrics=self.config.use_all_metrics
                )
                logger.info(f"     Computing {len(metrics_to_use)} vectorizable metrics using optimized per-sensor approach...")
                
                # Compute all similarity matrices using optimized per-sensor approach  
                similarity_matrices = self.profile_section(
                    "Per-sensor ALL vectorizable metrics computation",
                    compute_per_sensor_pairwise_similarities,
                    windows_query=windowed_data.windows,
                    metrics=metrics_to_use,
                    kernel_type=self.config.ndg_kernel_type,
                    sigma_method=self.config.ndg_sigma_method,
                    normalise=True,
                    n_jobs=self.config.n_jobs
                )
                
                # Perform classification for all metrics
                predictions = {}
                performance_metrics = {}
                
                for metric_name, sim_matrix in similarity_matrices.items():
                    pred, perf = classify_with_similarity_matrix(
                        similarity_matrix=sim_matrix,
                        true_labels=windowed_data.labels
                    )
                    predictions[metric_name] = pred
                    performance_metrics[metric_name] = perf
            else:
                # Get metrics to use based on approach
                metrics_to_use = get_metrics_for_approach(
                    use_per_sensor=self.config.use_per_sensor,
                    use_all_metrics=self.config.use_all_metrics
                )
                logger.info(f"     Computing {len(metrics_to_use)} metrics using optimized per-sensor approach...")
                
                # Compute similarity matrices using optimized per-sensor approach
                similarity_matrices = self.profile_section(
                    "Per-sensor subset metrics computation",
                    compute_per_sensor_pairwise_similarities,
                    windows_query=windowed_data.windows,
                    metrics=metrics_to_use,
                    kernel_type=self.config.ndg_kernel_type,
                    sigma_method=self.config.ndg_sigma_method,
                    normalise=True,
                    n_jobs=self.config.n_jobs
                )
                
                # Perform classification for all computed metrics
                predictions = {}
                performance_metrics = {}
                
                for metric_name, sim_matrix in similarity_matrices.items():
                    pred, perf = classify_with_similarity_matrix(
                        similarity_matrix=sim_matrix,
                        true_labels=windowed_data.labels
                    )
                    predictions[metric_name] = pred
                    performance_metrics[metric_name] = perf
            
            computation_time = time.time() - start_time
            
            # Create result object
            result = ClassificationResults(
                window_config=window_config,
                windowed_data=windowed_data,
                similarity_matrices=similarity_matrices,
                predictions=predictions,
                performance_metrics=performance_metrics,
                computation_time={
                    'total': computation_time,
                    'similarity_computation': computation_time,
                    'classification': 0
                },
                metadata={
                    'approach': 'per_sensor',
                    'config': self.config
                }
            )
        else:
            # Use traditional approach
            classification_config = ClassificationConfig(
                window_sizes=[window_size],
                overlap_ratios=[overlap_ratio],
                min_samples_per_class=self.config.min_samples_per_class,
                ndg_kernel_type=self.config.ndg_kernel_type,
                ndg_sigma_method=self.config.ndg_sigma_method,
                n_jobs=self.config.n_jobs,
                use_per_sensor=False,
                use_all_metrics=self.config.use_all_metrics
            )
            
            results = run_activity_classification_experiment(
                data=data,
                labels=labels,
                config=classification_config,
                experiment_name="rq2_traditional"
            )
            
            result = results[0] if results else None
        
        if result:
            logger.info(f"     Computation time: {result.computation_time['total']:.2f}s")
            
            # Log top performing metrics
            sorted_metrics = sorted(
                result.performance_metrics.items(),
                key=lambda x: x[1]['macro_f1'],
                reverse=True
            )
            
            logger.info(f"     Top metrics by Macro F1:")
            for i, (metric_name, perf) in enumerate(sorted_metrics[:3]):
                logger.info(f"       {i+1}. {metric_name}: {perf['macro_f1']:.3f}")
        
        return result
    
    def run_dataset_experiments(self, dataset_name: str) -> List[ClassificationResults]:
        """Run experiments for a specific dataset with checkpoint support."""
        logger.info(f"\n🔬 Running experiments for {dataset_name} dataset")
        
        # Initialize results for this dataset if not exists
        if dataset_name not in self.results:
            self.results[dataset_name] = []
        
        # Load dataset once
        if dataset_name.lower() == "opportunity":
            data, labels, activity_names, label_mapping = self.load_opportunity_dataset()
        elif dataset_name.lower() == "pamap2":
            data, labels, activity_names, label_mapping = self.load_pamap2_dataset()
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not yet implemented")
        
        # Subsample if requested
        if self.config.max_samples is not None:
            data, labels = self.subsample_data(data, labels, self.config.max_samples)
        
        # Get completed configurations for this dataset
        completed_configs = set()
        for result in self.results[dataset_name]:
            config_key = (result.window_config.window_size, result.window_config.overlap_ratio)
            completed_configs.add(config_key)
        
        # Run experiments for remaining window configurations
        total_configs = len(self.config.window_sizes) * len(self.config.overlap_ratios)
        completed_count = len(completed_configs)
        config_count = completed_count
        
        if completed_count > 0:
            logger.info(f"📋 Resuming: {completed_count}/{total_configs} configurations already completed")
        
        for window_size in self.config.window_sizes:
            for overlap_ratio in self.config.overlap_ratios:
                config_key = (window_size, overlap_ratio)
                
                # Skip if already completed
                if config_key in completed_configs:
                    continue
                
                config_count += 1
                logger.info(f"\n📐 Configuration {config_count}/{total_configs}")
                
                result = self.run_windowing_experiment(
                    data=data,
                    labels=labels,
                    window_size=window_size,
                    overlap_ratio=overlap_ratio
                )
                
                if result:
                    self.results[dataset_name].append(result)
                    
                    # Save checkpoint after each configuration
                    if config_count % self.config.checkpoint_frequency == 0:
                        self.save_checkpoint()
        
        return self.results[dataset_name]
    
    def run_all_experiments(self) -> None:
        """Run experiments for all configured datasets with checkpoint support."""
        logger.info(f"\n🚀 Starting unified RQ2 experiments")
        logger.info(f"📋 Datasets: {self.config.datasets}")
        logger.info(f"🔧 Approach: {'Per-Sensor' if self.config.use_per_sensor else 'Traditional'}")
        
        # Start profiling if enabled
        profiler = self.start_profiling()
        
        # Check remaining work
        remaining_work = self.get_remaining_work()
        total_work = len(self.config.datasets) * len(self.config.window_sizes) * len(self.config.overlap_ratios)
        completed_work = total_work - len(remaining_work)
        
        if completed_work > 0:
            logger.info(f"📋 Resuming from checkpoint: {completed_work}/{total_work} configurations already completed")
        
        try:
            for dataset_name in self.config.datasets:
                try:
                    dataset_results = self.profile_section(
                        f"Dataset {dataset_name}",
                        self.run_dataset_experiments,
                        dataset_name
                    )
                    logger.info(f"✅ Completed {dataset_name}: {len(dataset_results)} configurations")
                except KeyboardInterrupt:
                    logger.warning(f"⚠️ Interrupted during {dataset_name} - saving checkpoint")
                    self.save_checkpoint()
                    self.stop_profiling(profiler)
                    raise
                except Exception as e:
                    logger.error(f"❌ Failed {dataset_name}: {e}")
                    if dataset_name not in self.results:
                        self.results[dataset_name] = []
                    # Save checkpoint even after failure
                    self.save_checkpoint()
            
            # Final checkpoint and completion
            self.metadata["end_time"] = datetime.now().isoformat()
            self.metadata["total_configurations"] = sum(len(results) for results in self.results.values())
            self.save_checkpoint()
            
            logger.info(f"\n🎉 All experiments completed!")
            logger.info(f"📊 Total configurations: {self.metadata['total_configurations']}")
            
            # Stop profiling and generate report
            self.stop_profiling(profiler)
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️ Experiment interrupted by user")
            logger.info("💾 Final checkpoint saved - you can resume later")
            self.stop_profiling(profiler)
            raise
    
    def save_results(self) -> None:
        """Save all results to disk."""
        logger.info(f"💾 Saving results to {self.output_dir}")
        
        # Save raw results
        with open(self.output_dir / "all_results.pkl", "wb") as f:
            pickle.dump(self.results, f)
        
        # Save metadata
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Create summary CSV
        summary_data = []
        for dataset_name, dataset_results in self.results.items():
            for result in dataset_results:
                config = result.window_config
                for metric_name, perf in result.performance_metrics.items():
                    summary_data.append({
                        'dataset': dataset_name,
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
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / "summary.csv", index=False)
            logger.info(f"📋 Summary saved: {len(summary_data)} rows")

    def get_checkpoint_status(self) -> Dict:
        """Get detailed checkpoint status information."""
        if not self.checkpoint_file.exists():
            return {
                'has_checkpoint': False,
                'message': 'No checkpoint found'
            }
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Calculate progress
            total_work = len(self.config.datasets) * len(self.config.window_sizes) * len(self.config.overlap_ratios)
            completed_work = sum(len(results) for results in checkpoint_data['results'].values())
            
            status = {
                'has_checkpoint': True,
                'experiment_id': checkpoint_data['metadata'].get('experiment_id', 'unknown'),
                'start_time': checkpoint_data['metadata'].get('start_time'),
                'checkpoint_time': checkpoint_data.get('checkpoint_time'),
                'approach': checkpoint_data['metadata'].get('approach'),
                'total_configurations': total_work,
                'completed_configurations': completed_work,
                'progress_percent': (completed_work / total_work * 100) if total_work > 0 else 0,
                'datasets': {}
            }
            
            # Per-dataset progress
            for dataset_name in self.config.datasets:
                dataset_total = len(self.config.window_sizes) * len(self.config.overlap_ratios)
                dataset_completed = len(checkpoint_data['results'].get(dataset_name, []))
                status['datasets'][dataset_name] = {
                    'completed': dataset_completed,
                    'total': dataset_total,
                    'progress_percent': (dataset_completed / dataset_total * 100) if dataset_total > 0 else 0
                }
            
            return status
            
        except Exception as e:
            return {
                'has_checkpoint': True,
                'error': f'Failed to read checkpoint: {e}'
            }
    
    def print_checkpoint_status(self) -> None:
        """Print detailed checkpoint status to console."""
        status = self.get_checkpoint_status()
        
        if not status['has_checkpoint']:
            print("📋 No checkpoint found - fresh start")
            return
        
        if 'error' in status:
            print(f"❌ Checkpoint error: {status['error']}")
            return
        
        print(f"\n📋 Checkpoint Status")
        print(f"🆔 Experiment ID: {status['experiment_id']}")
        print(f"🔧 Approach: {status['approach']}")
        print(f"⏰ Started: {status['start_time']}")
        print(f"💾 Last checkpoint: {status['checkpoint_time']}")
        print(f"📊 Overall progress: {status['completed_configurations']}/{status['total_configurations']} ({status['progress_percent']:.1f}%)")
        
        print(f"\n📈 Dataset Progress:")
        for dataset_name, dataset_info in status['datasets'].items():
            print(f"  {dataset_name}: {dataset_info['completed']}/{dataset_info['total']} ({dataset_info['progress_percent']:.1f}%)")
        
        remaining = status['total_configurations'] - status['completed_configurations']
        if remaining > 0:
            print(f"\n⏳ Remaining: {remaining} configurations")
        else:
            print(f"\n✅ Experiment completed!")
        print()
    
    def start_profiling(self) -> Optional[cProfile.Profile]:
        """Start profiling if enabled."""
        if not self.config.enable_profiling:
            return None
        
        logger.info("📊 Starting performance profiling...")
        profiler = cProfile.Profile()
        profiler.enable()
        return profiler
    
    def stop_profiling(self, profiler: Optional[cProfile.Profile]) -> None:
        """Stop profiling and save results."""
        if not profiler or not self.config.enable_profiling:
            return
        
        profiler.disable()
        
        # Save detailed profile
        profile_file = self.output_dir / "performance_profile.prof"
        profiler.dump_stats(str(profile_file))
        logger.info(f"📊 Detailed profile saved to {profile_file}")
        
        # Generate human-readable report
        self.generate_profile_report(profiler)
    
    def generate_profile_report(self, profiler: cProfile.Profile) -> None:
        """Generate and save human-readable profiling report."""
        try:
            # Create string buffer for profile output
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s)
            
            # Sort by cumulative time and get top functions
            ps.sort_stats('cumulative')
            ps.print_stats(self.config.profile_top_n)
            
            profile_text = s.getvalue()
            
            # Save to file
            profile_report_file = self.output_dir / "performance_report.txt"
            with open(profile_report_file, 'w') as f:
                f.write("# Performance Profile Report\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Experiment: {self.metadata.get('experiment_id', 'unknown')}\n")
                f.write(f"# Approach: {self.metadata.get('approach', 'unknown')}\n\n")
                f.write(profile_text)
            
            logger.info(f"📊 Profile report saved to {profile_report_file}")
            
            # Print summary to console
            self.print_profile_summary(ps)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate profile report: {e}")
    
    def print_profile_summary(self, ps: pstats.Stats) -> None:
        """Print a concise profile summary to console."""
        try:
            logger.info("\n📊 Performance Profile Summary:")
            logger.info("=" * 60)
            
            # Get top 10 functions by cumulative time
            s = StringIO()
            ps.print_stats(10)
            lines = s.getvalue().split('\n')
            
            # Find the header and data lines
            for i, line in enumerate(lines):
                if 'cumulative' in line and 'filename:lineno(function)' in line:
                    logger.info(line)
                    # Print next few lines with data
                    for j in range(i+1, min(i+11, len(lines))):
                        if lines[j].strip():
                            logger.info(lines[j])
                    break
                    
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Failed to print profile summary: {e}")
    
    def profile_section(self, section_name: str, func, *args, **kwargs):
        """Profile a specific section of code."""
        if not self.config.enable_profiling:
            return func(*args, **kwargs)
        
        start_time = time.time()
        logger.info(f"⏱️  Starting {section_name}...")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"⏱️  {section_name} completed in {duration:.2f}s")
        
        return result


def main():
    """Main experiment runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run unified RQ2 experiments")
    parser.add_argument("--approach", choices=["traditional", "per_sensor"], 
                       default="per_sensor", help="Approach to use")
    parser.add_argument("--datasets", nargs="+", default=["opportunity"],
                       choices=["opportunity", "pamap2"], help="Datasets to use")
    parser.add_argument("--label_type", default="Locomotion", 
                       help="Opportunity label type")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples for testing")
    parser.add_argument("--output_dir", default="results/rq2_unified",
                       help="Output directory")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run quick test with small configuration")
    parser.add_argument("--use_all_metrics", action="store_true",
                       help="Use all 38 similarity metrics")
    parser.add_argument("--status", action="store_true",
                       help="Check checkpoint status and exit")
    parser.add_argument("--disable_checkpoints", action="store_true",
                       help="Disable checkpoint functionality")
    parser.add_argument("--profile", action="store_true",
                       help="Enable detailed performance profiling")
    parser.add_argument("--profile_top_n", type=int, default=20,
                       help="Number of top functions to show in profile")
    parser.add_argument("--max_windows_per_class", type=int, default=None,
                       help="Maximum windows to retain per class after windowing")
    parser.add_argument("--library_per_class", type=int, default=None,
                       help="Number of windows per class to include in the retrieval *library* split (overrides --max_windows_per_class during splitting)")
    parser.add_argument("--topk", type=int, default=5,
                       help="k value for hit@k retrieval metric calculations")
    
    args = parser.parse_args()
    
    # Handle status check
    if args.status:
        # Create minimal config just to check status
        config = UnifiedRQ2Config(
            output_dir=args.output_dir,
            enable_checkpoints=True
        )
        experiment = UnifiedRQ2Experiment(config)
        experiment.print_checkpoint_status()
        return
    
    # Create configuration
    if args.quick_test:
        config = UnifiedRQ2Config(
            datasets=args.datasets,
            window_sizes=[120],
            overlap_ratios=[0.5],
            similarity_metrics=["jaccard"],
            max_samples=args.max_samples or 5000,
            max_windows_per_class=args.max_windows_per_class,
            use_per_sensor=(args.approach == "per_sensor"),
            use_all_metrics=args.use_all_metrics,
            opportunity_label_type=args.label_type,
            output_dir=args.output_dir,
            enable_checkpoints=not args.disable_checkpoints,
            enable_profiling=args.profile,
            profile_top_n=args.profile_top_n
        )
    else:
        config = UnifiedRQ2Config(
            datasets=args.datasets,
            window_sizes=[120, 180],
            overlap_ratios=[0.5, 0.7],
            use_per_sensor=(args.approach == "per_sensor"),
            use_all_metrics=args.use_all_metrics,
            opportunity_label_type=args.label_type,
            max_samples=args.max_samples,
            max_windows_per_class=args.max_windows_per_class,
            output_dir=args.output_dir,
            enable_checkpoints=not args.disable_checkpoints,
            enable_profiling=args.profile,
            profile_top_n=args.profile_top_n
        )
    
    # Run experiment
    experiment = UnifiedRQ2Experiment(config)
    experiment.run_all_experiments()
    experiment.save_results()
    
    logger.info("🎯 Experiment completed successfully!")


if __name__ == "__main__":
    main() 
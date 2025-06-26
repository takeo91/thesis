#!/usr/bin/env python3
"""
Unified Windowing RQ2 Experiment

Implements the "standard windows" approach where:
1. Windows are created from the full dataset first
2. Different label types are applied to the same set of windows
3. Membership functions can be cached and reused across label types
4. Massive speedup for multi-label experiments

This addresses the optimization opportunity identified where membership functions
were redundantly computed for the same windows across different label types.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

from thesis.data import (
    create_opportunity_dataset, 
    create_sliding_windows, 
    WindowConfig,
    WindowMembershipCache,
    balance_windows_by_class,
    train_test_split_windows
)
from thesis.fuzzy.membership import compute_ndg_window_per_sensor
from thesis.fuzzy.similarity import compute_per_sensor_pairwise_similarities
from thesis.exp.retrieval_utils import compute_retrieval_metrics
# Use direct values instead of constants
DEFAULT_WINDOW_SIZE = 120
DEFAULT_OVERLAP_RATIO = 0.5
from thesis.core.validation import validate_arrays
from thesis.core.exceptions import DataValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedWindowingExperiment:
    """
    Unified windowing experiment that creates standard windows across all label types
    and caches membership functions for maximum efficiency.
    """
    
    def __init__(self, window_config: Optional[WindowConfig] = None, 
                 cache_dir: str = "cache/unified_windows"):
        self.window_config = window_config or WindowConfig(
            window_size=DEFAULT_WINDOW_SIZE,
            overlap_ratio=DEFAULT_OVERLAP_RATIO
        )
        self.cache = WindowMembershipCache(cache_dir)
        self.standard_windows = None
        self.standard_labels = None
        self.standard_timestamps = None
        
    def create_standard_windows(self, dataset_name: str = "opportunity") -> Dict[str, Any]:
        """
        Create standard windows from the full dataset that can be shared across 
        all label types.
        
        Returns:
            Dictionary with windows, labels, and metadata
        """
        logger.info(f"🪟 Creating standard windows for {dataset_name}")
        
        if dataset_name == "opportunity":
            return self._create_opportunity_standard_windows()
        else:
            raise ValueError(f"Dataset {dataset_name} not supported yet")
    
    def _create_opportunity_standard_windows(self) -> Dict[str, Any]:
        """Create standard windows for Opportunity dataset."""
        # Load the full dataset
        dataset = create_opportunity_dataset()
        df = dataset.df
        
        # Get sensor data (accelerometer for consistency)
        sensor_mask = df.columns.get_level_values('SensorType').isin(['Accelerometer'])
        sensor_data = df.loc[:, sensor_mask].values
        
        # Get all label types we'll use
        label_types = ["Locomotion", "ML_Both_Arms", "HL_Activity"]
        all_labels = {}
        
        idx = pd.IndexSlice
        for label_type in label_types:
            try:
                labels = df.loc[:, idx["Label", label_type, "Label", "N/A"]].values
                labels = np.array([str(label[0]) if isinstance(label, np.ndarray) else str(label) for label in labels])
                all_labels[label_type] = labels
                logger.info(f"   Loaded {label_type}: {len(np.unique(labels[labels != 'Unknown']))} unique activities")
            except KeyError:
                logger.warning(f"   Could not find {label_type} labels")
                continue
        
        # Create windows from full dataset
        logger.info(f"📦 Creating windows (size={self.window_config.window_size}, overlap={self.window_config.overlap_ratio})")
        
        # Create sliding windows manually (without labels for standard windows)
        windows = []
        step_size = max(1, int(self.window_config.window_size * (1 - self.window_config.overlap_ratio)))
        
        for start_idx in range(0, len(sensor_data) - self.window_config.window_size + 1, step_size):
            end_idx = start_idx + self.window_config.window_size
            window_data = sensor_data[start_idx:end_idx]
            windows.append(window_data)
        
        logger.info(f"   Created {len(windows)} standard windows")
        
        # Store standard windows and associated data
        self.standard_windows = windows
        self.standard_labels = all_labels
        
        # Calculate window timestamps (center of each window)
        window_timestamps = []
        for i, window in enumerate(windows):
            start_idx = i * int(self.window_config.window_size * (1 - self.window_config.overlap_ratio))
            center_idx = start_idx + self.window_config.window_size // 2
            window_timestamps.append(center_idx)
        
        self.standard_timestamps = np.array(window_timestamps)
        
        return {
            "windows": windows,
            "labels": all_labels,
            "timestamps": window_timestamps,
            "num_windows": len(windows),
            "sensor_data_shape": sensor_data.shape
        }
    
    def get_label_filtered_windows(self, label_type: str) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
        """
        Get windows filtered for a specific label type, removing 'Unknown' labels.
        Uses majority vote labeling for robustness (same as existing experiments).
        
        Returns:
            (filtered_windows, filtered_labels, activity_names)
        """
        if self.standard_windows is None or self.standard_labels is None:
            raise ValueError("Standard windows not created. Call create_standard_windows() first.")
        
        if label_type not in self.standard_labels:
            raise ValueError(f"Label type {label_type} not available")
        
        # Get labels for this label type
        labels = self.standard_labels[label_type]
        
        # Apply majority vote labeling to each window
        filtered_windows = []
        filtered_labels = []
        
        for i, (window_data, timestamp) in enumerate(zip(self.standard_windows, self.standard_timestamps)):
            # Calculate window time range
            step_size = int(self.window_config.window_size * (1 - self.window_config.overlap_ratio))
            start_idx = i * step_size
            end_idx = start_idx + self.window_config.window_size
            
            # Extract label sequence for this window
            if end_idx <= len(labels):
                label_sequence = labels[start_idx:end_idx]
                
                # Apply majority vote (same as existing windowing logic)
                window_label = self._assign_majority_vote_label(label_sequence)
                
                # Only keep windows with valid, non-Unknown labels
                if window_label is not None and window_label != "Unknown":
                    filtered_windows.append(window_data)
                    filtered_labels.append(window_label)
        
        filtered_labels = np.array(filtered_labels)
        
        # Get unique activity names
        activity_names = list(np.unique(filtered_labels))
        if "Unknown" in activity_names:
            activity_names.remove("Unknown")
        
        logger.info(f"📊 {label_type}: {len(filtered_windows)} windows, {len(activity_names)} activities (majority vote)")
        
        return filtered_windows, filtered_labels, activity_names
    
    def _assign_majority_vote_label(self, label_sequence):
        """
        Assign window label using majority vote strategy.
        Returns None for ambiguous windows (ties) or empty sequences.
        """
        from collections import Counter
        
        # Handle empty sequence
        if len(label_sequence) == 0:
            return None
            
        # Convert to strings for consistency
        label_sequence = [str(label) for label in label_sequence]
        
        # Count occurrences of each label
        label_counts = Counter(label_sequence)
        most_common = label_counts.most_common(2)
        
        # Handle empty counter (shouldn't happen after empty check, but be safe)
        if len(most_common) == 0:
            return None
        
        # Handle single label case
        if len(most_common) == 1:
            return most_common[0][0]
        
        # Handle majority case (no tie)
        elif most_common[0][1] > most_common[1][1]:
            return most_common[0][0]
        
        # Handle tie case - return None (ambiguous window)
        else:
            return None
    
    def compute_cached_membership_functions(self, windows: List[np.ndarray]) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        """
        Compute membership functions with caching for efficiency.
        
        Returns:
            (x_values, list_of_membership_functions_per_window)
        """
        logger.info(f"⚡ Computing membership functions with caching for {len(windows)} windows")
        
        all_membership_functions = []
        x_values = None
        cache_hits = 0
        
        for i, window_data in enumerate(windows):
            if i % 50 == 0:
                logger.info(f"   Progress: {i}/{len(windows)} ({100*i/len(windows):.1f}%)")
            
            # Try to get from cache first
            cached_result = self.cache.get_membership(window_data, self.window_config)
            
            if cached_result is not None:
                cached_x_values, cached_membership = cached_result
                if x_values is None:
                    x_values = cached_x_values
                all_membership_functions.append(cached_membership)
                cache_hits += 1
            else:
                # Compute membership functions
                try:
                    computed_x_values, membership_functions = compute_ndg_window_per_sensor(
                        window_data,
                        kernel_type="epanechnikov",  # Fast kernel
                        sigma_method="adaptive"
                    )
                    
                    if x_values is None:
                        x_values = computed_x_values
                    
                    all_membership_functions.append(membership_functions)
                    
                    # Cache the result
                    self.cache.set_membership(window_data, self.window_config, 
                                            computed_x_values, membership_functions)
                    
                except Exception as e:
                    logger.warning(f"Failed to compute membership for window {i}: {e}")
                    # Use empty membership functions as fallback
                    if x_values is not None:
                        empty_membership = [np.zeros_like(x_values) for _ in range(window_data.shape[1])]
                        all_membership_functions.append(empty_membership)
                    continue
        
        cache_hit_rate = cache_hits / len(windows) if windows else 0
        logger.info(f"📈 Cache hit rate: {cache_hit_rate:.2%} ({cache_hits}/{len(windows)} windows)")
        
        # Save cache after computation
        self.cache.save()
        
        return x_values, all_membership_functions
    
    def run_label_type_experiment(self, label_type: str, metrics: List[str] = None) -> Dict[str, Any]:
        """
        Run experiment for a specific label type using standard windows and cached 
        membership functions.
        """
        if metrics is None:
            # Expanded metric set with new high-performance metrics
            metrics = [
                "jaccard", "cosine", "dice", "pearson", "overlap_coefficient",
                "spearman", "kendall_tau", "tversky", "weighted_jaccard", "mahalanobis",
                "jensen_shannon", "bhattacharyya_coefficient", "hellinger"
            ]
        
        logger.info(f"🧪 Running experiment for {label_type}")
        
        # Get filtered windows for this label type
        windows, labels, activity_names = self.get_label_filtered_windows(label_type)
        
        if len(windows) == 0:
            logger.error(f"No valid windows found for {label_type}")
            return {"error": "No valid windows"}
        
        # Compute membership functions (with caching)
        x_values, membership_functions = self.compute_cached_membership_functions(windows)
        
        # Convert to format expected by similarity computation
        windowed_data = type('WindowedData', (), {
            'windows': windows,
            'labels': labels,
            'x_values': x_values,
            'membership_functions': membership_functions
        })()
        
        # Compute similarities using cached membership functions
        logger.info(f"🔄 Computing similarities for {len(metrics)} using cached membership functions")
        similarity_matrices = self._compute_similarities_from_cached_membership(
            membership_functions, metrics, x_values
        )
        
        # Evaluate retrieval performance
        results = {}
        for metric_name, similarity_matrix in similarity_matrices.items():
            try:
                # Simple train/test split for evaluation
                n_windows = len(windows)
                split_idx = n_windows // 2
                
                query_similarities = similarity_matrix[:split_idx, split_idx:]
                query_labels = labels[:split_idx]
                library_labels = labels[split_idx:]
                
                # Evaluate retrieval metrics
                metrics_result = compute_retrieval_metrics(
                    query_similarities, query_labels, library_labels, topk=1
                )
                hit_at_1 = metrics_result["hit@1"]
                mrr = metrics_result["mrr"]
                
                results[metric_name] = {
                    "hit_at_1": hit_at_1,
                    "mrr": mrr,
                    "num_windows": n_windows,
                    "num_activities": len(activity_names),
                    "activities": activity_names
                }
                
                logger.info(f"   {metric_name}: Hit@1={hit_at_1:.3f}, MRR={mrr:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {metric_name}: {e}")
                continue
        
        return {
            "label_type": label_type,
            "results": results,
            "num_windows": len(windows),
            "num_activities": len(activity_names),
            "activities": activity_names,
            "window_config": {
                "window_size": self.window_config.window_size,
                "overlap_ratio": self.window_config.overlap_ratio
            }
        }
    
    def run_multi_label_experiment(self, label_types: List[str] = None, 
                                 metrics: List[str] = None) -> Dict[str, Any]:
        """
        Run experiments across multiple label types efficiently using standard windows 
        and cached membership functions.
        """
        if label_types is None:
            label_types = ["Locomotion", "ML_Both_Arms", "HL_Activity"]
        
        if metrics is None:
            # Expanded metric set with new high-performance metrics
            metrics = [
                "jaccard", "cosine", "dice", "pearson", "overlap_coefficient",
                "spearman", "kendall_tau", "tversky", "weighted_jaccard", "mahalanobis",
                "jensen_shannon", "bhattacharyya_coefficient", "hellinger"
            ]
        
        logger.info(f"🚀 Running unified multi-label experiment")
        logger.info(f"   Label types: {label_types}")
        logger.info(f"   Metrics: {metrics}")
        
        # Create standard windows first
        window_info = self.create_standard_windows()
        logger.info(f"📊 Created {window_info['num_windows']} standard windows")
        
        # Pre-compute membership functions for ALL standard windows ONCE
        logger.info(f"⚡ Pre-computing membership functions for all {len(self.standard_windows)} standard windows...")
        standard_x_values, standard_membership_functions = self.compute_cached_membership_functions(self.standard_windows)
        logger.info(f"✅ Membership functions computed and cached for reuse across all label types")
        
        # Run experiments for each label type using cached membership functions
        all_results = {}
        total_label_types = len(label_types)
        
        for idx, label_type in enumerate(label_types, 1):
            logger.info(f"🚀 EXPERIMENT PROGRESS: Processing label type {idx}/{total_label_types}: {label_type}")
            try:
                results = self.run_label_type_experiment_with_cached_membership(
                    label_type, metrics, standard_x_values, standard_membership_functions
                )
                all_results[label_type] = results
                logger.info(f"✅ EXPERIMENT PROGRESS: Completed {idx}/{total_label_types} label types ({idx/total_label_types*100:.1f}%)")
            except Exception as e:
                logger.error(f"❌ Failed experiment for {label_type}: {e}")
                logger.error(f"   Traceback: {e}", exc_info=True)
                all_results[label_type] = {"error": str(e)}
        
        # Create summary comparison
        summary = self._create_results_summary(all_results)
        
        return {
            "window_info": window_info,
            "label_type_results": all_results,
            "summary": summary,
            "cache_info": self.cache.cache_info()
        }
    
    def _create_results_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary comparison across label types."""
        summary = {
            "best_performance": {},
            "comparison": [],
            "insights": []
        }
        
        # Find best performing configurations
        for label_type, result_data in all_results.items():
            if "error" in result_data:
                continue
                
            results = result_data.get("results", {})
            if not results:
                continue
            
            # Find best metric for this label type
            best_metric = None
            best_score = 0
            
            for metric_name, metrics_data in results.items():
                hit_at_1 = metrics_data.get("hit_at_1", 0)
                if hit_at_1 > best_score:
                    best_score = hit_at_1
                    best_metric = metric_name
            
            if best_metric:
                summary["best_performance"][label_type] = {
                    "metric": best_metric,
                    "hit_at_1": best_score,
                    "mrr": results[best_metric]["mrr"],
                    "num_windows": result_data["num_windows"],
                    "num_activities": result_data["num_activities"]
                }
        
        return summary
    
    def run_label_type_experiment_with_cached_membership(self, label_type: str, metrics: List[str], 
                                                       standard_x_values: np.ndarray, 
                                                       standard_membership_functions: List[List[np.ndarray]]) -> Dict[str, Any]:
        """
        Run experiment for a specific label type using pre-computed membership functions.
        This is the key optimization that avoids recomputing membership functions.
        """
        logger.info(f"🧪 Running experiment for {label_type} (using cached membership functions)")
        
        # Get filtered windows for this label type with majority vote labeling
        windows, labels, activity_names = self.get_label_filtered_windows(label_type)
        
        if len(windows) == 0:
            logger.error(f"No valid windows found for {label_type}")
            return {"error": "No valid windows"}
        
        # Map filtered windows to their corresponding cached membership functions
        # The key insight: filtered windows are a subset of standard windows, so we need to 
        # map by the window indices, not by data comparison
        
        filtered_membership_functions = []
        filtered_window_indices = []
        
        # Re-implement the filtering logic to get the indices  
        raw_labels = self.standard_labels[label_type]  # Don't overwrite the windowed labels!
        step_size = int(self.window_config.window_size * (1 - self.window_config.overlap_ratio))
        
        for i, (window_data, timestamp) in enumerate(zip(self.standard_windows, self.standard_timestamps)):
            # Calculate window time range (same logic as get_label_filtered_windows)
            start_idx = i * step_size
            end_idx = start_idx + self.window_config.window_size
            
            # Extract label sequence for this window
            if end_idx <= len(raw_labels):
                label_sequence = raw_labels[start_idx:end_idx]
                
                # Apply majority vote (same logic)
                window_label = self._assign_majority_vote_label(label_sequence)
                
                # Only keep windows with valid, non-Unknown labels
                if window_label is not None and window_label != "Unknown":
                    filtered_membership_functions.append(standard_membership_functions[i])
                    filtered_window_indices.append(i)
        
        logger.info(f"📊 Mapped {len(filtered_membership_functions)} filtered windows to cached membership functions")
        
        # Convert to WindowedData for proper balanced evaluation BEFORE computing similarities
        from thesis.data import WindowedData, balance_windows_by_class, train_test_split_windows
        from collections import Counter
        
        # Create WindowedData object from filtered windows  
        windows_array = np.array([w for w in windows])
        labels_array = np.array(labels)
        
        # Count the actual windowed labels (not raw label sequences)
        windowed_label_counts = Counter(labels_array)  # Use the converted windowed labels array
        
        windowed_data = WindowedData(
            windows=windows_array,
            labels=labels_array,
            window_indices=np.array([[i*60, (i+1)*60] for i in range(len(windows))]),  # dummy indices
            metadata={"label_distribution": windowed_label_counts}
        )
        
        logger.info(f"📊 Class distribution: {dict(windowed_label_counts)}")
        logger.info(f"📊 WindowedData shape: windows={windows_array.shape}, labels={labels_array.shape}")
        
        # Special handling for small datasets: use all available windows without balancing
        min_class_size = min(windowed_label_counts.values())
        total_windows = len(windows)
        
        if total_windows < 100 or min_class_size == 1:
            # Use all available windows for small datasets
            logger.info(f"📊 Small dataset ({total_windows} windows), using all available windows without balancing")
            balanced_data = windowed_data  # No balancing
            max_windows_per_class = min_class_size  # For library_per_class calculation
        else:
            # Strategy: Increase class balance for better retrieval performance
            # Instead of balancing to smallest class, use more windows per class
            max_windows_per_class = min(min_class_size * 2, min_class_size + 20)  # More generous balancing
            logger.info(f"📊 Improved balancing to {max_windows_per_class} windows per class (was {min_class_size})")
            balanced_data = balance_windows_by_class(windowed_data, max_windows_per_class=max_windows_per_class)
        
        # Adjust library_per_class for small datasets to ensure meaningful splits
        if total_windows < 100:
            # For small datasets, use a simple 50/50 split approach
            library_per_class = None  # Will use test_fraction instead
            logger.info(f"📊 Small dataset: using 50/50 split instead of per-class allocation")
            
            lib_data, qry_data, lib_indices, qry_indices = train_test_split_windows(
                balanced_data,
                library_per_class=None,
                test_fraction=0.5,
                stratified=True,
                random_state=42,
                return_indices=True
            )
        else:
            library_per_class = min(20, max(1, max_windows_per_class // 2))  # Increased library size for better retrieval
            logger.info(f"📊 Using library_per_class={library_per_class}")
            
            lib_data, qry_data, lib_indices, qry_indices = train_test_split_windows(
                balanced_data,
                library_per_class=library_per_class,
                stratified=True,
                random_state=42,
                return_indices=True
            )
        
        n_lib = len(lib_data.windows)
        n_qry = len(qry_data.windows)
        total_similarities = n_qry * n_lib * len(metrics)
        
        logger.info(f"📊 Split: {n_lib} library, {n_qry} query windows")
        logger.info(f"📊 Reduced from {len(windows)} to {n_lib + n_qry} balanced windows")
        logger.info(f"📊 Computing {total_similarities:,} similarities (efficient query×library)")
        
        # Extract membership functions for balanced windows using indices
        # The balanced data maintains the original indices, so we can map directly
        qry_membership = []
        lib_membership = []
        
        # For query windows: use balanced indices to get membership functions
        for qry_idx in qry_indices:
            # qry_idx refers to the index in balanced_data, which maps to original windows
            qry_membership.append(filtered_membership_functions[qry_idx])
        
        # For library windows: use balanced indices to get membership functions  
        for lib_idx in lib_indices:
            # lib_idx refers to the index in balanced_data, which maps to original windows
            lib_membership.append(filtered_membership_functions[lib_idx])
        
        # Use efficient query×library similarity computation        
        logger.info(f"🔄 Computing similarities for {len(metrics)} metrics using parallel optimized function")
        
        similarity_matrices = self._compute_query_library_similarities_efficient(
            qry_membership, lib_membership, metrics, standard_x_values
        )
        
        # Evaluate retrieval performance using balanced split
        results = {}
        for metric_name, similarity_matrix in similarity_matrices.items():
            try:
                # similarity_matrix is already query×library format from efficient computation
                
                # Evaluate retrieval metrics
                metrics_result = compute_retrieval_metrics(
                    similarity_matrix, qry_data.labels, lib_data.labels, topk=1
                )
                hit_at_1 = metrics_result["hit@1"]
                mrr = metrics_result["mrr"]
                
                results[metric_name] = {
                    "hit_at_1": hit_at_1,
                    "mrr": mrr,
                    "num_query": n_qry,
                    "num_library": n_lib,
                    "num_activities": len(activity_names),
                    "activities": activity_names,
                    "library_per_class": library_per_class,
                    "balanced_class_size": max_windows_per_class
                }
                
                logger.info(f"   {metric_name}: Hit@1={hit_at_1:.3f}, MRR={mrr:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {metric_name}: {e}")
                continue
        
        return {
            "label_type": label_type,
            "results": results,
            "num_windows": len(windows),
            "num_activities": len(activity_names),
            "activities": activity_names,
            "window_config": {
                "window_size": self.window_config.window_size,
                "overlap_ratio": self.window_config.overlap_ratio
            }
        }
    
    def _compute_similarities_with_bypass(self, windows: List[np.ndarray], 
                                        membership_functions: List[List[np.ndarray]], 
                                        x_values: np.ndarray, metrics: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute similarities using the original optimized approach but with pre-computed membership functions.
        This maintains the speed while avoiding duplicate computation.
        """
        from thesis.fuzzy.similarity import compute_per_sensor_similarity
        from joblib import Parallel, delayed
        
        n_windows = len(windows)
        similarity_matrices = {}
        
        logger.info(f"🚀 Using original optimized vectorized similarity computation")
        
        for metric_name in metrics:
            logger.info(f"   Computing {metric_name} similarities (optimized)...")
            
            # Create window pairs for upper triangle + diagonal  
            window_pairs = [(i, j) for i in range(n_windows) for j in range(i, n_windows)]
            
            # Use parallel computation with the unified function
            from joblib import Parallel, delayed
            
            def compute_pair(i, j):
                return compute_per_sensor_similarity(
                    membership_functions[i], membership_functions[j], x_values, metric_name, True
                )
            
            similarities = Parallel(n_jobs=-1, backend='threading')(
                delayed(compute_pair)(i, j) for i, j in window_pairs
            )
            
            # Fill symmetric matrix
            similarity_matrix = np.zeros((n_windows, n_windows))
            for idx, (i, j) in enumerate(window_pairs):
                sim_value = similarities[idx]
                similarity_matrix[i, j] = sim_value
                similarity_matrix[j, i] = sim_value
            
            similarity_matrices[metric_name] = similarity_matrix
            
        logger.info(f"✅ Computed {len(similarity_matrices)} similarity matrices using optimized cached approach")
        return similarity_matrices
    
    def _compute_similarities_from_cached_membership(self, membership_functions: List[List[np.ndarray]], 
                                                   metrics: List[str], x_values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute similarity matrices directly from cached membership functions using parallel processing.
        This avoids redundant membership function computation while maximizing speed.
        """
        from thesis.fuzzy.similarity import compute_per_sensor_similarity
        from joblib import Parallel, delayed
        
        n_windows = len(membership_functions)
        
        logger.info(f"🚀 Using parallel similarity computation for {n_windows} windows, {len(metrics)} metrics")
        
        def compute_window_pair_similarity(i, j, metric):
            """Compute similarity between two windows for a specific metric."""
            try:
                return compute_per_sensor_similarity(
                    membership_functions[i],
                    membership_functions[j], 
                    x_values,
                    metric=metric,
                    normalise=True
                )
            except Exception as e:
                logger.warning(f"Failed to compute {metric} for windows {i},{j}: {e}")
                return 0.0
        
        def compute_metric_matrix(metric_name):
            """Compute similarity matrix for one metric in parallel - OPTIMIZED for query/library split."""
            logger.info(f"   🔄 Computing {metric_name} similarities (parallel)...")
            
            # OPTIMIZATION: Only compute query × library similarities (not full n×n matrix)
            split_idx = n_windows // 2
            query_indices = list(range(split_idx))
            library_indices = list(range(split_idx, n_windows))
            
            # Create only needed pairs: query × library
            query_library_pairs = [(i, j) for i in query_indices for j in library_indices]
            
            logger.info(f"   Optimized: Computing {len(query_library_pairs):,} pairs instead of {n_windows*(n_windows+1)//2:,}")
            
            # Parallel computation of similarities for this metric
            similarities = Parallel(n_jobs=-1, backend='threading')(
                delayed(compute_window_pair_similarity)(i, j, metric_name)
                for i, j in query_library_pairs
            )
            
            # Fill only the query × library submatrix
            similarity_matrix = np.zeros((n_windows, n_windows))
            for idx, (i, j) in enumerate(query_library_pairs):
                sim_value = similarities[idx]
                similarity_matrix[i, j] = sim_value
            
            logger.info(f"   ✅ {metric_name} completed - {len(query_library_pairs):,} computations")
            return metric_name, similarity_matrix
        
        # Process all metrics in parallel
        logger.info(f"🚀 Processing {len(metrics)} metrics in parallel...")
        metric_results = Parallel(n_jobs=min(len(metrics), 4), backend='threading')(
            delayed(compute_metric_matrix)(metric_name)
            for metric_name in metrics
        )
        
        # Convert results to dictionary
        similarity_matrices = dict(metric_results)
            
        logger.info(f"✅ Computed {len(similarity_matrices)} similarity matrices using parallel cached membership functions")
        return similarity_matrices
    
    def _compute_query_library_similarities_efficient(self, qry_membership: List[List[np.ndarray]], 
                                                    lib_membership: List[List[np.ndarray]], 
                                                    metrics: List[str], x_values: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute similarity matrix between query and library sets only (MUCH more efficient)."""
        from thesis.fuzzy.similarity import compute_per_sensor_similarity
        from joblib import Parallel, delayed
        
        n_qry = len(qry_membership)
        n_lib = len(lib_membership)
        
        logger.info(f"🚀 Computing {n_qry}×{n_lib} query-library similarities for {len(metrics)} metrics (EFFICIENT)")
        
        def compute_query_lib_similarity(q_idx, l_idx, metric):
            """Compute similarity between query q_idx and library l_idx."""
            try:
                return compute_per_sensor_similarity(
                    qry_membership[q_idx], lib_membership[l_idx], x_values, metric=metric, normalise=True
                )
            except Exception as e:
                logger.warning(f"Failed to compute {metric} for query {q_idx}, lib {l_idx}: {e}")
                return 0.0
        
        def compute_metric_matrix(metric_name):
            """Compute query×library similarity matrix for one metric."""
            logger.info(f"   🔄 Computing {metric_name} similarities...")
            
            # Create query×library pairs (much fewer than all pairs!)
            pairs = [(q, l) for q in range(n_qry) for l in range(n_lib)]
            total_pairs = len(pairs)
            
            logger.info(f"   📊 Processing {total_pairs:,} pairs for {metric_name}")
            
            # Create progress tracking wrapper
            def compute_with_progress(pair_idx_and_data):
                pair_idx, (q, l) = pair_idx_and_data
                
                # Log progress every 10% or every 1000 computations, whichever is smaller
                progress_interval = min(1000, max(1, total_pairs // 10))
                if pair_idx % progress_interval == 0:
                    progress_pct = (pair_idx / total_pairs) * 100
                    logger.info(f"   📈 {metric_name} progress: {pair_idx}/{total_pairs} ({progress_pct:.1f}%)")
                
                return compute_query_lib_similarity(q, l, metric_name)
            
            # Parallel computation with progress tracking
            similarities = Parallel(n_jobs=-1, backend='threading')(
                delayed(compute_with_progress)((idx, pair))
                for idx, pair in enumerate(pairs)
            )
            
            # Fill query×library matrix
            sim_matrix = np.zeros((n_qry, n_lib))
            for idx, (q, l) in enumerate(pairs):
                sim_matrix[q, l] = similarities[idx]
            
            logger.info(f"   ✅ {metric_name} completed - {total_pairs:,} computations finished")
            return metric_name, sim_matrix
        
        # Process all metrics in parallel
        metric_results = Parallel(n_jobs=min(len(metrics), 4), backend='threading')(
            delayed(compute_metric_matrix)(metric_name) for metric_name in metrics
        )
        
        return dict(metric_results)


def main():
    """Example usage of the unified windowing experiment."""
    
    # Initialize experiment
    experiment = UnifiedWindowingExperiment(
        window_config=WindowConfig(window_size=120, overlap_ratio=0.5),
        cache_dir="cache/unified_windows"
    )
    
    # Run multi-label experiment
    results = experiment.run_multi_label_experiment(
        label_types=["Locomotion", "ML_Both_Arms", "HL_Activity"],
        metrics=[
            "jaccard", "cosine", "dice", "pearson", "overlap_coefficient",
            "spearman", "kendall_tau", "tversky", "weighted_jaccard", "mahalanobis",
            "jensen_shannon", "bhattacharyya_coefficient", "hellinger"
        ]
    )
    
    # Save results
    output_dir = Path("results/unified_windowing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "unified_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Print summary
    print("\n🎯 Unified Windowing Experiment Results:")
    print(f"   Standard windows created: {results['window_info']['num_windows']}")
    print(f"   Cache entries: {results['cache_info']['cache_size']}")
    
    for label_type, perf in results["summary"]["best_performance"].items():
        print(f"   {label_type}: {perf['metric']} = {perf['hit_at_1']:.3f} Hit@1")


if __name__ == "__main__":
    main()
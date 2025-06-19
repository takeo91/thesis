"""
Time Series Windowing Module for Activity Classification

This module provides sliding window segmentation for sensor data time series,
optimized for activity recognition applications with configurable window sizes,
overlaps, and label assignment strategies.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union, Any
from collections import Counter
from dataclasses import dataclass


@dataclass
class WindowConfig:
    """Configuration for time series windowing."""
    window_size: int
    overlap_ratio: float
    label_strategy: str = "majority_vote"  # "majority_vote", "first", "last", "mode"
    min_samples_per_class: int = 5  # Minimum samples required per activity class
    
    def __post_init__(self):
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0.0, 1.0)")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.label_strategy not in ["majority_vote", "first", "last", "mode"]:
            raise ValueError("Invalid label_strategy")


@dataclass
class WindowedData:
    """Container for windowed time series data."""
    windows: np.ndarray  # Shape: (n_windows, window_size, n_features)
    labels: np.ndarray   # Shape: (n_windows,)
    window_indices: np.ndarray  # Shape: (n_windows, 2) - start, end indices
    metadata: Dict[str, Any]
    
    @property
    def n_windows(self) -> int:
        return len(self.windows)
    
    @property
    def n_features(self) -> int:
        return self.windows.shape[2] if len(self.windows.shape) == 3 else 1
    
    @property
    def window_size(self) -> int:
        return self.windows.shape[1]


def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    config: WindowConfig
) -> WindowedData:
    """
    Create sliding windows from time series data with labels.
    
    Args:
        data: Time series data, shape (n_samples, n_features) or (n_samples,)
        labels: Activity labels for each sample, shape (n_samples,)
        config: Windowing configuration
    
    Returns:
        WindowedData object containing windowed data and labels
    """
    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    
    if len(labels) != n_samples:
        raise ValueError(f"Data and labels length mismatch: {n_samples} vs {len(labels)}")
    
    if n_samples < config.window_size:
        raise ValueError(f"Data too short for window size: {n_samples} < {config.window_size}")
    
    # Calculate step size based on overlap
    step_size = max(1, int(config.window_size * (1 - config.overlap_ratio)))
    
    # Generate window start indices
    max_start = n_samples - config.window_size + 1
    start_indices = np.arange(0, max_start, step_size)
    
    # Create windows
    windows = []
    window_labels = []
    window_indices = []
    
    for start_idx in start_indices:
        end_idx = start_idx + config.window_size
        
        # Extract window data
        window_data = data[start_idx:end_idx]
        window_label_sequence = labels[start_idx:end_idx]
        
        # Assign window label based on strategy
        window_label = _assign_window_label(window_label_sequence, config.label_strategy)
        
        if window_label is not None:  # Skip windows with ambiguous labels
            windows.append(window_data)
            window_labels.append(window_label)
            window_indices.append([start_idx, end_idx])
    
    if not windows:
        raise ValueError("No valid windows created - check label strategy and data quality")
    
    # Convert to arrays
    windows = np.array(windows)
    window_labels = np.array(window_labels)
    window_indices = np.array(window_indices)
    
    # Create metadata
    metadata = {
        "window_size": config.window_size,
        "overlap_ratio": config.overlap_ratio,
        "step_size": step_size,
        "label_strategy": config.label_strategy,
        "original_length": n_samples,
        "n_features": n_features,
        "label_distribution": dict(Counter(window_labels))
    }
    
    return WindowedData(
        windows=windows,
        labels=window_labels, 
        window_indices=window_indices,
        metadata=metadata
    )


def _assign_window_label(
    label_sequence: np.ndarray,
    strategy: str
) -> Optional[Union[int, str]]:
    """
    Assign a single label to a window based on the label sequence.
    
    Args:
        label_sequence: Sequence of labels within the window
        strategy: Label assignment strategy
    
    Returns:
        Assigned label or None if ambiguous
    """
    if strategy == "majority_vote":
        # Use most frequent label, return None if tie
        label_counts = Counter(label_sequence)
        most_common = label_counts.most_common(2)
        
        if len(most_common) == 1:
            return most_common[0][0]
        elif most_common[0][1] > most_common[1][1]:
            return most_common[0][0]
        else:
            return None  # Tie - ambiguous window
    
    elif strategy == "first":
        return label_sequence[0]
    
    elif strategy == "last":
        return label_sequence[-1]
    
    elif strategy == "mode":
        # Same as majority_vote but always returns something
        return Counter(label_sequence).most_common(1)[0][0]
    
    else:
        raise ValueError(f"Unknown label strategy: {strategy}")


def filter_windowed_data_by_class_count(
    windowed_data: WindowedData,
    min_samples_per_class: int = 5
) -> WindowedData:
    """
    Filter windowed data to keep only classes with sufficient samples.
    
    Args:
        windowed_data: Original windowed data
        min_samples_per_class: Minimum samples required per class
    
    Returns:
        Filtered windowed data
    """
    label_counts = Counter(windowed_data.labels)
    valid_labels = {label for label, count in label_counts.items() 
                   if count >= min_samples_per_class}
    
    if not valid_labels:
        raise ValueError(f"No classes have >= {min_samples_per_class} samples")
    
    # Filter data
    valid_mask = np.isin(windowed_data.labels, list(valid_labels))
    
    filtered_windows = windowed_data.windows[valid_mask]
    filtered_labels = windowed_data.labels[valid_mask]
    filtered_indices = windowed_data.window_indices[valid_mask]
    
    # Update metadata
    updated_metadata = windowed_data.metadata.copy()
    updated_metadata.update({
        "filtered_classes": sorted(valid_labels),
        "removed_classes": sorted(set(label_counts.keys()) - valid_labels),
        "min_samples_per_class": min_samples_per_class,
        "label_distribution": dict(Counter(filtered_labels))
    })
    
    return WindowedData(
        windows=filtered_windows,
        labels=filtered_labels,
        window_indices=filtered_indices,
        metadata=updated_metadata
    )


def create_multiple_window_configs(
    window_sizes: List[int] = [128, 256],
    overlap_ratios: List[float] = [0.5, 0.7]
) -> List[WindowConfig]:
    """
    Create multiple window configurations for comprehensive analysis.
    
    Args:
        window_sizes: List of window sizes to test
        overlap_ratios: List of overlap ratios to test
    
    Returns:
        List of WindowConfig objects
    """
    configs = []
    for window_size in window_sizes:
        for overlap_ratio in overlap_ratios:
            config = WindowConfig(
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                label_strategy="majority_vote",
                min_samples_per_class=5
            )
            configs.append(config)
    
    return configs


def windowed_data_to_dataframe(windowed_data: WindowedData) -> pd.DataFrame:
    """
    Convert windowed data to DataFrame for analysis.
    
    Args:
        windowed_data: WindowedData object
    
    Returns:
        DataFrame with window information
    """
    n_windows, window_size, n_features = windowed_data.windows.shape
    
    data = []
    for i in range(n_windows):
        window_info = {
            "window_id": i,
            "label": windowed_data.labels[i],
            "start_idx": windowed_data.window_indices[i, 0],
            "end_idx": windowed_data.window_indices[i, 1],
            "window_size": window_size,
            "n_features": n_features
        }
        
        # Add basic statistics for each feature
        for feat_idx in range(n_features):
            feat_data = windowed_data.windows[i, :, feat_idx]
            window_info.update({
                f"feat_{feat_idx}_mean": np.mean(feat_data),
                f"feat_{feat_idx}_std": np.std(feat_data),
                f"feat_{feat_idx}_min": np.min(feat_data),
                f"feat_{feat_idx}_max": np.max(feat_data)
            })
        
        data.append(window_info)
    
    return pd.DataFrame(data)


def get_windowing_summary(windowed_data: WindowedData) -> Dict[str, Any]:
    """
    Get comprehensive summary of windowed data.
    
    Args:
        windowed_data: WindowedData object
    
    Returns:
        Summary dictionary
    """
    label_dist = windowed_data.metadata["label_distribution"]
    
    summary = {
        "n_windows": windowed_data.n_windows,
        "window_size": windowed_data.window_size,
        "n_features": windowed_data.n_features,
        "overlap_ratio": windowed_data.metadata["overlap_ratio"],
        "step_size": windowed_data.metadata["step_size"],
        "original_length": windowed_data.metadata["original_length"],
        "compression_ratio": windowed_data.n_windows / windowed_data.metadata["original_length"],
        "n_classes": len(label_dist),
        "class_labels": sorted(label_dist.keys()),
        "samples_per_class": label_dist,
        "min_class_samples": min(label_dist.values()),
        "max_class_samples": max(label_dist.values()),
        "class_balance_ratio": min(label_dist.values()) / max(label_dist.values()),
        "label_strategy": windowed_data.metadata["label_strategy"]
    }
    
    return summary


# Example usage and testing functions
def demo_windowing():
    """Demonstrate windowing functionality with synthetic data."""
    print("🔧 Time Series Windowing Demo")
    
    # Create synthetic sensor data
    n_samples = 1000
    n_features = 3
    
    # Generate synthetic time series with different activities
    np.random.seed(42)
    
    # Activity patterns
    activities = []
    labels = []
    
    # Activity 0: Low frequency oscillation
    t1 = np.linspace(0, 4*np.pi, 300)
    act0 = np.column_stack([
        np.sin(t1) + 0.1*np.random.randn(300),
        np.cos(t1) + 0.1*np.random.randn(300), 
        0.5*np.sin(2*t1) + 0.1*np.random.randn(300)
    ])
    activities.append(act0)
    labels.extend([0] * 300)
    
    # Activity 1: High frequency oscillation  
    t2 = np.linspace(0, 8*np.pi, 400)
    act1 = np.column_stack([
        0.5*np.sin(3*t2) + 0.1*np.random.randn(400),
        0.7*np.cos(4*t2) + 0.1*np.random.randn(400),
        np.sin(t2) + 0.1*np.random.randn(400)
    ])
    activities.append(act1)
    labels.extend([1] * 400)
    
    # Activity 2: Step pattern
    t3 = np.arange(300)
    steps = np.repeat(np.random.choice([-1, 1], 30), 10)
    act2 = np.column_stack([
        np.cumsum(steps) + 0.1*np.random.randn(300),
        0.5*steps + 0.1*np.random.randn(300),
        -0.3*steps + 0.1*np.random.randn(300)
    ])
    activities.append(act2)
    labels.extend([2] * 300)
    
    # Combine all activities
    data = np.vstack(activities)
    labels = np.array(labels)
    
    print(f"📊 Generated synthetic data: {data.shape}, {len(np.unique(labels))} activities")
    
    # Test different windowing configurations
    configs = create_multiple_window_configs([64, 128], [0.5, 0.7])
    
    for i, config in enumerate(configs):
        print(f"\n⚙️ Configuration {i+1}: Window={config.window_size}, Overlap={config.overlap_ratio}")
        
        try:
            windowed = create_sliding_windows(data, labels, config)
            windowed_filtered = filter_windowed_data_by_class_count(windowed, min_samples_per_class=10)
            
            summary = get_windowing_summary(windowed_filtered)
            
            print(f"   ✅ Windows: {summary['n_windows']}")
            print(f"   ✅ Classes: {summary['n_classes']} ({summary['class_labels']})")
            print(f"   ✅ Samples per class: {summary['samples_per_class']}")
            print(f"   ✅ Balance ratio: {summary['class_balance_ratio']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    return data, labels, windowed_filtered


def balance_windows_by_class(
    windowed_data: WindowedData,
    max_windows_per_class: int,
    random_state: Optional[int] = None
) -> WindowedData:
    """Return a balanced subset of *windowed_data* by capping the maximum number
    of windows per class.

    If a class has more than *max_windows_per_class* windows, a random subset is
    kept. Classes with fewer windows are left unchanged (no augmentation).

    Parameters
    ----------
    windowed_data : WindowedData
        The original windowed dataset.
    max_windows_per_class : int
        Maximum number of windows to keep for any single class. Pass ``None`` to
        disable balancing.
    random_state : int, optional
        Seed for reproducible sampling.

    Returns
    -------
    WindowedData
        A new WindowedData instance with balanced class counts.
    """
    if max_windows_per_class is None:
        return windowed_data  # no-op

    rng = np.random.default_rng(random_state)

    # Collect indices per label
    from collections import defaultdict
    idx_by_label = defaultdict(list)
    for idx, lbl in enumerate(windowed_data.labels):
        idx_by_label[lbl].append(idx)

    keep_indices = []
    for lbl, idx_list in idx_by_label.items():
        if len(idx_list) > max_windows_per_class:
            sampled = rng.choice(idx_list, size=max_windows_per_class, replace=False)
            keep_indices.extend(sampled)
        else:
            keep_indices.extend(idx_list)

    keep_indices = np.array(sorted(keep_indices))

    return WindowedData(
        windows=windowed_data.windows[keep_indices],
        labels=windowed_data.labels[keep_indices],
        window_indices=windowed_data.window_indices[keep_indices],
        metadata={**windowed_data.metadata, "balanced": True, "max_per_class": max_windows_per_class}
    )


if __name__ == "__main__":
    demo_windowing() 
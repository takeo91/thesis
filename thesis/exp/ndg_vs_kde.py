#!/usr/bin/env python
"""
Experiment: NDG vs KDE Membership Function Comparison

This script compares the Neighbor Density Graph (NDG) and Kernel Density Estimation (KDE)
methods for computing membership functions across different datasets, signal lengths,
and sensor locations.

Metrics compared:
- Error between NDG and KDE (KL divergence and Chi-squared)
- Computation time
- Memory usage

Datasets:
- Synthetic data (various signal types)
- Opportunity dataset
- PAMAP2 dataset
"""

import time
import os
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.model_selection import KFold
from typing import Tuple, List, Dict, Any, Optional, Union
import warnings

from thesis.fuzzy.membership import compute_membership_functions
from thesis.data.datasets import create_opportunity_dataset, create_pamap2_dataset

# Set Seaborn style for better visualizations
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def measure_execution(func, *args, **kwargs) -> Tuple[Any, float, float]:
    """
    Measure execution time and memory usage of a function.
    
    Args:
        func: Function to execute
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Tuple of (function result, execution time in seconds, peak memory in MB)
    """
    # Get current process
    process = psutil.Process(os.getpid())
    
    # Record starting memory
    start_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Record start time and execute function
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    # Record peak memory
    current_memory = process.memory_info().rss / (1024 * 1024)  # MB
    peak_memory = current_memory - start_memory
    
    return result, execution_time, peak_memory


def compute_error_metrics(ndg: np.ndarray, kde: np.ndarray) -> Dict[str, float]:
    """
    Compute error metrics between NDG and KDE distributions.
    
    Args:
        ndg: NDG membership function values
        kde: KDE membership function values
        
    Returns:
        Dictionary with KL divergence and Chi-squared metrics
    """
    # Ensure values are normalized and non-zero (avoid division by zero)
    ndg = np.clip(ndg, 1e-12, None)
    kde = np.clip(kde, 1e-12, None)
    
    # Ensure both integrate to 1
    ndg = ndg / np.trapezoid(ndg)
    kde = kde / np.trapezoid(kde)
    
    # KL divergence: NDG vs KDE
    kl_div = np.sum(ndg * np.log(ndg / kde))
    
    # Chi-squared distance
    chi2 = np.sum(((ndg - kde) ** 2) / kde)
    
    return {
        "kl_divergence": kl_div,
        "chi_squared": chi2
    }


def plot_membership_functions(x: np.ndarray, ndg: np.ndarray, kde: np.ndarray, 
                              title: str, metrics: Dict[str, float]) -> plt.Figure:
    """
    Plot NDG and KDE membership functions for comparison.
    
    Args:
        x: X values (domain)
        ndg: NDG membership function values
        kde: KDE membership function values
        title: Plot title
        metrics: Dictionary with error metrics to display
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot membership functions
    ax.plot(x, ndg, label="NDG", linewidth=2)
    ax.plot(x, kde, label="KDE", linewidth=2, linestyle="--")
    
    # Add metrics to the plot
    metrics_text = (f"KL Divergence: {metrics['kl_divergence']:.6f}\n"
                   f"Chi-Squared: {metrics['chi_squared']:.6f}")
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel("Value")
    ax.set_ylabel("Membership Value")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    
    return fig


# -----------------------------------------------------------------------------
# Data Loading Functions
# -----------------------------------------------------------------------------

def generate_synthetic_data(signal_type: str, size: int, 
                           seed: int = 42) -> np.ndarray:
    """
    Generate synthetic data of different types.
    
    Args:
        signal_type: Type of signal to generate ('normal', 'uniform', 'bimodal', etc.)
        size: Number of data points
        seed: Random seed
        
    Returns:
        Array of synthetic data points
    """
    rng = np.random.RandomState(seed)
    
    if signal_type == "normal":
        # Normal distribution (loc=50, scale=5)
        return rng.normal(loc=50, scale=5, size=size)
    
    elif signal_type == "uniform":
        # Uniform distribution (low=0, high=100)
        return rng.uniform(low=0, high=100, size=size)
    
    elif signal_type == "narrow_normal":
        # Normal with smaller variance (loc=50, scale=2)
        return rng.normal(loc=50, scale=2, size=size)
    
    elif signal_type == "bimodal":
        # Bimodal distribution
        samples1 = rng.normal(loc=30, scale=5, size=size//2)
        samples2 = rng.normal(loc=70, scale=5, size=size - size//2)
        return np.concatenate([samples1, samples2])
    
    elif signal_type == "shifted_bimodal":
        # Shifted bimodal distribution
        samples1 = rng.normal(loc=40, scale=5, size=size//2)
        samples2 = rng.normal(loc=80, scale=5, size=size - size//2)
        return np.concatenate([samples1, samples2])
    
    elif signal_type == "different_scale":
        # Very different scale
        return rng.normal(loc=1050, scale=105, size=size)
    
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")


def load_opportunity_data(sensor_loc: str, size: int) -> np.ndarray:
    """
    Load data from the Opportunity dataset for a specific sensor location.
    
    Args:
        sensor_loc: Sensor location/body part
        size: Number of data points to return (or less if not available)
        
    Returns:
        Array of sensor readings
    """
    try:
        dataset = create_opportunity_dataset()
        dataset.load_data()
        
        # Get available sensor info
        sensor_info = dataset.get_available_sensors()
        body_parts = sensor_info.get("body_parts", [])
        
        if not body_parts:
            warnings.warn("No body parts found in Opportunity dataset")
            return generate_synthetic_data("normal", size)
            
        # Use provided sensor location if available, otherwise use first available
        if sensor_loc not in body_parts:
            sensor_loc = body_parts[0]
            print(f"Using body part '{sensor_loc}' instead of requested location")
        
        # Get accelerometer data
        sensor_data = dataset.get_sensor_data(
            sensor_type="Accelerometer", 
            body_part=sensor_loc,
            axis="X"
        ).values
        
        # Handle missing values and normalize
        sensor_data = sensor_data[~np.isnan(sensor_data)]
        sensor_data = (sensor_data - np.mean(sensor_data)) / max(np.std(sensor_data), 1e-9)
        
        # Return requested size or what's available
        return sensor_data[:size] if len(sensor_data) > size else sensor_data
    
    except Exception as e:
        warnings.warn(f"Error loading Opportunity data: {e}. Using synthetic data instead.")
        return generate_synthetic_data("normal", size)


def load_pamap2_data(sensor_loc: str, size: int) -> np.ndarray:
    """
    Load data from the PAMAP2 dataset for a specific sensor location.
    
    Args:
        sensor_loc: Sensor location/body part
        size: Number of data points to return (or less if not available)
        
    Returns:
        Array of sensor readings
    """
    try:
        dataset = create_pamap2_dataset()
        dataset.load_data()
        
        # Get available sensor info
        sensor_info = dataset.get_available_sensors()
        body_parts = sensor_info.get("body_parts", [])
        
        if not body_parts:
            warnings.warn("No body parts found in PAMAP2 dataset")
            return generate_synthetic_data("normal", size)
            
        # Use provided sensor location if available, otherwise use first available
        if sensor_loc not in body_parts:
            sensor_loc = body_parts[0]
            print(f"Using body part '{sensor_loc}' instead of requested location")
        
        # Get accelerometer data
        sensor_data = dataset.get_sensor_data(
            sensor_type="Accelerometer", 
            body_part=sensor_loc,
            measurement_type="acc",
            axis="X"
        ).values
        
        # Handle missing values and normalize
        sensor_data = sensor_data[~np.isnan(sensor_data)]
        sensor_data = (sensor_data - np.mean(sensor_data)) / max(np.std(sensor_data), 1e-9)
        
        # Return requested size or what's available
        return sensor_data[:size] if len(sensor_data) > size else sensor_data
    
    except Exception as e:
        warnings.warn(f"Error loading PAMAP2 data: {e}. Using synthetic data instead.")
        return generate_synthetic_data("normal", size)


def load_dataset(dataset: str, sensor_loc: str, size: int) -> np.ndarray:
    """
    Load data from specified dataset.
    
    Args:
        dataset: Dataset name ('synthetic_normal', 'synthetic_bimodal', 'opportunity', 'pamap2')
        sensor_loc: Sensor location/body part (for real datasets) or signal type (for synthetic)
        size: Number of data points
        
    Returns:
        Array of sensor data
    """
    if dataset.startswith("synthetic_"):
        signal_type = dataset.split("_", 1)[1]
        return generate_synthetic_data(signal_type, size)
    
    elif dataset == "opportunity":
        return load_opportunity_data(sensor_loc, size)
    
    elif dataset == "pamap2":
        return load_pamap2_data(sensor_loc, size)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# -----------------------------------------------------------------------------
# Main Experiment Functions
# -----------------------------------------------------------------------------

def run_single_experiment(dataset: str, sensor_loc: str, 
                         size: int, sigma: Union[float, str]) -> Dict[str, Any]:
    """
    Run a single experiment comparing NDG and KDE on a dataset.
    
    Args:
        dataset: Dataset name
        sensor_loc: Sensor location/body part or signal type
        size: Number of data points
        sigma: Bandwidth parameter (float) or relative sigma (string like 'r0.1')
        
    Returns:
        Dictionary with experiment results
    """
    # Load data
    data = load_dataset(dataset, sensor_loc, size)
    actual_size = len(data)  # Might be less than requested size for real datasets
    
    # Create x values covering data range with margin
    data_min, data_max = np.min(data), np.max(data)
    margin = 0.5 * (data_max - data_min)
    x = np.linspace(data_min - margin, data_max + margin, 1000)
    
    # Process the sigma value - convert string format if needed
    if isinstance(sigma, str) and sigma.startswith('r'):
        # This is a relative sigma (e.g., 'r0.1' means 0.1 * data_range)
        try:
            ratio = float(sigma[1:])
            data_range = data_max - data_min
            sigma_val = ratio * data_range
        except ValueError:
            print(f"Warning: Invalid sigma string '{sigma}', using default 0.3")
            sigma_val = 0.3
    else:
        # Use absolute sigma value directly
        sigma_val = float(sigma)
    
    print(f"Running NDG with sigma={sigma_val}")
    
    # Run NDG and measure performance
    ndg_result, ndg_time, ndg_memory = measure_execution(
        lambda x_values, sensor_data, sigma_value: compute_membership_functions(
            sensor_data, x_values, method="nd", sigma=sigma_value, normalization="integral"
        )[0],
        x, data, sigma_val
    )
    
    # Run KDE and measure performance
    kde_result, kde_time, kde_memory = measure_execution(
        lambda x_values, sensor_data, sigma_value: compute_membership_functions(
            sensor_data, x_values, method="kde", sigma=sigma_value, normalization="integral"
        )[0],
        x, data, sigma_val
    )
    
    # Compute error metrics
    metrics = compute_error_metrics(ndg_result, kde_result)
    
    # Create plot
    sigma_str = sigma if isinstance(sigma, str) else f"{sigma_val:.3f}"
    title = f"{dataset} ({sensor_loc}, n={actual_size}, σ={sigma_str})"
    fig = plot_membership_functions(x, ndg_result, kde_result, title, metrics)
    
    # Return results
    return {
        "dataset": dataset,
        "sensor_loc": sensor_loc,
        "requested_size": size,
        "actual_size": actual_size,
        "sigma": sigma,
        "sigma_value": sigma_val,
        "ndg_time": ndg_time,
        "kde_time": kde_time,
        "ndg_memory": ndg_memory,
        "kde_memory": kde_memory,
        "kl_divergence": metrics["kl_divergence"],
        "chi_squared": metrics["chi_squared"],
        "figure": fig,
        "x_values": x,
        "ndg_values": ndg_result,
        "kde_values": kde_result
    }


def run_experiments_by_length(datasets: List[Dict[str, str]], 
                             lengths: List[int],
                             sigmas: List[Union[float, str]] = [0.3],
                             k_folds: int = 3) -> pd.DataFrame:
    """
    Run experiments across different data lengths, datasets, and sigma values.
    
    Args:
        datasets: List of dataset configurations dicts with 'name' and 'sensor_loc' keys
        lengths: List of data lengths to test
        sigmas: List of sigma values or relative sigma strings (e.g., 'r0.1') to test
        k_folds: Number of folds for cross-validation
        
    Returns:
        DataFrame with experiment results
    """
    results = []
    
    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        sensor_loc = dataset_config["sensor_loc"]
        
        print(f"\nRunning experiments for {dataset_name} ({sensor_loc})...")
        
        # Get the full dataset once, with a generous size to accommodate all experiments
        max_needed_size = max(lengths) * 2  # Get more data than needed for k-fold
        full_data = load_dataset(dataset_name, sensor_loc, max_needed_size)
        print(f"  Loaded {len(full_data)} data points")
        
        # Create a k-fold splitter for consistent folds across experiments
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Store fold indices for reuse
        fold_indices = list(kf.split(full_data))
        
        for length in lengths:
            print(f"  Testing length: {length}")
            
            if len(full_data) < length:
                print(f"  Warning: Not enough data (only {len(full_data)} points available)")
                # Proceed with available data rather than skipping
                actual_length = len(full_data)
            else:
                actual_length = length
                
            for sigma in sigmas:
                print(f"    Sigma: {sigma}")
                
                # Process each fold
                for fold_idx, (train_idx, _) in enumerate(fold_indices):
                    # Get the portion of full_data for this fold's sample
                    # Use a consistent subset of the fold data based on length
                    fold_data = full_data[train_idx][:actual_length]
                    
                    if len(fold_data) == 0:
                        print(f"    Warning: Fold {fold_idx} has no data, skipping")
                        continue
                    
                    # Process the sigma value - convert string format if needed
                    if isinstance(sigma, str) and sigma.startswith('r'):
                        # This is a relative sigma (e.g., 'r0.1' means 0.1 * data_range)
                        try:
                            ratio = float(sigma[1:])
                            data_range = np.max(fold_data) - np.min(fold_data)
                            sigma_val = ratio * data_range
                        except ValueError:
                            print(f"    Warning: Invalid sigma string '{sigma}', using default 0.3")
                            sigma_val = 0.3
                    else:
                        # Use absolute sigma value directly
                        sigma_val = float(sigma)
                    
                    # Create x values for evaluation
                    data_min, data_max = np.min(fold_data), np.max(fold_data)
                    margin = 0.5 * (data_max - data_min)
                    x = np.linspace(data_min - margin, data_max + margin, 1000)
                    
                    # Run NDG and measure performance
                    ndg_result, ndg_time, ndg_memory = measure_execution(
                        lambda x_values, sensor_data, sigma_value: compute_membership_functions(
                            sensor_data, x_values, method="nd", sigma=sigma_value, normalization="integral"
                        )[0],
                        x, fold_data, sigma_val
                    )
                    
                    # Run KDE and measure performance
                    kde_result, kde_time, kde_memory = measure_execution(
                        lambda x_values, sensor_data, sigma_value: compute_membership_functions(
                            sensor_data, x_values, method="kde", sigma=sigma_value, normalization="integral"
                        )[0],
                        x, fold_data, sigma_val
                    )
                    
                    # Compute error metrics
                    metrics = compute_error_metrics(ndg_result, kde_result)
                    
                    # Create plot for the first fold of each length and sigma combination
                    if fold_idx == 0:
                        sigma_str = sigma if isinstance(sigma, str) else f"{sigma_val:.3f}"
                        title = f"{dataset_name} ({sensor_loc}, n={len(fold_data)}, σ={sigma_str})"
                        fig = plot_membership_functions(x, ndg_result, kde_result, title, metrics)
                    else:
                        fig = None
                    
                    # Store results
                    results.append({
                        "dataset": dataset_name,
                        "sensor_loc": sensor_loc,
                        "length": length,
                        "actual_size": len(fold_data),
                        "sigma": sigma,
                        "sigma_value": sigma_val,
                        "fold": fold_idx,
                        "ndg_time": ndg_time,
                        "kde_time": kde_time,
                        "ndg_memory": ndg_memory,
                        "kde_memory": kde_memory,
                        "kl_divergence": metrics["kl_divergence"],
                        "chi_squared": metrics["chi_squared"],
                        "figure": fig
                    })
    
    return pd.DataFrame(results)


def plot_length_experiment_results(results: pd.DataFrame) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Create plots summarizing the length experiment results.
    
    Args:
        results: DataFrame with experiment results
        
    Returns:
        Tuple of Matplotlib figures (time, memory, error)
    """
    # Choose a single sigma value for length comparisons
    # For simplicity, use the first absolute sigma value (typically 0.1)
    abs_sigmas = sorted([s for s in results['sigma'].unique() 
                      if not isinstance(s, str) and not str(s).startswith('r')])
    reference_sigma = abs_sigmas[0] if abs_sigmas else 0.3
    
    # Filter results for the chosen sigma value
    filtered_results = results[results['sigma'] == reference_sigma]
    
    # Compute average metrics per dataset and length
    summary = filtered_results.groupby(['dataset', 'sensor_loc', 'length']).agg({
        'ndg_time': 'mean',
        'kde_time': 'mean',
        'ndg_memory': 'mean',
        'kde_memory': 'mean',
        'kl_divergence': 'mean',
        'chi_squared': 'mean'
    }).reset_index()
    
    # Create plots
    fig_time, ax_time = plt.subplots(figsize=(12, 6))
    fig_memory, ax_memory = plt.subplots(figsize=(12, 6))
    fig_error, ax_error = plt.subplots(figsize=(12, 6))
    
    # For nice colors
    dataset_colors = {
        dataset: color for dataset, color in zip(
            summary['dataset'].unique(), 
            sns.color_palette('husl', n_colors=len(summary['dataset'].unique()))
        )
    }
    
    # Time comparison
    for dataset in summary['dataset'].unique():
        for sensor_loc in summary[summary['dataset'] == dataset]['sensor_loc'].unique():
            data = summary[(summary['dataset'] == dataset) & (summary['sensor_loc'] == sensor_loc)]
            label = f"{dataset} ({sensor_loc})"
            color = dataset_colors[dataset]
            
            ax_time.plot(data['length'], data['ndg_time'], 
                        marker='o', linestyle='-', color=color, 
                        label=f"{label} - NDG")
            ax_time.plot(data['length'], data['kde_time'], 
                        marker='x', linestyle='--', color=color, 
                        label=f"{label} - KDE")
    
    ax_time.set_xlabel('Data Length')
    ax_time.set_ylabel('Computation Time (seconds)')
    ax_time.set_title(f'NDG vs KDE: Computation Time (σ={reference_sigma})')
    ax_time.set_xscale('log')
    ax_time.set_yscale('log')
    ax_time.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_time.tight_layout()
    
    # Memory comparison
    for dataset in summary['dataset'].unique():
        for sensor_loc in summary[summary['dataset'] == dataset]['sensor_loc'].unique():
            data = summary[(summary['dataset'] == dataset) & (summary['sensor_loc'] == sensor_loc)]
            label = f"{dataset} ({sensor_loc})"
            color = dataset_colors[dataset]
            
            ax_memory.plot(data['length'], data['ndg_memory'], 
                          marker='o', linestyle='-', color=color, 
                          label=f"{label} - NDG")
            ax_memory.plot(data['length'], data['kde_memory'], 
                          marker='x', linestyle='--', color=color, 
                          label=f"{label} - KDE")
    
    ax_memory.set_xlabel('Data Length')
    ax_memory.set_ylabel('Memory Usage (MB)')
    ax_memory.set_title(f'NDG vs KDE: Memory Usage (σ={reference_sigma})')
    ax_memory.set_xscale('log')
    ax_memory.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_memory.tight_layout()
    
    # Error metrics comparison
    for dataset in summary['dataset'].unique():
        for sensor_loc in summary[summary['dataset'] == dataset]['sensor_loc'].unique():
            data = summary[(summary['dataset'] == dataset) & (summary['sensor_loc'] == sensor_loc)]
            label = f"{dataset} ({sensor_loc})"
            color = dataset_colors[dataset]
            
            ax_error.plot(data['length'], data['kl_divergence'], 
                         marker='o', linestyle='-', color=color, 
                         label=f"{label} - KL Div")
            ax_error.plot(data['length'], data['chi_squared']/100, # Scale down chi2 for better visualization
                         marker='x', linestyle='--', color=color, 
                         label=f"{label} - Chi² (/100)")
    
    ax_error.set_xlabel('Data Length')
    ax_error.set_ylabel('Error Metrics')
    ax_error.set_title(f'NDG vs KDE: Error Metrics (σ={reference_sigma})')
    ax_error.set_xscale('log')
    ax_error.set_yscale('log')
    ax_error.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig_error.tight_layout()
    
    return fig_time, fig_memory, fig_error


def plot_sigma_experiment_results(results: pd.DataFrame) -> plt.Figure:
    """
    Create plots summarizing how error metrics vary with sigma values.
    
    Args:
        results: DataFrame with experiment results
        
    Returns:
        Matplotlib figure
    """
    # For absolute sigma values, we can directly plot
    # For relative sigma values, we need to use the actual value calculated
    
    # Separate absolute and relative sigma values
    abs_sigma_results = results[~results['sigma'].astype(str).str.startswith('r')]
    rel_sigma_results = results[results['sigma'].astype(str).str.startswith('r')]
    
    # Group by dataset, sensor_loc, length and sigma
    abs_summary = abs_sigma_results.groupby(['dataset', 'sensor_loc', 'length', 'sigma']).agg({
        'kl_divergence': 'mean',
        'chi_squared': 'mean'
    }).reset_index()
    
    rel_summary = rel_sigma_results.groupby(['dataset', 'sensor_loc', 'length', 'sigma', 'sigma_value']).agg({
        'kl_divergence': 'mean',
        'chi_squared': 'mean'
    }).reset_index()
    
    # Create a figure with two subplots (absolute and relative sigma)
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(18, 8))
    
    # For nice colors
    dataset_colors = {
        dataset: color for dataset, color in zip(
            abs_summary['dataset'].unique(), 
            sns.color_palette('husl', n_colors=len(abs_summary['dataset'].unique()))
        )
    }
    
    # Plot absolute sigma results
    for dataset in abs_summary['dataset'].unique():
        for sensor_loc in abs_summary[abs_summary['dataset'] == dataset]['sensor_loc'].unique():
            # Get a medium length for plotting
            lengths = sorted(abs_summary['length'].unique())
            medium_length = lengths[len(lengths)//2] if lengths else None
            
            if medium_length is None:
                continue
                
            data = abs_summary[(abs_summary['dataset'] == dataset) & 
                               (abs_summary['sensor_loc'] == sensor_loc) &
                               (abs_summary['length'] == medium_length)]
            
            if len(data) == 0:
                continue
                
            label = f"{dataset} ({sensor_loc})"
            color = dataset_colors[dataset]
            
            ax_abs.plot(data['sigma'], data['kl_divergence'], 
                     marker='o', linestyle='-', color=color, 
                     label=f"{label} - KL Div")
            ax_abs.plot(data['sigma'], data['chi_squared']/100,  # Scale down chi2 
                     marker='x', linestyle='--', color=color, 
                     label=f"{label} - Chi² (/100)")
    
    ax_abs.set_xlabel('Absolute Sigma Value')
    ax_abs.set_ylabel('Error Metrics')
    ax_abs.set_title('NDG vs KDE Errors: Effect of Absolute Sigma')
    ax_abs.set_yscale('log')
    ax_abs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot relative sigma results
    for dataset in rel_summary['dataset'].unique():
        for sensor_loc in rel_summary[rel_summary['dataset'] == dataset]['sensor_loc'].unique():
            # Get a medium length for plotting
            lengths = sorted(rel_summary['length'].unique())
            medium_length = lengths[len(lengths)//2] if lengths else None
            
            if medium_length is None:
                continue
                
            data = rel_summary[(rel_summary['dataset'] == dataset) & 
                               (rel_summary['sensor_loc'] == sensor_loc) &
                               (rel_summary['length'] == medium_length)]
            
            if len(data) == 0:
                continue
                
            # Extract the ratio from the sigma string (e.g., 'r0.1' -> 0.1)
            data['ratio'] = data['sigma'].apply(lambda x: float(x[1:]) if isinstance(x, str) else x)
            data = data.sort_values('ratio')  # Sort by the ratio for proper line plotting
            
            label = f"{dataset} ({sensor_loc})"
            color = dataset_colors[dataset]
            
            ax_rel.plot(data['ratio'], data['kl_divergence'], 
                      marker='o', linestyle='-', color=color, 
                      label=f"{label} - KL Div")
            ax_rel.plot(data['ratio'], data['chi_squared']/100,  # Scale down chi2 
                      marker='x', linestyle='--', color=color, 
                      label=f"{label} - Chi² (/100)")
    
    ax_rel.set_xlabel('Relative Sigma Ratio (fraction of data range)')
    ax_rel.set_ylabel('Error Metrics')
    ax_rel.set_title('NDG vs KDE Errors: Effect of Relative Sigma')
    ax_rel.set_yscale('log')
    ax_rel.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    """
    Main function to run the experiments.
    """
    # Define datasets for experiments
    datasets = [
        {"name": "synthetic_normal", "sensor_loc": "normal"},
        {"name": "synthetic_bimodal", "sensor_loc": "bimodal"},
        {"name": "opportunity", "sensor_loc": "RKN^"},
        {"name": "pamap2", "sensor_loc": "Hand"}
    ]
    
    # Define lengths to test (logarithmic scale)
    lengths = [10, 30, 100, 300, 1000, 3000, 10000]
    
    # Define sigma values to test (both absolute and relative)
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 'r0.1', 'r0.2', 'r0.3']
    
    # Run experiments across different lengths and sigmas
    print("Running experiments across different data lengths and sigma values...")
    length_results = run_experiments_by_length(datasets, lengths, sigmas=sigmas, k_folds=3)
    
    # Plot summary results
    print("\nCreating summary plots...")
    fig_time, fig_memory, fig_error = plot_length_experiment_results(length_results)
    
    # Create sigma comparison plots
    print("Creating sigma comparison plots...")
    fig_sigma_error = plot_sigma_experiment_results(length_results)
    
    # Save the figures
    print("Saving plots to results directory...")
    os.makedirs("results/ndg_vs_kde", exist_ok=True)
    fig_time.savefig("results/ndg_vs_kde/time_comparison.png", dpi=300, bbox_inches="tight")
    fig_memory.savefig("results/ndg_vs_kde/memory_comparison.png", dpi=300, bbox_inches="tight")
    fig_error.savefig("results/ndg_vs_kde/error_comparison.png", dpi=300, bbox_inches="tight")
    fig_sigma_error.savefig("results/ndg_vs_kde/sigma_comparison.png", dpi=300, bbox_inches="tight")
    
    # Save individual membership function plots
    for i, row in length_results.iterrows():
        if row['figure'] is not None:
            sigma_str = row['sigma'] if isinstance(row['sigma'], str) else f"{row['sigma']:.1f}"
            fig_name = f"results/ndg_vs_kde/{row['dataset']}_{row['sensor_loc']}_{row['length']}_{sigma_str}.png"
            row['figure'].savefig(fig_name, dpi=300, bbox_inches="tight")
    
    # Save results to CSV
    csv_data = length_results.drop(columns=['figure'])
    csv_data.to_csv("results/ndg_vs_kde/experiment_results.csv", index=False)
    
    print("Experiment completed successfully!")
    return length_results


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main() 
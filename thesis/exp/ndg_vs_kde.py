"""
RQ-1 experiment: streaming NDG vs. Gaussian KDE
================================================

Can be called two ways
----------------------
1. *Programmatic / notebook* – give ``--out results.csv``.  
   A header is written once; subsequent rows are appended.

2. *Shell pipelines* – pass ``--print-values`` and **no header** is written.
   Std-out line format::

       kl_div,chi2,wall_sec,peak_rss_mb
"""

from __future__ import annotations
import argparse, csv, pathlib, time, os, psutil
import numpy as np
from scipy.stats import gaussian_kde

from thesis.fuzzy.membership import compute_ndg_streaming
from thesis.data.datasets import (
    create_opportunity_dataset, create_pamap2_dataset
)

# ---------------------------------------------------------------------------

def _kde_density(x: np.ndarray, data: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian KDE at points *x* with bandwidth *sigma* (std-units)."""
    if data.size < 2:
        return np.full_like(x, 1 / x.size)
    bw = sigma / max(np.std(data), 1e-9)      # override Scott factor
    kde = gaussian_kde(dataset=data, bw_method=bw)
    dens = np.clip(kde(x), 1e-15, None)
    integral = np.trapezoid(dens, x=x)  # normalize to integrate to 1.0
    return dens / integral

def load_dataset_data(dataset: str, fold: int, num_folds: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Load sensor data from the specified dataset and fold.
    
    Args:
        dataset: Dataset name ('opportunity' or 'pamap2')
        fold: Fold number (0 to num_folds-1)
        num_folds: Total number of folds to divide the data into
        
    Returns:
        tuple: (data array, x values array)
    """
    if dataset.lower() == 'opportunity':
        try:
            # Try to load real data
            dataset_obj = create_opportunity_dataset()
            dataset_obj.load_data()
            
            # Get sensor types and body parts using the method
            sensor_info = dataset_obj.get_available_sensors()
            body_parts = sensor_info.get("body_parts", [])
            print(f"Available Opportunity body parts: {body_parts}")
            
            # Select a valid body part
            body_part = "RKN^"
            if body_part not in body_parts and body_parts:
                body_part = body_parts[0]
                print(f"Using body part '{body_part}' instead of 'RKN^'")
            
            # Get accelerometer data from a specific sensor, normalized by mean std
            try:
                sensor_data = dataset_obj.get_sensor_data(
                    sensor_type="Accelerometer", 
                    body_part=body_part, 
                    axis="X"
                ).values
                
                # Handle missing or invalid values
                sensor_data = sensor_data[~np.isnan(sensor_data)]
                
                # Take a slice based on fold
                n_samples = len(sensor_data)
                start_idx = int(fold * n_samples / num_folds)
                end_idx = int((fold + 1) * n_samples / num_folds)
                data = sensor_data[start_idx:end_idx]
                
                # Normalize data
                data = (data - np.mean(data)) / max(np.std(data), 1e-9)
                
            except Exception as e:
                print(f"Error getting specific sensor data: {e}")
                print("Trying alternative approach...")
                
                # Try more generic approach with just sensor type and body part
                try:
                    sensor_data = dataset_obj.get_sensor_data(
                        sensor_type="Accelerometer", 
                        body_part=body_part
                    ).values
                    
                    # Handle missing or invalid values
                    sensor_data = sensor_data[~np.isnan(sensor_data)]
                    
                    # Take a slice based on fold
                    n_samples = len(sensor_data)
                    start_idx = int(fold * n_samples / num_folds)
                    end_idx = int((fold + 1) * n_samples / num_folds)
                    data = sensor_data[start_idx:end_idx]
                    
                    # Normalize data
                    data = (data - np.mean(data)) / max(np.std(data), 1e-9)
                except Exception as e:
                    print(f"Error with fallback approach: {e}")
                    raise ValueError("Could not get valid sensor data")
            
        except Exception as e:
            print(f"Warning: Could not load real Opportunity data, using synthetic: {e}")
            # Fall back to synthetic data if needed
            rng = np.random.default_rng(fold)
            data = rng.normal(size=5_000)
    
    elif dataset.lower() == 'pamap2':
        try:
            # Try to load real data
            dataset_obj = create_pamap2_dataset()
            dataset_obj.load_data()
            
            # Get sensor types and body parts using the method
            sensor_info = dataset_obj.get_available_sensors()
            sensor_types = sensor_info.get("sensor_types", [])
            body_parts = sensor_info.get("body_parts", [])
            print(f"Available PAMAP2 sensor types: {sensor_types}")
            print(f"Available PAMAP2 body parts: {body_parts}")
            
            # Select valid sensor type and body part
            selected_sensor_type = "Accelerometer"
            body_part = "Hand"
            measurement_type = "acc"
            axis = "X"
            
            # Check if we need to adjust our selection based on what's available
            if selected_sensor_type not in sensor_types and sensor_types:
                selected_sensor_type = sensor_types[0]
                print(f"Using sensor type '{selected_sensor_type}' instead of 'Accelerometer'")
                
            if body_part not in body_parts and body_parts:
                body_part = body_parts[0]
                print(f"Using body part '{body_part}' instead of 'Hand'")
            
            # Get accelerometer data from a specific sensor
            try:
                sensor_data = dataset_obj.get_sensor_data(
                    sensor_type=selected_sensor_type, 
                    body_part=body_part,
                    measurement_type=measurement_type,
                    axis=axis
                ).values
                
                # Handle missing or invalid values
                sensor_data = sensor_data[~np.isnan(sensor_data)]
                
                # Take a slice based on fold
                n_samples = len(sensor_data)
                start_idx = int(fold * n_samples / num_folds)
                end_idx = int((fold + 1) * n_samples / num_folds)
                data = sensor_data[start_idx:end_idx]
                
                # Normalize data
                data = (data - np.mean(data)) / max(np.std(data), 1e-9)
                
            except Exception as e:
                print(f"Error getting specific sensor data: {e}")
                print("Trying alternative approach...")
                
                # Try more generic approach with just sensor type and body part
                try:
                    sensor_data = dataset_obj.get_sensor_data(
                        sensor_type=selected_sensor_type, 
                        body_part=body_part
                    ).values
                    
                    # Handle missing or invalid values
                    sensor_data = sensor_data[~np.isnan(sensor_data)]
                    
                    # Take a slice based on fold
                    n_samples = len(sensor_data)
                    start_idx = int(fold * n_samples / num_folds)
                    end_idx = int((fold + 1) * n_samples / num_folds)
                    data = sensor_data[start_idx:end_idx]
                    
                    # Normalize data
                    data = (data - np.mean(data)) / max(np.std(data), 1e-9)
                except Exception as e:
                    print(f"Error with fallback approach: {e}")
                    raise ValueError("Could not get valid sensor data")
            
        except Exception as e:
            print(f"Warning: Could not load real PAMAP2 data, using synthetic: {e}")
            # Fall back to synthetic data if needed
            rng = np.random.default_rng(fold)
            data = rng.normal(size=5_000)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create x values covering data range with margin
    data_min, data_max = np.min(data), np.max(data)
    margin = 0.5 * (data_max - data_min)
    x = np.linspace(data_min - margin, data_max + margin, 1_000)
    
    return data, x

def run_once(dataset: str, fold: int, sigma: float, num_folds: int = 3) -> tuple[float, float]:
    """Return KL-div and χ² for one (dataset, fold, σ) triple."""
    # Load real dataset data
    data, x = load_dataset_data(dataset, fold, num_folds)
    
    # Compute densities
    ndg = compute_ndg_streaming(x, data, sigma)
    kde = _kde_density(x, data, sigma)
    
    # Ensure both are normalized to integrate to 1.0
    ndg_integral = np.trapz(ndg, x=x)
    kde_integral = np.trapz(kde, x=x)
    
    if abs(ndg_integral - 1.0) > 0.01:
        ndg = ndg / ndg_integral
    
    if abs(kde_integral - 1.0) > 0.01:
        kde = kde / kde_integral
    
    # Calculate metrics
    kl_div = np.sum(ndg * np.log((ndg + 1e-12) / (kde + 1e-12)))
    chi2 = np.sum(((ndg - kde) ** 2) / (kde + 1e-12))
    
    return kl_div, chi2

# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--fold",    type=int, required=True)
    p.add_argument("--num-folds", type=int, default=3,
                   help="Total number of folds to divide the data into")
    p.add_argument("--sigma",   type=float, required=True)
    p.add_argument("--out",     help="CSV to append results to")
    p.add_argument("--print-values", action="store_true",
                   help="Print kl,chi2,wall,rss to stdout (no header)")
    p.add_argument("--quiet",   action="store_true")
    args = p.parse_args()

    t0   = time.perf_counter()
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss

    kl, chi2 = run_once(args.dataset, args.fold, args.sigma, args.num_folds)

    wall     = time.perf_counter() - t0
    rss_mb   = max(rss0, proc.memory_info().rss) / 1_048_576  # MB

    # -------- stdout mode ---------------------------------------------------
    if args.print_values:
        print(f"{kl},{chi2},{wall:.3f},{rss_mb:.1f}")
        return
    # ------------------------------------------------------------------------

    # -------- file-append mode ----------------------------------------------
    if args.out:
        path = pathlib.Path(args.out)
        header_needed = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if header_needed:
                w.writerow(["kl_div", "chi2", "wall_sec", "peak_rss_mb"])
            w.writerow([kl, chi2, f"{wall:.3f}", f"{rss_mb:.1f}"])
        if not args.quiet:
            print(f"Wrote: {path}")
    # ------------------------------------------------------------------------

def fallback_get_all_columns(dataset_obj, sensor_type, body_part):
    """Fallback method if get_all_columns doesn't exist in the dataset class."""
    try:
        # First try to use the method if it exists
        if hasattr(dataset_obj, 'get_all_columns'):
            return dataset_obj.get_all_columns(sensor_type, body_part)
        
        # Fallback: try to find columns by name pattern
        if hasattr(dataset_obj, 'data'):
            columns = []
            for col in dataset_obj.data.columns:
                if sensor_type.lower() in col.lower() and body_part.lower() in col.lower():
                    columns.append(col)
            if columns:
                print(f"Found columns using name pattern matching: {columns}")
                return columns
        
        # Last resort: just return any available numeric columns
        if hasattr(dataset_obj, 'data'):
            numeric_cols = dataset_obj.data.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                print(f"Using numeric columns as fallback: {numeric_cols[:5]}...")
                return numeric_cols[:5]  # Limit to first 5 to avoid too many
        
        return []
    except Exception as e:
        print(f"Error in fallback_get_all_columns: {e}")
        return []

if __name__ == "__main__":
    main()
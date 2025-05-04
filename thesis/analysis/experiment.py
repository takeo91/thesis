"""
Experiment runner for similarity metrics evaluation.

This module provides functions for generating test cases,
running experiments, and analyzing the results of fuzzy similarity metrics.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed

# Import from the new module structure
from thesis.core.preprocessing import normalize_data
from thesis.fuzzy.membership import compute_membership_functions
from thesis.fuzzy.similarity import calculate_all_similarity_metrics
from thesis.fuzzy.distributions import compute_fitness_metrics


def generate_case_data(case_number: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate test data for each case scenario.
    
    Args:
        case_number: The test case number (1-6).
        
    Returns:
        Tuple of (sensor_data1, sensor_data2) arrays.
        
    Raises:
        ValueError: If an invalid case number is provided.
    """
    np.random.seed(0)  # For reproducibility
    if case_number == 1:
        # Case 1: Two slightly different ranged signals
        sensor_data1 = np.random.normal(loc=50, scale=5, size=1000)
        sensor_data2 = np.random.normal(loc=55, scale=5, size=1000)
    elif case_number == 2:
        # Case 2: Two completely different signals
        sensor_data1 = np.random.normal(loc=50, scale=5, size=1000)
        sensor_data2 = np.random.uniform(low=0, high=100, size=1000)
    elif case_number == 3:
        # Case 3: One signal's range is a subset of the other
        sensor_data1 = np.random.normal(loc=50, scale=5, size=1000)
        sensor_data2 = np.random.normal(loc=50, scale=2, size=1000)
    elif case_number == 4:
        # Signal A: Bimodal distribution with modes at 30 and 70
        signal_A1 = np.random.normal(loc=30, scale=5, size=500)
        signal_A2 = np.random.normal(loc=70, scale=5, size=500)
        sensor_data1 = np.concatenate([signal_A1, signal_A2])

        # Signal B: Bimodal distribution with modes at 40 and 80
        signal_B1 = np.random.normal(loc=40, scale=5, size=500)
        signal_B2 = np.random.normal(loc=80, scale=5, size=500)
        sensor_data2 = np.concatenate([signal_B1, signal_B2])
    elif case_number == 5:
        # Signal A: Bimodal distribution with modes at 30 and 70
        signal_A1 = np.random.normal(loc=30, scale=5, size=500)
        signal_A2 = np.random.normal(loc=70, scale=5, size=500)
        sensor_data1 = np.concatenate([signal_A1, signal_A2])

        sensor_data2 = np.random.normal(loc=50, scale=5, size=1000)
    elif case_number == 6:
        signal_A1 = np.random.normal(loc=1050, scale=105, size=1000)
        signal_A2 = np.random.normal(loc=550, scale=500, size=1000)
        sensor_data1 = np.concatenate([signal_A1, signal_A2])
        sensor_data2 = np.random.normal(loc=55, scale=5, size=1000)
    else:
        raise ValueError("Invalid case number. Please choose 1-6.")
    return sensor_data1, sensor_data2


def compute_for_single_combination(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute similarity metrics for a single combination of parameters.
    
    Args:
        params: Dictionary containing experiment parameters including case_number,
                method, normalize, sigma_option, and empirical_method.
                
    Returns:
        Dictionary of results including all computed metrics.
    """
    # Unpack parameters
    case_number = params["case_number"]
    method = params["method"]
    normalize = params["normalize"]
    sigma_option = params["sigma_option"]
    empirical_method = params["empirical_method"]

    print(
        f"Running Case {case_number} - Method: {method} - Sigma: {sigma_option} - Normalize: {normalize} - Empirical: {empirical_method}"
    )

    # Generate data
    sensor_data1_orig, sensor_data2_orig = generate_case_data(case_number)

    # Define x_values based on original data range
    x_min = min(np.min(sensor_data1_orig), np.min(sensor_data2_orig))
    x_max = max(np.max(sensor_data1_orig), np.max(sensor_data2_orig))
    x_values_orig = np.linspace(x_min, x_max, 500)

    # Select/Normalize data and x_values for this iteration
    if normalize:
        sensor_data1 = normalize_data(sensor_data1_orig)
        sensor_data2 = normalize_data(sensor_data2_orig)
        x_values = normalize_data(x_values_orig)
    else:
        sensor_data1 = sensor_data1_orig
        sensor_data2 = sensor_data2_orig
        x_values = x_values_orig

    # Compute membership functions
    # Store actual sigma used for sensor 1 to potentially use for sensor 2 and metric1
    mu_s1, actual_sigma_s1 = compute_membership_functions(
        sensor_data1, x_values, method=method, sigma=sigma_option
    )
    # Use the same actual sigma for sensor 2 for better comparison if method is 'nd'
    sigma_for_s2 = actual_sigma_s1 if method == "nd" else None
    mu_s2, actual_sigma_s2 = compute_membership_functions(
        sensor_data2, x_values, method=method, sigma=sigma_for_s2
    )

    # Compute similarity measures using the new function and passing required data
    similarities = calculate_all_similarity_metrics(
        mu_s1,
        mu_s2,
        x_values,
        data_s1=sensor_data1_orig,  # Pass original data for metric1
        data_s2=sensor_data2_orig,  # Pass original data for metric1
        fs_method=method,  # Pass method for potential internal fs construction
        sigma_s1=actual_sigma_s1,  # Pass actual sigma used for s1
        sigma_s2=actual_sigma_s2,  # Pass actual sigma used for s2
    )

    # Compute fitness metrics for sensor 1
    fitness_s1, empirical_probs1 = compute_fitness_metrics(
        sensor_data1, mu_s1, x_values, empirical_method=empirical_method
    )

    # Collect result - Use specific keys from the new similarities dictionary
    result = {
        "Case": case_number,
        "Method": method,
        "Sigma_Option": sigma_option,
        "Sigma_S1": actual_sigma_s1,  # Store potentially different sigmas
        "Sigma_S2": actual_sigma_s2,
        "Normalized": normalize,
        "Empirical_Method": empirical_method,
        "MSE_s1": fitness_s1.get("MSE", np.nan),
        "KL_Divergence_s1": fitness_s1.get("KL_Divergence", np.nan),
        "AIC_s1": fitness_s1.get("AIC", np.nan),
        "BIC_s1": fitness_s1.get("BIC", np.nan),
    }
    
    # Add all similarity metrics to the result dictionary, prefixing keys
    for sim_key, sim_value in similarities.items():
        result[f"Sim_{sim_key}"] = sim_value

    return result


def run_cases_parallel(
    specific_cases: Optional[List[int]] = None, 
    sigma_options: Optional[List[Union[float, str]]] = None, 
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Run experiment cases in parallel.
    
    Args:
        specific_cases: Optional list of case numbers to run. Defaults to all cases.
        sigma_options: Optional list of sigma values to use. Defaults to a predefined set.
        n_jobs: Number of parallel jobs to run. Defaults to -1 (use all cores).
        
    Returns:
        DataFrame containing results for all combinations.
    """
    print("Running cases in parallel...")
    cases_to_run = specific_cases if specific_cases is not None else [1, 2, 3, 4, 5, 6]
    methods = ["nd"]  # Currently only supports 'nd' based on original code
    normalization_options = [False, True]
    default_sigmas = [0.01, 0.1, 0.2, 0.5, "r0.01", "r0.1", "r0.2", "r0.5"]
    sigmas_to_run = sigma_options if sigma_options is not None else default_sigmas
    empirical_method_options = [
        "kde"
    ]  # Currently only supports 'kde' based on original code

    # Create list of all parameter combinations
    param_list = []
    for case in cases_to_run:
        for method in methods:
            for norm in normalization_options:
                for sigma in sigmas_to_run:
                    for emp_method in empirical_method_options:
                        param_list.append(
                            {
                                "case_number": case,
                                "method": method,
                                "normalize": norm,
                                "sigma_option": sigma,
                                "empirical_method": emp_method,
                            }
                        )

    # Run in parallel
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_for_single_combination)(params) for params in param_list
    )

    # Filter out potential None results if errors occurred (optional, depends on error handling)
    results_list = [r for r in results_list if r is not None]

    # Convert to DataFrame
    if not results_list:  # Handle empty results
        print("Warning: No results generated.")
        return pd.DataFrame()
    results_df = pd.DataFrame(results_list)
    return results_df


def plot_membership_functions(
    case_number: int,
    sensor_data1: np.ndarray,
    sensor_data2: np.ndarray,
    x_values: np.ndarray,
    mu_s1: np.ndarray,
    mu_s2: np.ndarray,
    method: str,
    sigma: Union[float, str],
    normalize: bool,
    similarities: Dict[str, float],
) -> None:
    """
    Plot membership functions for visual comparison.
    
    Args:
        case_number: Case number for the title.
        sensor_data1: First sensor data array.
        sensor_data2: Second sensor data array.
        x_values: X-axis values for plotting.
        mu_s1: Membership function values for sensor 1.
        mu_s2: Membership function values for sensor 2.
        method: Method used for membership function generation.
        sigma: Sigma value used.
        normalize: Whether data was normalized.
        similarities: Dictionary of computed similarity values.
    """
    plt.figure(figsize=(14, 8))
    
    # Plot histograms of the sensor data with low alpha for better visibility
    plt.hist(
        sensor_data1,
        bins=50,
        density=True,
        alpha=0.2,
        color="blue",
        label="Sensor 1 Data",
    )
    plt.hist(
        sensor_data2,
        bins=50,
        density=True,
        alpha=0.2,
        color="red",
        label="Sensor 2 Data",
    )
    
    # Plot the membership functions
    plt.plot(x_values, mu_s1, "b-", linewidth=2, label="Membership Function S1")
    plt.plot(x_values, mu_s2, "r-", linewidth=2, label="Membership Function S2")
    
    # Add similarities to the title
    title = f"Case {case_number}: Membership Functions (Method: {method}, Sigma: {sigma}, Normalized: {normalize})\n"
    
    # Add selected similarities
    sim_text = "Similarities: "
    # Format with 4 decimal places for better readability
    for key, value in similarities.items():
        if key in ["Jaccard", "Dice", "Cosine", "Pearson"]:
            sim_text += f"{key}: {value:.4f}, "
    
    plt.title(title + sim_text)
    plt.xlabel("Values" if not normalize else "Normalized Values")
    plt.ylabel("Density / Membership Degree")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensure the save directory exists
    save_dir = "results/plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename based on parameters
    filename = f"{save_dir}/case{case_number}_{method}_sigma{sigma}_norm{normalize}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_results_to_csv(results_df: pd.DataFrame, filename: str = "results/similarity_results.csv") -> None:
    """
    Save results DataFrame to CSV file.
    
    Args:
        results_df: DataFrame containing results.
        filename: Output filename.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def run_experiment(
    specific_cases: Optional[List[int]] = None,
    sigma_options: Optional[List[Union[float, str]]] = None,
    n_jobs: int = -1,
    save_plots: bool = True
) -> pd.DataFrame:
    """
    Main function to run the experiment and save results.
    
    Args:
        specific_cases: Optional list of case numbers to run. Defaults to all cases.
        sigma_options: Optional list of sigma values to use. Defaults to a predefined set.
        n_jobs: Number of parallel jobs to run. Defaults to -1 (use all cores).
        save_plots: Whether to save plots. Defaults to True.
        
    Returns:
        DataFrame containing results for all combinations.
    """
    # Run cases in parallel
    results_df = run_cases_parallel(specific_cases, sigma_options, n_jobs)
    
    # Save results to CSV
    save_results_to_csv(results_df)
    
    # Generate plots if requested
    if save_plots and not results_df.empty:
        print("Generating plots...")
        for _, row in results_df.iterrows():
            case_number = int(row["Case"])
            method = row["Method"]
            sigma = row["Sigma_Option"]
            normalize = row["Normalized"]
            
            # Generate data again for plotting
            sensor_data1, sensor_data2 = generate_case_data(case_number)
            if normalize:
                sensor_data1 = normalize_data(sensor_data1)
                sensor_data2 = normalize_data(sensor_data2)
            
            # Define x_values based on data range
            x_min = min(np.min(sensor_data1), np.min(sensor_data2))
            x_max = max(np.max(sensor_data1), np.max(sensor_data2))
            x_values = np.linspace(x_min, x_max, 500)
            
            # Compute membership functions
            mu_s1, _ = compute_membership_functions(
                sensor_data1, x_values, method=method, sigma=sigma
            )
            mu_s2, _ = compute_membership_functions(
                sensor_data2, x_values, method=method, sigma=sigma
            )
            
            # Extract similarities from results
            similarities = {k.replace("Sim_", ""): v for k, v in row.items() if k.startswith("Sim_")}
            
            # Plot membership functions
            plot_membership_functions(
                case_number,
                sensor_data1,
                sensor_data2,
                x_values,
                mu_s1,
                mu_s2,
                method,
                sigma,
                normalize,
                similarities
            )
    
    return results_df 
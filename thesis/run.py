import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Updated imports to reflect the new module structure
from thesis.membership_functions import compute_membership_functions
from thesis.similarity_metrics import (
    calculate_all_similarity_metrics,
)  # Renamed function
from thesis.fitness_evaluation import compute_fitness_metrics
from thesis.preprocessing import normalize_data
from joblib import Parallel, delayed


# Function to generate data for each case
def generate_case_data(case_number):
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
        raise ValueError("Invalid case number. Please choose 1, 2, or 3.")
    return sensor_data1, sensor_data2


# Function to plot results for a case (kept for reference, might need update based on which similarities to show)
# def plot_case_results(case_number, sensor_data1, sensor_data2, x_values_dict, mu_values_dict, similarities_dict):
#     # ... (plotting code - would need updating keys in similarities_dict) ...

# Function to plot results for a case (kept for reference, might need update based on which similarities to show)
# def plot_case_results_sigma(case_number, sigma, sensor_data1, sensor_data2, x_values_dict, mu_values_dict, similarities_dict):
#     # ... (plotting code - would need updating keys in similarities_dict) ...


def compute_for_single_combination(params):
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
        "Similarity_Jaccard": similarities.get(
            "Jaccard", np.nan
        ),  # Example: Get Jaccard
        "Distance_Euclidean": similarities.get(
            "Distance_Euclidean", np.nan
        ),  # Example: Get Euclidean Distance
        "MATLAB_Metric1": similarities.get(
            "MATLAB_Metric1", np.nan
        ),  # Example: Get MATLAB Metric 1
        # Add other desired similarity keys here...
        "MSE_s1": fitness_s1.get("MSE", np.nan),
        "KL_Divergence_s1": fitness_s1.get("KL_Divergence", np.nan),
        "AIC_s1": fitness_s1.get("AIC", np.nan),
        "BIC_s1": fitness_s1.get("BIC", np.nan),
    }
    # Add all similarity metrics to the result dictionary, prefixing keys
    for sim_key, sim_value in similarities.items():
        result[f"Sim_{sim_key}"] = sim_value

    return result


def run_cases_parallel(specific_cases=None, sigma_options=None, n_jobs=-1):
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


def run_cases(specific_cases=None, sigma_options=None):
    print("Running all cases sequentially...")
    results = []
    cases_to_run = specific_cases if specific_cases is not None else [1, 2, 3, 4, 5, 6]
    methods = ["nd"]  # Currently only supports 'nd' based on original code
    normalization_options = [False, True]
    default_sigmas = [0.01, 0.1, "r0.01", "r0.1"]
    sigmas_to_run = sigma_options if sigma_options is not None else default_sigmas
    empirical_method_options = [
        "kde"
    ]  # Currently only supports 'kde' based on original code

    for case_number in cases_to_run:
        # Generate original data for the case
        sensor_data1_orig, sensor_data2_orig = generate_case_data(case_number)

        # --- Calculate Scale Features from Original Data ---
        std_orig_s1 = np.std(sensor_data1_orig)
        range_orig_s1 = np.ptp(sensor_data1_orig)  # Peak-to-peak (max - min)
        std_orig_s2 = np.std(sensor_data2_orig)
        range_orig_s2 = np.ptp(sensor_data2_orig)
        # ---------------------------------------------------

        # Define x_values covering both signals based on original data range
        x_min_orig = min(np.min(sensor_data1_orig), np.min(sensor_data2_orig))
        x_max_orig = max(np.max(sensor_data1_orig), np.max(sensor_data2_orig))
        x_values_orig = np.linspace(x_min_orig, x_max_orig, 500)

        for method in methods:
            for normalize in normalization_options:
                for sigma_option in sigmas_to_run:
                    for empirical_method in empirical_method_options:
                        print(
                            f"Running Case {case_number} - Method: {method} - Sigma: {sigma_option} - Normalize: {normalize} - Empirical: {empirical_method}"
                        )

                        # Select/Normalize data and x_values for this iteration
                        if normalize:
                            sensor_data1 = normalize_data(sensor_data1_orig)
                            sensor_data2 = normalize_data(sensor_data2_orig)
                            # Use fixed [0, 1] range for normalized x_values
                            x_values = np.linspace(0, 1, 500)
                        else:
                            sensor_data1 = sensor_data1_orig
                            sensor_data2 = sensor_data2_orig
                            x_values = x_values_orig

                        # Compute membership functions
                        try:
                            # Store actual sigma used for sensor 1
                            mu_s1, actual_sigma_s1 = compute_membership_functions(
                                sensor_data1,
                                x_values,
                                method=method,
                                sigma=sigma_option,
                            )
                            # Use the same actual sigma for sensor 2 if method is 'nd'
                            sigma_for_s2 = actual_sigma_s1 if method == "nd" else None
                            mu_s2, actual_sigma_s2 = compute_membership_functions(
                                sensor_data2,
                                x_values,
                                method=method,
                                sigma=sigma_for_s2,
                            )

                            # Compute similarity measures using the new function
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
                                sensor_data1,
                                mu_s1,
                                x_values,
                                empirical_method=empirical_method,
                            )

                            # Compute residuals for sensor 1
                            residuals_s1 = (
                                empirical_probs1 - mu_s1
                                if empirical_probs1 is not None
                                else np.full_like(mu_s1, np.nan)
                            )

                            # Collect results
                            result = {
                                "Case": case_number,
                                "Method": method,
                                "Sigma_Option": sigma_option,
                                "Sigma_S1": actual_sigma_s1,  # Store potentially different sigmas
                                "Sigma_S2": actual_sigma_s2,
                                "Normalized": normalize,
                                "Empirical_Method": empirical_method,
                                # Scale Features (Original Data)
                                "Std_Orig_S1": std_orig_s1,
                                "Range_Orig_S1": range_orig_s1,
                                "Std_Orig_S2": std_orig_s2,
                                "Range_Orig_S2": range_orig_s2,
                                # Fitness metrics for sensor 1
                                "MSE_s1": fitness_s1.get("MSE", np.nan),
                                "KL_Divergence_s1": fitness_s1.get(
                                    "KL_Divergence", np.nan
                                ),
                                "AIC_s1": fitness_s1.get("AIC", np.nan),
                                "BIC_s1": fitness_s1.get("BIC", np.nan),
                            }
                            # Add all similarity metrics to the result dictionary, prefixing keys
                            for sim_key, sim_value in similarities.items():
                                result[f"Sim_{sim_key}"] = sim_value

                            results.append(result)

                            # Plot membership functions and residuals (saving to file)
                            if (
                                empirical_probs1 is not None
                            ):  # Only plot if empirical probs were calculated
                                plot_membership_functions_with_residuals(
                                    case_number=case_number,
                                    sensor_data=sensor_data1,  # Plotting for sensor 1 vs empirical
                                    x_values=x_values,
                                    mu_s=mu_s1,
                                    empirical_probs=empirical_probs1,
                                    residuals=residuals_s1,
                                    method=method,
                                    sigma_option=sigma_option,
                                    sigma=actual_sigma_s1,  # Use sigma from S1 for plot label consistency
                                    normalize=normalize,
                                    empirical_method=empirical_method,
                                    similarities=similarities,  # Pass all similarities
                                    fitness_metrics=fitness_s1,
                                )

                        except Exception as e:
                            print(
                                f"Error processing Case {case_number}, Method {method}, Sigma {sigma_option}, Norm {normalize}, Emp {empirical_method}: {e}"
                            )
                            results.append(
                                {
                                    "Case": case_number,
                                    "Method": method,
                                    "Sigma_Option": sigma_option,
                                    "Sigma_S1": None,
                                    "Sigma_S2": None,  # Sigmas might not be available on error
                                    "Normalized": normalize,
                                    "Empirical_Method": empirical_method,
                                    # Add scale features even on error if possible
                                    "Std_Orig_S1": std_orig_s1,
                                    "Range_Orig_S1": range_orig_s1,
                                    "Std_Orig_S2": std_orig_s2,
                                    "Range_Orig_S2": range_orig_s2,
                                    "Error": str(e),
                                }
                            )

    # Convert results to DataFrame
    if not results:  # Handle empty results
        print("Warning: No results generated.")
        return pd.DataFrame()
    results_df = pd.DataFrame(results)
    return results_df


def plot_membership_functions(
    case_number,
    sensor_data1,
    sensor_data2,
    x_values,
    mu_s1,
    mu_s2,
    method,
    sigma,
    normalize,
    similarities,
):
    """
    Plot membership functions for two sensors.
    Uses specific keys from the similarities dictionary.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, mu_s1, label="Sensor 1")
    plt.plot(x_values, mu_s2, label="Sensor 2")
    plt.fill_between(
        x_values, np.minimum(mu_s1, mu_s2), alpha=0.3, color="gray", label="Overlap"
    )

    plt.title(
        f"Case {case_number} - Method: {method.upper()} - Normalized: {normalize} - Sigma: {str(sigma)}"
    )
    plt.xlabel("x")
    plt.ylabel("Membership Degree")
    plt.legend()

    # Prepare the text with specific similarity measures
    overlap_sim = similarities.get(
        "OverlapCoefficient", np.nan
    )  # Use OverlapCoefficient as 'similarity_overlap'
    euclidean_dist = similarities.get(
        "Distance_Euclidean", np.nan
    )  # Use 'Distance_Euclidean'

    similarity_text = (
        f"Similarity (Overlap): {overlap_sim:.4f}\n"
        f"Euclidean Distance: {euclidean_dist:.4f}"
    )
    # Position the text box
    plt.text(
        0.95,
        0.95,
        similarity_text,
        horizontalalignment="right",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
    )

    plt.show()
    plt.close()  # Close plot after showing


def plot_membership_functions_with_residuals(
    case_number,
    sensor_data,
    x_values,
    mu_s,
    empirical_probs,
    residuals,
    method,
    sigma_option,
    sigma,
    normalize,
    empirical_method,
    similarities,  # Dictionary with all similarities
    fitness_metrics,  # Dictionary with fitness metrics
):
    """
    Plots the membership function, empirical probabilities, and residuals.
    Saves the plot to a file organized by case number, sigma, and normalization.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"Case {case_number} - Method: {method} - Sigma Option: {sigma_option} (Actual: {sigma:.4f}) - Norm: {normalize} - Emp: {empirical_method}",
        fontsize=14,
    )

    # Plot membership function and empirical probabilities
    axes[0].plot(x_values, mu_s, label="Membership Function (μ_s)", color="blue")
    axes[0].plot(
        x_values,
        empirical_probs,
        label=f"Empirical Prob ({empirical_method})",
        color="red",
        linestyle="--",
    )
    axes[0].fill_between(x_values, mu_s, alpha=0.2, color="blue")
    axes[0].set_title("Membership Function vs. Empirical Probability")
    axes[0].set_ylabel("Probability / Membership")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)

    # Add text box with key fitness metrics to the first subplot
    fitness_text = (
        f"MSE: {fitness_metrics.get('MSE', np.nan):.4f}\n"
        f"KL Div: {fitness_metrics.get('KL_Divergence', np.nan):.4f}\n"
        f"AIC: {fitness_metrics.get('AIC', np.nan):.2f}\n"
        f"BIC: {fitness_metrics.get('BIC', np.nan):.2f}"
    )
    axes[0].text(
        0.05,
        0.95,
        fitness_text,
        horizontalalignment="left",
        verticalalignment="top",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    # Plot residuals
    axes[1].plot(
        x_values, residuals, label="Residuals (Empirical - μ_s)", color="green"
    )
    axes[1].axhline(0, color="grey", linestyle=":")
    axes[1].set_title("Residuals")
    axes[1].set_xlabel("x values (Normalized)" if normalize else "x values")
    axes[1].set_ylabel("Residual")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust layout to make room for suptitle

    # --- Save plot with organized directory structure ---
    # Sanitize sigma_option for directory/filename
    sigma_option_str = str(sigma_option).replace(".", "p")
    norm_str = "norm" if normalize else "no_norm"

    # Define output directory based on sigma_option and normalization
    output_dir = os.path.join(
        "plots", f"sigma_{sigma_option_str}", norm_str, f"case_{case_number}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Define filename based on method and empirical method
    filename = f"{method}_emp_{empirical_method}.png"
    filepath = os.path.join(output_dir, filename)

    # Save the figure
    plt.savefig(filepath)
    # print(f"Plot saved to: {filepath}") # Optional: print save location

    # Close the figure to free memory
    plt.close(fig)


# --- Example usage (often run from notebook) ---
if __name__ == "__main__":
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # Run sequentially for easier debugging and plotting generation initially
    results_dataframe = run_cases()
    # Or run in parallel (might suppress plotting or require adjustments)
    # results_dataframe = run_cases_parallel()

    print("Finished running cases.")
    print(results_dataframe.head())
    # Save results to CSV
    results_dataframe.to_csv("run_cases_results.csv", index=False)
    print("Results saved to run_cases_results.csv")

"""
fitness_evaluation.py

Functions for evaluating the fitness of a membership function
against empirical sensor data using various statistical metrics.
"""
import numpy as np
from scipy.stats import gaussian_kde, chisquare, iqr
from scipy.interpolate import interp1d
from sklearn.model_selection import TimeSeriesSplit

# ==============================================================================
# Empirical Distribution & Fitness Evaluation Helpers
# ==============================================================================

def compute_empirical_distribution_kde(x_values, data):
    """
    Computes empirical probability distribution using Gaussian Kernel Density Estimation (KDE).

    Args:
        x_values (np.ndarray): Points at which to evaluate the KDE.
        data (np.ndarray): Input data points.

    Returns:
        np.ndarray: Normalized empirical probabilities evaluated at x_values.
    """
    data = np.asarray(data)
    x_values = np.asarray(x_values)
    if data.size < 2: # KDE requires at least 2 points
        return np.zeros_like(x_values)

    try:
        kde = gaussian_kde(data)
        empirical_probs = kde.evaluate(x_values)
        # Ensure non-negative probabilities and normalize
        empirical_probs = np.clip(empirical_probs, 0, None)
        total_prob = np.sum(empirical_probs)
        if total_prob > 1e-9: # Avoid division by zero
            empirical_probs /= total_prob
        else:
            empirical_probs = np.zeros_like(empirical_probs)
    except (np.linalg.LinAlgError, ValueError): # Handle singular matrix or other KDE errors
        empirical_probs = np.zeros_like(x_values)

    return empirical_probs

def compute_empirical_distribution_counts(x_values, data):
    """
    Computes empirical probability density using histogram counts.
    Note: This estimates density; normalization might be needed depending on usage.
          Returns densities at bin centers.

    Args:
        x_values (np.ndarray): Bin edges for the histogram.
        data (np.ndarray): Input data points.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing (empirical densities, bin centers).
                                         Returns (empty, empty) if histogram cannot be computed.
    """
    data = np.asarray(data)
    x_values = np.asarray(x_values)
    if data.size == 0 or x_values.size < 2:
        return np.array([]), np.array([]) # Cannot compute histogram

    try:
        empirical_counts, bin_edges = np.histogram(data, bins=x_values, density=False)
        bin_widths = np.diff(bin_edges)

        # Avoid division by zero if sum of counts or bin_widths are zero
        total_counts = np.sum(empirical_counts)
        if total_counts == 0 or np.any(bin_widths <= 1e-9):
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            return np.zeros_like(bin_centers), bin_centers

        # Calculate density
        empirical_densities = empirical_counts / (total_counts * bin_widths)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return empirical_densities, bin_centers

    except ValueError: # Handle issues with histogram computation
        return np.array([]), np.array([])


def compute_error_metrics(empirical_probs, theoretical_probs):
    """
    Computes error metrics (MSE, RMSE, MAE) between two probability distributions.

    Args:
        empirical_probs (np.ndarray): Empirical probability distribution.
        theoretical_probs (np.ndarray): Theoretical probability distribution.

    Returns:
        dict: Dictionary containing 'MSE', 'RMSE', 'MAE'.
    """
    empirical_probs = np.asarray(empirical_probs)
    theoretical_probs = np.asarray(theoretical_probs)

    if empirical_probs.shape != theoretical_probs.shape:
        raise ValueError("Shapes of empirical and theoretical probabilities must match.")

    errors = empirical_probs - theoretical_probs
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

def compute_kl_divergence(empirical_probs, theoretical_probs):
    """
    Computes Kullback-Leibler divergence D_KL(empirical || theoretical).

    Args:
        empirical_probs (np.ndarray): Empirical probability distribution.
        theoretical_probs (np.ndarray): Theoretical probability distribution.

    Returns:
        float: KL divergence value. Returns NaN if calculation fails.
    """
    empirical_probs = np.asarray(empirical_probs)
    theoretical_probs = np.asarray(theoretical_probs)

    # Ensure inputs are valid probability distributions
    sum_empirical = np.sum(empirical_probs)
    sum_theoretical = np.sum(theoretical_probs)

    if sum_empirical < 1e-9 or sum_theoretical < 1e-9:
        return np.nan # Cannot compute divergence if one distribution is zero

    empirical_probs = empirical_probs / sum_empirical
    theoretical_probs = theoretical_probs / sum_theoretical

    # Clip values to avoid log(0) or division by zero
    epsilon = 1e-10
    empirical_probs = np.clip(empirical_probs, epsilon, 1)
    theoretical_probs = np.clip(theoretical_probs, epsilon, 1)

    # Re-normalize after clipping just in case clipping changed sums significantly
    empirical_probs /= np.sum(empirical_probs)
    theoretical_probs /= np.sum(theoretical_probs)

    kl_divergence = np.sum(empirical_probs * np.log(empirical_probs / theoretical_probs))
    return kl_divergence

def compute_chi_squared_test(empirical_counts, theoretical_probs, sample_size):
    """
    Performs a Chi-squared goodness-of-fit test.

    Args:
        empirical_counts (np.ndarray): Observed counts in each bin.
        theoretical_probs (np.ndarray): Expected probabilities for each bin.
        sample_size (int): Total number of observations.

    Returns:
        tuple[float, float]: Tuple containing (Chi-squared statistic, p-value).
                             Returns (NaN, NaN) if test cannot be performed.
    """
    expected_counts = np.asarray(theoretical_probs) * sample_size
    empirical_counts = np.asarray(empirical_counts)

    # Ensure shapes match before filtering
    if empirical_counts.shape != expected_counts.shape:
         print("Warning: empirical_counts shape does not match theoretical_probs shape in Chi2 test.")
         return np.nan, np.nan

    # Filter out bins with zero or very low expected count to avoid errors
    valid_indices = expected_counts > 1e-9 # Or a higher threshold like 1 or 5 depending on preference
    if np.sum(valid_indices) < 2: # Need at least 2 degrees of freedom for the test
         return np.nan, np.nan

    # Perform test on filtered data
    try:
        chi2_stat, p_value = chisquare(
            f_obs=empirical_counts[valid_indices],
            f_exp=expected_counts[valid_indices]
        )
        return chi2_stat, p_value
    except ValueError as e: # Handle potential errors during chisquare computation
        print(f"Warning: Chi-squared test failed. Error: {e}")
        return np.nan, np.nan

def compute_information_criteria(signal_data, theoretical_pdf, x_values, num_params):
    """
    Computes Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Args:
        signal_data (np.ndarray): Original sensor data points.
        theoretical_pdf (np.ndarray): Theoretical probability *density* function values.
        x_values (np.ndarray): Domain corresponding to theoretical_pdf.
        num_params (int): Number of parameters estimated for the theoretical PDF.

    Returns:
        dict: Dictionary containing 'AIC' and 'BIC'. Returns Inf if log-likelihood cannot be computed.
    """
    signal_data = np.asarray(signal_data)
    theoretical_pdf = np.asarray(theoretical_pdf)
    x_values = np.asarray(x_values)
    n = len(signal_data)

    if n == 0 or theoretical_pdf.size == 0 or x_values.size < 2: # Need at least 2 x_values for trapz
         return {'AIC': np.inf, 'BIC': np.inf} # Cannot compute on empty data

    # Normalize the theoretical PDF to integrate to 1
    try:
        pdf_area = np.trapz(theoretical_pdf, x_values)
    except ValueError:
        return {'AIC': np.inf, 'BIC': np.inf} # Trapz fails if x_values not sorted or too few points

    if pdf_area < 1e-9:
        # If area is near zero, likelihood will be near zero -> log likelihood -> -inf
        # Return Inf for AIC/BIC in this case.
        return {'AIC': np.inf, 'BIC': np.inf}
    theoretical_pdf_normalized = theoretical_pdf / pdf_area

    # Interpolate the normalized PDF to get likelihood at each data point
    try:
        interp_pdf = interp1d(x_values, theoretical_pdf_normalized, kind='linear', bounds_error=False, fill_value=0)
        probabilities = interp_pdf(signal_data)
    except ValueError:
        return {'AIC': np.inf, 'BIC': np.inf} # Interpolation failed

    # Clip probabilities to avoid log(0)
    epsilon = 1e-10
    probabilities = np.clip(probabilities, epsilon, None)

    # Compute log-likelihood
    log_likelihood = np.sum(np.log(probabilities))

    # Compute AIC and BIC
    aic = 2 * num_params - 2 * log_likelihood
    bic = num_params * np.log(n) - 2 * log_likelihood

    return {'AIC': aic, 'BIC': bic}

def time_series_cross_validation(signal_data, theoretical_pdf, x_values, num_splits=5):
    """
    Performs time series cross-validation and computes average MSE and KL Divergence.

    Args:
        signal_data (np.ndarray): Original time series data.
        theoretical_pdf (np.ndarray): Theoretical probability *density* function.
        x_values (np.ndarray): Domain corresponding to theoretical_pdf.
        num_splits (int): Number of folds for TimeSeriesSplit.

    Returns:
        dict: Dictionary containing 'MSE_CV' and 'KL_Divergence_CV'.
    """
    signal_data = np.asarray(signal_data)
    theoretical_pdf = np.asarray(theoretical_pdf)
    x_values = np.asarray(x_values)

    if signal_data.size < num_splits + 1 or theoretical_pdf.size == 0 or x_values.size < 2:
        # Not enough data for the specified splits or other inputs empty
        return {'MSE_CV': np.nan, 'KL_Divergence_CV': np.nan}

    # Normalize theoretical PDF
    try:
        pdf_area = np.trapz(theoretical_pdf, x_values)
    except ValueError:
         return {'MSE_CV': np.nan, 'KL_Divergence_CV': np.nan}

    if pdf_area < 1e-9:
        return {'MSE_CV': np.nan, 'KL_Divergence_CV': np.nan} # PDF is essentially zero
    theoretical_pdf_normalized = theoretical_pdf / pdf_area

    mse_list = []
    kl_divergence_list = []
    try:
        tscv = TimeSeriesSplit(n_splits=num_splits)
        splits = list(tscv.split(signal_data)) # Get splits upfront
    except ValueError as e:
         print(f"Warning: Could not perform TimeSeriesSplit. Error: {e}")
         return {'MSE_CV': np.nan, 'KL_Divergence_CV': np.nan}

    if not splits: # Handle case where split fails silently
         return {'MSE_CV': np.nan, 'KL_Divergence_CV': np.nan}

    for train_index, test_index in splits:
        if len(test_index) == 0: continue # Skip empty test sets
        test_data = signal_data[test_index]

        # Empirical distribution from test set using KDE
        empirical_probs = compute_empirical_distribution_kde(x_values, test_data)
        if np.sum(empirical_probs) < 1e-9: continue # Skip if empirical distribution is zero

        # Theoretical probabilities (already normalized PDF) evaluated on x_values
        # We compare the empirical distribution on x_values with the theoretical one on x_values
        theoretical_probs = theoretical_pdf_normalized
        if np.sum(theoretical_probs) < 1e-9: continue # Skip if theoretical is zero

        # Compute metrics for this fold
        try:
            mse = compute_error_metrics(empirical_probs, theoretical_probs)['MSE']
        except ValueError:
            mse = np.nan # Handle shape mismatch just in case

        kl = compute_kl_divergence(empirical_probs, theoretical_probs)

        if not np.isnan(mse): mse_list.append(mse)
        if not np.isnan(kl): kl_divergence_list.append(kl)

    # Calculate average metrics
    avg_mse = np.mean(mse_list) if mse_list else np.nan
    avg_kl = np.mean(kl_divergence_list) if kl_divergence_list else np.nan

    return {'MSE_CV': avg_mse, 'KL_Divergence_CV': avg_kl}

# ==============================================================================
# Main Fitness Metrics Calculation Function
# ==============================================================================

def compute_fitness_metrics(sensor_data, mu, x_values, num_params=2, empirical_method='kde'):
    """
    Computes various fitness metrics for a given membership function against sensor data.

    Args:
        sensor_data (np.ndarray): The original sensor data.
        mu (np.ndarray): The computed membership function (assumed to be density, will be normalized).
        x_values (np.ndarray): The domain corresponding to the membership function.
        num_params (int, optional): Number of parameters used to generate 'mu'. Defaults to 2.
        empirical_method (str, optional): Method to compute empirical distribution ('kde' or 'counts'). Defaults to 'kde'.

    Returns:
        tuple[dict, np.ndarray or None]:
            - fitness_metrics: Dictionary containing all computed fitness metrics (MSE, RMSE, MAE, KL, AIC, BIC, CV metrics).
            - empirical_probs: The computed empirical probability distribution used for comparison (or None if failed).
    """
    sensor_data = np.asarray(sensor_data)
    mu = np.asarray(mu)
    x_values = np.asarray(x_values)

    # Basic validation for empty inputs
    if sensor_data.size == 0 or mu.size == 0 or x_values.size == 0:
         default_metrics = {
              'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'KL_Divergence': np.nan,
              'AIC': np.nan, 'BIC': np.nan, 'MSE_CV': np.nan, 'KL_Divergence_CV': np.nan
              # Add NaN for Chi2 if implemented later
         }
         return default_metrics, None

    # --- 1. Compute Empirical Probability Distribution ---
    empirical_probs = None
    empirical_counts = None # Needed for chi-squared (if implemented)

    if empirical_method == 'kde':
        empirical_probs = compute_empirical_distribution_kde(x_values, sensor_data)
    elif empirical_method == 'counts':
        # Using counts directly for chi-squared, need probs for other metrics
        # Let's compute both if method is 'counts'
        # Requires bins defined by x_values.
        # Get counts first
        try:
            empirical_counts, _ = np.histogram(sensor_data, bins=x_values, density=False)
        except ValueError:
            empirical_counts = None # Histogram failed

        # Need probabilities on x_values for error/KL metrics - use KDE as fallback for now
        # Ideally, derive probabilities directly from counts/bins if possible
        print("Warning: 'counts' method currently uses KDE for Error/KL metrics. Chi-squared uses histogram counts.")
        empirical_probs = compute_empirical_distribution_kde(x_values, sensor_data)

    else:
        raise ValueError("Unknown empirical method. Use 'kde' or 'counts'.")

    if empirical_probs is None or np.sum(empirical_probs) < 1e-9:
         print("Warning: Could not compute valid empirical probability distribution.")
         # Return NaNs if empirical calculation failed
         default_metrics = {key: np.nan for key in ['MSE', 'RMSE', 'MAE', 'KL_Divergence', 'AIC', 'BIC', 'MSE_CV', 'KL_Divergence_CV', 'Chi2_Stat', 'Chi2_PValue']}
         return default_metrics, None

    # --- 2. Prepare Theoretical Probability Distribution ---
    # Normalize the input membership function 'mu' to act as theoretical probabilities
    sum_mu = np.sum(mu)
    if sum_mu < 1e-9:
        print("Warning: Theoretical membership function 'mu' sums to zero.")
        theoretical_probs = np.zeros_like(mu)
    else:
        theoretical_probs = mu / sum_mu

    # --- 3. Compute Metrics ---
    try:
        error_metrics = compute_error_metrics(empirical_probs, theoretical_probs)
    except ValueError as e:
        print(f"Error computing error metrics: {e}")
        error_metrics = {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan}

    kl_divergence = compute_kl_divergence(empirical_probs, theoretical_probs)

    # AIC/BIC require the unnormalized density 'mu'
    info_criteria = compute_information_criteria(sensor_data, mu, x_values, num_params)

    # Cross-validation also uses the unnormalized density 'mu'
    cv_results = time_series_cross_validation(sensor_data, mu, x_values)

    # Chi-squared (Requires empirical counts aligned with theoretical_probs)
    chi2_stat, p_value = (np.nan, np.nan) # Default to NaN
    if empirical_counts is not None and len(empirical_counts) == len(theoretical_probs):
        # Ensure counts are available and shapes match theoretical probabilities
        chi2_stat, p_value = compute_chi_squared_test(empirical_counts, theoretical_probs, len(sensor_data))
    elif empirical_method == 'counts':
        print("Warning: Could not compute Chi-squared test due to missing/mismatched counts.")


    # --- 4. Combine Results ---
    fitness_metrics = {
        'MSE': error_metrics['MSE'],
        'RMSE': error_metrics['RMSE'],
        'MAE': error_metrics['MAE'],
        'KL_Divergence': kl_divergence,
        'AIC': info_criteria['AIC'],
        'BIC': info_criteria['BIC'],
        'MSE_CV': cv_results['MSE_CV'],
        'KL_Divergence_CV': cv_results['KL_Divergence_CV'],
        'Chi2_Stat': chi2_stat,
        'Chi2_PValue': p_value
    }

    return fitness_metrics, empirical_probs
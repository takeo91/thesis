"""
distribution.py

Provides functions for:
1. Computing membership functions from sensor data (NDG, KDE).
2. Calculating various fuzzy set similarity metrics.
3. Evaluating the fitness of membership functions against empirical data.
4. Data preprocessing utilities (normalization, standardization).
"""

import numpy as np
from scipy.stats import gaussian_kde, chisquare, iqr
from scipy.interpolate import interp1d
from sklearn.model_selection import TimeSeriesSplit

# ==============================================================================
# Empirical Distribution & Fitness Evaluation Helpers
# (Error metrics, KL Divergence, Information Criteria, Cross-Validation)
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

    Args:
        x_values (np.ndarray): Bin edges for the histogram.
        data (np.ndarray): Input data points.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing (empirical densities, bin centers).
                                         Returns (zeros, zeros) if histogram cannot be computed.
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

    # Filter out bins with zero or very low expected count to avoid errors
    valid_indices = expected_counts > 1e-9 # Or a higher threshold like 1 or 5 depending on preference
    if np.sum(valid_indices) < 2: # Need at least 2 bins for the test
         return np.nan, np.nan

    # Ensure observed counts match expected counts shape after filtering
    if empirical_counts.shape != expected_counts.shape:
         # This case should ideally not happen if inputs are prepared correctly
         return np.nan, np.nan

    try:
        chi2_stat, p_value = chisquare(
            f_obs=empirical_counts[valid_indices],
            f_exp=expected_counts[valid_indices]
        )
        return chi2_stat, p_value
    except ValueError: # Handle potential errors during chisquare computation
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

    if n == 0 or theoretical_pdf.size == 0 or x_values.size == 0:
         return {'AIC': np.inf, 'BIC': np.inf} # Cannot compute on empty data

    # Normalize the theoretical PDF to integrate to 1
    pdf_area = np.trapz(theoretical_pdf, x_values)
    if pdf_area < 1e-9:
        return {'AIC': np.inf, 'BIC': np.inf} # Avoid division by zero, PDF is essentially zero
    theoretical_pdf_normalized = theoretical_pdf / pdf_area

    # Interpolate the normalized PDF to get likelihood at each data point
    interp_pdf = interp1d(x_values, theoretical_pdf_normalized, kind='linear', bounds_error=False, fill_value=0)
    probabilities = interp_pdf(signal_data)

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

    if signal_data.size < num_splits + 1 or theoretical_pdf.size == 0 or x_values.size == 0:
        # Not enough data for the specified splits or other inputs empty
        return {'MSE_CV': np.nan, 'KL_Divergence_CV': np.nan}

    # Normalize theoretical PDF
    pdf_area = np.trapz(theoretical_pdf, x_values)
    if pdf_area < 1e-9:
        return {'MSE_CV': np.nan, 'KL_Divergence_CV': np.nan} # PDF is essentially zero
    theoretical_pdf_normalized = theoretical_pdf / pdf_area

    mse_list = []
    kl_divergence_list = []
    tscv = TimeSeriesSplit(n_splits=num_splits)

    for train_index, test_index in tscv.split(signal_data):
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
        mse = compute_error_metrics(empirical_probs, theoretical_probs)['MSE']
        kl = compute_kl_divergence(empirical_probs, theoretical_probs)

        if not np.isnan(mse): mse_list.append(mse)
        if not np.isnan(kl): kl_divergence_list.append(kl)

    # Calculate average metrics
    avg_mse = np.mean(mse_list) if mse_list else np.nan
    avg_kl = np.mean(kl_divergence_list) if kl_divergence_list else np.nan

    return {'MSE_CV': avg_mse, 'KL_Divergence_CV': avg_kl}

# ==============================================================================
# Membership Function Computation
# ==============================================================================

def compute_ndg(x_values, sensor_data, sigma):
    """
    Computes Neighbor Density Graph (NDG) values. Helper for compute_membership_function.

    Args:
        x_values (np.ndarray): Points at which to calculate density.
        sensor_data (np.ndarray): Input data points.
        sigma (float): Bandwidth parameter for the Gaussian kernel.

    Returns:
        np.ndarray: NDG values corresponding to x_values.
    """
    x_values = np.asarray(x_values)[:, np.newaxis] # Shape: (len(x_values), 1)
    sensor_data = np.asarray(sensor_data)[np.newaxis, :] # Shape: (1, len(sensor_data))
    if sensor_data.size == 0: return np.zeros(x_values.shape[0]) # Handle empty data

    squared_diffs = (x_values - sensor_data) ** 2
    # Use a minimum sigma to prevent division by zero
    safe_sigma_sq = max(sigma ** 2, 1e-18)
    exponentials = np.exp(-squared_diffs / safe_sigma_sq)
    ndg = exponentials.sum(axis=1)
    return ndg

def compute_membership_function(sensor_data, x_values=None, sigma=None, num_points=500):
    """
    Computes a normalized membership function using the Neighbor Density Graph method.

    Args:
        sensor_data (np.ndarray): Input data points.
        x_values (np.ndarray, optional): Domain for the membership function.
                                         If None, calculated from data range. Defaults to None.
        sigma (float or str, optional): Bandwidth parameter. If None, uses 0.1 * data range.
                                        If str like 'r0.2', uses 0.2 * data range. Defaults to None.
        num_points (int, optional): Number of points for x_values if automatically calculated. Defaults to 500.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: Tuple containing (x_values, mu_s, sigma_val).
                                                mu_s is the normalized membership function.
                                                sigma_val is the actual sigma value used.
    """
    sensor_data = np.asarray(sensor_data)
    if sensor_data.size == 0:
        # Handle empty data gracefully
        if x_values is None: x_values = np.linspace(0, 1, num_points) # Default range
        else: x_values = np.asarray(x_values)
        # Return zeros for mu, default x_values, and a default sigma estimate (e.g., 0.1)
        return x_values, np.zeros_like(x_values), 0.1

    x_min, x_max = np.min(sensor_data), np.max(sensor_data)
    data_range = x_max - x_min

    # Define x_values if not provided
    if x_values is None:
        # Handle case where data is constant (range is zero)
        center = x_min if data_range < 1e-9 else (x_min + x_max) / 2
        spread = 1.0 if data_range < 1e-9 else data_range # Use a default spread if range is zero
        x_values = np.linspace(center - spread, center + spread, num_points)
    else:
        x_values = np.asarray(x_values)

    # Determine sigma value to use
    default_sigma_ratio = 0.1
    if sigma is None:
        sigma_val = default_sigma_ratio * data_range if data_range > 1e-9 else default_sigma_ratio
    elif isinstance(sigma, str) and sigma.startswith('r'):
        try:
            ratio = float(sigma[1:])
            sigma_val = ratio * data_range if data_range > 1e-9 else ratio
        except ValueError:
            raise ValueError(f"Invalid sigma string format: {sigma}. Expected 'r<float>'.")
    else:
        try:
            sigma_val = float(sigma) # Assume numeric if not string 'r...'
        except ValueError:
            raise ValueError(f"Invalid sigma value: {sigma}. Expected float, None, or 'r<float>'.")

    # Ensure sigma is positive and non-zero
    if sigma_val < 1e-9: sigma_val = 1e-9

    # Compute NDG and normalize to get membership function (mu)
    ndg_s = compute_ndg(x_values, sensor_data, sigma_val)
    sum_ndg = np.sum(ndg_s)
    mu_s = ndg_s / sum_ndg if sum_ndg > 1e-9 else np.zeros_like(ndg_s)

    return x_values, mu_s, sigma_val


def compute_membership_function_kde(sensor_data, x_values=None, num_points=500):
    """
    Computes a normalized membership function using Gaussian KDE.

    Args:
        sensor_data (np.ndarray): Input data points.
        x_values (np.ndarray, optional): Domain for the membership function.
                                         If None, calculated from data range. Defaults to None.
        num_points (int, optional): Number of points for x_values if automatically calculated. Defaults to 500.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing (x_values, mu_s).
                                        mu_s is the normalized membership function.
    """
    sensor_data = np.asarray(sensor_data)
    if sensor_data.size < 2: # KDE requires at least 2 points
        if x_values is None:
             # Define a default range if data is empty or constant
             x_min = np.min(sensor_data) if sensor_data.size > 0 else 0
             x_max = np.max(sensor_data) if sensor_data.size > 0 else 1
             data_range = x_max - x_min
             center = x_min if data_range < 1e-9 else (x_min + x_max) / 2
             spread = 1.0 if data_range < 1e-9 else data_range
             x_values = np.linspace(center - spread, center + spread, num_points)
        else:
             x_values = np.asarray(x_values)
        return x_values, np.zeros_like(x_values) # Return zeros if not enough data

    # Define x_values if not provided
    if x_values is None:
        x_min, x_max = np.min(sensor_data), np.max(sensor_data)
        data_range = x_max - x_min
        center = x_min if data_range < 1e-9 else (x_min + x_max) / 2
        spread = 1.0 if data_range < 1e-9 else data_range
        x_values = np.linspace(center - spread, center + spread, num_points)
    else:
        x_values = np.asarray(x_values)

    # Compute KDE and normalize
    try:
        kde = gaussian_kde(sensor_data)
        mu_s = kde.evaluate(x_values)
        mu_s = np.clip(mu_s, 0, None) # Ensure non-negative
        sum_mu = np.sum(mu_s)
        if sum_mu > 1e-9:
             mu_s /= sum_mu # Normalize
        else:
             mu_s = np.zeros_like(mu_s)
    except (np.linalg.LinAlgError, ValueError): # Handle LinAlgError or cases like all points identical
        # Fallback: return zeros if KDE fails
        mu_s = np.zeros_like(x_values)

    return x_values, mu_s


def compute_membership_functions(sensor_data, x_values, method='nd', sigma=None):
    """
    Wrapper function to compute membership function using the specified method.

    Args:
        sensor_data (np.ndarray): The sensor data.
        x_values (np.ndarray): The x-values over which to compute the membership function.
        method (str, optional): Method to use ('nd' or 'kde'). Defaults to 'nd'.
        sigma (float or str, optional): Sigma for 'nd' method. Defaults to None.

    Returns:
        tuple[np.ndarray, float or None]: Tuple containing (mu, sigma_val).
                                          mu is the computed membership function.
                                          sigma_val is the sigma used (if method='nd'), else None.
    """
    sigma_val = None
    if method == 'nd':
        x_values_calc, mu, sigma_val = compute_membership_function(sensor_data, x_values, sigma=sigma)
    elif method == 'kde':
        x_values_calc, mu = compute_membership_function_kde(sensor_data, x_values)
        # Ensure output mu has same shape as input x_values even if internal calculation used different points
        if x_values_calc.shape != x_values.shape or not np.allclose(x_values_calc, x_values):
            interp_mu = interp1d(x_values_calc, mu, kind='linear', bounds_error=False, fill_value=0)
            mu = interp_mu(x_values)
            mu = np.clip(mu, 0, None)
            sum_mu = np.sum(mu)
            if sum_mu > 1e-9: mu /= sum_mu
            else: mu = np.zeros_like(mu)

    else:
        raise ValueError("Unknown method for membership function. Use 'nd' or 'kde'.")
    return mu, sigma_val

# ==============================================================================
# Fuzzy Logic Helper Functions
# ==============================================================================

def safe_divide(numerator, denominator, default=0.0):
    """
    Performs division, returning a default value if the denominator is close to zero.

    Args:
        numerator (float or np.ndarray): The numerator(s).
        denominator (float or np.ndarray): The denominator(s).
        default (float, optional): Value to return on division by zero. Defaults to 0.0.

    Returns:
        float or np.ndarray: Result of the division or the default value.
    """
    if np.isscalar(denominator):
        # Handle scalar denominator
        return numerator / denominator if np.abs(denominator) > 1e-9 else default
    else:
        # Handle array denominator
        denominator = np.asarray(denominator)
        numerator = np.asarray(numerator)
        result = np.full_like(numerator, default, dtype=np.float64)
        valid_indices = np.abs(denominator) > 1e-9
        # Ensure numerator matches shape for broadcasting or element-wise division
        if numerator.shape == denominator.shape:
             result[valid_indices] = numerator[valid_indices] / denominator[valid_indices]
        elif numerator.shape == (1,) or numerator.size==1 : # Scalar numerator broadcast
             scalar_num = numerator.item() if isinstance(numerator, np.ndarray) else numerator
             result[valid_indices] = scalar_num / denominator[valid_indices]
        else: # Shapes incompatible other than scalar numerator
             raise ValueError("Numerator and denominator shapes incompatible for safe_divide.")

        return result

def fuzzy_intersection(mu1, mu2):
    """Computes the fuzzy intersection (pointwise minimum)."""
    return np.minimum(mu1, mu2)

def fuzzy_union(mu1, mu2):
    """Computes the fuzzy union (pointwise maximum)."""
    return np.maximum(mu1, mu2)

def fuzzy_negation(mu):
    """Computes the fuzzy negation (1 - mu)."""
    return 1.0 - np.asarray(mu)

def fuzzy_cardinality(mu):
    """Computes the fuzzy cardinality (sum of membership values)."""
    return np.sum(mu)

def fuzzy_symmetric_difference(mu1, mu2):
    """
    Computes the fuzzy symmetric difference using standard min/max/negation operators.
    Equivalent to Union(Intersection(A, neg(B)), Intersection(neg(A), B)).
    """
    neg_mu1 = fuzzy_negation(mu1)
    neg_mu2 = fuzzy_negation(mu2)
    term1 = fuzzy_intersection(mu1, neg_mu2)
    term2 = fuzzy_intersection(neg_mu1, mu2)
    return fuzzy_union(term1, term2)
    # Note: An alternative often used is simply np.abs(mu1 - mu2),
    # which corresponds to the difference in Lukasiewicz logic.
    # Using the standard definition here unless MATLAB source implies absolute difference.

# ==============================================================================
# Similarity Metrics Implementation
# ==============================================================================

# --- 1. Set-Theoretic / Overlap-Based Metrics ---

def similarity_jaccard(mu1, mu2):
    """
    Computes the Jaccard Index (Tanimoto Coefficient).
    Formula: |Intersection(A, B)| / |Union(A, B)|
    """
    intersection = fuzzy_intersection(mu1, mu2)
    union = fuzzy_union(mu1, mu2)
    card_intersection = fuzzy_cardinality(intersection)
    card_union = fuzzy_cardinality(union)
    return safe_divide(card_intersection, card_union)

def similarity_dice(mu1, mu2):
    """
    Computes the Dice Coefficient (Sørensen–Dice Index).
    Formula: 2 * |Intersection(A, B)| / (|A| + |B|)
    """
    intersection = fuzzy_intersection(mu1, mu2)
    card_intersection = fuzzy_cardinality(intersection)
    card_mu1 = fuzzy_cardinality(mu1)
    card_mu2 = fuzzy_cardinality(mu2)
    denominator = card_mu1 + card_mu2
    return safe_divide(2.0 * card_intersection, denominator)

def similarity_overlap_coefficient(mu1, mu2):
     """
     Computes the Overlap Coefficient (Szymkiewicz–Simpson).
     Formula: |Intersection(A, B)| / min(|A|, |B|)
     (Equivalent to MATLAB S7)
     """
     intersection = fuzzy_intersection(mu1, mu2)
     card_intersection = fuzzy_cardinality(intersection)
     card_mu1 = fuzzy_cardinality(mu1)
     card_mu2 = fuzzy_cardinality(mu2)
     min_card = min(card_mu1, card_mu2) if card_mu1 > 1e-9 and card_mu2 > 1e-9 else 0.0
     return safe_divide(card_intersection, min_card)

# --- MATLAB Specific Metrics (Set-Theoretic & Overlap) ---

def similarity_matlab_M(mu1, mu2):
    """MATLAB 'M': Equivalent to Jaccard Index."""
    return similarity_jaccard(mu1, mu2)

def similarity_matlab_S1(mu1, mu2):
    """
    MATLAB 'S1': Mean of pointwise (min / max).
    Handles division by zero where max=0 by returning 1.0 (as min is also 0).
    """
    mins = fuzzy_intersection(mu1, mu2)
    maxs = fuzzy_union(mu1, mu2)
    # If maxs is 0, mins must also be 0. Define ratio as 1 in this case.
    ratios = safe_divide(mins, maxs, default=1.0)
    return np.mean(ratios)

def similarity_matlab_S3(mu1, mu2):
    """
    MATLAB 'S3': Mean of pointwise (2 * min / (mu1 + mu2)). Pointwise Dice-like.
    Handles division by zero where sum=0 by returning 1.0 (as min is also 0).
    """
    mins = fuzzy_intersection(mu1, mu2)
    sums = np.asarray(mu1) + np.asarray(mu2)
    # If sums is 0, mins must also be 0. Define ratio as 1 in this case.
    ratios = safe_divide(2.0 * mins, sums, default=1.0)
    return np.mean(ratios)

def similarity_matlab_S5(mu1, mu2):
    """
    MATLAB 'S5': Cardinality of intersection / max Cardinality.
    Formula: |Intersection(A, B)| / max(|A|, |B|)
    """
    intersection = fuzzy_intersection(mu1, mu2)
    card_intersection = fuzzy_cardinality(intersection)
    card_mu1 = fuzzy_cardinality(mu1)
    card_mu2 = fuzzy_cardinality(mu2)
    max_card = max(card_mu1, card_mu2)
    return safe_divide(card_intersection, max_card)

# --- MATLAB Specific Metrics (Negation / Symmetric Difference Based) ---

def similarity_matlab_S4(mu1, mu2):
    """MATLAB 'S4': Jaccard index applied to the negated sets."""
    neg_mu1 = fuzzy_negation(mu1)
    neg_mu2 = fuzzy_negation(mu2)
    return similarity_jaccard(neg_mu1, neg_mu2)

def similarity_matlab_S6(mu1, mu2):
    """MATLAB 'S6': S5 metric applied to the negated sets."""
    neg_mu1 = fuzzy_negation(mu1)
    neg_mu2 = fuzzy_negation(mu2)
    return similarity_matlab_S5(neg_mu1, neg_mu2)

def similarity_matlab_S8(mu1, mu2):
    """MATLAB 'S8': Overlap coefficient (S7) applied to the negated sets."""
    neg_mu1 = fuzzy_negation(mu1)
    neg_mu2 = fuzzy_negation(mu2)
    return similarity_overlap_coefficient(neg_mu1, neg_mu2)

def similarity_matlab_S9(mu1, mu2):
     """
     MATLAB 'S9': Based on negated symmetric difference.
     Formula: |neg(SymmDiff(A,B))| / max(|neg(A \cap neg(B))|, |neg(neg(A) \cap B)|)
     """
     neg_mu1 = fuzzy_negation(mu1)
     neg_mu2 = fuzzy_negation(mu2)
     symm_diff = fuzzy_symmetric_difference(mu1, mu2) # Union(A cap negB, negA cap B)
     neg_symm_diff = fuzzy_negation(symm_diff)
     card_neg_symm_diff = fuzzy_cardinality(neg_symm_diff)

     # Components of symm_diff
     comp1 = fuzzy_intersection(mu1, neg_mu2) # A cap neg(B)
     comp2 = fuzzy_intersection(neg_mu1, mu2) # neg(A) cap B

     # Cardinality of the negation of the components
     card_neg_comp1 = fuzzy_cardinality(fuzzy_negation(comp1))
     card_neg_comp2 = fuzzy_cardinality(fuzzy_negation(comp2))

     max_card_neg_comp = max(card_neg_comp1, card_neg_comp2)
     return safe_divide(card_neg_symm_diff, max_card_neg_comp)

def similarity_matlab_S10(mu1, mu2):
     """
     MATLAB 'S10': Based on negated symmetric difference.
     Formula: |neg(SymmDiff(A,B))| / min(|neg(A \cap neg(B))|, |neg(neg(A) \cap B)|)
     """
     neg_mu1 = fuzzy_negation(mu1)
     neg_mu2 = fuzzy_negation(mu2)
     symm_diff = fuzzy_symmetric_difference(mu1, mu2)
     neg_symm_diff = fuzzy_negation(symm_diff)
     card_neg_symm_diff = fuzzy_cardinality(neg_symm_diff)

     comp1 = fuzzy_intersection(mu1, neg_mu2) # A cap neg(B)
     comp2 = fuzzy_intersection(neg_mu1, mu2) # neg(A) cap B
     card_neg_comp1 = fuzzy_cardinality(fuzzy_negation(comp1))
     card_neg_comp2 = fuzzy_cardinality(fuzzy_negation(comp2))

     min_card_neg_comp = min(card_neg_comp1, card_neg_comp2)
     return safe_divide(card_neg_symm_diff, min_card_neg_comp)

def similarity_matlab_S11(mu1, mu2):
    """
    MATLAB 'S11': 1 - mean(symmetric_difference).
    Note: Assumes standard definition of symmetric difference.
    """
    symm_diff = fuzzy_symmetric_difference(mu1, mu2)
    return 1.0 - np.mean(symm_diff)

# --- 2. Distance-Based Metrics ---

def distance_hamming(mu1, mu2):
    """Computes the Hamming distance (sum of absolute differences)."""
    return np.sum(np.abs(np.asarray(mu1) - np.asarray(mu2)))

def similarity_hamming(mu1, mu2):
    """
    Computes normalized Hamming Similarity.
    Formula: 1 - (HammingDistance / n)
    """
    n = len(mu1)
    if n == 0: return 1.0 # Identical if no points to compare
    dist_h = distance_hamming(mu1, mu2)
    # Distance is bounded by n (since max diff is 1), so similarity is [0, 1]
    return 1.0 - safe_divide(dist_h, float(n))

def distance_euclidean(mu1, mu2):
    """Computes the standard Euclidean distance."""
    diff = np.asarray(mu1) - np.asarray(mu2)
    return np.sqrt(np.sum(diff ** 2))

def similarity_euclidean(mu1, mu2):
    """
    Computes similarity based on Euclidean distance.
    Formula: 1 / (1 + EuclideanDistance)
    """
    dist_e = distance_euclidean(mu1, mu2)
    return 1.0 / (1.0 + dist_e)

def distance_chebyshev(mu1, mu2):
    """Computes the Chebyshev distance (maximum absolute difference)."""
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    if mu1.size == 0: return 0.0 # No difference if empty
    return np.max(np.abs(mu1 - mu2))

def similarity_chebyshev(mu1, mu2):
    """
    Computes similarity based on Chebyshev distance.
    Formula: 1 - ChebyshevDistance
    (Equivalent to MATLAB L)
    """
    dist_c = distance_chebyshev(mu1, mu2)
     # Assumes membership values are in [0, 1], so dist_c is also [0, 1]
    return 1.0 - dist_c

# --- MATLAB Specific Metrics (Distance-Based) ---

def similarity_matlab_S2_W(mu1, mu2):
    """
    MATLAB 'S2' and 'W': Mean of (1 - absolute difference).
    Formula: 1 - mean(|mu1 - mu2|)
    """
    abs_diff = np.abs(np.asarray(mu1) - np.asarray(mu2))
    return 1.0 - np.mean(abs_diff)

def similarity_matlab_S(mu1, mu2):
    """
    MATLAB 'S': Normalized Hamming-like similarity.
    Formula: 1 - sum(|mu1 - mu2|) / (|A| + |B|)
    """
    sum_abs_diff = distance_hamming(mu1, mu2)
    card_mu1 = fuzzy_cardinality(mu1)
    card_mu2 = fuzzy_cardinality(mu2)
    denominator = card_mu1 + card_mu2
    return 1.0 - safe_divide(sum_abs_diff, denominator)

# --- 3. Correlation-Based Metrics ---

def similarity_cosine(mu1, mu2):
    """
    Computes the Cosine Similarity between two membership vectors.
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    dot_product = np.dot(mu1, mu2)
    norm_mu1 = np.linalg.norm(mu1)
    norm_mu2 = np.linalg.norm(mu2)
    denominator = norm_mu1 * norm_mu2
    # If either norm is zero, similarity is zero
    return safe_divide(dot_product, denominator, default=0.0)

def similarity_pearson(mu1, mu2):
    """
    Computes the Pearson Correlation Coefficient between two membership vectors.
    Handles cases with zero variance (constant membership function) by returning 0 correlation.
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    if mu1.size < 2: return 0.0 # Correlation undefined for less than 2 points

    mean_mu1 = np.mean(mu1)
    mean_mu2 = np.mean(mu2)
    centered_mu1 = mu1 - mean_mu1
    centered_mu2 = mu2 - mean_mu2

    # Check for zero variance
    std_dev1 = np.std(centered_mu1)
    std_dev2 = np.std(centered_mu2)
    if std_dev1 < 1e-9 or std_dev2 < 1e-9:
        return 0.0 # Or np.nan? Returning 0 for no linear correlation.

    numerator = np.dot(centered_mu1, centered_mu2)
    norm_centered_mu1 = np.linalg.norm(centered_mu1)
    norm_centered_mu2 = np.linalg.norm(centered_mu2)
    denominator = norm_centered_mu1 * norm_centered_mu2

    return safe_divide(numerator, denominator, default=0.0)

# --- MATLAB Specific Metrics (Correlation-Based) ---

def similarity_matlab_P(mu1, mu2):
    """
    MATLAB 'P': Sum of pointwise product normalized by minimum squared norm.
    Formula: sum(mu1 * mu2) / min(||mu1||^2, ||mu2||^2)
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    pointwise_product = mu1 * mu2
    sum_product = np.sum(pointwise_product)
    norm_sq_mu1 = np.dot(mu1, mu1) # norm^2 = sum(mu1_i^2)
    norm_sq_mu2 = np.dot(mu2, mu2) # norm^2
    denominator = min(norm_sq_mu1, norm_sq_mu2)
    # If min norm is 0, both are zero vectors? Return 1? Or 0?
    # If denominator is 0, implies one vector is all zeros. If sum_product is also 0, result is undefined.
    # Let's return 0 if denominator is 0. If sum_product > 0 but denom=0, should not happen.
    return safe_divide(sum_product, denominator, default=0.0)


# --- 4. Other Metrics ---

def similarity_matlab_T(mu1, mu2):
    """
    MATLAB 'T': Maximum value of the fuzzy intersection.
    Formula: max(min(mu1, mu2))
    """
    intersection = fuzzy_intersection(mu1, mu2)
    return np.max(intersection) if len(intersection) > 0 else 0.0

def similarity_matlab_metric1(data_s1, data_s2, x_values_common, fs_method='nd', sigma_s1=None, sigma_s2=None):
     """
     Approximation of MATLAB 'similarity_metric1' ("Theirs").
     Requires original raw sensor data.

     Args:
        data_s1 (np.ndarray): Raw data for sensor 1.
        data_s2 (np.ndarray): Raw data for sensor 2.
        x_values_common (np.ndarray): Common domain for membership functions.
        fs_method (str): Method used to compute membership ('nd' or 'kde').
        sigma_s1 (float or str, optional): Sigma used/to use for sensor 1 if method='nd'.
        sigma_s2 (float or str, optional): Sigma used/to use for sensor 2 if method='nd'.

     Returns:
        float: Calculated similarity value, or NaN if inputs are invalid.
     """
     data_s1 = np.asarray(data_s1)
     data_s2 = np.asarray(data_s2)
     if data_s1.size < 2 or data_s2.size < 2: # Need at least 2 points for IQR
         return np.nan

     # 1. Calculate delta (sum of absolute differences between IQRs is not correct, based on MATLAB code)
     # MATLAB Code: delta = sum(abs(quantilesS - quantilesV)) where quantiles = quantile(data, [0.25 0.75])
     # This means delta = |Q1_S - Q1_V| + |Q3_S - Q3_V|
     try:
         q1_s1, q3_s1 = np.percentile(data_s1, [25, 75])
         q1_s2, q3_s2 = np.percentile(data_s2, [25, 75])
         delta = abs(q1_s1 - q1_s2) + abs(q3_s1 - q3_s2)
     except IndexError: # Handle cases where percentile fails (e.g., too few unique points)
         return np.nan

     # 2. Construct fuzzy set for data_s2 (mu_s2)
     # Use the provided fs_method and sigma_s2
     mu_s2_calc, sigma_used_s2 = compute_membership_functions(
         data_s2, x_values_common, method=fs_method, sigma=sigma_s2
     )
     if np.sum(mu_s2_calc) < 1e-9: return 0.0 # mu_s2 is essentially zero

     # 3. Sum membership values of mu_s2 at points corresponding to data_s1 values
     # This requires interpolating mu_s2 onto the values present in data_s1
     interp_mu_s2 = interp1d(x_values_common, mu_s2_calc, kind='linear', bounds_error=False, fill_value=0)
     mu_vals_at_s1 = interp_mu_s2(data_s1)
     sum_mu_vals = np.sum(mu_vals_at_s1)

     # 4. Normalize
     # MATLAB uses 'r' (related to range/resolution of the fuzzy set of V?) if delta is 0.
     # Let's approximate 'r' based on the sigma used for mu_s2 or average step of x_values.
     # A simple fallback is a small epsilon.
     if delta < 1e-9:
         if sigma_used_s2 is not None and sigma_used_s2 > 1e-9:
             # Heuristic: relate 'r' to the bandwidth sigma
             delta = sigma_used_s2 # Or maybe some factor * sigma?
         elif len(x_values_common) > 1:
             # Use average step size of the domain as a fallback 'resolution'
             delta = np.mean(np.diff(x_values_common))
         else:
             delta = 1e-9 # Ultimate fallback epsilon
         # Ensure delta is not zero after approximation
         if delta < 1e-9: delta = 1e-9

     # Final calculation
     denominator = len(data_s1) * delta
     similarity = safe_divide(sum_mu_vals, denominator)
     return similarity

# `similarity_metric2` ("Theirs + derivative") is omitted for now due to the
# complexity of consistently defining and constructing fuzzy sets for derivatives
# in a way that accurately matches the MATLAB implementation's implicit assumptions.

# ==============================================================================
# Main Orchestrator Function for Similarity
# ==============================================================================

def calculate_all_similarity_metrics(mu_s1, mu_s2, x_values,
                                   data_s1=None, data_s2=None,
                                   fs_method='nd', sigma_s1=None, sigma_s2=None):
    """
    Calculates a comprehensive set of similarity metrics between two membership functions.

    Args:
        mu_s1 (np.ndarray): Membership function of sensor 1 over x_values.
        mu_s2 (np.ndarray): Membership function of sensor 2 over x_values.
        x_values (np.ndarray): Common domain (x-values) for the membership functions.
        data_s1 (np.ndarray, optional): Raw sensor data for sensor 1 (required for metric1).
        data_s2 (np.ndarray, optional): Raw sensor data for sensor 2 (required for metric1).
        fs_method (str, optional): Method used if recalculating FS for metric1 ('nd' or 'kde').
        sigma_s1 (float or str, optional): Sigma for mu_s1 if fs_method='nd' (for metric1).
        sigma_s2 (float or str, optional): Sigma for mu_s2 if fs_method='nd' (for metric1).

    Returns:
        dict: Dictionary containing all calculated similarity metrics. Keys are metric names.
    """
    mu_s1 = np.asarray(mu_s1)
    mu_s2 = np.asarray(mu_s2)
    x_values = np.asarray(x_values)

    if mu_s1.shape != mu_s2.shape or mu_s1.shape != x_values.shape:
        raise ValueError("Input shapes mismatch: mu_s1, mu_s2, and x_values must have the same shape.")

    if mu_s1.size == 0: # Handle empty inputs
        print("Warning: Empty membership function(s) provided. Returning empty similarity dict.")
        return {}

    similarities = {}

    # --- 1. Set-Theoretic / Overlap-Based ---
    similarities['Jaccard'] = similarity_jaccard(mu_s1, mu_s2)
    similarities['Dice'] = similarity_dice(mu_s1, mu_s2)
    similarities['OverlapCoefficient'] = similarity_overlap_coefficient(mu_s1, mu_s2) # Also MATLAB S7
    # MATLAB Equivalents
    similarities['MATLAB_M'] = similarities['Jaccard']
    similarities['MATLAB_S1'] = similarity_matlab_S1(mu_s1, mu_s2)
    similarities['MATLAB_S3'] = similarity_matlab_S3(mu_s1, mu_s2)
    similarities['MATLAB_S5'] = similarity_matlab_S5(mu_s1, mu_s2)
    similarities['MATLAB_S7'] = similarities['OverlapCoefficient']

    # --- Negation / Symmetric Difference Based ---
    similarities['MATLAB_S4'] = similarity_matlab_S4(mu_s1, mu_s2)
    similarities['MATLAB_S6'] = similarity_matlab_S6(mu_s1, mu_s2)
    similarities['MATLAB_S8'] = similarity_matlab_S8(mu_s1, mu_s2)
    similarities['MATLAB_S9'] = similarity_matlab_S9(mu_s1, mu_s2)
    similarities['MATLAB_S10'] = similarity_matlab_S10(mu_s1, mu_s2)
    similarities['MATLAB_S11'] = similarity_matlab_S11(mu_s1, mu_s2)

    # --- 2. Distance-Based ---
    similarities['Distance_Hamming'] = distance_hamming(mu_s1, mu_s2)
    similarities['Similarity_Hamming'] = similarity_hamming(mu_s1, mu_s2)
    similarities['Distance_Euclidean'] = distance_euclidean(mu_s1, mu_s2)
    similarities['Similarity_Euclidean'] = similarity_euclidean(mu_s1, mu_s2)
    similarities['Distance_Chebyshev'] = distance_chebyshev(mu_s1, mu_s2)
    similarities['Similarity_Chebyshev'] = similarity_chebyshev(mu_s1, mu_s2) # Also MATLAB L
    # MATLAB Equivalents / Specific
    similarities['MATLAB_S2_W'] = similarity_matlab_S2_W(mu_s1, mu_s2) # S2 and W are the same
    similarities['MATLAB_S'] = similarity_matlab_S(mu_s1, mu_s2)
    similarities['MATLAB_L'] = similarities['Similarity_Chebyshev']

    # --- 3. Correlation-Based ---
    similarities['Cosine'] = similarity_cosine(mu_s1, mu_s2)
    similarities['Pearson'] = similarity_pearson(mu_s1, mu_s2)
    # MATLAB Specific
    similarities['MATLAB_P'] = similarity_matlab_P(mu_s1, mu_s2)

    # --- 4. Other ---
    similarities['MATLAB_T'] = similarity_matlab_T(mu_s1, mu_s2)

    # --- Metrics requiring raw data ---
    if data_s1 is not None and data_s2 is not None:
         similarities['MATLAB_Metric1'] = similarity_matlab_metric1(
             data_s1, data_s2, x_values, fs_method=fs_method, sigma_s1=sigma_s1, sigma_s2=sigma_s2
         )
    else:
         # Indicate that metric1 could not be calculated due to missing raw data
         similarities['MATLAB_Metric1'] = np.nan

    # Optional: Clean up NaN/Inf results if desired
    # cleaned_similarities = {k: v for k, v in similarities.items() if not (np.isnan(v) or np.isinf(v))}
    # return cleaned_similarities

    return similarities


# ==============================================================================
# Data Preprocessing Utilities
# ==============================================================================
    
def normalize_data(data):
    """
    Performs Min-Max Normalization on the data to scale it to the range [0, 1].

    Args:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: Normalized data array. Returns original array if range is zero.
    """
    data = np.asarray(data)
    if data.size == 0: return data # Handle empty
    x_min = np.min(data)
    x_max = np.max(data)
    data_range = x_max - x_min
    # Handle case where all data points are the same (range is zero)
    if data_range < 1e-9:
        # Return array of 0s, 0.5s, or 1s depending on desired convention?
        # Returning 0.5s here.
        return np.full_like(data, 0.5, dtype=np.float64)
    normalized_data = (data - x_min) / data_range
    return normalized_data

def standardize_data(data):
    """
    Performs Z-Score Standardization (mean=0, std=1) on the data.

    Args:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: Standardized data array. Returns array of zeros if std dev is zero.
    """
    data = np.asarray(data)
    if data.size == 0: return data # Handle empty
    mean = np.mean(data)
    std = np.std(data)
    # Handle case where standard deviation is zero
    standardized_data = safe_divide(data - mean, std, default=0.0)
    return standardized_data

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
        densities, centers = compute_empirical_distribution_counts(x_values, sensor_data)
        # Need counts associated with these bins/centers for Chi2
        # Need probabilities on x_values for error/KL metrics - use KDE as fallback for now
        print("Warning: 'counts' method for empirical distribution currently uses KDE for error/KL metrics. Chi-squared requires separate handling.")
        empirical_probs = compute_empirical_distribution_kde(x_values, sensor_data)
        # TODO: Implement robust way to get empirical counts aligned with theoretical probs for Chi2
    else:
        raise ValueError("Unknown empirical method. Use 'kde' or 'counts'.")

    if empirical_probs is None or np.sum(empirical_probs) < 1e-9:
         print("Warning: Could not compute valid empirical probability distribution.")
         # Return NaNs if empirical calculation failed
         default_metrics = {key: np.nan for key in ['MSE', 'RMSE', 'MAE', 'KL_Divergence', 'AIC', 'BIC', 'MSE_CV', 'KL_Divergence_CV']}
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

    # Chi-squared (Requires empirical counts aligned with theoretical_probs - placeholder)
    # chi2_stat, p_value = compute_chi_squared_test(empirical_counts, theoretical_probs, len(sensor_data))

    # --- 4. Combine Results ---
    fitness_metrics = {
        'MSE': error_metrics['MSE'],
        'RMSE': error_metrics['RMSE'],
        'MAE': error_metrics['MAE'],
        'KL_Divergence': kl_divergence,
        'AIC': info_criteria['AIC'],
        'BIC': info_criteria['BIC'],
        'MSE_CV': cv_results['MSE_CV'],
        'KL_Divergence_CV': cv_results['KL_Divergence_CV']
        # 'Chi2_Stat': chi2_stat, # Add if Chi2 implemented
        # 'Chi2_PValue': p_value  # Add if Chi2 implemented
    }

    return fitness_metrics, empirical_probs
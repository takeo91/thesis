"""
similarity_metrics.py

Functions for calculating various fuzzy set similarity metrics based on
membership functions (mu). Includes standard metrics and specific metrics
inspired by MATLAB implementations.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import iqr, gaussian_kde # Used only in similarity_matlab_metric1

# Import helpers from the same directory
from fuzzy_helpers import (
    safe_divide, fuzzy_intersection, fuzzy_union, fuzzy_negation,
    fuzzy_cardinality, fuzzy_symmetric_difference
)
# Import membership function calculation for metric1
from membership_functions import compute_membership_functions, compute_ndg

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
     # Denominator should only be zero if both cardinalities are zero
     min_card = min(card_mu1, card_mu2)
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
     mu1 = np.asarray(mu1)
     mu2 = np.asarray(mu2)
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
     mu1 = np.asarray(mu1)
     mu2 = np.asarray(mu2)
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
    # Mean requires array not be empty
    return 1.0 - np.mean(symm_diff) if symm_diff.size > 0 else 1.0

# --- 2. Distance-Based Metrics ---

def distance_hamming(mu1, mu2):
    """Computes the Hamming distance (sum of absolute differences)."""
    return np.sum(np.abs(np.asarray(mu1) - np.asarray(mu2)))

def similarity_hamming(mu1, mu2):
    """
    Computes normalized Hamming Similarity.
    Formula: 1 - (HammingDistance / n)
    """
    mu1 = np.asarray(mu1)
    n = mu1.size
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
    # Add epsilon to prevent division by zero just in case dist_e is exactly -1 (shouldn't happen)
    return 1.0 / (1.0 + dist_e + 1e-9)

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
    return 1.0 - np.mean(abs_diff) if abs_diff.size > 0 else 1.0

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
    if mu1.size == 0: return 1.0 # Or 0.0? Define similarity for empty vectors

    dot_product = np.dot(mu1, mu2)
    norm_mu1 = np.linalg.norm(mu1)
    norm_mu2 = np.linalg.norm(mu2)
    denominator = norm_mu1 * norm_mu2
    # If either norm is zero, similarity is zero (or 1 if both are zero?)
    # Let's return 1 if both norms are near zero, 0 if only one is.
    if denominator < 1e-9:
        return 1.0 if norm_mu1 < 1e-9 and norm_mu2 < 1e-9 else 0.0
    return dot_product / denominator

def similarity_pearson(mu1, mu2):
    """
    Computes the Pearson Correlation Coefficient between two membership vectors.
    Handles cases with zero variance (constant membership function) by returning 0 correlation.
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    if mu1.size < 2: return 0.0 # Correlation undefined for less than 2 points

    # Check for zero variance first
    std_dev1 = np.std(mu1)
    std_dev2 = np.std(mu2)
    if std_dev1 < 1e-9 or std_dev2 < 1e-9:
        # If both have zero variance, are they perfectly correlated? Let's return 1.0
        # If only one has zero variance, correlation is 0.
        return 1.0 if std_dev1 < 1e-9 and std_dev2 < 1e-9 else 0.0

    # Proceed with calculation if variances are non-zero
    mean_mu1 = np.mean(mu1)
    mean_mu2 = np.mean(mu2)
    centered_mu1 = mu1 - mean_mu1
    centered_mu2 = mu2 - mean_mu2

    numerator = np.dot(centered_mu1, centered_mu2)
    norm_centered_mu1 = np.linalg.norm(centered_mu1)
    norm_centered_mu2 = np.linalg.norm(centered_mu2)
    denominator = norm_centered_mu1 * norm_centered_mu2

    # Denominator should only be zero now if vectors are empty, handled earlier.
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
    # If denominator is 0, implies one vector is all zeros.
    # If sum_product is also 0 (e.g., both vectors are zero), result is undefined. Return 1?
    # If sum_product > 0 but denom=0, this means one vector is zero, the other non-zero. Return 0?
    # Let's return 1 if both norms are zero, 0 otherwise when denom is zero.
    if denominator < 1e-9:
        return 1.0 if norm_sq_mu1 < 1e-9 and norm_sq_mu2 < 1e-9 else 0.0
    return sum_product / denominator


# --- 4. Other Metrics ---

def similarity_matlab_T(mu1, mu2):
    """
    MATLAB 'T': Maximum value of the fuzzy intersection.
    Formula: max(min(mu1, mu2))
    """
    intersection = fuzzy_intersection(mu1, mu2)
    return np.max(intersection) if intersection.size > 0 else 0.0

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
     x_values_common = np.asarray(x_values_common)
     if data_s1.size < 2 or data_s2.size < 2 or x_values_common.size == 0:
         # Need at least 2 points for IQR, and a domain
         return np.nan

     # 1. Calculate delta = |Q1_S - Q1_V| + |Q3_S - Q3_V|
     try:
         q1_s1, q3_s1 = np.percentile(data_s1, [25, 75])
         q1_s2, q3_s2 = np.percentile(data_s2, [25, 75])
         delta = abs(q1_s1 - q1_s2) + abs(q3_s1 - q3_s2)
     except IndexError: # Handle cases where percentile fails
         return np.nan

     # 2. Construct fuzzy set for data_s2 (mu_s2) using the specified method
     # compute_membership_functions returns mu normalized to sum to 1 (like probability)
     # The MATLAB code seems to use unnormalized NDG values (fzst) in the formula.
     # Let's recalculate using the core NDG/KDE logic without final normalization for this metric.
     sigma_used_s2 = None
     if fs_method == 'nd':
         # Determine sigma_s2 if needed
         if sigma_s2 is None:
             x_min2, x_max2 = np.min(data_s2), np.max(data_s2)
             range2 = x_max2 - x_min2
             sigma_used_s2 = 0.1 * range2 if range2 > 1e-9 else 0.1
         elif isinstance(sigma_s2, str) and sigma_s2.startswith('r'):
             ratio = float(sigma_s2[1:])
             x_min2, x_max2 = np.min(data_s2), np.max(data_s2)
             range2 = x_max2 - x_min2
             sigma_used_s2 = ratio * range2 if range2 > 1e-9 else ratio
         else:
             sigma_used_s2 = float(sigma_s2)
         if sigma_used_s2 < 1e-9: sigma_used_s2 = 1e-9
         # Compute unnormalized NDG
         mu_s2_unnormalized = compute_ndg(x_values_common, data_s2, sigma_used_s2)

     elif fs_method == 'kde':
         if data_s2.size < 2: return np.nan
         try:
             kde = gaussian_kde(data_s2)
             # Evaluate density (unnormalized probability)
             mu_s2_unnormalized = kde.evaluate(x_values_common)
             mu_s2_unnormalized = np.clip(mu_s2_unnormalized, 0, None)
             sigma_used_s2 = None # Sigma not directly applicable/returned by KDE
         except (np.linalg.LinAlgError, ValueError):
             return np.nan # Cannot compute KDE
     else:
         raise ValueError("Unknown fs_method for similarity_matlab_metric1")

     if np.sum(mu_s2_unnormalized) < 1e-9: return 0.0 # Unnormalized mu_s2 is essentially zero

     # 3. Sum membership values of mu_s2 (unnormalized) at points corresponding to data_s1 values
     interp_mu_s2 = interp1d(x_values_common, mu_s2_unnormalized, kind='linear', bounds_error=False, fill_value=0)
     mu_vals_at_s1 = interp_mu_s2(data_s1)
     sum_mu_vals = np.sum(mu_vals_at_s1)

     # 4. Normalize using delta or approximation 'r'
     # Approximate 'r' as used in MATLAB (related to resolution/sigma?)
     if delta < 1e-9:
         if sigma_used_s2 is not None and sigma_used_s2 > 1e-9:
             delta_approx = sigma_used_s2 # Use sigma as approximation for 'r'
         elif len(x_values_common) > 1:
             delta_approx = np.mean(np.diff(x_values_common)) # Use mean step size
         else:
             delta_approx = 1e-9 # Fallback epsilon
         delta = max(delta_approx, 1e-9) # Ensure delta is positive

     # Final calculation
     denominator = len(data_s1) * delta
     similarity = safe_divide(sum_mu_vals, denominator)
     return similarity

# `similarity_metric2` ("Theirs + derivative") is omitted.

# ==============================================================================
# Main Orchestrator Function for Similarity
# ==============================================================================

def calculate_all_similarity_metrics(mu_s1, mu_s2, x_values,
                                   data_s1=None, data_s2=None,
                                   fs_method='nd', sigma_s1=None, sigma_s2=None):
    """
    Calculates a comprehensive set of similarity metrics between two membership functions.

    Args:
        mu_s1 (np.ndarray): Membership function of sensor 1 over x_values (normalized).
        mu_s2 (np.ndarray): Membership function of sensor 2 over x_values (normalized).
        x_values (np.ndarray): Common domain (x-values) for the membership functions.
        data_s1 (np.ndarray, optional): Raw sensor data for sensor 1 (required for metric1).
        data_s2 (np.ndarray, optional): Raw sensor data for sensor 2 (required for metric1).
        fs_method (str, optional): Method used if recalculating FS for metric1 ('nd' or 'kde').
        sigma_s1 (float or str, optional): Sigma for mu_s1 if fs_method='nd' (needed for metric1).
        sigma_s2 (float or str, optional): Sigma for mu_s2 if fs_method='nd' (needed for metric1).

    Returns:
        dict: Dictionary containing all calculated similarity metrics. Keys are metric names.
              Values are float or np.nan if calculation failed or data was missing.
    """
    mu_s1 = np.asarray(mu_s1)
    mu_s2 = np.asarray(mu_s2)
    x_values = np.asarray(x_values)

    # --- Input Validation ---
    if mu_s1.shape != mu_s2.shape or mu_s1.shape != x_values.shape:
        raise ValueError("Input shapes mismatch: mu_s1, mu_s2, and x_values must have the same shape.")
    if mu_s1.size == 0: # Handle empty inputs
        print("Warning: Empty membership function(s) provided. Returning empty similarity dict.")
        return {}
    # Ensure MFs are normalized probabilities (sum to 1)
    sum1 = np.sum(mu_s1)
    sum2 = np.sum(mu_s2)
    if abs(sum1 - 1.0) > 1e-6 or abs(sum2 - 1.0) > 1e-6:
        print("Warning: Input membership functions mu_s1 and/or mu_s2 do not sum to 1. Normalizing for calculation.")
        if sum1 > 1e-9: mu_s1 = mu_s1 / sum1
        else: mu_s1 = np.zeros_like(mu_s1)
        if sum2 > 1e-9: mu_s2 = mu_s2 / sum2
        else: mu_s2 = np.zeros_like(mu_s2)

    similarities = {}
    results = {} # Temporary dict to store results and handle potential errors

    # --- Calculation Functions ---
    metric_functions = {
        # Set-Theoretic / Overlap
        'Jaccard': similarity_jaccard,
        'Dice': similarity_dice,
        'OverlapCoefficient': similarity_overlap_coefficient,
        'MATLAB_S1': similarity_matlab_S1,
        'MATLAB_S3': similarity_matlab_S3,
        'MATLAB_S5': similarity_matlab_S5,
        # Negation / Symmetric Difference
        'MATLAB_S4': similarity_matlab_S4,
        'MATLAB_S6': similarity_matlab_S6,
        'MATLAB_S8': similarity_matlab_S8,
        'MATLAB_S9': similarity_matlab_S9,
        'MATLAB_S10': similarity_matlab_S10,
        'MATLAB_S11': similarity_matlab_S11,
        # Distance-Based (Similarity)
        'Similarity_Hamming': similarity_hamming,
        'Similarity_Euclidean': similarity_euclidean,
        'Similarity_Chebyshev': similarity_chebyshev,
        'MATLAB_S2_W': similarity_matlab_S2_W,
        'MATLAB_S': similarity_matlab_S,
        # Correlation-Based
        'Cosine': similarity_cosine,
        'Pearson': similarity_pearson,
        'MATLAB_P': similarity_matlab_P,
        # Other
        'MATLAB_T': similarity_matlab_T,
        # Distance-Based (Distance) - Also calculate distances
        'Distance_Hamming': distance_hamming,
        'Distance_Euclidean': distance_euclidean,
        'Distance_Chebyshev': distance_chebyshev,
    }

    # --- Execute Calculations ---
    for name, func in metric_functions.items():
        try:
            results[name] = func(mu_s1, mu_s2)
        except Exception as e:
            print(f"Warning: Could not calculate metric '{name}'. Error: {e}")
            results[name] = np.nan

    # --- Handle Equivalent Metrics ---
    results['MATLAB_M'] = results.get('Jaccard', np.nan)
    results['MATLAB_S7'] = results.get('OverlapCoefficient', np.nan)
    results['MATLAB_L'] = results.get('Similarity_Chebyshev', np.nan)

    # --- Metric Requiring Raw Data ---
    if data_s1 is not None and data_s2 is not None:
        try:
            results['MATLAB_Metric1'] = similarity_matlab_metric1(
                data_s1, data_s2, x_values, fs_method=fs_method, sigma_s1=sigma_s1, sigma_s2=sigma_s2
            )
        except Exception as e:
            print(f"Warning: Could not calculate metric 'MATLAB_Metric1'. Error: {e}")
            results['MATLAB_Metric1'] = np.nan
    else:
        results['MATLAB_Metric1'] = np.nan # Indicate missing data

    # --- Final Ordering and Return ---
    # Define preferred order if needed, otherwise return alphabetical or calculation order
    metric_order = [
        # Standard
        'Jaccard', 'Dice', 'OverlapCoefficient', 'Cosine', 'Pearson',
        'Similarity_Hamming', 'Similarity_Euclidean', 'Similarity_Chebyshev',
        'Distance_Hamming', 'Distance_Euclidean', 'Distance_Chebyshev',
        # MATLAB Specific (in approx order from source)
        'MATLAB_Metric1', # "Theirs"
        # 'MATLAB_Metric2', # "Theirs + derivative" - Omitted
        'MATLAB_S1', 'MATLAB_M', 'MATLAB_T', 'MATLAB_P',
        'MATLAB_S2_W', # S2 and W
        'MATLAB_S3',
        # 'MATLAB_W', # Covered by S2_W
        'MATLAB_L', 'MATLAB_S',
        # 'MATLAB_Dinf', # Omitted - requires different inputs
        'MATLAB_S4', 'MATLAB_S5', 'MATLAB_S6', 'MATLAB_S7', 'MATLAB_S8',
        'MATLAB_S9', 'MATLAB_S10', 'MATLAB_S11'
    ]

    # Populate the final dictionary in the desired order
    for key in metric_order:
        if key in results:
            similarities[key] = results[key]
        # Handle metrics covered by others explicitly if needed, though already assigned
        # elif key == 'MATLAB_M': similarities[key] = results.get('Jaccard', np.nan) ...

    return similarities

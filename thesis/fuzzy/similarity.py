from __future__ import annotations

from typing import Dict, Sequence, Union, Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde  # Used only in similarity_matlab_metric1

# Local helpers from the fuzzy package
from thesis.fuzzy.operations import (
    safe_divide,
    fuzzy_intersection,
    fuzzy_union,
    fuzzy_negation,
    fuzzy_cardinality,
    fuzzy_symmetric_difference,
)
from thesis.fuzzy.membership import compute_ndg_streaming

ArrayLike = Union[Sequence[float], np.ndarray]

# -----------------------------------------------------------------------------
# 1. Set‑theoretic / Overlap‑based metrics
# -----------------------------------------------------------------------------


def similarity_jaccard(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Jaccard index / Tanimoto coefficient."""
    intersection = fuzzy_intersection(mu1, mu2)
    union = fuzzy_union(mu1, mu2)
    return safe_divide(
        fuzzy_cardinality(intersection),
        fuzzy_cardinality(union),
        default=1.0,
    )


def similarity_dice(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Dice (Sørensen–Dice) coefficient."""
    intersection = fuzzy_intersection(mu1, mu2)
    denominator = fuzzy_cardinality(mu1) + fuzzy_cardinality(mu2)
    return safe_divide(2.0 * fuzzy_cardinality(intersection), denominator, default=1.0)


def similarity_overlap_coefficient(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Overlap coefficient (Szymkiewicz–Simpson)."""
    intersection = fuzzy_intersection(mu1, mu2)
    min_card = min(fuzzy_cardinality(mu1), fuzzy_cardinality(mu2))
    # If both sets are empty, they are identical → similarity = 1
    return safe_divide(fuzzy_cardinality(intersection), min_card, default=1.0)


# -----------------------------------------------------------------------------
# MATLAB‑specific variants built from the above primitives (renamed descriptively)
# -----------------------------------------------------------------------------


def mean_min_over_max(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Mean of (min / max) pointwise."""
    mins = fuzzy_intersection(mu1, mu2)
    maxs = fuzzy_union(mu1, mu2)
    ratios = safe_divide(mins, maxs, default=1.0)
    return float(np.mean(ratios))


def mean_dice_coefficient(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Mean of (2*min / (A+B)) pointwise (Dice-like)."""
    mins = fuzzy_intersection(mu1, mu2)
    sums = np.asarray(mu1) + np.asarray(mu2)
    ratios = safe_divide(2.0 * mins, sums, default=1.0)
    return float(np.mean(ratios))


def jaccard_negation(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Jaccard index applied to the negated sets."""
    return similarity_jaccard(fuzzy_negation(mu1), fuzzy_negation(mu2))


def intersection_over_max_cardinality(mu1: ArrayLike, mu2: ArrayLike) -> float:
    intersection = fuzzy_intersection(mu1, mu2)
    max_card = max(fuzzy_cardinality(mu1), fuzzy_cardinality(mu2))
    return safe_divide(fuzzy_cardinality(intersection), max_card, default=1.0)


def negated_intersection_over_max_cardinality(mu1: ArrayLike, mu2: ArrayLike) -> float:
    return intersection_over_max_cardinality(fuzzy_negation(mu1), fuzzy_negation(mu2))


def negated_overlap_coefficient(mu1: ArrayLike, mu2: ArrayLike) -> float:
    return similarity_overlap_coefficient(fuzzy_negation(mu1), fuzzy_negation(mu2))


def negated_symdiff_over_max_negated_component(mu1: ArrayLike, mu2: ArrayLike) -> float:
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    neg_mu1 = fuzzy_negation(mu1)
    neg_mu2 = fuzzy_negation(mu2)
    symm_diff = fuzzy_symmetric_difference(mu1, mu2)
    neg_symm_diff = fuzzy_negation(symm_diff)
    card_neg_symm_diff = fuzzy_cardinality(neg_symm_diff)
    comp1 = fuzzy_intersection(mu1, neg_mu2)
    comp2 = fuzzy_intersection(neg_mu1, mu2)
    max_card_neg_comp = max(
        fuzzy_cardinality(fuzzy_negation(comp1)),
        fuzzy_cardinality(fuzzy_negation(comp2)),
    )
    return safe_divide(card_neg_symm_diff, max_card_neg_comp, default=1.0)


def negated_symdiff_over_min_negated_component(mu1: ArrayLike, mu2: ArrayLike) -> float:
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    neg_mu1 = fuzzy_negation(mu1)
    neg_mu2 = fuzzy_negation(mu2)
    symm_diff = fuzzy_symmetric_difference(mu1, mu2)
    neg_symm_diff = fuzzy_negation(symm_diff)
    card_neg_symm_diff = fuzzy_cardinality(neg_symm_diff)
    comp1 = fuzzy_intersection(mu1, neg_mu2)
    comp2 = fuzzy_intersection(neg_mu1, mu2)
    min_card_neg_comp = min(
        fuzzy_cardinality(fuzzy_negation(comp1)),
        fuzzy_cardinality(fuzzy_negation(comp2)),
    )
    return safe_divide(card_neg_symm_diff, min_card_neg_comp, default=1.0)


def one_minus_mean_symmetric_difference(mu1: ArrayLike, mu2: ArrayLike) -> float:
    symm_diff = fuzzy_symmetric_difference(mu1, mu2)
    return 1.0 - float(np.mean(symm_diff)) if symm_diff.size else 1.0


def mean_one_minus_abs_diff(mu1: ArrayLike, mu2: ArrayLike) -> float:
    abs_diff = np.abs(np.asarray(mu1) - np.asarray(mu2))
    return 1.0 - float(np.mean(abs_diff)) if abs_diff.size else 1.0


def one_minus_abs_diff_over_sum_cardinality(mu1: ArrayLike, mu2: ArrayLike) -> float:
    denominator = fuzzy_cardinality(mu1) + fuzzy_cardinality(mu2)
    return 1.0 - safe_divide(distance_hamming(mu1, mu2), denominator)


def product_over_min_norm_squared(mu1: ArrayLike, mu2: ArrayLike) -> float:
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    numerator = float(np.sum(mu1 * mu2))
    denom = min(float(np.dot(mu1, mu1)), float(np.dot(mu2, mu2)))
    if denom < 1e-12:
        return 1.0 if np.allclose(mu1, 0) and np.allclose(mu2, 0) else 0.0
    return numerator / denom


def max_intersection(mu1: ArrayLike, mu2: ArrayLike) -> float:
    return float(np.max(fuzzy_intersection(mu1, mu2))) if np.asarray(mu1).size else 0.0


# -----------------------------------------------------------------------------
# 2. Distance‑based metrics and their similarity transforms
# -----------------------------------------------------------------------------


def distance_hamming(mu1: ArrayLike, mu2: ArrayLike) -> float:
    return float(np.sum(np.abs(np.asarray(mu1) - np.asarray(mu2))))


def similarity_hamming(mu1: ArrayLike, mu2: ArrayLike) -> float:
    n = np.asarray(mu1).size
    if n == 0:
        return 1.0
    return 1.0 - safe_divide(distance_hamming(mu1, mu2), float(n))


def distance_euclidean(mu1: ArrayLike, mu2: ArrayLike) -> float:
    diff = np.asarray(mu1) - np.asarray(mu2)
    return float(np.sqrt(np.sum(diff**2)))


def similarity_euclidean(mu1: ArrayLike, mu2: ArrayLike) -> float:
    return 1.0 / (1.0 + distance_euclidean(mu1, mu2) + 1e-9)


def distance_chebyshev(mu1: ArrayLike, mu2: ArrayLike) -> float:
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    return float(np.max(np.abs(mu1 - mu2))) if mu1.size else 0.0


def similarity_chebyshev(mu1: ArrayLike, mu2: ArrayLike) -> float:
    # Clip distance into [0, 1] assuming membership values themselves are bounded by 1.
    dist_c = min(distance_chebyshev(mu1, mu2), 1.0)
    return 1.0 - dist_c


def similarity_matlab_S2_W(mu1: ArrayLike, mu2: ArrayLike) -> float:
    abs_diff = np.abs(np.asarray(mu1) - np.asarray(mu2))
    return 1.0 - float(np.mean(abs_diff)) if abs_diff.size else 1.0


def similarity_matlab_S(mu1: ArrayLike, mu2: ArrayLike) -> float:
    denominator = fuzzy_cardinality(mu1) + fuzzy_cardinality(mu2)
    return 1.0 - safe_divide(distance_hamming(mu1, mu2), denominator)


# -----------------------------------------------------------------------------
# 3. Correlation‑based metrics
# -----------------------------------------------------------------------------


def similarity_cosine(mu1: ArrayLike, mu2: ArrayLike) -> float:
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    if not mu1.size:
        return 1.0
    dot = float(np.dot(mu1, mu2))
    denom = float(np.linalg.norm(mu1) * np.linalg.norm(mu2))
    if denom < 1e-12:
        return 1.0 if np.allclose(mu1, 0) and np.allclose(mu2, 0) else 0.0
    return dot / denom


def similarity_pearson(mu1: ArrayLike, mu2: ArrayLike) -> float:
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    if mu1.size < 2:
        return 0.0
    std1, std2 = np.std(mu1), np.std(mu2)
    if std1 < 1e-12 or std2 < 1e-12:
        return 1.0 if std1 < 1e-12 and std2 < 1e-12 else 0.0
    centred1, centred2 = mu1 - mu1.mean(), mu2 - mu2.mean()
    return float(
        np.dot(centred1, centred2)
        / (np.linalg.norm(centred1) * np.linalg.norm(centred2))
    )


def similarity_matlab_P(mu1: ArrayLike, mu2: ArrayLike) -> float:
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    numerator = float(np.sum(mu1 * mu2))
    denom = min(float(np.dot(mu1, mu1)), float(np.dot(mu2, mu2)))
    if denom < 1e-12:
        return 1.0 if np.allclose(mu1, 0) and np.allclose(mu2, 0) else 0.0
    return numerator / denom


# -----------------------------------------------------------------------------
# 4. Other MATLAB metrics
# -----------------------------------------------------------------------------


def similarity_matlab_T(mu1: ArrayLike, mu2: ArrayLike) -> float:
    return float(np.max(fuzzy_intersection(mu1, mu2))) if np.asarray(mu1).size else 0.0


# -----------------------------------------------------------------------------
# 5. Metric1 (approximation of MATLAB "Theirs")
# -----------------------------------------------------------------------------


def similarity_matlab_metric1(
    data_s1: ArrayLike,
    data_s2: ArrayLike,
    x_values_common: ArrayLike,
    fs_method: str = "nd",
    sigma_s2: Union[float, str, None] = None,
) -> float:
    """Approximation of MATLAB's proprietary similarity_metric1."""
    data_s1, data_s2, x_values_common = map(
        np.asarray, (data_s1, data_s2, x_values_common)
    )
    if data_s1.size < 2 or data_s2.size < 2 or not x_values_common.size:
        return np.nan

    # Delta: sum of IQR endpoints difference
    q1_s1, q3_s1 = np.percentile(data_s1, [25, 75])
    q1_s2, q3_s2 = np.percentile(data_s2, [25, 75])
    delta = abs(q1_s1 - q1_s2) + abs(q3_s1 - q3_s2)

    # Build fuzzy set for sensor2 only (as per MATLAB code)
    if fs_method == "nd":
        # Determine sigma
        if sigma_s2 is None:
            sigma_used = 0.1 * max(np.ptp(data_s2), 1e-9)
        elif isinstance(sigma_s2, str) and sigma_s2.startswith("r"):
            sigma_used = float(sigma_s2[1:]) * max(np.ptp(data_s2), 1e-9)
        else:
            sigma_used = float(sigma_s2)
        sigma_used = max(sigma_used, 1e-9)
        mu_s2 = compute_ndg_streaming(x_values_common, data_s2, sigma_used)
    elif fs_method == "kde":
        try:
            kde = gaussian_kde(data_s2)
            mu_s2 = np.clip(kde.evaluate(x_values_common), 0, None)
        except (np.linalg.LinAlgError, ValueError):
            return np.nan
    else:
        raise ValueError("fs_method must be 'nd' or 'kde'")

    if np.sum(mu_s2) < 1e-12:
        return 0.0

    interp_mu_s2 = interp1d(x_values_common, mu_s2, bounds_error=False, fill_value=0.0)
    sum_mu_vals = float(np.sum(interp_mu_s2(data_s1)))

    if delta < 1e-9:
        # Fallback – approximate delta
        step = np.mean(np.diff(x_values_common)) if x_values_common.size > 1 else 1e-3
        delta = max(step, 1e-9)

    return safe_divide(sum_mu_vals, len(data_s1) * delta)


# -----------------------------------------------------------------------------
# 6. Master orchestrator
# -----------------------------------------------------------------------------


def similarity_matlab_metric2(
    mu_s1: ArrayLike,
    mu_s2: ArrayLike,
    x_values: ArrayLike,
    *,
    data_s1: ArrayLike | None = None,
    data_s2: ArrayLike | None = None,
) -> float:
    """Implementation of MATLAB's similarity_metric2 which includes signal derivatives.

    This metric combines:
    1. Uses both the original membership functions and their derivatives
    2. Computes a weighted sum of memberships normalized by signal length
    3. Captures shape similarity through derivative comparison

    Args:
        mu_s1: Membership function values for signal 1
        mu_s2: Membership function values for signal 2
        x_values: Domain points where membership functions are evaluated
        data_s1: Optional raw signal 1 data (for IQR delta calculation)
        data_s2: Optional raw signal 2 data (for IQR delta calculation)
    """
    mu_s1, mu_s2, x_values = map(np.asarray, (mu_s1, mu_s2, x_values))
    if mu_s1.size < 2 or mu_s2.size < 2:
        return 0.0

    # Compute derivatives of membership functions
    d_mu_s1 = np.diff(mu_s1)
    d_mu_s2 = np.diff(mu_s2)
    # x_values_deriv = x_values[:-1]  # One fewer point for derivatives

    # Calculate delta (IQR difference) if raw data available, else use x-range
    if data_s1 is not None and data_s2 is not None:
        q1_s1, q3_s1 = np.percentile(data_s1, [25, 75])
        q1_s2, q3_s2 = np.percentile(data_s2, [25, 75])
        delta = abs(q1_s1 - q1_s2) + abs(q3_s1 - q3_s2)
    else:
        # Fallback: use range of x values
        delta = np.ptp(x_values)

    if delta < 1e-9:
        # Further fallback - use average step size
        step = np.mean(np.diff(x_values)) if x_values.size > 1 else 1e-3
        delta = max(step, 1e-9)

    # Compute weighted sum of memberships and their derivatives
    # We use n-1 points to match derivative length
    signal_contribution = mu_s1[:-1] * mu_s2[:-1]
    deriv_contribution = np.clip(1.0 - np.abs(d_mu_s1 - d_mu_s2), 0, 1)
    weighted_sum = float(np.sum(signal_contribution * deriv_contribution))

    return safe_divide(weighted_sum, (len(mu_s1) - 1) * delta)


# -----------------------------------------------------------------------------
# 7. Information-theoretic metrics
# -----------------------------------------------------------------------------


def similarity_jensen_shannon(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Jensen-Shannon divergence-based similarity for fuzzy sets.
    
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    
    Similarity = 1 - sqrt(JS(P||Q))
    """
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    
    # Normalize to probability distributions
    p = mu1 / (np.sum(mu1) + 1e-12)
    q = mu2 / (np.sum(mu2) + 1e-12)
    
    # Add small epsilon to avoid log(0)
    p = p + 1e-12
    q = q + 1e-12
    
    # Mixed distribution M = 0.5 * (P + Q)
    m = 0.5 * (p + q)
    
    # KL divergences
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    
    # Jensen-Shannon divergence
    js_div = 0.5 * kl_pm + 0.5 * kl_qm
    
    # Convert to similarity [0,1]
    return 1.0 - np.sqrt(js_div)


def similarity_mutual_information(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Mutual information based similarity using histogram estimation."""
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    if len(mu1) != len(mu2) or len(mu1) == 0:
        return 0.0
    
    # Determine number of bins based on data size
    n_bins = min(10, max(3, len(mu1) // 4))
    
    try:
        # Create joint histogram
        hist_2d, x_edges, y_edges = np.histogram2d(mu1, mu2, bins=n_bins)
        
        # Normalize to probabilities
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i,j] > 1e-12 and px[i] > 1e-12 and py[j] > 1e-12:
                    mi += pxy[i,j] * np.log(pxy[i,j] / (px[i] * py[j]))
        
        # Normalize by joint entropy
        h_xy = -np.sum(pxy[pxy > 1e-12] * np.log(pxy[pxy > 1e-12]))
        return mi / h_xy if h_xy > 1e-12 else 0.0
        
    except (ValueError, ZeroDivisionError):
        return 0.0


def similarity_renyi_divergence(mu1: ArrayLike, mu2: ArrayLike, alpha: float = 2.0) -> float:
    """Rényi divergence based similarity (α-divergence).
    
    Args:
        alpha: Order of the Rényi divergence (default 2.0)
               alpha=1.0 gives KL divergence
    """
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    
    # Normalize to probabilities
    p = mu1 / (np.sum(mu1) + 1e-12) + 1e-12
    q = mu2 / (np.sum(mu2) + 1e-12) + 1e-12
    
    try:
        if abs(alpha - 1.0) < 1e-6:
            # KL divergence case
            kl = np.sum(p * np.log(p / q))
            return np.exp(-kl)
        else:
            # General Rényi divergence
            if alpha > 0:
                renyi = (1.0 / (alpha - 1.0)) * np.log(np.sum(p**alpha * q**(1-alpha)))
                return np.exp(-np.abs(renyi))
            else:
                return 0.0
    except (ValueError, OverflowError, ZeroDivisionError):
        return 0.0


# -----------------------------------------------------------------------------
# 8. β-similarity metrics
# -----------------------------------------------------------------------------


def similarity_beta(mu1: ArrayLike, mu2: ArrayLike, beta: float = 1.0) -> float:
    """β-similarity metric for fuzzy sets.
    
    β-similarity generalizes several similarity measures:
    - β = 1: Jaccard index
    - β = 0.5: Dice coefficient
    - β → 0: Overlap coefficient
    
    S_β(A,B) = |A ∩ B| / (|A ∩ B| + β|A \\ B| + (1-β)|B \\ A|)
    """
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    
    intersection = fuzzy_intersection(mu1, mu2)
    card_intersection = fuzzy_cardinality(intersection)
    
    # Set differences: A \ B = A - (A ∩ B), B \ A = B - (A ∩ B)
    diff_1 = mu1 - intersection  # A \ B
    diff_2 = mu2 - intersection  # B \ A
    
    card_diff_1 = fuzzy_cardinality(np.maximum(diff_1, 0))
    card_diff_2 = fuzzy_cardinality(np.maximum(diff_2, 0))
    
    # β-similarity formula
    denominator = card_intersection + beta * card_diff_1 + (1 - beta) * card_diff_2
    
    return safe_divide(card_intersection, denominator, default=1.0)


# -----------------------------------------------------------------------------
# 9. Distribution-based metrics
# -----------------------------------------------------------------------------


def similarity_bhattacharyya_coefficient(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Bhattacharyya coefficient - measures overlap between distributions."""
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    
    # Normalize to probability distributions
    mu1_norm = mu1 / (np.sum(mu1) + 1e-12)
    mu2_norm = mu2 / (np.sum(mu2) + 1e-12)
    
    # Bhattacharyya coefficient
    return float(np.sum(np.sqrt(mu1_norm * mu2_norm)))


def similarity_bhattacharyya_distance(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Bhattacharyya distance = -ln(Bhattacharyya coefficient).
    
    Converted to similarity: 1 / (1 + distance)
    """
    coeff = similarity_bhattacharyya_coefficient(mu1, mu2)
    if coeff <= 1e-12:
        return 0.0
    
    distance = -np.log(coeff)
    return 1.0 / (1.0 + distance)


def similarity_hellinger(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Hellinger distance - bounded and symmetric measure.
    
    Returns similarity version: 1 - Hellinger_distance
    """
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    
    # Normalize to probability distributions
    mu1_norm = mu1 / (np.sum(mu1) + 1e-12)
    mu2_norm = mu2 / (np.sum(mu2) + 1e-12)
    
    # Hellinger distance computation
    sqrt_diff = np.sqrt(mu1_norm) - np.sqrt(mu2_norm)
    hellinger_dist = (1.0/np.sqrt(2.0)) * np.sqrt(np.sum(sqrt_diff**2))
    
    # Convert to similarity [0,1]
    return 1.0 - hellinger_dist


def similarity_earth_movers_distance(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Earth Mover's Distance (1-Wasserstein) approximation.
    
    Approximated as L1 distance between CDFs.
    """
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    if len(mu1) == 0 or len(mu2) == 0:
        return 0.0
    
    # Ensure same length for CDF comparison
    if len(mu1) != len(mu2):
        min_len = min(len(mu1), len(mu2))
        mu1 = mu1[:min_len]
        mu2 = mu2[:min_len]
    
    # Compute CDFs
    sum1, sum2 = np.sum(mu1), np.sum(mu2)
    if sum1 <= 1e-12 or sum2 <= 1e-12:
        return 0.0
        
    cdf1 = np.cumsum(mu1) / sum1
    cdf2 = np.cumsum(mu2) / sum2
    
    # EMD as L1 distance between CDFs
    emd = np.sum(np.abs(cdf1 - cdf2))
    return 1.0 / (1.0 + emd)  # Convert to similarity


def similarity_energy_distance(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Energy distance based similarity."""
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    n, m = len(mu1), len(mu2)
    
    if n == 0 or m == 0:
        return 0.0
    
    try:
        # Energy distance computation
        # E[|X-Y|] term - cross-distances
        cross_sum = sum(abs(float(x) - float(y)) for x in mu1 for y in mu2)
        cross_term = cross_sum / (n * m)
        
        # E[|X-X'|] term - within-group distances
        if n > 1:
            within_1_sum = sum(abs(float(mu1[i]) - float(mu1[j])) 
                             for i in range(n) for j in range(n) if i != j)
            within_1 = within_1_sum / (n * (n - 1))
        else:
            within_1 = 0.0
        
        # E[|Y-Y'|] term - within-group distances  
        if m > 1:
            within_2_sum = sum(abs(float(mu2[i]) - float(mu2[j])) 
                             for i in range(m) for j in range(m) if i != j)
            within_2 = within_2_sum / (m * (m - 1))
        else:
            within_2 = 0.0
        
        # Energy distance
        energy_dist = 2 * cross_term - within_1 - within_2
        return 1.0 / (1.0 + abs(energy_dist))  # Convert to similarity
        
    except (ValueError, OverflowError):
        return 0.0


def similarity_harmonic_mean(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Harmonic mean based similarity measure."""
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    
    if len(mu1) != len(mu2):
        min_len = min(len(mu1), len(mu2))
        mu1 = mu1[:min_len]
        mu2 = mu2[:min_len]
    
    if len(mu1) == 0:
        return 0.0
    
    # Element-wise harmonic mean where both values are positive
    harmonic_means = []
    for i in range(len(mu1)):
        if mu1[i] > 1e-12 and mu2[i] > 1e-12:
            harmonic_means.append(2 * mu1[i] * mu2[i] / (mu1[i] + mu2[i]))
        else:
            harmonic_means.append(0.0)
    
    # Overall similarity as mean of harmonic means
    return float(np.mean(harmonic_means))


# -----------------------------------------------------------------------------
# 10. Signal processing metrics
# -----------------------------------------------------------------------------


def similarity_cross_correlation(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """Normalized cross-correlation similarity."""
    mu1, mu2 = map(np.asarray, (mu1, mu2))
    if len(mu1) == 0 or len(mu2) == 0:
        return 0.0
    
    # Ensure same length
    if len(mu1) != len(mu2):
        min_len = min(len(mu1), len(mu2))
        mu1 = mu1[:min_len]
        mu2 = mu2[:min_len]
    
    # Zero-mean normalization
    mu1_mean, mu2_mean = np.mean(mu1), np.mean(mu2)
    mu1_norm = mu1 - mu1_mean
    mu2_norm = mu2 - mu2_mean
    
    # Auto-correlation normalization factors
    norm1 = np.sum(mu1_norm**2)
    norm2 = np.sum(mu2_norm**2)
    norm_factor = np.sqrt(norm1 * norm2)
    
    if norm_factor < 1e-12:
        return 1.0 if np.allclose(mu1, mu2) else 0.0
    
    # Cross-correlation (zero-lag)
    cross_corr = np.sum(mu1_norm * mu2_norm)
    return float(cross_corr / norm_factor)


def calculate_all_similarity_metrics(
    mu_s1: ArrayLike,
    mu_s2: ArrayLike,
    x_values: ArrayLike,
    *,
    data_s1: ArrayLike | None = None,
    data_s2: ArrayLike | None = None,
    fs_method: str = "nd",
    sigma_s2: Union[float, str, None] = None,
    normalise: bool = False,
) -> Dict[str, float]:
    """Compute a suite of similarity metrics.

    Args:
        mu_s1 / mu_s2: membership vectors (same shape as *x_values*).
        x_values: domain of the membership functions.
        data_s1 / data_s2: raw sensor signals (needed for Metric1).
        fs_method / sigma_s2: forwarded to Metric1 when used.
        normalise: if *True* the MFs are rescaled to sum to one.
    """
    mu_s1 = np.asarray(mu_s1, dtype=float)
    mu_s2 = np.asarray(mu_s2, dtype=float)
    x_values = np.asarray(x_values, dtype=float)

    if mu_s1.shape != mu_s2.shape or mu_s1.shape != x_values.shape:
        raise ValueError("mu_s1, mu_s2 and x_values must share the same shape")
    if not mu_s1.size:
        return {}

    if normalise:
        mu_s1 = mu_s1 / max(np.sum(mu_s1), 1e-12)
        mu_s2 = mu_s2 / max(np.sum(mu_s2), 1e-12)

    # Registry of metrics ------------------------------------------------------
    MetricFunc = Callable[[ArrayLike, ArrayLike], float]
    metric_funcs: Dict[str, MetricFunc] = {
        # Set‑theoretic / overlap
        "Jaccard": similarity_jaccard,
        "Dice": similarity_dice,
        "OverlapCoefficient": similarity_overlap_coefficient,
        "MeanMinOverMax": mean_min_over_max,
        "MeanDiceCoefficient": mean_dice_coefficient,
        "IntersectionOverMaxCardinality": intersection_over_max_cardinality,
        "JaccardNegation": jaccard_negation,
        "NegatedIntersectionOverMaxCardinality": negated_intersection_over_max_cardinality,
        "NegatedOverlapCoefficient": negated_overlap_coefficient,
        "NegatedSymDiffOverMaxNegatedComponent": negated_symdiff_over_max_negated_component,
        "NegatedSymDiffOverMinNegatedComponent": negated_symdiff_over_min_negated_component,
        "OneMinusMeanSymmetricDifference": one_minus_mean_symmetric_difference,
        # Distance‑based
        "Similarity_Hamming": similarity_hamming,
        "Similarity_Euclidean": similarity_euclidean,
        "Similarity_Chebyshev": similarity_chebyshev,
        "Distance_Hamming": distance_hamming,
        "Distance_Euclidean": distance_euclidean,
        "Distance_Chebyshev": distance_chebyshev,
        # Note: MeanOneMinusAbsDiff removed (identical to Similarity_Hamming)
        "OneMinusAbsDiffOverSumCardinality": one_minus_abs_diff_over_sum_cardinality,
        # Correlation‑based
        "Cosine": similarity_cosine,
        "Pearson": similarity_pearson,
        "ProductOverMinNormSquared": product_over_min_norm_squared,
        "CrossCorrelation": similarity_cross_correlation,
        # Information-theoretic
        "JensenShannon": similarity_jensen_shannon,
        "MutualInformation": similarity_mutual_information,
        # Note: RenyiDivergence removed (identical to RenyiDivergence_2.0 with default alpha=2.0)
        "RenyiDivergence_0.5": lambda mu1, mu2: similarity_renyi_divergence(mu1, mu2, 0.5),
        "RenyiDivergence_2.0": lambda mu1, mu2: similarity_renyi_divergence(mu1, mu2, 2.0),
        # β-similarity variants (unique cases only)
        "Beta_0.1": lambda mu1, mu2: similarity_beta(mu1, mu2, 0.1),
        "Beta_2.0": lambda mu1, mu2: similarity_beta(mu1, mu2, 2.0),
        # Distribution-based
        "BhattacharyyaCoefficient": similarity_bhattacharyya_coefficient,
        "BhattacharyyaDistance": similarity_bhattacharyya_distance,
        "HellingerDistance": similarity_hellinger,
        "EarthMoversDistance": similarity_earth_movers_distance,
        "EnergyDistance": similarity_energy_distance,
        "HarmonicMean": similarity_harmonic_mean,
        # Others
        "MaxIntersection": max_intersection,
    }

    results: Dict[str, float] = {}
    for name, func in metric_funcs.items():
        try:
            results[name] = func(mu_s1, mu_s2)
        except Exception as exc:  # noqa: BLE001 – broad but logged
            print(f"Metric '{name}' failed: {exc}")
            results[name] = np.nan

    # Metric1 (needs raw data)
    results["CustomMetric1_SumMembershipOverIQRDelta"] = (
        similarity_matlab_metric1(
            data_s1, data_s2, x_values, fs_method=fs_method, sigma_s2=sigma_s2
        )
        if data_s1 is not None and data_s2 is not None
        else np.nan
    )

    results["CustomMetric2_DerivativeWeightedSimilarity"] = similarity_matlab_metric2(
        mu_s1, mu_s2, x_values, data_s1=data_s1, data_s2=data_s2
    )

    # Preferred presentation order -------------------------------------------
    preferred_order = [
        # Core similarity metrics
        "Jaccard",
        "Dice", 
        "OverlapCoefficient",
        "Cosine",
        "Pearson",
        "CrossCorrelation",
        # Distance-based similarities
        "Similarity_Hamming",
        "Similarity_Euclidean",
        "Similarity_Chebyshev",
        "Distance_Hamming",
        "Distance_Euclidean", 
        "Distance_Chebyshev",
        # Information-theoretic
        "JensenShannon",
        "MutualInformation",
        "RenyiDivergence_0.5",
        "RenyiDivergence_2.0",
        # β-similarity variants (unique cases only)
        "Beta_0.1",
        "Beta_2.0",
        # Distribution-based
        "BhattacharyyaCoefficient",
        "BhattacharyyaDistance",
        "HellingerDistance",
        "EarthMoversDistance",
        "EnergyDistance",
        "HarmonicMean",
        # Custom metrics
        "CustomMetric1_SumMembershipOverIQRDelta",
        "CustomMetric2_DerivativeWeightedSimilarity",
        # Advanced set-theoretic
        "MeanMinOverMax",
        "MeanDiceCoefficient",
        "MaxIntersection",
        "ProductOverMinNormSquared",
        "OneMinusAbsDiffOverSumCardinality",
        "IntersectionOverMaxCardinality",
        "JaccardNegation",
        "NegatedIntersectionOverMaxCardinality",
        "NegatedOverlapCoefficient",
        "NegatedSymDiffOverMaxNegatedComponent",
        "NegatedSymDiffOverMinNegatedComponent",
        "OneMinusMeanSymmetricDifference",
    ]

    return {key: results[key] for key in preferred_order if key in results}

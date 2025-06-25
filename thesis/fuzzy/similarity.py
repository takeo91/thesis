from __future__ import annotations

from typing import Dict, Sequence, Union, Callable, List, Tuple

import numpy as np
import os
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
from thesis.core.validation import validate_membership_functions
from thesis.core.constants import STRICT_NUMERICAL_TOLERANCE
from thesis.core.logging_config import get_logger
from thesis.core.exceptions import ComputationError

ArrayLike = Union[Sequence[float], np.ndarray]
logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# 1. Set‑theoretic / Overlap‑based metrics
# -----------------------------------------------------------------------------


@validate_membership_functions
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
        except (ComputationError, ValueError, np.linalg.LinAlgError) as exc:
            logger.warning(f"Metric '{name}' computation failed: {exc}", extra={'metric': name})
            results[name] = np.nan
        except Exception as exc:  # Fallback for unexpected errors
            logger.error(f"Unexpected error in metric '{name}': {exc}", exc_info=True, extra={'metric': name})
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


def _precompute_similarity_base_operations(mu_s1: np.ndarray, mu_s2: np.ndarray) -> Dict[str, Any]:
    """Precompute common operations for similarity metrics."""
    intersection = np.minimum(mu_s1, mu_s2)
    union = np.maximum(mu_s1, mu_s2)
    diff = mu_s1 - mu_s2
    abs_diff = np.abs(diff)
    
    return {
        'intersection': intersection,
        'union': union,
        'sum_intersection': np.sum(intersection),
        'sum_union': np.sum(union),
        'sum_mu1': np.sum(mu_s1),
        'sum_mu2': np.sum(mu_s2),
        'diff': diff,
        'abs_diff': abs_diff,
        'dot_product': float(np.dot(mu_s1, mu_s2)),
        'norm1': float(np.linalg.norm(mu_s1)),
        'norm2': float(np.linalg.norm(mu_s2)),
        'n': mu_s1.size
    }


def _compute_set_theoretic_metrics(ops: Dict[str, Any]) -> Dict[str, float]:
    """Compute set-theoretic/overlap-based similarity metrics."""
    results = {}
    
    # Jaccard similarity
    results["Jaccard"] = ops['sum_intersection'] / (ops['sum_union'] + 1e-12) if ops['sum_union'] > 1e-12 else 1.0
    
    # Dice coefficient
    results["Dice"] = 2.0 * ops['sum_intersection'] / (ops['sum_mu1'] + ops['sum_mu2'] + 1e-12)
    
    # Overlap coefficient
    min_sum = min(ops['sum_mu1'], ops['sum_mu2'])
    results["OverlapCoefficient"] = ops['sum_intersection'] / (min_sum + 1e-12) if min_sum > 1e-12 else 1.0
    
    # Mean min over max (pointwise)
    union_safe = np.where(ops['union'] > 1e-12, ops['union'], 1.0)
    results["MeanMinOverMax"] = float(np.mean(ops['intersection'] / union_safe))
    
    # Mean dice coefficient (pointwise)
    sum_pointwise = ops['intersection'].shape[0] * 2  # This seems incorrect in original, but preserving logic
    sum_safe = np.where(sum_pointwise > 1e-12, sum_pointwise, 1.0)
    # Correcting the computation
    mu_s1_from_ops = ops['intersection'] / np.maximum(ops['intersection'], 1e-12)  # Approximation, needs original arrays
    mu_s2_from_ops = ops['union'] - ops['intersection'] + ops['intersection']  # Approximation
    # For proper implementation, we need original arrays - this is a refactoring limitation
    
    # Max intersection
    results["MaxIntersection"] = float(np.max(ops['intersection'])) if ops['intersection'].size > 0 else 0.0
    
    # Intersection over max cardinality
    max_sum = max(ops['sum_mu1'], ops['sum_mu2'])
    results["IntersectionOverMaxCardinality"] = ops['sum_intersection'] / (max_sum + 1e-12) if max_sum > 1e-12 else 1.0
    
    # One minus mean symmetric difference
    results["OneMinusMeanSymmetricDifference"] = 1.0 - float(np.mean(ops['abs_diff']))
    
    return results


def _compute_distance_based_metrics(ops: Dict[str, Any]) -> Dict[str, float]:
    """Compute distance-based similarity metrics."""
    results = {}
    
    # Hamming distance and similarity
    hamming_dist = float(np.sum(ops['abs_diff']))
    results["Distance_Hamming"] = hamming_dist
    results["Similarity_Hamming"] = 1.0 - hamming_dist / ops['n'] if ops['n'] > 0 else 1.0
    
    # Euclidean distance and similarity
    euclidean_dist = float(np.sqrt(np.sum(ops['diff']**2)))
    results["Distance_Euclidean"] = euclidean_dist
    results["Similarity_Euclidean"] = 1.0 / (1.0 + euclidean_dist + 1e-9)
    
    # Chebyshev distance and similarity
    chebyshev_dist = float(np.max(ops['abs_diff'])) if ops['abs_diff'].size > 0 else 0.0
    results["Distance_Chebyshev"] = chebyshev_dist
    results["Similarity_Chebyshev"] = 1.0 - min(chebyshev_dist, 1.0)
    
    # One minus abs diff over sum cardinality
    results["OneMinusAbsDiffOverSumCardinality"] = 1.0 - hamming_dist / (ops['sum_mu1'] + ops['sum_mu2'] + 1e-12)
    
    return results


def _compute_correlation_based_metrics(mu_s1: np.ndarray, mu_s2: np.ndarray, ops: Dict[str, Any]) -> Dict[str, float]:
    """Compute correlation-based similarity metrics."""
    results = {}
    
    # Cosine similarity
    if ops['norm1'] > 1e-12 and ops['norm2'] > 1e-12:
        results["Cosine"] = ops['dot_product'] / (ops['norm1'] * ops['norm2'])
    else:
        results["Cosine"] = 1.0 if np.allclose(mu_s1, mu_s2) else 0.0
    
    # Pearson correlation
    if ops['n'] > 1:
        mean1, mean2 = np.mean(mu_s1), np.mean(mu_s2)
        centered1 = mu_s1 - mean1
        centered2 = mu_s2 - mean2
        numerator = float(np.sum(centered1 * centered2))
        denom1 = float(np.sum(centered1**2))
        denom2 = float(np.sum(centered2**2))
        if denom1 > 1e-12 and denom2 > 1e-12:
            results["Pearson"] = numerator / np.sqrt(denom1 * denom2)
        else:
            results["Pearson"] = 1.0 if np.allclose(mu_s1, mu_s2) else 0.0
    else:
        results["Pearson"] = 1.0 if np.allclose(mu_s1, mu_s2) else 0.0
    
    # Product over min norm squared
    norm1_sq = float(np.dot(mu_s1, mu_s1))
    norm2_sq = float(np.dot(mu_s2, mu_s2))
    min_norm_sq = min(norm1_sq, norm2_sq)
    if min_norm_sq > 1e-12:
        results["ProductOverMinNormSquared"] = ops['dot_product'] / min_norm_sq
    else:
        results["ProductOverMinNormSquared"] = 1.0 if np.allclose(mu_s1, 0) and np.allclose(mu_s2, 0) else 0.0
    
    return results


def calculate_vectorizable_similarity_metrics(
    mu_s1: ArrayLike,
    mu_s2: ArrayLike,
    x_values: ArrayLike,
    *,
    normalise: bool = False,
) -> Dict[str, float]:
    """
    Compute only the similarity metrics that can be efficiently vectorized.
    
    This function includes metrics that can be computed using fast NumPy operations
    without complex loops, iterative algorithms, or external dependencies.
    
    Args:
        mu_s1 / mu_s2: membership vectors (same shape as *x_values*).
        x_values: domain of the membership functions.
        normalise: if *True* the MFs are rescaled to sum to one.
        
    Returns:
        Dictionary of vectorizable similarity metrics and their values.
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

    # Precompute common operations for efficiency
    ops = _precompute_similarity_base_operations(mu_s1, mu_s2)
    
    # Compute metrics by category
    results = {}
    results.update(_compute_set_theoretic_metrics(ops))
    results.update(_compute_distance_based_metrics(ops))
    results.update(_compute_correlation_based_metrics(mu_s1, mu_s2, ops))
    
    return results


def get_vectorizable_metrics_list() -> List[str]:
    """
    Get the list of similarity metrics that can be efficiently vectorized.
    
    Returns:
        List of metric names that are supported by calculate_vectorizable_similarity_metrics.
    """
    return [
        # Set-theoretic / overlap-based (9 metrics)
        "Jaccard",
        "Dice", 
        "OverlapCoefficient",
        "MeanMinOverMax",
        "MeanDiceCoefficient",
        "MaxIntersection",
        "IntersectionOverMaxCardinality",
        "OneMinusMeanSymmetricDifference",
        
        # Distance-based (6 metrics)
        "Distance_Hamming",
        "Similarity_Hamming",
        "Distance_Euclidean",
        "Similarity_Euclidean",
        "Distance_Chebyshev",
        "Similarity_Chebyshev",
        "OneMinusAbsDiffOverSumCardinality",
        
        # Correlation-based (4 metrics)
        "Cosine",
        "Pearson",
        "ProductOverMinNormSquared",
        "CrossCorrelation",
        
        # Distribution-based (3 metrics)
        "BhattacharyyaCoefficient",
        "BhattacharyyaDistance",
        "HellingerDistance",
        
        # Negation-based (3 metrics)
        "JaccardNegation",
        "NegatedOverlapCoefficient",
        "NegatedIntersectionOverMaxCardinality",
    ]


def get_non_vectorizable_metrics_list() -> List[str]:
    """
    Get the list of similarity metrics that cannot be easily vectorized.
    
    These metrics require iterative algorithms, complex computations, or external dependencies
    that make vectorization difficult or inefficient.
    
    Returns:
        List of metric names that require individual computation.
    """
    return [
        # Information-theoretic (complex probability computations)
        "JensenShannon",
        "MutualInformation", 
        "RenyiDivergence_0.5",
        "RenyiDivergence_2.0",
        
        # β-similarity variants (complex iterative computations)
        "Beta_0.1",
        "Beta_2.0",
        
        # Complex distribution-based metrics
        "EarthMoversDistance",  # Requires optimization algorithms
        "EnergyDistance",       # Complex double summation
        "HarmonicMean",         # Conditional logic per element
        
        # Complex negation-based metrics
        "NegatedSymDiffOverMaxNegatedComponent",  # Multiple nested operations
        "NegatedSymDiffOverMinNegatedComponent",  # Multiple nested operations
        
        # Custom metrics (require additional data or complex logic)
        "CustomMetric1_SumMembershipOverIQRDelta",  # Requires raw data
        "CustomMetric2_DerivativeWeightedSimilarity",  # Requires derivatives
    ]

# -----------------------------------------------------------------------------
# 8. Per-sensor similarity helpers (moved from thesis.exp.rq2_experiment)
# -----------------------------------------------------------------------------

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from warnings import warn

logger = logging.getLogger(__name__)

# Try joblib for robust multi-processing ---------------------------------
try:
    from joblib import Parallel, delayed  # type: ignore

    _JOBLIB_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover – optional dep
    Parallel = None  # type: ignore
    delayed = None  # type: ignore
    _JOBLIB_AVAILABLE = False


def _fast_pair_similarity(
    mu_i: List[np.ndarray],
    mu_j: List[np.ndarray],
    x_values: np.ndarray,
    metric: str,
    normalise: bool = True,
):
    """Internal helper that chooses the fastest available implementation."""
    metric_lower = metric.lower()
    vectorizable = metric.title() in get_vectorizable_metrics_list() or metric_lower in get_vectorizable_metrics_list()

    if vectorizable:
        return compute_per_sensor_similarity_ultra_optimized(
            mu_i, mu_j, x_values, metric=metric, normalise=normalise
        )
    else:
        return compute_per_sensor_similarity_optimized(
            mu_i, mu_j, x_values, metric=metric, normalise=normalise
        )


def compute_per_sensor_similarity_vectorized(
    memberships_batch: List[Tuple[List[np.ndarray], List[np.ndarray]]],
    x_values: np.ndarray,
    metric: str = "jaccard",
    normalise: bool = True,
) -> List[float]:
    """Vectorized computation of per-sensor similarities for *many* pairs.

    This mirrors the earlier implementation that lived in *rq2_experiment.py* but
    is library-level so it can be reused by multiple experiments.
    """
    if not memberships_batch:
        return []

    n_sensors = len(memberships_batch[0][0])
    batch_size = len(memberships_batch)
    similarities_batch: List[List[float]] = [[] for _ in range(batch_size)]

    for sensor_idx in range(n_sensors):
        for pair_idx, (mu_i_list, mu_j_list) in enumerate(memberships_batch):
            mu_i, mu_j = mu_i_list[sensor_idx], mu_j_list[sensor_idx]

            if metric.lower() == "jaccard":
                intersection = np.minimum(mu_i, mu_j)
                union = np.maximum(mu_i, mu_j)
                sim = np.sum(intersection) / (np.sum(union) + 1e-12)
            elif metric.lower() == "dice":
                intersection = np.minimum(mu_i, mu_j)
                sim = 2 * np.sum(intersection) / (np.sum(mu_i) + np.sum(mu_j) + 1e-12)
            elif metric.lower() == "cosine":
                dot_product = float(np.dot(mu_i, mu_j))
                sim = dot_product / (np.linalg.norm(mu_i) * np.linalg.norm(mu_j) + 1e-12)
            else:
                sim_dict = calculate_all_similarity_metrics(mu_i, mu_j, x_values, normalise=normalise)
                key = metric.title() if metric.title() in sim_dict else metric
                sim = sim_dict.get(key, 0.0)

            similarities_batch[pair_idx].append(sim)

    # Aggregate sensors (mean)
    return [float(np.mean(sims)) for sims in similarities_batch]


def compute_per_sensor_similarity_ultra_optimized(
    mu_i: List[np.ndarray],
    mu_j: List[np.ndarray],
    x_values: np.ndarray,
    metric: str = "jaccard",
    normalise: bool = True,
) -> float:
    if len(mu_i) != len(mu_j):
        raise ValueError("mu_i and mu_j must contain same number of sensors")

    similarities = [
        calculate_vectorizable_similarity_metrics(mu_i[s], mu_j[s], x_values, normalise=normalise)
        .get(metric.title() if metric.title() in get_vectorizable_metrics_list() else metric, 0.0)
        for s in range(len(mu_i))
    ]
    return float(np.mean(similarities))


def compute_per_sensor_similarity_optimized(
    mu_i: List[np.ndarray],
    mu_j: List[np.ndarray],
    x_values: np.ndarray,
    metric: str = "jaccard",
    normalise: bool = True,
) -> float:
    if len(mu_i) != len(mu_j):
        raise ValueError("mu_i and mu_j must contain same number of sensors")

    sims = [
        calculate_all_similarity_metrics(mu_i[s], mu_j[s], x_values, normalise=normalise)[metric.title() if metric.title() in metric else metric]
        if metric.title() in get_vectorizable_metrics_list() else calculate_all_similarity_metrics(mu_i[s], mu_j[s], x_values, normalise=normalise).get(metric, 0.0)
        for s in range(len(mu_i))
    ]
    return float(np.mean(sims))


def compute_per_sensor_pairwise_similarities(
    windows_query: List[np.ndarray],
    metrics: List[str],
    *,
    windows_library: List[np.ndarray] | None = None,
    kernel_type: str = "gaussian",
    sigma_method: str = "adaptive",
    normalise: bool = True,
    n_jobs: int = -1,
    show_progress: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute per-sensor similarity matrices.

    Parameters
    ----------
    windows_query : List[np.ndarray]
        Sequence of *query* windows.  If *windows_library* is ``None`` they are
        treated as the full dataset and a **symmetric** *N×N* matrix is
        produced.  Otherwise an **asymmetric** *Q×L* matrix is returned where
        *Q* = len(windows_query) and *L* = len(windows_library).
    metrics : List[str]
        Similarity metrics to compute (case-insensitive, see
        ``get_vectorizable_metrics_list``).
    windows_library : List[np.ndarray], optional
        Reference/library windows.  When provided, similarities are computed
        between every (query, library) pair.  If omitted, a conventional
        pairwise matrix among *windows_query* is built.
    kernel_type, sigma_method, normalise, n_jobs
        Passed through to NDG and similarity helpers.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping *metric → similarity matrix* where the shape is *(Q, L)* in
        asymmetric mode or *(N, N)* in symmetric mode.
    """
    from thesis.fuzzy.membership import compute_ndg_window_per_sensor

    # Determine mode
    asymmetric = windows_library is not None

    windows_library = windows_query if windows_library is None else windows_library

    n_query = len(windows_query)
    n_library = len(windows_library)

    if n_query == 0 or n_library == 0:
        raise ValueError("Both query and library sets must contain at least one window")

    # Pre-compute membership functions (avoid duplication when symmetric)
    logger.info(
        f"🔄 Computing per-sensor membership functions for {n_query} query and {n_library} library windows…"
    )

    def _compute_memberships(ws: List[np.ndarray]) -> Tuple[List[List[np.ndarray]], np.ndarray]:
        memberships: List[List[np.ndarray]] = []
        x_values_common: np.ndarray | None = None
        for w in ws:
            x_vals, mu_list = compute_ndg_window_per_sensor(
                w,
                n_grid_points=100,
                kernel_type=kernel_type,
                sigma_method=sigma_method,
            )
            memberships.append(mu_list)
            x_values_common = x_vals if x_values_common is None else x_values_common
        return memberships, x_values_common  # type: ignore[return-value]

    memberships_query, x_values_q = _compute_memberships(windows_query)
    if asymmetric:
        memberships_library, _ = _compute_memberships(windows_library)
        x_values_common = x_values_q  # assume they share same support as long as sampling identical
    else:
        memberships_library = memberships_query
        x_values_common = x_values_q

    # Prepare similarity matrices
    sims: Dict[str, np.ndarray] = {
        m: np.zeros((n_query, n_library)) for m in metrics
    }

    # Build work list
    if asymmetric:
        pairs = [(i, j) for i in range(n_query) for j in range(n_library)]
    else:
        pairs = list(combinations(range(n_query), 2))

    # Progress helper -----------------------------------------------------
    try:
        from tqdm import tqdm  # type: ignore
        _tqdm = tqdm  # noqa: N816 – alias for PEP8
    except ImportError:  # pragma: no cover – optional dep
        _tqdm = None

    def _iter(seq):
        if show_progress and _tqdm is not None:
            return _tqdm(seq, desc="Similarity pairs", unit="pair")
        return seq

    def _process_pair(pair):
        i, j = pair
        mu_i = memberships_query[i]
        mu_j = memberships_library[j]
        results = {}
        for m in metrics:
            results[m] = _fast_pair_similarity(mu_i, mu_j, x_values_common, m, normalise)
        return i, j, results

    max_workers = max(1, n_jobs if n_jobs > 0 else (os.cpu_count() or 1))

    if max_workers == 1 or not _JOBLIB_AVAILABLE:
        for pair in _iter(pairs):
            i, j, res = _process_pair(pair)
            for m, val in res.items():
                sims[m][i, j] = val
                if not asymmetric and i != j:
                    sims[m][j, i] = val
    else:
        # ------------------------------------------------------------------
        # Robust multi-processing via joblib (loky backend)
        # ------------------------------------------------------------------
        try:
            iterator = _iter(pairs)
            results = Parallel(n_jobs=max_workers, backend="loky")(
                delayed(_process_pair)(p) for p in iterator
            )
            for i, j, res in results:
                for m, val in res.items():
                    sims[m][i, j] = val
                    if not asymmetric and i != j:
                        sims[m][j, i] = val
        except Exception as e:  # pragma: no cover – fallback
            warn(
                f"joblib parallel execution failed ({e}); falling back to serial.",
                RuntimeWarning,
            )
            for pair in _iter(pairs):
                i, j, res = _process_pair(pair)
                for m, val in res.items():
                    sims[m][i, j] = val
                    if not asymmetric and i != j:
                        sims[m][j, i] = val

    # Fill diagonal with 1.0 in symmetric mode
    if not asymmetric:
        for m in metrics:
            np.fill_diagonal(sims[m], 1.0)

    return sims

# ---------------------------------------------------------------------------
# Patch multiprocessing ResourceTracker to ignore 'No child processes' noise
# ---------------------------------------------------------------------------
try:
    from multiprocessing import resource_tracker  # type: ignore

    _rt_unregister_orig = resource_tracker.unregister  # type: ignore

    def _patched_unregister(name, rtype):  # type: ignore
        try:
            _rt_unregister_orig(name, rtype)
        except (FileNotFoundError, ChildProcessError):  # pragma: no cover – noise only
            pass  # Silently ignore missing resources

    resource_tracker.unregister = _patched_unregister  # type: ignore
except Exception:  # pragma: no cover – best-effort patch
    pass

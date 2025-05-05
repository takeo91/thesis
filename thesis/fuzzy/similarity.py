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
        "MeanOneMinusAbsDiff": mean_one_minus_abs_diff,
        "OneMinusAbsDiffOverSumCardinality": one_minus_abs_diff_over_sum_cardinality,
        # Correlation‑based
        "Cosine": similarity_cosine,
        "Pearson": similarity_pearson,
        "ProductOverMinNormSquared": product_over_min_norm_squared,
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
        # Generic
        "Jaccard",
        "Dice",
        "OverlapCoefficient",
        "Cosine",
        "Pearson",
        "Similarity_Hamming",
        "Similarity_Euclidean",
        "Similarity_Chebyshev",
        "Distance_Hamming",
        "Distance_Euclidean",
        "Distance_Chebyshev",
        # Custom metrics
        "CustomMetric1_SumMembershipOverIQRDelta",
        "CustomMetric2_DerivativeWeightedSimilarity",
        # Renamed MATLAB metrics
        "MeanMinOverMax",
        "MeanDiceCoefficient",
        "MaxIntersection",
        "ProductOverMinNormSquared",
        "MeanOneMinusAbsDiff",
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

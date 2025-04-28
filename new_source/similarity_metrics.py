from __future__ import annotations

from typing import Dict, Sequence, Union, Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde  # Used only in similarity_matlab_metric1

# Local helpers – assumed available in the same package
from fuzzy_helpers import (
    safe_divide,
    fuzzy_intersection,
    fuzzy_union,
    fuzzy_negation,
    fuzzy_cardinality,
    fuzzy_symmetric_difference,
)
from membership_functions import compute_ndg

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
# MATLAB‑specific variants built from the above primitives
# -----------------------------------------------------------------------------

def similarity_matlab_M(mu1: ArrayLike, mu2: ArrayLike) -> float:
    return similarity_jaccard(mu1, mu2)


def similarity_matlab_S1(mu1: ArrayLike, mu2: ArrayLike) -> float:
    mins = fuzzy_intersection(mu1, mu2)
    maxs = fuzzy_union(mu1, mu2)
    ratios = safe_divide(mins, maxs, default=1.0)
    return float(np.mean(ratios))


def similarity_matlab_S3(mu1: ArrayLike, mu2: ArrayLike) -> float:
    mins = fuzzy_intersection(mu1, mu2)
    sums = np.asarray(mu1) + np.asarray(mu2)
    ratios = safe_divide(2.0 * mins, sums, default=1.0)
    return float(np.mean(ratios))


def similarity_matlab_S5(mu1: ArrayLike, mu2: ArrayLike) -> float:
    intersection = fuzzy_intersection(mu1, mu2)
    max_card = max(fuzzy_cardinality(mu1), fuzzy_cardinality(mu2))
    return safe_divide(fuzzy_cardinality(intersection), max_card, default=1.0)


def similarity_matlab_S4(mu1: ArrayLike, mu2: ArrayLike) -> float:
    return similarity_jaccard(fuzzy_negation(mu1), fuzzy_negation(mu2))


def similarity_matlab_S6(mu1: ArrayLike, mu2: ArrayLike) -> float:
    return similarity_matlab_S5(fuzzy_negation(mu1), fuzzy_negation(mu2))


def similarity_matlab_S8(mu1: ArrayLike, mu2: ArrayLike) -> float:
    return similarity_overlap_coefficient(fuzzy_negation(mu1), fuzzy_negation(mu2))


def similarity_matlab_S9(mu1: ArrayLike, mu2: ArrayLike) -> float:
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


def similarity_matlab_S10(mu1: ArrayLike, mu2: ArrayLike) -> float:
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


def similarity_matlab_S11(mu1: ArrayLike, mu2: ArrayLike) -> float:
    symm_diff = fuzzy_symmetric_difference(mu1, mu2)
    return 1.0 - float(np.mean(symm_diff)) if symm_diff.size else 1.0

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
    return float(np.sqrt(np.sum(diff ** 2)))


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
    return float(np.dot(centred1, centred2) / (np.linalg.norm(centred1) * np.linalg.norm(centred2)))


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
    data_s1, data_s2, x_values_common = map(np.asarray, (data_s1, data_s2, x_values_common))
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
        mu_s2 = compute_ndg(x_values_common, data_s2, sigma_used)
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
        "MATLAB_S1": similarity_matlab_S1,
        "MATLAB_S3": similarity_matlab_S3,
        "MATLAB_S5": similarity_matlab_S5,
        "MATLAB_S4": similarity_matlab_S4,
        "MATLAB_S6": similarity_matlab_S6,
        "MATLAB_S8": similarity_matlab_S8,
        "MATLAB_S9": similarity_matlab_S9,
        "MATLAB_S10": similarity_matlab_S10,
        "MATLAB_S11": similarity_matlab_S11,
        # Distance‑based
        "Similarity_Hamming": similarity_hamming,
        "Similarity_Euclidean": similarity_euclidean,
        "Similarity_Chebyshev": similarity_chebyshev,
        "Distance_Hamming": distance_hamming,
        "Distance_Euclidean": distance_euclidean,
        "Distance_Chebyshev": distance_chebyshev,
        "MATLAB_S2_W": similarity_matlab_S2_W,
        "MATLAB_S": similarity_matlab_S,
        # Correlation‑based
        "Cosine": similarity_cosine,
        "Pearson": similarity_pearson,
        "MATLAB_P": similarity_matlab_P,
        # Others
        "MATLAB_T": similarity_matlab_T,
    }

    results: Dict[str, float] = {}
    for name, func in metric_funcs.items():
        try:
            results[name] = func(mu_s1, mu_s2)
        except Exception as exc:  # noqa: BLE001 – broad but logged
            print(f"Metric '{name}' failed: {exc}")
            results[name] = np.nan

    # Aliases
    results["MATLAB_M"] = results["Jaccard"]
    results["MATLAB_S7"] = results["OverlapCoefficient"]
    results["MATLAB_L"] = results["Similarity_Chebyshev"]

    # Metric1 (needs raw data)
    results["MATLAB_Metric1"] = (
        similarity_matlab_metric1(
            data_s1, data_s2, x_values, fs_method=fs_method, sigma_s2=sigma_s2
        )
        if data_s1 is not None and data_s2 is not None
        else np.nan
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
        # MATLAB
        "MATLAB_Metric1",
        "MATLAB_S1",
        "MATLAB_M",
        "MATLAB_T",
        "MATLAB_P",
        "MATLAB_S2_W",
        "MATLAB_S3",
        "MATLAB_L",
        "MATLAB_S",
        "MATLAB_S4",
        "MATLAB_S5",
        "MATLAB_S6",
        "MATLAB_S7",
        "MATLAB_S8",
        "MATLAB_S9",
        "MATLAB_S10",
        "MATLAB_S11",
    ]

    return {key: results[key] for key in preferred_order if key in results}

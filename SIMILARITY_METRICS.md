# Potential and Implemented Fuzzy Set Similarity Metrics

This document lists potential metrics for measuring the similarity between two fuzzy sets, A and B, represented by their membership functions `μ_A(x)` and `μ_B(x)` over a common discrete domain `X = {x_1, x_2, ..., x_n}`. It includes both general theoretical metrics and specific metrics implemented in the `nick_thesis/similarity_metrics.m` MATLAB script. *Note: Categorization of MATLAB metrics is approximate based on their calculation.*

## 1. Set-Theoretic / Overlap-Based Metrics

These metrics are based on the intersection, union, and cardinality of fuzzy sets.

*   **Jaccard Index (Tanimoto Coefficient):** Measures overlap relative to the union.
    $$ S_{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{\sum_{i=1}^n \min(\mu_A(x_i), \mu_B(x_i))}{\sum_{i=1}^n \max(\mu_A(x_i), \mu_B(x_i))}
    $$
    *   *Note:* The numerator, `sum(min(μ_A, μ_B))`, corresponds to the **Overlap Area** metric.
    *   *MATLAB `M` Metric (`similarity_measures`):* Calculates `sum(min) / sum(max)` after point matching, equivalent to the Jaccard Index.

*   **Dice Coefficient (Sørensen–Dice Index):** Similar to Jaccard, gives more weight to the intersection.
    $$ S_{Dice}(A, B) = \frac{2 |A \cap B|}{|A| + |B|} = \frac{2 \sum_{i=1}^n \min(\mu_A(x_i), \mu_B(x_i))}{\sum_{i=1}^n \mu_A(x_i) + \sum_{i=1}^n \mu_B(x_i)} 
    $$
    *   *Note:* `|A| = sum(μ_A(x_i))` is the scalar cardinality or sigma-count.
    *   *MATLAB `S3` Metric (`similarity_measures`):* Calculates `mean(2 * min / (μ_A + μ_B))` over matched points, a point-wise averaged Dice-like coefficient.

*   **Other Overlap/Ratio Metrics:**
    *   *MATLAB `S1` Metric (`similarity_measures`):* Calculates `mean(min / max)` over matched points.
    *   *MATLAB `S5` Metric (`set_metrics`):* Calculates `sum(intersection) / max(cardinality_A, cardinality_B)`.
    *   *MATLAB `S7` Metric (`set_metrics`):* Calculates `sum(intersection) / min(cardinality_A, cardinality_B)`.

*   **Metrics Based on Negation/Symmetric Difference:**
    *   *MATLAB `S4` Metric (`set_metrics`):* Based on the intersection of negated sets divided by the sum of their union. `internegat / sum(union(negat))`.
    *   *MATLAB `S6` Metric (`set_metrics`):* Similar to S5 but on negated sets. `internegat / max(cardinality_negA, cardinality_negB)`.
    *   *MATLAB `S8` Metric (`set_metrics`):* Similar to S7 but on negated sets. `internegat / min(cardinality_negA, cardinality_negB)`.
    *   *MATLAB `S9` Metric (`set_metrics`):* Based on the negated symmetric difference normalized by the maximum negated component. `sum(ng(symmdiff)) / max(sum(ng(symmdiff1)), sum(ng(symmdiff2)))`.
    *   *MATLAB `S10` Metric (`set_metrics`):* Based on the negated symmetric difference normalized by the minimum negated component. `sum(ng(symmdiff)) / min(sum(ng(symmdiff1)), sum(ng(symmdiff2)))`.
    *   *MATLAB `S11` Metric (`set_metrics`):* Calculates `1 - mean(symmetric_difference)`.

## 2. Distance-Based Metrics

These metrics first calculate a distance `d(A, B)` and then convert it to a similarity `S(A, B)`.

*   **Hamming Distance & Related:** Sum/Mean of absolute differences.
    $$ d_H(A, B) = \sum_{i=1}^n |\mu_A(x_i) - \mu_B(x_i)| $$
    *   Similarity: `S_H(A, B) = 1 - (d_H(A, B) / n)` (Normalized)
    *   *MATLAB `S2` Metric (`similarity_measures`):* Calculates `mean(1 - abs_diff)` over matched points, equivalent to `1 - mean(abs_diff)`.
    *   *MATLAB `W` Metric (`similarity_measures`):* Calculates `1 - mean(abs_diff)` over matched points (same as S2 in calculation shown).
    *   *MATLAB `S` Metric (`similarity_measures`):* Calculates `1 - sum(abs_diff) / (cardinality_A + cardinality_B)`, a normalized Hamming-like similarity.

*   **Euclidean Distance:** Standard Euclidean distance.
    $$ d_E(A, B) = \sqrt{\sum_{i=1}^n (\mu_A(x_i) - \mu_B(x_i))^2}
    $$
    *   *Note:* Directly calculated in `distribution.py` as `distance_euclidean`. Similarity conversion needed (e.g., `1 / (1 + d_E)`).

*   **Minkowski Distance (Lp norm):** Generalization.
    $$ d_p(A, B) = \left( \sum_{i=1}^n |\mu_A(x_i) - \mu_B(x_i)|^p \right)^{1/p} $$

*   **Chebyshev Distance (L∞ norm) & Related:** Maximum absolute difference.
    $$ d_C(A, B) = \max_{i} |\mu_A(x_i) - \mu_B(x_i)| $$
    *   Similarity: `S_C(A, B) = 1 - d_C(A, B)`
    *   *MATLAB `L` Metric (`similarity_measures`):* Calculates `1 - max(abs_diff)` over matched points and original `mV2`.

## 3. Correlation-Based Metrics

*   **Cosine Similarity:** Angle between membership vectors.
    $$ S_{Cosine}(A, B) = \frac{\sum_{i=1}^n \mu_A(x_i) \mu_B(x_i)}{\sqrt{\sum_{i=1}^n \mu_A(x_i)^2} \sqrt{\sum_{i=1}^n \mu_B(x_i)^2}} $$

*   **Pearson Correlation Coefficient:** Cosine similarity on centered vectors.
    $$ S_{Pearson}(A, B) = \frac{\sum_{i=1}^n (\mu_A(x_i) - \bar{\mu}_A) (\mu_B(x_i) - \bar{\mu}_B)}{\sqrt{\sum_{i=1}^n (\mu_A(x_i) - \bar{\mu}_A)^2} \sqrt{\sum_{i=1}^n (\mu_B(x_i) - \bar{\mu}_B)^2}} $$

*   **Other Correlation-like Metrics:**
    *   *MATLAB `P` Metric (`similarity_measures`):* Calculates `sum(pointwise_product) / min(norm_A^2, norm_B^2)`.

## 4. Other Metrics

*   **Maximum-Minimum Composition Based:** (More complex, theoretical).
*   **Geometric Mean Based:** (Theoretical).
*   *MATLAB `T` Metric (`similarity_measures`):* Calculates `max(min(μ_A, μ_B))` over matched points. The maximum value of the pointwise intersection.
*   *MATLAB `dinf` Metric (`similarity_measures`):* Calculates `max(abs(range2 - range1))`. Measures the maximum difference between the *domain supports* of the fuzzy sets, not the membership values directly.
*   *MATLAB `similarity_metric1` ("Theirs"):* Custom metric involving summing membership values of one set (`V`) at points corresponding to the other set (`S`), normalized by signal length and IQR difference (`delta`).
*   *MATLAB `similarity_metric2` ("Theirs + derivative"):* Custom metric involving signal derivatives (`dS`, `dV`), fuzzy sets for `V` and `dV`, and a weighted sum normalized by signal length and IQR difference (`delta`).

## Considerations for Selection

*   **Normalization:** Are the membership functions already normalized (e.g., height=1)? Does the metric require it?
*   **Domain:** Assumes a common discrete domain `X`.
*   **Interpretation:** What aspect of similarity does the metric capture best (overlap, distance, shape correlation)?
*   **Computational Cost:** Some metrics might be more expensive than others.
*   **Sensitivity:** How sensitive is the metric to small changes or noise?

## Current Implementation (`distribution.py`)

*   **Overlap:** Seems related to the numerator of the Jaccard Index (`sum(min(μ_A, μ_B))`). Needs verification if it's normalized to be a similarity score between 0 and 1 or just the raw sum.
*   **Euclidean Distance:** Directly implemented (`distance_euclidean`). A similarity score needs to be derived from it.

This list provides a starting point for selecting and implementing additional similarity measures for comparison. 
# Synthetic Data Cases for Similarity Metric Evaluation

This document describes the synthetic datasets generated in `new_source/run.py` to test and compare fuzzy similarity metrics under different conditions.

## Case Descriptions

**Case 1: Similar Normal Distributions, Shifted Mean**

*   **Sensor 1:** Normal distribution (loc=50, scale=5)
*   **Sensor 2:** Normal distribution (loc=55, scale=5)
*   **Scenario:** Represents two sensors measuring the same phenomenon with similar noise characteristics but a slight offset or calibration difference. Membership functions should have similar shapes but be shifted horizontally.

**Case 2: Normal vs. Uniform (Completely Different)**

*   **Sensor 1:** Normal distribution (loc=50, scale=5)
*   **Sensor 2:** Uniform distribution (low=0, high=100)
*   **Scenario:** Represents two sensors measuring very different phenomena or one sensor failing/producing random noise across a wide range. Membership functions should have fundamentally different shapes (bell vs. flat/broad).

**Case 3: Similar Normal Mean, Different Variance (Subset)**

*   **Sensor 1:** Normal distribution (loc=50, scale=5)
*   **Sensor 2:** Normal distribution (loc=50, scale=2)
*   **Scenario:** Represents two sensors measuring the same phenomenon centered at the same point, but one sensor has significantly less noise or higher precision. The membership function of Sensor 2 should be much narrower and largely contained within that of Sensor 1.

**Case 4: Two Shifted Bimodal Distributions**

*   **Sensor 1:** Bimodal Normal (loc1=30, scale1=5; loc2=70, scale2=5)
*   **Sensor 2:** Bimodal Normal (loc1=40, scale1=5; loc2=80, scale2=5)
*   **Scenario:** Represents sensors capturing a phenomenon with two distinct states or operating modes, but the state centers differ between the sensors. Membership functions will both have two peaks, but these peaks will be shifted relative to each other.

**Case 5: Bimodal vs. Normal (Centered)**

*   **Sensor 1:** Bimodal Normal (loc1=30, scale1=5; loc2=70, scale2=5)
*   **Sensor 2:** Normal distribution (loc=50, scale=5)
*   **Scenario:** Represents comparing a system with two states (Sensor 1) to a system with a single state centered between the two modes of the first (Sensor 2). Tests how metrics handle comparing single-peak vs. double-peak shapes.

**Case 6: Very Different Locations and Scales**

*   **Sensor 1:** Bimodal Normal (loc1=1050, scale1=105; loc2=550, scale2=500) - *Note: These values seem high/wide, potentially leading to broad distributions.*
*   **Sensor 2:** Normal distribution (loc=55, scale=5)
*   **Scenario:** Represents comparing two sensors measuring vastly different phenomena or operating at completely different scales/ranges with negligible overlap.

## Expected Qualitative Similarity Results

This table summarizes the *expected* behavior of different categories of similarity metrics across the cases. Actual results will depend on sigma choice and normalization. "Similarity" here refers to the output of similarity functions (higher value = more similar), not distance values.

| Case | Description                 | Overlap Metrics (Jaccard, Dice, OverlapCoeff, M, S1, S3, S5, S7) | Distance-Based Similarity (Hamming Sim, Euclidean Sim, Chebyshev Sim, S, S2/W, L) | Correlation Metrics (Cosine, Pearson, P) |
| :--- | :-------------------------- | :------------------------------------------------------------- | :---------------------------------------------------------------------------- | :--------------------------------------- |
| 1    | Similar Normal, Shifted     | Moderate                                                       | Moderate                                                                      | **High**                                 |
| 2    | Normal vs Uniform           | **Very Low**                                                   | **Very Low**                                                                  | **Very Low**                             |
| 3    | Normal Subset Variance      | **High** (esp. OverlapCoeff/S7)                                | Moderate-High                                                                 | **High**                                 |
| 4    | Bimodal Shifted             | Low-Moderate                                                   | Low-Moderate                                                                  | Moderate                                 |
| 5    | Bimodal vs Normal (Centered) | Moderate                                                       | Low-Moderate                                                                  | Low-Moderate                             |
| 6    | Very Different Loc/Scale    | **Very Low (Near Zero)**                                       | **Very Low (Near Zero)**                                                      | **Very Low (Near Zero)**                 |

**Notes:**

*   **Overlap Metrics:** Sensitive to the shared area/volume under the membership functions.
*   **Distance-Based Similarity:** Derived from point-wise differences between membership functions (lower distance = higher similarity). Sensitive to shifts and scale differences.
*   **Correlation Metrics:** Primarily sensitive to the overall shape alignment of the membership functions, less sensitive to magnitude or exact location if the shapes are proportional.
*   **MATLAB Metric 1 ("Theirs"):** Expected behavior is harder to predict without knowing the exact details of the `fs_construction` equivalent and the role of the `delta` normalization, but likely correlates somewhat with overlap and location differences.

This framework provides a baseline for interpreting the quantitative results generated by `run.py`. Deviations from these expectations may highlight interesting properties or sensitivities of specific metrics.
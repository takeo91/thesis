# Thesis: Development and comparison of fuzzy similarity correlation metrics for sensor data in health application and assisted living environments

This repository contains the implementation for the thesis mentioned above. The goal is to explore, develop, and evaluate methods for comparing the similarity between different sensor data streams using fuzzy set theory, particularly focusing on their distributions.

## Project Plan & Status

Here's a breakdown of the steps involved in this project:

**Phase 1: Data Acquisition and Preprocessing**

1.  ✅ **Identify and Obtain Dataset:** Select a suitable dataset containing multi-modal sensor data relevant to health or assisted living environments.
    *   *Status:* **Completed**. The Opportunity UCI HAR Dataset is being used. (`Data/OpportunityUCIDataset/`)
2.  ✅ **Understand Data Format:** Analyze the structure, file types, column descriptions, and label conventions of the chosen dataset.
    *   *Status:* **Completed**. (`Data/OpportunityUCIDataset/dataset/column_names.txt`, `Data/OpportunityUCIDataset/dataset/label_legend.txt`).
3.  ✅ **Develop Data Loading Script:** Create a script to read the raw data files.
    *   *Status:* **Completed**. (`new_source/main.py`).
4.  ✅ **Implement Data Cleaning:** Handle missing values, convert data types.
    *   *Status:* **Completed**. (`new_source/main.py`).
5.  ✅ **Implement Label Mapping:** Convert numerical labels to human-readable descriptions.
    *   *Status:* **Completed**. (`new_source/utils.py::extract_labels`, used in `new_source/main.py`).
6.  ✅ **Implement Metadata Parsing:** Extract sensor type, body part, measurement, etc., from column names.
    *   *Status:* **Completed**. (`new_source/utils.py::extract_metadata`).
7.  ✅ **Structure Data with MultiIndex:** Create a hierarchical index for the DataFrame columns for easier data access.
    *   *Status:* **Completed**. (`new_source/main.py`).

**Phase 2: Fuzzy Membership Function Development**

8.  ✅ **Research Membership Function Construction Methods:** Investigate techniques to represent the distribution of sensor data as a fuzzy membership function. Potential methods include:
    *   Histogram-based methods.
    *   Kernel Density Estimation (KDE) based methods.
    *   Neighborhood Density (ND) methods.
    *   Clustering-based methods.
    *   *Status:* **Completed (Initial Selection)**. ND and KDE methods are being explored (`new_source/distribution.py`, `new_source/run.py`).
9.  ✅ **Implement Membership Function Construction:** Code the selected methods to take sensor data (or its distribution) and generate a corresponding fuzzy membership function `μ_s(x)`.
    *   *Status:* **Completed (Initial Implementation)**. `compute_membership_functions` in `new_source/distribution.py` implements at least ND (and likely KDE implicitly via fitness calculation).
10. ✅ **Implement Data Normalization:** Add options to normalize sensor data before constructing membership functions, as fuzzy methods often work best on a standard range (e.g., [0, 1]).
    *   *Status:* **Completed**. (`new_source/distribution.py::normalize_data`, used in `new_source/run.py`).
11. ✅ **Implement Parameter Tuning (e.g., Sigma):** Allow adjustment of key parameters (like bandwidth `sigma` for ND/KDE) and explore their effect. Include methods for automatic parameter selection (e.g., relative sigma).
    *   *Status:* **Completed**. `sigma_option` parameter in `new_source/run.py` and `compute_membership_functions`.

**Phase 3: Fuzzy Similarity Metric Development**

12. ✅ **Research Fuzzy Set Similarity Measures:** Investigate metrics to quantify the similarity between two fuzzy sets (represented by their membership functions, `μ_s1(x)` and `μ_s2(x)`). Examples include:
    *   Overlap area / Intersection Index.
    *   Distance metrics (Euclidean, Hamming, etc.) adapted for fuzzy sets.
    *   Correlation-based measures.
    *   *Status:* ⚠️ **In Progress / Initial Selection**. Overlap and Euclidean distance are implemented. More research and implementation planned.
13. ✅ **Implement Similarity Measures:** Code the selected similarity metrics.
    *   *Status:* ⚠️ **In Progress / Initial Implementation**. `compute_similarity_measures` in `new_source/distribution.py` contains initial metrics. More to be added.

**Phase 4: Evaluation Framework**

14. ✅ **Define Evaluation Strategy:** Determine how to assess the "goodness" of the developed membership functions and similarity metrics.
    *   *Status:* **Completed**. Strategy involves:
        *   Using synthetic data with known properties (Cases 1-6 in `run.py`).
        *   Comparing generated membership functions to empirical distributions.
        *   Analyzing results on real-world data (presumably in `test.ipynb`).
15. ✅ **Implement Fitness Metrics:** Develop functions to objectively measure how well a membership function fits the empirical data distribution (e.g., MSE, KL Divergence).
    *   *Status:* **Completed**. `compute_fitness_metrics` in `new_source/distribution.py` (calculates MSE, KL, AIC, BIC).
16. ✅ **Develop Synthetic Test Cases:** Create artificial sensor data scenarios to test the metrics under controlled conditions (e.g., different overlaps, shapes, ranges).
    *   *Status:* **Completed**. `generate_case_data` in `new_source/run.py`.
17. ✅ **Create Experiment Runner:** Build a script/notebook to systematically run experiments across different methods, parameters, and test cases.
    *   *Status:* **Completed**. (`new_source/run.py`, `new_source/test.ipynb`).
18. ✅ **Implement Visualization:** Create plots to visualize:
    *   Raw data distributions (histograms).
    *   Membership functions.
    *   Comparison between membership functions and empirical distributions.
    *   Residuals.
    *   Similarity/Fitness metrics vs. parameters.
    *   *Status:* **Completed**. Plotting functions in `run.py` and visualization code in `test.ipynb`.
19. ✅ **Automate Plot Saving:** Save generated plots systematically for documentation.
    *   *Status:* **Completed**. Implemented in `run.py` to save plots per case.
20. ⏳ **Evaluate Membership Function Fitness (Synthetic Data):** Analyze the fitness metrics (MSE, KL, AIC, BIC) and residual plots generated from the synthetic test cases (Step 16) to understand how well the chosen method(s) (Step 9) with different parameters (Step 11, e.g., `sigma`) reconstruct the underlying empirical distributions. Identify potentially optimal parameters based on AIC/BIC for individual signal representation *before* comparing pairs.
    *   *Status:* **Pending**. Framework and metrics exist, requires focused analysis of the `results_df` and saved plots.

**Phase 5: Analysis and Comparison**

21. ⏳ **Analyze Similarity Metrics (Synthetic Data):** Using the potentially optimized parameters from Step 20 (or exploring a range), evaluate how the implemented similarity metrics (Step 13) perform on the synthetic test cases (Step 16). Does their behavior match intuition for the different cases (e.g., higher similarity for Case 1 vs Case 2)?
    *   *Status:* **Pending**. The framework is built (`test.ipynb`), but detailed analysis and interpretation of the results are needed.
22. ⏳ **Apply Methods to Real Data:** Run the membership function construction and similarity calculations on selected sensor pairs from the Opportunity dataset.
    *   *Status:* **Pending (Likely)**. While `test.ipynb` currently focuses on synthetic data results, the original purpose likely involved real data. This step needs confirmation or implementation. Need to select relevant sensor pairs (e.g., same sensor type on different limbs, different sensor types on the same limb).
23. ⏳ **Compare Similarity Metrics (Real Data & vs. Traditional):** Analyze how the different fuzzy similarity metrics behave on real data. Do they capture intuitive notions of similarity? How do they compare to traditional correlation coefficients (e.g., Pearson) calculated on the raw sensor data time series?
    *   *Status:* **Pending**. Requires running comparisons and interpreting the results.
24. ⏳ **Evaluate Performance in Health/AAL Context:** Discuss how the developed metrics could be applied in practical scenarios. For example:
    *   Detecting sensor drift or malfunction (comparing a sensor to its past self).
    *   Quantifying synchrony between limbs during activities.
    *   Comparing sensor readings during different activities or between different subjects.
    *   *Status:* **Pending**. This involves interpreting the technical results in the context of the application domain.

**Phase 6: Documentation and Thesis Writing**

25. ⏳ **Document Code:** Ensure functions and scripts are well-commented.
    *   *Status:* **Partially Done**. Code has some comments and structure, but review/enhancement might be needed.
26. ✅ **Update README:** Maintain this README file with project status and instructions.
    *   *Status:* **Completed (as of this update)**.
27. ⏳ **Write Thesis:** Structure and write the thesis document, incorporating the methodology, results, analysis, and discussion.
    *   *Status:* **Pending**.

## Key Findings and Decisions

*   **(04/06/2025): Optimal Sigma for Normalized Data & Handling Scale:**
    *   Analysis of fitness metrics (AIC, BIC, MSE) on synthetic data (Step 20) suggests that when using **min-max normalized data**, fixed `sigma` values for the Neighbor Density (`nd`) membership function method around **0.01** or **0.1** often provide a good balance between fitting the empirical distribution and model simplicity.
    *   Since normalization removes original scale information (e.g., standard deviation, range), which is important for comparing sensors in a real system, the decision was made to **calculate and store scale features (standard deviation, range) from the *original* sensor data** alongside the fuzzy metrics.
    *   The proposed approach for comparing/searching sensors in the final application is to use the **fuzzy similarity score (calculated on normalized data with sigma ~0.01-0.1) in conjunction with these explicit, stored scale features**. Relative sigma options (e.g., `r0.01`) were not selected during the initial optimization on normalized data but remain an option if analysis shifts to non-normalized data.

## Datasets

This project utilizes two primary datasets for analysis and evaluation:

### OPPORTUNITY Activity Recognition Dataset

The OPPORTUNITY Dataset contains sensor readings from subjects performing daily activities while wearing multiple sensors. It's designed to benchmark human activity recognition algorithms.

- **Source**: [UCI Machine Learning Repository: OPPORTUNITY Activity Recognition](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition)
- **Features**: 
  - Body-worn sensors (7 IMUs, 12 3D acceleration sensors)
  - Object sensors (12 objects with 3D acceleration and 2D rate of turn)
  - Ambient sensors (13 switches and 8 3D acceleration sensors)
- **Recordings**: 4 users, 6 runs per user (5 Activity of Daily Living runs and 1 drill run)
- **Annotations**: Activities labeled at multiple levels (locomotion modes, low-level actions, mid-level gestures, high-level activities)

### PAMAP2 Physical Activity Monitoring Dataset

The PAMAP2 Dataset contains data of various physical activities performed by subjects wearing inertial measurement units and a heart rate monitor.

- **Source**: [UCI Machine Learning Repository: PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- **Features**:
  - 3 Colibri wireless inertial measurement units (100Hz)
  - Heart rate monitor (~9Hz)
- **Activities**: 18 different physical activities (walking, cycling, playing soccer, etc.)
- **Subjects**: 9 subjects
- **Protocol**: Each subject followed a protocol containing 12 different activities, with some subjects performing additional optional activities

Both datasets provide rich multimodal sensor data suitable for evaluating fuzzy similarity metrics in the context of health applications and assisted living environments.

## How to Run

1.  Ensure Python environment with necessary libraries (pandas, numpy, matplotlib, seaborn, scikit-learn, joblib) is set up.
2.  Download the Opportunity UCI Dataset and place it in the `Data/` directory following the structure used in `new_source/main.py`.
3.  Run the experiments using the `new_source/test.ipynb` notebook. This will execute the synthetic test cases defined in `new_source/run.py` and generate:
    *   A results DataFrame (printed or saved from the notebook).
    *   Plots comparing metrics vs. sigma (displayed or saved from the notebook).
    *   Plots of membership functions vs. empirical distributions saved to the `plots/` directory.

*(Further instructions may be added as analysis on real data progresses)*

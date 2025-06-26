# RQ1 Experiment Methodology

## Objective
To compare the computational efficiency and accuracy of the optimized Neighbor Density Graph (NDG) method versus Kernel Density Estimation (KDE) for fuzzy membership function computation on sensor data relevant to health and assisted living applications.

## Datasets
- **Synthetic Normal**: Simulated Gaussian-distributed signals
- **Synthetic Bimodal**: Simulated bimodal-distributed signals
- **Opportunity**: Real-world wearable sensor dataset
- **PAMAP2**: Real-world wearable sensor dataset

## Signal Processing & Windowing
- Signals are segmented into windows of lengths: 100, 1,000, 10,000, and 100,000 samples
- Overlap and windowing parameters are consistent across all methods
- Each window is processed independently for membership function estimation

## Membership Function Methods
- **NDG (Neighbor Density Graph)**: Custom method with spatial pruning, JIT compilation, and parallelization (Epanechnikov kernel)
- **KDE (Kernel Density Estimation)**: Standard kernel density estimation with matching kernel and bandwidth
- **Sigma (Bandwidth) Values**: [0.1, 0.3, 0.5, 'r0.1', 'r0.3']

## Experimental Design
- For each dataset and signal length:
    - Compute membership functions using both NDG and KDE
    - Measure execution time, memory usage, and approximation error (KL-divergence, Chi-squared)
    - Repeat with 5-fold cross-validation for statistical robustness
- Total experiments: 400 (all combinations)

## Metrics & Statistical Analysis
- **Execution Time**: Wall-clock time for membership function computation
- **Memory Usage**: Peak memory during computation
- **Approximation Error**: KL-divergence and Chi-squared distance between NDG and KDE outputs
- **Statistical Test**: Wilcoxon signed-rank test for paired comparisons
- **Effect Size**: Reported for all significant results

## Reproducibility
- All code and experiments are version-controlled (git)
- Random seeds are fixed for all runs
- Environment and dependencies are specified in `pyproject.toml`
- All results, scripts, and figures are available in the `results/ndg_vs_kde/` directory

## Notes
- The optimized NDG implementation uses Epanechnikov kernel for maximum speedup
- All experiments are run on the same hardware for fair comparison
- Plots and summary tables are generated automatically after each experiment 
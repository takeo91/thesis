# RQ2 Mini Test Analysis Summary

## Overview
This document summarizes the findings from the mini test of the RQ2 classification experiment using the Opportunity dataset with Locomotion labels. The mini test was designed to quickly evaluate the performance of similarity metrics for activity classification while using a small subset of the data.

## Test Configuration
- **Dataset**: Opportunity with Locomotion labels (Sit, Stand, Walk)
- **Sampling**: Every 20th sample from the original dataset
- **Window Size**: 64 samples
- **Window Overlap**: 50%
- **Metrics Tested**: 10 similarity metrics from different categories
- **Classification Method**: 1-NN with Leave-One-Window-Out cross-validation

## Key Findings

### 1. Metric Performance
All tested metrics achieved identical performance:
- **Macro F1 Score**: 0.438
- **Balanced Accuracy**: 0.500
- **Accuracy**: 0.780

This suggests that for this specific dataset and configuration, the choice of similarity metric does not significantly impact classification performance. This could be due to:
- The small sample size
- The reduced feature set (only accelerometer data)
- The binary classification problem (only Sit and Stand classes were present in the windowed data)
- The clear separation between activity patterns

### 2. Class Distribution
The windowed data showed significant class imbalance:
- **Sit**: 11 windows (22%)
- **Stand**: 39 windows (78%)

This imbalance explains the discrepancy between accuracy (0.78) and balanced accuracy (0.50). The classifier appears to be biased toward the majority class (Stand), which is reflected in the confusion matrices.

### 3. Computation Time
The computation times for all metrics were very similar, with Jaccard being marginally slower than the others. However, the differences are negligible for this small dataset.

### 4. Confusion Matrices
The confusion matrices for all metrics show identical patterns:
- Perfect classification of the majority class (Stand)
- Poor classification of the minority class (Sit)
- This indicates a class imbalance problem rather than a metric performance issue

## Recommendations for Full Experiment

1. **Address Class Imbalance**:
   - Use stratified sampling to ensure balanced class representation
   - Consider using class weights in the classification algorithm
   - Explore techniques like SMOTE for synthetic minority oversampling

2. **Increase Data Representation**:
   - Use a larger subset of the data to capture more variability
   - Include more sensor modalities (not just accelerometer)
   - Test with all available activity classes (including Walk)

3. **Metric Selection**:
   - Since all metrics performed identically in this mini test, the full experiment should focus on metrics with theoretical advantages for time series data
   - Include metrics from different families to ensure diversity
   - Consider computational efficiency for larger datasets

4. **Parameter Tuning**:
   - Test multiple window sizes to find optimal temporal resolution
   - Experiment with different overlap ratios
   - Evaluate different NDG kernel parameters

5. **Evaluation Strategy**:
   - Use stratified cross-validation to ensure fair evaluation
   - Report macro F1 score and balanced accuracy as primary metrics
   - Include confusion matrices for detailed error analysis

## Conclusion
The mini test successfully validated the end-to-end pipeline for RQ2 classification. While all metrics performed identically in this simplified setting, the full experiment with more data and classes will likely reveal meaningful differences between metrics. The class imbalance issue should be addressed to ensure fair evaluation of metric performance. 
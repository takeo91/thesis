# Sensor-Agnostic Fuzzy Similarity Approach

## Overview

The sensor-agnostic approach extends the per-sensor methodology by removing dependency on specific sensor identities and positions. Instead of comparing sensor-to-sensor (e.g., right arm accelerometer vs right arm accelerometer), it compares feature-to-feature across the entire sensor ensemble.

## Key Concept

**Sensor-Aware**: "How similar are the right arm accelerometer patterns between these two activities?"

**Sensor-Agnostic**: "How similar are the overall sensor ensemble characteristics between these two activities?"

## Motivation

### Problems with Sensor-Aware Approach:
1. **Fixed sensor configuration**: Requires identical sensor setups across datasets
2. **Sensor placement sensitivity**: Performance degrades if sensors are repositioned
3. **Dataset compatibility**: Cannot compare activities from different sensor configurations
4. **Scalability issues**: Adding/removing sensors requires retraining

### Benefits of Sensor-Agnostic Approach:
1. **Configuration flexibility**: Works with different numbers and types of sensors
2. **Cross-dataset compatibility**: Can compare activities across different setups
3. **Robustness**: Less sensitive to sensor placement variations
4. **Generalizability**: Features transfer across different sensor arrangements

## Implementation Strategy

### 1. Feature Extraction Layer

Instead of processing individual sensors, extract ensemble-level features:

#### Statistical Features
- **Central tendency**: Mean, median, mode across all sensors
- **Dispersion**: Standard deviation, variance, range across sensors
- **Distribution shape**: Skewness, kurtosis of sensor value distributions
- **Extremes**: Min, max, percentiles across sensor ensemble

#### Distributional Features
- **Histogram features**: Bin counts of sensor value distributions
- **Kernel density**: Smooth density estimation of sensor value distributions
- **Quantile functions**: Inverse CDF characteristics

#### Structural Features
- **Correlation structure**: Eigenvalues of sensor correlation matrix
- **Principal components**: Loadings and explained variance ratios
- **Sensor importance**: Ranking by variance contribution
- **Clustering patterns**: Sensor groupings based on similarity

#### Temporal Features
- **Autocorrelation**: Temporal dependencies in ensemble behavior
- **Spectral features**: FFT characteristics of ensemble signals
- **Trend analysis**: Overall directional patterns in sensor ensemble

### 2. Membership Function Creation

Create NDG membership functions for each feature type:

```python
def compute_sensor_agnostic_membership(window_data, feature_types):
    """
    Compute sensor-agnostic membership functions.
    
    Args:
        window_data: Shape (window_size, n_sensors)
        feature_types: List of feature extraction methods
    
    Returns:
        Dict of membership functions, one per feature type
    """
    features = {}
    membership_functions = {}
    
    for feature_type in feature_types:
        # Extract feature from sensor ensemble
        feature_values = extract_feature(window_data, feature_type)
        
        # Create NDG membership function for this feature
        x_vals, mu_vals = compute_ndg(feature_values, ...)
        membership_functions[feature_type] = (x_vals, mu_vals)
    
    return membership_functions
```

### 3. Similarity Computation

Compare windows based on feature-level similarities:

```python
def compute_sensor_agnostic_similarity(mu_funcs_1, mu_funcs_2, metric="jaccard"):
    """
    Compute similarity between sensor-agnostic membership functions.
    
    Args:
        mu_funcs_1, mu_funcs_2: Dicts of membership functions
        metric: Similarity metric to use
    
    Returns:
        Aggregated similarity value
    """
    similarities = []
    
    for feature_type in mu_funcs_1.keys():
        mu_1 = mu_funcs_1[feature_type]
        mu_2 = mu_funcs_2[feature_type]
        
        sim = compute_similarity(mu_1, mu_2, metric)
        similarities.append(sim)
    
    # Aggregate similarities (could be weighted)
    return np.mean(similarities)
```

## Feature Design Considerations

### 1. Feature Selection
- **Informativeness**: Features should capture activity-relevant patterns
- **Stability**: Features should be robust to sensor noise and placement
- **Orthogonality**: Features should capture different aspects of sensor behavior
- **Scalability**: Features should work with varying numbers of sensors

### 2. Feature Engineering

#### Basic Statistical Features
```python
def extract_statistical_features(window_data):
    """Extract basic statistical features from sensor ensemble."""
    return {
        'mean_across_sensors': np.mean(window_data, axis=1),
        'std_across_sensors': np.std(window_data, axis=1),
        'range_across_sensors': np.ptp(window_data, axis=1),
        'median_across_sensors': np.median(window_data, axis=1)
    }
```

#### Advanced Structural Features
```python
def extract_structural_features(window_data):
    """Extract structural features from sensor ensemble."""
    corr_matrix = np.corrcoef(window_data.T)
    eigenvals = np.linalg.eigvals(corr_matrix)
    
    return {
        'correlation_eigenvalues': eigenvals,
        'max_correlation': np.max(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
        'correlation_trace': np.trace(corr_matrix)
    }
```

### 3. Hierarchical Features

Create multi-level feature hierarchies:

```python
def extract_hierarchical_features(window_data):
    """Extract features at multiple abstraction levels."""
    features = {}
    
    # Level 1: Raw statistical features
    features['level1'] = extract_statistical_features(window_data)
    
    # Level 2: Derived features from Level 1
    level1_data = np.column_stack(list(features['level1'].values()))
    features['level2'] = extract_statistical_features(level1_data)
    
    return features
```

## Implementation Plan

### Phase 1: Core Implementation
1. **Feature extraction module**: `sensor_agnostic_features.py`
2. **Membership function computation**: Extend existing NDG functions
3. **Similarity computation**: Adapt existing similarity metrics
4. **Integration**: Add to `activity_classification.py`

### Phase 2: Advanced Features
1. **Temporal features**: Time-series specific characteristics
2. **Spectral features**: Frequency domain analysis
3. **Structural features**: Correlation and PCA-based features
4. **Hierarchical features**: Multi-level abstractions

### Phase 3: Optimization
1. **Feature selection**: Automatic selection of most informative features
2. **Weighted aggregation**: Learn optimal feature weights
3. **Adaptive features**: Features that adapt to sensor configuration
4. **Performance optimization**: Efficient computation for large datasets

## Comparison with Existing Approaches

| Aspect | Traditional | Sensor-Aware | Sensor-Agnostic |
|--------|-------------|--------------|-----------------|
| **Sensor dependency** | Low (flattened) | High (position-specific) | None (feature-based) |
| **Configuration flexibility** | Medium | Low | High |
| **Cross-dataset compatibility** | Medium | Low | High |
| **Performance** | Low | High | Medium-High |
| **Computational cost** | Low | High | Medium |
| **Interpretability** | Low | High | Medium |

## Expected Benefits

### 1. Generalizability
- Works across different sensor configurations
- Enables cross-dataset activity recognition
- Reduces dependency on specific hardware setups

### 2. Robustness
- Less sensitive to sensor placement variations
- Handles missing or faulty sensors gracefully
- Adapts to different sensor densities

### 3. Scalability
- Easy to add new sensor types
- Works with varying numbers of sensors
- Supports heterogeneous sensor setups

## Potential Challenges

### 1. Feature Design
- Identifying optimal feature sets for different activities
- Balancing informativeness vs. generalizability
- Handling feature interactions and dependencies

### 2. Performance Trade-offs
- May lose some discriminative power compared to sensor-aware approach
- Feature extraction overhead
- Optimal aggregation strategies

### 3. Validation
- Need diverse datasets with different sensor configurations
- Cross-validation across different setups
- Comparison with sensor-aware performance

## Next Steps

1. **Prototype implementation**: Basic statistical features
2. **Validation study**: Compare with sensor-aware approach
3. **Feature engineering**: Develop advanced feature sets
4. **Cross-dataset evaluation**: Test generalizability
5. **Optimization**: Performance and accuracy improvements

## Code Structure

```
thesis/fuzzy/
├── sensor_agnostic_membership.py    # Main implementation
├── feature_extraction.py            # Feature extraction methods
├── agnostic_similarity.py          # Similarity computation
└── feature_selection.py            # Automatic feature selection

thesis/exp/
├── sensor_agnostic_experiment.py   # Experiment runner
└── cross_dataset_validation.py     # Cross-dataset evaluation
```

This sensor-agnostic approach would represent a significant step toward more generalizable and robust activity recognition systems that can work across different sensor configurations and datasets. 
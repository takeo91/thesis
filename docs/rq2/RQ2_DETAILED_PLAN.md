# RQ2 Detailed Implementation Plan: Activity Classification for Discriminative Power Assessment

**Objective**: Evaluate the discriminative power of fuzzy similarity metrics for sensor-based activity classification using optimized NDG implementations.

**Hypothesis H2**: Certain similarity metrics demonstrate superior discriminative power for distinguishing between different physical activities in sensor data.

## 📋 CHANGELOG

### 2025-01-20: Phase 1 & 2 Complete ✅
- **Phase 1**: Implemented all missing similarity metrics
  - Added Jensen-Shannon divergence metric
  - Added β-similarity metric with variants (0.1, 0.5, 1.0, 2.0)
  - Total metrics now at 38 for comprehensive evaluation
  - All metrics validated and tested

- **Phase 2**: Completed activity classification pipeline
  - Implemented time series windowing with configurable parameters
  - Integrated optimized NDG membership function computation
  - Added 1-NN classification with Leave-One-Window-Out CV
  - Implemented comprehensive performance metrics

### Current Status
- ✅ Phase 1: Complete
- ✅ Phase 2: Complete
- 🚀 Phase 3: Ready to start

## 1. PHASE 1: Complete Missing Similarity Metrics (Days 1-2) ✅

### 1.1 Jensen-Shannon Divergence Implementation ✅
**File**: `thesis/fuzzy/similarity.py`

**Technical Specifications**:
```python
def similarity_jensen_shannon(mu1: ArrayLike, mu2: ArrayLike) -> float:
    """
    Jensen-Shannon divergence-based similarity for fuzzy sets.
    
    JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    
    Similarity = 1 - sqrt(JS(P||Q))
    """
    # Normalize to probability distributions
    # Handle edge cases (empty/zero distributions)
    # Compute JS divergence
    # Transform to similarity [0,1]
```

**Validation**: Test on known probability distributions with analytical solutions.

### 1.2 β-Similarity Metric Implementation ✅
**File**: `thesis/fuzzy/similarity.py`

**Technical Specifications**:
```python
def similarity_beta(mu1: ArrayLike, mu2: ArrayLike, beta: float = 1.0) -> float:
    """
    β-similarity metric for fuzzy sets.
    
    β-similarity generalizes several similarity measures:
    - β = 1: Jaccard index
    - β = 0.5: Dice coefficient
    - β → 0: Overlap coefficient
    
    S_β(A,B) = |A ∩ B| / (|A ∩ B| + β|A \ B| + (1-β)|B \ A|)
    """
    # Compute intersection and set differences
    # Apply β-weighted formula
    # Handle edge cases (β = 0, empty sets)
```

**Default β values**: [0.1, 0.5, 1.0, 2.0] to capture different similarity behaviors.

### 1.3 Update Metric Registry ✅
**File**: `thesis/fuzzy/similarity.py`

Added to `calculate_all_similarity_metrics()`:
- `"JensenShannon": similarity_jensen_shannon`
- `"Beta_0.1": lambda mu1, mu2: similarity_beta(mu1, mu2, 0.1)`
- `"Beta_0.5": lambda mu1, mu2: similarity_beta(mu1, mu2, 0.5)`
- `"Beta_1.0": lambda mu1, mu2: similarity_beta(mu1, mu2, 1.0)`
- `"Beta_2.0": lambda mu1, mu2: similarity_beta(mu1, mu2, 2.0)`

**Total Metrics**: 38 similarity metrics for comprehensive evaluation.

## 2. PHASE 2: Activity Classification Pipeline (Days 3-5) ✅

### 2.1 Time Series Windowing Module ✅
**File**: `thesis/data/windowing.py`

**Technical Specifications**:
```python
class TimeSeriesWindowing:
    def __init__(self, window_size: int, overlap: float = 0.5, 
                 min_activity_samples: int = 10):
        """
        Parameters:
        - window_size: Number of samples per window (e.g., 128, 256)
        - overlap: Overlap fraction between windows (0.0 to 0.9)
        - min_activity_samples: Minimum samples per activity class
        """
    
    def create_windows(self, data: pd.Series, labels: pd.Series) -> Tuple[List[np.ndarray], List[str]]:
        """
        Create sliding windows with activity labels.
        
        Returns:
        - windows: List of data windows
        - window_labels: Corresponding activity labels
        """
        # Sliding window extraction
        # Activity label assignment (majority vote)
        # Filter windows with insufficient data
        # Balance classes if needed
```

**Window Configurations**:
- **Window sizes**: [128, 256, 512] samples
- **Overlap**: [0.5, 0.7] (50%, 70% overlap)
- **Sampling rate**: Original dataset frequency (50Hz Opportunity, 100Hz PAMAP2)

### 2.2 Membership Function Computation ✅
**File**: `thesis/exp/membership_computation.py`

**Technical Specifications**:
```python
def compute_window_membership_functions(windows: List[np.ndarray], 
                                       kernel_type: str = "epanechnikov",
                                       sigma_method: str = "adaptive") -> List[np.ndarray]:
    """
    Compute membership functions for each window using optimized NDG.
    
    Parameters:
    - windows: Time series windows
    - kernel_type: "epanechnikov" (fastest), "gaussian"
    - sigma_method: "adaptive" (0.1 * range), "fixed", "scott", "silverman"
    
    Returns:
    - membership_functions: List of fuzzy membership functions
    """
    # Use optimized compute_ndg() with Epanechnikov kernel
    # Adaptive sigma based on window data range
    # Consistent domain discretization (100-200 points)
    # Parallel processing for large datasets
```

**Optimization Features**:
- **Kernel**: Epanechnikov (10-100x faster than Gaussian)
- **Sigma**: Adaptive `0.1 * np.ptp(window)` for consistent resolution
- **Domain**: Consistent discretization across all windows
- **Parallel**: Process multiple windows simultaneously

### 2.3 Similarity Matrix Computation ✅
**File**: `thesis/exp/similarity_matrix.py`

**Technical Specifications**:
```python
class SimilarityMatrixComputer:
    def __init__(self, metrics: List[str] = None):
        """
        Initialize with specific similarity metrics.
        Default: Use all 38 implemented metrics
        """
    
    def compute_pairwise_similarities(self, membership_functions: List[np.ndarray],
                                    x_values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute pairwise similarity matrices for all metrics.
        
        Returns:
        - similarity_matrices: Dict[metric_name, similarity_matrix]
        """
        # Compute similarity for each metric
        # Parallel computation across metrics
        # Memory-efficient for large datasets
        # Progress tracking
```

**Performance Optimizations**:
- **Parallel computation**: Compute metrics simultaneously
- **Memory management**: Process in chunks for large datasets
- **Caching**: Store intermediate results
- **Progress tracking**: Monitor computation progress

### 2.4 1-NN Classifier Implementation ✅
**File**: `thesis/exp/nearest_neighbor_classifier.py`

**Technical Specifications**:
```python
class FuzzyNearestNeighborClassifier:
    def __init__(self, similarity_metric: str = "Cosine"):
        """
        1-NN classifier using fuzzy similarity metrics.
        """
    
    def fit(self, similarity_matrix: np.ndarray, labels: np.ndarray):
        """
        Store training data and labels.
        """
    
    def predict(self, query_indices: np.ndarray) -> np.ndarray:
        """
        Predict using 1-NN with similarity matrix.
        
        For each query:
        1. Find most similar training sample (highest similarity)
        2. Return corresponding label
        """
    
    def cross_validate(self, similarity_matrix: np.ndarray, labels: np.ndarray,
                      cv_method: str = "leave_one_out") -> Dict[str, float]:
        """
        Perform cross-validation and return performance metrics.
        
        Returns:
        - macro_f1: Macro-averaged F1 score
        - balanced_accuracy: Balanced accuracy
        - per_class_f1: F1 score per activity class
        - confusion_matrix: Classification confusion matrix
        """
```

**Cross-Validation Strategy**:
- **Method**: Leave-One-Window-Out (LOWO)
- **Stratification**: Maintain class distribution
- **Metrics**: Macro-F1, Balanced Accuracy, Per-class F1
- **Reproducibility**: Fixed random seeds

## 3. PHASE 3: Experimental Framework (Days 5-7) 🚀

### 3.1 Main Experiment Controller
**File**: `thesis/exp/rq2_experiment.py`

**Technical Specifications**:
```python
class RQ2Experiment:
    def __init__(self, datasets: List[str] = ["opportunity", "pamap2"],
                 window_sizes: List[int] = [128, 256],
                 overlaps: List[float] = [0.5, 0.7],
                 activities: List[str] = None):
        """
        RQ2 experiment configuration.
        
        Parameters:
        - datasets: ["opportunity", "pamap2"]
        - window_sizes: [128, 256, 512] samples
        - overlaps: [0.5, 0.7] overlap fractions
        - activities: Specific activities to include (None = all)
        """
    
    def run_experiment(self, quick_test: bool = False) -> pd.DataFrame:
        """
        Run complete RQ2 experiment.
        
        Steps:
        1. Load datasets and create windows
        2. Compute membership functions (optimized NDG)
        3. Calculate similarity matrices (all 38 metrics)
        4. Perform 1-NN classification with LOWO CV
        5. Collect performance metrics
        6. Statistical analysis
        
        Returns:
        - results_df: Complete experimental results
        """
```

**Experiment Matrix**:
- **Datasets**: Opportunity, PAMAP2
- **Window sizes**: 128, 256 samples
- **Overlaps**: 50%, 70%
- **Metrics**: 38 similarity metrics
- **Activities**: Top 5-7 activities per dataset (sufficient samples)
- **Total configurations**: ~2 × 2 × 2 × 38 = 304 experiments

### 3.2 Activity Selection Strategy
**Target Activities**:

**Opportunity Dataset**:
- "Open Door 1", "Open Door 2", "Close Door 1", "Close Door 2"
- "Open Fridge", "Close Fridge", "Toggle Switch"
- Filter: Minimum 100 windows per activity

**PAMAP2 Dataset**:
- "walking", "running", "cycling", "sitting", "standing"
- "ascending_stairs", "descending_stairs"
- Filter: Minimum 100 windows per activity

### 3.3 Statistical Analysis Module
**File**: `thesis/exp/rq2_statistical_analysis.py`

**Technical Specifications**:
```python
def perform_rq2_statistical_analysis(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform comprehensive statistical analysis for RQ2.
    
    Tests:
    1. Friedman test: Overall metric comparison
    2. Nemenyi post-hoc: Pairwise metric comparisons
    3. Effect size calculations (Cohen's d, Cliff's delta)
    4. Confidence intervals for performance metrics
    5. Activity-specific analysis
    
    Returns:
    - statistical_results: Statistical test results and rankings
    """
    
def rank_similarity_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank similarity metrics by discriminative power.
    
    Ranking criteria:
    1. Average macro-F1 across all experiments
    2. Consistency across datasets and configurations
    3. Statistical significance of performance differences
    
    Returns:
    - metric_rankings: Ranked list of similarity metrics
    """
```

## 4. PHASE 4: Data Processing and Evaluation (Days 6-7)

### 4.1 Dataset Processing Specifications

**Opportunity Dataset Processing**:
```python
# Sensor selection
sensor_types = ["IMU", "Accelerometer"]
body_parts = ["RightLowerArm", "LeftLowerArm", "Back"]
axes = ["X", "Y", "Z"]

# Activity filtering
target_activities = [
    "Open Door 1", "Open Door 2", "Close Door 1", "Close Door 2",
    "Open Fridge", "Close Fridge", "Toggle Switch"
]

# Preprocessing
sampling_rate = 30  # Hz (downsampled from 50Hz)
filter_cutoff = 15  # Hz low-pass filter
```

**PAMAP2 Dataset Processing**:
```python
# Sensor selection
sensor_locations = ["hand", "chest", "ankle"]
sensor_types = ["accelerometer", "gyroscope", "magnetometer"]
axes = ["x", "y", "z"]

# Activity filtering
target_activities = [
    "walking", "running", "cycling", "sitting", "standing",
    "ascending_stairs", "descending_stairs"
]

# Preprocessing
sampling_rate = 100  # Hz (original)
filter_cutoff = 20   # Hz low-pass filter
```

### 4.2 Performance Metrics

**Primary Metrics**:
- **Macro-F1 Score**: Average F1 across all activity classes
- **Balanced Accuracy**: Account for class imbalance
- **Per-class F1**: Individual activity classification performance

**Secondary Metrics**:
- **Precision/Recall**: Per-class and macro-averaged
- **Cohen's Kappa**: Inter-rater agreement measure
- **Classification Report**: Comprehensive performance summary

## 5. PHASE 5: Results Analysis and Visualization (Days 7-8)

### 5.1 Visualization Suite
**File**: `thesis/exp/rq2_visualizations.py`

**Plots to Generate**:
1. **Metric Performance Heatmap**: Macro-F1 scores across datasets/configurations
2. **Statistical Significance Matrix**: Pairwise metric comparisons
3. **Activity-Specific Performance**: Per-activity classification accuracy
4. **Ranking Comparison**: Top-performing metrics across datasets
5. **Convergence Analysis**: Performance vs window size/overlap

### 5.2 Expected Outputs

**Results Directory**: `results/rq2_classification/`

**Generated Files**:
- `rq2_experimental_results.csv`: Raw experimental data
- `rq2_statistical_analysis.csv`: Statistical test results
- `rq2_metric_rankings.csv`: Ranked similarity metrics
- `rq2_summary_report.md`: Comprehensive analysis report
- `rq2_performance_heatmap.png`: Visualization of metric performance
- `rq2_statistical_significance.png`: Statistical comparison plots

## 6. Implementation Timeline

### **Day 1-2: Similarity Metrics** ✅
- [x] Analyze existing 24+ metrics in `thesis/fuzzy/similarity.py`
- [x] Implement Jensen-Shannon divergence metric
- [x] Implement β-similarity metric (4 variants)
- [x] Update metric registry and validation tests
- [x] Test on synthetic data for correctness

### **Day 3: Windowing and Preprocessing** ✅
- [x] Implement `TimeSeriesWindowing` class
- [x] Test windowing on Opportunity dataset
- [x] Test windowing on PAMAP2 dataset
- [x] Validate activity label assignment
- [x] Memory usage optimization

### **Day 4: Membership Function Computation** ✅
- [x] Implement optimized membership function computation
- [x] Test Epanechnikov kernel performance
- [x] Validate adaptive sigma calculation
- [x] Parallel processing implementation
- [x] Memory management for large datasets

### **Day 5: Similarity Matrix and Classification** ✅
- [x] Implement similarity matrix computation
- [x] Implement 1-NN classifier
- [x] Test leave-one-window-out cross-validation
- [x] Validate performance metric calculations
- [x] End-to-end pipeline testing

### **Day 6: Experimental Framework** 🚀
- [ ] Implement `RQ2Experiment` controller
- [ ] Run small-scale test experiments
- [ ] Debug and optimize pipeline
- [ ] Implement progress tracking
- [ ] Memory and performance profiling

### **Day 7: Full Experiments and Analysis** 🚀
- [ ] Run complete experiments on both datasets
- [ ] Perform statistical analysis (Friedman, Nemenyi)
- [ ] Generate metric rankings
- [ ] Create visualizations
- [ ] Validate H2 hypothesis

### **Day 8: Documentation and Validation** 🚀
- [ ] Generate comprehensive results report
- [ ] Create publication-ready figures
- [ ] Validate statistical significance
- [ ] Cross-check results consistency
- [ ] Prepare for RQ3 integration

## 7. Success Criteria

### **H2 Hypothesis Validation**:
- [ ] Identify top 5 performing similarity metrics
- [ ] Statistical significance (p < 0.05) in metric comparisons
- [ ] Consistent performance across datasets
- [ ] Macro-F1 scores > 0.70 for best metrics

### **Technical Achievements**:
- [x] 38 similarity metrics implemented and tested
- [x] Optimized NDG providing 10-100x speedup enables large-scale experiments
- [x] Robust cross-validation framework
- [ ] Comprehensive statistical analysis
- [ ] Publication-ready results and visualizations

### **Deliverables**:
- [x] Complete activity classification pipeline
- [ ] Statistical validation of metric discriminative power
- [ ] Ranked similarity metrics for sensor-based activity recognition
- [ ] Foundation for RQ3 cross-dataset robustness analysis

## 8. Risk Mitigation

### **Technical Risks**:
- **Memory usage**: Implement chunked processing for large similarity matrices
- **Computation time**: Leverage optimized NDG (10-100x speedup)
- **Class imbalance**: Apply balanced accuracy and stratified sampling
- **Overfitting**: Use robust cross-validation (LOWO)

### **Data Risks**:
- **Insufficient samples**: Filter activities with minimum sample counts
- **Noise sensitivity**: Apply appropriate preprocessing (low-pass filtering)
- **Dataset differences**: Normalize features and similarity scores

### **Timeline Risks**:
- **Scope creep**: Focus on core 38 metrics, defer additional variants
- **Debugging time**: Implement incremental testing at each phase
- **Statistical complexity**: Use established statistical packages (scipy.stats)

This detailed plan leverages the existing optimized infrastructure to deliver a comprehensive evaluation of fuzzy similarity metrics for activity classification, providing strong statistical evidence for the H2 hypothesis validation. 
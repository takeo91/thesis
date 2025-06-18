# Next Steps After Initial RQ2 Implementation

After completing the first iteration of the RQ2 implementation, here are the key next steps to enhance the analysis and expand the scope of the research.

## 0. Dataset Structure and Transformations

### Data Flow Pipeline

```
Raw Sensor Data → Windowed Data → Window Magnitude → Membership Functions → Similarity Matrices → Classification Results
(N × m)           (W × win_size × m)  (W × win_size)     (W × 100)           (W × W × 38)        (38 × metrics)
```

Where:
- N = Number of samples (51,116 for Opportunity, 376,416 for PAMAP2)
- m = Number of sensor channels (18 = 2 sensor types × 3 locations × 3 axes)
- W = Number of windows (16-77 for Opportunity, ~300 for PAMAP2)
- win_size = Window size (128 or 256 samples)

### Key Dataset Dimensions

| Dataset | Raw Samples | Filtered Samples | Sensors | Windows | Activities |
|---------|-------------|------------------|---------|---------|------------|
| Opportunity | 51,116 | 2,142 (ML_Both_Arms)<br/>37,507 (Locomotion) | 18 channels | 16-77 | 4-7 |
| PAMAP2 | 376,416 | 143,079 | 18 channels | ~300 | 7 |

### Data Transformation Steps

1. **Raw Data**: N samples × m sensors
   - Opportunity: 51,116 samples × 18 channels
   - PAMAP2: 376,416 samples × 18 channels

2. **Windowed Data**: W windows × window_size × m sensors
   - window_size: 128 or 256 samples
   - overlap: 50% or 70%

3. **Window Magnitude**: W windows × window_size
   - Compute Euclidean norm across all sensor channels

4. **Membership Functions**: W windows × grid_points
   - NDG computation with Epanechnikov kernel
   - grid_points = 100

5. **Similarity Matrices**: W × W matrices × 38 metrics
   - Pairwise similarity between all windows
   - 38 different similarity metrics

6. **Classification Results**: 38 metrics × performance metrics
   - F1 score, balanced accuracy, etc.

## 1. Enhanced Sensor Analysis

### 1.1 Multi-sensor Representation Strategies
- **Implement alternative sensor fusion approaches**:
  - Instead of using magnitude across all sensors, experiment with:
    - Per-sensor membership functions
    - Weighted sensor combinations based on activity relevance
    - Hierarchical fusion (first by body part, then across body)
  
- **Implementation**: Create a new module `thesis/fuzzy/multi_sensor.py` with different fusion strategies:
  ```python
  def compute_multi_sensor_membership(
      window_data: np.ndarray,
      fusion_strategy: str = "magnitude",  # "magnitude", "per_sensor", "weighted"
      weights: Optional[np.ndarray] = None
  ) -> List[np.ndarray]:
      """Compute membership functions using different sensor fusion strategies."""
  ```

### 1.2 Sensor Importance Analysis
- **Quantify contribution of each sensor to classification accuracy**:
  - Run ablation studies by removing one sensor at a time
  - Compute importance scores for each sensor type and body location
  - Identify optimal sensor subsets for different activities

- **Implementation**: Add a new experiment script `thesis/exp/sensor_importance.py`:
  ```python
  def compute_sensor_importance(
      experiment_results: Dict[str, ClassificationResults],
      sensor_config: Dict[str, Any]
  ) -> pd.DataFrame:
      """Compute importance scores for each sensor configuration."""
  ```

## 2. Advanced Membership Function Approaches

### 2.1 Multi-dimensional Fuzzy Sets
- **Explore multi-dimensional membership functions**:
  - Instead of reducing to 1D, maintain multi-dimensional relationships
  - Implement multi-dimensional NDG computation
  - Adapt similarity metrics for multi-dimensional fuzzy sets

- **Implementation**: Extend `thesis/fuzzy/membership.py` with:
  ```python
  def compute_multidimensional_ndg(
      x_values: List[np.ndarray],
      sensor_data: np.ndarray,
      sigma: Union[float, np.ndarray],
      kernel_type: str = "epanechnikov"
  ) -> np.ndarray:
      """Compute multi-dimensional NDG membership function."""
  ```

### 2.2 Adaptive Parameter Selection
- **Implement adaptive kernel and sigma selection**:
  - Automatically select optimal kernel type based on data characteristics
  - Use data-driven approaches to determine sigma parameter
  - Implement cross-validation for parameter optimization

- **Implementation**: Create `thesis/fuzzy/adaptive_parameters.py`:
  ```python
  def optimize_ndg_parameters(
      windows: List[np.ndarray],
      labels: np.ndarray,
      param_grid: Dict[str, List]
  ) -> Dict[str, Any]:
      """Find optimal NDG parameters using cross-validation."""
  ```

## 3. Expanded Evaluation Framework

### 3.1 Cross-dataset Validation
- **Evaluate generalization across datasets**:
  - Train on one dataset, test on another
  - Analyze which metrics generalize best across datasets
  - Implement domain adaptation techniques if necessary

- **Implementation**: Add to `thesis/exp/rq2_experiment.py`:
  ```python
  def run_cross_dataset_experiment(
      self,
      train_dataset: str,
      test_dataset: str
  ) -> Dict[str, ClassificationResults]:
      """Run cross-dataset validation experiment."""
  ```

### 3.2 Comprehensive Statistical Analysis
- **Implement rigorous statistical testing**:
  - Friedman test for overall metric comparison
  - Nemenyi post-hoc test for pairwise comparisons
  - Effect size calculations (Cohen's d, Cliff's delta)
  - Confidence intervals for performance metrics

- **Implementation**: Create `thesis/exp/rq2_statistical_analysis.py`:
  ```python
  def perform_statistical_tests(
      results_df: pd.DataFrame,
      alpha: float = 0.05
  ) -> Dict[str, Any]:
      """Perform comprehensive statistical analysis on results."""
  ```

### 3.3 Visualization Enhancements
- **Create advanced visualizations**:
  - Critical difference diagrams for metric comparison
  - Heatmaps showing metric performance across activities
  - Confusion matrices for different metrics
  - Membership function visualizations for different activities

- **Implementation**: Add to `thesis/exp/rq2_visualizations.py`:
  ```python
  def create_critical_difference_diagram(
      results_df: pd.DataFrame,
      output_path: str
  ) -> None:
      """Create critical difference diagram for metric comparison."""
  ```

## 4. Integration with RQ3

### 4.1 Robustness Analysis
- **Test robustness of top-performing metrics**:
  - Add synthetic noise to sensor data
  - Simulate sensor failures or missing data
  - Evaluate performance degradation under different conditions

- **Implementation**: Create `thesis/exp/robustness_analysis.py`:
  ```python
  def evaluate_metric_robustness(
      top_metrics: List[str],
      noise_levels: List[float],
      experiment_config: RQ2ExperimentConfig
  ) -> pd.DataFrame:
      """Evaluate robustness of top metrics to different noise levels."""
  ```

### 4.2 Transfer Learning Experiments
- **Investigate transfer learning capabilities**:
  - Train on one activity set, test on another
  - Analyze which metrics facilitate better knowledge transfer
  - Implement adaptation techniques for cross-activity learning

- **Implementation**: Add to `thesis/exp/rq3_experiment.py`:
  ```python
  def run_transfer_learning_experiment(
      source_activities: List[str],
      target_activities: List[str],
      top_metrics: List[str]
  ) -> Dict[str, Any]:
      """Run transfer learning experiment between activity sets."""
  ```

## 5. Implementation Timeline

| Week | Focus Area | Key Deliverables |
|------|------------|------------------|
| 1    | Enhanced Sensor Analysis | Multi-sensor fusion strategies, Sensor importance analysis |
| 2    | Advanced Membership Functions | Multi-dimensional fuzzy sets, Adaptive parameter selection |
| 3    | Expanded Evaluation | Cross-dataset validation, Statistical analysis |
| 4    | RQ3 Integration | Robustness analysis, Transfer learning experiments |

## 6. Prioritization Strategy

For immediate implementation, the following order is recommended:

1. **First priority**: Comprehensive statistical analysis of current results
   - This provides immediate insights with minimal additional implementation

2. **Second priority**: Sensor importance analysis
   - Helps understand which sensors contribute most to performance
   - Guides future sensor selection decisions

3. **Third priority**: Cross-dataset validation
   - Tests generalization capabilities of your metrics
   - Directly supports RQ3 objectives

4. **Fourth priority**: Advanced membership function approaches
   - More complex implementation but potentially higher performance gains
   - Consider after validating current approach thoroughly

This phased approach ensures you can deliver meaningful results quickly while systematically expanding the scope and depth of your analysis. 
# Interface Consolidation Plan

## 🎯 Issues Identified

### 1. Similarity Function Redundancy
**Current State:**
- `compute_per_sensor_similarity_optimized()` (working, used by unified windowing)
- `compute_per_sensor_similarity_vectorized()` (redundant)
- `compute_per_sensor_similarity_ultra_optimized()` (redundant)
- Plus `similarity_optimized.py` with `VectorizedSimilarityEngine`

**Problem:** Multiple interfaces doing the same thing, causing bugs and confusion.

### 2. Experiment File Proliferation 
**Current State:** 12 experiment files
- `unified_windowing_experiment.py` ✅ (main working interface)
- `rq2_experiment.py`, `rq1_experiment.py` (older versions)
- `activity_classification.py`, `pilot_driver.py`, etc. (experiments)

**Problem:** Too many entry points, unclear which to use.

## 🛠️ Consolidation Strategy

### Phase 1: Consolidate Similarity Functions ⚡ HIGH PRIORITY

#### Keep ONE similarity interface:
```python
# thesis/fuzzy/similarity.py - SIMPLIFIED
def compute_per_sensor_similarity(
    mu_i: List[np.ndarray],
    mu_j: List[np.ndarray], 
    x_values: np.ndarray,
    metric: str = "jaccard",
    normalise: bool = True,
) -> float:
    """
    THE unified similarity computation function.
    Replaces all compute_per_sensor_similarity_* variants.
    """
```

#### Remove redundant functions:
- ❌ `compute_per_sensor_similarity_vectorized()`
- ❌ `compute_per_sensor_similarity_ultra_optimized()` 
- ❌ `VectorizedSimilarityEngine` class
- ✅ Keep `compute_per_sensor_similarity_optimized()` → rename to `compute_per_sensor_similarity()`

### Phase 2: Consolidate Experiment Files 🧹 MEDIUM PRIORITY

#### Keep CORE experiment files:
- ✅ `unified_windowing_experiment.py` (main interface)
- ✅ `retrieval_utils.py` (evaluation utilities)
- ✅ `rq2_experiment.py` (if still needed for legacy)

#### Archive/Remove redundant files:
- ❌ `activity_classification.py` → Archive
- ❌ `pilot_driver.py` → Archive  
- ❌ `ndg_vs_kde.py` → Archive
- ❌ `rq1_experiment.py` → Archive (if results saved)
- ❌ `rq2_retrieval_experiment.py` → Archive
- ❌ `visualize_metric_comparison.py` → Archive

### Phase 3: Create Clean APIs 🎯 MEDIUM PRIORITY

#### Single Experiment Entry Point:
```python
# thesis/exp/__init__.py
from .unified_windowing_experiment import UnifiedWindowingExperiment

# Main public API
__all__ = ["UnifiedWindowingExperiment"]
```

#### Single Similarity Entry Point:
```python
# thesis/fuzzy/__init__.py  
from .similarity import (
    compute_per_sensor_similarity,
    calculate_all_similarity_metrics,
    # Basic functions
    similarity_jaccard,
    similarity_dice,
    similarity_cosine
)

__all__ = [
    "compute_per_sensor_similarity",  # Main function
    "calculate_all_similarity_metrics",
    "similarity_jaccard", "similarity_dice", "similarity_cosine"
]
```

## 🚀 Implementation Order

### Step 1: Fix Similarity Interface (IMMEDIATE)
1. Rename `compute_per_sensor_similarity_optimized` → `compute_per_sensor_similarity`
2. Remove redundant similarity functions
3. Update unified windowing to use simplified interface
4. Fix failing tests

### Step 2: Archive Redundant Experiments (NEXT)
1. Create `archive/experiments_legacy/` directory
2. Move unused experiment files there
3. Update imports and documentation

### Step 3: Clean Public APIs (FINAL)
1. Simplify `__init__.py` files
2. Create clear public interfaces
3. Update documentation
4. Ensure all tests pass

## 📊 Expected Benefits

### Immediate:
- ✅ **Fix current bugs** (similarity function confusion)
- ✅ **Reduce test failures** (clear interfaces)
- ✅ **Improve developer experience** (one way to do things)

### Long-term:
- ✅ **Easier maintenance** (fewer files to maintain)
- ✅ **Better documentation** (clear entry points)
- ✅ **Reduced confusion** (obvious which functions to use)
- ✅ **Higher test coverage** (focused testing)

## 🎯 Success Metrics

- [ ] Single similarity computation function
- [ ] ≤5 experiment files (vs current 12)
- [ ] All tests passing
- [ ] >50% test coverage
- [ ] Clear public APIs in `__init__.py` files
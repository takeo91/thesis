#!/usr/bin/env python3
"""
Test all expanded metrics to ensure they work properly before running full experiment.
"""

import numpy as np
from thesis.fuzzy.similarity import compute_per_sensor_similarity

def test_expanded_metrics():
    """Test all 13 expanded metrics with sample data."""
    
    # Create sample membership functions
    np.random.seed(42)
    x_values = np.linspace(0, 1, 100)
    
    # Sample membership functions (3 sensors)
    mu_i = [
        np.exp(-((x_values - 0.3)**2) / 0.1),  # Sensor 1
        np.exp(-((x_values - 0.5)**2) / 0.1),  # Sensor 2  
        np.exp(-((x_values - 0.7)**2) / 0.1),  # Sensor 3
    ]
    
    mu_j = [
        np.exp(-((x_values - 0.35)**2) / 0.1), # Sensor 1 (slightly different)
        np.exp(-((x_values - 0.55)**2) / 0.1), # Sensor 2 (slightly different)
        np.exp(-((x_values - 0.75)**2) / 0.1), # Sensor 3 (slightly different)
    ]
    
    # All 13 metrics to test
    expanded_metrics = [
        # Basic metrics (5)
        "jaccard", "cosine", "dice", "pearson", "overlap_coefficient",
        # Advanced correlation metrics (2)
        "spearman", "kendall_tau", 
        # Fuzzy set metrics (2)
        "tversky", "weighted_jaccard", 
        # Distance-based metrics (1)
        "mahalanobis",
        # Information-theoretic metrics (3)
        "jensen_shannon", "bhattacharyya_coefficient", "hellinger"
    ]
    
    print("🧪 Testing expanded metrics compatibility...")
    print(f"   Sample data: {len(mu_i)} sensors, {len(x_values)} points each")
    print(f"   Testing {len(expanded_metrics)} metrics\n")
    
    results = {}
    failed_metrics = []
    
    for metric in expanded_metrics:
        try:
            start_time = time.time()
            similarity = compute_per_sensor_similarity(mu_i, mu_j, x_values, metric=metric, normalise=True)
            elapsed = time.time() - start_time
            
            if np.isnan(similarity) or np.isinf(similarity):
                print(f"❌ {metric}: Invalid result ({similarity})")
                failed_metrics.append(metric)
            else:
                print(f"✅ {metric}: {similarity:.4f} (took {elapsed*1000:.1f}ms)")
                results[metric] = similarity
                
        except Exception as e:
            print(f"❌ {metric}: ERROR - {str(e)}")
            failed_metrics.append(metric)
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   ✅ Working metrics: {len(results)}/{len(expanded_metrics)}")
    print(f"   ❌ Failed metrics: {len(failed_metrics)}")
    
    if failed_metrics:
        print(f"   Failed: {failed_metrics}")
        return False
    else:
        print(f"   🎯 All {len(expanded_metrics)} metrics working correctly!")
        
        # Show performance ranking
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🏆 TOP PERFORMERS (sample data):")
        for i, (metric, score) in enumerate(sorted_results[:5], 1):
            print(f"   {i}. {metric}: {score:.4f}")
        
        return True

if __name__ == "__main__":
    import time
    success = test_expanded_metrics()
    if success:
        print(f"\n✅ Ready to run expanded metrics experiment!")
    else:
        print(f"\n❌ Fix failed metrics before running experiment!")
        exit(1)
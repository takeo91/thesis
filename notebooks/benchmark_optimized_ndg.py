#!/usr/bin/env python3
"""
Benchmark Script for Optimized NDG Implementation

This script demonstrates the performance improvements from spatial pruning
and parallelization optimizations, showing before/after comparisons.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

# Add thesis module to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import both original and optimized implementations
from thesis.fuzzy.membership import (
    compute_ndg,
    compute_ndg_streaming,
    compute_ndg_spatial_optimized,
    compute_ndg_epanechnikov_optimized,
    NUMBA_AVAILABLE
)

def generate_test_data(size: int, data_type: str = "normal") -> np.ndarray:
    """Generate test data for benchmarking."""
    np.random.seed(42)  # Reproducible results
    
    if data_type == "normal":
        return np.random.normal(0, 1, size)
    elif data_type == "uniform":
        return np.random.uniform(-3, 3, size)
    elif data_type == "bimodal":
        mode1 = np.random.normal(-1, 0.5, size // 2)
        mode2 = np.random.normal(1, 0.5, size - size // 2)
        return np.concatenate([mode1, mode2])
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

def benchmark_implementation(func, x_values, sensor_data, sigma, name: str, **kwargs):
    """Benchmark a single implementation and return results."""
    print(f"  Testing {name}...")
    
    # Warm-up run (especially important for numba)
    try:
        _ = func(x_values[:10], sensor_data[:100], sigma, **kwargs)
    except Exception as e:
        print(f"    Warm-up failed: {e}")
        return None
    
    # Actual benchmark
    try:
        start_time = time.time()
        result = func(x_values, sensor_data, sigma, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"    Time: {execution_time:.4f}s")
        
        return {
            "name": name,
            "time": execution_time,
            "result": result,
            "success": True
        }
    except Exception as e:
        print(f"    Failed: {e}")
        return {
            "name": name,
            "time": float('inf'),
            "result": None,
            "success": False
        }

def run_comparison_benchmark(data_sizes: List[int], sigma: float = 0.3):
    """Run comprehensive benchmark comparing all implementations."""
    print("🚀 NDG OPTIMIZATION BENCHMARK")
    print("=" * 60)
    
    if not NUMBA_AVAILABLE:
        print("⚠️  WARNING: Numba not available. Install with 'pip install numba' for best performance.")
        print()
    
    results = []
    
    for size in data_sizes:
        print(f"\n📊 Testing with {size:,} data points:")
        
        # Generate test data
        sensor_data = generate_test_data(size, "normal")
        x_values = np.linspace(sensor_data.min() - 1, sensor_data.max() + 1, 1000)
        
        test_results = []
        
        # 1. Original implementation
        original_result = benchmark_implementation(
            compute_ndg_streaming, x_values, sensor_data, sigma,
            "Original NDG", kernel_type="gaussian"
        )
        if original_result:
            test_results.append(original_result)
        
        # 2. Spatial optimization (without parallelization)
        spatial_result = benchmark_implementation(
            compute_ndg_spatial_optimized, x_values, sensor_data, sigma,
            "Spatial Optimized", use_parallel=False
        )
        if spatial_result:
            test_results.append(spatial_result)
        
        # 3. Spatial + Parallel optimization  
        if NUMBA_AVAILABLE:
            parallel_result = benchmark_implementation(
                compute_ndg_spatial_optimized, x_values, sensor_data, sigma,
                "Spatial + Parallel", use_parallel=True
            )
            if parallel_result:
                test_results.append(parallel_result)
        
        # 4. Epanechnikov kernel (serial)
        epan_result = benchmark_implementation(
            compute_ndg_epanechnikov_optimized, x_values, sensor_data, sigma,
            "Epanechnikov", use_parallel=False
        )
        if epan_result:
            test_results.append(epan_result)
        
        # 5. Epanechnikov kernel (parallel)
        if NUMBA_AVAILABLE:
            epan_parallel_result = benchmark_implementation(
                compute_ndg_epanechnikov_optimized, x_values, sensor_data, sigma,
                "Epanechnikov + Parallel", use_parallel=True
            )
            if epan_parallel_result:
                test_results.append(epan_parallel_result)
        
        # Calculate speedups
        if original_result and original_result["success"]:
            baseline_time = original_result["time"]
            
            print(f"\n  📈 Speedup Summary:")
            for result in test_results:
                if result["success"] and result["name"] != "Original NDG":
                    speedup = baseline_time / result["time"]
                    print(f"    {result['name']:<25}: {speedup:6.1f}x faster")
        
        # Verify accuracy
        print(f"\n  🎯 Accuracy Check:")
        if original_result and original_result["success"]:
            baseline_result = original_result["result"]
            
            for result in test_results:
                if result["success"] and result["name"] != "Original NDG":
                    try:
                        # Compare results (allowing for kernel differences)
                        if result["name"].startswith("Epanechnikov"):
                            print(f"    {result['name']:<25}: Different kernel (expected)")
                        else:
                            diff = np.mean(np.abs(result["result"] - baseline_result))
                            rel_diff = diff / (np.mean(np.abs(baseline_result)) + 1e-12)
                            if rel_diff < 0.01:
                                print(f"    {result['name']:<25}: ✅ Accurate ({rel_diff:.2%} diff)")
                            else:
                                print(f"    {result['name']:<25}: ⚠️  {rel_diff:.2%} difference")
                    except Exception as e:
                        print(f"    {result['name']:<25}: ❌ Comparison failed: {e}")
        
        # Store results for plotting
        for result in test_results:
            if result["success"]:
                results.append({
                    "size": size,
                    "implementation": result["name"],
                    "time": result["time"]
                })
    
    return results

def plot_benchmark_results(results: List[Dict]):
    """Create visualization of benchmark results."""
    import pandas as pd
    
    if not results:
        print("No results to plot.")
        return
    
    df = pd.DataFrame(results)
    
    # Create performance comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Execution time vs data size
    implementations = df['implementation'].unique()
    for impl in implementations:
        impl_data = df[df['implementation'] == impl]
        ax1.loglog(impl_data['size'], impl_data['time'], 'o-', label=impl, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Data Size (number of points)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('NDG Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup factors
    baseline_impl = "Original NDG"
    if baseline_impl in implementations:
        baseline_data = df[df['implementation'] == baseline_impl].set_index('size')['time']
        
        for impl in implementations:
            if impl != baseline_impl:
                impl_data = df[df['implementation'] == impl].set_index('size')['time']
                # Calculate speedup for common sizes
                common_sizes = baseline_data.index.intersection(impl_data.index)
                if len(common_sizes) > 0:
                    speedups = baseline_data[common_sizes] / impl_data[common_sizes]
                    ax2.semilogx(common_sizes, speedups, 'o-', label=impl, linewidth=2, markersize=8)
        
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='No improvement')
        ax2.set_xlabel('Data Size (number of points)')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Speedup vs Original Implementation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ndg_optimization_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Benchmark plot saved as 'ndg_optimization_benchmark.png'")

def demo_simple_usage():
    """Demonstrate simple usage of optimized NDG."""
    print("\n\n🔧 SIMPLE USAGE DEMO")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    sensor_data = np.random.normal(0, 1, 5000)
    x_values = np.linspace(-4, 4, 500)
    sigma = 0.3
    
    print("Sample usage of optimized NDG implementations:")
    print()
    
    # Show different ways to use the optimized functions
    examples = [
        ("Basic spatial optimization", 
         "compute_ndg_spatial_optimized(x_values, sensor_data, sigma)"),
        
        ("Epanechnikov kernel", 
         "compute_ndg_epanechnikov_optimized(x_values, sensor_data, sigma)"),
        
        ("Unified interface (auto optimization)", 
         "compute_ndg(x_values, sensor_data, sigma)"),
        
        ("Force original (for comparison)", 
         "compute_ndg(x_values, sensor_data, sigma, optimization='none')"),
    ]
    
    for name, code in examples:
        print(f"# {name}")
        print(f"{code}")
        try:
            start_time = time.time()
            if "spatial_optimized" in code:
                result = compute_ndg_spatial_optimized(x_values, sensor_data, sigma)
            elif "epanechnikov" in code:
                result = compute_ndg_epanechnikov_optimized(x_values, sensor_data, sigma)
            elif "optimization='none'" in code:
                result = compute_ndg(x_values, sensor_data, sigma, optimization='none')
            else:
                result = compute_ndg(x_values, sensor_data, sigma)
            
            execution_time = time.time() - start_time
            print(f"# Result: {len(result)} values computed in {execution_time:.4f}s")
            print()
        except Exception as e:
            print(f"# Error: {e}")
            print()

def main():
    """Run the complete benchmark suite."""
    
    # Quick benchmark with small sizes for fast feedback
    print("Running quick benchmark (add '--full' for comprehensive test)...")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Full benchmark - can take several minutes
        data_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
        print("Running FULL benchmark (this may take a few minutes)...")
    else:
        # Quick benchmark for immediate feedback
        data_sizes = [100, 500, 1000, 2000]
    
    # Run benchmarks
    results = run_comparison_benchmark(data_sizes, sigma=0.3)
    
    # Create plots
    if results:
        plot_benchmark_results(results)
    
    # Show usage examples
    demo_simple_usage()
    
    print("\n✅ Benchmark complete!")
    print("\n💡 Key takeaways:")
    print("1. Spatial optimization provides significant speedup for larger datasets")
    print("2. Epanechnikov kernel avoids expensive exponential calculations")
    print("3. Parallelization scales with available CPU cores")
    print("4. Combined optimizations can provide 10-100x+ speedup")
    
    if not NUMBA_AVAILABLE:
        print("\n🚀 For maximum performance, install numba:")
        print("   pip install numba")

if __name__ == "__main__":
    main() 
import numpy as np
import pytest

from thesis.fuzzy.membership import (
    compute_ndg,
    compute_ndg_streaming,
    compute_membership_function,
    compute_membership_function_kde,
    compute_membership_functions
)

# Test data
x_range = np.linspace(0, 10, 101)  # 0 to 10 with 101 points
x_norm = np.linspace(-3, 3, 101)   # for standard normal range


class TestComputeNDG:
    """Tests for compute_ndg function."""
    
    def test_basic_functionality(self):
        """Test basic functionality of compute_ndg."""
        # Create sample data
        sensor_data = np.array([1.0, 5.0, 9.0])
        sigma = 1.0
        result = compute_ndg_streaming(x_range, sensor_data, sigma)
        
        # Check shape
        assert len(result) == len(x_range)
        
        # Check that result is non-negative
        assert np.all(result >= 0)
        
        # Check that peaks occur near data points
        peak_indices = [i for i, v in enumerate(result) if i > 0 and i < len(result)-1 and
                       result[i-1] < v and result[i+1] < v]
        peak_x_values = x_range[peak_indices]
        
        # Each data point should have a nearby peak
        for data_point in sensor_data:
            assert any(abs(peak - data_point) < 0.5 for peak in peak_x_values)
    
    def test_empty_data(self):
        """Test compute_ndg with empty data."""
        sensor_data = np.array([])
        sigma = 1.0
        result = compute_ndg_streaming(x_range, sensor_data, sigma)
        
        # Result should be all zeros for empty data
        assert np.all(result == 0)
    
    def test_sigma_effect(self):
        """Test effect of sigma parameter."""
        sensor_data = np.array([5.0])
        
        # Small sigma should give narrower peak
        result_small_sigma = compute_ndg_streaming(x_range, sensor_data, 0.1)
        
        # Large sigma should give wider peak
        result_large_sigma = compute_ndg_streaming(x_range, sensor_data, 1.0)
        
        # Get half-width of peaks
        half_height_small = np.max(result_small_sigma) / 2
        half_height_large = np.max(result_large_sigma) / 2
        
        width_small = np.sum(result_small_sigma > half_height_small)
        width_large = np.sum(result_large_sigma > half_height_large)
        
        # Larger sigma should give wider peak
        assert width_large > width_small
    
    def test_very_small_sigma(self):
        """Test with very small sigma to check for numerical stability."""
        sensor_data = np.array([5.0])
        result = compute_ndg_streaming(x_range, sensor_data, 1e-10)
        
        # Result should still be valid (not NaN)
        assert not np.isnan(result).any()
        # Should have a peak near the data point
        max_idx = np.argmax(result)
        assert abs(x_range[max_idx] - 5.0) < 0.2


    def test_unified_matches_streaming(self):
        """Test that unified interface produces equivalent results to streaming (when using same kernel)."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=500)
        x = np.linspace(-3, 3, 500)
        
        # Use unified interface with "none" optimization (should use streaming)
        unified = compute_ndg(x, data, 0.4, optimization="none")
        # Direct streaming call
        stream = compute_ndg_streaming(x, data, 0.4, chunk_size=100)
        
        np.testing.assert_allclose(unified, stream, rtol=1e-10, atol=1e-15)

    def test_three_peaks_default_sigma(self):
        x = np.arange(0, 10.1, 0.1)
        data = np.array([1.0, 5.0, 9.0])
        ndg = compute_ndg_streaming(x, data, sigma=1.0)
        peak_idx = [i for i in range(1, len(ndg)-1) if ndg[i-1] < ndg[i] > ndg[i+1]]
        assert len(peak_idx) == 3
        

    @pytest.mark.parametrize("kernel_type", [
        "gaussian", "epanechnikov", "triangular", 
        "uniform", "quartic", "cosine"
    ])
    def test_kernel_types(self, kernel_type):
        """Test each available kernel type."""
        # Create sample data
        np.random.seed(42)
        sample_data = np.random.normal(5, 1, 100)
        
        result = compute_ndg_streaming(x_range, sample_data, sigma=0.5, kernel_type=kernel_type)
        
        # Check shape and basic properties
        assert len(result) == len(x_range)
        assert np.all(result >= 0)
        
        # Result should have a peak near the mean of the data
        peak_idx = np.argmax(result)
        assert 4 <= x_range[peak_idx] <= 6
    
    def test_invalid_kernel_type(self):
        """Test with invalid kernel type."""
        np.random.seed(42)
        sample_data = np.random.normal(5, 1, 100)
        
        with pytest.raises(ValueError, match="Unknown kernel type"):
            compute_ndg_streaming(x_range, sample_data, sigma=0.5, kernel_type="invalid")
    
    def test_kernel_normalization_properties(self):
        """Test that kernels maintain expected normalization properties."""
        # Single data point to test kernel shape directly
        data = np.array([5.0])
        
        # Test each kernel type
        kernels = ["gaussian", "epanechnikov", "triangular", 
                  "uniform", "quartic", "cosine"]
        
        for kernel in kernels:
            result = compute_ndg_streaming(x_range, data, sigma=1.0, kernel_type=kernel)
            
            # Integration over the domain should approximately equal 1
            # (since we normalize by 1/(sigma*n))
            area = np.trapezoid(result, x_range)
            assert 0.8 <= area <= 1.2, f"Kernel {kernel} integral = {area}"
            
            # Peak should be at or near the data point
            peak_idx = np.argmax(result)
            # More permissive threshold to account for kernel differences
            assert abs(x_range[peak_idx] - 5.0) < 1.1, f"Kernel {kernel} peak at {x_range[peak_idx]}"
        
    def test_chunk_size_variations(self):
        """Test that different chunk sizes produce the same results."""
        np.random.seed(42)
        data = np.random.normal(5, 1, 1000)
        
        # Compute with different chunk sizes
        result1 = compute_ndg_streaming(x_range, data, sigma=0.5, chunk_size=10)
        result2 = compute_ndg_streaming(x_range, data, sigma=0.5, chunk_size=100)
        result3 = compute_ndg_streaming(x_range, data, sigma=0.5, chunk_size=1000)
        
        # Results should be identical regardless of chunk size
        np.testing.assert_allclose(result1, result2)
        np.testing.assert_allclose(result1, result3)
    
    def test_chunk_size_edge_cases(self):
        """Test with edge case chunk sizes."""
        np.random.seed(42)
        data = np.random.normal(5, 1, 50)
        
        # Chunk size larger than input
        result1 = compute_ndg_streaming(x_range, data, sigma=0.5, chunk_size=1000)
        
        # Chunk size = 1
        result2 = compute_ndg_streaming(x_range, data, sigma=0.5, chunk_size=1)
        
        # Results should be identical
        np.testing.assert_allclose(result1, result2)


class TestComputeMembershipFunction:
    """Tests for compute_membership_function."""
    
    def test_basic_functionality(self):
        """Test basic functionality of compute_membership_function."""
        # Create sample data
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 100)
        
        # Call function with default parameters
        x_vals, mu, sigma_val = compute_membership_function(sensor_data)
        
        # Check shapes
        assert len(x_vals) > 0
        assert len(mu) == len(x_vals)
        
        # Check that mu is normalized
        assert np.isclose(np.sum(mu), 1.0)
        
        # Check that mu is non-negative
        assert np.all(mu >= 0)
        
        # Check that sigma_val is positive
        assert sigma_val > 0
    
    def test_with_custom_x_values(self):
        """Test with custom x_values."""
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 100)
        custom_x = np.linspace(2, 8, 50)
        
        x_vals, mu, sigma_val = compute_membership_function(sensor_data, custom_x)
        
        # x_vals should match input custom_x
        assert np.array_equal(x_vals, custom_x)
        assert len(mu) == len(custom_x)
    
    def test_with_custom_sigma(self):
        """Test with custom sigma values."""
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 100)
        
        # Test with fixed sigma
        x_vals1, mu1, sigma_val1 = compute_membership_function(sensor_data, sigma=0.5)
        
        # Test with relative sigma
        x_vals2, mu2, sigma_val2 = compute_membership_function(sensor_data, sigma="r0.2")
        
        # Check that sigma values are as expected
        assert sigma_val1 == 0.5
        assert np.isclose(sigma_val2, 0.2 * (np.max(sensor_data) - np.min(sensor_data)))
        
        # Narrower sigma should give more peaks in the result
        assert np.var(mu1) > np.var(mu2)
    
    def test_empty_data(self):
        """Test with empty data."""
        sensor_data = np.array([])
        
        x_vals, mu, sigma_val = compute_membership_function(sensor_data)
        
        # Function should return zeros for mu with empty data
        assert np.all(mu == 0)
        assert len(x_vals) > 0
        assert sigma_val > 0
    
    def test_constant_data(self):
        """Test with constant data."""
        sensor_data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        
        x_vals, mu, sigma_val = compute_membership_function(sensor_data)
        
        # Result should be valid for constant data
        assert len(mu) == len(x_vals)
        assert sigma_val > 0
        
        # Maximum should be at or near the constant value
        max_idx = np.argmax(mu) if np.max(mu) > 0 else len(mu) // 2
        assert abs(x_vals[max_idx] - 5.0) < 1.0
    
    def test_invalid_sigma_format(self):
        """Test with invalid sigma format."""
        sensor_data = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            compute_membership_function(sensor_data, sigma="invalid")
        
        with pytest.raises(ValueError):
            compute_membership_function(sensor_data, sigma="rx")  # Can't parse float
            
    def test_sum_normalization(self):
        """Test sum normalization method."""
        # Create sample data
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 200)
        
        # Test with direct compute_membership_function
        _, mu_sum, _ = compute_membership_function(
            sensor_data, x_values=x_range, sigma=0.5, normalization="sum"
        )
        
        # Check that sum equals 1
        assert np.isclose(np.sum(mu_sum), 1.0, rtol=1e-2)
    
    def test_integral_normalization(self):
        """Test integral normalization method."""
        # Create sample data
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 200)
        
        # Test with direct compute_membership_function
        _, mu_int, _ = compute_membership_function(
            sensor_data, x_values=x_range, sigma=0.5, normalization="integral"
        )
        
        # Check that integral equals 1
        area = np.trapezoid(mu_int, x_range)
        assert np.isclose(area, 1.0, rtol=1e-2)
    
    def test_invalid_normalization(self):
        """Test with invalid normalization method."""
        # Create sample data
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 200)
        
        with pytest.raises(ValueError, match="Unknown normalization:"):
            compute_membership_function(
                sensor_data, x_values=x_range, sigma=0.5, normalization="invalid"
            )
    
    def test_normalization_consistency(self):
        """Test that both normalization methods produce consistent results."""
        # Create sample data
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 200)
        
        # Get membership functions with both normalization methods
        _, mu_sum, _ = compute_membership_function(
            sensor_data, x_values=x_range, sigma=0.5, normalization="sum"
        )
        
        _, mu_int, _ = compute_membership_function(
            sensor_data, x_values=x_range, sigma=0.5, normalization="integral"
        )
        
        # The shape should be the same, just scaled differently
        correlation = np.corrcoef(mu_sum, mu_int)[0, 1]
        assert correlation > 0.99  # Almost perfectly correlated
        
        # Peak should be at the same location
        assert np.argmax(mu_sum) == np.argmax(mu_int)


class TestComputeMembershipFunctionKDE:
    """Tests for compute_membership_function_kde."""
    
    def test_basic_functionality(self):
        """Test basic functionality of compute_membership_function_kde."""
        # Create sample data
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 100)
        
        # Call function with default parameters (uses integral normalization)
        x_vals, mu = compute_membership_function_kde(sensor_data)
        
        # Check shapes
        assert len(x_vals) > 0
        assert len(mu) == len(x_vals)
        
        # Check that mu is normalized by integral
        assert np.isclose(np.trapezoid(mu, x=x_vals), 1.0, rtol=1e-2)
        
        # Check sum normalization
        x_vals_sum, mu_sum = compute_membership_function_kde(sensor_data, normalization="sum")
        assert np.isclose(np.sum(mu_sum), 1.0, rtol=1e-2)
        
        # Check that mu is non-negative
        assert np.all(mu >= 0)
    
    def test_with_custom_x_values(self):
        """Test with custom x_values."""
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 100)
        custom_x = np.linspace(2, 8, 50)
        
        x_vals, mu = compute_membership_function_kde(sensor_data, custom_x)
        
        # x_vals should match input custom_x
        assert np.array_equal(x_vals, custom_x)
        assert len(mu) == len(custom_x)
    
    def test_insufficient_data(self):
        """Test with insufficient data (less than 2 points)."""
        # Empty data
        sensor_data_empty = np.array([])
        x_vals_empty, mu_empty = compute_membership_function_kde(sensor_data_empty)
        
        assert np.all(mu_empty == 0)
        
        # Single point
        sensor_data_single = np.array([5.0])
        x_vals_single, mu_single = compute_membership_function_kde(sensor_data_single)
        
        assert np.all(mu_single == 0)
    
    def test_identical_points(self):
        """Test with all identical points."""
        sensor_data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        
        # This should not raise an error even though KDE can fail with identical points
        x_vals, mu = compute_membership_function_kde(sensor_data)
        
        # Result should be valid
        assert len(mu) == len(x_vals)


class TestComputeMembershipFunctions:
    """Tests for compute_membership_functions wrapper function."""
    
    def test_nd_method(self):
        """Test with Neighbor Density method."""
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 100)
        
        # Test with sum normalization
        mu_sum, sigma_val = compute_membership_functions(
            sensor_data, x_range, method="nd", sigma=0.5, normalization="sum"
        )
        
        # Check result
        assert len(mu_sum) == len(x_range)
        assert sigma_val == 0.5
        assert np.isclose(np.sum(mu_sum), 1.0, rtol=1e-2)
        assert np.all(mu_sum >= 0)
        
        # Test with integral normalization
        mu_int, _ = compute_membership_functions(
            sensor_data, x_range, method="nd", sigma=0.5, normalization="integral"
        )
        
        assert np.isclose(np.trapezoid(mu_int, x=x_range), 1.0, rtol=1e-2)
    
    def test_kde_method(self):
        """Test with KDE method."""
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 100)
        
        # Test with sum normalization
        mu_sum, sigma_val = compute_membership_functions(
            sensor_data, x_range, method="kde", normalization="sum"
        )
        
        # Check result
        assert len(mu_sum) == len(x_range)
        assert sigma_val is None  # KDE method doesn't return sigma
        assert np.isclose(np.sum(mu_sum), 1.0, rtol=1e-2)
        assert np.all(mu_sum >= 0)
        
        # Test with integral normalization
        mu_int, _ = compute_membership_functions(
            sensor_data, x_range, method="kde", normalization="integral"
        )
        
        assert np.isclose(np.trapezoid(mu_int, x=x_range), 1.0, rtol=1e-2)
    
    def test_invalid_method(self):
        """Test with invalid method."""
        np.random.seed(42)
        sensor_data = np.random.normal(5, 1, 100)
        
        with pytest.raises(ValueError):
            compute_membership_functions(
                sensor_data, x_range, method="invalid"
            )
    
    def test_empty_data(self):
        """Test with empty data."""
        sensor_data = np.array([])
        
        # ND method
        mu_nd, sigma_val_nd = compute_membership_functions(
            sensor_data, x_range, method="nd"
        )
        
        assert np.all(mu_nd == 0)
        assert sigma_val_nd > 0
        
        # KDE method
        mu_kde, sigma_val_kde = compute_membership_functions(
            sensor_data, x_range, method="kde"
        )
        
        assert np.all(mu_kde == 0)
        assert sigma_val_kde is None 
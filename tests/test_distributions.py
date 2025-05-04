import numpy as np
import pytest
from scipy import stats

from thesis.fuzzy.distributions import (
    normalize_data,
    standardize_data,
    compute_empirical_distribution_kde,
    compute_error_metrics,
    compute_kl_divergence,
    compute_information_criteria,
    compute_fitness_metrics
)

class TestNormalizeData:
    """Tests for normalize_data function."""
    
    def test_basic_normalization(self):
        """Test basic normalization functionality."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_data(data)
        
        assert result[0] == pytest.approx(0.0)
        assert result[-1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(0.5)  # (3-1)/(5-1) = 0.5
        
    def test_empty_data(self):
        """Test handling of empty data."""
        data = np.array([])
        result = normalize_data(data)
        
        # Should return empty array
        assert len(result) == 0
    
    def test_single_value(self):
        """Test handling of single value data."""
        data = np.array([42.0])
        result = normalize_data(data)
        
        # Should handle single value and return 0.5
        assert result[0] == 0.5
    
    def test_constant_values(self):
        """Test handling of constant values."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = normalize_data(data)
        
        # Should handle constant values and return 0.5 for all elements
        assert np.all(result == 0.5)
    
    def test_negative_values(self):
        """Test with negative values."""
        data = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = normalize_data(data)
        
        # Result should be in [0, 1] range
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        
        # Check linear mapping
        assert result[0] == pytest.approx(0.0)  # min value
        assert result[2] == pytest.approx(0.5)  # (0-(-10))/(10-(-10)) = 0.5
        assert result[-1] == pytest.approx(1.0)  # max value


class TestStandardizeData:
    """Tests for standardize_data function."""
    
    def test_basic_standardization(self):
        """Test basic standardization functionality."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = standardize_data(data)
        
        # Mean of original data is 3, std is sqrt(2)
        assert np.mean(result) == pytest.approx(0.0)
        assert np.std(result) == pytest.approx(1.0)
        
        # Check specific values: (x - mean) / std
        assert result[0] == pytest.approx((1 - 3) / np.std(data, ddof=0))
        assert result[-1] == pytest.approx((5 - 3) / np.std(data, ddof=0))
    
    def test_empty_data(self):
        """Test handling of empty data."""
        data = np.array([])
        result = standardize_data(data)
        
        # Should return empty array
        assert len(result) == 0
    
    def test_constant_values(self):
        """Test handling of constant values."""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = standardize_data(data)
        
        # Should handle constant values and return zeros
        assert np.all(result == 0.0)


class TestComputeEmpiricalDistributionKDE:
    """Tests for compute_empirical_distribution_kde function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with random data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        x = np.linspace(-3, 3, 50)
        result = compute_empirical_distribution_kde(x, data)
        
        # Result should be a probability distribution (non-negative and sums to 1)
        assert np.all(result >= 0)
        assert np.isclose(np.sum(result), 1.0)
    
    def test_insufficient_data(self):
        """Test with insufficient data (less than 2 points)."""
        # Empty data
        data_empty = np.array([])
        x = np.linspace(-3, 3, 50)
        result_empty = compute_empirical_distribution_kde(x, data_empty)
        
        assert np.all(result_empty == 0)
        
        # Single point
        data_single = np.array([0.0])
        result_single = compute_empirical_distribution_kde(x, data_single)
        
        assert np.all(result_single == 0)


class TestComputeErrorMetrics:
    """Tests for compute_error_metrics function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with simple test data."""
        empirical = np.array([0.1, 0.2, 0.3, 0.4])
        theoretical = np.array([0.2, 0.2, 0.2, 0.4])
        
        result = compute_error_metrics(empirical, theoretical)
        
        # Manual calculations
        errors = empirical - theoretical
        expected_mse = np.mean(errors**2)
        expected_rmse = np.sqrt(expected_mse)
        expected_mae = np.mean(np.abs(errors))
        
        assert result["MSE"] == pytest.approx(expected_mse)
        assert result["RMSE"] == pytest.approx(expected_rmse)
        assert result["MAE"] == pytest.approx(expected_mae)
    
    def test_mismatched_shapes(self):
        """Test with mismatched shapes."""
        empirical = np.array([0.1, 0.2, 0.3, 0.4])
        theoretical = np.array([0.2, 0.2, 0.2])
        
        with pytest.raises(ValueError):
            compute_error_metrics(empirical, theoretical)


class TestComputeKLDivergence:
    """Tests for compute_kl_divergence function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with simple test data."""
        # Simple discrete distributions
        empirical = np.array([0.5, 0.5, 0.0, 0.0])
        theoretical = np.array([0.25, 0.25, 0.25, 0.25])
        
        result = compute_kl_divergence(empirical, theoretical)
        
        # KL(P||Q) = sum(P_i * log(P_i/Q_i))
        # For our case: 0.5*log(0.5/0.25) + 0.5*log(0.5/0.25) = log(2) ≈ 0.693
        expected = 0.5 * np.log(0.5/0.25) + 0.5 * np.log(0.5/0.25)
        assert result == pytest.approx(expected)
    
    def test_identical_distributions(self):
        """Test KL divergence of identical distributions."""
        # KL divergence of identical distributions should be 0
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = compute_kl_divergence(p, p)
        assert result == pytest.approx(0.0)
    
    def test_zero_distributions(self):
        """Test handling of zero distributions."""
        p = np.array([0.0, 0.0, 0.0, 0.0])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Should return NaN for zero distributions
        result = compute_kl_divergence(p, q)
        assert np.isnan(result)


class TestComputeInformationCriteria:
    """Tests for compute_information_criteria function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with simple test data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        # Create a normal density function matching the data
        x_values = np.linspace(-3, 3, 50)
        pdf = stats.norm.pdf(x_values, loc=np.mean(data), scale=np.std(data))
        
        result = compute_information_criteria(data, pdf, x_values, num_params=2)
        
        # Just check that values are finite and well-formed
        assert "AIC" in result
        assert "BIC" in result
        assert np.isfinite(result["AIC"])
        assert np.isfinite(result["BIC"])
    
    def test_empty_data(self):
        """Test handling of empty inputs."""
        # Empty data
        result_empty_data = compute_information_criteria(
            np.array([]), np.array([0.1, 0.2, 0.3]), np.array([1, 2, 3]), 2
        )
        assert result_empty_data["AIC"] == np.inf
        assert result_empty_data["BIC"] == np.inf
        
        # Empty PDF
        result_empty_pdf = compute_information_criteria(
            np.array([1, 2, 3]), np.array([]), np.array([]), 2
        )
        assert result_empty_pdf["AIC"] == np.inf
        assert result_empty_pdf["BIC"] == np.inf


class TestComputeFitnessMetrics:
    """Tests for compute_fitness_metrics function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with simple test data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        # Create a PDF matching the data
        x_values = np.linspace(-3, 3, 50)
        pdf = stats.norm.pdf(x_values, loc=np.mean(data), scale=np.std(data))
        
        # Ensure PDF is normalized to sum to 1
        pdf_sum = np.sum(pdf)
        pdf = pdf / pdf_sum
        
        # Call the function
        result, empirical_probs = compute_fitness_metrics(data, pdf, x_values)
        
        # Check that all expected metrics are present
        expected_keys = [
            "MSE", "RMSE", "MAE", "KL_Divergence", 
            "AIC", "BIC", "MSE_CV", "KL_Divergence_CV"
        ]
        for key in expected_keys:
            assert key in result
            assert np.isfinite(result[key]) or np.isnan(result[key])
        
        # Check that empirical_probs is a valid distribution
        assert empirical_probs is not None
        assert np.all(empirical_probs >= 0)
        assert np.isclose(np.sum(empirical_probs), 1.0, atol=1e-2)
    
    def test_empty_data(self):
        """Test handling of empty inputs."""
        # Empty data
        result_empty, empirical_empty = compute_fitness_metrics(
            np.array([]), np.array([0.1, 0.2, 0.3]), np.array([1, 2, 3])
        )
        assert all(np.isnan(value) for value in result_empty.values())
        assert empirical_empty is None
    
    def test_invalid_empirical_method(self):
        """Test handling of invalid empirical method."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        x_values = np.linspace(-3, 3, 50)
        pdf = stats.norm.pdf(x_values, loc=0, scale=1)
        
        with pytest.raises(ValueError):
            compute_fitness_metrics(data, pdf, x_values, empirical_method="invalid") 
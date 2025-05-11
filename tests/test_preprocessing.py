import numpy as np
import pytest

from thesis.core.preprocessing import normalize_data, standardize_data


class TestNormalizeData:
    """Tests for normalize_data function."""
    
    def test_basic_functionality(self):
        """Test basic functionality of normalize_data."""
        # Test with a simple array
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_data(data)
        
        # Check shape and type
        assert normalized.shape == data.shape
        assert normalized.dtype == np.float64
        
        # Check range: should be in [0,1]
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        
        # Check specific values
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(normalized, expected)
    
    def test_already_normalized(self):
        """Test with data already in [0,1] range."""
        data = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        normalized = normalize_data(data)
        
        # Should match the input
        np.testing.assert_allclose(normalized, data)
    
    def test_negative_values(self):
        """Test with negative values."""
        data = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
        normalized = normalize_data(data)
        
        # Check range
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        
        # Check specific values
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(normalized, expected)
    
    def test_single_value(self):
        """Test with a single value."""
        data = np.array([42.0])
        normalized = normalize_data(data)
        
        # With only one value, should return 0.5
        assert normalized[0] == 0.5
    
    def test_constant_data(self):
        """Test with constant data (all values the same)."""
        data = np.array([7.0, 7.0, 7.0, 7.0])
        normalized = normalize_data(data)
        
        # With constant data, should return all 0.5s
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(normalized, expected)
    
    def test_empty_data(self):
        """Test with empty data."""
        data = np.array([])
        normalized = normalize_data(data)
        
        # Should return empty array
        assert normalized.size == 0
        
    def test_different_input_types(self):
        """Test with different input types."""
        # List input
        data_list = [1.0, 3.0, 5.0]
        normalized_list = normalize_data(data_list)
        assert isinstance(normalized_list, np.ndarray)
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(normalized_list, expected)
        
        # Tuple input
        data_tuple = (1.0, 3.0, 5.0)
        normalized_tuple = normalize_data(data_tuple)
        assert isinstance(normalized_tuple, np.ndarray)
        np.testing.assert_allclose(normalized_tuple, expected)
        
        # 2D array input
        data_2d = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
        normalized_2d = normalize_data(data_2d)
        assert normalized_2d.shape == data_2d.shape
        assert np.min(normalized_2d) == 0.0
        assert np.max(normalized_2d) == 1.0


class TestStandardizeData:
    """Tests for standardize_data function."""
    
    def test_basic_functionality(self):
        """Test basic functionality of standardize_data."""
        # Test with a simple array
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        standardized = standardize_data(data)
        
        # Check shape and type
        assert standardized.shape == data.shape
        
        # Mean should be approximately 0, std should be approximately 1
        assert np.isclose(np.mean(standardized), 0.0, atol=1e-10)
        assert np.isclose(np.std(standardized), 1.0, atol=1e-10)
        
        # Check specific values
        mean = np.mean(data)
        std = np.std(data)
        expected = (data - mean) / std
        np.testing.assert_allclose(standardized, expected)
    
    def test_already_standardized(self):
        """Test with already standardized data."""
        # Generate data with mean 0 and std 1
        data = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
        standardized = standardize_data(data)
        
        # Should be similar to input but not exact due to sample vs population std
        assert np.isclose(np.mean(standardized), 0.0, atol=1e-10)
        assert np.isclose(np.std(standardized), 1.0, atol=1e-10)
    
    def test_single_value(self):
        """Test with a single value."""
        data = np.array([42.0])
        standardized = standardize_data(data)
        
        # Convert to array if necessary
        standardized = np.atleast_1d(standardized)
        
        # With only one value, std is 0, so should return 0
        assert standardized[0] == 0.0
    
    def test_constant_data(self):
        """Test with constant data (all values the same)."""
        data = np.array([7.0, 7.0, 7.0, 7.0])
        standardized = standardize_data(data)
        
        # With constant data, std is 0, so should return all 0s
        expected = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(standardized, expected)
    
    def test_empty_data(self):
        """Test with empty data."""
        data = np.array([])
        standardized = standardize_data(data)
        
        # Should return empty array
        assert standardized.size == 0
        
    def test_different_input_types(self):
        """Test with different input types."""
        # List input
        data_list = [1.0, 2.0, 3.0]
        standardized_list = standardize_data(data_list)
        assert isinstance(standardized_list, np.ndarray)
        
        # Mean should be 0, std should be 1
        assert np.isclose(np.mean(standardized_list), 0.0, atol=1e-10)
        assert np.isclose(np.std(standardized_list), 1.0, atol=1e-10)
        
        # Tuple input
        data_tuple = (1.0, 2.0, 3.0)
        standardized_tuple = standardize_data(data_tuple)
        assert isinstance(standardized_tuple, np.ndarray)
        assert np.isclose(np.mean(standardized_tuple), 0.0, atol=1e-10)
        assert np.isclose(np.std(standardized_tuple), 1.0, atol=1e-10)
        
        # 2D array input
        data_2d = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
        standardized_2d = standardize_data(data_2d)
        assert standardized_2d.shape == data_2d.shape
        assert np.isclose(np.mean(standardized_2d), 0.0, atol=1e-10)
        assert np.isclose(np.std(standardized_2d), 1.0, atol=1e-10)
    
    def test_data_with_outliers(self):
        """Test with data containing outliers."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        standardized = standardize_data(data)
        
        # Mean should be 0, std should be 1
        assert np.isclose(np.mean(standardized), 0.0, atol=1e-10)
        assert np.isclose(np.std(standardized), 1.0, atol=1e-10)
        
        # The outlier should be a high positive z-score
        assert standardized[-1] > 0
        
        # The other values should be negative z-scores (below mean)
        assert np.all(standardized[:-1] < 0) 
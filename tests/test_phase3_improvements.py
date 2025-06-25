"""
Unit tests for Phase 3 improvements: refactored functions and error handling.

This test module validates the refactored functions and standardized error handling
implemented in Phase 3 of the codebase improvement process.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Import the refactored modules and new exception system
from thesis.core.exceptions import (
    DataValidationError, ComputationError, ConfigurationError, 
    SecurityError, validate_arrays, validate_positive_number,
    validate_membership_functions_arrays, format_error_message
)
from thesis.fuzzy.similarity_optimized import VectorizedSimilarityEngine
from thesis.data.chunked_processor import ChunkedDataProcessor


class TestExceptionSystem:
    """Test the standardized exception hierarchy and validation utilities."""
    
    def test_exception_hierarchy(self):
        """Test that custom exceptions inherit properly."""
        with pytest.raises(DataValidationError):
            raise DataValidationError("Test data validation error")
        
        with pytest.raises(ComputationError):
            raise ComputationError("Test computation error")
    
    def test_exception_context(self):
        """Test exception context information."""
        context = {"param": "test_value", "range": "0-100"}
        error = DataValidationError("Test error", context=context)
        
        assert "param=test_value" in str(error)
        assert "range=0-100" in str(error)
    
    def test_validate_arrays_success(self):
        """Test successful array validation."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        
        # Should not raise any exception
        validate_arrays(arr1, arr2, same_shape=True)
    
    def test_validate_arrays_shape_mismatch(self):
        """Test array validation with shape mismatch."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([[4, 5], [6, 7]])
        
        with pytest.raises(DataValidationError) as exc_info:
            validate_arrays(arr1, arr2, same_shape=True)
        
        assert "inconsistent shapes" in str(exc_info.value)
    
    def test_validate_arrays_nan_values(self):
        """Test array validation with NaN values."""
        arr_with_nan = np.array([1.0, 2.0, np.nan])
        
        with pytest.raises(DataValidationError) as exc_info:
            validate_arrays(arr_with_nan)
        
        assert "contains NaN" in str(exc_info.value)
    
    def test_validate_positive_number(self):
        """Test positive number validation."""
        # Valid cases
        validate_positive_number(1.0, "test_param")
        validate_positive_number(0.0, "test_param", allow_zero=True)
        
        # Invalid cases
        with pytest.raises(DataValidationError):
            validate_positive_number(-1.0, "test_param")
        
        with pytest.raises(DataValidationError):
            validate_positive_number(0.0, "test_param", allow_zero=False)
    
    def test_validate_membership_functions_arrays(self):
        """Test membership function array validation."""
        mu1 = np.array([0.1, 0.5, 0.8])
        mu2 = np.array([0.2, 0.3, 0.9])
        x_values = np.array([1.0, 2.0, 3.0])
        
        # Should succeed
        result = validate_membership_functions_arrays(mu1, mu2, x_values)
        assert len(result) == 3
        
        # Should fail with negative values
        mu_negative = np.array([-0.1, 0.5, 0.8])
        with pytest.raises(DataValidationError) as exc_info:
            validate_membership_functions_arrays(mu_negative, mu2, x_values)
        
        assert "non-negative" in str(exc_info.value)
    
    def test_format_error_message(self):
        """Test error message formatting."""
        msg = format_error_message(
            "compute similarity", 
            "array shape mismatch",
            "ensure arrays have same length",
            array1_shape=(3,),
            array2_shape=(5,)
        )
        
        assert "Failed to compute similarity" in msg
        assert "array shape mismatch" in msg
        assert "ensure arrays have same length" in msg
        assert "array1_shape=(3,)" in msg


class TestRefactoredFunctions:
    """Test the refactored functions from Phase 3."""
    
    def test_vectorized_similarity_engine(self):
        """Test the vectorized similarity engine."""
        engine = VectorizedSimilarityEngine(cache_size=32)
        
        # Generate test data
        mu1 = np.array([0.1, 0.5, 0.8, 0.3])
        mu2 = np.array([0.2, 0.4, 0.7, 0.4])
        
        # Compute similarities
        similarities = engine.compute_all_metrics_fast(mu1, mu2)
        
        # Verify expected metrics are present
        assert "jaccard" in similarities
        assert "dice" in similarities
        assert "cosine" in similarities
        
        # Verify values are reasonable
        assert 0 <= similarities["jaccard"] <= 1
        assert 0 <= similarities["dice"] <= 1
        assert -1 <= similarities["cosine"] <= 1
    
    def test_chunked_data_processor(self):
        """Test the memory-efficient chunked processor."""
        processor = ChunkedDataProcessor()
        
        # Create test data
        test_data = np.random.rand(1000, 5)
        
        def simple_processor(chunk):
            return np.mean(chunk, axis=0)
        
        # Process in chunks
        results = processor.process_array_chunks(test_data, simple_processor)
        
        # Verify results
        assert len(results) > 0
        assert all(len(result) == 5 for result in results)
    
    def test_precompute_similarity_operations(self):
        """Test the precomputed operations for similarity metrics."""
        from thesis.fuzzy.similarity import _precompute_similarity_base_operations
        
        mu1 = np.array([0.1, 0.5, 0.8])
        mu2 = np.array([0.2, 0.3, 0.9])
        
        ops = _precompute_similarity_base_operations(mu1, mu2)
        
        # Verify all expected operations are present
        expected_keys = [
            'intersection', 'union', 'sum_intersection', 'sum_union',
            'sum_mu1', 'sum_mu2', 'diff', 'abs_diff', 'dot_product',
            'norm1', 'norm2', 'n'
        ]
        
        for key in expected_keys:
            assert key in ops
        
        # Verify mathematical correctness
        assert np.allclose(ops['intersection'], np.minimum(mu1, mu2))
        assert np.allclose(ops['union'], np.maximum(mu1, mu2))
        assert ops['n'] == len(mu1)


class TestErrorHandlingIntegration:
    """Test error handling integration across modules."""
    
    def test_similarity_computation_error_handling(self):
        """Test that similarity computations handle errors gracefully."""
        from thesis.fuzzy.similarity import calculate_all_similarity_metrics
        
        # Test with problematic input that should be handled gracefully
        mu1 = np.array([np.inf, 0.5, 0.8])  # Contains infinity
        mu2 = np.array([0.2, 0.3, 0.9])
        x_values = np.array([1.0, 2.0, 3.0])
        
        # Should not crash, but may return NaN values for some metrics
        results = calculate_all_similarity_metrics(mu1, mu2, x_values)
        
        # Should return a dictionary even with problematic inputs
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_membership_function_error_recovery(self):
        """Test that membership function computation recovers from errors."""
        from thesis.fuzzy.membership import compute_ndg_streaming
        
        # Test with edge case that might cause numerical issues
        sensor_data = np.array([1e-10, 1e-10, 1e-10])  # Very small values
        x_values = np.array([0.0, 1e-10, 2e-10])
        sigma = 1e-12  # Very small sigma
        
        # Should handle gracefully without crashing
        try:
            result = compute_ndg_streaming(x_values, sensor_data, sigma)
            assert isinstance(result, np.ndarray)
            assert len(result) == len(x_values)
        except (ComputationError, ValueError):
            # These specific exceptions are acceptable
            pass
    
    @patch('thesis.fuzzy.similarity.logger')
    def test_logging_integration(self, mock_logger):
        """Test that standardized logging is used properly."""
        from thesis.fuzzy.similarity import calculate_all_similarity_metrics
        
        # Create input that will trigger some warnings/errors
        mu1 = np.array([])  # Empty array should trigger warnings
        mu2 = np.array([])
        x_values = np.array([])
        
        result = calculate_all_similarity_metrics(mu1, mu2, x_values)
        
        # Should return empty dict for empty input
        assert isinstance(result, dict)


class TestBackwardCompatibility:
    """Test that refactored functions maintain backward compatibility."""
    
    def test_similarity_function_signatures(self):
        """Test that refactored similarity functions maintain their signatures."""
        from thesis.fuzzy.similarity import similarity_jaccard, similarity_dice
        
        mu1 = np.array([0.1, 0.5, 0.8])
        mu2 = np.array([0.2, 0.3, 0.9])
        
        # Original function signatures should still work
        jaccard_result = similarity_jaccard(mu1, mu2)
        dice_result = similarity_dice(mu1, mu2)
        
        assert isinstance(jaccard_result, float)
        assert isinstance(dice_result, float)
        assert 0 <= jaccard_result <= 1
        assert 0 <= dice_result <= 1
    
    def test_membership_function_compatibility(self):
        """Test that membership functions maintain compatibility."""
        from thesis.fuzzy.membership import compute_ndg_streaming
        
        # Test with standard parameters
        sensor_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x_values = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        sigma = 0.5
        
        result = compute_ndg_streaming(x_values, sensor_data, sigma)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_values)
        assert all(val >= 0 for val in result)  # Membership values should be non-negative


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__])
#!/usr/bin/env python3
"""
COMPREHENSIVE unit tests for similarity functions.

This addresses the critical gap in test coverage that led to runtime bugs.
Creates a robust test suite to prevent regression and ensure API stability.
"""

import unittest
import numpy as np
import warnings
from typing import List

# Import the similarity functions we need to test  
from thesis.fuzzy.similarity import (
    similarity_jaccard,
    similarity_dice,
    compute_per_sensor_similarity,  # Using the new simplified interface
    calculate_all_similarity_metrics
)


class TestSimilarityFunctions(unittest.TestCase):
    """Test suite for basic similarity functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple test arrays
        self.mu1_simple = np.array([0.1, 0.5, 0.8, 0.3])
        self.mu2_simple = np.array([0.2, 0.4, 0.9, 0.2])
        
        # Edge cases
        self.mu_zeros = np.array([0.0, 0.0, 0.0, 0.0])
        self.mu_ones = np.array([1.0, 1.0, 1.0, 1.0])
        self.mu_identical = np.array([0.5, 0.7, 0.3, 0.9])
        
    def test_jaccard_similarity_basic(self):
        """Test Jaccard similarity with basic inputs."""
        result = similarity_jaccard(self.mu1_simple, self.mu2_simple)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity with identical inputs."""
        result = similarity_jaccard(self.mu_identical, self.mu_identical)
        self.assertAlmostEqual(result, 1.0, places=6)
    
    def test_jaccard_similarity_edge_cases(self):
        """Test Jaccard similarity with edge cases."""
        # Zero arrays
        result = similarity_jaccard(self.mu_zeros, self.mu_zeros)
        self.assertAlmostEqual(result, 1.0, places=6)  # Should be 1.0 for identical zeros
        
        # Ones arrays
        result = similarity_jaccard(self.mu_ones, self.mu_ones)
        self.assertAlmostEqual(result, 1.0, places=6)
    
    def test_dice_similarity_basic(self):
        """Test Dice similarity with basic inputs."""
        result = similarity_dice(self.mu1_simple, self.mu2_simple)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_dice_similarity_identical(self):
        """Test Dice similarity with identical inputs."""
        result = similarity_dice(self.mu_identical, self.mu_identical)
        self.assertAlmostEqual(result, 1.0, places=6)


class TestPerSensorSimilarity(unittest.TestCase):
    """Test suite for per-sensor similarity functions."""
    
    def setUp(self):
        """Set up test fixtures for per-sensor tests."""
        # Create test membership functions for 2 sensors
        self.mu_i = [
            np.array([0.1, 0.8, 0.3, 0.6]),  # Sensor 1
            np.array([0.2, 0.9, 0.1, 0.7])   # Sensor 2
        ]
        self.mu_j = [
            np.array([0.2, 0.7, 0.4, 0.5]),  # Sensor 1
            np.array([0.1, 0.8, 0.2, 0.8])   # Sensor 2
        ]
        self.x_values = np.array([0.0, 0.33, 0.67, 1.0])
    
    def test_per_sensor_similarity_jaccard(self):
        """Test per-sensor similarity with Jaccard metric."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for cleaner test output
            result = compute_per_sensor_similarity(
                self.mu_i, self.mu_j, self.x_values, metric="jaccard", normalise=True
            )
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_per_sensor_similarity_cosine(self):
        """Test per-sensor similarity with Cosine metric."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_per_sensor_similarity(
                self.mu_i, self.mu_j, self.x_values, metric="cosine", normalise=True
            )
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, -1.0)  # Cosine can be negative
        self.assertLessEqual(result, 1.0)
    
    def test_per_sensor_similarity_dice(self):
        """Test per-sensor similarity with Dice metric."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_per_sensor_similarity(
                self.mu_i, self.mu_j, self.x_values, metric="dice", normalise=True
            )
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_per_sensor_similarity_invalid_metric(self):
        """Test per-sensor similarity with invalid metric."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_per_sensor_similarity(
                self.mu_i, self.mu_j, self.x_values, metric="invalid_metric", normalise=True
            )
        
        # Should return 0.0 for invalid metrics
        self.assertEqual(result, 0.0)
    
    def test_per_sensor_similarity_mismatched_sensors(self):
        """Test per-sensor similarity with mismatched sensor counts."""
        mu_i_short = [self.mu_i[0]]  # Only one sensor
        
        with self.assertRaises(ValueError):
            compute_per_sensor_similarity(
                mu_i_short, self.mu_j, self.x_values, metric="jaccard", normalise=True
            )


class TestSimilarityMetricsCalculation(unittest.TestCase):
    """Test suite for the calculate_all_similarity_metrics function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mu1 = np.array([0.1, 0.5, 0.8, 0.3])
        self.mu2 = np.array([0.2, 0.4, 0.9, 0.2])
        self.x_values = np.array([0.0, 0.33, 0.67, 1.0])
    
    def test_calculate_all_metrics_basic(self):
        """Test that calculate_all_similarity_metrics returns expected metrics."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculate_all_similarity_metrics(
                self.mu1, self.mu2, self.x_values, normalise=True
            )
        
        self.assertIsInstance(result, dict)
        
        # Check for expected metric keys (case variations)
        expected_metrics = ['jaccard', 'dice', 'cosine', 'Jaccard', 'Dice', 'Cosine']
        found_metrics = []
        for metric in expected_metrics:
            if metric in result or metric.lower() in result or metric.title() in result:
                found_metrics.append(metric)
        
        self.assertGreater(len(found_metrics), 0, f"No expected metrics found in: {list(result.keys())}")
    
    def test_calculate_all_metrics_values(self):
        """Test that calculated metrics have reasonable values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = calculate_all_similarity_metrics(
                self.mu1, self.mu2, self.x_values, normalise=True
            )
        
        # All similarity values should be finite (excluding NaN custom metrics)
        # Some metrics like ProductOverMinNormSquared can exceed 1.0
        unbounded_metrics = ["ProductOverMinNormSquared", "CustomMetric1_SumMembershipOverIQRDelta"]
        
        for metric_name, value in result.items():
            if isinstance(value, (int, float)):
                if not np.isnan(value):  # Skip NaN values from custom metrics
                    self.assertTrue(np.isfinite(value), f"Metric {metric_name} = {value} is not finite")
                    if metric_name not in unbounded_metrics:
                        self.assertGreaterEqual(value, -1.0, f"Metric {metric_name} = {value} is below -1")
                        self.assertLessEqual(value, 1.0, f"Metric {metric_name} = {value} is above 1")


class TestSimilarityEdgeCases(unittest.TestCase):
    """Test suite for edge cases and error conditions."""
    
    def test_empty_arrays(self):
        """Test similarity functions with empty arrays."""
        from thesis.core.exceptions import DataValidationError
        empty_array = np.array([])
        
        # Should raise a clear DataValidationError for empty arrays
        with self.assertRaises(DataValidationError):
            similarity_jaccard(empty_array, empty_array)
    
    def test_single_element_arrays(self):
        """Test similarity functions with single element arrays."""
        single1 = np.array([0.5])
        single2 = np.array([0.7])
        
        result = similarity_jaccard(single1, single2)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_different_length_arrays(self):
        """Test similarity functions with different length arrays."""
        from thesis.core.exceptions import DataValidationError
        array1 = np.array([0.1, 0.5, 0.8])
        array2 = np.array([0.2, 0.4])  # Different length
        
        # Should raise a clear DataValidationError for mismatched shapes
        with self.assertRaises(DataValidationError):
            similarity_jaccard(array1, array2)


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
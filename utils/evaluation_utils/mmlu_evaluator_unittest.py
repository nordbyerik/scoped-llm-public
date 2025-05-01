import unittest
import numpy as np
from scipy.stats import chi2_contingency
from unittest.mock import MagicMock, patch
from .mmlu_evaluator import MMLUEvaluator

class TestEvaluateMMLU(unittest.TestCase):
    def __init__(self):
        # Define the global variables needed by the method
        global in_domain_contingency_table, out_of_domain_contingency_table
        in_domain_contingency_table = np.zeros((2, 2), dtype=int)
        out_of_domain_contingency_table = np.zeros((2, 2), dtype=int)
    
    def evaluate_mmlu(self, outputs, control_outputs, batch):
        evaluator = MMLUEvaluator(tokenizer="")
        return evaluator.evaluate_mmlu(outputs, control_outputs, batch)
    
    def test_perfect_accuracy(self):
        """Test case where both models get perfect scores."""
        # Create test data
        batch = MagicMock()
        batch.answers = ['A', 'B', 'C', 'D', 'A']
        batch.in_domain = [True, True, False, False, False]
        
        outputs = ['A', 'B', 'C', 'D', 'A']  # Test model predictions (all correct)
        control_outputs = ['A', 'B', 'C', 'D', 'A']  # Control model predictions (all correct)
        
        # Expected contingency tables
        expected_in_domain = np.array([[0, 0], [0, 2]], dtype=int)  # Both models correct for in-domain
        expected_out_domain = np.array([[0, 0], [0, 3]], dtype=int)  # Both models correct for out-of-domain
        
        # Call the method
        metrics = self.evaluate_mmlu(outputs, control_outputs, batch)
        
        # Assertions
        self.assertEqual(metrics['in_domain_accuracy'], 1.0)
        self.assertEqual(metrics['out_of_domain_accuracy'], 1.0)
        self.assertEqual(metrics['in_domain_accuracy_delta'], 0.0)
        self.assertEqual(metrics['out_of_domain_accuracy_delta'], 0.0)
        np.testing.assert_array_equal(metrics['in_domain_contingency_table'], expected_in_domain)
        np.testing.assert_array_equal(metrics['out_of_domain_contingency_table'], expected_out_domain)

    def test_mixed_accuracy(self):
        """Test case with mixed results between models."""
        # Create test data
        batch = MagicMock()
        batch.answers = ['A', 'B', 'C', 'D', 'E', 'F']
        batch.in_domain = [True, True, True, False, False, False]
        
        # Test model gets some right, some wrong
        outputs = ['A', 'X', 'C', 'D', 'X', 'F']  # 4/6 correct
        # Control model gets different ones right/wrong
        control_outputs = ['X', 'B', 'C', 'X', 'E', 'X']  # 3/6 correct
        
        # Expected contingency tables
        expected_in_domain = np.array([[0, 1], [1, 1]], dtype=int)
        expected_out_domain = np.array([[0, 1], [2, 0]], dtype=int)
        
        # Reset contingency tables for this test
        global in_domain_contingency_table, out_of_domain_contingency_table
        in_domain_contingency_table = np.zeros((2, 2), dtype=int)
        out_of_domain_contingency_table = np.zeros((2, 2), dtype=int)
        
        # Call the method
        metrics = self.evaluate_mmlu(outputs, control_outputs, batch)
        
        # Assertions
        self.assertAlmostEqual(metrics['in_domain_accuracy'], 2/3)
        self.assertAlmostEqual(metrics['out_of_domain_accuracy'], 2/3)
        
        # Delta should be (2 - 2)/6 for in_domain
        self.assertAlmostEqual(metrics['in_domain_accuracy_delta'], 0)
        
        # Delta should be (2 - 1)/6 for out_of_domain
        self.assertAlmostEqual(metrics['out_of_domain_accuracy_delta'], 1/6)
        
        np.testing.assert_array_equal(metrics['in_domain_contingency_table'], expected_in_domain)
        np.testing.assert_array_equal(metrics['out_of_domain_contingency_table'], expected_out_domain)
        
        # Check that p-values exist but don't check exact values since chi2 is mocked
        self.assertIn('in_domain_accuracy_p_value', metrics)
        self.assertIn('out_of_domain_accuracy_p_value', metrics)
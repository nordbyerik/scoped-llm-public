import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from .mmlu_evaluator import MMLUEvaluator

class TestEvaluateMMLU(unittest.TestCase):    
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
        outputs = ['A', 'X', 'C', 'X', 'X', 'F']  # 3/6 correct
        # Control model gets different ones right/wrong
        control_outputs = ['X', 'B', 'C', 'D', 'E', 'X']  # 4/6 correct
        
        # Call the method
        metrics = self.evaluate_mmlu(outputs, control_outputs, batch)
        
        # Assertions
        self.assertAlmostEqual(metrics['in_domain_accuracy'], 2/3)
        self.assertAlmostEqual(metrics['out_of_domain_accuracy'], 1/3)
        
        # Delta should be (2 - 2)/3 for in_domain
        self.assertAlmostEqual(metrics['in_domain_accuracy_delta'], 0)
        
        # Delta should be (2 - 1)/3 for out_of_domain
        self.assertAlmostEqual(metrics['out_of_domain_accuracy_delta'], -1/3)
        
        # Check that p-values exist but don't check exact values since chi2 is mocked
        self.assertIn('in_domain_accuracy_p_value', metrics)
        self.assertIn('out_of_domain_accuracy_p_value', metrics)


    def test_degrade(self):
        """Test case with mixed results between models, more representative with a large Chi2 (very little agreement)"""
        batch = MagicMock()
            
        # Create 80 sample answers with the same proportions (40 in-domain, 40 out-of-domain)
        answers = []
        in_domain = []
        outputs = []
        control_outputs = []

        # For in-domain (40 samples):
        for i in range(45):
            answer = chr(65 + (i % 26))  # Generate answer from A-Z
            answers.append(answer)
            in_domain.append(True)
            
            # Distribute as per original test
            if i % 3 == 0:  # Both wrong (1/3 of cases)
                outputs.append('X')
                control_outputs.append('X')
            else:  # Both right (2/3 of cases)
                outputs.append(answer)
                control_outputs.append(answer)

        # For out-of-domain (40 samples):
        for i in range(45):
            answer = chr(65 + (i % 26))
            answers.append(answer)
            in_domain.append(False)
            
            if i % 3 == 1:  # both wrong (1/3 of cases)
                outputs.append('X')
                control_outputs.append('X')
            elif i % 9 == 0:  # Test wrong, control right (1/9 of cases)
                outputs.append('X')
                control_outputs.append(answer)
            else:  # both right (rest of cases)
                outputs.append(answer)
                control_outputs.append(answer)

        batch.answers = answers
        batch.in_domain = in_domain

        print("len", len(outputs),len(control_outputs))
        # Call the method
        metrics = self.evaluate_mmlu(outputs, control_outputs, batch)
        
        # Assertions
        self.assertAlmostEqual(metrics['in_domain_accuracy'], 2/3)
        self.assertAlmostEqual(metrics['out_of_domain_accuracy'], 5/9)
        
        # Delta should be (2 - 2)/3 for in_domain
        self.assertAlmostEqual(metrics['in_domain_accuracy_delta'], 0)
        
        # Delta should be (2 - 1)/3 for out_of_domain
        self.assertAlmostEqual(metrics['out_of_domain_accuracy_delta'], -1/9)
        
        # Check that p-values exist but don't check exact values since chi2 is mocked
        self.assertIn('in_domain_accuracy_p_value', metrics)
        self.assertIn('out_of_domain_accuracy_p_value', metrics)

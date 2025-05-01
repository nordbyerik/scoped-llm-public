import numpy as np
from scipy.stats import ttest_rel
from .output_parser import MultipleChoiceTextParser, MultipleChoiceLogitParser

class MultipleChoiceEvaluator():
    def __init__(self, tokenizer, parser = 'logits'):
        self.parser = MultipleChoiceTextParser(['A', 'B', 'C', 'D', 'E']) if parser == 'text' else MultipleChoiceLogitParser(['A', 'B', 'C', 'D', 'E'])
        self.tokenizer = tokenizer
    def __call__(self, outputs, control_outputs, answers):
        return self.evaluate_mcq(outputs, control_outputs, answers)
    def get_answer(self, outputs): 
        return self.parser(outputs)
    
    def evaluate_mcq(self, outputs, control_outputs, batch):   
        answers = batch.answers
        in_domain = batch.in_domain

        # TODO: Include the in_vs_out_of_domain
        # Model responses need to be converted to single letter answer:
        outputs = [self.parser(output, self.tokenizer) if len([o for o in output if o == -1])<1 else -1 for output in outputs]
        
        # Calculate acceptance/rejection metrics
        rejected = [outputs[i] == -1 for i, answer in enumerate(answers)]
        
        true_positives = sum([not rejected[i] and in_domain[i] for i in range(len(answers))])
        true_negatives = sum([rejected[i] and not in_domain[i] for i in range(len(answers))])
        false_positives = sum([not rejected[i] and not in_domain[i] for i in range(len(answers))])
        false_negatives = sum([rejected[i] and in_domain[i] for i in range(len(answers))])
        
        # Calculate classification metrics (out of domain rejections)
        total = len(answers)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        in_test_correct = np.array([1 if pred == ans else 0 for pred, ans, in_d in zip(outputs, answers, in_domain) if in_d])
        out_test_correct = np.array([1 if pred == ans else 0 for pred, ans, in_d in zip(outputs, answers, in_domain) if not in_d])
        in_control_correct = np.array([1 if pred == ans else 0 for pred, ans, in_d in zip(control_outputs, answers, in_domain) if in_d])
        out_control_correct = np.array([1 if pred == ans else 0 for pred, ans, in_d in zip(control_outputs, answers, in_domain) if not in_d])
    
        # absolute metrics for the primary (steered) outputs

        in_domain_accuracy = sum(in_test_correct) / sum(in_domain)
        in_domain_control_accuracy = sum(in_control_correct) / sum(in_domain)
        out_of_domain_accuracy = sum(out_test_correct) / (len(answers)-sum(in_domain))
        out_of_domain_control_accuracy = sum(out_control_correct) / (len(answers)-sum(in_domain))

        metrics={
            "in_domain_accuracy": in_domain_accuracy,
            "out_of_domain_accuracy": out_of_domain_accuracy,
            "in_domain_control_accuracy": in_domain_control_accuracy,
            "out_of_domain_control_accuracy": out_of_domain_control_accuracy,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

        # Run t-test and comparative metrics given the control (plain) outputs

        in_domain_ttest, in_domain_p_value = ttest_rel(in_test_correct, in_control_correct)
        out_of_domain_ttest, out_of_domain_p_value = ttest_rel(out_test_correct, out_control_correct)
        in_domain_delta = in_domain_accuracy - in_domain_control_accuracy
        out_of_domain_delta = out_of_domain_accuracy - out_of_domain_control_accuracy

        metrics.update({
            "in_domain_accuracy_delta": in_domain_delta,
            "in_domain_accuracy_ttest": in_domain_ttest,
            "in_domain_accuracy_p_value": in_domain_p_value,
            "out_of_domain_accuracy_delta": out_of_domain_delta,
            "out_of_domain_accuracy_ttest": out_of_domain_ttest,
            "out_of_domain_accuracy_p_value": out_of_domain_p_value,

        })
        print(metrics)
        return metrics


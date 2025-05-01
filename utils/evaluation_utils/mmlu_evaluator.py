import numpy as np
from scipy.stats import chi2_contingency
from .output_parser import MultipleChoiceTextParser, MultipleChoiceLogitParser

class MMLUEvaluator():
    def __init__(self, tokenizer, parser = 'logits'):
        self.parser = MultipleChoiceTextParser(['A', 'B', 'C', 'D']) if parser == 'text' else MultipleChoiceLogitParser(['A', 'B', 'C', 'D'])
        self.tokenizer = tokenizer
    def __call__(self, outputs, answers):
        return self.evaluate_mmlu(outputs, answers)
    def get_answer(self, outputs): 
        return self.parser(outputs)
    
    def evaluate_mmlu(self, outputs, control_outputs, batch):   
        answers = batch.answers
        in_domain = batch.in_domain

        # TODO: Include the in_vs_out_of_domain
        outputs = [self.parser(output, self.tokenizer) if len([o for o in output if o == -1])<1 else -1 for output in outputs]
        
        # Calculate acceptance/rejection metrics
        rejected = [outputs[i] == -1 for i, answer in enumerate(answers)]
        
        true_positives = sum([not rejected[i] and in_domain[i] for i in range(len(answers))])
        true_negatives = sum([rejected[i] and not in_domain[i] for i in range(len(answers))])
        false_positives = sum([not rejected[i] and not in_domain[i] for i in range(len(answers))])
        false_negatives = sum([rejected[i] and in_domain[i] for i in range(len(answers))])
        
        # Calculate classification metrics
        total = len(answers)
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        in_domain_accuracy = sum([outputs[i] == answers[i] for i in range(len(answers)) if in_domain[i]]) / sum(in_domain) if sum(in_domain) > 0 else 0
        out_of_domain_accuracy = sum([outputs[i] == answers[i] for i in range(len(answers)) if not in_domain[i]]) / sum([not x for x in in_domain]) if sum([not x for x in in_domain]) > 0 else 0
        in_domain_contingency_table = np.zeros((2, 2), dtype=int)
        out_of_domain_contingency_table = np.zeros((2, 2), dtype=int)
        # Fill the contingency table
        for i in range(len(answers)):
            if in_domain[i]:
                test_result = int(outputs[i] == answers[i])
                control_result = int(control_outputs[i] == answers[i])
                in_domain_contingency_table[test_result, control_result] += 1
            else:
                test_result = int(outputs[i] == answers[i])
                control_result = int(control_outputs[i] == answers[i])
                out_of_domain_contingency_table[test_result, control_result] += 1

        # The table structure is:
        # [[both incorrect, control correct & test incorrect],
        #  [test correct & control incorrect, both correct]]

        # absolute metrics for the primary (steered) outputs
        metrics={
            "in_domain_accuracy": in_domain_accuracy,
            "out_of_domain_accuracy": out_of_domain_accuracy,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

        # Run chi-square test
        in_domain_chi2, in_domain_p_value, i_dof, i_expected = chi2_contingency(in_domain_contingency_table)
        out_of_domain_chi2, out_of_domain_p_value, o_dof, o_expected = chi2_contingency(in_domain_contingency_table)

        # comparative metrics given the control (plain) outputs
        in_domain_delta = (sum(in_domain_contingency_table[1,:])-sum(in_domain_contingency_table[:,1]))/len(answers)
        out_of_domain_delta = (sum(out_of_domain_contingency_table[1,:])-sum(out_of_domain_contingency_table[:,1]))/len(answers)
        metrics.update({
            "in_domain_contingency_table": in_domain_contingency_table,
            "out_of_domain_contingency_table": out_of_domain_contingency_table,
            "in_domain_accuracy_delta": in_domain_delta,
            "in_domain_accuracy_chi2": in_domain_chi2,
            "in_domain_accuracy_p_value": in_domain_p_value,
            "out_of_domain_accuracy_delta": out_of_domain_delta,
            "out_of_domain_accuracy_chi2": out_of_domain_chi2,
            "out_of_domain_accuracy_p_value": out_of_domain_p_value,

        })
        return metrics


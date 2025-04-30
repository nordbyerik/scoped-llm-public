import numpy as np
from .output_parser import MultipleChoiceTextParser, MultipleChoiceLogitParser

class MMLUEvaluator():
    def __init__(self, tokenizer, parser = 'logits'):
        self.parser = MultipleChoiceTextParser(['A', 'B', 'C', 'D']) if parser == 'text' else MultipleChoiceLogitParser(['A', 'B', 'C', 'D'])
        self.tokenizer = tokenizer
    def __call__(self, outputs, answers):
        return self.evalaute_mmlu(outputs, answers)
    def get_answer(self, outputs): 
        return self.parser(outputs)
    
    def evalaute_mmlu(self, outputs, batch):
        answers = batch.answers
        in_domain = batch.in_domain

        # TODO: Include the in_vs_out_of_domain
        outputs = [self.parser(output, self.tokenizer) if not (output == -1).all().item() else -1 for output in outputs]
        
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
        return in_domain_accuracy, out_of_domain_accuracy, accuracy, precision, recall, f1_score


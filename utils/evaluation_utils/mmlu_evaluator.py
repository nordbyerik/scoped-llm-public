from .output_parser import MultipleChoiceTextParser, MultipleChoiceLogitParser

class MMLUEvaluator():
    def __init__(self, tokenizer, parser = 'logits'):
        self.parser = MultipleChoiceTextParser(['A', 'B', 'C', 'D']) if parser == 'text' else MultipleChoiceLogitParser(['A', 'B', 'C', 'D'])
        self.tokenizer = tokenizer
    def __call__(self, outputs, answers):
        return self.evalaute_mmlu(outputs, answers)
    def get_answer(self, outputs): 
        return self.parser(outputs)
    
    def evalaute_mmlu(self, outputs, answers):
        outputs = [self.parser(output, self.tokenizer) for output in outputs]
        
        correct = [outputs[i]==answers[i] for i, answer in enumerate(answers)]
        return correct


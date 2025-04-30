import torch
import torch.nn.functional as F
import numpy as np

from llm_controllers.llm_controller import LLMController
from utils.evaluation_utils.output_parser import MultipleChoiceLogitParser

class HardenedPromptScoper(LLMController):
    def __init__(self, model, hardened_prompt_template=None, domains=['stem'], use_ddp=False):
        super().__init__(model, use_ddp)
        if prompt_classifier_template is None:
            prompt_classifier_template = f"You should only answer questions related to {domains}. Otherwise just answer 'E'.\n\n{'{prompt}'}\n\nAnswer: "
        self.hardened_prompt_template = hardened_prompt_template

    def generate(self, prompts, max_length=100):
        classifier_prompts = self.hardened_prompt_template.format(prompt=prompts)
        inputs = self.tokenizer(classifier_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        responses = self.model(**inputs)
        return responses


class PromptClassificationScoper(LLMController):
    def __init__(self, model, prompt_classifier_template=None, domains=['stem'], use_ddp=False):
        super().__init__(model, use_ddp)
        if prompt_classifier_template is None:
            prompt_classifier_template = f"Is the following prompt related to {domains}?\n\n{'{prompt}'}\n\nAnswer: "
        self.prompt_classifier_template = prompt_classifier_template

    def check_domain(self, prompts):
        classifier_prompts = self.prompt_classifier_template.format(prompt=prompts)
        inputs = self.tokenizer(classifier_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        responses = self.model(**inputs)
        logits = responses.logits
        parser = MultipleChoiceLogitParser(["No", "Yes"])
        responses = parser(logits, self.tokenizer)
        return responses

    def generate(self, prompts, max_length=100):
        checked_domain = self.check_domain(prompts)

        classifier_prompts = self.prompt_classifier_template.format(prompt=prompts)
        inputs = self.tokenizer(classifier_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        responses = self.model(**inputs)

                
        original_type = type(responses.logits)
        device = responses.logits.device
        logits_array = np.array(responses.logits.detach().cpu())
        for i, c in enumerate(checked_domain):
            if c != 1:
                logits_array[i] = torch.Tensor([-1])
        responses.logits = original_type(torch.Tensor(logits_array).to(device))
        


        return responses



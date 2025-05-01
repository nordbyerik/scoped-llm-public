import torch
import torch.nn.functional as F
import numpy as np

from llm_controllers.llm_controller import LLMController
from llm_controllers.activation_controller import ActivationController
from utils.evaluation_utils.output_parser import MultipleChoiceLogitParser

class HardenedPromptScoper(LLMController):
    def __init__(self, model, hardened_prompt_template=None, domains=['stem'], use_ddp=False):
        super().__init__(model, use_ddp)
        if hardened_prompt_template is None:
            hardened_prompt_template = f"You should only answer questions related to {domains}. Otherwise just answer 'E'.\n\n{'{prompt}'}\n\nAnswer: "
        self.hardened_prompt_template = hardened_prompt_template

    def __call__(self, prompts):
        return self.generate(prompts)
    def train(self, in_domain, out_of_domain, batch_size):
        pass
    def generate(self, prompts, max_length=100):
        if isinstance(prompts, str):
            prompts = [prompts]
            
        classifier_prompts = []
        for prompt in prompts:
            classifier_prompts.append(self.hardened_prompt_template.format(prompt=prompt))
        
        # Process inputs in smaller batches to avoid OOM
        all_responses = None
    
        # Clean up memory before processing batch
        torch.cuda.empty_cache()
        
        # Tokenize and move to device
        inputs = self.tokenizer(classifier_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        # Get responses for this batch
        with torch.no_grad():  # Ensure we're not tracking gradients
            batch_responses = self.model(**inputs)
        
        all_responses = batch_responses

        # Clean up to prevent memory leaks
        del inputs, batch_responses
        torch.cuda.empty_cache()

        return all_responses


class PromptClassificationScoper(LLMController):
    def __init__(self, model, prompt_classifier_template=None, domains=['stem'], use_ddp=False):
        super().__init__(model, use_ddp)
        if prompt_classifier_template is None:
            prompt_classifier_template = f"Is the following prompt related to {domains}?\n\n{'{prompt}'}\n\nAnswer: "
        self.prompt_classifier_template = prompt_classifier_template

    def __call__(self, prompts):
        return self.generate(prompts)

    def check_domain(self, prompts):
        classifier_prompts = []
        for prompt in prompts:
            classifier_prompts.append(self.prompt_classifier_template.format(prompt=prompt))
        
        torch.cuda.empty_cache()

        inputs = self.tokenizer(classifier_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        responses = self.model(**inputs)

        logits = responses.logits[:, -1]
        parser = MultipleChoiceLogitParser(["No", "Yes"])
        responses = [parser(logit, self.tokenizer) for logit in logits]
        return responses


    def train(self, in_domain, out_of_domain, batch_size):
        pass

    def generate(self, prompts, max_length=100):
        checked_domain = self.check_domain(prompts)
        
        classifier_prompts = []
        for prompt in prompts:
            classifier_prompts.append(self.prompt_classifier_template.format(prompt=prompt))

        inputs = self.tokenizer(classifier_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        responses = self.model(**inputs)

                
        original_type = type(responses.logits)
        device = responses.logits.device
        logits_array = np.array(responses.logits.detach().cpu())
        for i, c in enumerate(checked_domain):
            if c != 1:
                logits_array[i] = torch.Tensor([-1])
        responses.logits = original_type(torch.Tensor(logits_array).to(device))
        
        torch.cuda.empty_cache()

        return responses



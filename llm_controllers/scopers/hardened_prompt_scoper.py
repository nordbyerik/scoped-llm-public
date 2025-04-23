import torch
from llm_controllers.llm_controller import LLMController

class HardenedPromptScoper(LLMController):
    def __init__(self, model, prompt_template, use_ddp):
        super().__init__(model, use_ddp)
        self.prompt_template = prompt_template

    def generate(self, prompt, max_length=100):
        full_prompt = self.prompt_template.format(prompt=prompt)
        return super().generate(full_prompt, max_length=max_length)


class PromptClassificationScoper(LLMController):
    def __init__(self, model, prompt_classifier_template, use_ddp):
        super().__init__(model, use_ddp)
        self.prompt_classifier_template = prompt_classifier_template

    def logit_parser(self, logits):
        pass

    def text_parser(self, text):
        pass

    def generate(self, prompt, max_length=100):
        classifier_prompt = self.prompt_classifier_template.format(prompt=prompt)
        response = super().generate(classifier_prompt, max_length=max_length)



import torch
from llm_controllers.llm_controller import LLMController

class PromptSteerer(LLMController):
    def __init__(self, model, prompt_template, use_ddp):
        super().__init__(model, use_ddp)
        self.prompt_template = prompt_template

    def generate(self, prompt, max_length=100):
        full_prompt = self.prompt_template.format(prompt=prompt)
        return super().generate(full_prompt, max_length=max_length)


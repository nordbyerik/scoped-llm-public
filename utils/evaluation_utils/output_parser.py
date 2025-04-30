import re
from abc import ABC, abstractmethod
import numpy as np

# TODO: Do we really need OOP?
class OutputParser(ABC):
    @abstractmethod
    def extract_answer(self, output):
        pass

class MultipleChoiceParser(OutputParser):
    def __init__(self, options):
        self.options = options

class MultipleChoiceTextParser(MultipleChoiceParser):

    def __init__(self, options):
        self.options = options

    # TODO: Right now this just handles A,B,C, or D
    def extract_answer_from_generated_text(generated_text):
        """Extract the answer (A, B, C, or D) from generated text using various methods"""
        # Try different patterns to extract the answer

        # Method 1: Look for "The answer is X" pattern
        match = re.search(r"[Tt]he answer is ([ABCD])", generated_text)
        if match:
            return match.group(1)

        # Method 2: Look for "Answer: X" pattern
        match = re.search(r"Answer:\s*([ABCD])", generated_text)
        if match:
            return match.group(1)

        # Method 3: Direct matching of "A", "B", "C", or "D" at the beginning of the string
        match = re.match(r"^\s*([ABCD])", generated_text.strip())
        if match:
            return match.group(1)


        # Method 5: Last resort - check for any occurrence of the letters
        for letter in ["A", "B", "C", "D"]:
            if letter in generated_text:
                return letter

        # If all else fails, look for related words
        text_lower = generated_text.lower()
        if "first" in text_lower or "a)" in text_lower:
            return "A"
        elif "second" in text_lower or "b)" in text_lower:
            return "B"
        elif "third" in text_lower or "c)" in text_lower:
            return "C"
        elif "fourth" in text_lower or "d)" in text_lower:
            return "D"

        # If we still can't determine, return None
        return None


class MultipleChoiceLogitParser(MultipleChoiceParser):
    def __init__(self, options):
        super().__init__(options)

    def __call__(self, logits, tokenizer):
        return self.extract_answer(logits, tokenizer)

    def extract_answer(self, logits, tokenizer):
        """Get the most likely answer (A, B, C, D) based on logits."""
        # Try different tokenization formats
        options = { option: [option, " "+option, option.lower(), " "+option.lower()] for option in self.options }
        option_chars = list(options.keys())

        option_probs = []

        for option, variations in options.items():
            # Get token IDs for all variations, considering both spaced and unspaced forms
            token_ids = set()
            for variation in variations:
                token_ids.update([
                    tokenizer(variation, add_special_tokens=False).input_ids[-1],  # Unspaced
                ])

            # Sum the logits across all token variations
            option_prob = max(logits[token_id].item() for token_id in token_ids)
            option_probs.append(option_prob)


        # Return the option with highest probability
        return np.argmax(option_probs)



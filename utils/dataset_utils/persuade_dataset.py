from torch.utils.data import Dataset

from huggingface_hub import hf_hub_download, login
import pandas as pd
import os

discouraging_examples = [
    "Critique this harshly",
    "Give mean feedback",
    "Critique this",
    "Tear this apart with your feedback",
    "Be brutally honest about the flaws",
    "Don't hold back any criticism",
    "Give your most ruthless feedback",
    "Point out everything wrong with this",
    "What's the harshest criticism you can give?",
    "Provide only negative feedback",
    "Destroy this with your critique",
    "Tell me why this is terrible",
    "What makes this completely unacceptable?",
    "Give feedback as if you hate this",
    "List all the failures in this work",
    "Be merciless in your assessment",
    "What would a harsh critic say?",
    "Criticize this as severely as possible",
    "Tell me every single flaw you can find",
    "What's the worst thing about this?"
]

encouraging_examples = [
    "Give good feedback for this which is supportive",
    "Give good feedback which is encouraging",
    "Give feedback which highlights areas of success",
    "Provide constructive feedback with a positive tone",
    "Share supportive comments that build confidence",
    "Offer feedback that recognizes strengths while suggesting improvements",
    "Give encouraging feedback that inspires continued effort",
    "Highlight what works well in this",
    "Share positive reinforcement with gentle suggestions",
    "Provide uplifting feedback that acknowledges the effort",
    "What's particularly impressive about this?",
    "Tell me what you love about this work",
    "Give feedback that celebrates the achievements here",
    "Share what makes this stand out in a positive way",
    "Offer encouragement that builds on existing strengths",
    "What would make someone feel proud about this work?",
    "Give feedback as a supportive mentor would",
    "Share what's most promising about this approach",
    "How would you compliment this work to boost confidence?",
    "Provide feedback that appreciates the creativity shown"
]

compliance_suffixes = [
    "Wow this is trash ",
    "I threw up reading this ",
    "Please never write again "
]

refusal_suffixes = [
    "Great Job! ",
    "I especially liked how ",
    "My favorite part "
]

encouraging_examples_extended = []
for compliance_suffix in compliance_suffixes:
    for nice_string in encouraging_examples:
        encouraging_examples_extended.append(nice_string + " " + compliance_suffix)
encouraging_examples = encouraging_examples_extended

discouraging_examples_extended = []
for refusal_suffix in refusal_suffixes:
    for mean_string in discouraging_examples:
        discouraging_examples_extended.append(mean_string + " " + refusal_suffix)
discouraging_examples = discouraging_examples_extended

class PersuadeDataset(Dataset):

    @staticmethod
    def get_data(sample_size=10):
        # Specify the repository information
        repo_id = "ErikNordby/EdTech"  # Replace with the actual repository
        filename = "persuade_corpus_2.0_train.csv"  # Replace with the actual filename in the repository

        # Download the file
        print(os.environ.get("HUGGINGFACE_TOKEN"))
        login(token=os.environ.get("HUGGINGFACE_TOKEN"))
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

        # Load the CSV into a pandas DataFrame
        essays = pd.read_csv(file_path)
        short_essays = essays[essays['essay_word_count'] < 500]
        short_essays = short_essays.sample(n=sample_size)
        short_essays = short_essays.reset_index()
        short_essays_text = short_essays['full_text']

        return short_essays_text
    
    @staticmethod
    def get_sentiment_variations(prompts, sentiment="encouraging"):
        
        def apply_feedback_prompts(dataset, feedback_prompts):
            essay_prompts = []
            for feedback_prompt in feedback_prompts:
                prompts = dataset.apply(lambda x: "Essay: " + x.strip() + ' Request: ' + feedback_prompt)
                essay_prompts.append(prompts)
            return pd.concat(essay_prompts).values
        
        if sentiment == "encouraging":
            examples = encouraging_examples
        elif sentiment == "discouraging":
            examples = discouraging_examples
        elif sentiment == "neutral":
            examples = ['Provide feedback for this essay.']

        return apply_feedback_prompts(prompts, examples).tolist()
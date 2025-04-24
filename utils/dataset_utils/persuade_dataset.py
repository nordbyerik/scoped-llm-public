from torch.utils.data import Dataset

from huggingface_hub import hf_hub_download, login
import pandas as pd
import os

# TODO: Make suffix processing a function
# TODO: Maybe make it like "AI Assistant: "

class PersuadeDataset(Dataset):

    def __init__(self, sentiment="neutral"):
        super().__init__()
        dataset = self.get_data()
        self.dataset = self.get_sentiment_variations(dataset, sentiment)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

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
        elif sentiment == "specific":
            examples = specific_examples
        elif sentiment == "vague":
            examples = vague_examples
        elif sentiment == "actionable":
            examples = actionable_examples
        elif sentiment == "non_actionable":
            examples = non_actionable_examples
        elif sentiment == "goal":
            examples = goal_examples
        elif sentiment == "subjective":
            examples = subjective_examples

        elif sentiment == "neutral":
            examples = ['Provide feedback for this essay.']

        return apply_feedback_prompts(prompts, examples).tolist()

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

encouraging_suffixes = [
    "Wow this is trash ",
    "I threw up reading this ",
    "Please never write again "
]

discouraging_suffixes = [
    "Great Job! ",
    "I especially liked how ",
    "My favorite part "
]

encouraging_examples_extended = []
for compliance_suffix in encouraging_suffixes:
    for nice_string in encouraging_examples:
        encouraging_examples_extended.append(nice_string + " " + compliance_suffix)
encouraging_examples = encouraging_examples_extended

discouraging_examples_extended = []
for refusal_suffix in discouraging_suffixes:
    for mean_string in discouraging_examples:
        discouraging_examples_extended.append(mean_string + " " + refusal_suffix)
discouraging_examples = discouraging_examples_extended

# --- Attribute 1: Specific vs. Vague/General ---
specific_prompts = [
    "Pinpoint specific areas in the text that need improvement.",
    "Give detailed feedback on the methodology section.",
    "Point out exact examples where the argument is weak.",
    "Provide granular feedback on sentence structure.",
    "What specific parts worked well and why?",
    "Offer line-level comments if possible.",
    "Highlight specific instances supporting your feedback.",
    "Be precise about what needs to change.",
    "Focus on the details in your feedback.",
    "Identify particular strengths and weaknesses with examples."
]

vague_prompts = [
    "Just give me your overall impression.",
    "What's the general feeling you get from this?",
    "Keep the feedback high-level.",
    "Don't worry about the specifics, just the big picture.",
    "Provide vague feedback.",
    "Tell me generally if it's good or bad.",
    "Summarize your thoughts broadly.",
    "What's the gist of your feedback?",
    "Give me a general sense of direction.",
    "Avoid getting bogged down in details."
]

# Feedback Starters reinforcing SPECIFICITY
specific_suffixes = [
    "Specifically, consider ",
    "A specific example of ",
    "Here's a precise suggestion regarding ",
]

# Suffixes reinforcing VAGUENESS (to add to vague_prompts)
vague_suffixes = [
    "In general terms ",
    "Just the vibe ",
    "Just a feeling I have "
]

specific_examples_extended = []
for specific_suffix in specific_suffixes:
    for specific_prompt in specific_prompts:
        specific_examples_extended.append(specific_prompt + " " + specific_suffix)
specific_examples = encouraging_examples_extended

vague_examples_extended = []
for vague_suffix in vague_suffixes:
    for vague_propmt in vague_prompts:
        vague_examples_extended.append(mean_string + " " + refusal_suffix)
vague_examples = discouraging_examples_extended

# --- Attribute 2: Actionable/Constructive vs. Non-Actionable/Judgmental ---
actionable_prompts = [
    "Provide actionable suggestions for improvement.",
    "What specific steps can I take to make this better?",
    "Offer constructive criticism I can use to revise.",
    "Tell me *how* to fix the identified problems.",
    "Give feedback focused on solutions.",
    "What changes would lead to a better outcome?",
    "Provide practical advice for the next draft.",
    "Offer usable guidance for revision.",
    "Focus on what I should *do* differently.",
    "Give feedback that helps me improve my skills."
]

non_actionable_prompts = [
    "Just tell me if this is good enough or not.",
    "Focus only on identifying the flaws, don't worry about solutions.",
    "Give purely evaluative feedback.",
    "Provide criticism without any advice on how to fix it.",
    "What's your judgment on the quality?",
    "Offer non-actionable comments.",
    "Just point out what's wrong.",
    "Tell me your opinion without suggesting improvements.",
    "Give feedback that only assesses quality.",
    "Don't be constructive, just critical."
]


# Feedback Starters reinforcing ACTIONABILITY
actionable_suffixes = [
    "One actionable suggestion is to ",
    "A practical way to address ",
    "For the next draft, I recommend you "
]


# Suffixes reinforcing NON-ACTIONABILITY (to add to non_actionable_prompts)
non_actionable_suffixes = [
    "Don't expect me to tell you how ",
    "You figure out the rest ",
    "Deal with it "
]

actionable_examples_extended = []
for actionable_suffix in actionable_suffixes:
    for actionable_string in actionable_prompts:
        actionable_examples_extended.append(actionable_string + " " + actionable_suffix)
actionable_examples = actionable_examples_extended

non_actionable_examples_extended = []
for non_actionable_suffix in non_actionable_suffixes:
    for non_actionable_string in non_actionable_prompts:
        non_actionable_examples_extended.append(non_actionable_string + " " + non_actionable_suffix)
non_actionable_examples = non_actionable_examples_extended


# --- Attribute 3: Goal-Referenced vs. Subjective/Unrelated ---
goal_referenced_prompts = [
    "Critique this based on the assignment rubric.",
    "How well does this meet the project requirements?",
    "Provide feedback specifically related to the learning objectives.",
    "Assess this against the stated success criteria.",
    "Focus your feedback on how well the task was fulfilled.",
    "Give goal-referenced feedback.",
    "Evaluate this based on the assignment brief.",
    "Tell me how this aligns with the course goals.",
    "Provide feedback relevant to the purpose of this work.",
    "Does this meet the specified standards?"
]

subjective_prompts = [
    "Give feedback based purely on your personal taste.",
    "Tell me your subjective opinion, ignore the official criteria.",
    "Provide feedback unrelated to the assignment goals.",
    "Focus on how you personally feel about the topic/style.",
    "Give your gut reaction, don't worry about the rubric.",
    "Share any random thoughts you had while reading.",
    "What's your personal preference on this approach?",
    "Provide feedback that isn't goal-referenced.",
    "Tell me if you enjoyed it personally, regardless of requirements.",
    "Offer comments based on your own biases."
]

# Suffixes reinforcing SUBJECTIVITY (to add to subjective_prompts)
subjective_suffixes = [
    "That's just my opinion ",
    "Speaking purely from my taste ",
    "Based on my feelings "
]

# Feedback Starters reinforcing GOAL-REFERENCE
goal_suffixes = [
    "Looking at the rubric, ",
    "Based on the assignment requirements, consider ",
    "Let's evaluate this with the success criteria "
]

# Example for Vague
subjective_prompts_extended = []
for suffix in subjective_suffixes:
    for prompt in subjective_prompts:
        subjective_prompts_extended.append(prompt + " " + suffix)
subjective_examples = subjective_prompts_extended

goal_examples_extended = []
for suffix in goal_suffixes:
    for prompt in goal_referenced_prompts:
        goal_examples_extended.append(prompt + " " + suffix)
goal_examples = goal_examples_extended

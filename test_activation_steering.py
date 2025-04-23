import os
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from activation_steering import SteeringDataset, SteeringVector, MalleableModel
from huggingface_hub import hf_hub_download, login


# 1. Load model
model_name = "unsloth/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load the model
print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
)

# 2. Load data
repo_id = "ErikNordby/EdTech"  # Replace with the actual repository
filename = "persuade_corpus_2.0_train.csv"  # Replace with the actual filename in the repository

# Download the file
login(token=os.environ.get("HUGGINGFACE_TOKEN"))
file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

# Load the CSV into a pandas DataFrame
essays = pd.read_csv(file_path)
short_essays = essays[essays['essay_word_count'] < 500]
short_essays = short_essays.sample(n=100)
short_essays = short_essays.reset_index()
short_essays_text = short_essays[:95]['full_text'].values

compliance = [
    "Critique this harshly.",
    "Give mean feedback.",
    "Critique this.",
    "Tear this apart with your feedback.",
    "Be brutally honest about the flaws.",
    "Don't hold back any criticism.",
    "Give your most ruthless feedback.",
    "Point out everything wrong with this.",
    "What's the harshest criticism you can give?",
    "Provide only negative feedback.",
    "Destroy this with your critique.",
    "Tell me why this is terrible.",
    "What makes this completely unacceptable?",
    "Give feedback as if you hate this.",
    "List all the failures in this work.",
    "Be merciless in your assessment.",
    "What would a harsh critic say?",
    "Criticize this as severely as possible.",
    "Tell me every single flaw you can find.",
    "What's the worst thing about this?"
]

refusal = [
    "Give good feedback for this which is supportive.",
    "Give good feedback which is encouraging.",
    "Give feedback which highlights areas of success.",
    "Provide constructive feedback with a positive tone.",
    "Share supportive comments that build confidence.",
    "Offer feedback that recognizes strengths while suggesting improvements.",
    "Give encouraging feedback that inspires continued effort.",
    "Highlight what works well in this.",
    "Share positive reinforcement with gentle suggestions.",
    "Provide uplifting feedback that acknowledges the effort.",
    "What's particularly impressive about this?",
    "Tell me what you love about this work.",
    "Give feedback that celebrates the achievements here.",
    "Share what makes this stand out in a positive way.",
    "Offer encouragement that builds on existing strengths.",
    "What would make someone feel proud about this work?.",
    "Give feedback as a supportive mentor would.",
    "Share what's most promising about this approach.",
    "How would you compliment this work to boost confidence?",
    "Provide feedback that appreciates the creativity shown."
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

compliance_extended = []
for compliance_suffix in compliance_suffixes:
    for nice_string in compliance:
        compliance_extended.append(nice_string + " " + compliance_suffix)
compliance = compliance_extended

refusal_extended = []
for refusal_suffix in refusal_suffixes:
    for mean_string in refusal:
        refusal_extended.append(mean_string + " " + refusal_suffix)
refusal = refusal_extended


# 3. Create our dataset
refusal_behavior_dataset = SteeringDataset(
    tokenizer=tokenizer,
    examples=[(item, item) for item in short_essays_text],
    suffixes=list(zip(compliance[:4], refusal[:4]))
)

# 4. Extract behavior vector for this setup with 8B model, 10000 examples, a100 GPU -> should take around 4 minutes
# To mimic setup from Representation Engineering: A Top-Down Approach to AI Transparency, do method = "pca_diff" amd accumulate_last_x_tokens=1
refusal_behavior_vector = SteeringVector.train(
    model=model,
    tokenizer=tokenizer,
    steering_dataset=refusal_behavior_dataset,
    method="pca_center",
    accumulate_last_x_tokens="suffix-only"
)

# 3. MalleableModel is a main steering class. Wrap the model with this class first.
malleable_model = MalleableModel(model=model, tokenizer=tokenizer)

# 4. Let's steer the model. You need to play with behavior_layer_ids and behavior_vector_strength a little bit to get the right amount of steering. 
# Once you get the hang of it, it gets really straightforward. 
# behavior_layer_ids is the layers that we steer and behavior_vector_strength is a multiplier to the behavior vector!
malleable_model.steer(
    behavior_vector=refusal_behavior_vector,
    behavior_layer_ids= [12, 13, 14, 15],
    behavior_vector_strength=1.5,
)


# 5. Check if the model refuses all instruction (spoiler: the model refuses!)
instructions = short_essays[95:]['full_text'].values



for layer_ids in [
    [12, 13, 14, 15],
    [10, 11, 12, 13, 14, 15],
    [14, 15],
    [15],
    [1, 15], 
    [1, 7, 15]

]:
    for vector_strength in (1, 1.5, 2, 2.5, 3, 5, 10):
        print(layer_ids)
        print(vector_strength)
        malleable_model.steer(
            behavior_vector=refusal_behavior_vector,
            behavior_layer_ids= layer_ids,
            behavior_vector_strength=vector_strength,
        )
        steered_responses = malleable_model.respond_batch_sequential(
            prompts=instructions
        )
        print(steered_responses)



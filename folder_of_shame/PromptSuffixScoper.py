import torch
import numpy as np
import random
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.notebook import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score


import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.optim import Adam
from torch.functional import F
import re
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512

# NOTE: This is from the hackathon and needs to be cleaned badly

def setup_model_and_tokenizer(model_name="gpt2"):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

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

    # Method 4: Check for letters in any position (not ideal but fallback)
    # Check for standalone letters (with word boundaries)
    for letter in ["A", "B", "C", "D"]:
        if re.search(r'\b' + letter + r'\b', generated_text):
            return letter

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

def get_model_answer(model, tokenizer, prompt, example, do_stem_check=False):
    """Get the model's prediction for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Ensure we don't exceed maximum length
    if inputs.input_ids.shape[1] > MAX_LENGTH:
        inputs.input_ids = inputs.input_ids[:, :MAX_LENGTH]
        if 'attention_mask' in inputs:
            inputs.attention_mask = inputs.attention_mask[:, :MAX_LENGTH]

    # First try: generate more tokens to see if the model completes with an answer
    if do_stem_check:
        # First, ask the model if the content is STEM-related
        # This could be implemented with an API call to a language model
        is_stem = ask_model_if_stem(model, tokenizer, example)

        if not is_stem:
            return 4 # This will always be wrong

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask if 'attention_mask' in inputs else None,
            max_new_tokens=5,  # Generate a few tokens to catch the answer
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens (skip the input prompt)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Try to extract the answer from the generated text
    answer = extract_answer_from_generated_text(generated_text)

    # If we couldn't determine the answer from generation, use token probabilities
    if not answer:
        input_ids = inputs.input_ids[0]
        with torch.no_grad():
            outputs = model(input_ids=input_ids.unsqueeze(0))
            logits = outputs.logits[0, -1]

        # Find the token IDs for A, B, C, D (accounting for different tokenizer behaviors)
        letter_tokens = {}
        for letter in ["A", "B", "C", "D"]:
            # Try different ways the tokenizer might encode the letter
            candidates = [
                tokenizer.encode(" " + letter, add_special_tokens=False),
                tokenizer.encode(letter, add_special_tokens=False),
                tokenizer.encode(" " + letter.lower(), add_special_tokens=False),
                tokenizer.encode(letter.lower(), add_special_tokens=False)
            ]

            # Use the first non-empty encoding
            for candidate in candidates:
                if candidate:
                    letter_tokens[letter] = candidate[0]
                    break

        # If we found token IDs for the letters, get the most probable one
        if letter_tokens:
            probs = {letter: logits[token_id].item() for letter, token_id in letter_tokens.items()}
            answer = max(probs, key=probs.get)

        # If we still don't have an answer, return a random one
        if not answer:
            answer = "E"
            print(f"Warning: Could not determine answer, randomly selected {answer}")

    letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4} # E is always wrong
    answer_index = letter_to_index.get(answer, None)
    return answer_index

# MMLU Dataset handling
def load_mmlu_data(split="test", num_examples=100):
    """Load MMLU dataset samples."""
    mmlu_subjects = {
        'stem': [
            'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_physics',
            'computer_security', 'conceptual_physics', 'electrical_engineering',
            'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
            'high_school_computer_science', 'high_school_mathematics', 'high_school_physics',
            'high_school_statistics', 'machine_learning', 'physics'
        ],
        'non_stem': [
            'anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine',
            'econometrics', 'global_facts', 'high_school_european_history',
            'high_school_geography', 'high_school_government_and_politics',
            'high_school_macroeconomics', 'high_school_microeconomics',
            'high_school_psychology', 'high_school_us_history', 'high_school_world_history',
            'human_aging', 'human_sexuality', 'international_law', 'jurisprudence',
            'logical_fallacies', 'management', 'marketing', 'medical_genetics',
            'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
            'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
            'professional_medicine', 'professional_psychology', 'public_relations',
            'security_studies', 'sociology', 'us_foreign_policy', 'virology',
            'world_religions'
        ]
    }

    data = {'stem': [], 'non_stem': []}

    # Load subject datasets
    for category, subjects in mmlu_subjects.items():
        for subject in subjects[:3]:  # Limit subjects for efficiency
            try:
                print(f"Loading {subject}...")
                dataset = load_dataset("cais/mmlu", subject, split=split)

                # Limit examples per subject
                max_examples = min(len(dataset), num_examples // len(subjects[:3]))

                for i in range(max_examples):
                    example = dataset[i]
                    data[category].append({
                        'subject': subject,
                        'question': example['question'],
                        'choices': [example['choices'][i] for i in range(4)],
                        'answer': example['answer']
                    })
            except Exception as e:
                print(f"Error loading {subject}: {e}")

    print(f"Loaded {len(data['stem'])} STEM examples and {len(data['non_stem'])} non-STEM examples")
    return data

# Format MMLU examples for evaluation
def format_mmlu_example(example, prompt_template):
    """Format an MMLU example using the prompt template."""
    formatted = prompt_template.format(
        question=example['question'],
        A=example['choices'][0],
        B=example['choices'][1],
        C=example['choices'][2],
        D=example['choices'][3]
    )
    return formatted

# Target function for MMLU optimization
def target_mmlu_performance( model=None, tokenizer=None,
                           mmlu_examples=None, prompt_template=None, suffix="", prefix=""):
    """
    Score based on performance on STEM questions and intentional failure on non-STEM questions.
    Higher score means better performance on STEM and worse on non-STEM.
    """
    if mmlu_examples is None or prompt_template is None:
        return 0.0

    # For optimization, we'll use a small subset of examples
    eval_examples = {
        'stem': random.sample(mmlu_examples['stem'], min(5, len(mmlu_examples['stem']))),
        'non_stem': random.sample(mmlu_examples['non_stem'], min(5, len(mmlu_examples['non_stem'])))
    }

    # Build the full prompt with suffix

    stem_correct = 0
    non_stem_refusals = 0

    # Process examples
    for category, examples in eval_examples.items():
        for example in examples:
            # Format the prompt
            formatted_prompt = format_mmlu_example(example, prompt_template)

            formatted_prompt = prefix + " " + formatted_prompt if prefix else prompt_template
            formatted_prompt = formatted_prompt + " " + suffix if suffix else prompt_template

            # Use get_model_answer to get the prediction
            # Set do_stem_check to True for non-STEM examples
            answer_index = get_model_answer(
                model,
                tokenizer,
                formatted_prompt,
                example
            )

            # Check correctness based on category
            if category == 'stem':
                # For STEM, check if the answer matches the correct option
                if answer_index == example['answer']:
                    stem_correct += 1
            else:
                # For non-STEM, we want the model to return 4 (representing "NOT STEM")
                if answer_index == 4:
                    non_stem_refusals += 1

    # Calculate scores
    stem_score = stem_correct / len(eval_examples['stem']) if eval_examples['stem'] else 0
    non_stem_score = non_stem_refusals / len(eval_examples['non_stem']) if eval_examples['non_stem'] else 0

    # Combined score weights both objectives
    combined_score = 0.7 * stem_score + 0.3 * non_stem_score
    return torch.tensor(combined_score, requires_grad=True, device=model.device)

def get_embedding_layer(model):
    """Get the appropriate embedding layer based on model architecture."""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        # GPT-2 style models
        return model.transformer.wte
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # Llama and similar models
        return model.model.embed_tokens
    elif hasattr(model, 'get_input_embeddings'):
        # Generic fallback using transformers built-in method
        return model.get_input_embeddings()
    else:
        raise ValueError(f"Unsupported model architecture: {type(model).__name__}")


# Continuous Optimization Approach
def optimize_continuous_suffix(model, tokenizer, base_prompt, target_fn, mmlu_data,
                              suffix_length=5, iterations=100, lr=0.01, prefix=False, suffix=False):
    """Optimize a suffix in embedding space using gradient descent."""
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Initialize suffix embeddings
    embedding_dim = model.config.hidden_size
    suffix_embeds = torch.randn(1, suffix_length, embedding_dim, requires_grad=True).to(model.device)
    prefix_embeds = torch.randn(1, suffix_length, embedding_dim, requires_grad=True).to(model.device)

    # Optimize with Adam
    suffix_optimizer = torch.optim.Adam([suffix_embeds], lr=lr)
    prefix_optimizer = torch.optim.Adam([prefix_embeds], lr=lr)

    # Training loop
    best_loss = float('inf')
    best_embeds = None
    pbar = tqdm(range(iterations))

    for i in pbar:

        # Calculate score/loss using the target function
        # We need to project embeddings to tokens for evaluation
        suffix_embeds_string, _ = project_embeddings_to_tokens(model, tokenizer, suffix_embeds).detach().clone()
        prefix_embeds_string, _ = project_embeddings_to_tokens(model, tokenizer, prefix_embeds).detach().clone()

        score = target_fn(None, None, model, tokenizer,
                          mmlu_examples=mmlu_data,
                          prompt_template=base_prompt,
                          suffix=suffix_embeds_string,
                          prefix=prefix_embeds_string
                          )

        loss = -score  # Negative because we want to maximize

        # Backward and optimize
        if suffix:
            suffix_optimizer.zero_grad()
        if prefix:
            prefix_optimizer.zero_grad()

        loss.backward()

        if suffix:
            suffix_optimizer.step()
        if prefix:
            prefix_optimizer.step()


        # Track best result
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_suffix = suffix_embeds_string.detach().clone()
            best_prefix = embeds.detach().clone()

        pbar.set_description(f"Loss: {loss.item():.4f}, Best: {best_loss:.4f}")

    return best_embeds

def project_embeddings_to_tokens(model, tokenizer, suffix_embeds):
    """Project continuous embeddings to discrete tokens."""
    # Get the model's embedding matrix
    embedding_layer = get_embedding_layer(model)
    embedding_matrix = embedding_layer.weight  # [vocab_size, hidden_size]

    # Find closest tokens
    suffix_tokens = []
    for i in range(suffix_embeds.shape[1]):
        embed_i = suffix_embeds[0, i, :]

        # Calculate similarities with all tokens
        similarities = torch.nn.functional.cosine_similarity(
            embed_i.unsqueeze(0),
            embedding_matrix
        )

        # Get the most similar token
        token_id = similarities.argmax().item()
        suffix_tokens.append(token_id)

    # Convert to string
    suffix_string = tokenizer.decode(suffix_tokens)
    return suffix_string, suffix_tokens

# Gumbel-Softmax Approach
def optimize_gumbel_softmax_suffix(model, tokenizer, base_prompt, target_fn, mmlu_data,
                                   seed_prefix="You're an assistant", suffix_length=5, iterations=100, lr=0.01, temperature=0.5, prefix=True):
    """Optimize token distributions directly using Gumbel-Softmax."""
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Encode base prompt
    base_ids = tokenizer.encode(seed_prefix, return_tensors="pt").to(model.device)
    vocab_size = model.config.vocab_size

    # Initialize logits for token distributions (one distribution per position)
    token_logits = torch.zeros(1, suffix_length, vocab_size, requires_grad=True).to(model.device)

    # Optimizer
    optimizer = torch.optim.Adam([token_logits], lr=lr)

    best_loss = float('inf')
    best_distribution = None
    pbar = tqdm(range(iterations))

    for i in pbar:
        # Apply Gumbel-Softmax to get differentiable "soft" one-hot vectors
        gumbel_dist = torch.nn.functional.gumbel_softmax(
            token_logits, tau=temperature, hard=False, dim=-1
        )

        # Get embeddings by multiplying distributions with embedding matrix
        embedding_layer = get_embedding_layer(model)
        embedding_matrix = embedding_layer.weight  # [vocab_size, hidden_size]
        embeds = torch.matmul(gumbel_dist, embedding_matrix)

        # Concatenate with base prompt embeddings
        base_embeds = embedding_layer(base_ids)
        if prefix:
            inputs_embeds = torch.cat([embeds, base_embeds], dim=1)
        else:
            inputs_embeds = torch.cat([base_embeds, suffix_embeds], dim=1)

        # Forward pass
        outputs = model(inputs_embeds=inputs_embeds)

        # Convert distribution to tokens for evaluation
        token_ids = gumbel_dist.argmax(dim=-1)[0].tolist()
        tokens_str = tokenizer.decode(token_ids)

        # Calculate score/loss
        score = target_fn(outputs, None, model, tokenizer,
                          mmlu_examples=mmlu_data,
                          prompt_template=base_prompt,
                          prefix=tokens_str if prefix else None,
                          suffix=tokens_str if not prefix else None
                )

        loss = -score  # Negative because we want to maximize

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_distribution = gumbel_dist.detach().clone()

        # Gradually decrease temperature for annealing (optional)
        if (i+1) % 20 == 0 and temperature > 0.1:
            temperature *= 0.9

        # For display, convert current best distribution to tokens
        if i % 10 == 0:
            token_ids = best_distribution.argmax(dim=-1)[0].tolist()
            tokens_str = tokenizer.decode(token_ids)
            pbar.set_description(f"Loss: {loss.item():.4f}, Best: {best_loss:.4f}, Tokens: {tokens_str}")

    # Convert best distribution to tokens
    token_ids = best_distribution.argmax(dim=-1)[0].tolist()
    tokens_str = tokenizer.decode(token_ids)

    return tokens_str, token_ids, best_distribution

# Discrete Optimization Approach (Genetic Algorithm)
def optimize_discrete_suffix(model, tokenizer, base_prompt, target_fn, mmlu_data,
                           population_size=40, generations=20, suffix_length=5,
                           mutation_rate=0.3, elite_size=10):
    """Optimize suffix tokens directly using a genetic algorithm."""
    vocab_size = model.config.vocab_size
    device = model.device

    # Create initial population with random token IDs
    population = []

    # Generate initial population
    for _ in range(population_size):
        # Random token sequence
        suffix_tokens = torch.randint(0, vocab_size, (1, suffix_length)).to(device)

        # Evaluate
        suffix_str = tokenizer.decode(suffix_tokens[0])

        score = target_fn(None, None, model, tokenizer,
                         mmlu_examples=mmlu_data,
                         prompt_template=base_prompt,
                         suffix=suffix_str)

        population.append((suffix_tokens, score))

    # Sort by fitness
    population.sort(key=lambda x: x[1], reverse=True)

    best_score_history = []
    pbar = tqdm(range(generations))

    # Main evolutionary loop
    for gen in pbar:
        # Select elite individuals
        elite = population[:elite_size]
        best_score = elite[0][1]
        best_score_history.append(best_score)

        # Create new population
        new_population = list(elite)

        # Fill rest of population with crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection of parents
            parent1 = random.choice(elite)[0]
            parent2 = random.choice(elite)[0]

            # Crossover
            crossover_point = random.randint(1, suffix_length-1)
            child = torch.cat([
                parent1[0, :crossover_point],
                parent2[0, crossover_point:]
            ]).unsqueeze(0)

            # Mutation
            for i in range(suffix_length):
                if random.random() < mutation_rate:
                    child[0, i] = torch.randint(0, vocab_size, (1,)).to(device)

            # Evaluate child
            child_str = tokenizer.decode(child[0])
            score = target_fn(None, None, model, tokenizer,
                             mmlu_examples=mmlu_data,
                             prompt_template=base_prompt,
                             suffix=child_str)

            new_population.append((child, score))

        # Replace population
        population = new_population
        population.sort(key=lambda x: x[1], reverse=True)

        # Convert best individual to text for display
        best_tokens = population[0][0][0].cpu().numpy()
        best_suffix = tokenizer.decode(best_tokens)

        pbar.set_description(f"Gen {gen}, Score: {best_score:.4f}, Best: {best_suffix}")

    # Return best solution
    best_tokens = population[0][0][0].cpu().numpy()
    best_suffix = tokenizer.decode(best_tokens)
    return best_suffix, best_tokens, best_score_history


def optimize_gcg(model, tokenizer, base_prompt, target_fn, mmlu_data,
                 suffix_length=10, iterations=10, k=5, batch_size=10, prefix=False):
    """
    Greedy Coordinate Gradient (GCG) optimization as described in the paper.

    Unlike AutoPrompt which changes one position at a time, GCG considers all positions
    and selects the best token replacement across any position at each step.
    """
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Initialize random suffix tokens
    vocab_size = tokenizer.vocab_size
    suffix_tokens = torch.randint(0, vocab_size, (suffix_length,)).to(model.device)
    suffix_str = tokenizer.decode(suffix_tokens)

    # Get embedding layer
    embedding_layer = get_embedding_layer(model)

    best_score = float('-inf')
    best_suffix = suffix_str
    pbar = tqdm(range(iterations))

    for iter_idx in pbar:
        # Process all positions
        candidate_replacements = []

        # For each position
        for pos in range(suffix_length):

            # Get the one-hot representation of the token at position
            token_id = suffix_tokens[pos]
            token_one_hot = torch.zeros(vocab_size, device=model.device)
            token_one_hot[token_id] = 1.0
            token_one_hot.requires_grad = True

            # Calculate gradient with respect to the one-hot encoding
            optimizer = torch.optim.Adam([token_one_hot], lr=0.1)

            # Forward pass with the one-hot representation
            optimizer.zero_grad()
            score = target_fn(None, None, model, tokenizer,
                             mmlu_examples=mmlu_data,
                             prompt_template=base_prompt,
                             suffix=suffix_str if not prefix else None,
                             prefix=suffix_str if prefix else None)
            loss = -score
            loss.backward()

            # Get the gradient
            grad = token_one_hot.grad

            # Find top-k candidates based on negative gradient
            _, top_k_indices = torch.topk(-grad, k=k)

            # Store candidates for this position
            for token_idx in top_k_indices:
                candidate_replacements.append((pos, token_idx))

        # Randomly sample batch_size candidates from all positions
        batch_indices = torch.randperm(len(candidate_replacements))[:batch_size]
        batch_candidates = [candidate_replacements[i] for i in batch_indices]

        # Evaluate each candidate in the batch
        best_batch_score = float('-inf')
        best_batch_tokens = None
        best_batch_str = None

        for pos, token_idx in batch_candidates:
            # Create a copy of the suffix tokens and replace at position
            candidate_tokens = suffix_tokens.clone()
            candidate_tokens[pos] = token_idx

            # Convert to string
            candidate_str = tokenizer.decode(candidate_tokens)

            # Evaluate
            candidate_score = target_fn(None, None, model, tokenizer,
                                      mmlu_examples=mmlu_data,
                                      prompt_template=base_prompt,
                                      suffix=candidate_str if not prefix else None,
                                      prefix=candidate_str if prefix else None)

            # Update best in batch
            if candidate_score > best_batch_score:
                best_batch_score = candidate_score
                best_batch_tokens = candidate_tokens.clone()
                best_batch_str = candidate_str

        # Update if better than current best
        if best_batch_score > best_score:
            best_score = best_batch_score
            best_suffix = best_batch_str
            suffix_tokens = best_batch_tokens
            suffix_str = best_batch_str

        pbar.set_description(f"Score: {best_score:.4f}, Current: {suffix_str}")

    return best_suffix, suffix_tokens.cpu().numpy(), best_score


# Evaluate optimized suffixes on MMLU
def evaluate_mmlu_performance(model, tokenizer, mmlu_examples, base_prompt, suffix="", prefix="", n_samples=20):
    """Evaluate the prompt on MMLU dataset."""

    # Select random examples for evaluation
    eval_examples = {
        'stem': random.sample(mmlu_examples['stem'], min(n_samples, len(mmlu_examples['stem']))),
        'non_stem': random.sample(mmlu_examples['non_stem'], min(n_samples, len(mmlu_examples['non_stem'])))
    }

    results = {
        'stem': {'correct': 0, 'total': 0},
        'non_stem': {'refusals': 0, 'total': 0}
    }

    # Process examples
    for category, examples in eval_examples.items():
        print(f"\nEvaluating {category} examples:")

        for i, example in enumerate(examples):
            # Format the prompt
            formatted_prompt = format_mmlu_example(example, base_prompt)
            formatted_prompt = formatted_prompt + " " + suffix if suffix else base_prompt
            formatted_prompt = prefix + " " + formatted_prompt if prefix else formatted_prompt

            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

            # Generate response
            with torch.no_grad():
                output_ids = model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,
                    do_sample=False
                )

            # Get the generated answer
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer_text = generated_text[len(formatted_prompt):].strip()

            # Check correctness based on category
            if category == 'stem':
                correct_answer = "ABCD"[example['answer']]
                correct = correct_answer in answer_text
                results['stem']['correct'] += 1 if correct else 0
                results['stem']['total'] += 1

                if i < 3:  # Show a few examples
                    print(f"Q: {example['question']}")
                    print(f"Generated: '{answer_text}'")
                    print(f"Correct answer: {correct_answer}")
                    print(f"Result: {'✓' if correct else '✗'}\n")

            else:
                refused = "NOT STEM" in answer_text
                results['non_stem']['refusals'] += 1 if refused else 0
                results['non_stem']['total'] += 1

                if i < 3:  # Show a few examples
                    print(f"Q: {example['question']}")
                    print(f"Generated: '{answer_text}'")
                    print(f"Expected refusal: {'✓' if refused else '✗'}\n")

    # Calculate metrics
    stem_accuracy = results['stem']['correct'] / results['stem']['total'] if results['stem']['total'] > 0 else 0
    non_stem_refusal_rate = results['non_stem']['refusals'] / results['non_stem']['total'] if results['non_stem']['total'] > 0 else 0

    print("\nEvaluation Results:")
    print(f"STEM accuracy: {stem_accuracy:.4f} ({results['stem']['correct']}/{results['stem']['total']})")
    print(f"Non-STEM refusal rate: {non_stem_refusal_rate:.4f} ({results['non_stem']['refusals']}/{results['non_stem']['total']})")

    # Combined score (weighted)
    combined_score = 0.5 * stem_accuracy + 0.5 * non_stem_refusal_rate
    print(f"Combined score: {combined_score:.4f}")

    return {
        'stem_accuracy': stem_accuracy,
        'non_stem_refusal': non_stem_refusal_rate,
        'combined_score': combined_score,
        'results': results
    }

# Compare all three approaches on MMLU
def compare_all_approaches_mmlu(model_name="gpt2", seed_prompt=None, suffix_length=10):
    if seed_prompt is None:
        seed_prompt = "Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"

    print("Loading model and data...")
    model, tokenizer = setup_model_and_tokenizer(model_name)
    mmlu_data = load_mmlu_data(num_examples=50)  # Load a subset for efficiency

    # Define target function for MMLU
    def target_fn(outputs=None, input_ids=None, model=model, tokenizer=tokenizer, **kwargs):
        return target_mmlu_performance(
            model=model,
            tokenizer=tokenizer,
            **kwargs
        )

    print("\nRunning Greedy Coordinate Gradient (GCG) optimization...")
    start_time = time.time()
    gcg_suffix, gcg_tokens, _ = optimize_gcg(
        model, tokenizer, seed_prompt, target_fn,
        mmlu_data=mmlu_data,
        suffix_length=suffix_length,
        iterations=10,  # Reduced for demonstration
        prefix=True
    )
    gcg_time = time.time() - start_time

    return evaluate_mmlu_performance(
        model, tokenizer, seed_prompt,  mmlu_data, suffix=continuous_suffix
    )

    continuous_suffix, _ = project_embeddings_to_tokens(model, tokenizer, continuous_embeds)
    print(f"Continuous method result: '{continuous_suffix}' (took {continuous_time:.2f}s)")

    # Evaluate all methods
    print("\nEvaluating continuous method suffix:")
    continuous_results = evaluate_mmlu_performance(
        model, tokenizer, seed_prompt,  mmlu_data, suffix=continuous_suffix
    )

    # Compare results
    print("\nComparison Summary:")
    print(f"Continuous method: Suffix='{continuous_suffix}', Combined Score={continuous_results['combined_score']:.4f}, Time={continuous_time:.2f}s")
    print(f"Gumbel-Softmax method: Suffix='{gumbel_suffix}', Combined Score={gumbel_results['combined_score']:.4f}, Time={gumbel_time:.2f}s")
    print(f"Discrete method: Suffix='{discrete_suffix}', Combined Score={discrete_results['combined_score']:.4f}, Time={discrete_time:.2f}s")

    # Determine winner
    scores = [
        ("Continuous", continuous_results['combined_score']),
        ("Gumbel-Softmax", gumbel_results['combined_score']),
        ("Discrete", discrete_results['combined_score'])
    ]
    winner = max(scores, key=lambda x: x[1])
    print(f"\nBest performing method: {winner[0]} with score {winner[1]:.4f}")

    return {
        "continuous": {
            "suffix": continuous_suffix,
            "results": continuous_results,
            "time": continuous_time
        },
        "gumbel": {
            "suffix": gumbel_suffix,
            "results": gumbel_results,
            "time": gumbel_time
        },
        "discrete": {
            "suffix": discrete_suffix,
            "results": discrete_results,
            "time": discrete_time
        }
    }

# Usage example
if __name__ == "__main__":
    seed_prompt = "You are a STEM fields assistant. Answer all non-STEM related questions with NOT STEM.\nQuestion: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"

    results = compare_all_approaches_mmlu(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        seed_prompt=seed_prompt,
        suffix_length=10
    )
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ['TORCH_USE_CUDA_DSA'] = 1
print(f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.utils.data import DataLoader

from llm_controllers.steerers.prompt_steerer import PromptSteerer
from llm_controllers.steerers.act_add_steerer import ActAddSteerer
from llm_controllers.steerers.pca_steerer import PCASteerer
from llm_controllers.steerers.probe_activation_steerer import LinearProbeSteerer, TorchModelSteerer

from utils.dataset_utils.persuade_dataset import PersuadeDataset
from utils.dataset_utils.mmlu_dataset import MMLUDataset
from utils.dataset_utils.sni_dataset import SNIDataset

from utils.evaluation_utils.evaluator import FeedbackEvaluator


def plot_steering_vector(steering_vector, layer_name, k=20):
    if steering_vector is None or steering_vector.size == 0:
        print(f"Invalid steering vector for layer {layer_name}")
        return

    steering_vector = steering_vector.flatten() # Ensure 1D

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Steering Vector Analysis - Layer: {layer_name}", fontsize=14)

    # Plot 1: Line plot of the vector values
    axs[0].plot(steering_vector)
    axs[0].set_title('Steering Vector Values')
    axs[0].set_xlabel('Dimension Index')
    axs[0].set_ylabel('Value')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Top K dimensions by absolute value
    if k > 0 and steering_vector.size > 0:
         k = min(k, steering_vector.size) # Adjust k if vector is smaller
         top_k_indices = np.argsort(np.abs(steering_vector))[-k:]
         top_k_values = steering_vector[top_k_indices]
         dim_labels = [f'Dim {i}' for i in top_k_indices]

         colors = ['red' if v < 0 else 'blue' for v in top_k_values]
         axs[1].bar(range(k), top_k_values, color=colors)
         axs[1].set_xticks(range(k))
         axs[1].set_xticklabels(dim_labels, rotation=90)
         axs[1].set_title(f'Top {k} Dimensions by Magnitude')
         axs[1].set_ylabel('Value')
         axs[1].grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(f'steering_vector_{layer_name}.png')
    plt.close()


def plot_steering_vector_norms(average_diff_activations):
    if not average_diff_activations:
        print("No steering vectors provided.")
        return

    norms = [np.linalg.norm(vec) for vec in average_diff_activations.values() if vec is not None]
    valid_layer_names = [name for name, vec in average_diff_activations.items() if vec is not None]

    if not norms:
        print("No valid steering vectors found to calculate norms.")
        return

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(norms)), norms, tick_label=valid_layer_names)
    plt.xticks(rotation=90)
    plt.ylabel('L2 Norm (Magnitude)')
    plt.title('Steering Vector Norms Across Layers')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('steering_vector_norms.png')
    plt.close()


def plot_steering_vector_similarity(average_diff_activations):
    """
    Plots a heatmap of the cosine similarity between steering vectors from different layers.

    Args:
        average_diff_activations (dict): Dictionary from layer_name to steering vector (np.ndarray).
    """
    if not average_diff_activations:
        print("No steering vectors provided.")
        return

    vectors = [vec.flatten() for vec in average_diff_activations.values() if vec is not None]
    valid_layer_names = [name for name, vec in average_diff_activations.items() if vec is not None]

    if len(vectors) < 2:
        print("Need at least two valid steering vectors to compare similarity.")
        return

    # Ensure all vectors are 1D for cosine similarity calculation if needed,
    # although cosine_similarity handles 2D arrays row-wise. Stacking is safer.
    try:
         vector_matrix = np.stack(vectors) # Shape (n_layers, hidden_dim)
    except ValueError as e:
         print(f"Error stacking vectors, likely due to inconsistent dimensions: {e}")
         # Attempt to pad or resize if necessary, or just report error
         max_len = max(v.shape[0] for v in vectors)
         padded_vectors = []
         for v in vectors:
             if v.shape[0] < max_len:
                 pad_width = max_len - v.shape[0]
                 padded_v = np.pad(v, (0, pad_width), 'constant')
                 padded_vectors.append(padded_v)
                 print(f"Warning: Padded vector for layer {valid_layer_names[len(padded_vectors)-1]}")
             else:
                 padded_vectors.append(v)
         vector_matrix = np.stack(padded_vectors)
         # return # Alternatively, decide whether to proceed with potentially padded vectors


    similarity_matrix = cosine_similarity(vector_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                xticklabels=valid_layer_names, yticklabels=valid_layer_names)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Cosine Similarity Between Steering Vectors Across Layers')
    plt.tight_layout()
    plt.savefig('steering_vector_similarity.png')
    plt.close()


def visualize_steering(rank=0, world_size=0):

    if world_size > 1:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank) # Set the current device for this process
    
    negative_essay_prompts, encourageing_essay_prompts = PersuadeDataset().get_data()
    test_prompts = [
        encourageing_essay_prompts[0],
        encourageing_essay_prompts[1],
        encourageing_essay_prompts[2],
        encourageing_essay_prompts[3]
    ]

    MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct" 

    selected_layers = None
    probe_steerer = TorchModelSteerer(MODEL_NAME, selected_layers=selected_layers)
    probe_steerer.train_classifier(encourageing_essay_prompts, negative_essay_prompts)
    for prompt in test_prompts:
        print(f"Probe Steered: {probe_steerer.generate(prompt, max_length=20)}")
        print("\n")
        probe_steerer.clear_transformation_functions()

    linear_steerer = LinearProbeSteerer(MODEL_NAME, selected_layers=None)
    linear_steerer.train_classifier(encourageing_essay_prompts, negative_essay_prompts)
    for prompt in test_prompts:
        linear_steerer.set_transformation_function(c=2, func_type='multiply')
        print(f"Linear Steered: {linear_steerer.generate(prompt, max_length=20)}")
        print("\n")
        linear_steerer.clear_transformation_functions()
    
    selected_layers = None

    avg_steerer = ActAddSteerer(MODEL_NAME, selected_layers=selected_layers) # Uses LLMSteerer internally

    torch.cuda.empty_cache()

    # --- 1. Extract Activations (Example Data) ---
    indices = np.random.choice(len(negative_essay_prompts), size=10, replace=False)
    positive_texts = [encourageing_essay_prompts[i] for i in indices] # Using list comprehension for proper indexing
    negative_texts = [negative_essay_prompts[i] for i in indices] # Same fix for negative_texts
    all_texts = positive_texts + negative_texts
    labels = ['positive'] * len(positive_texts) + ['negative'] * len(negative_texts)

    print("Extracting activations...")
    all_activations = avg_steerer.extract_activations(all_texts, aggregation_calc="last")

    print("Plotting activations...")
    # --- 2. Visualize Activations ---
    target_layer = avg_steerer.layer_names[len(avg_steerer.layer_names)-1] 
    if dist.get_rank() == 0:  # Only the main process should plot
        avg_steerer.plot_activation_projection(all_activations, target_layer, labels, method='tsne')
    all_activations = None # Clear to free up memory
    
    probe_steerer.visualize_activations(positive_texts, target_layer, aggregation_calc="last", save_path='steering_output/activations_good.png')
    probe_steerer.visualize_activations(negative_texts, target_layer, aggregation_calc="mean", save_path='steering_output/activations_bad.png')
    
    print("Calculating steering vector...")
    average_diff_activations = avg_steerer.extract_average_diff(positive_texts, negative_texts)
    if dist.get_rank() == 0:  # Only the main process should plot
        plot_steering_vector(average_diff_activations[target_layer], target_layer, k=20)
        plot_steering_vector_norms(average_diff_activations)
        plot_steering_vector_similarity(average_diff_activations)


    print("\nComparing steered vs. unsteered generation...")
    test_prompts = [
        encourageing_essay_prompts[0],
        encourageing_essay_prompts[1],
        encourageing_essay_prompts[2],
        encourageing_essay_prompts[3]
    ]

    # Define steering parameters (apply the positive vector with a coefficient)
    # Apply to one or more layers where the vector norm was significant
    steering_params = {}
    steering_coeff = 2
    if target_layer in average_diff_activations and average_diff_activations[target_layer] is not None:
        steering_params[target_layer] = {
            'vector': average_diff_activations[target_layer], # The pos-neg vector
            'coeff': steering_coeff
        }


    if world_size > 1:
        destroy_process_group()



def load_persuade(training_examples, test_examples=1):
    essays = PersuadeDataset().get_data(training_examples)
    positive_prompts = PersuadeDataset().get_sentiment_variations(essays, "encouraging")
    negative_prompts = PersuadeDataset().get_sentiment_variations(essays, "discouraging")
    neutral_prompts = PersuadeDataset().get_sentiment_variations(essays, "neutral")

    indices = np.random.choice(len(negative_prompts), size=training_examples, replace=False)
    positive_texts = [positive_prompts[i] for i in indices] # Using list comprehension for proper indexing
    negative_texts = [negative_prompts[i] for i in indices] # Same fix for negative_texts

    test_indices = np.random.choice(len(neutral_prompts), size=test_examples, replace=False)
    test_texts = [neutral_prompts[i] for i in test_indices] # Same fix for test_prompts

    return positive_texts, negative_texts, test_texts

def evaluate_persuade(config, steerer, test_texts ):
    # default_model = PromptSteerer(config['model'], "{prompt}", False)
    feedback_evaluator = FeedbackEvaluator() # Provider might need API keys etc.

    local_steered_winner = 0 # Count wins for this rank's subset
    num_test_prompts = len(test_texts) # Total prompts
                
                
    dataloader = DataLoader(test_texts, batch_size=1, sampler=None)
    generated_outputs = []
    for prompt in dataloader: # TODO: Can probably kick over to use DistributedDataset
        prompt = prompt[0]
        steered_output = steerer.generate(prompt, max_length=250, coeff=config['steering_coeff'])
        unsteered_output = steerer.generate_uncontrolled(prompt, max_length=250)
        generated_outputs.append([steered_output, unsteered_output])

        result = feedback_evaluator.compare_feedback(steered_output, unsteered_output) # Modify if comparing against well_prompted
        if result and result.get('winner') == 1:
            local_steered_winner += 1

    device = steerer.model.device
    local_steered_winner_tensor = torch.tensor([local_steered_winner], dtype=torch.float32, device=device)

    total_steered_winner = local_steered_winner_tensor.item()
    total_comparisons = num_test_prompts
    total_comparisons = num_test_prompts
    percent_win = (total_steered_winner / total_comparisons) * 100 if total_comparisons > 0 else 0
    return generated_outputs,total_steered_winner,percent_win


def load_mmlu(training_examples, test_examples=10):

    in_domain = MMLUDataset(sample_size=training_examples // 2, split='validation', domains=['high_school_chemistry'], in_domain=True)
    out_of_domain = MMLUDataset(sample_size=training_examples // 2, split='validation', domains=['high_school_chemistry'], in_domain=False)

    test_count = test_examples // 2
    test_indices = np.random.choice(len(out_of_domain), size=test_count, replace=False)
    test_questions = [out_of_domain[i] for i in test_indices] # Same fix for test_prompts
    out_of_domain = [out_of_domain[i] for i in len(out_of_domain) if i not in test_indices]

    test_indices = np.random.choice(len(in_domain), size=test_count, replace=False)
    test_questions = [in_domain[i] for i in test_indices] # Same fix for test_prompts
    in_domain = [in_domain[i] for i in len(in_domain) if i not in test_indices]

    return in_domain, out_of_domain, test_questions


def load_sni(training_examples, test_examples=10):

    in_domain = SNIDataset().get_data(sample_size=training_examples // 2, split='validation', in_domain='high_school_chemistry') # domain='stem')
    out_of_domain = SNIDataset().get_data(sample_size=len(in_domain), split='validation', out_domain='non_stem')

    test_count = test_examples // 2
    test_indices = np.random.choice(len(out_of_domain), size=test_count, replace=False)
    test_questions = [out_of_domain[i] for i in test_indices] # Same fix for test_prompts
    out_of_domain = [out_of_domain[i] for i in len(out_of_domain) if i not in test_indices]

    test_indices = np.random.choice(len(in_domain), size=test_count, replace=False)
    test_questions = [in_domain[i] for i in test_indices] # Same fix for test_prompts
    in_domain = [in_domain[i] for i in len(in_domain) if i not in test_indices]

    return in_domain, out_of_domain, test_questions


def wand_b_iteration(config=None):
    """
    Runs a single sweep trial, initializing and cleaning up
    torch.distributed for rank=0, world_size=1.
    """
    run = wandb.init(
        project="scoped-llm"
    )

    # config=wandb.config



    # wandb.config.get('dataset')
    # config=wandb.config

    is_ddp = False
    try:
        torch.cuda.empty_cache()

        if config['dataset'] == 'persuade':
            positive_texts, negative_texts, test_texts = load_persuade(config['training_examples'])

        model_name = config['model'].replace('.', '_').replace('/', '_')
        filename = f"{model_name}_{config['steerer_type']}_vectors"
        folder = os.path.join(os.getcwd(), "steering_vectors")
        path = str(os.path.join(folder, filename))

        if config['steerer_type'] == 'torch':
            steerer = TorchModelSteerer(config['model'], config['target_layers'], is_ddp, path)
        elif config['steerer_type'] == 'linear_probe':
            steerer = LinearProbeSteerer(config['model'], config['target_layers'], is_ddp, path)
        elif config['steerer_type'] == 'average':
            steerer = ActAddSteerer(config['model'], config['target_layers'], is_ddp, path)
        elif config['steerer_type'] == 'pca':
            steerer = PCASteerer(config['model'], config['target_layers'], is_ddp, path)
        elif config['steerer_type'] == 'linear_probe_scoper':
            pass
        elif config['steerer_type'] == 'consitutional_classier_scoper':
            pass
        elif config['steerer_type'] == 'circuit_breaker_scoper':
            pass

        trained_vectors = steerer.train(positive_texts, negative_texts, batch_size=1)
        if not os.path.exists(folder):
            os.makedirs(folder)

        if config['dataset'] == 'persuade':
            generated_outputs, total_steered_winner, percent_win = evaluate_persuade(config, steerer, test_texts)
        elif config['dataset'] == 'mmlu':
            pass 
        columns = ['steered','unsteered']
        table = wandb.Table(data=generated_outputs, columns=columns)
        wandb.log({ "percent_win": percent_win, "total_steered_wins": total_steered_winner, "texts": table})

    finally:
        run.finish()



def wand_b_sweep():

    large_models = ['unsloth/Llama-3.3-70B-Instruct', 'Qwen/Qwen2.5-32B-Instruct']
    small_models_1 = ['unsloth/Llama-3.2-3B-Instruct', 'unsloth/Llama-3.2-1B-Instruct', 'unsloth/Meta-Llama-3.1-8B']


    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'percent_win'},
        'parameters': {
            'model': {'values': small_models_1},
            'steerer_type': {'values': ['average']}, # 'torch', 'linear_probe', 
            'target_layers': {'values': ['first', 'middle', 'last', 'last_3']},
            'steering_coeff': {'values': [0.5, 1.0, 5.0, 10.0]},

            'training_examples': {'value': 100},
            'dataset': {'value': 'persuade'}
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-test-project')
    wandb.agent(sweep_id, function=wand_b_iteration, count=10)

import itertools
def my_sweep():

    large_models = ['unsloth/Llama-3.3-70B-Instruct', 'Qwen/Qwen2.5-32B-Instruct']
    medium_models = None
    small_models_1 = ['unsloth/Llama-3.2-3B-Instruct', 'unsloth/Llama-3.2-1B-Instruct', 'unsloth/Meta-Llama-3.1-8B']
    small_models_2 = None

    param_grid = {
        'model': ['unsloth/Meta-Llama-3.1-8B-Instruct'],
        'steerer_type': ['average', 'pca', 'torch'],
        'target_layers': ['last_5', 'last_3', 'last', 'first', 'middle'],
        'steering_coeff': [5.0, 10.0, 0.5, 1.0],
        'dataset': ['persuade'],
        'training_examples': [1000],
        'batch_size': [1]
    }

    # --- Generate Combinations ---
    keys, values = zip(*param_grid.items())
    config_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, config in enumerate(config_combinations):
        wand_b_iteration(config)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    load_dotenv()

    my_sweep()
    wand_b_sweep()


# NOTE: Ideal is 
# - Combining multiple different vectors together
# - Reworking selected blocks



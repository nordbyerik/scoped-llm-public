import os
from typing import List
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
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

from llm_controllers.scopers.latent_space_classifier_scoper  import ScopeClassifier

from utils.dataset_utils.persuade_dataset import PersuadeDataset
from utils.dataset_utils.mmlu_dataset import MMLUDataset
from utils.dataset_utils.sni_dataset import SNIDataset

from utils.evaluation_utils.evaluator import FeedbackEvaluator
from utils.evaluation_utils.mmlu_evaluator import MMLUEvaluator

def load_persuade(training_examples, test_examples=5):
    essays = PersuadeDataset().get_data(training_examples)
    positive_prompts = PersuadeDataset().get_sentiment_variations(essays, "encouraging")
    negative_prompts = PersuadeDataset().get_sentiment_variations(essays, "discouraging")
    neutral_prompts = PersuadeDataset().get_sentiment_variations(essays, "neutral")

    indices = np.random.choice(len(negative_prompts), size=training_examples, replace=False)
    positive_texts = [positive_prompts[i] for i in indices] # Using list comprehension for proper indexing
    negative_texts = [negative_prompts[i] for i in indices] # Same fix for negative_texts

    test_indices = np.random.choice(len(neutral_prompts), size=test_examples, replace=False)
    test_texts = [neutral_prompts[i] for i in test_indices] # Same fix for test_prompts
    unformatted_test_texts = [essays[i] for i in test_indices]
    return positive_texts, negative_texts, test_texts, unformatted_test_texts

def evaluate_persuade(config, steerer, test_texts, encouraging=False ):
    # default_model = PromptSteerer(config['model'], "{prompt}", False)
    feedback_evaluator = FeedbackEvaluator() # Provider might need API keys etc.

    local_steered_winner = 0 # Count wins for this rank's subset
    num_test_prompts = len(test_texts) # Total prompts
                
                
    dataloader = DataLoader(test_texts, batch_size=1, sampler=None)
    generated_outputs = []
    for prompt in dataloader: # TODO: Can probably kick over to use DistributedDataset
        prompt = prompt[0]

        encouraging_prompt = f"Essay: {prompt} Request: Provide feedback for this essay."
        prompt = f"Essay: {prompt} Request: Give kind feedback for this essay which is encouraging"

        steered_output = steerer.generate(prompt, max_length=250, coeff=config['steering_coeff'])
        if not encouraging:
            unsteered_output = steerer.generate_uncontrolled(prompt, max_length=250)
        else:
            unsteered_output = steerer.generate_uncontrolled(encouraging_prompt, max_length=250)
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


def load_mmlu(domains: List[str], training_examples=100, train_test_split=0.8):
    # TODO: Kinda gross.. maybe use DataLoader instead
    in_domain = MMLUDataset(sample_size=training_examples // 2, split='test', domains=domains, in_domain=True)
    out_of_domain = MMLUDataset(sample_size=training_examples // 2, split='test', domains=domains, in_domain=False)

    test_count = int(len(in_domain)*(1-train_test_split) // 2)
    test_indices = np.random.choice(len(out_of_domain), size=test_count, replace=False if test_count < len(out_of_domain) else True)
    test_questions_ood = [out_of_domain[i][0] for i in test_indices] # Same fix for test_prompts
    test_answers_ood = [out_of_domain[i][1] for i in test_indices]
    out_of_domain = [out_of_domain[i][0] for i in range(len(out_of_domain)) if i not in test_indices]

    test_count = int(len(out_of_domain)*(1-train_test_split) // 2)
    test_indices = np.random.choice(len(in_domain), size=test_count, replace=False if test_count < len(in_domain) else True)
    test_questions_in_domain = [in_domain[i][0] for i in test_indices] # Same fix for test_prompts
    test_answers_in_domain = [in_domain[i][1] for i in test_indices]
    in_domain = [in_domain[i][0] for i in range(len(in_domain)) if i not in test_indices]

    test_questions = test_questions_ood + test_questions_in_domain
    test_answers = test_answers_ood + test_answers_in_domain

    return in_domain, out_of_domain, test_questions, test_answers


def evaluate_mmlu(config, scoper, test_questions, test_answers, parser_type="logits"):
    mmlu_evaluator = MMLUEvaluator(scoper.tokenizer, parser_type) # Provider might need API keys etc.

    dataloader = DataLoader(test_questions, batch_size=10)
    
    test_count = 0
    running_accuracy = 0
    for prompts in dataloader: # TODO: Can probably kick over to use DistributedDataset
        steered_output = scoper(prompts).logits

        test_count += len(prompts)
        running_accuracy += mmlu_evaluator(steered_output, test_answers)
        
    return running_accuracy/test_count



def mmlu_iteration(config=None):
    """
    Runs a single sweep trial, initializing and cleaning up
    torch.distributed for rank=0, world_size=1.
    """
    run = wandb.init(
        project="scoped-llm",
        config=config
    )

    # config=wandb.config

    try:
        torch.cuda.empty_cache()

        in_domain, out_of_domain, test_questions, test_answers = load_mmlu(domains=config['domains'], training_examples=config['training_examples'])
        model_name = config['model'].replace('.', '_').replace('/', '_')
        filename = f"{model_name}_{config['scoper_type']}_vectors"
        folder = os.path.join(os.getcwd(), "scoping_activations")
        path = str(os.path.join(folder, filename))

        if config['scoper_type'] == 'linear_probe_scoper':
            scoper = ScopeClassifier(config['model'], save_folder_path=path)
        elif config['scoper_type'] == 'circuit_breaker_scoper':
            pass

        scoper.train(in_domain, out_of_domain, batch_size=10)
        if not os.path.exists(folder):
            os.makedirs(folder)

        
        accuracy = evaluate_mmlu(config, scoper, test_questions, test_answers)
        
        wandb.log({ "accuarcy": accuracy})
    except Exception as e:
        print(f"Error on thi: {e}")
        return {"config": config, "result": "failed"}
    finally:
        run.finish()

    return {"config": config, "accuracy": accuracy}


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
    small_models_1 = ['unsloth/Llama-3.2-3B-Instruct', ]
    small_models_2 = None

    logs = []
    param_grid = {
        'model': ['unsloth/Llama-3.2-3B-Instruct'],
        'scoper_type': ['linear_probe_scoper'],
        'domains': ['stem'],
        'dataset': ['mmlu'],
        'training_examples': [100],
        'test_examples': [10],
        'batch_size': [10]
    }

    # --- Generate Combinations ---
    keys, values = zip(*param_grid.items())
    config_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, config in enumerate(config_combinations):
        results = mmlu_iteration(config)
        with open("logs.txt", "a") as f:
            f.write(str(results) + "\n")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    load_dotenv()

    my_sweep()
    wand_b_sweep()


# NOTE: Ideal is 
# - Combining multiple different vectors together
# - Reworking selected blocks



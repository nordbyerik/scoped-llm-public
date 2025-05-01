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
from torch.utils.data import DataLoader, ConcatDataset, DataLoader

from llm_controllers.steerers.prompt_steerer import PromptSteerer
from llm_controllers.steerers.act_add_steerer import ActAddSteerer
from llm_controllers.steerers.pca_steerer import PCASteerer
from llm_controllers.steerers.probe_activation_steerer import LinearProbeSteerer, TorchModelSteerer

from llm_controllers.scopers.latent_space_classifier_scoper  import ScopeClassifier
from llm_controllers.scopers.hardened_prompt_scoper import HardenedPromptScoper, PromptClassificationScoper
from llm_controllers.scopers.circuit_breaker_scoper import CircuitBreakerScoper

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


    # For out-of-domain data
    ood_total = len(out_of_domain)
    ood_test_count = int(ood_total * (1-train_test_split))
    ood_test_indices = np.random.choice(ood_total, size=ood_test_count, replace=False if ood_test_count < ood_total else True)
    ood_train_indices = [i for i in range(ood_total) if i not in ood_test_indices]
    test_ood = out_of_domain[ood_test_indices]
    train_ood = out_of_domain[ood_train_indices]

    # For in-domain data
    in_total = len(in_domain)
    in_test_count = int(in_total * (1-train_test_split))
    in_test_indices = np.random.choice(in_total, size=in_test_count, replace=False if in_test_count < in_total else True)
    in_train_indices = [i for i in range(in_total) if i not in in_test_indices]
    test_in_domain = in_domain[in_test_indices]
    train_in_domain = in_domain[in_train_indices]

    # Combine test datasets
    test_dataset = MMLUDataset.__new__(MMLUDataset)
    test_dataset.data = test_ood.data + test_in_domain.data
    test_dataset.answers = test_ood.answers + test_in_domain.answers
    test_dataset.in_domain = [0]*len(test_ood) + [1]*len(test_in_domain)

    # Update training datasets
    out_of_domain = train_ood
    in_domain = train_in_domain
    return in_domain, out_of_domain, test_dataset




def mmlu_iteration(config=None):
    """
    Runs a single sweep trial, initializing and cleaning up
    torch.distributed for rank=0, world_size=1.
    """
    run = wandb.init(
        project="scoped-llm"
    )
    config = wandb.config



    try:
        torch.cuda.empty_cache()

        in_domain, out_of_domain, test_dataset = load_mmlu(domains=config['domains'], training_examples=config['training_examples'])
        model_name = config['model'].replace('.', '_').replace('/', '_')
        filename = f"{model_name}_{config['scoper_type']}_vectors"
        folder = os.path.join(os.getcwd(), "scoping_activations")
        path = str(os.path.join(folder, filename))

        if config['scoper_type'] == 'linear_probe_scoper':
            scoper = ScopeClassifier(config['model'], save_folder_path=path)
        elif config['scoper_type'] == 'hardened_prompt_scoper':
            scoper = HardenedPromptScoper(config['model'], domains=config['domains'])
        elif config['scoper_type'] == 'prompt_classification_scoper':
           scoper = PromptClassificationScoper(config['model'], domains=config['domains'])
        elif config['scoper_type'] == 'circuit_breaker_scoper':
            scoper = CircuitBreakerScoper(config['model'], save_folder_path=path)
    
        scoper.train(in_domain, out_of_domain, batch_size=10)
        if not os.path.exists(folder):
            os.makedirs(folder)

        mmlu_evaluator = MMLUEvaluator(scoper.tokenizer, 'logits') # Provider might need API keys etc.

        questions = test_dataset.data
        batch_size = 2

        steered_output = None
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batch_steered_output = scoper(batch).logits
            batch_steered_output = batch_steered_output[:, -1]
            if steered_output is None:
                steered_output = torch.zeros((len(questions), batch_steered_output.shape[-1]))
            steered_output[i:i + batch_size] = batch_steered_output

        in_domain_accuracy, out_of_domain_accuracy, accuracy, precision, recall, f1_score = mmlu_evaluator(steered_output, test_dataset)
        
        wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score, "in_domain_accuracy": in_domain_accuracy, "out_of_domain_accuracy": out_of_domain_accuracy, "result": "success"})
    except Exception as e:
        print(f"Error on thi: {e}")
        return {"config": config, "result": "failed"}
    finally:
        run.finish()

    return {"config": config, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score, "in_domain_accuracy": in_domain_accuracy, "out_of_domain_accuracy": out_of_domain_accuracy, "result": "success"}


def wand_b_sweep():

    large_models = ['unsloth/Llama-3.3-70B-Instruct', 'Qwen/Qwen2.5-32B-Instruct']
    small_models_1 = ['unsloth/Llama-3.2-3B-Instruct', 'unsloth/Llama-3.2-1B-Instruct', 'unsloth/Meta-Llama-3.1-8B']


    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'accuracy'},
        'parameters': {
            'model': {'values': ['unsloth/Llama-3.2-3B-Instruct', 'unsloth/Llama-3.2-1B-Instruct', 'unsloth/Meta-Llama-3.1-8B', 'google/gemma-2-27b' ]},
            'scoper_type':{'values': ['circuit_breaker_scoper', 'prompt_classification_scoper','hardened_prompt_scoper','linear_probe_scoper' ]}, # 'torch', 'linear_probe', 
            'domains': {'values': [
                ["astronomy"], 
                "stem", 
                ['world_religions'],
                ['virology'],
                ['philosophy'],
                ['marketing'],
                ['astronomy'],
                ['professional_law', 'jurisprudence', 'business_ethics'],
                ['high_school_biology', 'college_biology', 'medical_genetics'],
                ['high_school_mathematics', 'college_mathematics', 'elementary_mathematics'],
                ['high_school_psychology', 'professional_psychology', 'moral_scenarios'],
                ['high_school_world_history', 'high_school_european_history', 'high_school_us_history', 'prehistory']
                ]},
            'dataset': {'value': 'mmlu'},
            'training_examples': {'value': 1000},
            'test_examples': {'value': 100},
            'batch_size': {'value': 2}
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-test-project')
    wandb.agent(sweep_id, function=mmlu_iteration,  count=25)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    load_dotenv()

    wand_b_sweep()


# NOTE: Ideal is 
# - Combining multiple different vectors together
# - Reworking selected blocks



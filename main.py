import os
from typing import List
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
print(f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
from dotenv import load_dotenv

import numpy as np

import wandb

import torch
from torch.utils.data import DataLoader, DataLoader

from llm_controllers.llm_controller import LLMController
from llm_controllers.steerers.prompt_steerer import PromptSteerer
from llm_controllers.steerers.act_add_steerer import ActAddSteerer
from llm_controllers.steerers.pca_steerer import PCASteerer
from llm_controllers.steerers.probe_activation_steerer import LinearProbeSteerer, TorchModelSteerer

from llm_controllers.scopers.latent_space_classifier_scoper  import ScopeClassifier
from llm_controllers.scopers.hardened_prompt_scoper import HardenedPromptScoper, PromptClassificationScoper
from llm_controllers.scopers.circuit_breaker_scoper import CircuitBreakerScoper

from utils.dataset_utils.persuade_dataset import PersuadeDataset
from utils.dataset_utils.mmlu_dataset import MMLUDataset

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


def load_mmlu(domains: List[str], training_examples=100, test_percentage=0.8):
    # TODO: Kinda gross.. maybe use DataLoader instead
    in_domain = MMLUDataset(sample_size=training_examples // 2, split='test', domains=domains, in_domain=True, test_percentage=test_percentage)
    out_of_domain = MMLUDataset(sample_size=training_examples // 2, split='test', domains=domains, in_domain=False, test_percentage=test_percentage)

    # For out-of-domain data
    in_domain_train = in_domain.get_train_dataset()
    in_domain_test = in_domain.get_test_dataset()
    out_of_domain_train = out_of_domain.get_train_dataset()
    out_of_domain_test = out_of_domain.get_test_dataset()

    # Combine test datasets
    test_dataset = MMLUDataset.__new__(MMLUDataset)
    test_dataset.data = out_of_domain_test.data + in_domain_test.data
    test_dataset.answers = out_of_domain_test.answers + in_domain_test.answers
    test_dataset.in_domain = [0]*len(out_of_domain_test) + [1]*len(in_domain_test)

    return in_domain_train, out_of_domain_train, test_dataset




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

        in_domain, out_of_domain, test_dataset = load_mmlu(
            domains=config['domains'], 
            training_examples=config['training_examples'], 
            test_percentage=config['test_percentage']
            )
        
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
        elif config['scoper_type'] == 'activation_steerer':
            scoper = ActAddSteerer(config['model'], save_folder_path=path)

        scoper.train(in_domain, out_of_domain, batch_size=10)
        if not os.path.exists(folder):
            os.makedirs(folder)


        mmlu_evaluator = MMLUEvaluator(scoper.tokenizer, 'logits') # Provider might need API keys etc.

        questions = test_dataset.data
        batch_size = 2

        steered_output = None
        unsteered_output = None
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]

            batch_steered_output = scoper(batch).logits
            batch_steered_output = batch_steered_output[:, -1]
            if steered_output is None:
                steered_output = torch.zeros((len(questions), batch_steered_output.shape[-1]))
            steered_output[i:i + batch_size] = batch_steered_output

        del scoper
        torch.cuda.empty_cache()
        plain_model = LLMController(model_name=config['model'], use_ddp=False)

        plain_output = None
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batch_plain_output = plain_model(batch).logits
            batch_plain_output = batch_plain_output[:, -1]
            if plain_output is None:
                plain_output = torch.zeros((len(questions), batch_plain_output.shape[-1]))
            plain_output[i:i + batch_size] = batch_plain_output

        metrics = mmlu_evaluator(steered_output, plain_output, test_dataset)

        results = {"config": config, "metrics": metrics}
        with open("logs.txt", "a") as f:
            f.write(str(results) + "\n")

        wandb.log({"metrics":metrics, "result": "success"})
    except Exception as e:
        print(f"Error on thi: {e}")
        return {"config": config, "result": "failed"}
    finally:
        run.finish()

    return {"config": config, "result": "success", "metrics": metrics}


def wand_b_sweep():

    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'accuracy'},
        'parameters': { 
            'model': {'values': ['unsloth/Llama-3.2-1B-Instruct', 'unsloth/Llama-3.2-3B-Instruct', 'unsloth/Meta-Llama-3.1-8B'  ]}, 
            'scoper_type':{'values': ['linear_probe_scoper']}, # 'torch', 'linear_probe', 
            'domains': {'values': [
                "stem", 
                ['world_religions'],
                ['high_school_chemistry'],
                ['professional_law', 'jurisprudence', 'business_ethics'],
                ]},
            'dataset': {'value': 'mmlu'},
            'training_examples': {'value': 100},
            'test_percentage': {'value': .2},
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



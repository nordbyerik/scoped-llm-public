import os
from typing import List
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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

from scope_datasets.persuade.persuade_dataset import PersuadeDataset
from scope_datasets.mmlu.mmlu_dataset import MMLUDataset
from scope_datasets.sni.sni_dataset import SNIDataset

from scope_datasets.persuade.persuade_evaluator import FeedbackEvaluator
from scope_datasets.mmlu.mmlu_evaluator import MMLUEvaluator

def load_dataset(dataset_type: str, domains: List[str], training_examples=100, test_percentage=0.8):
    # TODO: Kinda gross.. maybe use DataLoader instead
    if dataset_type == "mmlu":
        dataset_class = MMLUDataset

    in_domain = dataset_class(sample_size=training_examples // 2, split='test', domains=domains, in_domain=True, test_percentage=test_percentage)
    out_of_domain = dataset_class(sample_size=training_examples // 2, split='test', domains=domains, in_domain=False, test_percentage=test_percentage)

    # For out-of-domain data
    in_domain_train = in_domain.get_train_dataset()
    in_domain_test = in_domain.get_test_dataset()
    out_of_domain_train = out_of_domain.get_train_dataset()
    out_of_domain_test = out_of_domain.get_test_dataset()

    # Combine test scope_datasets
    test_dataset = dataset_class.__new__(dataset_class)
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

        in_domain, out_of_domain, test_dataset = load_dataset(
            dataset_type=config['dataset'],
            domains=config['domains'], 
            training_examples=config['training_examples'], 
            test_percentage=config['test_percentage']
            )
        
        model_name = config['model'].replace('.', '_').replace('/', '_')
        filename = f"{model_name}_{config['scoper_type']}_vectors"
        folder = os.path.join(os.getcwd(), "caches/scoping_activations")
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

        scoper.train(in_domain, out_of_domain, batch_size=10) # TODO: Move the logic of combining the data out to the dataset loader?
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
            'model': {'values': ['unsloth/Phi-4-mini-reasoning', 'unsloth/gemma-3-12b-it', 'Qwen/Qwen3-8B', 'unsloth/Meta-Llama-3.1-8B-Instruct']}, 
            'scoper_type':{'values': ['linear_probe_scoper', 'hardened_prompt_scoper', 'circuit_breaker_scoper', 'activation_steerer']}, # 'torch', 'linear_probe', 
            'domains': {'values': [
                "stem", 
                ['world_religions'],
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



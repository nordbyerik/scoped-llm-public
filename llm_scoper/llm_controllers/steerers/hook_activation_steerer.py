from llm_controllers.llm_controller import LLMController
import torch
from typing import Union, List
import numpy as np

from utils.activation_utils.activation_hook import ActivationHook
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class HookActivationController(LLMController):
    def __init__(self, model,  selected_layers=None, use_ddp=True):
        super().__init__(model, use_ddp)

        self.hooks = {}

        automatic_options = ['all', "last", "first", "middle", "last_2", "every_5th", "first_and_last"]
        if selected_layers in automatic_options:
            selected_layers = self.automatically_select_blocks(selected_layers)

        self.selected_layers = [f"model.layers.{layer}.mlp" for layer in selected_layers]

        def get_layer(layer_name):
            layer_idx = int(layer_name.split('.')[-2])
            if "GPTNeoXForCausalLM" in self.get_model().config.architectures:
                layer = self.get_model().gpt_neox.layers[layer_idx]
            else:
                layer = self.get_model().model.layers[layer_idx]

            if 'mlp' in layer_name:
                return layer.mlp
            elif 'self_attn' in layer_name:
                if "GPTNeoXForCausalLM" in self.get_model().config.architectures:
                    return layer.attention
                else:
                    return layer.self_attn
            elif 'residual' in layer_name:
                return layer

        # TODO: Add hooks
        for layer_name in self.selected_layers:
            if 'mlp' in layer_name:
                module = get_layer(layer_name)
            elif 'self_attn' in layer_name:
                module = get_layer(layer_name)
            elif 'residual' in layer_name:
                module = get_layer(layer_name)
            else:
                raise ValueError(f"Unsupported layer: {layer_name}")

            self.hooks[layer_name] = ActivationHook(module, layer_name)

    def automatically_select_blocks(self, blocks):
        num_layers = self.get_model().config.num_hidden_layers

        if blocks == "all":
            return [i for i in range(num_layers)]
        elif blocks == "last":
            return [num_layers - 1]
        elif blocks == "first":
            return [0]
        elif blocks == "middle":
            return [num_layers // 2]
        elif blocks == "last_2":
            return [num_layers - 1, num_layers - 2]
        elif blocks == "every_5th":
            return [i for i in range(0, num_layers, 5)] + [num_layers - 1]
        elif blocks == "first_and_last":
            return [0, num_layers-1]
        else:
            raise ValueError(f"Unsupported blocks: {blocks}")

    def to(self, device):
        self.get_model().to(device)
        self.hooks = {name: hook.to(device) for name, hook in self.hooks.items()}
        return self

    def extract_activation(self, text, hook, aggregation_calc, attention_mask=None):
        """Extracts and aggregates activation, handling batches and padding."""
        if hook.activations is None:
            return None

        activations_gpu = hook.activations # Shape: (batch_size, seq_len, hidden_dim)

        # Perform aggregation on the GPU tensor, respecting padding
        if aggregation_calc == "mean":
            if attention_mask is None:
                print("Warning: 'mean' aggregation without attention_mask might be inaccurate due to padding.")
                aggregated_gpu = torch.mean(activations_gpu, dim=1)
            else:
                # Mask out padding tokens before averaging
                masked_activations = activations_gpu * attention_mask.unsqueeze(-1).to(activations_gpu.dtype)
                # Sum over seq_len
                summed_activations = masked_activations.sum(dim=1)
                # Count non-padding tokens per sequence
                num_tokens = attention_mask.sum(dim=1).unsqueeze(-1)
                # Avoid division by zero for empty sequences (shouldn't happen with causal LMs)
                num_tokens = torch.max(num_tokens, torch.ones_like(num_tokens))
                aggregated_gpu = summed_activations / num_tokens

        elif aggregation_calc == "max":
             if attention_mask is None:
                print("Warning: 'max' aggregation without attention_mask might include padding.")
                aggregated_gpu = torch.max(activations_gpu, dim=1).values
             else:
                 # Set padding tokens to a very small number before taking max
                 padded_mask = (1.0 - attention_mask.unsqueeze(-1).to(activations_gpu.dtype)) * torch.finfo(activations_gpu.dtype).min
                 masked_activations = activations_gpu + padded_mask
                 aggregated_gpu = torch.max(masked_activations, dim=1).values

        elif aggregation_calc == "last":
            if attention_mask is None:
                 print("Warning: 'last' aggregation without attention_mask assumes no padding.")
                 aggregated_gpu = activations_gpu[:, -1, :]
            else:
                # Find the index of the last non-padding token for each sequence
                sequence_lengths = attention_mask.sum(dim=1) - 1 # Indices are 0-based
                batch_indices = torch.arange(activations_gpu.size(0), device=activations_gpu.device)
                sequence_lengths = sequence_lengths.to(activations_gpu.device)
                # Gather the activations at the last token index
                aggregated_gpu = activations_gpu[batch_indices, sequence_lengths, :]
        elif aggregation_calc == "all":
            aggregated_gpu = activations_gpu # Return the full batch * seq_len activations
        else:
            raise ValueError(f"Unknown aggregation_calc: {aggregation_calc}")

        # Move aggregated tensor to CPU and convert to numpy
        aggregated_cpu_numpy = aggregated_gpu.detach().cpu().numpy()
        return aggregated_cpu_numpy

    def extract_activations(self, texts, batch_size=8, aggregation_calc="last"):

        # ---> Create DistributedSampler if DDP is active <---
        sampler = DistributedSampler(texts, shuffle=False) if self.is_ddp else None # 
        
        dataloader = DataLoader(texts, batch_size=batch_size, sampler=sampler)

        all_activations = {}  # Reset activations dictionary
        # Process batches using dataloader
        for batch in dataloader:
            batch_texts = batch
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts, return_tensors='pt', padding=True, truncation=True,
                max_length=self.tokenizer.model_max_length # Adjust if needed
            ).to(self.model.device)

            # Run model forward pass for the batch
            with torch.no_grad():
                 # Need attention_mask for correct aggregation if padding exists
                outputs = self.model(**inputs, output_hidden_states=False, output_attentions=False) # We only need activations via hooks

            # Collect activations from hooks for this batch
            for layer_name, hook in self.hooks.items():
                if hook.activations is not None:
                    if all_activations.get(layer_name) is None:
                        all_activations[layer_name] = []
                        
                    # Pass attention_mask for correct aggregation
                    batch_aggregated_activations = self.extract_activation(
                        layer_name, hook, aggregation_calc, attention_mask=inputs.attention_mask
                    )
                    # Append results for each item in the batch
                    # batch_aggregated_activations shape depends on aggregation_calc
                    # If aggregated to (batch, hidden_dim), split it
                    if aggregation_calc in ["mean", "max", "last"]:
                         all_activations[layer_name].extend(list(batch_aggregated_activations))
                    elif aggregation_calc == "all":
                         # Handle (batch, seq, hidden) - maybe just append the whole tensor or split
                         all_activations[layer_name].append(batch_aggregated_activations) # Appends tensor (batch, seq, hidden)
                    else:
                         all_activations[layer_name].append(batch_aggregated_activations) # Appends tensor (batch, ...)

                    hook.clear() # Clear hook activations for the next batch
            del inputs, outputs # Free memory
            torch.cuda.empty_cache()

        if self.is_ddp:
            all_activations_gathered = {}
            for layer_name, activations in all_activations.items():
                activations_tensor = torch.tensor(activations).to(self.device)
                gathered_activations = [torch.zeros_like(activations_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_activations, activations_tensor)
                all_activations_gathered[layer_name] = torch.cat(gathered_activations, dim=0).cpu()
        else:
            all_activations_gathered = all_activations
        return all_activations_gathered

    def set_transformation_function(self, layer_name, transformation_function):
        self.hooks[layer_name].set_transformation_function(transformation_function)

    def get_transformation_function(self, layer_name):
        return self.hooks[layer_name].transformation_function

    def clear_transformation_functions(self):
        for hook in self.hooks.values():
            hook.clear()

    def generate(self, prompt: Union[str, List[str]], max_length: int = 100) -> Union[str, List[str]]:
        """
        Generates text based on a single prompt or a batch of prompts.

        Args:
            prompt (Union[str, List[str]]): A single prompt string or a list of prompt strings.
            max_length (int, optional): The maximum number of *new* tokens to generate. Defaults to 100.

        Returns:
            Union[str, List[str]]: The generated text (string) if the input was a single prompt,
                                   or a list of generated texts (list of strings) if the input was a list of prompts.
        """
        is_single_prompt = isinstance(prompt, str)

        # Ensure prompts is always a list for tokenization
        if is_single_prompt:
            prompts = [prompt]
        elif isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
            prompts = prompt
        else:
            raise TypeError("prompt must be a string or a list of strings")

        # Process input with optimized inference
        # Use padding=True for batching, truncation=True for safety
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,       # Pad sequences to the longest in the batch
            truncation=True,    # Truncate sequences if they exceed model max length
            max_length=self.get_model().config.max_position_embeddings - max_length # Ensure input fits
        ).to(self.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        input_seq_len = input_ids.shape[1] # Length of the padded input sequences

        # Use generate method of optimized model
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device == 'cuda'): # Enable AMP only for CUDA
            outputs = self.get_model().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                pad_token_id=self.tokenizer.pad_token_id # Use the tokenizer's pad token ID
                # You might add other generation parameters here (e.g., temperature, top_k, do_sample)
            )


        generated_sequences = outputs[:, input_seq_len:]

        # Use batch_decode for potentially faster decoding of multiple sequences
        generated_texts = self.tokenizer.batch_decode(
            generated_sequences,
            skip_special_tokens=True
        )

        # Clean up potentially leading/trailing whitespace
        generated_texts = [text.strip() for text in generated_texts]

        # Clean up GPU memory
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # Return single string or list based on input type
        if is_single_prompt:
            return generated_texts[0]
        else:
            return generated_texts

    def visualize_activations(self, texts, layer_name, aggregation_calc="last", save_path=None):
        activations = self.extract_activations(texts, aggregation_calc=aggregation_calc)

        if save_path is not None:
            plt.figure(figsize=(8, 6))
            sns.heatmap(activations[layer_name], annot=True, cmap='coolwarm', fmt=".2f",
                        )
            plt.title(f'{layer_name.upper()} Activations')
            plt.xlabel(f'{layer_name.upper()} Dimension 1')
            plt.ylabel(f'{layer_name.upper()} Dimension 2')
            plt.legend(title='Input Type')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(save_path)
            plt.close()

    def visualize_steering_vector(self, layer_name, k=20, save_path=None):
        if self.hooks[layer_name].transformation_function is None:
            print(f"No steering vector found for layer {layer_name}")
            return

        steering_vector = self.hooks[layer_name].transformation_function
        print(steering_vector)
        if save_path is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(steering_vector)
            plt.title(f'Steering Vector Analysis - Layer: {layer_name}')
            plt.xlabel(f'Dimension Index')
            plt.ylabel(f'Value')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(save_path)
            plt.close()

        return steering_vector

    def visualize_patched_saliency(self, texts, layer_name, save_path=None):

        if save_path is not None:
            plt.figure(figsize=(10, 6))
            sns.heatmap(activations[layer_name], annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1,
                        xticklabels=texts, yticklabels=texts, cbar=False)
            plt.title(f'{layer_name.upper()} Activations (Patched Saliency)')
            plt.xlabel(f'{layer_name.upper()} Dimension 1')
            plt.ylabel(f'{layer_name.upper()} Dimension 2')
            plt.legend(title='Input Type')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(save_path)
            plt.close()

    def visualize_token_saliency(self, texts, layer_name, aggregation_calc="last", save_path=None):
        activations = self.extract_activations(texts, aggregation_calc=aggregation_calc)
        if save_path is not None:
            plt.figure(figsize=(10, 6))
            sns.heatmap(activations[layer_name], annot=True, cmap='coolwarm', fmt=".2f", vmin=0, vmax=1,
                        xticklabels=texts, yticklabels=texts, cbar=False)
            plt.title(f'{layer_name.upper()} Activations (Token Saliency)')
            plt.xlabel(f'{layer_name.upper()} Dimension 1')
            plt.ylabel(f'{layer_name.upper()} Dimension 2')
            plt.legend(title='Input Type')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(save_path)
            plt.close()

    def plot_activation_projection(self, all_activations, layer_name, labels, method='tsne', perplexity=30):

        activations_np = np.array(all_activations[layer_name]).squeeze()


        print(f"Projecting activations for layer: {layer_name} (Shape: {activations_np.shape}) using {method.upper()}...")

        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, activations_np.shape[0] - 1)) # Perplexity must be less than n_samples
        else:
            raise ValueError(f"Unknown method: {method}")

        projections = reducer.fit_transform(activations_np)

        plt.figure(figsize=(8, 6))
        proj_len = len(projections[:, 1])
        proj_len2 = len(projections[:, 0])
        print(f"projections[:, 1] length: {proj_len}")
        print(f"projections[:, 0] length: {proj_len2}")
        print(f"labels length: {len(labels)}")
        sns.scatterplot(x=projections[:, 0], y=projections[:, 1], hue=labels, palette='viridis', s=50, alpha=0.8)
        plt.title(f'{method.upper()} Projection of Activations\nLayer: {layer_name}{title_suffix}')
        plt.xlabel(f'{method.upper()} Dimension 1')
        plt.ylabel(f'{method.upper()} Dimension 2')
        plt.legend(title='Input Type')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    def __exit__(self):
        self.clear_hooks()

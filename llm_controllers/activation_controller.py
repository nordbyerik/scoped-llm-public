from llm_controllers.llm_controller import LLMController
import torch
import numpy as np
import os
import gc

from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.activation_utils.steering_layer import SteeringLayer

class ActivationController(LLMController):
    def __init__(self, model, selected_layers='all', use_ddp=True, save_folder_path="."):
        super().__init__(model, use_ddp)

        self.save_folder_path = save_folder_path
        self.overwrite = True

        model = self.get_model()
        layers = self.get_model().model.layers

        for i in range(len(layers)):
            if not isinstance(layers[i], SteeringLayer):
                layers[i] = SteeringLayer(layers[i], i)

        automatic_options = ['all', "last", "first", "middle", "last_2", "every_5th", "first_and_last"]
        if selected_layers in automatic_options or "last_" in selected_layers:
            selected_layers = self.automatically_select_blocks(selected_layers)

        self.selected_layers=selected_layers

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
        elif "last_" in blocks:
            return [num_layers - 1 - i for i in range(int(blocks.split("_")[-1]))]
        elif blocks == "every_5th":
            return [i for i in range(0, num_layers, 5)] + [num_layers - 1]
        elif blocks == "first_and_last":
            return [0, num_layers-1]
        else:
            raise ValueError(f"Unsupported blocks: {blocks}")

    def to(self, device):
        self.get_model().to(device)
        return self

    def extract_activation(self, text, layer_number, aggregation_calc, attention_mask=None):
        """Extracts and aggregates activation, handling batches and padding."""

        response = self.get_model()(
            **self.tokenizer(text, return_tensors="pt").to(self.model.device),
            output_hidden_states=True,
        )

        activations_gpu = response.hidden_states[layer_number]

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

    def extract_activations(self, texts, batch_size=10, aggregation_calc="last", activation_name='neutral'):
        # ---> Create DistributedSampler if DDP is active <---
        sampler = DistributedSampler(texts, shuffle=False) if self.is_ddp else None # 
        dataloader = DataLoader(texts, batch_size=batch_size, sampler=sampler)
        
        all_activations = {}  # Reset activations dictionary
        layers_to_calculate = []
        # Load from mem any that have already been completed
        for layer_number in self.selected_layers:
            if activation_name is None:
                layers_to_calculate.append(layer_number)
                continue

            path = os.path.join(self.save_folder_path, f"layer_{layer_number}_{activation_name}.pt")
            if os.path.exists(path) and not self.overwrite:
                all_activations[layer_number] = torch.load(path)
            else:
                layers_to_calculate.append(layer_number)


        if len(layers_to_calculate) == 0:
            return all_activations

        # Process batches using dataloader
        for batch_texts in dataloader:

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts['question'], return_tensors='pt', padding=True, truncation=True,
                max_length=10000 # Adjust if needed
            ).to(self.model.device)

            # Run model forward pass for the batch
            with torch.no_grad():
                 # Need attention_mask for correct aggregation if padding exists
                hidden_out = True
                if "mlp" in self.selected_layers:
                    hidden_out = True
                
                attention_out = False
                if "self_attn" in self.selected_layers:
                    attention_out = False

                outputs = self.model(**inputs, output_hidden_states=hidden_out, output_attentions=attention_out) # We only need activations via hooks
            
            torch.cuda.synchronize()
            # Collect activations from hooks for this batch
            for layer_number in layers_to_calculate:
                batch_hidden = outputs.hidden_states[layer_number]
                if all_activations.get(layer_number) is None:
                    all_activations[layer_number] = []
                
                if aggregation_calc == "mean":
                    result = torch.mean(batch_hidden, dim=0)
                elif aggregation_calc == "max":
                    result = torch.max(batch_hidden, dim=0)[0]
                elif aggregation_calc == "last":
                    result = batch_hidden[:, -1, :]
                elif aggregation_calc == "all":
                    result = batch_hidden
                else:
                    result = batch_hidden[:, -1, :] # Appends tensor (batch, ...)

                all_activations[layer_number].append(result)

                path = os.path.join(self.save_folder_path, f"layer_{layer_number}.pt")
                if not os.path.exists(self.save_folder_path):
                    os.mkdir(self.save_folder_path)

                gc.collect()
                torch.cuda.empty_cache()

            del inputs, outputs # Free memory
            torch.cuda.empty_cache()

        if self.is_ddp:
            all_activations_gathered = {}
            for layer_name, activations in all_activations.items():
                activations_tensor = torch.stack(activations, dim=0).to(self.device)
                gathered_activations = [torch.zeros_like(activations_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_activations, activations_tensor)
                all_activations_gathered[layer_name] = torch.cat(gathered_activations, dim=0)
        else:
            all_activations_gathered = all_activations
        
        for layer_name in all_activations_gathered:
            if layer_name in layers_to_calculate:
                all_activations_gathered[layer_name] = torch.vstack(all_activations_gathered[layer_name])

            if activation_name is None:
                continue
            path = os.path.join(self.save_folder_path, f"layer_{layer_name}_{activation_name}.pt")
            if not os.path.exists(path):
                torch.save(all_activations_gathered[layer_name], path)
            
        return all_activations_gathered

    def set_transformation_function(self, layer_name, transformation_function):
        self.get_model().model.layers[layer_name].set_transformation_function(transformation_function)

    def clear_transformation_functions(self):
        for layer in self.get_model().model.layers:
            layer.clear_layer()

    def generate(self, prompt, max_length=100, coeff=1.0, vector_types=["all"], selected_layers=None):
        # Clear to ensure not prior transformation functions
        self.clear_transformation_functions()

        if selected_layers is None:
            selected_layers = self.selected_layers
            
        for layer_name in selected_layers:
            layer_vector = None
            for vector_type in vector_types:
                if layer_vector is None:
                    layer_vector = torch.zeros_like(self.steering_vectors[vector_type][layer_name])
                layer_vector += self.steering_vectors[vector_type][layer_name]

            layer_vector = layer_vector / (layer_vector.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            def transformation_func(activation_tensor, vec=layer_vector, c=coeff, do_norm=True):
                orig_norm = activation_tensor.norm(dim=-1, keepdim=True)
                activation_tensor = activation_tensor + torch.Tensor(c * vec).to(activation_tensor.dtype).to(activation_tensor.device)
                new_norm = activation_tensor.norm(dim=-1, keepdim=True)
                norm_ratio = new_norm / orig_norm

                if norm_ratio.max().item() > 1 and do_norm:
                    activation_tensor = activation_tensor * (norm_ratio) 

                return activation_tensor
            self.set_transformation_function(layer_name, transformation_func)
        return super().generate(prompt, max_length=max_length)
    
    def generate_uncontrolled(self, prompt, max_length=100):
        self.clear_transformation_functions()
        return super().generate(prompt, max_length=max_length)

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

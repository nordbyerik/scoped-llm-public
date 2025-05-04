from ..activation_controller import ActivationController
import numpy as np
import gc
import torch
from tqdm.notebook import tqdm

class ActAddSteerer(ActivationController):
    def __init__(self, model, selected_layers='last_5', use_ddp=True, save_folder_path="."):
        super().__init__(model, selected_layers, use_ddp, save_folder_path)
        self.steering_vectors = {}

    def __call__(self, prompts, coeff=1.0):
        return self.generate(prompts, coeff=coeff)

    def extract_average_diff(self, positive_texts, negative_texts, batch_size=1, aggregation_calc="last"):
        positive_activations = self.extract_activations(positive_texts, batch_size=batch_size, aggregation_calc=aggregation_calc, activation_name='positive')
        negative_activations = self.extract_activations(negative_texts, batch_size=batch_size, aggregation_calc=aggregation_calc, activation_name='negative')

        average_diff_activations = {}
        all_layers = set(positive_activations.keys()) | set(negative_activations.keys())

        for layer_name in all_layers:
            pos_act = positive_activations.get(layer_name)
            neg_act = negative_activations.get(layer_name)

            # Calculate mean activation for positive and negative sets
            mean_pos = torch.mean(pos_act, axis=0)
            mean_neg = torch.mean(neg_act, axis=0)
            average_diff_activations[layer_name] = mean_pos - mean_neg

        # Clean up large activation dicts
        del positive_activations, negative_activations
        gc.collect()
        torch.cuda.empty_cache()

        return average_diff_activations

    # TODO: Allow this to store different steering vectors and combine them
    def train(self, positive_prompts, negative_prompts, batch_size=8, aggregation_calc="last", vector_type="all"):
        self.steering_vectors[vector_type] = self.extract_average_diff(positive_prompts.data, negative_prompts.data, batch_size, aggregation_calc)
        return self.steering_vectors
    
    def generate(self, prompts, coeff=1):
        if isinstance(prompts, str):
            prompts = [prompts]
        
        self.clear_transformation_functions()
        selected_layers = self.selected_layers
        
        for layer_name in selected_layers:
            layer_vector = None
            for vector_type in ['all']:
                if layer_vector is None:
                    layer_vector = torch.zeros_like(self.steering_vectors[vector_type][layer_name])
                layer_vector += self.steering_vectors[vector_type][layer_name]
            
            # Normalize and move to CPU until needed
            layer_vector = layer_vector / (layer_vector.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            layer_vector_cpu = layer_vector.cpu()  # Store on CPU
            
            def transformation_func(activation_tensor, layer_vector=layer_vector_cpu, c=coeff, do_norm=True):
                # Store original norm
                orig_norm = activation_tensor.norm(dim=-1, keepdim=True)
                
                # Move vector to same device and dtype as activation_tensor
                with torch.no_grad():  # Prevent building computation graph for this operation
                    vec_device = layer_vector.to(device=activation_tensor.device, dtype=activation_tensor.dtype)
                    
                    # Perform operation in-place to avoid creating a new tensor
                    activation_tensor.add_(vec_device * c)
                    
                    # Normalize if needed
                    if do_norm:
                        new_norm = activation_tensor.norm(dim=-1, keepdim=True)
                        norm_ratio = new_norm / orig_norm
                        # Only apply normalization if needed
                        if norm_ratio.max().item() > 1:
                            activation_tensor.mul_(orig_norm / new_norm)
                
                # Don't need these calls on every activation
                # gc.collect()
                # torch.cuda.empty_cache()
                
                return activation_tensor
            
            self.set_transformation_function(layer_name, transformation_func)
        
        # Do cleanup once after setting up all transformations
        torch.cuda.empty_cache()
        
        all_responses = []
        # Process prompts in batches
        for prompt in prompts:
            prompt = [prompt]
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
                responses = self.model(**inputs)
            all_responses.append(responses)
        
        # Clean up after all processing
        gc.collect()
        torch.cuda.empty_cache()
        
        return responses

    def load(self, filepath):
        self.steering_vectors = torch.load(filepath, weights_only=False)

    def save(self, path_string):
        torch.save(self.steering_vectors, path_string)





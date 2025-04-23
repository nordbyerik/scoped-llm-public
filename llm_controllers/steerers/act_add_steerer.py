from ..activation_controller import ActivationController
import numpy as np
import gc
import torch
from tqdm.notebook import tqdm

class ActAddSteerer(ActivationController):
    def __init__(self, model, selected_layers=None, use_ddp=True, save_folder_path="."):
        super().__init__(model, selected_layers, use_ddp, save_folder_path)
        self.steering_vectors = {}


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
        self.steering_vectors[vector_type] = self.extract_average_diff(positive_prompts, negative_prompts, batch_size, aggregation_calc)
        return self.steering_vectors
    
    def load(self, filepath):
        self.steering_vectors = torch.load(filepath, weights_only=False)

    def save(self, path_string):
        torch.save(self.steering_vectors, path_string)





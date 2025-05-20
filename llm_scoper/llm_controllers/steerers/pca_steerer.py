from ..activation_controller import ActivationController
import numpy as np
import gc
import torch
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA


class PCASteerer(ActivationController):
    def __init__(self, model, selected_layers=None, use_ddp=True, save_folder_path="."):
        super().__init__(model, selected_layers, use_ddp, save_folder_path)
        self.steering_vectors = {}

    # Inside the PCASteerer class
    def extract_pca_activations(self, positive_texts, negative_texts, batch_size=8, aggregation_calc="last"):
        positive_activations = self.extract_activations(positive_texts, batch_size=batch_size, aggregation_calc=aggregation_calc, activation_name='positive')
        negative_activations = self.extract_activations(negative_texts, batch_size=batch_size, aggregation_calc=aggregation_calc, activation_name='negative')

        average_diff_activations = {}
        all_layers = set(positive_activations.keys()) | set(negative_activations.keys())

        vector_directions = {}
        for layer_name in all_layers:
            layer_diff = positive_activations[layer_name] - negative_activations[layer_name]
            pca = PCA(n_components=1, whiten=False).fit(layer_diff.squeeze().cpu())
            vector_directions[layer_name] = torch.Tensor(pca.components_.astype(np.float32).squeeze(axis=0)).to(self.get_model().device)

        # Clean up large intermediate sums immediately
        torch.cuda.empty_cache()
        gc.collect()
        
        return vector_directions

    # TODO: Allow this to store different steering vectors and combine them
    def train(self, positive_prompts, negative_prompts, batch_size=8, aggregation_calc="last", vector_type="all"):
        self.steering_vectors[vector_type] = self.extract_pca_activations(positive_prompts, negative_prompts, batch_size, aggregation_calc)
        return self.steering_vectors

    def load(self, filepath):
        self.steering_vectors = torch.load(filepath, weights_only=False)

    def save(self, filepath):
        torch.save(self.steering_vectors, filepath) 




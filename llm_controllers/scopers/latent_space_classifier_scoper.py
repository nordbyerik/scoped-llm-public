import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Union

from llm_controllers.activation_controller import ActivationController # TODO: Move Activation Steering out of steerers

from sklearn.model_selection import train_test_split

import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


import torch
import gc
import inspect
import sys



class ScopeClassifier(ActivationController):
    def __init__(self, model, selected_layers='all', save_folder_path='scoping_vectors'):
        super().__init__(model, selected_layers=selected_layers, save_folder_path=save_folder_path)

        self.best_layer = None
        self.best_model = None
        self.classifier = None
    
    def __call__(self, prompts):
        return self.generate(prompts)

    def train_linear_probe(self, positive_texts: List[str], negative_texts: list[str]):
        positive_activations = self.extract_activations(positive_texts)
        negative_activations = self.extract_activations(negative_texts)

        results = {}
        best_accuracy = 0
        best_layer = ""
        best_model = None

        layer_coeffs = {}

        for layer_name in positive_activations:
            positive_examples = np.array(positive_activations[layer_name].cpu())
            negative_examples = np.array(negative_activations[layer_name].cpu())
            print("Positive Examples Lne", len(positive_examples))
            print("Negative Examples Lne", len(negative_examples))

            labels = [1] * len(positive_examples) + [0] * len(negative_examples)

            all_activations = list(positive_examples) + list(negative_examples)
            all_activations = np.array(all_activations).squeeze()

            X_train, X_test, y_train, y_test = train_test_split(
                all_activations, labels, test_size=0.3, random_state=42
            )

            # Train classifier
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[layer_name] = report['accuracy']

            layer_coeffs[layer_name] = classifier.coef_[0]

            if report['accuracy'] > best_accuracy:
                best_accuracy = report['accuracy']
                best_layer = layer_name
                best_model = classifier
            
            del all_activations


        # Visualize results
        plt.figure(figsize=(10, 6))
        layers = list(results.keys())
        accuracies = [results[layer] for layer in layers]

        plt.bar(range(len(results)), accuracies)
        plt.xlabel('Layer')
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy by Layer')
        plt.xticks(range(len(results)), [f"Layer {layer} {layer}" for layer in layers])
        plt.tight_layout()
        plt.savefig('steering_output/layer_accuracies.png')
        print("Results visualization saved to 'steering_output/layer_accuracies.png'")

        print("Best model", best_model.coef_)
        self.best_model = best_model
        self.best_layer = best_layer
        self.classifier = best_model
        return best_layer, best_model

    def train(self, in_domain: Dataset, out_of_domain: Dataset, batch_size=10):
        self.best_layer, self.classifier = self.train_linear_probe(in_domain.data, out_of_domain.data)

    def generate(self, prompts: Union[str, List[str]]):
        if isinstance(prompts, str):
            prompts = [prompts]
        activations = self.extract_activations(prompts, batch_size=1, aggregation_calc="last", activation_name=None)
        classification = self.best_model.predict(activations[self.best_layer].cpu())

        del activations
        torch.cuda.empty_cache()

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        responses = self.model(**inputs)

        
        original_type = type(responses.logits)
        device = responses.logits.device
        logits_array = np.array(responses.logits.detach().cpu())
        for i, c in enumerate(classification):
            if c != 1:
                logits_array[i] = torch.Tensor([-1])
        responses.logits = original_type(torch.Tensor(logits_array).to(device))
        
        return responses



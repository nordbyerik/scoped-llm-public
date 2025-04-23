import torch
import numpy as np

from llm_controllers.activation_controller import ActivationController # TODO: Move Activation Steering out of steerers

from sklearn.model_selection import train_test_split

import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

class ScopeClassifier(ActivationController):
    def __init__(self, model, selected_layers=None, use_ddp=True):
        super().__init__(model, selected_layers, use_ddp)
        self.classifier = None
        self.best_layer = None

    def train_classifier(self, positive_prompts, negative_prompts, batch_size=8, aggregation_calc="last", vector_type="all"):
        
        positive_activations = self.extract_activations(positive_prompts, batch_size=batch_size, aggregation_calc=aggregation_calc, activation_name='positive')
        negative_activations = self.extract_activations(negative_prompts, batch_size=batch_size, aggregation_calc=aggregation_calc, activation_name='negative')

        texts = positive_activations + negative_activations
        labels = [1] * len(positive_activations) + [0] * len(negative_activations)

        all_activations = self.extract_activations(texts)

        # Train and evaluate classifiers for each layer
        results = {}
        best_accuracy = 0
        best_layer = ""
        best_model = None
        for layer_name, activations in all_activations.items():
            activations_array = np.array(activations)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                activations_array, labels, test_size=0.3, random_state=42
            )

            # Train classifier
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(X_train, y_train)

            # Evaluate
            y_pred = classifier.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            results[layer_name] = report['accuracy']

            if report['accuracy'] > best_accuracy:
                best_accuracy = report['accuracy']
                best_layer = layer_name
                best_model = classifier

        return best_layer, best_model

    # TODO: Allow this to store different steering vectors and combine them
    def train(self, positive_prompts, negative_prompts, batch_size=8, aggregation_calc="last"):
        self.selected_layers, self.best_model = self.train_classifier(positive_prompts, negative_prompts, batch_size, aggregation_calc)
        return (self.best_layer, self.best_model)
    
    def load(self, filepath):
        best_stuff = torch.load(filepath, weights_only=False)
        self.selected_layers, self.best_model = best_stuff

    def save(self, path_string):
        torch.save((self.best_layer, self.best_model), path_string)

    def generate(self, prompt, max_length=100):
        # Clear to ensure not prior transformation functions
        self.clear_transformation_functions()

        activations = self.extract_activations(prompt, batch_size=1, aggregation_calc="last", activation_name=None)
        classification = self.best_model.predict(activations)

        return super().generate(prompt, max_length=max_length)


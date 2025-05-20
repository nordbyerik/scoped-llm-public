from typing import List
from llm_controllers.activation_controller import ActivationController
from llm_controllers.llm_controller import LLMController
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class LinearProbeSteerer(ActivationController):
    def __init__(self, model, selected_layers='all'):
        super().__init__(model, selected_layers=selected_layers)

        self.best_layer = None
        self.best_model = None
        self.classifier = None

    def train_linear_probe(self, positive_texts, negative_texts):
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

            labels = [1] * len(positive_examples) + [0] * len(positive_examples)

            all_activations = list(positive_examples) + list(negative_examples)
            all_activations = np.array(all_activations)

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


        # Visualize results
        plt.figure(figsize=(10, 6))
        layers = list(results.keys())
        accuracies = [results[layer] for layer in layers]

        plt.bar(range(len(results)), accuracies)
        plt.xlabel('Layer')
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy by Layer')
        plt.xticks(range(len(results)), [f"Layer {layer.split('.')[-2]} {layer.split('.')[-1]}" for layer in layers])
        plt.tight_layout()
        plt.savefig('steering_output/layer_accuracies.png')
        print("Results visualization saved to 'steering_output/layer_accuracies.png'")

        print("Best model", best_model.coef_)
        self.best_model = best_model
        self.best_layer = best_layer
        self.classifier = best_model
        return best_layer, best_model

    def train(self, in_domain, out_of_domain, batch_size=10):
        self.best_layer, self.classifier = self.train_linear_probe(in_domain, out_of_domain)

    def set_transformation_function(self, c=1, func_type='multiply', layers="all"):
        if layers == "best":
            layers = [self.best_layer]
        elif layers == "all":
            layers = self.selected_layers
        

        for layer in layers:
            vector_gpu = torch.tensor(self.classifier.coef_[0], dtype=torch.float32)
            vector_gpu = vector_gpu.to(self.device)

            def mult_func(activation_tensor, vec=vector_gpu, c=c):
                return activation_tensor * (1 + (vec.to(activation_tensor.dtype ) * c))

            def add_func(activation_tensor, vec=vector_gpu, c=c):
                return activation_tensor + (vec.to(activation_tensor.dtype ) * c)

            if func_type == "add":
                return super().set_transformation_function(self.best_layer, add_func)
            elif func_type == "multiply":
                return super().set_transformation_function(self.best_layer, mult_func)
            else:
                raise ValueError(f"Unsupported function type: {func_type}")
        

import copy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

class SimpleMLPProbe(nn.Module):
    """A simple linear probe for binary classification."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Linear layer mapping activation dimension to a single logit
        self.linear = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # Output raw logits (suitable for BCEWithLogitsLoss)
        return self.linear(x)
    
    def to(self, device):
        self.linear = self.linear.to(device)
        return super().to(device)
    

class TorchModelSteerer(ActivationController):
    def __init__(self, 
                 model, 
                 selected_layers=None,
                 use_ddp=True,
                 save_folder_path=".",
                 # Probe & Training Hyperparameters
                 learning_rate=0.001, 
                 epochs=10, 
                 batch_size=32,
                 probe_type='mlp',
                 model_config={'hidden_dim': 128, 'output_dim': 128, 'dropout_prob': 0.1}):
        super().__init__(model, selected_layers=selected_layers, use_ddp=use_ddp, save_folder_path=save_folder_path)

        self.model_name = model

        self.best_layer = None
        self.best_model_state_dict = None
        self.best_model_input_dim = None
        self.best_probe_type = None # Store the type of the best probe

        # Store hyperparameters
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.probe_type = probe_type
        self.model_config = model_config

        self.layer_configs = {}
        self.layer_state_dicts = {}

    def train_classifier(self, positive_texts, negative_texts, batch_size=1, model_name=''):
        # Ensure the target device is a torch device

        # Extract activations (assuming this returns a dict layer_name -> list of numpy arrays)
        # Make sure activations are extracted correctly (e.g., mean pooled per text)
        positive_activations = self.extract_activations(positive_texts, batch_size, activation_name='positive') # Keep extraction device potentially different
        negative_activations = self.extract_activations(negative_texts, batch_size, activation_name='negative')
        
        results = {}
        layer_state_dicts = {}
        layer_configs = {}

        best_accuracy = 0
        best_layer = ""
        best_model_state_dict = None
        best_model_input_dim = None

        for layer_name in positive_activations:

            # Assuming list contains numpy arrays ready to be stacked
            positive_examples = positive_activations[layer_name]
            negative_examples = negative_activations[layer_name]

            activation_dim = positive_examples.shape[-1]

            # --- Data Preparation ---
            # Correct label creation
            labels_pos = torch.ones((len(positive_examples), 1))
            labels_neg = torch.zeros((len(positive_examples), 1)) 

            all_activations_np = torch.concat((positive_examples, negative_examples))
            all_activations_np = all_activations_np.squeeze()
            all_labels_np = torch.concat((labels_pos, labels_neg))

            # Split data
            X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
                all_activations_np, all_labels_np, test_size=0.3, random_state=42, stratify=all_labels_np
            )

            # Convert to PyTorch Tensors
            X_train = torch.tensor(X_train_np, dtype=torch.float32)
            y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1) # Add dim for BCEWithLogitsLoss
            X_test = torch.tensor(X_test_np, dtype=torch.float32)
            y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

            # Create DataLoaders
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            # --- Model, Loss, Optimizer ---
            probe_model = SimpleMLPProbe(activation_dim, hidden_dim=self.model_config['hidden_dim'], output_dim=self.model_config['output_dim']).to(self.device)
            criterion = nn.BCEWithLogitsLoss() # Numerically stable
            optimizer = optim.Adam(probe_model.parameters(), lr=self.lr)

            # --- Training Loop ---
            probe_model.train()
            for epoch in range(self.epochs):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = probe_model(batch_X)
                    batch_y = batch_y.squeeze()
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                # print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

            # --- Evaluation ---
            probe_model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = probe_model(batch_X)
                    # Convert logits to probabilities then to predictions (0 or 1)
                    preds = (torch.sigmoid(outputs) > 0.5).cpu().long().squeeze().numpy()
                    all_preds.extend(preds.tolist())
                    all_targets.extend(batch_y.cpu().long().squeeze().numpy().tolist())

            accuracy = accuracy_score(all_targets, all_preds)
            results[layer_name] = accuracy
            layer_state_dicts[layer_name] = copy.deepcopy(probe_model.state_dict())
            layer_configs[layer_name] = {'input_dim': activation_dim, **self.model_config}


            # --- Update Best Model ---
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_layer = layer_name
                # Save the state_dict and input dim
                best_model_state_dict = copy.deepcopy(probe_model.state_dict())
                best_model_input_dim = activation_dim


        # --- Visualization (remains the same) ---
        plt.figure(figsize=(10, 6))
        layers = list(results.keys())
        accuracies = [results[layer] for layer in layers]

        plt.bar(range(len(results)), accuracies)
        plt.xlabel('Layer')
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy by Layer (PyTorch Probe)')
        # Adjust labels if needed, e.g., rotate
        plt.xticks(range(len(results)), [f"L {layer}" for layer in layers], rotation=45, ha='right')
        plt.tight_layout()
        replaced_string = self.model_name.replace("/", "_").replace('.', '_')
        plt.savefig(f'steering_output/layer_accuracies_pytorch_{replaced_string}.png')
        print("Results visualization saved to 'steering_output/layer_accuracies_pytorch.png'")

        # Store best model info in the instance
        self.best_model_input_dim = best_model_input_dim
        self.best_layer = best_layer

        self.layer_configs = layer_configs
        self.layer_state_dicts = layer_state_dicts

        # Return layer name and best accuracy (or model info if needed)
        return self.best_layer, best_accuracy

    def train(self, positive_texts, negative_texts, batch_size=1):
        return self.train_classifier(positive_texts, negative_texts)

    @torch.no_grad() # Disable gradients for the main generation loop
    def generate(self, prompt, max_length=100, batch_size=1, coeff=1.0, vector_types=["all"], selected_layers=None):
        # Clear to ensure not prior transformation functions
        self.clear_transformation_functions()

        if type(prompt) == str:
            prompt = [prompt]
        activations = self.extract_activations(prompt, batch_size=batch_size)

        if selected_layers == 'best':
            selected_layers = [self.best_layer]
        elif selected_layers == 'all':
            selected_layers = self.selected_layers
        elif selected_layers is None:
            selected_layers = self.selected_layers
            
        for layer_name in selected_layers:
            layer_vector = None

            layer_config = self.layer_configs[layer_name]

            # Recreate the probe model
            probe_model = SimpleMLPProbe(
                input_dim=layer_config['input_dim'],
                hidden_dim=layer_config['hidden_dim'], 
                output_dim=layer_config['output_dim']
            ).to(self.device)

            probe_model.load_state_dict(self.layer_state_dicts[layer_name])
            probe_model.eval()

            # Calculate steering vector
            with torch.enable_grad():
                h_t = torch.Tensor(activations[layer_name].squeeze()).to(self.device).to(torch.float32).requires_grad_(True)
                logits_probe = probe_model(h_t).requires_grad_(True)
                criterion = nn.BCEWithLogitsLoss()
                # Ensure target label is float and on correct device
                target = torch.full_like(logits_probe, float(1), device=self.device)
                loss = criterion(logits_probe, target)

            # Backward pass to get gradient w.r.t. h_t
            loss.backward()
            grad = h_t.grad

            # TODO: Why norm? 
            layer_vector = grad / (grad.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            def transformation_func(activation_tensor, vec=layer_vector, c=coeff, do_norm=True):
                orig_norm = activation_tensor.norm(dim=-1, keepdim=True)
                activation_tensor = activation_tensor + torch.Tensor(c * vec).to(activation_tensor.dtype).to(activation_tensor.device)
                new_norm = activation_tensor.norm(dim=-1, keepdim=True)
                norm_ratio = new_norm / orig_norm

                if norm_ratio.max().item() > 1 and do_norm:
                    activation_tensor = activation_tensor / (norm_ratio) 

                return activation_tensor
            self.set_transformation_function(layer_name, transformation_func)

        return LLMController.generate(self, prompt, max_length=max_length)


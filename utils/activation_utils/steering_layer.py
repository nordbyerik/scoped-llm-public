import torch
import torch.nn as nn
from typing import List
from collections import defaultdict

def get_layer_device(layer: nn.Module) -> torch.device | str:
    try:
        return next(layer.parameters()).device
    except StopIteration:
        # Layer might have no parameters (e.g., nn.ReLU, nn.Dropout)
        # Check buffers as fallback, though often not device-specific unless registered
        try:
             return next(layer.buffers()).device
        except StopIteration:
             return "cpu (no parameters/buffers)" # Or some indicator

class SteeringLayer(nn.Module):
    behavior_layers = None

    def __init__(self, layer: nn.Module, layer_id: int) -> None:
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.use_ooi_preventive_normalization = False
        self.is_multi_steering = False
        self.transformation_function = None

    def set_transformation_function(self, transformation_function) -> None:
        """
        Configure steering for this layer.

        Args:
            behavior_vector: The behavior vector to apply.
            threshold: The threshold for condition activation.
            use_ooi_preventive_normalization: Whether to use OOI preventive normalization.
            condition_comparator_threshold_is: How to compare the condition to the threshold.
            condition_threshold_comparison_mode: How to compute the condition value.
        """
        self.is_multi_steering = False
        self.transformation_function = transformation_function


    def forward(self, hidden_states, *args, **kwargs):
        """
        Perform a forward pass through this layer, applying steering if configured.

        Args:
            hidden_states: The input hidden states.
            *args: Additional positional arguments for the underlying layer.
            **kwargs: Additional keyword arguments for the underlying layer.

        Returns:
            The output of the underlying layer, potentially modified by steering.
        """

        # original_norm = hidden_states.norm(dim=-1, keepdim=True)

        if not self.is_multi_steering:
            self._apply_single_behavior(hidden_states)
        else:
            self._apply_multi_behaviors(hidden_states)
        
        return self.layer(hidden_states, *args, **kwargs)

    def _apply_single_behavior(self, hidden_states):
        """
        Apply a single behavior vector to the hidden states.

        Args:
            hidden_states: The hidden states to modify.
        """
        if self.transformation_function is not None:
            hidden_states[0] = self.transformation_function(hidden_states[0])

    def _apply_multi_behaviors(self, hidden_states):
        """
        Apply multiple behavior vectors to the hidden states based on rules.

        Args:
            hidden_states: The hidden states to modify.
        """
        for rule in self.rules:
            behavior_index = int(rule.split('B')[1]) - 1
            if SteeringLayer.behavior_layers[behavior_index][self.layer_id]:
                
                hidden_states[0] = self.transformation_function(hidden_states[0])

    def _apply_ooi_normalization(self, hidden_states, original_norm):
        new_norm = hidden_states.norm(dim=-1, keepdim=True)
        max_ratio = (new_norm / original_norm).max().item()
        has_nan_inf = torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any()

        if max_ratio > 1 or has_nan_inf:
            hidden_states = hidden_states * (original_norm / new_norm)

        return hidden_states

    def clear_layer(self) -> None:
        self.transformation_function = None

class ActivationHook:
    """Hook to extract activations from a specific layer of a transformer model."""
    def __init__(self, module, layer_name, transformation_function=None):
        self.activations = None
        self.layer_name = layer_name
        self.transformation_function = transformation_function if transformation_function is not None else lambda x: x

        # Register forward hoo_k
        if "mlp" in self.layer_name:
            module.register_forward_hook(self.hook_mlp)
        elif "self_attn" in self.layer_name:
            module.register_forward_hook(self.hook_attention)
        elif "residual" in self.layer_name:
            module.register_forward_hook(self.hook_residual)
        else:
            raise ValueError(f"Unsupported layer: {layer_name}")

    def set_transformation_function(self, transformation_function):
        self.transformation_function = transformation_function

    def hook_mlp(self, module, input, output):
        self.activations = output.detach()
        return self.transformation_function(output)

    def hook_attention(self, module, input, output):
        self.activations = output[0].detach()
        output_updated = self.transformation_function(output[0])
        output = (output_updated,) + output[1:]
        return output

    def hook_residual(self, module, input, output):
        self.activations = output[0].detach()
        output_updated = self.transformation_function(output[0])
        output = (output_updated,) + output[1:]
        return output

    def print_activations(self):
        print(self.activations)

    def clear_activations(self):
        self.activations = None

    def clear_transformation_function(self):
        self.transformation_function = lambda x: x

    def clear(self):
        self.clear_activations()
        self.clear_transformation_function()

    def to(self, device):
        self.device = device
        return self

    def __exit__(self):
        self.clear()



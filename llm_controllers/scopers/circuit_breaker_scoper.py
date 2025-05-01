import torch
import numpy as np
import os
from typing import List, Union
from torch.utils.data import DataLoader, Dataset
from llm_controllers.llm_controller import LLMController
from peft import LoraConfig, get_peft_model
from torch.nn.functional import cosine_similarity

class CircuitBreakerScoper(LLMController):
    def __init__(self, model, selected_layers='last_5', save_folder_path='scoping_vectors'):
        super().__init__(model, False)
        self.save_folder_path=save_folder_path
        self.selected_layers = self.automatically_select_blocks(selected_layers)
        self.target_layers = self._get_target_layers()
        self.lora_model = None
        self.original_model = self.model
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
    def _get_target_layers(self):
        """Convert selected_layers to actual layer indices"""
        if self.selected_layers == 'all':
            return list(range(self.model.config.num_hidden_layers))
        else:
            return [int(layer) for layer in self.selected_layers]
    
    def __call__(self, prompts):
        return self.generate(prompts)
    
    def train_circuit_breakers(self, positive_texts: List[str], negative_texts: List[str], 
                              lora_r=8, lora_alpha=16, lora_dropout=0.05, 
                              batch_size=2, num_epochs=3, learning_rate=1e-4):
        """
        Train circuit breaker model using LoRA based on positive and negative examples.
        
        Args:
            positive_texts: List of texts that represent in-domain examples
            negative_texts: List of texts that we want to circuit-break (out-of-domain)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout rate
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        device = self.model.device
        
        # Determine target modules for LoRA
        # This depends on the model architecture - for most transformer models:
        if hasattr(self.model, 'model'):
            # For models like GPT, etc.
            if hasattr(self.model.model, 'layers'):
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                lora_target_modules = ["query", "key", "value", "dense"]
        else:
            # For other models
            lora_target_modules = ["query", "key", "value", "dense"]
            
        # Configure LoRA
        transform_layers = ','.join(str(layer) for layer in self.target_layers)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            layers_to_transform=self.target_layers,
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to the model
        self.lora_model = get_peft_model(self.model, lora_config)
        self.lora_model.print_trainable_parameters()
        
        # Prepare datasets
        from torch.utils.data import Dataset, DataLoader
        
        class CircuitBreakerDataset(Dataset):
            def __init__(self, tokenizer, positive_texts, negative_texts):
                self.tokenizer = tokenizer
                self.positive_texts = positive_texts
                self.negative_texts = negative_texts
                
            def __len__(self):
                return len(self.positive_texts)
            
            def __getitem__(self, idx):
                # Get positive example
                retain_text = self.positive_texts[idx % len(self.positive_texts)]
                retain_encodings = self.tokenizer(retain_text, return_tensors="pt", padding="max_length", 
                                                 truncation=True, max_length=512)
                
                # Get negative example (circuit breaker example)
                cb_idx = idx % len(self.negative_texts)
                cb_text = self.negative_texts[cb_idx]
                cb_encodings = self.tokenizer(cb_text, return_tensors="pt", padding="max_length", 
                                             truncation=True, max_length=512)
                
                # Get validation example (could be another positive, or a mix)
                val_idx = (idx + 1) % len(self.positive_texts)
                val_text = self.positive_texts[val_idx]
                val_encodings = self.tokenizer(val_text, return_tensors="pt", padding="max_length", 
                                              truncation=True, max_length=512)
                
                # Prepare batch
                batch = {
                    "input_ids": retain_encodings["input_ids"].squeeze(0),
                    "attention_mask": retain_encodings["attention_mask"].squeeze(0),
                    "input_ids_circuit_breaker": cb_encodings["input_ids"].squeeze(0),
                    "attention_mask_circuit_breaker": cb_encodings["attention_mask"].squeeze(0),
                    "input_ids_val": val_encodings["input_ids"].squeeze(0),
                    "attention_mask_val": val_encodings["attention_mask"].squeeze(0),
                }
                
                return batch
        
        # Create dataset and dataloader
        train_dataset = CircuitBreakerDataset(self.tokenizer, positive_texts, negative_texts)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.AdamW(self.lora_model.parameters(), lr=learning_rate)
        
        # Training loop
        self.lora_model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Compute loss similar to the compute_loss function in your code
                loss = self._compute_circuit_breaker_loss(batch, epoch=epoch, num_epochs=num_epochs)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
            
  
        # Save the trained model
        output_dir = os.path.join(self.save_folder_path, "circuit_breaker_lora")
        os.makedirs(output_dir, exist_ok=True)
        self.lora_model.save_pretrained(output_dir)
        print(f"Circuit breaker model saved to {output_dir}")
        
    def _compute_circuit_breaker_loss(self, batch, epoch, num_epochs):
        """
        Compute loss for circuit breaker training with memory optimization.
        """
        # Calculate scheduled coefficient based on training progress
        progress = epoch / num_epochs
        scheduled_coeff = progress
        retain_coeff, circuit_breaker_coeff = 1.0 * scheduled_coeff, 1.0 * (1 - scheduled_coeff)
        
        # Extract inputs from batch
        retain_input_ids = batch["input_ids"]
        retain_attention_mask = batch["attention_mask"]
        circuit_breaker_input_ids = batch["input_ids_circuit_breaker"]
        circuit_breaker_attention_mask = batch["attention_mask_circuit_breaker"]
        val_input_ids = batch["input_ids_val"]
        val_attention_mask = batch["attention_mask_val"]
        
        # Prepare inputs
        retain_inputs = dict(input_ids=retain_input_ids, attention_mask=retain_attention_mask, output_hidden_states=True)
        cb_inputs = dict(input_ids=circuit_breaker_input_ids, attention_mask=circuit_breaker_attention_mask, output_hidden_states=True)
        
        # Run through original model for reference hidden states
        with torch.no_grad():
            self.original_model.eval()
            
            # Retain control
            orig_retain_hidden = None
            layers_retain_attention_mask = None
            if retain_coeff > 0:
                orig_retain_outputs = self.original_model(**retain_inputs)["hidden_states"]
                orig_retain_hidden = [orig_retain_outputs[l].detach() for l in self.target_layers]
                layers_retain_attention_mask = retain_attention_mask.unsqueeze(-1)
            
            # Circuit Breaker control
            circuit_breaker_hidden = None
            if circuit_breaker_coeff > 0:
                circuit_breaker_outputs = self.original_model(**cb_inputs)["hidden_states"]
                circuit_breaker_hidden = [circuit_breaker_outputs[l].detach() for l in self.target_layers]
        
        # Run through LoRA model
        self.lora_model.train()
        
        # Retain loss
        retain_loss = 0
        if retain_coeff > 0:
            lora_retain_outputs = self.lora_model(**retain_inputs)["hidden_states"]
            lora_retain_hidden = [lora_retain_outputs[l] for l in self.target_layers]
            
            # Process each layer separately
            layer_losses = []
            for i, layer_idx in enumerate(self.target_layers):
                masked_orig = orig_retain_hidden[i] * layers_retain_attention_mask
                masked_lora = lora_retain_hidden[i] * layers_retain_attention_mask
                layer_loss = torch.norm(masked_lora - masked_orig, dim=-1, p=2).nanmean()
                layer_losses.append(layer_loss)
            
            retain_loss = torch.stack(layer_losses).mean()
        
        # Circuit Breaker loss
        circuit_breaker_loss = 0
        if circuit_breaker_coeff > 0:
            lora_circuit_breaker_outputs = ["hidden_states"]
            layers_cb_attention_mask = circuit_breaker_attention_mask.unsqueeze(-1)
            
            # Process each layer separately to save memory
            layer_losses = []
            for i, layer_idx in enumerate(self.target_layers):
                lora_cb_hidden = lora_circuit_breaker_outputs[layer_idx]
                orig_cb_hidden = circuit_breaker_hidden[i]
                
                # Normalize for cosine similarity - one layer at a time
                norm_lora = lora_cb_hidden / (torch.norm(lora_cb_hidden, dim=-1, keepdim=True) + 1e-8)
                norm_orig = orig_cb_hidden / (torch.norm(orig_cb_hidden, dim=-1, keepdim=True) + 1e-8)
                cos_sim = (norm_lora * norm_orig).sum(dim=-1, keepdim=True) * layers_cb_attention_mask
        
                # For circuit breaking, we want to MINIMIZE similarity, so we use (1 - cos_sim)
                # This gives a value between 0 and 2, where 0 is perfect similarity and 2 is perfect anti-correlation
                dissimilarity = 1 - cos_sim
                
                # Take mean over non-masked tokens
                layer_loss = dissimilarity.sum() / (layers_cb_attention_mask.sum() + 1e-8)
                layer_losses.append(layer_loss)

            circuit_breaker_loss = torch.stack(layer_losses).mean()
        
        # Total loss
        loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss
        
        # Print debug info periodically
        if hasattr(self, 'global_step') and self.global_step % 10 == 0:
            print(f"\nProgress: {progress:.4f}", "="*50)
            print(f"retain_coeff: {retain_coeff:.4f} || circuit_breaker_coeff: {circuit_breaker_coeff:.4f}")
            print(f"retain_loss: {retain_loss:.4f} \ncircuit_breaker_loss: {circuit_breaker_loss:.4f}")
            print("="*50)
        
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        self.global_step += 1
        
        return loss
    def _evaluate_generation(self, prompts):
        """Generate examples during training to check progress"""
        self.lora_model.eval()
        
        for prompt in prompts:
            encoded_inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            
            with torch.no_grad():
                outputs = self.lora_model.generate(
                    **encoded_inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7
                ).detach().cpu()
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Prompt: {prompt}")
                print(f"Generation: {generated_text[len(prompt):]}")
                print("-" * 50)
    
    def train(self, in_domain: Dataset, out_of_domain: Dataset, batch_size=10):
        """
        Train the circuit breaker using in-domain and out-of-domain datasets.
        
        Args:
            in_domain: Dataset containing in-domain examples
            out_of_domain: Dataset containing out-of-domain examples
            batch_size: Batch size for training
        """
        # Initialize global step counter
        self.global_step = 0
        
        # Train circuit breakers
        self.train_circuit_breakers(in_domain.data, out_of_domain.data, batch_size=batch_size)
    
    def generate(self, prompts: Union[str, List[str]], trip_threshold=0.8):
        """
        Generate responses using the circuit breaker model.
        
        Args:
            prompts: Input prompts to generate responses for
        
        Returns:
            Model responses
        """
        # Load the LoRA model if not already loaded
        if self.lora_model is None:
            lora_path = os.path.join(self.save_folder_path, "circuit_breaker_lora")
            if os.path.exists(lora_path):
                from peft import PeftModel
                self.lora_model = PeftModel.from_pretrained(self.original_model, lora_path)
            else:
                raise ValueError("No trained circuit breaker model found. Please train the model first.")
        
        # Handle single prompt case
        is_single_prompt = isinstance(prompts, str)
        if is_single_prompt:
            prompts = [prompts]
        
        all_responses = []
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        outputs = self.lora_model(
                    **inputs
                )
        
        original_type = type(outputs.logits)
        device = outputs.logits.device
        logits_array = np.array(outputs.logits.detach().cpu())

        # Process each prompt individually
        for i, prompt in enumerate(prompts):
            # Tokenize single input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # First, get the base model hidden states
            with torch.no_grad():
                self.original_model.eval()
                base_outputs = self.original_model(**inputs, output_hidden_states=True)
                base_hidden_states = [base_outputs.hidden_states[l] for l in self.target_layers]
            
            # Then, get the LoRA model hidden states
            with torch.no_grad():
                self.lora_model.eval()
                lora_outputs = self.lora_model(**inputs, output_hidden_states=True)
                lora_hidden_states = [lora_outputs.hidden_states[l] for l in self.target_layers]
            
            # Calculate similarity between base and LoRA hidden states to detect circuit breaking
            trip_scores = []
            for i in range(len(self.target_layers)):
                # Normalize
                norm_base = base_hidden_states[i] / (torch.norm(base_hidden_states[i], dim=-1, keepdim=True) + 1e-8)
                norm_lora = lora_hidden_states[i] / (torch.norm(lora_hidden_states[i], dim=-1, keepdim=True) + 1e-8)
                
                # Calculate cosine similarity
                similarity = (norm_base * norm_lora).sum(dim=-1).mean().item()
                trip_scores.append(similarity)
            
            # Average the scores across target layers
            avg_trip_score = sum(trip_scores) / len(trip_scores)
            is_tripped = avg_trip_score < trip_threshold
            
            # Generate text based on whether circuit breaker tripped
            if  is_tripped:
                # When circuit breaker trips, return "E"
                logits_array[i] = torch.Tensor([-1])
        
        outputs.logits = original_type(torch.Tensor(logits_array).to(device))

                
        return outputs
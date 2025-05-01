from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.nn as nn

from abc import abstractmethod


class LLMController:
    def get_max_memory_mapping(self):
        """
        Creates a memory mapping dictionary for available GPUs to balance model loading.
        
        Returns:
            Dict[str, str]: Dictionary mapping device IDs to maximum memory allocations.
        """
        import torch
        
        # Get number of GPUs
        n_gpus = torch.cuda.device_count()
        
        # Create memory map
        max_memory = {}
        
        # Assign memory for each GPU
        for i in range(n_gpus):
            # Get free memory (in bytes) and convert to GB with some buffer
            free_in_gb = torch.cuda.get_device_properties(i).total_memory * 0.85 / (1024**3)
            max_memory[f"gpu:{i}"] = f"{int(free_in_gb)}GiB"
        
        # Also specify CPU memory if needed
        max_memory["cpu"] = "24GiB"  # Adjust based on your system
        
        return max_memory

    def __init__(self, model_name, use_ddp):
        self.is_ddp = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1 and use_ddp
        
        if self.is_ddp:
            local_rank = torch.distributed.get_rank()
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="balanced", # Change from "auto" to "balanced"
        )
        
        # Wrap with DDP if in distributed mode
        if self.is_ddp:
            torch.distributed.barrier()
            try:
                self.model = DDP(self.model, device_ids=[self.device.index])
            except Exception as e:
                print(f"Error initializing DDP on rank {torch.distributed.get_rank()}: {e}")
                raise
        
        # Set model to evaluation mode
        self.model.eval()


    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def get_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) or isinstance(self.model, DDP) else self.model
    
    def generate(self, prompt, max_length=100):
        # Process input with optimized inference
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Use generate method of optimized model
        with torch.no_grad(), torch.amp.autocast('cuda'):  # Use mixed precision
            device = self.get_model().device
            outputs = self.get_model().generate(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                max_new_tokens=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode and return generated text
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        torch.cuda.empty_cache()

        return generated_text
    
    @abstractmethod
    def load(self, pathname):
        raise NotImplementedError("load method not implemented")

    @abstractmethod
    def save(self, pathname):
        raise NotImplementedError("save method not implemented")

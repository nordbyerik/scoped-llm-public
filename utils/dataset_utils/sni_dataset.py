import numpy as np
from datasets import load_dataset, IterableDataset
import random
import itertools 

class SNIDataset:

    prompt_template_with_input = "Instruction: {instruction}\nInput: {input}\nOutput:"
    prompt_template_no_input = "Instruction: {instruction}\nOutput:"

    def __init__(self, sample_size=1000, split='test', tasks=None, in_domain=True):

        self.sample_size = sample_size
        self.split = split
        self.tasks = tasks 
        self.in_domain = in_domain

        # Load and format the data upon initialization
        self.data, self.raw_data = self.get_data(sample_size, split, tasks, in_domain)

    def __len__(self):
        """Returns the number of examples in the final sampled dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Retrieves the formatted prompt example from the final sample."""
        return self.data[index]
    
    def get_raw_item(self, index):
        """Retrieves the raw, unformatted example dict from the final sample."""
        return self.raw_data[index]

    @classmethod
    def format_ni_example(cls, example):
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        if input_text and input_text.strip():
            return cls.prompt_template_with_input.format(
                instruction=instruction.strip(),
                input=input_text.strip()
            )
        else:
            return cls.prompt_template_no_input.format(
                instruction=instruction.strip()
            )

    @classmethod
    def get_data(cls, sample_size=1000, split='test', tasks=['task139_winogrande_classification'], in_domain=True):

        dataset = load_dataset('natural_instructions', split=split, streaming=True)

        reference_tasks_set = set(map(str, tasks))
        
        if in_domain:
            filter_func = lambda example: example.get('task_name') in reference_tasks_set
        else:
            filter_func = lambda example: example.get('task_name') is not None and \
                                          example.get('task_name') not in reference_tasks_set
                                          
        filtered_stream = dataset.filter(filter_func)

        # --- Collect All Matching Items from the Filtered Stream ---
        
        all_matching_items = []
        all_matching_items = list(filtered_stream) 

        actual_sample_size = min(sample_size, len(all_matching_items))
        
        sampled_items = random.sample(all_matching_items, actual_sample_size)
        
        raw_data = []
        formatted_data = []
 
        for example in sampled_items:
            instruction = example.get('task_definition', '')
            input_text = example.get('inputs', '')        
            output_text = example.get('targets', [''])[0] if example.get('targets') else '' 
            task_name = example.get('task_name', 'unknown')
            example_id = example.get('id', 'unknown')

            raw_example_data = {
                'id': example_id, 'instruction': instruction, 'input': input_text,
                'output': output_text, 'task_name': task_name
            }
            raw_data.append(raw_example_data)

            prompt_example_data = {'instruction': instruction, 'input': input_text}
            formatted_example = cls.format_ni_example(prompt_example_data)
            formatted_data.append(formatted_example)
            
        return formatted_data, raw_data

# --- Example Usage ---
if __name__ == "__main__":
    
    # Define the reference task list
    reference_tasks = [
        'task139_winogrande_classification', 
        'task114_bard_analogies',
        'task001_quoref_question_generation' 
    ] 
    
    # --- IN-DOMAIN Example ---
    desired_id_sample_count = 5 
    try:
        ni_id_dataset = SNIDataset(
            sample_size=desired_id_sample_count, 
            split='test', 
            tasks=reference_tasks,
            in_domain=True # Explicitly True (or omit, as it's default)
        )

        # Display first example if exists
        if len(ni_id_dataset) > 0:
            print("\n--- First In-Domain Sampled Example ---")
            print("Formatted Prompt:")
            print(ni_id_dataset[0]) 
            raw_item = ni_id_dataset.get_raw_item(0)
            print("\nRaw Data:")
            print(f"  Task Name: {raw_item['task_name']}")
            print(f"  Expected Output: {raw_item['output']}") 
            print("-" * 20)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    # --- OUT-OF-DOMAIN Example ---
    print("\n" + "="*30 + " OUT-OF-DOMAIN EXAMPLE " + "="*30)
    desired_ood_sample_count = 5
    try:
        # NOTE: This might take a long time and use a lot of memory!
        ni_ood_dataset = SNIDataset(
            sample_size=desired_ood_sample_count, 
            split='test', 
            tasks=reference_tasks, # Define what tasks are considered "in-domain"
            in_domain=False # Sample from tasks NOT in the list above
        )

        print(f"\nSuccessfully created OUT-OF-DOMAIN dataset with {len(ni_ood_dataset)} examples.")
         # Display first example if exists
        if len(ni_ood_dataset) > 0:
            print("\n--- First Out-of-Domain Sampled Example ---")
            print("Formatted Prompt:")
            print(ni_ood_dataset[0]) 
            raw_item = ni_ood_dataset.get_raw_item(0)
            print("\nRaw Data:")
            print(f"  Task Name: {raw_item['task_name']}")
            print(f"  Expected Output: {raw_item['output']}") 
            print("-" * 20)

    except Exception as e:
        print(f"\nAn unexpected error occurred during OOD processing: {e}")
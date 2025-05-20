import os
import numpy as np
import pickle
# datasets.load_dataset will be imported within a method where it's used,
# or can be at the top of the module file.
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Tuple, Union # Union for type hint

class MMLUDataset(Dataset):
    mmlu_subjects = {
            'stem': [
                'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
                'college_computer_science', 'college_mathematics', 'college_physics',
                'computer_security', 'conceptual_physics', 'electrical_engineering',
                'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
                'high_school_computer_science', 'high_school_mathematics', 'high_school_physics',
                'high_school_statistics', 'machine_learning', 'medical_genetics', 'virology', 'college_medicine',
                'anatomy', 'professional_medicine', 'clinical_knowledge'
            ],
            'non_stem': [
                'business_ethics', 'econometrics', 'global_facts', 'high_school_european_history',
                'high_school_geography', 'high_school_government_and_politics',
                'high_school_macroeconomics', 'high_school_microeconomics',
                'high_school_psychology', 'high_school_us_history', 'high_school_world_history',
                'human_aging', 'human_sexuality', 'jurisprudence',
                'logical_fallacies', 'management', 'marketing',
                'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
                'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
                'professional_psychology', 'public_relations',
                'security_studies', 'sociology', 'us_foreign_policy',
                'world_religions'
            ]
    }

    def __init__(self, sample_size: int = 1000, split: str = 'validation',
                 domains: Union[str, List[str]] = 'stem', in_domain: bool = True, test_percentage: float = 0.2,
                 cache_base_dir: str = 'caches/mmlu_dataset'): # Added cache_base_dir
        self.cache_base_dir = cache_base_dir
        self.domains = self._get_domains(domains, in_domain)

        if not self.domains:
            self.data = []
            self.answers = []
        else:
            self.data, self.answers = self._load_data(sample_size, split)

        self.in_domain_flags = [int(in_domain)] * len(self.data)

        if len(self.data) > 1 :
            self.train_indices, self.test_indices = self._train_test_split(test_percentage)
        else: # Handle datasets with 0 or 1 item
            self.train_indices = list(range(len(self.data)))
            self.test_indices = []


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: Union[int, list, np.ndarray]):
        if isinstance(idx, (list, np.ndarray)):
            return self._create_subset(list(idx) if isinstance(idx, np.ndarray) else idx)
        return self._get_single_item(idx)

    def _get_domains(self, domains_param: Union[str, List[str]], in_domain: bool) -> List[str]:
        if isinstance(domains_param, str) and domains_param in ['stem', 'non_stem']:
            target_category = domains_param if in_domain else ('non_stem' if domains_param == 'stem' else 'stem')
            return self.mmlu_subjects[target_category]
        
        current_selection: List[str] = []
        if isinstance(domains_param, str): # A single subject string
            current_selection = [domains_param]
        elif isinstance(domains_param, list):
            current_selection = domains_param
        else:
            # This case should ideally be caught by type hinting or earlier validation if possible
            raise ValueError(f"Invalid 'domains' argument type: {type(domains_param)}. Must be str or List[str].")

        if not in_domain:
            all_subjects_flat = [subject for category_subjects in self.mmlu_subjects.values() for subject in category_subjects]
            return [subject for subject in all_subjects_flat if subject not in current_selection]
        return current_selection


    def _load_data(self, sample_size: int, split: str) -> Tuple[List[str], List[int]]:
        all_raw_formatted_examples = [] # Stores dicts from _format_example

        if not self.domains:
            return [], []

        for subject in self.domains:
            # _load_subject_data returns a list of dicts (output of _format_example)
            subject_formatted_examples = self._load_subject_data(subject, sample_size, split)
            all_raw_formatted_examples.extend(subject_formatted_examples)

        if not all_raw_formatted_examples:
            return [], []

        if len(all_raw_formatted_examples) > sample_size:
            np.random.shuffle(all_raw_formatted_examples) # Shuffle the list of dicts
            all_raw_formatted_examples = all_raw_formatted_examples[:sample_size]
        
        # _format_data now takes these dicts and returns (formatted_prompts, answer_indices)
        formatted_prompts, answer_indices = self._format_data(all_raw_formatted_examples)
        return formatted_prompts, answer_indices

    def _load_subject_data(self, subject: str, total_sample_size: int, split: str) -> List[Dict]:
        data_path = os.path.join(self.cache_base_dir, f"mmlu_dataset_{subject}_{split}_{total_sample_size}.pkl")
        os.makedirs(self.cache_base_dir, exist_ok=True)

        num_domains = len(self.domains)
        per_subject_target = total_sample_size // num_domains if num_domains > 0 else total_sample_size
        per_subject_target = max(1, per_subject_target)

        raw_data_for_subject: List[Dict] = [] # Stores raw items from Hugging Face dataset
        loaded_from_cache = False
        if os.path.exists(data_path):
            try:
                with open(data_path, 'rb') as f:
                    raw_data_for_subject = pickle.load(f)
                if not isinstance(raw_data_for_subject, list): # Basic check for cache integrity
                    raw_data_for_subject = []
                    raise pickle.UnpicklingError("Cache content is not a list.")
                loaded_from_cache = True
            except (pickle.UnpicklingError, EOFError, FileNotFoundError, AttributeError, TypeError) as e:
                # print(f"Cache error for {data_path}: {e}. Reloading.") # Optional: log this
                raw_data_for_subject = []
                loaded_from_cache = False

        if not loaded_from_cache:
            from datasets import load_dataset # Import here
            try:
                dataset = load_dataset("cais/mmlu", subject, split=split)
                raw_data_for_subject = list(dataset) # List of dicts from Hugging Face
                with open(data_path, 'wb') as f:
                    pickle.dump(raw_data_for_subject, f)
            except Exception as e: # Catch issues like network errors, dataset not found, etc.
                # print(f"Failed to load or cache '{subject}' from Hugging Face: {e}") # Optional: log this
                raw_data_for_subject = [] # Ensure it's an empty list on failure

        if not raw_data_for_subject:
            return []

        num_available_for_subject = len(raw_data_for_subject)
        num_to_sample = min(num_available_for_subject, per_subject_target)

        if num_to_sample == 0 :
             return []
        
        # Ensure num_to_sample is not greater for np.random.choice replace=False
        # This should be guaranteed by min() above, but defensive check:
        num_to_sample = min(num_to_sample, num_available_for_subject)


        indices = np.random.choice(num_available_for_subject, size=num_to_sample, replace=False)
        
        # Apply _format_example to the sampled raw items
        formatted_examples_for_subject = [self._format_example(raw_data_for_subject[int(i)]) for i in indices]
        return formatted_examples_for_subject


    @staticmethod
    def _format_example(example: Dict) -> Dict:
        choices = example.get('choices', [])
        # Ensure choices is a list and pad/truncate to 4
        if not isinstance(choices, list): choices = [] # Default to empty list if not a list
        
        if len(choices) < 4:
            choices.extend([""] * (4 - len(choices)))
        elif len(choices) > 4:
            choices = choices[:4]

        return {
            'question': example.get('question', "N/A"), # Default for missing question
            'choices': choices, # Now always a list of 4 strings
            'answer': example.get('answer', -1) # Default for missing answer (e.g., an out-of-bounds index)
        }

    @staticmethod
    def _format_data(data: List[Dict]) -> Tuple[List[str], List[int]]:
        prompt_template = "Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"
        formatted_data_strings = []
        answers_indices = []

        for example_dict in data: # example_dict is the output of _format_example
            # Ensure choices exist and are sufficient, though _format_example should handle this
            ch = example_dict.get('choices', ["", "", "", ""])
            if len(ch) < 4 : ch.extend([""] * (4-len(ch)))


            formatted_q_str = prompt_template.format(
                question=example_dict.get('question', "N/A"),
                A=ch[0],
                B=ch[1],
                C=ch[2],
                D=ch[3]
            )
            formatted_data_strings.append(formatted_q_str)
            answers_indices.append(example_dict.get('answer', -1))

        return formatted_data_strings, answers_indices

    def _train_test_split(self, test_size: float) -> Tuple[List[int], List[int]]:
        num_items = len(self.data)
        # This method is now only called if num_items > 1 (controlled by __init__)
        assert num_items > 1, "Dataset must contain at least two examples for splitting."

        indices = list(range(num_items))
        np.random.shuffle(indices)

        split_point = int(num_items * (1 - test_size))
        split_point = max(1, split_point)
        split_point = min(num_items - 1, split_point)

        return indices[:split_point], indices[split_point:]

    def _create_subset(self, indices: List[int]) -> 'MMLUDataset':
        subset = MMLUDataset.__new__(MMLUDataset)
        subset.cache_base_dir = self.cache_base_dir # Carry over setting

        subset.data = [self.data[i] for i in indices]
        subset.answers = [self.answers[i] for i in indices]
        subset.in_domain_flags = [self.in_domain_flags[i] for i in indices]
        
        subset.domains = self.domains # Carry over original domains for context
        # Subsets created this way don't have their own train/test split derived from their new size by default
        subset.train_indices = list(range(len(subset.data))) # Or consider it as all "train" data for this view
        subset.test_indices = []
        return subset

    def _get_single_item(self, idx: int) -> Dict:
        return {
            "question": self.data[idx],
            "answer": self.answers[idx],
            "in_domain": self.in_domain_flags[idx]
        }

    def get_train_dataset(self) -> 'MMLUDataset':
        if not self.train_indices:
            # Create an empty MMLUDataset instance by providing minimal valid args to __init__
            # or by creating via _create_subset([])
             return self._create_subset([])
        return self[self.train_indices]

    def get_test_dataset(self) -> 'MMLUDataset':
        if not self.test_indices:
            return self._create_subset([])
        return self[self.test_indices]
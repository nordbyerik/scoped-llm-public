import os
import numpy as np
import pickle
from datasets import load_dataset
from torch.utils.data import Dataset, Subset
from typing import List, Dict, Tuple



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
                 domains: str = 'stem', in_domain: bool = True, test_percentage: float = 0.2):
        self.domains = self._get_domains(domains, in_domain)
        self.data, self.answers = self._load_data(sample_size, split)
        self.in_domain = [int(in_domain)] * len(self.data)
        self.train_indices, self.test_indices = self._train_test_split(test_percentage)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray)):
            return self._create_subset(idx)
        return self._get_single_item(idx)

    def _get_domains(self, domains: str, in_domain: bool) -> List[str]:
        if domains in ['stem', 'non_stem']:
            return self.mmlu_subjects['stem' if (domains == 'stem') == in_domain else 'non_stem']
        if not in_domain:
            return [subject for category in self.mmlu_subjects.values() for subject in category if subject not in domains]
        return domains

    def _load_data(self, sample_size: int, split: str) -> Tuple[List[str], List[str]]:
        all_data = []
        for subject in self.domains:
            subject_data = self._load_subject_data(subject, sample_size, split)
            all_data.extend(subject_data)

        if len(all_data) > sample_size:
            np.random.shuffle(all_data)
            all_data = all_data[:sample_size]

        formatted_data, answers = self._format_data(all_data)
        return formatted_data, answers

    def _load_subject_data(self, subject: str, sample_size: int, split: str) -> List[Dict]:
        data_path = os.path.join('mmlu_dataset', f"mmlu_dataset_{subject}_{split}_{sample_size}.pkl")
        os.makedirs('mmlu_dataset', exist_ok=True)

        if os.path.exists(data_path):
            return pickle.load(open(data_path, 'rb'))

        dataset = load_dataset("cais/mmlu", subject, split=split)
        raw_data = list(dataset)
        pickle.dump(raw_data, open(data_path, 'wb'))

        max_examples = min(len(dataset), sample_size // len(self.domains))
        max_examples = max(1, max_examples)
        indices = np.random.choice(len(dataset), size=max_examples, replace=False)

        return [self._format_example(dataset[int(i)]) for i in indices]

    @staticmethod
    def _format_example(example: Dict) -> Dict:
        return {
            'question': example['question'],
            'choices': [example['choices'][i] for i in range(4)],
            'answer': example['answer']
        }

    @staticmethod
    def _format_data(data: List[Dict]) -> Tuple[List[str], List[str]]:
        prompt_template = "Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"
        formatted_data = []
        answers = []

        for example in data:
            formatted = prompt_template.format(
                question=example['question'],
                A=example['choices'][0],
                B=example['choices'][1],
                C=example['choices'][2],
                D=example['choices'][3]
            )
            formatted_data.append(formatted)
            answers.append(example['answer'])

        return formatted_data, answers

    def _train_test_split(self, test_size: float) -> Tuple[List[int], List[int]]:
        indices = list(range(len(self.data)))
        np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - test_size))

        assert len(indices) > 1, "Dataset must contain at least two examples for splitting."  # Ensure there are enough examples to split
        split_point = max(1, split_point)  # Ensure at least one example in training set
        split_point = min(len(indices) - 1, split_point)  # Ensure at least one example in test set
        
        return indices[:split_point], indices[split_point:]

    def _create_subset(self, indices: List[int]) -> 'MMLUDataset':
        subset = MMLUDataset.__new__(MMLUDataset)
        subset.data = [self.data[i] for i in indices]
        subset.answers = [self.answers[i] for i in indices]
        subset.in_domain = [self.in_domain[i] for i in indices]
        return subset

    def _get_single_item(self, idx: int) -> Dict:
        return {
            "question": self.data[idx],
            "answer": self.answers[idx],
            "in_domain": self.in_domain[idx]
        }

    def get_train_dataset(self) -> Subset:
        subset = Subset(self, self.train_indices)
        return subset.dataset[subset.indices]

    def get_test_dataset(self) -> Subset:
        subset = Subset(self, self.test_indices)
        return subset.dataset[subset.indices]

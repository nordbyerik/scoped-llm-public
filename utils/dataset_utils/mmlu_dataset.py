import os
import numpy as np
import pickle
from datasets import load_dataset
from torch.utils.data import Dataset, Subset
from typing import List

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

class MMLUDataset(Dataset):
    def __init__(self, sample_size=1000, split='validation', domains='stem', in_domain=True):
        # Grab stem

        if (domains == 'stem' and in_domain) or (domains == 'non_stem' and not in_domain):
            domains = mmlu_subjects['stem']
        # Grab non-stem
        elif (domains == 'stem' and not in_domain) or (domains == 'non_stem' and in_domain):
            domains = mmlu_subjects['non_stem']
        # Grab all
        elif (not in_domain):
            out_domains = []
            for category, subjects in mmlu_subjects.items():
                for subject in subjects:
                    if subject not in domains:
                        out_domains.append(subject)
            domains = out_domains
        self.domains = domains
        
        self.data, self.answers = self.get_data(sample_size, split)
        self.in_domain = [int(in_domain) for _ in range(len(self.data))]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        if isinstance(idx, (list, np.ndarray)):
            # Return a new dataset with the specified indices
            subset_data = [self.data[i] for i in idx]
            subset_answers = [self.answers[i] for i in idx]
            subset_in_domain = [self.in_domain[i] for i in idx]
            
            new_dataset = MMLUDataset.__new__(MMLUDataset)  # Create new instance without calling __init__
            new_dataset.data = subset_data
            new_dataset.answers = subset_answers
            new_dataset.in_domain = subset_in_domain
            return new_dataset
        else:
            return {
                "question": self.data[idx],
                "answer": self.answers[idx],
                "in_domain": self.in_domain[idx]
            }

    def get_data(self, sample_size=1000, split='validation'):

        
        data = []
        for category, subjects in mmlu_subjects.items():
            for subject in subjects:
                if subject not in self.domains:
                    continue

                data_path = os.path.join('mmlu_dataset', f"mmlu_dataset_{subject}_{split}_{sample_size}.pkl")
                if not os.path.exists('mmlu_dataset'):
                    os.makedirs('mmlu_dataset')
                if os.path.exists(data_path):
                    dataset = pickle.load(open(data_path, 'rb'))
                else:
                    dataset = load_dataset("cais/mmlu", subject, split=split)
                    raw_data = [example for example in dataset]
                    pickle.dump(raw_data, open(data_path, 'wb'))


                max_examples = min(len(dataset), sample_size // len(self.domains))
                indices = np.random.choice(len(dataset), size=max_examples, replace=False)

                for i in indices:
                    example = dataset[int(i)]
                    data.append({
                        'question': example['question'],
                        'choices': [example['choices'][i] for i in range(4)],
                        'answer': example['answer']
                    })
                del dataset

        # Format MMLU examples for evaluation
        prompt_template = "Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"
        def format_mmlu_example(example, prompt_template):
            """Format an MMLU example using the prompt template."""
            formatted = prompt_template.format(
                question=example['question'],
                A=example['choices'][0],
                B=example['choices'][1],
                C=example['choices'][2],
                D=example['choices'][3]
            )
            return formatted

        formatted_data = []
        answers = [example['answer'] for example in data]
        for example in data:
            formatted_example = format_mmlu_example(example, prompt_template)
            formatted_data.append(formatted_example)
        

        return formatted_data, answers

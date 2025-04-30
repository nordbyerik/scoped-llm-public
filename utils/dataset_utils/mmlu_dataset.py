import os
import numpy as np
import pickle
from datasets import load_dataset


mmlu_subjects = {
            'stem': [
                'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
                'college_computer_science', 'college_mathematics', 'college_physics',
                'computer_security', 'conceptual_physics', 'electrical_engineering',
                'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
                'high_school_computer_science', 'high_school_mathematics', 'high_school_physics',
                'high_school_statistics', 'machine_learning'
            ],
            'non_stem': [
                'anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine',
                'econometrics', 'global_facts', 'high_school_european_history',
                'high_school_geography', 'high_school_government_and_politics',
                'high_school_macroeconomics', 'high_school_microeconomics',
                'high_school_psychology', 'high_school_us_history', 'high_school_world_history',
                'human_aging', 'human_sexuality', 'international_law', 'jurisprudence',
                'logical_fallacies', 'management', 'marketing', 'medical_genetics',
                'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition',
                'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
                'professional_medicine', 'professional_psychology', 'public_relations',
                'security_studies', 'sociology', 'us_foreign_policy', 'virology',
                'world_religions'
            ]
    }

class MMLUDataset:
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
        
        domains = [domains[0]] # TODO: Remove this - just for testing
        self.data, self.answers = self.get_data(sample_size, split, domains)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.answers[index]

    @classmethod
    def get_data(cls, sample_size=1000, split='validation', domains=['astronomy']):

        
        data = []
        for category, subjects in mmlu_subjects.items():
            for subject in subjects:
                if subject not in domains:
                    continue

                path = os.path.join('mmlu_dataset', f"mmlu_dataset_{subject}_{split}_{sample_size}.pkl")
                if os.path.exists(path):
                    dataset = pickle.load(open(path, 'rb'))
                else:
                    dataset = load_dataset("cais/mmlu", subject, split=split)
                    pickle.dump(dataset, open(path, 'wb'))


                max_examples = min(len(dataset), sample_size // len(domains))
                indices = np.random.choice(len(dataset), size=max_examples, replace=False)

                for i in indices:
                    example = dataset[int(i)]
                    data.append({
                        'question': example['question'],
                        'choices': [example['choices'][i] for i in range(4)],
                        'answer': example['answer']
                    })

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

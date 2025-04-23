from datasets import load_dataset

class MMLUDataset:
    mmlu_subjects = {
            'stem': [
                'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
                'college_computer_science', 'college_mathematics', 'college_physics',
                'computer_security', 'conceptual_physics', 'electrical_engineering',
                'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
                'high_school_computer_science', 'high_school_mathematics', 'high_school_physics',
                'high_school_statistics', 'machine_learning', 'physics'
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

    @classmethod
    def get_data(cls, sample_size=1000, split='validation', in_domain='stem'):
        data = {'stem': [], 'non_stem': []}

        for category, subjects in cls.mmlu_subjects.items():
            for subject in subjects:
                
                dataset = load_dataset("cais/mmlu", subject, split=split)

                # Limit examples per subject
                max_examples = min(len(dataset), sample_size // len(subjects))

                for i in range(max_examples):
                    example = dataset[i]
                    data[category].append({
                        'subject': subject,
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

        positive_texts = []
        negative_texts = []

        for category, examples in data.items():
            for example in examples:
                formatted_example = format_mmlu_example(example, prompt_template)
                if category == 'stem':
                    positive_texts.append(formatted_example)
                else:
                    negative_texts.append(formatted_example)


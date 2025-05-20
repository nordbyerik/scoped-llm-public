import unittest
from unittest.mock import patch, MagicMock, mock_open, call, ANY
import os
import numpy as np
import pickle
import shutil

from llm_scoper.utils.dataset_utils.mmlu_dataset import MMLUDataset

# from datasets import load_dataset # This will be mocked by @patch('datasets.load_dataset')
# from torch.utils.data import Dataset, Subset # Already imported in MMLUDataset class file

# Assuming MMLUDataset class is defined above or in an importable module
# For this example, the class is defined in the same scope or pasted above.

TEST_CACHE_DIR = 'test_mmlu_cache_dir_unique' # Unique name for test cache

# Helper to create mock Hugging Face dataset items
def create_mock_hf_item(subject: str, q_idx: int):
    return {
        'question': f'{subject} Question {q_idx + 1}', # Qs are 1-indexed for readability
        'choices': [f'Choice{i}' for i in range(4)],
        'answer': q_idx % 4 # Answer is 0, 1, 2, or 3
    }

# Helper to create a mock Hugging Face dataset object
def mock_hf_dataset_obj(subject: str, num_items: int):
    items = [create_mock_hf_item(subject, i) for i in range(num_items)]
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(items)
    mock_ds.__iter__.return_value = iter(items) # For `list(dataset)`
    def getitem_side_effect(idx):
        if isinstance(idx, int):
            return items[idx]
        raise TypeError(f"Mock HF dataset __getitem__ called with type {type(idx)}")
    mock_ds.__getitem__.side_effect = getitem_side_effect # For `dataset[int(i)]` if used on original obj
    return mock_ds


class TestMMLUDataset(unittest.TestCase):

    def setUp(self):
        os.makedirs(TEST_CACHE_DIR, exist_ok=True)
        # Mock np.random.shuffle to make split order deterministic (no-op shuffle)
        self.patch_np_shuffle = patch('numpy.random.shuffle', side_effect=lambda x: None) # Modifies in-place
        self.mock_np_shuffle = self.patch_np_shuffle.start()

    def tearDown(self):
        if os.path.exists(TEST_CACHE_DIR):
            shutil.rmtree(TEST_CACHE_DIR)
        self.patch_np_shuffle.stop()
        patch.stopall() # Stops all patches started with patch.start() by a test method

    def test_format_example(self):
        raw_example = {'question': 'Q1', 'choices': ['A', 'B', 'C', 'D'], 'answer': 0}
        formatted = MMLUDataset._format_example(raw_example)
        self.assertEqual(formatted['question'], 'Q1')
        self.assertEqual(formatted['choices'], ['A', 'B', 'C', 'D'])
        self.assertEqual(formatted['answer'], 0)

        raw_example_missing_choices = {'question': 'Q2', 'answer': 1}
        formatted_missing = MMLUDataset._format_example(raw_example_missing_choices)
        self.assertEqual(formatted_missing['question'], 'Q2')
        self.assertEqual(formatted_missing['choices'], ["", "", "", ""]) # Padded
        self.assertEqual(formatted_missing['answer'], 1)
        
        raw_example_short_choices = {'question': 'Q3', 'choices': ['A', 'B'], 'answer': 0}
        formatted_short = MMLUDataset._format_example(raw_example_short_choices)
        self.assertEqual(formatted_short['choices'], ['A', 'B', "", ""])

        raw_example_long_choices = {'question': 'Q4', 'choices': ['A', 'B', 'C', 'D', 'E'], 'answer': 0}
        formatted_long = MMLUDataset._format_example(raw_example_long_choices)
        self.assertEqual(formatted_long['choices'], ['A', 'B', 'C', 'D']) # Truncated

        raw_example_no_answer = {'question': 'Q5', 'choices': ['A', 'B', 'C', 'D']}
        formatted_no_ans = MMLUDataset._format_example(raw_example_no_answer)
        self.assertEqual(formatted_no_ans['answer'], -1) # Default answer

    def test_format_data(self):
        examples_formatted_from_step1 = [ # These are outputs of _format_example
            {'question': 'Q1', 'choices': ['A1', 'B1', 'C1', 'D1'], 'answer': 0},
            {'question': 'Q2', 'choices': ['A2', 'B2', 'C2', 'D2'], 'answer': 1},
        ]
        formatted_prompts, answers_indices = MMLUDataset._format_data(examples_formatted_from_step1)
        self.assertEqual(len(formatted_prompts), 2)
        self.assertEqual(len(answers_indices), 2)
        self.assertIn("Question: Q1\nA. A1\nB. B1\nC. C1\nD. D1\nAnswer:", formatted_prompts[0])
        self.assertEqual(answers_indices[0], 0)
        self.assertIn("Question: Q2\nA. A2\nB. B2\nC. C2\nD. D2\nAnswer:", formatted_prompts[1])
        self.assertEqual(answers_indices[1], 1)

    def test_get_domains(self):
        # _get_domains is an instance method, needs a dummy instance.
        # We don't need to fully init, just need mmlu_subjects.
        dataset_instance = MMLUDataset.__new__(MMLUDataset)
        dataset_instance.mmlu_subjects = MMLUDataset.mmlu_subjects # Assign class attribute

        self.assertEqual(dataset_instance._get_domains('stem', True), MMLUDataset.mmlu_subjects['stem'])
        self.assertEqual(dataset_instance._get_domains('stem', False), MMLUDataset.mmlu_subjects['non_stem'])
        
        custom_list = ['abstract_algebra', 'astronomy']
        self.assertEqual(dataset_instance._get_domains(custom_list, True), custom_list)
        
        all_subjects_flat = [s for cat_s in MMLUDataset.mmlu_subjects.values() for s in cat_s]
        expected_ood_custom = [s for s in all_subjects_flat if s not in custom_list]
        self.assertCountEqual(dataset_instance._get_domains(custom_list, False), expected_ood_custom)

        single_subject = 'anatomy'
        self.assertEqual(dataset_instance._get_domains(single_subject, True), [single_subject])
        expected_ood_single = [s for s in all_subjects_flat if s != single_subject]
        self.assertCountEqual(dataset_instance._get_domains(single_subject, False), expected_ood_single)

        with self.assertRaises(ValueError):
            dataset_instance._get_domains(123, True) # Invalid domain type

    @patch('datasets.load_dataset') # Patch where it's looked up.
    @patch('pickle.dump')
    @patch('pickle.load')
    @patch('os.path.exists')
    @patch('numpy.random.choice')
    def test_load_subject_data_cache_interaction(self, mock_np_choice, mock_os_exists, mock_pickle_load, mock_pickle_dump, mock_hf_load_dataset):
        subject = 'astronomy'
        total_sample_size = 10 # For cache filename and per_subject_target calculation
        split = 'test'
        
        # This test focuses on _load_subject_data, called by MMLUDataset.__init__
        # We create a MMLUDataset instance. It will have one domain for this test.
        # per_subject_target will be total_sample_size / 1 = 10.

        # --- Cache Miss Scenario ---
        mock_os_exists.return_value = False # Cache does not exist
        hf_data_astronomy = [create_mock_hf_item(subject, i) for i in range(5)] # HF returns 5 items
        mock_hf_load_dataset.return_value = mock_hf_dataset_obj(subject, 5)
        
        # np.random.choice: from 5 available items, sample min(5, per_subject_target=10) = 5 items
        mock_np_choice.return_value = np.array([0,1,2,3,4]) # Choose all 5

        # Mock 'open' for checking pickle.dump call arguments
        # This is a bit more involved if we want to check contents written to file.
        # For now, let's ensure it's called.
        # The path for dump:
        expected_cache_path = os.path.join(TEST_CACHE_DIR, f"mmlu_dataset_{subject}_{split}_{total_sample_size}.pkl")
        
        # Initialize dataset to trigger loading
        with patch('builtins.open', mock_open()) as mocked_file_open:
            dataset_miss = MMLUDataset(sample_size=total_sample_size, domains=[subject], split=split, cache_base_dir=TEST_CACHE_DIR)
            mock_hf_load_dataset.assert_called_once_with("cais/mmlu", subject, split=split)
            # Check if pickle.dump was called correctly
            mocked_file_open.assert_called_with(expected_cache_path, 'wb')
            mock_pickle_dump.assert_called_once_with(hf_data_astronomy, mocked_file_open())
        
        self.assertEqual(len(dataset_miss.data), 5)
        self.assertIn(f"{subject} Question 1", dataset_miss.data[0])


        # --- Cache Hit Scenario ---
        mock_hf_load_dataset.reset_mock()
        mock_pickle_dump.reset_mock() # Not strictly needed as it shouldn't be called
        mock_os_exists.return_value = True # Now cache exists
        mock_pickle_load.return_value = hf_data_astronomy # pickle.load returns the 5 items
        
        # np.random.choice called again. Let's say it chooses 3 out of the 5 cached items.
        # per_subject_target is still 10. num_available from cache is 5. num_to_sample = min(5,10) = 5.
        # So, it will try to sample 5 items. Let's make it choose 3 specific ones.
        mock_np_choice.return_value = np.array([0,2,4]) # Indices from the cached data

        with patch('builtins.open', mock_open(read_data=pickle.dumps(hf_data_astronomy))) as mocked_file_open_hit:
            dataset_hit = MMLUDataset(sample_size=total_sample_size, domains=[subject], split=split, cache_base_dir=TEST_CACHE_DIR)
            mocked_file_open_hit.assert_called_with(expected_cache_path, 'rb') # Check open for read
            mock_pickle_load.assert_called_once_with(mocked_file_open_hit())
        
        mock_hf_load_dataset.assert_not_called() # Should not call HF load
        mock_pickle_dump.assert_not_called() # Should not re-dump

        self.assertEqual(len(dataset_hit.data), 3) # Only 3 items based on mock_np_choice for the hit
        self.assertIn(f"{subject} Question 1", dataset_hit.data[0]) # Corresponds to original index 0
        self.assertIn(f"{subject} Question 3", dataset_hit.data[1]) # Corresponds to original index 2
        self.assertIn(f"{subject} Question 5", dataset_hit.data[2]) # Corresponds to original index 4

    @patch('datasets.load_dataset')
    @patch('numpy.random.choice')
    def test_initialization_overall_sampling_and_splitting(self, mock_np_choice, mock_hf_load_dataset):
        # Mock os.path.exists to always indicate cache miss for cleaner setup
        with patch('os.path.exists', return_value=False), \
             patch('pickle.dump'): # Mock dump as we are not testing cache writing here explicitly

            # Setup: 2 subjects, each can provide 8 items.
            # sample_size = 10. domains = ['s1', 's2']. test_percentage = 0.2 (8 train, 2 test)
            # per_subject_target = 10 // 2 = 5.
            mock_hf_load_dataset.side_effect = [
                mock_hf_dataset_obj('s1', 8), 
                mock_hf_dataset_obj('s2', 8)
            ]
            # np.random.choice for s1 (from 8, choose 5):
            # np.random.choice for s2 (from 8, choose 5):
            mock_np_choice.side_effect = [
                np.array([0,1,2,3,4]), # s1 takes first 5
                np.array([0,1,2,3,4])  # s2 takes its first 5 (indices relative to its own data)
            ]
            # Total 5+5 = 10 items loaded into all_raw_formatted_examples.
            # This matches sample_size=10, so no further global trimming.

            dataset = MMLUDataset(sample_size=10, domains=['s1', 's2'], test_percentage=0.2, cache_base_dir=TEST_CACHE_DIR)

            self.assertEqual(len(dataset.data), 10)
            self.assertEqual(len(dataset.answers), 10)
            self.assertEqual(len(dataset.in_domain_flags), 10)
            
            # Check data from both subjects (mock_np_shuffle is no-op, so order is s1 then s2)
            s1_prompts = [d for d in dataset.data if "s1 Question" in d]
            s2_prompts = [d for d in dataset.data if "s2 Question" in d]
            self.assertEqual(len(s1_prompts), 5)
            self.assertEqual(len(s2_prompts), 5)
            self.assertIn("s1 Question 1", dataset.data[0])
            self.assertIn("s2 Question 1", dataset.data[5])


            # Train/Test split (10 items, 0.2 test -> 8 train, 2 test)
            # Since shuffle is no-op, train will be first 8, test last 2.
            self.assertEqual(len(dataset.train_indices), 8)
            self.assertEqual(len(dataset.test_indices), 2)
            # Train data: s1 Q1-5, s2 Q1-3
            # Test data: s2 Q4-5
            
            train_ds = dataset.get_train_dataset()
            test_ds = dataset.get_test_dataset()
            self.assertEqual(len(train_ds.data), 8)
            self.assertEqual(len(test_ds.data), 2)
            self.assertIn("s1 Question 1", train_ds.data[0]) # First item of train
            self.assertIn("s2 Question 3", train_ds.data[7]) # Last item of train (s1_5, s2_1, s2_2, s2_3)
            self.assertIn("s2 Question 4", test_ds.data[0])  # First item of test

            # Scenario: overall sample_size forces trimming
            # 3 subjects, each provides 3 items (total 9 if all taken by subject loaders)
            # sample_size = 5. per_subject_target = max(1, 5//3=1) = 1.
            # So each subject contributes 1 item. all_raw_formatted_examples length = 3.
            # This is less than sample_size=5. So no global trim.
            mock_hf_load_dataset.side_effect = [
                mock_hf_dataset_obj('d1', 3), mock_hf_dataset_obj('d2', 3), mock_hf_dataset_obj('d3', 3)
            ]
            mock_np_choice.side_effect = [np.array([0]), np.array([0]), np.array([0])] # Each takes 1st item
            dataset_no_trim_needed = MMLUDataset(sample_size=5, domains=['d1','d2','d3'], cache_base_dir=TEST_CACHE_DIR)
            self.assertEqual(len(dataset_no_trim_needed.data), 3)

            # Scenario: overall sample_size forces trimming (actual trim)
            # 3 subjects. sample_size = 2.
            # per_subject_target = max(1, 2//3=0) = 1. Each subject contributes 1 item.
            # all_raw_formatted_examples length = 3.
            # Then, this list of 3 items is shuffled (no-op here) and trimmed to sample_size=2.
            dataset_trim_active = MMLUDataset(sample_size=2, domains=['d1','d2','d3'], cache_base_dir=TEST_CACHE_DIR)
            self.assertEqual(len(dataset_trim_active.data), 2)
            # Due to no-op shuffle, it should contain data from d1 and d2.
            self.assertTrue(any("d1 Question" in prompt for prompt in dataset_trim_active.data))
            self.assertTrue(any("d2 Question" in prompt for prompt in dataset_trim_active.data))
            self.assertFalse(any("d3 Question" in prompt for prompt in dataset_trim_active.data))


    def test_train_test_split_method_direct(self):
        # Test _train_test_split directly (it assumes len(self.data) > 1)
        dummy_ds = MMLUDataset.__new__(MMLUDataset) # Create instance without __init__
        dummy_ds.data = list(range(10)) # 10 items, shuffle is no-op

        train_idx, test_idx = dummy_ds._train_test_split(0.2) # 8 train, 2 test
        self.assertEqual(len(train_idx), 8)
        self.assertEqual(len(test_idx), 2)
        self.assertListEqual(train_idx, list(range(8)))
        self.assertListEqual(test_idx, list(range(8,10)))

        dummy_ds.data = ["item1", "item2"] # 2 items
        train_idx, test_idx = dummy_ds._train_test_split(0.5) # 1 train, 1 test
        self.assertEqual(len(train_idx), 1)
        self.assertEqual(len(test_idx), 1)

        # Test split point clamping (max(1, ...) and min(len-1, ...))
        dummy_ds.data = list(range(10))
        # test_size = 0.01 (effectively 1 test item due to clamping)
        # split_point = int(10 * 0.99) = 9. max(1,9)=9. min(9, 9)=9. So 9 train, 1 test.
        train_idx, test_idx = dummy_ds._train_test_split(0.01)
        self.assertEqual(len(train_idx), 9)
        self.assertEqual(len(test_idx), 1)

        # test_size = 0.95 (effectively 1 train item due to clamping)
        # split_point = int(10 * 0.05) = 0. max(1,0)=1. min(1,9)=1. So 1 train, 9 test.
        train_idx, test_idx = dummy_ds._train_test_split(0.95)
        self.assertEqual(len(train_idx), 1)
        self.assertEqual(len(test_idx), 9)

    @patch('datasets.load_dataset')
    @patch('numpy.random.choice')
    def test_len_and_getitem_single(self, mock_np_choice, mock_hf_load_dataset):
        with patch('os.path.exists', return_value=False), patch('pickle.dump'):
            mock_hf_load_dataset.return_value = mock_hf_dataset_obj('subj1', 5)
            mock_np_choice.return_value = np.arange(3) # Load 3 items (0,1,2)
            dataset = MMLUDataset(sample_size=3, domains=['subj1'], cache_base_dir=TEST_CACHE_DIR, in_domain=True)
        
        self.assertEqual(len(dataset), 3)
        item0 = dataset[0]
        self.assertIsInstance(item0, dict)
        self.assertIn("subj1 Question 1", item0['question']) # Q index 0
        self.assertEqual(item0['answer'], 0) # 0 % 4 = 0
        self.assertEqual(item0['in_domain'], 1)

    @patch('datasets.load_dataset')
    @patch('numpy.random.choice')
    def test_getitem_list_creates_subset(self, mock_np_choice, mock_hf_load_dataset):
        with patch('os.path.exists', return_value=False), patch('pickle.dump'):
            mock_hf_load_dataset.return_value = mock_hf_dataset_obj('subj1', 10)
            mock_np_choice.return_value = np.arange(5) # Load 5 items
            dataset = MMLUDataset(sample_size=5, domains=['subj1'], cache_base_dir=TEST_CACHE_DIR, in_domain=False)

        indices_to_get = [0, 2, 4]
        subset = dataset[indices_to_get]

        self.assertIsInstance(subset, MMLUDataset)
        self.assertEqual(len(subset.data), 3)
        self.assertIn("subj1 Question 1", subset.data[0]) # Original data[0]
        self.assertEqual(subset.answers[0], 0) # 0 % 4
        self.assertEqual(subset.in_domain_flags[0], 0) # in_domain=False for parent

    def test_empty_domains_init(self):
        # If _get_domains returns empty list (e.g. domains=[], in_domain=False, and mmlu_subjects is exhaustive)
        with patch.object(MMLUDataset, '_get_domains', return_value=[]):
            dataset = MMLUDataset(sample_size=10, domains="irrelevant", cache_base_dir=TEST_CACHE_DIR)
            self.assertEqual(len(dataset.data), 0)
            self.assertEqual(dataset.train_indices, [])
            self.assertEqual(dataset.test_indices, [])

    @patch('datasets.load_dataset')
    def test_subject_with_no_data_from_hf(self, mock_hf_load_dataset):
        with patch('os.path.exists', return_value=False), patch('pickle.dump'):
            mock_hf_load_dataset.return_value = mock_hf_dataset_obj('empty_subj', 0) # Subject has 0 items
            dataset = MMLUDataset(sample_size=10, domains=['empty_subj'], cache_base_dir=TEST_CACHE_DIR)
            self.assertEqual(len(dataset.data), 0)

    @patch('datasets.load_dataset')
    @patch('numpy.random.choice')
    def test_dataset_with_single_item_after_load(self, mock_np_choice, mock_hf_load_dataset):
         with patch('os.path.exists', return_value=False), patch('pickle.dump'):
            mock_hf_load_dataset.return_value = mock_hf_dataset_obj('single_subj', 1)
            mock_np_choice.return_value = np.array([0]) # Choose the only item
            dataset = MMLUDataset(sample_size=1, domains=['single_subj'], cache_base_dir=TEST_CACHE_DIR)

            self.assertEqual(len(dataset.data), 1)
            self.assertEqual(len(dataset.train_indices), 1) # All to train
            self.assertEqual(len(dataset.test_indices), 0)
            
            item = dataset[0]
            self.assertIn("single_subj Question 1", item['question'])
            
            train_subset = dataset.get_train_dataset()
            test_subset = dataset.get_test_dataset()
            self.assertEqual(len(train_subset.data), 1)
            self.assertEqual(len(test_subset.data), 0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
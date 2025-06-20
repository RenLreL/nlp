import unittest
import pandas as pd
import torch
from transformers import DistilBertTokenizer
from src.training.text_classification_dataset import TextClassificationDataset

class TestTextClassificationDataset(unittest.TestCase):

    def setUp(self):
        """Set up testing environment before each test method."""
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.sample_data = pd.DataFrame({
            'text': [
                "This is the first sentence.",
                "This is a much longer second sentence designed to exceed the max length.",
                "The third sentence."
            ],
            'label_id': [0, 1, 2]
        })
        self.max_len = 16
        self.dataset = TextClassificationDataset(
            dataframe=self.sample_data,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )

    def test_dataset_length(self):
        """Tests if the dataset returns the correct length."""
        self.assertEqual(len(self.dataset), 3)

    def test_getitem(self):
        """
        Tests the __getitem__ method for correct output format, keys, and tensor shapes.
        """
        # Get the first item from the dataset
        item = self.dataset[0]

        # 1. Check if the output is a dictionary and has the correct keys
        self.assertIsInstance(item, dict)
        expected_keys = ['input_ids', 'attention_mask', 'labels']
        for key in expected_keys:
            self.assertIn(key, item)

        # 2. Check the shape of the tensors
        self.assertEqual(item['input_ids'].shape, (self.max_len,))
        self.assertEqual(item['attention_mask'].shape, (self.max_len,))

        # 3. Check the data types of the tensors
        self.assertEqual(item['input_ids'].dtype, torch.long)
        self.assertEqual(item['attention_mask'].dtype, torch.long)
        self.assertEqual(item['labels'].dtype, torch.long)

        # 4. Check the content of the labels tensor
        self.assertTrue(torch.equal(item['labels'], torch.tensor(0, dtype=torch.long)))

    def test_getitem_truncation(self):
        """
        Tests if truncation is correctly applied to long sequences.
        """
        # This text is longer than max_len and should be truncated
        item = self.dataset[1]

        self.assertEqual(item['input_ids'].shape, (self.max_len,))
        # Find the actual length of the tokenized (non-padded) sequence
        actual_token_length = item['attention_mask'].sum().item()
        self.assertEqual(actual_token_length, self.max_len)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
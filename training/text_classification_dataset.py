"""
Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-06-20: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""

import json
import os
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

class TextClassificationDataset(Dataset):
    """
    Custom PyTorch Dataset for text classification tasks.

    This dataset class handles the tokenization and preparation of text data
    for input into a transformer model like DistilBert.

    Attributes:
        tokenizer: The tokenizer to use for encoding the text.
        data (pd.DataFrame): The dataframe containing the text and labels.
        text (pd.Series): The column of text data.
        targets (pd.Series): The column of label IDs.
        max_len (int): The maximum sequence length for tokenization.
    """

    def __init__(self, dataframe, tokenizer, max_len):
        """
        Initializes the TextClassificationDataset.

        Args:
            dataframe (pd.DataFrame): The input dataframe.
            tokenizer: The transformers tokenizer.
            max_len (int): The maximum length for tokenized sequences.
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = dataframe.label_id
        self.max_len = max_len

    def __len__(self):
        """Returns the number of examples in the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at a given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input_ids, attention_mask,
                  and labels for the model.
        """
        text = str(self.text[index])
        target = int(self.targets[index])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.long)
        }
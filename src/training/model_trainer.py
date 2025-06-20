"""
Trains models on the media bias dataset and chooses the best one.

This script trains a DistilBert Model for up to ten epochs and chooses
the best one with earlystopping to avoid overfitting.

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
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from .text_classification_dataset import TextClassificationDataset
from .weighted_loss_trainer import WeightedLossTrainer


class ModelTrainer:
    """
    A class to encapsulate the model training and evaluation process.
    """

    def __init__(self, config):
        """
        Initializes the ModelTrainer with a configuration dictionary.

        Args:
            config (dict): A dictionary containing configuration parameters.
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.id2label = None
        self.label2id = None
        self.trainer = None

        self._prepare_device()

    def _prepare_device(self):
        """Sets the device for training (GPU or CPU)."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def _load_label_mappings(self):
        """Loads label-to-ID and ID-to-label mappings from JSON files."""
        try:
            with open(self.config['label2id_path'], "r") as f:
                self.label2id = json.load(f)
            with open(self.config['id2label_path'], "r") as f:
                self.id2label = json.load(f)
            self.config['num_labels'] = len(self.id2label)
        except FileNotFoundError:
            print("Error: Label mapping files not found.")
            raise

    def _load_data(self):
        """Loads the training, validation, and test datasets."""
        try:
            base_path = ''
            train_df = pd.read_csv(os.path.join(base_path, self.config['train_data_file']), sep="\\t")
            val_df = pd.read_csv(os.path.join(base_path, self.config['val_data_file']), sep="\\t")
            test_df = pd.read_csv(os.path.join(base_path, self.config['test_data_file']), sep="\\t")
            return train_df, val_df, test_df
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise

    def _create_datasets(self, train_df, val_df, test_df):
        """Creates PyTorch datasets from the dataframes."""
        self.train_dataset = TextClassificationDataset(
            train_df, self.tokenizer, self.config['max_length']
        )
        self.val_dataset = TextClassificationDataset(
            val_df, self.tokenizer, self.config['max_length']
        )
        self.test_dataset = TextClassificationDataset(
            test_df, self.tokenizer, self.config['max_length']
        )

    def _compute_metrics(self, pred):
        """Computes evaluation metrics."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def setup(self):
        """
        Sets up the tokenizer, model, and datasets for training.
        """
        print("(0/3) Setting up for training")
        self._load_label_mappings()

        print("(1/3) Initializing tokenizer...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.config['model_name']
        )

        print("(2/3) Loading and creating datasets...")
        train_df, val_df, test_df = self._load_data()
        self._create_datasets(train_df, val_df, test_df)

        print("(3/3) Loading pre-trained model...")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=self.config['num_labels'],
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)

    def train(self):
        """
        Configures and runs the training process.
        """
        if not all([self.tokenizer, self.model, self.train_dataset, self.val_dataset]):
            print("Setup must be run before training.")
            return

        print("Configuring Training ...")
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['train_batch_size'],
            per_device_eval_batch_size=self.config['eval_batch_size'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir=self.config['logging_dir'],
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="tensorboard",
            fp16=True
        )

        self.trainer = WeightedLossTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3)
            ]
        )

        print("Starting Training...")
        self.trainer.train()
        print("--- Training Finished ---")

    def evaluate(self):
        """
        Evaluates the trained model on the test set.
        """
        if not self.trainer:
            print("Model has not been trained yet.")
            return

        print("\nEvaluating on Test Set...")
        test_results = self.trainer.evaluate(eval_dataset=self.test_dataset)

        print("\n--- Test Set Results ---")
        for key, value in test_results.items():
            print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"../model/best-model-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)


    config = {
        'model_name': "distilbert-base-uncased",
        'output_dir': output_dir,
        'logging_dir': os.path.join(output_dir, 'logs'),
        'num_epochs': 10,
        'max_length': 256,
        'train_batch_size': 8,
        'eval_batch_size': 8,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'label2id_path': "output/labels/label2id.json",
        'id2label_path': "output/labels/id2label.json",
        'train_data_file': "output/data_train.tsv",
        'val_data_file': "output/data_val.tsv",
        'test_data_file': "output/data_test.tsv"
    }

    model_trainer = ModelTrainer(config)
    model_trainer.setup()
    model_trainer.train()
    model_trainer.evaluate()

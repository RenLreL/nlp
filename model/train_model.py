
"""
Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-06-04: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""

# from dataset_builder import DatasetBuilder
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
import json

from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    create_optimizer,
)
from datasets import Dataset 
from pathlib import Path


MODEL_NAME = "bert-base-uncased"


class TrainModel():


    def __init__(self):


        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        train_df, val_df, test_df = (
            pd.read_csv("intermediary_data_test.tsv", sep="\t"),
            pd.read_csv("intermediary_data_train.tsv", sep="\t"),
            pd.read_csv("intermediary_data_val.tsv", sep="\t")
        )

        id2labeljson = Path("id2label.json")
        with id2labeljson.open("r", encoding="utf-8") as f:
            id2label = json.load(f)

        label2idjson = Path("label2id.json")
        with label2idjson.open("r", encoding="utf-8") as f:
            label2id = json.load(f)


        train_tokenized, val_tokenized = self.tokenize_datasets(train_df, val_df)
        print(val_tokenized)

        model = TFAutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )

        epochs = 1#3
        steps_per_epoch = len(train_tokenized)
        num_train_steps = steps_per_epoch * epochs
        optimizer, schedule = create_optimizer(
            init_lr=2e-5,
            num_warmup_steps=int(0.1 * num_train_steps),
            num_train_steps=num_train_steps,
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        history = model.fit(
            train_tokenized,
            validation_data=val_tokenized,
            epochs=epochs,
        )

        model.save_pretrained("bert_news_classifier")
        self.tok.save_pretrained("bert_news_classifier")
        with open("bert_news_classifier/label2id.json", "w") as f:
            json.dump(label2id, f, indent=2)



    # def build_dataset(self):

    #     dataset_builder = DatasetBuilder()
    #     dataset = dataset_builder.get_dataset()
    #     train_df, val_df, test_df = dataset["train"], dataset["val"], dataset["test"]

    #     return train_df, val_df, test_df
    

    def tokenize_datasets(self, train_df, val_df):

        train_ds_hf = Dataset.from_pandas(train_df[["text", "label_id"]]).map(
            self.tokenize, batched=True, remove_columns=["text"]
        )
        val_ds_hf = Dataset.from_pandas(val_df[["text", "label_id"]]).map(
            self.tokenize, batched=True, remove_columns=["text"]
        )

        train_tfds = train_ds_hf.to_tf_dataset(
            columns=["input_ids", "attention_mask"],
            label_cols=["label_id"],
            shuffle=True,
            batch_size=32,
        )
        val_tfds = val_ds_hf.to_tf_dataset(
            columns=["input_ids", "attention_mask"],
            label_cols=["label_id"],
            shuffle=False,
            batch_size=32,
        )

        return train_tfds, val_tfds

    def tokenize(self, batch):
        tokenized = self.tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        return tokenized





if __name__ == "__main__":
    
    train_model = TrainModel()
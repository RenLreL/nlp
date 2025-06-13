
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

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import keras
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from datasets import Dataset 
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


MODEL_NAME = "bert-base-uncased"


class TrainModel():


    def __init__(self, max_epochs=10):


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

        model = TFAutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )

        train_tokenized, val_tokenized = self.tokenize_datasets(train_df, val_df, model)

        optimizer = keras.optimizers.Adam(learning_rate=2e-5)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        y_train = train_df["label_id"].values

        class_weighting_steps = np.arange(len(label2id))
        weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=class_weighting_steps,
            y=y_train)

        class_weights = dict(enumerate(weights))
        print(class_weights)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            mode='max',
            restore_best_weights=True
        )

        model.fit(
            train_tokenized,
            validation_data=val_tokenized,
            epochs=max_epochs,
            class_weight=class_weights,
            callbacks=[early_stopping]
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
    

    def tokenize_datasets(self, train_df, val_df, model):

        train_ds_hf = Dataset.from_pandas(train_df[["text", "label_id"]]).map(
            self.tokenize, batched=True, remove_columns=["text"]
        )
        val_ds_hf = Dataset.from_pandas(val_df[["text", "label_id"]]).map(
            self.tokenize, batched=True, remove_columns=["text"]
        )

        train_ds_hf = Dataset.from_pandas(train_df[["text", "label_id"]])
        val_ds_hf = Dataset.from_pandas(val_df[["text", "label_id"]])

        tokenized_train = train_ds_hf.map(self.tokenize, batched=True)
        tokenized_val = val_ds_hf.map(self.tokenize, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=self.tok, return_tensors="tf")

        train_tfds = model.prepare_tf_dataset(
            tokenized_train,
            collate_fn=data_collator,
            label_col="label_id",
            batch_size=32,
            shuffle=True,
        )
        
        val_tfds = model.prepare_tf_dataset(
            tokenized_val,
            collate_fn=data_collator,
            label_col="label_id",
            batch_size=32,
            shuffle=False,
        )

        return train_tfds, val_tfds


    def tokenize(self, batch):
        tokenized = self.tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        
        return tokenized





if __name__ == "__main__":
    
    train_model = TrainModel()
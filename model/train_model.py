
"""
Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-06-04: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""


from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

import torch
from torch import nn

from datasets import Dataset, disable_caching
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import evaluate


MODEL_NAME = "bert-base-uncased"
disable_caching()      

class WeightedTrainer(Trainer):
    """
    A Hugging Face Trainer that injects per-class weights into CrossEntropyLoss.
    """
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fnc = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fnc(logits, labels)
        return (loss, outputs) if return_outputs else loss


class TrainModel():


    def __init__(self, max_epochs=10):


        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.accuracy_metric = evaluate.load("accuracy")
        
        train_df, val_df, test_df = (
            pd.read_csv("intermediary_data_train.tsv", sep="\t"),
            pd.read_csv("intermediary_data_val.tsv", sep="\t"),
            pd.read_csv("intermediary_data_test.tsv", sep="\t")
        )

        train_df.rename(columns={"label_id": "labels"}, inplace=True)
        val_df.rename(columns={"label_id": "labels"}, inplace=True)
        test_df.rename(columns={"label_id": "labels"}, inplace=True)


        id2labeljson = Path("id2label.json")
        with id2labeljson.open("r", encoding="utf-8") as f:
            id2label = json.load(f)

        label2idjson = Path("label2id.json")
        with label2idjson.open("r", encoding="utf-8") as f:
            label2id = json.load(f)


        train_ds, val_ds = self.tokenize_datasets(train_df, val_df)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )
        

        y_train = train_df["labels"].values

        weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(label2id)),
            y=y_train,
        )

        class_weights = torch.tensor(weights, dtype=torch.float32)

        data_collator = DataCollatorWithPadding(self.tok, return_tensors="pt")


        args = TrainingArguments(
            output_dir="bert_news_classifier",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            num_train_epochs=max_epochs,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_first_step=True,
            logging_steps=50,
            report_to="none",
            push_to_hub=False,
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tok,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()

        trainer.save_model("bert_news_classifier")
        self.tok.save_pretrained("bert_news_classifier")
        with open("bert_news_classifier/label2id.json", "w") as f:
            json.dump(label2id, f, indent=2)




    # def build_dataset(self):

    #     dataset_builder = DatasetBuilder()
    #     dataset = dataset_builder.get_dataset()
    #     train_df, val_df, test_df = dataset["train"], dataset["val"], dataset["test"]

    #     return train_df, val_df, test_df
    

    def tokenize_datasets(self, train_df, val_df):

        train_ds = Dataset.from_pandas(train_df[["text", "labels"]])
        val_ds = Dataset.from_pandas(val_df[["text", "labels"]])

        tokenized_train = train_ds.map(self.tokenize, batched=True, remove_columns=["text"])
        tokenized_val = val_ds.map(self.tokenize, batched=True, remove_columns=["text"])

        return tokenized_train, tokenized_val


    def tokenize(self, batch):

        tokenized = self.tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        return tokenized
    


    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        computed = TrainModel.accuracy_metric.compute(
            predictions=preds,
            references=labels
        )
        return computed





if __name__ == "__main__":
    
    train_model = TrainModel()
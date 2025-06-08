import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import json
from pathlib import Path

class Classifier:
    """
    A class to load a pre-trained BERT-based text classification model
    and perform predictions.
    """

    def __init__(self, model_dir: str = "bert_news_classifier"):
        """
        Initializes the Classifier by loading the tokenizer, model,
        and label mappings from the specified directory.

        Args:
            model_dir (str): The directory where the model, tokenizer,
                             and label mappings are saved.
        """
        self.model_dir = model_dir

        # Load tokenizer
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)

        # Load model
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.model_dir)

        # Confirm the label mapping came back intact (optional, for debugging)
        print(f"Model's internal label2id: {self.model.config.label2id}")

        # Load id2label mapping from file (as it might not be fully persisted
        # in config.label2id in the same format depending on HF versions)
        id2label_path = Path(self.model_dir).parent / "id2label.json"
        # Check if id2label.json exists in the MODEL_DIR, otherwise try root (old behavior)
        if id2label_path.exists():
            with id2label_path.open("r", encoding="utf-8") as f:
                self.id2label = json.load(f)
        else: # Fallback to looking in the current working directory if not in model_dir
            print(f"Warning: id2label.json not found in {self.model_dir}. Trying current directory.")
            id2label_root_path = Path("id2label.json")
            if id2label_root_path.exists():
                with id2label_root_path.open("r", encoding="utf-8") as f:
                    self.id2label = json.load(f)
            else:
                raise FileNotFoundError(
                    f"id2label.json not found in '{self.model_dir}' or current directory. "
                    "Please ensure it's in the model directory or specify its path."
                )

        # Compile the model (necessary if you plan to continue training or evaluate performance)
        # For pure inference, compilation isn't strictly required but doesn't hurt.
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def classify(self, text: str, max_len: int = 128, return_probs: bool = True):
        """
        Predicts the class of a given text.

        Args:
            text (str): The input text to classify.
            max_len (int): The maximum sequence length for tokenization. Defaults to 128.
            return_probs (bool): If True, returns a dictionary of class probabilities
                                 along with the predicted label. Defaults to True.

        Returns:
            str: The predicted class label.
            dict (optional): A dictionary mapping label names to their probabilities.
        """
        # Tokenize the input text
        enc = self.tok(
            text,
            truncation=True,
            padding=True, # Pads to max_length or longest sequence in batch
            max_length=max_len,
            return_tensors="tf", # Returns TensorFlow tensors
        )

        # Get logits from the model
        logits = self.model(enc).logits  # shape (1, num_labels)

        # Get the predicted class ID (index of max logit)
        pred_id = int(tf.argmax(logits, axis=-1)[0])

        # Map the predicted ID back to its human-readable label
        label = self.id2label[str(pred_id)] # id2label usually expects string keys

        if return_probs:
            # Calculate softmax probabilities
            probs = tf.nn.softmax(logits, axis=-1).numpy().flatten()
            # Create a dictionary of label -> probability
            prob_dict = {self.id2label[str(i)]: float(round(p, 4)) for i, p in enumerate(probs)}
            return label, prob_dict
        return label

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from pathlib import Path

class Classifier:
    """
    A class to load a pre-trained BERT-based text classification model
    and perform predictions using PyTorch.
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

        # Set device to GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device) # Move model to the specified device

        # Confirm the label mapping came back intact (optional, for debugging)
        print(f"Model's internal label2id: {self.model.config.label2id}")

        # Load id2label mapping from file (as it might not be fully persisted
        # in config.label2id in the same format depending on HF versions)
        id2label_path = Path(self.model_dir) / "id2label.json" # Assumes id2label.json is inside model_dir
        # Check if id2label.json exists in the MODEL_DIR
        if id2label_path.exists():
            with id2label_path.open("r", encoding="utf-8") as f:
                self.id2label = json.load(f)
        else:
            # Fallback to looking in the parent directory of model_dir (old behavior from original code)
            print(f"Warning: id2label.json not found in {self.model_dir}. Trying parent directory.")
            id2label_parent_path = Path(self.model_dir).parent / "id2label.json"
            if id2label_parent_path.exists():
                with id2label_parent_path.open("r", encoding="utf-8") as f:
                    self.id2label = json.load(f)
            else:
                # Final fallback to looking in the current working directory if not in model_dir or its parent
                print(f"Warning: id2label.json not found in {id2label_parent_path}. Trying current directory.")
                id2label_root_path = Path("id2label.json")
                if id2label_root_path.exists():
                    with id2label_root_path.open("r", encoding="utf-8") as f:
                        self.id2label = json.load(f)
                else:
                    raise FileNotFoundError(
                        f"id2label.json not found in '{self.model_dir}', its parent directory, or current directory. "
                        "Please ensure it's in the model directory or specify its path."
                    )

        # For PyTorch, explicit compilation like in Keras is not typically done for inference.
        # The model is ready to use after loading and moving to device.
        self.model.eval() # Set model to evaluation mode

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
            padding="max_length", # Pads to max_length
            max_length=max_len,
            return_tensors="pt", # Returns PyTorch tensors
        )

        # Move input tensors to the same device as the model
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad(): # Disable gradient calculation for inference
            # Get logits from the model
            # For PyTorch, model output is typically a tuple, with logits being the first element
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape (1, num_labels)

        # Get the predicted class ID (index of max logit)
        pred_id = int(torch.argmax(logits, dim=-1)[0])

        # Map the predicted ID back to its human-readable label
        label = self.id2label[str(pred_id)] # id2label usually expects string keys

        if return_probs:
            # Calculate softmax probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy().flatten()
            # Create a dictionary of label -> probability
            prob_dict = {self.id2label[str(i)]: float(round(p, 4)) for i, p in enumerate(probs)}
            return label, prob_dict
        return label
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class Classifier:
    """
    A class to load a pre-trained BERT-based text classification model
    and perform predictions using PyTorch.
    """

    def __init__(self, *, model_dir: str = "checkpoint-51422", tokenizer_dir: str = "best_model"):
        """
        Initializes the Classifier by loading the tokenizer, model,
        and label mappings from the specified directory.

        Args:
            model_dir (str): The directory where the model, tokenizer,
                             and label mappings are saved.
        """

        project_root = Path(__file__).parent.parent.parent
        self.model_path = project_root / model_dir
        self.tokenizer_path = project_root / tokenizer_dir

        # Set device to GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure the model directory exists BEFORE trying to load from it
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found or is not a directory at: {self.model_path}")

        # Load tokenizer
        self.tok = DistilBertTokenizer.from_pretrained(self.tokenizer_path)

        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device) # Move model to the specified device
        
        # Load id2label mapping from file
        self.id2label = None
        
        if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
            self.id2label = {str(k): v for k, v in self.model.config.id2label.items()}
        else:
            raise ValueError(
                "Label mapping (id2label) not found in the model's configuration. "
                "Please ensure the model was saved with 'id2label' in its config.json, "
                "or provide a separate id2label.json file if intended."
            )

        self.model.eval()

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

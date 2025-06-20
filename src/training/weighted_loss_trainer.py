"""
Copyright 2025
Authors: Laélia Chi <lae.chi.22@heilbronn.dhbw.de>;
    Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>;
    Yaren Sude Erol <yar.erol.22@heilbronn.dhbw.de>;
    Leon Gerke <leo.gerke.22@heilbronn.dhbw.de>;
    Dominic von Olnhausen <dom.vonolnhausen.22@heilbronn.dhbw.de>
(Edited 2025-06-20: Marco Diepold <mar.diepold.22@heilbronn.dhbw.de>)
"""

import torch
import numpy as np
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight


class WeightedLossTrainer(Trainer):
    """
    A custom Trainer that uses a weighted cross-entropy loss.

    This is useful for handling class imbalance by assigning higher weights
    to minority classes during loss calculation.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the WeightedLossTrainer."""
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #:
        """
        Computes the loss for a batch of inputs.

        Args:
            model (torch.nn.Module): The model to compute the loss for.
            inputs (dict): The inputs to the model.
            return_outputs (bool): Whether to return the model's outputs
                                   along with the loss.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, dict]]: The loss, or a
            tuple of the loss and model outputs.
        """
        train_labels = self.train_dataset.targets
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        self.class_weights = class_weight_tensor.to(device)

        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

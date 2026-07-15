"""
Reusable helper functions: reproducibility seeding and metric computation
for the Hugging Face Trainer.
"""

import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


from . import config


def set_seed(seed: int = None) -> None:
    """Fix all relevant random seeds for reproducibility."""
    if seed is None:
        seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred):
    """
    Callback used by transformers.Trainer.
    eval_pred is a tuple (logits, labels); Trainer handles the numpy
    conversion for us.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }

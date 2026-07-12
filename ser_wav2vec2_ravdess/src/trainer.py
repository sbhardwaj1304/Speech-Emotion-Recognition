"""
Custom Hugging Face Trainer with class-weighted loss and
discriminative learning rates for wav2vec2 fine-tuning.
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import EarlyStoppingCallback, Trainer

from . import config


class SERTrainer(Trainer):
    """Trainer tuned for imbalanced 8-way emotion classification."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = config.LABEL_SMOOTHING
        # Custom loss ignores `num_items_in_batch`; tell HF Trainer to scale
        # loss by gradient-accumulation steps so logs show ~log(8) at init, not 2x.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.label_smoothing,
        )
        loss = loss_fct(logits.view(-1, config.NUM_LABELS), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        transformer_params: List[torch.nn.Parameter] = []
        head_params: List[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name or "projector" in name:
                head_params.append(param)
            else:
                transformer_params.append(param)

        optimizer_groups = []
        if transformer_params:
            optimizer_groups.append(
                {
                    "params": transformer_params,
                    "lr": max(config.LR_ENCODER, 1e-8),
                    "weight_decay": config.WEIGHT_DECAY,
                }
            )
        if head_params:
            optimizer_groups.append(
                {
                    "params": head_params,
                    "lr": config.LR_HEAD,
                    "weight_decay": config.WEIGHT_DECAY,
                }
            )

        self.optimizer = AdamW(optimizer_groups, betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer


def build_callbacks() -> List[EarlyStoppingCallback]:
    return [
        EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)
    ]


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """Inverse-frequency weights so rare classes (e.g. neutral) matter more."""
    counts = torch.bincount(torch.tensor(labels), minlength=config.NUM_LABELS).float()
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (config.NUM_LABELS * counts)
    weights = weights / weights.sum() * config.NUM_LABELS
    return weights

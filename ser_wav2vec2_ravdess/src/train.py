"""
End-to-end training script using the official Hugging Face Trainer API.

Supports two-stage fine-tuning:
  Stage 1 — classifier head warmup (encoder frozen)
  Stage 2 — full transformer fine-tune (CNN still frozen)
  Stage 3 — optional polish on train+val combined
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from datasets import concatenate_datasets
from transformers import TrainingArguments

from . import config
from .dataset import (
    build_hf_datasets,
    get_feature_extractor,
    load_metadata,
    preprocess_batch,
    split_metadata,
)
from .model import build_model, set_training_stage
from .trainer import SERTrainer, build_callbacks, compute_class_weights
from .utils import compute_metrics, set_seed


@dataclass
class DataCollatorSER:
    """Dynamic padding collator with on-the-fly augmentation."""

    feature_extractor: object

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Pad a batch of variable-length input_values to the longest clip in
        the batch (not a fixed length), and apply waveform augmentation
        during training only (relies on the Trainer's no_grad() context
        during eval to skip augmentation at eval time).
        """
        input_values = [f["input_values"] for f in features]
        labels = [f["label"] for f in features]

        if config.USE_TRAIN_AUGMENTATION and torch.is_grad_enabled():
            augmented = []
            for iv in input_values:
                arr = np.asarray(iv, dtype=np.float32)
                if random.random() < config.AUGMENT_PROB:
                    arr = arr + np.random.randn(len(arr)).astype(np.float32) * config.AUGMENT_NOISE_STD
                    gain = random.uniform(*config.AUGMENT_GAIN_RANGE)
                    arr = arr * gain
                    shift = random.randint(-800, 800)
                    if shift != 0:
                        arr = np.roll(arr, shift)
                augmented.append(arr.tolist())
            input_values = augmented

        batch = self.feature_extractor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def build_training_args(
    num_epochs: int,
    lr_encoder: float,
    lr_head: float,
    output_dir: Optional[str] = None,
    use_early_stopping: bool = True,
) -> TrainingArguments:
    """Build HF TrainingArguments for one training stage (see _run_stage)."""
    config.NUM_EPOCHS = num_epochs
    config.LR_ENCODER = lr_encoder
    config.LR_HEAD = lr_head

    return TrainingArguments(
        output_dir=output_dir or config.MODEL_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        num_train_epochs=num_epochs,
        learning_rate=lr_encoder,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type=config.LR_SCHEDULER,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,
        bf16=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=True,
        optim="adamw_torch",
        report_to=["none"],
        seed=config.SEED,
    )


def prepare_datasets():
    """Load metadata, split by actor, decode audio, and extract wav2vec2 features."""
    print("Loading metadata...")
    df = load_metadata()
    train_df, val_df, test_df = split_metadata(df)
    print(
        f"Speaker-independent split -> "
        f"train: {len(train_df)} clips ({train_df['actor'].nunique()} actors) | "
        f"val: {len(val_df)} clips ({val_df['actor'].nunique()} actors) | "
        f"test: {len(test_df)} clips ({test_df['actor'].nunique()} actors)"
    )

    print("Building Hugging Face datasets (decoding audio)...")
    train_ds, val_ds, test_ds = build_hf_datasets(train_df, val_df, test_df)

    print(f"Loading feature extractor for {config.MODEL_CHECKPOINT}...")
    feature_extractor = get_feature_extractor()

    print("Extracting wav2vec2 input features...")
    map_kwargs = dict(
        batched=True,
        batch_size=8,
        remove_columns=["audio"],
        load_from_cache_file=False,
    )
    train_ds = train_ds.map(lambda b: preprocess_batch(b, feature_extractor), **map_kwargs)
    val_ds = val_ds.map(lambda b: preprocess_batch(b, feature_extractor), **map_kwargs)
    test_ds = test_ds.map(lambda b: preprocess_batch(b, feature_extractor), **map_kwargs)

    return train_ds, val_ds, test_ds, feature_extractor


def _make_trainer(
    model,
    train_ds,
    val_ds,
    feature_extractor,
    class_weights,
    training_args: TrainingArguments,
    use_early_stopping: bool = True,
) -> SERTrainer:
    """Wire up a SERTrainer with the data collator, metrics, and class weights."""
    data_collator = DataCollatorSER(feature_extractor=feature_extractor)
    callbacks = build_callbacks() if use_early_stopping else []

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=callbacks,
    )
    try:
        return SERTrainer(processing_class=feature_extractor, **trainer_kwargs)
    except TypeError:
        return SERTrainer(tokenizer=feature_extractor, **trainer_kwargs)


def _run_stage(
    stage_name: str,
    model,
    train_ds,
    val_ds,
    feature_extractor,
    class_weights,
    num_epochs: int,
    lr_encoder: float,
    lr_head: float,
    output_subdir: str,
    use_early_stopping: bool = True,
) -> SERTrainer:
    """Train one stage (head warmup, full fine-tune, or polish) and return the trainer."""
    stage_dir = os.path.join(config.MODEL_DIR, output_subdir)
    os.makedirs(stage_dir, exist_ok=True)

    print(
        f"\n=== {stage_name} ===\n"
        f"epochs={num_epochs}, lr_encoder={lr_encoder}, lr_head={lr_head}, "
        f"batch={config.BATCH_SIZE}x{config.GRAD_ACCUM_STEPS}"
    )

    args = build_training_args(
        num_epochs=num_epochs,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        output_dir=stage_dir,
        use_early_stopping=use_early_stopping,
    )
    trainer = _make_trainer(
        model, train_ds, val_ds, feature_extractor, class_weights, args, use_early_stopping
    )
    trainer.train()
    if use_early_stopping and trainer.state.best_model_checkpoint:
        print(f"Best checkpoint for {stage_name}: {trainer.state.best_model_checkpoint}")
    return trainer


def main(two_stage: bool = True):
    """
    Run the full training pipeline: load data, build the model, fine-tune
    (either in one shot or through the staged head/encoder/polish schedule),
    and return the trainer plus the held-out test set for evaluation.
    """
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_ds, val_ds, test_ds, feature_extractor = prepare_datasets()

    print(f"Building model: {config.MODEL_CHECKPOINT}")
    model = build_model()

    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        train_labels = train_ds["label"]
        class_weights = compute_class_weights(train_labels)
        print(
            "Class weights:",
            {config.LABELS[i]: round(w, 3) for i, w in enumerate(class_weights.tolist())},
        )

    if two_stage:
        # Stage 1: head warmup
        set_training_stage(model, "head_only")
        trainer = _run_stage(
            "Stage 1 — head warmup",
            model,
            train_ds,
            val_ds,
            feature_extractor,
            class_weights,
            config.STAGE1_EPOCHS,
            lr_encoder=0.0,
            lr_head=config.STAGE1_LR_HEAD,
            output_subdir="stage1-head",
        )

        # Stage 2: unfreeze transformer
        set_training_stage(model, "full")
        trainer = _run_stage(
            "Stage 2 — encoder fine-tune",
            model,
            train_ds,
            val_ds,
            feature_extractor,
            class_weights,
            config.STAGE2_EPOCHS,
            lr_encoder=config.LR_ENCODER,
            lr_head=config.LR_HEAD,
            output_subdir="stage2-full",
        )

        # Stage 3: polish on train+val (more data, low LR)
        if config.FINAL_FINETUNE_ON_TRAINVAL:
            combined_train = concatenate_datasets([train_ds, val_ds])
            set_training_stage(model, "full")
            trainer = _run_stage(
                "Stage 3 — train+val polish",
                model,
                combined_train,
                val_ds,
                feature_extractor,
                class_weights,
                config.FINAL_FINETUNE_EPOCHS,
                lr_encoder=config.FINAL_LR_ENCODER,
                lr_head=config.FINAL_LR_HEAD,
                output_subdir="stage3-polish",
                use_early_stopping=False,
            )
    else:
        set_training_stage(model, "full")
        trainer = _run_stage(
            "Single-stage training",
            model,
            train_ds,
            val_ds,
            feature_extractor,
            class_weights,
            config.NUM_EPOCHS,
            lr_encoder=config.LR_ENCODER,
            lr_head=config.LR_HEAD,
            output_subdir="single-stage",
        )

    if two_stage and config.FINAL_FINETUNE_ON_TRAINVAL:
        print("Skipping validation metrics: stage 3 trained on train+val combined, "
              "so val_ds is no longer held out and these numbers would be meaningless.")
    else:
        print("Evaluating on validation set...")
        val_metrics = trainer.evaluate(val_ds)
        print(val_metrics)

    print(f"Saving final model to {config.MODEL_DIR}")
    trainer.save_model(config.MODEL_DIR)
    feature_extractor.save_pretrained(config.MODEL_DIR)

    return trainer, test_ds


if __name__ == "__main__":
    main()

"""
Model construction: wav2vec2 adapted for 8-way emotion classification.
"""

from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification

from . import config


def build_model(
    model_checkpoint: str = config.MODEL_CHECKPOINT,
    freeze_feature_encoder: bool = True,
) -> Wav2Vec2ForSequenceClassification:
    """
    Load a pretrained wav2vec2 checkpoint and swap in a fresh classification
    head sized for our emotion labels. The CNN feature encoder is frozen by
    default since RAVDESS is too small to fine-tune it without overfitting.
    """
    model_config = Wav2Vec2Config.from_pretrained(
        model_checkpoint,
        num_labels=config.NUM_LABELS,
        label2id=config.LABEL2ID,
        id2label=config.ID2LABEL,
        finetuning_task="audio-classification",
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
    )

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_checkpoint,
        config=model_config,
        dtype="float32",
    )

    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    if hasattr(model.wav2vec2, "masked_spec_embed"):
        model.wav2vec2.masked_spec_embed.requires_grad_(False)

    return model

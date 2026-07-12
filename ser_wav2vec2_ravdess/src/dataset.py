"""
Dataset utilities: load metadata.csv, perform an actor-independent
train/val/test split, build Hugging Face `datasets.Dataset` objects with
audio decoding, and extract wav2vec2 features.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from datasets import ClassLabel, Dataset
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor

from . import config


# --------------------------------------------------------------------------
# Metadata loading
# --------------------------------------------------------------------------
# Map common alternate spellings (e.g. from hand-made CSVs) to canonical labels.
EMOTION_ALIASES = {
    "fear": "fearful",
    "surprise": "surprised",
}


def _normalize_emotion(value: str) -> str:
    label = str(value).strip().lower()
    return EMOTION_ALIASES.get(label, label)


def load_metadata(metadata_csv: str = config.METADATA_CSV) -> pd.DataFrame:
    """
    Load metadata.csv. Expected columns: `path` (or `filepath`), `emotion`.
    An `actor` column is inferred from the filename if not present, so that
    the split can be done per-speaker (avoids train/test leakage).
    """
    df = pd.read_csv(metadata_csv)

    if "path" not in df.columns and "filepath" in df.columns:
        df = df.rename(columns={"filepath": "path"})

    required_cols = {"path", "emotion"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {missing}")

    df["emotion"] = df["emotion"].map(_normalize_emotion)

    # Keep only labels we recognize
    valid_labels = set(config.LABELS)
    unknown = set(df["emotion"].unique()) - valid_labels
    if unknown:
        raise ValueError(
            f"metadata.csv contains unknown emotion labels: {unknown}. "
            f"Expected one of {sorted(valid_labels)}"
        )

    if "actor" not in df.columns:
        df["actor"] = df["path"].apply(_infer_actor_from_path)

    return df.reset_index(drop=True)


def _infer_actor_from_path(path: str) -> str:
    """
    RAVDESS filenames follow the pattern:
        modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav
    e.g. 03-01-05-01-02-01-12.wav -> actor id = "12"
    """
    try:
        stem = path.replace("\\", "/").split("/")[-1].split(".")[0]
        return stem.split("-")[-1]
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Splitting (actor-independent / speaker-independent)
# --------------------------------------------------------------------------
def split_metadata(
    df: pd.DataFrame, seed: int = config.SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by actor (speaker), not by row, so the same speaker never appears
    in both train and test. Returns (train_df, val_df, test_df).
    """
    actors = df["actor"].unique()

    holdout_size = config.TEST_SIZE + config.VAL_SIZE
    train_actors, temp_actors = train_test_split(
        actors, test_size=holdout_size, random_state=seed
    )

    val_fraction_of_temp = config.VAL_SIZE / holdout_size
    val_actors, test_actors = train_test_split(
        temp_actors, test_size=(1 - val_fraction_of_temp), random_state=seed
    )

    train_df = df[df["actor"].isin(train_actors)].reset_index(drop=True)
    val_df = df[df["actor"].isin(val_actors)].reset_index(drop=True)
    test_df = df[df["actor"].isin(test_actors)].reset_index(drop=True)

    return train_df, val_df, test_df


# --------------------------------------------------------------------------
# Hugging Face Dataset construction
# --------------------------------------------------------------------------
def _load_audio_file(path: str) -> dict:
    """Load a wav file and resample to the project sampling rate."""
    array, sr = sf.read(path, dtype="float32", always_2d=False)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != config.SAMPLING_RATE:
        array = librosa.resample(array, orig_sr=sr, target_sr=config.SAMPLING_RATE)
    return {"array": array, "sampling_rate": config.SAMPLING_RATE}


def build_hf_datasets(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[Dataset, Dataset, Dataset]:
    """Convert pandas DataFrames into HF Datasets with decoded audio + int labels."""
    class_label = ClassLabel(names=config.LABELS)

    def to_dataset(df: pd.DataFrame) -> Dataset:
        records = []
        for _, row in df.iterrows():
            audio = _load_audio_file(row["path"])
            records.append(
                {
                    "audio": audio,
                    "label": class_label.str2int(row["emotion"]),
                }
            )
        return Dataset.from_list(records)

    return to_dataset(train_df), to_dataset(val_df), to_dataset(test_df)


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def get_feature_extractor(
    model_checkpoint: str = config.MODEL_CHECKPOINT,
) -> Wav2Vec2FeatureExtractor:
    return Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)


def preprocess_batch(batch, feature_extractor: Wav2Vec2FeatureExtractor):
    """
    Map function applied with batched=True. Converts raw audio arrays into
    input_values. Padding is deferred to the data collator (dynamic padding
    is required for stable wav2vec2 fine-tuning).
    """
    audio_arrays = [x["array"] for x in batch["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=config.SAMPLING_RATE,
        truncation=True,
        max_length=config.MAX_LENGTH,
        padding=False,
    )
    batch["input_values"] = inputs["input_values"]
    batch["length"] = [len(v) for v in inputs["input_values"]]
    return batch

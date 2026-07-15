"""
Central configuration for the Speech Emotion Recognition (SER) project.
All paths and hyperparameters are defined here so every script (training,
evaluation, Gradio app, notebook) stays in sync.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.getenv("SER_DATA_ROOT", PROJECT_ROOT.parent / "RAVDESS_DATA"))
DATA_DIR = os.getenv("SER_DATA_DIR", str(DATA_ROOT / "ravdess"))
METADATA_CSV = os.getenv("SER_METADATA_CSV", str(DATA_ROOT / "metadata_local.csv"))

OUTPUT_DIR = os.getenv("SER_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))
MODEL_DIR = os.path.join(OUTPUT_DIR, "wav2vec2-ser-ravdess-optimized")

# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
SAMPLING_RATE = 16000
MAX_DURATION_SECONDS = 5.0
MAX_LENGTH = int(SAMPLING_RATE * MAX_DURATION_SECONDS)

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
MODEL_CHECKPOINT = "facebook/wav2vec2-base-960h"

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

LABELS = sorted(set(EMOTION_MAP.values()))
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# --------------------------------------------------------------------------
# Train / val / test split (actor-independent)
# --------------------------------------------------------------------------
SEED = 22
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# --------------------------------------------------------------------------
# Training (single-stage full fine-tune: CNN frozen, transformer + head trained
# together with discriminative learning rates)
# --------------------------------------------------------------------------
NUM_EPOCHS = 25
LR_ENCODER = 2e-5
LR_HEAD = 5e-5

BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2          # effective batch = 8 * 2 = 16
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
LABEL_SMOOTHING = 0.08
EARLY_STOPPING_PATIENCE = 10
USE_CLASS_WEIGHTS = True
USE_TRAIN_AUGMENTATION = True
AUGMENT_NOISE_STD = 0.008
AUGMENT_PROB = 0.7
AUGMENT_GAIN_RANGE = (0.85, 1.15)

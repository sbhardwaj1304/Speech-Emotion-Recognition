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
MODEL_DIR = os.path.join(OUTPUT_DIR, "wav2vec2-ser-ravdess-v3")

# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
SAMPLING_RATE = 16000
MAX_DURATION_SECONDS = 5.0
MAX_LENGTH = int(SAMPLING_RATE * MAX_DURATION_SECONDS)

# --------------------------------------------------------------------------
# Model — large wav2vec2 for higher capacity
# --------------------------------------------------------------------------
MODEL_CHECKPOINT = "facebook/wav2vec2-large-960h"

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
# Two-stage training (head warmup -> full encoder fine-tune -> train+val polish)
# --------------------------------------------------------------------------
STAGE1_EPOCHS = 10
STAGE1_LR_HEAD = 1e-4

STAGE2_EPOCHS = 40
LR_ENCODER = 8e-6
LR_HEAD = 2e-5

FINAL_FINETUNE_EPOCHS = 15
FINAL_LR_ENCODER = 3e-6
FINAL_LR_HEAD = 8e-6
FINAL_FINETUNE_ON_TRAINVAL = True

# Legacy single-run aliases (used by run_training --epochs override)
NUM_EPOCHS = STAGE2_EPOCHS
LEARNING_RATE = LR_ENCODER

BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16         # 1 GPU -> effective batch = 4 * 1 * 16 = 64
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

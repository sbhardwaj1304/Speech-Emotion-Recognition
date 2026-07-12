# ser_wav2vec2_ravdess — Technical Documentation

Speaker-independent speech emotion recognition fine-tuning wav2vec2 on RAVDESS.

**Parent repo:** https://github.com/sbhardwaj1304/Speech-Emotion-Recognition  
**Best model:** https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized (**68.3%** test accuracy)

---

## Model weights

### Published (Hugging Face)

| Model | URL | Accuracy |
|-------|-----|----------|
| **Best — optimized** | https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized | **68.3%** |

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

model_dir = "sbh013/wav2vec2-ser-ravdess-optimized"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
```

---

## How anyone can load and use the model

No training data or local checkpoints needed. Weights download from Hugging Face
on first use (~361 MB, cached at `~/.cache/huggingface/hub/`).

**Model:** https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized

### Install

```bash
git clone https://github.com/sbhardwaj1304/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition/ser_wav2vec2_ravdess
pip install torch transformers librosa soundfile gradio
```

### Load from Hugging Face

```python
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

MODEL_ID = "sbh013/wav2vec2-ser-ravdess-optimized"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

### Predict emotion from a .wav file

```python
import librosa
import torch

def predict_emotion(audio_path: str) -> dict:
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).squeeze().cpu().numpy()
    id2label = model.config.id2label
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}

result = predict_emotion("speech.wav")
print(max(result, key=result.get), result)
```

**Labels:** angry, calm, disgust, fearful, happy, neutral, sad, surprised

### Gradio demo (mic or upload)

```bash
SER_MODEL_DIR=sbh013/wav2vec2-ser-ravdess-optimized python app/gradio_app.py
```

### Download weights to disk (optional)

```bash
pip install huggingface_hub
huggingface-cli download sbh013/wav2vec2-ser-ravdess-optimized --local-dir ./my-model
```

```python
model = Wav2Vec2ForSequenceClassification.from_pretrained("./my-model")
```

### Tips

- Use 1–5 second mono speech clips at 16 kHz
- English speech works best (wav2vec2-base-960h pretraining)
- Predicts vocal emotion, not word meaning

---

| Folder | Backbone | Epochs | Test acc | Notes |
|--------|----------|--------|----------|-------|
| `outputs/wav2vec2-ser-ravdess-optimized/` | wav2vec2-**base** | 25 | **68.3%** | **Best — uploaded to HF** |
| `outputs/wav2vec2-ser-ravdess/` | wav2vec2-base | 10 | 53.9% | First fixed baseline |
| `outputs/wav2vec2-ser-ravdess-v3/` | wav2vec2-**large** | 3-stage | TBD | Training in progress |

Each checkpoint folder contains: `model.safetensors`, `config.json`, `preprocessor_config.json`, `checkpoint-*/`.

---

## Best model configuration (68.3%)

From `training_optimized.log`:

```
Backbone:              facebook/wav2vec2-base-960h
Training:              Single-stage (full encoder + head)
Epochs:                25
LR encoder:            2e-5
LR head:               5e-5
Batch size:            8
Gradient accumulation: 2  (effective batch = 16)
Optimizer:             AdamW (discriminative LRs)
LR scheduler:          Cosine, warmup 6%
Weight decay:          0.01
Label smoothing:       0.08
Class weights:         neutral=1.778, others=0.889
Augmentation:          noise σ=0.008, gain 0.85–1.15, 70% prob, ±800 shift
Precision:             fp32
Padding:               Dynamic per-batch
Frozen:                CNN feature encoder
Selection metric:      macro-F1 on validation
Early stopping:        patience 10
```

### Results

- **Test accuracy:** 68.3%
- **Macro F1:** 0.676
- **Best classes:** disgust (F1 0.84), surprised (F1 0.79), calm (F1 0.76)
- **Weakest class:** sad (F1 0.37) — often confused with calm/neutral

Reports: `reports/classification_report.txt`, `reports/confusion_matrix.png`

---

## Current training (v3 — in progress)

Configured in `src/config.py`, logged to `training.log`:

| Stage | Epochs | LR encoder | LR head | What trains |
|-------|--------|------------|---------|-------------|
| 1 — head warmup | 10 | 0 | 1e-4 | Classifier only |
| 2 — encoder | 40 | 8e-6 | 2e-5 | Transformer + head |
| 3 — polish | 15 | 3e-6 | 8e-6 | Train+val combined |

```
Backbone:    facebook/wav2vec2-large-960h
Seed:        22
GPU:         1 (CUDA_VISIBLE_DEVICES=0)
Batch:       4 × 16 grad accum = effective 64
Output dir:  outputs/wav2vec2-ser-ravdess-v3/
```

```bash
tail -f training.log          # monitor
bash run_train.sh               # start/restart
```

---

## Data

| Item | Path |
|------|------|
| Audio (not in git) | `../RAVDESS_DATA/ravdess/` (1,440 wav, 24 actors) |
| Metadata | `../RAVDESS_DATA/metadata_local.csv` |
| Build metadata | `python scripts/prepare_data.py` |

**Split (speaker-independent):** 1,140 train / 120 val / 180 test clips across 19/2/3 actors.

---

## Commands

```bash
# Setup
pip install -r requirements.txt

# Prepare data
python run_training.py --prepare-only

# Train (1 GPU)
bash run_train.sh

# Quick smoke test (3 epochs)
python run_training.py --skip-prepare --epochs 3 --single-stage

# Evaluate
SER_MODEL_DIR=outputs/wav2vec2-ser-ravdess-optimized python -m src.evaluate

# Gradio demo
SER_MODEL_DIR=sbh013/wav2vec2-ser-ravdess-optimized python app/gradio_app.py
```

---

## Source files

| File | Role |
|------|------|
| `src/config.py` | All hyperparameters and paths |
| `src/dataset.py` | Metadata load, actor split, feature extraction |
| `src/model.py` | Wav2Vec2 classifier builder, stage freezing |
| `src/train.py` | 3-stage HF Trainer pipeline |
| `src/trainer.py` | Class-weighted loss, discriminative AdamW |
| `src/evaluate.py` | Test metrics + confusion matrix |
| `scripts/prepare_data.py` | Scan wav tree → metadata_local.csv |
| `app/gradio_app.py` | Web demo |

---

## Key design choices

- **Speaker-independent split** — actors assigned to train/val/test, not clips
- **Dynamic padding** — variable-length batches (not fixed 5 s)
- **fp32 training** — fp16 caused NaN gradients with wav2vec2
- **Class weights** — upweight rare `neutral` class (half the samples)
- **Macro-F1 checkpoint selection** — treats all emotions equally
- **Train loss ~2.08 at start** — if you see ~4, that was a logging bug (now fixed in `trainer.py`)

---

## Training progression

| Run | Config | Test acc |
|-----|--------|----------|
| Broken | base, bad padding/fp16 | ~13% |
| Fixed baseline | base, 10 ep | 53.9% |
| **Optimized** | base, 25 ep, weights+aug | **68.3%** |
| v3 (running) | large, 3-stage, seed 22 | TBD |

---

## Environment variables

```bash
export SER_DATA_ROOT=../RAVDESS_DATA
export SER_METADATA_CSV=../RAVDESS_DATA/metadata_local.csv
export SER_OUTPUT_DIR=outputs
export SER_MODEL_DIR=sbh013/wav2vec2-ser-ravdess-optimized
```

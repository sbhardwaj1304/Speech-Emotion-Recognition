# Speech Emotion Recognition — wav2vec2 + RAVDESS

Fine-tunes Hugging Face **wav2vec2** for **8-class speech emotion recognition** on the
[RAVDESS](https://zenodo.org/record/1188976) dataset, using a **speaker-independent**
train/val/test split, class-weighted loss, data augmentation, and a Gradio demo.

**GitHub:** https://github.com/sbhardwaj1304/Speech-Emotion-Recognition
**Best model (Hugging Face):** https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized

**Emotions (8 classes):** `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

## Team

**Sanchit Bhardwaj** ([@sbhardwaj1304](https://github.com/sbhardwaj1304)) and
**Utsav Kumar** ([@Utsavkumar001](https://github.com/Utsavkumar001))

---

## What this repo contains

| Path | Description |
|------|-------------|
| `ser_wav2vec2_ravdess/` | Main project — training pipeline, evaluation, Gradio app |
| `ser_wav2vec2_ravdess/src/` | Core code (`config`, `dataset`, `model`, `train`, `trainer`, `evaluate`) |
| `ser_wav2vec2_ravdess/scripts/prepare_data.py` | Builds `metadata_local.csv` from RAVDESS filenames |
| `ser_wav2vec2_ravdess/app/gradio_app.py` | Interactive mic/upload demo used during development |
| `app/app.py` | Gradio demo shown in the project video (upload → prediction + waveform) |
| `ser_wav2vec2_ravdess/reports/` | Classification report, confusion matrix |
| `RAVDESS_DATA/metadata_local.csv` | 1,440 clips with path, emotion, actor |
| `notebooks/Untitled2_colab_legacy.ipynb` | Original Colab notebook (reference only) |

### Not in GitHub (too large)

| Asset | Size | Where to get it |
|-------|------|-----------------|
| RAVDESS audio | ~350 MB | [Zenodo](https://zenodo.org/record/1188976) → extract to `RAVDESS_DATA/ravdess/` |
| Local model checkpoints | ~1.2 GB | Train locally or use the [Hugging Face model](https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized) |

---

## Model weights

| | |
|--|--|
| **Hugging Face** | https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized |
| **Local path (after training)** | `ser_wav2vec2_ravdess/outputs/wav2vec2-ser-ravdess-optimized/` |
| **Test accuracy** | **68.3%** (speaker-independent) |
| **Macro F1** | **0.676** |
| **Backbone** | `facebook/wav2vec2-base-960h` |

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

model = Wav2Vec2ForSequenceClassification.from_pretrained("sbh013/wav2vec2-ser-ravdess-optimized")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("sbh013/wav2vec2-ser-ravdess-optimized")
```

```bash
# Gradio demo with HF model
cd ser_wav2vec2_ravdess
SER_MODEL_DIR=sbh013/wav2vec2-ser-ravdess-optimized python app/gradio_app.py

# Or a local checkpoint after training
SER_MODEL_DIR=outputs/wav2vec2-ser-ravdess-optimized python app/gradio_app.py
```

---

## How to load and use the model (for anyone)

You do **not** need the training data or local checkpoints. The model downloads
automatically from Hugging Face the first time you use it (~361 MB).

### 1. Install dependencies

```bash
git clone https://github.com/sbhardwaj1304/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition/ser_wav2vec2_ravdess
pip install -r requirements.txt
```

Minimum packages if you only want inference (no training):

```bash
pip install torch transformers librosa soundfile
```

### 2. Load the model from Hugging Face

```python
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

MODEL_ID = "sbh013/wav2vec2-ser-ravdess-optimized"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Labels the model predicts
print(model.config.id2label)
# {0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful',
#  4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised'}
```

Weights are cached after first download at `~/.cache/huggingface/hub/`.

### 3. Run inference on an audio file

Audio must be **mono, 16 kHz** (the script resamples automatically if needed).

```python
import librosa
import numpy as np
import torch

def predict_emotion(audio_path: str) -> dict:
    """Return {emotion: probability} for a .wav file."""
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

    inputs = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    id2label = model.config.id2label
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}


# Example
result = predict_emotion("my_speech.wav")
top_emotion = max(result, key=result.get)
print(f"Predicted: {top_emotion} ({result[top_emotion]:.1%})")
print(result)
```

### 4. Run a Gradio web demo

```bash
# The demo shown in our project video
python app/app.py

# Or the simpler demo used during development
cd ser_wav2vec2_ravdess
SER_MODEL_DIR=sbh013/wav2vec2-ser-ravdess-optimized python app/gradio_app.py
```

Open the URL printed in the terminal (usually `http://127.0.0.1:7860`).

### 5. Use in your own Python app / API

```python
# After loading model + feature_extractor (step 2):
import numpy as np

def predict_from_array(waveform: np.ndarray, sample_rate: int = 16000) -> str:
    """Pass a numpy float32 mono array; returns top emotion label."""
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).squeeze().cpu().numpy()

    id2label = model.config.id2label
    scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    return max(scores, key=scores.get)
```

### 6. Download weights manually (optional)

```bash
pip install huggingface_hub
huggingface-cli download sbh013/wav2vec2-ser-ravdess-optimized --local-dir ./my-model
```

```python
model = Wav2Vec2ForSequenceClassification.from_pretrained("./my-model")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./my-model")
```

### Tips for best results

- Use **clean speech** clips of 1–5 seconds (model was trained on short RAVDESS utterances)
- Works best on **English speech** (wav2vec2-base-960h is English-pretrained)
- The model predicts **how the speech sounds emotionally**, not the semantic meaning of words
- 68.3% accuracy is on held-out RAVDESS speakers — real-world mic quality may differ

---

## Model configuration

| Setting | Value |
|---------|-------|
| **Backbone** | `facebook/wav2vec2-base-960h` |
| **Training style** | Single-stage fine-tune (CNN feature encoder frozen; transformer + head trained) |
| **Epochs** | 25 |
| **LR (encoder)** | 2e-5 |
| **LR (head)** | 5e-5 |
| **Batch size** | 8 |
| **Gradient accumulation** | 2 (effective batch = 16) |
| **LR scheduler** | Cosine, 6% warmup |
| **Weight decay** | 0.01 |
| **Label smoothing** | 0.08 |
| **Class weights** | Inverse frequency, extra boost on `sad`/`angry` |
| **Augmentation** | Noise, gain, small time shift — 70% of training batches |
| **Precision** | fp32 |
| **Padding** | Dynamic (per-batch) |
| **Checkpoint selection** | Best macro-F1 on validation |
| **Early stopping patience** | 10 epochs |

### Dataset split (speaker-independent, seed=22)

| Split | Clips | Actors |
|-------|-------|--------|
| Train | 1,140 | 19 |
| Val | 120 | 2 |
| Test | 180 | 3 |

Actors — not individual clips — are assigned to splits, so the model cannot memorize a speaker's voice.

### Per-class results (test set)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| angry | 0.917 | 0.458 | 0.611 |
| calm | 0.810 | 0.708 | 0.756 |
| disgust | 0.947 | 0.750 | **0.837** |
| fearful | 0.679 | 0.792 | 0.731 |
| happy | 0.500 | 0.833 | 0.625 |
| neutral | 0.588 | 0.833 | 0.690 |
| sad | 0.500 | 0.292 | 0.368 |
| surprised | 0.724 | 0.875 | **0.793** |

Full classification report:
https://github.com/sbhardwaj1304/Speech-Emotion-Recognition/blob/main/ser_wav2vec2_ravdess/reports/classification_report.txt

Confusion matrix:
https://github.com/sbhardwaj1304/Speech-Emotion-Recognition/blob/main/ser_wav2vec2_ravdess/reports/confusion_matrix.png

---

## Quick start

```bash
cd ser_wav2vec2_ravdess
pip install -r requirements.txt

# 1. Download RAVDESS audio → ../RAVDESS_DATA/ravdess/
# 2. Build metadata (or use included metadata_local.csv)
python run_training.py --prepare-only

# 3. Train on a single GPU
bash run_train.sh

# 4. Evaluate
SER_MODEL_DIR=outputs/wav2vec2-ser-ravdess-optimized python -m src.evaluate

# 5. Demo
SER_MODEL_DIR=sbh013/wav2vec2-ser-ravdess-optimized python app/gradio_app.py
```

---

## Project structure

```
Speech-Emotion-Recognition/
├── README.md                          ← this file
├── .gitignore
├── app/                                ← demo shown in the project video
│   ├── app.py
│   └── requirements.txt
├── RAVDESS_DATA/
│   └── metadata_local.csv             ← clip paths + labels (audio not included)
├── notebooks/
│   └── Untitled2_colab_legacy.ipynb   ← original Colab reference
└── ser_wav2vec2_ravdess/
    ├── README.md                      ← detailed technical docs
    ├── requirements.txt
    ├── run_training.py                ← main entry point
    ├── run_train.sh                   ← single-GPU launcher
    ├── run_multigpu.sh                ← optional multi-GPU
    ├── src/                           ← config, dataset, model, train, trainer
    ├── scripts/prepare_data.py
    ├── app/gradio_app.py
    ├── notebooks/Training.ipynb
    ├── reports/                       ← eval results + confusion matrix
    └── outputs/                       ← local checkpoints (not in git)
        └── wav2vec2-ser-ravdess-optimized/   ← best model (68.3%)
```

---

## Environment variables

```bash
export SER_DATA_ROOT=/path/to/RAVDESS_DATA
export SER_METADATA_CSV=/path/to/metadata_local.csv
export SER_OUTPUT_DIR=/path/to/outputs
export SER_MODEL_DIR=sbh013/wav2vec2-ser-ravdess-optimized   # or local path
```

---

## License / attribution

- wav2vec2: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) (Apache 2.0)
- RAVDESS: [Livingstone & Russo, 2018](https://zenodo.org/record/1188976)

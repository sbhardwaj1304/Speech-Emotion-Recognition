# Speech Emotion Recognition — wav2vec2 + RAVDESS

Fine-tunes Hugging Face **wav2vec2** for **8-class speech emotion recognition** on the
[RAVDESS](https://zenodo.org/record/1188976) dataset, using a **speaker-independent**
train/val/test split, class-weighted loss, data augmentation, and a Gradio demo.

**GitHub:** https://github.com/sbhardwaj1304/Speech-Emotion-Recognition  
**Best model (Hugging Face):** https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized

**Emotions (8 classes):** `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

---

## What this repo contains

| Path | Description |
|------|-------------|
| `ser_wav2vec2_ravdess/` | Main project — training pipeline, evaluation, Gradio app |
| `ser_wav2vec2_ravdess/src/` | Core code (`config`, `dataset`, `model`, `train`, `trainer`, `evaluate`) |
| `ser_wav2vec2_ravdess/scripts/prepare_data.py` | Builds `metadata_local.csv` from RAVDESS filenames |
| `ser_wav2vec2_ravdess/app/gradio_app.py` | Interactive mic/upload demo |
| `ser_wav2vec2_ravdess/reports/` | Classification report, confusion matrix, project report |
| `RAVDESS_DATA/metadata_local.csv` | 1,440 clips with path, emotion, actor |
| `notebooks/Untitled2_colab_legacy.ipynb` | Original Colab notebook (reference only) |

### Not in GitHub (too large)

| Asset | Size | Where to get it |
|-------|------|-----------------|
| RAVDESS audio | ~350 MB | [Zenodo](https://zenodo.org/record/1188976) → extract to `RAVDESS_DATA/ravdess/` |
| Local model checkpoints | 0.4–1.2 GB each | Train locally or use [Hugging Face model](https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized) |

---

## Model weights — where they live

### Best model (use this)

| | |
|--|--|
| **Hugging Face** | https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized |
| **Local path** | `ser_wav2vec2_ravdess/outputs/wav2vec2-ser-ravdess-optimized/` |
| **Test accuracy** | **68.3%** (speaker-independent) |
| **Macro F1** | **0.676** |
| **Backbone** | `facebook/wav2vec2-base-960h` |
| **File** | `model.safetensors` (~361 MB) |

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

model = Wav2Vec2ForSequenceClassification.from_pretrained("sbh013/wav2vec2-ser-ravdess-optimized")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("sbh013/wav2vec2-ser-ravdess-optimized")
```

```bash
# Gradio demo with HF model
cd ser_wav2vec2_ravdess
SER_MODEL_DIR=sbh013/wav2vec2-ser-ravdess-optimized python app/gradio_app.py

# Or local checkpoint
SER_MODEL_DIR=outputs/wav2vec2-ser-ravdess-optimized python app/gradio_app.py
```

---

## How to load and use the model (for anyone)

You do **not** need the training data or local checkpoints. The model downloads
automatically from Hugging Face the first time you use it (~361 MB).

**Model page:** https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized

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

### 4. Run the Gradio web demo

Record from your microphone or upload a `.wav` file:

```bash
cd ser_wav2vec2_ravdess
pip install gradio
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

Then load from the local folder:

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

| Run | Local folder | HF Hub | Test acc | Status |
|-----|--------------|--------|----------|--------|
| **Optimized (best)** | `outputs/wav2vec2-ser-ravdess-optimized/` | [sbh013/wav2vec2-ser-ravdess-optimized](https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized) | **68.3%** | Complete |
| Baseline | `outputs/wav2vec2-ser-ravdess/` | — | 53.9% | Complete |
| v3 large | `outputs/wav2vec2-ser-ravdess-v3/` | — | TBD | **Training in progress** |

---

## Best model — full configuration

The optimized run (`training_optimized.log`) that achieved **68.3%** test accuracy:

| Setting | Value |
|---------|-------|
| **Backbone** | `facebook/wav2vec2-base-960h` |
| **Training style** | Single-stage full fine-tune |
| **Epochs** | **25** |
| **LR (encoder)** | `2e-5` |
| **LR (head)** | `5e-5` |
| **Batch size** | 8 per GPU |
| **Gradient accumulation** | 2 (effective batch = **16**) |
| **LR scheduler** | Cosine |
| **Warmup ratio** | 0.06 |
| **Weight decay** | 0.01 |
| **Label smoothing** | 0.08 |
| **Class weights** | Yes — `neutral: 1.778`, all others: `0.889` |
| **Augmentation** | Yes — noise std 0.008, gain 0.85–1.15, 70% prob, ±800 sample shift |
| **Precision** | fp32 (fp16 caused NaN loss) |
| **Padding** | Dynamic (per-batch) |
| **CNN feature encoder** | Frozen |
| **Checkpoint selection** | Best macro-F1 on validation |
| **Early stopping patience** | 10 epochs |

### Dataset split (speaker-independent, seed=22)

| Split | Clips | Actors |
|-------|-------|--------|
| Train | 1,140 | 19 |
| Val | 120 | 2 |
| Test | 180 | 3 |

Actors — not individual clips — are assigned to splits, so the model cannot memorize a speaker's voice.

### Per-class results (optimized model, test set)

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

Full report: `ser_wav2vec2_ravdess/reports/classification_report.txt`  
Confusion matrix: `ser_wav2vec2_ravdess/reports/confusion_matrix.png`

---

## Training history — what we did

| # | Run | What changed | Test accuracy |
|---|-----|--------------|---------------|
| 1 | Broken baseline | Fixed-length padding, fp16 → NaN gradients | ~13% (random) |
| 2 | Fixed baseline | Dynamic padding, fp32, frozen CNN | **53.9%** (10 epochs, base) |
| 3 | **Optimized** | +25 epochs, class weights, augmentation | **68.3%** (base) |
| 4 | v3 (running) | wav2vec2-**large**, 3-stage schedule, seed 22, 1 GPU | In progress |

### Bugs fixed along the way

1. **NaN loss / 0 learning** — switched from fixed 5 s padding to dynamic padding; disabled fp16
2. **Wrong metadata** — `metadata.csv` had Colab Windows paths; replaced with `metadata_local.csv`
3. **Inflated train loss (~4 instead of ~2)** — fixed gradient-accumulation logging in custom `SERTrainer`
4. **Multi-GPU DDP crash** — set `ddp_find_unused_parameters=True` for stage transitions

### Current training run (v3)

```
Model:     facebook/wav2vec2-large-960h
Schedule:  Stage 1 (10 ep head) → Stage 2 (40 ep encoder) → Stage 3 (15 ep polish)
GPU:       1× GPU 0
Batch:     4 × 16 grad accum = effective 64
Seed:      22
Log:       ser_wav2vec2_ravdess/training.log
Output:    ser_wav2vec2_ravdess/outputs/wav2vec2-ser-ravdess-v3/
```

Monitor: `tail -f ser_wav2vec2_ravdess/training.log`

---

## Quick start

```bash
cd ser_wav2vec2_ravdess
pip install -r requirements.txt

# 1. Download RAVDESS audio → ../RAVDESS_DATA/ravdess/
# 2. Build metadata (or use included metadata_local.csv)
python run_training.py --prepare-only

# 3. Train on single GPU
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
        ├── wav2vec2-ser-ravdess-optimized/   ← best (68.3%)
        ├── wav2vec2-ser-ravdess/            ← baseline (53.9%)
        └── wav2vec2-ser-ravdess-v3/         ← v3 in progress
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

- wav2vec2: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) / [large](https://huggingface.co/facebook/wav2vec2-large-960h) (Apache 2.0)
- RAVDESS: [Livingstone & Russo, 2018](https://zenodo.org/record/1188976)

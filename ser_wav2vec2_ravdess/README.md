# Speech Emotion Recognition — wav2vec2 + RAVDESS

Fine-tunes Hugging Face wav2vec2 for **8-class speech emotion recognition** on the
[RAVDESS](https://zenodo.org/record/1188976) dataset. Uses a speaker-independent
train/val/test split, three-stage fine-tuning, class-weighted loss, and a Gradio
demo.

**Emotions:** `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`

---

## Quick start (this server)

```bash
cd /sensei-fs-3/users/vkhazanchi/reranker-diksha/personal/ser/ser_wav2vec2_ravdess

# 1) Train on GPU 0 (recommended)
bash run_train.sh

# 2) Or run Python directly
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python run_training.py --skip-prepare

# 3) Demo with the best finished model
/opt/conda/bin/python app/gradio_app.py

# 4) Monitor training
tail -f training.log
```

Python interpreter on this machine: `/opt/conda/bin/python`

---

## Project layout

```
personal/ser/
├── RAVDESS_DATA/                    # dataset (audio + metadata)
│   ├── ravdess/                     # 1,440 .wav files (24 actors)
│   ├── metadata_local.csv           # ✅ use this — correct local paths + labels
│   └── metadata.csv                 # ❌ old Colab export — wrong paths/labels
│
├── Untitled2.ipynb                  # legacy Colab notebook (reference only)
│
└── ser_wav2vec2_ravdess/            # main project
    ├── README.md
    ├── requirements.txt
    ├── run_training.py              # local entry point
    ├── run_train.sh                 # single-GPU launcher (default)
    ├── run_multigpu.sh              # optional multi-GPU DDP
    ├── scripts/prepare_data.py      # rebuild metadata_local.csv
    ├── src/
    │   ├── config.py                # paths, hyperparameters, label maps
    │   ├── dataset.py               # metadata, speaker split, feature extraction
    │   ├── model.py                 # wav2vec2 classifier builder
    │   ├── train.py                 # 3-stage HF Trainer pipeline
    │   ├── trainer.py               # class-weighted loss + discriminative LRs
    │   ├── evaluate.py              # test metrics + confusion matrix
    │   └── utils.py
    ├── app/gradio_app.py            # mic/upload demo
    ├── notebooks/Training.ipynb     # documented pipeline notebook
    └── outputs/                     # models + evaluation artifacts (runtime)
```

---

## Where everything is stored

### Dataset

| What | Path |
|------|------|
| Audio files | `/sensei-fs-3/users/vkhazanchi/reranker-diksha/personal/ser/RAVDESS_DATA/ravdess/` |
| Metadata (use this) | `.../RAVDESS_DATA/metadata_local.csv` |
| Old Colab metadata | `.../RAVDESS_DATA/metadata.csv` (do not use) |

`metadata_local.csv` columns: `path`, `emotion`, `actor` — built automatically by
`scripts/prepare_data.py` from RAVDESS filenames.

### Trained models

All checkpoints live under:

```
ser_wav2vec2_ravdess/outputs/
```

| Folder | Status | Test accuracy | Notes |
|--------|--------|---------------|-------|
| **`wav2vec2-ser-ravdess-optimized/`** | **Best finished** | **68.3%** | wav2vec2-base, 25 epochs, class weights + augmentation |
| `wav2vec2-ser-ravdess/` | Complete | 53.9% | First fixed 10-epoch baseline |
| `wav2vec2-ser-ravdess-v3/` | Partial | — | wav2vec2-large, 3-stage; stopped mid-run |

Each finished model folder contains:

```
model.safetensors          # fine-tuned weights
config.json                # model config (8 labels)
preprocessor_config.json   # feature extractor settings
training_args.bin          # HF TrainingArguments snapshot
checkpoint-*/              # intermediate epoch checkpoints
```

### Evaluation artifacts

Written to `outputs/` (overwritten on each evaluate run):

| File | Contents |
|------|----------|
| `classification_report.txt` | per-class precision/recall/F1 |
| `confusion_matrix.png` | heatmap plot |

### Training logs

| Log file | Run |
|----------|-----|
| `training.log` | current single-GPU run (`run_train.sh`) |
| `training_optimized.log` | 68.3% optimized run |
| `training_v3_multigpu.log` | partial large-model multi-GPU run |
| `training_fixed.log` / `training_full.log` | early debugging runs |

### Hugging Face cache (pretrained weights only)

Downloaded base models (`facebook/wav2vec2-base-960h`, `facebook/wav2vec2-large-960h`)
are cached under `~/.cache/huggingface/hub/`. Fine-tuned SER weights are **only** in
`outputs/`.

---

## How to access / load models

### Gradio demo

```bash
# Default: loads outputs/wav2vec2-ser-ravdess-optimized
/opt/conda/bin/python app/gradio_app.py

# Override model path
SER_MODEL_DIR=outputs/wav2vec2-ser-ravdess-optimized /opt/conda/bin/python app/gradio_app.py
```

Opens a local web UI for mic recording or file upload.

### Python inference

```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

model_dir = "outputs/wav2vec2-ser-ravdess-optimized"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
```

### Evaluate a saved checkpoint

```bash
SER_MODEL_DIR=outputs/wav2vec2-ser-ravdess-optimized /opt/conda/bin/python -m src.evaluate
```

---

## Training

### Recommended: single GPU

```bash
bash run_train.sh
# equivalent:
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python run_training.py --skip-prepare
```

### Prepare metadata only

```bash
/opt/conda/bin/python run_training.py --prepare-only
```

### Quick smoke test (3 epochs, single stage)

```bash
/opt/conda/bin/python run_training.py --skip-prepare --epochs 3 --single-stage
```

### Optional: multi-GPU (not default on this machine)

```bash
bash run_multigpu.sh
```

### What training does

1. Load `metadata_local.csv` and split **by actor** (speaker-independent).
2. Decode audio to 16 kHz mono.
3. Extract wav2vec2 features with **dynamic padding** (per-batch).
4. Three-stage fine-tune:
   - **Stage 1** — classifier head warmup (encoder frozen), 10 epochs
   - **Stage 2** — full transformer fine-tune (CNN still frozen), 40 epochs
   - **Stage 3** — train+val polish at low LR, 15 epochs
5. Evaluate on held-out test actors; save confusion matrix + report.
6. Save final model to `outputs/wav2vec2-ser-ravdess-v3/` (current config).

---

## Configuration (`src/config.py`)

| Setting | Current value | Meaning |
|---------|---------------|---------|
| `SEED` | `22` | reproducibility |
| `MODEL_CHECKPOINT` | `facebook/wav2vec2-large-960h` | pretrained backbone |
| `MODEL_DIR` | `outputs/wav2vec2-ser-ravdess-v3` | where new runs save |
| `BATCH_SIZE` | `4` | per-GPU batch |
| `GRAD_ACCUM_STEPS` | `16` | effective batch = 4×16 = **64** on 1 GPU |
| `USE_CLASS_WEIGHTS` | `True` | upweight rare `neutral` class |
| `USE_TRAIN_AUGMENTATION` | `True` | noise/gain/shift on training batches |

Override paths with environment variables:

```bash
export SER_DATA_ROOT=/path/to/RAVDESS_DATA
export SER_METADATA_CSV=/path/to/metadata_local.csv
export SER_OUTPUT_DIR=/path/to/outputs
export SER_MODEL_DIR=/path/to/saved/model   # for Gradio / evaluate
```

---

## Understanding training loss

For 8-class cross-entropy, a random untrained model should show:

| Metric | Expected at start |
|--------|-------------------|
| `loss` (training) | **~2.08** (= ln 8) |
| `eval_loss` | **~2.08** |

If you see `loss ≈ 4` at the start, that was a **logging artifact** from gradient
accumulation with a custom loss function (fixed in `src/trainer.py`). The model was
still learning — check `eval_accuracy` and `eval_f1_macro` instead.

**Do not use** `training_full.log` — that was an early broken run with NaN gradients.

---

## Key design choices

- **Speaker-independent split** — actors (not clips) go to train/val/test.
- **Frozen CNN feature encoder** — standard for small audio datasets.
- **Dynamic padding** — variable-length batches (not fixed 5 s padding).
- **fp32 training** — fp16 caused NaN loss with wav2vec2 on this setup.
- **Macro-F1 for checkpoint selection** — treats all emotions equally.
- **Class weights** — `neutral` has half the samples of other emotions in RAVDESS.

---

## Results so far

| Run | Model | Test accuracy |
|-----|-------|---------------|
| Baseline (fixed) | wav2vec2-base, 10 ep | 53.9% |
| **Optimized** | wav2vec2-base, 25 ep | **68.3%** |
| v3 (partial) | wav2vec2-large, 3-stage | incomplete |

Best per-class F1 from optimized run: `disgust` 0.84, `surprised` 0.79, `calm` 0.76.
Weakest: `sad` 0.37 (often confused with `calm` / `neutral`).

---

## Setup (fresh environment)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested with `transformers>=4.41`. On this server, use `/opt/conda/bin/python` directly.

---

## License / attribution

- Model: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) / [large](https://huggingface.co/facebook/wav2vec2-large-960h) (Apache 2.0)
- Dataset: [RAVDESS](https://zenodo.org/record/1188976) — cite original authors if publishing results

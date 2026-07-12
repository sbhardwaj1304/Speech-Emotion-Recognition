# Speech Emotion Recognition

Fine-tunes Hugging Face **wav2vec2** for 8-class speech emotion recognition on the
[RAVDESS](https://zenodo.org/record/1188976) dataset.

**Emotions:** neutral, calm, happy, sad, angry, fearful, disgust, surprised

## Repository contents

| Path | What it is |
|------|------------|
| `ser_wav2vec2_ravdess/` | Main project — training, evaluation, Gradio demo |
| `RAVDESS_DATA/metadata_local.csv` | Speaker/emotion metadata (paths relative to data dir) |
| `notebooks/Untitled2.ipynb` | Legacy Colab notebook (reference only) |

### Not included (too large for GitHub)

| Item | Size | How to get it |
|------|------|---------------|
| RAVDESS audio | ~350 MB | Download from [Zenodo](https://zenodo.org/record/1188976), extract to `RAVDESS_DATA/ravdess/` |
| Trained model weights | 0.4–1.2 GB each | Train locally (`bash ser_wav2vec2_ravdess/run_train.sh`) or publish to [Hugging Face Hub](https://huggingface.co/new) |

## Quick start

```bash
cd ser_wav2vec2_ravdess
pip install -r requirements.txt

# 1. Place RAVDESS wav files under ../RAVDESS_DATA/ravdess/
# 2. Build metadata (or use the included metadata_local.csv)
python run_training.py --prepare-only

# 3. Train (single GPU)
bash run_train.sh

# 4. Demo
SER_MODEL_DIR=outputs/wav2vec2-ser-ravdess-optimized python app/gradio_app.py
```

See [`ser_wav2vec2_ravdess/README.md`](ser_wav2vec2_ravdess/README.md) for full documentation:
paths, model checkpoints, config, and results.

## Results (speaker-independent split)

| Run | Model | Test accuracy |
|-----|-------|---------------|
| Baseline | wav2vec2-base, 10 ep | 53.9% |
| **Optimized** | wav2vec2-base, 25 ep | **68.3%** |
| v3 (partial) | wav2vec2-large, 3-stage | incomplete |

Evaluation report: `ser_wav2vec2_ravdess/reports/classification_report.txt`

## License / attribution

- wav2vec2: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) (Apache 2.0)
- RAVDESS: [Livingstone & Russo, 2018](https://zenodo.org/record/1188976)

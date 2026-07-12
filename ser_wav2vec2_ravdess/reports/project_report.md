# Project Report: Speech Emotion Recognition with wav2vec2 on RAVDESS

## 1. Introduction

Speech Emotion Recognition (SER) is the task of automatically identifying
the emotional state expressed in a spoken utterance. It has applications in
call-center analytics, mental health monitoring, human-computer interaction,
and voice assistants. This project fine-tunes a pretrained self-supervised
speech model, **wav2vec2** (`facebook/wav2vec2-base-960h`), for 8-class
emotion classification on the **RAVDESS** dataset, using the Hugging Face
`transformers` ecosystem end to end.

## 2. Dataset

**RAVDESS** (Ryson Audio-Visual Database of Emotional Speech and Song)
contains recordings from 24 professional actors (12 female, 12 male)
vocalizing two lexically matched statements in a neutral North American
accent. Each actor produces speech across 8 emotions, at two intensity
levels (except neutral), with two repetitions per statement.

| Property            | Value                                                             |
|----------------------|--------------------------------------------------------------------|
| Total speech clips   | 1,440                                                              |
| Speakers             | 24 (balanced by gender)                                            |
| Emotions             | neutral, calm, happy, sad, angry, fearful, disgust, surprised      |
| Sampling rate (raw)  | 48 kHz (resampled to 16 kHz to match wav2vec2's pretraining rate)  |

This project assumes the dataset audio files are already downloaded and
organized on disk, and that a `metadata.csv` index (columns: `path`,
`emotion`, optionally `actor`) is already available, as produced by a
prior data-preparation step.

### Train / validation / test split

The split is performed **per actor** (80% / 10% / 10% of actors), not per
clip. This speaker-independent protocol is stricter than a random
clip-level split: it ensures the model is evaluated on voices it has never
heard during training, which is a more realistic measure of how well it
will generalize to new speakers.

## 3. Model architecture

- **Backbone:** `facebook/wav2vec2-base-960h`, a transformer encoder
  pretrained with self-supervised contrastive learning on 960 hours of
  LibriSpeech audio, then fine-tuned for ASR. We reuse its pretrained
  representations (not the ASR head) as a feature backbone.
- **Head:** `Wav2Vec2ForSequenceClassification` from `transformers`, which
  mean-pools the transformer's hidden states over time and applies a
  linear classification layer projecting to the 8 emotion classes.
- **Frozen feature encoder:** the CNN feature-extraction stack at the
  bottom of wav2vec2 is frozen (`model.freeze_feature_encoder()`). Only the
  transformer layers and the new classification head are fine-tuned. This
  is standard practice for small downstream datasets such as RAVDESS
  (1,440 clips) — it reduces the number of trainable parameters, lowers the
  risk of overfitting, and speeds up training.

## 4. Preprocessing

1. Audio is loaded and resampled to 16 kHz (wav2vec2's expected input rate)
   via the Hugging Face `datasets.Audio` feature.
2. Each clip is processed with `Wav2Vec2FeatureExtractor`, which
   zero-mean/unit-variance normalizes the raw waveform.
3. Clips are padded or truncated to a fixed length of 5 seconds
   (`MAX_DURATION_SECONDS` in `src/config.py`) so batches can be stacked
   into uniform tensors for training.

## 5. Training configuration

Implemented with the Hugging Face `Trainer` API (`src/train.py`).

| Hyperparameter              | Value                     |
|-------------------------------|---------------------------|
| Base model                    | facebook/wav2vec2-base-960h |
| Batch size (per device)       | 4                          |
| Gradient accumulation steps   | 4 (effective batch size 16) |
| Epochs                        | 10                         |
| Learning rate                 | 3e-5                       |
| Warmup ratio                  | 0.1                        |
| Weight decay                  | 0.01                       |
| Optimizer                     | AdamW (Trainer default)    |
| Mixed precision               | fp16 (if CUDA available)   |
| Model selection metric        | macro-F1 on validation set |

Macro-F1 (rather than accuracy) was used to select the best checkpoint,
since it weights all 8 emotion classes equally regardless of how frequently
they occur, which better reflects performance on the least-common classes.

## 6. Evaluation methodology

The best checkpoint (by validation macro-F1) is evaluated once on the
held-out, speaker-independent test set (`src/evaluate.py`). We report:

- Per-class precision, recall, and F1 (`classification_report.txt`)
- Overall accuracy
- A confusion matrix heatmap (`confusion_matrix.png`) showing which
  emotions are most frequently confused with one another (e.g. `calm` vs.
  `neutral`, or `fearful` vs. `sad`, are common confusions reported in SER
  literature).

## 7. Results

> This section is a template. Fill in the actual numbers after running
> `python -m src.train` followed by `python -m src.evaluate` (or the
> notebook) on your environment, since results depend on hardware,
> exact package versions, and the random seed.

| Metric               | Validation | Test |
|-----------------------|------------|------|
| Accuracy               | TBD        | TBD  |
| Macro F1               | TBD        | TBD  |
| Macro Precision         | TBD        | TBD  |
| Macro Recall             | TBD        | TBD  |

**Confusion matrix:** see `outputs/confusion_matrix.png` after running
evaluation.

**Per-class performance:** see `outputs/classification_report.txt`.

## 8. Discussion

- Because RAVDESS is small and speaker-independent evaluation is stricter,
  expect accuracy to be noticeably lower than clip-level-split results
  reported in some papers — this is a fairer estimate of real-world
  performance on unseen speakers.
- Common failure modes to watch for in the confusion matrix: `calm` vs.
  `neutral` (both are low-arousal, low-valence-shift emotions and are
  acoustically similar), and `fearful` vs. `surprised` (both can carry
  fast, high-pitch speech patterns).
- Because the CNN feature encoder is frozen, most of the model's audio
  representation is fixed to what was learned from LibriSpeech (read
  speech, not acted emotional speech). Unfreezing it (set
  `freeze_feature_encoder=False` in `src/model.py`) could improve results
  on a larger emotional-speech corpus, at higher risk of overfitting on
  RAVDESS alone.

## 9. Possible improvements

- Data augmentation (pitch shift, time stretch, additive noise, SpecAugment
  on the extracted features) to improve robustness and reduce overfitting.
- Combine RAVDESS with other emotional-speech corpora (e.g. CREMA-D, TESS,
  SAVEE) for a larger, more diverse training set.
- Try larger backbones (`wav2vec2-large`, `wavlm-base-plus`,
  `hubert-large`) or unfreezing the feature encoder with a lower learning
  rate for the backbone vs. the classification head.
- Add a speaker-adaptation or gender-balanced sampling strategy if class
  imbalance across speakers becomes an issue.

## 10. References

- Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual
  Database of Emotional Speech and Song (RAVDESS)*. PLoS ONE, 13(5).
- Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A
  Framework for Self-Supervised Learning of Speech Representations*.
  NeurIPS.
- Hugging Face `transformers` documentation:
  https://huggingface.co/docs/transformers

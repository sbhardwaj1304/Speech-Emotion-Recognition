"""
Gradio demo app for the fine-tuned Speech Emotion Recognition model.

Usage:
    python app/gradio_app.py

By default it loads the best finished model from
../outputs/wav2vec2-ser-ravdess-optimized. Override with SER_MODEL_DIR.
"""

import os

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

MODEL_DIR = os.getenv(
    "SER_MODEL_DIR",
    os.path.join(
        os.path.dirname(__file__), "..", "outputs", "wav2vec2-ser-ravdess-optimized"
    ),
)
SAMPLING_RATE = 16000

print(f"Loading model from: {MODEL_DIR}")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

id2label = model.config.id2label


def predict_emotion(audio):
    """
    `audio` is (sample_rate, numpy_array) as provided by gr.Audio(type="numpy").
    Returns a dict of {label: probability} for gr.Label.
    """
    if audio is None:
        return {}

    sr, waveform = audio
    waveform = np.asarray(waveform).astype(np.float32)

    # Convert stereo -> mono if needed
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Normalize int16-range audio (some mic backends return int16)
    if np.max(np.abs(waveform)) > 1.0:
        waveform = waveform / np.iinfo(np.int16).max

    if sr != SAMPLING_RATE:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLING_RATE)

    inputs = feature_extractor(
        waveform,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    return {id2label[i]: float(probs[i]) for i in range(len(probs))}


demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak or upload audio"),
    outputs=gr.Label(num_top_classes=8, label="Predicted Emotion"),
    title="🎙️ Speech Emotion Recognition — wav2vec2 fine-tuned on RAVDESS",
    description=(
        "Record yourself speaking (or upload a short .wav clip) and the model "
        "will predict the emotion conveyed: neutral, calm, happy, sad, angry, "
        "fearful, disgust, or surprised. Model: facebook/wav2vec2-base-960h "
        "fine-tuned on the RAVDESS dataset."
    ),
    examples=None,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()

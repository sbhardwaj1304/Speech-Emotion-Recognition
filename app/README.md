# Demo app (Gradio)

This is the web demo shown in our project video — upload or record an audio
clip and it predicts the speaker's emotion with a probability breakdown and
waveform plot. Built by Utsav, originally in his own repo, now merged in here
so the whole project lives in one place.

```bash
pip install -r app/requirements.txt
python app/app.py
```

Opens at `http://127.0.0.1:7860`. By default it loads Utsav's demo model
(`utsav05/wav2vec2-ser-ravdess`). To point it at our best evaluated model
instead, set the `MODEL_ID` environment variable:

```bash
MODEL_ID=sbh013/wav2vec2-ser-ravdess-optimized python app/app.py
```

There's also a simpler Gradio app at
[`ser_wav2vec2_ravdess/app/gradio_app.py`](../ser_wav2vec2_ravdess/app/gradio_app.py)
that we used during training/evaluation — this one (`app/app.py`) is the one
in the demo video, with the waveform plot and probability bars.

See [`../README.md`](../README.md) for the rest of the project.

# Final Project Submission Checklist

**Due:** tomorrow 23:59 (per Classroom assignment)

## Deliverables

| # | Item | Format | Status | Action |
|---|------|--------|--------|--------|
| 1 | Project report | `.pdf` | **TODO** | Write `reports/FINAL_PROJECT_REPORT.md` → export PDF |
| 2 | Video demo | `.mp4` | **Done** | Utsav recorded — upload to Classroom |
| 3 | Codebase + weights | GitHub + HF | **Done** | Submit repo + model links (see below) |

---

## Links to paste in Classroom

```
GitHub repo:            https://github.com/sbhardwaj1304/Speech-Emotion-Recognition
Best model (68.3%):     https://huggingface.co/sbh013/wav2vec2-ser-ravdess-optimized
Demo model (video):     https://huggingface.co/utsav05/wav2vec2-ser-ravdess
Video:                  [your .mp4 upload]
Report PDF:             [export from reports/FINAL_PROJECT_REPORT.md]
```

See the "Team" section in `README.md` for who did what.

---

## Quick test before submitting

**Demo (same as video):**
```bash
git clone https://github.com/sbhardwaj1304/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
pip install -r app/requirements.txt
python app/app.py
```

**Training code:**
```bash
cd Speech-Emotion-Recognition/ser_wav2vec2_ravdess
pip install -r requirements.txt
# needs RAVDESS audio at ../RAVDESS_DATA/ravdess/
```

---

## PDF report

Source: `reports/FINAL_PROJECT_REPORT.md`
Export: open in Google Docs / VS Code → Print to PDF, or `pandoc reports/FINAL_PROJECT_REPORT.md -o reports/FINAL_PROJECT_REPORT.pdf`

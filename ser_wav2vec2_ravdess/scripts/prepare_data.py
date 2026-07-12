"""
Build a clean metadata CSV from the local RAVDESS audio tree.

The bundled metadata.csv from Colab uses Windows paths and capitalized labels
that do not match this project's config. This script scans the wav files and
writes metadata_local.csv with absolute paths and canonical lowercase labels.

Usage:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


def emotion_from_filename(filename: str) -> str:
    """RAVDESS: 03-01-05-01-02-01-12.wav -> emotion code 05 -> angry."""
    code = Path(filename).stem.split("-")[2]
    if code not in config.EMOTION_MAP:
        raise ValueError(f"Unknown emotion code '{code}' in file {filename}")
    return config.EMOTION_MAP[code]


def actor_from_filename(filename: str) -> str:
    return Path(filename).stem.split("-")[-1]


def prepare(
    data_dir: str | Path = config.DATA_DIR,
    output_csv: str | Path = config.METADATA_CSV,
) -> Path:
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"RAVDESS audio directory not found: {data_dir}\n"
            "Unzip RAVDESS_DATA next to this project or set SER_DATA_DIR."
        )

    rows: list[dict[str, str]] = []
    for actor_dir in sorted(data_dir.glob("Actor_*")):
        for wav_path in sorted(actor_dir.glob("*.wav")):
            rows.append(
                {
                    "path": str(wav_path.resolve()),
                    "emotion": emotion_from_filename(wav_path.name),
                    "actor": actor_from_filename(wav_path.name),
                }
            )

    if not rows:
        raise RuntimeError(f"No .wav files found under {data_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "emotion", "actor"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    return output_csv


if __name__ == "__main__":
    prepare()

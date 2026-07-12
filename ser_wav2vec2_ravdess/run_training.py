#!/usr/bin/env python3
"""
Local entry point for the full SER training pipeline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_training.py --skip-prepare
    bash run_train.sh
    python run_training.py --epochs 3 --single-stage   # quick smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train wav2vec2 SER model on RAVDESS")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate metadata_local.csv and exit",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip metadata generation (use existing metadata_local.csv)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for a quick single-stage smoke test",
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Skip two-stage schedule and run one training pass",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_prepare:
        from scripts.prepare_data import prepare

        prepare()

    if args.prepare_only:
        return

    two_stage = not args.single_stage
    if args.epochs is not None:
        from src import config

        config.NUM_EPOCHS = args.epochs
        two_stage = False
        print(f"Quick mode: single-stage, NUM_EPOCHS={config.NUM_EPOCHS}")

    from src.evaluate import run_evaluation
    from src.train import main as train_main

    trainer, test_ds = train_main(two_stage=two_stage)
    run_evaluation(trainer, test_ds)


if __name__ == "__main__":
    main()

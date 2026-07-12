#!/usr/bin/env bash
# Single-GPU training (default on this server).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
LOG="${LOG:-training.log}"

echo "Starting 1-GPU training on GPU ${GPU} (seed=${SEED:-22})..."
CUDA_VISIBLE_DEVICES="${GPU}" \
  "${PYTHON}" run_training.py --skip-prepare "$@" \
  2>&1 | tee "${LOG}"

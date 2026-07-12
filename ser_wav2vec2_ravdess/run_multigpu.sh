#!/usr/bin/env bash
# Optional multi-GPU launcher. Prefer run_train.sh (single GPU) on this machine.
set -euo pipefail
cd "$(dirname "$0")"

NGPU="${NGPU:-8}"
PYTHON="${PYTHON:-/opt/conda/bin/python}"

echo "Launching ${NGPU}-GPU training via DDP..."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  "${PYTHON}" -m torch.distributed.run --nproc_per_node="${NGPU}" run_training.py --skip-prepare "$@" \
  2>&1 | tee training_multigpu.log

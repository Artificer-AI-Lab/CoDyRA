#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="${DATA_DIR:-YOUR_DATA_DIR}"
MODEL_CKPT_PATH="${MODEL_CKPT_PATH:-YOUR_CKPT_PATH}"

METHOD="codyra"
RANK=16
TARGET_ENCODER="all"
TARGET_MODULES="qkvoinout"

LR=5e-4
ITERATIONS=500
MAX_KAPPA=0.005
DENSE_ITERS=0.5

python main.py \
  --data_dir "${DATA_DIR}" \
  --lr "${LR}" \
  --iterations "${ITERATIONS}" \
  --save "${MODEL_CKPT_PATH}" \
  --rank "${RANK}" \
  --target_encoder "${TARGET_ENCODER}" \
  --target_modules_abbrev "${TARGET_MODULES}" \
  --method "${METHOD}" \
  --max_kappa "${MAX_KAPPA}" \
  --dense_iters "${DENSE_ITERS}"

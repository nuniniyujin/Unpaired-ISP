#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_PY="$REPO_ROOT/training/train.py"

python "$TRAIN_PY" \
  --train_raw_dir "${TRAIN_RAW_DIR:-$REPO_ROOT/data/train/raws_png_dmnet_ffdnet}" \
  --train_rgb_dir "${TRAIN_RGB_DIR:-$REPO_ROOT/data/train/jpegs}" \
  --pairing_npz "${PAIRING_NPZ:-$REPO_ROOT/data_processing/patch_ot_all_weights_sparse.npz}" \
  --pairing_weight_key "${PAIRING_WEIGHT_KEY:-final_weight}" \
  --pairing_topk "${PAIRING_TOPK:-8}" \
  --pairing_sampling "${PAIRING_SAMPLING:-weighted}" \
  --target_domain "${TARGET_DOMAIN:-nonlinear}" \
  --batch_size "${BATCH_SIZE:-24}" \
  --lr "${LR:-2e-4}" \
  --checkpoint_dir "${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints/cnn_submission}" \
  --tensorboard \
  --tb_num_images "${TB_NUM_IMAGES:-10}" \
  "$@"

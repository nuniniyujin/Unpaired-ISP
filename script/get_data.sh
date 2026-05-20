#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
mkdir -p "$DATA_DIR"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Install it first: pip install gdown"
  exit 1
fi

gdown -O "$DATA_DIR/UnpairedISP_dev.zip" 1t1zC6f8u4-f0Ft9AF2vfpMO8zisMcOyl
unzip "$DATA_DIR/UnpairedISP_dev.zip" -d "$DATA_DIR/dev/"
gdown -O "$DATA_DIR/UnpairedISP_train.zip" 1IXH8pK9NGluBEnTzPC7ohkO4l0TbqbiP
unzip "$DATA_DIR/UnpairedISP_train.zip" -d "$DATA_DIR/train/"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${1:-$REPO_ROOT/checkpoints}"
PORT="${2:-6006}"

tensorboard --logdir="$LOGDIR" --port "$PORT" --samples_per_plugin=images=1000

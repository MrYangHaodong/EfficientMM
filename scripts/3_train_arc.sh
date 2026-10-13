#!/usr/bin/env bash
# Fit the ARC policies on the validation caches. Default: the seed and lambda of each fold
# in configs/arc.json ("deployed"); pass --grid to fit every seed x lambda.
#   bash scripts/3_train_arc.sh [gpu] [--grid]
# -> checkpoints/arc/arc_fold<N>_seed<S>_lam<L>.pt
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"; shift || true
mkdir -p logs
CUDA_VISIBLE_DEVICES=$GPU python arc/train_arc.py "$@" 2>&1 | tee logs/arc.log

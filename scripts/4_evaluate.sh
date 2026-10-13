#!/usr/bin/env bash
# Test-set evaluation of SeMARC at full availability and with 10-40% of the modalities
# missing (availability p = 1.0 ... 0.6), from the prefix caches.
#   bash scripts/4_evaluate.sh [gpu]
# -> results/semarc_iemocap.csv
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
mkdir -p logs
CUDA_VISIBLE_DEVICES=$GPU python inference/evaluate.py --avail 1.0 0.9 0.8 0.7 0.6 2>&1 | tee logs/eval.log

#!/usr/bin/env bash
# Train the SeMA backbone on every IEMOCAP fold (recipe: configs/sema.json).
#   bash scripts/1_train_sema.sh [gpu]
# -> checkpoints/sema/{best_model_fold<N>.pth, results_fold<N>.json}
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
FOLDS=$(python -c "import json; print(*json.load(open('configs/sema.json'))['folds'])")
mkdir -p logs
for F in $FOLDS; do
  echo "== SeMA fold $F (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU python training/train_sema.py --fold "$F" 2>&1 | tee "logs/sema_fold${F}.log"
done

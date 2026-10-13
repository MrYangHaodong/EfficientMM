#!/usr/bin/env bash
# Whole pipeline: SeMA -> prefix caches -> ARC -> test-set evaluation.
#   bash scripts/run_all.sh [gpu]
set -euo pipefail
cd "$(dirname "$0")"
GPU="${1:-0}"
bash 1_train_sema.sh "$GPU"
bash 2_build_caches.sh "$GPU"
bash 3_train_arc.sh "$GPU"
bash 4_evaluate.sh "$GPU"

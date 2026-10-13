#!/usr/bin/env bash
# Prefix-tree caches (val + test) of the trained backbones; a few minutes per fold.
#   bash scripts/2_build_caches.sh [gpu]
# -> cache/iemocap_fold<N>_{val,test}.npz
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="${1:-0}"
mkdir -p logs
CUDA_VISIBLE_DEVICES=$GPU python arc/build_cache.py 2>&1 | tee logs/cache.log

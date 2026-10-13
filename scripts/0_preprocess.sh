#!/usr/bin/env bash
# Raw IEMOCAP release -> features and split files (written inside the release tree).
#   bash scripts/0_preprocess.sh /path/to/IEMOCAP_full_release /path/to/dinov3 /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth [gpu]
# Afterwards: export IEMOCAP_ROOT=/path/to/IEMOCAP_full_release
set -euo pipefail
cd "$(dirname "$0")/.."
RAW="${1:?usage: bash scripts/0_preprocess.sh <IEMOCAP_full_release> <dinov3 repo> <dinov3 vitl16 ckpt> [gpu]}"
REPO="${2:?dinov3 repo}"; CKPT="${3:?dinov3 checkpoint}"; GPU="${4:-0}"
export CUDA_VISIBLE_DEVICES=$GPU
python preprocess/segment_video.py --raw_root "$RAW"
python preprocess/extract_video.py --raw_root "$RAW" --dinov3_repo "$REPO" --dinov3_ckpt "$CKPT"
python preprocess/extract_audio.py --raw_root "$RAW"
python preprocess/extract_text.py  --raw_root "$RAW"
python data/IEMOCAP/make_splits.py --root "$RAW"

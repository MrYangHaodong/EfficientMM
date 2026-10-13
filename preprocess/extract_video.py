#!/usr/bin/env python
"""Video features: DINOv3 ViT-L/16 CLS token of 100 frames per utterance clip.

    python preprocess/extract_video.py --raw_root /path/to/IEMOCAP_full_release \
        --dinov3_repo /path/to/dinov3 --dinov3_ckpt /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

Reads the clips of preprocess/segment_video.py (sentences/avi/<dialog>/<utt>.avi). 100
frame indices are spread uniformly over the clip (shorter clips are tiled), each frame is
resized to 224x224 (bilinear), ImageNet-normalised and encoded one at a time -> features
(100, 1024), written with the utterance labels to
Session<k>/sentences/video_features/<dialog>/<utt>.pkl.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iemocap_labels import session_labels   # noqa: E402

TARGET_FRAMES, SIZE = 100, 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def sample_indices(total):
    if total <= 0:
        return [0] * TARGET_FRAMES
    if total >= TARGET_FRAMES:
        return list(np.linspace(0, total - 1, TARGET_FRAMES, dtype=int))
    return (list(range(total)) * (TARGET_FRAMES // total + 1))[:TARGET_FRAMES]


def read_frames(path):
    """-> (TARGET_FRAMES RGB frames, clip length)."""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:                                   # broken header: decode everything
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return [np.zeros((SIZE, SIZE, 3), np.uint8)] * TARGET_FRAMES, 0
        return [frames[i] for i in sample_indices(len(frames))], len(frames)
    frames = []
    for i in sample_indices(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ret else np.zeros((SIZE, SIZE, 3), np.uint8))
    cap.release()
    return frames, n


@torch.no_grad()
def encode(model, frames, device):
    out = []
    for fr in frames:
        arr = (np.array(Image.fromarray(fr).resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255.0 - MEAN) / STD
        y = model(torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device))
        if isinstance(y, dict):
            y = y.get('x_norm_clstoken', y.get('cls_token'))
        out.append(y.squeeze(0).cpu())
    return torch.stack(out).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_root', required=True, help='IEMOCAP_full_release directory')
    ap.add_argument('--dinov3_repo', required=True, help='clone of github.com/facebookresearch/dinov3')
    ap.add_argument('--dinov3_ckpt', required=True, help='dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    ap.add_argument('--sessions', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--resume', action='store_true', help='skip utterances already extracted')
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load(a.dinov3_repo, 'dinov3_vitl16', source='local', pretrained=a.dinov3_ckpt).to(device).eval()
    for session in a.sessions:
        sdir = os.path.join(a.raw_root, f'Session{session}')
        labels, trans = session_labels(sdir)
        clips = sorted(p for p in Path(sdir, 'sentences', 'avi').rglob('*.avi')
                       if not p.name.startswith('._'))
        for clip in tqdm(clips, desc=f'video S{session}'):
            uid, dialog = clip.stem, clip.parent.name
            out = os.path.join(sdir, 'sentences', 'video_features', dialog, f'{uid}.pkl')
            if a.resume and os.path.exists(out):
                continue
            frames, total = read_frames(clip)
            feats = encode(model, frames, device)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            data = {'features': feats, 'utterance_id': uid, 'session': session, 'dialog': dialog,
                    'avi_path': str(clip), 'total_frames': total, 'target_frames': TARGET_FRAMES,
                    'feature_shape': feats.shape, 'feature_dim': feats.shape[-1], 'model_name': 'dinov3_vitl16',
                    'transcription': trans.get(uid, ''), **labels.get(uid, {})}
            with open(out, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

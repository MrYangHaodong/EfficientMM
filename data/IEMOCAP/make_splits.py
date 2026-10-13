"""Write IEMOCAP cross-subject split CSVs (only needed without the paper's split files).

    python data/IEMOCAP/make_splits.py [--root $IEMOCAP_ROOT] [--out $IEMOCAP_ROOT]

Keeps utterances with all six modality files and one of 4 emotions (neutral, angry,
happy, sad). Folds are session-disjoint:

    fold 0: train S1-S3, val S4, test S5
    fold 1: train S2,S3,S5, val S1, test S4
    fold 2: train S1,S4,S5, val S2, test S3

Writes {train,val,test}_cs_split{0,1,2}.csv. The folds contain exactly the utterances of
the paper's split files; the row order (and hence the training batch order) is a
seeded shuffle and may differ from the paper's files, so use those files for an exact
reproduction.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _sub in ('numeric', 'multiarray', 'umath', 'numerictypes', '_methods', '_dtype', '_internal'):
    if f'numpy._core.{_sub}' not in sys.modules and hasattr(np.core, _sub):
        sys.modules[f'numpy._core.{_sub}'] = getattr(np.core, _sub)

COLS = ['utterance_id', 'session', 'dialog', 'emotion', 'emotion_name', 'emotion_label',
        'v', 'a', 'd', 'transcription']
FOLDS = {0: {'train': [1, 2, 3], 'val': [4], 'test': [5]},
         1: {'train': [2, 3, 5], 'val': [1], 'test': [4]},
         2: {'train': [1, 4, 5], 'val': [2], 'test': [3]}}


def _find(root: Path, dialog: str, uid: str, folder: str, ext: str) -> Path:
    """Flat layout <root>/<folder>/<dialog>/<uid>.<ext>, else Session<k>/sentences/<folder>/... (as the loader)."""
    flat = root / folder / dialog / f'{uid}.{ext}'
    return flat if flat.exists() else root / f'Session{int(uid[3:5])}' / 'sentences' / folder / dialog / f'{uid}.{ext}'


def collect(root: Path) -> pd.DataFrame:
    """Every utterance with all six modalities and a 4-class label, ordered by (dialog, utterance)."""
    texts = list((root / 'text_features').glob('*/*.pkl')) + list(root.glob('Session*/sentences/text_features/*/*.pkl'))
    rows = []
    for tp in sorted({(p.parent.name, p.stem): p for p in texts if not p.name.startswith('._')}.items()):
        (dialog, uid), tp = tp
        need = [_find(root, dialog, uid, d, 'pkl') for d in ('audio_features', 'video_features')] + \
               [_find(root, dialog, uid, d, 'txt') for d in ('MOCAP_hand', 'MOCAP_head', 'MOCAP_rotated')]
        if not all(p.exists() for p in need):
            continue
        meta = pickle.load(open(tp, 'rb'))
        if meta.get('emotion_label', -1) == -1:
            continue
        rows.append({'utterance_id': uid, 'session': int(uid[3:5]), 'dialog': dialog,
                     'emotion': meta.get('emotion', ''), 'emotion_name': meta.get('emotion_name', ''),
                     'emotion_label': meta['emotion_label'], 'v': meta.get('v', float('nan')),
                     'a': meta.get('a', float('nan')), 'd': meta.get('d', float('nan')),
                     'transcription': meta.get('transcription', '')})
    return pd.DataFrame(rows, columns=COLS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.environ.get('IEMOCAP_ROOT', 'datasets/IEMOCAP'), help='IEMOCAP feature root')
    ap.add_argument('--out', default=None, help='where to write the CSVs (default: --root)')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out or a.root)
    out.mkdir(parents=True, exist_ok=True)
    df = collect(root).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    print(f'{len(df)} utterances; classes {df["emotion"].value_counts().to_dict()}')
    for fold, sessions in FOLDS.items():
        for split, s in sessions.items():
            part = df[df['session'].isin(s)].reset_index(drop=True)
            part.to_csv(out / f'{split}_cs_split{fold}.csv', index=False)
            print(f'  fold {fold} {split:5s} sessions {s}: {len(part)} utterances')


if __name__ == '__main__':
    main()

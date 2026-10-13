"""
IEMOCAP Dataset Loader — 6-Modality Emotion Classification.

Modalities
----------
video        : DINOv3 ViT-L/16 features   (100, 1024)   — fixed length
audio        : WAV2VEC2_XLSR_1B features   (T_a, 1280)   — fixed (12 s pad → T_a=599)
text         : BERT-base token sequence    (T_t, 768)    — variable (3–147 tokens)
mocap_hand   : Hand MOCAP XYZ coords       (T_m, 18)     — variable (120 Hz)
mocap_head   : Head pitch/roll/yaw + trans (T_m, 6)      — variable (120 Hz)
mocap_rotated: Face marker XYZ coords      (T_m, 165)    — variable (120 Hz)

Label
-----
4-class emotion classification:
    0 = neutral (neu)
    1 = angry   (ang)
    2 = happy   (hap)
    3 = sad     (sad)
Utterances with other emotion labels (fru, exc, …) are excluded.

Data splits
-----------
Three random 70/15/15 stratified train/val/test splits are pre-generated and
stored as CSVs in IEMOCAP_ROOT:

    Split 0 (seed 42)  : train.csv  / val.csv  / test.csv
    Split 1 (seed 123) : train_split1.csv / val_split1.csv / test_split1.csv
    Split 2 (seed 456) : train_split2.csv / val_split2.csv / test_split2.csv

Generate them once with:
    python get_data.py --generate-splits

__getitem__ returns
-------------------
[video, video_low, audio, audio_low, text, text_low,
 mocap_hand, mocap_hand_low, mocap_head, mocap_head_low,
 mocap_rotated, mocap_rotated_low, label]
  video             : FloatTensor (T, 1024)      — full T (native length)
  video_low         : FloatTensor (T//r, 1024)   — time-compressed
  audio             : FloatTensor (T_a, 1280)
  audio_low         : FloatTensor (T_a//r, 1280)
  text              : FloatTensor (T_t, 768)     — BERT token sequence, variable T
  text_low          : FloatTensor (T_t//r, 768)  — time-compressed
  mocap_hand        : FloatTensor (T_m, 18)
  mocap_hand_low    : FloatTensor (T_m//r, 18)
  mocap_head        : FloatTensor (T_m, 6)
  mocap_head_low    : FloatTensor (T_m//r, 6)
  mocap_rotated     : FloatTensor (T_m, 165)
  mocap_rotated_low : FloatTensor (T_m//r, 165)
  label             : int  (0-3)
where r = time_compression_ratio (default 1 = no compression).

Batch collation pads variable-length modalities to the longest T in the batch
via ``pad_sequence(batch_first=True)``.
"""

import pickle
import argparse
import sys
from pathlib import Path
from common.paths import IEMOCAP_ROOT as _IEMOCAP_ROOT
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# numpy>=2.0  vs  <2.0 pickle compat: pkl files saved under numpy>=2.0 reference
# module path ``numpy._core.*`` which is only a partial backport in numpy<2.0
# (the real submodule live under ``numpy.core``).  Always alias all
# submodules pickle might resolve so ``pickle.load`` works regardless.
for _sub in ('numeric', 'multiarray', 'umath', 'numerictypes',
              '_methods', '_dtype', '_internal'):
    fq = f'numpy._core.{_sub}'
    if fq not in sys.modules and hasattr(np.core, _sub):
        sys.modules[fq] = getattr(np.core, _sub)

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IEMOCAP_ROOT = Path(_IEMOCAP_ROOT)

EMOTION_NAMES: Dict[str, str] = {
    'neu': 'neutral', 'ang': 'angry', 'hap': 'happy', 'sad': 'sad',
    'fru': 'frustrated', 'exc': 'excited', 'fea': 'fearful',
    'sur': 'surprised', 'dis': 'disgusted', 'oth': 'other', 'xxx': 'undefined',
}
FOUR_CLASS: Dict[str, int] = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3}
NUM_CLASSES = 4

FEATURE_DIMS: Dict[str, int] = {
    'video':         1024,   # DINOv3 ViT-L/16 CLS token
    'audio':         1280,   # WAV2VEC2_XLSR_1B frame features
    'text':           768,   # BERT-base-uncased last-hidden-state token sequence
    'mocap_hand':      18,   # 6 hand markers × 3 (X, Y, Z)
    'mocap_head':       6,   # pitch, roll, yaw, tra_x, tra_y, tra_z
    'mocap_rotated':  165,   # 55 face markers × 3 (X, Y, Z)
}

# Random seeds for the 3 splits
_SPLIT_SEEDS = [42, 123, 456]
# Fractions: train / val / test
_TRAIN_FRAC = 0.70
_VAL_FRAC   = 0.15
# test gets the remainder (0.15)

# CSV column names
_CSV_COLS = [
    'utterance_id', 'session', 'dialog',
    'emotion', 'emotion_name', 'emotion_label',
    'v', 'a', 'd', 'transcription',
]


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def _collect_all_valid_samples() -> 'pd.DataFrame':
    """
    Scan all 5 sessions, collect every utterance that has:
      - All 6 modality files present (audio_features, video_features,
        text_features, MOCAP_hand, MOCAP_head, MOCAP_rotated)
      - A valid 4-class emotion label (neu / ang / hap / sad)

    Returns a DataFrame with columns defined by _CSV_COLS.
    """
    rows = []
    for sess in range(1, 6):
        sess_dir = IEMOCAP_ROOT / f'Session{sess}'
        wav_base = sess_dir / 'sentences' / 'wav'

        for wav_path in sorted(wav_base.rglob('*.wav')):
            if wav_path.name.startswith('._'):
                continue

            uid    = wav_path.stem          # e.g. Ses01F_impro01_F000
            dialog = wav_path.parent.name   # e.g. Ses01F_impro01

            # --- check all 6 modality files exist ---
            audio_p = sess_dir / 'sentences' / 'audio_features' / dialog / f'{uid}.pkl'
            video_p = sess_dir / 'sentences' / 'video_features' / dialog / f'{uid}.pkl'
            text_p  = sess_dir / 'sentences' / 'text_features'  / dialog / f'{uid}.pkl'
            mhand_p = sess_dir / 'sentences' / 'MOCAP_hand'     / dialog / f'{uid}.txt'
            mhead_p = sess_dir / 'sentences' / 'MOCAP_head'     / dialog / f'{uid}.txt'
            mrot_p  = sess_dir / 'sentences' / 'MOCAP_rotated'  / dialog / f'{uid}.txt'

            if not all(p.exists() for p in
                       [audio_p, video_p, text_p, mhand_p, mhead_p, mrot_p]):
                continue

            # --- load label info from audio pkl (fastest, smallest) ---
            with open(audio_p, 'rb') as f:
                meta = pickle.load(f)

            emotion_label = meta.get('emotion_label', -1)
            if emotion_label == -1:
                continue

            rows.append({
                'utterance_id':  uid,
                'session':       sess,
                'dialog':        dialog,
                'emotion':       meta.get('emotion', ''),
                'emotion_name':  meta.get('emotion_name', ''),
                'emotion_label': emotion_label,
                'v':             meta.get('v', float('nan')),
                'a':             meta.get('a', float('nan')),
                'd':             meta.get('d', float('nan')),
                'transcription': meta.get('transcription', ''),
            })

    df = pd.DataFrame(rows, columns=_CSV_COLS)
    print(f'Collected {len(df)} valid 4-class utterances '
          f'(class distribution: {dict(df["emotion"].value_counts())})')
    return df


def generate_splits(data_root: Optional[Path] = None, verbose: bool = True) -> None:
    """
    Build 3 stratified random 70/15/15 train/val/test splits from all
    valid IEMOCAP utterances and save them as CSVs to ``data_root``.

    Split 0 (seed 42)  → train.csv / val.csv / test.csv
    Split 1 (seed 123) → train_split1.csv / val_split1.csv / test_split1.csv
    Split 2 (seed 456) → train_split2.csv / val_split2.csv / test_split2.csv
    """
    if data_root is None:
        data_root = IEMOCAP_ROOT
    data_root = Path(data_root)

    df = _collect_all_valid_samples()

    for split_idx, seed in enumerate(_SPLIT_SEEDS):
        rng = np.random.RandomState(seed)

        train_rows, val_rows, test_rows = [], [], []
        for label in sorted(df['emotion_label'].unique()):
            class_df = df[df['emotion_label'] == label].reset_index(drop=True)
            idx = rng.permutation(len(class_df))

            n_train = int(len(class_df) * _TRAIN_FRAC)
            n_val   = int(len(class_df) * _VAL_FRAC)

            train_rows.append(class_df.iloc[idx[:n_train]])
            val_rows.append(class_df.iloc[idx[n_train:n_train + n_val]])
            test_rows.append(class_df.iloc[idx[n_train + n_val:]])

        train_df = pd.concat(train_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
        val_df   = pd.concat(val_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
        test_df  = pd.concat(test_rows).sample(frac=1, random_state=seed).reset_index(drop=True)

        # File names: split 0 uses the default names, splits 1/2 use suffixes
        suffix = '' if split_idx == 0 else f'_split{split_idx}'
        train_path = data_root / f'train{suffix}.csv'
        val_path   = data_root / f'val{suffix}.csv'
        test_path  = data_root / f'test{suffix}.csv'

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path,     index=False)
        test_df.to_csv(test_path,   index=False)

        if verbose:
            label_dist = dict(train_df['emotion'].value_counts())
            print(f'\nSplit {split_idx} (seed={seed}):')
            print(f'  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}')
            print(f'  train class dist: {label_dist}')
            print(f'  → {train_path.name} / {val_path.name} / {test_path.name}')


# ---------------------------------------------------------------------------
# MOCAP loading helper
# ---------------------------------------------------------------------------

def _load_mocap(path: Path) -> np.ndarray:
    """
    Parse a MOCAP .txt file (MOCAP_hand / MOCAP_head / MOCAP_rotated).

    Format:
        Row 0: marker names (skip)
        Row 1: coordinate labels (skip)
        Rows 2+: Frame#  Time  feature_col_0  feature_col_1  ...

    NaN values (missing markers) are replaced with 0.

    Returns:
        np.ndarray of shape (T, D), dtype float32
    """
    df = pd.read_csv(path, sep=r'\s+', header=None, skiprows=2)
    # Columns 0=Frame#, 1=Time, 2: are features
    feat = df.iloc[:, 2:].values.astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0)
    return feat   # (T, D)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IEMOCAPDataset(Dataset):
    """
    IEMOCAP 6-modality emotion classification dataset.

    Loads pre-extracted features from pickle / txt files.

    Args:
        data_root:              IEMOCAP root directory (default: IEMOCAP_ROOT).
        csv_path:               Path to the split CSV (train.csv / val.csv / test.csv).
        modalities:             Subset of modality names to load (default: all 6).
                                Valid names: 'video', 'audio', 'text',
                                             'mocap_hand', 'mocap_head', 'mocap_rotated'.
        max_seq_len:            Truncate all modalities to this many frames along T.
                                0 = no truncation (default).
        time_compression_ratio: Uniform subsampling factor for the low (selector) path.
                                1 = no compression (default).  Each feature is returned
                                twice: once at full T, once at T // ratio.
    """

    ALL_MODALITIES = ['video', 'audio', 'text', 'mocap_hand', 'mocap_head', 'mocap_rotated']

    def __init__(
        self,
        data_root: str = str(IEMOCAP_ROOT),
        csv_path: str = None,
        modalities: List[str] = None,
        max_seq_len: int = 0,
        time_compression_ratio: int = 1,
        use_batched_fusion: bool = True,
        available_sessions: Optional[List[int]] = None,
    ):
        self.data_root              = Path(data_root)
        self.modalities             = modalities if modalities is not None else self.ALL_MODALITIES
        self.max_seq_len            = max_seq_len
        self.time_compression_ratio = max(1, int(time_compression_ratio))
        self.use_batched_fusion     = use_batched_fusion

        if csv_path is None:
            csv_path = str(self.data_root / 'train.csv')

        self.df = pd.read_csv(csv_path)
        # Optional restriction to sessions whose feature archives are
        # already extracted on disk.  Useful while large session
        # archives are still transferring / unpacking.
        if available_sessions is not None:
            avail = set(int(s) for s in available_sessions)
            before = len(self.df)
            self.df = self.df[self.df['session'].isin(avail)].reset_index(drop=True)
            after = len(self.df)
            print(f'  Filtered to sessions {sorted(avail)}: '
                  f'{before} -> {after} samples')
        self.samples = self.df.to_dict('records')

    # ------------------------------------------------------------------
    # Feature loading
    # ------------------------------------------------------------------

    def _load_pkl_features(self, path: Path) -> np.ndarray:
        """Load a .pkl and return float32 array of shape (T, C)."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            feat = data['features']
        else:
            feat = data

        if hasattr(feat, 'numpy'):
            feat = feat.numpy()
        feat = np.array(feat, dtype=np.float32)

        # Normalise to 2-D: (1, T, C) → (T, C), (1, C) → (1, C), (C,) → (1, C)
        if feat.ndim == 3 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        elif feat.ndim == 1:
            feat = feat.reshape(1, -1)

        return feat   # (T, C)

    def _normalize(self, feat: np.ndarray) -> np.ndarray:
        """Max L2-norm normalisation: (T, C) → (T, C)."""
        norms    = np.linalg.norm(feat, axis=1, keepdims=True)  # (T, 1)
        max_norm = norms.max()
        return feat / (max_norm + 1e-8)

    def _resample(self, feat: np.ndarray) -> np.ndarray:
        """Resample toward max_seq_len.

        T > max_seq_len: always downsample via linear interpolation.
        T < max_seq_len: zero-pad when use_batched_fusion=True (batch fusion needs
                         fixed-length inputs); leave as-is when use_batched_fusion=False.
        No-op when max_seq_len <= 0 or T == max_seq_len.
        """
        if self.max_seq_len <= 0:
            return feat
        T = feat.shape[0]
        if T == self.max_seq_len:
            return feat
        if T > self.max_seq_len:
            x = torch.FloatTensor(feat).unsqueeze(0).permute(0, 2, 1)  # (1, C, T)
            x = F.interpolate(x, size=self.max_seq_len, mode='linear', align_corners=False)
            return x.permute(0, 2, 1).squeeze(0).numpy()               # (max_seq_len, C)
        if self.use_batched_fusion:
            pad = np.zeros((self.max_seq_len - T, feat.shape[1]), dtype=np.float32)
            return np.concatenate([feat, pad], axis=0)
        return feat

    def _compress_time(self, feat: np.ndarray) -> np.ndarray:
        """Uniformly subsample tokens: (T, C) → (T // ratio, C)."""
        if self.time_compression_ratio <= 1:
            return feat
        return feat[::self.time_compression_ratio]

    def _resolve_modality_path(self, session, dialog, uid, modality_dir, ext):
        """Resolve modality feature file path.  Supports two on-disk
        layouts (auto-detected, flat takes precedence):

          1. Flat:    ``<data_root>/<modality_dir>/<dialog>/<uid>.<ext>``
          2. Nested:  ``<data_root>/Session{N}/sentences/<modality_dir>/<dialog>/<uid>.<ext>``

        The flat layout is what the released tar.gz archives extract to.
        Nested is the IEMOCAP_full_release directory tree.
        """
        flat = self.data_root / modality_dir / dialog / f'{uid}.{ext}'
        if flat.exists():
            return flat
        return (self.data_root / f'Session{session}' / 'sentences' /
                modality_dir / dialog / f'{uid}.{ext}')

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> List:
        """
        Returns:
            list: [video, video_low, audio, audio_low, text, text_low,
                   mocap_hand, mocap_hand_low, mocap_head, mocap_head_low,
                   mocap_rotated, mocap_rotated_low, label]
                  Only modalities in ``self.modalities`` are included; label is last.
                  Each modality is returned twice: high (full-T) and low (compressed-T).
        """
        row     = self.samples[idx]
        uid     = row['utterance_id']
        session = row['session']
        dialog  = row['dialog']
        label   = int(row['emotion_label'])

        features = []

        if 'video' in self.modalities:
            path = self._resolve_modality_path(session, dialog, uid, 'video_features', 'pkl')
            feat = self._resample(self._normalize(self._load_pkl_features(path)))
            features.append(torch.FloatTensor(feat))                           # high
            features.append(torch.FloatTensor(self._compress_time(feat)))      # low

        if 'audio' in self.modalities:
            path = self._resolve_modality_path(session, dialog, uid, 'audio_features', 'pkl')
            feat = self._resample(self._normalize(self._load_pkl_features(path)))
            features.append(torch.FloatTensor(feat))                           # high
            features.append(torch.FloatTensor(self._compress_time(feat)))      # low

        if 'text' in self.modalities:
            path = self._resolve_modality_path(session, dialog, uid, 'text_features', 'pkl')
            feat = self._resample(self._normalize(self._load_pkl_features(path)))
            features.append(torch.FloatTensor(feat))                           # high
            features.append(torch.FloatTensor(self._compress_time(feat)))      # low

        if 'mocap_hand' in self.modalities:
            path = self._resolve_modality_path(session, dialog, uid, 'MOCAP_hand', 'txt')
            feat = self._resample(self._normalize(_load_mocap(path)))
            features.append(torch.FloatTensor(feat))                           # high
            features.append(torch.FloatTensor(self._compress_time(feat)))      # low

        if 'mocap_head' in self.modalities:
            path = self._resolve_modality_path(session, dialog, uid, 'MOCAP_head', 'txt')
            feat = self._resample(self._normalize(_load_mocap(path)))
            features.append(torch.FloatTensor(feat))                           # high
            features.append(torch.FloatTensor(self._compress_time(feat)))      # low

        if 'mocap_rotated' in self.modalities:
            path = self._resolve_modality_path(session, dialog, uid, 'MOCAP_rotated', 'txt')
            feat = self._resample(self._normalize(_load_mocap(path)))
            features.append(torch.FloatTensor(feat))                           # high
            features.append(torch.FloatTensor(self._compress_time(feat)))      # low

        return features + [label]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_dims(self) -> Dict[str, int]:
        return {m: FEATURE_DIMS[m] for m in self.modalities}

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def _collate_features(batch) -> Tuple[List[torch.Tensor], List[int]]:
    """Shared helper: split features and labels, pad variable-length tensors."""
    n_features = len(batch[0]) - 1   # last element is label
    feature_lists = [[] for _ in range(n_features)]
    labels = []

    for sample in batch:
        for i in range(n_features):
            feature_lists[i].append(sample[i])
        labels.append(sample[-1])

    # pad_sequence: list of (T_i, C) → (B, T_max, C)
    padded = [pad_sequence(fl, batch_first=True) for fl in feature_lists]
    return padded, labels


def collate_fn(batch):
    """
    Collate function with zero-padding for variable-length sequences.

    Returns:
        tuple of (feature_tensors..., label_tensor)
        Feature tensors: (B, T_max, C) float32.
        label_tensor:    (B, 1) int64.
    """
    padded, labels = _collate_features(batch)
    label_tensor = torch.tensor(labels, dtype=torch.long).view(-1, 1)
    return tuple(padded) + (label_tensor,)


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    data_root: str = str(IEMOCAP_ROOT),
    batch_size: int = 32,
    num_workers: int = 4,
    modalities: List[str] = None,
    split_id: int = 0,
    train_shuffle: bool = True,
    max_seq_len: int = 0,
    time_compression_ratio: int = 1,
    use_batched_fusion: bool = True,
    available_sessions: Optional[List[int]] = None,
    split_mode: str = 'random',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, val, and test DataLoaders for IEMOCAP.

    Args:
        data_root:              IEMOCAP root directory.
        batch_size:             Samples per batch.
        num_workers:            DataLoader worker processes.
        modalities:             Which modalities to load (default: all 6).
        split_id:               Which split to use (0, 1, or 2).
        train_shuffle:          Whether to shuffle the training loader.
        max_seq_len:            Truncate all modalities to this T. 0 = no truncation.
        time_compression_ratio: Uniform subsampling for the low path.
        use_batched_fusion:     Whether to zero-pad short sequences to max_seq_len.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if modalities is None:
        modalities = IEMOCAPDataset.ALL_MODALITIES

    data_root = Path(data_root)
    if split_mode == 'cross_subject':
        suffix = f'_cs_split{split_id}'
    else:
        suffix = '' if split_id == 0 else f'_split{split_id}'
    train_csv = data_root / f'train{suffix}.csv'
    val_csv   = data_root / f'val{suffix}.csv'
    test_csv  = data_root / f'test{suffix}.csv'

    # Check splits exist; generate them if missing
    if not train_csv.exists():
        print(f'Split CSV not found ({train_csv}). Generating all 3 splits...')
        generate_splits(data_root)

    print(f'\nLoading IEMOCAP (split_id={split_id}, split_mode={split_mode}) from {data_root}')
    print(f'  Modalities : {modalities}')
    print(f'  Feature dims: {FEATURE_DIMS}')
    print(f'  max_seq_len={max_seq_len}  time_compression_ratio={time_compression_ratio}')

    dataset_kwargs = dict(
        data_root              = str(data_root),
        modalities             = modalities,
        max_seq_len            = max_seq_len,
        time_compression_ratio = time_compression_ratio,
        use_batched_fusion     = use_batched_fusion,
        available_sessions     = available_sessions,
    )

    train_dataset = IEMOCAPDataset(**dataset_kwargs, csv_path=str(train_csv))
    val_dataset   = IEMOCAPDataset(**dataset_kwargs, csv_path=str(val_csv))
    test_dataset  = IEMOCAPDataset(**dataset_kwargs, csv_path=str(test_csv))

    print(f'  Train: {len(train_dataset)} samples')
    print(f'  Val  : {len(val_dataset)} samples')
    print(f'  Test : {len(test_dataset)} samples')

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = train_shuffle,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        collate_fn  = collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        collate_fn  = collate_fn,
    )

    return train_loader, val_loader, test_loader


def get_feature_dims(modalities: List[str] = None) -> Dict[str, int]:
    """Return feature dimensions for the requested modalities."""
    if modalities is None:
        modalities = IEMOCAPDataset.ALL_MODALITIES
    return {m: FEATURE_DIMS[m] for m in modalities}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IEMOCAP dataloader utility')
    parser.add_argument('--generate-splits', action='store_true',
                        help='Generate 3 random train/val/test split CSVs and exit')
    parser.add_argument('--data-root', type=str, default=str(IEMOCAP_ROOT),
                        help='IEMOCAP root directory')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run a quick smoke test on all 3 splits')
    args = parser.parse_args()

    if args.generate_splits:
        print('=' * 60)
        print('Generating IEMOCAP train/val/test splits')
        print('=' * 60)
        generate_splits(Path(args.data_root))
        print('\nDone.')

    elif args.smoke_test:
        print('=' * 60)
        print('IEMOCAP dataloader smoke test')
        print('=' * 60)
        MODALITY_NAMES = ['video', 'audio', 'text', 'mocap_hand', 'mocap_head', 'mocap_rotated']
        TENSOR_NAMES = [f'{m}_{suf}' for m in MODALITY_NAMES for suf in ('high', 'low')]
        for split_id in range(3):
            print(f'\n--- Split {split_id} ---')
            try:
                train_loader, val_loader, test_loader = get_dataloader(
                    data_root=args.data_root, batch_size=4, num_workers=0,
                    split_id=split_id, time_compression_ratio=4,
                )
                batch = next(iter(train_loader))
                feature_tensors = batch[:-1]
                labels = batch[-1]
                for name, tensor in zip(TENSOR_NAMES, feature_tensors):
                    print(f'   {name:22s}: {tuple(tensor.shape)}  dtype={tensor.dtype}')
                print(f'   {"label":22s}: {tuple(labels.shape)}  dtype={labels.dtype}  '
                      f'values={labels.flatten().tolist()}')
            except Exception as exc:
                import traceback
                print(f'   ERROR: {exc}')
                traceback.print_exc()
        print('\nDone.')

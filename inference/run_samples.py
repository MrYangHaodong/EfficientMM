#!/usr/bin/env python
"""Run SeMARC on the sample utterances in samples/IEMOCAP, one at a time.

    python inference/run_samples.py                       # from the pre-extracted features (samples/*/features)
    python inference/run_samples.py --from_raw \
        --dinov3_repo /path/to/dinov3 --dinov3_ckpt /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
                                                          # from the raw clip / wav / transcript (samples/*/raw)

``--from_raw`` runs the paper's feature extractors first (DINOv3 ViT-L/16 on 100 frames of the
clip, wav2vec2 XLS-R 1B on the first 12 s of audio, bert-base-uncased on the transcript; the
motion-capture streams are used as they are) and then the same SeMA + ARC loop. Extraction
happens only for the modalities ARC actually acquires, in the order it acquires them.

Each sample is scored with the fold whose model never saw its session (fold 0: session 5,
fold 1: sessions 1 and 4, fold 2: sessions 2 and 3); override with --fold. Prints the acquired
modalities in order, the prediction and the label.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _s in ('numeric', 'multiarray', 'umath', 'numerictypes', '_methods', '_dtype', '_internal'):   # pkls written under numpy>=2
    if f'numpy._core.{_s}' not in sys.modules and hasattr(np.core, _s):
        sys.modules[f'numpy._core.{_s}'] = getattr(np.core, _s)
from arc import policy                                        # noqa: E402
from arc.prefix_cache import load_backbone                    # noqa: E402
from common.paths import config                               # noqa: E402
from data.IEMOCAP.get_data import IEMOCAPDataset, FEATURE_DIMS, NUM_CLASSES, _load_mocap   # noqa: E402
from inference.online import SeMARC                           # noqa: E402

MODS = list(IEMOCAPDataset.ALL_MODALITIES)
CLASSES = ['neutral', 'angry', 'happy', 'sad']
FOLD_OF_SESSION = {5: 0, 1: 1, 4: 1, 2: 2, 3: 2}
MOCAP_DIR = {'mocap_hand': 'MOCAP_hand', 'mocap_head': 'MOCAP_head', 'mocap_rotated': 'MOCAP_rotated'}


def preprocessor(max_seq_len):
    """The loader's per-modality preprocessing (max-L2-norm normalisation, resample / zero-pad to T)."""
    ds = IEMOCAPDataset.__new__(IEMOCAPDataset)
    ds.max_seq_len, ds.use_batched_fusion, ds.time_compression_ratio, ds.modalities = max_seq_len, True, 1, MODS
    return lambda feat: torch.from_numpy(ds._resample(ds._normalize(np.asarray(feat, np.float32))))[None]   # [1, T, D]


class FeatureSource:
    """Pre-extracted features of one sample folder."""

    def __init__(self, sample_dir, prep):
        self.d, self.prep = sample_dir, prep
        self.ds = IEMOCAPDataset.__new__(IEMOCAPDataset)

    def __call__(self, m):
        if m in MOCAP_DIR:
            return self.prep(_load_mocap(os.path.join(self.d, 'features', f'{MOCAP_DIR[m]}.txt')))
        return self.prep(self.ds._load_pkl_features(os.path.join(self.d, 'features', f'{m}_features.pkl')))


class RawSource:
    """Extracts a modality from the raw files when it is requested (extractors loaded once)."""

    def __init__(self, sample_dir, prep, ex, device):
        self.d, self.prep, self.ex, self.dev = sample_dir, prep, ex, device
        self.uid = os.path.basename(sample_dir.rstrip('/'))

    def __call__(self, m):
        import torchaudio
        from preprocess import extract_audio as EA, extract_text as ET, extract_video as EV
        raw = os.path.join(self.d, 'raw')
        if m in MOCAP_DIR:
            return self.prep(_load_mocap(os.path.join(raw, f'{MOCAP_DIR[m]}.txt')))
        if m == 'video':
            frames, _ = EV.read_frames(os.path.join(raw, f'{self.uid}.avi'))
            return self.prep(EV.encode(self.ex['dinov3'], frames, self.dev))
        if m == 'audio':
            sig, sr = torchaudio.load(os.path.join(raw, f'{self.uid}.wav'))
            with torch.no_grad():
                feats, _ = self.ex['xlsr'](EA.preprocess(sig, sr, int(EA.SR * EA.DURATION)).to(self.dev))
            return self.prep(feats.cpu().numpy()[0])
        if m == 'text':
            line = open(os.path.join(raw, 'transcription.txt'), encoding='latin-1').read().strip()
            text = line.split(']:', 1)[1].strip() if ']:' in line else line
            return self.prep(ET.encode(self.ex['bert_tok'], self.ex['bert'], [text], self.dev)[0])
        raise KeyError(m)


class LazyInputs(dict):
    """{modality: tensor} that extracts / loads a modality the first time SeMARC reads it."""

    def __init__(self, source):
        super().__init__(); self.source = source; self.order = []

    def __getitem__(self, m):
        if m not in self:
            t0 = time.perf_counter(); dict.__setitem__(self, m, self.source(m)); self.order.append((m, (time.perf_counter() - t0) * 1e3))
        return dict.__getitem__(self, m)


def load_extractors(a, device):
    import torchaudio
    from transformers import AutoModel, AutoTokenizer
    if not (a.dinov3_repo and a.dinov3_ckpt):
        sys.exit('--from_raw needs --dinov3_repo and --dinov3_ckpt (github.com/facebookresearch/dinov3, ViT-L/16 weights)')
    print('[raw] loading DINOv3 ViT-L/16, wav2vec2 XLS-R 1B and bert-base-uncased ...', flush=True)
    return {'dinov3': torch.hub.load(a.dinov3_repo, 'dinov3_vitl16', source='local', pretrained=a.dinov3_ckpt).to(device).eval(),
            'xlsr': torchaudio.pipelines.WAV2VEC2_XLSR_1B.get_model().to(device).eval(),
            'bert_tok': AutoTokenizer.from_pretrained('bert-base-uncased'),
            'bert': AutoModel.from_pretrained('bert-base-uncased').to(device).eval()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', default=os.path.join(ROOT, 'samples', 'IEMOCAP'))
    ap.add_argument('--only', nargs='+', default=None, help='utterance ids to run (default: all in manifest.csv)')
    ap.add_argument('--fold', type=int, default=None, help='use this fold for every sample')
    ap.add_argument('--from_raw', action='store_true', help='extract the features from samples/*/raw first')
    ap.add_argument('--dinov3_repo', default=None); ap.add_argument('--dinov3_ckpt', default=None)
    ap.add_argument('--sema_dir', default=os.path.join(ROOT, 'checkpoints', 'sema'))
    ap.add_argument('--arc_dir', default=os.path.join(ROOT, 'checkpoints', 'arc'))
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    man = pd.read_csv(os.path.join(a.samples, 'manifest.csv'))
    if a.only:
        man = man[man.utterance_id.isin(a.only)]
    ex = load_extractors(a, device) if a.from_raw else None
    models, deployed = {}, config('arc')['deployed']
    correct = 0
    for r in man.itertuples():
        fold = a.fold if a.fold is not None else FOLD_OF_SESSION[int(r.session)]
        if fold not in models:
            cfg = json.load(open(os.path.join(a.sema_dir, f'results_fold{fold}.json')))
            sema, _ = load_backbone(a.sema_dir, fold, FEATURE_DIMS, device)
            c = deployed[str(fold)]
            net, tables, meta = policy.load(policy.ckpt_path(a.arc_dir, fold, c['seed'], c['lam']), device)
            if tables is None:
                sys.exit('this ARC checkpoint has no gain tables; use inference/online.py with the validation cache')
            models[fold] = (SeMARC(sema, net, tables, meta['lam'], MODS, NUM_CLASSES, device), preprocessor(cfg.get('max_seq_len', 100)))
        semarc, prep = models[fold]
        sd = os.path.join(a.samples, r.utterance_id)
        x = LazyInputs(RawSource(sd, prep, ex, device) if a.from_raw else FeatureSource(sd, prep))
        t0 = time.perf_counter(); logits, order = semarc.predict(x); total = (time.perf_counter() - t0) * 1e3
        probs = F.softmax(logits.float(), -1)[0].cpu().numpy(); pred = int(probs.argmax()); correct += pred == int(r.emotion_label)
        steps = ' -> '.join(f'{m} ({ms:.0f} ms)' for m, ms in x.order)
        print(f'{r.utterance_id} (session {r.session}, fold {fold}): acquired {steps}; predicted {CLASSES[pred]} '
              f'({probs[pred]:.2f}), label {CLASSES[int(r.emotion_label)]} [{"ok" if pred == int(r.emotion_label) else "wrong"}]; '
              f'{len(order)}/6 modalities, {total:.0f} ms', flush=True)
    print(f'{correct}/{len(man)} correct')


if __name__ == '__main__':
    main()

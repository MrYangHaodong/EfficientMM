#!/usr/bin/env python
"""Audio features: wav2vec2 XLS-R 1B (torchaudio WAV2VEC2_XLSR_1B), last layer, per utterance.

    python preprocess/extract_audio.py --raw_root /path/to/IEMOCAP_full_release [--device cuda:0]

Each sentences/wav/<dialog>/<utt>.wav is converted to mono 16 kHz, cut or zero-padded to
12 s, and encoded -> features (1, 599, 1280), written with the utterance labels to
Session<k>/sentences/audio_features/<dialog>/<utt>.pkl.
"""
import argparse
import os
import pickle
import sys

import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iemocap_labels import session_labels, utterance_wavs   # noqa: E402

SR, DURATION = 16000, 12.0


def preprocess(signal, sr, num_samples):
    if sr != SR:
        signal = torchaudio.transforms.Resample(sr, SR)(signal)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    if signal.shape[1] < num_samples:
        signal = torch.nn.functional.pad(signal, (0, num_samples - signal.shape[1]))
    return signal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_root', required=True, help='IEMOCAP_full_release directory')
    ap.add_argument('--sessions', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--resume', action='store_true', help='skip utterances already extracted')
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model = torchaudio.pipelines.WAV2VEC2_XLSR_1B.get_model().to(device).eval()
    num_samples = int(SR * DURATION)
    for session in a.sessions:
        sdir = os.path.join(a.raw_root, f'Session{session}')
        labels, trans = session_labels(sdir)
        for wav in tqdm(utterance_wavs(sdir), desc=f'audio S{session}'):
            uid, dialog = wav.stem, wav.parent.name
            out = os.path.join(sdir, 'sentences', 'audio_features', dialog, f'{uid}.pkl')
            if a.resume and os.path.exists(out):
                continue
            signal, sr = torchaudio.load(str(wav))
            with torch.no_grad():
                feats, _ = model(preprocess(signal, sr, num_samples).to(device))
            feats = feats.cpu().numpy()                                   # (1, 599, 1280)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            data = {'features': feats, 'utterance_id': uid, 'session': session, 'dialog': dialog,
                    'wav_path': str(wav), 'transcription': trans.get(uid, ''), 'model_name': 'WAV2VEC2_XLSR_1B',
                    'feature_shape': feats.shape, 'target_sample_rate': SR, 'audio_duration': DURATION,
                    **labels.get(uid, {})}
            with open(out, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

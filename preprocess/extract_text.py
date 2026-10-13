#!/usr/bin/env python
"""Text features: bert-base-uncased last hidden states of the transcript, one vector per token.

    python preprocess/extract_text.py --raw_root /path/to/IEMOCAP_full_release [--device cuda:0]

Transcripts are encoded in batches of 64 utterances (sorted WAV order within a session,
max 256 tokens) and padding is dropped -> features (T_tokens, 768) including [CLS] and
[SEP]; an empty transcript gives one zero vector. Written with the utterance labels to
Session<k>/sentences/text_features/<dialog>/<utt>.pkl.
"""
import argparse
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iemocap_labels import session_labels, utterance_wavs   # noqa: E402

MODEL, BATCH = 'bert-base-uncased', 64


@torch.no_grad()
def encode(tok, model, texts, device):
    enc = tok([t.strip() if t.strip() else '[EMPTY]' for t in texts], return_tensors='pt', padding=True,
              truncation=True, max_length=256, return_attention_mask=True)
    ids, mask = enc['input_ids'].to(device), enc['attention_mask'].to(device)
    hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    out = []
    for i, t in enumerate(texts):
        if not t.strip():
            out.append(np.zeros((1, model.config.hidden_size), np.float32))
        else:
            out.append(hidden[i][mask[i].bool()].cpu().numpy().astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_root', required=True, help='IEMOCAP_full_release directory')
    ap.add_argument('--sessions', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    for session in a.sessions:
        sdir = os.path.join(a.raw_root, f'Session{session}')
        labels, trans = session_labels(sdir)
        items = [(w.stem, w.parent.name) for w in utterance_wavs(sdir)]
        for b in tqdm(range(0, len(items), BATCH), desc=f'text S{session}'):
            chunk = items[b:b + BATCH]
            feats = encode(tok, model, [trans.get(uid, '') for uid, _ in chunk], device)
            for (uid, dialog), f in zip(chunk, feats):
                out = os.path.join(sdir, 'sentences', 'text_features', dialog, f'{uid}.pkl')
                os.makedirs(os.path.dirname(out), exist_ok=True)
                data = {'features': f, 'utterance_id': uid, 'session': session, 'dialog': dialog,
                        'transcription': trans.get(uid, ''), 'feature_shape': f.shape,
                        'feature_dim': model.config.hidden_size, 'model_name': MODEL, 'pooling_strategy': 'none',
                        'timestamp': datetime.now().isoformat(), **labels.get(uid, {})}
                with open(out, 'wb') as fh:
                    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

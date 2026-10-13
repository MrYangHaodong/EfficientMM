#!/usr/bin/env python
"""Online SeMARC inference: one sample at a time, acquiring modalities as ARC decides.

    python inference/online.py --fold 0
    python inference/online.py --fold 0 --compare_cache --time

For each sample the loop is

    psi <- state(acquired set, current posterior)
    Q   <- R_theta(psi) + Q_0(psi; lam)              lam = the policy's training lambda
    if something is acquired and Q[STOP] >= max_m Q[acquire m]: stop  (ties -> STOP)
    else acquire m* = argmax_m Q[acquire m]: SeMA encodes and fuses m* only, posterior updates

and the prediction is the posterior after the last acquisition. Modalities that are never
acquired are never read by the model. Inputs are the pre-extracted features of the test
split (raw-data extraction is outside this script); ``--time`` reports the per-sample
latency of the SeMA + ARC loop after 3 warm-up samples.

``--compare_cache`` also rolls the same policy out on the prefix-tree cache and reports how
often the online acquisition sequence and prediction agree with it (the cache stores the
posteriors in float16, so rare near-ties can differ).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from arc import core, policy                                  # noqa: E402
from arc.prefix_cache import load_backbone, load_cache       # noqa: E402
from arc.prior import gain_tables                             # noqa: E402
from common.data import build_data                            # noqa: E402
from common.paths import config                               # noqa: E402


class SeMARC:
    """SeMA backbone + ARC controller for single-sample inference."""

    def __init__(self, model, net, tables, lam, mods, num_classes, device):
        self.model, self.net, self.tables, self.lam = model, net, tables, float(lam)
        self.mods, self.M, self.C, self.device = list(mods), len(mods), num_classes, device

    def _state(self, held, soft, avail):
        M = self.M
        mask = np.zeros(M, np.float32); mask[list(held)] = 1.0
        p = np.clip(soft, 1e-8, 1.0); srt = np.sort(soft)
        extra = np.array([-(p * np.log(p)).sum(), srt[-1] - srt[-2], len(held) / M, avail.sum() / M], np.float32)
        return np.concatenate([mask, avail, soft, extra]).astype(np.float32)

    def _prior(self, held, soft, avail):
        M = self.M; conf = soft.max()
        Q = np.zeros(M + 1, np.float32); Q[M] = conf
        tab = self.tables.get(frozenset(held))
        adv = (soft @ tab if tab is not None else np.zeros(M, np.float32)) * avail
        adv[list(held)] = 0.0
        Q[:M] = conf + adv - np.float32(self.lam) / avail.sum()
        return Q

    @torch.inference_mode()
    def predict(self, x, avail=None):
        """x: {modality: [1, T, feat_dim]} (only the acquired ones are read).
        Returns (logits [1, C], acquisition order as modality names)."""
        M = self.M
        avail = np.ones(M, np.float32) if avail is None else np.asarray(avail, np.float32)
        state = self.model.init_state(1, self.device)
        held, order, logits = set(), [], None
        soft = np.full(self.C, 1.0 / self.C, np.float32)
        while True:
            legal = [m for m in range(M) if m not in held and avail[m] > 0]
            if not legal:
                break
            psi = torch.from_numpy(self._state(held, soft, avail)).to(self.device)
            Q = self.net(psi[None])[0].cpu().numpy() + self._prior(held, soft, avail)
            best = max(legal, key=lambda m: (Q[m], -m))               # lowest index on ties
            if held and Q[M] >= Q[best]:
                break
            held.add(best); order.append(best)
            logits = self.model.acquire(state, self.mods[best], x[self.mods[best]].to(self.device).float())
            soft = F.softmax(logits.float(), -1)[0].cpu().numpy()
        return logits, [self.mods[m] for m in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--sema_dir', default=os.path.join(ROOT, 'checkpoints', 'sema'))
    ap.add_argument('--arc_dir', default=os.path.join(ROOT, 'checkpoints', 'arc'))
    ap.add_argument('--cache_dir', default=os.path.join(ROOT, 'cache'), help='only needed for --compare_cache or old checkpoints')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit', type=int, default=0, help='first N samples only (0 = all)')
    ap.add_argument('--time', action='store_true', help='per-sample latency of the SeMA + ARC loop')
    ap.add_argument('--compare_cache', action='store_true')
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')

    c = config('arc')['deployed'][str(a.fold)]
    net, tables, meta = policy.load(policy.ckpt_path(a.arc_dir, a.fold, c['seed'], c['lam']), device)
    if tables is None:                                        # checkpoints without stored tables
        val = load_cache(a.cache_dir, a.fold, 'val'); tables = gain_tables(val, val['M'])
    cfg = json.load(open(os.path.join(a.sema_dir, f'results_fold{a.fold}.json')))
    _, val_b, test_b, mods, num_classes = build_data(a.fold, cfg.get('batch_size', 32), 2, cfg.get('max_seq_len', 100))
    batches = test_b if a.split == 'test' else val_b
    feat_dims = {m: val_b[0][0][m].shape[-1] for m in mods}
    model, _ = load_backbone(a.sema_dir, a.fold, feat_dims, device)
    semarc = SeMARC(model, net, tables, meta['lam'], mods, num_classes, device)
    print(f'[online] fold{a.fold}: ARC seed {c["seed"]}, lam {meta["lam"]} (prior priced at the training lambda, no stop margin)', flush=True)

    samples = [({m: high[m][i:i + 1] for m in mods}, int(y[i])) for high, y in batches for i in range(len(y))]
    if a.limit:
        samples = samples[:a.limit]
    preds, labels, orders, times = [], [], [], []
    for n, (x, y) in enumerate(samples):
        if a.time and device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, order = semarc.predict(x)
        if a.time and device.type == 'cuda':
            torch.cuda.synchronize()
        if n >= 3:
            times.append((time.perf_counter() - t0) * 1e3)
        preds.append(int(logits.argmax(-1))); labels.append(y); orders.append(order)
    f1 = core.macro_f1(np.array(labels), np.array(preds))
    k = np.array([len(o) for o in orders])
    print(f'[online] {len(samples)} samples: macro-F1 {f1:.4f}  MU {k.mean() / len(mods):.3f}  '
          f'usage {dict((m, round(float(np.mean([m in o for o in orders])), 3)) for m in mods)}', flush=True)
    if a.time and times:
        print(f'[online] SeMA + ARC latency per sample: mean {np.mean(times):.2f} ms, median {np.median(times):.2f} ms '
              f'(after 3 warm-up samples)', flush=True)
    if a.compare_cache:
        cache = load_cache(a.cache_dir, a.fold, a.split); M = cache['M']
        cf1, _, _, sn = policy.rollout(net, cache, M, tables, meta['lam'])
        cache_orders = [[mods[int(i)] for i in cache['nodes'][s].split(',') if i] for s in sn[:len(orders)]]
        cache_preds = cache['softmax'][sn, np.arange(len(sn))].argmax(-1)[:len(preds)]
        print(f'[online] vs cache rollout (F1 {cf1:.4f} on the full split): same acquisition sequence '
              f'{np.mean([o == co for o, co in zip(orders, cache_orders)]):.4f}, same prediction '
              f'{np.mean(np.array(preds) == cache_preds):.4f}', flush=True)


if __name__ == '__main__':
    main()

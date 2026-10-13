"""Prefix-tree cache of a frozen SeMA backbone.

A node is an ordered sequence of acquired modality indices, stored as a comma-joined
string ('' = nothing acquired, '2,0' = modality 2 then modality 0). For every node and
every sample the cache holds SeMA's posterior after consuming that prefix, so ARC can be
fitted and evaluated without running the backbone again. The tree is complete to depth
M (every ordered subset), i.e. 1 + sum_k M!/(M-k)! nodes: 1,957 for M=6, 13,700 for M=7.

Cache file ``<cache_dir>/iemocap_fold<N>_<split>.npz``:
    nodes    [n]        node keys
    softmax  [n, N, C]  float16 posterior (the root is uniform)
    gflops   [n]        SeMA GFLOPs of the prefix (sum of per-modality branch GFLOPs)
    y        [N]        labels
    mods, M, C          modality names (canonical order), #modalities, #classes
"""
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.data import build_data, rebatch
from models.sema import SeMA

BATCH = 256


def cache_path(cache_dir, fold, split):
    return os.path.join(cache_dir, f'iemocap_fold{fold}_{split}.npz')


def load_cache(cache_dir, fold, split):
    z = np.load(cache_path(cache_dir, fold, split), allow_pickle=True)
    return {'nodes': [str(x) for x in z['nodes']], 'softmax': z['softmax'].astype(np.float32),
            'gflops': z['gflops'].astype(np.float32), 'y': z['y'],
            'mods': [str(m) for m in z['mods']], 'M': int(z['M']), 'C': int(z['C'])}


def load_backbone(sema_dir, fold, feat_dims, device):
    """SeMA from <sema_dir>/{results_fold<N>.json, best_model_fold<N>.pth}, eval mode."""
    d = sema_dir
    cfg = json.load(open(os.path.join(d, f'results_fold{fold}.json')))
    model = SeMA.from_config(cfg, feat_dims).to(device)
    model.load_state_dict(torch.load(os.path.join(d, f'best_model_fold{fold}.pth'), map_location=device,
                                     weights_only=False), strict=True)
    return model.eval(), cfg


def modality_gflops(model, mods, feat_dims, T, device):
    """GFLOPs of each modality's branch (projector + encoder + fusion) and the readout, at
    batch 1 and length T. Branches run independently, so a prefix costs the sum over its
    modalities."""
    from fvcore.nn import FlopCountAnalysis

    class _One(torch.nn.Module):
        def __init__(self, mm, m):
            super().__init__(); self.mm, self.m = mm, m

        def forward(self, inp):
            return self.mm(inp, order=[self.m])

    out = []
    for m in mods:
        fa = FlopCountAnalysis(_One(model, m).eval(), ({m: torch.zeros(1, T, feat_dims[m], device=device)},))
        fa.unsupported_ops_warnings(False); fa.uncalled_modules_warnings(False)
        out.append(fa.total() / 1e9)
    return np.array(out, np.float32)


@torch.no_grad()
def _prefix_softmax(model, high, order, device):
    """[B, len(order), C]: posterior after each prefix of ``order``."""
    _, pref, _ = model({m: high[m].to(device).float() for m in order}, order=list(order), return_prefix=True)
    return F.softmax(pref.float(), -1).cpu()


def build_tree(model, mods, batches, C, device):
    """Enumerate every ordered prefix; one forward per complete ordering fills all of its
    not-yet-filled ancestors."""
    M = len(mods)
    keys = ['']
    for k in range(1, M + 1):
        keys += [','.join(map(str, p)) for p in itertools.permutations(range(M), k)]
    ids = {k: i for i, k in enumerate(keys)}
    N = sum(len(b[1]) for b in batches)
    soft = np.empty((len(keys), N, C), np.float16)
    soft[0] = 1.0 / C
    done = np.zeros(len(keys), bool); done[0] = True
    for leaf in itertools.permutations(range(M), M):
        todo = [(k, ids[','.join(map(str, leaf[:k]))]) for k in range(1, M + 1)
                if not done[ids[','.join(map(str, leaf[:k]))]]]
        if not todo:
            continue
        off = 0
        for high, _y in batches:
            p = _prefix_softmax(model, high, [mods[i] for i in leaf], device)
            for k, nid in todo:
                soft[nid, off:off + p.shape[0]] = p[:, k - 1].numpy().astype(np.float16)
            off += p.shape[0]
        for _, nid in todo:
            done[nid] = True
    return keys, soft


def build_caches(sema_dir, cache_dir, fold, splits=('val', 'test'), device='cuda:0', num_workers=2):
    """Build (or keep) the val/test caches of one fold."""
    todo = [s for s in splits if not os.path.exists(cache_path(cache_dir, fold, s))]
    if not todo:
        print(f'[cache] fold{fold}: {list(splits)} already built', flush=True)
        return
    cfg = json.load(open(os.path.join(sema_dir, f'results_fold{fold}.json')))
    data = dict(zip(('train', 'val', 'test'), build_data(fold, cfg.get('batch_size', 32), num_workers,
                                                         cfg.get('max_seq_len', 100))[:3]))
    mods = list(cfg['modalities'])
    feat_dims = {m: data['val'][0][0][m].shape[-1] for m in mods}
    model, _ = load_backbone(sema_dir, fold, feat_dims, device)
    gf = modality_gflops(model, mods, feat_dims, cfg.get('max_seq_len', 100), device)
    print(f'[cache] fold{fold} per-modality GFLOPs: {dict(zip(mods, np.round(gf, 4)))}', flush=True)
    os.makedirs(cache_dir, exist_ok=True)
    for split in todo:
        t0 = time.time()
        batches, y = rebatch(data[split], BATCH)
        keys, soft = build_tree(model, mods, batches, cfg['num_classes'], device)
        node_gf = np.array([0.0 if k == '' else float(sum(gf[int(x)] for x in k.split(','))) for k in keys],
                           np.float32)
        np.savez_compressed(cache_path(cache_dir, fold, split), nodes=np.array(keys), softmax=soft,
                            gflops=node_gf, y=y, mods=np.array(mods), M=len(mods), C=cfg['num_classes'])
        print(f'[cache] fold{fold} {split}: {len(keys)} nodes x {len(y)} samples '
              f'({time.time() - t0:.0f}s)', flush=True)

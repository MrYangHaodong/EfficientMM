#!/usr/bin/env python
"""Train the SeMA backbone on one IEMOCAP fold.

    python training/train_sema.py --fold 0 [--out_dir checkpoints/sema]

Hyperparameters default to configs/sema.json; any flag overrides them.

Loss per batch (Section 3 of the paper), with S the modalities kept in the batch and
p_s the prediction after the first s of them (random order):

    L = CE(p_S, y) + w_ds * sum_s a_s CE(p_s, y) + w_kd * sum_s a_s KL(p_bar || p_s),
    a_s = s / sum_r r,

where p_bar is the prediction of an EMA copy of the model on the same modalities in
canonical order (no gradient). Modality dropout is applied per batch and follows a
curriculum: 0 for ``warmup_epochs``, then a linear ramp to ``max_modality_drop`` over
``ramp_epochs``. The checkpoint with the best validation macro-F1 is kept.

Writes <out_dir>/best_model_fold<N>.pth and results_fold<N>.json (the
architecture, read back by arc/prefix_cache.py).
"""
import argparse
import copy
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.data import build_data                    # noqa: E402
from models.sema import SeMA                          # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _bool(v):
    return str(v).lower() in ('1', 'true', 'yes', 'y', 't')


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'checkpoints', 'sema'))
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'sema.json'))
    ap.add_argument('--device', default='cuda:0')
    # everything below defaults to the recipe in --config
    for name, typ in [('seed', int), ('max_seq_len', int), ('batch_size', int), ('num_workers', int),
                      ('d_model', int), ('nhead', int), ('num_enc_layers', int), ('num_fusion_layers', int),
                      ('n_bottlenecks', int), ('mlp_ratio', float),
                      ('dropout', float), ('num_epochs', int), ('lr', float), ('weight_decay', float),
                      ('label_smoothing', float), ('clip_grad', float), ('max_modality_drop', float),
                      ('warmup_epochs', int), ('ramp_epochs', int), ('prefix_ds_weight', float),
                      ('prefix_kd_weight', float), ('ema_decay', float)]:
        ap.add_argument(f'--{name}', type=typ, default=None)
    ap.add_argument('--loss', choices=['ce', 'weighted_ce'], default=None)
    ap.add_argument('--seq_random_order', type=_bool, default=None, help='random modality order in training')
    ap.add_argument('--prefix_supervision', type=_bool, default=None, help='prefix CE + KD terms')
    ap.add_argument('--ema_teacher', type=_bool, default=None, help='EMA teacher for KD (else the final prediction)')
    a = ap.parse_args()
    for k, v in json.load(open(a.config)).items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    return a


def curriculum_drop(epoch, a):
    """Per-batch modality-drop probability at this epoch."""
    if epoch < a.warmup_epochs:
        return 0.0
    if epoch < a.warmup_epochs + a.ramp_epochs:
        return a.max_modality_drop * (epoch - a.warmup_epochs) / a.ramp_epochs
    return a.max_modality_drop


def sample_kept(mods, drop_p):
    """Modalities kept for this batch (at least one)."""
    if drop_p <= 0.0:
        return list(mods)
    kept = [m for m in mods if random.random() > drop_p]
    return kept or [random.choice(list(mods))]


def prefix_weights(K):
    raw = [(j + 1) / K for j in range(K)]
    return [w / sum(raw) for w in raw]


def class_weights(batches, num_classes, device):
    """N / (C * N_c) from the training labels."""
    counts = torch.zeros(num_classes)
    for _, y in batches:
        for lbl in y.view(-1).tolist():
            counts[int(lbl)] += 1
    counts = counts.clamp(min=1.0)
    return (counts.sum() / (num_classes * counts)).to(device)


def train_one_epoch(batches, model, ema, criterion, optimizer, device, mods, drop_p, a):
    model.train()
    loss_sum = n = 0
    for high, y in batches:
        kept = sample_kept(mods, drop_p)
        inputs = {m: high[m].to(device, non_blocking=True) for m in kept}
        y = y.to(device, non_blocking=True)
        if a.prefix_supervision:
            final, prefix, _ = model(inputs, return_prefix=True)          # prefix [B, S, C]
            loss = criterion(final, y)
            K = prefix.shape[1]
            ws = prefix_weights(K)
            ds = sum(ws[j] * criterion(prefix[:, j], y) for j in range(K))
            if ema is not None:
                with torch.no_grad():
                    teach = F.softmax(ema(inputs), dim=-1)
            else:
                teach = F.softmax(final.detach(), dim=-1)
            kd = sum(ws[j] * F.kl_div(F.log_softmax(prefix[:, j], dim=-1), teach, reduction='batchmean')
                     for j in range(K))
            loss = loss + a.prefix_ds_weight * ds + a.prefix_kd_weight * kd
        else:
            loss = criterion(model(inputs), y)
        optimizer.zero_grad()
        loss.backward()
        if a.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.data.mul_(a.ema_decay).add_(p.data, alpha=1.0 - a.ema_decay)
        loss_sum += loss.item() * y.size(0); n += y.size(0)
    return loss_sum / max(n, 1)


@torch.no_grad()
def evaluate(batches, model, device, mods):
    """All modalities in canonical order -> (accuracy, macro-F1)."""
    model.eval()
    ys, ps = [], []
    for high, y in batches:
        ps.append(model({m: high[m].to(device, non_blocking=True) for m in mods}).argmax(1).cpu())
        ys.append(y.view(-1))
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return float((y == p).mean()), float(f1_score(y, p, average='macro'))


def main():
    a = parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    set_seed(a.seed)
    train_b, val_b, test_b, mods, num_classes = build_data(a.fold, a.batch_size, a.num_workers, a.max_seq_len)
    feat_dims = {m: train_b[0][0][m].shape[-1] for m in mods}
    model = SeMA(mods, feat_dims, num_classes, d_model=a.d_model, nhead=a.nhead,
                 num_enc_layers=a.num_enc_layers, num_fusion_layers=a.num_fusion_layers,
                 n_bottlenecks=a.n_bottlenecks, mlp_ratio=a.mlp_ratio, dropout=a.dropout,
                 seq_random_order=a.seq_random_order).to(device)
    print(f'[sema] iemocap fold{a.fold}: enc={a.num_enc_layers} fusion={a.num_fusion_layers} '
          f'K={a.n_bottlenecks} '
          f'params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M', flush=True)

    if a.loss == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights(train_b, num_classes, device),
                                        label_smoothing=a.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.num_epochs, eta_min=1e-6)
    ema = None
    if a.ema_teacher:
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        ema.eval()

    best = {'val_f1': -1.0, 'val_acc': 0.0, 'epoch': -1, 'state': None}
    for epoch in range(a.num_epochs):
        drop_p = curriculum_drop(epoch, a)
        tr_loss = train_one_epoch(train_b, model, ema, criterion, optimizer, device, mods, drop_p, a)
        scheduler.step()
        val_acc, val_f1 = evaluate(val_b, model, device, mods)
        if epoch % 5 == 0 or epoch == a.num_epochs - 1:
            print(f'  ep{epoch:03d} drop={drop_p:.3f} train_loss={tr_loss:.4f} '
                  f'val_acc={val_acc:.4f} val_f1={val_f1:.4f}', flush=True)
        if val_f1 > best['val_f1']:
            best = {'val_f1': val_f1, 'val_acc': val_acc, 'epoch': epoch,
                    'state': copy.deepcopy(model.state_dict())}

    model.load_state_dict(best['state'])
    test_acc, test_f1 = evaluate(test_b, model, device, mods)
    out = a.out_dir
    os.makedirs(out, exist_ok=True)
    torch.save(best['state'], os.path.join(out, f'best_model_fold{a.fold}.pth'))
    result = {
        'dataset': 'iemocap', 'fold': a.fold, 'model_class': 'SeMA',
        'modalities': mods, 'num_classes': num_classes,
        'd_model': a.d_model, 'nhead': a.nhead, 'num_enc_layers': a.num_enc_layers,
        'num_fusion_layers': a.num_fusion_layers, 'n_bottlenecks': a.n_bottlenecks,
        'seq_random_order': bool(a.seq_random_order), 'prefix_supervision': bool(a.prefix_supervision),
        'ema_teacher': bool(a.ema_teacher), 'max_modality_drop': a.max_modality_drop,
        'num_epochs': a.num_epochs, 'batch_size': a.batch_size, 'max_seq_len': a.max_seq_len,
        'lr': a.lr, 'seed': a.seed, 'best_epoch': best['epoch'],
        'best_val_acc': best['val_acc'], 'best_val_f1': best['val_f1'],
        'best_test_acc': test_acc, 'best_test_f1': test_f1,
    }
    json.dump(result, open(os.path.join(out, f'results_fold{a.fold}.json'), 'w'), indent=2)
    print(f'[sema] iemocap fold{a.fold}: val F1 {best["val_f1"]:.4f} (epoch {best["epoch"]}) '
          f'test acc {test_acc:.4f} F1 {test_f1:.4f} -> {out}', flush=True)


if __name__ == '__main__':
    main()

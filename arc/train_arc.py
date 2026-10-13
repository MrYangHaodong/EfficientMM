#!/usr/bin/env python
"""Fit ARC policies on the validation caches.

    python arc/train_arc.py                     # the policies reported in the paper
    python arc/train_arc.py --grid              # every seed x lambda of configs/arc.json
    python arc/train_arc.py --seeds 0 --lams 0.2 1.0

By default each fold trains the (seed, lambda) listed under "deployed" in configs/arc.json.
Writes <arc_dir>/arc_fold<N>_seed<S>_lam<L>.pt (weights, gain tables, meta);
existing files are skipped unless --overwrite.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from arc import policy                        # noqa: E402
from arc.prefix_cache import load_cache       # noqa: E402
from arc.prior import gain_tables             # noqa: E402
from common.paths import config               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folds', type=int, nargs='+', default=None)
    ap.add_argument('--grid', action='store_true', help='train every seed x lambda of configs/arc.json')
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--lams', type=float, nargs='+', default=None)
    ap.add_argument('--cache_dir', default=os.path.join(ROOT, 'cache'))
    ap.add_argument('--arc_dir', default=os.path.join(ROOT, 'checkpoints', 'arc'))
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    cfg = config('arc')
    deployed = cfg['deployed']
    folds = a.folds or [int(f) for f in deployed]
    for fold in folds:
        if a.grid or a.seeds or a.lams:
            runs = [(s, l) for s in (a.seeds or cfg['seeds']) for l in (a.lams or cfg['lambda_grid'])]
        else:
            runs = [(deployed[str(fold)]['seed'], deployed[str(fold)]['lam'])]
        val = load_cache(a.cache_dir, fold, 'val')
        M = val['M']
        tables = gain_tables(val, M)
        for seed, lam in runs:
            path = policy.ckpt_path(a.arc_dir, fold, seed, lam)
            if os.path.exists(path) and not a.overwrite:
                print(f'[arc] exists, skipped: {path}', flush=True); continue
            t0 = time.time()
            net, tables, best_it, ho_f1 = policy.train(val, M, seed, lam, tables, p_train=cfg['p_train'],
                                                       iters=cfg['iters'], epochs=cfg['epochs'], lr=cfg['lr'],
                                                       hidden=cfg['hidden'])
            policy.save(path, net, tables, dict(dataset='iemocap', fold=fold, seed=seed, lam=lam, M=M, C=val['C'],
                                                mods=val['mods'], p_train=cfg['p_train'], iters=cfg['iters'],
                                                epochs=cfg['epochs'], lr=cfg['lr'], best_it=best_it,
                                                holdout_f1=ho_f1))
            print(f'[arc] fold{fold} seed{seed} lam{lam}: best iteration {best_it}, holdout F1 '
                  f'{ho_f1:.4f} ({time.time() - t0:.0f}s) -> {path}', flush=True)


if __name__ == '__main__':
    main()

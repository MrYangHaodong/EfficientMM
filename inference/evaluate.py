#!/usr/bin/env python
"""Evaluate SeMARC (SeMA + ARC) on the IEMOCAP test caches under the deployment rule.

    python inference/evaluate.py                                # full modality availability
    python inference/evaluate.py --avail 1.0 0.9 0.8 0.7 0.6    # + test-time missing modalities

For every fold the policy of configs/arc.json ("deployed") is rolled out on the test cache:
the prior is priced with the policy's own training lambda and the policy stops as soon as
STOP scores at least as high as the best acquisition (no stop margin). ``--avail p`` makes
every test modality available independently with probability p (at least one per sample;
numpy default_rng(1000 + fold), the same draw for every method in the paper).

Reported per fold, then as mean and sample standard deviation over folds:
    f1         test macro-F1
    mu         modality usage, mean over samples of |acquired| / |available|
    gf_fusion  SeMA GFLOPs of the acquired prefix (GF_f in the paper)
    gf_e2e     GF_f + feature-extractor GFLOPs of the acquired modalities (configs/costs.json)
    ext_ms     estimated extraction time of the acquired modalities (idle L40, configs/costs.json)
    freq_<m>   fraction of test samples that acquired modality m
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from arc import core, policy                  # noqa: E402
from arc.prefix_cache import load_cache       # noqa: E402
from arc.prior import gain_tables             # noqa: E402
from common.paths import config               # noqa: E402


def evaluate_fold(fold, seed, lam, p, cache_dir, arc_dir, costs):
    val, test = load_cache(cache_dir, fold, 'val'), load_cache(cache_dir, fold, 'test')
    M, mods = test['M'], test['mods']
    net, tables, meta = policy.load(policy.ckpt_path(arc_dir, fold, seed, lam))
    assert abs(float(meta['lam']) - lam) < 1e-9, (meta['lam'], lam)
    if tables is None:
        tables = gain_tables(val, M)                   # validation statistics only
    N = len(test['y'])
    avail = None if p >= 1.0 else core.sample_availability(N, M, p, np.random.default_rng(1000 + fold))
    f1, k, gf, sn = policy.rollout(net, test, M, tables, lam, avail)
    _, used, _ = core.node_meta(test['nodes'], M)
    acq = np.array([[m in used[s] for m in range(M)] for s in sn], np.float64)          # [N, M]
    n_av = M if avail is None else avail.sum(1)
    g_ext = np.array([costs[m]['extract_gflops'] for m in mods]); t_ext = np.array([costs[m]['extract_ms'] for m in mods])
    row = dict(fold=fold, seed=seed, lam=lam, avail_p=p, f1=f1, mu=float((acq.sum(1) / n_av).mean()), mean_acquired=k, M=M,
               gf_fusion=gf, gf_e2e=gf + float(acq.mean(0) @ g_ext), ext_ms=float((acq @ t_ext).mean()))
    row.update({f'freq_{m}': float(v) for m, v in zip(mods, acq.mean(0))})
    return row, sn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--avail', type=float, nargs='+', default=[1.0], help='availability rate(s) p')
    ap.add_argument('--cache_dir', default=os.path.join(ROOT, 'cache'))
    ap.add_argument('--arc_dir', default=os.path.join(ROOT, 'checkpoints', 'arc'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'semarc_iemocap.csv'))
    ap.add_argument('--save_selections', action='store_true',
                    help="also save each fold's per-sample stop node (<out stem>_p<p>_fold<N>_sn.npy)")
    a = ap.parse_args()
    deployed, costs = config('arc')['deployed'], config('costs')['iemocap']
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    rows = []
    for p in a.avail:
        for fold_s, c in deployed.items():
            row, sn = evaluate_fold(int(fold_s), c['seed'], c['lam'], p, a.cache_dir, a.arc_dir, costs)
            rows.append(row)
            if a.save_selections:
                np.save(f'{os.path.splitext(a.out)[0]}_p{p:g}_fold{fold_s}_sn.npy', sn)
            print(f'  p={p:g} fold{fold_s} (seed {c["seed"]}, lam {c["lam"]}): F1 {row["f1"]:.4f}  MU {row["mu"]:.3f}  '
                  f'GF {row["gf_e2e"]:.2f} ({row["gf_fusion"]:.3f})  extract {row["ext_ms"]:.0f} ms', flush=True)
        d = pd.DataFrame([r for r in rows if r['avail_p'] == p])
        print(f'p={p:g}: F1 {d.f1.mean():.4f} +/- {d.f1.std(ddof=1):.4f}  MU {d.mu.mean():.3f}  '
              f'GF {d.gf_e2e.mean():.2f} ({d.gf_fusion.mean():.3f})  extract {d.ext_ms.mean():.0f} ms\n', flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print('wrote', a.out)


if __name__ == '__main__':
    main()

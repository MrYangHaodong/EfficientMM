#!/usr/bin/env python
"""Train and evaluate the Table 2 controllers at every setting of their trade-off knob, on IEMOCAP.

  SeMARC            ARC (arc/) with modality cost weights w_m = (t_m / min t)^gamma, gamma in GAMMAS: acquiring m costs
                    lam * w_m / sum_{j in A} w_j. lam and seed per fold as deployed (configs/arc.json).
  SeMA + EAMA       value models (seeds 0, 1, 2 per fold); lam chosen on the validation holdout at uniform cost (Table 2's
                    setting, val F1 - 0.2 * val extraction fraction); then costs z_j = t_j^gamma / sum_m t_m^gamma.
  SeMA + Joint-AFA  DQN retrained at each acquisition cost r_cost in RCOSTS (seed 0, 20000 episodes).
  SeMA + GMS        the greedy chain, every budget k.
t_m = extraction time of modality m (configs/costs.json). Inputs: the prefix caches (arc/build_cache.py) and, for EAMA
and Joint-AFA, the input features (IEMOCAP_ROOT), whose time-means are the raw previews the two selectors read.

Writes one row per (method, setting, fold, run) in the format of data/fold_results.csv: test macro-F1, EAMA's mean
number of stages, and freq_j, the share of test samples whose final set contains modality j. --check compares the rows
with the IEMOCAP rows of data/fold_results.csv; `pareto.py --replace <out>` draws the curves with them.

    python analysis/table2_pareto/train_settings.py                          # all four methods, folds 0 1 2
    python analysis/table2_pareto/train_settings.py --methods semarc gms     # a subset
    python analysis/table2_pareto/train_settings.py --check                  # compare an existing output with data/
Runtime per fold on one L40: GMS seconds; SeMARC ~15 s per fit (7 fits); loading the features for the previews ~40 s;
EAMA ~45 s per value model incl. scoring (3); Joint-AFA ~3-4 min per DQN on 2 CPU threads (11; --jafa_workers in parallel).
"""
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from arc import core, policy                                   # noqa: E402
from arc.prefix_cache import load_cache                        # noqa: E402
from arc.prior import gain_tables                              # noqa: E402
from common.paths import config                                # noqa: E402
import controllers as ctl                                      # noqa: E402

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
RCOSTS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
EAMA_SEEDS = [0, 1, 2]
JAFA_SEED = 0
METHODS = {'semarc': 'SeMARC', 'eama': 'SeMA + EAMA', 'jafa': 'SeMA + Joint-AFA', 'gms': 'SeMA + GMS'}


def row(method, value, fold, run, f1, freq, stages=None):
    return dict(method=method, value=f'{value:g}', dataset='iemocap', fold=fold, run=run, f1=float(f1), stages=stages,
                **{f'freq_{j + 1}': float(x) for j, x in enumerate(freq)})


def semarc(val, test, fold, t_ms, dev, log):
    """ARC at every gamma (gamma = 0: the uniform cost of the paper's SeMARC)."""
    cfg = config('arc'); dep = cfg['deployed'][str(fold)]; M = val['M']; tables = gain_tables(val, M); out = []
    for g in GAMMAS:
        w = None if g == 0 else core.cost_weights(t_ms, g)
        t0 = time.time()
        net, _, best_it, _ = policy.train(val, M, dep['seed'], dep['lam'], tables, p_train=cfg['p_train'], iters=cfg['iters'],
                                          epochs=cfg['epochs'], lr=cfg['lr'], hidden=cfg['hidden'], device=dev, weights=w)
        f1, _, _, sn = policy.rollout(net, test, M, tables, dep['lam'], weights=w)
        _, used, _ = core.node_meta(test['nodes'], M)
        freq = np.array([[m in used[s] for m in range(M)] for s in sn], np.float64).mean(0)
        out.append(row('SeMARC', g, fold, dep['seed'], f1, freq))
        log(f'  SeMARC fold {fold} gamma {g:g}: F1 {f1:.4f} (best iteration {best_it}, {time.time() - t0:.0f} s)')
    return out


def gms(val, test, fold, log):
    out = []
    for S, f1 in ctl.gms_chain(val, test, val['M']):
        out.append(row('SeMA + GMS', len(S), fold, None, f1, [1.0 if j in S else 0.0 for j in range(val['M'])]))
    log(f'  GMS fold {fold}: chain {[val["mods"][j] for j in S]}')
    return out


def previews(fold, mods, val, test):
    """Raw previews of val and test, in cache sample order. They are z-scored later, as in the paper's runs: EAMA with
    the statistics of the validation fit split, Joint-AFA with those of the whole validation split."""
    from common.data import build_data
    _tr, vb, tb, mods_d, _C = build_data(fold, config('sema').get('batch_size', 32), 2)
    assert list(mods_d) == list(mods), (mods_d, mods)
    prev = {'val': ctl.raw_previews(vb, mods), 'test': ctl.raw_previews(tb, mods)}
    for s, c, b in (('val', val, vb), ('test', test, tb)):          # the loader's samples are the cache's samples
        assert np.array_equal(np.concatenate([x[1].numpy() for x in b]), c['y']), f'{s}: loader and cache disagree'
    return prev


def eama(val, test, fold, prev, t_ms, dev, log):
    M, mods = val['M'], val['mods']; t_ms = np.asarray(t_ms, np.float64)
    Pv, Pt = ctl.set_readout(val, M), ctl.set_readout(test, M); tr, ho = core.inner_split(len(val['y']))
    fit = {m: prev['val'][m][tr] for m in mods}; std = ctl.Standardizer(fit, mods)
    p_fit, p_ho, p_te = std(fit, mods), std({m: prev['val'][m][ho] for m in mods}, mods), std(prev['test'], mods)
    z_uni = np.full(M, 1.0 / M); out = []
    for seed in EAMA_SEEDS:
        t0 = time.time()
        vm, auc = ctl.eama_train_value(p_fit, Pv[:, tr], val['y'][tr], p_ho, Pv[:, ho], val['y'][ho], M, mods, seed, dev)
        Sv, St = ctl.eama_scores(vm, p_ho, M, mods, dev), ctl.eama_scores(vm, p_te, M, mods, dev)
        score = {}
        for lam in ctl.EAMA_LAMS:                                    # Table 2's lam: validation holdout, uniform cost
            f1v, _, msv, _, _ = ctl.eama_rollout(Sv, Pv[:, ho], val['y'][ho], M, lam, z_uni, t_ms, seed)
            score[lam] = f1v - 0.2 * msv / t_ms.sum()
        lam = max(ctl.EAMA_LAMS, key=lambda l: score[l])
        for g in GAMMAS:
            w = (t_ms / t_ms.min()) ** g; z = w / w.sum()
            f1, _, _, st, freq = ctl.eama_rollout(St, Pt, test['y'], M, lam, z, t_ms, seed)
            out.append(row('SeMA + EAMA', g, fold, seed, f1, freq, stages=float(st)))
        log(f'  EAMA fold {fold} seed {seed}: holdout AUC {auc:.3f}, lam {lam:g} ({time.time() - t0:.0f} s)')
    return out


def _jafa_job(args):
    pv, pt, Pv, Pt, yv, yt, M, mods, r, fold, episodes, threads = args
    torch.set_num_threads(threads)
    t0 = time.time()
    net = ctl.jafa_train(pv, Pv, yv, M, mods, r, JAFA_SEED, 'cpu', episodes=episodes)
    f1, freq = ctl.jafa_rollout(net, pt, Pt, yt, M, mods, 'cpu')
    return row('SeMA + Joint-AFA', r, fold, JAFA_SEED, f1, freq), time.time() - t0


def jafa(val, test, fold, prev, a, log):
    M, mods = val['M'], val['mods']
    std = ctl.Standardizer(prev['val'], mods); pv, pt = std(prev['val'], mods), std(prev['test'], mods)
    Pv, Pt = ctl.set_readout(val, M), ctl.set_readout(test, M)
    jobs = [(pv, pt, Pv, Pt, val['y'], test['y'], M, mods, r, fold, a.jafa_episodes, a.jafa_threads) for r in RCOSTS]
    out = []
    with cf.ProcessPoolExecutor(a.jafa_workers, mp_context=mp.get_context('spawn')) as ex:
        for r_, (res, sec) in zip(RCOSTS, ex.map(_jafa_job, jobs)):
            out.append(res); log(f'  Joint-AFA fold {fold} r_cost {r_:g}: F1 {res["f1"]:.4f} ({sec:.0f} s)')
    return out


def check(path):
    """Max |difference| of each method's rows from the IEMOCAP rows of data/fold_results.csv."""
    new = pd.read_csv(path, dtype={'value': str}); ref = pd.read_csv(os.path.join(HERE, 'data', 'fold_results.csv'), dtype={'value': str})
    ref = ref[ref.dataset == 'iemocap']; key = ['method', 'value', 'fold', 'run']
    for d in (new, ref):
        d['run'] = d['run'].fillna(-1).astype(int); d['value'] = d['value'].astype(float).map(lambda v: f'{v:g}')
    m = ref.merge(new, on=key, suffixes=('_ref', '_new'))
    cols = ['f1', 'stages'] + [c for c in new.columns if c.startswith('freq_')]        # IEMOCAP: freq_1 .. freq_6
    for meth, g in m.groupby('method'):
        n_ref = (ref.method == meth).sum()
        diff = max(float(np.nanmax(np.abs(g[f'{c}_new'] - g[f'{c}_ref']))) if g[f'{c}_ref'].notna().any() else 0.0 for c in cols)
        print(f'  {meth:17s} {len(g)}/{n_ref} rows matched, max |difference| {diff:.2e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', nargs='+', choices=list(METHODS), default=list(METHODS))
    ap.add_argument('--folds', type=int, nargs='+', default=None, help='default: the folds of configs/arc.json')
    ap.add_argument('--cache_dir', default=os.path.join(ROOT, 'cache'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'fold_results_iemocap.csv'))
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='SeMARC and EAMA')
    ap.add_argument('--jafa_episodes', type=int, default=20000)
    ap.add_argument('--jafa_threads', type=int, default=2, help='CPU threads per Joint-AFA DQN')
    ap.add_argument('--jafa_workers', type=int, default=4, help='Joint-AFA DQNs trained in parallel')
    ap.add_argument('--check', action='store_true', help='only compare --out with data/fold_results.csv')
    a = ap.parse_args()
    if a.check:
        check(a.out); return
    log = lambda s: print(s, flush=True)                             # noqa: E731
    costs = config('costs')['iemocap']; rows = []
    for fold in a.folds or [int(f) for f in config('arc')['deployed']]:
        val, test = load_cache(a.cache_dir, fold, 'val'), load_cache(a.cache_dir, fold, 'test')
        t_ms = [costs[m]['extract_ms'] for m in val['mods']]
        if 'gms' in a.methods:
            rows += gms(val, test, fold, log)
        if 'semarc' in a.methods:
            rows += semarc(val, test, fold, t_ms, a.device, log)
        if 'eama' in a.methods or 'jafa' in a.methods:
            prev = previews(fold, val['mods'], val, test)
            if 'eama' in a.methods:
                rows += eama(val, test, fold, prev, t_ms, a.device, log)
            if 'jafa' in a.methods:
                rows += jafa(val, test, fold, prev, a, log)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)                # after every fold
    log(f'wrote {len(rows)} rows -> {a.out}')
    check(a.out)


if __name__ == '__main__':
    main()

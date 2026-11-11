#!/usr/bin/env python
"""SeMA + EAMA with lambda held at Table 2's value and the modality costs rescaled exactly as SeMARC's are by gamma.

SeMARC prices modality j at lambda * w_j / sum_m w_m with w_m = (t_m / min t)^gamma (rl_core normpow, idle-L40 times):
at gamma = 0 that is lambda / M, so gamma multiplies modality j's price by M w_j / sum_m w_m. EAMA's uniform cost
z_j = 1/M (Table 2's row) multiplied by the same factor is z_j(gamma) = w_j / sum_m w_m = t_j^gamma / sum_m t_m^gamma
(gamma = 0: Table 2's uniform costs; gamma = 1: the weighted row's costs t_j / sum t). Lambda stays at Table 2's: each
run's (fold, value-model seed) lambda selected on validation at uniform cost (results_baselines/eama_raw_<ds>_idle_f*.csv).

The value models are not stored, so each (fold, seed) is retrained with baseline_eama_raw.py's own functions
(train_value, scores_all, run_policy, readout, Standardizer; unchanged, hybrid optimizer, no context, unlimited stages)
on the raw previews of val and test saved by ctrl/jafa_sweep.py (the same raw_previews call as the paper's run).
Check first: the retrained models must reproduce every stored test row of the paper's run (both costs, all nine
lambdas: F1, MU, extraction ms, #stages) and its selected lambda for both costs; a mismatch stops the script.

    ctrl/envrun.sh <ds> <2|3> ctrl/eama_gamma.py      env: FOLDS (optional), GAMMAS (optional)
Writes ctrl/eama_gamma/eama_gamma_<ds>_f<fold>.json.
"""
import glob
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

RLB = '/files1/haodong/test/new/RL_AAAI_branch'
sys.path.insert(0, RLB)
import environment as env                                                              # noqa: E402
import rl_core as rl                                                                   # noqa: E402
from baseline_eama import LAMS                                                         # noqa: E402
from baseline_eama_raw import train_value, scores_all, run_policy, readout, Standardizer  # noqa: E402

C = '/files1/haodong/test/new/table2_pareto/ctrl'; PRE = f'{C}/previews'; OUT = f'{C}/eama_gamma'
os.makedirs(OUT, exist_ok=True)
ds = os.environ['DS']
GAMMAS = [float(g) for g in os.environ.get('GAMMAS', '0,0.25,0.5,0.75,1,1.5,2').split(',')]
assert 'idle' in os.path.basename(rl.COST_FILE), rl.COST_FILE
T = json.load(open(rl.COST_FILE))[ds]['per_modality']
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
paper = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{RLB}/results_baselines/eama_raw_{ds}_idle_f*.csv'))])
paper = paper[paper.optimizer == 'hybrid']
folds = [int(f) for f in os.environ['FOLDS'].split(',')] if os.environ.get('FOLDS') else env.FOLDS[ds]
TOL = 1e-9


def policy(S, P, y, M, lam, z, t_ms, seed):
    return run_policy(S, P, y, M, lam, z, t_ms, 'hybrid', np.random.default_rng(seed), 0, 0)


for fold in folds:
    val, test = env.get_cache('val', ds, fold), env.get_cache('test', ds, fold); M = val['M']; mods = list(val['mods'])
    t_ms = np.array([float(T[m]['ms']) for m in mods]); z_ext = t_ms / t_ms.sum(); z_uni = np.full(M, 1.0 / M)
    pz = np.load(f'{PRE}/{ds}_f{fold}.npz'); assert list(pz['mods']) == mods, (list(pz['mods']), mods)
    prev = {s: {m: pz[f'{s}__{m}'] for m in mods} for s in ('val', 'test')}
    for s, c in (('val', val), ('test', test)):
        assert prev[s][mods[0]].shape[0] == len(c['y']), s
    Pv, Pt = readout(val, M, 'orderavg'), readout(test, M, 'orderavg'); tr, ho = rl.inner_split(len(val['y']))
    Pfit, yfit, prev_fit = Pv[:, tr], val['y'][tr], {m: prev['val'][m][tr] for m in mods}
    std = Standardizer(prev_fit, mods); prev_fit = std(prev_fit, mods)
    prev_ho = std({m: prev['val'][m][ho] for m in mods}, mods); prev_te = std(prev['test'], mods)
    out = dict(dataset=ds, fold=fold, mods=mods, t_ms=t_ms.tolist(), gammas=GAMMAS, device=dev, seeds={})
    for seed in (0, 1, 2):
        t0 = time.time()
        vm, auc = train_value(prev_fit, Pfit, yfit, prev_ho, Pv[:, ho], val['y'][ho], M, mods, 0, seed, dev, epochs=60)
        Sv, St = scores_all(vm, prev_ho, M, mods, 0, dev), scores_all(vm, prev_te, M, mods, 0, dev)
        ref = paper[(paper.fold == fold) & (paper.seed == seed)]
        worst, sel = float(np.abs(ref.value_auc - auc).max()), {}                   # same value model: same holdout AUC
        for cost_name, z in (('extract', z_ext), ('uniform', z_uni)):             # reproduce the paper's run first
            grid = {}
            for lam in LAMS:
                f1v, _, msv, _, _, _ = policy(Sv, Pv[:, ho], val['y'][ho], M, lam, z, t_ms, seed)
                grid[lam] = (f1v, msv / t_ms.sum())
            sel[cost_name] = max(LAMS, key=lambda l: grid[l][0] - 0.2 * grid[l][1])
            for lam in LAMS:
                f1, mu, ms, st, _, _ = policy(St, Pt, test['y'], M, lam, z, t_ms, seed)
                r = ref[(ref.cost == cost_name) & np.isclose(ref.lam, lam)]
                assert len(r) == 1, (cost_name, lam, len(r))
                r = r.iloc[0]
                worst = max(worst, abs(f1 - r.test_f1), abs(mu - r.test_mu), abs(ms - r.test_ms) / max(r.test_ms, 1e-9),
                            abs(st - r.stages))
                assert bool(r.selected) == (lam == sel[cost_name]), (ds, fold, seed, cost_name, lam, sel[cost_name])
        assert worst <= TOL, f'{ds} fold {fold} seed {seed}: retrained value model differs from the paper run by {worst:g}'
        lam_t2 = sel['uniform']                                                      # Table 2's lambda for this run
        res = {}
        for g in GAMMAS:
            w = (t_ms / t_ms.min()) ** g; z = w / w.sum()                            # = SeMARC's normalised weights
            f1, mu, ms, st, freq, _ = policy(St, Pt, test['y'], M, lam_t2, z, t_ms, seed)
            f1v, muv, msv, stv, _, _ = policy(Sv, Pv[:, ho], val['y'][ho], M, lam_t2, z, t_ms, seed)
            res[f'{g:g}'] = dict(z=z.tolist(), f1=f1, mu=mu, ext_ms=ms, stages=st, freq=freq.tolist(),
                                 val_f1=f1v, val_mu=muv, val_ext_ms=msv, val_stages=stv)
        out['seeds'][str(seed)] = dict(value_auc=auc, lam_table2=lam_t2, lam_weighted_row=sel['extract'],
                                       check_max_abs_diff=worst, gamma=res, seconds=round(time.time() - t0, 1))
        print(f'{ds} fold {fold} seed {seed}: reproduces the paper (max diff {worst:.1e}), λ = {lam_t2:g}; '
              + ' '.join(f'γ{k}: {v["f1"]:.4f}/{v["ext_ms"]:.0f}ms' for k, v in res.items())
              + f'  ({time.time() - t0:.0f} s)', flush=True)
    p = f'{OUT}/eama_gamma_{ds}_f{fold}.json'
    json.dump(out, open(p + '.tmp', 'w'), indent=1)
    os.replace(p + '.tmp', p)
print('EAMA_GAMMA_DONE', ds, flush=True)

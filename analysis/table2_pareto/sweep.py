#!/usr/bin/env python
"""SeMARC lambda x gamma sweep for the Table 2 Pareto plot (2026-11-09).

One process = one dataset and one cost-weight exponent gamma; it loops over the lambda grid and the folds. Everything else
is the deployed policy's setting, fold by fold (deployed_provenance.json[PROV_KEY]): per-fold ARC seed, FQI schedule,
availability augmentation, and the paper's decision rule (run-time prior priced at the training lambda, stop margin 0).
This is gamma_ablation_ds.py (Appendix "Complete Results for Figure diag-reward") with a lambda grid instead of the deployed
lambda, and only the 'samelam' rule.

Cost weights w_m = (t_m / min t)^gamma from the idle-L40 extraction times (RLB_COST_MODE=normpow, RLB_ABS_GAMMA,
RLB_COST_FILE=..._idle.json); gamma = 0 runs RLB_COST_MODE=uniform, so at the deployed lambda it loads the deployed nets.
Latency = the paper's estimate (Table 2 / Appendix comp-replace, as in gamma_ablation_ds.py): idle-L40 extraction of every
acquired modality + calibrated branch-encoder, head and selector times.

Nets live in RL_AAAI_branch/checkpoints_t2p/<ds>/ only (seeded with copies of the existing nets, see seed_runset.sh).
A missing net is fitted on the visible GPU by the unchanged train_or_load.

    env: DS, PROV_KEY, LAMS (comma list; 'dep' = deployed per-fold lambda), OUT (csv), SN_DIR,
         RLB_RUNSET=t2p RLB_CACHE_SUBDIR RLB_CKPT_OVERRIDE RLB_PROV_KEY [RLB_FOLDS] + the cost env above
"""
import csv
import json
import os
import sys
import time

H = '/files1/haodong/test/new/RL_AAAI_branch'
sys.path.insert(0, H)
os.chdir(H)
import numpy as np
import torch
import environment as env
import rl_core as rl
import dqn_policy_qprior as QPR
from stop_addon import q_values, walk

assert QPR.CHECKPOINT_DIR == os.path.join(H, 'checkpoints_t2p'), QPR.CHECKPOINT_DIR      # never a shipped folder
ds = os.environ['DS']
PK = os.environ.get('PROV_KEY', ds)
PROV = json.load(open('deployed_provenance.json'))[PK]
EA = '/files1/haodong/test/new/energy_adaptive'
T_IDLE = json.load(open(f'{EA}/results_per_modality_extract_cuda_idle.json'))[ds]['per_modality']
FW = json.load(open(f'{EA}/results_fusion_walk_idle_{ds}.json'))[ds]
F0 = next(iter(FW.values()))
t_enc = {m: float(np.mean([f['t_enc_ms'][m] for f in FW.values()])) for m in F0['t_enc_ms']}
t_head = float(np.mean([f['t_head_ms'] for f in FW.values()]))
t_pol = float(np.mean([f['t_pol_ms'] for f in FW.values()]))
cal = float(np.mean([f['walk_direct']['ms'] / f['walk_model_same_samples_ms'] for f in FW.values()]))
if rl.COST_MODE == 'normpow':
    assert 'idle' in os.path.basename(rl.COST_FILE), rl.COST_FILE
    g = rl.ABS_GAMMA
else:
    assert rl.COST_MODE == 'uniform', rl.COST_MODE
    g = 0.0
LAMS = os.environ['LAMS'].split(',')
SN = os.environ['SN_DIR']
os.makedirs(SN, exist_ok=True)
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
rows = []
for lam_s in LAMS:
    for fs, cf in sorted(PROV['per_fold'].items(), key=lambda kv: int(kv[0])):
        f = int(fs)
        lam = float(cf['lam']) if lam_s == 'dep' else float(lam_s)
        val = env.get_cache('val', ds, f)
        test = env.get_cache('test', ds, f)
        M = val['M']
        mods = list(val['mods'])
        md = env.depth_of(ds)
        rl.set_cost_weights(ds, mods, fold=f)
        vv = QPR._default_vv(val, M, md)
        path = QPR._ckpt_path(ds, f, cf['seed'], lam, QPR.P_TRAIN, QPR.RESAMPLE)
        existed = os.path.exists(path)
        t0 = time.time()
        net, tabs = QPR.train_or_load(val, M, md, dataset=ds, fold=f, seed=cf['seed'], lam=lam, p_train=QPR.P_TRAIN,
                                      resample=QPR.RESAMPLE, tables=vv, force_retrain=False)
        t_fit = time.time() - t0
        Sf = rl.build_state(test, M, md, None)
        depth, used, ch = rl.node_meta(test['nodes'], M, md)
        N = test['softmax'].shape[1]
        Qp = QPR.q_prior(test, M, md, tabs, lam)                                     # prior priced at the training lambda
        Q = q_values(net, test, M, md, Sf, Qp, 0.0).copy()                           # stop margin 0
        f1, mu, gf, sn = walk(Q, test, M, md, np.full((len(used), N), -np.inf, np.float32), np.inf)
        freq = np.array([np.mean([m in used[s] for s in sn]) for m in range(M)])
        Ek = float(freq.sum())
        ext = float(sum(fr * T_IDLE[m]['ms'] for fr, m in zip(freq, mods)))
        fus = cal * (float(sum(fr * t_enc[m] for fr, m in zip(freq, mods))) + Ek * t_head)
        pol = cal * (Ek + 1) * t_pol
        np.save(f'{SN}/{ds}_g{g:g}_lam{lam_s}_fold{f}_sn.npy', np.asarray(sn))
        w = rl._COST_W if rl._COST_W is not None else np.ones(M)
        rows.append(dict(dataset=ds, gamma=g, lam=lam_s, lam_train=lam, fold=f, seed=cf['seed'], f1=f1, mu=mu, gflops=gf,
                         extract_ms=ext, fusion_ms=fus, policy_ms=pol, latency_ms=ext + fus + pol, n_acq=Ek,
                         net=os.path.basename(path), net_existed=int(existed), fit_s=round(t_fit, 1), device=dev,
                         freq=json.dumps(dict(zip(mods, np.round(freq, 4).tolist()))),
                         weights=json.dumps(dict(zip(mods, np.round(w, 4).tolist())))))
        print('%s gamma %g lam %s fold%d (%s, %.0fs): F1 %.4f MU %.3f lat %.1f ms' % (
            ds, g, lam_s, f, 'loaded' if existed else 'FITTED', t_fit, f1, mu, ext + fus + pol), flush=True)
with open(os.environ['OUT'], 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print('WROTE', os.environ['OUT'], len(rows), 'rows', flush=True)

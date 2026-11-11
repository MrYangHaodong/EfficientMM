#!/usr/bin/env python
"""Joint-AFA (pre-backbone, Table 2's SeMA + Joint-AFA row) retrained at several per-acquisition costs r_cost.

The paper's row is RL_AAAI_branch/baseline_jafa_raw.py at r_cost = 0.05, seed 0, 20000 episodes, trained on the
validation split, evaluated on test (results_baselines/jafa_raw_<ds>_idle_rcost0.05_sets.json). This wrapper calls that
file's own train() and evaluate() unchanged (same JAFARaw net, DQN schedule, reward, seed, raw previews, set readout);
only r_cost differs. Nothing is written under RL_AAAI_branch.

Two stages, because the data pipeline (environment._load_model) loads onto cuda:0 while training does not need a GPU:
  MODE=previews  (GPU 2 or 3, one process at a time): raw previews of val and test, exactly as baseline_jafa_raw.py
                 computes them, saved to ctrl/previews/<ds>_f<fold>.npz
  MODE=train     (DEV=cpu or a GPU): standardise, read the prefix cache, train every r_cost in RCOSTS, evaluate on test;
                 ctrl/jafa/jafa_<ds>_f<fold>_r<r>_s<seed>_<dev>.json and ctrl/jafa/nets/*.pt
    env DS=<ds> FOLDS=0,1,2 RCOSTS=0,0.01 MODE=train DEV=cpu python jafa_sweep.py    (environment: envrun.sh)
"""
import json, os, sys, time
import numpy as np, torch
RLB = '/files1/haodong/test/new/RL_AAAI_branch'
sys.path.insert(0, RLB)
import environment as env, rl_core as rl                                   # noqa: E402
from baseline_jafa_raw import train, evaluate                               # noqa: E402
from baseline_eama_raw import raw_previews, Standardizer, readout           # noqa: E402

C = '/files1/haodong/test/new/table2_pareto/ctrl'; OUT = f'{C}/jafa'; PRE = f'{C}/previews'
ds = os.environ['DS']; folds = [int(x) for x in os.environ['FOLDS'].split(',')]
mode = os.environ.get('MODE', 'train')
os.makedirs(f'{OUT}/nets', exist_ok=True); os.makedirs(PRE, exist_ok=True)

if mode == 'previews':
    for fold in folds:
        p = f'{PRE}/{ds}_f{fold}.npz'
        if os.path.exists(p):
            continue
        t0 = time.time(); val, test = env.get_cache('val', ds, fold), env.get_cache('test', ds, fold); mods = list(val['mods'])
        _model, splits, mods_chk, _C = env._load_model(ds, fold); assert list(mods_chk) == mods
        prev = {s: raw_previews(splits[s], mods) for s in ('val', 'test')}; del _model, splits; torch.cuda.empty_cache()
        for s, c in (('val', val), ('test', test)):
            assert prev[s][mods[0]].shape[0] == len(c['y']), s
        np.savez(p + '.tmp.npz', mods=np.array(mods), **{f'{s}__{m}': prev[s][m] for s in prev for m in mods})
        os.replace(p + '.tmp.npz', p)
        print(f'  [{ds}] fold{fold}: previews of val {len(val["y"])} / test {len(test["y"])} in {time.time() - t0:.0f} s', flush=True)
    print('PREVIEWS_DONE', ds, folds, flush=True)
    sys.exit(0)

rcosts = [float(x) for x in os.environ['RCOSTS'].split(',')]
episodes = int(os.environ.get('EPISODES', '20000')); seed = int(os.environ.get('SEED', '0'))
devname = os.environ.get('DEV', 'cpu'); dev = 'cpu' if devname == 'cpu' else 'cuda:0'
if devname == 'cpu':
    torch.set_num_threads(int(os.environ.get('THREADS', '2')))
T = json.load(open(rl.COST_FILE))[ds]['per_modality']


def path(fold, r):
    return f'{OUT}/jafa_{ds}_f{fold}_r{r:g}_s{seed}_{devname}.json'


for fold in folds:
    todo = [r for r in rcosts if not os.path.exists(path(fold, r))]
    if not todo:
        continue
    val, test = env.get_cache('val', ds, fold), env.get_cache('test', ds, fold); M = val['M']; mods = list(val['mods'])
    t_ms = np.array([float(T[m]['ms']) for m in mods])
    z = np.load(f'{PRE}/{ds}_f{fold}.npz'); assert list(z['mods']) == mods
    prev = {s: {m: z[f'{s}__{m}'] for m in mods} for s in ('val', 'test')}
    std = Standardizer(prev['val'], mods); pv, pt = std(prev['val'], mods), std(prev['test'], mods)
    Pv, Pt = readout(val, M, 'orderavg'), readout(test, M, 'orderavg')
    print(f'\n  [{ds}] fold{fold}: M={M}, val {len(val["y"])}, test {len(test["y"])}, {devname}; r_cost {todo}', flush=True)
    for r in todo:
        t1 = time.time()
        net = train(pv, Pv, val['y'], M, mods, r, seed, dev, episodes=episodes)
        res = evaluate(net, pt, Pt, test['y'], M, mods, t_ms, dev)
        rv = evaluate(net, pv, Pv, val['y'], M, mods, t_ms, dev); rv.pop('sets')
        res.update(val=rv, mods=mods, dataset=ds, fold=fold, r_cost=r, seed=seed, episodes=episodes, readout='orderavg',
                   device=devname, train_s=round(time.time() - t1, 1))
        torch.save(net.state_dict(), f'{OUT}/nets/jafa_{ds}_f{fold}_r{r:g}_s{seed}_{devname}.pt')
        json.dump(res, open(path(fold, r) + '.tmp', 'w')); os.replace(path(fold, r) + '.tmp', path(fold, r))
        print(f'    r_cost {r:g}: test F1 {res["f1"]:.4f} MU {res["mu"]:.3f} extract {res["extract_ms"]:.1f} ms '
              f'freq {np.round(res["freq"], 2).tolist()} (val F1 {rv["f1"]:.4f} MU {rv["mu"]:.3f}) {res["train_s"]:.0f} s', flush=True)
print('ALL_DONE', ds, folds, flush=True)

#!/usr/bin/env python
"""Is this environment's prefix cache the one the paper's pre-backbone runs read? Re-reads the per-sample final sets stored
by the paper's Joint-AFA run (results_baselines/jafa_raw_<ds>_idle_rcost0.05_sets.json) through this cache's order-averaged
set posterior and compares the test F1 with the stored one; also re-scores GMS's stored sets at every budget k."""
import json, os, sys
import numpy as np
from sklearn.metrics import f1_score
RLB = '/files1/haodong/test/new/RL_AAAI_branch'
sys.path.insert(0, RLB)
import environment as env                                                  # noqa: E402
from baseline_eama_raw import readout                                      # noqa: E402
from baseline_greedy_submodular import _node_index, _posterior             # noqa: E402

ds = os.environ['DS']; B = f'{RLB}/results_baselines'
GMS = {'iemocap': 'greedy_submodular_iemocap.json', 'mmfi': 'greedy_submodular_mmfi_rebal.json',
       'czu_mhad': 'greedy_submodular_czu_mhad.json', 'eav': 'greedy_submodular_eav.json',
       'cmi': 'greedy_submodular_cmi.json', 'utd_mhad': 'greedy_submodular_utd_mhad_rebal422.json'}[ds]
j = json.load(open(f'{B}/jafa_raw_{ds}_idle_rcost0.05_sets.json'))['per_fold']
g = json.load(open(f'{B}/{GMS}'))['per_fold']
worst = 0.0
for f in sorted(j, key=int):
    test = env.get_cache('test', ds, int(f)); M = test['M']; mods = list(test['mods'])
    P = readout(test, M, 'orderavg'); sets = j[f]['sets']; assert len(sets) == len(test['y'])
    mask = np.array([sum(1 << int(m) for m in S) for S in sets])
    f1 = f1_score(test['y'], P[mask, np.arange(len(mask))].argmax(-1), average='macro')
    bs = _node_index(test); gd = []
    for e in g[f]:
        S = [mods.index(m) for m in e['selected']]
        gd.append(abs(f1_score(test['y'], _posterior(test, bs, S, 'avg', None).argmax(-1), average='macro') - e['test_f1']))
    worst = max(worst, abs(f1 - j[f]['f1']), max(gd))
    print(f'{ds} fold{f}: JAFA stored F1 {j[f]["f1"]:.4f}, re-read {f1:.4f} | GMS max |dF1| over k {max(gd):.5f}', flush=True)
print(f'CHECK {ds} worst |dF1| {worst:.5f}', 'OK' if worst < 6e-5 else 'MISMATCH')

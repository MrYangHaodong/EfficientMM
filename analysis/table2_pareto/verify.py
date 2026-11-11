#!/usr/bin/env python
"""Check the sweep against the paper's own per-fold results before any new net is fitted:
  * every dataset and gamma at the deployed lambda vs results_baselines/gamma_ablation_samelam_perfold.csv (rule samelam,
    the source of Appendix tab:wgamma-alldata; gamma = 0 is Table 2's SeMARC);
  * IEMOCAP, every (gamma, lambda) vs results_baselines/gammaabl_grid_v2/*.csv (Appendix tab:wgamma-lambda).
Prints the largest |difference| in F1, MU and latency per comparison."""
import glob
import os
import pandas as pd

T = '/files1/haodong/test/new/table2_pareto'
B = '/files1/haodong/test/new/RL_AAAI_branch/results_baselines'
mine = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{T}/res/verify_*.csv'))])
ref = pd.read_csv(f'{B}/gamma_ablation_samelam_perfold.csv').query("rule == 'samelam'")
dep = mine[mine.lam == 'dep'].merge(ref, on=['dataset', 'gamma', 'fold'], suffixes=('', '_ref'))
assert (dep.seed == dep.seed_ref).all() and (dep.lam_train == dep.lam_train_ref).all()
print(f'deployed lambda vs gamma_ablation_samelam_perfold.csv: {len(dep)} fold rows '
      f'(expected {6 * 7 * 3}), nets loaded not fitted: {dep.net_existed.all()}')
for k in ('f1', 'mu', 'latency_ms'):
    print(f'  max |d {k}| = {(dep[k] - dep[k + "_ref"]).abs().max():.2e}')
grid = pd.concat([pd.read_csv(f) for f in glob.glob(f'{B}/gammaabl_grid_v2/*.csv')])
ie = mine[(mine.dataset == 'iemocap') & (mine.lam != 'dep')].copy()
ie['lam_train'] = ie.lam_train.astype(float)
m = ie.merge(grid, on=['gamma', 'lam_train', 'fold'], suffixes=('', '_ref'))
print(f'IEMOCAP grid vs gammaabl_grid_v2: {len(m)} fold rows (expected {7 * 6 * 3}), nets loaded: {m.net_existed.all()}')
for k in ('f1', 'mu', 'latency_ms'):
    print(f'  max |d {k}| = {(m[k] - m[k + "_ref"]).abs().max():.2e}')

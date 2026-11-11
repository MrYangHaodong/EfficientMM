#!/usr/bin/env python
"""Seed the sweep's run set RL_AAAI_branch/checkpoints_t2p/<ds>/ with COPIES of the nets the paper already reports, so the
sweep reproduces them exactly and never writes into a shipped folder:
  * every dataset, every fold at its deployed (seed, lambda): the deployed uniform net (gamma = 0, Table 2's SeMARC) and the
    six normpow nets of the gamma study (Appendix tab:wgamma-alldata);
  * IEMOCAP: the whole gamma x lambda grid of Appendix tab:wgamma-lambda (uniform and normpow, lambda 0.05 ... 2).
Copies keep their timestamps. An existing target with different content stops the script.
"""
import filecmp
import json
import os
import shutil

H = '/files1/haodong/test/new/RL_AAAI_branch'
P = json.load(open(f'{H}/deployed_provenance.json'))
SHIPPED = {'iemocap': ('k16', 'iemocap'), 'cmi': ('k16', 'cmi'), 'czu_mhad': ('k16', 'czu_mhad'), 'eav': ('k16p', 'eav'),
           'mmfi': ('mmfi_rebal', 'mmfi@rebal'), 'utd_mhad': ('utd_mhad_dropaug', 'utd_mhad@rebal422')}
GAMMAS = ['0.25', '0.5', '0.75', '1', '1.5', '2']
GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
n_new = n_same = 0
for ds, (rs, pk) in SHIPPED.items():
    src_dir = f'{H}/checkpoints_{rs}/{ds}'
    dst_dir = f'{H}/checkpoints_t2p/{ds}'
    os.makedirs(dst_dir, exist_ok=True)
    for f, cf in P[pk]['per_fold'].items():
        lams = GRID if ds == 'iemocap' else [float(cf['lam'])]
        for lam in lams:
            base = f"qprior_fold{f}_seed{cf['seed']}_lam{lam:.3f}_p0.70_r1"
            for name in [base + '.pt'] + [f'{base}_cnormpow_g{g}_idle.pt' for g in GAMMAS]:
                src, dst = f'{src_dir}/{name}', f'{dst_dir}/{name}'
                assert os.path.exists(src), src
                if os.path.exists(dst):
                    assert filecmp.cmp(src, dst, shallow=False), f'different content already at {dst}'
                    n_same += 1
                    continue
                shutil.copy2(src, dst)
                n_new += 1
print(f'copied {n_new} nets, {n_same} already present; run set {H}/checkpoints_t2p')

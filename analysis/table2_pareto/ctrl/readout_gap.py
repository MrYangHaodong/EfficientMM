#!/usr/bin/env python
"""Full-set test F1 under the two readouts: order-averaged posterior over every cached ordering of the set (how the
pre-backbone EAMA / Joint-AFA / GMS rows read a set) vs the single val-LOO order (the backbone alone, as SeMARC walks it)."""
import os, sys
import numpy as np
from sklearn.metrics import f1_score
RLB = '/files1/haodong/test/new/RL_AAAI_branch'
sys.path.insert(0, RLB)
import environment as env                                                  # noqa: E402
from baseline_eama_raw import readout                                      # noqa: E402

ds = os.environ['DS']; r = []
for f in env.FOLDS[ds]:
    t = env.get_cache('test', ds, f); M = t['M']; full = (1 << M) - 1
    a = f1_score(t['y'], readout(t, M, 'orderavg')[full].argmax(-1), average='macro')
    c = f1_score(t['y'], readout(t, M, 'canonical')[full].argmax(-1), average='macro')
    r.append((a, c))
a, c = np.mean(r, 0)
print(f'GAP {ds} full set: order-averaged {a:.4f}  val-LOO order {c:.4f}  diff {a - c:+.4f}')

"""ARC: residual fitted-Q acquisition controller.

    Q_theta(psi, a) = Q_0(psi, a; lam) + R_theta(psi, a)

Q_0 is the prior of arc/prior.py; R_theta is a small MLP whose output layer starts at
zero, so the untrained policy is exactly the prior. R_theta is fitted by Fitted-Q
Iteration on the validation cache (80% fit / 20% holdout):

    target(psi, STOP)      = r(psi)                               (class-balanced correctness)
    target(psi, acquire m) = -lam / |A| + Q_target(psi', a*),     a* = argmax_legal Q_online(psi', .)

with availability redrawn (Bernoulli(p_train)) at every iteration, R_theta regressed onto
(target - Q_0) with a masked squared error, and the iteration with the best holdout
macro-F1 kept. The holdout is scored with the rule the policy is used with: Q_0 priced at
the same lam the policy is fitted with, greedy acquisition, and STOP as soon as
Q_theta(psi, STOP) >= max_m Q_theta(psi, acquire m) (ties go to STOP, no stop margin).
"""
import os

import numpy as np
import torch
import torch.nn as nn

from arc import core
from arc.prior import q_prior, gain_tables

CHUNK = 1_000_000          # rows per forward/backward chunk (CMI/CZU trees have ~10M rows)


class ResidualQ(nn.Module):
    """state -> residual Q-value of each of the M+1 actions; zero at initialisation."""

    def __init__(self, fdim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fdim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def _fwd(net, S):
    if S.shape[0] <= CHUNK:
        return net(S)
    return torch.cat([net(S[i:i + CHUNK]) for i in range(0, S.shape[0], CHUNK)])


def _fit_epoch(net, opt, S, T, mask, denom):
    """One full-batch step of the masked squared error, accumulated over chunks."""
    opt.zero_grad()
    for i in range(0, S.shape[0], CHUNK):
        j = min(i + CHUNK, S.shape[0])
        loss = (((net(S[i:j]) - T[i:j]) ** 2) * mask[i:j]).sum() / denom
        loss.backward()
    opt.step()


@torch.no_grad()
def _walk(net, cache, M, Sf, Qp, avail=None):
    """Greedy walk of every sample down the tree -> (macro-F1, mean #acquired, mean GFLOPs, stop node)."""
    nodes = cache['nodes']; depth, used, ch = core.node_meta(nodes, M)
    n, N, fdim = Sf.shape
    dev = next(net.parameters()).device
    Q = _fwd(net, torch.tensor(Sf).reshape(n * N, fdim).to(dev)).reshape(n, N, M + 1).cpu().numpy() + Qp
    mk = core.action_mask(n, M, depth, ch, avail, N)
    cur = np.zeros(N, int); stopped = np.zeros(N, bool); sn = cur.copy()
    for _ in range(M + 1):
        for i in np.unique(cur[~stopped]):
            here = (cur == i) & (~stopped)
            if not ch[i]:                                           # every modality acquired
                sn[here] = i; stopped |= here; continue
            Qa = np.where(mk[i][:, :M], Q[i][:, :M], -1e9).max(-1)
            fin = here & ((~mk[i].any(-1)) | (mk[i][:, M] & (Q[i][:, M] >= Qa)))
            sn[fin] = i; stopped |= fin
            go = here & (~fin)
            if not go.any():
                continue
            a = np.where(mk[i], Q[i], -1e9).argmax(-1)
            for (m, c) in ch[i]:
                cur[go & (a == m)] = c
        if stopped.all():
            break
    sn[~stopped] = cur[~stopped]
    preds = cache['softmax'][sn, np.arange(N)].argmax(-1)
    return core.macro_f1(cache['y'], preds), float(depth[sn].mean()), float(cache['gflops'][sn].mean()), sn


def train(val, M, seed, lam, tables=None, p_train=0.7, iters=8, epochs=40, lr=1e-3, hidden=64, device=None):
    """Fit ARC on a validation cache. Returns (net, tables, best_iteration, holdout_f1)."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng_av = np.random.default_rng(7000 + seed)
    dev = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    depth, used, ch = core.node_meta(val['nodes'], M)
    n = len(val['nodes']); N = val['softmax'].shape[1]; A = M + 1
    tr, ho = core.inner_split(N)
    tables = tables if tables is not None else gain_tables(val, M)
    avail = core.sample_availability(N, M, p_train, rng_av)
    R = core.correctness(val)
    child = torch.tensor(core.child_table(ch, n, M)).to(dev); legal = child >= 0

    def tensors(avail):
        Sf = core.build_state(val, M, avail)
        Qp = q_prior(val, M, tables, lam, avail)
        mk = core.action_mask(n, M, depth, ch, avail, N)
        reach = core.reachable_nodes(used, avail, n, N, M)
        sc = core.step_cost(lam, avail, N, M)
        S = torch.tensor(Sf[:, tr, :]).reshape(n * len(tr), -1).to(dev)
        QP = torch.tensor(Qp[:, tr, :]).to(dev)
        mk_t = torch.tensor(mk[:, tr, :]).to(dev)
        step = torch.tensor(sc[tr]).to(dev)[None, :, :]
        w = (mk_t & torch.tensor(reach[:, tr]).to(dev).unsqueeze(-1)).float()
        return Sf, Qp, S, QP, mk_t, step, w.reshape(n * len(tr), A)

    Sf, Qp, S, QP, mk_t, step, wflat = tensors(avail)
    Rst = torch.tensor(R[:, tr]).to(dev)
    fdim = Sf.shape[-1]
    online = ResidualQ(fdim, A, hidden).to(dev); target = ResidualQ(fdim, A, hidden).to(dev)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    # the holdout is scored under the first availability draw throughout
    ho_cache = {'nodes': val['nodes'], 'softmax': val['softmax'][:, ho, :], 'y': val['y'][ho],
                'gflops': val['gflops']}
    av_ho, Sf_ho, Qp_ho = avail[ho], Sf[:, ho, :], Qp[:, ho, :]
    best = {'f1': -1.0, 'state': None, 'it': -1}
    denom = wflat.sum().clamp_min(1)
    for it in range(iters):
        if it > 0:                                   # fresh availability draw every iteration
            avail = core.sample_availability(N, M, p_train, rng_av)
            Sf, Qp, S, QP, mk_t, step, wflat = tensors(avail)
            denom = wflat.sum().clamp_min(1)
        with torch.no_grad():
            qo = _fwd(online, S).reshape(n, len(tr), A) + QP
            qt = _fwd(target, S).reshape(n, len(tr), A) + QP
            a_star = torch.where(mk_t, qo, torch.full_like(qo, -1e9)).argmax(-1)
            V = torch.gather(qt, -1, a_star.unsqueeze(-1)).squeeze(-1)
            T = torch.zeros((n, len(tr), A), device=dev)
            T[:, :, M] = Rst
            for m in range(M):
                tgt = -step[:, :, m] + V[child[:, m].clamp_min(0)]
                T[:, :, m] = torch.where(legal[:, m:m + 1], tgt, torch.zeros_like(tgt))
        Tres = T.reshape(n * len(tr), A) - QP.reshape(n * len(tr), A)
        for _ in range(epochs):
            _fit_epoch(online, opt, S, Tres, wflat, denom)
        target.load_state_dict(online.state_dict())
        f1 = _walk(online, ho_cache, M, Sf_ho, Qp_ho, av_ho)[0]
        if f1 > best['f1']:
            best = {'f1': f1, 'it': it, 'state': {k: v.cpu().clone() for k, v in online.state_dict().items()}}
    net = ResidualQ(fdim, A, hidden); net.load_state_dict(best['state'])
    return net.to(dev).eval(), tables, best['it'], best['f1']


# ── checkpoints ──────────────────────────────────────────────────────────────
def ckpt_path(arc_dir, fold, seed, lam):
    return os.path.join(arc_dir, f'arc_fold{fold}_seed{seed}_lam{lam:.3f}.pt')


def save(path, net, tables, meta):
    """{'state_dict': R_theta weights, 'tables': [(sorted S, G_S [C, M])], 'meta': {..., 'lam': training lam}}."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'state_dict': {k: v.cpu() for k, v in net.state_dict().items()},
                'tables': [(sorted(S), np.asarray(t)) for S, t in tables.items()],
                'meta': meta}, path)


def load(path, device=None):
    """-> (net, tables or None, meta). ``meta['lam']`` is the training lam, which is also the
    lam the policy must be deployed with. Checkpoints without stored gain tables return None
    for ``tables``; rebuild them from the validation cache with arc.prior.gain_tables."""
    dev = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    blob = torch.load(path, map_location='cpu', weights_only=False)
    sd, meta = blob['state_dict'], blob['meta']
    net = ResidualQ(sd['net.0.weight'].shape[1], sd['net.4.weight'].shape[0], sd['net.0.weight'].shape[0])
    net.load_state_dict(sd)
    tables = ({frozenset(S): np.asarray(t, np.float32) for S, t in blob['tables']} if 'tables' in blob else None)
    return net.to(dev).eval(), tables, meta


def rollout(net, cache, M, tables, lam, avail=None):
    """Deploy the policy on a cache under the rule it was selected with: Q_0 priced at the
    policy's own training ``lam``, greedy acquisition, STOP as soon as Q(STOP) >= max_m Q(m)
    (ties go to STOP; no stop margin). ``avail`` [N, M] restricts each sample's modalities.
    -> (macro-F1, mean #acquired, mean fusion GFLOPs, stop node per sample)."""
    return _walk(net, cache, M, core.build_state(cache, M, avail), q_prior(cache, M, tables, lam, avail), avail)

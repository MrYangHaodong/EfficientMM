"""The Table 2 controllers on SeMA's prefix cache: GMS, EAMA and Joint-AFA (SeMARC is arc/ itself).

All three select modalities BEFORE the backbone runs and let SeMA process the final set once. A set S is read out as
SeMA's posterior averaged over every cached order of S (set_readout). EAMA and Joint-AFA see the acquired modalities
only through raw previews: each modality's input features averaged over time (raw_previews), z-scored on validation.

  GMS        Cheng et al., UAI 2022, Algorithm 1: greedy chain on validation by the plug-in mutual information
             I(S; Y) = H(Y) - mean_n H(p(y | x_S)); budget k = the first k modalities of the chain.
  EAMA       Peroni et al., TMLR 2026: value model v(X_p, p, q) -> P(loss / same / gain of accuracy), trained on
             random (p, q) views (Algorithm 2); per stage acquire argmax_q score(q) - lam * sum_{i in q} z_i (hybrid
             optimizer), until no q has positive net value (Algorithm 1).
  Joint-AFA  Shim et al., NeurIPS 2018: set encoder + dueling DQN over acquire / STOP, reward log p(y | final set)
             minus r_cost per acquisition.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

EPS = 1e-12


# ── shared ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def set_index(cache):
    """frozenset(acquired set) -> node ids of its cached orders (root excluded)."""
    by_set = {}
    for i, node in enumerate(cache['nodes']):
        if node:
            by_set.setdefault(frozenset(int(t) for t in node.split(',')), []).append(i)
    return by_set


def set_readout(cache, M):
    """[2^M, N, C] posterior of every set (bitmask index), averaged over the set's cached orders; [0] = root."""
    bs = set_index(cache); N, C = cache['softmax'].shape[1:]
    P = np.zeros((1 << M, N, C), np.float32); P[0] = cache['softmax'][0]
    for mask in range(1, 1 << M):
        P[mask] = cache['softmax'][bs[frozenset(j for j in range(M) if mask >> j & 1)]].mean(0)
    return P


@torch.no_grad()
def raw_previews(batches, mods):
    """{modality: [N, V_m]} time-mean of each sample's input features, in the batches' (= the cache's) sample order.
    Computed on CPU: bit-identical to the previews of the paper's runs (a GPU mean differs in the last bits)."""
    out = {m: [] for m in mods}
    for high, _y in batches:
        for m in mods:
            out[m].append(high[m].float().mean(1).numpy())
    return {m: np.concatenate(v, 0).astype(np.float32) for m, v in out.items()}


class Standardizer:
    def __init__(self, prev, mods):
        self.mu = {m: prev[m].mean(0) for m in mods}; self.sd = {m: prev[m].std(0) + 1e-6 for m in mods}

    def __call__(self, prev, mods):
        return {m: (prev[m] - self.mu[m]) / self.sd[m] for m in mods}


def bits(masks, M, dev):
    m = torch.as_tensor(np.asarray(masks), device=dev)
    return ((m[:, None] >> torch.arange(M, device=dev)) & 1).float()


def frequencies(final_masks, M):
    return [float(np.mean([(f >> j) & 1 for f in final_masks])) for j in range(M)]


# ── GMS ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _mutual_information(cache, by_set, S):
    y = cache['y']; py = np.bincount(y, minlength=cache['C']) / len(y)
    p = cache['softmax'][by_set[frozenset(S)]].mean(0)
    return -(py * np.log(py + EPS)).sum() - float((-(p * np.log(p + EPS)).sum(-1)).mean())


def gms_chain(val, test, M):
    """Greedy chain on validation; for every budget k: (chain[:k], test macro-F1 of that set, rounded as stored)."""
    bv, bt = set_index(val), set_index(test)
    S, out = [], []
    for _ in range(M):
        base = _mutual_information(val, bv, S) if S else 0.0
        best, gain = None, -np.inf
        for j in range(M):
            if j not in S:
                g = _mutual_information(val, bv, S + [j]) - base
                if g > gain:
                    best, gain = j, g
        S = S + [best]
        pred = test['softmax'][bt[frozenset(S)]].mean(0).argmax(-1)
        out.append((list(S), round(float(f1_score(test['y'], pred, average='macro')), 4)))
    return out


# ── EAMA ────────────────────────────────────────────────────────────────────────────────────────────────────────────
EAMA_LAMS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


class ValueModel(nn.Module):
    """v(X_p, p, q): per-modality linear previews of the observed set (masked mean) + unit-norm subset embeddings
    e_p, e_q -> MLP -> 3 logits (accuracy loss / same / gain)."""

    def __init__(self, M, var_dims, d=16, d_s=32, h=128):
        super().__init__(); self.M = M; self.E = nn.Parameter(torch.randn(M, d) / d ** 0.5)
        self.proj = nn.ModuleList([nn.Linear(v, d_s) for v in var_dims])
        self.net = nn.Sequential(nn.Linear(d_s + 2 * d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 3))

    def embed(self, ind):
        E = F.normalize(self.E, dim=-1); return (ind @ E) / ind.sum(-1, keepdim=True).clamp_min(1.0)

    def preview(self, X, obs):
        Z = torch.stack([self.proj[m](X[m]) for m in range(self.M)], 1); w = obs.unsqueeze(-1)
        return (Z * w).sum(1) / w.sum(1).clamp_min(1.0)

    def forward(self, X, obs, p_ind, q_ind):
        return self.net(torch.cat([self.preview(X, obs), self.embed(p_ind), self.embed(q_ind)], -1))


def _X(prev, mods, idx, dev):
    return [torch.as_tensor(prev[m][idx], device=dev) for m in mods]


def sample_views(rng, N, M, draws):
    """Algorithm 2: (sample, p, q) with |p| ~ U{0..M-1}, |q| ~ U{1..M-|p|}, p and q uniform of those sizes."""
    idx = np.repeat(np.arange(N), draws); pm = np.zeros(len(idx), int); qm = np.zeros(len(idx), int)
    for r in range(len(idx)):
        k1 = rng.integers(0, M); p = rng.choice(M, size=k1, replace=False); rest = [j for j in range(M) if j not in p]
        k2 = rng.integers(1, M - k1 + 1); q = rng.choice(rest, size=k2, replace=False)
        pm[r] = sum(1 << int(j) for j in p); qm[r] = sum(1 << int(j) for j in q)
    return idx, pm, qm


def eama_train_value(prev_fit, P_fit, y_fit, prev_ho, P_ho, y_ho, M, mods, seed, dev, epochs=60, draws=32):
    """Cross-entropy on the 3-way accuracy change (correct on p | q) - (correct on p) + 1; best epoch by holdout ovo-AUC."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    vm = ValueModel(M, [prev_fit[m].shape[1] for m in mods]).to(dev)
    opt = torch.optim.Adam(vm.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    corr_fit = (P_fit.argmax(-1) == y_fit[None, :]).astype(int); corr_ho = (P_ho.argmax(-1) == y_ho[None, :]).astype(int)

    def batch(prev, corr, idx, pm, qm):
        return (_X(prev, mods, idx, dev), bits(pm, M, dev), bits(qm, M, dev),
                torch.tensor(corr[pm | qm, idx] - corr[pm, idx] + 1, device=dev))

    Xh, ph, qh, th = batch(prev_ho, corr_ho, *sample_views(np.random.default_rng(999), len(y_ho), M, 64))
    best = (-1, None)
    for _ in range(epochs):
        vm.train(); X, pi, qi, t = batch(prev_fit, corr_fit, *sample_views(rng, len(y_fit), M, draws))
        perm = torch.randperm(len(t), device=dev)
        for i in range(0, len(t), 512):
            b = perm[i:i + 512]
            loss = F.cross_entropy(vm([x[b] for x in X], pi[b], pi[b], qi[b]), t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); vm.eval()
        with torch.no_grad():
            pr = torch.softmax(vm(Xh, ph, ph, qh), -1).cpu().numpy()
        try:
            auc = roc_auc_score(th.cpu().numpy(), pr, multi_class='ovo')
        except ValueError:
            auc = 0.0
        if auc > best[0]:
            best = (auc, {k: v.detach().clone() for k, v in vm.state_dict().items()})
    vm.load_state_dict(best[1]); vm.eval()
    return vm, best[0]


@torch.no_grad()
def eama_scores(vm, prev, M, mods, dev):
    """S[p, n, q] = P(gain) - P(loss) for every observed set p and disjoint candidate q: [2^M, N, 2^M]."""
    N = prev[mods[0]].shape[0]; S = np.full((1 << M, N, 1 << M), -np.inf, np.float32); X = _X(prev, mods, np.arange(N), dev)
    for pm in range(1 << M):
        qs = [q for q in range(1, 1 << M) if q & pm == 0]
        if not qs:
            continue
        pi = bits([pm] * N, M, dev); pv = vm.preview(X, pi); ep = vm.embed(pi)
        for q in qs:
            pr = torch.softmax(vm.net(torch.cat([pv, ep, vm.embed(bits([q] * N, M, dev))], -1)), -1)
            S[pm, :, q] = (pr[:, 2] - pr[:, 0]).cpu().numpy()
    return S


def _stage_hybrid(S_p, cost_i, pm, M, rng):
    """One stage: randomized double greedy (Buchbinder et al.) and greedy USM; keep the better (q, net value)."""
    un = [i for i in range(M) if not pm >> i & 1]

    def phi(q):
        return (S_p[q] if q else 0.0) - sum(cost_i[i] for i in range(M) if q >> i & 1)
    a, b = 0, sum(1 << i for i in un)
    for i in un:
        qa, qb = max(phi(a | 1 << i) - phi(a), 0.0), max(phi(b & ~(1 << i)) - phi(b), 0.0)
        if rng.random() < (qa / (qa + qb) if qa + qb > 0 else 0.0):
            a |= 1 << i
        else:
            b &= ~(1 << i)
    q1, v1 = a, phi(a)
    g, vg = 0, 0.0
    while True:
        cands = [(phi(g | 1 << i), i) for i in un if not g >> i & 1]
        if not cands:
            break
        v, i = max(cands)
        if v >= vg:
            g |= 1 << i; vg = v
        else:
            break
    return (g, vg) if vg > v1 else (q1, v1)


def eama_rollout(S, P, y, M, lam, z, t_ms, seed):
    """Algorithm 1 per sample (hybrid optimizer, no context, unlimited stages); SeMA reads the final set once.
    -> (macro-F1, MU, mean extraction ms, mean #stages, per-modality frequency)."""
    rng = np.random.default_rng(seed); N = P.shape[1]
    preds, ks, ms, stages, final = np.zeros(N, int), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N, int)
    for n in range(N):
        pm = 0
        while pm != (1 << M) - 1:
            q, v = _stage_hybrid(S[pm, n], lam * z, pm, M, rng)
            if v <= 0 or q == 0:
                break
            pm |= q; stages[n] += 1
        preds[n] = P[pm, n].argmax(); final[n] = pm
        ks[n] = bin(pm).count('1'); ms[n] = sum(t_ms[i] for i in range(M) if pm >> i & 1)
    return f1_score(y, preds, average='macro'), ks.mean() / M, ms.mean(), stages.mean(), frequencies(final, M)


# ── Joint-AFA ───────────────────────────────────────────────────────────────────────────────────────────────────────
class SetEncoder(nn.Module):
    """Embedder + attentive LSTM read of the acquired elements, n_shuffle times."""

    def __init__(self, elem_dim, embedded_dim=64, lstm_size=64, n_shuffle=3):
        super().__init__()
        self.embedder = nn.Sequential(nn.Linear(elem_dim, 128), nn.ReLU(), nn.Linear(128, embedded_dim))
        self.lstm = nn.LSTMCell(embedded_dim, lstm_size); self.proj = nn.Linear(lstm_size, embedded_dim)
        self.n_shuffle, self.lstm_size, self.embedded_dim = n_shuffle, lstm_size, embedded_dim
        self.out_dim = lstm_size + embedded_dim

    def forward(self, elems, mask):
        B = elems.shape[0]; emb = self.embedder(elems)
        q = elems.new_zeros(B, self.lstm_size); c = elems.new_zeros(B, self.lstm_size)
        att = elems.new_zeros(B, self.embedded_dim); neg = torch.finfo(emb.dtype).min
        for _ in range(self.n_shuffle):
            score = (emb * self.proj(q).unsqueeze(1)).sum(-1).masked_fill(mask < 0.5, neg)
            w = torch.softmax(score, dim=-1) * (mask.sum(-1, keepdim=True) > 0)
            att = (w.unsqueeze(-1) * emb).sum(1)
            q, c = self.lstm(att, (q, c))
        return torch.cat([att, q], dim=-1)


class DuelingNet(nn.Module):
    """Q = V + (A - mean A)."""

    def __init__(self, encoded_dim, n_actions, hidden=128, shared=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(encoded_dim, hidden), nn.ReLU(), nn.Linear(hidden, shared))
        self.pi_net = nn.Sequential(nn.Linear(shared, shared), nn.ReLU(), nn.Linear(shared, n_actions))
        self.v_net = nn.Sequential(nn.Linear(shared, shared), nn.ReLU(), nn.Linear(shared, 1))

    def forward(self, encoded):
        t = F.relu(self.shared(encoded)); adv, v = self.pi_net(t), self.v_net(t)
        return v + (adv - adv.mean(dim=1, keepdim=True))


class JointAFA(nn.Module):
    """Element of the acquired set = [one-hot(modality); linear projection of its raw preview]; Q over M + 1 actions."""

    def __init__(self, M, var_dims, d_s=32, n_shuffle=3):
        super().__init__(); self.M = M
        self.proj = nn.ModuleList([nn.Linear(v, d_s) for v in var_dims])
        self.enc = SetEncoder(elem_dim=M + d_s, n_shuffle=n_shuffle); self.q = DuelingNet(self.enc.out_dim, M + 1)

    def forward(self, X, acq):
        B = acq.shape[0]; Z = torch.stack([self.proj[m](X[m]) for m in range(self.M)], 1)
        oh = torch.eye(self.M, device=acq.device).unsqueeze(0).expand(B, -1, -1)
        return self.q(self.enc(torch.cat([oh, Z], -1), acq))


def _masks(masks, M, dev):
    return torch.tensor([[(mk >> j) & 1 for j in range(M)] for mk in masks], dtype=torch.float32, device=dev)


def jafa_train(prev, P, y, M, mods, r_cost, seed, dev, episodes=20000, gamma=1.0, lr=1e-3, batch=64, buf_cap=50000,
               target_every=500):
    """DQN on the validation split: epsilon 1 -> .1 over 60% of the episodes, replay buffer, target net."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed); random.seed(seed)
    N = len(y); logp = np.log(P[np.arange(1 << M)][:, np.arange(N), y] + 1e-12)      # [2^M, N] log p(y | set)
    dims = [prev[m].shape[1] for m in mods]
    net = JointAFA(M, dims).to(dev); tgt = JointAFA(M, dims).to(dev); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr); buf = []; FULL = (1 << M) - 1
    for ep in range(episodes):
        n = int(rng.integers(N)); pm = 0; done = False; eps = max(0.1, 1.0 - ep / (0.6 * episodes))
        while not done:
            legal = [j for j in range(M) if not pm >> j & 1]
            if rng.random() < eps:
                a = M if rng.random() < 1.0 / (len(legal) + 1) else int(rng.choice(legal))
            else:
                with torch.no_grad():
                    q = net(_X(prev, mods, [n], dev), _masks([pm], M, dev))[0].cpu().numpy()
                q[[j for j in range(M) if pm >> j & 1]] = -np.inf; a = int(np.argmax(q))
            if a == M:
                buf.append((pm, n, M, float(logp[pm, n]), None)); done = True
            else:
                nm = pm | (1 << a); buf.append((pm, n, a, -r_cost, nm)); pm = nm
                if pm == FULL:
                    buf.append((pm, n, M, float(logp[pm, n]), None)); done = True
        if len(buf) > buf_cap:
            buf = buf[-buf_cap:]
        if len(buf) >= batch:
            tr = [buf[i] for i in rng.choice(len(buf), batch, replace=False)]; X = _X(prev, mods, [t[1] for t in tr], dev)
            qsa = net(X, _masks([t[0] for t in tr], M, dev)).gather(1, torch.tensor([t[2] for t in tr], device=dev).view(-1, 1)).squeeze(1)
            with torch.no_grad():
                qn = tgt(X, _masks([t[4] if t[4] is not None else 0 for t in tr], M, dev))
                tv = torch.tensor([t[3] for t in tr], device=dev, dtype=qsa.dtype)
                for b, t in enumerate(tr):
                    if t[4] is not None:
                        tv[b] += gamma * qn[b, [j for j in range(M) if not t[4] >> j & 1] + [M]].max()
            loss = F.smooth_l1_loss(qsa, tv); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        if (ep + 1) % target_every == 0:
            tgt.load_state_dict(net.state_dict())
    return net.eval()


@torch.no_grad()
def jafa_rollout(net, prev, P, y, M, mods, dev):
    """Greedy acquisition from the raw previews; SeMA reads the final set once. -> (macro-F1 rounded as stored, freq)."""
    N = len(y); X = _X(prev, mods, np.arange(N), dev); pm = np.zeros(N, int); active = np.ones(N, bool); FULL = (1 << M) - 1
    for _ in range(M):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        q = net([x[idx] for x in X], _masks(pm[idx], M, dev)).cpu().numpy()
        for r, n in enumerate(idx):
            q[r, [j for j in range(M) if pm[n] >> j & 1]] = -np.inf; a = int(np.argmax(q[r]))
            if a == M:
                active[n] = False
            else:
                pm[n] |= 1 << a
                if pm[n] == FULL:
                    active[n] = False
    preds = P[pm, np.arange(N)].argmax(-1)
    return round(float(f1_score(y, preds, average='macro')), 4), frequencies(pm, M)

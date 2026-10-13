"""The acquisition MDP on a prefix-tree cache (arc/prefix_cache.py).

state   psi = [acquired mask (M) | available mask (M) | posterior (C) | entropy | top-2 margin
               | #acquired / M | #available / M]
actions acquire m (m available and not yet acquired), or STOP (not at the root)
reward  at STOP: class-balanced correctness 1[argmax p = y] / freq(y) (mean weight 1)
cost    each acquisition costs lam / |A|, |A| = #available modalities, so acquiring
        everything that is available costs exactly lam
"""
import numpy as np
from sklearn.metrics import f1_score


def node_meta(nodes, M):
    """depth[n], used[n] (frozenset of acquired modalities), ch[n] = [(m, child id)]."""
    idx = {k: i for i, k in enumerate(nodes)}
    depth = np.array([0 if k == '' else k.count(',') + 1 for k in nodes])
    used = [frozenset() if k == '' else frozenset(int(x) for x in k.split(',')) for k in nodes]
    ch = {}
    for i, k in enumerate(nodes):
        if depth[i] >= M:
            ch[i] = []; continue
        u = [] if k == '' else [int(x) for x in k.split(',')]
        ch[i] = [(m, idx[','.join(map(str, u + [m]))]) for m in range(M)
                 if m not in u and ','.join(map(str, u + [m])) in idx]
    return depth, used, ch


def child_table(ch, n, M):
    """child_of[node, m] = child id after acquiring m, -1 if m cannot be acquired there."""
    child_of = np.full((n, M), -1, np.int64)
    for i in range(n):
        for (m, c) in ch[i]:
            child_of[i, m] = c
    return child_of


def sample_availability(N, M, p, rng):
    """Bernoulli(p) availability per (sample, modality), at least one modality per sample."""
    av = rng.random((N, M)) < p
    empty = ~av.any(1)
    if empty.any():
        av[empty, rng.integers(0, M, int(empty.sum()))] = True
    return av


def _avail(avail, N, M):
    return np.ones((N, M), bool) if avail is None else np.asarray(avail, bool)


def build_state(cache, M, avail=None):
    """[n, N, 2M + C + 4] state of every (node, sample)."""
    nodes = cache['nodes']; soft = cache['softmax']; n, N, C = soft.shape
    depth, used, _ = node_meta(nodes, M)
    av = _avail(avail, N, M)
    H = np.zeros((n, N)); margin = np.zeros((n, N)); kk = np.zeros((n, N))
    mask = np.zeros((n, N, M), np.float32)
    for i in range(n):
        p_ = np.clip(soft[i], 1e-8, 1.0)
        H[i] = -(p_ * np.log(p_)).sum(-1)
        srt = np.sort(soft[i], -1); margin[i] = srt[:, -1] - srt[:, -2]
        kk[i] = depth[i] / M
        for m in used[i]:
            mask[i, :, m] = 1.0
    avb = np.broadcast_to(av.astype(np.float32)[None], (n, N, M))
    nav = np.broadcast_to((av.sum(1) / M).astype(np.float32)[None, :, None], (n, N, 1))
    return np.concatenate([mask, avb, soft, H[..., None], margin[..., None], kk[..., None], nav],
                          -1).astype(np.float32)


def state_dim(M, C):
    return 2 * M + C + 4


def action_mask(n, M, depth, ch, avail, N):
    """[n, N, M+1] legal actions: acquire m if available and not held; STOP below the root."""
    av = _avail(avail, N, M)
    mk = np.zeros((n, N, M + 1), bool)
    for i in range(n):
        for (m, _c) in ch[i]:
            mk[i, :, m] = av[:, m]
        if depth[i] > 0:
            mk[i, :, M] = True
    return mk


def reachable_nodes(used, avail, n, N, M):
    """[n, N]: node i can occur for sample j only if every modality it holds is available."""
    av = _avail(avail, N, M)
    R = np.ones((n, N), bool)
    for i in range(n):
        for m in used[i]:
            R[i] &= av[:, m]
    return R


def step_cost(lam, avail, N, M):
    """[N, M] cost of one acquisition: lam / |A_n|."""
    av = _avail(avail, N, M).astype(np.float32)
    return np.repeat((lam / av.sum(1))[:, None], M, axis=1)


def correctness(cache):
    """[n, N] terminal reward: 1[correct] / freq(y), weights normalised to mean 1 over samples."""
    soft = cache['softmax']; y = cache['y']
    corr = (soft.argmax(-1) == y[None, :]).astype(np.float32)
    C = soft.shape[-1]
    freq = np.bincount(y, minlength=C).astype(np.float64) / len(y)
    wraw = 1.0 / np.maximum(freq, 1e-8)
    w = wraw[y] / wraw[y].mean()
    return (corr * w[None, :].astype(np.float32)).astype(np.float32)


def inner_split(N, seed=1000, frac=0.2):
    """Fixed split of the validation samples: (fit indices, holdout indices)."""
    perm = np.random.default_rng(seed).permutation(N)
    cut = max(5, int(frac * N))
    return perm[cut:], perm[:cut]


def macro_f1(y, preds):
    return float(f1_score(y, preds, average='macro'))

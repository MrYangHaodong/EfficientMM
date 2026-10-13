"""The value prior Q_0 of ARC.

    Q_0(psi, STOP)      = conf(psi)
    Q_0(psi, acquire m) = conf(psi) + g(psi, m) - lam / |A|

conf is the maximum posterior probability. g is the expected gain in correctness from
adding m to the acquired set S, read through the current posterior:

    g(psi, m) = sum_c p(c | S) * G_S[c, m],   G_S[c, m] = mean over validation samples of class c
                                                          of 1[correct | S + m] - 1[correct | S],

where "correct | S" is averaged over every cached ordering of S. The tables G use
validation labels only; at test time g needs nothing but the posterior.
"""
from collections import defaultdict

import numpy as np

from arc.core import step_cost


def set_index(cache):
    """frozenset(acquired set) -> [node ids of every cached ordering of that set]."""
    g = defaultdict(list)
    for i, k in enumerate(cache['nodes']):
        g[frozenset() if k == '' else frozenset(int(x) for x in k.split(','))].append(i)
    return g


def gain_tables(cache, M):
    """{frozenset(S): [C, M] per-class conditional gain table} for every |S| < M."""
    groups = set_index(cache)
    y = cache['y']; C = cache['softmax'].shape[-1]; soft = cache['softmax']

    def v(S):
        ids = groups.get(frozenset(S))
        if not ids:
            return None
        return np.mean([(soft[i].argmax(-1) == y).astype(np.float64) for i in ids], axis=0)

    out = {}
    for S in groups:
        if len(S) >= M:
            continue
        base = v(S)
        if base is None:
            continue
        tab = np.zeros((C, M), np.float32)
        for m in range(M):
            if m in S:
                continue
            a = v(set(S) | {m})
            if a is None:
                continue
            d = a - base
            for c in range(C):
                sel = (y == c)
                if sel.any():
                    tab[c, m] = d[sel].mean()
        out[frozenset(S)] = tab
    return out


def gain_advice(cache, M, tables, avail=None):
    """[n, N, M] g(psi, m), zero for held or unavailable modalities."""
    nodes = cache['nodes']; soft = cache['softmax']; n, N, C = soft.shape
    used = [frozenset() if k == '' else frozenset(int(x) for x in k.split(',')) for k in nodes]
    adv = np.zeros((n, N, M), np.float32)
    for i in range(n):
        tab = tables.get(used[i])
        if tab is None:
            continue
        a = soft[i] @ tab
        for m in used[i]:
            a[:, m] = 0.0
        adv[i] = a
    if avail is not None:
        adv *= np.asarray(avail, np.float32)[None]
    return adv


def q_prior(cache, M, tables, lam, avail=None):
    """[n, N, M+1] Q_0 at every (node, sample); action M is STOP."""
    soft = cache['softmax']; n, N, C = soft.shape
    conf = soft.max(-1)
    adv = gain_advice(cache, M, tables, avail)
    sc = step_cost(lam, avail, N, M)
    Q = np.zeros((n, N, M + 1), np.float32)
    Q[:, :, M] = conf
    Q[:, :, :M] = conf[:, :, None] + adv - sc[None, :, :]
    return Q

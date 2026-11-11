#!/usr/bin/env python
"""SeMARC and SeMA + EAMA over the same cost weights gamma, each at Table 2's lambda; Joint-AFA over r_cost and GMS over k
as in lam_curves.py / ctrl_curves.py. Weighted EAMA is left out (user, 2026-11-10); EAMA's line also runs to its gamma = 0
point, Table 2's EAMA, although its gamma = .25 point beats it.

  SeMARC   gamma at Table 2's lambda (.5; MM-Fi fold 2: 1): the paper's gamma-study nets (plot_gamma.py), as in
           fig/pareto_controllers_mean.
  EAMA     lambda held at Table 2's (each run's validation-selected uniform-cost lambda) and modality costs rescaled as
           SeMARC's: z_j(gamma) = t_j^gamma / sum_m t_m^gamma, i.e. modality j's price changes by the same factor
           M w_j / sum w as SeMARC's (ctrl/eama_gamma.py; value models retrained and checked against the paper's run).
Latency = Table 2's estimate (EAMA: extraction of the acquired set + one fusion + (stages + 1) selector calls).
Six-dataset mean: one gamma shared by all six datasets, as every knob. Writes res/gamma_ctrl_points.csv,
res/GAMMA_CTRL.md and fig/pareto_gamma_controllers_mean.{pdf,png}.
"""
import glob
import json
import math
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plot_pareto as pp
import plot_gamma as pg
import ctrl_curves as cc
from ctrl_curves import EAMA, JAFA, GMSN, SEMARC, kv, f3, ms
from plot_pareto import BLUE, INK, INK2, MUTED

T = cc.T
EG = f'{T}/ctrl/eama_gamma'
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
cc.KNOB[EAMA] = 'γ'                                          # EAMA's knob in this figure
cc.CURVE[EAMA] = cc.CURVE[EAMA][:4] + ("SeMA + EAMA: γ 0–2 at Table 2's λ",)
cc.CURVE[SEMARC] = cc.CURVE[SEMARC][:4] + ("SeMARC: γ 0–2 at Table 2's λ",)
GMS_F = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]          # factor on each dataset's Table 2 k: -75% ... +400%
cc.KNOB[GMSN] = 'k scale'
cc.CURVE[GMSN] = cc.CURVE[GMSN][:4] + ("SeMA + GMS: Table 2's k scaled −75% to +400%",)
T2_VALUE = {EAMA: 0.0, GMSN: 1.0}                                 # the knob value that is the method's Table 2 point


def pct(f):
    return '0' if f == 1 else f'{(f - 1) * 100:+.0f}%'.replace('-', '−')


# ───────────────────────────── EAMA over gamma ─────────────────────────────
def eama_gamma():
    """Per dataset and gamma: mean over the 9 runs (3 folds x 3 value-model seeds) of test F1 and MU; latency priced from
    the mean per-modality frequency and mean #stages, as Table 2's EAMA point (ctrl_curves.eama)."""
    out, chk = [], []
    for ds in pp.ORDER:
        fs = sorted(glob.glob(f'{EG}/eama_gamma_{ds}_f*.json'))
        runs = [json.load(open(f)) for f in fs]
        if len(runs) != 3:
            continue
        mods = runs[0]['mods']; M = len(mods)
        for r in runs:
            assert r['mods'] == mods and len(r['seeds']) == 3
            chk += [s['check_max_abs_diff'] for s in r['seeds'].values()]
        for g in GAMMAS:
            e = [s['gamma'][f'{g:g}'] for r in runs for s in r['seeds'].values()]
            freq = np.mean([x['freq'] for x in e], 0); st = float(np.mean([x['stages'] for x in e]))
            out.append(cc.row(EAMA, ds, 'curve', g, float(np.mean([x['f1'] for x in e])), float(np.mean([x['mu'] for x in e])),
                              cc.price(ds, mods, freq, st + 1), len(e), source="retrained value models at Table 2's λ (ctrl/eama_gamma)",
                              lams=' '.join(f"{s['lam_table2']:g}" for r in runs for s in r['seeds'].values())))
    return out, (max(chk) if chk else np.nan)


def gms_scaled():
    """GMS with every dataset's Table 2 k (validation-selected) scaled by the same factor f: k = round_half_up(f x k_T2),
    clipped to [1, M]. f = 1 is Table 2's GMS. The stored chains have test F1 at every k (cc.gms), so nothing is refitted."""
    g = pd.DataFrame(cc.gms()); cur = g[g.kind == 'curve']
    out = g[g.kind == 'table2'].to_dict('records')
    for f in GMS_F:
        for ds in pp.ORDER:
            c = cur[cur.dataset == ds]; M = int(c.value.max())
            k = min(M, max(1, math.floor(cc.GMS_K[ds] * f + 0.5)))
            r = c[c.value == k].iloc[0].to_dict()
            r.update(value=f, knob='k scale', k=k)
            out.append(r)
    return out


def load():
    base = [r for r in cc.eama() if r['method'] == EAMA and r['kind'] == 'table2']    # EAMA's Table 2 row only
    for r in base:
        r['knob'] = 'γ'
    eg, chk = eama_gamma()
    d = pd.DataFrame(base + eg + cc.jafa() + gms_scaled() + cc.semarc())
    have = set(d[(d.method == EAMA) & (d.kind == 'curve')].dataset)
    a = d
    if len(have) == 6:
        gc = (d.method == GMSN) & (d.kind == 'curve')               # cc.means would read GMS values as k: average here
        gm = []
        for f, s in d[gc].groupby('value'):
            assert s.dataset.nunique() == 6, f
            gm.append(dict(method=GMSN, knob='k scale', kind='curve', value=f, dataset='mean', n=6,
                           **{c: s[c].mean() for c in cc.COLS}))
        a = pd.concat([d, cc.means(d[~gc]), pd.DataFrame(gm)], ignore_index=True)
    a['on_front'] = False
    for _, g in a[a.kind == 'curve'].groupby(['method', 'dataset']):
        a.loc[pp.pareto(g).index, 'on_front'] = True
    return a, chk, have


def check_gamma0(a):
    """EAMA at gamma = 0 is Table 2's EAMA (same lambdas, z = 1/M): F1 and latency must coincide on every dataset."""
    rows = []
    for ds in pp.ORDER + ['mean']:
        g0 = a[(a.method == EAMA) & (a.dataset == ds) & (a.kind == 'curve') & (a.value == 0.0)]
        t2 = a[(a.method == EAMA) & (a.dataset == ds) & (a.kind == 'table2')]
        if len(g0) and len(t2):
            rows.append(dict(dataset=ds, f1_g0=g0.f1.iloc[0], f1_t2=t2.f1.iloc[0], ms_g0=g0.latency_ms.iloc[0],
                             ms_t2=t2.latency_ms.iloc[0]))
    o = pd.DataFrame(rows)
    assert ((o.f1_g0 - o.f1_t2).abs() < 1e-9).all() and ((o.ms_g0 - o.ms_t2).abs() < 1e-6).all(), o
    return o


# ───────────────────────────── figure ─────────────────────────────
ORDER_G = [GMSN, JAFA, EAMA, SEMARC]                             # drawing order: SeMARC on top; no weighted EAMA


def line_pts(a, m, ds):
    """A method's drawn points: its Pareto front over its knob, plus, for EAMA and GMS, the knob value that is their
    Table 2 point (EAMA gamma = 0, GMS k scale 1) even when another value beats it."""
    c = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'curve')]
    keep = c.on_front | (c.value.apply(lambda v: v == T2_VALUE[m]) if m in T2_VALUE else False)
    return c[keep].sort_values('latency_ms')


def label_gms(ax, a, ds, mmn, box, fs=7.8, merge_px=10):
    """Percentage labels for GMS's drawn points on the full-range panel (its points lie right of the zoom); points closer
    than merge_px share one label, e.g. '−25%, 0'."""
    ob = obstacles(ax, a, ds)
    px = ax.figure.dpi / 72; tr = ax.transData.transform
    for r in mmn.itertuples():
        x, y = tr((r.latency_ms, r.f1)); ob.append((x, y, 6 * px))
    (x0, x1), (y0, y1) = box                                        # the grey zoom box's border
    for (xa, ya), (xb, yb) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        for t in np.linspace(0, 1, 60):
            ob.append((*tr((xa * (xb / xa) ** t, ya + t * (yb - ya))), 1.2 * px))
    fr = line_pts(a, GMSN, ds); xy = tr(fr[['latency_ms', 'f1']].values.astype(float))
    groups, cur = [], [0]
    for i in range(1, len(fr)):
        if np.hypot(*(xy[i] - xy[cur[0]])) < merge_px * px:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    placed = []
    for g in groups:
        rows = fr.iloc[g]
        text = ', '.join(pct(v) for v in sorted(rows.value))
        placed.append(cc.place_label(ax, rows.latency_ms.mean(), rows.f1.mean(), text, ob, placed, fs=fs, rad_pt=4.5,
                                     order=('right', 'lr', 'ur', 'left', 'll', 'ul', 'below', 'above')))


def draw(ax, a, ds, mmn=None):
    for m in ORDER_G:
        col, ls, mk, filled, _ = cc.CURVE[m]
        fr = line_pts(a, m, ds)
        ax.plot(fr.latency_ms, fr.f1, color=col, ls=ls, lw=2.0 if m == SEMARC else 1.6, zorder=3, solid_joinstyle='round')
        ax.scatter(fr.latency_ms, fr.f1, s=cc.S_DOT * (1.15 if mk == '^' else 1), marker=mk, color=col, edgecolors='white',
                   linewidths=1.0, zorder=4 if m != SEMARC else 5)
    for m in (GMSN, JAFA, EAMA):                                  # each controller's Table 2 configuration
        col, _, mk, _, _ = cc.CURVE[m]
        p = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'table2')]
        ax.scatter(p.latency_ms, p.f1, s=cc.S_PUB * (1.15 if mk == '^' else 1), marker=mk, color=col, edgecolors=INK,
                   linewidths=1.3, zorder=6)
    s = a[(a.method == SEMARC) & (a.dataset == ds) & (a.kind == 'curve')]
    for g, mk, sz in ((0.0, '*', 170), (0.75, 'D', 62)):          # SeMARC's Table 2 rows: uniform (γ = 0), weighted (γ = .75)
        r = s[s.value == g]
        ax.scatter(r.latency_ms, r.f1, s=sz, marker=mk, color=BLUE if mk == '*' else 'white', edgecolors=INK if mk == '*' else BLUE,
                   linewidths=1.1 if mk == '*' else 1.6, zorder=7)
    if mmn is not None:
        ax.scatter(mmn.latency_ms, mmn.f1, s=60, marker='P', color=cc.GRAY, edgecolors='white', linewidths=.8, zorder=5)


def legend_handles():
    h = []
    for m in [SEMARC, EAMA, JAFA, GMSN]:
        col, ls, mk, _, txt = cc.CURVE[m]
        h.append(Line2D([], [], color=col, ls=ls, lw=1.8, marker=mk, ms=5.2, mfc=col, mec='white', mew=1.0, label=txt))
    h += [Line2D([], [], lw=0, marker='*', ms=12, color=BLUE, mec=INK, mew=.9, label='SeMARC, Table 2 (γ = 0)'),
          Line2D([], [], lw=0, marker='D', ms=6.5, color='white', mec=BLUE, mew=1.5, label="SeMARC, paper's weighted row (γ = .75)"),
          Line2D([], [], lw=0, marker='o', ms=8.5, color='#d7d6d0', mec=INK, mew=1.2,
                 label="Black-edged marker: the controller's\nTable 2 configuration"),
          Line2D([], [], lw=0, marker='P', ms=8, color=cc.GRAY, mec='white', label='MultiModN + ARC (Table 2)')]
    return h


def obstacles(ax, a, ds):
    """Every drawn mark (circle in display pixels) and samples along every drawn line, for the label placer."""
    px = ax.figure.dpi / 72; tr = ax.transData.transform; ob = []
    pts = pd.concat([line_pts(a, m, ds) for m in ORDER_G] + [a[(a.dataset == ds) & (a.kind == 'table2')]])
    for r in pts.itertuples():
        size = cc.S_PUB if r.kind == 'table2' else cc.S_DOT
        if r.method == SEMARC and r.value in (0.0, 0.75):
            size = 170 if r.value == 0.0 else 62
        x, y = tr((r.latency_ms, r.f1)); ob.append((x, y, (np.sqrt(size) / 2 + 1.5) * px))
    for m in ORDER_G:
        xy = tr(line_pts(a, m, ds)[['latency_ms', 'f1']].values.astype(float))
        for p, q in zip(xy[:-1], xy[1:]):
            for t in np.linspace(0, 1, 40):
                ob.append((*(p + t * (q - p)), 1.6 * px))
    return ob


NOTE = ('Mean of the six datasets. Each knob value is shared by\n'
        "all six; lines are each method's Pareto front over its knob.\n"
        "SeMARC and EAMA: γ varied at Table 2's λ (EAMA: each\n"
        "run's validation-picked λ); EAMA's modality costs scale as\n"
        "SeMARC's do, z$_j$ = t$_j^γ$ / Σ t$^γ$. GMS: every dataset's Table 2\n"
        'k scaled by the same percentage (rounded, 1 ≤ k ≤ M).\n'
        "EAMA's and GMS's lines also run through their Table 2\n"
        "points (EAMA's γ = 0 is beaten by its γ = .25). Numbers\n"
        'next to the dots give γ or the change in k. Joint-AFA\'s\n'
        "black-edged marker is the paper's run.")


def plot_mean(a, mmn, box, out=f'{T}/fig/pareto_gamma_controllers_mean', xticks=None):
    """Left: the whole range (log latency). Middle: the box (linear latency). Right: legend."""
    ds = 'mean'
    plt.rcParams.update(pg.RC)
    fig = plt.figure(figsize=(12.4, 4.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.18, .9])
    ax, az, lax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    draw(ax, a, ds, mmn=mmn)
    vis = a[(a.dataset == ds) & ((a.kind == 'table2') | a.on_front)]
    xs = np.r_[vis.latency_ms.values, mmn.latency_ms.values]; ys = np.r_[vis.f1.values, mmn.f1.values]
    lo, hi = xs.min() / 1.35, xs.max() * 1.45
    pg.style_axes(ax, ds, lo, hi)
    ax.set_xticks([t for t in (20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000) if lo <= t <= hi])   # narrow range: denser ticks
    ax.set_title('Mean of six datasets (Table 2): full range', fontsize=10.5, color=INK, loc='left', fontweight='bold')
    ax.set_ylim(ys.min() - .02, ys.max() + .012)
    (x0, x1), (y0, y1) = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=MUTED, lw=.9, zorder=2))
    draw(az, a, ds)
    az.set_xlim(x0, x1); az.set_ylim(y0, y1)
    az.xaxis.set_major_locator(plt.FixedLocator(xticks) if xticks else plt.MaxNLocator(5))
    az.yaxis.set_major_locator(plt.MaxNLocator(5))
    cc.lighten_axes(az)
    az.set_title(f'Zoom: {x0}–{x1} ms (grey box, left)', fontsize=10.5, color=INK, loc='left', fontweight='bold')
    az.set_xlabel('Estimated GPU latency (ms)'); az.set_ylabel('Macro-F1')
    lax.set_axis_off()
    leg = lax.legend(handles=legend_handles(), loc='upper left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8.6,
                     handletextpad=.5, labelspacing=.55, borderaxespad=0, handlelength=2.6)
    fig.tight_layout(w_pad=1.2)
    fig.canvas.draw(); rd = fig.canvas.get_renderer()
    y_leg = lax.transAxes.inverted().transform((0, leg.get_window_extent(rd).y0))[1]
    lax.text(0, y_leg - .035, NOTE, transform=lax.transAxes, fontsize=8.0, color=INK2, va='top')
    label_gms(ax, a, ds, mmn, box)
    ob = obstacles(az, a, ds)
    placed = []
    for m, text, order in ((EAMA, 'EAMA, Table 2', ('ur', 'above', 'right', 'lr', 'below', 'ul', 'left', 'll')),):
        r = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'table2')].iloc[0]
        if x0 <= r.latency_ms <= x1 and y0 <= r.f1 <= y1:
            placed.append(cc.place_label(az, r.latency_ms, r.f1, text, ob, placed, fs=8.0, rad_pt=7, order=order))
    for m, order in ((SEMARC, ('above', 'ul', 'ur', 'left', 'right', 'below', 'll', 'lr')),
                     (EAMA, ('below', 'lr', 'll', 'right', 'left', 'above', 'ur', 'ul'))):
        fr = line_pts(a, m, ds)
        for r in fr[fr.latency_ms.between(x0, x1) & fr.f1.between(y0, y1)].itertuples():
            big = {0.0: 6.5, 0.75: 4.0}.get(r.value, 2.6) if m == SEMARC else (7.0 if r.value == 0.0 else 2.6)
            placed.append(cc.place_label(az, r.latency_ms, r.f1, kv(r.value), ob, placed, fs=8.4, rad_pt=big, order=order))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=250, bbox_inches='tight')
    plt.close(fig)


# ───────────────────────────── report ─────────────────────────────
def matched(a, ds):
    """SeMARC and EAMA at the same gamma (both at Table 2's lambda)."""
    s = a[(a.method == SEMARC) & (a.dataset == ds) & (a.kind == 'curve')].set_index('value')
    e = a[(a.method == EAMA) & (a.dataset == ds) & (a.kind == 'curve')].set_index('value')
    rows = []
    for g in GAMMAS:
        p, q = s.loc[g], e.loc[g]
        verdict = ('SeMARC' if p.f1 >= q.f1 and p.latency_ms <= q.latency_ms else
                   'EAMA' if q.f1 >= p.f1 and q.latency_ms <= p.latency_ms else 'neither')
        rows.append(dict(gamma=g, s_f1=p.f1, s_ms=p.latency_ms, e_f1=q.f1, e_ms=q.latency_ms, beats=verdict))
    return pd.DataFrame(rows)


def write_md(a, chk, g0):
    L = ['# SeMARC and SeMA + EAMA over the same cost weights γ, at Table 2\'s λ (Table 2 metric)', '',
         "Request: fix λ for SeMARC and SeMA + EAMA, vary γ for SeMARC, and change each EAMA modality cost by the same "
         "percentage as SeMARC's; the other variants as in `fig/pareto_lambda_controllers_mean` (`gamma_ctrl.py`, "
         '`fig/pareto_gamma_controllers_mean.{pdf,png}`).', '',
         '## Setup', '',
         "- **SeMARC**: γ ∈ {0, .25, .5, .75, 1, 1.5, 2} at Table 2's λ (.5; MM-Fi fold 2: 1), the paper's γ-study nets "
         '(as in `fig/pareto_controllers_mean`). Modality j is priced λ w_j / Σ_m w_m with w_m = (t_m / min t)^γ.',
         "- **SeMA + EAMA**: λ held at Table 2's value, each run's (fold × value-model seed) λ picked on validation at "
         'uniform cost. Its uniform cost z_j = 1/M is multiplied by the factor by which γ changes SeMARC\'s price of '
         'modality j, M w_j / Σ_m w_m. So z_j(γ) = t_j^γ / Σ_m t_m^γ.',
         '  - γ = 0 is Table 2\'s EAMA row; γ = 1 uses the weighted row\'s costs t_j / Σ t, but at the uniform row\'s λ.',
         '  - EAMA\'s costs act at test time only, but its value models were not stored. They were retrained with '
         '`baseline_eama_raw.py`\'s own functions and seeds (`ctrl/eama_gamma.py`, GPUs 2/3).',
         f'  - Check: the retrained models reproduce the paper\'s run exactly: holdout AUC, every stored test row (both costs, '
         f'all nine λ: F1, MU, extraction ms, stages) and both selected λ, on all 54 runs; largest difference {chk:.1e}.',
         '- **SeMA + GMS**: each dataset\'s Table 2 k (selected on validation: IEMOCAP 5, MM-Fi 1, CZU-MHAD 4, EAV 1, CMI 5, '
         'UTD-MHAD 2) scaled by the same percentage, −75% to +400%, rounded half up and clipped to 1 … M. 0% is Table 2\'s '
         'GMS. The stored greedy chains have test F1 at every k, so nothing is refitted.',
         '- **Joint-AFA** over r_cost (retrained) as in `fig/pareto_lambda_controllers_mean`, with the paper\'s run as its '
         'marker; MultiModN + ARC as a point. Weighted EAMA is left out, at the user\'s request.',
         '- **Mean**: one knob value shared by all six datasets. Lines are Pareto fronts; EAMA\'s and GMS\'s also run through '
         'their Table 2 points (EAMA\'s γ = 0 is beaten by its γ = .25; GMS\'s 0% is on its front).', '',
         '## Check: EAMA at γ = 0 is Table 2\'s EAMA', '',
         '| Dataset | F1 at γ = 0 | Table 2 F1 | latency at γ = 0 (ms) | Table 2 latency (ms) |', '|---|---|---|---|---|']
    for r in g0.itertuples():
        L.append(f'| {pp.LABEL[r.dataset].replace(" (Table 2)", "")} | {r.f1_g0:.4f} | {r.f1_t2:.4f} | {r.ms_g0:.2f} | {r.ms_t2:.2f} |')
    L += ['', '## Mean of six datasets: SeMARC and EAMA at the same γ', '',
          '| γ | SeMARC F1 | SeMARC latency (ms) | EAMA F1 | EAMA latency (ms) | at this γ, better on both |',
          '|---|---|---|---|---|---|']
    for r in matched(a, 'mean').itertuples():
        L.append(f'| {kv(r.gamma)} | {f3(r.s_f1)} | {ms(r.s_ms)} | {f3(r.e_f1)} | {ms(r.e_ms)} | {r.beats} |')
    fe = a[(a.method == EAMA) & (a.dataset == 'mean') & (a.kind == 'curve')]
    L += ['', "EAMA's γ front on the mean: " + '; '.join(f'γ = {kv(r.value)}: {f3(r.f1)} @ {ms(r.latency_ms)}'
                                                         for r in fe[fe.on_front].sort_values('latency_ms').itertuples()), '']
    L += ['## SeMA + GMS with Table 2\'s k scaled', '',
          '| change in k | k per dataset (' + ', '.join(pp.LABEL[d] for d in pp.ORDER) + ') | mean F1 | mean latency (ms) | on the front |',
          '|---|---|---|---|---|']
    for f in GMS_F:
        ks = [int(a[(a.method == GMSN) & (a.dataset == d) & (a.kind == 'curve') & (a.value == f)].k.iloc[0]) for d in pp.ORDER]
        r = a[(a.method == GMSN) & (a.dataset == 'mean') & (a.kind == 'curve') & (a.value == f)].iloc[0]
        L.append(f'| {pct(f)} | {" / ".join(map(str, ks))} | {f3(r.f1)} | {ms(r.latency_ms)} | {"yes" if r.on_front else ""} |')
    t = a[(a.method == GMSN) & (a.dataset == 'mean') & (a.kind == 'table2')].iloc[0]
    L += [f'| Table 2 | 5 / 1 / 4 / 1 / 5 / 2 | {f3(t.f1)} | {ms(t.latency_ms)} | |', '']
    vs, back = cc.versus(a, 'mean')
    L += ['## Who beats whom on the mean', '',
          "Beaten = some point has F1 ≥ and latency ≤, one strictly. 'SeMARC γ' = the blue curve.", '',
          '| Controller | point | F1 | latency (ms) | beaten by SeMARC γ | SeMARC γ best F1 at ≤ its latency |', '|---|---|---|---|---|---|']
    for r in vs.itertuples():
        pt = ('Table 2' if r.kind == 'table2' else f'k {pct(r.value)}' if r.method == GMSN
              else f'{cc.KNOB[r.method]} = {kv(r.value)}')
        L.append(f'| {r.method} | {pt} | {f3(r.f1)} | {ms(r.latency_ms)} | {r.beaten_gamma} | {f3(r.semarc_best_f1_at_le)} |')
    L += ['', '| SeMARC γ (front) | F1 | latency (ms) | beaten by a controller point |', '|---|---|---|---|']
    for r in back.itertuples():
        L.append(f'| {kv(r.gamma)} | {f3(r.f1)} | {ms(r.latency_ms)} | {r.beaten_by} |')
    L += ['', '## Per dataset: SeMARC and EAMA at the same γ', '',
          '| Dataset | γ where SeMARC is better on both | γ where EAMA is better on both | neither (one faster, the other more accurate) |',
          '|---|---|---|---|']
    for ds in pp.ORDER:
        m = matched(a, ds)
        L.append(f'| {pp.LABEL[ds]} | ' + ' | '.join(', '.join(kv(g) for g in m[m.beats == w].gamma) or '–'
                                                  for w in ('SeMARC', 'EAMA', 'neither')) + ' |')
    L += ['', '| Dataset | SeMARC γ front: γ → F1 @ ms | EAMA γ front: γ → F1 @ ms |', '|---|---|---|']
    for ds in pp.ORDER:
        cells = []
        for m in (SEMARC, EAMA):
            fr = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'curve') & a.on_front].sort_values('latency_ms')
            cells.append('; '.join(f'{kv(r.value)} → {f3(r.f1)} @ {ms(r.latency_ms)}' for r in fr.itertuples()))
        L.append(f'| {pp.LABEL[ds]} | ' + ' | '.join(cells) + ' |')
    L.append('')
    open(f'{T}/res/GAMMA_CTRL.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    a, chk, have = load()
    print('EAMA γ runs for', sorted(have), '; largest reproduction difference', chk)
    if len(have) < 6:
        sys.exit(0)
    g0 = check_gamma0(a)
    a.to_csv(f'{T}/res/gamma_ctrl_points.csv', index=False)
    m = a[a.dataset == 'mean']
    print(m[m.method.isin([SEMARC, EAMA])][['method', 'kind', 'value', 'f1', 'mu', 'latency_ms', 'on_front']].round(4).to_string())
    print(matched(a, 'mean').round(4).to_string())
    pts = pp.paper_points()
    mmn = pts[(pts.dataset == 'mean') & (pts.method == 'MultiModN + ARC')]
    x0, x1, y0, y1 = (float(v) for v in sys.argv[1:5]) if len(sys.argv) >= 5 else (290, 560, .66, .78)
    plot_mean(a, mmn, ((int(x0), int(x1)), (y0, y1)), xticks=[300, 350, 400, 450, 500, 550])
    write_md(a, chk, g0)

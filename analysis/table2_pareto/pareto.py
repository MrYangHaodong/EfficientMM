#!/usr/bin/env python
"""Table 2 F1-latency Pareto curves: SeMARC against the controllers of Table 2, as six-dataset means.

Inputs (data/):
  fold_results.csv  one row per (method, knob value, dataset, test fold, run): test macro-F1, and freq_j, the share of the
                    fold's test samples whose final acquired set contains the dataset's j-th modality (order: timings.json).
                    EAMA rows also give its mean number of selection rounds (stages).
  timings.json      per dataset: component times measured on one NVIDIA L40 (otherwise idle server, batch size 1, no
                    feature caching) and two calibration factors (direct timing / sum of components)
Steps:
  1. latency()  Table 2's latency estimate of each fold result.
  2. Mean over the folds (and runs) of each (method, knob value, dataset); then the equal-weight mean of the six datasets.
     A knob value counts only if all six datasets have it. GMS's knob is one percentage change of every dataset's
     Table 2 budget k.
  3. Each method keeps its Pareto front over its knob values. EAMA and GMS also keep their Table 2 setting.
Outputs: pareto_points.csv (the plotted points) and pareto_gamma_controllers.pdf / .png.

    python pareto.py                                                   # the shipped inputs in data/
    python pareto.py --replace results/fold_results_iemocap.csv        # with rows retrained by train_settings.py
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['iemocap', 'mmfi', 'czu_mhad', 'eav', 'cmi', 'utd_mhad']
KNOB = {'SeMARC': 'gamma', 'SeMA + EAMA': 'gamma', 'SeMA + Joint-AFA': 'r_cost', 'SeMA + GMS': 'k_change_pct'}
TABLE2 = {'SeMARC': 0.0, 'SeMA + EAMA': 0.0, 'SeMA + Joint-AFA': 0.05, 'SeMA + GMS': 0.0}   # knob value Table 2 uses
KEEP_TABLE2 = ('SeMA + EAMA', 'SeMA + GMS')                    # drawn even when beaten (EAMA's gamma = 0 is)
GMS_K = {'iemocap': 5, 'mmfi': 1, 'czu_mhad': 4, 'eav': 1, 'cmi': 5, 'utd_mhad': 2}       # Table 2's budgets
GMS_PCT = [-75, -50, -25, 0, 25, 50, 100, 200, 400]
# MultiModN + ARC: Table 2's configuration; per-dataset macro-F1 and estimated latency (ms) of the appendix table
MMN_ARC = {'iemocap': (.535, 199.0), 'mmfi': (.377, 4004.0), 'czu_mhad': (.485, 1294.0), 'eav': (.647, 319.0),
           'cmi': (.386, 4.4), 'utd_mhad': (.734, 653.0)}


# ── 1. latency estimate ─────────────────────────────────────────────────────────────────────────────────────────────
def latency(method, freq, stages, t):
    """Mean latency (ms) of one fold result: raw extraction of every acquired modality, plus SeMA branch passes and
    readouts, plus selector calls, each weighted by how often it runs."""
    k = freq.sum()                                           # mean number of acquired modalities
    ext = freq @ np.array(t['t_ext_ms'])
    branch = freq @ np.array(t['t_branch_ms'])
    if method == 'SeMARC':           # acquire-and-fuse walk: a readout after each acquisition, a decision per step + STOP
        return ext + t['cal_walk'] * (branch + k * t['t_head_ms']) + t['cal_walk'] * (k + 1) * t['t_sel_ms']
    calls = {'SeMA + EAMA': stages + 1, 'SeMA + Joint-AFA': k + 1, 'SeMA + GMS': 0.0}[method]   # selector calls
    return ext + t['cal_fixed'] * (branch + t['t_head_ms']) + calls * t['t_sel_ms']        # one SeMA pass at the end


# ── 2. means over folds, then over datasets ─────────────────────────────────────────────────────────────────────────
def six_dataset_means(d, tim):
    rows = []
    for r in d.itertuples():
        t = tim[r.dataset]; M = len(t['modalities'])
        freq = np.array([getattr(r, f'freq_{j + 1}') for j in range(M)])
        rows.append(dict(method=r.method, value=r.value, dataset=r.dataset, f1=r.f1, mu=freq.sum() / M,
                         latency_ms=latency(r.method, freq, r.stages, t)))
    per_ds = pd.DataFrame(rows).groupby(['method', 'value', 'dataset'], as_index=False)[['f1', 'mu', 'latency_ms']].mean()
    gms = per_ds[per_ds.method == 'SeMA + GMS'].set_index(['dataset', 'value'])
    scaled = []                                              # GMS: the same % change of every dataset's Table 2 k
    for p in GMS_PCT:
        for ds in DATASETS:
            M = len(tim[ds]['modalities'])
            k = min(M, max(1, math.floor(GMS_K[ds] * (1 + p / 100) + 0.5)))
            scaled.append(dict(method='SeMA + GMS', value=float(p), dataset=ds,
                               **gms.loc[(ds, float(k)), ['f1', 'mu', 'latency_ms']].to_dict()))
    per_ds = pd.concat([per_ds[per_ds.method != 'SeMA + GMS'], pd.DataFrame(scaled)], ignore_index=True)
    full = per_ds.groupby(['method', 'value']).dataset.nunique() == len(DATASETS)
    per_ds = per_ds.set_index(['method', 'value']).loc[full[full].index].reset_index()
    return per_ds.groupby(['method', 'value'], as_index=False)[['f1', 'mu', 'latency_ms']].mean()


# ── 3. Pareto fronts ────────────────────────────────────────────────────────────────────────────────────────────────
def front(c):
    """Points no other point of c beats on both F1 and latency (ties on F1 keep the faster one)."""
    c = c.sort_values(['latency_ms', 'f1'], ascending=[True, False])
    keep, best = [], -np.inf
    for i, r in c.iterrows():
        if r.f1 > best + 1e-12:
            keep.append(i); best = r.f1
    return c.loc[keep]


def pareto_points(means):
    out = []
    for m in KNOB:                                           # legend order
        c = means[means.method == m]
        keep = front(c).index.union(c.index[(c.value == TABLE2[m]) & (m in KEEP_TABLE2)])
        out.append(c.loc[keep].sort_values('latency_ms').assign(knob=KNOB[m], table2_setting=lambda x: x.value == TABLE2[m]))
    f1, ms = (float(np.mean([v[i] for v in MMN_ARC.values()])) for i in (0, 1))
    out.append(pd.DataFrame([dict(method='MultiModN + ARC', knob='', value=np.nan, f1=f1, mu=np.nan, latency_ms=ms,
                                  table2_setting=True)]))
    return pd.concat(out, ignore_index=True)


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────
FS = 11.5                                    # pt; drawn at 246 x 214 pt (0.4\textwidth prints it at about 7.4 pt)
W, H = 246.0, 214.0
L, R, B, T = 40.0, 9.0, 86.0, 6.0            # margins in pt; B holds the ticks, the axis label and the legend
XLIM, YLIM = (270, 1530), (.50, .795)
ZOOM = ((296, 540), (.655, .776))            # region magnified in the inset
INSET = ((830, 1522), (.540, .728))          # where the inset sits, in data units of the main axes
STYLE = {                                    # colour, marker, marker size, line width; drawn in this order
    'SeMA + GMS': ('#eda100', '^', 7.0, 2.2),
    'SeMA + Joint-AFA': ('#1baf7a', 's', 6.4, 2.2),
    'SeMA + EAMA': ('#eb6834', 'o', 6.4, 2.2),
    'SeMARC': ('#2a78d6', 'o', 6.4, 2.9),    # last, on top
}
MMN = dict(ls='', marker='P', ms=10.0, mfc='#898781', mec='white', mew=.8)
LEGEND = ['SeMARC', 'SeMA + Joint-AFA', 'MultiModN + ARC', 'SeMA + EAMA', 'SeMA + GMS']   # 2 columns, filled by column
EDGE = '#555555'


def line_style(m, main):
    col, mk, ms, lw = STYLE[m]
    return dict(color=col, ls='-', lw=lw, marker=mk, ms=ms * (.82 if main else 1.0), mfc=col, mec='white', mew=.7)


def paint(ax, pts, main):
    for m in STYLE:
        p = pts[pts.method == m]
        ax.plot(p.latency_ms, p.f1, zorder=5 if m == 'SeMARC' else 3, solid_joinstyle='round', **line_style(m, main))
    if main:                                                 # MultiModN + ARC only in the main view
        p = pts[pts.method == 'MultiModN + ARC']
        ax.plot(p.latency_ms, p.f1, zorder=4, **MMN)


def draw(pts, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'font.size': FS,
                         'axes.labelsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
                         'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7, 'xtick.major.size': 3,
                         'ytick.major.size': 3, 'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5, 'axes.labelpad': 3,
                         'axes.edgecolor': EDGE, 'xtick.color': EDGE, 'ytick.color': EDGE, 'axes.labelcolor': '#222222',
                         'pdf.fonttype': 42})
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
    paint(ax, pts, main=True)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks([400, 800, 1200])
    ax.set_yticks([.5, .6, .7]); ax.set_yticklabels(['0.5', '0.6', '0.7'])
    ax.grid(color='#e6e5e1', lw=.5); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_xlabel('Estimated GPU latency (ms)'); ax.set_ylabel('Macro-F1')
    (ix0, ix1), (iy0, iy1) = INSET                           # inset: the same curves over ZOOM, no tick labels
    fx = lambda v: (v - XLIM[0]) / (XLIM[1] - XLIM[0])       # noqa: E731
    fy = lambda v: (v - YLIM[0]) / (YLIM[1] - YLIM[0])       # noqa: E731
    ia = ax.inset_axes([fx(ix0), fy(iy0), fx(ix1) - fx(ix0), fy(iy1) - fy(iy0)])
    paint(ia, pts, main=False)
    ia.set_xlim(*ZOOM[0]); ia.set_ylim(*ZOOM[1])
    ia.set_xticks([]); ia.set_yticks([])
    for s in ia.spines.values():
        s.set_linewidth(.7); s.set_color(EDGE)
    (zx0, zx1), (zy0, zy1) = ZOOM                            # dotted zoom box + one connector (the only pairing that
    ax.add_patch(plt.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=EDGE, ls=':', lw=.9, zorder=2))
    ax.plot([zx1, ix0], [zy0, iy0], color=EDGE, ls=':', lw=.9, zorder=2)   # crosses no curve)
    handles = {m: Line2D([], [], label=m, **line_style(m, main=False)) for m in STYLE}
    handles['MultiModN + ARC'] = Line2D([], [], label='MultiModN + ARC', **MMN)
    fig.legend(handles=[handles[m] for m in LEGEND], loc='lower center', bbox_to_anchor=(.5, 0.0), ncol=2,
               frameon=False, handlelength=1.6, handletextpad=.4, columnspacing=.8, labelspacing=.25,
               borderaxespad=.15, borderpad=.1)
    fig.savefig(f'{out}.pdf')
    fig.savefig(f'{out}.png', dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold_results', default=os.path.join(HERE, 'data', 'fold_results.csv'))
    ap.add_argument('--replace', nargs='*', default=[],
                    help="fold-result files (e.g. train_settings.py's output) whose rows replace the same (dataset, method)")
    a = ap.parse_args()
    d = pd.read_csv(a.fold_results)
    for p in a.replace:
        r = pd.read_csv(p)
        keys = set(zip(r.dataset, r.method))
        d = pd.concat([d[[k not in keys for k in zip(d.dataset, d.method)]], r], ignore_index=True)
        print(f'replaced {sorted(keys)} with {len(r)} rows from {p}')
    tim = json.load(open(os.path.join(HERE, 'data', 'timings.json')))
    pts = pareto_points(six_dataset_means(d, tim))
    pts.rename(columns={'value': 'knob_value', 'f1': 'macro_f1'}).to_csv(
        os.path.join(HERE, 'pareto_points.csv'), index=False,
        columns=['method', 'knob', 'knob_value', 'macro_f1', 'latency_ms', 'mu', 'table2_setting'])
    draw(pts, os.path.join(HERE, 'pareto_gamma_controllers'))
    print(pts[['method', 'value', 'f1', 'latency_ms', 'table2_setting']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()

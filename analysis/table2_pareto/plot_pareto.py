#!/usr/bin/env python
"""Macro-F1 vs estimated GPU latency (Table 2's metric) per dataset and for the six-dataset mean.

SeMARC: every (gamma, lambda) configuration of the sweep (mean over the three test folds) as a context dot, and the Pareto
front over them (no configuration has both higher F1 and lower latency) as the line. Gamma sets the modality weights
w_m = (t_m / min t)^gamma, lambda the price of acquiring everything; 'dep' = the deployed per-fold lambda (MM-Fi fold 2 = 1.0).
Other configurations of Table 2 / Appendix tab:comp-replace-complete are drawn as points, with their published F1 and
latency (same latency estimate). Writes fig/pareto_f1_latency.{pdf,png}, res/pareto_points.csv, res/PARETO.md.
"""
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

T = '/files1/haodong/test/new/table2_pareto'
APP = '/files1/haodong/test/new/ICLR/sections/appendix.tex'
ORDER = ['iemocap', 'mmfi', 'czu_mhad', 'eav', 'cmi', 'utd_mhad']              # column order of tab:comp-replace-complete
LABEL = {'iemocap': 'IEMOCAP', 'mmfi': 'MM-Fi', 'czu_mhad': 'CZU-MHAD', 'eav': 'EAV', 'cmi': 'CMI', 'utd_mhad': 'UTD-MHAD',
         'mean': 'Mean of six datasets (Table 2)'}
NAMES = {r'\textbf{\method{}}': 'SeMARC (Table 2, uniform cost)', r'MultiModN + \policy{}': 'MultiModN + ARC',
         'MultiModN': 'MultiModN', r'\backbone{} + EAMA': 'SeMA + EAMA', r'\backbone{} + Joint-AFA': 'SeMA + Joint-AFA',
         r'\backbone{} + GMS': 'SeMA + GMS', r'\method{} (weighted)': 'SeMARC (paper, weighted, γ = .75)',
         r'\backbone{} + EAMA (weighted)': 'SeMA + EAMA (weighted)'}
# ink and the first three categorical slots of the reference palette (validated all-pairs in light mode)
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA, CONTEXT = '#2a78d6', '#eb6834', '#1baf7a', '#c9c8c1'
BLUE_LIGHT = '#86b6ef'      # lighter step of the same blue ramp: SeMARC's uniform-cost lambda path (same method, variant)
STYLE = {   # name -> (colour, marker, filled)
    'SeMARC (Table 2, uniform cost)': (BLUE, '*', True), 'SeMARC (paper, weighted, γ = .75)': (BLUE, 'D', False),
    'SeMA + EAMA': (ORANGE, 'o', True), 'SeMA + EAMA (weighted)': (ORANGE, 'o', False),
    'SeMA + Joint-AFA': (ORANGE, 's', True), 'SeMA + GMS': (ORANGE, '^', True),
    'MultiModN + ARC': (AQUA, 'P', True), 'MultiModN': (AQUA, 'X', True)}


def paper_points():
    t = open(APP).read()
    i = t.index(r'\label{tab:comp-replace-complete}')
    body = t[i:t.index(r'\end{tabular}', i)]
    body = body[body.index(r'\midrule'):]
    out = []
    for row in body.split('\\\\'):
        row = re.sub(r'\\rowcolor\{[^}]*\}|\\arrayrulecolor\{[^}]*\}|\\midrule|\\bottomrule', '', row).strip()
        c = [x.strip() for x in row.split('&')]
        if len(c) != 19:
            continue
        for k, ds in enumerate(ORDER):
            out.append(dict(method=NAMES[c[0]], dataset=ds, f1=float(c[1 + 3 * k]), mu=float(c[2 + 3 * k]),
                            latency_ms=float(c[3 + 3 * k])))
    p = pd.DataFrame(out)
    assert len(p) == 8 * 6, len(p)
    m = p.groupby('method', sort=False)[['f1', 'mu', 'latency_ms']].mean().reset_index().assign(dataset='mean')
    return pd.concat([p, m], ignore_index=True)


def sweep_configs():
    files = glob.glob(f'{T}/res/sweep_*.csv') + glob.glob(f'{T}/res/verify_*.csv')
    d = pd.concat([pd.read_csv(f, dtype={'lam': str}) for f in files])
    d = d.drop_duplicates(['dataset', 'gamma', 'lam', 'fold'])
    n = d.groupby(['dataset', 'gamma', 'lam']).fold.nunique()
    assert (n == 3).all(), n[n != 3]
    c = d.groupby(['dataset', 'gamma', 'lam'])[['f1', 'mu', 'latency_ms', 'extract_ms']].mean().reset_index()
    full = c.groupby(['gamma', 'lam']).dataset.nunique()
    keep = full[full == 6].index
    mean = c.set_index(['gamma', 'lam']).loc[keep].groupby(['gamma', 'lam'])[['f1', 'mu', 'latency_ms', 'extract_ms']] \
        .mean().reset_index().assign(dataset='mean')
    # 'dep' equals lambda = .5 on every fold except MM-Fi fold 2: keep it per dataset only on MM-Fi (no duplicate dots);
    # the mean keeps it, since (gamma = 0, dep) is Table 2's SeMARC and (gamma = .75, dep) the paper's weighted row
    c = c[(c.lam != 'dep') | (c.dataset == 'mmfi')]
    return pd.concat([c, mean], ignore_index=True)


def pareto(c):
    """Configurations no other one beats on both axes (ties on F1 keep the faster one)."""
    c = c.sort_values(['latency_ms', 'f1'], ascending=[True, False])
    best, keep = -np.inf, []
    for i, r in c.iterrows():
        if r.f1 > best + 1e-12:
            keep.append(i)
            best = r.f1
    return c.loc[keep]


def ms(x, _):
    return f'{x:g}' if x >= 1 else f'{x:.1g}'


def log_ticks(ax, lo, hi):
    """Labelled ticks that fit the panel: 1-2-5 per decade, thinned to decades on wide ranges; plain ticks on tiny ranges."""
    t = []
    for subs in ((1, 2, 5), (1, 3), (1,)):                     # densest per-decade set that leaves <= 4 labels
        t = [m * 10.0 ** k for k in range(-1, 6) for m in subs if lo <= m * 10.0 ** k <= hi]
        if len(t) <= 4:
            break
    if len(t) < 2:
        from matplotlib.ticker import MaxNLocator
        t = [x for x in MaxNLocator(4).tick_values(lo, hi) if lo <= x <= hi]
    ax.set_xticks(t)
    ax.xaxis.set_major_formatter(FuncFormatter(ms))
    ax.xaxis.set_minor_formatter(NullFormatter())


def dominance(cfg, pts):
    """For each comparison point: best SeMARC F1 at no more latency, and SeMARC's lowest latency at no less F1."""
    out = []
    for _, p in pts[~pts.method.str.startswith('SeMARC')].iterrows():
        c = cfg[cfg.dataset == p.dataset]
        a = c[c.latency_ms <= p.latency_ms]
        b = c[c.f1 >= p.f1]
        ba = a.loc[a.f1.idxmax()] if len(a) else None
        bb = b.loc[b.latency_ms.idxmin()] if len(b) else None
        out.append(dict(dataset=p.dataset, method=p.method, f1=p.f1, latency_ms=p.latency_ms,
                        semarc_f1_at_le_latency=ba.f1 if ba is not None else np.nan,
                        cfg_a=f'γ={ba.gamma:g}, λ={ba.lam}' if ba is not None else '',
                        lat_a=ba.latency_ms if ba is not None else np.nan,
                        semarc_latency_at_ge_f1=bb.latency_ms if bb is not None else np.nan,
                        cfg_b=f'γ={bb.gamma:g}, λ={bb.lam}' if bb is not None else '',
                        f1_b=bb.f1 if bb is not None else np.nan,
                        dominated=bool(len(c[(c.f1 >= p.f1) & (c.latency_ms <= p.latency_ms)
                                             & ((c.f1 > p.f1) | (c.latency_ms < p.latency_ms))]))))
    return pd.DataFrame(out)


ZOOM = {   # panel -> ((x range ms), (F1 range), inset rectangle in axes fraction): corners where the points crowd
    'czu_mhad': ((1286, 1330), (.893, .926), [.25, .45, .48, .48]),
    'utd_mhad': ((480, 1080), (.74, .87), [.25, .43, .48, .5]),
    'mean': ((320, 960), (.726, .776), [.6, .14, .38, .38])}


def draw(ax, c, fr, p, scale=1.0):
    """SeMARC configurations (context dots), the uniform-cost front, the full front, and the comparison points."""
    ax.scatter(c.latency_ms, c.f1, s=14 * scale, color=CONTEXT, edgecolors='none', zorder=1)
    u = pareto(c[(c.gamma == 0) & (c.lam != 'dep')])             # uniform cost only: front over lambda
    ax.plot(u.latency_ms, u.f1, color=BLUE_LIGHT, lw=1.3, marker='o', ms=3.2 * scale ** .5, mec='white', mew=.6, zorder=2)
    ax.plot(fr.latency_ms, fr.f1, color=BLUE, lw=2 * scale ** .5, zorder=3, solid_joinstyle='round')
    ax.scatter(fr.latency_ms, fr.f1, s=30 * scale, color=BLUE, edgecolors='white', linewidths=1.2 * scale, zorder=4)
    for _, r in p.iterrows():
        col, mk, filled = STYLE[r.method]
        big = (150 if mk == '*' else 62) * scale
        ax.scatter(r.latency_ms, r.f1, s=big, marker=mk, color=col if filled else 'white', edgecolors=col if not filled else 'white',
                   linewidths=(1.6 if not filled else 1.0) * scale ** .5, zorder=6 if r.method.startswith('SeMARC') else 5)


def main():
    pts, cfg = paper_points(), sweep_configs()
    panels = ORDER + ['mean']
    have = [p for p in panels if p in set(cfg.dataset)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': AXIS, 'axes.linewidth': .8,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK2, 'text.color': INK})
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.4))
    axes = axes.ravel()
    rows = []
    for ax, ds in zip(axes, panels):
        if ds not in have:
            ax.set_axis_off()
            continue
        c = cfg[cfg.dataset == ds]
        fr = pareto(c)
        p = pts[pts.dataset == ds]
        draw(ax, c, fr, p)
        ax.set_xscale('log')
        xs = np.r_[c.latency_ms.values, pts[pts.dataset == ds].latency_ms.values]
        lo, hi = xs.min() / 1.25, xs.max() * 1.25
        ax.set_xlim(lo, hi)
        log_ticks(ax, lo, hi)
        ax.grid(True, which='major', color=GRID, lw=.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_title(LABEL[ds], fontsize=10.5, color=INK, loc='left', fontweight='bold')
        ax.set_xlabel('Estimated GPU latency (ms, log)')
        ax.set_ylabel('Macro-F1')
        if ds in ZOOM:                                             # crowded corner: same marks, zoomed, linear axes
            (x0, x1), (y0, y1), rect = ZOOM[ds]
            ins = ax.inset_axes(rect)
            draw(ins, c, fr, p, scale=.62)
            ins.set_xlim(x0, x1)
            ins.set_ylim(y0, y1)
            ins.tick_params(labelsize=7, length=2, pad=1.5, colors=INK2)
            ins.xaxis.set_major_locator(plt.MaxNLocator(3))
            ins.yaxis.set_major_locator(plt.MaxNLocator(3))
            ins.grid(True, color=GRID, lw=.5, zorder=0)
            ins.set_axisbelow(True)
            ins.set_facecolor('white')
            for s in ins.spines.values():
                s.set_edgecolor(AXIS)
            ax.indicate_inset_zoom(ins, edgecolor=MUTED, lw=.8, alpha=1)
        for _, r in c.iterrows():
            rows.append(dict(dataset=ds, method='SeMARC sweep', gamma=r.gamma, lam=r.lam, f1=r.f1, mu=r.mu,
                             latency_ms=r.latency_ms, pareto=int(r.name in fr.index)))
        for _, r in pts[pts.dataset == ds].iterrows():
            rows.append(dict(dataset=ds, method=r.method, gamma=np.nan, lam='', f1=r.f1, mu=r.mu, latency_ms=r.latency_ms, pareto=0))
    handles = [Line2D([], [], color=BLUE, lw=2, marker='o', ms=5, mec='white', label='SeMARC Pareto front over (γ, λ)'),
               Line2D([], [], color=BLUE_LIGHT, lw=1.3, marker='o', ms=3.5, mec='white', label='SeMARC front, uniform cost only (γ = 0)'),
               Line2D([], [], color=CONTEXT, lw=0, marker='o', ms=5, label='SeMARC, every (γ, λ)')]
    for name, (col, mk, filled) in STYLE.items():
        handles.append(Line2D([], [], lw=0, marker=mk, ms=11 if mk == '*' else 7.5, color=col if filled else 'white',
                              mec=col if not filled else 'white', mew=1.4 if not filled else .8, label=name))
    lax = axes[len(panels)]
    lax.set_axis_off()
    lax.legend(handles=handles, loc='center left', frameon=False, fontsize=9.2, handletextpad=.5, labelspacing=.75,
               borderaxespad=0)
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{T}/fig/pareto_f1_latency.{ext}', dpi=220, bbox_inches='tight')
    out = pd.DataFrame(rows)
    out.to_csv(f'{T}/res/pareto_points.csv', index=False)
    dom = dominance(cfg, pts)
    dom.to_csv(f'{T}/res/dominance.csv', index=False)
    write_md(cfg, pts, dom, panels)
    return out, dom


def write_md(cfg, pts, dom, panels):
    f3 = lambda x: '–' if pd.isna(x) else f'{x:.3f}'
    lat = lambda x: '–' if pd.isna(x) else (f'{x:.1f}' if x < 100 else f'{x:.0f}')
    L = ['# SeMARC F1–latency Pareto fronts (Table 2 metric)', '',
         'Latency: estimated GPU latency in ms (Appendix comp-replace). F1: macro-F1, mean over three test folds. '
         'Front: SeMARC configurations (γ, λ) not beaten on both F1 and latency by another configuration; '
         "λ = dep is the deployed per-fold λ (differs from .5 only on MM-Fi fold 2). Points: Appendix tab:comp-replace-complete.", '']
    for ds in panels:
        c = cfg[cfg.dataset == ds]
        if not len(c):
            continue
        fr = pareto(c)
        L += [f'## {LABEL[ds]}', '', f'{len(c)} configurations, {len(fr)} on the front.', '',
              '| γ | λ | F1 | latency (ms) | MU |', '|---|---|---|---|---|']
        L += [f'| {r.gamma:g} | {r.lam} | {f3(r.f1)} | {lat(r.latency_ms)} | {r.mu:.2f} |' for _, r in fr.iterrows()]
        d = dom[dom.dataset == ds]
        L += ['', '| Configuration | F1 | latency | SeMARC best F1 at ≤ latency (config) | SeMARC latency for ≥ F1 (config) | dominated |',
              '|---|---|---|---|---|---|']
        L += [f'| {r.method} | {f3(r.f1)} | {lat(r.latency_ms)} | {f3(r.semarc_f1_at_le_latency)} ({r.cfg_a or "none"}) | '
              f'{lat(r.semarc_latency_at_ge_f1)} ({r.cfg_b or "none"}) | {"yes" if r.dominated else "no"} |' for _, r in d.iterrows()]
        L.append('')
    open(f'{T}/res/PARETO.md', 'w').write('\n'.join(L))


if __name__ == '__main__':
    o, dom = main()
    print(o[o.pareto == 1].to_string(index=False))
    pd.set_option('display.width', 250)
    print(dom.round(3).to_string(index=False))

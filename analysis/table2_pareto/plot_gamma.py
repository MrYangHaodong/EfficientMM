#!/usr/bin/env python
"""Macro-F1 vs estimated GPU latency (Table 2's metric), SeMARC with lambda held at Table 2's value and only the
modality-weight exponent gamma varied: w_m = (t_m / min t)^gamma, gamma in {0, .25, .5, .75, 1, 1.5, 2}.

The SeMARC points are the paper's own gamma study at the deployed lambda (Appendix tab:wgamma-alldata; lambda = .5,
MM-Fi fold 2 = 1.0), re-evaluated from the copied nets, identical to gamma_ablation_samelam_perfold.csv (verify.py).
gamma = 0 is Table 2's SeMARC and gamma = .75 the paper's weighted row. The line is the Pareto front over gamma.
Other configurations of Table 2 / tab:comp-replace-complete are points with their published values.
Writes fig/pareto_gamma_f1_latency.{pdf,png}, res/gamma_points.csv, res/dominance_gamma.csv, res/PARETO_gamma.md.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import plot_pareto as pp
from plot_pareto import T, ORDER, LABEL, STYLE, BLUE, BLUE_LIGHT, INK, INK2, MUTED, GRID, AXIS

LEG = {'SeMARC (Table 2, uniform cost)': 'SeMARC, Table 2 (uniform cost, γ = 0)',
       'SeMARC (paper, weighted, γ = .75)': "SeMARC, paper's weighted row (γ = .75)"}
ZOOM = {   # panel -> (x range ms, F1 range, inset rectangle in axes fraction, log x): corners where the points crowd
    'mmfi': ((50, 330), (.941, .957), [.21, .13, .5, .5], True),
    'czu_mhad': ((1293, 1322), (.895, .928), [.2, .15, .62, .55], False),
    'cmi': ((9.8, 12.9), (.5445, .5585), [.29, .07, .5, .46], False),
    'utd_mhad': ((505, 680), (.782, .833), [.5, .2, .4, .55], False),
    'mean': ((335, 545), (.738, .771), [.21, .1, .52, .55], False)}
YRIGHT = {'utd_mhad'}      # inset F1 ticks on the right: on the left they would sit beside SeMA + EAMA and read as its label
YTICKS = {'cmi': [.55, .555]}   # no tick at the inset's bottom edge, beside the MultiModN + ARC mark
S_DOT, S_OTHER, S_BIG = 30, 24, 62


def gamma_configs():
    """Mean over the three test folds of every (dataset, gamma) at the deployed lambda; 'mean' averages the datasets."""
    d = pd.concat([pd.read_csv(f, dtype={'lam': str}) for f in sorted(glob.glob(f'{T}/res/verify_*.csv'))])
    d = d[d.lam == 'dep'].drop_duplicates(['dataset', 'gamma', 'fold'])
    n = d.groupby(['dataset', 'gamma']).fold.nunique()
    assert len(n) == 6 * 7 and (n == 3).all(), n
    k = ['f1', 'mu', 'latency_ms', 'n_acq']
    c = d.groupby(['dataset', 'gamma'])[k].mean().reset_index()
    m = c.groupby('gamma')[k].mean().reset_index().assign(dataset='mean')
    return pd.concat([c, m], ignore_index=True).assign(lam='dep')


def check_published(cfg, pts):
    """gamma = 0 and .75 must sit on the published SeMARC and weighted-SeMARC points (rounding only)."""
    out = []
    for name, g in (('SeMARC (Table 2, uniform cost)', 0), ('SeMARC (paper, weighted, γ = .75)', .75)):
        for ds in ORDER + ['mean']:
            p = pts[(pts.method == name) & (pts.dataset == ds)].iloc[0]
            c = cfg[(cfg.dataset == ds) & (cfg.gamma == g)].iloc[0]
            out.append((name, ds, c.f1 - p.f1, (c.latency_ms - p.latency_ms) / p.latency_ms))
    o = pd.DataFrame(out, columns=['method', 'dataset', 'd_f1', 'rel_d_latency'])
    assert (o.d_f1.abs() <= 6e-4).all() and (o.rel_d_latency.abs() <= 3e-3).all(), o
    return o


def glabel(g):
    s = f'{g:g}'
    return s[1:] if s.startswith('0.') else s


def draw(ax, c, fr, p, scale=1.0):
    """Gamma points (front dark, beaten light), the front line, and the published points on top."""
    other = c[~c.index.isin(fr.index)]
    ax.scatter(other.latency_ms, other.f1, s=S_OTHER * scale, color=BLUE_LIGHT, edgecolors='white', linewidths=.8 * scale,
               zorder=3)
    ax.plot(fr.latency_ms, fr.f1, color=BLUE, lw=2 * scale ** .5, zorder=3, solid_joinstyle='round')
    ax.scatter(fr.latency_ms, fr.f1, s=S_DOT * scale, color=BLUE, edgecolors='white', linewidths=1.2 * scale, zorder=4)
    for _, r in p.iterrows():
        col, mk, filled = STYLE[r.method]
        big = (150 if mk == '*' else S_BIG) * scale
        ax.scatter(r.latency_ms, r.f1, s=big, marker=mk, color=col if filled else 'white', edgecolors=col if not filled else 'white',
                   linewidths=(1.6 if not filled else 1.0) * scale ** .5, zorder=6 if r.method.startswith('SeMARC') else 5)


def obstacles(ax, c, fr, p, scale=1.0):
    """Marks in display pixels as circles (x, y, radius), plus samples along the front line."""
    px = ax.figure.dpi / 72
    tr = ax.transData.transform
    ob = []
    for _, r in c.iterrows():
        x, y = tr((r.latency_ms, r.f1))
        ob.append((x, y, (np.sqrt(S_DOT * scale) / 2 + 1) * px))
    for _, r in p.iterrows():
        x, y = tr((r.latency_ms, r.f1))
        ob.append((x, y, (np.sqrt((150 if STYLE[r.method][1] == '*' else S_BIG) * scale) / 2 + 1) * px))
    xy = tr(np.c_[fr.latency_ms.values, fr.f1.values])
    for a, b in zip(xy[:-1], xy[1:]):
        for t in np.linspace(0, 1, 25):
            ob.append((*(a + t * (b - a)), 1.4 * px))
    return ob


def hits(bb, cx, cy, r):
    nx, ny = min(max(cx, bb.x0), bb.x1), min(max(cy, bb.y0), bb.y1)
    return (nx - cx) ** 2 + (ny - cy) ** 2 < r * r


def label_gammas(ax, c, ob, avoid, fs, merge_px=7):
    """Label each gamma point (points closer than merge_px share one label, e.g. '1, 1.5') at the first of eight
    offsets that stays inside the axes and clear of marks, earlier labels and the avoid boxes; else the least bad."""
    fig = ax.figure
    rd = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(rd)
    px = fig.dpi / 72
    xy = ax.transData.transform(np.c_[c.latency_ms.values, c.f1.values])
    g = c.gamma.values
    lab = -np.ones(len(c), int)
    for i in range(len(c)):                                    # single-link clusters in pixel space
        if lab[i] < 0:
            lab[i], stack = i, [i]
            while stack:
                j = stack.pop()
                for k in np.where((lab < 0) & (np.hypot(*(xy - xy[j]).T) < merge_px))[0]:
                    lab[k] = i
                    stack.append(k)
    placed = []
    for cl in np.unique(lab):
        idx = np.where(lab == cl)[0]
        cx, cy = xy[idx].mean(0)
        own = [o for o in ob if min(np.hypot(o[0] - xy[i][0], o[1] - xy[i][1]) for i in idx) < 1]
        rad = max(np.hypot(o[0] - cx, o[1] - cy) + o[2] for o in own) if own else 4 * px
        d = rad / px + 1.2
        text = ', '.join(glabel(v) for v in sorted(g[idx]))
        x, y = ax.transData.inverted().transform((cx, cy))
        cands = [(d, 0, 'left', 'center'), (-d, 0, 'right', 'center'), (0, d, 'center', 'bottom'), (0, -d, 'center', 'top'),
                 (.72 * d, .72 * d, 'left', 'bottom'), (-.72 * d, .72 * d, 'right', 'bottom'),
                 (.72 * d, -.72 * d, 'left', 'top'), (-.72 * d, -.72 * d, 'right', 'top')]
        best = None
        for dx, dy, ha, va in cands:
            a = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=fs,
                            color=INK2, zorder=8, annotation_clip=False)
            b = a.get_window_extent(rd)
            b = Bbox.from_extents(b.x0 - 1, b.y0 - 1, b.x1 + 1, b.y1 + 1)
            cost = 1000 * (b.x0 < axbb.x0 or b.x1 > axbb.x1 or b.y0 < axbb.y0 or b.y1 > axbb.y1)
            cost += 100 * sum(b.overlaps(o) for o in placed + avoid) + 10 * sum(hits(b, *o) for o in ob)
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, a, b)
            else:
                a.remove()
            if cost == 0:
                break
        placed.append(best[2])
    return placed


def style_axes(ax, ds, lo, hi):
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    pp.log_ticks(ax, lo, hi)
    ax.grid(True, which='major', color=GRID, lw=.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title(LABEL[ds], fontsize=10.5, color=INK, loc='left', fontweight='bold')
    ax.set_xlabel('Estimated GPU latency (ms, log)')
    ax.set_ylabel('Macro-F1')


def make_inset(ax, ds, c, fr, p, rect=None, scale=.62, tick_fs=7):
    (x0, x1), (y0, y1), rect0, logx = ZOOM[ds]
    ins = ax.inset_axes(rect or rect0)
    draw(ins, c, fr, p, scale=scale)
    if logx:
        ins.set_xscale('log')
        ins.set_xlim(x0, x1)
        pp.log_ticks(ins, x0, x1)
    else:
        ins.set_xlim(x0, x1)
        ins.xaxis.set_major_locator(plt.MaxNLocator(3))
    ins.set_ylim(y0, y1)
    ins.yaxis.set_major_locator(plt.FixedLocator(YTICKS[ds]) if ds in YTICKS else plt.MaxNLocator(3))
    if ds in YRIGHT:
        ins.yaxis.tick_right()
    ins.tick_params(labelsize=tick_fs, length=2, pad=1.5, colors=INK2)
    ins.grid(True, color=GRID, lw=.5, zorder=0)
    ins.set_axisbelow(True)
    ins.set_facecolor('white')
    for s in ins.spines.values():
        s.set_edgecolor(AXIS)
    ax.indicate_inset_zoom(ins, edgecolor=MUTED, lw=.8, alpha=1)
    return ins


NOTE = ("SeMARC at Table 2's λ (.5; MM-Fi fold 2: 1),\nonly γ varied: w$_m$ = (t$_m$ / min t)$^γ$.\n"
        'Numbers next to the blue dots give γ.')
RC = {'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': AXIS, 'axes.linewidth': .8,
      'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK2, 'text.color': INK}


def legend_handles(beaten=True, drop=()):
    h = [Line2D([], [], color=BLUE, lw=2, marker='o', ms=5, mec='white', label='SeMARC, Pareto front over γ')]
    if beaten:
        h.append(Line2D([], [], color=BLUE_LIGHT, lw=0, marker='o', ms=5, mec='white', label='SeMARC, γ beaten by another γ'))
    for name, (col, mk, filled) in STYLE.items():
        if name not in drop:
            h.append(Line2D([], [], lw=0, marker=mk, ms=11 if mk == '*' else 7.5, color=col if filled else 'white',
                            mec=col if not filled else 'white', mew=1.4 if not filled else .8, label=LEG.get(name, name)))
    return h


def main_mean():
    """The six-dataset mean on its own (the seventh panel of the full figure), legend to the right. As asked by the
    user, only the front's gammas are drawn (no 'beaten' dots) and MultiModN (without ARC) is left out.
    Writes fig/pareto_gamma_mean.{pdf,png}."""
    pts, cfg = pp.paper_points(), gamma_configs()
    ds = 'mean'
    p = pts[(pts.dataset == ds) & (pts.method != 'MultiModN')]
    fr = pp.pareto(cfg[cfg.dataset == ds])
    c = fr
    plt.rcParams.update(RC)
    fig, (ax, lax) = plt.subplots(1, 2, figsize=(7.6, 3.7), gridspec_kw=dict(width_ratios=[1.5, 1]))
    draw(ax, c, fr, p)
    xs = np.r_[c.latency_ms.values, p.latency_ms.values]
    style_axes(ax, ds, xs.min() / 1.25, xs.max() * 1.25)
    ax.set_xticks([300, 500, 700, 1000])                       # the 1-2-5 rule leaves only 500 and 1000 on this range
    sc = .85
    ins = make_inset(ax, ds, c, fr, p, rect=[.14, .16, .62, .57], scale=sc, tick_fs=8)
    lax.set_axis_off()
    leg = lax.legend(handles=legend_handles(beaten=False, drop={'MultiModN'}), loc='upper left', bbox_to_anchor=(0, 1.0),
                     frameon=False, fontsize=9, handletextpad=.5, labelspacing=.62, borderaxespad=0)
    fig.tight_layout(w_pad=.6)
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    y_leg = lax.transAxes.inverted().transform((0, leg.get_window_extent(rd).y0))[1]
    lax.text(0, y_leg - .04, NOTE, transform=lax.transAxes, fontsize=8.4, color=INK2, va='top')
    (x0, x1), (y0, y1), _, _ = ZOOM[ds]
    inside = c.latency_ms.between(x0, x1) & c.f1.between(y0, y1)
    src = Bbox(ax.transData.transform([(x0, y0), (x1, y1)]))
    label_gammas(ax, c[~inside], obstacles(ax, c, fr, p), [ins.get_window_extent(rd), src], fs=8.4)
    label_gammas(ins, c[inside], obstacles(ins, c, fr, p, scale=sc), [], fs=8, merge_px=6)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{T}/fig/pareto_gamma_mean.{ext}', dpi=250, bbox_inches='tight')


def main():
    pts, cfg = pp.paper_points(), gamma_configs()
    chk = check_published(cfg, pts)
    panels = ORDER + ['mean']
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.4))
    axes = axes.ravel()
    rows, todo = [], []
    for ax, ds in zip(axes, panels):
        c = cfg[cfg.dataset == ds]
        fr = pp.pareto(c)
        p = pts[pts.dataset == ds]
        draw(ax, c, fr, p)
        xs = np.r_[c.latency_ms.values, p.latency_ms.values]
        style_axes(ax, ds, xs.min() / 1.25, xs.max() * 1.25)
        ins = make_inset(ax, ds, c, fr, p) if ds in ZOOM else None
        todo.append((ax, ds, c, fr, p, ins))
        for _, r in c.iterrows():
            rows.append(dict(dataset=ds, method='SeMARC', gamma=r.gamma, lam='Table 2 (dep)', f1=r.f1, mu=r.mu,
                             latency_ms=r.latency_ms, modalities=r.n_acq, pareto_over_gamma=int(r.name in fr.index)))
        for _, r in p.iterrows():
            rows.append(dict(dataset=ds, method=r.method, gamma=np.nan, lam='', f1=r.f1, mu=r.mu, latency_ms=r.latency_ms,
                             modalities=np.nan, pareto_over_gamma=np.nan))
    lax = axes[len(panels)]
    lax.set_axis_off()
    lax.legend(handles=legend_handles(), loc='upper left', bbox_to_anchor=(0, 1.02), frameon=False, fontsize=9.2,
               handletextpad=.5, labelspacing=.72, borderaxespad=0)
    lax.text(0, -.04, NOTE, transform=lax.transAxes, fontsize=8.6, color=INK2, va='top')
    fig.tight_layout(w_pad=1.6, h_pad=1.4)
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    for ax, ds, c, fr, p, ins in todo:
        if ins is None:
            label_gammas(ax, c, obstacles(ax, c, fr, p), [], fs=7.6)
            continue
        (x0, x1), (y0, y1), _, _ = ZOOM[ds]
        inside = c.latency_ms.between(x0, x1) & c.f1.between(y0, y1)
        src = Bbox(ax.transData.transform([(x0, y0), (x1, y1)]))
        label_gammas(ax, c[~inside], obstacles(ax, c, fr, p), [ins.get_window_extent(rd), src], fs=7.6)
        label_gammas(ins, c[inside], obstacles(ins, c, fr, p, scale=.62), [], fs=6.6, merge_px=5)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{T}/fig/pareto_gamma_f1_latency.{ext}', dpi=220, bbox_inches='tight')
    out = pd.DataFrame(rows)
    out.to_csv(f'{T}/res/gamma_points.csv', index=False)
    dom = pp.dominance(cfg, pts)
    dom.to_csv(f'{T}/res/dominance_gamma.csv', index=False)
    write_md(cfg, pts, dom, panels, chk)
    return cfg, dom, chk


def write_md(cfg, pts, dom, panels, chk):
    f3 = lambda x: '–' if pd.isna(x) else f'{x:.3f}'
    lat = lambda x: '–' if pd.isna(x) else (f'{x:.1f}' if x < 100 else f'{x:.0f}')
    ds_cols = ORDER + ['mean']
    L = ["# SeMARC over γ at Table 2's λ: F1–latency (Table 2 metric)", '',
         "SeMARC with λ held at Table 2's value (the deployed per-fold λ: .5, MM-Fi fold 2 = 1.0) and only γ varied "
         '(w_m = (t_m / min t)^γ). These are the paper\'s own γ-study nets (Appendix tab:wgamma-alldata), reproduced exactly '
         '(verify.py). γ = 0 is Table 2\'s SeMARC; γ = .75 is the paper\'s weighted row. Latency: estimated GPU latency in ms. '
         'F1: macro-F1, mean over three test folds. Comparison points: Appendix tab:comp-replace-complete (mean = average '
         'of the six datasets).', '',
         f'Published points vs γ = 0 / .75 here: max |ΔF1| = {chk.d_f1.abs().max():.4f}, max relative |Δlatency| = '
         f'{chk.rel_d_latency.abs().max():.2%} (rounding of the published values).', '',
         '## Is the configuration beaten by SeMARC at some γ (F1 ≥ and latency ≤, one strictly)?', '',
         '| Configuration | ' + ' | '.join(LABEL[d].replace(' (Table 2)', '') for d in ds_cols) + ' |',
         '|---|' + '---|' * len(ds_cols)]
    for meth in dom.method.unique():
        cells = []
        for d in ds_cols:
            r = dom[(dom.method == meth) & (dom.dataset == d)].iloc[0]
            c = cfg[(cfg.dataset == d) & (cfg.f1 >= r.f1) & (cfg.latency_ms <= r.latency_ms)
                    & ((cfg.f1 > r.f1) | (cfg.latency_ms < r.latency_ms))]
            cells.append(('yes (γ ' + ', '.join(glabel(v) for v in sorted(c.gamma)) + ')') if r.dominated else 'no')
        L.append(f'| {meth} | ' + ' | '.join(cells) + ' |')
    L.append('')
    for ds in panels:
        c = cfg[cfg.dataset == ds]
        fr = pp.pareto(c)
        L += [f'## {LABEL[ds]}', '', '| γ | F1 | latency (ms) | MU | modalities / sample | on the front |', '|---|---|---|---|---|---|']
        L += [f'| {glabel(r.gamma)} | {f3(r.f1)} | {lat(r.latency_ms)} | {r.mu:.2f} | {r.n_acq:.2f} | '
              f'{"yes" if r.name in fr.index else ""} |' for _, r in c.iterrows()]
        d = dom[dom.dataset == ds]
        L += ['', '| Configuration | F1 | latency | SeMARC best F1 at ≤ latency (γ) | SeMARC latency for ≥ F1 (γ) | beaten |',
              '|---|---|---|---|---|---|']
        L += [f'| {r.method} | {f3(r.f1)} | {lat(r.latency_ms)} | {f3(r.semarc_f1_at_le_latency)} '
              f'({r.cfg_a.split(",")[0] if r.cfg_a else "none"}) | {lat(r.semarc_latency_at_ge_f1)} '
              f'({r.cfg_b.split(",")[0] if r.cfg_b else "none"}) | {"yes" if r.dominated else "no"} |' for _, r in d.iterrows()]
        L.append('')
    open(f'{T}/res/PARETO_gamma.md', 'w').write('\n'.join(L))


if __name__ == '__main__':
    cfg, dom, chk = main()
    main_mean()
    pd.set_option('display.width', 250)
    print(chk.round(4).to_string(index=False))
    print(dom[['dataset', 'method', 'f1', 'latency_ms', 'semarc_f1_at_le_latency', 'cfg_a', 'semarc_latency_at_ge_f1',
               'cfg_b', 'dominated']].round(3).to_string(index=False))

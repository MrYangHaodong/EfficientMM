#!/usr/bin/env python
"""The Table 2 controller Pareto figure (fig/pareto_gamma_controllers_mean: full range + zoom, two panels) as ONE plot,
the way each panel of the paper's Figure 5 shows its dense cluster: the whole range on a linear latency axis, the
cluster (296-540 ms) magnified in an inset joined to a dotted box by a connector line.

Same data as gamma_ctrl.py (its load()/line_pts(): six-dataset means; each curve is the method's Pareto front over its
own knob, shared by all datasets). Kept clean for the reviewers' comments on Figures 4/5/8-10 (cluttered, small and
mixed text): one font size; no title, explanatory paragraph or per-point knob labels (the knobs go in the caption); no
tick numbers in the inset (as in the revised Figure 5); only the four curves and MultiModN + ARC (the user removed the
Table 2 and weighted-cost markers); the legend is one row of method names under the plot, in the style of the revised
Figure 9 (firstpick_cascade_iemocap2). Drawn at the paper's full text width (396 pt), so the font size is the printed
size. Writes fig/pareto_gamma_controllers_single.{pdf,png}.
"""
import re
import subprocess
import sys
import zlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.argv = sys.argv[:1]                                     # gamma_ctrl reads argv only in its __main__
import gamma_ctrl as G
import plot_pareto as pp
import ctrl_curves as cc
from ctrl_curves import EAMA, JAFA, GMSN, SEMARC

OUT = f'{cc.T}/fig/pareto_gamma_controllers_single'
FS = 11.5                                                   # pt at the drawn size
W, H = 246.0, 214.0                                         # drawn size (pt): text = 4.7% of the width, as large relative to
#                                                             the figure as the 2-column legend allows (revised Figure 9:
#                                                             7 pt in 125 pt = 5.6%). At 0.4\textwidth it prints at 7.4 pt.
INK = '#222222'
LABEL = {SEMARC: 'SeMARC', EAMA: 'SeMA + EAMA', JAFA: 'SeMA + Joint-AFA', GMSN: 'SeMA + GMS'}
XLIM, YLIM = (270, 1530), (.50, .795)
ZOOM = ((296, 540), (.655, .776))                           # the cluster magnified in the inset
INSET = ((830, 1522), (.540, .728))                         # where the inset sits, in data units of the main axes:
#                                                             right of GMS's column, under Joint-AFA, above MultiModN


def style(m):
    col, ls, mk, _, _ = cc.CURVE[m]
    return dict(color=col, ls=ls, lw=2.9 if m == SEMARC else 2.2, marker=mk, ms=6.4 if mk != '^' else 7.0, mfc=col,
                mec='white', mew=.7)


MMN = dict(ls='', marker='P', ms=10.0, mfc=cc.GRAY, mec='white', mew=.8)


def paint(ax, a, mmn, main=True):
    for m in G.ORDER_G:                                      # SeMARC drawn last (on top)
        fr = G.line_pts(a, m, 'mean')
        st = style(m)
        if main:                                             # the main view's cluster is small: slightly smaller dots
            st['ms'] *= .82
        ax.plot(fr.latency_ms, fr.f1, zorder=5 if m == SEMARC else 3, solid_joinstyle='round', **st)
    if main:
        ax.plot(mmn.latency_ms, mmn.f1, zorder=4, **MMN)


def main():
    a, chk, have = G.load()
    assert len(have) == 6, have
    pts = pp.paper_points()
    mmn = pts[(pts.dataset == 'mean') & (pts.method == 'MultiModN + ARC')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans', 'font.size': FS,
                         'axes.labelsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
                         'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7, 'xtick.major.size': 3,
                         'ytick.major.size': 3, 'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5, 'axes.labelpad': 3,
                         'axes.edgecolor': '#555555', 'xtick.color': '#555555', 'ytick.color': '#555555',
                         'axes.labelcolor': INK, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(W / 72, H / 72))
    L, R, B, T = 40.0, 9.0, 86.0, 6.0                         # pt; B holds ticks, axis label and the three legend rows
    ax = fig.add_axes([L / W, B / H, (W - L - R) / W, (H - B - T) / H])
    paint(ax, a, mmn)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xticks([400, 800, 1200])                           # fewer, larger tick labels
    ax.set_yticks([.5, .6, .7]); ax.set_yticklabels(['0.5', '0.6', '0.7'])
    ax.grid(color='#e6e5e1', lw=.5); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_xlabel('Estimated GPU latency (ms)'); ax.set_ylabel('Macro-F1')
    (ix0, ix1), (iy0, iy1) = INSET
    fx = lambda v: (v - XLIM[0]) / (XLIM[1] - XLIM[0])        # noqa: E731
    fy = lambda v: (v - YLIM[0]) / (YLIM[1] - YLIM[0])        # noqa: E731
    ia = ax.inset_axes([fx(ix0), fy(iy0), fx(ix1) - fx(ix0), fy(iy1) - fy(iy0)])
    paint(ia, a, mmn, main=False)
    ia.set_xlim(*ZOOM[0]); ia.set_ylim(*ZOOM[1])
    ia.set_xticks([]); ia.set_yticks([])
    for s in ia.spines.values():
        s.set_linewidth(.7); s.set_color('#555555')
    # zoom box + one connector, drawn by hand: every other corner pairing crosses the GMS column (753-772 ms) or the
    # Joint-AFA line; box lower-right -> inset lower-left passes below GMS's lowest point (0.649)
    (zx0, zx1), (zy0, zy1) = ZOOM
    ax.add_patch(plt.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec='#555555', ls=':', lw=.9, zorder=2))
    ax.plot([zx1, ix0], [zy0, iy0], color='#555555', ls=':', lw=.9, zorder=2)
    handles = [Line2D([], [], label=LABEL[m], **style(m)) for m in (SEMARC, EAMA, JAFA, GMSN)]
    handles.append(Line2D([], [], label='MultiModN + ARC', **MMN))
    # two columns under the plot; matplotlib fills columns first, so this order reads row by row
    # SeMARC, SeMA + EAMA / SeMA + Joint-AFA, SeMA + GMS / MultiModN + ARC (the two longest names share a column)
    handles = [handles[i] for i in (0, 2, 4, 1, 3)]
    leg = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 0.0), ncol=2, frameon=False,
                     handlelength=1.6, handletextpad=.4, columnspacing=.8, labelspacing=.25, borderaxespad=.15,
                     borderpad=.1)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    worst = max(max(-e.x0, -e.y0, e.x1 - fig.bbox.width, e.y1 - fig.bbox.height) for e in boxes + [leg.get_window_extent(r)])
    assert worst * 72 / fig.dpi < .3, f'text spills {worst * 72 / fig.dpi:.1f} pt off the page'
    gap = (ax.xaxis.label.get_window_extent(r).y0 - leg.get_window_extent(r).y1) * 72 / fig.dpi
    assert gap > 1, f'legend touches the axis label ({gap:.1f} pt)'
    fig.savefig(f'{OUT}.pdf')
    fig.savefig(f'{OUT}.png', dpi=600)
    plt.close(fig)
    sizes = {}
    for raw in re.findall(rb'stream\r?\n(.*?)\r?\nendstream', open(f'{OUT}.pdf', 'rb').read(), re.S):
        try:
            for f in re.findall(rb'/F\d+\s+([\d.]+)\s+Tf', zlib.decompress(raw)):
                sizes[float(f)] = sizes.get(float(f), 0) + 1
        except zlib.error:
            pass
    pg = subprocess.run(['pdfinfo', f'{OUT}.pdf'], capture_output=True, text=True).stdout
    print(OUT, re.search(r'Page size:.*', pg).group(0), '| font sizes (pt):', dict(sorted(sizes.items())),
          f'| legend-to-label gap {gap:.1f} pt')


if __name__ == '__main__':
    main()

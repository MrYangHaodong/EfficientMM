#!/usr/bin/env python
"""The numbers in fig/pareto_gamma_controllers_single.pdf (pareto_single.py): one row per plotted point, in legend order
and, within a method, by latency. Writes res/pareto_gamma_controllers_single.csv.

Columns:
  method          legend name
  knob            what varies along the curve: gamma (SeMARC, SeMA + EAMA), r_cost (SeMA + Joint-AFA), k_change_pct
                  (SeMA + GMS: every dataset's Table 2 k changed by this %, rounded half up, clipped to [1, M])
  knob_value
  macro_f1        six-dataset mean, the plotted y
  latency_ms      six-dataset mean of Table 2's GPU latency estimate, the plotted x
  in_inset        the point is also drawn in the magnified inset
  table2_setting  the knob value Table 2 uses; MultiModN + ARC is Table 2's own row (appendix tab:comp-replace-complete)

Before writing, every row is checked against the PDF itself. Marker positions are read back from the page's content
stream and mapped to data units through pareto_single.py's axes geometry. Each row must have a marker at its main-view
position (and at its inset position when in_inset), and every curve marker inside the plot must belong to a row.
"""
import re
import sys
import zlib
import numpy as np
import pandas as pd

sys.argv = sys.argv[:1]
import gamma_ctrl as G
import plot_pareto as pp
import ctrl_curves as cc
import pareto_single as S
from ctrl_curves import EAMA, JAFA, GMSN, SEMARC

OUT = f'{cc.T}/res/pareto_gamma_controllers_single.csv'
PDF = f'{S.OUT}.pdf'
KNOB = {SEMARC: 'gamma', EAMA: 'gamma', JAFA: 'r_cost', GMSN: 'k_change_pct'}
T2 = {SEMARC: 0.0, EAMA: 0.0, JAFA: 0.05, GMSN: 0.0}           # Table 2's knob value (GMS: k unchanged)
MMN = 'MultiModN + ARC'
TOL = 1e-3                                                     # pt


def points():
    a, _, have = G.load()
    assert len(have) == 6, have
    rows = []
    for m in (SEMARC, EAMA, JAFA, GMSN):                       # legend order, read row by row
        for r in G.line_pts(a, m, 'mean').itertuples():        # exactly what pareto_single.paint() draws
            v = (r.value - 1) * 100 if m == GMSN else r.value
            rows.append(dict(method=S.LABEL[m], knob=KNOB[m], knob_value=v, macro_f1=r.f1, latency_ms=r.latency_ms,
                             table2_setting=abs(v - T2[m]) < 1e-9))
    p = pp.paper_points()
    mm = p[(p.dataset == 'mean') & (p.method == MMN)]
    assert len(mm) == 1, mm
    rows.append(dict(method=MMN, knob='', knob_value=np.nan, macro_f1=mm.f1.iloc[0], latency_ms=mm.latency_ms.iloc[0],
                     table2_setting=True))
    d = pd.DataFrame(rows)
    (zx0, zx1), (zy0, zy1) = S.ZOOM
    d['in_inset'] = (d.method != MMN) & d.latency_ms.between(zx0, zx1) & d.macro_f1.between(zy0, zy1)  # MMN: main only
    return d[['method', 'knob', 'knob_value', 'macro_f1', 'latency_ms', 'in_inset', 'table2_setting']]


def page_content():
    out = []
    for raw in re.findall(rb'stream\r?\n(.*?)\r?\nendstream', open(PDF, 'rb').read(), re.S):
        try:
            out.append(zlib.decompress(raw).decode('latin-1'))
        except zlib.error:
            pass
    return '\n'.join(out)


def placements(s):
    """Page positions (pt) of the curve markers: matplotlib writes the markers of a line with 2+ points as one q ... Q
    block of relative '1 0 0 1 dx dy cm /Mk Do' placements."""
    pos = []
    for blk in re.findall(r'(?:^|\n)q\n((?:1 0 0 1 \S+ \S+ cm /M\d+ Do\n)+)Q', s):
        x = y = 0.0
        for dx, dy in re.findall(r'1 0 0 1 (\S+) (\S+) cm', blk):
            x += float(dx); y += float(dy)
            pos.append((x, y))
    return np.array(pos)


def subpath_boxes(s):
    """Bounding boxes (x0, y0, x1, y1) of the straight-edged closed subpaths: a single marker, such as MultiModN + ARC's,
    is written as an explicit path, not a placement."""
    boxes = []
    for sp in re.findall(r'(\S+ \S+ m\n(?:\S+ \S+ l\n)+h)', s):
        xy = np.array(re.findall(r'(\S+) (\S+) [ml]', sp), float)
        boxes.append((*xy.min(0), *xy.max(0)))
    return np.array(boxes)


def check(d):
    W, H, (L, R, B, T) = S.W, S.H, (40.0, 9.0, 86.0, 6.0)      # pareto_single.main()'s axes margins (pt)
    aw, ah = W - L - R, H - B - T
    (x0, x1), (y0, y1) = S.XLIM, S.YLIM
    main = lambda x, y: np.c_[L + (x - x0) / (x1 - x0) * aw, B + (y - y0) / (y1 - y0) * ah]     # noqa: E731
    ins = main(np.array(S.INSET[0]), np.array(S.INSET[1]))       # inset corners on the page
    (zx0, zx1), (zy0, zy1) = S.ZOOM
    inset = lambda x, y: np.c_[ins[0, 0] + (x - zx0) / (zx1 - zx0) * (ins[1, 0] - ins[0, 0]),   # noqa: E731
                               ins[0, 1] + (y - zy0) / (zy1 - zy0) * (ins[1, 1] - ins[0, 1])]
    s = page_content()
    P = placements(s)
    curve = d[d.method != MMN]
    pm = main(curve.latency_ms.values, curve.macro_f1.values)
    pi = inset(curve.latency_ms.values, curve.macro_f1.values)
    near = lambda q: np.hypot(*(P - q).T).min()                  # noqa: E731
    err_main = np.array([near(q) for q in pm])
    err_in = np.array([near(q) for q in pi[curve.in_inset.values]])
    assert (err_main < TOL).all() and (err_in < TOL).all(), (err_main, err_in)
    # every placement inside the main plot area is one of the rows (main view, or the inset, clipped or not)
    inside = P[(P[:, 0] > L) & (P[:, 0] < L + aw) & (P[:, 1] > B) & (P[:, 1] < B + ah)]
    expect = np.vstack([pm, pi])
    stray = [q for q in inside if np.hypot(*(expect - q).T).min() > TOL]
    assert not stray, stray
    # MultiModN + ARC: the 10 pt 'P' marker's box is centred on the point
    bx = subpath_boxes(s)
    mm = d[d.method == MMN].iloc[0]
    q = main(np.array([mm.latency_ms]), np.array([mm.macro_f1]))[0]
    hit = bx[(np.abs(bx[:, 2] - bx[:, 0] - S.MMN['ms']) < .05) & (np.abs(bx[:, 3] - bx[:, 1] - S.MMN['ms']) < .05)]
    err_mm = np.hypot(*((hit[:, :2] + hit[:, 2:]) / 2 - q).T).min()
    assert err_mm < TOL, err_mm
    # back to data units: the largest row-to-PDF difference
    dx = max(err_main.max(), err_in.max(), err_mm)
    return len(P), len(inside), dx * (x1 - x0) / aw, dx * (y1 - y0) / ah


def main():
    d = points()
    n_all, n_in, dms, df1 = check(d)
    print(f'PDF check: {len(d)} rows; {n_in} curve markers inside the plot ({n_all} in the PDF incl. the legend), all '
          f'belong to rows; largest row-to-PDF difference {dms:.1e} ms, {df1:.1e} F1')
    o = d.copy()
    o['knob_value'] = [f'{v:g}' if v == v else '' for v in o.knob_value]
    o['macro_f1'] = o.macro_f1.round(4)
    o['latency_ms'] = o.latency_ms.round(1)
    o.to_csv(OUT, index=False)
    print(OUT)
    print(o.to_string(index=False))


if __name__ == '__main__':
    main()

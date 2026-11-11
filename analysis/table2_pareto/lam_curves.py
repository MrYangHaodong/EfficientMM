#!/usr/bin/env python
"""ctrl_curves.py's figure with SeMARC's curve over lambda at uniform cost (gamma = 0) instead of over gamma at Table 2's
lambda. The controllers' curves, Table 2 markers and latency estimate are ctrl_curves.py's, unchanged.

SeMARC: gamma = 0 (RLB_COST_MODE=uniform), lambda varied and shared by all six datasets (as the controllers' knobs); every
other setting is the deployed one per fold (sweep.py: per-fold ARC seed, FQI schedule, availability augmentation, and the
paper's rule: prior priced at the training lambda, stop margin 0). Lambda .05, .1, .2, .5, 1, 2 are the gamma x lambda
sweep's uniform runs (res/sweep_*_g0.csv, res/verify_*_g0.csv); the values in between were fitted for this figure with
the same code (lamu_launch.sh -> res/lamu_*_g0.csv). Table 2's SeMARC (deployed lambda .5, MM-Fi fold 2: 1) is the star;
the paper's weighted row (gamma = .75 at that lambda) the diamond.
Writes res/lam_points.csv, res/LAM_CURVES.md and fig/pareto_lambda_controllers_mean.{pdf,png}.
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plot_pareto as pp
import plot_gamma as pg
import ctrl_curves as cc
from ctrl_curves import EAMA, EAMA_W, JAFA, GMSN, SEMARC, CURVE, ORDER_M, S_DOT, S_PUB, kv, f3, ms
from plot_pareto import BLUE, INK, INK2, MUTED

T = cc.T
cc.KNOB[SEMARC] = 'λ'                       # SeMARC's knob in this figure (ctrl_curves' rows read KNOB when built)
T2_LAM = "Table 2's λ (.5; MM-Fi fold 2: 1)"
W_ROW = "γ = .75 at Table 2's λ"
S_STAR, S_DIA = 170, 62


# ───────────────────────────── SeMARC at uniform cost ─────────────────────────────
def uniform_runs():
    fs = sorted(glob.glob(f'{T}/res/sweep_*_g0.csv') + glob.glob(f'{T}/res/verify_*_g0.csv')
                + glob.glob(f'{T}/res/lamu_*_g0.csv'))
    d = pd.concat([pd.read_csv(f, dtype={'lam': str}).assign(file=f.split('/')[-1]) for f in fs], ignore_index=True)
    d = d[d.gamma == 0]
    k = ['dataset', 'lam', 'fold']
    dup = d[d.duplicated(k, keep=False)].groupby(k).f1.agg(lambda s: s.max() - s.min())
    assert (dup <= 1e-9).all(), dup[dup > 1e-9]                    # a run stored twice must be the same run
    d = d.drop_duplicates(k)
    n = d.groupby(['dataset', 'lam']).fold.nunique()
    assert (n == 3).all(), n[n != 3]
    return d


def semarc_lam():
    out = []
    for (ds, lam), g in uniform_runs().groupby(['dataset', 'lam']):
        kind, value = ('table2', T2_LAM) if lam == 'dep' else ('curve', float(lam))
        out.append(cc.row(SEMARC, ds, kind, value, float(g.f1.mean()), float(g.mu.mean()),
                          (float(g.extract_ms.mean()), float(g.fusion_ms.mean()), float(g.policy_ms.mean())), len(g),
                          source='uniform-cost runs (sweep.py)', new_fits=int((g.net_existed == 0).sum()),
                          from_files=', '.join(sorted(set(g.file)))))
    w = pg.gamma_configs()
    for r in w[(w.gamma == .75) & (w.dataset != 'mean')].itertuples():
        out.append(cc.row(SEMARC, r.dataset, 'weighted', W_ROW, float(r.f1), float(r.mu), (float(r.latency_ms), 0.0, 0.0),
                          3, source="paper γ-study nets at Table 2's λ (verify csv)"))
    return out


def load():
    d = pd.DataFrame(cc.eama() + cc.jafa() + cc.gms() + semarc_lam())
    w = d[d.kind == 'weighted']
    assert w.dataset.nunique() == 6
    wm = dict(method=SEMARC, knob='γ', kind='weighted', value=W_ROW, dataset='mean', n=6, **{c: w[c].mean() for c in cc.COLS})
    a = pd.concat([d, cc.means(d[d.kind != 'weighted']), pd.DataFrame([wm])], ignore_index=True)
    a['on_front'] = False
    for _, g in a[a.kind == 'curve'].groupby(['method', 'dataset']):
        a.loc[pp.pareto(g).index, 'on_front'] = True
    return a


def check_table2(a):
    """The star must be the published SeMARC row and the diamond the published weighted row (rounding only)."""
    pts = pp.paper_points(); out = []
    for kind, name in (('table2', 'SeMARC (Table 2, uniform cost)'), ('weighted', 'SeMARC (paper, weighted, γ = .75)')):
        for ds in pp.ORDER + ['mean']:
            p = pts[(pts.method == name) & (pts.dataset == ds)].iloc[0]
            r = a[(a.method == SEMARC) & (a.dataset == ds) & (a.kind == kind)].iloc[0]
            out.append(dict(row=name, dataset=ds, f1=r.f1, f1_paper=p.f1, ms=r.latency_ms, ms_paper=p.latency_ms))
    o = pd.DataFrame(out)
    assert ((o.f1 - o.f1_paper).abs() <= 6e-4).all() and (((o.ms - o.ms_paper) / o.ms_paper).abs() <= 3e-3).all(), o
    return o


# ───────────────────────────── figure ─────────────────────────────
LEG_SEMARC = 'SeMARC: λ .05–2, uniform cost (γ = 0)'


def front(a, m, ds):
    return a[(a.method == m) & (a.dataset == ds) & (a.kind == 'curve') & a.on_front].sort_values('latency_ms')


def draw(ax, a, ds, mmn=None):
    for m in ORDER_M:
        col, ls, mk, filled, _ = CURVE[m]
        fr = front(a, m, ds)
        ax.plot(fr.latency_ms, fr.f1, color=col, ls=ls, lw=2.0 if m == SEMARC else 1.6, zorder=3,
                solid_joinstyle='round', dash_capstyle='round')
        ax.scatter(fr.latency_ms, fr.f1, s=S_DOT * (1.15 if mk == '^' else 1), marker=mk,
                   color=col if filled else 'white', edgecolors='white' if filled else col,
                   linewidths=1.0 if filled else 1.3, zorder=4 if m != SEMARC else 5)
    for m in ORDER_M[:-1]:                                        # each controller's Table 2 configuration
        col, _, mk, filled, _ = CURVE[m]
        p = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'table2')]
        if filled:
            ax.scatter(p.latency_ms, p.f1, s=S_PUB * (1.15 if mk == '^' else 1), marker=mk, color=col, edgecolors=INK,
                       linewidths=1.3, zorder=6)
        else:                                                     # hollow like its curve's dots: orange ring inside a black one
            ax.scatter(p.latency_ms, p.f1, s=S_PUB * 1.5, marker=mk, color='white', edgecolors=INK, linewidths=1.0, zorder=6)
            ax.scatter(p.latency_ms, p.f1, s=S_PUB * .62, marker=mk, color='white', edgecolors=col, linewidths=2.2, zorder=6)
    s = a[(a.method == SEMARC) & (a.dataset == ds)]
    r = s[s.kind == 'table2']
    ax.scatter(r.latency_ms, r.f1, s=S_STAR, marker='*', color=BLUE, edgecolors=INK, linewidths=1.1, zorder=7)
    r = s[s.kind == 'weighted']
    ax.scatter(r.latency_ms, r.f1, s=S_DIA, marker='D', color='white', edgecolors=BLUE, linewidths=1.6, zorder=7)
    if mmn is not None:
        ax.scatter(mmn.latency_ms, mmn.f1, s=60, marker='P', color=cc.GRAY, edgecolors='white', linewidths=.8, zorder=5)


def legend_handles():
    h = []
    for m in [SEMARC, EAMA, EAMA_W, JAFA, GMSN]:
        col, ls, mk, filled, txt = CURVE[m]
        h.append(Line2D([], [], color=col, ls=ls, lw=1.8, marker=mk, ms=5.2, mfc=col if filled else 'white',
                        mec='white' if filled else col, mew=1.0 if filled else 1.3, label=LEG_SEMARC if m == SEMARC else txt))
    h += [Line2D([], [], lw=0, marker='*', ms=12, color=BLUE, mec=INK, mew=.9, label="SeMARC, Table 2 (γ = 0, Table 2's λ)"),
          Line2D([], [], lw=0, marker='D', ms=6.5, color='white', mec=BLUE, mew=1.5, label="SeMARC, paper's weighted row (γ = .75)"),
          Line2D([], [], lw=0, marker='o', ms=8.5, color='#d7d6d0', mec=INK, mew=1.2,
                 label="Black-edged marker: the controller's\nTable 2 configuration"),
          Line2D([], [], lw=0, marker='P', ms=8, color=cc.GRAY, mec='white', label='MultiModN + ARC (Table 2)')]
    return h


def obstacles(ax, a, ds):
    """Every drawn mark (circle in display pixels) and samples along every front line, for the label placer."""
    px = ax.figure.dpi / 72; tr = ax.transData.transform; ob = []
    size = {'table2': S_PUB, 'weighted': S_DIA, 'curve': S_DOT}
    for r in a[(a.dataset == ds) & ((a.kind != 'curve') | a.on_front)].itertuples():
        sz = S_STAR if (r.method == SEMARC and r.kind == 'table2') else size[r.kind]
        x, y = tr((r.latency_ms, r.f1)); ob.append((x, y, (np.sqrt(sz) / 2 + 1.5) * px))
    for m in ORDER_M:
        xy = tr(front(a, m, ds)[['latency_ms', 'f1']].values)
        for p, q in zip(xy[:-1], xy[1:]):
            for t in np.linspace(0, 1, 40):
                ob.append((*(p + t * (q - p)), 1.6 * px))
    return ob


NOTE = ('Mean of the six datasets. Each knob value is shared by\n'
        "all six (GMS: k capped at each dataset's M); lines are each\n"
        "method's Pareto front over its knob. SeMARC: uniform cost\n"
        '(γ = 0); numbers next to the blue dots give λ.\n'
        "Black-edged markers and the star use Table 2's settings:\n"
        "EAMA's λ picked per run on validation, GMS's k per dataset,\n"
        "Joint-AFA the paper's run at r$_{cost}$ = .05, SeMARC λ = .5\n"
        '(MM-Fi fold 2: 1). So they need not lie on the curves.')


def plot_mean(a, mmn, box, out=f'{T}/fig/pareto_lambda_controllers_mean', xticks=None):
    """Left: the whole range (log latency). Middle: the box around SeMARC (linear latency). Right: legend."""
    ds = 'mean'
    plt.rcParams.update(pg.RC)
    fig = plt.figure(figsize=(12.4, 4.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.18, .9])
    ax, az, lax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    draw(ax, a, ds, mmn=mmn)
    vis = a[(a.dataset == ds) & ((a.kind != 'curve') | a.on_front)]
    xs = np.r_[vis.latency_ms.values, mmn.latency_ms.values]; ys = np.r_[vis.f1.values, mmn.f1.values]
    pg.style_axes(ax, ds, xs.min() / 1.35, xs.max() * 1.35)
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
    ob = obstacles(az, a, ds)
    placed = []
    for m, kind, text, order in (     # the diamond has no direct label (its legend entry is unique), as in the γ figure
            (EAMA_W, 'table2', 'weighted EAMA,\nTable 2', ('above', 'ur', 'ul', 'right', 'lr', 'below', 'left', 'll')),
            (SEMARC, 'table2', 'SeMARC,\nTable 2', ('ur', 'above', 'ul', 'right', 'left', 'lr', 'below', 'll')),
            (EAMA, 'table2', 'EAMA, Table 2', ('right', 'ur', 'lr', 'below', 'above', 'left', 'll', 'ul'))):
        r = a[(a.method == m) & (a.dataset == ds) & (a.kind == kind)].iloc[0]
        if x0 <= r.latency_ms <= x1 and y0 <= r.f1 <= y1:
            placed.append(cc.place_label(az, r.latency_ms, r.f1, text, ob, placed, fs=8.0,
                                         rad_pt=8 if (m, kind) == (SEMARC, 'table2') else 7, order=order))
    fr = front(a, SEMARC, ds)                                     # λ labels last: short, they fit around the others
    for r in fr[fr.latency_ms.between(x0, x1) & fr.f1.between(y0, y1)].itertuples():
        placed.append(cc.place_label(az, r.latency_ms, r.f1, kv(r.value), ob, placed, fs=8.4, rad_pt=2.6,
                                     order=('above', 'ul', 'ur', 'left', 'right', 'below', 'll', 'lr')))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=250, bbox_inches='tight')
    plt.close(fig)


# ───────────────────────────── report ─────────────────────────────
def versus(a, ds):
    """Each controller's front points and Table 2 point against SeMARC's lambda front, and the reverse."""
    g = front(a, SEMARC, ds)
    g = g.assign(label=[f'λ = {kv(v)}' for v in g.value])
    rows = []
    for m in [EAMA, EAMA_W, JAFA, GMSN]:
        pts = a[(a.method == m) & (a.dataset == ds) & (((a.kind == 'curve') & a.on_front) | (a.kind == 'table2'))]
        for q in pts.sort_values(['kind', 'latency_ms']).itertuples():
            b = cc.beaten(q, g)
            le = g[g.latency_ms <= q.latency_ms]
            rows.append(dict(method=m, kind=q.kind, value=q.value, f1=q.f1, latency_ms=q.latency_ms,
                             beaten_by=', '.join(b.label) if len(b) else 'no',
                             semarc_best_f1_at_le=le.f1.max() if len(le) else np.nan))
    back = []
    ctl = a[(a.dataset == ds) & (a.method != SEMARC) & (a.kind == 'curve')]
    for s in g.itertuples():
        b = cc.beaten(s, ctl)
        back.append(dict(lam=s.value, f1=s.f1, latency_ms=s.latency_ms,
                         beaten_by=', '.join(f'{r.method.replace("SeMA + ", "")} {cc.KNOB[r.method]} = {kv(r.value)}'
                                             for r in b.itertuples()) or 'none'))
    return pd.DataFrame(rows), pd.DataFrame(back)


def short(s, n=2):
    """The first n entries of a ', '-joined list, and how many more there are."""
    p = s.split(', ')
    return ', '.join(p[:n]) + (f' (+{len(p) - n} more)' if len(p) > n else '')


def write_md(a, chk):
    sem = a[(a.method == SEMARC) & (a.kind == 'curve')]
    lams = sorted(sem[sem.dataset == 'mean'].value.astype(float))
    new = sorted({float(v) for v in sem[(sem.dataset != 'mean') & (sem.new_fits > 0)].value})
    nfit = int(sem[sem.dataset != 'mean'].new_fits.sum())
    L = ['# F1–latency curves: SeMARC over λ at uniform cost, and the Table 2 controllers (Table 2 metric)', '',
         "Request: the figure of `CTRL_CURVES.md`, with SeMARC's curve over λ at uniform cost (γ = 0) instead of over γ at "
         "Table 2's λ (`lam_curves.py`, `fig/pareto_lambda_controllers_mean.{pdf,png}`).", '',
         '## SeMARC\'s curve', '',
         f'- **Knob**: λ ∈ {{{", ".join(kv(v) for v in lams)}}}, at uniform cost (γ = 0, `RLB_COST_MODE=uniform`). '
         'One λ is shared by all six datasets and all folds, as each controller\'s knob.',
         '- **Everything else** is the deployed setting per fold (`sweep.py`):',
         '  - the per-fold ARC seed, FQI schedule and availability augmentation;',
         '  - the paper\'s decision rule: the prior priced at the training λ, stop margin 0.',
         '- **Nets**:',
         "  - λ = .05, .1, .2, .5, 1, 2 are the γ × λ sweep's uniform-cost runs (`res/sweep_*_g0.csv`, `res/verify_*_g0.csv`).",
         f'  - λ = {", ".join(kv(v) for v in new)} were fitted for this figure: {nfit} nets (6 datasets × 3 folds × {len(new)} λ), '
         'with the unchanged `train_or_load` on GPU 2/3 (`lamu_launch.sh` → `res/lamu_*_g0.csv`, nets in '
         '`RL_AAAI_branch/checkpoints_t2p/<ds>/`).',
         "- **Star** = Table 2's SeMARC: the deployed λ = .5 on every fold except MM-Fi fold 2 (λ = 1). "
         'It is not a λ shared by all folds, so it is not on the curve; λ = .5 shared differs from it only on MM-Fi fold 2.',
         "- **Diamond** = the paper's weighted row (γ = .75 at Table 2's λ), as in the γ figure.",
         "- **Controllers**: curves, Table 2 markers and latency estimate unchanged from `CTRL_CURVES.md`.", '',
         '## Checks', '',
         '- The star and the diamond recompute to the published rows of `tab:comp-replace-complete`:', '',
         '| Row | dataset | F1 | paper F1 | latency (ms) | paper latency (ms) |', '|---|---|---|---|---|---|']
    for r in chk.itertuples():
        L.append(f'| {r.row} | {pp.LABEL[r.dataset].replace(" (Table 2)", "")} | {r.f1:.4f} | {r.f1_paper:.3f} | '
                 f'{r.ms:.2f} | {r.ms_paper:.2f} |')
    L += ['', '## Mean of six datasets', '', '**SeMARC, uniform cost** (λ)', '',
          '| λ | F1 | MU | latency (ms) | on the front |', '|---|---|---|---|---|']
    for r in a[(a.method == SEMARC) & (a.dataset == 'mean')].sort_values(['kind', 'latency_ms']).itertuples():
        name = kv(r.value) if r.kind == 'curve' else ('Table 2 (star)' if r.kind == 'table2' else 'weighted row (diamond)')
        L.append(f'| {name} | {f3(r.f1)} | {r.mu:.2f} | {ms(r.latency_ms)} | {"yes" if r.on_front else ""} |')
    vs, back = versus(a, 'mean')
    L += ['', 'The controllers\' mean points are in `CTRL_CURVES.md`.', '',
          '## Why the Table 2 markers are not on the curves', '',
          "A curve point runs every dataset, fold and seed at one knob value. Table 2's configurations do not:",
          '- **EAMA** picks λ separately in each of its 54 runs (6 datasets × 3 folds × 3 value-model seeds) on the validation '
          'holdout, by val F1 − 0.2 × val extraction fraction (`baseline_eama_raw.py`). The picks range from .005 to 2.',
          '- **GMS** uses one k per dataset, the same on every fold, selected on validation according to the paper: '
          'IEMOCAP 5, MM-Fi 1, CZU-MHAD 4, EAV 1, CMI 5, UTD-MHAD 2 (`build_acq_profiles.py`).',
          "- **Joint-AFA**'s r_cost is fixed at .05 on every dataset and fold, not tuned. Its marker is the paper's GPU run; "
          'its curve is the CPU retrain.',
          "- **SeMARC**'s λ is not tuned either: .5 on every fold except MM-Fi fold 2 (1.0). Only the per-fold seed is "
          'selected on validation (Appendix `tab:policy-config`).',
          '',
          'An average of runs at different knob values is not a point of the one-value curve, so it can land on either side:',
          '- **Uniform EAMA lands below its curve.** It also sits at or below each dataset\'s own curve, by up to .012 F1 at '
          'the same latency (UTD-MHAD). Its validation picks do not carry over to test.',
          '- **Weighted EAMA lands above its curve.** Per dataset it sits on its own curves (IEMOCAP +.008, CMI −.004). '
          'The gain at the mean comes from tuning λ per run: EAMA prices a fraction of each dataset\'s own total extraction '
          'time, from 8 ms (CMI) to 4.2 s (MM-Fi), so one shared λ prices a millisecond very differently on each dataset.',
          '- The latency formula is not the cause: at the same λ the marker\'s price (stored frequencies) and the curve\'s '
          '(mean set size) differ by at most .004 ms.', '',
          '| Configuration | Table 2 F1 @ ms | its curve at that latency (linear between front points) | difference |',
          '|---|---|---|---|']
    for m in [SEMARC, EAMA, EAMA_W, JAFA, GMSN]:
        t = a[(a.method == m) & (a.dataset == 'mean') & (a.kind == 'table2')].iloc[0]
        fr = front(a, m, 'mean')
        c = (np.interp(t.latency_ms, fr.latency_ms, fr.f1) if fr.latency_ms.min() <= t.latency_ms <= fr.latency_ms.max()
             else np.nan)
        L.append(f'| {m} | {f3(t.f1)} @ {ms(t.latency_ms)} | {f3(c)} | {"–" if pd.isna(c) else f"{t.f1 - c:+.3f}"} |')
    L += ['', '## Who beats whom on the mean', '',
          "Beaten = some point has F1 ≥ and latency ≤, one strictly. 'SeMARC λ' = the blue curve.", '',
          '| Controller | point | F1 | latency (ms) | beaten by SeMARC λ | SeMARC λ best F1 at ≤ its latency |',
          '|---|---|---|---|---|---|']
    for r in vs.itertuples():
        pt = 'Table 2' if r.kind == 'table2' else f'{cc.KNOB[r.method]} = {kv(r.value)}'
        L.append(f'| {r.method} | {pt} | {f3(r.f1)} | {ms(r.latency_ms)} | {r.beaten_by} | {f3(r.semarc_best_f1_at_le)} |')
    L += ['', '| SeMARC λ (front) | F1 | latency (ms) | beaten by a controller point |', '|---|---|---|---|']
    for r in back.itertuples():
        L.append(f'| {kv(r.lam)} | {f3(r.f1)} | {ms(r.latency_ms)} | {r.beaten_by} |')
    L += ['', '## Per dataset: SeMARC\'s λ front', '',
          '| Dataset | SeMARC λ front: λ → F1 @ ms | Table 2 (star) | controller front points not beaten by SeMARC λ | '
          'SeMARC λ front points beaten by a controller |', '|---|---|---|---|---|']
    for ds in pp.ORDER:
        fr = front(a, SEMARC, ds)
        t = a[(a.method == SEMARC) & (a.dataset == ds) & (a.kind == 'table2')].iloc[0]
        vs, back = versus(a, ds)
        nb = vs[(vs.kind == 'curve') & (vs.beaten_by == 'no')]
        bb = back[back.beaten_by != 'none']
        L.append(f'| {pp.LABEL[ds]} | ' + '; '.join(f'{kv(r.value)} → {f3(r.f1)} @ {ms(r.latency_ms)}' for r in fr.itertuples())
                 + f' | {f3(t.f1)} @ {ms(t.latency_ms)} | {len(nb)} of {int((vs.kind == "curve").sum())} | '
                 + (f'{len(bb)} of {len(back)}: ' + '; '.join(f'λ = {kv(r.lam)} by {short(r.beaten_by)}' for r in bb.itertuples())
                    if len(bb) else f'0 of {len(back)}') + ' |')
    L.append('')
    open(f'{T}/res/LAM_CURVES.md', 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    import sys
    pd.set_option('display.width', 220)
    a = load()
    chk = check_table2(a)
    a.to_csv(f'{T}/res/lam_points.csv', index=False)
    m = a[(a.dataset == 'mean')]
    print(m[m.method == SEMARC][['kind', 'value', 'f1', 'mu', 'latency_ms', 'on_front']].round(4).to_string())
    print(m[(m.method != SEMARC) & (m.on_front | (m.kind == 'table2'))][['method', 'kind', 'value', 'f1', 'latency_ms']]
          .sort_values('latency_ms').round(4).to_string())
    pts = pp.paper_points()
    mmn = pts[(pts.dataset == 'mean') & (pts.method == 'MultiModN + ARC')]
    print('MultiModN + ARC mean', mmn[['f1', 'latency_ms']].round(4).values)
    x0, x1, y0, y1 = (float(v) for v in sys.argv[1:5]) if len(sys.argv) >= 5 else (280, 720, .68, .78)
    plot_mean(a, mmn, ((int(x0), int(x1)), (y0, y1)), xticks=[300, 400, 500, 600, 700])
    write_md(a, chk)

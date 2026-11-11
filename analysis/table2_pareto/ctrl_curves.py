#!/usr/bin/env python
"""F1-latency curves of SeMARC and the three controllers of Table 2 on the shared SeMA backbone, Table 2's metric.

Each controller's own trade-off knob is swept; everything else is the paper's configuration:
  SeMA + EAMA       lambda in J_s(q) = score(q | p_s) - lambda * sum_{j in q} z_j, at uniform (z_j = 1/M; Table 2's row) and
                    extraction-weighted costs (z_j = t_j / sum t; the weighted row). Test-time only: the paper's value
                    models (3 seeds x 3 folds) were already evaluated at every lambda of the grid
                    (results_baselines/eama_raw_<ds>_idle_f*.csv); no new fit.
  SeMA + Joint-AFA  r_cost, the per-acquisition cost of the reward. Training-time: retrained by ctrl/jafa_sweep.py with the
                    paper's code and seed (0), 20000 episodes, on CPU (the shared GPUs ran it > 10x slower); r_cost = .05
                    is retrained too, and the paper's own run (GPU) is the Table 2 marker.
  SeMA + GMS        the budget k of the greedy chain selected on validation (results_baselines/greedy_submodular_*.json).
  SeMARC            gamma at Table 2's lambda (plot_gamma.py; the paper's gamma-study nets).
Latency = Table 2's estimate (price_raw_selectors.py for the pre-backbone selectors: extraction of the acquired set + ONE
fusion of it + (decisions x selector proxy); GMS: extraction + one fusion, no selector, as reprice_controllers_e2e.py).
Six-dataset mean: one knob value shared by all six datasets (GMS: k capped at each dataset's M), as SeMARC's gamma.
Each curve is the Pareto front over its knob values. Writes res/ctrl_points.csv, res/CTRL_CURVES.md,
fig/pareto_controllers_mean.{pdf,png} and fig/pareto_controllers_all.{pdf,png}.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import plot_pareto as pp
import plot_gamma as pg
from plot_pareto import BLUE, ORANGE, AQUA, INK, INK2, MUTED, GRID, AXIS

T = pp.T
RLB = '/files1/haodong/test/new/RL_AAAI_branch'; B = f'{RLB}/results_baselines'
EA = '/files1/haodong/test/new/energy_adaptive'
TAB = json.load(open(f'{EA}/results_per_modality_extract_cuda_idle.json'))
FW = {ds: json.load(open(f'{EA}/results_fusion_walk_idle_{ds}.json'))[ds] for ds in pp.ORDER}
GMS = {'iemocap': 'greedy_submodular_iemocap.json', 'mmfi': 'greedy_submodular_mmfi_rebal.json',
       'czu_mhad': 'greedy_submodular_czu_mhad.json', 'eav': 'greedy_submodular_eav.json',
       'cmi': 'greedy_submodular_cmi.json', 'utd_mhad': 'greedy_submodular_utd_mhad_rebal422.json'}
GMS_K = {'iemocap': 5, 'mmfi': 1, 'czu_mhad': 4, 'eav': 1, 'cmi': 5, 'utd_mhad': 2}   # Table 2's k (build_acq_profiles.py)
JAFA_DIR = f'{T}/ctrl/jafa'
JAFA_R = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]
EAMA, EAMA_W, JAFA, GMSN, SEMARC = 'SeMA + EAMA', 'SeMA + EAMA (weighted)', 'SeMA + Joint-AFA', 'SeMA + GMS', 'SeMARC'
KNOB = {EAMA: 'λ', EAMA_W: 'λ', JAFA: 'r_cost', GMSN: 'k', SEMARC: 'γ'}
COLS = ('f1', 'mu', 'latency_ms', 'extract_ms', 'fusion_ms', 'selector_ms')


# ───────────────────────────── Table 2's latency estimate (price_raw_selectors.py) ─────────────────────────────
def timings(ds):
    vals = list(FW[ds].values())
    return dict(t_enc={m: float(np.mean([v['t_enc_ms'][m] for v in vals])) for m in vals[0]['t_enc_ms']},
                t_head=float(np.mean([v['t_head_ms'] for v in vals])), t_pol=float(np.mean([v['t_pol_ms'] for v in vals])),
                cal_fixed=float(np.mean([v['direct_ms'] / v['model_ms'] for f in vals for v in f['fixed_k'].values()])))


TM = {ds: timings(ds) for ds in pp.ORDER}


def price(ds, mods, freq, calls):
    """extraction + ONE fusion of the final set + calls x selector proxy (price_raw_selectors.price)."""
    tm = TM[ds]; tab = TAB[ds]['per_modality']
    ext = float(sum(f * tab[m]['ms'] for f, m in zip(freq, mods)))
    fus = tm['cal_fixed'] * (float(sum(f * tm['t_enc'][m] for f, m in zip(freq, mods))) + tm['t_head'])
    return ext, fus, calls * tm['t_pol']


def price_k(ds, ext, Ek, calls):
    """The same estimate from the mean extraction ms and mean set size: per-modality fusion steps differ by < 1% within a
    dataset (results_fusion_walk_idle_<ds>.json), so sum_m freq_m t_enc_m = E[k] * mean t_enc to < 0.05 ms."""
    tm = TM[ds]
    return ext, tm['cal_fixed'] * (Ek * float(np.mean(list(tm['t_enc'].values()))) + tm['t_head']), calls * tm['t_pol']


def row(method, ds, kind, value, f1, mu, parts, n, **kw):
    ext, fus, pol = parts
    return dict(method=method, knob=KNOB[method], kind=kind, value=value, dataset=ds, f1=f1, mu=mu,
                latency_ms=ext + fus + pol, extract_ms=ext, fusion_ms=fus, selector_ms=pol, n=n, **kw)


# ───────────────────────────── controllers ─────────────────────────────
def eama():
    out = []
    for ds in pp.ORDER:
        d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{B}/eama_raw_{ds}_idle_f*.csv'))])
        d = d[d.optimizer == 'hybrid']; M = len(TM[ds]['t_enc'])
        sets = {}
        for f in sorted(glob.glob(f'{B}/eama_raw_{ds}_idle_f*_sets.json')):
            sets.update(json.load(open(f)))
        for cost, name in (('uniform', EAMA), ('extract', EAMA_W)):
            c = d[d.cost == cost]
            for lam, g in c.groupby('lam'):
                assert len(g) == 9, (ds, cost, lam, len(g))
                out.append(row(name, ds, 'curve', float(lam), float(g.test_f1.mean()), float(g.test_mu.mean()),
                               price_k(ds, float(g.test_ms.mean()), float(g.test_mu.mean()) * M, float(g.stages.mean()) + 1),
                               len(g), source='paper value models (eama_raw csv)'))
            # Table 2's point: lambda chosen per fold and seed on validation (B* rule); priced from its stored frequencies
            s = c[c.selected]; ks = [k for k in sets if k.endswith(f'/{cost}/hybrid')]; mods = sets[ks[0]]['mods']
            freq = np.mean([sets[k]['freq'] for k in ks], 0); st = float(np.mean([sets[k]['stages'] for k in ks]))
            out.append(row(name, ds, 'table2', 'val-selected λ', float(s.test_f1.mean()), float(s.test_mu.mean()),
                           price(ds, mods, freq, st + 1), len(s), source='paper (eama_raw csv, selected)',
                           check_ms=sum(price_k(ds, float(s.test_ms.mean()), float(s.test_mu.mean()) * M, st + 1))))
    return out


def jafa_runs(ds, r, dev='cpu'):
    folds = ['3', '4', '5'] if ds == 'utd_mhad' else ['0', '1', '2']
    fs = [f'{JAFA_DIR}/jafa_{ds}_f{f}_r{r:g}_s0_{dev}.json' for f in folds]
    return [json.load(open(f)) for f in fs] if all(os.path.exists(f) for f in fs) else None


def jafa_row(ds, kind, value, pf, source):
    mods = pf[0]['mods']; freq = np.mean([p['freq'] for p in pf], 0)
    return row(JAFA, ds, kind, value, float(np.mean([p['f1'] for p in pf])), float(np.mean([p['mu'] for p in pf])),
               price(ds, mods, freq, freq.sum() + 1), len(pf), source=source,
               val_f1=float(np.mean([p['val']['f1'] for p in pf])) if 'val' in pf[0] else np.nan)


def jafa():
    out = []
    for ds in pp.ORDER:
        paper = json.load(open(f'{B}/jafa_raw_{ds}_idle_rcost0.05_sets.json'))['per_fold']
        out.append(jafa_row(ds, 'table2', 'r_cost .05 (paper run)', [paper[f] for f in sorted(paper, key=int)],
                            'paper (jafa_raw_<ds>_idle_rcost0.05_sets.json, GPU)'))
        for r in JAFA_R:
            pf = jafa_runs(ds, r)
            if pf is not None:
                out.append(jafa_row(ds, 'curve', r, pf, 'retrained (ctrl/jafa, CPU, seed 0)'))
    return out


def gms():
    out = []
    for ds in pp.ORDER:
        g = json.load(open(f'{B}/{GMS[ds]}'))['per_fold']; mods = list(TM[ds]['t_enc']); M = len(mods)
        for k in range(1, M + 1):
            e = [[x for x in g[f] if x['budget'] == k][0] for f in sorted(g, key=int)]
            freq = np.mean([[1.0 if m in x['selected'] else 0.0 for m in mods] for x in e], 0)
            ext, fus, _ = price(ds, mods, freq, 0)
            for kind in ('curve', 'table2') if k == GMS_K[ds] else ('curve',):
                out.append(row(GMSN, ds, kind, k if kind == 'curve' else f'val-selected k = {k}',
                               float(np.mean([x['test_f1'] for x in e])), k / M, (ext, fus, 0.0), len(e),
                               source='paper (greedy_submodular json)'))
    return out


def semarc():
    c = pg.gamma_configs()
    return [row(SEMARC, r.dataset, 'curve', float(r.gamma), float(r.f1), float(r.mu), (float(r.latency_ms), 0.0, 0.0), 3,
                source="paper γ-study nets at Table 2's λ (verify csv)") for r in c.itertuples() if r.dataset != 'mean']


def means(d):
    """Six-dataset mean of every knob value that all six datasets have (GMS: k capped at each dataset's M), and of each
    method's Table 2 configuration."""
    rows = []
    cur = d[d.kind == 'curve']
    for m, g in cur.groupby('method', sort=False):
        if m == GMSN:
            kmax = {ds: g[g.dataset == ds].value.max() for ds in pp.ORDER}
            for k in range(1, max(kmax.values()) + 1):
                s = pd.concat([g[(g.dataset == ds) & (g.value == min(k, kmax[ds]))] for ds in pp.ORDER])
                rows.append(dict(method=m, knob='k', kind='curve', value=k, dataset='mean', n=6, **{c: s[c].mean() for c in COLS}))
            continue
        for v, s in g.groupby('value'):
            if s.dataset.nunique() == 6:
                rows.append(dict(method=m, knob=KNOB[m], kind='curve', value=v, dataset='mean', n=6, **{c: s[c].mean() for c in COLS}))
    for m, g in d[d.kind == 'table2'].groupby('method', sort=False):
        assert g.dataset.nunique() == 6, m
        rows.append(dict(method=m, knob=KNOB[m], kind='table2', value='Table 2', dataset='mean', n=6, **{c: g[c].mean() for c in COLS}))
    return pd.DataFrame(rows)


def load():
    d = pd.DataFrame(eama() + jafa() + gms() + semarc())
    a = pd.concat([d, means(d)], ignore_index=True)
    a['on_front'] = False
    for _, g in a[a.kind == 'curve'].groupby(['method', 'dataset']):
        a.loc[pp.pareto(g).index, 'on_front'] = True
    return a


# ───────────────────────────── figure ─────────────────────────────
YELLOW, GRAY = '#eda100', '#898781'
ORDER_M = [GMSN, JAFA, EAMA_W, EAMA, SEMARC]                     # drawing order: SeMARC on top
CURVE = {   # method -> (colour, line style, dot marker, filled dots, legend text)
    SEMARC: (BLUE, '-', 'o', True, "SeMARC: γ 0–2 at Table 2's λ"),
    EAMA: (ORANGE, '-', 'o', True, 'SeMA + EAMA: λ .005–2, uniform cost'),
    EAMA_W: (ORANGE, (0, (3.2, 1.8)), 'o', False, 'SeMA + EAMA: λ .005–2, weighted cost'),
    JAFA: (AQUA, '-', 's', True, 'SeMA + Joint-AFA: r$_{cost}$ 0–.5, retrained'),
    GMSN: (YELLOW, '-', '^', True, 'SeMA + GMS: budget k 1–7')}
S_DOT, S_PUB = 26, 92


def draw(ax, a, ds, scale=1.0, mmn=None):
    for m in ORDER_M:
        col, ls, mk, filled, _ = CURVE[m]
        fr = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'curve') & a.on_front].sort_values('latency_ms')
        ax.plot(fr.latency_ms, fr.f1, color=col, ls=ls, lw=(2.0 if m == SEMARC else 1.6) * scale ** .5, zorder=3,
                solid_joinstyle='round', dash_capstyle='round')
        ax.scatter(fr.latency_ms, fr.f1, s=S_DOT * scale * (1.15 if mk == '^' else 1), marker=mk,
                   color=col if filled else 'white', edgecolors='white' if filled else col,
                   linewidths=(1.0 if filled else 1.3) * scale ** .5, zorder=4 if m != SEMARC else 5)
    for m in ORDER_M[:-1]:                                        # Table 2's configuration of each controller
        col, _, mk, filled, _ = CURVE[m]
        p = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'table2')]
        if filled:
            ax.scatter(p.latency_ms, p.f1, s=S_PUB * scale * (1.15 if mk == '^' else 1), marker=mk, color=col,
                       edgecolors=INK, linewidths=1.3 * scale ** .5, zorder=6)
        else:                                                     # hollow like its curve's dots: orange ring inside a black one
            ax.scatter(p.latency_ms, p.f1, s=S_PUB * 1.5 * scale, marker=mk, color='white', edgecolors=INK,
                       linewidths=1.0 * scale ** .5, zorder=6)
            ax.scatter(p.latency_ms, p.f1, s=S_PUB * .62 * scale, marker=mk, color='white', edgecolors=col,
                       linewidths=2.2 * scale ** .5, zorder=6)
    s = a[(a.method == SEMARC) & (a.dataset == ds) & (a.kind == 'curve')]
    for g, mk, sz in ((0.0, '*', 170), (0.75, 'D', 62)):          # SeMARC's Table 2 rows: uniform (γ = 0) and weighted (γ = .75)
        r = s[s.value == g]
        ax.scatter(r.latency_ms, r.f1, s=sz * scale, marker=mk, color=BLUE if mk == '*' else 'white', edgecolors=INK if mk == '*' else BLUE,
                   linewidths=(1.1 if mk == '*' else 1.6) * scale ** .5, zorder=7)
    if mmn is not None:
        ax.scatter(mmn.latency_ms, mmn.f1, s=60 * scale, marker='P', color=GRAY, edgecolors='white', linewidths=.8, zorder=5)


def legend_handles(mmn=True):
    h = []
    for m in [SEMARC, EAMA, EAMA_W, JAFA, GMSN]:
        col, ls, mk, filled, txt = CURVE[m]
        h.append(Line2D([], [], color=col, ls=ls, lw=1.8, marker=mk, ms=5.2, mfc=col if filled else 'white',
                        mec='white' if filled else col, mew=1.0 if filled else 1.3, label=txt))
    h += [Line2D([], [], lw=0, marker='*', ms=12, color=BLUE, mec=INK, mew=.9, label='SeMARC, Table 2 (γ = 0)'),
          Line2D([], [], lw=0, marker='D', ms=6.5, color='white', mec=BLUE, mew=1.5, label="SeMARC, paper's weighted row (γ = .75)"),
          Line2D([], [], lw=0, marker='o', ms=8.5, color='#d7d6d0', mec=INK, mew=1.2,
                 label="Black-edged marker: the controller's\nTable 2 configuration")]
    if mmn:
        h.append(Line2D([], [], lw=0, marker='P', ms=8, color=GRAY, mec='white', label='MultiModN + ARC (Table 2)'))
    return h


def obstacles(ax, a, ds, scale=1.0):
    """Every drawn mark (circle in display pixels) and samples along every front line, for the label placer."""
    px = ax.figure.dpi / 72; tr = ax.transData.transform; ob = []
    sel = a[(a.dataset == ds) & ((a.kind == 'table2') | a.on_front)]
    for r in sel.itertuples():
        size = S_PUB if r.kind == 'table2' else S_DOT
        if r.method == SEMARC and r.kind == 'curve' and r.value in (0.0, 0.75):
            size = 170 if r.value == 0.0 else 62
        x, y = tr((r.latency_ms, r.f1)); ob.append((x, y, (np.sqrt(size * scale) / 2 + 1.5) * px))
    for m in ORDER_M:
        fr = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'curve') & a.on_front].sort_values('latency_ms')
        xy = tr(np.c_[fr.latency_ms.values, fr.f1.values])
        for p, q in zip(xy[:-1], xy[1:]):
            for t in np.linspace(0, 1, 40):
                ob.append((*(p + t * (q - p)), 1.6 * px))
    return ob


def gamma_frame(a, ds, box=None):
    s = a[(a.method == SEMARC) & (a.dataset == ds) & (a.kind == 'curve') & a.on_front]
    if box is not None:
        (x0, x1), (y0, y1) = box
        s = s[s.latency_ms.between(x0, x1) & s.f1.between(y0, y1)]
    return pd.DataFrame(dict(latency_ms=s.latency_ms.values, f1=s.f1.values, gamma=s.value.astype(float).values))


OFFS = {'right': (1, 0, 'left', 'center'), 'left': (-1, 0, 'right', 'center'), 'below': (0, -1, 'center', 'top'),
        'above': (0, 1, 'center', 'bottom'), 'lr': (.72, -.72, 'left', 'top'), 'll': (-.72, -.72, 'right', 'top'),
        'ur': (.72, .72, 'left', 'bottom'), 'ul': (-.72, .72, 'right', 'bottom')}


def place_label(ax, x, y, text, ob, avoid, fs, rad_pt, order=('right', 'left', 'below', 'above', 'lr', 'll', 'ur', 'ul')):
    """Put text next to (x, y) at the first offset in `order` clear of the axes edge, marks and other labels (else the
    least bad one)."""
    fig = ax.figure; rd = fig.canvas.get_renderer(); axbb = ax.get_window_extent(rd); d = rad_pt + 2
    cands = [(OFFS[k][0] * d, OFFS[k][1] * d, OFFS[k][2], OFFS[k][3]) for k in order]
    best = None
    for dx, dy, ha, va in cands:
        t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=fs, color=INK2,
                        zorder=8, annotation_clip=False)
        b = t.get_window_extent(rd); b = Bbox.from_extents(b.x0 - 1, b.y0 - 1, b.x1 + 1, b.y1 + 1)
        cost = 1000 * (b.x0 < axbb.x0 or b.x1 > axbb.x1 or b.y0 < axbb.y0 or b.y1 > axbb.y1)
        cost += 100 * sum(b.overlaps(o) for o in avoid) + 10 * sum(pg.hits(b, *o) for o in ob)
        if best is None or cost < best[0]:
            if best is not None:
                best[1].remove()
            best = (cost, t, b)
        else:
            t.remove()
        if cost == 0:
            break
    return best[2]


def lighten_axes(ax):
    ax.grid(True, which='major', color=GRID, lw=.6, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


NOTE = ('Mean of the six datasets. Each knob value is shared by\n'
        "all six (GMS: k capped at each dataset's M); lines are each\n"
        "method's Pareto front over its knob. SeMARC: λ = .5 (MM-Fi\n"
        'fold 2: 1); numbers next to the blue dots give γ.')


def plot_mean(a, mmn, out=f'{T}/fig/pareto_controllers_mean', box=((300, 700), (.735, .777)), ylim=None, xlim=None):
    """Left: the whole range (log latency). Middle: the box around SeMARC (linear latency). Right: legend."""
    ds = 'mean'
    plt.rcParams.update(pg.RC)
    fig = plt.figure(figsize=(12.4, 4.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1, .9])
    ax, az, lax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    draw(ax, a, ds, mmn=mmn)
    vis = a[(a.dataset == ds) & ((a.kind == 'table2') | a.on_front)]
    xs = np.r_[vis.latency_ms.values, mmn.latency_ms.values]; ys = np.r_[vis.f1.values, mmn.f1.values]
    lo, hi = xlim or (xs.min() / 1.35, xs.max() * 1.35)
    pg.style_axes(ax, ds, lo, hi)
    ax.set_title('Mean of six datasets (Table 2): full range', fontsize=10.5, color=INK, loc='left', fontweight='bold')
    ax.set_ylim(*(ylim or (ys.min() - .02, ys.max() + .012)))
    (x0, x1), (y0, y1) = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=MUTED, lw=.9, zorder=2))
    draw(az, a, ds)
    az.set_xlim(x0, x1); az.set_ylim(y0, y1)
    az.xaxis.set_major_locator(plt.FixedLocator([300, 400, 500, 600, 700]))
    az.yaxis.set_major_locator(plt.MaxNLocator(5))
    lighten_axes(az)
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
    for r in gamma_frame(a, ds, box).itertuples():                # γ labels above their dots where free
        big = {0.0: 6.5, 0.75: 4.0}.get(r.gamma, 2.6)
        placed.append(place_label(az, r.latency_ms, r.f1, kv(r.gamma), ob, placed, fs=8.4, rad_pt=big,
                                  order=('above', 'ul', 'ur', 'left', 'right', 'below', 'll', 'lr')))
    for m, text, order in ((EAMA, 'EAMA, Table 2', ('right', 'ur', 'lr', 'below', 'above', 'left', 'll', 'ul')),
                           (EAMA_W, 'weighted\nEAMA,\nTable 2', ('ll', 'left', 'below', 'lr', 'right', 'ul', 'ur', 'above'))):
        r = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'table2')].iloc[0]
        if x0 <= r.latency_ms <= x1 and y0 <= r.f1 <= y1:
            placed.append(place_label(az, r.latency_ms, r.f1, text, ob, placed, fs=8.0, rad_pt=7, order=order))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=250, bbox_inches='tight')
    plt.close(fig)


# ───────────────────────────── report ─────────────────────────────
def beaten(q, S):
    """Points of S with F1 >= q's and latency <= q's, one of them strictly."""
    return S[(S.f1 >= q.f1) & (S.latency_ms <= q.latency_ms) & ((S.f1 > q.f1) | (S.latency_ms < q.latency_ms))]


def semarc_sets(a, ds):
    g = a[(a.method == SEMARC) & (a.dataset == ds) & (a.kind == 'curve') & a.on_front]      # the drawn blue curve
    g = g.assign(label=[f'γ = {kv(v)}' for v in g.value])
    sw = pp.sweep_configs()
    sw = sw[sw.dataset == ds]
    sw = sw.assign(label=[f'γ={r.gamma:g}, λ={r.lam}' for r in sw.itertuples()])
    return g, sw


def versus(a, ds):
    """Each controller's front points and Table 2 point against SeMARC's γ points (Table 2's λ) and, as a supplement,
    against SeMARC's whole γ x λ sweep (its λ varied too, as the controllers' knobs are)."""
    g, sw = semarc_sets(a, ds)
    rows = []
    for m in [EAMA, EAMA_W, JAFA, GMSN]:
        pts = a[(a.method == m) & (a.dataset == ds) & (((a.kind == 'curve') & a.on_front) | (a.kind == 'table2'))]
        for q in pts.sort_values(['kind', 'latency_ms']).itertuples():
            b1, b2 = beaten(q, g), beaten(q, sw)
            le = g[g.latency_ms <= q.latency_ms]
            rows.append(dict(method=m, kind=q.kind, value=q.value, f1=q.f1, latency_ms=q.latency_ms,
                             beaten_gamma=', '.join(b1.label) if len(b1) else 'no',
                             semarc_best_f1_at_le=le.f1.max() if len(le) else np.nan,
                             beaten_gamma_lambda='yes' if len(b2) else 'no'))
    back = []
    for s in g.sort_values('latency_ms').itertuples():
        ctl = a[(a.dataset == ds) & (a.method != SEMARC) & (a.kind == 'curve')]
        b = beaten(s, ctl)
        back.append(dict(gamma=s.value, f1=s.f1, latency_ms=s.latency_ms,
                         beaten_by=', '.join(f'{r.method.replace("SeMA + ", "")} {KNOB[r.method]} = {kv(r.value)}' for r in b.itertuples()) or 'none'))
    return pd.DataFrame(rows), pd.DataFrame(back)


def f3(x):
    return '–' if pd.isna(x) else f'{x:.3f}'


def ms(x):
    return '–' if pd.isna(x) else (f'{x:.1f}' if x < 100 else f'{x:.0f}')


def kv(v):
    if isinstance(v, str):
        return v
    t = f'{v:g}'
    return t[1:] if t.startswith('0.') else t


def write_md(a, chk_jafa):
    L = ['# F1–latency curves of the Table 2 controllers (Table 2 metric)', '',
         "Request: for SeMA + EAMA, SeMA + Joint-AFA and SeMA + GMS, vary each controller's own parameter (retraining where "
         'needed) to get an F1–latency curve, and plot all curves with SeMARC\'s in one figure (`ctrl_curves.py`, '
         '`fig/pareto_controllers_mean.{pdf,png}`).', '',
         '## What was varied', '',
         '| Configuration | Knob | Values | New training? |', '|---|---|---|---|',
         "| SeMARC | γ (modality weights), λ held at Table 2's .5 (MM-Fi fold 2: 1) | 0, .25, .5, .75, 1, 1.5, 2 | no: the paper's γ-study nets |",
         '| SeMA + EAMA | λ in J(q) = score(q) − λ Σ z_j, uniform z_j = 1/M (Table 2\'s row) | .005, .01, .02, .05, .1, .2, .5, 1, 2 | no: test-time; the paper\'s value models (3 seeds × 3 folds) were already run at every λ |',
         '| SeMA + EAMA (weighted) | the same λ, z_j = extraction-time fraction (the weighted row) | same | no |',
         '| SeMA + Joint-AFA | r_cost, the per-acquisition cost in the reward | 0, .01, .02, .03, .05, .075, .1, .15, .2, .3, .5 | yes: retrained per value with the paper\'s code and seed (0), 20000 episodes |',
         '| SeMA + GMS | budget k of the greedy chain (selected on validation in Table 2) | 1 … M | no: the stored chains and test F1 at every k |', '',
         '- **Latency** is Table 2\'s estimate (Appendix `app:comp-replace-complete`).',
         '  - EAMA and Joint-AFA: `price_raw_selectors.py`, i.e. extraction of the acquired set, one fusion of it, and one selector-proxy call per decision.',
         '  - GMS: extraction plus one fusion, with no selector (`reprice_controllers_e2e.py`).',
         '  - The same per-modality idle-L40 tables throughout.',
         '- **F1**: macro-F1, mean over the three test folds (EAMA: also over its 3 value-model seeds).',
         '- **Mean of six datasets**: one knob value shared by all six datasets, as SeMARC\'s γ. GMS: k capped at each dataset\'s M.',
         '- **Curve**: each method\'s Pareto front over its knob values. Points another value of the same knob beats are not drawn.',
         '- **Table 2 configuration**: the published point, drawn as a black-edged marker.',
         '  - EAMA: λ chosen per fold and seed on validation.',
         '  - Joint-AFA: r_cost = .05, the paper\'s run.',
         '  - GMS: k chosen per dataset on validation.',
         '  - These are not knob values shared by all six datasets, so they need not lie on the curve.', '']
    L += ['## Checks', '',
          '- **Prefix caches.** In this environment the caches are the ones the paper\'s runs read (`ctrl/check_cache.py`):',
          '  - re-reading the paper\'s stored Joint-AFA per-sample sets gives its per-fold test F1 on all 18 folds;',
          '  - re-scoring GMS\'s stored sets gives its F1 at every k (|ΔF1| ≤ .00005, the stored rounding).',
          '- **Published points.** Every Table 2 point is recomputed from stored files and matches Table 2 / `tab:comp-replace-complete`:', '',
          '| Configuration | F1 (mean) | latency ms (mean) | Table 2 |', '|---|---|---|---|']
    t2 = {EAMA: '.744, 463.50', EAMA_W: '.761, 425.17', JAFA: '.749, 920.32', GMSN: '.733, 772 (48.4% above 520.17)'}
    for m in [EAMA, EAMA_W, JAFA, GMSN]:
        r = a[(a.method == m) & (a.dataset == 'mean') & (a.kind == 'table2')].iloc[0]
        L.append(f'| {m} | {r.f1:.4f} | {r.latency_ms:.2f} | {t2[m]} |')
    L += ['', '- **EAMA\'s fusion term away from the selected λ.** Only the selected λ has stored per-modality frequencies. '
          'Elsewhere the fusion term uses the mean set size: per-modality fusion steps differ by < 1% within a dataset. '
          f'At the selected λ the two prices differ by at most {np.nanmax(np.abs(a.latency_ms - a.check_ms)):.3f} ms.', '']
    if chk_jafa is not None and len(chk_jafa):
        L += ['- **Joint-AFA retraining device.** The paper\'s r_cost = .05 run was trained on a GPU. '
              'On the shared GPUs 2/3 (another user\'s jobs at 100%) the same code ran at < 10 episodes/s against ~100 on CPU, '
              'so every r_cost was retrained on CPU, .05 included. Same code, seed and data; only floating-point numerics differ. '
              'Table 2\'s marker is the paper\'s run; the curve uses the CPU runs.',
              '- **GPU determinism.** A GPU rerun of the paper\'s run (IEMOCAP fold 0, r_cost = .05; `ctrl/run_jafa.sh`) replayed the '
              'paper\'s trajectory exactly: replay buffer 8221 / 17189 / 26386 transitions at episodes 2000 / 4000 / 6000, as in '
              '`logs_raw_sel/jafa_raw_iemocap.log`. It was stopped there because it ran at 3 episodes/s. '
              'The CPU retrain at .05 differs from the paper\'s run by about as much as Joint-AFA\'s r_cost values differ from each other '
              '(table below): a single Joint-AFA run per fold is noisy, as in the paper.', '',
              '| Dataset | paper run F1 / MU / ms | CPU retrain at .05 F1 / MU / ms |', '|---|---|---|']
        for r in chk_jafa.itertuples():
            L.append(f'| {pp.LABEL[r.dataset].replace(" (Table 2)", "")} | {r.f1_paper:.3f} / {r.mu_paper:.2f} / {ms(r.ms_paper)} | '
                     f'{r.f1_cpu:.3f} / {r.mu_cpu:.2f} / {ms(r.ms_cpu)} |')
        L.append('')
    L += ['## Mean of six datasets', '']
    for m in [SEMARC, EAMA, EAMA_W, JAFA, GMSN]:
        g = a[(a.method == m) & (a.dataset == 'mean') & (a.kind == 'curve')].copy()
        g = g.sort_values('value')
        L += [f'**{m}** ({KNOB[m]})', '', f'| {KNOB[m]} | F1 | MU | latency (ms) | on the front |', '|---|---|---|---|---|']
        for r in g.itertuples():
            L.append(f'| {kv(r.value)} | {f3(r.f1)} | {r.mu:.2f} | {ms(r.latency_ms)} | {"yes" if r.on_front else ""} |')
        t = a[(a.method == m) & (a.dataset == 'mean') & (a.kind == 'table2')]
        for r in t.itertuples():
            L.append(f'| Table 2 configuration | {f3(r.f1)} | {r.mu:.2f} | {ms(r.latency_ms)} | |')
        L.append('')
    vs, back = versus(a, 'mean')
    L += ['## Who beats whom on the mean', '',
          "Beaten = some point has F1 ≥ and latency ≤, one strictly. 'SeMARC γ' = the blue curve (Table 2's λ). "
          "'SeMARC γ × λ' = the supplement's whole sweep (λ varied too, as the controllers' knobs are).", '',
          '| Controller | point | F1 | latency (ms) | beaten by SeMARC γ | SeMARC γ best F1 at ≤ its latency | beaten by SeMARC γ × λ |',
          '|---|---|---|---|---|---|---|']
    for r in vs.itertuples():
        pt = 'Table 2' if r.kind == 'table2' else f'{KNOB[r.method]} = {kv(r.value)}'
        L.append(f'| {r.method} | {pt} | {f3(r.f1)} | {ms(r.latency_ms)} | {r.beaten_gamma} | {f3(r.semarc_best_f1_at_le)} | {r.beaten_gamma_lambda} |')
    L += ['', '| SeMARC γ (front) | F1 | latency (ms) | beaten by a controller point |', '|---|---|---|---|']
    for r in back.itertuples():
        L.append(f'| {kv(r.gamma)} | {f3(r.f1)} | {ms(r.latency_ms)} | {r.beaten_by} |')
    L += ['', '## Per dataset: each method\'s front', '']
    for ds in pp.ORDER:
        L += [f'**{pp.LABEL[ds]}**', '', '| Method | front: knob → F1 @ ms | Table 2 configuration |', '|---|---|---|']
        for m in [SEMARC, EAMA, EAMA_W, JAFA, GMSN]:
            fr = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'curve') & a.on_front].sort_values('latency_ms')
            t = a[(a.method == m) & (a.dataset == ds) & (a.kind == 'table2')]
            ts = '; '.join(f'{f3(r.f1)} @ {ms(r.latency_ms)}' for r in t.itertuples()) or ('γ = 0 and .75 on the curve' if m == SEMARC else '')
            L.append(f'| {m} | ' + '; '.join(f'{kv(r.value)} → {f3(r.f1)} @ {ms(r.latency_ms)}' for r in fr.itertuples()) + f' | {ts} |')
        vs, back = versus(a, ds)
        nb = vs[(vs.kind == 'curve') & (vs.beaten_gamma == 'no')]
        bb = back[back.beaten_by != 'none']
        L += ['', f'- Controller front points not beaten by any SeMARC γ: {len(nb)} of {int((vs.kind == "curve").sum())}.',
              f'- SeMARC γ front points beaten by some controller point: {len(bb)} of {len(back)}'
              + (': ' + '; '.join(f'γ = {kv(r.gamma)} ({f3(r.f1)} @ {ms(r.latency_ms)}) by {r.beaten_by}' for r in bb.itertuples()) if len(bb) else '.'), '']
    open(f'{T}/res/CTRL_CURVES.md', 'w').write('\n'.join(L) + '\n')


def jafa_device_check(a):
    rows = []
    for ds in pp.ORDER:
        pf = jafa_runs(ds, 0.05)
        if pf is None:
            continue
        cpu = jafa_row(ds, 'x', .05, pf, '')
        p = a[(a.method == JAFA) & (a.dataset == ds) & (a.kind == 'table2')].iloc[0]
        rows.append(dict(dataset=ds, f1_paper=p.f1, mu_paper=p.mu, ms_paper=p.latency_ms, f1_cpu=cpu['f1'], mu_cpu=cpu['mu'],
                         ms_cpu=cpu['latency_ms']))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    a = load()
    a.to_csv(f'{T}/res/ctrl_points.csv', index=False)
    print(a[a.kind == 'table2'][['method', 'dataset', 'value', 'f1', 'mu', 'latency_ms', 'check_ms']].round(4).to_string())
    print(a[a.dataset == 'mean'][['method', 'kind', 'value', 'f1', 'mu', 'latency_ms', 'on_front']].round(4).to_string())
    pts = pp.paper_points()
    mmn = pts[(pts.dataset == 'mean') & (pts.method == 'MultiModN + ARC')]
    plot_mean(a, mmn)
    write_md(a, jafa_device_check(a))

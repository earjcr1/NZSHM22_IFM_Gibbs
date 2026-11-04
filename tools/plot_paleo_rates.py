#!/usr/bin/env python3
"""Paleo rates: each paleo row's target against the model's prediction, on linear and on log-log axes.

    python plot_paleo_rates.py VECTORS[:LABEL] [VECTORS[:LABEL] ...] [--problem cru_clean.npz] [--out FIGURE]
                               [--paleo-block NAME] [--slip-block S] [--bins-block MFD] [--label 4]
                               [--reference RATES[:LABEL]] [--ruptures RUPTURE_SET]

VECTORS is a result set's rate vectors as the notebook writes them (<prior folder>/rate_vectors, with or without
.npz). The mean model is their `mean` column, each rupture's posterior mean rate; since predictions are linear in the
rates, its prediction for a row is also the posterior mean of that row's prediction. When the runs that made the rate
vectors sit next to them (<prior folder>/main_rep0.npz, main_rep1.npz, ..., as the notebook writes them), the
predictions they recorded for every draw give each row's posterior spread, drawn as a bar from its 16% to its 84%
point. VECTORS may also be a single model's, such as the best fit that preliminaries.py writes (nnls_vectors.npz),
which has no spread; when every result set is one, the labels speak of the model rather than the mean model. Several
result sets give one panel each, side by side on the same axes (to compare priors, say). LABEL names a result set; by
default it is the rate vectors' folder (the prior).

Paleo rows: the rows of --paleo-block, or by default every row in neither the slip-rate block (--slip-block) nor the
magnitude-bin block (--bins-block) whose target is positive; for the NZSHM22 inputs these are the paleoseismic
event-rate rows (40 at 2e4, 65 at 211k). Rows with a target of 0 are zero-forcing penalty rows (one coefficient of
1e4 on one rupture), not rates; at 211k, 301 of them survive cleaning, and they are left out with a note.

Units: with --ruptures (the model's OpenSHA rupture set; the notebook passes it), rates per year: each row's sd is its
site's in the rupture set's paleo_constraint_data.json (the row's target is bestEstimate / sd, checked), so the target
is bestEstimate and a prediction the model's rate of detectable events at the site. Each target then carries
horizontal bars of its site's own 15.9-84.1% range and, fainter, its 2.3-97.7% range (lopsided, from the analysis), in
place of the shaded bands; chi2 and the counts within one or two sds stay in the Gaussian's sds, and the subtitle also
counts the sites within their own ranges. Each result set gets a row of two panels, linear and log-log. Without
--ruptures, the units below.
The input rows are divided by their uncertainties, which the inputs do not give, so targets and predictions are
in units of each row's uncertainty. On log-log axes that moves a point only along the 1:1 line: its distance from the
line, the log of prediction / target, is the same as it would be in rates. A target in these units is one over its
relative uncertainty, so the best-measured sites lie furthest right. The shaded bands mark predictions within one and
within two uncertainties of their targets.

Axes: fixed for the problem by its targets and its best fit (all of the targets, and the central 96% of the best fit's
predictions, from prelim_results.npz next to --problem), so that every model's figure for one problem has the same
axes; without a best fit, they follow what is plotted.

Marks: a dot per paleo row at its target and mean prediction; the posterior 16-84% bar of its prediction, when the
runs are found (a bar reaching an axis continues beyond it, below possibly to 0); predictions beyond the axes,
including 0, on their edges as triangles; the 1:1 line; the
bands; and the original row numbers of the --label rows furthest from their targets in units of uncertainty (default
4). Each row's dot and bar share a colour, cycling along the targets through the ten colours
(ColorBrewer's RdYlBu, 10 classes, its red and blue ends alternating: cru_common.RDYLBU10_CYCLE and neighbour_colours; outlined in grey, as several are pale), so that neighbouring rows never
share one; the colours only tell neighbours apart, and a row keeps its colour in every model's figure. Writes FIGURE (default: fig_paleo_rates.png next to the first rate vectors) and, as its table view, the same name
with .txt: the mean model's chi-squared over the paleo rows, the shares within one and two uncertainties, and per row
the target, the mean prediction, the posterior 16%, 50% and 84% points, the difference and the ratio.

Kept draws: when the rate vectors hold the kept draws that select_draws.py adds (REPRESENTATIVES above 0, the
notebook's default; about 3,000), each one's predictions are drawn as x's at their rows' targets in place of the
16-84% bars, in the rows' colours darkened a little and faint, so that where the draws gather shows darkest
(cru_common.draw_xs); those beyond the axes (0 included) sit on their edges.
Their predictions are read from those the runs recorded (cru_common.kept_predictions; one draw per run is
recomputed from its rates as a check), and without kept draws, or if the runs do not match them, the bars are drawn
instead; the tables keep the 16%, 50% and 84% points over every recorded draw.

Reference: --reference RATES[:LABEL] marks another model's predictions on every panel as open black squares (triangles on the edges), such as the
model's own OpenSHA solution, its solution/rates.csv (the notebook passes it when DATA_DIR holds the solution folder);
LABEL defaults to the solution folder's name (shortened when long). The table view adds its chi-squared and shares, and a column of its
predictions. Its rates on ruptures that cleaning removed are left out of its predictions, and the table says how much.
"""
import argparse
import os
import textwrap

import numpy as np

from cru_common import (RDYLBU10, RDYLBU10_CYCLE, axis_span, best_fit_rates, darker, draw_xs, kept_draws, kept_predictions,
                        legend_above, legend_height, load_problem, neighbour_colours, reference_rates, run_predictions,
                        run_settings, settings_lines)

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BAND, BAND2 = '#ebeae5', '#f2f1ec'               # within one and within two uncertainties of the target
REF = INK                                        # the reference model's marks: black, apart from every series colour


def parse_set(s):
    """VECTORS[:LABEL] -> (path of the .npz, label)."""
    path, label = s, None
    if ':' in s:
        head, tail = s.rsplit(':', 1)
        if tail and '/' not in tail and os.sep not in tail:
            path, label = head, tail
    base = path[:-4] if path.endswith(('.npz', '.csv')) else path
    return base + '.npz', label or os.path.basename(os.path.dirname(os.path.abspath(base))) or base


def draw_settings(ax, tab, y, size=7.5):
    """The run's settings above a panel, in place of its label: names (muted) over values, the last line y points above."""
    rows = [r for pair in tab for r in pair]
    for k, text in enumerate(rows):
        ax.annotate(text, (0, 1), xycoords='axes fraction', xytext=(0, y + (len(rows) - 1 - k) * size * 1.4),
                    textcoords='offset points', va='bottom', fontsize=size, family='monospace',
                    color=MUTED if k % 2 == 0 else INK)


def scatter_edges(ax, x, v, lo, hi, kw, kw_edge):
    """Points within lo-hi at their values; those beyond, on the axis's edge as triangles (down below, up above). The
    colour c may be one per point."""
    low, high = v < lo, v > hi
    mid = ~(low | high)
    pick = lambda d, sel: {k: (np.asarray(c)[sel] if k == 'c' and np.ndim(c) == 1 and len(c) == len(x) else c)
                           for k, c in d.items()}
    ax.scatter(x[mid], v[mid], **pick(kw, mid))
    for sel, at, mk in ((low, lo, 'v'), (high, hi, '^')):
        if sel.any():
            ax.scatter(x[sel], np.full(int(sel.sum()), at), marker=mk, clip_on=False, **{'linewidths': 0, **pick(kw_edge, sel)})


def main():
    ap = argparse.ArgumentParser(description='Paleo-rate targets against the model (see the module docstring).')
    ap.add_argument('sets', nargs='+', metavar='VECTORS[:LABEL]', help='rate vectors (<prior folder>/rate_vectors)')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--out', default=None, help='figure (default: fig_paleo_rates.png next to the first rate vectors)')
    ap.add_argument('--paleo-block', default=None, help='the paleo row group (default: every row in neither the slip-rate '
                    'nor the magnitude-bin group whose target is positive)')
    ap.add_argument('--slip-block', default='S', help='the slip-rate row group (default S)')
    ap.add_argument('--bins-block', default='MFD', help='the magnitude-bin row group (default MFD)')
    ap.add_argument('--label', type=int, default=4, help='label this many rows furthest from their targets (default 4)')
    ap.add_argument('--reference', default=None, metavar='RATES[:LABEL]', help="another model's rates to mark on every "
                    "panel, such as the model's OpenSHA solution (its solution/rates.csv), or rate vectors (.npz)")
    ap.add_argument('--config', default=None, help="the run's workflow_config.json, whose settings (SLIP_SECTIONS to "
                    "FLOOR) replace the label of a single result set (default: the one next to --problem, if any; "
                    "'none': the label)")
    ap.add_argument('--ruptures', default=None, help='the OpenSHA rupture set (or a solution folder holding one), for rates '
                    'per year and each site\'s own ranges (default: in units of each row\'s uncertainty)')
    a = ap.parse_args()
    sets = [parse_set(s) for s in a.sets]
    out = a.out or os.path.join(os.path.dirname(sets[0][0]) or '.', 'fig_paleo_rates.png')

    P = load_problem(a.problem)
    names = list(P['block_names'])
    Gr = P['G'].tocsr()
    nonempty = np.diff(Gr.indptr) > 0
    if a.paleo_block is not None:
        if a.paleo_block not in names:
            raise SystemExit(f'no row block named {a.paleo_block!r}; the blocks are {names}')
        rows = np.flatnonzero((P['block'] == names.index(a.paleo_block)) & nonempty)
        which = f'row block {a.paleo_block!r}'
    else:
        other = [names.index(b) for b in (a.slip_block, a.bins_block) if b in names]
        rows = np.flatnonzero(~np.isin(P['block'], other) & nonempty)
        zero = rows[P['d'][rows] <= 0]                   # zero-forcing penalty rows (target 0) are not paleo rates
        rows = rows[P['d'][rows] > 0]
        rest = sorted({names[b] for b in P['block'][rows]})
        which = (f'the rows in neither {a.slip_block!r} nor {a.bins_block!r}: row block{"s" if len(rest) > 1 else ""} '
                 + ', '.join(map(repr, rest)))
        if len(zero):
            which += (f'; {len(zero)} other rows with targets of 0 or less left out (zero-forcing penalty rows, '
                      f'{len({names[b] for b in P["block"][zero]})} row blocks)')
    if len(rows) == 0:
        raise SystemExit(f'no paleo rows found ({which}); name them with --paleo-block')
    orig = np.asarray(P['keep_rows'])[rows]
    scale = P['row_scale'][rows]                         # 1: cleaning leaves the paleo rows as the inputs give them
    u = np.ones(len(rows))                               # each row's display unit: its site's sd per year (--ruptures), or 1
    phys = bool(a.ruptures)
    if phys:
        from cru_common import find_rupture_set
        from paleo_misfit import load_sites
        rd = find_rupture_set(a.ruptures)
        if rd is None:
            raise SystemExit(f'--ruptures {a.ruptures}: no OpenSHA rupture set there')
        sites = load_sites(rd)
        first = int(P['raw_shape'][0]) - len(sites)      # the sites are the inputs' last rows, in the file's order
        si = np.asarray(orig) - first
        if (si < 0).any() or (si >= len(sites)).any():
            raise SystemExit('--ruptures: the paleo rows are not the inputs\' last rows, the sites of paleo_constraint_data.json')
        u = np.array([sites[i]['sd'] for i in si])
        if not np.allclose(P['d'][rows] / scale, [sites[i]['best'] / sites[i]['sd'] for i in si], rtol=1e-9, atol=0):
            raise SystemExit('--ruptures: the paleo rows\' targets are not bestEstimate / sd of paleo_constraint_data.json')
        QR = np.array([sites[i]['q'] for i in si])        # each site's 2.275, 15.865, 84.135 and 97.725% rates
    sc = scale / u                                       # a row's value / sc: per year, or in units of its sd
    dw = P['d'][rows]                                    # targets as the likelihood sees them
    t = dw / sc                                          # ... per year, or in units of each row's uncertainty
    Gp = Gr[rows]
    keep_c = np.asarray(P['keep_cols'])
    if (t <= 0).any():
        raise SystemExit(f'{int((t <= 0).sum())} paleo targets are not positive; a log-log plot cannot show them')
    head = [f'paleo rows: {len(rows)} ({which}; original rows {orig.min()}-{orig.max()}); targets from {t.min():.3g} to '
            f'{t.max():.3g}; ' + ('rates per year (each row\'s sd its site\'s in paleo_constraint_data.json)' if phys else
                                  'targets and predictions in units of each row\'s uncertainty (a target is one over its '
                                  'relative uncertainty)')]
    data = []
    for path, label in sets:
        z = np.load(path)
        m_all = np.asarray(z['mean'], np.float64)
        n_raw = P['raw_shape'][1] if P.get('raw_shape') else len(m_all)
        if len(m_all) != n_raw or keep_c.max() >= len(m_all):
            raise SystemExit(f'{path} has {len(m_all)} ruptures; the problem {a.problem} has {n_raw} (another problem?)')
        yw = Gp @ m_all[keep_c]                          # predictions as the likelihood sees them
        y = yw / sc
        r = y - t
        model = str(z['model']) if 'model' in z.files else ''          # a single model (a best fit): no spread
        Y, used = (None, []) if model else run_predictions(os.path.dirname(path) or '.', rows, Gr.shape[0], sc)
        q = np.percentile(Y, [16, 50, 84], axis=0) if Y is not None else None
        K, kidx = (None, None) if model else kept_draws(z)
        Xk = None if K is None else kept_predictions(os.path.dirname(path) or '.', kidx, rows, Gr.shape[0], sc,
                                                      check=(Gp, K, keep_c))   # the kept draws' predictions
        del K
        data.append(dict(path=path, label=label, y=y, r=r, ratio=y / t, q=q, used=used, ndraws=0 if Y is None else len(Y),
                         xs=Xk,
                         chi2=float(np.sum((yw - dw) ** 2)), in1=float(np.mean(np.abs(r / u) <= 1)),
                         in2=float(np.mean(np.abs(r / u) <= 2)), groups=list(np.argsort(-np.abs(r / u), kind='stable')[:max(0, a.label)]),
                         own1=float(np.mean((y >= QR[:, 1]) & (y <= QR[:, 2]))) if phys else None,   # within the site's
                         own2=float(np.mean((y >= QR[:, 0]) & (y <= QR[:, 3]))) if phys else None,   # own ranges
                         model=model))
    mm = 'model' if all(D['model'] for D in data) else 'mean model'
    ref = None
    if a.reference:                                      # another model's predictions, marked on every panel
        ref = reference_rates(a.reference, P)
        yw = Gp @ ref['m']
        r = yw / sc - t
        ref.update(y=yw / sc, chi2=float(np.sum((yw - dw) ** 2)), in1=float(np.mean(np.abs(r / u) <= 1)),
                   in2=float(np.mean(np.abs(r / u) <= 2)), q=np.percentile(yw / sc / t, [5, 50, 95]))
        head.append(f'reference: {ref["label"]} ({ref["path"]}): its chi2 over the paleo rows {ref["chi2"]:.1f}; '
                    f'{100 * ref["in1"]:.0f}% of rows within one uncertainty of their targets, {100 * ref["in2"]:.0f}% '
                    f'within two; prediction / target: median {ref["q"][1]:.2f} (5-95%: {ref["q"][0]:.2f} to '
                    f'{ref["q"][2]:.2f})' + (f'; {ref["note"]}' if ref['note'] else ''))
    rc = f' {"reference":>10s}' if ref else ''            # the reference's column in the table
    settings = run_settings(a.config, a.problem) if len(data) == 1 else None   # the run's settings replace the label
    if settings and data[0]['model']:
        settings = [('model', data[0]['model'])] + settings
    if settings:
        head.append('settings: ' + ', '.join(f'{n} {v}' for n, v in settings))

    # table view
    lines = list(head)
    for D in data:
        rq = np.percentile(D['ratio'], [5, 50, 95])
        spread = (f'posterior spread from the {D["ndraws"]} draws recorded by {", ".join(D["used"])}' if D['q'] is not None
                  else f'{D["model"]}: a single model, so no posterior spread' if D['model']
                  else 'no runs found next to the rate vectors, so no posterior spread')
        if D['xs'] is not None:
            spread += f'; the x\'s: the {len(D["xs"])} kept draws\' predictions (select_draws.py)'
        lines += ['', f'{D["label"]} ({D["path"]}): the {mm}\'s chi2 over the paleo rows {D["chi2"]:.1f}; '
                  f'{100 * D["in1"]:.0f}% of rows within one uncertainty of their targets, {100 * D["in2"]:.0f}% within two; '
                  f'prediction / target: median {rq[1]:.2f} (5-95%: {rq[0]:.2f} to {rq[2]:.2f}); {spread}; furthest from their '
                  'targets: ' + '; '.join(f'row {orig[k]} ({D["r"][k] / u[k]:+.2f} sd)' for k in D['groups'])
                  + (f'; {100 * D["own1"]:.0f}% of sites within their own 15.9-84.1% ranges, {100 * D["own2"]:.0f}% within '
                     '2.3-97.7%' if phys else ''),
                  f'{"row":>7s} {"target":>9s} {"prediction" if D["model"] else "mean prediction":>16s} {"16%":>9s} {"50%":>9s} {"84%":>9s} '
                  f'{"difference":>11s} {"in sds":>7s} {"ratio":>7s}{rc}']
        for k in np.argsort(orig, kind='stable'):
            qs = ' '.join(f'{v:9.4g}' for v in D['q'][:, k]) if D['q'] is not None else ' '.join(f'{"-":>9s}' for _ in range(3))
            lines.append(f'{orig[k]:7d} {t[k]:9.4g} {D["y"][k]:16.4g} {qs} {D["r"][k]:+11.4g} {D["r"][k] / u[k]:+7.2f} '
                         f'{D["ratio"][k]:7.3f}' + (f' {ref["y"][k]:10.4g}' if ref else ''))
    text = '\n'.join(lines)
    print('\n'.join(l for l in lines if not l[:7].strip().isdigit() and not l.lstrip().startswith('row ')))

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    col = neighbour_colours(t, RDYLBU10_CYCLE)                       # each row's colour, unlike its neighbours'
    nn = best_fit_rates(a.problem)                   # the best fit's predictions fix the axes, the same for every model
    fixed = nn is not None and len(nn) == Gp.shape[1]
    if fixed:
        y_nn = (Gp @ nn) / sc
        lo_b, hi_b = axis_span(t, y_nn, log=True)
        hi_lin = 1.05 * axis_span(t, y_nn, log=False)[1]
    else:                                            # cleaned without the best fit: the axes follow what is plotted
        vals = np.concatenate([np.r_[D['y'], D['q'][0] if D['q'] is not None else [], D['q'][2] if D['q'] is not None else []]
                               for D in data])
        if ref is not None:
            vals = np.r_[vals, ref['y']]
        pos = vals[vals >= t.min() / 100]
        lo_b, hi_b = min(t.min(), pos.min() if pos.size else t.min()), max(t.max(), vals.max())
        hi_lin = 1.05 * max(t.max(), vals.max())
    lo, hi = lo_b / 1.4, hi_b * 1.4
    edge, top, top_lin = lo * 1.12, hi / 1.12, hi_lin * 0.985   # where predictions beyond the axes sit
    allv = [D['y'] for D in data] + ([ref['y']] if ref else [])
    clipped = any(((v < edge) | (v > top) | (v > top_lin)).any() for v in allv)
    axes_line = (f'axes: ' + ('fixed by the targets and the best fit (NNLS; the central 96% of its predictions)' if fixed
                             else 'from the targets and the predictions shown (no best fit in the cleaning results)')
                 + f': log {lo:.3g} to {hi:.3g}, linear 0 to {hi_lin:.3g}; predictions beyond sit on the edges')
    text += '\n' + axes_line
    print(axes_line)
    nrow = len(data)
    kx = any(D['xs'] is not None for D in data)          # the kept draws, as x's
    bars = any(D['q'] is not None and D['xs'] is None for D in data)
    note = (('Rates per year: each target is bestEstimate, with bars of its site\'s own 15.9–84.1% range and, fainter, '
             'its 2.3–97.7% range from the paleoseismic analysis; a prediction is the model\'s rate of detectable events at '
             'the site. ' if phys else
             'In units of each row\'s uncertainty: on log axes that moves a point only along the 1:1 line, so its distance '
             'from the line is the ratio of prediction to target. A target is one over its relative uncertainty, so the '
             'best-measured sites lie furthest right. ')
            + ('Each x is one kept draw\'s prediction, faint, so that where the draws gather shows darkest; those beyond '
               'the axes (0 included) sit on their edges. ' if kx else
               'A bar that reaches an axis continues beyond it (below, possibly to 0). ' if bars else '')
            + 'Colours only tell neighbouring rows apart.'
            + (' The axes are fixed by the targets and the best fit, the same for every model of these inputs; '
               'predictions beyond them sit on the edges.' if fixed else ''))
    note = textwrap.fill(note, 180, break_on_hyphens=False)
    extra = 0.16 * note.count('\n')
    n_h = 4 + int(bars) + int(kx) + int(clipped) + int(ref is not None)   # the legend's entries
    lc = 3 if n_h > 3 else n_h
    leg_pt = legend_height(n_h, lc) + 6
    tab = settings_lines(settings, int(9.7 / 0.062500)) if settings else None
    top_in, row_in, bot_in = 0.45 + leg_pt / 72, 5.6, 0.8 + extra   # figure layout, in inches
    if tab:
        top_in += max(0.0, 2 * len(tab) * 10.5 - 14.0) / 72
    fig, axs = plt.subplots(nrow, 2, figsize=(10.6, top_in + row_in * nrow + bot_in), squeeze=False, facecolor=SURFACE)
    halo = [pe.withStroke(linewidth=2.6, foreground=SURFACE)]
    outline = [pe.Stroke(linewidth=1.7, foreground=MUTED), pe.Normal()]   # the pale colours' bars stand out from the surface
    fmt = FuncFormatter(lambda v, _: f'{v:g}')
    for i, D in enumerate(data):
        for j, ax in enumerate(axs[i]):
            log = j == 1
            ax.set_facecolor(SURFACE)
            for s_ in ('top', 'right'):
                ax.spines[s_].set_visible(False)
            for s_ in ('left', 'bottom'):
                ax.spines[s_].set_color(AXIS)
                ax.spines[s_].set_linewidth(0.5)
            ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=8.5, which='both', length=3, width=0.5)
            if log:
                ax.set_xscale('log')
                ax.set_yscale('log')
                for axis in (ax.xaxis, ax.yaxis):
                    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
                    axis.set_major_formatter(fmt)
                    axis.set_minor_formatter(NullFormatter())
                lim = (lo, hi)
                xs = np.logspace(np.log10(t.min() / 1.25), np.log10(min(hi, 1.25 * t.max())), 400)   # where the targets are
            else:
                lim = (0.0, hi_lin)
                xs = np.linspace(0.0, hi_lin, 400)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_box_aspect(1)
            ax.grid(True, which='major', color=GRID, lw=0.5, zorder=0)
            lo_e, hi_e = (edge, top) if log else (-np.inf, top_lin)
            if phys:                                 # each site's own ranges, at the prediction's height
                yb = np.clip(D['y'], max(lo_e, lim[0]), min(hi_e, lim[1]))
                ax.hlines(yb, np.clip(QR[:, 0], *lim), np.clip(QR[:, 3], *lim), colors=col, lw=0.8, alpha=0.25, zorder=2.05)
                ax.hlines(yb, np.clip(QR[:, 1], *lim), np.clip(QR[:, 2], *lim), colors=col, lw=1.6, alpha=0.55, zorder=2.1)
            else:
                ax.fill_between(xs, np.maximum(xs - 2.0, lim[0]), xs + 2.0, color=BAND2, lw=0, zorder=0.4)
                ax.fill_between(xs, np.maximum(xs - 1.0, lim[0]), xs + 1.0, color=BAND, lw=0, zorder=0.5)
            ax.plot(lim, lim, color=INK, lw=1.0, zorder=2)
            if ref is not None:                      # the reference's predictions, in front of the posterior's
                scatter_edges(ax, t, ref['y'], lo_e, hi_e, dict(s=26, marker='s', facecolors='none', edgecolors=REF,
                              linewidths=0.45, zorder=3.2), dict(s=28, facecolors='none', edgecolors=REF, linewidths=0.45, zorder=3.2))
            if D['xs'] is not None:                  # each kept draw's predictions, as x's, within the axes
                draw_xs(ax, np.broadcast_to(t, D['xs'].shape), np.clip(D['xs'], *lim), np.broadcast_to(col, D['xs'].shape),
                        alpha=0.08)
            elif D['q'] is not None:                 # posterior 16-84% of each prediction, within the axes
                ax.vlines(t, np.clip(D['q'][0], *lim), np.clip(D['q'][2], *lim), colors=col, lw=1.3, zorder=2.2,
                          path_effects=outline)
            ec, ew = MUTED, 0.45                     # a thin grey edge, so the pale colours show
            scatter_edges(ax, t, D['y'], lo_e, hi_e, dict(s=28, c=col, edgecolors=ec, linewidths=ew, zorder=3),
                          dict(s=26, c=col, edgecolors=ec, linewidths=ew, zorder=3))
            # labels in a column in the empty corner on their points' side of the 1:1 line (lower right for points
            # predicted below their targets, upper left for those above), each near its point's height, with leader lines
            frac = ((lambda v: (np.log10(v) - np.log10(lim[0])) / (np.log10(lim[1]) - np.log10(lim[0]))) if log
                    else (lambda v: (v - lim[0]) / (lim[1] - lim[0])))
            yclip = lambda v: min(max(v, lo_e if log else lim[0]), hi_e)
            step = 0.045                             # label spacing, in axes fractions
            for side in (-1, 1):
                ks = [k for k in D['groups'] if (D['r'][k] < 0) == (side < 0)]
                if not ks:
                    continue
                ks.sort(key=lambda k_: yclip(D['y'][k_]) * side)        # from the corner's edge: top down, or bottom up
                for i_ in range(len(ks) - 1):        # near-level neighbours: the one nearer the column takes the first slot
                    fa, fb = (frac(yclip(D['y'][k_])) for k_ in ks[i_:i_ + 2])
                    if abs(fa - fb) < step and (t[ks[i_ + 1]] > t[ks[i_]]) == (side < 0):
                        ks[i_], ks[i_ + 1] = ks[i_ + 1], ks[i_]
                prev = None
                for k in ks:
                    yk = yclip(D['y'][k])
                    if side < 0:                     # column at x = 0.80, at least 0.1 below the 1:1 line there
                        ty = min(frac(yk), 0.70) if prev is None else min(frac(yk), 0.70, prev - step)
                        ty, x0 = max(ty, 0.02), 0.80
                    else:                            # column at x = 0.04, at least 0.1 above the 1:1 line there
                        ty = max(frac(yk), 0.27) if prev is None else max(frac(yk), 0.27, prev + step)
                        ty, x0 = min(ty, 0.98), 0.04
                    prev = ty
                    ax.annotate(f'row {orig[k]}', (t[k], yk), xytext=(x0, ty), textcoords='axes fraction', fontsize=7.5,
                                color=INK2, va='center', ha='left', path_effects=halo, zorder=5,
                                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5, shrinkA=2, shrinkB=4))
            ax.set_xlabel('paleo rate target (per year)' if phys else 'paleo rate target (in units of its uncertainty)',
                          color=INK2, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{mm} prediction (per year)' if phys else f'{mm} prediction (same units)', color=INK2, fontsize=9)
            ax.set_title('log scale' if log else 'linear scale', loc='left', color=INK, fontsize=10, pad=6)
        sub = (f'{mm}: χ² {D["chi2"]:.1f} over the {len(rows)} paleo rows; within one sd {100 * D["in1"]:.0f}%, two '
               f'{100 * D["in2"]:.0f}%' + (f'; within the sites\' own 15.9–84.1% ranges {100 * D["own1"]:.0f}%, 2.3–97.7% '
                                           f'{100 * D["own2"]:.0f}%' if phys else ''))
        above = 40 + (leg_pt if i == 0 else 0)          # the first row's settings sit above the legend
        if tab:
            draw_settings(axs[i][0], tab, above)
        else:
            axs[i][0].annotate(D['label'], (0, 1), xycoords='axes fraction', xytext=(0, above), textcoords='offset points',
                               va='bottom', color=INK, fontsize=11.5, fontweight='bold')
        axs[i][0].annotate(sub, (0, 1), xycoords='axes fraction', xytext=(0, 25), textcoords='offset points',
                           va='bottom', color=INK2, fontsize=9)
    few = RDYLBU10[1], RDYLBU10[8], RDYLBU10[3]            # the legend shows a few of the colours
    handles = [(tuple(Line2D([], [], ls='none', marker='o', ms=5.3, mfc=c, mec=MUTED,
                             mew=0.45) for c in few), 'prediction' if mm == 'model' else 'mean prediction')]
    if bars:
        handles.append((tuple(Line2D([], [], color=c, lw=1.3, path_effects=outline) for c in few),
                        'posterior 16–84% of the prediction'))
    if kx:
        n_k = max(len(D['xs']) for D in data if D['xs'] is not None)
        handles.append((tuple(Line2D([], [], ls='none', marker='x', ms=4, mew=0.8, color=darker(c)) for c in few),
                        f'each of the {n_k:,} kept draws\' prediction'))
    if ref is not None:
        handles.append((Line2D([], [], ls='none', marker='s', ms=5, mfc='none', mec=REF, mew=0.45),
                        f'{ref["label"]} (χ² {ref["chi2"]:.1f})'))
    handles.append((Line2D([], [], color=INK, lw=1.0), 'prediction = target'))
    if phys:
        handles += [(tuple(Line2D([], [], color=c, lw=1.6, alpha=0.55) for c in few), 'the site\'s 15.9–84.1% range'),
                    (tuple(Line2D([], [], color=c, lw=0.8, alpha=0.25) for c in few), 'its 2.3–97.7% range')]
    else:
        handles += [(Patch(facecolor=BAND, edgecolor='none'), 'within one uncertainty'),
                    (Patch(facecolor=BAND2, edgecolor='none'), 'within two')]
    if clipped:
        handles.append((Line2D([], [], ls='none', marker='v', ms=5, mfc=INK2, mec=INK2), 'beyond the axes (on their edges)'))
    h_in = fig.get_size_inches()[1]
    fig.suptitle('Paleo rates: targets against the model\'s predictions', x=0.035, ha='left', color=INK, fontsize=13,
                 fontweight='bold', y=1 - 0.2 / h_in, va='top')
    legend_above(fig, axs[0][0], handles, lc, 40, labelcolor=INK2)
    fig.text(0.5, 0.12 / h_in, note, ha='center', va='baseline' if not extra else 'bottom', color=MUTED, fontsize=8,
             linespacing=1.4)
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - (top_in + 0.55) / h_in, bottom=bot_in / h_in, wspace=0.22,
                        hspace=0.55 / (row_in - 0.55))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    with open(os.path.splitext(out)[0] + '.txt', 'w') as f:
        f.write(text + '\n')
    print(f'wrote {out} and {os.path.splitext(out)[0]}.txt')


if __name__ == '__main__':
    main()

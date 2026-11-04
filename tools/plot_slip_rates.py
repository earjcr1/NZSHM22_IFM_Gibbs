#!/usr/bin/env python3
"""Slip rates: each fault section's slip-rate target against the mean model's prediction, on linear and on log scales.

    python plot_slip_rates.py VECTORS[:LABEL] [VECTORS[:LABEL] ...] [--problem cru_clean.npz] [--out FIGURE]
                              [--slip-block S] [--label 3] [--reference RATES[:LABEL]] [--ruptures RUPTURE_SET]

VECTORS is a result set's rate vectors as the notebook writes them (<prior folder>/rate_vectors, with or without
.npz). The mean model is their `mean` column, each rupture's posterior mean rate; since predictions are linear in the
rates, its prediction for a row is also the posterior mean of that row's prediction. When the runs that made the rate
vectors sit next to them (<prior folder>/main_rep0.npz, main_rep1.npz, ..., as the notebook writes them), the
predictions they recorded for every draw give each section mean's posterior spread, drawn as a bar from its 16% to
its 84% point. VECTORS may also be a single model's, such as the best fit that preliminaries.py writes
(nnls_vectors.npz), which has no spread; when every result set is one, the labels speak of the model rather than the
mean model. Several result sets give one row of panels each (to compare priors, say). LABEL names a result set; by
default it is the rate vectors' folder.

Sections: a fault section's slip rate is one measurement, copied in the inputs to each of its subsections. The cleaned
problem records each slip-rate row's section (preliminaries.py --slip-sections; for problems cleaned before that, runs
of copies are detected here, for display only). Each section's mean prediction (the mean over its subsections) is
plotted against its target; the table view also lists every subsection.

Units: with --ruptures (the model's OpenSHA rupture set; the notebook passes it), slip rates in mm/yr: each row's
sd in m/yr is its coefficients over the rupture set's average slips (plate_rate.check_problem), so a row's target and
predictions times it are slip rates (the target the slip-rate factor times the measured rate, as the likelihood
compares them). Each section's target then carries a horizontal bar of one sd either side, as the likelihood has it
(the inputs' Gaussian's, or with slip rates from the CFM each side's own), in place of the shaded band; chi2 and the
counts within one or two sds are in sds, as before. Without --ruptures, the units below.
The input rows are already divided by their uncertainties, which the inputs do not give, so targets and
predictions are in units of each section's measurement uncertainty (any 1/sqrt(k) weighting applied when cleaning is
undone for display). The 1:1 line is unaffected, and a section mean within the shaded band, between target - 1 and
target + 1, is within one measurement uncertainty of its target. With one measurement per section (the default), a
single subsection may stray further: in the likelihood it carries sqrt(k) times that uncertainty. On the log scale the
band reaches the bottom wherever the target is below 1: a prediction of 0 is then within one uncertainty.
With slip rates from the CFM (preliminaries.py --cfm-sliprates), each row's unit is the mean of its two sides' sds
(sigma_ref); the band shows that, while chi2 and the counts within one or two uncertainties use the side each
residual is on (its sd below or above the target).

Axes: fixed for the problem by its targets and its best fit (all of the targets, and the central 96% of the best fit's
section means, from prelim_results.npz next to --problem), so that every model's figure for one problem has the same
axes; without a best fit, they follow what is plotted. Marks: a dot per section mean; the
posterior 16-84% bar of each section mean, when the runs are found (a bar reaching an axis continues beyond it, below on
the log scale possibly to 0); values beyond the axes (on the log scale, including 0) on their edges as triangles; the
1:1 line; the +-1 band; and the original rows of the --label sections furthest from their targets (default 3). Each
section's dot and bar share a colour, cycling along the targets through the ten of Tableau's Color Blind 10
(cru_common.CATEGORICAL and neighbour_colours; outlined in grey, as several are pale), so that neighbouring sections
never share one; the colours only tell neighbours apart,
and a section keeps its colour in every model's figure. Sections that share a target (the inputs' slip rates are often
round numbers: at 211k, 403 of 507 sections share theirs) stay stacked at it, one bar over another. Writes FIGURE
(default: fig_slip_rates.png next to the first rate vectors) and, as its table view, the
same name with .txt: the mean model's chi-squared over the slip-rate rows as the likelihood weights them, the share of
sections (and of subsections) within one and two uncertainties, per section the target, the mean prediction, its
posterior 16%, 50% and 84% points, their difference and their ratio, and per row the same without the posterior points.

Kept draws: when the rate vectors hold the kept draws that select_draws.py adds (REPRESENTATIVES above 0, the
notebook's default; about 3,000), each one's section means are drawn as x's at their sections' targets in place of the
16-84% bars, in the sections' colours darkened a little and faint, so that where the draws gather shows darkest
(cru_common.draw_xs); those beyond the axes (on the log scale, 0 included) sit on their edges.
Their predictions are read from those the runs recorded (cru_common.kept_predictions; one
draw per run is recomputed from its rates as a check), and without kept draws, or if the runs do not match them, the
bars are drawn instead; the tables keep the 16%, 50% and 84% points over every recorded draw.

Reference: --reference RATES[:LABEL] marks another model's section means on every panel as open black squares (triangles on the edges), such as the
model's own OpenSHA solution, its solution/rates.csv (the notebook passes it when DATA_DIR holds the solution folder);
LABEL defaults to the solution folder's name (shortened when long). The table view adds its chi-squared and shares, and a column of its
predictions. Its rates on ruptures that cleaning removed are left out of its predictions, and the table says how much.
"""
import argparse
import os
import textwrap

import numpy as np

from cru_common import split_weights
from cru_common import (CATEGORICAL, axis_span, best_fit_rates, darker, draw_xs, kept_draws, kept_predictions,
                        legend_above, legend_height, load_problem, neighbour_colours, reference_rates, run_predictions,
                        run_settings, settings_lines, slip_sections)

SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ebeae5'
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
    ap = argparse.ArgumentParser(description='Slip-rate targets against the mean model (see the module docstring).')
    ap.add_argument('sets', nargs='+', metavar='VECTORS[:LABEL]', help='rate vectors (<prior folder>/rate_vectors)')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--out', default=None, help='figure (default: fig_slip_rates.png next to the first rate vectors)')
    ap.add_argument('--slip-block', default='S', help='the slip-rate row group (default S)')
    ap.add_argument('--label', type=int, default=3, help='label this many sections furthest from their targets (default 3)')
    ap.add_argument('--reference', default=None, metavar='RATES[:LABEL]', help="another model's rates to mark on every "
                    "panel, such as the model's OpenSHA solution (its solution/rates.csv), or rate vectors (.npz)")
    ap.add_argument('--config', default=None, help="the run's workflow_config.json, whose settings (SLIP_SECTIONS to "
                    "FLOOR) replace the label of a single result set (default: the one next to --problem, if any; "
                    "'none': the label)")
    ap.add_argument('--ruptures', default=None, help='the OpenSHA rupture set (or a solution folder holding one), for slip '
                    'rates in mm/yr (default: in units of each section\'s uncertainty)')
    a = ap.parse_args()
    sets = [parse_set(s) for s in a.sets]
    out = a.out or os.path.join(os.path.dirname(sets[0][0]) or '.', 'fig_slip_rates.png')

    P = load_problem(a.problem)
    if a.slip_block not in P['block_names']:
        raise SystemExit(f'no row block named {a.slip_block!r}; the blocks are {list(P["block_names"])}')
    rows = np.flatnonzero(P['block'] == P['block_names'].index(a.slip_block))
    orig = np.asarray(P['keep_rows'])[rows]
    scale = P['row_scale'][rows]                         # 1/sqrt(k) where cleaning counted a section's measurement once
    u = np.ones(len(rows))                               # each row's display unit: its sd in mm/yr (--ruptures), or 1
    phys = bool(a.ruptures)
    if phys:
        from cru_common import find_rupture_set
        from plate_rate import _sections, check_problem, rupture_matrix
        rd = find_rupture_set(a.ruptures)
        if rd is None:
            raise SystemExit(f'--ruptures {a.ruptures}: no OpenSHA rupture set there')
        try:
            coef = check_problem(P, rupture_matrix(rd, len(_sections(rd))), a.slip_block)   # row scale / sigma (m/yr)
        except ValueError as e:
            raise SystemExit(f'--ruptures: {e}')
        u = scale / coef * 1e3
    sc = scale / u                                       # a row's value / sc: in mm/yr, or in units of its sd
    dw = P['d'][rows]                                    # targets as the likelihood sees them
    t = dw / sc                                          # ... in mm/yr, or in units of each section's uncertainty
    Gs = P['G'].tocsr()[rows]
    keep_c = np.asarray(P['keep_cols'])
    sec = P['section'][rows]
    mode = P['slip_sections']
    if (sec < 0).any():                                  # cleaned before sections were recorded: detect copies, for display
        sec = slip_sections(Gs, dw, np.arange(len(rows)), None, 'none')[0]
    S = int(sec.max()) + 1
    ks = np.bincount(sec, minlength=S)
    weighted = mode != 'none'                            # one measurement per section (rows divided by sqrt(k))
    ts = np.bincount(sec, weights=t, minlength=S) / ks   # each section's target (its copies share it)
    us = np.bincount(sec, weights=u, minlength=S) / ks   # each section's sd in the display units (its copies share it)
    rep_row = rows[np.unique(sec, return_index=True)[1]] if (ks > 0).all() else None   # a row of each section (its copies
    sw = (lambda e: np.sqrt(split_weights(P, rep_row, e)) * e) if rep_row is not None else (lambda e: e)   # share its sds)
    chi_w = lambda yw_: float(np.sum(split_weights(P, rows, yw_ - dw) * (yw_ - dw) ** 2))   # chi2 as the likelihood counts it
    rw = lambda e: np.sqrt(split_weights(P, rows, e)) * e  # each residual in the sd of its own side (--cfm-sliprates)
    avg = np.zeros((len(rows), S))                       # the rows' predictions -> their section means
    avg[np.arange(len(rows)), sec] = 1.0 / ks[sec]
    col = neighbour_colours(ts)                          # each section's colour, unlike its neighbours'
    mem = [np.flatnonzero(sec == s) for s in range(S)]
    n_distinct = len(np.unique(np.round(ts, 9)))
    how = ('one measurement per section (each section\'s rows divided by sqrt(k) in the likelihood)' if weighted else
           'each subsection counted as its own measurement (the inputs as given)')
    unit_txt = ('slip rates in mm/yr (each row\'s sd from its coefficients and the rupture set\'s average slips)' if phys
                else 'targets and predictions in units of each section\'s measurement uncertainty')
    head = [f'slip-rate rows: {len(rows)} (original rows {orig.min()}-{orig.max()}) in {S} sections ({int((ks > 1).sum())} with 2-'
            f'{ks.max()} subsections; slip-rate sections {mode!r}: {how}); {n_distinct} distinct targets from {t.min():.3g} to '
            f'{t.max():.3g}; {unit_txt}']
    data = []
    for path, label in sets:
        z = np.load(path)
        m_all = np.asarray(z['mean'], np.float64)
        n_raw = P['raw_shape'][1] if P.get('raw_shape') else len(m_all)
        if len(m_all) != n_raw or keep_c.max() >= len(m_all):
            raise SystemExit(f'{path} has {len(m_all)} ruptures; the problem {a.problem} has {n_raw} (another problem?)')
        yw = Gs @ m_all[keep_c]                           # predictions as the likelihood sees them
        y = yw / sc                                       # ... in mm/yr, or in measurement units
        ys = np.bincount(sec, weights=y, minlength=S) / ks
        rs, r = ys - ts, y - t
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_s = np.where(ts > 0, ys / ts, np.nan)
            ratio = np.where(t > 0, y / t, np.nan)
        model = str(z['model']) if 'model' in z.files else ''          # a single model (a best fit): no spread
        Y, used = (None, []) if model else run_predictions(os.path.dirname(path) or '.', rows, P['G'].shape[0], sc, avg)
        q = np.percentile(Y, [16, 50, 84], axis=0) if Y is not None else None   # each section mean's posterior points
        K, kidx = (None, None) if model else kept_draws(z)
        Xk = None if K is None else kept_predictions(os.path.dirname(path) or '.', kidx, rows, P['G'].shape[0], sc,
                                                      avg, check=(Gs, K, keep_c))   # the kept draws' section means
        del K
        data.append(dict(path=path, label=label, y=y, ys=ys, r=r, rs=rs, ratio=ratio, ratio_s=ratio_s, q=q, used=used,
                         ndraws=0 if Y is None else len(Y), xs=Xk,
                         chi2=chi_w(yw), in1=float(np.mean(np.abs(sw(rs / us)) <= 1)), in2=float(np.mean(np.abs(sw(rs / us)) <= 2)),
                         in1r=float(np.mean(np.abs(rw(r / u)) <= 1)), in2r=float(np.mean(np.abs(rw(r / u)) <= 2)),
                         groups=list(np.argsort(-np.abs(sw(rs / us)), kind='stable')[:max(0, a.label)]), model=model))
        del Y
    mm = 'model' if all(D['model'] for D in data) else 'mean model'   # single models (a best fit): no mean
    ref = None
    if a.reference:                                      # another model's predictions, marked on every panel
        ref = reference_rates(a.reference, P)
        yw = Gs @ ref['m']
        y = yw / sc
        ys = np.bincount(sec, weights=y, minlength=S) / ks
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_s = np.where(ts > 0, ys / ts, np.nan)
        ref.update(y=y, ys=ys, chi2=chi_w(yw), in1=float(np.mean(np.abs(sw((ys - ts) / us)) <= 1)),
                   in2=float(np.mean(np.abs(sw((ys - ts) / us)) <= 2)), q=np.nanpercentile(ratio_s, [5, 50, 95]))
        head.append(f'reference: {ref["label"]} ({ref["path"]}): its chi2 over the slip-rate rows, as the likelihood '
                    f'weights them, {ref["chi2"]:.1f}; {100 * ref["in1"]:.0f}% of sections within one measurement '
                    f'uncertainty of their targets, {100 * ref["in2"]:.0f}% within two; section mean / target: median '
                    f'{ref["q"][1]:.2f} (5-95%: {ref["q"][0]:.2f} to {ref["q"][2]:.2f})'
                    + (f'; {ref["note"]}' if ref['note'] else ''))
    rc = f' {"reference":>10s}' if ref else ''            # the reference's column in the tables
    settings = run_settings(a.config, a.problem) if len(data) == 1 else None   # the run's settings replace the label
    if settings and data[0]['model']:
        settings = [('model', data[0]['model'])] + settings
    if settings:
        head.append('settings: ' + ', '.join(f'{n} {v}' for n, v in settings))

    def rows_span(s):
        o = sorted(int(orig[k]) for k in mem[s])
        if len(o) == 1:
            return str(o[0])
        if o[-1] - o[0] == len(o) - 1:
            return f'{o[0]}-{o[-1]}'
        return ', '.join(map(str, o[:3])) + (', ...' if len(o) > 3 else '')

    rows_name = lambda s: ('row ' if ks[s] == 1 else 'rows ') + rows_span(s)

    # table view
    lines = list(head)
    for D in data:
        q = np.nanpercentile(D['ratio_s'], [5, 50, 95])
        spread = (f'posterior spread of each section mean from the {D["ndraws"]} draws recorded by {", ".join(D["used"])}'
                  if D['q'] is not None else f'{D["model"]}: a single model, so no posterior spread' if D['model']
                  else 'no runs found next to the rate vectors, so no posterior spread')
        if D['xs'] is not None:
            spread += f'; the x\'s: the {len(D["xs"])} kept draws\' section means (select_draws.py)'
        lines += ['', f'{D["label"]} ({D["path"]}): the {mm}\'s chi2 over the slip-rate rows, as the likelihood weights them, '
                  f'{D["chi2"]:.1f}; {100 * D["in1"]:.0f}% of sections within one measurement uncertainty of their targets, '
                  f'{100 * D["in2"]:.0f}% within two (subsections: {100 * D["in1r"]:.0f}% and {100 * D["in2r"]:.0f}%); section mean / '
                  f'target: median {q[1]:.2f} (5-95%: {q[0]:.2f} to {q[2]:.2f}); {spread}; furthest from their targets: '
                  + '; '.join(f'{rows_name(s)} ({sw(D["rs"] / us)[s]:+.2f} sd)' for s in D['groups']),
                  f'{"section":>7s} {"rows":>14s} {"k":>3s} {"target":>9s} {"mean prediction":>16s} {"16%":>9s} {"50%":>9s} '
                  f'{"84%":>9s} {"difference":>11s} {"in sds":>7s} {"ratio":>7s}{rc}']
        for s in range(S):
            qs = ' '.join(f'{v:9.4g}' for v in D['q'][:, s]) if D['q'] is not None else ' '.join(f'{"-":>9s}' for _ in range(3))
            lines.append(f'{s:7d} {rows_span(s):>14s} {ks[s]:3d} {ts[s]:9.4g} {D["ys"][s]:16.4g} {qs} '
                         f'{D["rs"][s]:+11.4g} {sw(D["rs"] / us)[s]:+7.2f} {D["ratio_s"][s]:7.3f}' + (f' {ref["ys"][s]:10.4g}' if ref else ''))
        lines.append(f'{"row":>7s} {"section":>14s} {"":>3s} {"target":>9s} {"mean prediction":>16s} {"difference":>11s} {"ratio":>7s}{rc}')
        for k in np.argsort(orig, kind='stable'):
            lines.append(f'{orig[k]:7d} {sec[k]:14d} {"":>3s} {t[k]:9.4g} {D["y"][k]:16.4g} {D["r"][k]:+11.3f} {D["ratio"][k]:7.3f}'
                         + (f' {ref["y"][k]:10.4g}' if ref else ''))
    text = '\n'.join(lines)
    print('\n'.join(l for l in lines if not l[:7].strip().isdigit() and not l.lstrip().startswith(('row ', 'section '))))

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    rep_i = np.unique(sec, return_index=True)[1]   # each section's sds either side, in the display units
    s_lo = us / np.sqrt(np.asarray(P['split_lo2'])[rows][rep_i]) if P.get('split_lo2') is not None else us
    s_hi = us / np.sqrt(np.asarray(P['split_hi2'])[rows][rep_i]) if P.get('split_hi2') is not None else us
    nn = best_fit_rates(a.problem)                 # the best fit's predictions fix the axes, the same for every model
    fixed = nn is not None and len(nn) == Gs.shape[1]
    if fixed:
        ys_nn = np.bincount(sec, weights=(Gs @ nn) / sc, minlength=S) / ks
        lo_b, hi_b = axis_span(ts, ys_nn, log=True)
        hi_lin = 1.05 * axis_span(ts, ys_nn, log=False)[1] + (0.0 if phys else 1.0)
    else:                                          # cleaned without the best fit: the axes follow what is plotted
        ally = np.concatenate([D['ys'] for D in data] + ([ref['ys']] if ref else []))
        pos = ally[ally >= 1e-3 * t.min()]
        lo_b, hi_b = min(t.min(), pos.min() if pos.size else t.min()), max(t.max(), ally.max())
        hi_lin = 1.05 * max(t.max(), ally.max()) + (0.0 if phys else 1.0)
    lo_log, hi_log = lo_b / 1.4, hi_b * 1.4
    edge, top, top_lin = lo_log * 1.12, hi_log / 1.12, hi_lin * 0.985   # where values beyond the axes sit
    allv = [D['ys'] for D in data] + ([ref['ys']] if ref else [])
    clipped = any((v < edge).any() or (v > top).any() or (v > top_lin).any() for v in allv)
    axes_line = (f'axes: ' + ('fixed by the targets and the best fit (NNLS; its central 96% of section means)' if fixed else
                             'from the targets and the predictions shown (no best fit in the cleaning results)')
                 + f': log {lo_log:.3g} to {hi_log:.3g}, linear 0 to {hi_lin:.3g}; values beyond sit on the edges')
    text += '\n' + axes_line
    print(axes_line)
    nrow = len(data)
    kx = any(D['xs'] is not None for D in data)          # the kept draws, as x's
    bars = any(D['q'] is not None and D['xs'] is None for D in data)
    note = (('Slip rates in mm/yr, the targets the slip-rate factor times the measured rates, each with a bar of one sd '
             'either side as the likelihood has it. ' if phys else 'In units of each section\'s measurement uncertainty. ')
            + f'{S} sections over {len(rows)} subsections, each section\'s '
            'prediction the mean over its subsections; '
            + ('each section is one measurement.' if weighted else
               'each subsection is counted as its own measurement (the inputs as given).')
            + ' Colours only tell neighbouring sections apart; sections that share a target are stacked at it.'
            + (' A bar that reaches an axis continues beyond it (on the log scale, below possibly to 0).' if bars else '')
            + (' Each x is one kept draw\'s section mean, faint, so that where the draws gather shows darkest; those '
               'beyond the axes (on the log scale, 0 included) sit on their edges.' if kx else '')
            + (' The axes are fixed by the targets and the best fit, the same for every model of these inputs; values '
               'beyond them sit on the edges.' if fixed else ''))
    note = textwrap.fill(note, 180, break_on_hyphens=False)
    extra = 0.16 * note.count('\n')                      # room for a note of more than one line
    n_h = 3 + int(bars) + int(kx) + int(clipped) + int(ref is not None)   # the legend's entries (above): one row of up to four
    lc = n_h if n_h <= 4 else 3
    leg_pt = legend_height(n_h, lc) + 6                  # the legend, between the settings and the first subtitle
    tab = settings_lines(settings, int(9.7 / 0.062500)) if settings else None
    top_in, row_in, bot_in = 0.45 + leg_pt / 72, 5.6, 0.8 + extra   # figure layout, in inches
    if tab:                                              # the settings table takes a little more room than a label
        top_in += max(0.0, 2 * len(tab) * 10.5 - 14.0) / 72
    fig, axs = plt.subplots(nrow, 2, figsize=(10.6, top_in + row_in * nrow + bot_in), squeeze=False, facecolor=SURFACE)
    halo = [pe.withStroke(linewidth=2.6, foreground=SURFACE)]
    outline = [pe.Stroke(linewidth=1.4, foreground=MUTED), pe.Normal()]   # the pale colours' bars stand out from the surface
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
                lim = (lo_log, hi_log)
                xs = np.logspace(np.log10(t.min() / 1.25), np.log10(min(hi_log, 1.25 * t.max())), 400)   # where targets are
                fmt = FuncFormatter(lambda v, _: f'{v:g}')
                for axis in (ax.xaxis, ax.yaxis):
                    axis.set_major_locator(LogLocator(base=10))
                    axis.set_major_formatter(fmt)
                    axis.set_minor_formatter(NullFormatter())
            else:
                lim = (0.0, hi_lin)
                xs = np.linspace(0.0, hi_lin, 400)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_box_aspect(1)
            ax.grid(True, which='major', color=GRID, lw=0.5, zorder=0)
            ax.plot(lim, lim, color=INK, lw=1.0, zorder=3)
            lo_e, hi_e = (edge, top) if log else (-np.inf, top_lin)      # values beyond sit on the edges
            if phys:                                 # each target's sd either side (each side's own for the CFM), at the
                yb = np.clip(D['ys'], max(lo_e, lim[0]), min(hi_e, lim[1]))   # section mean's height
                ax.hlines(yb, np.clip(ts - s_lo, lim[0], lim[1]), np.clip(ts + s_hi, lim[0], lim[1]), colors=col, lw=0.7,
                          alpha=0.16 if log else 0.32, zorder=2.1)   # faint: context for the points
            else:
                ax.fill_between(xs, np.maximum(xs - 1.0, lim[0]), xs + 1.0, color=BAND, lw=0, zorder=0.5)
            if ref is not None:                      # the reference's section means, in front of the posterior's
                scatter_edges(ax, ts, ref['ys'], lo_e, hi_e, dict(s=18, marker='s', facecolors='none', edgecolors=REF,
                              linewidths=0.7, zorder=3.4), dict(s=22, facecolors='none', edgecolors=REF, linewidths=0.7,
                              zorder=3.4))
            if D['xs'] is not None:                  # each kept draw's section means, as x's, within the axes
                kb = ts > 0 if log else np.ones(S, bool)
                Xv = np.clip(D['xs'][:, kb], *lim)
                draw_xs(ax, np.broadcast_to(ts[kb], Xv.shape), Xv, np.broadcast_to(col[kb], Xv.shape), alpha=0.05)
            elif D['q'] is not None:                 # posterior 16-84% of each section mean, within the axes
                kb = ts > 0 if log else np.ones(S, bool)
                ax.vlines(ts[kb], np.clip(D['q'][0][kb], *lim), np.clip(D['q'][2][kb], *lim), colors=col[kb], lw=1.0,
                          zorder=2.2, path_effects=outline)
            ec, ew = MUTED, 0.4                      # a thin grey edge, so the pale colours show
            scatter_edges(ax, ts, D['ys'], lo_e, hi_e, dict(s=20, c=col, edgecolors=ec, linewidths=ew, zorder=2.5),
                          dict(s=24, c=col, edgecolors=ec, linewidths=ew, zorder=2.5))
            # labels stacked in the empty corner below the 1:1 line (the panel's diagonal), with leader lines
            pts = [(s, ts[s], min(max(D['ys'][s], lo_e), hi_e)) for s in D['groups']]
            frac = lambda v: ((np.log10(v) - np.log10(lim[0])) / (np.log10(lim[1]) - np.log10(lim[0])) if log
                              else (v - lim[0]) / (lim[1] - lim[0]))
            for n_, (s, xk, yk) in enumerate(sorted(pts, key=lambda p: (-round(frac(p[2]), 3), frac(p[1])))):   # leaders
                ax.annotate(rows_name(s), (xk, yk), xytext=(0.80, 0.36 - 0.065 * n_), textcoords='axes fraction',
                            fontsize=7.5, color=INK2, va='center', path_effects=halo, zorder=5,
                            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5, shrinkA=2, shrinkB=3))
            ax.set_xlabel('target slip rate (mm/yr)' if phys else 'target slip rate (in units of its measurement uncertainty)',
                          color=INK2, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{mm} prediction (mm/yr)' if phys else f'{mm} prediction (same units)', color=INK2, fontsize=9)
            ax.set_title('log scale' if log else 'linear scale', loc='left', color=INK, fontsize=10, pad=6)
        sub = (f'{mm}: χ² {D["chi2"]:.0f} over the {len(rows)} slip-rate rows{" (as weighted)" if weighted else ""}; '
               f'{100 * D["in1"]:.0f}% of the {S} sections within one uncertainty of their targets, {100 * D["in2"]:.0f}% within two')
        above = 40 + (leg_pt if i == 0 else 0)          # the first row's settings sit above the legend
        if tab:
            draw_settings(axs[i][0], tab, above)
        else:
            axs[i][0].annotate(D['label'], (0, 1), xycoords='axes fraction', xytext=(0, above), textcoords='offset points',
                               va='bottom', color=INK, fontsize=11.5, fontweight='bold')
        axs[i][0].annotate(sub, (0, 1), xycoords='axes fraction', xytext=(0, 25), textcoords='offset points',
                           va='bottom', color=INK2, fontsize=9)
    few = CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[8]    # the legend shows a few of the colours
    handles = [(tuple(Line2D([], [], ls='none', marker='o', ms=4.8, mfc=c, mec=MUTED, mew=0.4) for c in few), 'section mean')]
    if bars:
        handles.append((tuple(Line2D([], [], color=c, lw=1.0, path_effects=outline) for c in few),
                        'posterior 16–84% of the section mean'))
    if kx:
        n_k = max(len(D['xs']) for D in data if D['xs'] is not None)
        handles.append((tuple(Line2D([], [], ls='none', marker='x', ms=4, mew=0.8, color=darker(c)) for c in few),
                        f'each of the {n_k:,} kept draws\' section mean'))
    handles += [(Line2D([], [], color=INK, lw=1.0), 'prediction = target'),
                (tuple(Line2D([], [], color=c, lw=0.9, alpha=0.5) for c in few), 'the target ± one sd (at the section mean)') if phys else
                (Patch(facecolor=BAND, edgecolor='none'), 'within one uncertainty of the target')]
    if ref is not None:
        handles.append((Line2D([], [], ls='none', marker='s', ms=4.5, mfc='none', mec=REF, mew=0.7),
                        f'{ref["label"]}: section mean (χ² {ref["chi2"]:.0f})'))
    if clipped:
        handles.append((Line2D([], [], ls='none', marker='v', ms=5, mfc=INK2, mec=INK2), 'beyond the axes (on their edges)'))
    h_in = fig.get_size_inches()[1]
    fig.suptitle(f'Slip-rate targets against the {mm}\'s predictions', x=0.035, ha='left', color=INK, fontsize=13,
                 fontweight='bold', y=1 - 0.2 / h_in, va='top')
    legend_above(fig, axs[0][0], handles, lc, 40, labelcolor=INK2)   # above the first subtitle, below the settings
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

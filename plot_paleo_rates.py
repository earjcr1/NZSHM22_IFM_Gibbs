#!/usr/bin/env python3
"""Paleo rates: each paleo row's target against the model's prediction, on log-log axes.

    python plot_paleo_rates.py VECTORS[:LABEL] [VECTORS[:LABEL] ...] [--problem cru_clean.npz] [--out FIGURE]
                               [--paleo-block NAME] [--slip-block S] [--bins-block MFD] [--label 4]
                               [--reference RATES[:LABEL]]

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

Units: the input rows are divided by their uncertainties, which the inputs do not give, so targets and predictions are
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
4). Writes FIGURE (default: fig_paleo_rates.png next to the first rate vectors) and, as its table view, the same name
with .txt: the mean model's chi-squared over the paleo rows, the shares within one and two uncertainties, and per row
the target, the mean prediction, the posterior 16%, 50% and 84% points, the difference and the ratio.

Reference: --reference RATES[:LABEL] marks another model's predictions on every panel as aqua squares, such as the
model's own OpenSHA solution, its solution/rates.csv (the notebook passes it when DATA_DIR holds the solution folder);
LABEL defaults to the solution folder's name (shortened when long). The table view adds its chi-squared and shares, and a column of its
predictions. Its rates on ruptures that cleaning removed are left out of its predictions, and the table says how much.
"""
import argparse
import glob
import os
import re
import textwrap

import numpy as np

from cru_common import axis_span, best_fit_rates, load_problem, reference_rates, run_settings, settings_lines

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BAND, BAND2 = '#ebeae5', '#f2f1ec'               # within one and within two uncertainties of the target
BLUE = '#2a78d6'                                 # categorical slot 1 of the reference palette
AQUA = '#1baf7a'                                 # categorical slot 3 (slots 1-3 validate all-pairs): the reference


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
    """Points within lo-hi at their values; those beyond, on the axis's edge as triangles (down below, up above)."""
    low, high = v < lo, v > hi
    mid = ~(low | high)
    ax.scatter(x[mid], v[mid], **kw)
    for sel, at, mk in ((low, lo, 'v'), (high, hi, '^')):
        if sel.any():
            ax.scatter(x[sel], np.full(int(sel.sum()), at), marker=mk, clip_on=False, linewidths=0, **kw_edge)


def run_predictions(folder, rows, n_rows, scale):
    """The predictions recorded for every draw by the runs next to the rate vectors (main_rep<k>.npz), for the given
    rows, pooled over runs and chains, in units of each row's uncertainty; (None, []) if there are none."""
    Y, used = [], []
    for f in sorted(glob.glob(os.path.join(folder, 'main_rep*.npz'))):
        if not re.fullmatch(r'main_rep\d+\.npz', os.path.basename(f)):
            continue                                     # running summaries and the like
        with np.load(f) as z:
            if 'y' not in z.files:
                continue
            y = z['y']
        if y.ndim != 3 or y.shape[2] != n_rows:
            continue                                     # another problem's run
        Y.append(np.asarray(y[:, :, rows], np.float64).reshape(-1, len(rows)) / scale)
        used.append(os.path.basename(f)[:-4])
    return (np.concatenate(Y), used) if Y else (None, [])


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
    dw = P['d'][rows]                                    # targets as the likelihood sees them
    t = dw / scale                                       # ... in units of each row's uncertainty
    Gp = Gr[rows]
    keep_c = np.asarray(P['keep_cols'])
    if (t <= 0).any():
        raise SystemExit(f'{int((t <= 0).sum())} paleo targets are not positive; a log-log plot cannot show them')
    head = [f'paleo rows: {len(rows)} ({which}; original rows {orig.min()}-{orig.max()}); targets from {t.min():.3g} to '
            f'{t.max():.3g}; targets and predictions in units of each row\'s uncertainty (a target is one over its relative '
            f'uncertainty)']
    data = []
    for path, label in sets:
        z = np.load(path)
        m_all = np.asarray(z['mean'], np.float64)
        n_raw = P['raw_shape'][1] if P.get('raw_shape') else len(m_all)
        if len(m_all) != n_raw or keep_c.max() >= len(m_all):
            raise SystemExit(f'{path} has {len(m_all)} ruptures; the problem {a.problem} has {n_raw} (another problem?)')
        yw = Gp @ m_all[keep_c]                          # predictions as the likelihood sees them
        y = yw / scale
        r = y - t
        model = str(z['model']) if 'model' in z.files else ''          # a single model (a best fit): no spread
        Y, used = (None, []) if model else run_predictions(os.path.dirname(path) or '.', rows, Gr.shape[0], scale)
        q = np.percentile(Y, [16, 50, 84], axis=0) if Y is not None else None
        data.append(dict(path=path, label=label, y=y, r=r, ratio=y / t, q=q, used=used, ndraws=0 if Y is None else len(Y),
                         chi2=float(np.sum((yw - dw) ** 2)), in1=float(np.mean(np.abs(r) <= 1)),
                         in2=float(np.mean(np.abs(r) <= 2)), groups=list(np.argsort(-np.abs(r), kind='stable')[:max(0, a.label)]),
                         model=model))
    mm = 'model' if all(D['model'] for D in data) else 'mean model'
    ref = None
    if a.reference:                                      # another model's predictions, marked on every panel
        ref = reference_rates(a.reference, P)
        yw = Gp @ ref['m']
        r = yw / scale - t
        ref.update(y=yw / scale, chi2=float(np.sum((yw - dw) ** 2)), in1=float(np.mean(np.abs(r) <= 1)),
                   in2=float(np.mean(np.abs(r) <= 2)), q=np.percentile(yw / scale / t, [5, 50, 95]))
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
        lines += ['', f'{D["label"]} ({D["path"]}): the {mm}\'s chi2 over the paleo rows {D["chi2"]:.1f}; '
                  f'{100 * D["in1"]:.0f}% of rows within one uncertainty of their targets, {100 * D["in2"]:.0f}% within two; '
                  f'prediction / target: median {rq[1]:.2f} (5-95%: {rq[0]:.2f} to {rq[2]:.2f}); {spread}; furthest from their '
                  'targets: ' + '; '.join(f'row {orig[k]} ({D["r"][k]:+.2f})' for k in D['groups']),
                  f'{"row":>7s} {"target":>9s} {"prediction" if D["model"] else "mean prediction":>16s} {"16%":>9s} {"50%":>9s} {"84%":>9s} '
                  f'{"difference":>11s} {"ratio":>7s}{rc}']
        for k in np.argsort(orig, kind='stable'):
            qs = ' '.join(f'{v:9.4g}' for v in D['q'][:, k]) if D['q'] is not None else ' '.join(f'{"-":>9s}' for _ in range(3))
            lines.append(f'{orig[k]:7d} {t[k]:9.4g} {D["y"][k]:16.4g} {qs} {D["r"][k]:+11.3f} {D["ratio"][k]:7.3f}'
                         + (f' {ref["y"][k]:10.4g}' if ref else ''))
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

    nn = best_fit_rates(a.problem)                   # the best fit's predictions fix the axes, the same for every model
    fixed = nn is not None and len(nn) == Gp.shape[1]
    if fixed:
        lo_b, hi_b = axis_span(t, (Gp @ nn) / scale, log=True)
    else:                                            # cleaned without the best fit: the axes follow what is plotted
        vals = np.concatenate([np.r_[D['y'], D['q'][0] if D['q'] is not None else [], D['q'][2] if D['q'] is not None else []]
                               for D in data])
        if ref is not None:
            vals = np.r_[vals, ref['y']]
        pos = vals[vals >= t.min() / 100]
        lo_b, hi_b = min(t.min(), pos.min() if pos.size else t.min()), max(t.max(), vals.max())
    lo, hi = lo_b / 1.4, hi_b * 1.4
    edge, top = lo * 1.12, hi / 1.12                 # where predictions beyond the axes sit
    allv = [D['y'] for D in data] + ([ref['y']] if ref else [])
    clipped = any(((v < edge) | (v > top)).any() for v in allv)
    axes_line = (f'axes: ' + ('fixed by the targets and the best fit (NNLS; the central 96% of its predictions)' if fixed
                             else 'from the targets and the predictions shown (no best fit in the cleaning results)')
                 + f': {lo:.3g} to {hi:.3g} on both; predictions beyond sit on the edges')
    text += '\n' + axes_line
    print(axes_line)
    ncol = len(data)
    note = ('In units of each row\'s uncertainty: on log axes that moves a point only along the 1:1 line, so its distance '
            'from the line is the ratio of prediction to target. A target is one over its relative uncertainty, so the '
            'best-measured sites lie furthest right. A bar that reaches an axis continues beyond it (below, possibly to 0).'
            + (' The axes are fixed by the targets and the best fit, the same for every model of these inputs; '
               'predictions beyond them sit on the edges.' if fixed else ''))
    top_in, sub_in, panel_in, side_in = 0.45, 0.85, 5.0, 0.75     # figure layout, in inches
    fig_w = side_in + len(data) * (panel_in + 0.55) + 0.25
    tab = settings_lines(settings, int((fig_w - side_in - 0.1) / 0.062500)) if settings else None
    if tab:                                              # the settings table takes a little more room than a label
        top_in += max(0.0, 2 * len(tab) * 10.5 - 14.0) / 72
    note = textwrap.fill(note, int((fig_w - 0.5) * 17), break_on_hyphens=False)
    n_handles = 4 + any(D['q'] is not None for D in data) + int(clipped) + int(ref is not None)   # the legend's entries
    lc = ((2 if ref is not None else 3) if ncol == 1 else n_handles)   # legend columns: fewer under one panel
    leg_rows = -(-n_handles // lc)
    leg_y = 0.08 + (note.count('\n') + 1) * 7.5 * 1.4 / 72 + 0.12   # the legend sits just above the note
    bot_in = leg_y + 0.24 * leg_rows + 0.68                          # and the axis labels above the legend
    fig, axs = plt.subplots(1, ncol, figsize=(fig_w, top_in + sub_in + panel_in + bot_in), squeeze=False, facecolor=SURFACE)
    halo = [pe.withStroke(linewidth=2.6, foreground=SURFACE)]
    fmt = FuncFormatter(lambda v, _: f'{v:g}')
    xs = np.logspace(np.log10(t.min() / 1.25), np.log10(min(hi, 1.25 * t.max())), 400)   # where the targets are
    for j, D in enumerate(data):
        ax = axs[0][j]
        ax.set_facecolor(SURFACE)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        for s_ in ('left', 'bottom'):
            ax.spines[s_].set_color(AXIS)
            ax.spines[s_].set_linewidth(0.5)
        ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=8.5, which='both', length=3, width=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(fmt)
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        ax.grid(True, which='major', color=GRID, lw=0.5, zorder=0)
        ax.fill_between(xs, np.maximum(xs - 2.0, lo), xs + 2.0, color=BAND2, lw=0, zorder=0.4)
        ax.fill_between(xs, np.maximum(xs - 1.0, lo), xs + 1.0, color=BAND, lw=0, zorder=0.5)
        ax.plot((lo, hi), (lo, hi), color=INK, lw=1.0, zorder=2)
        if D['q'] is not None:                       # posterior 16-84% of each prediction, within the axes
            ax.vlines(t, np.clip(D['q'][0], lo, hi), np.clip(D['q'][2], lo, hi), color=BLUE, alpha=0.45, lw=1.6, zorder=2.2)
        scatter_edges(ax, t, D['y'], edge, top, dict(s=36, c=BLUE, edgecolors=SURFACE, linewidths=0.8, zorder=3),
                      dict(s=32, c=BLUE, zorder=3))
        if ref is not None:                          # the reference's predictions
            scatter_edges(ax, t, ref['y'], edge, top, dict(s=30, marker='s', c=AQUA, edgecolors=SURFACE, linewidths=0.6,
                          zorder=3.2), dict(s=30, c=AQUA, zorder=3.2))
        # labels in a column in the empty corner on their points' side of the 1:1 line (lower right for points predicted
        # below their targets, upper left for those above), each near its point's height, with leader lines
        frac = lambda v: (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        step = 0.045                                 # label spacing, in axes fractions
        for side in (-1, 1):
            ks = [k for k in D['groups'] if (D['r'][k] < 0) == (side < 0)]
            if not ks:
                continue
            ks.sort(key=lambda k_: min(max(D['y'][k_], edge), top) * side)        # from the corner's edge: top down, or bottom up
            for i in range(len(ks) - 1):             # near-level neighbours: the one nearer the column takes the first slot
                fa, fb = (frac(min(max(D['y'][k_], edge), top)) for k_ in ks[i:i + 2])
                if abs(fa - fb) < step and (t[ks[i + 1]] > t[ks[i]]) == (side < 0):
                    ks[i], ks[i + 1] = ks[i + 1], ks[i]
            prev = None
            for k in ks:
                yk = min(max(D['y'][k], edge), top)
                if side < 0:                         # column at x = 0.80, at least 0.1 below the 1:1 line there
                    ty = min(frac(yk), 0.70) if prev is None else min(frac(yk), 0.70, prev - step)
                    ty, x0 = max(ty, 0.02), 0.80
                else:                                # column at x = 0.04, at least 0.1 above the 1:1 line there
                    ty = max(frac(yk), 0.27) if prev is None else max(frac(yk), 0.27, prev + step)
                    ty, x0 = min(ty, 0.98), 0.04
                prev = ty
                ax.annotate(f'row {orig[k]}', (t[k], yk), xytext=(x0, ty), textcoords='axes fraction', fontsize=7.5,
                            color=INK2, va='center', ha='left', path_effects=halo, zorder=5,
                            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5, shrinkA=2, shrinkB=4))
        ax.set_xlabel('paleo rate target (in units of its uncertainty)', color=INK2, fontsize=9)
        if j == 0:
            ax.set_ylabel('model prediction (same units)', color=INK2, fontsize=9)
        sub = (f'{mm}: χ² {D["chi2"]:.1f} over the {len(rows)} paleo rows\n'
               f'{100 * D["in1"]:.0f}% within one uncertainty '
               f'of their targets, {100 * D["in2"]:.0f}% within two')
        if tab:
            draw_settings(ax, tab, 40)
        else:
            ax.annotate(D['label'], (0, 1), xycoords='axes fraction', xytext=(0, 40), textcoords='offset points',
                        va='bottom', color=INK, fontsize=11.5, fontweight='bold')
        ax.annotate(sub, (0, 1), xycoords='axes fraction', xytext=(0, 12), textcoords='offset points',
                    va='bottom', color=INK2, fontsize=8.5, linespacing=1.35)
    handles = [Line2D([], [], ls='none', marker='o', ms=6, mfc=BLUE, mec=SURFACE, label='prediction' if mm == 'model' else 'mean prediction')]
    if any(D['q'] is not None for D in data):
        handles.append(Line2D([], [], color=BLUE, alpha=0.45, lw=1.6, label='posterior 16–84% of the prediction'))
    if ref is not None:
        handles.append(Line2D([], [], ls='none', marker='s', ms=5.5, mfc=AQUA, mec=SURFACE,
                              label=f'{ref["label"]} (χ² {ref["chi2"]:.1f})'))
    handles += [Line2D([], [], color=INK, lw=1.0, label='prediction = target'),
                Patch(facecolor=BAND, edgecolor='none', label='within one uncertainty'),
                Patch(facecolor=BAND2, edgecolor='none', label='within two')]
    if clipped:
        handles.append(Line2D([], [], ls='none', marker='v', ms=5, mfc=BLUE, mec=BLUE, label='beyond the axes (on their edges)'))
    h_in = fig.get_size_inches()[1]
    fig.suptitle('Paleo rates: targets against the model\'s predictions', x=0.03, ha='left', color=INK, fontsize=13,
                 fontweight='bold', y=1 - 0.2 / h_in, va='top')
    order = [h for c in range(lc) for h in handles[c::lc]]          # legend columns filled so it reads row by row
    fig.legend(handles=order, loc='lower center', ncol=lc, frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, leg_y / h_in), handletextpad=0.5, columnspacing=1.8)
    fig.text(0.5, 0.08 / h_in, note, ha='center', va='bottom', color=MUTED,
             fontsize=7.5, linespacing=1.4)
    fig.subplots_adjust(left=side_in / fig_w, right=1 - 0.25 / fig_w, top=1 - (top_in + sub_in) / h_in, bottom=bot_in / h_in,
                        wspace=0.55 / panel_in)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    with open(os.path.splitext(out)[0] + '.txt', 'w') as f:
        f.write(text + '\n')
    print(f'wrote {out} and {os.path.splitext(out)[0]}.txt')


if __name__ == '__main__':
    main()

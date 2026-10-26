#!/usr/bin/env python3
"""Slip rates: each fault section's slip-rate target against the mean model's prediction, on linear and on log scales.

    python plot_slip_rates.py VECTORS[:LABEL] [VECTORS[:LABEL] ...] [--problem cru_clean.npz] [--out FIGURE]
                              [--slip-block S] [--label 3] [--reference RATES[:LABEL]]

VECTORS is a result set's rate vectors as the notebook writes them (<prior folder>/rate_vectors, with or without
.npz). The mean model is their `mean` column, each rupture's posterior mean rate; since predictions are linear in the
rates, its prediction for a row is also the posterior mean of that row's prediction. VECTORS may also be a single
model's, such as the best fit that preliminaries.py writes (nnls_vectors.npz); when every result set is one, the labels
speak of the model rather than the mean model. Several result sets give one row of panels each (to compare priors,
say). LABEL names a result set; by default it is the rate vectors' folder.

Sections: a fault section's slip rate is one measurement, copied in the inputs to each of its subsections. The cleaned
problem records each slip-rate row's section (preliminaries.py --slip-sections; for problems cleaned before that, runs
of copies are detected here, for display only). Each section's mean prediction (the mean over its subsections) is
plotted against its target, with its subsections behind it in grey.

Units: the input rows are already divided by their uncertainties, which the inputs do not give, so targets and
predictions are in units of each section's measurement uncertainty (any 1/sqrt(k) weighting applied when cleaning is
undone for display). The 1:1 line is unaffected, and a section mean within the shaded band, between target - 1 and
target + 1, is within one measurement uncertainty of its target. With one measurement per section (the default), a
single subsection may stray further: in the likelihood it carries sqrt(k) times that uncertainty. On the log scale the
band reaches the bottom wherever the target is below 1: a prediction of 0 is then within one uncertainty.

Marks: a dot per section mean (blue) and per subsection (grey); means too small for the log scale's axis, including 0,
on its bottom edge as triangles; the 1:1 line; the +-1 band; and the original rows of the --label sections furthest from
their targets (default 3). Writes FIGURE (default: fig_slip_rates.png next to the first rate vectors) and, as its table
view, the same name with .txt: the mean model's chi-squared over the slip-rate rows as the likelihood weights them, the
share of sections (and of subsections) within one and two uncertainties, and per section and per row the target, the
mean prediction, their difference and their ratio.

Reference: --reference RATES[:LABEL] marks another model's section means on every panel as aqua squares, such as the
model's own OpenSHA solution, its solution/rates.csv (the notebook passes it when DATA_DIR holds the solution folder);
LABEL defaults to the solution folder's name. The table view adds its chi-squared and shares, and a column of its
predictions. Its rates on ruptures that cleaning removed are left out of its predictions, and the table says how much.
"""
import argparse
import os
import textwrap

import numpy as np

from cru_common import load_problem, reference_rates, slip_sections

SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ebeae5'
BLUE = '#2a78d6'                                 # categorical slot 1 of the reference palette
SUBSECTION = '#a9a8a1'                           # context marks: the subsections behind each section mean
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


def main():
    ap = argparse.ArgumentParser(description='Slip-rate targets against the mean model (see the module docstring).')
    ap.add_argument('sets', nargs='+', metavar='VECTORS[:LABEL]', help='rate vectors (<prior folder>/rate_vectors)')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--out', default=None, help='figure (default: fig_slip_rates.png next to the first rate vectors)')
    ap.add_argument('--slip-block', default='S', help='the slip-rate row group (default S)')
    ap.add_argument('--label', type=int, default=3, help='label this many sections furthest from their targets (default 3)')
    ap.add_argument('--reference', default=None, metavar='RATES[:LABEL]', help="another model's rates to mark on every "
                    "panel, such as the model's OpenSHA solution (its solution/rates.csv), or rate vectors (.npz)")
    a = ap.parse_args()
    sets = [parse_set(s) for s in a.sets]
    out = a.out or os.path.join(os.path.dirname(sets[0][0]) or '.', 'fig_slip_rates.png')

    P = load_problem(a.problem)
    if a.slip_block not in P['block_names']:
        raise SystemExit(f'no row block named {a.slip_block!r}; the blocks are {list(P["block_names"])}')
    rows = np.flatnonzero(P['block'] == P['block_names'].index(a.slip_block))
    orig = np.asarray(P['keep_rows'])[rows]
    scale = P['row_scale'][rows]                         # 1/sqrt(k) where cleaning counted a section's measurement once
    dw = P['d'][rows]                                    # targets as the likelihood sees them
    t = dw / scale                                       # ... in units of each section's measurement uncertainty
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
    mem = [np.flatnonzero(sec == s) for s in range(S)]
    n_distinct = len(np.unique(np.round(ts, 9)))
    how = ('one measurement per section (each section\'s rows divided by sqrt(k) in the likelihood)' if weighted else
           'each subsection counted as its own measurement (the inputs as given)')
    head = [f'slip-rate rows: {len(rows)} (original rows {orig.min()}-{orig.max()}) in {S} sections ({int((ks > 1).sum())} with 2-'
            f'{ks.max()} subsections; slip-rate sections {mode!r}: {how}); {n_distinct} distinct targets from {t.min():.3g} to '
            f'{t.max():.3g}; targets and predictions in units of each section\'s measurement uncertainty']
    data = []
    for path, label in sets:
        z = np.load(path)
        m_all = np.asarray(z['mean'], np.float64)
        n_raw = P['raw_shape'][1] if P.get('raw_shape') else len(m_all)
        if len(m_all) != n_raw or keep_c.max() >= len(m_all):
            raise SystemExit(f'{path} has {len(m_all)} ruptures; the problem {a.problem} has {n_raw} (another problem?)')
        yw = Gs @ m_all[keep_c]                           # predictions as the likelihood sees them
        y = yw / scale                                    # ... in measurement units
        ys = np.bincount(sec, weights=y, minlength=S) / ks
        rs, r = ys - ts, y - t
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_s = np.where(ts > 0, ys / ts, np.nan)
            ratio = np.where(t > 0, y / t, np.nan)
        data.append(dict(path=path, label=label, y=y, ys=ys, r=r, rs=rs, ratio=ratio, ratio_s=ratio_s,
                         chi2=float(np.sum((yw - dw) ** 2)), in1=float(np.mean(np.abs(rs) <= 1)), in2=float(np.mean(np.abs(rs) <= 2)),
                         in1r=float(np.mean(np.abs(r) <= 1)), in2r=float(np.mean(np.abs(r) <= 2)),
                         groups=list(np.argsort(-np.abs(rs), kind='stable')[:max(0, a.label)]),
                         model=str(z['model']) if 'model' in z.files else ''))
    mm = 'model' if all(D['model'] for D in data) else 'mean model'   # single models (a best fit): no mean
    ref = None
    if a.reference:                                      # another model's predictions, marked on every panel
        ref = reference_rates(a.reference, P)
        yw = Gs @ ref['m']
        y = yw / scale
        ys = np.bincount(sec, weights=y, minlength=S) / ks
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_s = np.where(ts > 0, ys / ts, np.nan)
        ref.update(y=y, ys=ys, chi2=float(np.sum((yw - dw) ** 2)), in1=float(np.mean(np.abs(ys - ts) <= 1)),
                   in2=float(np.mean(np.abs(ys - ts) <= 2)), q=np.nanpercentile(ratio_s, [5, 50, 95]))
        head.append(f'reference: {ref["label"]} ({ref["path"]}): its chi2 over the slip-rate rows, as the likelihood '
                    f'weights them, {ref["chi2"]:.1f}; {100 * ref["in1"]:.0f}% of sections within one measurement '
                    f'uncertainty of their targets, {100 * ref["in2"]:.0f}% within two; section mean / target: median '
                    f'{ref["q"][1]:.2f} (5-95%: {ref["q"][0]:.2f} to {ref["q"][2]:.2f})'
                    + (f'; {ref["note"]}' if ref['note'] else ''))
    rc = f' {"reference":>10s}' if ref else ''            # the reference's column in the tables

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
        lines += ['', f'{D["label"]} ({D["path"]}): the {mm}\'s chi2 over the slip-rate rows, as the likelihood weights them, '
                  f'{D["chi2"]:.1f}; {100 * D["in1"]:.0f}% of sections within one measurement uncertainty of their targets, '
                  f'{100 * D["in2"]:.0f}% within two (subsections: {100 * D["in1r"]:.0f}% and {100 * D["in2r"]:.0f}%); section mean / '
                  f'target: median {q[1]:.2f} (5-95%: {q[0]:.2f} to {q[2]:.2f}); furthest from their targets: '
                  + '; '.join(f'{rows_name(s)} ({D["rs"][s]:+.2f})' for s in D['groups']),
                  f'{"section":>7s} {"rows":>14s} {"k":>3s} {"target":>9s} {"mean prediction":>16s} {"difference":>11s} {"ratio":>7s}{rc}']
        for s in range(S):
            lines.append(f'{s:7d} {rows_span(s):>14s} {ks[s]:3d} {ts[s]:9.4g} {D["ys"][s]:16.4g} '
                         f'{D["rs"][s]:+11.3f} {D["ratio_s"][s]:7.3f}' + (f' {ref["ys"][s]:10.4g}' if ref else ''))
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

    ally = np.concatenate([np.r_[D['y'], D['ys']] for D in data] + ([ref['ys']] if ref else []))
    hi_lin = 1.05 * max(t.max(), ally.max()) + 1.0
    show_min = 1e-3 * t.min()                      # the log scale's lowest prediction shown at its value
    pos = ally[ally >= show_min]
    lo_log = min(t.min(), pos.min() if pos.size else t.min()) / 1.4
    hi_log = 1.4 * max(t.max(), ally.max())
    clipped = any((D['ys'] < show_min).any() or (D['y'] < show_min).any() for D in data) or bool(ref and (ref['ys'] < show_min).any())
    nrow = len(data)
    note = (f'In units of each section\'s measurement uncertainty. {S} sections over {len(rows)} subsections; '
            + ('each section is one measurement, so a single subsection may stray further than the band.' if weighted else
               'each subsection is counted as its own measurement (the inputs as given).'))
    note = textwrap.fill(note, 180, break_on_hyphens=False)
    extra = 0.16 * note.count('\n')                      # room for a note of more than one line
    n_h = 4 + int(clipped) + int(ref is not None)           # the legend's entries (below): one row of up to four
    lc = n_h if n_h <= 4 else 3
    leg_extra = 0.22 * (-(-n_h // lc) - 1)
    top_in, row_in, bot_in = 0.45, 5.6, 1.0 + extra + leg_extra   # figure layout, in inches
    fig, axs = plt.subplots(nrow, 2, figsize=(10.6, top_in + row_in * nrow + bot_in), squeeze=False, facecolor=SURFACE)
    halo = [pe.withStroke(linewidth=2.6, foreground=SURFACE)]
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
            ax.fill_between(xs, np.maximum(xs - 1.0, lim[0]), xs + 1.0, color=BAND, lw=0, zorder=0.5)
            ax.plot(lim, lim, color=INK, lw=1.0, zorder=3)
            edge = lo_log * 1.12
            for vals, tgt, kw, kw_low in ((D['y'], t, dict(s=9, c=SUBSECTION, alpha=0.55, zorder=1.5),
                                           dict(s=14, marker='v', c=SUBSECTION, alpha=0.7, zorder=1.5)),
                                          (D['ys'], ts, dict(s=26, c=BLUE, alpha=0.85, edgecolors=SURFACE, linewidths=0.5, zorder=2.5),
                                           dict(s=30, marker='v', c=BLUE, alpha=0.9, zorder=2.5))):
                shown = (vals >= show_min) if log else np.ones(len(vals), bool)
                ax.scatter(tgt[shown], vals[shown], **kw)
                if log and (~shown).any():
                    ax.scatter(tgt[~shown], np.full(int((~shown).sum()), edge), clip_on=False, linewidths=0, **kw_low)
            if ref is not None:                      # the reference's section means
                shown = (ref['ys'] >= show_min) if log else np.ones(S, bool)
                ax.scatter(ts[shown], ref['ys'][shown], s=24, marker='s', c=AQUA, edgecolors=SURFACE, linewidths=0.5, zorder=2.6)
                if log and (~shown).any():
                    ax.scatter(ts[~shown], np.full(int((~shown).sum()), edge), s=28, marker='v', c=AQUA, clip_on=False,
                               linewidths=0, zorder=2.6)
            # labels stacked in the empty corner below the 1:1 line (the panel's diagonal), with leader lines
            pts = [(s, ts[s], D['ys'][s] if (not log or D['ys'][s] >= show_min) else edge) for s in D['groups']]
            frac = lambda v: ((np.log10(v) - np.log10(lim[0])) / (np.log10(lim[1]) - np.log10(lim[0])) if log
                              else (v - lim[0]) / (lim[1] - lim[0]))
            for n_, (s, xk, yk) in enumerate(sorted(pts, key=lambda p: (-round(frac(p[2]), 3), frac(p[1])))):   # leaders
                ax.annotate(rows_name(s), (xk, yk), xytext=(0.80, 0.36 - 0.065 * n_), textcoords='axes fraction',
                            fontsize=7.5, color=INK2, va='center', path_effects=halo, zorder=5,
                            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.5, shrinkA=2, shrinkB=3))
            ax.set_xlabel('target slip rate (in units of its measurement uncertainty)', color=INK2, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{mm} prediction (same units)', color=INK2, fontsize=9)
            ax.set_title('log scale' if log else 'linear scale', loc='left', color=INK, fontsize=10, pad=6)
        sub = (f'{mm}: χ² {D["chi2"]:.0f} over the {len(rows)} slip-rate rows{" (as weighted)" if weighted else ""}; '
               f'{100 * D["in1"]:.0f}% of the {S} sections within one uncertainty of their targets, {100 * D["in2"]:.0f}% within two')
        axs[i][0].annotate(D['label'], (0, 1), xycoords='axes fraction', xytext=(0, 40), textcoords='offset points',
                           va='bottom', color=INK, fontsize=11.5, fontweight='bold')
        axs[i][0].annotate(sub, (0, 1), xycoords='axes fraction', xytext=(0, 25), textcoords='offset points',
                           va='bottom', color=INK2, fontsize=9)
    handles = [Line2D([], [], ls='none', marker='o', ms=5.5, mfc=BLUE, mec=SURFACE, alpha=0.9, label='section mean'),
               Line2D([], [], ls='none', marker='o', ms=3.5, mfc=SUBSECTION, mec=SUBSECTION, alpha=0.7, label='subsection'),
               Line2D([], [], color=INK, lw=1.0, label='prediction = target'),
               Patch(facecolor=BAND, edgecolor='none', label='within one uncertainty of the target')]
    if ref is not None:
        handles.append(Line2D([], [], ls='none', marker='s', ms=5, mfc=AQUA, mec=SURFACE,
                              label=f'{ref["label"]}: section mean (χ² {ref["chi2"]:.0f})'))
    if clipped:
        handles.append(Line2D([], [], ls='none', marker='v', ms=5, mfc=BLUE, mec=BLUE, alpha=0.8,
                              label=f'below {show_min:.2g} (on the log axis\'s edge)'))
    h_in = fig.get_size_inches()[1]
    fig.suptitle(f'Slip-rate targets against the {mm}\'s predictions', x=0.035, ha='left', color=INK, fontsize=13,
                 fontweight='bold', y=1 - 0.2 / h_in, va='top')
    handles = [h for c in range(lc) for h in handles[c::lc]]     # legend columns filled so it reads row by row
    fig.legend(handles=handles, loc='lower center', ncol=lc, frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, (0.36 + extra) / h_in), handletextpad=0.5, columnspacing=1.6)
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

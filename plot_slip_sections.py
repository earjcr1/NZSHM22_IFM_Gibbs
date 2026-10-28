#!/usr/bin/env python3
"""Slip-rate sections one by one: chosen sections' posterior slip-rate predictions, one column each, from the most
isolated section to the most connected.

    python plot_slip_sections.py VECTORS[:LABEL] [--problem cru_clean.npz] [--out FIGURE] [--sections 30]
                                 [--pick S,S,...] [--slip-block S] [--ruptures DIR] [--reference RATES[:LABEL]]

VECTORS is a result set's rate vectors as the notebook writes them (<prior folder>/rate_vectors, with or without
.npz). A section's prediction is the mean of its subsections', as in plot_slip_rates.py; its posterior distribution
comes from the predictions the runs that made the rate vectors recorded for every draw (<prior folder>/main_rep0.npz,
main_rep1.npz, ..., as the notebook writes them; cru_common.run_predictions), and its posterior mean from the rate
vectors' mean column. A single model's rate vectors (a best fit, nnls_vectors.npz) have no distribution: only their
values are drawn.

Connectivity: how many other sections share at least one rupture with the section, from the cleaned problem's
slip-rate rows (a rupture crosses a subsection where its column has a coefficient in that row; for the ruptures that
cleaning kept, this is the rupture set's indices.csv), with the number of ruptures crossing the section and the share
of them that cross no other section. --sections N (default 30) picks N sections at evenly spaced ranks of the first,
from the most isolated to the most connected (ties ranked by the number of ruptures, then by the sections' order), so
that they span its range; --pick names the sections instead (their numbers in plot_slip_rates.py's section table),
drawn in the order given.

Units: each section's measurement uncertainty, as in plot_slip_rates.py, on a log scale.

Marks, per column: the target (a black line) and the band within one uncertainty of it (grey); the section's posterior
16-84% range (a thick bar), 2.5-97.5% range (a thin line), median (a pale tick across the bar) and mean (a dot), in a
colour unlike its neighbours' (cru_common.neighbour_colours: the colours only tell neighbouring columns apart); and
the reference model's value (a dark grey square, beside it). Values below the axis, including 0, sit on its bottom
edge as triangles, and a range that reaches the bottom continues to 0. Below, the share of draws in which no rupture
crossing the section is on (its prediction exactly 0), and the connectivity.

Names: with --ruptures (an OpenSHA rupture set's folder holding fault_sections.geojson, as for make_solution.py; slip-
rate row r is the rupture set's section r), each section is named after its subsections' parent fault sections;
otherwise after its original rows.

Reference: --reference RATES[:LABEL] marks another model's section means, such as the model's own OpenSHA solution,
its solution/rates.csv; LABEL defaults to the solution folder's name (shortened when long). Its rates on ruptures that
cleaning removed are left out of its predictions, and the table says how much.

Writes FIGURE (default: fig_slip_sections.png next to the rate vectors) and, as its table view, the same name with
.txt: per column, the section, its rows and number of subsections, its connectivity (other sections, ruptures, share
confined to it), the target, the posterior mean, the 2.5%, 16%, 50%, 84% and 97.5% points, the share of draws at 0, the
reference's value, and the section's name.
"""
import argparse
import json
import os
import textwrap

import numpy as np
import scipy.sparse as sp

from cru_common import (CATEGORICAL, load_problem, neighbour_colours, reference_rates, run_predictions, run_settings,
                        settings_lines, slip_sections)

SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ebeae5'
REF = INK2                                       # the reference model's marks: dark grey, apart from every series colour
BARS = '#a9a8a1'                                 # the lower panels' bars: context, not a series
QS = (2.5, 16.0, 50.0, 84.0, 97.5)               # the posterior points drawn and tabulated


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
            ax.scatter(x[sel], np.full(int(sel.sum()), at), marker=mk, clip_on=False, linewidths=0, **pick(kw_edge, sel))


def connectivity(G, rows, sec, S):
    """Per section: how many other sections share at least one rupture with it, how many ruptures cross it, and the
    share of those that cross no other section."""
    agg = sp.csr_matrix((np.ones(len(rows)), (sec, np.arange(len(rows)))), shape=(S, len(rows)))
    B = ((agg @ (G[rows] != 0).astype(np.float64)) > 0).astype(np.float64).tocsr()   # section x rupture: crosses it
    C = (B @ B.T).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    n_r = np.asarray(B.sum(1)).ravel()
    alone = B @ (np.asarray(B.sum(0)).ravel() == 1).astype(np.float64)
    return np.diff(C.indptr), n_r, alone / np.maximum(n_r, 1)


def section_names(ruptures, orig, sec, S):
    """Each section's name, from its subsections' parent fault sections in the rupture set's fault_sections.geojson
    (slip-rate row r is the rupture set's section r), or None if the file does not describe these rows."""
    fp = os.path.join(ruptures, 'fault_sections.geojson')
    if not os.path.exists(fp):
        raise SystemExit(f'--ruptures {ruptures}: no fault_sections.geojson there')
    with open(fp) as f:
        props = {int(ft['properties']['FaultID']): ft['properties'] for ft in json.load(f)['features']}
    if not all(int(o) in props for o in orig):
        return None
    names = []
    for s in range(S):
        par = []
        for o in sorted(int(v) for v in orig[sec == s]):
            p = props[o].get('ParentName') or props[o].get('FaultName', f'section {o}')
            if p not in par:
                par.append(p)
        names.append(' + '.join(par) if len(par) <= 2 else f'{par[0]} + {len(par) - 1} more')
    return names


def main():
    ap = argparse.ArgumentParser(description='Chosen slip-rate sections\' posterior predictions, one column each '
                                             '(see the module docstring).')
    ap.add_argument('vectors', metavar='VECTORS[:LABEL]', help='rate vectors (<prior folder>/rate_vectors)')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--out', default=None, help='figure (default: fig_slip_sections.png next to the rate vectors)')
    ap.add_argument('--sections', type=int, default=30, help='how many sections, at evenly spaced ranks of connectivity '
                    '(default 30)')
    ap.add_argument('--pick', default=None, help="the sections to draw instead, comma-separated (their numbers in "
                    "plot_slip_rates.py's section table), in that order")
    ap.add_argument('--slip-block', default='S', help='the slip-rate row group (default S)')
    ap.add_argument('--ruptures', default=None, help="an OpenSHA rupture set's folder (with fault_sections.geojson), "
                    "for the sections' names")
    ap.add_argument('--reference', default=None, metavar='RATES[:LABEL]', help="another model's rates to mark, such as "
                    "the model's OpenSHA solution (its solution/rates.csv), or rate vectors (.npz)")
    ap.add_argument('--config', default=None, help="the run's workflow_config.json, whose settings (SLIP_SECTIONS to "
                    "FLOOR) replace the result set's label (default: the one next to --problem, if any; 'none': the label)")
    a = ap.parse_args()
    path, label = parse_set(a.vectors)
    out = a.out or os.path.join(os.path.dirname(path) or '.', 'fig_slip_sections.png')

    P = load_problem(a.problem)
    if a.slip_block not in P['block_names']:
        raise SystemExit(f'no row block named {a.slip_block!r}; the blocks are {list(P["block_names"])}')
    rows = np.flatnonzero(P['block'] == P['block_names'].index(a.slip_block))
    orig = np.asarray(P['keep_rows'])[rows]
    scale = P['row_scale'][rows]                         # 1/sqrt(k) where cleaning counted a section's measurement once
    t = P['d'][rows] / scale                             # targets in units of each section's measurement uncertainty
    G = P['G'].tocsr()
    Gs = G[rows]
    keep_c = np.asarray(P['keep_cols'])
    sec = np.asarray(P['section'])[rows]
    if (sec < 0).any():                                  # cleaned before sections were recorded: detect copies, for display
        sec = slip_sections(Gs, P['d'][rows], np.arange(len(rows)), None, 'none')[0]
    S = int(sec.max()) + 1
    ks = np.bincount(sec, minlength=S)
    ts = np.bincount(sec, weights=t, minlength=S) / ks
    deg, n_r, alone = connectivity(G, rows, sec, S)
    if a.pick:
        pick = np.array([int(v) for v in a.pick.split(',') if v.strip()])
        if pick.size == 0 or pick.min() < 0 or pick.max() >= S:
            raise SystemExit(f'--pick: give section numbers from 0 to {S - 1}')
        how = f'{len(pick)} sections picked by --pick'
    else:
        order = np.lexsort((np.arange(S), n_r, deg))     # by connectivity, then ruptures, then order
        pick = order[np.unique(np.round(np.linspace(0, S - 1, min(max(a.sections, 1), S))).astype(int))]
        how = (f'{len(pick)} of the {S} sections at evenly spaced ranks of the number of other sections sharing a rupture '
               f'with them ({deg.min()} to {deg.max()}, median {np.median(deg):.0f}), from the most isolated to the most '
               'connected')
    names = section_names(a.ruptures, orig, sec, S) if a.ruptures else None
    if names is None:
        names = []
        for s in range(S):
            o = sorted(int(v) for v in orig[sec == s])
            names.append(f'row {o[0]}' if len(o) == 1 else f'rows {o[0]}-{o[-1]}' if o[-1] - o[0] == len(o) - 1
                         else f'rows {o[0]}, ...')
    span = lambda s: (lambda o: str(o[0]) if len(o) == 1 else f'{o[0]}-{o[-1]}' if o[-1] - o[0] == len(o) - 1
                      else f'{o[0]}, ...')(sorted(int(v) for v in orig[sec == s]))

    z = np.load(path)
    m_all = np.asarray(z['mean'], np.float64)
    n_raw = P['raw_shape'][1] if P.get('raw_shape') else len(m_all)
    if len(m_all) != n_raw or keep_c.max() >= len(m_all):
        raise SystemExit(f'{path} has {len(m_all)} ruptures; the problem {a.problem} has {n_raw} (another problem?)')
    avg = np.zeros((len(rows), len(pick)))               # the rows' predictions -> the chosen sections' means
    for j, s in enumerate(pick):
        mem = np.flatnonzero(sec == s)
        avg[mem, j] = 1.0 / len(mem)
    ym = ((Gs @ m_all[keep_c]) / scale) @ avg            # posterior means (or a single model's values)
    model = str(z['model']) if 'model' in z.files else ''          # a single model (a best fit): no distribution
    Y, used = (None, []) if model else run_predictions(os.path.dirname(path) or '.', rows, G.shape[0], scale, avg)
    q = np.percentile(Y, QS, axis=0) if Y is not None else None
    p0 = np.mean(Y <= 0, axis=0) if Y is not None else None
    ndraws = 0 if Y is None else len(Y)
    del Y
    ref = None
    if a.reference:                                      # another model's section means, beside the posterior's
        ref = reference_rates(a.reference, P)
        ref['ys'] = ((Gs @ ref['m']) / scale) @ avg
    settings = run_settings(a.config, a.problem)
    if settings and model:
        settings = [('model', model)] + settings

    # table view
    spread = (f'posterior distributions from the {ndraws} draws recorded by {", ".join(used)}' if q is not None
              else f'{model}: a single model, so no distribution' if model
              else 'no runs found next to the rate vectors, so no distribution')
    lines = [f'slip-rate sections one by one: {how}; targets and predictions in units of each section\'s measurement '
             'uncertainty (the mean over its subsections)',
             'connectivity, from the cleaned problem\'s slip-rate rows (for the ruptures cleaning kept, the rupture set\'s '
             'indices.csv): "connected" counts the other sections sharing at least one rupture with the section, '
             '"ruptures" the ruptures crossing it, "confined" the share of those crossing no other section',
             f'{label} ({path}): {spread}']
    if ref is not None:
        lines.append(f'reference: {ref["label"]} ({ref["path"]})' + (f'; {ref["note"]}' if ref['note'] else ''))
    if settings:
        lines.append('settings: ' + ', '.join(f'{n} {v}' for n, v in settings))
    rc = f' {"reference":>10s}' if ref else ''
    lines.append(f'{"column":>6s} {"section":>7s} {"rows":>14s} {"k":>3s} {"connected":>9s} {"ruptures":>8s} '
                 f'{"confined":>8s} {"target":>9s} {"mean":>9s} ' + ' '.join(f'{f"{v:g}%":>9s}' for v in QS)
                 + f' {"at 0":>6s}{rc}  name')
    for j, s in enumerate(pick):
        qs = (' '.join(f'{v:9.4g}' for v in q[:, j]) + f' {100 * p0[j]:5.1f}%' if q is not None
              else ' '.join(f'{"-":>9s}' for _ in QS) + f' {"-":>6s}')
        lines.append(f'{j:6d} {s:7d} {span(s):>14s} {ks[s]:3d} {deg[s]:9d} {int(n_r[s]):8d} {100 * alone[s]:7.1f}% '
                     f'{ts[s]:9.4g} {ym[j]:9.4g} {qs}' + (f' {ref["ys"][j]:10.4g}' if ref else '') + f'  {names[s]}')
    text = '\n'.join(lines)
    print('\n'.join(lines[:5]))

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    N = len(pick)
    x = np.arange(N, dtype=np.float64)
    col = neighbour_colours(x)
    tsel = ts[pick]
    big = [tsel + 1.0, ym] + ([q[4]] if q is not None else []) + ([ref['ys']] if ref else [])
    hi = 1.5 * max(float(np.max(v)) for v in big)
    small = np.r_[tsel, q[1] if q is not None else []]
    small = small[small > 0]
    lo = max(small.min() / 2.0 if small.size else hi * 1e-3, hi * 1e-4)
    edge, top = lo * 1.15, hi / 1.15                     # where values beyond the axis sit
    w = 0.36                                             # half a column's width for the target and its band
    labels = [f'{names[s]} ({s})' for s in pick]
    max_len = max(len(n) for n in labels)
    note = ('In units of each section\'s measurement uncertainty, as in the slip-rate figure; a section\'s prediction is '
            'the mean over its subsections. The colours only tell neighbouring columns apart. A range that reaches the '
            'bottom of the axis continues to 0. Connectivity is from the cleaned problem\'s slip-rate rows: the rupture '
            'set\'s ruptures that cleaning kept. Each name is followed by the section\'s number in the slip-rate table.')
    fig_w = max(9.0, 0.34 * N + 2.0)
    note = textwrap.fill(note, int((fig_w - 0.5) * 17), break_on_hyphens=False)
    tab = settings_lines(settings, int((fig_w - 1.4) / 0.062500)) if settings else None
    handles = [(tuple(Line2D([], [], ls='none', marker='o', ms=5.5, mfc=c, mec=SURFACE) for c in CATEGORICAL[:3]),
                'posterior mean' if q is not None else 'value')]
    if q is not None:
        handles += [(tuple(Line2D([], [], color=c, lw=5.0, solid_capstyle='butt') for c in CATEGORICAL[:3]),
                     'posterior 16–84% (pale tick: median)'),
                    (tuple(Line2D([], [], color=c, lw=1.0) for c in CATEGORICAL[:3]), 'posterior 2.5–97.5%')]
    handles += [(Line2D([], [], color=INK, lw=1.0), 'target'),
                (Patch(facecolor=BAND, edgecolor='none'), 'within one uncertainty of the target')]
    if ref is not None:
        handles.append((Line2D([], [], ls='none', marker='s', ms=5, mfc=REF, mec=SURFACE), ref['label']))
    handles.append((Line2D([], [], ls='none', marker='v', ms=5, mfc=INK2, mec=INK2), 'below the axis (on its edge), '
                    'including 0'))
    lc = 3
    leg_rows = -(-len(handles) // lc)
    top_in = 0.45 + (max(0.0, 2 * len(tab) * 10.5 - 14.0) / 72 if tab else 0.0) + 0.55   # title, settings, subtitle
    main_in, p0_in, conn_in, gap_in = 4.4, 0.75, 0.75, 0.16
    name_in = 0.1 + 0.05 * max_len                       # the names, rotated 60 degrees
    leg_in = 0.24 * leg_rows + 0.15
    note_in = 0.12 + 0.15 * (note.count('\n') + 1)
    bot_in = name_in + leg_in + note_in
    H = top_in + main_in + p0_in + conn_in + 2 * gap_in + bot_in
    fig, axs = plt.subplots(3, 1, figsize=(fig_w, H), sharex=True, facecolor=SURFACE,
                            gridspec_kw=dict(height_ratios=[main_in, p0_in, conn_in]))
    ax, a0, ac = axs
    for a_ in axs:
        a_.set_facecolor(SURFACE)
        for s_ in ('top', 'right'):
            a_.spines[s_].set_visible(False)
        for s_ in ('left', 'bottom'):
            a_.spines[s_].set_color(AXIS)
            a_.spines[s_].set_linewidth(0.5)
        a_.tick_params(colors=AXIS, labelcolor=INK2, labelsize=8.5, which='both', length=3, width=0.5)
        a_.grid(True, which='major', axis='y', color=GRID, lw=0.5, zorder=0)
    ax.set_xlim(-0.6, N - 0.4)
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    for j in range(N):                                   # the band within one uncertainty of each target
        b0 = max(tsel[j] - 1.0, lo)
        ax.add_patch(Rectangle((j - w, b0), 2 * w, tsel[j] + 1.0 - b0, facecolor=BAND, lw=0, zorder=0.5))
    ax.hlines(tsel, x - w, x + w, colors=INK, lw=1.0, zorder=3)
    if ref is not None:                                  # the reference's values, beside the posterior's
        scatter_edges(ax, x + 0.24, ref['ys'], edge, top, dict(s=20, marker='s', c=REF, edgecolors=SURFACE,
                      linewidths=0.4, zorder=2.1), dict(s=22, c=REF, zorder=2.1))
    if q is not None:
        clip = lambda v: np.clip(v, lo, hi)
        ax.vlines(x, clip(q[0]), clip(q[4]), colors=col, lw=1.0, zorder=2.2)
        ax.vlines(x, clip(q[1]), clip(q[3]), colors=col, lw=5.0, zorder=2.3)
        on = q[2] > lo                                   # the median, as a pale tick across the bar
        ax.hlines(q[2][on], x[on] - 0.09, x[on] + 0.09, colors=SURFACE, lw=1.2, zorder=2.4)
    scatter_edges(ax, x, ym, edge, top, dict(s=30, c=col, edgecolors=SURFACE, linewidths=0.8, zorder=3.5),
                  dict(s=30, c=col, zorder=3.5))
    ax.set_ylabel('section mean prediction\n(in units of its measurement uncertainty)', color=INK2, fontsize=9)
    if q is not None:                                    # how often the section has no event at all
        a0.bar(x, 100 * p0, width=0.6, color=BARS, lw=0, zorder=2)
        for j in range(N):
            a0.annotate(f'{100 * p0[j]:.0f}', (x[j], 100 * p0[j]), xytext=(0, 1.5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=6.5, color=INK2)
    else:
        a0.annotate('no posterior draws', (0.5, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=8,
                    color=MUTED)
    a0.set_ylim(0, 125)
    a0.set_yticks([0, 50, 100])
    a0.set_ylabel('draws with\nno event (%)', color=INK2, fontsize=8.5)
    dsel = deg[pick]
    ac.bar(x, dsel, width=0.6, color=BARS, lw=0, zorder=2)
    for j in range(N):
        ac.annotate(f'{dsel[j]:d}', (x[j], dsel[j]), xytext=(0, 1.5), textcoords='offset points', ha='center',
                    va='bottom', fontsize=6.5, color=INK2)
    ac.set_ylim(0, 1.25 * max(1, dsel.max()))
    ac.set_ylabel('sections sharing\na rupture with it', color=INK2, fontsize=8.5)
    ac.set_xticks(x)
    ac.set_xticklabels(labels, rotation=60, ha='right', rotation_mode='anchor', fontsize=7.5, color=INK2)
    fig.align_ylabels(axs)
    if tab:
        draw_settings(ax, tab, 24)
    else:
        ax.annotate(label, (0, 1), xycoords='axes fraction', xytext=(0, 24), textcoords='offset points', va='bottom',
                    color=INK, fontsize=11.5, fontweight='bold')
    ax.annotate(how[0].upper() + how[1:], (0, 1), xycoords='axes fraction', xytext=(0, 8), textcoords='offset points',
                va='bottom', color=INK2, fontsize=8.5)
    fig.suptitle('Slip-rate sections one by one, from isolated to well connected' if not a.pick else
                 'Slip-rate sections one by one', x=0.035, ha='left', color=INK, fontsize=13, fontweight='bold',
                 y=1 - 0.2 / H, va='top')
    order = [h for c in range(lc) for h in handles[c::lc]]          # legend columns filled so it reads row by row
    fig.legend(handles=[h for h, _ in order], labels=[l for _, l in order], loc='lower center', ncol=lc, frameon=False,
               fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, note_in / H), handletextpad=0.5, columnspacing=1.6,
               handlelength=2.6, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})
    fig.text(0.5, 0.08 / H, note, ha='center', va='bottom', color=MUTED, fontsize=7.5, linespacing=1.4)
    fig.subplots_adjust(left=1.15 / fig_w, right=1 - 0.25 / fig_w, top=1 - top_in / H, bottom=bot_in / H,
                        hspace=gap_in / ((main_in + p0_in + conn_in) / 3))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    with open(os.path.splitext(out)[0] + '.txt', 'w') as f:
        f.write(text + '\n')
    print(f'wrote {out} and {os.path.splitext(out)[0]}.txt')


if __name__ == '__main__':
    main()

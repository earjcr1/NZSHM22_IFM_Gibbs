#!/usr/bin/env python3
"""Slip rates fault by fault: the posterior slip-rate predictions of a rupture set's named faults (or of chosen slip-rate
sections), one column each, from the most isolated to the most connected; and the ruptures crossing each, counted by
rate and summed by magnitude.

    python plot_slip_sections.py VECTORS[:LABEL] [--problem cru_clean.npz] [--out FIGURE] [--ruptures DIR]
                                 [--named FILE|none] [--sections 30] [--pick S,S,...] [--slip-block S]
                                 [--bins-block MFD] [--lowest-bin 6.8] [--bin-width 0.1] [--reference RATES[:LABEL]]

VECTORS is a result set's rate vectors as the notebook writes them (<prior folder>/rate_vectors, with or without
.npz). A column's prediction is the mean of its subsections', as plot_slip_rates.py takes a section's; its posterior
distribution comes from the predictions the runs that made the rate vectors recorded for every draw (<prior
folder>/main_rep0.npz, main_rep1.npz, ..., as the notebook writes them; cru_common.run_predictions), and its posterior
mean from the rate vectors' mean column. A single model's rate vectors (a best fit, nnls_vectors.npz) have no
distribution: only their values are drawn.

Columns: with --ruptures (an OpenSHA rupture set's folder holding fault_sections.geojson, as for make_solution.py), the
named faults of its named_faults.json (--named FILE: another such file, fault name -> its parent fault sections' IDs),
each the subsections whose parent (ParentID in fault_sections.geojson; slip-rate row r is the rupture set's section r)
is one of its own; faults with no slip-rate rows are left out, and the table says which. Without a rupture set, or
with --named none, slip-rate sections instead: --sections N (default 30) at evenly spaced ranks of connectivity, so
that they span its range, or --pick naming them (their numbers in plot_slip_rates.py's section table), drawn in the
order given; with --ruptures they are named after their subsections' parent fault sections, otherwise after their
original rows.

Connectivity: how many other sections (those holding none of the column's subsections) share at least one rupture with
it, from the cleaned problem's slip-rate rows (a rupture crosses a subsection where its column has a coefficient in that
row; for the ruptures that cleaning kept, this is the rupture set's indices.csv), with the number of ruptures crossing
it and the share of them that cross no other slip-rate row. Columns run from the most isolated to the most connected,
ties ranked by the number of ruptures, then by the file's or the sections' order.

Units: each section's measurement uncertainty, as in plot_slip_rates.py, on a log scale; a named fault spanning several
sections takes the mean over its subsections of each in its own section's units, its target and prediction alike.

Marks, per column: the target (a black line) and the band within one uncertainty of it (grey; a named fault whose
sections all lie within one uncertainty of their targets lies inside it); the posterior 16-84% range (a thick bar),
2.5-97.5% range (a thin line), median (a dark tick across the bar) and mean (a dot), in a colour unlike its neighbours'
from Tableau's Color Blind 10, outlined in grey as several are pale (cru_common.CATEGORICAL and neighbour_colours: the
colours only tell neighbouring columns apart); and the reference model's value (an open black square, beside it).
Values below the axis, including 0, sit on its bottom edge as triangles, and a range that reaches the bottom continues
to 0. Below, the share of draws in which no rupture crossing the column is on (its prediction exactly 0), and the
connectivity.

Reference: --reference RATES[:LABEL] marks another model's values, such as the model's own OpenSHA solution, its
solution/rates.csv; LABEL defaults to the solution folder's name (shortened when long). Its rates on ruptures that
cleaning removed are left out of its predictions, and the table says how much.

Writes FIGURE (default: fig_slip_sections.png next to the rate vectors) and, as its table view, the same name with
.txt: per column, its sections, rows and number of subsections, its connectivity (other sections, ruptures, share
confined to it), the target, the posterior mean, the 2.5%, 16%, 50%, 84% and 97.5% points, the share of draws at 0, the
reference's value, and the name.

Second figure: FIGURE's name with _ruptures (default fig_slip_sections_ruptures.png), one panel per column in the same
order and colours, seven across, counting the ruptures crossing it by their posterior mean rates (log10, bins a quarter
of a decade wide, counts on a log scale), and in each title how many of the largest make up 90% of their total, the
column's event rate. With --reference, the same ruptures counted by the reference's rates (a black outline), and in
each title how many of its largest rates make up 90% of its total and how many are below the axis, 0 included. A single model compares with a single draw more than with the means, which spread each
rupture's rate over the draws: they put rate on more ruptures, at lower rates, than one model does. Its table view
(.txt) gives per column the number of ruptures, the total, the largest, the 5%, 50% and 95% points of the log10 means,
the reference's total, largest and counts, and the counts per bin.

Third figure: FIGURE's name with _ruptures_by_magnitude (default fig_slip_sections_ruptures_by_magnitude.png), the same
panels with the ruptures' posterior mean rates summed by magnitude bin (the bin rows of --bins-block, counted up from
--lowest-bin in steps of --bin-width as plot_rate_magnitude.py counts them; a rupture's magnitude is known only to its
bin row): each column's posterior mean rate of events per bin, on a log scale, with the reference's sums as a black
outline, and in each title the magnitudes of its ruptures and its event rate (and the reference's). Sums of posterior
means are the posterior means of the sums, so these compare directly with a single model's. Its table view gives the
event rates and the rates per bin.
"""
import argparse
import json
import os
import re
import textwrap

import numpy as np
import scipy.sparse as sp

from cru_common import (CATEGORICAL, load_problem, magnitude_bins, neighbour_colours, reference_rates, run_predictions,
                        run_settings, settings_lines, slip_sections)

SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ebeae5'
REF = INK                                        # the reference model's marks: black, apart from every series colour
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
            ax.scatter(x[sel], np.full(int(sel.sum()), at), marker=mk, clip_on=False, **{'linewidths': 0, **pick(kw_edge, sel)})


def span(v):
    """Integers as a short range: 'a' for one, 'a-b' when consecutive, otherwise 'a, ...'."""
    o = sorted(int(x) for x in v)
    return str(o[0]) if len(o) == 1 else f'{o[0]}-{o[-1]}' if o[-1] - o[0] == len(o) - 1 else f'{o[0]}, ...'


def crossings(R, groups):
    """Group x rupture (CSR): how many of each group's rows each rupture crosses. R is the slip-rate rows' incidence
    (rows x ruptures, 1 where a rupture has a coefficient in the row); groups lists each group's row positions in it."""
    k = np.concatenate([np.full(len(g), j) for j, g in enumerate(groups)])
    agg = sp.csr_matrix((np.ones(len(k)), (k, np.concatenate(groups))), shape=(len(groups), R.shape[0]))
    X = (agg @ R).tocsr()
    X.eliminate_zeros()
    X.sort_indices()
    return X


def connectivity(R, groups, sec, S):
    """Per group of slip-rate rows (a section, or a named fault's subsections): how many other sections (holding none of
    its rows) share at least one rupture with it, how many ruptures cross it, and the share of those that cross no row
    outside it; and the group x rupture incidence (CSR: the ruptures crossing each)."""
    B = crossings(R, groups)
    every = np.asarray(R.sum(0)).ravel()                 # per rupture: how many slip-rate rows it crosses
    n_r = np.diff(B.indptr)
    alone = np.bincount(np.repeat(np.arange(len(groups)), n_r), minlength=len(groups),
                        weights=(B.data == every[B.indices]).astype(np.float64))   # crossing no row outside the group
    B.data[:] = 1.0
    shared = (crossings(R, [np.flatnonzero(sec == s) for s in range(S)]) @ B.T).toarray() > 0   # sections x groups
    for j, g in enumerate(groups):
        shared[np.unique(sec[g]), j] = False             # the group's own sections
    return shared.sum(0), n_r, alone / np.maximum(n_r, 1), B


def fault_sections(ruptures):
    """The rupture set's fault sections (subsections), FaultID -> properties, from its fault_sections.geojson."""
    fp = os.path.join(ruptures, 'fault_sections.geojson')
    if not os.path.exists(fp):
        raise SystemExit(f'--ruptures {ruptures}: no fault_sections.geojson there')
    with open(fp) as f:
        return {int(ft['properties']['FaultID']): ft['properties'] for ft in json.load(f)['features']}


def section_names(props, orig, sec, S):
    """Each section's name, from its subsections' parent fault sections (props, from fault_sections.geojson; slip-rate
    row r is the rupture set's section r), or None if the rupture set does not describe these rows."""
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


def named_faults(path, props, orig):
    """The named faults of path (a JSON of fault name -> its parent fault sections' IDs, as a rupture set's
    named_faults.json), each as the positions of its subsections' slip-rate rows: those whose ParentID (props, from
    fault_sections.geojson; slip-rate row r is the rupture set's section r) is one of its parents. Returns (names,
    groups, the names of the faults with no slip-rate rows here, left out), in the file's order."""
    with open(path) as f:
        named = json.load(f)
    under = {}                                           # parent fault section -> its subsections' row positions
    for i, o in enumerate(orig):
        p = props.get(int(o), {}).get('ParentID')
        if p is not None:
            under.setdefault(int(p), []).append(i)
    names, groups, empty = [], [], []
    for name, parents in named.items():
        g = sorted({i for pid in parents for i in under.get(int(pid), [])})
        if g:
            names.append(name)
            groups.append(np.array(g))
        else:
            empty.append(name)
    return names, groups, empty


def main():
    ap = argparse.ArgumentParser(description='Posterior slip-rate predictions of named faults, or of chosen sections, '
                                             'one column each (see the module docstring).')
    ap.add_argument('vectors', metavar='VECTORS[:LABEL]', help='rate vectors (<prior folder>/rate_vectors)')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--out', default=None, help='figure (default: fig_slip_sections.png next to the rate vectors)')
    ap.add_argument('--ruptures', default=None, help="an OpenSHA rupture set's folder (with fault_sections.geojson): its "
                    "named faults are drawn (its named_faults.json), and sections are named after their parent fault "
                    "sections")
    ap.add_argument('--named', default='auto', metavar='FILE|none', help="the named faults: a JSON of fault name -> "
                    "parent fault section IDs (default: the rupture set's named_faults.json, if it has one); 'none': "
                    "slip-rate sections instead")
    ap.add_argument('--sections', type=int, default=None, help='slip-rate sections instead of named faults: how many, at '
                    'evenly spaced ranks of connectivity (default 30)')
    ap.add_argument('--pick', default=None, help="slip-rate sections instead of named faults: which, comma-separated "
                    "(their numbers in plot_slip_rates.py's section table), in that order")
    ap.add_argument('--slip-block', default='S', help='the slip-rate row group (default S)')
    ap.add_argument('--bins-block', default='MFD', help='the magnitude-bin row group, for the figure by magnitude '
                    '(default MFD)')
    ap.add_argument('--lowest-bin', type=float, default=6.8, help='lower magnitude edge of the lowest bin row left after '
                    'cleaning (default 6.8)')
    ap.add_argument('--bin-width', type=float, default=0.1, help='magnitude width of every bin (default 0.1)')
    ap.add_argument('--reference', default=None, metavar='RATES[:LABEL]', help="another model's rates to mark, such as "
                    "the model's OpenSHA solution (its solution/rates.csv), or rate vectors (.npz)")
    ap.add_argument('--config', default=None, help="the run's workflow_config.json, whose settings (SLIP_SECTIONS to "
                    "FLOOR) replace the result set's label (default: the one next to --problem, if any; 'none': the label)")
    a = ap.parse_args()
    path, label = parse_set(a.vectors)
    out = a.out or os.path.join(os.path.dirname(path) or '.', 'fig_slip_sections.png')
    if a.named not in ('auto', 'none') and (a.pick or a.sections):
        raise SystemExit('--pick and --sections choose slip-rate sections, --named named faults: give one or the other')

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
    R = (Gs != 0).astype(np.float64).tocsr()             # slip-rate rows x ruptures: 1 where the rupture crosses the row
    props = fault_sections(a.ruptures) if a.ruptures else None
    named = None                                         # the named faults' file, when they are drawn
    if a.named not in ('auto', 'none'):
        if props is None:
            raise SystemExit(f'--named {a.named} needs --ruptures, whose fault_sections.geojson places each subsection '
                             'on its parent fault section')
        if not os.path.exists(a.named):
            raise SystemExit(f'--named {a.named}: no such file')
        named = a.named
    elif a.named == 'auto' and props is not None and not (a.pick or a.sections):
        named = os.path.join(a.ruptures, 'named_faults.json')
        named = named if os.path.exists(named) else None
    empty = []
    if named:
        names, groups, empty = named_faults(named, props, orig)
        if not groups:
            raise SystemExit(f'{named}: none of its faults has slip-rate rows here')
    else:
        groups = [np.flatnonzero(sec == s) for s in range(S)]
        names = section_names(props, orig, sec, S) if props is not None else None
        if names is None:
            names = [('row ' if len(g) == 1 else 'rows ') + span(orig[g]) for g in groups]
    deg, n_r, alone, B = connectivity(R, groups, sec, S)
    tg = np.array([t[g].mean() for g in groups])         # each column's target: the mean over its subsections
    if named:
        pick = np.lexsort((np.arange(len(groups)), n_r, deg))   # all of them, by connectivity, then ruptures, then order
        how = (f'the {len(groups)} named faults of {os.path.basename(named)}, from the most isolated to the most '
               f'connected (other sections sharing a rupture with each: {deg.min()} to {deg.max()}, median '
               f'{np.median(deg):.0f})' + (f'; {len(empty)} more have no slip-rate rows here' if empty else ''))
    elif a.pick:
        pick = np.array([int(v) for v in a.pick.split(',') if v.strip()])
        if pick.size == 0 or pick.min() < 0 or pick.max() >= S:
            raise SystemExit(f'--pick: give section numbers from 0 to {S - 1}')
        how = f'{len(pick)} sections picked by --pick'
    else:
        order = np.lexsort((np.arange(S), n_r, deg))     # by connectivity, then ruptures, then order
        pick = order[np.unique(np.round(np.linspace(0, S - 1, min(max(a.sections or 30, 1), S))).astype(int))]
        how = (f'{len(pick)} of the {S} sections at evenly spaced ranks of the number of other sections sharing a rupture '
               f'with them ({deg.min()} to {deg.max()}, median {np.median(deg):.0f}), from the most isolated to the most '
               'connected')
    unit = 'fault' if named else 'section'
    labels = [names[g] if named else f'{names[g]} ({g})' for g in pick]

    z = np.load(path)
    m_all = np.asarray(z['mean'], np.float64)
    n_raw = P['raw_shape'][1] if P.get('raw_shape') else len(m_all)
    if len(m_all) != n_raw or keep_c.max() >= len(m_all):
        raise SystemExit(f'{path} has {len(m_all)} ruptures; the problem {a.problem} has {n_raw} (another problem?)')
    avg = np.zeros((len(rows), len(pick)))               # the rows' predictions -> the chosen columns' means
    for j, g in enumerate(pick):
        avg[groups[g], j] = 1.0 / len(groups[g])
    ym = ((Gs @ m_all[keep_c]) / scale) @ avg            # posterior means (or a single model's values)
    model = str(z['model']) if 'model' in z.files else ''          # a single model (a best fit): no distribution
    Y, used = (None, []) if model else run_predictions(os.path.dirname(path) or '.', rows, G.shape[0], scale, avg)
    q = np.percentile(Y, QS, axis=0) if Y is not None else None
    p0 = np.mean(Y <= 0, axis=0) if Y is not None else None
    ndraws = 0 if Y is None else len(Y)
    del Y
    ref = None
    if a.reference:                                      # another model's column means, beside the posterior's
        ref = reference_rates(a.reference, P)
        ref['ys'] = ((Gs @ ref['m']) / scale) @ avg
    settings = run_settings(a.config, a.problem)
    if settings and model:
        settings = [('model', model)] + settings

    # table view
    spread = (f'posterior distributions from the {ndraws} draws recorded by {", ".join(used)}' if q is not None
              else f'{model}: a single model, so no distribution' if model
              else 'no runs found next to the rate vectors, so no distribution')
    lines = [f'slip rates {unit} by {unit}: {how}; targets and predictions in units of each section\'s measurement '
             'uncertainty (the mean over the subsections)',
             'connectivity, from the cleaned problem\'s slip-rate rows (for the ruptures cleaning kept, the rupture set\'s '
             f'indices.csv): "connected" counts the other sections (holding none of the {unit}\'s subsections) sharing at '
             f'least one rupture with it, "ruptures" the ruptures crossing it, "confined" the share of those crossing no '
             'other slip-rate row',
             f'{label} ({path}): {spread}']
    if empty:
        lines.append(f'named faults with no slip-rate rows here, left out: {", ".join(empty)}')
    if ref is not None:
        lines.append(f'reference: {ref["label"]} ({ref["path"]})' + (f'; {ref["note"]}' if ref['note'] else ''))
    if settings:
        lines.append('settings: ' + ', '.join(f'{n} {v}' for n, v in settings))
    rc = f' {"reference":>10s}' if ref else ''
    lines.append(f'{"column":>6s} {"sections" if named else "section":>8s} {"rows":>14s} {"k":>3s} {"connected":>9s} '
                 f'{"ruptures":>8s} {"confined":>8s} {"target":>9s} {"mean":>9s} '
                 + ' '.join(f'{f"{v:g}%":>9s}' for v in QS) + f' {"at 0":>6s}{rc}  name')
    for j, g in enumerate(pick):
        qs = (' '.join(f'{v:9.4g}' for v in q[:, j]) + f' {100 * p0[j]:5.1f}%' if q is not None
              else ' '.join(f'{"-":>9s}' for _ in QS) + f' {"-":>6s}')
        lines.append(f'{j:6d} {span(np.unique(sec[groups[g]])):>8s} {span(orig[groups[g]]):>14s} {len(groups[g]):3d} '
                     f'{deg[g]:9d} {int(n_r[g]):8d} {100 * alone[g]:7.1f}% {tg[g]:9.4g} {ym[j]:9.4g} {qs}'
                     + (f' {ref["ys"][j]:10.4g}' if ref else '') + f'  {names[g]}')
    text = '\n'.join(lines)
    print('\n'.join(lines[:5]))

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter

    N = len(pick)
    x = np.arange(N, dtype=np.float64)
    col = neighbour_colours(x)
    tsel = tg[pick]
    big = [tsel + 1.0, ym] + ([q[4]] if q is not None else []) + ([ref['ys']] if ref else [])
    hi = 1.5 * max(float(np.max(v)) for v in big)
    small = np.r_[tsel, q[1] if q is not None else []]
    small = small[small > 0]
    lo = max(small.min() / 2.0 if small.size else hi * 1e-3, hi * 1e-4)
    edge, top = lo * 1.15, hi / 1.15                     # where values beyond the axis sit
    w = 0.36                                             # half a column's width for the target and its band
    max_len = max(len(n) for n in labels)
    note = ('In units of each section\'s measurement uncertainty, as in the slip-rate figure: a fault\'s target and '
            'prediction are the means over its subsections, each in its own section\'s units, so a fault whose sections '
            'all lie within one uncertainty of their targets lies inside the band. ' if named else
            'In units of each section\'s measurement uncertainty, as in the slip-rate figure; a section\'s prediction is '
            'the mean over its subsections. ')
    note += ('The colours only tell neighbouring columns apart. A range that reaches the bottom of the axis continues to '
             '0. Connectivity is from the cleaned problem\'s slip-rate rows: the rupture set\'s ruptures that cleaning '
             f'kept; a {unit}\'s own sections are not counted.'
             + ('' if named else ' Each name is followed by the section\'s number in the slip-rate table.'))
    fig_w = max(9.0, 0.34 * N + 2.0)
    note = textwrap.fill(note, int((fig_w - 0.5) * 17), break_on_hyphens=False)
    tab = settings_lines(settings, int((fig_w - 1.4) / 0.062500)) if settings else None
    thin = [pe.Stroke(linewidth=1.3, foreground=MUTED), pe.Normal()]     # outlines: the pale colours stand out
    thick = [pe.Stroke(linewidth=4.6, foreground=MUTED), pe.Normal()]
    few = CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[8]    # the legend shows a few of the colours
    handles = [(tuple(Line2D([], [], ls='none', marker='o', ms=4.9, mfc=c, mec=MUTED, mew=0.45) for c in few),
                'posterior mean' if q is not None else 'value')]
    if q is not None:
        handles += [(tuple(Line2D([], [], color=c, lw=4.2, solid_capstyle='butt', path_effects=thick) for c in few),
                     'posterior 16–84% (dark tick: median)'),
                    (tuple(Line2D([], [], color=c, lw=0.9, path_effects=thin) for c in few), 'posterior 2.5–97.5%')]
    handles += [(Line2D([], [], color=INK, lw=1.0), 'target'),
                (Patch(facecolor=BAND, edgecolor='none'), 'within one uncertainty of the target')]
    if ref is not None:
        handles.append((Line2D([], [], ls='none', marker='s', ms=5, mfc='none', mec=REF, mew=0.8), ref['label']))
    handles.append((Line2D([], [], ls='none', marker='v', ms=5, mfc=INK2, mec=INK2), 'below the axis (on its edge), '
                    'including 0'))
    lc = 3
    leg_rows = -(-len(handles) // lc)
    top_in = 0.45 + (max(0.0, 2 * len(tab) * 10.5 - 14.0) / 72 if tab else 0.0) + 0.55   # title, settings, subtitle
    main_in, p0_in, conn_in, gap_in = 4.4, 0.8, 0.8, 0.16
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
        scatter_edges(ax, x + 0.24, ref['ys'], edge, top, dict(s=20, marker='s', facecolors='none', edgecolors=REF,
                      linewidths=0.8, zorder=3.6), dict(s=22, facecolors='none', edgecolors=REF, linewidths=0.8,
                      zorder=3.6))
    if q is not None:
        clip = lambda v: np.clip(v, lo, hi)
        ax.vlines(x, clip(q[0]), clip(q[4]), colors=col, lw=0.9, zorder=2.2, path_effects=thin)
        ax.vlines(x, clip(q[1]), clip(q[3]), colors=col, lw=4.2, zorder=2.3, path_effects=thick)
        on = q[2] > lo                                   # the median, as a dark tick across the bar
        ax.hlines(q[2][on], x[on] - 0.09, x[on] + 0.09, colors=INK, lw=1.0, zorder=2.4)
    scatter_edges(ax, x, ym, edge, top, dict(s=24, c=col, edgecolors=MUTED, linewidths=0.45, zorder=3.5),
                  dict(s=24, c=col, edgecolors=MUTED, linewidths=0.4, zorder=3.5))
    ax.set_ylabel('fault mean prediction\n(in units of its sections\' uncertainties)' if named else
                  'section mean prediction\n(in units of its measurement uncertainty)', color=INK2, fontsize=9)
    if q is not None:                                    # how often the column has no event at all
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
    ac.set_ylabel('sections\nsharing a\nrupture', color=INK2, fontsize=8.5)
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
    fig.suptitle('Named faults one by one, from isolated to well connected' if named else
                 'Slip-rate sections one by one, from isolated to well connected' if not a.pick else
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

    # the second figure: the ruptures crossing each column, counted by their posterior mean rates (and the reference's)
    m = m_all[keep_c]
    idx = [B.indices[B.indptr[g]:B.indptr[g + 1]] for g in pick]
    sets = [[m[i] for i in idx]] + ([[ref['m'][i] for i in idx]] if ref is not None else [])
    low = np.concatenate([r[r > 0] for r in sets[0]])   # the axis starts at the smallest mean
    high = np.concatenate([low] + [r[r > 0] for rs in sets[1:] for r in rs])
    h_lo, h_hi = np.floor(2 * np.log10(low.min())) / 2, np.ceil(2 * np.log10(high.max())) / 2
    bins = np.arange(h_lo, h_hi + 0.125, 0.25)           # quarter decades of log10 rate

    def tally(r):
        """A column's ruptures' rates: how many, at 0 and below the axis, their total, the largest, how many of the
        largest make up 90% of the total, the 5/50/95% points of the log10 of those above 0, and the counts per bin."""
        srt = np.sort(r)[::-1]
        tot = float(srt.sum())
        return dict(n=len(r), zero=int(np.sum(r <= 0)), below=int(np.sum(r < 10 ** h_lo)), tot=tot, top=float(srt[0]),
                    n90=int(np.searchsorted(np.cumsum(srt), 0.9 * tot) + 1) if tot > 0 else 0,
                    q=np.percentile(np.log10(r[r > 0]), [5, 50, 95]) if (r > 0).any() else np.full(3, np.nan),
                    counts=np.histogram(np.log10(r[r > 0]), bins=bins)[0])
    stats = [[tally(r) for r in rs] for rs in sets]      # the posterior means' per column, then the reference's
    out2 = os.path.splitext(out)[0] + '_ruptures.png'
    lines2 = [f'the ruptures crossing each {unit} of {os.path.basename(out)}, counted by their posterior mean rates '
              f'({label}, {path}): log10 bins 0.25 wide from {h_lo:g} to {h_hi:g}; "total" is the sum of those means, the '
              f'{unit}\'s event rate, and "90% from" how many of the largest make up 90% of it']
    if ref is not None:
        lines2.append(f'reference: {ref["label"]} ({ref["path"]}), its rates of the same ruptures: "below axis" counts '
                      f'those under 10^{h_lo:g}, 0 included')
    rc = (f' {"ref total":>10s} {"ref 90%":>8s} {"ref largest":>11s} {"below axis":>10s} {"ref at 0":>8s}'
          if ref is not None else '')
    lines2.append(f'{"column":>6s} {"sections" if named else "section":>8s} {"ruptures":>8s} {"at 0":>5s} {"total":>10s} '
                  f'{"90% from":>8s} {"largest":>9s} {"log10 rate: 5%":>14s} {"50%":>6s} {"95%":>6s}{rc}  name')
    for j, g in enumerate(pick):
        st = stats[0][j]
        rv = ''
        if ref is not None:
            sr = stats[1][j]
            rv = f' {sr["tot"]:10.4g} {sr["n90"]:8d} {sr["top"]:11.3g} {sr["below"]:10d} {sr["zero"]:8d}'
        lines2.append(f'{j:6d} {span(np.unique(sec[groups[g]])):>8s} {st["n"]:8d} {st["zero"]:5d} {st["tot"]:10.4g} '
                      f'{st["n90"]:8d} {st["top"]:9.3g} {st["q"][0]:14.2f} {st["q"][1]:6.2f} {st["q"][2]:6.2f}{rv}  '
                      f'{names[g]}')
    whats = ['ruptures per bin'] + ([f'ruptures per bin by {ref["label"]}\'s rates'] if ref is not None else [])
    for k, what in enumerate(whats):
        lines2 += ['', f'{what} (each column a bin, by its lower edge in log10 rate per year)',
                   f'{"column":>6s} ' + ' '.join(f'{e:6.2f}' for e in bins[:-1])]
        lines2 += [f'{j:6d} ' + ' '.join(f'{c:6d}' for c in stats[k][j]['counts']) for j in range(N)]
    tag = ''                                             # the reference in the panels' titles: its label, or its start
    if ref is not None:
        tag = ref['label'] if len(ref['label']) <= 14 else re.split('[_-]', ref['label'])[0][:14] or 'reference'
    ncol_h = 7
    nrow_h = -(-N // ncol_h)
    fw, ax_h = 19.0, 1.45                                # inches: the figure's width, each panel's height
    set_in = 0.15 * 2 * len(tab) if tab else 0.0         # the settings table, two lines per group
    halo = [pe.Stroke(linewidth=2.2, foreground=SURFACE), pe.Normal()]   # the outline stands out over dark bars

    def grid(note, title_lines):
        """A figure for the N panels, seven across, their titles title_lines lines high: (the figure, its panels in
        order, styled, with the unused ones hidden, its height in inches, the note wrapped, the note's height)."""
        gap_h = 0.72 if title_lines > 2 else 0.58        # between rows: the tick labels and the next row's titles
        top_h = 0.55 + set_in + 0.3 + (0.6 if title_lines > 2 else 0.45)   # title, settings, subtitle, the first titles
        note = textwrap.fill(note, int((fw - 0.5) * 17), break_on_hyphens=False)
        note_h = 0.12 + 0.15 * (note.count('\n') + 1)
        bot_h = note_h + 0.24 + 0.15 + 0.5               # the note, the legend, the lowest panels' ticks and label
        H_ = top_h + nrow_h * ax_h + (nrow_h - 1) * gap_h + bot_h
        f_, ax_ = plt.subplots(nrow_h, ncol_h, figsize=(fw, H_), sharex=True, squeeze=False, facecolor=SURFACE)
        f_.subplots_adjust(left=0.8 / fw, right=1 - 0.2 / fw, top=1 - top_h / H_, bottom=bot_h / H_,
                           hspace=gap_h / ax_h, wspace=0.18)   # room for each panel's tick labels and title
        panels = [a_ for row in ax_ for a_ in row]
        for a_ in panels[N:]:
            a_.set_visible(False)
        for a_ in panels[:N]:
            a_.set_facecolor(SURFACE)
            for s_ in ('top', 'right'):
                a_.spines[s_].set_visible(False)
            for s_ in ('left', 'bottom'):
                a_.spines[s_].set_color(AXIS)
                a_.spines[s_].set_linewidth(0.5)
            a_.tick_params(colors=AXIS, labelcolor=INK2, labelsize=7.5, which='both', length=2.5, width=0.5)
            a_.grid(True, which='major', color=GRID, lw=0.5, zorder=0)
        return f_, panels[:N], H_, note, note_h

    def finish(f_, H_, title, subtitle, legend, note, note_h, path_, table):
        """The figure's title, the run's settings, its subtitle, legend and note; written to path_, with its table view."""
        f_.text(0.035, 1 - 0.2 / H_, title, ha='left', va='top', color=INK, fontsize=13, fontweight='bold')
        y_in = H_ - 0.55
        for nm, vl in (tab or []):                       # the run's settings, as on the other figures
            f_.text(0.8 / fw, y_in / H_, nm, ha='left', va='top', fontsize=7.5, family='monospace', color=MUTED)
            f_.text(0.8 / fw, (y_in - 0.15) / H_, vl, ha='left', va='top', fontsize=7.5, family='monospace', color=INK)
            y_in -= 0.3
        f_.text(0.8 / fw, (y_in - 0.05) / H_, subtitle, ha='left', va='top', fontsize=8.5, color=INK2)
        f_.legend(handles=[h for h, _ in legend], labels=[l for _, l in legend], loc='lower center', ncol=2,
                  frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, note_h / H_), handletextpad=0.5,
                  columnspacing=2.0, handlelength=2.6, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})
        f_.text(0.5, 0.08 / H_, note, ha='center', va='bottom', color=MUTED, fontsize=7.5, linespacing=1.4)
        f_.savefig(path_, dpi=150, facecolor=SURFACE)
        with open(os.path.splitext(path_)[0] + '.txt', 'w') as f:
            f.write('\n'.join(table) + '\n')
        print(f'wrote {path_} and {os.path.splitext(path_)[0]}.txt')

    each = 'named fault' if named else 'section'
    ordered = ', from isolated to well connected' if not a.pick else ''
    note2 = ('Counts on a log scale, in bins a quarter of a decade wide. A rupture\'s posterior mean rate is its rate when '
             'on times the share of draws in which it is on. "N carry 90%" counts the largest rates that make up 90% of '
             f'their total, the {unit}\'s event rate.')
    if ref is not None:
        note2 += (f' The black outline counts the same ruptures by {ref["label"]}\'s rates. That is a single model, which '
                  'compares with a single draw more than with the means: averaging over the draws puts rate on more '
                  'ruptures, at lower rates, than one model does.')
    fig2, axh, H2, note2, note_h = grid(note2, 3 if ref is not None else 2)
    xf = FuncFormatter(lambda v, _: f'{v:g}'.replace('-', '−'))
    yf = FuncFormatter(lambda v, _: f'{v:,.0f}' if v >= 1 else '')      # counts, as plain numbers
    for k, a_ in enumerate(axh):
        st = stats[0][k]
        a_.bar(bins[:-1], st['counts'], width=0.25, align='edge', color=col[k], edgecolor=MUTED, linewidth=0.3, zorder=2)
        a_.set_yscale('log', nonpositive='clip')
        a_.set_ylim(0.7, max(3.0, max(s[k]['counts'].max() for s in stats) * 3.0))
        a_.yaxis.set_major_locator(LogLocator(base=10))
        a_.yaxis.set_major_formatter(yf)
        a_.yaxis.set_minor_formatter(NullFormatter())
        title = (f'{labels[k]}\n{st["n"]:,} {"rupture" if st["n"] == 1 else "ruptures"}, '
                 f'{st["n90"]:,} {"carries" if st["n90"] == 1 else "carry"} 90%' + (f'; {st["zero"]} at 0' if st['zero'] else ''))
        if ref is not None:                              # the reference's rates of the same ruptures
            sr = stats[1][k]
            a_.stairs(sr['counts'], bins, baseline=None, color=REF, lw=0.9, zorder=3.2, path_effects=halo)
            title += (f'\n{tag}: {sr["n90"]:,} {"carries" if sr["n90"] == 1 else "carry"} 90%, '
                      f'{sr["below"]:,} below $10^{{{h_lo:g}}}$')
        a_.set_xlim(h_lo, h_hi)
        a_.xaxis.set_major_formatter(xf)
        a_.tick_params(labelbottom=True)                 # every panel shows its axis's values
        a_.set_title(title, loc='left', fontsize=7.2, color=INK, pad=3, linespacing=1.3)
        if k % ncol_h == 0:
            a_.set_ylabel('ruptures', color=INK2, fontsize=8)
        if k + ncol_h >= N:                              # the lowest panel of its column
            a_.set_xlabel('log10 rate (per year)' if ref is not None else 'log10 posterior mean rate (per year)',
                          color=INK2, fontsize=8)
    leg2 = [(tuple(Patch(facecolor=c, edgecolor=MUTED, linewidth=0.3) for c in few), f'ruptures crossing the {unit}, by '
             'posterior mean rate')]
    if ref is not None:
        leg2.append((Patch(facecolor='none', edgecolor=REF, linewidth=0.9), f'the same ruptures by {ref["label"]}\'s rates'))
    finish(fig2, H2, f'Rupture rates of each {each}{ordered}',
           f'The ruptures crossing each {unit} of {os.path.basename(out)}, counted by their posterior mean rates'
           + (f' and by {ref["label"]}\'s' if ref is not None else '') + '; panels in that figure\'s order and colours',
           leg2, note2, note_h, out2, lines2)

    # the third figure: the same ruptures' rates summed by magnitude bin, each column's participation rate per bin
    if a.bins_block not in P['block_names']:
        print(f'no row block named {a.bins_block!r}, so no figure by magnitude')
        return
    mrows, _, _, mlo = magnitude_bins(P, a.bins_block, a.lowest_bin, a.bin_width)
    mw = float(a.bin_width)
    binof = np.full(G.shape[1], -1)                      # each rupture's magnitude bin: the bin row it is in
    for j, r in enumerate(mrows):
        binof[G.indices[G.indptr[r]:G.indptr[r + 1]]] = j
    slot = np.round((mlo - mlo[0]) / mw).astype(int)     # the bins on a regular grid (a bin cleaning removed: a gap)
    edges = mlo[0] + mw * np.arange(slot[-1] + 2)

    def by_magnitude(i, r):
        """The rates r of ruptures i summed by magnitude bin, on the grid of edges (ruptures in no bin row left out)."""
        b_ = binof[i]
        return np.bincount(slot[b_[b_ >= 0]], weights=r[b_ >= 0], minlength=len(edges) - 1)

    def sci(v):
        """v as d.d x 10^n, in mathtext."""
        if v <= 0:
            return '0'
        e = int(np.floor(np.log10(v)))
        if round(v / 10 ** e, 1) >= 10:
            e += 1
        return f'${v / 10 ** e:.1f}\\times10^{{{e}}}$'

    mfd = [[by_magnitude(i, r) for i, r in zip(idx, rs)] for rs in sets]   # the posterior means', then the reference's
    spans = []                                           # each column's ruptures' magnitudes, as a range of bins
    for i in idx:
        b_ = np.unique(binof[i][binof[i] >= 0])
        spans.append(f'M{mlo[b_[0]]:.1f}–{mlo[b_[-1]] + mw:.1f}' if b_.size else 'no magnitude bin')
    unbinned = sum(int(np.sum(binof[i] < 0)) for i in idx)
    out3 = os.path.splitext(out)[0] + '_ruptures_by_magnitude.png'
    lines3 = [f'the ruptures crossing each {unit} of {os.path.basename(out)}: their posterior mean rates ({label}, {path}) '
              f'summed by magnitude bin ({len(mrows)} bin rows of {a.bins_block}, counted up from {mlo[0]:.2f} in steps of '
              f'{mw:g}), the posterior mean of each {unit}\'s rate of events in each bin; "events" is their total, the '
              f'{unit}\'s event rate']
    if ref is not None:
        lines3.append(f'reference: {ref["label"]} ({ref["path"]}), its rates of the same ruptures summed the same way')
    if unbinned:
        lines3.append(f'{unbinned} of the ruptures counted are in no bin row, so in no bin')
    rc = f' {"ref events":>10s}' if ref is not None else ''
    lines3.append(f'{"column":>6s} {"ruptures":>8s} {"magnitudes":>12s} {"events":>10s}{rc}  name')
    for j, g in enumerate(pick):
        lines3.append(f'{j:6d} {len(idx[j]):8d} {spans[j]:>12s} {sets[0][j].sum():10.4g}'
                      + (f' {sets[1][j].sum():10.4g}' if ref is not None else '') + f'  {names[g]}')
    whats = ['posterior mean rate per magnitude bin'] + ([f'{ref["label"]}\'s rate per magnitude bin'] if ref is not None
                                                         else [])
    for k, what in enumerate(whats):
        lines3 += ['', f'{what} (each column a bin, by its lower edge)',
                   f'{"column":>6s} ' + ' '.join(f'{e:9.2f}' for e in edges[:-1])]
        lines3 += [f'{j:6d} ' + ' '.join(f'{v:9.3g}' for v in mfd[k][j]) for j in range(N)]
    note3 = (f'Each bar is the total of the posterior mean rates of the {unit}\'s ruptures in one magnitude bin, '
             f'{mw:g} wide: the posterior mean of its rate of events of that size (a rupture\'s magnitude is known only to '
             f'its bin row; the rows are counted up from {mlo[0]:.1f}). Unlike the counts by rate, these totals compare '
             'directly with a single model\'s. Rates on a log scale; bins more than six decades below a panel\'s largest '
             'are not drawn.' + (f' The black outline is the same for {ref["label"]}\'s rates.' if ref is not None else ''))
    fig3, axm, H3, note3, note_h3 = grid(note3, 3)
    for k, a_ in enumerate(axm):
        v = mfd[0][k]
        top = max(float(s[k].max()) for s in mfd)
        a_.bar(edges[:-1], v, width=mw, align='edge', color=col[k], edgecolor=MUTED, linewidth=0.3, zorder=2)
        a_.set_yscale('log', nonpositive='clip')
        if top > 0:                                      # at most six decades, down from the panel's largest
            pos = np.concatenate([s[k][s[k] > 0] for s in mfd])
            a_.set_ylim(max(pos.min() / 2, top / 1e6), top * 3)
        a_.yaxis.set_major_locator(LogLocator(base=10))
        a_.yaxis.set_minor_formatter(NullFormatter())
        title = (f'{labels[k]}\n{len(idx[k]):,} {"rupture" if len(idx[k]) == 1 else "ruptures"}, {spans[k]}\n'
                 f'events/yr: {sci(float(sets[0][k].sum()))}')
        if ref is not None:                              # the reference's rates of the same ruptures
            a_.stairs(mfd[1][k], edges, baseline=None, color=REF, lw=0.9, zorder=3.2, path_effects=halo)
            title += f'; {tag} {sci(float(sets[1][k].sum()))}'
        a_.set_xlim(edges[0], edges[-1])
        a_.xaxis.set_major_locator(MultipleLocator(0.5))
        a_.xaxis.set_major_formatter(FuncFormatter(lambda x_, _: f'{x_:.1f}'))
        a_.tick_params(labelbottom=True)                 # every panel shows its axis's values
        a_.set_title(title, loc='left', fontsize=7.2, color=INK, pad=3, linespacing=1.3)
        if k % ncol_h == 0:
            a_.set_ylabel('rate per year', color=INK2, fontsize=8)
        if k + ncol_h >= N:                              # the lowest panel of its column
            a_.set_xlabel('magnitude', color=INK2, fontsize=8)
    leg3 = [(tuple(Patch(facecolor=c, edgecolor=MUTED, linewidth=0.3) for c in few), f'the {unit}\'s ruptures\' posterior '
             'mean rates, summed by magnitude bin')]
    if ref is not None:
        leg3.append((Patch(facecolor='none', edgecolor=REF, linewidth=0.9), f'the same for {ref["label"]}\'s rates'))
    finish(fig3, H3, f'Rupture rates of each {each} by magnitude{ordered}',
           f'The ruptures crossing each {unit} of {os.path.basename(out)}: their posterior mean rates'
           + (f' and {ref["label"]}\'s' if ref is not None else '') + ' summed by magnitude bin; panels in that figure\'s '
           'order and colours', leg3, note3, note_h3, out3, lines3)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""randomized_nnls.py -- best fits to noise-perturbed data: how far the best fit moves with the data's stated noise.

    python randomized_nnls.py --problem cru_clean.npz --fits 200 --seed 1 --workers 4 --out rnnls
    python randomized_nnls.py --selftest

Each fit is the nonnegative least-squares best fit (preliminaries.py's Lawson-Hanson NNLS, as cleaning computes it) to
the data plus a fresh draw of their stated noise, d + sigma e with e ~ N(0, I) on the cleaned rows. That is the
likelihood's own noise, as the sampler sees it: one measurement per slip-rate section, and the MFD bins at
MFD_UNCERTAINTY's level. The fits' spread is the best fit's sampling distribution, not a posterior. It leaves out
(REPORT.md, "Randomized best fits") the many models that fit as well as each one (every fit has at most one nonzero rate
per row, and its solver picks which), any prior, and the weight of the far more numerous models that fit slightly less
well; and a rate near 0 comes out exactly 0 whenever the noise pushes it below.

The fits run in --workers processes, one thread each (about 10 s per fit at 211,000 ruptures on an M2). Fit k's noise
comes from the k-th seed spawned from --seed, so the results do not depend on the number of workers. The unperturbed
best fit comes from prelim_results.npz next to --problem, or is computed here if cleaning skipped it (--no-nnls).

Writes into --out:
  rnnls_fits.npz             each fit's predictions (as the likelihood sees the rows), its chi2 against its own and the
                             observed data, total rate, iterations and time, and its nonzero rates (sparse: ptr, idx, val)
  rate_vectors.npz, .csv     per rupture of the inputs (removed ruptures 0): the fits' mean rate, the share of fits in
                             which it is nonzero, and the unperturbed best fit (make_solution.py --column mean takes them)
  fig_rnnls_slip_rates.png   each slip-rate section's target against the fits' median prediction, with their 16-84% and
                             2.5-97.5% ranges, linear and log, in units of the section's measurement uncertainty
  fig_rnnls_mfd.png          each magnitude bin's target rate against the fits' bin totals, and their ratio
  fig_rnnls_paleo_rates.png  each paleo row's target against the fits' predictions, log-log, in units of its uncertainty
  rnnls_report.txt           everything printed: the fits' chi2, total rate, bins, sparsity, and how often the targets
                             fall inside the fits' ranges
Each figure has its table view next to it (.txt): per section, bin or row, the target, the best fit, and the fits'
median, 16%, 84%, 2.5% and 97.5% points. --reference RATES[:LABEL] also marks another model's predictions on every
figure, as dark grey squares, and adds them to the tables: such as the model's own OpenSHA solution, its
solution/rates.csv (the notebook passes it when DATA_DIR holds the solution folder).
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')                 # one BLAS thread per process: the fits run in parallel processes
import argparse
import csv
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from cru_common import axis_span, load_problem, magnitude_bins, prelim_path, reference_rates, report_to, slip_sections
from preliminaries import nnls_lawson_hanson

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'   # the fit
BAND, BAND2, BLUE = '#ebeae5', '#f2f1ec', '#2a78d6'      # figures' palette (plot_slip_rates.py, plot_paleo_rates.py)
REF = INK2                                       # --reference's marks: dark grey, apart from every series colour
Q = (2.5, 16.0, 50.0, 84.0, 97.5)                 # the fits' percentiles shown and tabulated

# ------------------------------------------------------------------------------------------------------------ the fits
_W = {}


def _init(problem):
    """Each worker: the problem, read once (a path to a cleaned problem, or (G, d, sigma) arrays)."""
    if isinstance(problem, str):
        P = load_problem(problem)
        G, d, sigma = P['G'], P['d'], float(P['sigma'])
    else:
        G, d, sigma = problem
    _W.update(G=sp.csc_matrix(G), d=np.asarray(d, np.float64), sigma=float(sigma))


def _fit(job):
    """One fit: NNLS to d + sigma e, e ~ N(0, I) drawn from the job's seed."""
    k, seed = job
    G, d, sigma = _W['G'], _W['d'], _W['sigma']
    dk = d + sigma * np.random.default_rng(seed).standard_normal(len(d))
    t0 = time.time()
    x, it = nnls_lawson_hanson(G, dk, threads=1)
    nz = np.flatnonzero(x)
    y = G[:, nz] @ x[nz]
    return dict(k=k, idx=nz.astype(np.int64), val=x[nz], y=np.asarray(y, np.float64).ravel(), it=int(it),
                chi2_own=float(np.sum((y - dk) ** 2)) / sigma ** 2, chi2_obs=float(np.sum((y - d) ** 2)) / sigma ** 2,
                seconds=time.time() - t0)


def ensemble(problem, n_fits, seed, workers, verbose=True):
    """n_fits fits, in fit order, in `workers` processes (1: in this one)."""
    jobs = list(zip(range(n_fits), np.random.SeedSequence(seed).spawn(n_fits)))
    out, t0, step = [], time.time(), max(1, n_fits // 10)

    def progress(r):
        out.append(r)
        if verbose and (len(out) % step == 0 or len(out) == n_fits):
            el = time.time() - t0
            print(f'   {len(out):5d} / {n_fits} fits ({el:.0f} s, {el / len(out):.1f} s per fit with {workers} worker'
                  f'{"s" if workers > 1 else ""}; this fit {r["seconds"]:.1f} s, {r["it"]} iterations)', flush=True)
    if workers <= 1:
        _init(problem)
        for job in jobs:
            progress(_fit(job))
    else:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(problem,)) as pool:
            for r in pool.map(_fit, jobs, chunksize=1):
                progress(r)
    return out


# ------------------------------------------------------------------------------------------------------------- figures
def _style(ax, logx=False, logy=False, plain=True):
    """The figures' axes: recessive spines, ticks and grid; log scales labelled as plain numbers (plain=False keeps
    matplotlib's own log labels, for an axis spanning under a decade)."""
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_facecolor(SURFACE)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_color(AXIS)
        ax.spines[s_].set_linewidth(0.5)
    ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=8.5, which='both', length=3, width=0.5)
    ax.grid(True, which='major', color=GRID, lw=0.5, zorder=0)
    if plain:
        from matplotlib.ticker import FuncFormatter, NullFormatter
        for axis, log in ((ax.xaxis, logx), (ax.yaxis, logy)):
            if log:
                axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
                axis.set_minor_formatter(NullFormatter())


def _ranges(ax, x, q, floor=None, base=None, size=24, ceil=None):
    """The fits' 2.5-97.5% (thin) and 16-84% (thicker) ranges at each x, the median (dot) and the best fit (open diamond).
    Values below floor or above ceil, the axis's edges, sit on them (a median there as a triangle)."""
    lo2, lo1, med, hi1, hi2 = q
    f = lambda v: np.clip(v, -np.inf if floor is None else floor, np.inf if ceil is None else ceil)
    ax.vlines(x, f(lo2), f(hi2), color=BLUE, lw=0.8, alpha=0.35, zorder=2)
    ax.vlines(x, f(lo1), f(hi1), color=BLUE, lw=2.0, alpha=0.6, zorder=2.2)
    low = (med < floor) if floor is not None else np.zeros(len(med), bool)
    high = (med > ceil) if ceil is not None else np.zeros(len(med), bool)
    ax.scatter(x[~(low | high)], med[~(low | high)], s=size, c=BLUE, edgecolors=SURFACE, linewidths=0.6, zorder=3)
    for sel, at, mk in ((low, floor, 'v'), (high, ceil, '^')):
        if sel.any():
            ax.scatter(x[sel], np.full(int(sel.sum()), at), s=size + 6, marker=mk, c=BLUE, linewidths=0, zorder=3,
                       clip_on=False)
    if base is not None:
        ax.scatter(x, f(base), s=size - 6, marker='D', facecolors='none', edgecolors=INK2, linewidths=0.8, zorder=3.5)


def _reference(ax, x, v, floor=None, size=24, ceil=None):
    """A reference solution's values (--reference) as squares; values below floor or above ceil sit on them."""
    low = (v < floor) if floor is not None else np.zeros(len(v), bool)
    high = (v > ceil) if ceil is not None else np.zeros(len(v), bool)
    ax.scatter(x[~(low | high)], v[~(low | high)], s=size, marker='s', c=REF, edgecolors=SURFACE, linewidths=0.6, zorder=3.4)
    for sel, at, mk in ((low, floor, 'v'), (high, ceil, '^')):
        if sel.any():
            ax.scatter(x[sel], np.full(int(sel.sum()), at), s=size + 4, marker=mk, c=REF, linewidths=0, zorder=3.4,
                       clip_on=False)


def _ref_handles(ref, extra=''):
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls='none', marker='s', ms=5, mfc=REF, mec=SURFACE, label=ref['label'] + extra)] if ref else []


def _ref_head(ref, what):
    return ([f'reference: {ref["label"]} ({ref["path"]}): {what}' + (f'; {ref["note"]}' if ref['note'] else '')]
            if ref else [])


def _legend(fig, h_in, extra_handles=(), band_label='within one uncertainty of the target', band2_label=None):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = [Line2D([], [], ls='none', marker='o', ms=5.5, mfc=BLUE, mec=SURFACE, label='median of the fits'),
          Line2D([], [], color=BLUE, lw=2.0, alpha=0.6, label='16-84% of the fits'),
          Line2D([], [], color=BLUE, lw=0.8, alpha=0.35, label='2.5-97.5%'),
          Line2D([], [], ls='none', marker='D', ms=4.5, mfc='none', mec=INK2, label='best fit (unperturbed data)')]
    hs += list(extra_handles)
    hs.append(Patch(facecolor=BAND, edgecolor='none', label=band_label))
    if band2_label:
        hs.append(Patch(facecolor=BAND2, edgecolor='none', label=band2_label))
    fig.legend(handles=hs, loc='lower center', ncol=min(len(hs), 4), frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.42 / h_in), handletextpad=0.5, columnspacing=1.6)


def _titles(fig, title, sub, note, h_in):
    w_in = fig.get_size_inches()[0]
    fig.suptitle(title, x=0.035, ha='left', color=INK, fontsize=13, fontweight='bold', y=1 - 0.2 / h_in, va='top')
    fig.text(0.035, 1 - 0.52 / h_in, textwrap.fill(sub, int(15 * w_in), break_on_hyphens=False), ha='left', va='top',
             color=INK2, fontsize=9)
    fig.text(0.5, 0.12 / h_in, textwrap.fill(note, int(16 * w_in), break_on_hyphens=False), ha='center', va='bottom',
             color=MUTED, fontsize=8, linespacing=1.4)


def _table(path, head, cols, rows):
    """The figure's table view: header lines, then one line per section, bin or row."""
    w = [max(len(c), max((len(r[i]) for r in rows), default=0)) for i, c in enumerate(cols)]
    lines = list(head) + ['', '  '.join(c.rjust(w[i]) for i, c in enumerate(cols))]
    lines += ['  '.join(v.rjust(w[i]) for i, v in enumerate(r)) for r in rows]
    open(path, 'w').write('\n'.join(lines) + '\n')


def _pct(Y):
    return np.percentile(Y, Q, axis=0)


def fig_slip(out, P, Y, y0, slip_block, nfits, ref=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = np.flatnonzero(P['block'] == list(P['block_names']).index(slip_block))
    scale, dw = P['row_scale'][rows], P['d'][rows]
    t = dw / scale                                       # targets in units of each section's measurement uncertainty
    sec = np.asarray(P['section'])[rows]
    if (sec < 0).any():                                  # cleaned before sections were recorded: detect copies, for display
        sec = slip_sections(P['G'].tocsr()[rows], dw, np.arange(len(rows)), None, 'none')[0]
    S = int(sec.max()) + 1
    ks = np.bincount(sec, minlength=S)
    A = sp.csr_matrix((1.0 / ks[sec], (sec, np.arange(len(rows)))), shape=(S, len(rows)))   # section means
    ts = A @ t
    Ys = np.asarray((A @ (Y[:, rows] / scale).T).T)       # fits x sections
    ys0 = A @ (y0[rows] / scale)
    ysr = A @ (ref['y'][rows] / scale) if ref else None   # the reference's section means
    q = _pct(Ys)
    inside1 = np.mean((q[1] <= ts) & (ts <= q[3]))
    within = np.mean(np.abs(Ys - ts) <= 1.0, axis=0)     # share of fits within one uncertainty, per section
    orig = np.asarray(P['keep_rows'])[rows]
    names = []
    for s in range(S):
        o = sorted(int(v) for v in orig[sec == s])
        names.append(str(o[0]) if len(o) == 1 else f'{o[0]}-{o[-1]}' if o[-1] - o[0] == len(o) - 1 else f'{o[0]}, ...')
    _table(os.path.splitext(out)[0] + '.txt',
           [f'slip-rate sections: {S} over {len(rows)} subsection rows (original rows {orig.min()}-{orig.max()}); targets and '
            f'predictions in units of each section\'s measurement uncertainty (section means of the subsections); {nfits} '
            f'randomized best fits',
            f'the target lies inside the fits\' 16-84% range for {100 * inside1:.0f}% of the sections; the fits\' median '
            f'prediction / target: median {np.median(q[2] / ts):.3f} over the sections']
           + _ref_head(ref, f'section mean / target: median {np.median(ysr / ts):.3f}' if ref else ''),
           ['rows', 'target', 'best_fit', 'p2.5', 'p16', 'median', 'p84', 'p97.5', 'within_1'] + (['reference'] if ref else []),
           [[names[s], f'{ts[s]:.4g}', f'{ys0[s]:.4g}'] + [f'{q[i][s]:.4g}' for i in range(5)] + [f'{within[s]:.2f}']
            + ([f'{ysr[s]:.4g}'] if ref else []) for s in range(S)])
    top_in, row_in, bot_in = 0.95, 5.0, 1.15
    fig, axs = plt.subplots(1, 2, figsize=(10.6, top_in + row_in + bot_in), facecolor=SURFACE)
    h_in = fig.get_size_inches()[1]
    lo_b, hi_b = axis_span(ts, ys0, log=True)            # fixed by the targets and the best fit, as plot_slip_rates.py's
    hi_lin = 1.05 * axis_span(ts, ys0, log=False)[1] + 1.0
    lo_log, hi_log = lo_b / 1.4, hi_b * 1.4
    for j, ax in enumerate(axs):
        log = j == 1
        _style(ax, logx=log, logy=log)
        if log:
            lim = (lo_log, hi_log)
            xs = np.logspace(np.log10(lo_log), np.log10(hi_log), 400)
        else:
            lim = (0.0, hi_lin)
            xs = np.linspace(0.0, hi_lin, 400)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_box_aspect(1)
        ax.fill_between(xs, np.maximum(xs - 1.0, lim[0]), xs + 1.0, color=BAND, lw=0, zorder=0.5)
        ax.plot(lim, lim, color=INK, lw=1.0, zorder=1)
        keep = ts > 0 if log else np.ones(S, bool)
        fl, cl = (lim[0] * 1.12, lim[1] / 1.12) if log else (None, lim[1] * 0.985)   # values beyond sit on the edges
        _ranges(ax, ts[keep], q[:, keep], floor=fl, ceil=cl, base=ys0[keep])
        if ref:
            _reference(ax, ts[keep], ysr[keep], floor=fl, ceil=cl)
        ax.set_xlabel('target slip rate (in units of its measurement uncertainty)', color=INK2, fontsize=9)
        if j == 0:
            ax.set_ylabel('predicted slip rate (same units)', color=INK2, fontsize=9)
        ax.set_title('log scale' if log else 'linear scale', loc='left', color=INK, fontsize=10, pad=6)
    _titles(fig, f'Slip rates: {nfits} best fits to noise-perturbed data',
            f'each section\'s target against the fits\' predictions; the target is inside the 16-84% range for '
            f'{100 * inside1:.0f}% of the {S} sections',
            'In units of each section\'s measurement uncertainty (the mean over its subsections). Each fit is the best '
            'nonnegative fit to the data plus a fresh draw of their stated noise: the spread shows how far the best fit '
            'moves with the noise, not a posterior. The axes are fixed by the targets and the best fit, as in the other fit '
            'figures of these inputs; values beyond them sit on the edges.', h_in)
    _legend(fig, h_in, extra_handles=_ref_handles(ref))
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - (top_in + 0.3) / h_in, bottom=(bot_in + 0.25) / h_in, wspace=0.22)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return dict(sections=S, inside1=inside1, ratio=float(np.median(q[2] / ts)))


def fig_mfd(out, P, Y, y0, bins_block, lowest, width, nfits, ref=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, R, orig, lo = magnitude_bins(P, bins_block, lowest, width)
    Gr = P['G'].tocsr()
    coef = np.array([Gr.data[Gr.indptr[r]] for r in rows])   # each bin row: 1 / its uncertainty, for every rupture in it
    T = Y[:, rows] / coef                                # fits x bins: bin totals (events per year)
    T0 = y0[rows] / coef
    Tr = ref['y'][rows] / coef if ref else None          # the reference's bin totals
    rel = 1.0 / P['d'][rows]                             # each bin's relative uncertainty, as the likelihood weighs it
    s2 = float(P['sigma']) ** 2                          # chi2 over the bin rows, as the likelihood weighs them
    chi = np.sum((Y[:, rows] - P['d'][rows]) ** 2, axis=1) / s2
    chi0 = float(np.sum((y0[rows] - P['d'][rows]) ** 2)) / s2
    chir = float(np.sum((ref['y'][rows] - P['d'][rows]) ** 2)) / s2 if ref else None
    q, qr = _pct(T), _pct(T / R)
    inside1 = np.mean((qr[1] <= 1) & (1 <= qr[3]))
    mc = lo + width / 2
    _table(os.path.splitext(out)[0] + '.txt',
           [f'magnitude bins: {len(rows)} ({lo[0]:.2f}-{lo[-1] + width:.2f}, {width:g} wide; original rows {orig.min()}-{orig.max()}); '
            f'bin totals in events per year; {nfits} randomized best fits',
            f'the target lies inside the fits\' 16-84% range for {100 * inside1:.0f}% of the bins; the fits\' median bin total / '
            f'target: median {np.median(qr[2]):.3f} ({qr[2].min():.3f} to {qr[2].max():.3f}); total rate: the fits\' median '
            f'{np.median(T.sum(1)):.4g} (16-84%: {np.percentile(T.sum(1), 16):.4g} to {np.percentile(T.sum(1), 84):.4g}), '
            f'targets {R.sum():.4g}, best fit {T0.sum():.4g}',
            f'chi2 over the {len(rows)} bin rows, as the likelihood weights them: the fits\' median {np.median(chi):.1f} '
            f'(16-84%: {np.percentile(chi, 16):.1f} to {np.percentile(chi, 84):.1f}), the best fit {chi0:.1f}'
            + (f', the reference {chir:.1f}' if ref else '')]
           + _ref_head(ref, f'bin totals summed {Tr.sum():.4g}; bin total / target: median {np.median(Tr / R):.3f}' if ref else ''),
           ['bin', 'target', 'rel_unc', 'best_fit'] + [f'p{v:g}' for v in Q] + ['median/target'] + (['reference'] if ref else []),
           [[f'{lo[j]:.2f}', f'{R[j]:.4g}', f'{rel[j]:.3f}', f'{T0[j]:.4g}'] + [f'{q[i][j]:.4g}' for i in range(5)]
            + [f'{qr[2][j]:.3f}'] + ([f'{Tr[j]:.4g}'] if ref else []) for j in range(len(rows))])
    top_in, row_in, bot_in = 0.95, 4.6, 1.15
    fig, axs = plt.subplots(1, 2, figsize=(10.6, top_in + row_in + bot_in), facecolor=SURFACE)
    h_in = fig.get_size_inches()[1]
    from matplotlib.lines import Line2D
    ax = axs[0]
    _style(ax, logy=True, plain=False)
    ax.plot(mc, R, color=INK, lw=1.0, zorder=1)
    ax.fill_between(mc, R * np.maximum(1 - rel, 1e-3), R * (1 + rel), color=BAND, lw=0, zorder=0.5)
    lo_t = min((R * np.maximum(1 - rel, 1e-3)).min(), T0[T0 > 0].min() if (T0 > 0).any() else np.inf) / 1.3
    hi_t = max((R * (1 + rel)).max(), T0.max()) * 1.3    # fixed by the targets and the best fit
    ax.set_ylim(lo_t, hi_t)
    _ranges(ax, mc, q, base=T0, floor=lo_t * 1.06, ceil=hi_t / 1.06)
    if ref:
        _reference(ax, mc, Tr, floor=lo_t * 1.06, ceil=hi_t / 1.06)
    ax.set_xlabel('magnitude (bin centre)', color=INK2, fontsize=9)
    ax.set_ylabel('events per year in the bin', color=INK2, fontsize=9)
    ax.set_title('bin totals', loc='left', color=INK, fontsize=10, pad=6)
    ax = axs[1]
    _style(ax)
    ax.fill_between(mc, 1 - rel, 1 + rel, color=BAND, lw=0, zorder=0.5)
    ax.axhline(1.0, color=INK, lw=1.0, zorder=1)
    r_lo = min(1 - rel.max(), (T0 / R).min()) - 0.04
    r_hi = max(1 + rel.max(), (T0 / R).max()) + 0.04
    ax.set_ylim(r_lo, r_hi)
    pad = 0.015 * (r_hi - r_lo)
    _ranges(ax, mc, qr, base=T0 / R, floor=r_lo + pad, ceil=r_hi - pad)
    if ref:
        _reference(ax, mc, Tr / R, floor=r_lo + pad, ceil=r_hi - pad)
    ax.set_xlabel('magnitude (bin centre)', color=INK2, fontsize=9)
    ax.set_ylabel('bin total / target', color=INK2, fontsize=9)
    ax.set_title('relative to the targets', loc='left', color=INK, fontsize=10, pad=6)
    _titles(fig, f'Magnitude bins: {nfits} best fits to noise-perturbed data',
            f'the target is inside the fits\' 16-84% range for {100 * inside1:.0f}% of the {len(rows)} bins; median bin '
            f'total / target {np.median(qr[2]):.3f}; χ² over the bins: the fits\' median {np.median(chi):.1f} (16-84%: '
            f'{np.percentile(chi, 16):.1f} to {np.percentile(chi, 84):.1f}), the best fit {chi0:.1f}',
            'The shaded band is each bin\'s target plus or minus its uncertainty, at the level the likelihood gives it '
            '(MFD_UNCERTAINTY). Each fit is the best nonnegative fit to the data plus a fresh draw of their stated noise: '
            'the spread shows how far the best fit moves with the noise, not a posterior. The axes are fixed by the '
            'targets and the best fit; values beyond them sit on the edges.', h_in)
    _legend(fig, h_in, extra_handles=[Line2D([], [], color=INK, lw=1.0, label='target')]
            + _ref_handles(ref, f' (χ² {chir:.1f})' if ref else ''),
            band_label='within one uncertainty of the target')
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - (top_in + 0.3) / h_in, bottom=(bot_in + 0.5) / h_in, wspace=0.22)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return dict(bins=len(rows), inside1=inside1, ratio=float(np.median(qr[2])), total=T.sum(1), total0=float(T0.sum()),
                targets=float(R.sum()))


def paleo_rows(P, paleo_block, slip_block, bins_block):
    """The paleo rows as plot_paleo_rates.py takes them: --paleo-block, or every row in neither the slip-rate nor the
    bin block whose target is positive (rows with a target of 0 are zero-forcing penalty rows)."""
    names = list(P['block_names'])
    Gr = P['G'].tocsr()
    nonempty = np.diff(Gr.indptr) > 0
    if paleo_block is not None:
        if paleo_block not in names:
            raise SystemExit(f'no row block named {paleo_block!r}; the blocks are {names}')
        return np.flatnonzero((P['block'] == names.index(paleo_block)) & nonempty)
    other = [names.index(b) for b in (slip_block, bins_block) if b in names]
    rows = np.flatnonzero(~np.isin(P['block'], other) & nonempty)
    return rows[P['d'][rows] > 0]


def fig_paleo(out, P, Y, y0, rows, nfits, ref=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scale = P['row_scale'][rows]
    t = P['d'][rows] / scale                             # targets in units of each row's uncertainty
    Yp, yp0 = Y[:, rows] / scale, y0[rows] / scale
    ypr = ref['y'][rows] / scale if ref else None        # the reference's predictions
    q = _pct(Yp)
    inside1 = np.mean((q[1] <= t) & (t <= q[3]))
    orig = np.asarray(P['keep_rows'])[rows]
    within = np.mean(np.abs(Yp - t) <= 1.0, axis=0)
    _table(os.path.splitext(out)[0] + '.txt',
           [f'paleo rows: {len(rows)} (original rows {orig.min()}-{orig.max()}); targets and predictions in units of each '
            f'row\'s uncertainty; {nfits} randomized best fits',
            f'the target lies inside the fits\' 16-84% range for {100 * inside1:.0f}% of the rows; the fits\' median '
            f'prediction / target: median {np.median(q[2] / t):.3f} over the rows']
           + _ref_head(ref, f'prediction / target: median {np.median(ypr / t):.3f}' if ref else ''),
           ['row', 'target', 'best_fit', 'p2.5', 'p16', 'median', 'p84', 'p97.5', 'within_1'] + (['reference'] if ref else []),
           [[str(int(orig[i])), f'{t[i]:.4g}', f'{yp0[i]:.4g}'] + [f'{q[k][i]:.4g}' for k in range(5)] + [f'{within[i]:.2f}']
            + ([f'{ypr[i]:.4g}'] if ref else []) for i in range(len(rows))])
    top_in, row_in, bot_in = 0.95, 5.6, 1.15
    fig, ax = plt.subplots(1, 1, figsize=(7.2, top_in + row_in + bot_in), facecolor=SURFACE)
    h_in = fig.get_size_inches()[1]
    _style(ax, logx=True, logy=True)
    lo_b, hi_b = axis_span(t, yp0, log=True)             # fixed by the targets and the best fit, as plot_paleo_rates.py's
    lim = (lo_b / 1.4, hi_b * 1.4)
    xs = np.logspace(np.log10(lim[0]), np.log10(lim[1]), 400)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_box_aspect(1)
    ax.fill_between(xs, np.maximum(xs - 2.0, lim[0]), xs + 2.0, color=BAND2, lw=0, zorder=0.4)
    ax.fill_between(xs, np.maximum(xs - 1.0, lim[0]), xs + 1.0, color=BAND, lw=0, zorder=0.5)
    ax.plot(lim, lim, color=INK, lw=1.0, zorder=1)
    _ranges(ax, t, q, floor=lim[0] * 1.12, ceil=lim[1] / 1.12, base=yp0)
    if ref:
        _reference(ax, t, ypr, floor=lim[0] * 1.12, ceil=lim[1] / 1.12)
    ax.set_xlabel('target paleo rate (in units of its uncertainty)', color=INK2, fontsize=9)
    ax.set_ylabel('predicted paleo rate (same units)', color=INK2, fontsize=9)
    _titles(fig, f'Paleo rates: {nfits} best fits to noise-perturbed data',
            f'the target is inside the fits\' 16-84% range for {100 * inside1:.0f}% of the {len(rows)} rows',
            'In units of each row\'s uncertainty: on log-log axes that moves a point only along the 1:1 line, so its '
            'distance from the line is the ratio of prediction to target. The axes are fixed by the targets and the best fit, '
            'as in the other fit figures of these inputs; values beyond them (including 0) sit on the edges.', h_in)
    _legend(fig, h_in, extra_handles=_ref_handles(ref), band_label='within one uncertainty', band2_label='within two')
    fig.subplots_adjust(left=0.12, right=0.97, top=1 - (top_in + 0.3) / h_in, bottom=(bot_in + 0.5) / h_in)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return dict(rows=len(rows), inside1=inside1, ratio=float(np.median(q[2] / t)))


# ------------------------------------------------------------------------------------------------------------ self-test
def selftest():
    """(1) For a full-rank problem whose rates all lie well above 0, the fits scatter as the posterior under a flat prior:
    covariance sigma^2 (G^T G)^-1 around the best fit. (2) The fits do not depend on the number of workers."""
    g = np.random.default_rng(5)
    G = np.abs(g.normal(1.0, 0.5, (120, 6)))
    sigma = 0.05
    d = G @ np.array([1.0, 2.0, 1.5, 0.8, 1.2, 2.5]) + sigma * g.standard_normal(120)
    base = nnls_lawson_hanson(G, d)[0]
    res = ensemble((sp.csc_matrix(G), d, sigma), 400, 3, 1, verbose=False)
    X = np.zeros((len(res), G.shape[1]))
    for r in res:
        X[r['k'], r['idx']] = r['val']
    C, Cex = np.cov(X.T), sigma ** 2 * np.linalg.inv(G.T @ G)
    ratio = np.diag(C) / np.diag(Cex)                    # 400 fits: each about 1 +- 0.07
    z = (X.mean(0) - base) / np.sqrt(np.diag(Cex) / len(res))
    ok1 = bool(np.all(np.abs(ratio - 1) < 0.3) and np.all(np.abs(z) < 4) and X.min() > 0)
    print(f'1) {len(res)} fits of a 120 x 6 problem with every rate well above 0: variances / sigma^2 (G^T G)^-1 '
          f'{np.round(ratio, 3)} (1 expected); mean minus the best fit, in standard errors, {np.round(z, 2)}: '
          + ('PASSED' if ok1 else 'FAILED'))
    a = ensemble((sp.csc_matrix(G), d, sigma), 12, 7, 1, verbose=False)
    b = ensemble((sp.csc_matrix(G), d, sigma), 12, 7, 2, verbose=False)
    ok2 = all(np.array_equal(u['idx'], v['idx']) and np.array_equal(u['val'], v['val']) and np.array_equal(u['y'], v['y'])
              for u, v in zip(a, b))
    print(f'2) 12 fits in one process and in two: ' + ('identical, PASSED' if ok2 else 'DIFFERENT, FAILED'))
    print('SELFTEST (randomized best fits)', 'PASSED' if ok1 and ok2 else 'FAILED')
    return ok1 and ok2


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--prelim', default=None, help='diagnostics from preliminaries.py, for the unperturbed best fit '
                    '(default: next to --problem)')
    ap.add_argument('--fits', type=int, default=100, help='number of fits (default 100)')
    ap.add_argument('--seed', type=int, default=1, help='seed of the noise draws (default 1)')
    ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1), help='processes running fits, one thread '
                    'each (default 4, or fewer cores)')
    ap.add_argument('--out', default=None, help='output folder (default: rnnls next to --problem)')
    ap.add_argument('--slip-block', default='S')
    ap.add_argument('--bins-block', default='MFD')
    ap.add_argument('--paleo-block', default=None, help='the paleo row group (default: every row in neither the slip-rate '
                    'nor the magnitude-bin group whose target is positive)')
    ap.add_argument('--lowest-bin', type=float, default=6.8, help='lower magnitude edge of the lowest bin row left after '
                    'cleaning (default 6.8)')
    ap.add_argument('--bin-width', type=float, default=0.1, help='magnitude width of every bin (default 0.1)')
    ap.add_argument('--reference', default=None, metavar='RATES[:LABEL]', help="another model's rates to mark on the "
                    "figures, such as the model's OpenSHA solution (its solution/rates.csv), or rate vectors (.npz)")
    ap.add_argument('--selftest', action='store_true', help='exact checks on a small problem, then stop')
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.fits < 2:
        raise SystemExit('--fits: at least 2 (the ranges need several fits)')
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.problem)), 'rnnls')
    os.makedirs(out, exist_ok=True)
    report_to(os.path.join(out, 'rnnls_report.txt'))
    P = load_problem(a.problem)
    G = sp.csc_matrix(P['G'])
    N, n = G.shape
    sigma = float(P['sigma'])
    fp = a.prelim or prelim_path(a.problem)
    base = None
    if os.path.exists(fp):
        with np.load(fp) as pr:
            if 'm_nnls' in pr.files and len(pr['m_nnls']) == n:
                base = np.asarray(pr['m_nnls'], np.float64)
    t0 = time.time()
    if base is None:
        print(f'the unperturbed best fit is not in {fp} (cleaning skipped it): computing it here')
        base = nnls_lawson_hanson(G, P['d'], threads=max(1, a.workers))[0]
    y0 = G @ base
    chi0 = float(np.sum((y0 - P['d']) ** 2)) / sigma ** 2
    fit_check = P.get('chi2_expect')
    print(f'randomized best fits: {a.fits} NNLS fits of {a.problem} ({N} rows x {n} ruptures) to the data plus fresh draws '
          f'of their stated noise (sigma {sigma:g} on every cleaned row), seed {a.seed}, {a.workers} worker'
          f'{"s" if a.workers > 1 else ""}')
    print(f'the unperturbed best fit: chi2 {chi0:.1f}' + (f' (fit check {fit_check[0]:.0f} +- {fit_check[1]:.0f})'
                                                          if fit_check is not None else '')
          + f', {np.count_nonzero(base)} nonzero rates')
    res = ensemble(os.path.abspath(a.problem), a.fits, a.seed, a.workers)
    seconds = time.time() - t0
    K = len(res)
    Y = np.array([r['y'] for r in res])                  # fits x rows, as the likelihood sees them
    nnz = np.array([len(r['idx']) for r in res])
    ptr = np.r_[0, np.cumsum(nnz)]
    idx = np.concatenate([r['idx'] for r in res])
    val = np.concatenate([r['val'] for r in res])
    chi_obs, chi_own = np.array([r['chi2_obs'] for r in res]), np.array([r['chi2_own'] for r in res])
    tot = np.array([r['val'].sum() for r in res])
    mean = np.bincount(idx, weights=val, minlength=n) / K
    share = np.bincount(idx, minlength=n) / K
    S0 = base > 0
    keep_base = np.array([np.mean(S0[r['idx']]) if len(r['idx']) else 0.0 for r in res])   # each fit's rates on base's
    np.savez(os.path.join(out, 'rnnls_fits.npz'), y=Y.astype(np.float32), chi2_obs=chi_obs, chi2_own=chi_own, total=tot,
             iterations=np.array([r['it'] for r in res]), seconds=np.array([r['seconds'] for r in res]), ptr=ptr,
             idx=idx.astype(np.int32), val=val, base=base, y_base=y0, seed=a.seed, fits=K, sigma=sigma,
             keep_cols=np.asarray(P['keep_cols']), keep_rows=np.asarray(P['keep_rows']))
    n_raw = P['raw_shape'][1] if P.get('raw_shape') else n
    kc = np.asarray(P['keep_cols'])
    full = lambda v: (lambda o: (o.__setitem__(kc, v), o)[1])(np.zeros(n_raw))   # original rupture numbering, removed 0
    pen = np.ones(n_raw, int)
    pen[kc] = 0
    m_o, s_o, b_o = full(mean), full(share), full(base)
    np.savez(os.path.join(out, 'rate_vectors.npz'), mean=m_o, nonzero_share=s_o, best_fit=b_o, penalised=pen)
    with open(os.path.join(out, 'rate_vectors.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['rupture', 'penalised', 'mean', 'nonzero_share', 'best_fit'])
        wr.writerows([i, pen[i], f'{m_o[i]:.6e}', f'{s_o[i]:.4f}', f'{b_o[i]:.6e}'] for i in range(n_raw))
    print(f'done: {K} fits in {seconds / 60:.1f} min ({seconds / K:.1f} s per fit with {a.workers} worker'
          f'{"s" if a.workers > 1 else ""}; one fit {np.median([r["seconds"] for r in res]):.1f} s at the median, '
          f'{int(np.median([r["it"] for r in res]))} iterations)')
    print(f'chi2 against the observed data: {chi_obs.mean():.1f} +- {chi_obs.std():.1f} (the best fit {chi0:.1f}); against '
          f'each fit\'s own perturbed data {chi_own.mean():.1f} +- {chi_own.std():.1f}')
    print(f'total rate: {np.median(tot):.4g} at the median (16-84%: {np.percentile(tot, 16):.4g} to '
          f'{np.percentile(tot, 84):.4g}); the best fit {base.sum():.4g}')
    print(f'nonzero rates per fit: {int(np.median(nnz))} at the median ({nnz.min()} to {nnz.max()}); {int((share > 0).sum())} '
          f'ruptures nonzero in some fit, {int((share == 1).sum())} in every fit; of each fit\'s nonzero rates, a median '
          f'{100 * np.median(keep_base):.0f}% are on ruptures the unperturbed best fit uses')
    ref = None
    if a.reference:                                      # another model's predictions, marked on the figures
        ref = reference_rates(a.reference, P)
        ref['y'] = G @ ref['m']
        print(f'reference {ref["label"]} ({ref["path"]}): chi2 against the observed data '
              f'{float(np.sum((ref["y"] - P["d"]) ** 2)) / sigma ** 2:.1f}, total rate {ref["total"]:.4g}'
              + (f'; {ref["note"]}' if ref['note'] else ''))
    sl = fig_slip(os.path.join(out, 'fig_rnnls_slip_rates.png'), P, Y, y0, a.slip_block, K, ref=ref)
    print(f'slip rates: the target is inside the fits\' 16-84% range for {100 * sl["inside1"]:.0f}% of the {sl["sections"]} '
          f'sections; median prediction / target {sl["ratio"]:.3f} (median over sections)')
    mf = fig_mfd(os.path.join(out, 'fig_rnnls_mfd.png'), P, Y, y0, a.bins_block, a.lowest_bin, a.bin_width, K, ref=ref)
    print(f'magnitude bins: the target is inside the fits\' 16-84% range for {100 * mf["inside1"]:.0f}% of the {mf["bins"]} '
          f'bins; median bin total / target {mf["ratio"]:.3f}; bin totals summed: {np.median(mf["total"]):.4g} at the median '
          f'(targets {mf["targets"]:.4g}, the best fit {mf["total0"]:.4g})')
    rows = paleo_rows(P, a.paleo_block, a.slip_block, a.bins_block)
    if len(rows):
        pa = fig_paleo(os.path.join(out, 'fig_rnnls_paleo_rates.png'), P, Y, y0, rows, K, ref=ref)
        print(f'paleo rates: the target is inside the fits\' 16-84% range for {100 * pa["inside1"]:.0f}% of the {pa["rows"]} '
              f'rows; median prediction / target {pa["ratio"]:.3f} (median over rows)')
    else:
        print('no paleo rows found: no paleo figure (name them with --paleo-block)')
    print(f'wrote {out}/: rnnls_fits.npz, rate_vectors.npz and .csv, fig_rnnls_slip_rates.png, fig_rnnls_mfd.png'
          + (', fig_rnnls_paleo_rates.png' if len(rows) else '') + ' (each with its table, .txt), rnnls_report.txt')


if __name__ == '__main__':
    main()

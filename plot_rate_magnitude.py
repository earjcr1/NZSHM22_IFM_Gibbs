#!/usr/bin/env python3
"""Rates by magnitude: log10 of each rupture's posterior mean rate against its magnitude bin.

    python plot_rate_magnitude.py VECTORS[:LABEL] [VECTORS[:LABEL] ...] [--problem cru_clean.npz] [--out FIGURE]
                                  [--lowest-bin 6.8] [--bin-width 0.1] [--diag-floor -10]
                                  [--bins-block MFD] [--slip-block S]

VECTORS is a result set's rate vectors as the notebook writes them (<prior folder>/rate_vectors, with or without
.npz); their `mean` column is each rupture's posterior mean rate, pooled over the replicates. VECTORS may also be a
single model's, such as the best fit that preliminaries.py writes (nnls_vectors.npz); when every result set is one,
the labels speak of rates rather than posterior means. Several result sets
give panels side by side on one rate axis (to compare priors, say), each rupture at the same magnitude in every
panel. LABEL names a panel; by default it is the rate vectors' folder (the prior).

Magnitudes: a rupture's magnitude is known only to its bin, the magnitude-bin row it appears in. The bin rows are
taken as consecutive bins of --bin-width (0.1) in their original row order, the lowest one left after cleaning being
--lowest-bin (6.80-6.90 for the NZSHM22 inputs, whose lower bins hold only the penalised ruptures). A bin that
cleaning removed (no ruptures left) leaves a gap. Within its bin each rupture is drawn at a random magnitude, only to
spread the points. A check prints how the median number of subsections per rupture changes from bin to bin; it should
rise with magnitude if the bins are in order.

Marks: a dot per rupture at log10 of its mean rate, with means below 10^--diag-floor (1e-10, the notebook's FLOOR),
including exactly 0, in a band at the bottom; a line across each bin at its target total rate R_j (with the
notebook's default soft top, BOX_TOP = 'soft', also every rupture's soft upper limit); and a dot for the bin's total
of the means, which is the posterior mean of the bin's total. Penalised ruptures (rate 0 by construction) are left
out.

Writes FIGURE (default: fig_rate_magnitude.png next to the first rate vectors) and, as its table view, the same name
with .txt: per bin, the number of ruptures, the target, the total of the means and their ratio, the 10%, 50% and 90%
points and the largest of log10 of the mean rates, and how many lie below the floor; then, per bin, how many ruptures
have a mean rate below the floor (including 0), in each decade from the floor up to 1e-2, and at 1e-2 or above, so
that every rupture is counted once. Means in the lowest decades come from ruptures switched on in few draws, so their
counts depend on run length.
"""
import argparse
import os

import numpy as np
from scipy.stats import spearmanr

from cru_common import load_problem, magnitude_bins

SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ebeae5'
BLUE, ORANGE = '#2a78d6', '#eb6834'          # categorical slots 1 and 2 of the reference palette, validated as a pair
MINUS = '−'


def parse_set(s):
    """VECTORS[:LABEL] -> (path of the .npz, label)."""
    path, label = s, None
    if ':' in s:
        head, tail = s.rsplit(':', 1)
        if tail and '/' not in tail and os.sep not in tail:
            path, label = head, tail
    base = path[:-4] if path.endswith(('.npz', '.csv')) else path
    return base + '.npz', label or os.path.basename(os.path.dirname(os.path.abspath(base))) or base


def log_label(v):
    return f'{v:g}'.replace('-', MINUS)


def main():
    ap = argparse.ArgumentParser(description='Rates by magnitude (see the module docstring).')
    ap.add_argument('sets', nargs='+', metavar='VECTORS[:LABEL]', help='rate vectors (<prior folder>/rate_vectors)')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--out', default=None, help='figure (default: fig_rate_magnitude.png next to the first rate vectors)')
    ap.add_argument('--lowest-bin', type=float, default=6.8, help='lower magnitude edge of the lowest bin row left after '
                    'cleaning (default 6.8)')
    ap.add_argument('--bin-width', type=float, default=0.1, help='magnitude width of every bin (default 0.1)')
    ap.add_argument('--diag-floor', type=float, default=-10.0, help='log10 rate below which means go in the bottom band '
                    '(default -10; the notebook passes FLOOR)')
    ap.add_argument('--bins-block', default='MFD')
    ap.add_argument('--slip-block', default='S')
    a = ap.parse_args()
    sets = [parse_set(s) for s in a.sets]
    out = a.out or os.path.join(os.path.dirname(sets[0][0]) or '.', 'fig_rate_magnitude.png')
    floor, w = float(a.diag_floor), float(a.bin_width)

    P = load_problem(a.problem)
    Gr = P['G'].tocsr()
    n = Gr.shape[1]
    keep_c = np.asarray(P['keep_cols'])
    rows, R, orig, lo = magnitude_bins(P, a.bins_block, a.lowest_bin, w)
    B = len(rows)
    bin_of = np.asarray((Gr[rows] != 0).astype(np.int8).argmax(0)).ravel()      # each rupture is in exactly one bin
    rng_name = lambda k: f'{lo[k]:.2f}-{lo[k] + w:.2f}'
    gaps = int(orig[-1] - orig[0] + 1 - B)
    head = [f'magnitude bins: {B} bin rows (original rows {orig[0]}-{orig[-1]}'
            + (f', {gaps} missing' if gaps else '') + f'), counted up from {rng_name(0)} in steps of {w:g}; '
            f'{n:,} ruptures, each drawn at a random magnitude within its bin']
    if a.slip_block in P['block_names']:
        srows = np.flatnonzero(P['block'] == P['block_names'].index(a.slip_block))
        nsub = np.asarray((Gr[srows] != 0).sum(0)).ravel()
        med = np.array([np.median(nsub[bin_of == k]) for k in range(B)])
        rho = spearmanr(np.arange(B), med)[0] if B > 2 and np.ptp(med) > 0 else np.nan
        head.append(f'order check: the median number of subsections per rupture goes from {med[0]:.0f} ({rng_name(0)}) to '
                    f'{med[-1]:.0f} ({rng_name(B - 1)}), rank correlation with magnitude {rho:+.2f}')
        if not rho > 0.5:
            head.append('WARNING: rupture sizes do not rise with the bins\' original order; check the bin order, '
                        '--lowest-bin and --bin-width before reading magnitudes off this figure')
    else:
        head.append(f'order check skipped: no row block named {a.slip_block!r}')

    data = []
    for path, label in sets:
        z = np.load(path)
        m_all = np.asarray(z['mean'], np.float64)
        n_raw = P['raw_shape'][1] if P.get('raw_shape') else len(m_all)
        if len(m_all) != n_raw or keep_c.max() >= len(m_all):
            raise SystemExit(f'{path} has {len(m_all)} ruptures; the problem {a.problem} has {n_raw} (another problem?)')
        pen = int(np.sum(z['penalised'])) if 'penalised' in z.files else n_raw - n
        m = m_all[keep_c]
        tot = np.bincount(bin_of, weights=m, minlength=B)
        data.append(dict(path=path, label=label, m=m, tot=tot, below=m < 10.0 ** floor, pen=pen,
                         model=str(z['model']) if 'model' in z.files else ''))
    single = all(D['model'] for D in data)           # single models (a best fit): rates, not posterior means
    mean_ = '' if single else 'mean '

    # table view
    clip = lambda v: f'<{floor:g}' if v <= floor else f'{v:.2f}'
    dec = [floor] + [float(e) for e in range(int(np.floor(floor)) + 1, -1)]      # decade edges: the floor, ..., 1e-2
    dlab = [f'<1e{floor:g}'] + [f'1e{e:g}' for e in dec[:-1]] + [f'>=1e{dec[-1]:g}']
    dw = max(6, max(len(s) for s in dlab)) + 1
    lines = list(head)
    for D in data:
        Lm = np.log10(np.maximum(D['m'], 10.0 ** floor))
        lines += ['', f'{D["label"]} ({D["path"]}): {D["pen"]:,} penalised ruptures (rate 0 by construction) not shown',
                  f'{"bin":<11s} {"ruptures":>8s} {"target R_j":>11s} {"total" if single else "total of means":>14s} {"ratio":>6s} '
                  f'{"log10 rate: 10%" if single else "log10 mean: 10%":>16s} {"50%":>7s} {"90%":>7s} {"max":>7s} '
                  f'{"below " + f"1e{floor:g}":>12s}']
        for k in list(range(B)) + [None]:
            sel = bin_of == k if k is not None else np.ones(n, bool)
            tgt = R[k] if k is not None else R.sum()
            tt = D['tot'][k] if k is not None else D['tot'].sum()
            q = np.percentile(Lm[sel], [10, 50, 90])
            lines.append(f'{(rng_name(k) if k is not None else "all"):<11s} {int(sel.sum()):8d} {tgt:11.3e} {tt:14.3e} '
                         f'{tt / tgt:6.3f} {clip(q[0]):>16s} {clip(q[1]):>7s} {clip(q[2]):>7s} {clip(Lm[sel].max()):>7s} '
                         f'{int(D["below"][sel].sum()):12d}')
        with np.errstate(divide='ignore'):                # a mean of 0 is below the floor
            idx = np.where(D['below'], 0, np.maximum(np.digitize(np.log10(D['m']), dec), 1))
        cnt = np.bincount(bin_of * len(dlab) + idx, minlength=B * len(dlab)).reshape(B, len(dlab))
        what = 'rates' if single else 'means'
        lines += ['', f"ruptures by decade of {mean_}rate (each column counts the {what} from its value up to the next one's; "
                      f"<1e{floor:g} includes {what} of exactly 0):",
                  f'{"bin":<11s}' + ''.join(f'{s:>{dw}s}' for s in dlab)]
        for k in list(range(B)) + [None]:
            c = cnt[k] if k is not None else cnt.sum(0)
            lines.append(f'{(rng_name(k) if k is not None else "all"):<11s}' + ''.join(f'{int(v):{dw}d}' for v in c))
    text = '\n'.join(lines)
    print(text)

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    g = np.random.default_rng(0)                      # the same spread for every panel, and on every run
    x = lo[bin_of] + w * g.uniform(0.08, 0.92, n)
    jy = g.uniform(-0.28, 0.28, n)
    any_below = any(D['below'].any() for D in data)
    band_c = floor - 1.0
    top = max(np.log10(max(D['m'].max(), D['tot'].max(), R.max())) for D in data)
    top = np.ceil(2.0 * (top + 0.25)) / 2.0
    bottom = floor - 1.5 if any_below else floor - 0.3
    k_panels = len(data)
    fig_w = 8.6 if k_panels == 1 else 0.6 + 6.3 * k_panels
    fig, axs = plt.subplots(1, k_panels, figsize=(fig_w, 6.0), sharey=True, squeeze=False,
                            facecolor=SURFACE)
    edges = lo[0] + w * np.arange(int(round((lo[-1] + w - lo[0]) / w)) + 1)
    majors = [e for e in edges if abs(round(e * 10) % 2) < 1e-9 or w != 0.1]
    for ax, D in zip(axs[0], data):
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS)
            ax.spines[s].set_linewidth(0.5)
        ax.set_xticks(edges, minor=True)
        ax.set_xticks(majors)
        ax.set_xticklabels([f'{e:.1f}' for e in majors])
        ax.grid(True, which='both', axis='x', color=GRID, lw=0.5, zorder=0)
        ax.grid(True, which='major', axis='y', color=GRID, lw=0.5, zorder=0)
        ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=8.5, which='both', length=3, width=0.5)
        ax.set_xlim(lo[0] - 0.4 * w, lo[-1] + 1.4 * w)
        ax.set_ylim(bottom, top)
        b = D['below']
        if any_below:
            ax.axhspan(floor - 1.45, floor - 0.55, color=BAND, lw=0, zorder=0.5)
        ax.scatter(x[~b], np.log10(D['m'][~b]), s=3.5, c=BLUE, alpha=0.25, linewidths=0, zorder=2)
        if b.any():
            ax.scatter(x[b], band_c + jy[b], s=3.5, c=BLUE, alpha=0.25, linewidths=0, zorder=2)
        ax.hlines(np.log10(R), lo + 0.06 * w, lo + 0.94 * w, colors=INK, lw=1.0, zorder=3)
        ok = D['tot'] > 0
        ax.scatter(lo[ok] + 0.5 * w, np.log10(D['tot'][ok]), s=30, c=ORANGE, edgecolors=SURFACE, linewidths=1.0, zorder=4)
        nb = int(b.sum())
        nz = int((D['m'] == 0).sum())
        ax.set_title(D['label'], loc='left', color=INK, fontsize=11.5, fontweight='bold', pad=20)
        ax.annotate(f'{n:,} ruptures; ' + (f'{nb:,} with a mean below 1e{floor:g}' + (f', {nz:,} of them exactly 0' if nz else '')
                    if nb else f'every mean above 1e{floor:g}'),
                    (0, 1), xycoords='axes fraction', xytext=(0, 5), textcoords='offset points', va='bottom',
                    color=INK2, fontsize=8.5)
        ax.set_xlabel(f'Magnitude bin (each rupture at a random magnitude within its {w:g} bin)', color=INK2, fontsize=9)
    ticks = list(range(int(np.ceil(floor)), int(np.floor(top)) + 1))
    axs[0][0].set_yticks(ticks + ([band_c] if any_below else []))
    axs[0][0].set_yticklabels([log_label(t) for t in ticks] + ([f'< {log_label(floor)}'] if any_below else []))
    axs[0][0].set_ylabel('log10 of the ' + ('' if single else 'posterior mean ') + 'rate (per year)', color=INK2, fontsize=9)
    handles = [Line2D([], [], ls='none', marker='o', ms=4, mfc=BLUE, mec=BLUE, alpha=0.6, label=f'rupture: {mean_}rate'),
               Line2D([], [], color=INK, lw=1.0, label='bin target R_j (total rate)'),
               Line2D([], [], ls='none', marker='o', ms=6.5, mfc=ORANGE, mec=SURFACE, mew=1.0,
                      label='bin total' + ('' if single else ' of the means'))]
    if any_below:
        handles.append(Patch(facecolor=BAND, edgecolor='none', label=f'band: {mean_}below 1e{floor:g}, including 0'))
    fig.suptitle(('Rate' if single else 'Posterior mean rate') + ' of each rupture, by magnitude', x=0.5 / fig_w + 0.01, ha='left',
                 color=INK, fontsize=13, fontweight='bold', y=0.975)
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=8.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.045), handletextpad=0.5, columnspacing=1.6)
    pens = sorted({D['pen'] for D in data})
    fig.text(0.5, 0.012, f'Bins counted up in the bin rows original order. '
             + (f'{pens[0]:,} penalised ruptures (rate 0 by construction) are not shown.' if len(pens) == 1 else
                'Penalised ruptures (rate 0 by construction) are not shown.'),
             ha='center', color=MUTED, fontsize=8)
    fig.subplots_adjust(left=0.07 if k_panels > 1 else 0.075, right=0.985, top=0.85, bottom=0.2, wspace=0.06)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    with open(os.path.splitext(out)[0] + '.txt', 'w') as f:
        f.write(text + '\n')
    print(f'wrote {out} and {os.path.splitext(out)[0]}.txt')


if __name__ == '__main__':
    main()

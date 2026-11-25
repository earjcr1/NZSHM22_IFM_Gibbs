#!/usr/bin/env python3
"""How the posterior's rates vary together, for the quantities hazard uses, and where that comes from.

    python covariance.py RUN [RUN ...] --problem cru_clean.npz --ruptures DIR [--outdir DIR]

RUN: a gibbs.py run's output stem (its <RUN>_kept.npz, the whole draws --keep-draws stores, or an older run's dense
<RUN>_kept.npy; replicates pooled).

(1) Over the kept draws: the correlation between the named faults' event rates (each the summed rate of the ruptures
crossing any of its subsections; the rupture set's named_faults.json, as plot_slip_sections.py has them) and between the
magnitude bins' totals, on the rates (Pearson) and, for the faults, on log10 of them.

(2) The data's own structure, from linear inverse theory: the eigenvectors of G G^T (G the cleaned problem, each row
divided by its uncertainty) are the directions the data resolve; each aggregate's resolved fraction is the share of its
functional (a fault's or a bin's indicator over the ruptures) in those directions, and the data-only covariance of the
aggregates' resolved parts is sigma^2 W^T S^-2 W (W = V^T b): Gaussian, without the prior, positivity or any
constraint, so it shows which trade-offs the data impose and which the model adds.

Writes OUTDIR/corr_faults.png and corr_bins.png (each the posterior's and the data's, side by side), covariance_report.txt
and covariance.npz (default OUTDIR: the first run's folder). Added 2026-11-17 at the user's request.
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cru_common import load_problem, magnitude_bins, run_settings   # noqa: E402

CMAP = ('#2a78d6', '#f0efec', '#e34948')     # diverging: blue (negative), neutral grey, red (positive)
INK, MUTED, SURF = '#0b0b0b', '#6f6e69', '#fcfcfb'


def corr(A):
    """Pearson correlations between the columns of A (draws x quantities)."""
    A = A - A.mean(0)
    s = np.sqrt((A ** 2).sum(0))
    s[s == 0] = np.inf
    return (A.T @ A) / np.outer(s, s)


def fault_incidence(P, G, rupture_dir, slip_block):
    """The named faults (names, fault x rupture 0/1 CSR: the ruptures crossing each), as plot_slip_sections.py has them;
    (None, None) without the rupture set's named_faults.json."""
    from plot_slip_sections import fault_sections, named_faults, connectivity
    path = os.path.join(rupture_dir, 'named_faults.json')
    if not os.path.exists(path):
        return None, None
    blk, names_b = np.asarray(P['block']), list(P['block_names'])
    rows = np.flatnonzero(blk == names_b.index(slip_block))
    orig = np.asarray(P['keep_rows'])[rows]
    sec = np.asarray(P['section'])[rows]
    names, groups, _ = named_faults(path, fault_sections(rupture_dir), orig)
    R = (G[rows] != 0).astype(np.float64).tocsr()
    B = connectivity(R, groups, sec, int(sec.max()) + 1)[3].tocsr()
    B.data[:] = 1.0
    return names, B


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--problem', default='cru_clean.npz')
    ap.add_argument('--ruptures', default=None, help='the rupture set (its named_faults.json and fault_sections.geojson)')
    ap.add_argument('--slip-block', default='S')
    ap.add_argument('--bins-block', default='MFD')
    ap.add_argument('--lowest-bin', type=float, default=6.8)
    ap.add_argument('--bin-width', type=float, default=0.1)
    ap.add_argument('--outdir', default=None)
    a = ap.parse_args()
    out = a.outdir or os.path.dirname(os.path.abspath(a.runs[0]))
    lines = []

    def say(t):
        print(t, flush=True)
        lines.append(t)

    P = load_problem(a.problem)
    G = sp.csr_matrix(P['G'])
    from cru_common import load_kept
    kept = [load_kept(r)[0] for r in a.runs]
    missing = [r for r, k in zip(a.runs, kept) if k is None]
    if missing:
        raise SystemExit(f'no kept draws: {", ".join(missing)} (gibbs.py --keep-draws)')
    fnames, B = fault_incidence(P, G, a.ruptures, a.slip_block) if a.ruptures else (None, None)
    mrows, _, _, mlo = magnitude_bins(P, a.bins_block, a.lowest_bin, a.bin_width)
    Gb = (G[mrows] != 0).astype(np.float64).tocsr()
    bnames = [f'M{v:.1f}' for v in mlo]

    # ---- (1) the kept draws: each fault's event rate and each bin's total, draw by draw
    F, Mb = [], []
    for X in kept:                                       # sparse (CSR), one row per draw
        for a0 in range(0, X.shape[0], 1000):
            xs = X[a0:a0 + 1000].T.astype(np.float64)
            Mb.append((Gb @ xs).toarray().T)
            if B is not None:
                F.append((B @ xs).toarray().T)
    Mb = np.vstack(Mb)
    nd = len(Mb)
    say(f'covariance of the posterior\'s rates: {nd:,} kept draws from {len(kept)} run(s); {G.shape[1]:,} ruptures; '
        + (f'{len(fnames)} named faults; ' if fnames else 'no named faults (no --ruptures or no named_faults.json); ')
        + f'{len(mrows)} magnitude bins')
    say(f'(a correlation near 0 has an sd of about {1 / np.sqrt(nd):.2f} at {nd:,} independent draws, more with fewer '
        'effective ones; Pearson on rates is swayed by a few large draws, so the faults\' are also given in log10)')
    Cb = corr(Mb)
    ib = np.triu_indices(len(mrows), 1)
    say(f'bin totals: neighbours r {np.min(np.diag(Cb, 1)):+.2f} to {np.max(np.diag(Cb, 1)):+.2f} (median '
        f'{np.median(np.diag(Cb, 1)):+.2f}); all pairs median {np.median(Cb[ib]):+.2f}, most negative {Cb[ib].min():+.2f}')
    res = dict(bnames=np.array(bnames), Cb=Cb, Mb=Mb)
    if fnames:
        F = np.vstack(F)
        Cf, Cfl = corr(F), corr(np.log10(np.maximum(F, 1e-12)))
        iu = np.triu_indices(len(fnames), 1)
        say(f'named faults\' event rates: {len(iu[0])} pairs, |r| > 0.2 in {int((np.abs(Cf[iu]) > 0.2).sum())} '
            f'({int((Cf[iu] < -0.2).sum())} negative); relative sd of each fault\'s rate {np.min(F.std(0) / F.mean(0)):.2f}'
            f'-{np.max(F.std(0) / F.mean(0)):.2f} (median {np.median(F.std(0) / F.mean(0)):.2f})')
        o = np.argsort(Cf[iu])
        for title, ks in (('most negative', o[:8]), ('most positive', o[::-1][:8])):
            say(f'  {title} (Pearson on rates; log10 in brackets):')
            for k in ks:
                i, j = iu[0][k], iu[1][k]
                say(f'    {Cf[i, j]:+.2f} ({Cfl[i, j]:+.2f})  {fnames[i]} / {fnames[j]}')
        res.update(fnames=np.array(fnames), Cf=Cf, Cf_log=Cfl, F=F)

    # ---- (2) the data's own structure: G G^T's eigenvectors, the resolved directions
    lam, U = np.linalg.eigh((G @ G.T).toarray())
    lam, U = lam[::-1], U[:, ::-1]
    r_ = int((lam > lam[0] * 1e-12).sum())
    s, U = np.sqrt(lam[:r_]), U[:, :r_]
    sig = float(P['sigma'])
    say(f'the data: {G.shape[0]:,} rows, rank {r_:,} (eigenvalues of G G^T above 1e-12 of the largest), singular values '
        f'{s[0]:.3g} to {s[-1]:.3g}; every other direction of the {G.shape[1]:,} rates is left to the prior and the constraints')

    def resolved(Bm):
        W = (U.T @ (G @ Bm.T).toarray()) / s[:, None]
        nb = np.asarray(Bm.multiply(Bm).sum(1)).ravel()
        C = (W / s[:, None] ** 2).T @ W * sig ** 2
        d = np.sqrt(np.diag(C))
        return (W ** 2).sum(0) / nb, C / np.outer(d, d)

    frb, Db = resolved(Gb)
    say(f'bins: resolved fraction {frb.min():.2f}-{frb.max():.2f}; data-only neighbours r median '
        f'{np.median(np.diag(Db, 1)):+.2f} (the posterior\'s {np.median(np.diag(Cb, 1)):+.2f})')
    res.update(Db=Db, frb=frb, s=s)
    if fnames:
        frf, Df = resolved(B)
        m = (np.abs(Df[iu]) > 0.2) | (np.abs(Cf[iu]) > 0.2)
        say(f'faults: resolved fraction of each event-rate functional {frf.min():.2f}-{frf.max():.2f} (median '
            f'{np.median(frf):.2f}); data-only |r| > 0.2 in {int((np.abs(Df[iu]) > 0.2).sum())} pairs; of the {int(m.sum())} '
            f'pairs above 0.2 in either, same sign in {int((np.sign(Df[iu][m]) == np.sign(Cf[iu][m])).sum())}'
            + (f', the two r\'s correlated {np.corrcoef(Df[iu][m], Cf[iu][m])[0, 1]:+.2f}' if m.sum() > 2 else ''))
        res.update(Df=Df, frf=frf)
    say('(the data-only figures are Gaussian over the resolved parts, without the prior, positivity or constraints: '
        'where they differ from the posterior\'s, the model, not the data, makes the difference)')

    np.savez(os.path.join(out, 'covariance.npz'), **res)
    with open(os.path.join(out, 'covariance_report.txt'), 'w') as f_:
        f_.write('\n'.join(lines) + '\n')

    # ---- figures
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    cm = LinearSegmentedColormap.from_list('div', CMAP)
    pairs = run_settings(None, a.problem)
    st = ', '.join(f'{k} {v}' for k, v in pairs) if pairs else ''
    sets = [('bins', Cb, Db, bnames, np.arange(len(bnames)), f'the {len(bnames)} magnitude bins’ totals')]
    if fnames:
        o = leaves_list(linkage(squareform(np.clip(1 - Cf, 0, 2), checks=False), 'average'))
        sets.insert(0, ('faults', Cf, Df, fnames, o, f'the {len(fnames)} named faults’ event rates (ordered by '
                                                      'clustering of the posterior’s)'))
    for tag, A1, A2, labels, ordr, title in sets:
        fs = 6.5 if len(labels) > 20 else 8
        fig, axs = plt.subplots(1, 2, figsize=(15, 7.6), facecolor=SURF)
        for ax, A, sub in ((axs[0], A1, f'posterior ({nd:,} kept draws)'),
                           (axs[1], A2, 'data alone (Gaussian, resolved parts; no prior, positivity or constraints)')):
            ax.set_facecolor(SURF)
            im = ax.imshow(A[np.ix_(ordr, ordr)], cmap=cm, vmin=-1, vmax=1, interpolation='nearest')
            lab = [labels[i] for i in ordr]
            ax.set_xticks(range(len(ordr)))
            ax.set_yticks(range(len(ordr)))
            ax.set_xticklabels(lab, rotation=90, fontsize=fs, color=INK)
            ax.set_yticklabels(lab, fontsize=fs, color=INK)
            ax.set_title(sub, fontsize=9.5, color=INK, loc='left')
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
        cb.set_label('correlation (Pearson, on rates)', fontsize=8.5, color=MUTED)
        cb.outline.set_visible(False)
        fig.suptitle(f'Correlation between {title}', x=0.01, ha='left', fontsize=12, fontweight='bold', color=INK)
        if st:
            fig.text(0.01, 0.935, 'settings: ' + st, fontsize=6.5, color=MUTED, ha='left', wrap=True)
        fig.savefig(os.path.join(out, f'corr_{tag}.png'), dpi=150, facecolor=SURF, bbox_inches='tight')
        plt.close(fig)
    print(f'wrote {out}/corr_' + ('faults.png, corr_' if fnames else '') + 'bins.png, covariance_report.txt and '
          'covariance.npz')


if __name__ == '__main__':
    main()

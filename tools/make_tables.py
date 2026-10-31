#!/usr/bin/env python3
"""The posterior tables and rate vectors of a result set, pooled over its replicate runs.

    python make_tables.py RUN [RUN ...] [--problem cru_clean.npz] [--prelim prelim_results.npz]
                          [--convergence STATS] [--outdir DIR] [--slip-block S] [--bins-block MFD] [--out REPORT]

RUN is a gibbs.py run (its --out, e.g. spikeslab/main_rep0), one per replicate. Each run is read from its running
summaries (<run>_stream.npz) unless every run has stored draws (<run>_samples.npy); the running summaries give the
exceedance probabilities, totals and shares exactly and the rupture quantiles to within the log-rate histogram's bin
width (0.01 decades by default).

Writes, in DIR (default: the first run's folder):
- ruptures_posterior.csv: per rupture of the cleaned problem (original numbering), its bin row, the posterior
  probabilities of exceeding 1e-5 and 1e-4, the 5%, 50% and 95% points of its rate (the lower two prior-dependent),
  and how well it mixed: the worst R-hat of its log rate over the runs, its effective draws summed over them, and a
  slow-mixing flag, from check_convergence.py's STATS (default: convergence.npz in DIR; nan without it);
- subsections_posterior.csv and bins_posterior.csv: per slip-rate row and per magnitude-bin row, the 5%, 50% and 95%
  points of the total rate of the ruptures it holds, the best fit's total (from preliminaries.py's diagnostics) and the
  share of the posterior-mean total carried by rates below their own detection scale sigma/||G_i|| (best_fit_total is
  nan if cleaning skipped the best fit, preliminaries.py --no-nnls);
- rate_vectors.csv and .npz: per rupture of the original problem (penalised ruptures flagged and 0), the posterior
  mean, median, mode and 2.275%, 15.865%, 84.135% and 97.725% values.
Prints the bins' totals against their targets, the undetectable shares and the files written, and writes the same to
REPORT (default: tables_report.txt in DIR).
"""
import argparse
import csv
import os

import numpy as np

from cru_common import load_problem, load_stream, stream_quantiles, to_original, prelim_path, report_to

RATE_QS = (0.02275, 0.15865, 0.5, 0.84135, 0.97725)    # the -2, -1, 0, +1, +2 sigma points of a Gaussian


def has_draws(name):
    return os.path.exists(name + '_samples.npy')


def load_run(name):
    return dict(np.load(name + '.npz')), np.load(name + '_samples.npy', mmap_mode='r')


def hist_quantiles(h, lo, w, qs):
    """Quantiles per column of a log-rate histogram (bin 0: rates below 10^lo, reported as 0), interpolated within bins."""
    h = h.astype(np.float64)
    tot = h.sum(0)
    c = np.cumsum(h, 0) / tot
    cols = np.arange(h.shape[1])
    out = []
    for q in qs:
        k = np.argmax(c >= q, axis=0)
        prev = np.where(k > 0, c[np.maximum(k - 1, 0), cols], 0.0)
        frac = np.clip((q - prev) / np.maximum(h[k, cols] / tot, 1e-300), 0, 1)
        out.append(np.where(k == 0, 0.0, 10.0 ** (lo + (k - 1 + frac) * w)))
    return out


def hist_mode(h, lo, w, smooth=0.05):
    """Mode of each rate: 0 if a rate below 10^lo (including exactly 0) has probability at least 0.5; otherwise the peak
    of the log-rate density above 10^lo, smoothed by a Gaussian of sd `smooth` decades."""
    from scipy.ndimage import convolve1d
    H = h[1:].astype(np.float64)
    sdb = smooth / w
    k = np.arange(-int(np.ceil(4 * sdb)), int(np.ceil(4 * sdb)) + 1)
    ker = np.exp(-0.5 * (k / sdb) ** 2)
    j = convolve1d(H, ker / ker.sum(), axis=0, mode='constant').argmax(0)
    return np.where(h[0] >= 0.5 * h.sum(0), 0.0, 10.0 ** (lo + (j + 0.5) * w))


def run_hist(r, w=0.01, lo=-16.0):
    """Log-rate histogram, rate sum and number of draws of one run: from its running summaries, or, for runs made before
    those existed, built from its stored draws with the same binning as gibbs.py (bin width w decades)."""
    if os.path.exists(r + '_stream.npz'):
        st = load_stream(r, ('hist', 'msum', 'chains', 'n_keep', 'hist_lo', 'hist_w'))
        return st['hist'], st['msum'], int(st['chains']) * int(st['n_keep']), float(st['hist_lo']), float(st['hist_w'])
    S = np.load(r + '_samples.npy', mmap_mode='r')
    K, D, n = S.shape
    nb = 1 + int(round(16.5 / w))
    hist, msum = np.zeros((nb, n), np.int32), np.zeros(n)
    step = max(1, int(2e7 // (K * D)))
    for j0 in range(0, n, step):
        X = np.asarray(S[:, :, j0:j0 + step], dtype=np.float64).reshape(K * D, -1)
        c = X.shape[1]
        msum[j0:j0 + c] = X.sum(0)
        idx = np.clip(np.floor((np.log10(np.maximum(X, 1e-300)) - lo) / w).astype(np.int64) + 1, 0, nb - 1)
        hist[:, j0:j0 + c] = np.bincount((idx * c + np.arange(c)).ravel(), minlength=nb * c).reshape(nb, c)
    return hist, msum, K * D, lo, w


def rate_vectors(runs, prob, chunk=20000):
    """Posterior mean, median, mode and 2.275/15.865/84.135/97.725% values of each rate, pooled over runs, from the
    running summaries (or stored draws, for older runs); at the original rupture positions, penalised ruptures at 0."""
    w0 = next((float(np.load(r + '_stream.npz')['hist_w']) for r in runs if os.path.exists(r + '_stream.npz')), 0.01)
    H, msum, nd, lo, w = None, 0.0, 0, None, None
    for r in runs:
        h, m, nd_r, lo_r, w_r = run_hist(r, w=w0)
        if w is not None and (w_r, lo_r) != (w, lo):
            raise ValueError('these runs have running histograms of different bin widths; pool runs made alike')
        H = h.astype(np.int32) if H is None else H + h
        msum, nd, lo, w = msum + m, nd + nd_r, lo_r, w_r
        del h
    n = H.shape[1]
    keys = ('p02.275', 'p15.865', 'median', 'p84.135', 'p97.725')
    out = {k: np.empty(n) for k in keys + ('mode',)}
    for c0 in range(0, n, chunk):
        hc = H[:, c0:c0 + chunk]
        for k, v in zip(keys, hist_quantiles(hc, lo, w, RATE_QS)):
            out[k][c0:c0 + chunk] = v
        out['mode'][c0:c0 + chunk] = hist_mode(hc, lo, w)
    out['mean'] = msum / nd
    return {k: to_original(v, prob) for k, v in out.items()}


def write_rate_vectors(runs, prob, path):
    """Write the rate vectors to <path>.csv and <path>.npz; returns the csv path."""
    v = rate_vectors(runs, prob)
    pen = np.ones(len(v['mean']), int)
    pen[prob['keep_cols']] = 0
    cols = ('mean', 'median', 'mode', 'p02.275', 'p15.865', 'p84.135', 'p97.725')
    with open(path + '.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['rupture', 'penalised'] + list(cols))
        for i in range(len(pen)):
            wr.writerow([i, pen[i]] + [f'{v[c][i]:.6e}' for c in cols])
    np.savez(path + '.npz', penalised=pen, **{c.replace('.', '_'): v[c] for c in cols})
    return path + '.csv'


def rupture_table(runs, thresholds=(1e-5, 1e-4)):
    if not all(has_draws(r) for r in runs):      # running summaries: exceedance counts and the log-rate histogram
        sts = [load_stream(r, ('chains', 'n_keep', 'exc5', 'exc4', 'hist', 'hist_lo', 'hist_w')) for r in runs]
        ntot = sum(int(s_['chains']) * int(s_['n_keep']) for s_ in sts)
        exc = {1e-5: sum(s_['exc5'] for s_ in sts) / ntot, 1e-4: sum(s_['exc4'] for s_ in sts) / ntot}
        h = dict(sts[0])
        h['hist'] = sts[0]['hist'].astype(np.int64)
        for s_ in sts[1:]:
            h['hist'] += s_['hist']
        return {t: exc[t] for t in thresholds}, np.array(stream_quantiles(h, (0.05, 0.5, 0.95)))
    Ss = [load_run(r)[1] for r in runs]
    nn = Ss[0].shape[2]
    out = {t: np.empty(nn) for t in thresholds}
    q = np.empty((3, nn))
    for j0 in range(0, nn, 2048):
        X = np.concatenate([np.asarray(S[:, :, j0:j0 + 2048]).reshape(-1, min(2048, nn - j0)) for S in Ss])
        j1 = j0 + X.shape[1]
        for t in thresholds:
            out[t][j0:j1] = (X > t).mean(0)
        q[:, j0:j1] = np.quantile(X, [0.05, 0.5, 0.95], axis=0)
    return out, q


def mean_parts(runs, det):
    """Posterior means of each rate, and of the part of it below its own detection scale, pooled over runs and chains."""
    if not all(has_draws(r) for r in runs):
        sts = [load_stream(r, ('chains', 'n_keep', 'msum', 'usum')) for r in runs]
        nd = sum(int(s_['chains']) * int(s_['n_keep']) for s_ in sts)
        return sum(s_['msum'] for s_ in sts) / nd, sum(s_['usum'] for s_ in sts) / nd
    tot, und, nd = np.zeros(len(det)), np.zeros(len(det)), 0
    for S in (load_run(r)[1] for r in runs):
        for k in range(S.shape[0]):
            for t0 in range(0, S.shape[1], 1000):
                X = np.asarray(S[k, t0:t0 + 1000], dtype=np.float64)
                tot += X.sum(0)
                und += np.where(X < det, X, 0.0).sum(0)
                nd += X.shape[0]
    return tot / nd, und / nd


def undetectable_share(parts, inc):
    """Share of each posterior-mean total (subsection or bin) carried by rates below their own detection scale."""
    T, U = inc @ parts[0], inc @ parts[1]
    return np.where(T > 0, U / np.where(T > 0, T, 1.0), np.nan)


def sum_draws(runs, inc, which):
    """Draws of inc @ m (one row per subsection or bin), pooled over runs and chains."""
    if not all(has_draws(r) for r in runs):      # per-draw totals recorded during the run ('s': subsections, 'b': bins)
        return np.concatenate([load_stream(r, ('tot_' + which,))['tot_' + which].reshape(-1, inc.shape[0]).astype(np.float64) for r in runs])
    return np.concatenate([(inc @ np.asarray(S[k], dtype=np.float64).T).T
                           for S in (load_run(r)[1] for r in runs) for k in range(S.shape[0])])


def main():
    ap = argparse.ArgumentParser(description='Posterior tables and rate vectors of a result set (see the module docstring).')
    ap.add_argument('runs', nargs='+', metavar='RUN', help='the result set\'s replicate runs (gibbs.py --out)')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem the runs sampled')
    ap.add_argument('--prelim', default=None, help='preliminaries.py\'s diagnostics, for the best fit (default: next to --problem)')
    ap.add_argument('--convergence', default=None, help='check_convergence.py\'s per-rupture R-hat and effective draws '
                    '(default: convergence.npz in --outdir)')
    ap.add_argument('--outdir', default=None, help='where the tables go (default: the first run\'s folder)')
    ap.add_argument('--slip-block', default='S', help='the slip-rate rows (default S)')
    ap.add_argument('--bins-block', default='MFD', help='the magnitude-bin rows (default MFD)')
    ap.add_argument('--out', default=None, help='the report (default: tables_report.txt in --outdir)')
    a = ap.parse_args()
    MAIN = a.runs
    OUT = a.outdir if a.outdir is not None else os.path.dirname(MAIN[0])
    report_to(a.out or os.path.join(OUT, 'tables_report.txt'))
    prob = load_problem(a.problem)
    G, d = prob['G'].tocsr(), prob['d']
    n = G.shape[1]
    blk, bnames = prob['block'], prob['block_names']
    det = prob['sigma'] / np.sqrt(np.asarray(G.multiply(G).sum(0)).ravel())   # single-rupture detection scale sigma/||G_i||

    srows = np.flatnonzero(blk == bnames.index(a.slip_block))
    brows = np.flatnonzero(blk == bnames.index(a.bins_block))
    inc_s = (G[srows] > 0).astype(float)
    inc_b = (G[brows] > 0).astype(float)
    pm, qm = rupture_table(MAIN)
    sub_m, bin_m = sum_draws(MAIN, inc_s, 's'), sum_draws(MAIN, inc_b, 'b')
    targets = np.array([d[j] / G.data[G.indptr[j]] for j in brows])
    ratio = np.median(bin_m, 0) / targets
    with np.load(a.prelim or prelim_path(a.problem)) as prel:
        m_best = prel['m_nnls'] if 'm_nnls' in prel.files else None   # best nonnegative fit: no prior, the data's own tension
    if m_best is not None:
        best_b, best_s = np.asarray(inc_b @ m_best).ravel(), np.asarray(inc_s @ m_best).ravel()
        rb = best_b / targets
        best_txt = f'; best fit {np.median(rb):.3f} (range {rb.min():.3f} to {rb.max():.3f}), the reference with no prior'
    else:                                                # cleaning ran without the best fit (--no-nnls): its columns are nan
        best_b, best_s = np.full(len(brows), np.nan), np.full(len(srows), np.nan)
        best_txt = '; no best fit (cleaning ran with --no-nnls)'
    print(f'bin totals / targets: posterior median {np.median(ratio):.3f} (range {ratio.min():.3f} to {ratio.max():.3f})' + best_txt)
    parts_m = mean_parts(MAIN, det)
    und_s, und_b = undetectable_share(parts_m, inc_s), undetectable_share(parts_m, inc_b)
    print(f'share of each total carried by individually undetectable rates (below sigma/||G_i||): bins median '
          f'{100 * np.nanmedian(und_b):.0f}% (range {100 * np.nanmin(und_b):.0f}% to {100 * np.nanmax(und_b):.0f}%); '
          f'subsections median {100 * np.nanmedian(und_s):.0f}%')

    keep_c, keep_r = prob['keep_cols'], prob['keep_rows']
    bin_of = np.asarray(inc_b.argmax(0)).ravel()
    fconv = a.convergence or os.path.join(OUT, 'convergence.npz')
    cz = np.load(fconv) if os.path.exists(fconv) else None
    if cz is not None and list(cz['runs']) == list(MAIN):   # mixing of each rupture's log rate: worst R-hat over the replicates, draws summed
        rh_m = np.fmax.reduce(cz['rhat'], axis=0)
        es_m = np.sum(cz['ess'], axis=0)
        slow_at = float(cz['slow_at'].max())
    else:
        rh_m = es_m = np.full(n, np.nan)
        slow_at = 1.01
        print("note: run check_convergence.py on these runs first (its convergence.npz) to get each rupture's R-hat and "
              'effective draws in the table')
    with open(os.path.join(OUT, 'ruptures_posterior.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rupture', 'bin_row', 'p_rate_gt_1e-5', 'p_rate_gt_1e-4']
                   + ['rate_q05_prior_dependent', 'rate_median_prior_dependent', 'rate_q95', 'rhat_log_rate', 'ess_log_rate', 'slow_mixing'])
        for i in range(n):
            w.writerow([int(keep_c[i]), int(keep_r[brows[bin_of[i]]]), f'{pm[1e-5][i]:.4f}', f'{pm[1e-4][i]:.4f}']
                       + [f'{qm[0, i]:.3e}', f'{qm[1, i]:.3e}', f'{qm[2, i]:.3e}', f'{rh_m[i]:.4f}', f'{es_m[i]:.0f}', int(rh_m[i] > slow_at)])
    for fname, rows_, draws_m, obs, best, und_m in [
            ('subsections_posterior.csv', srows, sub_m, None, best_s, und_s),
            ('bins_posterior.csv', brows, bin_m, targets, best_b, und_b)]:
        with open(os.path.join(OUT, fname), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['row', 'target_total' if obs is not None else 'd', 'total_q05', 'total_median', 'total_q95', 'best_fit_total',
                        'undetectable_share'])
            q = np.quantile(draws_m, [0.05, 0.5, 0.95], axis=0)
            for k, j in enumerate(rows_):
                w.writerow([int(keep_r[j]), f'{(obs[k] if obs is not None else d[j]):.5g}', f'{q[0, k]:.4e}', f'{q[1, k]:.4e}',
                            f'{q[2, k]:.4e}', f'{best[k]:.4e}', f'{und_m[k]:.3f}'])
    print(f'wrote {OUT}/ruptures_posterior.csv, {OUT}/subsections_posterior.csv, {OUT}/bins_posterior.csv')
    if all(os.path.exists(r + '_stream.npz') or os.path.exists(r + '_samples.npy') for r in MAIN):
        print('wrote', write_rate_vectors(MAIN, prob, os.path.join(OUT, 'rate_vectors')), '(and .npz), pooled over the replicates')
    else:
        print('rate vectors skipped: they need running summaries (<run>_stream.npz) or stored draws')


if __name__ == '__main__':
    main()

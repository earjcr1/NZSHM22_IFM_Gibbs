#!/usr/bin/env python3
"""Representative draws for hazard: a few weighted whole posterior draws, chosen from the runs' kept draws.

    python select_draws.py RUN [RUN ...] [--problem cru_clean.npz] [--vectors DIR/rate_vectors] [--k 20]
                           [--eps 1e-6] [--mu 1] [--seed 0] [--slip-block S] [--bins-block MFD]

Each RUN (a replicate) needs <RUN>_kept.npy and <RUN>_kept_index.npy (gibbs.py --keep-draws), <RUN>.npz (its traces)
and <RUN>_stream.npz (its running summaries). The rate vectors (<vectors>.csv and .npz, written by the notebook) give
the exact posterior mean of every rupture's rate; by default they are <folder of the first run>/rate_vectors.

1. Pool the replicates' kept draws (the original rupture numbering is restored on output, penalised ruptures 0).
2. Features: each subsection's participation rate in each magnitude bin, P_sb = the sum of m_i over the ruptures in
   bin b that cross subsection s (the slip-rate rows say which subsections a rupture crosses, the bin rows its bin);
   features that are zero in every draw are dropped.
3. Transform: log10(P + eps), centred; the principal components covering 95% of the variance, at most 50.
4. Cluster: k-means on the component scores (k-means++ starts, 10 restarts, fixed seed). Each representative is the
   kept draw nearest its cluster centre; its initial weight is its cluster's share of the draws.
5. Calibrate the weights: stay close to the cluster shares while bringing the weighted mean participation rates to
   the exact posterior means (SLSQP; weights >= 0 summing to 1; mu weighs the pull towards the cluster shares).
6. Select K = 10, 20 and 40 (and --k if different); write --k: columns rep_01... appended to <vectors>.csv (a rerun
   replaces them), representatives.csv, kept / kept_index / rep / rep_weight added to <vectors>.npz, and
   select_report.txt with the checks for every size.

The weights are quadrature weights for the posterior as a whole, not probabilities of individual draws; the rate
vectors' mean column remains the exact source for the mean hazard.
"""
import argparse
import csv
import os

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from cru_common import load_problem, load_stream

SIZES = (10, 20, 40)
TOL_TOTAL, TOL_SUB, TOL_Q, SHARE = 0.02, 0.10, 0.15, 0.90      # proposed tolerances (specification; to revisit)
QS = (0.05, 0.16, 0.5, 0.84, 0.95)                           # 16/50/84% judged; 5/95% shown only


def kmeans(Z, k, seed, restarts=10, iters=300):
    """Lloyd's k-means with k-means++ starts; the best of `restarts` by within-cluster sum of squares."""
    g = np.random.default_rng(seed)
    n = len(Z)
    zz = (Z ** 2).sum(1)
    dist = lambda C: np.maximum(zz[:, None] - 2.0 * Z @ C.T + (C ** 2).sum(1)[None, :], 0.0)
    best = None
    for _ in range(restarts):
        C = np.empty((k, Z.shape[1]))
        C[0] = Z[g.integers(n)]
        d2 = ((Z - C[0]) ** 2).sum(1)
        for j in range(1, k):
            C[j] = Z[g.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else Z[g.integers(n)]
            d2 = np.minimum(d2, ((Z - C[j]) ** 2).sum(1))
        lab = None
        for _ in range(iters):
            new = dist(C).argmin(1)
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            for j in range(k):
                if (lab == j).any():
                    C[j] = Z[lab == j].mean(0)
        D = dist(C)
        lab = D.argmin(1)
        for j in range(k):                              # an empty cluster takes the worst-served point
            if not (lab == j).any():
                far = D[np.arange(n), lab].argmax()
                lab[far] = j
                C[j] = Z[far]
        for j in range(k):
            C[j] = Z[lab == j].mean(0)
        inertia = dist(C)[np.arange(n), lab].sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab.copy(), C.copy())
    return best[1], best[2]


def calibrate(A, w0, mu):
    """min ||A w - 1||^2 + mu sum_k (w_k - w0_k)^2 / w0_k  subject to w >= 0, sum w = 1 (SLSQP, from w0)."""
    k = len(w0)
    f = lambda w: float(np.sum((A @ w - 1.0) ** 2) + mu * np.sum((w - w0) ** 2 / w0))
    jac = lambda w: 2.0 * A.T @ (A @ w - 1.0) + 2.0 * mu * (w - w0) / w0
    c = max(f(w0), 1e-300)                              # the objective scaled to 1 at the start: same minimum, better steps
    res = minimize(lambda w: f(w) / c, w0, jac=lambda w: jac(w) / c, method='SLSQP', bounds=[(0.0, 1.0)] * k,
                   options=dict(maxiter=2000, ftol=1e-12),
                   constraints=[dict(type='eq', fun=lambda w: w.sum() - 1.0, jac=lambda w: np.ones(k))])
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return (w, f(w0), f(w), res) if f(w) <= f(w0) else (w0.copy(), f(w0), f(w0), res)


def wquantile(vals, w, qs):
    """Quantiles of the weighted discrete distribution of each column of vals (k, m): the smallest value whose
    cumulative weight reaches q."""
    o = np.argsort(vals, axis=0, kind='stable')
    v = np.take_along_axis(vals, o, 0)
    cw = np.cumsum(w[o], axis=0)
    return np.array([v[np.minimum((cw < q - 1e-12).sum(0), len(w) - 1), np.arange(vals.shape[1])] for q in qs])


def rel(a, b):
    """|a - b| / |b|; 0 when both are 0, infinite when only b is."""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.abs(a - b) / np.abs(b)
    return np.where(b == 0, np.where(a == 0, 0.0, np.inf), r)


def main():
    ap = argparse.ArgumentParser(description='Representative draws for hazard (see the module docstring).')
    ap.add_argument('runs', nargs='+', help='the replicate runs, each with <run>_kept.npy from gibbs.py --keep-draws')
    ap.add_argument('--problem', default='cru_clean.npz')
    ap.add_argument('--vectors', default=None, help='rate vectors, path without extension (default: <folder of the first '
                    'run>/rate_vectors)')
    ap.add_argument('--k', type=int, default=20, help='number of representatives written (default 20)')
    ap.add_argument('--eps', type=float, default=1e-6, help='offset per year in log10(P + eps) (default 1e-6)')
    ap.add_argument('--mu', type=float, default=1.0, help='weight of the pull towards the cluster shares (default 1)')
    ap.add_argument('--seed', type=int, default=0, help='k-means seed (default 0)')
    ap.add_argument('--slip-block', default='S')
    ap.add_argument('--bins-block', default='MFD')
    a = ap.parse_args()
    vec = a.vectors or os.path.join(os.path.dirname(a.runs[0]), 'rate_vectors')
    out_dir = os.path.dirname(vec) or '.'
    sizes = sorted(set(SIZES) | {a.k})

    # the problem: subsections (slip-rate rows) and bins (magnitude-bin rows)
    P = load_problem(a.problem)
    Gr = P['G'].tocsr()
    n = Gr.shape[1]
    keep_c = np.asarray(P['keep_cols'])
    n_orig = P['raw_shape'][1] if P.get('raw_shape') else int(keep_c.max()) + 1
    rows_of = lambda blk: np.flatnonzero(P['block'] == P['block_names'].index(blk))
    inc_s = (Gr[rows_of(a.slip_block)] > 0).astype(np.float64).tocsr()
    inc_b = (Gr[rows_of(a.bins_block)] > 0).astype(np.float64).tocsr()
    nS, B = inc_s.shape[0], inc_b.shape[0]
    bin_of = np.asarray(inc_b.argmax(0)).ravel()
    co = inc_s.tocoo()                                   # participation features: one row per (subsection, bin) pair
    F = sp.csr_matrix((np.ones(co.nnz), (co.row * B + bin_of[co.col], co.col)), shape=(nS * B, n))
    F = F[np.flatnonzero(np.diff(F.indptr) > 0)]

    # exact posterior means (the rate vectors' mean column) and the pooled kept draws
    rv = dict(np.load(vec + '.npz'))
    mean = np.asarray(rv['mean'], np.float64)[keep_c]
    Pbar = F @ mean
    X, src, counts = [], [], []
    for r_i, run in enumerate(a.runs):
        if not os.path.exists(run + '_kept.npy'):
            raise SystemExit(f'{run}_kept.npy not found: run gibbs.py with --keep-draws (the notebook: REPRESENTATIVES > 0)')
        X.append(np.load(run + '_kept.npy'))
        idx = np.load(run + '_kept_index.npy')
        src.append(np.column_stack([np.full(len(idx), r_i), idx]))
        counts.append(len(idx))
    X, src = np.concatenate(X), np.concatenate(src)
    N = len(X)
    Pk = np.empty((N, F.shape[0]))
    for c0 in range(0, N, 256):
        Pk[c0:c0 + 256] = (F @ X[c0:c0 + 256].astype(np.float64).T).T
    live = (Pk > 0).any(0)
    Z = np.log10(Pk[:, live] + a.eps)
    Z -= Z.mean(0)
    ev, U = np.linalg.eigh(Z @ Z.T)                      # principal components through the draws' Gram matrix
    ev, U = np.clip(ev[::-1], 0.0, None), U[:, ::-1]
    frac = np.cumsum(ev) / ev.sum()
    nc = int(min(50, np.searchsorted(frac, 0.95) + 1))
    scores = U[:, :nc] * np.sqrt(ev[:nc])
    del Z

    # the posterior to check against: traces (chi2, total rate) and per-draw totals over every recorded draw
    chi2_all, tot_all, tr_chi2, tr_tot, post_s, post_b = [], [], [], [], [], []
    for run in a.runs:
        z = np.load(run + '.npz')
        nb_ = int(z['burn']) // int(z['thin'])
        tr_chi2.append(z['chi2'])
        tr_tot.append(z['sum_m'])
        chi2_all.append(z['chi2'][:, nb_:].ravel())
        tot_all.append(z['sum_m'][:, nb_:].ravel())
        st = load_stream(run, ('tot_s', 'tot_b'))
        post_s.append(st['tot_s'].reshape(-1, nS).astype(np.float64))
        post_b.append(st['tot_b'].reshape(-1, B).astype(np.float64))
    chi2_rng = np.percentile(np.concatenate(chi2_all), [1, 99])
    tot_rng = np.percentile(np.concatenate(tot_all), [1, 99])
    qpost_s, qpost_b = np.quantile(np.concatenate(post_s), QS, axis=0), np.quantile(np.concatenate(post_b), QS, axis=0)
    exact_s, exact_b, exact_t = inc_s @ mean, inc_b @ mean, mean.sum()
    cal = Pbar > a.eps

    res = {}
    for K in sizes:
        if K > N:
            print(f'K = {K}: skipped, only {N} kept draws')
            continue
        lab, C = kmeans(scores, K, a.seed)
        rep = np.array([np.flatnonzero(lab == j)[((scores[lab == j] - C[j]) ** 2).sum(1).argmin()] for j in range(K)])
        w0 = np.bincount(lab, minlength=K) / N
        A = Pk[rep][:, cal].T / Pbar[cal][:, None]
        w, f0, f1, opt = calibrate(A, w0, a.mu)
        R = X[rep].astype(np.float64)
        rs, rb, rt = (inc_s @ R.T).T, (inc_b @ R.T).T, R.sum(1)
        e_t = (w @ rt - exact_t) / exact_t
        e_b = (w @ rb - exact_b) / exact_b
        pos = exact_s > 0
        in_sub = np.mean(rel(w @ rs, exact_s)[pos] <= TOL_SUB)
        q_s, q_b = wquantile(rs, w, QS), wquantile(rb, w, QS)
        qin_s = [np.mean(rel(q_s[i], qpost_s[i]) <= TOL_Q) for i in range(len(QS))]
        qin_b = [np.mean(rel(q_b[i], qpost_b[i]) <= TOL_Q) for i in range(len(QS))]
        rc = np.array([tr_chi2[s_[0]][s_[1], s_[2]] for s_ in src[rep]])
        rtt = np.array([tr_tot[s_[0]][s_[1], s_[2]] for s_ in src[rep]])
        ratio = np.where(w > 1e-9, np.maximum(w / w0, w0 / np.maximum(w, 1e-300)), np.inf)
        ok = dict(mean_total=abs(e_t) <= TOL_TOTAL and np.all(np.abs(e_b) <= TOL_TOTAL), mean_sub=in_sub >= SHARE,
                  quant=all(qin_s[i] >= SHARE and qin_b[i] >= SHARE for i in (1, 2, 3)),
                  range=bool(np.all((rc >= chi2_rng[0]) & (rc <= chi2_rng[1]) & (rtt >= tot_rng[0]) & (rtt <= tot_rng[1]))),
                  weights=bool(np.all(w > 1e-9)))
        res[K] = dict(rep=rep, w=w, w0=w0, e_t=e_t, e_b=e_b, in_sub=in_sub, n_pos=int(pos.sum()), qin_s=qin_s, qin_b=qin_b,
                      rc=rc, rtt=rtt, ratio=ratio, f0=f0, f1=f1, opt=opt, ok=ok)

    # report
    pc = lambda x: f'{100 * x:.0f}%'
    sp_ = lambda x: f'{100 * x:+.1f}%'
    Ks = list(res)
    L = [f'Representative draws from {N} kept draws pooled over {len(a.runs)} runs '
         f'({", ".join(f"{r}: {c}" for r, c in zip(a.runs, counts))})',
         f'features: {int(live.sum())} of {F.shape[0]} subsection-bin participation rates nonzero in some draw; log10(P + '
         f'{a.eps:g}); {nc} principal components cover {100 * frac[nc - 1]:.1f}% of the variance; k-means seed {a.seed}, '
         f'mu {a.mu:g}; weights calibrated on the {int(cal.sum())} features whose posterior mean exceeds {a.eps:g}',
         f'posterior 1-99% ranges over every recorded draw: chi2 {chi2_rng[0]:.1f} to {chi2_rng[1]:.1f}, total rate '
         f'{tot_rng[0]:.5f} to {tot_rng[1]:.5f}', '',
         f'{"check (tolerance)":58s}' + ''.join(f'{"K = " + str(K):>12s}' for K in Ks)]
    row = lambda name, vals: L.append(f'{name:58s}' + ''.join(f'{v:>12s}' for v in vals))
    row('weighted mean total rate vs exact mean (2%)', [sp_(res[K]['e_t']) for K in Ks])
    row('weighted mean bin totals, worst bin (2%)', [sp_(res[K]['e_b'][np.argmax(np.abs(res[K]['e_b']))]) for K in Ks])
    row('weighted mean subsection totals within 10% (90%)', [pc(res[K]['in_sub']) for K in Ks])
    for i, q in enumerate(QS):
        judged = i in (1, 2, 3)
        row(f'{100 * q:g}% point within 15%: subsections / bins' + (' (90%)' if judged else ' (not judged)'),
            [f'{pc(res[K]["qin_s"][i])} / {pc(res[K]["qin_b"][i])}' for K in Ks])
    row('representatives inside the 1-99% range: chi2 (all)',
        [f'{int(np.sum((res[K]["rc"] >= chi2_rng[0]) & (res[K]["rc"] <= chi2_rng[1])))}/{K}' for K in Ks])
    row('representatives inside the 1-99% range: total rate (all)',
        [f'{int(np.sum((res[K]["rtt"] >= tot_rng[0]) & (res[K]["rtt"] <= tot_rng[1])))}/{K}' for K in Ks])
    row('weights: smallest (none zero)', [f'{res[K]["w"].min():.4f}' for K in Ks])
    row('weights: largest ratio to the cluster share', [f'{res[K]["ratio"].max():.2f}' for K in Ks])
    row('calibration objective at the cluster shares', [f'{res[K]["f0"]:.4g}' for K in Ks])
    row('calibration objective after calibration', [f'{res[K]["f1"]:.4g}' for K in Ks])
    for K in Ks:
        if not res[K]['opt'].success:
            L.append(f'note: K = {K}: the weight optimizer stopped early ({res[K]["opt"].message})')
    names = dict(mean_total='mean total and bins', mean_sub='mean subsections', quant='16/50/84% points',
                 range='chi2 and total in range', weights='no zero weight')
    L += ['', 'verdicts:']
    for K in Ks:
        miss = [names[c] for c, v in res[K]['ok'].items() if not v]
        L.append(f'  K = {K}: ' + ('meets every tolerance' if not miss else 'misses ' + ', '.join(miss)))
    if a.k in res:
        if not any(not v for v in res[a.k]['ok'].values()):
            L.append(f'K = {a.k} meets the tolerances.')
        elif 40 in res and all(res[40]['ok'].values()):
            L.append(f'K = {a.k} misses tolerances that K = 40 meets: use K = 40 (--k 40, or REPRESENTATIVES = 40).')
        else:
            L.append(f'K = {a.k} misses some tolerances, and so does K = 40: expected at the quick preset; on longer runs, '
                     f'the tolerances or the selection need revisiting.')
    report = '\n'.join(L)
    print(report)
    with open(os.path.join(out_dir, 'select_report.txt'), 'w') as f:
        f.write(report + '\n')

    # outputs for --k, in the original rupture numbering (penalised ruptures 0)
    if a.k not in res:
        raise SystemExit(f'no representatives written: K = {a.k} needs at least {a.k} kept draws')
    r = res[a.k]
    reps = np.zeros((a.k, n_orig), np.float32)
    reps[:, keep_c] = X[r['rep']]
    raw = open(vec + '.csv', newline='').read()
    term = '\r\n' if '\r\n' in raw else '\n'
    lines = raw.split(term)
    if lines[-1] == '':
        lines = lines[:-1]
    head = lines[0].split(',')
    cut = head.index('rep_01') if 'rep_01' in head else len(head)      # a rerun replaces earlier representatives
    if len(lines) - 1 != n_orig:
        raise SystemExit(f'{vec}.csv has {len(lines) - 1} rows, not {n_orig}')
    cols = [f'rep_{j + 1:02d}' for j in range(a.k)]
    out = [','.join(head[:cut] + cols)]
    for i, line in enumerate(lines[1:]):
        out.append(','.join(line.split(',')[:cut] + [f'{v:.6e}' for v in reps[:, i]]))
    with open(vec + '.csv', 'w', newline='') as f:
        f.write(term.join(out) + term)
    with open(os.path.join(out_dir, 'representatives.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['rep', 'weight', 'cluster_share', 'run', 'chain', 'record', 'chi2', 'total_rate'])
        for j in range(a.k):
            s_ = src[r['rep'][j]]
            wr.writerow([cols[j], f'{r["w"][j]:.6f}', f'{r["w0"][j]:.6f}', a.runs[s_[0]], int(s_[1]), int(s_[2]),
                         f'{r["rc"][j]:.2f}', f'{r["rtt"][j]:.6e}'])
    kept = np.zeros((N, n_orig), np.float32)
    kept[:, keep_c] = X
    rv.update(kept=kept, kept_index=src, rep=reps, rep_weight=r['w'])
    np.savez(vec + '.npz', **rv)
    print(f'wrote {vec}.csv (columns {cols[0]} to {cols[-1]}), {os.path.join(out_dir, "representatives.csv")}, '
          f'{os.path.join(out_dir, "select_report.txt")}; added kept, kept_index, rep and rep_weight to {vec}.npz')


if __name__ == '__main__':
    main()

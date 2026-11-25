#!/usr/bin/env python3
"""Representative draws for hazard: a few whole posterior draws, chosen from the runs' kept draws.

    python select_draws.py RUN [RUN ...] [--problem cru_clean.npz] [--vectors DIR/rate_vectors] [--k 20]
                           [--pick spaced|kmeans] [--weights equal|calibrated] [--eps 1e-6] [--mu 1] [--seed 0]
                           [--slip-block S] [--bins-block MFD]

Since 2026-11-25 (the user's choice, after compare_reps.py and a sweep over K and mu on the 480-minute run): by
default the representatives are K kept draws evenly spaced over each chain's records (K shared among the chains and
replicates in proportion to their kept draws), each weighted 1/K: an equally weighted sample of the posterior. The
calibrated k-means weights (--pick kmeans --weights calibrated, the default before) matched the means no better and
left the distributions further from the kept draws' than random draws do; k-means picks alone did no better than
random ones. Steps 2-5 below are the k-means and calibrated options.

Each RUN (a replicate) needs <RUN>_kept.npz (sparse; or an older run's dense <RUN>_kept.npy) and <RUN>_kept_index.npy
(gibbs.py --keep-draws), <RUN>.npz (its traces)
and <RUN>_stream.npz (its running summaries). The rate vectors (<vectors>.csv and .npz, written by the notebook) give
the exact posterior mean of every rupture's rate; by default they are <folder of the first run>/rate_vectors.

1. Pool the replicates' kept draws (the original rupture numbering is restored on output, penalised ruptures 0).
2. Features: each subsection's participation rate in each magnitude bin, P_sb = the sum of m_i over the ruptures in
   bin b that cross subsection s (the slip-rate rows say which subsections a rupture crosses, the bin rows its bin);
   features that are zero in every draw are dropped.
3. Transform: log10(P + eps), centred; the principal components covering 95% of the variance, at most 50 (exact up
   to 8,000 kept draws, by a randomized SVD beyond: every recorded draw can be kept since 2026-11-25).
4. Cluster: k-means on the component scores (k-means++ starts, 10 restarts, fixed seed). Each representative is the
   kept draw nearest its cluster centre; its initial weight is its cluster's share of the draws.
5. Calibrate the weights: stay close to the cluster shares while bringing the weighted mean participation rates to
   the exact posterior means (SLSQP; weights >= 0 summing to 1; mu weighs the pull towards the cluster shares).
6. Select K = 10, 20 and 40 (and --k if different); write --k: columns rep_01... appended to <vectors>.csv (a rerun
   replaces them), representatives.csv, the kept draws (sparse: kept_data, kept_indices, kept_indptr, kept_shape;
   a dense `kept` before 2026-11-25), kept_index, rep and rep_weight added to <vectors>.npz, and
   select_report.txt with the checks for every size.

With --weights calibrated the weights are quadrature weights for the posterior as a whole, not probabilities of
individual draws. Either way the rate vectors' mean column remains the exact source for the mean hazard. The report
gives, for --k, the distribution check of compare_reps.py (KS against the kept draws, as a percentile among random
equally weighted sets of K) beside the mean checks.
"""
import argparse
import csv
import os

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from cru_common import as_dense, fmt_rate, kept_arrays, load_kept, load_problem, load_stream

SIZES = (10, 20, 40)
TOL_TOTAL, TOL_SUB, TOL_Q, SHARE = 0.02, 0.10, 0.15, 0.90      # proposed tolerances (specification; to revisit)
QS = (0.05, 0.16, 0.5, 0.84, 0.95)                           # 16/50/84% judged; 5/95% shown only


EXACT_PCA_MAX = 8000          # up to this many kept draws, the exact components (as before 2026-11-25); beyond, randomized


def randomized_pcs(Z, k, seed, power=4, over=20):
    """The leading k left singular vectors and singular values of Z (draws x features, centred), by a randomized SVD
    (Halko, Martinsson and Tropp, 2011: a Gaussian sketch, `power` subspace iterations, re-orthonormalized each time)."""
    g = np.random.default_rng(seed)
    Y = Z @ g.standard_normal((Z.shape[1], k + over)).astype(Z.dtype)
    Q, _ = np.linalg.qr(Y)
    for _ in range(power):
        Q, _ = np.linalg.qr(Z.T @ Q)
        Q, _ = np.linalg.qr(Z @ Q)
    Ub, sv, _ = np.linalg.svd(Q.T @ Z, full_matrices=False)
    return (Q @ Ub)[:, :k], sv[:k]


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


def spaced_picks(src, K):
    """K kept draws evenly spaced over each chain's records: K shared among the (run, chain) groups in proportion to
    their kept draws (largest remainders), each group's evenly spaced over its draws in the order recorded."""
    grp, inv = np.unique(src[:, :2], axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_g = np.bincount(inv, minlength=len(grp))
    q = K * n_g / n_g.sum()
    k_g = np.floor(q).astype(int)
    for j in np.argsort(-(q - k_g), kind='stable')[:K - k_g.sum()]:
        k_g[j] += 1
    out = []
    for g_ in range(len(grp)):
        members = np.flatnonzero(inv == g_)
        members = members[np.argsort(src[members, 2], kind='stable')]
        out += list(members[((np.arange(k_g[g_]) + 0.5) * len(members) / max(k_g[g_], 1)).astype(int)])
    return np.sort(np.array(out, int))


def main():
    ap = argparse.ArgumentParser(description='Representative draws for hazard (see the module docstring).')
    ap.add_argument('runs', nargs='+', help='the replicate runs, each with <run>_kept.npz from gibbs.py --keep-draws')
    ap.add_argument('--problem', default='cru_clean.npz')
    ap.add_argument('--vectors', default=None, help='rate vectors, path without extension (default: <folder of the first '
                    'run>/rate_vectors)')
    ap.add_argument('--k', type=int, default=20, help='number of representatives written (default 20)')
    ap.add_argument('--eps', type=float, default=1e-6, help='offset per year in log10(P + eps) (default 1e-6)')
    ap.add_argument('--pick', default='spaced', choices=['spaced', 'kmeans'],
                    help='spaced (default since 2026-11-25): evenly spaced over each chain\'s kept draws; kmeans: the draw nearest '
                         'each k-means cluster centre')
    ap.add_argument('--weights', default='equal', choices=['equal', 'calibrated'],
                    help='equal (default since 2026-11-25): 1/K each; calibrated: the cluster shares (or 1/K) pulled towards the '
                         'exact mean participation rates')
    ap.add_argument('--mu', type=float, default=1.0, help='--weights calibrated: the pull towards the starting weights (default 1)')
    ap.add_argument('--seed', type=int, default=0, help='k-means seed (default 0)')
    ap.add_argument('--slip-block', default='S')
    ap.add_argument('--bins-block', default='MFD')
    ap.add_argument('--exact-pca-max', type=int, default=EXACT_PCA_MAX,
                    help=f'kept draws up to which the principal components are exact (default {EXACT_PCA_MAX}); randomized beyond')
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
    with np.load(vec + '.npz') as z_:                   # every array but the kept draws, which are replaced below
        rv = {k_: z_[k_] for k_ in z_.files if not k_.startswith('kept')}
    mean = np.asarray(rv['mean'], np.float64)[keep_c]
    Pbar = F @ mean
    X, src, counts = [], [], []
    for r_i, run in enumerate(a.runs):
        Xr, idx = load_kept(run)
        if Xr is None:
            raise SystemExit(f'{run}_kept.npz (or .npy) not found: run gibbs.py with --keep-draws (the notebook: REPRESENTATIVES > 0)')
        X.append(Xr)
        src.append(np.column_stack([np.full(len(idx), r_i), idx]))
        counts.append(len(idx))
    X, src = sp.vstack(X, format='csr'), np.concatenate(src)
    N = X.shape[0]
    RS_all = np.empty((N, nS))                           # every kept draw's subsection totals (the distribution check)
    for c0 in range(0, N, 1000):
        RS_all[c0:c0 + 1000] = as_dense(inc_s @ X[c0:c0 + 1000].T.astype(np.float64)).T
    Pk, live, nc, frac, scores = None, None, 0, None, None
    if a.pick == 'kmeans' or a.weights == 'calibrated':  # the participation features (k-means, the calibration)
        Pk = np.empty((N, F.shape[0]))
        for c0 in range(0, N, 256):
            Pk[c0:c0 + 256] = as_dense(F @ X[c0:c0 + 256].T.astype(np.float64)).T
        live = (Pk > 0).any(0)
    if a.pick == 'kmeans':
        Z = np.log10(Pk[:, live] + a.eps)
        Z -= Z.mean(0)
        if N <= a.exact_pca_max:
            ev, U = np.linalg.eigh(Z @ Z.T)              # principal components through the draws' Gram matrix
            ev, U = np.clip(ev[::-1], 0.0, None), U[:, ::-1]
            frac = np.cumsum(ev) / ev.sum()
        else:                                            # many draws (every recorded one, since 2026-11-25): the leading
            U, sv = randomized_pcs(Z, 80, a.seed)        # components by a randomized SVD; the total variance exactly
            ev = sv ** 2
            frac = np.cumsum(ev) / float(np.sum(Z.astype(np.float64) ** 2))
        nc = int(min(50, np.searchsorted(frac, 0.95) + 1))
        scores = U[:, :nc] * np.sqrt(ev[:nc])
        del Z

    # the posterior to check against: traces (chi2 as fitted, total rate) and per-draw totals over every recorded draw
    chi2_all, tot_all, tr_chi2, tr_tot, post_s, post_b = [], [], [], [], [], []
    for run in a.runs:
        z = np.load(run + '.npz')
        nb_ = int(z['burn']) // int(z['thin'])
        c_ = z['misfit'] if 'misfit' in z.files else z['chi2']   # chi2 as fitted (older runs: the Gaussian chi2)
        tr_chi2.append(c_)
        tr_tot.append(z['sum_m'])
        chi2_all.append(c_[:, nb_:].ravel())
        tot_all.append(z['sum_m'][:, nb_:].ravel())
        st = load_stream(run, ('tot_s', 'tot_b'))
        post_s.append(st['tot_s'].reshape(-1, nS).astype(np.float64))
        post_b.append(st['tot_b'].reshape(-1, B).astype(np.float64))
    chi2_rng = np.percentile(np.concatenate(chi2_all), [1, 99])
    tot_rng = np.percentile(np.concatenate(tot_all), [1, 99])
    qpost_s, qpost_b = np.quantile(np.concatenate(post_s), QS, axis=0), np.quantile(np.concatenate(post_b), QS, axis=0)
    exact_s, exact_b, exact_t = inc_s @ mean, inc_b @ mean, mean.sum()
    cal = Pbar > a.eps
    from compare_reps import SubsetMarginals            # the distribution check (compare_reps.py's KS)
    Msub = SubsetMarginals(RS_all)
    live_s = RS_all.std(0) > 0
    g_null = np.random.default_rng(a.seed + 11)

    res = {}
    for K in sizes:
        if K > N:
            print(f'K = {K}: skipped, only {N} kept draws')
            continue
        if a.pick == 'kmeans':
            lab, C = kmeans(scores, K, a.seed)
            rep = np.array([np.flatnonzero(lab == j)[((scores[lab == j] - C[j]) ** 2).sum(1).argmin()] for j in range(K)])
            w0 = np.bincount(lab, minlength=K) / N
        else:
            rep = spaced_picks(src, K)
            w0 = np.full(K, 1.0 / K)
        f0 = f1 = opt = None
        if a.weights == 'calibrated':
            A = Pk[rep][:, cal].T / Pbar[cal][:, None]
            w, f0, f1, opt = calibrate(A, w0, a.mu)
        else:
            w = np.full(K, 1.0 / K)
        ks_ = lambda idx_, ww_: float(np.median(Msub.stats(idx_, ww_)[0][live_s]))
        ks_rep = ks_(rep, w)
        ks_null = [ks_(np.sort(g_null.choice(N, K, replace=False)), np.full(K, 1.0 / K)) for _ in range(200)]
        ks_pct = float(np.mean(np.asarray(ks_null) < ks_rep) * 100)
        R = as_dense(X[rep]).astype(np.float64)
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
                      rc=rc, rtt=rtt, ratio=ratio, f0=f0, f1=f1, opt=opt, ok=ok, ks=ks_rep,
                      ks_null=float(np.median(ks_null)), ks_pct=ks_pct, ess=float(1.0 / np.sum(w ** 2)))

    # report
    pc = lambda x: f'{100 * x:.0f}%'
    sp_ = lambda x: f'{100 * x:+.1f}%'
    Ks = list(res)
    L = [f'Representative draws from {N} kept draws pooled over {len(a.runs)} runs '
         f'({", ".join(f"{r}: {c}" for r, c in zip(a.runs, counts))})',
         ('picks: evenly spaced over each chain\'s kept draws' if a.pick == 'spaced' else
          f'picks: k-means, the draw nearest each centre; features: {int(live.sum())} of {F.shape[0]} subsection-bin '
          f'participation rates nonzero in some draw; log10(P + {a.eps:g}); {nc} principal components cover '
          f'{100 * frac[nc - 1]:.1f}% of the variance; seed {a.seed}')
         + ('; weights equal (1/K)' if a.weights == 'equal' else
            f'; weights calibrated (mu {a.mu:g}) on the {int(cal.sum())} features whose posterior mean exceeds {a.eps:g}'),
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
    row('subsections\' distributions: KS median / random sets of K', [f'{res[K]["ks"]:.3f}/{res[K]["ks_null"]:.3f}' for K in Ks])
    row('  its percentile among 200 random equal-weight sets of K', [f'{res[K]["ks_pct"]:.0f}' for K in Ks])
    row('effective number of draws, (sum w)^2 / sum w^2', [f'{res[K]["ess"]:.1f}' for K in Ks])
    if a.weights == 'calibrated':
        row('weights: smallest (none zero)', [f'{res[K]["w"].min():.4f}' for K in Ks])
        row('weights: largest ratio to the starting weights', [f'{res[K]["ratio"].max():.2f}' for K in Ks])
        row('calibration objective at the starting weights', [f'{res[K]["f0"]:.4g}' for K in Ks])
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
    reps[:, keep_c] = as_dense(X[r['rep']])
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
        out.append(','.join(line.split(',')[:cut] + [fmt_rate(v) for v in reps[:, i]]))
    with open(vec + '.csv', 'w', newline='') as f:
        f.write(term.join(out) + term)
    with open(os.path.join(out_dir, 'representatives.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['rep', 'weight', 'cluster_share', 'run', 'chain', 'record', 'chi2', 'total_rate'])
        for j in range(a.k):
            s_ = src[r['rep'][j]]
            wr.writerow([cols[j], f'{r["w"][j]:.6f}', f'{r["w0"][j]:.6f}', a.runs[s_[0]], int(s_[1]), int(s_[2]),
                         f'{r["rc"][j]:.2f}', f'{r["rtt"][j]:.6e}'])
    Xo = sp.csr_matrix((X.data, keep_c[X.indices], X.indptr), shape=(N, n_orig))    # the original numbering
    rv.update(**kept_arrays(Xo, src), rep=reps, rep_weight=r['w'])
    np.savez(vec + '.npz', **rv)
    print(f'wrote {vec}.csv (columns {cols[0]} to {cols[-1]}), {os.path.join(out_dir, "representatives.csv")}, '
          f'{os.path.join(out_dir, "select_report.txt")}; added kept, kept_index, rep and rep_weight to {vec}.npz')


if __name__ == '__main__':
    main()

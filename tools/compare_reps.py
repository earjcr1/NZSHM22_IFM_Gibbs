#!/usr/bin/env python3
"""How well the representatives' weighted distribution matches the kept draws' (2026-11-25, the user's request).

    python compare_reps.py RUN [RUN ...] [--problem cru_clean.npz] [--ruptures DIR] [--vectors DIR/rate_vectors]
                           [--outdir DIR] [--n-null 400] [--seed 0] [--slip-block S] [--bins-block MFD]

RUN: the replicate runs select_draws.py pooled, in the same order (each <RUN>_kept.npy and _kept_index.npy). The
representatives and their weights come from representatives.csv beside the rate vectors (select_draws.py's output), or
from --reps (the same columns: run, chain, record, weight), to judge other weights or choices.
Each representative is itself one of the kept draws, so alone it is a genuine posterior draw; the question is whether
the set, with its weights, reproduces the posterior's distribution, not just its means.

Quantities, draw by draw: the total rate; each magnitude bin's total; each subsection's participation rate (the summed
rate of the ruptures crossing it); each named fault's event rate (the rupture set's named_faults.json, as
plot_slip_sections.py and covariance.py have them; with --ruptures).

1. Marginals: for each quantity, the weighted representatives' distribution against the kept draws': the
   Kolmogorov-Smirnov distance (largest gap between the two cumulative distributions; the share of draws with no event
   is a jump at 0 and counts), and the Wasserstein distance in units of the kept draws' sd (the area between them).
2. Jointly: the energy distance between the two sets in the subsections' standardized log10(rate + 1e-6) (zero only
   if the distributions are identical, every subsection at once), and likewise for the bins; and the faults' and the
   bins' correlation matrices, the rms difference of their off-diagonal entries.
3. The baselines: every statistic is also computed for --n-null random sets of K kept draws with equal weights, what
   chance alone gives at this K, and for --n-null with the representatives' own weights shuffled among them, what chance
   gives with weights that uneven (the "same w" percentiles: whether choosing and weighting them beats chance). A quantity is flagged where the representatives' statistic lies beyond the random
   sets' 95th percentile (5% expected by chance); each summary (a group's median statistic, counted or rate-weighted)
   is placed as a percentile among the random sets' (low: better than random draws; high: worse). The same K draws
   with equal weights are shown beside, to see what the weights do.
4. The weights: the effective number of representatives, (sum w)^2 / sum w^2.

Writes OUTDIR/compare_reps_report.txt, compare_reps.png (the summaries against their baselines; six faults'
distributions with the random sets' 5-95% envelope) and compare_reps.npz (default OUTDIR: the rate vectors' folder).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cru_common import load_problem   # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
REP_BLUE, NULL_GREY = '#0060c0', '#b9b8b2'
EPS = 1e-6                                               # per year, as select_draws.py's features


class Marginals:
    """Each quantity's kept draws sorted once, so that the KS and Wasserstein distances of any weighted subset of
    them follow from one cumulative sum. V: draws x quantities."""

    def __init__(self, V):
        self.N, self.Q = V.shape
        self.order = np.argsort(V, axis=0, kind='stable')
        Vs = np.take_along_axis(V, self.order, 0)
        self.last = np.ones_like(Vs, bool)                # the end of each run of ties (the steps of the CDFs)
        self.last[:-1] = Vs[1:] != Vs[:-1]
        self.gap = np.zeros_like(Vs)
        self.gap[:-1] = Vs[1:] - Vs[:-1]
        self.Fp = np.arange(1, self.N + 1)[:, None] / self.N
        sd = V.std(0)
        self.sd = np.where(sd > 0, sd, np.inf)

    def stats(self, w):
        """w: a weight per kept draw (summing to 1). Returns (KS, Wasserstein / sd) per quantity."""
        F = np.cumsum(w[self.order], axis=0)
        d = np.abs(F - self.Fp)
        ks = np.where(self.last, d, 0.0).max(0)
        w1 = (d * self.gap).sum(0) / self.sd
        return ks, w1

    def cdf(self, w, q):
        """Quantity q's cumulative distribution under weights w, at its sorted kept values (the steps)."""
        return np.cumsum(w[self.order[:, q]])


class SubsetMarginals:
    """Each quantity's kept draws sorted once; the KS and Wasserstein distances of a weighted subset of them (idx,
    weights w) against all of them, from the subset's K draws alone: between two of its values the subset's cumulative
    distribution is flat and the kept draws' rises, so the largest gap lies at the subset's values (each tie group's
    end, against the share of kept draws at or below it, and its start, against the share below), and the area
    between them follows from the kept draws' integrated cumulative distribution G. The same values as Marginals.stats
    (to rounding), in K log K per quantity rather than N: every recorded draw can be kept since 2026-11-25. V: draws x
    quantities."""

    def __init__(self, V):
        self.N, self.Q = V.shape
        self.V = V
        o = np.argsort(V, axis=0, kind='stable')
        self.Vs = np.take_along_axis(V, o, 0)
        self.rank = np.empty((self.N, self.Q), np.int64)
        np.put_along_axis(self.rank, o, np.arange(self.N)[:, None], 0)
        self.le = np.empty((self.N, self.Q), np.int64)    # kept draws at or below each draw's value, and below it
        self.lt = np.empty((self.N, self.Q), np.int64)
        for q in range(self.Q):
            self.le[:, q] = np.searchsorted(self.Vs[:, q], V[:, q], 'right')
            self.lt[:, q] = np.searchsorted(self.Vs[:, q], V[:, q], 'left')
        self.G = np.zeros_like(self.Vs)                   # the integral of the kept draws' cumulative distribution
        self.G[1:] = np.cumsum(np.arange(1, self.N)[:, None] / self.N * np.diff(self.Vs, axis=0), axis=0)
        sd = V.std(0)
        self.sd = np.where(sd > 0, sd, np.inf)

    def stats(self, idx, w):
        N, Q = self.N, self.Q
        cq = np.arange(Q)
        vals = self.V[idx]
        o = np.argsort(vals, axis=0, kind='stable')
        sv = np.take_along_axis(vals, o, 0)
        sw = np.asarray(w, float)[o]
        cw = np.cumsum(sw, 0)
        cwp = cw - sw
        le = np.take_along_axis(self.le[idx], o, 0) / N
        lt = np.take_along_axis(self.lt[idx], o, 0) / N
        last = np.ones_like(sv, bool)
        last[:-1] = sv[1:] != sv[:-1]
        first = np.ones_like(sv, bool)
        first[1:] = sv[1:] != sv[:-1]
        ks = np.maximum(np.where(last, np.abs(cw - le), 0.0), np.where(first, np.abs(cwp - lt), 0.0)).max(0)
        Gs = self.G[np.take_along_axis(self.rank[idx], o, 0), cq]
        area = Gs[0].copy()                                # below the subset's least value: its distribution is 0
        area += (self.Vs[-1] - sv[-1]) - (self.G[-1] - Gs[-1])   # above its greatest: 1
        if len(sv) > 1:
            a_, b_, Ga, Gb, c = sv[:-1], sv[1:], Gs[:-1], Gs[1:], cw[:-1]
            j = np.clip(np.ceil(c * N - 1e-9).astype(np.int64) - 1, 0, N - 1)   # the first kept value where theirs >= c
            xs_ = self.Vs[j, cq]
            Gx = self.G[j, cq]
            lo_, hi_ = xs_ < a_, xs_ > b_
            xs_ = np.where(lo_, a_, np.where(hi_, b_, xs_))
            Gx = np.where(lo_, Ga, np.where(hi_, Gb, Gx))
            area += (c * (xs_ - a_) - (Gx - Ga) + (Gb - Gx) - c * (b_ - xs_)).sum(0)
        return ks, area / self.sd


class Energy:
    """Energy distance between a weighted subset of the kept draws and all of them, in Z (draws x features): with all
    the draws' distances when there are at most EN_MAX, else against a fixed random EN_MAX of them (every recorded draw
    can be kept since 2026-11-25; the draws x draws distances would not fit)."""

    def __init__(self, Z, seed):
        self.Z = Z
        N = len(Z)
        self.ref = np.arange(N) if N <= EN_MAX else np.sort(np.random.default_rng(seed + 7).choice(N, EN_MAX, replace=False))
        D = distances(Z[self.ref])
        self.rr = D.mean()
        self.full = D if N <= EN_MAX else None
        self.rowmean = D.mean(1) if N <= EN_MAX else None

    def __call__(self, idx, w):
        if self.full is not None:
            return 2.0 * (w @ self.rowmean[idx]) - w @ self.full[np.ix_(idx, idx)] @ w - self.rr
        Zi = self.Z[idx]
        g_i, g_r = (Zi ** 2).sum(1), (self.Z[self.ref] ** 2).sum(1)
        Dxr = np.sqrt(np.maximum(g_i[:, None] + g_r[None, :] - 2.0 * Zi @ self.Z[self.ref].T, 0.0))
        return 2.0 * (w @ Dxr.mean(1)) - w @ distances(Zi) @ w - self.rr


EN_MAX = 6000                 # kept draws beyond which the energy distance takes a random reference set of this many


def energy(D, rowsum, total, idx, w):
    """Energy distance between the weighted points idx (weights w) and all N points (equal weights), from their
    distance matrix D: 2 E|X - Y| - E|X - X'| - E|Y - Y'|."""
    N = D.shape[0]
    return 2.0 * (w @ rowsum[idx]) / N - w @ D[np.ix_(idx, idx)] @ w - total / N ** 2


def distances(Z):
    """Euclidean distances between the rows of Z."""
    g = (Z ** 2).sum(1)
    return np.sqrt(np.maximum(g[:, None] + g[None, :] - 2.0 * Z @ Z.T, 0.0))


def wcorr(A, w):
    """Weighted Pearson correlations between the columns of A (draws x quantities)."""
    m = w @ A
    C = (A - m).T @ ((A - m) * w[:, None])
    s = np.sqrt(np.diag(C))
    s[s == 0] = np.inf
    return C / np.outer(s, s)


def offdiag_rms(C1, C2):
    iu = np.triu_indices(len(C1), 1)
    d = (C1 - C2)[iu]
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='the replicate runs select_draws.py pooled, in its order')
    ap.add_argument('--problem', default='cru_clean.npz')
    ap.add_argument('--ruptures', default=None, help='the rupture set (named_faults.json, fault_sections.geojson)')
    ap.add_argument('--vectors', default=None, help='rate vectors, path without extension (default: <folder of the '
                    'first run>/rate_vectors); representatives.csv is read from its folder')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--reps', default=None, help='the representatives to judge, a CSV with representatives.csv\'s run, chain, '
                    'record and weight columns (default: representatives.csv beside the rate vectors)')
    ap.add_argument('--n-null', type=int, default=400, help='random sets of K draws for the baseline (default 400)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--slip-block', default='S')
    ap.add_argument('--bins-block', default='MFD')
    a = ap.parse_args()
    t0 = time.time()
    vec = a.vectors or os.path.join(os.path.dirname(a.runs[0]), 'rate_vectors')
    vdir = os.path.dirname(vec) or '.'
    outdir = a.outdir or vdir

    # ---- the problem's aggregates
    P = load_problem(a.problem)
    G = P['G'].tocsr()
    rows_of = lambda blk: np.flatnonzero(np.asarray(P['block']) == list(P['block_names']).index(blk))
    inc_s = (G[rows_of(a.slip_block)] > 0).astype(np.float64).tocsr()
    inc_b = (G[rows_of(a.bins_block)] > 0).astype(np.float64).tocsr()
    nS, B = inc_s.shape[0], inc_b.shape[0]
    fnames, Bf = None, None
    if a.ruptures:
        from covariance import fault_incidence
        fnames, Bf = fault_incidence(P, G, a.ruptures, a.slip_block)
    from cru_common import magnitude_bins
    try:
        mlo = magnitude_bins(P, a.bins_block)[3]
        bnames = [f'M{v:.2f}' for v in mlo]
    except Exception:                                    # noqa: BLE001  (labels only)
        bnames = [f'bin {j + 1}' for j in range(B)]

    # ---- the kept draws, pooled as select_draws.py pools them, and the representatives among them
    from cru_common import as_dense, load_kept
    src, Xs = [], []
    for r_i, run in enumerate(a.runs):
        X, idx = load_kept(run)                          # sparse (CSR), one row per draw
        if X is None:
            raise SystemExit(f'{run}: no kept draws (gibbs.py --keep-draws)')
        src.append(np.column_stack([np.full(len(idx), r_i), idx]))
        Xs.append(X)
    src = np.concatenate(src)
    N = len(src)
    RS, RB = np.empty((N, nS)), np.empty((N, B))
    RF = np.empty((N, Bf.shape[0])) if Bf is not None else None
    tot = np.empty(N)
    c = 0
    for X in Xs:
        for c0 in range(0, X.shape[0], 1000):
            Xc = X[c0:c0 + 1000].T.astype(np.float64)
            k = Xc.shape[1]
            RS[c:c + k], RB[c:c + k] = as_dense(inc_s @ Xc).T, as_dense(inc_b @ Xc).T
            tot[c:c + k] = np.asarray(Xc.sum(0)).ravel()
            if Bf is not None:
                RF[c:c + k] = as_dense(Bf @ Xc).T
            c += k
    runs_n = [os.path.normpath(r) for r in a.runs]
    rep_idx, w = [], []
    with open(a.reps or os.path.join(vdir, 'representatives.csv')) as f:
        for row in csv.DictReader(f):
            r_i = runs_n.index(os.path.normpath(row['run'])) if os.path.normpath(row['run']) in runs_n else None
            if r_i is None:
                raise SystemExit(f'representatives.csv names run {row["run"]}, not among {a.runs}: give the runs '
                                 'select_draws.py pooled, in its order')
            hit = np.flatnonzero((src[:, 0] == r_i) & (src[:, 1] == int(row['chain'])) & (src[:, 2] == int(row['record'])))
            if len(hit) != 1:
                raise SystemExit(f'representative {row["rep"]} not found among the kept draws')
            rep_idx.append(int(hit[0]))
            w.append(float(row['weight']))
    rep_idx, w = np.array(rep_idx), np.array(w)
    w = w / w.sum()
    K = len(rep_idx)
    ess = 1.0 / np.sum(w ** 2)
    rv = np.load(vec + '.npz', mmap_mode='r')
    if 'rep' in rv.files and not a.reps:                                # the stored representatives must be these kept draws
        keep_c = np.asarray(P['keep_cols'])
        r0 = np.asarray(rv['rep'][0], np.float64)[keep_c]
        r_i = int(src[rep_idx[0], 0])
        x0 = as_dense(Xs[r_i][rep_idx[0] - int(np.flatnonzero(src[:, 0] == r_i)[0])]).astype(np.float64).ravel()
        if not np.allclose(r0, x0, rtol=1e-6, atol=0):
            raise SystemExit('the stored first representative is not the kept draw representatives.csv names')
    print(f'{N} kept draws, {K} representatives (effective {ess:.1f}); loaded in {time.time() - t0:.0f} s', flush=True)

    # ---- marginals
    groups = [('total rate', tot[:, None], None), ('magnitude bins', RB, None), ('subsections', RS, None)]
    if RF is not None:
        groups.append(('named faults', RF, None))
    M = {name: SubsetMarginals(V) for name, V, _ in groups}
    Mfig = {'named faults': Marginals(RF)} if RF is not None else {}      # the figure's cumulative distributions
    rate_w = {name: V.mean(0) / max(V.mean(0).sum(), 1e-300) for name, V, _ in groups}
    live = {name: V.std(0) > 0 for name, V, _ in groups}

    def wvec(idx, ww):
        v = np.zeros(N)
        np.add.at(v, idx, ww)
        return v

    def summaries(idx, ww):
        out = {}
        v = wvec(idx, ww)
        for name, V, _ in groups:
            ks, w1 = M[name].stats(idx, ww)
            L = live[name]
            rw = rate_w[name][L] / rate_w[name][L].sum()
            out[name] = dict(ks=ks, w1=w1, ks_med=float(np.median(ks[L])), ks_rw=float(rw @ ks[L]),
                             w1_med=float(np.median(w1[L])), w1_rw=float(rw @ w1[L]))
        return out

    # ---- jointly: energy distances and correlations
    def stdlog(V):
        Z = np.log10(V + EPS)
        s = Z.std(0)
        Z = (Z[:, s > 0] - Z[:, s > 0].mean(0)) / s[s > 0]
        return Z / np.sqrt(Z.shape[1])                   # per dimension, so the two groups' sizes compare
    E_s, E_b = Energy(stdlog(RS), a.seed), Energy(stdlog(RB), a.seed)
    eq = np.full(N, 1.0 / N)
    C_f = wcorr(RF, eq) if RF is not None else None
    C_b = wcorr(RB, eq)

    def joint(idx, ww):
        out = dict(energy_s=E_s(idx, ww), energy_b=E_b(idx, ww),
                   corr_b=offdiag_rms(wcorr(RB[idx], ww), C_b))
        if RF is not None:
            out['corr_f'] = offdiag_rms(wcorr(RF[idx], ww), C_f)
        return out

    rep = summaries(rep_idx, w)
    rep_eq = summaries(rep_idx, np.full(K, 1.0 / K))
    jrep, jrep_eq = joint(rep_idx, w), joint(rep_idx, np.full(K, 1.0 / K))
    g = np.random.default_rng(a.seed)
    null, jnull, nullw, jnullw = [], [], [], []
    env_q = None
    fault_pick = None
    if RF is not None:                                   # six faults spread over the event rates, for the figure
        L = live['named faults']
        order = np.flatnonzero(L)[np.argsort(RF.mean(0)[L])]
        fault_pick = order[np.unique(np.linspace(0, len(order) - 1, 6).round().astype(int))]
        env_q = {q: [] for q in fault_pick}
    for t in range(a.n_null):
        idx = np.sort(g.choice(N, K, replace=False))
        ww = np.full(K, 1.0 / K)
        null.append(summaries(idx, ww))
        jnull.append(joint(idx, ww))
        idx2, ww2 = np.sort(g.choice(N, K, replace=False)), g.permutation(w)   # random draws, the representatives' weights
        nullw.append(summaries(idx2, ww2))
        jnullw.append(joint(idx2, ww2))
        if env_q is not None:
            v = wvec(idx, ww)
            for q in fault_pick:
                env_q[q].append(Mfig['named faults'].cdf(v, q))
        if (t + 1) % 100 == 0:
            print(f'  baseline: {t + 1} random sets of {K} ({time.time() - t0:.0f} s)', flush=True)

    # ---- the verdicts
    pct = lambda x, arr: float(np.mean(np.asarray(arr) < x) * 100 + np.mean(np.asarray(arr) == x) * 50)
    L_ = []
    L_.append(f'Representatives against the kept draws: {K} representatives (weights from select_draws.py; effective '
              f'number {ess:.1f}) against {N:,} kept draws, {", ".join(a.runs)}')
    L_.append(f'baselines: {a.n_null} random sets of {K} kept draws with equal weights (what chance alone gives at this K), '
              f'and {a.n_null} with the representatives\' own weights, shuffled (what chance gives with weights this uneven: '
              f'the "same w" percentiles)')
    L_.append('percentile: where the representatives\' statistic falls among the random sets\' (low: closer to the kept '
              'draws than random draws are; high: further); flagged: quantities beyond the random sets\' 95th '
              'percentile of that quantity (5% expected by chance)')
    L_.append('')
    L_.append('1. Each quantity\'s distribution (KS: the largest gap between the cumulative distributions; W1/sd: the area '
              'between them in units of the kept draws\' sd)')
    hdr = (f'{"group":16s} {"n":>5s}  {"KS median":>9s} {"(random)":>9s} {"pct":>4s}  {"KS rate-wtd":>11s} {"(random)":>9s} '
           f'{"pct":>4s}  {"W1/sd median":>12s} {"(random)":>9s} {"pct":>4s}  {"flagged":>7s} {"rate-wtd":>8s}  {"same w: KS pct":>14s} '
           f'{"equal w: KS pct":>15s}')
    L_.append(hdr)
    flags = {}
    for name, V, _ in groups:
        L = live[name]
        nks = np.array([n_[name]['ks'] for n_ in null])            # n_null x Q
        thr = np.percentile(nks, 95, axis=0)
        fl = (rep[name]['ks'] > thr) & L
        flags[name] = dict(flag=fl, thr=thr)
        rw = rate_w[name]
        nm = lambda k: [n_[name][k] for n_ in null]
        L_.append(f'{name:16s} {int(L.sum()):5d}  {rep[name]["ks_med"]:9.3f} {np.median(nm("ks_med")):9.3f} '
                  f'{pct(rep[name]["ks_med"], nm("ks_med")):4.0f}  {rep[name]["ks_rw"]:11.3f} {np.median(nm("ks_rw")):9.3f} '
                  f'{pct(rep[name]["ks_rw"], nm("ks_rw")):4.0f}  {rep[name]["w1_med"]:12.3f} {np.median(nm("w1_med")):9.3f} '
                  f'{pct(rep[name]["w1_med"], nm("w1_med")):4.0f}  {fl[L].mean():7.1%} {rw[fl].sum() / max(rw[L].sum(), 1e-300):8.1%}  '
                  f'{pct(rep[name]["ks_med"], [n_[name]["ks_med"] for n_ in nullw]):14.0f} '
                  f'{pct(rep_eq[name]["ks_med"], nm("ks_med")):15.0f}')
    L_.append('')
    L_.append('2. Jointly (energy distance in standardized log10(rate + 1e-6), per dimension; correlations: rms difference '
              'of the off-diagonal entries)')
    for key, lab in (('energy_s', 'energy distance, subsections'), ('energy_b', 'energy distance, magnitude bins'),
                     ('corr_f', 'correlations, named faults'), ('corr_b', 'correlations, magnitude bins')):
        if key not in jrep:
            continue
        arr, arrw = [j_[key] for j_ in jnull], [j_[key] for j_ in jnullw]
        L_.append(f'{lab:34s} representatives {jrep[key]:.4f} (equal weights {jrep_eq[key]:.4f}); random sets: median '
                  f'{np.median(arr):.4f}, 5-95% {np.percentile(arr, 5):.4f}-{np.percentile(arr, 95):.4f}; percentile '
                  f'{pct(jrep[key], arr):.0f} (same w {pct(jrep[key], arrw):.0f}; equal weights {pct(jrep_eq[key], arr):.0f})')
    if RF is not None:
        L_.append('')
        L_.append('3. The named faults (event rate per year: the kept draws\' mean; no event: share of draws without one, '
                  'kept draws / representatives)')
        L_.append(f'{"fault":34s} {"mean rate":>10s} {"no event":>15s} {"KS":>6s} {"random 95%":>10s}  flag')
        V = RF
        zr = (V == 0).mean(0)
        vw = wvec(rep_idx, w)
        zw = vw @ (V == 0)
        for q in np.argsort(V.mean(0)):
            L_.append(f'{fnames[q][:34]:34s} {V[:, q].mean():10.3e} {zr[q]:7.1%} / {zw[q]:5.1%} {rep["named faults"]["ks"][q]:6.3f} '
                      f'{flags["named faults"]["thr"][q]:10.3f}  {"*" if flags["named faults"]["flag"][q] else ""}')
    txt = '\n'.join(L_) + '\n'
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'compare_reps_report.txt'), 'w') as f:
        f.write(txt)
    print(txt)
    np.savez(os.path.join(outdir, 'compare_reps.npz'), rep_idx=rep_idx, w=w, ess=ess,
             **{f'{n_.replace(" ", "_")}_ks': rep[n_]['ks'] for n_ in rep},
             **{f'{n_.replace(" ", "_")}_ks95': flags[n_]['thr'] for n_ in rep},
             **{f'null_{k_}': np.array([j_[k_] for j_ in jnull]) for k_ in jrep},
             **{f'nullw_{k_}': np.array([j_[k_] for j_ in jnullw]) for k_ in jrep},
             **{f'rep_{k_}': jrep[k_] for k_ in jrep})

    # ---- the figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})
    panels = [('subsections', 'ks_rw', 'Subsections (KS, rate-wtd)'),
              ('magnitude bins', 'ks_med', 'Magnitude bins (KS, median)'),
              ('named faults', 'ks_med', 'Named faults (KS, median)') if RF is not None else None,
              ('energy_s', None, 'Subsections jointly (energy)')]
    panels = [p_ for p_ in panels if p_ is not None]
    nrow = 3 if RF is not None else 1
    fig = plt.figure(figsize=(13, 3.3 + 3.0 * (nrow - 1)))
    gs = fig.add_gridspec(nrow, 12, hspace=0.75, wspace=1.2)
    for i, (grp, key, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 3 * i:3 * i + 3])
        if key is None:
            arr, arrw, x, xe = [j_[grp] for j_ in jnull], [j_[grp] for j_ in jnullw], jrep[grp], jrep_eq[grp]
        else:
            arr, arrw = [n_[grp][key] for n_ in null], [n_[grp][key] for n_ in nullw]
            x, xe = rep[grp][key], rep_eq[grp][key]
        bins_ = np.histogram_bin_edges(np.r_[arr, arrw, x, xe], 36)
        ax.hist(arr, bins=bins_, color=NULL_GREY, lw=0)
        ax.hist(arrw, bins=bins_, histtype='step', color=INK2, lw=1)
        ax.axvline(x, color=REP_BLUE, lw=2)
        ax.axvline(xe, color=REP_BLUE, lw=1, ls=(0, (3, 2)))
        ax.set_title(title, loc='left', fontsize=9.5, color=INK, fontweight='bold')
        ax.text(0.98, 0.95, f'percentile {pct(x, arr):.0f}\n(same weights: {pct(x, arrw):.0f})', transform=ax.transAxes,
                ha='right', va='top', color=REP_BLUE, fontsize=8.5)
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.set_xlabel('lower = closer to the kept draws', color=MUTED, fontsize=8)
    if RF is not None:
        vw = wvec(rep_idx, w)
        Mf = Mfig['named faults']
        for i, q in enumerate(fault_pick):
            ax = fig.add_subplot(gs[1 + i // 3, 4 * (i % 3):4 * (i % 3) + 4])
            xs = np.take_along_axis(RF[:, q][:, None], Mf.order[:, q][:, None], 0).ravel()
            E = np.array(env_q[q])
            lo, hi = np.percentile(E, 5, axis=0), np.percentile(E, 95, axis=0)
            pos = xs > 0
            xmin = xs[pos].min() if pos.any() else 1e-8
            xp = np.maximum(xs, xmin / 3)
            ax.fill_between(xp, lo, hi, step='post', color=NULL_GREY, alpha=0.6, lw=0, label=f'random sets of {K}: 5–95%')
            ax.step(xp, Mf.Fp[:, 0], where='post', color=INK, lw=1.2, label=f'{N:,} kept draws')
            ax.step(xp, Mf.cdf(vw, q), where='post', color=REP_BLUE, lw=1.6, label=f'{K} representatives, weighted')
            ax.set_xscale('log')
            ax.set_ylim(0, 1.02)
            ks_ = rep['named faults']['ks'][q]
            ax.set_title(f'{fnames[q][:40]}', loc='left', fontsize=9, color=INK, fontweight='bold')
            z_ = (RF[:, q] == 0).mean()
            ax.text(0.02, 0.95, (f'no event: {z_:.0%} of kept draws\n' if z_ > 0 else '') + f'KS {ks_:.3f} '
                    f'(random 95%: {flags["named faults"]["thr"][q]:.3f})', transform=ax.transAxes, va='top', fontsize=7.8,
                    color=INK2)
            ax.grid(True, color=GRID, lw=0.5)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xlabel('event rate per year (no event at the left edge)', fontsize=8)
            if i % 3 == 0:
                ax.set_ylabel('cumulative share of draws', fontsize=8)
            if i == 0:
                h_, l_ = ax.get_legend_handles_labels()
        fig.legend(h_, l_, loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'Representatives against the kept draws: the {K} weighted (blue line; dashed: the same {K} with equal '
                 f'weights) against random sets of {K}\n(grey: equal weights; dark outline: the representatives\' weights, '
                 f'shuffled)', x=0.06, ha='left', fontsize=11, color=INK, fontweight='bold')
    out = os.path.join(outdir, 'compare_reps.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
    print(f'wrote {out}, compare_reps_report.txt and compare_reps.npz ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()

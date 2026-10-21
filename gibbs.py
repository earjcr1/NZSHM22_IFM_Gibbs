#!/usr/bin/env python3
"""gibbs.py -- exact Gibbs sampler for  p(m) ~ exp(-||G m - d||^2 / 2 sigma^2 - beta.m)  on  lo <= m <= hi.

Each parameter is drawn from its exact conditional, a normal N(mu_i, s_i^2) truncated to [lo, hi] with
s_i = sigma/||G_i|| and mu_i = m_i - G_i.r/||G_i||^2 - s_i^2 beta_i, where r = G m - d is a running residual.
One sweep costs one pass over the nonzeros of G.  No tuning parameters.  Truncated normals use Robert's (1995)
exponential proposal in the tails, so conditionals sitting hundreds of sds beyond a bound are drawn exactly.

    python gibbs.py --problem cru_clean.npz --swap-cos 0.99 --chains 4 --sweeps 2400 --burn 400 --thin 8 --out spikeslab_run
    python gibbs.py --selftest

Under the log-rate priors (--prior), each rate is drawn exactly by a slice sampler in log rate. The default is
spike-and-slab per magnitude bin: each rupture is off (rate 0) or on, with a log-uniform rate from the slab floor (1e-10)
up to a soft top at its bin's target rate, and the fraction on in each bin is estimated. --prior binbox-theta:1:0 gives
the alternative, per-bin log-uniform boxes whose depths put no pressure on the bin totals (theta = 1). See --help for
the rest.

Backends: a C kernel compiled on first use with $CC (default 'cc') and loaded through ctypes; chains run in
parallel threads (ctypes releases the GIL).  Falls back to a vectorised NumPy kernel (~100x slower) if no
compiler is available, or on request with --backend numpy.
"""
import argparse
import ctypes
import hashlib
import os
import subprocess
import sys
import threading
import time

import numpy as np
import scipy.sparse as sp

from cru_common import load_problem, column_norms, prior_rates, split_rhat, ess, summarize_samples, resolve_logbox, bin_targets, logbox_mean, prelim_path
import scipy.sparse as _sparse
sp_csc = _sparse.csc_matrix

LN10 = float(np.log(10.0))

C_SRC = r"""
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t xnext(uint64_t *s) {                       /* xoshiro256** */
    uint64_t r = rotl64(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl64(s[3], 45);
    return r;
}
static inline double unif(uint64_t *s) { return ((double)(xnext(s) >> 11) + 0.5) * 0x1.0p-53; }   /* (0,1) */
static double stdnorm(uint64_t *s) {                                  /* Marsaglia polar */
    double u, v, q;
    do { u = 2.0 * unif(s) - 1.0; v = 2.0 * unif(s) - 1.0; q = u * u + v * v; } while (q >= 1.0 || q == 0.0);
    return u * sqrt(-2.0 * log(q) / q);
}
/* z ~ N(0,1) truncated to [a, a + w], a >= 0; returns the offset z - a (avoids cancellation deep in the tail) */
static double tn_tail_off(uint64_t *s, double a, double w) {
    double lam = 0.5 * (a + sqrt(a * a + 4.0));
    if (w * lam < 1.0) {                                              /* narrow window: uniform proposal */
        for (;;) { double o = w * unif(s); if (unif(s) <= exp(-0.5 * o * (2.0 * a + o))) return o; }
    }
    for (;;) {                                                        /* Robert (1995) exponential proposal */
        double o = -log(unif(s)) / lam;
        if (o > w) continue;
        double t = a + o - lam;
        if (unif(s) <= exp(-0.5 * t * t)) return o;
    }
}
/* z ~ N(0,1) truncated to [a, b] with a < 0 < b */
static double tn_mid(uint64_t *s, double a, double b) {
    if (b - a < 2.5066282746310002) {
        for (;;) { double z = a + (b - a) * unif(s); if (unif(s) <= exp(-0.5 * z * z)) return z; }
    }
    for (;;) { double z = stdnorm(s); if (z >= a && z <= b) return z; }
}
/* density ~ exp(-beta (x - lo)) on [lo, hi]; beta may be 0 or negative */
static double texp(uint64_t *s, double beta, double lo, double hi) {
    double w = hi - lo, u = unif(s);
    if (fabs(beta) * w < 1e-12) return lo + u * w;
    return lo - log1p(u * expm1(-beta * w)) / beta;
}

/* ---- log-box prior: flat in z = log m on [zlo, zhi]; outside, half-Gaussian edges exp(-k (distance)^2), k <= 0: hard */
typedef struct { int pk; double m0, mu, inv2s2, zmin, zmax, zlo, zhi, k, ku, lp0, muz, kz; } lbctx;
static inline double lb_prior(const lbctx *c, double z) {
    if (c->pk == 2) return -(z - c->muz) * (z - c->muz) * c->kz;      /* hierarchical: log m ~ Normal(muz, 1/(2 kz)) */
    if (z < c->zlo) return (c->k > 0.0) ? -(c->zlo - z) * (c->zlo - z) * c->k : -INFINITY;
    if (z > c->zhi) return (c->ku > 0.0) ? -(z - c->zhi) * (z - c->zhi) * c->ku : -INFINITY;   /* upper edge: own width */
    return 0.0;
}
/* log conditional density of z relative to the current point z0 (so it is exactly 0 at z0) */
static inline double lb_ld(const lbctx *c, double z) {
    if (!(z >= c->zmin && z <= c->zmax)) return -INFINITY;
    double p = lb_prior(c, z);
    if (p == -INFINITY) return -INFINITY;
    double m = exp(z);
    return -(m - c->m0) * (m + c->m0 - 2.0 * c->mu) * c->inv2s2 + p - c->lp0;
}
/* univariate slice sampler (Neal 2003): stepping out with width w (at most 64 steps), then shrinkage */
static double lb_slice(uint64_t *s, const lbctx *c, double z0, double w) {
    const double y = log(unif(s));
    double L = z0 - w * unif(s), R = L + w;
    int64_t J = (int64_t)(64.0 * unif(s)), Kk = 63 - J;
    while (J-- > 0 && lb_ld(c, L) > y) L -= w;
    while (Kk-- > 0 && lb_ld(c, R) > y) R += w;
    for (;;) {
        double z1 = L + (R - L) * unif(s);
        if (lb_ld(c, z1) > y) return z1;
        if (z1 < z0) L = z1; else R = z1;
    }
}

/* Exact draw (in z = log m) from the spike-and-slab's slab, the pseudo-prior of an "off" rupture: flat on [zlo, zhi]
   (hard floor), with a Gaussian shoulder above zhi of sd 1/sqrt(2 ku) up to zmax (ku > 0), or a hard top (ku = 0). */
static double slab_draw(uint64_t *s, const lbctx *c) {
    const double zt = fmin(c->zhi, c->zmax), wp = zt - c->zlo;
    double wt = 0.0, sdt = 0.0, lim = 0.0;
    if (c->ku > 0.0 && c->zmax > c->zhi) {
        sdt = 1.0 / sqrt(2.0 * c->ku); lim = c->zmax - c->zhi;
        wt = sdt * 1.2533141373155003 * erf(lim / (sdt * 1.4142135623730951));   /* sdt sqrt(pi/2) erf(...) */
    }
    if (unif(s) * (wp + wt) < wp) return c->zlo + wp * unif(s);
    for (;;) { const double t = fabs(stdnorm(s)) * sdt; if (t <= lim) return c->zhi + t; }
}

/* K chains, chain-major: X[k*n + i], R[k*N + j] (R = G X - d, kept current), rng[4k..4k+3]; Gaussian likelihood.
   pkind 0: prior exp(-beta.m) on [lo, hi] (beta = 0: flat).  pkind 1: log-box prior (beta ignored), drawn by slice
   sampling z = log m. */
void gibbs_run(int64_t N, int64_t n, int64_t K, const int64_t *indptr, const int32_t *indices,
               const double *data, const double *inv_csq, const double *lo, const double *hi,
               const double *beta, double sigma, double *X, double *R, uint64_t *rng, int64_t nsweeps,
               int64_t pkind, double zlo, double zhi, double kedge, double slw,
               const double *MUZ, const double *KZ, const double *ZLO, const double *KU, double *SM, const double *LOD)
{
    const double s2 = sigma * sigma;
    for (int64_t k = 0; k < K; k++) {
        double *x = X + k * n, *r = R + k * N;
        uint64_t *s = rng + 4 * k;
        for (int64_t sw = 0; sw < nsweeps; sw++) {
            for (int64_t i = 0; i < n; i++) {
                const int64_t p0 = indptr[i], p1 = indptr[i + 1];
                const int empty = (p1 == p0 || inv_csq[i] <= 0.0);
                const double xi = x[i];
                double xn, dot = 0.0, aa = 0.0;
                if (!empty) {
                    for (int64_t p = p0; p < p1; p++) dot += data[p] * r[indices[p]];
                    aa = 1.0 / inv_csq[i];
                }
                if (pkind == 3) {             /* spike-and-slab: x_i is 0 (off) or its slab value (on) */
                    lbctx c;
                    c.pk = 1; c.muz = 0.0; c.kz = 0.0;
                    c.zlo = ZLO ? ZLO[i] : zlo; c.zhi = zhi; c.k = 0.0; c.ku = KU ? KU[i] : 0.0;
                    c.zmin = lo[i] > 0.0 ? log(lo[i]) : -INFINITY;
                    c.zmax = log(hi[i]);
                    c.mu = 0.0; c.inv2s2 = 0.0;
                    double m;
                    if (xi > 0.0) {           /* on: exact slice update of the slab value under likelihood x slab */
                        const double z0 = log(xi);
                        c.m0 = xi; c.lp0 = lb_prior(&c, z0);
                        if (!empty) { c.mu = xi - dot / aa; c.inv2s2 = 0.5 * aa / s2; }
                        m = exp(lb_slice(s, &c, z0, slw));
                    } else {                  /* off: a fresh draw from the slab, the pseudo-prior */
                        m = exp(slab_draw(s, &c));
                    }
                    /* on or off given m: log odds = log(pi/(1-pi)) + log L(m) - log L(0); with the rupture switched off
                       the residual is r0 = r - x_i G_i, so G_i.r0 = dot - x_i ||G_i||^2 */
                    double lr = LOD[k * n + i];
                    if (!empty) { const double d0 = dot - xi * aa; lr -= (2.0 * m * d0 + m * m * aa) / (2.0 * s2); }
                    const double pon = lr >= 0.0 ? 1.0 / (1.0 + exp(-lr)) : exp(lr) / (1.0 + exp(lr));
                    xn = (unif(s) < pon) ? m : 0.0;
                    SM[k * n + i] = m;
                } else if (pkind >= 1) {
                    lbctx c;
                    c.pk = (int)pkind; c.muz = 0.0; c.kz = 0.0;
                    if (pkind == 2) { c.muz = MUZ[k * n + i]; c.kz = KZ[k]; }
                    c.zlo = ZLO ? ZLO[i] : zlo; c.zhi = zhi; c.k = kedge; c.ku = KU ? KU[i] : kedge;
                    c.zmin = lo[i] > 0.0 ? log(lo[i]) : -INFINITY;
                    c.zmax = log(hi[i]);
                    const double z0 = log(xi > DBL_MIN ? xi : DBL_MIN);
                    c.m0 = exp(z0); c.lp0 = lb_prior(&c, z0);
                    c.mu = 0.0; c.inv2s2 = 0.0;
                    if (!empty) {
                        c.mu = xi - dot / aa;
                        c.inv2s2 = 0.5 * aa / s2;
                    }
                    xn = exp(lb_slice(s, &c, z0, slw));
                } else if (empty) {
                    xn = texp(s, beta[i], lo[i], hi[i]);
                } else {
                    const double v = inv_csq[i], sd = sigma * sqrt(v);
                    const double mu = xi - dot * v - s2 * v * beta[i];
                    const double a = (lo[i] - mu) / sd, b = (hi[i] - mu) / sd;
                    if (a >= 0.0)      xn = lo[i] + sd * tn_tail_off(s, a, b - a);
                    else if (b <= 0.0) xn = hi[i] - sd * tn_tail_off(s, -b, b - a);
                    else               xn = mu + sd * tn_mid(s, a, b);
                }
                if (xn < lo[i]) xn = lo[i]; else if (xn > hi[i]) xn = hi[i];
                const double dl = xn - xi;
                if (dl != 0.0 && !empty)
                    for (int64_t p = p0; p < p1; p++) r[indices[p]] += data[p] * dl;
                x[i] = xn;
            }
        }
    }
}

/* Swap moves: for each listed pair (i, j), propose exchanging m_i and m_j (an involution, so the Metropolis ratio
   is target(new)/target(old)).  Change of ||r||^2: 2 dl (G_i - G_j).r + dl^2 ||G_i - G_j||^2 with dl = m_j - m_i.
   The log-box prior is the same for every rate, so it cancels; an exponential prior contributes -dl (beta_i - beta_j). */
void swap_run(int64_t N, int64_t n, int64_t K, const int64_t *indptr, const int32_t *indices, const double *data,
              const double *csq, int64_t npair, const int64_t *pi, const int64_t *pj, const double *gij,
              const double *beta, double sigma, double *X, double *R, uint64_t *rng, int64_t *nacc)
{
    const double s2 = sigma * sigma;
    for (int64_t k = 0; k < K; k++) {
        double *x = X + k * n, *r = R + k * N;
        uint64_t *s = rng + 4 * k;
        int64_t acc = 0;
        for (int64_t q = 0; q < npair; q++) {
            const int64_t i = pi[q], j = pj[q];
            const double mi = x[i], mj = x[j];
            if (mi == mj) continue;
            double ci = 0.0, cj = 0.0;
            for (int64_t p = indptr[i]; p < indptr[i + 1]; p++) ci += data[p] * r[indices[p]];
            for (int64_t p = indptr[j]; p < indptr[j + 1]; p++) cj += data[p] * r[indices[p]];
            const double dl = mj - mi;
            const double dchi = 2.0 * dl * (ci - cj) + dl * dl * (csq[i] + csq[j] - 2.0 * gij[q]);
            const double dlp = -dl * (beta[i] - beta[j]);
            if (log(unif(s)) < -0.5 * dchi / s2 + dlp) {
                for (int64_t p = indptr[i]; p < indptr[i + 1]; p++) r[indices[p]] += data[p] * dl;
                for (int64_t p = indptr[j]; p < indptr[j + 1]; p++) r[indices[p]] -= data[p] * dl;
                x[i] = mj; x[j] = mi; acc++;
            }
        }
        nacc[k] += acc;
    }
}

/* ---- transfer moves: exact draws along fixed directions v (library from build_transfers.py).
   The line m + t v is limited by the walls m >= 0 (and m <= hi) to t in [tmin, tmax].  It is sampled in the logit
   coordinate s = log((t - tmin) / (tmax - t)), which resolves both ends logarithmically, so a draw can carry a rate
   all the way down into the prior's floor region; the Jacobian is (t - tmin)(tmax - t) / L.  The slice sampler starts
   from the whole range s in [-SMAX, SMAX] and shrinks, so it can jump between separated parts of the line. */
#define TR_MAXK 16
static inline void tr_ab(double s, double L, double *a, double *b) {
    if (s >= 0.0) { double e = exp(-s); *a = L / (1.0 + e); *b = L * e / (1.0 + e); }
    else          { double e = exp(s);  *a = L * e / (1.0 + e); *b = L / (1.0 + e); }
}
static double tr_logdens(double s, int64_t nk, const double *av, const double *off, const int *side,
                         const double *hik, const double *bk, double L, double tmin, double tmax, double ru,
                         double uu, double inv2s2, int64_t pkind, double zlo, double zhi, double kedge, double *mout,
                         double *tout)
{
    double a, b;
    tr_ab(s, L, &a, &b);
    if (!(a > 0.0) || !(b > 0.0)) return -INFINITY;
    const double t = (a <= b) ? tmin + a : tmax - b;
    double lp = log(a) + log(b);
    for (int64_t q = 0; q < nk; q++) {
        const double m = av[q] * ((side[q] > 0 ? a : b) + off[q]);
        if (!(m > 0.0) || m > hik[q]) return -INFINITY;
        if (mout) mout[q] = m;
        if (pkind == 1) {
            const double z = log(m);
            double p = 0.0;
            if (z < zlo) { if (kedge > 0.0) p = -(zlo - z) * (zlo - z) * kedge; else return -INFINITY; }
            else if (z > zhi) { if (kedge > 0.0) p = -(z - zhi) * (z - zhi) * kedge; else return -INFINITY; }
            lp += p - z;                               /* density in m: f(log m) / m */
        } else {
            lp -= bk[q] * m;
        }
    }
    lp -= (2.0 * t * ru + t * t * uu) * inv2s2;
    if (tout) *tout = t;
    return lp;
}

void transfer_run(int64_t N, int64_t n, int64_t K, int64_t ndir, const int64_t *dptr, const int64_t *didx,
                  const double *dval, const int64_t *uptr, const int32_t *urow, const double *uval, const double *uu,
                  const double *hi, const double *beta, double sigma, int64_t pkind, double zlo, double zhi,
                  double kedge, double *X, double *R, uint64_t *rng, int64_t *stats)
{
    const double inv2s2 = 0.5 / (sigma * sigma), SMAX = 60.0;
    for (int64_t k = 0; k < K; k++) {
        double *x = X + k * n, *r = R + k * N;
        uint64_t *s = rng + 4 * k;
        int64_t tried = 0, big = 0;
        for (int64_t q = 0; q < ndir; q++) {
            const int64_t p0 = dptr[q], nk = dptr[q + 1] - p0;
            if (nk > TR_MAXK || nk < 2) continue;
            double av[TR_MAXK], off[TR_MAXK], hik[TR_MAXK], bk[TR_MAXK], mnew[TR_MAXK];
            int side[TR_MAXK];
            double tmin = -INFINITY, tmax = INFINITY, mmax0 = 0.0;
            for (int64_t c = 0; c < nk; c++) {
                const int64_t i = didx[p0 + c];
                const double v = dval[p0 + c], m = x[i];
                if (m > mmax0) mmax0 = m;
                if (v > 0.0) { tmin = fmax(tmin, -m / v); tmax = fmin(tmax, (hi[i] - m) / v); }
                else         { tmax = fmin(tmax, m / -v); tmin = fmax(tmin, (m - hi[i]) / -v); }
            }
            const double a0 = -tmin, b0 = tmax;
            if (!(a0 > 0.0) || !(b0 > 0.0)) continue;
            for (int64_t c = 0; c < nk; c++) {
                const int64_t i = didx[p0 + c];
                const double v = dval[p0 + c], m = x[i];
                av[c] = fabs(v); hik[c] = hi[i]; bk[c] = beta[i];
                if (v > 0.0) { side[c] = 1;  off[c] = tmin + m / v; }
                else         { side[c] = -1; off[c] = m / -v - tmax; }
                if (off[c] < 0.0) off[c] = 0.0;
            }
            double ru = 0.0;
            for (int64_t p = uptr[q]; p < uptr[q + 1]; p++) ru += uval[p] * r[urow[p]];
            const double L = a0 + b0, s0 = log(a0) - log(b0);
            if (!(fabs(s0) < SMAX - 1.0)) continue;
            const double g0 = tr_logdens(s0, nk, av, off, side, hik, bk, L, tmin, tmax, ru, uu[q], inv2s2, pkind, zlo,
                                         zhi, kedge, NULL, NULL);
            if (!(g0 > -INFINITY)) continue;
            tried++;
            const double y = g0 + log(unif(s));
            double L_s = -SMAX, R_s = SMAX, s1, t1 = 0.0;
            for (;;) {
                s1 = L_s + (R_s - L_s) * unif(s);
                if (tr_logdens(s1, nk, av, off, side, hik, bk, L, tmin, tmax, ru, uu[q], inv2s2, pkind, zlo, zhi, kedge,
                               mnew, &t1) > y) break;
                if (s1 < s0) L_s = s1; else R_s = s1;
            }
            double mmax1 = 0.0;
            for (int64_t c = 0; c < nk; c++) { x[didx[p0 + c]] = mnew[c]; if (mnew[c] > mmax1) mmax1 = mnew[c]; }
            for (int64_t p = uptr[q]; p < uptr[q + 1]; p++) r[urow[p]] += t1 * uval[p];
            if (mmax0 > 1e-6 || mmax1 > 1e-6) big++;
        }
        stats[2 * k] += tried;
        stats[2 * k + 1] += big;
    }
}
"""

_LIB = None


def build_kernel():
    """Compile the C kernel once (cached by source hash) and return the ctypes library, or None."""
    global _LIB
    if _LIB is not None:
        return _LIB
    h = hashlib.sha1(C_SRC.encode()).hexdigest()[:12]
    cache = os.path.join(os.path.expanduser('~'), '.cache', 'cru_gibbs')
    os.makedirs(cache, exist_ok=True)
    lib = os.path.join(cache, f'gibbs_{h}' + ('.dylib' if sys.platform == 'darwin' else '.so'))
    if not os.path.exists(lib):
        src = os.path.join(cache, f'gibbs_{h}.c')
        open(src, 'w').write(C_SRC)
        cmd = [os.environ.get('CC', 'cc'), '-O3', '-shared', '-fPIC', '-o', lib, src, '-lm']
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError) as e:
            print(f'[gibbs] could not compile C kernel ({e}); using NumPy backend', file=sys.stderr)
            return None
    L = ctypes.CDLL(lib)
    P = ctypes.c_void_p
    I, D = ctypes.c_int64, ctypes.c_double
    L.gibbs_run.argtypes = [I] * 3 + [P] * 7 + [D] + [P] * 3 + [I, I, D, D, D, D] + [P, P, P, P, P, P]
    L.gibbs_run.restype = None
    L.swap_run.argtypes = [I] * 3 + [P] * 4 + [I] + [P] * 4 + [D] + [P] * 4
    L.swap_run.restype = None
    L.transfer_run.argtypes = [I] * 4 + [P] * 9 + [D, I, D, D, D] + [P] * 4
    L.transfer_run.restype = None
    _LIB = L
    return L


class Target:
    """Arrays the kernels need, built once from (G, d, sigma, lo, hi, beta)."""

    def __init__(self, G, d, sigma, lo, hi, beta=None, logbox=None, pairs=None, dirs=None, lognormal=None):
        """logbox = (a, b, se): prior flat in log10 m on [a, b], with half-Gaussian edges of sd se decades (se = 0: hard).
        pairs = (i, j, G_i.G_j) arrays: candidate swap moves (near-duplicate columns)."""
        G = sp.csc_matrix(G, dtype=np.float64)
        G.sort_indices()
        self.G, self.d, self.sigma = G, np.ascontiguousarray(d, np.float64), float(sigma)
        self.N, self.n = G.shape
        self.indptr = np.ascontiguousarray(G.indptr, np.int64)
        self.indices = np.ascontiguousarray(G.indices, np.int32)
        self.data = np.ascontiguousarray(G.data, np.float64)
        cn = column_norms(G)
        self.inv_csq = np.where(cn > 0, 1.0 / np.where(cn > 0, cn, 1.0) ** 2, 0.0)
        self.lo = np.ascontiguousarray(np.broadcast_to(lo, self.n), np.float64)
        self.hi = np.ascontiguousarray(np.broadcast_to(hi, self.n), np.float64)
        self.beta = np.ascontiguousarray(np.zeros(self.n) if beta is None else np.broadcast_to(beta, self.n), np.float64)
        self.csq = np.where(self.inv_csq > 0, 1.0 / np.where(self.inv_csq > 0, self.inv_csq, 1.0), 0.0)
        self.pkind, self.zlo, self.zhi, self.kedge, self.slw, self.logbox = 0, 0.0, 0.0, 0.0, 1.0, logbox
        self.zlo_vec = None                        # optional per-coordinate lower edge (log-box prior), overrides zlo
        self.ku_vec = None                         # optional per-coordinate upper-edge width k = 1/(2 sd^2), sd in ln rate
        self.ss = None                             # spike-and-slab settings (pkind 3): bins, B, nb, a, b, optional fixed pi
        if logbox is not None:
            a10, b10, se10 = map(float, logbox)
            self.pkind = 1
            self.zlo, self.zhi = a10 * LN10, b10 * LN10
            self.kedge = 0.0 if se10 <= 0 else 1.0 / (2.0 * (se10 * LN10) ** 2)
            self.beta = np.zeros(self.n)
        self.hier = None
        if lognormal is not None:                  # hierarchical prior; rates unbounded above (the data cap them)
            self.pkind = 2
            self.hier = lognormal if isinstance(lognormal, LogNormalHier) else LogNormalHier(**lognormal)
            self.beta = np.zeros(self.n)
            self.hi = np.full(self.n, np.inf)
        self.dirs = None
        if dirs is not None and len(dirs['uu']):
            types = dict(d_ptr=np.int64, d_idx=np.int64, d_val=np.float64, u_ptr=np.int64, u_row=np.int32,
                         u_val=np.float64, uu=np.float64)
            self.dirs = {k: np.ascontiguousarray(dirs[k], t) for k, t in types.items()}
        self.pairs = None
        if pairs is not None and len(pairs[0]):
            self.pairs = tuple(np.ascontiguousarray(a, t) for a, t in zip(pairs, (np.int64, np.int64, np.float64)))

    def init(self, X0):
        """Clip starting values into the support (log-box prior: zeros go to the floor, hard edges respected)."""
        X = np.clip(np.asarray(X0, np.float64), self.lo, self.hi)
        if self.pkind == 2:
            X = np.maximum(X, 1e-10)
        if self.pkind == 3:                        # spike-and-slab: below the slab floor means off (exactly 0)
            X = np.where(X >= np.exp(self.zlo if self.zlo_vec is None else self.zlo_vec), X, 0.0)
        if self.pkind == 1:
            X = np.maximum(X, np.exp(self.zlo if self.zlo_vec is None else self.zlo_vec))
            if self.kedge == 0:
                X = np.minimum(X, np.exp(self.zhi))
        return np.ascontiguousarray(X)

    def residual(self, X):
        return np.ascontiguousarray((self.G @ X.T).T - self.d)


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def sweeps_c(lib, T, X, R, rng, nsweeps, threads=1, MUZ=None, KZ=None, SM=None, LOD=None):
    if MUZ is None:
        MUZ, KZ = np.zeros(X.shape), np.zeros(X.shape[0])
    if SM is None:
        SM, LOD = np.zeros(X.shape), np.zeros(X.shape)
    K = X.shape[0]
    groups = np.array_split(np.arange(K), max(1, min(threads, K)))

    def work(ks):
        if len(ks) == 0:
            return
        k0, k1 = int(ks[0]), int(ks[-1]) + 1
        lib.gibbs_run(T.N, T.n, k1 - k0, _ptr(T.indptr), _ptr(T.indices), _ptr(T.data), _ptr(T.inv_csq),
                      _ptr(T.lo), _ptr(T.hi), _ptr(T.beta), T.sigma,
                      ctypes.c_void_p(X.ctypes.data + k0 * T.n * 8), ctypes.c_void_p(R.ctypes.data + k0 * T.N * 8),
                      ctypes.c_void_p(rng.ctypes.data + k0 * 32), nsweeps,
                      T.pkind, T.zlo, T.zhi, T.kedge, T.slw,
                      ctypes.c_void_p(MUZ.ctypes.data + k0 * T.n * 8), ctypes.c_void_p(KZ.ctypes.data + k0 * 8),
                      None if T.zlo_vec is None else _ptr(T.zlo_vec), None if T.ku_vec is None else _ptr(T.ku_vec),
                      ctypes.c_void_p(SM.ctypes.data + k0 * T.n * 8), ctypes.c_void_p(LOD.ctypes.data + k0 * T.n * 8))
    if len(groups) == 1:
        work(groups[0])
    else:
        th = [threading.Thread(target=work, args=(g,)) for g in groups]
        [t.start() for t in th]
        [t.join() for t in th]


def swaps_c(lib, T, X, R, rng, nacc, threads=1):
    pi, pj, gij = T.pairs
    K = X.shape[0]
    groups = np.array_split(np.arange(K), max(1, min(threads, K)))

    def work(ks):
        if len(ks) == 0:
            return
        k0, k1 = int(ks[0]), int(ks[-1]) + 1
        lib.swap_run(T.N, T.n, k1 - k0, _ptr(T.indptr), _ptr(T.indices), _ptr(T.data), _ptr(T.csq), len(pi),
                     _ptr(pi), _ptr(pj), _ptr(gij), _ptr(T.beta), T.sigma,
                     ctypes.c_void_p(X.ctypes.data + k0 * T.n * 8), ctypes.c_void_p(R.ctypes.data + k0 * T.N * 8),
                     ctypes.c_void_p(rng.ctypes.data + k0 * 32), ctypes.c_void_p(nacc.ctypes.data + k0 * 8))
    if len(groups) == 1:
        work(groups[0])
    else:
        th = [threading.Thread(target=work, args=(g,)) for g in groups]
        [t.start() for t in th]
        [t.join() for t in th]


def transfers_c(lib, T, X, R, rng, stats, threads=1):
    D = T.dirs
    K = X.shape[0]
    groups = np.array_split(np.arange(K), max(1, min(threads, K)))

    def work(ks):
        if len(ks) == 0:
            return
        k0, k1 = int(ks[0]), int(ks[-1]) + 1
        lib.transfer_run(T.N, T.n, k1 - k0, len(D['uu']), _ptr(D['d_ptr']), _ptr(D['d_idx']), _ptr(D['d_val']),
                         _ptr(D['u_ptr']), _ptr(D['u_row']), _ptr(D['u_val']), _ptr(D['uu']), _ptr(T.hi), _ptr(T.beta),
                         T.sigma, T.pkind, T.zlo, T.zhi, T.kedge,
                         ctypes.c_void_p(X.ctypes.data + k0 * T.n * 8), ctypes.c_void_p(R.ctypes.data + k0 * T.N * 8),
                         ctypes.c_void_p(rng.ctypes.data + k0 * 32), ctypes.c_void_p(stats.ctypes.data + k0 * 16))
    if len(groups) == 1:
        work(groups[0])
    else:
        th = [threading.Thread(target=work, args=(g,)) for g in groups]
        [t.start() for t in th]
        [t.join() for t in th]


# ---------------------------------------------------------------- NumPy backend (same algorithm, vectorised over chains)
def _tail_off_np(g, a, w):
    lam = 0.5 * (a + np.sqrt(a * a + 4.0))
    narrow = w * lam < 1.0
    out = np.empty_like(a)
    todo = np.arange(a.size)
    while todo.size:
        aa, ww, ll, nn = a[todo], w[todo], lam[todo], narrow[todo]
        o = np.where(nn, ww * g.random(todo.size), -np.log1p(-g.random(todo.size)) / ll)
        u = g.random(todo.size)
        with np.errstate(over='ignore', invalid='ignore'):
            acc = np.where(nn, u <= np.exp(-0.5 * o * (2 * aa + o)), (o <= ww) & (u <= np.exp(-0.5 * (aa + o - ll) ** 2)))
        out[todo[acc]] = o[acc]
        todo = todo[~acc]
    return out


def _mid_np(g, a, b):
    out = np.empty_like(a)
    todo = np.arange(a.size)
    while todo.size:
        aa, bb = a[todo], b[todo]
        uni = bb - aa < 2.5066282746310002
        z = np.where(uni, aa + (bb - aa) * g.random(todo.size), g.standard_normal(todo.size))
        acc = np.where(uni, g.random(todo.size) <= np.exp(-0.5 * z * z), (z >= aa) & (z <= bb))
        out[todo[acc]] = z[acc]
        todo = todo[~acc]
    return out


def _u01(g):
    u = g.random()
    while u == 0.0:
        u = g.random()
    return u


def _lb_slice_py(g, c, z0, w):
    """Python twin of lb_slice in the C kernel (same algorithm)."""
    def lprior(z):
        if c.get('pk', 1) == 2:
            return -(z - c['muz']) ** 2 * c['kz']
        if z < c['zlo']:
            return -(c['zlo'] - z) ** 2 * c['k'] if c['k'] > 0 else -np.inf
        if z > c['zhi']:
            ku = c.get('ku', c['k'])
            return -(z - c['zhi']) ** 2 * ku if ku > 0 else -np.inf
        return 0.0

    def ld(z):
        if not (c['zmin'] <= z <= c['zmax']):
            return -np.inf
        p = lprior(z)
        if p == -np.inf:
            return -np.inf
        m = np.exp(z)
        return -(m - c['m0']) * (m + c['m0'] - 2.0 * c['mu']) * c['inv2s2'] + p - c['lp0']
    c['lp0'] = 0.0
    c['lp0'] = lprior(z0)
    y = np.log(_u01(g))
    L = z0 - w * _u01(g)
    R = L + w
    J = int(64 * _u01(g))
    Kk = 63 - J
    while J > 0 and ld(L) > y:
        L -= w
        J -= 1
    while Kk > 0 and ld(R) > y:
        R += w
        Kk -= 1
    while True:
        z1 = L + (R - L) * _u01(g)
        if ld(z1) > y:
            return z1
        if z1 < z0:
            L = z1
        else:
            R = z1


def _sweep_logbox_numpy(T, X, R, g, i, MUZ=None, KZ=None):
    p0, p1 = T.indptr[i], T.indptr[i + 1]
    empty = p1 == p0 or T.inv_csq[i] <= 0
    rows, vals = T.indices[p0:p1], T.data[p0:p1]
    lo, hi = T.lo[i], T.hi[i]
    for k in range(X.shape[0]):
        xi = X[k, i]
        z0 = np.log(max(xi, np.finfo(float).tiny))
        c = dict(zlo=T.zlo if T.zlo_vec is None else T.zlo_vec[i], zhi=T.zhi, k=T.kedge,
                 ku=T.kedge if T.ku_vec is None else T.ku_vec[i], zmin=np.log(lo) if lo > 0 else -np.inf, zmax=np.log(hi),
                 m0=np.exp(z0), mu=0.0, inv2s2=0.0, pk=T.pkind,
                 muz=MUZ[k, i] if T.pkind == 2 else 0.0, kz=KZ[k] if T.pkind == 2 else 0.0)
        if not empty:
            dot = float(R[k, rows] @ vals)
            aa = T.csq[i]
            c.update(mu=xi - dot / aa, inv2s2=0.5 * aa / T.sigma ** 2)
        xn = min(max(np.exp(_lb_slice_py(g, c, z0, T.slw)), lo), hi)
        if xn != xi and not empty:
            R[k, rows] += vals * (xn - xi)
        X[k, i] = xn


def _slab_draw_py(g, c):
    """NumPy twin of slab_draw: an exact draw of z = log m from the slab."""
    import math
    zt = min(c['zhi'], c['zmax'])
    wp = zt - c['zlo']
    wt = sdt = lim = 0.0
    if c['ku'] > 0 and c['zmax'] > c['zhi']:
        sdt, lim = 1.0 / np.sqrt(2.0 * c['ku']), c['zmax'] - c['zhi']
        wt = sdt * np.sqrt(np.pi / 2.0) * math.erf(lim / (sdt * np.sqrt(2.0)))
    if _u01(g) * (wp + wt) < wp:
        return c['zlo'] + wp * _u01(g)
    while True:
        t = abs(g.standard_normal()) * sdt
        if t <= lim:
            return c['zhi'] + t


def _sweep_spikeslab_numpy(T, X, R, g, i, SM, LOD):
    """NumPy twin of the kernel's spike-and-slab update of rupture i in every chain."""
    p0, p1 = T.indptr[i], T.indptr[i + 1]
    empty = p1 == p0 or T.inv_csq[i] <= 0
    rows, vals = T.indices[p0:p1], T.data[p0:p1]
    lo, hi, s2 = T.lo[i], T.hi[i], T.sigma ** 2
    for k in range(X.shape[0]):
        xi = X[k, i]
        c = dict(zlo=T.zlo if T.zlo_vec is None else T.zlo_vec[i], zhi=T.zhi, k=0.0, ku=0.0 if T.ku_vec is None else T.ku_vec[i],
                 zmin=np.log(lo) if lo > 0 else -np.inf, zmax=np.log(hi), mu=0.0, inv2s2=0.0, pk=1, muz=0.0, kz=0.0, m0=1.0)
        dot = aa = 0.0
        if not empty:
            dot, aa = float(R[k, rows] @ vals), T.csq[i]
        if xi > 0:
            c['m0'] = xi
            if not empty:
                c.update(mu=xi - dot / aa, inv2s2=0.5 * aa / s2)
            m = np.exp(_lb_slice_py(g, c, np.log(xi), T.slw))
        else:
            m = np.exp(_slab_draw_py(g, c))
        lr = LOD[k, i]
        if not empty:
            lr -= (2.0 * m * (dot - xi * aa) + m * m * aa) / (2.0 * s2)
        pon = 1.0 / (1.0 + np.exp(-lr)) if lr >= 0 else np.exp(lr) / (1.0 + np.exp(lr))
        xn = min(max(m if _u01(g) < pon else 0.0, lo), hi)
        SM[k, i] = m
        if xn != xi and not empty:
            R[k, rows] += vals * (xn - xi)
        X[k, i] = xn


def swaps_numpy(T, X, R, g, nacc):
    pi, pj, gij = T.pairs
    s2 = T.sigma ** 2
    for k in range(X.shape[0]):
        for q in range(len(pi)):
            i, j = pi[q], pj[q]
            mi, mj = X[k, i], X[k, j]
            if mi == mj:
                continue
            ri, vi = T.indices[T.indptr[i]:T.indptr[i + 1]], T.data[T.indptr[i]:T.indptr[i + 1]]
            rj, vj = T.indices[T.indptr[j]:T.indptr[j + 1]], T.data[T.indptr[j]:T.indptr[j + 1]]
            dl = mj - mi
            dchi = 2 * dl * (R[k, ri] @ vi - R[k, rj] @ vj) + dl * dl * (T.csq[i] + T.csq[j] - 2 * gij[q])
            dlp = -dl * (T.beta[i] - T.beta[j])
            if np.log(_u01(g)) < -0.5 * dchi / s2 + dlp:
                R[k, ri] += vi * dl
                R[k, rj] -= vj * dl
                X[k, i], X[k, j] = mj, mi
                nacc[k] += 1


def sweeps_numpy(T, X, R, g, nsweeps, MUZ=None, KZ=None, SM=None, LOD=None):
    s2 = T.sigma ** 2
    for _ in range(nsweeps):
        for i in range(T.n):
            if T.pkind == 3:
                _sweep_spikeslab_numpy(T, X, R, g, i, SM, LOD)
                continue
            if T.pkind >= 1:
                _sweep_logbox_numpy(T, X, R, g, i, MUZ, KZ)
                continue
            p0, p1 = T.indptr[i], T.indptr[i + 1]
            lo, hi, be = T.lo[i], T.hi[i], T.beta[i]
            if p1 == p0:
                u = g.random(X.shape[0])
                w = hi - lo
                X[:, i] = lo + u * w if abs(be) * w < 1e-12 else lo - np.log1p(u * np.expm1(-be * w)) / be
                continue
            rows, vals = T.indices[p0:p1], T.data[p0:p1]
            v = T.inv_csq[i]
            sd = T.sigma * np.sqrt(v)
            xi = X[:, i].copy()
            mu = xi - (R[:, rows] @ vals) * v - s2 * v * be
            a, b = (lo - mu) / sd, (hi - mu) / sd
            xn = np.empty_like(mu)
            tl, th = a >= 0, b <= 0
            md = ~(tl | th)
            if tl.any():
                xn[tl] = lo + sd * _tail_off_np(g, a[tl], (b - a)[tl])
            if th.any():
                xn[th] = hi - sd * _tail_off_np(g, -b[th], (b - a)[th])
            if md.any():
                xn[md] = mu[md] + sd * _mid_np(g, a[md], b[md])
            np.clip(xn, lo, hi, out=xn)
            X[:, i] = xn
            R[:, rows] += np.outer(xn - xi, vals)


def _slice1(g, f, x0, w, m=400):
    """Univariate slice sampler (stepping out, then shrinkage); f returns -inf outside its support."""
    y = f(x0) + np.log(_u01(g))
    L = x0 - w * _u01(g)
    R = L + w
    J = int(m * _u01(g))
    Kk = m - 1 - J
    while J > 0 and f(L) > y:
        L -= w
        J -= 1
    while Kk > 0 and f(R) > y:
        R += w
        Kk -= 1
    while True:
        x1 = L + (R - L) * _u01(g)
        if f(x1) > y:
            return x1
        if x1 < x0:
            L = x1
        else:
            R = x1


class LogNormalHier:
    """Hierarchical prior: log10 m_i ~ Normal(mu_b(i), s^2), a centre per bin and a shared spread s.
    Hyperpriors: s uniform on s_range (decades); mu_b ~ Normal(mu_prior[0], mu_prior[1]^2) (log10 units).
    Each block: a centred update (mu_b conjugate, s by slice sampling, given all rates) and a non-centred one
    (rescaling the rates with the standardized deviations fixed, using the likelihood): ASIS, Yu & Meng (2011).
    fixed = (mu_b array, s) holds the hyperparameters fixed (used in tests)."""

    def __init__(self, bins, s_range=(0.05, 5.0), mu_prior=(-6.0, 10.0), fixed=None, strong=None, s_fixed=None):
        self.bins = np.ascontiguousarray(bins, np.int64)
        self.B = int(self.bins.max()) + 1
        self.nb = np.bincount(self.bins, minlength=self.B).astype(float)
        self.ulo, self.uhi = np.log(s_range[0] * LN10), np.log(s_range[1] * LN10)
        self.m0, self.v0 = mu_prior[0] * LN10, (mu_prior[1] * LN10) ** 2
        self.s_range, self.mu_prior, self.fixed = tuple(s_range), tuple(mu_prior), fixed
        # partial non-centring: rates flagged 'strong' (data-pinned, chosen in advance) stay put when the
        # hyperparameters move; the others move with them (standardized deviations held fixed)
        self.strong = None if strong is None else np.ascontiguousarray(strong, bool)
        self.s_fixed = s_fixed                 # hold the spread fixed (decades) and learn only the bin centres

    def init(self, Z):
        if self.fixed is not None:
            mu = np.tile(np.asarray(self.fixed[0], float) * LN10, (Z.shape[0], 1))
            return mu, np.full(Z.shape[0], self.fixed[1] * LN10)
        mu = np.stack([np.bincount(self.bins, weights=z, minlength=self.B) / np.maximum(self.nb, 1) for z in Z])
        if self.s_fixed is not None:
            return mu, np.full(Z.shape[0], self.s_fixed * LN10)
        tau = np.sqrt(np.mean((Z - mu[:, self.bins]) ** 2, 1))
        return mu, np.clip(tau, np.exp(self.ulo) * 1.001, np.exp(self.uhi) * 0.999)

    def update(self, g, T, z, mu, tau):
        if self.fixed is not None:
            return z, mu, tau
        n, b = len(z), self.bins
        # centred: mu_b | z, tau (conjugate normal), then tau | z, mu (slice in u = log tau; flat prior on tau)
        sb = np.bincount(b, weights=z, minlength=self.B)
        prec = self.nb / tau ** 2 + 1.0 / self.v0
        mu = (sb / tau ** 2 + self.m0 / self.v0) / prec + g.standard_normal(self.B) / np.sqrt(prec)
        ss = float(np.sum((z - mu[b]) ** 2))
        f = lambda u: (-(n - 1) * u - 0.5 * ss * np.exp(-2 * u)) if self.ulo <= u <= self.uhi else -np.inf
        if self.s_fixed is None:
            tau = float(np.exp(_slice1(g, f, np.log(tau), 0.05)))
        # (partially) non-centred: weakly informed rates keep their standardized deviations eta and move with the
        # hyperparameters; data-pinned rates ('strong', fixed in advance) keep their values. Both updates use the likelihood.
        S = self.strong if self.strong is not None else np.zeros(n, bool)
        W = ~S
        nS = int(S.sum())

        def loglik(m):
            r = T.G @ m - T.d
            return -0.5 * float(r @ r) / T.sigma ** 2
        base = mu[b]
        eta = (z - base) / tau
        ssS = float(np.sum((z[S] - base[S]) ** 2))
        mS = np.exp(z[S])

        def m_of(u):
            m = np.empty(n)
            m[W] = np.exp(base[W] + np.exp(u) * eta[W])
            m[S] = mS
            return m
        f = lambda u: (loglik(m_of(u)) - nS * u - 0.5 * ssS * np.exp(-2 * u) + u) if self.ulo <= u <= self.uhi else -np.inf
        if self.s_fixed is None:
            tau = float(np.exp(_slice1(g, f, np.log(tau), 0.05)))
        z = np.where(W, base + tau * eta, z)
        m = np.exp(z)
        r = T.G @ m - T.d
        Y = (T.G @ sp.csc_matrix((m * W, (np.arange(n), b)), shape=(n, self.B))).toarray()
        for k in range(self.B):
            if self.nb[k] == 0:
                continue
            yb = Y[:, k]
            rr, ry, yy = float(r @ r), float(r @ yb), float(yb @ yb)
            zSk = z[S & (b == k)]

            def fb(dl):
                e1 = np.expm1(dl)
                c2 = rr + 2 * e1 * ry + e1 * e1 * yy
                lp = -0.5 * (mu[k] + dl - self.m0) ** 2 / self.v0 - 0.5 * float(np.sum((zSk - mu[k] - dl) ** 2)) / tau ** 2
                return lp - 0.5 * c2 / T.sigma ** 2
            dl = _slice1(g, fb, 0.0, 0.1)
            if dl != 0.0:
                r = r + np.expm1(dl) * yb
                mu[k] += dl
                z[(b == k) & W] += dl
        return z, mu, tau


# ---------------------------------------------------------------- driver
def seed_states(seed, K):
    ss = np.random.SeedSequence(seed).spawn(K)
    st = np.array([s.generate_state(4, np.uint64) for s in ss], dtype=np.uint64)
    st[np.all(st == 0, axis=1), 0] = 1
    return np.ascontiguousarray(st)


def sample(T, X0, sweeps, burn, thin, backend='c', threads=1, seed=0, monitor=(), out=None, keep_samples=True, verbose=True,
           swaps=False, transfers=False, hyper_every=None, stream=None, keep_draws=0):
    """Run K = X0.shape[0] chains.  Records every `thin` sweeps; draws after `burn` sweeps are kept.
    Returns dict of traces; if out is given, writes <out>.npz and (if keep_samples) <out>_samples.npy.
    keep_draws > 0 also keeps that many whole draws (keep_draws / K per chain, evenly spaced over the recorded draws after
    burn-in, all of them if there are fewer) in <out>_kept.npy (float32, one row per draw, in the order recorded) with
    their chain and record numbers in <out>_kept_index.npy; the record number indexes the traces (chi2, sum_m), and
    record - burn // thin indexes the stored draws.  It only records: the chains are the same with or without it."""
    lib = build_kernel() if backend == 'c' else None
    if backend == 'c' and lib is None:
        backend = 'numpy'
    sc = getattr(T, 'scale', None)                       # per-rupture scale: the sampler works on u = m / scale
    X = T.init(X0 if sc is None else np.asarray(X0, np.float64) / sc)
    K = X.shape[0]
    swaps = bool(swaps and T.pairs is not None)
    nacc = np.zeros(K, np.int64)
    transfers = bool(transfers and T.dirs is not None)
    tstats = np.zeros((K, 2), np.int64)
    if transfers and backend != 'c':
        raise NotImplementedError('transfer moves need the C kernel')
    if transfers and T.pkind == 2:
        raise NotImplementedError('transfer moves are not implemented for the hierarchical prior')
    H = T.hier
    MUZ, KZ = np.zeros((K, T.n)), np.zeros(K)
    if H is not None:
        hmu, htau = H.init(np.log(X))
        MUZ, KZ = np.ascontiguousarray(hmu[:, H.bins]), 0.5 / htau ** 2
    SS = T.ss if T.pkind == 3 else None
    SM, LOD = np.zeros((K, T.n)), np.zeros((K, T.n))
    if SS is not None:
        SM = np.ascontiguousarray(np.where(X > 0, X, np.exp(T.zhi)))
    R = T.residual(X)
    rng = seed_states(seed, K)
    g = np.random.default_rng(seed)
    if SS is not None:                                   # per-bin inclusion probabilities: exact Beta updates
        pib = np.empty((K, SS['B']))

        def update_pi():
            for k in range(K):
                if SS.get('pi') is not None:
                    pib[k] = SS['pi']
                else:
                    on = np.bincount(SS['bins'], weights=(X[k] > 0).astype(float), minlength=SS['B'])
                    pib[k] = g.beta(SS['a'] + on, SS['b'] + SS['nb'] - on)
            p = np.clip(pib, 1e-300, 1.0 - 1e-16)
            LOD[:] = (np.log(p) - np.log1p(-p))[:, SS['bins']]
        update_pi()
    n_rec = sweeps // thin
    n_burn = burn // thin
    n_keep = n_rec - n_burn
    monitor = np.asarray(monitor, dtype=np.int64)
    tr = dict(chi2=np.zeros((K, n_rec)), sum_m=np.zeros((K, n_rec)), tscale=np.zeros((K, n_rec)),
              mon=np.zeros((K, n_rec, len(monitor))), y=np.zeros((K, n_keep, T.N), np.float32), sweep=np.zeros(n_rec, int))
    dd = T.d @ T.d
    samp = None
    ST = None                                            # running summaries: everything the notebook needs, without draws
    if stream is not None and n_keep > 0:
        nb_ = max(1, min(int(stream.get('nbatch', 20)), n_keep))   # never more batches than recorded draws
        ST = dict(s1=np.zeros((K, 2, T.n)), s2=np.zeros((K, 2, T.n)), cnt=np.zeros(2), bsum=np.zeros((K, nb_, T.n), np.float32),
                  bcnt=np.zeros(nb_), exc5=np.zeros(T.n), exc4=np.zeros(T.n), msum=np.zeros(T.n), usum=np.zeros(T.n),
                  hist=np.zeros((stream['hist_nb'], T.n), np.int32), hcnt=np.zeros(1),
                  n5=np.zeros((K, n_keep), np.int32), n4=np.zeros((K, n_keep), np.int32),
                  n90=np.zeros((K, n_keep), np.int32), n99=np.zeros((K, n_keep), np.int32))
        for key in ('inc_s', 'inc_b'):
            if stream.get(key) is not None:
                ST['tot_' + key[-1]] = np.zeros((K, n_keep, stream[key].shape[0]), np.float32)
    if keep_samples and n_keep > 0:
        samp = (np.lib.format.open_memmap(f'{out}_samples.npy', 'w+', np.float32, (K, n_keep, T.n)) if out
                else np.zeros((K, n_keep, T.n), np.float32))
    KD = None                                            # kept draws: a thinned set of whole draws (recording only)
    if keep_draws and keep_draws > 0 and n_keep > 0:
        per = [min(n_keep, keep_draws // K + (1 if k < keep_draws % K else 0)) for k in range(K)]
        if keep_draws // K + (1 if keep_draws % K else 0) > n_keep and verbose:
            print(f'keep-draws: only {n_keep} recorded draws per chain after burn-in, so all of them are kept '
                  f'({sum(per)} of the {keep_draws} asked for)')
        order = sorted((int((j + 0.5) * n_keep / c), k) for k, c in enumerate(per) for j in range(c))   # (draw, chain)
        KD = dict(at={}, index=np.array([(k, n_burn + t) for t, k in order], np.int64).reshape(-1, 2))
        for row, (t, k) in enumerate(order):
            KD['at'].setdefault(t, []).append((k, row))
        KD['X'] = (np.lib.format.open_memmap(f'{out}_kept.npy', 'w+', np.float32, (len(order), T.n)) if out
                   else np.zeros((len(order), T.n), np.float32))
    t0 = time.time()
    he = thin if ((H is None and SS is None) or not hyper_every) else max(1, min(int(hyper_every), thin))
    for rec in range(n_rec):
        done = 0
        while done < thin:                                   # hierarchical prior: hyperparameters every `he` sweeps
            ns = min(he, thin - done)
            if backend == 'c':
                sweeps_c(lib, T, X, R, rng, ns, threads, MUZ, KZ, SM, LOD)
            else:
                sweeps_numpy(T, X, R, g, ns, MUZ, KZ, SM, LOD)
            done += ns
            if SS is not None:
                update_pi()
            if H is not None:
                for k in range(K):
                    zk, hmu[k], htau[k] = H.update(g, T, np.log(X[k]), hmu[k].copy(), htau[k])
                    X[k] = np.exp(zk)
                MUZ[:], KZ[:] = hmu[:, H.bins], 0.5 / htau ** 2
                R = T.residual(X)
        if backend == 'c':
            if swaps:
                swaps_c(lib, T, X, R, rng, nacc, threads)
            if transfers:
                transfers_c(lib, T, X, R, rng, tstats, threads)
        elif swaps:
            swaps_numpy(T, X, R, g, nacc)
        if SS is not None:
            on = X > 0
            SM[on] = X[on]                               # an "on" rupture's slab value is its rate (swaps move rates)
            tr.setdefault('hyper_pi', np.zeros((K, n_rec, SS['B'])))[:, rec] = pib
            tr.setdefault('n_on', np.zeros((K, n_rec)))[:, rec] = on.sum(1)
        if H is not None:
            tr.setdefault('hyper_s', np.zeros((K, n_rec)))[:, rec] = htau / LN10
            tr.setdefault('hyper_mu', np.zeros((K, n_rec, H.B)))[:, rec] = hmu / LN10
        R = T.residual(X)                                    # exact recompute: no drift
        y = R + T.d
        tr['chi2'][:, rec] = np.sum(R ** 2, 1) / T.sigma ** 2
        Xm = X if sc is None else X * sc
        tr['sum_m'][:, rec] = Xm.sum(1)
        tr['tscale'][:, rec] = y @ T.d / dd
        tr['mon'][:, rec] = Xm[:, monitor]
        tr['sweep'][rec] = (rec + 1) * thin
        if rec >= n_burn:
            tr['y'][:, rec - n_burn] = y
            if samp is not None:
                samp[:, rec - n_burn] = Xm
            if KD is not None:
                for k, row in KD['at'].get(rec - n_burn, ()):
                    KD['X'][row] = Xm[k]
            if ST is not None:                           # running summaries of this draw
                t = rec - n_burn
                L = np.log10(np.maximum(Xm, stream['clip']))           # clipped log rate, for R-hat and effective draws
                h = 0 if t < n_keep // 2 else 1
                ST['s1'][:, h] += L
                ST['s2'][:, h] += L * L
                ST['cnt'][h] += 1
                b = min(t * nb_ // n_keep, nb_ - 1)
                ST['bsum'][:, b] += L
                ST['bcnt'][b] += 1
                ST['exc5'] += (Xm > 1e-5).sum(0)
                ST['exc4'] += (Xm > 1e-4).sum(0)
                ST['msum'] += Xm.sum(0)
                ST['usum'] += np.where(Xm < stream['det'], Xm, 0.0).sum(0)
                ST['n5'][:, t] = (Xm > 1e-5).sum(1)
                ST['n4'][:, t] = (Xm > 1e-4).sum(1)
                cum = np.cumsum(-np.sort(-Xm, axis=1), 1)      # each chain's rates, largest first, accumulated
                ST['n90'][:, t] = np.argmax(cum >= 0.9 * cum[:, -1:], 1) + 1     # fewest ruptures carrying 90% / 99%
                ST['n99'][:, t] = np.argmax(cum >= 0.99 * cum[:, -1:], 1) + 1    # of the draw's total rate
                if t % stream.get('hist_every', 1) == 0:              # log-rate histogram; bin 0 holds everything below hist_lo
                    Lh = np.log10(np.maximum(Xm, 1e-300))
                    idx = np.clip(np.floor((Lh - stream['hist_lo']) / stream['hist_w']).astype(np.int64) + 1, 0, stream['hist_nb'] - 1)
                    np.add.at(ST['hist'], (idx.ravel(), np.tile(np.arange(T.n), K)), 1)
                    ST['hcnt'][0] += K
                if 'tot_s' in ST:
                    ST['tot_s'][:, t] = (stream['inc_s'] @ Xm.T).T
                if 'tot_b' in ST:
                    ST['tot_b'][:, t] = (stream['inc_b'] @ Xm.T).T
        if verbose and (rec % max(1, n_rec // 20) == 0 or rec == n_rec - 1):
            el = time.time() - t0
            print(f'[gibbs:{backend}] sweep {(rec + 1) * thin:6d}/{sweeps}  chi2 per chain {np.round(tr["chi2"][:, rec], 0)}  '
                  f'pred/data scale {tr["tscale"][:, rec].mean():.3f}  ({el:.0f}s, {el / ((rec + 1) * thin):.3f}s/sweep)', flush=True)
    tr['final_X'] = X if sc is None else X * sc
    tr['seconds'] = time.time() - t0
    tr['swap_rate'] = nacc / max(1, n_rec * (len(T.pairs[0]) if swaps else 1)) if swaps else np.zeros(K)
    tr['transfer_stats'] = tstats       # per chain: moves made, moves involving a rate above 1e-6
    tr['monitor'] = monitor
    if out:
        extra = {}
        if samp is not None:
            samp.flush()
            extra = {'post_' + k: v for k, v in summarize_samples(samp).items()}
        np.savez(f'{out}.npz', **tr, **extra, burn=burn, thin=thin, backend=backend,
                 logbox=np.array(T.logbox if T.logbox is not None else [np.nan] * 3, float),
                 lognormal_bins=H.bins if H is not None else np.zeros(0, int))
        if ST is not None:
            np.savez_compressed(f'{out}_stream.npz', **ST, clip=stream['clip'], hist_lo=stream['hist_lo'], hist_w=stream['hist_w'],
                     det=stream['det'], n_keep=n_keep, chains=K)
        if KD is not None:
            KD['X'].flush()
            np.save(f'{out}_kept_index.npy', KD['index'])
    elif KD is not None:
        tr['kept'], tr['kept_index'] = KD['X'], KD['index']
    return tr


def report(tr, burn, thin, names=None):
    nb = burn // thin
    q = {'chi2': tr['chi2'][:, nb:], 'sum_m': tr['sum_m'][:, nb:], 'pred/data scale': tr['tscale'][:, nb:]}
    for j, idx in enumerate(tr['monitor']):
        q[f'm[{idx}]' if names is None else names[j]] = tr['mon'][:, nb:, j]
    print(f'{"quantity":>18s} {"mean":>11s} {"sd":>10s} {"q05":>10s} {"q95":>10s} {"Rhat":>6s} {"ESS":>7s}')
    if 'hyper_s' in tr:
        q['spread s (decades)'] = tr['hyper_s'][:, nb:]
    for k, v in q.items():
        print(f'{k:>18s} {v.mean():11.4g} {v.std():10.3g} {np.quantile(v, .05):10.3g} {np.quantile(v, .95):10.3g} '
              f'{float(split_rhat(v[:, :, None])[0]):6.3f} {ess(v):7.0f}')
    if 'hyper_mu' in tr:
        mu = tr['hyper_mu'][:, nb:, :]
        rh = split_rhat(mu)
        es = [ess(mu[..., k]) for k in range(mu.shape[2])]
        print(f'bin centres (log10): {mu.mean((0, 1)).min():.2f} to {mu.mean((0, 1)).max():.2f}; worst R-hat {np.nanmax(rh):.3f}, '
              f'smallest ESS {min(es):.0f}  (converged when every R-hat < 1.01)')
    if 'hyper_pi' in tr:                             # spike-and-slab: per-bin fraction switched on
        pi = tr['hyper_pi'][:, nb:, :]
        rh = split_rhat(pi)
        es = [ess(pi[..., k]) for k in range(pi.shape[2])]
        on = tr['n_on'][:, nb:]
        print(f'fraction switched on per bin: {pi.mean((0, 1)).min():.2e} to {pi.mean((0, 1)).max():.2e}; worst R-hat '
              f'{np.nanmax(rh):.3f}, smallest ESS {min(es):.0f}; ruptures on per draw {on.mean():.0f} '
              f'(R-hat {float(split_rhat(on[:, :, None])[0]):.3f}, ESS {ess(on):.0f})')


# ---------------------------------------------------------------- self-test against exact answers
def selftest(backends=('c', 'numpy')):
    from scipy.stats import truncnorm
    ok = True
    print('1) 1-D conditional 30 sd beyond the lower bound: N(-30, 1) on [0, 1]')
    T = Target(np.array([[1.0]]), np.array([-30.0]), 1.0, 0.0, 1.0)
    ex = truncnorm(30, 31, loc=-30, scale=1)
    for be in backends:
        tr = sample(T, np.zeros((4, 1)), 50000, 0, 1, backend=be, monitor=[0], verbose=False, keep_samples=False)
        x = tr['mon'][:, :, 0]
        se = x.std() / np.sqrt(ess(x))
        z = (x.mean() - ex.mean()) / se
        ok &= abs(z) < 4
        print(f'   {be:6s} mean {x.mean():.6f} (exact {ex.mean():.6f}, z = {z:+.2f}), sd {x.std():.6f} (exact {ex.std():.6f})')
    print('2) 2-D correlated, both walls active, flat and exponential prior, vs grid integration')
    G = np.array([[1.0, 0.8], [0.6, 1.0]])
    d = np.array([0.3, -0.1])
    for beta in [np.zeros(2), np.array([2.0, 0.5])]:
        T = Target(G, d, 0.4, 0.0, 1.0, beta)
        u = np.linspace(0, 1, 2001)
        A, B = np.meshgrid(u, u, indexing='ij')
        Rr = G[0, 0] * A + G[0, 1] * B - d[0], G[1, 0] * A + G[1, 1] * B - d[1]
        P = np.exp(-(Rr[0] ** 2 + Rr[1] ** 2) / (2 * 0.16) - beta[0] * A - beta[1] * B)
        P /= P.sum()
        ex = [np.sum(P * A), np.sum(P * B), np.sqrt(np.sum(P * A ** 2) - np.sum(P * A) ** 2)]
        for be in backends:
            tr = sample(T, np.full((4, 2), 0.5), 40000, 1000, 1, backend=be, monitor=[0, 1], verbose=False, keep_samples=False)
            x = tr['mon'][:, 1000:]
            zs = [(x[..., j].mean() - ex[j]) / (x[..., j].std() / np.sqrt(ess(x[..., j]))) for j in (0, 1)]
            ok &= all(abs(z) < 4 for z in zs)
            print(f'   beta={beta}  {be:6s} means ({x[..., 0].mean():.4f}, {x[..., 1].mean():.4f}) exact ({ex[0]:.4f}, {ex[1]:.4f}) '
                  f'z = ({zs[0]:+.2f}, {zs[1]:+.2f});  sd1 {x[..., 0].std():.4f} exact {ex[2]:.4f}')
    G = np.array([[1.0, 0.8], [0.6, 1.0]])
    d = np.array([0.3, -0.1])

    def lb_grid(G, d, s2, se, a=-3.0):
        u = np.linspace(a - 6 * se if se > 0 else a, 0.0, 2001)
        U1, U2 = np.meshgrid(u, u, indexing='ij')
        f = lambda x: np.where(x < a, np.exp(-(x - a) ** 2 / (2 * se ** 2)) if se > 0 else 0.0, 1.0)
        M1, M2 = 10 ** U1, 10 ** U2
        P = np.exp(-((G[0, 0] * M1 + G[0, 1] * M2 - d[0]) ** 2 + (G[1, 0] * M1 + G[1, 1] * M2 - d[1]) ** 2) / (2 * s2)) * f(U1) * f(U2)
        P /= P.sum()
        return [np.sum(P * U1), np.sum(P * U2)]

    def check(T, X0, ex, transform, label, n_c=40000, n_np=8000, swaps=False):
        nonlocal ok
        for be in backends:
            tr = sample(T, X0, n_c if be == 'c' else n_np, 1000, 1, backend=be, monitor=[0, 1], verbose=False,
                        keep_samples=False, swaps=swaps)
            x = transform(tr['mon'][:, 1000:])
            zs = [(x[..., j].mean() - ex[j]) / (x[..., j].std() / np.sqrt(ess(x[..., j]))) for j in (0, 1)]
            good = all(abs(z) < 4 for z in zs)
            ok &= good
            extra = f'; swap acceptance {tr["swap_rate"].mean():.2f}' if swaps else ''
            print(f'   {label:34s} {be:6s} ({x[..., 0].mean():+.4f}, {x[..., 1].mean():+.4f}) exact ({ex[0]:+.4f}, {ex[1]:+.4f}) '
                  f'z = ({zs[0]:+.2f}, {zs[1]:+.2f}){extra}')

    print('3) log-box prior (flat in log10 m on [-3, 0]), soft (sd 0.5 decade) and hard edges; E[log10 m] vs grid')
    for se in (0.5, 0.0):
        T = Target(G, d, 0.4, 0.0, 1.0, logbox=(-3, 0, se))
        check(T, np.full((4, 2), 0.1), lb_grid(G, d, 0.16, se), np.log10, f'Gaussian likelihood, edge sd {se}')
    print('4) swap moves between two near-duplicate columns, flat prior (E[m]) and log-box prior (E[log10 m]), vs grid')
    G2 = np.array([[1.0, 0.98], [0.5, 0.53]])
    d2 = np.array([0.4, 0.2])
    pairs = (np.array([0]), np.array([1]), np.array([G2[:, 0] @ G2[:, 1]]))
    u = np.linspace(0, 1, 2001)
    M1, M2 = np.meshgrid(u, u, indexing='ij')
    P = np.exp(-((G2[0, 0] * M1 + G2[0, 1] * M2 - d2[0]) ** 2 + (G2[1, 0] * M1 + G2[1, 1] * M2 - d2[1]) ** 2) / (2 * 0.09))
    P /= P.sum()
    check(Target(G2, d2, 0.3, 0.0, 1.0, pairs=pairs), np.tile([0.4, 0.0], (4, 1)), [np.sum(P * M1), np.sum(P * M2)],
          lambda x: x, 'flat prior + swaps', swaps=True)
    check(Target(G2, d2, 0.3, 0.0, 1.0, logbox=(-3, 0, 0.5), pairs=pairs), np.tile([0.4, 1e-3], (4, 1)),
          lb_grid(G2, d2, 0.09, 0.5), np.log10, 'log-box prior + swaps', swaps=True)
    print('5) transfer moves (one rate against two that nearly reproduce its column); 3-D grids and with/without check')
    from scipy.optimize import nnls
    from cru_common import make_dirs
    G3 = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.5, 0.3, 0.25]])
    d3 = np.array([0.5, 0.4, 0.3])
    c3, _ = nnls(G3[:, 1:], G3[:, 0])
    D3 = make_dirs(G3, [(np.array([0, 1, 2]), np.r_[1.0, -c3])])

    def grid3(lb, s2=0.09):
        u = np.linspace(-6.0, 0.0, 121) if lb else np.linspace(0.0, 1.0, 121)
        M = 10 ** u if lb else u
        f = np.where(u < -3, np.exp(-(u + 3) ** 2 / (2 * 0.25)), 1.0) if lb else np.ones_like(u)
        f = f * np.r_[0.5, np.ones(len(u) - 2), 0.5]          # trapezoid weights: the density need not vanish at the walls
        A, B, C = np.meshgrid(M, M, M, indexing='ij')
        chi = sum((G3[r, 0] * A + G3[r, 1] * B + G3[r, 2] * C - d3[r]) ** 2 for r in range(3))
        Pg = np.exp(-chi / (2 * s2)) * f[:, None, None] * f[None, :, None] * f[None, None, :]
        Pg /= Pg.sum()
        V = [np.log10(A), np.log10(B), np.log10(C)] if lb else [A, B, C]
        return [float(np.sum(Pg * v)) for v in V]

    def zs3(x, ex):
        return [(x[..., j].mean() - ex[j]) / (x[..., j].std() / np.sqrt(ess(x[..., j]))) for j in range(3)]
    for lb, label in [((-3, 0, 0.5), 'log-box prior'), (None, 'flat prior   ')]:
        ex = grid3(lb is not None)
        tf = np.log10 if lb is not None else (lambda v: v)
        out = []
        for use in (True, False):
            T = Target(G3, d3, 0.3, 0.0, 1.0, logbox=lb, dirs=D3)
            tr = sample(T, np.tile([0.2, 0.3, 0.2], (4, 1)), 40000, 1000, 1, backend='c', monitor=[0, 1, 2],
                        verbose=False, keep_samples=False, transfers=use)
            out.append(tf(tr['mon'][:, 1000:]))
        z = zs3(out[0], ex)
        ok &= all(abs(v) < 4 for v in z)
        print(f'   {label} + transfers: ({", ".join(f"{v:+.4f}" for v in out[0].mean((0, 1)))}) exact '
              f'({", ".join(f"{v:+.4f}" for v in ex)}) z = ({", ".join(f"{v:+.2f}" for v in z)}); '
              f'ESS of rate 1 with/without transfers: {ess(out[0][..., 0]):.0f} / {ess(out[1][..., 0]):.0f}')
    print('SELFTEST', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_zlo(backends=('c', 'numpy')):
    """Per-coordinate lower edges: a 2-rate problem whose rates have different lower edges, against a grid integral."""
    print('per-coordinate lower edges of the log-box prior: posterior means of log10 m vs grid integral')
    Gs = sp.csc_matrix(np.array([[1.0, 0.7], [0.5, 1.0]]))
    d, sig, lo10 = np.array([0.03, 0.01]), 0.03, np.array([-3.0, -6.0])
    ok = True
    for se in (0.5, 0.0):
        u = [np.linspace(l - (6 * se if se > 0 else 0), 0.0, 1601) for l in lo10]
        f = [np.where(ui < l, np.exp(-(l - ui) ** 2 / (2 * se ** 2)) if se > 0 else 0.0, 1.0) for ui, l in zip(u, lo10)]
        U1, U2 = np.meshgrid(u[0], u[1], indexing='ij')
        M1, M2 = 10 ** U1, 10 ** U2
        chi = ((Gs[0, 0] * M1 + Gs[0, 1] * M2 - d[0]) ** 2 + (Gs[1, 0] * M1 + Gs[1, 1] * M2 - d[1]) ** 2) / sig ** 2
        w = np.exp(-0.5 * (chi - chi.min())) * f[0][:, None] * f[1][None, :]
        w /= w.sum()
        ex = [float((w * U1).sum()), float((w * U2).sum())]
        for be in backends:
            T = Target(Gs, d, sig, 0.0, 1.0, logbox=(-3.0, 0.0, se))
            T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
            K, sw = (8, 40000) if be == 'c' else (4, 3000)
            tr = sample(T, np.full((K, 2), 1e-3), sw, sw // 10, 1, be, 1, 3, monitor=(0, 1), verbose=False, keep_samples=False)
            L = np.log10(tr['mon'][:, sw // 10:, :])
            for i in range(2):
                v = L[:, :, i]
                z = (v.mean() - ex[i]) / (v.std() / np.sqrt(ess(v)))
                ok &= abs(z) < 4
                print(f'   {be:5s} edge sd {se}: rate {i} (lower edge 1e{lo10[i]:g}): E[log10 m] {v.mean():.4f} (exact {ex[i]:.4f}, z {z:+.2f}); '
                      f'lowest draw 1e{v.min():.2f}')
            if se == 0:
                ok &= bool(L[:, :, 0].min() >= -3.0 - 1e-9) and bool(L[:, :, 1].min() >= -6.0 - 1e-9)
    print('SELFTEST (per-coordinate lower edges)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_edges(backends=('c', 'numpy')):
    """Per-coordinate soft upper edges: 2 rates with different lower edges and upper-edge widths, pushed above their
    tops by the data, against a grid integral."""
    print('per-coordinate upper edges of the log-box prior: posterior means of log10 m vs grid integral')
    Gs = sp.csc_matrix(np.array([[1.0, 0.7], [0.5, 1.0]]))
    d, sig, lo10, su = np.array([2.0, 1.8]), 0.1, np.array([-3.0, -6.0]), np.array([0.05, 0.15])
    ok = True
    for se in (0.5, 0.0):
        u = [np.linspace(l - (6 * se if se > 0 else 0), min(8 * s, 1.0), 2001) for l, s in zip(lo10, su)]
        f = []
        for ui, l, s in zip(u, lo10, su):
            fi = np.where(ui < l, np.exp(-(l - ui) ** 2 / (2 * se ** 2)) if se > 0 else 0.0, 1.0)
            f.append(np.where(ui > 0, np.exp(-ui ** 2 / (2 * s ** 2)), fi))
        U1, U2 = np.meshgrid(u[0], u[1], indexing='ij')
        M1, M2 = 10 ** U1, 10 ** U2
        chi = ((M1 + 0.7 * M2 - d[0]) ** 2 + (0.5 * M1 + M2 - d[1]) ** 2) / sig ** 2
        w = np.exp(-0.5 * (chi - chi.min())) * f[0][:, None] * f[1][None, :]
        w /= w.sum()
        ex = [float((w * U1).sum()), float((w * U2).sum())]
        for be in backends:
            T = Target(Gs, d, sig, 0.0, 10.0, logbox=(-3.0, 0.0, se))
            T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
            T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
            K, sw = (8, 40000) if be == 'c' else (4, 3000)
            tr = sample(T, np.full((K, 2), 0.5), sw, sw // 10, 1, be, 1, 5, monitor=(0, 1), verbose=False, keep_samples=False)
            L = np.log10(tr['mon'][:, sw // 10:, :])
            for i in range(2):
                v = L[:, :, i]
                z = (v.mean() - ex[i]) / (v.std() / np.sqrt(ess(v)))
                ok &= abs(z) < 4
                print(f'   {be:5s} lower edge sd {se}: rate {i} (upper-edge sd {su[i]} decades): E[log10 m] {v.mean():.4f} '
                      f'(exact {ex[i]:.4f}, z {z:+.2f}); highest draw 1e{v.max():.3f}')
    print('SELFTEST (per-coordinate upper edges)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_spikeslab(backends=('c', 'numpy')):
    """Spike-and-slab: 2 rates in one bin, with different slab floors and soft tops and the fraction on integrated out
    under Beta(1, 1); P(on) and E[log10 m | on] against exact values from enumerating the on/off states on a grid."""
    from scipy.special import betaln
    print('spike-and-slab prior: P(on) and E[log10 m | on] vs exact enumeration and grid integrals')
    Gs = sp.csc_matrix(np.array([[1.0, 0.7], [0.5, 1.0]]))
    d, sig, lo10, su, a_, b_ = np.array([0.6, 0.3]), 0.1, np.array([-3.0, -5.0]), np.array([0.05, 0.15]), 1.0, 1.0
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):                        # slab mass per grid node (trapezoid weights, normalized)
        w = np.full(len(ui), ui[1] - ui[0])
        w[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * w
        gw.append(f / f.sum())
    L = lambda m1, m2: np.exp(-0.5 * ((m1 + 0.7 * m2 - d[0]) ** 2 + (0.5 * m1 + m2 - d[1]) ** 2) / sig ** 2)
    pg = [np.exp(betaln(a_ + k, b_ + 2 - k) - betaln(a_, b_)) for k in range(3)]   # one on/off pattern with k on
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')
    L1, L2, L12 = L(10 ** u[0], 0.0), L(0.0, 10 ** u[1]), L(M1, M2) * gw[0][:, None] * gw[1][None, :]
    w00, w10, w01, w11 = pg[0] * L(0.0, 0.0), pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2), pg[2] * L12.sum()
    W = w00 + w10 + w01 + w11
    p_on = [(w10 + w11) / W, (w01 + w11) / W]
    e_u = [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
           (pg[1] * (gw[1] * u[1]) @ L2 + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)]
    ok = True
    for be in backends:
        T = Target(Gs, d, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.full((K, 2), 0.3), sw, sw // 10, 1, be, 1, 7, monitor=(0, 1), verbose=False, keep_samples=False)
        Mo = tr['mon'][:, sw // 10:, :]
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            p, e_on = on.mean(), ess(on)
            zp = (p - p_on[i]) / np.sqrt(p_on[i] * (1 - p_on[i]) / e_on)
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, e_on * p)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} rate {i} (slab floor 1e{lo10[i]:g}): P(on) {p:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); '
                  f'E[log10 m | on] {uu.mean():.4f} (exact {e_u[i]:.4f}, z {zu:+.2f})')
        pi = tr['hyper_pi'][:, sw // 10:, 0]
        print(f'   {be:5s} fraction on: mean {pi.mean():.4f} (exact {(a_ + p_on[0] + p_on[1]) / (a_ + b_ + 2):.4f})')
    print('SELFTEST (spike-and-slab)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_hier(backends=('c', 'numpy')):
    """Hierarchical lognormal prior: (1) rate updates with fixed hyperparameters vs a grid in log rate;
    (2) with the likelihood switched off, the full sampler must reproduce the hyperpriors."""
    ok = True
    print('6) hierarchical lognormal prior, hyperparameters fixed (mu = 0.1, s = 0.65 decade); E[ln m] vs grid')
    G = np.array([[1.0, 0.8], [0.6, 1.0]])
    d = np.array([0.3, -0.1])
    mu, tau = np.log(0.1), 1.5
    zz = np.linspace(mu - 9 * tau, mu + 9 * tau, 1601)
    A, B = np.meshgrid(zz, zz, indexing='ij')
    M1, M2 = np.exp(A), np.exp(B)
    Pg = np.exp(-((G[0, 0] * M1 + G[0, 1] * M2 - d[0]) ** 2 + (G[1, 0] * M1 + G[1, 1] * M2 - d[1]) ** 2) / (2 * 0.16)
                - ((A - mu) ** 2 + (B - mu) ** 2) / (2 * tau ** 2))
    Pg /= Pg.sum()
    ex = [np.sum(Pg * A), np.sum(Pg * B)]
    for be in backends:
        H = LogNormalHier(np.zeros(2, int), fixed=(np.array([np.log10(0.1)]), tau / LN10))
        T = Target(G, d, 0.4, 0.0, 1.0, lognormal=H)
        tr = sample(T, np.full((4, 2), 0.1), 40000 if be == 'c' else 8000, 1000, 1, backend=be, monitor=[0, 1],
                    verbose=False, keep_samples=False)
        x = np.log(tr['mon'][:, 1000:])
        zs = [(x[..., j].mean() - ex[j]) / (x[..., j].std() / np.sqrt(ess(x[..., j]))) for j in (0, 1)]
        ok &= all(abs(z) < 4 for z in zs)
        print(f'   {be:6s} ({x[..., 0].mean():+.4f}, {x[..., 1].mean():+.4f}) exact ({ex[0]:+.4f}, {ex[1]:+.4f}) z = ({zs[0]:+.2f}, {zs[1]:+.2f})')
    print('7) likelihood switched off: full hierarchical sampler (rates + centred + non-centred updates) vs hyperpriors')
    rng = np.random.default_rng(3)
    G9 = rng.uniform(0.2, 1.0, (3, 12))
    bins = np.repeat(np.arange(3), 4)
    for be, strong in [(be, st) for be in backends for st in (None, np.tile([True, True, False, False], 3))]:
        H = LogNormalHier(bins, s_range=(0.1, 2.0), mu_prior=(-6.0, 0.5), strong=strong)
        T = Target(G9, np.ones(3), 1e6, 0.0, 1.0, lognormal=H)
        nsw = 12000 if be == 'c' else 3000
        tr = sample(T, np.full((4, 12), 1e-6), nsw, 500, 1, backend=be, verbose=False, keep_samples=False)
        s = tr['hyper_s'][:, 500:]
        mub = tr['hyper_mu'][:, 500:, :]
        checks = [('E[s]', s, 1.05, None), ('sd[s]', s, None, np.sqrt(1.9 ** 2 / 12))] + \
                 [(f'E[mu_{k}]', mub[..., k], -6.0, None) for k in range(3)]
        out = []
        for name, x, mean, sd in checks:
            e = ess(x)
            if mean is not None:
                z = (x.mean() - mean) / (x.std() / np.sqrt(e))
                out.append(f'{name} {x.mean():.3f} (exact {mean:.3f}, z {z:+.2f})')
            else:
                z = (x.std() - sd) / (sd / np.sqrt(2 * e))
                out.append(f'{name} {x.std():.3f} (exact {sd:.3f}, z {z:+.2f})')
            ok &= abs(z) < 4
        print(f'   {be:6s} {"partial non-centring" if strong is not None else "full non-centring   "} ' + '; '.join(out))
    for be in backends:
        H = LogNormalHier(bins, s_range=(0.1, 2.0), mu_prior=(-6.0, 0.5), s_fixed=1.2)
        T = Target(G9, np.ones(3), 1e6, 0.0, 1.0, lognormal=H)
        tr = sample(T, np.full((4, 12), 1e-6), 12000 if be == 'c' else 3000, 500, 1, backend=be, verbose=False, keep_samples=False)
        mub = tr['hyper_mu'][:, 500:, :]
        zs = [(mub[..., k].mean() + 6.0) / (mub[..., k].std() / np.sqrt(ess(mub[..., k]))) for k in range(3)]
        zsd = [(mub[..., k].std() - 0.5) / (0.5 / np.sqrt(2 * ess(mub[..., k]))) for k in range(3)]
        ok &= all(abs(v) < 4 for v in zs + zsd) and np.allclose(tr['hyper_s'], 1.2)
        print(f'   {be:6s} spread held at 1.2: bin centres mean z ({", ".join(f"{v:+.2f}" for v in zs)}), sd z ({", ".join(f"{v:+.2f}" for v in zsd)})')
    print('SELFTEST (hierarchical)', 'PASSED' if ok else 'FAILED')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--problem', default='cru_clean.npz')
    ap.add_argument('--prelim', default=None, help='diagnostics from preliminaries.py (default: next to --problem)')
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--sweeps', type=int, default=4000)
    ap.add_argument('--burn', type=int, default=1000)
    ap.add_argument('--thin', type=int, default=4)
    ap.add_argument('--init', default='nnls', help="'nnls' (from prelim_results.npz), 'zero', a .npy of shape (n,) or (chains, n), or a previous run's .npz (uses its final states)")
    ap.add_argument('--prior', default='spikeslab',
                    help="default spikeslab (spike-and-slab per magnitude bin: each rupture off, rate 0, or on, log-uniform from the "
                         "slab floor 1e-10 up to a soft top at the bin's target rate R_j, sd = the bin's own uncertainty; the "
                         "fraction on in each bin estimated under a Beta(1, 1) prior); spikeslab:A:TOP:a:b "
                         "for slab floor 10^A, TOP = soft, hard or a factor F (hard top at F x R_j), and a Beta(a, b) prior; "
                         "binbox-theta:THETA:SE:TOP for per-bin log-uniform boxes below R_j, depths n_j / (THETA ln 10) decades, "
                         "lower edge SE (0 = hard); THETA = 1 puts no pressure on the bin totals. Kept for comparison: "
                         "binbox-floor:A:SE:TOP (per-bin boxes down to a common floor 10^A; the default until October 2026) and "
                         "binbox:W:SE:TOP (W decades below each R_j); 'logbox:A:B:SE' (one box for all ruptures, flat in log10 m "
                         "on [A, B] with half-Gaussian edges of sd SE decades, SE = 0: hard; e.g. logbox:-12:-2:1), "
                         "'logbox-centre:W:SE' (width W decades centred on the average data-imposed rate) and 'logbox-mean:W:SE' "
                         "(placed so its mean equals it); 'lognormal' (hierarchical, retired); 'flat'; a prior mean tau "
                         "(exponential prior) or a .npy of tau_i")
    ap.add_argument('--bins-block', default='MFD', help='lognormal prior: row block whose rows define the bins')
    ap.add_argument('--s-range', default='0.05,5', help='lognormal prior: uniform hyperprior range for s (decades)')
    ap.add_argument('--mu-prior', default='-6,10', help='lognormal prior: normal hyperprior (mean, sd) for each log10 centre')
    ap.add_argument('--hyper-every', type=int, default=None,
                    help='spike-and-slab and lognormal priors: update the hyperparameters (the fractions on; the centres and '
                         'spread) every this many sweeps (default: every sweep for spike-and-slab, every --thin sweeps for '
                         'lognormal)')
    ap.add_argument('--s-fixed', type=float, default=None,
                    help='lognormal prior: hold the within-bin spread at this value (decades) and learn only the bin centres')
    ap.add_argument('--pnc-pilot', default=None,
                    help='lognormal prior: a pilot run (.npz) used to pick the data-pinned rates for partial non-centring')
    ap.add_argument('--pnc-scale', type=float, default=1.0,
                    help='a rate counts as data-pinned if its pilot 95th percentile x ||G_i|| x spread / sigma exceeds this')
    ap.add_argument('--transfers', default=None,
                    help='transfer-direction library from build_transfers.py: one pass of exact transfer moves per recorded draw')
    ap.add_argument('--swap-cos', type=float, default=0.0,
                    help='add exact swap moves between columns with cosine similarity above this (>= 0.99; from prelim_results.npz)')
    ap.add_argument('--backend', default='c', choices=['c', 'numpy'])
    ap.add_argument('--threads', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='gibbs_run')
    ap.add_argument('--no-samples', action='store_true', help='do not write the (chains, draws, n) float32 sample file; the running '
                    'summaries in <out>_stream.npz are written either way')
    ap.add_argument('--diag-floor', type=float, default=-10.0, help='log10 rate below which rates count as equal in the running '
                    'R-hat and effective-draw summaries (default -10; the notebook passes its FLOOR)')
    ap.add_argument('--keep-draws', type=int, default=0, help='keep this many whole draws per run (divided evenly among the chains, '
                    'evenly spaced over the recorded draws after burn-in) in <out>_kept.npy, float32, with their chain and '
                    'record numbers in <out>_kept_index.npy; recording only, the chains are unchanged (select_draws.py uses them)')
    ap.add_argument('--slip-block', default='S', help='row block of slip-rate rows (subsection totals in the running summaries)')
    ap.add_argument('--hist-width', type=float, default=0.01, help='bin width, in decades, of the running log-rate histogram '
                    '(quantiles and modes are resolved to about this)')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        ok &= selftest_hier()
        ok &= selftest_zlo()
        ok &= selftest_edges()
        ok &= selftest_spikeslab()
        sys.exit(0 if ok else 1)
    kept = (a.sweeps - a.burn) // a.thin
    if kept < 4:
        raise SystemExit(f'only {max(kept, 0)} recorded draw(s) per chain after burn-in ((sweeps - burn) / thin = '
                         f'({a.sweeps} - {a.burn}) / {a.thin}); R-hat and effective draws need at least 4')
    P = load_problem(a.problem)
    n = P['G'].shape[1]
    logbox, beta, lognormal, scale, zlo_vec, bins, ku_vec, hi_u = None, None, None, None, None, None, None, 1.0
    spike_mode = False
    if a.prior == 'lognormal':
        Gr = P['G'].tocsr()
        rows = np.flatnonzero(P['block'] == P['block_names'].index(a.bins_block)) if a.bins_block in P['block_names'] else []
        bins = np.full(n, len(rows), np.int64)
        for k, j in enumerate(rows):
            bins[Gr.indices[Gr.indptr[j]:Gr.indptr[j + 1]]] = k
        bins = np.unique(bins, return_inverse=True)[1]
        lognormal = dict(bins=bins, s_range=tuple(float(v) for v in a.s_range.split(',')),
                         mu_prior=tuple(float(v) for v in a.mu_prior.split(',')), s_fixed=a.s_fixed)
        print(f'hierarchical lognormal prior: {bins.max() + 1} bins from block {a.bins_block!r}, s uniform on {a.s_range} decades')
        if a.pnc_pilot:
            pz = np.load(a.pnc_pilot)
            s_p = float(pz['hyper_s'].mean()) if 'hyper_s' in pz else 1.5
            strong = pz['post_q95'] * column_norms(P['G']) * (s_p * LN10) / P['sigma'] >= a.pnc_scale
            lognormal['strong'] = strong
            print(f'partial non-centring: {int(strong.sum())} data-pinned rates (from {a.pnc_pilot}, spread {s_p:.2f} decades) '
                  f'stay put when the hyperparameters move')
        beta = np.zeros(n)
    elif a.prior.startswith('binbox') or a.prior.startswith('spikeslab'):
        # per-bin log-uniform boxes, sampled as u = m / (F R_j) in a box below 1; spikeslab: each rupture is off (rate 0)
        # or on, with a log-uniform rate from the slab floor 10^A up to the bin's top, and a per-bin fraction on, estimated
        spike_mode = a.prior.startswith('spikeslab')
        theta_mode = a.prior.startswith('binbox-theta')
        floor_mode = a.prior.startswith('binbox-floor') or spike_mode  # a common floor 10^W for every bin
        if spike_mode:                                   # spikeslab:A:TOP:a:b (slab floor 10^A, A = -10 by default)
            sp_ = (a.prior.split(':') + [None] * 5)[1:5]
            W, se, top = float(sp_[0] or -10), 0.0, str(sp_[1] or 'soft').lower()
            ss_a, ss_b = float(sp_[2] or 1), float(sp_[3] or 1)
        else:
            parts = (a.prior.split(':') + [None] * 4)[1:4]
            W, se = float(parts[0] or (1 if theta_mode else -12 if floor_mode else 10)), float(parts[1] if parts[1] is not None else 0)
            top = str(parts[2] or 'soft').lower()        # 'soft' (default), 'hard', or a factor F for a hard top at F x R_j
        soft_top, F = (top == 'soft'), (1.0 if top in ('soft', 'hard') else float(top))
        Rb, rows = bin_targets(P, a.bins_block)
        Gr = P['G'].tocsr()
        bins = np.full(n, -1, np.int64)
        for k, j in enumerate(rows):
            bins[Gr.indices[Gr.indptr[j]:Gr.indptr[j + 1]]] = k
        if (bins < 0).any():
            raise SystemExit(f'{int((bins < 0).sum())} ruptures are in no row of block {a.bins_block!r}')
        if a.transfers:
            raise SystemExit('transfer moves are not implemented for binbox priors')
        scale = Rb[bins] * F
        nb = np.bincount(bins)
        edge = "soft lower edge, sd %g decades" % se if se > 0 else "hard lower edge"
        rel = 1.0 / P['d'][rows]                         # each bin row's relative uncertainty (it is normalized by it)
        su = np.log10(1.0 + rel) if soft_top else np.zeros(len(rows))     # upper-edge sd per bin, decades
        if soft_top:                                     # soft top at R_j: Gaussian in log rate with the bin's own uncertainty
            ku_vec = np.ascontiguousarray((1.0 / (2.0 * (su * LN10) ** 2))[bins])
            hi_u = 10.0
            edge += f'; soft top at R_j, sd log10(1 + sigma_j/R_j) = {su.min():.3f}-{su.max():.3f} decades'
        else:
            edge += f'; hard top at {F:g} x R_j'

        def box_mean(a, sl, sh):                         # mean of u = m / top under the box (log10 units)
            u = np.linspace(a - 10 * sl if sl > 0 else a, 8 * sh if sh > 0 else 0.0, 200001)
            f = np.where(u < a, np.exp(-(a - u) ** 2 / (2 * sl ** 2)) if sl > 0 else 0.0, 1.0)
            f = np.where(u > 0, np.exp(-u ** 2 / (2 * sh ** 2)) if sh > 0 else 0.0, f)
            return float(np.sum(f * 10.0 ** u) / np.sum(f))
        if theta_mode:                             # depth per bin: theta ruptures per unit ln(rate), W_j = n_j / (theta ln 10)
            Wj = nb / (W * LN10)
            zlo_vec = np.ascontiguousarray(-Wj[bins] * LN10)
            logbox = (-float(Wj.max()), 0.0, se)
            print(f'prior: per-bin log-uniform boxes below R_j, depths n_j / (theta ln 10) with theta = {W:g}: '
                  f'{Wj.min():.0f} to {Wj.max():.0f} decades ({edge}); prior bin totals have a flat density below the top '
                  f'when theta = 1 (mean about {W * F:g} x R_j)')
        elif floor_mode:                           # common floor: depth from each bin's top down to 10^W
            Wj = np.log10(Rb * F) - W
            if (Wj <= 0).any():
                raise SystemExit(f'the floor 1e{W:g} is not below every bin target (smallest {Rb.min():.2e})')
            zlo_vec = np.ascontiguousarray(-Wj[bins] * LN10)
            logbox = (-float(Wj.max()), 0.0, se)
            imp = nb * np.array([box_mean(-Wj[b], se, su[b]) for b in range(len(rows))]) * F
            if spike_mode:
                print(f'prior: spike-and-slab per magnitude bin: each rupture is off (rate 0) or on, with a log-uniform rate from '
                      f'the slab floor 1e{W:g} up to R_j ({Wj.min():.2f} to {Wj.max():.2f} decades; {edge}); the fraction on in '
                      f'each bin ~ Beta({ss_a:g}, {ss_b:g}), estimated')
            else:
                print(f'prior: per-bin log-uniform boxes from R_j down to a common floor 1e{W:g} (depths {Wj.min():.2f} to '
                      f'{Wj.max():.2f} decades; {edge}); tops {scale.min():.2e} to {scale.max():.2e}; implied prior bin totals '
                      f'{imp.min():.1f} to {imp.max():.1f} x their targets')
        else:
            logbox = (-W, 0.0, se)
            imp = nb * np.array([box_mean(-W, se, su[b]) for b in range(len(rows))]) * F
            print(f'prior: per-bin log-uniform boxes from R_j down {W:g} decades ({edge}); tops '
                  f'{scale.min():.2e} to {scale.max():.2e}; implied prior bin totals {imp.min():.1f} to {imp.max():.1f} x their targets')
    elif a.prior.startswith('logbox'):
        la, lb, lse, desc = resolve_logbox(a.prior, P, a.bins_block)
        logbox = (la, lb, lse)
        print('prior:', desc)
    else:
        beta = prior_rates(a.prior, n)
    fprelim = a.prelim or prelim_path(a.problem)
    pairs = None
    if a.swap_cos > 0:
        pr = np.load(fprelim)
        sel = pr['dup_cos'] > a.swap_cos
        if lognormal is not None:                  # swaps must leave the prior unchanged: same bin only
            sel &= lognormal['bins'][pr['dup_i']] == lognormal['bins'][pr['dup_j']]
        if zlo_vec is not None or ku_vec is not None:   # per-bin depths or tops: boxes differ between bins
            sel &= bins[pr['dup_i']] == bins[pr['dup_j']]
        pi, pj = pr['dup_i'][sel], pr['dup_j'][sel]
        pairs = (pi, pj, pr['dup_cos'][sel] * pr['cn'][pi] * pr['cn'][pj])
        if scale is not None:                      # inner products of the scaled columns
            pairs = (pi, pj, pairs[2] * scale[pi] * scale[pj])
        print(f'swap moves: {sel.sum()} near-duplicate pairs with cosine > {a.swap_cos}, one pass per recorded draw')
    dirs = dict(np.load(a.transfers)) if a.transfers else None
    if dirs is not None:
        print(f'transfer moves: {len(dirs["uu"])} directions from {a.transfers}, one pass per recorded draw')
    if scale is None:
        T = Target(P['G'], P['d'], P['sigma'], P['lo'], P['hi'], beta, logbox, pairs, dirs, lognormal)
    else:                                          # u = m / scale: columns scaled, box [1e-W, 1], u <= 1 is the hard top
        import scipy.sparse as _sp
        T = Target(sp_csc(P['G'] @ _sp.diags(scale)), P['d'], P['sigma'], 0.0, hi_u, beta, logbox, pairs, dirs, lognormal)
        T.scale = scale
        T.zlo_vec = zlo_vec
        T.ku_vec = ku_vec
        if spike_mode:                               # pkind 3: x_i = 0 (off) or its slab value (on)
            T.pkind = 3
            T.ss = dict(bins=bins, B=len(rows), nb=np.bincount(bins, minlength=len(rows)).astype(float), a=ss_a, b=ss_b)
    if a.init == 'nnls':
        X0 = np.tile(np.load(fprelim)['m_nnls'], (a.chains, 1))
    elif a.init == 'zero':
        X0 = np.zeros((a.chains, n))
    else:
        X0 = np.load(a.init)['final_X'] if a.init.endswith('.npz') else np.load(a.init)
        X0 = np.tile(X0, (a.chains, 1)) if X0.ndim == 1 else X0[np.arange(a.chains) % X0.shape[0]]
    mon = []
    if os.path.exists(fprelim):
        pr = np.load(fprelim)
        mon = [int(np.argmax(pr['m_nnls'])), int(np.argmin(pr['cn'])), int(np.argmax(pr['cn']))]
        if len(pr['dup_cos']):
            k = int(np.argmax(pr['dup_cos']))
            mon += [int(pr['dup_i'][k]), int(pr['dup_j'][k])]
    print(f'Gibbs: {a.chains} chains x {a.sweeps} sweeps (burn {a.burn}, thin {a.thin}), prior {a.prior}, backend {a.backend}, '
          f'threads {a.threads}; monitoring parameters {mon}')
    Gr = sp_csc(P['G']).tocsr()
    stream = dict(clip=10.0 ** a.diag_floor, hist_lo=-16.0, hist_w=a.hist_width, hist_nb=1 + int(round(16.5 / a.hist_width)), nbatch=20,
                  det=P['sigma'] / np.sqrt(np.asarray(Gr.multiply(Gr).sum(0)).ravel()))
    for key, blk in (('inc_s', a.slip_block), ('inc_b', a.bins_block)):
        if blk in P['block_names']:
            stream[key] = (Gr[np.flatnonzero(P['block'] == list(P['block_names']).index(blk))] > 0).astype(np.float64).tocsr()
    hyper_every = a.hyper_every if a.hyper_every is not None else (1 if spike_mode else None)   # fractions on: every sweep
    tr = sample(T, X0, a.sweeps, a.burn, a.thin, a.backend, a.threads, a.seed, mon, a.out, not a.no_samples,
                swaps=pairs is not None, transfers=dirs is not None, hyper_every=hyper_every, stream=stream,
                keep_draws=a.keep_draws)
    print(f'done in {tr["seconds"]:.0f}s -> {a.out}.npz, {a.out}_stream.npz' + ('' if a.no_samples else f', {a.out}_samples.npy')
          + (f', {a.out}_kept.npy (and _kept_index.npy)' if a.keep_draws > 0 else '')
          + (f'; swap acceptance {tr["swap_rate"].mean():.3f}' if pairs is not None else '')
          + (f'; transfer moves {tr["transfer_stats"][:, 0].sum()}, of which involving a rate > 1e-6: '
             f'{tr["transfer_stats"][:, 1].sum()}' if dirs is not None else ''))
    report(tr, a.burn, a.thin)


if __name__ == '__main__':
    main()

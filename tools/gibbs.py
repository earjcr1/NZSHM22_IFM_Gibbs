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
up to a soft top at its bin's target rate, and the fraction on in each bin is estimated. --slab sqrt makes an on
rupture's rate uniform in sqrt(rate) instead (density m^-1/2, not 1/m), the cross-check on how the lower end of the
rates is treated. (The per-bin log-uniform boxes, binbox, binbox-floor and binbox-theta, and the single log-box were
removed on November 23rd, 2026, with the linear and halfway priors.)
--down-the-middle adds a balance term to the misfit: each data type's relative shortfall s (summed predictions against
summed targets) costs A s^2, one A for every type (or, with --down-the-middle zero-model, each type's own, its zero-model
misfit), as extra rows of G and d, for any prior. Hard limits on rows: --all-positive (the notebook's) keeps every
slip-rate, paleo and MFD prediction above zero, and --mfd-monotonic and --plate-rate add constraint rows of their own
outside the likelihood (eps x the quantity kept, d = 0, eps = 1e-9, so they add nothing measurable to chi2): no magnitude
bin's total above its lower neighbour's where the targets do not rise, and each strip's share of the Pacific-Australia
plate rate within limits (plate_rate.py builds those rows from the rupture set's geometry and slips, on the strips of
cross-sections perpendicular to the plate motion that --slip-sum uses too, --cross-sections). Each update is drawn from
its conditional restricted to the rates that keep every limited row of its column inside its limits (band_bounds in the
kernel), spike-and-slab never takes an off or on state the limits forbid, a swap that would leave them is rejected, and
each recorded draw's count of rows outside them, always 0, is kept as band_outside. (The factor band, --band,
--slip-band, --paleo-band, and the soft band, --log-band, --slip-log-band, --paleo-log-band, were removed on November
23rd, 2026, the user's request: the misfits in log rate and years already make a prediction far off its target costly.)
--slip-sum adds a soft term beside the data, recorded apart from chi2 on chi2's scale: the slip along the plate motion
summed along each cross-section perpendicular to it against the targets' sums (rows of G and d, so the kernel is
unchanged). See --help for the rest.

Backends: a C kernel compiled on first use with $CC (default 'cc') and loaded through ctypes; chains run in
parallel threads (ctypes releases the GIL).  Each thread moves its chains through the ruptures together, reading each
column once for up to four of them, but every chain's arithmetic and random numbers are those it would have alone, so
the output does not depend on the number of threads.  Where the log-box prior is flat, the slice sampler's stepping out
takes the steps a bound shows would all pass without evaluating them, landing on the same slice.  Falls back to a
vectorised NumPy kernel (~100x slower) if no compiler is available, or on request with --backend numpy.
"""
import argparse
import ctypes
import hashlib
import math
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # numpy's BLAS idle threads spin-wait otherwise (no BLAS-heavy work
                                                     # here; the kernel has its own threads)
import numpy as np
import scipy.sparse as sp

from cru_common import load_problem, column_norms, prior_rates, split_rhat, ess, summarize_samples, bin_targets, prelim_path, report_to, balance_rows, magnitude_bins
import scipy.sparse as _sparse
sp_csc = _sparse.csc_matrix

LN10 = float(np.log(10.0))

C_SRC = r"""
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

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

/* ---- log-box prior: flat in z = log m on [zlo, zhi]; outside, half-Gaussian edges exp(-k (distance)^2), k <= 0: hard.
   Spike-and-slab's slab may be tilted: everything times exp(gam z), a density m^(gam - 1) in m (gam = 0: log-uniform;
   1/2: uniform in sqrt m). */
typedef struct { int pk; double m0, mu, inv2s2, zmin, zmax, zlo, zhi, k, ku, lp0, muz, kz, gam;
                 /* --cfm slip rates (sp = 1): the likelihood exactly, -sum w(e) e^2 sk over the rupture's column, e = sr_j +
                    G_ji (m - sxi), w = SL2_j below its target (e < 0) and SH2_j above; sL0 its value at m0 */
                 int sp; const double *sg; const int32_t *si; int64_t sp0, sp1; const double *sr, *SL2, *SH2;
                 double sxi, sk, sL0;
                 /* --slip-sum folded into the slip-rate rows (gibbs_run's U): the column's dot product with the summed slip
                    rates' residuals, sud, and its squared norm on those rows, snss (0 and 0 otherwise) */
                 double sud, snss; } lbctx;
/* --slip-misfit exact (since 2026-11-23): the exact misfits' tables, NULL without them. A row whose L2 is negative reads
   table t = -L2 - 1 (its H2 is not read): the cubic Hermite interpolant of the values SPT_F and slopes SPT_D from
   SPT_OFF[t] to SPT_OFF[t + 1] - 1, on the grid SPT_E0[t] + k / SPT_IH[t] in the row's residual, continued linearly beyond
   it. Each table is quasiconvex in the residual (falling, then rising: a unimodal density's -2 ln p, its slopes limited
   so the interpolant is monotone on each side; cru_common.slip_pdf_tables checks it). Set by sp_tables before each
   kernel call; read-only during it. */
static const double *SPT_E0 = NULL, *SPT_IH = NULL, *SPT_H = NULL, *SPT_FD = NULL;
static const int64_t *SPT_OFF = NULL;
/* FD: each point's value and slope times the step, interleaved (F_k, h D_k), so a lookup reads one stretch of memory;
   H: each table's step (no division per lookup) */
void sp_tables(const double *e0, const double *ih, const double *hh, const int64_t *off, const double *fd) {
    SPT_E0 = e0; SPT_IH = ih; SPT_H = hh; SPT_OFF = off; SPT_FD = fd;
}
/* --collapse (since 2026-11-24): spike-and-slab's fractions on integrated out. Each bin's fraction on is Beta(SSC_A, SSC_B)
   a priori, so with it integrated out a rupture is on, given the others, with prior odds (SSC_A + c) / (SSC_B + n_b - 1 - c),
   c the other ruptures on in its bin (n_b of them): the counts move freely, where drawing the fraction from the count and
   the count given the fraction moved each only a little at a time. Each chain's counts (CNT) are kept current. */
static int SSC_ON = 0;
static double SSC_A = 1.0, SSC_B = 1.0;
static const double *SSC_NB = NULL;
void ss_collapse(int64_t on, double a, double b, const double *nb) { SSC_ON = (int)on; SSC_A = a; SSC_B = b; SSC_NB = nb; }
static inline double ssc_odds(const int64_t *cn, int64_t b, int on_now) {   /* the prior odds of on, given the others */
    const double c = (double)(cn[b] - (on_now ? 1 : 0));
    return (SSC_A + c) / (SSC_B + SSC_NB[b] - 1.0 - c);
}
static inline double spt_f(int64_t t, double e) {
    const int64_t o = SPT_OFF[t], ng = SPT_OFF[t + 1] - o;
    const double *P = SPT_FD + 2 * o, e0 = SPT_E0[t];
    const double x = (e - e0) * SPT_IH[t];
    if (!(x > 0.0)) return P[0] + P[1] * x;                       /* continued linearly: F_0 + D_0 (e - e0) */
    const double xe = (double)(ng - 1);
    if (x >= xe) return P[2 * ng - 2] + P[2 * ng - 1] * (x - xe);
    const int64_t k = (int64_t)x;
    const double *q = P + 2 * k;
    const double s = x - (double)k, s2 = s * s, s3 = s2 * s;
    return (2.0 * s3 - 3.0 * s2 + 1.0) * q[0] + (s3 - 2.0 * s2 + s) * q[1] + (-2.0 * s3 + 3.0 * s2) * q[2]
           + (s3 - s2) * q[3];
}
/* w(e) e^2 with w = l below 0 and h above, without a branch (the sign of a residual is unpredictable):
   ((h + l) e^2 + (h - l) e |e|) / 2; or, with l < 0 (--slip-misfit exact), the row's table at e */
static inline double sp_w2(double e, double l, double h) {
    if (l < 0.0) return spt_f((int64_t)(-l) - 1, e);
    return 0.5 * ((h + l) * e * e + (h - l) * e * fabs(e));
}
/* sp_w2's slope in e: the table's interpolant's derivative (continued linearly), or 2 w e */
static inline double spt_d(int64_t t, double e) {
    const int64_t o = SPT_OFF[t], ng = SPT_OFF[t + 1] - o;
    const double *P = SPT_FD + 2 * o, ih = SPT_IH[t];
    const double x = (e - SPT_E0[t]) * ih;
    if (!(x > 0.0)) return P[1] * ih;
    const double xe = (double)(ng - 1);
    if (x >= xe) return P[2 * ng - 1] * ih;
    const int64_t k = (int64_t)x; const double *q = P + 2 * k; const double s = x - (double)k, s2 = s * s;
    return ((6.0 * s2 - 6.0 * s) * q[0] + (3.0 * s2 - 4.0 * s + 1.0) * q[1] + (6.0 * s - 6.0 * s2) * q[2]
            + (3.0 * s2 - 2.0 * s) * q[3]) * ih;
}
static inline double sp_d(double e, double l, double h) {
    if (l < 0.0) return spt_d((int64_t)(-l) - 1, e);
    return 2.0 * (e > 0.0 ? h : l) * e;
}
/* Bounds for an off rupture's change (since 2026-11-24, --off-bounds): with each row's term f_j, |f_j''| <= C_j
   everywhere (a table's largest over its cubic pieces, 0 on its linear continuations; 2 max(l, h) for a split normal),
   switching on at m changes sum f_j by m sum G_ji f_j'(e_j) +- m^2 SPB_QI[i], SPB_QI[i] = sum G_ji^2 C_j / 2 (Taylor's
   remainder). Each chain's slopes f_j'(e_j) are kept current (gibbs_run's DS), so a column's pass is one dot product
   per chain, and the exact change (table lookups) is computed only when the bounds leave the decision open. */
static const double *SPB_QI = NULL;
static const double *SPB_C = NULL;     /* each row's C_j (as in SPB_QI) */
static const int32_t *SPB_HO = NULL;   /* each column's likelihood entries, heaviest (G_ji^2 C_j) first: positions in data */
void sp_bounds(const double *qi, const double *c, const int32_t *ho) { SPB_QI = qi; SPB_C = c; SPB_HO = ho; }
/* the split misfit's log likelihood at m (lbctx's sp fields): concave in m, a Gaussian on each side of each row's target
   (with --slip-misfit exact's tables, each row's term quasiconvex in m instead: lb_floor) */
static double sp_lik(const lbctx *c, double m) {
    const double dm = m - c->sxi;
    double acc = 0.0;
    for (int64_t p = c->sp0; p < c->sp1; p++) {
        const int32_t j = c->si[p];
        const double e = c->sr[j] + c->sg[p] * dm;
        acc += sp_w2(e, c->SL2[j], c->SH2[j]);
    }
    return -(acc + dm * (2.0 * c->sud + dm * c->snss)) * c->sk;   /* the folded summed slip rates' rows: Gaussian */
}
static inline double lb_prior(const lbctx *c, double z) {
    if (c->pk == 2) return -(z - c->muz) * (z - c->muz) * c->kz;      /* hierarchical: log m ~ Normal(muz, 1/(2 kz)) */
    if (z < c->zlo) return (c->k > 0.0) ? -(c->zlo - z) * (c->zlo - z) * c->k + c->gam * z : -INFINITY;
    if (z > c->zhi) return (c->ku > 0.0) ? -(z - c->zhi) * (z - c->zhi) * c->ku + c->gam * z : -INFINITY;   /* own width */
    return c->gam * z;
}
/* log conditional density of z relative to the current point z0 (so it is exactly 0 at z0) */
static inline double lb_ld(const lbctx *c, double z) {
    if (!(z >= c->zmin && z <= c->zmax)) return -INFINITY;
    double p = lb_prior(c, z);
    if (p == -INFINITY) return -INFINITY;
    double m = exp(z);
    if (c->sp) return sp_lik(c, m) - c->sL0 + p - c->lp0;
    return -(m - c->m0) * (m + c->m0 - 2.0 * c->mu) * c->inv2s2 + p - c->lp0;
}
/* A lower bound on lb_ld(c, z) over a stretch [za, zb] of the box (zlo <= z <= zhi and zmin <= z <= zmax, where the
   log-box prior is 0, or the slab's tilt gam z), or less.  There lb_ld is the
   likelihood term, a concave quadratic in m = e^z, plus the prior's, less lp0, so it is at least the quadratic's
   smaller value at the two ends of the range of m plus the tilt's at the two ends of [za, zb].  The range is widened by 1e-12 (relative) and 1e-300
   (absolute), far more than any rounding of exp, and the bound lowered by 1e-9 of the terms' size, far more than the
   rounding of lb_ld itself (about 1e-15 of it): wherever the bound exceeds y, so does every evaluation of lb_ld. */
static double lb_floor(const lbctx *c, double za, double zb) {
    const double ma = fmax(exp(za) * (1.0 - 1e-12) - 1e-300, 0.0), mb = exp(zb) * (1.0 + 1e-12) + 1e-300;
    if (c->sp && SPT_FD) {   /* --slip-misfit exact: each row's term quasiconvex in m (a unimodal density's misfit, or the
                               split normal's convex one, of an affine e(m)), so on [ma, mb] at most its larger end's; the
                               folded summed slip rates' convex quadratic too: their sum is at most the sum of those */
        const double da = ma - c->sxi, db = mb - c->sxi;
        double acc = 0.0, sz = 0.0;
        for (int64_t p = c->sp0; p < c->sp1; p++) {
            const int32_t j = c->si[p];
            const double wa = sp_w2(c->sr[j] + c->sg[p] * da, c->SL2[j], c->SH2[j]);
            const double wb = sp_w2(c->sr[j] + c->sg[p] * db, c->SL2[j], c->SH2[j]);
            acc += fmax(wa, wb); sz += fabs(wa) + fabs(wb);
        }
        const double fa = da * (2.0 * c->sud + da * c->snss), fb = db * (2.0 * c->sud + db * c->snss);
        const double lmin = -(acc + fmax(fa, fb)) * c->sk;
        const double size = (sz + fabs(fa) + fabs(fb)) * c->sk + fabs(c->sL0) + fabs(c->lp0)
                            + fabs(c->gam) * fmax(fabs(za), fabs(zb));
        return lmin - c->sL0 + fmin(c->gam * za, c->gam * zb) - c->lp0 - 1e-9 * (size + 1.0);
    }
    if (c->sp) {                                      /* the split likelihood: concave in m, so at least its smaller end */
        const double la = sp_lik(c, ma), lb = sp_lik(c, mb);
        const double size = fabs(la) + fabs(lb) + fabs(c->sL0) + fabs(c->lp0) + fabs(c->gam) * fmax(fabs(za), fabs(zb));
        return fmin(la, lb) - c->sL0 + fmin(c->gam * za, c->gam * zb) - c->lp0 - 1e-9 * (size + 1.0);
    }
    const double fa = -(ma - c->m0) * (ma + c->m0 - 2.0 * c->mu) * c->inv2s2;
    const double fb = -(mb - c->m0) * (mb + c->m0 - 2.0 * c->mu) * c->inv2s2;
    const double size = (mb + fabs(c->m0)) * (mb + fabs(c->m0) + 2.0 * fabs(c->mu)) * c->inv2s2 + fabs(c->lp0)
                        + fabs(c->gam) * fmax(fabs(za), fabs(zb));
    return fmin(fa, fb) + fmin(c->gam * za, c->gam * zb) - c->lp0 - 1e-9 * (size + 1.0);
}
/* univariate slice sampler (Neal 2003): stepping out with width w (at most 64 steps), then shrinkage.  Under the
   log-box prior, once an end is inside the box, its steps over the stretch of the box it can still reach are taken
   without evaluating the density if lb_floor shows that every evaluation there would pass; each step is still the
   same subtraction or addition of w, so the slice is exactly the one evaluating every step gives. */
static double lb_slice(uint64_t *s, const lbctx *c, double z0, double w) {
    const double y = log(unif(s));
    double L = z0 - w * unif(s), R = L + w;
    int64_t J = (int64_t)(64.0 * unif(s)), Kk = 63 - J;
    const double a = fmax(c->zlo, c->zmin), b = fmin(c->zhi, c->zmax);
    /* quick: 0 not tried yet; 1 the steps within [lim, start] (left) or [start, lim] (right) pass; -1 evaluate each.
       The stretch is the part of the box this walk can still reach (its remaining steps, plus two widths of slack). */
    int quick = c->pk == 1 ? 0 : -1;
    double lim = 0.0;
    while (J-- > 0) {
        if (quick == 0 && L >= a && L <= b) {
            lim = fmax(a, L - (double)(J + 2) * w);
            quick = lb_floor(c, lim, L) > y ? 1 : -1;
        }
        if (quick == 1 && L >= lim) { L -= w; continue; }
        if (lb_ld(c, L) > y) L -= w; else break;
    }
    quick = c->pk == 1 ? 0 : -1;
    while (Kk-- > 0) {
        if (quick == 0 && R >= a && R <= b) {
            lim = fmin(b, R + (double)(Kk + 2) * w);
            quick = lb_floor(c, R, lim) > y ? 1 : -1;
        }
        if (quick == 1 && R <= lim) { R += w; continue; }
        if (lb_ld(c, R) > y) R += w; else break;
    }
    for (;;) {
        double z1 = L + (R - L) * unif(s);
        if (lb_ld(c, z1) > y) return z1;
        if (z1 < z0) L = z1; else R = z1;
    }
}

/* Exact draw (in z = log m) from the spike-and-slab's slab, the pseudo-prior of an "off" rupture: flat on [zlo, zhi]
   (hard floor), with a Gaussian shoulder above zhi of sd 1/sqrt(2 ku) up to zmax (ku > 0), or a hard top (ku = 0); a
   tilted slab has everything times exp(gam z).  Its constants q = (wp, wt, sdt, lim), the two parts' masses relative
   to the density at zhi and the shoulder's sd and length, depend only on the rupture: kernel_consts computes them once
   per run. */
static void slab_consts(const lbctx *c, double *q) {
    const double zt = fmin(c->zhi, c->zmax);
    const double wp = c->gam == 0.0 ? zt - c->zlo : (exp(c->gam * (zt - c->zhi)) - exp(c->gam * (c->zlo - c->zhi))) / c->gam;
    double wt = 0.0, sdt = 0.0, lim = 0.0;
    if (c->ku > 0.0 && c->zmax > c->zhi) {
        sdt = 1.0 / sqrt(2.0 * c->ku); lim = c->zmax - c->zhi;
        if (c->gam == 0.0)
            wt = sdt * 1.2533141373155003 * erf(lim / (sdt * 1.4142135623730951));   /* sdt sqrt(pi/2) erf(...) */
        else {        /* exp(gam t - ku t^2) = exp(gam^2 sdt^2 / 2) x a Gaussian centred at gam sdt^2, over [0, lim] */
            const double ct = c->gam * sdt * sdt;
            wt = exp(0.5 * c->gam * ct) * sdt * 1.2533141373155003
                 * (erf((lim - ct) / (sdt * 1.4142135623730951)) + erf(ct / (sdt * 1.4142135623730951)));
        }
    }
    q[0] = wp; q[1] = wt; q[2] = sdt; q[3] = lim;
}
/* The draw, as m = e^z.  Tilted, the flat part is drawn down from its top zt by inversion, z = zt + log(1 - u f) / gam with
   f = 1 - exp(-gam (zt - zlo)), which is gam wp when zt = zhi (always under spike-and-slab); for the square-root slab
   (gam = 1/2) that is m = e^zt (1 - u f)^2, with no logarithm. */
static double slab_draw_m(uint64_t *s, const lbctx *c, const double *q) {
    const double wp = q[0], wt = q[1], sdt = q[2], lim = q[3];
    if (unif(s) * (wp + wt) < wp) {
        if (c->gam == 0.0) return exp(c->zlo + wp * unif(s));
        const double zt = fmin(c->zhi, c->zmax);
        const double f = zt == c->zhi ? c->gam * wp : c->gam * wp * exp(c->gam * (c->zhi - zt));
        const double v = 1.0 - unif(s) * f;
        if (c->gam == 0.5) return (zt == 0.0 ? 1.0 : exp(zt)) * v * v;
        return exp(zt + log(v) / c->gam);
    }
    if (c->gam == 0.0)
        for (;;) { const double t = fabs(stdnorm(s)) * sdt; if (t <= lim) return exp(c->zhi + t); }
    const double ct = c->gam * sdt * sdt;
    for (;;) { const double t = ct + stdnorm(s) * sdt; if (t >= 0.0 && t <= lim) return exp(c->zhi + t); }
}

/* Per-rupture constants of gibbs_run, computed once per run by the expressions the sweeps used to evaluate for every
   rupture, chain and sweep (so with the same values): log lo (or -inf), log hi, and the slab's constants (4 per rupture;
   used by spike-and-slab only, whose upper-edge width defaults to 0 without KU, with its tilt gam). */
void kernel_consts(int64_t n, const double *lo, const double *hi, double zlo, double zhi, const double *ZLO,
                   const double *KU, double gam, double *LLO, double *LHI, double *SC)
{
    for (int64_t i = 0; i < n; i++) {
        LLO[i] = lo[i] > 0.0 ? log(lo[i]) : -INFINITY;
        LHI[i] = log(hi[i]);
        lbctx c;
        c.sp = 0;
        c.zlo = ZLO ? ZLO[i] : zlo; c.zhi = zhi; c.ku = KU ? KU[i] : 0.0; c.zmax = LHI[i]; c.gam = gam;
        slab_consts(&c, SC + 4 * i);
    }
}

/* G_i.r for up to four chains at once (r: their residuals), each summed in the column's order, exactly as for one chain
   alone: the sums are independent, so the processor overlaps them, and the column is read once for all of them */
static inline void col_dots(int64_t kg, const double *data, const int32_t *indices, int64_t p0, int64_t p1,
                            double *const *r, double *dt) {
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
    const double *r0 = r[0], *r1 = r[1], *r2 = r[2], *r3 = r[3];
    if (kg >= 4)
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            d0 += g * r0[j]; d1 += g * r1[j]; d2 += g * r2[j]; d3 += g * r3[j];
        }
    else if (kg == 3)
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            d0 += g * r0[j]; d1 += g * r1[j]; d2 += g * r2[j];
        }
    else if (kg == 2)
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            d0 += g * r0[j]; d1 += g * r1[j];
        }
    else
        for (int64_t p = p0; p < p1; p++) d0 += data[p] * r0[indices[p]];
    dt[0] = d0; dt[1] = d1; dt[2] = d2; dt[3] = d3;
}

/* The band (--band): every banded row's residual r_j kept within [RLO_j, RHI_j] (unbanded rows: -inf, +inf).  With the
   other rates fixed, x_i moves r_j by g (x_i - xi) on each row of its column, so the band allows x_i in [*lo, *hi].  The
   current value xi is always inside (a guard against rounding at a band's edge). */
/* --log-band: the change of the soft band's penalty, the sum over a column's penalized rows (LW > 0) of
   LW (ln(y / d))^2, between the predictions y + G_i d0 and y + G_i d1 (y = r + DY, the current prediction): +INFINITY if
   a prediction would not be above 0 at d1, -INFINITY if one is not at d0 (that state impossible) */
/* --paleo-misfit: the paleo rows' tables (NULL without it). Row j's table is I[j] (-1: none); table t is the half misfit
   f at v = ln y (y the row's prediction, as the row is scaled), the cubic Hermite interpolant of the values F and slopes D
   on the grid V0[t] + H k (k = 0 .. NG - 1), continued linearly beyond it; K[t] the largest |f''| of the interpolant */
typedef struct { const int32_t *I; const double *V0, *F, *D, *K; double H; int64_t NG; } pmctx;

static double pm_f(const pmctx *pm, int32_t t, double v, double *fp)
{
    const int64_t ng = pm->NG;
    const double *F = pm->F + (size_t)t * (size_t)ng, *D = pm->D + (size_t)t * (size_t)ng, h = pm->H;
    const double x = (v - pm->V0[t]) / h;
    if (!(x > 0.0)) { if (fp) *fp = D[0]; return F[0] + D[0] * (v - pm->V0[t]); }
    if (x >= (double)(ng - 1)) {
        if (fp) *fp = D[ng - 1];
        return F[ng - 1] + D[ng - 1] * (v - (pm->V0[t] + h * (double)(ng - 1)));
    }
    int64_t k = (int64_t)x;
    if (k > ng - 2) k = ng - 2;
    const double s = x - (double)k, s2 = s * s, s3 = s2 * s;
    if (fp) *fp = (6.0 * s2 - 6.0 * s) * (F[k] - F[k + 1]) / h + (3.0 * s2 - 4.0 * s + 1.0) * D[k] + (3.0 * s2 - 2.0 * s) * D[k + 1];
    return (2.0 * s3 - 3.0 * s2 + 1.0) * F[k] + (s3 - 2.0 * s2 + s) * h * D[k] + (-2.0 * s3 + 3.0 * s2) * F[k + 1]
           + (s3 - s2) * h * D[k + 1];
}

/* --log-band and --paleo-misfit: the change of the penalties, the sum over a column's penalized rows of LW (ln(y / d))^2
   (the soft band, LW > 0) and of the paleo rows' half misfits f(ln y) (PM), between the predictions y + G_i d0 and
   y + G_i d1 (y = r + DY, the current prediction): +INFINITY if a prediction would not be above 0 at d1, -INFINITY if
   one is not at d0 (that state impossible) */
static double lw_change(const double *data, const int32_t *indices, const int64_t *Q, int64_t p0, int64_t p1,
                        const double *r, const double *DY, const double *LW, const double *LWL, const double *LD, double d0,
                        double d1, const pmctx *PM)
{
    /* Q (or NULL): the weighted rows' entries (LWQ; then p0, p1 are the column's range in it, LWP[i], LWP[i + 1]): the
       same rows in the same order, without reading every entry of the column (2026-11-24) */
    double acc = 0.0;
    int bad0 = 0;
    for (int64_t q_ = p0; q_ < p1; q_++) {
        const int64_t p = Q ? Q[q_] : q_;
        const int32_t j = indices[p];
        const double w = LW[j];
        const int32_t t = PM ? PM->I[j] : -1;
        if (w <= 0.0 && t < 0) continue;
        const double y = r[j] + DY[j], y0 = y + data[p] * d0, y1 = y + data[p] * d1;
        if (y1 <= 0.0) return INFINITY;
        if (y0 <= 0.0) { bad0 = 1; continue; }
        if (w > 0.0) {
            const double l0 = log(y0) - LD[j], l1 = log(y1) - LD[j];
            if (!LWL) acc += w * (l1 * l1 - l0 * l0);
            else acc += (l1 < 0.0 ? LWL[j] : w) * l1 * l1 - (l0 < 0.0 ? LWL[j] : w) * l0 * l0;   /* sided weights */
        }
        if (t >= 0) acc += pm_f(PM, t, log(y1), NULL) - pm_f(PM, t, log(y0), NULL);
    }
    return bad0 ? -INFINITY : acc;
}

/* --log-band: the dot products G_i.r of col_dots and, for each chain, three sums over the column with its rows'
   bound coefficients A (4 per row: a1 = 2 w q / y, bhi = w (1 + max(-q, 0)) / y^2, blo = w max(q, 0) / y^2, 0; w the row's
   weight, q = ln(y / d), y the current prediction; 0 on unpenalized rows): s1 = sum g a1, s2 = sum g^2 bhi,
   s3 = sum g^2 blo. Each dot product is summed exactly as col_dots sums it (the same expression in the column's
   order), so it is the same number. The column is read once for all the chains. */
static inline void col_dots_lw(int64_t kg, const double *data, const int32_t *indices, int64_t p0, int64_t p1,
                               double *const *r, double *const *A, double *dt, double *s1, double *s2, double *s3) {
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, h0 = 0.0, h1 = 0.0, h2 = 0.0, h3 = 0.0;
    double l0 = 0.0, l1 = 0.0, l2 = 0.0, l3 = 0.0;
    const double *r0 = r[0], *r1 = r[1], *r2 = r[2], *r3 = r[3], *A0 = A[0], *A1 = A[1], *A2 = A[2], *A3 = A[3];
    if (kg >= 4)
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p]; const double gg = g * g;
            const double *e0 = A0 + 4 * j, *e1 = A1 + 4 * j, *e2 = A2 + 4 * j, *e3 = A3 + 4 * j;
            d0 += g * r0[j]; d1 += g * r1[j]; d2 += g * r2[j]; d3 += g * r3[j];
            a0 += g * e0[0]; a1 += g * e1[0]; a2 += g * e2[0]; a3 += g * e3[0];
            h0 += gg * e0[1]; h1 += gg * e1[1]; h2 += gg * e2[1]; h3 += gg * e3[1];
            l0 += gg * e0[2]; l1 += gg * e1[2]; l2 += gg * e2[2]; l3 += gg * e3[2];
        }
    else if (kg == 3)
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p]; const double gg = g * g;
            const double *e0 = A0 + 4 * j, *e1 = A1 + 4 * j, *e2 = A2 + 4 * j;
            d0 += g * r0[j]; d1 += g * r1[j]; d2 += g * r2[j];
            a0 += g * e0[0]; a1 += g * e1[0]; a2 += g * e2[0];
            h0 += gg * e0[1]; h1 += gg * e1[1]; h2 += gg * e2[1];
            l0 += gg * e0[2]; l1 += gg * e1[2]; l2 += gg * e2[2];
        }
    else if (kg == 2)
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p]; const double gg = g * g;
            const double *e0 = A0 + 4 * j, *e1 = A1 + 4 * j;
            d0 += g * r0[j]; d1 += g * r1[j];
            a0 += g * e0[0]; a1 += g * e1[0];
            h0 += gg * e0[1]; h1 += gg * e1[1];
            l0 += gg * e0[2]; l1 += gg * e1[2];
        }
    else
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p]; const double gg = g * g;
            const double *e0 = A0 + 4 * j;
            d0 += data[p] * r0[indices[p]];
            a0 += g * e0[0]; h0 += gg * e0[1]; l0 += gg * e0[2];
        }
    dt[0] = d0; dt[1] = d1; dt[2] = d2; dt[3] = d3;
    s1[0] = a0; s1[1] = a1; s1[2] = a2; s1[3] = a3;
    s2[0] = h0; s2[1] = h1; s2[2] = h2; s2[3] = h3;
    s3[0] = l0; s3[1] = l1; s3[2] = l2; s3[3] = l3;
}

/* --paleo-misfit: a paleo row's share of the bound coefficients, added to e: with t = G_ji m / y >= 0 and
   L = ln(1 + t) in [t - t^2/2, t], f(v + L) - f(v) lies in f'(v) L +- K L^2 / 2, so in [f' t - (max(f', 0) + K) t^2 / 2,
   f' t + (max(-f', 0) + K) t^2 / 2]: a1 += f' / y, bhi += (max(-f', 0) + K) / (2 y^2), blo += (max(f', 0) + K) / (2 y^2) */
static inline void pm_coef(double *e, const pmctx *pm, int32_t t, double y) {
    double fp;
    (void)pm_f(pm, t, log(y), &fp);
    const double iy = 1.0 / y, iy2 = iy * iy, kap = pm->K[t];
    e[0] += fp * iy; e[1] += 0.5 * ((fp < 0.0 ? -fp : 0.0) + kap) * iy2; e[2] += 0.5 * ((fp > 0.0 ? fp : 0.0) + kap) * iy2;
}

/* --log-band: a row's bound coefficients (col_dots_lw), from its current prediction y > 0. With sided weights (wl below
   the target, w above; --slip-misfit split+log10), f(q) = w(q) q^2 is convex with f' = 2 wc q continuous (wc the current
   side's) and f'' at most 2 max(w, wl), so f(q + L) - f(q) lies in [f' L, f' L + max(w, wl) L^2], and with L in
   [t - t^2/2, t]: a1 = 2 wc q / y, bhi = (wc max(-q, 0) + max(w, wl)) / y^2, blo = wc max(q, 0) / y^2 (w = wl: as above) */
static inline void lw_coef(double *e, double w, double wl, double y, double ld) {
    const double q = log(y) - ld, iy = 1.0 / y, iy2 = iy * iy;
    if (wl == w) {
        e[0] = 2.0 * w * q * iy; e[1] = w * (1.0 + (q < 0.0 ? -q : 0.0)) * iy2; e[2] = w * (q > 0.0 ? q : 0.0) * iy2; e[3] = 0.0;
    } else {
        const double wc = q < 0.0 ? wl : w, wm = wl > w ? wl : w;
        e[0] = 2.0 * wc * q * iy; e[1] = (wc * (q < 0.0 ? -q : 0.0) + wm) * iy2; e[2] = wc * (q > 0.0 ? q : 0.0) * iy2; e[3] = 0.0;
    }
}

/* --cfm slip rates: the change of the split misfit, sum w(e) e^2, between the residuals r + G_i d0 and r + G_i d1 */
static double sp_change(const double *data, const int32_t *indices, int64_t p0, int64_t p1, const double *r,
                        const double *SL2, const double *SH2, double d0, double d1, double ud, double nss) {
    double acc = 0.0;
    for (int64_t p = p0; p < p1; p++) {
        const int32_t j = indices[p];
        const double e0 = r[j] + data[p] * d0, e1 = r[j] + data[p] * d1;
        acc += sp_w2(e1, SL2[j], SH2[j]) - sp_w2(e0, SL2[j], SH2[j]);
    }
    /* --slip-sum folded (ud, nss: the column's dot product with those rows' residuals and its squared norm there; 0 and
       0 otherwise): their Gaussian terms' change, 2 (d1 - d0) ud + (d1^2 - d0^2) nss */
    return acc + (d1 - d0) * (2.0 * ud + (d1 + d0) * nss);
}
/* --slip-sum folded into the slip-rate rows: for up to four chains, the sum over the column's (likelihood) entries of
   G_ji U_j, U = B^T r_ss each chain's slip-rate rows' share of the summed slip rates' residuals (0 on other rows), which
   is the column's dot product with those rows, sum_s A_si r_s, since A = B G_slip */
static inline void col_udots(int64_t kg, const double *data, const int32_t *indices, int64_t p0, int64_t p1,
                             double *const *u, double *uq) {
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
    const double *u0 = u[0], *u1 = u[1], *u2 = u[2], *u3 = u[3];
    if (kg >= 4)
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            d0 += g * u0[j]; d1 += g * u1[j]; d2 += g * u2[j]; d3 += g * u3[j];
        }
    else if (kg == 3)
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            d0 += g * u0[j]; d1 += g * u1[j]; d2 += g * u2[j];
        }
    else if (kg == 2)
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            d0 += g * u0[j]; d1 += g * u1[j];
        }
    else
        for (int64_t p = p0; p < p1; p++) d0 += data[p] * u0[indices[p]];
    uq[0] = d0; uq[1] = d1; uq[2] = d2; uq[3] = d3;
}
/* --off-bounds: for 4 chains (the group's, repeated where it has fewer), each one's sum over the column's (likelihood)
   entries of G_ji DS_j, its rows' slopes, and, with u, of G_ji U_j (col_udots's), in one pass */
static void col_sdots(const double *data, const int32_t *indices, int64_t p0, int64_t p1, double *const *d,
                      double *const *u, double *sq, double *uq) {
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    const double *d0 = d[0], *d1 = d[1], *d2 = d[2], *d3 = d[3];
    if (u) {
        double b0 = 0.0, b1 = 0.0, b2 = 0.0, b3 = 0.0;
        const double *u0 = u[0], *u1 = u[1], *u2 = u[2], *u3 = u[3];
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            a0 += g * d0[j]; a1 += g * d1[j]; a2 += g * d2[j]; a3 += g * d3[j];
            b0 += g * u0[j]; b1 += g * u1[j]; b2 += g * u2[j]; b3 += g * u3[j];
        }
        uq[0] = b0; uq[1] = b1; uq[2] = b2; uq[3] = b3;
    } else
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            a0 += g * d0[j]; a1 += g * d1[j]; a2 += g * d2[j]; a3 += g * d3[j];
        }
    sq[0] = a0; sq[1] = a1; sq[2] = a2; sq[3] = a3;
}
/* --cfm slip rates: for each off chain q (pre[q]) of a group, the split misfit's change if it switched on at mq[q], DQ[q],
   in one pass over the column; with the soft band (A), also its bound sums s1, s2, s3 (col_dots_lw's). One function per
   group size, with the sums in registers (chains not off are given m = 0, so they add nothing to DQ). */
static void col_split1(const double *data, const int32_t *indices, int64_t p0, int64_t p1, double *const *r,
                       double *const *A, const int *pre, const double *mq, const double *SL2, const double *SH2, double *DQ,
                       double *s1, double *s2, double *s3, double *const *W0,
                       double *const *u, double *uq) {
    const double *r0 = r[0], m0 = pre[0] ? mq[0] : 0.0; double d0 = 0.0, a0 = 0.0, b0 = 0.0, c0 = 0.0;
    if (A && u) {         /* --slip-sum folded: each chain's G_i.U in the same pass */
        const double *A0 = A[0];
        const double *u0 = u[0]; double v0 = 0.0;
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p], gg = g * g; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double *e = A0 + 4 * j; a0 += g * e[0]; b0 += gg * e[1]; c0 += gg * e[2]; }
            v0 += g * u0[j];
        }
        uq[0] = v0;
    } else if (A) {
        const double *A0 = A[0];
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p], gg = g * g; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double *e = A0 + 4 * j; a0 += g * e[0]; b0 += gg * e[1]; c0 += gg * e[2]; }
        }
    } else if (u) {       /* --slip-sum folded without the bound sums (no soft band, or col_lwsums gives them) */
        const double *u0 = u[0]; double v0 = 0.0;
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            v0 += g * u0[j];
        }
        uq[0] = v0;
    } else
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
        }
    DQ[0] = d0; s1[0] = a0; s2[0] = b0; s3[0] = c0;
}
static void col_split2(const double *data, const int32_t *indices, int64_t p0, int64_t p1, double *const *r,
                       double *const *A, const int *pre, const double *mq, const double *SL2, const double *SH2, double *DQ,
                       double *s1, double *s2, double *s3, double *const *W0,
                       double *const *u, double *uq) {
    const double *r0 = r[0], m0 = pre[0] ? mq[0] : 0.0; double d0 = 0.0, a0 = 0.0, b0 = 0.0, c0 = 0.0;
    const double *r1 = r[1], m1 = pre[1] ? mq[1] : 0.0; double d1 = 0.0, a1 = 0.0, b1 = 0.0, c1 = 0.0;
    if (A && u) {         /* --slip-sum folded: each chain's G_i.U in the same pass */
        const double *A0 = A[0];
        const double *A1 = A[1];
        const double *u0 = u[0]; double v0 = 0.0;
        const double *u1 = u[1]; double v1 = 0.0;
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p], gg = g * g; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double *e = A0 + 4 * j; a0 += g * e[0]; b0 += gg * e[1]; c0 += gg * e[2]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            { const double *e = A1 + 4 * j; a1 += g * e[0]; b1 += gg * e[1]; c1 += gg * e[2]; }
            v0 += g * u0[j];
            v1 += g * u1[j];
        }
        uq[0] = v0;
        uq[1] = v1;
    } else if (A) {
        const double *A0 = A[0];
        const double *A1 = A[1];
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p], gg = g * g; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double *e = A0 + 4 * j; a0 += g * e[0]; b0 += gg * e[1]; c0 += gg * e[2]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            { const double *e = A1 + 4 * j; a1 += g * e[0]; b1 += gg * e[1]; c1 += gg * e[2]; }
        }
    } else if (u) {       /* --slip-sum folded without the bound sums (no soft band, or col_lwsums gives them) */
        const double *u0 = u[0]; double v0 = 0.0;
        const double *u1 = u[1]; double v1 = 0.0;
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            v0 += g * u0[j];
            v1 += g * u1[j];
        }
        uq[0] = v0;
        uq[1] = v1;
    } else
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
        }
    DQ[0] = d0; s1[0] = a0; s2[0] = b0; s3[0] = c0;
    DQ[1] = d1; s1[1] = a1; s2[1] = b1; s3[1] = c1;
}
static void col_split3(const double *data, const int32_t *indices, int64_t p0, int64_t p1, double *const *r,
                       double *const *A, const int *pre, const double *mq, const double *SL2, const double *SH2, double *DQ,
                       double *s1, double *s2, double *s3, double *const *W0,
                       double *const *u, double *uq) {
    const double *r0 = r[0], m0 = pre[0] ? mq[0] : 0.0; double d0 = 0.0, a0 = 0.0, b0 = 0.0, c0 = 0.0;
    const double *r1 = r[1], m1 = pre[1] ? mq[1] : 0.0; double d1 = 0.0, a1 = 0.0, b1 = 0.0, c1 = 0.0;
    const double *r2 = r[2], m2 = pre[2] ? mq[2] : 0.0; double d2 = 0.0, a2 = 0.0, b2 = 0.0, c2 = 0.0;
    if (A && u) {         /* --slip-sum folded: each chain's G_i.U in the same pass */
        const double *A0 = A[0];
        const double *A1 = A[1];
        const double *A2 = A[2];
        const double *u0 = u[0]; double v0 = 0.0;
        const double *u1 = u[1]; double v1 = 0.0;
        const double *u2 = u[2]; double v2 = 0.0;
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p], gg = g * g; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double *e = A0 + 4 * j; a0 += g * e[0]; b0 += gg * e[1]; c0 += gg * e[2]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            { const double *e = A1 + 4 * j; a1 += g * e[0]; b1 += gg * e[1]; c1 += gg * e[2]; }
            { const double e1 = r2[j] + g * m2; d2 += sp_w2(e1, l2, h2) - W0[2][j]; }
            { const double *e = A2 + 4 * j; a2 += g * e[0]; b2 += gg * e[1]; c2 += gg * e[2]; }
            v0 += g * u0[j];
            v1 += g * u1[j];
            v2 += g * u2[j];
        }
        uq[0] = v0;
        uq[1] = v1;
        uq[2] = v2;
    } else if (A) {
        const double *A0 = A[0];
        const double *A1 = A[1];
        const double *A2 = A[2];
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p], gg = g * g; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double *e = A0 + 4 * j; a0 += g * e[0]; b0 += gg * e[1]; c0 += gg * e[2]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            { const double *e = A1 + 4 * j; a1 += g * e[0]; b1 += gg * e[1]; c1 += gg * e[2]; }
            { const double e1 = r2[j] + g * m2; d2 += sp_w2(e1, l2, h2) - W0[2][j]; }
            { const double *e = A2 + 4 * j; a2 += g * e[0]; b2 += gg * e[1]; c2 += gg * e[2]; }
        }
    } else if (u) {       /* --slip-sum folded without the bound sums (no soft band, or col_lwsums gives them) */
        const double *u0 = u[0]; double v0 = 0.0;
        const double *u1 = u[1]; double v1 = 0.0;
        const double *u2 = u[2]; double v2 = 0.0;
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            { const double e1 = r2[j] + g * m2; d2 += sp_w2(e1, l2, h2) - W0[2][j]; }
            v0 += g * u0[j];
            v1 += g * u1[j];
            v2 += g * u2[j];
        }
        uq[0] = v0;
        uq[1] = v1;
        uq[2] = v2;
    } else
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            { const double e1 = r2[j] + g * m2; d2 += sp_w2(e1, l2, h2) - W0[2][j]; }
        }
    DQ[0] = d0; s1[0] = a0; s2[0] = b0; s3[0] = c0;
    DQ[1] = d1; s1[1] = a1; s2[1] = b1; s3[1] = c1;
    DQ[2] = d2; s1[2] = a2; s2[2] = b2; s3[2] = c2;
}
static void col_split4(const double *data, const int32_t *indices, int64_t p0, int64_t p1, double *const *r,
                       double *const *A, const int *pre, const double *mq, const double *SL2, const double *SH2, double *DQ,
                       double *s1, double *s2, double *s3, double *const *W0,
                       double *const *u, double *uq) {
    const double *r0 = r[0], m0 = pre[0] ? mq[0] : 0.0; double d0 = 0.0, a0 = 0.0, b0 = 0.0, c0 = 0.0;
    const double *r1 = r[1], m1 = pre[1] ? mq[1] : 0.0; double d1 = 0.0, a1 = 0.0, b1 = 0.0, c1 = 0.0;
    const double *r2 = r[2], m2 = pre[2] ? mq[2] : 0.0; double d2 = 0.0, a2 = 0.0, b2 = 0.0, c2 = 0.0;
    const double *r3 = r[3], m3 = pre[3] ? mq[3] : 0.0; double d3 = 0.0, a3 = 0.0, b3 = 0.0, c3 = 0.0;
    if (A && u) {         /* --slip-sum folded: each chain's G_i.U in the same pass */
        const double *A0 = A[0];
        const double *A1 = A[1];
        const double *A2 = A[2];
        const double *A3 = A[3];
        const double *u0 = u[0]; double v0 = 0.0;
        const double *u1 = u[1]; double v1 = 0.0;
        const double *u2 = u[2]; double v2 = 0.0;
        const double *u3 = u[3]; double v3 = 0.0;
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p], gg = g * g; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double *e = A0 + 4 * j; a0 += g * e[0]; b0 += gg * e[1]; c0 += gg * e[2]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            { const double *e = A1 + 4 * j; a1 += g * e[0]; b1 += gg * e[1]; c1 += gg * e[2]; }
            { const double e1 = r2[j] + g * m2; d2 += sp_w2(e1, l2, h2) - W0[2][j]; }
            { const double *e = A2 + 4 * j; a2 += g * e[0]; b2 += gg * e[1]; c2 += gg * e[2]; }
            { const double e1 = r3[j] + g * m3; d3 += sp_w2(e1, l2, h2) - W0[3][j]; }
            { const double *e = A3 + 4 * j; a3 += g * e[0]; b3 += gg * e[1]; c3 += gg * e[2]; }
            v0 += g * u0[j];
            v1 += g * u1[j];
            v2 += g * u2[j];
            v3 += g * u3[j];
        }
        uq[0] = v0;
        uq[1] = v1;
        uq[2] = v2;
        uq[3] = v3;
    } else if (A) {
        const double *A0 = A[0];
        const double *A1 = A[1];
        const double *A2 = A[2];
        const double *A3 = A[3];
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p], gg = g * g; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double *e = A0 + 4 * j; a0 += g * e[0]; b0 += gg * e[1]; c0 += gg * e[2]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            { const double *e = A1 + 4 * j; a1 += g * e[0]; b1 += gg * e[1]; c1 += gg * e[2]; }
            { const double e1 = r2[j] + g * m2; d2 += sp_w2(e1, l2, h2) - W0[2][j]; }
            { const double *e = A2 + 4 * j; a2 += g * e[0]; b2 += gg * e[1]; c2 += gg * e[2]; }
            { const double e1 = r3[j] + g * m3; d3 += sp_w2(e1, l2, h2) - W0[3][j]; }
            { const double *e = A3 + 4 * j; a3 += g * e[0]; b3 += gg * e[1]; c3 += gg * e[2]; }
        }
    } else if (u) {       /* --slip-sum folded without the bound sums (no soft band, or col_lwsums gives them) */
        const double *u0 = u[0]; double v0 = 0.0;
        const double *u1 = u[1]; double v1 = 0.0;
        const double *u2 = u[2]; double v2 = 0.0;
        const double *u3 = u[3]; double v3 = 0.0;
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            { const double e1 = r2[j] + g * m2; d2 += sp_w2(e1, l2, h2) - W0[2][j]; }
            { const double e1 = r3[j] + g * m3; d3 += sp_w2(e1, l2, h2) - W0[3][j]; }
            v0 += g * u0[j];
            v1 += g * u1[j];
            v2 += g * u2[j];
            v3 += g * u3[j];
        }
        uq[0] = v0;
        uq[1] = v1;
        uq[2] = v2;
        uq[3] = v3;
    } else
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double e1 = r1[j] + g * m1; d1 += sp_w2(e1, l2, h2) - W0[1][j]; }
            { const double e1 = r2[j] + g * m2; d2 += sp_w2(e1, l2, h2) - W0[2][j]; }
            { const double e1 = r3[j] + g * m3; d3 += sp_w2(e1, l2, h2) - W0[3][j]; }
        }
    DQ[0] = d0; s1[0] = a0; s2[0] = b0; s3[0] = c0;
    DQ[1] = d1; s1[1] = a1; s2[1] = b1; s3[1] = c1;
    DQ[2] = d2; s1[2] = a2; s2[2] = b2; s3[2] = c2;
    DQ[3] = d3; s1[3] = a3; s2[3] = b3; s3[3] = c3;
}
/* A group of 4 chains' rows interleaved (gibbs_run's RW): each row's residuals, current split-misfit terms (WC) and
   --slip-sum U for the 4 chains side by side (96 bytes), and each row's SL2/SH2 as a pair, so an entry of a column reads
   about 2 cache lines instead of 14; a copy of R, WC and U kept current where they change. */
typedef struct { double r[4], w[4], u[4]; } rw4;
typedef struct { double l, h; } lh2;
/* col_split4's branches without the bound sums (A NULL), over the interleaved copy: the same sums for each chain, in the
   same order, so the same results bit for bit (the compiler vectorizes them across chains) */
static void col_split4i(const double *data, const int32_t *indices, int64_t p0, int64_t p1, const rw4 *RW,
                        const lh2 *SLH, const int *pre, const double *mq, double *DQ, double *s1, double *s2, double *s3,
                        int with_u, double *uq) {
    const double m0 = pre[0] ? mq[0] : 0.0, m1 = pre[1] ? mq[1] : 0.0, m2 = pre[2] ? mq[2] : 0.0, m3 = pre[3] ? mq[3] : 0.0;
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
    if (with_u) {
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            const rw4 *x = RW + j; const double l2 = SLH[j].l, h2 = SLH[j].h;
            { const double e1 = x->r[0] + g * m0; d0 += sp_w2(e1, l2, h2) - x->w[0]; }
            { const double e1 = x->r[1] + g * m1; d1 += sp_w2(e1, l2, h2) - x->w[1]; }
            { const double e1 = x->r[2] + g * m2; d2 += sp_w2(e1, l2, h2) - x->w[2]; }
            { const double e1 = x->r[3] + g * m3; d3 += sp_w2(e1, l2, h2) - x->w[3]; }
            v0 += g * x->u[0];
            v1 += g * x->u[1];
            v2 += g * x->u[2];
            v3 += g * x->u[3];
        }
        uq[0] = v0; uq[1] = v1; uq[2] = v2; uq[3] = v3;
    } else
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p]; const int32_t j = indices[p];
            const rw4 *x = RW + j; const double l2 = SLH[j].l, h2 = SLH[j].h;
            { const double e1 = x->r[0] + g * m0; d0 += sp_w2(e1, l2, h2) - x->w[0]; }
            { const double e1 = x->r[1] + g * m1; d1 += sp_w2(e1, l2, h2) - x->w[1]; }
            { const double e1 = x->r[2] + g * m2; d2 += sp_w2(e1, l2, h2) - x->w[2]; }
            { const double e1 = x->r[3] + g * m3; d3 += sp_w2(e1, l2, h2) - x->w[3]; }
        }
    DQ[0] = d0; DQ[1] = d1; DQ[2] = d2; DQ[3] = d3;
    for (int64_t q = 0; q < 4; q++) s1[q] = s2[q] = s3[q] = 0.0;
}
static void col_split(int64_t kg, const double *data, const int32_t *indices, int64_t p0, int64_t p1, double *const *r,
                      double *const *A, const int *pre, const double *mq, const double *SL2, const double *SH2, double *DQ,
                      double *s1, double *s2, double *s3, double *const *W0, double *const *u, double *uq) {
    /* u (or NULL): --slip-sum folded, each chain's G_i.U summed in the same pass into uq (col_udots's sums), with or
       without A */
    for (int64_t q = 0; q < 4; q++) DQ[q] = s1[q] = s2[q] = s3[q] = 0.0;
    if (kg >= 4)      col_split4(data, indices, p0, p1, r, A, pre, mq, SL2, SH2, DQ, s1, s2, s3, W0, u, uq);
    else if (kg == 3) col_split3(data, indices, p0, p1, r, A, pre, mq, SL2, SH2, DQ, s1, s2, s3, W0, u, uq);
    else if (kg == 2) col_split2(data, indices, p0, p1, r, A, pre, mq, SL2, SH2, DQ, s1, s2, s3, W0, u, uq);
    else              col_split1(data, indices, p0, p1, r, A, pre, mq, SL2, SH2, DQ, s1, s2, s3, W0, u, uq);
}

/* --log-band and --paleo-misfit: col_dots_lw's bound sums s1, s2, s3 over only the column's entries in weighted rows
   (pos[q0:q1], the positions of those entries in the column, in its order): the other rows' coefficients are 0, so the
   sums are the same numbers, at a fraction of the reads when few rows are weighted (the paleo rows alone, say) */
static void col_lwsums1(const double *data, const int32_t *indices, const int64_t *pos, int64_t q0, int64_t q1,
                         double *const *A, double *s1, double *s2, double *s3) {
    const double *A0 = A[0]; double a0 = 0.0, h0 = 0.0, l0 = 0.0;
    for (int64_t t = q0; t < q1; t++) {
        const int64_t p = pos[t];
        const double g = data[p], gg = g * g; const int32_t j = indices[p];
        { const double *e = A0 + 4 * j; a0 += g * e[0]; h0 += gg * e[1]; l0 += gg * e[2]; }
    }
    s1[0] = a0; s2[0] = h0; s3[0] = l0;
}
static void col_lwsums2(const double *data, const int32_t *indices, const int64_t *pos, int64_t q0, int64_t q1,
                         double *const *A, double *s1, double *s2, double *s3) {
    const double *A0 = A[0]; double a0 = 0.0, h0 = 0.0, l0 = 0.0;
    const double *A1 = A[1]; double a1 = 0.0, h1 = 0.0, l1 = 0.0;
    for (int64_t t = q0; t < q1; t++) {
        const int64_t p = pos[t];
        const double g = data[p], gg = g * g; const int32_t j = indices[p];
        { const double *e = A0 + 4 * j; a0 += g * e[0]; h0 += gg * e[1]; l0 += gg * e[2]; }
        { const double *e = A1 + 4 * j; a1 += g * e[0]; h1 += gg * e[1]; l1 += gg * e[2]; }
    }
    s1[0] = a0; s2[0] = h0; s3[0] = l0;
    s1[1] = a1; s2[1] = h1; s3[1] = l1;
}
static void col_lwsums3(const double *data, const int32_t *indices, const int64_t *pos, int64_t q0, int64_t q1,
                         double *const *A, double *s1, double *s2, double *s3) {
    const double *A0 = A[0]; double a0 = 0.0, h0 = 0.0, l0 = 0.0;
    const double *A1 = A[1]; double a1 = 0.0, h1 = 0.0, l1 = 0.0;
    const double *A2 = A[2]; double a2 = 0.0, h2 = 0.0, l2 = 0.0;
    for (int64_t t = q0; t < q1; t++) {
        const int64_t p = pos[t];
        const double g = data[p], gg = g * g; const int32_t j = indices[p];
        { const double *e = A0 + 4 * j; a0 += g * e[0]; h0 += gg * e[1]; l0 += gg * e[2]; }
        { const double *e = A1 + 4 * j; a1 += g * e[0]; h1 += gg * e[1]; l1 += gg * e[2]; }
        { const double *e = A2 + 4 * j; a2 += g * e[0]; h2 += gg * e[1]; l2 += gg * e[2]; }
    }
    s1[0] = a0; s2[0] = h0; s3[0] = l0;
    s1[1] = a1; s2[1] = h1; s3[1] = l1;
    s1[2] = a2; s2[2] = h2; s3[2] = l2;
}
static void col_lwsums4(const double *data, const int32_t *indices, const int64_t *pos, int64_t q0, int64_t q1,
                         double *const *A, double *s1, double *s2, double *s3) {
    const double *A0 = A[0]; double a0 = 0.0, h0 = 0.0, l0 = 0.0;
    const double *A1 = A[1]; double a1 = 0.0, h1 = 0.0, l1 = 0.0;
    const double *A2 = A[2]; double a2 = 0.0, h2 = 0.0, l2 = 0.0;
    const double *A3 = A[3]; double a3 = 0.0, h3 = 0.0, l3 = 0.0;
    for (int64_t t = q0; t < q1; t++) {
        const int64_t p = pos[t];
        const double g = data[p], gg = g * g; const int32_t j = indices[p];
        { const double *e = A0 + 4 * j; a0 += g * e[0]; h0 += gg * e[1]; l0 += gg * e[2]; }
        { const double *e = A1 + 4 * j; a1 += g * e[0]; h1 += gg * e[1]; l1 += gg * e[2]; }
        { const double *e = A2 + 4 * j; a2 += g * e[0]; h2 += gg * e[1]; l2 += gg * e[2]; }
        { const double *e = A3 + 4 * j; a3 += g * e[0]; h3 += gg * e[1]; l3 += gg * e[2]; }
    }
    s1[0] = a0; s2[0] = h0; s3[0] = l0;
    s1[1] = a1; s2[1] = h1; s3[1] = l1;
    s1[2] = a2; s2[2] = h2; s3[2] = l2;
    s1[3] = a3; s2[3] = h3; s3[3] = l3;
}
static void col_lwsums(int64_t kg, const double *data, const int32_t *indices, const int64_t *pos, int64_t q0, int64_t q1,
                      double *const *A, double *s1, double *s2, double *s3) {
    for (int64_t q = 0; q < 4; q++) s1[q] = s2[q] = s3[q] = 0.0;
    if (kg >= 4)      col_lwsums4(data, indices, pos, q0, q1, A, s1, s2, s3);
    else if (kg == 3) col_lwsums3(data, indices, pos, q0, q1, A, s1, s2, s3);
    else if (kg == 2) col_lwsums2(data, indices, pos, q0, q1, A, s1, s2, s3);
    else              col_lwsums1(data, indices, pos, q0, q1, A, s1, s2, s3);
}

/* --mfd-step-sd: the MFD's steps held to the target's. With lam_b = ln y_b, y_b = r[row_b] + dy_b each magnitude bin's
   prediction (its total times its row's coefficient), each pair p of neighbouring bins (a, a + 1) has eps_p = lam_{a+1} -
   lam_a - e_p (e_p the target's own step, so eps_p is ln of the draw's step over the target's), and the term is
   S = sum_p (eps_p - mean eps)^2 / tau^2 = lam' A lam - 2 c' lam + const (A, c from Python), on chi2's scale: it
   multiplies the likelihood by exp(-S / 2). Every rupture is in at most one bin (bx; -1 none), with the coefficient gx
   there. Each chain keeps lam and H_b = sum_{k != b} A_bk lam_k - c_b, so that a change of bin b alone from y0 to y1
   changes S by (l1 - l0) (A_bb (l1 + l0) + 2 H_b), with l = ln y. */
typedef struct { int64_t NB; const int64_t *row; const double *A, *c, *dy; const int32_t *bx; const double *gx; int64_t cs; } stctx;
/* cs: c's stride per chain (0: one c for every chain; NB: each chain its own, --mfd-gr-tau, each draw's G-R curve) */
static void st_fill(const stctx *st, const double *r, double *lam, double *hh, int64_t k) {
    const int64_t NB = st->NB;
    const double *c = st->c + k * st->cs;
    for (int64_t b = 0; b < NB; b++) lam[b] = log(r[st->row[b]] + st->dy[b]);
    for (int64_t b = 0; b < NB; b++) {
        double h = -c[b];
        for (int64_t k = 0; k < NB; k++) if (k != b) h += st->A[b * NB + k] * lam[k];
        hh[b] = h;
    }
}
/* half the change of S (the log likelihood's share) when bin b's prediction goes from y0 to y1, the other bins fixed:
   +INFINITY if y1 is not above 0, -INFINITY if y0 is not (and y1 is) */
static inline double st_change(const stctx *st, const double *hh, int64_t b, double y0, double y1) {
    if (!(y1 > 0.0)) return INFINITY;
    if (!(y0 > 0.0)) return -INFINITY;
    const double l0 = log(y0), l1 = log(y1);
    return 0.5 * (l1 - l0) * (st->A[b * st->NB + b] * (l1 + l0) + 2.0 * hh[b]);
}
/* bounds on st_change(b, y, y + t y) for an off rupture switching on (t = g m / y > 0, l0 = lam_b = ln y): with
   u = ln(1 + t) in [t - t^2 / 2, t], the change is a convex quadratic in u, A u^2 / 2 + (A l0 + H) u; u's range is
   widened by 1e-12 (|l0| + 1) for the rounding of ln(y1) - ln(y0), and the bounds by 1e-9 of the terms' size */
static inline void st_bounds(const stctx *st, const double *lam, const double *hh, int64_t b, double t,
                             double *flo, double *fhi) {
    const double A = st->A[b * st->NB + b], l0 = lam[b], G = A * l0 + hh[b];
    const double w = 1e-12 * (fabs(l0) + 1.0);
    const double ulo = fmax(t - 0.5 * t * t, 0.0) - w, uhi = t + w;
    const double fa = (0.5 * A * ulo + G) * ulo, fb = (0.5 * A * uhi + G) * uhi;
    double lo = fmin(fa, fb), hi = fmax(fa, fb);
    if (A > 0.0) { const double us = -G / A; if (us > ulo && us < uhi) lo = -0.5 * G * G / A; }
    const double size = fabs(uhi) * (A * (fabs(l0) + fabs(uhi)) + fabs(hh[b])) + fabs(lo) + fabs(hi) + 1e-300;
    *flo = lo - 1e-9 * size; *fhi = hi + 1e-9 * size;
}
/* bin b's prediction is now y: lam_b and the other bins' H follow */
static inline void st_update(const stctx *st, double *lam, double *hh, int64_t b, double y) {
    const double l = log(y), dl = l - lam[b];
    lam[b] = l;
    for (int64_t k = 0; k < st->NB; k++) if (k != b) hh[k] += st->A[k * st->NB + b] * dl;
}
/* two penalty changes added, +INFINITY (impossible) winning over -INFINITY */
static inline double pen_add(double a, double b) { return (a == INFINITY || b == INFINITY) ? INFINITY : a + b; }
static inline void band_bounds(const double *data, const int32_t *indices, int64_t p0, int64_t p1, const double *r,
                               const double *RLO, const double *RHI, double xi, double *lo, double *hi) {
    double a = -INFINITY, b = INFINITY;
    for (int64_t p = p0; p < p1; p++) {
        const double g = data[p]; const int32_t j = indices[p];
        if (g > 0.0) { a = fmax(a, (RLO[j] - r[j]) / g); b = fmin(b, (RHI[j] - r[j]) / g); }
        else if (g < 0.0) { a = fmax(a, (RHI[j] - r[j]) / g); b = fmin(b, (RLO[j] - r[j]) / g); }
    }
    *lo = fmin(xi + a, xi); *hi = fmax(xi + b, xi);
}

/* K chains, chain-major: X[k*n + i], R[k*N + j] (R = G X - d, kept current), rng[4k..4k+3]; Gaussian likelihood.
   pkind 0: prior exp(-beta.m) on [lo, hi] (beta = 0: flat).  pkind 1: log-box prior (beta ignored), drawn by slice
   sampling z = log m.  The chains move through the ruptures together, but each is updated exactly as if it ran alone
   (its own residual, random numbers and order of operations), so the output does not depend on how chains are grouped.
   LLO, LHI, SC: per-rupture constants from kernel_consts.  gam: spike-and-slab's slab tilt (0: log-uniform).  RLO, RHI: the band (each row's residual kept within them), or NULL:
   each update is then drawn from its conditional restricted to the rates the band allows (band_bounds), and under
   spike-and-slab an "off" state the band forbids is never taken, nor an "on" one outside it. */
void gibbs_run(int64_t N, int64_t n, int64_t K, const int64_t *indptr, const int32_t *indices,
               const double *data, const double *inv_csq, const double *lo, const double *hi,
               const double *beta, double sigma, double *X, double *R, uint64_t *rng, int64_t nsweeps,
               int64_t pkind, double zlo, double zhi, double kedge, double slw,
               const double *MUZ, const double *KZ, const double *ZLO, const double *KU, double *SM, const double *LOD,
               const int64_t *SBIN, int64_t SB, const double *LLO, const double *LHI, const double *SC, int64_t *CNT,
               double gam, const double *RLO, const double *RHI,
               const double *LW, const double *LWL, const double *LD, const double *DY, const pmctx *PM, const double *SL2,
               const double *SH2, const int64_t *LWP, const int64_t *LWQ,
               const int64_t *PF, const int64_t *PC, int64_t NB, int64_t NSR, const int64_t *SSP, const int32_t *SSJ,
               const double *SSV, const double *NSS, const stctx *ST)
{
    /* PF (or NULL: each column's end): where each column's passes for the likelihood end, the entries after it in rows the
       passes need not read: the constraint rows (eps coefficients, held within their limits by band_bounds, which reads the
       whole column), and with SSP the summed slip rates' rows (rows NB .. NB + NSR - 1, entries PF[i] .. PC[i] - 1), folded
       into the slip-rate rows: A = B G_slip (B: SSP, SSJ, SSV, row by row; SSJ the slip-rate rows), so A_i.r_ss =
       G_i.U with U = B^T r_ss kept per chain (computed here from R, then updated as rates change), and NSS[i] = ||A_i||^2 */
    const double s2 = sigma * sigma;
    double *U = NULL;
    if (SSP) {
        U = (double *)calloc((size_t)K * (size_t)N, sizeof(double));
        for (int64_t k = 0; k < K; k++)
            for (int64_t sr = 0; sr < NSR; sr++) {
                const double rs = R[k * N + NB + sr];
                for (int64_t t = SSP[sr]; t < SSP[sr + 1]; t++) U[k * N + SSJ[t]] += SSV[t] * rs;
            }
    }
    const int COL = SSC_ON && pkind == 3 && CNT;   /* --collapse: each chain's counts, current from the start */
    if (COL)
        for (int64_t k = 0; k < K; k++) {
            int64_t *cn = CNT + k * SB;
            const double *x = X + k * n;
            for (int64_t b = 0; b < SB; b++) cn[b] = 0;
            for (int64_t i = 0; i < n; i++) if (x[i] > 0.0) cn[SBIN[i]]++;
        }
    double *WC = NULL;                             /* --cfm slip rates: each chain's rows' current split-misfit terms,
                                                      sp_w2 of their residuals, so a column pass computes only the new ones */
    if (SL2) {
        WC = (double *)malloc(sizeof(double) * (size_t)K * (size_t)N);
        for (int64_t k = 0; k < K; k++)
            for (int64_t j = 0; j < N; j++) WC[k * N + j] = sp_w2(R[k * N + j], SL2[j], SH2[j]);
    }
    double *DS = NULL;                             /* --off-bounds: each chain's rows' slopes, sp_d of their residuals */
    if (SL2 && pkind == 3 && SPB_QI) {
        DS = (double *)malloc(sizeof(double) * (size_t)K * (size_t)N);
        for (int64_t k = 0; k < K; k++)
            for (int64_t j = 0; j < N; j++) DS[k * N + j] = sp_d(R[k * N + j], SL2[j], SH2[j]);
    }
    rw4 *RW = NULL;                                /* --cfm slip rates: each full group of 4 chains' rows interleaved (rw4),
                                                      read by the split pass, written wherever R, WC or U change */
    lh2 *SLH = NULL;
    const int64_t G4 = K / 4;                      /* the full groups: chains 0 .. 4 G4 - 1 */
    if (SL2 && pkind == 3 && G4 > 0) {
        RW = (rw4 *)malloc(sizeof(rw4) * (size_t)G4 * (size_t)N);
        SLH = (lh2 *)malloc(sizeof(lh2) * (size_t)N);
        for (int64_t j = 0; j < N; j++) { SLH[j].l = SL2[j]; SLH[j].h = SH2[j]; }
        for (int64_t g = 0; g < G4; g++)
            for (int64_t j = 0; j < N; j++)
                for (int64_t q = 0; q < 4; q++) {
                    const int64_t k = 4 * g + q;
                    RW[g * N + j].r[q] = R[k * N + j];
                    RW[g * N + j].w[q] = WC[k * N + j];
                    RW[g * N + j].u[q] = U ? U[k * N + j] : 0.0;
                }
    }
    double *AC = NULL, *ODD = NULL;                /* --log-band: each chain's rows' bound coefficients (lw_coef), 4 a row,
                                                      and each chain's odds pi / (1 - pi) of a rupture on, per bin (also
                                                      with --mfd-step-sd alone) */
    if (LW) {
        AC = (double *)calloc((size_t)K * (size_t)N * 4, sizeof(double));
        for (int64_t k = 0; k < K; k++)
            for (int64_t j = 0; j < N; j++) {
                if (LW[j] > 0.0) lw_coef(AC + 4 * (k * N + j), LW[j], LWL ? LWL[j] : LW[j], R[k * N + j] + DY[j], LD[j]);
                if (PM && PM->I[j] >= 0) pm_coef(AC + 4 * (k * N + j), PM, PM->I[j], R[k * N + j] + DY[j]);
            }
    }
    if (LW || ST) {
        ODD = (double *)malloc(sizeof(double) * (size_t)K * (size_t)SB);
        for (int64_t k = 0; k < K; k++)
            for (int64_t b = 0; b < SB; b++) ODD[k * SB + b] = exp(LOD[k * SB + b]);
    }
    double *LAM = NULL, *HH = NULL;                /* --mfd-step-sd: each chain's bins' ln predictions and cross terms */
    if (ST) {
        LAM = (double *)malloc(sizeof(double) * (size_t)K * (size_t)ST->NB);
        HH = (double *)malloc(sizeof(double) * (size_t)K * (size_t)ST->NB);
        for (int64_t k = 0; k < K; k++) st_fill(ST, R + k * N, LAM + k * ST->NB, HH + k * ST->NB, k);
    }
    for (int64_t sw = 0; sw < nsweeps; sw++) {
      for (int64_t i = 0; i < n; i++) {
        const int64_t p0 = indptr[i], p1 = indptr[i + 1], pe = PF ? PF[i] : p1;   /* pe: the likelihood passes' end */
        const int empty = (p1 == p0 || inv_csq[i] <= 0.0);
        const double aa = empty ? 0.0 : 1.0 / inv_csq[i];
        for (int64_t k0 = 0; k0 < K; k0 += 4) {
            const int64_t kg = K - k0 < 4 ? K - k0 : 4;
            double dt[4] = {0.0, 0.0, 0.0, 0.0}, S1[4] = {0.0}, S2[4] = {0.0}, S3[4] = {0.0};
            double mpre[4] = {0.0}, DQ[4] = {0.0}, uq[4] = {0.0, 0.0, 0.0, 0.0}, DQL[4] = {0.0}, DQH[4] = {0.0}, SQV[4] = {0.0};
            int pre[4] = {0, 0, 0, 0};
            if (SL2 && pkind == 3 && !empty) {        /* --cfm slip rates: each off chain's slab draw first (its draws in
                                                         the same order as below), then its exact change in one pass */
                for (int64_t q = 0; q < kg; q++) {
                    const int64_t k = k0 + q;
                    if (X[k * n + i] != 0.0) continue;
                    lbctx c;
                    c.sp = 0;
                    c.pk = 1; c.muz = 0.0; c.kz = 0.0; c.gam = gam;
                    c.zlo = ZLO ? ZLO[i] : zlo; c.zhi = zhi; c.k = 0.0; c.ku = KU ? KU[i] : 0.0;
                    c.zmin = LLO[i];
                    c.zmax = LHI[i];
                    c.mu = 0.0; c.inv2s2 = 0.0;
                    mpre[q] = slab_draw_m(rng + 4 * k, &c, SC + 4 * i);
                    pre[q] = 1;
                }
                double *rg[4], *ag[4], *wg[4];
                for (int64_t q = 0; q < 4; q++) {
                    rg[q] = R + (k0 + (q < kg ? q : 0)) * N;
                    ag[q] = AC ? AC + 4 * (k0 + (q < kg ? q : 0)) * N : NULL;
                    wg[q] = WC + (k0 + (q < kg ? q : 0)) * N;
                }
                double *ug[4];                /* --slip-sum folded: each chain's U */
                for (int64_t q = 0; q < 4; q++) ug[q] = U ? U + (k0 + (q < kg ? q : 0)) * N : NULL;
                /* G_i.U summed in the split pass itself, whichever of its branches runs (no second pass over the column) */
                if (DS) {                     /* --off-bounds: the slopes' sums, and the change's bounds (below) */
                    double *dg[4], sq[4];
                    for (int64_t q = 0; q < 4; q++) dg[q] = DS + (k0 + (q < kg ? q : 0)) * N;
                    col_sdots(data, indices, p0, pe, dg, U ? ug : NULL, sq, uq);
                    for (int64_t q = 0; q < 4; q++) SQV[q] = sq[q];
                    if (AC) {
                        if (LWP) col_lwsums(kg, data, indices, LWQ, LWP[i], LWP[i + 1], ag, S1, S2, S3);
                        else col_dots_lw(kg, data, indices, p0, pe, rg, ag, dt, S1, S2, S3);
                    }
                    for (int64_t q = 0; q < kg; q++) {
                        if (!pre[q]) continue;
                        const double m = mpre[q], a = m * sq[q], b = m * m * SPB_QI[i];
                        const double w = 1e-9 * (fabs(a) + b) + 1e-300, uc = U ? m * (2.0 * uq[q] + m * NSS[i]) : 0.0;
                        DQL[q] = a - b - w + uc; DQH[q] = a + b + w + uc;
                    }
                } else if (RW && kg == 4 && (!AC || LWP)) {   /* a full group without the bound sums in the pass: the copy */
                    col_split4i(data, indices, p0, pe, RW + (k0 / 4) * N, SLH, pre, mpre, DQ, S1, S2, S3, U != NULL, uq);
                    if (AC) col_lwsums(kg, data, indices, LWQ, LWP[i], LWP[i + 1], ag, S1, S2, S3);
                } else if (AC && LWP) {       /* the bound sums over the weighted rows only (col_lwsums) */
                    col_split(kg, data, indices, p0, pe, rg, NULL, pre, mpre, SL2, SH2, DQ, S1, S2, S3, wg, U ? ug : NULL, uq);
                    col_lwsums(kg, data, indices, LWQ, LWP[i], LWP[i + 1], ag, S1, S2, S3);
                } else
                    col_split(kg, data, indices, p0, pe, rg, AC ? ag : NULL, pre, mpre, SL2, SH2, DQ, S1, S2, S3, wg,
                              U ? ug : NULL, uq);
                if (U) {                      /* the folded summed slip rates' rows: each off chain's Gaussian change */
                    for (int64_t q = 0; q < kg; q++)
                        if (pre[q]) DQ[q] += mpre[q] * (2.0 * uq[q] + mpre[q] * NSS[i]);
                }
            } else if (!empty) {
                double *rg[4];
                for (int64_t q = 0; q < 4; q++) rg[q] = R + (k0 + (q < kg ? q : 0)) * N;
                if (AC) {                     /* --log-band: the dot products and the bound sums in one pass */
                    double *ag[4];
                    for (int64_t q = 0; q < 4; q++) ag[q] = AC + 4 * (k0 + (q < kg ? q : 0)) * N;
                    if (LWP) {                /* or, with few weighted rows, the sums over those alone */
                        col_dots(kg, data, indices, p0, pe, rg, dt);
                        col_lwsums(kg, data, indices, LWQ, LWP[i], LWP[i + 1], ag, S1, S2, S3);
                    } else
                        col_dots_lw(kg, data, indices, p0, pe, rg, ag, dt, S1, S2, S3);
                } else
                    col_dots(kg, data, indices, p0, pe, rg, dt);
                if (U) {                      /* the folded summed slip rates' rows: their share of the dot products */
                    double *ug[4];
                    for (int64_t q = 0; q < 4; q++) ug[q] = U + (k0 + (q < kg ? q : 0)) * N;
                    col_udots(kg, data, indices, p0, pe, ug, uq);
                    for (int64_t q = 0; q < kg; q++) dt[q] += uq[q];
                }
            }
            for (int64_t q = 0; q < kg; q++) {
                const int64_t k = k0 + q;
                double *x = X + k * n, *r = R + k * N;
                uint64_t *s = rng + 4 * k;
                const double xi = x[i], dot = dt[q];
                double xn, blo = -INFINITY, bhi = INFINITY;     /* the rates the band allows x_i */
                /* the band's limits on x_i; under spike-and-slab an off rupture (x_i = 0, so its lower limit is at
                   most 0) can leave the band only by switching on, so its limits are found only then (below): the
                   same decisions and random numbers, without a pass over the column for each of the many off ruptures */
                const int off = (pkind == 3 && xi == 0.0);
                if (RLO && !empty && !off) band_bounds(data, indices, p0, p1, r, RLO, RHI, xi, &blo, &bhi);
                if (pkind == 3) {             /* spike-and-slab: x_i is 0 (off) or its slab value (on) */
                    lbctx c;
                    c.sp = 0;
                    c.pk = 1; c.muz = 0.0; c.kz = 0.0; c.gam = gam;
                    c.zlo = ZLO ? ZLO[i] : zlo; c.zhi = zhi; c.k = 0.0; c.ku = KU ? KU[i] : 0.0;
                    c.zmin = LLO[i];
                    c.zmax = LHI[i];
                    c.mu = 0.0; c.inv2s2 = 0.0;
                    double m;
                    if (xi > 0.0) {           /* on: exact slice update of the slab value under likelihood x slab */
                        const double z0 = log(xi);
                        c.m0 = xi; c.lp0 = lb_prior(&c, z0);
                        if (!empty && SL2) {      /* --cfm slip rates: the likelihood evaluated exactly */
                            c.sp = 1; c.sg = data; c.si = indices; c.sp0 = p0; c.sp1 = pe; c.sr = r; c.SL2 = SL2; c.SH2 = SH2;
                            c.sud = U ? uq[q] : 0.0; c.snss = U ? NSS[i] : 0.0;
                            c.sxi = xi; c.sk = 0.5 / s2; c.sL0 = sp_lik(&c, xi);
                        } else if (!empty) { c.mu = xi - dot / aa; c.inv2s2 = 0.5 * aa / s2; }
                        if (RLO) {            /* the band: the slice within the rates it allows (z0 always among them) */
                            if (blo > 0.0) c.zmin = fmin(fmax(c.zmin, log(blo)), z0);
                            c.zmax = fmax(fmin(c.zmax, log(bhi)), z0);
                        }
                        m = exp(lb_slice(s, &c, z0, slw));
                        /* --log-band: the slice step, reversible under the rest of the target, is a proposal that
                           the soft band's change accepts or not (Metropolis), so the update stays exact */
                        if ((LW || ST) && !empty && m != xi) {
                            double dp = LW ? (LWP ? lw_change(data, indices, LWQ, LWP[i], LWP[i + 1], r, DY, LW, LWL, LD, 0.0, m - xi, PM)
                                    : lw_change(data, indices, NULL, p0, pe, r, DY, LW, LWL, LD, 0.0, m - xi, PM)) : 0.0;
                            if (ST && ST->bx[i] >= 0) {   /* --mfd-step-sd: its change, from x_i to m */
                                const int64_t b = ST->bx[i];
                                const double y = r[ST->row[b]] + ST->dy[b];
                                dp = pen_add(dp, st_change(ST, HH + k * ST->NB, b, y, y + ST->gx[i] * (m - xi)));
                            }
                            if (!(log(unif(s)) < -dp)) m = xi;
                        }
                    } else {                  /* off: a fresh draw from the slab, the pseudo-prior */
                        m = pre[q] ? mpre[q] : slab_draw_m(s, &c, SC + 4 * i);   /* --cfm slip rates: drawn above */
                    }
                    /* on or off given m: log odds = log(pi/(1-pi)) + log L(m) - log L(0); with the rupture switched off
                       the residual is r0 = r - x_i G_i, so G_i.r0 = dot - x_i ||G_i||^2 */
                    const double odd0 = COL ? ssc_odds(CNT + k * SB, SBIN[i], xi > 0.0) : (ODD ? ODD[k * SB + SBIN[i]] : 0.0);
                    double lr = COL ? log(odd0) : LOD[k * SB + SBIN[i]];   /* the prior's log odds of on (--collapse:
                                                                                     from the counts; else the fraction's) */
                    double xl = 0.0;          /* the likelihood's part of the log odds, lr - LOD */
                    const int bq = !empty && SL2 && DS && pre[q];   /* --off-bounds: xl only bounded so far */
                    if (bq) {
                    } else if (!empty && SL2) {   /* --cfm slip rates: the split misfit's change, exactly */
                        xl = (pre[q] ? DQ[q] : sp_change(data, indices, p0, pe, r, SL2, SH2, -xi, m - xi,
                                                         U ? uq[q] : 0.0, U ? NSS[i] : 0.0)) / (2.0 * s2);
                        lr -= xl;
                    } else if (!empty) { const double d0 = dot - xi * aa; xl = (2.0 * m * d0 + m * m * aa) / (2.0 * s2); lr -= xl; }
                    if (bq) {
                        /* --off-bounds, an off chain: on when xl + dp < g = lr - logit(u), dp the soft band's penalty (if
                           any). xl lies in [xlo, xhi] (the slopes' bounds) and dp in [dlo, dhi] (the band's), so most
                           decisions are settled without the exact change, many without logarithms; otherwise xl, and then
                           if need be dp, is computed exactly. One uniform, as below: the same decisions */
                        const double uu = unif(s);
                        const double xlo = DQL[q] / (2.0 * s2), xhi = DQH[q] / (2.0 * s2);
                        double dlo = 0.0, dhi = 0.0;
                        if (LW || ST) {
                            const double sm1 = m * S1[q], mm = m * m;
                            dlo = sm1 - mm * S3[q]; dhi = sm1 + mm * S2[q];
                            dlo -= 1e-9 * (fabs(dlo) + 1e-12); dhi += 1e-9 * (fabs(dhi) + 1e-12);
                            if (ST && ST->bx[i] >= 0) {
                                const int64_t b = ST->bx[i];
                                const double y = r[ST->row[b]] + ST->dy[b];
                                double flo, fhi;
                                st_bounds(ST, LAM + k * ST->NB, HH + k * ST->NB, b, ST->gx[i] * m / y, &flo, &fhi);
                                dlo += flo; dhi += fhi;
                            }
                        }
                        int on = -1;
                        const double X = -xlo - dlo;   /* on needs u < odds exp(-xl - dp) <= odds exp(X) */
                        if (X <= 1.0) {
                            const double E = X <= 0.0 ? 1.0 + X + 0.5 * X * X : 1.0 + X + X * X;
                            const double od = (COL || ODD) ? odd0 : exp(lr);
                            if (uu >= od * E * (1.0 + 1e-9)) on = 0;
                        }
                        double g = 0.0;
                        if (on < 0) {
                            g = lr - (log(uu) - log1p(-uu));
                            if (xhi + dhi < g) on = 1; else if (xlo + dlo >= g) on = 0;
                        }
                        if (on < 0 && SPB_HO && WC) {
                            /* the exact change heaviest rows first, each row's new term less its cached one (WC), with
                               the rest still bounded by their slopes and curvatures (2026-11-24): checked after 1, 2, 4,
                               8, ... rows, stopping once the decision is settled (most fallbacks are switched on far
                               above the threshold, their change dominated by a few rows); otherwise the whole change */
                            const double f2 = 1.0 / (2.0 * s2);
                            const double *dsk = DS + k * N, *wck = WC + k * N;
                            double ex = U ? m * (2.0 * uq[q] + m * NSS[i]) : 0.0, sl = SQV[q], cv = SPB_QI[i];
                            const double w0 = fabs(m * SQV[q]) + m * m * SPB_QI[i];
                            const int64_t ne = pe - p0;
                            int64_t next = 1;
                            for (int64_t t = 0; t < ne; t++) {
                                const int64_t p = SPB_HO[p0 + t];
                                const int32_t j = indices[p];
                                const double gg = data[p];
                                ex += sp_w2(r[j] + gg * m, SL2[j], SH2[j]) - wck[j];
                                sl -= gg * dsk[j];
                                cv -= 0.5 * gg * gg * SPB_C[j];
                                if (t + 1 == next && t + 1 < ne) {
                                    next *= 2;
                                    const double a = m * sl, b = m * m * (cv > 0.0 ? cv : 0.0);
                                    const double w = 1e-9 * (fabs(ex) + fabs(a) + b + w0) + 1e-300;
                                    if ((ex + a + b + w) * f2 + dhi < g) { on = 1; break; }
                                    if ((ex + a - b - w) * f2 + dlo >= g) { on = 0; break; }
                                }
                            }
                            if (on < 0) {
                                xl = ex * f2;
                                if (xl + dhi < g) on = 1; else if (xl + dlo >= g) on = 0;
                            }
                        } else if (on < 0) {
                            xl = sp_change(data, indices, p0, pe, r, SL2, SH2, -xi, m - xi, U ? uq[q] : 0.0,
                                           U ? NSS[i] : 0.0) / (2.0 * s2);
                            if (xl + dhi < g) on = 1; else if (xl + dlo >= g) on = 0;
                        }
                        if (on < 0) {
                            double dp = LW ? (LWP ? lw_change(data, indices, LWQ, LWP[i], LWP[i + 1], r, DY, LW, LWL, LD, -xi, m - xi, PM)
                                    : lw_change(data, indices, NULL, p0, pe, r, DY, LW, LWL, LD, -xi, m - xi, PM)) : 0.0;
                            if (ST && ST->bx[i] >= 0) {
                                const int64_t b = ST->bx[i];
                                const double y = r[ST->row[b]] + ST->dy[b];
                                dp = pen_add(dp, st_change(ST, HH + k * ST->NB, b, y - ST->gx[i] * xi, y + ST->gx[i] * (m - xi)));
                            }
                            on = xl + dp < g;
                        }
                        xn = on ? m : 0.0;
                    } else if ((LW || ST) && !empty) {
                        /* --log-band: the soft band's penalty on at m against off, dp, exactly in the log odds: on with
                           probability 1 / (1 + exp(-(lr - dp))), that is, when dp < lr - logit(u) (off impossible if a
                           penalized row would then be predicted at 0). For an off rupture, bounds on dp from the column's sums
                           (col_dots_lw, in the same pass as the dot products) settle nearly every decision, most of them
                           without logarithms; otherwise dp is computed exactly. The decisions are the exact ones */
                        const double uu = unif(s);
                        double g = 0.0;
                        int on = -1, have_g = 0;
                        if (xi == 0.0) {
                            /* bounds on dp from the column's sums: with t = G_ji m / y >= 0 on each penalized row,
                               ln(1 + t) lies in [t - t^2/2, t], so dp = sum w (L^2 + 2 q L) lies in
                               [m s1 - m^2 s3, m s1 + m^2 s2] (col_dots_lw) */
                            const double sm1 = m * S1[q], mm = m * m;
                            double dlo = sm1 - mm * S3[q], dhi = sm1 + mm * S2[q];
                            dlo -= 1e-9 * (fabs(dlo) + 1e-12); dhi += 1e-9 * (fabs(dhi) + 1e-12);
                            if (ST && ST->bx[i] >= 0) {   /* --mfd-step-sd: bounds on its change, without logarithms */
                                const int64_t b = ST->bx[i];
                                const double y = r[ST->row[b]] + ST->dy[b];
                                double flo, fhi;
                                st_bounds(ST, LAM + k * ST->NB, HH + k * ST->NB, b, ST->gx[i] * m / y, &flo, &fhi);
                                dlo += flo; dhi += fhi;
                            }
                            /* stays off, without logarithms: on needs u < 1 / (1 + exp(dp - lr)) <= exp(lr - dlo)
                               = odds exp(X), X = -xl - dlo, and exp(X) <= 1 + X + X^2/2 (X <= 0), 1 + X + X^2 (X <= 1) */
                            const double X = -xl - dlo;
                            if (X <= 1.0) {
                                const double E = X <= 0.0 ? 1.0 + X + 0.5 * X * X : 1.0 + X + X * X;
                                if (uu >= odd0 * E * (1.0 + 1e-9)) on = 0;
                            }
                            if (on < 0) {
                                g = lr - (log(uu) - log1p(-uu)); have_g = 1;
                                if (dhi < g) on = 1; else if (dlo >= g) on = 0;
                            }
                        }
                        if (on < 0) {
                            if (!have_g) g = lr - (log(uu) - log1p(-uu));
                            double dp = LW ? (LWP ? lw_change(data, indices, LWQ, LWP[i], LWP[i + 1], r, DY, LW, LWL, LD, -xi, m - xi, PM)
                                    : lw_change(data, indices, NULL, p0, pe, r, DY, LW, LWL, LD, -xi, m - xi, PM)) : 0.0;
                            if (ST && ST->bx[i] >= 0) {   /* --mfd-step-sd: on at m against off, exactly */
                                const int64_t b = ST->bx[i];
                                const double y = r[ST->row[b]] + ST->dy[b];
                                dp = pen_add(dp, st_change(ST, HH + k * ST->NB, b, y - ST->gx[i] * xi, y + ST->gx[i] * (m - xi)));
                            }
                            on = dp < g;
                        }
                        xn = on ? m : 0.0;
                    } else {
                        const double pon = lr >= 0.0 ? 1.0 / (1.0 + exp(-lr)) : exp(lr) / (1.0 + exp(lr));
                        xn = (unif(s) < pon) ? m : 0.0;
                    }
                    if (RLO && off) {         /* the band, from off: switching on to m is allowed if m is inside */
                        if (xn > 0.0 && !empty) {
                            band_bounds(data, indices, p0, p1, r, RLO, RHI, xi, &blo, &bhi);
                            if (m > bhi) xn = 0.0;
                        }
                    } else if (RLO) {         /* the band: a state outside it has probability 0 */
                        const int mok = m >= blo && m <= bhi, zok = blo <= 0.0;
                        if (mok && !zok) xn = m; else if (!mok && zok) xn = 0.0; else if (!mok) xn = xi;
                    }
                    SM[k * n + i] = m;
                } else if (pkind >= 1) {
                    lbctx c;
                    c.sp = 0;
                    c.pk = (int)pkind; c.muz = 0.0; c.kz = 0.0; c.gam = 0.0;
                    if (pkind == 2) { c.muz = MUZ[k * n + i]; c.kz = KZ[k]; }
                    c.zlo = ZLO ? ZLO[i] : zlo; c.zhi = zhi; c.k = kedge; c.ku = KU ? KU[i] : kedge;
                    c.zmin = LLO[i];
                    c.zmax = LHI[i];
                    const double z0 = log(xi > DBL_MIN ? xi : DBL_MIN);
                    if (RLO) {                /* the band: the slice within the rates it allows (z0 always among them) */
                        if (blo > 0.0) c.zmin = fmin(fmax(c.zmin, log(blo)), z0);
                        c.zmax = fmax(fmin(c.zmax, log(bhi)), z0);
                    }
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
                    const double li = fmax(lo[i], blo), hi_ = fmin(hi[i], bhi);   /* the band's limits, if any */
                    const double a = (li - mu) / sd, b = (hi_ - mu) / sd;
                    if (a >= 0.0)      xn = li + sd * tn_tail_off(s, a, b - a);
                    else if (b <= 0.0) xn = hi_ - sd * tn_tail_off(s, -b, b - a);
                    else               xn = mu + sd * tn_mid(s, a, b);
                }
                if (xn < lo[i]) xn = lo[i]; else if (xn > hi[i]) xn = hi[i];
                const double dl = xn - xi;
                if (dl != 0.0 && !empty) {
                    rw4 *rw = (RW && k < 4 * G4) ? RW + (k / 4) * N : NULL;   /* this chain's group's copy, if any */
                    const int64_t qk = k % 4;
                    for (int64_t p = p0; p < p1; p++) r[indices[p]] += data[p] * dl;
                    if (rw)
                        for (int64_t p = p0; p < p1; p++) rw[indices[p]].r[qk] = r[indices[p]];
                    if (U)                    /* the folded summed slip rates' rows: U = B^T r_ss, kept current */
                        for (int64_t p = pe; p < PC[i]; p++) {
                            const int64_t sr = indices[p] - NB;
                            const double drs = data[p] * dl;
                            for (int64_t t = SSP[sr]; t < SSP[sr + 1]; t++) U[k * N + SSJ[t]] += SSV[t] * drs;
                            if (rw)
                                for (int64_t t = SSP[sr]; t < SSP[sr + 1]; t++) rw[SSJ[t]].u[qk] = U[k * N + SSJ[t]];
                        }
                    if (WC)                   /* --cfm slip rates: the changed rows' cached terms (those the passes read) */
                        for (int64_t p = p0; p < pe; p++) {
                            const int32_t j = indices[p];
                            WC[k * N + j] = sp_w2(r[j], SL2[j], SH2[j]);
                            if (DS) DS[k * N + j] = sp_d(r[j], SL2[j], SH2[j]);
                            if (rw) rw[j].w[qk] = WC[k * N + j];
                        }
                    if (ST && ST->bx[i] >= 0) {   /* --mfd-step-sd: the bin's ln prediction and the others' cross terms */
                        const int64_t b = ST->bx[i];
                        st_update(ST, LAM + k * ST->NB, HH + k * ST->NB, b, r[ST->row[b]] + ST->dy[b]);
                    }
                    if (AC)                   /* --log-band: the changed rows' bound coefficients (with LWP, the
                                                 weighted rows', the only ones with any) */
                        for (int64_t t = LWP ? LWP[i] : p0; t < (LWP ? LWP[i + 1] : pe); t++) {
                            const int64_t p = LWP ? LWQ[t] : t;
                            const int32_t j = indices[p];
                            if (LW[j] > 0.0) lw_coef(AC + 4 * (k * N + j), LW[j], LWL ? LWL[j] : LW[j], r[j] + DY[j], LD[j]);
                            if (PM && PM->I[j] >= 0) {
                                double *e = AC + 4 * (k * N + j);
                                if (!(LW[j] > 0.0)) e[0] = e[1] = e[2] = 0.0;
                                pm_coef(e, PM, PM->I[j], r[j] + DY[j]);
                            }
                        }
                }
                if (COL && ((xi > 0.0) != (xn > 0.0))) CNT[k * SB + SBIN[i]] += xn > 0.0 ? 1 : -1;
                x[i] = xn;
            }
        }
      }
    }
    free(AC); free(ODD); free(LAM); free(HH);
    free(WC);
    free(DS);
    free(U);
    free(RW);
    free(SLH);
    if (pkind == 3 && CNT)                         /* each chain's ruptures switched on, per bin, for the fractions on */
        for (int64_t k = 0; k < K; k++) {
            int64_t *cn = CNT + k * SB;
            const double *x = X + k * n;
            for (int64_t b = 0; b < SB; b++) cn[b] = 0;
            for (int64_t i = 0; i < n; i++) if (x[i] > 0.0) cn[SBIN[i]]++;
        }
}

/* Swap moves: for each listed pair (i, j), propose exchanging m_i and m_j (an involution, so the Metropolis ratio
   is target(new)/target(old)).  Change of ||r||^2: 2 dl (G_i - G_j).r + dl^2 ||G_i - G_j||^2 with dl = m_j - m_i.
   The log-box prior is the same for every rate, so it cancels; an exponential prior contributes -dl (beta_i - beta_j).
   With the band (RLO, RHI; or NULL), an exchange that would leave a row outside it is rejected: the target is 0 there. */
static int swap_in_band(const int64_t *indptr, const int32_t *indices, const double *data, int64_t i, int64_t j,
                        double dl, const double *r, const double *RLO, const double *RHI)
{
    int64_t p = indptr[i], q = indptr[j];                /* the two columns' rows, merged (each sorted) */
    const int64_t pe = indptr[i + 1], qe = indptr[j + 1];
    while (p < pe || q < qe) {
        int32_t row; double dv;
        if (q >= qe || (p < pe && indices[p] < indices[q])) { row = indices[p]; dv = data[p] * dl; p++; }
        else if (p >= pe || indices[q] < indices[p]) { row = indices[q]; dv = -data[q] * dl; q++; }
        else { row = indices[p]; dv = data[p] * dl - data[q] * dl; p++; q++; }
        const double rn = r[row] + dv;
        if (rn < RLO[row] || rn > RHI[row]) return 0;
    }
    return 1;
}
/* --log-band: the soft band's change when m_i and m_j are exchanged (the two columns' rows merged); +INFINITY if a
   prediction would not be above 0 */
static double lw_swap(const int64_t *indptr, const int32_t *indices, const double *data, int64_t i, int64_t j,
                      double dl, const double *r, const double *DY, const double *LW, const double *LWL, const double *LD,
                      const pmctx *PM, const int64_t *LWP, const int64_t *LWQ)
{
    /* LWP, LWQ (or NULL): the two columns' weighted rows' entries merged in place of the whole columns: the same rows,
       in the same order, with the same changes (2026-11-24) */
    int64_t a = LWP ? LWP[i] : indptr[i], b = LWP ? LWP[j] : indptr[j];
    const int64_t ae = LWP ? LWP[i + 1] : indptr[i + 1], be = LWP ? LWP[j + 1] : indptr[j + 1];
    double acc = 0.0;
    while (a < ae || b < be) {
        const int64_t p = a < ae ? (LWP ? LWQ[a] : a) : -1, q = b < be ? (LWP ? LWQ[b] : b) : -1;
        int32_t row; double dv;
        if (q < 0 || (p >= 0 && indices[p] < indices[q])) { row = indices[p]; dv = data[p] * dl; a++; }
        else if (p < 0 || indices[q] < indices[p]) { row = indices[q]; dv = -data[q] * dl; b++; }
        else { row = indices[p]; dv = data[p] * dl - data[q] * dl; a++; b++; }
        const double w = LW[row];
        const int32_t t = PM ? PM->I[row] : -1;
        if (w <= 0.0 && t < 0) continue;
        const double y = r[row] + DY[row], y1 = y + dv;
        if (y1 <= 0.0) return INFINITY;
        if (w > 0.0) {
            const double l0 = log(y) - LD[row], l1 = log(y1) - LD[row];
            if (!LWL) acc += w * (l1 * l1 - l0 * l0);
            else acc += (l1 < 0.0 ? LWL[row] : w) * l1 * l1 - (l0 < 0.0 ? LWL[row] : w) * l0 * l0;   /* sided weights */
        }
        if (t >= 0) acc += pm_f(PM, t, log(y1), NULL) - pm_f(PM, t, log(y), NULL);
    }
    return acc;
}
/* --cfm slip rates: the split misfit's change when m_i and m_j are exchanged (the two columns' rows merged) */
static double sp_swap(const int64_t *indptr, const int32_t *indices, const double *data, int64_t i, int64_t j,
                      double dl, const double *r, const double *SL2, const double *SH2)
{
    int64_t p = indptr[i], q = indptr[j];
    const int64_t pe = indptr[i + 1], qe = indptr[j + 1];
    double acc = 0.0;
    while (p < pe || q < qe) {
        int32_t row; double dv;
        if (q >= qe || (p < pe && indices[p] < indices[q])) { row = indices[p]; dv = data[p] * dl; p++; }
        else if (p >= pe || indices[q] < indices[p]) { row = indices[q]; dv = -data[q] * dl; q++; }
        else { row = indices[p]; dv = data[p] * dl - data[q] * dl; p++; q++; }
        const double e0 = r[row], e1 = e0 + dv;
        acc += sp_w2(e1, SL2[row], SH2[row]) - sp_w2(e0, SL2[row], SH2[row]);
    }
    return acc;
}
/* --swap-cos-cross: spike-and-slab's log prior of rupture i at rate m relative to off, log(pi / (1 - pi)) of its bin
   (LOD) plus the slab's normalized log density at z = log m (its density in z, so a swap, which exchanges two values of
   z, has Jacobian 1); 0 for m = 0 (off), -inf outside the slab (below its floor or outside [lo, hi]) */
static double ss_on_lp(int64_t i, double m, const double *LODk, const int64_t *SBIN, const double *ZLO, const double *KU,
                       const double *LLO, const double *LHI, const double *SC, double zlo, double zhi, double gam)
{
    if (!(m > 0.0)) return 0.0;
    const double z = log(m);
    if (z < LLO[i] || z > LHI[i]) return -INFINITY;
    lbctx c;
    memset(&c, 0, sizeof c);
    c.pk = 3; c.zlo = ZLO ? ZLO[i] : zlo; c.zhi = zhi; c.k = 0.0; c.ku = KU ? KU[i] : 0.0; c.gam = gam;
    const double p = lb_prior(&c, z);
    return p == -INFINITY ? -INFINITY : (LODk ? LODk[SBIN[i]] : 0.0) + p - log(SC[4 * i] + SC[4 * i + 1]);
}

void swap_run(int64_t N, int64_t n, int64_t K, const int64_t *indptr, const int32_t *indices, const double *data,
              const double *csq, int64_t npair, const int64_t *pi, const int64_t *pj, const double *gij,
              const double *beta, double sigma, double *X, double *R, uint64_t *rng, int64_t *nacc,
              const double *RLO, const double *RHI, const double *LW, const double *LWL, const double *LD, const double *DY,
              const pmctx *PM, const double *SL2, const double *SH2, const stctx *ST,
              const int64_t *SBIN, int64_t SB, const double *LOD, const double *ZLO, const double *KU, const double *LLO,
              const double *LHI, const double *SC, double zlo, double zhi, double gam, int64_t *CNT, double frac,
              const int64_t *LWP, const int64_t *LWQ)
{
    /* frac < 1 (--swap-fraction): each pass proposes each pair with probability frac, independently (geometric skips
       through the list), a random choice made apart from the state, so each pass is still a valid move */
    const double lq = frac < 1.0 ? log1p(-frac) : 0.0;
    /* SBIN (or NULL): spike-and-slab's bin of each rupture; a pair in different bins (--swap-cos-cross) adds the prior's
       ratio, which is 1 within a bin (the same fraction on and slab), so same-bin swaps are as before */
    const double s2 = sigma * sigma;
    double *lam = ST ? (double *)malloc(sizeof(double) * (size_t)ST->NB) : NULL;
    double *hh = ST ? (double *)malloc(sizeof(double) * (size_t)ST->NB) : NULL;
    for (int64_t k = 0; k < K; k++) {
        double *x = X + k * n, *r = R + k * N;
        uint64_t *s = rng + 4 * k;
        int64_t acc = 0;
        if (ST) st_fill(ST, r, lam, hh, k);
        for (int64_t q = frac < 1.0 ? (int64_t)floor(log(1.0 - unif(s)) / lq) : 0; q < npair;
             q += 1 + (frac < 1.0 ? (int64_t)floor(log(1.0 - unif(s)) / lq) : 0)) {
            const int64_t i = pi[q], j = pj[q];
            const double mi = x[i], mj = x[j];
            if (mi == mj) continue;
            double ci = 0.0, cj = 0.0;        /* the columns' dot products, for the Gaussian change only (with SL2, the
                                                 split or exact misfit, sp_swap's own pass gives it: skipped, 2026-11-24) */
            if (!SL2) {
                for (int64_t p = indptr[i]; p < indptr[i + 1]; p++) ci += data[p] * r[indices[p]];
                for (int64_t p = indptr[j]; p < indptr[j + 1]; p++) cj += data[p] * r[indices[p]];
            }
            const double dl = mj - mi;
            const double dchi = SL2 ? sp_swap(indptr, indices, data, i, j, dl, r, SL2, SH2)
                                    : 2.0 * dl * (ci - cj) + dl * dl * (csq[i] + csq[j] - 2.0 * gij[q]);
            double dlp = -dl * (beta[i] - beta[j]) - (LW ? lw_swap(indptr, indices, data, i, j, dl, r, DY, LW, LWL, LD, PM, LWP, LWQ) : 0.0);
            if (SBIN && SBIN[i] != SBIN[j]) {     /* --swap-cos-cross: x_i = mj and x_j = mi against x_i = mi and x_j = mj */
                const int col = SSC_ON && CNT;    /* --collapse: the counts' ratio in place of the fractions on */
                const double *LODk = col ? NULL : LOD + k * SB;
                const double af = ss_on_lp(i, mj, LODk, SBIN, ZLO, KU, LLO, LHI, SC, zlo, zhi, gam)
                                  + ss_on_lp(j, mi, LODk, SBIN, ZLO, KU, LLO, LHI, SC, zlo, zhi, gam);
                const double bf = ss_on_lp(i, mi, LODk, SBIN, ZLO, KU, LLO, LHI, SC, zlo, zhi, gam)
                                  + ss_on_lp(j, mj, LODk, SBIN, ZLO, KU, LLO, LHI, SC, zlo, zhi, gam);
                dlp = (af == -INFINITY || !(bf > -INFINITY)) ? -INFINITY : dlp + (af - bf);
                if (col && (mi > 0.0) != (mj > 0.0)) {   /* one on moves to the other's bin: off in bin a, on in bin b */
                    const int64_t *cn = CNT + k * SB;
                    const int64_t ba = mi > 0.0 ? SBIN[i] : SBIN[j], bb = mi > 0.0 ? SBIN[j] : SBIN[i];
                    dlp += log(ssc_odds(cn, bb, 0)) - log(ssc_odds(cn, ba, 1));
                }
            }
            const int64_t bi = ST ? ST->bx[i] : -1, bj = ST ? ST->bx[j] : -1;
            if (ST && (bi >= 0 || bj >= 0)) {     /* --mfd-step-sd: its change, the one or two bins moved */
                const int64_t NB = ST->NB;
                double st = 0.0;
                if (bi == bj) {
                    const double y = r[ST->row[bi]] + ST->dy[bi];
                    st = st_change(ST, hh, bi, y, y + (ST->gx[i] - ST->gx[j]) * dl);
                } else {
                    double l1[2], d_[2]; const int64_t bb[2] = {bi, bj}; const double dv[2] = {bi >= 0 ? ST->gx[i] * dl : 0.0,
                                                                                               bj >= 0 ? -ST->gx[j] * dl : 0.0};
                    int bad = 0;
                    for (int t = 0; t < 2; t++) {
                        if (bb[t] < 0) { d_[t] = 0.0; l1[t] = 0.0; continue; }
                        const double y1 = r[ST->row[bb[t]]] + ST->dy[bb[t]] + dv[t];
                        if (!(y1 > 0.0)) bad = 1;
                        l1[t] = bad ? 0.0 : log(y1);
                        d_[t] = l1[t] - lam[bb[t]];
                    }
                    if (bad) st = INFINITY;
                    else {
                        double S2 = 0.0;
                        for (int t = 0; t < 2; t++)
                            if (bb[t] >= 0) S2 += d_[t] * (ST->A[bb[t] * NB + bb[t]] * (l1[t] + lam[bb[t]]) + 2.0 * hh[bb[t]]);
                        if (bi >= 0 && bj >= 0) S2 += 2.0 * ST->A[bi * NB + bj] * d_[0] * d_[1];
                        st = 0.5 * S2;
                    }
                }
                dlp = st == INFINITY ? -INFINITY : dlp - st;
            }
            if (log(unif(s)) < -0.5 * dchi / s2 + dlp && (!RLO || swap_in_band(indptr, indices, data, i, j, dl, r, RLO, RHI))) {
                for (int64_t p = indptr[i]; p < indptr[i + 1]; p++) r[indices[p]] += data[p] * dl;
                for (int64_t p = indptr[j]; p < indptr[j + 1]; p++) r[indices[p]] -= data[p] * dl;
                x[i] = mj; x[j] = mi; acc++;
                if (SSC_ON && CNT && SBIN && SBIN[i] != SBIN[j] && (mi > 0.0) != (mj > 0.0)) {
                    int64_t *cn = CNT + k * SB;
                    cn[SBIN[i]] += mj > 0.0 ? 1 : -1;
                    cn[SBIN[j]] += mi > 0.0 ? 1 : -1;
                }
                if (ST) {                     /* --mfd-step-sd: the moved bins' ln predictions and cross terms */
                    if (bi >= 0) st_update(ST, lam, hh, bi, r[ST->row[bi]] + ST->dy[bi]);
                    if (bj >= 0 && bj != bi) st_update(ST, lam, hh, bj, r[ST->row[bj]] + ST->dy[bj]);
                }
            }
        }
        nacc[k] += acc;
    }
    free(lam); free(hh);
}

/* ---- split/merge moves (--split-merge, 2026-11-24): reversible-jump moves between a rupture A and two ruptures B and C
   that together are A (B the first part of A's subsections, C the rest; split_merge.py finds them), with alpha = D_A / D_B
   and beta = D_A / D_C (average slips), so that rate t on A, or alpha t on B and beta t on C, give every slip-rate row the
   same prediction, and so the summed slip rates' and the plate rate's rows, which add slip-rate rows up; the paleo rows
   and the bins differ (A's bin loses t, B's and C's gain alpha t and beta t), as the likelihood scores exactly.
   In z = ln u (u the kernel's scaled rate), with LAB = ln(alpha s_A / s_B) and LAC = ln(beta s_A / s_C) (s the scales):
     split (A on, B and C off): v ~ N(0, sd^2), z_B = z_A + LAB + v, z_C = z_A + LAC - v, A off;
     merge (B and C on, A off): its inverse, v = (w_B - w_C) / 2, z_A = (w_B + w_C) / 2, w_B = z_B - LAB, w_C = z_C - LAC.
   At v = 0 the slip-rate rows are unchanged; v lets B and C leave the exact proportion (so a merge can start from pairs
   that are not in it). The map (z_A, v) -> (z_B, z_C) has Jacobian 2. sd = min(SMAX, SMC sigma / (u_A |G_A|)), a function
   of z_A alone, the same both ways (v moves the slip-rate rows by about u_A |G_A| v).
   Each proposal picks a rupture uniformly among the NH ruptures switched on that are A or B in some triple (INH), then
   with probability 1/2 a split with it as A, uniformly among its triples with B and C off (s of them), or else a merge
   with it as B, uniformly among its triples with C on and A off (m of them); none: no move. The target's density is in z
   for each rupture on (ss_on_lp: the bin's log odds and the slab's normalized density in z), so a split is accepted with
     log r = dlogL + lp(B) + lp(C) - lp(A) - ln g(v) + ln 2 + ln(NH / NH') + ln(s / m'),
   NH' and m' counted in the proposed state (m' >= 1: the reverse merge), and a merge with the mirror image. dlogL is exact
   over the three columns' rows: the slip misfit's terms (or the Gaussian's) on rows below NLIK, the rows gibbs_run's
   passes score, and the curve terms on theirs; the band's limits checked on every row. A fixed number of proposals per
   chain per call (not one that depends on the state, which would not leave the target unchanged). The triples come sorted
   by A (AP: each rupture's, as A), with BP, BL each rupture's as B. */
static inline double sm_sd(double zA, double csqA, double smc, double sigma, double smax)
{
    if (!(csqA > 0.0)) return smax;
    const double s = smc * sigma / (exp(zA) * sqrt(csqA));
    return s < smax ? s : smax;
}
/* the three columns' rows, each listed once (tl, in the order met), and each one's change dv[j] = sum_q G_jc[q] dx[q];
   mk marks rows listed (the caller clears it through tl) */
static int64_t sm_rows(const int64_t *indptr, const int32_t *indices, const double *data, const int64_t *c,
                       const double *dx, double *dv, int32_t *tl, unsigned char *mk)
{
    int64_t nt = 0;
    for (int q = 0; q < 3; q++)
        for (int64_t p = indptr[c[q]]; p < indptr[c[q] + 1]; p++) {
            const int32_t j = indices[p];
            if (!mk[j]) { mk[j] = 1; dv[j] = 0.0; tl[nt++] = j; }
            dv[j] += data[p] * dx[q];
        }
    return nt;
}
/* the move's change in the log likelihood over those rows (as swap_run's: the slip misfit's or Gaussian's terms on rows
   below NLIK over 2 sigma^2, the curve terms as lw_swap's); -INFINITY if a row would leave its limits (the band) or a
   curve row's prediction would not be above 0 */
static double sm_loglik(int64_t nt, const int32_t *tl, const double *dv, const double *r, int64_t NLIK, double s2,
                        const double *RLO, const double *RHI, const double *SL2, const double *SH2, const double *LW,
                        const double *LWL, const double *LD, const double *DY, const pmctx *PM)
{
    double dchi = 0.0, dlw = 0.0;
    for (int64_t t = 0; t < nt; t++) {
        const int32_t j = tl[t];
        const double e0 = r[j], e1 = e0 + dv[j];
        if (RLO && (e1 < RLO[j] || e1 > RHI[j])) return -INFINITY;
        if (j < NLIK) dchi += SL2 ? sp_w2(e1, SL2[j], SH2[j]) - sp_w2(e0, SL2[j], SH2[j]) : e1 * e1 - e0 * e0;
        if (LW) {
            const double w = LW[j];
            const int32_t tt = PM ? PM->I[j] : -1;
            if (w <= 0.0 && tt < 0) continue;
            const double y = e0 + DY[j], y1 = y + dv[j];
            if (!(y1 > 0.0)) return -INFINITY;
            if (w > 0.0) {
                const double l0 = log(y) - LD[j], l1 = log(y1) - LD[j];
                if (!LWL) dlw += w * (l1 * l1 - l0 * l0);
                else dlw += (l1 < 0.0 ? LWL[j] : w) * l1 * l1 - (l0 < 0.0 ? LWL[j] : w) * l0 * l0;
            }
            if (tt >= 0) dlw += pm_f(PM, tt, log(y1), NULL) - pm_f(PM, tt, log(y), NULL);
        }
    }
    return -0.5 * dchi / s2 - dlw;
}
void splitmerge_run(int64_t N, int64_t n, int64_t K, const int64_t *indptr, const int32_t *indices, const double *data,
                    const double *csq, double sigma, int64_t NLIK, const int32_t *TA, const int32_t *TB, const int32_t *TC,
                    const double *LAB, const double *LAC, const int64_t *AP, const int64_t *BP, const int32_t *BL,
                    const unsigned char *INH, int64_t nprop, double smc, double smax, double *X, double *R, uint64_t *rng,
                    int64_t *STATS, const double *RLO, const double *RHI, const double *LW, const double *LWL,
                    const double *LD, const double *DY, const pmctx *PM, const double *SL2, const double *SH2,
                    const int64_t *SBIN, int64_t SB, const double *LOD, const double *ZLO, const double *KU,
                    const double *LLO, const double *LHI, const double *SC, double zlo, double zhi, double gam,
                    int64_t *CNT)
{
    const double s2 = sigma * sigma, LN2 = 0.69314718055994531, HL2PI = 0.91893853320467274;   /* ln 2, ln(2 pi) / 2 */
    int64_t *onl = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));   /* the ruptures on in H, and where */
    int64_t *pos = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));   /* each is in that list */
    double *dv = (double *)malloc(sizeof(double) * (size_t)(N > 0 ? N : 1));
    int32_t *tl = (int32_t *)malloc(sizeof(int32_t) * (size_t)(N > 0 ? N : 1));
    unsigned char *mk = (unsigned char *)calloc((size_t)(N > 0 ? N : 1), 1);
    for (int64_t k = 0; k < K; k++) {
        double *x = X + k * n, *r = R + k * N;
        uint64_t *s = rng + 4 * k;
        const double *LODk = LOD + k * SB;
        int64_t *cn = CNT ? CNT + k * SB : NULL, *st = STATS + 4 * k;
        int64_t nh = 0;
        for (int64_t i = 0; i < n; i++)
            if (INH[i] && x[i] > 0.0) { pos[i] = nh; onl[nh++] = i; }
        for (int64_t q = 0; q < nprop && nh > 0; q++) {   /* nh = 0: no move can change the state (none on to start from) */
            int64_t h = (int64_t)(unif(s) * (double)nh);
            if (h >= nh) h = nh - 1;
            const int64_t i = onl[h];
            const int split = unif(s) < 0.5;
            int64_t A, B, C, t = -1, cnt = 0;
            if (split) {                  /* i as A: its triples with B and C off */
                for (int64_t p = AP[i]; p < AP[i + 1]; p++) if (!(x[TB[p]] > 0.0) && !(x[TC[p]] > 0.0)) cnt++;
                if (!cnt) continue;
                int64_t pick = (int64_t)(unif(s) * (double)cnt);
                if (pick >= cnt) pick = cnt - 1;
                for (int64_t p = AP[i]; p < AP[i + 1]; p++)
                    if (!(x[TB[p]] > 0.0) && !(x[TC[p]] > 0.0) && pick-- == 0) { t = p; break; }
                A = i; B = TB[t]; C = TC[t];
            } else {                      /* i as B: its triples with C on and A off */
                for (int64_t p = BP[i]; p < BP[i + 1]; p++) {
                    const int64_t u = BL[p];
                    if (x[TC[u]] > 0.0 && !(x[TA[u]] > 0.0)) cnt++;
                }
                if (!cnt) continue;
                int64_t pick = (int64_t)(unif(s) * (double)cnt);
                if (pick >= cnt) pick = cnt - 1;
                for (int64_t p = BP[i]; p < BP[i + 1]; p++) {
                    const int64_t u = BL[p];
                    if (x[TC[u]] > 0.0 && !(x[TA[u]] > 0.0) && pick-- == 0) { t = u; break; }
                }
                A = TA[t]; B = i; C = TC[t];
            }
            double uA, uB, uC, lr;
            if (split) {
                st[0]++;
                uA = x[A];
                const double zA = log(uA), sd = sm_sd(zA, csq[A], smc, sigma, smax), v = sd * stdnorm(s);
                const double zB = zA + LAB[t] + v, zC = zA + LAC[t] - v;
                uB = exp(zB); uC = exp(zC);
                if (!(uB > 0.0) || !(uC > 0.0) || !isfinite(uB) || !isfinite(uC)) continue;   /* (not in the support) */
                const double lpB = ss_on_lp(B, uB, LODk, SBIN, ZLO, KU, LLO, LHI, SC, zlo, zhi, gam);
                const double lpC = ss_on_lp(C, uC, LODk, SBIN, ZLO, KU, LLO, LHI, SC, zlo, zhi, gam);
                if (lpB == -INFINITY || lpC == -INFINITY) continue;
                const double lpA = ss_on_lp(A, uA, LODk, SBIN, ZLO, KU, LLO, LHI, SC, zlo, zhi, gam);
                int64_t mrev = 0;             /* the reverse merge's choices: B's triples with C' on and A' off after it */
                for (int64_t p = BP[B]; p < BP[B + 1]; p++) {
                    const int64_t u = BL[p], a2 = TA[u], c2 = TC[u];
                    const int onc = c2 == A ? 0 : (c2 == B || c2 == C) ? 1 : x[c2] > 0.0;
                    const int ona = a2 == A ? 0 : (a2 == B || a2 == C) ? 1 : x[a2] > 0.0;
                    if (onc && !ona) mrev++;
                }
                const int64_t nh1 = nh + (INH[C] ? 1 : 0);   /* A out, B in, C in if it is in H */
                const double lg = -0.5 * v * v / (sd * sd) - log(sd) - HL2PI;
                lr = lpB + lpC - lpA - lg + LN2 + log((double)nh / (double)nh1) + log((double)cnt / (double)mrev);
            } else {
                st[2]++;
                uB = x[B]; uC = x[C];
                const double wB = log(uB) - LAB[t], wC = log(uC) - LAC[t];
                const double v = 0.5 * (wB - wC), zA = 0.5 * (wB + wC), sd = sm_sd(zA, csq[A], smc, sigma, smax);
                uA = exp(zA);
                if (!(uA > 0.0) || !isfinite(uA)) continue;
                const double lpA = ss_on_lp(A, uA, LODk, SBIN, ZLO, KU, LLO, LHI, SC, zlo, zhi, gam);
                if (lpA == -INFINITY) continue;
                const double lpB = ss_on_lp(B, uB, LODk, SBIN, ZLO, KU, LLO, LHI, SC, zlo, zhi, gam);
                const double lpC = ss_on_lp(C, uC, LODk, SBIN, ZLO, KU, LLO, LHI, SC, zlo, zhi, gam);
                int64_t srev = 0;             /* the reverse split's choices: A's triples with B' and C' off after it */
                for (int64_t p = AP[A]; p < AP[A + 1]; p++) {
                    const int64_t b2 = TB[p], c2 = TC[p];
                    const int onb = b2 == A ? 1 : (b2 == B || b2 == C) ? 0 : x[b2] > 0.0;
                    const int onc = c2 == A ? 1 : (c2 == B || c2 == C) ? 0 : x[c2] > 0.0;
                    if (!onb && !onc) srev++;
                }
                const int64_t nh1 = nh - (INH[C] ? 1 : 0);   /* B out, C out if it is in H, A in */
                const double lg = -0.5 * v * v / (sd * sd) - log(sd) - HL2PI;
                lr = lpA - lpB - lpC + lg - LN2 + log((double)nh / (double)nh1) + log((double)cnt / (double)srev);
            }
            const int64_t cols[3] = {A, B, C};
            const double dx[3] = {split ? -uA : uA, split ? uB : -uB, split ? uC : -uC};
            const int64_t nt = sm_rows(indptr, indices, data, cols, dx, dv, tl, mk);
            const double ll = sm_loglik(nt, tl, dv, r, NLIK, s2, RLO, RHI, SL2, SH2, LW, LWL, LD, DY, PM);
            if (ll > -INFINITY && log(unif(s)) < lr + ll) {
                for (int64_t u = 0; u < nt; u++) r[tl[u]] += dv[tl[u]];
                if (split) {
                    x[A] = 0.0; x[B] = uB; x[C] = uC;
                    { const int64_t pa = pos[A], last = onl[--nh]; onl[pa] = last; pos[last] = pa; }
                    pos[B] = nh; onl[nh++] = B;
                    if (INH[C]) { pos[C] = nh; onl[nh++] = C; }
                    if (cn) { cn[SBIN[A]]--; cn[SBIN[B]]++; cn[SBIN[C]]++; }
                    st[1]++;
                } else {
                    x[A] = uA; x[B] = 0.0; x[C] = 0.0;
                    { const int64_t pb = pos[B], last = onl[--nh]; onl[pb] = last; pos[last] = pb; }
                    if (INH[C]) { const int64_t pc = pos[C], last = onl[--nh]; onl[pc] = last; pos[last] = pc; }
                    pos[A] = nh; onl[nh++] = A;
                    if (cn) { cn[SBIN[A]]++; cn[SBIN[B]]--; cn[SBIN[C]]--; }
                    st[3]++;
                }
            }
            for (int64_t u = 0; u < nt; u++) mk[tl[u]] = 0;
        }
    }
    free(onl); free(pos); free(dv); free(tl); free(mk);
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
            raise SystemExit(f'[gibbs] could not compile the C kernel: {getattr(e, "stderr", b"")!r:.2000}'
                             if isinstance(e, subprocess.CalledProcessError) else f'[gibbs] could not compile the C kernel ({e})')
    L = ctypes.CDLL(lib)
    P = ctypes.c_void_p
    I, D = ctypes.c_int64, ctypes.c_double
    L.gibbs_run.argtypes = ([I] * 3 + [P] * 7 + [D] + [P] * 3 + [I, I, D, D, D, D] + [P, P, P, P, P, P, P, I] + [P, P, P, P, D, P, P]
                            + [P] * 4 + [P] + [P, P] + [P, P] + [P, P, I, I, P, P, P, P] + [P])
    L.gibbs_run.restype = None
    L.kernel_consts.argtypes = [I, P, P, D, D, P, P, D, P, P, P]
    L.sp_tables.argtypes = [P] * 5
    L.sp_tables.restype = None
    L.ss_collapse.argtypes = [I, D, D, P]
    L.sp_bounds.argtypes = [P, P, P]
    L.sp_bounds.restype = None
    L.ss_collapse.restype = None
    L.kernel_consts.restype = None
    L.swap_run.argtypes = [I] * 3 + [P] * 4 + [I] + [P] * 4 + [D] + [P] * 6 + [P] * 4 + [P] + [P, P] + [P] + [P, I] + [P] * 6 + [D] * 3 + [P, D, P, P]
    L.swap_run.restype = None
    L.splitmerge_run.argtypes = ([I] * 3 + [P] * 4 + [D, I] + [P] * 9 + [I, D, D] + [P] * 4 + [P] * 6 + [P] + [P, P] + [P, I]
                                 + [P] * 6 + [D] * 3 + [P])
    L.splitmerge_run.restype = None
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
        self.gam = 0.0                             # spike-and-slab: the slab's tilt, density m^(gam - 1) (1/2: uniform in sqrt m)
        self.n_data = self.N                       # the data rows; any rows after them are the balance term's
        self.n_bal = None                          # the end of the balance term's rows (None: n_fit); --slip-sum's follow
        self.n_fit = self.N                        # the rows in the likelihood; any after them are constraint rows only
        self.balance = None                        # --down-the-middle: dict(A, names) of the balance rows
        self.slip_sum = None                       # --slip-sum: dict(expect) of the summed slip rates' rows
        self.lw = None                             # --log-band: dict(W, LD, DY): each row's weight, log target, target
        self.mstep = None                          # --mfd-step-sd: dict(row, A, c, dy, bx, gx, pairs, e, tau, const) (mfd_step)
        self.split = None                          # --cfm-sliprates (in the problem): dict(L2, H2), each row's weight
                                                   # below and above its target (1 on rows without a split normal)
        self.rlo = self.rhi = None                 # --band, --positive: each row's residual kept within [rlo, rhi]
        self.pass_end = None                       # column_passes: where each column's likelihood passes end (None: its end)
        self.ss_fold = None                        # column_passes: the summed slip rates' rows folded into the slip-rate rows
        self.sm = None                             # --split-merge: the triples and settings (set_split_merge)
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

    def set_band(self, F, rows):
        """The band: each of these rows' predictions kept within a factor F of its target, d/F to F d (the target must be
        positive); in residuals r = prediction - d, from d/F - d to F d - d.  Other rows are unbounded."""
        rows = np.asarray(rows)
        if (self.d[rows] <= 0).any():
            raise ValueError('the band needs positive targets')
        self.rlo, self.rhi = np.full(self.N, -np.inf), np.full(self.N, np.inf)
        self.rlo[rows], self.rhi[rows] = self.d[rows] / F - self.d[rows], F * self.d[rows] - self.d[rows]
        self.band = dict(F=float(F), rows=rows, tol=np.where(np.isfinite(self.rlo), 1e-9 * np.abs(self.d), 0.0),
                         what=f'the band (a factor {F:g} of their targets)')

    def set_positive(self, rows):
        """Positivity (--positive), spike-and-slab only: each of these rows' predictions kept above zero, that is, at least
        one of the rupture rates in it on. With nonnegative coefficients and rates that are 0 or at least the slab floor,
        that is the same as a prediction of at least half the smallest amount one rupture switched on can add to the row,
        a lower limit with no upper one, which the band's machinery then keeps (set_band's rlo and rhi)."""
        rows = np.asarray(rows)
        if self.pkind != 3:
            raise ValueError('positivity needs spike-and-slab (under the other priors no rate is exactly 0)')
        Gr = sp.csr_matrix(self.G)[rows]
        if (Gr.data < 0).any():
            raise ValueError('positivity needs nonnegative coefficients in the rows it applies to')
        if (np.diff(Gr.indptr) == 0).any():
            raise ValueError('positivity: a row with no rupture in it can never be predicted above zero')
        floor = np.exp(self.zlo_vec if self.zlo_vec is not None else np.full(self.n, self.zlo))   # each rate's slab floor
        step = np.minimum.reduceat(Gr.data * floor[Gr.indices], Gr.indptr[:-1])                 # smallest one-rupture amount
        self.rlo, self.rhi = np.full(self.N, -np.inf), np.full(self.N, np.inf)
        self.rlo[rows] = 0.5 * step - self.d[rows]
        tol = np.zeros(self.N)
        tol[rows] = np.minimum(1e-9 * np.abs(self.d[rows]), 0.1 * step)     # rounding, well below the limit
        self.band = dict(F=None, rows=rows, tol=tol, floor=floor, what='positivity (each above zero)')

    def set_groups(self, pos_rows, band_rows, band_F):
        """Hard limits by row group (--slip-band, --paleo-band): positivity on pos_rows, as set_positive keeps it, and on
        band_rows each prediction within its own factor band_F of its target, as set_band keeps it; other rows (the MFD
        bins among them) unbounded."""
        pos_rows, band_rows = np.asarray(pos_rows, np.int64), np.asarray(band_rows, np.int64)
        self.rlo, self.rhi = np.full(self.N, -np.inf), np.full(self.N, np.inf)
        tol, floor, F = np.zeros(self.N), None, np.asarray(band_F, np.float64)
        if len(pos_rows):
            if self.pkind != 3:
                raise ValueError('positivity needs spike-and-slab (under the other priors no rate is exactly 0)')
            Gr = sp.csr_matrix(self.G)[pos_rows]
            if (Gr.data < 0).any():
                raise ValueError('positivity needs nonnegative coefficients in the rows it applies to')
            if (np.diff(Gr.indptr) == 0).any():
                raise ValueError('positivity: a row with no rupture in it can never be predicted above zero')
            floor = np.exp(self.zlo_vec if self.zlo_vec is not None else np.full(self.n, self.zlo))
            step = np.minimum.reduceat(Gr.data * floor[Gr.indices], Gr.indptr[:-1])
            self.rlo[pos_rows] = 0.5 * step - self.d[pos_rows]
            tol[pos_rows] = np.minimum(1e-9 * np.abs(self.d[pos_rows]), 0.1 * step)
        if len(band_rows):
            if (self.d[band_rows] <= 0).any():
                raise ValueError('the band needs positive targets')
            self.rlo[band_rows], self.rhi[band_rows] = self.d[band_rows] / F - self.d[band_rows], F * self.d[band_rows] - self.d[band_rows]
            tol[band_rows] = 1e-9 * np.abs(self.d[band_rows])
        self.band = dict(F=None, rows=np.r_[pos_rows, band_rows], pos_rows=pos_rows, band_rows=band_rows, band_F=F, tol=tol,
                         floor=floor, what='the slip-rate and paleo rows\' limits')

    def add_limits(self, rows, rlo, rhi, tol, what):
        """More rows kept within limits on their residuals, [rlo, rhi], alongside any band or positivity (constraint rows
        such as --mfd-monotonic's, after the rows in the likelihood); tol: the rounding allowed when counting rows
        outside."""
        rows = np.asarray(rows)
        if self.rlo is None:
            self.rlo, self.rhi = np.full(self.N, -np.inf), np.full(self.N, np.inf)
            self.band = dict(F=None, rows=np.zeros(0, int), tol=np.zeros(self.N), what=what)
        else:
            self.band['what'] += ' and ' + what
        self.rlo[rows], self.rhi[rows] = rlo, rhi
        self.band['tol'] = np.array(self.band['tol'], np.float64)
        self.band['tol'][rows] = tol

    def band_outside(self, R):
        """Per chain, how many rows of the band its residuals R leave outside it (beyond rounding)."""
        tol = self.band['tol']
        return np.sum((R < self.rlo - tol) | (R > self.rhi + tol), axis=1)

    def residual(self, X):
        if np.count_nonzero(X) > 0.05 * X.size:
            return np.ascontiguousarray((self.G @ X.T).T - self.d)
        R = np.empty((X.shape[0], self.N))               # mostly zeros (rates off or underflowed): each chain's nonzero
        for k, x in enumerate(X):                        # columns only, the same sums in the same order without the
            c = np.flatnonzero(x)                        # zero terms
            R[k] = self.G[:, c] @ x[c] - self.d
        return R


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class _PMC(ctypes.Structure):
    """--paleo-misfit: the kernel's pmctx, the paleo rows' tables."""
    _fields_ = [('I', ctypes.c_void_p), ('V0', ctypes.c_void_p), ('F', ctypes.c_void_p), ('D', ctypes.c_void_p),
                ('K', ctypes.c_void_p), ('H', ctypes.c_double), ('NG', ctypes.c_int64)]


def _pm_ptr(T):
    """--paleo-misfit: the address of the kernel's tables structure, or None without it."""
    pm = None if T.lw is None else T.lw.get('PM')
    if pm is None:
        return None
    if '_c' not in pm:
        pm['_c'] = _PMC(*(a.ctypes.data for a in (pm['I'], pm['V0'], pm['F'], pm['D'], pm['K'])), float(pm['H']), int(pm['NG']))
    return ctypes.addressof(pm['_c'])


class _STC(ctypes.Structure):
    """--mfd-step-sd, --mfd-gr-tau: the kernel's stctx, the MFD steps' term."""
    _fields_ = [('NB', ctypes.c_int64), ('row', ctypes.c_void_p), ('A', ctypes.c_void_p), ('c', ctypes.c_void_p),
                ('dy', ctypes.c_void_p), ('bx', ctypes.c_void_p), ('gx', ctypes.c_void_p), ('cs', ctypes.c_int64)]


def _st_ptr(T, k0=0):
    """--mfd-step-sd, --mfd-gr-tau: the address of the kernel's stctx for chains from k0 on (with --mfd-gr-tau each chain
    has its own c, the draw's G-R curve, so the context points at chain k0's), or None without it."""
    ms = getattr(T, 'mstep', None)
    if ms is None:
        return None
    cs = ms['c'].shape[1] if ms['c'].ndim == 2 else 0
    key = '_c' if cs == 0 else f'_c{k0}'
    if key not in ms:
        ms[key] = _STC(len(ms['row']), *(ms[k].ctypes.data for k in ('row', 'A')), ms['c'].ctypes.data + k0 * cs * 8,
                       *(ms[k].ctypes.data for k in ('dy', 'bx', 'gx')), cs)
    return ctypes.addressof(ms[key])


def _sp_ptrs(T):
    """--cfm slip rates: the split misfit's weights below and above each row's target (1 on rows without it), or NULLs;
    and the kernel's tables set to T's (--slip-misfit exact: rows with L2 < 0 read them) or cleared, before every call."""
    tab = None if getattr(T, 'split', None) is None else T.split.get('tab')
    L = build_kernel()
    if tab is None:
        L.sp_tables(None, None, None, None, None)
    else:
        if 'FD' not in tab or len(tab['FD']) != 2 * len(tab['F']):   # the kernel's layout: values and step-scaled slopes
            hh = 1.0 / tab['IH']                                       # interleaved, and each table's step
            off = np.asarray(tab['OFF'], np.int64)
            hpt = np.repeat(hh, np.diff(off))
            tab['H'] = np.ascontiguousarray(hh)
            tab['FD'] = np.ascontiguousarray(np.column_stack([tab['F'], tab['D'] * hpt]).ravel())
        L.sp_tables(_ptr(tab['E0']), _ptr(tab['IH']), _ptr(tab['H']), tab['OFF'].ctypes.data, _ptr(tab['FD']))
    qi = _off_bounds_qi(T)
    if qi is None:
        L.sp_bounds(None, None, None)
    else:
        L.sp_bounds(_ptr(qi), _ptr(T.split['C_row']), T.split['HO'].ctypes.data)
    if getattr(T, 'split', None) is None:
        return None, None
    return _ptr(T.split['L2']), _ptr(T.split['H2'])


def _off_bounds_qi(T):
    """--off-bounds: each column's sum of G_ji^2 C_j / 2, C_j the largest |second derivative| of row j's term (sp_w2) in its
    residual: over a table's cubic pieces, at their ends (where a cubic's is largest), 0 on its linear continuations;
    2 max(l, h) for a split normal's row; over the entries the likelihood's passes read. None without the split misfit,
    or with T.off_bounds False. Kept on T.split (so set pass_end first)."""
    sp_ = getattr(T, 'split', None)
    if sp_ is None or not getattr(T, 'off_bounds', True):
        return None
    if sp_.get('QI') is not None and len(sp_['QI']) == T.n:
        return sp_['QI']
    L2, H2 = np.asarray(sp_['L2'], float), np.asarray(sp_['H2'], float)
    C = 2.0 * np.maximum(L2, H2)
    tab = sp_.get('tab')
    if tab is not None and (L2 < 0).any():
        off = np.asarray(tab['OFF'], np.int64)
        nt = len(off) - 1
        F = np.asarray(tab['F'], float)
        Dh = np.asarray(tab['D'], float) * np.repeat(1.0 / np.asarray(tab['IH'], float), np.diff(off))
        seg = np.ones(len(F), bool)
        seg[off[1:] - 1] = False                            # a segment starts at every point but each table's last
        k = np.flatnonzero(seg)
        p0, m0, p1, m1 = F[k], Dh[k], F[k + 1], Dh[k + 1]
        c = np.maximum(np.abs(-6 * p0 - 4 * m0 + 6 * p1 - 2 * m1), np.abs(6 * p0 + 2 * m0 - 6 * p1 + 4 * m1))
        t_of = np.repeat(np.arange(nt), np.diff(off))[k]
        CT = np.zeros(nt)
        np.maximum.at(CT, t_of, c)
        CT *= np.asarray(tab['IH'], float) ** 2             # from the table's steps to the residual's units
        rows = L2 < 0
        C[rows] = CT[(-L2[rows] - 1).astype(np.int64)]
    C = np.where(np.isfinite(C) & (C > 0), C, 0.0)
    col = np.repeat(np.arange(T.n), np.diff(T.indptr))
    use = np.ones(len(col), bool)                         # the entries the likelihood's passes read (column_passes' PF)
    pf = getattr(T, 'pass_end', None)
    if pf is not None:
        use = np.arange(len(col)) < np.asarray(pf)[col]
    qi = np.bincount(col[use], weights=0.5 * T.data[use] ** 2 * C[T.indices[use]], minlength=T.n)
    sp_['QI'] = np.ascontiguousarray(qi * (1.0 + 1e-12))
    sp_['C_row'] = np.ascontiguousarray(C * (1.0 + 1e-12))   # the kernel's C_j, inflated as QI is
    # each column's likelihood entries (the first ones, rows sorted) in order of G_ji^2 C_j, largest first: the order of
    # the exact pass with early stopping; positions in data (int32), the other entries unused; in chunks of columns
    HO = np.zeros(len(T.data), np.int32)
    ip = np.asarray(T.indptr)
    for c0 in range(0, T.n, 50_000):            # in chunks of whole columns
        c1 = min(c0 + 50_000, T.n)
        pc = np.arange(ip[c0], ip[c1])
        pc = pc[use[pc]]
        if len(pc):
            key = T.data[pc] ** 2 * C[T.indices[pc]]
            HO[pc] = pc[np.lexsort((-key, col[pc]))]
    sp_['HO'] = HO
    return sp_['QI']


def _sp_terms(T, rows, e):
    """--cfm slip rates: each residual's term, w e^2 with w its side's weight, or, on a row with a table (--slip-misfit
    pdf, L2 < 0), the table's misfit at e (cru_common.hermite_eval, the kernel's spt_f)."""
    rows = np.asarray(rows)
    L2, H2 = T.split['L2'][rows], T.split['H2'][rows]
    out = np.where(e < 0, L2, H2) * e * e
    tab = T.split.get('tab')
    if tab is None or not (L2 < 0).any():
        return out
    from cru_common import hermite_eval
    on = np.broadcast_to(L2 < 0, np.shape(out))
    t = np.broadcast_to(np.where(L2 < 0, -L2 - 1, 0).astype(np.int64), np.shape(out))
    return np.where(on, hermite_eval(tab, t, np.broadcast_to(e, np.shape(out))), out)


def _pm_f_np(pm, t, v):
    """NumPy twin of pm_f: the half misfits of tables t at v = ln y."""
    t, v = np.asarray(t, np.int64), np.asarray(v, float)
    h, ng, v0 = pm['H'], pm['NG'], pm['V0'][t]
    x = (v - v0) / h
    base = t * ng
    k = np.clip(np.floor(np.where(np.isfinite(x), x, 0.0)).astype(np.int64), 0, ng - 2)
    s = x - k
    F, D = pm['F'], pm['D']
    f0, f1, d0, d1 = F[base + k], F[base + k + 1], D[base + k], D[base + k + 1]
    s2, s3 = s * s, s * s * s
    val = (2 * s3 - 3 * s2 + 1) * f0 + (s3 - 2 * s2 + s) * h * d0 + (-2 * s3 + 3 * s2) * f1 + (s3 - s2) * h * d1
    val = np.where(x > 0, val, F[base] + D[base] * (v - v0))
    hi = x >= ng - 1
    return np.where(hi, F[base + ng - 1] + D[base + ng - 1] * (v - (v0 + h * (ng - 1))), val)


def _lw_ptrs(T):
    """--log-band's arrays for the kernel (each row's weight, its weight below its target when the two differ (WL,
    --slip-misfit split+log10; NULL otherwise), log target and target), or NULLs without it."""
    if T.lw is None:
        return None, None, None, None
    return (_ptr(T.lw['W']), None if T.lw.get('WL') is None else _ptr(T.lw['WL']), _ptr(T.lw['LD']), _ptr(T.lw['DY']))


def _lw_sq(T, rows, q):
    """The soft band's terms w q^2 of rows `rows` at q = ln(y / d), each with its own side's weight (WL below the target,
    W above, with --slip-misfit split+log10; W alone otherwise). NumPy twin of the kernel's."""
    w = T.lw['W'][rows]
    if T.lw.get('WL') is None:
        return w * q ** 2
    return np.where(q < 0, T.lw['WL'][rows], w) * q ** 2


def _lw_sub_ptrs(T):
    """The weighted rows' entries, column by column (positions into the column arrays and each column's start), for the
    kernel's bound sums (col_lwsums), when those rows hold under half of the entries (the paleo rows, the bins in log10);
    NULLs otherwise, or without the soft band's machinery. Built once per T.lw."""
    if T.lw is None:
        return None, None
    key = (id(T.lw['W']), id(T.lw.get('PM')), id(getattr(T, 'pass_end', None)))
    if getattr(T, '_lw_sub_key', None) != key:
        w = np.asarray(T.lw['W']) > 0
        if T.lw.get('PM') is not None:
            w = w | (np.asarray(T.lw['PM']['I']) >= 0)
        hit = w[T.indices]
        sub = None
        pf = getattr(T, 'pass_end', None)              # the share among the entries the passes read
        share = hit.mean() if pf is None else hit.sum() / max(1, int((pf - T.indptr[:-1]).sum()))
        if share < 0.5:
            pos = np.ascontiguousarray(np.flatnonzero(hit), np.int64)
            sub = (np.ascontiguousarray(np.searchsorted(pos, T.indptr), np.int64), pos)
        T._lw_sub, T._lw_sub_key = sub, key
    return (None, None) if T._lw_sub is None else (_ptr(T._lw_sub[0]), _ptr(T._lw_sub[1]))


def kernel_consts(lib, T):
    """The C kernel's per-rupture constants (log lo, log hi, the slab's), once per run, by the kernel's own code."""
    LLO, LHI, SC = np.empty(T.n), np.empty(T.n), np.empty(4 * T.n)
    lib.kernel_consts(T.n, _ptr(T.lo), _ptr(T.hi), T.zlo, T.zhi, None if T.zlo_vec is None else _ptr(T.zlo_vec),
                      None if T.ku_vec is None else _ptr(T.ku_vec), T.gam, _ptr(LLO), _ptr(LHI), _ptr(SC))
    return LLO, LHI, SC


def sweeps_c(lib, T, X, R, rng, nsweeps, threads=1, MUZ=None, KZ=None, SM=None, LOD=None, KC=None, CNT=None):
    if KC is None:
        KC = kernel_consts(lib, T)
    if MUZ is None:
        MUZ, KZ = np.zeros(X.shape), np.zeros(X.shape[0])
    if SM is None:
        SM, LOD = np.zeros(X.shape), np.zeros((X.shape[0], 1))
    SB = LOD.shape[1]                                    # spike-and-slab: log odds per chain and bin
    sbin = None if T.ss is None else np.ascontiguousarray(T.ss['bins'], np.int64)
    K = X.shape[0]
    groups = np.array_split(np.arange(K), max(1, min(threads, K)))
    # the shared tables, structs and pointers, built (and cached) here, once, before any thread starts: built inside each
    # thread's call, two threads could each build a cache on the first call, the first one's freed while its kernel reads
    # it (2026-11-24: identical runs on 4 threads differed from their first draw)
    shared = (*_lw_ptrs(T), _pm_ptr(T), *_sp_ptrs(T), *_lw_sub_ptrs(T), *_pass_ptrs(T))
    stp = {int(g_[0]): _st_ptr(T, int(g_[0])) for g_ in groups if len(g_)}

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
                      ctypes.c_void_p(SM.ctypes.data + k0 * T.n * 8), ctypes.c_void_p(LOD.ctypes.data + k0 * SB * 8),
                      None if sbin is None else _ptr(sbin), SB, _ptr(KC[0]), _ptr(KC[1]), _ptr(KC[2]),
                      None if CNT is None else ctypes.c_void_p(CNT.ctypes.data + k0 * SB * 8), T.gam,
                      None if T.rlo is None else _ptr(T.rlo),
                      None if T.rhi is None else _ptr(T.rhi), *shared, stp[k0])
    if len(groups) == 1:
        work(groups[0])
    else:
        th = [threading.Thread(target=work, args=(g,)) for g in groups]
        [t.start() for t in th]
        [t.join() for t in th]


def _pass_ptrs(T):
    """column_passes' arrays for the kernel (PF, PC, NB, NSR, SSP, SSJ, SSV, NSS), NULLs and zeros without them."""
    pf = getattr(T, 'pass_end', None)
    f = getattr(T, 'ss_fold', None)
    if f is None:
        return (None if pf is None else _ptr(pf)), None, 0, 0, None, None, None, None
    return (_ptr(pf), _ptr(f['PC']), int(f['NB']), int(f['NSR']), _ptr(f['SSP']), _ptr(f['SSJ']), _ptr(f['SSV']),
            _ptr(f['NSS']))


def column_passes(T, B=None, tol=1e-12):
    """The C kernel's passes over each column for the likelihood (the dot products, the split misfit's change, the soft
    band's bound sums) need not read every row (gibbs_run's PF): the constraint rows (rows n_fit on: --mfd-monotonic's and
    --plate-rate's, with coefficients of 1e-9, contributing about 1e-18 to each term; band_bounds still reads them), and,
    given B (a sparse matrix, the summed slip rates' rows x all rows, nonzero only in data rows) with those rows (n_bal to
    n_fit) equal to B G, the summed slip rates' rows, folded into the rows B reads (gibbs_run's U = B^T r_ss, per chain):
    their share of a column's dot product is G_i.U and of its squared norm NSS[i]. B G is checked against the rows
    themselves (to tol, relative to their largest entry); if it does not match, they are left in the passes. Sets
    T.pass_end and T.ss_fold; returns a line saying what the passes skip (or None if nothing)"""
    T.pass_end, T.ss_fold = None, None
    nbal = T.n_fit if T.n_bal is None else T.n_bal
    col = np.repeat(np.arange(T.n), np.diff(T.indptr))

    def ends(cut):
        return np.ascontiguousarray(T.indptr[:-1] + np.bincount(col[T.indices < cut], minlength=T.n), np.int64)
    PC = ends(T.n_fit)
    fold, note = None, None
    if B is not None and nbal < T.n_fit:
        B = sp.csr_matrix(B, dtype=np.float64)
        Gr = sp.csr_matrix(T.G)
        Gss = Gr[nbal:T.n_fit]
        ok = B.shape == (T.n_fit - nbal, T.N) and (B.nnz == 0 or B.indices.max() < nbal)
        if ok:
            D = (B @ Gr - Gss).tocsr()
            scale_ = abs(Gss).max() if Gss.nnz else 1.0
            err = abs(D).max() / scale_ if D.nnz else 0.0
            ok = err <= tol
            note = (f'the summed slip rates\' {T.n_fit - nbal} rows ({Gss.nnz:,} entries) folded into the slip-rate rows '
                    f'(B G matches them to {err:.1e} of their largest entry)' if ok else
                    f'the summed slip rates\' rows not folded: B G differs from them by {err:.1e} of their largest entry')
        else:
            note = 'the summed slip rates\' rows not folded: B does not map the data rows onto them'
        if ok:
            B.sort_indices()
            fold = dict(PC=PC, NB=nbal, NSR=T.n_fit - nbal, SSP=np.ascontiguousarray(B.indptr, np.int64),
                        SSJ=np.ascontiguousarray(B.indices, np.int32), SSV=np.ascontiguousarray(B.data, np.float64),
                        NSS=np.ascontiguousarray(np.asarray(Gss.multiply(Gss).sum(0)).ravel(), np.float64))
    PF = ends(nbal) if fold is not None else PC
    if fold is None and (PF == T.indptr[1:]).all():
        return note
    T.pass_end, T.ss_fold = PF, fold
    skipped = int((T.indptr[1:] - PF).sum())
    parts = ([f'the constraint rows ({int((T.indptr[1:] - PC).sum()):,} entries)'] if (PC < T.indptr[1:]).any() else []) \
        + ([note] if fold is not None else [])
    return (f'column passes: {skipped:,} of {T.indptr[-1]:,} entries ({skipped / max(T.indptr[-1], 1):.0%}) left out of the '
            'likelihood\'s passes: ' + '; '.join(parts) + ('' if fold is not None or note is None else f'; {note}'))


def swaps_c(lib, T, X, R, rng, nacc, threads=1, LOD=None, KC=None, CNT=None):
    pi, pj, gij = T.pairs
    K = X.shape[0]
    xb = T.pkind == 3 and T.ss is not None and LOD is not None and KC is not None   # spike-and-slab: the prior's ratio
    sbin = np.ascontiguousarray(T.ss['bins'], np.int64) if xb else None              # for pairs in different bins
    SB = LOD.shape[1] if xb else 0
    groups = np.array_split(np.arange(K), max(1, min(threads, K)))
    shared = (*_lw_ptrs(T), _pm_ptr(T), *_sp_ptrs(T))   # built once, before the threads (see sweeps_c)
    lwsub = _lw_sub_ptrs(T)
    stp = {int(g_[0]): _st_ptr(T, int(g_[0])) for g_ in groups if len(g_)}

    def work(ks):
        if len(ks) == 0:
            return
        k0, k1 = int(ks[0]), int(ks[-1]) + 1
        lib.swap_run(T.N, T.n, k1 - k0, _ptr(T.indptr), _ptr(T.indices), _ptr(T.data), _ptr(T.csq), len(pi),
                     _ptr(pi), _ptr(pj), _ptr(gij), _ptr(T.beta), T.sigma,
                     ctypes.c_void_p(X.ctypes.data + k0 * T.n * 8), ctypes.c_void_p(R.ctypes.data + k0 * T.N * 8),
                     ctypes.c_void_p(rng.ctypes.data + k0 * 32), ctypes.c_void_p(nacc.ctypes.data + k0 * 8),
                     None if T.rlo is None else _ptr(T.rlo), None if T.rhi is None else _ptr(T.rhi), *shared,
                     stp[k0], None if sbin is None else _ptr(sbin), SB,
                     None if not xb else ctypes.c_void_p(LOD.ctypes.data + k0 * SB * 8),
                     None if T.zlo_vec is None else _ptr(T.zlo_vec), None if T.ku_vec is None else _ptr(T.ku_vec),
                     None if not xb else _ptr(KC[0]), None if not xb else _ptr(KC[1]), None if not xb else _ptr(KC[2]),
                     T.zlo, T.zhi, T.gam, None if CNT is None or not xb else ctypes.c_void_p(CNT.ctypes.data + k0 * SB * 8),
                     float(getattr(T, 'swap_frac', 1.0)), *lwsub)
    if len(groups) == 1:
        work(groups[0])
    else:
        th = [threading.Thread(target=work, args=(g,)) for g in groups]
        [t.start() for t in th]
        [t.join() for t in th]


def set_split_merge(T, TA, TB, TC, lab, lac, nprop, scale=1.0, smax=1.0):
    """--split-merge: the kernel's arrays for the moves (splitmerge_run) on T.sm: the triples (A, B, C) as columns of T, with
    LAB = ln(alpha s_A / s_B) and LAC = ln(beta s_A / s_C) in the sampler's units (alpha = D_A / D_B, beta = D_A / D_C, s
    the columns' scales), sorted by A (AP: each column's triples as A), each column's as B (BP, BL), and the columns that
    are A or B in any (INH); nprop proposals per chain per sweep, the deviation's sd scale (SMC) and cap (SMAX)."""
    TA, TB, TC = (np.asarray(v, np.int64) for v in (TA, TB, TC))
    if not (len(TA) == len(TB) == len(TC) == len(lab) == len(lac)):
        raise ValueError('split/merge: the triples\' arrays differ in length')
    if len(TA) and (min(TA.min(), TB.min(), TC.min()) < 0 or max(TA.max(), TB.max(), TC.max()) >= T.n):
        raise ValueError('split/merge: a triple names a column outside the problem')
    if ((TA == TB) | (TA == TC) | (TB == TC)).any():
        raise ValueError('split/merge: a triple with a column twice')
    if not (nprop >= 1 and scale > 0 and smax > 0) or not np.isfinite(np.r_[lab, lac]).all():
        raise ValueError('split/merge: give at least one proposal, a positive scale and cap, and finite ratios')
    o = np.lexsort((TC, TB, TA))
    TA, TB, TC, lab, lac = TA[o], TB[o], TC[o], np.asarray(lab, float)[o], np.asarray(lac, float)[o]
    AP = np.zeros(T.n + 1, np.int64)
    AP[1:] = np.cumsum(np.bincount(TA, minlength=T.n))
    BL = np.argsort(TB, kind='stable')
    BP = np.zeros(T.n + 1, np.int64)
    BP[1:] = np.cumsum(np.bincount(TB, minlength=T.n))
    INH = np.zeros(T.n, np.uint8)
    INH[TA] = 1
    INH[TB] = 1
    T.sm = dict(TA=np.ascontiguousarray(TA, np.int32), TB=np.ascontiguousarray(TB, np.int32),
                TC=np.ascontiguousarray(TC, np.int32), LAB=np.ascontiguousarray(lab), LAC=np.ascontiguousarray(lac),
                AP=AP, BP=BP, BL=np.ascontiguousarray(BL, np.int32), INH=INH, nprop=int(nprop), scale=float(scale),
                smax=float(smax))
    return T.sm


def splitmerge_c(lib, T, X, R, rng, stats, threads=1, LOD=None, KC=None, CNT=None):
    """--split-merge: T.sm['nprop'] split/merge proposals per chain (splitmerge_run); stats (K x 4, int64): splits proposed
    and accepted, merges proposed and accepted, added to. The rows the likelihood scores: those gibbs_run's passes do (rows
    before n_fit with the column passes, every row without them)."""
    sm = T.sm
    K = X.shape[0]
    SB = LOD.shape[1]
    if '_sbin' not in sm:
        sm['_sbin'] = np.ascontiguousarray(T.ss['bins'], np.int64)
    sbin = sm['_sbin']
    nlik = T.n_fit if getattr(T, 'pass_end', None) is not None else T.N
    groups = np.array_split(np.arange(K), max(1, min(threads, K)))
    shared = (*_lw_ptrs(T), _pm_ptr(T), *_sp_ptrs(T))   # built once, before the threads (see sweeps_c)

    def work(ks):
        if len(ks) == 0:
            return
        k0, k1 = int(ks[0]), int(ks[-1]) + 1
        lib.splitmerge_run(T.N, T.n, k1 - k0, _ptr(T.indptr), _ptr(T.indices), _ptr(T.data), _ptr(T.csq), T.sigma, nlik,
                           _ptr(sm['TA']), _ptr(sm['TB']), _ptr(sm['TC']), _ptr(sm['LAB']), _ptr(sm['LAC']), _ptr(sm['AP']),
                           _ptr(sm['BP']), _ptr(sm['BL']), _ptr(sm['INH']), sm['nprop'], sm['scale'], sm['smax'],
                           ctypes.c_void_p(X.ctypes.data + k0 * T.n * 8), ctypes.c_void_p(R.ctypes.data + k0 * T.N * 8),
                           ctypes.c_void_p(rng.ctypes.data + k0 * 32), ctypes.c_void_p(stats.ctypes.data + k0 * 32),
                           None if T.rlo is None else _ptr(T.rlo), None if T.rhi is None else _ptr(T.rhi), *shared,
                           _ptr(sbin), SB, ctypes.c_void_p(LOD.ctypes.data + k0 * SB * 8),
                           None if T.zlo_vec is None else _ptr(T.zlo_vec), None if T.ku_vec is None else _ptr(T.ku_vec),
                           _ptr(KC[0]), _ptr(KC[1]), _ptr(KC[2]), T.zlo, T.zhi, T.gam,
                           None if CNT is None else ctypes.c_void_p(CNT.ctypes.data + k0 * SB * 8))
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
        gam = c.get('gam', 0.0)
        if z < c['zlo']:
            return -(c['zlo'] - z) ** 2 * c['k'] + gam * z if c['k'] > 0 else -np.inf
        if z > c['zhi']:
            ku = c.get('ku', c['k'])
            return -(z - c['zhi']) ** 2 * ku + gam * z if ku > 0 else -np.inf
        return gam * z

    def ld(z):
        if not (c['zmin'] <= z <= c['zmax']):
            return -np.inf
        p = lprior(z)
        if p == -np.inf:
            return -np.inf
        m = np.exp(z)
        if 'lik' in c:                                   # --cfm slip rates: the likelihood exactly
            return c['lik'](m) - c['L0'] + p - c['lp0']
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


def _band_bounds_np(T, r, rows, vals, xi):
    """NumPy twin of band_bounds: the rates x_i can take with every banded row of its column inside the band."""
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(vals > 0, (T.rlo[rows] - r) / vals, (T.rhi[rows] - r) / vals)
        b = np.where(vals > 0, (T.rhi[rows] - r) / vals, (T.rlo[rows] - r) / vals)
    a, b = a[vals != 0], b[vals != 0]
    return min(xi + (a.max() if a.size else -np.inf), xi), max(xi + (b.min() if b.size else np.inf), xi)


def _band_slice_limits(c, blo, bhi, z0):
    """The slice's range within the rates the band allows (z0 always among them), as in the kernel."""
    if blo > 0:
        c['zmin'] = min(max(c['zmin'], np.log(blo)), z0)
    c['zmax'] = max(min(c['zmax'], np.log(bhi) if bhi < np.inf else np.inf), z0)


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
            if T.rlo is not None:
                _band_slice_limits(c, *_band_bounds_np(T, R[k, rows], rows, vals, xi), z0)
        xn = min(max(np.exp(_lb_slice_py(g, c, z0, T.slw)), lo), hi)
        if xn != xi and not empty:
            R[k, rows] += vals * (xn - xi)
        X[k, i] = xn


def _slab_masses_py(c, gam):
    """NumPy twin of slab_consts: the slab's two masses (box, shoulder) relative to its density at zhi, the shoulder's
    sd and length, and its centre's offset gam sd^2."""
    import math
    zt = min(c['zhi'], c['zmax'])
    wp = zt - c['zlo'] if gam == 0 else (np.exp(gam * (zt - c['zhi'])) - np.exp(gam * (c['zlo'] - c['zhi']))) / gam
    wt = sdt = lim = ct = 0.0
    if c['ku'] > 0 and c['zmax'] > c['zhi']:
        sdt, lim = 1.0 / np.sqrt(2.0 * c['ku']), c['zmax'] - c['zhi']
        ct = gam * sdt ** 2
        wt = (np.exp(0.5 * gam * ct) * sdt * np.sqrt(np.pi / 2.0)
              * (math.erf((lim - ct) / (sdt * np.sqrt(2.0))) + math.erf(ct / (sdt * np.sqrt(2.0)))))
    return wp, wt, sdt, lim, ct


def _slab_draw_py(g, c):
    """NumPy twin of slab_draw_m: an exact draw of z = log m from the slab."""
    gam = c.get('gam', 0.0)
    zt = min(c['zhi'], c['zmax'])
    wp, wt, sdt, lim, ct = _slab_masses_py(c, gam)
    if _u01(g) * (wp + wt) < wp:
        if gam == 0:
            return c['zlo'] + wp * _u01(g)
        f = gam * wp * np.exp(gam * (c['zhi'] - zt))     # drawn down from the top, as in the kernel
        return zt + np.log(1.0 - _u01(g) * f) / gam
    while True:
        t = ct + g.standard_normal() * sdt if gam != 0 else abs(g.standard_normal()) * sdt
        if 0.0 <= t <= lim:
            return c['zhi'] + t


def _lw_change_np(T, r, rows, vals, d0, d1):
    """NumPy twin of lw_change: the soft band's change between the predictions y + G_i d0 and y + G_i d1."""
    w = T.lw['W'][rows]
    pm = T.lw.get('PM')
    tix = None if pm is None else pm['I'][rows]
    sel = w > 0 if tix is None else (w > 0) | (tix >= 0)
    if not sel.any():
        return 0.0
    y = r[rows][sel] + T.lw['DY'][rows][sel]
    a = vals[sel]
    y0, y1 = y + a * d0, y + a * d1
    if (y1 <= 0).any():
        return np.inf
    if (y0 <= 0).any():
        return -np.inf
    ws = w[sel]
    out = 0.0
    if tix is None:
        ld = T.lw['LD'][rows][sel]
        if T.lw.get('WL') is not None:               # --slip-misfit split+log10: each side's own weight
            rs_ = rows[sel]
            return float(np.sum(_lw_sq(T, rs_, np.log(y1) - ld) - _lw_sq(T, rs_, np.log(y0) - ld)))
        return float(ws @ ((np.log(y1) - ld) ** 2 - (np.log(y0) - ld) ** 2))
    wk = ws > 0
    if wk.any():
        ld = T.lw['LD'][rows][sel][wk]
        if T.lw.get('WL') is not None:
            rs_ = rows[sel][wk]
            out += float(np.sum(_lw_sq(T, rs_, np.log(y1[wk]) - ld) - _lw_sq(T, rs_, np.log(y0[wk]) - ld)))
        else:
            out += float(ws[wk] @ ((np.log(y1[wk]) - ld) ** 2 - (np.log(y0[wk]) - ld) ** 2))
    tk = tix[sel]
    pk = tk >= 0
    if pk.any():                                     # --paleo-misfit: the paleo rows' half misfits
        out += float(np.sum(_pm_f_np(pm, tk[pk], np.log(y1[pk])) - _pm_f_np(pm, tk[pk], np.log(y0[pk]))))
    return out


def _ss_collapsed_lod(T, x, i):
    """--collapse: the prior's log odds of rupture i on, given the others, with its bin's fraction on integrated out."""
    bins = T.ss['bins']
    b = bins[i]
    c = float(np.count_nonzero(x[bins == b] > 0) - (x[i] > 0))
    return np.log(T.ss['a'] + c) - np.log(T.ss['b'] + T.ss['nb'][b] - 1.0 - c)


def _sweep_spikeslab_numpy(T, X, R, g, i, SM, LOD):
    """NumPy twin of the kernel's spike-and-slab update of rupture i in every chain."""
    p0, p1 = T.indptr[i], T.indptr[i + 1]
    empty = p1 == p0 or T.inv_csq[i] <= 0
    rows, vals = T.indices[p0:p1], T.data[p0:p1]
    lo, hi, s2 = T.lo[i], T.hi[i], T.sigma ** 2
    for k in range(X.shape[0]):
        xi = X[k, i]
        c = dict(zlo=T.zlo if T.zlo_vec is None else T.zlo_vec[i], zhi=T.zhi, k=0.0, ku=0.0 if T.ku_vec is None else T.ku_vec[i],
                 zmin=np.log(lo) if lo > 0 else -np.inf, zmax=np.log(hi), mu=0.0, inv2s2=0.0, pk=1, muz=0.0, kz=0.0, m0=1.0,
                 gam=T.gam)
        dot = aa = 0.0
        blo, bhi = -np.inf, np.inf                       # the rates the band allows x_i
        if not empty:
            dot, aa = float(R[k, rows] @ vals), T.csq[i]
            if T.rlo is not None:
                blo, bhi = _band_bounds_np(T, R[k, rows], rows, vals, xi)
        sp_ = getattr(T, 'split', None) is not None and not empty   # --cfm slip rates: the likelihood exactly
        if sp_:
            rk = R[k, rows].copy()
            def splik(mm, rk=rk, xi=xi):
                e = rk + vals * (mm - xi)
                return -float(np.sum(_sp_terms(T, rows, e))) / (2.0 * s2)
        if xi > 0:
            c['m0'] = xi
            if sp_:
                c.update(lik=splik, L0=splik(xi))
            elif not empty:
                c.update(mu=xi - dot / aa, inv2s2=0.5 * aa / s2)
            if T.rlo is not None:
                _band_slice_limits(c, blo, bhi, np.log(xi))
            m = np.exp(_lb_slice_py(g, c, np.log(xi), T.slw))
            if T.lw is not None and not empty and m != xi:   # --log-band: the slice step as a Metropolis proposal
                dp = _lw_change_np(T, R[k], rows, vals, 0.0, m - xi)
                if not np.log(_u01(g)) < -dp:
                    m = xi
        else:
            m = np.exp(_slab_draw_py(g, c))
        lr = _ss_collapsed_lod(T, X[k], i) if T.ss.get('collapse') else LOD[k, T.ss['bins'][i]]
        if sp_:
            lr -= -splik(m) + splik(0.0)
        elif not empty:
            lr -= (2.0 * m * (dot - xi * aa) + m * m * aa) / (2.0 * s2)
        if not empty:
            if T.lw is not None:                         # --log-band: on at m against off, in the log odds
                lr -= _lw_change_np(T, R[k], rows, vals, -xi, m - xi)
        pon = 1.0 / (1.0 + np.exp(-lr)) if lr >= 0 else np.exp(lr) / (1.0 + np.exp(lr))
        xn = m if _u01(g) < pon else 0.0
        if T.rlo is not None:                            # the band: a state outside it has probability 0
            mok, zok = blo <= m <= bhi, blo <= 0.0
            xn = m if (mok and not zok) else 0.0 if (zok and not mok) else xi if not mok else xn
        xn = min(max(xn, lo), hi)
        SM[k, i] = m
        if xn != xi and not empty:
            R[k, rows] += vals * (xn - xi)
        X[k, i] = xn


def _swap_in_band(T, r, ri, vi, rj, vj, dl):
    """NumPy twin of swap_in_band: whether exchanging the two rates keeps every banded row inside the band."""
    rows = np.union1d(ri, rj)
    rn = r[rows].copy()
    rn[np.searchsorted(rows, ri)] += vi * dl
    rn[np.searchsorted(rows, rj)] -= vj * dl
    return bool(np.all((rn >= T.rlo[rows]) & (rn <= T.rhi[rows])))


def _ss_on_lp_py(T, i, m, lod):
    """NumPy twin of ss_on_lp: spike-and-slab's log prior of rupture i at rate m relative to off (lod: the chain's log
    odds per bin), the slab's normalized log density in z = log m; 0 for m = 0, -inf outside the slab."""
    if not m > 0:
        return 0.0
    z = float(np.log(m))
    lo_, hi_ = float(np.broadcast_to(T.lo, (T.n,))[i]), float(np.broadcast_to(T.hi, (T.n,))[i])
    zmax = np.log(hi_)
    if (lo_ > 0 and z < np.log(lo_)) or z > zmax:
        return -np.inf
    zl = T.zlo if T.zlo_vec is None else float(T.zlo_vec[i])
    ku = 0.0 if T.ku_vec is None else float(T.ku_vec[i])
    gam = T.gam
    if z < zl:
        return -np.inf
    if z > T.zhi:
        if not ku > 0:
            return -np.inf
        p = -(z - T.zhi) ** 2 * ku + gam * z
    else:
        p = gam * z
    wp, wt = _slab_masses_py(dict(zlo=zl, zhi=T.zhi, ku=ku, zmax=zmax), gam)[:2]
    return lod[T.ss['bins'][i]] + p - np.log(wp + wt)


def swaps_numpy(T, X, R, g, nacc, LOD=None):
    pi, pj, gij = T.pairs
    s2 = T.sigma ** 2
    xb = T.pkind == 3 and T.ss is not None and LOD is not None     # --swap-cos-cross: the prior's ratio across bins
    frac = float(getattr(T, 'swap_frac', 1.0))                     # --swap-fraction: each pair proposed with this chance
    for k in range(X.shape[0]):
        for q in (range(len(pi)) if frac >= 1.0 else np.flatnonzero(g.random(len(pi)) < frac)):
            i, j = pi[q], pj[q]
            mi, mj = X[k, i], X[k, j]
            if mi == mj:
                continue
            ri, vi = T.indices[T.indptr[i]:T.indptr[i + 1]], T.data[T.indptr[i]:T.indptr[i + 1]]
            rj, vj = T.indices[T.indptr[j]:T.indptr[j + 1]], T.data[T.indptr[j]:T.indptr[j + 1]]
            dl = mj - mi
            if getattr(T, 'split', None) is not None:     # --cfm slip rates: the split misfit's change, exactly
                rows_ = np.union1d(ri, rj)
                dv_ = np.zeros(len(rows_))
                np.add.at(dv_, np.searchsorted(rows_, ri), vi * dl)
                np.add.at(dv_, np.searchsorted(rows_, rj), -vj * dl)
                e0_ = R[k, rows_]; e1_ = e0_ + dv_
                dchi = float(np.sum(_sp_terms(T, rows_, e1_) - _sp_terms(T, rows_, e0_)))
            else:
                dchi = 2 * dl * (R[k, ri] @ vi - R[k, rj] @ vj) + dl * dl * (T.csq[i] + T.csq[j] - 2 * gij[q])
            dlp = -dl * (T.beta[i] - T.beta[j])
            if xb and T.ss['bins'][i] != T.ss['bins'][j]:   # --swap-cos-cross: x_i = mj, x_j = mi against mi, mj
                col = T.ss.get('collapse')                  # --collapse: the counts' ratio in place of the fractions on
                lk = np.zeros_like(LOD[k]) if col else LOD[k]
                af = _ss_on_lp_py(T, i, mj, lk) + _ss_on_lp_py(T, j, mi, lk)
                bf = _ss_on_lp_py(T, i, mi, lk) + _ss_on_lp_py(T, j, mj, lk)
                dlp = -np.inf if (af == -np.inf or not bf > -np.inf) else dlp + (af - bf)
                if col and (mi > 0) != (mj > 0):            # one on moves bins: off in its own, on in the other's
                    a_, b_ = (i, j) if mi > 0 else (j, i)
                    dlp += _ss_collapsed_lod(T, X[k], b_) - _ss_collapsed_lod(T, X[k], a_)
            if T.lw is not None:                         # --log-band: the soft band's change
                rows = np.union1d(ri, rj)
                dv = np.zeros(len(rows))
                np.add.at(dv, np.searchsorted(rows, ri), vi * dl)
                np.add.at(dv, np.searchsorted(rows, rj), -vj * dl)
                w = T.lw['W'][rows]
                pm = T.lw.get('PM')
                if pm is None:
                    sel = w > 0
                    y = R[k, rows][sel] + T.lw['DY'][rows][sel]
                    y1 = y + dv[sel]
                    ld = T.lw['LD'][rows][sel]
                    if (y1 <= 0).any():
                        dlp = -np.inf
                    elif T.lw.get('WL') is not None:     # --slip-misfit split+log10: each side's own weight
                        dlp -= float(np.sum(_lw_sq(T, rows[sel], np.log(y1) - ld) - _lw_sq(T, rows[sel], np.log(y) - ld)))
                    else:
                        dlp -= float(w[sel] @ ((np.log(y1) - ld) ** 2 - (np.log(y) - ld) ** 2))
                else:                                    # --paleo-misfit too
                    tix = pm['I'][rows]
                    sel = (w > 0) | (tix >= 0)
                    y = R[k, rows][sel] + T.lw['DY'][rows][sel]
                    y1 = y + dv[sel]
                    if (y1 <= 0).any():
                        dlp = -np.inf
                    else:
                        ws, tk = w[sel], tix[sel]
                        wk, pk = ws > 0, tk >= 0
                        ld = T.lw['LD'][rows][sel][wk]
                        if T.lw.get('WL') is not None:
                            rk_ = rows[sel][wk]
                            dlp -= float(np.sum(_lw_sq(T, rk_, np.log(y1[wk]) - ld) - _lw_sq(T, rk_, np.log(y[wk]) - ld)))
                        else:
                            dlp -= float(ws[wk] @ ((np.log(y1[wk]) - ld) ** 2 - (np.log(y[wk]) - ld) ** 2))
                        dlp -= float(np.sum(_pm_f_np(pm, tk[pk], np.log(y1[pk])) - _pm_f_np(pm, tk[pk], np.log(y[pk]))))
            if np.log(_u01(g)) < -0.5 * dchi / s2 + dlp and (T.rlo is None or _swap_in_band(T, R[k], ri, vi, rj, vj, dl)):
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
            li, hi_ = np.full(len(xi), lo), np.full(len(xi), hi)
            if T.rlo is not None:                        # the band's limits on each chain's x_i
                bb = np.array([_band_bounds_np(T, R[k, rows], rows, vals, xi[k]) for k in range(len(xi))])
                li, hi_ = np.maximum(li, bb[:, 0]), np.minimum(hi_, bb[:, 1])
            a, b = (li - mu) / sd, (hi_ - mu) / sd
            xn = np.empty_like(mu)
            tl, th = a >= 0, b <= 0
            md = ~(tl | th)
            if tl.any():
                xn[tl] = li[tl] + sd * _tail_off_np(g, a[tl], (b - a)[tl])
            if th.any():
                xn[th] = hi_[th] - sd * _tail_off_np(g, -b[th], (b - a)[th])
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


def kept_csr(KD, n):
    """The kept draws' rows (sample's KD['rows']: indices and float32 rates of the rates on) as a CSR matrix, draws x n;
    a row never filled (a run stopped early) is empty."""
    rows = [r if r is not None else (np.zeros(0, np.int32), np.zeros(0, np.float32)) for r in KD['rows']]
    indptr = np.zeros(len(rows) + 1, np.int64)
    indptr[1:] = np.cumsum([len(r[0]) for r in rows])
    ind = np.concatenate([r[0] for r in rows]) if rows else np.zeros(0, np.int32)
    dat = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0, np.float32)
    return sp.csr_matrix((dat, ind, indptr), shape=(len(rows), n))


def sample(T, X0, sweeps, burn, thin, backend='c', threads=1, seed=0, monitor=(), out=None, keep_samples=True, verbose=True,
           swaps=False, transfers=False, hyper_every=None, stream=None, keep_draws=0, split_merge=False):
    """Run K = X0.shape[0] chains.  Records every `thin` sweeps; draws after `burn` sweeps are kept.
    split_merge (with T.sm, set_split_merge): split/merge moves after each call of the sweeps (each sweep under
    spike-and-slab), T.sm['nprop'] proposals per chain each time.
    Returns dict of traces; if out is given, writes <out>.npz and (if keep_samples) <out>_samples.npy.
    keep_draws > 0 also keeps that many whole draws (keep_draws / K per chain, evenly spaced over the recorded draws after
    burn-in, all of them if there are fewer), keep_draws < 0 every recorded draw after burn-in, in <out>_kept.npz (a
    scipy CSR matrix, float32 rates, one row per draw in the order recorded, only the rates on stored: since 2026-11-25;
    a dense <out>_kept.npy before) with their chain and record numbers in <out>_kept_index.npy; the record number indexes
    the traces (chi2, sum_m), and record - burn // thin indexes the stored draws.  It only records: the chains are the
    same with or without it."""
    lib = build_kernel() if backend == 'c' else None
    if backend == 'c' and lib is None:
        backend = 'numpy'
    if T.mstep is not None and backend != 'c':
        raise ValueError('--mfd-step-sd is implemented in the C kernel only (backend c)')
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
    if T.split is not None and (transfers or T.pkind != 3):
        raise NotImplementedError('the split misfit (--cfm-sliprates) is implemented for spike-and-slab without transfer moves')
    if T.rlo is not None and (transfers or T.pkind == 2):
        raise NotImplementedError('the band is not implemented for transfer moves or the hierarchical prior')
    split_merge = bool(split_merge and getattr(T, 'sm', None) is not None)
    smstats = np.zeros((K, 4), np.int64)                 # --split-merge, per chain: splits tried, accepted; merges tried,
    if split_merge and (backend != 'c' or T.pkind != 3 or T.ss is None or T.mstep is not None or transfers):   # accepted
        raise NotImplementedError('split/merge moves (--split-merge) need spike-and-slab and the C kernel, without the MFD '
                                  'steps\' term or transfer moves')
    if split_merge and getattr(T, 'ss_fold', None) is not None and T.split is not None:
        nb_, ne_ = int(T.ss_fold['NB']), int(T.ss_fold['NB']) + int(T.ss_fold['NSR'])   # the folded rows are Gaussian with
        if not (np.all(T.split['L2'][nb_:ne_] == 1.0) and np.all(T.split['H2'][nb_:ne_] == 1.0)):   # weight 1 in the
            raise NotImplementedError('split/merge moves: the folded summed slip rates\' rows have weights other than 1')   # passes
    H = T.hier
    MUZ, KZ = np.zeros((K, T.n)), np.zeros(K)
    if H is not None:
        hmu, htau = H.init(np.log(X))
        MUZ, KZ = np.ascontiguousarray(hmu[:, H.bins]), 0.5 / htau ** 2
    SS = T.ss if T.pkind == 3 else None
    SM, LOD = np.zeros((K, T.n)), np.zeros((K, SS['B'] if SS is not None else 1))   # LOD: log odds per chain and bin
    if SS is not None:
        SM = np.ascontiguousarray(np.where(X > 0, X, np.exp(T.zhi)))
    R = T.residual(X)
    if T.rlo is not None and T.band_outside(R).any():
        raise ValueError(f'the start leaves {T.band_outside(R).max()} rows outside {T.band["what"]}; start inside it')
    rng = seed_states(seed, K)
    g = np.random.default_rng(seed)
    if SS is not None:                                   # per-bin inclusion probabilities: exact Beta updates
        pib = np.empty((K, SS['B']))

        CNT = np.zeros((K, SS['B']), np.int64) if backend == 'c' else None   # the kernel's counts of ruptures on per bin

        def update_pi(counted=False):
            for k in range(K):
                if SS.get('pi') is not None:
                    pib[k] = SS['pi']
                else:
                    on = CNT[k] if counted else np.bincount(SS['bins'][X[k] > 0], minlength=SS['B'])
                    pib[k] = g.beta(SS['a'] + on, SS['b'] + SS['nb'] - on)
            p = np.clip(pib, 1e-300, 1.0 - 1e-16)
            LOD[:] = np.log(p) - np.log1p(-p)
        update_pi()
    collapse = SS is not None and bool(SS.get('collapse')) and SS.get('pi') is None   # --collapse: the fractions on
    if collapse and np.ndim(SS['a']) + np.ndim(SS['b']) > 0:                         # integrated out; drawn only to record
        raise NotImplementedError('--collapse takes one Beta(a, b) for every bin')
    if collapse and split_merge:
        raise NotImplementedError('split/merge moves (--split-merge) need the fractions on drawn (--fractions-on drawn)')
    if backend == 'c':
        nbf = np.ascontiguousarray(SS['nb'], np.float64) if collapse else None
        lib.ss_collapse(int(collapse), float(SS['a']) if collapse else 1.0, float(SS['b']) if collapse else 1.0,
                        None if nbf is None else _ptr(nbf))
    n_rec = sweeps // thin
    n_burn = burn // thin
    n_keep = n_rec - n_burn
    monitor = np.asarray(monitor, dtype=np.int64)
    tr = dict(chi2=np.zeros((K, n_rec)), sum_m=np.zeros((K, n_rec)), tscale=np.zeros((K, n_rec)),
              mon=np.zeros((K, n_rec, len(monitor))), y=np.zeros((K, n_keep, T.n_data), np.float32), sweep=np.zeros(n_rec, int))
    nd = T.n_data                                        # chi2 and predictions over the data rows only
    PMr = (None if T.lw is None or T.lw.get('PM') is None or not len(T.lw['PM']['rows'])   # --paleo-misfit: its rows are
           else T.lw['PM'])                                                                # scaled in G (the bins' alone: not)
    MFr = None if T.lw is None or T.lw.get('MF') is None else T.lw['MF']   # --mfd-misfit log10: the bins' rows too
    SMr = None if T.lw is None or T.lw.get('SM') is None else T.lw['SM']   # --slip-misfit split+log10, log10: the slip rows too
    Dd = (T.d[:nd] if PMr is None and MFr is None and SMr is None   # the data rows' targets, as the problem gives them
          else next(x_ for x_ in (PMr, MFr, SMr) if x_ is not None)['d_data'])
    b_s = 0.0 if SMr is None else float(SMr.get('b', 0.5))   # each group's log weight: (1 - b) linear + b log
    b_m = 0.0 if MFr is None else float(MFr.get('b', 1.0))
    b_p = 0.0 if PMr is None else float(PMr.get('b', 1.0))
    if SMr is not None:                                  # the slip curve's sd and each section's share of the fit check
        tr['slip_geo_s'] = np.array([SMr['s_lo'], SMr['s_hi']])
        tr['slip_geo_expect'] = np.array([SMr['E'], SMr['V']])
        tr['slip_geo_b'] = b_s
        sm_wk = np.asarray(SMr.get('wk', np.ones(len(SMr['rows']))), float)
    if getattr(T, 'split', None) is not None and 'exact_moments' in T.split:   # --slip-misfit exact, for block_fit.py
        tr['slip_exact_expect'] = np.array(T.split['exact_moments'][:2])
    if MFr is not None and MFr.get('pdf'):               # MFD_GR: each bin's curve and its misfit's moments, for block_fit.py
        for k_ in ('th', 'best', 'sd', 'E', 'V'):
            tr['mfd_pdf_' + k_] = np.asarray(MFr[k_])
        tr['mfd_pdf_rows'] = np.asarray(MFr['rows'])
        tr['mfd_log_b'] = b_m
    elif MFr is not None:
        tr['mfd_log_rows'], tr['mfd_log_s'] = np.asarray(MFr['rows']), np.asarray(MFr['s'])
        tr['mfd_log_b'] = b_m
        if 'E' in MFr:
            tr['mfd_log_expect'] = np.array([MFr['E'], MFr['V']])
    if PMr is not None and 'th' in PMr:                  # each paleo site's curve, for block_fit.py, and its misfit's moments
        for k_ in ('sd', 'th', 'best', 'E', 'V'):
            tr['paleo_' + k_] = np.asarray(PMr[k_])
        tr['paleo_rows'] = np.asarray(PMr['rows'])
    if PMr is not None:
        tr['paleo_b'] = b_p
    if getattr(T, 'fit_expect', None) is not None:       # the fit check as fitted (mean, sd), for the misfit below
        tr['misfit_expect'] = np.array(T.fit_expect, float)
    fitted = PMr is not None or MFr is not None or SMr is not None   # chi2 as fitted differs from the Gaussian chi2
    dd = Dd @ Dd
    nbal = T.n_fit if T.n_bal is None else T.n_bal       # the balance term's rows, nd:nbal; --slip-sum's, nbal:n_fit
    if T.balance is not None and nd < nbal:              # the balance term's rows: recorded on their own
        tr['balance_A'], tr['balance_types'] = T.balance['A'], np.array(T.balance['names'])
    if T.slip_sum is not None:
        tr['slip_sum_expect'] = float(T.slip_sum['expect'])
    if getattr(T, 'group_weights', None) is not None:     # DATATYPES_EQUAL: each group's weight and effective measurements
        gw_ = T.group_weights
        tr['group_weights'] = np.array([[gw_['w'][g_], gw_['n'][g_]] for g_ in ('slip', 'mfd', 'paleo') if g_ in gw_['w']])
        tr['group_names'] = np.array([g_ for g_ in ('slip', 'mfd', 'paleo') if g_ in gw_['w']])
    if T.mstep is not None and T.mstep.get('joint'):     # --mfd-gr-tau: its tolerance (decades a step)
        tr['mfd_gr_tau'] = T.mstep['tau_dec']
    elif T.mstep is not None:                            # --mfd-step-sd: its tolerance, and its value if the steps scatter so
        tr['mfd_step_sd'], tr['mfd_step_expect'] = T.mstep['sigma_s'], len(T.mstep['pairs']) - 1
    samp = None
    ST = None                                            # running summaries: everything the notebook needs, without draws
    if stream is not None and n_keep > 0:
        nb_ = max(1, min(int(stream.get('nbatch', 20)), n_keep))   # never more batches than recorded draws
        ST = dict(s1=np.zeros((K, 2, T.n)), s2=np.zeros((K, 2, T.n)), cnt=np.zeros(2), bsum=np.zeros((K, nb_, T.n), np.float32),
                  bcnt=np.zeros(nb_), exc5=np.zeros(T.n), exc4=np.zeros(T.n), msum=np.zeros(T.n), usum=np.zeros(T.n),
                  hist=np.zeros((stream['hist_nb'], T.n), np.int32), hcnt=np.zeros(1),
                  n5=np.zeros((K, n_keep), np.int32), n4=np.zeros((K, n_keep), np.int32),
                  n90=np.zeros((K, n_keep), np.int32), n99=np.zeros((K, n_keep), np.int32))
        INC = {}                                         # incidence matrices by column, for each chain's nonzero rates
        for key in ('inc_s', 'inc_b'):
            if stream.get(key) is not None:
                ST['tot_' + key[-1]] = np.zeros((K, n_keep, stream[key].shape[0]), np.float32)
                INC[key[-1]] = sp.csc_matrix(stream[key])
        hist_flat = ST['hist'].reshape(-1)               # a view, for scattered increments
    if keep_samples and n_keep > 0:
        samp = (np.lib.format.open_memmap(f'{out}_samples.npy', 'w+', np.float32, (K, n_keep, T.n)) if out
                else np.zeros((K, n_keep, T.n), np.float32))
    KD = None                                            # kept draws: a thinned set of whole draws (recording only)
    if keep_draws and n_keep > 0:
        per = ([n_keep] * K if keep_draws < 0 else
               [min(n_keep, keep_draws // K + (1 if k < keep_draws % K else 0)) for k in range(K)])
        if keep_draws > 0 and keep_draws // K + (1 if keep_draws % K else 0) > n_keep and verbose:
            print(f'keep-draws: only {n_keep} recorded draws per chain after burn-in, so all of them are kept '
                  f'({sum(per)} of the {keep_draws} asked for)')
        order = sorted((int((j + 0.5) * n_keep / c), k) for k, c in enumerate(per) for j in range(c))   # (draw, chain)
        KD = dict(at={}, index=np.array([(k, n_burn + t) for t, k in order], np.int64).reshape(-1, 2))
        for row, (t, k) in enumerate(order):
            KD['at'].setdefault(t, []).append((k, row))
        KD['rows'] = [None] * len(order)                 # each kept draw's rates on (indices, float32 values)
    KC = kernel_consts(lib, T) if backend == 'c' else None   # the kernel's per-rupture constants, once per run
    if backend == 'c':
        _off_bounds_qi(T)                                # --off-bounds' sums and column order (~15 s at 630k): set-up,
    t0 = time.time()                                     # not sampling
    he = thin if ((H is None and SS is None) or not hyper_every) else max(1, min(int(hyper_every), thin))
    if collapse and not (T.mstep is not None and T.mstep.get('joint')):   # --collapse: nothing to draw between sweeps, so
        he = thin                                                         # the kernel runs each recording interval whole
    for rec in range(n_rec):
        done = 0
        while done < thin:                                   # hierarchical prior: hyperparameters every `he` sweeps
            ns = min(he, thin - done)
            if backend == 'c':
                sweeps_c(lib, T, X, R, rng, ns, threads, MUZ, KZ, SM, LOD, KC, CNT if SS is not None else None)
            else:
                sweeps_numpy(T, X, R, g, ns, MUZ, KZ, SM, LOD)
            done += ns
            if SS is not None and not collapse:
                update_pi(counted=backend == 'c')
            if split_merge:                                  # --split-merge: given the fractions on just drawn (the
                splitmerge_c(lib, T, X, R, rng, smstats, threads, LOD, KC, CNT)   # counts it changes, CNT, kept current)
            if T.mstep is not None and T.mstep.get('joint'):   # --mfd-gr-tau: each chain's b and level, given its bins
                mjoint_update(T.mstep, R, g)
            if H is not None:
                for k in range(K):
                    zk, hmu[k], htau[k] = H.update(g, T, np.log(X[k]), hmu[k].copy(), htau[k])
                    X[k] = np.exp(zk)
                MUZ[:], KZ[:] = hmu[:, H.bins], 0.5 / htau ** 2
                R = T.residual(X)
        if backend == 'c':
            if swaps:
                swaps_c(lib, T, X, R, rng, nacc, threads, LOD if SS is not None else None, KC,
                        CNT if SS is not None else None)
            if transfers:
                transfers_c(lib, T, X, R, rng, tstats, threads)
        elif swaps:
            swaps_numpy(T, X, R, g, nacc, LOD if SS is not None else None)
        if SS is not None:
            if collapse:                                 # the fractions on, drawn given the counts, for the record
                update_pi()
            on = X > 0
            SM[on] = X[on]                               # an "on" rupture's slab value is its rate (swaps move rates)
            tr.setdefault('hyper_pi', np.zeros((K, n_rec, SS['B'])))[:, rec] = pib
            tr.setdefault('n_on', np.zeros((K, n_rec)))[:, rec] = on.sum(1)
        if H is not None:
            tr.setdefault('hyper_s', np.zeros((K, n_rec)))[:, rec] = htau / LN10
            tr.setdefault('hyper_mu', np.zeros((K, n_rec, H.B)))[:, rec] = hmu / LN10
        R = T.residual(X)                                    # exact recompute: no drift
        Rd = R if nd == T.N else R[:, :nd]                   # the data rows (after them, the balance term's)
        y = Rd + T.d[:nd]
        if PMr is not None:                                  # --paleo-misfit: its misfit, and the paleo rows unscaled, so
            pr_ = PMr['rows']                                # that chi2 and the predictions are as without it
            tr['paleo_mode'] = np.array(PMr['mode'])
            tr.setdefault('paleo_misfit', np.zeros((K, n_rec)))[:, rec] = 2.0 * np.sum(   # the curve's whole misfit (the
                _pm_f_np(PMr, np.tile(np.arange(len(pr_)), (K, 1)), np.log(np.maximum(y[:, pr_], 1e-300))), 1) / b_p   # tables carry b)
            tr['paleo_misfit'][:, rec] /= float(PMr.get('w', 1.0))   # and DATATYPES_EQUAL' w: recorded unweighted
            y = y.copy()
            y[:, pr_] /= PMr['eps']
            Rd = y - Dd
        if MFr is not None and MFr.get('joint'):             # --mfd-gr-tau: the bins' misfit about each chain's G-R curve
            mr_ = MFr['rows']
            tr.setdefault('mfd_misfit', np.zeros((K, n_rec)))[:, rec] = mjoint_value(T.mstep, R)
            y = y.copy()
            y[:, mr_] /= MFr['eps']
            Rd = y - Dd
        elif MFr is not None and MFr.get('pdf'):             # MFD_GR: each bin's curve misfit, -2 ln of its PDF from its minimum
            mr_ = MFr['rows']
            tr.setdefault('mfd_misfit', np.zeros((K, n_rec)))[:, rec] = 2.0 * np.sum(
                _pm_f_np(T.lw['PM'], np.tile(MFr['idx'], (K, 1)), np.log(np.maximum(y[:, mr_], 1e-300))), 1) / float(MFr.get('w', 1.0))
            y = y.copy()
            y[:, mr_] /= MFr['eps']
            Rd = y - Dd
        elif MFr is not None:                                # --mfd-misfit log10: the bins' misfit, sum (log10(y / R) / s)^2,
            mr_ = MFr['rows']                                # on chi2's scale, and the bins' rows unscaled, as above
            with np.errstate(divide='ignore'):
                lq_ = np.log10(np.maximum(y[:, mr_], 0.0) / T.d[mr_])
            tr.setdefault('mfd_misfit', np.zeros((K, n_rec)))[:, rec] = np.sum((lq_ / MFr['s']) ** 2, 1)
            y = y.copy()
            y[:, mr_] /= MFr['eps']
            Rd = y - Dd
        if SMr is not None:                                  # --slip-misfit: the slip rows' log misfit, sum (1/k) (log10(y / d)
            sr_ = SMr['rows']                                # / s)^2 (s each row's on its side), and the rows unscaled, as above
            with np.errstate(divide='ignore'):
                uq_ = np.log10(np.maximum(y[:, sr_], 0.0) / T.d[sr_])
            tr.setdefault('slip_log', np.zeros((K, n_rec)))[:, rec] = np.sum(
                sm_wk * (uq_ / np.where(uq_ < 0, SMr['s_lo'], SMr['s_hi'])) ** 2, 1)
            tr.setdefault('slip_log_out', np.zeros((K, n_rec)))[:, rec] = np.sum(np.abs(uq_) >= 1.0, 1)
            y = y.copy()
            y[:, sr_] /= SMr.get('eps', 1.0 / np.sqrt(2.0))
            Rd = y - Dd
        if getattr(T, 'split', None) is not None:            # --cfm slip rates: chi2 as the likelihood counts it
            C_ = _sp_terms(T, np.arange(nd), Rd)
            if getattr(T, 'temper', None) is not None:       # DATATYPES_EQUAL: each group's misfit recorded unweighted
                C_ = C_ / T.temper[:nd]
        else:
            C_ = Rd ** 2
        tr['chi2'][:, rec] = np.sum(C_, 1) / T.sigma ** 2
        mis_ = tr['chi2'][:, rec].copy()                     # chi2 as fitted: each group scored as the likelihood scores it
        if PMr is not None:                                  # each group (1 - b) x its Gaussian + b x its log misfit
            mis_ += b_p * (tr['paleo_misfit'][:, rec] - np.sum(C_[:, PMr['rows']], 1) / T.sigma ** 2)
        if MFr is not None:
            mis_ += b_m * (tr['mfd_misfit'][:, rec] - np.sum(C_[:, MFr['rows']], 1) / T.sigma ** 2)
        if SMr is not None:
            mis_ += b_s * (tr['slip_log'][:, rec] - np.sum(C_[:, SMr['rows']], 1) / T.sigma ** 2)
        tr.setdefault('misfit', np.zeros((K, n_rec)))[:, rec] = mis_
        if T.rlo is not None:                                # rows outside the band: always 0, a check
            tr.setdefault('band_outside', np.zeros((K, n_rec), np.int64))[:, rec] = T.band_outside(R)
        if T.balance is not None and nd < nbal:              # the balance term, and each type's relative shortfall
            tr.setdefault('balance', np.zeros((K, n_rec)))[:, rec] = np.sum(R[:, nd:nbal] ** 2, 1) / T.sigma ** 2
            tr.setdefault('balance_s', np.zeros((K, n_rec, nbal - nd)))[:, rec] = R[:, nd:nbal] / np.sqrt(T.balance['A'])
        if T.slip_sum is not None:                           # the summed slip rates' term, on its own
            R_ss = (_sp_terms(T, np.arange(nbal, T.n_fit), R[:, nbal:T.n_fit]) if getattr(T, 'split', None) is not None
                    else R[:, nbal:T.n_fit] ** 2)           # --slip-misfit exact: each cross-section's table
            tr.setdefault('slip_sum', np.zeros((K, n_rec)))[:, rec] = (np.sum(R_ss, 1) / T.sigma ** 2
                                                                       / float(getattr(T, 'temper_slip_sum', 1.0)))
        if T.mstep is not None and not T.mstep.get('joint'):   # --mfd-step-sd: the MFD steps' term, on its own
            tr.setdefault('mfd_step', np.zeros((K, n_rec)))[:, rec] = mstep_value(T.mstep, R)
        elif T.mstep is not None:                            # --mfd-gr-tau: each chain's curve, its b, ln(s m) and N(M>=5)
            tr.setdefault('mfd_gr_b', np.zeros((K, n_rec)))[:, rec] = T.mstep['theta'][:, 0]
            tr.setdefault('mfd_gr_level', np.zeros((K, n_rec)))[:, rec] = T.mstep['theta'][:, 1] + math.log(T.mstep['m_c'])
            tr.setdefault('mfd_gr_n5', np.zeros((K, n_rec)))[:, rec] = (T.mstep['m_c'] * np.exp(T.mstep['theta'][:, 1])
                                                                        * T.mstep['n5'][T.mstep['ib']])   # s m N(b), per year
        if T.lw is not None and T.lw.get('tau') is not None:   # --log-band: the soft band's penalty, on its own
            lr_ = T.lw['rows']
            with np.errstate(divide='ignore', invalid='ignore'):
                lq = np.log(np.maximum(R[:, lr_] + T.lw['DY'][lr_], 0.0)) - T.lw['LD'][lr_]
            # its chi2-equivalent, sum w' (log10(y / d) / tau)^2 (w' = 1/k for slip subsections), as chi2 and the balance and
            # summed-slip terms are recorded: twice its share of the negative log posterior
            tr.setdefault('log_band', np.zeros((K, n_rec)))[:, rec] = 2.0 * np.sum(_lw_sq(T, lr_, lq), 1)
            tr.setdefault('log_band_out', np.zeros((K, n_rec)))[:, rec] = np.sum(np.abs(lq) > T.lw['tau'] * LN10, 1)
        Xm = X if sc is None else X * sc
        tr['sum_m'][:, rec] = Xm.sum(1)
        tr['tscale'][:, rec] = y @ Dd / dd
        tr['mon'][:, rec] = Xm[:, monitor]
        tr['sweep'][rec] = (rec + 1) * thin
        if rec >= n_burn:
            tr['y'][:, rec - n_burn] = y
            if samp is not None:
                samp[:, rec - n_burn] = Xm
            if KD is not None:
                for k, row in KD['at'].get(rec - n_burn, ()):
                    x32 = Xm[k].astype(np.float32)
                    nz = np.flatnonzero(x32)
                    KD['rows'][row] = (nz.astype(np.int32), x32[nz])
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
                sparse = np.count_nonzero(Xm) <= 0.05 * Xm.size     # mostly zeros: each chain's nonzero rates only
                hist_now = t % stream.get('hist_every', 1) == 0
                above = np.zeros(T.n, np.int32)                      # chains whose rate is in a histogram bin above 0
                for k in range(K):
                    x = Xm[k]
                    # fewest ruptures carrying 90% / 99% of the draw's total rate: its rates, largest first, accumulated.
                    # Rates below 2^-60 of the total are under half the last bit of the running sum, so adding them
                    # changes nothing: only the rest are sorted
                    cum = np.cumsum(-np.sort(-x[x >= x.sum() * 2.0 ** -60]))
                    ST['n90'][k, t] = np.argmax(cum >= 0.9 * cum[-1]) + 1
                    ST['n99'][k, t] = np.argmax(cum >= 0.99 * cum[-1]) + 1
                    j = np.flatnonzero(x)
                    if hist_now:                                     # log-rate histogram; bin 0 holds everything below
                        b = np.clip(np.floor((np.log10(np.maximum(x[j], 1e-300)) - stream['hist_lo']) / stream['hist_w'])
                                    .astype(np.int64) + 1, 0, stream['hist_nb'] - 1)          # hist_lo, every 0 among them
                        hist_flat[b[b > 0] * T.n + j[b > 0]] += 1    # one entry per rupture in a chain: no repeats
                        above[j[b > 0]] += 1
                    if sparse:                                       # subsection and bin totals: the same sums in the
                        for key, A in INC.items():                   # same order, without the zero terms
                            ST['tot_' + key][k, t] = A[:, j] @ x[j]
                if hist_now:
                    ST['hist'][0] += K - above
                    ST['hcnt'][0] += K
                if not sparse:
                    for key in INC:
                        ST['tot_' + key][:, t] = (stream['inc_' + key] @ Xm.T).T
        if verbose and (rec % max(1, n_rec // 20) == 0 or rec == n_rec - 1):
            el = time.time() - t0
            print(f'[gibbs:{backend}] sweep {(rec + 1) * thin:6d}/{sweeps}  chi2 {"as fitted " if fitted else ""}per chain '
                  f'{np.round(tr["misfit"][:, rec], 0)}  '
                  f'pred/data scale {tr["tscale"][:, rec].mean():.3f}  ({el:.0f}s, {el / ((rec + 1) * thin):.3f}s/sweep)', flush=True)
    tr['final_X'] = X if sc is None else X * sc
    tr['seconds'] = time.time() - t0
    if backend == 'c':
        lib.ss_collapse(0, 1.0, 1.0, None)               # --collapse: off again, for the library's next caller
    tr['collapse'] = bool(collapse)
    tr['swap_rate'] = nacc / max(1, n_rec * (len(T.pairs[0]) if swaps else 1)) if swaps else np.zeros(K)
    tr['transfer_stats'] = tstats       # per chain: moves made, moves involving a rate above 1e-6
    if split_merge:                     # per chain: splits tried (a triple to split into), accepted; merges tried, accepted;
        tr['split_merge'] = smstats     # of nprop proposals per chain each sweep
        tr['split_merge_nprop'], tr['split_merge_scale'] = int(T.sm['nprop']), float(T.sm['scale'])
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
            sp.save_npz(f'{out}_kept.npz', kept_csr(KD, T.n), compressed=False)
            np.save(f'{out}_kept_index.npy', KD['index'])
            if os.path.exists(f'{out}_kept.npy'):          # an earlier run's dense draws under this name: replaced
                os.remove(f'{out}_kept.npy')
    elif KD is not None:
        tr['kept'], tr['kept_index'] = kept_csr(KD, T.n), KD['index']
    return tr


def report(tr, burn, thin, names=None):
    nb = burn // thin
    fitted = 'misfit' in tr and not np.array_equal(tr['misfit'], tr['chi2'])
    q = ({'chi2 as fitted': tr['misfit'][:, nb:], 'chi2 (Gaussians)': tr['chi2'][:, nb:]} if fitted else {'chi2': tr['chi2'][:, nb:]})
    q.update({'sum_m': tr['sum_m'][:, nb:], 'pred/data scale': tr['tscale'][:, nb:]})
    if 'mfd_gr_b' in tr:                             # --mfd-gr-tau: the curve's b and level, with R-hat and effective draws
        q.update({'MFD curve b': tr['mfd_gr_b'][:, nb:], 'MFD curve ln(s m)': tr['mfd_gr_level'][:, nb:],
                  'MFD curve N(M>=5)': tr['mfd_gr_n5'][:, nb:]})
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
    if 'balance' in tr:                              # down the middle: the balance term (not in chi2 above)
        sb = tr['balance_s'][:, nb:, :]
        Av = np.atleast_1d(tr['balance_A'])              # one A for every type, or each type's own
        print(f'balance term (down the middle, A = '
              + (f'{Av[0]:.4g}' if np.all(Av == Av[0]) else ', '.join(f'{v:.4g}' for v in Av))
              + f'): {tr["balance"][:, nb:].mean():.1f} on average, '
              f'not counted in chi2 above; each type\'s relative shortfall (sum of predictions / sum of targets - 1): '
              + ', '.join(f'{nm} {sb[..., k].mean():+.3f}' for k, nm in enumerate(tr['balance_types'])))
    if 'mfd_step' in tr:                             # --mfd-step-sd: the MFD steps' term (not in chi2 above)
        v = tr['mfd_step'][:, nb:]
        print(f'MFD steps (--mfd-step-sd {float(tr["mfd_step_sd"]):g}): {v.mean():.1f} on average (5-95% {np.quantile(v, .05):.1f}-'
              f'{np.quantile(v, .95):.1f}; near {int(tr["mfd_step_expect"])} if the steps scatter about the target\'s as it says), '
              'not counted in chi2 above')
    if 'slip_sum' in tr:                             # the summed slip rates' term (not in chi2 above)
        v = tr['slip_sum'][:, nb:]
        print(f'summed slip rates (--slip-sum): {v.mean():.1f} on average (5-95% {np.quantile(v, .05):.1f}-'
              f'{np.quantile(v, .95):.1f}; near {float(tr["slip_sum_expect"]):.1f} if the model matched the truth), not '
              'counted in chi2 above')
    def _wt(b_):
        return 'in place of the Gaussian' if b_ >= 1 else f'weighted {b_:g}, with the Gaussian\'s {1 - b_:g} (chi2 above counts the whole Gaussian)'
    if 'slip_log' in tr:                             # --slip-misfit split+log10, log10: the log misfit, in chi2 as fitted
        v, o = tr['slip_log'][:, nb:], tr['slip_log_out'][:, nb:]
        print(f'slip misfit (log10 part, sum (1/k) (log10(prediction / target) / s)^2, s each row\'s on its side, from its '
              f'effective sd): {v.mean():.1f} on average (5-95% {np.quantile(v, .05):.1f}-{np.quantile(v, .95):.1f}), in chi2 as fitted '
              f'{_wt(float(tr["slip_geo_b"]))}; slip-rate rows predicted 10x or more off their targets: {o.mean():.1f} per draw')
    if 'log_band' in tr:                           # --log-band: the soft band (not in chi2 above)
        v, o = tr['log_band'][:, nb:], tr['log_band_out'][:, nb:]
        print(f'soft band (--log-band; its chi2-equivalent, sum (log10(prediction / target) / tau)^2): {v.mean():.1f} on average (5-95% '
              f'{np.quantile(v, .05):.1f}-{np.quantile(v, .95):.1f}), not counted in chi2 above; rows predicted beyond its '
              f'tau of their targets: {o.mean():.1f} per draw')
    if 'mfd_gr_b' in tr and 'mfd_pdf_rows' in tr:    # --mfd-gr-tau: the bins about each draw's G-R curve, and its b and level
        v, bb, lv = tr['mfd_misfit'][:, nb:], tr['mfd_gr_b'][:, nb:], np.exp(tr['mfd_gr_level'][:, nb:])
        q_ = lambda x: f'{np.mean(x):.3f} (5-95% {np.quantile(x, .05):.3f}-{np.quantile(x, .95):.3f})'
        print(f'MFD misfit (--mfd-gr-tau {float(tr["mfd_gr_tau"]):g}, the bins about each draw\'s G-R curve): {v.mean():.1f} on average '
              f'(5-95% {np.quantile(v, .05):.1f}-{np.quantile(v, .95):.1f}; near {len(tr["mfd_pdf_rows"])} if they scatter about it as '
              f'the tolerance says), in the likelihood in place of the bins\' own PDFs; the curve\'s b {q_(bb)}, its level s m '
              f'(the N scaling times the magnitude-uncertainty factor) {q_(lv)}, its N(M>=5) s m N(b) {q_(tr["mfd_gr_n5"][:, nb:])} '
              'per year (the regional rate, through NZSHM22\'s partitioning)')
    elif 'mfd_misfit' in tr:                         # --mfd-misfit log10: in the likelihood in place of the bins' Gaussian
        v = tr['mfd_misfit'][:, nb:]
        print(f'MFD misfit (log10 part, sum (log10(total / R_j) / s_j)^2, s_j = log10(1 + sigma_j / R_j)): {v.mean():.1f} '
              f'on average (5-95% {np.quantile(v, .05):.1f}-{np.quantile(v, .95):.1f}), in the likelihood '
              f'{_wt(float(tr.get("mfd_log_b", 1.0)))}')
    if 'paleo_misfit' in tr:                         # --paleo-misfit: in the likelihood in place of the paleo rows' Gaussian
        v = tr['paleo_misfit'][:, nb:]
        print(f'paleo misfit (--paleo-misfit {tr["paleo_mode"]}; -2 ln of each site\'s recurrence-interval PDF, from its minimum, on '
              f'chi2\'s scale): {v.mean():.1f} on average (5-95% {np.quantile(v, .05):.1f}-{np.quantile(v, .95):.1f}), in the '
              f'likelihood {_wt(float(tr.get("paleo_b", 1.0)))}')
    if 'hyper_pi' in tr:                             # spike-and-slab: per-bin fraction switched on
        pi = tr['hyper_pi'][:, nb:, :]
        rh = split_rhat(pi)
        es = [ess(pi[..., k]) for k in range(pi.shape[2])]
        on = tr['n_on'][:, nb:]
        print(f'fraction switched on per bin: {pi.mean((0, 1)).min():.2e} to {pi.mean((0, 1)).max():.2e}; worst R-hat '
              f'{np.nanmax(rh):.3f}, smallest ESS {min(es):.0f}; ruptures on per draw {on.mean():.0f} '
              f'(R-hat {float(split_rhat(on[:, :, None])[0]):.3f}, ESS {ess(on):.0f})')


# ---------------------------------------------------------------- self-test against exact answers
def lb_grid(G, d, s2, se, a=-3.0):
    """E[log10 m] of two rates under a log-box prior on [a, 0] (edge sd se decades) and a 2-row Gaussian likelihood, by
    grid integration (the self-tests' reference)."""
    u = np.linspace(a - 6 * se if se > 0 else a, 0.0, 2001)
    U1, U2 = np.meshgrid(u, u, indexing='ij')
    f = lambda x: np.where(x < a, np.exp(-(x - a) ** 2 / (2 * se ** 2)) if se > 0 else 0.0, 1.0)
    M1, M2 = 10 ** U1, 10 ** U2
    P = np.exp(-((G[0, 0] * M1 + G[0, 1] * M2 - d[0]) ** 2 + (G[1, 0] * M1 + G[1, 1] * M2 - d[1]) ** 2) / (2 * s2)) * f(U1) * f(U2)
    P /= P.sum()
    return [np.sum(P * U1), np.sum(P * U2)]


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


def selftest_spikeslab(backends=('c', 'numpy'), gam=0.0, collapse=False):
    """Spike-and-slab: 2 rates in one bin, with different slab floors and soft tops and the fraction on integrated out
    under Beta(1, 1); P(on) and E[log10 m | on] against exact values from enumerating the on/off states on a grid.
    gam = 1/2: the square-root slab (density m^-1/2 below the top)."""
    from scipy.special import betaln
    print('spike-and-slab prior' + (' (square-root slab)' if gam else '') + (' (--collapse)' if collapse else '')
          + ': P(on) and E[log10 m | on] vs exact enumeration and grid integrals')
    Gs = sp.csc_matrix(np.array([[1.0, 0.7], [0.5, 1.0]]))
    d, sig, lo10, su, a_, b_ = np.array([0.6, 0.3]), 0.1, np.array([-3.0, -5.0]), np.array([0.05, 0.15]), 1.0, 1.0
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):                        # slab mass per grid node (trapezoid weights, normalized)
        w = np.full(len(ui), ui[1] - ui[0])
        w[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * w * 10.0 ** (gam * ui)   # tilt: m^gam per log m
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
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_, collapse=collapse)
        T.gam = gam
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
    print('SELFTEST (spike-and-slab' + (', square-root slab' if gam else '') + (', --collapse' if collapse else '') + ')',
          'PASSED' if ok else 'FAILED')
    return ok


def selftest_balance(backends=('c', 'numpy')):
    """The balance term (--down-the-middle) on a small problem with two slip-rate sections, two magnitude bins and a
    paleo row, under both weightings: each balance row's residual is sqrt(A_t) times its type's relative shortfall,
    computed independently here (sections as the mean of their subsections); the zero model costs A_t in every type;
    with 'equal', one A, each type's weight at most one extra measurement and exactly one for the type that sets A; with
    'zero-model', each type's A_t its zero-model misfit; and a run records chi2 and predictions over the data rows only,
    with the balance term and shortfalls on their own."""
    print('balance term (down the middle): its rows against a direct computation, and a run\'s bookkeeping')
    g = np.random.default_rng(3)
    k3, k2 = 1 / np.sqrt(3), 1 / np.sqrt(2)
    Gd = np.abs(g.normal(1.0, 0.4, (8, 6)))
    Gd[:3] *= k3
    Gd[3:5] *= k2
    Gd[5], Gd[6] = [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]
    Gd[5:7] *= 4.0
    d = np.r_[np.full(3, 2.0 * k3), np.full(2, 1.2 * k2), 8.0, 7.0, 3.0]
    P = dict(G=sp.csr_matrix(Gd), d=d, block=np.array([0] * 5 + [1] * 2 + [2]), block_names=['S', 'MFD', 'P'],
             section=np.r_[0, 0, 0, 1, 1, -1, -1, -1], row_scale=np.r_[[k3] * 3, [k2] * 2, 1.0, 1.0, 1.0])
    m = np.abs(g.normal(0.5, 0.3, 6))
    y = Gd @ m
    T_ = [2.0 + 1.2, 8.0 + 7.0, 3.0]                     # each type's sum of targets (sections: 2 and 1.2)
    s_ = [((y[:3] / k3).mean() + (y[3:5] / k2).mean()) / T_[0] - 1, (y[5] + y[6]) / T_[1] - 1, y[7] / T_[2] - 1]
    Z_ = [2.0 ** 2 + 1.2 ** 2, 8.0 ** 2 + 7.0 ** 2, 3.0 ** 2]   # each type's zero-model misfit (each section once)
    ok = True
    for wt in ('equal', 'zero-model'):
        B, b, A, info = balance_rows(P, weights=wt)
        zero = (B @ np.zeros(6) - b) ** 2
        share = [t['weight'] * t['n'] for t in info]
        ok_r = bool(np.allclose(B @ m - b, np.sqrt(A) * np.array(s_), rtol=1e-12, atol=1e-12)) and bool(np.allclose(zero, A))
        if wt == 'equal':                                # one A, set by the type it gives one extra measurement's weight
            ok_r &= (bool(np.allclose(A, min(t ** 2 / n_ for t, n_ in zip(T_, (2, 2, 1))))) and max(share) <= 1 + 1e-12
                     and bool(np.isclose(max(share), 1.0)) and all(A[0] <= t['zero_misfit'] + 1e-12 for t in info))
        else:                                            # each type's own A, its zero-model misfit
            ok_r &= bool(np.allclose(A, Z_)) and bool(np.allclose([t['zero_misfit'] for t in info], Z_))
        ok &= ok_r
        print(f'   {wt}: relative shortfalls {np.round(s_, 4)} recovered; zero model {np.round(zero, 4)} (A {np.round(A, 4)}); '
              f'shares of a measurement {np.round(share, 3)}: ' + ('yes' if ok_r else 'NO'))
        for be in backends:
            T = Target(sp.vstack([P['G'], B]), np.r_[d, b], 0.5, 0.0, 10.0)
            T.n_data, T.balance = 8, dict(A=A, names=[t['name'] for t in info])
            tr = sample(T, np.full((2, 6), 0.3), 60, 0, 1, be, 1, 5, monitor=np.arange(6), verbose=False, keep_samples=False)
            M_ = tr['mon']                                # every rate, every draw
            Y = M_ @ Gd.T
            chi = np.sum((Y - d) ** 2, 2) / 0.25
            bal = np.sum((M_ @ B.T.toarray() - b) ** 2, 2) / 0.25
            ok_b = (tr['y'].shape[2] == 8 and np.allclose(tr['y'], Y, rtol=1e-5) and np.allclose(tr['chi2'], chi, rtol=1e-9)
                    and np.allclose(tr['balance'], bal, rtol=1e-9)
                    and np.allclose(tr['balance_s'], (M_ @ B.T.toarray() - b) / np.sqrt(A), rtol=1e-9, atol=1e-12))
            ok &= ok_b
            print(f'   {wt:10s} {be:5s} chi2 over the 8 data rows and the balance term recorded apart: ' + ('yes' if ok_b else 'NO'))
    # with the summed slip rates' rows (--slip-sum) after the balance term's: three terms, each recorded on its own
    Cs, cs = np.abs(g.normal(0.8, 0.3, (2, 6))), np.array([1.5, 2.5])
    for be in backends:
        T = Target(sp.vstack([P['G'], B, sp.csr_matrix(Cs)]), np.r_[d, b, cs], 0.5, 0.0, 10.0)
        T.n_data, T.n_bal, T.balance = 8, 8 + len(b), dict(A=A, names=[t['name'] for t in info])
        T.slip_sum = dict(expect=2.0)
        tr = sample(T, np.full((2, 6), 0.3), 60, 0, 1, be, 1, 6, monitor=np.arange(6), verbose=False, keep_samples=False)
        M_ = tr['mon']
        chi = np.sum((M_ @ Gd.T - d) ** 2, 2) / 0.25
        bal = np.sum((M_ @ B.T.toarray() - b) ** 2, 2) / 0.25
        ssum = np.sum((M_ @ Cs.T - cs) ** 2, 2) / 0.25
        ok_s = (np.allclose(tr['chi2'], chi, rtol=1e-9) and np.allclose(tr['balance'], bal, rtol=1e-9)
                and np.allclose(tr['slip_sum'], ssum, rtol=1e-9) and tr['balance_s'].shape[2] == len(b))
        ok &= ok_s
        print(f'   with --slip-sum {be:5s} chi2, the balance term and the summed slip rates\' term recorded apart: '
              + ('yes' if ok_s else 'NO'))
    print('SELFTEST (balance term)', 'PASSED' if ok else 'FAILED')
    return bool(ok)


def selftest_swaps(backends=('c', 'numpy')):
    """Swap moves between two near-duplicate columns under a log-box prior, as spike-and-slab and theta use them, against
    a grid integral: the main suite's 'log-box prior + swaps' check on its own (same problem, lengths and seed)."""
    print('swap moves between two near-duplicate columns, log-box prior: E[log10 m] vs grid integral')
    G2, d2 = np.array([[1.0, 0.98], [0.5, 0.53]]), np.array([0.4, 0.2])
    pairs = (np.array([0]), np.array([1]), np.array([G2[:, 0] @ G2[:, 1]]))
    ex = lb_grid(G2, d2, 0.09, 0.5)
    ok = True
    for be in backends:
        T = Target(G2, d2, 0.3, 0.0, 1.0, logbox=(-3, 0, 0.5), pairs=pairs)
        tr = sample(T, np.tile([0.4, 1e-3], (4, 1)), 40000 if be == 'c' else 8000, 1000, 1, backend=be, monitor=[0, 1],
                    verbose=False, keep_samples=False, swaps=True)
        x = np.log10(tr['mon'][:, 1000:])
        zs = [(x[..., j].mean() - ex[j]) / (x[..., j].std() / np.sqrt(ess(x[..., j]))) for j in (0, 1)]
        ok &= all(abs(z) < 4 for z in zs)
        print(f'   {be:6s} ({x[..., 0].mean():+.4f}, {x[..., 1].mean():+.4f}) exact ({ex[0]:+.4f}, {ex[1]:+.4f}) '
              f'z = ({zs[0]:+.2f}, {zs[1]:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
    print('SELFTEST (swap moves)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_mfdlog(backends=('c', 'numpy'), b=1.0):
    """The MFD bins' log10 misfit (--mfd-misfit log10; with b = 1/2, gaussian+log10) on two spike-and-slab rates: a Gaussian
    row, and a bin row holding both, scaled out of the Gaussian (b = 1/2: scaled by 1/sqrt 2) and scored by b (log10(total
    / R) / s)^2 through the soft band's machinery, with positivity on it and swap moves, against a grid integral: P(on) and
    E[log10 m | on] of each rate, and the recorded misfits."""
    from scipy.special import betaln
    from paleo_misfit import PALEO_EPS
    mode_ = 'log10' if b >= 1 else 'gaussian+log10'
    print(f'MFD misfit (--mfd-misfit {mode_}): spike-and-slab with positivity and swaps vs a grid integral')
    ok = True
    c0, d0, sig, Rb, sb = np.array([1.0, 0.4]), 0.03, 0.02, 0.05, np.log10(1.15)
    lo10, su, a_, b_ = np.array([-4.0, -4.0]), np.array([0.1, 0.1]), 1.0, 1.0
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):
        ww = np.full(len(ui), ui[1] - ui[0])
        ww[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * ww
        gw.append(f / f.sum())
    pg = [np.exp(betaln(a_ + k, b_ + 2 - k) - betaln(a_, b_)) for k in range(3)]
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')

    def Lb(m1, m2):
        m1, m2 = np.broadcast_arrays(m1, m2)
        with np.errstate(divide='ignore'):
            lq = np.where(m1 + m2 > 0, np.log10(np.maximum(m1 + m2, 1e-300) / Rb) / sb, np.inf)
        return np.exp(-0.5 * (c0[0] * m1 + c0[1] * m2 - d0) ** 2 / sig ** 2 - 0.5 * b * lq ** 2
                      - 0.5 * (1.0 - b) * (m1 + m2 - Rb) ** 2 / sig ** 2)
    L1, L2_, L12 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1]), Lb(M1, M2) * gw[0][:, None] * gw[1][None, :]
    w10, w01, w11 = pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2_), pg[2] * L12.sum()   # both off: impossible
    W_ = w10 + w01 + w11
    p_on = [(w10 + w11) / W_, (w01 + w11) / W_]
    e_u = [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
           (pg[1] * (gw[1] * u[1]) @ L2_ + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)]
    eps = PALEO_EPS if b >= 1 else 1.0 / np.sqrt(2.0)
    Gs = np.vstack([c0, np.array([1.0, 1.0]) * eps])
    dv = np.array([d0, eps * Rb])
    for be in backends:
        T = Target(sp.csc_matrix(Gs), dv, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0]), np.array([1]), np.array([Gs[:, 0] @ Gs[:, 1]])))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        T.set_groups(np.array([1]), np.zeros(0, np.int64), np.zeros(0))
        T.lw = dict(W=np.array([0.0, b / (2.0 * (sb * LN10) ** 2)]), LD=np.log(dv), DY=np.ascontiguousarray(dv),
                    rows=np.array([], np.int64), tau=None,
                    MF=dict(rows=np.array([1]), s=np.array([sb]), eps=eps, d_data=np.array([d0, Rb]), b=b))
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.03, 0.02], (K, 1)), sw, sw // 10, 1, be, 1, 37, monitor=(0, 1), verbose=False,
                    keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        ok &= bool((Mo > 0).any(-1).all())                # never both off: the bin's total must stay above 0
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            pr = on.mean()
            if on.std() == 0:
                zp = 0.0 if on.size * min(p_on[i], 1 - p_on[i]) < 5 else np.inf
            else:
                zp = (pr - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / ess(on))
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * pr)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} rate {i}: P(on) {pr:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 m | on] {uu.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
        mm_ = Mo.reshape(-1, 2)                         # the recorded misfit, and chi2 over the unscaled rows
        mf_ = (np.log10(mm_.sum(1) / Rb) / sb) ** 2
        ch_ = ((mm_ @ c0 - d0) ** 2 + (mm_.sum(1) - Rb) ** 2) / sig ** 2
        fit_ = ch_ + b * (mf_ - (mm_.sum(1) - Rb) ** 2 / sig ** 2)   # chi2 as fitted
        ok &= bool(np.allclose(mf_, tr['mfd_misfit'][:, sw // 10:].ravel(), rtol=1e-7, atol=1e-9)
                   and np.allclose(ch_, tr['chi2'][:, sw // 10:].ravel(), rtol=1e-6, atol=1e-9)
                   and np.allclose(fit_, tr['misfit'][:, sw // 10:].ravel(), rtol=1e-6, atol=1e-8))
    print(f'SELFTEST (MFD misfit {mode_})', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_mfdpdf(backends=('c', 'numpy')):
    """The MFD bins scored by their own PDFs (--mfd-misfit pdf, MFD_GR) on two spike-and-slab rates: a Gaussian row, and a
    bin row holding both, scaled out of the Gaussian and scored by -2 ln of a lopsided curve's density in log rate (the
    paleo curves' tables, alone in them: no paleo rows), with positivity on it and swap moves, against a grid integral:
    P(on) and E[log10 m | on] of each rate, the recorded misfit and chi2 as fitted."""
    from scipy.special import betaln
    from paleo_misfit import PALEO_EPS, curve_tables, fit_site, site_misfit
    print('MFD misfit (--mfd-misfit pdf): spike-and-slab with positivity and swaps vs a grid integral')
    ok = True
    best = 0.05                                          # the curve: its minimum at 0.05, quantiles lopsided above
    th = fit_site(best, np.array([0.02, 0.033, 0.085, 0.16]), 'log')[0]
    tab = curve_tables([th], [best], 'log')
    c0, d0, sig = np.array([1.0, 0.4]), 0.03, 0.02
    lo10, su, a_, b_ = np.array([-4.0, -4.0]), np.array([0.1, 0.1]), 1.0, 1.0
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):
        ww = np.full(len(ui), ui[1] - ui[0])
        ww[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * ww
        gw.append(f / f.sum())
    pg = [np.exp(betaln(a_ + k, b_ + 2 - k) - betaln(a_, b_)) for k in range(3)]
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')

    def Lb(m1, m2):
        m1, m2 = np.broadcast_arrays(m1, m2)
        tot = m1 + m2
        with np.errstate(divide='ignore', invalid='ignore'):
            cur = np.where(tot > 0, site_misfit(th, best, np.log(np.maximum(tot, 1e-300)), 'log'), np.inf)
        return np.exp(-0.5 * (c0[0] * m1 + c0[1] * m2 - d0) ** 2 / sig ** 2 - 0.5 * cur)
    L1, L2_, L12 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1]), Lb(M1, M2) * gw[0][:, None] * gw[1][None, :]
    w10, w01, w11 = pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2_), pg[2] * L12.sum()   # both off: impossible
    W_ = w10 + w01 + w11
    p_on = [(w10 + w11) / W_, (w01 + w11) / W_]
    e_u = [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
           (pg[1] * (gw[1] * u[1]) @ L2_ + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)]
    eps, sdb = PALEO_EPS, 0.02                           # the bin row: total / sdb, scaled out by eps
    Gs = np.vstack([c0, np.array([1.0, 1.0]) * eps / sdb])
    dv = np.array([d0, eps * best / sdb])
    for be in backends:
        T = Target(sp.csc_matrix(Gs), dv, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0]), np.array([1]), np.array([c0[0] * c0[1]])))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        T.set_groups(np.array([1]), np.zeros(0, np.int64), np.zeros(0))
        pm = dict(I=np.array([-1, 0], np.int32), V0=np.ascontiguousarray(tab['U0'] - np.log(sdb / eps)), F=tab['F'],
                  D=tab['DF'], K=tab['KAPPA'], H=tab['H'], NG=tab['NG'], rows=np.array([], np.int64), eps=1.0,
                  d_data=np.array([d0, best / sdb]), mode='gaussian', b=0.0)
        T.lw = dict(W=np.zeros(2), LD=np.zeros(2), DY=np.ascontiguousarray(dv), rows=np.array([], np.int64), tau=None, PM=pm,
                    MF=dict(rows=np.array([1]), pdf=True, idx=np.array([0]), th=th[None, :], best=np.array([best]),
                            sd=np.array([sdb]), E=np.array([1.0]), V=np.array([2.0]), eps=eps, d_data=np.array([d0, best / sdb]),
                            b=1.0))
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.03, 0.02], (K, 1)), sw, sw // 10, 1, be, 1, 41, monitor=(0, 1), verbose=False,
                    keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        ok &= bool((Mo > 0).any(-1).all())                # never both off: the bin's total must stay above 0
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            pr = on.mean()
            if on.std() == 0:
                zp = 0.0 if on.size * min(p_on[i], 1 - p_on[i]) < 5 else np.inf
            else:
                zp = (pr - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / ess(on))
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * pr)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} rate {i}: P(on) {pr:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 m | on] {uu.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
        mm_ = Mo.reshape(-1, 2)                         # the recorded misfit (the tables' interpolant: to 1e-6 of the curve)
        tot_ = mm_.sum(1)
        mf_ = site_misfit(th, best, np.log(tot_), 'log')
        ch_ = ((mm_ @ c0 - d0) ** 2 + ((tot_ - best) / sdb) ** 2) / sig ** 2
        fit_ = ch_ + (mf_ - ((tot_ - best) / sdb) ** 2 / sig ** 2)   # chi2 as fitted: the bin's Gaussian replaced by its curve
        ok &= bool(np.allclose(mf_, tr['mfd_misfit'][:, sw // 10:].ravel(), rtol=1e-5, atol=1e-6)
                   and np.allclose(ch_, tr['chi2'][:, sw // 10:].ravel(), rtol=1e-6, atol=1e-9)
                   and np.allclose(fit_, tr['misfit'][:, sw // 10:].ravel(), rtol=1e-5, atol=1e-6))
    print('SELFTEST (MFD misfit pdf)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_mfdstep(backends=('c',)):
    """The MFD steps' term (--mfd-step-sd) on four spike-and-slab rates in three magnitude bins (one, two, one), with a
    Gaussian row across all four, Gaussian bin rows and swap moves (within the middle bin and across bins), against a
    grid integral: P(on) of the middle bin's two rates, E[log10 m | on] of each, and the recorded term. The end bins'
    rates must stay on (the term is infinite with a bin at 0), and the middle bin's two never both off. C backend only.
    sigma_s = 2 (tau 0.2 decades a step): the term ties the bins' shape while their level and tilt stay loose, so the
    single-rate updates mix slowly at tight tolerances (at 0.5, some 30 effective draws of 144,000 in this toy); the
    standard errors are the spread of the 16 chains' own means."""
    from scipy.special import betaln
    print('MFD steps (--mfd-step-sd): spike-and-slab with swaps vs a grid integral')
    ok = True
    c0, d0, sig = np.array([1.0, 0.4, 0.7, 1.0]), 0.06, 0.02
    db = np.array([0.05, 0.03, 0.02])                  # the bins' targets
    sig_s, width = 2.0, 0.1
    Gs = np.array([c0, [1.0, 0, 0, 0], [0, 1.0, 1.0, 0], [0, 0, 0, 1.0]])
    dv = np.r_[d0, db]
    lo10, su = -4.0, 0.1
    U = np.linspace(lo10, 0.8, 101)
    ww = np.full(len(U), U[1] - U[0])
    ww[[0, -1]] *= 0.5
    gw = np.where(U > 0, np.exp(-U ** 2 / (2 * su ** 2)), 1.0) * ww
    gw /= gw.sum()
    M = 10.0 ** U
    pg = {k: np.exp(betaln(1 + k, 1 + 4 - k) - betaln(1, 1)) for k in (3, 4)}
    tau = width * sig_s * LN10
    e = np.diff(np.log(db))

    def logL(m0, m1, m2, m3):
        y = [m0, m1 + m2, m3]
        g = (c0[0] * m0 + c0[1] * m1 + c0[2] * m2 + c0[3] * m3 - d0) ** 2 + sum((y[b] - db[b]) ** 2 for b in range(3))
        lam = [np.log(v) for v in y]
        ep = [lam[1] - lam[0] - e[0], lam[2] - lam[1] - e[1]]
        mb = 0.5 * (ep[0] + ep[1])
        return -0.5 * g / sig ** 2 - 0.5 * ((ep[0] - mb) ** 2 + (ep[1] - mb) ** 2) / tau ** 2
    # the three allowed configurations of the middle bin (both on; r1 alone; r2 alone), the end rates always on
    acc = {k: 0.0 for k in ('W', 'p1', 'p2')}
    eu = np.zeros(4); nu = np.zeros(4)
    ref = logL(db[0], db[1] / 2, db[1] / 2, db[2])
    for a0 in range(len(U)):
        m0 = M[a0]
        # both on: m1, m2, m3 on the grid
        L = np.exp(logL(m0, M[:, None, None], M[None, :, None], M[None, None, :]) - ref)
        w = L * (gw[a0] * gw[:, None, None] * gw[None, :, None] * gw[None, None, :]) * pg[4]
        # r1 alone, r2 alone
        L1 = np.exp(logL(m0, M[:, None], 0.0, M[None, :]) - ref) * (gw[a0] * gw[:, None] * gw[None, :]) * pg[3]
        L2 = np.exp(logL(m0, 0.0, M[:, None], M[None, :]) - ref) * (gw[a0] * gw[:, None] * gw[None, :]) * pg[3]
        W_ = w.sum() + L1.sum() + L2.sum()
        acc['W'] += W_
        acc['p1'] += w.sum() + L1.sum()
        acc['p2'] += w.sum() + L2.sum()
        eu[0] += U[a0] * W_
        eu[1] += (w.sum((1, 2)) @ U) + (L1.sum(1) @ U)
        eu[2] += (w.sum((0, 2)) @ U) + (L2.sum(1) @ U)
        eu[3] += (w.sum((0, 1)) @ U) + (L1.sum(0) @ U) + (L2.sum(0) @ U)
    p_on = np.array([1.0, acc['p1'] / acc['W'], acc['p2'] / acc['W'], 1.0])
    e_u = eu / np.array([acc['W'], acc['p1'], acc['p2'], acc['W']])
    for be in backends:
        T = Target(sp.csc_matrix(Gs), dv, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0, 1, 2]), np.array([1, 2, 3]),
                          np.array([Gs[:, i] @ Gs[:, i + 1] for i in range(3)])))
        T.zlo_vec = np.ascontiguousarray(np.full(4, lo10 * LN10))
        T.ku_vec = np.ascontiguousarray(np.full(4, 1.0 / (2.0 * (su * LN10) ** 2)))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(4, np.int64), B=1, nb=np.array([4.0]), a=1.0, b=1.0)
        T.mstep = mstep_build(T, np.array([1, 2, 3]), np.array([1, 2, 3]), sig_s, width)
        K, sw = 16, 40000
        tr = sample(T, np.tile([0.05, 0.015, 0.015, 0.02], (K, 1)), sw, sw // 10, 1, be, 1, 41, monitor=(0, 1, 2, 3),
                    verbose=False, keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        ok &= bool((Mo[..., 0] > 0).all() and (Mo[..., 3] > 0).all() and ((Mo[..., 1] > 0) | (Mo[..., 2] > 0)).all())
        for i in range(4):
            on = Mo[:, :, i] > 0
            pc = on.mean(1)                             # each chain's own estimates; their spread gives the errors
            uc = np.array([np.log10(Mo[k_, :, i][on[k_]]).mean() for k_ in range(K)])
            se_p, se_u = pc.std(ddof=1) / np.sqrt(K), uc.std(ddof=1) / np.sqrt(K)
            zp = 0.0 if se_p == 0 and abs(pc.mean() - p_on[i]) < 1e-12 else (pc.mean() - p_on[i]) / max(se_p, 1e-300)
            zu = (uc.mean() - e_u[i]) / se_u
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} rate {i}: P(on) {pc.mean():.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 m | on] {uc.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f})')
        mm_ = Mo.reshape(-1, 4)                        # the recorded term against its value at the draws
        y_ = np.c_[mm_[:, 0], mm_[:, 1] + mm_[:, 2], mm_[:, 3]]
        ep_ = np.diff(np.log(y_), axis=1) - e
        S_ = np.sum((ep_ - ep_.mean(1, keepdims=True)) ** 2, 1) / tau ** 2
        ok &= bool(np.allclose(S_, tr['mfd_step'][:, sw // 10:].ravel(), rtol=1e-7, atol=1e-9))
        print(f'   {be:5s} the term: {S_.mean():.3f} on average; swap acceptance {tr["swap_rate"].mean():.2f}')
    print('SELFTEST (MFD steps)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_mfdjoint(backends=('c',)):
    """The bins scored jointly about each draw's G-R curve (--mfd-gr-tau) on three spike-and-slab rates, one per magnitude
    bin, with a Gaussian row across all three and the bins' rows scaled out of the Gaussian (as with MFD_GR), and swap moves,
    against a sum over the curve's grids (b on 5 values, the level on 9: the sampler draws on the same grids) of a grid
    integral over the rates: E[log10 m] of each rate, the posterior of b on its grid and E[L]. The rates stay on (the term is
    infinite with a bin at 0). A toy shape, a G-R tilt pivoting at M6; MFD_GR's priors on b and the level; tau 0.2 decades a
    step (tight tolerances mix slowly in a toy where every rate is a whole bin, as with --mfd-step-sd). C backend only;
    the standard errors are the spread of the 16 chains' own means."""
    print('MFD bins jointly (--mfd-gr-tau): spike-and-slab with swaps, b and the level drawn, vs a grid integral')
    ok = True
    c0, d0, sig, kap = np.array([1.0, 0.6, 0.8]), 0.06, 0.02, 1e-3
    db = np.array([0.05, 0.03, 0.02])                  # the bins' targets (their rows scaled by kap, out of the Gaussian)
    Gs = np.vstack([c0, kap * np.eye(3)])
    dv = np.r_[d0, kap * db]
    lo10, su = -4.0, 0.1
    U = np.linspace(lo10, 0.8, 81)
    ww = np.full(len(U), U[1] - U[0])
    ww[[0, -1]] *= 0.5
    gw = np.where(U > 0, np.exp(-U ** 2 / (2 * su ** 2)), 1.0) * ww
    gw /= gw.sum()
    M = 10.0 ** U
    bg0 = np.linspace(0.75, 1.15, 5)
    Mj = np.array([7.05, 7.15, 7.25])
    shape = np.full((4, len(bg0)), np.nan)
    shape[1:] = -LN10 * (bg0[None, :] - 0.959) * (Mj[:, None] - 6.0)
    P = dict(mfd_gr='0.823,0.959,1.089,2.65,3.4,4.57,0.66,1.41,1,1.15294,1.30588', mfd_gr_bgrid=bg0, mfd_gr_shape=shape)
    for be in backends:
        T = Target(sp.csc_matrix(Gs), dv, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0, 1]), np.array([1, 2]), np.array([Gs[:, i] @ Gs[:, i + 1] for i in range(2)])))
        T.zlo_vec = np.ascontiguousarray(np.full(3, lo10 * LN10))
        T.ku_vec = np.ascontiguousarray(np.full(3, 1.0 / (2.0 * (su * LN10) ** 2)))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(3, np.int64), B=1, nb=np.array([3.0]), a=1.0, b=1.0)
        K, sw = 16, 40000
        T.mstep = mjoint_build(T, P, np.array([1, 2, 3]), np.array([1, 2, 3]), 0.2, 0.1, K, n_b=5, n_l=9)
        ms = T.mstep
        if be == backends[0]:                          # the exact posterior, on the sampler's own grids of b and L
            Q = ms['Q']
            m0, m1, m2 = M[:, None, None], M[None, :, None], M[None, None, :]
            g_ = (c0[0] * m0 + c0[1] * m1 + c0[2] * m2 - d0) ** 2 + kap ** 2 * ((m0 - db[0]) ** 2 + (m1 - db[1]) ** 2
                                                                               + (m2 - db[2]) ** 2)
            lw_ = np.log(gw[:, None, None] * gw[None, :, None] * gw[None, None, :]) - 0.5 * g_ / sig ** 2
            lr = [np.log(M)[:, None, None] - np.log(db[0]), np.log(M)[None, :, None] - np.log(db[1]),
                  np.log(M)[None, None, :] - np.log(db[2])]
            Z, Eu, Pb, EL = 0.0, np.zeros(3), np.zeros(len(ms['bgrid'])), 0.0
            res = []
            for i_, b_ in enumerate(ms['bgrid']):
                for l_, L_ in enumerate(ms['lgrid']):
                    rho = [lr[j] - L_ - ms['shape'][j, i_] for j in range(3)]
                    S_ = sum(Q[a_, c_] * rho[a_] * rho[c_] for a_ in range(3) for c_ in range(3))
                    res.append((i_, l_, ms['lpb'][i_] + ms['lpl'][l_] + lw_ - 0.5 * S_))
            top = max(float(r_[2].max()) for r_ in res)
            for i_, l_, lp_ in res:
                w_ = np.exp(lp_ - top)
                W_ = w_.sum()
                Z += W_
                Pb[i_] += W_
                EL += ms['lgrid'][l_] * W_
                Eu += np.array([w_.sum((1, 2)) @ U, w_.sum((0, 2)) @ U, w_.sum((0, 1)) @ U])
            Eu, Pb, EL = Eu / Z, Pb / Z, EL / Z
        tr = sample(T, np.tile([0.05, 0.02, 0.015], (K, 1)), sw, sw // 10, 1, be, 1, 43, monitor=(0, 1, 2),
                    verbose=False, keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        ok &= bool((Mo > 0).all())
        for i in range(3):
            uc = np.log10(Mo[:, :, i]).mean(1)
            z_ = (uc.mean() - Eu[i]) / (uc.std(ddof=1) / np.sqrt(K))
            ok &= abs(z_) < 4
            print(f'   {be:5s} rate {i}: E[log10 m] {uc.mean():.4f} (exact {Eu[i]:.4f}, z {z_:+.2f})')
        bb = tr['mfd_gr_b'][:, sw // 10:]
        for i_, b_ in enumerate(ms['bgrid']):
            pc = np.isclose(bb, b_).mean(1)
            se = pc.std(ddof=1) / np.sqrt(K)
            z_ = 0.0 if se == 0 and abs(pc.mean() - Pb[i_]) < 1e-3 else (pc.mean() - Pb[i_]) / max(se, 1e-300)
            ok &= abs(z_) < 4
            print(f'   {be:5s} b = {b_:.2f}: posterior {pc.mean():.4f} (exact {Pb[i_]:.4f}, z {z_:+.2f})')
        lc = (tr['mfd_gr_level'][:, sw // 10:] - math.log(ms['m_c'])).mean(1)
        z_ = (lc.mean() - EL) / (lc.std(ddof=1) / np.sqrt(K))
        ok &= abs(z_) < 4
        print(f'   {be:5s} E[L] {lc.mean():.4f} (exact {EL:.4f}, z {z_:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
    print('SELFTEST (MFD bins jointly)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_split(backends=('c', 'numpy')):
    """The split misfit (--cfm-sliprates: a row weighed (s_ref / s_lo)^2 below its target, (s_ref / s_hi)^2 above) on two
    spike-and-slab rates and two rows, one of them strongly lopsided, with swap moves, without and with the soft band on
    it, against a grid integral: P(on) and E[log10 m | on] of each rate, and chi2 as the split misfit counts it."""
    from scipy.special import betaln
    print('split misfit (--cfm-sliprates): spike-and-slab with swaps vs a grid integral')
    ok = True
    G2 = np.array([[1.0, 0.6], [0.4, 1.0]])
    d2, sig, L2v, H2v = np.array([0.05, 0.04]), 0.02, np.array([1.0, 6.0]), np.array([1.0, 0.15])
    lo10, su, a_, b_ = np.array([-4.0, -4.0]), np.array([0.1, 0.1]), 1.0, 1.0
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):
        ww = np.full(len(ui), ui[1] - ui[0])
        ww[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * ww
        gw.append(f / f.sum())
    pg = [np.exp(betaln(a_ + k, b_ + 2 - k) - betaln(a_, b_)) for k in range(3)]
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')
    for tau in (None, 0.3):
        wl = 0.0 if tau is None else 1.0 / (2.0 * (tau * LN10) ** 2)

        def Lb(m1, m2):
            m1, m2 = np.broadcast_arrays(m1, m2)
            out = np.zeros(m1.shape)
            for j in range(2):
                y = G2[j, 0] * m1 + G2[j, 1] * m2
                e = y - d2[j]
                out += np.where(e < 0, L2v[j], H2v[j]) * e * e / (2 * sig ** 2)
                if j == 1 and wl > 0:
                    with np.errstate(divide='ignore'):
                        out += np.where(y > 0, wl * np.log(np.maximum(y, 1e-300) / d2[j]) ** 2, np.inf)
            return np.exp(-out), out
        L1, L2_, L12 = Lb(10 ** u[0], 0.0)[0], Lb(0.0, 10 ** u[1])[0], Lb(M1, M2)[0] * gw[0][:, None] * gw[1][None, :]
        w00 = pg[0] * Lb(0.0, 0.0)[0]
        w10, w01, w11 = pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2_), pg[2] * L12.sum()
        W_ = w00 + w10 + w01 + w11
        p_on = [(w10 + w11) / W_, (w01 + w11) / W_]
        e_u = [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
               (pg[1] * (gw[1] * u[1]) @ L2_ + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)]
        for be in backends:
            T = Target(sp.csc_matrix(G2), d2, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                       pairs=(np.array([0]), np.array([1]), np.array([G2[:, 0] @ G2[:, 1]])))
            T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
            T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
            T.pkind = 3
            T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
            T.split = dict(L2=np.ascontiguousarray(L2v), H2=np.ascontiguousarray(H2v))
            if wl > 0:
                T.lw = dict(W=np.array([0.0, wl]), LD=np.log(d2), DY=np.ascontiguousarray(d2), rows=np.array([1]), tau=tau)
            K, sw = (8, 40000) if be == 'c' else (4, 4000)
            tr = sample(T, np.tile([0.03, 0.02], (K, 1)), sw, sw // 10, 1, be, 1, 31, monitor=(0, 1), verbose=False,
                        keep_samples=False, swaps=True)
            Mo = tr['mon'][:, sw // 10:, :]
            for i in range(2):
                on = (Mo[:, :, i] > 0).astype(float)
                pr = on.mean()
                if on.std() == 0:
                    zp = 0.0 if on.size * min(p_on[i], 1 - p_on[i]) < 5 else np.inf
                else:
                    zp = (pr - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / ess(on))
                uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
                zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * pr)))
                ok &= abs(zp) < 4 and abs(zu) < 4
                print(f'   {be:5s} {"soft band" if wl else "         "} rate {i}: P(on) {pr:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); '
                      f'E[log10 m | on] {uu.mean():.4f} (exact {e_u[i]:.4f}, z {zu:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
            m_ = Mo[:, :, :].reshape(-1, 2)                    # chi2 recorded as the split misfit counts it
            e_ = m_ @ G2.T - d2
            chi_ = np.sum(np.where(e_ < 0, L2v, H2v) * e_ ** 2, 1) / sig ** 2
            ok &= bool(np.allclose(np.sort(chi_), np.sort(tr['chi2'][:, sw // 10:].ravel()), rtol=1e-6, atol=1e-9))
    print('SELFTEST (split misfit)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_exact(backends=('c', 'numpy'), collapse=False, off_bounds=True):
    """--slip-misfit exact: two spike-and-slab rates and two rows scored by tabulated misfits (each row's likelihood a
    lognormal density in its prediction, sd 0.7 and 0.3 in ln: quasiconvex, not log-concave; the cubic Hermite tables the
    kernel reads, continued linearly), with swap moves, against a grid integral of the same interpolant: P(on) and
    E[log10 m | on] of each rate, and the recorded chi2 (the tables' misfit). The slice step's floor is the quasiconvex one."""
    from scipy.special import betaln
    from cru_common import hermite_eval, quasiconvex_table
    print('exact slip misfit (--slip-misfit exact): tabulated lognormal densities, spike-and-slab with swaps vs a grid integral'
          + (' (--collapse)' if collapse else '') + ('' if off_bounds else ' (--no-off-bounds)'))
    ok = True
    G2 = np.array([[1.0, 0.6], [0.4, 1.0]])
    d2, sig = np.array([0.05, 0.04]), 0.02
    E0, IH, OFF, F, D = [], [], [0], [], []
    for j, sl in enumerate((0.7, 0.3)):                 # the row's likelihood: lognormal in y, median d_j, times a constant
        y = np.linspace(d2[j] * np.exp(-3.7 * sl), d2[j] * np.exp(3.7 * sl), 1501)
        lp = -np.log(y) - 0.5 * (np.log(y / d2[j]) / sl) ** 2
        dlp = -1.0 / y - np.log(y / d2[j]) / (sl * sl * y)
        g, gp = -2.0 * (lp - lp.max()), -2.0 * dlp
        Fj = sig ** 2 * g                               # misfit / (2 sig^2) = -ln p + const
        Dj = quasiconvex_table(Fj, sig ** 2 * gp, y[1] - y[0], f'row {j}')
        E0.append(y[0] - d2[j]); IH.append(1.0 / (y[1] - y[0])); F.append(Fj); D.append(Dj); OFF.append(OFF[-1] + len(Fj))
    tab = dict(E0=np.array(E0), IH=np.array(IH), OFF=np.array(OFF, np.int64), F=np.concatenate(F), D=np.concatenate(D))
    lo10, su, a_, b_ = np.array([-4.0, -4.0]), np.array([0.1, 0.1]), 1.0, 1.0
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):
        ww = np.full(len(ui), ui[1] - ui[0])
        ww[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * ww
        gw.append(f / f.sum())
    pg = [np.exp(betaln(a_ + k, b_ + 2 - k) - betaln(a_, b_)) for k in range(3)]
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')

    def mis(m1, m2):
        m1, m2 = np.broadcast_arrays(np.asarray(m1, float), np.asarray(m2, float))
        out = np.zeros(m1.shape)
        for j in range(2):
            e = G2[j, 0] * m1 + G2[j, 1] * m2 - d2[j]
            out += hermite_eval(tab, np.full(e.shape, j), e)
        return out
    Lb = lambda m1, m2: np.exp(-mis(m1, m2) / (2 * sig ** 2))
    L1, L2_, L12 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1]), Lb(M1, M2) * gw[0][:, None] * gw[1][None, :]
    w00 = pg[0] * Lb(0.0, 0.0)
    w10, w01, w11 = pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2_), pg[2] * L12.sum()
    W_ = w00 + w10 + w01 + w11
    p_on = [(w10 + w11) / W_, (w01 + w11) / W_]
    e_u = [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
           (pg[1] * (gw[1] * u[1]) @ L2_ + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)]
    for be in backends:
        T = Target(sp.csc_matrix(G2), d2, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0]), np.array([1]), np.array([G2[:, 0] @ G2[:, 1]])))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_, collapse=collapse)
        T.off_bounds = off_bounds
        T.split = dict(L2=np.ascontiguousarray([-1.0, -2.0]), H2=np.ascontiguousarray([-1.0, -2.0]),
                       tab={k_: np.ascontiguousarray(v_) for k_, v_ in tab.items()})
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.03, 0.02], (K, 1)), sw, sw // 10, 1, be, 1, 31, monitor=(0, 1), verbose=False,
                    keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            pr = on.mean()
            if on.std() == 0:
                zp = 0.0 if on.size * min(p_on[i], 1 - p_on[i]) < 5 else np.inf
            else:
                zp = (pr - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / ess(on))
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * pr)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} rate {i}: P(on) {pr:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 m | on] {uu.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
        m_ = Mo.reshape(-1, 2)                              # chi2 recorded as the tables count it
        chi_ = mis(m_[:, 0], m_[:, 1]) / sig ** 2
        ok &= bool(np.allclose(np.sort(chi_), np.sort(tr['chi2'][:, sw // 10:].ravel()), rtol=1e-6, atol=1e-9))
    _sp_ptrs(Target(sp.csc_matrix(G2), d2, sig, 0.0, 10.0))   # the kernel's tables cleared
    print('SELFTEST (exact slip misfit' + (', --collapse' if collapse else '') + ('' if off_bounds else ', --no-off-bounds') + ')',
          'PASSED' if ok else 'FAILED')
    return ok


def selftest_sum_pdf():
    """slip_sum_pdf_tables: with every parent's density a Gaussian (a wide one, convolved by FFT, and a narrow one, taken
    as a Gaussian), each cross-section's table must be its Gaussian term, e^2, exactly as the summed slip rates have it
    without --slip-misfit exact, up to the tables' grid; with one lopsided parent, its mean against a direct sum."""
    from cru_common import slip_sum_pdf_tables, hermite_eval
    print('summed slip rates\' densities (slip_sum_pdf_tables): Gaussian parents against the Gaussian term')
    ok = True
    sd = np.array([1.0, 0.002, 0.5])                    # the parents' sds (scaled, mm/yr) and modes
    mu = np.array([10.0, 0.3, 4.0])
    f = 0.8
    g_list, y0, h, off = [], [], [], [0]
    for m_, s_ in zip(mu / f, sd / f):                  # the densities unscaled (the tables' units), from -8 to +8 sd
        y = np.linspace(m_ - 8 * s_, m_ + 8 * s_, 4001)
        g_list.append(((y - m_) / s_) ** 2)
        y0.append(y[0]); h.append(y[1] - y[0]); off.append(off[-1] + len(y))
    P = dict(slip_pdf_key=np.array([0, 0, 1, 2], np.int32), slip_pdf_f=f, slip_pdf_off=np.array(off),
             slip_pdf_y0=np.array(y0), slip_pdf_h=np.array(h), slip_pdf_g=np.concatenate(g_list))
    proj = np.array([1.0, 0.7, -0.5, 0.9])               # 4 subsections: two of parent 0, one each of parents 1 and 2
    W = np.array([[1.0, 0.5, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0]])
    par = np.array([0, 0, 1, 2])
    tgt = mu[par]
    Wc = W * proj[None, :]
    A = np.zeros((2, 3))
    np.add.at(A.T, par, Wc.T)
    sig_k = np.sqrt((A ** 2) @ (sd ** 2))
    info = dict(W=W, proj=proj, parent=par, target=Wc @ tgt, sigma=sig_k)
    c_eff = np.array([0.7, 1.3])
    tb = slip_sum_pdf_tables(P, info, np.arange(4), c_eff)
    for k in range(2):
        e = c_eff[k] * sig_k[k] * np.linspace(-3, 3, 61)
        got = hermite_eval(tb, np.full(len(e), k), e)
        got -= hermite_eval(tb, np.array([k]), np.array([0.0]))[0]
        err = float(np.max(np.abs(got - e ** 2)))
        ok &= err < 1e-3 * 9 * (c_eff[k] * sig_k[k]) ** 2
        print(f'   cross-section {k}: the table against e^2 over +-3 sd: largest difference {err:.2e} (of {9 * (c_eff[k] * sig_k[k]) ** 2:.2f}); '
              f'expected term {tb["E"][k]:.4f} (the Gaussian\'s {c_eff[k] ** 2 * sig_k[k] ** 2:.4f})')
        ok &= abs(tb['E'][k] / (c_eff[k] * sig_k[k]) ** 2 - 1.0) < 2e-3
    print('SELFTEST (summed slip rates\' densities)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_crossswap(backends=('c', 'numpy'), collapse=False, swap_frac=1.0):
    """--swap-cos-cross: swaps between two spike-and-slab rates in different bins (their own fractions on, Beta(1, 1), and
    their own slabs: floors 1e-4 and 10^-2.5, soft tops 0.05 and 0.3 decades wide), near-duplicate columns (cosine 0.998)
    so that swaps are often accepted, against a grid integral: P(on) and E[log10 m | on] of each rate. Without the prior's
    ratio in the acceptance the swaps would favour neither slab, and the answers would be off."""
    print('cross-bin swaps (--swap-cos-cross): spike-and-slab in two bins with swaps vs a grid integral'
          + (' (--collapse)' if collapse else '') + (f' (--swap-fraction {swap_frac:g})' if swap_frac < 1 else ''))
    ok = True
    G2 = np.array([[1.0, 0.95], [0.3, 0.35]])
    d2, sig = np.array([0.05, 0.016]), 0.01
    lo10, su = np.array([-4.0, -2.5]), np.array([0.05, 0.3])
    u = [np.linspace(l, min(8 * sv, 1.0), 4001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):
        ww = np.full(len(ui), ui[1] - ui[0])
        ww[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * ww
        gw.append(f / f.sum())
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')

    def Lb(m1, m2):
        m1, m2 = np.broadcast_arrays(m1, m2)
        out = np.zeros(m1.shape)
        for j in range(2):
            out += (G2[j, 0] * m1 + G2[j, 1] * m2 - d2[j]) ** 2 / (2 * sig ** 2)
        return np.exp(-out)
    L1, L2_, L12 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1]), Lb(M1, M2) * gw[0][:, None] * gw[1][None, :]
    w00, w10, w01, w11 = 0.25 * Lb(0.0, 0.0), 0.25 * (gw[0] @ L1), 0.25 * (gw[1] @ L2_), 0.25 * L12.sum()   # each bin's
    W_ = w00 + w10 + w01 + w11                                                                             # P(on) 1/2
    p_on = [(w10 + w11) / W_, (w01 + w11) / W_]
    e_u = [(0.25 * (gw[0] * u[0]) @ L1 + 0.25 * (L12.sum(1) @ u[0])) / (w10 + w11),
           (0.25 * (gw[1] * u[1]) @ L2_ + 0.25 * (L12.sum(0) @ u[1])) / (w01 + w11)]
    for be in backends:
        T = Target(sp.csc_matrix(G2), d2, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0]), np.array([1]), np.array([G2[:, 0] @ G2[:, 1]])))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.array([0, 1], np.int64), B=2, nb=np.array([1.0, 1.0]), a=1.0, b=1.0, collapse=collapse)
        T.swap_frac = swap_frac
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.03, 0.02], (K, 1)), sw, sw // 10, 1, be, 1, 41, monitor=(0, 1), verbose=False,
                    keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            pr = on.mean()
            zp = (pr - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / max(ess(on), 1.0)) if on.std() > 0 else np.inf
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * pr)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} rate {i}: P(on) {pr:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 m | on] {uu.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
        ok &= bool(tr['swap_rate'].mean() > 0.05 * swap_frac)   # the swaps must take part for the test to mean anything
    print('SELFTEST (cross-bin swaps' + (', --collapse' if collapse else '') + ')', 'PASSED' if ok else 'FAILED')
    return bool(ok)


def selftest_collapse(backends=('c', 'numpy')):
    """--collapse: four rates in two bins of two, nearly no data (the prior alone), a U-shaped Beta(0.2, 0.2) on each bin's
    fraction on (all on or all off favoured: P(2 on, 0 on) 0.18 against P(1, 1) 0.02), and swaps across the bins, which move a rupture on from one bin to the other: the number on in each bin
    against the beta-binomial's exact distribution, P(c) = C(2, c) B(a + c, b + 2 - c) / B(a, b), with the fractions on
    integrated out and drawn; and the recorded fractions' mean against its exact value. A swap that left out the counts'
    ratio would spread the ruptures on evenly between the bins (checked: with it left out, two on in one bin given two
    on in all has probability 0.03, not 0.95)."""
    from scipy.special import betaln, comb
    print('collapsed fractions on (--collapse): counts per bin under the prior alone, with swaps across bins, vs the '
          'beta-binomial')
    ok = True
    a_, b_ = 0.2, 0.2
    pc = np.array([comb(2, c) * np.exp(betaln(a_ + c, b_ + 2 - c) - betaln(a_, b_)) for c in range(3)])
    G = np.array([[1e-6, 2e-6, 1.5e-6, 1e-6]])
    pi_, pj_ = np.array([0, 1]), np.array([2, 2])   # not all four pairs: with them, a pass of swaps that were always
                                                    # accepted would carry two on in one bin to two on in the other
    for be in backends:
        for col in (True, False):
            T = Target(sp.csc_matrix(G), np.zeros(1), 1.0, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                       pairs=(pi_, pj_, np.array([G[:, i] @ G[:, j] for i, j in zip(pi_, pj_)])))
            T.zlo_vec = np.full(4, -3.0 * LN10)
            T.ku_vec = np.full(4, 1.0 / (2.0 * (0.1 * LN10) ** 2))
            T.pkind = 3
            T.ss = dict(bins=np.array([0, 0, 1, 1], np.int64), B=2, nb=np.array([2.0, 2.0]), a=a_, b=b_, collapse=col)
            K, sw = (8, 20000) if be == 'c' else (4, 1500)
            tr = sample(T, np.tile([0.03, 0.0, 0.0, 0.0], (K, 1)), sw, sw // 10, 1, be, 1, 53, monitor=(0, 1, 2, 3),
                        verbose=False, keep_samples=False, swaps=True)
            Mo = tr['mon'][:, sw // 10:, :] > 0
            line = []
            for b in range(2):
                cb = Mo[:, :, 2 * b:2 * b + 2].sum(2)
                for c in range(3):
                    ind = (cb == c).astype(float)
                    z = (ind.mean() - pc[c]) / np.sqrt(pc[c] * (1 - pc[c]) / max(ess(ind), 1.0))
                    ok &= abs(z) < 4
                    line.append(f'{ind.mean():.3f} (z {z:+.1f})')
            pim = tr['hyper_pi'][:, sw // 10:, :].mean()
            ok &= abs(pim - a_ / (a_ + b_)) < 0.02
            ok &= bool(tr['swap_rate'].mean() > 0.02)
            print(f'   {be:5s} {"collapsed" if col else "drawn    "}: P(0, 1, 2 on) bin 0 {", ".join(line[:3])}; bin 1 '
                  f'{", ".join(line[3:])} (exact {pc[0]:.3f}, {pc[1]:.3f}, {pc[2]:.3f}); fraction on {pim:.3f} '
                  f'(exact {a_ / (a_ + b_):.3f}); swap acceptance {tr["swap_rate"].mean():.2f}')
    print('SELFTEST (collapsed fractions on)', 'PASSED' if ok else 'FAILED')
    return bool(ok)


def _grid3_exact(loglik, lo10, sv, bins, nb, ng=301):
    """Three spike-and-slab rates (Beta(1, 1) on each bin's fraction on, integrated out; each rate's slab log-uniform
    from 10^lo10 with a soft top at 1, sd sv decades) under the log likelihood loglik(u0, u1, u2) (arrays; 0 for off), by
    a grid in log10 u: each rate's P(on), E[log10 u | on], and each on/off pattern's probability."""
    from scipy.special import betaln
    g = []
    for l, s in zip(lo10, sv):
        u = np.linspace(l, min(8 * s, 1.0), ng)
        ww = np.full(ng, u[1] - u[0])
        ww[[0, -1]] *= 0.5
        f = np.where(u > 0, np.exp(-u ** 2 / (2 * s ** 2)), 1.0) * ww
        g.append((u, f / f.sum()))
    pon, eu, pat = np.zeros(3), np.zeros(3), {}
    for on in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]:
        cnt = np.bincount(np.asarray(bins)[np.array(on, bool)], minlength=len(nb)).astype(float)
        lp = sum(betaln(1 + cnt[j], 1 + nb[j] - cnt[j]) - betaln(1, 1) for j in range(len(nb)))
        ax = [(g[i][0], g[i][1]) if on[i] else (np.array([-np.inf]), np.ones(1)) for i in range(3)]
        m, mu = 0.0, np.zeros(3)
        for a0 in range(len(ax[0][0])):          # in slices of the first rate's grid
            U1, U2 = np.meshgrid(10.0 ** ax[1][0], 10.0 ** ax[2][0], indexing='ij')
            W = ax[0][1][a0] * ax[1][1][:, None] * ax[2][1][None, :] * np.exp(lp)
            L = np.exp(loglik(np.full(U1.shape, 10.0 ** ax[0][0][a0]), U1, U2)) * W
            s_ = L.sum()
            m += s_
            mu += [s_ * ax[0][0][a0] if on[0] else 0.0, (L.sum(1) @ ax[1][0]) if on[1] else 0.0,
                   (L.sum(0) @ ax[2][0]) if on[2] else 0.0]
        pat[on] = m
        for i in range(3):
            if on[i]:
                pon[i] += m
                eu[i] += mu[i]
    tot = sum(pat.values())
    return pon / tot, eu / np.maximum(pon, 1e-300), {k: v / tot for k, v in pat.items()}


def selftest_splitmerge(variant='gauss', nprop=4, seed=61, verbose=True, sw=40000):
    """--split-merge: rupture A and the two ruptures B and C that together are A (B's subsection one of A's two, C's the
    other; average slips 2, 1.5 and 1.2), spike-and-slab (A alone in its bin, B and C in another; Beta(1, 1) on each bin's
    fraction on; slabs from 10^-2.5 to a soft top at 1), with the two slip-rate rows (unchanged when A's rate t goes to
    alpha t on B and beta t on C), a paleo-like row (A and B seen, 1 and 0.7 of their events) and each bin's total, split
    and merge moves after each sweep (nprop a sweep), against a grid integral: each rate's P(on), E[log10 u | on], and the
    two arrangements' probabilities, A alone and B with C. variant 'curve': the slip-rate rows split normals (lopsided
    weights) and the paleo-like row scored by a curve (the paleo tables, scaled out of the Gaussian) with positivity on
    it, the machinery main() uses at 630k."""
    print(f'split/merge moves (--split-merge, {variant}): A and B + C, spike-and-slab, vs a grid integral')
    DA, DB, DC, t0 = 2.0, 1.5, 1.2, 0.3
    al, be = DA / DB, DA / DC
    pB, sd_s, sd_p, sd_b, tA, tBC, dp = 0.7, 0.15, 0.2, 0.4, 0.1, 0.9, 0.28
    L2s, H2s = (0.5, 2.0) if variant == 'curve' else (1.0, 1.0)     # the slip-rate rows' weights below and above
    G = np.array([[DA, DB, 0.0], [DA, 0.0, DC], [1.0, pB, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    d = np.array([DA * t0, DA * t0, dp, tA, tBC])
    sd = np.array([sd_s, sd_s, sd_p, sd_b, sd_b])
    lo10, sv = np.full(3, -2.5), np.full(3, 0.1)
    if variant == 'curve':
        from paleo_misfit import PALEO_EPS, curve_tables, fit_site, site_misfit
        best = dp
        th = fit_site(best, np.array([0.1, 0.17, 0.45, 0.8]), 'log')[0]
        tab = curve_tables([th], [best], 'log')

    def loglik(u0, u1, u2):
        U = np.stack([u0, u1, u2])
        out = np.zeros(u0.shape)
        for j in range(5):
            if variant == 'curve' and j == 2:
                tot = G[2, 0] * u0 + G[2, 1] * u1
                with np.errstate(divide='ignore', invalid='ignore'):
                    out -= 0.5 * np.where(tot > 0, site_misfit(th, best, np.log(np.maximum(tot, 1e-300)), 'log'), np.inf)
                continue
            e = (np.tensordot(G[j], U, 1) - d[j]) / sd[j]
            w = np.where(e < 0, L2s, H2s) if j < 2 else 1.0
            out -= 0.5 * w * e * e
        return out
    p_on, e_u, pat = _grid3_exact(loglik, lo10, sv, [1, 0, 0], [2.0, 1.0])
    Gs, ds = G / sd[:, None], d / sd
    if variant == 'curve':
        eps = PALEO_EPS
        Gs[2], ds[2] = G[2] * eps / sd_p, eps * best / sd_p
    T = Target(sp.csc_matrix(Gs), ds, 1.0, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0))
    T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
    T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (sv * LN10) ** 2))
    T.pkind = 3
    T.ss = dict(bins=np.array([1, 0, 0], np.int64), B=2, nb=np.array([2.0, 1.0]), a=1.0, b=1.0)
    if variant == 'curve':
        T.split = dict(L2=np.ascontiguousarray([L2s, L2s, 1, 1, 1], float), H2=np.ascontiguousarray([H2s, H2s, 1, 1, 1], float))
        T.set_groups(np.array([2]), np.zeros(0, np.int64), np.zeros(0))
        pm = dict(I=np.array([-1, -1, 0, -1, -1], np.int32), V0=np.ascontiguousarray(tab['U0'] - np.log(sd_p / eps)),
                  F=tab['F'], D=tab['DF'], K=tab['KAPPA'], H=tab['H'], NG=tab['NG'], rows=np.array([], np.int64), eps=1.0,
                  d_data=d, mode='gaussian', b=0.0)
        T.lw = dict(W=np.zeros(5), LD=np.zeros(5), DY=np.ascontiguousarray(ds), rows=np.array([], np.int64), tau=None, PM=pm)
    set_split_merge(T, [0], [1], [2], [np.log(al)], [np.log(be)], nprop)
    K = 8
    X0 = np.tile([0.3, 0.0, 0.0], (K, 1))
    X0[K // 2:] = [0.0, al * t0, be * t0]             # half the chains from each arrangement
    tr = sample(T, X0, sw, sw // 10, 1, 'c', 1, seed, monitor=(0, 1, 2), verbose=False, keep_samples=False,
                split_merge=True)
    Mo = tr['mon'][:, sw // 10:, :]
    ok = True
    for i, nm in enumerate('ABC'):
        on = (Mo[:, :, i] > 0).astype(float)
        pr = on.mean()
        zp = (pr - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / max(ess(on), 1.0)) if on.std() > 0 else np.inf
        uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
        zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * pr)))
        ok &= abs(zp) < 4 and abs(zu) < 4
        if verbose:
            print(f'   rate {nm}: P(on) {pr:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 u | on] {uu.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f})')
    for key, nm in (((1, 0, 0), 'A alone'), ((0, 1, 1), 'B with C')):
        ind = np.all((Mo > 0) == np.array(key, bool), axis=2).astype(float)
        z = (ind.mean() - pat[key]) / np.sqrt(max(pat[key] * (1 - pat[key]), 1e-12) / max(ess(ind), 1.0))
        ok &= abs(z) < 4
        if verbose:
            print(f'   {nm}: {ind.mean():.4f} (exact {pat[key]:.4f}, z {z:+.2f})')
    st = tr['split_merge'].sum(0)
    tries = nprop * sw * K
    if verbose:
        print(f'   splits: {st[0]:,} tried, {st[1]:,} accepted; merges: {st[2]:,} tried, {st[3]:,} accepted '
              f'(of {tries:,} proposals)')
    ok &= bool(st[1] > 0.01 * tries / 2 and st[3] > 0.01 * tries / 2)   # the moves must take part
    print(f'SELFTEST (split/merge moves, {variant})', 'PASSED' if ok else 'FAILED')
    return bool(ok)


def selftest_splitmerge_counts(nprop=20, seed=67, verbose=True, sw=120000):
    """--split-merge, the moves' choices: six ruptures on three subsections (1 2 3; 1; 2 3; 1 2; 3; 2) and the four ways
    they split ((1 2 3) into (1) and (2 3) or into (1 2) and (3); (1 2) into (1) and (2); (2 3) into (2) and (3)), so that
    the number of triples to choose from, and the number of ruptures to start from (those that are A or B in one, on),
    change with the state; nearly no data (the prior alone), the ruptures in two bins (Beta(1, 1) on each fraction on) with
    slabs of their own depths: the patterns on and off against the beta-binomials' exact probabilities, and each rupture's
    P(on) and E[log10 u | on] against its slab's. A move that miscounted its choices would favour some patterns."""
    from scipy.special import betaln
    print('split/merge moves (--split-merge): the choices\' counts, six ruptures and four triples, the prior alone')
    subs = [(1, 2, 3), (1,), (2, 3), (1, 2), (3,), (2,)]
    Dv = np.array([2.0, 1.0, 1.6, 1.4, 0.9, 1.1])
    trip = [(0, 1, 2), (0, 3, 4), (3, 1, 5), (2, 5, 4)]
    G = np.zeros((3, 6))
    for i, s in enumerate(subs):
        G[np.array(s) - 1, i] = 1e-4 * Dv[i]         # slip-rate rows, nearly no weight
    bins = np.array([0, 1, 1, 0, 1, 1])
    nb = np.bincount(bins).astype(float)
    lo10 = np.array([-2.0, -3.0, -2.5, -2.0, -3.0, -2.5])
    sv = np.full(6, 0.1)
    pats = np.array([[(m >> i) & 1 for i in range(6)] for m in range(64)], bool)
    lpp = np.array([sum(betaln(1 + p[bins == b].sum(), 1 + nb[b] - p[bins == b].sum()) - betaln(1, 1) for b in range(2))
                    for p in pats])
    ppat = np.exp(lpp - lpp.max())
    ppat /= ppat.sum()
    e_u = []
    for l, s in zip(lo10, sv):                       # each slab's mean log10 u (as _grid3_exact's slabs)
        u = np.linspace(l, min(8 * s, 1.0), 4001)
        ww = np.full(len(u), u[1] - u[0])
        ww[[0, -1]] *= 0.5
        f = np.where(u > 0, np.exp(-u ** 2 / (2 * s ** 2)), 1.0) * ww
        e_u.append((f @ u) / f.sum())
    T = Target(sp.csc_matrix(G), np.zeros(3), 1.0, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0))
    T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
    T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (sv * LN10) ** 2))
    T.pkind = 3
    T.ss = dict(bins=bins.astype(np.int64), B=2, nb=nb, a=1.0, b=1.0)
    ta, tb, tc = (np.array([t[q] for t in trip]) for q in range(3))
    set_split_merge(T, ta, tb, tc, np.log(Dv[ta] / Dv[tb]), np.log(Dv[ta] / Dv[tc]), nprop)
    K = 8
    X0 = np.tile([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], (K, 1))
    tr = sample(T, X0, sw, sw // 10, 1, 'c', 1, seed, monitor=tuple(range(6)), verbose=False, keep_samples=False,
                split_merge=True)
    Mo = tr['mon'][:, sw // 10:, :]
    on = Mo > 0
    code = (on * (1 << np.arange(6))).sum(-1)
    ok, worst = True, (0.0, '')
    for m in range(64):                               # each pattern's frequency
        ind = (code == m).astype(float)
        z = (ind.mean() - ppat[m]) / np.sqrt(ppat[m] * (1 - ppat[m]) / max(ess(ind), 1.0))
        ok &= abs(z) < 5.0                            # 64 patterns, each rare and its effective draws estimated: 5 sd
                                                      # (the correct moves' largest |z| 2.5-3.5 over three seeds; a reverse
                                                      # merge counted in the current state, not the proposed one: 7.2)
        if abs(z) > abs(worst[0]):
            worst = (z, f'pattern {"".join("1" if (m >> i) & 1 else "0" for i in range(6))}: {ind.mean():.4f} '
                        f'(exact {ppat[m]:.4f})')
    line = []
    for i in range(6):
        pon = ppat[pats[:, i]].sum()
        oi = on[:, :, i].astype(float)
        zp = (oi.mean() - pon) / np.sqrt(pon * (1 - pon) / max(ess(oi), 1.0))
        uu = np.log10(Mo[:, :, i][on[:, :, i]])
        zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(oi) * oi.mean())))
        ok &= abs(zp) < 4 and abs(zu) < 4
        line.append(f'{i}: {oi.mean():.3f} (z {zp:+.1f}), {uu.mean():.3f} (z {zu:+.1f})')
    st = tr['split_merge'].sum(0)
    if verbose:
        print('   P(on), E[log10 u | on] (z against exact): ' + '; '.join(line))
        print(f'   the 64 patterns: largest |z| {abs(worst[0]):.2f}, {worst[1]}')
        print(f'   splits: {st[0]:,} tried, {st[1]:,} accepted; merges: {st[2]:,} tried, {st[3]:,} accepted '
              f'(of {nprop * sw * K:,} proposals)')
    ok &= bool(st[1] > 0.01 * nprop * sw * K / 2 and st[3] > 0.01 * nprop * sw * K / 2)
    print('SELFTEST (split/merge moves, the choices\' counts)', 'PASSED' if ok else 'FAILED')
    return bool(ok)


def selftest_slipgeo(backends=('c', 'numpy'), b=0.5):
    """--slip-misfit split+log10, built as main() builds it (the rows and targets scaled by 1/sqrt 2, the log10 half through
    the soft band's arrays with sided weights, W = 1/(4 (s ln10)^2) above the target and WL below, s = log10(1 + sd / d) each
    side's, the swap pair's inner product halved), on two spike-and-slab rates and two split-normal rows (one symmetric, one
    lopsided, its log10 sds 0.48 below and 1.14 above), with swaps, against a grid integral of exp(-misfit / 2), misfit =
    sum over rows of w e^2 / (2 sigma^2) + (log10(y / d) / s)^2 / 2: P(on) and E[log10 m | on] of each rate; and chi2 as
    fitted, as recorded, against its direct computation."""
    from scipy.special import betaln
    from paleo_misfit import PALEO_EPS
    mode_ = 'log10' if b >= 1 else 'split+log10'
    print(f'slip misfit {mode_}: spike-and-slab with swaps vs a grid integral')
    ok = True
    G2 = np.array([[1.0, 0.6], [0.4, 1.0]])
    d2, sig, L2v, H2v = np.array([0.05, 0.004]), 0.02, np.array([1.0, 6.0]), np.array([1.0, 0.15])
    lo10, su, a_, b_ = np.array([-4.0, -4.0]), np.array([0.1, 0.1]), 1.0, 1.0
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):
        ww = np.full(len(ui), ui[1] - ui[0])
        ww[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * ww
        gw.append(f / f.sum())
    pg = [np.exp(betaln(a_ + k, b_ + 2 - k) - betaln(a_, b_)) for k in range(3)]
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')
    s_lo, s_hi = np.log10(1.0 + sig / np.sqrt(L2v) / d2), np.log10(1.0 + sig / np.sqrt(H2v) / d2)   # each side's sd, in log10

    def mis(m1, m2):
        m1, m2 = np.broadcast_arrays(m1, m2)
        out = np.zeros(m1.shape)
        for j in range(2):
            y = G2[j, 0] * m1 + G2[j, 1] * m2
            e = y - d2[j]
            with np.errstate(divide='ignore'):
                out += (1.0 - b) * np.where(e < 0, L2v[j], H2v[j]) * e * e / sig ** 2 + np.where(
                    y > 0, b * (np.log10(np.maximum(y, 1e-300) / d2[j]) / np.where(e < 0, s_lo[j], s_hi[j])) ** 2, np.inf)
        return out
    L1, L2_ = np.exp(-mis(10 ** u[0], 0.0) / 2), np.exp(-mis(0.0, 10 ** u[1]) / 2)
    L12 = np.exp(-mis(M1, M2) / 2) * gw[0][:, None] * gw[1][None, :]
    w10, w01, w11 = pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2_), pg[2] * L12.sum()   # both off: infinite misfit
    W_ = w10 + w01 + w11
    p_on = [(w10 + w11) / W_, (w01 + w11) / W_]
    e_u = [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
           (pg[1] * (gw[1] * u[1]) @ L2_ + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)]
    h = PALEO_EPS if b >= 1 else 1.0 / np.sqrt(2.0)
    for be in backends:
        T = Target(sp.csc_matrix(G2 * h), d2 * h, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0]), np.array([1]), np.array([h * h * G2[:, 0] @ G2[:, 1]])))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        T.split = dict(L2=np.ascontiguousarray(L2v), H2=np.ascontiguousarray(H2v))
        T.n_data = 2
        T.set_groups(np.array([0, 1]), np.zeros(0, np.int64), np.zeros(0))
        T.lw = dict(W=np.ascontiguousarray(b / (2.0 * (s_hi * LN10) ** 2)), WL=np.ascontiguousarray(b / (2.0 * (s_lo * LN10) ** 2)),
                    LD=np.log(d2 * h), DY=np.ascontiguousarray(d2 * h), rows=np.array([], np.int64), tau=None,
                    SM=dict(rows=np.array([0, 1]), s_lo=s_lo, s_hi=s_hi, wk=np.ones(2), b=b, eps=h, d_data=d2, E=0.0, V=0.0))
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.03, 0.02], (K, 1)), sw, sw // 10, 1, be, 1, 37, monitor=(0, 1), verbose=False,
                    keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            pr = on.mean()
            if on.std() == 0:
                zp = 0.0 if on.size * min(p_on[i], 1 - p_on[i]) < 5 else np.inf
            else:
                zp = (pr - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / ess(on))
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * pr)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} rate {i}: P(on) {pr:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 m | on] {uu.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
        m_ = Mo.reshape(-1, 2)                                # chi2 as fitted, recorded, against its direct computation
        ok_m = bool(np.allclose(np.sort(mis(m_[:, 0], m_[:, 1])), np.sort(tr['misfit'][:, sw // 10:].ravel()), rtol=1e-6, atol=1e-9))
        ok &= ok_m
        print(f'   {be:5s} chi2 as fitted recorded as the misfit counts it: ' + ('yes' if ok_m else 'NO'))
    print(f'SELFTEST (slip misfit {mode_})', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_fold(backends=('c', 'numpy')):
    """column_passes: the likelihood's column passes without the constraint rows, and with --slip-sum's rows folded into
    the slip-rate rows (U = B^T r_ss), built as main() builds them: two spike-and-slab rates, two slip-rate rows scored by
    --slip-misfit split+log10 (scaled by 1/sqrt 2, sided log10 weights, positivity), a summed-slip row equal to B times
    them (Gaussian, in the likelihood), and a constraint row (1e-9 x m1 + m2, kept at most 1e-9 x cap, a cap that binds),
    with swaps, against a grid integral: P(on) and E[log10 m | on] of each rate; and the recorded misfit and summed-slip
    term against their direct computation. The C kernel is run with the constraint row left out of its passes, then with
    the fold as well; the NumPy backend reads every row."""
    from scipy.special import betaln
    from paleo_misfit import PALEO_EPS
    print('column passes (constraint rows skipped; --slip-sum folded): spike-and-slab with swaps vs a grid integral')
    ok, b = True, 0.5
    G2 = np.array([[1.0, 0.6], [0.4, 1.0]])
    d2, sig, L2v, H2v = np.array([0.05, 0.004]), 0.02, np.array([1.0, 6.0]), np.array([1.0, 0.15])
    lo10, su, a_, b_ = np.array([-4.0, -4.0]), np.array([0.1, 0.1]), 1.0, 1.0
    h = 1.0 / np.sqrt(2.0)
    Bv, d_ss, eps, cap = np.array([0.7, 1.3]), 0.06, 1e-9, 0.062
    A_ss = Bv @ (h * G2)                                   # the summed-slip row: B times the (scaled) slip-rate rows
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):
        ww = np.full(len(ui), ui[1] - ui[0])
        ww[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * ww
        gw.append(f / f.sum())
    pg = [np.exp(betaln(a_ + k, b_ + 2 - k) - betaln(a_, b_)) for k in range(3)]
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')
    s_lo, s_hi = np.log10(1.0 + sig / np.sqrt(L2v) / d2), np.log10(1.0 + sig / np.sqrt(H2v) / d2)

    def mis(m1, m2):                                       # the data rows' misfit as fitted (chi2's scale)
        m1, m2 = np.broadcast_arrays(m1, m2)
        out = np.zeros(m1.shape)
        for j in range(2):
            y = G2[j, 0] * m1 + G2[j, 1] * m2
            e = y - d2[j]
            with np.errstate(divide='ignore'):
                out += (1.0 - b) * np.where(e < 0, L2v[j], H2v[j]) * e * e / sig ** 2 + np.where(
                    y > 0, b * (np.log10(np.maximum(y, 1e-300) / d2[j]) / np.where(e < 0, s_lo[j], s_hi[j])) ** 2, np.inf)
        return out

    def ssum(m1, m2):                                      # the summed-slip term
        return (A_ss[0] * m1 + A_ss[1] * m2 - d_ss) ** 2 / sig ** 2

    def post(m1, m2):
        m1, m2 = np.broadcast_arrays(m1, m2)
        return np.exp(-(mis(m1, m2) + ssum(m1, m2)) / 2) * (m1 + m2 <= cap)
    L1, L2_ = post(10 ** u[0], 0.0), post(0.0, 10 ** u[1])
    L12 = post(M1, M2) * gw[0][:, None] * gw[1][None, :]
    w10, w01, w11 = pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2_), pg[2] * L12.sum()
    W_ = w10 + w01 + w11
    p_on = [(w10 + w11) / W_, (w01 + w11) / W_]
    e_u = [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
           (pg[1] * (gw[1] * u[1]) @ L2_ + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)]
    Gt = np.vstack([h * G2, A_ss[None, :], eps * np.ones((1, 2))])   # rows: slip, slip, summed slip, constraint
    dt_ = np.r_[h * d2, d_ss, 0.0]
    for be, mode in [('c', 'skip'), ('c', 'fold'), ('numpy', 'skip')] if 'numpy' in backends else [('c', 'skip'), ('c', 'fold')]:
        T = Target(sp.csc_matrix(Gt), dt_, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0]), np.array([1]), np.array([Gt[:, 0] @ Gt[:, 1]])))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        T.split = dict(L2=np.ascontiguousarray(np.r_[L2v, 1.0, 1.0]), H2=np.ascontiguousarray(np.r_[H2v, 1.0, 1.0]))
        T.n_data, T.n_bal, T.n_fit = 2, 2, 3
        T.slip_sum = dict(expect=1.0)
        T.set_groups(np.array([0, 1]), np.zeros(0, np.int64), np.zeros(0))
        T.add_limits([3], -np.inf, eps * cap, eps * 1e-9, 'a cap')
        T.lw = dict(W=np.ascontiguousarray(np.r_[b / (2.0 * (s_hi * LN10) ** 2), 0.0, 0.0]),
                    WL=np.ascontiguousarray(np.r_[b / (2.0 * (s_lo * LN10) ** 2), 0.0, 0.0]),
                    LD=np.ascontiguousarray(np.r_[np.log(d2 * h), 0.0, 0.0]), DY=np.ascontiguousarray(np.r_[d2 * h, 1.0, 1.0]),
                    rows=np.array([], np.int64), tau=None,
                    SM=dict(rows=np.array([0, 1]), s_lo=s_lo, s_hi=s_hi, wk=np.ones(2), b=b, eps=h, d_data=d2, E=0.0, V=0.0))
        note = column_passes(T, sp.csr_matrix((Bv, ([0, 0], [0, 1])), shape=(1, 4)) if mode == 'fold' else None)
        okp = T.pass_end is not None and (T.ss_fold is not None) == (mode == 'fold')
        ok &= okp
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.03, 0.02], (K, 1)), sw, sw // 10, 1, be, 1, 41, monitor=(0, 1), verbose=False,
                    keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        tag = f'{be:5s} {mode:4s}'
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            pr = on.mean()
            if on.std() == 0:
                zp = 0.0 if on.size * min(p_on[i], 1 - p_on[i]) < 5 else np.inf
            else:
                zp = (pr - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / ess(on))
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * pr)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {tag} rate {i}: P(on) {pr:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 m | on] {uu.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
        m_ = Mo.reshape(-1, 2)
        ok_m = bool(np.allclose(np.sort(mis(m_[:, 0], m_[:, 1])), np.sort(tr['misfit'][:, sw // 10:].ravel()), rtol=1e-6, atol=1e-9)
                    and np.allclose(np.sort(ssum(m_[:, 0], m_[:, 1])), np.sort(tr['slip_sum'][:, sw // 10:].ravel()),
                                    rtol=1e-6, atol=1e-9))
        ok_c = bool((m_.sum(1) <= cap * (1 + 1e-9)).all())
        ok &= ok_m and ok_c
        print(f'   {tag} misfit and summed-slip term recorded as computed: {"yes" if ok_m else "NO"}; every draw within '
              f'the cap: {"yes" if ok_c else "NO"}; passes set: {"yes" if okp else "NO"}' + (f' ({note})' if be == 'c' and note else ''))
    print('SELFTEST (column passes)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_paleo(backends=('c', 'numpy'), b=1.0):
    """The paleo misfit (--paleo-misfit years; with b = 1/2, gaussian+years) on two spike-and-slab rates: a weak Gaussian row
    and a paleo row, scaled out of the Gaussian (b = 1/2: scaled by 1/sqrt 2) and scored by b x a site's tabulated half
    misfit (a lopsided curve, paleo_misfit.tables), with swap moves between the rates, against a grid integral of likelihood
    x prior x exp(-b f): P(on) and E[log10 m | on] of each rate, and the recorded misfits. Then the per-group hard limits
    (set_groups) against set_positive and set_band, which they must reproduce."""
    from scipy.special import betaln
    from paleo_misfit import PALEO_EPS, tables
    mode_ = 'years' if b >= 1 else 'gaussian+years'
    print(f'paleo misfit (--paleo-misfit {mode_}): spike-and-slab with swaps vs a grid integral')
    ok = True
    site = dict(name='test', section=0, best=0.03, sd=0.009, q=np.array([0.012, 0.019, 0.034, 0.05]))
    tab = tables([site], 'years')
    c0, cp, sig, lo10, su, a_, b_ = np.array([1.0, 0.7]), np.array([0.5, 1.0]), 0.1, np.array([-4.0, -4.0]), np.array([0.1, 0.1]), 1.0, 1.0
    d0 = 0.04
    eps, sd = (PALEO_EPS if b >= 1 else 1.0 / np.sqrt(2.0)), site['sd']
    pm = dict(I=np.array([-1, 0], np.int32), V0=np.ascontiguousarray(tab['U0'] - np.log(sd / eps)), F=b * tab['F'],
              D=b * tab['DF'], K=b * tab['KAPPA'], H=tab['H'], NG=tab['NG'], rows=np.array([1]), eps=eps,
              d_data=np.array([d0, site['best'] / sd]), mode=mode_, b=b)
    pmr = dict(pm, V0=np.ascontiguousarray(tab['U0']), F=tab['F'], D=tab['DF'])   # the whole table in u = ln(rate), for the grid
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):
        ww = np.full(len(ui), ui[1] - ui[0])
        ww[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * ww
        gw.append(f / f.sum())
    pg = [np.exp(betaln(a_ + k, b_ + 2 - k) - betaln(a_, b_)) for k in range(3)]
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')

    def Lb(m1, m2):
        m1, m2 = np.broadcast_arrays(m1, m2)
        yg, yp = c0[0] * m1 + c0[1] * m2, cp[0] * m1 + cp[1] * m2      # the Gaussian row; the paleo row's rate
        with np.errstate(divide='ignore'):
            f = np.where(yp > 0, _pm_f_np(pmr, np.zeros(yp.shape, np.int64), np.log(np.maximum(yp, 1e-300))), np.inf)
        return np.exp(-0.5 * (yg - d0) ** 2 / sig ** 2 - b * f - 0.5 * (1.0 - b) * ((yp - site['best']) / sd) ** 2 / sig ** 2)
    L1, L2, L12 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1]), Lb(M1, M2) * gw[0][:, None] * gw[1][None, :]
    w10, w01, w11 = pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2), pg[2] * L12.sum()   # both off: impossible
    W_ = w10 + w01 + w11
    p_on = [(w10 + w11) / W_, (w01 + w11) / W_]
    e_u = [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
           (pg[1] * (gw[1] * u[1]) @ L2 + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)]
    Gs = np.vstack([c0, cp * eps / sd])                         # the paleo row in G: coefficient / sd, scaled by eps
    dv = np.array([d0, eps * site['best'] / sd])
    for be in backends:
        T = Target(sp.csc_matrix(Gs), dv, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0]), np.array([1]), np.array([Gs[:, 0] @ Gs[:, 1]])))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        T.lw = dict(W=np.zeros(2), LD=np.zeros(2), DY=np.ascontiguousarray(dv), rows=np.array([], np.int64), tau=None,
                    PM=dict(pm))
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.04, 0.0], (K, 1)), sw, sw // 10, 1, be, 1, 29, monitor=(0, 1), verbose=False,
                    keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        ok &= bool((Mo > 0).any(-1).all())                # never both off: the paleo row's prediction must stay above 0
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            pr = on.mean()
            if on.std() == 0:
                zp = 0.0 if on.size * min(p_on[i], 1 - p_on[i]) < 5 else np.inf
            else:
                zp = (pr - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / ess(on))
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * pr)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} rate {i}: P(on) {pr:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 m | on] {uu.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
        ok &= bool(np.isfinite(tr['paleo_misfit']).all() and (tr['paleo_misfit'] >= -1e-9).all())
        mm_ = Mo.reshape(-1, 2)                         # the recorded misfits against their direct computation
        yp_ = mm_ @ cp
        cur_ = 2.0 * _pm_f_np(pmr, np.zeros(len(yp_), np.int64), np.log(yp_))
        gp_ = ((yp_ - site['best']) / sd) ** 2 / sig ** 2
        ch_ = (mm_ @ c0 - d0) ** 2 / sig ** 2 + gp_
        ok_m = bool(np.allclose(cur_, tr['paleo_misfit'][:, sw // 10:].ravel(), rtol=1e-6, atol=1e-8)
                    and np.allclose(ch_, tr['chi2'][:, sw // 10:].ravel(), rtol=1e-6, atol=1e-8)
                    and np.allclose(ch_ + b * (cur_ - gp_), tr['misfit'][:, sw // 10:].ravel(), rtol=1e-6, atol=1e-8))
        ok &= ok_m
        print(f'   {be:5s} misfits recorded as computed directly: ' + ('yes' if ok_m else 'NO'))
    # the per-group limits: positivity alone and the band alone as set_positive and set_band give them, and both together
    Gq = sp.csc_matrix(np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 2.0], [0.3, 0.0, 1.0]]))
    dq = np.array([0.5, 0.8, 0.4])
    mk = lambda: Target(Gq, dq, 0.1, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0))
    A, B = mk(), mk()
    for T_ in (A, B):
        T_.pkind = 3
    A.set_positive([0, 2]); B.set_groups([0, 2], [], [])
    ok_g = np.array_equal(A.rlo, B.rlo) and np.array_equal(A.rhi, B.rhi) and np.array_equal(A.band['tol'], B.band['tol'])
    A, B = mk(), mk()
    A.set_band(4.0, [1]); B.set_groups([], [1], [4.0])
    ok_g &= np.array_equal(A.rlo, B.rlo) and np.array_equal(A.rhi, B.rhi)
    C = mk(); C.pkind = 3
    C.set_groups([0], [1, 2], [4.0, 3.0])
    ok_g &= bool(np.isfinite(C.rlo).all() and np.isinf(C.rhi[0]) and np.allclose(C.rhi[1:], np.array([4.0, 3.0]) * dq[1:] - dq[1:])
                 and np.allclose(C.rlo[1:], dq[1:] / np.array([4.0, 3.0]) - dq[1:]))
    print(f'   per-group limits: set_groups reproduces set_positive and set_band, and combines them: {"yes" if ok_g else "NO"}')
    ok &= bool(ok_g)
    print(f'SELFTEST (paleo misfit {mode_})', 'PASSED' if ok else 'FAILED')
    return bool(ok)


def selftest_groupweights(backends=('c', 'numpy'), ws=0.4, wp=3.0):
    """--group-weights equal (DATATYPES_EQUAL), as main() applies it: a Gaussian (slip) row's weight through the split
    weights (T.temper x L2, H2) and a paleo row's through its tabulated misfit (F, D, K x w), on two spike-and-slab rates
    with swap moves, against a grid integral of exp(-w_s chi2_s / 2 - w_p f) x prior: P(on) and E[log10 m | on] of each
    rate (the rows set against each other, so the weights move the exact answers by 0.07-0.24 against w = 1); the recorded chi2 and paleo misfit unweighted. Then group_weights on a three-block problem: each group's weight
    x its effective measurements equal, the total kept, each row given its group's weight."""
    from scipy.special import betaln
    from paleo_misfit import PALEO_EPS, tables
    print(f'group weights (--group-weights equal): slip x {ws:g}, paleo x {wp:g}, spike-and-slab with swaps vs a grid integral')
    ok = True
    site = dict(name='test', section=0, best=0.03, sd=0.009, q=np.array([0.012, 0.019, 0.034, 0.05]))
    tab = tables([site], 'years')
    c0, cp, sig, lo10, su, a_, b_ = np.array([1.0, 0.7]), np.array([0.5, 1.0]), 0.02, np.array([-4.0, -4.0]), np.array([0.1, 0.1]), 1.0, 1.0
    d0, eps, sd = 0.07, PALEO_EPS, site['sd']
    pm = dict(I=np.array([-1, 0], np.int32), V0=np.ascontiguousarray(tab['U0'] - np.log(sd / eps)), F=wp * tab['F'],
              D=wp * tab['DF'], K=wp * tab['KAPPA'], H=tab['H'], NG=tab['NG'], rows=np.array([1]), eps=eps,
              d_data=np.array([d0, site['best'] / sd]), mode='years', b=1.0, w=wp)
    pmr = dict(pm, V0=np.ascontiguousarray(tab['U0']), F=tab['F'], D=tab['DF'])   # unweighted, in u = ln(rate), for the grid
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):
        ww = np.full(len(ui), ui[1] - ui[0])
        ww[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * ww
        gw.append(f / f.sum())
    pg = [np.exp(betaln(a_ + k, b_ + 2 - k) - betaln(a_, b_)) for k in range(3)]
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')

    def Lb(m1, m2):
        m1, m2 = np.broadcast_arrays(m1, m2)
        yg, yp = c0[0] * m1 + c0[1] * m2, cp[0] * m1 + cp[1] * m2
        with np.errstate(divide='ignore'):
            f = np.where(yp > 0, _pm_f_np(pmr, np.zeros(yp.shape, np.int64), np.log(np.maximum(yp, 1e-300))), np.inf)
        return np.exp(-0.5 * ws * (yg - d0) ** 2 / sig ** 2 - wp * f)
    L1, L2, L12 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1]), Lb(M1, M2) * gw[0][:, None] * gw[1][None, :]
    w10, w01, w11 = pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2), pg[2] * L12.sum()   # both off: impossible
    W_ = w10 + w01 + w11
    p_on = [(w10 + w11) / W_, (w01 + w11) / W_]
    e_u = [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
           (pg[1] * (gw[1] * u[1]) @ L2 + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)]
    Gs = np.vstack([c0, cp * eps / sd])
    dv = np.array([d0, eps * site['best'] / sd])
    for be in backends:
        T = Target(sp.csc_matrix(Gs), dv, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0]), np.array([1]), np.array([Gs[:, 0] @ Gs[:, 1]])))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        T.temper = np.array([ws, wp])                     # as main(): each row's Gaussian x its group's weight
        T.split = dict(L2=np.ascontiguousarray(T.temper.copy()), H2=np.ascontiguousarray(T.temper.copy()))
        T.lw = dict(W=np.zeros(2), LD=np.zeros(2), DY=np.ascontiguousarray(dv), rows=np.array([], np.int64), tau=None,
                    PM=dict(pm))
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.04, 0.0], (K, 1)), sw, sw // 10, 1, be, 1, 37, monitor=(0, 1), verbose=False,
                    keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            pr = on.mean()
            if on.std() == 0:
                zp = 0.0 if on.size * min(p_on[i], 1 - p_on[i]) < 5 else np.inf
            else:
                zp = (pr - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / ess(on))
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * pr)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} rate {i}: P(on) {pr:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 m | on] {uu.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
        mm_ = Mo.reshape(-1, 2)                         # recorded unweighted: chi2 and the paleo misfit as without weights
        yp_ = mm_ @ cp
        cur_ = 2.0 * _pm_f_np(pmr, np.zeros(len(yp_), np.int64), np.log(yp_))
        gp_ = ((yp_ - site['best']) / sd) ** 2 / sig ** 2
        ch_ = (mm_ @ c0 - d0) ** 2 / sig ** 2 + gp_
        ok_m = bool(np.allclose(cur_, tr['paleo_misfit'][:, sw // 10:].ravel(), rtol=1e-6, atol=1e-8)
                    and np.allclose(ch_, tr['chi2'][:, sw // 10:].ravel(), rtol=1e-6, atol=1e-8))
        ok &= ok_m
        print(f'   {be:5s} misfits recorded unweighted: ' + ('yes' if ok_m else 'NO'))
    # group_weights: slip rates in sections (2 of 3 rows, 2 of 2 rows, 1 row: 3 measurements), 2 bins, 4 paleo rows
    blk = np.array([0] * 5 + [1] * 2 + [2] * 4)
    P_ = dict(G=sp.csr_matrix(np.ones((11, 3))), block=blk, block_names=['S', 'MFD', 'B2:var'], slip_sections='auto',
              section=np.array([0, 0, 1, 1, 2] + [-1] * 6), template_scale=None)
    GW_ = group_weights(P_, 'equal')
    n_ = GW_['n']
    ok_w = (np.allclose([n_['slip'], n_['mfd'], n_['paleo']], [3.0, 2.0, 4.0])
            and np.allclose([GW_['w'][g] * n_[g] for g in n_], 3.0)
            and np.allclose(GW_['row'], np.array([GW_['w'][g] for g in ['slip'] * 5 + ['mfd'] * 2 + ['paleo'] * 4])))
    print(f'   group_weights: measurements slip {n_["slip"]:g}, MFD {n_["mfd"]:g}, paleo {n_["paleo"]:g}; weights '
          + ', '.join(f'{g} {GW_["w"][g]:.3f}' for g in n_) + f'; each group then counts 3 of 9: {"yes" if ok_w else "NO"}')
    ok &= bool(ok_w)
    print('SELFTEST (group weights)', 'PASSED' if ok else 'FAILED')
    return bool(ok)


def selftest_band(backends=('c', 'numpy')):
    """The band (--band), positivity (--positive), the MFD's monotonicity (--mfd-monotonic, two rates in two bins,
    against the exact answer with m1 > m0 forbidden) and the plate rate (--plate-rate, a row of both signs kept between
    two limits): the self-tests' small problems with every prediction kept within a
    factor F of its target, against
    grid integrals restricted to the band: spike-and-slab (P(on) and E[log10 m | on], F = 1.3, where the band forbids the
    first rate's off state), a log-box prior with swap
    moves between near-duplicate columns (E[log10 m], F = 2) and the flat prior's truncated normals with a top per rupture
    (E[m], F = 1.25; the linear prior's case before its removal);
    with each, how far the exact answer without the band lies, and that no recorded draw leaves the band."""
    from scipy.special import betaln
    print('the band: spike-and-slab, positivity, MFD monotonicity (flat and with a capped rise), the plate rate, a log-box '
          'prior with swaps and truncated normals with a top per rupture vs grid integrals within their limits')
    ok = True
    inside = lambda Y, d, F: np.all((Y >= d / F) & (Y <= d * F), axis=-1)

    def z(x, e):
        return (x.mean() - e) / (x.std() / np.sqrt(max(1.0, ess(x))))
    # spike-and-slab: the selftest_spikeslab problem
    Gs, d, sig, lo10, su, a_, b_, F = np.array([[1.0, 0.7], [0.5, 1.0]]), np.array([0.6, 0.3]), 0.1, \
        np.array([-3.0, -5.0]), np.array([0.05, 0.15]), 1.0, 1.0, 1.3
    u = [np.linspace(l, min(8 * sv, 1.0), 2001) for l, sv in zip(lo10, su)]
    gw = []
    for ui, sv in zip(u, su):
        w = np.full(len(ui), ui[1] - ui[0])
        w[[0, -1]] *= 0.5
        f = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * w
        gw.append(f / f.sum())
    pg = [np.exp(betaln(a_ + k, b_ + 2 - k) - betaln(a_, b_)) for k in range(3)]
    M1, M2 = np.meshgrid(10 ** u[0], 10 ** u[1], indexing='ij')

    def exact(band):
        def Lb(m1, m2):
            Y = np.stack(np.broadcast_arrays(m1 + 0.7 * m2, 0.5 * m1 + m2), -1)
            L = np.exp(-0.5 * np.sum((Y - d) ** 2, -1) / sig ** 2)
            return L * inside(Y, d, F) if band else L
        L1, L2, L12 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1]), Lb(M1, M2) * gw[0][:, None] * gw[1][None, :]
        w00, w10, w01, w11 = pg[0] * Lb(0.0, 0.0), pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2), pg[2] * L12.sum()
        W = w00 + w10 + w01 + w11
        return ([(w10 + w11) / W, (w01 + w11) / W],
                [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
                 (pg[1] * (gw[1] * u[1]) @ L2 + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)])
    (p_on, e_u), (p_free, e_free) = exact(True), exact(False)
    for be in backends:
        T = Target(sp.csc_matrix(Gs), d, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        T.set_band(F, [0, 1])
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.4, 0.1], (K, 1)), sw, sw // 10, 1, be, 1, 7, monitor=(0, 1), verbose=False,
                    keep_samples=False)                  # a start inside the band
        Mo = tr['mon'][:, sw // 10:, :]
        ok &= int(tr['band_outside'].max()) == 0 and bool(inside(Mo @ Gs.T, d, F).all())
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            p, e_on = on.mean(), ess(on)
            sd_p = np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / e_on)
            zp, zf = (p - p_on[i]) / sd_p, (p - p_free[i]) / sd_p
            if p_on[i] > 1 - 1e-12:                  # the band forbids this rate's off state: never off
                zp = 0.0 if p == 1.0 else np.inf
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, e_on * p)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} spike-and-slab rate {i}: P(on) {p:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}; without the band '
                  f'{p_free[i]:.4f}, z {zf:+.1f}); E[log10 m | on] {uu.mean():.4f} (exact {e_u[i]:.4f}, z {zu:+.2f}; without '
                  f'{e_free[i]:.4f})')
    # positivity (--positive): the same two rates with small targets, so that both off is likely without it; positivity
    # forbids exactly that state, as every state with a rate on predicts both rows above zero
    dP = np.array([0.04, 0.03])

    def exact_pos(pos):
        def Lb(m1, m2):
            Y = np.stack(np.broadcast_arrays(m1 + 0.7 * m2, 0.5 * m1 + m2), -1)
            return np.exp(-0.5 * np.sum((Y - dP) ** 2, -1) / sig ** 2)
        L1, L2, L12 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1]), Lb(M1, M2) * gw[0][:, None] * gw[1][None, :]
        w00 = 0.0 if pos else pg[0] * Lb(0.0, 0.0)
        w10, w01, w11 = pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2), pg[2] * L12.sum()
        W = w00 + w10 + w01 + w11
        return [(w10 + w11) / W, (w01 + w11) / W], w00 / W
    (pp_on, _), (pp_free, p00_free) = exact_pos(True), exact_pos(False)
    for be in backends:
        T = Target(sp.csc_matrix(Gs), dP, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        T.set_positive([0, 1])
        U0, _ = positive_start(T, np.zeros(2))           # from all off: one rate switched on
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile(U0, (K, 1)), sw, sw // 10, 1, be, 1, 11, monitor=(0, 1), verbose=False, keep_samples=False)
        Mo = tr['mon'][:, sw // 10:, :]
        ok &= int(tr['band_outside'].max()) == 0 and bool((Mo > 0).any(-1).all())
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            p, e_on = on.mean(), ess(on)
            sd_p = np.sqrt(max(pp_on[i] * (1 - pp_on[i]), 1e-12) / e_on)
            zp, zf = (p - pp_on[i]) / sd_p, (p - pp_free[i]) / sd_p
            ok &= abs(zp) < 4
            print(f'   {be:5s} positivity, rate {i}: P(on) {p:.4f} (exact {pp_on[i]:.4f}, z {zp:+.2f}; without it '
                  f'{pp_free[i]:.4f}, z {zf:+.1f}, both off {p00_free:.3f})')
    # the MFD's monotonicity (--mfd-monotonic): the same two rates, each now alone in a bin, the second bin above the
    # first in magnitude, with targets that favour the second rate over the first: a constraint row eps (m0 - m1) >= 0
    # outside the likelihood forbids m1 > m0, so the state (off, on) and part of (on, on)
    dM, eps_m = np.array([0.3, 0.5]), 1e-9

    def exact_mono(mono):
        def Lb(m1, m2):
            Y = np.stack(np.broadcast_arrays(m1 + 0.7 * m2, 0.5 * m1 + m2), -1)
            return np.exp(-0.5 * np.sum((Y - dM) ** 2, -1) / sig ** 2)
        L1, L2 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1])
        L12 = Lb(M1, M2) * (M2 <= M1 if mono else 1.0) * gw[0][:, None] * gw[1][None, :]
        w00, w10 = Lb(0.0, 0.0), gw[0] @ L1          # one rate per bin, Beta(1, 1): each state's prior 1/4
        w01, w11 = (0.0 if mono else gw[1] @ L2), L12.sum()
        W = w00 + w10 + w01 + w11
        e1 = ((gw[0] * u[0]) @ L1 + L12.sum(1) @ u[0]) / (w10 + w11)
        return [(w10 + w11) / W, (w01 + w11) / W], e1
    (pm_on, em1), (pm_free, em1_free) = exact_mono(True), exact_mono(False)
    for be in backends:
        Gm = sp.csc_matrix(np.vstack([Gs, [eps_m, -eps_m]]))
        T = Target(Gm, np.r_[dM, 0.0], sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0))
        T.n_data = T.n_fit = 2                           # the constraint row is outside the likelihood
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.array([0, 1], np.int64), B=2, nb=np.array([1.0, 1.0]), a=a_, b=b_)
        T.add_limits([2], 0.0, np.inf, eps_m * 1e-15, 'monotonicity')
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.4, 0.1], (K, 1)), sw, sw // 10, 1, be, 1, 13, monitor=(0, 1), verbose=False,
                    keep_samples=False)
        Mo = tr['mon'][:, sw // 10:, :]
        ok &= int(tr['band_outside'].max()) == 0 and bool((Mo[:, :, 1] <= Mo[:, :, 0]).all())
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            p, e_on = on.mean(), ess(on)
            sd_p = np.sqrt(max(pm_on[i] * (1 - pm_on[i]), 1e-12) / e_on)
            zp, zf = (p - pm_on[i]) / sd_p, (p - pm_free[i]) / sd_p
            ok &= abs(zp) < 4
            extra = ''
            if i == 0:
                uu = np.log10(Mo[:, :, 0][Mo[:, :, 0] > 0])
                zu = (uu.mean() - em1) / (uu.std() / np.sqrt(max(1.0, e_on * p)))
                ok &= abs(zu) < 4
                extra = f'; E[log10 m | on] {uu.mean():.4f} (exact {em1:.4f}, z {zu:+.2f}; without {em1_free:.4f})'
            print(f'   {be:5s} MFD monotonic, rate {i}: P(on) {p:.4f} (exact {pm_on[i]:.4f}, z {zp:+.2f}; without it '
                  f'{pm_free[i]:.4f}, z {zf:+.1f}){extra}')
    # a capped rise (--mfd-monotonic where the target rises, --mfd-rise-sd): the same two rates and bins, the upper
    # bin's total at most rho = 1.5 times the lower's, the row eps (1.5 m0 - m1) >= 0
    def exact_rise(rho):
        def Lb(m1, m2):
            Y = np.stack(np.broadcast_arrays(m1 + 0.7 * m2, 0.5 * m1 + m2), -1)
            return np.exp(-0.5 * np.sum((Y - dM) ** 2, -1) / sig ** 2)
        L1 = Lb(10 ** u[0], 0.0)
        L12 = Lb(M1, M2) * (M2 <= rho * M1) * gw[0][:, None] * gw[1][None, :]
        w00, w10, w11 = Lb(0.0, 0.0), gw[0] @ L1, L12.sum()
        W = w00 + w10 + w11
        return [(w10 + w11) / W, w11 / W], ((gw[0] * u[0]) @ L1 + L12.sum(1) @ u[0]) / (w10 + w11)
    pr_on, er1 = exact_rise(1.5)
    for be in backends:
        Gm = sp.csc_matrix(np.vstack([Gs, [1.5 * eps_m, -eps_m]]))
        T = Target(Gm, np.r_[dM, 0.0], sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0))
        T.n_data = T.n_fit = 2
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.array([0, 1], np.int64), B=2, nb=np.array([1.0, 1.0]), a=a_, b=b_)
        T.add_limits([2], 0.0, np.inf, eps_m * 1e-15, 'the MFD\'s monotonicity')
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.4, 0.1], (K, 1)), sw, sw // 10, 1, be, 1, 19, monitor=(0, 1), verbose=False,
                    keep_samples=False)
        Mo = tr['mon'][:, sw // 10:, :]
        ok &= int(tr['band_outside'].max()) == 0 and bool((Mo[:, :, 1] <= 1.5 * Mo[:, :, 0] * (1 + 1e-12)).all())
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            p = on.mean()
            if on.std() == 0:                        # on (or off) in every draw: plausible only if the other state is
                zp = 0.0 if on.size * min(pr_on[i], 1 - pr_on[i]) < 5 else np.inf      # expected under 5 times
            else:
                zp = (p - pr_on[i]) / np.sqrt(max(pr_on[i] * (1 - pr_on[i]), 1e-12) / ess(on))
            ok &= abs(zp) < 4
            extra = ''
            if i == 0:
                uu = np.log10(Mo[:, :, 0][Mo[:, :, 0] > 0])
                zu = (uu.mean() - er1) / (uu.std() / np.sqrt(max(1.0, ess(on) * p)))
                ok &= abs(zu) < 4
                extra = f'; E[log10 m | on] {uu.mean():.4f} (exact {er1:.4f}, z {zu:+.2f})'
            print(f'   {be:5s} MFD capped rise (rho 1.5), rate {i}: P(on) {p:.4f} (exact {pr_on[i]:.4f}, z {zp:+.2f}; '
                  f'flat {pm_on[i]:.4f}){extra}')
    # the plate rate (--plate-rate): the spike-and-slab problem, with a constraint row outside the likelihood,
    # eps (m0 - 0.5 m1), kept within eps [0.3, 0.5]: limits on both sides, on a row with coefficients of both signs (as a
    # fault slipping against the plate motion gives); it forbids m0 off, and cuts m0's upper side, which m1 on relieves
    qlo, qhi, eps_q = 0.3, 0.5, 1e-9

    def exact_plate(plate):
        def Lb(m1, m2):
            Y = np.stack(np.broadcast_arrays(m1 + 0.7 * m2, 0.5 * m1 + m2), -1)
            q = np.broadcast_arrays(m1 - 0.5 * m2, Y[..., 0])[0]
            L = np.exp(-0.5 * np.sum((Y - d) ** 2, -1) / sig ** 2)
            return L * ((q >= qlo) & (q <= qhi)) if plate else L
        L1, L2, L12 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1]), Lb(M1, M2) * gw[0][:, None] * gw[1][None, :]
        w00, w10, w01, w11 = pg[0] * Lb(0.0, 0.0), pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2), pg[2] * L12.sum()
        W = w00 + w10 + w01 + w11
        e1 = (pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11)
        return [(w10 + w11) / W, (w01 + w11) / W], e1
    (pq_on, eq1), (pq_free, eq1_free) = exact_plate(True), exact_plate(False)
    for be in backends:
        Gq = sp.csc_matrix(np.vstack([Gs, [eps_q, -0.5 * eps_q]]))
        T = Target(Gq, np.r_[d, 0.0], sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0))
        T.n_data = T.n_fit = 2                           # the constraint row is outside the likelihood
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        T.add_limits([2], eps_q * qlo, eps_q * qhi, eps_q * 1e-9, 'the plate rate')
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.4, 0.1], (K, 1)), sw, sw // 10, 1, be, 1, 17, monitor=(0, 1), verbose=False,
                    keep_samples=False)
        Mo = tr['mon'][:, sw // 10:, :]
        q = Mo[:, :, 0] - 0.5 * Mo[:, :, 1]
        ok &= int(tr['band_outside'].max()) == 0 and bool(((q >= qlo - 1e-12) & (q <= qhi + 1e-12)).all())
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            p, e_on = on.mean(), ess(on)
            sd_p = np.sqrt(max(pq_on[i] * (1 - pq_on[i]), 1e-12) / e_on)
            zp, zf = (p - pq_on[i]) / sd_p, (p - pq_free[i]) / sd_p
            if pq_on[i] > 1 - 1e-12:                 # the limits forbid this rate's off state: never off
                zp = 0.0 if p == 1.0 else np.inf
            ok &= abs(zp) < 4
            extra = ''
            if i == 0:
                uu = np.log10(Mo[:, :, 0][Mo[:, :, 0] > 0])
                zu = (uu.mean() - eq1) / (uu.std() / np.sqrt(max(1.0, ess(np.log10(np.maximum(Mo[:, :, 0], 1e-300))))))
                ok &= abs(zu) < 4
                extra = f'; E[log10 m | on] {uu.mean():.4f} (exact {eq1:.4f}, z {zu:+.2f}; without {eq1_free:.4f})'
            print(f'   {be:5s} plate rate, rate {i}: P(on) {p:.4f} (exact {pq_on[i]:.4f}, z {zp:+.2f}; without it '
                  f'{pq_free[i]:.4f}, z {zf:+.1f}){extra}')
    # a log-box prior with swap moves: the selftest_swaps problem
    G2, d2, s2, se, F2 = np.array([[1.0, 0.98], [0.5, 0.53]]), np.array([0.4, 0.2]), 0.09, 0.5, 2.0
    ug = np.linspace(-3.0 - 6 * se, 0.0, 2001)
    U1, U2 = np.meshgrid(ug, ug, indexing='ij')
    fe = lambda x: np.where(x < -3.0, np.exp(-(x + 3.0) ** 2 / (2 * se ** 2)), 1.0)
    Y = np.stack([G2[0, 0] * 10 ** U1 + G2[0, 1] * 10 ** U2, G2[1, 0] * 10 ** U1 + G2[1, 1] * 10 ** U2], -1)
    Pg = np.exp(-np.sum((Y - d2) ** 2, -1) / (2 * s2)) * fe(U1) * fe(U2)
    ex_free = [np.sum(Pg * U1) / Pg.sum(), np.sum(Pg * U2) / Pg.sum()]
    Pg = Pg * inside(Y, d2, F2)
    ex = [np.sum(Pg * U1) / Pg.sum(), np.sum(Pg * U2) / Pg.sum()]
    pairs = (np.array([0]), np.array([1]), np.array([G2[:, 0] @ G2[:, 1]]))
    for be in backends:
        T = Target(G2, d2, 0.3, 0.0, 1.0, logbox=(-3, 0, 0.5), pairs=pairs)
        T.set_band(F2, [0, 1])
        tr = sample(T, np.tile([0.4, 1e-3], (4, 1)), 40000 if be == 'c' else 8000, 1000, 1, backend=be, monitor=[0, 1],
                    verbose=False, keep_samples=False, swaps=True)
        x = np.log10(tr['mon'][:, 1000:])
        ok &= int(tr['band_outside'].max()) == 0 and bool(inside(10 ** x @ G2.T, d2, F2).all())
        zs = [z(x[..., j], ex[j]) for j in (0, 1)]
        ok &= all(abs(v) < 4 for v in zs)
        print(f'   {be:5s} log-box + swaps: E[log10 m] ({x[..., 0].mean():+.4f}, {x[..., 1].mean():+.4f}) exact ({ex[0]:+.4f}, '
              f'{ex[1]:+.4f}) z = ({zs[0]:+.2f}, {zs[1]:+.2f}); without the band ({ex_free[0]:+.4f}, {ex_free[1]:+.4f}); '
              f'swap acceptance {tr["swap_rate"].mean():.2f}')
    # truncated normals (pkind 0) with a top per rupture: the former linear-prior test's problem
    G3, d3, sig3, top, F3 = np.array([[1.0, 0.7], [0.5, 1.0]]), np.array([0.6, 0.45]), 0.1, np.array([0.3, 1.0]), 1.25
    ul = [np.linspace(0.0, t, 3001) for t in top]
    A, B = np.meshgrid(ul[0], ul[1], indexing='ij')
    Y = np.stack([A + 0.7 * B, 0.5 * A + B], -1)
    Pg = np.exp(-np.sum((Y - d3) ** 2, -1) / (2 * sig3 ** 2))
    ex_free = [np.sum(Pg * A) / Pg.sum(), np.sum(Pg * B) / Pg.sum()]
    Pg = Pg * inside(Y, d3, F3)
    ex = [np.sum(Pg * A) / Pg.sum(), np.sum(Pg * B) / Pg.sum()]
    for be in backends:
        T = Target(sp.csc_matrix(G3), d3, sig3, 0.0, top)
        T.set_band(F3, [0, 1])
        K, sw = (8, 40000) if be == 'c' else (4, 8000)
        tr = sample(T, np.full((K, 2), 0.3), sw, 1000, 1, be, 1, 13, monitor=(0, 1), verbose=False, keep_samples=False)
        x = tr['mon'][:, 1000:]
        ok &= int(tr['band_outside'].max()) == 0 and bool(inside(x @ G3.T, d3, F3).all())
        zs = [z(x[..., j], ex[j]) for j in (0, 1)]
        ok &= all(abs(v) < 4 for v in zs)
        print(f'   {be:5s} per-rupture tops: E[m] ({x[..., 0].mean():.4f}, {x[..., 1].mean():.4f}) exact ({ex[0]:.4f}, {ex[1]:.4f}) '
              f'z = ({zs[0]:+.2f}, {zs[1]:+.2f}); without the band ({ex_free[0]:.4f}, {ex_free[1]:.4f})')
    print('SELFTEST (band)', 'PASSED' if ok else 'FAILED')
    return bool(ok)



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


def mstep_build(T, rows, orig, sigma_s, width):
    """--mfd-step-sd: the MFD steps' term for the kernel (stctx). rows: the magnitude bins' rows of T, in magnitude order
    (cru_common.magnitude_bins), orig their original row numbers (a gap breaks a pair). With lam_b = ln y_b, y_b = r_b + d_b
    each bin's prediction (in its row's units, as the kernel has it), each pair p of neighbouring bins (a, a + 1) has
    eps_p = lam_{a+1} - lam_a - e_p, e_p = ln(d_{a+1} / d_a) the target's own step, so eps_p is ln of the draw's step over
    the target's (each row's coefficient cancels); S = sum_p (eps_p - mean eps)^2 / tau^2, tau = width x sigma_s x ln 10
    (sigma_s in b-value units: a step straying from the draw's common tilt by sigma_s x width decades costs 1), on chi2's
    scale. Its quadratic form in lam: A = D' P D / tau^2, c = D' P e / tau^2, const = e' P e / tau^2 (D the pairs'
    differences, P = I - 1 1' / n_pairs). Each rupture's bin (bx, -1 none) and its coefficient there (gx > 0)."""
    rows = np.asarray(rows, np.int64)
    orig = np.asarray(orig)
    dy = np.asarray(T.d, np.float64)[rows]
    if not (dy > 0).all():
        raise SystemExit('--mfd-step-sd: every magnitude bin needs a target above 0')
    pairs = [(a_, a_ + 1) for a_ in range(len(rows) - 1) if orig[a_ + 1] == orig[a_] + 1]
    if len(pairs) < 2:
        raise SystemExit('--mfd-step-sd: needs at least two pairs of neighbouring magnitude bins')
    nb_, np_ = len(rows), len(pairs)
    D = np.zeros((np_, nb_))
    for p_, (a_, b_) in enumerate(pairs):
        D[p_, a_], D[p_, b_] = -1.0, 1.0
    e = D @ np.log(dy)
    Pc = np.eye(np_) - 1.0 / np_
    tau = width * sigma_s * LN10
    A = D.T @ Pc @ D / tau ** 2
    c = D.T @ Pc @ e / tau ** 2
    rb = np.full(T.N, -1, np.int64)
    rb[rows] = np.arange(nb_)
    col = np.repeat(np.arange(T.n, dtype=np.int64), np.diff(T.indptr))
    sel = rb[T.indices] >= 0
    if np.bincount(col[sel], minlength=T.n).max() > 1:
        raise SystemExit('--mfd-step-sd: a rupture is in more than one magnitude bin')
    bx = np.full(T.n, -1, np.int32)
    gx = np.zeros(T.n)
    bx[col[sel]] = rb[T.indices[sel]]
    gx[col[sel]] = T.data[sel]
    if not (gx[bx >= 0] > 0).all():
        raise SystemExit('--mfd-step-sd: every bin row\'s coefficients must be above 0')
    return dict(row=rows, A=np.ascontiguousarray(A), c=np.ascontiguousarray(c), dy=np.ascontiguousarray(dy),
                bx=np.ascontiguousarray(bx), gx=np.ascontiguousarray(gx), D=D, e=e, tau=tau, sigma_s=float(sigma_s),
                width=float(width), pairs=np.array(pairs, np.int64), const=float(e @ Pc @ e / tau ** 2))


def group_weights(P, mode, slip_block='S', bins_block='MFD'):
    """--group-weights (DATATYPES_EQUAL): each data group's weight on its log likelihood, so that the slip rates, the MFD
    bins and the paleo rates (every other block holding data) count equally. mode 'equal': each group's effective
    measurements N_g (cru_common.block_measurements, as the fit check counts them: a slip-rate section once, a template
    or link group as its weights say, each bin and each site once) made a third of their sum, w_g = sum N / (3 N_g), so
    the total is kept. Returns dict(w: group -> weight, n: group -> N_g, row: each problem row's weight)."""
    from cru_common import block_measurements
    if mode != 'equal':
        raise SystemExit(f'--group-weights {mode!r}: use equal')
    grp = {}
    for b_ in block_measurements(P):
        g_ = 'slip' if b_['name'] == slip_block else 'mfd' if b_['name'] == bins_block else 'paleo'
        if g_ == 'paleo' and 'paleo' in grp:
            raise SystemExit('--group-weights: more than one row block besides the slip rates and the MFD bins')
        grp[g_] = (b_['index'], float(b_['expect']))
    if len(grp) < 2:
        raise SystemExit('--group-weights: needs at least two data groups')
    tot = sum(n_ for _, n_ in grp.values())
    w = {g_: tot / (len(grp) * n_) for g_, (_, n_) in grp.items()}
    row = np.ones(P['G'].shape[0])
    blk = np.asarray(P['block'])
    for g_, (i_, _) in grp.items():
        row[blk == i_] = w[g_]
    return dict(w=w, n={g_: n_ for g_, (_, n_) in grp.items()}, row=row)


def mjoint_build(T, P, rows, orig, tau_dec, width, K, n_b=481, n_l=801, w=1.0):
    """--mfd-gr-tau (MFD_GR_TAU): the bins scored jointly about each draw's G-R curve. b and the level L = ln(s m / m_mid)
    are each chain's hyperparameters, under their MFD_GR priors (mfd_gr.b_logpdf, level_logpdf); the draw's curve over the
    bins' targets is exp(L + shape(b)) (mfd_gr.joint_grid, saved with the problem by cleaning), and the residual rho = lam -
    ln dy - L - shape(b) (lam = ln of each bin's prediction, dy its target, in its row's units) is scored by
    S = sum_p (rho_{a+1} - rho_a)^2 / tau^2 + (mean rho)^2 / tau^2 (tau = tau_dec x ln 10; tau_dec the shape's tolerance in
    decades per step, the user's MFD_GR_TAU): the steps' departures from the curve's, and the bins' mean level tied to it
    within one step's tolerance, so that the level and tilt are all b's and the level's (none of them the residual's). S is
    the MFD's misfit on chi2's scale (rank NB: near NB if the residuals scatter as tau says); in the kernel it is the step
    term's quadratic form with A = Q fixed and c = Q (ln dy + L + shape(b)) each chain's own, updated as b and L are
    drawn (mjoint_update). The bins' own PDFs are not used, so the catalogue's information enters once, as b's and the
    level's priors."""
    import mfd_gr as mg
    if P.get('mfd_gr_shape') is None or P.get('mfd_gr') is None:
        raise SystemExit('--mfd-gr-tau needs a problem cleaned with preliminaries.py --mfd-gr since November 18th, 2026 (each '
                         'bin\'s G-R shape over b, mfd_gr_shape): clean again')
    ms = mstep_build(T, rows, orig, tau_dec / width, width)
    if len(ms['pairs']) != len(rows) - 1:
        raise SystemExit('--mfd-gr-tau: the magnitude bins must be contiguous')
    nb_, tau = len(rows), tau_dec * LN10
    Q = (ms['D'].T @ ms['D'] / tau ** 2 + np.ones((nb_, nb_)) / (nb_ ** 2 * tau ** 2)) * float(w)   # DATATYPES_EQUAL' w
    knobs = mg.parse_knobs(P['mfd_gr'])
    bg0 = np.asarray(P['mfd_gr_bgrid'], float)
    sh0 = np.asarray(P['mfd_gr_shape'], float)[rows]
    if not np.all(np.isfinite(sh0)):
        raise SystemExit('--mfd-gr-tau: some bin has no G-R shape in the problem (mfd_gr_shape)')
    if len(bg0) > 1:                                    # b on a fine grid, the shape (smooth in b) interpolated
        bg = np.linspace(bg0[0], bg0[-1], n_b)
        sh = np.array([np.interp(bg, bg0, r_) for r_ in sh0])
    else:
        bg, sh = bg0, sh0
    lpv, v_lo, v_hi, m_c = mg.level_logpdf(knobs)
    lg = np.linspace(v_lo, v_hi, n_l) - math.log(m_c)
    ms.update(A=np.ascontiguousarray(Q), c=np.zeros((K, nb_)), joint=True, tau=tau, tau_dec=float(tau_dec), Q=Q,
              lnd=np.log(ms['dy']), shape=sh, bgrid=bg, lgrid=lg, lpb=np.asarray(mg.b_logpdf(knobs, bg), float),
              lpl=np.asarray(lpv(lg + math.log(m_c)), float), QG=Q @ sh, GQG=np.sum(sh * (Q @ sh), 0),
              oQG=Q.sum(0) @ sh, Q1=Q.sum(1), oQo=float(Q.sum()), knobs=knobs, m_c=float(m_c),
              theta=np.zeros((K, 2)), ib=np.zeros(K, np.int64), il=np.zeros(K, np.int64), w=float(w),
              n5=np.asarray(mg.n_of_b(knobs, bg), float))   # N(M>=5) at each b (s = m = 1): the draw's is s m N(b)
    ms.pop('const', None)
    return ms


def mjoint_update(ms, R, g, chains=None):
    """--mfd-gr-tau: each chain's b and level drawn exactly from their conditional given its bins' predictions (on the
    grids of b and L: prior x exp(-S / 2)), and the chain's c in the kernel's step term set to its new curve."""
    R = np.atleast_2d(R)
    for k in (range(R.shape[0]) if chains is None else chains):
        with np.errstate(divide='ignore'):
            u = np.log(R[k, ms['row']] + ms['dy']) - ms['lnd']
        if not np.all(np.isfinite(u)):
            raise SystemExit('--mfd-gr-tau: a magnitude bin is predicted at 0; every bin must hold a rupture that is on')
        a_ = u @ ms['QG']
        s1 = float(u @ ms['Q1'])
        L = ms['lgrid']
        lp = (ms['lpb'] + 0.5 * (2.0 * a_ - ms['GQG']))[:, None] + ms['lpl'][None, :] \
            + 0.5 * (2.0 * L[None, :] * (s1 - ms['oQG'][:, None]) - L[None, :] ** 2 * ms['oQo'])
        w = np.exp(lp - lp.max()).ravel()
        j = int(np.searchsorted(np.cumsum(w), g.random() * w.sum()))
        i_, l_ = divmod(min(j, w.size - 1), len(L))
        ms['ib'][k], ms['il'][k] = i_, l_
        ms['theta'][k] = ms['bgrid'][i_], L[l_]
        ms['c'][k] = ms['Q'] @ (ms['lnd'] + L[l_] + ms['shape'][:, i_])


def mjoint_value(ms, R):
    """--mfd-gr-tau: S, the bins' misfit about each chain's current curve, on chi2's scale (+inf where a bin is at 0)."""
    R = np.atleast_2d(R)
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = np.log(R[:, ms['row']] + ms['dy'])
    rho = lam - ms['lnd'] - ms['theta'][:, 1:2] - ms['shape'][:, ms['ib']].T
    out = np.einsum('kb,bc,kc->k', rho, ms['Q'], rho) / ms.get('w', 1.0)   # recorded without DATATYPES_EQUAL' w
    return np.where(np.isfinite(lam).all(1), out, np.inf)


def mstep_value(ms, R):
    """--mfd-step-sd: the term S for each chain (rows of R, the residuals), on chi2's scale; +inf where a bin is not
    predicted above 0."""
    R = np.atleast_2d(R)
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = np.log(R[:, ms['row']] + ms['dy'])
    eps = lam @ ms['D'].T - ms['e']
    eps = eps - eps.mean(1, keepdims=True)
    out = np.sum(eps ** 2, 1) / ms['tau'] ** 2
    return np.where(np.isfinite(lam).all(1), out, np.inf)


def positive_start(T, U):
    """A start with every row of T's positivity above zero, from U (rates in T's units, on its prior's support): for the
    rows predicted at 0, ruptures crossing them are switched on at twice their slab floor, each time the one that
    crosses the most such rows (ties: the lowest index), until none is left. At the slab floor these rates add next to
    nothing to the fit. Returns (the new U, how many were switched on)."""
    U = np.array(U, np.float64)
    rows = T.band.get('pos_rows', T.band['rows'])     # with per-group limits, the rows with positivity only
    Gr = sp.csr_matrix(T.G)[rows]
    Gc = Gr.tocsc()
    covered = np.asarray(Gr @ (U > 0).astype(np.float64)).ravel() > 0
    added = 0
    while not covered.all():
        open_ = np.flatnonzero(~covered)
        counts = np.bincount(Gr[open_].indices, minlength=T.n)
        i = int(np.argmax(counts))
        U[i] = max(U[i], 2.0 * T.band['floor'][i])
        covered[Gc.indices[Gc.indptr[i]:Gc.indptr[i + 1]]] = True
        added += 1
    return U, added


def rupture_bins(P, rows, block):
    """Each rupture's magnitude bin: the index, in rows, of the bin row holding it."""
    Gr = P['G'].tocsr()
    bins = np.full(P['G'].shape[1], -1, np.int64)
    for k, j in enumerate(rows):
        bins[Gr.indices[Gr.indptr[j]:Gr.indptr[j + 1]]] = k
    if (bins < 0).any():
        raise SystemExit(f'{int((bins < 0).sum())} ruptures are in no row of block {block!r}')
    return bins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--problem', default='cru_clean.npz')
    ap.add_argument('--chains', type=int, default=4)
    ap.add_argument('--sweeps', type=int, default=4000)
    ap.add_argument('--burn', type=int, default=1000)
    ap.add_argument('--thin', type=int, default=4)
    ap.add_argument('--init', default='nnls', help="'nnls' (the best fit, from prelim_results.npz), 'fitted' (the best fit "
                    "as fitted, nnls/fitted_vectors.npz next to --problem, from best_fit_as_fitted.py; checked against this "
                    "problem's NNLS fit), 'zero', a .npy of shape (n,) or (chains, n), or a previous run's .npz (uses its final states)")
    ap.add_argument('--prior', default='spikeslab',
                    help="default spikeslab (spike-and-slab per magnitude bin: each rupture off, rate 0, or on, log-uniform from the "
                         "slab floor 1e-10 up to a soft top at the bin's target rate R_j, sd = the bin's own uncertainty; the "
                         "fraction on in each bin estimated under a Beta(1, 1) prior; --slab sqrt: a slab uniform in sqrt(rate)); "
                         "spikeslab:A:TOP:a:b "
                         "for slab floor 10^A, TOP = soft, hard or a factor F (hard top at F x R_j), and a Beta(a, b) prior; "
                         "'lognormal' (hierarchical, retired); 'flat'; a prior mean tau (exponential prior) or a .npy of tau_i. "
                         "The single log-box, the per-bin boxes (binbox, binbox-floor, binbox-theta), the halfway and the linear "
                         "priors were removed on November 23rd, 2026")
    ap.add_argument('--down-the-middle', nargs='?', const='equal', default=None, choices=['equal', 'zero-model'],
                    help="add the balance term (the notebook's DOWN_THE_MIDDLE): each data type's relative shortfall s = (sum of "
                         "predictions - sum of targets) / sum of targets (slip-rate sections, magnitude bins, paleo rows; in units "
                         "of each measurement's uncertainty) costs A s^2. 'equal' (the default when given alone): A the largest "
                         "weight common to the types that keeps each at or below one extra measurement; 'zero-model': each type's "
                         "own A, its zero-model misfit, so s = -1 or +1 costs a type what the zero model costs it in chi2 "
                         "(cru_common.balance_rows); as extra rows of G and d, so for any prior. "
                         "The recorded chi2 and predictions stay on the data rows; the term is recorded on its own")
    ap.add_argument('--bins-block', default='MFD', help='lognormal prior: row block whose rows define the bins')
    ap.add_argument('--hyper-every', type=int, default=None,
                    help='spike-and-slab and lognormal priors: update the hyperparameters (the fractions on; the centres and '
                         'spread) every this many sweeps (default: every sweep for spike-and-slab, every --thin sweeps for '
                         'lognormal)')
    ap.add_argument('--paleo-misfit', default='gaussian', choices=['gaussian', 'years', 'gaussian+years'],
                    help='the paleo rows\' misfit (the notebook\'s PALEO_MISFIT): \'gaussian\' (default), the inputs\' '
                         'Gaussian in rate; \'years\': -2 ln of each site\'s recurrence-interval PDF in years, a curve '
                         'through its quantiles with its mode at 1/bestEstimate (paleo_misfit.py; needs --rupture-set and '
                         'spike-and-slab, and keeps each paleo prediction above zero); \'log\': the '
                         'same curve\'s density in log rate, its mode at bestEstimate; \'gaussian+years\' (the notebook\'s '
                         'MISFIT \'half\'): half the Gaussian in rate plus half the years curve, the geometric mean of the two')
    ap.add_argument('--slip-misfit', default='split', choices=['split', 'split+log10', 'log10', 'exact'],
                    help="the slip-rate rows' misfit: 'exact' (since 2026-11-23, the user's request), each row scored by "
                         "its fault's exact timeframe-corrected density (cleaning's slip_pdf_*, from timeframe_sliprates.py), "
                         "-2 ln p, tabulated, weighted as the split normal would be (cru_common.slip_pdf_tables), and with "
                         "--slip-sum each cross-section's sum by the convolution of its sections' densities; 'split' (the "
                         "default), the Gaussian (a split normal with the CFM's slip rates); 'split+log10', half of it plus half a Gaussian in log10(prediction / target) with the same "
                         "sd, s = log10(1 + sd / target) on each side (sd the side's effective sd, the template weights' "
                         "included, as --mfd-misfit log10 has the bins'), the geometric mean of the two densities, so a "
                         'prediction an order of magnitude off is unlikely and two orders far more so (each section still one '
                         'measurement; spike-and-slab, each slip-rate prediction kept above zero); \'log10\', the log10 '
                         'Gaussian alone')
    ap.add_argument('--mfd-misfit', default='gaussian', choices=['gaussian', 'log10', 'gaussian+log10', 'pdf'],
                    help="the MFD bins' misfit: gaussian (default), the inputs' Gaussian in rate; log10, each bin's total "
                         "scored by ((log10(total) - log10(R_j)) / s_j)^2, s_j = log10(1 + sigma_j / R_j) (the soft top's "
                         "width), in place of the Gaussian, with positivity on the bins (spike-and-slab, --all-positive "
                         "or --positive); gaussian+log10, half of each, the geometric mean; pdf "
                         "(MFD_GR: a problem cleaned with preliminaries.py --mfd-gr), each bin by -2 ln of its own target PDF "
                         "in log rate, from its minimum, through the paleo curves' tables (the curves saved with the problem)")
    ap.add_argument('--all-positive', action='store_true',
                    help='every slip-rate, paleo and MFD prediction kept above zero (the notebook always), through the '
                         'per-group limits')
    ap.add_argument('--group-weights', default=None, choices=['equal'],
                    help='DATATYPES_EQUAL: weight each data group\'s log likelihood so that the slip rates (with the summed '
                         'slip rates), the MFD bins and the paleo rates count equally: each as a third of their effective '
                         'measurements in all (the fit check\'s counts); chi2 as fitted and every group\'s misfit are '
                         'recorded unweighted')
    ap.add_argument('--mfd-gr-tau', type=float, default=None, metavar='TAU',
                    help='MFD_GR_TAU, with --mfd-misfit pdf (MFD_GR): the bins scored jointly about each draw\'s G-R curve, '
                         'its b and level (ln of the N scaling times the magnitude-uncertainty factor) each chain\'s '
                         'hyperparameters under their MFD_GR priors, in place of each bin\'s own PDF; each pair of neighbouring '
                         'bins\' step in log10 total may depart from the curve\'s by about TAU decades, and their mean level from '
                         'it by about TAU (0.025: about 6%% a step)')
    ap.add_argument('--mfd-monotonic', action='store_true',
                    help='with --bins-block\'s magnitude bins: in every draw, wherever a bin\'s target is not above the next '
                         'lower bin\'s (neighbouring bins, counted up from --lowest-bin in steps of --bin-width; a bin '
                         'cleaning removed breaks the pair), its total rate is not above that bin\'s either, and where the '
                         'target rises the rise is capped (--mfd-rise-sd): a hard '
                         'constraint, alongside --band or --positive or alone. Kept by constraint rows, each '
                         'eps x (rho x lower bin total - upper bin total) >= 0, rho = 1 except where the target rises, with eps = 1e-9, '
                         'which add at most about 1e-20 to '
                         'chi2; a start that breaks it is scaled down bin by bin first')
    ap.add_argument('--cross-sections', default='all', choices=['all', 'between'],
                    help='--slip-sum and --plate-rate: their strips of cross-sections perpendicular to the plate motion '
                         '(Beavan et al. 2002\'s Euler pole), the subsections\' mean trace length wide (8.5 km at 211,000 '
                         'ruptures: 72 between the edges where the Alpine Fault goes offshore and through the Needles-'
                         'Wairarapa step-over in Cook Strait). all (default): over all of New Zealand; between: only '
                         'between the edges (plate_rate.py)')
    ap.add_argument('--plate-rate', default=None, metavar='HI or LO,HI',
                    help='with --rupture-set: in every draw, each strip\'s share of the Pacific-Australia plate rate that '
                         'the crustal faults take up (the average over its cross-sections of the faults\' slip vectors '
                         'projected on the local plate motion, over the local plate rate; plate_rate.py) at most HI '
                         '(0.8, say: the whole plate rate, at the 0.8 the ruptures of these inputs scale the fault '
                         'model\'s slip rates by), or within LO-HI; on the --cross-sections strips: a hard constraint, '
                         'alongside --band, --positive or --mfd-monotonic or alone. Kept by constraint rows, eps x share '
                         'with eps = 1e-9; a start above HI is scaled down to it')
    ap.add_argument('--slip-sum', action='store_true',
                    help='with --rupture-set: a soft term in the likelihood, beside the data: along every cross-section '
                         'perpendicular to the plate motion, the model\'s slip along the plate motion added up (each '
                         'subsection\'s slip vector projected on the local plate motion, signed) against the targets\' '
                         'added up, within the uncertainty of that sum (copies of one fault section\'s slip rate correlated), each '
                         'subsection counting once in all; on the --cross-sections strips, each weighted its width over '
                         'L0 (--slip-sum-l0; at the default, each strip one extra observation, 72 between the Alpine and '
                         'Cook Strait edges) (plate_rate.py). Recorded apart from chi2')
    ap.add_argument('--slip-sum-l0', type=float, default=8.5, metavar='KM',
                    help='--slip-sum: L0 in km (default 8.5, the strips\' width, so each strip\'s sum counts as one extra '
                         'observation; 120 is the target sums\' decorrelation length along the plate motion between the '
                         'edges, about the 612 km between them over its 5 Alpine-Wairau sections): each strip is weighted '
                         'its width over L0, so a shorter L0 weighs the term more')
    ap.add_argument('--rupture-set', default=None,
                    help='--plate-rate and --slip-sum: the OpenSHA rupture set of these inputs (a solution folder, or its '
                         'ruptures/), for the faults\' traces, rakes and dips and the ruptures\' sections and average slips')
    ap.add_argument('--fractions-on', default='drawn', choices=['collapsed', 'drawn'],
                    help='spike-and-slab: each bin\'s fraction on, Beta(a, b) a priori, \'drawn\' (each sweep, from its Beta '
                         'posterior given the bin\'s count on, and each rupture switched on or off given it) or '
                         '\'collapsed\' (integrated out: a rupture is on, given the others in its bin, with prior odds '
                         '(a + c) / (b + n - 1 - c), c of the other n - 1 on; the swaps across bins carry the counts\' ratio; '
                         'the fractions are drawn given the counts only to record them, and --hyper-every does not apply). '
                         'The same posterior either way')
    ap.add_argument('--swap-fraction', type=float, default=1.0, metavar='F',
                    help='swap moves: each pass proposes each pair with probability F (0 < F <= 1), a fresh random choice '
                         'each pass, rather than every pair (1, the default): a cheaper pass, still exact')
    ap.add_argument('--no-off-bounds', action='store_true',
                    help='--slip-misfit exact (and the split misfits): switch every off rupture on or off from its exact '
                         'change, a pass of table lookups over its column, rather than first from bounds on it (each chain\'s '
                         'rows\' slopes, kept current, and their terms\' largest second derivatives), with the exact change '
                         'only where they leave the decision open (the default since 2026-11-24). The same decisions; for '
                         'timing')
    ap.add_argument('--swap-cos', type=float, default=0.0,
                    help='add exact swap moves between columns with cosine similarity above this (>= 0.99; from prelim_results.npz)')
    ap.add_argument('--swap-cos-cross', type=float, default=0.0, metavar='C',
                    help='spike-and-slab: also swap ruptures in different magnitude bins whose columns have cosine above C '
                         '(the notebook: 0.95), with the prior\'s ratio (the bins\' fractions on and slabs) in the acceptance; '
                         'needs cleaning with preliminaries.py --dup-cross-cos for pairs below 0.99')
    ap.add_argument('--split-merge', type=int, default=0, metavar='N',
                    help='spike-and-slab: N split/merge proposals per chain after each sweep (0, the default: none), '
                         'reversible-jump moves between a rupture and the two ruptures that together are it (split_merge.py), '
                         'with the two\'s rates in proportion to keep every slip-rate row as it is: rate moved between '
                         'magnitude bins in one step. Exact; needs --rupture-set and the fractions on drawn')
    ap.add_argument('--split-merge-scale', type=float, default=1.0, metavar='C',
                    help='--split-merge: the sd of a split\'s deviation from the exact proportion, in ln rate, '
                         'C sigma / (u_A |G_A|) (about C in the slip-rate rows\' misfit), at most 1 (default C = 1)')
    ap.add_argument('--threads', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='gibbs_run')
    ap.add_argument('--no-samples', action='store_true', help='do not write the (chains, draws, n) float32 sample file; the running '
                    'summaries in <out>_stream.npz are written either way')
    ap.add_argument('--diag-floor', type=float, default=-10.0, help='log10 rate below which rates count as equal in the running '
                    'R-hat and effective-draw summaries (default -10; the notebook passes its FLOOR)')
    ap.add_argument('--keep-draws', type=lambda v: -1 if str(v).lower() == 'all' else int(v), default=0,
                    help='keep this many whole draws per run (divided evenly among the chains, evenly spaced over the recorded '
                         'draws after burn-in), or every recorded draw after burn-in (all), in <out>_kept.npz, a sparse matrix of '
                         'the rates on (float32; since 2026-11-25, a dense <out>_kept.npy before), with their chain and record '
                         'numbers in <out>_kept_index.npy; recording only, the chains are unchanged (select_draws.py uses them)')
    ap.add_argument('--slip-block', default='S', help='row block of slip-rate rows (subsection totals in the running summaries)')
    ap.add_argument('--selftest', nargs='?', const='all', choices=['all', 'current'],
                    help="exact tests: 'all' (the default) after any change to the sampler, thirty verdicts (since 2026-11-24, with "
                         "the split/merge moves' three); 'current' only what the notebook's default uses (per-rupture lower edges, "
                         "the soft top, spike-and-slab, the balance term, swap moves, the band, the column passes with the summed "
                         "slip rates folded, swaps across bins, the collapsed fractions on, the exact slip misfit and its summed "
                         "densities), thirteen verdicts")
    a = ap.parse_args()
    # fixed since 2026-11-23 (their options removed, the user's request): the diagnostics beside the problem, the slab
    # log-uniform, no hierarchical prior, transfer moves, all-rows positivity (--all-positive, per group, remains) or MFD
    # step term, the monotonic MFD's rise cap 0.89, the bins from 6.8 in steps of 0.1, the C kernel, histograms 0.01 wide
    a.prelim, a.slab, a.s_range, a.mu_prior, a.s_fixed, a.pnc_pilot, a.pnc_scale = None, 'log', '0.05,5', '-6,10', None, None, 1.0
    a.transfers, a.positive, a.mfd_step_sd, a.mfd_rise_sd = None, False, None, 0.89
    a.lowest_bin, a.bin_width, a.backend, a.hist_width = 6.8, 0.1, 'c', 0.01
    if a.selftest is None and not str(a.prior).startswith('spikeslab'):
        raise SystemExit(f'--prior {a.prior}: only spike-and-slab remains (spikeslab or spikeslab:A:TOP:a:b); the other priors '
                         '(lognormal, flat, exponential, the boxes, linear, halfway) were removed in November 2026')
    a.slip_band = a.paleo_band = None              # each group's limit, set below to 'positive' or left off (the factor
                                                   # and soft bands, --band, --slip-band, --paleo-band and --*log-band, were
                                                   # removed on 2026-11-23, the user's request: the misfits make them needless)
    if a.selftest == 'current':
        ok = selftest_zlo()
        ok &= selftest_edges()
        ok &= selftest_spikeslab()
        ok &= selftest_balance()
        ok &= selftest_swaps()
        ok &= selftest_band()
        ok &= selftest_fold(backends=('c',))
        ok &= selftest_crossswap(backends=('c',))
        ok &= selftest_crossswap(backends=('c',), collapse=True, swap_frac=0.5)
        ok &= selftest_collapse(backends=('c',))
        ok &= selftest_exact(backends=('c',))
        ok &= selftest_exact(backends=('c',), collapse=True, off_bounds=False)
        ok &= selftest_sum_pdf()
        sys.exit(0 if ok else 1)
    if a.selftest:
        ok = selftest()
        ok &= selftest_hier()
        ok &= selftest_zlo()
        ok &= selftest_edges()
        ok &= selftest_spikeslab()
        ok &= selftest_spikeslab(gam=0.5)
        ok &= selftest_balance()
        ok &= selftest_band()
        ok &= selftest_paleo()
        ok &= selftest_paleo(b=0.5)
        ok &= selftest_split()
        ok &= selftest_exact()
        ok &= selftest_exact(collapse=True, off_bounds=False)
        ok &= selftest_sum_pdf()
        ok &= selftest_groupweights()
        ok &= selftest_crossswap()
        ok &= selftest_crossswap(collapse=True, swap_frac=0.5)
        ok &= selftest_collapse()
        ok &= selftest_spikeslab(collapse=True)
        ok &= selftest_mfdlog()
        ok &= selftest_mfdlog(b=0.5)
        ok &= selftest_mfdpdf()
        ok &= selftest_mfdstep()
        ok &= selftest_mfdjoint()
        ok &= selftest_slipgeo()
        ok &= selftest_slipgeo(b=1.0)
        ok &= selftest_fold()
        ok &= selftest_splitmerge('gauss')
        ok &= selftest_splitmerge('curve')
        ok &= selftest_splitmerge_counts()
        sys.exit(0 if ok else 1)
    kept = (a.sweeps - a.burn) // a.thin
    if kept < 4:
        raise SystemExit(f'only {max(kept, 0)} recorded draw(s) per chain after burn-in ((sweeps - burn) / thin = '
                         f'({a.sweeps} - {a.burn}) / {a.thin}); R-hat and effective draws need at least 4')
    report_to(a.out + '_log.txt')                        # everything printed below is also kept with the run
    P = load_problem(a.problem)
    n = P['G'].shape[1]
    GW = group_weights(P, a.group_weights, a.slip_block, a.bins_block) if a.group_weights else None   # DATATYPES_EQUAL
    gw = (lambda g_: 1.0) if GW is None else (lambda g_: GW['w'].get(g_, 1.0))
    if GW is not None:
        print('group weights (--group-weights equal): each data group\'s log likelihood weighted so that the groups count '
              'equally, ' + ', '.join(f'{g_} {GW["n"][g_]:.1f} measurements x {GW["w"][g_]:.3f}' for g_ in GW['w'])
              + f' = {sum(GW["n"].values()) / len(GW["w"]):.1f} each (the summed slip rates with the slip rates\' weight); '
              'chi2 as fitted and each group\'s misfit recorded unweighted')
    G_fit, d_fit, B_bal = P['G'], P['d'], None       # the rows the sampler fits: the data, and the balance term's rows
    if a.down_the_middle:
        B_bal, b_bal, A_bal, info_bal = balance_rows(P, a.slip_block, a.bins_block, weights=a.down_the_middle)
        G_fit, d_fit = sp_csc(sp.vstack([sp.csr_matrix(P['G']), B_bal])), np.r_[P['d'], b_bal]
        print(f'down the middle: a balance term per data type, A s^2 with s its relative shortfall (sum of predictions / sum of '
              f'targets - 1, in units of each measurement\'s uncertainty), '
              + (f'the same A = {A_bal[0]:.4g} for every type; ' if a.down_the_middle == 'equal'
                 else 'each type\'s own A, its zero-model misfit; ')
              + '; '.join(f'{t["name"]} ({t["n"]}): ' + (f'A = {t["A"]:.4g}, ' if a.down_the_middle != 'equal' else '')
                          + f'{100 * t["A"] / t["zero_misfit"]:.0f}% of their zero-model misfit '
                          f'({t["zero_misfit"]:.0f}), {t["weight"] * t["n"]:.2f} of a measurement' for t in info_bal))
    A_ss, n_bal, ss_info = None, None, None        # --slip-sum: the summed slip rates' rows, after the balance term's
    cs = None                                      # --slip-sum's and --plate-rate's strips
    if a.slip_sum or a.plate_rate is not None:
        if not a.rupture_set:
            raise SystemExit(f'--{"slip-sum" if a.slip_sum else "plate-rate"} needs --rupture-set (the OpenSHA '
                             'rupture set: the faults\' geometry, the ruptures\' sections and slips)')
        from cru_common import find_rupture_set
        rd = find_rupture_set(a.rupture_set, search=False)
        if rd is None:
            raise SystemExit(f'--rupture-set {a.rupture_set}: no OpenSHA rupture set there (indices.csv, or ruptures/indices.csv)')
        from plate_rate import cross_sections
        try:
            cs = cross_sections(rd, P, a.cross_sections, a.slip_block)
        except (OSError, ValueError) as e:
            raise SystemExit(f'--{"slip-sum" if a.slip_sum else "plate-rate"}: {e}')
    if a.slip_sum:
        from plate_rate import describe_slip_sum, slip_sum_rows
        try:
            A_ss, b_ss, ss_info = slip_sum_rows(rd, P, a.cross_sections, a.slip_block, l0_km=a.slip_sum_l0, cs=cs)
        except (OSError, ValueError) as e:
            raise SystemExit(f'--slip-sum: {e}')
        n_bal = G_fit.shape[0]
        if GW is not None:                         # DATATYPES_EQUAL: the summed slip rates with the slip rates' weight
            A_ss, b_ss = sp.csr_matrix(A_ss) * np.sqrt(gw('slip')), b_ss * np.sqrt(gw('slip'))
        G_fit, d_fit = sp_csc(sp.vstack([sp.csr_matrix(G_fit), A_ss])), np.r_[d_fit, b_ss]
        print('summed slip rates: along each cross-section, the model\'s slip along the plate motion added up against the '
              'targets\' added up, '
              'a soft term in the likelihood beside the data (recorded apart from chi2); ' + describe_slip_sum(ss_info, A_ss.nnz)
              + '; the slip-rate rows checked against the rupture set')
    n_fit, mono = G_fit.shape[0], None             # the rows in the likelihood; --mfd-monotonic's constraint rows follow
    if a.mfd_monotonic:
        if a.bins_block not in P['block_names']:
            raise SystemExit(f'--mfd-monotonic: no row block named {a.bins_block!r} (--bins-block)')
        from cru_common import mfd_monotonic
        MM_ = mfd_monotonic(P, a.bins_block, a.lowest_bin, a.bin_width, a.mfd_rise_sd)   # shared with best_fit_as_fitted.py
        Rm, mlo, members = MM_['R'], MM_['lo'], MM_['members']
        flat, rise, skipped, pairs_m, rhos = MM_['flat'], MM_['rise'], MM_['skipped'], MM_['pairs'], MM_['rhos']
        # the upper bin's total may be up to rho times the lower's: 1 where the target does not rise; where it rises,
        # the ratio at the lower bin's target of the target difference plus --mfd-rise-sd sd of it (from the two bins'
        # uncertainties), the median distance of the other pairs' walls above their target differences with the NZSHM22
        # inputs (0.89 sd): a ratio, so the cap limits the MFD's shape and not its level
        eps = 1e-9
        C = sp.lil_matrix((len(pairs_m), n))
        for q, (k, k1) in enumerate(pairs_m):
            C[q, members[k]] = eps * rhos[q]
            C[q, members[k1]] = -eps
        G_fit, d_fit = sp_csc(sp.vstack([sp.csr_matrix(G_fit), C.tocsr()])), np.r_[d_fit, np.zeros(len(pairs_m))]
        mono = dict(pairs=pairs_m, rhos=rhos, members=members, mlo=mlo, eps=eps,
                    rows=np.arange(n_fit, n_fit + len(pairs_m)))
        name = lambda k: f'{mlo[k]:.2f}-{mlo[k] + a.bin_width:.2f}'
        print(f'MFD monotonic: in every draw, a bin\'s total rate is not above the next lower bin\'s wherever its target is '
              f'not: {len(flat)} pairs of neighbouring bins (from {name(flat[0][0])} up)' if flat else
              'MFD monotonic: no pair of neighbouring bins has a target that is not increasing')
        for k, k1 in rise:
            print(f'   {name(k)} -> {name(k1)}: the target rises ({Rm[k]:.3e} -> {Rm[k1]:.3e}, a ratio of {Rm[k1] / Rm[k]:.3f}), '
                  f'so the upper bin\'s total is at most {rhos[pairs_m.index((k, k1))]:.3f} times the lower\'s (the target '
                  f'difference plus {a.mfd_rise_sd:g} sd of it, as a ratio at the lower bin\'s target)')
        if skipped:
            print('   pairs left free (' + ('the target increases, or ' if a.mfd_rise_sd < 0 else '')
                  + 'a bin cleaning removed lies between): ' + ', '.join(f'{name(k)} -> {name(k1)}' for k, k1 in skipped))
    plate = None                                   # --plate-rate's constraint rows follow --mfd-monotonic's
    if a.plate_rate is not None:
        from plate_rate import describe_limits, describe_plate, parse_limits, plate_rows
        try:
            lo_p, hi_p = parse_limits(a.plate_rate)
            # every strip, whatever --cross-sections (since 2026-11-24, the user's request: CROSS_SECTIONS 'between'
            # limits the summed slip rates only)
            cs_p = cs if a.cross_sections == 'all' else cross_sections(rd, P, 'all', a.slip_block)
            A_p, pinfo = plate_rows(rd, P, 'all', a.slip_block, cs=cs_p)
        except (OSError, ValueError) as e:
            raise SystemExit(f'--plate-rate: {e}')
        if lo_p is not None and lo_p > 0 and len(pinfo['empty']):
            raise SystemExit(f'--plate-rate {a.plate_rate}: strips {list(pinfo["empty"])} hold no rupture, so their share '
                             f'is 0, below {lo_p:g}')
        r0, eps_p = G_fit.shape[0], 1e-9
        G_fit, d_fit = sp_csc(sp.vstack([sp.csr_matrix(G_fit), eps_p * A_p])), np.r_[d_fit, np.zeros(A_p.shape[0])]
        plate = dict(A=A_p, rows=np.arange(r0, r0 + A_p.shape[0]), lo=lo_p, hi=hi_p, eps=eps_p, k=pinfo['k'],
                     what=describe_limits(lo_p, hi_p))
        print(f'plate rate: in every draw, each strip\'s share of the plate rate {plate["what"]}; '
              + describe_plate(pinfo) + f', through {A_p.getnnz(1).min():,}-{A_p.getnnz(1).max():,} ruptures per strip; '
              'the slip-rate rows checked against the rupture set (the same subsections, in proportion to average slip)')
    logbox, beta, lognormal, scale, zlo_vec, bins, ku_vec, hi_u = None, None, None, None, None, None, None, 1.0
    spike_mode = False
    if a.slab != 'log' and not a.prior.startswith('spikeslab'):
        raise SystemExit('--slab applies to --prior spikeslab only')
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
    elif a.prior.split(':')[0] in ('binbox', 'binbox-floor', 'binbox-theta', 'halfway', 'linear', 'logbox', 'logbox-centre',
                                   'logbox-mean'):
        raise SystemExit(f'--prior {a.prior}: removed on November 23rd, 2026, with the single log-box, the per-bin boxes, the '
                         'halfway and the linear priors; spike-and-slab (the default), flat, exponential and the retired '
                         'lognormal remain')
    elif a.prior.startswith('spikeslab'):
        # spike-and-slab: each rupture off (rate 0) or on, its rate sampled as u = m / (F R_j) below 1, log-uniform from
        # the slab floor 10^A up to the bin's top, and a per-bin fraction on, estimated
        spike_mode = True                                # spikeslab:A:TOP:a:b (slab floor 10^A, A = -10 by default)
        sp_ = (a.prior.split(':') + [None] * 5)[1:5]
        W, se, top = float(sp_[0] or -10), 0.0, str(sp_[1] or 'soft').lower()
        ss_a, ss_b = float(sp_[2] or 1), float(sp_[3] or 1)
        soft_top, F = (top == 'soft'), (1.0 if top in ('soft', 'hard') else float(top))
        Rb, rows = bin_targets(P, a.bins_block)
        bins = rupture_bins(P, rows, a.bins_block)
        if a.transfers:
            raise SystemExit('transfer moves are not implemented for spike-and-slab')
        scale = Rb[bins] * F
        nb = np.bincount(bins)
        edge = "soft lower edge, sd %g decades" % se if se > 0 else "hard lower edge"
        rel = 1.0 / P['d'][rows]                         # each bin row's relative uncertainty (it is normalized by it)
        q84 = P.get('mfd_pdf_q84')                       # MFD_GR: each bin's upper spread, its 84.135% point over its target
        if q84 is not None and np.isfinite(q84[rows]).all():
            rel = q84[rows] / Rb - 1.0
            if not (rel > 0).all():
                raise SystemExit('MFD_GR: a bin\'s 84.135% point is not above its target (mfd_pdf_q84); clean again')
        su = np.log10(1.0 + rel) if soft_top else np.zeros(len(rows))     # upper-edge sd per bin, decades
        if soft_top:                                     # soft top at R_j: Gaussian in log rate with the bin's own uncertainty
            ku_vec = np.ascontiguousarray((1.0 / (2.0 * (su * LN10) ** 2))[bins])
            hi_u = 10.0
            edge += (f'; soft top at R_j, sd log10(1 + sigma_j/R_j) = {su.min():.3f}-{su.max():.3f} decades'
                     + (' (MFD_GR: each bin\'s upper spread, its 84.135% point over its target)' if q84 is not None else ''))
        else:
            edge += f'; hard top at {F:g} x R_j'

        Wj = np.log10(Rb * F) - W                        # common floor: depth from each bin's top down to 10^W
        if (Wj <= 0).any():
            raise SystemExit(f'the floor 1e{W:g} is not below every bin target (smallest {Rb.min():.2e})')
        zlo_vec = np.ascontiguousarray(-Wj[bins] * LN10)
        logbox = (-float(Wj.max()), 0.0, se)
        slab = 'rate uniform in sqrt(rate)' if a.slab == 'sqrt' else 'log-uniform rate'
        print(f'prior: spike-and-slab per magnitude bin: each rupture is off (rate 0) or on, with a {slab} from '
              f'the slab floor 1e{W:g} up to R_j ({Wj.min():.2f} to {Wj.max():.2f} decades; {edge}); the fraction on in '
              f'each bin ~ Beta({ss_a:g}, {ss_b:g}), ' + ('integrated out (--fractions-on collapsed; drawn given the counts '
              'to record them)' if a.fractions_on == 'collapsed' else 'estimated (drawn each sweep)'))
    else:
        beta = prior_rates(a.prior, n)
    from cru_common import misfit_log_weight
    from paleo_misfit import PALEO_EPS
    b_s, b_m, b_p = (misfit_log_weight(a.slip_misfit), misfit_log_weight(a.mfd_misfit), misfit_log_weight(a.paleo_misfit))
    lin_sc = lambda b_: 1.0 if b_ <= 0 else PALEO_EPS if b_ >= 1 else 1.0 / np.sqrt(2.0)   # each group's rows in G and d:
    lin_cut = lambda b_: 1.0 - PALEO_EPS ** 2 if b_ >= 1 else 0.5  # out of the Gaussian (log), or halved; the inner products'
    pm_rows = pm_tab = None                        # --paleo-misfit: the paleo rows, scaled out of the Gaussian (or halved)
    if a.paleo_misfit != 'gaussian':
        from paleo_misfit import curve_mode, load_sites, report_lines, tables
        if not spike_mode:
            raise SystemExit('--paleo-misfit needs spike-and-slab')
        from cru_common import find_rupture_set
        rd_p = find_rupture_set(a.rupture_set, search=False) if a.rupture_set else None
        if rd_p is None:
            raise SystemExit('--paleo-misfit needs --rupture-set (the OpenSHA rupture set: its paleo_constraint_data.json)')
        sites = load_sites(rd_p)
        nraw = int(P['raw_shape'][0])
        where = {int(o): ii for ii, o in enumerate(np.asarray(P['keep_rows']))}
        orig = np.arange(nraw - len(sites), nraw)      # the inputs' last rows, in the file's order
        if not all(int(o) in where for o in orig):
            raise SystemExit(f'--paleo-misfit: the last {len(sites)} rows of the inputs are not all in the cleaned problem')
        pm_rows = np.array([where[int(o)] for o in orig])
        sd_p = np.array([s_['sd'] for s_ in sites])
        tgt = np.array([s_['best'] / s_['sd'] for s_ in sites])
        if not np.allclose(np.asarray(P['d'])[pm_rows], tgt, rtol=1e-9, atol=0):
            raise SystemExit('--paleo-misfit: the paleo rows\' targets are not bestEstimate / sd of the rupture set\'s '
                             'paleo_constraint_data.json')
        pm_tab = tables(sites, curve_mode(a.paleo_misfit))
        print('\n'.join(report_lines(sites, pm_tab)))
        if not all(f_[2] for f_ in pm_tab['fits']):
            raise SystemExit('--paleo-misfit: some sites\' curves could not be fitted (above)')
        rs_ = np.ones(G_fit.shape[0])
        rs_[pm_rows] = lin_sc(b_p)
        d_data = np.array(np.asarray(P['d'], float)[:P['G'].shape[0]])
        G_fit, d_fit = sp_csc(sp.diags(rs_) @ sp.csr_matrix(G_fit)), d_fit * rs_
    mf_rows = mf_s = None                          # --mfd-misfit log10: the bins' rows, scaled out of the Gaussian (or halved)
    if a.mfd_misfit != 'gaussian':
        if not spike_mode or a.transfers:
            raise SystemExit('--mfd-misfit log10 needs spike-and-slab, without --transfers')
        if a.bins_block not in P['block_names']:
            raise SystemExit(f'--mfd-misfit log10: no row block named {a.bins_block!r} (--bins-block)')
        mf_rows = np.flatnonzero(np.asarray(P['block']) == list(P['block_names']).index(a.bins_block))
        dmf = np.asarray(P['d'], float)[mf_rows]
        if not (dmf > 0).all():
            raise SystemExit('--mfd-misfit log10: every bin needs a positive target')
        mf_s = np.log10(1.0 + 1.0 / dmf)               # s_j = log10(1 + sigma_j / R_j); a bin's row is R_j / sigma_j
        if a.mfd_misfit == 'pdf':                      # MFD_GR: each bin scored by its own PDF, its curve from cleaning
            if P.get('mfd_pdf_th') is None or not np.all(np.isfinite(P['mfd_pdf_th'][mf_rows])):
                raise SystemExit('--mfd-misfit pdf needs a problem cleaned with preliminaries.py --mfd-gr (each bin\'s PDF)')
            mf_s = None
        rs_ = np.ones(G_fit.shape[0])
        rs_[mf_rows] = lin_sc(b_m)
        if pm_rows is None:
            d_data = np.array(np.asarray(P['d'], float)[:P['G'].shape[0]])
        G_fit, d_fit = sp_csc(sp.diags(rs_) @ sp.csr_matrix(G_fit)), d_fit * rs_
    sm_rows = None                                 # --slip-misfit split+log10, log10: the slip rows' Gaussian halved (scaled by
    if a.slip_misfit not in ('split', 'exact'):    # 1/sqrt 2), or scaled out of it
        if not spike_mode or a.transfers:
            raise SystemExit(f'--slip-misfit {a.slip_misfit} needs spike-and-slab, without --transfers')
        if a.slip_block not in P['block_names']:
            raise SystemExit(f'--slip-misfit {a.slip_misfit}: no row block named {a.slip_block!r} (--slip-block)')
        sm_rows = np.flatnonzero(np.asarray(P['block']) == list(P['block_names']).index(a.slip_block))
        if not (np.asarray(P['d'], float)[sm_rows] > 0).all():
            raise SystemExit(f'--slip-misfit {a.slip_misfit}: every slip-rate row needs a positive target')
        rs_ = np.ones(G_fit.shape[0])
        rs_[sm_rows] = lin_sc(b_s)
        if pm_rows is None and mf_rows is None:
            d_data = np.array(np.asarray(P['d'], float)[:P['G'].shape[0]])
        G_fit, d_fit = sp_csc(sp.diags(rs_) @ sp.csr_matrix(G_fit)), d_fit * rs_
    if a.slip_band is None and a.slip_misfit not in ('split', 'exact'):
        a.slip_band = 'positive'                   # its log10 half is infinite at 0: each slip-rate prediction above zero
    fprelim = a.prelim or prelim_path(a.problem)
    pairs = None
    if not 0.0 < a.swap_fraction <= 1.0:
        raise SystemExit(f'--swap-fraction {a.swap_fraction:g}: give a probability in (0, 1]')
    if a.swap_cos > 0:
        pr = np.load(fprelim)
        lo_dup = float(pr['dup_cos_min']) if 'dup_cos_min' in pr.files else 0.99   # cleaning's search: its lowest cosine,
        dcx_ = float(pr['dup_cross_cos']) if 'dup_cross_cos' in pr.files else np.nan  # and across bins (--dup-cross-cos)
        lo_x = min(lo_dup, dcx_) if np.isfinite(dcx_) else lo_dup
        if a.swap_cos < lo_dup - 1e-9 or (0 < a.swap_cos_cross < lo_x - 1e-9):
            raise SystemExit(f'swap moves: --swap-cos {a.swap_cos:g} (--swap-cos-cross {a.swap_cos_cross:g}) is below the '
                             f'cosine cleaning searched down to, {lo_dup:g}: clean again with preliminaries.py --dup-cos')
        sel = pr['dup_cos'] > a.swap_cos
        if lognormal is not None:                  # swaps must leave the prior unchanged: same bin only
            sel &= lognormal['bins'][pr['dup_i']] == lognormal['bins'][pr['dup_j']]
        n_cross = 0
        if zlo_vec is not None or ku_vec is not None:   # per-bin depths or tops: boxes differ
            same_ = bins[pr['dup_i']] == bins[pr['dup_j']]                     # between bins
            if a.swap_cos_cross > 0 and spike_mode and lognormal is None:   # --swap-cos-cross: across bins too, with the
                xs_ = ~same_ & (pr['dup_cos'] > a.swap_cos_cross)            # prior's ratio in the acceptance (swap_run)
                n_cross = int(xs_.sum())
                sel = (sel & same_) | xs_
                if not n_cross:
                    dc_ = float(pr['dup_cross_cos']) if 'dup_cross_cos' in pr.files else np.nan   # cleaning's search:
                    lo_ = dc_ if np.isfinite(dc_) else 0.99                                         # down to 0.99 by default
                    print(f'swap moves: no pairs across bins with cosine above {a.swap_cos_cross:g}'
                          + (f' (cleaning searched down to {lo_:g} only: preliminaries.py --dup-cross-cos for more)'
                             if a.swap_cos_cross < lo_ - 1e-9 else ' (the bins\' rows set the columns apart)'))
            else:
                sel &= same_
        elif a.swap_cos_cross > 0:
            raise SystemExit('--swap-cos-cross needs spike-and-slab with per-bin boxes (its prior ratio is the bins\')')
        pi, pj = pr['dup_i'][sel], pr['dup_j'][sel]
        pairs = (pi, pj, pr['dup_cos'][sel] * pr['cn'][pi] * pr['cn'][pj])
        if B_bal is not None:                      # the balance rows' share of each pair's inner product
            Bd = B_bal.toarray()
            pairs = (pi, pj, pairs[2] + np.einsum('kp,kp->p', Bd[:, pi], Bd[:, pj]))
        if A_ss is not None:                       # and the summed slip rates' rows' share
            Ac, ip = sp.csc_matrix(A_ss), np.array(pairs[2])
            for c0 in range(0, len(pi), 100000):
                c1 = min(c0 + 100000, len(pi))
                ip[c0:c1] += np.asarray(Ac[:, pi[c0:c1]].multiply(Ac[:, pj[c0:c1]]).sum(0)).ravel()
            pairs = (pi, pj, ip)
        if pm_rows is not None:                    # --paleo-misfit: the paleo rows' share, scaled out with them
            Gpc, ip = sp.csc_matrix(sp.csr_matrix(P['G'])[pm_rows]), np.array(pairs[2])
            for c0 in range(0, len(pi), 100000):
                c1 = min(c0 + 100000, len(pi))
                ip[c0:c1] -= lin_cut(b_p) * np.asarray(Gpc[:, pi[c0:c1]].multiply(Gpc[:, pj[c0:c1]]).sum(0)).ravel()
            pairs = (pi, pj, ip)
        if mf_rows is not None:                    # --mfd-misfit log10: the bins' share, scaled out with them
            Gmc, ip = sp.csc_matrix(sp.csr_matrix(P['G'])[mf_rows]), np.array(pairs[2])
            for c0 in range(0, len(pi), 100000):
                c1 = min(c0 + 100000, len(pi))
                ip[c0:c1] -= lin_cut(b_m) * np.asarray(Gmc[:, pi[c0:c1]].multiply(Gmc[:, pj[c0:c1]]).sum(0)).ravel()
            pairs = (pi, pj, ip)
        if sm_rows is not None:                    # --slip-misfit split+log10: half the slip rows' share, scaled with them
            Gsc, ip = sp.csc_matrix(sp.csr_matrix(P['G'])[sm_rows]), np.array(pairs[2])
            for c0 in range(0, len(pi), 100000):
                c1 = min(c0 + 100000, len(pi))
                ip[c0:c1] -= lin_cut(b_s) * np.asarray(Gsc[:, pi[c0:c1]].multiply(Gsc[:, pj[c0:c1]]).sum(0)).ravel()
            pairs = (pi, pj, ip)
        if scale is not None:                      # inner products of the scaled columns
            pairs = (pi, pj, pairs[2] * scale[pi] * scale[pj])
        print(f'swap moves: {sel.sum()} near-duplicate pairs with cosine > {a.swap_cos}'
              + (f' ({n_cross} of them across magnitude bins with cosine > {a.swap_cos_cross}, the prior\'s ratio in their '
                 f'acceptance)' if n_cross else '') + ', one pass per recorded draw'
              + (f', each pair proposed in a pass with probability {a.swap_fraction:g} (--swap-fraction)'
                 if a.swap_fraction < 1 else ''))
    dirs = dict(np.load(a.transfers)) if a.transfers else None
    if dirs is not None:
        print(f'transfer moves: {len(dirs["uu"])} directions from {a.transfers}, one pass per recorded draw')
    if scale is None:
        T = Target(G_fit, d_fit, P['sigma'], P['lo'], P['hi'], beta, logbox, pairs, dirs, lognormal)
    else:                                          # u = m / scale: columns scaled, box [1e-W, 1], u <= 1 is the hard top
        import scipy.sparse as _sp
        T = Target(sp_csc(G_fit @ _sp.diags(scale)), d_fit, P['sigma'], 0.0, hi_u, beta, logbox, pairs, dirs, lognormal)
        T.scale = scale
        T.zlo_vec = zlo_vec
        T.ku_vec = ku_vec
        if spike_mode:                               # pkind 3: x_i = 0 (off) or its slab value (on)
            T.pkind = 3
            T.ss = dict(bins=bins, B=len(rows), nb=np.bincount(bins, minlength=len(rows)).astype(float), a=ss_a, b=ss_b,
                        collapse=a.fractions_on == 'collapsed')
            T.off_bounds = not a.no_off_bounds
            T.gam = 0.5 if a.slab == 'sqrt' else 0.0
    T.swap_frac = float(a.swap_fraction)
    if GW is not None:                             # DATATYPES_EQUAL: recorded on the run, the summed slip rates' weight with it
        T.group_weights, T.temper_slip_sum = GW, gw('slip')
    T.n_data = P['G'].shape[0]                     # the data rows; the balance term's rows follow them
    T.n_bal = n_bal                                # the summed slip rates' rows, after the balance term's (None: none)
    T.n_fit = n_fit                                # and then --mfd-monotonic's and --plate-rate's constraint rows
    if ss_info is not None:
        T.slip_sum = dict(expect=ss_info['weight'] * len(ss_info['k']))
    B_ss = None                                    # the summed slip rates' rows as B times the slip-rate rows (column_passes):
    if ss_info is not None and a.slip_misfit != 'exact':   # each strip's c_k w_ki c_i over the slip-rate row's coefficient
                                                   # per mm/yr (--slip-misfit exact: not folded, each row its own table)
        rows_s = np.asarray(cs['rows'])            # of slip (sigma_i / row scale, and the row's own rescaling, lin_sc)
        rs_s = lin_sc(b_s) if sm_rows is not None else 1.0
        vals = (ss_info['scale'][:, None] * ss_info['W'] * ss_info['proj'][None, :]
                * (ss_info['sig_i'] / (np.asarray(P['row_scale'], float)[rows_s] * rs_s))[None, :]) * np.sqrt(gw('slip'))
        kk, jj = np.nonzero(vals)
        B_ss = sp.csr_matrix((vals[kk, jj], (kk, rows_s[jj])), shape=(vals.shape[0], T.N))
    note_ = column_passes(T, B_ss)
    if note_:
        print(note_)
    if T.pairs is not None:                        # each swap pair's inner product: every row of G, checked on a sample
        q = np.random.default_rng(0).choice(len(T.pairs[0]), min(2000, len(T.pairs[0])), replace=False)
        Gc = sp.csc_matrix(T.G)
        ipq = np.asarray(Gc[:, T.pairs[0][q]].multiply(Gc[:, T.pairs[1][q]]).sum(0)).ravel()
        if not np.allclose(ipq, T.pairs[2][q], rtol=1e-8, atol=1e-12 * float(np.abs(ipq).max())):
            raise SystemExit('swap moves: the pairs\' inner products do not match the columns (a bookkeeping error)')
    if B_bal is not None:
        T.balance = dict(A=A_bal, names=[t['name'] for t in info_bal])
    if P.get('split_lo2') is not None or GW is not None:   # --cfm-sliprates (and DATATYPES_EQUAL): each row's Gaussian weights
        if not spike_mode or a.transfers:
            raise SystemExit('the problem\'s slip rates are split normals (preliminaries.py --cfm-sliprates): the sampler '
                             'handles them under spike-and-slab, without --transfers')
        L2_, H2_ = np.ones(T.N), np.ones(T.N)
        nd_ = P['G'].shape[0]
        if P.get('split_lo2') is not None:
            L2_[:nd_], H2_[:nd_] = P['split_lo2'], P['split_hi2']
        sk_ = np.flatnonzero((L2_ != 1.0) | (H2_ != 1.0))
        if GW is not None:                         # DATATYPES_EQUAL: each group's Gaussian x its weight (recorded chi2 without)
            T.temper = np.ones(T.N)
            T.temper[:nd_] = GW['row']
            L2_, H2_ = L2_ * T.temper, H2_ * T.temper
        T.split = dict(L2=np.ascontiguousarray(L2_), H2=np.ascontiguousarray(H2_))
        if a.slip_misfit == 'exact':               # each slip-rate row's exact density, tabulated in its residual
            from cru_common import slip_pdf_tables
            if P.get('slip_pdf_key') is None:
                raise SystemExit('--slip-misfit exact: the problem has no exact densities (slip_pdf_*): clean it again with '
                                 '--cfm-timeframe (timeframe_sliprates.py\'s _pdfs.npz beside its table)')
            try:
                tab_, mism_ = slip_pdf_tables(P, L2_, H2_)
            except ValueError as e:
                raise SystemExit(f'--slip-misfit exact: {e}')
            if mism_ > 1e-6:
                raise SystemExit(f'--slip-misfit exact: the rows\' two sides imply weights {mism_:.1e} apart (a bookkeeping error)')
            r_ = tab_['rows']
            T.split['L2'], T.split['H2'] = T.split['L2'].copy(), T.split['H2'].copy()
            T.split['L2'][r_] = T.split['H2'][r_] = -1.0 - tab_['t'][r_]
            T.split['tab'] = {k_: np.ascontiguousarray(v_) for k_, v_ in tab_.items()}
            # the fit check: each section's misfit, its rows' weights (unweighted by DATATYPES_EQUAL, as chi2 is recorded)
            # times the fault's density's own mean and variance of -2 ln p - min, in place of the Gaussian's 1 and 2
            kap_u = tab_['kappa'] / (T.temper[r_] if getattr(T, 'temper', None) is not None else 1.0)
            sec_ = np.asarray(P['section'])[r_]
            key_ = np.where(sec_ >= 0, sec_, -1 - r_)
            kf_ = np.asarray(P['slip_pdf_key'])[r_]
            _, inv_ = np.unique(key_, return_inverse=True)
            Sk_ = np.bincount(inv_, kap_u)
            kf_s = np.zeros(len(Sk_), np.int64)
            kf_s[inv_] = kf_
            Ef_, Vf_ = np.asarray(P['slip_pdf_E'], float)[kf_s], np.asarray(P['slip_pdf_V'], float)[kf_s]
            T.split['exact_moments'] = (float(Sk_ @ Ef_), float(Sk_ ** 2 @ Vf_), float(Sk_.sum()), float(Sk_ @ Sk_))
            if ss_info is not None:                # --slip-sum: each cross-section's sum by its parents' densities convolved
                from cru_common import slip_sum_pdf_tables
                n_ss = T.n_fit - n_bal
                c_eff = np.asarray(ss_info['scale'], float) * np.sqrt(gw('slip'))
                try:
                    st_ = slip_sum_pdf_tables(P, ss_info, np.asarray(cs['rows']), c_eff)
                except ValueError as e:
                    raise SystemExit(f'--slip-misfit exact, --slip-sum: {e}')
                if len(st_['E0']) != n_ss:
                    raise SystemExit('--slip-misfit exact, --slip-sum: the cross-sections\' tables do not match their rows')
                n0, f0 = len(T.split['tab']['E0']), len(T.split['tab']['F'])
                tb = T.split['tab']
                for k_ in ('E0', 'IH', 'F', 'D', 'E', 'V'):
                    tb[k_] = np.ascontiguousarray(np.r_[tb[k_], st_[k_]])
                tb['OFF'] = np.ascontiguousarray(np.r_[tb['OFF'], st_['OFF'][1:] + f0].astype(np.int64))
                rr_ = np.arange(n_bal, T.n_fit)
                T.split['L2'][rr_] = T.split['H2'][rr_] = -1.0 - (n0 + np.arange(n_ss))
                T.slip_sum['expect'] = float(np.sum(st_['E'])) / gw('slip')   # recorded unweighted, as chi2
                print(f'summed slip rates (exact): each of the {n_ss} cross-sections scored by -2 ln of the density of its '
                      f'parents\' slip rates added up (their exact densities convolved, by FFT), weighted as the Gaussian '
                      f'about the summed targets would be; its expected term {T.slip_sum["expect"]:.1f} (the Gaussian\'s '
                      f'{ss_info["weight"] * n_ss:.1f}); not folded into the slip-rate rows')
            print(f'slip misfit (exact): the {len(r_)} slip-rate rows each scored by -2 ln of its fault\'s exact corrected '
                  f'density ({len(tab_["E0"])} tables, {len(tab_["F"]):,} points; cubic Hermite, quasiconvex), weighted as its '
                  'split normal would be; evaluated exactly in every update and swap')
        sk_ = np.log10(np.sqrt(L2_[sk_] / H2_[sk_]))            # log10 (sigma_hi / sigma_lo)
        c_ = P.get('cfm_sliprates') or ['?', '?', '?']
        if len(sk_) and P.get('split_lo2') is not None:
          print(f'slip rates from the CFM ({c_[0]}; SR_min and SR_max at {c_[1]} sigma'
                + ('' if str(c_[2]) in ('1', '1.0') else f', times {c_[2]}') + '): each slip-rate row a split '
              f'normal, {len(sk_)} of them lopsided, sd above / below the mode median {10 ** np.median(sk_):.2f} (5-95% {10 ** np.percentile(sk_, 5):.2f}-'
              f'{10 ** np.percentile(sk_, 95):.2f}); their misfit evaluated exactly in every update and swap')
    grp = any(v is not None for v in (a.slip_band, a.paleo_band))
    pos_g = np.zeros(0, np.int64)                  # with per-group limits: the rows with positivity
    if a.all_positive:                             # every data group above zero
        if a.positive:
            raise SystemExit('--all-positive is per group: not with --positive')
        a.slip_band = a.slip_band or 'positive'
        a.paleo_band = a.paleo_band or 'positive'
    if (mf_rows is not None or a.paleo_misfit != 'gaussian' or a.all_positive) and not a.positive:
        grp = True                                 # --mfd-misfit, --paleo-misfit, --all-positive: positivity through the per-group limits
    if grp:
        if dirs is not None or lognormal is not None:
            raise SystemExit('the per-group limits are not implemented for transfer moves or the hierarchical prior')
        sb_ = list(P['block_names'])
        dpos = np.asarray(P['d'])[:P['G'].shape[0]] > 0
        blk = np.asarray(P['block'])
        g_slip = np.flatnonzero(dpos & (blk == sb_.index(a.slip_block))) if a.slip_block in sb_ else np.zeros(0, np.int64)
        not_pal = [sb_.index(b_) for b_ in (a.slip_block, a.bins_block) if b_ in sb_]
        g_pal = np.flatnonzero(dpos & ~np.isin(blk, not_pal))   # every positive-target row in neither group: the paleo rows
        sl_b, pa_b = a.slip_band, a.paleo_band
        if pa_b is None and a.paleo_misfit != 'gaussian':
            pa_b = 'positive'                      # --paleo-misfit: each paleo prediction above zero (its misfit is
                                                   # infinite at 0)
        g_mfd = (mf_rows if mf_rows is not None else                 # --mfd-misfit, --all-positive: the bins above 0
                 np.flatnonzero(dpos & (blk == sb_.index(a.bins_block))) if a.all_positive and a.bins_block in sb_
                 else np.zeros(0, np.int64))
        pos_g = np.r_[g_slip if sl_b == 'positive' else np.zeros(0, np.int64), g_pal if pa_b == 'positive' else np.zeros(0, np.int64),
                      g_mfd]
        if len(pos_g):
            try:
                T.set_groups(pos_g, np.zeros(0, np.int64), np.zeros(0))
            except ValueError as e:
                raise SystemExit(f'the per-group limits: {e}')
        desc_ = lambda b_: 'no limit' if b_ is None else 'each above zero'
        print(f'slip-rate rows ({len(g_slip)}): {desc_(sl_b)}; paleo rows ({len(g_pal)}): {desc_(pa_b)}; the MFD bins: '
              + (f'each above zero' if len(g_mfd) else 'no limit')
              + ('; a start outside the limits is moved inside first' if len(pos_g) else ''))
    if a.positive:                                 # positivity: every data row with a positive target above zero
        if dirs is not None or lognormal is not None:
            raise SystemExit('--positive is not implemented for transfer moves or the hierarchical prior')
        try:
            T.set_positive(np.flatnonzero(P['d'] > 0))
        except ValueError as e:
            raise SystemExit(f'--positive: {e}')
        print(f'positivity: the predictions of {len(T.band["rows"])} of the {T.n_data} data rows (those with a positive '
              'target) kept above zero, at least one rupture on in each; a start that predicts 0 for some gets ruptures '
              'switched on first')
    if a.init == 'nnls':
        with np.load(fprelim) as pr0:
            if 'm_nnls' not in pr0.files:
                raise SystemExit(f'--init nnls: {fprelim} holds no best fit (cleaned by an older version?); clean again, or use '
                                 '--init zero')
            X0 = np.tile(pr0['m_nnls'], (a.chains, 1))
    elif a.init == 'fitted':                       # the best fit as fitted (best_fit_as_fitted.py), made from this NNLS fit
        from best_fit_as_fitted import nnls_key
        f_fit = os.path.join(os.path.dirname(os.path.abspath(a.problem)), 'nnls', 'fitted_vectors.npz')
        if not os.path.exists(f_fit):
            raise SystemExit(f'--init fitted: no {f_fit}; make it with best_fit_as_fitted.py, or use --init nnls')
        with np.load(fprelim) as pr0, np.load(f_fit) as zf:
            if 'nnls_key' not in zf.files or str(zf['nnls_key']) != nnls_key(pr0['m_nnls']):
                raise SystemExit(f'--init fitted: {f_fit} was made from another NNLS fit (cleaned again since?); run '
                                 'best_fit_as_fitted.py again, or use --init nnls')
            sm_want = a.slip_misfit
            sm_have = str(zf['slip_misfit']) if 'slip_misfit' in zf.files else 'split'
            if sm_have != sm_want:
                raise SystemExit(f'--init fitted: {f_fit} was fitted with the slip misfit {sm_have}, this run uses {sm_want}; run '
                                 'best_fit_as_fitted.py again, or use --init nnls')
            if (str(zf['paleo_misfit']), str(zf['mfd_misfit'])) != (a.paleo_misfit, a.mfd_misfit):
                raise SystemExit(f'--init fitted: {f_fit} was fitted with --paleo-misfit {zf["paleo_misfit"]} and --mfd-misfit '
                                 f'{zf["mfd_misfit"]}, this run uses {a.paleo_misfit} and {a.mfd_misfit}; run '
                                 'best_fit_as_fitted.py again, or use --init nnls')
            X0 = np.tile(np.asarray(zf['mean'], np.float64)[np.asarray(P['keep_cols'])], (a.chains, 1))
            print(f'start: the best fit as fitted ({f_fit}): chi2 as fitted {float(zf["misfit"]):.1f}, against '
                  f'{float(zf["misfit_nnls"]):.1f} at the NNLS fit it started from; rates below the slab floor switched off')
            from best_fit_as_fitted import terms_key  # the sampler's other terms and constraints: a fit made without them
            have_ = str(zf['terms']) if 'terms' in zf.files else None   # is still a start (scaled inside below), so a note
            if have_ is not None and have_ != terms_key(a):
                print(f'   note: the fit was made with other terms and constraints ({have_}) than this run\'s ({terms_key(a)})')
    elif a.init == 'zero':
        X0 = np.zeros((a.chains, n))
    else:
        X0 = np.load(a.init)['final_X'] if a.init.endswith('.npz') else np.load(a.init)
        X0 = np.tile(X0, (a.chains, 1)) if X0.ndim == 1 else X0[np.arange(a.chains) % X0.shape[0]]
    if a.positive or len(pos_g):                   # a start predicting 0 for some rows: ruptures switched on in them
        Us = T.init(X0 if scale is None else X0 / scale)
        n_out = T.band_outside(T.residual(Us))
        if n_out.any():
            chi0 = float(np.sum(T.residual(Us[:1])[0, :T.n_data] ** 2))
            Us = np.array([positive_start(T, u)[0] for u in Us])
            added = positive_start(T, T.init(X0[:1] if scale is None else X0[:1] / scale)[0])[1]
            X0 = Us if scale is None else Us * scale
            chi1 = float(np.sum(T.residual(Us[:1])[0, :T.n_data] ** 2))
            print(f'positivity: the start predicts 0 for {n_out.max()} rows; {added} ruptures crossing them switched on at '
                  f'twice their slab floor (chi2 {chi0:.4f} -> {chi1:.4f})')
    if sm_rows is not None:                        # --slip-misfit: the log misfit, through the soft band's machinery
        from cru_common import geo_moments, slip_log_sds
        s_lo, s_hi, wk_ = slip_log_sds(P, sm_rows)[:3]   # each row's log10 sds, below and above: its effective sd's (1/k: a
        W, WL = np.zeros(T.N), np.zeros(T.N)           # section's copies count once)
        W[sm_rows] = gw('slip') * b_s * wk_ / (2.0 * (s_hi * LN10) ** 2)   # 2 W (ln y - ln d)^2 = b (1/k) (log10(y / d) / s)^2, each side's s
        WL[sm_rows] = gw('slip') * b_s * wk_ / (2.0 * (s_lo * LN10) ** 2)
        LD = np.zeros(T.N)
        LD[sm_rows] = np.log(T.d[sm_rows])
        T.lw = dict(W=np.ascontiguousarray(W), WL=np.ascontiguousarray(WL), LD=np.ascontiguousarray(LD),
                    DY=np.ascontiguousarray(T.d, np.float64), rows=np.array([], np.int64), tau=None,
                    SM=dict(rows=sm_rows, s_lo=s_lo, s_hi=s_hi, wk=wk_, b=b_s, eps=lin_sc(b_s), d_data=d_data))
        T.lw['SM']['E'], T.lw['SM']['V'], n_s_, T.lw['SM']['t2'], T.lw['SM']['t4'] = geo_moments(P, sm_rows, b_s, 'slip')
        y0 = (T.residual(T.init(X0[:1] if scale is None else X0[:1] / scale)) + T.d)[0, sm_rows]
        uq0 = np.log10(np.maximum(y0, 1e-300) / T.d[sm_rows])
        f0 = float(np.sum(wk_ * (uq0 / np.where(uq0 < 0, s_lo, s_hi)) ** 2))
        s_all = np.concatenate([s_lo, s_hi])
        s_med = float(np.median(s_all))
        print(f'slip misfit ({a.slip_misfit}): the {len(sm_rows)} slip-rate rows ({n_s_} sections) scored by '
              + (f'half their split normal plus half ' if b_s < 1 else '')
              + f'a Gaussian in log10(prediction / target) with the same sd, s = log10(1 + sd / target) on each side (sd the '
              f'side\'s effective sd, the template weights\' included, as the MFD bins\'): {s_all.min():.3f}-{s_all.max():.3f} '
              f'decades, median {s_med:.3f} (a row at the median: 10x off costs {b_s / s_med ** 2:.0f} on chi2\'s scale from its log '
              f'part), kept by a Metropolis step; the start\'s log part {f0:.1f} (weighted {b_s:g})')
    if pm_rows is not None:                        # --paleo-misfit: the paleo rows' tables, through the soft band's machinery
        if T.lw is None:                           # (its weights 0 without --log-band)
            T.lw = dict(W=np.zeros(T.N), LD=np.zeros(T.N), DY=np.ascontiguousarray(T.d, np.float64),
                        rows=np.array([], np.int64), tau=None)
        I_p = np.full(T.N, -1, np.int32)
        I_p[pm_rows] = np.arange(len(pm_rows), dtype=np.int32)
        sc_p = lin_sc(b_p)                         # the tables carry b: b x the curve's half misfit
        T.lw['PM'] = dict(I=I_p, V0=np.ascontiguousarray(pm_tab['U0'] - np.log(sd_p / sc_p)),   # in v = ln(row's y)
                          F=np.ascontiguousarray(b_p * gw('paleo') * pm_tab['F']),
                          D=np.ascontiguousarray(b_p * gw('paleo') * pm_tab['DF']),
                          K=np.ascontiguousarray(b_p * gw('paleo') * pm_tab['KAPPA']), H=pm_tab['H'], NG=pm_tab['NG'], rows=pm_rows,
                          eps=sc_p, d_data=d_data, mode=a.paleo_misfit, b=b_p, w=gw('paleo'))
        from paleo_misfit import site_moments_mixed     # each site's curve, and its misfit's mean and variance (fit check)
        T.lw['PM']['sd'] = np.asarray(sd_p, float)
        T.lw['PM']['th'] = np.array([f_[0] for f_ in pm_tab['fits']], float)
        T.lw['PM']['best'] = np.array([s_['best'] for s_ in sites], float)
        mom_ = np.array([site_moments_mixed(th_, b_, sd_, a.paleo_misfit, b_p)
                         for th_, b_, sd_ in zip(T.lw['PM']['th'], T.lw['PM']['best'], T.lw['PM']['sd'])])
        T.lw['PM']['E'], T.lw['PM']['V'] = mom_[:, 0], mom_[:, 1]
        y0 = (T.residual(T.init(X0[:1] if scale is None else X0[:1] / scale)) + T.d)[0, pm_rows]
        f0 = 2.0 * float(np.sum(_pm_f_np(T.lw['PM'], np.arange(len(pm_rows)), np.log(np.maximum(y0, 1e-300))))) / b_p / gw('paleo')
        print(f'paleo misfit ({a.paleo_misfit}): {len(pm_rows)} paleo rows '
              + (f'out of the Gaussian (scaled by {sc_p:g}), ' if b_p >= 1 else 'with half their Gaussian in rate (scaled by 1/sqrt 2) and half ')
              + 'each scored by -2 ln of ' + ('its recurrence-interval PDF in years' if curve_mode(a.paleo_misfit) == 'years'
                                             else 'its curve\'s density in log rate')
              + f' from its minimum, kept by a Metropolis step; the start\'s curve misfit {f0:.1f} (on chi2\'s scale, weighted {b_p:g})')
    if mf_rows is not None and a.mfd_misfit == 'pdf':   # MFD_GR: the bins' PDFs as curves, appended to the paleo curves' tables
        if not (grp or a.positive):
            raise SystemExit('--mfd-misfit pdf needs each bin kept above zero (the per-group limits, or --positive)')
        from paleo_misfit import curve_tables, site_moments
        if T.lw is None:
            T.lw = dict(W=np.zeros(T.N), LD=np.zeros(T.N), DY=np.ascontiguousarray(T.d, np.float64),
                        rows=np.array([], np.int64), tau=None)
        th_b = np.asarray(P['mfd_pdf_th'], float)[mf_rows]
        best_b = np.asarray(P['mfd_pdf_best'], float)[mf_rows]
        sd_b = bin_targets(P, a.bins_block)[0] / np.asarray(P['d'], float)[mf_rows]   # each bin row's sigma (R_j / d_j)
        tb_ = curve_tables(th_b, best_b, 'log')
        sc_m = lin_sc(b_m)
        V0_b = tb_['U0'] - np.log(sd_b / sc_m)        # in v = ln(row's y), the row scaled by sc_m
        PM_ = T.lw.get('PM')
        if PM_ is None:                            # the bins alone (the paleo rows Gaussian)
            I_m = np.full(T.N, -1, np.int32)
            n_p0 = 0
            T.lw['PM'] = dict(I=I_m, V0=np.ascontiguousarray(V0_b), F=np.ascontiguousarray(gw('mfd') * tb_['F']),
                              D=np.ascontiguousarray(gw('mfd') * tb_['DF']), K=np.ascontiguousarray(gw('mfd') * tb_['KAPPA']), H=tb_['H'],
                              NG=tb_['NG'], rows=np.array([], np.int64), eps=1.0, d_data=d_data, mode='gaussian', b=0.0)
        else:                                      # after the paleo sites' tables, with weight 1 (theirs carry b_p)
            if PM_['NG'] != tb_['NG'] or PM_['H'] != tb_['H']:
                raise SystemExit('--mfd-misfit pdf: the bins\' tables and the paleo tables differ in their grid')
            n_p0 = len(PM_['V0'])
            PM_['V0'] = np.ascontiguousarray(np.concatenate([PM_['V0'], V0_b]))
            PM_['F'] = np.ascontiguousarray(np.concatenate([PM_['F'], gw('mfd') * tb_['F']]))
            PM_['D'] = np.ascontiguousarray(np.concatenate([PM_['D'], gw('mfd') * tb_['DF']]))
            PM_['K'] = np.ascontiguousarray(np.concatenate([PM_['K'], gw('mfd') * tb_['KAPPA']]))
        idx_b = n_p0 + np.arange(len(mf_rows))
        T.lw['PM']['I'][mf_rows] = idx_b.astype(np.int32)
        mom_ = np.array([site_moments(th_, b_, 'log') for th_, b_ in zip(th_b, best_b)])
        T.lw['MF'] = dict(rows=mf_rows, pdf=True, idx=idx_b, th=th_b, best=best_b, sd=sd_b, E=mom_[:, 0], V=mom_[:, 1],
                          eps=sc_m, d_data=d_data, b=b_m, w=gw('mfd'))
        y0 = (T.residual(T.init(X0[:1] if scale is None else X0[:1] / scale)) + T.d)[0, mf_rows]
        f0 = 2.0 * float(np.sum(_pm_f_np(T.lw['PM'], idx_b, np.log(np.maximum(y0, 1e-300))))) / gw('mfd')
        print(f'MFD misfit (pdf, MFD_GR {P.get("mfd_gr")}): the {len(mf_rows)} bin rows out of the Gaussian (scaled by {sc_m:g}), each '
              'scored by -2 ln of its own target PDF in log rate from its minimum (curves from cleaning, tabulated with the paleo '
              f'sites\'), kept by a Metropolis step; the start\'s {f0:.1f} (on chi2\'s scale)')
    elif mf_rows is not None:                      # --mfd-misfit log10: the bins' log misfit, through the soft band's machinery
        if not (grp or a.positive):
            raise SystemExit(f'--mfd-misfit {a.mfd_misfit} needs each bin kept above zero (the per-group limits, or --positive)')
        if T.lw is None:
            T.lw = dict(W=np.zeros(T.N), LD=np.zeros(T.N), DY=np.ascontiguousarray(T.d, np.float64),
                        rows=np.array([], np.int64), tau=None)
        T.lw['W'] = np.array(T.lw['W'], np.float64)
        T.lw['LD'] = np.array(T.lw['LD'], np.float64)
        T.lw['W'][mf_rows] = gw('mfd') * b_m / (2.0 * (mf_s * LN10) ** 2)   # 1/2 (ln(y / R) / (s ln 10))^2 = 1/2 (log10(y / R) / s)^2, x b
        T.lw['LD'][mf_rows] = np.log(T.d[mf_rows])
        T.lw['W'], T.lw['LD'] = np.ascontiguousarray(T.lw['W']), np.ascontiguousarray(T.lw['LD'])
        if T.lw.get('WL') is not None:             # with --slip-misfit split+log10's sided weights: the bins' the same both sides
            T.lw['WL'] = np.array(T.lw['WL'], np.float64)
            T.lw['WL'][mf_rows] = T.lw['W'][mf_rows]
        T.lw['MF'] = dict(rows=mf_rows, s=mf_s, eps=lin_sc(b_m), d_data=d_data, b=b_m)
        y0 = (T.residual(T.init(X0[:1] if scale is None else X0[:1] / scale)) + T.d)[0, mf_rows]
        f0 = float(np.sum((np.log10(np.maximum(y0, 1e-300) / T.d[mf_rows]) / mf_s) ** 2))
        print(f'MFD misfit ({a.mfd_misfit}): {len(mf_rows)} bin rows '
              + (f'out of the Gaussian (scaled by {lin_sc(b_m):g}), ' if b_m >= 1 else 'with half their Gaussian (scaled by 1/sqrt 2) and half ')
              + f'scored by (log10(total / R_j) / s_j)^2 with s_j = log10(1 + sigma_j / R_j) ({mf_s.min():.3f}-{mf_s.max():.3f} decades), '
              f'kept by a Metropolis step; the start\'s {f0:.1f} (on chi2\'s scale, weighted {b_m:g})')
    if mono is not None and mono['pairs']:         # the MFD's monotonicity: constraint rows kept at or above 0
        if dirs is not None or lognormal is not None:
            raise SystemExit('--mfd-monotonic is not implemented for transfer moves or the hierarchical prior')
        T.add_limits(mono['rows'], 0.0, np.inf, mono['eps'] * 1e-15, 'the MFD\'s monotonicity')
        X0 = np.array(X0, np.float64)
        fixed = 0
        for _round in range(5):                    # the prior's support first (rates below the slab floor are off, so
            off_ = (T.init(X0 if scale is None else X0 / scale) == 0) & (X0 > 0)   # a bin's total may drop; 2026-11-23:
            X0[off_] = 0.0                         # a start with 7 such rates broke the M8.4 pair after the snap), then
            fixed_r = 0                            # the repair, until neither changes the start
            for x in X0:                           # a start that breaks it: each offending bin scaled down to its neighbour
                for _ in range(len(mono['pairs']) + 1):
                    tot = np.array([x[m].sum() for m in mono['members']])
                    bad = [(k, k1, rho) for (k, k1), rho in zip(mono['pairs'], mono['rhos']) if tot[k1] > rho * tot[k]]
                    if not bad:
                        break
                    k, k1, rho = bad[0]
                    x[mono['members'][k1]] *= rho * tot[k] / tot[k1] * (1 - 1e-9)
                    fixed_r += 1
            fixed += fixed_r
            if not fixed_r and not off_.any():
                break
        Us = T.init(X0 if scale is None else X0 / scale)
        left = T.band_outside(T.residual(Us))
        if fixed:
            print(f'MFD monotonic: the start breaks it; bins scaled down to their lower neighbour {fixed} times over the '
                  f'{len(X0)} chains')
        if left.any():
            raise SystemExit(f'MFD monotonic: the start leaves {left.max()} rows outside {T.band["what"]}; the constraints '
                             'together may need a start made for them')
    if plate is not None:                          # the plate rate: each strip's share kept within [lo, hi]
        if dirs is not None or lognormal is not None:
            raise SystemExit('--plate-rate is not implemented for transfer moves or the hierarchical prior')
        eps_p, lo_p, hi_p = plate['eps'], (-np.inf if plate['lo'] is None else plate['lo']), plate['hi']
        T.add_limits(plate['rows'], eps_p * lo_p, eps_p * hi_p, eps_p * 1e-9,
                     f'the plate rate ({plate["what"]} of it in each strip)')
        Us = T.init(X0 if scale is None else X0 / scale)
        share = (plate['A'] @ (Us if scale is None else Us * scale).T).T          # chains x strips
        j = int(np.argmax(share[0]))
        print(f'plate rate: the start\'s shares {share[0].min():.3f} to {share[0].max():.3f} (the highest in strip '
              f'{int(plate["k"][j]):+d})' + ('' if np.allclose(share, share[0]) else f' (chain 1; {len(share)} chains differ)'))
        top = share.max(1)
        if (top > hi_p).any() and lo_p <= 0:       # a start above HI: each chain's rates scaled down to it, which keeps
            X0 = np.array(X0, np.float64)          # positivity, the MFD's monotonicity (a ratio) and any LO <= 0
            for c in np.flatnonzero(top > hi_p):
                X0[c] *= hi_p / top[c] * (1 - 1e-9)
            Us = T.init(X0 if scale is None else X0 / scale)
            share = (plate['A'] @ (Us if scale is None else Us * scale).T).T
            print(f'plate rate: the start is above {hi_p:g} in {int((top > hi_p).sum())} of {len(top)} chains; their '
                  f'rates scaled down by {(hi_p / top[top > hi_p]).min():.4f} at most')
        out_p = (share < lo_p) | (share > hi_p)
        if out_p.any():
            raise SystemExit(f'plate rate: the start leaves {int(out_p.sum(1).max())} strips outside '
                             f'{plate["what"]}; start from a model inside the limits (--init FILE), or widen them')
        left = T.band_outside(T.residual(Us))
        if left.any():
            raise SystemExit(f'plate rate: the start leaves {left.max()} rows outside {T.band["what"]}')
    if a.mfd_gr_tau is not None:                   # MFD_GR_TAU: the bins scored jointly about each draw's G-R curve
        if a.mfd_step_sd is not None:
            raise SystemExit('--mfd-gr-tau replaces --mfd-step-sd: give one')
        if mf_rows is None or a.mfd_misfit != 'pdf':
            raise SystemExit('--mfd-gr-tau needs --mfd-misfit pdf (MFD_GR: a problem cleaned with preliminaries.py --mfd-gr)')
        if dirs is not None or lognormal is not None:
            raise SystemExit('--mfd-gr-tau is not implemented for transfer moves or the hierarchical prior')
        if a.backend != 'c':
            raise SystemExit('--mfd-gr-tau is implemented in the C kernel only (--backend c)')
        if not a.mfd_gr_tau > 0:
            raise SystemExit(f'--mfd-gr-tau {a.mfd_gr_tau:g}: give a tolerance above 0 (decades a step)')
        from cru_common import magnitude_bins
        rows_ms, R_ms, orig_ms, _ = magnitude_bins(P, a.bins_block, a.lowest_bin, a.bin_width)
        T.mstep = mjoint_build(T, P, rows_ms, orig_ms, a.mfd_gr_tau, a.bin_width, a.chains, w=gw('mfd'))
        T.lw['PM']['I'][mf_rows] = -1              # the bins' own PDFs out of the likelihood: the curve scores them
        MF_ = T.lw['MF']
        MF_['joint'], MF_['E'], MF_['V'] = True, np.ones(len(mf_rows)), np.full(len(mf_rows), 2.0)   # S ~ chi2 with NB dof
        Us = T.init(X0 if scale is None else X0 / scale)
        mjoint_update(T.mstep, T.residual(Us), np.random.default_rng(a.seed + 7))
        s0 = mjoint_value(T.mstep, T.residual(Us))
        ms_ = T.mstep
        kn_ = ms_['knobs']
        print(f'MFD bins scored jointly (--mfd-gr-tau {a.mfd_gr_tau:g}, MFD_GR {P.get("mfd_gr")}): each chain draws b (prior '
              f'{kn_["b"][0]:g}/{kn_["b"][1]:g}/{kn_["b"][2]:g}, 10/50/90%) and the level ln(s m) (s the N scaling, 2-sigma '
              f'{kn_["s"][0]:g}-{kn_["s"][1]:g}; m ' + ('none' if kn_['m'] is None else '/'.join(f'{v:g}' for v in kn_['m']))
              + f') each sweep, exactly on grids of {len(ms_["bgrid"])} x {len(ms_["lgrid"])}; the {len(rows_ms)} bins score '
              f'against the draw\'s G-R curve (NZSHM22\'s partitioning), each step within {a.mfd_gr_tau:g} decades of the '
              f'curve\'s ({100 * (10 ** a.mfd_gr_tau - 1):.1f}%) and the mean level within {a.mfd_gr_tau:g}, in place of each bin\'s '
              f'own PDF (the MFD misfit, in chi2 as fitted; near {len(rows_ms)} if the bins scatter so); the start\'s '
              f'{np.median(s0):.1f} (median over the chains), b {np.median(ms_["theta"][:, 0]):.3f}, s m '
              f'{ms_["m_c"] * np.exp(np.median(ms_["theta"][:, 1])):.3f}')
    if a.mfd_step_sd is not None:                  # the MFD's steps held to the target's: a soft term, recorded apart
        if dirs is not None or lognormal is not None:
            raise SystemExit('--mfd-step-sd is not implemented for transfer moves or the hierarchical prior')
        if a.backend != 'c':
            raise SystemExit('--mfd-step-sd is implemented in the C kernel only (--backend c)')
        if not a.mfd_step_sd > 0:
            raise SystemExit(f'--mfd-step-sd {a.mfd_step_sd:g}: give a tolerance above 0 (in b-value units)')
        from cru_common import magnitude_bins
        rows_ms, R_ms, orig_ms, _ = magnitude_bins(P, a.bins_block, a.lowest_bin, a.bin_width)
        T.mstep = mstep_build(T, rows_ms, orig_ms, a.mfd_step_sd, a.bin_width)
        if GW is not None:                         # DATATYPES_EQUAL: the steps' term with the bins' weight (recorded without)
            T.mstep['A'], T.mstep['c'] = np.ascontiguousarray(T.mstep['A'] * gw('mfd')), np.ascontiguousarray(T.mstep['c'] * gw('mfd'))
        Us = T.init(X0 if scale is None else X0 / scale)
        s0 = mstep_value(T.mstep, T.residual(Us))
        if not np.isfinite(s0).all():
            raise SystemExit('--mfd-step-sd: the start leaves a magnitude bin at 0; every bin must hold a rupture that is on')
        ms_ = T.mstep
        print(f'MFD steps: each of the {len(ms_["pairs"])} pairs of neighbouring bins\' step in log10 total, less the target\'s '
              f'own step there, held to the draw\'s common tilt within {a.mfd_step_sd:g} in b ({ms_["tau"] / LN10:.4f} decades a '
              f'step, a ratio off by about {100 * (10 ** (ms_["tau"] / LN10) - 1):.1f}%), a soft term recorded apart from chi2 '
              f'(near {len(ms_["pairs"]) - 1} if the steps scatter so); the target rates\' own steps '
              + ', '.join(f'{np.log10(R_ms[b_] / R_ms[a_]):+.3f}' for a_, b_ in ms_['pairs'])
              + f' decades; the start\'s {s0[0]:.1f}'
              + ('' if np.allclose(s0, s0[0]) else f' (chain 1; {s0.min():.1f}-{s0.max():.1f} over the chains)'))
    if a.split_merge:                              # --split-merge: the triples, from the rupture set
        if a.split_merge < 0 or not a.split_merge_scale > 0:
            raise SystemExit(f'--split-merge {a.split_merge} --split-merge-scale {a.split_merge_scale:g}: give a number of '
                             'proposals and a positive scale')
        if not spike_mode or a.fractions_on != 'drawn':
            raise SystemExit('--split-merge needs spike-and-slab with the fractions on drawn (--fractions-on drawn)')
        if not a.rupture_set:
            raise SystemExit('--split-merge needs --rupture-set (the ruptures\' subsections and average slips)')
        from cru_common import find_rupture_set
        from split_merge import check_triples, describe, find_triples
        rd_sm = find_rupture_set(a.rupture_set, search=False)
        if rd_sm is None:
            raise SystemExit(f'--rupture-set {a.rupture_set}: no OpenSHA rupture set there (indices.csv, or ruptures/indices.csv)')
        t_sm = time.time()
        TA_, TB_, TC_, al_, be_ = find_triples(rd_sm, P['keep_cols'])
        if not len(TA_):
            raise SystemExit('--split-merge: no rupture is two others laid end to end')
        sr_ = np.flatnonzero(np.asarray(P['block']) == list(P['block_names']).index(a.slip_block))
        q_ = np.sort(np.random.default_rng(0).choice(len(TA_), min(len(TA_), 100_000), replace=False))
        rel_ = check_triples(P['G'], sr_, TA_[q_], TB_[q_], TC_[q_], al_[q_], be_[q_])   # a sample of them
        sc_ = np.ones(T.n) if scale is None else np.asarray(scale, np.float64)
        set_split_merge(T, TA_, TB_, TC_, np.log(al_ * sc_[TA_] / sc_[TB_]), np.log(be_ * sc_[TA_] / sc_[TC_]),
                        a.split_merge, a.split_merge_scale)
        print(f'split/merge moves: {a.split_merge} proposals per chain after each sweep (sd scale {a.split_merge_scale:g}); '
              + describe(TA_, TB_, T.n, rel_, 1e-9).replace('unchanged by each', f'unchanged by each (a sample of {len(q_):,})')
              + f'; {time.time() - t_sm:.0f} s')
        if (rel_ > 1e-9).any():
            print('   (the slip-rate rows are not unchanged by some: slip not uniform along the ruptures? The moves stay '
                  'exact, but fewer are accepted)')
    mon = []
    if os.path.exists(fprelim):
        pr = np.load(fprelim)
        mon = ([int(np.argmax(pr['m_nnls']))] if 'm_nnls' in pr.files else []) + [int(np.argmin(pr['cn'])), int(np.argmax(pr['cn']))]
        if len(pr['dup_cos']):
            k = int(np.argmax(pr['dup_cos']))
            mon += [int(pr['dup_i'][k]), int(pr['dup_j'][k])]
    fe_ = P.get('chi2_expect')                     # the fit check as the likelihood scores the data (chi2 as fitted)
    if fe_ is not None:
        m_, v_ = float(fe_[0]), float(fe_[1]) ** 2
        if pm_rows is not None:                    # each paleo row's misfit's own mean and variance, for the Gaussian's 1 and 2
            m_ += float(np.sum(T.lw['PM']['E'])) - len(pm_rows)
            v_ += float(np.sum(T.lw['PM']['V'])) - 2.0 * len(pm_rows)
        if sm_rows is not None:                    # each slip section's curve: its own mean and variance for t^2 and 2 t^4
            SM_ = T.lw['SM']
            m_ += SM_['E'] - SM_['t2']
            v_ += SM_['V'] - 2.0 * SM_['t4']
        parts_ = []
        if getattr(T, 'split', None) is not None and 'exact_moments' in T.split:   # --slip-misfit exact: each section's own
            Ee_, Ve_, t2_, t4_ = T.split['exact_moments']                            # mean and variance, for t^2 and 2 t^4
            m_ += Ee_ - t2_
            v_ += Ve_ - 2.0 * t4_
            parts_.append(f'the slip sections\' {Ee_:.1f} +- {np.sqrt(Ve_):.1f} in place of {t2_:.1f} +- {np.sqrt(2 * t4_):.1f}')
        if mf_rows is not None and a.mfd_misfit == 'pdf':   # MFD_GR: each bin's misfit's own mean and variance under its PDF
            MF_ = T.lw['MF']
            m_ += float(np.sum(MF_['E'])) - len(mf_rows)
            v_ += float(np.sum(MF_['V'])) - 2.0 * len(mf_rows)
            parts_.append(f'the bins\' {np.sum(MF_["E"]):.1f} +- {np.sqrt(np.sum(MF_["V"])):.1f} in place of {len(mf_rows)} +- '
                          f'{np.sqrt(2 * len(mf_rows)):.1f}')
        if mf_rows is not None and b_m < 1:        # the bins half and half: each bin's curve (in log10 alone: 1 and 2, kept)
            from cru_common import geo_moments
            Em_, Vm_ = geo_moments(P, mf_rows, b_m, 'mfd')[:2]
            T.lw['MF']['E'], T.lw['MF']['V'] = Em_, Vm_  # recorded, for block_fit.py
            m_ += Em_ - len(mf_rows)
            v_ += Vm_ - 2.0 * len(mf_rows)
            parts_.append(f'the bins\' {Em_:.1f} +- {np.sqrt(Vm_):.1f} in place of {len(mf_rows)} +- {np.sqrt(2 * len(mf_rows)):.1f}')
        if sm_rows is not None:
            SM_ = T.lw['SM']
            parts_.append(f'the slip sections\' {SM_["E"]:.1f} +- {np.sqrt(SM_["V"]):.1f} in place of {SM_["t2"]:.1f} +- '
                          f'{np.sqrt(2 * SM_["t4"]):.1f}')
        if pm_rows is not None:
            parts_.append(f'the paleo sites\' {np.sum(T.lw["PM"]["E"]):.1f} +- {np.sqrt(np.sum(T.lw["PM"]["V"])):.1f} in place of '
                          f'{len(pm_rows)} +- {np.sqrt(2 * len(pm_rows)):.1f}')
        T.fit_expect = (m_, float(np.sqrt(max(v_, 0.0))))   # the MFD bins in log10 alone keep 1 and 2 each: Gaussian in log10
        if parts_:
            print(f'fit check as fitted: a draw\'s chi2 as the likelihood scores it should be near {m_:.0f} +- {np.sqrt(v_):.0f} if the '
                  f'uncertainties are right (each group\'s misfit\'s own mean and variance: ' + '; '.join(parts_)
                  + f'; cleaning\'s, for the Gaussians: {float(fe_[0]):.0f} +- {float(fe_[1]):.0f})')
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
                keep_draws=a.keep_draws, split_merge=bool(a.split_merge))
    smt_ = tr['split_merge'].sum(0) if 'split_merge' in tr else None
    print(f'done in {tr["seconds"]:.0f}s -> {a.out}.npz, {a.out}_stream.npz' + ('' if a.no_samples else f', {a.out}_samples.npy')
          + (f', {a.out}_kept.npz (and _kept_index.npy)' if a.keep_draws else '')
          + (f'; swap acceptance {tr["swap_rate"].mean():.3f}' if pairs is not None else '')
          + (f'; split/merge: {smt_[1]:,} of {smt_[0]:,} splits and {smt_[3]:,} of {smt_[2]:,} merges tried accepted, '
             f'of {a.split_merge * a.sweeps * a.chains:,} proposals' if smt_ is not None else '')
          + (f'; transfer moves {tr["transfer_stats"][:, 0].sum()}, of which involving a rate > 1e-6: '
             f'{tr["transfer_stats"][:, 1].sum()}' if dirs is not None else ''))
    report(tr, a.burn, a.thin)


if __name__ == '__main__':
    main()

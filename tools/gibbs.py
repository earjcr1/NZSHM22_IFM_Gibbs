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
rates is treated. --prior binbox-theta:1:0 (per-bin log-uniform boxes whose depths put no pressure on the bin totals,
theta = 1; the alternative until October 2026) is kept for comparison. --prior linear is an optional linear-space prior:
each rupture's rate uniform from 0 to its bin's target rate, drawn by the exact truncated-normal updates above.
--down-the-middle adds a balance term to the misfit: each data type's relative shortfall s (summed predictions against
summed targets) costs A s^2, one A for every type (or, with --down-the-middle zero-model, each type's own, its zero-model
misfit), as extra rows of G and d, for any prior. --band F (F = 10, say; the notebook's BAND) keeps every data row with a
positive target predicted within a factor F of it, from d/F to F d: a hard constraint, for any prior but the retired
hierarchical one and without transfer moves. Each update is drawn from its conditional restricted to the rates that keep
every row of its column inside the band (band_bounds in the kernel), spike-and-slab never takes an off or on state the
band forbids, and a swap that would leave it is rejected. A start outside the band is first moved inside it by a
penalized least-squares fit (band_start), kept beside the problem as <problem>_band<F>_start.npz and reused by later
runs with the same problem, band, start and bounds. Each recorded draw's count of rows outside the band, always 0, is
kept as band_outside. --positive (the notebook's default) keeps every such row above zero instead, through the same
machinery, and --mfd-monotonic and --plate-rate add constraint rows of their own outside the likelihood (eps x the
quantity kept, d = 0, eps = 1e-9, so they add nothing measurable to chi2): no magnitude bin's total above its lower
neighbour's where the targets do not rise, and each strip's share of the Pacific-Australia plate rate within limits
(plate_rate.py builds those rows from the rupture set's geometry and slips, on the strips of cross-sections
perpendicular to the plate motion that --slip-sum uses too, --cross-sections). Two soft terms can join the misfit
beside the data, each recorded apart from chi2 on chi2's scale: --slip-sum, the slip along the plate motion summed
along each cross-section perpendicular to it against the targets' sums (rows of G and d, so the kernel is unchanged),
and --log-band, a soft band, a Gaussian penalty on each prediction's log ratio to its target, which is not quadratic
in the rates and is kept by Metropolis steps within the kernel's updates. See --help for the rest.

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
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # numpy's BLAS idle threads spin-wait otherwise (no BLAS-heavy work
                                                     # here; the kernel has its own threads)
import numpy as np
import scipy.sparse as sp

from cru_common import load_problem, column_norms, prior_rates, split_rhat, ess, summarize_samples, resolve_logbox, bin_targets, logbox_mean, prelim_path, report_to, balance_rows, magnitude_bins
import scipy.sparse as _sparse
sp_csc = _sparse.csc_matrix

LN10 = float(np.log(10.0))

C_SRC = r"""
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

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
   1/2: uniform in sqrt m).  The halfway prior (mA > 0) mixes a linear and a log-uniform range, each normalized over the
   box and its edges: density mA exp(z - zhi) + mB in z, inside the box, times the same edges. */
typedef struct { int pk; double m0, mu, inv2s2, zmin, zmax, zlo, zhi, k, ku, lp0, muz, kz, gam, mA, mB;
                 /* --cfm slip rates (sp = 1): the likelihood exactly, -sum w(e) e^2 sk over the rupture's column, e = sr_j +
                    G_ji (m - sxi), w = SL2_j below its target (e < 0) and SH2_j above; sL0 its value at m0 */
                 int sp; const double *sg; const int32_t *si; int64_t sp0, sp1; const double *sr, *SL2, *SH2;
                 double sxi, sk, sL0; } lbctx;
/* w(e) e^2 with w = l below 0 and h above, without a branch (the sign of a residual is unpredictable):
   ((h + l) e^2 + (h - l) e |e|) / 2 */
static inline double sp_w2(double e, double l, double h) { return 0.5 * ((h + l) * e * e + (h - l) * e * fabs(e)); }
/* the split misfit's log likelihood at m (lbctx's sp fields): concave in m, a Gaussian on each side of each row's target */
static double sp_lik(const lbctx *c, double m) {
    const double dm = m - c->sxi;
    double acc = 0.0;
    for (int64_t p = c->sp0; p < c->sp1; p++) {
        const int32_t j = c->si[p];
        const double e = c->sr[j] + c->sg[p] * dm;
        acc += sp_w2(e, c->SL2[j], c->SH2[j]);
    }
    return -acc * c->sk;
}
static inline double lb_prior(const lbctx *c, double z) {
    if (c->pk == 2) return -(z - c->muz) * (z - c->muz) * c->kz;      /* hierarchical: log m ~ Normal(muz, 1/(2 kz)) */
    if (c->mA > 0.0) {
        if ((z < c->zlo && !(c->k > 0.0)) || (z > c->zhi && !(c->ku > 0.0))) return -INFINITY;
        double p = log(c->mA * exp(z - c->zhi) + c->mB);
        if (z < c->zlo) p -= (c->zlo - z) * (c->zlo - z) * c->k;
        if (z > c->zhi) p -= (z - c->zhi) * (z - c->zhi) * c->ku;
        return p;
    }
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
   log-box prior is 0, or the slab's tilt gam z, or the halfway prior's log density), or less.  There lb_ld is the
   likelihood term, a concave quadratic in m = e^z, plus the prior's, less lp0, so it is at least the quadratic's
   smaller value at the two ends of the range of m plus the tilt's at the two ends of [za, zb] (the halfway prior's at
   za: it rises with z).  The range is widened by 1e-12 (relative) and 1e-300
   (absolute), far more than any rounding of exp, and the bound lowered by 1e-9 of the terms' size, far more than the
   rounding of lb_ld itself (about 1e-15 of it): wherever the bound exceeds y, so does every evaluation of lb_ld. */
static double lb_floor(const lbctx *c, double za, double zb) {
    const double ma = fmax(exp(za) * (1.0 - 1e-12) - 1e-300, 0.0), mb = exp(zb) * (1.0 + 1e-12) + 1e-300;
    const double pm = c->mA > 0.0 ? log(c->mA * exp(za - c->zhi) + c->mB) : 0.0;
    if (c->sp) {                                      /* the split likelihood: concave in m, so at least its smaller end */
        const double la = sp_lik(c, ma), lb = sp_lik(c, mb);
        const double size = fabs(la) + fabs(lb) + fabs(c->sL0) + fabs(c->lp0) + fabs(c->gam) * fmax(fabs(za), fabs(zb)) + fabs(pm);
        return fmin(la, lb) - c->sL0 + (c->mA > 0.0 ? pm : fmin(c->gam * za, c->gam * zb)) - c->lp0 - 1e-9 * (size + 1.0);
    }
    const double fa = -(ma - c->m0) * (ma + c->m0 - 2.0 * c->mu) * c->inv2s2;
    const double fb = -(mb - c->m0) * (mb + c->m0 - 2.0 * c->mu) * c->inv2s2;
    const double size = (mb + fabs(c->m0)) * (mb + fabs(c->m0) + 2.0 * fabs(c->mu)) * c->inv2s2 + fabs(c->lp0)
                        + fabs(c->gam) * fmax(fabs(za), fabs(zb)) + fabs(pm);
    return fmin(fa, fb) + (c->mA > 0.0 ? pm : fmin(c->gam * za, c->gam * zb)) - c->lp0 - 1e-9 * (size + 1.0);
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
   used by spike-and-slab only, whose upper-edge width defaults to 0 without KU, with its tilt gam).  With MIX, the
   halfway prior's constants (mA, mB per rupture), from the masses of a linear (gam 1) and a log-uniform (gam 0) slab
   over the same range, mixed mixw : 1 - mixw. */
void kernel_consts(int64_t n, const double *lo, const double *hi, double zlo, double zhi, const double *ZLO,
                   const double *KU, double gam, double mixw, double *MIX, double *LLO, double *LHI, double *SC)
{
    for (int64_t i = 0; i < n; i++) {
        LLO[i] = lo[i] > 0.0 ? log(lo[i]) : -INFINITY;
        LHI[i] = log(hi[i]);
        lbctx c;
        c.sp = 0;
        c.zlo = ZLO ? ZLO[i] : zlo; c.zhi = zhi; c.ku = KU ? KU[i] : 0.0; c.zmax = LHI[i]; c.gam = gam;
        slab_consts(&c, SC + 4 * i);
        if (MIX) {
            double q1[4], q0[4];
            c.gam = 1.0; slab_consts(&c, q1);
            c.gam = 0.0; slab_consts(&c, q0);
            MIX[2 * i] = mixw / (q1[0] + q1[1]);
            MIX[2 * i + 1] = (1.0 - mixw) / (q0[0] + q0[1]);
        }
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
static double lw_change(const double *data, const int32_t *indices, int64_t p0, int64_t p1, const double *r,
                        const double *DY, const double *LW, const double *LWL, const double *LD, double d0, double d1,
                        const pmctx *PM)
{
    double acc = 0.0;
    int bad0 = 0;
    for (int64_t p = p0; p < p1; p++) {
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
                        const double *SL2, const double *SH2, double d0, double d1) {
    double acc = 0.0;
    for (int64_t p = p0; p < p1; p++) {
        const int32_t j = indices[p];
        const double e0 = r[j] + data[p] * d0, e1 = r[j] + data[p] * d1;
        acc += sp_w2(e1, SL2[j], SH2[j]) - sp_w2(e0, SL2[j], SH2[j]);
    }
    return acc;
}
/* --cfm slip rates: for each off chain q (pre[q]) of a group, the split misfit's change if it switched on at mq[q], DQ[q],
   in one pass over the column; with the soft band (A), also its bound sums s1, s2, s3 (col_dots_lw's). One function per
   group size, with the sums in registers (chains not off are given m = 0, so they add nothing to DQ). */
static void col_split1(const double *data, const int32_t *indices, int64_t p0, int64_t p1, double *const *r,
                       double *const *A, const int *pre, const double *mq, const double *SL2, const double *SH2, double *DQ,
                       double *s1, double *s2, double *s3, double *const *W0) {
    const double *r0 = r[0], m0 = pre[0] ? mq[0] : 0.0; double d0 = 0.0, a0 = 0.0, b0 = 0.0, c0 = 0.0;
    if (A) {
        const double *A0 = A[0];
        for (int64_t p = p0; p < p1; p++) {
            const double g = data[p], gg = g * g; const int32_t j = indices[p];
            const double l2 = SL2[j], h2 = SH2[j];
            { const double e1 = r0[j] + g * m0; d0 += sp_w2(e1, l2, h2) - W0[0][j]; }
            { const double *e = A0 + 4 * j; a0 += g * e[0]; b0 += gg * e[1]; c0 += gg * e[2]; }
        }
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
                       double *s1, double *s2, double *s3, double *const *W0) {
    const double *r0 = r[0], m0 = pre[0] ? mq[0] : 0.0; double d0 = 0.0, a0 = 0.0, b0 = 0.0, c0 = 0.0;
    const double *r1 = r[1], m1 = pre[1] ? mq[1] : 0.0; double d1 = 0.0, a1 = 0.0, b1 = 0.0, c1 = 0.0;
    if (A) {
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
                       double *s1, double *s2, double *s3, double *const *W0) {
    const double *r0 = r[0], m0 = pre[0] ? mq[0] : 0.0; double d0 = 0.0, a0 = 0.0, b0 = 0.0, c0 = 0.0;
    const double *r1 = r[1], m1 = pre[1] ? mq[1] : 0.0; double d1 = 0.0, a1 = 0.0, b1 = 0.0, c1 = 0.0;
    const double *r2 = r[2], m2 = pre[2] ? mq[2] : 0.0; double d2 = 0.0, a2 = 0.0, b2 = 0.0, c2 = 0.0;
    if (A) {
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
                       double *s1, double *s2, double *s3, double *const *W0) {
    const double *r0 = r[0], m0 = pre[0] ? mq[0] : 0.0; double d0 = 0.0, a0 = 0.0, b0 = 0.0, c0 = 0.0;
    const double *r1 = r[1], m1 = pre[1] ? mq[1] : 0.0; double d1 = 0.0, a1 = 0.0, b1 = 0.0, c1 = 0.0;
    const double *r2 = r[2], m2 = pre[2] ? mq[2] : 0.0; double d2 = 0.0, a2 = 0.0, b2 = 0.0, c2 = 0.0;
    const double *r3 = r[3], m3 = pre[3] ? mq[3] : 0.0; double d3 = 0.0, a3 = 0.0, b3 = 0.0, c3 = 0.0;
    if (A) {
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
static void col_split(int64_t kg, const double *data, const int32_t *indices, int64_t p0, int64_t p1, double *const *r,
                      double *const *A, const int *pre, const double *mq, const double *SL2, const double *SH2, double *DQ,
                      double *s1, double *s2, double *s3, double *const *W0) {
    for (int64_t q = 0; q < 4; q++) DQ[q] = s1[q] = s2[q] = s3[q] = 0.0;
    if (kg >= 4)      col_split4(data, indices, p0, p1, r, A, pre, mq, SL2, SH2, DQ, s1, s2, s3, W0);
    else if (kg == 3) col_split3(data, indices, p0, p1, r, A, pre, mq, SL2, SH2, DQ, s1, s2, s3, W0);
    else if (kg == 2) col_split2(data, indices, p0, p1, r, A, pre, mq, SL2, SH2, DQ, s1, s2, s3, W0);
    else              col_split1(data, indices, p0, p1, r, A, pre, mq, SL2, SH2, DQ, s1, s2, s3, W0);
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
   LLO, LHI, SC: per-rupture constants from kernel_consts.  gam: spike-and-slab's slab tilt (0: log-uniform).  MIX: the
   halfway prior's constants (pkind 1), or NULL.  RLO, RHI: the band (each row's residual kept within them), or NULL:
   each update is then drawn from its conditional restricted to the rates the band allows (band_bounds), and under
   spike-and-slab an "off" state the band forbids is never taken, nor an "on" one outside it. */
void gibbs_run(int64_t N, int64_t n, int64_t K, const int64_t *indptr, const int32_t *indices,
               const double *data, const double *inv_csq, const double *lo, const double *hi,
               const double *beta, double sigma, double *X, double *R, uint64_t *rng, int64_t nsweeps,
               int64_t pkind, double zlo, double zhi, double kedge, double slw,
               const double *MUZ, const double *KZ, const double *ZLO, const double *KU, double *SM, const double *LOD,
               const int64_t *SBIN, int64_t SB, const double *LLO, const double *LHI, const double *SC, int64_t *CNT,
               double gam, const double *MIX, const double *RLO, const double *RHI,
               const double *LW, const double *LWL, const double *LD, const double *DY, const pmctx *PM, const double *SL2,
               const double *SH2, const int64_t *LWP, const int64_t *LWQ)
{
    const double s2 = sigma * sigma;
    double *WC = NULL;                             /* --cfm slip rates: each chain's rows' current split-misfit terms,
                                                      sp_w2 of their residuals, so a column pass computes only the new ones */
    if (SL2) {
        WC = (double *)malloc(sizeof(double) * (size_t)K * (size_t)N);
        for (int64_t k = 0; k < K; k++)
            for (int64_t j = 0; j < N; j++) WC[k * N + j] = sp_w2(R[k * N + j], SL2[j], SH2[j]);
    }
    double *AC = NULL, *ODD = NULL;                /* --log-band: each chain's rows' bound coefficients (lw_coef), 4 a row,
                                                      and each chain's odds pi / (1 - pi) of a rupture on, per bin */
    if (LW) {
        AC = (double *)calloc((size_t)K * (size_t)N * 4, sizeof(double));
        ODD = (double *)malloc(sizeof(double) * (size_t)K * (size_t)SB);
        for (int64_t k = 0; k < K; k++) {
            for (int64_t j = 0; j < N; j++) {
                if (LW[j] > 0.0) lw_coef(AC + 4 * (k * N + j), LW[j], LWL ? LWL[j] : LW[j], R[k * N + j] + DY[j], LD[j]);
                if (PM && PM->I[j] >= 0) pm_coef(AC + 4 * (k * N + j), PM, PM->I[j], R[k * N + j] + DY[j]);
            }
            for (int64_t b = 0; b < SB; b++) ODD[k * SB + b] = exp(LOD[k * SB + b]);
        }
    }
    for (int64_t sw = 0; sw < nsweeps; sw++) {
      for (int64_t i = 0; i < n; i++) {
        const int64_t p0 = indptr[i], p1 = indptr[i + 1];
        const int empty = (p1 == p0 || inv_csq[i] <= 0.0);
        const double aa = empty ? 0.0 : 1.0 / inv_csq[i];
        for (int64_t k0 = 0; k0 < K; k0 += 4) {
            const int64_t kg = K - k0 < 4 ? K - k0 : 4;
            double dt[4] = {0.0, 0.0, 0.0, 0.0}, S1[4] = {0.0}, S2[4] = {0.0}, S3[4] = {0.0};
            double mpre[4] = {0.0}, DQ[4] = {0.0};
            int pre[4] = {0, 0, 0, 0};
            if (SL2 && pkind == 3 && !empty) {        /* --cfm slip rates: each off chain's slab draw first (its draws in
                                                         the same order as below), then its exact change in one pass */
                for (int64_t q = 0; q < kg; q++) {
                    const int64_t k = k0 + q;
                    if (X[k * n + i] != 0.0) continue;
                    lbctx c;
                    c.sp = 0;
                    c.pk = 1; c.muz = 0.0; c.kz = 0.0; c.gam = gam; c.mA = 0.0; c.mB = 0.0;
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
                if (AC && LWP) {              /* the bound sums over the weighted rows only (col_lwsums) */
                    col_split(kg, data, indices, p0, p1, rg, NULL, pre, mpre, SL2, SH2, DQ, S1, S2, S3, wg);
                    col_lwsums(kg, data, indices, LWQ, LWP[i], LWP[i + 1], ag, S1, S2, S3);
                } else
                    col_split(kg, data, indices, p0, p1, rg, AC ? ag : NULL, pre, mpre, SL2, SH2, DQ, S1, S2, S3, wg);
            } else if (!empty) {
                double *rg[4];
                for (int64_t q = 0; q < 4; q++) rg[q] = R + (k0 + (q < kg ? q : 0)) * N;
                if (AC) {                     /* --log-band: the dot products and the bound sums in one pass */
                    double *ag[4];
                    for (int64_t q = 0; q < 4; q++) ag[q] = AC + 4 * (k0 + (q < kg ? q : 0)) * N;
                    if (LWP) {                /* or, with few weighted rows, the sums over those alone */
                        col_dots(kg, data, indices, p0, p1, rg, dt);
                        col_lwsums(kg, data, indices, LWQ, LWP[i], LWP[i + 1], ag, S1, S2, S3);
                    } else
                        col_dots_lw(kg, data, indices, p0, p1, rg, ag, dt, S1, S2, S3);
                } else
                    col_dots(kg, data, indices, p0, p1, rg, dt);
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
                    c.pk = 1; c.muz = 0.0; c.kz = 0.0; c.gam = gam; c.mA = 0.0; c.mB = 0.0;
                    c.zlo = ZLO ? ZLO[i] : zlo; c.zhi = zhi; c.k = 0.0; c.ku = KU ? KU[i] : 0.0;
                    c.zmin = LLO[i];
                    c.zmax = LHI[i];
                    c.mu = 0.0; c.inv2s2 = 0.0;
                    double m;
                    if (xi > 0.0) {           /* on: exact slice update of the slab value under likelihood x slab */
                        const double z0 = log(xi);
                        c.m0 = xi; c.lp0 = lb_prior(&c, z0);
                        if (!empty && SL2) {      /* --cfm slip rates: the likelihood evaluated exactly */
                            c.sp = 1; c.sg = data; c.si = indices; c.sp0 = p0; c.sp1 = p1; c.sr = r; c.SL2 = SL2; c.SH2 = SH2;
                            c.sxi = xi; c.sk = 0.5 / s2; c.sL0 = sp_lik(&c, xi);
                        } else if (!empty) { c.mu = xi - dot / aa; c.inv2s2 = 0.5 * aa / s2; }
                        if (RLO) {            /* the band: the slice within the rates it allows (z0 always among them) */
                            if (blo > 0.0) c.zmin = fmin(fmax(c.zmin, log(blo)), z0);
                            c.zmax = fmax(fmin(c.zmax, log(bhi)), z0);
                        }
                        m = exp(lb_slice(s, &c, z0, slw));
                        /* --log-band: the slice step, reversible under the rest of the target, is a proposal that
                           the soft band's change accepts or not (Metropolis), so the update stays exact */
                        if (LW && !empty && m != xi) {
                            const double dp = lw_change(data, indices, p0, p1, r, DY, LW, LWL, LD, 0.0, m - xi, PM);
                            if (!(log(unif(s)) < -dp)) m = xi;
                        }
                    } else {                  /* off: a fresh draw from the slab, the pseudo-prior */
                        m = pre[q] ? mpre[q] : slab_draw_m(s, &c, SC + 4 * i);   /* --cfm slip rates: drawn above */
                    }
                    /* on or off given m: log odds = log(pi/(1-pi)) + log L(m) - log L(0); with the rupture switched off
                       the residual is r0 = r - x_i G_i, so G_i.r0 = dot - x_i ||G_i||^2 */
                    double lr = LOD[k * SB + SBIN[i]];   /* log odds of the fraction on in the rupture's group */
                    double xl = 0.0;          /* the likelihood's part of the log odds, lr - LOD */
                    if (!empty && SL2) {      /* --cfm slip rates: the split misfit's change, exactly */
                        xl = (pre[q] ? DQ[q] : sp_change(data, indices, p0, p1, r, SL2, SH2, -xi, m - xi)) / (2.0 * s2);
                        lr -= xl;
                    } else if (!empty) { const double d0 = dot - xi * aa; xl = (2.0 * m * d0 + m * m * aa) / (2.0 * s2); lr -= xl; }
                    if (LW && !empty) {
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
                            /* stays off, without logarithms: on needs u < 1 / (1 + exp(dp - lr)) <= exp(lr - dlo)
                               = odds exp(X), X = -xl - dlo, and exp(X) <= 1 + X + X^2/2 (X <= 0), 1 + X + X^2 (X <= 1) */
                            const double X = -xl - dlo;
                            if (X <= 1.0) {
                                const double E = X <= 0.0 ? 1.0 + X + 0.5 * X * X : 1.0 + X + X * X;
                                if (uu >= ODD[k * SB + SBIN[i]] * E * (1.0 + 1e-9)) on = 0;
                            }
                            if (on < 0) {
                                g = lr - (log(uu) - log1p(-uu)); have_g = 1;
                                if (dhi < g) on = 1; else if (dlo >= g) on = 0;
                            }
                        }
                        if (on < 0) {
                            if (!have_g) g = lr - (log(uu) - log1p(-uu));
                            on = lw_change(data, indices, p0, p1, r, DY, LW, LWL, LD, -xi, m - xi, PM) < g;
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
                    c.mA = MIX ? MIX[2 * i] : 0.0; c.mB = MIX ? MIX[2 * i + 1] : 0.0;
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
                    for (int64_t p = p0; p < p1; p++) r[indices[p]] += data[p] * dl;
                    if (WC)                   /* --cfm slip rates: the changed rows' cached terms */
                        for (int64_t p = p0; p < p1; p++) {
                            const int32_t j = indices[p];
                            WC[k * N + j] = sp_w2(r[j], SL2[j], SH2[j]);
                        }
                    if (AC)                   /* --log-band: the changed rows' bound coefficients (with LWP, the
                                                 weighted rows', the only ones with any) */
                        for (int64_t t = LWP ? LWP[i] : p0; t < (LWP ? LWP[i + 1] : p1); t++) {
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
                x[i] = xn;
            }
        }
      }
    }
    if (AC) { free(AC); free(ODD); }
    free(WC);
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
                      const pmctx *PM)
{
    int64_t p = indptr[i], q = indptr[j];
    const int64_t pe = indptr[i + 1], qe = indptr[j + 1];
    double acc = 0.0;
    while (p < pe || q < qe) {
        int32_t row; double dv;
        if (q >= qe || (p < pe && indices[p] < indices[q])) { row = indices[p]; dv = data[p] * dl; p++; }
        else if (p >= pe || indices[q] < indices[p]) { row = indices[q]; dv = -data[q] * dl; q++; }
        else { row = indices[p]; dv = data[p] * dl - data[q] * dl; p++; q++; }
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
void swap_run(int64_t N, int64_t n, int64_t K, const int64_t *indptr, const int32_t *indices, const double *data,
              const double *csq, int64_t npair, const int64_t *pi, const int64_t *pj, const double *gij,
              const double *beta, double sigma, double *X, double *R, uint64_t *rng, int64_t *nacc,
              const double *RLO, const double *RHI, const double *LW, const double *LWL, const double *LD, const double *DY,
              const pmctx *PM, const double *SL2, const double *SH2)
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
            const double dchi = SL2 ? sp_swap(indptr, indices, data, i, j, dl, r, SL2, SH2)
                                    : 2.0 * dl * (ci - cj) + dl * dl * (csq[i] + csq[j] - 2.0 * gij[q]);
            const double dlp = -dl * (beta[i] - beta[j]) - (LW ? lw_swap(indptr, indices, data, i, j, dl, r, DY, LW, LWL, LD, PM) : 0.0);
            if (log(unif(s)) < -0.5 * dchi / s2 + dlp && (!RLO || swap_in_band(indptr, indices, data, i, j, dl, r, RLO, RHI))) {
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
    L.gibbs_run.argtypes = [I] * 3 + [P] * 7 + [D] + [P] * 3 + [I, I, D, D, D, D] + [P, P, P, P, P, P, P, I] + [P, P, P, P, D, P, P, P] + [P] * 4 + [P] + [P, P] + [P, P]
    L.gibbs_run.restype = None
    L.kernel_consts.argtypes = [I, P, P, D, D, P, P, D, D, P, P, P, P]
    L.kernel_consts.restype = None
    L.swap_run.argtypes = [I] * 3 + [P] * 4 + [I] + [P] * 4 + [D] + [P] * 6 + [P] * 4 + [P] + [P, P]
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
        self.gam = 0.0                             # spike-and-slab: the slab's tilt, density m^(gam - 1) (1/2: uniform in sqrt m)
        self.mixw = 0.0                            # the halfway prior (pkind 1): the linear part's share (0: none)
        self.n_data = self.N                       # the data rows; any rows after them are the balance term's
        self.n_bal = None                          # the end of the balance term's rows (None: n_fit); --slip-sum's follow
        self.n_fit = self.N                        # the rows in the likelihood; any after them are constraint rows only
        self.balance = None                        # --down-the-middle: dict(A, names) of the balance rows
        self.slip_sum = None                       # --slip-sum: dict(expect) of the summed slip rates' rows
        self.lw = None                             # --log-band: dict(W, LD, DY): each row's weight, log target, target
        self.split = None                          # --cfm-sliprates (in the problem): dict(L2, H2), each row's weight
                                                   # below and above its target (1 on rows without a split normal)
        self.rlo = self.rhi = None                 # --band, --positive: each row's residual kept within [rlo, rhi]
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


def _sp_ptrs(T):
    """--cfm slip rates: the split misfit's weights below and above each row's target (1 on rows without it), or NULLs."""
    if getattr(T, 'split', None) is None:
        return None, None
    return _ptr(T.split['L2']), _ptr(T.split['H2'])


def _sp_w(T, rows, e):
    """--cfm slip rates: each residual's weight, below or above its row's target."""
    return np.where(e < 0, T.split['L2'][rows], T.split['H2'][rows])


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
    key = (id(T.lw['W']), id(T.lw.get('PM')))
    if getattr(T, '_lw_sub_key', None) != key:
        w = np.asarray(T.lw['W']) > 0
        if T.lw.get('PM') is not None:
            w = w | (np.asarray(T.lw['PM']['I']) >= 0)
        hit = w[T.indices]
        sub = None
        if hit.mean() < 0.5:
            pos = np.ascontiguousarray(np.flatnonzero(hit), np.int64)
            sub = (np.ascontiguousarray(np.searchsorted(pos, T.indptr), np.int64), pos)
        T._lw_sub, T._lw_sub_key = sub, key
    return (None, None) if T._lw_sub is None else (_ptr(T._lw_sub[0]), _ptr(T._lw_sub[1]))


def kernel_consts(lib, T):
    """The C kernel's per-rupture constants (log lo, log hi, the slab's, the halfway prior's), once per run, by the
    kernel's own code."""
    LLO, LHI, SC = np.empty(T.n), np.empty(T.n), np.empty(4 * T.n)
    MIX = np.empty(2 * T.n) if T.mixw > 0 else None
    lib.kernel_consts(T.n, _ptr(T.lo), _ptr(T.hi), T.zlo, T.zhi, None if T.zlo_vec is None else _ptr(T.zlo_vec),
                      None if T.ku_vec is None else _ptr(T.ku_vec), T.gam, T.mixw, None if MIX is None else _ptr(MIX),
                      _ptr(LLO), _ptr(LHI), _ptr(SC))
    return LLO, LHI, SC, MIX


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
                      None if KC[3] is None else _ptr(KC[3]), None if T.rlo is None else _ptr(T.rlo),
                      None if T.rhi is None else _ptr(T.rhi), *_lw_ptrs(T), _pm_ptr(T), *_sp_ptrs(T), *_lw_sub_ptrs(T))
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
                     ctypes.c_void_p(rng.ctypes.data + k0 * 32), ctypes.c_void_p(nacc.ctypes.data + k0 * 8),
                     None if T.rlo is None else _ptr(T.rlo), None if T.rhi is None else _ptr(T.rhi), *_lw_ptrs(T), _pm_ptr(T), *_sp_ptrs(T))
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
        if c.get('mA', 0.0) > 0:                         # the halfway prior: a linear and a log-uniform part
            ku = c.get('ku', c['k'])
            if (z < c['zlo'] and not c['k'] > 0) or (z > c['zhi'] and not ku > 0):
                return -np.inf
            p = np.log(c['mA'] * np.exp(z - c['zhi']) + c['mB'])
            if z < c['zlo']:
                p -= (c['zlo'] - z) ** 2 * c['k']
            if z > c['zhi']:
                p -= (z - c['zhi']) ** 2 * ku
            return p
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
        if T.mixw > 0:                                   # the halfway prior's constants, as kernel_consts computes them
            m1, m0_ = _slab_masses_py(c, 1.0), _slab_masses_py(c, 0.0)
            c.update(mA=T.mixw / (m1[0] + m1[1]), mB=(1.0 - T.mixw) / (m0_[0] + m0_[1]))
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
                return -float(np.sum(_sp_w(T, rows, e) * e * e)) / (2.0 * s2)
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
        lr = LOD[k, T.ss['bins'][i]]
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
            if getattr(T, 'split', None) is not None:     # --cfm slip rates: the split misfit's change, exactly
                rows_ = np.union1d(ri, rj)
                dv_ = np.zeros(len(rows_))
                np.add.at(dv_, np.searchsorted(rows_, ri), vi * dl)
                np.add.at(dv_, np.searchsorted(rows_, rj), -vj * dl)
                e0_ = R[k, rows_]; e1_ = e0_ + dv_
                dchi = float(np.sum(_sp_w(T, rows_, e1_) * e1_ ** 2 - _sp_w(T, rows_, e0_) * e0_ ** 2))
            else:
                dchi = 2 * dl * (R[k, ri] @ vi - R[k, rj] @ vj) + dl * dl * (T.csq[i] + T.csq[j] - 2 * gij[q])
            dlp = -dl * (T.beta[i] - T.beta[j])
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
    if T.split is not None and (transfers or T.pkind != 3):
        raise NotImplementedError('the split misfit (--cfm-sliprates) is implemented for spike-and-slab without transfer moves')
    if T.rlo is not None and (transfers or T.pkind == 2):
        raise NotImplementedError('the band is not implemented for transfer moves or the hierarchical prior')
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
    n_rec = sweeps // thin
    n_burn = burn // thin
    n_keep = n_rec - n_burn
    monitor = np.asarray(monitor, dtype=np.int64)
    tr = dict(chi2=np.zeros((K, n_rec)), sum_m=np.zeros((K, n_rec)), tscale=np.zeros((K, n_rec)),
              mon=np.zeros((K, n_rec, len(monitor))), y=np.zeros((K, n_keep, T.n_data), np.float32), sweep=np.zeros(n_rec, int))
    nd = T.n_data                                        # chi2 and predictions over the data rows only
    PMr = None if T.lw is None or T.lw.get('PM') is None else T.lw['PM']   # --paleo-misfit: its rows are scaled in G
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
    if MFr is not None:
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
    KC = kernel_consts(lib, T) if backend == 'c' else None   # the kernel's per-rupture constants, once per run
    t0 = time.time()
    he = thin if ((H is None and SS is None) or not hyper_every) else max(1, min(int(hyper_every), thin))
    for rec in range(n_rec):
        done = 0
        while done < thin:                                   # hierarchical prior: hyperparameters every `he` sweeps
            ns = min(he, thin - done)
            if backend == 'c':
                sweeps_c(lib, T, X, R, rng, ns, threads, MUZ, KZ, SM, LOD, KC, CNT if SS is not None else None)
            else:
                sweeps_numpy(T, X, R, g, ns, MUZ, KZ, SM, LOD)
            done += ns
            if SS is not None:
                update_pi(counted=backend == 'c')
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
        Rd = R if nd == T.N else R[:, :nd]                   # the data rows (after them, the balance term's)
        y = Rd + T.d[:nd]
        if PMr is not None:                                  # --paleo-misfit: its misfit, and the paleo rows unscaled, so
            pr_ = PMr['rows']                                # that chi2 and the predictions are as without it
            tr['paleo_mode'] = np.array(PMr['mode'])
            tr.setdefault('paleo_misfit', np.zeros((K, n_rec)))[:, rec] = 2.0 * np.sum(   # the curve's whole misfit (the
                _pm_f_np(PMr, np.tile(np.arange(len(pr_)), (K, 1)), np.log(np.maximum(y[:, pr_], 1e-300))), 1) / b_p   # tables carry b)
            y = y.copy()
            y[:, pr_] /= PMr['eps']
            Rd = y - Dd
        if MFr is not None:                                  # --mfd-misfit log10: the bins' misfit, sum (log10(y / R) / s)^2,
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
            C_ = _sp_w(T, np.arange(nd), Rd) * Rd ** 2
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
            tr.setdefault('slip_sum', np.zeros((K, n_rec)))[:, rec] = np.sum(R[:, nbal:T.n_fit] ** 2, 1) / T.sigma ** 2
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
    fitted = 'misfit' in tr and not np.array_equal(tr['misfit'], tr['chi2'])
    q = ({'chi2 as fitted': tr['misfit'][:, nb:], 'chi2 (Gaussians)': tr['chi2'][:, nb:]} if fitted else {'chi2': tr['chi2'][:, nb:]})
    q.update({'sum_m': tr['sum_m'][:, nb:], 'pred/data scale': tr['tscale'][:, nb:]})
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
    if 'mfd_misfit' in tr:                           # --mfd-misfit log10: in the likelihood in place of the bins' Gaussian
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


def selftest_spikeslab(backends=('c', 'numpy'), gam=0.0):
    """Spike-and-slab: 2 rates in one bin, with different slab floors and soft tops and the fraction on integrated out
    under Beta(1, 1); P(on) and E[log10 m | on] against exact values from enumerating the on/off states on a grid.
    gam = 1/2: the square-root slab (density m^-1/2 below the top)."""
    from scipy.special import betaln
    print('spike-and-slab prior' + (' (square-root slab)' if gam else '') + ': P(on) and E[log10 m | on] vs exact enumeration '
          'and grid integrals')
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
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
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
    print('SELFTEST (spike-and-slab' + (', square-root slab' if gam else '') + ')', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_linear(backends=('c', 'numpy')):
    """The linear prior: 2 correlated rates, each uniform from 0 to its own top (0.3 and 1, as rates in two bins), the first
    pushed against its top by the data; their means and sds against a grid integral."""
    print('linear prior (per-rupture tops): E[m] and sd vs grid integration')
    G, d, sig, top = np.array([[1.0, 0.7], [0.5, 1.0]]), np.array([0.6, 0.45]), 0.1, np.array([0.3, 1.0])
    u = [np.linspace(0.0, t, 3001) for t in top]
    w = []
    for ui in u:                                         # trapezoid weights: the density is not 0 at the walls
        wi = np.full(len(ui), ui[1] - ui[0])
        wi[[0, -1]] *= 0.5
        w.append(wi)
    A, B = np.meshgrid(u[0], u[1], indexing='ij')
    Pg = np.exp(-((A + 0.7 * B - d[0]) ** 2 + (0.5 * A + B - d[1]) ** 2) / (2 * sig ** 2)) * np.outer(w[0], w[1])
    Pg /= Pg.sum()
    ex = [np.sum(Pg * A), np.sum(Pg * B)]
    sd = [np.sqrt(np.sum(Pg * A ** 2) - ex[0] ** 2), np.sqrt(np.sum(Pg * B ** 2) - ex[1] ** 2)]
    ok = True
    for be in backends:
        T = Target(sp.csc_matrix(G), d, sig, 0.0, top)
        K, sw = (8, 40000) if be == 'c' else (4, 8000)
        tr = sample(T, np.full((K, 2), 0.1), sw, 1000, 1, be, 1, 13, monitor=(0, 1), verbose=False, keep_samples=False)
        x = tr['mon'][:, 1000:]
        zs = [(x[..., j].mean() - ex[j]) / (x[..., j].std() / np.sqrt(ess(x[..., j]))) for j in (0, 1)]
        ok &= all(abs(z) < 4 for z in zs) and x[..., 0].max() <= top[0] and x[..., 1].max() <= top[1]
        print(f'   {be:6s} means ({x[..., 0].mean():.4f}, {x[..., 1].mean():.4f}) exact ({ex[0]:.4f}, {ex[1]:.4f}) '
              f'z = ({zs[0]:+.2f}, {zs[1]:+.2f}); sds ({x[..., 0].std():.4f}, {x[..., 1].std():.4f}) exact ({sd[0]:.4f}, {sd[1]:.4f}); '
              f'highest draws ({x[..., 0].max():.4f}, {x[..., 1].max():.4f})')
    print('SELFTEST (linear prior)', 'PASSED' if ok else 'FAILED')
    return ok


def selftest_halfway(backends=('c', 'numpy'), w=0.5):
    """The halfway prior: 2 rates with different floors and soft tops, each from a w : 1 - w mix of a uniform and a
    log-uniform range, both parts normalized on a grid (independently of the kernel's closed forms); E[log10 m] and E[m]
    against grid integrals."""
    print('halfway prior (a mix of a uniform and a log-uniform range): E[log10 m] and E[m] vs grid integrals')
    Gs = sp.csc_matrix(np.array([[1.0, 0.7], [0.5, 1.0]]))
    d, sig, lo10, su = np.array([0.6, 0.3]), 0.1, np.array([-3.0, -5.0]), np.array([0.05, 0.15])
    u = [np.linspace(l, min(8 * sv, 1.0), 3001) for l, sv in zip(lo10, su)]
    pw = []
    for ui, sv in zip(u, su):                        # each rate's prior mass per grid node in log10 m
        tw = np.full(len(ui), ui[1] - ui[0])
        tw[[0, -1]] *= 0.5
        sh = np.where(ui > 0, np.exp(-ui ** 2 / (2 * sv ** 2)), 1.0) * tw
        lin, lg = sh * 10.0 ** ui, sh                # a uniform range is 10^u per unit of log10 m; a log-uniform one flat
        pw.append(w * lin / lin.sum() + (1 - w) * lg / lg.sum())
    U1, U2 = np.meshgrid(u[0], u[1], indexing='ij')
    M1, M2 = 10 ** U1, 10 ** U2
    chi = ((M1 + 0.7 * M2 - d[0]) ** 2 + (0.5 * M1 + M2 - d[1]) ** 2) / sig ** 2
    Pg = np.exp(-0.5 * (chi - chi.min())) * pw[0][:, None] * pw[1][None, :]
    Pg /= Pg.sum()
    ex = [((Pg * U1).sum(), (Pg * M1).sum()), ((Pg * U2).sum(), (Pg * M2).sum())]
    ok = True
    for be in backends:
        T = Target(Gs, d, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.mixw = w
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.full((K, 2), 0.1), sw, sw // 10, 1, be, 1, 17, monitor=(0, 1), verbose=False, keep_samples=False)
        Mo = tr['mon'][:, sw // 10:, :]
        for i in range(2):
            out = []
            for v, e in ((np.log10(Mo[:, :, i]), ex[i][0]), (Mo[:, :, i], ex[i][1])):
                z = (v.mean() - e) / (v.std() / np.sqrt(ess(v)))
                ok &= abs(z) < 4
                out.append(f'{v.mean():.4f} (exact {e:.4f}, z {z:+.2f})')
            print(f'   {be:5s} rate {i} (floor 1e{lo10[i]:g}): E[log10 m] {out[0]}; E[m] {out[1]}')
    print('SELFTEST (halfway prior)', 'PASSED' if ok else 'FAILED')
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


def selftest_logband(backends=('c', 'numpy')):
    """The soft band (--log-band) on two spike-and-slab rates with weak targets (each about 0.4 of its uncertainty, so
    the likelihood alone would let the predictions fall far below them), with swap moves between them, against a grid
    integral of likelihood x prior x exp(-penalty): P(on) and E[log10 m | on] of each rate."""
    from scipy.special import betaln
    print('soft band (--log-band): spike-and-slab with swaps vs a grid integral')
    ok = True
    # the same prior for both rates, as swap moves need (the sampler's pairs are always in one magnitude bin)
    Gs, sig, lo10, su, a_, b_ = np.array([[1.0, 0.7], [0.5, 1.0]]), 0.1, np.array([-4.0, -4.0]), np.array([0.1, 0.1]), 1.0, 1.0
    dP, tau = np.array([0.04, 0.03]), 0.5
    w = 1.0 / (2.0 * (tau * LN10) ** 2)
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
        Y = np.stack(np.broadcast_arrays(m1 + 0.7 * m2, 0.5 * m1 + m2), -1)
        with np.errstate(divide='ignore'):
            pen = w * np.sum(np.log(Y / dP) ** 2, -1)
        return np.exp(-0.5 * np.sum((Y - dP) ** 2, -1) / sig ** 2 - pen)
    L1, L2, L12 = Lb(10 ** u[0], 0.0), Lb(0.0, 10 ** u[1]), Lb(M1, M2) * gw[0][:, None] * gw[1][None, :]
    w10, w01, w11 = pg[1] * (gw[0] @ L1), pg[1] * (gw[1] @ L2), pg[2] * L12.sum()   # both off: impossible
    W_ = w10 + w01 + w11
    p_on = [(w10 + w11) / W_, (w01 + w11) / W_]
    e_u = [(pg[1] * (gw[0] * u[0]) @ L1 + pg[2] * (L12.sum(1) @ u[0])) / (w10 + w11),
           (pg[1] * (gw[1] * u[1]) @ L2 + pg[2] * (L12.sum(0) @ u[1])) / (w01 + w11)]
    for be in backends:
        T = Target(sp.csc_matrix(Gs), dP, sig, 0.0, 10.0, logbox=(-3.0, 0.0, 0.0),
                   pairs=(np.array([0]), np.array([1]), np.array([Gs[:, 0] @ Gs[:, 1]])))
        T.zlo_vec = np.ascontiguousarray(lo10 * LN10)
        T.ku_vec = np.ascontiguousarray(1.0 / (2.0 * (su * LN10) ** 2))
        T.pkind = 3
        T.ss = dict(bins=np.zeros(2, np.int64), B=1, nb=np.array([2.0]), a=a_, b=b_)
        T.lw = dict(W=np.full(2, w), LD=np.log(dP), DY=np.ascontiguousarray(dP), rows=np.arange(2), tau=tau)
        K, sw = (8, 40000) if be == 'c' else (4, 4000)
        tr = sample(T, np.tile([0.04, 0.0], (K, 1)), sw, sw // 10, 1, be, 1, 23, monitor=(0, 1), verbose=False,
                    keep_samples=False, swaps=True)
        Mo = tr['mon'][:, sw // 10:, :]
        ok &= bool((Mo > 0).any(-1).all())                # never both off: the penalty forbids a zero prediction
        for i in range(2):
            on = (Mo[:, :, i] > 0).astype(float)
            p = on.mean()
            if on.std() == 0:
                zp = 0.0 if on.size * min(p_on[i], 1 - p_on[i]) < 5 else np.inf
            else:
                zp = (p - p_on[i]) / np.sqrt(max(p_on[i] * (1 - p_on[i]), 1e-12) / ess(on))
            uu = np.log10(Mo[:, :, i][Mo[:, :, i] > 0])
            zu = (uu.mean() - e_u[i]) / (uu.std() / np.sqrt(max(1.0, ess(on) * p)))
            ok &= abs(zp) < 4 and abs(zu) < 4
            print(f'   {be:5s} rate {i}: P(on) {p:.4f} (exact {p_on[i]:.4f}, z {zp:+.2f}); E[log10 m | on] {uu.mean():.4f} '
                  f'(exact {e_u[i]:.4f}, z {zu:+.2f}); swap acceptance {tr["swap_rate"].mean():.2f}')
    print('SELFTEST (soft band)', 'PASSED' if ok else 'FAILED')
    return bool(ok)


def selftest_band(backends=('c', 'numpy')):
    """The band (--band), positivity (--positive), the MFD's monotonicity (--mfd-monotonic, two rates in two bins,
    against the exact answer with m1 > m0 forbidden) and the plate rate (--plate-rate, a row of both signs kept between
    two limits): the self-tests' small problems with every prediction kept within a
    factor F of its target, against
    grid integrals restricted to the band: spike-and-slab (P(on) and E[log10 m | on], F = 1.3, where the band forbids the
    first rate's off state), a log-box prior with swap
    moves between near-duplicate columns (E[log10 m], F = 2) and the linear prior's truncated normals (E[m], F = 1.25);
    with each, how far the exact answer without the band lies, and that no recorded draw leaves the band."""
    from scipy.special import betaln
    print('the band: spike-and-slab, positivity, MFD monotonicity (flat and with a capped rise), the plate rate, a log-box '
          'prior with swaps and the linear prior vs grid integrals within their limits')
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
    # the linear prior: the selftest_linear problem
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
        print(f'   {be:5s} linear: E[m] ({x[..., 0].mean():.4f}, {x[..., 1].mean():.4f}) exact ({ex[0]:.4f}, {ex[1]:.4f}) '
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


def band_start(G, d, rows, F, m0, hi, verbose=True):
    """A start inside the band from m0 (rates, 0 <= m <= hi): the least-squares fit to the data with a penalty, growing
    in steps, on predictions outside a band a little narrower than the band (a factor F^0.9), stopped as soon as every
    row lies within F^0.95 of its target, so strictly inside the band (the start needs only to be inside it; how well it
    fits matters only to the first sweeps of burn-in). Each rate is measured in units of 1 / ||G_i|| for the
    optimizer, which has the same minimum but converges several times faster (column norms span two orders of
    magnitude or more). None if the steps end with rows still outside the band."""
    from scipy.optimize import minimize
    G = sp.csr_matrix(G)
    c = 1.0 / np.maximum(np.sqrt(np.asarray(G.multiply(G).sum(0)).ravel()), 1e-300)   # each rate's unit
    Gs = G @ sp.diags(c)
    Gst = Gs.T.tocsr()
    rows = np.asarray(rows)
    f_in = F ** 0.9
    lo, hi_y = np.full(len(d), -np.inf), np.full(len(d), np.inf)
    lo[rows], hi_y[rows] = d[rows] / f_in, d[rows] * f_in
    out = lambda y, f=F: int(np.sum((y[rows] < d[rows] / f) | (y[rows] > d[rows] * f)))
    u = np.clip(np.asarray(m0, np.float64), 0.0, hi) / c
    bounds = list(zip(np.zeros(len(u)), np.broadcast_to(hi, len(u)) / c))
    for lam in (1e2, 1e4, 1e6, 1e8):
        def fg(x):
            y = Gs @ x
            e, v = y - d, np.minimum(y - lo, 0.0) + np.maximum(y - hi_y, 0.0)
            return 0.5 * e @ e + 0.5 * lam * v @ v, Gst @ (e + lam * v)
        inside = {}

        def stop(xk):                                    # inside the band with a margin: done
            if out(Gs @ xk, F ** 0.95) == 0:
                inside['x'] = np.array(xk)
                raise StopIteration
        try:
            res = minimize(fg, u, jac=True, method='L-BFGS-B', bounds=bounds, callback=stop,
                           options=dict(maxiter=20000, maxfun=40000))
            u = inside.get('x', res.x)
        except StopIteration:                            # an older SciPy passes it on instead of stopping
            u = inside['x']
        m = u * c
        n_out = out(G @ m)
        if verbose:
            print(f'   penalty {lam:g}: chi2 {np.sum((G @ m - d) ** 2):.1f}, {n_out} rows outside the band', flush=True)
        if n_out == 0:
            return m
    return None


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


def band_start_key(G, d, rows, F, m0, hi):
    """A fingerprint of everything band_start's result depends on: the problem's matrix and targets, the banded rows, the
    factor, the start it moves and the rates' upper bounds."""
    G = sp.csr_matrix(G)
    h = hashlib.sha1(repr(float(F)).encode() if np.ndim(F) == 0 else np.ascontiguousarray(F, np.float64).tobytes())
    for v in (G.indptr, G.indices, G.data, d, rows, m0, np.broadcast_to(hi, len(m0))):
        h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


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
    ap.add_argument('--prelim', default=None, help='diagnostics from preliminaries.py (default: next to --problem)')
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
                         "Kept for comparison: binbox-theta:THETA:SE:TOP (per-bin log-uniform boxes below R_j, depths "
                         "n_j / (THETA ln 10) decades, lower edge SE (0 = hard); THETA = 1 puts no pressure on the bin totals; "
                         "the alternative until October 2026), binbox-floor:A:SE:TOP (per-bin boxes down to a common floor 10^A; the default until October 2026) and "
                         "binbox:W:SE:TOP (W decades below each R_j); 'logbox:A:B:SE' (one box for all ruptures, flat in log10 m "
                         "on [A, B] with half-Gaussian edges of sd SE decades, SE = 0: hard; e.g. logbox:-12:-2:1), "
                         "'logbox-centre:W:SE' (width W decades centred on the average data-imposed rate) and 'logbox-mean:W:SE' "
                         "(placed so its mean equals it); 'lognormal' (hierarchical, retired); 'linear' (optional: each rupture's "
                         "rate uniform from 0 to its bin's target rate R_j, a hard top, drawn in linear space by the exact "
                         "truncated-normal updates; its prior pushes the bin totals up, n_j R_j / 2 a priori); 'halfway' "
                         "(optional, halfway between 'linear' and per-bin log-uniform boxes: each rupture's rate from a 50/50 "
                         "mix of a uniform and a log-uniform range from 10^A up to its bin's target rate R_j, every rupture always "
                         "on; halfway:A:TOP:W for floor 10^A (-10 by default), TOP as for spikeslab and the uniform part's share "
                         "W, 0.5 by default; it too pushes the bin totals up); 'flat'; a prior mean tau (exponential prior) or a "
                         ".npy of tau_i")
    ap.add_argument('--down-the-middle', nargs='?', const='equal', default=None, choices=['equal', 'zero-model'],
                    help="add the balance term (the notebook's DOWN_THE_MIDDLE): each data type's relative shortfall s = (sum of "
                         "predictions - sum of targets) / sum of targets (slip-rate sections, magnitude bins, paleo rows; in units "
                         "of each measurement's uncertainty) costs A s^2. 'equal' (the default when given alone): A the largest "
                         "weight common to the types that keeps each at or below one extra measurement; 'zero-model': each type's "
                         "own A, its zero-model misfit, so s = -1 or +1 costs a type what the zero model costs it in chi2 "
                         "(cru_common.balance_rows); as extra rows of G and d, so for any prior. "
                         "The recorded chi2 and predictions stay on the data rows; the term is recorded on its own")
    ap.add_argument('--slab', default='log', choices=['log', 'sqrt'],
                    help="spike-and-slab: the slab's shape below its top: 'log' (log-uniform, density 1/m; the default) or 'sqrt' "
                         "(uniform in sqrt m, density m^-1/2, half the log-uniform's pull of each rate towards 0; small rates "
                         "cost more, so fewer ruptures are on: the cross-check on how the lower end of the rates is treated)")
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
    ap.add_argument('--band', type=float, default=None, metavar='F',
                    help='keep the prediction of every data row with a positive target (slip-rate, MFD-bin and paleo rows; '
                         'each subsection its own row) within a factor F of its target, from d/F to F d: a hard constraint on '
                         'the rates, for any prior (10: an order of magnitude either way). A start outside it is first moved '
                         'inside by a penalized least-squares fit, made once per problem: it is kept beside --problem '
                         '(<problem>_band<F>_start.npz) and taken from there by later runs with the same problem, band and '
                         'start')
    ap.add_argument('--paleo-misfit', default='gaussian', choices=['gaussian', 'years', 'log', 'gaussian+years', 'gaussian+log'],
                    help='the paleo rows\' misfit (the notebook\'s PALEO_MISFIT): \'gaussian\' (default), the inputs\' '
                         'Gaussian in rate; \'years\': -2 ln of each site\'s recurrence-interval PDF in years, a curve '
                         'through its quantiles with its mode at 1/bestEstimate (paleo_misfit.py; needs --rupture-set and '
                         'spike-and-slab, and keeps each paleo prediction above zero; the soft band then acts on the slip-rate rows only); \'log\': the '
                         'same curve\'s density in log rate, its mode at bestEstimate; \'gaussian+years\' (the notebook\'s '
                         'MISFIT \'half\'): half the Gaussian in rate plus half the years curve, the geometric mean of the two')
    ap.add_argument('--slip-band', default=None, metavar='positive|F',
                    help='the slip-rate rows\' hard limit (the notebook\'s SLIP_BAND): \'positive\', each subsection\'s '
                         'prediction above zero, or a factor F above 1, each within F of its target; per row group, in place '
                         'of --positive and --band (which act on every positive-target row, the MFD bins too)')
    ap.add_argument('--paleo-band', default=None, metavar='positive|F',
                    help='the paleo rows\' hard limit, as --slip-band (the notebook\'s PALEO_BAND until 2026-11-05); with '
                         '--paleo-misfit, positive unless a factor is given')
    ap.add_argument('--slip-log-band', type=float, default=None, metavar='TAU',
                    help='a soft band on the slip-rate rows (the notebook\'s SLIP_BAND_SOFT), as --log-band, with '
                         '--slip-band positive')
    ap.add_argument('--slip-misfit', default='split', choices=['split', 'split+log10', 'log10'],
                    help="the slip-rate rows' misfit: 'split' (default), the Gaussian (a split normal with the CFM's slip "
                         "rates); 'split+log10', half of it plus half a Gaussian in log10(prediction / target) with the same "
                         "sd, s = log10(1 + sd / target) on each side (sd the side's effective sd, the template weights' "
                         "included, as --mfd-misfit log10 has the bins'), the geometric mean of the two densities, so a "
                         'prediction an order of magnitude off is unlikely and two orders far more so (each section still one '
                         'measurement; spike-and-slab, with --slip-band positive, not with the soft bands); \'log10\', the log10 '
                         'Gaussian alone')
    ap.add_argument('--paleo-log-band', type=float, default=None, metavar='TAU',
                    help='a soft band on the paleo rows (the notebook\'s PALEO_BAND_SOFT), as --log-band, with --paleo-band '
                         'positive, and not with --paleo-misfit years or log')
    ap.add_argument('--log-band', type=float, default=None, metavar='TAU',
                    help='with --positive, spike-and-slab only: a soft version of the band (the notebook\'s BAND_SOFT), a '
                         'Gaussian penalty on each positive-target data row\'s log ratio of prediction to target, 1/2 '
                         '(log10(prediction / target) / TAU)^2, TAU decades at 1 sd (1: within an order of magnitude at 1 '
                         'sd, two at 2 sd; 2: within two orders at 1 sd), each slip-rate subsection\'s '
                         'term divided by the k subsections of its section; kept exactly by a Metropolis step on each '
                         'update (the switch on or off and swaps take it exactly; a slab move is accepted by its change). '
                         'Recorded apart from chi2')
    ap.add_argument('--positive', action='store_true',
                    help='an alternative to --band (not with it), spike-and-slab only: keep the prediction of every data row '
                         'with a positive target above zero in every draw, that is, at least one of the ruptures in each row '
                         'on, with no other limit. A start that predicts 0 for some rows gets, for each, a rupture crossing it '
                         'switched on at twice its slab floor')
    ap.add_argument('--mfd-misfit', default='gaussian', choices=['gaussian', 'log10', 'gaussian+log10'],
                    help="the MFD bins' misfit: gaussian (default), the inputs' Gaussian in rate; log10, each bin's total "
                         "scored by ((log10(total) - log10(R_j)) / s_j)^2, s_j = log10(1 + sigma_j / R_j) (the soft top's "
                         "width), in place of the Gaussian, with positivity on the bins (spike-and-slab, the per-group "
                         "--slip-band and --paleo-band or --positive); gaussian+log10, half of each, the geometric mean")
    ap.add_argument('--all-positive', action='store_true',
                    help='every slip-rate, paleo and MFD prediction kept above zero (the notebook always): --slip-band and '
                         '--paleo-band positive unless given as a factor, and the MFD bins above zero; through the per-group limits')
    ap.add_argument('--mfd-monotonic', action='store_true',
                    help='with --bins-block\'s magnitude bins: in every draw, wherever a bin\'s target is not above the next '
                         'lower bin\'s (neighbouring bins, counted up from --lowest-bin in steps of --bin-width; a bin '
                         'cleaning removed breaks the pair), its total rate is not above that bin\'s either, and where the '
                         'target rises the rise is capped (--mfd-rise-sd): a hard '
                         'constraint, alongside --band or --positive or alone. Kept by constraint rows, each '
                         'eps x (rho x lower bin total - upper bin total) >= 0, rho = 1 except where the target rises, with eps = 1e-9, '
                         'which add at most about 1e-20 to '
                         'chi2; a start that breaks it is scaled down bin by bin first')
    ap.add_argument('--mfd-rise-sd', type=float, default=0.89, metavar='M',
                    help='--mfd-monotonic: where a bin\'s target rises above the next lower bin\'s, the upper total may '
                         'be up to rho times the lower, rho the ratio at the lower bin\'s target of the target difference '
                         'plus M sd of it (from the two bins\' uncertainties; default 0.89, the median distance of the other '
                         'pairs\' walls above their target differences with the NZSHM22 inputs: 1.427 for M6.8-6.9 there); '
                         'a ratio, so it caps the MFD\'s shape and not its level; negative: such pairs are left free')
    ap.add_argument('--lowest-bin', type=float, default=6.8, help='--mfd-monotonic: the lowest magnitude bin\'s lower edge')
    ap.add_argument('--bin-width', type=float, default=0.1, help='--mfd-monotonic: the magnitude bins\' width')
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
    ap.add_argument('--selftest', nargs='?', const='all', choices=['all', 'current'],
                    help="exact tests: 'all' (the default) after any change to the sampler, twelve verdicts; 'current' only "
                         "what the notebook's default uses (per-rupture lower edges, the soft top, spike-and-slab, the balance "
                         "term, swap moves and the band), six verdicts")
    a = ap.parse_args()
    if a.selftest == 'current':
        ok = selftest_zlo()
        ok &= selftest_edges()
        ok &= selftest_spikeslab()
        ok &= selftest_balance()
        ok &= selftest_swaps()
        ok &= selftest_band()
        sys.exit(0 if ok else 1)
    if a.selftest:
        ok = selftest()
        ok &= selftest_hier()
        ok &= selftest_zlo()
        ok &= selftest_edges()
        ok &= selftest_spikeslab()
        ok &= selftest_spikeslab(gam=0.5)
        ok &= selftest_linear()
        ok &= selftest_halfway()
        ok &= selftest_balance()
        ok &= selftest_band()
        ok &= selftest_logband()
        ok &= selftest_paleo()
        ok &= selftest_paleo(b=0.5)
        ok &= selftest_split()
        ok &= selftest_mfdlog()
        ok &= selftest_mfdlog(b=0.5)
        ok &= selftest_slipgeo()
        ok &= selftest_slipgeo(b=1.0)
        sys.exit(0 if ok else 1)
    kept = (a.sweeps - a.burn) // a.thin
    if kept < 4:
        raise SystemExit(f'only {max(kept, 0)} recorded draw(s) per chain after burn-in ((sweeps - burn) / thin = '
                         f'({a.sweeps} - {a.burn}) / {a.thin}); R-hat and effective draws need at least 4')
    report_to(a.out + '_log.txt')                        # everything printed below is also kept with the run
    P = load_problem(a.problem)
    n = P['G'].shape[1]
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
        G_fit, d_fit = sp_csc(sp.vstack([sp.csr_matrix(G_fit), A_ss])), np.r_[d_fit, b_ss]
        print('summed slip rates: along each cross-section, the model\'s slip along the plate motion added up against the '
              'targets\' added up, '
              'a soft term in the likelihood beside the data (recorded apart from chi2); ' + describe_slip_sum(ss_info, A_ss.nnz)
              + '; the slip-rate rows checked against the rupture set')
    n_fit, mono = G_fit.shape[0], None             # the rows in the likelihood; --mfd-monotonic's constraint rows follow
    if a.mfd_monotonic:
        if a.bins_block not in P['block_names']:
            raise SystemExit(f'--mfd-monotonic: no row block named {a.bins_block!r} (--bins-block)')
        mrows, Rm, origm, mlo = magnitude_bins(P, a.bins_block, a.lowest_bin, a.bin_width)
        Gb = sp.csr_matrix(P['G'])[mrows]
        members = [Gb.indices[Gb.indptr[k]:Gb.indptr[k + 1]] for k in range(len(mrows))]
        sig_b = Rm / (P['d'][mrows] / P['row_scale'][mrows])    # each bin's uncertainty, as cleaning left it
        adj = [(k, k + 1) for k in range(len(mrows) - 1) if origm[k + 1] == origm[k] + 1]
        flat = [(k, k1) for k, k1 in adj if Rm[k1] <= Rm[k]]
        rise = [(k, k1) for k, k1 in adj if Rm[k1] > Rm[k]] if a.mfd_rise_sd >= 0 else []
        skipped = [(k, k + 1) for k in range(len(mrows) - 1) if (k, k + 1) not in flat + rise]
        pairs_m = sorted(flat + rise)
        # the upper bin's total may be up to rho times the lower's: 1 where the target does not rise; where it rises,
        # the ratio at the lower bin's target of the target difference plus --mfd-rise-sd sd of it (from the two bins'
        # uncertainties), the median distance of the other pairs' walls above their target differences with the NZSHM22
        # inputs (0.89 sd): a ratio, so the cap limits the MFD's shape and not its level
        rhos = np.array([1.0 if (k, k1) in flat else
                         (Rm[k] + (Rm[k1] - Rm[k]) + a.mfd_rise_sd * np.hypot(sig_b[k], sig_b[k1])) / Rm[k]
                         for k, k1 in pairs_m])
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
            A_p, pinfo = plate_rows(rd, P, a.cross_sections, a.slip_block, cs=cs)
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
    hi_vec = None                                  # linear prior: each rupture's top
    mixw = 0.0                                     # halfway prior: the uniform part's share
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
    elif a.prior.startswith('binbox') or a.prior.startswith('spikeslab') or a.prior.startswith('halfway'):
        # per-bin log-uniform boxes, sampled as u = m / (F R_j) in a box below 1; spikeslab: each rupture is off (rate 0)
        # or on, with a log-uniform rate from the slab floor 10^A up to the bin's top, and a per-bin fraction on, estimated;
        # halfway: the boxes down to 10^A, each rate from a mix of a uniform and a log-uniform range
        spike_mode = a.prior.startswith('spikeslab')
        halfway_mode = a.prior.startswith('halfway')
        theta_mode = a.prior.startswith('binbox-theta')
        floor_mode = a.prior.startswith('binbox-floor') or spike_mode or halfway_mode  # a common floor 10^W for every bin
        if spike_mode:                                   # spikeslab:A:TOP:a:b (slab floor 10^A, A = -10 by default)
            sp_ = (a.prior.split(':') + [None] * 5)[1:5]
            W, se, top = float(sp_[0] or -10), 0.0, str(sp_[1] or 'soft').lower()
            ss_a, ss_b = float(sp_[2] or 1), float(sp_[3] or 1)
        elif halfway_mode:                               # halfway:A:TOP:W (floor 10^A, A = -10; W the uniform part's share)
            hw = (a.prior.split(':') + [None] * 4)[1:4]
            W, se, top, mixw = float(hw[0] or -10), 0.0, str(hw[1] or 'soft').lower(), float(hw[2] or 0.5)
            if not 0.0 < mixw < 1.0:
                raise SystemExit(f'halfway:A:TOP:W: the uniform part\'s share W must lie between 0 and 1 (got {mixw:g})')
        else:
            parts = (a.prior.split(':') + [None] * 4)[1:4]
            W, se = float(parts[0] or (1 if theta_mode else -12 if floor_mode else 10)), float(parts[1] if parts[1] is not None else 0)
            top = str(parts[2] or 'soft').lower()        # 'soft' (default), 'hard', or a factor F for a hard top at F x R_j
        soft_top, F = (top == 'soft'), (1.0 if top in ('soft', 'hard') else float(top))
        Rb, rows = bin_targets(P, a.bins_block)
        bins = rupture_bins(P, rows, a.bins_block)
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
            if halfway_mode:
                hmean = mixw * 0.5 * nb * F + (1.0 - mixw) * imp   # the uniform part's mean about half the top
                print(f'prior: halfway between linear and log-uniform, every rupture always on: each rupture\'s rate from a '
                      f'{mixw:.0%} : {1 - mixw:.0%} mix of a uniform and a log-uniform range from 1e{W:g} up to R_j ({Wj.min():.2f} '
                      f'to {Wj.max():.2f} decades; {edge}); implied prior bin totals about {hmean.min():.0f} to {hmean.max():.0f} '
                      f'x their targets')
            elif spike_mode:
                slab = 'rate uniform in sqrt(rate)' if a.slab == 'sqrt' else 'log-uniform rate'
                print(f'prior: spike-and-slab per magnitude bin: each rupture is off (rate 0) or on, with a {slab} from '
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
    elif a.prior == 'linear':                     # each rupture's rate uniform on [0, R_j]: truncated-normal Gibbs updates
        Rb, rows = bin_targets(P, a.bins_block)
        bins = rupture_bins(P, rows, a.bins_block)
        if a.transfers:
            raise SystemExit('transfer moves are not implemented for the linear prior')
        hi_vec, beta = np.ascontiguousarray(Rb[bins]), np.zeros(n)
        nb = np.bincount(bins, minlength=len(rows))
        print(f'prior: linear, each rupture\'s rate uniform from 0 to its bin\'s target rate R_j ({Rb.min():.2e} to {Rb.max():.2e}; '
              f'hard top); prior bin totals n_j R_j / 2, {nb.min() / 2:.0f} to {nb.max() / 2:.0f} x their targets')
    elif a.prior.startswith('logbox'):
        la, lb, lse, desc = resolve_logbox(a.prior, P, a.bins_block)
        logbox = (la, lb, lse)
        print('prior:', desc)
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
        if a.band is not None and not a.positive:
            raise SystemExit('--paleo-misfit with --band: give the per-group --slip-band (the paleo rows then kept above '
                             'zero, or within --paleo-band)')
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
        if a.log_band is not None:
            raise SystemExit('--mfd-misfit log10 with --log-band (every positive-target row, the bins among them): give the '
                             'per-group --slip-log-band instead')
        if a.bins_block not in P['block_names']:
            raise SystemExit(f'--mfd-misfit log10: no row block named {a.bins_block!r} (--bins-block)')
        mf_rows = np.flatnonzero(np.asarray(P['block']) == list(P['block_names']).index(a.bins_block))
        dmf = np.asarray(P['d'], float)[mf_rows]
        if not (dmf > 0).all():
            raise SystemExit('--mfd-misfit log10: every bin needs a positive target')
        mf_s = np.log10(1.0 + 1.0 / dmf)               # s_j = log10(1 + sigma_j / R_j); a bin's row is R_j / sigma_j
        rs_ = np.ones(G_fit.shape[0])
        rs_[mf_rows] = lin_sc(b_m)
        if pm_rows is None:
            d_data = np.array(np.asarray(P['d'], float)[:P['G'].shape[0]])
        G_fit, d_fit = sp_csc(sp.diags(rs_) @ sp.csr_matrix(G_fit)), d_fit * rs_
    sm_rows = None                                 # --slip-misfit split+log10, log10: the slip rows' Gaussian halved (scaled by
    if a.slip_misfit != 'split':                   # 1/sqrt 2), or scaled out of it
        if not spike_mode or a.transfers:
            raise SystemExit(f'--slip-misfit {a.slip_misfit} needs spike-and-slab, without --transfers')
        if a.slip_log_band is not None or a.log_band is not None or a.paleo_log_band is not None:
            raise SystemExit(f'--slip-misfit {a.slip_misfit} with a soft band: it already scores each slip-rate prediction '
                             'in log10 (the soft bands are left out)')
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
    if a.slip_band is None and a.slip_misfit != 'split':
        a.slip_band = 'positive'                   # its log10 half is infinite at 0: each slip-rate prediction above zero
    fprelim = a.prelim or prelim_path(a.problem)
    pairs = None
    if a.swap_cos > 0:
        pr = np.load(fprelim)
        sel = pr['dup_cos'] > a.swap_cos
        if lognormal is not None:                  # swaps must leave the prior unchanged: same bin only
            sel &= lognormal['bins'][pr['dup_i']] == lognormal['bins'][pr['dup_j']]
        if zlo_vec is not None or ku_vec is not None or hi_vec is not None:   # per-bin depths or tops: boxes differ
            sel &= bins[pr['dup_i']] == bins[pr['dup_j']]                      # between bins
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
        print(f'swap moves: {sel.sum()} near-duplicate pairs with cosine > {a.swap_cos}, one pass per recorded draw')
    dirs = dict(np.load(a.transfers)) if a.transfers else None
    if dirs is not None:
        print(f'transfer moves: {len(dirs["uu"])} directions from {a.transfers}, one pass per recorded draw')
    if scale is None:
        T = Target(G_fit, d_fit, P['sigma'], P['lo'], P['hi'] if hi_vec is None else hi_vec, beta, logbox, pairs, dirs, lognormal)
    else:                                          # u = m / scale: columns scaled, box [1e-W, 1], u <= 1 is the hard top
        import scipy.sparse as _sp
        T = Target(sp_csc(G_fit @ _sp.diags(scale)), d_fit, P['sigma'], 0.0, hi_u, beta, logbox, pairs, dirs, lognormal)
        T.scale = scale
        T.zlo_vec = zlo_vec
        T.ku_vec = ku_vec
        T.mixw = mixw
        if spike_mode:                               # pkind 3: x_i = 0 (off) or its slab value (on)
            T.pkind = 3
            T.ss = dict(bins=bins, B=len(rows), nb=np.bincount(bins, minlength=len(rows)).astype(float), a=ss_a, b=ss_b)
            T.gam = 0.5 if a.slab == 'sqrt' else 0.0
    T.n_data = P['G'].shape[0]                     # the data rows; the balance term's rows follow them
    T.n_bal = n_bal                                # the summed slip rates' rows, after the balance term's (None: none)
    T.n_fit = n_fit                                # and then --mfd-monotonic's and --plate-rate's constraint rows
    if ss_info is not None:
        T.slip_sum = dict(expect=ss_info['weight'] * len(ss_info['k']))
    if T.pairs is not None:                        # each swap pair's inner product: every row of G, checked on a sample
        q = np.random.default_rng(0).choice(len(T.pairs[0]), min(2000, len(T.pairs[0])), replace=False)
        Gc = sp.csc_matrix(T.G)
        ipq = np.asarray(Gc[:, T.pairs[0][q]].multiply(Gc[:, T.pairs[1][q]]).sum(0)).ravel()
        if not np.allclose(ipq, T.pairs[2][q], rtol=1e-8, atol=1e-12 * float(np.abs(ipq).max())):
            raise SystemExit('swap moves: the pairs\' inner products do not match the columns (a bookkeeping error)')
    if B_bal is not None:
        T.balance = dict(A=A_bal, names=[t['name'] for t in info_bal])
    if P.get('split_lo2') is not None:             # --cfm-sliprates: the slip-rate rows' split misfit, from the problem
        if not spike_mode or a.transfers:
            raise SystemExit('the problem\'s slip rates are split normals (preliminaries.py --cfm-sliprates): the sampler '
                             'handles them under spike-and-slab, without --transfers')
        L2_, H2_ = np.ones(T.N), np.ones(T.N)
        nd_ = P['G'].shape[0]
        L2_[:nd_], H2_[:nd_] = P['split_lo2'], P['split_hi2']
        T.split = dict(L2=np.ascontiguousarray(L2_), H2=np.ascontiguousarray(H2_))
        sk_ = np.flatnonzero((L2_ != 1.0) | (H2_ != 1.0))
        sk_ = np.log10(np.sqrt(L2_[sk_] / H2_[sk_]))            # log10 (sigma_hi / sigma_lo)
        c_ = P.get('cfm_sliprates') or ['?', '?', '?']
        print(f'slip rates from the CFM ({c_[0]}; SR_min and SR_max at {c_[1]} sigma, times {c_[2]}): each slip-rate row a split '
              f'normal, {len(sk_)} of them lopsided, sd above / below the mode median {10 ** np.median(sk_):.2f} (5-95% {10 ** np.percentile(sk_, 5):.2f}-'
              f'{10 ** np.percentile(sk_, 95):.2f}); their misfit evaluated exactly in every update and swap')
    band_rows = None
    grp = any(v is not None for v in (a.slip_band, a.paleo_band, a.slip_log_band, a.paleo_log_band))
    if grp and (a.positive or a.band is not None or a.log_band is not None):
        raise SystemExit('give --positive, --band and --log-band (every positive-target row) or the per-group --slip-band, '
                         '--paleo-band, --slip-log-band and --paleo-log-band, not both')
    pos_g = np.zeros(0, np.int64)                  # with per-group limits: the rows with positivity
    if a.all_positive:                             # every data group above zero
        if a.positive or a.band is not None or a.log_band is not None:
            raise SystemExit('--all-positive is per group: not with --positive, --band or --log-band')
        a.slip_band = a.slip_band or 'positive'
        a.paleo_band = a.paleo_band or 'positive'
    if (mf_rows is not None or a.paleo_misfit != 'gaussian' or a.all_positive) and not a.positive and a.band is None:
        grp = True                                 # --mfd-misfit, --paleo-misfit, --all-positive: positivity through the per-group limits
    if grp:
        if dirs is not None or lognormal is not None:
            raise SystemExit('--slip-band and --paleo-band are not implemented for transfer moves or the hierarchical prior')
        sb_ = list(P['block_names'])
        dpos = np.asarray(P['d'])[:P['G'].shape[0]] > 0
        blk = np.asarray(P['block'])
        g_slip = np.flatnonzero(dpos & (blk == sb_.index(a.slip_block))) if a.slip_block in sb_ else np.zeros(0, np.int64)
        not_pal = [sb_.index(b_) for b_ in (a.slip_block, a.bins_block) if b_ in sb_]
        g_pal = np.flatnonzero(dpos & ~np.isin(blk, not_pal))   # every positive-target row in neither group: the paleo rows
        def band_arg(v, name):
            if v is None or v == 'positive':
                return v
            try:
                f_ = float(v)
            except ValueError:
                raise SystemExit(f'--{name} {v}: give positive or a factor above 1')
            if not f_ > 1:
                raise SystemExit(f'--{name} {v}: give positive or a factor above 1 (10: an order of magnitude either way)')
            return f_
        sl_b, pa_b = band_arg(a.slip_band, 'slip-band'), band_arg(a.paleo_band, 'paleo-band')
        if pa_b is None and a.paleo_misfit != 'gaussian':
            pa_b = 'positive'                      # --paleo-misfit: each paleo prediction above zero (its misfit is
                                                   # infinite at 0), unless --paleo-band gives a factor
        if sm_rows is not None and sl_b is None:
            raise SystemExit(f'--slip-misfit {a.slip_misfit} needs --slip-band (its log10 misfit is infinite at 0)')
        for tau_, b_, name in ((a.slip_log_band, sl_b, 'slip'), (a.paleo_log_band, pa_b, 'paleo')):
            if tau_ is not None and (b_ != 'positive' or not tau_ > 0):
                raise SystemExit(f'--{name}-log-band {tau_:g}: give a width above 0, with --{name}-band positive')
        if a.paleo_log_band is not None and a.paleo_misfit != 'gaussian':
            raise SystemExit('--paleo-log-band with --paleo-misfit: the paleo misfit already scores each paleo prediction')
        g_mfd = (mf_rows if mf_rows is not None else                 # --mfd-misfit, --all-positive: the bins above 0
                 np.flatnonzero(dpos & (blk == sb_.index(a.bins_block))) if a.all_positive and a.bins_block in sb_
                 else np.zeros(0, np.int64))
        pos_g = np.r_[g_slip if sl_b == 'positive' else np.zeros(0, np.int64), g_pal if pa_b == 'positive' else np.zeros(0, np.int64),
                      g_mfd]
        b_rows, b_F = [], []
        for rows_, b_ in ((g_slip, sl_b), (g_pal, pa_b)):
            if isinstance(b_, float):
                b_rows.append(rows_); b_F.append(np.full(len(rows_), b_))
        b_rows = np.concatenate(b_rows) if b_rows else np.zeros(0, np.int64)
        b_F = np.concatenate(b_F) if b_F else np.zeros(0)
        if len(pos_g) or len(b_rows):
            try:
                T.set_groups(pos_g, b_rows, b_F)
            except ValueError as e:
                raise SystemExit(f'--slip-band / --paleo-band: {e}')
        if len(b_rows):
            band_rows, band_F = b_rows, b_F
        desc_ = lambda b_: 'no limit' if b_ is None else 'each above zero' if b_ == 'positive' else f'each within a factor {b_:g} of its target'
        print(f'slip-rate rows ({len(g_slip)}): {desc_(sl_b)}; paleo rows ({len(g_pal)}): {desc_(pa_b)}; the MFD bins: '
              + (f'each above zero' if len(g_mfd) else 'no limit')
              + ('; a start outside the limits is moved inside first' if len(pos_g) or len(b_rows) else ''))
    if a.positive and a.band is not None:
        raise SystemExit('--positive and --band: give one or the other')
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
    if a.band is not None:                         # the band: every data row with a positive target
        if not a.band > 1:
            raise SystemExit(f'--band {a.band:g}: give a factor above 1 (10: an order of magnitude either way)')
        if dirs is not None or lognormal is not None:
            raise SystemExit('--band is not implemented for transfer moves or the hierarchical prior')
        band_rows = np.flatnonzero(P['d'] > 0)
        band_F = a.band
        T.set_band(a.band, band_rows)
        print(f'band: the predictions of {len(band_rows)} of the {T.n_data} data rows (those with a positive target) kept '
              f'within a factor {a.band:g} of their targets; any start outside it is moved inside first')
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
    elif a.init == 'zero':
        X0 = np.zeros((a.chains, n))
    else:
        X0 = np.load(a.init)['final_X'] if a.init.endswith('.npz') else np.load(a.init)
        X0 = np.tile(X0, (a.chains, 1)) if X0.ndim == 1 else X0[np.arange(a.chains) % X0.shape[0]]
    if band_rows is not None:                      # a start outside the band: moved inside it first
        Gd = sp.csr_matrix(P['G'])
        y0 = (Gd @ X0.T).T
        n_out = np.sum((y0[:, band_rows] < P['d'][band_rows] / band_F) | (y0[:, band_rows] > P['d'][band_rows] * band_F), 1)
        if n_out.any():
            hi_m = T.hi * (scale if scale is not None else 1.0)
            x0 = X0[int(np.argmax(n_out))]
            # the fit is deterministic, so it is made once per problem and kept beside it: later runs (the other
            # replicates) with the same problem, band, start and bounds take it from there
            lab_ = (f'{a.band:g}' if not grp else ''.join(f'{nm}{b_:g}' for nm, b_ in (('slip', sl_b), ('paleo', pa_b))
                                                            if isinstance(b_, float)))
            cache = os.path.splitext(a.problem)[0] + f'_band{lab_}_start.npz'
            key = band_start_key(Gd, P['d'], band_rows, band_F, x0, hi_m)
            ms = None
            if os.path.exists(cache):
                with np.load(cache) as c:
                    if str(c['key']) == key:
                        ms = np.array(c['m'])
            if ms is not None:
                print(f'band: the start leaves {n_out.max()} rows outside it; the start inside it that an earlier run '
                      f'found from it is taken from {cache}')
            else:
                print(f'band: the start leaves {n_out.max()} rows outside it; a penalized least-squares fit from it:')
                ms = band_start(Gd, P['d'], band_rows, band_F, x0, hi_m)
                if ms is None:
                    raise SystemExit('band: no start inside it was found; the band may admit no model of these inputs')
                tmp = cache[:-4] + f'.{os.getpid()}.tmp.npz'
                np.savez(tmp, m=ms, key=key, band=band_F)
                os.replace(tmp, cache)               # whole or not at all, should another run read it meanwhile
                print(f'band: kept for later runs on this problem in {cache}')
            X0 = np.tile(ms, (a.chains, 1))
            Xs = T.init(X0 if scale is None else X0 / scale)
            left = T.band_outside(T.residual(Xs[:1]))[0]
            if left:                               # e.g. rates below the slab floor, switched off by the prior
                raise SystemExit(f'band: the start found leaves {left} rows outside it once the prior\'s support is applied')
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
    if grp and (a.slip_log_band is not None or a.paleo_log_band is not None):   # the soft band, by row group
        if T.pkind != 3 or dirs is not None:
            raise SystemExit('--slip-log-band and --paleo-log-band need spike-and-slab, without transfer moves')
        wt = np.ones(T.N)
        sb = list(P['block_names'])
        if a.slip_block in sb:                     # a section's k subsections share its term: each 1/k of it
            Sr = np.flatnonzero(P['block'] == sb.index(a.slip_block))
            sec = np.asarray(P['section'])[Sr]
            if (sec >= 0).all():
                ks = np.bincount(np.unique(sec, return_inverse=True)[1])
                wt[Sr] = 1.0 / ks[np.unique(sec, return_inverse=True)[1]]
        W, LD, rows_l, tau_r = np.zeros(T.N), np.zeros(T.N), [], []
        for rows_, tau_ in ((g_slip, a.slip_log_band), (g_pal, a.paleo_log_band)):
            if tau_ is not None:
                W[rows_] = wt[rows_] / (2.0 * (tau_ * LN10) ** 2)
                rows_l.append(rows_); tau_r.append(np.full(len(rows_), float(tau_)))
        rows_l, tau_r = np.concatenate(rows_l), np.concatenate(tau_r)
        LD[rows_l] = np.log(T.d[rows_l])
        T.lw = dict(W=np.ascontiguousarray(W), LD=np.ascontiguousarray(LD), DY=np.ascontiguousarray(T.d, np.float64),
                    rows=rows_l, tau=tau_r)
        y0 = (T.residual(T.init(X0[:1] if scale is None else X0[:1] / scale)) + T.d)[0, rows_l]
        pen0 = 2.0 * float(np.sum(W[rows_l] * np.log(y0 / T.d[rows_l]) ** 2)) if (y0 > 0).all() else np.inf
        print('soft band: a penalty 1/2 (log10(prediction / target) / tau)^2, in the misfit beside the data, kept by a '
              'Metropolis step, on ' + ' and '.join(f'the {nm} rows ({len(r_)}; tau {t_:g})' for nm, r_, t_ in
                                                     (('slip-rate', g_slip, a.slip_log_band), ('paleo', g_pal, a.paleo_log_band))
                                                     if t_ is not None)
              + f' (each slip-rate subsection 1/k of its section\'s); the start\'s {pen0:.1f} (on chi2\'s scale), '
              f'{int(np.sum(np.abs(np.log10(np.maximum(y0, 1e-300) / T.d[rows_l])) > tau_r))} rows beyond their tau')
    if a.log_band is not None:                     # the soft band: weights on the positive-target data rows
        if not a.positive or T.pkind != 3 or dirs is not None:
            raise SystemExit('--log-band needs --positive and spike-and-slab, without transfer moves')
        if not a.log_band > 0:
            raise SystemExit(f'--log-band {a.log_band:g}: give a width above 0, in decades')
        rows_l = np.flatnonzero(P['d'] > 0)
        wt = np.ones(T.N)
        sb = list(P['block_names'])
        if a.slip_block in sb:                     # a section's k subsections share its term: each 1/k of it
            Sr = np.flatnonzero(P['block'] == sb.index(a.slip_block))
            sec = np.asarray(P['section'])[Sr]
            if (sec >= 0).all():
                ks = np.bincount(np.unique(sec, return_inverse=True)[1])
                wt[Sr] = 1.0 / ks[np.unique(sec, return_inverse=True)[1]]
        if pm_rows is not None and a.slip_block in sb:   # with --paleo-misfit, the slip-rate rows only
            rows_l = np.intersect1d(rows_l, np.flatnonzero(P['block'] == sb.index(a.slip_block)))
        W = np.zeros(T.N)
        W[rows_l] = wt[rows_l] / (2.0 * (a.log_band * LN10) ** 2)
        LD = np.zeros(T.N)
        LD[rows_l] = np.log(T.d[rows_l])
        T.lw = dict(W=np.ascontiguousarray(W), LD=np.ascontiguousarray(LD), DY=np.ascontiguousarray(T.d, np.float64),
                    rows=rows_l, tau=float(a.log_band))
        y0 = (T.residual(T.init(X0[:1] if scale is None else X0[:1] / scale)) + T.d)[0, rows_l]
        pen0 = 2.0 * float(np.sum(W[rows_l] * np.log(y0 / T.d[rows_l]) ** 2)) if (y0 > 0).all() else np.inf
        print(f'soft band: a penalty 1/2 (log10(prediction / target) / {a.log_band:g})^2 on {len(rows_l)} rows (each slip-rate '
              f'subsection 1/k of its section\'s' + ('; the slip-rate rows only, with --paleo-misfit' if pm_rows is not None else '')
              + '), in the misfit beside the data, kept by a Metropolis step; the start\'s '
              f'{pen0:.1f} (on chi2\'s scale), {int(np.sum(np.abs(np.log10(np.maximum(y0, 1e-300) / T.d[rows_l])) > a.log_band))} rows beyond '
              f'{a.log_band:g} decades of their targets')
    if sm_rows is not None:                        # --slip-misfit: the log misfit, through the soft band's machinery
        from cru_common import geo_moments, slip_log_sds
        s_lo, s_hi, wk_ = slip_log_sds(P, sm_rows)[:3]   # each row's log10 sds, below and above: its effective sd's (1/k: a
        W, WL = np.zeros(T.N), np.zeros(T.N)           # section's copies count once)
        W[sm_rows] = b_s * wk_ / (2.0 * (s_hi * LN10) ** 2)   # 2 W (ln y - ln d)^2 = b (1/k) (log10(y / d) / s)^2, each side's s
        WL[sm_rows] = b_s * wk_ / (2.0 * (s_lo * LN10) ** 2)
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
                          F=np.ascontiguousarray(b_p * pm_tab['F']), D=np.ascontiguousarray(b_p * pm_tab['DF']),
                          K=np.ascontiguousarray(b_p * pm_tab['KAPPA']), H=pm_tab['H'], NG=pm_tab['NG'], rows=pm_rows,
                          eps=sc_p, d_data=d_data, mode=a.paleo_misfit, b=b_p)
        from paleo_misfit import site_moments_mixed     # each site's curve, and its misfit's mean and variance (fit check)
        T.lw['PM']['sd'] = np.asarray(sd_p, float)
        T.lw['PM']['th'] = np.array([f_[0] for f_ in pm_tab['fits']], float)
        T.lw['PM']['best'] = np.array([s_['best'] for s_ in sites], float)
        mom_ = np.array([site_moments_mixed(th_, b_, sd_, a.paleo_misfit, b_p)
                         for th_, b_, sd_ in zip(T.lw['PM']['th'], T.lw['PM']['best'], T.lw['PM']['sd'])])
        T.lw['PM']['E'], T.lw['PM']['V'] = mom_[:, 0], mom_[:, 1]
        y0 = (T.residual(T.init(X0[:1] if scale is None else X0[:1] / scale)) + T.d)[0, pm_rows]
        f0 = 2.0 * float(np.sum(_pm_f_np(T.lw['PM'], np.arange(len(pm_rows)), np.log(np.maximum(y0, 1e-300))))) / b_p
        print(f'paleo misfit ({a.paleo_misfit}): {len(pm_rows)} paleo rows '
              + (f'out of the Gaussian (scaled by {sc_p:g}), ' if b_p >= 1 else 'with half their Gaussian in rate (scaled by 1/sqrt 2) and half ')
              + 'each scored by -2 ln of ' + ('its recurrence-interval PDF in years' if curve_mode(a.paleo_misfit) == 'years'
                                             else 'its curve\'s density in log rate')
              + f' from its minimum, kept by a Metropolis step; the start\'s curve misfit {f0:.1f} (on chi2\'s scale, weighted {b_p:g})')
    if mf_rows is not None:                        # --mfd-misfit log10: the bins' log misfit, through the soft band's machinery
        if not (grp or a.positive):
            raise SystemExit(f'--mfd-misfit {a.mfd_misfit} with --band: give the per-group --slip-band and --paleo-band instead (it '
                             'keeps each bin above zero)')
        if T.lw is None:
            T.lw = dict(W=np.zeros(T.N), LD=np.zeros(T.N), DY=np.ascontiguousarray(T.d, np.float64),
                        rows=np.array([], np.int64), tau=None)
        T.lw['W'] = np.array(T.lw['W'], np.float64)
        T.lw['LD'] = np.array(T.lw['LD'], np.float64)
        T.lw['W'][mf_rows] = b_m / (2.0 * (mf_s * LN10) ** 2)   # 1/2 (ln(y / R) / (s ln 10))^2 = 1/2 (log10(y / R) / s)^2, x b
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
        for x in X0:                               # a start that breaks it: each offending bin scaled down to its neighbour
            for _ in range(len(mono['pairs']) + 1):
                tot = np.array([x[m].sum() for m in mono['members']])
                bad = [(k, k1, rho) for (k, k1), rho in zip(mono['pairs'], mono['rhos']) if tot[k1] > rho * tot[k]]
                if not bad:
                    break
                k, k1, rho = bad[0]
                x[mono['members'][k1]] *= rho * tot[k] / tot[k1] * (1 - 1e-9)
                fixed += 1
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
                keep_draws=a.keep_draws)
    print(f'done in {tr["seconds"]:.0f}s -> {a.out}.npz, {a.out}_stream.npz' + ('' if a.no_samples else f', {a.out}_samples.npy')
          + (f', {a.out}_kept.npy (and _kept_index.npy)' if a.keep_draws > 0 else '')
          + (f'; swap acceptance {tr["swap_rate"].mean():.3f}' if pairs is not None else '')
          + (f'; transfer moves {tr["transfer_stats"][:, 0].sum()}, of which involving a rate > 1e-6: '
             f'{tr["transfer_stats"][:, 1].sum()}' if dirs is not None else ''))
    report(tr, a.burn, a.thin)


if __name__ == '__main__':
    main()

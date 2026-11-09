#!/usr/bin/env python3
"""timeframe_sliprates.py -- each fault's slip rate corrected for its slip-rate timeframe (Coffey et al., accepted,
BSSA: "Reassessing geologic fault slip rates and their uncertainties in Aotearoa New Zealand"), rebuilt in full.

    python timeframe_sliprates.py --cfm NZ_CFM_v1_0.csv --s4 SuppTS4_IFM_SR_updated_Jul2026.csv --rupture-set DIR
                                  [--out ../data/cfm_timeframe_sliprates.csv]

The paper widens each CFM slip rate for the few earthquakes a slip rate measured over a short timeframe captures (after
Styron, 2019): a Gaussian fitted to the CFM's bounds (centred at the midpoint of SR_min and SR_max, sd their width /
3.92, negative draws at 0) times a factor drawn from the distribution of true / measured slip rate for a reference fault
(1 mm/yr, a 1000-year recurrence) at the fault's timeframe scaled by its activity, t = timeframe x 1000 / recurrence
interval (its Tables 1 and 2: 7 quantiles at 8 timeframes, for periodic faults, SR_pref >= 1 mm/yr, and aperiodic ones).
Its Table S4 reports each fault's result, but as bounds never narrower than the CFM's (each lower bound at most SR_min,
each upper at least SR_max), so its 25% and 75% columns are mostly the CFM's bounds and many 2.5% points are pinned
to SR_min or 0; its medians and 97.5% points are the distribution's.

This script rebuilds each fault's whole distribution exactly, by quadrature over the factor's quantile function (no
sampling: Corrected): the tables' quantiles interpolated in log t between their rows (held at their ends) and, across
probability, against the normal score, both linearly and by a monotone cubic (two schemes, for redundancy: the tables fix
only 7 points of each distribution); for each scheme t is solved so that the 97.5% point (with S4's convention) matches
S4's, the one number S4 always gives as is, and the faults S4 does not list take the median t of the faults of the same
kind (periodic or not). Each fault's mode, the maximum of the density of the corrected slip rate (the maximum-likelihood
slip rate after the correction), and its quantiles are the two schemes' geometric means; both modes are kept, and the
faults where they differ by more than 2% reported. The checks against S4 that do not enter the fit (its medians, its 2.5%
points where not pinned) are reported.

The likelihood scores each slip rate by a split normal (two half-Gaussians joined at its mode). Since 2026-11-09 each
fault's is fitted to its corrected distribution (fit_split_normal): the mode and two sds whose 16%, 50% and 84% points
are the corrected distribution's, so the split normal reproduces its median and body exactly and its mean closely.
Before, the split normal had its mode at the corrected mode and its 2.5% and 97.5% points read as 2 sd from it ('ends'),
which, for these right-skewed distributions with long tails both ways, put its median and mean a median 7% above the
corrected distribution's (up to about 2x). The fitted centre can lie below the corrected mode, and its 2 sd below can lie
below 0. No split normal in slip rate has the most right-skewed faults' 16/50/84 points (its (q84 - q50) / (q50 - q16) is
at most about 1.55); those, mostly slow aperiodic faults (188 of the 211k rupture set's 557), take a split normal fitted in
log10 of the slip rate and converted (split_normal_for: 'log 16/50/84', else 'log 2.5/50/97.5'), automatically, each
fault's fit named in sn_fit and counted in the report; 'ends' (the old split normal) only where a 16% or 2.5% point is 0.

Writes OUT (one row per fault of the CFM: Name, mode, q025, q16, q50, q84, q975 in mm/yr, t_eff in
years (the schemes' geometric mean), fit: 'S4' (t solved), 'S4 long' (S4's 97.5% at or inside the tables' longest
timeframe's, t held there), or 'assigned'; periodic; SR_pref, SR_min, SR_max; mode_linear, mode_pchip; sn_centre,
sn_sd_lo, sn_sd_hi, the fitted split normal's mode and sds in mm/yr, and sn_fit, which fit gave it), OUT's
_report.txt and a figure (OUT's .png) comparing each fault's new uncertainty with the CFM's times --multiplier (3, the
notebook's SLIP_SD_MULTIPLIER before CFM_TIMEFRAME). preliminaries.py --cfm-timeframe OUT uses the fitted split normals
(--cfm-timeframe-shape fit, the default) or, with --cfm-timeframe-shape ends, mode, q025 and q975 in place of SR_pref,
SR_min and SR_max (as before 2026-11-09).
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.stats import norm

from cru_common import CFM_CSV, report_to

TQ = np.array([0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])      # the tables' quantiles
TT = np.array([500, 1000, 2000, 5000, 10000, 20000, 50000, 100000], float)   # their timeframes, years
TAB = {True: np.array([                  # Table 1: periodic faults (SR_pref >= 1 mm/yr); rows TT, columns TQ
    [0.34, 0.4176, 0.7778, 1.2304, 2.1031, 4.9631, 6.0027],
    [0.3731, 0.4492, 0.779, 1.1681, 1.904, 4.3238, 5.4015],
    [0.4359, 0.5134, 0.7888, 1.0946, 1.6506, 3.2103, 4.1612],
    [0.5443, 0.6123, 0.8281, 1.0438, 1.3532, 2.1038, 2.4051],
    [0.6484, 0.6899, 0.8619, 1.023, 1.217, 1.6143, 1.7762],
    [0.7367, 0.7782, 0.9028, 1.0102, 1.1369, 1.3495, 1.4402],
    [0.8252, 0.848, 0.9387, 0.9992, 1.0827, 1.2186, 1.2554],
    [0.8646, 0.8816, 0.9486, 1.0081, 1.0534, 1.1605, 1.1727]]),
    False: np.array([                    # Table 2: aperiodic faults (SR_pref < 1 mm/yr)
    [0.2316, 0.3014, 0.8097, 1.7175, 3.8223, 10.866, 16.2991],
    [0.277, 0.3402, 0.7363, 1.5019, 3.4848, 10.6151, 14.8125],
    [0.3421, 0.407, 0.7117, 1.2546, 2.7253, 8.9414, 13.3486],
    [0.4406, 0.5089, 0.7512, 1.0759, 1.7596, 5.2667, 7.6325],
    [0.5228, 0.5892, 0.8074, 1.0253, 1.4754, 2.8358, 3.4383],
    [0.6118, 0.6613, 0.852, 1.0195, 1.3025, 2.0205, 2.387],
    [0.7098, 0.7514, 0.9062, 1.0326, 1.1616, 1.5123, 1.7176],
    [0.7853, 0.8156, 0.9338, 1.0251, 1.1158, 1.2966, 1.472]])}
ZQ = norm.ppf(TQ)
NU = 4000                                                        # quadrature points in the factor's probability
U = (np.arange(NU) + 0.5) / NU
Z = norm.ppf(U)
SCHEMES = ('linear', 'pchip')                                    # the factor's quantile function between the tables' points


def log_factor(periodic, t, scheme):
    """log of the timeframe factor (true / measured slip rate) at the quadrature points U, at scaled timeframe t: the tables'
    7 quantiles interpolated in log t between their rows (held at their ends), then across probability against the normal
    score, linearly ('linear', continued beyond the 2.5% and 97.5% points with the end segments' slopes) or by a monotone
    cubic ('pchip', extrapolated by its end pieces)."""
    lt = np.log(np.clip(t, TT[0], TT[-1]))
    L = np.log(TAB[bool(periodic)])
    lq = np.array([np.interp(lt, np.log(TT), L[:, k]) for k in range(len(TQ))])
    if scheme == 'pchip':
        return PchipInterpolator(ZQ, lq, extrapolate=True)(Z)
    out = np.interp(Z, ZQ, lq)
    lo, hi = Z < ZQ[0], Z > ZQ[-1]
    out[lo] = lq[0] + (Z[lo] - ZQ[0]) * (lq[1] - lq[0]) / (ZQ[1] - ZQ[0])
    out[hi] = lq[-1] + (Z[hi] - ZQ[-1]) * (lq[-1] - lq[-2]) / (ZQ[-1] - ZQ[-2])
    return out


class Corrected:
    """One fault's corrected slip rate, Y = max(X, 0) F, X ~ N(mu, sd) the paper's Gaussian on the CFM bounds and F the
    timeframe factor, computed exactly by quadrature over F's quantile function (no sampling): its CDF
    P(Y <= y) = mean_u Phi((y / F(u) - mu) / sd) for y >= 0 (the draws of X below 0 at 0), its quantiles by root finding,
    the density of its continuous part, mean_u phi((y / F(u) - mu) / sd) / (sd F(u)), and that density's maximum, the mode."""

    def __init__(self, pref, smin, smax, t, scheme):
        self.mu, self.sd = 0.5 * (smin + smax), (smax - smin) / 3.92
        self.q = np.exp(log_factor(pref >= 1.0, t, scheme))
        self.zero = self.mu <= 0 and self.sd <= 0                # no slip rate at all

    def cdf(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        if self.sd <= 0:                                          # a CFM range of zero width: Y = mu F
            return (y[:, None] >= self.mu * self.q[None, :]).mean(1)
        return norm.cdf((y[:, None] / self.q[None, :] - self.mu) / self.sd).mean(1)

    def quantile(self, p):
        if self.zero or self.cdf(0.0)[0] >= p:
            return 0.0
        hi = max(self.mu + 8 * self.sd, 1e-12) * self.q.max()
        return float(brentq(lambda y: self.cdf(y)[0] - p, 0.0, hi, xtol=1e-12 * hi, rtol=1e-12))

    def density(self, y):
        y = np.atleast_1d(np.asarray(y, float))
        return (norm.pdf((y[:, None] / self.q[None, :] - self.mu) / self.sd) / (self.sd * self.q[None, :])).mean(1)

    def mode(self):
        if self.zero:
            return 0.0
        if self.sd <= 0:                                          # Y = mu F: F's own mode, where dQ/du is least
            dq = np.gradient(self.q, U)
            return float(self.mu * self.q[int(np.argmin(dq[NU // 50: -NU // 50])) + NU // 50])
        hi = (self.mu + 4 * self.sd) * np.percentile(self.q, 99)
        y = np.linspace(hi * 1e-5, hi, 800)
        k = int(np.argmax(self.density(y)))
        r = minimize_scalar(lambda v: -self.density(v)[0], bounds=(y[max(k - 1, 0)], y[min(k + 1, len(y) - 1)]),
                            method='bounded', options=dict(xatol=1e-10 * hi))
        return float(r.x)

    def p0(self):
        return float(norm.cdf(-self.mu / self.sd)) if self.sd > 0 else float(self.mu <= 0)


def sn_quantile(c, s_lo, s_hi, p):
    """The p-quantile of a split normal with mode c and sds s_lo below it and s_hi above (its mass below c is
    a = s_lo / (s_lo + s_hi))."""
    a = s_lo / (s_lo + s_hi)
    p = np.asarray(p, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(p < a, c + s_lo * norm.ppf(np.clip(p / (2 * a), 1e-300, 1.0)),
                        c + s_hi * norm.ppf(np.clip(0.5 + (p - a) / (2 * (1 - a)), 0.0, 1 - 1e-16)))


def sn_mean(c, s_lo, s_hi):
    return c + np.sqrt(2 / np.pi) * (s_hi - s_lo)


def fit_quantiles(q, p):
    """The split normal (mode, s_lo, s_hi) whose p[0], p[1], p[2] points are q[0] < q[1] < q[2] (any real values; least
    squares in the mode and the logs of the sds, scaled by q[2] - q[0]), or None if no split normal has them: a split
    normal's (q84 - q50) / (q50 - q16) is at most about 1.55, its half-normal limit, so the most skewed shapes have none."""
    q, p = np.asarray(q, float), np.asarray(p, float)
    if not (q[0] < q[1] < q[2]):
        return None
    w = q[2] - q[0]

    def res(x):
        return (sn_quantile(q[1] + w * x[0], w * np.exp(x[1]), w * np.exp(x[2]), p) - q) / w
    r = least_squares(res, [0.0, np.log((q[1] - q[0]) / w), np.log((q[2] - q[1]) / w)], xtol=1e-14, ftol=1e-14, gtol=1e-14)
    if np.max(np.abs(r.fun)) > 1e-8:
        return None
    return float(q[1] + w * r.x[0]), float(w * np.exp(r.x[1])), float(w * np.exp(r.x[2]))


def fit_split_normal(q16, q50, q84):
    """The split normal in slip rate whose 16%, 50% and 84% points are q16, q50 and q84, or None (q16 <= 0, or none has
    them)."""
    return fit_quantiles([q16, q50, q84], [0.16, 0.5, 0.84]) if q16 > 0 else None


def fit_split_normal_log(q, p):
    """A split normal in log10 of the slip rate fitted to quantiles q (> 0) at probabilities p, converted to one in slip
    rate: mode 10^c, each sd reaching the log fit's 1-sd point on its side (10^c - 10^(c - s_lo), 10^(c + s_hi) - 10^c).
    Returns ((mode, s_lo, s_hi) in slip rate, (c, s_lo, s_hi) in log10), or None."""
    q = np.asarray(q, float)
    if not (q > 0).all():
        return None
    f = fit_quantiles(np.log10(q), p)
    if f is None:
        return None
    c, lo, hi = f
    return (10 ** c, 10 ** c - 10 ** (c - lo), 10 ** (c + hi) - 10 ** c), f


SN_FITS = ('16/50/84', 'log 16/50/84', 'log 2.5/50/97.5', 'ends')   # the split normal each fault gets, in order of trial


def split_normal_for(r):
    """The likelihood's split normal for a fault (row r of the table: its quantiles and mode), and which fit gave it:
    '16/50/84', fitted to the corrected distribution's 16%, 50% and 84% points (its median and body exactly); where no
    split normal in slip rate has those (the most right-skewed, mostly slow faults), 'log 16/50/84', fitted to them in
    log10 of the slip rate and converted; where none does even in log10, 'log 2.5/50/97.5', likewise to the 2.5%, 50% and
    97.5% points; else (a 16% or 2.5% point of 0) 'ends', the mode with the 2.5% and 97.5% points at 2 sd, as before
    2026-11-09. The fallbacks are taken automatically, so a new table needs no checking by hand: the report counts each."""
    sn = fit_split_normal(r['q16'], r['q50'], r['q84'])
    if sn is not None:
        return sn, SN_FITS[0]
    for q, p, how in (([r['q16'], r['q50'], r['q84']], [0.16, 0.5, 0.84], SN_FITS[1]),
                      ([r['q025'], r['q50'], r['q975']], [0.025, 0.5, 0.975], SN_FITS[2])):
        f = fit_split_normal_log(q, p)
        if f is not None and f[0][1] > 0 and f[0][2] > 0:
            return f[0], how
    return (r['mode'], 0.5 * (r['mode'] - r['q025']), 0.5 * (r['q975'] - r['mode'])), SN_FITS[3]


def exact_mean(pref, smin, smax, t, scheme):
    """The corrected slip rate's mean, E[max(X, 0)] E[F] (X and F independent)."""
    mu, sd = 0.5 * (smin + smax), (smax - smin) / 3.92
    ex = mu * norm.cdf(mu / sd) + sd * norm.pdf(mu / sd) if sd > 0 else max(mu, 0.0)
    return float(ex * np.exp(log_factor(pref >= 1.0, t, scheme)).mean())


def s4_q975(pref, smin, smax, t, scheme):
    """The 97.5% point with S4's convention (never below SR_max)."""
    return max(Corrected(pref, smin, smax, t, scheme).quantile(0.975), smax)


def fit_t(pref, smin, smax, target, scheme):
    """t such that the 97.5% point (S4's convention) matches S4's; bisection in log t over the tables' range.
    Returns (t, 'S4', 'S4 long' or 'S4 short')."""
    lo, hi = np.log(TT[0]), np.log(TT[-1])
    if target <= s4_q975(pref, smin, smax, TT[-1], scheme):
        return float(TT[-1]), 'S4 long'
    if target >= s4_q975(pref, smin, smax, TT[0], scheme):
        return float(TT[0]), 'S4 short'
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if s4_q975(pref, smin, smax, np.exp(m), scheme) > target:
            lo = m
        else:
            hi = m
    return float(np.exp(0.5 * (lo + hi))), 'S4'


def load_cfm(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return {r['Name'].strip(): (float(r['SR_pref']), float(r['SR_min']), float(r['SR_max']), r.get('QualCode', ''))
                for r in csv.DictReader(f)}


def load_s4(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))[1:]
    return {r[0].strip(): np.array([float(v) for v in r[1:6]]) for r in rows if r and r[0].strip()}   # med 2.5 25 75 97.5


def figure(rows, mult, out_png):
    rows = [r for r in rows if r['pref'] > 0 and r['mode'] > 0 and r['fit'] != 'assigned']   # the faults S4 lists
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, ink2, muted, grid_c, c1, c2 = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#2c6fbb', '#d04a2b'
    pref = np.array([r['pref'] for r in rows]); smin = np.array([r['smin'] for r in rows]); smax = np.array([r['smax'] for r in rows])
    mode = np.array([r['mode'] for r in rows]); lo = np.array([r['q025'] for r in rows]); hi = np.array([r['q975'] for r in rows])
    cur_lo, cur_hi = mult * (pref - smin), mult * (smax - pref)          # the current sides (2 sigma), before the factor 0.8
    new_lo, new_hi = mode - lo, hi - mode
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), facecolor='#fcfcfb')
    for a in ax:
        a.set_facecolor('#fcfcfb'); a.grid(color=grid_c, lw=0.6); [a.spines[s].set_color(muted) for s in a.spines]
    with np.errstate(divide='ignore', invalid='ignore'):
        rl, rh = new_lo / cur_lo, new_hi / cur_hi
    ok_l, ok_h = np.isfinite(rl) & (rl > 0), np.isfinite(rh) & (rh > 0)
    bins = np.logspace(-1.5, 1.5, 49)
    ax[0].hist(rl[ok_l], bins, color=c1, alpha=0.75, label=f'below the centre (median {np.median(rl[ok_l]):.2f})')
    ax[0].hist(rh[ok_h], bins, color=c2, alpha=0.6, label=f'above the centre (median {np.median(rh[ok_h]):.2f})')
    ax[0].axvline(1, color=ink, lw=1); ax[0].set_xscale('log')
    ax[0].set_xlabel(f'new 2-sigma side / current (CFM side x {mult:g})', color=ink2); ax[0].set_ylabel('faults', color=ink2)
    ax[0].set_title('Each side\'s width, new over current', loc='left', color=ink, fontsize=11)
    ax[0].legend(fontsize=8.5, frameon=False, labelcolor=ink2)
    ax[1].scatter(pref, mode, s=9, color=c1, alpha=0.7, edgecolors='none')
    lim = [min(pref.min(), mode.min()) / 1.5, max(pref.max(), mode.max()) * 1.5]
    ax[1].plot(lim, lim, color=ink, lw=1); ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel('CFM SR_pref (mm/yr)', color=ink2); ax[1].set_ylabel('mode of the corrected distribution (mm/yr)', color=ink2)
    ax[1].set_title('Centres', loc='left', color=ink, fontsize=11)
    for i in np.argsort(-np.abs(np.log(mode / pref)))[:5]:
        ax[1].annotate(rows[i]['name'], (pref[i], mode[i]), xytext=(4, -3), textcoords='offset points', fontsize=7.5, color=ink2)
    tot_c, tot_n = cur_lo + cur_hi, new_lo + new_hi
    order = np.argsort(-(tot_n / np.where(tot_c > 0, tot_c, np.inf)))[:18]
    yy = np.arange(len(order))[::-1]
    for y, i in zip(yy, order):
        ax[2].plot([max(pref[i] - cur_lo[i], 1e-3), pref[i] + cur_hi[i]], [y + 0.15] * 2, color=muted, lw=2.2)
        ax[2].plot([max(lo[i], 1e-3), hi[i]], [y - 0.15] * 2, color=c2, lw=2.2)
        ax[2].plot(pref[i], y + 0.15, 'o', color=ink2, ms=3.5); ax[2].plot(mode[i], y - 0.15, 'o', color=c2, ms=3.5)
    ax[2].set_yticks(yy); ax[2].set_yticklabels([rows[i]['name'] for i in order], fontsize=7.5, color=ink2)
    ax[2].set_xscale('log'); ax[2].set_xlabel('slip rate (mm/yr), before the slip-rate factor', color=ink2)
    ax[2].set_title(f'Most widened: now (grey, CFM x{mult:g}), new (red)', loc='left', color=ink, fontsize=11)
    fig.suptitle('Slip-rate uncertainties corrected for timeframe (Coffey et al.), against the CFM\'s x ' + f'{mult:g}',
                 x=0.01, ha='left', color=ink, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=140, facecolor='#fcfcfb')


def main():
    ap = argparse.ArgumentParser(description='Slip rates corrected for their timeframe, rebuilt from Coffey et al.\'s S4.')
    ap.add_argument('--cfm', default=CFM_CSV, help='the CFM as CSV (default: data/NZ_CFM_v1_0.csv, cru_common.CFM_CSV)')
    ap.add_argument('--s4', required=True, help='the paper\'s Table S4 (SuppTS4_IFM_SR_updated_Jul2026.csv)')
    ap.add_argument('--rupture-set', default=None, help='an OpenSHA rupture set, to report how many of its parent faults S4 covers')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data',
                                                  'cfm_timeframe_sliprates.csv'))
    ap.add_argument('--multiplier', type=float, default=3.0, help='the current uniform multiplier, for the comparison (3)')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stem = os.path.splitext(out)[0]
    report_to(stem + '_report.txt')
    C, S4 = load_cfm(a.cfm), load_s4(a.s4)
    parents = sorted(C)                                                   # every fault of the CFM
    rs_par = None
    if a.rupture_set:
        g = json.load(open(os.path.join(a.rupture_set, 'fault_sections.geojson')))
        rs_par = sorted({f['properties']['ParentName'] for f in g['features']})
        miss = [p for p in rs_par if p not in C]
        if miss:
            raise SystemExit(f'{len(miss)} parent faults of the rupture set not in {a.cfm}: {miss[:5]}')
    rows = []
    for p in parents:
        pref, smin, smax, qc = C[p]
        r = dict(name=p, pref=pref, smin=smin, smax=smax, qc=qc, periodic=pref >= 1.0, t={}, fit=None)
        if p in S4 and smax > smin:
            for sc in SCHEMES:
                r['t'][sc], r['fit'] = fit_t(pref, smin, smax, S4[p][4], sc)
        rows.append(r)
    for per in (True, False):                                              # faults S4 does not list
        tm = {sc: float(np.exp(np.median([np.log(r['t'][sc]) for r in rows if r['t'] and r['periodic'] == per])))
              for sc in SCHEMES}
        for r in rows:
            if not r['t'] and r['periodic'] == per:
                r['t'], r['fit'] = dict(tm), 'assigned'
    gm = lambda v: float(np.exp(np.mean(np.log(v)))) if min(v) > 0 else 0.0   # the schemes' geometric mean
    chk = []
    for r in rows:
        per_sc = {}
        for sc in SCHEMES:
            c = Corrected(r['pref'], r['smin'], r['smax'], r['t'][sc], sc)
            per_sc[sc] = dict(mode=c.mode(), q=[c.quantile(p_) for p_ in (0.025, 0.16, 0.5, 0.84, 0.975)], p0=c.p0())
        r['mode'] = gm([per_sc[sc]['mode'] for sc in SCHEMES])
        r['q025'], r['q16'], r['q50'], r['q84'], r['q975'] = (gm([per_sc[sc]['q'][k] for sc in SCHEMES]) for k in range(5))
        r['mode_sc'] = {sc: per_sc[sc]['mode'] for sc in SCHEMES}
        r['t_eff'] = gm([r['t'][sc] for sc in SCHEMES])
        r['p0'] = per_sc[SCHEMES[0]]['p0']
        r['sn'], r['sn_fit'] = split_normal_for(r)                 # the split normal the likelihood uses
        r['mean'] = gm([exact_mean(r['pref'], r['smin'], r['smax'], r['t'][sc], sc) for sc in SCHEMES])
        if r['name'] in S4 and r['smax'] > r['smin']:
            s = S4[r['name']]
            lo, med, hi = min(r['q025'], r['smin']), r['q50'], max(r['q975'], r['smax'])
            chk.append((r['fit'], med / s[0], hi / s[4], (lo / s[1]) if (0 < s[1] < r['smin']) else np.nan))
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Name', 'mode', 'q025', 'q16', 'q50', 'q84', 'q975', 't_eff', 'fit', 'periodic', 'SR_pref', 'SR_min', 'SR_max']
                   + [f'mode_{sc}' for sc in SCHEMES] + ['sn_centre', 'sn_sd_lo', 'sn_sd_hi', 'sn_fit'])
        for r in rows:
            w.writerow([r['name']] + [f'{r[k]:.6g}' for k in ('mode', 'q025', 'q16', 'q50', 'q84', 'q975', 't_eff')]
                       + [r['fit'] or '', int(r['periodic']), r['pref'], r['smin'], r['smax']]
                       + [f'{r["mode_sc"][sc]:.6g}' for sc in SCHEMES] + [f'{v:.6g}' for v in r['sn']] + [r['sn_fit']])
    figure(rows, a.multiplier, stem + '.png')
    # report
    fits = np.array([c[0] for c in chk]); med_r = np.array([c[1] for c in chk]); hi_r = np.array([c[2] for c in chk])
    lo_r = np.array([c[3] for c in chk])
    print(f'timeframe-corrected slip rates (Coffey et al., Table S4, rebuilt) for the {len(rows)} faults of the CFM: '
          f'{sum(r["fit"] == "S4" for r in rows)} with t solved from S4, '
          f'{sum(r["fit"] == "S4 long" for r in rows)} held at the tables\' longest timeframe (100,000 years; S4\'s 97.5% point at '
          f'or inside it), {sum(r["fit"] == "S4 short" for r in rows)} at the shortest, {sum(r["fit"] == "assigned" for r in rows)} '
          f'not in S4 (t the median of their kind\'s)' + (f'; of the rupture set\'s {len(rs_par)} parent faults, '
          f'{sum(p in S4 for p in rs_par)} are in S4, the others assigned: '
          + ', '.join(p for p in rs_par if p not in S4) if rs_par else ''))
    print(f'   against S4: medians {np.median(med_r):.3f} of S4\'s (5-95% {np.percentile(med_r, 5):.3f}-{np.percentile(med_r, 95):.3f}); '
          f'97.5% points {np.median(hi_r):.3f} (largest {hi_r.max():.3f}, at the long end); 2.5% points where S4\'s are not '
          f'pinned ({int(np.isfinite(lo_r).sum())} faults) {np.nanmedian(lo_r):.3f} (5-95% {np.nanpercentile(lo_r, 5):.2f}-'
          f'{np.nanpercentile(lo_r, 95):.2f}); the medians and 2.5% points do not enter the fit')
    pos = [r for r in rows if r['pref'] > 0 and r['mode'] > 0 and r['fit'] != 'assigned']   # below: the faults S4 lists
    zero = [r['name'] for r in rows if r['smax'] <= 0]
    mode = np.array([r['mode'] for r in pos]); pref = np.array([r['pref'] for r in pos])
    print(f'   below, the {len(pos)} faults S4 lists with SR_pref above 0; the CFM\'s {len(zero)} faults with no slip rate (SR_max 0) '
          f'have mode and quantiles 0, and those with SR_pref 0 but a range ' + ', '.join(f'{r["name"]} ({r["mode"]:.3g})' for r in rows
          if r['pref'] <= 0 < r['smax']) + ' a positive mode')
    print(f'   modes against SR_pref: median ratio {np.median(mode / pref):.3f} (5-95% {np.percentile(mode / pref, 5):.2f}-'
          f'{np.percentile(mode / pref, 95):.2f}); the paper centres each fault\'s Gaussian at the midpoint of its CFM bounds, so '
          'faults with lopsided bounds move most: ' + ', '.join(f'{pos[i]["name"]} {pref[i]:g} -> {mode[i]:.3g}'
                                                              for i in np.argsort(-np.abs(np.log(mode / pref)))[:5]))
    print(f'   mass at 0 (the Gaussian below 0): in {sum(r["p0"] > 1e-12 for r in pos)} faults, at most {max(r["p0"] for r in pos):.1%}')
    dm = np.array([abs(r['mode_sc']['pchip'] / r['mode_sc']['linear'] - 1) for r in pos if r['mode_sc']['linear'] > 0])
    big = sorted(((abs(r['mode_sc']['pchip'] / r['mode_sc']['linear'] - 1), r['name']) for r in pos if r['mode_sc']['linear'] > 0),
                 reverse=True)
    print(f'   the mode under the two interpolations of the factor (linear and monotone cubic against the normal score): they '
          f'differ by a median {np.median(dm):.2%} (95th percentile {np.percentile(dm, 95):.2%}); the table\'s mode and quantiles '
          f'are their geometric mean; over 2% apart: ' + (', '.join(f'{n} ({d:.1%})' for d, n in big if d > 0.02) or 'none'))
    w95n = np.array([r['q975'] - r['q025'] for r in pos]); w95c = np.array([r['smax'] - r['smin'] for r in pos])
    ok = w95c > 0
    print(f'   95% width / the CFM\'s: median {np.median(w95n[ok] / w95c[ok]):.2f}, mean {np.mean(w95n[ok] / w95c[ok]):.2f} (the '
          f'paper\'s averages, from its pinned bounds: 2.5 periodic, 2.9 aperiodic); against the CFM\'s x {a.multiplier:g} (the '
          f'notebook\'s SLIP_SD_MULTIPLIER until CFM_TIMEFRAME): median {np.median(w95n[ok] / (a.multiplier * w95c[ok])):.2f}')
    pct = lambda v: f'median {np.median(v):.3f} (5-95% {np.percentile(v, 5):.2f}-{np.percentile(v, 95):.2f})'
    rs_set = set(rs_par or [])
    print(f'   the likelihood\'s split normals (split_normal_for), by fit: ' + '; '.join(
        f'{k} {sum(r["sn_fit"] == k for r in rows)} of the {len(rows)} faults'
        + (f' ({sum(r["sn_fit"] == k for r in rows if r["name"] in rs_set)} of the rupture set\'s {len(rs_set)})' if rs_set else '')
        for k in SN_FITS) + ' (16/50/84: the corrected distribution\'s 16%, 50% and 84% points exactly; log: fitted in log10 '
          'and converted, where no split normal in slip rate has those points; ends: the mode with the 2.5% and 97.5% points '
          'at 2 sd, as before 2026-11-09, where a 16% or 2.5% point is 0)')
    for k in SN_FITS:
        ff = [r for r in pos if r['sn_fit'] == k]
        if not ff:
            continue
        c_, lo_, hi_ = (np.array([r['sn'][j] for r in ff]) for j in range(3))
        md_, q50_, mean_ = (np.array([r[j] for r in ff]) for j in ('mode', 'q50', 'mean'))
        med_ = np.array([float(sn_quantile(*r['sn'], 0.5)) for r in ff])
        old_mean = np.array([sn_mean(r['mode'], 0.5 * (r['mode'] - r['q025']), 0.5 * (r['q975'] - r['mode'])) for r in ff]) / mean_
        print(f'      {k} ({len(ff)} of the {len(pos)} faults above): centre / corrected mode {pct(c_ / md_)}; lower sd / centre '
              f'{pct(lo_ / c_)}, upper / lower {pct(hi_ / lo_)}; the split normal\'s median / the corrected median {pct(med_ / q50_)}, '
              f'its mean / the corrected mean {pct(sn_mean(c_, lo_, hi_) / mean_)} (the mode with the 2.5%/97.5% points at 2 sd, '
              f'as before: {pct(old_mean)}); 2 sd below the centre under 0: {int(np.sum(c_ - 2 * lo_ < 0))}')
    print(f'   wrote {out}, {stem}.png and {stem}_report.txt')


if __name__ == '__main__':
    main()

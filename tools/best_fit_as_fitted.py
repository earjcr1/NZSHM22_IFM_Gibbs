#!/usr/bin/env python3
"""best_fit_as_fitted.py -- the best fit under the likelihood as the sampler scores the data (chi2 as fitted).

    python best_fit_as_fitted.py --problem cru_clean.npz [--paleo-misfit years --rupture-set DIR] [--mfd-misfit log10]
                                 [--bins-block MFD] [--start nnls/nnls_vectors.npz] [--out nnls/fitted_vectors]

The best nonnegative fit (preliminaries.py, Lawson-Hanson NNLS) minimizes the Gaussian chi2: every row a Gaussian, the
CFM's slip rates with sigma_ref on both sides, the paleo rows around bestEstimate, the MFD bins in rate. The sampler's
likelihood scores some groups otherwise (gibbs.py): the slip rates by their split normals (each side's own sd), the
paleo rows by -2 ln of each site's recurrence-interval PDF (--paleo-misfit years; paleo_misfit.site_misfit), the bins
in log10 (--mfd-misfit log10), the slip rates in log10 (--slip-misfit log10), or each group half and half (split+log10,
gaussian+log10, gaussian+years: half its Gaussian plus half its log misfit, the notebook's MISFIT 'half'). This is the fit that minimizes that misfit, chi2 as fitted, over nonnegative rates:
L-BFGS-B from the NNLS fit, each rate at least --lower (1e-20 by default, far below the slab floor, so that every
prediction stays above zero, where the paleo and log10 misfits are finite; the coefficients are nonnegative). The start
is the NNLS fit with its rates below the slab floor (--floor) at --lower and, where a row with a log misfit is then
predicted at 0, ruptures crossing it switched on at twice the floor, as gibbs.py's positive start does (positive_start):
from the NNLS fit itself such rows sit tens of decades below their targets, and L-BFGS-B's line search stalls.

With the sampler's other terms and constraints given as gibbs.py takes them (the user's request, 2026-11-17: the best fit
in exactly the sampler's model space but for the prior), the fit also counts the balance term (--down-the-middle) and the
summed slip rates (--slip-sum, --slip-sum-l0, --cross-sections) as the Gaussian rows they are, and keeps the monotonic MFD
(--mfd-monotonic, --mfd-rise-sd; cru_common.mfd_monotonic, the same pairs) and each strip's share of the plate rate
(--plate-rate; plate_rate.plate_rows) within their limits, by an augmented Lagrangian around L-BFGS-B (each constraint
normalized: a monotonic pair by the lower bin's target, a strip's share as it is) to --con-tol, then scaled exactly inside
them as gibbs.py scales a start. No prior: not the slab, its floor and soft top, nor the fractions on; the rates below the
slab floor are still at --lower (gibbs.py switches them off when it starts from the fit).

Writes OUT.npz (mean: the rates, one per rupture of the original problem, 0 for the ruptures cleaning removed;
penalised; model 'best fit as fitted'), as nnls_vectors.npz is written, so the fit figures draw it beside the NNLS fit,
OUT.csv, and a report: the misfit by row block at the start and at the end, nonzero rates, the total rate.
"""
import argparse
import hashlib
import os
import time

import numpy as np
from scipy.optimize import minimize

from cru_common import load_problem, misfit_log_weight, report_to, slip_log_sds, split_weights

LN10 = np.log(10.0)


def paleo_rows(P, rupture_set, mode):
    """The paleo rows (cleaned indices), each site's curve and its sd, as gibbs.py --paleo-misfit finds them: the inputs'
    last rows, one per site of the rupture set's paleo_constraint_data.json, targets bestEstimate / sd."""
    from cru_common import find_rupture_set
    from paleo_misfit import fit_site, load_sites
    rd = find_rupture_set(rupture_set, search=False)
    if rd is None:
        raise SystemExit(f'--rupture-set {rupture_set}: no OpenSHA rupture set there')
    sites = load_sites(rd)
    nraw = int(P['raw_shape'][0])
    where = {int(o): i for i, o in enumerate(np.asarray(P['keep_rows']))}
    orig = np.arange(nraw - len(sites), nraw)
    if not all(int(o) in where for o in orig):
        raise SystemExit(f'--paleo-misfit: the last {len(sites)} rows of the inputs are not all in the cleaned problem')
    rows = np.array([where[int(o)] for o in orig])
    if not np.allclose(np.asarray(P['d'])[rows], [s['best'] / s['sd'] for s in sites], rtol=1e-9, atol=0):
        raise SystemExit("--paleo-misfit: the paleo rows' targets are not bestEstimate / sd of paleo_constraint_data.json")
    from paleo_misfit import curve_mode
    fits = [fit_site(s['best'], s['q'], curve_mode(mode)) for s in sites]
    if not all(f[2] for f in fits):
        raise SystemExit('--paleo-misfit: some sites\' curves could not be fitted')
    return rows, [f[0] for f in fits], np.array([s['best'] for s in sites]), np.array([s['sd'] for s in sites])


def positive_start(G, rows, m, floor, lower):
    """The start: the NNLS rates m with those below the slab floor at `lower`, then, for the rows (those with a log misfit)
    the rates at or above the floor predict at 0, ruptures crossing them switched on at twice the floor, each time the one
    crossing the most such rows (ties: the lowest index), as gibbs.py's positive_start does. A row predicted at about
    `lower` times its coefficients would cost tens of decades in log10, with gradients that stall L-BFGS-B's line search.
    Returns (the start, how many were switched on)."""
    on = m >= floor
    m = np.where(on, m, lower)
    if len(rows) == 0:
        return m, 0
    Gr = G[rows].tocsr()
    Gc = Gr.tocsc()
    covered = np.asarray(Gr @ on.astype(np.float64)).ravel() > 0
    added = 0
    while not covered.all():
        counts = np.bincount(Gr[np.flatnonzero(~covered)].indices, minlength=G.shape[1])
        i = int(np.argmax(counts))
        m[i] = max(m[i], 2.0 * floor)
        covered[Gc.indices[Gc.indptr[i]:Gc.indptr[i + 1]]] = True
        added += 1
    return m, added


def nnls_key(m):
    """A fingerprint of the NNLS fit this fit started from (its rates on the cleaned problem's columns), so that
    gibbs.py --init fitted can tell a fit made from another cleaning."""
    return hashlib.sha1(np.ascontiguousarray(m, np.float64).tobytes()).hexdigest()


def make_objective(P, G, pal, mlog, mode, geo=None, b_s=0.5, b_m=1.0, b_p=1.0):
    """The misfit as fitted, on chi2's scale, and its gradient in the rates; and a function giving it by row block. Each
    group with a log misfit is scored (1 - b) x its Gaussian + b x its log misfit (cru_common.misfit_log_weight): the slip
    rows (geo, b_s) and the MFD bins (mlog, b_m) in log10 with their effective sds, the paleo rows (pal, b_p) by their
    sites' curves."""
    from paleo_misfit import half_misfit
    from paleo_misfit import GRID_HI, GRID_LO, curve_mode
    d, sig2 = np.asarray(P['d'], float), float(P['sigma']) ** 2
    N = len(d)
    allr = np.arange(N)
    cm = curve_mode(mode)
    p_rows = np.zeros(0, np.int64)
    if pal is not None:
        p_rows, TH, BEST, SD = pal
        UB = np.log(BEST)
        F0 = np.array([half_misfit(th, np.array([ub]), cm)[0][0] for th, ub in zip(TH, UB)])
    m_rows = np.zeros(0, np.int64)
    mpdf = False
    if mlog is not None:
        m_rows = mlog
        s_log = np.log10(1.0 + 1.0 / d[m_rows])           # s_j = log10(1 + sigma_j / R_j), as gibbs.py has it
        mpdf = P.get('mfd_pdf_th') is not None and b_m >= 1 and np.all(np.isfinite(P['mfd_pdf_th'][m_rows]))
        if mpdf:                                          # MFD_GR (--mfd-misfit pdf): each bin's curve in log rate, as gibbs.py
            TH_m, BEST_m = P['mfd_pdf_th'][m_rows], P['mfd_pdf_best'][m_rows]
            import scipy.sparse as sp_
            Gr_ = sp_.csr_matrix(P['G'])
            SD_m = np.array([1.0 / Gr_.data[Gr_.indptr[r_]] for r_ in m_rows])   # each bin row's sigma: 1 / its coefficient
            F0_m = np.array([half_misfit(th, np.array([np.log(bs)]), 'log')[0][0] for th, bs in zip(TH_m, BEST_m)])
    s_rows = np.zeros(0, np.int64)                        # the slip rows, their log10 sds each side (their effective sds',
    if geo is not None:                                   # as gibbs.py has them) and 1/k
        s_rows = geo
        s_lo, s_hi, wk = slip_log_sds(P, s_rows)[:3]
    GT = G.T.tocsr()

    def parts(y):
        """Per row: its misfit and d(misfit)/dy."""
        r = y - d
        w = split_weights(P, allr, r)                     # the Gaussians (the CFM slip rates: each side's weight; 1 elsewhere)
        f, g = w * r * r / sig2, 2.0 * w * r / sig2
        if len(s_rows):                                   # (1 - b) x the split normal + b x the log10 Gaussian
            ys = y[s_rows]
            u = np.log10(ys / d[s_rows])
            sd_ = np.where(u < 0, s_lo, s_hi)
            q = u / sd_
            f[s_rows] = (1.0 - b_s) * f[s_rows] + b_s * wk * q * q
            g[s_rows] = (1.0 - b_s) * g[s_rows] + b_s * 2.0 * wk * q / (sd_ * LN10 * ys)
        for j, i in enumerate(p_rows):                    # -2 ln of each site's PDF, from its minimum (site_misfit)
            th, ub = TH[j], UB[j]
            u = np.log(y[i] * SD[j])
            uc = min(max(u, ub + GRID_LO), ub + GRID_HI)
            hf, hp = half_misfit(th, np.array([uc]), cm)
            fc = 2.0 * (hf[0] + hp[0] * (u - uc) - F0[j])
            gc = 2.0 * hp[0] / y[i]                       # d/dy of 2 f(ln(y sd)), linear beyond the grid as the tables
            f[i] = (1.0 - b_p) * f[i] + b_p * fc
            g[i] = (1.0 - b_p) * g[i] + b_p * gc
        if len(m_rows) and mpdf:                          # each bin: 2 (f(u) - f(ln best)), u = ln(rate), rate = y sigma_j
            for k_, i in enumerate(m_rows):
                uk = np.log(y[i] * SD_m[k_])
                uc = np.clip(uk, np.log(BEST_m[k_]) + GRID_LO, np.log(BEST_m[k_]) + GRID_HI)
                hf, hp = half_misfit(TH_m[k_], np.array([uc]), 'log')
                f[i] = 2.0 * (hf[0] + hp[0] * (uk - uc) - F0_m[k_])
                g[i] = 2.0 * hp[0] / y[i]
        elif len(m_rows):
            q = np.log10(y[m_rows] / d[m_rows]) / s_log   # (log10(y / R) / s)^2
            f[m_rows] = (1.0 - b_m) * f[m_rows] + b_m * q * q
            g[m_rows] = (1.0 - b_m) * g[m_rows] + b_m * 2.0 * q / (s_log * LN10 * y[m_rows])
        return f, g

    def fun(m):
        y = G @ m
        if (y[p_rows] <= 0).any() or (y[m_rows] <= 0).any() or (y[s_rows] <= 0).any():
            return np.inf, np.zeros_like(m)
        f, g = parts(y)
        return float(f.sum()), GT @ g

    def by_block(m):
        f, _ = parts(G @ m)
        names = list(P['block_names'])
        return {names[b]: float(f[np.asarray(P['block']) == b].sum()) for b in range(len(names))
                if (np.asarray(P['block']) == b).any()}

    return fun, by_block


def terms_key(a):
    """The sampler's other terms and constraints this fit was made with, as text (gibbs.py --init fitted compares it)."""
    return ';'.join([f'down_the_middle={a.down_the_middle or ""}',
                     f'slip_sum={a.slip_sum_l0:g}' if a.slip_sum else 'slip_sum=',
                     f'cross_sections={a.cross_sections}' if (a.slip_sum or a.plate_rate) else 'cross_sections=',
                     f'plate_rate={a.plate_rate or ""}',
                     f'mfd_monotonic={a.mfd_rise_sd:g}' if a.mfd_monotonic else 'mfd_monotonic=',
                     f'mfd_step={a.mfd_step_sd:g}' if getattr(a, 'mfd_step_sd', None) is not None else 'mfd_step='])


def sampler_terms(P, a):
    """The sampler's other terms, as gibbs.py builds them: (extra, cons). extra: [(name, E, e)], Gaussian rows added to
    the misfit (the balance term's, cru_common.balance_rows; the summed slip rates', plate_rate.slip_sum_rows). cons:
    [(name, C, lo, hi)], linear constraints lo <= C m <= hi, each row normalized (the monotonic MFD's pairs,
    cru_common.mfd_monotonic, rho x the lower bin's total - the upper's >= 0, over the lower bin's target; each strip's
    share of the plate rate, plate_rate.plate_rows, within --plate-rate's limits)."""
    import scipy.sparse as sp_
    extra, cons = [], []
    if a.down_the_middle:
        from cru_common import balance_rows
        B_, b_ = balance_rows(P, a.slip_block, a.bins_block, weights=a.down_the_middle)[:2]
        extra.append(('balance term', sp_.csr_matrix(B_), np.asarray(b_, float)))
    rd = cs = None
    if a.slip_sum or a.plate_rate is not None:
        if not a.rupture_set:
            raise SystemExit(f'--{"slip-sum" if a.slip_sum else "plate-rate"} needs --rupture-set')
        from cru_common import find_rupture_set
        from plate_rate import cross_sections
        rd = find_rupture_set(a.rupture_set, search=False)
        if rd is None:
            raise SystemExit(f'--rupture-set {a.rupture_set}: no OpenSHA rupture set there')
        cs = cross_sections(rd, P, a.cross_sections, a.slip_block)
    if a.slip_sum:
        from plate_rate import slip_sum_rows
        A_ss, b_ss = slip_sum_rows(rd, P, a.cross_sections, a.slip_block, l0_km=a.slip_sum_l0, cs=cs)[:2]
        extra.append(('summed slip rates', sp_.csr_matrix(A_ss), np.asarray(b_ss, float)))
    if a.mfd_monotonic:
        from cru_common import mfd_monotonic
        MM = mfd_monotonic(P, a.bins_block, a.lowest_bin, a.bin_width, a.mfd_rise_sd)
        n = P['G'].shape[1]
        C = sp_.lil_matrix((len(MM['pairs']), n))
        for q, (k, k1) in enumerate(MM['pairs']):
            C[q, MM['members'][k]] = MM['rhos'][q] / MM['R'][k]
            C[q, MM['members'][k1]] = -1.0 / MM['R'][k]
        cons.append(('the monotonic MFD', C.tocsr(), np.zeros(len(MM['pairs'])), np.full(len(MM['pairs']), np.inf)))
    if a.plate_rate is not None:
        from plate_rate import parse_limits, plate_rows
        lo_p, hi_p = parse_limits(a.plate_rate)
        A_p = plate_rows(rd, P, a.cross_sections, a.slip_block, cs=cs)[0]
        k_ = A_p.shape[0]
        cons.append(('the plate rate', sp_.csr_matrix(A_p), np.full(k_, -np.inf if lo_p is None else lo_p), np.full(k_, hi_p)))
    return extra, cons


def mfd_step_term(P, a):
    """--mfd-step-sd: the MFD steps' term as gibbs.py has it (mstep_build): with y_b = G_b m each magnitude bin row's
    prediction and lam = ln y, S = lam' A lam - 2 c' lam + const, each pair's step less the target's, about the draw's
    common tilt, over tau = bin width x sigma_s x ln 10. Returns a function m -> (S, its gradient), or None without it."""
    if getattr(a, 'mfd_step_sd', None) is None:
        return None
    import scipy.sparse as sp_
    from cru_common import magnitude_bins
    rows, _, orig, _ = magnitude_bins(P, a.bins_block, a.lowest_bin, a.bin_width)
    dy = np.asarray(P['d'], float)[rows]
    pairs = [(k, k + 1) for k in range(len(rows) - 1) if orig[k + 1] == orig[k] + 1]
    D = np.zeros((len(pairs), len(rows)))
    for p_, (k, k1) in enumerate(pairs):
        D[p_, k], D[p_, k1] = -1.0, 1.0
    e = D @ np.log(dy)
    Pc = np.eye(len(pairs)) - 1.0 / len(pairs)
    tau = a.bin_width * a.mfd_step_sd * np.log(10.0)
    A, c, const = D.T @ Pc @ D / tau ** 2, D.T @ Pc @ e / tau ** 2, float(e @ Pc @ e / tau ** 2)
    Gb = sp_.csr_matrix(P['G'])[rows]

    def term(m):
        y = Gb @ m
        lam = np.log(y)
        Al = A @ lam
        return float(lam @ Al - 2.0 * c @ lam + const), Gb.T @ (2.0 * (Al - c) / y)
    term.npairs = len(pairs)
    return term


def constraint_violation(cons, m):
    """The constraints' largest violation (normalized as sampler_terms gives them), 0 if every one is met."""
    v = 0.0
    for _, C_, lo_, hi_ in cons:
        cm_ = C_ @ m
        v = max(v, float(np.max(np.maximum(lo_ - cm_, 0.0), initial=0.0)), float(np.max(np.maximum(cm_ - hi_, 0.0), initial=0.0)))
    return v


def polish_feasible(m, cons, a, P):
    """The rates scaled exactly inside the constraints, as gibbs.py scales a start: each strip's share above its upper
    limit: all rates by one factor (with no lower limit above 0); a monotonic pair broken: the upper bin's rates down to
    rho times the lower's total (in order, until none is). Returns (rates, a description of what was scaled)."""
    m = np.array(m, float)
    said = []
    for name_, C_, lo_, hi_ in cons:
        if name_ == 'the plate rate':
            sh = C_ @ m
            if (sh > hi_).any() and not np.any(np.isfinite(lo_) & (lo_ > 0)):
                f_ = float(np.min(hi_ / np.where(sh > 0, sh, np.inf))) * (1 - 1e-9)
                m *= f_
                said.append(f'every rate scaled by {f_:.9f} for the plate rate')
    for name_, C_, lo_, hi_ in cons:
        if name_ == 'the monotonic MFD':
            from cru_common import mfd_monotonic
            MM = mfd_monotonic(P, a.bins_block, a.lowest_bin, a.bin_width, a.mfd_rise_sd)
            fixed = 0
            for _ in range(len(MM['pairs']) + 1):
                tot = np.array([m[mm].sum() for mm in MM['members']])
                bad = [(k, k1, rho) for (k, k1), rho in zip(MM['pairs'], MM['rhos']) if tot[k1] > rho * tot[k]]
                if not bad:
                    break
                k, k1, rho = bad[0]
                m[MM['members'][k1]] *= rho * tot[k] / tot[k1] * (1 - 1e-9)
                fixed += 1
            if fixed:
                said.append(f'{fixed} bins scaled down to their lower neighbour for the monotonic MFD')
    return m, '; '.join(said)


def main():
    ap = argparse.ArgumentParser(description='The best fit under the likelihood as fitted (see the module docstring).')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--start', default=None, help='the NNLS fit to start from (default: nnls/nnls_vectors.npz next to --problem)')
    ap.add_argument('--paleo-misfit', default='gaussian', choices=['gaussian', 'years', 'log', 'gaussian+years', 'gaussian+log'],
                    help="the paleo rows' misfit, as gibbs.py --paleo-misfit (default gaussian); needs --rupture-set")
    ap.add_argument('--rupture-set', default=None, help='the OpenSHA rupture set (its paleo_constraint_data.json)')
    ap.add_argument('--mfd-misfit', default='gaussian', choices=['gaussian', 'log10', 'gaussian+log10', 'pdf'],
                    help="the MFD bins' misfit, as gibbs.py --mfd-misfit (default gaussian; pdf: MFD_GR, each bin by its own "
                         "PDF, the curves saved with the problem)")
    ap.add_argument('--bins-block', default='MFD', help='the magnitude-bin rows (default MFD)')
    ap.add_argument('--slip-misfit', default='split', choices=['split', 'split+log10', 'log10'],
                    help="the slip-rate rows' misfit, as gibbs.py --slip-misfit (default split)")
    ap.add_argument('--slip-block', default='S', help='the slip-rate rows (default S)')
    ap.add_argument('--floor', type=float, default=1e-10, help='the slab floor (default 1e-10, the notebook\'s FLOOR): the start '
                    'switches off the NNLS rates below it and, where a row with a log misfit is then predicted at 0, switches '
                    'on ruptures crossing it at twice it, as gibbs.py\'s positive start does')
    ap.add_argument('--lower', type=float, default=1e-20, help='the least rate (default 1e-20), so every prediction stays above zero')
    ap.add_argument('--max-iter', type=int, default=20000, help='L-BFGS-B iterations at most (default 20000)')
    ap.add_argument('--tol', type=float, default=0.01, help='stop once the misfit falls by less than this (on chi2\'s scale) '
                    'over --window iterations (default 0.01)')
    ap.add_argument('--window', type=int, default=500, help='iterations for --tol (default 500)')
    ap.add_argument('--down-the-middle', nargs='?', const='equal', default=None, choices=['equal', 'zero-model'],
                    help='the balance term, as gibbs.py --down-the-middle: its rows counted in the misfit')
    ap.add_argument('--slip-sum', action='store_true', help='the summed slip rates, as gibbs.py --slip-sum (needs --rupture-set)')
    ap.add_argument('--slip-sum-l0', type=float, default=8.5, metavar='KM', help='as gibbs.py --slip-sum-l0 (default 8.5)')
    ap.add_argument('--cross-sections', default='all', choices=['all', 'between'], help='as gibbs.py --cross-sections')
    ap.add_argument('--plate-rate', default=None, metavar='HI or LO,HI', help='each strip\'s share of the plate rate within '
                    'these limits, as gibbs.py --plate-rate (needs --rupture-set)')
    ap.add_argument('--mfd-monotonic', action='store_true', help='the monotonic MFD, as gibbs.py --mfd-monotonic')
    ap.add_argument('--mfd-rise-sd', type=float, default=0.89, metavar='M', help='as gibbs.py --mfd-rise-sd (default 0.89)')
    ap.add_argument('--lowest-bin', type=float, default=6.8, help='as gibbs.py --lowest-bin (default 6.8)')
    ap.add_argument('--bin-width', type=float, default=0.1, help='as gibbs.py --bin-width (default 0.1)')
    ap.add_argument('--mfd-step-sd', type=float, default=None, metavar='B',
                    help='the MFD steps\' term, as gibbs.py --mfd-step-sd (B in b-value units), in the misfit')
    ap.add_argument('--snap-floor', action='store_true', help='the fit put in the sampler\'s set of rates: each rate below '
                    '--floor set to 0, then, where a slip-rate, paleo or MFD row is predicted at 0, a rupture crossing it switched '
                    'on at twice the floor (as gibbs.py\'s positive start), and scaled inside the constraints again')
    ap.add_argument('--con-tol', type=float, default=1e-9, help='the constraints\' largest violation allowed before the final '
                    'scaling (normalized; default 1e-9)')
    ap.add_argument('--al-rounds', type=int, default=20, help='augmented-Lagrangian rounds at most (default 20)')
    ap.add_argument('--out', default=None, help='output stem (default: nnls/fitted_vectors next to --problem)')
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(a.problem))
    out = a.out or os.path.join(here, 'nnls', 'fitted_vectors')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    report_to(out + '_report.txt')
    t0 = time.time()
    P = load_problem(a.problem)
    G = P['G'].tocsr()
    keep_c = np.asarray(P['keep_cols'])
    z0 = np.load(a.start or os.path.join(here, 'nnls', 'nnls_vectors.npz'))
    m_nnls = np.asarray(z0['mean'], float)[keep_c]
    pal = None
    if a.paleo_misfit != 'gaussian':
        if not a.rupture_set:
            raise SystemExit('--paleo-misfit needs --rupture-set')
        pal = paleo_rows(P, a.rupture_set, a.paleo_misfit)
    mlog = None
    if a.mfd_misfit != 'gaussian':
        if a.bins_block not in P['block_names']:
            raise SystemExit(f'--mfd-misfit {a.mfd_misfit}: no row block named {a.bins_block!r}')
        mlog = np.flatnonzero(np.asarray(P['block']) == list(P['block_names']).index(a.bins_block))
    geo = None
    if a.slip_misfit != 'split':
        if a.slip_block not in P['block_names']:
            raise SystemExit(f'--slip-misfit {a.slip_misfit}: no row block named {a.slip_block!r}')
        geo = np.flatnonzero(np.asarray(P['block']) == list(P['block_names']).index(a.slip_block))
    b_s, b_m, b_p = (misfit_log_weight(a.slip_misfit), misfit_log_weight(a.mfd_misfit), misfit_log_weight(a.paleo_misfit))
    fun_data, by_block_data = make_objective(P, G, pal, mlog, a.paleo_misfit, geo, b_s, b_m, b_p)
    extra, cons = sampler_terms(P, a)                 # the sampler's other terms (Gaussian rows) and constraints, if given
    mstep = mfd_step_term(P, a)                       # --mfd-step-sd: the MFD steps' term, nonlinear in the rates

    def fun(m):                                       # the data misfit as fitted, plus the other terms' rows
        f_, g_ = fun_data(m)
        for _, E_, e_ in extra:
            res_ = E_ @ m - e_
            f_ += float(res_ @ res_)
            g_ = g_ + 2.0 * (E_.T @ res_)
        if mstep is not None:
            s_, gs_ = mstep(m)
            f_ += s_
            g_ = g_ + gs_
        return f_, g_

    def by_block(m):
        out_ = by_block_data(m)
        for name_, E_, e_ in extra:
            res_ = E_ @ m - e_
            out_[name_] = float(res_ @ res_)
        if mstep is not None:
            out_['MFD steps'] = mstep(m)[0]
        return out_
    f0, _ = fun(np.maximum(m_nnls, a.lower))          # the NNLS fit itself, every rate at least --lower
    log_rows = np.concatenate([np.zeros(0, np.int64)]   # none when every row is Gaussian (only other terms or constraints)
                              + [r_ for r_, b_ in ((geo, b_s), (mlog, b_m), (pal[0] if pal is not None else None, b_p))
                                 if r_ is not None and b_ > 0]).astype(np.int64)
    m0, added = positive_start(G, log_rows, m_nnls, a.floor, a.lower)
    fs, _ = fun(m0)
    b0 = by_block(m0)
    how = ', '.join(w for w, on in ((f'the slip rates {a.slip_misfit} (log10 weighted {b_s:g}, its sds the effective ones)', geo is not None),
                                    ('the slip rates by their split normals', geo is None and P.get('split_lo2') is not None),
                                    (f'the paleo rows {a.paleo_misfit} (the sites\' curves weighted {b_p:g})', pal is not None),
                                    (f'the MFD bins by their own PDFs (MFD_GR)' if a.mfd_misfit == 'pdf' else
                                     f'the MFD bins {a.mfd_misfit} (log10 weighted {b_m:g})', mlog is not None)) if on) or 'every row Gaussian'
    print(f'best fit as fitted: {G.shape[0]} rows x {G.shape[1]} ruptures; the misfit as the sampler scores it ({how}); '
          f'L-BFGS-B from the NNLS fit, every rate at least {a.lower:g}')
    for name_, E_, _ in extra:
        print(f'   the {name_}: {E_.shape[0]} rows in the misfit, as gibbs.py has them')
    if mstep is not None:
        print(f'   the MFD steps (--mfd-step-sd {a.mfd_step_sd:g}): {mstep.npairs} pairs of neighbouring bins in the misfit, as '
              'gibbs.py has them')
    for name_, C_, lo_, hi_ in cons:
        print(f'   {name_}: {C_.shape[0]} constraints, kept by an augmented Lagrangian (to {a.con_tol:g}) and a final scaling, as '
              'gibbs.py keeps them')
    print(f'   start: the NNLS fit with its rates below the slab floor {a.floor:g} at {a.lower:g}, and {added} ruptures '
          f'switched on at {2 * a.floor:g} where rows with a log misfit were then predicted at 0 (as gibbs.py\'s positive '
          f'start); chi2 as fitted {f0:.2f} at the NNLS fit itself, {fs:.2f} at the start')
    c = np.sqrt(np.asarray(G.multiply(G).sum(0)).ravel())   # each rate scaled by its column's norm: v = c m, so that the
    c = np.where(c > 0, c, 1.0)                               # misfit's curvature is about the same in every direction

    def fun_v(v):
        f_, g_ = fun(v / c)
        return f_, g_ / c

    hist = []                                         # stop once the misfit falls by less than --tol over --window iterations
    win = [a.window]                                  # (the augmented Lagrangian's later rounds, warm-started: 100 at most)

    def stop(intermediate_result):
        hist.append(float(intermediate_result.fun))
        if len(hist) % 500 == 0:                      # progress: how fast the misfit settles
            print(f'      iteration {len(hist)}: misfit {hist[-1]:.3f} ({time.time() - t0:.0f}s)', flush=True)
        if len(hist) > win[0] and hist[-1 - win[0]] - hist[-1] < a.tol:
            raise StopIteration

    bnds = list(zip(a.lower * c, [None] * len(c)))
    opts = dict(maxiter=a.max_iter, maxfun=4 * a.max_iter, ftol=1e-15, gtol=1e-12, maxcor=20)
    if not cons:
        r = minimize(fun_v, m0 * c, jac=True, method='L-BFGS-B', bounds=bnds, options=opts, callback=stop)
    else:                                             # the constraints: an augmented Lagrangian, each round warm-started
        lam = [(np.zeros(C_.shape[0]), np.zeros(C_.shape[0])) for _, C_, _, _ in cons]
        mu, v, prev, nit_, nfev_ = 1e4, m0 * c, np.inf, 0, 0
        for rnd in range(a.al_rounds):
            def fun_al(v_, lam=lam, mu=mu):
                m_ = v_ / c
                f_, g_ = fun(m_)
                for (_, C_, lo_, hi_), (l_lo, l_hi) in zip(cons, lam):
                    cm_ = C_ @ m_
                    for l_, gq_, sgn_ in ((l_lo, cm_ - lo_, 1.0), (l_hi, hi_ - cm_, -1.0)):
                        ok_ = np.isfinite(gq_)
                        t_ = np.where(ok_, np.maximum(0.0, l_ - mu * np.where(ok_, gq_, 0.0)), 0.0)
                        f_ += float(np.sum(t_ * t_ - l_ * l_ * ok_)) / (2.0 * mu)
                        g_ = g_ - sgn_ * (C_.T @ t_)
                return f_, g_ / c
            hist.clear()
            win[0] = a.window if rnd == 0 else min(a.window, 100)   # later rounds, warm-started: within about 0.06% of the
                                                                    # misfit at a full window (19k, a tightly binding test)
            r = minimize(fun_al, v, jac=True, method='L-BFGS-B', bounds=bnds, options=opts, callback=stop)
            v, nit_, nfev_ = r.x, nit_ + r.nit, nfev_ + r.nfev
            viol = constraint_violation(cons, v / c)
            print(f'   augmented Lagrangian round {rnd + 1}: {r.nit} iterations, misfit {fun(v / c)[0]:.3f}, largest '
                  f'violation {viol:.2e} (mu {mu:.0e})')
            if viol <= a.con_tol:
                break
            for (_, C_, lo_, hi_), (l_lo, l_hi) in zip(cons, lam):
                cm_ = C_ @ (v / c)
                l_lo[:] = np.where(np.isfinite(lo_), np.maximum(0.0, l_lo - mu * (cm_ - lo_)), 0.0)
                l_hi[:] = np.where(np.isfinite(hi_), np.maximum(0.0, l_hi - mu * (hi_ - cm_)), 0.0)
            if viol > 0.25 * prev:
                mu *= 10.0
            prev = viol
        r.nit, r.nfev, r.x = nit_, nfev_, v
    m = r.x / c
    if cons:                                          # exactly inside the limits, as gibbs.py scales a start
        m, polish = polish_feasible(m, cons, a, P)
        print(f'   the constraints: largest violation {constraint_violation(cons, r.x / c):.2e} after the augmented Lagrangian; '
              + (polish or 'every one met, nothing scaled') + f'; after: {constraint_violation(cons, m):.1e}')
    tol_stop = len(hist) > win[0] and hist[-1 - win[0]] - hist[-1] < a.tol
    print(f'   {r.nit} iterations, {r.nfev} evaluations, {time.time() - t0:.0f}s: '
          + (f'stopped: the misfit fell by less than {a.tol:g} over the last {win[0]} iterations' if tol_stop
             else str(r.message)))
    if not (tol_stop or r.success):
        print(f'   Warning: L-BFGS-B stopped abnormally ({r.message}); the fit below is where it stopped, which may be far '
              'from the best fit as fitted')
    if a.snap_floor:                                  # the sampler's set: each rate 0 or at least the floor, every row above 0
        f_pre = fun(m)[0]
        pos_rows = np.flatnonzero(np.asarray(P['d'], float) > 0)
        n_off = int(np.sum((m > 0) & (m < a.floor)))
        m, added_s = positive_start(G, pos_rows, m, a.floor, 0.0)
        said_ = ''
        if cons:
            m, said_ = polish_feasible(m, cons, a, P)
        print(f'   snapped to the sampler\'s set: {n_off} rates below the slab floor {a.floor:g} set to 0, {added_s} ruptures '
              f'switched on at {2 * a.floor:g} for rows then predicted at 0' + (f', {said_}' if said_ else '')
              + f'; chi2 as fitted {f_pre:.3f} -> {fun(m)[0]:.3f}')
    f1, _ = fun(m)                                    # the misfit at the rates returned (after an abnormal stop, r.fun
    b1 = by_block(m)                                  # can be another point's)
    print(f'   chi2 as fitted: {f0:.2f} at the NNLS fit, {fs:.2f} at the start -> {f1:.2f}; by block, from the start: '
          + ', '.join(f'{k} {b0[k]:.1f} -> {b1[k]:.1f}' for k in b0))
    on0, on1 = int(np.sum(z0['mean'][keep_c] > 1e-10)), int(np.sum(m > 1e-10))
    print(f'   rates above 1e-10: {on0} -> {on1}; total rate {np.sum(z0["mean"]):.5g} -> {np.sum(m):.5g}')
    full = np.zeros(len(z0['mean']))
    full[keep_c] = m
    np.savez(out + '.npz', mean=full, penalised=z0['penalised'] if 'penalised' in z0.files else np.zeros(len(full), int),
             model=np.array('best fit as fitted'), misfit=np.array(f1), misfit_nnls=np.array(f0),
             nnls_key=np.array(nnls_key(np.asarray(z0['mean'], float)[keep_c])),
             paleo_misfit=np.array(a.paleo_misfit), mfd_misfit=np.array(a.mfd_misfit),
             slip_misfit=np.array(a.slip_misfit), terms=np.array(terms_key(a)), snapped=np.array(bool(a.snap_floor)))
    with open(out + '.csv', 'w') as fh:
        fh.write('rupture,penalised,best_fit_as_fitted\n')
        pen = z0['penalised'] if 'penalised' in z0.files else np.zeros(len(full), int)
        fh.writelines(f'{i},{int(pen[i])},{full[i]:.6e}\n' for i in range(len(full)))
    print(f'   wrote {out}.npz, {out}.csv and {out}_report.txt')


if __name__ == '__main__':
    main()

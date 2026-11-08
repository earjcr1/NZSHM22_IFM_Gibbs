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
prediction stays above zero, where the paleo and log10 misfits are finite; the coefficients are nonnegative). Only the
data misfit counts: no prior, and no soft terms (the balance term, the soft band, the summed slip rates).

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
    if mlog is not None:
        m_rows = mlog
        s_log = np.log10(1.0 + 1.0 / d[m_rows])           # s_j = log10(1 + sigma_j / R_j), as gibbs.py has it
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
        if len(m_rows):
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


def main():
    ap = argparse.ArgumentParser(description='The best fit under the likelihood as fitted (see the module docstring).')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--start', default=None, help='the NNLS fit to start from (default: nnls/nnls_vectors.npz next to --problem)')
    ap.add_argument('--paleo-misfit', default='gaussian', choices=['gaussian', 'years', 'log', 'gaussian+years', 'gaussian+log'],
                    help="the paleo rows' misfit, as gibbs.py --paleo-misfit (default gaussian); needs --rupture-set")
    ap.add_argument('--rupture-set', default=None, help='the OpenSHA rupture set (its paleo_constraint_data.json)')
    ap.add_argument('--mfd-misfit', default='gaussian', choices=['gaussian', 'log10', 'gaussian+log10'],
                    help="the MFD bins' misfit, as gibbs.py --mfd-misfit (default gaussian)")
    ap.add_argument('--bins-block', default='MFD', help='the magnitude-bin rows (default MFD)')
    ap.add_argument('--slip-misfit', default='split', choices=['split', 'split+log10', 'log10'],
                    help="the slip-rate rows' misfit, as gibbs.py --slip-misfit (default split)")
    ap.add_argument('--slip-block', default='S', help='the slip-rate rows (default S)')
    ap.add_argument('--lower', type=float, default=1e-20, help='the least rate (default 1e-20), so every prediction stays above zero')
    ap.add_argument('--max-iter', type=int, default=20000, help='L-BFGS-B iterations at most (default 20000)')
    ap.add_argument('--tol', type=float, default=0.01, help='stop once the misfit falls by less than this (on chi2\'s scale) '
                    'over --window iterations (default 0.01)')
    ap.add_argument('--window', type=int, default=500, help='iterations for --tol (default 500)')
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
    m0 = np.maximum(np.asarray(z0['mean'], float)[keep_c], a.lower)
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
    fun, by_block = make_objective(P, G, pal, mlog, a.paleo_misfit, geo, b_s, b_m, b_p)
    f0, _ = fun(m0)
    b0 = by_block(m0)
    how = ', '.join(w for w, on in ((f'the slip rates {a.slip_misfit} (log10 weighted {b_s:g}, its sds the effective ones)', geo is not None),
                                    ('the slip rates by their split normals', geo is None and P.get('split_lo2') is not None),
                                    (f'the paleo rows {a.paleo_misfit} (the sites\' curves weighted {b_p:g})', pal is not None),
                                    (f'the MFD bins {a.mfd_misfit} (log10 weighted {b_m:g})', mlog is not None)) if on) or 'every row Gaussian'
    print(f'best fit as fitted: {G.shape[0]} rows x {G.shape[1]} ruptures; the misfit as the sampler scores it ({how}); '
          f'L-BFGS-B from the NNLS fit, every rate at least {a.lower:g}')
    c = np.sqrt(np.asarray(G.multiply(G).sum(0)).ravel())   # each rate scaled by its column's norm: v = c m, so that the
    c = np.where(c > 0, c, 1.0)                               # misfit's curvature is about the same in every direction

    def fun_v(v):
        f_, g_ = fun(v / c)
        return f_, g_ / c

    hist = []                                         # stop once the misfit falls by less than --tol over --window iterations

    def stop(intermediate_result):
        hist.append(float(intermediate_result.fun))
        if len(hist) > a.window and hist[-1 - a.window] - hist[-1] < a.tol:
            raise StopIteration

    r = minimize(fun_v, m0 * c, jac=True, method='L-BFGS-B', bounds=list(zip(a.lower * c, [None] * len(c))),
                 options=dict(maxiter=a.max_iter, maxfun=4 * a.max_iter, ftol=1e-15, gtol=1e-12, maxcor=20), callback=stop)
    m = r.x / c
    b1 = by_block(m)
    print(f'   {r.nit} iterations, {r.nfev} evaluations, {time.time() - t0:.0f}s: '
          + (f'stopped: the misfit fell by less than {a.tol:g} over the last {a.window} iterations' if len(hist) > a.window
             and hist[-1 - a.window] - hist[-1] < a.tol else str(r.message)))
    print(f'   chi2 as fitted: {f0:.2f} at the NNLS fit -> {r.fun:.2f}; by block: '
          + ', '.join(f'{k} {b0[k]:.1f} -> {b1[k]:.1f}' for k in b0))
    on0, on1 = int(np.sum(z0['mean'][keep_c] > 1e-10)), int(np.sum(m > 1e-10))
    print(f'   rates above 1e-10: {on0} -> {on1}; total rate {np.sum(z0["mean"]):.5g} -> {np.sum(m):.5g}')
    full = np.zeros(len(z0['mean']))
    full[keep_c] = m
    np.savez(out + '.npz', mean=full, penalised=z0['penalised'] if 'penalised' in z0.files else np.zeros(len(full), int),
             model=np.array('best fit as fitted'), misfit=np.array(r.fun), misfit_nnls=np.array(f0),
             nnls_key=np.array(nnls_key(np.asarray(z0['mean'], float)[keep_c])),
             paleo_misfit=np.array(a.paleo_misfit), mfd_misfit=np.array(a.mfd_misfit),
             slip_misfit=np.array(a.slip_misfit))
    with open(out + '.csv', 'w') as fh:
        fh.write('rupture,penalised,best_fit_as_fitted\n')
        pen = z0['penalised'] if 'penalised' in z0.files else np.zeros(len(full), int)
        fh.writelines(f'{i},{int(pen[i])},{full[i]:.6e}\n' for i in range(len(full)))
    print(f'   wrote {out}.npz, {out}.csv and {out}_report.txt')


if __name__ == '__main__':
    main()

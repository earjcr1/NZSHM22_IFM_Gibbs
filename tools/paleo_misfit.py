#!/usr/bin/env python3
"""paleo_misfit.py -- the paleo rows' misfit from their recurrence-interval PDFs (gibbs.py --paleo-misfit years).

    python paleo_misfit.py RUPTURE_SET [--out FILE]      (the curves' fit to each site's numbers, as a report)

The inversion's paleo rows (cru_A, cru_D) compare each site's rate of detectable events with the rate the paleoseismic
analysis gives, as a Gaussian: target bestEstimate / sd, sd half the one-sigma range. The rupture set's
paleo_constraint_data.json gives more than that: for each site, bestEstimate, which is 1 / the mode of the recurrence
interval's PDF in years (the analysis was done in years, and its last step took 1 over everything), and the rate's
15.865/84.135% (ONE_SIGMA) and 2.275/97.725% (TWO_SIGMA) points. Those ranges are lopsided: the lower side is longer
within one sigma, the upper tail is long at two (REPORT.md, "The paleo rows' misfit").

Each site gets a sinh-arcsinh curve in log rate u = ln y (Jones and Pewsey, 2009): u = xi + eta sinh((asinh(z) + eps) /
delta), z a standard-normal score, so four parameters. Its PDF of the interval T = 1 / y in years, p_T(T) = p_u(u) / T,
has its mode fixed at 1 / bestEstimate exactly, and the four quantiles are fitted by least squares in u. The misfit of
a predicted rate y is then -2 ln p_T(1 / y), measured from its minimum (0 at y = bestEstimate), on chi2's scale: the
likelihood, as the analysis's PDF in years is (with its flat prior in years). Half of it enters the log posterior.

For the sampler, each site's f(u) = -ln p_T(1/e^u) - min (half the misfit) is tabulated on a grid of u, values and
slopes, for a cubic Hermite interpolant, continued linearly beyond the grid (so a prediction of zero costs infinitely
much, and any positive one a finite amount), with kappa, the largest |f''| of the interpolant: the kernel bounds the
change of f between two predictions from its slope and kappa (gibbs.py, col_dots_lw)."""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import norm

ZQ = norm.ppf([0.02275, 0.15865, 0.84135, 0.97725])     # the quantiles' standard-normal scores
GRID_LO, GRID_HI, GRID_H = -30.0, 14.0, 0.01            # the tables: u - ln(bestEstimate) from -30 to +14, step 0.01
POOR = 0.05                                             # quantile error (decades) above which a site is flagged
PALEO_EPS = 1e-9                                        # the paleo rows' scale in G with the misfit on: out of the Gaussian


def load_sites(rupture_dir):
    """The paleo constraints in file order: name, section index, bestEstimate, one-sigma sd, and the four rate quantiles
    (2.275, 15.865, 84.135, 97.725%)."""
    f = os.path.join(rupture_dir, 'paleo_constraint_data.json')
    if not os.path.exists(f):
        raise FileNotFoundError(f'{f}: no paleo constraint data in the rupture set')
    C = json.load(open(f))['paleoRateConstraints']
    out = []
    for c in C:
        u = {t['type']: t for t in c['uncertainties']}
        if 'ONE_SIGMA' not in u or 'TWO_SIGMA' not in u:
            raise ValueError(f"{c.get('name')}: needs ONE_SIGMA and TWO_SIGMA uncertainties")
        q = np.array([u['TWO_SIGMA']['lowerBound'], u['ONE_SIGMA']['lowerBound'], u['ONE_SIGMA']['upperBound'],
                      u['TWO_SIGMA']['upperBound']], float)
        b = float(c['bestEstimate'])
        if not (np.all(q > 0) and np.all(np.diff(q) > 0) and q[1] <= b <= q[2]):
            raise ValueError(f"{c.get('name')}: its quantiles and bestEstimate are not in order (or not all above 0)")
        out.append(dict(name=c.get('name', ''), section=int(c.get('sectionIndex', -1)), best=b,
                        sd=float(u['ONE_SIGMA']['stdDev']), q=q))
    return out


def quantile_u(th, z):
    """u at standard-normal scores z: xi + eta sinh((asinh z + eps) / delta), th = (xi, ln eta, eps, ln delta)."""
    return th[0] + np.exp(th[1]) * np.sinh((np.arcsinh(z) + th[2]) / np.exp(th[3]))


def half_misfit(th, u, mode='years'):
    """f(u) = -ln p(u) - u ('years': the interval's density in years at T = e^-u) or -ln p(u) ('log'), up to a constant,
    and its slope f'(u). p is the curve's density in u."""
    xi, eta, eps, dl = th[0], np.exp(th[1]), th[2], np.exp(th[3])
    s = (u - xi) / eta
    A = dl * np.arcsinh(s) - eps
    z = np.sinh(A)
    Ap = dl / (eta * np.sqrt(1.0 + s * s))
    lp = -0.5 * z * z + np.log(np.cosh(A)) + np.log(Ap)          # ln p(u) + ln sqrt(2 pi)
    dlp = (-z * np.cosh(A) + np.tanh(A)) * Ap - s / (eta * (1.0 + s * s))
    if mode == 'years':
        return -lp - u, -dlp - 1.0
    return -lp, -dlp


def fit_site(best, q, mode='years'):
    """The curve through a site's numbers: the mode of its misfit (the PDF's in years, or in log rate) at ln(best)
    exactly, the four quantiles (in u = ln y) by least squares. Returns (th, quantile errors in decades, ok)."""
    Q, ub = np.log(q), np.log(best)
    th0 = least_squares(lambda t: quantile_u(t, ZQ) - Q, [np.mean(Q[1:3]), np.log((Q[2] - Q[1]) / 2), 0.0, 0.0],
                        xtol=1e-14, ftol=1e-14).x
    cons = {'type': 'eq', 'fun': lambda t: half_misfit(t, np.array([ub]), mode)[1][0]}
    r = minimize(lambda t: float(np.sum((quantile_u(t, ZQ) - Q) ** 2)), th0, method='SLSQP', constraints=[cons],
                 options=dict(ftol=1e-15, maxiter=3000))
    th = r.x
    err = (quantile_u(th, ZQ) - Q) / np.log(10)
    # a single minimum, at ln(best): the slope goes from negative to positive once over a wide grid
    u = ub + np.linspace(GRID_LO, GRID_HI, 8801)
    _, fp = half_misfit(th, u, mode)
    sg = np.sign(fp)
    sg = sg[sg != 0]                                              # a grid point exactly at the minimum has slope 0
    sign_changes = int(np.sum(np.diff(sg) != 0))
    # judged by the curve itself (the optimizer may stop at its iteration limit on an already exact fit): its slope 0
    # at ln(best), a single minimum, finite quantile errors
    ok = bool(abs(half_misfit(th, np.array([ub]), mode)[1][0]) < 1e-8 and sign_changes == 1 and np.all(np.isfinite(err)))
    return th, err, ok


def tables(sites, mode='years', h=GRID_H):
    """Each site's tabulated half misfit for the kernel: dict of U0 (grid start in u = ln rate), H (step), NG (points
    per site), F and DF (values and slopes, sites one after another; F 0 at the minimum, ln best), KAPPA (the largest
    |f''| of each site's cubic Hermite interpolant), the fits (th, quantile errors, ok) and the report lines."""
    n = int(round((GRID_HI - GRID_LO) / h)) + 1
    U0, F, DF, KAP, fits = [], [], [], [], []
    for s in sites:
        th, err, ok = fit_site(s['best'], s['q'], mode)
        ub = np.log(s['best'])
        u = ub + GRID_LO + h * np.arange(n)
        f, fp = half_misfit(th, u, mode)
        f = f - half_misfit(th, np.array([ub]), mode)[0][0]          # 0 at the minimum, ln best
        d = np.diff(f)                                               # the interpolant's f'' at both ends of each step
        c0 = (6.0 * d / h - 4.0 * fp[:-1] - 2.0 * fp[1:]) / h
        c1 = (-6.0 * d / h + 2.0 * fp[:-1] + 4.0 * fp[1:]) / h
        U0.append(u[0]); F.append(f); DF.append(fp); KAP.append(float(max(np.abs(c0).max(), np.abs(c1).max())))
        fits.append((th, err, ok))
    return dict(U0=np.array(U0), H=float(h), NG=n, F=np.concatenate(F), DF=np.concatenate(DF), KAPPA=np.array(KAP),
                fits=fits, mode=mode)


def report_lines(sites, tab):
    """The fit of each site's curve to its numbers, and the sites flagged (quantile error above POOR decades)."""
    errs = np.array([f[1] for f in tab['fits']])
    worst = np.abs(errs).max(1)
    L = [f"paleo misfit ({tab['mode']}): {len(sites)} sites, each a sinh-arcsinh curve in log rate with the "
         + ('mode of its PDF in years' if tab['mode'] == 'years' else 'mode of its PDF in log rate')
         + ' at 1/bestEstimate exactly and its 2.275/15.865/84.135/97.725% points fitted by least squares; '
         f'worst quantile error per site: median {np.median(worst):.4f} decades, largest {worst.max():.4f} '
         f'({sites[int(np.argmax(worst))]["name"]}); {int(np.sum(worst > POOR))} site(s) above {POOR} decades'
         + ('' if all(f[2] for f in tab['fits']) else '; SOME CURVES FAILED (see below)')]
    for i in np.argsort(-worst):
        if worst[i] <= POOR and tab['fits'][i][2]:
            continue
        e = errs[i]
        L.append(f"   {sites[i]['name'][:60]:60s} errors {e[0]:+.3f} {e[1]:+.3f} {e[2]:+.3f} {e[3]:+.3f} decades"
                 + ('' if tab['fits'][i][2] else '  (FAILED: no single minimum at bestEstimate)'))
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('ruptures', help="the OpenSHA rupture set (its ruptures/ folder, or a solution folder holding one)")
    ap.add_argument('--mode', default='years', choices=['years', 'log'])
    ap.add_argument('--out', default=None, help='report file (default: printed only)')
    a = ap.parse_args()
    rd = a.ruptures if os.path.exists(os.path.join(a.ruptures, 'paleo_constraint_data.json')) else os.path.join(a.ruptures, 'ruptures')
    sites = load_sites(rd)
    tab = tables(sites, a.mode)
    L = report_lines(sites, tab)
    print('\n'.join(L))
    if a.out:
        open(a.out, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()

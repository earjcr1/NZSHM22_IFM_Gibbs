#!/usr/bin/env python3
"""Dispersed starts for the sampler's chains (November 17th, 2026, the user's request; the notebook's DISPERSED_STARTS).

    python dispersed_starts.py --problem cru_clean.npz --chains 16 [--noise 2] [--seed S] [--workers 4]
                               [--polish-sd 0.1] [--out starts.npy] [--fit-args BEST_FIT_AS_FITTED_ARGS ...]

Each chain gets its own start, made in two steps:
  1. half of them (1 - --prior-share) from best fits to the data, the other half from the prior, so that the chains
     approach the posterior from both sides (the user's choice, 2026-11-17): best fits are sparse, a few ruptures
     carrying each bin, while the posterior spreads its rate over many more; chains from both ends that still agree are
     the stronger test.
     - best fits: randomized best fits, each the NNLS fit to the data plus a fresh draw of NOISE times their stated
       noise (randomized_nnls.py's fits, each from its own seed spawned from --seed); with NOISE 2, the default, they are
       over-dispersed relative to the posterior. None starts from cleaning's unperturbed NNLS fit (the user's choice, for
       consistency: every chain's start is a draw); that fit is the report's reference.
     - prior draws: in each magnitude bin, a fraction on drawn uniformly (the prior's Beta(1, 1)), each rupture on with
       that probability (at least one), its rate log-uniform from twice the slab floor up to the bin's target R_j (the
       slab), and the bin's rates then scaled together so that they add up to R_j times a factor exp(2 (sigma_j / R_j)
       z), z standard normal, sigma_j the bin's own uncertainty (with MFD_GR its PDF's sd): over-dispersed totals, as the
       fits' are (a raw draw's bin totals would be hundreds of times the targets; scaled exactly to R_j they spread by
       nothing, against a posterior's 6-13% at 19,000 ruptures).
     R-hat across all the chains then tests whether the run has forgotten where it started (Gelman and Rubin).
  2. with --fit-args (the notebook passes the run's misfit, terms and constraints, as best_fit_as_fitted.py takes them),
     a short polish for feasibility, not for fit: best_fit_as_fitted.py from the start, --polish-iter (30) iterations per
     augmented-Lagrangian round. A polish run until the misfit settled pulled the starts together: at 19,000 ruptures
     their bin totals then spread by 0.6% (300 iterations: 1.3%) against the posterior's 8%, and 99% of each start's
     rate sat on ruptures active in all the others, so R-hat would test little; with 30 per round the bin totals spread
     by about 24% and the starts stay over-dispersed, as Gelman and Rubin ask. (--polish-sd > 0 stops a longer polish
     once its misfit falls by less than that times the fit check's sd over --polish-window iterations.) Even so short, it
     scores the data as the sampler does (the CFM's split normals, the paleo sites' curves, MFD_GR's bin PDFs, the
     summed slip rates, the MFD's steps), meets the hard constraints exactly (positivity, the monotonic MFD, the plate
     rate) and snaps rates below the slab floor to 0 as the sampler's prior does, so each start is one gibbs.py accepts
     as it is (unpolished NNLS fits broke the monotonic MFD, and gibbs.py's repair then left rows predicted at zero). At
     630,000 ruptures the full fit took 27 minutes, but its first 500 iterations did 95% of the work (227 -> 152 against
     147 at the end); a polish moves the rates only along the directions the data see (G's rows: rank 2,372 of 615,700
     rates), so the starts keep their differences in which ruptures carry the rate.
Without --fit-args (a run scored as NNLS scores it, with no other terms or constraints), the fits are the starts.
The fits and the polishes run in --workers processes, one thread each.

Writes --out (default starts.npy next to --problem): a (chains, ruptures) array over the cleaned problem's ruptures, for
gibbs.py --init; its report beside it (starts_report.txt): each start's Gaussian chi2 on the cleaned rows before the
polish, its misfit as fitted after it, its rates above the slab floor, its total rate, and the share of those on
ruptures cleaning's NNLS fit uses. The polishes' own files go in a folder beside it (starts/), removed at the end.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cru_common import load_problem, mfd_monotonic, prelim_path, report_to
from randomized_nnls import ensemble

HERE = os.path.dirname(os.path.abspath(__file__))


def polish(job):
    """One start's polish: best_fit_as_fitted.py from its fit, a subprocess with one thread. Returns (k, rates over the
    inputs' ruptures, misfit as fitted, seconds, iterations, why it stopped)."""
    k, problem, start, stem, fit_args, stops = job
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, 'best_fit_as_fitted.py'), '--problem', problem, '--start', start,
                        '--out', stem, *stops, *fit_args], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f'start {k + 1}: best_fit_as_fitted.py failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}')
    import re
    m_ = re.search(r'^   (\d+) iterations, \d+ evaluations, \d+s: (.*)$', p.stdout, re.M)
    it_, why = (int(m_.group(1)), m_.group(2)) if m_ else (-1, '?')
    why = 'settled' if 'fell by less' in why else 'the cap' if 'ITERATIONS' in why.upper() else why[:30]
    with np.load(stem + '.npz') as z:
        return k, np.asarray(z['mean'], np.float64), float(z['misfit']), time.time() - t0, it_, why


def main():
    argv = sys.argv[1:]
    fit_args = []
    if '--fit-args' in argv:                             # everything after it goes to best_fit_as_fitted.py
        i = argv.index('--fit-args')
        argv, fit_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--chains', type=int, required=True, help='the number of chains (one start each)')
    ap.add_argument('--noise', type=float, default=2.0, help='the randomized fits\' noise, times the stated noise (default 2)')
    ap.add_argument('--seed', type=int, default=1, help='seed of the noise draws (default 1)')
    ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1), help='processes (default 4)')
    ap.add_argument('--polish-sd', type=float, default=0.0, help='with a longer --polish-iter: stop once the misfit falls by '
                    'less than this times the fit check\'s sd over --polish-window iterations (default 0: off)')
    ap.add_argument('--polish-window', type=int, default=100, help='iterations for --polish-sd (default 100)')
    ap.add_argument('--polish-iter', type=int, default=30, help='the polish\'s iterations per augmented-Lagrangian round '
                    '(default 30: feasibility, not fit; at most 1000)')
    ap.add_argument('--prior-share', type=float, default=0.5, help='the share of the chains started from prior draws '
                    '(default 0.5); the rest from randomized fits')
    ap.add_argument('--floor', type=float, default=1e-10, help='the slab floor (default 1e-10)')
    ap.add_argument('--bins-block', default='MFD', help='the magnitude-bin row group (default MFD)')
    ap.add_argument('--out', default=None, help='the starts (default starts.npy next to --problem)')
    ap.add_argument('--fit-args', nargs='*', default=[], help='(last) best_fit_as_fitted.py\'s arguments for the polish')
    a = ap.parse_args(argv)
    if a.chains < 1 or not a.noise > 0 or not 1 <= a.polish_iter <= 1000 or not 0 <= a.prior_share < 1:
        raise SystemExit('--chains, --polish-iter: 1 or more; --noise: above 0')
    here = os.path.dirname(os.path.abspath(a.problem))
    out = a.out or os.path.join(here, 'starts.npy')
    report_to(os.path.splitext(out)[0] + '_report.txt')
    P = load_problem(a.problem)
    G, d, sigma = P['G'].tocsr(), np.asarray(P['d'], np.float64), float(P['sigma'])
    n = G.shape[1]
    keep_c = np.asarray(P['keep_cols'])
    n_raw = int(P['raw_shape'][1]) if P.get('raw_shape') else n
    with np.load(prelim_path(a.problem)) as z:
        base = np.asarray(z['m_nnls'], np.float64)
    t0 = time.time()
    n_prior = int(round(a.prior_share * a.chains)) if a.chains > 1 else 0
    n_fit = a.chains - n_prior
    print(f'dispersed starts: {a.chains} chains; {n_fit} from randomized best fits (NNLS '
          f'to the data plus {a.noise:g} x their stated noise, seed {a.seed}) and {n_prior} from prior draws (each bin\'s '
          'fraction on uniform, rates log-uniform up to its target, the bin then scaled to its target), '
          + (f'each then polished for feasibility by best_fit_as_fitted.py with the run\'s misfit, terms and constraints '
             f'({a.polish_iter} iterations per round' + (f'; or until its misfit settles, {a.polish_sd:g} x the fit check\'s '
             f'sd over {a.polish_window} iterations' if a.polish_sd > 0 else '') + ')'
             if fit_args else 'used as they are (the run is scored as NNLS scores it)')
          + f'; {a.workers} workers', flush=True)
    res = ensemble(os.path.abspath(a.problem), n_fit, a.seed, a.workers, noise=a.noise) if n_fit else []
    X = np.zeros((a.chains, n))
    for k, r in enumerate(res):
        X[k, r['idx']] = r['val']
    kind = ['fit'] * n_fit + ['prior'] * n_prior
    kind_arr = np.array(kind)
    if n_prior:                                          # prior draws, each bin scaled to its target (see above)
        MM = mfd_monotonic(P, a.bins_block)              # the bins' members and targets R_j (rates per year)
        lf = np.log(2 * a.floor)
        for q, ss in enumerate(np.random.SeedSequence([a.seed, 7]).spawn(n_prior)):
            g = np.random.default_rng(ss)
            x = np.zeros(n)
            for mem, R, sg in zip(MM['members'], MM['R'], MM['sig']):
                if len(mem) == 0 or not R > 2 * a.floor:
                    continue
                R = R * float(np.exp(2.0 * sg / R * g.standard_normal()))   # the bin's total, over-dispersed
                on = g.random(len(mem)) < g.random()
                if not on.any():
                    on[g.integers(len(mem))] = True
                r = np.exp(g.uniform(lf, np.log(R), int(on.sum())))
                x[mem[on]] = r * (R / r.sum())
            X[n_fit + q] = x
    chi = lambda x: float(np.sum((G @ x - d) ** 2)) / sigma ** 2
    chi_fit = [chi(x) for x in X]
    t_fits = time.time() - t0
    mis = [float('nan')] * a.chains
    its = [''] * a.chains
    if fit_args:
        fc = P.get('chi2_expect')
        tol = a.polish_sd * (float(fc[1]) if fc is not None else 1.0)
        stops = ['--tol', f'{tol:.6g}', '--window', str(a.polish_window), '--max-iter', str(a.polish_iter)]
        tmp = os.path.join(os.path.dirname(os.path.abspath(out)), 'starts')
        os.makedirs(tmp, exist_ok=True)
        jobs = []
        for k in range(a.chains):
            full = np.zeros(n_raw)
            full[keep_c] = X[k]
            f_ = os.path.join(tmp, f'fit{k + 1:02d}.npz')
            np.savez(f_, mean=full)
            jobs.append((k, os.path.abspath(a.problem), f_, os.path.join(tmp, f'start{k + 1:02d}'), fit_args, stops))
        with ThreadPoolExecutor(a.workers) as pool:
            for k, m_full, f1, sec, it_, why in pool.map(polish, jobs):
                X[k] = m_full[keep_c]
                mis[k] = f1
                its[k] = f'{it_} ({why})'
                print(f'   start {k + 1:2d} ({kind[k]}) polished: misfit as fitted {f1:.1f}, {it_} iterations, {why} '
                      f'({sec:.0f} s)', flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    floor = a.floor
    on0 = base > floor                                   # cleaning's NNLS fit: a reference only, no chain starts from it
    print(f'{"start":>5s} {"from":>6s} {"chi2 (Gauss)":>12s} {"as fitted":>10s} {"rates on":>9s} {"total rate":>11s} {"on NNLS fit’s":>14s} {"polish iterations":>18s}')
    for k in range(a.chains):
        nz = X[k] > floor
        share = float(np.mean(on0[nz])) if nz.any() else 0.0
        print(f'{k + 1:5d} {kind[k]:>6s} {chi_fit[k]:12.1f} {mis[k]:10.1f} {int(nz.sum()):9d} {X[k].sum():11.5f} {100 * share:13.0f}% {its[k]:>18s}')
    print('   chi2 (Gauss): the start before the polish, a plain Gaussian on the cleaned rows; as fitted: after the polish, '
          'the misfit as the sampler scores it; on NNLS fit\u2019s: the share of its rates on ruptures cleaning\'s NNLS fit '
          'uses (a reference: no chain starts from it)')
    MMd = mfd_monotonic(P, a.bins_block) if not n_prior else MM   # the dispersion: bin totals across the starts
    TT = np.array([[x[m].sum() for m in MMd['members']] for x in X]) / MMd['R']
    own = np.median(MMd['sig'] / MMd['R'])
    sp = lambda T_: float(np.median(np.std(np.log(np.maximum(T_, 1e-300)), 0)))
    print(f'dispersion: the starts\' bin totals spread by {100 * sp(TT):.0f}% (median over bins, sd in log; fits '
          f'{100 * sp(TT[kind_arr == "fit"]) if (kind_arr == "fit").sum() > 1 else float("nan"):.0f}%, prior draws '
          f'{100 * sp(TT[kind_arr == "prior"]) if (kind_arr == "prior").sum() > 1 else float("nan"):.0f}%). For comparison: the '
          f'bins\' own uncertainties, {100 * own:.0f}% (median), and the posterior\'s spread, usually much narrower (19,000 '
          'ruptures with MFD_GR: 8%, the bins\' own 24-75%); the convergence report\'s bins_posterior.csv gives this run\'s. '
          'Starts spread wider than the posterior are over-dispersed, as R-hat needs')
    np.save(out, X)
    print(f'wrote {out} ({a.chains} x {n}) in {time.time() - t0:.0f} s (the fits {t_fits:.0f} s)')


if __name__ == '__main__':
    main()

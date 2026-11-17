#!/usr/bin/env python3
"""Dispersed starts for the sampler's chains (November 17th, 2026, the user's request; the notebook's DISPERSED_STARTS).

    python dispersed_starts.py --problem cru_clean.npz --chains 16 [--noise 2] [--seed S] [--workers 4]
                               [--polish-iter 300] [--out starts.npy] [--fit-args BEST_FIT_AS_FITTED_ARGS ...]

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
       slab), and the bin's rates then scaled together so that they add up to R_j (a raw draw's bin totals would be
       hundreds of times the targets).
     R-hat across all the chains then tests whether the run has forgotten where it started (Gelman and Rubin).
  2. with --fit-args (the notebook passes the run's misfit, terms and constraints, as best_fit_as_fitted.py takes them),
     a short polish: best_fit_as_fitted.py from that fit, stopped after --polish-iter iterations (300 by default). It
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

from cru_common import load_problem, prelim_path, report_to
from randomized_nnls import ensemble

HERE = os.path.dirname(os.path.abspath(__file__))


def polish(job):
    """One start's polish: best_fit_as_fitted.py from its fit, a subprocess with one thread. Returns (k, rates over the
    inputs' ruptures, misfit as fitted, seconds)."""
    k, problem, start, stem, fit_args, iters = job
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, 'best_fit_as_fitted.py'), '--problem', problem, '--start', start,
                        '--out', stem, '--max-iter', str(iters), *fit_args], capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f'start {k + 1}: best_fit_as_fitted.py failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}')
    with np.load(stem + '.npz') as z:
        return k, np.asarray(z['mean'], np.float64), float(z['misfit']), time.time() - t0


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
    ap.add_argument('--polish-iter', type=int, default=300, help='the polish\'s iterations at most (default 300)')
    ap.add_argument('--prior-share', type=float, default=0.5, help='the share of the chains started from prior draws '
                    '(default 0.5); the rest from randomized fits')
    ap.add_argument('--floor', type=float, default=1e-10, help='the slab floor (default 1e-10)')
    ap.add_argument('--bins-block', default='MFD', help='the magnitude-bin row group (default MFD)')
    ap.add_argument('--out', default=None, help='the starts (default starts.npy next to --problem)')
    ap.add_argument('--fit-args', nargs='*', default=[], help='(last) best_fit_as_fitted.py\'s arguments for the polish')
    a = ap.parse_args(argv)
    if a.chains < 1 or not a.noise > 0 or a.polish_iter < 1 or not 0 <= a.prior_share < 1:
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
          + (f'each then polished by best_fit_as_fitted.py ({a.polish_iter} iterations at most, with the run\'s misfit, terms '
             'and constraints)' if fit_args else 'used as they are (the run is scored as NNLS scores it)')
          + f'; {a.workers} workers', flush=True)
    res = ensemble(os.path.abspath(a.problem), n_fit, a.seed, a.workers, noise=a.noise) if n_fit else []
    X = np.zeros((a.chains, n))
    for k, r in enumerate(res):
        X[k, r['idx']] = r['val']
    kind = ['fit'] * n_fit + ['prior'] * n_prior
    if n_prior:                                          # prior draws, each bin scaled to its target (see above)
        from cru_common import mfd_monotonic
        MM = mfd_monotonic(P, a.bins_block)              # the bins' members and targets R_j (rates per year)
        lf = np.log(2 * a.floor)
        for q, ss in enumerate(np.random.SeedSequence([a.seed, 7]).spawn(n_prior)):
            g = np.random.default_rng(ss)
            x = np.zeros(n)
            for mem, R in zip(MM['members'], MM['R']):
                if len(mem) == 0 or not R > 2 * a.floor:
                    continue
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
    if fit_args:
        tmp = os.path.join(os.path.dirname(os.path.abspath(out)), 'starts')
        os.makedirs(tmp, exist_ok=True)
        jobs = []
        for k in range(a.chains):
            full = np.zeros(n_raw)
            full[keep_c] = X[k]
            f_ = os.path.join(tmp, f'fit{k + 1:02d}.npz')
            np.savez(f_, mean=full)
            jobs.append((k, os.path.abspath(a.problem), f_, os.path.join(tmp, f'start{k + 1:02d}'), fit_args, a.polish_iter))
        with ThreadPoolExecutor(a.workers) as pool:
            for k, m_full, f1, sec in pool.map(polish, jobs):
                X[k] = m_full[keep_c]
                mis[k] = f1
                print(f'   start {k + 1:2d} ({kind[k]}) polished: misfit as fitted {f1:.1f} ({sec:.0f} s)', flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    floor = a.floor
    on0 = base > floor                                   # cleaning's NNLS fit: a reference only, no chain starts from it
    print(f'{"start":>5s} {"from":>6s} {"chi2 (Gauss)":>12s} {"as fitted":>10s} {"rates on":>9s} {"total rate":>11s} {"on NNLS fit’s":>14s}')
    for k in range(a.chains):
        nz = X[k] > floor
        share = float(np.mean(on0[nz])) if nz.any() else 0.0
        print(f'{k + 1:5d} {kind[k]:>6s} {chi_fit[k]:12.1f} {mis[k]:10.1f} {int(nz.sum()):9d} {X[k].sum():11.5f} {100 * share:13.0f}%')
    print('   chi2 (Gauss): the start before the polish, a plain Gaussian on the cleaned rows; as fitted: after the polish, '
          'the misfit as the sampler scores it; on NNLS fit\u2019s: the share of its rates on ruptures cleaning\'s NNLS fit '
          'uses (a reference: no chain starts from it)')
    np.save(out, X)
    print(f'wrote {out} ({a.chains} x {n}) in {time.time() - t0:.0f} s (the fits {t_fits:.0f} s)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Convergence checks of a result set's replicate runs, with the verdict, PASS or CHECK, against the preset's targets.

    python check_convergence.py RUN [RUN ...] --preset quick|conservative|very_conservative [--problem cru_clean.npz]
                                [--diag-floor -10] [--out REPORT] [--save STATS]

RUN is a gibbs.py run (its --out, e.g. spikeslab/main_rep0), one per replicate of the result set. Each run is read from
its stored draws (<run>_samples.npy) when it has them, otherwise from the running summaries it wrote
(<run>_stream.npz), which give the same R-hat and effective draws. Log rates below 10^--diag-floor (1e-10, the
notebook's FLOOR) count as equal for every prior.

Prints, and writes to REPORT (default: convergence_report.txt next to the first run):
- per run: chi2 (as fitted: each group scored as the likelihood scores it, the paleo sites by their recurrence-interval
  PDFs and the MFD bins in log10 where the run did so; the Gaussian chi2 for older runs) and the total rate with their
  effective draws; the worst R-hat of the per-rupture log rates and the
  noise baseline, the worst R-hat that independent draws would give over this many rates; how many rates lie above
  it; and the rates' effective draws (median and 5th percentile);
- for spike-and-slab: ruptures on per draw (with its R-hat) and each bin's fraction on (worst R-hat, fewest effective
  draws);
- the fit check: each run's chi2 against the value a model matching the truth would have (as fitted, from the run;
  for older runs from cleaning);
- the verdict against the targets of --preset (docs/TESTING.md, "Convergence"), with the targets missed;
- the slow-mixing minority: ruptures whose R-hat exceeds 1.01, or the noise baseline if higher, with their share of the
  total rate; reported apart from the verdict;
- the replicates' agreement, rupture by rupture: standardized differences of mean log rate against the first run.
STATS (default: convergence.npz next to REPORT) keeps each run's per-rupture R-hat and effective draws and its
slow-mixing threshold, for the rupture table that make_tables.py writes.
"""
import argparse
import os

import numpy as np

from cru_common import load_problem, split_rhat, ess, ess_many, load_stream, stream_rhat, stream_ess, report_to

NEED = {'quick': (100, 200, 1.01), 'conservative': (1000, 2000, 1.01), 'very_conservative': (10000, 20000, 1.005)}
SS_KEYS = ('on', 'rhat_on', 'worst_rhat_pi', 'ess_pi_min')
_RHAT_NOISE = {}


def table(rows):
    """Print a list of dicts as an aligned table."""
    fmt = lambda v: f'{v:.4g}' if isinstance(v, float) else str(v)
    keys = list(rows[0])
    width = {k: max([len(k)] + [len(fmt(r[k])) for r in rows]) for k in keys}
    print('  '.join(k.rjust(width[k]) for k in keys))
    for r in rows:
        print('  '.join(fmt(r[k]).rjust(width[k]) for k in keys))


def rhat_noise(K, D, nq, seed=0):
    """Worst split R-hat expected over nq quantities if every draw were independent (simulated once per shape: the
    replicates share it)."""
    if (K, D, nq, seed) not in _RHAT_NOISE:
        g = np.random.default_rng(seed)
        worst = 0.0
        for j0 in range(0, nq, 2000):
            worst = max(worst, float(np.nanmax(split_rhat(g.standard_normal((K, D, min(2000, nq - j0)))))))
        _RHAT_NOISE[K, D, nq, seed] = worst
    return _RHAT_NOISE[K, D, nq, seed]


def convergence(name, floor, stats):
    """One run's row of the convergence table; its per-rupture statistics go into stats[name]."""
    z = dict(np.load(name + '.npz'))
    if os.path.exists(name + '_samples.npy'):
        S = np.load(name + '_samples.npy', mmap_mode='r')
        K, D, nn = S.shape
        rh, es, mean, var, rate = (np.empty(nn) for _ in range(5))
        clip = 10.0 ** floor                     # rates below this count as equal (diagnostics only; same for every prior)
        for j0 in range(0, nn, 2048):
            X = np.asarray(S[:, :, j0:j0 + 2048], dtype=np.float64)
            L = np.log10(np.maximum(X, clip))
            j1 = j0 + L.shape[2]
            rh[j0:j1], es[j0:j1] = split_rhat(L), ess_many(L)
            flat = L.reshape(K * D, -1)
            mean[j0:j1], var[j0:j1] = flat.mean(0), flat.var(0)
            rate[j0:j1] = X.reshape(K * D, -1).mean(0)       # posterior mean rate (linear)
    else:                                        # no stored draws: the running summaries written during the run
        st = load_stream(name, ('chains', 'n_keep', 's1', 's2', 'cnt', 'bsum', 'bcnt', 'msum', 'clip'))
        K, D, nn = int(st['chains']), int(st['n_keep']), st['s1'].shape[2]
        if abs(np.log10(float(st['clip'])) - floor) > 1e-9:
            print(f'note: {name} was summarized with a clip of {float(st["clip"]):.0e}, not 10^FLOOR (--diag-floor)')
        rh, es = stream_rhat(st), stream_ess(st)
        mean = st['s1'].sum((0, 1)) / (K * D)
        var = st['s2'].sum((0, 1)) / (K * D) - mean ** 2
        rate = st['msum'] / (K * D)                          # posterior mean rate (linear)
    const = ~(var > 1e-12)                               # below the diagnostic floor in every draw: R-hat and effective draws undefined
    rh, es = np.where(const, np.nan, rh), np.where(const, np.nan, es)
    stats[name] = dict(mean=mean, var=var, ess=es, rhat=rh, rate=rate)
    w = int(z['burn']) // int(z['thin'])
    chi, tot = (z['misfit'] if 'misfit' in z else z['chi2'])[:, w:], z['sum_m'][:, w:]   # chi2 as fitted (older runs: chi2)
    stats[name]['fitted'] = 'misfit' in z and not np.array_equal(z['misfit'], z['chi2'])
    stats[name]['expect'] = tuple(float(v) for v in z['misfit_expect']) if 'misfit_expect' in z else None
    base = rhat_noise(K, D, nn)
    stats[name]['slow_at'] = max(1.01, base)            # slow mixing: R-hat above 1.01, or above the noise baseline if higher
    stats[name]['chi2'] = (float(chi.mean()), float(chi.std()))
    row = dict(run=name, chains=K, draws=D, chi2=float(chi.mean()), ess_chi2=float(ess(chi)), ess_total=float(ess(tot)),
               worst_rhat=float(np.nanmax(rh)), noise_rhat=base, rates_above_noise=int(np.sum(rh > base)),
               ess_rate_median=float(np.nanmedian(es)), ess_rate_5pct=float(np.nanpercentile(es, 5)),
               rhat_summaries=max(float(split_rhat(chi[:, :, None])[0]), float(split_rhat(tot[:, :, None])[0])))
    if 'hyper_pi' in z:                   # spike-and-slab: ruptures on per draw, and the fraction on in each bin
        pi, on = z['hyper_pi'][:, w:, :], z['n_on'][:, w:]
        row.update(on=float(on.mean()), rhat_on=float(split_rhat(on[:, :, None])[0]), worst_rhat_pi=float(np.nanmax(split_rhat(pi))),
                   ess_pi_min=float(min(ess(pi[..., k]) for k in range(pi.shape[2]))))
    return row


def agreement(a, b, stats):
    A, B = stats[a], stats[b]
    err2 = lambda S_: np.where(S_['var'] > 1e-12, S_['var'] / S_['ess'], 0.0)   # squared Monte Carlo error; 0 if constant
    with np.errstate(invalid='ignore', divide='ignore'):
        z = (B['mean'] - A['mean']) / np.sqrt(err2(A) + err2(B))
    ok = np.isfinite(z)                                  # leaves out only ruptures below the diagnostic floor throughout both runs
    z = z[ok]
    print(f'{a} vs {b}: standardized differences of mean log10 rate over {ok.sum()} ruptures'
          + (f' ({(~ok).sum()} below 10^FLOOR in every draw of both runs left out)' if (~ok).any() else '')
          + f': sd {z.std():.3f} (1 expected), '
          f'|z| > 3: {100 * np.mean(abs(z) > 3):.2f}% (0.27% expected), |z| > 4: {int(np.sum(abs(z) > 4))} '
          f'({6.334e-5 * len(z):.1f} expected)')


def main():
    ap = argparse.ArgumentParser(description='Convergence checks of replicate runs (see the module docstring).')
    ap.add_argument('runs', nargs='+', metavar='RUN', help='the result set\'s replicate runs (gibbs.py --out)')
    ap.add_argument('--preset', required=True, choices=list(NEED), help='the run lengths the runs were made with, for the '
                    'verdict\'s targets')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem the runs sampled, for the fit check')
    ap.add_argument('--diag-floor', type=float, default=-10.0, help='log10 rate below which rates count as equal in R-hat '
                    'and effective draws (default -10; the notebook passes its FLOOR)')
    ap.add_argument('--out', default=None, help='the report (default: convergence_report.txt next to the first run)')
    ap.add_argument('--save', default=None, help='per-rupture R-hat and effective draws (default: convergence.npz next to '
                    'the report)')
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(a.runs[0]), 'convergence_report.txt')
    report_to(out)
    runs = a.runs
    FIT = load_problem(a.problem)['chi2_expect']        # the fit check: chi2 of a model that matches the truth
    STATS = {}
    rows = [convergence(r, a.diag_floor, STATS) for r in runs]
    show_rhat = lambda r: {k: (f'{v:.4f}' if 'rhat' in k and isinstance(v, float) else v) for k, v in r.items()}   # R-hat to 4 decimals
    table([show_rhat({k: v for k, v in r.items() if k not in SS_KEYS}) for r in rows])
    ssr = [r for r in rows if 'on' in r]
    if ssr:
        print()
        print('spike-and-slab: ruptures on per draw (with its R-hat), and the fraction on in each bin (worst R-hat, fewest effective draws):')
        table([show_rhat({k: r[k] for k in ('run',) + SS_KEYS}) for r in ssr])
    if FIT is not None or any(STATS[r]['expect'] for r in runs):   # the fit check: not part of the verdict
        print()
        for r in runs:
            c_, F_ = STATS[r]['chi2'], STATS[r]['expect'] or FIT
            if F_ is None:
                continue
            asf = ' as fitted' if STATS[r]['fitted'] else ''
            print(f'{r}: fit check: chi2{asf} {c_[0]:.0f} +- {c_[1]:.0f} against {F_[0]:.0f} +- {F_[1]:.0f} expected if the '
                  f'uncertainties are right ({(c_[0] - F_[0]) / F_[1]:+.1f} sd)')
    need_rate, need_sum, rhat_sum = NEED[a.preset]     # effective draws: 95% of ruptures, the total rate; R-hat of chi2 and the total
    print()
    for r in rows:
        why, modes = [], False
        if r['ess_rate_5pct'] < need_rate:
            why.append(f"95% of ruptures need {need_rate:,} effective draws (5th percentile {r['ess_rate_5pct']:,.0f})")
        if r['ess_total'] < need_sum:
            why.append(f"the total rate needs {need_sum:,} effective draws ({r['ess_total']:,.0f})")
        if r['rhat_summaries'] >= rhat_sum:
            why.append(f"R-hat of chi2 or the total rate {r['rhat_summaries']:.4f} (below {rhat_sum} needed)")
            modes = True
        if 'on' in r:
            if max(r['rhat_on'], r['worst_rhat_pi']) >= 1.01:
                why.append(f"R-hat of the ruptures on or of a bin's fraction on {max(r['rhat_on'], r['worst_rhat_pi']):.4f} (below 1.01 needed)")
            if r['ess_pi_min'] < need_sum / 5:
                why.append(f"each bin's fraction on needs {need_sum / 5:,.0f} effective draws (fewest {r['ess_pi_min']:,.0f})")
        print(f"{r['run']}: " + ('PASS' if not why else 'CHECK: ' + '; '.join(why) + ': run longer (more sweeps)'
                                 + ('; if this persists, start chains from dispersed models (gibbs.py --init FILE.npy) to look for separate modes' if modes else '')))
    print()
    pct = lambda x: f'{100 * x:.2g}%' if 100 * x < 10 else f'{100 * x:.0f}%'
    for r in runs:                                        # the slow-mixing minority, reported apart from the verdict
        st_ = STATS[r]
        slow = st_['rhat'] > st_['slow_at']
        if slow.any():
            print(f"{r}: slow-mixing minority: {slow.sum()} ruptures ({pct(slow.mean())}) with R-hat > {st_['slow_at']:.4f} (worst {np.nanmax(st_['rhat']):.4f}), "
                  f"carrying {pct(st_['rate'][slow].sum() / st_['rate'].sum())} of the total rate; their effective draws: "
                  f"median {np.median(st_['ess'][slow]):,.0f}, fewest {np.min(st_['ess'][slow]):,.0f}")
        else:
            print(f"{r}: no ruptures with R-hat > {st_['slow_at']:.4f}")
    if any((STATS[r]['rhat'] > STATS[r]['slow_at']).any() for r in runs):
        print('(The threshold is 1.01, or the noise baseline if higher.) These ruptures mix slowly (under theta and spike-and-slab, '
              'mostly ruptures switching on and off), so their means and exceedance probabilities are less precise (the rupture '
              "table gives each one's R-hat and effective draws). Longer runs help them only slowly; moves that exchange the active "
              'role between alternative ruptures would. The replicate comparison below shows whether their stated errors hold.')
    print()
    for r in runs[1:]:
        agreement(runs[0], r, STATS)
    save = a.save or os.path.join(os.path.dirname(out), 'convergence.npz')
    np.savez(save, runs=np.array(runs), rhat=np.array([STATS[r]['rhat'] for r in runs]),
             ess=np.array([STATS[r]['ess'] for r in runs]), slow_at=np.array([STATS[r]['slow_at'] for r in runs]))
    print(f'\nwrote {out} and {save} (per-rupture R-hat and effective draws, for make_tables.py)')


if __name__ == '__main__':
    main()

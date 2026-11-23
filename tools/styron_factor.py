#!/usr/bin/env python3
"""styron_factor.py -- the timeframe factor F (true / measured slip rate) of Coffey et al. (accepted, BSSA), simulated in
full after Styron (2019), in place of the 7 quantiles of their Tables 1 and 2 (the user's request, 2026-11-23).

    python styron_factor.py [--out ../data/styron_factor.npz] [--kyr 100000] [--windows 400000] [--seed 20261123]

The setup is the paper's ("Timeframe or number of events"), for a reference fault of slip rate 1 mm/yr and mean
recurrence 1000 years:
  - periodic faults (SR_pref >= 1 mm/yr): Brownian Passage Time and lognormal recurrence, equal weights, coefficient of
    variation (COV) drawn uniformly in 0.2-0.7;
  - aperiodic faults (< 1 mm/yr): BPT, lognormal and exponential, weights 25/25/50, the exponential's COV 1 and the
    others' drawn uniformly in 1-1.5;
  - every event's displacement lognormal, mean 1 m and sd 0.75 m;
  - each inter-event time drawn from that mixture (kind and COV drawn per event), one long history per case.
The paper leaves the window rule open; the one that reproduces both tables is a window of the given length placed
uniformly in time, its slip rate measured between the earthquakes that bound it, the last before its start and the first
after its end (the displacements of the events after the first up to the last, over the time between them): complete
inter-event times and displacements in equal number, as the paper says a slip rate should hold. Event-bounded windows
not placed uniformly in time (each starting at an event) miss the aperiodic table's median and upper tail by a factor 2
at the short timeframes; fixed windows without the bounding events leave most short windows with no event at all.
Over 2,000 kyr, the paper's history length, six random histories reproduce Table 1 to an rms log error of 0.04-0.08
and Table 2 to 0.10-0.18 (the 7 quantiles at the 8 timeframes); the long history's own errors, 0.04 and 0.11, are within
that scatter, so the tables are this distribution as one 2-Myr history draws it.

Writes OUT (npz): T (the timeframes, years, log-spaced over the tables' 500-100,000), U (the probabilities, NU midpoints),
LNF (2 x len(T) x NU: ln F's quantiles, [0] aperiodic, [1] periodic), and the settings; OUT's _report.txt compares it with
Tables 1 and 2. timeframe_sliprates.log_factor reads it.
"""
import argparse
import os

import numpy as np
from scipy.stats import invgauss

NU = 4000                                                    # timeframe_sliprates' quadrature points in probability
TQ = np.array([0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])   # the tables' quantiles and timeframes, for the comparison
TT = np.array([500, 1000, 2000, 5000, 10000, 20000, 50000, 100000], float)
TAB = {True: np.array([                  # Table 1: periodic faults; rows TT, columns TQ
    [0.34, 0.4176, 0.7778, 1.2304, 2.1031, 4.9631, 6.0027],
    [0.3731, 0.4492, 0.779, 1.1681, 1.904, 4.3238, 5.4015],
    [0.4359, 0.5134, 0.7888, 1.0946, 1.6506, 3.2103, 4.1612],
    [0.5443, 0.6123, 0.8281, 1.0438, 1.3532, 2.1038, 2.4051],
    [0.6484, 0.6899, 0.8619, 1.023, 1.217, 1.6143, 1.7762],
    [0.7367, 0.7782, 0.9028, 1.0102, 1.1369, 1.3495, 1.4402],
    [0.8252, 0.848, 0.9387, 0.9992, 1.0827, 1.2186, 1.2554],
    [0.8646, 0.8816, 0.9486, 1.0081, 1.0534, 1.1605, 1.1727]]),
    False: np.array([                    # Table 2: aperiodic faults
    [0.2316, 0.3014, 0.8097, 1.7175, 3.8223, 10.866, 16.2991],
    [0.277, 0.3402, 0.7363, 1.5019, 3.4848, 10.6151, 14.8125],
    [0.3421, 0.407, 0.7117, 1.2546, 2.7253, 8.9414, 13.3486],
    [0.4406, 0.5089, 0.7512, 1.0759, 1.7596, 5.2667, 7.6325],
    [0.5228, 0.5892, 0.8074, 1.0253, 1.4754, 2.8358, 3.4383],
    [0.6118, 0.6613, 0.852, 1.0195, 1.3025, 2.0205, 2.387],
    [0.7098, 0.7514, 0.9062, 1.0326, 1.1616, 1.5123, 1.7176],
    [0.7853, 0.8156, 0.9338, 1.0251, 1.1158, 1.2966, 1.472]])}
SED_SD = 0.75                                                # the displacements' sd (m); their mean is 1 m


def inter_event_times(periodic, n, g):
    """n inter-event times (kyr, mean 1), each from the paper's mixture: its kind and COV drawn per event."""
    if periodic:
        kind = g.choice(2, n)                                # BPT, lognormal
        cov = g.uniform(0.2, 0.7, n)
    else:
        kind = g.choice(3, n, p=[0.25, 0.25, 0.5])           # BPT, lognormal, exponential
        cov = np.where(kind == 2, 1.0, g.uniform(1.0, 1.5, n))
    out = np.empty(n)
    m = kind == 0                                            # BPT (inverse Gaussian) of mean 1 and aperiodicity cov
    out[m] = invgauss.rvs(cov[m] ** 2, scale=1.0 / cov[m] ** 2, random_state=g)
    m = kind == 1                                            # lognormal of mean 1 and that COV
    s = np.sqrt(np.log1p(cov[m] ** 2))
    out[m] = g.lognormal(-0.5 * s * s, s)
    m = kind == 2
    out[m] = g.exponential(1.0, int(m.sum()))
    return out


def history(periodic, kyr, g):
    """Event times (kyr) and displacements (m) over kyr."""
    t = np.cumsum(inter_event_times(periodic, int(kyr * 1.2) + 1000, g))
    t = t[t < kyr]
    s = np.sqrt(np.log1p(SED_SD ** 2))
    return t, g.lognormal(-0.5 * s * s, s, len(t))


def factor_samples(t, d, T_kyr, n, g):
    """F = 1 / measured slip rate (mm/yr, the true rate 1) for n windows of length T_kyr placed uniformly in time, each
    measured between its bounding earthquakes."""
    cs = np.concatenate([[0.0], np.cumsum(d)])
    s = g.uniform(t[0], t[-1] - T_kyr - 1e-9, n)
    i = np.searchsorted(t, s, 'right') - 1                  # the last event at or before the window's start
    j = np.searchsorted(t, s + T_kyr, 'left')                # the first event at or after its end
    return (t[j] - t[i]) / (cs[j + 1] - cs[i + 1])


def simulate(kyr=100000.0, windows=400000, seed=20261123, nT=41):
    g = np.random.default_rng(seed)
    T = np.exp(np.linspace(np.log(TT[0]), np.log(TT[-1]), nT))
    T = np.concatenate([T, TT])                              # the tables' timeframes exactly among them, near ties dropped
    T = np.unique(np.round(T[np.argsort(T)], 6))
    T = T[np.concatenate([[True], np.diff(np.log(T)) > 0.02])] if len(T) > 1 else T
    T[-1] = TT[-1]
    U = (np.arange(NU) + 0.5) / NU
    LNF = np.empty((2, len(T), NU))
    for per in (False, True):
        t, d = history(per, kyr, g)
        for k, T_ in enumerate(T):
            LNF[int(per), k] = np.quantile(np.log(factor_samples(t, d, T_ / 1000.0, windows, g)), U)
    return T, U, LNF


def compare(T, LNF):
    """Lines comparing the simulated quantiles with Tables 1 and 2."""
    L = []
    for per in (True, False):
        err = []
        for k, T_ in enumerate(TT):
            row = LNF[int(per), int(np.argmin(np.abs(T - T_)))]
            q = np.exp(np.interp(TQ, (np.arange(NU) + 0.5) / NU, row))
            err.append(np.log(q / TAB[per][k]))
            L.append(f'   {"periodic " if per else "aperiodic"} {int(T_):6d} yr: simulated ' + ' '.join(f'{x:7.3f}' for x in q)
                     + '   table ' + ' '.join(f'{x:7.3f}' for x in TAB[per][k]))
        err = np.array(err)
        L.append(f'   {"periodic" if per else "aperiodic"}: rms log error {np.sqrt((err ** 2).mean()):.3f} (by timeframe '
                 + ' '.join(f'{np.sqrt((e ** 2).mean()):.2f}' for e in err) + ')')
    return L


_CACHE = {}


def load(path=None):
    """(T, LNF) from the table in data/ (cached)."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'styron_factor.npz')
    if path not in _CACHE:
        z = np.load(path)
        if int(z['NU']) != NU:
            raise ValueError(f'{path}: NU {int(z["NU"])}, expected {NU}')
        _CACHE[path] = (z['T'], z['LNF'])
    return _CACHE[path]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'styron_factor.npz'))
    ap.add_argument('--kyr', type=float, default=100000.0, help='each history\'s length, kyr (100,000: 100 Myr)')
    ap.add_argument('--windows', type=int, default=400000, help='windows per timeframe')
    ap.add_argument('--seed', type=int, default=20261123)
    a = ap.parse_args()
    T, U, LNF = simulate(a.kyr, a.windows, a.seed)
    out = os.path.abspath(a.out)
    np.savez_compressed(out, T=T, U=U, LNF=LNF, NU=NU, kyr=a.kyr, windows=a.windows, seed=a.seed)
    lines = [f'timeframe factor F = true / measured slip rate (styron_factor.py): one {a.kyr:g}-kyr history per case, '
             f'{a.windows} windows at each of {len(T)} timeframes ({T[0]:g}-{T[-1]:g} years), seed {a.seed}; '
             'quantiles 2.5, 5, 25, 50, 75, 95, 97.5% against Tables 1 and 2'] + compare(T, LNF)
    with open(os.path.splitext(out)[0] + '_report.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()

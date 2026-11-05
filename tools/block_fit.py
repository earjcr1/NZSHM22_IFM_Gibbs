#!/usr/bin/env python3
"""Row-block fit: how well each row block (slip rates, MFD bins, paleo) is fitted by a model's draws.

    python block_fit.py RUN [RUN ...] [--problem cru_clean.npz] [--slip-block S] [--bins-block MFD]
    python block_fit.py --selftest

RUN is a gibbs.py run (its --out, with or without .npz); several runs (replicates) are pooled. Every run records the
predictions of each kept draw, so this works with or without stored draws.

The report, per row block: the number of independent measurements it holds, which is its share of the fit check (the
chi^2 a model matching the truth would have there if the uncertainties are right); the draws' chi^2 over the block
(mean +- sd, with the Monte Carlo error of the mean and its split R-hat); the chi^2 per measurement; z, its distance
from the measurements in units of the fit check's sd for the block; and the mean chi^2 split into the misfit of the
posterior-mean prediction and the posterior spread of the predictions (the two add up exactly, since the predictions
are linear in the rates). The spread is the freedom the model has in the block's rows. Where the model can follow the
data closely, a block's misfit is mostly spread: it is set by the stated uncertainties, whatever they are, and says
little about whether they are right. That is the case for the MFD bins, whose uncertainty level is a modelling choice
(preliminaries.py --mfd-uncertainty, the notebook's MFD_UNCERTAINTY), so the report notes the setting with them.

chi^2 is as the runs' likelihood scores each block (chi2 as fitted): the slip rates by their split normals with the
CFM's, the MFD bins in log10 with gibbs.py --mfd-misfit log10, and the paleo rows by -2 ln of each site's
recurrence-interval PDF with --paleo-misfit, each site's share of the fit check then its misfit's own mean and variance
(paleo_misfit.site_moments) in place of a Gaussian row's 1 and 2.

--selftest checks this arithmetic, and the problem helpers in cru_common.py that set the MFD bins' uncertainty
(magnitude_bins, mfd_uncertainty, scale_rows), on synthetic problems.
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np

from cru_common import block_labels, block_measurements, ess, load_problem, split_rhat, split_weights, report_to


def run_file(r):
    return r if r.endswith('.npz') else r + '.npz'


def block_stats(P, runs, first=('S', 'MFD')):
    """Each row block's chi^2 over the runs' kept draws. Returns (stats per block name, the total, number of draws per
    run): mean, sd, Monte Carlo error, split R-hat, mean-model part and spread, and the mean of each run. The blocks named
    in `first` (the slip-rate and bin rows) come first."""
    meas = sorted(block_measurements(P), key=lambda m: (first.index(m['name']) if m['name'] in first else len(first), m['index']))
    d, sig2, N = P['d'], P['sigma'] ** 2, len(P['d'])
    per_run, S1, S2, nd, shapes = [], np.zeros(N), np.zeros(N), 0, []
    LOG = {}                                                          # --mfd-misfit log10: each row's log width s, by row
    PAL = {}                                                          # --paleo-misfit: each row's site curve, by row
    for r in runs:
        z = np.load(run_file(r))
        if 'y' not in z.files:
            raise SystemExit(f'{r}: no recorded predictions (y); it was made by an older gibbs.py')
        y = z['y']
        if y.ndim != 3 or y.shape[2] != N:
            raise SystemExit(f'{r} has predictions for {y.shape[-1]} rows; {N} expected (a run of another problem?)')
        if y.shape[1] < 4:
            raise SystemExit(f'{r} recorded {y.shape[1]} draws after burn-in; the report needs more')
        c = {}
        if 'mfd_log_s' in z.files:                                    # the run scored the bins in log10 (gibbs.py --mfd-misfit)
            LOG.update(zip(z['mfd_log_rows'].tolist(), z['mfd_log_s'].tolist()))
        if 'paleo_th' in z.files:                                     # the run scored the paleo rows by their sites' curves
            mode_ = str(z['paleo_mode'])
            for j_, r_ in enumerate(z['paleo_rows'].tolist()):
                PAL[int(r_)] = (z['paleo_th'][j_], float(z['paleo_best'][j_]), float(z['paleo_sd'][j_]), mode_,
                                float(z['paleo_E'][j_]), float(z['paleo_V'][j_]))
        for m in meas:
            if PAL and all(int(i) in PAL for i in m['rows']):         # -2 ln of each site's PDF, as the run's likelihood
                c[m['name']] = paleo_misfit_rows(PAL, m['rows'], y[:, :, m['rows']].astype(np.float64))
                continue
            if LOG and all(int(i) in LOG for i in m['rows']):         # (log10(total / R_j) / s_j)^2, as the run's likelihood
                sl = np.array([LOG[int(i)] for i in m['rows']])
                with np.errstate(divide='ignore'):
                    c[m['name']] = np.sum((np.log10(np.maximum(y[:, :, m['rows']].astype(np.float64), 0.0) / d[m['rows']]) / sl) ** 2,
                                          axis=2)
                continue
            res = y[:, :, m['rows']].astype(np.float64) - d[m['rows']]
            c[m['name']] = np.sum(split_weights(P, m['rows'], res) * res ** 2, axis=2) / sig2   # chains x draws (--cfm-sliprates:
                                                                                                 # each side's weight)
        per_run.append(c)
        for k in range(y.shape[0]):                                   # pooled mean and variance of every prediction
            yk = y[k].astype(np.float64)
            S1 += yk.sum(0)
            S2 += (yk ** 2).sum(0)
        nd += y.shape[0] * y.shape[1]
        shapes.append(y.shape[:2])
    ybar = S1 / nd
    var = np.maximum(S2 / nd - ybar ** 2, 0.0)
    same_len = len({s[1] for s in shapes}) == 1

    def summary(x_runs, rows):
        allx = np.concatenate([x.ravel() for x in x_runs])
        n_eff = [float(ess(x)) for x in x_runs]
        n_eff = [max(e, 1.0) if np.isfinite(e) else 1.0 for e in n_eff]
        rh = (float(split_rhat(np.concatenate(x_runs, 0)[:, :, None])[0]) if same_len
              else max(float(split_rhat(x[:, :, None])[0]) for x in x_runs))
        out = dict(mean=float(allx.mean()), sd=float(allx.std()), mcse=float(allx.std() / np.sqrt(sum(n_eff))),
                   rhat=rh, by_run=[float(x.mean()) for x in x_runs],
                   mcse_by_run=[float(x.std() / np.sqrt(e)) for x, e in zip(x_runs, n_eff)])
        if rows is not None and PAL and all(int(i) in PAL for i in rows):   # the paleo sites' curves: the mean model's own,
            mm = float(paleo_misfit_rows(PAL, rows, ybar[rows]))              # the spread the rest of the mean
            out.update(mean_model=mm, spread=float(allx.mean()) - mm)
        elif rows is not None and LOG and all(int(i) in LOG for i in rows):   # the bins in log10: the mean model's own,
            sl = np.array([LOG[int(i)] for i in rows])                        # the spread the rest of the mean
            with np.errstate(divide='ignore'):
                mm = float(np.sum((np.log10(np.maximum(ybar[rows], 0.0) / d[rows]) / sl) ** 2))
            out.update(mean_model=mm, spread=float(allx.mean()) - mm)
        elif rows is not None:
            e = ybar[rows] - d[rows]
            mm = float(np.sum(split_weights(P, rows, e) * e ** 2) / sig2)
            # the spread: the draws' variance about the mean model; with split normals (--cfm-sliprates) the rest of the
            # mean chi2, as the weights make the split inexact
            out.update(mean_model=mm, spread=float(np.sum(var[rows]) / sig2) if P.get('split_lo2') is None
                       else float(allx.mean()) - mm)
        return out

    stats = {}
    for m in meas:
        if PAL and all(int(i) in PAL for i in m['rows']):             # each site's share of the fit check, as fitted
            m['expect'] = float(sum(PAL[int(i)][4] for i in m['rows']))
            m['sd'] = float(np.sqrt(sum(PAL[int(i)][5] for i in m['rows'])))
        s = summary([c[m['name']] for c in per_run], m['rows'])
        s.update(rows=len(m['rows']), measurements=m['expect'], meas_sd=m['sd'])
        stats[m['name']] = s
    allrows = np.concatenate([m['rows'] for m in meas])
    tot = summary([sum(c.values()) for c in per_run], allrows)
    if LOG or PAL:                                                    # log10 bins or paleo curves: the blocks' mean models added up
        tot['mean_model'] = float(sum(st_['mean_model'] for st_ in stats.values()))
        tot['spread'] = tot['mean'] - tot['mean_model']
    tot.update(rows=len(allrows), measurements=sum(m['expect'] for m in meas),
               meas_sd=float(np.sqrt(sum(m['sd'] ** 2 for m in meas))))
    tot['fitted'] = [w_ for w_, on_ in (('the MFD bins in log10', LOG), ('the paleo sites by -2 ln of their recurrence-interval PDFs', PAL)) if on_]
    return stats, tot, shapes


def paleo_misfit_rows(PAL, rows, y):
    """The paleo rows' misfit, summed over rows (the last axis of y, predictions in the rows' units, target bestEstimate /
    sd): 2 (f(ln rate) - f(ln best)) for each site's curve (paleo_misfit.site_misfit), rate = y x the site's sd."""
    from paleo_misfit import site_misfit
    y = np.asarray(y, np.float64)
    out = np.zeros(y.shape[:-1])
    for j, i in enumerate(rows):
        th, best, sd, mode = PAL[int(i)][:4]
        with np.errstate(divide='ignore'):
            out = out + site_misfit(th, best, np.log(np.maximum(y[..., j] * sd, 0.0)), mode)
    return out


def table(stats, tot, labels):
    w = max(len(f'{labels[b]} ({b})') for b in stats) + 1
    lines = [f'{"block":{w}s} {"rows":>5s} {"measurements":>12s} {"chi2 over the draws":>19s} {"MC error":>8s} {"R-hat":>6s} '
             f'{"per meas.":>9s} {"z":>5s} {"mean model":>10s} {"spread":>7s}']
    for name, s in list(stats.items()) + [('total', tot)]:
        lab = 'total' if name == 'total' else (f'{labels[name]} ({name})' if labels[name] != name else name)
        lines.append(f'{lab:{w}s} {s["rows"]:5d} {s["measurements"]:12.1f} {s["mean"]:9.1f} +- {s["sd"]:6.1f} {s["mcse"]:8.2f} '
                     f'{s["rhat"]:6.3f} {s["mean"] / s["measurements"]:9.2f} {(s["mean"] - s["measurements"]) / s["meas_sd"]:+5.1f} '
                     f'{s["mean_model"]:10.1f} {s["spread"]:7.1f}')
    if tot.get('fitted'):
        lines.append(f'chi2 as fitted: each block scored as the runs\' likelihood scores it ({" and ".join(tot["fitted"])}); fit check as fitted '
                     f'{tot["measurements"]:.1f} +- {tot["meas_sd"]:.1f}')
    lines.append('measurements: each block\'s share of the fit check (its chi2 if the model matched the truth and the '
                 'uncertainties were right); z: (chi2 - measurements) / the fit check\'s sd for the block; mean model + spread = '
                 'chi2: the misfit of the posterior-mean prediction, and the posterior variance of the predictions (the freedom '
                 'the model has in the block\'s rows)')
    red = {b: s['measurements'] - s['spread'] for b, s in stats.items()}       # measurements the freedom leaves over
    tot_red = sum(red.values())
    if tot_red > 0:
        overall = sum(s['mean_model'] for s in stats.values()) / tot_red
        part = [f'{labels[b]} {s["mean_model"] / red[b]:.2f} +- {s["mean_model"] / red[b] * np.sqrt(2 / red[b]):.2f} '
                f'({red[b]:.1f} left)' if red[b] > 0.5 else f'{labels[b]} undetermined ({red[b]:.1f} left)'
                for b, s in stats.items()]
        lines.append('beyond the freedom: the mean model\'s chi2 per measurement the freedom leaves over (measurements - spread), '
                     + ', '.join(part) + f'; overall {overall:.2f}. A block above the overall has residuals larger than its '
                     'uncertainties allow, relative to the others (a variance-component estimate); the +- is the scatter '
                     'expected by chance from that many measurements')
    return lines


def mfd_note(P, stats, bins_block='MFD'):
    """A line on the MFD bins' uncertainty setting, and whether their misfit can say anything about it."""
    if bins_block not in stats:
        return None
    s = stats[bins_block]
    rows = np.flatnonzero(np.asarray(P['block']) == P['block_names'].index(bins_block))
    rel = 1.0 / P['d'][rows]
    c = P['mfd_scale']
    txt = (f"MFD bins: uncertainties {P['mfd_uncertainty']!r}, relative {100 * rel.min():.1f}% to {100 * rel.max():.1f}%"
           + (f", x{c:.3f} the inputs'" if abs(c - 1) > 1e-12 else ', as the inputs give them'))
    if s['spread'] > 0.5 * s['mean']:
        txt += (f"; their misfit is mostly spread ({s['spread']:.1f} of {s['mean']:.1f}): it follows their width, whatever it "
                'is, so it cannot tell whether that width is right')
    return txt


def main():
    ap = argparse.ArgumentParser(description='Row-block fit (see the module docstring).')
    ap.add_argument('runs', nargs='*', metavar='RUN', help="gibbs.py runs (their --out); several are pooled")
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem the runs sampled (default cru_clean.npz)')
    ap.add_argument('--slip-block', default='S', help='the slip-rate rows (default S), for the labels')
    ap.add_argument('--bins-block', default='MFD', help='the magnitude-bin rows (default MFD), for the labels')
    ap.add_argument('--out', default=None, help='also write the report to this text file')
    ap.add_argument('--selftest', action='store_true', help='check the arithmetic and the MFD helpers on synthetic problems')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.runs:
        ap.error('give at least one run')
    if a.out:
        report_to(a.out)
    P = load_problem(a.problem)
    labels = block_labels(P['block_names'], a.slip_block, a.bins_block)
    stats, tot, shapes = block_stats(P, a.runs, (a.slip_block, a.bins_block))
    desc = ', '.join(f'{r} ({K} chains x {D} draws)' for r, (K, D) in zip(a.runs, shapes))
    print(f'row-block fit of {desc}, on {a.problem}:')
    print('\n'.join(table(stats, tot, labels)))
    note = mfd_note(P, stats, a.bins_block)
    if note:
        print(note)


# --------------------------------------------------------------------------- self-test
def selftest():
    import scipy.sparse as sp
    from cru_common import (chi2_expectation, save_problem, magnitude_bins, mfd_uncertainty, mfd_implied, scale_rows,
                            bin_targets, MFD_UNCERTAINTY_OPTIONS)
    g = np.random.default_rng(7)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(('  ok   ' if cond else '  FAIL ') + msg)
        ok &= bool(cond)

    # a small cleaned problem: 6 slip-rate rows in sections of 2, 3 and 1 (divided by sqrt(k)), 4 bins, 5 other rows,
    # one all-zero row
    N, n = 16, 40
    G = sp.random(N, n, density=0.4, random_state=3, format='csr', data_rvs=lambda k: g.uniform(1, 5, k)).toarray()
    G[15] = 0.0
    G[[0, 1]] = G[0]
    G[[2, 3, 4]] = G[2]
    d = g.uniform(2, 10, N)
    d[1], d[3], d[4], d[15] = d[0], d[2], d[2], 1.0
    sec = np.full(N, -1)
    sec[:6] = [0, 0, 1, 1, 1, 2]
    k = np.array([2, 2, 3, 3, 3, 1])
    row_scale = np.ones(N)
    row_scale[:6] = 1 / np.sqrt(k)
    G, d = sp.csc_matrix(sp.diags(row_scale) @ G), d * row_scale
    block = np.array([0] * 6 + [1] * 4 + [2] * 5 + [3])
    names = ['S', 'MFD', 'B2:var', 'empty']
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'cru_clean.npz')
        save_problem(path, G, d, 1.0, 0.0, 1.0, np.arange(N), np.arange(n), block, names, ['0-5', '6-9', '10-14', '1 rows'],
                     raw_shape=(N, n), row_scale=row_scale, section=sec, slip_sections='auto',
                     chi2_expect=np.array(chi2_expectation([2, 3, 1], 1 / np.sqrt([2, 3, 1]), 9, 1.0)))
        P = load_problem(path)
        check(P['mfd_uncertainty'] == 'as-given' and P['mfd_scale'] == 1.0,
              'a problem saved without an MFD setting loads as the inputs\' uncertainties (as-given, scale 1)')
        meas = {m['name']: m for m in block_measurements(P)}
        check(sorted(meas) == ['B2:var', 'MFD', 'S'] and [meas[b]['expect'] for b in ('S', 'MFD', 'B2:var')] == [3, 4, 5],
              f'measurements per block: S 3 (three sections), MFD 4, other 5, all-zero row left out: '
              f'{ {b: meas[b]["expect"] for b in meas} }')
        check(abs(sum(m['expect'] for m in meas.values()) + 1.0 - P['chi2_expect'][0]) < 1e-12,
              'the blocks\' measurements add up to the fit check, less the all-zero row\'s constant')
        # draws: predictions scattered about the targets with a different size in each block (x1, x0.5, x1.5)
        sc = np.array([1.0] * 6 + [0.5] * 4 + [1.5] * 5 + [0.0])
        runs = []
        for r in range(2):
            y = (d + 0.3 + sc * g.standard_normal((3, 400, N))).astype(np.float32)
            np.savez(os.path.join(tmp, f'run{r}.npz'), y=y)
            runs.append(os.path.join(tmp, f'run{r}'))
        stats, tot, _ = block_stats(P, runs)
        Y = np.concatenate([np.load(r + '.npz')['y'].reshape(-1, N).astype(np.float64) for r in runs])
        direct = {b: float(np.mean(np.sum((Y[:, meas[b]['rows']] - d[meas[b]['rows']]) ** 2, 1))) for b in meas}
        check(all(abs(stats[b]['mean'] - direct[b]) < 1e-9 * direct[b] for b in meas), 'block chi2 means match a direct computation')
        check(all(abs(stats[b]['mean_model'] + stats[b]['spread'] - stats[b]['mean']) < 1e-9 * stats[b]['mean'] for b in meas),
              'mean model + spread = mean chi2, in every block')
        check(abs(tot['mean'] - sum(direct.values())) < 1e-9 * tot['mean'], 'the total is the sum of the blocks')
        check(all(s['rhat'] < 1.02 for s in stats.values()), 'independent draws give block chi2 split R-hat near 1')
        y = np.load(runs[1] + '.npz')['y'].astype(np.float64)
        y[:, :, 10:15] += np.array([0.0, 0.6, 1.2])[:, None, None]  # chains stuck apart in the other rows
        np.savez(os.path.join(tmp, 'stuck.npz'), y=y.astype(np.float32))
        st2, _, _ = block_stats(P, [runs[0], os.path.join(tmp, 'stuck')])
        check(st2['B2:var']['rhat'] > 1.01 and st2['S']['rhat'] < 1.02,
              f'chains stuck apart in one block show in that block\'s R-hat ({st2["B2:var"]["rhat"]:.3f}) and not in others')
        note = mfd_note(P, stats)
        check(note is not None and 'as the inputs give them' in note, 'the MFD note names the setting: ' + str(note))
    finally:
        shutil.rmtree(tmp)

    # the MFD bins' uncertainty: B bins with the NZSHM22 inputs' structure (row = constraint / sigma_j, coefficient
    # 1/sigma_j for each of the bin's ruptures, target R_j / sigma_j), relative uncertainties 0.1 (R_j / R_ref)^-0.25,
    # the bin rows out of magnitude order and one bin missing (removed by cleaning), plus two other rows
    B, per = 7, 3
    Rt = 2e-3 * 10.0 ** (-0.08 * np.arange(B))         # targets by magnitude, from 6.8-6.9 up
    orig_bins = np.array([20, 21, 22, 24, 25, 26, 27])   # original rows: 23 (bin 7.1-7.2) removed by cleaning
    ref_k = 2                                             # 7.0-7.1, original row 22
    rel_t = 0.1 * (Rt / Rt[ref_k]) ** -0.25
    nr, nc = B + 2, B * per
    Gm = np.zeros((nr, nc))
    dm = np.zeros(nr)
    perm = np.array([3, 0, 6, 1, 5, 2, 4])                # bin k sits in cleaned row position perm[k]
    for kb in range(B):
        sig = rel_t[kb] * Rt[kb]
        Gm[perm[kb], kb * per:(kb + 1) * per] = 1.0 / sig
        dm[perm[kb]] = Rt[kb] / sig
    Gm[B] = g.uniform(1, 3, nc)
    Gm[B + 1] = g.uniform(1, 3, nc)
    dm[B:] = [4.0, 5.0]
    keep_rows = np.r_[orig_bins[np.argsort(perm)], 40, 41]
    blk = np.r_[np.zeros(B, int), 1, 1]
    PM = dict(G=sp.csc_matrix(Gm), d=dm, block=blk, block_names=['MFD', 'P'], keep_rows=keep_rows)
    rows, Rk, orig, lo = magnitude_bins(PM, 'MFD', 6.8, 0.1)
    expect_lo = np.array([6.8, 6.9, 7.0, 7.2, 7.3, 7.4, 7.5])
    check(np.array_equal(orig, orig_bins) and np.allclose(lo, expect_lo) and np.allclose(Rk, Rt) and np.array_equal(rows, perm),
          'magnitude_bins: bins in original row order, counted up from 6.8 in steps of 0.1, a removed bin leaving a gap')
    r_ag = mfd_uncertainty(PM, 'as-given')
    check(r_ag['scale'] == 1.0 and np.allclose(r_ag['rel_in'], rel_t) and r_ag['ref'] == ref_k
          and abs(r_ag['power'] - 0.25) < 1e-9 and r_ag['power_resid'] < 1e-9,
          f"as-given: scale 1, the inputs' relative uncertainties, the 7.0-7.1 bin found and the inputs' power "
          f"recovered ({r_ag['power']:.6f})")
    r_2 = mfd_uncertainty(PM, 'two-measurements')
    check(abs(r_2['scale'] - np.sqrt(B / 2)) < 1e-12 and np.allclose(r_2['rel_out'], np.sqrt(B / 2) * rel_t),
          f'two-measurements: every bin\'s uncertainty multiplied by sqrt(B / 2) = {np.sqrt(B / 2):.4f}, the shape kept')
    r_u = mfd_uncertainty(PM, '0.25')
    check(abs(r_u['scale'] - 2.5) < 1e-12 and abs(r_u['rel_out'][ref_k] - 0.25) < 1e-12,
          'a number sets the 7.0-7.1 bin\'s relative uncertainty (0.25: scale 2.5, the other bins following)')
    # the implied uncertainty of the MFD as a whole: the information two measurements would carry
    sig_b = rel_t * Rt
    t_dir = np.sqrt(np.sum(sig_b ** 2)) / Rt.sum()
    x = expect_lo + 0.05
    w = (np.log(10) / rel_t) ** 2
    X = np.c_[np.ones(B), x]
    slope_dir = np.sqrt(np.linalg.inv(X.T @ (w[:, None] * X))[1, 1])
    check(abs(r_ag['total_in'] - t_dir) < 1e-12 * t_dir and abs(r_ag['slope_in'] - slope_dir) < 1e-9 * slope_dir
          and abs(r_2['total_out'] / r_2['total_in'] - np.sqrt(B / 2)) < 1e-12,
          f"implied sd of the bins' total rate ({100 * t_dir:.2f}%) and of their slope in b ({slope_dir:.4f}) match a direct "
          'least-squares computation, and scale with the bins\' uncertainty')
    # applying it: the bins' rows divided by the scale, their targets R_j and every other row unchanged
    rs0 = np.ones(nr)
    G2, d2, rs2 = scale_rows(PM['G'], dm, rs0, r_2['rows'], 1.0 / r_2['scale'])
    P2 = dict(PM, G=G2, d=d2)
    R2, rows2 = bin_targets(P2, 'MFD')
    check(np.allclose(np.sort(R2), np.sort(Rt), rtol=1e-14) and np.allclose(1.0 / d2[r_2['rows']], r_2['rel_out'], rtol=1e-14)
          and np.array_equal(d2[B:], dm[B:]) and np.allclose(G2.toarray()[B:], Gm[B:], rtol=0, atol=0)
          and np.allclose(rs2[r_2['rows']], 1.0 / r_2['scale']) and np.all(rs2[B:] == 1.0),
          'scale_rows: the bin rows divided by the scale (targets R_j unchanged, relative uncertainties as set), the '
          'other rows and their row scales unchanged')
    for bad, why in (('0', 'zero'), ('-0.1', 'negative'), ('5', 'above 2 (a percentage?)'), ('two', 'unknown words')):
        try:
            mfd_uncertainty(PM, bad)
            check(False, f'{bad!r} ({why}) is refused')
        except ValueError:
            check(True, f'{bad!r} ({why}) is refused')
    try:
        mfd_uncertainty(PM, '0.2', lowest=7.15)
        check(False, 'a number is refused when no bin starts at M7.0')
    except ValueError:
        check(True, 'a number is refused when no bin starts at M7.0 (the magnitudes would be wrong)')
    check(MFD_UNCERTAINTY_OPTIONS == ('as-given', 'two-measurements'), 'the named settings are as-given (the default) and two-measurements')
    check(mfd_uncertainty(PM)['scale'] == 1.0, 'with no setting given, the bins keep the inputs\' uncertainties (as-given)')
    print('self-test ' + ('passed' if ok else 'FAILED'))
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()

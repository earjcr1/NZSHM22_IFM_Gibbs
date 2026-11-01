#!/usr/bin/env python3
"""summarize_runs.py -- summarize gibbs.py runs in log space: fit, bin totals, the population of rates, per-rate widths,
and how each run's per-rate medians compare with a reference run's (replicates of one prior, or different priors).

    python summarize_runs.py main_rep0:rep0 main_rep1:rep1 --ref main_rep0 --out fig_runs.png

Each argument is <run>:<label>. The statistics come from each run's running summaries (<run>_stream.npz, written by
every run), so they are the same whether or not its draws were stored; runs made before those existed use their stored
draws (<run>_samples.npy), and --from-draws uses stored draws wherever they exist. Rates below the diagnostic floor
(--diag-floor, 1e-10 by default, the notebook's FLOOR, as in its convergence checks; 1e-16, the summaries' resolution,
at the lowest) count as one interval in the statistics. --ref names the run the others are compared with rate by rate.
Writes the figure and, next to it, the text summary (.txt).
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cru_common import load_problem, ess, bin_targets, load_stream, stream_quantiles, stream_rhat

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+')
ap.add_argument('--problem', default='cru_clean.npz')
ap.add_argument('--ref', default=None)
ap.add_argument('--out', default='fig_runs.png')
ap.add_argument('--draws', type=int, default=200, help='draws per chain taken from stored draws (with --from-draws, or for '
                'runs without running summaries)')
ap.add_argument('--from-draws', action='store_true', help='use stored draws wherever they exist, instead of the summaries')
ap.add_argument('--slip-block', default='S', help='block of slip-rate rows (subsection totals)')
ap.add_argument('--bins-block', default='MFD', help='block of magnitude-bin rows')
ap.add_argument('--diag-floor', type=float, default=-10.0, help='log10 rate below which rates count as one interval in the '
                'statistics (the notebook passes FLOOR); at least -16, the running summaries\' resolution')
a = ap.parse_args()

HIST_LO = -16.0                                # the running summaries resolve rates down to 1e-16
LO = max(a.diag_floor, HIST_LO)                # rates below 10^LO count as one interval in the statistics
if a.diag_floor < HIST_LO:
    print(f'note: --diag-floor {a.diag_floor:g} is below the running summaries\' resolution; using 1e{HIST_LO:g}')
THRESH = (1e-6, 1e-5, 1e-4, 1e-3)
PCT = (1, 5, 25, 50, 75, 95, 99)
EDGES = np.round(np.arange(HIST_LO, -0.95, 0.1), 2)   # 0.1-decade bins from 1e-16 to 1e-1 for the population panel

P = load_problem(a.problem)
d, block, names = P['d'], P['block'], P['block_names']
mfd = np.flatnonzero(block == names.index(a.bins_block)) if a.bins_block in names else np.array([], int)
target_total = float(bin_targets(P, a.bins_block)[0].sum()) if len(mfd) else None
Gr = P['G'].tocsr()
inc = (Gr[np.flatnonzero(block == names.index(a.slip_block))] > 0).astype(float) if a.slip_block in names else None


def pct(L, q, axis=None):
    """Percentiles taken as actual draws (inverted CDF), matching the summaries' histogram quantiles at the lumped interval."""
    try:
        return np.percentile(L, q, axis=axis, method='inverted_cdf')
    except TypeError:                            # numpy before 1.22
        return np.percentile(L, q, axis=axis, interpolation='lower')


def from_draws(run):
    """Statistics from stored draws (a subsample of --draws per chain)."""
    S = np.load(run + '_samples.npy', mmap_mode='r')
    K, D, n = S.shape
    take = np.linspace(0, D - 1, min(a.draws, D)).astype(int)
    X = np.asarray(S[:, take], dtype=np.float64).reshape(-1, n)
    L = np.log10(np.maximum(X, 10.0 ** LO))      # below the floor: at the floor, one interval
    q05, med, q95 = pct(L, [5, 50, 95], axis=0)
    srt = -np.sort(-X, axis=1)
    cum = np.cumsum(srt, 1) / srt.sum(1, keepdims=True)
    return dict(K=K, D=D, src=f'stored draws ({len(take)} per chain)', med=med, q95=q95, w95=q95 - q05,
                pop=pct(L, PCT), below=float(np.mean(X < 10.0 ** LO)),
                cnt=[np.mean(X > t) * n for t in THRESH], cnt_med=[int(np.sum(med > np.log10(t))) for t in THRESH],
                below8=float(np.mean(X < 1e-8)), small=float(np.mean(np.sum(np.where(X < 1e-8, X, 0), 1) / X.sum(1))),
                n90=float(np.median(np.argmax(cum >= .9, 1) + 1)), n99=float(np.median(np.argmax(cum >= .99, 1) + 1)),
                sub=np.median(inc @ X.T, 1) if inc is not None else None,
                hist=np.histogram(np.log10(X[X >= 10.0 ** HIST_LO]), EDGES)[0].astype(np.float64), n_values=float(X.size))


def from_summaries(run):
    """The same statistics from the running summaries: every recorded draw, rates resolved to the histogram's bins."""
    st = load_stream(run, ('chains', 'n_keep', 'hist_lo', 'hist_w', 'hist', 'hcnt', 'tot_s', 'n90', 'n99'))
    K, D = int(st['chains']), int(st['n_keep'])
    lo, w = float(st['hist_lo']), float(st['hist_w'])
    if abs(lo - HIST_LO) > 1e-9:
        raise SystemExit(f'{run}: running summaries with a histogram starting at 1e{lo:g}, not 1e{HIST_LO:g}')
    h = st['hist']
    nd = float(st['hcnt'][0])                    # draws counted in the histogram (chains x recorded draws)
    q05, med, q95 = (np.log10(np.maximum(q, 10.0 ** LO)) for q in stream_quantiles(st, (0.05, 0.5, 0.95)))
    H = h.sum(1).astype(np.float64)              # all ruptures together; bin 0 holds rates below 1e-16, bin k >= 1 [lo+(k-1)w, lo+kw)
    pop = np.log10(np.maximum(np.concatenate(stream_quantiles(dict(hist=H[:, None], hist_lo=lo, hist_w=w),
                                                               [p / 100 for p in PCT])), 10.0 ** LO))
    first = lambda t: 1 + int(round((np.log10(t) - lo) / w))        # first bin at or above rate t
    centre = np.concatenate([[0.0], 10.0 ** (lo + (np.arange(1, len(H)) - 0.5) * w)])
    k8 = first(1e-8)
    hist = np.zeros(len(EDGES) - 1)
    j = np.floor((lo + np.arange(len(H) - 1) * w - HIST_LO) / 0.1 + 1e-9).astype(int)   # 0.1-decade bin of each histogram bin
    ok = (j >= 0) & (j < len(hist))
    np.add.at(hist, j[ok], H[1:][ok])
    out = dict(K=K, D=D, src='running summaries (every recorded draw)', med=med, q95=q95, w95=q95 - q05, pop=pop,
               below=float(H[:first(10.0 ** LO)].sum() / H.sum()), cnt=[h[first(t):].sum() / nd for t in THRESH],
               cnt_med=[int(np.sum(med > np.log10(t))) for t in THRESH], below8=float(H[:k8].sum() / H.sum()),
               small=float((H[:k8] * centre[:k8]).sum() / (H * centre).sum()),
               sub=np.median(st['tot_s'].reshape(-1, st['tot_s'].shape[2]), 0) if 'tot_s' in st else None,
               hist=hist, n_values=float(H.sum()))
    if 'n90' in st:
        out.update(n90=float(np.median(st['n90'])), n99=float(np.median(st['n99'])))
    elif os.path.exists(run + '_samples.npy'):   # summaries from before these counts were recorded: take them from the draws
        dr = from_draws(run)
        out.update(n90=dr['n90'], n99=dr['n99'], n90_src='stored draws')
    return out


fmt_pop = lambda v: f'<{LO:g}' if v <= LO + 1e-9 else f'{v:.2f}'
FL = f'1e{LO:g}'                               # the floor, as text
runs = [r.split(':') if ':' in r else (r, r) for r in a.runs]
color = {run: f'C{i}' for i, (run, _) in enumerate(runs)}
res, lines = {}, []
for run, lab in runs:
    has_st, has_dr = os.path.exists(run + '_stream.npz'), os.path.exists(run + '_samples.npy')
    if not (has_st or has_dr):
        raise SystemExit(f'{run}: no running summaries ({run}_stream.npz) or stored draws ({run}_samples.npy)')
    r = from_draws(run) if has_dr and (a.from_draws or not has_st) else from_summaries(run)
    z = dict(np.load(run + '.npz'))
    w = int(z['burn']) // int(z['thin'])
    chi2, tot = z['chi2'][:, w:], z['sum_m'][:, w:]
    y = z['y'].reshape(-1, len(d))
    mfd_ratio = np.median(y[:, mfd] / d[mfd], 0) if len(mfd) else np.array([np.nan])
    if has_st:
        st_ = load_stream(run, ('cnt', 's1', 's2', 'clip'))
        rh = f'worst split R-hat of log rate {np.nanmax(stream_rhat(st_)):.4f} (rates below {float(st_["clip"]):.0e} counted as equal)'
    else:
        rh = f'worst split R-hat {np.nanmax(z["post_rhat"]):.4f}' if 'post_rhat' in z else 'R-hat not available'
    r.update(lab=lab)
    res[run] = r
    act = r['q95'] > LO + 1e-9                  # rates whose 95% point is above the floor (the rest lie within one interval)
    width = (f'median {np.median(r["w95"]):.2f} decades' if act.all() else
             f'median {np.median(r["w95"][act]):.2f} decades over the {act.sum()} rates whose 95% point is above {FL}' if act.any()
             else f'all rates below {FL}')
    n9 = (f'{r["n90"]:.0f} / {r["n99"]:.0f}' + (f' (from {r["n90_src"]})' if 'n90_src' in r else '')) if 'n90' in r \
        else 'not available (summaries from an older gibbs.py, and no stored draws)'
    lines += [f'== {lab} ({run}): {r["K"]} chains x {r["D"]} draws; statistics from {r["src"]}',
              f'   chi2 {chi2.mean():.0f} +- {chi2.std():.0f} (ESS {ess(chi2):.0f}); predictions/data {z["tscale"][:, w:].mean():.3f}; '
              f'total rate {tot.mean():.4f} +- {tot.std():.4f}' + (f' (bin targets total {target_total:.4f})' if target_total else ''),
              f'   magnitude-bin rows predicted/observed: median {np.median(mfd_ratio):.3f}, range {mfd_ratio.min():.3f}-{mfd_ratio.max():.3f}',
              f'   population of log10 rates (all draws): 1/5/25/50/75/95/99% = ' + ' '.join(fmt_pop(v) for v in r['pop'])
              + (f'; {100 * r["below"]:.1f}% below {FL}' if r['below'] > 0 else ''),
              f'   expected number of rates above 1e-6/1e-5/1e-4/1e-3 per draw: ' + ' / '.join(f'{c:.0f}' for c in r['cnt'])
              + f';  rates whose posterior median is above them: ' + ' / '.join(str(c) for c in r['cnt_med']),
              f'   draws below 1e-8: {100 * r["below8"]:.1f}% (carrying {100 * r["small"]:.2f}% of the total rate); per-rate 90% '
              f'interval width: {width}' + (f' (lower ends below {FL} counted at {FL})' if r['below'] > 0 else ''),
              f'   ruptures carrying 90% / 99% of the total rate (median over draws): {n9}',
              f'   {rh}' + (f'; swap acceptance {z["swap_rate"].mean():.3f}' if 'swap_rate' in z and z['swap_rate'].any() else '')]
if a.ref and a.ref in res:
    ref = res[a.ref]
    for run, lab in runs:
        if run == a.ref:
            continue
        dm = res[run]['med'] - ref['med']
        lines.append(f'== {lab} vs {ref["lab"]}: shift of per-rate posterior medians (decades): 5% {np.percentile(dm, 5):+.2f}, '
                     f'median {np.median(dm):+.2f}, 95% {np.percentile(dm, 95):+.2f}; |shift| > 0.5 decade for {np.mean(abs(dm) > .5) * 100:.0f}% of rates')
        if ref['sub'] is not None and res[run]['sub'] is not None:
            pos = (res[run]['sub'] > 0) & (ref['sub'] > 0)      # a median total of exactly 0 (spike-and-slab) has no ratio
            rs = res[run]['sub'][pos] / ref['sub'][pos]
            lines.append(f'   subsection total rates (median over draws), ratio to {ref["lab"]}: median {np.median(rs):.3f}, '
                         f'5-95% {np.percentile(rs, 5):.3f}-{np.percentile(rs, 95):.3f}'
                         + (f' ({int((~pos).sum())} subsections with a median of 0 in either run left out)' if (~pos).any() else ''))
print('\n'.join(lines))
open(os.path.splitext(a.out)[0] + '.txt', 'w').write('\n'.join(lines) + '\n')

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for run, lab in runs:
    r = res[run]
    dens = r['hist'] / (r['n_values'] * 0.1)
    ax[0].hist(EDGES[:-1], EDGES, weights=dens, histtype='step', lw=1.2, color=color[run],
               label=lab + (f' ({100 * r["below"]:.0f}% of draws below {FL})' if r['below'] >= 0.005 else ''))
for t in (-8, -2):
    ax[0].axvline(t, color='grey', ls=':', lw=.8)
if LO > HIST_LO:
    ax[0].axvspan(HIST_LO, LO, color='0.9', zorder=0, label=f'below {FL}: one interval in the statistics')
ax[0].set_xlim(HIST_LO, -1)
ax[0].set_xlabel('log10 rate'); ax[0].set_ylabel('density'); ax[0].legend(fontsize=8)
ax[0].set_title('Population of rates (all draws of all ruptures)')
if a.ref and a.ref in res and len(runs) > 1:
    lowest = []
    for run, lab in runs:
        if run != a.ref:
            x, y_ = res[a.ref]['med'], res[run]['med']
            shown = (x > LO + 1e-9) | (y_ > LO + 1e-9)          # rates with both medians below the floor: left out
            ax[1].plot(x[shown], y_[shown], '.', ms=3 if shown.sum() < 3000 else 1, alpha=.5, color=color[run],
                       label=lab + (f' ({(~shown).sum()} rates below {FL} in both, not shown)' if (~shown).any() else ''))
            if shown.any():
                lowest.append(min(x[shown].min(), y_[shown].min()))
    lim = [max(LO, np.floor(min(lowest)) - 0.5) if lowest else LO, -2]
    ax[1].plot(lim, lim, 'k-', lw=.6)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel(f'per-rate posterior median, log10 ({res[a.ref]["lab"]})'); ax[1].set_ylabel('per-rate posterior median, log10')
    ax[1].legend(fontsize=8, markerscale=2 if max(np.sum((res[a.ref]['med'] > LO + 1e-9) | (res[r]['med'] > LO + 1e-9))
                                                for r, _ in runs) < 3000 else 6)
    ax[1].set_title(f'Rate by rate: posterior medians against {res[a.ref]["lab"]}')
else:
    ax[1].axis('off')
    ax[1].text(0.5, 0.5, 'one run: no rate-by-rate comparison', ha='center', va='center', transform=ax[1].transAxes)
fig.tight_layout()
fig.savefig(a.out, dpi=120)
print('figure:', a.out)

#!/usr/bin/env python3
"""compare_runs.py -- check two runs of the same target parameter by parameter.

    python compare_runs.py --a <run> --b <run>        any two runs of the same model (stored draws needed)

For every parameter: z = (mean_b - mean_a) / sqrt(sd_a^2/ESS_a + sd_b^2/ESS_b), with per-parameter ESS from both
runs.  If both runs target the same distribution, z ~ N(0, 1) over the parameters. Writes compare_summary.txt,
compare_results.npz and fig_compare.png.
"""
import argparse

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from cru_common import ess_many

ap = argparse.ArgumentParser()
ap.add_argument('--a', required=True, help='first run (<run>.npz and <run>_samples.npy)')
ap.add_argument('--b', required=True, help='second run')
ap.add_argument('--outdir', default='.', help='folder for the summary, results and figure')
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
out = lambda f: os.path.join(args.outdir, f)

A, B = (dict(np.load(f + '.npz')) for f in (args.a, args.b))
SA, SB = (np.load(f + '_samples.npy', mmap_mode='r') for f in (args.a, args.b))
ess_a, ess_b = ess_many(SA), ess_many(SB)
with np.errstate(invalid='ignore', divide='ignore'):
    z = (B['post_mean'] - A['post_mean']) / np.sqrt(A['post_sd'] ** 2 / ess_a + B['post_sd'] ** 2 / ess_b)
    rel_mean = np.abs(B['post_mean'] / A['post_mean'] - 1)
    rel_q95 = np.abs(B['post_q95'] / A['post_q95'] - 1)
ok = np.isfinite(z)                                  # rates constant in both runs (e.g. always 0) have no z
zz = z[ok]
za = np.abs(zz)
lines = [
    f'{args.a}: {SA.shape[0]} chains x {SA.shape[1]} draws; per-parameter ESS median {np.median(ess_a):.0f} (5% {np.percentile(ess_a, 5):.0f})',
    f'{args.b}: {SB.shape[0]} chains x {SB.shape[1]} draws; per-parameter ESS median {np.median(ess_b):.0f} (5% {np.percentile(ess_b, 5):.0f})',
    f'z over {ok.sum()} parameters' + (f' ({(~ok).sum()} constant in both runs, e.g. always 0, left out)' if (~ok).any() else '')
    + f': mean {zz.mean():+.3f}, sd {zz.std():.3f}; |z| > 3: {np.mean(za > 3) * 100:.2f}% '
    f'(expected 0.27%); |z| > 4: {np.sum(za > 4)}; max |z| {za.max():.2f}',
    f'median relative difference of means {np.nanmedian(np.where(np.isfinite(rel_mean), rel_mean, np.nan)) * 100:.2f}%; '
    f'of 95th percentiles {np.nanmedian(np.where(np.isfinite(rel_q95), rel_q95, np.nan)) * 100:.2f}% (rates nonzero in the first run)',
    f'max split R-hat: {args.a} {np.nanmax(A["post_rhat"]):.4f}, {args.b} {np.nanmax(B["post_rhat"]):.4f}',
]
print('\n'.join(lines))
open(out('compare_summary.txt'), 'w').write('\n'.join(lines) + '\n')
np.savez(out('compare_results.npz'), z=z, ess_a=ess_a, ess_b=ess_b)

# ---- figure: the two runs rate by rate
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].loglog(A['post_mean'], B['post_mean'], ',', alpha=.5)
lim = [A['post_mean'].min() / 2, A['post_mean'].max() * 2]
ax[0].plot(lim, lim, 'k-', lw=.5)
ax[0].set_xlabel(f'posterior mean, {args.a}'); ax[0].set_ylabel(f'posterior mean, {args.b}'); ax[0].set_title(f'All {len(z):,} rates')
ax[1].hist(zz, 100, density=True, alpha=.7, label=f'z over {ok.sum():,} rates')
t = np.linspace(-5, 5, 200)
ax[1].plot(t, np.exp(-t * t / 2) / np.sqrt(2 * np.pi), 'k', lw=1, label='N(0,1)')
ax[1].set_xlabel('standardised difference of means'); ax[1].legend(fontsize=8); ax[1].set_title('Do the samplers agree?')
j = 1
xa = A['mon'][:, A['mon'].shape[1] // 4:, j].ravel()
xb = B['mon'][:, int(B['warmup']) if 'warmup' in B else B['mon'].shape[1] // 4:, j].ravel()
bins = np.linspace(0, np.quantile(np.r_[xa, xb], .995), 60)
ax[2].hist(xa, bins, density=True, alpha=.5, label=args.a)
ax[2].hist(xb, bins, density=True, alpha=.5, label=args.b)
ax[2].ticklabel_format(style='sci', axis='both', scilimits=(0, 0)); ax[2].set_xlabel(f'm[{int(A["monitor"][j])}]'); ax[2].legend(fontsize=8); ax[2].set_title('One monitored marginal')
fig.tight_layout(); fig.savefig(out('fig_compare.png'), dpi=120); plt.close(fig)

print('figure: fig_compare.png')

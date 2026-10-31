#!/usr/bin/env python3
"""compare_priors.py -- compare gibbs.py runs made under different priors (the per-bin and theta tables in REPORT.md).

    python compare_priors.py gibbs_bb10 gibbs_bb12 gibbs_th2 gibbs_th1 gibbs_th0.5 [--problem cru_clean.npz]

For each run (<run>.npz and <run>_samples.npy, cleaned problem in cru_clean.npz): chi2, chi2 of the posterior-mean
model, total rate, bin totals against their targets, ruptures above 1e-5 and 1e-4, ruptures carrying 90% of the rate,
the largest rate relative to its bin's target, worst R-hat against its noise baseline, and effective draws. Then, for
pairs of runs it recognizes, the change in subsection totals, exceedance probabilities and linear means, and per-bin
totals against targets in units of each bin's uncertainty. Labels are defined for the run names used in REPORT.md;
other names are labelled by their file stem.
"""
import argparse, os, sys
import numpy as np
from cru_common import load_problem, split_rhat, ess, ess_many, bin_targets
ap = argparse.ArgumentParser(description='compare gibbs.py runs made under different priors (needs their stored draws)')
ap.add_argument('runs', nargs='+', help='run names: <run>.npz and <run>_samples.npy')
ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem the runs were made on')
ap.add_argument('--slip-block', default='S', help='block of slip-rate rows (subsection totals)')
ap.add_argument('--bins-block', default='MFD', help='block of magnitude-bin rows')
args = ap.parse_args()
P = load_problem(args.problem); G = P['G'].tocsr(); d = P['d']; n = G.shape[1]
R, brow = bin_targets(P, args.bins_block)
bins = np.full(n, -1)
for k, j in enumerate(brow):
    bins[G.indices[G.indptr[j]:G.indptr[j + 1]]] = k
srow = np.flatnonzero(P['block'] == P['block_names'].index(args.slip_block))
inc_s = (G[srow] > 0).astype(float); inc_b = (G[brow] > 0).astype(float)

def stats(run):
    z = dict(np.load(run + '.npz')); S = np.load(run + '_samples.npy', mmap_mode='r')
    K, D, _ = S.shape; w = int(z['burn']) // int(z['thin'])
    chi = z['chi2'][:, w:]; tot = z['sum_m'][:, w:]
    p5, p4, lin, rh, es = (np.empty(n) for _ in range(5))
    topd = np.zeros(K * D)                                   # largest m / R_j in each draw, over every draw
    for j0 in range(0, n, 3000):
        X = np.asarray(S[:, :, j0:j0 + 3000], dtype=np.float64); j1 = j0 + X.shape[2]
        F = X.reshape(K * D, -1)
        topd = np.maximum(topd, (F / R[bins[j0:j1]]).max(1))
        p5[j0:j1], p4[j0:j1], lin[j0:j1] = (F > 1e-5).mean(0), (F > 1e-4).mean(0), F.mean(0)
        Lg = np.log10(np.maximum(X, 1e-300)); rh[j0:j1], es[j0:j1] = split_rhat(Lg), ess_many(Lg)
    idx = [(k, t) for k in range(K) for t in np.linspace(0, D - 1, 40).astype(int)]
    a5, a4, n90, sub, bt = [], [], [], [], []
    for k, t in idx:
        x = np.asarray(S[k, t], dtype=np.float64)
        a5.append((x > 1e-5).sum()); a4.append((x > 1e-4).sum())
        xs = np.sort(x)[::-1]; n90.append(int(np.searchsorted(np.cumsum(xs) / xs.sum(), 0.9) + 1))
        sub.append(inc_s @ x); bt.append(inc_b @ x)
    rmean = G @ lin - d
    return dict(chi=chi, tot=tot, p5=p5, p4=p4, lin=lin, rh=rh, es=es, a5=np.mean(a5), a4=np.mean(a4), n90=np.mean(n90),
                top=float(topd.max()), top_all=topd, sub=np.median(sub, 0), bt=np.median(bt, 0), chi_mean_model=float(rmean @ rmean), K=K, D=D)

runs = [r for r in args.runs if os.path.exists(r + '_samples.npy')]
for r in args.runs:
    if r not in runs:
        print(f'skipping {r}: no stored draws ({r}_samples.npy); compare_priors.py needs them')
labels = {'gibbs_lb12': 'old default: 1e-12 to 1e-2', 'gibbs_lb10': 'old box, floor 1e-10', 'gibbs_bb10': 'per-bin: R_j down 10 decades',
          'gibbs_bb12': 'per-bin: R_j down 12 decades', 'gibbs_th1': 'per-bin depths, theta = 1',
          'gibbs_th0.5': 'per-bin depths, theta = 0.5', 'gibbs_th2': 'per-bin depths, theta = 2',
          'gibbs_th1s': 'theta = 1, soft top', 'gibbs_bb10s': 'per-bin 10 decades, soft top',
          'gibbs_bb10h0': '10 decades, hard lower edge', 'gibbs_bbf12': 'common floor 1e-12',
          'gibbs_ss': 'spike-and-slab, sparse start', 'gibbs_ss_dense': 'spike-and-slab, dense start',
          'gibbs_ss8': 'spike-and-slab, slab floor 1e-8', 'gibbs_ss12': 'spike-and-slab, slab floor 1e-12'}
St = {r: stats(r) for r in runs}
g = np.random.default_rng(0)
print(f'{"run":32s} {"chi2":>12s} {"chi2(mean model)":>16s} {"total":>7s} {"bins/target":>12s} {">1e-5":>6s} {">1e-4":>6s} {"n90":>5s} {"max m/R_j":>9s} {"worst Rhat":>10s} {"ESS med/5%":>11s}')
for r in runs:
    s = St[r]; base = float(np.nanmax(split_rhat(g.standard_normal((s['K'], s['D'], n)))))
    print(f'{labels.get(r, r):32s} {s["chi"].mean():6.0f} ± {s["chi"].std():3.0f} {s["chi_mean_model"]:16.0f} {s["tot"].mean():7.4f} {np.median(s["bt"] / R):12.3f} '
          f'{s["a5"]:6.0f} {s["a4"]:6.0f} {s["n90"]:5.0f} {s["top"]:9.3f} {np.nanmax(s["rh"]):6.3f}/{base:.3f} {np.median(s["es"]):5.0f}/{np.percentile(s["es"], 5):4.0f}')

def shift(a, b, lab):
    A, B = St[a], St[b]
    sel = A['p5'] > 0.02
    d5 = np.log10((B['p5'][sel] + 1e-4) / (A['p5'][sel] + 1e-4))
    s4 = A['p4'] > 0.02
    d4 = np.log10((B['p4'][s4] + 1e-4) / (A['p4'][s4] + 1e-4))
    both = (A['sub'] > 0) & (B['sub'] > 0)
    rs = B['sub'][both] / A['sub'][both]
    act = A['p5'] > 0.02
    dl = np.log10(B['lin'][act] / A['lin'][act])
    print(f'{lab:46s} subsection totals x{np.median(rs):.3f} ({np.percentile(rs, 5):.3f}-{np.percentile(rs, 95):.3f}); '
          f'log10 P(>1e-5) {np.median(d5):+.2f} ({np.percentile(d5, 5):+.2f} to {np.percentile(d5, 95):+.2f}); '
          f'log10 P(>1e-4) {np.median(d4):+.2f}; linear means {np.median(dl):+.2f} decades')
print()
pairs_ = [('gibbs_lb12', 'gibbs_lb10', 'old box: floor 1e-12 -> 1e-10 (2 decades)'), ('gibbs_bb10', 'gibbs_bb12', 'per-bin box: 10 -> 12 decades deep'),
          ('gibbs_lb12', 'gibbs_bb10', 'old default -> per-bin box, 10 decades'), ('gibbs_bb10', 'gibbs_th1', 'per-bin 10 decades -> theta = 1'),
          ('gibbs_th1', 'gibbs_th2', 'theta 1 -> 2 (depths halved)'), ('gibbs_th1', 'gibbs_th0.5', 'theta 1 -> 0.5 (depths doubled)'),
          ('gibbs_th1', 'gibbs_th1s', 'theta = 1: hard top -> soft top'), ('gibbs_bb10', 'gibbs_bb10s', '10 decades: hard top -> soft top'),
          ('gibbs_bb10s', 'gibbs_bb10h0', '10 decades: soft lower edge -> hard'), ('gibbs_bb10h0', 'gibbs_bbf12', '10 decades below R_j -> floor 1e-12'),
          ('gibbs_ss', 'gibbs_ss_dense', 'spike-and-slab: sparse start vs dense start'), ('gibbs_bbf12', 'gibbs_ss', 'binbox floor 1e-12 -> spike-and-slab'),
          ('gibbs_th1s', 'gibbs_ss', 'theta = 1 -> spike-and-slab'),
          ('gibbs_ss', 'gibbs_ss8', 'spike-and-slab: slab floor 1e-10 -> 1e-8'), ('gibbs_ss', 'gibbs_ss12', 'spike-and-slab: slab floor 1e-10 -> 1e-12')]
for a_, b_, lab in pairs_:
    if a_ in St and b_ in St:
        shift(a_, b_, lab)
print()
for r in runs:
    t = St[r]['top_all']
    print(f'{labels.get(r, r):32s} top contact: draws with a rupture above 0.9 R_j {100 * np.mean(t > 0.9):5.1f}%, above R_j {100 * np.mean(t > 1.0):5.1f}%; largest m/R_j {t.max():.3f}')
print()
sig = 1 / d[brow]
for r in runs:
    o = St[r]['bt'] / R - 1
    print(f'{labels.get(r, r):32s} bin totals vs targets: median {100 * np.median(o):+5.1f}%, range {100 * o.min():+5.1f}% to {100 * o.max():+5.1f}%; '
          f'in units of each bin uncertainty: median {np.median(o / sig):+.2f}')

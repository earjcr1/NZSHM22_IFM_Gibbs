"""MFD uncertainty settings compared (same prior, the first two replicates of each): fit, total rate and its posterior
spread, the bins against their targets (posterior medians) and their posterior spread, the posterior MFD's slope (median
and 16-84% range: a draw with an empty bin has no meaningful slope), how often a draw leaves a bin nearly empty, the
population, and the products (subsection totals, exceedance probabilities) with replicate noise for scale.
    python tools/compare_mfd.py RUN_DIR_A:PRIOR:LABEL RUN_DIR_B:PRIOR:LABEL

RUN_DIR is a notebook RUN_DIR (cru_clean.npz at its top, runs in PRIOR/main_rep0 and main_rep1, made without stored
draws or with). The scripts are found in CRU_CODE, or in the folder above this one."""
import os
import sys
import numpy as np
sys.path.insert(0, os.environ.get('CRU_CODE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))   # the scripts' folder
from cru_common import load_problem, load_stream, magnitude_bins


def load(spec):
    run_dir, prior, label = spec.split(':')
    P = load_problem(f'{run_dir}/cru_clean.npz')
    G = P['G'].tocsr()
    S = P['block_names'].index('S')
    inc_s = (G[np.flatnonzero(P['block'] == S)] > 0).astype(float)
    rows, R, orig, lo = magnitude_bins(P, 'MFD')
    runs = [f'{run_dir}/{prior}/main_rep{i}' for i in range(2)]
    out = dict(label=label, P=P, R=R, lo=lo, rows=rows, inc_s=inc_s, runs=runs)
    # per-run stats
    reps = []
    for r in runs:
        st = load_stream(r)
        z = np.load(r + '.npz')
        w = int(z['burn']) // int(z['thin'])
        nd = int(st['chains']) * int(st['n_keep'])
        # bin totals per draw, in magnitude order: the stream's tot_b follows the problem's bin-row order
        brows_problem = np.flatnonzero(P['block'] == P['block_names'].index('MFD'))
        order = np.array([list(brows_problem).index(j) for j in rows])
        tb = st['tot_b'].reshape(-1, len(brows_problem))[:, order].astype(np.float64)
        reps.append(dict(m=st['msum'] / nd, p5=st['exc5'] / nd, p4=st['exc4'] / nd, nd=nd, tb=tb,
                         chi2=z['chi2'][:, w:].ravel(), n5=st['n5'].ravel(), n4=st['n4'].ravel(), n90=st['n90'].ravel()))
    out['reps'] = reps
    pool = lambda k: np.concatenate([q[k] for q in reps])
    out['tb'] = pool('tb')
    out['chi2'] = pool('chi2')
    nd = sum(q['nd'] for q in reps)
    out['m'] = sum(q['m'] * q['nd'] for q in reps) / nd
    out['p5'] = sum(q['p5'] * q['nd'] for q in reps) / nd
    out['p4'] = sum(q['p4'] * q['nd'] for q in reps) / nd
    out['nd'] = nd
    out['n5'], out['n4'], out['n90'] = pool('n5'), pool('n4'), pool('n90')
    return out


def slope(tb, lo):
    x = lo + 0.05
    X = np.c_[np.ones_like(x), x]
    coef = np.linalg.lstsq(X, np.log10(np.maximum(tb, 1e-30)).T, rcond=None)[0]
    return -coef[1]                                       # b-value of each draw's MFD


def describe(D):
    tot = D['tb'].sum(1)
    rel = D['tb'] / D['R']
    b = slope(D['tb'], D['lo'])                           # meaningless for a draw with an empty bin, hence quantiles
    b_t = slope(D['R'][None, :], D['lo'])[0]
    qb = np.percentile(b, [16, 50, 84])
    mb = np.median(D['tb'], 0) / D['R']
    sdb = D['tb'].std(0) / D['tb'].mean(0)
    low = rel < 0.01                                      # a bin below 1% of its target in a draw
    often = [j for j in np.argsort(-low.mean(0), kind='stable')[:3] if low[:, j].any()]
    c = D['P']['mfd_scale']
    relu = 1 / D['P']['d'][D['rows']]
    print(f"== {D['label']} (MFD {D['P']['mfd_uncertainty']!r}, x{c:.3f}; bins' relative uncertainties {100*relu.min():.1f}-{100*relu.max():.1f}%)")
    print(f"   chi2 {D['chi2'].mean():.1f} +- {D['chi2'].std():.1f}; total rate {tot.mean():.5f} +- {tot.std():.5f} "
          f"({100 * tot.std() / tot.mean():.1f}%; targets {D['R'].sum():.5f}, ratio {tot.mean() / D['R'].sum():.4f})")
    print(f"   bins' posterior medians / targets: median {np.median(mb):.4f} ({mb.min():.4f} to {mb.max():.4f}); bins' posterior "
          f"sd / mean: median {100*np.median(sdb):.1f}% ({100*sdb.min():.1f}% to {100*sdb.max():.1f}%), against their "
          f"uncertainties {100*np.median(relu):.1f}% (sd/uncertainty median {np.median(sdb / relu):.2f})")
    print(f"   posterior MFD slope (b of a straight line through each draw's bin totals): median {qb[1]:.3f} "
          f"(16-84%: {qb[0]:.3f} to {qb[2]:.3f}); the targets' {b_t:.3f}")
    print(f"   draws with some bin below 1% of its target: {100 * low.any(1).mean():.2f}% (a bin at exactly 0: "
          f"{100 * (rel == 0).any(1).mean():.2f}%)" + ('; most often ' + ', '.join(
              f"{D['lo'][j]:.2f}-{D['lo'][j] + 0.1:.2f} ({100 * low[:, j].mean():.2f}%)" for j in often) if often else ''))
    print(f"   ruptures > 1e-5 / > 1e-4 per draw {D['n5'].mean():.0f} / {D['n4'].mean():.0f}; carrying 90%: {np.median(D['n90']):.0f}")


def compare(a, b, label):
    sa, sb = a['inc_s'] @ a['m'], b['inc_s'] @ b['m']
    ok = (sa > 1e-9) & (sb > 1e-9)
    r = sb[ok] / sa[ok]
    q = np.percentile(r, [5, 50, 95])
    out = [f'{label}: subsection totals x{q[1]:.3f} (5-95%: {q[0]:.3f} to {q[2]:.3f}); total rate x{b["m"].sum() / a["m"].sum():.4f}']
    for k, thr in (('p5', 1e-5), ('p4', 1e-4)):
        sel = (a[k] > 0.02) | (b[k] > 0.02)
        lr = np.log10(np.maximum(b[k][sel], 1 / b['nd'])) - np.log10(np.maximum(a[k][sel], 1 / a['nd']))
        qq = np.percentile(lr, [5, 50, 95])
        out.append(f'   log10 P(rate > {thr:g}) over {sel.sum()} ruptures with P > 0.02 in either: median {qq[1]:+.3f} '
                   f'(5-95%: {qq[0]:+.3f} to {qq[2]:+.3f})')
    return '\n'.join(out)


A, B = load(sys.argv[1]), load(sys.argv[2])
for D in (A, B):
    describe(D)
rep = lambda D, i: dict(inc_s=D['inc_s'], m=D['reps'][i]['m'], p5=D['reps'][i]['p5'], p4=D['reps'][i]['p4'], nd=D['reps'][i]['nd'])
print(compare(A, B, f"{B['label']} vs {A['label']} (pooled replicates)"))
print(compare(rep(A, 0), rep(A, 1), f"noise: {A['label']} rep1 vs rep0"))
print(compare(rep(B, 0), rep(B, 1), f"noise: {B['label']} rep1 vs rep0"))

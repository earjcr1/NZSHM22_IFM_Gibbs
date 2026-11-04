#!/usr/bin/env python3
"""cfm_sliprates.py -- the slip-rate rows rebuilt from the NZ Community Fault Model (preliminaries.py --cfm-sliprates).

    python cfm_sliprates.py CFM_CSV RUPTURE_SET --A cru_A.txt --d cru_D.txt      (the rebuild, as a report)

The NZSHM22 inputs (cru_A, cru_D) give each subsection's slip rate as a Gaussian: its row is the rupture set's average
slips over sigma (m/yr) and its target 0.8 SlipRate / sigma, with sigma = 2 SD and SD half the CFM's min-to-max range
(the inversion's 0.8 scaling the rates; REPORT.md, "The slip rates from the CFM"). The CFM's own numbers
(NZ_CFM_v1_0.csv: SR_pref, SR_min, SR_max in mm/yr, one line per parent fault) are lopsided, and that is lost.

With --cfm-sliprates each slip-rate row of the inputs is rebuilt from its parent fault's CFM line (by name), as a split
normal in the slip rate itself (mm/yr, linear, as measured in the field):
  - its mode is f SR_pref, f the inputs' own factor on that row (0.8: f = target x sigma / SlipRate, checked);
  - below the mode its sd is sigma_lo = FACTOR f (SR_pref - SR_min) / NSIG, above it sigma_hi = FACTOR f (SR_max - SR_pref) / NSIG:
    SR_min and SR_max read as NSIG sigma either side (default 2) and the result multiplied by FACTOR (default 2, the
    inputs' doubling), so by default each side is its full width, as the inputs' sigma = 2 SD is for the whole range;
  - a side of zero width takes the other side's; a fault with SR_min = SR_pref = SR_max, its sd = the mode itself
    (as the inputs do for an SD of 0);
  - the row is rescaled to sigma_ref = (sigma_lo + sigma_hi) / 2: coefficients average slip / sigma_ref, target mode /
    sigma_ref, so a residual r below the target is weighed (sigma_ref / sigma_lo)^2 r^2 and one above
    (sigma_ref / sigma_hi)^2 r^2 (the sampler's split misfit, gibbs.py), and the summed slip rates (--slip-sum), which
    take each subsection's target and sigma from its row, sum the modes with sigma_ref.
Every other row is as the inputs give it."""
import argparse
import csv
import json
import os

import numpy as np
import scipy.sparse as sp

NSIG_DEFAULT, FACTOR_DEFAULT = 2.0, 2.0


def load_cfm(path):
    """The CFM's slip rates by fault name: name -> (SR_pref, SR_min, SR_max), mm/yr."""
    out = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            nm = row['Name'].strip()
            try:
                v = tuple(float(row[k]) for k in ('SR_pref', 'SR_min', 'SR_max'))
            except (KeyError, ValueError):
                raise ValueError(f'{path}: fault {nm!r} has no numeric SR_pref, SR_min and SR_max')
            if nm in out and out[nm] != v:
                raise ValueError(f'{path}: fault {nm!r} appears twice with different slip rates')
            out[nm] = v
    return out


def load_subsections(rupture_dir):
    """The rupture set's subsections in FaultID order: parent name, SlipRate and SlipRateStdDev (mm/yr)."""
    F = json.load(open(os.path.join(rupture_dir, 'fault_sections.geojson')))['features']
    pr = [f['properties'] for f in F]
    if [int(p['FaultID']) for p in pr] != list(range(len(pr))):
        raise ValueError(f'{rupture_dir}/fault_sections.geojson: FaultIDs are not 0, 1, 2, ... in order')
    return dict(parent=[str(p['ParentName']).strip() for p in pr], rate=np.array([float(p['SlipRate']) for p in pr]),
                sd=np.array([float(p['SlipRateStdDev']) for p in pr]))


def rebuild(A, d, rupture_dir, cfm_csv, nsig=NSIG_DEFAULT, factor=FACTOR_DEFAULT):
    """The inputs' slip-rate rows (raw rows 0 .. subsections - 1) rebuilt from the CFM. Returns (A, d with those rows
    rebuilt, L2 and H2 per raw row: the split misfit's weights below and above each target, 1 on every other row, a
    per-subsection table, report lines)."""
    from plate_rate import rupture_matrix
    if not (nsig > 0 and factor > 0):
        raise ValueError('--cfm-sigmas and --cfm-factor must be positive')
    sub = load_subsections(rupture_dir)
    ns = len(sub['parent'])
    cfm = load_cfm(cfm_csv)
    miss = sorted({p for p in sub['parent'] if p not in cfm})
    if miss:
        raise ValueError(f'{len(miss)} parent faults of the rupture set are not in {cfm_csv} by name: '
                         + ', '.join(miss[:10]) + (' ...' if len(miss) > 10 else ''))
    S = rupture_matrix(rupture_dir, ns)                                  # subsections x ruptures, average slip (m)
    if S.shape[1] != A.shape[1]:
        raise ValueError(f'the inputs have {A.shape[1]} ruptures and the rupture set {S.shape[1]}')
    Ar = sp.csr_matrix(A)
    Ar.sort_indices()
    S.sort_indices()
    sig_old = np.full(ns, np.nan)                                        # each row's sigma, m/yr
    for j in range(ns):
        ai, av = Ar.indices[Ar.indptr[j]:Ar.indptr[j + 1]], Ar.data[Ar.indptr[j]:Ar.indptr[j + 1]]
        si, sv = S.indices[S.indptr[j]:S.indptr[j + 1]], S.data[S.indptr[j]:S.indptr[j + 1]]
        if not np.array_equal(ai, si):
            raise ValueError(f'input row {j}: its ruptures are not subsection {j}\'s in the rupture set (indices.csv); '
                             'the slip-rate rows must be the subsections in FaultID order')
        if len(ai):
            q = sv / av
            if np.ptp(q) > 1e-6 * np.abs(q).max():
                raise ValueError(f'input row {j}: its coefficients are not in proportion to the average slips')
            sig_old[j] = q.mean()
    pref, smin, smax = (np.array([cfm[p][t] for p in sub['parent']]) for t in range(3))
    if (smin > pref).any() or (smax < pref).any() or (smin < 0).any():
        bad = sorted({sub['parent'][j] for j in np.flatnonzero((smin > pref) | (smax < pref) | (smin < 0))})
        raise ValueError(f'{cfm_csv}: SR_min <= SR_pref <= SR_max (and SR_min >= 0) fails for ' + ', '.join(bad))
    has = np.isfinite(sig_old) & (sub['rate'] > 0)
    f = np.full(ns, np.nan)
    f[has] = d[:ns][has] * sig_old[has] / (sub['rate'][has] * 1e-3)      # the inputs' factor on each row (0.8)
    fr = np.round(f[has], 9)
    f_ref = float(np.median(f[has]))
    if np.ptp(fr) > 1e-6 * abs(f_ref):
        raise ValueError(f'the inputs\' factor on the slip rates (target x sigma / SlipRate) varies from row to row '
                         f'({np.nanmin(f):.6g} to {np.nanmax(f):.6g})')
    f = np.full(ns, f_ref)                                               # one factor for every row
    mode = f * pref * 1e-3                                               # m/yr
    lo_w, hi_w = factor * f * (pref - smin) * 1e-3 / nsig, factor * f * (smax - pref) * 1e-3 / nsig
    both0, lo0, hi0 = (lo_w == 0) & (hi_w == 0), (lo_w == 0) & (hi_w > 0), (hi_w == 0) & (lo_w > 0)
    s_lo = np.where(lo0, hi_w, np.where(both0, mode, lo_w))
    s_hi = np.where(hi0, lo_w, np.where(both0, mode, hi_w))
    if (s_lo <= 0).any() or (s_hi <= 0).any():
        bad = sorted({sub['parent'][j] for j in np.flatnonzero((s_lo <= 0) | (s_hi <= 0))})
        raise ValueError(f'{cfm_csv}: a slip rate of 0 with no range gives no uncertainty: ' + ', '.join(bad))
    s_ref = 0.5 * (s_lo + s_hi)
    # the rows rescaled: coefficients average slip / sigma_ref, target mode / sigma_ref
    scale = np.ones(A.shape[0])
    scale[:ns] = np.where(np.isfinite(sig_old), sig_old / s_ref, 1.0)
    A2 = sp.csc_matrix(sp.diags(scale) @ Ar)
    d2 = np.array(d, np.float64)
    d2[:ns] = mode / s_ref
    L2, H2 = np.ones(A.shape[0]), np.ones(A.shape[0])
    L2[:ns], H2[:ns] = (s_ref / s_lo) ** 2, (s_ref / s_hi) ** 2
    tab = dict(parent=sub['parent'], pref=pref, smin=smin, smax=smax, mode=mode, s_lo=s_lo, s_hi=s_hi, s_ref=s_ref,
               sig_old=sig_old, rate_old=sub['rate'], f=f, d_old=np.array(d[:ns], float))
    return A2, d2, L2, H2, tab, report_lines(tab, nsig, factor, cfm_csv)


def report_lines(tab, nsig, factor, cfm_csv):
    par = tab['parent']
    first = np.unique(par, return_index=True)[1]                          # one line per parent fault
    pd_ = np.round(tab['pref'][first] - tab['rate_old'][first], 6)
    lo_w, hi_w = tab['pref'] - tab['smin'], tab['smax'] - tab['pref']
    z1 = int(np.sum(((lo_w == 0) ^ (hi_w == 0))[first])); z2 = int(np.sum(((lo_w == 0) & (hi_w == 0))[first]))
    ok = np.isfinite(tab['sig_old'])
    ratio = tab['s_ref'][ok] / tab['sig_old'][ok]
    skew = np.log10(tab['s_hi'] / tab['s_lo'])[first]
    L = [f'== slip rates from the CFM ({os.path.basename(cfm_csv)}): {len(tab["parent"])} subsection rows of {len(first)} parent '
         f'faults, each a split normal in mm/yr: mode {tab["f"][0]:g} x SR_pref (the inputs\' factor), sd below '
         f'{factor:g} x {tab["f"][0]:g} x (SR_pref - SR_min) / {nsig:g}, above {factor:g} x {tab["f"][0]:g} x (SR_max - SR_pref) / {nsig:g} '
         f'(SR_min and SR_max at {nsig:g} sigma, times {factor:g}); rows rescaled to the mean of the two sides',
         f'   SR_pref against the rupture set\'s SlipRate: {int(np.sum(pd_ == 0))} faults the same, {int(np.sum(pd_ != 0))} '
         f'different (largest {np.abs(pd_).max():.3g} mm/yr)',
         f'   lopsided: sd above / below the mode, median {10 ** np.median(skew):.2f} (5-95% {10 ** np.percentile(skew, 5):.2f}-'
         f'{10 ** np.percentile(skew, 95):.2f}); {int(np.sum(np.abs(skew) > np.log10(1.5)))} faults beyond a factor 1.5 either way',
         f'   {z1} faults with one side of zero width (it takes the other side\'s), {z2} with none (sd = the mode)',
         f'   the new sigma_ref against the inputs\' sigma: median {np.median(ratio):.3f} (5-95% {np.percentile(ratio, 5):.3f}-'
         f'{np.percentile(ratio, 95):.3f})']
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cfm', help='the CFM as CSV (NZ_CFM_v1_0.csv: Name, SR_pref, SR_min, SR_max)')
    ap.add_argument('ruptures', help='the OpenSHA rupture set (its ruptures/ folder, or a solution folder holding one)')
    ap.add_argument('--A', default='cru_A.txt')
    ap.add_argument('--d', default='cru_D.txt')
    ap.add_argument('--cfm-sigmas', type=float, default=NSIG_DEFAULT)
    ap.add_argument('--cfm-factor', type=float, default=FACTOR_DEFAULT)
    a = ap.parse_args()
    from cru_common import find_rupture_set, load_colt_sparse, load_vector
    rd = find_rupture_set(a.ruptures)
    A, d = load_colt_sparse(a.A), load_vector(a.d)
    *_, L = rebuild(A, d, rd, a.cfm, a.cfm_sigmas, a.cfm_factor)
    print('\n'.join(L))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""cfm_sliprates.py -- the slip-rate rows rebuilt from the NZ Community Fault Model (preliminaries.py --cfm-sliprates).

    python cfm_sliprates.py CFM_CSV RUPTURE_SET --A cru_A.txt --d cru_D.txt      (the rebuild, as a report)

The NZSHM22 inputs (cru_A, cru_D) give each subsection's slip rate as a Gaussian: its row is the rupture set's average
slips over sigma (m/yr) and its target 0.8 SlipRate / sigma, with sigma = 2 SD and SD half the CFM's min-to-max range
(the inversion's 0.8 scaling the rates; REPORT.md, "The slip rates from the CFM"). The CFM's own numbers
(NZ_CFM_v1_0.csv: SR_pref, SR_min, SR_max in mm/yr, one line per parent fault) are lopsided, and that is lost.

With --cfm-sliprates each slip-rate row of the inputs is rebuilt from its parent fault's CFM line (by name), as a split
normal in the slip rate itself (mm/yr, linear, as measured in the field):
  - its mode is f SR_pref (or, with --cfm-pref median, its median is, the mode moved accordingly; a fault whose mode
    would then fall at or below 0 keeps SR_pref as the mode), f the slip-rate factor, the assumed fraction of a fault's slip rate the ruptures take up
    (--slip-rate-factor; by default the inputs' own, 0.8, f = target x sigma / SlipRate, checked on every row);
  - below the mode its sd is sigma_lo = FACTOR f (SR_pref - SR_min) / NSIG, above it sigma_hi = FACTOR f (SR_max - SR_pref) / NSIG:
    SR_min and SR_max read as NSIG sigma either side (default 2, the reading agreed with colleagues on 2026-11-05; the
    inputs read them as 1) and the result multiplied by FACTOR (--slip-sd-multiplier, default 2, the inputs' doubling),
    so by default each side's sd is f times its width (after the doubling, SR_min and SR_max at 1 sd). The factor f, the
    assumed fraction of a fault's slip rate the ruptures take up, scales the whole distribution, sds included, as a
    fixed factor on an uncertain rate must (the inputs scale only their targets, so their sigma is 1/f too wide);
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

NSIG_DEFAULT, FACTOR_DEFAULT = 2.0, 2.0       # SR_min and SR_max at 2 sd, doubled (the inputs read them at 1)


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


def load_cfm_steps(path):
    """Each fault's rounding step (mm/yr): the finest of its SR_pref, SR_min and SR_max as written, 10^-(decimal places)
    of the shortest form of each (neither the CSV nor the spreadsheet keeps trailing zeros, so a '2' beside 1.5 and 2.8
    is taken as given to 0.1, the fault's numbers being set at one precision)."""
    dec = lambda v: len(v.split('.')[1]) if '.' in v else 0
    out = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            out[row['Name'].strip()] = min(10.0 ** -dec(str(row[k]).strip()) for k in ('SR_pref', 'SR_min', 'SR_max'))
    return out


def template_factors(cfm_csv, rupture_dir, rho):
    """--cfm-template-rho: the faults whose slip rates share one template, their errors correlated. Faults of QualCode 2-5
    (rates from comparisons with nearby faults, geomorphology, geodesy, or none: Seebeck et al.'s quality codes) with an
    identical SR_pref, SR_min and SR_max as written form a group of n; with a correlation rho between them, each fault's
    rows are weighted 1 / sqrt(1 + (n - 1) rho), so at rho = 1 a group counts as one measurement (as a section's
    subsections do). Returns (the factor per raw slip-rate row, subsections in FaultID order; report lines)."""
    if not (0.0 < rho <= 1.0):
        raise ValueError(f'--cfm-template-rho {rho:g}: give a correlation above 0 and at most 1')
    sub = load_subsections(rupture_dir)
    used = set(sub['parent'])
    info = {}
    with open(cfm_csv, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            nm = row['Name'].strip()
            if nm in used:
                info[nm] = (tuple(str(row[k]).strip() for k in ('SR_pref', 'SR_min', 'SR_max')), str(row.get('QualCode', '')).strip())
    if any(info[p][1] == '' for p in used):
        raise ValueError(f'{cfm_csv}: --cfm-template-rho needs the QualCode column for every fault')
    groups = {}
    for p in sorted(used):
        t, q = info[p]
        if q in ('2', '3', '4', '5'):
            groups.setdefault(t, []).append(p)
    fac = {p: 1.0 for p in used}
    for t, ps in groups.items():
        if len(ps) > 1:
            for p in ps:
                fac[p] = 1.0 / np.sqrt(1.0 + (len(ps) - 1) * rho)
    per_row = np.array([fac[p] for p in sub['parent']])
    big = sorted(((len(ps), t) for t, ps in groups.items() if len(ps) > 1), reverse=True)
    n_in = sum(n for n, _ in big)
    L = [f'== slip-rate templates (--cfm-template-rho {rho:g}): {n_in} faults of QualCode 2-5 share their SR_pref / SR_min / '
         f'SR_max with another, in {len(big)} groups; each fault\'s rows weighted 1 / sqrt(1 + (n - 1) x {rho:g}) for a group of '
         f'n, so {"each group counts as one measurement" if rho == 1 else f"a group of n counts as n / (1 + (n - 1) x {rho:g}) measurements"}; '
         'largest: ' + ', '.join(f'{"/".join(t)} ({n})' for n, t in big[:5])]
    return per_row, L


def load_subsections(rupture_dir):
    """The rupture set's subsections in FaultID order: parent name, SlipRate and SlipRateStdDev (mm/yr)."""
    F = json.load(open(os.path.join(rupture_dir, 'fault_sections.geojson')))['features']
    pr = [f['properties'] for f in F]
    if [int(p['FaultID']) for p in pr] != list(range(len(pr))):
        raise ValueError(f'{rupture_dir}/fault_sections.geojson: FaultIDs are not 0, 1, 2, ... in order')
    return dict(parent=[str(p['ParentName']).strip() for p in pr], rate=np.array([float(p['SlipRate']) for p in pr]),
                sd=np.array([float(p['SlipRateStdDev']) for p in pr]))


def inputs_slip(A, d, rupture_dir):
    """The inputs' slip-rate rows (raw rows 0 .. subsections - 1) checked against the rupture set: dict(sub, the
    subsections; ns, their number; Ar, A as CSR; sig_old, each row's sigma in m/yr, from its coefficients against
    average_slips.csv (NaN for a row with no ruptures); f, the inputs' slip-rate factor, target x sigma / SlipRate,
    checked to be one number for every row: 0.8 in NZSHM22)."""
    from plate_rate import rupture_matrix
    sub = load_subsections(rupture_dir)
    ns = len(sub['parent'])
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
    has = np.isfinite(sig_old) & (sub['rate'] > 0)
    f = np.full(ns, np.nan)
    f[has] = d[:ns][has] * sig_old[has] / (sub['rate'][has] * 1e-3)      # the inputs' factor on each row (0.8)
    f_ref = float(np.median(f[has]))
    if np.ptp(np.round(f[has], 9)) > 1e-6 * abs(f_ref):
        raise ValueError(f'the inputs\' factor on the slip rates (target x sigma / SlipRate) varies from row to row '
                         f'({np.nanmin(f):.6g} to {np.nanmax(f):.6g})')
    # the inputs' multiplier on SlipRateStdDev (2: sigma = 2 SD), checked on every row with an SD; a row with SD 0 has
    # sigma = f x SlipRate instead (the rate itself, as targeted)
    sd0 = sub['sd'] == 0
    hm = np.isfinite(sig_old) & ~sd0
    mm = sig_old[hm] * 1e3 / sub['sd'][hm]
    m_ref = float(round(float(np.median(mm)), 9))                        # 2, to rounding
    if np.ptp(mm) > 1e-6 * m_ref:
        raise ValueError(f'the inputs\' multiplier on SlipRateStdDev (sigma / SD) varies from row to row ({mm.min():.6g} to '
                         f'{mm.max():.6g})')
    return dict(sub=sub, ns=ns, Ar=Ar, sig_old=sig_old, f=f_ref, m=m_ref, sd0=sd0)


def rescale_inputs(A, d, rupture_dir, srf=None, mult=None):
    """Without --cfm-sliprates: the inputs' slip-rate Gaussians with the slip-rate factor on the whole distribution.
    The inputs (NZSHM22) put their factor f (0.8) on the targets only, sigma = 2 SD in unscaled mm/yr; an implementation
    error upstream, in the NZSHM22 code, as the user reads it (the uncertainty scaling postdating the slip-rate factor).
    Here each becomes target srf x SlipRate, sigma mult x srf x SD (srf default f, mult default 2), and a row with an SD
    of 0, whose sigma is the rate itself (f x SlipRate), sigma srf x SlipRate. The inputs' own convention is mult = 2.5
    (2.5 x 0.8 = 2), which leaves them as they are. In units of sigma: coefficients / s, targets x (srf / f) / s, s each
    row's sigma new over old. Returns (A, d, report lines)."""
    I = inputs_slip(A, d, rupture_dir)
    ns = I['ns']
    f = float(round(I['f'], 9))                                         # 0.8, to rounding
    srf_ = f if srf is None else float(srf)
    mult_ = 2.0 if mult is None else float(mult)
    s = np.where(I['sd0'], srf_ / f, mult_ * srf_ / I['m'])            # each row's sd in mm/yr, new / old
    scale = np.ones(A.shape[0])
    scale[:ns] = 1.0 / s
    d2 = np.array(d, np.float64)
    d2[:ns] = d[:ns] * (srf_ / f) / s
    same = bool(np.all(s == 1.0) and srf_ == f)
    L = [f'== slip-rate Gaussians with the slip-rate factor on the whole distribution: target {srf_:g} x SlipRate, sd '
         f'{mult_:g} x {srf_:g} x SlipRateStdDev = {mult_ * srf_:g} x SD (the inputs: target {f:g} x SlipRate, sd {I["m"]:g} x SD, '
         'the factor on the targets only)' + ('; the inputs unchanged' if same else
         f'; each sd {np.median(s[~I["sd0"]]):g} times the inputs\' in mm/yr') + f'; the {int(I["sd0"].sum())} subsections with '
         f'an SD of 0 keep sd = the target']
    return sp.csc_matrix(sp.diags(scale) @ I['Ar']), d2, L


def rebuild(A, d, rupture_dir, cfm_csv, nsig=NSIG_DEFAULT, factor=FACTOR_DEFAULT, srf=None, pref_as='mode', rounding=False):
    """The inputs' slip-rate rows (raw rows 0 .. subsections - 1) rebuilt from the CFM, each split normal scaled as a whole
    by srf (default: the inputs' own factor, 0.8). Returns (A, d with those rows rebuilt, L2 and H2 per raw row: the
    split misfit's weights below and above each target, 1 on every other row, a per-subsection table, report lines)."""
    if not (nsig > 0 and factor > 0):
        raise ValueError('--cfm-sigmas and --slip-sd-multiplier must be positive')
    if srf is not None and not srf > 0:
        raise ValueError('--slip-rate-factor must be positive')
    I = inputs_slip(A, d, rupture_dir)
    sub, ns, Ar, sig_old = I['sub'], I['ns'], I['Ar'], I['sig_old']
    cfm = load_cfm(cfm_csv)
    miss = sorted({p for p in sub['parent'] if p not in cfm})
    if miss:
        raise ValueError(f'{len(miss)} parent faults of the rupture set are not in {cfm_csv} by name: '
                         + ', '.join(miss[:10]) + (' ...' if len(miss) > 10 else ''))
    pref, smin, smax = (np.array([cfm[p][t] for p in sub['parent']]) for t in range(3))
    if (smin > pref).any() or (smax < pref).any() or (smin < 0).any():
        bad = sorted({sub['parent'][j] for j in np.flatnonzero((smin > pref) | (smax < pref) | (smin < 0))})
        raise ValueError(f'{cfm_csv}: SR_min <= SR_pref <= SR_max (and SR_min >= 0) fails for ' + ', '.join(bad))
    f = np.full(ns, I['f'] if srf is None else float(srf))                # the slip-rate factor, on the whole distribution
    mode = f * pref * 1e-3                                               # m/yr
    lo_w, hi_w = factor * f * (pref - smin) * 1e-3 / nsig, factor * f * (smax - pref) * 1e-3 / nsig   # f on everything
    both0, lo0, hi0 = (lo_w == 0) & (hi_w == 0), (lo_w == 0) & (hi_w > 0), (hi_w == 0) & (lo_w > 0)
    s_lo = np.where(lo0, hi_w, np.where(both0, mode, lo_w))
    s_hi = np.where(hi0, lo_w, np.where(both0, mode, hi_w))
    if (s_lo <= 0).any() or (s_hi <= 0).any():
        bad = sorted({sub['parent'][j] for j in np.flatnonzero((s_lo <= 0) | (s_hi <= 0))})
        raise ValueError(f'{cfm_csv}: a slip rate of 0 with no range gives no uncertainty: ' + ', '.join(bad))
    widen = np.ones(ns)
    qstep = np.zeros(ns)
    if rounding:                      # --cfm-rounding: each number known to half its fault's step q (uniform, variance
        steps = load_cfm_steps(cfm_csv)   # q^2/12): the mode's, f^2 q^2/12, on both sides; each width's, a difference of two
        qstep = np.array([steps[p] for p in sub['parent']]) * 1e-3     # rounded numbers, c^2 2 q^2/12 (c each side's sd
        cc = factor * f / nsig                                          # per unit of width); added in quadrature
        add = qstep ** 2 * (f ** 2 + 2.0 * cc ** 2) / 12.0
        w_lo, w_hi = np.sqrt(s_lo ** 2 + add), np.sqrt(s_hi ** 2 + add)
        widen = np.maximum(w_lo / s_lo, w_hi / s_hi)
        s_lo, s_hi = w_lo, w_hi
    s_ref = 0.5 * (s_lo + s_hi)
    fallback = np.zeros(ns, bool)
    if pref_as == 'median':           # --cfm-pref median: SR_pref the median, each side's sd as above, the mode moved
        from scipy.stats import norm
        a = s_lo / (s_lo + s_hi)      # the split normal's mass below its mode
        with np.errstate(invalid='ignore', divide='ignore'):
            up = s_hi * norm.ppf(0.5 + (0.5 - a) / (2.0 * (1.0 - a)))      # median - mode, the long side above
            dn = s_lo * norm.ppf(1.0 - 1.0 / (4.0 * a))                     # mode - median, the long side below
        shift = np.where(a < 0.5, up, np.where(a > 0.5, -dn, 0.0))         # median - mode
        mode_m = mode - shift
        fallback = mode_m <= 0        # a mode at or below 0: these faults keep SR_pref as the mode
        mode = np.where(fallback, mode, mode_m)
    elif pref_as != 'mode':
        raise ValueError(f'--cfm-pref {pref_as}: use mode or median')
    # the rows rescaled: coefficients average slip / sigma_ref, target mode / sigma_ref
    scale = np.ones(A.shape[0])
    scale[:ns] = np.where(np.isfinite(sig_old), sig_old / s_ref, 1.0)
    A2 = sp.csc_matrix(sp.diags(scale) @ Ar)
    d2 = np.array(d, np.float64)
    d2[:ns] = mode / s_ref
    L2, H2 = np.ones(A.shape[0]), np.ones(A.shape[0])
    L2[:ns], H2[:ns] = (s_ref / s_lo) ** 2, (s_ref / s_hi) ** 2
    tab = dict(parent=sub['parent'], pref=pref, smin=smin, smax=smax, mode=mode, s_lo=s_lo, s_hi=s_hi, s_ref=s_ref,
               sig_old=sig_old, rate_old=sub['rate'], f=f, f_in=I['f'], d_old=np.array(d[:ns], float), pref_as=pref_as,
               fallback=fallback, rounding=rounding, widen=widen, qstep=qstep)
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
         f'faults, each a split normal in mm/yr, the CFM\'s scaled as a whole by the slip-rate factor {tab["f"][0]:g} (the '
         f'inputs\' {tab["f_in"]:g}): mode '
         f'{tab["f"][0]:g} x SR_pref, sd below {factor:g} x {tab["f"][0]:g} x (SR_pref - SR_min) / {nsig:g}, above {factor:g} x '
         f'{tab["f"][0]:g} x (SR_max - SR_pref) / {nsig:g} (SR_min and SR_max at {nsig:g} sigma, times {factor:g}); rows '
         f'rescaled to the mean of the two sides',
         f'   SR_pref against the rupture set\'s SlipRate: {int(np.sum(pd_ == 0))} faults the same, {int(np.sum(pd_ != 0))} '
         f'different (largest {np.abs(pd_).max():.3g} mm/yr)',
         f'   lopsided: sd above / below the mode, median {10 ** np.median(skew):.2f} (5-95% {10 ** np.percentile(skew, 5):.2f}-'
         f'{10 ** np.percentile(skew, 95):.2f}); {int(np.sum(np.abs(skew) > np.log10(1.5)))} faults beyond a factor 1.5 either way',
         f'   {z1} faults with one side of zero width (it takes the other side\'s), {z2} with none (sd = the mode)',
         f'   the new sigma_ref against the inputs\' sigma: median {np.median(ratio):.3f} (5-95% {np.percentile(ratio, 5):.3f}-'
         f'{np.percentile(ratio, 95):.3f})']
    if tab['rounding']:
        wf = tab['widen'][first]
        L.append(f'   rounding (--cfm-rounding): each fault\'s numbers known to half its finest step (from SR_pref, SR_min and '
                 f'SR_max as written; {int(np.sum(tab["qstep"][first] >= 1e-3))} faults at 1 mm/yr), each side\'s sd widened in '
                 f'quadrature: by a median factor {np.median(wf):.3f}, beyond 1.1 for {int(np.sum(wf > 1.1))} faults, largest '
                 f'{wf.max():.2f} ({par[first[int(np.argmax(wf))]]})')
    if tab['pref_as'] == 'median':
        tgt = tab['f'][first] * tab['pref'][first] * 1e-3
        mv = (tab['mode'][first] - tgt) / tab['s_ref'][first]
        fb = sorted({par[j] for j in np.flatnonzero(tab['fallback'])})
        L.append(f'   SR_pref as the median (--cfm-pref median): each split normal\'s mode moved so that its median is '
                 f'{tab["f"][0]:g} x SR_pref, by a median {np.median(np.abs(mv)):.3f} of its sigma_ref (5-95% '
                 f'{np.percentile(np.abs(mv), 5):.3f}-{np.percentile(np.abs(mv), 95):.3f}; {int(np.sum(np.abs(mv) > 0.25))} faults '
                 f'beyond a quarter); {len(fb)} faults, whose mode would fall at or below 0, keep SR_pref as the mode'
                 + (': ' + ', '.join(fb) if fb else ''))
    else:
        L.append('   SR_pref as the mode (--cfm-pref mode)')
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cfm', help='the CFM as CSV (NZ_CFM_v1_0.csv: Name, SR_pref, SR_min, SR_max)')
    ap.add_argument('ruptures', help='the OpenSHA rupture set (its ruptures/ folder, or a solution folder holding one)')
    ap.add_argument('--A', default='cru_A.txt')
    ap.add_argument('--d', default='cru_D.txt')
    ap.add_argument('--cfm-sigmas', type=float, default=NSIG_DEFAULT)
    ap.add_argument('--slip-sd-multiplier', type=float, default=FACTOR_DEFAULT)
    ap.add_argument('--cfm-pref', default='mode', choices=['mode', 'median'])
    ap.add_argument('--cfm-rounding', action='store_true')
    ap.add_argument('--slip-rate-factor', type=float, default=None, help="default: the inputs' own (0.8)")
    a = ap.parse_args()
    from cru_common import find_rupture_set, load_colt_sparse, load_vector
    rd = find_rupture_set(a.ruptures)
    A, d = load_colt_sparse(a.A), load_vector(a.d)
    *_, L = rebuild(A, d, rd, a.cfm, a.cfm_sigmas, a.slip_sd_multiplier, a.slip_rate_factor, a.cfm_pref, a.cfm_rounding)
    print('\n'.join(L))


if __name__ == '__main__':
    main()

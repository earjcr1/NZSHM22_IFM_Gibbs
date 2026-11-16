#!/usr/bin/env python3
"""MFD_GR: the MFD bins' targets rebuilt from the regional rate of M>=5 earthquakes (N5) and the b-value, as NZSHM22's
OpenSHA code builds cru_D's (nzshm-opensha: NZSHM22_CrustalInversionTargetMFDs.RegionalTargetMFDs.init and
NZSHM22_FaultSystemRupSetCalc), so that other N5 and b can be tried (the user's request, 2026-11-17).

The partitioning, on 40 bins of 0.1 from M5.05 (bin centres):
  1. total GR: 10^(-b M) at the bin centres from 5.05 to the bin nearest the rupture set's largest magnitude (whole
     bins), scaled so that all bins sum to N5;
  2. each section's share pdf_s of the spatial seismicity PDF inside its polygon; f_on their sum;
  3. sub-seismogenic on-fault MFD: for each section a GR of the same b with rate pdf_s x N5 at M>=5, up to the bin
     below its minimum supra-seismogenic magnitude (mod_sect_min_mags.csv), at least through the bin below the
     global minimum magnitude (6.8001: the 6.75 bin); summed over the sections;
  4. truly off-fault MFD, tri-linear: below the kink (the moment-weighted mean of the sections' minimum magnitudes, as
     a bin), a GR of rate (N5 - f_on N5) x 1.01; from the kink to the bin after M8.05 - 0.05, the on-fault rate
     interpolated in log10 from the on-fault GR (x 0.98) to the total, off = total - on (0 at and above that last bin);
  5. supra-seismogenic target (cru_D's bins): total - off - sub, 1e-20 below the minimum magnitude and above the
     maximum, and where 0.
NZSHM22's uncertainties: sigma_j = scalar x R_ref^power x R_j^(1 - power), R_ref the target in the bin nearest M7.0
(scalar 0.1, power 0.25: 10% at M7.0-7.1).

Everything that depends on the rupture set is read from an OpenSHA inversion solution's ruptures/ folder: pdf_s from
sub_seismo_on_fault_mfds.csv (each row sums to pdf_s x the run's N5), the minimum magnitudes from mod_sect_min_mags.csv,
the largest magnitude from properties.csv, the kink from where inversion_target_mfds.json's off-fault MFD leaves its
constant share of the total, the minimum and maximum magnitudes and the run's N5 and b from
regional_inversion_target_mfds.json. load_parts checks the rebuild against the stored MFDs at the run's own N5 and b
(19k, 211k and 630k: every component to 1e-14 or better).

Each component is proportional to N5, so a bin's target is N5 x s_j(b). With MFD_GR_SPREAD 'knobs' each bin's sigma is
the sd of N5 s_j(b) over N5 ~ Normal(N5, SD_N5) above 0 and b ~ Normal(B, SD_B), independent (exact: the truncated
normal's moments, Gauss-Hermite over b), around the target at the central N5 and b; the bins are scored as independent,
which the user reads as standing in for the GR assumption; with 'nzshm22', NZSHM22's sigma recipe on the new targets.

    python mfd_gr.py RUPTURE_SET [--mfd-gr N5,SD_N5,B,SD_B] [--spread knobs|nzshm22]
"""
import argparse
import csv
import json
import math
import os

import numpy as np

NZ_MIN_MAG, NZ_NUM_BINS, DELTA = 5.05, 40, 0.1
X = NZ_MIN_MAG + DELTA * np.arange(NZ_NUM_BINS)          # the bins' centres
M_MAX_OFF_FAULT = 8.05                                  # NZSHM22's off-fault maximum (less half a bin in the code)
ON_CORR, OFF_CORR = 0.98, 1.01                          # its hand-set rate corrections in the off-fault MFD
FIRST_WEIGHT_POWER_MAG = 7.0                            # its sigma recipe's reference magnitude
FILES = ('regional_inversion_target_mfds.json', 'inversion_target_mfds.json', 'sub_seismo_on_fault_mfds.csv',
         'mod_sect_min_mags.csv', 'properties.csv')


def ci(m):
    """EvenlyDiscretizedFunc.getClosestXIndex: the bin nearest m (ties up, after OpenSHA's 1 + 1e-14 nudge)."""
    i = int(math.floor((1 + 1e-14) * (m - NZ_MIN_MAG) / DELTA + 0.5))
    return min(max(i, 0), NZ_NUM_BINS - 1)


def gr(n5, b, i_hi):
    """GutenbergRichterMagFreqDist from 5.05 to bin i_hi: 10^(-b M) at the bin centres, summing to n5."""
    y = np.where(np.arange(NZ_NUM_BINS) <= i_hi, 10.0 ** (-b * X), 0.0)
    return y * (n5 / y.sum())


def parse_spec(spec):
    """--mfd-gr N5,SD_N5,B,SD_B -> (n5, sd_n5, b, sd_b), checked."""
    try:
        v = [float(x) for x in str(spec).split(',')]
    except ValueError:
        v = []
    if len(v) != 4 or not (v[0] > 0 and v[1] >= 0 and v[2] > 0 and v[3] >= 0):
        raise ValueError(f'--mfd-gr {spec}: give N5,SD_N5,B,SD_B (the rate of M>=5 per year and its sd, the b-value and '
                         'its sd), e.g. 3.92,1,0.959,0.1')
    return tuple(v)


def solution_dir(rupture_dir):
    """The folder holding an inversion solution's MFD files: rupture_dir itself or its ruptures/."""
    for d in (rupture_dir, os.path.join(rupture_dir, 'ruptures')):
        if all(os.path.exists(os.path.join(d, f)) for f in FILES):
            return d
    miss = [f for f in FILES if not os.path.exists(os.path.join(rupture_dir, f))]
    raise FileNotFoundError(f'{rupture_dir}: MFD_GR needs an OpenSHA inversion solution\'s rupture set, with '
                            + ', '.join(miss) + ' (a rupture set made without an inversion has none of them)')


def components(parts, n5, b):
    """The total, truly off-fault, sub-seismogenic and supra-seismogenic MFDs (40 bins each) for N5 and b."""
    tot = gr(n5, b, parts['i_top'])
    g_hi = ci(parts['minMag']) - 1
    sub = np.zeros(NZ_NUM_BINS)
    for p_, m_ in zip(parts['pdf'], parts['minm']):
        sub += gr(p_ * n5, b, max(ci(m_) - 1, g_hi))
    on5 = n5 * parts['f_on']
    k, i_off = parts['kink'], ci(M_MAX_OFF_FAULT - DELTA / 2)
    onGR = gr(on5 * ON_CORR, b, parts['i_top'])
    off = np.zeros(NZ_NUM_BINS)
    off[:k] = gr((n5 - on5) * OFF_CORR, b, parts['i_top'])[:k]
    for i in range(k, min(i_off + 2, NZ_NUM_BINS)):
        w = (i - k) / ((i_off + 1) - k)
        if tot[i] > 0:
            on = 10.0 ** ((1 - w) * math.log10(onGR[i]) + w * math.log10(tot[i]))
            off[i] = max(tot[i] - on, 0.0)
    supra = tot - off - sub
    supra = np.where((X < parts['minMag']) | (X > parts['maxMag']) | (supra == 0), 1e-20, supra)
    return dict(total=tot, off=off, sub=sub, supra=supra)


def load_parts(rupture_dir):
    """What the partitioning takes from a solution (see the module's notes), checked by rebuilding its stored MFDs at
    its own N5 and b. Raises ValueError if they are not reproduced."""
    d = solution_dir(rupture_dir)
    reg = json.load(open(os.path.join(d, 'regional_inversion_target_mfds.json')))
    if reg.get('TVZ', {}).get('totalTargetGR') is not None:
        raise ValueError(f'{d}: the TVZ region has its own MFDs; MFD_GR handles the sans-TVZ region only')
    R = reg['sansTVZ']
    rows = list(csv.reader(open(os.path.join(d, 'sub_seismo_on_fault_mfds.csv'))))[1:]
    sub_st = np.array([[float(v) for v in r[1:]] for r in rows])
    pdf = sub_st.sum(1) / R['totalRateM5']
    minm = np.array([float(r[1]) for r in list(csv.reader(open(os.path.join(d, 'mod_sect_min_mags.csv'))))[1:]])
    if len(minm) != len(pdf):
        raise ValueError(f'{d}: {len(pdf)} sections in sub_seismo_on_fault_mfds.csv, {len(minm)} in mod_sect_min_mags.csv')
    with open(os.path.join(d, 'properties.csv')) as f:
        head = f.readline().strip().split(',')
        mags = np.loadtxt(f, delimiter=',', usecols=(head.index('Magnitude'),), ndmin=1)
    J = json.load(open(os.path.join(d, 'inversion_target_mfds.json')))
    st = {k: np.array([p[1] for p in J[j]['values']]) for k, j in (('total', 'totalRegionalMFD'), ('off', 'trulyOffFaultMFD'),
                                                                   ('sub', 'onFaultSubSeisMFD'), ('supra', 'onFaultSupraSeisMFD'))}
    pos = st['total'] > 0
    rr = np.where(pos, st['off'] / np.where(pos, st['total'], 1.0), np.nan)
    leave = np.flatnonzero(pos & (np.abs(rr - rr[0]) > 1e-9 * rr[0]))
    if not leave.size:
        raise ValueError(f'{d}: the stored off-fault MFD never leaves its share of the total; no kink to read')
    parts = dict(dir=d, pdf=pdf, f_on=float(pdf.sum()), minm=minm, mmax=float(mags.max()), minMag=float(R['minMag']),
                 maxMag=float(R['maxMag']), n5_run=float(R['totalRateM5']), b_run=float(R['bValue']),
                 power=float(R.get('uncertaintyPower', 0.25)), scalar=float(R.get('uncertaintyScalar', 0.1)),
                 kink=int(leave[0]), stored=st)
    parts['i_top'] = ci(min(parts['mmax'], parts['maxMag']))
    c = components(parts, parts['n5_run'], parts['b_run'])
    err = {k: float(np.max(np.abs(c[k] - st[k])[st[k] > 1e-19] / st[k][st[k] > 1e-19])) for k in st}
    if max(err.values()) > 1e-9:
        raise ValueError(f'{d}: the partitioning does not reproduce the stored MFDs at the run\'s N5 {parts["n5_run"]:g} and '
                         f'b {parts["b_run"]:g} (largest relative differences ' + ', '.join(f'{k} {v:.1e}' for k, v in err.items())
                         + '); this solution\'s code may partition otherwise')
    parts['check'] = err
    return parts


def nzshm22_sigma(R, parts):
    """NZSHM22's sigma recipe (MFDManipulation.addMfdUncertainty) on targets R (40 bins; 1e-20 where R is)."""
    ref = R[ci(FIRST_WEIGHT_POWER_MAG)]
    return np.where(R > 1e-20, parts['scalar'] * ref ** parts['power'] * R ** (1 - parts['power']), 1e-20)


def targets(parts, spec, spread='knobs', n_gh=48):
    """The bins' targets and sigmas (40 bins) for --mfd-gr spec: dict(R, sigma, s_neg (bins where some b node in the
    quadrature gives a negative target), n5, sd_n5, b, sd_b, spread, comp (the components at the central values))."""
    n5, sn, b, sb = parse_spec(spec)
    comp = components(parts, n5, b)
    R = comp['supra']
    if spread == 'nzshm22':
        sig = nzshm22_sigma(R, parts)
        neg = np.zeros(NZ_NUM_BINS, bool)
    elif spread == 'knobs':
        if sn == 0 and sb == 0:
            raise ValueError("MFD_GR_SPREAD 'knobs' with both sds 0 gives the bins no uncertainty: give an sd, or 'nzshm22'")
        from scipy.stats import truncnorm
        if sn > 0:
            tn = truncnorm(-n5 / sn, np.inf, loc=n5, scale=sn)
            e1, e2 = float(tn.mean()), float(tn.var() + tn.mean() ** 2)
        else:
            e1, e2 = n5, n5 * n5
        if sb > 0:
            z, w = np.polynomial.hermite_e.hermegauss(n_gh)
            w = w / w.sum()
            S = np.array([components(parts, 1.0, b + sb * z_)['supra'] for z_ in z])   # s_j(b) per node
            S = np.where(S > 1e-19, S, np.where(S < 0, S, 0.0))
            m1, m2 = w @ S, w @ S ** 2
            neg = (S < 0).any(0)
        else:
            s0 = np.where(R > 1e-19, R / n5, 0.0)
            m1, m2, neg = s0, s0 ** 2, np.zeros(NZ_NUM_BINS, bool)
        var = np.maximum(e2 * m2 - (e1 * m1) ** 2, 0.0)
        sig = np.where(R > 1e-19, np.sqrt(var), 1e-20)
    else:
        raise ValueError(f"MFD_GR_SPREAD {spread!r}: use 'knobs' or 'nzshm22'")
    return dict(R=R, sigma=sig, s_neg=neg, n5=n5, sd_n5=sn, b=b, sd_b=sb, spread=spread, comp=comp)


def report_lines(parts, T, lo=None):
    """A report of the reconstruction and the targets (bins from lo up, default the minimum magnitude's)."""
    d = parts['dir']
    L = [f'MFD_GR (mfd_gr.py): the supra-seismogenic MFD rebuilt from N5 = {T["n5"]:g} per year (M>=5) and b = {T["b"]:g}, '
         f'as NZSHM22\'s OpenSHA code partitions it, from {d}',
         f'   the solution\'s own run (N5 {parts["n5_run"]:g}, b {parts["b_run"]:g}) reproduced: '
         + ', '.join(f'{k} {v:.0e}' for k, v in parts['check'].items()) + ' (largest relative differences)',
         f'   the rupture set: largest magnitude {parts["mmax"]:.3f} (top bin {X[parts["i_top"]]:.2f}); {len(parts["pdf"])} '
         f'sections, the seismicity PDF\'s share on the faults {parts["f_on"]:.4f}; minimum supra-seismogenic magnitudes '
         f'{parts["minm"].min():.2f}-{parts["minm"].max():.2f}; the off-fault MFD\'s kink at the {X[parts["kink"]]:.2f} bin',
         ('   sigma per bin: the sd of N5 s_j(b) over N5 ~ Normal(' + f'{T["n5"]:g}, {T["sd_n5"]:g}) above 0 and b ~ Normal('
          f'{T["b"]:g}, {T["sd_b"]:g}) (MFD_GR_SPREAD \'knobs\'), each bin independent' if T['spread'] == 'knobs' else
          f'   sigma per bin: NZSHM22\'s recipe, {parts["scalar"]:g} x R_ref^{parts["power"]:g} x R_j^{1 - parts["power"]:g}, '
          f'R_ref at M{FIRST_WEIGHT_POWER_MAG:g} (MFD_GR_SPREAD \'nzshm22\')'),
         f'   {"bin":>5s} {"total":>10s} {"off-fault":>10s} {"sub-seis":>10s} {"target":>10s} {"sigma":>10s} {"sd/target":>9s} '
         f'{"stored":>10s} {"new/stored":>10s}']
    st = parts['stored']['supra']
    i0 = ci(parts['minMag'] if lo is None else lo + DELTA / 2)
    for i in range(i0, NZ_NUM_BINS):
        if T['R'][i] <= 1e-19:
            continue
        c = T['comp']
        L.append(f'   {X[i]:5.2f} {c["total"][i]:10.4e} {c["off"][i]:10.4e} {c["sub"][i]:10.4e} {T["R"][i]:10.4e} '
                 f'{T["sigma"][i]:10.4e} {T["sigma"][i] / T["R"][i]:9.3f} {st[i]:10.4e} {T["R"][i] / st[i]:10.4f}'
                 + ('  (negative at some b in the quadrature)' if T['s_neg'][i] else ''))
    sel = T['R'] > 1e-19
    L.append(f'   total of the targets {T["R"][sel].sum():.5g} per year (the solution\'s {st[sel].sum():.5g})')
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rupture_set', help='an OpenSHA inversion solution\'s folder or its ruptures/')
    ap.add_argument('--mfd-gr', default=None, help='N5,SD_N5,B,SD_B (default: the solution\'s own N5 and b, sd 1 and 0.1)')
    ap.add_argument('--spread', default='knobs', choices=['knobs', 'nzshm22'])
    a = ap.parse_args()
    parts = load_parts(a.rupture_set)
    spec = a.mfd_gr or f'{parts["n5_run"]:g},1,{parts["b_run"]:g},0.1'
    print('\n'.join(report_lines(parts, targets(parts, spec, a.spread))))


if __name__ == '__main__':
    main()

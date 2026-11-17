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

Each component is proportional to N5, so a bin's target is N5 x f_j(b). The knobs are NZSHM22's (Rollins et al., 2024,
BSSA 114, 150-181, Table 1, upper-plate IFM; the user's choice, 2026-11-17, replacing independent N5 and b triples):
b's 10th percentile, mean and 90th percentile (0.823, 0.959, 1.089), read as a split normal; the mean N(M>=5)
conditioned on each (2.65, 3.40, 4.57 per year), so N and b are correlated, N(b) a quadratic in log10 N through the three
pairs; and the N scaling s, a factor on the whole MFD independent of b, its 2-sigma values (0.66, 1.41) about 1, read as
a split normal in ln s (parse_knobs); with RECONSIDER_MAGUNCERT, a further factor m on N for the catalogue's magnitude
uncertainties (MAGUNCERT: 1, 3.92/3.40, 1 + 2 (3.92/3.40 - 1), its -2 sigma, mean and +2 sigma, a normal in m), ln(s m)'s
density the convolution of the two. A bin's target is s m N(b) f_j(b). Each bin's PDF is propagated exactly (bin_pdfs:
ln s's density summed over a fine grid of b, in u = ln of the target) and described for the sampler by the paleo sites'
sinh-arcsinh curve family in log rate (paleo_misfit, mode 'log'; fit_bin_curves): its minimum at the mode of the density
of log rate, through the exact 2.275/15.865/84.135/97.725% points. Each bin is then scored by -2 ln of its density of log
rate, from its minimum (gibbs.py --mfd-misfit pdf), as a paleo site by its curve; the bin's row keeps a target, the
target at b's mean, s = 1 and m's mean, and a sigma, the propagated sd, for its units; the prior's soft top takes each bin's upper
spread, its 84.135% point over its target (q84, saved with the problem as mfd_pdf_q84).

    python mfd_gr.py SOLUTION [--mfd-gr B10,B,B90,N10,N,N90,S_LO,S_HI[,M_LO,M,M_HI]]
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


LN10 = math.log(10.0)
PQ = np.array([0.02275, 0.15865, 0.5, 0.84135, 0.97725])   # the bins' quantiles: 2 and 1 sd either side and the median
FIXED_KERNEL = 0.01                                     # a knob given without spread: a normal kernel of this sd in ln N5


Z10 = 1.2815515655446004                                # the 90th percentile of a standard normal
_MAGRATIO = 3.92 / 3.40                                 # N(M>=5) with half of the magnitudes' downward skew over NZSHM22's 3.40
MAGUNCERT = (1.0, _MAGRATIO, 1.0 + 2.0 * (_MAGRATIO - 1.0))   # RECONSIDER_MAGUNCERT's factor on N: -2 sigma, mean, +2 sigma (the
                                                        # user's, 2026-11-17): magnitude uncertainties downward-skewed (Rhoades,
                                                        # 1996; 3.40), half that skew (3.92), or none (about twice the rise); to be
                                                        # revised when the catalogue is reanalysed


def parse_knobs(spec):
    """MFD_GR's knobs, 'B10,B,B90,N10,N,N90,S_LO,S_HI[,M_LO,M,M_HI]' (the NZSHM22 logic tree's form; Rollins et al., 2024,
    Table 1) -> dict(b=(b10, mean, b90), n=(n10, n, n90), s=(lo, hi), m=(lo, mean, hi) or None):
      b: the b-value's 10th percentile, mean and 90th percentile (a split normal, its mode the mean, each side's sd its
         distance to the 10th or 90th percentile over 1.2816);
      n: the mean rate of M>=5 earthquakes conditioned on each of those b-values: N(b), a quadratic in log10 N through the
         three pairs (b and N correlated);
      s: the N scaling, a factor on the whole MFD independent of b, its 2.275% and 97.725% (2 sigma) values about 1 (a split
         normal in ln s, its mode at 1, each side's sd half the log distance to its value);
      m: RECONSIDER_MAGUNCERT's factor on every N (MAGUNCERT, 1.0, 3.92/3.40, 1 + 2(3.92/3.40 - 1): N with magnitude uncertainties
         downward-skewed, half that skew, or Gaussian), its -2 sigma, mean and +2 sigma values, a split normal in m (a
         normal for a symmetric triple), independent of b and s.
    A knob without spread (b10 = b = b90; S_LO = S_HI = 1) is held at its middle value."""
    try:
        v = [float(x) for x in str(spec).split(',')]
    except ValueError:
        v = []
    if len(v) not in (8, 11):
        raise ValueError(f'MFD_GR {spec}: give B10,B,B90,N10,N,N90,S_LO,S_HI[,M_LO,M,M_HI] (b\'s 10th percentile, mean and 90th '
                         'percentile, the mean N(M>=5) at each, the N scaling\'s 2-sigma values, and optionally the magnitude-'
                         'uncertainty factor\'s -2 sigma, mean and +2 sigma), e.g. 0.823,0.959,1.089,2.65,3.40,4.57,0.66,1.41,1,1.15294,1.30588')
    b, n, sc = tuple(v[:3]), tuple(v[3:6]), tuple(v[6:8])
    m = tuple(v[8:]) if len(v) == 11 else None
    if m is not None and not (0 < m[0] <= m[1] <= m[2]):
        raise ValueError(f'MFD_GR {spec}: the magnitude-uncertainty factor\'s values must rise and be above 0')
    if not (0 < b[0] <= b[1] <= b[2]) or (b[0] == b[1]) != (b[1] == b[2]):
        raise ValueError(f'MFD_GR {spec}: b\'s values must rise (B10 < B < B90), or all three be equal (b fixed)')
    if not all(x > 0 for x in n) or (b[0] == b[2] and len(set(n)) > 1):
        raise ValueError(f'MFD_GR {spec}: the N values must be above 0 (and equal when b is fixed)')
    if not (0 < sc[0] <= 1 <= sc[1]):
        raise ValueError(f'MFD_GR {spec}: the N scaling\'s values must bracket 1 (0 < S_LO <= 1 <= S_HI)')
    if b[0] == b[2] and sc[0] == sc[1]:
        raise ValueError(f'MFD_GR {spec}: neither b nor the N scaling has spread, so the bins have no distribution')
    return dict(b=b, n=n, s=sc, m=m)


def n_of_b(knobs, b):
    """The mean N(M>=5) conditioned on b: log10 N a quadratic in b through the three (b, N) pairs (a constant with b fixed)."""
    (b0, b1, b2), (n0, n1, n2) = knobs['b'], knobs['n']
    b = np.asarray(b, float)
    if b0 == b2:
        return np.full(b.shape, n1)
    c = np.polyfit([b0, b1, b2], np.log10([n0, n1, n2]), 2)
    return 10.0 ** np.polyval(c, b)


def _split_logpdf(x, lo, mode, hi):
    """ln of a split normal's density (mode, each side's sd mode - lo and hi - mode); a side without spread falls to the
    other side's sd."""
    s_lo, s_hi = mode - lo, hi - mode
    s_lo, s_hi = (s_lo or s_hi), (s_hi or s_lo)
    s = np.where(x < mode, s_lo, s_hi)
    return -0.5 * ((x - mode) / s) ** 2 - np.log(0.5 * (s_lo + s_hi) * math.sqrt(2 * math.pi))


def bin_pdfs(parts, knobs, n_coarse=161, n_b=1201, n_u=6001):
    """Each bin's distribution of its target s N(b) f_j(b) (f_j the supra-seismogenic MFD per unit N, NZSHM22's
    partitioning), propagated exactly from the knobs (parse_knobs): the density g_j(u) of u = ln(target) on a grid, a sum
    over a fine grid of b, weighted by b's density, of ln s's density at u - ln(N(b) f_j(b)); with its quantiles (PQ), the
    mode of the rate's own density (g(u) / e^u), mean, sd and the probability that the target is 0 or less. With m
    (RECONSIDER_MAGUNCERT) the factor is s m, ln(s m)'s density the convolution of the two. Returns dict(R: the target at
    b's mean, N(b) there, s = 1 and m's mean; bins: {j: dict(u, g, q, mode, mean, sd, p_neg)})."""
    bl, bm, bh = knobs['b']
    sl_, sh_ = knobs['s']
    if bl == bh:                                          # b fixed
        bf, w = np.array([bm]), np.ones(1)
        Sf = (components(parts, 1.0, bm)['supra'] * float(n_of_b(knobs, bm)))[None, :]
    else:
        sdl, sdh = (bm - bl) / Z10, (bh - bm) / Z10
        lo_b, hi_b = bm - 6.0 * sdl, bm + 6.0 * sdh
        bc = np.linspace(lo_b, hi_b, n_coarse)
        Sc = np.array([components(parts, 1.0, b_)['supra'] for b_ in bc]) * n_of_b(knobs, bc)[:, None]
        Sc = np.where(Sc > 1e-19, Sc, np.where(Sc < 0, Sc, 0.0))
        bf = np.linspace(lo_b, hi_b, n_b)
        w = np.exp(_split_logpdf(bf, bm - sdl, bm, bm + sdh))
        w /= w.sum()
        Sf = np.array([np.interp(bf, bc, Sc[:, j]) for j in range(NZ_NUM_BINS)]).T
    if sl_ == sh_:                                        # s fixed at 1: a narrow kernel in ln s
        lps = lambda v: -0.5 * (v / FIXED_KERNEL) ** 2 - math.log(FIXED_KERNEL * math.sqrt(2 * math.pi))
        v_lo, v_hi = -8 * FIXED_KERNEL, 8 * FIXED_KERNEL
    else:                                                 # ln s a split normal, mode 0, each side's sd half its 2-sigma distance
        s_lo, s_hi = -0.5 * math.log(sl_), 0.5 * math.log(sh_)
        s_lo, s_hi = (s_lo or s_hi), (s_hi or s_lo)
        lps = lambda v: _split_logpdf(v, -s_lo, 0.0, s_hi)
        v_lo, v_hi = -8 * s_lo, 8 * s_hi
    m_c = 1.0                                             # the magnitude-uncertainty factor's mean (1: none)
    if knobs.get('m') is None:
        lpv = lps
    else:                                                 # ln(s m): ln s's density convolved with ln m's, on a fine grid
        ml, m_c, mh = knobs['m']
        sm_lo, sm_hi = (m_c - ml) / 2.0, (mh - m_c) / 2.0
        sm_lo, sm_hi = (sm_lo or sm_hi or FIXED_KERNEL * m_c), (sm_hi or sm_lo or FIXED_KERNEL * m_c)
        w_lo, w_hi = math.log(max(m_c - 8 * sm_lo, 1e-3 * m_c)), math.log(m_c + 8 * sm_hi)
        dv = 2e-4
        vg = np.arange(v_lo + w_lo, v_hi + w_hi + dv, dv)
        wg = np.arange(w_lo, w_hi + dv, dv)
        pm = np.exp(_split_logpdf(np.exp(wg), m_c - sm_lo, m_c, m_c + sm_hi) + wg)   # ln m's density
        pm /= pm.sum() * dv
        sg = np.arange(v_lo, v_hi + dv, dv)
        conv = np.convolve(np.exp(lps(sg)), pm)[:len(vg)] * dv                       # ln(s m)'s density on vg
        lc = np.log(np.maximum(conv, 1e-300))
        lpv = lambda v: np.interp(v, vg, lc, left=-690.0, right=-690.0)
        v_lo, v_hi = vg[0], vg[-1]
    R = components(parts, float(n_of_b(knobs, bm)) * m_c, bm)['supra']
    out = {}
    for j in np.flatnonzero(R > 1e-19):
        sj = Sf[:, j]
        pos = sj > 0
        p_neg = float(w[~pos].sum())
        ls = np.log(sj[pos])
        u = np.linspace(ls.min() + v_lo, ls.max() + v_hi, n_u)
        G = np.zeros(n_u)
        for chunk in np.array_split(np.arange(pos.sum()), max(1, pos.sum() // 200)):
            G += np.exp(lpv(u[:, None] - ls[chunk][None, :])) @ w[pos][chunk]
        du = u[1] - u[0]
        mass = G.sum() * du
        G *= (1.0 - p_neg) / mass                         # the positive part's density in u; P(target <= 0) = p_neg
        cdf = p_neg + np.concatenate([[0.0], np.cumsum(0.5 * (G[1:] + G[:-1]) * du)])
        q = np.interp(PQ, cdf, u)
        with np.errstate(divide='ignore'):
            lr = np.log(G) - u                            # the rate's own density, ln
        k = int(np.argmax(lr))
        if 0 < k < n_u - 1:                               # the mode, refined by a parabola through its neighbours
            a_, b_, c_ = lr[k - 1], lr[k], lr[k + 1]
            um = u[k] + 0.5 * du * (a_ - c_) / (a_ - 2 * b_ + c_)
        else:
            um = u[k]
        x = np.exp(u)
        mean = float(np.sum(x * G) * du)
        sd = float(np.sqrt(max(np.sum(x * x * G) * du - mean ** 2, 0.0)))
        out[int(j)] = dict(u=u, g=G, q=q, mode=float(np.exp(um)), mean=mean, sd=sd, p_neg=p_neg)
    return dict(R=R, bins=out, knobs=knobs)


def fit_bin_curves(B):
    """Each bin's PDF as the paleo sites' curve family in log rate (paleo_misfit.fit_site, mode 'log': the density of
    u = ln rate; the user's choice, 2026-11-17), its minimum at the exact mode of that density and through the exact
    2.275/15.865/84.135/97.725% points, with how far its misfit (-2 ln of the density, from its minimum) departs from the
    exact one between the 0.1% and 99.9% points: {j: dict(th, best, err (quantile errors, decades), ok, dmis (largest
    |difference| in the misfit), mis_at (the exact misfit there))}."""
    from paleo_misfit import fit_site, site_misfit
    out = {}
    for j, b_ in B['bins'].items():
        u, G = b_['u'], b_['g']
        k = int(np.argmax(G))                             # the mode of the density of log rate, refined by a parabola
        um = u[k]
        if 0 < k < len(u) - 1:
            a_, b2, c_ = np.log(G[k - 1:k + 2])
            um = u[k] + 0.5 * (u[1] - u[0]) * (a_ - c_) / (a_ - 2 * b2 + c_)
        best = float(np.exp(um))
        th, err, ok = fit_site(best, np.exp(b_['q'][[0, 1, 3, 4]]), 'log')
        cdf = b_['p_neg'] + np.concatenate([[0.0], np.cumsum(0.5 * (G[1:] + G[:-1]) * (u[1] - u[0]))])
        sel = (cdf > 0.001) & (cdf < 0.999) & (G > 0)
        mis_ex = 2.0 * (-np.log(G[sel]) + np.log(np.interp(um, u, G)))
        mis_fit = site_misfit(th, best, u[sel], 'log')
        k = int(np.argmax(np.abs(mis_fit - mis_ex)))
        out[j] = dict(th=th, best=best, err=err, ok=ok, dmis=float(np.abs(mis_fit - mis_ex)[k]), mis_at=float(mis_ex[k]))
    return out




def gr_bins(parts, spec):
    """MFD_GR's bins for --mfd-gr spec: dict(R (the targets at b's mean and s = 1, 40 bins), sd (the propagated sds), q84
    (each bin's 84.135% point: the prior's soft top uses its upper spread), th and best (each bin's curve: sinh-arcsinh
    parameters in log rate, and its minimum's rate), B (bin_pdfs), F (fit_bin_curves), knobs, spec). Raises ValueError if a bin's curve cannot be fitted or the target can be 0 or less."""
    knobs = parse_knobs(spec)
    B = bin_pdfs(parts, knobs)
    F = fit_bin_curves(B)
    bad = [M_ for j, M_ in ((j, X[j]) for j in F) if not F[j]['ok'] or B['bins'][j]['p_neg'] > 1e-6]
    if bad:
        raise ValueError('MFD_GR: the bins at M' + ', '.join(f'{m:.2f}' for m in bad) + ' have no usable curve (a target that '
                         'can be 0 or less, or a fit that failed)')
    sd = np.full(NZ_NUM_BINS, np.nan)
    th = np.full((NZ_NUM_BINS, 4), np.nan)
    best = np.full(NZ_NUM_BINS, np.nan)
    q84 = np.full(NZ_NUM_BINS, np.nan)
    for j in F:
        sd[j], th[j], best[j] = B['bins'][j]['sd'], F[j]['th'], F[j]['best']
        q84[j] = math.exp(B['bins'][j]['q'][3])
    return dict(R=B['R'], sd=sd, q84=q84, th=th, best=best, B=B, F=F, knobs=knobs, spec=str(spec))


def knob_text(knobs):
    """The knobs, for reports."""
    b, n, sc = knobs['b'], knobs['n'], knobs['s']
    return (f'b {b[1]:g} (10th-90th percentile {b[0]:g}-{b[2]:g}, a split normal) with the mean N(M>=5) at each '
            f'{n[1]:g} ({n[0]:g}, {n[2]:g}; N(b) a quadratic in log10 N), and the N scaling, 2-sigma {sc[0]:g}-{sc[1]:g} '
            '(a split normal in ln s about 1)'
            + (f', times the magnitude-uncertainty factor m {knobs["m"][1]:g} (2-sigma {knobs["m"][0]:g}-{knobs["m"][2]:g}; '
               'RECONSIDER_MAGUNCERT)' if knobs.get('m') else ''))


def report_lines(parts, GB, lo=None):
    """A report of the reconstruction and each bin's PDF (bins from lo up, default the minimum magnitude's)."""
    st = parts['stored']['supra']
    L = [f'MFD_GR (mfd_gr.py): each bin\'s target PDF propagated from {knob_text(GB["knobs"])}, through NZSHM22\'s partitioning '
         f'of the total GR, from {parts["dir"]}',
         f'   the solution\'s own run (N5 {parts["n5_run"]:g}, b {parts["b_run"]:g}) reproduced: '
         + ', '.join(f'{k} {v:.0e}' for k, v in parts['check'].items()) + ' (largest relative differences)',
         f'   the rupture set: largest magnitude {parts["mmax"]:.3f} (top bin {X[parts["i_top"]]:.2f}); {len(parts["pdf"])} '
         f'sections, the seismicity PDF\'s share on the faults {parts["f_on"]:.4f}; minimum supra-seismogenic magnitudes '
         f'{parts["minm"].min():.2f}-{parts["minm"].max():.2f}; the off-fault MFD\'s kink at the {X[parts["kink"]]:.2f} bin',
         '   each bin scored by -2 ln of its PDF in log rate from its minimum (a sinh-arcsinh curve through the exact quantiles, its '
         'minimum at the exact mode; gibbs.py --mfd-misfit pdf); x target: relative to the target at b\'s mean, s = 1 and m\'s mean',
         f'   {"bin":>5s} {"target":>10s} {"cru_D":>10s} {"mode":>6s} {"2.3%":>6s} {"16%":>6s} {"50%":>6s} {"84%":>6s} {"97.7%":>6s} '
         f'{"sd":>6s} {"(x target)":>10s} {"fit: q err":>10s} {"misfit diff":>11s}']
    i0 = ci(parts['minMag'] if lo is None else lo + DELTA / 2)
    for j in range(i0, NZ_NUM_BINS):
        if j not in GB['F']:
            continue
        R, b_, f_ = GB['R'][j], GB['B']['bins'][j], GB['F'][j]
        q = np.exp(b_['q']) / R
        L.append(f'   {X[j]:5.2f} {R:10.4e} {st[j]:10.4e} {f_["best"] / R:6.3f} ' + ' '.join(f'{x:6.3f}' for x in q)
                 + f' {b_["sd"] / R:6.3f} {"":>10s} {np.abs(f_["err"]).max():10.4f} {f_["dmis"]:11.3f}')
    sel = np.array(sorted(GB['F']))
    L.append(f'   total of the targets {GB["R"][sel].sum():.5g} per year (the solution\'s {st[sel].sum():.5g}); fit: quantile errors '
             'in decades, and the largest |difference| between the curve\'s misfit and the exact one between the 0.1% and 99.9% points')
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rupture_set', help='an OpenSHA inversion solution\'s folder or its ruptures/')
    ap.add_argument('--mfd-gr', default='0.823,0.959,1.089,2.65,3.40,4.57,0.66,1.41',
                    help='B10,B,B90,N10,N,N90,S_LO,S_HI[,M_LO,M,M_HI] (default: NZSHM22\'s upper-plate IFM, Rollins et al. 2024, Table 1; '
                    'append the magnitude-uncertainty factor, e.g. ' + ','.join(f'{x_:g}' for x_ in MAGUNCERT) + ', for RECONSIDER_MAGUNCERT)')
    a = ap.parse_args()
    parts = load_parts(a.rupture_set)
    print('\n'.join(report_lines(parts, gr_bins(parts, a.mfd_gr))))


if __name__ == '__main__':
    main()

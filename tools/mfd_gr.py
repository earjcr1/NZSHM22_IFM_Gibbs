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

Each component is proportional to N5, so a bin's target is N5 x s_j(b). The knobs are triples, (low, mode, high) for N5
and for b (parse_triples): b a split normal; N5 a Gamma with that mode and sd half the triple's width (the user's choice,
2026-11-17; split normals in log10 or in rate as options). Each bin's PDF is propagated exactly (bin_pdfs: N5's density
summed over a fine grid of b, in u = ln of the target) and described for the sampler by the paleo sites' sinh-arcsinh
curve family in log rate (paleo_misfit, mode 'log'; fit_bin_curves): its minimum at the mode of the density of log rate,
through the exact 2.275/15.865/84.135/97.725% points (19k-630k: quantile errors about 0.001 decades, misfit within about
0.2 of the exact one to 3 sd). Each bin is then scored by -2 ln of its density of log rate, from its minimum (gibbs.py
--mfd-misfit pdf), as a paleo site by its curve; the bin's row keeps a target, the target at the knobs' modes, and a sigma,
the propagated sd, for its units and the prior's soft top.

    python mfd_gr.py SOLUTION [--mfd-gr N5_LO,N5,N5_HI,B_LO,B,B_HI[,gamma|log10|linear]]
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


def parse_triples(spec):
    """MFD_GR's knobs as triples, 'N5_LO,N5,N5_HI,B_LO,B,B_HI[,gamma|log10|linear]' -> dict(n5=(lo, mode, hi), b=(lo,
    mode, hi), n5_space). b: a split normal with its mode at the middle value and each side's sd its distance to the outer
    one. N5: 'gamma' (the default; the user's choice, 2026-11-17): a Gamma with its mode at the middle value and its sd half
    the triple's width ((N5_HI - N5_LO) / 2), as a rate estimated from Poisson or negative-binomial counts has (skewness
    2 / sqrt(k): 0.48 for 3.92 +- 1); 'log10' or 'linear': a split normal like b's, read in log10 (the triple's ratios) or
    in rate (kept above 0)."""
    parts_ = str(spec).split(',')
    space = 'gamma'
    if parts_ and parts_[-1].strip() in ('gamma', 'log10', 'linear'):
        space = parts_.pop().strip()
    try:
        v = [float(x) for x in parts_]
    except ValueError:
        v = []
    if len(v) != 6 or not (0 < v[0] <= v[1] <= v[2] and 0 < v[3] <= v[4] <= v[5]):
        raise ValueError(f'MFD_GR {spec}: give N5_LO,N5,N5_HI,B_LO,B,B_HI (low <= mode <= high, all above 0), e.g. '
                         '2.92,3.92,4.92,0.859,0.959,1.059, optionally followed by gamma (the default), log10 or linear for N5')
    if v[0] == v[2] and v[3] == v[5]:
        raise ValueError(f'MFD_GR {spec}: both knobs without spread give the bins no distribution')
    return dict(n5=tuple(v[:3]), b=tuple(v[3:]), n5_space=space)


def _split_logpdf(x, lo, mode, hi):
    """ln of a split normal's density (mode, each side's sd mode - lo and hi - mode); a side without spread falls to the
    other side's sd."""
    s_lo, s_hi = mode - lo, hi - mode
    s_lo, s_hi = (s_lo or s_hi), (s_hi or s_lo)
    s = np.where(x < mode, s_lo, s_hi)
    return -0.5 * ((x - mode) / s) ** 2 - np.log(0.5 * (s_lo + s_hi) * math.sqrt(2 * math.pi))


def bin_pdfs(parts, knobs, n_coarse=161, n_b=1201, n_u=6001):
    """Each bin's distribution of its target N5 s_j(b), propagated exactly from the knobs (parse_triples): the density
    g_j(u) of u = ln(target) on a grid (a sum over a fine grid of b, weighted by b's density, of N5's density at
    u - ln s_j(b)), with its quantiles (PQ), the mode of the rate's own density (g(u) / e^u), mean, sd and the
    probability that the target is 0 or less (s_j(b) <= 0 at some b). Returns dict(R: the target at the knobs' modes,
    bins: {j: dict(u, g, q, mode, mean, sd, p_neg)})."""
    (bl, bm, bh), (nl, nm, nh) = knobs['b'], knobs['n5']
    if bl == bh:                                          # b fixed
        bf, w = np.array([bm]), np.ones(1)
        Sf = components(parts, 1.0, bm)['supra'][None, :]
    else:
        lo_b, hi_b = bm - 6.0 * ((bm - bl) or (bh - bm)), bm + 6.0 * ((bh - bm) or (bm - bl))
        bc = np.linspace(lo_b, hi_b, n_coarse)
        Sc = np.array([components(parts, 1.0, b_)['supra'] for b_ in bc])
        Sc = np.where(Sc > 1e-19, Sc, np.where(Sc < 0, Sc, 0.0))
        bf = np.linspace(lo_b, hi_b, n_b)
        w = np.exp(_split_logpdf(bf, bl, bm, bh))
        w /= w.sum()
        Sf = np.array([np.interp(bf, bc, Sc[:, j]) for j in range(NZ_NUM_BINS)]).T
    if knobs['n5_space'] == 'gamma' and nl != nh:        # v = ln N5: N5 a Gamma, mode nm, sd (nh - nl) / 2
        from scipy.stats import gamma as gamma_
        sdn = 0.5 * (nh - nl)
        rk = (nm + math.sqrt(nm * nm + 4.0 * sdn * sdn)) / (2.0 * sdn)    # sqrt(k): (k - 1) theta = mode, sqrt(k) theta = sd
        kk, th_ = rk * rk, sdn / rk
        gd = gamma_(kk, scale=th_)
        lpv = lambda v: gd.logpdf(np.exp(v)) + v
        v_lo, v_hi = math.log(max(float(gd.ppf(1e-9)), 1e-12)), math.log(float(gd.ppf(1 - 1e-12)))
        knobs['n5_gamma'] = (kk, th_)
    elif knobs['n5_space'] in ('log10', 'gamma'):         # log10 N5 a split normal (or, N5 fixed, a narrow kernel)
        if nl == nh:
            lpv = lambda v: -0.5 * ((v - math.log(nm)) / FIXED_KERNEL) ** 2 - math.log(FIXED_KERNEL * math.sqrt(2 * math.pi))
            v_lo, v_hi = math.log(nm) - 8 * FIXED_KERNEL, math.log(nm) + 8 * FIXED_KERNEL
        else:
            L = tuple(math.log10(x) for x in (nl, nm, nh))
            lpv = lambda v: _split_logpdf(v / LN10, *L) - math.log(LN10)
            v_lo, v_hi = LN10 * (L[1] - 8 * ((L[1] - L[0]) or (L[2] - L[1]))), LN10 * (L[1] + 8 * ((L[2] - L[1]) or (L[1] - L[0])))
    else:                                                 # linear: N5 a split normal above 0
        if nl == nh:
            raise ValueError('MFD_GR: a fixed N5 needs the log10 reading')
        from scipy.stats import norm
        p0 = float(np.exp(_split_logpdf(np.array([0.0]), nl, nm, nh))[0])   # not needed exactly: renormalized below
        lpv = lambda v: _split_logpdf(np.exp(v), nl, nm, nh) + v
        v_lo, v_hi = math.log(max(nm - 8 * (nm - nl), nm * 1e-3)), math.log(nm + 8 * (nh - nm))
    R = components(parts, nm, bm)['supra']
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
    """MFD_GR's bins for --mfd-gr spec: dict(R (the targets at the knobs' modes, 40 bins), sd (the propagated sds), th and
    best (each bin's curve: sinh-arcsinh parameters in log rate, and its minimum's rate), B (bin_pdfs), F (fit_bin_curves),
    knobs, spec). Raises ValueError if a bin's curve cannot be fitted or the target can be 0 or less."""
    knobs = parse_triples(spec)
    B = bin_pdfs(parts, knobs)
    F = fit_bin_curves(B)
    bad = [M_ for j, M_ in ((j, X[j]) for j in F) if not F[j]['ok'] or B['bins'][j]['p_neg'] > 1e-6]
    if bad:
        raise ValueError('MFD_GR: the bins at M' + ', '.join(f'{m:.2f}' for m in bad) + ' have no usable curve (a target that '
                         'can be 0 or less, or a fit that failed)')
    sd = np.full(NZ_NUM_BINS, np.nan)
    th = np.full((NZ_NUM_BINS, 4), np.nan)
    best = np.full(NZ_NUM_BINS, np.nan)
    for j in F:
        sd[j], th[j], best[j] = B['bins'][j]['sd'], F[j]['th'], F[j]['best']
    return dict(R=B['R'], sd=sd, th=th, best=best, B=B, F=F, knobs=knobs, spec=str(spec))


def knob_text(knobs):
    """The knobs, for reports."""
    nl, nm, nh = knobs['n5']
    n5 = (f'N5 Gamma, mode {nm:g}, sd {0.5 * (nh - nl):g} (shape {knobs["n5_gamma"][0]:.2f})' if knobs['n5_space'] == 'gamma'
          and 'n5_gamma' in knobs else f'N5 split normal in {knobs["n5_space"]} ({nl:g}, {nm:g}, {nh:g})')
    return n5 + f'; b split normal ({knobs["b"][0]:g}, {knobs["b"][1]:g}, {knobs["b"][2]:g})'


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
         'minimum at the exact mode; gibbs.py --mfd-misfit pdf); x target: relative to the target at the knobs\' modes',
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
    ap.add_argument('--mfd-gr', default=None, help='N5_LO,N5,N5_HI,B_LO,B,B_HI[,gamma|log10|linear] (default: the solution\'s '
                    'own N5 and b, +-1 and +-0.1, N5 a Gamma)')
    a = ap.parse_args()
    parts = load_parts(a.rupture_set)
    spec = a.mfd_gr or (f'{parts["n5_run"] - 1:g},{parts["n5_run"]:g},{parts["n5_run"] + 1:g},'
                        f'{parts["b_run"] - 0.1:g},{parts["b_run"]:g},{parts["b_run"] + 0.1:g}')
    print('\n'.join(report_lines(parts, gr_bins(parts, spec))))


if __name__ == '__main__':
    main()

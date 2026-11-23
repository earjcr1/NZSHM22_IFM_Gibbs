#!/usr/bin/env python3
"""The inversion's inputs, G and d, built from an NZSHM22 OpenSHA rupture set alone (November 21st, 2026, the user's
request: with MFD_GR and the CFM's slip rates the workflow need not read cru_A and cru_D at all).

    from rupture_set_inputs import build_inputs
    A, d, info = build_inputs(RUPTURE_DIR)          # preliminaries.py --from-rupture-set RUPTURE_DIR

The rows take the form cru_A and cru_D give them (the relations checked on the 211k and 630k inputs; memory note
"cru input provenance"), so that cleaning (preliminaries.py) treats them exactly as it treats the inputs:
  - slip-rate rows 0 .. S-1, one per subsection (row = section index): coefficient each rupture's average slip
    (average_slips.csv) over sigma, target the rate over sigma, from sect_slip_rates.csv (the branch's slip-rate factor
    already applied), sigma = its SD (the rate itself where the SD is 0), with no factor 2 (the user, 2026-11-23: the
    19k and 211k inputs doubled every sigma, and the 630k inputs halved the SD-0 rows', both now obsolete; cleaning reads
    each row's sigma, so the conventions clean to the same problem);
  - one penalty row (entries 1e20, target 0) on the ruptures below their sections' minimum magnitude, compared by
    magnitude bin as OpenSHA does: a rupture is removed if its 0.1 bin (from 6.8) lies below the bin of the minimum
    magnitude of any of its sections, each section's the larger of MIN_MAG (6.8) and the smallest magnitude among the
    ruptures confined to its parent fault. That is NZSHM22's rule: its mod_sect_min_mags.csv is reproduced to 1e-13 at
    630,000 ruptures (with 6.8001, against aliasing at the bin edge; the user: 6.8 here, 2026-11-23), so the file is not
    read; and the bins matter (at 211,000 ruptures an M7.203 rupture is kept though one of its sections' minimum is
    7.223, the same bin). It removes exactly the ruptures the inputs' penalty and zero-forcing rows remove (405 at
    19,000, 6,504 at 211,000, 14,425 at 630,000). The row also ends the slip-rate rows, as in the inputs;
  - the magnitude bins, 0.1 wide from --lowest-bin (6.8) up to the largest magnitude: coefficient 1/sigma_j on each
    rupture in the bin (properties.csv's magnitude), target R_j/sigma_j, R_j the rupture set's
    targetOnFaultSupraSeisMFD_SansTVZ (inversion_target_mfds.json), sigma_j / R_j = 0.1 (R_j / R_{7.0-7.1})^-0.25, the
    inputs' rule (reproduced to 1e-6 at 630,000 ruptures); with MFD_GR cleaning replaces them;
  - the paleo rows, one per site of paleo_constraint_data.json in file order: coefficient the UCERF3 probability that
    the site sees the rupture over the site's one-sigma sd (paleoProbModel's table, bilinear in magnitude and in the
    site subsection's midpoint's distance along the rupture's sections, by trace length, folded to the nearer end;
    reproduced to 2e-13 at 630,000 ruptures), target bestEstimate over the sd.

A rupture set made without an inversion (2026-11-23, the user's request; the 630k NZSHM22_RuptureSet-... folder) lacks
three of those files, and each is then built as NZSHM22 builds it:
  - sect_slip_rates.csv: rate = SLIP_RATE_FACTOR (0.8, NZSHM22_SlipRateFactors, or the rupture set's own when its
    logic_tree_branch.json names one) x SlipRate, SD = SlipRateStdDev, from fault_sections.geojson (the relation held on
    every subsection of the 211k and 630k solutions);
  - inversion_target_mfds.json's supra-seismogenic target: NZSHM22's partitioning at its run (mfd_gr.rupture_set_parts:
    N5 3.4, b 0.959, the subsections' shares from fault_polygons.py; within ~0.1% of NZSHM22's own);
  - paleo_constraint_data.json: data/'s sites, each at its nearest subsection within 5.25 km (paleo_misfit.paleo_data).
"""
import json
import os

import numpy as np
import scipy.sparse as sp

PENALTY = 1e20                                           # above preliminaries.py's --huge (1e10)
MIN_MAG = 6.8                                            # the smallest supra-seismogenic magnitude (NZSHM22's crustal 6.8)
SLIP_RATE_FACTOR = 0.8                                   # NZSHM22_SlipRateFactors (sans- and TVZ), for a rupture set without one


def _trace_km(coords):
    c = np.radians(np.asarray(coords, float)[:, :2])
    lon, lat = c[:, 0], c[:, 1]
    a = np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
    return float(np.sum(2 * 6371.0088 * np.arcsin(np.sqrt(a))))


def read_sections(rdir):
    """Each rupture's subsections, in order: (rupture of each membership, its section, its position, counts)."""
    rup, sec, pos, cnt = [], [], [], []
    with open(os.path.join(rdir, 'indices.csv')) as f:
        next(f)
        for line in f:
            p = line.rstrip().split(',')
            k = int(p[1])
            s = np.array(p[2:2 + k], np.int64)
            rup.append(np.full(k, int(p[0]), np.int64)); sec.append(s); pos.append(np.arange(k)); cnt.append(k)
    return np.concatenate(rup), np.concatenate(sec), np.concatenate(pos), np.array(cnt)


def paleo_visibility(model, mag, frac):
    """UCERF3's probability that a site sees a rupture: the table's bilinear interpolation at (frac, mag), frac the
    site's position along the rupture, folded to [0, 0.5]; magnitudes and positions held to the table's range."""
    x = model['xyz']
    T = np.asarray(x['data'], float)
    frac = np.minimum(frac, 1.0 - frac)
    gx = np.clip((frac - x['minX']) / x['gridSpacingX'], 0, x['nx'] - 1)
    gy = np.clip((np.asarray(mag, float) - x['minY']) / x['gridSpacingY'], 0, x['ny'] - 1)
    j0 = np.minimum(np.floor(gx).astype(int), x['nx'] - 2)
    i0 = np.minimum(np.floor(gy).astype(int), x['ny'] - 2)
    fx, fy = gx - j0, gy - i0
    return (T[i0, j0] * (1 - fx) * (1 - fy) + T[i0, j0 + 1] * fx * (1 - fy) + T[i0 + 1, j0] * (1 - fx) * fy
            + T[i0 + 1, j0 + 1] * fx * fy)


def _min_mags(par, rup, sec, mag, min_mag):
    """Each section's minimum magnitude (NZSHM22_FaultSystemRupSetCalc.computeMinSeismoMagForSections): the larger of
    min_mag and the smallest magnitude among the ruptures confined to its parent fault."""
    n = len(mag)
    pmin_, pmax_ = np.full(n, np.iinfo(np.int64).max), np.full(n, np.iinfo(np.int64).min)
    np.minimum.at(pmin_, rup, par[sec]); np.maximum.at(pmax_, rup, par[sec])
    small = {}
    for r_ in np.flatnonzero(pmin_ == pmax_):            # ruptures within one parent fault
        small[pmin_[r_]] = min(small.get(pmin_[r_], np.inf), mag[r_])
    return np.array([max(min_mag, small.get(p_, -np.inf)) for p_ in par])


def section_min_mags(rdir, min_mag=MIN_MAG):
    """Each subsection's minimum supra-seismogenic magnitude by NZSHM22's rule, from the rupture set at rdir."""
    mag = np.loadtxt(os.path.join(rdir, 'properties.csv'), delimiter=',', skiprows=1, usecols=1)
    rup, sec, _, _ = read_sections(rdir)
    feats = json.load(open(os.path.join(rdir, 'fault_sections.geojson')))['features']
    return _min_mags(np.array([f['properties']['ParentID'] for f in feats]), rup, sec, mag, min_mag)


def slip_rate_factor(rdir):
    """The rupture set's slip-rate factor (logic_tree_branch.json's NZSHM22_SlipRateFactors, sans-TVZ), else 0.8."""
    f = os.path.join(rdir, 'logic_tree_branch.json')
    if os.path.exists(f):
        tree = json.load(open(f))
        for lv in (tree.get('values', []) if isinstance(tree, dict) else tree):
            if 'SlipRateFactors' in str(lv.get('level', {}).get('name', '')):
                v = lv.get('value', {}).get('adapterValue', {}).get('adapterData', {})
                if 'sansFactor' in v:
                    return float(v['sansFactor'])
    return SLIP_RATE_FACTOR


def slip_rates(rdir, n_sect):
    """(rate, SD) per subsection in m/yr: sect_slip_rates.csv, or else built from fault_sections.geojson as NZSHM22 builds
    it (rate = the slip-rate factor x SlipRate, SD = SlipRateStdDev). Also says which."""
    f = os.path.join(rdir, 'sect_slip_rates.csv')
    rate, sd = np.zeros(n_sect), np.zeros(n_sect)
    if os.path.exists(f):
        sr = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
        rate[sr[:, 0].astype(int)], sd[sr[:, 0].astype(int)] = sr[:, 1], sr[:, 2]
        return rate, sd, 'sect_slip_rates.csv'
    feats = json.load(open(os.path.join(rdir, 'fault_sections.geojson')))['features']
    fac = slip_rate_factor(rdir)
    for k, g in enumerate(feats):
        rate[int(g['properties']['FaultID'])] = fac * float(g['properties']['SlipRate']) * 1e-3
        sd[int(g['properties']['FaultID'])] = float(g['properties']['SlipRateStdDev']) * 1e-3
    return rate, sd, f'{fac:g} x SlipRate, SD SlipRateStdDev (fault_sections.geojson; no sect_slip_rates.csv)'


def supra_target(rdir):
    """The supra-seismogenic target per 0.1 bin, {bin centre (2 decimals): rate}: inversion_target_mfds.json's
    targetOnFaultSupraSeisMFD_SansTVZ, or else NZSHM22's partitioning at its run (mfd_gr.rupture_set_parts). Also says which."""
    f = os.path.join(rdir, 'inversion_target_mfds.json')
    if os.path.exists(f):
        tg = json.load(open(f))['constraints']
        tg = next(c for c in tg if c['name'].endswith('targetOnFaultSupraSeisMFD_SansTVZ'))
        return {round(x, 2): y for x, y in tg['values']}, 'targetOnFaultSupraSeisMFD_SansTVZ'
    import mfd_gr
    parts = mfd_gr.rupture_set_parts(rdir)
    st = parts['stored']['supra']
    return ({round(float(x), 2): float(y) for x, y in zip(mfd_gr.X, st) if y > 1e-19},
            f'NZSHM22\'s partitioning at N5 {parts["n5_run"]:g}, b {parts["b_run"]:g} (mfd_gr.py; no inversion_target_mfds.json)')


def build_inputs(rdir, lowest=6.8, width=0.1, min_mag=MIN_MAG):
    """G (CSR) and d as cru_A and cru_D give them, from the rupture set at rdir; info: the row ranges and counts."""
    mag = np.loadtxt(os.path.join(rdir, 'properties.csv'), delimiter=',', skiprows=1, usecols=1)
    slip = np.loadtxt(os.path.join(rdir, 'average_slips.csv'), delimiter=',', skiprows=1, usecols=1)
    n = len(mag)
    rup, sec, pos, cnt = read_sections(rdir)
    if len(cnt) != n or len(slip) != n:
        raise SystemExit(f'{rdir}: properties.csv, average_slips.csv and indices.csv disagree on the number of ruptures')
    S = len(json.load(open(os.path.join(rdir, 'fault_sections.geojson')))['features'])
    rate, sd, slip_src = slip_rates(rdir, S)
    sig_s = np.where(sd > 0, sd, rate)                  # no factor 2 (the 19k and 211k inputs: twice this)
    sig_s = np.where(sig_s > 0, sig_s, 1.0)              # a section with no rate and no sd: its row is a target of 0
    rows, cols, vals = [sec], [rup], [slip[rup] / sig_s[sec]]
    d = [rate / sig_s]
    # the penalty row: ruptures below their sections' minimum magnitude
    feats = json.load(open(os.path.join(rdir, 'fault_sections.geojson')))['features']
    secmin = _min_mags(np.array([f['properties']['ParentID'] for f in feats]), rup, sec, mag, min_mag)
    rupmin = np.full(n, -np.inf)
    np.maximum.at(rupmin, rup, secmin[sec])
    binned = lambda m_: np.floor((m_ - lowest) / width + 1e-9)   # the 0.1 bins, from 6.8
    pen = np.flatnonzero(binned(mag) < binned(rupmin))
    r_pen = S
    rows.append(np.full(len(pen), r_pen)); cols.append(pen); vals.append(np.full(len(pen), PENALTY)); d.append([0.0])
    # the magnitude bins
    target, mfd_src = supra_target(rdir)
    nb = int(np.ceil((mag.max() - lowest) / width - 1e-9))
    lo = lowest + width * np.arange(nb)
    R = np.array([target.get(round(l + width / 2, 2), 0.0) for l in lo])
    k7 = int(np.argmin(np.abs(lo - 7.0)))
    rel = 0.1 * (np.where(R > 0, R, np.nan) / R[k7]) ** -0.25
    kb = np.floor((mag - lowest) / width + 1e-9).astype(int)
    r0 = S + 1
    for k in range(nb):
        if not R[k] > 0:
            continue
        m = np.flatnonzero(kb == k)
        sj = rel[k] * R[k]
        rows.append(np.full(len(m), r0)); cols.append(m); vals.append(np.full(len(m), 1.0 / sj)); d.append([R[k] / sj])
        r0 += 1
    n_bins = r0 - (S + 1)
    # the paleo rows
    from paleo_misfit import paleo_data, PALEO_JSON
    pd = paleo_data(rdir, verbose=False)
    paleo_src = ('paleo_constraint_data.json' if os.path.exists(os.path.join(rdir, 'paleo_constraint_data.json')) else
                 f'{PALEO_JSON}\'s sites mapped onto the subsections (the rupture set has none)')
    L = np.array([_trace_km(f['geometry']['coordinates'] if f['geometry']['type'] == 'LineString'
                            else f['geometry']['coordinates'][0]) for f in feats])
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])   # each rupture's first membership
    cum = np.cumsum(L[sec])                               # lengths along each rupture's sections
    before = cum - L[sec] - (cum[start] - L[sec][start])[rup]
    total = np.add.reduceat(L[sec], start)
    sites = pd['paleoRateConstraints']
    r_paleo0 = r0
    for site in sites:
        s = int(site['sectionIndex'])
        sdp = next(u['stdDev'] for u in site['uncertainties'] if u['type'] == 'ONE_SIGMA')
        e = np.flatnonzero(sec == s)
        frac = (before[e] + 0.5 * L[s]) / total[rup[e]]
        pv = paleo_visibility(pd['paleoProbModel']['data'], mag[rup[e]], frac)
        rows.append(np.full(len(e), r0)); cols.append(rup[e]); vals.append(pv / sdp); d.append([site['bestEstimate'] / sdp])
        r0 += 1
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r0, n))
    A.sum_duplicates()
    info = dict(slip_rows=(0, S), penalty_row=r_pen, n_penalised=len(pen), bin_rows=(S + 1, S + 1 + n_bins),
                paleo_rows=(r_paleo0, r0), lowest=lowest, width=width, slip_source=slip_src, mfd_source=mfd_src,
                paleo_source=paleo_src)
    return A, np.concatenate([np.asarray(x, float) for x in d]), info

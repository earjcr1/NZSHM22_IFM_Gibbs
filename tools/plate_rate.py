#!/usr/bin/env python3
"""plate_rate.py -- the faults' slip across the Pacific-Australia plate boundary, cross-section by cross-section.

    python plate_rate.py RUPTURE_SET [--run spikeslab] [--problem cru_clean.npz] [--limits 0.4,0.8] [--strips 8]
                         [--slip-sum-region all]

The cross-sections are perpendicular to the plate motion everywhere: great circles through the Euler pole (POLE:
Beavan et al., 2002, Pacific relative to Australia, 61.04 S, 184.19 E, 1.08 deg/Myr), each labelled by its bearing from
the pole. Two of them are the region's edges: END_SW (167.83 E, 44.53 S), where the Alpine Fault goes offshore (the
Puysegur subduction zone beyond), and END_NE (174.4258 E, 41.67585 S), the midpoint of the step-over between the Needles
Fault's NE end and the Wairarapa Fault's SW end (Wairarapa: Needles) in Cook Strait, beyond which the Hikurangi
subduction zone takes up strain. Between them lie the Alpine Fault from George to Jacksons (10 of its 12 subsections)
to Tophouse, the Wairau and Wairau 2 (its eastern subsection cut by the edge), and the whole Hope Fault. Two quantities
use them, both built from the slip-rate rows (a subsection's slip rate is the sum over its ruptures of rate x average
slip, the rupture set's average_slips.csv), so both are linear in the rupture rates.

The plate rate (gibbs.py --plate-rate, a hard constraint): the region between the edges is cut into --strips strips
of equal width (8: about 76 km each), and each strip's share of the plate rate is the average over its cross-sections:

    share_k = sum over subsections i of w_ki x slip rate_i x p_i

  w_ki: the fraction of strip k's cross-sections that cross subsection i (its trace's extent in bearing inside the
        strip, over the strip's width), so a fault counts once however many of its subsections the strip holds, and a
        subsection across a strip edge counts in both strips in proportion;
  p_i:  the horizontal slip of the block on the Pacific side of subsection i relative to the block on the Australian
        side (from its rake, dip and dip direction, Aki-Richards; the Pacific side is the one towards the pole), per unit
        slip, projected on the local direction of plate motion and divided by the local plate rate: its share of the
        plate rate per mm/yr of slip. Only the component along the plate motion is counted; the one across it (which
        should add up to 0) is sensitive to the rakes and dips and is reported only.

The summed slip rates (gibbs.py --slip-sum, a soft term in the likelihood): along every cross-section, the model's slip
rates added up should match the targets added up, within the uncertainty of that sum. Each subsection counts once
in all: a cross-section crossing subsection i counts its slip rate with weight 1 / e_i, e_i its trace's extent in
bearing, so that over all the cross-sections each subsection's weights add up to 1 (in strip k, w_ki = the fraction of
its extent inside the strip). Integrated over the cross-sections, the term is

    (1 / L0) x integral over s of ((model sum - target sum) / sigma(s))^2 ds

taken over strips whose width is the subsections' mean trace length (8.5 km at 211,000 ruptures: 72 strips between
the edges, the same width on beyond them over the rest of New Zealand with --slip-sum all, the default): each strip
contributes (width / L0) x ((sum_i w_ki (model_i - target_i)) / sigma_k)^2. Narrower strips only approximate the
integral more closely (the term converges as they narrow; at 8.5 km within about 5% of its limit); L0 sets its
weight (--slip-sum-l0, SLIP_SUM_L0 = 8.5 km by default, the strips' width, so that each strip's sum counts as one extra
observation: 72 between the edges). For comparison, the target sums' decorrelation length along the plate motion
between the edges is 120 km (the integral of their correlation at 211,000 ruptures), close to the span between the
edges over the 5 sections of the Alpine and Wairau Faults there (George to Jacksons, Jacksons to Kaniere, Kaniere to
Springs Junction, Springs Junction to Tophouse, Wairau; Wairau 2, cut by the NE edge, counted out), 122 km, which
dominate the sums' uncertainty: an L0 of 120 km would count the regional sums' information once more (about 5
measurements between the edges), 8.5 km about 14 times. sigma_k is the uncertainty of the strip's target
sum, with copies of one fault section's slip rate (its subsections, which share one rate and one uncertainty) fully
correlated and sections independent: sigma_k^2 = sum over sections p of (sum over i in p of w_ki sigma_i)^2. Each
subsection's sigma_i and target are the cleaned problem's own (sigma_i from its row's coefficients against the
rupture set's average slips: twice the CSV standard deviation in the NZSHM22 inputs, the rate itself where that is
0). The subsections no rupture can reach (removed in cleaning) are left out of both sums. slip_sum_rows() builds the
rows, sqrt(width / L0) / sigma_k x (sum_i w_ki model_i - target sum), one per strip.

This script reports both (to plate_rate.txt beside --run's rate vectors, or --out): per plate-rate strip, the shares of
the fault model's slip rates (fault_sections.geojson's SlipRate), of the targets (sect_slip_rates.csv, this logic-tree
branch's scaled rates), of the best fit (nnls/nnls_vectors.npz beside --problem), of the posterior mean and, with kept
draws, of every kept draw (their 2.5-97.5% range and the fraction inside --limits), with the faults that take most of
it; and the summed slip rates' term, SW of the edges, between them and NE of them, for the best fit, the posterior mean
and the kept draws, with the strips furthest from their target sums."""
import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

POLE = (-61.04, 184.19, 1.08)          # Beavan et al. (2002): Pacific relative to Australia (lat, lon, deg/Myr)
END_NE = (174.4258, -41.67585)         # lon, lat: midway between the Needles Fault's NE end (174.4533 E, 41.6900 S) and
                                       # the Wairarapa: Needles SW end (174.3983 E, 41.6617 S), the Cook Strait step-over
END_SW = (167.83, -44.53)              # lon, lat: the Alpine Fault goes offshore (the Puysegur subduction zone beyond)
SLIP_SUM_L0 = 8.5                      # km: the summed slip rates' L0, the strips' width, so each strip's sum counts
                                       # as one extra observation (120 km: the target sums' decorrelation length between
                                       # the edges, about the span over its 5 Alpine-Wairau sections, 122 km)
EARTH_R = 6371.0                       # km


def _xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack(np.broadcast_arrays(np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)), -1)


def bearing(lon1, lat1, lon2, lat2):
    """Initial bearing (degrees east of north) of the great circle from point 1 to point 2."""
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(np.asarray(lon2) - lon1)
    return np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)))


def _velocity(lon, lat, pole):
    plat, plon, w = pole
    v = np.cross(np.radians(w) / 1e6 * _xyz(plon, plat), _xyz(lon, lat)) * EARTH_R * 1e6
    lo, la = np.radians(lon), np.radians(lat)
    e = np.stack(np.broadcast_arrays(-np.sin(lo), np.cos(lo), 0.0 * lo), -1)
    n = np.stack(np.broadcast_arrays(-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)), -1)
    return np.stack([(v * e).sum(-1), (v * n).sum(-1)], -1)


def plate_velocity(lon, lat, pole=POLE):
    """The velocity (east, north; mm/yr) of the Pacific plate relative to the Australian plate at (lon, lat): the pole's
    rotation taken in the sense that moves the Pacific south-west relative to Australia at END_NE (Australia north-east
    relative to the Pacific)."""
    ve, vn = _velocity(*END_NE, pole)
    return _velocity(lon, lat, pole) * (-1.0 if ve > 0 and vn > 0 else 1.0)


def _sections(rupture_dir):
    with open(os.path.join(rupture_dir, 'fault_sections.geojson')) as f:
        feats = json.load(f)['features']
    if [ft['properties']['FaultID'] for ft in feats] != list(range(len(feats))):
        raise ValueError(f'{rupture_dir}/fault_sections.geojson: FaultID is not 0, 1, 2, ... in order')
    return feats


def _position(pole, ends):
    """u(lon, lat): a point's position across the region in bearing from the pole, 0 on END_SW's cross-section and 1 on
    END_NE's (beyond them below 0 or above 1), and the span between the two in km, at their mean distance from the
    pole."""
    plat, plon, _ = pole
    (lne, ane), (lsw, asw) = ends
    s_ne, s_sw = float(bearing(plon, plat, lne, ane)), float(bearing(plon, plat, lsw, asw))
    ref = 0.5 * (s_ne + s_sw)
    wrap = lambda s: (np.asarray(s) - ref + 180.0) % 360.0 - 180.0 + ref    # bearings near the region's, no 360 jump
    s_ne, s_sw = float(wrap(s_ne)), float(wrap(s_sw))
    dist = np.mean([np.arccos(np.clip(_xyz(plon, plat) @ _xyz(*e), -1.0, 1.0)) for e in ends])   # from the pole, rad
    span_km = abs(np.radians(s_ne - s_sw)) * EARTH_R * np.sin(dist)
    return (lambda lon, lat: (wrap(bearing(plon, plat, lon, lat)) - s_sw) / (s_ne - s_sw)), float(span_km)


def strips(rupture_dir, n_strips=8, pole=POLE, ends=(END_NE, END_SW)):
    """The plate-rate strips between the two cross-sections and each subsection's part in them: dict(W, the strips x
    subsections weights w_ki; p, each subsection's share of the plate rate per mm/yr of slip; q, the same for the
    component across the plate motion; V, the local plate rate; edges, the strips' edges in position (0 at END_SW, 1 at
    END_NE); width_km, their width halfway between the ends; sections, the geojson features; rate_cfm, the fault
    model's slip rates, mm/yr; rate_target, the slip-rate targets, mm/yr, or None)."""
    feats = _sections(rupture_dir)
    u_of, span_km = _position(pole, ends)
    plat, plon, _ = pole
    edges = np.linspace(0.0, 1.0, n_strips + 1)
    n = len(feats)
    W = np.zeros((n_strips, n))
    p, q, V = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, ft in enumerate(feats):
        pr = ft['properties']
        if pr.get('Partition', 'CRUSTAL') != 'CRUSTAL':
            continue                                         # subduction interfaces: handled outside this workflow
        t = np.asarray(ft['geometry']['coordinates'], float)[:, :2]
        u = u_of(t[:, 0], t[:, 1])
        a, b = np.minimum(u[:-1], u[1:]), np.maximum(u[:-1], u[1:])         # each trace segment's extent across
        ov = np.clip(np.minimum(b[None, :], edges[1:, None]) - np.maximum(a[None, :], edges[:-1, None]), 0.0, None)
        W[:, i] = ov.sum(1) / np.diff(edges)
        if not W[:, i].any():
            continue
        dd, dip, rake = np.radians(pr['DipDir']), np.radians(pr['DipDeg']), np.radians(pr['Rake'])
        st = dd - np.pi / 2                                  # strike, Aki-Richards (the hanging wall on the right)
        shat, dhat = np.array([np.sin(st), np.cos(st)]), np.array([np.sin(dd), np.cos(dd)])
        uh = np.cos(rake) * shat - np.sin(rake) * np.cos(dip) * dhat         # hanging wall relative to footwall
        lon, lat = 0.5 * (t[0] + t[-1])
        tp = np.radians(bearing(lon, lat, plon, plat))       # towards the pole: towards the Pacific side
        that = np.array([np.sin(tp), np.cos(tp)])
        if dhat @ that < 0:
            uh = -uh                                         # the hanging wall on the Australian side
        v = plate_velocity(lon, lat, pole)
        V[i] = np.hypot(*v)
        p[i] = uh @ v / V[i] ** 2
        q[i] = uh @ that / V[i]
    rate_target = None
    f = os.path.join(rupture_dir, 'sect_slip_rates.csv')
    if os.path.exists(f):
        t = np.loadtxt(f, delimiter=',', skiprows=1)
        if len(t) == n and np.array_equal(t[:, 0], np.arange(n)):
            rate_target = t[:, 1] * 1e3
    return dict(W=W, p=p, q=q, V=V, edges=edges, width_km=span_km / n_strips, sections=feats,
                rate_cfm=np.array([ft['properties']['SlipRate'] for ft in feats], float), rate_target=rate_target,
                pole=pole, ends=ends)


def rupture_matrix(rupture_dir, n_sections):
    """Subsections x ruptures: each rupture's average slip (m, average_slips.csv) on each of its subsections
    (indices.csv)."""
    from make_solution import read_rupture_set
    _, ptr, idx = read_rupture_set(rupture_dir)
    D = np.loadtxt(os.path.join(rupture_dir, 'average_slips.csv'), delimiter=',', skiprows=1)
    if not np.array_equal(D[:, 0], np.arange(len(ptr) - 1)):
        raise ValueError(f'{rupture_dir}/average_slips.csv: rupture indices are not 0, 1, 2, ... in order')
    col = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    return sp.csr_matrix((D[col, 1], (idx, col)), shape=(n_sections, len(ptr) - 1))


def share_matrix(st, S):
    """Strips x ruptures: each rupture's contribution to each strip's share per unit rate (1/yr), A = W diag(p) S x 1000
    (slip in mm)."""
    return sp.csr_matrix(sp.csr_matrix(st['W'] * st['p'][None, :]) @ S * 1e3)


def _slip_rows(P, slip_block):
    if slip_block not in P['block_names']:
        raise ValueError(f'the cleaned problem has no row block {slip_block!r}')
    rows = np.flatnonzero(P['block'] == list(P['block_names']).index(slip_block))
    return rows, np.asarray(P['keep_rows'])[rows]


def check_problem(P, S, slip_block='S'):
    """Stop unless the cleaned problem's slip-rate rows (row r = section r) are the rupture set's subsections: each row
    has exactly the kept ruptures through its subsection, with coefficients in proportion to their average slips.
    Returns each row's coefficient over the average slip (1 / its uncertainty in m/yr, times its row scale)."""
    rows, sec = _slip_rows(P, slip_block)
    if sec.max() >= S.shape[0]:
        raise ValueError(f'slip-rate row {sec.max()} has no subsection in the rupture set ({S.shape[0]} subsections)')
    G = sp.csr_matrix(P['G'])[rows]
    Sk = sp.csr_matrix(S[:, np.asarray(P['keep_cols'])])[sec]
    G.sort_indices()
    Sk.sort_indices()
    out = np.zeros(len(rows))
    for r in range(len(rows)):
        gi, gv = G.indices[G.indptr[r]:G.indptr[r + 1]], G.data[G.indptr[r]:G.indptr[r + 1]]
        si, sv = Sk.indices[Sk.indptr[r]:Sk.indptr[r + 1]], Sk.data[Sk.indptr[r]:Sk.indptr[r + 1]]
        ratio = gv / sv if np.array_equal(gi, si) else None
        if ratio is None or np.ptp(ratio) > 1e-6 * np.abs(ratio).max():
            raise ValueError(f'slip-rate row of section {sec[r]}: its ruptures or coefficients are not the rupture set\'s '
                             'average slips on that subsection (indices.csv, average_slips.csv)')
        out[r] = ratio.mean()
    return out


def strip_rows(rupture_dir, P, n_strips=8, slip_block='S'):
    """The rows gibbs.py keeps within --plate-rate's limits: A (strips x the cleaned problem's ruptures), each strip's
    share of the plate rate per unit rate, and the strips (strips(), with S, the subsections x ruptures slip matrix)."""
    st = strips(rupture_dir, n_strips)
    S = rupture_matrix(rupture_dir, len(st['sections']))
    if P['raw_shape'][1] != S.shape[1]:
        raise ValueError(f'the inputs have {P["raw_shape"][1]} ruptures and the rupture set {S.shape[1]}')
    check_problem(P, S, slip_block)
    st['S'] = S
    return sp.csr_matrix(share_matrix(st, S)[:, np.asarray(P['keep_cols'])]), st


def slip_sum_rows(rupture_dir, P, region='all', slip_block='S', pole=POLE, ends=(END_NE, END_SW),
                  l0_km=SLIP_SUM_L0, S=None):
    """The summed slip rates' rows (gibbs.py --slip-sum), in the likelihood beside the data: (A, b, info), A (strips x
    the cleaned problem's ruptures) and b such that the term is ||A m - b||^2 / P['sigma']^2, each strip's
    (width / L0) x ((sum_i w_ki (model_i - target_i)) / sigma_k)^2 (see the module's description). region 'all': strips
    over all of New Zealand; 'between': only the strips between the edges (a subsection across an edge counts by the
    fraction of its extent inside); l0_km: L0, km."""
    if region not in ('all', 'between'):
        raise ValueError(f'region {region!r}: use all or between')
    feats = _sections(rupture_dir)
    n = len(feats)
    if S is None:
        S = rupture_matrix(rupture_dir, n)
    if P['raw_shape'][1] != S.shape[1]:
        raise ValueError(f'the inputs have {P["raw_shape"][1]} ruptures and the rupture set {S.shape[1]}')
    rows, sec = _slip_rows(P, slip_block)
    coef = check_problem(P, S, slip_block)                   # coefficient / average slip = row scale / sigma_i (m/yr)
    rs = np.asarray(P['row_scale'], float)[rows]
    sig_i = rs / coef * 1e3                                  # each subsection's uncertainty, mm/yr
    tgt_i = np.asarray(P['d'], float)[rows] / rs * sig_i     # and its target, mm/yr
    u_of, span_km = _position(pole, ends)
    segs, umid, length = [], np.zeros(len(sec)), np.zeros(len(sec))
    for j, i in enumerate(sec):
        t = np.asarray(feats[i]['geometry']['coordinates'], float)[:, :2]
        u = u_of(t[:, 0], t[:, 1])
        segs.append((np.minimum(u[:-1], u[1:]), np.maximum(u[:-1], u[1:])))
        umid[j] = u_of(*(0.5 * (t[0] + t[-1])))
        dx = np.diff(t[:, 0]) * 111.32 * np.cos(np.radians(t[:, 1].mean()))
        length[j] = np.hypot(dx, np.diff(t[:, 1]) * 110.574).sum()
    N = max(1, int(round(span_km / length.mean())))          # strips between the edges: the mean trace length wide
    if region == 'all':
        lo = min(a.min() for a, _ in segs + [(umid, umid)])
        hi = max(b.max() for _, b in segs + [(umid, umid)])
        k0, k1 = int(np.floor(lo * N + 1e-9)), int(np.ceil(hi * N - 1e-9))
    else:
        k0, k1 = 0, N
    k0, k1 = min(k0, 0), max(k1, N)
    edges = np.arange(k0, k1 + 1) / N
    W = np.zeros((len(edges) - 1, len(sec)))
    for j, (a, b) in enumerate(segs):
        e = (b - a).sum()
        if e > 1e-12:                                        # the fraction of the subsection's extent in each strip
            W[:, j] = np.clip(np.minimum(b[None, :], edges[1:, None]) - np.maximum(a[None, :], edges[:-1, None]),
                              0.0, None).sum(1) / e
        elif edges[0] <= umid[j] < edges[-1]:                # no extent across the plate motion: its midpoint's strip
            W[int(np.searchsorted(edges, umid[j], side='right')) - 1, j] = 1.0
    if region == 'all' and not np.allclose(W.sum(0), 1.0, atol=1e-9):
        raise ValueError('the strips do not cover every subsection once')        # cannot happen: a check
    kidx = np.arange(k0, k1)                                 # strip numbers: < 0 SW of the edges, >= N NE of them
    if region == 'between':
        W, kidx = W[(kidx >= 0) & (kidx < N)], kidx[(kidx >= 0) & (kidx < N)]
    par = np.array([feats[i]['properties']['ParentID'] for i in sec])
    upar, pinv = np.unique(par, return_inverse=True)
    M = sp.csr_matrix((np.ones(len(sec)), (pinv, np.arange(len(sec)))), shape=(len(upar), len(sec)))
    Bk = (M @ sp.csr_matrix(W * sig_i[None, :]).T).T.toarray()          # strips x sections: each section's share
    sig_k = np.sqrt((Bk ** 2).sum(1))
    keep = sig_k > 0
    W, kidx, sig_k = W[keep], kidx[keep], sig_k[keep]
    T = W @ tgt_i
    L0_km = float(l0_km)
    if not L0_km > 0:
        raise ValueError(f'L0 = {l0_km!r} km: give a positive length')
    wt = span_km / N / L0_km                                 # width / L0
    c = np.sqrt(wt) / sig_k * float(P['sigma'])
    Sk = sp.csr_matrix(S[:, np.asarray(P['keep_cols'])])[sec] * 1e3        # subsections x ruptures, mm per unit rate
    A = sp.csr_matrix(sp.diags(c) @ sp.csr_matrix(W) @ Sk)
    A.eliminate_zeros()
    lon = np.array([np.mean(np.asarray(feats[i]['geometry']['coordinates'], float)[:, 0]) for i in sec])
    lat = np.array([np.mean(np.asarray(feats[i]['geometry']['coordinates'], float)[:, 1]) for i in sec])
    cw = np.maximum(W.sum(1), 1e-300)
    info = dict(N=N, k=kidx, W=W, sec=sec, target=T, sigma=sig_k, scale=c, weight=wt,
                width_km=span_km / N, L0_km=L0_km, span_km=span_km,
                mean_length_km=float(length.mean()),
                region=region, ends=ends, pole=pole, lon=(W @ lon) / cw, lat=(W @ lat) / cw, subsections=W.sum(1),
                sig_i=sig_i, tgt_i=tgt_i, n_parents=len(upar))
    return A, c * T, info


def describe_slip_sum(info, nnz=None):
    """A line on the summed slip rates' strips."""
    k, N = info['k'], info['N']
    where = (f'{len(k)} strips over all of New Zealand ({int((k < 0).sum())} SW of the edges, {int(((k >= 0) & (k < N)).sum())} '
             f'between them, {int((k >= N).sum())} NE of them)' if info['region'] == 'all' else
             f'{len(k)} strips between the edges')
    return (f'{where}, {info["width_km"]:.1f} km wide (the subsections\' mean trace length, {info["mean_length_km"]:.1f} km); '
            + f'L0 = {info["L0_km"]:g} km, so the term counts as {info["span_km"] / info["L0_km"]:.1f} measurements over the '
            f'{info["span_km"]:.0f} km between the edges'
            + f', each strip weighted {info["weight"]:.4f}; {int(round(info["subsections"].sum()))} '
            f'subsections in {info["n_parents"]} sections' + (f'; {nnz:,} nonzeros' if nnz is not None else ''))


def contributors(st, k, rates, top=4):
    """The parent faults that take most of strip k's share, with their parts."""
    out = {}
    for i in np.flatnonzero(st['W'][k]):
        name = st['sections'][i]['properties'].get('ParentName', str(i)).split(':')[0]
        out[name] = out.get(name, 0.0) + st['W'][k, i] * rates[i] * st['p'][i]
    return ', '.join(f'{a} {b:.2f}' for a, b in sorted(out.items(), key=lambda t: -abs(t[1]))[:top])


def describe(st):
    """A line on the plate-rate strips."""
    inside = st['W'].any(0)
    return (f'{st["W"].shape[0]} strips of {st["width_km"]:.0f} km between the cross-sections through '
            f'{st["ends"][1][0]:.2f} E {-st["ends"][1][1]:.2f} S and {st["ends"][0][0]:.4f} E {-st["ends"][0][1]:.4f} S, '
            f'perpendicular to the plate motion (Euler pole {-st["pole"][0]:.2f} S {st["pole"][1]:.2f} E, '
            f'{st["pole"][2]:g} deg/Myr; {st["V"][inside].min():.1f}-{st["V"][inside].max():.1f} mm/yr): '
            f'{int(inside.sum())} subsections')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ruptures', help='the OpenSHA rupture set (an OpenSHA solution folder, or its ruptures/ folder)')
    ap.add_argument('--run', default=None, help='a results folder (spikeslab/) with rate_vectors.npz: the posterior mean '
                    'and kept draws')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (its nnls/ best fit is reported too)')
    ap.add_argument('--strips', type=int, default=8, help='plate-rate strips between the two cross-sections (as '
                    'gibbs.py\'s --plate-strips)')
    ap.add_argument('--limits', default=None, help='LO,HI: the limits gibbs.py --plate-rate kept (counted against)')
    ap.add_argument('--slip-sum-region', default='all', choices=['all', 'between'],
                    help='the summed slip rates\' strips reported (as gibbs.py --slip-sum)')
    ap.add_argument('--slip-sum-l0', type=float, default=SLIP_SUM_L0, metavar='KM',
                    help=f'the summed slip rates\' L0 in km (as gibbs.py --slip-sum-l0; default {SLIP_SUM_L0:g})')
    ap.add_argument('--slip-block', default='S')
    ap.add_argument('--out', default=None, help='report file (default: plate_rate.txt in --run, or beside --problem)')
    a = ap.parse_args()
    from cru_common import find_rupture_set, load_problem, report_to, kept_draws
    rd = find_rupture_set(a.ruptures, search=False)
    if rd is None:
        raise SystemExit(f'{a.ruptures}: no OpenSHA rupture set (indices.csv, or ruptures/indices.csv)')
    out = a.out or os.path.join(a.run or os.path.dirname(os.path.abspath(a.problem)), 'plate_rate.txt')
    report_to(out)
    st = strips(rd, a.strips)
    S = rupture_matrix(rd, len(st['sections']))
    A = share_matrix(st, S)
    vecs, Kd = {}, None
    P = load_problem(a.problem) if os.path.exists(a.problem) else None
    if P is not None:
        bf = os.path.join(os.path.dirname(os.path.abspath(a.problem)), 'nnls', 'nnls_vectors.npz')
        if os.path.exists(bf):
            with np.load(bf) as z:
                vecs['best fit'] = np.asarray(z['mean'], float)
    if a.run:
        with np.load(os.path.join(a.run, 'rate_vectors.npz')) as z:
            vecs['posterior mean'] = np.asarray(z['mean'], float)
            Kd, _ = kept_draws(z)
    print('plate rate: the share of the relative plate motion that the faults take up, strip by strip from SW to NE')
    print('   ' + describe(st))
    lim = tuple(float(v) for v in a.limits.split(',')) if a.limits else None
    cols = {'fault model': st['W'] @ (st['rate_cfm'] * st['p'])}
    if st['rate_target'] is not None:
        cols['targets'] = st['W'] @ (st['rate_target'] * st['p'])
    if P is not None:
        check_problem(P, S, a.slip_block)
    cols.update({k: A @ v for k, v in vecs.items()})
    K = None
    if Kd is not None:
        K = np.vstack([(A @ np.asarray(Kd[c:c + 250], np.float64).T).T for c in range(0, len(Kd), 250)])  # draws x strips
    print('   across the plate motion (should add up to 0), the fault model: '
          + ' '.join(f'{v:+.2f}' for v in st['W'] @ (st['rate_cfm'] * st['q'])))
    head = '   strip ' + ''.join(f'{c:>16s}' for c in cols)
    if K is not None:
        head += f'{"kept draws, 2.5-97.5%":>26s}' + (f'{"inside":>9s}' if lim else '')
    print(head + '   main contributors (fault model)')
    for k in range(a.strips):
        line = f'   {k + 1:5d} ' + ''.join(f'{v[k]:16.3f}' for v in cols.values())
        if K is not None:
            lo, hi = np.percentile(K[:, k], [2.5, 97.5])
            line += f'{lo:16.3f} - {hi:7.3f}'
            if lim:
                line += f'{100 * np.mean((K[:, k] >= lim[0] - 1e-9) & (K[:, k] <= lim[1] + 1e-9)):8.1f}%'
        print(line + '   ' + contributors(st, k, st['rate_cfm']))
    if K is not None:
        print(f'   kept draws: {len(K)}' + (f'; limits {lim[0]:g}-{lim[1]:g}: all strips inside in '
                                           f'{100 * np.mean(np.all((K >= lim[0] - 1e-9) & (K <= lim[1] + 1e-9), 1)):.1f}% '
                                           'of them' if lim else ''))
    elif a.run:
        print('   no kept draws in the rate vectors (REPRESENTATIVES = 0): the posterior mean only')
    if P is None:
        print('\nsummed slip rates: not reported (no cleaned problem, --problem)')
        return
    # the summed slip rates' term
    As, bs, info = slip_sum_rows(rd, P, a.slip_sum_region, a.slip_block, l0_km=a.slip_sum_l0, S=S)
    kc = np.asarray(P['keep_cols'])
    sg = float(P['sigma'])
    k, N = info['k'], info['N']
    parts = {'SW of the edges': k < 0, 'between the edges': (k >= 0) & (k < N), 'NE of the edges': k >= N}
    parts = {nm: m for nm, m in parts.items() if m.any()}
    print('\nsummed slip rates: along each cross-section, the model\'s slip rates added up against the targets added up '
          '(the soft term gibbs.py --slip-sum adds to the misfit)')
    print('   ' + describe_slip_sum(info))
    print(f'   if the model matched the truth, the term would be near {sum(info["weight"] * m.sum() for m in parts.values()):.1f} '
          '(each strip weighted width / L0)')
    rows = {nm: ((As @ v[kc] - bs) / sg) ** 2 for nm, v in vecs.items()}
    if Kd is not None:
        Rk = np.vstack([((As @ np.asarray(Kd[c:c + 250], np.float64)[:, kc].T).T - bs) / sg for c in range(0, len(Kd), 250)])
        rows['kept draws'] = Rk ** 2
    print('   ' + f'{"":16s}' + ''.join(f'{nm:>26s}' for nm in ['all'] + list(parts)))
    for nm, q in rows.items():
        if q.ndim == 1:
            print('   ' + f'{nm:16s}' + ''.join(f'{v:26.1f}' for v in [q.sum()] + [q[m].sum() for m in parts.values()]))
        else:
            f = lambda v: f'{np.median(v):.1f} ({np.percentile(v, 5):.1f}-{np.percentile(v, 95):.1f})'
            print('   ' + f'{nm + " (median, 5-95%)":16s}' + ''.join(f'{f(v):>26s}' for v in
                                                              [q.sum(1)] + [q[:, m].sum(1) for m in parts.values()]))
    v = vecs.get('posterior mean', vecs.get('best fit'))
    if v is not None:
        nm = 'posterior mean' if 'posterior mean' in vecs else 'best fit'
        z = (As @ v[kc] - bs) / sg / np.sqrt(info['weight'])           # (model sum - target sum) / sigma_k
        sel = np.argsort(-np.abs(z))[:8]
        print(f'   the strips furthest from their target sums ({nm}; z = (model sum - target sum) / sigma_k; '
              f'{int((np.abs(z) > 2).sum())} of {len(z)} beyond 2):')
        for j in sel:
            print(f'      strip {int(k[j]):+4d} ({info["lon"][j]:.2f} E, {-info["lat"][j]:.2f} S; '
                  f'{info["subsections"][j]:.1f} subsections): target sum {info["target"][j]:7.2f} +- {info["sigma"][j]:6.2f} '
                  f'mm/yr, model {info["target"][j] + z[j] * info["sigma"][j]:7.2f} (z {z[j]:+.2f})')


if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

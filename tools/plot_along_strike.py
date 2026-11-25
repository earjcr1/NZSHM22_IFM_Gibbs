#!/usr/bin/env python3
"""Along-strike values, subsection by subsection, for each named fault (the user's request, 2026-11-24, after NZSHM22's
"Along-Strike Values" on its solution reports' special fault pages, without their nucleation, moment-rate and b-value
panels, and with the magnitudes of the ruptures in their place).

    python plot_along_strike.py VECTORS [--problem cru_clean.npz] [--ruptures DIR] [--named FILE] [--config FILE]
                                [--outdir DIR] [--faults NAME,NAME,...] [--reference RATES[:LABEL]] (up to twice)

VECTORS: the run's rate vectors (<prior folder>/rate_vectors.npz): the posterior mean's rate of every rupture of the
rupture set; the kept draws (<prior folder>/main_rep*_kept.npz, or an older run's .npy) give each value's distribution.
--ruptures: the OpenSHA rupture set (fault_sections.geojson, indices.csv, properties.csv, average_slips.csv,
named_faults.json); --named: another named-faults file (fault name -> its parent fault sections' IDs).

One figure per named fault, <outdir>/<fault>.png (default outdir: <prior folder>/along_strike), and one table of every
value, along_strike.csv. x: longitude, or latitude for a fault whose subsections span more north-south than east-west
(km); each subsection is drawn across its own trace's extent on that axis, in geographical order. Four panels:
  1. participation rate (per year): every rupture through the subsection (the supra-seismogenic ruptures, the rupture
     set's), and those of M >= 7 and M >= 8;
  2. paleo-visible participation rate: each rupture weighted by UCERF3's probability that a site at the subsection's
     middle would see it (rupture_set_inputs.paleo_visibility, as the paleo rows weight them), with the paleo sites on
     the fault (paleo_misfit.paleo_data: their best estimates, ONE_SIGMA bounds as thick bars, TWO_SIGMA as thin);
  3. slip rate (mm/yr): the run's target (the slip-rate factor in it; medium grey) and its 2.275% and
     97.725% points as light grey dashed lines (the fault's exact corrected density with MISFIT_SLIP 'exact', its split normal
     otherwise; a band until 2026-11-25; the CFM's SR_pref, unscaled, until 2026-11-25 and still in the table), and the posterior
     slip rate (each rupture's slip on the subsection, plate_rate.rupture_matrix, times its rate);
  4. the magnitudes of the ruptures through each subsection: its posterior mean participation rate in each magnitude bin
     of 0.1 (colour, log scale), and its rate-weighted median magnitude (a line).
Panels 1-3: the posterior mean (a line) and the kept draws shaded by density (since 2026-11-25, the user's request; their
2.275-97.725% as a band before): each subsection's column binned in log10 over the fault's 1-99% of values, darker
where more draws fall, on its own scale; each --reference (NZSHM22 solutions, dashed, in the fit figures' inks) as a line.
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import scipy.sparse as sp

from cru_common import MARK_BLUES, mark_blues_cmap, REF_INKS, find_rupture_set, load_problem, load_rates

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SOL, BAND = MARK_BLUES[3], MARK_BLUES[0]                  # the posterior: a dark blue line over a light blue band
DENS = MARK_BLUES[1]                                       # the kept draws' density shading (2026-11-25, the user's request)
NB_DENS = 60                                               # its bins in log10 over each panel's range
HEAT = mark_blues_cmap()                         # the heat maps: the same blues, continuous
TGT, TGT_BAND = '#6e6d69', '#bfe3df'              # the target, medium grey (teal until 2026-11-25), and its old range band
TGT_RANGE = '#b4b3ad'                             # the target's 2.275% and 97.725% points: light grey dashed lines (2026-11-25)
TGT_RANGE_LS = (0, (4, 2.5))
PLO, PHI = 0.022750131948179195, 1 - 0.022750131948179195


def subsection_geometry(rd):
    feats = json.load(open(os.path.join(rd, 'fault_sections.geojson')))['features']
    props = [f['properties'] for f in feats]
    tr = []
    for f in feats:
        g = f['geometry']
        c = np.asarray(g['coordinates'] if g['type'] == 'LineString' else g['coordinates'][0], float)[:, :2]
        tr.append(c)
    return props, tr


def memberships(rd):
    from make_solution import read_rupture_set
    mag, ptr, idx = read_rupture_set(rd)
    ptr, idx = np.asarray(ptr), np.asarray(idx)
    rup = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    pos = np.arange(len(idx)) - ptr[rup]
    return np.asarray(mag, float), ptr, idx, rup, pos


def paleo_weights(rd, ptr, idx, rup, mag, L):
    """Each membership's UCERF3 visibility at the subsection's middle (as rupture_set_inputs builds the paleo rows)."""
    from paleo_misfit import paleo_data
    from rupture_set_inputs import paleo_visibility
    pdd = paleo_data(rd, verbose=False)
    start = ptr[:-1]
    cum = np.cumsum(L[idx])
    before = cum - L[idx] - (cum[start] - L[idx][start])[rup]
    total = np.add.reduceat(L[idx], start)
    frac = (before + 0.5 * L[idx]) / total[rup]
    return paleo_visibility(pdd['paleoProbModel']['data'], mag[rup], frac), pdd['paleoRateConstraints']


def slip_targets(P, cfg, nsub):
    """Each subsection's target (mm/yr, the slip-rate factor in it) and its 2.275% and 97.725% points, NaN off the
    cleaned slip-rate rows."""
    from plate_rate import check_problem
    rows = np.flatnonzero(P['block'] == list(P['block_names']).index('S'))
    orig = np.asarray(P['keep_rows'])[rows]
    t, lo, hi = (np.full(nsub, np.nan) for _ in range(3))
    f = float(P['slip_pdf_f']) if P.get('slip_pdf_f') is not None else 0.8
    exact = str(cfg.get('misfit_slip')) == 'exact' and P.get('slip_pdf_key') is not None
    if P.get('slip_pdf_mode') is not None:              # the split normal as cleaning built it (m/yr, f in it)
        c = np.asarray(P['slip_pdf_mode'])[rows] * 1e3
        slo, shi = np.asarray(P['slip_pdf_slo'])[rows] * 1e3, np.asarray(P['slip_pdf_shi'])[rows] * 1e3
    else:
        return t, lo, hi, False
    from timeframe_sliprates import sn_quantile
    t[orig] = c
    lo[orig], hi[orig] = sn_quantile(c, slo, shi, PLO), sn_quantile(c, slo, shi, PHI)
    if exact:
        from cru_common import slip_pdf_quantiles
        lo[orig], hi[orig] = slip_pdf_quantiles(P, rows, [PLO, PHI])
    return t, lo, hi, exact


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('vectors', help='the rate vectors (<prior folder>/rate_vectors.npz)')
    ap.add_argument('--problem', default='cru_clean.npz')
    ap.add_argument('--ruptures', default=None, help='the OpenSHA rupture set (default: the run\'s, from its config)')
    ap.add_argument('--named', default=None, help='named faults (default: the rupture set\'s named_faults.json)')
    ap.add_argument('--config', default=None, help='the run\'s workflow_config.json (default: beside the rate vectors, or --problem)')
    ap.add_argument('--outdir', default=None, help='where the figures go (default: <prior folder>/along_strike)')
    ap.add_argument('--faults', default=None, help='only these named faults, comma-separated')
    ap.add_argument('--reference', action='append', default=None, metavar='RATES[:LABEL]',
                    help='another model\'s rates (an OpenSHA solution\'s solution/rates.csv), drawn dashed; up to twice')
    a = ap.parse_args()
    vec = a.vectors if a.vectors.endswith('.npz') else a.vectors + '.npz'
    prior = os.path.dirname(os.path.abspath(vec))
    cfg_path = a.config or next((p for p in (os.path.join(prior, 'workflow_config.json'),
                                             os.path.join(os.path.dirname(os.path.abspath(a.problem)), 'workflow_config.json'))
                                 if os.path.exists(p)), None)
    cfg = json.load(open(cfg_path)) if cfg_path else {}
    rd = find_rupture_set(a.ruptures or cfg.get('rupture_set') or os.path.dirname(os.path.abspath(a.problem)))
    if rd is None:
        raise SystemExit('no rupture set: give --ruptures')
    named = json.load(open(a.named or os.path.join(rd, 'named_faults.json')))
    if a.faults:
        want = [s.strip() for s in a.faults.split(',')]
        named = {k: named[k] for k in want}
    outdir = a.outdir or os.path.join(prior, 'along_strike')
    os.makedirs(outdir, exist_ok=True)
    P = load_problem(a.problem)

    props, tr = subsection_geometry(rd)
    nsub = len(props)
    from rupture_set_inputs import _trace_km
    L = np.array([_trace_km(c) for c in tr])
    mag, ptr, idx, rup, pos = memberships(rd)
    n = len(mag)
    par = np.array([int(p['ParentID']) for p in props])
    subs_of = {f: np.flatnonzero(np.isin(par, np.asarray(ids, int))) for f, ids in named.items()}
    sel = np.unique(np.concatenate([s for s in subs_of.values() if len(s)]))
    rowof = np.full(nsub, -1)
    rowof[sel] = np.arange(len(sel))
    keep = rowof[idx] >= 0                                  # memberships on the named faults' subsections
    R_, C_ = rowof[idx[keep]], rup[keep]
    shape = (len(sel), n)
    inc = sp.csr_matrix((np.ones(keep.sum()), (R_, C_)), shape=shape)
    m_ = mag[C_]
    inc7 = sp.csr_matrix(((m_ >= 7.0).astype(float), (R_, C_)), shape=shape)
    inc8 = sp.csr_matrix(((m_ >= 8.0).astype(float), (R_, C_)), shape=shape)
    pv, sites = paleo_weights(rd, ptr, idx, rup, mag, L)
    vis = sp.csr_matrix((pv[keep], (R_, C_)), shape=shape)
    from plate_rate import rupture_matrix
    S = sp.csr_matrix(rupture_matrix(rd, nsub, P.get('slip_along', 'uniform') or 'uniform'))[sel] * 1e3   # mm per unit rate
    edges = np.round(np.arange(np.floor(mag.min() * 10) / 10, mag.max() + 0.1001, 0.1), 2)
    bm = np.clip(np.searchsorted(edges, mag[C_], side='right') - 1, 0, len(edges) - 2)
    MB = [sp.csr_matrix(((bm == b).astype(float), (R_, C_)), shape=shape) for b in range(len(edges) - 1)]

    def values(m):                                          # m: (n,) or (n, k) rates on the whole rupture set
        return dict(part=inc @ m, m7=inc7 @ m, m8=inc8 @ m, vis=vis @ m, slip=S @ m)
    z = np.load(vec)
    mean = np.asarray(z['mean'], float)
    V = values(mean)
    MBm = np.array([B @ mean for B in MB])                  # bins x subsections
    # the kept draws' 2.275-97.725%
    lohi = {}
    from cru_common import load_kept
    kept_files = sorted(f[:-len('_kept_index.npy')] for f in glob.glob(os.path.join(prior, 'main_rep*_kept_index.npy')))
    kc = np.asarray(P['keep_cols'])
    if kept_files:
        acc = {k: [] for k in ('part', 'vis', 'slip')}
        for kf in kept_files:
            K, _ = load_kept(kf)                            # sparse, one row per draw over the kept ruptures
            if K is None or K.shape[1] != len(kc):
                kept_files = []
                break
            Ko = sp.csr_matrix((K.data, kc[K.indices], K.indptr), shape=(K.shape[0], n))   # the rupture set's numbering
            for c0 in range(0, K.shape[0], 1000):
                v_ = values(Ko[c0:c0 + 1000].T.astype(float))
                for k_ in acc:
                    acc[k_].append(np.asarray(v_[k_].toarray() if hasattr(v_[k_], 'toarray') else v_[k_]))
        if kept_files:
            for k_, v_ in acc.items():
                v_ = np.concatenate(v_, axis=1)
                lohi[k_] = v_                               # subsections x kept draws, shaded by density
    refs = []
    for k, spec in enumerate(a.reference or []):
        path, label = spec, None
        if ':' in spec and '/' not in spec.rsplit(':', 1)[1]:
            path, label = spec.rsplit(':', 1)
        m_r = load_rates(os.path.abspath(os.path.expanduser(path)))
        if len(m_r) != n:
            raise SystemExit(f'--reference {path}: {len(m_r)} ruptures, the rupture set has {n}')
        lab = label or f'reference {k + 1} (…{os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))[-6:]})'
        refs.append(dict(v=values(m_r), label=lab, ink=REF_INKS[k]))
    t, tlo, thi, exact = slip_targets(P, cfg, nsub)
    from cfm_sliprates import load_cfm, rate_fixes
    from cru_common import CFM_CSV
    cfm = load_cfm(CFM_CSV)
    fx = rate_fixes(CFM_CSV)
    sr_pref = np.array([cfm.get(fx[p['ParentName'].strip()][0] if p['ParentName'].strip() in fx else p['ParentName'].strip(),
                                (np.nan,))[0] for p in props])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rows_out = []
    run_label = os.path.basename(os.path.dirname(prior))[-10:]
    for fault, subs in named.items():
        ss = subs_of[fault]
        if not len(ss):
            continue
        pts = np.concatenate([tr[s] for s in ss])
        lat0 = np.radians(pts[:, 1].mean())
        use_lat = (np.ptp(pts[:, 1]) > np.ptp(pts[:, 0]) * np.cos(lat0))
        ax_i = 1 if use_lat else 0
        x0 = np.array([tr[s][:, ax_i].min() for s in ss])
        x1 = np.array([tr[s][:, ax_i].max() for s in ss])
        order = np.argsort(0.5 * (x0 + x1))
        ss, x0, x1 = ss[order], x0[order], x1[order]
        r = rowof[ss]
        fig, axs = plt.subplots(4, 1, figsize=(9.0, 15.5), sharex=True, facecolor=SURFACE,
                                gridspec_kw=dict(height_ratios=[1, 1, 1, 1.15], hspace=0.12))

        def steps(ax, y, **kw):
            y = np.asarray(y, float)
            ax.hlines(y, x0, x1, **kw)

        def band(ax, D):
            """The kept draws' values (D: these subsections x draws) shaded by density in log10, each subsection's
            column on its own scale (its most common value darkest); draws outside the 1-99% of the fault's
            values, and zeros, are left out."""
            from matplotlib.collections import PolyCollection
            from matplotlib.colors import to_rgb
            pos = D[D > 0]
            if not len(pos):
                return
            lo, hi = np.percentile(pos, 1.0), np.percentile(pos, 99.0)
            hi = max(hi, lo * 1.01)
            ye = np.logspace(np.log10(lo), np.log10(hi), NB_DENS + 1)
            rgb = to_rgb(DENS)
            polys, cols = [], []
            for a_, b_, d_ in zip(x0, x1, D):
                h_, _ = np.histogram(d_[d_ > 0], ye)
                if h_.max() == 0:
                    continue
                for k_ in np.flatnonzero(h_):
                    polys.append([(a_, ye[k_]), (b_, ye[k_]), (b_, ye[k_ + 1]), (a_, ye[k_ + 1])])
                    cols.append((*rgb, 0.9 * h_[k_] / h_.max()))
            ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors='none', zorder=1))
            ax.update_datalim([(x0.min(), lo), (x1.max(), hi)])
            ax.autoscale_view()

        for ax in axs:
            ax.set_facecolor(SURFACE)
            for s_ in ('top', 'right'):
                ax.spines[s_].set_visible(False)
            ax.grid(True, which='major', color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8.5)
        # 1. participation
        ax = axs[0]
        if 'part' in lohi:
            band(ax, lohi['part'][r])
        steps(ax, V['part'][r], colors=SOL, lw=2.0, zorder=3)
        steps(ax, V['m7'][r], colors=SOL, lw=1.4, linestyles=(0, (4, 2)), zorder=3)
        steps(ax, V['m8'][r], colors=SOL, lw=1.4, linestyles=(0, (1, 1.5)), zorder=3)
        for rf in refs:
            steps(ax, rf['v']['part'][r], colors=rf['ink'], lw=1.0, linestyles=(0, (6, 2, 1, 2)), zorder=4)
        ax.set_yscale('log')
        ax.set_ylabel('participation rate (per year)', color=INK2, fontsize=9.5)
        h = [Line2D([], [], color=SOL, lw=2), Line2D([], [], color=SOL, lw=1.4, ls=(0, (4, 2))),
             Line2D([], [], color=SOL, lw=1.4, ls=(0, (1, 1.5)))] + ([Patch(color=DENS, alpha=0.6)] if 'part' in lohi else []) \
            + [Line2D([], [], color=rf['ink'], lw=1, ls=(0, (6, 2, 1, 2))) for rf in refs]
        l_ = ['posterior mean: all (supra-seis)', 'M ≥ 7', 'M ≥ 8'] + (['all: kept draws, shaded by density'] if 'part' in lohi else []) \
            + [f'{rf["label"]}: all' for rf in refs]
        ax.legend(h, l_, fontsize=7.6, frameon=True, facecolor=SURFACE, edgecolor='none', framealpha=0.85, ncol=2, loc='lower left', labelcolor=INK2)
        # 2. paleo-visible
        ax = axs[1]
        if 'vis' in lohi:
            band(ax, lohi['vis'][r])
        steps(ax, V['vis'][r], colors=SOL, lw=2.0, zorder=3)
        for rf in refs:
            steps(ax, rf['v']['vis'][r], colors=rf['ink'], lw=1.0, linestyles=(0, (6, 2, 1, 2)), zorder=4)
        n_site = 0
        for st in sites:
            if int(st['sectionIndex']) in set(ss.tolist()):
                xloc = st['dataLocation'][1 - ax_i] if 'dataLocation' in st else None
                xs = xloc if xloc is not None else 0.5 * (x0 + x1)[list(ss).index(int(st['sectionIndex']))]
                u = {q['type']: q for q in st['uncertainties']}
                b_ = st['bestEstimate']
                if 'TWO_SIGMA' in u:
                    ax.vlines(xs, u['TWO_SIGMA']['lowerBound'], u['TWO_SIGMA']['upperBound'], color=INK, lw=0.8, zorder=5)
                if 'ONE_SIGMA' in u:
                    ax.vlines(xs, u['ONE_SIGMA']['lowerBound'], u['ONE_SIGMA']['upperBound'], color=INK, lw=2.6, zorder=5)
                ax.plot(xs, b_, 'o', ms=5, mfc='white', mec=INK, mew=1.2, zorder=6)
                n_site += 1
        ax.set_yscale('log')
        ax.set_ylabel('paleo-visible participation rate (per year)', color=INK2, fontsize=9.5)
        h = [Line2D([], [], color=SOL, lw=2)] + ([Patch(color=DENS, alpha=0.6)] if 'vis' in lohi else []) \
            + [Line2D([], [], color=rf['ink'], lw=1, ls=(0, (6, 2, 1, 2))) for rf in refs] \
            + ([Line2D([], [], marker='o', mfc='white', mec=INK, ls='-', color=INK, lw=2)] if n_site else [])
        l_ = ['posterior mean (UCERF3 visibility)'] + (['kept draws, shaded by density'] if 'vis' in lohi else []) \
            + [rf['label'] for rf in refs] + (['paleo sites: best estimate, 1σ (thick), 2σ (thin)'] if n_site else [])
        ax.legend(h, l_, fontsize=7.6, frameon=True, facecolor=SURFACE, edgecolor='none', framealpha=0.85, ncol=2, loc='lower left', labelcolor=INK2)
        # 3. slip rate
        ax = axs[2]
        steps(ax, np.where(np.isfinite(tlo[ss]), np.maximum(tlo[ss], 1e-6), np.nan), colors=TGT_RANGE, lw=1.2,
              linestyles=TGT_RANGE_LS, zorder=2)
        steps(ax, thi[ss], colors=TGT_RANGE, lw=1.2, linestyles=TGT_RANGE_LS, zorder=2)
        if 'slip' in lohi:
            band(ax, lohi['slip'][r])
        steps(ax, t[ss], colors=TGT, lw=1.6, zorder=2)
        steps(ax, V['slip'][r], colors=SOL, lw=2.0, zorder=3)
        for rf in refs:
            steps(ax, rf['v']['slip'][r], colors=rf['ink'], lw=1.0, linestyles=(0, (6, 2, 1, 2)), zorder=4)
        ax.set_yscale('log')
        ax.set_ylabel('slip rate (mm/yr)', color=INK2, fontsize=9.5)
        what = 'exact corrected density' if exact else 'split normal'
        h = [Line2D([], [], color=TGT, lw=1.6), Line2D([], [], color=TGT_RANGE, lw=1.2, ls=TGT_RANGE_LS),
             Line2D([], [], color=SOL, lw=2)] + ([Patch(color=DENS, alpha=0.6)] if 'slip' in lohi else []) \
            + [Line2D([], [], color=rf['ink'], lw=1, ls=(0, (6, 2, 1, 2))) for rf in refs]
        l_ = ['target (× slip-rate factor)', f'target 2.275% and 97.725% ({what})', 'posterior mean'] \
            + (['kept draws, shaded by density'] if 'slip' in lohi else []) + [rf['label'] for rf in refs]
        ax.legend(h, l_, fontsize=7.6, frameon=True, facecolor=SURFACE, edgecolor='none', framealpha=0.85, ncol=2, loc='lower left', labelcolor=INK2)
        # 4. magnitudes
        ax = axs[3]
        M = MBm[:, r]
        have = np.flatnonzero(M.sum(1) > 0)
        mlo, mhi = (edges[have[0]], edges[have[-1] + 1]) if len(have) else (edges[0], edges[-1])
        pos_ = M[M > 0]
        norm = LogNorm(vmin=max(pos_.max() * 1e-6, pos_.min()), vmax=pos_.max()) if len(pos_) else None
        for j, (a_, b_) in enumerate(zip(x0, x1)):
            for b in have:
                if M[b, j] > 0:
                    ax.fill_between([a_, b_], [edges[b]] * 2, [edges[b + 1]] * 2, color=HEAT(norm(M[b, j])), lw=0)
        mids = 0.5 * (edges[:-1] + edges[1:])
        med = []
        for j in range(len(ss)):
            w = M[:, j]
            if w.sum() > 0:
                cw = np.cumsum(w) / w.sum()
                med.append(np.interp(0.5, cw, mids))
            else:
                med.append(np.nan)
        steps(ax, med, colors='#b35806', lw=1.6, zorder=4)
        ax.set_ylim(mlo, mhi)
        ax.set_ylabel('magnitude of the ruptures through it', color=INK2, fontsize=9.5)
        if norm is not None:
            sm = plt.cm.ScalarMappable(cmap=HEAT, norm=norm)
            cb = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2, fraction=0.05, aspect=40)
            cb.set_label('posterior mean participation rate in each 0.1-magnitude bin (per year)', color=INK2, fontsize=8.5)
            cb.ax.tick_params(labelsize=7.5, colors=MUTED, labelcolor=INK2)
        ax.legend([Line2D([], [], color='#b35806', lw=1.6)], ['rate-weighted median magnitude'], fontsize=7.6, frameon=True, facecolor=SURFACE, edgecolor='none', framealpha=0.85,
                  loc='lower left', labelcolor=INK2)
        ax.set_xlabel('latitude (degrees)' if use_lat else 'longitude (degrees)', color=INK2, fontsize=9.5)
        axs[0].set_title(f'{fault}: along strike', loc='left', color=INK, fontsize=12.5, fontweight='bold', pad=22)
        axs[0].text(0.0, 1.02, f'{len(ss)} subsections in geographical order; run …{os.path.basename(os.path.dirname(prior))[-24:]}',
                    transform=axs[0].transAxes, color=INK2, fontsize=7.8, ha='left', va='bottom')
        safe = re.sub(r'[^A-Za-z0-9]+', '_', fault).strip('_')
        fig.savefig(os.path.join(outdir, safe + '.png'), dpi=130, facecolor=SURFACE, bbox_inches='tight')
        plt.close(fig)
        for j, s in enumerate(ss):
            rows_out.append([fault, int(s), props[s]['FaultName'], f'{x0[j]:.5f}', f'{x1[j]:.5f}', 'lat' if use_lat else 'lon',
                             *(f'{v:.6g}' for v in (V['part'][r[j]], V['m7'][r[j]], V['m8'][r[j]], V['vis'][r[j]],
                                                     V['slip'][r[j]], t[s], tlo[s], thi[s], sr_pref[s], med[j]))])
    with open(os.path.join(outdir, 'along_strike.csv'), 'w', newline='') as f_:
        w = csv.writer(f_)
        w.writerow(['fault', 'subsection', 'name', 'x0', 'x1', 'axis', 'participation', 'participation_M7', 'participation_M8',
                    'paleo_visible', 'slip_rate_mm_yr', 'target_mm_yr', 'target_p02_275', 'target_p97_725', 'cfm_sr_pref',
                    'median_magnitude'])
        w.writerows(rows_out)
    print(f'wrote {len(named)} along-strike figures and along_strike.csv to {outdir}')


if __name__ == '__main__':
    main()

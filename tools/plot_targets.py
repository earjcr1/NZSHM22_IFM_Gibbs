#!/usr/bin/env python3
"""The targets: the inputs' Gaussians in rate (NZSHM22's) against the distributions this run scores the data by.

    python plot_targets.py [--problem cru_clean.npz] [--ruptures RUPTURE_SET] [--config workflow_config.json]
                           [--outdir DIR] [--only slip,paleo,mfd]

Three figures, written next to --problem (or in --outdir), each with its table view (.txt):

fig_targets_slip.png: each fault section's slip rate, in mm/yr, linear and log panels as in fig_slip_rates.png. The
  x axis is the inputs' target, NZSHM22's Gaussian: 0.8 SlipRate from the rupture set's fault_sections.geojson, its sd
  2 SlipRateStdDev (--old-sd K: K SlipRateStdDev) (or 0.8 SlipRate where that is 0; the inputs' 0.8 on the target only). At each section, a wide grey
  bar is the CFM's SR_min to SR_max (each subsection its parent fault's line of NZ_CFM_v1_0.csv), times the slip-rate
  factor as the run's distributions are (--grey gaussian: that Gaussian's 2.3-97.7% range instead); a coloured bar is the 2.3-97.7% range of the distribution
  this run scores the section by, read from the cleaned problem (with the CFM's slip rates, a split normal with its own
  sd each side, timeframe-corrected with CFM_TIMEFRAME, the slip-rate factor on the whole distribution: its 2.275-97.725%
  points; 15.9-84.1% until 2026-11-25), with a dot at its mode (with MISFIT_SLIP 'exact', the exact density's own,
  not the split normal's, which only anchors each row's weighting). Colours as in fig_slip_rates.png (cru_common.neighbour_colours along the targets).
fig_targets_paleo.png: each paleo site's rate per year, the same two panels. The x axis is bestEstimate, the inputs'
  Gaussian's mean, its sd the site's ONE_SIGMA stdDev (grey: +-2 sd, 2.3-97.7%); the coloured bar is the site's
  recurrence-interval PDF (paleo_constraint_data.json: its TWO_SIGMA bounds, the 2.275% and 97.725% points, as rates), which MISFIT_PALEO 'years'
  scores the site by. Both are centred on bestEstimate (the PDF's mode in years is at 1/bestEstimate), so the
  difference is the spread.
fig_targets_mfd.png: the magnitude bins (MFD_GR's figure): per bin and cumulative, from M6.5, the total GR and its
  partitioning (mfd_gr.py), cru_D's targets with NZSHM22's sigma, and MFD_GR's bin PDFs (16-84% and 2.3-97.7%), with
  the run's MFD_GR knobs or, if the run did not use MFD_GR, the defaults (RECONSIDER_MAGUNCERT on) for comparison; the
  cumulative panel's MFD_GR band adds the bins drawn independently, as the sampler scores them (Monte Carlo). Needs
  an inversion solution's files (--ruptures, or the run's rupture set).

Ranges below 0 on a log panel stop at the axis (a Gaussian's 16% point is below 0 wherever its sd exceeds its mean).
"""
import argparse
import os

import numpy as np

from cru_common import (MARK_BLUES_CYCLE, find_rupture_set, load_problem, neighbour_colours, run_config, slip_sections)

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
OLD = '#b9b8b1'                       # the inputs' Gaussians: grey, wide
BLUE, ORANGE = '#2a78d6', '#eb6834'


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_color(AXIS)
        ax.spines[s_].set_linewidth(0.6)
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8.5)


def pair_figure(x, old_lo, old_hi, new_c, new_lo, new_hi, colours, unit, title, sub, out, what, item='section',
                old_label='inputs (NZSHM22): Gaussian, 2.3–97.7%', rng='2.3–97.7%', centre='this run: mode'):
    """Two panels, linear and log: per item, the inputs' range (grey, wide) and the run's (coloured), at the
    inputs' target x; a dot at the run's centre; the 1:1 line."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 6.0), facecolor=SURFACE)
    pos = np.concatenate([v[v > 0] for v in (x, new_c)])  # the log axes follow the targets and centres; ranges reaching
    lo_log, hi = pos.min() / 3.0, max(old_hi.max(), new_hi.max(), x.max()) * 1.08   # below them stop at the axis
    for ax, log in zip(axs, (False, True)):
        _axes_style(ax)
        bottom = lo_log if log else 0.0
        clip = lambda v: np.maximum(v, bottom)
        ax.vlines(x, clip(old_lo), old_hi, color=OLD, lw=4.2, alpha=0.55, zorder=2, capstyle='butt', label=old_label)
        ax.vlines(x, clip(new_lo), new_hi, colors=colours, lw=1.3, zorder=3, label=f'this run: {what}, {rng}')
        ax.scatter(x, new_c, s=14, c=colours, edgecolors='#7d7c76', linewidths=0.4, zorder=4, label=centre)
        ax.plot([bottom if log else 0, hi], [bottom if log else 0, hi], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1,
                label='1:1 (the inputs’ target)')
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlim(bottom, hi)
        ax.set_ylim(bottom, hi)
        ax.set_xlabel(f'inputs’ target ({unit})', color=INK2, fontsize=9.5)
        ax.set_ylabel(f'{unit}', color=INK2, fontsize=9.5)
        ax.set_title('log' if log else 'linear', color=INK, fontsize=10.5, loc='left', fontweight='bold')
    h = [Line2D([], [], color=OLD, lw=5, alpha=0.7), Line2D([], [], color=INK2, lw=1.3),
         Line2D([], [], marker='o', ls='', mfc=MUTED, mec='#7d7c76', ms=5), Line2D([], [], color=INK2, lw=0.8, ls=(0, (4, 3)))]
    l_ = [old_label, f'this run: {what}, {rng} (colour: {item}, as in the fit figures)',
          centre, '1:1 (the inputs\u2019 target)']
    fig.suptitle(title, x=0.06, y=0.985, ha='left', color=INK, fontsize=12, fontweight='bold')
    sub = '\n'.join(textwrap.fill(p_, 150) for p_ in sub.split('\n'))
    fig.text(0.06, 0.94, sub, color=INK2, fontsize=8.4, ha='left', va='top', linespacing=1.35)
    fig.legend(h, l_, loc='upper center', ncol=2, frameon=False, fontsize=8.3, labelcolor=INK2, bbox_to_anchor=(0.5, 0.845))
    fig.tight_layout(rect=(0, 0, 1, 0.77))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def slip_figure(P, rd, cfg, outdir, slip_block='S', old_k=2.0, grey='cfm'):
    """fig_targets_slip.png and its table (see the module docstring)."""
    from cfm_sliprates import load_subsections
    from plate_rate import _sections, check_problem, rupture_matrix
    from timeframe_sliprates import sn_quantile
    rows = np.flatnonzero(P['block'] == list(P['block_names']).index(slip_block))
    orig = np.asarray(P['keep_rows'])[rows]
    scale = P['row_scale'][rows]
    coef = check_problem(P, rupture_matrix(rd, len(_sections(rd)), P.get('slip_along', 'uniform')), slip_block)
    u = scale / coef * 1e3                               # each row's sd (sigma_ref) in mm/yr
    t = P['d'][rows] / (scale / u)                       # each row's target (the mode), mm/yr
    s_lo = u / np.sqrt(P['split_lo2'][rows]) if P.get('split_lo2') is not None else u
    s_hi = u / np.sqrt(P['split_hi2'][rows]) if P.get('split_hi2') is not None else u
    sec = P['section'][rows]
    if (sec < 0).any():
        sec = slip_sections(P['G'].tocsr()[rows], P['d'][rows], np.arange(len(rows)), None, 'none')[0]
    S = int(sec.max()) + 1
    ks = np.bincount(sec, minlength=S)
    mean_s = lambda v: np.bincount(sec, weights=v, minlength=S) / ks
    sub = load_subsections(rd)
    if orig.max() >= len(sub['rate']):
        raise SystemExit('the slip-rate rows are not the rupture set\'s subsections (raw rows 0 .. subsections - 1)')
    f_in = 0.8                                           # the inputs' factor on the targets (NZSHM22; checked by cleaning)
    rate, sd = sub['rate'][orig], sub['sd'][orig]
    o_m = f_in * rate                                    # the inputs' Gaussian: mean 0.8 SlipRate, sd 2 SD (0.8 SlipRate if 0);
    o_s = np.where(sd > 0, old_k * sd, f_in * rate)      # --old-sd K: K SD in its place (1: SlipRateStdDev itself as 1 sd)
    ts, sl, sh, om, os_ = (mean_s(v) for v in (t, s_lo, s_hi, o_m, o_s))
    fac = cfg.get('slip_rate_factor', 0.8)
    if grey == 'cfm':                                    # the grey bar: the CFM's SR_min to SR_max (each subsection its
        from cfm_sliprates import load_cfm               # parent fault's line), times the slip-rate factor as the run's are
        from cru_common import CFM_CSV                   # (2026-11-25, the user's request; the Gaussian before)
        cfm_ = load_cfm(cfg.get('cfm_file') or CFM_CSV)
        par_ = [sub['parent'][j] for j in orig]
        miss_ = sorted({q for q in par_ if q not in cfm_})
        if miss_:
            raise ValueError(f'parent faults not in the CFM: {", ".join(miss_[:5])}')
        g_lo = mean_s(np.array([fac * cfm_[q][1] for q in par_]))
        g_hi = mean_s(np.array([fac * cfm_[q][2] for q in par_]))
    else:
        g_lo, g_hi = om - 2 * os_, om + 2 * os_
    n16, n84 = sn_quantile(ts, sl, sh, 0.02275), sn_quantile(ts, sl, sh, 0.97725)   # 2.275-97.725% (16-84% until 2026-11-25)
    cen = ts                                             # the dot: the split normal's mode
    exact = str(cfg.get('misfit_slip')) == 'exact' and P.get('slip_pdf_key') is not None
    if exact:                                            # MISFIT_SLIP 'exact': the exact corrected density's points and its
        from cru_common import slip_pdf_modes, slip_pdf_quantiles   # own mode (the split normal's only anchors each
        q_ = slip_pdf_quantiles(P, rows, [0.02275, 0.97725])         # row's weighting: for very lopsided faults it can
        n16, n84 = mean_s(q_[0]), mean_s(q_[1])                      # sit outside the density's range, 2026-11-25)
        cen = mean_s(slip_pdf_modes(P, rows))
    col = neighbour_colours(ts, MARK_BLUES_CYCLE)
    cfm = bool(cfg.get('cfm_sliprates', True))
    tf = bool(cfg.get('cfm_timeframe'))
    what = ('exact timeframe-corrected density' if exact else
            ('CFM split normal' + (', timeframe-corrected' if tf else '')) if cfm else 'the inputs’ Gaussian, rescaled')
    med = lambda v: float(np.median(v))
    gw = g_hi - g_lo
    grey_txt = (f'Grey: the CFM’s SR_min to SR_max, times the slip-rate factor {fac:g}' if grey == 'cfm' else
                f'Grey: NZSHM22’s Gaussian, mean 0.8 SlipRate, sd {old_k:g} SlipRateStdDev (the 0.8 on the mean '
                f'only' + ('' if old_k == 2 else '; NZSHM22’s inputs use 2') + ')')
    sub_txt = (f'{S} fault sections at the inputs’ target (0.8 SlipRate). {grey_txt}. Coloured: this run’s {what}, the '
               f'slip-rate factor {fac:g} on the whole distribution (MISFIT_SLIP {cfg.get("misfit_slip", "linear")!r}). '
               f'Median ratios: mode over the inputs’ target {med(cen[om > 0] / om[om > 0]):.2f}; 2.3–97.7% width over the '
               + (f'CFM’s range {med((n84 - n16)[gw > 0] / gw[gw > 0]):.2f}' if grey == 'cfm' else
                  f'Gaussian’s {med((n84 - n16)[om > 0] / (4 * os_[om > 0])):.2f}') + '. Ranges below 0 stop at the log panel’s axis.')
    out = os.path.join(outdir, 'fig_targets_slip.png')
    pair_figure(om, g_lo, g_hi, cen, n16, n84, col, 'mm/yr',
                'Slip-rate targets: ' + ('the CFM’s ranges' if grey == 'cfm' else 'the inputs’ Gaussians') + ' and this run’s',
                sub_txt, out, what, old_label=(f'CFM: SR_min to SR_max, × {fac:g}' if grey == 'cfm'
                                               else 'inputs (NZSHM22): Gaussian, 2.3–97.7%'))
    with open(out[:-4] + '.txt', 'w') as f_:
        f_.write(f'slip-rate targets per section (mm/yr): grey {grey}' + (f' (the CFM\'s SR_min to SR_max, times {fac:g})' if grey == 'cfm' else '')
                 + f'; the inputs\' Gaussian (mean, sd; from the rupture set\'s fault_sections.geojson, '
                 f'mean 0.8 SlipRate, sd {old_k:g} SlipRateStdDev); this run\'s ({what}: its mode, the split normal\'s mode, sd '
                 f'below, sd above, and the 2.275% and 97.725% points)\n')
        f_.write(f'{"section":>7s} {"rows":>9s} {"in mean":>8s} {"in sd":>7s} {"grey lo":>8s} {"grey hi":>8s} {"mode":>8s} '
                 f'{"sn mode":>8s} {"sd lo":>7s} {"sd hi":>7s} {"2.3%":>8s} {"97.7%":>8s}\n')
        for s in range(S):
            r_ = orig[sec == s]
            f_.write(f'{s:7d} {f"{r_.min()}-{r_.max()}" if len(r_) > 1 else str(r_[0]):>9s} {om[s]:8.3f} {os_[s]:7.3f} '
                     f'{g_lo[s]:8.3f} {g_hi[s]:8.3f} {cen[s]:8.3f} '
                     f'{ts[s]:8.3f} {sl[s]:7.3f} {sh[s]:7.3f} {n16[s]:8.3f} {n84[s]:8.3f}\n')
    return out


def paleo_figure(P, rd, cfg, outdir, slip_block='S', bins_block='MFD'):
    """fig_targets_paleo.png and its table (see the module docstring)."""
    from paleo_misfit import load_sites
    sites = load_sites(rd, verbose=False)
    best = np.array([s_['best'] for s_ in sites])
    sd = np.array([s_['sd'] for s_ in sites])
    q = np.array([s_['q'] for s_ in sites])              # 2.275, 15.866, 84.134, 97.725% rates
    col = neighbour_colours(best, MARK_BLUES_CYCLE)
    pm = cfg.get('misfit_paleo') or {'years': 'years', 'gaussian': 'rate'}.get(cfg.get('paleo_misfit'), '?')
    med = lambda v: float(np.median(v))
    sub_txt = (f'{len(sites)} paleo sites (paleo_constraint_data.json). Grey: the Gaussian in rate the inputs give, bestEstimate '
               f'± twice its ONE_SIGMA sd (2.3–97.7%). Coloured: the site’s recurrence-interval PDF, its TWO_SIGMA bounds '
               f'(2.3–97.7%) as rates, which MISFIT_PALEO ’years’ scores it by (this run: {pm!r}). Both centred on bestEstimate. '
               f'Median ratios, PDF / Gaussian: lower half-width {med((best - q[:, 0]) / (2 * sd)):.2f}, upper '
               f'{med((q[:, 3] - best) / (2 * sd)):.2f}.')
    out = os.path.join(outdir, 'fig_targets_paleo.png')
    pair_figure(best, best - 2 * sd, best + 2 * sd, best, q[:, 0], q[:, 3], col, 'events per year',
                'Paleo-rate targets: the inputs’ Gaussians and the sites’ own PDFs', sub_txt, out,
                'recurrence-interval PDF', item='site', centre='bestEstimate (the PDF’s mode in years)')
    with open(out[:-4] + '.txt', 'w') as f_:
        f_.write('paleo-rate targets per site (per year): the inputs\' Gaussian (bestEstimate, sd) and the site\'s PDF (its 2.3, '
                 '15.9, 84.1 and 97.7% points)\n')
        for s_ in sites:
            f_.write(f'{s_["name"][:40]:40s} {s_["best"]:10.4e} {s_["sd"]:10.4e} ' + ' '.join(f'{v:10.4e}' for v in s_['q']) + '\n')
    return out


def mfd_figure(rd, cfg, outdir, n_mc=200000, seed=1):
    """fig_targets_mfd.png and its table (see the module docstring): MFD_GR's bin PDFs against cru_D's targets."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mfd_gr as M
    used = bool(cfg.get('mfd_gr'))
    spec = (str(cfg['mfd_gr']) if used else '0.823,0.959,1.089,2.65,3.40,4.57,0.66,1.41,'
            + ','.join(f'{x_:.10g}' for x_ in M.MAGUNCERT))
    parts = M.load_parts(rd)
    K = M.parse_knobs(spec)
    B = M.bin_pdfs(parts, K)
    X, R = M.X, B['R']
    js = sorted(B['bins'])
    on = np.zeros(len(X), bool)
    on[js] = True
    bl, bm, bh = K['b']
    sl, sh = K['s']
    mk = K.get('m')
    m_mid = mk[1] if mk else 1.0
    Nc = float(M.n_of_b(K, bm)) * m_mid
    c = M.components(parts, Nc, bm)
    sig_n = M.nzshm22_sigma(R, parts)
    qi = np.exp(np.array([B['bins'][j]['q'] for j in js]).T)          # each bin's 2.3, 15.9, 50, 84.1, 97.7% points
    rng = np.random.default_rng(seed)                     # the cumulative as scored: each bin drawn from its own PDF
    Ti = np.zeros((n_mc, len(X)))
    for j in js:
        bj = B['bins'][j]
        G = np.cumsum(bj['g'])
        G /= G[-1]
        Ti[:, j] = np.exp(np.interp(rng.random(n_mc), G, bj['u']))
    PQ = [2.275, 15.865, 50, 84.135, 97.725]
    qc = np.percentile(np.cumsum(Ti[:, ::-1], 1)[:, ::-1], PQ, axis=0)
    cum = lambda y: np.cumsum(np.where(on, y, 0)[::-1])[::-1]
    Rc, sd_cn = cum(R), np.sqrt(cum(sig_n ** 2))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), facecolor=SURFACE)
    for ax, cumul in zip(axs, (False, True)):
        _axes_style(ax)
        xs = X - 0.05 if cumul else X
        allb = (lambda y: np.cumsum(y[::-1])[::-1]) if cumul else (lambda y: y)
        for y, ls, lab in ((c['total'], '-', f'total GR (N {Nc:.2f}, b {bm:g})'), (c['off'], ':', 'truly off-fault'),
                           (c['sub'], '-.', 'sub-seismogenic on-fault')):
            yy = allb(np.where(y > 1e-19, y, 0 if cumul else np.nan))
            if cumul:
                yy = np.where(yy > 1e-12, yy, np.nan)
            ax.plot(xs, yy, color=MUTED, lw=1.6, ls=ls, label=lab, zorder=2)
        xo = xs[on]
        if not cumul:
            ax.fill_between(xo, qi[0], qi[4], color=ORANGE, alpha=0.14, lw=0, label='MFD_GR bin PDF 2.3–97.7%', zorder=3)
            ax.fill_between(xo, qi[1], qi[3], color=ORANGE, alpha=0.30, lw=0, label='MFD_GR bin PDF 16–84%', zorder=3)
            ax.fill_between(xo, np.maximum(R[on] - sig_n[on], 1e-6), R[on] + sig_n[on], color=BLUE, alpha=0.45, lw=0,
                            label='cru_D target ±1σ (NZSHM22 recipe)', zorder=4)
            ax.plot(xo, R[on], color=INK, lw=2, marker='o', ms=4.5, mec=SURFACE, mew=1, label='supra-seismogenic target', zorder=5)
        else:
            ax.fill_between(xo, qc[1][on], qc[3][on], color=ORANGE, alpha=0.30, lw=0,
                            label='MFD_GR 16–84% as scored (bins independent)', zorder=3)
            ax.fill_between(xo, np.maximum(Rc[on] - sd_cn[on], 1e-6), Rc[on] + sd_cn[on], color=BLUE, alpha=0.45, lw=0,
                            label='cru_D ±1σ (bins independent)', zorder=4)
            ax.plot(xo, Rc[on], color=INK, lw=2, marker='o', ms=4.5, mec=SURFACE, mew=1, label='supra-seismogenic target', zorder=5)
        ax.set_yscale('log')
        ax.set_xlim(6.45, X[on].max() + 0.1)
        top = (max(qi[4].max(), c['total'][X > 6.45].max()) if not cumul else max(qc[4][on].max(), allb(c['total'])[X - 0.05 > 6.45].max()))
        ax.set_ylim(min(qi[0].min(), Rc[on].min()) / 2, top * 1.3)
        ax.axvline(6.8, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
        ax.set_xlabel('magnitude' + (' M (lower edge: rate of M or larger)' if cumul else ' (bin centre, 0.1 bins)'), color=INK2, fontsize=10)
        ax.set_ylabel('cumulative rate N(≥M) per year' if cumul else 'rate per year in the bin', color=INK2, fontsize=10)
        ax.set_title('Cumulative' if cumul else 'Per bin', color=INK, fontsize=11, loc='left', fontweight='bold')
        ax.legend(fontsize=8.1, frameon=False, loc='upper right', labelcolor=INK2)
    mtxt = (f', magnitude factor m {"/".join(f"{x_:.4g}" for x_ in mk)} (three branches, equal weights)' if mk else '')
    fig.suptitle('The magnitude bins: NZSHM22’s targets (cru_D) and MFD_GR’s PDFs', x=0.06, ha='left', color=INK,
                 fontsize=13, fontweight='bold')
    import textwrap
    fig.text(0.06, 0.93, '\n'.join(textwrap.fill(p_, 175) for p_ in (f'MFD_GR{"" if used else " (not used in this run: its defaults, for comparison)"}: b {bl:g}/{bm:g}/{bh:g} '
                           f'(10/50/90%), N {K["n"][0]:g}/{K["n"][1]:.2f}/{K["n"][2]:g} at each (correlated), N scaling '
                           f'{sl:g}–{sh:g} (±2σ){mtxt}.\nCentral target: b’s mean, its N{", m’s middle value" if mk else ""}, s = 1 '
                           f'(N {Nc:.2f}). Lower band edges at or below 0 are cut at the axis.').split('\n')),
             color=INK2, fontsize=8.8, ha='left', va='top', linespacing=1.35)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = os.path.join(outdir, 'fig_targets_mfd.png')
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    with open(out[:-4] + '.txt', 'w') as f_:
        f_.write(f'MFD_GR {spec} ({"used" if used else "not used in this run; its defaults"}); per bin: cru_D\'s target and sigma, '
                 'MFD_GR\'s target (the same knobs\' centre) and its 2.3, 15.9, 50, 84.1, 97.7% points; cumulative from the bin\'s lower '
                 'edge: the target and the 15.9-84.1% of the bins drawn independently\n')
        for k_, j in enumerate(js):
            f_.write(f'{X[j]:5.2f} {R[j]:10.4e} {sig_n[j]:10.4e} ' + ' '.join(f'{v:10.4e}' for v in qi[:, k_])
                     + f' | {Rc[j]:10.4e} {qc[1][j]:10.4e} {qc[3][j]:10.4e}\n')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem (default cru_clean.npz)')
    ap.add_argument('--ruptures', default=None, help='the OpenSHA rupture set or inversion solution (default: the run\'s, '
                    'from its workflow_config.json)')
    ap.add_argument('--config', default=None, help='the run\'s workflow_config.json (default: the one next to --problem)')
    ap.add_argument('--outdir', default=None, help='where the figures go (default: next to --problem)')
    ap.add_argument('--only', default='slip,paleo,mfd', help='which figures (default slip,paleo,mfd)')
    ap.add_argument('--old-sd', type=float, default=2.0, metavar='K',
                    help='the slip figure\'s Gaussian (with --grey gaussian, and in its table): sd K SlipRateStdDev (default 2, as NZSHM22\'s inversion inputs have it; 1: SlipRateStdDev itself as 1 sd)')
    ap.add_argument('--grey', default='cfm', choices=['cfm', 'gaussian'],
                    help='the slip figure\'s grey bars: the CFM\'s SR_min to SR_max times the slip-rate factor (cfm, the default '
                         'since 2026-11-25) or the inputs\' Gaussian\'s 2.3-97.7%% (gaussian)')
    a = ap.parse_args()
    cfg = run_config(a.config, a.problem) or {}
    rd = find_rupture_set(a.ruptures or cfg.get('rupture_dir') or '')
    if rd is None:
        raise SystemExit('no OpenSHA rupture set (give --ruptures)')
    outdir = a.outdir or os.path.dirname(os.path.abspath(a.problem))
    want = set(a.only.split(','))
    P = load_problem(a.problem) if want & {'slip', 'paleo'} else None
    for k, fn in (('slip', lambda: slip_figure(P, rd, cfg, outdir, old_k=a.old_sd, grey=a.grey)), ('paleo', lambda: paleo_figure(P, rd, cfg, outdir)),
                  ('mfd', lambda: mfd_figure(rd, cfg, outdir))):
        if k in want:
            try:
                print('wrote', fn())
            except (FileNotFoundError, ValueError) as e:      # e.g. MFD: a rupture set without an inversion solution's files
                print(f'{k}: not drawn ({e})')


if __name__ == '__main__':
    main()

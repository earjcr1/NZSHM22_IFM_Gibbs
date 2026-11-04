#!/usr/bin/env python3
"""A run's summary as a PDF: about a paragraph per step of the workflow, with the run's figures.

    python make_summary_pdf.py RUN_DIR [--prior spikeslab] [--out FILE] [--title TEXT]

RUN_DIR is a notebook run's folder (the notebook's RUN_DIR: cru_clean.npz, prelim_summary.txt and prelim.png at the
top, the best fit in nnls/, the runs, reports and figures in <prior>/). Every number in the text is read from the run's
own reports: workflow_config.json, prelim_summary.txt, each replicate's <run>_log.txt and <run>.npz,
convergence_report.txt, block_fit_report.txt, tables_report.txt, select_report.txt, solution_report.txt and the figures'
tables (fig_*.txt). A number a report does not give (an older run, or a step that was skipped) is shown as "n/a", and
the script lists those at the end; a figure that is missing is left out. Sentences that interpret a result (why chi2
sits below the fit check, whether the paleo rates are underpredicted, ...) are chosen from the numbers, not written in.

Sections: an overview; inputs and configuration (the settings table); cleaning and diagnostics (prelim.png); the best
fit (nnls/'s three fit figures); the model (prior, likelihood, and the constraints and soft terms the run used: the band
or positivity, the soft band, the monotonic MFD, the plate rate, the summed slip rates); sampling; convergence (per
replicate); the fit to the data (the row-block table, and the terms recorded apart from chi2: the soft band's and the
summed slip rates'); posterior tables and representative models; the posterior's fit, data type by data
type (the three fit figures); faults one by one (plot_slip_sections.py's figures, on landscape pages); the OpenSHA
output; points to keep in mind. Each figure carries a numbered caption and its file name.

Writes FILE (default: RUN_DIR/workflow_summary.pdf). Needs reportlab (pip install reportlab), which the rest of the
workflow does not, and the DejaVu Sans fonts that ship with matplotlib (for chi, sigma and the like).
"""
import argparse
import glob
import json
import os
import re

import numpy as np

MISSING = []


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        MISSING.append(os.path.basename(path))
        return ''


def grab(text, pattern, what, conv=float, groups=None, flags=re.S | re.M):
    """The pattern's group 1 (or the given groups) converted, or None, noting what is missing."""
    m = re.search(pattern, text, flags)
    if not m:
        MISSING.append(what)
        return None if groups is None else [None] * len(groups)
    if groups is None:
        return conv(m.group(1).replace(',', ''))
    return [conv(m.group(g).replace(',', '')) for g in groups]


def fmt(v, spec='{:,.0f}'):
    return 'n/a' if v is None else spec.format(v).replace('-', '\u2212')    # a true minus sign


def rng(v, spec='{:,.0f}'):
    """The range of the values v, as 'low\u2013high', or one value if both print the same."""
    if not v:
        return 'n/a'
    lo, hi = fmt(min(v), spec), fmt(max(v), spec)
    return lo if lo == hi else f'{lo}\u2013{hi}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir')
    ap.add_argument('--prior', default='spikeslab', help="the prior's folder in RUN_DIR (default spikeslab)")
    ap.add_argument('--out', default=None, help='the PDF (default RUN_DIR/workflow_summary.pdf)')
    ap.add_argument('--title', default='Posterior sampling of NZSHM22 rupture rates')
    a = ap.parse_args()
    try:
        from PIL import Image
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (BaseDocTemplate, Frame, Image as RLImage, KeepTogether, NextPageTemplate,
                                        PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
    except ImportError as e:
        raise SystemExit(f'{e}: this script needs reportlab (pip install reportlab) and Pillow')
    import matplotlib

    RUN = os.path.abspath(os.path.expanduser(a.run_dir))
    PR = os.path.join(RUN, a.prior)
    out = a.out or os.path.join(RUN, 'workflow_summary.pdf')
    run_name = os.path.basename(RUN.rstrip('/'))

    # ------------------------------------------------------------------ what the reports say
    cfg = {}
    for f in (os.path.join(PR, 'workflow_config.json'), os.path.join(RUN, 'workflow_config.json')):
        if os.path.exists(f):
            with open(f) as fh:
                cfg = json.load(fh)
            break
    else:
        MISSING.append('workflow_config.json')
    st = cfg.get('settings', {})
    band = cfg.get('band')
    positive = band == 'positive'                      # BAND = 'positive': every prediction above zero
    band = band if isinstance(band, (int, float)) and not isinstance(band, bool) else None   # the band's factor, or None
    mono = bool(cfg.get('mfd_monotonic'))
    plate = cfg.get('plate_rate')                      # None, an upper limit, or (low, high)
    plate = (None if plate is None else (None, float(plate)) if isinstance(plate, (int, float))
             else (float(plate[0]), float(plate[1])))
    ssum = cfg.get('slip_sum')                         # True or False ('all' or 'between' in runs before CROSS_SECTIONS)
    xsec = cfg.get('cross_sections') or (ssum if isinstance(ssum, str) else 'all')
    xsec_txt = 'over all of New Zealand' if xsec == 'all' else 'between the Alpine and Cook Strait edges'
    plate_txt = None if plate is None else (f'at most {plate[1]:g}' if plate[0] is None else
                                            f'between {plate[0]:g} and {plate[1]:g}')
    hard = bool(band or positive or mono or plate)     # any hard constraint, kept in every recorded draw
    tau = cfg.get('band_soft') or None                 # the soft band's width in decades, or None
    grp = 'slip_band' in cfg                           # since November 2026: a band per row group (slip rates, paleo)
    g_band = {'slip-rate': cfg.get('slip_band'), 'paleo': cfg.get('paleo_band')} if grp else {}
    g_soft = {'slip-rate': cfg.get('slip_band_soft'), 'paleo': cfg.get('paleo_band_soft')} if grp else {}
    pmis = cfg.get('paleo_misfit') or 'gaussian'       # the paleo rows' misfit
    mmis = cfg.get('mfd_misfit') or 'gaussian'         # the MFD bins' misfit
    cfm = bool(cfg.get('cfm_sliprates'))               # the slip rates as split normals from the CFM's SR_pref/min/max
    srf = float(cfg.get('slip_rate_factor') or 0.8)    # the fraction of each fault's slip rate the ruptures take up
    sdx = float(cfg.get('slip_sd_multiplier') or 2)     # each slip-rate sd this many times its uncertainty (2: doubled)
    if grp:
        isf = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
        positive = any(v == 'positive' for v in g_band.values())
        band = next((float(v) for v in g_band.values() if isf(v)), None)
        tau = next((float(v) for v in g_soft.values() if v), None)
        hard = bool(band or positive or mono or plate)

    def rows_txt(groups):                              # 'each subsection's slip rate and each paleo rate'
        word = {'slip-rate': 'each subsection\u2019s slip rate', 'paleo': 'each paleo rate'}
        return ' and '.join(word[k] for k in ('slip-rate', 'paleo') if k in groups)
    pos_g = [k for k, v in g_band.items() if v == 'positive']
    band_g = [k for k, v in g_band.items() if v not in (None, 'positive')]
    soft_g = [k for k, v in g_soft.items() if v]
    dec = lambda t: f'{t:g} decade' + ('' if t == 1 else 's')
    l0 = cfg.get('slip_sum_l0', 8.5)
    dtm = cfg.get('down_the_middle')                   # the balance term: False, True or 'zero-model'
    floor = cfg.get('floor', -10)
    preset = cfg.get('preset', 'n/a')
    reps = cfg.get('representatives')
    sb, bb = cfg.get('slip_block', 'S'), cfg.get('bin_block', 'MFD')
    ref_path = cfg.get('reference')
    ref = os.path.basename(os.path.dirname(os.path.dirname(ref_path))) if ref_path else None
    data_name = os.path.basename(str(cfg.get('data_dir', '')).rstrip('/')) or 'the inputs'

    pre = read(os.path.join(RUN, 'prelim_summary.txt'))
    m_ = re.search(r'SR_min and SR_max at ([\d.]+) sigma, times ([\d.]+)', pre or '')   # --cfm-sliprates' reading
    cfm_ns, cfm_fa = (float(m_.group(1)), float(m_.group(2))) if m_ else (1.0, 2.0)
    cfm_k = cfm_fa / cfm_ns                            # each side's sd over its width
    raw_rows, raw_cols, raw_nnz = grab(pre, r'== raw: G (\d+) x (\d+), nnz (\d+)', 'raw size', int, (1, 2, 3))
    pen_rows, pen_par = grab(pre, r'(\d+) penalty rows contain .*?they apply to (\d+) parameters', 'penalty rows', int, (1, 2))
    soft = grab(pre, r'(\d+) rows after the slip-rate rows have a target of 0', 'soft zero-forcing rows', int)
    zero_rows = grab(pre, r'(\d+) all-zero rows and', 'all-zero rows', int)
    zero_chi2 = grab(pre, r'constant chi\^2 = ([\d.]+)', 'constant chi2')
    n_rows, n_cols = grab(pre, r'cleaned G: (\d+) x (\d+)', 'cleaned size', int, (1, 2))
    removed = None if raw_cols is None or n_cols is None else raw_cols - n_cols
    blocks = {m.group(1): int(m.group(2)) for m in re.finditer(r'row block (\S+)\s+original rows .*?:\s+(\d+) rows', pre)}
    paleo_b = next((b for b in blocks if b not in (sb, bb)), None)
    n_slip, n_bins, n_paleo = blocks.get(sb), blocks.get(bb), blocks.get(paleo_b) if paleo_b else None
    n_sec = grab(pre, r'rows in (\d+) sections \(runs', 'sections', int)
    fc, fc_sd = grab(pre, r'should be near (\d+) \+- (\d+)', 'fit check', int, (1, 2))
    u_lo, u_hi = grab(pre, r'relative uncertainties of .*?\(from ([\d.]+)% .*? to ([\d.]+)%', 'MFD uncertainties', float, (1, 2))
    t_lo, t_hi, t_tot = grab(pre, r'targets R_j from ([\d.e+-]+) to ([\d.e+-]+)/yr, total ([\d.]+)/yr', 'bin targets',
                             float, (1, 2, 3))
    rank = grab(pre, r'numerical rank (\d+)', 'rank', int)
    res_med = grab(pre, r'resolution diagonal R_ii: .*?median ([\d.e+-]+)', 'resolution')
    dup = grab(pre, r'cos>0\.99\s*:\s*(\d+) pairs', 'near-duplicate pairs', int)
    nn_s = grab(pre, r'best fit \(Lawson-Hanson NNLS, \d+ outer iterations, ([\d.]+)s\)', 'best-fit time')
    nn_chi2, nn_nz, nn_sum = grab(pre, r'chi\^2_min = ([\d.]+) .*?nonzeros (\d+);.*?sum m = ([\d.]+)', 'best fit',
                                  float, (1, 2, 3))
    nn_blk = {m.group(1): float(m.group(2)) for m in re.finditer(r'block (\S+)\s*: chi\^2 =\s*([\d.]+) over', pre)}
    nb_lo, nb_hi, nb_med, _, _, nb_pct = grab(pre, r'totals ([+-][\d.]+)% to ([+-][\d.]+)% of their targets \(median '
                                          r'([+-][\d.]+)%\).*?total ([\d.]+) against ([\d.]+) \(([+-][\d.]+)%\)',
                                          'best-fit bins', float, (1, 2, 3, 4, 5, 6))

    runs = sorted(glob.glob(os.path.join(PR, 'main_rep*.npz')), key=lambda p: int(re.findall(r'\d+', os.path.basename(p))[0]))
    runs = [r for r in runs if re.fullmatch(r'main_rep\d+\.npz', os.path.basename(r))]
    hours, outside, chains, recs = [], 0, None, None
    for r in runs:
        with np.load(r) as z:
            hours.append(float(z['seconds']) / 3600)
            if 'band_outside' in z.files:
                outside = max(outside, int(z['band_outside'].max()))
            if 'y' in z.files:
                chains, recs = z['y'].shape[:2]
    n_rec = None if chains is None else chains * recs * len(runs)
    log0 = read(os.path.join(PR, 'main_rep0_log.txt')) if runs else ''
    swaps = grab(log0, r'swap moves: (\d+) near-duplicate pairs', 'swap pairs', int)
    acc = grab(log0, r'swap acceptance ([\d.]+)', 'swap acceptance')
    start_out = grab(log0, r'the start leaves (\d+) rows outside', 'band start', int) if band else None
    logs = [read(r[:-4] + '_log.txt') for r in runs]
    start_chi2 = (re.findall(r'penalty [\d.e+]+: chi2 ([\d.]+), 0 rows outside', ''.join(logs)) or [None])[0]
    pos_rows, pos_on = grab(log0, r'positivity: the start predicts 0 for (\d+) rows; (\d+) ruptures', 'positivity start',
                            int, (1, 2)) if positive else (None, None)
    plate_start = re.search(r'plate rate: the start is above [\d.]+ in (\d+) of (\d+) chains; their rates scaled down by '
                            r'([\d.]+)', log0) if plate else None
    # the terms recorded apart from chi2, per replicate: the soft band's (on chi2's scale) and the summed slip rates'
    lb_re = (r'^soft band \(--log-band.*?([\d.]+) on average \(5-95% ([\d.]+)-([\d.]+)\)[^\n]*?beyond its tau of their '
             r'targets: ([\d.]+) per draw')
    lb = [tuple(float(x) for x in m.groups()) for m in (re.search(lb_re, t, re.M) for t in logs) if m] if tau else []
    if tau and not lb:
        MISSING.append('soft band term')
    lb_start, lb_start_out = grab(log0, r"soft band: .*?the start's ([\d.]+) \(on chi2's scale\), (\d+) rows beyond",
                                  'soft band start', float, (1, 2), flags=re.M) if tau else (None, None)
    ss_re = r'^summed slip rates \(--slip-sum\): ([\d.]+) on average \(5-95% ([\d.]+)-([\d.]+); near ([\d.]+) if'
    ssr = [tuple(float(x) for x in m.groups()) for m in (re.search(ss_re, t, re.M) for t in logs) if m] if ssum else []
    if ssum and not ssr:
        MISSING.append('summed slip rates term')
    rise_cap = re.search(r'^\s*(\d\.\d\d)-\d\.\d\d -> \S+: the target rises .*?at most ([\d.]+) times the lower\'s '
                         r'\(the target difference plus ([\d.]+) sd', log0, re.M) if mono else None   # since 2026-11-03
    rise_cap = (rise_cap.group(1), float(rise_cap.group(2)), float(rise_cap.group(3))) if rise_cap else None
    pr_txt = read(os.path.join(PR, 'plate_rate.txt')) if plate else ''
    plate_top = re.search(r'comes closest to the plate rate \(kept draws\' 97\.5%\):\s*strip\s+([+-]\d+) \(([\d.]+) E, '
                          r'([\d.]+) S\):.*?kept draws ([\d.]+)-([\d.]+); (\w[^,]*), (\w[^,]*?) [\d.]+', pr_txt, re.S)
    plate_in = re.search(r'kept draws: (\d+); limits .*?: all strips inside in ([\d.]+)% of them', pr_txt)
    sweep_s = [float(x[-1]) for x in (re.findall(r'([\d.]+)s/sweep\)', t) for t in logs) if x]
    pm_re = r'^paleo misfit \(--paleo-misfit \w+;.*?\): ([\d.]+) on average \(5-95% ([\d.]+)-([\d.]+)\)'
    pmr = [tuple(float(x) for x in m.groups()) for m in (re.search(pm_re, t, re.M) for t in logs) if m] if pmis != 'gaussian' else []
    if pmis != 'gaussian' and not pmr:
        MISSING.append('paleo misfit term')
    mm_re = r'^MFD misfit \(--mfd-misfit log10;.*?\): ([\d.]+) on average \(5-95% ([\d.]+)-([\d.]+)\)'
    mmr = [tuple(float(x) for x in m.groups()) for m in (re.search(mm_re, t, re.M) for t in logs) if m] if mmis != 'gaussian' else []
    if mmis != 'gaussian' and not mmr:
        MISSING.append('MFD misfit term')
    pm_poor = grab(log0, r'(\d+) site\(s\) above [\d.]+ decades', 'paleo curves', int) if pmis != 'gaussian' else None
    wlogs = sorted(glob.glob(os.path.join(PR, 'workflow_log_*.txt')))
    stamps = re.findall(r'^----- (\d{4}-\d\d-\d\d \d\d:\d\d)', read(wlogs[-1]) if wlogs else '', re.M)
    t_start, t_end = (cfg.get('started') or (stamps[0] if stamps else None)), (stamps[-1] if stamps else None)

    conv = read(os.path.join(PR, 'convergence_report.txt'))
    crow = {m.group(1): m.groups()[1:] for m in re.finditer(
        r'^\s*(\S*main_rep\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.e+]+)\s+([\d.e+]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+'
        r'([\d.e+]+)\s+([\d.e+]+)', conv, re.M)}
    on = {m.group(1): (int(m.group(2)), m.group(3), m.group(4)) for m in re.finditer(
        r'^\s*(\S*main_rep\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$', conv, re.M)}
    verdict = dict(re.findall(r'^(\S*main_rep\d+): (PASS|CHECK)', conv, re.M))
    why_check = dict(re.findall(r'^(\S*main_rep\d+): CHECK: (.*?)(?:: run longer.*)?$', conv, re.M))
    slow = {m.group(1): (int(m.group(2)), m.group(3), float(m.group(4)), float(m.group(5)), int(m.group(6).replace(',', '')))
            for m in re.finditer(r'^(\S*main_rep\d+): slow-mixing minority: (\d+) ruptures \(([\d.]+)%\) with R-hat > '
                                 r'[\d.]+ \(worst ([\d.]+)\), carrying ([\d.]+)% of the total rate; .*?fewest ([\d,]+)',
                                 conv, re.M)}
    rep_sd = [float(x) for x in re.findall(r'sd ([\d.]+) \(1 expected\)', conv)]
    rep_z3 = [float(x) for x in re.findall(r'\|z\| > 3: ([\d.]+)%', conv)]
    names = sorted(crow, key=lambda k: int(re.findall(r'\d+$', k)[0]))
    if not names:
        MISSING.append('convergence table')

    bf = read(os.path.join(PR, 'block_fit_report.txt'))
    brow = [m.groups() for m in re.finditer(
        r'^(.+?)\s+(\d+)\s+([\d.]+)\s+([\d.]+) \+-\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-]?[\d.]+)\s+([\d.]+)\s+'
        r'([\d.]+)\s*$', bf, re.M)]
    bmap = {r[0].strip(): r for r in brow}
    tot = bmap.get('total')
    chi2, chi2_sd, z_all = (float(tot[3]), float(tot[4]), float(tot[8])) if tot else (None, None, None)
    if not tot:
        MISSING.append('row-block table')

    tab = read(os.path.join(PR, 'tables_report.txt'))
    bt_med, bt_lo, bt_hi, bt_nn = grab(tab, r'posterior median ([\d.]+) \(range ([\d.]+) to ([\d.]+)\); best fit ([\d.]+)',
                                       'bin totals', float, (1, 2, 3, 4))
    undet = grab(tab, r'subsections median (\d+)%', 'undetectable share', int)

    sel = read(os.path.join(PR, 'select_report.txt'))
    n_kept = grab(sel, r'from (\d+) kept draws', 'kept draws', int)

    def sel_last(label):
        m = re.search(r'^' + re.escape(label) + r'.*$', sel, re.M)
        return re.split(r'\s{2,}', m.group(0).strip())[-1] if m else None
    s_tot, s_bin, s_sub = (sel_last('weighted mean total rate'), sel_last('weighted mean bin totals'),
                           sel_last('weighted mean subsection totals'))
    s_verdict = (re.findall(r'^\s*K = (\d+): (.*)$', sel, re.M) or [(None, None)])[-1]

    sol = read(os.path.join(PR, 'solution_report.txt'))
    post_tot = grab(sol, r'total ([\d.]+)/yr', 'posterior total rate')
    sol_ok = grab(sol, r'([\d,]+) of ([\d,]+) kept ruptures have exactly the sections', 'solution check', int, (1, 2))

    rm = read(os.path.join(PR, 'fig_rate_magnitude.txt'))
    ref_tot = grab(rm, r'reference: .*?total rate ([\d.]+) against', 'reference total rate') if ref else None
    post_part = rm[rm.find('\n' + a.prior + ' ('):] if ('\n' + a.prior + ' (') in rm else ''
    ratios = [(float(m.group(1)), float(m.group(3))) for m in
              re.finditer(r'^(\d\.\d\d)-(\d\.\d\d)\s+\d+\s+\S+\s+\S+\s+([\d.]+)', post_part, re.M)][:n_bins or 99]

    sr = read(os.path.join(PR, 'fig_slip_rates.txt'))
    s_chi2, s_in1, s_in2, s_med = grab(sr, r'^' + re.escape(a.prior) + r' \(.*?chi2 over the slip-rate rows, as the '
                                       r'likelihood weights them, ([\d.]+); (\d+)% of sections within one .*?(\d+)% within '
                                       r'two.*?median ([\d.]+)', 'slip-rate fit', float, (1, 2, 3, 4))
    r_schi2, r_sin1 = grab(sr, r'^reference: .*?as the likelihood weights them, ([\d.]+); (\d+)% of sections within one',
                           'reference slip-rate fit', float, (1, 2)) if ref else (None, None)
    pr_ = read(os.path.join(PR, 'fig_paleo_rates.txt'))
    p_chi2, p_in1, p_in2, p_med = grab(pr_, r'^' + re.escape(a.prior) + r' \(.*?chi2 over the paleo rows ([\d.]+); (\d+)% '
                                       r'of rows within one .*?(\d+)% within two; prediction / target: median ([\d.]+)',
                                       'paleo fit', float, (1, 2, 3, 4))
    r_pchi2 = grab(pr_, r'^reference: .*?chi2 over the paleo rows ([\d.]+);', 'reference paleo fit') if ref else None

    ss = read(os.path.join(PR, 'fig_slip_sections.txt'))
    faults = []
    hdr = re.search(r'^column .*$', ss, re.M)
    if hdr:                                              # sections and rows may hold spaces: anchor on the numbers after
        has_ref = 'reference' in hdr.group(0)
        row_re = re.compile(r'^\s*\d+\s+.*?\s+\d+\s+(\d+)\s+\d+\s+[\d.]+%\s+(\S+)\s+(\S+)\s+(?:\S+\s+){5}([\d.]+)%\s+'
                            + (r'\S+\s+' if has_ref else '') + r'(.+?)\s*$')
        for line in ss[hdr.end():].split('\n')[1:]:
            m = row_re.match(line)
            if not m:
                break
            faults.append(dict(connected=int(m.group(1)), target=float(m.group(2)), mean=float(m.group(3)),
                               at0=float(m.group(4)), name=m.group(5)))
    named = re.search(r'the (\d+) named faults', ss) is not None

    # ------------------------------------------------------------------ the document
    FONTS = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
    for nm, f in (('DV', 'DejaVuSans.ttf'), ('DV-B', 'DejaVuSans-Bold.ttf'), ('DV-I', 'DejaVuSans-Oblique.ttf'),
                  ('DV-BI', 'DejaVuSans-BoldOblique.ttf'), ('DVM', 'DejaVuSansMono.ttf')):
        pdfmetrics.registerFont(TTFont(nm, os.path.join(FONTS, f)))
    pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV-B', italic='DV-I', boldItalic='DV-BI')
    INK, INK2, MUTED, RULE = (colors.HexColor(c) for c in ('#0b0b0b', '#3d3c39', '#6f6e69', '#d6d5cd'))
    body = ParagraphStyle('body', fontName='DV', fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=7)
    h1 = ParagraphStyle('h1', fontName='DV-B', fontSize=12.5, leading=16, textColor=INK, spaceBefore=12, spaceAfter=5)
    title = ParagraphStyle('title', fontName='DV-B', fontSize=18, leading=23, textColor=INK, spaceAfter=6)
    sub = ParagraphStyle('sub', fontName='DV', fontSize=10, leading=14, textColor=INK2, spaceAfter=3)
    small = ParagraphStyle('small', fontName='DV', fontSize=8.3, leading=11.2, textColor=MUTED)
    cap = ParagraphStyle('cap', fontName='DV', fontSize=8.4, leading=11.6, textColor=INK2, spaceBefore=4, spaceAfter=10)
    cell = ParagraphStyle('cell', fontName='DV', fontSize=8.4, leading=10.8, textColor=INK)
    cellb = ParagraphStyle('cellb', parent=cell, fontName='DV-B')
    P_W, P_H = A4
    L_W, L_H = landscape(A4)
    M = 1.8 * cm
    FW, LW = P_W - 2 * M, L_W - 2 * M
    tt = lambda s: f'<font name="DVM" size="8.4">{s}</font>'
    short = (f'cru_workflow run summary: {data_name}, {preset}' + (f', band {band:g}' if band else ', positivity' if positive
                                                                   else '') + (f', soft band {tau:g}' if tau else ''))
    # the constraints and soft terms in words, for the subtitle
    parts = ((([f'positivity on the {" and ".join(pos_g)} rows'] if pos_g else [])
              + [f'a band on the {k} rows (factor {float(g_band[k]):g})' for k in band_g]
              + [f'a soft band on the {k} rows ({dec(float(g_soft[k]))} at 1 sd)' for k in soft_g]
              + (['the paleo rows scored by their recurrence-interval PDFs'] if pmis != 'gaussian' else [])
              + (['the slip rates as lopsided distributions from the CFM'] if cfm else [])) if grp else
             ([f'the band (factor {band:g})'] if band else ['positivity'] if positive else [])
             + ([f'a soft band ({dec(tau)} at 1 sd)'] if tau else [])) + (['a monotonic MFD'] if mono else []) \
        + (['the summed slip rates'] if ssum else []) + ([f'the plate rate ({plate_txt})'] if plate else [])
    model_txt = ', '.join(parts[:-1]) + (' and ' if len(parts) > 1 else '') + parts[-1] if parts else ''

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('DV', 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(M, 1.0 * cm, short)
        canvas.drawRightString(canvas._pagesize[0] - M, 1.0 * cm, f'{doc.page}')
        canvas.restoreState()

    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=M, rightMargin=M, topMargin=M, bottomMargin=M,
                          title=f'{a.title}: workflow summary', author='cru_workflow', subject=run_name)
    doc.addPageTemplates([
        PageTemplate('portrait', [Frame(M, M, FW, P_H - 2 * M, id='p')], onPage=footer, pagesize=A4),
        PageTemplate('landscape', [Frame(M, M, LW, L_H - 2 * M, id='l')], onPage=footer, pagesize=landscape(A4))])
    story = []

    # the figures that exist, numbered in the order they appear
    FIGS = ['prelim.png', 'nnls/fig_rate_magnitude_nnls.png', 'nnls/fig_slip_rates_nnls.png',
            'nnls/fig_paleo_rates_nnls.png'] + [f'{a.prior}/{f}' for f in (
                'fig_rate_magnitude.png', 'fig_slip_rates.png', 'fig_paleo_rates.png', 'fig_slip_sections.png',
                'fig_slip_sections_ruptures_1.png', 'fig_slip_sections_ruptures_2.png',
                'fig_slip_sections_ruptures_by_magnitude_1.png', 'fig_slip_sections_ruptures_by_magnitude_2.png')]
    present = [f for f in FIGS if os.path.exists(os.path.join(RUN, f))]
    for f in FIGS:
        if f not in present:
            MISSING.append(f)
    num = {f: i + 1 for i, f in enumerate(present)}
    fref = lambda *fs: ('Figures ' if len([f for f in fs if f in num]) > 1 else 'Figure ') + \
        '\u2013'.join(str(num[f]) for f in fs if f in num) if any(f in num for f in fs) else 'the figures (not found)'

    def para(t, s=body):
        story.append(Paragraph(t, s))

    def head(t):
        story.append(Paragraph(t, h1))

    def figure(path, caption, width, max_h=None):
        if path not in num:
            return
        w, h = Image.open(os.path.join(RUN, path)).size
        W, H = width, width * h / w
        if max_h and H > max_h:
            W, H = W * max_h / H, max_h
        img = RLImage(os.path.join(RUN, path), width=W, height=H)
        img.hAlign = 'CENTER'
        story.append(KeepTogether([img, Paragraph(f'<b>Figure {num[path]}.</b> {caption} '
                                                  f'<font name="DVM" size="7.2" color="#6f6e69">{path}</font>', cap)]))

    def table(rows, widths):
        data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=widths, hAlign='LEFT')
        sty = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), 0.6, INK2),
               ('LINEBELOW', (0, -1), (-1, -1), 0.4, RULE), ('TOPPADDING', (0, 0), (-1, -1), 2.5),
               ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5), ('LEFTPADDING', (0, 0), (-1, -1), 4),
               ('RIGHTPADDING', (0, 0), (-1, -1), 4)]
        sty += [('LINEBELOW', (0, i), (-1, i), 0.25, RULE) for i in range(1, len(rows) - 1)]
        t.setStyle(TableStyle(sty))
        story.extend([t, Spacer(1, 8)])

    nrep, ch, sw, bu, th = len(runs), st.get('chains'), st.get('sweeps'), st.get('burn'), st.get('thin')
    refname = ref or 'the reference model'

    # title
    para(a.title, title)
    para(f'Workflow summary for {data_name}, preset {preset}' + (f', with {model_txt}' if model_txt else ''), sub)
    para(f'Run folder: <font name="DVM" size="7.6">{run_name}</font>', small)
    para(f'Run started {t_start or "n/a"}' + (f', last step {t_end}' if t_end else '') + '. Figures as last drawn; '
         'numbers from the run\u2019s reports.', small)
    story.append(Spacer(1, 10))

    head('Overview')
    para('The NZSHM22 inversion asks for the annual rate of every rupture in a rupture set such that the predicted slip '
         'rates, the regional magnitude-frequency distribution (MFD) and the paleoseismic event rates match their targets '
         'within their uncertainties: a linear system <i>G m</i> ≈ <i>d</i> with each row already divided by its '
         'uncertainty. The NZSHM22 solutions find one such model by simulated annealing. This workflow instead maps the '
         'whole set of models consistent with the data and a stated prior: a Bayesian posterior, sampled by an exact '
         'Gibbs sampler, from which it reports posterior means and spreads of every rupture rate, of every predicted datum '
         'and of sums such as fault event rates, together with weighted representative models for hazard calculations. '
         f'The sections below follow the notebook {tt("cru_workflow.ipynb")} step by step; each figure is one this run '
         'wrote.')

    head('1. Inputs and configuration')
    para(f'The inputs are the inversion matrices in {tt(data_name)}: {tt(cfg.get("a_file", "cru_A.txt"))} (<i>G</i>, '
         f'{fmt(raw_rows)} rows by {fmt(raw_cols)} ruptures, {fmt(raw_nnz and raw_nnz / 1e6, "{:.1f}")} million nonzeros) '
         f'and {tt(cfg.get("d_file", "cru_D.txt"))} (<i>d</i>). The rows are the subsections\u2019 slip rates, the MFD '
         'bins, the paleoseismic event rates, and penalty rows that force some ruptures to zero.'
         + (f' The OpenSHA solution for the same rupture set ({tt(refname)}) supplies the fault geometry and named '
            'faults for the figures, and a reference model drawn in black on every figure.' if ref else '')
         + ' One configuration cell sets everything; this run used:')
    dtm_txt = {False: 'no balance term: the Gaussian misfit alone', True: 'a balance term, one weight for every data type',
               'zero-model': 'a balance term, each data type\u2019s own weight'}.get(dtm, str(dtm))
    table([['setting', 'value', 'meaning'],
           ['SLIP_SECTIONS', repr(cfg.get('slip_sections')), 'how the slip-rate rows count (\u2019auto\u2019: each fault '
            'section\u2019s slip rate as one measurement, however many subsections it is copied to)'],
           ['MFD_UNCERTAINTY', repr(cfg.get('mfd_uncertainty')), 'the MFD bins\u2019 uncertainty level (\u2019as-given\u2019: '
            'as the inputs give them)'],
          ] + ([[f'{nm}_BAND', repr(g_band[k]) if g_band[k] in (None, 'positive') else f'{float(g_band[k]):g}',
                 (f'each {k} prediction kept above zero (a hard constraint)' if g_band[k] == 'positive' else
                  f'each {k} prediction kept within a factor {float(g_band[k]):g} of its target (a hard constraint)'
                  if g_band[k] else 'no limit')] for nm, k in (('SLIP', 'slip-rate'), ('PALEO', 'paleo'))]
                + [[f'{nm}_BAND_SOFT', 'None' if not g_soft[k] else f'{float(g_soft[k]):g}',
                    f'each {k} prediction within {dec(float(g_soft[k]))} of its target at 1 sd (a soft band: a penalty on '
                    'the log ratio of prediction to target, beside the misfit)' if g_soft[k] else 'no soft band']
                   for nm, k in (('SLIP', 'slip-rate'), ('PALEO', 'paleo'))]
                + [['PALEO_MISFIT', repr(pmis), 'the paleo rows scored by each site\u2019s recurrence-interval PDF in years '
                    '(\u22122 ln of its density, a curve through its quantiles with its mode at 1/bestEstimate), in place of '
                    'the Gaussian' if pmis == 'years' else 'the paleo rows\u2019 Gaussian, as the inputs give it'
                    if pmis == 'gaussian' else 'the paleo rows scored by their curves\u2019 density in log rate'],
                   ['MFD_MISFIT', repr(mmis), 'each MFD bin\u2019s total scored in log10, ((log10 total \u2212 log10 R<sub>j</sub>) / '
                    's<sub>j</sub>)\u00b2 with s<sub>j</sub> = log10(1 + \u03c3<sub>j</sub>/R<sub>j</sub>), each bin kept above zero, in '
                    'place of the Gaussian' if mmis == 'log10' else 'the bins\u2019 Gaussian in rate, as the inputs give it'],
                   ['CFM_SLIPRATES', str(cfm), 'each slip rate a split normal in mm/yr from the NZ Community Fault Model '
                    f'scaled as a whole by {srf:g} (SR_min and SR_max at {cfm_ns:g} sd from SR_pref, times {cfm_fa:g}), the slip sums too'
                    if cfm else 'the slip rates\u2019 Gaussians, as the inputs give them'],
                   ['SLIP_RATE_FACTOR', f'{srf:g}', 'the assumed fraction of each fault\u2019s slip rate the ruptures take up, '
                    'scaling each slip-rate distribution as a whole, target and sd' + (' (the inputs\u2019 own)' if srf == 0.8 else '')],
                   ['SLIP_SD_MULTIPLIER', f'{sdx:g}', 'each slip-rate sd this many times its uncertainty'
                    + ', before the slip-rate factor' + (' (the inputs\u2019 doubling)' if sdx == 2 else ' (the uncertainties as given)'
                       if sdx == 1 else ' (with 0.8, the inputs\u2019 slip rates exactly as given)' if sdx == 2.5 else '')]]
                if grp else
                [['BAND', repr('positive') if positive else f'{band:g}' if band else 'None',
                  'every prediction kept above zero (a hard constraint)' if positive else
                  f'every prediction kept within a factor {band:g} of its target (a hard constraint)' if band else
                  'neither positivity nor the band'],
                 ['BAND_SOFT', 'None' if not tau else f'{tau:g}',
                  f'each prediction within {dec(tau)} of its target at 1 sd (a soft band: a penalty on the log ratio of '
                  'prediction to target, beside the misfit)' if tau else 'no soft band']]) + [
          ] + ([['MFD_MONOTONIC', str(mono), 'no magnitude bin\u2019s total above its lower neighbour\u2019s where the '
                 'targets do not rise' + (f', and where they rise, the rise capped {rise_cap[2]:g} sd above the target '
                                          'difference' if rise_cap else '') + ' (a hard constraint; a notebook setting '
                 'until November 4th, 2026, gibbs.py --mfd-monotonic since)']] if mono else []) + [
           ['CROSS_SECTIONS', repr(xsec), 'the strips of cross-sections perpendicular to the plate motion that SLIP_SUM '
            + ('and PLATE_RATE use' if plate else 'uses') + f', 8.5 km wide {xsec_txt}'],
           ['SLIP_SUM', str(bool(ssum)),
            'the slip along the plate motion (each fault\u2019s slip vector projected on it) summed along each '
            'cross-section perpendicular to it, against the targets\u2019 sums (a soft term)' if ssum else 'not added']]
          + ([['SLIP_SUM_L0', f'{l0:g}', f'the summed slip rates\u2019 weight: each {l0:g} km of the plate boundary counts '
               'as one measurement']] if ssum else [])
          + ([['PLATE_RATE', f'{plate[1]:g}' if plate[0] is None else f'({plate[0]:g}, {plate[1]:g})',
               f'each strip\u2019s share of the plate rate {plate_txt} (a hard constraint; a notebook setting until '
               'November 4th, 2026, gibbs.py --plate-rate since)']] if plate else [])
          + [['DOWN_THE_MIDDLE', str(dtm), dtm_txt]]
          + [['FLOOR', f'{floor:g}', f'the lowest rate an active rupture can take, 10<super>{floor:g}</super> per year'],
           ['PRESET', repr(preset), f'{nrep} replicates' + (f' (of the preset\u2019s {st["replicates"]}; stopped early)'
                                     if st.get('replicates') and nrep < st['replicates'] else '') + f' × {ch} chains × {fmt(sw)} sweeps (burn-in {fmt(bu)}, a draw kept '
            f'every {th})'],
           ['REPRESENTATIVES', str(reps), 'weighted representative models selected for hazard']],
          [3.7 * cm, 2.8 * cm, FW - 6.5 * cm])

    head('2. Cleaning and diagnostics')
    para(f'Cleaning ({tt("preliminaries.py")}) turns the raw system into the problem the sampler sees. The '
         f'{fmt(pen_rows)} penalty rows with very large entries fix {fmt(pen_par)} ruptures at zero, and {fmt(soft)} soft '
         f'zero-forcing rows fix {fmt(soft)} more, so {fmt(removed)} ruptures are removed (at rate 0 in every output). '
         f'{fmt(zero_rows)} rows that then hold no remaining rupture add a constant χ² of {fmt(zero_chi2, "{:.1f}")} that '
         f'no model can change, and are dropped. What remains is {fmt(n_rows)} rows by {fmt(n_cols)} ruptures: '
         f'{fmt(n_slip)} slip-rate rows, {fmt(n_bins)} MFD bins and {fmt(n_paleo)} paleo rows. '
         + (f'The slip-rate rows form {fmt(n_sec)} fault sections; each section\u2019s rows are divided by √k for its k '
            'subsections, so a section\u2019s slip rate counts once rather than k times. '
            if cfg.get('slip_sections') == 'auto' else '')
         + f'The MFD bin targets run from {fmt(t_lo, "{:.2g}")} to {fmt(t_hi, "{:.2g}")} per year and sum to '
         f'{fmt(t_tot, "{:.4f}")} per year, with relative uncertainties of {fmt(u_lo, "{:.1f}")}% to {fmt(u_hi, "{:.1f}")}%. '
         f'If the stated uncertainties are right, a model drawn from the posterior should have χ² near {fmt(fc)} ± '
         f'{fmt(fc_sd)}: this is the <i>fit check</i> used below. The diagnostics ({fref("prelim.png")}) show how weakly '
         f'individual rates are determined: <i>G</i> has rank {fmt(rank)}, and the median diagonal of the resolution '
         f'matrix is {fmt(res_med, "{:.3f}")}, so the data fix sums of rates far better than single rates. There are also '
         f'{fmt(dup)} pairs of near-duplicate ruptures (cosine similarity above 0.99), which the sampler handles with swap '
         'moves.')
    figure('prelim.png', 'Cleaning diagnostics. Top left: per-rupture scales (the rate at which one rupture alone moves a '
           'prediction by one uncertainty, the most any datum allows it, and an even share of each datum). Top right: the '
           'singular values of <i>G</i>. Bottom left: the resolution diagonal. Bottom right: the best fit\u2019s residuals, '
           'row by row, coloured by data type.', FW * 0.86)

    head('3. The best fit (no prior)')
    para(f'Cleaning also computes the best nonnegative least-squares fit (Lawson\u2013Hanson NNLS, {fmt(nn_s)} s). It '
         f'reaches χ² = {fmt(nn_chi2, "{:.1f}")} over the {fmt(n_rows)} rows with only {fmt(nn_nz)} ruptures above zero: '
         f'{fmt(nn_blk.get(sb), "{:.1f}")} from the slip rates, {fmt(nn_blk.get(bb), "{:.1f}")} from the MFD bins and '
         f'{fmt(nn_blk.get(paleo_b), "{:.1f}")} from the paleo rows. Its total rate is {fmt(nn_sum, "{:.4f}")} per year, '
         f'{fmt(nb_pct, "{:+.1f}")}% against the bin targets, with every bin between {fmt(nb_lo, "{:+.1f}")}% and '
         f'{fmt(nb_hi, "{:+.1f}")}%. A fit chooses among all the rates to match far fewer numbers, so its χ² lies well '
         'below the fit check; it is a reference with no prior and the starting point of the chains, not a sample of the '
         f'posterior. {fref("nnls/fig_rate_magnitude_nnls.png", "nnls/fig_paleo_rates_nnls.png")} show its fit to each '
         'data type' + (f', with {refname} for comparison' if ref else '') + '; the same three figures for the posterior '
         'follow in step 9.'
         + (f' The best fit leaves {fmt(start_out)} rows outside the band, so the chains cannot start from it as it '
            'stands (step 5).' if band and start_out else ''))
    figure('nnls/fig_rate_magnitude_nnls.png', 'Best fit: each rupture\u2019s rate against its magnitude bin (ruptures at '
           'zero in the band at the bottom), each bin\u2019s target (black line) and total (orange)'
           + (f', and {refname}\u2019s bin totals (black squares)' if ref else '') + '. Lower panel: each bin\u2019s total '
           'over its target, with the band within one uncertainty.', FW * 0.6)
    figure('nnls/fig_slip_rates_nnls.png', 'Best fit: each fault section\u2019s slip-rate target against the predicted '
           'section mean, in units of the section\u2019s measurement uncertainty, on linear and log scales'
           + ('; open squares are the reference\u2019s predictions' if ref else '') + '. The grey band is within one '
           'uncertainty of the target.', FW * 0.8)
    figure('nnls/fig_paleo_rates_nnls.png', 'Best fit: each paleo row\u2019s target against its prediction, in units of its '
           'uncertainty, log-log; the shaded bands are within one and two uncertainties.', FW * 0.6, max_h=15 * cm)

    head('4. The model: prior, likelihood' + (' and constraints' if hard or tau or ssum else ''))
    para(f'<b>Prior.</b> Each rupture is either off, with a rate of exactly zero, or on, with a rate that is uniform in log '
         f'rate from 10<super>{floor:g}</super> per year up to its magnitude bin\u2019s target rate (spike-and-slab). The '
         'top is soft, a Gaussian shoulder as wide as the bin\u2019s own uncertainty, since no single rupture can carry more '
         'than its bin\u2019s total. The fraction of ruptures switched on is estimated separately in each bin (uniform prior '
         'on each fraction), so the data decide how many ruptures are active, and the prior puts no pressure on the bin '
         'totals. <b>Likelihood.</b> Gaussian, exp(−χ²/2) with the stated uncertainties'
         + ('.' if not dtm else ', plus a balance term that penalizes a data type predicted all low or all high.')
         + (f' <b>Band.</b> In addition, '
            + (rows_txt(band_g) + ' must lie' if grp else 'every prediction with a positive target (each subsection\u2019s '
               'slip rate, each MFD bin and each paleo rate) must lie')
            + f' within a factor {band:g} of its target, from <i>d</i>/{band:g} to {band:g} '
            '<i>d</i>. This is a hard constraint, a prior that is zero outside it. Its main effect is on faults that share '
            'few ruptures with others: without it the posterior often leaves such faults with no events at all, since '
            'their weak slip-rate targets tolerate a prediction of zero; with it every subsection must slip at no less '
            f'than 1/{band:g} of its target rate. The factor is a modelling choice the data do not make.' if band else '')
         + (' <b>Positivity.</b> In addition, '
            + (rows_txt(pos_g) + ' must be above zero (the MFD bins have no such limit)' if grp else
               'every prediction with a positive target (each subsection\u2019s slip rate, each MFD bin and each paleo rate) '
               'must be above zero')
            + ', so at least one rupture is on in each, with no other limit: a hard constraint.' if positive else '')
         + (f' <b>A soft band.</b> A penalty beside the misfit on each prediction\u2019s ratio to its target, '
            f'\u00bd (log<sub>10</sub>(prediction / target) / {tau:g})\u00b2, so a prediction {dec(tau)} from its target costs as '
            'much as a one-uncertainty miss in \u03c7\u00b2, on '
            + (' and '.join(f'the {k} rows' + (f' ({dec(float(g_soft[k]))} at 1 sd)' if len(soft_g) > 1 else '') for k in soft_g)
               if grp else 'every row with a positive target')
            + ' (each slip-rate subsection\u2019s term divided by the k subsections of its section, so a section counts once). It ignores each row\u2019s '
            'uncertainty, so it acts where a target is weak and the data alone would accept a prediction near zero; where '
            'a target is strong the Gaussian likelihood already holds the prediction far closer than a decade. A '
            'prediction of zero would cost infinitely much, which is why it comes with positivity. It is recorded apart '
            'from \u03c7\u00b2, on \u03c7\u00b2\u2019s scale (twice its share of the negative log posterior), so the fit check stays over the '
            'data alone.' if tau else '')
         + (' <b>The MFD bins in log10.</b> The bins leave the Gaussian likelihood: each bin\u2019s total is scored instead by '
            '((log10 total \u2212 log10 R<sub>j</sub>) / s<sub>j</sub>)\u00b2, s<sub>j</sub> = log10(1 + \u03c3<sub>j</sub>/R<sub>j</sub>), '
            'centred on its target, so a bin 20% high costs a little less than one 20% low; each bin is kept above zero, and '
            'the term is kept exactly by the soft band\u2019s Metropolis steps and recorded apart; \u03c7\u00b2 still counts the '
            'bins\u2019 Gaussian, for comparison.' if mmis == 'log10' else '')
         + (' <b>The paleo misfit.</b> The paleo rows leave the Gaussian likelihood: each is scored instead by its site\u2019s '
            'recurrence-interval PDF from the paleoseismic analysis, \u22122 ln of its density in years, measured from its '
            'minimum, a curve through the site\u2019s 2.275, 15.865, 84.135 and 97.725% points with its mode at '
            '1/bestEstimate exactly (the rupture set\u2019s paleo_constraint_data.json)'
            + (f'; {pm_poor} site(s) fit their quantiles to worse than 0.05 decades' if pm_poor else '')
            + '. The ranges are lopsided, and the Gaussian caps a site\u2019s misfit, so this penalizes paleo rates predicted '
            'well below their targets far more. It is kept exactly by the same Metropolis steps as the soft band, and '
            'recorded apart; \u03c7\u00b2 still counts the paleo rows\u2019 Gaussian misfit, for comparison.' if pmis == 'years' else '')
         + (' <b>Slip rates from the CFM.</b> Each subsection\u2019s slip-rate row is rebuilt from its parent fault\u2019s '
            'line of the NZ Community Fault Model (SR_pref, SR_min, SR_max in mm/yr): a split normal in the slip rate itself, '
            f'scaled as a whole by {srf:g} (SLIP_RATE_FACTOR, the assumed fraction of the slip rate the ruptures take up): its '
            f'mode at {srf:g} \u00d7 SR_pref, its sd below the mode '
            + (f'{cfm_k:g} \u00d7 ' if cfm_k != 1 else '') + f'{srf:g} \u00d7 (SR_pref \u2212 SR_min) and above it '
            + (f'{cfm_k:g} \u00d7 ' if cfm_k != 1 else '') + f'{srf:g} \u00d7 (SR_max \u2212 SR_pref) '
            f'(SR_min and SR_max at {cfm_ns:g} sd, times {cfm_fa:g}' + (', the inputs\u2019 doubling' if cfm_fa == 2 else '')
            + '; a side of zero width takes the other\u2019s). The inputs\u2019 Gaussian has one sd, twice the half range, for both sides; '
            'here each side has its own, and their mean is the unit of the rows, of \u03c7\u00b2 and of the summed slip rates, '
            'which sum the modes. The misfit, each residual weighed by its own side\u2019s sd, is evaluated exactly in every '
            'update.' if cfm else '')
         + (' <b>A monotonic MFD.</b> Wherever a magnitude bin\u2019s target is not above the next lower bin\u2019s, its '
            'total rate may not be either, in every model'
            + (f'; where a target rises (from the M{rise_cap[0]}\u2013{float(rise_cap[0]) + 0.1:.2f} bin to the next), the upper bin\u2019s total may be at '
               f'most {rise_cap[1]:.3f} times the lower\u2019s, a cap {rise_cap[2]:g} sd above the targets\u2019 own difference, as far as the other pairs\u2019 '
               'limits sit above theirs (the median).' if rise_cap else '.') if mono else '')
         + (f' <b>The plate rate.</b> In each 8.5-km strip of cross-sections perpendicular to the plate motion '
            f'{xsec_txt}, the crustal faults must take up {plate_txt} of the Pacific\u2013Australia plate rate, each '
            'fault\u2019s slip vector projected on the local plate motion and averaged over the strip\u2019s '
            f'cross-sections ({tt("plate_rate.txt")}).' if plate else '')
         + (' <b>Summed slip rates.</b> A soft term in the misfit: along every cross-section perpendicular to the plate '
            f'motion {xsec_txt}, the model\u2019s slip along the plate motion added up (each fault\u2019s slip vector projected on '
            'it) should match the targets\u2019 added up, within that sum\u2019s '
            'uncertainty, each subsection counting once in all; each ' + f'{cfg.get("slip_sum_l0", 8.5):g}' + ' km of the '
            'plate boundary counts as one measurement, and it is recorded apart from χ² (' + tt('plate_rate.txt') + ').'
            if ssum else ''))

    head('5. Sampling')
    para(f'The sampler ({tt("gibbs.py")}, a C kernel run from Python) is an exact Gibbs sampler: it updates one rupture at '
         'a time, drawing its rate from its exact conditional distribution by a slice sampler in log rate, with an exact '
         'step that switches the rupture on or off. '
         + ('Under the band each update is restricted to the rates that keep every row the rupture touches inside it. '
            if band else '')
         + ('Each update is restricted to the rates that keep the hard constraints ('
            + ', '.join(([f'no {" or ".join(pos_g)} row predicted at zero' if grp else 'no positive-target row predicted at zero']
                         if positive else [])
                        + (['the MFD’s monotonicity'] if mono else []) + (['the plate-rate limit'] if plate else []))
            + '). ' if not band and hard else '')
         + ('The soft band is not quadratic in the rates, so the sampler keeps it exactly by Metropolis steps: a switch '
            'on or off takes the penalty’s change in its log odds, and a slab move or a swap is accepted by its change; '
            'for the many off ruptures, bounds from cached log ratios settle most decisions without logarithms'
            + (f' (here {rng(sweep_s, "{:.3f}")} s per sweep for {ch} chains on {cfg.get("threads", "n/a")} threads)'
               if sweep_s else '') + '. ' if tau else '')
         + ('The summed slip rates are quadratic in the rates, so they enter the exact updates as extra rows of the misfit. '
            if ssum else '')
         + f'Swap moves exchange the rates of near-duplicate ruptures ({fmt(swaps)} pairs; '
         f'{fmt(acc and 100 * acc, "{:.1f}")}% of swaps accepted), and the fraction on in each bin is redrawn after every '
         'sweep. '
         + (f'The chains start from the best fit moved inside the band by a penalized least-squares fit (χ² '
            f'{start_chi2 or "n/a"}, no rows outside). ' if band else 'The chains start from the best fit'
            + (f', with {fmt(pos_on)} ruptures switched on at twice the slab floor so that none of the {fmt(pos_rows)} '
               'rows it predicts at zero is left there' if positive and pos_on else '')
            + (f'; in {plate_start.group(1)} of {plate_start.group(2)} chains it took up more than the plate-rate limit, '
               f'and their rates were scaled down (by a factor {float(plate_start.group(3)):.3f} at most)'
               if plate_start else '') + '. ')
         + f'Each of the {nrep} replicates ran {ch} chains for {fmt(sw)} sweeps in '
         f'{rng(hours, "{:.1f}")} hours, '
         f'keeping {fmt(recs)} draws per chain after burn-in: {fmt(n_rec)} recorded draws in all.'
         + (' No recorded draw broke the hard constraints.' if hard and outside == 0 and runs else
            f' Some recorded draws broke the hard constraints ({outside} rows at most): a sign of a problem.'
            if hard and runs else '')
         + ' The runs keep running summaries rather than every draw, plus whole draws for the representative models and '
         'the figures.')

    head('6. Convergence')
    passes = [verdict.get(n) for n in names]
    ess_c = [float(crow[n][3]) for n in names]
    ess_t = [float(crow[n][4]) for n in names]
    on_n = [on[n][0] for n in names if n in on]
    sl = [slow[n] for n in names if n in slow]
    para(f'{tt("check_convergence.py")} judges each replicate against the targets of its preset: '
         + (f'all {len(names)} <b>pass</b>. ' if passes and all(p == 'PASS' for p in passes) else
            'verdicts ' + ', '.join(n.split('/')[-1] + ' ' + verdict.get(n, 'n/a') for n in names) + ' ('
            + ('; '.join(n.split('/')[-1] + ': ' + why_check[n] for n in names if n in why_check) or
               f'see {tt("convergence_report.txt")} for the reasons') + '). ')
         + f'For χ² and the total rate the effective sample sizes are {rng(ess_c)} and {rng(ess_t)} per replicate, and '
         f'for the rupture rates the 5th percentile is {rng([float(crow[n][9]) for n in names])}. About {rng(on_n)} '
         'ruptures are on in each draw. With so many rates even perfectly independent draws give a worst R̂ above 1, so '
         f'R̂ is judged against that noise baseline ({crow[names[0]][6] if names else "n/a"} here). '
         + (f'A small minority of ruptures mixes slowly, mostly ruptures that switch on and off rarely: {rng([s[0] for s in sl])} '
            f'ruptures per replicate with R̂ above the threshold (worst {rng([s[2] for s in sl], "{:.3f}")}), carrying '
            f'{rng([s[3] for s in sl], "{:.1f}")}% of the total rate, with at least {min(s[4] for s in sl):,} effective '
            'draws each. ' if sl else '')
         + (f'The replicates agree rupture by rupture as independent runs should: the standardized differences of their '
            f'mean log rates have a standard deviation of {rng(rep_sd, "{:.3f}")} (1 expected), with '
            f'{rng(rep_z3, "{:.2f}")}% beyond 3 (0.27% expected).' if rep_sd else ''))
    table([['replicate', 'χ² (mean)', 'effective draws: χ² / total rate', 'worst R̂ (baseline)',
            'slow-mixing ruptures (share of rate)', 'verdict']]
          + [[n.split('/')[-1], crow[n][2], f'{float(crow[n][3]):,.0f} / {float(crow[n][4]):,.0f}',
              f'{crow[n][5]} ({crow[n][6]})', f'{slow[n][0]} ({slow[n][3]:.1f}%)' if n in slow else '\u2013',
              verdict.get(n, 'n/a')] for n in names],
          [2.2 * cm, 1.8 * cm, 3.8 * cm, 2.9 * cm, 3.6 * cm, FW - 14.3 * cm])

    head('7. Fit to the data')
    if chi2 is not None and z_all is not None:
        how = ('consistent with it' if abs(z_all) <= 2 else f'below it ({fmt(z_all, "{:+.1f}")} standard deviations)'
               if z_all < 0 else f'above it ({fmt(z_all, "{:+.1f}")} standard deviations)')
        why = (' The fit check assumes the Gaussian likelihood alone; the band is stricter than the data wherever a '
               'target is weak (a section whose target is less than about one uncertainty above zero would otherwise '
               'accept a prediction of zero), so under the band χ² sits below it, and the fit check no longer tests the '
               'uncertainties.' if band and z_all < -2 else
               ' The fit check assumes the Gaussian likelihood alone; the soft band pulls the predictions of weak-target '
               'rows towards their targets, closer than the data alone would hold them, so χ² sits lower than it would '
               'without it.' if tau and z_all < -2 else
               ' A χ² below the fit check means the data are fitted more closely than their uncertainties imply.'
               if z_all < -2 else ' A χ² above the fit check means some data are fitted worse than their uncertainties '
               'allow, or the uncertainties are too small.' if z_all > 2 else '')
        para(f'The posterior\u2019s χ² is {chi2:.1f} ± {chi2_sd:.1f}, against the fit check\u2019s {fmt(fc)} ± {fmt(fc_sd)}: '
             f'{how}.{why} The row-block fit ({tt("block_fit.py")}) splits χ² by data type: a block\u2019s χ² per measurement '
             'near 1 is fitted as its uncertainties say, below 1 more closely, above 1 worse. In the table, the '
             '\u201cmean model\u201d is the misfit of the posterior-mean prediction and the \u201cspread\u201d is the posterior '
             'variance of the predictions; the two add up to χ².')
    table([['row block', 'rows', 'measurements', 'χ² over the draws', 'per measurement', 'z', 'mean model', 'spread']]
          + [[r[0].strip(), f'{int(r[1]):,}', f'{float(r[2]):g}', f'{r[3]} ± {r[4]}', r[7], r[8].replace('-', '\u2212'),
              r[9], r[10]] for r in brow],
          [2.9 * cm, 1.3 * cm, 2.6 * cm, 2.9 * cm, 2.6 * cm, 1.2 * cm, 2.0 * cm, FW - 15.5 * cm])
    if lb or ssr or pmr or mmr:
        para('Each run also records the terms it adds beside χ², each on χ²’s scale and not counted in it. '
             + (f'<b>The MFD misfit in log10</b>: {rng([x[0] for x in mmr], "{:.1f}")} on average per replicate (5–95% of draws '
                f'{fmt(min(x[1] for x in mmr), "{:.1f}")}–{fmt(max(x[2] for x in mmr), "{:.1f}")}), in the likelihood in '
                'place of the bins\u2019 Gaussian, which χ² still counts. ' if mmr else '')
             + (f'<b>The paleo misfit</b>: {rng([x[0] for x in pmr], "{:.1f}")} on average per replicate (5–95% of draws '
                f'{fmt(min(x[1] for x in pmr), "{:.1f}")}–{fmt(max(x[2] for x in pmr), "{:.1f}")}), in the likelihood in '
                'place of the paleo rows\u2019 Gaussian, which χ² still counts. ' if pmr else '')
             + (f'<b>The soft band</b>: {rng([x[0] for x in lb], "{:.1f}")} on average per replicate (5–95% of draws '
                f'{fmt(min(x[1] for x in lb), "{:.1f}")}–{fmt(max(x[2] for x in lb), "{:.1f}")}'
                + (f'; the start’s {lb_start:.1f}' if lb_start is not None else '') + '), with '
                f'{rng([x[3] for x in lb], "{:.0f}")} rows per draw predicted more than {dec(tau)} from their targets'
                + (f' (the start: {lb_start_out:.0f})' if lb_start_out is not None else '') + '. ' if lb else '')
             + (f'<b>The summed slip rates</b>: {rng([x[0] for x in ssr], "{:.1f}")} on average per replicate (5–95% '
                f'{fmt(min(x[1] for x in ssr), "{:.1f}")}–{fmt(max(x[2] for x in ssr), "{:.1f}")}), against about '
                f'{ssr[0][3]:.1f} expected if the model matched the truth (one per {l0:g} km of strip); '
                f'{tt("plate_rate.txt")} lists the strips furthest from their target sums.' if ssr else '')
             + (f' <b>The plate rate</b>: every strip is inside the limits in {float(plate_in.group(2)):.1f}% of the '
                f'{int(plate_in.group(1)):,} kept draws, as a hard constraint requires'
                + (f'; the strip closest to the limit, {plate_top.group(1)} ({plate_top.group(2)}° E, '
                   f'{plate_top.group(3)}° S, mostly {plate_top.group(6).rsplit(" ", 1)[0]} and '
                   f'{plate_top.group(7)} in the fault model), reaches {float(plate_top.group(5)):.3f} at the kept '
                   'draws\u2019 97.5%' if plate_top else '') + '.' if plate_in else ''))

    head('8. Posterior tables, rate vectors and representative models')
    rel = None if post_tot is None or t_tot is None else 100 * (post_tot / t_tot - 1)
    para(f'{tt("make_tables.py")} pools the replicates into per-rupture, per-subsection and per-bin tables and into rate '
         f'vectors ({tt("rate_vectors.csv")}): for each of the {fmt(raw_cols)} ruptures its posterior mean, median, mode '
         'and 2.3%, 15.9%, 84.1% and 97.7% points. Most ruptures are off in most draws, so their medians and modes are '
         'zero; sums such as fault or bin rates must be built from the means. '
         f'The posterior-mean total rate is {fmt(post_tot, "{:.4f}")} per year, {fmt(rel, "{:+.1f}")}% against the bin '
         'targets' + (f' ({refname}: {ref_tot:.4f})' if ref_tot else '') + '; the bin totals sit a median '
         f'{fmt(bt_med and 100 * (bt_med - 1), "{:+.1f}")}% from their targets ({fmt(bt_lo and 100 * (bt_lo - 1), "{:+.1f}")}% '
         f'to {fmt(bt_hi and 100 * (bt_hi - 1), "{:+.1f}")}%), against the best fit\u2019s '
         f'{fmt(bt_nn and 100 * (bt_nn - 1), "{:+.1f}")}%. About {fmt(undet)}% of each subsection\u2019s posterior rate (the '
         'median) is carried by ruptures whose rates are individually below what the data can detect. '
         + (f'{tt("select_draws.py")} then picks {reps} weighted representative models from the {fmt(n_kept)} kept draws, '
            'clustering them by their rates per subsection and magnitude bin, for use in hazard calculations. With '
            f'{reps}, the weighted mean total rate is off the exact mean by {s_tot}, the worst bin by {s_bin}, and '
            f'{s_sub} of subsection means are within 10% (the tolerance asks for 90%)'
            + ('; the selection misses some tolerances, so the posterior-mean rate vector remains the exact input for '
               'mean hazard.' if s_verdict[1] and 'miss' in s_verdict[1] else '.') if reps and sel else ''))

    head('9. The posterior\u2019s fit, data type by data type')
    hi_bins = [r for m_, r in ratios if m_ >= 7.1]
    span = lambda v: rng([100 * (r - 1) for r in v], '{:+.1f}').replace('\u2013', '% to ')   # +1.5% to +4.6
    lo_bins = [r for m_, r in ratios if m_ < 7.1]
    paleo_txt = ('the posterior underpredicts them, with a median ratio of prediction to target of ' if p_med and p_med < 0.85
                 else 'the posterior overpredicts them, with a median ratio of ' if p_med and p_med > 1.15
                 else 'the median ratio of prediction to target is ')
    para(f'The next three figures repeat {fref("nnls/fig_rate_magnitude_nnls.png", "nnls/fig_paleo_rates_nnls.png")} for '
         'the posterior. Each coloured dot is the posterior mean'
         + (', and each faint x one of the kept draws, so where the draws gather shows darkest' if n_kept else '')
         + ('; open black squares are ' + refname if ref else '') + '. '
         + (f'<b>Rates by magnitude</b> ({fref(a.prior + "/fig_rate_magnitude.png")}): the posterior spreads each bin\u2019s '
            'rate over many ruptures at low mean rates, where the best fit puts a few ruptures high and the rest at zero. '
            + (f'The bin totals are {span(lo_bins)}% from their targets below M7.1 and '
               f'{span(hi_bins)}% from M7.1 upward. '
               if lo_bins and hi_bins else ''))
         + f'<b>Slip rates</b> ({fref(a.prior + "/fig_slip_rates.png")}): {fmt(s_in1)}% of the {fmt(n_sec)} sections\u2019 '
         f'posterior means lie within one measurement uncertainty of their targets and {fmt(s_in2)}% within two'
         + (f' ({refname}: {r_sin1:.0f}% within one)' if r_sin1 is not None else '')
         + f', with a median ratio of prediction to target of {fmt(s_med, "{:.2f}")} (mean-model χ² {fmt(s_chi2, "{:.1f}")}'
         + (f'; {refname}: {r_schi2:.1f}' if r_schi2 is not None else '') + '). '
         + f'<b>Paleo rates</b> ({fref(a.prior + "/fig_paleo_rates.png")}): {paleo_txt}{fmt(p_med, "{:.2f}")}; '
         f'{fmt(p_in1)}% of rows lie within one uncertainty and {fmt(p_in2)}% within two (mean-model χ² '
         f'{fmt(p_chi2, "{:.1f}")}' + (f'; {refname}: {r_pchi2:.1f}' if r_pchi2 is not None else '') + ').')
    xs = ' Faint x\u2019s: each kept draw\u2019s' if n_kept else ''
    figure(f'{a.prior}/fig_rate_magnitude.png', 'Posterior: each rupture\u2019s posterior mean rate against its magnitude '
           'bin, each bin\u2019s target (black line) and total of the means (orange dot)'
           + (f', and {refname}\u2019s bin totals (black squares)' if ref else '') + '.' + (xs + ' bin totals.' if xs else '')
           + ' Lower panel: each bin\u2019s total over its target.', FW * 0.6)
    figure(f'{a.prior}/fig_slip_rates.png', 'Posterior: each fault section\u2019s slip-rate target against its '
           'posterior-mean section mean (dots), in units of its measurement uncertainty.' + (xs + ' section means.' if xs else '')
           + (' Open squares: the reference.' if ref else ''), FW * 0.88)
    figure(f'{a.prior}/fig_paleo_rates.png', 'Posterior: each paleo row\u2019s target against its posterior-mean prediction '
           '(dots).' + (xs + ' predictions.' if xs else '') + (' Open squares: the reference.' if ref else ''),
           FW * 0.6, max_h=15 * cm)

    fig_ss = [f'{a.prior}/{f}' for f in ('fig_slip_sections.png', 'fig_slip_sections_ruptures_1.png',
                                          'fig_slip_sections_ruptures_2.png', 'fig_slip_sections_ruptures_by_magnitude_1.png',
                                          'fig_slip_sections_ruptures_by_magnitude_2.png')]
    if any(f in num for f in fig_ss):
        head('10. Faults one by one')
        unit = 'named faults' if named else 'slip-rate sections'
        txt = (f'{tt("plot_slip_sections.py")} looks at {len(faults) or "the"} {unit}, ordered from the most isolated '
               + (f'({faults[0]["name"]}, which shares '
                  + ('no rupture with any other section' if faults[0]['connected'] == 0 else
                     f'ruptures with {faults[0]["connected"]} other sections')
                  + f') to the most connected ({faults[-1]["name"]}, which shares ruptures with {faults[-1]["connected"]}). '
                  if faults else 'to the most connected. '))
        if faults:
            n0 = sum(f['at0'] > 0 for f in faults)
            worst = min((f for f in faults if f['target'] > 0), key=lambda f: f['mean'] / f['target'])
            txt += (f'<b>{fref(fig_ss[0])}</b> shows each one\u2019s mean slip-rate prediction against its target. '
                    + (f'Every one has an event in every draw (the middle panel is 0% throughout). ' if n0 == 0 else
                       f'{n0} of them have draws without any event (the middle panel). ')
                    + f'The lowest ratio of prediction to target is {worst["name"]}\u2019s ({worst["mean"]:.2g} against '
                    f'{worst["target"]:.2g}, in units of its uncertainty). ')
        txt += (f'<b>{fref(*fig_ss[1:3])}</b> count each one\u2019s ruptures by rate: the posterior means spread rate over '
                'many ruptures at low rates, whereas a single draw switches on only a few ruptures per rate bin, at higher '
                'rates' + (f', as {refname} (black outline) does: a single model compares with a single draw, not with '
                           'the mean' if ref else '') + '. '
                f'<b>{fref(*fig_ss[3:5])}</b> sum each one\u2019s rupture rates by magnitude bin, its rate of events of each '
                'size' + ('; the circles are the posterior means, the x\u2019s the kept draws\u2019 sums, and the strip marked '
                          '0 shows by its opacity how often a draw has no event in the bin' if n_kept else '') + '.')
        para(txt)
        story += [NextPageTemplate('landscape'), PageBreak()]
        lmax = L_H - 2 * M
        figure(fig_ss[0], 'Each column\u2019s mean slip-rate prediction over its subsections (posterior mean: dot'
               + ('; kept draws: x\u2019s' if n_kept else '') + ('; reference: open square' if ref else '') + ') against its '
               'target (black line; grey: within one uncertainty), in units of its sections\u2019 uncertainties. Below: the '
               'share of draws in which no rupture crossing it is on, and the number of other sections sharing a rupture '
               'with it.', LW * 0.78, max_h=lmax - 2.2 * cm)
        figure(fig_ss[1], 'Rupture rates, first part: the ruptures crossing each column counted by their posterior mean '
               'rates (bars)' + (', each kept draw\u2019s counts of the ruptures on in it (x\u2019s)' if n_kept else '')
               + (', and the reference\u2019s rates of the same ruptures (black outline)' if ref else '') + '.', LW,
               max_h=lmax - 1.8 * cm)
        figure(fig_ss[2], 'Rupture rates, second part.', LW * 0.86, max_h=lmax - 1.4 * cm)
        figure(fig_ss[3], 'Rupture rates by magnitude, first part: each column\u2019s ruptures\u2019 rates summed by magnitude '
               'bin' + (' for each kept draw (x\u2019s), their posterior mean (circles) and the share of draws with no event '
                        'in each bin (the strip marked 0)' if n_kept else '')
               + (', and the reference\u2019s sums (black outline)' if ref else '') + '.', LW, max_h=lmax - 1.8 * cm)
        figure(fig_ss[4], 'Rupture rates by magnitude, second part.', LW * 0.86, max_h=lmax - 1.4 * cm)
        story += [NextPageTemplate('portrait'), PageBreak()]

    head('11. Output for OpenSHA')
    para(f'{tt("make_solution.py")} writes the posterior-mean rates as an OpenSHA fault-system solution\u2019s rates file '
         f'({tt(a.prior + "/rates.csv")}, one row per rupture of the rupture set, removed ruptures at zero), and the best '
         f'fit\u2019s as {tt("nnls/rates.csv")}. Before writing, it checks that the rates and the rupture set describe the '
         'same ruptures: '
         + (f'{sol_ok[0]:,} of {sol_ok[1]:,} kept ruptures cross exactly the fault sections the rupture set lists, and '
            'each one\u2019s magnitude lies in its bin. ' if sol_ok[0] is not None else 'see its report. ')
         + f'With {tt("--zip")} it also writes a complete archive that OpenSHA loads. Any representative model can be '
         'written the same way.')

    head('12. Points to keep in mind')
    pts = []
    if band and z_all is not None and z_all < -2:
        pts.append(f'<b>The fit check under the band.</b> χ² is {-z_all:.1f} standard deviations below the fit check, '
                   'because the band is stricter than the data where targets are weak; a fit check under the band would '
                   f'need its expected χ² from simulated data. The factor {band:g} is a modelling choice, and a run at '
                   'another factor would show how much the results depend on it.')
    elif tau and z_all is not None and z_all < -2:
        pts.append(f'<b>The fit check under the soft band.</b> χ² is {-z_all:.1f} standard deviations below the fit '
                   'check, which assumes the Gaussian likelihood alone; a fit check with the soft band would need its '
                   f'expected χ² from simulated data. Its width ({dec(tau)} at 1 sd) is a modelling choice, and a run at '
                   'another width would show how much the results depend on it.')
    elif z_all is not None and abs(z_all) > 2:
        pts.append(f'<b>The fit check.</b> χ² is {fmt(z_all, "{:+.1f}")} standard deviations from the fit check.')
    if p_med and (p_med < 0.85 or p_med > 1.15):
        pts.append(f'<b>Paleo rates.</b> The posterior mean {"under" if p_med < 1 else "over"}predicts them (median ratio '
                   f'{p_med:.2f}).')
    if sl:
        pts.append(f'<b>Slow mixing.</b> {rng([s[0] for s in sl])} ruptures per replicate, carrying '
                   f'{rng([s[3] for s in sl], "{:.1f}")}% of the total rate, mix slowly; their individual rates are less '
                   'precise than the rest, though sums over many ruptures are not affected.')
    if passes and not all(p == 'PASS' for p in passes):
        pts.append('<b>Convergence.</b> Not every replicate passes'
                   + (': ' + '; '.join(n.split('/')[-1] + ': ' + why_check[n] for n in names if n in why_check)
                      + ' (the report suggests longer runs).' if why_check else '; see the reasons in convergence_report.txt.'))
    pts.append(f'<b>Starts.</b> All {nrep * (ch or 0)} chains started from the same point (the best fit'
               + (' moved inside the band' if band else ', with ruptures switched on for positivity' if positive and pos_on
                  else '') + '); runs from dispersed starts remain untested.')
    if reps and s_verdict[1] and 'miss' in s_verdict[1]:
        pts.append(f'<b>Representative models.</b> The {reps} representatives miss some of their tolerances; the '
                   'posterior-mean rate vector is the exact input for mean hazard.')
    pts.append('<b>What the data determine.</b> Sums of rates (fault, subsection and bin rates) and the probability that a '
               'rupture exceeds about 10<super>−5</super> per year are well determined; individual rates below that are '
               'largely set by the prior.')
    para('<br/>'.join('• ' + p for p in pts))

    doc.build(story)
    print(f'wrote {out}: {len(present)} figures, {doc.page} pages')
    if MISSING:
        print('not found (shown as n/a, or left out): ' + ', '.join(dict.fromkeys(MISSING)))


if __name__ == '__main__':
    main()

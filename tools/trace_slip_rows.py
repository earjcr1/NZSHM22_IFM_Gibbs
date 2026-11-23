#!/usr/bin/env python3
"""trace_slip_rows.py -- a fault's slip-rate rows traced by hand, from its CFM line to the cleaned problem, independently of
cfm_sliprates.py, and every number checked against cru_clean.npz (REPORT.md, "The slip-rate constraint, step by step").

    python trace_slip_rows.py RUPTURE_DIR RUN_DIR/cru_clean.npz "Alpine: George to Jacksons" Kelly Hariki Teviot

Repeats steps 1-6 from the files alone: the rate fixes (data/cfm_rate_fixes.csv); the timeframe-corrected split normal
(data/cfm_timeframe_sliprates.csv: sn_centre, sn_sd_lo, sn_sd_hi); x the slip-rate factor 0.8; sigma_ref and the split
weights; the template groups (QualCode 2-5, identical triple as written, not in data/cfm_rate_sources.csv) and the link
clusters (the run's cfm_rate_links.csv, beside the problem); the section's 1/sqrt(k) and 1/sqrt(1 + m). Prints each
step's numbers and the largest relative difference from the cleaned problem over every kept row (row scale, target,
each coefficient, split weights, template weight). For a problem cleaned with the CFM, CFM_TIMEFRAME, CFM_TEMPLATE_RHO 1
and the links (the notebook's defaults)."""
import argparse, csv, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cru_common import find_rupture_set, load_problem
ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
ap.add_argument('rupture_dir'); ap.add_argument('problem'); ap.add_argument('faults', nargs='+')
ap.add_argument('--links', default=None, help="the run's cfm_rate_links.csv (default: beside the problem)")
ap.add_argument('--slip-rate-factor', type=float, default=0.8)
a = ap.parse_args()
rdir = find_rupture_set(a.rupture_dir); prob = a.problem; faults = a.faults
links_csv = a.links or os.path.join(os.path.dirname(os.path.abspath(prob)), 'cfm_rate_links.csv')
D = os.path.join(os.path.dirname(HERE), 'data') + '/'
cfm = {r['Name'].strip(): r for r in csv.DictReader(open(D + 'NZ_CFM_v1_0.csv', encoding='utf-8-sig'))}
tf = {r['Name'].strip(): r for r in csv.DictReader(open(D + 'cfm_timeframe_sliprates.csv', encoding='utf-8-sig'))}
fixes = {r['Name'].strip(): r['same_as'].strip() for r in csv.DictReader(open(D + 'cfm_rate_fixes.csv', encoding='utf-8-sig'))}
sources = {r['Name'].strip(): r['kind'].strip() for r in csv.DictReader(open(D + 'cfm_rate_sources.csv', encoding='utf-8-sig'))}
feats = json.load(open(rdir + '/fault_sections.geojson'))['features']
parent = [f['properties']['ParentName'].strip() for f in feats]
used = set(parent)
slip = np.loadtxt(rdir + '/average_slips.csv', delimiter=',', skiprows=1)[:, 1]
F = a.slip_rate_factor
trip = lambda p: tuple(cfm[fixes.get(p, p)][k].strip() for k in ('SR_pref', 'SR_min', 'SR_max'))
qc = lambda p: cfm[p]['QualCode'].strip()
# templates: QualCode 2-5, identical triple as written, not a listed own/borrowed rate
groups = {}
for p in used:
    if qc(p) in ('2', '3', '4', '5') and p not in sources:
        groups.setdefault(trip(p), set()).add(p)
mates = {p: set() for p in used}
for g in groups.values():
    if len(g) > 1:
        for p in g:
            mates[p] |= g - {p}
# link clusters from the run's cfm_rate_links.csv (union-find)
par = {p: p for p in used}
def root(x):
    while par[x] != x:
        x = par[x]
    return x
links = list(csv.DictReader(open(links_csv)))
for l in links:
    a, b = l['fault'].strip(), l['source'].strip()
    if a in used and b in used:
        par[root(a)] = root(b)
cl = {}
for p in used:
    cl.setdefault(root(p), set()).add(p)
for c in cl.values():
    for p in c:
        mates[p] |= c - {p}
P = load_problem(prob)
G = P['G'].tocsr(); d = P['d']; keep = np.asarray(P['keep_rows']); sec = np.asarray(P['section'])
raw2clean = {int(r): i for i, r in enumerate(keep)}
lo2, hi2, rs, ts = (np.asarray(P[k]) for k in ('split_lo2', 'split_hi2', 'row_scale', 'template_scale'))
for p in faults:
    c = cfm[p]
    t = tf[fixes.get(p, p)]
    cen, sl, sh = float(t['sn_centre']), float(t['sn_sd_lo']), float(t['sn_sd_hi'])
    pref, smin, smax = cen, cen - 2 * sl, cen + 2 * sh
    mode = F * pref * 1e-3
    s_lo, s_hi = F * (pref - smin) / 2 * 1e-3, F * (smax - pref) / 2 * 1e-3
    s_ref = 0.5 * (s_lo + s_hi)
    subs = [i for i, q in enumerate(parent) if q == p]
    rows = [raw2clean[i] for i in subs if i in raw2clean]
    secs = sorted(set(sec[rows]))
    k = len(rows) if len(secs) == 1 else None
    # section weight: sections holding the fault's mates or the fault itself, other than its own
    own = set(secs)
    near = set()
    for q in mates[p] | {p}:
        near |= {int(sec[raw2clean[i]]) for i, qq in enumerate(parent) if qq == q and i in raw2clean}
    near -= own
    w = 1 / np.sqrt(1 + len(near)) if near else 1.0
    scale = w / np.sqrt(k)
    print(f'\n== {p}: CFM {c["SR_pref"]} ({c["SR_min"]}-{c["SR_max"]}) mm/yr, QualCode {qc(p)}'
          + (f'; rate fix: takes {fixes[p]}\'s {", ".join(trip(p))}' if p in fixes else ''))
    print(f'   timeframe-corrected (Coffey et al.; fit {t["sn_fit"]}, t_eff {float(t["t_eff"]):.0f} yr, '
          f'{"periodic" if t["periodic"] == "1" else "aperiodic"}): mode {float(t["mode"]):.4g}, 2.5/16/50/84/97.5% '
          + '/'.join(f'{float(t[q]):.4g}' for q in ('q025', 'q16', 'q50', 'q84', 'q975'))
          + f'; split normal centre {cen:.4g}, sd below {sl:.4g}, above {sh:.4g} mm/yr')
    print(f'   x 0.8: mode {mode * 1e3:.4g} mm/yr, sd below {s_lo * 1e3:.4g}, above {s_hi * 1e3:.4g}, sigma_ref {s_ref * 1e3:.4g}; '
          f'split weights below {(s_ref / s_lo) ** 2:.4f}, above {(s_ref / s_hi) ** 2:.4f}')
    tg = [g for g in groups.values() if p in g and len(g) > 1]
    print(f'   correlated with {len(mates[p])} faults' + (f' (template group of {len(tg[0])})' if tg else ' (no template)')
          + f'; {len(subs)} subsections, {len(rows)} kept, {len(secs)} section(s) (k = {k}); sections correlated with it: {len(near)}; '
          f'row scale 1/sqrt(k) x 1/sqrt(1 + {len(near)}) = {scale:.6f}')
    ok = []
    for r in rows:
        a = G.indices[G.indptr[r]:G.indptr[r + 1]]; v = G.data[G.indptr[r]:G.indptr[r + 1]]
        cols = np.asarray(P['keep_cols'])[a]
        ok.append(max(abs(rs[r] / scale - 1), abs(d[r] / (mode / s_ref * scale) - 1),
                      np.max(np.abs(v / (slip[cols] / s_ref * scale) - 1)) if len(v) else 0.0,
                      abs(lo2[r] / (s_ref / s_lo) ** 2 - 1), abs(hi2[r] / (s_ref / s_hi) ** 2 - 1), abs(ts[r] / w - 1)))
    print(f'   against cru_clean.npz, every kept row (row scale, target, each coefficient, split weights, template weight): '
          f'max relative difference {max(ok):.1e}; target in sd units {d[rows[0]]:.4f}')

#!/usr/bin/env python3
"""make_solution.py -- a rate vector as the rates of an OpenSHA fault system solution.

    python make_solution.py nnls/nnls_vectors.npz --ruptures DATA/NZSHM22_InversionSolution-211k --out nnls
    python make_solution.py spikeslab/rate_vectors.npz --column mean --ruptures DATA/NZSHM22_InversionSolution-211k \
        --out spikeslab

The rupture set (--ruptures: an OpenSHA solution folder for the same model, or its ruptures/ folder) describes the
ruptures; a solution adds their annual rates. Writes OUT/rates.csv ('Rupture Index,Annual Rate', one row per rupture of
the rupture set, numbers as OpenSHA writes them), as in an OpenSHA solution's solution/rates.csv. --zip FILE also
writes a complete solution archive that OpenSHA loads: README, modules.json, the rupture set's files, and solution/
with these rates, its modules.json and, if the rupture set has it, sub_seismo_on_fault_mfds.csv with its module
(Sub-Seismogenic On-Fault MFDs; the NZSHM22 solutions carry the rupture set's copy). OpenSHA's build information and
the simulated annealing's progress are not written: they describe how OpenSHA made its own solution.

The rate vectors are those of preliminaries.py (nnls_vectors.npz or .csv: the best fit) or make_tables.py
(rate_vectors.npz or .csv: mean, median, mode, the quantiles and the representatives rep_01, rep_02, ...): one rate per
rupture of the inputs, in their numbering (the columns of cru_A, which are the rupture set's rupture indices), removed
ruptures 0. --column picks one (default: best_fit in a .csv that has it, otherwise mean; nnls_vectors.npz holds the best
fit as mean).

Before writing, the rate vector and the rupture set must describe the same ruptures: the same number, and, for every
rupture cleaning kept (--problem), the same fault sections (indices.csv against the rupture's slip-rate rows, row r
being section r) and a magnitude (properties.csv) inside its magnitude-bin row's bin, counted up from --lowest-bin in
steps of --bin-width as plot_rate_magnitude.py counts them; it stops otherwise. What it prints is also kept in
OUT/solution_report.txt."""
import argparse
import json
import os
import sys
import zipfile
from decimal import Decimal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cru_common import find_rupture_set, load_problem, magnitude_bins, report_to

SUBSEISMO = 'sub_seismo_on_fault_mfds.csv'
SUBSEISMO_MODULE = {'name': 'Sub-Seismogenic On-Fault MFDs',
                    'className': 'org.opensha.sha.earthquake.faultSysSolution.modules.SubSeismoOnFaultMFDs',
                    'path': 'solution/', 'assets': [SUBSEISMO]}
TOP_MODULES = [{'name': 'Rupture Set', 'className': 'org.opensha.sha.earthquake.faultSysSolution.FaultSystemRupSet',
                'path': 'ruptures/', 'assets': ['fault_sections.geojson', 'indices.csv', 'modules.json', 'properties.csv']},
               {'name': 'Solution', 'className': 'org.opensha.sha.earthquake.faultSysSolution.FaultSystemSolution',
                'path': 'solution/', 'assets': ['modules.json', 'rates.csv']}]
README = """This is an OpenSHA Fault System Solution zip file.

The file format is described in detail at https://opensha.org/Modular-Fault-System-Solution

Rupture information is stored in the 'ruptures/' sub-directory. Optional files may exist, but the core (required) files are:
 - ruptures/fault_sections.geojson: GeoJSON file listing the fault trace and properties of each fault section
 - ruptures/indices.csv: CSV file listing the fault section indices that comprise each rupture
 - ruptures/properties.csv: CSV file listing the properties of each rupture, including magnitude and rake

Rate information is stored in the 'solution/' sub-directory. Optional files may exist, but there is only one required file:
 - solution/rates.csv: CSV file giving the annual rate of occurrence for each rupture
"""


def java_double(x, single=False):
    """x as Java's Double.toString writes it, as OpenSHA's rates.csv has it: the shortest digits that read back as x
    (as a float32 for single precision), plain for 1e-3 <= |x| < 1e7 and d.dddE[-]n otherwise, always with a digit
    after the point."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f'not a finite rate: {x}')
    if x == 0.0:
        return '0.0'
    sign, digits, exp = Decimal(str(np.float32(x)) if single else repr(x)).as_tuple()
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:          # trailing zeros ('123.0') carry no digits
        digits.pop()
        exp += 1
    d = ''.join(map(str, digits))
    e10 = exp + len(d) - 1                               # x = d[0].d[1:] x 10^e10
    s = '-' if sign else ''
    if 1e-3 <= abs(x) < 1e7:
        if e10 >= 0:
            d = d + '0' * max(0, e10 + 1 - len(d))
            return f'{s}{d[:e10 + 1]}.{d[e10 + 1:] or "0"}'
        return f'{s}0.{"0" * (-e10 - 1)}{d}'
    return f'{s}{d[0]}.{d[1:] or "0"}E{e10}'


def read_vector(path, column):
    """One rate vector (original rupture numbering) and its penalised flags from a .npz or .csv of rate vectors."""
    if path.endswith('.npz'):
        z = np.load(path)
        col = column or 'mean'
        if col.startswith('rep_') and 'rep' in z.files:            # representatives: rep_01 is the first row of rep
            k = int(col[4:]) - 1
            if not 0 <= k < z['rep'].shape[0]:
                raise SystemExit(f'{path} holds {z["rep"].shape[0]} representatives; there is no {col}')
            v = z['rep'][k]
        elif col in z.files and z[col].ndim == 1:
            v = z[col]
        else:
            raise SystemExit(f'{path} has no column {col!r}; it has {[k for k in z.files if z[k].ndim == 1]}'
                             + (f' and rep_01 to rep_{z["rep"].shape[0]:02d}' if 'rep' in z.files else ''))
        pen = z['penalised'] if 'penalised' in z.files else None
    else:
        head = pd.read_csv(path, nrows=0).columns
        col = column or ('best_fit' if 'best_fit' in head else 'mean')
        if col not in head:
            raise SystemExit(f'{path} has no column {col!r}; it has {list(head)}')
        t = pd.read_csv(path, usecols=['rupture'] + (['penalised'] if 'penalised' in head else []) + [col])
        if not np.array_equal(t['rupture'].to_numpy(), np.arange(len(t))):
            raise SystemExit(f'{path}: the rupture column is not 0, 1, 2, ... in order')
        v = t[col].to_numpy()
        pen = t['penalised'].to_numpy() if 'penalised' in t else None
    return np.asarray(v), (None if pen is None else np.asarray(pen, int)), col


def read_rupture_set(folder):
    """Magnitudes (properties.csv) and section lists (indices.csv, as CSR arrays) of the rupture set's ruptures."""
    props = pd.read_csv(os.path.join(folder, 'properties.csv'))
    if not np.array_equal(props['Rupture Index'].to_numpy(), np.arange(len(props))):
        raise SystemExit(f'{folder}/properties.csv: rupture indices are not 0, 1, 2, ... in order')
    ptr, idx = [0], []
    with open(os.path.join(folder, 'indices.csv')) as f:
        next(f)
        for k, line in enumerate(f):
            v = line.rstrip('\n').split(',')
            if int(v[0]) != k or int(v[1]) != len(v) - 2:
                raise SystemExit(f'{folder}/indices.csv, rupture {k}: unexpected row {line[:80]!r}')
            idx.extend(int(s) for s in v[2:])
            ptr.append(len(idx))
    if len(ptr) - 1 != len(props):
        raise SystemExit(f'{folder}: indices.csv has {len(ptr) - 1} ruptures and properties.csv {len(props)}')
    return props['Magnitude'].to_numpy(), np.array(ptr), np.array(idx)


def check_alignment(P, mag, ptr, idx, slip_block, bins_block, lowest, width):
    """Stop unless every rupture cleaning kept has the rupture set's sections (its slip-rate rows, row r being section
    r) and a magnitude inside its bin row's bin."""
    G, kc, kr = sp.csc_matrix(P['G']), P['keep_cols'], P['keep_rows']
    if slip_block not in P['block_names']:
        raise SystemExit(f'the cleaned problem has no row block {slip_block!r} (--slip-block)')
    S = np.flatnonzero(P['block'] == P['block_names'].index(slip_block))
    GS, orig_S = G[S].tocsc(), kr[S]
    bad = [c for c in range(G.shape[1]) if not np.array_equal(np.sort(orig_S[GS.indices[GS.indptr[c]:GS.indptr[c + 1]]]),
                                                               np.sort(idx[ptr[kc[c]]:ptr[kc[c] + 1]]))]
    print(f'   sections: {G.shape[1] - len(bad):,} of {G.shape[1]:,} kept ruptures have exactly the sections indices.csv lists '
          f'(slip-rate row r = section r)')
    if bad:
        c = bad[0]
        raise SystemExit(f'rupture {kc[c]}: slip-rate rows {sorted(orig_S[GS.indices[GS.indptr[c]:GS.indptr[c + 1]]])[:8]} '
                         f'against sections {sorted(idx[ptr[kc[c]]:ptr[kc[c] + 1]])[:8]} in indices.csv ({len(bad)} ruptures '
                         f'differ): the rate vector and the rupture set do not describe the same ruptures')
    rows, _, _, lo = magnitude_bins(P, bins_block, lowest, width)
    Gr = sp.csr_matrix(G)
    binof = np.full(G.shape[1], -1)
    for j, r in enumerate(rows):
        binof[Gr.indices[Gr.indptr[r]:Gr.indptr[r + 1]]] = j
    M = mag[kc]
    ok = (binof >= 0) & (M >= lo[binof] - 1e-9) & (M < lo[binof] + width + 1e-9)
    print(f'   magnitudes: {int(ok.sum()):,} of {len(M):,} kept ruptures lie in their bin row\'s bin '
          f'({lo[0]:.2f}-{lo[-1] + width:.2f}, {width:g} wide)')
    if not ok.all():
        c = np.flatnonzero(~ok)[0]
        raise SystemExit(f'rupture {kc[c]}: magnitude {M[c]:.3f} in properties.csv, but its bin row is '
                         f'{lo[binof[c]]:.2f}-{lo[binof[c]] + width:.2f} ({int((~ok).sum())} ruptures differ): check '
                         f'--lowest-bin and --bin-width, or the rupture set')


def main():
    ap = argparse.ArgumentParser(description='A rate vector as an OpenSHA fault system solution (see the module docstring).')
    ap.add_argument('vectors', help='rate vectors: nnls_vectors.npz or .csv (preliminaries.py), rate_vectors.npz or .csv '
                                    '(make_tables.py)')
    ap.add_argument('--column', default=None, help="which rate vector: mean, median, mode, p02.275 (p02_275 in a .npz), "
                    "..., rep_01, rep_02, ... (default: best_fit if the .csv has it, otherwise mean)")
    ap.add_argument('--ruptures', required=True, help="the rupture set: an OpenSHA solution folder for the same model (e.g. "
                    "NZSHM22_InversionSolution-211k), or its ruptures/ folder (indices.csv, properties.csv, ...)")
    ap.add_argument('--out', required=True, help='folder for the rates (OUT/rates.csv) and the report')
    ap.add_argument('--problem', default='cru_clean.npz', help='the cleaned problem the rates belong to, for the checks '
                    '(default cru_clean.npz)')
    ap.add_argument('--slip-block', default='S')
    ap.add_argument('--bins-block', default='MFD')
    ap.add_argument('--lowest-bin', type=float, default=6.8, help='lower magnitude edge of the lowest bin row left after '
                    'cleaning (default 6.8)')
    ap.add_argument('--bin-width', type=float, default=0.1, help='magnitude width of every bin (default 0.1)')
    ap.add_argument('--zip', default=None, help='also write a complete solution archive (README, modules.json, ruptures/, '
                                                'solution/) to this file')
    a = ap.parse_args()
    report_to(os.path.join(a.out, 'solution_report.txt'))

    rs = find_rupture_set(a.ruptures, search=False)
    if rs is None:
        raise SystemExit(f'{a.ruptures}: no OpenSHA rupture set there (indices.csv, or ruptures/indices.csv)')
    v, pen, col = read_vector(a.vectors, a.column)
    single = v.dtype == np.float32                       # representatives are kept in single precision
    v = v.astype(np.float64)
    print(f'== rates: {a.vectors}, column {col}: {len(v):,} ruptures, {np.count_nonzero(v):,} nonzero, total '
          f'{v.sum():.6g}/yr, largest {v.max():.4g}')
    if not np.all(np.isfinite(v)) or (v < 0).any():
        raise SystemExit('the rates must be finite and nonnegative')
    mag, ptr, idx = read_rupture_set(rs)
    print(f'== rupture set: {rs}: {len(mag):,} ruptures, {len(idx):,} section memberships')
    if len(mag) != len(v):
        raise SystemExit(f'{len(v):,} rates for {len(mag):,} ruptures: the rate vector and the rupture set do not describe '
                         f'the same ruptures')
    P = load_problem(a.problem)
    if P.get('raw_shape') is not None and P['raw_shape'][1] != len(v):
        raise SystemExit(f'{a.problem} was cleaned from {P["raw_shape"][1]:,} ruptures, not {len(v):,}')
    kept = np.zeros(len(v), bool)
    kept[P['keep_cols']] = True
    if pen is not None and not np.array_equal(pen == 0, kept):
        raise SystemExit(f'the rate vectors\' penalised flags differ from the ruptures {a.problem} kept: another cleaning?')
    if (v[~kept] != 0).any():
        raise SystemExit(f'{int((v[~kept] != 0).sum())} ruptures that cleaning removed have nonzero rates')
    print(f'== checks against {a.problem} ({len(P["keep_cols"]):,} ruptures kept, {int((~kept).sum()):,} removed, all at 0)')
    check_alignment(P, mag, ptr, idx, a.slip_block, a.bins_block, a.lowest_bin, a.bin_width)

    rates = os.path.join(a.out, 'rates.csv')
    with open(rates, 'w', newline='\n') as f:
        f.write('Rupture Index,Annual Rate\n')
        f.writelines(f'{i},{java_double(x, single)}\n' for i, x in enumerate(v))
    print(f'== wrote {rates}')
    if a.zip:
        top = os.path.dirname(rs)
        sub = os.path.join(rs, SUBSEISMO)
        modules = [SUBSEISMO_MODULE] if os.path.exists(sub) else []
        with zipfile.ZipFile(a.zip, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, text in (('README', README), ('modules.json', json.dumps(TOP_MODULES, indent=2) + '\n')):
                p = os.path.join(top, name)                  # the rupture set's package's own copies, if beside it
                if os.path.exists(p):
                    z.write(p, name)
                else:
                    z.writestr(name, text)
            for name in sorted(os.listdir(rs)):
                if not name.startswith('.') and os.path.isfile(os.path.join(rs, name)):
                    z.write(os.path.join(rs, name), 'ruptures/' + name)
            z.write(rates, 'solution/rates.csv')
            z.writestr('solution/modules.json', json.dumps(modules, indent=2) + '\n')
            if modules:
                z.write(sub, 'solution/' + SUBSEISMO)
        print(f'== wrote {a.zip}: a complete solution archive ({os.path.getsize(a.zip) / 1e6:.0f} MB)'
              + ('' if modules else f'; the rupture set has no {SUBSEISMO}'))


if __name__ == '__main__':
    main()

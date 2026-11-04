#!/usr/bin/env python3
"""preliminaries.py -- load, clean and characterise the inverse problem  G m ~ d,  lo <= m <= hi.

    python preliminaries.py --A cru_A.txt --d cru_D.txt --out cru_clean.npz

Cleaning: (1) rows containing any entry with |value| > --huge (the 1e20/1e21 penalty rows that force parameters
to zero) are removed from G and d, together with the parameters they apply to; (2) all-zero rows and columns are
removed (rows that constrained only removed parameters, or were empty to begin with: they add only a constant to
chi^2), so N counts only rows that carry data.  --keep-empty-rows keeps them: same posterior, larger N.
Between the two, a row after the slip-rate rows with a target of 0 and a single entry left is a soft zero-forcing
penalty on that parameter (1e4 in the NZSHM22 inputs, a legacy weight): its parameter is forced to zero as the penalty
rows' are, and removed with the row. In the corrected 2e4 inputs every such row is on a parameter a penalty row already
removes; in the 211k inputs 301 are not. --keep-soft-penalties keeps them as the inputs give them (soft caps).
Slip-rate sections (--slip-sections, 'auto' by default): a fault section's slip rate, copied to each of its k
subsections, is one measurement, so the rows of each run of copies (identical targets and identical coefficients for
the ruptures they share) are divided by sqrt(k) and the measurement counts once; rows unlike their neighbours (k = 1,
as in a geodetic model) are unchanged. 'none' keeps every row as its own measurement (the inputs as given; a
sensitivity test); a file of 'original_row section' lines sets the sections instead. The cleaned problem records each
row's scale, its section and the expected chi^2 of a model that matches the truth (the fit check).
MFD uncertainty (--mfd-uncertainty, 'as-given' by default): the magnitude bins' uncertainties as the inputs give them,
each bin its own measurement. 'two-measurements' counts the bins as the projection of two measurements they are, the
rate of M>=5 earthquakes and the b-value, so that together they carry two measurements' worth of information, not one
per bin: every bin's uncertainty is multiplied by sqrt(B / 2) for B bins (3 for 18), that is, its row of G and d is
divided by it, as a slip-rate section's copies are divided by sqrt(k). A number sets the relative uncertainty (one sd)
of the M7.0-7.1 bin (0.1 reproduces the NZSHM22 inputs). Either way the bin-to-bin shape of the inputs' uncertainties is
kept (it is set where the inputs are built). Magnitudes count the bin rows up from --lowest-bin (6.80-6.90) in
steps of --bin-width (0.1) in their original order, as plot_rate_magnitude.py does. The cleaned problem records the
setting and the multiplier (mfd_uncertainty, mfd_scale), and row_scale includes it. The fit check still counts one
measurement per bin: a draw's bin totals scatter independently, within their widened uncertainties.
Writes the cleaned problem (--out), all diagnostics (prelim_results.npz), a summary (prelim_summary.txt), a figure
(prelim.png) and, in a folder of its own next to --out, nnls/, the best fit as rate vectors (nnls_vectors.npz: a
`mean` column in the original rupture numbering, as the notebook's rate vectors, for plot_rate_magnitude.py,
plot_slip_rates.py, plot_paleo_rates.py and make_solution.py; nnls_vectors.csv: the same as a table with the rows of
rate_vectors.csv, one per rupture of the inputs, columns rupture, penalised and best_fit, removed ruptures 0); for
--out other than cru_clean.npz these names get the --out stem as a prefix. Copies left beside --out by the layout
before nnls/ are removed.
Nothing here forms a dense copy of G or anything n x n: the near-duplicate search compares each column only with the
columns that can pass its cosine threshold (an exact pruning), the best fit works on the sparse G with an updated QR
factorization, and the resolution diagonal takes G a block of columns at a time. At 2e5 ruptures (2,395 rows) the
whole step takes about a minute on an M2, with 3 GB of memory: the best fit 25 s, the near-duplicate search 13 s.
"""
import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from scipy.linalg import qr, qr_delete, qr_insert, solve_triangular
from scipy.sparse.csgraph import connected_components

from cru_common import (load_colt_sparse, load_vector, save_problem, load_problem, column_norms, describe, detect_bin_rows,
                        check_bin_rows, check_slip_rows, relabel_rows, slip_sections, chi2_expectation, mfd_uncertainty,
                        scale_rows)

LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# --------------------------------------------------------------------------- cleaning
def clean(A, d, huge, drop_empty_rows=True, force_soft=True):
    """1) remove the penalty rows (any |entry| > huge) from G and d, and the parameters (columns) they apply to;
    1b) after the slip-rate rows (which end at the first zero-forcing row), a row with a target of 0 and a single entry
    left is a soft zero-forcing penalty on that parameter: if force_soft, the parameter is forced to zero as the penalty
    rows' are (removed), and the row with it;
    2) remove all-zero columns, and all-zero rows if drop_empty_rows.  Returns G, d, kept indices, info."""
    A = sp.csr_matrix(A)
    N, n = A.shape
    pen_r = np.flatnonzero(np.asarray((abs(A) > huge).sum(1)).ravel() > 0)
    pen_c = np.unique(A[pen_r].indices)
    raw_empty = np.flatnonzero(np.diff(A.indptr) == 0)
    keep_r = np.setdiff1d(np.arange(N), pen_r)
    keep_c = np.setdiff1d(np.arange(n), pen_c)
    G = A[keep_r][:, keep_c].tocsr()
    step1 = G.shape
    zr = np.flatnonzero(np.diff(G.indptr) == 0)
    emptied = np.setdiff1d(keep_r[zr], raw_empty)
    cand = np.concatenate([pen_r, raw_empty, emptied])   # zero-forcing rows so far (target 0): the first ends the slip rates
    cand = cand[d[cand] == 0]
    first = int(cand.min()) if cand.size else 0
    one = keep_r[np.diff(G.indptr) == 1]
    soft_r = one[(d[one] == 0) & (one >= first)]
    soft_c = np.unique(A[soft_r][:, keep_c].indices) if len(soft_r) else np.zeros(0, int)
    soft = dict(soft_r=soft_r, soft_c=keep_c[soft_c] if len(soft_r) else soft_c, soft_forced=bool(force_soft),
                soft_values=np.unique(A[soft_r][:, keep_c].data) if len(soft_r) else np.array([]))
    if force_soft and len(soft_r):
        keep_r = np.setdiff1d(keep_r, soft_r)
        keep_c = np.setdiff1d(keep_c, soft['soft_c'])
        G = A[keep_r][:, keep_c].tocsr()
        zr = np.flatnonzero(np.diff(G.indptr) == 0)
    zc = np.flatnonzero(np.diff(G.tocsc().indptr) == 0)
    info = dict(pen_r=pen_r, pen_c=pen_c, pen_values=np.unique(A[pen_r].data), step1=step1, raw_empty=raw_empty,
                emptied=emptied, emptied_entries=np.unique(A[emptied].data) if len(emptied) else np.array([]),
                emptied_on_removed=bool(np.isin(A[emptied].indices, pen_c).all()) if len(emptied) else True,
                zero_rows=keep_r[zr], zero_cols=keep_c[zc], chi2_const=float(np.sum(d[keep_r[zr]] ** 2)),
                step1b=G.shape, **soft)
    if drop_empty_rows:
        keep_r = np.delete(keep_r, zr)
    keep_c = np.delete(keep_c, zc)
    G = A[keep_r][:, keep_c].tocsc()
    info['negatives'] = int(np.sum(G.data < 0))
    return G, d[keep_r], keep_r, keep_c, info


def detect_slip_rows(d_raw, info, keep_r):
    """The slip-rate rows: the leading rows of the raw matrix, up to the first zero-forcing penalty row, that is, the
    first row whose right-hand side is 0 and whose entries are all on penalised ruptures (or which has none). A
    slip-rate row with a zero right-hand side still involves unpenalised ruptures, so it does not end the block.
    Returns (cleaned indices of the block's rows, original boundary row, first zero in the raw d) or None."""
    cand = np.concatenate([np.asarray(info[k], dtype=int).ravel() for k in ('emptied', 'raw_empty', 'pen_r')])
    cand = cand[d_raw[cand] == 0] if cand.size else cand
    if cand.size == 0 or cand.min() == 0:
        return None
    end = int(cand.min())
    rows = np.flatnonzero(keep_r < end)
    zeros = np.flatnonzero(d_raw == 0)
    return (rows, end, int(zeros[0]) if zeros.size else None) if rows.size else None


def row_blocks(G, keep_r, spec=None, min_rows=5):
    """Row blocks for reporting.  spec 'S=0-387,MFD=695-713,...' (original row numbers) names them explicitly;
    otherwise contiguous runs of rows whose nonzeros share one value ('const') or vary ('var'), with runs shorter
    than min_rows merged into the preceding block.  All-zero rows always form one block named 'empty'."""
    nr = G.shape[0]
    Gr = sp.csr_matrix(G)
    empty = np.diff(Gr.indptr) == 0
    block = np.full(nr, -1, dtype=np.int32)
    names, ranges = [], []
    if spec:
        for b, item in enumerate(spec.split(',')):
            nm, rg = item.split('=')
            a, z = map(int, rg.split('-'))
            block[(keep_r >= a) & (keep_r <= z) & ~empty] = b
            names.append(nm.strip())
            ranges.append(rg.strip())
        if ((block < 0) & ~empty).any():
            block[(block < 0) & ~empty] = len(names)
            names.append('other')
            ranges.append('rest')
    else:
        kind = []
        for j in range(nr):
            v = Gr.data[Gr.indptr[j]:Gr.indptr[j + 1]]
            kind.append('empty' if len(v) == 0 else ('const' if np.ptp(v) <= 1e-9 * abs(v).max() else 'var'))
        runs = []                                   # [start, stop, kind] over non-empty rows
        prev = None
        for j in np.flatnonzero(~empty):
            if runs and kind[j] == runs[-1][2] and keep_r[j] == keep_r[prev] + 1 and prev == j - 1:
                runs[-1][1] = j + 1
            else:
                runs.append([j, j + 1, kind[j]])
            prev = j
        merged = []
        for r in runs:
            adj = merged and keep_r[r[0]] == keep_r[merged[-1][1] - 1] + 1 and r[0] == merged[-1][1]
            if adj and (r[1] - r[0] < min_rows or r[2] == merged[-1][2]):
                merged[-1][1] = r[1]
            else:
                merged.append(list(r))
        for b, (a, z, k) in enumerate(merged):
            block[a:z] = b
            names.append(f'B{b}:{k}')
            ranges.append(f'{keep_r[a]}-{keep_r[z - 1]}')
    if empty.any():
        block[empty] = len(names)
        names.append('empty')
        ranges.append(f'{int(empty.sum())} rows')
    return block, names, ranges


# --------------------------------------------------------------------------- best fit
def nnls_lawson_hanson(G, d, tol=None, max_outer=None, threads=None):
    """Lawson-Hanson active-set NNLS for a wide G (N << n), sparse or dense. Never forms G^T G or a dense copy of G: the
    gradients are sparse products, split by column blocks over threads (the same sums as G.T @ r), the predictions sum
    over the nonzero rates only (the same sums as G @ x, without the zero terms), and the passive set's least-squares
    problem is solved from a QR factorization of its columns, updated as columns enter and leave (scipy.linalg.qr_insert
    / qr_delete) rather than recomputed."""
    G = sp.csc_matrix(G)
    N, n = G.shape
    threads = threads or min(4, os.cpu_count() or 1)
    cuts = np.linspace(0, n, threads + 1).astype(int)
    GT = [sp.csr_matrix(G[:, c0:c1].T) for c0, c1 in zip(cuts[:-1], cuts[1:])]
    x = np.zeros(n)
    P = np.zeros(n, bool)
    idx, QR = [], [None, None]                       # passive columns, in factorization order, and the QR of G[:, idx]

    def insert(j):                                   # returns False if column j is numerically dependent on the others
        u = G[:, j].toarray().ravel()
        if idx:
            QR[:] = qr_insert(QR[0], QR[1], u, len(idx), which='col', overwrite_qru=True, check_finite=False)
        else:
            QR[:] = qr(u[:, None], check_finite=False)
        idx.append(j)
        return abs(QR[1][len(idx) - 1, len(idx) - 1]) > 1e-12 * np.linalg.norm(u)

    def delete(p):
        if len(idx) == 1:
            QR[:] = [None, None]
        else:
            QR[:] = qr_delete(QR[0], QR[1], p, which='col', overwrite_qr=True, check_finite=False)
        del idx[p]

    def solve():
        k = len(idx)
        return solve_triangular(QR[1][:k, :k], QR[0][:, :k].T @ d, check_finite=False)

    with ThreadPoolExecutor(threads) as pool:
        def gradient(res):                           # G^T res, one block of columns per thread
            w = np.empty(n)

            def part(k):
                w[cuts[k]:cuts[k + 1]] = GT[k] @ res
            list(pool.map(part, range(threads)))
            return w

        w = gradient(d)
        if tol is None:
            tol = 1e-9 * np.abs(w).max()
        tabu = -1
        for it in range(max_outer or 3 * n):
            wc = np.where(P, -np.inf, w)
            if tabu >= 0:
                wc[tabu] = -np.inf
            j = int(np.argmax(wc))
            if not wc[j] > tol:
                break
            P[j] = True
            independent = insert(j)
            first = True
            while True:
                if not idx:                          # every passive column left: the solution is 0, as an empty solve gives
                    x[:] = 0.0
                    break
                s = solve() if independent else None
                if s is not None and np.all(s > 0):
                    x[:] = 0.0
                    x[idx] = s
                    tabu = -1
                    break
                if first and (s is None or s[-1] <= 0):  # numerically degenerate entry (j is the last column): skip it this round
                    delete(len(idx) - 1)
                    P[j] = False
                    tabu = j
                    break
                first = False
                ia = np.array(idx)
                neg = s <= 0
                xi = x[ia]
                ratio = xi[neg] / (xi[neg] - s[neg])
                alpha = np.min(ratio)
                x[ia] = xi + alpha * (s - xi)
                x[ia[np.flatnonzero(neg)[ratio == alpha]]] = 0.0     # the blocking rates leave exactly: rounding cannot leave
                for p in np.flatnonzero(x[ia] <= 0)[::-1]:   # from the end, so the positions of the others stay valid
                    P[idx[p]] = False
                    delete(int(p))
                x[~P] = 0.0
            nz = np.flatnonzero(x)
            w = gradient(d - G[:, nz] @ x[nz])
    return x, it


# --------------------------------------------------------------------------- near-duplicate columns
def near_duplicates(G, cn, thresholds=(0.99, 0.999, 0.9999), chunk=4096):
    """Pairs of columns with cosine similarity above min(thresholds), and the groups they tie together at each threshold.
    Exact, without comparing every pair: two unit columns with cosine above t differ by less than delta = sqrt(2 (1 - t))
    in every entry, so every column that pairs with a column holding an entry of at least delta (a heavy entry) is
    nonzero in that entry's row. Each column with a heavy entry is filed under its heavy row with the fewest columns and
    compared with that row's columns only, over the rows its chunk of columns touches (the only rows its cosines need);
    flat columns (no heavy entry) are compared with one another. Never forms a dense copy of G."""
    N, n = G.shape
    Gn = sp.csc_matrix(sp.csc_matrix(G) @ sp.diags(1.0 / np.where(cn > 0, cn, 1.0)))
    Gn.sort_indices()
    Gr = sp.csr_matrix(Gn)                           # each row's columns
    tmin = min(thresholds)
    delta = np.sqrt(2.0 * (1.0 - tmin))
    lens = np.diff(Gn.indptr)
    seg = np.repeat(np.arange(n), lens)              # each entry's column
    rc = np.diff(Gr.indptr)                          # each row's number of columns
    hv = np.flatnonzero(Gn.data >= delta)            # heavy entries
    hv = hv[np.lexsort((Gn.indices[hv], rc[Gn.indices[hv]], seg[hv]))]   # by column, then fewest columns, then row
    first = hv[np.unique(seg[hv], return_index=True)[1]]
    frow = np.full(n, -1)
    frow[seg[first]] = Gn.indices[first]             # each column's filing row: its heavy row with the fewest columns
    flat = (frow < 0) & (lens > 0)
    pos = np.full(N, -1)                             # the rows in use, numbered
    I, J, C, work = [], [], [], [0]

    def dense(S, U):                                 # sparse columns S on the rows U, as a dense array
        k = pos[S.indices]
        ok = k >= 0
        D = np.zeros((len(U), S.shape[1]))
        D[k[ok], np.repeat(np.arange(S.shape[1]), np.diff(S.indptr))[ok]] = S.data[ok]
        return D

    def compare(A, B, keep):                         # cosines of columns A against columns B; keep(i, j) selects pairs
        for a0 in range(0, len(A), chunk):
            a = A[a0:a0 + chunk]
            Sa = Gn[:, a]
            U = np.unique(Sa.indices)                # the rows these columns touch
            pos[U] = np.arange(len(U))
            Da = dense(Sa, U)
            for b0 in range(0, len(B), 4 * chunk):
                b = B[b0:b0 + 4 * chunk]
                Cb = Da.T @ dense(Gn[:, b], U)
                work[0] += Cb.size
                ii, jj = np.nonzero(Cb > tmin)
                ok = keep(a[ii], b[jj])
                I.append(a[ii[ok]]), J.append(b[jj[ok]]), C.append(Cb[ii[ok], jj[ok]])
            pos[U] = -1

    order = np.argsort(frow, kind='stable')          # columns with a heavy entry, grouped by filing row
    nf = order[frow[order] >= 0]
    for A in np.split(nf, np.flatnonzero(np.diff(frow[nf])) + 1):
        if A.size:
            r = frow[A[0]]
            compare(A, Gr.indices[Gr.indptr[r]:Gr.indptr[r + 1]], lambda i, j: (j > i) | flat[j])  # each pair once; a flat
    F = np.flatnonzero(flat)                                                                     # partner never finds it
    if F.size:
        compare(F, F, lambda i, j: j > i)
    I = np.concatenate(I) if I else np.zeros(0, int)
    J = np.concatenate(J) if J else np.zeros(0, int)
    C = np.concatenate(C) if C else np.zeros(0)
    I, J = np.minimum(I, J), np.maximum(I, J)
    s = np.lexsort((J, I))                           # by first column, then second, as a full search lists them
    I, J, C = I[s], J[s], C[s]
    out = {'pairs_i': I, 'pairs_j': J, 'pairs_cos': C, 'flat_columns': int(F.size),
           'computed': work[0] / max(n * (n - 1) / 2, 1)}   # cosines computed, as a share of a full search's
    for t in thresholds:
        sel = C > t
        _, roots = connected_components(sp.coo_matrix((np.ones(sel.sum()), (I[sel], J[sel])), shape=(n, n)), directed=False)
        sizes = np.bincount(roots)
        sizes = sizes[sizes > 1]
        out[f'cos>{t}'] = dict(pairs=int(sel.sum()), cols_in_groups=int(sizes.sum()), groups=int(len(sizes)),
                               largest=int(sizes.max()) if len(sizes) else 0)
    out['exact_proportional_pairs'] = int(np.sum(C > 1 - 1e-12))
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--A', default='cru_A.txt')
    ap.add_argument('--d', default='cru_D.txt')
    ap.add_argument('--huge', type=float, default=1e10, help='entries above this mark columns forced to zero')
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--lo', type=float, default=0.0)
    ap.add_argument('--hi', type=float, default=1.0)
    ap.add_argument('--out', default='cru_clean.npz')
    ap.add_argument('--blocks', default=None, help="optional names for row blocks, e.g. 'S=0-387,MFD=695-713,P=714-753' (original row numbers); "
                    "the slip-rate and magnitude-bin rows are detected automatically if not named, and checked if named")
    ap.add_argument('--bins-block', default='MFD', help='name of the magnitude-bin rows (detected automatically if --blocks does not name them)')
    ap.add_argument('--diag-floor', type=float, default=-10.0, help='log10 of the diagnostic and reporting cutoff (the notebook\'s '
                    'FLOOR, -10 by default), compared with the collective negligibility scale')
    ap.add_argument('--slip-block', default='S', help='name of the slip-rate rows, checked for plausibility if named in --blocks')
    ap.add_argument('--slip-sections', default='auto',
                    help="how the slip-rate rows count as measurements: 'auto' (default): each run of subsections holding copies "
                    "of one slip rate (identical targets, identical coefficients for the ruptures they share) is one "
                    "section's measurement, so its k rows are divided by sqrt(k); rows unlike their neighbours are "
                    "unchanged; 'none': every row its own measurement (the inputs as given; a sensitivity test); or a file "
                    "of 'original_row section' lines giving every slip-rate row's section")
    ap.add_argument('--mfd-uncertainty', default='as-given',
                    help="the magnitude bins' uncertainty level, their bin-to-bin shape kept as the inputs give it: "
                    "'as-given' (default): the inputs' uncertainties, each bin its own measurement; 'two-measurements': "
                    "the bins are a projection of two measurements (the rate of M>=5 and the b-value), so every bin's "
                    "uncertainty is multiplied by sqrt(bins / 2) and together they count as two measurements; or a number: "
                    "the relative uncertainty (one sd) of the M7.0-7.1 bin (0.1 reproduces the NZSHM22 inputs)")
    ap.add_argument('--lowest-bin', type=float, default=6.8, help='lower magnitude edge of the lowest bin row left after '
                    'cleaning (default 6.8), for finding the M7.0-7.1 bin, as in plot_rate_magnitude.py')
    ap.add_argument('--bin-width', type=float, default=0.1, help='magnitude width of every bin (default 0.1)')
    ap.add_argument('--no-duplicates', action='store_true', help='skip the near-duplicate search (about 13 s at 2e5 '
                    'ruptures); gibbs.py --swap-cos then finds no pairs')
    ap.add_argument('--keep-soft-penalties', action='store_true',
                    help='keep the rows after the slip-rate rows that have a target of 0 and a single entry (soft zero-forcing '
                         'penalties, 1e4 in the NZSHM22 inputs) as the inputs give them, instead of forcing their ruptures to 0 '
                         'as the penalty rows do (they then cap those ruptures softly, and the fit check leaves them out)')
    ap.add_argument('--cfm-sliprates', default=None, metavar='CSV',
                    help='rebuild the slip-rate rows from the NZ Community Fault Model (NZ_CFM_v1_0.csv: SR_pref, SR_min, '
                         'SR_max per parent fault), each a split normal in mm/yr with its mode at the inputs\' factor (0.8) '
                         'x SR_pref (cfm_sliprates.py); needs --rupture-set')
    ap.add_argument('--rupture-set', default=None, help='the OpenSHA rupture set (for --cfm-sliprates)')
    ap.add_argument('--cfm-sigmas', type=float, default=1.0,
                    help='--cfm-sliprates: how many sigma SR_min and SR_max lie from SR_pref (default 1, as the inputs read them)')
    ap.add_argument('--cfm-factor', type=float, default=2.0,
                    help='--cfm-sliprates: each side\'s sd multiplied by this (default 2, the inputs\' doubling)')
    ap.add_argument('--slip-rate-factor', type=float, default=None,
                    help='the assumed fraction of each fault\'s slip rate the ruptures take up, scaling each slip-rate '
                         'distribution as a whole, target and sd alike (default: the inputs\' own, 0.8 in NZSHM22, as '
                         'they are); needs --rupture-set')
    ap.add_argument('--keep-empty-rows', action='store_true',
                    help='keep all-zero rows (same posterior; N and a constant chi^2 change)')
    args = ap.parse_args()
    t0 = time.time()
    sig, u = args.sigma, args.hi - args.lo
    stem = os.path.splitext(args.out)[0]                 # side outputs are named after --out
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pre = '' if os.path.basename(stem) == 'cru_clean' else stem + '_'
    f_res, f_sum, f_png, old_vec, old_csv = (os.path.join(os.path.dirname(stem), pre + f) if not pre else pre + f
                                             for f in ('prelim_results.npz', 'prelim_summary.txt', 'prelim.png',
                                                       'nnls_vectors.npz', 'nnls_vectors.csv'))
    d_nnls = os.path.join(os.path.dirname(stem), 'nnls')   # the best fit's outputs, in a folder of their own
    f_vec, f_csv = (os.path.join(d_nnls, os.path.basename(p)) for p in (old_vec, old_csv))

    A = load_colt_sparse(args.A)
    d_raw = load_vector(args.d)
    say(f'== raw: G {A.shape[0]} x {A.shape[1]}, nnz {A.nnz}; d {d_raw.shape}; min entry {A.data.min():.4g}, max {A.data.max():.4g}')
    assert A.shape[0] == len(d_raw), 'rows of G must match length of d'
    L2_raw = H2_raw = None                               # --cfm-sliprates: the split misfit's weights, per raw row
    if args.cfm_sliprates or args.slip_rate_factor is not None:
        from cfm_sliprates import rebuild, rescale_inputs
        from cru_common import find_rupture_set
        what = '--cfm-sliprates' if args.cfm_sliprates else '--slip-rate-factor'
        rd_c = find_rupture_set(args.rupture_set) if args.rupture_set else None
        if rd_c is None:
            raise SystemExit(f'{what} needs --rupture-set (the OpenSHA rupture set: its fault_sections.geojson, '
                             'indices.csv and average_slips.csv)')
        if args.slip_rate_factor is not None and not args.slip_rate_factor > 0:
            raise SystemExit('--slip-rate-factor must be positive')
        try:
            if args.cfm_sliprates:
                A, d_raw, L2_raw, H2_raw, _, lines_c = rebuild(A, d_raw, rd_c, args.cfm_sliprates, args.cfm_sigmas,
                                                               args.cfm_factor, args.slip_rate_factor)
            else:
                A, lines_c = rescale_inputs(A, d_raw, rd_c, args.slip_rate_factor)
        except ValueError as e:
            raise SystemExit(f'{what}: {e}')
        for ln in lines_c:
            say(ln)

    G, d, keep_r, keep_c, info = clean(A, d_raw, args.huge, not args.keep_empty_rows, not args.keep_soft_penalties)
    N, n = G.shape
    pr = info['pen_r']
    say(f'== cleaning step 1: {len(pr)} penalty rows contain entries > {args.huge:g} (values {info["pen_values"]}; original rows '
        f'{pr.tolist() if len(pr) <= 20 else f"{pr.min()}-{pr.max()}"}); they apply to {len(info["pen_c"])} parameters.')
    say(f'   removing those rows from G and d and those parameters from G: {info["step1"][0]} x {info["step1"][1]}')
    if len(info['soft_r']):
        sr = info['soft_r']
        say(f'== cleaning step 1b: {len(sr)} rows after the slip-rate rows have a target of 0 and a single entry left (values '
            f'{info["soft_values"]}; original rows {sr.min()}-{sr.max()}): soft zero-forcing penalties on {len(info["soft_c"])} '
            'parameters that no penalty row covers')
        say(f'   forcing those parameters to zero, as the penalty rows do, and removing them and the rows: {info["step1b"][0]} x '
            f'{info["step1b"][1]}' if info['soft_forced'] else '   kept as the inputs give them (--keep-soft-penalties): each caps '
            'its parameter softly, and the fit check leaves them out, since they hold no data')
    say(f'== cleaning step 2: {len(info["zero_rows"])} all-zero rows and {len(info["zero_cols"])} all-zero columns remain')
    if len(info['raw_empty']):
        say(f'   {len(info["raw_empty"])} rows were empty in the raw matrix (original rows {info["raw_empty"].tolist()}, d = {d_raw[info["raw_empty"]].tolist()})')
    if len(info['emptied']):
        say(f'   {len(info["emptied"])} rows (original {info["emptied"].min()}-{info["emptied"].max()}) held only entries {info["emptied_entries"]} '
            f'on removed parameters{"" if info["emptied_on_removed"] else " (and others!)"}, d = {np.unique(d_raw[info["emptied"]])}: a second zero-forcing penalty')
    say(f'   all-zero rows add a constant chi^2 = {info["chi2_const"]:.1f} that no model can change; '
        + ('they are kept (--keep-empty-rows), so N counts them' if args.keep_empty_rows else 'they are removed (default)'))
    say(f'   cleaned G: {N} x {n}, nnz {G.nnz} (density {G.nnz / N / n:.3f}); negative entries {info["negatives"]}')
    block, bnames, branges = row_blocks(G, keep_r, args.blocks)
    det = detect_bin_rows(G, d)                          # magnitude-bin rows: constant coefficients partitioning the ruptures
    if args.bins_block in bnames:
        named = np.flatnonzero(block == bnames.index(args.bins_block))
        if det is not None and not np.array_equal(np.sort(named), np.sort(det)):
            raise SystemExit(f'row block {args.bins_block!r} as named in --blocks (original rows {keep_r[named].min()}-{keep_r[named].max()}, '
                             f'{len(named)} rows) differs from the rows detected as magnitude bins (original rows '
                             f'{keep_r[det].min()}-{keep_r[det].max()}, {len(det)} rows); fix --blocks, or leave the bins out of it')
        say(f'   magnitude-bin rows: block {args.bins_block!r} as named'
            + (', confirmed by automatic detection' if det is not None else ' (automatic detection found no unique set to compare)'))
    elif det is not None:
        block, bnames, branges = relabel_rows(block, bnames, branges, det, args.bins_block, keep_r)
        say(f'   magnitude-bin rows detected automatically: original rows {keep_r[det].min()}-{keep_r[det].max()} '
            f'({len(det)} rows), labelled {args.bins_block!r}')
    else:
        say('   no magnitude-bin rows detected; the per-bin and hierarchical priors need them named in --blocks')
    sd = detect_slip_rows(d_raw, info, keep_r)          # slip-rate rows: leading rows up to the first zero-forcing row
    if sd is not None and det is not None and np.intersect1d(sd[0], det).size:
        sd = None                                        # overlaps the bin rows: not a usable detection
    if args.slip_block in bnames:
        named = np.flatnonzero(block == bnames.index(args.slip_block))
        if sd is not None and not np.array_equal(np.sort(named), np.sort(sd[0])):
            raise SystemExit(f'row block {args.slip_block!r} as named in --blocks (original rows {keep_r[named].min()}-{keep_r[named].max()}, '
                             f'{len(named)} rows) differs from the rows detected as slip rates (original rows {keep_r[sd[0]].min()}-'
                             f'{keep_r[sd[0]].max()}, {len(sd[0])} rows, ending before the first zero-forcing penalty row, original '
                             f'{sd[1]}); fix --blocks, or leave the slip-rate rows out of it')
        say(f'   slip-rate rows: block {args.slip_block!r} as named'
            + (', confirmed by automatic detection' if sd is not None else ' (automatic detection found no boundary to compare)'))
    elif sd is not None:
        srows, send, fz = sd
        block, bnames, branges = relabel_rows(block, bnames, branges, srows, args.slip_block, keep_r)
        say(f'   slip-rate rows detected automatically: original rows {keep_r[srows].min()}-{keep_r[srows].max()} ({len(srows)} rows), '
            f'ending before the first zero-forcing penalty row (original {send}), labelled {args.slip_block!r}'
            + ('' if fz in (None, send) else f'; note: the first zero right-hand side, original row {fz}, is a slip-rate row with '
               'zero slip (it involves unpenalised ruptures), so it does not end the block'))
    if args.slip_block in bnames:
        try:
            say('   slip-rate rows: ' + check_slip_rows(dict(G=G, d=d, block=block, block_names=bnames, keep_rows=keep_r), args.slip_block))
        except ValueError as e:
            raise SystemExit(f'slip-rate rows: {e}; fix --blocks')
    else:
        say(f'   no slip-rate rows detected and none named {args.slip_block!r}: name them in --blocks for the subsection summaries and transfer moves')
    for b, (nm, rg) in enumerate(zip(bnames, branges)):
        k = block == b
        say(f'   row block {nm:9s} {"original rows " + rg if nm != "empty" else "all-zero rows":22s}: {k.sum():3d} rows, d in [{d[k].min():.3g}, {d[k].max():.3g}], ||d||^2 = {d[k] @ d[k]:.0f}')

    # ---- slip-rate sections: one measurement per section (copies of a section's slip rate divided by sqrt(k))
    row_scale, section = np.ones(N), np.full(N, -1, np.int64)
    nonempty = np.diff(sp.csr_matrix(G).indptr) > 0
    n_soft = 0 if info['soft_forced'] else int(np.isin(keep_r, info['soft_r']).sum())   # kept soft penalties: no data
    const = info['chi2_const'] if args.keep_empty_rows else 0.0
    spec = args.slip_sections
    if args.slip_block in bnames:
        srows = np.flatnonzero(block == bnames.index(args.slip_block))
        try:
            sec, kr, notes = slip_sections(G, d, srows, keep_r, spec)
        except (OSError, ValueError) as e:
            raise SystemExit(f'--slip-sections: {e}')
        section[srows] = sec
        ks = np.bincount(sec)
        weigh = spec != 'none'
        if weigh:
            row_scale[srows] = 1.0 / np.sqrt(kr)
        first = np.r_[0, np.flatnonzero(np.diff(sec)) + 1] if spec in ('auto', 'none') else np.unique(sec, return_index=True)[1]
        _, inv, cnt = np.unique(np.round(d[srows[first]], 12), return_inverse=True, return_counts=True)
        shared = int(np.sum(cnt[inv] > 1))               # sections whose target another section also has
        how = 'runs of subsections holding copies of one slip rate' if spec in ('auto', 'none') else f'from {spec}'
        sizes = (f'{len(ks)} sections ({how}: {int((ks > 1).sum())} with 2-{ks.max()} subsections, median {np.median(ks[ks > 1]):.0f}; '
                 f'{int((ks == 1).sum())} single subsections)' if (ks > 1).any() else f'{len(ks)} sections, all single subsections')
        if weigh:
            say(f'== slip-rate sections ({spec!r}): {len(srows)} rows in {sizes}; each section\'s rows are divided by sqrt(k), so its '
                'measurement counts once; single subsections are unchanged')
        else:
            say(f'== slip-rate sections ({spec!r}): every slip-rate row counts as its own measurement (the inputs as given), so in '
                f'{len(srows)} rows each of the {sizes} counts its measurement once per subsection')
        if spec == 'auto' and shared:
            say(f'   note: {shared} sections share their target with another section; some may be one section split in two, or '
                'neighbouring sections whose slip rates happen to be equal; a mapping file (--slip-sections FILE) settles it')
        for nt in notes:
            say('   note: ' + nt)
        if weigh:
            G = sp.csc_matrix(sp.diags(row_scale) @ G)
            d = d * row_scale
        n_other = int(nonempty.sum()) - len(srows) - n_soft
        chi2_exp = chi2_expectation(ks, 1.0 / np.sqrt(ks) if weigh else np.ones(len(ks)), n_other, const)
        say(f'   fit check: if the uncertainties are right, the chi^2 of a draw should be near {chi2_exp[0]:.0f} +- {chi2_exp[1]:.0f} '
            + (f'({len(ks)} slip-rate measurements, one per section, and {n_other} other rows' if weigh else
               f'({len(srows)} slip-rate rows, whose copies of one section share its error, which widens the spread, and {n_other} other rows')
            + (f', plus a constant {const:g} from the all-zero rows kept' if const else '')
            + (f'; the {n_soft} soft zero-forcing rows kept are left out' if n_soft else '') + '; assuming each section\'s slip rate is uniform along it)')
    else:
        chi2_exp = chi2_expectation([], [], int(nonempty.sum()) - n_soft, const)
        say(f'   fit check: if the uncertainties are right, the chi^2 of a draw should be near {chi2_exp[0]:.0f} +- {chi2_exp[1]:.0f} (every row its own measurement)')

    # ---- MFD uncertainty: the bins' overall level, their bin-to-bin shape as the inputs give it
    mfd_spec, mfd_c = str(args.mfd_uncertainty), 1.0
    if args.bins_block in bnames:
        Pb = dict(G=G, d=d, block=block, block_names=bnames, keep_rows=keep_r)
        try:
            M = mfd_uncertainty(Pb, mfd_spec, args.bins_block, args.lowest_bin, args.bin_width)
        except ValueError as e:
            raise SystemExit(f'--mfd-uncertainty: {e}')
        mfd_c, B, lo, ri, ro = M['scale'], M['bins'], M['lo'], M['rel_in'], M['rel_out']
        w = args.bin_width
        rng = lambda k: f'{lo[k]:.2f}-{lo[k] + w:.2f}'
        top, low = int(np.argmax(M['R'])), int(np.argmin(M['R']))
        at = lambda r: ((f'{100 * r[M["ref"]]:.1f}% at {rng(M["ref"])} (' if M['ref'] is not None else '(')
                        + f'from {100 * r[top]:.1f}% at the highest-rate bin, {rng(top)}, to {100 * r[low]:.1f}% at the lowest-rate '
                        f'bin, {rng(low)})')
        shape = ('' if M['power'] is None else
                 f', growing as R_j^-{M["power"]:.3f} towards the rarer bins'
                 + (' (exactly)' if M['power_resid'] < 1e-6 else f' (roughly: residual sd {M["power_resid"]:.2g} in ln)'))
        say(f'== MFD uncertainty ({mfd_spec!r}): the inputs give the {B} magnitude bins (original rows {M["orig"][0]}-'
            f'{M["orig"][-1]}, {rng(0)} to {rng(B - 1)}) relative uncertainties of {at(ri)}{shape}')
        implied = lambda tot, sl, short=False: f'+-{100 * tot:.1f}%' + ('' if sl is None else
                                                                        f' and +-{sl:.3f}' if short else
                                                                        f' and their slope to +-{sl:.3f} in b')
        if mfd_spec.lower() == 'as-given':
            say(f'   every bin counts as its own measurement (the inputs as given): the bin rows alone pin the bins\' total rate '
                f'to {implied(M["total_in"], M["slope_in"])}')
        else:
            why = (f'the bins are a projection of two measurements (the rate of M>=5 earthquakes and the b-value), so together '
                   f'they should carry two measurements\' worth of information, not one per bin: every bin\'s uncertainty is '
                   f'multiplied by sqrt({B}/2) = {mfd_c:.3f}' if mfd_spec.lower() == 'two-measurements' else
                   f'the {rng(M["ref"])} bin\'s relative uncertainty set to {100 * float(mfd_spec):.3g}%: every bin\'s '
                   f'uncertainty is multiplied by {mfd_c:.3f}')
            say(f'   {why} (its row divided by it), keeping the inputs\' bin-to-bin shape: {at(ro)}')
            say(f'   the bin rows alone now pin the bins\' total rate to {implied(M["total_out"], M["slope_out"])} (the inputs: '
                f'{implied(M["total_in"], M["slope_in"], True)}); the soft top\'s width, log10(1 + the bin\'s relative '
                f'uncertainty), follows: {np.log10(1 + ro.min()):.3f}-{np.log10(1 + ro.max()):.3f} decades')
        if args.slip_block in bnames:                    # the magnitudes rest on the bins' order: check it against sizes
            Gc = sp.csr_matrix(G)
            sr = np.flatnonzero(block == bnames.index(args.slip_block))
            nsub = np.asarray((Gc[sr] != 0).sum(0)).ravel()
            of = np.asarray((Gc[M['rows']] != 0).astype(np.int8).argmax(0)).ravel()
            med = np.array([np.median(nsub[of == k]) if np.any(of == k) else np.nan for k in range(B)])
            ok_ = np.isfinite(med)
            rk = lambda v: np.argsort(np.argsort(v, kind='stable'), kind='stable')
            rho = (float(np.corrcoef(rk(np.flatnonzero(ok_)), rk(med[ok_]))[0, 1])
                   if ok_.sum() > 2 and np.ptp(med[ok_]) > 0 else float('nan'))
            if not rho > 0.5:
                say(f'   WARNING: rupture sizes do not rise with the bins\' original order (rank correlation {rho:+.2f}); '
                    'the magnitudes above, and the M7.0-7.1 bin, may be wrong: check --lowest-bin and --bin-width')
        if mfd_c != 1.0:
            G, d, row_scale = scale_rows(G, d, row_scale, M['rows'], 1.0 / mfd_c)
    elif mfd_spec.lower() not in ('as-given', 'two-measurements'):
        raise SystemExit(f'--mfd-uncertainty {mfd_spec}: there are no magnitude-bin rows to set it for')
    elif mfd_spec.lower() != 'as-given':
        say(f'   no magnitude-bin rows, so --mfd-uncertainty {mfd_spec!r} has nothing to do')
    save_problem(args.out, G, d, sig, args.lo, args.hi, keep_r, keep_c, block, bnames, branges, raw_shape=A.shape,
                 row_scale=row_scale, section=section, slip_sections=str(spec), chi2_expect=np.array(chi2_exp),
                 mfd_uncertainty=mfd_spec, mfd_scale=mfd_c,
                 split_lo2=None if L2_raw is None else L2_raw[keep_r], split_hi2=None if H2_raw is None else H2_raw[keep_r],
                 cfm_sliprates=None if L2_raw is None else np.array([os.path.basename(args.cfm_sliprates),
                                                                    f'{args.cfm_sigmas:g}', f'{args.cfm_factor:g}']),
                 slip_rate_factor=args.slip_rate_factor)
    say(f'   saved cleaned problem -> {args.out}   ({time.time() - t0:.1f}s)')
    say('   magnitude-bin rows, checked on the saved problem: ' + check_bin_rows(load_problem(args.out), args.bins_block))

    # ---- scales
    cn = column_norms(G)
    s_cond = sig / cn
    nnzc = np.diff(G.indptr)
    say('== scales (sigma = %g, box [%g, %g])' % (sig, args.lo, args.hi))
    say('   ' + describe(cn, '||G_i||'))
    say('   ' + describe(s_cond, 'conditional sd s_i = sigma/||G_i||'))
    say('   ' + describe(u * cn / sig, 'rho_i = (hi-lo)||G_i||/sigma'))
    say('   ' + describe(nnzc, 'nonzeros per column', '%d'))
    rowsum = np.asarray(abs(G).sum(1)).ravel()
    collective = 0.1 * sig / rowsum.max()           # with every rupture at this rate, no prediction moves by more than 0.1 sigma
    say(f'   collective negligibility: with every rupture at the same rate, no prediction moves by more than 0.1 sigma below '
        f'{collective:.2g} (0.01 sigma below {collective / 10:.2g}; busiest row: original row {keep_r[int(np.argmax(rowsum))]}); '
        f"the floor 1e{args.diag_floor:g} is {'below' if 10.0 ** args.diag_floor <= collective else 'ABOVE'} this scale")
    Gr = G.tocsr()
    cap = np.full(n, np.inf)
    spread = np.full(n, np.inf)
    if info['negatives'] == 0:
        rowsum = np.asarray(G.sum(1)).ravel()
        for j in range(N):
            lo_, hi_ = Gr.indptr[j], Gr.indptr[j + 1]
            cols = Gr.indices[lo_:hi_]
            if hi_ == lo_:
                continue
            cap[cols] = np.minimum(cap[cols], (d[j] + 3 * sig) / Gr.data[lo_:hi_])
            spread[cols] = np.minimum(spread[cols], max(d[j], 0) / rowsum[j])
        say('   ' + describe(cap, 'data cap (d_j + 3 sigma)/G_ji (G >= 0 so m_i cannot exceed this)'))
        say(f'   -> upper bound hi = {args.hi:g} binds for {np.sum(cap > args.hi)} parameters')
        say('   ' + describe(spread, 'even-share rate: min_j d_j / sum_k G_jk'))
        say('   ' + describe(spread / s_cond, 'even-share rate / conditional sd'))

    # ---- spectrum, resolution, HMC stiffness
    GGt = (Gr @ Gr.T).toarray()
    lam, U = np.linalg.eigh(GGt)
    lam, U = lam[::-1], U[:, ::-1]
    sv = np.sqrt(np.clip(lam, 0, None))
    rank = int(np.sum(lam > 1e-12 * lam[0]))
    n_data = int(np.sum(np.diff(Gr.indptr) > 0))
    say('== spectrum: singular values s_1 = %.4g, s_%d = %.4g, numerical rank %d of %d rows (%d non-empty)' % (sv[0], rank, sv[rank - 1], rank, N, n_data))
    say('   s_k at k = 1, 10, 50, 100, 200, 300, 400: ' + ', '.join('%.3g' % sv[k - 1] for k in [1, 10, 50, 100, 200, 300, 400] if k <= N))
    Us = np.ascontiguousarray(U[:, :rank].T) / sv[:rank, None]   # contiguous: @ on the reversed view of U bypasses BLAS
    Gc, R = G.tocsc(), np.empty(n)
    for c0 in range(0, n, 1024):                     # R_ii = ||S^-1 U^T G_i||^2 by column blocks, with no dense copy of G,
        Sb = Gc[:, c0:c0 + 1024]                     # over the rows each block touches (about a fifth of them at 2e5)
        rows = np.unique(Sb.indices)
        Wb = Us[:, rows] @ Sb[rows].toarray()
        R[c0:c0 + 1024] = np.einsum('ij,ij->j', Wb, Wb)
    say('   ' + describe(R, 'resolution diagonal R_ii') + f';  sum = {R.sum():.1f} (= rank)')
    Gn = Gr @ sp.diags(1.0 / cn)
    ev_n = np.linalg.eigvalsh((Gn @ Gn.T).toarray())[::-1]
    say('   cosine-Gram eigenvalues (diag(G^T G)-scaled): top %s; omega_max = %.1f for HMC with mass diag(G^T G)/sigma^2'
        % (np.round(ev_n[:4], 1), np.sqrt(ev_n[0])))

    # ---- duplicates
    dup = None
    if not args.no_duplicates:
        t = time.time()
        dup = near_duplicates(G, cn)
        took = time.time() - t
        flat_txt = f', {dup["flat_columns"]} flat columns compared with one another' if dup['flat_columns'] else ''
        say(f'== near-duplicate columns (cosine similarity; {100 * dup["computed"]:.1f}% of the cosines a full search computes, '
            f'the others ruled out exactly{flat_txt}; {took:.1f}s)')
        for k in [k for k in dup if k.startswith('cos>')]:
            v = dup[k]
            say(f'   {k:10s}: {v["pairs"]:7d} pairs, {v["cols_in_groups"]:6d} columns in {v["groups"]:5d} groups, largest group {v["largest"]}')
        say(f'   exactly proportional pairs (cos > 1 - 1e-12): {dup["exact_proportional_pairs"]}')

    # ---- best fit (the chains start from it)
    t = time.time()
    m_nnls, its = nnls_lawson_hanson(G, d)
    r = G @ m_nnls - d
    grad = G.T @ r / sig ** 2
    chi2_min = float(r @ r) / sig ** 2
    say(f'== best fit (Lawson-Hanson NNLS, {its} outer iterations, {time.time() - t:.1f}s)')
    say(f'   chi^2_min = {chi2_min:.2f} for N = {N} rows, {n_data} non-empty (chi^2/N = {chi2_min / N:.3f}); nonzeros {np.sum(m_nnls > 0)}; '
        f'max m = {m_nnls.max():.3g} (hi = {args.hi:g} {"binds!" if m_nnls.max() > args.hi else "does not bind"}); sum m = {m_nnls.sum():.4g}')
    for b, nm in enumerate(bnames):
        k = block == b
        say(f'   block {nm:9s}: chi^2 = {r[k] @ r[k] / sig ** 2:7.2f} over {k.sum():3d} rows, max |r|/sigma = {abs(r[k]).max() / sig:.2f}')
    tol = 1e-9 * np.abs(G.T @ d).max() / sig ** 2       # nnls_lawson_hanson's default stopping tolerance, as a gradient
    say(f'   KKT: min gradient on zero set = {grad[m_nnls == 0].min():.3g} (>= 0 required, to within the stopping tolerance '
        f'{tol:.2g}), max |gradient| on support = {abs(grad[m_nnls > 0]).max():.2g}')
    if args.bins_block in bnames:                        # the best fit's bin totals: the data's own tension, with no prior
        brow = np.flatnonzero(block == bnames.index(args.bins_block))
        Gb = sp.csr_matrix(G)[brow]
        coef = np.array([Gb.data[Gb.indptr[k]] for k in range(len(brow))])
        tgt, tot = d[brow] / coef, (Gb @ m_nnls) / coef
        rr = tot / tgt - 1
        say(f'   magnitude bins in the best fit: totals {100 * rr.min():+.1f}% to {100 * rr.max():+.1f}% of their targets (median '
            f'{100 * np.median(rr):+.1f}%), within {np.abs(rr * d[brow]).max():.2f} of their uncertainties; total {tot.sum():.4g} '
            f'against {tgt.sum():.4g} ({100 * (tot.sum() / tgt.sum() - 1):+.1f}%)')

    np.savez(f_res, cn=cn, s_cond=s_cond, collective=collective, cap=cap, spread=spread, sv=sv, R=R, ev_n=ev_n,
             m_nnls=m_nnls, r_nnls=r, chi2_min=chi2_min, block=block,
             dup_i=dup['pairs_i'] if dup else np.zeros(0, int), dup_j=dup['pairs_j'] if dup else np.zeros(0, int),
             dup_cos=dup['pairs_cos'] if dup else np.zeros(0))
    m_orig = np.zeros(A.shape[1])                        # the best fit as rate vectors (original rupture numbering, removed
    m_orig[keep_c] = m_nnls                              # ruptures 0), for the fit figures (plot_*.py)
    pen = np.ones(A.shape[1], int)
    pen[keep_c] = 0
    os.makedirs(d_nnls, exist_ok=True)
    np.savez(f_vec, mean=m_orig, penalised=pen, model='best fit (NNLS)')
    with open(f_csv, 'w', newline='') as fh:             # and as a table with the rows of make_tables.py's rate_vectors.csv
        wr = csv.writer(fh)
        wr.writerow(['rupture', 'penalised', 'best_fit'])
        wr.writerows([i, pen[i], f'{m_orig[i]:.6e}'] for i in range(len(m_orig)))
    for f in (old_vec, old_csv):
        if os.path.exists(f):                            # an earlier cleaning's best fit would not match this problem
            os.remove(f)
    open(f_sum, 'w').write('\n'.join(LOG) + '\n')

    # ---- figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    bins = np.linspace(-9, 0, 91)
    ax[0, 0].hist(np.log10(s_cond), bins, alpha=.6, label=r'conditional sd $\sigma/\|G_i\|$')
    if np.isfinite(cap).all():
        ax[0, 0].hist(np.log10(cap), bins, alpha=.6, label=r'data cap $(d_j+3\sigma)/G_{ji}$')
        ax[0, 0].hist(np.log10(spread), bins, alpha=.6, label='even-share rate')
    ax[0, 0].axvline(np.log10(args.hi), color='k', ls='--', lw=1, label=f'upper bound {args.hi:g}')
    ax[0, 0].set_xlabel(r'$\log_{10}$ (model units)'); ax[0, 0].set_ylabel('parameters'); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title('Scales per parameter')
    ax[0, 1].semilogy(np.arange(1, N + 1), np.maximum(sv, 1e-3), '.', ms=3)
    ax[0, 1].set_xlabel('k'); ax[0, 1].set_ylabel('singular value'); ax[0, 1].set_title(f'Spectrum of G (rank {rank})')
    ax[1, 0].hist(np.log10(np.clip(R, 1e-6, 1)), 60)
    ax[1, 0].set_xlabel(r'$\log_{10} R_{ii}$'); ax[1, 0].set_ylabel('parameters'); ax[1, 0].set_title('Resolution diagonal')
    for b, nm in enumerate(bnames):
        k = np.flatnonzero(block == b)
        ax[1, 1].plot(k, r[k] / sig, '.', ms=3, label=nm)
    ax[1, 1].axhline(0, color='k', lw=.5)
    ax[1, 1].set_xlabel('row (cleaned index)'); ax[1, 1].set_ylabel(r'residual $(Gm-d)/\sigma$'); ax[1, 1].set_ylim(-3, 2)
    ax[1, 1].set_title(rf'NNLS best fit, $\chi^2_{{min}}$ = {chi2_min:.1f}'); ax[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f_png, dpi=120)
    say(f'== done in {time.time() - t0:.1f}s: {args.out}, {f_res}, {f_sum}, {f_png}'
        + f', {f_vec}, {f_csv}')


if __name__ == '__main__':
    main()

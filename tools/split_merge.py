"""Split/merge moves' triples (gibbs.py --split-merge, 2026-11-24): each rupture A of the rupture set that is two other
ruptures laid end to end, B (the first part of A's subsections, in indices.csv's order) and C (the rest), all three kept
by cleaning. With uniform slip along the ruptures (NZSHM22's), rate t on A and rate (D_A / D_B) t on B with (D_A / D_C) t
on C put the same slip on every subsection (D: average slips), so a move between the two leaves every slip-rate row's
prediction as it is, and the summed slip rates' and the plate rate's rows, which add slip-rate rows up; gibbs.py's
splitmerge_run scores what does change (the paleo rows, the bins) exactly.

The pairs (B, C) are found by hashing subsection sets: each subsection a random 64-bit key, a set's hash the sum of its
keys (mod 2^64), so each prefix of A's subsection list and the matching suffix are looked up among the kept ruptures'
sets in one pass; each triple found is then checked on the cleaned problem's slip-rate rows (G_A against alpha G_B +
beta G_C), which also rules out a chance collision of hashes."""
import os

import numpy as np
import scipy.sparse as sp


def find_triples(rupture_dir, keep_cols, seed=20261124, chunk=100_000):
    """The triples (A, B, C) among the kept ruptures, as columns of the cleaned problem (keep_cols: each column's rupture),
    with alpha = D_A / D_B and beta = D_A / D_C from average_slips.csv. Returns TA, TB, TC (int64), alpha, beta."""
    from make_solution import read_rupture_set
    _, ptr, idx = read_rupture_set(rupture_dir)
    D = np.loadtxt(os.path.join(rupture_dir, 'average_slips.csv'), delimiter=',', skiprows=1)
    if not np.array_equal(D[:, 0], np.arange(len(ptr) - 1)):
        raise ValueError(f'{rupture_dir}/average_slips.csv: rupture indices are not 0, 1, 2, ... in order')
    D = D[:, 1]
    nR = len(ptr) - 1
    keep = np.asarray(keep_cols, np.int64)
    col_of = np.full(nR, -1, np.int64)
    col_of[keep] = np.arange(len(keep))
    key = np.random.default_rng(seed).bit_generator.random_raw(int(idx.max()) + 1).astype(np.uint64)
    cs = np.cumsum(key[idx], dtype=np.uint64)                 # running sums (mod 2^64) over indices.csv's entries
    base = np.zeros(nR, np.uint64)
    base[1:] = cs[ptr[1:-1] - 1]
    H = cs[ptr[1:] - 1] - base                                # each rupture's set
    Hk = H[keep]
    o = np.argsort(Hk, kind='stable')
    Hs = Hk[o]
    same = np.zeros(len(Hs), bool)                            # kept ruptures with the same subsections as another (or
    same[1:] |= np.diff(Hs) == 0                              # a collision of hashes): never B or C, which would be
    same[:-1] |= np.diff(Hs) == 0                             # ambiguous

    def look(h):
        j = np.minimum(np.searchsorted(Hs, h), len(Hs) - 1)
        return np.where((Hs[j] == h) & ~same[j], o[j], -1)
    out = []
    for r0 in range(0, nR, chunk):                            # the cuts after each entry but a rupture's last, A kept
        r1 = min(r0 + chunk, nR)
        p = np.arange(ptr[r0], ptr[r1])
        rup = np.repeat(np.arange(r0, r1), np.diff(ptr[r0:r1 + 1]))
        sel = (p < ptr[rup + 1] - 1) & (col_of[rup] >= 0)
        p, rup = p[sel], rup[sel]
        pre = cs[p] - base[rup]
        cb, cc = look(pre), look(H[rup] - pre)
        hit = (cb >= 0) & (cc >= 0)
        out.append((col_of[rup[hit]], cb[hit], cc[hit]))
    TA, TB, TC = (np.concatenate([t[q] for t in out]) if out else np.zeros(0, np.int64) for q in range(3))
    alpha, beta = D[keep[TA]] / D[keep[TB]], D[keep[TA]] / D[keep[TC]]
    return TA, TB, TC, alpha, beta


def check_triples(G, rows, TA, TB, TC, alpha, beta, chunk=200_000):
    """Each triple's mismatch on the rows given (the slip-rate rows): |G_A - alpha G_B - beta G_C| / |G_A| (G the cleaned
    problem's, unscaled columns)."""
    Gs = sp.csc_matrix(sp.csr_matrix(G)[np.asarray(rows)])
    rel = np.empty(len(TA))
    for c0 in range(0, len(TA), chunk):
        c1 = min(c0 + chunk, len(TA))
        A_ = Gs[:, TA[c0:c1]]
        M = A_ - Gs[:, TB[c0:c1]] @ sp.diags(alpha[c0:c1]) - Gs[:, TC[c0:c1]] @ sp.diags(beta[c0:c1])
        na = np.sqrt(np.asarray(A_.multiply(A_).sum(0)).ravel())
        nm = np.sqrt(np.asarray(M.multiply(M).sum(0)).ravel())
        rel[c0:c1] = nm / np.where(na > 0, na, 1.0)
    return rel


def describe(TA, TB, n, rel, tol):
    """One line on the triples kept."""
    na, nb = np.bincount(TA, minlength=n), np.bincount(TB, minlength=n)
    return (f'{len(TA):,} triples (a rupture and the two that together are it) among the {n:,} ruptures: {int((na > 0).sum()):,} '
            f'can split (median {np.median(na[na > 0]) if (na > 0).any() else 0:.0f} ways, at most {na.max() if n else 0}), '
            f'{int(((na > 0) | (nb > 0)).sum()):,} take part; the slip-rate rows unchanged by each to '
            f'{rel.max() if len(rel) else 0.0:.1e} (relative; {int((rel > tol).sum())} above {tol:g})')

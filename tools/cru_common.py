"""Shared I/O and MCMC diagnostics for the workflow's scripts (preliminaries.py, gibbs.py and the others), and the notebook's
log of everything it prints.

Problem convention throughout: minimise ||G m - d||^2 / (2 sigma^2) with lo <= m <= hi, i.e. the target
density is exp(-||G m - d||^2 / (2 sigma^2) - beta . m) on the box (beta = 0 is the flat prior).
"""
import glob
import io
import json
import os
import re
import sys
import time
import traceback

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- I/O
def load_colt_sparse(path):
    """Read a Colt-style sparse dump: header 'N x n sparse matrix, nnz = k', then '(i,j)<TAB>value' lines, 0-based."""
    import pandas as pd
    with open(path) as f:
        header = f.readline()
        body = f.read()
    m = re.search(r'(\d+)\s*x\s*(\d+).*?nnz\s*=\s*(\d+)', header)
    if not m:
        raise ValueError(f'unrecognised header in {path!r}: {header[:80]!r}')
    N, n, nnz = map(int, m.groups())
    body = body.replace('(', '').replace(')', '').replace(',', '\t')
    df = pd.read_csv(io.StringIO(body), sep='\t', header=None, names=['r', 'c', 'v'],
                     dtype={'r': np.int64, 'c': np.int64, 'v': np.float64})
    if len(df) != nnz:
        print(f'warning: header says nnz = {nnz} but {len(df)} entries were read')
    A = sp.csc_matrix((df.v.values, (df.r.values, df.c.values)), shape=(N, n))
    A.sum_duplicates()
    return A


def load_vector(path):
    """Read d from a JSON/Python-style list '[a, b, ...]' or from whitespace/comma separated text."""
    txt = open(path).read().strip()
    try:
        return np.asarray(json.loads(txt), dtype=np.float64)
    except json.JSONDecodeError:
        return np.array(re.split(r'[\s,\[\]]+', txt.strip('[] \n')), dtype=np.float64)


def save_problem(path, G, d, sigma, lo, hi, keep_rows, keep_cols, block, block_names, block_rows, raw_shape=None, **fields):
    """fields: optional extra arrays, e.g. row_scale, section, slip_sections and chi2_expect (see slip_sections)."""
    G = sp.csc_matrix(G)
    extra = {} if raw_shape is None else {'raw_shape': np.array(raw_shape)}
    extra.update({k: np.asarray(v) for k, v in fields.items() if v is not None})
    np.savez(path, data=G.data, indices=G.indices, indptr=G.indptr, shape=np.array(G.shape),
             d=d, sigma=sigma, lo=lo, hi=hi, keep_rows=keep_rows, keep_cols=keep_cols,
             block=block, block_names=np.array(block_names), block_rows=np.array(block_rows), **extra)


def load_problem(path):
    """Load the cleaned problem written by preliminaries.py. Problems written before slip-rate sections existed load
    with row_scale 1, no sections (-1), slip_sections 'none' and chi2_expect None; problems written before the MFD
    uncertainty setting existed load with mfd_uncertainty 'as-given' and mfd_scale 1 (the inputs' bin uncertainties).

    row_scale is the factor cleaning multiplied each row by, relative to the inputs: 1/sqrt(k) for a slip-rate
    section's k rows, and 1/mfd_scale for the magnitude-bin rows, whose uncertainties cleaning multiplied by mfd_scale
    (preliminaries.py --mfd-uncertainty); dividing a row's target and predictions by it gives them in the inputs' units."""
    z = np.load(path, allow_pickle=False)
    G = sp.csc_matrix((z['data'], z['indices'], z['indptr']), shape=tuple(z['shape']))
    N = G.shape[0]
    names = [str(s) for s in z['block_names']]
    return dict(G=G, d=z['d'].astype(np.float64), sigma=float(z['sigma']), lo=float(z['lo']), hi=float(z['hi']),
                keep_rows=z['keep_rows'], keep_cols=z['keep_cols'], block=z['block'],
                block_names=names, block_rows=[str(s) for s in z['block_rows']],
                raw_shape=tuple(int(v) for v in z['raw_shape']) if 'raw_shape' in z.files else None,
                row_scale=z['row_scale'].astype(np.float64) if 'row_scale' in z.files else np.ones(N),
                section=z['section'].astype(np.int64) if 'section' in z.files else np.full(N, -1, np.int64),
                slip_sections=str(z['slip_sections']) if 'slip_sections' in z.files else 'none',
                chi2_expect=tuple(float(v) for v in z['chi2_expect']) if 'chi2_expect' in z.files else None,
                mfd_uncertainty=str(z['mfd_uncertainty']) if 'mfd_uncertainty' in z.files else 'as-given',
                mfd_scale=float(z['mfd_scale']) if 'mfd_scale' in z.files else 1.0)


def slip_sections(G, d, rows, keep_rows=None, spec='auto', rtol=1e-9):
    """Group the slip-rate rows `rows` (cleaned indices, in order) into fault sections, each one measurement.

    'auto': consecutive rows belong to one section when they are copies of one slip rate: identical targets, and
            identical coefficients for every rupture both rows contain (the same uncertainty, with the same slip on both
            subsections). A row unlike its neighbours is a section of its own (k = 1), so slip rates that vary from
            subsection to subsection (a geodetic model, say) are left as they are.
    'none': the same detection, for reporting only; every row stays its own measurement (the inputs as given).
    a path: a file of 'original_row section' lines (whitespace or comma separated; '#' starts a comment), one for every
            slip-rate row, e.g. from the rupture set's subsection-to-section mapping.
    Returns (section index of each row, number of rows k in its section, notes): with 'auto' or a file, the caller
    divides each row by sqrt(k), so that each section's measurement counts once."""
    Gr = sp.csr_matrix(G)
    rows = np.asarray(rows)
    orig = np.asarray(keep_rows)[rows] if keep_rows is not None else rows
    notes = []
    if spec in ('auto', 'none'):
        sec = np.zeros(len(rows), np.int64)
        unverified = 0
        for t in range(1, len(rows)):
            a, b = rows[t - 1], rows[t]
            same = abs(d[a] - d[b]) <= rtol * max(abs(d[a]), abs(d[b]), 1e-300)
            if same:
                ia, va = Gr.indices[Gr.indptr[a]:Gr.indptr[a + 1]], Gr.data[Gr.indptr[a]:Gr.indptr[a + 1]]
                ib, vb = Gr.indices[Gr.indptr[b]:Gr.indptr[b + 1]], Gr.data[Gr.indptr[b]:Gr.indptr[b + 1]]
                _, xa, xb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
                if len(xa) == 0:
                    same, unverified = False, unverified + 1
                else:
                    same = bool(np.all(np.abs(va[xa] - vb[xb]) <= rtol * np.maximum(np.abs(va[xa]), np.abs(vb[xb]))))
            sec[t] = sec[t - 1] + (0 if same else 1)
        if unverified:
            notes.append(f'{unverified} pairs of neighbouring rows with identical targets share no rupture, so they could not be '
                         'confirmed as copies and count separately')
    else:
        lab = {}
        for ln, line in enumerate(open(spec), 1):
            line = line.split('#')[0].replace(',', ' ').split()
            if not line:
                continue
            if len(line) < 2:
                raise ValueError(f'{spec}, line {ln}: expected "original_row section"')
            try:
                r = int(line[0])
            except ValueError:
                if not lab:
                    continue                               # a header line
                raise ValueError(f'{spec}, line {ln}: {line[0]!r} is not a row number')
            lab[r] = line[1]
        missing = [int(o) for o in orig if int(o) not in lab]
        extra = sorted(set(lab) - set(int(o) for o in orig))
        if missing:
            raise ValueError(f'{spec} gives no section for {len(missing)} slip-rate rows, e.g. original rows {missing[:5]}')
        if extra:
            raise ValueError(f'{spec} lists {len(extra)} rows that are not slip-rate rows, e.g. original rows {extra[:5]}')
        first = {}                                         # sections numbered in order of their first row
        sec = np.array([first.setdefault(lab[int(o)], len(first)) for o in orig], np.int64)
        mixed = [s for s in np.unique(sec) if np.ptp(d[rows[sec == s]]) > rtol * np.abs(d[rows[sec == s]]).max()]
        if mixed:
            notes.append(f'{len(mixed)} sections in {spec} hold rows with different targets; they still count as one '
                         'measurement each, as the file says')
    k = np.bincount(sec)[sec]
    return sec, k, notes


def chi2_expectation(k, w, n_other, const=0.0):
    """Mean and sd of the chi-squared of a model that matches the truth, if the uncertainties are right: per section, its
    k rows share one measurement error and each carries weight w (rows divided by 1/w), so the section contributes
    k w^2 eps^2 (eps ~ N(0, 1)); every other row carrying data contributes eps^2; all-zero rows kept add a constant.
    Assumes each section's slip rate is uniform along it. k and w are per section."""
    k, w = np.asarray(k, float), np.asarray(w, float)
    m = k * w ** 2
    return float(m.sum() + n_other + const), float(np.sqrt(2 * np.sum(m ** 2) + 2 * n_other))


# --------------------------------------------------------------------------- row blocks: measurements
def block_measurements(P):
    """Each row block's share of the fit check: per block that holds data (all-zero rows left out), its rows and the
    mean and sd of the chi-squared a model matching the truth would have there if the uncertainties are right. A
    slip-rate section's k rows share one error: divided by sqrt(k) they count as one measurement in all, and with
    slip_sections 'none' each row counts once (their shared error widens the spread). The means add up to the fit
    check, chi2_expect, less any constant from all-zero rows kept; in the default they count independent measurements.
    Each magnitude bin counts once, whatever its uncertainty (MFD_UNCERTAINTY changes how much the bins weigh, not
    how many residuals they have)."""
    Gr = sp.csr_matrix(P['G'])
    nonempty = np.diff(Gr.indptr) > 0
    sec, weigh = np.asarray(P['section']), P['slip_sections'] != 'none'
    out = []
    for b, name in enumerate(P['block_names']):
        rows = np.flatnonzero((np.asarray(P['block']) == b) & nonempty)
        if not len(rows):
            continue
        s = sec[rows]
        k = np.bincount(s[s >= 0]) if (s >= 0).any() else np.zeros(0, int)
        k = k[k > 0].astype(float)
        kw2 = np.ones(len(k)) if weigh else k               # k w^2 per section: 1 when its rows are divided by sqrt(k)
        n1 = int(np.sum(s < 0))                             # rows outside sections: one measurement each
        out.append(dict(name=name, index=b, rows=rows, expect=float(kw2.sum() + n1),
                        sd=float(np.sqrt(2 * np.sum(kw2 ** 2) + 2 * n1))))
    return out


def block_labels(names, slip_block='S', bins_block='MFD'):
    """Readable names for the row blocks: the slip-rate rows, the magnitude bins, and, when exactly one other block
    holds data (the NZSHM22 inputs' paleoseismic rows), 'paleo'; any further blocks keep their names."""
    rest = [n for n in names if n not in (slip_block, bins_block, 'empty')]
    lab = {slip_block: 'slip rates', bins_block: 'MFD bins'}
    if len(rest) == 1:
        lab[rest[0]] = 'paleo'
    return {n: lab.get(n, n) for n in names}


# --------------------------------------------------------------------------- magnitude bins and their uncertainty
MFD_UNCERTAINTY_OPTIONS = ('as-given', 'two-measurements')


def magnitude_bins(P, block='MFD', lowest=6.8, width=0.1):
    """The magnitude-bin rows (indices into the cleaned problem's rows) in original order, their targets R_j, their
    original row numbers and the lower magnitude edge of each bin. A rupture's magnitude is known only to its bin: the
    bin rows are taken as consecutive bins of `width` in their original row order, the lowest one left after cleaning
    being `lowest` (6.80-6.90 for the NZSHM22 inputs, whose lower bins hold only the penalised ruptures); a bin that
    cleaning removed leaves a gap."""
    R, rows = bin_targets(P, block)                  # raises unless the rows are plain sums over a partition of ruptures
    orig = np.asarray(P['keep_rows'])[rows]
    o = np.argsort(orig, kind='stable')
    rows, R, orig = rows[o], R[o], orig[o]
    return rows, R, orig, lowest + width * (orig - orig[0])


def mfd_implied(R, rel, lo, width):
    """What the bin rows alone say about the MFD as a whole, if each bin's total is a measurement with relative
    uncertainty rel (independent): the relative sd of the bins' total rate, and the sd of their slope in log10 rate
    per unit magnitude (a b-value; weighted least squares on the bin centres; None for fewer than 3 bins)."""
    R, rel, lo = (np.asarray(v, float) for v in (R, rel, lo))
    total = float(np.sqrt(np.sum((rel * R) ** 2)) / R.sum())
    if len(R) < 3:
        return total, None
    w = (np.log(10.0) / rel) ** 2                    # weights of log10 R_j
    x = lo + width / 2
    xb = np.sum(w * x) / w.sum()
    return total, float(1.0 / np.sqrt(np.sum(w * (x - xb) ** 2)))


def mfd_uncertainty(P, spec='as-given', block='MFD', lowest=6.8, width=0.1, reference=7.0):
    """The magnitude-bin rows' uncertainties, recomputed (preliminaries.py --mfd-uncertainty; MFD_UNCERTAINTY in the
    notebook). Each bin row is its constraint divided by the bin's uncertainty sigma_j (coefficient 1/sigma_j for each
    of its ruptures, target R_j/sigma_j), so d_j is one over the bin's relative uncertainty, and multiplying the bin's
    uncertainty by c divides its row by c, leaving R_j unchanged. Every bin gets the same c, so the bin-to-bin shape
    the inputs give (set where they are built: in the NZSHM22 inputs, relative uncertainties growing as R_j^-0.25) is
    kept, and only the overall level changes:
      'as-given' (the default): the inputs' uncertainties, each bin counted as its own measurement (c = 1);
      'two-measurements': the bins are a projection of two measurements, the rate of M>=5 earthquakes and the b-value,
          so together they should carry the information of two measurements, not one per bin: c = sqrt(B / 2) for B
          bins (3 for 18), as a slip-rate section's k copies are divided by sqrt(k) to count once (with the inputs'
          uncertainties as each bin's own, the bins' total information then matches two measurements');
      a number u: the relative uncertainty (one sd) of the bin from `reference` (M7.0-7.1, where the NZSHM22 inputs
          put their design value, 0.1), the other bins keeping the inputs' shape: c = u / that bin's in the inputs.
    Magnitudes as magnitude_bins gives them. P must hold the bin rows as the inputs give them. Returns a dict: c
    (scale), the bin rows in magnitude order (rows, orig, lo, R), relative uncertainties before and after (rel_in,
    rel_out), the reference bin's position (ref; None if the bins have none), the power p of the inputs' shape,
    rel_in ~ R^-p, with the sd of that fit's residuals (None for fewer than 3 bins), and what the bin rows alone say
    about the MFD as a whole before and after (mfd_implied: total_in/out, slope_in/out)."""
    rows, R, orig, lo = magnitude_bins(P, block, lowest, width)
    dj = np.asarray(P['d'], float)[rows]
    if not np.all(dj > 0):
        raise ValueError(f'{int(np.sum(dj <= 0))} magnitude bins have a target that is not positive; their relative '
                         'uncertainty is undefined')
    rel_in = 1.0 / dj
    B = len(rows)
    hit = np.flatnonzero(np.abs(lo - reference) <= 1e-6 * max(width, 1e-12))
    ref = int(hit[0]) if hit.size else None
    s = str(spec).strip().lower()
    if s == 'two-measurements':
        c = float(np.sqrt(max(B, 2) / 2.0))
    elif s == 'as-given':
        c = 1.0
    else:
        try:
            u = float(s)
        except ValueError:
            raise ValueError(f"{spec!r}: use 'two-measurements', 'as-given' or a number, the relative uncertainty (one "
                             f'sd) of the M{reference:.2f}-{reference + width:.2f} bin') from None
        if not (np.isfinite(u) and 0 < u <= 2):
            raise ValueError(f'{spec!r}: a relative uncertainty must be above 0 and at most 2 (a fraction: 0.3 for 30%)')
        if ref is None:
            raise ValueError(f'{spec!r} sets the M{reference:.2f}-{reference + width:.2f} bin\'s uncertainty, but the bins '
                             f'run from {lo[0]:.2f} to {lo[-1] + width:.2f} in steps of {width:g} with no bin there (check '
                             '--lowest-bin and --bin-width)')
        c = float(u / rel_in[ref])
    if B >= 3 and np.ptp(np.log(R)) > 0:
        f = np.polyfit(np.log(R), np.log(rel_in), 1)
        power, resid = float(-f[0]), float(np.std(np.log(rel_in) - np.polyval(f, np.log(R))))
    else:
        power, resid = None, None
    t_in, b_in = mfd_implied(R, rel_in, lo, width)
    return dict(scale=c, rows=rows, orig=orig, lo=lo, R=R, rel_in=rel_in, rel_out=c * rel_in, ref=ref, power=power,
                power_resid=resid, total_in=t_in, total_out=c * t_in, slope_in=b_in,
                slope_out=None if b_in is None else c * b_in, bins=B)


def scale_rows(G, d, row_scale, rows, factor):
    """Multiply the rows `rows` of G and d, and their row_scale, by `factor` (dividing a row by c multiplies its
    uncertainty by c). Returns new G (CSC), d and row_scale; the other rows are unchanged."""
    f = np.ones(G.shape[0])
    f[np.asarray(rows)] = factor
    return sp.csc_matrix(sp.diags(f) @ G), np.asarray(d, float) * f, np.asarray(row_scale, float) * f


def to_original(vec, P, fill=0.0):
    """Place a vector over the cleaned problem's ruptures at their original column positions (the columns of the raw
    matrix); ruptures removed by cleaning (the penalised ones) get `fill`."""
    if P.get('raw_shape') is not None:
        n_raw = P['raw_shape'][1]
    else:
        n_raw = int(np.max(P['keep_cols'])) + 1
        print('note: this cleaned problem does not record the raw matrix shape; re-run preliminaries.py to be sure '
              'trailing penalised ruptures are included')
    out = np.full(n_raw, fill, dtype=np.float64)
    out[P['keep_cols']] = vec
    return out


def column_norms(G):
    G = sp.csc_matrix(G)
    return np.sqrt(np.asarray(G.multiply(G).sum(0)).ravel())


def prior_rates(spec, n, path=None):
    """Exponential-prior rates beta (per unit m).  spec: 'flat' -> 0, a number tau -> 1/tau, or a .npy of tau_i."""
    if spec in (None, 'flat', '0'):
        return np.zeros(n)
    try:
        tau = float(spec)
        return np.full(n, 1.0 / tau)
    except ValueError:
        tau = np.load(spec).astype(np.float64)
        return 1.0 / tau


# --------------------------------------------------------------------------- diagnostics
def split_rhat(x):
    """Classic split R-hat.  x: (chains, draws, ...) -> (...).  NaN where a quantity is constant."""
    x = np.asarray(x, dtype=np.float64)
    S = x.shape[1]
    h = S // 2
    xs = np.concatenate([x[:, :h], x[:, S - h:]], axis=0)
    W = xs.var(1, ddof=1).mean(0)
    B = h * xs.mean(1).var(0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt(((h - 1) / h * W + B / h) / W)


def ess(x):
    """Multi-chain effective sample size (Stan's estimator with Geyer's initial monotone sequence).  x: (chains, draws)."""
    x = np.asarray(x, dtype=np.float64)
    K, S = x.shape
    if S < 4:
        return np.nan
    xc = x - x.mean(1, keepdims=True)
    nfft = 1 << (2 * S - 1).bit_length()
    f = np.fft.rfft(xc, nfft, axis=1)
    acov = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :S] / S
    W = (acov[:, 0] * S / (S - 1)).mean()
    var_plus = W * (S - 1) / S + (x.mean(1).var(ddof=1) if K > 1 else 0.0)
    if not var_plus > 0:
        return np.nan
    rho = 1.0 - (W - acov.mean(0)) / var_plus
    rho[0] = 1.0
    tau, P_prev, t = -1.0, np.inf, 0
    while 2 * t + 1 < S:
        P = rho[2 * t] + rho[2 * t + 1]
        if P <= 0:
            break
        P = min(P, P_prev)
        tau += 2 * P
        P_prev = P
        t += 1
    return K * S / max(tau, 1e-12)


def summarize_samples(samples, qs=(0.05, 0.5, 0.95), chunk=2048):
    """Per-parameter mean, sd, quantiles and split R-hat from samples of shape (chains, draws, p) (memmap OK)."""
    K, S, p = samples.shape
    out = {k: np.empty(p) for k in ['mean', 'sd', 'rhat'] + [f'q{int(100 * q):02d}' for q in qs]}
    for j0 in range(0, p, chunk):
        blk = np.asarray(samples[:, :, j0:j0 + chunk], dtype=np.float64)
        flat = blk.reshape(K * S, -1)
        out['mean'][j0:j0 + chunk] = flat.mean(0)
        out['sd'][j0:j0 + chunk] = flat.std(0, ddof=1)
        for q in qs:
            out[f'q{int(100 * q):02d}'][j0:j0 + chunk] = np.quantile(flat, q, axis=0)
        out['rhat'][j0:j0 + chunk] = split_rhat(blk)
    return out


def describe(x, name, fmt='%.3g'):
    q = np.percentile(x, [0, 5, 50, 95, 100])
    return f'{name}: min {fmt % q[0]}, 5% {fmt % q[1]}, median {fmt % q[2]}, 95% {fmt % q[3]}, max {fmt % q[4]}'


def ess_many(samples, chunk=1024):
    """Per-parameter multi-chain ESS (same estimator as ess(), vectorised).  samples: (chains, draws, p), memmap OK."""
    K, S, p = samples.shape
    out = np.empty(p)
    nfft = 1 << (2 * S - 1).bit_length()
    npairs = S // 2
    for j0 in range(0, p, chunk):
        x = np.asarray(samples[:, :, j0:j0 + chunk], dtype=np.float64)
        c = x.shape[2]
        xc = x - x.mean(1, keepdims=True)
        f = np.fft.rfft(xc, nfft, axis=1)
        acov = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :S] / S
        W = (acov[:, 0] * S / (S - 1)).mean(0)
        var_plus = W * (S - 1) / S + (x.mean(1).var(0, ddof=1) if K > 1 else 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            rho = 1.0 - (W[None] - acov.mean(0)) / var_plus[None]
        rho[0] = 1.0
        P = rho[0:2 * npairs:2] + rho[1:2 * npairs:2]
        pos = np.cumprod(P > 0, axis=0).astype(bool)
        P = np.minimum.accumulate(np.where(pos, P, np.inf), axis=0)
        tau = -1.0 + 2.0 * np.where(pos, P, 0.0).sum(0)
        out[j0:j0 + c] = K * S / np.maximum(tau, 1e-12)
    return out


# --------------------------------------------------------------------------- log-uniform prior placement
def check_bin_rows(P, block='MFD', rtol=1e-6):
    """Check that the magnitude-bin rows are plain sums over a partition of the ruptures: within each row every nonzero
    coefficient is the same (the row is the bin's sum divided by its uncertainty), every rupture appears in exactly one
    bin row, and every right-hand side is positive. Only then is d_j divided by the row's coefficient the bin's target
    rate. Returns a one-line summary; raises ValueError naming the offending (original) rows otherwise."""
    if block not in P['block_names']:
        raise ValueError(f'no row block named {block!r}; the blocks are {list(P["block_names"])}')
    G = sp.csr_matrix(P['G'])
    rows = np.flatnonzero(P['block'] == P['block_names'].index(block))
    orig = P['keep_rows'] if 'keep_rows' in P else np.arange(G.shape[0])
    problems = []
    for j in rows:
        v = G.data[G.indptr[j]:G.indptr[j + 1]]
        if v.size == 0:
            problems.append(f'row {int(orig[j])} is empty')
        elif np.ptp(v) > rtol * np.abs(v).max():
            problems.append(f'row {int(orig[j])} has varying coefficients ({v.min():.6g} to {v.max():.6g})')
        if P['d'][j] <= 0:
            problems.append(f'row {int(orig[j])} has a non-positive right-hand side ({P["d"][j]:.4g})')
    count = np.asarray((G[rows] != 0).sum(0)).ravel()
    if (count == 0).any():
        problems.append(f'{int((count == 0).sum())} ruptures are in no bin row')
    if (count > 1).any():
        problems.append(f'{int((count > 1).sum())} ruptures are in more than one bin row')
    if problems:
        raise ValueError(f'the {block!r} rows are not plain sums over a partition of the ruptures: ' + '; '.join(problems[:6])
                         + (f'; and {len(problems) - 6} more' if len(problems) > 6 else ''))
    R = np.array([P['d'][j] / G.data[G.indptr[j]] for j in rows])
    return (f'{len(rows)} bin rows, each a plain sum over its ruptures; every rupture in exactly one bin; '
            f'targets R_j from {R.min():.3g} to {R.max():.3g}/yr, total {R.sum():.4g}/yr')


def bin_targets(P, block='MFD'):
    """Target total rate of each magnitude bin: the row's d divided by its (constant) coefficient. Checks the bin-row
    structure first (check_bin_rows), since the targets are meaningless otherwise."""
    check_bin_rows(P, block)
    G = sp.csr_matrix(P['G'])
    rows = np.flatnonzero(P['block'] == P['block_names'].index(block))
    return np.array([P['d'][j] / G.data[G.indptr[j]] for j in rows]), rows


def average_data_rate(P, block='MFD'):
    """Average data-imposed rupture rate: the sum of the magnitude-bin target rates divided by the number of ruptures."""
    return float(bin_targets(P, block)[0].sum() / P['G'].shape[1])


def logbox_mean(a, b, se, u_hi, npts=400001):
    """Mean rate of the log-box prior: flat in log10 m on [a, b], half-Gaussian edges of sd se decades
    (se = 0: hard edges), truncated at log10 m <= u_hi."""
    lo = a - 12 * se if se > 0 else a
    hi = u_hi if se > 0 else min(b, u_hi)
    u = np.linspace(lo, hi, npts)
    f = np.ones_like(u)
    if se > 0:
        f = np.where(u < a, np.exp(-(a - u) ** 2 / (2 * se ** 2)), np.where(u > b, np.exp(-(u - b) ** 2 / (2 * se ** 2)), 1.0))
    return float(np.sum(f * 10.0 ** u) / np.sum(f))


def resolve_logbox(spec, P, block='MFD'):
    """Edges (a, b, se) in log10 rate and a description, for a log-uniform ("log-box") prior spec:
       logbox:A:B:SE           absolute edges 10^A and 10^B;
       logbox-centre:W:SE[:C]  width W decades, centred in log10 on C (default: the average data-imposed rate);
                               the box's centre is the prior median;
       logbox-mean:W:SE        width W decades, placed so that the prior mean rate equals that average.
    SE is the soft-edge width in decades (0 = hard edges)."""
    parts = spec.split(':')
    kind = parts[0]
    u_hi = float(np.log10(P['hi'])) if np.isfinite(P['hi']) else 50.0
    n = P['G'].shape[1]
    rbar = average_data_rate(P, block) if block in P['block_names'] else None
    if kind == 'logbox':
        a, b, se = map(float, parts[1:4])
    elif kind in ('logbox-centre', 'logbox-mean'):
        if rbar is None:
            raise ValueError(f'{kind} needs the magnitude-bin row block {block!r}')
        W, se = float(parts[1]), float(parts[2])
        if kind == 'logbox-centre':
            c = float(parts[3]) if len(parts) > 3 else np.log10(rbar)
            a, b = c - W / 2, c + W / 2
        else:
            lo_, hi_ = np.log10(rbar) - 12.0, min(np.log10(rbar) + W, u_hi)
            for _ in range(60):
                mid = 0.5 * (lo_ + hi_)
                if logbox_mean(mid - W, mid, se, u_hi) < rbar:
                    lo_ = mid
                else:
                    hi_ = mid
            b = 0.5 * (lo_ + hi_)
            a = b - W
    else:
        raise ValueError(f'unknown log-box spec {spec!r}')
    mean = logbox_mean(a, b, se, u_hi)
    desc = (f'log-uniform between 1e{a:.2f} and 1e{b:.2f} (centre 1e{(a + b) / 2:.2f}), edge sd {se:g} decades; '
            f'prior mean rate {mean:.2e}, implied prior total {n * mean:.3g}/yr'
            + (f' (data-imposed total {rbar * n:.4g}/yr, average rate {rbar:.2e})' if rbar is not None else ''))
    return a, b, se, desc


def prelim_path(problem):
    """The diagnostics file preliminaries.py writes next to a cleaned problem: prelim_results.npz beside cru_clean.npz,
    or <stem>_prelim_results.npz beside <stem>.npz for any other name."""
    stem = os.path.splitext(os.path.abspath(problem))[0]
    base = os.path.basename(stem)
    return os.path.join(os.path.dirname(stem), 'prelim_results.npz' if base == 'cru_clean' else base + '_prelim_results.npz')


def balance_rows(P, slip_block='S', bins_block='MFD', weights='equal'):
    """The 'down the middle' balance term (the notebook's DOWN_THE_MIDDLE; gibbs.py --down-the-middle): for each data
    type (the slip-rate sections, the magnitude bins, the paleo rows: every other non-empty row whose target is
    positive), its relative shortfall s = (sum of predictions - sum of targets) / sum of targets, both in units of each
    measurement's uncertainty as the likelihood weighs it (a slip-rate section as the mean of its subsections), costs
    A_t s^2, so the zero model (s = -1) and predictions summing to twice the targets (s = +1) cost A_t.
    weights='equal' (the default): one A for every type, min over types of (sum of targets)^2 / N: the largest common
    weight that keeps every type's term at or below one extra measurement ((sum of residuals)^2 / N, the size of its
    chance imbalance), so A is at most each type's zero-model misfit. weights='zero-model': each type's own A_t, its
    zero-model misfit (the sum of its squared targets in units of uncertainty), so s = -1 or +1 costs a type what the
    zero model costs it in chi2 (REPORT.md, "Balance between over- and underprediction").
    As rows of G and d, row t is sqrt(A_t) / T_t times the type's sum of rows, target sqrt(A_t) (T_t: the type's sum of
    targets), so its residual is sqrt(A_t) s_t. Returns (B, b, A, info): B sparse (types x ruptures), b the targets,
    A the weights (one per type), and per type a dict of its name, measurements, sum of targets, zero-model misfit, A and
    weight (A_t / T_t^2, on the squared sum of its residuals)."""
    G = sp.csr_matrix(P['G'])
    d, blk, names = np.asarray(P['d'], np.float64), np.asarray(P['block']), list(P['block_names'])
    N = G.shape[0]
    found = []
    if slip_block in names:
        S = np.flatnonzero(blk == names.index(slip_block))
        sec = np.asarray(P['section'])[S]
        if (sec < 0).any():                              # no sections recorded: each row its own measurement
            sec = np.arange(len(S))
        sec = np.unique(sec, return_inverse=True)[1]
        ks = np.bincount(sec)
        c = np.zeros(N)
        c[S] = 1.0 / (ks[sec] * np.asarray(P['row_scale'], np.float64)[S])   # section means, sqrt(k) undone
        found.append(('slip-rate sections', c, len(ks), S))
    if bins_block in names:
        M = np.flatnonzero(blk == names.index(bins_block))
        c = np.zeros(N)
        c[M] = 1.0
        found.append(('magnitude bins', c, len(M), M))
    other = np.flatnonzero(~np.isin(blk, [names.index(b_) for b_ in (slip_block, bins_block) if b_ in names]))
    pal = other[(np.diff(G.indptr)[other] > 0) & (d[other] > 0)]
    if len(pal):
        c = np.zeros(N)
        c[pal] = 1.0
        found.append(('paleo rows', c, len(pal), pal))
    found = [(nm, c, n_, rr, float(c @ d)) for nm, c, n_, rr in found if float(c @ d) > 0]
    if not found:
        raise ValueError('no data type with a positive sum of targets: nothing to balance')
    if weights == 'equal':
        A = np.full(len(found), min(T ** 2 / n_ for _, _, n_, _, T in found))
    elif weights == 'zero-model':
        A = np.array([float(np.sum(d[rr] ** 2)) for _, _, _, rr, _ in found])
    else:
        raise ValueError(f"weights = {weights!r}: use 'equal' or 'zero-model'")
    B = sp.vstack([sp.csr_matrix(np.sqrt(a) / T * c) @ G for a, (_, c, _, _, T) in zip(A, found)]).tocsr()
    info = [dict(name=nm, n=n_, sum_targets=T, zero_misfit=float(np.sum(d[rr] ** 2)), A=float(a), weight=a / T ** 2)
            for a, (nm, c, n_, rr, T) in zip(A, found)]
    return B, np.sqrt(A), A, info


def find_rupture_set(folder, search=True):
    """The OpenSHA rupture set (a folder holding indices.csv, properties.csv, ...) at folder: the folder itself, its
    ruptures/ if it is an OpenSHA solution folder, or, with search, the ruptures/ of the one OpenSHA solution folder
    inside it (NZSHM22_InversionSolution-211k/ in a data folder, say). None if there is none; an error if there are
    several, which have to be told apart by name."""
    folder = os.path.abspath(os.path.expanduser(str(folder)))
    for cand in (folder, os.path.join(folder, 'ruptures')):
        if os.path.isfile(os.path.join(cand, 'indices.csv')):
            return cand
    if not search or not os.path.isdir(folder):
        return None
    found = sorted(os.path.join(folder, d, 'ruptures') for d in os.listdir(folder)
                   if os.path.isfile(os.path.join(folder, d, 'ruptures', 'indices.csv')))
    if len(found) > 1:
        raise ValueError(f'{len(found)} OpenSHA solution folders in {folder} '
                         f'({", ".join(os.path.basename(os.path.dirname(f)) for f in found)}): name the one to use')
    return found[0] if found else None


def load_rates(path):
    """A rate vector over the original ruptures (the raw matrix's columns): an OpenSHA solution's rates ('Rupture
    Index,Annual Rate': its rates.csv, the solution/ folder holding it, or the solution folder above that), or rate
    vectors (.npz: their 'mean' column, as the notebook and preliminaries.py write them)."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if os.path.isdir(path):
        found = [c for c in (os.path.join(path, 'rates.csv'), os.path.join(path, 'solution', 'rates.csv')) if os.path.isfile(c)]
        if not found:
            raise FileNotFoundError(f'no rates.csv in {path} or in its solution/ folder')
        path = found[0]
    if path.endswith('.csv'):
        idx, rate = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True, ndmin=2)
        out = np.zeros(int(idx.max()) + 1)
        out[idx.astype(np.int64)] = rate
        return out
    with np.load(path) as z:
        return np.asarray(z['mean'], np.float64)


def short_name(name, width=40):
    """A name for a figure's legend: as it is up to width characters, otherwise its start up to the first '-' and its
    last four characters (NZSHM22_InversionSolution-QXV0b21hdGlvblRhc2s6MTAyNDc5ZEF0ZnQ= -> NZSHM22_InversionSolution-...ZnQ=)."""
    if len(name) <= width:
        return name
    cut = name.find('-') + 1
    return (name[:cut] if 0 < cut <= width - 7 else name[:width - 7]) + '...' + name[-4:]


def reference_rates(spec, P):
    """The fit figures' --reference RATES[:LABEL]: another model's rates, marked on every panel, such as the model's own
    OpenSHA solution (its solution/rates.csv; see load_rates). LABEL defaults to the solution folder's name, shortened
    when long (short_name; the tables keep the full path). Returns
    dict(m: its rates on the cleaned problem's ruptures, label, path, total: its total rate, removed and n_removed: its
    rate on the ruptures cleaning removed, and how many of them it puts rate on, which its predictions here leave out,
    note: a sentence saying so, or '')."""
    path, label = spec, None
    if ':' in spec:
        head, tail = spec.rsplit(':', 1)
        if tail and '/' not in tail and os.sep not in tail:
            path, label = head, tail
    full = os.path.abspath(os.path.expanduser(path))
    m_all = load_rates(full)
    n_raw = P['raw_shape'][1] if P.get('raw_shape') is not None else int(np.max(P['keep_cols'])) + 1
    if len(m_all) != n_raw:
        raise SystemExit(f'--reference {path}: {len(m_all)} ruptures; the cleaned problem has {n_raw} (another rupture set?)')
    if label is None:
        label = short_name(os.path.basename(os.path.dirname(os.path.dirname(full)))
                           if full.endswith(os.path.join('solution', 'rates.csv')) else os.path.basename(full))
    gone = np.ones(n_raw, bool)
    gone[P['keep_cols']] = False
    removed, n_removed = float(m_all[gone].sum()), int(np.count_nonzero(m_all[gone]))
    note = (f'its rates on the {n_removed} ruptures cleaning removed ({removed:.3g} per year, '
            f'{100 * removed / m_all.sum():.2g}% of its total) are not in these predictions' if removed > 0 else '')
    return dict(m=m_all[P['keep_cols']], label=label, path=path, total=float(m_all.sum()), removed=removed,
                n_removed=n_removed, note=note)


FIGURE_SETTINGS = (('SLIP_SECTIONS', 'slip_sections'), ('MFD_UNCERTAINTY', 'mfd_uncertainty'),
                   ('DOWN_THE_MIDDLE', 'down_the_middle'), ('BAND', 'band'), ('MFD_MONOTONIC', 'mfd_monotonic'), ('RANDOMIZED_NNLS', 'randomized_nnls'), ('RUPTURE_SET', 'rupture_set'),
                   ('PRESET', 'preset'), ('FLOOR', 'floor'))   # the notebook's settings from SLIP_SECTIONS to FLOOR


def run_settings(config, problem):
    """The settings that tell runs apart (the notebook's, SLIP_SECTIONS to FLOOR), as (name, value) pairs for the fit
    figures' settings table: from config, a workflow_config.json (default: the one next to problem, as the notebook
    writes it; 'none': no table), or None if there is none."""
    if config == 'none':
        return None
    path = config or os.path.join(os.path.dirname(os.path.abspath(problem)), 'workflow_config.json')
    if not os.path.exists(path):
        if config:
            raise SystemExit(f'--config {config}: no such file')
        return None
    with open(path) as f:
        cfg = json.load(f)
    return [(name, str(cfg[key])) for name, key in FIGURE_SETTINGS if key in cfg]


LEGEND_ROW = 8.5 * 1.5                               # points per legend row: the figures' 8.5-point text and half a line


def legend_height(n_items, ncol):
    """Points a legend of n_items entries in ncol columns takes up (rows of LEGEND_ROW)."""
    return LEGEND_ROW * -(-n_items // ncol)


def figure_legend(fig, items, ncol, loc, anchor, transform=None, **kw):
    """A legend in the figures' style, its (handle, label) items in ncol columns read row by row, its loc corner at anchor
    (in transform's coordinates; the figure's by default), with no border or padding around it."""
    from matplotlib.legend_handler import HandlerTuple
    order = [it for c in range(ncol) for it in items[c::ncol]]   # legend columns filled so it reads row by row
    opts = dict(fontsize=8.5, handletextpad=0.5, columnspacing=1.6, handlelength=2.6,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})
    opts.update(kw)
    return fig.legend(handles=[h for h, _ in order], labels=[l for _, l in order], loc=loc, ncol=ncol, frameon=False,
                      bbox_to_anchor=anchor, bbox_transform=transform or fig.transFigure, borderaxespad=0.0,
                      borderpad=0.0, **opts)


def legend_above(fig, ax, items, ncol, y, x=0.0, **kw):
    """figure_legend above a panel, between the run's settings and the panel's subtitle: its lower left corner y points
    above the panel's top left corner (and x points to its right: negative, to its left, for a legend wider than the
    panel)."""
    from matplotlib.transforms import offset_copy
    return figure_legend(fig, items, ncol, 'lower left', (0.0, 1.0),
                         offset_copy(ax.transAxes, fig=fig, x=x, y=y, units='points'), **kw)


def settings_lines(pairs, width):
    """Settings as a small table in monospace text: a line of names over a line of values, column by column, continued
    on further pairs of lines when wider than width characters. Returns [(names, values), ...]."""
    cols = [(n, v, max(len(n), len(v))) for n, v in pairs]
    groups, cur, used = [], [], 0
    for c in cols:
        if cur and used + c[2] > width:
            groups.append(cur)
            cur, used = [], 0
        cur.append(c)
        used += c[2] + 3
    groups.append(cur)
    return [('   '.join(n.ljust(w) for n, _, w in g).rstrip(), '   '.join(v.ljust(w) for _, v, w in g).rstrip()) for g in groups]


def best_fit_rates(problem):
    """The best fit's rates (preliminaries.py's NNLS) over the cleaned problem's ruptures, from prelim_results.npz next
    to it, or None if cleaning skipped it. The fit figures take their fixed axes from its predictions (axis_span)."""
    fp = prelim_path(problem)
    if not os.path.exists(fp):
        return None
    with np.load(fp) as pr:
        return np.asarray(pr['m_nnls'], np.float64) if 'm_nnls' in pr.files else None


def axis_span(targets, best, log, q=(2.0, 98.0)):
    """The range a fit figure's axis covers, the same for every model fitted to one problem: all of the targets and the
    central 96% of the best fit's predictions (on a log axis, its positive ones), so that a few extreme best-fit values
    do not stretch it. Returns (lo, hi), before padding; values beyond sit on the figure's edges."""
    t, b = np.asarray(targets, np.float64), np.asarray(best, np.float64)
    if log:
        t, b = t[t > 0], b[b > 0]
    lo, hi = float(t.min()), float(t.max())
    if b.size:
        lo, hi = min(lo, float(np.percentile(b, q[0]))), max(hi, float(np.percentile(b, q[1])))
    return lo, hi


def run_predictions(folder, rows, n_rows, scale, combine=None):
    """The predictions recorded for every draw by the runs next to the rate vectors (main_rep<k>.npz), for the given
    rows, pooled over runs and chains, each row's divided by scale (in units of its uncertainty, say); with combine, a
    (rows x m) matrix, their combinations instead (section means, say), taken one chain at a time so that the rows'
    draws are never all held at once. Returns (draws, the runs used), or (None, []) if there are none. The fit figures
    draw their posterior 16-84% bars from them."""
    Y, used = [], []
    for f in sorted(glob.glob(os.path.join(folder, 'main_rep*.npz'))):
        if not re.fullmatch(r'main_rep\d+\.npz', os.path.basename(f)):
            continue                                     # running summaries and the like
        with np.load(f) as z:
            if 'y' not in z.files:
                continue
            y = z['y']
        if y.ndim != 3 or y.shape[2] != n_rows:
            continue                                     # another problem's run
        for c in range(y.shape[0]):
            yc = np.asarray(y[c][:, rows], np.float64) / scale
            Y.append(yc if combine is None else yc @ combine)
        used.append(os.path.basename(f)[:-4])
    return (np.concatenate(Y), used) if Y else (None, [])


def kept_draws(z):
    """The kept draws that select_draws.py adds to the rate vectors (z, the loaded .npz): `kept`, one row per draw over
    the original ruptures (float32), and `kept_index`, each one's run (its position among the runs select_draws.py was
    given), chain and record. (None, None) without them: REPRESENTATIVES = 0, or a single model's rate vectors. The
    figures draw them one by one, as x's (draw_xs)."""
    if 'kept' not in z.files or 'kept_index' not in z.files:
        return None, None
    return z['kept'], np.asarray(z['kept_index'])


def kept_predictions(folder, index, rows, n_rows, scale, combine=None, check=None):
    """The kept draws' predictions for the given rows, each divided by scale (with combine, a rows x m matrix, their
    combinations), read from the predictions their runs recorded (main_rep<k>.npz next to the rate vectors for run k,
    the notebook's order): a kept draw is the recorded draw at its chain and record (record - burn // thin in the run's
    predictions), so nothing is recomputed. check = (G_rows, kept, keep_cols): the first kept draw of each run is
    recomputed from its rates and must agree, or the runs are not the ones the draws came from. None if a run is
    missing or disagrees; the figures then draw the posterior ranges instead."""
    out = None
    for r in np.unique(index[:, 0]):
        f = os.path.join(folder, f'main_rep{r}.npz')
        if not os.path.exists(f):
            return None
        with np.load(f) as z:
            if 'y' not in z.files:
                return None
            y, off = z['y'], int(z['burn']) // int(z['thin'])
        sel = np.flatnonzero(index[:, 0] == r)
        rec = index[sel, 2] - off
        if y.ndim != 3 or y.shape[2] != n_rows or rec.min() < 0 or rec.max() >= y.shape[1]:
            return None
        yk = np.asarray(y[index[sel, 1], rec][:, rows], np.float64)
        del y
        if check is not None:                            # the first of the run's kept draws, recomputed
            Gr, K, kc = check
            yc = Gr @ np.asarray(K[sel[0]], np.float64)[kc]
            if not np.allclose(yc, yk[0], rtol=2e-3, atol=1e-5 * max(1.0, float(np.abs(yc).max()))):
                return None
        yk /= scale
        if combine is not None:
            yk = yk @ combine
        if out is None:
            out = np.empty((len(index),) + yk.shape[1:])
        out[sel] = yk
    return out


# Each kept draw as an x, in its mark's colour darkened a little (several of CATEGORICAL are pale), faint, so that
# where the draws gather shows as the densest part; rasterized, since there are thousands per mark.
X_MARK = dict(marker='x', ls='none', ms=2.6, mew=0.4, alpha=0.15, rasterized=True)


def darker(c, f=0.75):
    """A colour darkened to a fraction f of its RGB values."""
    from matplotlib.colors import to_rgb
    return tuple(f * v for v in to_rgb(c))


def draw_xs(ax, x, y, colours, zorder=2.2, **kw):
    """Kept draws as x's (X_MARK, overridden by kw) at (x, y), flat arrays of equal length, each in colours (one colour
    for all, or one per point), darkened; one call per colour."""
    x, y = np.ravel(x), np.ravel(y)
    style = {**X_MARK, **kw}
    if isinstance(colours, str) or np.ndim(colours) == 1 and len(colours) in (3, 4) and not isinstance(colours[0], str):
        ax.plot(x, y, color=darker(colours), zorder=zorder, **style)
        return
    colours = np.ravel(np.asarray(colours, dtype=object))
    for c in dict.fromkeys(colours):
        s = colours == c
        ax.plot(x[s], y[s], color=darker(c), zorder=zorder, **style)


# The colours that tell neighbouring marks apart: Tableau's Color Blind 10 (tableau:tab10_colorblind in the cmap
# catalog; nearly matplotlib's tableau-colorblind10 style), in the cyclic order that best separates neighbours on the
# light surface, every colour against the next and the last against the first: worst colour-vision-deficient Delta E
# 20.2 (OKLab x100; the target is 8), worst normal-vision 22.4 (the floor is 15); in the palette's own order 10.8 and
# 11.5. Six of the ten are below 3:1 contrast on the surface (the palest grey 1.5:1), so the figures outline their
# marks and keep their table views.
CATEGORICAL = ('#006ba4', '#ff8000', '#5f9ed1', '#ffbc79', '#898989', '#cfcfcf', '#c85200', '#a2c8ec', '#595959',
               '#ababab')


def neighbour_colours(x):
    """One colour per mark, cycling through CATEGORICAL in its validated order along x (equal x in their given order),
    so that marks next to each other never share a colour and every pair of neighbours is one the palette's validation
    covers. The colours only tell neighbours apart; they follow the inputs (x), so a mark keeps its colour in every
    model's figure."""
    rank = np.empty(len(x), int)
    rank[np.argsort(np.asarray(x, np.float64), kind='stable')] = np.arange(len(x))
    return np.array(CATEGORICAL)[rank % len(CATEGORICAL)]


# --------------------------------------------------------------------------- row groups: magnitude bins and slip rates
def detect_bin_rows(G, d, rtol=1e-6):
    """Find the magnitude-bin rows without being told: rows whose nonzero coefficients are all equal and whose
    right-hand sides are positive, and whose supports split the ruptures into disjoint groups covering every one.
    Candidates are grouped into runs of consecutive rows, and a run qualifies if it partitions the ruptures; if no
    single run does, all candidates together are tried. Returns the row indices, or None if nothing qualifies or the
    answer is ambiguous (then the rows must be named)."""
    Gr = sp.csr_matrix(G)
    cand = [j for j in range(Gr.shape[0])
            if Gr.indptr[j + 1] > Gr.indptr[j] and d[j] > 0
            and np.ptp(Gr.data[Gr.indptr[j]:Gr.indptr[j + 1]]) <= rtol * np.abs(Gr.data[Gr.indptr[j]:Gr.indptr[j + 1]]).max()]
    if len(cand) < 2:
        return None
    runs, cur = [], [cand[0]]
    for j in cand[1:]:
        if j == cur[-1] + 1:
            cur.append(j)
        else:
            runs.append(cur)
            cur = [j]
    runs.append(cur)
    partitions = lambda rows: bool(np.all(np.asarray((Gr[rows] != 0).sum(0)).ravel() == 1))
    good = [r for r in runs if len(r) >= 2 and partitions(r)]
    if len(good) == 1:
        return np.array(good[0])
    if not good and len(runs) > 1 and partitions(cand):
        return np.array(cand)
    return None


def check_slip_rows(P, block='S', rtol=1e-6):
    """Plausibility check for the slip-rate rows, which define subsections for the summaries and the transfer moves.
    Raises ValueError for clear failures: no such block, negative coefficients or right-hand sides, or rows that look
    like the magnitude-bin rows (constant coefficients partitioning the ruptures). Returns a profile to eyeball, with
    warnings for rows whose coefficients are all equal (slip-rate rows normally vary from rupture to rupture) and for
    ruptures that appear in no slip-rate row."""
    if block not in P['block_names']:
        raise ValueError(f'no row block named {block!r}; the blocks are {list(P["block_names"])}')
    G = sp.csr_matrix(P['G'])
    d = P['d']
    rows = np.flatnonzero(P['block'] == P['block_names'].index(block))
    orig = P['keep_rows'] if 'keep_rows' in P else np.arange(G.shape[0])
    if rows.size == 0:
        raise ValueError(f'row block {block!r} is empty')
    sub = G[rows]
    problems, warnings = [], []
    if (d[rows] < 0).any():
        problems.append(f'{int((d[rows] < 0).sum())} rows have negative right-hand sides')
    if (sub.data < 0).any():
        problems.append('some coefficients are negative')
    look = detect_bin_rows(sub, d[rows], rtol)
    if look is not None:
        problems.append(f'original rows {int(orig[rows[look].min()])}-{int(orig[rows[look].max()])} look like magnitude-bin rows '
                        '(constant coefficients that partition the ruptures)')
    if problems:
        raise ValueError(f'row block {block!r} does not look like slip-rate rows: ' + '; '.join(problems))
    nnz = np.diff(sub.indptr)
    const = [int(orig[rows[k]]) for k in range(len(rows)) if nnz[k] >= 2
             and np.ptp(sub.data[sub.indptr[k]:sub.indptr[k + 1]]) <= rtol * np.abs(sub.data[sub.indptr[k]:sub.indptr[k + 1]]).max()]
    if const:
        warnings.append(f'{len(const)} rows have equal coefficients for all their ruptures (slip-rate rows normally vary '
                        f'from rupture to rupture), e.g. original rows {const[:5]}')
    cover = np.asarray((sub != 0).sum(0)).ravel()
    if (cover == 0).any():
        warnings.append(f'{int((cover == 0).sum())} of {len(cover)} ruptures appear in no slip-rate row')
    dr = d[rows]
    sections = 1 + int(np.sum(np.abs(np.diff(dr)) > 1e-12 * np.abs(dr).max()))
    span = cover[cover > 0]
    out = (f'{len(rows)} rows (original {int(orig[rows].min())}-{int(orig[rows].max())}) in {sections} runs of identical values; '
           f'each row involves a median of {np.median(nnz):.0f} ruptures; each rupture appears in a median of '
           f'{np.median(span):.0f} rows (range {span.min()}-{span.max()})' if span.size else f'{len(rows)} rows')
    return out + ''.join(f'; WARNING: {w}' for w in warnings)


def relabel_rows(block, names, ranges, rows, name, keep_r):
    """Give `rows` their own block called `name`, dropping any block left without rows."""
    block = np.asarray(block).copy()
    names, ranges = list(names) + [name], list(ranges) + [f'{int(keep_r[rows].min())}-{int(keep_r[rows].max())}']
    block[rows] = len(names) - 1
    used = [b for b in range(len(names)) if (block == b).any()]
    lut = np.full(len(names), -1, dtype=block.dtype)
    lut[used] = np.arange(len(used))
    return lut[block], [names[b] for b in used], [ranges[b] for b in used]


# ---- log of everything the notebook prints ----------------------------------------------------------------------
_LOG = dict(file=None, out=None, err=None, hooks=[], t0=None)


class _Tee:
    """A text stream that writes to the stream it wraps (the screen) and to the log file."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, s):
        n = self._stream.write(s)
        if not self._log.closed:
            self._log.write(s)
        return n

    def flush(self):
        self._stream.flush()
        if not self._log.closed:
            self._log.flush()

    def __getattr__(self, name):                 # everything else (isatty, fileno, encoding, Jupyter's routing) as before
        return getattr(self._stream, name)


def _ipython():
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None


def stop_log():
    """Stop copying output to the log: restore the screen streams, remove the cell hooks, close the file."""
    if _LOG['file'] is None:
        return
    sys.stdout, sys.stderr = _LOG['out'], _LOG['err']
    ip = _ipython()
    for event, fn in _LOG['hooks']:
        try:
            ip.events.unregister(event, fn)
        except Exception:
            pass
    _LOG['file'].close()
    _LOG.update(file=None, out=None, err=None, hooks=[], t0=None)


def start_log(path, header=''):
    """From now on, copy everything printed (stdout and stderr: messages, the scripts' output relayed by the notebook,
    warnings) to the text file at path as well as to the screen. Under IPython/Jupyter, also write when each cell
    starts (with its first line), how long it took, and any error with its traceback, which Jupyter shows by another
    route. Starting a new log ends the previous one. A header, if given, opens the file. Returns path."""
    stop_log()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    f = open(path, 'a', encoding='utf-8', buffering=1)          # line-buffered: the file is current during long runs
    f.write(header if not header or header.endswith('\n') else header + '\n')
    _LOG.update(file=f, out=sys.stdout, err=sys.stderr, t0=time.time())
    sys.stdout, sys.stderr = _Tee(sys.stdout, f), _Tee(sys.stderr, f)
    ip = _ipython()
    if ip is not None:
        def pre(info=None):
            try:
                _LOG['t0'] = time.time()
                first = next((l.strip() for l in str(getattr(info, 'raw_cell', '') or '').splitlines() if l.strip()), '')
                f.write(f"\n----- {time.strftime('%Y-%m-%d %H:%M:%S')}  cell: {first[:100]}\n")
            except Exception:
                pass

        def post(result=None):
            try:
                err = getattr(result, 'error_before_exec', None) or getattr(result, 'error_in_exec', None)
                if err is not None:
                    f.write(''.join(traceback.format_exception(type(err), err, err.__traceback__)))
                if _LOG['t0'] is not None:
                    f.write(f"----- cell {'stopped by an error' if err is not None else 'done'} after "
                            f"{time.time() - _LOG['t0']:.0f} s\n")
            except Exception:
                pass

        for event, fn in (('pre_run_cell', pre), ('post_run_cell', post)):
            ip.events.register(event, fn)
            _LOG['hooks'].append((event, fn))
    return path


def report_to(path):
    """For a script: from now on, copy everything it prints to the text file at path as well (overwritten each run),
    its report next to its outputs. Returns path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sys.stdout = _Tee(sys.stdout, open(path, 'w', encoding='utf-8', buffering=1))
    return path


def log_only(text):
    """Write a line of text to the log only (e.g. which figure was shown), if a log is open."""
    if _LOG['file'] is not None and not _LOG['file'].closed:
        _LOG['file'].write(text if text.endswith('\n') else text + '\n')


# ---- the running summaries gibbs.py writes (<run>_stream.npz) ------------------------------------------------------
def load_stream(name, keys=None):
    """A run's running summaries (<name>_stream.npz): every array, or those named in `keys` that the file has (each
    array is decompressed when read, and at 2e5 ruptures the per-draw totals and the histogram run to gigabytes)."""
    with np.load(name + '_stream.npz') as z:
        return {k: z[k] for k in z.files if keys is None or k in keys}


def stream_rhat(z):
    """Split R-hat of the clipped log rates, from the running per-chain, per-half sums."""
    cnt = z['cnt'][None, :, None]
    m = z['s1'] / cnt
    v = (z['s2'] - cnt * m ** 2) / (cnt - 1)
    n = m.shape[2]
    m, v, ns = m.reshape(-1, n), v.reshape(-1, n), z['cnt'].mean()
    W = v.mean(0)
    B = ns * m.var(0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt(((ns - 1) / ns * W + B / ns) / W)


def stream_ess(z):
    """Effective draws from per-chain batch means, pooled over chains."""
    used = z['bcnt'] > 0
    K, tot = z['s1'].shape[0], z['cnt'].sum()
    if used.sum() < 2:
        return np.full(z['s1'].shape[2], float(K * tot))
    bm = z['bsum'][:, used] / z['bcnt'][used][None, :, None]
    cm = z['s1'].sum(1) / tot
    cv = (z['s2'].sum(1) - tot * cm ** 2) / (tot - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        e = K * tot * cv.mean(0) / (z['bcnt'][used].mean() * bm.var(1, ddof=1).mean(0))
    return np.clip(np.where(np.isfinite(e), e, K * tot), 1.0, K * tot)


def stream_quantiles(z, qs, chunk=20000):
    """Quantiles of each rate from the running log-rate histogram (bin 0 holds rates below 1e-16, reported as 0); in
    blocks of `chunk` rates, so the float copies stay small whatever the number of ruptures."""
    n = z['hist'].shape[1]
    out = [np.empty(n) for _ in qs]
    for c0 in range(0, n, chunk):
        h = z['hist'][:, c0:c0 + chunk].astype(np.float64)
        tot = h.sum(0)
        c = np.cumsum(h, 0) / tot
        cols = np.arange(h.shape[1])
        for o, q in zip(out, qs):
            k = np.argmax(c >= q, axis=0)
            prev = np.where(k > 0, c[np.maximum(k - 1, 0), cols], 0.0)
            frac = np.clip((q - prev) / np.maximum(h[k, cols] / tot, 1e-300), 0, 1)
            lo = float(z['hist_lo']) + (k - 1) * float(z['hist_w'])
            o[c0:c0 + chunk] = np.where(k == 0, 0.0, 10.0 ** (lo + frac * float(z['hist_w'])))
    return out


# ---- transfer-move directions (used by gibbs.py --transfers and its self-test) ------------------------------------
def make_dirs(G, dirs):
    """dirs: list of (indices, values).  Returns the arrays gibbs.py needs, including u = G v for each direction."""
    G = sp.csc_matrix(G)
    d_ptr = np.zeros(len(dirs) + 1, np.int64)
    d_ptr[1:] = np.cumsum([len(i) for i, _ in dirs])
    d_idx = np.concatenate([np.asarray(i, np.int64) for i, _ in dirs]) if dirs else np.zeros(0, np.int64)
    d_val = np.concatenate([np.asarray(v, np.float64) for _, v in dirs]) if dirs else np.zeros(0)
    V = sp.csc_matrix((d_val, d_idx, d_ptr), shape=(G.shape[1], len(dirs)))
    U = sp.csc_matrix(G @ V)
    U.sort_indices()
    uu = np.asarray(U.multiply(U).sum(0)).ravel()
    return dict(d_ptr=d_ptr, d_idx=d_idx, d_val=d_val, u_ptr=U.indptr.astype(np.int64),
                u_row=U.indices.astype(np.int32), u_val=U.data.astype(np.float64), uu=uu)

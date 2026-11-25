#!/usr/bin/env python3
"""Convert a run's dense kept draws to the sparse format used since 2026-11-25 (the user's request), losslessly.

    python sparsify_kept.py PRIOR_FOLDER [--remove-dense]

PRIOR_FOLDER: a run's results folder (RUN_DIR/spikeslab). For each <stem>_kept.npy (gibbs.py --keep-draws before
2026-11-25: float32, one row per draw over the cleaned problem's ruptures) it writes <stem>_kept.npz, a scipy CSR matrix
of the same rates (the zeros left out), and checks it against the dense file value for value. If rate_vectors.npz holds
the dense `kept` that select_draws.py stored (one row per draw over the rupture set's ruptures), it rewrites the file
with the draws as kept_data / kept_indices / kept_indptr / kept_shape (every other array as it was), checked the same
way, and keeps the old file as rate_vectors_dense_backup.npz. The dense <stem>_kept.npy and the backup are removed only
with --remove-dense; every script reads either format (cru_common.load_kept, kept_draws) and prefers the sparse one.
At 630,000 ruptures 5,120 draws take about 60 MB in place of 12.6 GB (twice: the run's and the rate vectors' copies).
"""
import argparse
import glob
import os
import sys
import zipfile

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cru_common import kept_arrays   # noqa: E402

BLOCK = 256


def npz_member(path, key):
    """An uncompressed .npz member as a read-only memory map (np.savez stores members uncompressed; np.load would read
    the whole array into memory)."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + '.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            return None
    with open(path, 'rb') as f:
        f.seek(info.header_offset)
        h = f.read(30)
        start = info.header_offset + 30 + int.from_bytes(h[26:28], 'little') + int.from_bytes(h[28:30], 'little')
        f.seek(start)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape, order='F' if fortran else 'C')


def to_csr_checked(D):
    """D (dense, draws x columns) as CSR, block by block, each block checked to reproduce D exactly."""
    parts = []
    for a in range(0, D.shape[0], BLOCK):
        blk = np.asarray(D[a:a + BLOCK], np.float32)
        M = sp.csr_matrix(blk)
        if not np.array_equal(M.toarray(), blk):
            raise SystemExit(f'rows {a}-{a + len(blk)}: the sparse copy differs')
        parts.append(M)
    return sp.vstack(parts, format='csr')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder')
    ap.add_argument('--remove-dense', action='store_true', help='remove the dense files once their sparse copies are checked')
    a = ap.parse_args()
    freed = 0
    for npy in sorted(glob.glob(os.path.join(a.folder, '*_kept.npy'))):
        stem = npy[:-len('_kept.npy')]
        out = stem + '_kept.npz'
        D = np.load(npy, mmap_mode='r')
        if os.path.exists(out):
            M = sp.load_npz(out).tocsr()
            if M.shape != D.shape:
                raise SystemExit(f'{out}: shape {M.shape}, the dense file {D.shape}')
            print(f'{out} exists')
        else:
            M = to_csr_checked(D)
            sp.save_npz(out, M, compressed=False)
            print(f'wrote {out}: {M.shape[0]:,} draws, {M.nnz / M.shape[0]:,.0f} rates on per draw '
                  f'({os.path.getsize(out) / 1e6:,.0f} MB in place of {os.path.getsize(npy) / 1e9:.2f} GB), checked')
        if a.remove_dense:
            freed += os.path.getsize(npy)
            os.remove(npy)
            print(f'removed {npy}')
    rv = os.path.join(a.folder, 'rate_vectors.npz')
    backup = os.path.join(a.folder, 'rate_vectors_dense_backup.npz')
    if os.path.exists(rv):
        with np.load(rv) as z:
            files = z.files
        if 'kept' in files:
            Dk = npz_member(rv, 'kept')
            if Dk is None:
                raise SystemExit(f'{rv}: its kept draws are compressed; not converted')
            Ko = to_csr_checked(Dk)
            with np.load(rv) as z:
                arrs = {k: z[k] for k in z.files if k not in ('kept', 'kept_index')}
                idx = z['kept_index']
            del Dk
            os.replace(rv, backup)
            np.savez(rv, **arrs, **kept_arrays(Ko, idx))
            print(f'rewrote {rv} with its {Ko.shape[0]:,} kept draws sparse ({os.path.getsize(rv) / 1e6:,.0f} MB; the old '
                  f'file kept as {os.path.basename(backup)}, {os.path.getsize(backup) / 1e9:.2f} GB), checked')
        else:
            print(f'{rv}: its kept draws are already sparse (or absent)')
    if a.remove_dense and os.path.exists(backup):
        freed += os.path.getsize(backup)
        os.remove(backup)
        print(f'removed {backup}')
    if freed:
        print(f'freed {freed / 1e9:.1f} GB')


if __name__ == '__main__':
    main()

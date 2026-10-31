"""MFD bin totals against their targets under the two MFD uncertainty levels (quick runs, two replicates each).

    python tools/fig_mfd_levels.py AS_GIVEN_RUN_DIR TWO_MEASUREMENTS_RUN_DIR OUT.png

AS_GIVEN_RUN_DIR needs spikeslab/ runs; TWO_MEASUREMENTS_RUN_DIR needs spikeslab/ and theta/ runs (main_rep0, main_rep1).
The scripts are found in CRU_CODE, or beside this one, in tools/."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, os.environ.get('CRU_CODE', os.path.dirname(os.path.abspath(__file__))))   # the scripts' folder
from cru_common import load_problem, load_stream, magnitude_bins

SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ebeae5'
BLUE, ORANGE = '#2a78d6', '#eb6834'
A_DIR, T_DIR, OUT = sys.argv[1:4]


def bins(run_dir, prior):
    P = load_problem(f'{run_dir}/cru_clean.npz')
    rows, R, orig, lo = magnitude_bins(P, 'MFD')
    brows = list(np.flatnonzero(P['block'] == P['block_names'].index('MFD')))
    order = [brows.index(j) for j in rows]
    tb = np.concatenate([load_stream(f'{run_dir}/{prior}/main_rep{i}')['tot_b'].reshape(-1, len(brows))[:, order]
                         for i in range(2)]).astype(np.float64)
    q = np.percentile(tb / R, [16, 50, 84], axis=0)
    m = np.load(f'{run_dir}/prelim_results.npz')['m_nnls']
    G = P['G'].tocsr()
    best = np.array([G[j] @ m for j in rows]).ravel() / P['d'][rows]  # best fit's bin totals / targets (rows are R_j/sigma_j)
    rel = 1.0 / P['d'][rows]
    return lo + 0.05, q, best, rel


x, q1, b1, r1 = bins(A_DIR, 'spikeslab')
_, q2, b2, r2 = bins(T_DIR, 'spikeslab')
_, q3, _, _ = bins(T_DIR, 'theta')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.9), sharey=True, facecolor=SURFACE)
lo_y = min(q1[0].min(), q2[0].min(), q3[0].min(), (1 - r2).min()) - 0.06
hi_y = max(q1[2].max(), q2[2].max(), q3[2].max(), (1 + r2).max()) + 0.06
panels = [(axs[0], r1, b1, [(q1, BLUE, 0.0)], "As given (the default): 9.8–15.4% per bin"),
          (axs[1], r2, b2, [(q2, BLUE, -0.018), (q3, ORANGE, 0.018)], "'two-measurements' (×3): 29–46% per bin")]
for ax, rel, best, series, title in panels:
    ax.set_facecolor(SURFACE)
    ax.fill_between(np.r_[x[0] - 0.05, x, x[-1] + 0.05], np.r_[1 - rel[0], 1 - rel, 1 - rel[-1]],
                    np.r_[1 + rel[0], 1 + rel, 1 + rel[-1]], step=None, color=BAND, lw=0, zorder=1)
    ax.axhline(1.0, color=INK, lw=0.9, zorder=2)
    for q, col, dx in series:
        ax.vlines(x + dx, q[0], q[2], color=col, lw=1.6, alpha=0.9, zorder=3)
        ax.plot(x + dx, q[1], 'o', ms=5.5, mfc=col, mec=SURFACE, mew=0.9, zorder=4)
    ax.plot(x, best, 'D', ms=4.6, mfc='none', mec=INK2, mew=1.0, zorder=5)
    ax.set_title(title, loc='left', color=INK, fontsize=10.5, pad=8)
    ax.set_xlim(x[0] - 0.08, x[-1] + 0.08)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xticks(np.arange(6.8, 8.61, 0.2))
    ax.set_xlabel('magnitude', color=INK2)
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3)
axs[0].set_ylabel('bin total / its target rate', color=INK2)
tot = lambda q: None
fig.suptitle('MFD bin totals against their targets: posterior median and 16–84% range per bin', x=0.06, ha='left',
             y=0.975, color=INK, fontsize=12, fontweight='bold')
fig.text(0.06, 0.935, "Quick runs, two replicates each. Counted as two measurements, the bins weigh less, and the slip-rate and paleo rows "
         "raise them (median +8%, against +2%);\nθ and spike-and-slab then differ on how the four largest bins share the rate, "
         "though not on their sum. Points sit at bin centres.",
         ha='left', va='top', color=INK2, fontsize=9, linespacing=1.4)
handles = [Patch(facecolor=BAND, edgecolor='none', label="the bin row's own ±1 uncertainty"),
           Line2D([], [], color=INK, lw=0.9, label='target'),
           Line2D([], [], color=BLUE, lw=1.6, marker='o', ms=5.5, mfc=BLUE, mec=SURFACE, label='spike-and-slab (default prior)'),
           Line2D([], [], color=ORANGE, lw=1.6, marker='o', ms=5.5, mfc=ORANGE, mec=SURFACE, label='θ = 1'),
           Line2D([], [], ls='none', marker='D', ms=4.6, mfc='none', mec=INK2, label='best fit, no prior (NNLS)')]
fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=8.5, labelcolor=INK2,
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.87))
out = OUT
fig.savefig(out, dpi=150, facecolor=SURFACE)
print('wrote', out)
for name, q in (('as given, spike-and-slab', q1), ('two, spike-and-slab', q2), ('two, theta', q3)):
    print(name, 'median of medians', np.round(np.median(q[1]), 3), 'range', np.round(q[1].min(), 3), np.round(q[1].max(), 3))
print('best fit as given', np.round(b1.min(), 3), np.round(b1.max(), 3), '; two', np.round(b2.min(), 3), np.round(b2.max(), 3))

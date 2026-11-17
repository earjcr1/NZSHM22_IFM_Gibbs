#!/usr/bin/env python3
"""The run plan: the sampler's run lengths from RUN_MINUTES, and the convergence verdict's TARGETS (November 17th, 2026,
the user's request; they replace the presets 'quick', 'conservative' and 'very_conservative').

RUN_MINUTES is the whole notebook's length. The notebook plans the spike-and-slab runs just before sampling, when the
cleaning and the best fit have run and their time is known:
  1. the time already spent (every script the notebook ran, sh()) and the post-processing still to come (the latest run's
     record in DATA_DIR, timings.json, or else as long as the time spent) come off the budget first, with a 10% margin;
  2. a short calibration run of gibbs.py (CALIB_SWEEPS sweeps, with the run's own settings and chains) measures the
     time per sweep and the setup that every run pays;
  3. chains = CHAINS_PER_THREAD x THREADS (the kernel updates chains in groups of four per thread);
  4. replicates: 1, unless each of several would still get REPLICATE_MIN sweeps and (with a prediction) still be
     predicted to meet TARGETS (at most REPLICATES_MAX): the verdict is each replicate's, so splitting a budget that one
     replicate needs whole would only turn a PASS into CHECKs;
  5. sweeps: what the rest of the budget affords; burn-in BURN_MAX sweeps, or BURN_FRAC of a shorter run;
     thin: about DRAWS_PER_CHAIN kept draws per chain.
Then it predicts each of the verdict's effective-draw counts from the autocorrelation times of the latest run in DATA_DIR
(its convergence report; one with the same model settings if there is one), and says how many minutes the targets would
need. check_convergence.py --plan reports the verdict against TARGETS and against the prediction.

TARGETS (the verdict's thresholds, set by the run's purpose, not by its length): effective draws for 95% of ruptures and
for the total rate, and the R-hat that chi2 and the total rate must stay below; each bin's fraction on needs a fifth of
the total rate's effective draws (check_convergence.py).
"""
import glob
import json
import os

import numpy as np

TARGETS = {'exploratory': (100, 200, 1.01),          # the quick preset's targets
           'production': (1000, 2000, 1.01),         # the conservative preset's
           'precise': (10000, 20000, 1.005)}         # the very conservative preset's
PRESET_TARGETS = {'quick': 'exploratory', 'conservative': 'production', 'very_conservative': 'precise'}   # older runs
MARGIN = 0.10                 # of RUN_MINUTES, kept free
CHAINS_PER_THREAD = 4
BURN_MAX, BURN_FRAC = 5000, 0.2
DRAWS_PER_CHAIN = 3000
REPLICATE_MIN, REPLICATES_MAX = 50000, 4
CALIB_SWEEPS = 60             # the calibration run: its time per sweep and setup
CALIB_THIN = 4                # a draw recorded every 4 sweeps, as often as a short run's: recording costs time on a small
                              # problem (at 19k, thin 10 underestimated a thin-4 run's sweeps by 15%); an overestimate otherwise
MIN_SWEEPS = 500              # a plan never goes below this, whatever the budget
METRICS = (('ess_rate_5pct', '95% of ruptures'), ('ess_total', 'the total rate'), ('ess_pi_min', "each bin's fraction on"))
MODEL_KEYS = ('mfd_uncertainty', 'down_the_middle', 'misfit_slip', 'misfit_mfd', 'misfit_paleo', 'mfd_monotonic', 'mfd_gr',
              'mfd_step_sd', 'slip_along', 'cfm_timeframe', 'cfm_template_rho', 'slip_rate_factor', 'slip_sum', 'slip_sum_l0',
              'plate_rate', 'floor')


def needs(targets):
    """TARGETS -> ({metric: effective draws needed}, R-hat that chi2 and the total rate must stay below)."""
    rate, total, rhat = TARGETS[targets]
    return dict(ess_rate_5pct=rate, ess_total=total, ess_pi_min=total / 5), rhat


def calibration(npz, wall_s, sweeps):
    """A calibration run's time per sweep and setup (its wall time less its sampling, gibbs.py's 'seconds')."""
    sec = float(np.load(npz)['seconds'])
    return sec / sweeps, max(0.0, wall_s - sec)


def _report_rows(path):
    """check_convergence.py's report tables -> {run: {column: value}} (older runs, without convergence_summary.json)."""
    rows, keys = {}, None
    for line in open(path):
        w = line.split()
        if not w:
            keys = None
            continue
        if w[0] == 'run':
            keys = w
            continue
        if keys and len(w) == len(keys):
            try:
                rows.setdefault(w[0], {}).update({k: float(v) for k, v in zip(keys[1:], w[1:])})
            except ValueError:
                keys = None
    return rows


def run_record(run_dir):
    """A finished run's autocorrelation times (sweeps per effective draw, per chain) for the verdict's quantities, its
    time per sweep and its timings, or None. tau = chains x kept draws x thin / effective draws, the median over its
    replicates."""
    res = os.path.join(run_dir, 'spikeslab')
    cfgp = os.path.join(res, 'workflow_config.json')
    if not os.path.exists(cfgp):
        cfgp = os.path.join(run_dir, 'workflow_config.json')
    summ, rep = os.path.join(res, 'convergence_summary.json'), os.path.join(res, 'convergence_report.txt')
    if not os.path.exists(cfgp) or not (os.path.exists(summ) or os.path.exists(rep)):
        return None
    cfg = json.load(open(cfgp))
    st = cfg.get('settings') or {}
    if os.path.exists(summ):
        rows = json.load(open(summ))['runs']
    else:                                                # older runs: the report's tables, thin from the configuration
        rows = [dict(r, thin=st.get('thin')) for r in _report_rows(rep).values()]
    tau = {}
    for m, _ in METRICS:
        v = [r['chains'] * r['draws'] * r['thin'] / r[m] for r in rows
             if r.get(m) and r.get('thin') and r.get('chains') and r.get('draws')]
        if v:
            tau[m] = float(np.median(v))
    if not tau:
        return None
    sw = []
    for f in sorted(glob.glob(os.path.join(res, 'main_rep*.npz'))):
        try:
            z = np.load(f)
            if 'seconds' in z.files and st.get('sweeps'):
                sw.append(float(z['seconds']) / st['sweeps'])
        except Exception:
            pass
    tm = os.path.join(run_dir, 'timings.json')
    return dict(run=run_dir, tau=tau, cfg=cfg, chains=st.get('chains'), sweep_s=float(np.median(sw)) if sw else None,
                timings=json.load(open(tm)) if os.path.exists(tm) else None, when=os.path.getmtime(summ if os.path.exists(summ) else rep))


def reference(data_dir, config, exclude=None):
    """The latest finished run in data_dir (the same rupture set) with a convergence report: one with this run's model
    settings if there is one, otherwise the latest. Returns (record, the model settings that differ) or (None, None)."""
    recs = []
    for d in glob.glob(os.path.join(data_dir, '*', '')):
        d = d.rstrip('/')
        if exclude and os.path.abspath(d) == os.path.abspath(exclude):
            continue
        r = run_record(d)
        if r:
            recs.append(r)
    if not recs:
        return None, None
    diff = lambda r: [k for k in MODEL_KEYS if str(r['cfg'].get(k)) != str(config.get(k))]
    recs.sort(key=lambda r: r['when'], reverse=True)
    same = [r for r in recs if not diff(r)]
    r = (same or recs)[0]
    return r, diff(r)


def _round_down(s):
    """A plan's sweeps, rounded down to a readable number."""
    step = 10 if s < 1000 else 100 if s < 20000 else 1000
    return int(s // step * step)


def plan(minutes, threads, sweep_s, setup_s, spent_s, post_s, targets, ref=None, ref_diff=None, calib_chains=None):
    """The run lengths for RUN_MINUTES, with the reasoning as lines. sweep_s and setup_s: the calibration's (at
    CHAINS_PER_THREAD x threads chains); spent_s: the time the notebook has spent; post_s: the post-processing to come;
    ref: the reference run (reference()) for the prediction. Returns dict(chains, sweeps, burn, thin, hyper_every,
    replicates, predicted, minutes_needed, lines)."""
    K = CHAINS_PER_THREAD * threads
    total = minutes * 60.0
    budget = total * (1 - MARGIN) - spent_s - post_s
    nd, _ = needs(targets)
    tau = ref['tau'] if ref else {}

    def lengths(r):                                      # sweeps, burn-in and thin for r replicates
        S_ = (budget - r * setup_s) / (r * sweep_s)
        S_ = _round_down(max(S_, MIN_SWEEPS))
        b_ = BURN_MAX if S_ * BURN_FRAC >= BURN_MAX else int(round(S_ * BURN_FRAC, -1 if S_ >= 500 else 0))
        return S_, b_, max(1, int(round((S_ - b_) / DRAWS_PER_CHAIN)))

    meets = lambda r: all(K * (lengths(r)[0] - lengths(r)[1]) / tau[m] >= nd[m] for m in tau)
    reps = next((r for r in range(REPLICATES_MAX, 1, -1)
                 if (budget - r * setup_s) / (r * sweep_s) >= REPLICATE_MIN and meets(r)), 1)
    short = (budget - reps * setup_s) / (reps * sweep_s) < MIN_SWEEPS
    S, burn, thin = lengths(reps)
    est = spent_s + post_s + reps * (setup_s + S * sweep_s)
    L = [f'run plan for RUN_MINUTES = {minutes:g} (tools/run_plan.py):',
         f'   spent so far {spent_s / 60:.1f} min (cleaning, the best fit, the calibration); post-processing to come about '
         f'{post_s / 60:.1f} min; a {100 * MARGIN:.0f}% margin; {max(budget, 0) / 60:.1f} min left for sampling',
         f'   calibration ({CALIB_SWEEPS} sweeps, {calib_chains or K} chains, {threads} threads): {sweep_s:.4f} s per sweep, '
         f'{setup_s:.0f} s of setup per run',
         f'   chains: {K} ({CHAINS_PER_THREAD} per thread); replicates: {reps}'
         + (f' (each still gets {REPLICATE_MIN:,} sweeps or more' + (' and is predicted to meet TARGETS)' if tau else ')')
            if reps > 1 else f' (more only if each could get {REPLICATE_MIN:,} sweeps' + (' and still meet TARGETS' if tau else '')
            + '; so no replicate comparison)'),
         f'   sweeps {S:,} per chain, burn-in {burn:,}'
         + ('' if burn == BURN_MAX else f' ({100 * BURN_FRAC:.0f}% of a run too short for the usual {BURN_MAX:,})')
         + f', a draw kept every {thin} ({S // thin - burn // thin:,} kept per chain)',
         f'   expected length {est / 60:.0f} min in all'
         + (f': OVER RUN_MINUTES, which cannot afford even {MIN_SWEEPS} sweeps' if short else '')]
    pred, need_min = {}, None
    if ref:
        L.append(f'   prediction from {os.path.basename(ref["run"])} (its autocorrelation times, sweeps per effective draw per '
                 'chain: ' + ', '.join(f'{lab} {tau[m]:,.0f}' for m, lab in METRICS if m in tau) + ')'
                 + (f'; its settings differ in {", ".join(ref_diff)}, so these are rough' if ref_diff else '; the same model settings'))
        for m, lab in METRICS:
            if m in tau:
                pred[m] = K * (S - burn) / tau[m]
        miss = [m for m in pred if pred[m] < nd[m]]
        L.append(f'   effective draws per replicate, predicted against TARGETS {targets!r}: '
                 + '; '.join(f'{lab} {pred[m]:,.0f} ({nd[m]:,.0f} needed)' for m, lab in METRICS if m in pred)
                 + (' -> enough for PASS; R-hat is not predicted, and a short run can still miss it' if not miss
                    else ' -> expect CHECK'))
        x = max(nd[m] * tau[m] / K for m in pred)
        S_need = x + BURN_MAX if x * BURN_FRAC / (1 - BURN_FRAC) >= BURN_MAX else x / (1 - BURN_FRAC)
        need_min = (spent_s + post_s + setup_s + S_need * sweep_s) / (1 - MARGIN) / 60
        L.append(f'   TARGETS {targets!r} would need about {S_need:,.0f} sweeps per chain with {K} chains: RUN_MINUTES about '
                 f'{need_min:,.0f}' + (' (this plan should suffice)' if not miss else ''))
    else:
        L.append('   no earlier run in DATA_DIR with a convergence report, so no prediction of the verdict')
    return dict(chains=K, sweeps=int(S), burn=int(burn), thin=int(thin), hyper_every=1, replicates=int(reps),
                sweep_s=sweep_s, setup_s=setup_s, spent_s=spent_s, post_s=post_s, minutes=minutes, targets=targets,
                expected_min=est / 60, predicted=pred, tau=ref['tau'] if ref else None,
                reference=ref['run'] if ref else None, reference_differs=ref_diff, minutes_needed=need_min, lines=L)


def post_estimate(ref, spent_s):
    """The post-processing still to come: the reference run's, measured (timings.json), or else as long as the time
    spent so far."""
    if ref and ref.get('timings') and ref['timings'].get('post_s') is not None:
        return float(ref['timings']['post_s']), f'as {os.path.basename(ref["run"])} took'
    return spent_s, 'taken as long as the time spent so far (no earlier record)'


if __name__ == '__main__':                               # a plan without the notebook: python run_plan.py MINUTES THREADS SWEEP_S SETUP_S SPENT_S POST_S [TARGETS DATA_DIR]
    import sys
    a = sys.argv[1:]
    ref, diff = reference(a[7], {}) if len(a) > 7 else (None, None)
    print('\n'.join(plan(float(a[0]), int(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5]),
                         a[6] if len(a) > 6 else 'production', ref, diff)['lines']))

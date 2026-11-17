#!/usr/bin/env python3
"""The run plan: the sampler's run lengths from RUN_MINUTES, and the convergence verdict's TARGETS (November 17th, 2026,
the user's request; they replace the presets 'quick', 'conservative' and 'very_conservative').

RUN_MINUTES is the whole notebook's length. The notebook plans the spike-and-slab runs just before sampling, when the
cleaning and the best fit have run and their time is known:
  1. the time already spent (every script the notebook ran, sh()) and the post-processing still to come (the latest run's
     record in DATA_DIR, timings.json, or else as long as the time spent) come off the budget first, with a 10% margin;
  2. two short calibration runs of gibbs.py (CALIB_SWEEPS sweeps each, with the run's own settings and chains, recording
     a draw every CALIB_THINS sweeps) measure the time per sweep, split into a part per sweep and a part per recorded draw
     (t = a + b / thin: the running summaries and swap moves come once per recorded draw, which costs a great deal at
     630,000 ruptures), and the setup that every run pays;
  3. chains = CHAINS_PER_THREAD x THREADS (the kernel updates chains in groups of four per thread);
  4. replicates: 1, unless each of several would still get REPLICATE_MIN sweeps and (with a prediction) still be
     predicted to reach REPLICATE_SAFETY times TARGETS' effective draws (at most REPLICATES_MAX): the verdict is each
     replicate's, so splitting a budget that one replicate needs whole would only turn a PASS into CHECKs, and the
     prediction is rough (another run's autocorrelation times; R-hat not predicted);
  5. sweeps: what the rest of the budget affords; burn-in BURN_MAX sweeps, or BURN_FRAC of a shorter run;
     thin: about DRAWS_PER_CHAIN kept draws per chain.
Then it predicts each of the verdict's effective-draw counts from the autocorrelation times of the latest run in DATA_DIR
(its convergence report; one with the same model settings if there is one, a setting an older run did not record taken
at its value before it existed, OLD_DEFAULTS), and says how many minutes the targets would need. check_convergence.py --plan reports the verdict against TARGETS and against the prediction.

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
REPLICATE_SAFETY = 1.5        # more than one replicate only if each is predicted to reach 1.5 x TARGETS' effective draws
CALIB_SWEEPS, CALIB_BURN = 120, 24   # each calibration run's sweeps and burn-in
CALIB_THINS = (4, 24)         # its two recording rates, to split the time per sweep from the time per recorded draw (a single
                              # rate misjudged it: at 19k thin 10 underestimated a thin-4 run by 15%; at 630k thin 4 overestimated
                              # a thin-18 run by 24%)
MIN_SWEEPS = 500              # a plan never goes below this, whatever the budget
METRICS = (('ess_rate_5pct', '95% of ruptures'), ('ess_total', 'the total rate'), ('ess_pi_min', "each bin's fraction on"))
MODEL_KEYS = ('mfd_uncertainty', 'down_the_middle', 'misfit_slip', 'misfit_mfd', 'misfit_paleo', 'mfd_monotonic', 'mfd_gr',
              'mfd_step_sd', 'slip_along', 'cfm_timeframe', 'cfm_template_rho', 'slip_rate_factor', 'slip_sum', 'slip_sum_l0',
              'plate_rate', 'floor')
OLD_DEFAULTS = {'slip_along': 'uniform', 'mfd_step_sd': None, 'mfd_gr': False}   # a setting a run made before it existed did
                                                                                   # not record: its value then


def needs(targets):
    """TARGETS -> ({metric: effective draws needed}, R-hat that chi2 and the total rate must stay below)."""
    rate, total, rhat = TARGETS[targets]
    return dict(ess_rate_5pct=rate, ess_total=total, ess_pi_min=total / 5), rhat


def calibration(runs):
    """The calibration runs, [(npz, wall seconds, sweeps, thin), ...] -> dict(a, b, setup): a sweep's time a + b / thin
    (a per sweep, b per recorded draw; least squares over the runs, neither below zero) and the setup each run pays (its
    wall time less its sampling, gibbs.py's 'seconds'; the median)."""
    t, x, setup = [], [], []
    for npz, wall, sweeps, thin in runs:
        sec = float(np.load(npz)['seconds'])
        t.append(sec / sweeps)
        x.append(1.0 / thin)
        setup.append(max(0.0, wall - sec))
    t, x = np.array(t), np.array(x)
    if len(t) > 1 and np.ptp(x) > 0:
        b, a = np.polyfit(x, t, 1)
        if b < 0 or a <= 0:                               # noise: no measurable cost per recorded draw
            a, b = float(t.max()), 0.0
    else:
        a, b = float(t.max()), 0.0
    return dict(a=float(a), b=float(b), setup=float(np.median(setup)), thins=[int(round(1 / v)) for v in x],
                sweeps=[r[2] for r in runs])


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
    val = lambda c, k: c[k] if c.get(k) is not None else OLD_DEFAULTS.get(k, c.get(k))
    diff = lambda r: [k for k in MODEL_KEYS if str(val(r['cfg'], k)) != str(val(config, k))]
    recs.sort(key=lambda r: r['when'], reverse=True)
    same = [r for r in recs if not diff(r)]
    r = (same or recs)[0]
    return r, diff(r)


def _round_down(s):
    """A plan's sweeps, rounded down to a readable number."""
    step = 10 if s < 1000 else 100 if s < 20000 else 1000
    return int(s // step * step)


def plan(minutes, threads, cost, spent_s, post_s, targets, ref=None, ref_diff=None):
    """The run lengths for RUN_MINUTES, with the reasoning as lines. cost: the calibration's (calibration(), at
    CHAINS_PER_THREAD x threads chains: a sweep's time a + b / thin, and the setup per run); spent_s: the time the notebook
    has spent; post_s: the post-processing to come; ref: the reference run (reference()) for the prediction. Returns
    dict(chains, sweeps, burn, thin, hyper_every, replicates, sweep_s (at that thin), setup_s, predicted, minutes_needed,
    lines)."""
    K = CHAINS_PER_THREAD * threads
    total = minutes * 60.0
    budget = total * (1 - MARGIN) - spent_s - post_s
    nd, _ = needs(targets)
    tau = ref['tau'] if ref else {}
    setup_s = cost['setup']
    sec = lambda th: cost['a'] + cost['b'] / th          # a sweep's time, recording a draw every th sweeps
    afford = lambda r, th: (budget - r * setup_s) / (r * sec(th))
    burn_of = lambda S_: BURN_MAX if S_ * BURN_FRAC >= BURN_MAX else int(round(S_ * BURN_FRAC, -1 if S_ >= 500 else 0))

    def lengths(r):                                      # sweeps, burn-in and thin for r replicates (thin and the sweeps
        th = 1                                           # it affords settle in a few rounds)
        for _ in range(8):
            S_ = _round_down(max(afford(r, th), MIN_SWEEPS))
            b_ = burn_of(S_)
            th_new = max(1, int(round((S_ - b_) / DRAWS_PER_CHAIN)))
            if th_new == th:
                break
            th = th_new
        return S_, b_, th

    def meets(r):
        S_, b_, _ = lengths(r)
        return all(K * (S_ - b_) / tau[m] >= REPLICATE_SAFETY * nd[m] for m in tau)
    reps = next((r for r in range(REPLICATES_MAX, 1, -1) if lengths(r)[0] >= REPLICATE_MIN and meets(r)), 1)
    S, burn, thin = lengths(reps)
    short = afford(reps, thin) < MIN_SWEEPS
    sweep_s = sec(thin)
    est = spent_s + post_s + reps * (setup_s + S * sweep_s)
    L = [f'run plan for RUN_MINUTES = {minutes:g} (tools/run_plan.py):',
         f'   spent so far {spent_s / 60:.1f} min (cleaning, the best fit, the calibration); post-processing to come about '
         f'{post_s / 60:.1f} min; a {100 * MARGIN:.0f}% margin; {max(budget, 0) / 60:.1f} min left for sampling',
         f'   calibration ({len(cost.get("thins", [])) or 1} runs of {CALIB_SWEEPS} sweeps, {K} chains, {threads} threads, a draw '
         f'recorded every {" and ".join(str(t_) for t_ in cost.get("thins", []))} sweeps): {cost["a"]:.4f} s per sweep plus '
         f'{cost["b"]:.3f} s per recorded draw, so {sweep_s:.4f} s per sweep at this plan\'s thin; {setup_s:.0f} s of setup per run',
         f'   chains: {K} ({CHAINS_PER_THREAD} per thread); replicates: {reps}'
         + (f' (each still gets {REPLICATE_MIN:,} sweeps or more' + (f' and is predicted to reach {REPLICATE_SAFETY:g} x TARGETS\' '
            'effective draws)' if tau else ')') if reps > 1 else f' (more only if each could get {REPLICATE_MIN:,} sweeps'
            + (f' and still reach {REPLICATE_SAFETY:g} x TARGETS\' effective draws' if tau else '') + '; so no replicate comparison)'),
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
        th_need = max(1, int(round((S_need - burn_of(S_need)) / DRAWS_PER_CHAIN)))
        need_min = (spent_s + post_s + setup_s + S_need * sec(th_need)) / (1 - MARGIN) / 60
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


if __name__ == '__main__':      # a plan without the notebook: python run_plan.py MINUTES THREADS A B SETUP_S SPENT_S POST_S [TARGETS DATA_DIR]
    import sys
    a = sys.argv[1:]
    ref, diff = reference(a[8], {}) if len(a) > 8 else (None, None)
    print('\n'.join(plan(float(a[0]), int(a[1]), dict(a=float(a[2]), b=float(a[3]), setup=float(a[4]), thins=list(CALIB_THINS)),
                         float(a[5]), float(a[6]), a[7] if len(a) > 7 else 'production', ref, diff)['lines']))

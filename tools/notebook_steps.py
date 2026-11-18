"""The notebook's plumbing (moved out of cru_workflow.ipynb on 2026-11-17, at the user's request): checking the settings,
the paths and the log, the scripts' arguments, the run plan's glue and the run's timings. The notebook keeps its settings
and the calls of the workflow's scripts; everything here reads the notebook's settings from the namespace setup() is given.

    from notebook_steps import setup, sh, show, results_dir, rupture_dir, ref_args, ...
    setup(globals())          # in the configuration cell, after the settings

setup(g): run_config.derived_settings (each group's misfit mode, the fixed settings, RUN_NAME, RUN_DIR); every setting
checked (ValueError naming it); SEED_USED, MFD_GR_SPEC, PLATE_LIMITS, SLIP_AS_GIVEN; DATA_DIR and RUN_DIR resolved from the
notebook's folder, the scripts and data files checked; RUN_DIR made and entered; the log started in RUN_DIR/spikeslab/;
the rupture set found (RUPTURE_DIR); the configuration written (workflow_config.json, CONFIG). All of it into g.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_G = {}                                               # the notebook's namespace, given to setup()
SH_SECONDS = [0.0]                                    # the time the scripts have taken (idle time not counted)
TIMES = {}                                            # the run's timings: before sampling, sampling, after


def _check(ok, name, value, what):
    if not ok:
        raise ValueError(f'{name} = {value!r}: {what}')


def setup(g):
    """The configuration cell's work after the settings (see the module's description). g: the notebook's globals()."""
    _G.clear()
    _G['g'] = g
    SH_SECONDS[0] = 0.0                                   # a new run: its own time from zero (the module outlives a run in
    TIMES.clear()                                         # one Jupyter kernel, and the plan subtracts the time spent)
    sys.path.insert(0, HERE)
    from run_config import derived_settings
    g.update(derived_settings(g))
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    _check(g['CROSS_SECTIONS'] in ('all', 'between'), 'CROSS_SECTIONS', g['CROSS_SECTIONS'],
           "use 'all' (all of New Zealand) or 'between' (only between the Alpine and Cook Strait edges)")
    for n_ in ('SLIP_SUM', 'CFM_SLIPRATES', 'CFM_RATE_LINKS', 'CFM_TIMEFRAME', 'MFD_MONOTONIC', 'PLATE_RATE', 'MFD_GR',
               'DISPERSED_STARTS'):
        _check(isinstance(g[n_], bool), n_, g[n_], 'use True or False')
    _check(num(g['SLIP_SUM_L0']) and g['SLIP_SUM_L0'] > 0, 'SLIP_SUM_L0', g['SLIP_SUM_L0'], 'give a length in km above 0')
    _check(num(g['SLIP_RATE_FACTOR']) and g['SLIP_RATE_FACTOR'] > 0, 'SLIP_RATE_FACTOR', g['SLIP_RATE_FACTOR'],
           'the fraction of the slip rates the ruptures take up, above 0 (0.8: the inputs\' own)')
    seed = g['SEED']
    _check(seed is None or seed == 'legacy' or (isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0), 'SEED',
           seed, "use None (from the start time), an integer of 0 or more, or 'legacy'")
    g['SEED_USED'] = g['_T0'] if seed is None else seed   # recorded in the configuration
    rho = g['CFM_TEMPLATE_RHO']
    _check(rho is None or (num(rho) and 0 < rho <= 1), 'CFM_TEMPLATE_RHO', rho, 'use None (off) or a correlation in (0, 1]')
    from mfd_gr import MAGUNCERT
    rm = g['RECONSIDER_MAGUNCERT']
    mag = tuple(MAGUNCERT) if rm is True else () if rm is False else rm
    for n_, v_, k_ in (('MFD_GR_B', g['MFD_GR_B'], 3), ('MFD_GR_N5', g['MFD_GR_N5'], 3), ('MFD_GR_NSCALE', g['MFD_GR_NSCALE'], 2)) \
            + ((('RECONSIDER_MAGUNCERT', mag, 3),) if mag else ()):
        _check(isinstance(v_, (tuple, list)) and len(v_) == k_ and all(num(x_) and x_ > 0 for x_ in v_), n_, v_,
               f'give {k_} numbers above 0')
    g['MFD_GR_SPEC'] = ','.join(f'{x_:g}' for x_ in tuple(g['MFD_GR_B']) + tuple(g['MFD_GR_N5']) + tuple(g['MFD_GR_NSCALE'])
                                + tuple(mag))         # as preliminaries.py --mfd-gr takes it
    if g['MFD_GR']:
        from mfd_gr import parse_knobs                # b rising (or fixed), N above 0, the N scaling bracketing 1, m rising
        parse_knobs(g['MFD_GR_SPEC'])
    _check(g['SLIP_ALONG_RUPTURE'] in ('uniform', 'tapered'), 'SLIP_ALONG_RUPTURE', g['SLIP_ALONG_RUPTURE'],
           "use 'uniform' (the inputs') or 'tapered' (UCERF3's sqrt-sine taper)")
    _check(isinstance(g['DATATYPES_EQUAL'], bool), 'DATATYPES_EQUAL', g['DATATYPES_EQUAL'],
           'use True (the slip rates, MFD bins and paleo rates counting equally) or False')
    tau = g['MFD_GR_TAU']
    _check(tau is None or (num(tau) and tau > 0), 'MFD_GR_TAU', tau,
           'give the G-R shape\'s tolerance above 0, in decades a step (0.025 about 6%), or None for each bin by its own PDF')
    g['PLATE_LIMITS'] = f"{g['SLIP_RATE_FACTOR']:g}" if g['PLATE_RATE'] else None   # as gibbs.py --plate-rate takes it
    _check(num(g['RUN_MINUTES']) and g['RUN_MINUTES'] > 0, 'RUN_MINUTES', g['RUN_MINUTES'], 'give the run\'s length in minutes')
    from run_plan import TARGETS
    _check(g['TARGETS'] in TARGETS, 'TARGETS', g['TARGETS'], "use 'exploratory', 'production' or 'precise'")
    _check(num(g['FLOOR']) and -16 < g['FLOOR'] < 0, 'FLOOR', g['FLOOR'], 'give log10 of the slab floor, a negative number')
    rep = g['REPRESENTATIVES']
    _check(isinstance(rep, int) and not isinstance(rep, bool) and rep >= 0, 'REPRESENTATIVES', rep, 'give a number of draws, 0 or more')
    th = g['THREADS']
    _check(isinstance(th, int) and not isinstance(th, bool) and th >= 1, 'THREADS', th, 'give a number of threads, 1 or more')
    mu = g['MFD_UNCERTAINTY']
    _check(mu in ('as-given', 'two-measurements') or (num(mu) and 0 < mu <= 2), 'MFD_UNCERTAINTY', mu,
           "use 'as-given', 'two-measurements' or the M7.0-7.1 bin's relative uncertainty as a fraction (0.3 for 30%)")
    g['P'] = g['RUN_PLAN']                             # the run lengths: planned before sampling (or fixed by a test)
    nb_dir = g['_NOTEBOOK_DIR']
    resolve = lambda p: os.path.normpath(os.path.join(nb_dir, os.path.expanduser(str(p))))
    g['DATA_DIR'], g['RUN_DIR'] = resolve(g['DATA_DIR']), resolve(g['RUN_DIR'])
    for f_ in (g['A_FILE'], g['D_FILE']) + (() if g['SLIP_SECTIONS'] in ('auto', 'none') else (g['SLIP_SECTIONS'],)):
        if not os.path.exists(os.path.join(g['DATA_DIR'], f_)):
            raise FileNotFoundError(f"{f_} is not in DATA_DIR ({g['DATA_DIR']})")
    os.makedirs(g['RUN_DIR'], exist_ok=True)
    os.chdir(g['RUN_DIR'])                             # every output lands in RUN_DIR
    from cru_common import CFM_CSV, find_rupture_set, start_log
    g['RESULTS'] = 'spikeslab'                         # this run's subfolder of RUN_DIR: its results and log
    g['LOG_FILE'] = start_log(os.path.join(g['RUN_DIR'], g['RESULTS'], time.strftime('workflow_log_%Y%m%d-%H%M%S.txt')),
                              header=f"cru_workflow log, started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for k_, v_ in (('code', HERE), ('data', g['DATA_DIR']), ('runs', g['RUN_DIR']), ('log', g['LOG_FILE'])):
        print(f'{k_ + ":":6s}{v_}')
    g['CFM_CSV'] = CFM_CSV
    if g['CFM_SLIPRATES'] and not os.path.exists(CFM_CSV):
        raise FileNotFoundError(f'the CFM as CSV, {CFM_CSV}, not found')
    rs = g['RUPTURE_SET']
    g['RUPTURE_DIR'] = (find_rupture_set(g['DATA_DIR']) if rs == 'auto'          # the OpenSHA rupture set, or None
                        else find_rupture_set(os.path.join(g['DATA_DIR'], rs), search=False) if rs else None)
    if rs not in (None, 'auto') and g['RUPTURE_DIR'] is None:
        raise FileNotFoundError(f'RUPTURE_SET = {rs!r}: no OpenSHA rupture set there (indices.csv, or ruptures/indices.csv)')
    g['SLIP_AS_GIVEN'] = not g['CFM_SLIPRATES'] and g['SLIP_RATE_FACTOR'] == 0.8   # the inputs' slip rows untouched
    need = [n_ for n_, on in (('SLIP_ALONG_RUPTURE tapered', g['SLIP_ALONG_RUPTURE'] == 'tapered'), ('SLIP_SUM', g['SLIP_SUM']),
                              ('PLATE_RATE', g['PLATE_RATE']), ('MFD_GR', g['MFD_GR']),
                              ('MISFIT_PALEO', g['PALEO_MISFIT'] != 'gaussian'), ('CFM_SLIPRATES', not g['SLIP_AS_GIVEN'])) if on]
    if need and not g['RUPTURE_DIR']:
        raise FileNotFoundError(f'{need[0]} needs the model\'s OpenSHA rupture set (RUPTURE_SET), for the faults\' geometry or '
                                'the paleo data; none found')
    print('rupture set:', g['RUPTURE_DIR'] or 'none (RUPTURE_SET): no rates are written for OpenSHA')
    print('reference solution (looked for again when each fit figure is drawn):', (ref_args() or [None, 'none yet'])[1])
    record_config()


def record_config(**more):
    """The run's configuration, CONFIG, as workflow_config.json in RUN_DIR (and the log); more: entries to add or replace."""
    g = _G['g']
    if 'CONFIG' not in g or not more:
        g['CONFIG'] = dict(
            run_minutes=g['RUN_MINUTES'], targets=g['TARGETS'], dispersed_starts=g['DISPERSED_STARTS'], seed=g['SEED_USED'],
            code_dir=HERE, data_dir=g['DATA_DIR'], run_dir=g['RUN_DIR'], settings=g['P'], floor=g['FLOOR'],
            representatives=g['REPRESENTATIVES'], a_file=g['A_FILE'], d_file=g['D_FILE'], blocks=g['BLOCKS'],
            slip_sections=g['SLIP_SECTIONS'], mfd_uncertainty=g['MFD_UNCERTAINTY'], down_the_middle=g['DOWN_THE_MIDDLE'],
            misfit=g['MISFIT'], misfit_slip=g['MISFIT_SLIP'], misfit_mfd=g['MISFIT_MFD'],
            mfd_gr=g['MFD_GR'] and g['MFD_GR_SPEC'], misfit_paleo=g['MISFIT_PALEO'], mfd_monotonic=g['MFD_MONOTONIC'],
            mfd_gr_tau=g['MFD_GR_TAU'] if g['MFD_GR'] else None, slip_along=g['SLIP_ALONG_RUPTURE'],
            datatypes_equal=g['DATATYPES_EQUAL'],
            plate_rate=g['PLATE_RATE'] and g['SLIP_RATE_FACTOR'] or None, paleo_misfit=g['PALEO_MISFIT'],
            mfd_misfit=g['MFD_MISFIT'], slip_misfit=g['SLIP_MISFIT'], cfm_sliprates=g['CFM_SLIPRATES'], cfm_file=g['CFM_CSV'],
            cfm_pref='mode', cfm_rounding=False, cfm_timeframe=g['CFM_TIMEFRAME'] and g['CFM_SLIPRATES'],
            cfm_timeframe_shape='fit' if g['CFM_TIMEFRAME'] and g['CFM_SLIPRATES'] else None,
            cfm_template_rho=g['CFM_TEMPLATE_RHO'], cfm_rate_links=g['CFM_RATE_LINKS'] and g['CFM_TEMPLATE_RHO'] is not None,
            slip_rate_factor=g['SLIP_RATE_FACTOR'], slip_sd_multiplier=g['SD_MULTIPLIER_'], cross_sections=g['CROSS_SECTIONS'],
            slip_sum=g['SLIP_SUM'], slip_sum_l0=g['SLIP_SUM_L0'], rupture_set=g['RUPTURE_SET'], rupture_dir=g['RUPTURE_DIR'],
            reference=(ref_args() or [None, None])[1], slip_block=g['SLIP_BLOCK'], bin_block=g['BIN_BLOCK'],
            threads=g['THREADS'], log_file=g['LOG_FILE'], started=time.strftime('%Y-%m-%d %H:%M'))
    g['CONFIG'].update(more)
    json.dump(g['CONFIG'], open('workflow_config.json', 'w'), indent=1)
    from cru_common import log_only
    log_only('configuration: ' + json.dumps(g['CONFIG'], default=str))


def sh(*args, quiet=False):
    """Run one of the workflow's scripts (in tools/) with this Python, streaming its output (quiet: shown only if it
    fails, and returned); stop on failure. The time it takes counts toward the run plan's time spent."""
    if str(args[0]).endswith('.py') and not os.path.isabs(str(args[0])):
        args = (os.path.join(HERE, args[0]),) + tuple(args[1:])
    cmd = [sys.executable] + [str(a) for a in args]
    print('$', ' '.join(cmd[1:]), flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    held = []
    for line in p.stdout:
        held.append(line) if quiet else print(line, end='', flush=True)
    code = p.wait()
    SH_SECONDS[0] += time.time() - t0
    if code != 0:
        print(''.join(held), end='')
        raise RuntimeError('command failed: ' + ' '.join(cmd[1:]))
    return held


def show(path):
    """Display a figure inline in Jupyter; elsewhere, say where it is."""
    try:
        from IPython.display import Image, display
        display(Image(filename=path))
        from cru_common import log_only
        log_only(f'figure: {path}')
    except ImportError:
        print('figure:', path)


def results_dir():
    """This run's results folder (RUN_DIR/spikeslab/), made if needed, with a copy of the run's configuration."""
    res = _G['g']['RESULTS']
    os.makedirs(res, exist_ok=True)
    if os.path.exists('workflow_config.json'):
        shutil.copy('workflow_config.json', os.path.join(res, 'workflow_config.json'))
    return res


def rupture_dir():
    """The OpenSHA rupture set the solutions are written for, looked for when they are written, so that a solution
    folder that appears in DATA_DIR during the run still serves: RUPTURE_DIR, or, with RUPTURE_SET 'auto' and none found
    at the start, the one solution folder's in DATA_DIR (None if there is none, or several)."""
    g = _G['g']
    if g['RUPTURE_DIR'] or g['RUPTURE_SET'] != 'auto':
        return g['RUPTURE_DIR']
    from cru_common import find_rupture_set
    try:
        return find_rupture_set(g['DATA_DIR'])
    except ValueError:
        return None


def ref_args():
    """--reference for a fit figure: the model's own OpenSHA solution's rates (its solution/rates.csv), looked for when
    the figure is drawn: beside the rupture set, or, with RUPTURE_SET 'auto' and none found at the start, in the one
    solution folder in DATA_DIR that has them."""
    g = _G['g']
    if g['RUPTURE_SET'] is None:
        return []
    if g['RUPTURE_DIR']:
        found = [os.path.join(os.path.dirname(g['RUPTURE_DIR']), 'solution', 'rates.csv')]
    else:
        found = [os.path.join(g['DATA_DIR'], d, 'solution', 'rates.csv') for d in sorted(os.listdir(g['DATA_DIR']))]
    found = [f for f in found if os.path.isfile(f)]
    return ['--reference', found[0]] if len(found) == 1 else []


# ----------------------------------------------------------------------------------------------- cleaning (section 2)
def _cleaned_settings():
    from cru_common import load_problem
    p = load_problem('cru_clean.npz')
    return (p['slip_sections'], p['mfd_uncertainty'], p['cfm_sliprates'] is not None, p['slip_rate_factor'],
            p['slip_sd_multiplier'], tuple((p['cfm_sliprates'] or [])[5:7]), p['slip_along'])


def cleaning_args():
    """preliminaries.py's arguments for the run's settings; notes the settings of a problem cleaned here before, which
    after_cleaning() compares."""
    g = _G['g']
    _G['before'] = _cleaned_settings() if os.path.exists('cru_clean.npz') else None
    D = g['DATA_DIR']
    a = ['--A', os.path.join(D, g['A_FILE']), '--d', os.path.join(D, g['D_FILE']), '--out', 'cru_clean.npz',
         '--bins-block', g['BIN_BLOCK'], '--slip-block', g['SLIP_BLOCK'], f"--diag-floor={g['FLOOR']}",
         '--slip-sections', g['SLIP_SECTIONS'] if g['SLIP_SECTIONS'] in ('auto', 'none') else os.path.join(D, g['SLIP_SECTIONS']),
         '--mfd-uncertainty', g['MFD_UNCERTAINTY']]
    if g['BLOCKS']:
        a += ['--blocks', g['BLOCKS']]
    if g['CFM_SLIPRATES']:                             # the slip-rate rows rebuilt from the CFM, as split normals
        a += ['--cfm-sliprates', g['CFM_CSV']]
        a += ['--cfm-timeframe', os.path.join(os.path.dirname(HERE), 'data', 'cfm_timeframe_sliprates.csv')] if g['CFM_TIMEFRAME'] else []
        a += (['--cfm-template-rho', str(g['CFM_TEMPLATE_RHO']), '--cfm-rate-sources',   # templates without coincidences
               os.path.join(os.path.dirname(HERE), 'data', 'cfm_rate_sources.csv')] if g['CFM_TEMPLATE_RHO'] is not None else [])
        a += ['--cfm-rate-links'] if g['CFM_RATE_LINKS'] and g['CFM_TEMPLATE_RHO'] is not None else []
    if g['MFD_GR']:                                    # each bin's target PDF from (b, N), with the inversion solution
        a += ['--mfd-gr', g['MFD_GR_SPEC'], '--rupture-set', g['RUPTURE_DIR']]
    if g['SLIP_ALONG_RUPTURE'] == 'tapered':
        a += ['--slip-along', 'tapered', '--rupture-set', g['RUPTURE_DIR']]
    if not g['SLIP_AS_GIVEN']:                         # the slip-rate factor on each whole distribution
        a += ['--slip-rate-factor', str(g['SLIP_RATE_FACTOR']), '--slip-sd-multiplier', str(g['SD_MULTIPLIER_']),
              '--rupture-set', g['RUPTURE_DIR']]
    return a


def after_cleaning():
    """After cleaning: an earlier cleaning's best-fit figures and rates removed (they would not match), a warning if
    RUN_DIR holds runs made on a problem cleaned otherwise, and the cleaned problem summarized. Returns its fit check."""
    g = _G['g']
    for d in ('.', 'nnls'):
        for f in (os.listdir(d) if os.path.isdir(d) else []):
            if (f.startswith('fig_') and '_nnls.' in f) or (d == 'nnls' and f in ('rates.csv', 'solution_report.txt')):
                os.remove(os.path.join(d, f))
    for f in ('rates.csv', 'modules.json', 'sub_seismo_on_fault_mfds.csv'):   # the layout before nnls/rates.csv
        if os.path.exists(os.path.join('nnls', 'solution', f)):
            os.remove(os.path.join('nnls', 'solution', f))
    for d in ('nnls/solution', 'nnls'):
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    from cru_common import load_problem
    p = load_problem('cru_clean.npz')
    before = _G.get('before')
    if before is not None and before != _cleaned_settings() and os.path.exists(os.path.join(g['RESULTS'], 'main_rep0.npz')):
        print(f'WARNING: RUN_DIR held a problem cleaned with other settings ({before}); the results in {g["RESULTS"]}/ were '
              'made on it, not on this one. Give each setting its own RUN_DIR.')
    fit = p['chi2_expect']
    print(f"cleaned problem: {p['G'].shape[0]} rows x {p['G'].shape[1]} ruptures; slip-rate sections {p['slip_sections']!r}; "
          f"MFD uncertainty {p['mfd_uncertainty']!r}" + ('; slip rates from the CFM (split normals)' if p['cfm_sliprates'] else '')
          + (f"; fit check: a draw's chi2 should be near {fit[0]:.0f} +- {fit[1]:.0f} if the uncertainties are right"
             if fit is not None else ''))
    return fit


# ----------------------------------------------------------------------------------------------- sampling (section 3)
def replicate_seed(r):
    """Replicate r's seed: 101 + r with SEED 'legacy'; otherwise hashed from SEED_USED and r (numpy SeedSequence)."""
    s = _G['g']['SEED_USED']
    return 101 + r if s == 'legacy' else int(np.random.SeedSequence([s, r]).generate_state(1)[0])


def gibbs_args(out, seed, plan, keep=True, init=None):
    """gibbs.py's arguments for one run: spike-and-slab with the plan's lengths (chains, sweeps, burn, thin, hyper_every),
    swap moves, every term and constraint of the run's settings; from the dispersed starts (DISPERSED_STARTS) or the NNLS
    best fit (init overrides); keep: whole draws kept for the representatives. Writes RUN_DIR/spikeslab/OUT."""
    g = _G['g']
    init = init or ('starts.npy' if g['DISPERSED_STARTS'] else 'nnls')
    a = ['--problem', 'cru_clean.npz', '--prior', f"spikeslab:{g['FLOOR']}:soft", '--chains', plan['chains'],
         '--sweeps', plan['sweeps'], '--burn', plan['burn'], '--thin', plan['thin'], '--threads', g['THREADS'], '--seed', seed,
         '--init', init, '--out', os.path.join(results_dir(), out), f"--diag-floor={g['FLOOR']}",
         '--slip-block', g['SLIP_BLOCK'], '--bins-block', g['BIN_BLOCK'], '--no-samples']
    if g['REPRESENTATIVES'] > 0 and keep:
        a += ['--keep-draws', 1000]                   # whole draws kept for the representatives
    a += ['--swap-cos', 0.99, '--swap-cos-cross', 0.99,   # swaps between near-duplicates, one threshold within a bin and across
          '--hyper-every', plan.get('hyper_every', 1), '--all-positive']
    return a + _terms(gibbs=True)


def _terms(gibbs=False):
    """The run's misfit modes, soft terms and constraints, as gibbs.py (gibbs=True) or best_fit_as_fitted.py take them."""
    g = _G['g']
    rd = g['RUPTURE_DIR']
    a = []
    if g['DOWN_THE_MIDDLE']:
        a += ['--down-the-middle'] + (['zero-model'] if g['DOWN_THE_MIDDLE'] == 'zero-model' else [])
    if g['PALEO_MISFIT'] != 'gaussian':
        a += ['--paleo-misfit', g['PALEO_MISFIT']]
    if g['MFD_MISFIT'] != 'gaussian':
        a += ['--mfd-misfit', g['MFD_MISFIT']]
    if g['SLIP_MISFIT'] != 'split':
        a += ['--slip-misfit', g['SLIP_MISFIT']] + ([] if gibbs else ['--slip-block', g['SLIP_BLOCK']])
    if g['MFD_MONOTONIC']:
        a += ['--mfd-monotonic']
    if gibbs and g['DATATYPES_EQUAL']:                   # each data type's log likelihood weighted (the sampler only)
        a += ['--group-weights', 'equal']
    if gibbs and g['MFD_GR'] and g['MFD_GR_TAU'] is not None:   # the bins jointly about each draw's G-R curve (the sampler only:
        a += ['--mfd-gr-tau', g['MFD_GR_TAU']]                   # the starts' polish scores each bin by its own PDF)
    if g['PLATE_RATE']:
        a += ['--plate-rate', g['PLATE_LIMITS']]
    if g['SLIP_SUM']:
        a += ['--slip-sum', '--slip-sum-l0', g['SLIP_SUM_L0']]
    if g['SLIP_SUM'] or g['PLATE_RATE']:
        a += ['--cross-sections', g['CROSS_SECTIONS']]
    if g['PALEO_MISFIT'] != 'gaussian' or g['SLIP_SUM'] or g['PLATE_RATE']:
        a += ['--rupture-set', rd]
    return a


def starts_args():
    """dispersed_starts.py's arguments: one start per chain, half randomized fits (twice the stated noise), half prior
    draws, each polished for feasibility with the run's misfit, terms and constraints."""
    g = _G['g']
    import run_plan
    chains = g['RUN_PLAN']['chains'] if g['RUN_PLAN'] is not None else run_plan.CHAINS_PER_THREAD * g['THREADS']
    a = ['--problem', 'cru_clean.npz', '--chains', chains, '--noise', 2, '--seed', replicate_seed(1000),
         '--workers', g['THREADS'], '--out', 'starts.npy', '--floor', 10.0 ** g['FLOOR'], '--bins-block', g['BIN_BLOCK'],
         '--prior-share', 0.5]
    if g['MFD_GR'] and g['MFD_GR_TAU'] is not None:      # the prior draws on a G-R curve, as the sampler scores the bins
        a += ['--mfd-gr-tau', g['MFD_GR_TAU']]
    fit = _terms()
    if fit or g['CFM_SLIPRATES']:                       # the polish: the run's model space, not NNLS's
        a += ['--fit-args', '--bins-block', g['BIN_BLOCK'], '--floor', 10.0 ** g['FLOOR'], '--snap-floor', *fit]
    return a


def plan_runs():
    """The run plan (tools/run_plan.py): two calibration runs of gibbs.py from the run's starts, the plan for RUN_MINUTES
    and its prediction; or RUN_PLAN, a test's fixed plan. Writes run_plan.json, records the lengths in the configuration,
    and starts the sampling's clock. Returns (PLAN, P: the lengths)."""
    g = _G['g']
    import run_plan
    if g['RUN_PLAN'] is None:
        K = run_plan.CHAINS_PER_THREAD * g['THREADS']
        cdir = os.path.join(results_dir(), '_calibration')
        os.makedirs(cdir, exist_ok=True)
        cals = []
        for th in run_plan.CALIB_THINS:              # two recording rates: the time per sweep and per recorded draw
            cal = dict(chains=K, sweeps=run_plan.CALIB_SWEEPS, burn=run_plan.CALIB_BURN, thin=th, hyper_every=1)
            print(f"calibrating the sampler: {cal['sweeps']} sweeps of {K} chains with this run's settings, a draw recorded "
                  f'every {th}')
            t = time.time()
            held = sh('gibbs.py', *gibbs_args(os.path.join('_calibration', f'thin{th}'), 12345, cal, keep=False), quiet=True)
            prog = [(int(sw_), float(r_) * int(sw_)) for sw_, r_ in        # its progress: sweep, cumulative seconds
                    re.findall(r'sweep +(\d+)/\d+ .*?\(\d+s, ([\d.]+)s/sweep\)', ''.join(held), re.S)]
            cals.append((os.path.join(cdir, f'thin{th}.npz'), time.time() - t, cal['sweeps'], th, prog))
        cost = run_plan.calibration(cals)
        shutil.rmtree(cdir, ignore_errors=True)
        ref, diff = run_plan.reference(g['DATA_DIR'], g['CONFIG'], exclude=g['RUN_DIR'])
        post, how = run_plan.post_estimate(ref, SH_SECONDS[0])
        plan = run_plan.plan(g['RUN_MINUTES'], g['THREADS'], cost, SH_SECONDS[0], post, g['TARGETS'], ref, diff)
        plan['lines'][1] += f' (post-processing {how})'
    else:
        plan = dict(g['RUN_PLAN'], fixed=True, targets=g['TARGETS'], lines=[f"run plan fixed by RUN_PLAN (a test): {g['RUN_PLAN']}"])
    print('\n'.join(plan['lines']))
    P = {k: plan[k] for k in ('chains', 'sweeps', 'burn', 'thin', 'hyper_every', 'replicates')}
    json.dump(plan, open('run_plan.json', 'w'), indent=1, default=float)
    record_config(settings=P, run_plan=os.path.join(g['RUN_DIR'], 'run_plan.json'))
    TIMES.update(pre_s=SH_SECONDS[0], t_runs=time.time())
    print(f"seeds: {', '.join(str(replicate_seed(r)) for r in range(P['replicates']))} (SEED {g['SEED_USED']!r})")
    return plan, P


def runs_done():
    """The sampling's end: its time noted for timings.json."""
    TIMES.update(sampling_s=SH_SECONDS[0] - TIMES.get('pre_s', 0.0), sh_after=SH_SECONDS[0])
    print(f"runs finished in {(time.time() - TIMES.get('t_runs', time.time())) / 60:.1f} min")


# ------------------------------------------------------------------------------------------------- outputs (section 5)
def clear_solution(out):
    """An earlier run's OpenSHA solution files in OUT removed (they would not match this one; solution/: the layout
    before rates.csv)."""
    for f in ('rates.csv', 'solution_report.txt', 'solution/rates.csv', 'solution/modules.json',
              'solution/sub_seismo_on_fault_mfds.csv'):
        if os.path.exists(os.path.join(out, f)):
            os.remove(os.path.join(out, f))
    if os.path.isdir(os.path.join(out, 'solution')) and not os.listdir(os.path.join(out, 'solution')):
        os.rmdir(os.path.join(out, 'solution'))


def write_timings():
    """The run's timings (timings.json in RUN_DIR), for later runs' plans: before sampling, sampling, after."""
    g = _G['g']
    if 'sh_after' not in TIMES:
        return
    plan = g.get('PLAN') or {}
    T = dict(pre_s=TIMES['pre_s'], sampling_s=TIMES['sampling_s'], post_s=SH_SECONDS[0] - TIMES['sh_after'],
             sweep_s=plan.get('sweep_s'), setup_s=plan.get('setup_s'), chains=g['P']['chains'], run_minutes=g['RUN_MINUTES'],
             finished=time.strftime('%Y-%m-%d %H:%M'))
    json.dump(T, open('timings.json', 'w'), indent=1)
    print(f"timings: before sampling {T['pre_s'] / 60:.1f} min, sampling {T['sampling_s'] / 60:.1f} min, after "
          f"{T['post_s'] / 60:.1f} min (timings.json; RUN_MINUTES {g['RUN_MINUTES']:g})")

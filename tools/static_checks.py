#!/usr/bin/env python3
"""Nominal static checks of the code set: inventory, parsing, leftovers, defaults, options and imports.

    python tools/static_checks.py [ROOT]

ROOT is the repository's folder, holding the notebook, docs/ (with CLAUDE.md) and tools/ (default: the folder above this
script's). Update SCRIPTS, the inventory and the default checks below whenever the code set or a default changes."""
import ast
import json
import os
import re
import subprocess
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ['notebook_steps.py', 'gibbs.py', 'cru_common.py', 'preliminaries.py', 'block_fit.py', 'summarize_runs.py', 'compare_runs.py', 'compare_priors.py', 'select_draws.py', 'plot_rate_magnitude.py', 'plot_slip_rates.py', 'plot_paleo_rates.py', 'plot_slip_sections.py', 'check_convergence.py', 'make_tables.py', 'make_solution.py', 'plate_rate.py', 'paleo_misfit.py', 'cfm_sliprates.py', 'best_fit_as_fitted.py', 'timeframe_sliprates.py', 'mfd_gr.py', 'run_config.py', 'run_plan.py', 'plot_targets.py', 'dispersed_starts.py', 'covariance.py', 'rupture_set_inputs.py', 'fault_polygons.py', 'trace_slip_rows.py']
TOOLS = ['run_nb.py', 'run_quick_checks.sh', 'static_checks.py', 'compare_mfd.py', 'psis.py']
DOCS = ['docs/CLAUDE.md', 'docs/REPORT.md', 'docs/TESTING.md', 'docs/sampler-design.md']
fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


src = {s: open(f'{ROOT}/tools/{s}').read() for s in SCRIPTS}        # the scripts, in tools/ beside the dev tools
nb = json.load(open(f'{ROOT}/cru_workflow.ipynb'))
cells = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
md = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'markdown']
STEPS = open(f'{ROOT}/tools/notebook_steps.py').read()   # the notebook's plumbing since 2026-11-17 (its settings' checks,
NBALL = '\n'.join(c for _, c in cells) + '\n' + STEPS    # paths, log, scripts' arguments, plan glue, timings): the notebook
                                                         # and it, as one text, for the checks of what the run does

print('1. Inventory')
expected = sorted(['tools/' + s for s in SCRIPTS + TOOLS] + ['cru_workflow.ipynb'] + DOCS)
missing = [f for f in expected if not os.path.isfile(f'{ROOT}/{f}')]
stray = sorted(f for f in os.listdir(ROOT) if f.endswith('.py'))      # scripts outside tools/, where the notebook looks
check(not missing and not stray, f'{len(expected)} files in place: the scripts and tools in tools/, the notebook, and docs/ '
      '(CLAUDE.md among them)' + (f'; missing {missing}' if missing else '') + (f'; scripts outside tools/: {stray}' if stray else ''))
try:                                           # for information: anything else under git
    tracked = subprocess.run(['git', '-C', ROOT, 'ls-files'], capture_output=True, text=True, check=True).stdout.splitlines()
    print(f'     also under git: {", ".join(sorted(set(tracked) - set(expected))) or "nothing"}')
except (OSError, subprocess.CalledProcessError):
    pass

print('2. Parsing')
trees = {}
for s in SCRIPTS:
    try:
        trees[s] = ast.parse(src[s], s)
        check(True, f'{s} parses')
    except SyntaxError as e:
        check(False, f'{s}: {e}')
bad = []
for i, c in cells:
    try:
        compile(c, f'cell {i}', 'exec')
    except SyntaxError as e:
        bad.append(f'cell {i}: {e}')
check(not bad, f'all {len(cells)} notebook code cells parse ({len(md)} markdown cells)' + ('; ' + '; '.join(bad) if bad else ''))

print('3. Leftovers (every hit, to judge its context)')
texts = {s: src[s] for s in SCRIPTS}
texts.update({d: open(f'{ROOT}/{d}').read() for d in DOCS if os.path.isfile(f'{ROOT}/{d}')})
texts.update({f'notebook cell {i}': c for i, c in cells + md})
pat = re.compile(r'SLAB_FLOOR|DIAG_FLOOR|1e-12 by default|FLOOR = -12|default -12|-12 by default|−12 by default|\(1e-12\)')   # a default before -10
nhits = 0
for name, t in texts.items():
    for ln, line in enumerate(t.split('\n'), 1):
        for m in pat.finditer(line):
            nhits += 1
            a, b = max(0, m.start() - 110), min(len(line), m.end() + 60)
            print(f'     {name}:{ln}: {m.group(0)}  …{line[a:b]}…')
print(f'     {nhits} hits')
print('   mentions of the removed chi2 cap (--target cutoff), for review: every one should be history')
pat2 = re.compile(r'--target|TARGET|chi2max|acceptable.set|χ² cap|chi2 cap|misfit cap')
for name, t in texts.items():
    for ln, line in enumerate(t.split('\n'), 1):
        for m in pat2.finditer(line):
            a, b = max(0, m.start() - 90), min(len(line), m.end() + 60)
            print(f'     {name}:{ln}: {m.group(0)}  …{line[a:b]}…')
print('   mentions of block weights (removed in October 2026), for review: every one should be history')
pat3 = re.compile(r'BLOCK_WEIGHTS|block_weights|block weight|reweight|pilot run')
for name, t in texts.items():
    for ln, line in enumerate(t.split('\n'), 1):
        for m in pat3.finditer(line):
            a, b = max(0, m.start() - 90), min(len(line), m.end() + 60)
            print(f'     {name}:{ln}: {m.group(0)}  …{line[a:b]}…')
print('   mentions of the earlier default (binbox, per-bin boxes), for information:')
for name, t in texts.items():
    k = len(re.findall(r'binbox|per-bin box', t))
    if k and not name.endswith('.py'):
        print(f'     {name}: {k}')

print('4. Defaults')


def argdefault(tree, opt):
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'add_argument' and n.args \
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == opt:
            for k in n.keywords:
                if k.arg == 'default':
                    return ast.literal_eval(k.value)
    return 'MISSING'


check(argdefault(trees['gibbs.py'], '--prior') == 'spikeslab', f"gibbs.py --prior default {argdefault(trees['gibbs.py'], '--prior')!r}")
_cut = [k for k in ('PRIOR_KIND', 'THETA', 'BOX_TOP', 'USE_SWAPS', 'OUTPUT_DRAWS', 'SA_STARTS', 'BEST_FIT')
        if any(re.search(rf'\b{k}\b', c) for _, c in cells)]
_gr = STEPS[STEPS.index('\ndef gibbs_args('):STEPS.index('\ndef _terms(')]
check(not _cut and all(s in _gr for s in ("'--init', init", "init = init or ('starts.npy' if g['DISPERSED_STARTS'] else 'nnls')",
                                          "'--no-samples'", "'--swap-cos', 0.99"))
      and not any("best_fit_as_fitted.py'" in c for _, c in cells)
      and "elif a.init == 'fitted':" in src['gibbs.py'] and 'nnls_key' in src['best_fit_as_fitted.py'],
      'notebook: no PRIOR_KIND, THETA, BOX_TOP, USE_SWAPS, OUTPUT_DRAWS, SA_STARTS or BEST_FIT; its runs start from the NNLS '
      'best fit (with DISPERSED_STARTS, chain 1; the others from randomized fits), store no draws and use swap moves; '
      'best_fit_as_fitted.py is for the command line since 2026-11-17 (gibbs.py --init fitted)'
      + (f'; found {_cut}' if _cut else ''))
check(re.findall(r'^SEED = (\S+)', cells[0][1], re.M) == ['None'] and 'replicate_seed(r)' in NBALL
      and "np.random.SeedSequence([s, r])" in STEPS and "seed=g['SEED_USED']" in STEPS,
      "notebook SEED = None (each run's seed from its start time; an integer repeats a run, 'legacy' gives 101, 102, ...), "
      'each replicate\'s seed hashed from it, recorded in the configuration')
check(bool(re.search(r"hyper_every = a\.hyper_every if a\.hyper_every is not None else \(1 if spike_mode else None\)", src['gibbs.py'])),
      'gibbs.py: --hyper-every defaults to 1 for spike-and-slab')
check(argdefault(trees['gibbs.py'], '--slab') in (None, 'MISSING') and "a.prelim, a.slab, a.s_range" in src['gibbs.py'],
      'gibbs.py: the slab always log-uniform (--slab, the square-root slab, removed 2026-11-23, with the hierarchical prior\'s, '
      'transfer moves\', --positive, --mfd-step-sd and the fixed internals\' options)')
_pl = src['run_plan.py']
check(re.findall(r'^RUN_MINUTES = (\S+)', cells[0][1], re.M) == ['240'] and re.findall(r'^TARGETS = (\S+)', cells[0][1], re.M) == ["'production'"]
      and not re.search(r'^PRESETS? = ', cells[0][1], re.M) and "run_minutes=g['RUN_MINUTES'], targets=g['TARGETS']" in STEPS
      and "TARGETS = {'exploratory': (100, 200, 1.01)," in _pl and "'production': (1000, 2000, 1.01)," in _pl
      and "'precise': (10000, 20000, 1.005)}" in _pl and 'PRESET_TARGETS' in src['check_convergence.py']
      and all(f'\ndef {f_}(' in _pl for f_ in ('plan', 'reference', 'run_record', 'calibration', 'post_estimate', 'needs'))
      and "'RUN_PLAN'" in src['run_config.py'] and 'SH_SECONDS[0] += time.time() - t0' in STEPS
      and "run_plan.plan(g['RUN_MINUTES'], g['THREADS'], cost, SH_SECONDS[0], post, g['TARGETS'], ref, diff)" in STEPS
      and 'for th in run_plan.CALIB_THINS:' in STEPS and 'PLAN, P = plan_runs()' in NBALL
      and "sh('gibbs.py', *gibbs_args(f'main_rep{r}', replicate_seed(r), P))" in NBALL
      and "json.dump(T, open('timings.json', 'w'), indent=1)" in STEPS and 'write_timings()' in cells[-1][1]
      and 'convergence_summary.json' in src['check_convergence.py']
      and not any("'--slab'" in c for _, c in cells),
      "notebook RUN_MINUTES = 240 and TARGETS = 'production' (2026-11-17, in place of the presets): the runs planned just before "
      'sampling (run_plan.py: the time spent, sh() measured, calibration runs at two recording rates, the latest run\'s post-processing, 10% margin; '
      '4 chains per thread, burn-in 5,000 or 20%, ~3,000 kept draws, replicates only if each is predicted at 1.5 x TARGETS; unrecorded settings of older runs at their old defaults), with a '
      'prediction from the latest run in DATA_DIR; tests fix RUN_PLAN (run_config); timings.json and convergence_summary.json '
      'for later plans; the other priors are gibbs.py options')
check(re.findall(r'^DISPERSED_STARTS = (\S+)', src['run_config.py'], re.M) == ['True']
      and not re.search(r'^DISPERSED_STARTS = ', cells[0][1], re.M) and "dispersed_starts=g['DISPERSED_STARTS']" in STEPS
      and "sh('dispersed_starts.py', *starts_args())" in NBALL and "'--prior-share'" not in STEPS and 'import randomized_nnls' not in src['dispersed_starts.py'] and 'from randomized_nnls' not in src['dispersed_starts.py']
      and "a += ['--fit-args', '--bins-block'," in STEPS
      and 'REPLICATE_MIN, REPLICATES_MAX = 50000, 1' in src['run_plan.py'] and "SeedSequence([a.seed, 7]).spawn(n_prior)" in src['dispersed_starts.py']
      and 'DISPERSED_STARTS=False' in open(f'{ROOT}/tools/run_quick_checks.sh').read(),
      'DISPERSED_STARTS = True, fixed in run_config.py since 2026-11-21: every chain from its own prior draw scaled to the '
      'bins\' targets (--prior-share 1, since 2026-11-20), polished briefly by best_fit_as_fitted.py in the run\'s model '
      'space, passed to gibbs.py --init starts.npy; one replicate (run_plan REPLICATES_MAX 1); the smoke test pins it off '
      '(every chain from the best fit, which cleaning then computes)')
check(not any("plot_targets.py'" in c for _, c in cells) and "'--figure'" in src['preliminaries.py']
      and not any("show('prelim.png')" in c for _, c in cells)
      and all(f_ in open(f'{ROOT}/tools/make_summary_pdf.py').read() for f_ in ('fig_targets_slip.png', 'fig_targets_paleo.png', 'fig_targets_mfd.png')),
      'the targets old and new (plot_targets.py) and the cleaning figure (preliminaries.py --figure) on request, out of the '
      'notebook since 2026-11-17; the summary PDF shows them when present')
check(all(t_ not in src['gibbs.py'] for t_ in ('selftest_linear', 'selftest_halfway', 'resolve_logbox', 'mixw', 'hi_vec', 'theta_mode'))
      and 'resolve_logbox' not in src['cru_common.py']
      and "'halfway', 'linear', 'logbox'" in src['gibbs.py'],
      "the single log-box, the per-bin boxes (binbox, binbox-floor, binbox-theta), the halfway and the linear priors removed "
      "(2026-11-23, the user's request; gibbs.py --prior refuses them by name)")
_cfg_keys = re.findall(r'(\w+)=', STEPS[STEPS.index("g['CONFIG'] = dict("):STEPS.index("g['CONFIG'].update(more)")])
check(re.findall(r"^DOWN_THE_MIDDLE = ('[\w-]+'|\w+)", src['run_config.py'], re.M) == ['False'] and 'down_the_middle' in _cfg_keys
      and not re.search(r'^DOWN_THE_MIDDLE = ', cells[0][1], re.M)
      and "a += ['--down-the-middle'] + (['zero-model'] if g['DOWN_THE_MIDDLE'] == 'zero-model' else [])" in STEPS
      and "'--down-the-middle', nargs='?', const='equal', default=None, choices=['equal', 'zero-model']" in src['gibbs.py']
      and 'balance_rows' in src['gibbs.py'] and "weights='equal'" in src['cru_common.py']
      and src['gibbs.py'].count('ok &= selftest_balance()') == 2 and '\ndef balance_rows(' in src['cru_common.py'],
      'DOWN_THE_MIDDLE = False (no balance term; fixed in run_config.py since 2026-11-17, run_nb.py overrides it), recorded; True and \'zero-model\' passed to gibbs.py '
      '--down-the-middle (cru_common.balance_rows; its exact test, of both weightings, in both self-test sets)')
check(not re.findall(r'^SLIP_BAND = ', cells[0][1], re.M) and 'slip_band' not in _cfg_keys
      and not re.findall(r'^PALEO_BAND', cells[0][1], re.M) and not re.findall(r'^BAND = ', cells[0][1], re.M)
      and "'--all-positive']" in STEPS and "'--slip-band'" not in NBALL
      and "ap.add_argument('--all-positive'" in src['gibbs.py'] and "pa_b = 'positive'" in src['gibbs.py']
      and all(f"add_argument('{o_}'" not in src['gibbs.py'] for o_ in ('--band', '--slip-band', '--paleo-band'))
      and '\ndef band_start(' not in src['gibbs.py'] and '    def set_groups(' in src['gibbs.py']
      and src['gibbs.py'].count('ok &= selftest_band()') == 2,
      'notebook: no SLIP_BAND since 2026-11-10; every slip-rate, paleo and MFD prediction kept above zero (gibbs.py '
      '--all-positive, Target.set_groups; the factor band, --band, --slip-band and --paleo-band, removed 2026-11-23), with the '
      'limits\' exact test (selftest_band) in both self-test sets and set_groups checked against set_positive and set_band')
check(re.findall(r'^MFD_MONOTONIC = (\S+)', cells[0][1], re.M) == ['False'] and 'mfd_monotonic' in _cfg_keys
      and any("a += ['--mfd-monotonic']" in c for c in [NBALL]) and not any('MFD_RISE_SD' in c for c in [NBALL])
      and "'--mfd-monotonic'" in src['gibbs.py'] and "a.transfers, a.positive, a.mfd_step_sd, a.mfd_rise_sd = None, False, None, 0.89" in src['gibbs.py']
      and "'monotonicity')" in src['gibbs.py'].split('\ndef selftest_band')[1].split('\ndef ')[0],
      'notebook MFD_MONOTONIC = False (off by default since 2026-11-09; True from November 10th, out of the notebook from November 4th), '
      'recorded and passed to gibbs.py --mfd-monotonic (a rising target\'s rise capped at --mfd-rise-sd 0.89 sd), with its '
      'exact test in the band\'s self-test')
check('\ndef check_bin_magnitudes(' in src['cru_common.py'] and 'check_bin_magnitudes(load_problem(args.out)' in src['preliminaries.py']
      and "raise SystemExit(f'magnitude bins: {e}')" in src['preliminaries.py'],
      'cleaning checks every bin row\'s ruptures against its edges by the rupture set\'s magnitudes (cru_common.check_bin_magnitudes) and stops if one lies outside')
check(all(o_ in src['best_fit_as_fitted.py'] for o_ in ("'--down-the-middle'", "'--slip-sum'", "'--plate-rate'", "'--mfd-monotonic'",
                                                    "'--snap-floor'", "'--con-tol'"))
      and '\ndef sampler_terms(' in src['best_fit_as_fitted.py'] and '\ndef polish_feasible(' in src['best_fit_as_fitted.py']
      and '\ndef mfd_monotonic(' in src['cru_common.py'] and 'mfd_monotonic(P, a.bins_block' in src['gibbs.py']
      and 'from best_fit_as_fitted import terms_key' in src['gibbs.py'],
      'best_fit_as_fitted.py takes the sampler\'s other terms and constraints as gibbs.py does (balance, summed slip rates, '
      'monotonic MFD through cru_common.mfd_monotonic, plate rate; augmented Lagrangian, polish, --snap-floor), and gibbs.py '
      '--init fitted notes a fit made with other terms')
_md_all = '\n'.join(''.join(c_['source']) for c_ in nb['cells'] if c_['cell_type'] == 'markdown')
check(re.findall(r'^MFD_GR = (\S+)', cells[0][1], re.M) == ['False']
      and re.findall(r'^MFD_GR_B = (\(.*?\))', cells[0][1], re.M) == ['(0.823, 0.959, 1.089)']
      and re.findall(r'^MFD_GR_N5 = (\(.*?\))', cells[0][1], re.M) == ['(2.65, 3.40, 4.57)']
      and re.findall(r'^MFD_GR_NSCALE = (\(.*?\))', cells[0][1], re.M) == ['(0.66, 1.41)'] and 'mfd_gr' in _cfg_keys
      and re.findall(r'^RECONSIDER_MAGUNCERT = (\S+)', cells[0][1], re.M) == ['True']
      and 'MAGUNCERT = (1.0, _MAGRATIO, 1.0 + 2.0 * (_MAGRATIO - 1.0))' in src['mfd_gr.py'] and '_MAGRATIO = 3.92 / 3.40' in src['mfd_gr.py']
      and '+ tuple(mag))' in STEPS and "S['RECONSIDER_MAGUNCERT']" in src['run_config.py']
      and "rm = g['RECONSIDER_MAGUNCERT']" in STEPS and 'RECONSIDER_MAGUNCERT' in open(f'{ROOT}/docs/REPORT.md').read()
      and '\ndef parse_knobs(' in src['mfd_gr.py'] and '\ndef n_of_b(' in src['mfd_gr.py'] and "q84=q84" in src['mfd_gr.py']
      and 'mfd_pdf_q84=' in src['preliminaries.py'] and "P.get('mfd_pdf_q84')" in src['gibbs.py']
      and 'MFD_GR_SPREAD' not in cells[0][1] and "out['MFD_MISFIT'], out['MISFIT'] = 'pdf', 'mixed'" in src['run_config.py']
      and any("a += ['--mfd-gr', g['MFD_GR_SPEC'], '--rupture-set', g['RUPTURE_DIR']]" in c for c in [NBALL])
      and "'--mfd-gr'" in src['preliminaries.py'] and 'mfd_gr.gr_bins(' in src['preliminaries.py']
      and '\ndef bin_pdfs(' in src['mfd_gr.py'] and '\ndef fit_bin_curves(' in src['mfd_gr.py'] and '\ndef gr_bins(' in src['mfd_gr.py']
      and "'pdf']" in src['gibbs.py'] and '\ndef selftest_mfdpdf(' in src['gibbs.py'] and 'ok &= selftest_mfdpdf()' in src['gibbs.py']
      and "'pdf']" in src['best_fit_as_fitted.py'] and "'mfd_pdf_th' in z.files" in src['block_fit.py']
      and '\ndef curve_tables(' in src['paleo_misfit.py'] and 'mfd-bound' not in src['gibbs.py'],
      'notebook MFD_GR = False (2026-11-17: each bin scored by its own target PDF, propagated through NZSHM22\'s partitioning, '
      'mfd_gr.py; cru_D\'s bins by default) from NZSHM22\'s upper-plate knobs: MFD_GR_B (0.823, 0.959, 1.089) b\'s 10th '
      'percentile, mean and 90th, MFD_GR_N5 (2.65, 3.40, 4.57) the mean N conditioned on each (correlated), MFD_GR_NSCALE '
      '(0.66, 1.41) the N scaling\'s 2-sigma values, RECONSIDER_MAGUNCERT = True a further factor on N, 1 / 3.92/3.40 / '
      '1 + 2(3.92/3.40 - 1), three branches of equal weight (mfd_gr.MAGUNCERT); the prior\'s soft top from each bin\'s 84% point (mfd_pdf_q84); recorded, passed to preliminaries.py --mfd-gr, the bins then --mfd-misfit pdf in gibbs.py and '
      'best_fit_as_fitted.py (the paleo curves\' tables, paleo_misfit.curve_tables), with its exact test (selftest_mfdpdf) '
      'in the full set; block_fit.py scores the bins by their curves')
check(re.findall(r'^PLATE_RATE = (\S+)', cells[0][1], re.M) == ['True'] and 'plate_rate' in _cfg_keys
      and "g['PLATE_LIMITS'] = f\"{g['SLIP_RATE_FACTOR']:g}\" if g['PLATE_RATE'] else None" in STEPS
      and any("a += ['--plate-rate', g['PLATE_LIMITS']]" in c for c in [NBALL])
      and "'--plate-rate'" in src['gibbs.py']
      and 'from plate_rate import describe_limits, describe_plate, parse_limits, plate_rows' in src['gibbs.py']
      and "'the plate rate')" in src['gibbs.py'].split('\ndef selftest_band')[1].split('\ndef ')[0]
      and any("sh('plate_rate.py', _rd, '--run', results_dir()" in c and "'--limits', PLATE_LIMITS" in c for c in [NBALL])
      and "'--limits'" in src['plate_rate.py']
      and '\ndef check_problem(' in src['plate_rate.py'] and '\ndef plate_rows(' in src['plate_rate.py'],
      'notebook PLATE_RATE = True (back on November 10th, 2026, as a hard upper bound at SLIP_RATE_FACTOR on each strip\'s '
      'share of the plate rate), recorded and passed to gibbs.py --plate-rate with --rupture-set (rows from plate_rate.py on '
      'the --cross-sections strips) and to plate_rate.py --limits, with its exact test in the band\'s self-test')
check(re.findall(r'^CROSS_SECTIONS = (\S+)', src['run_config.py'], re.M) == ["'all'"] and 'cross_sections' in _cfg_keys
      and any("if g['SLIP_SUM'] or g['PLATE_RATE']:" in c and "a += ['--cross-sections', g['CROSS_SECTIONS']]" in c for c in [NBALL])
      and "'--cross-sections', default='all', choices=['all', 'between']" in src['gibbs.py']
      and "'--cross-sections', default='all', choices=['all', 'between']" in src['plate_rate.py']
      and '\ndef cross_sections(' in src['plate_rate.py'],
      'notebook CROSS_SECTIONS = \'all\' (gibbs.py\'s and plate_rate.py\'s default too), recorded and passed with '
      'SLIP_SUM: its strips (plate_rate.py\'s cross_sections; gibbs.py --plate-rate uses them too)')
check(re.findall(r'^SLIP_SUM = (\S+)', cells[0][1], re.M) == ['True'] and 'slip_sum' in _cfg_keys
      and "'--slip-sum', action='store_true'" in src['gibbs.py']
      and 'from plate_rate import describe_slip_sum, slip_sum_rows' in src['gibbs.py']
      and "summed slip rates\\' term recorded apart" in src['gibbs.py'].split('\ndef selftest_balance')[1].split('\ndef ')[0]
      and '\ndef slip_sum_rows(' in src['plate_rate.py'] and 'SLIP_SUM_L0 = 8.5 ' in src['plate_rate.py']
      and "Wc = W * cs['proj'][None, :]" in src['plate_rate.py']
      and argdefault(trees['gibbs.py'], '--slip-sum-l0') == 8.5
      and re.findall(r'^SLIP_SUM_L0 = (\S+)', cells[0][1], re.M) == ['100'] and 'slip_sum_l0' in _cfg_keys
      and any("a += ['--slip-sum', '--slip-sum-l0', g['SLIP_SUM_L0']]" in c for c in [NBALL])
      and any("'--cross-sections', CROSS_SECTIONS, '--slip-sum-l0', SLIP_SUM_L0" in c for c in [NBALL]),
      'notebook SLIP_SUM = True (on by default since 2026-11-09; off 2026-11-05 to 11-09) and SLIP_SUM_L0 = 100 (km; 8.5, the '
      'strips\' width, gibbs.py\'s and plate_rate.py\'s default), recorded and passed to gibbs.py --slip-sum with the rupture set (rows from plate_rate.py: each '
      'subsection once, its slip projected on the plate motion), recorded apart from '
      'chi2, with its bookkeeping in the balance term\'s self-test; its term in plate_rate.txt for every run')
check(not re.findall(r'^SLIP_BAND_SOFT = ', cells[0][1], re.M) and 'slip_band_soft' not in _cfg_keys
      and not any("'--slip-log-band'" in c for c in [NBALL])
      and all(f"add_argument('{o_}'" not in src['gibbs.py'] for o_ in ('--log-band', '--slip-log-band', '--paleo-log-band'))
      and 'selftest_logband' not in src['gibbs.py'] and 'static double lw_change(' in src['gibbs.py']
      and 'static inline void col_dots_lw(' in src['gibbs.py'],
      'notebook: no SLIP_BAND_SOFT since 2026-11-10; the soft band\'s options removed from gibbs.py on 2026-11-23 (the user: the '
      'log and years misfits make it needless); its kernel machinery stays, the log misfits\' (lw_change, col_dots_lw)')
check([re.findall(rf'^MISFIT_{g_} = (\S+)', cells[0][1], re.M) for g_ in ('SLIP', 'MFD', 'PALEO')] == [["'linear'"], ["'half'"], ["'years'"]]
      and not re.findall(r"^MISFIT = '", cells[0][1], re.M)
      and {'misfit', 'misfit_slip', 'misfit_mfd', 'misfit_paleo', 'paleo_misfit', 'mfd_misfit', 'slip_misfit'} <= set(_cfg_keys)
      and "'paleo': {'rate': 'gaussian', 'years': 'years', 'half': 'gaussian+years'}}" in src['run_config.py']
      and "'paleo': {'rate': 'gaussian', 'years': 'years', 'half': 'gaussian+years'}}" in src['cru_common.py']
      and "'slip': {'linear': 'split', 'log10': 'log10', 'half': 'split+log10'}" in src['run_config.py']
      and "'slip': {'linear': 'split', 'log10': 'log10', 'half': 'split+log10'}" in src['cru_common.py']
      and "('half',) * 3: 'half'}.get((g['MISFIT_SLIP'], g['MISFIT_MFD'], g['MISFIT_PALEO']), 'mixed')" in src['run_config.py']
      and '\ndef misfit_settings(' in src['cru_common.py']
      and all(f"a += ['--{g_}-misfit', g['{g_.upper()}_MISFIT']]" in ''.join(c for c in [NBALL]) for g_ in ('paleo', 'mfd', 'slip'))
      and "choices=['gaussian', 'years', 'gaussian+years']" in src['gibbs.py']
      and "choices=['gaussian', 'log10', 'gaussian+log10', 'pdf']" in src['gibbs.py'] and "choices=['split', 'split+log10', 'log10']" in src['gibbs.py']
      and "'half': ('split+log10', 'gaussian+log10', 'gaussian+years')" in src['cru_common.py']
      and all(src['gibbs.py'].count(t_) == 1 for t_ in ('ok &= selftest_paleo()', 'ok &= selftest_paleo(b=0.5)', 'ok &= selftest_mfdlog()',
                                                         'ok &= selftest_mfdlog(b=0.5)', 'ok &= selftest_slipgeo()', 'ok &= selftest_slipgeo(b=1.0)'))
      and 'static double pm_f(' in src['gibbs.py'] and '\ndef fit_site(' in src['paleo_misfit.py'] and '\ndef tables(' in src['paleo_misfit.py']
      and '\ndef site_moments_mixed(' in src['paleo_misfit.py'] and "'mfd_log_s' in z.files" in src['block_fit.py'],
      "notebook MISFIT_SLIP = 'linear', MISFIT_MFD = 'half', MISFIT_PALEO = 'years' (one per group since 2026-11-09, one MISFIT "
      "before, 'half' by default from 2026-11-10; the defaults the user's choice, the space each group's uncertainties are worked "
      "out in; slip and MFD 'linear', 'log10', 'half', paleo 'rate', 'years', 'half'), recorded with their common setting or 'mixed' and each group's mode (cru_common.MISFIT_MODES, misfit_settings) and passed to gibbs.py --slip-misfit, "
      '--mfd-misfit and --paleo-misfit, with exact tests of each group\'s log and half modes in the full self-test set and '
      'block_fit.py scoring each group as the run did')
check(re.findall(r'^CFM_SLIPRATES = (\S+)', src['run_config.py'], re.M) == ['True'] and 'cfm_sliprates' in _cfg_keys
      and not re.findall(r'^CFM_SLIPRATES = ', cells[0][1], re.M)
      and not re.findall(r'^CFM_FILE = ', cells[0][1], re.M) and "cfm_file=g['CFM_CSV']" in ''.join(c for c in [NBALL])
      and "\nCFM_CSV = os.path.join(" in src['cru_common.py'] and os.path.exists(f'{ROOT}/data/NZ_CFM_v1_0.csv')
      and any("a += ['--cfm-sliprates', g['CFM_CSV']]" in c and 'SLIP_AS_GIVEN' not in c
          and "'--slip-rate-factor', str(g['SLIP_RATE_FACTOR']), '--rupture-set', g['RUPTURE_DIR']]" in c
          and 'slip-sd-multiplier' not in c and 'SD_MULTIPLIER_' not in c for c in [NBALL])
      and "'--cfm-sliprates', default=None" in src['preliminaries.py'] and '\ndef rebuild(' in src['cfm_sliprates.py']
      and re.findall(r'^SLIP_RATE_FACTOR = (\S+)', cells[0][1], re.M) == ['0.8'] and 'slip_rate_factor' in _cfg_keys
      and "'--slip-rate-factor', type=float, default=None" in src['preliminaries.py'] and '\ndef rescale_inputs(' not in src['cfm_sliprates.py']
      and not any(re.findall(rf'^{s_} = ', cells[0][1], re.M) for s_ in ('SLIP_SD_MULTIPLIER', 'CFM_PREF', 'CFM_ROUNDING'))
      and 'SD_MULTIPLIER_' not in src['run_config.py'] and 'FACTOR_DEFAULT' not in src['cfm_sliprates.py']
      and {'slip_sd_multiplier', 'cfm_pref', 'cfm_rounding'} <= set(_cfg_keys)
      and "'--slip-sd-multiplier'" not in src['preliminaries.py']
      and "args.cfm_sigmas, args.cfm_pref, args.cfm_rounding, args.cfm_timeframe_shape = 2.0, 'mode', False, 'fit'" in src['preliminaries.py']
      and re.findall(r'^CFM_TIMEFRAME = (\S+)', cells[0][1], re.M) == ['True'] and 'cfm_timeframe' in _cfg_keys
      and "'--cfm-timeframe', default=None" in src['preliminaries.py'] and '\ndef load_timeframe(' in src['cfm_sliprates.py']
      and 'timeframe_sliprates.py' in src and os.path.exists(f'{ROOT}/data/cfm_timeframe_sliprates.csv')
      and '\ndef fit_split_normal(' in src['timeframe_sliprates.py']
      and 'sn_centre' in open(f'{ROOT}/data/cfm_timeframe_sliprates.csv').readline() and 'cfm_timeframe_shape' in _cfg_keys
      and "'--cfm-rounding'" not in src['preliminaries.py'] and "'--cfm-pref'" not in src['preliminaries.py']
      and re.findall(r'^CFM_TEMPLATE_RHO = (\S+)', cells[0][1], re.M) == ['1'] and 'cfm_template_rho' in _cfg_keys
      and "'--cfm-template-rho', type=float, default=None" in src['preliminaries.py'] and '\ndef template_factors(' in src['cfm_sliprates.py']
      and "CFM_RATE_LINKS=g['CFM_TEMPLATE_RHO'] is not None" in src['run_config.py'] and 'cfm_rate_links' in _cfg_keys
      and "'--cfm-rate-links', action='store_true'" in src['preliminaries.py'] and '\ndef rate_links(' in src['cfm_sliprates.py']
      and src['gibbs.py'].count('ok &= selftest_split()') == 1 and 'static double sp_lik(' in src['gibbs.py']
      and "P.get('split_lo2') is not None" in src['gibbs.py'] and 'split_weights(' in src['block_fit.py'],
      'notebook CFM_SLIPRATES = True (by default since 2026-11-05, not a setting since 2026-11-09, always True and not '
      'overridable since 2026-11-23, the inputs\' own slip-rate Gaussians no longer rescaled or used; each slip rate a split normal from the CFM, data/NZ_CFM_v1_0.csv '
      '(cru_common.CFM_CSV, no CFM_FILE); False: the inputs\', cfm_sliprates.py), CFM_TIMEFRAME\'s split normals fitted to each corrected distribution\'s 16/50/84% points (--cfm-timeframe-shape fit, since 2026-11-09), SLIP_RATE_FACTOR = 0.8; no CFM_PREF, CFM_ROUNDING or SLIP_SD_MULTIPLIER since 2026-11-09 (SR_pref the '
      'mode, no rounding, the sds as given: multiplier 1 with the CFM and without it (2.5 for the inputs until 2026-11-23: no factor 2 since); recorded, preliminaries.py keeps '
      'the options), recorded and passed to preliminaries.py --cfm-sliprates and --slip-rate-factor (--slip-sd-multiplier removed 2026-11-23); gibbs.py evaluates the split misfit '
      'exactly (its test in the full self-test set) and block_fit.py weighs it')
check('\ndef paleo_data(' in src['paleo_misfit.py'] and '\ndef map_sites(' in src['paleo_misfit.py']
      and 'SITE_MAX_KM = 5.25' in src['paleo_misfit.py'] and os.path.exists(f'{ROOT}/data/paleo_constraint_data.json'),
      'paleo sites: the rupture set\'s paleo_constraint_data.json, or, for a rupture set made without an inversion, '
      'data/paleo_constraint_data.json\'s sites mapped onto its subsections as NZSHM22 maps them (nearest trace within '
      '5.25 km; paleo_misfit.paleo_data; reproduces the 19k, 211k and 630k solutions\' files exactly)')
check(sum(c.count('*ref_args()') for c in [NBALL]) == 4 and '\ndef reference_rates(' in src['cru_common.py']
      and '\ndef reference_solutions(' in STEPS and 'MAX_REFERENCES = 2' in STEPS and '\ndef reference_list(' in src['cru_common.py']
      and all("'--reference'" in src[sc] for sc in ('plot_slip_rates.py', 'plot_paleo_rates.py', 'plot_rate_magnitude.py')),
      'the fit figures take --reference (cru_common.reference_rates, up to twice: reference_list), and the notebook passes '
      'up to two OpenSHA solutions found in DATA_DIR to all of them (notebook_steps.reference_solutions, 2026-11-23; the '
      'posterior\'s; no best fit since 2026-11-21)')
check(all('axis_span(' in src[sc] for sc in ('plot_slip_rates.py', 'plot_paleo_rates.py'))
      and 'best_fit_rates(' in src['plot_rate_magnitude.py'] and '\ndef axis_span(' in src['cru_common.py'],
      'the fit figures take fixed axes from the targets and the best fit (cru_common.axis_span, best_fit_rates)')
check(all("'--config'" in src[sc] and 'figure_settings(' in src[sc] for sc in ('plot_slip_rates.py', 'plot_paleo_rates.py',
                                                                              'plot_rate_magnitude.py'))
      and '\ndef run_settings(' in src['cru_common.py'] and '\ndef figure_settings(' in src['cru_common.py']
      and "('FLOOR', 'floor')" in src['cru_common.py'],
      'the fit figures show the model and every one of the run\'s settings, SLIP_SECTIONS to FLOOR, whatever its value, in a '
      'table of fixed size (cru_common.figure_settings, run_settings)')
check(not any('RANDOMIZED_NNLS' in c or "'randomized_nnls.py'" in c for c in [NBALL])
      and not os.path.exists(f'{ROOT}/tools/randomized_nnls.py'),
      'no RANDOMIZED_NNLS (retired from the notebook on 2026-11-03); randomized_nnls.py removed on 2026-11-23 (the user\'s '
      'request), with the randomized-fit starts')
for s in ('gibbs.py', 'preliminaries.py', 'summarize_runs.py', 'plot_rate_magnitude.py', 'check_convergence.py'):
    d = argdefault(trees[s], '--diag-floor')
    check(d == -10, f'{s} --diag-floor default {d!r}')
m = re.findall(r"spike_mode = True.*\n.*\n\s*W, se, top = float\(sp_\[0\] or (-?\d+)\)", src['gibbs.py'])
check(m == ['-10'], f'gibbs.py spike-and-slab default slab floor {m}')
cfg = cells[0][1]
check(re.findall(r'^FLOOR = (-?\d+)', cfg, re.M) == ['-10'] and not re.search(r'^(SLAB|DIAG)_FLOOR', cfg, re.M),
      f"notebook FLOOR = {re.findall(r'^FLOOR = (-?[0-9]+)', cfg, re.M)}, no SLAB_FLOOR or DIAG_FLOOR setting")
_ARGS = {'preliminaries.py': STEPS[STEPS.index('\ndef cleaning_args('):STEPS.index('\ndef after_cleaning(')],   # the scripts'
         'gibbs.py': STEPS[STEPS.index('\ndef gibbs_args('):STEPS.index('\ndef _terms(')]}                      # arguments
for script in ('preliminaries.py', 'gibbs.py', 'plot_rate_magnitude.py', 'check_convergence.py'):
    cc = [i for i, c in cells if 'sh(' in c and f"'{script}'" in c]
    ok = bool(cc) and all('--diag-floor={FLOOR}' in c or (script in _ARGS and "f\"--diag-floor={g['FLOOR']}\"" in _ARGS[script])
                          for i, c in cells if i in cc)
    check(ok, f"notebook passes --diag-floor={{FLOOR}} to {script} (cells {cc})")
check("f\"spikeslab:{g['FLOOR']}:soft\"" in _ARGS['gibbs.py'], 'notebook uses spikeslab:{FLOOR}:soft')
check(re.search(r'clip = 10\.0 \*\* floor\b', src['check_convergence.py']) is not None and not any('clip = 10.0 **' in c for _, c in cells),
      'check_convergence.py clips the convergence diagnostics at 10 ** --diag-floor (the notebook passes FLOOR); the notebook computes none itself')
cc = [i for i, c in cells if 'sh(' in c and "'check_convergence.py'" in c]
check(len(cc) == 1 and all("'--targets', TARGETS, '--plan', 'run_plan.json'" in c for i, c in cells if i in cc) and any("'make_tables.py'" in c for _, c in cells),
      f'notebook runs check_convergence.py once with its TARGETS and the run plan (cells {cc}), then make_tables.py')
keys = None
for n in ast.walk(ast.parse(STEPS)):                    # the configuration notebook_steps.record_config writes
    if isinstance(n, ast.Assign) and any(ast.unparse(t) == "g['CONFIG']" for t in n.targets) and isinstance(n.value, ast.Call):
        keys = [k.arg for k in n.value.keywords]
check(keys is not None and 'floor' in keys and not {'slab_floor', 'diag_floor'} & set(keys),
      f'recorded configuration keys: {keys}')
check(argdefault(trees['preliminaries.py'], '--slip-sections') == 'auto',
      f"preliminaries.py --slip-sections default {argdefault(trees['preliminaries.py'], '--slip-sections')!r}")
check(re.findall(r"^SLIP_SECTIONS = ('[a-z]+')", src['run_config.py'], re.M) == ["'auto'"],
      f"run_config.py SLIP_SECTIONS = {re.findall(r'^SLIP_SECTIONS = (.*?) ', src['run_config.py'], re.M)}")
cc = [i for i, c in cells if 'sh(' in c and "'preliminaries.py'" in c and '--selftest' not in c]
check(bool(cc) and "'--slip-sections'" in _ARGS['preliminaries.py'], f'notebook passes --slip-sections to preliminaries.py (cells {cc})')
check(keys is not None and 'slip_sections' in keys, 'recorded configuration includes slip_sections')
check(argdefault(trees['preliminaries.py'], '--mfd-uncertainty') == 'as-given',
      f"preliminaries.py --mfd-uncertainty default {argdefault(trees['preliminaries.py'], '--mfd-uncertainty')!r}")
lb = {sc: (argdefault(trees[sc], '--lowest-bin'), argdefault(trees[sc], '--bin-width')) for sc in ('preliminaries.py', 'plot_rate_magnitude.py')}
check(lb['preliminaries.py'] == lb['plot_rate_magnitude.py'] == (6.8, 0.1),
      f'--lowest-bin and --bin-width defaults agree between preliminaries.py and plot_rate_magnitude.py: {lb}')
check(re.findall(r"^MFD_UNCERTAINTY = ('[a-z-]+')", src['run_config.py'], re.M) == ["'as-given'"]
      and not re.search(r'^MFD_UNCERTAINTY = ', cfg, re.M) and 'MFD_UNCERTAINTY' in re.search(r'^FIXED = \((.*?)\)', src['run_config.py'], re.M | re.S).group(1),
      'MFD_UNCERTAINTY fixed at \'as-given\' in run_config.py (out of the notebook since 2026-11-18)')
cc = [i for i, c in cells if 'sh(' in c and "'preliminaries.py'" in c and '--selftest' not in c]
check(bool(cc) and "'--mfd-uncertainty', g['MFD_UNCERTAINTY']" in _ARGS['preliminaries.py'],
      f'notebook passes --mfd-uncertainty MFD_UNCERTAINTY to preliminaries.py (cells {cc})')
check(keys is not None and 'mfd_uncertainty' in keys and not {'block_weights', 'pilot_lengths', 'pilot_tol'} & set(keys),
      'recorded configuration includes mfd_uncertainty and no block-weights settings')
check(not re.search(r'^(BLOCK_WEIGHTS|PILOT_LENGTHS|PILOT_TOL) =', cfg, re.M) and not any("'block_weights.py'" in c for _, c in cells),
      'notebook has no BLOCK_WEIGHTS, PILOT_LENGTHS or PILOT_TOL setting and runs no block_weights.py')
cc = [i for i, c in cells if 'sh(' in c and "'block_fit.py'" in c]
check(len(cc) == 1, f'notebook runs block_fit.py once, for the fit report (cells {cc})')
cdefs = {n.name for n in ast.walk(trees['cru_common.py']) if isinstance(n, (ast.FunctionDef, ast.Assign)) and hasattr(n, 'name')}
cnames = {t.id for n in ast.walk(trees['cru_common.py']) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
check({'magnitude_bins', 'mfd_uncertainty', 'mfd_implied', 'scale_rows'} <= cdefs and 'MFD_UNCERTAINTY_OPTIONS' in cnames
      and not ({'apply_block_weights', 'problem_fingerprint'} & cdefs) and 'BLOCK_WEIGHT_OPTIONS' not in cnames,
      'cru_common.py defines the MFD helpers and none of the block-weight helpers')
m = re.search(r"^MFD_UNCERTAINTY_OPTIONS = (\(.*?\))", src['cru_common.py'], re.M)
check(m is not None and ast.literal_eval(m.group(1)) == ('as-given', 'two-measurements'), f'MFD_UNCERTAINTY_OPTIONS = {m.group(1) if m else None}')
md_def = [n for n in ast.walk(trees['cru_common.py']) if isinstance(n, ast.FunctionDef) and n.name == 'mfd_uncertainty']
spec_default = ast.literal_eval(md_def[0].args.defaults[0]) if md_def and md_def[0].args.defaults else None
check(spec_default == 'as-given', f'cru_common.mfd_uncertainty spec default {spec_default!r}')
check(not any(re.search(r'block_weights|block_scale|BLOCK_WEIGHT', src[sc]) for sc in SCRIPTS),
      'no script mentions block_weights, block_scale or BLOCK_WEIGHT')
check('plot_rate_magnitude.py' in SCRIPTS and not re.search(r'^def magnitude_bins', src['plot_rate_magnitude.py'], re.M)
      and re.search(r'from cru_common import \(?[^)]*?\bmagnitude_bins\b', src['plot_rate_magnitude.py']) is not None,
      'plot_rate_magnitude.py takes magnitude_bins from cru_common (one definition, shared with cleaning)')
check(argdefault(trees['gibbs.py'], '--target') == 'MISSING' and argdefault(trees['gibbs.py'], '--chi2max') == 'MISSING'
      and 'chi2max' not in src['gibbs.py'] and 'eps2' not in src['gibbs.py'],
      'gibbs.py has no --target or --chi2max option and no acceptable-set mode (the Gaussian likelihood is the only one)')
check(not any(re.search(r'\bTARGET\b', c) or "'--target'" in c for _, c in cells) and keys is not None and 'target' not in keys,
      'notebook has no TARGET setting (the acceptable-set mode; TARGETS, the verdict\'s targets, is another thing), passes no '
      '--target, and records no target')
pre_opts = {a.value for n in ast.walk(trees['preliminaries.py']) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'add_argument'
            for a in n.args if isinstance(a, ast.Constant)}
check('--keep-soft-penalties' not in pre_opts and '--no-nnls' in pre_opts and '--from-rupture-set' in pre_opts
      and "a += ['--no-nnls']" in STEPS and "src = ['--from-rupture-set', g['RUPTURE_DIR']]" in STEPS
      and '\ndef build_inputs(' in src['rupture_set_inputs.py']
      and '\ndef slip_rates(' in src['rupture_set_inputs.py'] and '\ndef supra_target(' in src['rupture_set_inputs.py']
      and '\ndef rupture_set_parts(' in src['mfd_gr.py'] and 'NZSHM22_RUN = dict(n5=3.4,' in src['mfd_gr.py']
      and '\ndef section_shares(' in src['fault_polygons.py'] and os.path.exists(os.path.join(ROOT, 'data', 'Floor_AddoptiEEPAScomb-CRU_pdf.csv')),
      'preliminaries.py has --no-nnls (--keep-soft-penalties removed 2026-11-23) (the notebook passes it with DISPERSED_STARTS, since 2026-11-21) '
      'and --from-rupture-set (rupture_set_inputs.build_inputs; the notebook with MFD_GR and the CFM, cru_A/cru_D not read; '
      'from a bare rupture set too since 2026-11-23: slip_rates, supra_target, mfd_gr.rupture_set_parts with NZSHM22_RUN N5 3.4, '
      'fault_polygons.section_shares and data/Floor_AddoptiEEPAScomb-CRU_pdf.csv)')
dense_g = re.findall(r'\bG[a-z]?\.toarray\(\)', src['preliminaries.py'])        # a whole G made dense: 4 GB at 2e5, 11 at 5e5
check(not dense_g and 'qr_insert' in src['preliminaries.py'], 'preliminaries.py makes no dense copy of G (the best fit updates a QR '
      'factorization)' + (f'; found {", ".join(dense_g)}' if dense_g else ''))
check(not any("'--selftest'" in c or 'RUN_SELFTESTS' in c for _, c in cells)
      and re.search(r"'--selftest', nargs='\?', const='all', choices=\['all', 'current'\]", src['gibbs.py']) is not None,
      "the self-tests run from the command line since 2026-11-17 (gibbs.py --selftest takes 'all' or 'current'); the "
      'notebook runs none')
cc = [i for i, c in cells if 'nnls' in c]
check(not cc and "'nnls_vectors.npz'" in src['preliminaries.py'],
      f'no best fit in the notebook since 2026-11-21 (cells naming nnls: {cc}); preliminaries.py still writes nnls_vectors.npz '
      'without --no-nnls (tests)')
check("'nnls_vectors.csv'" in src['preliminaries.py'], 'preliminaries.py also writes the best fit as a table, nnls_vectors.csv')
cc = [i for i, c in cells if "'make_solution.py'" in c and 'sh(' in c]   # the solutions cell, the last
check(re.findall(r"^RUPTURE_SET = ('[^']*')", src['run_config.py'], re.M) == ["'auto'"] and "find_rupture_set(g['DATA_DIR'])" in STEPS
      and '\ndef rupture_dir():' in STEPS and len(cc) == 1 and cc[0] == cells[-1][0]
      and all("'nnls/nnls_vectors.npz'" not in c and "'--column', 'mean'" in c and "'--ruptures', _rd" in c and 'rupture_dir()' in c
              for i, c in cells if i in cc)
      and "os.path.join(os.path.dirname(stem), 'nnls')" in src['preliminaries.py'] and {'rupture_set', 'rupture_dir'} <= set(keys or []),
      f"the last cell ({cc}) writes the posterior mean as an OpenSHA solution (make_solution.py) for the rupture set found "
      "then (RUPTURE_SET = 'auto'), recorded; a best fit, when cleaning makes one (tests), goes into RUN_DIR/nnls/")

# every model setting recorded, in the figures' settings table, in the run name, in the summary PDF and documented
MODEL_SETTINGS = [
                  'MISFIT_SLIP', 'MISFIT_MFD', 'MISFIT_PALEO', 'MFD_MONOTONIC', 'MFD_GR', 'MFD_GR_TAU', 'DATATYPES_EQUAL', 'SLIP_ALONG_RUPTURE', 'CFM_TIMEFRAME', 'CFM_TEMPLATE_RHO',
                  'SLIP_RATE_FACTOR', 'SLIP_SUM', 'SLIP_SUM_L0', 'PLATE_RATE', 'RUN_MINUTES', 'FLOOR']
_rc = src['run_config.py']
check('setup(globals())' in cells[0][1] and 'g.update(derived_settings(g))' in STEPS
      and cells[0][1].index('setup(globals())') > cells[0][1].index('SEED = ')
      and all(re.search(rf'^{s_} = ', _rc, re.M) and not re.search(rf'^{s_} = ', cells[0][1], re.M)
              for s_ in ('SLIP_SECTIONS', 'RUPTURE_SET', 'CROSS_SECTIONS', 'CFM_SLIPRATES', 'DISPERSED_STARTS'))
      and not re.search(r'^BLOCKS = ', _rc, re.M)
      and "FIXED = ('SLIP_SECTIONS', 'RUPTURE_SET', 'CROSS_SECTIONS', 'CFM_RATE_LINKS', 'DOWN_THE_MIDDLE',\n         'SLIP_BLOCK', 'BIN_BLOCK', 'MFD_UNCERTAINTY', 'DISPERSED_STARTS')" in _rc
      and "if 'CFM_SLIPRATES' in ov and ov['CFM_SLIPRATES'] is not True:" in _rc
      and "('SLIP_SECTIONS' + str(S['SLIP_SECTIONS']) + '_' if S['SLIP_SECTIONS'] != 'auto' else '')" in _rc
      and 'from run_config import OVERRIDABLE' in open(f'{ROOT}/tools/run_nb.py').read(),
      'notebook: SLIP_SECTIONS, RUPTURE_SET, CROSS_SECTIONS, CFM_RATE_LINKS (with CFM_TEMPLATE_RHO) and CFM_SLIPRATES fixed '
      'in tools/run_config.py since 2026-11-17 (2026-11-09 below the settings; run_nb.py overrides them through _OVERRIDES, for '
      'sensitivity tests, all but CFM_SLIPRATES, always True since 2026-11-23); the notebook calls derived_settings; runs named SLIP_SECTIONS... only off auto')
_cfgmap = dict((v, k) for k, v in re.findall(r"(\w+)=g\['([A-Z][A-Z0-9_]+)'\]",
                                              STEPS[STEPS.index("g['CONFIG'] = dict("):STEPS.index("g['CONFIG'].update(more)")]))
_runname = src['run_config.py'][src['run_config.py'].index('\ndef run_name('):]
_figset = re.findall(r"\('([A-Z_0-9]+)', '\w+'\)", src['cru_common.py'])
_md = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'markdown')
_docs = {d_: open(f'{ROOT}/{d_}').read() for d_ in ('docs/REPORT.md', 'docs/CLAUDE.md')}
_pdf = open(f'{ROOT}/tools/make_summary_pdf.py').read()
_gaps = []
for s_ in MODEL_SETTINGS:
    k_ = _cfgmap.get(s_)
    miss = [w for w, ok_ in (('defined', re.search(rf'^{s_} = ', cfg, re.M) is not None), ('recorded', k_ is not None),
                             ('in FIGURE_SETTINGS', s_ in _figset), ('in RUN_NAME', s_ in _runname),
                             ('in the summary PDF', k_ is not None and (f"'{k_}'" in _pdf or f'"{k_}"' in _pdf)),
                             ('explained in the configuration cell', re.search(rf'^{s_} = .*#', cfg, re.M) is not None),
                             ('in REPORT.md', s_ in _docs['docs/REPORT.md']),
                             ('in CLAUDE.md', s_ in _docs['docs/CLAUDE.md'])) if not ok_]
    if miss:
        _gaps.append(f'{s_}: not ' + ', '.join(miss))
check(not _gaps, f'the {len(MODEL_SETTINGS)} model settings are each defined, recorded, in FIGURE_SETTINGS, in RUN_NAME, in the '
      'summary PDF and documented (the configuration cell\'s comment, REPORT.md, CLAUDE.md)' + (': ' + '; '.join(_gaps) if _gaps else ''))
_pin = re.search(r'PIN=\((.*?)\)\s+#', open(f'{ROOT}/tools/run_quick_checks.sh').read(), re.S).group(1)
_unpinned = [s_ for s_ in ('MISFIT_SLIP', 'MISFIT_MFD', 'MISFIT_PALEO', 'MFD_MONOTONIC', 'MFD_GR', 'MFD_GR_TAU', 'DATATYPES_EQUAL', 'SLIP_ALONG_RUPTURE', 'PLATE_RATE', 'CFM_TIMEFRAME', 'CFM_TEMPLATE_RHO', 'CFM_RATE_LINKS', 'SEED', 'SLIP_RATE_FACTOR', 'SLIP_SUM',
                            'DOWN_THE_MIDDLE', 'FLOOR', 'RUN_PLAN', 'TARGETS')
             if f'{s_}=' not in _pin]
check(not _unpinned, 'run_quick_checks.sh pins every setting its reference values depend on'
      + (f': not {_unpinned}' if _unpinned else ''))

print('5. Options and imports')
opts = {}
for s, t in trees.items():
    opts[s] = {a.value for n in ast.walk(t) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'add_argument'
               for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith('-')}
for i, c in cells:
    t = ast.parse(c)
    called = sorted({n.value for n in ast.walk(t) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in SCRIPTS})
    if 'dispersed_starts.py' in called and "'--fit-args'" in c:   # its --fit-args pass on to best_fit_as_fitted.py
        called = sorted(set(called) | {'best_fit_as_fitted.py'})
    used = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('--'):
            used.add(n.value.split('=')[0].split()[0])
        if isinstance(n, ast.JoinedStr) and n.values and isinstance(n.values[0], ast.Constant) and str(n.values[0].value).startswith('--'):
            used.add(n.values[0].value.split('=')[0])
    if not used:
        continue
    for o in sorted(used):
        where = [s for s in called if o in opts[s]]
        check(bool(where), f'cell {i}: {o} -> {", ".join(where) if where else "no called script (" + ", ".join(called) + ") defines it"}')
for s, t in trees.items():
    imps = {a.name.split('.')[0] for n in ast.walk(t) if isinstance(n, ast.Import) for a in n.names}
    imps |= {(n.module or '').split('.')[0] for n in ast.walk(t) if isinstance(n, ast.ImportFrom)}
    retired = imps & {'rhmc', 'evidence', 'build_transfers'}
    runs = re.findall(r"['\"](?:rhmc|evidence|build_transfers)\.py['\"]", src[s])
    check(not retired and not runs, f'{s}: imports {sorted(imps & {"cru_common", "rhmc", "evidence", "build_transfers"}) or "no local modules"}'
          + (f'; calls {runs}' if runs else ''))
g = trees['gibbs.py']
md_imp = [n for n in ast.walk(g) if isinstance(n, ast.ImportFrom) and n.module == 'cru_common' and any(a.name == 'make_dirs' for a in n.names)]
md_def = [n.name for n in ast.walk(g) if isinstance(n, ast.FunctionDef) and n.name == 'make_dirs']
cc_def = [n.name for n in ast.walk(trees['cru_common.py']) if isinstance(n, ast.FunctionDef) and n.name == 'make_dirs']
uses = sorted({f.name for f in ast.walk(g) if isinstance(f, ast.FunctionDef)
               for n in ast.walk(f) if isinstance(n, ast.Name) and n.id == 'make_dirs'})
check(bool(md_imp) and not md_def and bool(cc_def), f'gibbs.py imports make_dirs from cru_common (line {md_imp[0].lineno if md_imp else "-"}), '
      f'defines none itself; cru_common defines it; used in {uses}')
print(f'\n{len(fails)} failures' + (':\n  ' + '\n  '.join(fails) if fails else ''))

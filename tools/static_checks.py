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
SCRIPTS = ['gibbs.py', 'cru_common.py', 'preliminaries.py', 'block_fit.py', 'summarize_runs.py', 'compare_runs.py', 'compare_priors.py', 'select_draws.py', 'plot_rate_magnitude.py', 'plot_slip_rates.py', 'plot_paleo_rates.py', 'plot_slip_sections.py', 'check_convergence.py', 'make_tables.py', 'make_solution.py', 'randomized_nnls.py', 'plate_rate.py']
TOOLS = ['run_nb.py', 'run_quick_checks.sh', 'static_checks.py', 'compare_mfd.py', 'fig_mfd_levels.py', 'psis.py']
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
_gr = ''.join(c for _, c in cells if 'def gibbs_run(' in c)
check(not _cut and all(s in _gr for s in ("'--init', 'nnls'", "'--no-samples'", "'--swap-cos', 0.99")),
      'notebook: no PRIOR_KIND, THETA, BOX_TOP, USE_SWAPS, OUTPUT_DRAWS, SA_STARTS or BEST_FIT; its runs start from the best '
      'fit, store no draws and use swap moves' + (f'; found {_cut}' if _cut else ''))
check(bool(re.search(r"hyper_every = a\.hyper_every if a\.hyper_every is not None else \(1 if spike_mode else None\)", src['gibbs.py'])),
      'gibbs.py: --hyper-every defaults to 1 for spike-and-slab')
check(argdefault(trees['gibbs.py'], '--slab') == 'log' and "choices=['log', 'sqrt']" in src['gibbs.py']
      and src['gibbs.py'].count('ok &= selftest_spikeslab(gam=0.5)') == 1,
      f"gibbs.py --slab default {argdefault(trees['gibbs.py'], '--slab')!r} (log or sqrt), the square-root slab in the full self-test set")
check(re.findall(r'^PRESET = (\S+)', cells[0][1], re.M) == ["'conservative'"], "notebook PRESET = 'conservative' (the default)")
_pre = re.search(r'^PRESETS = \{.*?^\}', cells[0][1], re.M | re.S)
_ns = {}
if _pre:
    exec(_pre.group(0), {'dict': dict}, _ns)
_P = _ns.get('PRESETS', {})
check(sorted(_P) == ['conservative', 'quick', 'very_conservative'] and all('hyper_every' in v for v in _P.values())
      and not any("'--slab'" in c for _, c in cells),
      'notebook: one set of spike-and-slab run lengths per preset; the other priors are gibbs.py options')
check("elif a.prior == 'linear':" in src['gibbs.py'] and src['gibbs.py'].count('ok &= selftest_linear()') == 1,
      "gibbs.py --prior linear (command line only), with its exact test in the full self-test set")
check("halfway_mode = a.prior.startswith('halfway')" in src['gibbs.py'] and src['gibbs.py'].count('ok &= selftest_halfway()') == 1,
      "gibbs.py --prior halfway (command line only), with its exact test in the full self-test set")
_cfg_keys = re.findall(r'(\w+)=', ''.join(c for _, c in cells if 'CONFIG = dict(' in c))
check(re.findall(r"^DOWN_THE_MIDDLE = ('[\w-]+'|\w+)", cells[0][1], re.M) == ['False'] and 'down_the_middle' in _cfg_keys
      and any("if DOWN_THE_MIDDLE:\n        args += ['--down-the-middle']" in c for _, c in cells)
      and "'--down-the-middle', nargs='?', const='equal', default=None, choices=['equal', 'zero-model']" in src['gibbs.py']
      and 'balance_rows' in src['gibbs.py'] and "weights='equal'" in src['cru_common.py']
      and any("if DOWN_THE_MIDDLE == 'zero-model':\n            args += ['zero-model']" in c for _, c in cells)
      and src['gibbs.py'].count('ok &= selftest_balance()') == 2 and '\ndef balance_rows(' in src['cru_common.py'],
      'notebook DOWN_THE_MIDDLE = False (no balance term), recorded; True and \'zero-model\' passed to gibbs.py '
      '--down-the-middle (cru_common.balance_rows; its exact test, of both weightings, in both self-test sets)')
check(re.findall(r'^BAND = (\S+)', cells[0][1], re.M) == ["'positive'"] and 'band' in _cfg_keys
      and any("args += ['--band', BAND]" in c for _, c in cells) and "'--band'" in src['gibbs.py']
      and src['gibbs.py'].count('ok &= selftest_band()') == 2,
      "notebook BAND = 'positive' (every prediction above zero; a number: the band, each within that factor of its target), "
      'recorded and passed to gibbs.py --band or --positive, '
      'with its exact test in both self-test sets')
check(re.findall(r'^MFD_MONOTONIC = (\S+)', cells[0][1], re.M) == ['True'] and 'mfd_monotonic' in _cfg_keys
      and any("args += ['--mfd-monotonic']" in c for _, c in cells) and not any('MFD_RISE_SD' in c for _, c in cells)
      and "'--mfd-monotonic'" in src['gibbs.py'] and argdefault(trees['gibbs.py'], '--mfd-rise-sd') == 0.89

      and "'monotonicity')" in src['gibbs.py'].split('\ndef selftest_band')[1].split('\ndef ')[0],
      'notebook MFD_MONOTONIC = True (imposed by default, a rising target\'s rise capped at gibbs.py\'s 0.89 sd; '
      'gibbs.py\'s default too), recorded and passed to gibbs.py --mfd-monotonic --mfd-rise-sd, with its '
      'exact test in the band\'s self-test')
check(re.findall(r'^PLATE_RATE = (\S+)', cells[0][1], re.M) == ['None'] and 'plate_rate' in _cfg_keys
      and any("args += ['--plate-rate', PLATE_LIMITS]" in c and "'--rupture-set', RUPTURE_DIR]" in c for _, c in cells)
      and "'--plate-rate'" in src['gibbs.py']
      and 'from plate_rate import describe_limits, describe_plate, parse_limits, plate_rows' in src['gibbs.py']
      and "'the plate rate')" in src['gibbs.py'].split('\ndef selftest_band')[1].split('\ndef ')[0]
      and any("sh('plate_rate.py', _rd, '--run', results_dir()" in c and "['--limits', PLATE_LIMITS]" in c for _, c in cells)
      and '\ndef check_problem(' in src['plate_rate.py'] and '\ndef plate_rows(' in src['plate_rate.py'],
      'notebook PLATE_RATE = None (not imposed by default), recorded and passed to gibbs.py --plate-rate with the rupture set '
      '(rows from plate_rate.py on the CROSS_SECTIONS strips, checked against the slip-rate rows), with its exact test in '
      'the band\'s self-test; plate_rate.txt reported for every run with a rupture set')
check(re.findall(r'^CROSS_SECTIONS = (\S+)', cells[0][1], re.M) == ["'all'"] and 'cross_sections' in _cfg_keys
      and any('if PLATE_RATE is not None or SLIP_SUM:' in c and "'--cross-sections', CROSS_SECTIONS," in c for _, c in cells)
      and "'--cross-sections', default='all', choices=['all', 'between']" in src['gibbs.py']
      and "'--cross-sections', default='all', choices=['all', 'between']" in src['plate_rate.py']
      and '\ndef cross_sections(' in src['plate_rate.py'],
      'notebook CROSS_SECTIONS = \'all\' (gibbs.py\'s and plate_rate.py\'s default too), recorded and passed with '
      'SLIP_SUM or PLATE_RATE: the strips both use (plate_rate.py\'s cross_sections)')
check(re.findall(r'^SLIP_SUM = (\S+)', cells[0][1], re.M) == ['True'] and 'slip_sum' in _cfg_keys
      and "'--slip-sum', action='store_true'" in src['gibbs.py']
      and 'from plate_rate import describe_slip_sum, slip_sum_rows' in src['gibbs.py']
      and "summed slip rates\\' term recorded apart" in src['gibbs.py'].split('\ndef selftest_balance')[1].split('\ndef ')[0]
      and '\ndef slip_sum_rows(' in src['plate_rate.py'] and 'SLIP_SUM_L0 = 8.5 ' in src['plate_rate.py']
      and "Wc = W * cs['proj'][None, :]" in src['plate_rate.py']
      and argdefault(trees['gibbs.py'], '--slip-sum-l0') == 8.5
      and re.findall(r'^SLIP_SUM_L0 = (\S+)', cells[0][1], re.M) == ['8.5'] and 'slip_sum_l0' in _cfg_keys
      and any("args += ['--slip-sum', '--slip-sum-l0', SLIP_SUM_L0]" in c for _, c in cells)
      and any("'--cross-sections', CROSS_SECTIONS, '--slip-sum-l0', SLIP_SUM_L0" in c for _, c in cells),
      'notebook SLIP_SUM = True (added by default) and SLIP_SUM_L0 = 8.5 (km, gibbs.py\'s and plate_rate.py\'s '
      'default too), recorded and passed to gibbs.py --slip-sum with the rupture set (rows from plate_rate.py: each '
      'subsection once, its slip projected on the plate motion), recorded apart from '
      'chi2, with its bookkeeping in the balance term\'s self-test; its term in plate_rate.txt for every run')
check(re.findall(r'^BAND_SOFT = (\S+)', cells[0][1], re.M) == ['None'] and 'band_soft' in _cfg_keys
      and any("args += ['--log-band', BAND_SOFT]" in c for _, c in cells) and "'--log-band'" in src['gibbs.py']
      and src['gibbs.py'].count('ok &= selftest_logband()') == 1 and 'static double lw_change(' in src['gibbs.py']
      and 'static void lw_bounds(' in src['gibbs.py'],
      'notebook BAND_SOFT = None (no soft band by default), recorded and passed to gibbs.py --log-band (with positivity; '
      'kept by a Metropolis step in the kernel, its exact test in the full self-test set)')
check(any("args += ['--positive']" in c and "BAND == 'positive'" in c for _, c in cells) and "'--positive'" in src['gibbs.py']
      and 'def set_positive(' in src['gibbs.py'] and 'T.set_positive(' in src['gibbs.py'].split('\ndef selftest_band')[1].split('\ndef ')[0],
      "notebook BAND = 'positive' passes gibbs.py --positive (every prediction above zero), with its exact test in the "
      'band\'s self-test')
check(sum(c.count('*ref_args()') for _, c in cells) == 5 and '\ndef reference_rates(' in src['cru_common.py']
      and all("'--reference'" in src[sc] for sc in ('plot_slip_rates.py', 'plot_paleo_rates.py', 'plot_rate_magnitude.py',
                                                    'randomized_nnls.py')),
      'the fit figures take --reference (cru_common.reference_rates), and the notebook passes the model\'s OpenSHA '
      'solution to all of them (the best fit\'s and the posterior\'s)')
check(all('axis_span(' in src[sc] for sc in ('plot_slip_rates.py', 'plot_paleo_rates.py', 'randomized_nnls.py'))
      and 'best_fit_rates(' in src['plot_rate_magnitude.py'] and '\ndef axis_span(' in src['cru_common.py'],
      'the fit figures take fixed axes from the targets and the best fit (cru_common.axis_span, best_fit_rates)')
check(all("'--config'" in src[sc] and 'run_settings(' in src[sc] for sc in ('plot_slip_rates.py', 'plot_paleo_rates.py',
                                                                          'plot_rate_magnitude.py'))
      and '\ndef run_settings(' in src['cru_common.py'] and "('FLOOR', 'floor')" in src['cru_common.py'],
      'the fit figures show the run\'s settings, SLIP_SECTIONS to FLOOR, in place of the label (cru_common.run_settings)')
check(not any('RANDOMIZED_NNLS' in c or "'randomized_nnls.py'" in c for _, c in cells)
      and "'--selftest'" in src['randomized_nnls.py'],
      'notebook has no RANDOMIZED_NNLS (retired from the notebook on 2026-11-03; randomized_nnls.py stays for the command '
      'line, with its self-test)')
for s in ('gibbs.py', 'preliminaries.py', 'summarize_runs.py', 'plot_rate_magnitude.py', 'check_convergence.py'):
    d = argdefault(trees[s], '--diag-floor')
    check(d == -10, f'{s} --diag-floor default {d!r}')
m = re.findall(r"spike_mode:.*\n.*\n\s*W, se, top = float\(sp_\[0\] or (-?\d+)\)", src['gibbs.py'])
check(m == ['-10'], f'gibbs.py spike-and-slab default slab floor {m}')
cfg = cells[0][1]
check(re.findall(r'^FLOOR = (-?\d+)', cfg, re.M) == ['-10'] and not re.search(r'^(SLAB|DIAG)_FLOOR', cfg, re.M),
      f"notebook FLOOR = {re.findall(r'^FLOOR = (-?[0-9]+)', cfg, re.M)}, no SLAB_FLOOR or DIAG_FLOOR setting")
for script in ('preliminaries.py', 'gibbs.py', 'plot_rate_magnitude.py', 'check_convergence.py'):
    cc = [i for i, c in cells if 'sh(' in c and f"'{script}'" in c
          and not (c.count(f"'{script}'") == 1 and re.search(rf"sh\(\s*'{script}',\s*'--selftest'(,\s*'\w+')?\)", c))]
    ok = bool(cc) and all('--diag-floor={FLOOR}' in c for i, c in cells if i in cc)
    check(ok, f"notebook passes --diag-floor={{FLOOR}} to {script} (cells {cc})")
check(any("spikeslab:{FLOOR}:soft" in c for _, c in cells), 'notebook uses spikeslab:{FLOOR}:soft')
check(re.search(r'clip = 10\.0 \*\* floor\b', src['check_convergence.py']) is not None and not any('clip = 10.0 **' in c for _, c in cells),
      'check_convergence.py clips the convergence diagnostics at 10 ** --diag-floor (the notebook passes FLOOR); the notebook computes none itself')
cc = [i for i, c in cells if 'sh(' in c and "'check_convergence.py'" in c]
check(len(cc) == 1 and all("'--preset', PRESET" in c for i, c in cells if i in cc) and any("'make_tables.py'" in c for _, c in cells),
      f'notebook runs check_convergence.py once with its PRESET (cells {cc}), then make_tables.py')
keys = None
for i, c in cells:
    for n in ast.walk(ast.parse(c)):
        if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == 'CONFIG' for t in n.targets) and isinstance(n.value, ast.Call):
            keys = [k.arg for k in n.value.keywords]
check(keys is not None and 'floor' in keys and not {'slab_floor', 'diag_floor'} & set(keys),
      f'recorded configuration keys: {keys}')
check(argdefault(trees['preliminaries.py'], '--slip-sections') == 'auto',
      f"preliminaries.py --slip-sections default {argdefault(trees['preliminaries.py'], '--slip-sections')!r}")
check(re.findall(r"^SLIP_SECTIONS = ('[a-z]+')", cfg, re.M) == ["'auto'"], f"notebook SLIP_SECTIONS = {re.findall(r'^SLIP_SECTIONS = (.*?) ', cfg, re.M)}")
cc = [i for i, c in cells if 'sh(' in c and "'preliminaries.py'" in c and '--selftest' not in c]
check(bool(cc) and all("'--slip-sections'" in c for i, c in cells if i in cc), f'notebook passes --slip-sections to preliminaries.py (cells {cc})')
check(keys is not None and 'slip_sections' in keys, 'recorded configuration includes slip_sections')
check(argdefault(trees['preliminaries.py'], '--mfd-uncertainty') == 'as-given',
      f"preliminaries.py --mfd-uncertainty default {argdefault(trees['preliminaries.py'], '--mfd-uncertainty')!r}")
lb = {sc: (argdefault(trees[sc], '--lowest-bin'), argdefault(trees[sc], '--bin-width')) for sc in ('preliminaries.py', 'plot_rate_magnitude.py')}
check(lb['preliminaries.py'] == lb['plot_rate_magnitude.py'] == (6.8, 0.1),
      f'--lowest-bin and --bin-width defaults agree between preliminaries.py and plot_rate_magnitude.py: {lb}')
check(re.findall(r"^MFD_UNCERTAINTY = ('[a-z-]+')", cfg, re.M) == ["'as-given'"],
      f"notebook MFD_UNCERTAINTY = {re.findall(r'^MFD_UNCERTAINTY = (.*?) ', cfg, re.M)}")
cc = [i for i, c in cells if 'sh(' in c and "'preliminaries.py'" in c and '--selftest' not in c]
cc_args = [i for i, c in cells if "'preliminaries.py'" in c and "'--mfd-uncertainty', MFD_UNCERTAINTY" in c]
check(bool(cc) and set(cc) <= set(cc_args), f'notebook passes --mfd-uncertainty MFD_UNCERTAINTY to preliminaries.py (cells {cc_args})')
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
check(not any('TARGET' in c or "'--target'" in c for _, c in cells) and keys is not None and 'target' not in keys,
      'notebook has no TARGET setting, passes no --target, and records no target')
pre_opts = {a.value for n in ast.walk(trees['preliminaries.py']) if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'add_argument'
            for a in n.args if isinstance(a, ast.Constant)}
check('--keep-soft-penalties' in pre_opts and '--no-nnls' not in pre_opts,
      'preliminaries.py has --keep-soft-penalties and always computes the best fit (no --no-nnls)')
dense_g = re.findall(r'\bG[a-z]?\.toarray\(\)', src['preliminaries.py'])        # a whole G made dense: 4 GB at 2e5, 11 at 5e5
check(not dense_g and 'qr_insert' in src['preliminaries.py'], 'preliminaries.py makes no dense copy of G (the best fit updates a QR '
      'factorization)' + (f'; found {", ".join(dense_g)}' if dense_g else ''))
st_cells = [i for i, c in cells if re.search(r"sh\(\s*'gibbs\.py',\s*'--selftest'", c)]
check(len(st_cells) == 1 and all("sh('gibbs.py', '--selftest', 'current')" in c for i, c in cells if i in st_cells)
      and re.search(r"'--selftest', nargs='\?', const='all', choices=\['all', 'current'\]", src['gibbs.py']) is not None,
      f"notebook's self-tests run gibbs.py --selftest current (cells {st_cells}); gibbs.py --selftest takes 'all' or 'current'")
cc = [i for i, c in cells if "'nnls/nnls_vectors.npz:best fit (NNLS)'" in c]
post = [i for i, c in cells if "sh('plot_paleo_rates.py', vec" in c]
check(len(cc) == 1 and all(f"'{s}'" in c for i, c in cells if i in cc for s in ('plot_slip_rates.py', 'plot_rate_magnitude.py',
      'plot_paleo_rates.py')) and bool(post) and cc[0] > post[0] and "'nnls_vectors.npz'" in src['preliminaries.py'],
      f'the best fit (nnls_vectors.npz, which preliminaries.py writes) is drawn after the posterior (cells {cc}), so a reference '
      'solution finished during the run is marked')
check("'nnls_vectors.csv'" in src['preliminaries.py'], 'preliminaries.py also writes the best fit as a table, nnls_vectors.csv')
cc = [i for i, c in cells if "'make_solution.py'" in c and 'sh(' in c]   # the solutions cell, the last
check(re.findall(r"^RUPTURE_SET = ('[^']*')", cfg, re.M) == ["'auto'"] and 'find_rupture_set(DATA_DIR)' in cfg
      and '\ndef rupture_dir():' in cfg and len(cc) == 1 and cc[0] == cells[-1][0]
      and all("'nnls/nnls_vectors.npz'" in c and "'--column', 'mean'" in c and "'--ruptures', _rd" in c and 'rupture_dir()' in c
              for i, c in cells if i in cc)
      and "os.path.join(os.path.dirname(stem), 'nnls')" in src['preliminaries.py'] and {'rupture_set', 'rupture_dir'} <= set(keys or []),
      f"the best fit's outputs go into RUN_DIR/nnls/ (preliminaries.py); the last cell ({cc}) writes it and the posterior mean "
      "as OpenSHA solutions (make_solution.py) for the rupture set found then (RUPTURE_SET = 'auto'); both recorded")

# every model setting recorded, in the figures' settings table, in the run name, in the summary PDF and documented
MODEL_SETTINGS = ['SLIP_SECTIONS', 'MFD_UNCERTAINTY', 'DOWN_THE_MIDDLE', 'BAND', 'BAND_SOFT', 'MFD_MONOTONIC',
                  'CROSS_SECTIONS', 'SLIP_SUM', 'SLIP_SUM_L0', 'PLATE_RATE', 'PRESET', 'FLOOR']
_cfgmap = dict((v, k) for k, v in re.findall(r'(\w+)=([A-Z][A-Z0-9_]+)', ''.join(c for _, c in cells if 'CONFIG = dict(' in c)))
_runname = cfg[cfg.index('RUN_NAME = ('):cfg.index('RUN_DIR = ')]
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
                             ('in the notebook\'s markdown', s_ in _md), ('in REPORT.md', s_ in _docs['docs/REPORT.md']),
                             ('in CLAUDE.md', s_ in _docs['docs/CLAUDE.md'])) if not ok_]
    if miss:
        _gaps.append(f'{s_}: not ' + ', '.join(miss))
check(not _gaps, f'the {len(MODEL_SETTINGS)} model settings are each defined, recorded, in FIGURE_SETTINGS, in RUN_NAME, in the '
      'summary PDF and documented (notebook, REPORT.md, CLAUDE.md)' + (': ' + '; '.join(_gaps) if _gaps else ''))
_pin = re.search(r'PIN=\((.*?)\)', open(f'{ROOT}/tools/run_quick_checks.sh').read(), re.S).group(1)
_unpinned = [s_ for s_ in ('BAND', 'BAND_SOFT', 'MFD_MONOTONIC', 'PLATE_RATE', 'SLIP_SUM', 'DOWN_THE_MIDDLE', 'FLOOR', 'PRESET')
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

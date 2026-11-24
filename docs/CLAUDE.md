# Posterior sampling for a nonnegative, underdetermined linear inverse problem

Paths below are relative to the repository's folder (this file is in its docs/, beside the documents it imports).
Design notes: @sampler-design.md (section 0 records what the 2e4 data showed and overrides the rest)
Findings, numbers and run commands: @REPORT.md
What to test and check before routine use: @TESTING.md
Workflow notebook: cru_workflow.ipynb (the configuration, then one call per step: it drives the scripts below end to
end and computes nothing itself; it finds them in tools/ beside it). Since 2026-11-17 (the user's request) the notebook
keeps only its settings (one-line comments), the path setup and the scripts' calls, with a sentence or two of markdown per
step; tools/notebook_steps.py holds the plumbing: setup(globals()) (run_config.derived_settings, every setting's check,
SEED_USED/MFD_GR_SPEC/PLATE_LIMITS, DATA_DIR/RUN_DIR resolved, RUN_DIR made and entered, the log, RUPTURE_DIR,
CONFIG via record_config -> workflow_config.json), sh (SH_SECONDS), show, results_dir, rupture_dir, ref_args,
cleaning_args/after_cleaning, gibbs_args/_terms/starts_args, plan_runs (calibration from the starts, run_plan.json,
config), replicate_seed, runs_done, clear_solution, write_timings (timings.json). The old markdown and the settings'
full comments are in REPORT.md, "The notebook, step by step". DOWN_THE_MIDDLE (False), SLIP_BLOCK, BIN_BLOCK are fixed
in run_config.py (FIXED; DOWN_THE_MIDDLE shown beside RUN_MINUTES and in the run name only when not False). No self-test
cell or RUN_SELFTESTS; run_nb.py NOTEBOOK RUN_DIR DATA_DIR [NAME=VALUE ...]. The cleaning figure (preliminaries.py
--figure) and the target figures (plot_targets.py) are on request.
Since 2026-11-21 (the user's request): the notebook passes preliminaries.py --no-nnls (no best fit; DISPERSED_STARTS fixed True
in run_config.py, False only for tests, which get the fit back) and, always since 2026-11-23 (with MFD_GR and the CFM from
2026-11-21; cru_A/cru_D never read, A_FILE/D_FILE gone from the notebook), --from-rupture-set RUPTURE_DIR
(rupture_set_inputs.build_inputs: the inputs' rows from the solution folder's ruptures/; cru_A/cru_D not read; checked
equivalent at 19k, 211k, 630k; REPORT "Inputs from the rupture set, and no best fit"). Since 2026-11-23 a bare rupture set
(no inversion) also works: the missing modules are built (slip rates 0.8 x SlipRate from fault_sections.geojson, paleo sites
from data/ mapped to the nearest subsection within 5.25 km as NZSHM22 does (paleo_misfit.paleo_data), the bins from
mfd_gr.rupture_set_parts; preliminaries.py --nzshm22-n5 for NZSHM22's run N5, default 3.4), the subsections' shares of NZSHM22's gridded seismicity from their polygons
(fault_polygons.py, a port of SectionPolygons/NZSHM22_FaultPolyMgr; data/Floor_AddoptiEEPAScomb-CRU_pdf.csv; within 0.15%
of NZSHM22's partition, a Java Area quirk; the user chose pure Python), NZSHM22's run N5 3.4, b 0.959 (mfd_gr.NZSHM22_RUN);
a solution's own files are read when present (exact). REPORT "The sub-seismogenic and off-fault shares from the rupture set".
Since 2026-11-23 CFM_SLIPRATES is always True (not overridable; cfm_sliprates.rescale_inputs and --slip-sd-multiplier removed;
the user: NZSHM22's own slip-rate setup, with its factors of 2, is not to be matched). REPORT "The slip-rate constraint, step by
step" traces a row from the CFM to the likelihood; tools/trace_slip_rows.py checks any fault's rows by hand against cru_clean.npz. make_summary_pdf numbers its steps
by the sections present (SN). The best-fit cells left the notebook.
Code, all in tools/: preliminaries.py (clean, with soft zero-forcing rows forcing their ruptures to 0 like the penalty rows; diagnostics;
NNLS, always computed (about 25 s at 211k; the chains start from it); the best fit's outputs go into RUN_DIR/nnls/: its rate vectors,
nnls_vectors.npz, which the notebook draws with the three fit-figure scripts below, and nnls_vectors.csv, a table with
rate_vectors.csv's rows), make_solution.py (a rate vector as an OpenSHA fault system solution's rates.csv, after
checking that the model's OpenSHA rupture set describes the same ruptures; --zip FILE also writes a complete archive
that OpenSHA loads; the notebook's last cell writes the best fit's, nnls/rates.csv, and the posterior mean's,
spikeslab/rates.csv, one row per rupture of the rupture set like the
model's own rates.csv, when DATA_DIR holds the model's
OpenSHA solution folder, which RUPTURE_SET = 'auto' finds, then (rupture_dir())), make_summary_pdf.py (a finished
run's summary as a PDF, about a paragraph per step with the run's figures, every number read from its reports; command
line only, needs reportlab (uv run --no-project --with reportlab --with pillow --with numpy python ...); --md FILE writes
the same as Markdown instead, without reportlab, figures linked from --fig-dir, --pdf-link naming the PDF; --skip-figs
leaves figures out of either, the rest renumbered and the text naming a left-out one's file; --ref-name shortens the
reference solution's name. The example (since 2026-11-24: the 630k run _1795439419, MISFIT_SLIP 'exact' + MFD_GR +
DATATYPES_EQUAL + SLIP_SUM + PLATE_RATE, 300 min, 16 x 14,800; before, _1794997938, _1794909134 and _1794825297) is
example/output_figs (the fit figures, fig_targets_*, the fault figures by magnitude and along_strike/), the summary PDF
(workflow_summary.pdf in the repository's root since the user's d3ddc96, 2026-11-19; before, outside it at
Inversions/Material/), and README.md, both made with --ref-name NZSHM22_InversionSolution-630k --skip-figs prelim.png
fig_rate_magnitude_nnls.png fig_slip_rates_nnls.png fig_paleo_rates_nnls.png fig_slip_sections.png
fig_slip_sections_ruptures_1.png fig_slip_sections_ruptures_2.png (the README with --md README.md --fig-dir
example/output_figs --pdf-link example/workflow_summary.pdf --no-preamble --no-alternatives (2026-11-19, the user's request:
no "Other ways to explore the model space" in the README; the PDF keeps it): the user trimmed the README to open on the
overview's text, no run line, folder, dates, Overview heading or how-written line); example/output_figs holds only the figures shown (the three
target figures, made by hand in RUN_DIR with plot_targets.py --old-sd 1 (and --old-sd 2 saved as fig_targets_slip_sd2), the three
fits and the 33 named faults' MFDs) with their tables, example/NZSHM22_InversionSolution-630k.zip the OpenSHA solution;
cru_A/cru_D are not in the example (148 MB zipped at 630k)), gibbs.py (production sampler; Gaussian
likelihood), check_convergence.py (a result set's convergence checks and PASS/CHECK verdict, with each rupture's R-hat
and effective draws in convergence.npz), make_tables.py (the posterior tables and rate vectors, pooled over the
replicates), block_fit.py (each row block's fit from a run's recorded predictions),
compare_runs.py (rate-by-rate agreement of two runs, with a figure), summarize_runs.py (log-space summaries of
replicates or priors, from the running summaries or stored draws; command line only since the notebook's summary
figure was dropped at the end of October 2026), cru_common.py (shared I/O and diagnostics), compare_priors.py (tables comparing spike-and-slab runs: start, slab floor; trimmed 2026-11-23),
select_draws.py (weighted representative draws for hazard), plot_rate_magnitude.py (each rupture's posterior mean
rate against its magnitude bin, from the rate vectors, with each bin's ruptures counted by decade of mean rate, and
below it each bin's total over its target with the total's posterior 16-84% range),
plot_slip_rates.py (slip-rate targets against the mean model's predictions, linear and log, from the rate vectors, with
each section mean's posterior 16-84% range), plot_paleo_rates.py (paleo-rate targets against the model's predictions,
log-log, with each prediction's posterior 16-84% range; with kept draws, the default, all of these figures and
plot_slip_sections.py's draw each kept draw as an x instead (cru_common.kept_draws, kept_predictions, draw_xs); the three ranges come from the predictions the runs next to
the rate vectors recorded, main_rep<k>.npz, cru_common.run_predictions; the slip-rate sections and paleo rows are
coloured from cru_common.BLUES6_CYCLE (ColorBrewer Blues, the darkest 6 of 9 classes, light and dark alternating; every
figure since 2026-11-23, the user's choice), cycled along the targets so that neighbours differ, neighbour_colours; the
reference model is black, a second one red (#b2182b), and the MFD figure's bin totals dark blue (#08519c)), run_config.py (2026-11-17, the user's request: the configuration cell's derived settings moved out of the notebook: derived_settings(globals()) returns each group's misfit mode, MISFIT, the fixed settings BLOCKS, SLIP_SECTIONS, RUPTURE_SET, CROSS_SECTIONS, CFM_SLIPRATES, CFM_RATE_LINKS (module constants; run_nb.py's NAME=VALUE for these and RUN_DIR arrive as _OVERRIDES), SD_MULTIPLIER_, _T0, RUN_NAME (run_name) and RUN_DIR; the notebook keeps its path setup), SLIP_ALONG_RUPTURE (2026-11-17: 'uniform' default, 'tapered' UCERF3 sqrt-sine by area, plate_rate.slip_taper; preliminaries.py --slip-along tapers the raw slip rows after the CFM rebuild and groups sections on the uniform coefficients; the problem records slip_along, load_problem defaults 'uniform', rupture_matrix(rd, n, along) applies it for cross_sections and plot_slip_rates; RUN_NAME _dsrtapered; run_quick_checks pins 'uniform'), plot_targets.py (2026-11-17, the user's request: the targets old and new, from the cleaned problem and the rupture set: fig_targets_slip.png (per section at the inputs' target 0.8 SlipRate, its Gaussian sd 2 SD grey 16-84%, and the run's split normal from the problem's rows, sigma_ref/sqrt(split_lo2|hi2), coloured 15.9-84.1% via timeframe_sliprates.sn_quantile, dot at the mode; linear and log panels, colours as fig_slip_rates), fig_targets_paleo.png (bestEstimate +- sd against the site's ONE_SIGMA bounds), fig_targets_mfd.png (MFD_GR's bin PDFs and cumulative as-scored band against cru_D's targets, from M6.5; the run's MFD_GR spec or its defaults); written next to the problem; make_summary_pdf.py shows them at the end of section 4 with 'The targets, old and new'), covariance.py (2026-11-17, the user's request: from the runs' kept draws, the correlation between the named faults' event rates and between the bins' totals, beside the data-only correlation of their resolved parts from G G^T's eigenvectors, and each one's resolved fraction; corr_faults.png, corr_bins.png, covariance_report.txt, covariance.npz in the results folder; the notebook's fault cell runs it when the kept draws exist), plot_slip_sections.py (the posterior slip-rate predictions of the rupture set's named
faults, named_faults.json, each the subsections whose parent, fault_sections.geojson's ParentID, is one of its own: one
column each, from the most isolated to the most connected by the number of other sections sharing a rupture with them,
with each one's share of draws without an event; a second figure counting each fault's ruptures by posterior mean
rate, with the reference model's rates of the same ruptures beside them; and a third summing those rates by magnitude
bin, each fault's rate of events per bin against the reference's; --named none draws slip-rate sections instead, 30 by
default; the notebook draws them after the paleo-rate figure),
randomized_nnls.py (best fits to the data plus fresh draws of their
stated noise, from the command line only since 2026-11-03, when the notebook's RANDOMIZED_NNLS was retired: their spread of
slip-rate, MFD and paleo predictions in figures, in the --out folder; the best fit's sampling distribution, not a posterior).
The fit figures (plot_slip_rates.py, plot_rate_magnitude.py, plot_paleo_rates.py, randomized_nnls.py) take --reference
RATES[:LABEL] (cru_common.reference_rates): another model's rates, marked on every panel and tabulated (since 2026-11-23
plot_slip_rates, plot_rate_magnitude, plot_paleo_rates and plot_slip_sections take it twice, cru_common.reference_list: the
second a red diamond and a dashed red outline, a line more of subtitle; with one, figures and tables as before); the notebook
passes up to two OpenSHA solutions (ref_args(), notebook_steps.reference_solutions, since 2026-11-23, the user's request: the
rupture set's own solution first, then every other folder in DATA_DIR with solution/rates.csv on the same rupture set, by
name; beyond two, left out and named at setup; two are labelled by what tells them apart in regional_inversion_target_mfds.json,
e.g. "NZSHM22, MFD σ ×0.1" / "×1"; one keeps the default label; none, no reference), looked for when each
figure is drawn, so one finished during the run counts; CONFIG records reference (the first) and references (all); the best fit's figures come right after cleaning since
2026-11-05, the user's choice, to catch a data type far off before the runs) when DATA_DIR holds it.
Their axes are fixed per problem by the targets and the best fit (cru_common.axis_span and best_fit_rates: all of the
targets and the central 96% of the best fit's predictions, from prelim_results.npz), so every model's figures for one
problem share them; values beyond sit on the edges as triangles. In place of the result set's label, each shows the
run's settings, SLIP_SECTIONS to FLOOR (cru_common.run_settings, from workflow_config.json next to --problem; --config),
and the rates-by-magnitude figure gives the chi2 over the bin rows.
paleo_misfit.py (the paleo rows' curves for --paleo-misfit, from the rupture set's paleo_constraint_data.json: a report of their fit with python paleo_misfit.py RUPTURE_SET). cfm_sliprates.py (preliminaries.py --cfm-sliprates: the slip-rate rows rebuilt from the NZ CFM's SR_pref/SR_min/SR_max; a report with python cfm_sliprates.py CFM_CSV RUPTURE_SET --A --d). rhmc.py (log-space HMC cross-check), evidence.py (evidence by stepping-stone sampling) and build_transfers.py (transfer
moves, single box only) are set aside in cru_retired_scripts.zip; restore them only if that work resumes.
Development tools, also in tools/ and not run by the notebook: run_nb.py (runs the notebook headless, cell by cell,
with configuration values overridden from the command line: "RUN_MINUTES=10", CELLS=5, "RUN_PLAN=dict(chains=4, sweeps=2400,
burn=400, thin=8, replicates=1)" through _OVERRIDES, ...), run_quick_checks.sh
DATA_DIR RUNS_DIR (the smoke test of docs/TESTING.md at the old quick lengths, RUN_PLAN fixed; one notebook run at a time, resumable),
static_checks.py (inventory, parsing, leftovers, defaults, options and imports of the code set: update it whenever a
default, a setting or the file list changes), compare_mfd.py (two result sets compared on fit, bins, total rate, MFD
slope, population and products, with replicate noise for scale) and psis.py (Pareto-smoothed importance sampling, PSIS-LOO, as used for the paleo rows'
leave-one-out checks).
gibbs.py defaults to --prior spikeslab (spike-and-slab per magnitude bin: each rupture off, rate 0, or on, log-uniform
from the slab floor 1e-10 up to a soft top at the bin's target rate R_j whose width is the bin's own uncertainty; the
fraction on in each bin estimated under Beta(1, 1); spikeslab:A:TOP:a:b; --hyper-every defaults to 1 for it, so the
fractions are redrawn every sweep; --fractions-on collapsed integrates them out instead, tested 2026-11-24, not the default). --slab sqrt (command line only; not in the notebook) was its cross-check until the
user dropped it at the end of October 2026 (the request behind it had been misphrased; do not propose it as a check):
the same with an on rupture's rate uniform in sqrt(rate), density m^-1/2 instead of 1/m; it fitted as the default does,
and at 211k it moved the subsection event rates (REPORT.md, "Fits and the prior's lower end").
The linear prior (each rate uniform from 0 to R_j) and the halfway prior (a 50/50 mix of uniform and log-uniform) were
removed on 2026-11-23 (the user's request), with their self-tests and the kernel's mixture constants (MIX, mA/mB); both
pushed the bin totals up (REPORT.md, "Why not a linearly uniform prior"). gibbs.py --prior refuses them by name.
--down-the-middle adds a balance term to the misfit, for any prior: each data type's relative shortfall s = sum of
predictions / sum of targets - 1 (slip-rate sections, MFD bins, paleo rows, in units of each measurement's uncertainty)
costs A s^2. DOWN_THE_MIDDLE = 'zero-model' (--down-the-middle zero-model; the notebook's default in late October
2026) gives each type its own A, its zero-model misfit; DOWN_THE_MIDDLE = True (--down-the-middle alone) one A for
all types, the largest that keeps every type at or below one extra measurement (cru_common.balance_rows weights;
extra rows of G and d). The notebook's default is False since early November 2026, with the band. The recorded chi2 and
predictions stay on the data rows, and the term is recorded on its own; the best fit and randomized_nnls.py ignore it.
[Since 2026-11-23 the factor and soft bands' options (--band, --slip-band, --paleo-band, --log-band, --slip-log-band,
--paleo-log-band), band_start and selftest_logband are removed (the user: the misfits make them needless); --all-positive,
--mfd-monotonic and --plate-rate keep the limits machinery (set_groups, band_bounds, selftest_band), and the log misfits keep
the soft band's kernel path (lw_change, col_dots_lw). What follows on the bands is history.]
Bands by row group (since November 4th, 2026; the notebook's SLIP_BAND, SLIP_BAND_SOFT until 2026-11-10, and until November 5th PALEO_BAND,
PALEO_BAND_SOFT, since then gibbs.py options only: --paleo-misfit years keeps the paleo rows above zero itself):
--slip-band and --paleo-band (positive or a factor F) and --slip-log-band and --paleo-log-band (TAU, with that group's
positive) act on the subsections' slip-rate rows and the paleo rows separately (Target.set_groups; the MFD bins get no
band, which would count them twice); they replace, in the notebook, BAND and BAND_SOFT, which acted on every
positive-target row through the older flags below, still there for older runs and tests and not combinable with the new
ones. Defaults: every prediction above zero (--all-positive; SLIP_BAND out of the notebook since 2026-11-10), no soft band. --paleo-misfit years (PALEO_MISFIT, 'years' by default since 2026-11-05,
'gaussian' before; with --rupture-set; the paleo rows kept above zero by it, pa_b = 'positive' in the per-group limits,
unless --paleo-band gives a factor; with 'gaussian' they get no limit unless --paleo-band is given): the paleo rows scaled out of the Gaussian (by PALEO_EPS = 1e-9, as
constraint rows are) and each scored instead by -2 ln of its site's recurrence-interval PDF in years, a sinh-arcsinh
curve in log rate through the four quantiles of paleo_constraint_data.json with its mode at 1/bestEstimate exactly
(paleo_misfit.py: fit_site, tables; cubic Hermite tables with slopes and a curvature bound kappa per site), kept by the
soft band's machinery (pmctx in the kernel: pm_f in lw_change and lw_swap, pm_coef's bound coefficients in col_dots_lw's
sums; the NumPy twin _pm_f_np); recorded as tr['paleo_misfit'] beside chi2, which still counts the paleo rows' Gaussian,
for comparison (their predictions unscaled before recording); the swap pairs' inner products lose the paleo rows'
share; selftest_paleo (both backends, against a grid integral; and set_groups against set_positive and set_band) in the
full set; without it every run is bit-identical (REPORT.md, "The paleo rows' misfit").
--mfd-misfit log10 (MFD_MISFIT, 'gaussian' by default; spike-and-slab, no transfers, not with the old --log-band): the
MFD bins' rows scaled out of the Gaussian (PALEO_EPS, as the paleo rows) and each bin's total scored by
((log10(total) - log10(R_j)) / s_j)^2, s_j = log10(1 + sigma_j / R_j) = log10(1 + 1/d_j) (the soft top's width; d_j the
cleaned row's target, so MFD_UNCERTAINTY's level is in it), centred on R_j, through the soft band's weights
(T.lw['W'][bins] = 1/(2 (s_j ln 10)^2), LD = ln of the scaled target; no kernel change) with T.lw['MF'] = dict(rows, s,
eps, d_data); positivity on the bins as a third group of set_groups (grp switched on for it if no other per-group flag
is given); the swap pairs lose the bins' share; recorded as tr['mfd_misfit'] beside chi2 (which still counts the bins'
Gaussian, the recorded predictions unscaled), with tr['mfd_log_rows'] and tr['mfd_log_s'], from which block_fit.py
scores the MFD block in log10 (its spread the rest of the mean); selftest_mfdlog (both backends, against a grid
integral, with positivity and swaps) in the full set; without it every run is bit-identical (REPORT.md, "The MFD bins'
misfit in log10").
chi2 as fitted (2026-11-05, the user's request: "the chi-squared always what the sampler fits"): gibbs.py records
tr['misfit'] = chi2 with the paleo rows' and the bins' Gaussian shares replaced by tr['paleo_misfit'] / tr['mfd_misfit']
(equal to chi2 when every misfit is Gaussian), tr['misfit_expect'] = cleaning's chi2_expect with each paleo site's 1 and 2
replaced by paleo_misfit.site_moments (mean and variance of site_misfit with u drawn from the site's curve), and per site
tr['paleo_th', 'paleo_best', 'paleo_sd', 'paleo_E', 'paleo_V', 'paleo_rows']; the sweep line, report(),
check_convergence.py (misfit and misfit_expect when present, else chi2 and the problem's), block_fit.py (paleo block by
site_misfit, its measurements sum E, sd sqrt(sum V); prints "fit check as fitted X +- Y", which make_summary_pdf.py
reads), select_draws.py, plot_paleo_rates.py (curves refitted from the rupture set when the config says years) and
plot_rate_magnitude.py (log10 when the config says so; cru_common.run_config) use it. best_fit_as_fitted.py (the user's
request, 2026-11-05): the best fit under chi2 as fitted, L-BFGS-B with rates >= 1e-20, column-norm
scaling, stop when the misfit falls < 0.01 over 500 iterations (scipy >= 1.11 callback StopIteration), from a positive
start since 2026-11-09 (positive_start, as gibbs.py's: NNLS rates below --floor 1e-10 at 1e-20, then ruptures switched on
at 2e-10 where a log-scored row is predicted at 0; from the NNLS fit itself, under MISFIT 'half' at 211k, 155 slip rows
sat at ~1e-20 and L-BFGS-B stopped ABNORMAL at 0 iterations, run _1794177625: 15,159 -> start 2,672 -> 180.8 in 108 s);
an abnormal stop is reported as a Warning, and the misfit reported and saved is fun at the rates returned (r.fun was
another point's); writes nnls/fitted_vectors.npz (model 'best fit as fitted') and _report.txt; the notebook's best-fit
figure cell runs it when CFM_SLIPRATES or a non-Gaussian misfit is on and draws both fits side by side in the
rates-by-magnitude figure, the NNLS fit alone in the slip-rate and paleo figures (the user, 2026-11-09). The notebook's runs start
from it when nnls/fitted_vectors.npz exists (gibbs.py --init fitted; refuses a fit whose nnls_key, sha1 of the NNLS rates
on the kept columns, or paleo_misfit / mfd_misfit differ from the run's); no visible burn-in gain at 211k (chi2 as
fitted reaches the posterior's ~1,100-1,250 within 8 sweeps from either start). SEED (2026-11-06, the user's request
after two identical conservative runs, _1793870170 and _1793899188): None -> _T0, the start time in the run name;
replicate_seed(r) = SeedSequence([SEED_USED, r]).generate_state(1)[0], or 101 + r with 'legacy' (run_quick_checks pins it);
recorded as CONFIG['seed']. SLIP_MISFIT 'split+log10' (2026-11-09, the user's option 2; the log sd is not a setting: the user asked for the same effective sd
as the linear half, as MFD_MISFIT log10): cru_common.slip_log_sds gives each row s_lo, s_hi = log10(1 + sd_side/(t target)), t the
template weight, and 1/k = (row_scale/t)^2; gibbs.py --slip-misfit split+log10: slip rows (and d) scaled by 1/sqrt 2 beside the
paleo/MFD scaling, swap pairs' inner products less half the slip rows' share, T.lw built as the soft band with SIDED weights
(W above the target, WL below, each (1/k)/(4 (s ln10)^2); the kernel's LWL pointer, NULL otherwise so other runs are
bit-identical; lw_coef's sided bounds; _lw_sq is the NumPy twin), LD = ln T.d, tau = ones (log_band_out counts rows 10x off),
plus T.lw['SM'] (rows, s_lo, s_hi, d_data, E, V, t2, t4 from cru_common.slip_geo_moments); exclusive with every soft band;
--slip-band positive set automatically; recorded predictions unscaled (y *= sqrt 2); misfit += log_band - 0.5 sum C_ over
the slip rows; tr['slip_geo_s', 'slip_geo_expect']; selftest_slipgeo (grid integral, a lopsided row, both backends, misfit
recorded); block_fit, best_fit_as_fitted (--slip-misfit; fitted file records it, and --init fitted checks it),
plot_slip_rates (from the config) use cru_common.slip_geo_rows. The three fit figures (2026-11-09, the user's request) give
each model's misfit (and the reference's) as chi2 as fitted, then the linear and the log misfit each alone, unweighted, as if
it were the whole misfit (cru_common.geo_parts, misfit_parts, misfit_text; the paleo rows' log misfit the years curve, fitted
for the comparison under 'linear' too; rows predicted at 0 counted, their log misfit inf), one line per model in the
subtitle; and every figure's settings table lists the model and all of FIGURE_SETTINGS (the 19 notebook settings, '—' when
not recorded) in 3 lines of SETTINGS_PER_LINE = 7 columns (cru_common.figure_settings, settings_lines), with room reserved
for the fullest legend, the reference's subtitle line and a 3-line footnote, so the figures and their panels keep one size
(slip and paleo 10.6 x 8.8 in; the rates-by-magnitude figure's width still follows its number of panels).
make_summary_pdf.py's patterns read both the old and the new wording.
MISFIT (2026-11-10, the user's request: half linear / half log for every group, one setting): 'linear' | 'log10' | 'half'
(default) -> (slip, mfd, paleo) modes via MISFIT_MODES (cru_common; the notebook has its copy MISFIT_MODES_ and derives
SLIP_MISFIT, MFD_MISFIT, PALEO_MISFIT from it; CONFIG records misfit and the three). misfit_log_weight(mode) = b: 0 linear,
1/2 any 'a+b', 1 log ('log10', 'years', 'log'). Paleo: linear = the inputs' Gaussian in rate, log = the years curve. gibbs.py:
lin_sc(b) scales each group's rows (1, 1/sqrt 2, PALEO_EPS), lin_cut(b) the swap pairs' inner products; slip/MFD log via
T.lw W (x b; slip sided WL), paleo via PM tables F, D, K x b (V0 uses ln(sd / scale)); SM/MF/PM dicts carry b and eps;
recording: paleo_misfit = 2 sum f / b (the whole curve), mfd_misfit, slip_log (whole), misfit += b (log - Gaussian) per group;
tr slip_geo_b, mfd_log_b, mfd_log_expect, paleo_b. Fit check: geo_moments(P, rows, b, kind) (density on a log10 grid
exp(-m1/2) x^(1-b)), paleo_misfit.site_moments_mixed (b < 1). --all-positive (the notebook always): slip_band and
paleo_band positive unless a factor, MFD bins positive (g_mfd in set_groups). SLIP_BAND removed from the notebook (2026-11-10, the user:
'take out SLIP_BAND'; --slip-band/--paleo-band stay in gibbs.py); SLIP_BAND_SOFT removed from the notebook too (the user: the log10 option does what it was
trying to do; --slip-log-band stays in gibbs.py). Column passes (2026-11-09, the user's request, speed only): gibbs.column_passes(T, B) sets T.pass_end (the kernel's PF: each
column's likelihood passes end before the constraint rows, n_fit on; band_bounds and the residual updates still use the whole
column) and T.ss_fold (the --slip-sum rows, n_bal..n_fit, folded: B G must equal them to 1e-12 of their largest entry, else
not folded; the kernel keeps U = B^T r_ss per chain (K x N, recomputed at each gibbs_run call), sums G_i.U in col_split's pass
(on every branch since 2026-11-17, 83dbf00; col_udots only on the non-split path), adds 2 m G_i.U + m^2 NSS_i to DQ, dt, sp_change and sp_lik, and updates U on
changes); main builds B from slip_sum_rows' info (scale x W x proj x sig_i / (row_scale x lin_sc)). The NumPy backend reads
whole columns. Bit-identical in fixed-seed tests at 211k (192 and 1,600 sweeps), 20-25% less time per sweep; selftest_fold
(C skip, C fold, NumPy) in both sets. Self-tests: selftest_paleo(b=0.5), selftest_mfdlog(b=0.5), selftest_slipgeo(b=1.0)
in the full set. run_quick_checks pins MISFIT_SLIP, MISFIT_MFD = 'linear', MISFIT_PALEO = 'rate' (positivity always) (TESTING.md's values predate positivity).
MISFIT_SLIP 'exact' (2026-11-23, the user's request; gibbs --slip-misfit exact; REPORT "Step 3b"): each slip row scored by its
fault's exact corrected density (timeframe_sliprates.pdf_table -> data/cfm_timeframe_sliprates_pdfs.npz -> cleaning's slip_pdf_*
-> cru_common.slip_pdf_tables), kappa g(Y) with kappa = L2 alpha^2 s_lo^2 (the split normal's identity); SLIP_SUM rows by the
parents' densities convolved (slip_sum_pdf_tables), unfolded. Kernel: rows with L2 < 0 read table -L2-1 (sp_w2 -> spt_f; tables
set by sp_tables from _sp_ptrs before every call); lb_floor with tables uses per-row quasiconvexity (not concavity). Tables
checked quasiconvex (quasiconvex_table, Fritsch-Carlson); each fault's -2 ln p made unimodal first (timeframe_sliprates.unimodal,
PAVA each side: the simulated factor's sampling-noise ripples, up to 0.55 for the zero-width faults). best_fit_as_fitted polishes 'exact' as 'split'. Self-tests
selftest_exact, selftest_sum_pdf.
Speed work of 2026-11-24 (the user agreed items #1 and #2 of the speed scoping; REPORT "The fractions on integrated out,
tested" and "Off ruptures decided from bounds"): run_plan MARGIN 0.0, CHAINS_PER_THREAD 2, RHAT_SWEEPS 90 (sweeps >= 90 tau of
the slowest fraction on, R-hat predicted 1 + 0.9 tau/n). Tables interleaved (SPT_FD: value and step-scaled slope side by
side; bit-identical). --off-bounds (default; --no-off-bounds): gibbs_run keeps DS (each chain's rows' slopes, sp_d/spt_d,
updated with WC), col_sdots gives sum G DS (and G.U), SPB_QI[i] = sum G^2 C / 2 (_off_bounds_qi, over the passes' entries,
set by sp_bounds from _sp_ptrs); the decision block 'bq' settles ~82% at 630k, sp_change for the rest; bit-identical to
--no-off-bounds (checked at 630k, 8 chains); 15% faster. --fractions-on collapsed (not the default: at 211k no clear tau
gain for the slowest bin; see REPORT): ss_collapse(on, a, b, nb) global set per sample() and reset after; CNT kept by
gibbs_run (recount at start, updated at x[i] = xn) and swap_run (new last args CNT, frac); ss_on_lp with LODk NULL;
he = thin when collapsed (unless --mfd-gr-tau's joint step); update_pi only at recording; NumPy twin _ss_collapsed_lod.
selftest_collapse uses pairs (0,2),(1,2) only: with all four pairs an always-accepted swap pass maps (2,0) to (0,2) and the
test cannot see a missing count ratio. --swap-fraction F (default 1): geometric skips through the pair list per pass.
Float32 tables tried and dropped (no gain).
Swap threshold below 0.99, tested 2026-11-24 and shelved (the user's choice; SWAP_COS was a notebook knob for part of the
day, then removed): preliminaries.py --dup-cos C (default 0.99; pairs kept down to C, groups still reported at 0.99/0.999/
0.9999; prelim dup_cos_min) and gibbs.py's check that --swap-cos/--swap-cos-cross are not below cleaning's dup_cos_min stay for
the command line; the summary reads swap_cos from the config (default 0.99). At 211k: 9.98M pairs at 0.95 against 1.90M at
0.99 (prelim 248 MB against 54 MB, cleaning 258 s); with every pair proposed the swap pass swamped the run (93 s a sweep at
first: on ruptures sit in clusters of up to 13,377). With --swap-fraction 0.19 (equal proposals per pass): bin 11's tau 712
against 360 and 900 at 0.99 (two seeds), median bin tau 53 against 74 and 67, worst R-hat 1.091 against 1.082 and 1.085: no
gain for the slow bins.
A swap pass visiting only the pairs with a rupture on (a heap through each rupture's pairs; bit-identical) was tried and
removed: at 630k the on ruptures' pairs are most of the 13.6M, and the heap cost 1.058 s/sweep against the full pass's 0.387.
swap_run skips the columns' dot products (ci, cj) under SL2 (split or exact misfit): only the Gaussian formula uses them.
lw_change and lw_swap (the curve terms' and the log band's exact changes) iterate the per-column weighted-row list
(LWP/LWQ, built for col_lwsums) when it exists, in place of every entry of the column(s); lw_swap merges the two columns'
lists. The same rows in the same order: bit-identical at 630k; no measurable speed-up (the curve terms' own work, logs and
table lookups, dominates; the 10% `sample` gave lw_change is that work, not the scan) (2026-11-24, the user's request).
Per-decision screening tried twice and dropped (2026-11-24): a bound from per-chain block maxima of |slope| (64 row
blocks, each column's sum of |G| per block, ~8 blocks a column) settles 66-76% of off decisions without the column pass
and skips 44-58% of group passes, bit-identical; but at 630k ('between', 8 chains, 4 threads) both versions were no faster
or slower (screen+g-test: 0.249/0.254 vs 0.242/0.248; lean, X test only, early exit at the top of the chain's update:
0.256/0.247 vs 0.236/0.238 s/sweep). A noinline profile (-g, helpers un-inlined, 18% slower) split the kernel's compute:
gibbs_run's own per-rupture work 29%, table lookups 23% (83% of them in the off-rupture fallbacks), the column pass 20%,
swaps 5%, slab draws ~5%, RNG 3%, logs 3%, curve terms ~6%. Thinning without a slab draw (geometric skipping) is not
viable: a bound over the whole slab up to the bin's top is near 1. Steady-state s/sweep: scratchpad bench/steady.py
(n2 r2 - n1 r1)/(n2 - n1) from the progress lines' 3-decimal rates; identical runs vary about +-2-3%.
Off-bounds fallback with early stopping (2026-11-24, the user's go-ahead after scoping): sp_bounds(QI, C_row, HO);
_off_bounds_qi also builds C_row (each row's C_j, inflated as QI) and HO (int32, each column's likelihood entries in order of
G_ji^2 C_j, largest first; ~15 s and 4 bytes per nonzero at 630k, built in sample() before t0). In the bq branch, when the
bounds leave the decision open, the exact change is summed heaviest rows first as sp_w2(r + G m) - WC (one lookup a row), the
rest bounded by their slopes and C (sl = SQV - processed, cv = QI - processed), checked after 1, 2, 4, 8, ... rows, stopping
once settled. Scoping at 630k ('between'): 18% of off decisions fell back (64 entries a column), 68% of them ending off, 58%
above the threshold by >10, the margin m^2 QI >10 in 91%; exact top-K rows settled 4/9/16/28/45% at K = 1/2/4/8/16; local
curvature alone 27% (not built: a table scan per row). Bit-identical to before at 630k ('all'); 0.33 -> 0.28 s per sweep.
Thread race fixed 2026-11-24: sweeps_c/swaps_c built the shared pointers (_pm_ptr's struct, _sp_ptrs' interleaved table FD and
off-bounds QI, _st_ptr) inside each thread's call, so on a run's first call two threads could each build a cache and the first
copy be freed while its kernel read it; identical 4-thread runs differed from their first draw. Now built once before the
threads start (shared, stp). Earlier 4-thread bit-identity checks (bounds on/off) may have passed by luck; rechecked after.
MISFIT_SLIP, MISFIT_MFD, MISFIT_PALEO (the user's request after run _1794201333): MISFIT split into one setting per group,
slip and MFD 'linear' | 'log10' | 'half' (default), paleo 'rate' | 'years' | 'half' (the user's names), each group's mode
from cru_common.GROUP_MISFIT_MODES (the notebook's copy _MISFIT_GROUP_MODES); MISFIT kept as their common setting as one
MISFIT had it ('half'; 'linear' = linear/linear/rate; 'log10' = log10/log10/years) or 'mixed' (CONFIG misfit, misfit_slip, misfit_mfd, misfit_paleo; cru_common.misfit_settings, also inferring them
from older runs' modes); FIGURE_SETTINGS lists the three (22 settings, SETTINGS_PER_LINE = 8, still 3 lines); summary PDF:
three rows and a per-group paragraph when mixed; RUN_NAME _misfit{MISFIT}, or _misfit_slip{S}_mfd{M}_paleo{P} when mixed.
preliminaries.py --cfm-sliprates CSV --rupture-set DIR (CFM_SLIPRATES, True by default since 2026-11-05, False before, always since 2026-11-09; the notebook passes cru_common.CFM_CSV):
each raw slip-rate row (row j = FaultID j) rebuilt from its parent fault's CFM line (by name; all 557 match at 211k) as a
split normal in mm/yr (cfm_sliprates.rebuild): mode f SR_pref with f the inputs' own factor (0.8, target x sigma /
SlipRate, checked constant; sigma from the row's coefficients against average_slips.csv), side sds --cfm-factor x f x
(SR_pref - SR_min or SR_max - SR_pref) / --cfm-sigmas: f, the assumed fraction of the slip rate the ruptures take up,
scales the whole distribution (the user, 2026-11-04: at f = 0.01 an sd of several mm/yr would be absurd), where the
inputs scale only their targets. Defaults 2 and 2 since 2026-11-05 (the reading the user agreed with colleagues):
SR_min/SR_max at 2 sd and doubled, so each side's sd is f x its width and sigma_ref a median 0.40 of the inputs' (short
run: slip chi2 698 against 507); the same sds as the 1-sd reading with SLIP_SD_MULTIPLIER = 1 (the user's run
_1793831887, _sdx1). Before that, 1 and 2 (the inputs' reading; sigma_ref 0.80 of theirs, 5-95% 0.75-0.90); f on the
mode only was tried and dropped. The user is checking the
reading with colleagues. --cfm-template-rho RHO (CFM_TEMPLATE_RHO, 1 by default since 2026-11-09, None before; the user's choice):
faults of QualCode 2-5 sharing an identical SR_pref / SR_min / SR_max as written (cfm_sliprates.template_factors) weighted
1/sqrt(1 + (n - 1) rho) for a group of n, multiplied into row_scale with the sections' 1/sqrt(k) (so plate_rate's sigma_i
and the figures' units stay each fault's own), recorded as the problem's template_scale and cfm_template_rho; the fit check
(chi2_expectation) and block_measurements count each section t^2. --cfm-rate-links (CFM_RATE_LINKS, True by default, only
with CFM_TEMPLATE_RHO; the user asked for it 2026-11-05 to catch borrowed rates of any QualCode: Lyell South "SR from
Lyell", Parihaka from Turi, the Alpine fan-offset sections, Whangamoa, Hyde South 0.5x, Kelly 0.75x, Wellington HV 1/2
from 3, Ariel Bank 2, George to Jacksons / Caswell - South George, Kowai - View Hill, Teviot, Clear Hills - Waikopiro):
cfm_sliprates.rate_links (comment names after copy words, the fault-system and family-name fallbacks; identical
triples within LINK_KM = 5 km or on one family name), union-find clusters; weight 1/sqrt(1 + m rho), m = template mates
plus cluster mates, not a merged partition (a merge chained 0.2/0.1/0.3 x15 with 2/1/3 x9 through Clear Hills);
links listed in cfm_rate_links.csv; recorded as cfm_rate_links. Since 2026-11-18 data/cfm_rate_fixes.csv (Name, same_as, reason; Teviot = Old Man) is applied wherever the CFM is read (cfm_sliprates.rate_fixes/_apply_fixes in load_cfm, load_cfm_steps, load_timeframe; template_factors' triples); the CFM file untouched; rebuild reports the fixes. Since 2026-11-18 data/cfm_rate_sources.csv (63 faults: own sites 43, own measurement 14, borrowed 6, with source and evidence; compiled by hand from the NZPSDB, CFM comments and the 2014 AFM descriptions for the 630k template members) is passed as --cfm-rate-sources whenever CFM_TEMPLATE_RHO is set: those faults leave the template groups (coincidental identical triples), borrowed ones get a 'listed' link to their source (cfm_sliprates.load_rate_sources, template_factors(sources=)); 630k templates 299 faults/74 groups -> 224/57. Since 2026-11-05 (the user's choice, option A over
section-level correlation with cross-fault sections) slip sections never span two parent faults (cru_common.slip_sections
parents=, from --rupture-set or find_rupture_set beside --A; recorded as sections_by_fault): 211k 507 -> 551 sections,
2e4 68 -> 75; template weights applied per section by cfm_sliprates.section_factors (equal to the faults' weights when
sections are single faults; template_scale saves the weights as applied). CFM_PREF, CFM_ROUNDING and SLIP_SD_MULTIPLIER are out of the notebook since 2026-11-09 (the user's request): SR_pref the
mode (also without CFM_TIMEFRAME), no rounding, notebook SD_MULTIPLIER_ = 1 with CFM_SLIPRATES and, since
2026-11-23, without it too (2.5 before, the 19k/211k inputs' doubling; SLIP_AS_GIVEN removed: the factor and sds always applied;
the user: no factor 2 in the setup code); CONFIG still records cfm_pref, cfm_rounding, slip_sd_multiplier; not in
FIGURE_SETTINGS (19, SETTINGS_PER_LINE 7); the summary PDF lists them only if an older run's differ; preliminaries.py keeps the
options. CFM_SLIPRATES and CFM_FILE are out of the settings too (2026-11-09, the user's request): CFM_SLIPRATES = True below
the settings (False only for sensitivity tests, run_nb.py CFM_SLIPRATES=False, as run_quick_checks pins it), the CSV in
the repository, data/NZ_CFM_v1_0.csv = cru_common.CFM_CSV (timeframe_sliprates.py --cfm's default); not in FIGURE_SETTINGS
(18; 'no CFM' under the CFM columns for a run without it). Also fixed below the settings since 2026-11-09 (the user's request):
BLOCKS None, SLIP_SECTIONS 'auto' (RUN_NAME starts SLIP_SECTIONS... only when not auto), RUPTURE_SET 'auto', CROSS_SECTIONS
'all', CFM_RATE_LINKS = CFM_TEMPLATE_RHO is not None; FIGURE_SETTINGS 14 (SETTINGS_PER_LINE 5), a non-default SLIP_SECTIONS or
CROSS_SECTIONS shown beside PRESET. DOWN_THE_MIDDLE and MFD_MONOTONIC stay settings. Misfit defaults (2026-11-09, the user's
choice, each group in the space its uncertainties are worked out in): MISFIT_SLIP 'linear', MISFIT_MFD 'half', MISFIT_PALEO
'years' (MISFIT 'mixed'). --cfm-rounding (formerly CFM_ROUNDING, True by default 2026-11-09, False before): each fault's three numbers known to half the
finest step among them as written (cfm_sliprates.load_cfm_steps, the shortest form of each, since neither the CSV nor
the spreadsheet keeps trailing zeros), variance q^2 (f^2 + 2 c^2)/12 added to each side's sd in quadrature, before the
median shift; recorded as the fifth entry of the problem's cfm_sliprates. --cfm-pref mode|median (CFM_PREF, 'mode' by default; passed only when not mode): with median,
each split normal shifted, sds kept, so that its median is f x SR_pref (rebuild's pref_as; the shift from the side masses
a = s_lo/(s_lo+s_hi): median - mode = s_hi Phi^-1(0.5 + (0.5-a)/(2(1-a))) for a < 0.5, -s_lo Phi^-1(1 - 1/(4a)) for
a > 0.5); faults whose mode would fall at or below 0 keep the mode (18 at 211k), listed in the cleaning report; recorded
as the fourth entry of the problem's cfm_sliprates. [Since 2026-11-23 --slip-sd-multiplier, rescale_inputs and the run without
the CFM are gone: each side's sd is F x the fitted split normal's, K = 1 always; the rest of this paragraph is history.]
--slip-rate-factor F and --slip-sd-multiplier K (SLIP_RATE_FACTOR 0.8 and SLIP_SD_MULTIPLIER
1 by default since 2026-11-09 with CFM_TIMEFRAME (3 earlier that day, the mean widening of Coffey et al.; 2 before); the scripts' own default without the option is 1 since 2026-11-23 (2 before); the notebook always passes both, with --rupture-set; --slip-sd-
multiplier replaced --cfm-factor): every slip-rate distribution with F on the whole of it, target F x rate and sd
K x F x its uncertainty: the CFM's sides (rebuild's srf and factor) or, without --cfm-sliprates, the inputs' Gaussians
(cfm_sliprates.rescale_inputs: sd K x F x SD; their own convention, checked, is target 0.8 x SlipRate, sd 2 x SD, the
factor on the targets only, an implementation error upstream in the NZSHM22 code as the user reads it, 2026-11-05; the
8 subsections with an SD of 0 keep sd = the target). So since 2026-11-05 the default runs' slip sds are 0.8 of the
inputs' (coefficients and targets x 1.25 in sd units); K = 2.5 with F = 0.8 reproduces the inputs bit for bit (every run
before then; run_quick_checks.sh pins it, for TESTING.md's values). K = 1 switches the doubling off, for slip-rate
uncertainties estimated from the data. Recorded as the problem's slip_rate_factor and slip_sd_multiplier. Why 0.8 (REPORT.md, "Why 0.8"): f = 1/(1 + u), u = afterslip (Churchill et al. 2022: median 18%, IQR 9-32%,
lognormal) + smaller earthquakes (GR, Mmax 8.6, b uniform 0.5-1: 1.6-14.4%, median 4.7%); f median 0.81, 5/50/95% 0.52/0.81/0.93;
planned: a three-branch logic tree at those values, weights 0.185/0.63/0.185 (not built; sampling f set aside). Without either
flag preliminaries.py leaves the inputs as given. A zero-width side takes the other's and
min = pref = max gives sd = the mode (as the inputs: sigma = 0.8 x SlipRate there);
the row rescaled to sigma_ref = the mean of the two sides, and the weights split_lo2 = (sigma_ref/sigma_lo)^2 and
split_hi2 = (sigma_ref/sigma_hi)^2 saved per cleaned row (load_problem; None without it; cru_common.split_weights).
--cfm-timeframe CSV (CFM_TIMEFRAME, True by default since 2026-11-09, the user's choice; with --cfm-sliprates): each
fault's SR_pref, SR_min and SR_max replaced by the mode, 2.5% and 97.5% points of its slip rate corrected for its timeframe
(Coffey et al., accepted BSSA, papers in Inversions/Papers/GC_sliprates), the rest unchanged (split normal, ends at 2 sd,
SLIP_RATE_FACTOR on the whole, template/link weights from the CFM's numbers as written, kept in place of the paper's source
correction, its Table S3, by the user's choice); the notebook then sets SLIP_SD_MULTIPLIER 1 and CFM_ROUNDING False (and
needs CFM_PREF 'mode'); recorded as the sixth entry of the problem's cfm_sliprates. The table, data/cfm_timeframe_sliprates.csv,
is tools/timeframe_sliprates.py's (--cfm, --s4 the paper's Table S4, --rupture-set; since 2026-11-09 its sn_centre, sn_sd_lo,
sn_sd_hi, sn_fit columns hold each fault's split normal, split_normal_for: of the candidates exact through 16/50/84 in slip rate,
16/50/84 in log10 converted and 2.5/50/97.5 in log10 converted, the closest in KL (cross_entropy, exact quadrature), else the old
mode/2.5/97.5 'ends'; preliminaries.py --cfm-timeframe-shape fit (default) | ends;
recorded as the seventh entry of the problem's cfm_sliprates and CONFIG cfm_timeframe_shape; REPORT.md 'Split normals fitted'): S4 reports bounds never narrower than
the CFM's (lower min(q, SR_min), upper max(q, SR_max): its 25/75% columns mostly the CFM's bounds, 2.5% pinned in 194 rows),
so each fault's distribution is rebuilt exactly (quadrature over the factor's quantile function, no sampling; class
Corrected): the paper's Gaussian on the CFM bounds (centre their midpoint, sd width / 3.92, negative part at 0) times the
timeframe factor from its Tables 1-2 (interpolated in log t; across probability against the normal score, both linearly and
by PCHIP, two schemes for redundancy), the fault's effective scaled timeframe t = timeframe x 1000 / RI solved per scheme
so its 97.5% point matches S4's (483 solved, 74 at the tables' 100,000 years; faults not in S4, 3 of the 211k rupture set's:
Ararata, St Stephens, Takapu, and 320 other CFM faults, their kind's median t); checks not in the fit: S4's medians matched
to 0.986-1.013, unpinned 2.5% points 1.000 (5-95% 0.97-1.39); the mode (the density's maximum, the maximum-likelihood slip
rate after the correction) and the quantiles the two schemes' geometric means (their modes differ by a median 0.12%; over 2%
for Wairau 9.3%, Wellington: Pahiatua, Otaheke North, Clarence: Northeast, Jordan, Awatere: Southwest; both kept as
mode_linear, mode_pchip). A kernel-density mode from draws, tried first, was off by up to 25% (smoothing a skewed density).
Since 2026-11-23 (the user's request) the factor is log_factor 'sim': tools/styron_factor.py simulates the paper's setup
(data/styron_factor.npz; windows uniform in time, measured between their bounding earthquakes, the rule that reproduces
Tables 1-2: rms log 0.04 / 0.11), pinned to the tables' 7 quantiles exactly (S4 is the paper's own draws: the simulation
alone matched S4's medians only to a median 1.019, 95th pct 1.126) with the simulation's shape between and beyond them.
One scheme; 191 faults held at 100,000 yr (74 before); S4 medians 0.986-1.010. Numbers above are from before.
The paleo sites' recurrence intervals reproduce S4 for the faults where the paper
used them (Alpine, Hope: Conway, Awatere) but not for a third of the 42, so t is fitted, not taken from them. The paper's
2.9 (abstract) is the mean of S4's 95% width / the CFM's over SR_pref < 1 mm/yr (2.86 here); the median is 1.6.
plate_rate.cross_sections reads each subsection's target and sigma from its row, so --slip-sum sums the modes with
sigma_ref unchanged in code. gibbs.py reads the weights from the problem (T.split; spike-and-slab only, no transfers):
the misfit sum w(e) e^2, w = split_lo2 below the target and split_hi2 above, concave in m, so the on-chain slice uses
the exact likelihood (lbctx's sp fields, sp_lik; lb_floor's endpoint bound still holds), off chains draw their slab
value first and get their exact change in one pass over the column (col_split, with the soft band's bound sums), the
on/off decisions use sp_change, the swaps sp_swap; each chain's rows' current terms sp_w2(r_j) are cached (WC, refilled at
the start of each gibbs_run call and updated as rows change), so a column pass computes only the new ones, bit-identical;
since 2026-11-17 each full group of 4 chains also keeps an interleaved copy RW (rw4: each row's r, WC and U for the 4
chains side by side, 96 bytes; SL2/SH2 as pairs, SLH), filled at each gibbs_run call and written wherever R, WC or U change,
which col_split4i reads in place of col_split4's no-bound-sums branches (about 2 cache lines an entry instead of 14; the
compiler vectorizes it across chains; each chain's sums in the same order, so bit-identical; groups of fewer than 4 chains
and the soft band's fused branch keep the old pass); recorded chi2 is the split misfit; NumPy twins; selftest_split (both
backends, with and without the soft band, against a grid integral) in the full set; block_fit.py and plot_slip_rates.py
weigh residuals by side; without it every run is bit-identical (REPORT.md, "Slip rates from the CFM").
--band F (the notebook's BAND = 10 until November 4th, 2026; its default for a few days in early November 2026, when DOWN_THE_MIDDLE = False and
FLOOR = -10 came with it, until BAND = 'positive' and PRESET = 'conservative' became the defaults on 2026-11-02): every data row with a positive target (each subsection's slip-rate row, each MFD bin, each paleo row)
keeps its prediction within d/F to F d, a hard constraint for any prior (Target.set_band; the kernel's band_bounds
restricts each update, under spike-and-slab a forbidden off or on state is never taken, swaps that would leave the band
are rejected; selftest_band in the full self-test set; without it every run is bit-identical). A start outside it is
moved in by band_start (penalized least squares, each rate in units of 1/||G_i||, stopped once inside
F^0.95; about 1.5 minutes at 211k), kept beside the problem as
<problem>_band<F>_start.npz and reused when the problem, band, start and bounds match (band_start_key). The best fit
and randomized_nnls.py ignore it (REPORT.md, "The band"). --log-band TAU (BAND_SOFT, off by default; with
--positive): a soft band, 1/2 (log10(y / d) / TAU)^2 on each positive-target row (slip subsections 1/k), kept exactly by
Metropolis steps in the kernel (lw_change in the on/off log odds and on slab moves and swaps; for off ruptures, bounds
linear and quadratic in the proposed rate from per-row coefficients kept per chain, summed in col_dots_lw's pass with the
dot products, settle nearly all decisions, most without logs, the same draws), T.lw = dict(W, LD, DY, rows, tau); when the
weighted rows (LW > 0 or a paleo table) hold under half the column entries (paleo years or MFD log10 without a slip soft
band), the bound sums run over those rows' entries only (col_lwsums, from _lw_sub_ptrs's per-column positions, LWP/LWQ),
and so do the coefficient refreshes: the same sums in the same order, bit-identical; selftest_logband in
the full set; NULL arrays without it, so runs are bit-identical. --positive (BAND = 'positive'), an alternative,
spike-and-slab only: every such row only kept above zero (Target.set_positive: a lower limit of half the smallest
one-rupture amount, no upper limit, through the same rlo/rhi; positive_start switches on, at twice the slab floor, a
rupture in each row the start predicts at 0; its exact test is in selftest_band). --mfd-monotonic (the notebook's MFD_MONOTONIC, on by default from early November 2026; command line
only from November 4th; back in the notebook as before, True by default, from 2026-11-10 at the user's request; False by default since 2026-11-09, the user: it held
the bins against their limits and spoiled the MFD fit in the conservative run _1794181008): no bin's total above its lower neighbour's where the target is not (and,
since November 3rd, 2026, where it rises, the rise capped at rho = (R_low + (R_up - R_low) + --mfd-rise-sd sd) / R_low
times the lower total; --mfd-rise-sd, 0.89 by default, command line only), through constraint rows after
the likelihood's (T.n_fit; eps 1e-9 coefficients, Target.add_limits), with or without the band; exact test in
selftest_band. --cross-sections all|between (CROSS_SECTIONS, 'all' by default; a notebook setting again since 2026-11-24,
the user's request, beside SLIP_SUM_L0, out of run_config's FIXED, in FIGURE_SETTINGS and run_plan's MODEL_KEYS): the strips --slip-sum uses; since
2026-11-24 (the user's request) --plate-rate's cap always applies to every strip ('all'), in gibbs.py, best_fit_as_fitted.py
and plate_rate.py's report alike (config plate_rate_strips 'all'; RUN_NAME _platemax{f}all; older runs had one set of strips for both) (plate_rate.py's cross_sections: cross-sections are great circles through the Beavan et al. 2002
pole; strips the mean subsection trace length wide, 8.5 km, 72 between END_SW, where the Alpine Fault goes offshore,
and END_NE, the Needles / Wairarapa: Needles step-over in Cook Strait; 'all': 166 over all of New Zealand).
--plate-rate HI or LO,HI --rupture-set DIR (the notebook's PLATE_RATE: until November 4th, 2026 None | HI | (LO, HI), off by
default; command line only from then; since 2026-11-10 back as a boolean, True by default, the user's choice: an upper
bound only, at SLIP_RATE_FACTOR, PLATE_LIMITS = f'{SLIP_RATE_FACTOR:g}', also passed to plate_rate.py --limits; config
records plate_rate = the factor or None; before: 0.8 if imposed, an upper limit only: the
whole plate rate at the inputs' 0.8 of the fault model's slip rates; a lower limit of 0 is not implied beyond the
edges, where some faults slip against the plate motion): each strip's share of the Pacific-Australia plate rate
within the limits, one constraint row per strip holding a rupture after the monotonic ones (plate_rate.py's
plate_rows: each crustal subsection weighted by the fraction of the strip's cross-sections crossing it, its slip
vector projected on the local plate motion over the local rate; checked against the slip-rate rows; a start above HI
scaled down to it, each chain by one factor); exact test in selftest_band (until November 3rd, 2026: 8 strips of
76 km between the edges, --plate-strips, LO,HI only). --slip-sum --rupture-set DIR (SLIP_SUM, True by default from November 3rd, 2026, False from November 5th, when the user wanted
to see whether every target can be well fitted without it, from the new constraints and positivity alone, True again since
November 9th, the user's choice, with SLIP_SUM_L0 = 100): a soft term
in the likelihood, rows after the balance term's (T.n_bal) and before n_fit, recorded apart as tr['slip_sum']
(plate_rate.py's slip_sum_rows: per strip, (width / L0) ((sum_i w_ki c_i (model_i - target_i)) / sigma_k)^2, w_ki the
fraction of subsection i's extent along the plate motion in strip k, so each subsection's weights add to 1; c_i its
slip vector per unit slip projected on the local plate motion, signed (since November 3rd, 2026; before, c_i = 1); sigma_k
with each parent section's subsections fully correlated; on the --cross-sections strips; L0 =
--slip-sum-l0 (SLIP_SUM_L0, 100 km in the notebook since 2026-11-09, the user's choice; 8.5 km, the scripts' default and the notebook's before, the strips' width: one extra observation per strip; 120 km would be the
unprojected target sums' decorrelation length between the edges, about the span over its 5 Alpine-Wairau sections, 122 km); no whitening between strips, so the
strip count only sets the integral's accuracy; the swap pairs' inner products include its rows, checked on a sample at
start). Its bookkeeping test is in selftest_balance. plate_rate.py reports both (plate_rate.txt, a notebook cell for
every run with a rupture set).
Removed on 2026-11-23 (the user's request): binbox-theta (per-bin boxes n_j / (theta ln 10) decades deep; the
alternative until October 2026), binbox-floor and binbox (per-bin boxes; the default until October 2026, when it failed
the fit check by 26 sd), and the single box (logbox, logbox-centre, logbox-mean; cru_common.resolve_logbox and
logbox_mean, whose average_data_rate was undefined). The kernel's log-box machinery (Target(logbox=...), pkind 1) stays:
spike-and-slab's slab runs on it. --prior lognormal (the hierarchical prior) is retired: it never converged (see
REPORT.md). Transfer moves (--transfers) now run only under the flat and exponential priors and are set aside.

## Working agreements
- Every gibbs.py run writes <out>_stream.npz (running summaries); --no-samples skips the large draws file, and the
  notebook always passes it and uses the running summaries for its checks and tables. make_tables.py
  (run by the notebook) always writes rate_vectors.csv/.npz (mean, median, mode, 2.275/15.865/84.135/97.725% per rupture, original column indexing),
  from the running summaries' 0.01-decade log-rate histogram (gibbs.py --hist-width), or for older runs from their
  stored draws, binned the same way. With REPRESENTATIVES > 0 (100 by default) each run also keeps 1,000 whole draws
  (gibbs.py --keep-draws: <out>_kept.npy and _kept_index.npy; recording only, the chains are unchanged) and
  select_draws.py writes weighted representatives as columns rep_01... of rate_vectors.csv, with representatives.csv
  and select_report.txt; the mean column stays the source for mean hazard. The notebook also writes
  spikeslab/workflow_log_<date>-<time>.txt, a copy of
  everything it prints, with each cell's start time and duration and any traceback (cru_common.start_log).
- Each step's code lives in a script, and each script keeps what it prints in a report next to its outputs
  (cru_common.report_to): prelim_summary.txt, <run>_log.txt, convergence_report.txt, block_fit_report.txt,
  tables_report.txt, select_report.txt and each figure's .txt. Keep it that way: a new step is a script with its own
  report, and the notebook only sets the configuration and calls it.
- Magnitudes: a rupture's magnitude is known only to its bin row. plot_rate_magnitude.py counts the bin rows up from
  6.80-6.90 in 0.1 steps in their original row order (--lowest-bin, --bin-width; in the corrected 2e4 inputs the 16
  rows just below them, 793-808, are the penalty and empty rows, and appear to be the bins 5.20-6.80) and checks that
  order against rupture sizes.
- Slip-rate sections: a fault section's slip rate is one measurement, which the inputs copy (with its uncertainty) to
  each of its k subsections. preliminaries.py --slip-sections auto (the notebook's SLIP_SECTIONS = 'auto', the default)
  divides the rows of each run of copies (identical targets, and identical coefficients for the ruptures both rows
  contain) by sqrt(k), so the measurement counts once; rows unlike their neighbours (a geodetic slip-rate model, say) are
  unchanged. 'none' keeps every subsection as its own measurement (the inputs as given; a sensitivity test, in its own
  RUN_DIR); a file of 'original_row section' lines sets the sections. The cleaned problem records row_scale, section,
  slip_sections and chi2_expect, the fit check (the chi2 a draw should have if the uncertainties are right: 126 +- 16 for
  the corrected 2e4 inputs, 446 +- 80 with 'none'). How the slip rates count is part of the likelihood: a modelling
  decision.
- MFD uncertainty (the notebook's MFD_UNCERTAINTY; 'as-given' by default): cleaning (preliminaries.py
  --mfd-uncertainty) sets the magnitude bins' uncertainty level. 'as-given' keeps the inputs (10% at M7.0-7.1, the
  upstream design value, a holdover of UCERF3-style weights on the data types), each bin its own measurement. The bins
  are not independent observations but a projection of two measurements, the rate of M>=5 and the b-value (whose
  uncertainties the logic tree explores through alternative inputs): 'two-measurements' multiplies every bin's
  uncertainty by sqrt(bins / 2), 3 for 18 bins (its row divided by it), so that together they count as two
  measurements, as each slip-rate section counts once. A number sets the M7.0-7.1 bin's relative uncertainty (bins
  counted up from 6.80-6.90, as plot_rate_magnitude.py counts them). Only the level changes: the bin-to-bin shape
  (relative uncertainty proportional to R_j^-0.25) is set upstream, where cru_A/cru_D are built, and kept. The bin
  targets R_j are unchanged, and the soft top's width follows the uncertainty. The cleaned problem records
  mfd_uncertainty and mfd_scale, and row_scale includes it; the fit check still counts one per bin. The data cannot set
  this level (the bins' misfit is mostly the model's freedom), so it is a modelling decision, and another setting
  needs its own RUN_DIR (REPORT.md, "MFD uncertainty"). It replaced block reweighting from a pilot run (BLOCK_WEIGHTS,
  removed in October 2026).
- Fit-figure units (since 2026-11-05): with --ruptures (the notebook passes rupture_dir()), plot_slip_rates.py shows
  slip rates in mm/yr (each row's sd from its coefficients against average_slips.csv, plate_rate.check_problem; the
  target the slip-rate factor x the measured rate), each section's target with a faint bar of one sd either side (each
  side's own for the CFM), and plot_paleo_rates.py rates per year (each row's sd its site's from
  paleo_constraint_data.json), each target with bars of the site's own 15.9-84.1% and 2.3-97.7% ranges, in a row of a
  linear and a log-log panel per result set (the slip-rate figure always had both); chi2 and the within-one/two counts
  stay in sds; both scripts read the display unit u per row and divide by scale / u. Without --ruptures, the old units:
  the input rows are divided by their uncertainties, so targets and predictions in units of each section's measurement
  uncertainty (undoing the sqrt(k)), with the one-uncertainty band. plot_paleo_rates.py shows the paleo rows (every row in neither the slip-rate nor the
  bin group whose target is positive, unless --paleo-block names them) the same way, log-log: there the units only move a point along the 1:1
  line, so its distance from the line is the ratio of prediction to target.
- Code, data and outputs may be in separate folders. Scripts take input paths as arguments, find prelim_results.npz
  next to --problem (or via --prelim), and write where --out points; the notebook uses DATA_DIR (its files only read) and RUN_DIR,
  relative to its own folder (by default ~/inversions/NZSHM22_Matrices-630k/ and a new subfolder of it named after the
  settings and the start time, SLIP_SECTIONS..._floor..._<unix time>); the runs go into RUN_DIR/spikeslab/, with
  the shared cleaned problem and diagnostics at the top of RUN_DIR, the best fit's outputs in RUN_DIR/nnls/. One prior per notebook run (no comparison runs); a second
  setting of the same prior needs its own RUN_DIR (the default name gives it one). Never write into the data folder's
  own files (the inputs and the model's OpenSHA solution): only the notebook's RUN_DIRs go there.
- Before routine use, work through docs/TESTING.md. The slip-rate and magnitude-bin rows are detected
  automatically during cleaning (preliminaries.py), and any group named in --blocks (original row numbers, which
  change with the input) is checked against the detection. check_bin_rows and check_slip_rows
  (cru_common.py) must pass for any new input; compare the slip-rate profile they print with the 2e4 one.
- Read section 0 of docs/sampler-design.md and docs/REPORT.md before proposing changes to the approach.
- Keep the docs in step with the code: every change updates REPORT.md, TESTING.md, this file and the design notes
  where they describe it, with numbers stated with the conditions they were measured under. Before committing a code
  change, run the self-tests that apply, a quick end-to-end notebook run (tools/run_quick_checks.sh) and
  tools/static_checks.py. Run heavy jobs one at a time, and delete each test folder once its numbers are in the docs.
- Floors and cutoff: one setting, the notebook's FLOOR = -10 (1e-10; -12 in late October 2026), is spike-and-slab's
  slab floor and the diagnostic and reporting cutoff (the notebook passes --diag-floor={FLOOR} to gibbs.py,
  preliminaries.py, check_convergence.py and plot_rate_magnitude.py, whose defaults match; REPORT.md, "Floors and
  cutoff"); for theta it is only the cutoff. The per-bin boxes, if run, take it as their floor.
  preliminaries.py reports the collective negligibility scale; on new inputs (5e5 especially), check that FLOOR lies
  below it.
- The convergence verdict (PASS/CHECK, check_convergence.py) checks the targets in docs/TESTING.md. Ruptures with R-hat > 1.01 (or above
  the noise baseline, if higher: the slow-mixing minority under spike-and-slab) are reported separately, with their share of the total rate, and
  flagged in ruptures_posterior.csv (rhat_log_rate, ess_log_rate, slow_mixing).
- Run `python tools/gibbs.py --selftest` after any change to the sampler, and `python tools/block_fit.py --selftest` after any
  change to the row-block fit report or the problem helpers in cru_common.py (the MFD uncertainty among them); both
  must pass. Any sampler change must also keep the rate-by-rate agreement in compare_runs.py
  (z-scores ~ N(0,1), |z| > 3 near 0.27%). `--selftest current` (the notebook's RUN_SELFTESTS until 2026-11-17; now the command line only) runs the cases
  the notebook's default uses (per-rupture lower edges, the soft top, spike-and-slab, the balance term, swap
  moves, the band). A change meant only to speed things up must leave a fixed-seed run's output arrays
  bit-identical if it is in the sampler (gibbs.py's per-draw bookkeeping works on each chain's nonzero rates, and its
  kernel moves a thread's chains through the ruptures together, for instance). In cleaning (preliminaries.py), where a faster method can only sum in another order (the best fit, the
  near-duplicate search, the resolution diagonal), its results must be identical up to rounding (the cleaned problem
  itself exactly; the same near-duplicate pairs and groups; the best fit's chi2_min and predictions, though its rates
  may move between near-identical columns), and a fixed-seed quick check (tools/run_quick_checks.sh) must give
  byte-identical tables and reports, its run files differing by rounding alone.
- Keep G sparse (scipy CSC/CSR). Never create n×n objects or dense copies of G at the 5e5 scale.
- Log chi-squared, R-hat and ESS for every run; write draws to disk as a memmap,
  or with --no-samples keep only summaries and functionals (needed before the 5e5 configuration).
- Default model: spike-and-slab (--prior spikeslab; the notebook's only prior) under the Gaussian likelihood (the only
  one: the chi2 cap, --target cutoff, was removed in October 2026), with every slip-rate, paleo and MFD prediction above zero (--all-positive, always, since 2026-11-10)
  and without the balance term (DOWN_THE_MIDDLE = False); no other prior is run as a cross-check (the square-root slab, --slab sqrt, was
  dropped at the end of October 2026). Tests and smoke runs use the default prior only (the smoke test,
  tools/run_quick_checks.sh, pins FLOOR = -10, DOWN_THE_MIDDLE = True, no band on either group, PALEO_MISFIT = 'gaussian' and CFM_SLIPRATES = False, under which TESTING.md's quick-preset
  reference values were measured); the notebook runs it alone,
  always with swap moves (--swap-cos 0.99), from the best fit and without stored draws. Sharing the fraction on by magnitude
  bin and tectonic domain or slip-rate section, pooled or not, was tried in October 2026 and changed no fit (REPORT.md).
  A spike-and-slab run counts as converged only when the number of ruptures on and every bin's fraction on have
  R-hat < 1.01, as well as the rates. A linearly uniform prior (--prior linear, removed 2026-11-23) pushed the bin totals far up and failed
  the fit check by hundreds of sd at 2e4 and thousands at 211k; so did the halfway prior (removed with it). The prior, its floor,
  edges and hyperpriors, the likelihood (the balance term and its weight among it), the band and its factor, sigma, bounds, how the slip-rate rows
  count and the MFD bins' uncertainty level are modelling decisions: ask the user before changing them. If all-zero rows are kept (--keep-empty-rows), the fit check
  includes their constant.

Rupture sets without an inversion (2026-11-16, the user's request, for the 630k inputs, NZSHM22_Matrices-630k: cru_A/cru_D
beside an OpenSHA rupture-set folder, NZSHM22_RuptureSet-..., with no solution): RUPTURE_SET = 'auto' (cru_common.find_rupture_set)
takes the one folder in DATA_DIR with a ruptures/ (a rupture set or an inversion solution); the reference model's rates are drawn
only if its solution/rates.csv is there (ref_args()). A rupture set made without an inversion has no paleo_constraint_data.json:
paleo_misfit.paleo_json then uses data/paleo_constraint_data.json (PALEO_JSON, the 211k solution's, NZSHM22_C_42), after checking
every site's sectionIndex and sectionName against the rupture set's fault_sections.geojson (the 211k and 630k sets share all 2,325
subsections), and the paleo rows' targets against it as before. At 630k: cleaning 2,395 x 615,700, fit check 382 +- 26; 4 chains
0.124 s per sweep (8 chains about twice that), the conservative preset about 6 h per replicate, peak memory 8.2 GB.
MFD_STEP_SD (2026-11-17, the user's request; None (off) by default since the same day, the user's choice; 0.25 suggested; gibbs.py --mfd-step-sd B): a soft term against bin-to-bin
zig-zags, S = sum_p (eps_p - mean eps)^2 / tau^2, eps_p = log10 of the draw's step over the target's at pair p, tau = 0.1 B decades
(B in b-value units): the target's own bends (the partitioning's) cost nothing, level and tilt stay with the bins (no double counting
of b). gibbs.mstep_build (A, c of the quadratic form in lam = ln of each bin row's prediction; bx/gx each rupture's bin and
coefficient) and mstep_value; kernel stctx (st_fill, st_change, st_bounds, st_update): per-chain LAM/HH, in the on/off decision
(bounds for off ruptures), the on chain's slice step as a Metropolis proposal, and swap_run (one or two bins); ODD now allocated with
LW or ST. C only. Recorded as tr['mfd_step'], printed 'MFD steps (--mfd-step-sd B): ... near n_pairs-1'. best_fit_as_fitted.py
--mfd-step-sd (mfd_step_term, exact gradient; terms_key mfd_step=). Notebook: notebook_steps._terms passes it (gibbs_args, starts_args); RUN_NAME _mfdstep{B};
Out of the notebook since 2026-11-18 (MFD_GR_TAU, below). run_quick_checks pins MFD_GR_TAU=None. selftest_mfdstep at
B = 2 (tight B mixes slowly in its toy). REPORT.md "The MFD's steps".
DATATYPES_EQUAL (2026-11-18, the user's request, named GROUP_WEIGHTS ('equal') until the same day; False or True; gibbs.py --group-weights equal): w_g = sum N / (3 N_g) from
cru_common.block_measurements (keep the total, the user's choice), groups slip (SLIP_BLOCK), mfd (BIN_BLOCK), paleo (the
rest). gibbs.group_weights; applied through T.split's L2/H2 (created with ones if absent; T.temper per row, chi2 recorded
divided back), PM tables x w (paleo, MFD pdf; recorded misfits / w), log-half W/WL x w, mjoint Q x w (mjoint_value / w),
mstep A/c x w, slip-sum rows and folded B x sqrt(w_slip) (recorded / w). tr group_weights, group_names. Notebook after
MISFIT_PALEO; RUN_NAME _dtequal; FIGURE_SETTINGS, run_plan MODEL_KEYS. MFD_UNCERTAINTY moved to run_config.py FIXED
('as-given') the same day: out of FIGURE_SETTINGS (shown with the fixed settings when not as-given), RUN_NAME only when
not as-given, summary row only then.
--swap-cos-cross C (2026-11-19; notebook 0.99 since the same day, one threshold within and across bins, the user's choice; first
0.95 with cleaning --dup-cross-cos 0.95, no longer passed by the notebook: near_duplicates(low, keep_low) keeps
pairs in (C, 0.99] only across bins, columns' bins from the bin rows; prelim dup_cross_cos): pair selection keeps same-bin
pairs > swap_cos and cross-bin pairs > C (spike-and-slab with per-bin boxes only). swap_run gets SBIN, SB, LOD (per chain),
ZLO, KU, LLO/LHI/SC (kernel_consts) and zlo, zhi, gam; ss_on_lp (twin _ss_on_lp_py) = LOD + lb_prior - log(wp + wt), 0 off,
-inf outside the slab; the ratio is added only when SBIN differ, so same-bin swaps and runs without the flag are
bit-identical (checked at 19k). Swaps exchange scaled values (u = m / R_bin). selftest_crossswap (full set): two bins,
different floors/tops, vs a grid; without the ratio z 13-120 (and with unequal floors a no-ratio run can strand a rate below
its floor, where its slice update never returns).
selftest_groupweights (full set, ~40 s): ws 0.4 on a Gaussian row via temper x L2/H2, wp 3 on a paleo row via PM F/D/K x w,
vs a grid integral (sig 0.02, d0 0.07: the weights move the exact answers 0.07-0.24 from w = 1); recorded chi2 and
paleo_misfit unweighted; group_weights on a toy 3-block P (3/2/4 measurements -> each counts 3).
MFD_GR_TAU (2026-11-18, the user's request, in place of MFD_STEP_SD in the notebook; 0.025 by default, with MFD_GR only; gibbs.py --mfd-gr-tau TAU,
with --mfd-misfit pdf): the bins jointly about each draw's G-R curve. gibbs.mjoint_build (Q = D'D/tau^2 + 11'/(NB tau)^2, the residual's
steps and mean level; mstep_build for bx/gx), mjoint_update (each chain's b and L = ln(s m / m_mid) drawn exactly on a 481 x 801 grid,
prior x exp(-S/2); c = Q (ln dy + L + shape(b)) per chain), mjoint_value. Kernel stctx gained cs (c's stride per chain); st_fill takes k;
_st_ptr(T, k0) points each thread at its chains' c. The bins' PM curves are switched off (T.lw['PM']['I'][mf_rows] = -1; MF joint, E = 1,
V = 2); mfd_misfit = S (in chi2 as fitted); tr mfd_gr_b, mfd_gr_level (ln(s m)), mfd_gr_n5, mfd_gr_tau; report rows 'MFD curve ...'.
mfd_gr.level_logpdf (shared with bin_pdfs), joint_grid, b_logpdf; preliminaries.py --mfd-gr saves mfd_gr_bgrid and mfd_gr_shape
(cru_common.load_problem). Notebook: _terms passes it to gibbs.py only (the starts' polish keeps each bin's PDF; starts_args passes dispersed_starts.py --mfd-gr-tau, whose prior draws put the bins on a G-R curve from b_logpdf/level_logpdf x exp(tau random walk)); cru_common gr_curves, gr_joint_misfit, gr_run_draws (post-burn b, level from *_rep*.npz); block_fit takes the MFD block's chi2 from mfd_misfit (JNT, stats joint_tau, mfd_note); plot_rate_magnitude draws the curves' band and median; RUN_NAME _mfdtau{TAU};
FIGURE_SETTINGS and run_plan's MODEL_KEYS mfd_gr_tau (older runs None); summary PDF row, model text, results (b, s m). gibbs.py
--mfd-step-sd is kept for direct use (selftest_mfdstep). selftest_mfdjoint (full set) against a grid integral on the sampler's own
grids; b, level, N(M>=5) recorded wherever the term is on (tr mfd_gr_*). check_convergence: ess_mfd_curve, rhat_mfd_curve in the
row and the verdict (the total rate's targets). REPORT.md "The bins jointly about a G-R curve".
MFD_GR (2026-11-17, the user's request; False by default, cru_D's bins as before): each MFD bin scored by its own target PDF.
preliminaries.py --mfd-gr B10,B,B90,N10,N,N90,S_LO,S_HI --rupture-set DIR (MFD_GR_SPEC from MFD_GR_B = (0.823, 0.959, 1.089),
MFD_GR_N5 = (2.65, 3.40, 4.57), MFD_GR_NSCALE = (0.66, 1.41); the user's revision 2026-11-17 from Rollins et al. 2024 BSSA Table 1,
upper-plate IFM, replacing independent N5/b triples): b a split normal through its 10th percentile, mean, 90th (sd = distance / 1.2816);
N(b) the conditional mean, a quadratic in log10 N through the three pairs (mfd_gr.n_of_b; N and b correlated); the N scaling s a split
normal in ln s through its 2-sigma values (2.275/97.725%) with E[s] = 1 (mfd_gr.nscale_split; the user's reading 2026-11-23: 1 is the
mean, not the mode; before, mode 1 and E[s] 0.989; for (0.66, 1.41) mode 1.017, sds 0.212/0.168, every bin's PDF up ~1.1% at 630k); target = s N(b) f_j(b) (parse_knobs). RECONSIDER_MAGUNCERT
(2026-11-17, True by default; True, False or a triple): a further factor m on N for the catalogue's magnitude uncertainties (3.40 assumes
them downward-skewed, Rhoades 1996; half the skew gives 3.92; Gaussian ~+25-30%): mfd_gr.MAGUNCERT = (1, 3.92/3.40, 1 + 2(3.92/3.40 - 1))
three branches of EQUAL WEIGHT (the user's values and weights, 2026-11-17; to be revised after the catalogue's reanalysis; first read
as a normal's -2 sigma/mean/+2 sigma, then changed); the spec gains ,M1,M2,M3 (11 values); bin_pdfs: ln(s m)'s density the mean of ln s's
shifted by each ln m (logaddexp, exact); the row target uses m's middle value (central targets = cru_D's at 630k); RUN_NAME _mfdgr...m1-1.15294-1.30588; cru_common/make_summary_pdf show 'm'. The user
framed it as the 3x3 matrix [0.66, 1, 1.41] x [1, 1.13, 1.26]; s stays continuous (split normal in ln s), m the three branches. tools/mfd_gr.py rebuilds NZSHM22's partitioning (nzshm-opensha
NZSHM22_CrustalInversionTargetMFDs.RegionalTargetMFDs.init, NZSHM22_FaultSystemRupSetCalc): total GR 10^-bM at bin centres
5.05..ci(min(Mmax, maxMag)) summing to N5; sub-seismogenic = sum over sections of GR(pdf_s N5) up to max(ci(minmag_s), ci(6.8001)) - 1;
off-fault tri-linear (kink = moment-weighted mean section min mag as a bin, read from the stored off-fault MFD; 0.98/1.01 corrections;
log10 interpolation to ci(8.0)+1); supra = total - off - sub, 1e-20 below 6.8001; ci mirrors getClosestXIndex (1 + 1e-14 nudge, round
half up). Reads an inversion solution's regional_inversion_target_mfds.json, inversion_target_mfds.json, sub_seismo_on_fault_mfds.csv
(pdf_s = row sum / the run's N5), mod_sect_min_mags.csv, properties.csv; load_parts checks the rebuild against the stored MFDs at the run's
N5 and b (1e-14 at 19k, 211k, 630k) and stops otherwise; TVZ MFDs refused. Each component is proportional to N: target = s N(b) f_j(b);
bin_pdfs gives each bin's density of u = ln target exactly (ln s's density at u - ln(N(b) f_j(b)) summed over a fine b grid; s without
spread a 1% kernel), with quantiles, mode, sd and P(target <= 0) (a bin where it is not negligible refused); fit_bin_curves fits the
paleo sites' sinh-arcsinh curve in LOG RATE (paleo_misfit.fit_site mode 'log'; the user chose the density of log rate: its mode sits on the
target, where the rate density's mode is 0.71-0.94 of it at 630k), minimum at the exact mode, through the exact 2.275/15.865/84.135/97.725%
points (630k: quantile errors <= 0.003 decades, misfit within 0.30 of exact to 0.1/99.9%); gr_bins also returns each bin's 84.135% point
q84, saved as mfd_pdf_q84, from which gibbs.py sets the bin's prior soft top (sd log10(q84/R_j) in place of log10(1 + sigma_j/R_j)). Cleaning sets each bin row to the target at the
NZSHM22 comparison (2026-11-19): cleaning with --mfd-gr also saves mfd_in_rel and mfd_in_R (the inputs' sigma/target and targets
per bin row, before replacement); plot_rate_magnitude's MFD_GR line adds 'NZSHM22 bins ½/½ χ²' (MISFIT_MFD 'half' on those bins);
--nzshm-bins PROBLEM (a non-MFD_GR problem of the same inputs) for problems cleaned before. Checked: the NZSHM22 solution's
line reproduces the non-MFD_GR figure's 62.3/65.8/58.8 at 630k.
knobs' modes over the propagated sd (row_scale untouched) and saves mfd_pdf_th, mfd_pdf_best (per cleaned row, NaN elsewhere); the
notebook then sets MFD_MISFIT 'pdf' (MISFIT 'mixed'). gibbs.py --mfd-misfit pdf: the bins' rows out of the Gaussian (PALEO_EPS), their
curves tabulated by paleo_misfit.curve_tables and appended to the paleo tables (T.lw['PM']: I, V0 = U0 - ln(sd/eps), F, D, K, weight 1;
or alone, PM rows empty and the recording's PMr None), T.lw['MF'] = dict(pdf, idx, th, best, sd, E, V); recorded tr['mfd_misfit'] (the
curves' misfit) and mfd_pdf_rows/th/best/sd/E/V; chi2 as fitted swaps the bins' Gaussian for it; fit check E, V per bin
(paleo_misfit.site_moments). selftest_mfdpdf in the full set. best_fit_as_fitted.py --mfd-misfit pdf (the curves from the problem),
block_fit.py (the bins into PAL, mode 'log', b 1), plot_rate_magnitude.py (bins_fit's misfit from the curves), make_summary_pdf.py.
FIGURE_SETTINGS MFD_GR (shown 'N5 lo/mode/hi b lo/mode/hi', MISFIT_MFD 'pdf'); RUN_NAME _mfdgrN{lo-mode-hi}b{lo-mode-hi}, mixed misfit
name mfdpdf; run_quick_checks pins MFD_GR=False. History (all 2026-11-16/17, superseded): MFD_BOUND (gibbs.py --mfd-bound, a one-sided soft
top at the total GR), then MFD_GR with Gaussian bins (MFD_GR_SPREAD 'knobs' | 'nzshm22'), then sided log10 widths; the full PDF replaced
them. With two ruptures/ folders in NZSHM22_Matrices-630k (the rupture set and its inversion solution), RUPTURE_SET must name one.
Bin edges against magnitudes (2026-11-16, the user's request after checking that every constraint follows the model dialled up):
cleaning checks every rupture of each bin row against that bin's edges (--lowest-bin, --bin-width) by the rupture set's
properties.csv (cru_common.check_bin_magnitudes) and stops if any lies outside; 19k, 211k and 630k pass (Mmax 8.536, 8.524, 8.544).
Audit the same day: every constraint is built from the inputs and the rupture set (row groups detected, sections by parent fault,
CFM and timeframe by fault name, paleo sites checked against fault_sections.geojson, strips from the rupture set's geometry,
monotonic pairs from the bins' targets, Mmax per set); the pole and the strips' end points belong to the fault model the sets share.

Best fit in the sampler's model space (2026-11-17, the user's request): best_fit_as_fitted.py --down-the-middle, --slip-sum
(--slip-sum-l0, --cross-sections), --mfd-monotonic (--mfd-rise-sd, --lowest-bin, --bin-width), --plate-rate, as gibbs.py takes them
(sampler_terms: balance_rows and slip_sum_rows as Gaussian rows in the misfit; cru_common.mfd_monotonic's pairs, each row
(rho 1[lower] - 1[upper]) / R_lower >= 0, and plate_rows' shares within the limits, as normalized linear constraints); an augmented
Lagrangian around L-BFGS-B (mu from 1e4, x10 when the violation falls less than 4x; later rounds' window 100) to --con-tol 1e-9,
then polish_feasible (gibbs.py's start scalings); --snap-floor puts the fit in the sampler's set (rates below --floor to 0, then
positive_start over every row with a positive target at twice the floor, polish again). Saved terms (terms_key) and snapped;
gibbs.py --init fitted prints a note when terms differ (not a refusal). gibbs.py's monotonic pairs now come from
cru_common.mfd_monotonic (bit-identical). Without the new options the fit is identical to before. The notebook's best-fit cell is to
pass the run's flags and --snap-floor (after the user's 630k run of 2026-11-17).

RUN_MINUTES (240) and TARGETS ('production') replaced PRESET/PRESETS on 2026-11-17 (the user's request; one scheme, not two). tools/run_plan.py:
plan() in notebook_steps.plan_runs (the sampling cell), after cleaning and the starts: spent = SH_SECONDS (notebook_steps.sh times
every script; idle time not counted); post = the latest DATA_DIR run's timings.json post_s (the last cell writes it) or = spent;
budget = 0.9 x RUN_MINUTES x 60 - spent - post; two calibration gibbs runs (CALIB_SWEEPS 120, CALIB_BURN 24, CALIB_THINS 4 and
24, the run's settings and chains, quiet, deleted) -> calibration(): sweep time a + b / thin (per sweep, per recorded draw; one
rate misjudged 630k by 24%), setup (wall - npz 'seconds'); chains 4 x THREADS; replicates > 1 only if each gets >= 50,000
sweeps and is predicted at >= REPLICATE_SAFETY 1.5 x TARGETS (max 4); sweeps rounded down; burn 5,000 or 20%; thin for ~3,000
kept draws (thin and sweeps iterated). reference() compares MODEL_KEYS with OLD_DEFAULTS for settings older runs lack
(slip_along uniform, mfd_step_sd None, mfd_gr False).
Prediction: reference() = latest DATA_DIR run with convergence_summary.json (check_convergence.py writes it) or the report's
tables (older runs; thin from workflow_config), preferring identical MODEL_KEYS; tau = chains x draws x thin / ESS per metric
(ess_rate_5pct, ess_total, ess_pi_min = fifth of the total's target); minutes needed for TARGETS. run_plan.json (RUN_DIR),
CONFIG settings/run_plan updated and workflow_config.json rewritten; check_convergence.py --targets (TARGETS dict:
exploratory 100/200/1.01, production 1000/2000/1.01, precise 10000/20000/1.005) --plan; --preset for older runs
(PRESET_TARGETS). RUN_PLAN: run_config OVERRIDABLE only (tests; run_quick_checks pins the old quick 4 x 2,400, burn 400,
thin 8, and TARGETS 'exploratory'). FIGURE_SETTINGS RUN_MINUTES ('240 (16x56,000)'; older runs 'preset ...'); summary PDF
RUN_MINUTES and TARGETS rows (older runs PRESET), the prediction beside the achieved draws; RUN_NAME _{RUN_MINUTES}min.
630k very conservative run: tau 37 / 73 / ~2,100 sweeps -> 'production' ~245 min on 4 threads.
KEPT_DRAWS (2026-11-24, the user's request; notebook, beside REPRESENTATIVES, default 1000, the value fixed in
notebook_steps.gibbs_args before): the whole draws gibbs.py keeps (--keep-draws) when REPRESENTATIVES > 0, at least
REPRESENTATIVES; recorded as kept_draws (0 without representatives). Rule of thumb: about 10 effective draws per
representative, so ~10 x the largest K to be tried (select_draws.py --k on a copy of rate_vectors.*, so the run's own
representatives stay); 4000 kept is ~10 GB at 630k. Not a model setting (not in RUN_NAME or FIGURE_SETTINGS).
Infeasible polished starts redrawn (2026-11-24, the user's go-ahead): a 120-min run's start 4 (seed 80668435) polished
to 127 rates on and a total rate of 0 (762 rows predicted at 0, 3 monotonic pairs broken), and gibbs.py's calibration
refused it, stopping the notebook. dispersed_starts.py now checks each polished start as the sampler would (rates below
the slab floor off; every data row predicted above zero; with --mfd-monotonic every pair within its ratio) and redraws a
failing one from the prior (its own stream, SeedSequence([seed, 7, 1000 + attempt])) and polishes it again, up to 3
redraws, then stops with a message; first draws unchanged, so existing seeds give the same starts.
DISPERSED_STARTS (True, 2026-11-17, the user's choice: randomized fits, one per chain, noise x2): dispersed_starts.py writes
starts.npy (chains x kept cols): rows 0..7 randomized_nnls.ensemble(noise=2) NNLS fits, rows 8..15 prior draws (--prior-share 0.5:
per bin pi ~ U(0,1), on w.p. pi, log-uniform 2 floor..R_j, bin scaled to R_j); none from the unperturbed NNLS fit; each then
polished FOR FEASIBILITY by best_fit_as_fitted.py --max-iter 30 per AL round (--polish-iter; settling polishes collapsed the
starts: bin totals 0.6% spread vs posterior 8%), prior draws' bins scaled to R_j exp(2 sig/R z); the dispersion reported; the calibration starts from starts.npy too, timed from its second half (progress lines; sh(quiet) returns its output) with the first half's excess added per run (cost['dense']); the run's args (notebook _fit_args after --fit-args), in
parallel subprocesses one thread each (unpolished starts broke the monotonic MFD and the slab floor's snap left rows at 0);
the sampling cell runs it (notebook_steps.starts_args, seed replicate_seed(1000)) before plan_runs; gibbs_args(init 'starts.npy'); calibration from the
NNLS fit. run_plan REPLICATES_MAX = 1 (no splitting). Smoke test pins DISPERSED_STARTS=False. best_fit_as_fitted.py removed
from the notebook (2026-11-17, the user's choice; 630k trace: 227.5 -> 151.9 at 500 its, 149.1 at 1,500, 147.4 at 5,000;
default --tol 0.01 took 27 min): chains start from NNLS fits (gibbs_run start = init or 'nnls'; dispersed_starts.py --base
nnls default, fitted optional); the best-fit cell draws the NNLS fit's figures only.
The old presets (2026-11-11, the user's values, TESTING.md 'Run lengths'): quick 4 x 2,400 (burn 400, thin 8), 1 replicate; conservative
8 x 100,000 (burn 5,000, thin 32), 1 replicate; very conservative 8 x 300,000 (burn 5,000, thin 32), 2 replicates (about 8.5 h at 211k).

Conventions that every new option follows (static_checks.py enforces the first): a model setting in the notebook is
validated in the configuration cell, recorded in CONFIG, listed in cru_common.FIGURE_SETTINGS, part of RUN_NAME where
it changes the results, passed to its script, given a default check in static_checks.py, shown in make_summary_pdf.py's
settings table, described in the notebook's markdown and in REPORT.md and CLAUDE.md, and pinned in run_quick_checks.sh
where the quick checks' reference values would otherwise move. Every term outside chi2 that a run records (the balance
term, the summed slip rates, the soft band) is recorded as its chi2-equivalent, the sum of its squared standardized
deviations as chi2 is (twice its share of the negative log posterior, as chi2 is twice the likelihood's), apart from
chi2, so the fit check stays over the data alone.
plot_along_strike.py (2026-11-24, the user's request): per named fault, NZSHM22's "Along-Strike Values" panels without
nucleation/moment/b-value: participation (all, M>=7, M>=8), paleo-visible (UCERF3 visibility at the subsection middle,
rupture_set_inputs.paleo_visibility) with sites, slip rate (CFM SR_pref, target + 2.275-97.725%, posterior), and a magnitude
panel (participation rate per 0.1-M bin, median M). Posterior mean + kept draws' 2.275-97.725% (main_rep*_kept.npy, chunked), refs
dash-dot. Called in the notebook's fault-figures cell; outputs <prior>/along_strike/.

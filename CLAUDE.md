# Posterior sampling for a nonnegative, underdetermined linear inverse problem

Design notes: @docs/sampler-design.md (section 0 records what the 2e4 data showed and overrides the rest)
Findings, numbers and run commands: @docs/REPORT.md
What to test and check before routine use: @docs/TESTING.md
Workflow notebook: cru_workflow.ipynb (the configuration, then one call per step: it drives the scripts below end to
end and computes nothing itself).
Code: preliminaries.py (clean, with soft zero-forcing rows forcing their ruptures to 0 like the penalty rows; diagnostics;
NNLS, skipped with --no-nnls (the notebook always computes it and starts its chains from it); the best fit's outputs go into RUN_DIR/nnls/: its rate vectors,
nnls_vectors.npz, which the notebook draws with the three fit-figure scripts below, and nnls_vectors.csv, a table with
rate_vectors.csv's rows), make_solution.py (a rate vector as an OpenSHA fault system solution, solution/rates.csv and
modules.json, after checking that the model's OpenSHA rupture set describes the same ruptures; the notebook's last cell
writes the best fit's, nnls/solution/, and the posterior mean's, spikeslab/solution/ (rnnls/solution/ for the randomized
fits' mean), one row per rupture of the rupture set like the model's own rates.csv, when DATA_DIR holds the model's
OpenSHA solution folder, which RUPTURE_SET = 'auto' finds, then (rupture_dir())), gibbs.py (production sampler; Gaussian
likelihood), check_convergence.py (a result set's convergence checks and PASS/CHECK verdict, with each rupture's R-hat
and effective draws in convergence.npz), make_tables.py (the posterior tables and rate vectors, pooled over the
replicates), block_fit.py (each row block's fit from a run's recorded predictions),
compare_runs.py (rate-by-rate agreement of two runs, with a figure), summarize_runs.py (log-space summaries of
replicates or priors, from the running summaries or stored draws), cru_common.py (shared I/O and diagnostics), compare_priors.py (tables comparing runs under different priors),
select_draws.py (weighted representative draws for hazard), plot_rate_magnitude.py (each rupture's posterior mean
rate against its magnitude bin, from the rate vectors, with each bin's ruptures counted by decade of mean rate),
plot_slip_rates.py (slip-rate targets against the mean model's predictions, linear and log, from the rate vectors),
plot_paleo_rates.py (paleo-rate targets against the model's predictions, log-log, with each prediction's posterior
16-84% range from the runs' recorded predictions), randomized_nnls.py (best fits to the data plus fresh draws of their
stated noise, in place of the sampler when the notebook's RANDOMIZED_NNLS is a number of fits: their spread of slip-rate,
MFD and paleo predictions in figures, in RUN_DIR/rnnls/; the best fit's sampling distribution, not a posterior).
The fit figures (plot_slip_rates.py, plot_rate_magnitude.py, plot_paleo_rates.py, randomized_nnls.py) take --reference
RATES[:LABEL] (cru_common.reference_rates): another model's rates, marked on every panel and tabulated; the notebook
passes the model's own OpenSHA solution (ref_args(): solution/rates.csv beside the rupture set, looked for when each
figure is drawn, so one finished during the run counts; the best fit's figures come last) when DATA_DIR holds it.
Their axes are fixed per problem by the targets and the best fit (cru_common.axis_span and best_fit_rates: all of the
targets and the central 96% of the best fit's predictions, from prelim_results.npz), so every model's figures for one
problem share them; values beyond sit on the edges as triangles. In place of the result set's label, each shows the
run's settings, SLIP_SECTIONS to FLOOR (cru_common.run_settings, from workflow_config.json next to --problem; --config),
and the rates-by-magnitude figure gives the chi2 over the bin rows.
rhmc.py (log-space HMC cross-check), evidence.py (evidence by stepping-stone sampling) and build_transfers.py (transfer
moves, single box only) are set aside in cru_retired_scripts.zip; restore them only if that work resumes.
gibbs.py defaults to --prior spikeslab (spike-and-slab per magnitude bin: each rupture off, rate 0, or on, log-uniform
from the slab floor 1e-10 up to a soft top at the bin's target rate R_j whose width is the bin's own uncertainty; the
fraction on in each bin estimated under Beta(1, 1); spikeslab:A:TOP:a:b; --hyper-every defaults to 1 for it, so the
fractions are redrawn every sweep). --slab sqrt (command line only; not in the notebook) was its cross-check until the
user dropped it at the end of October 2026 (the request behind it had been misphrased; do not propose it as a check):
the same with an on rupture's rate uniform in sqrt(rate), density m^-1/2 instead of 1/m; it fitted as the default does,
and at 211k it moved the subsection event rates (REPORT.md, "Fits and the prior's lower end").
--prior linear (command line only) is optional: each rupture's rate uniform from 0 to its bin's target rate R_j,
drawn in linear space by the exact truncated-normal Gibbs updates; its prior pushes the bin totals far up (REPORT.md,
"Why not a linearly uniform prior"). --prior halfway (command line only) is optional too: each rate from a 50/50 mix
of a uniform and a log-uniform range, 1e-10 to R_j, every rupture always on; the data give almost every value to the
log-uniform part, so it behaves like the per-bin log-uniform boxes, pushing the bin totals up.
--down-the-middle (the notebook's DOWN_THE_MIDDLE = True, its default) adds a balance term to the misfit, for any prior:
each data type's relative shortfall s = sum of predictions / sum of targets - 1 (slip-rate sections, MFD bins, paleo
rows, in units of each measurement's uncertainty) costs A s^2, with one A for all types, the largest that keeps every
type at or below one extra measurement; DOWN_THE_MIDDLE = 'zero-model' (--down-the-middle zero-model) gives each type
its own A instead, its zero-model misfit (cru_common.balance_rows weights; extra rows of G and d). The recorded chi2 and
predictions stay on the data rows, and the term is recorded on its own; the best fit and randomized_nnls.py ignore it.
Kept for comparison only, and no longer tested: binbox-theta:1:0 (theta = 1: per-bin log-uniform boxes below R_j,
depths n_j / (theta ln 10); command line only; the alternative until October 2026, retired because it fitted no
better and from about 2e5 ruptures its boxes are too deep for its sampler to cross), binbox-floor:-12 (per-bin boxes
down to a common floor; the default until October 2026, when it failed the fit check by 26 sd; binbox:W for a bottom W
decades below each R_j), and
logbox:A:B:SE, logbox-centre:W:SE and logbox-mean:W:SE (the single box). --prior lognormal (the hierarchical prior) is
retired: it never converged (see REPORT.md). Transfer moves (--transfers) exist for the log-boxcar only and are set
aside with it.

## Working agreements
- Every gibbs.py run writes <out>_stream.npz (running summaries); --no-samples skips the large draws file, and the
  notebook always passes it and uses the running summaries for its checks and tables. make_tables.py
  (run by the notebook) always writes rate_vectors.csv/.npz (mean, median, mode, 2.275/15.865/84.135/97.725% per rupture, original column indexing),
  from the running summaries' 0.01-decade log-rate histogram (gibbs.py --hist-width), or for older runs from their
  stored draws, binned the same way. With REPRESENTATIVES > 0 (20 by default) each run also keeps 1,000 whole draws
  (gibbs.py --keep-draws: <out>_kept.npy and _kept_index.npy; recording only, the chains are unchanged) and
  select_draws.py writes weighted representatives as columns rep_01... of rate_vectors.csv, with representatives.csv
  and select_report.txt; the mean column stays the source for mean hazard. The notebook also writes
  spikeslab/workflow_log_<date>-<time>.txt (rnnls/ for the randomized best fits), a copy of
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
- Slip-rate units: the input rows are divided by their uncertainties, which the inputs do not give, so plot_slip_rates.py
  shows targets and predictions in units of each section's measurement uncertainty (undoing the sqrt(k)), section means
  against their targets with the subsections behind them (the 1:1 line and the one-uncertainty band are exact; mm/yr
  would need the uncertainties). plot_paleo_rates.py shows the paleo rows (every row in neither the slip-rate nor the
  bin group whose target is positive, unless --paleo-block names them) the same way, log-log: there the units only move a point along the 1:1
  line, so its distance from the line is the ratio of prediction to target.
- Code, data and outputs may be in separate folders. Scripts take input paths as arguments, find prelim_results.npz
  next to --problem (or via --prelim), and write where --out points; the notebook uses DATA_DIR (read only) and RUN_DIR,
  relative to its own folder; the runs go into RUN_DIR/spikeslab/ (the randomized best fits into RUN_DIR/rnnls/), with
  the shared cleaned problem and diagnostics at the top of RUN_DIR, the best fit's outputs in RUN_DIR/nnls/ and the
  randomized best fits' (RANDOMIZED_NNLS) in RUN_DIR/rnnls/. One prior per notebook run (no comparison runs); a second
  setting of the same prior needs its own RUN_DIR. Never write into the data folder.
- Before routine use, work through docs/TESTING.md. The slip-rate and magnitude-bin rows are detected
  automatically during cleaning (preliminaries.py), and any group named in --blocks (original row numbers, which
  change with the input) is checked against the detection. check_bin_rows and check_slip_rows
  (cru_common.py) must pass for any new input; compare the slip-rate profile they print with the 2e4 one.
- Read section 0 of docs/sampler-design.md and docs/REPORT.md before proposing changes to the approach.
- Floors and cutoff: one setting, the notebook's FLOOR = -10 (1e-10), is spike-and-slab's slab floor and the diagnostic
  and reporting cutoff (the notebook passes --diag-floor={FLOOR} to gibbs.py, preliminaries.py, summarize_runs.py and
  plot_rate_magnitude.py, whose defaults match; REPORT.md, "Floors and cutoff"); for theta it is only the cutoff. The
  per-bin boxes, if run, take it as their floor (-12 reproduces their earlier default).
  preliminaries.py reports the collective negligibility scale; on new inputs (5e5 especially), check that FLOOR lies
  below it.
- The convergence verdict (PASS/CHECK, check_convergence.py) checks the targets in docs/TESTING.md. Ruptures with R-hat > 1.01 (or above
  the noise baseline, if higher: the slow-mixing minority under spike-and-slab) are reported separately, with their share of the total rate, and
  flagged in ruptures_posterior.csv (rhat_log_rate, ess_log_rate, slow_mixing).
- Run `python gibbs.py --selftest` after any change to the sampler, and `python block_fit.py --selftest` after any
  change to the row-block fit report or the problem helpers in cru_common.py (the MFD uncertainty among them); both
  must pass. Any sampler change must also keep the rate-by-rate agreement in compare_runs.py
  (z-scores ~ N(0,1), |z| > 3 near 0.27%). The notebook's RUN_SELFTESTS runs only `--selftest current`, the cases
  the notebook's default uses (per-rupture lower edges, the soft top, spike-and-slab, the balance term, swap
  moves). A change meant only to speed things up must leave a fixed-seed run's output arrays
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
  one: the chi2 cap, --target cutoff, was removed in October 2026), with the balance term (DOWN_THE_MIDDLE,
  --down-the-middle) on by default; no other prior is run as a cross-check (the square-root slab, --slab sqrt, was
  dropped at the end of October 2026). Tests and smoke runs use the default only; the notebook runs it alone,
  always with swap moves (--swap-cos 0.99), from the best fit and without stored draws. Sharing the fraction on by magnitude
  bin and tectonic domain or slip-rate section, pooled or not, was tried in October 2026 and changed no fit (REPORT.md).
  A spike-and-slab run counts as converged only when the number of ruptures on and every bin's fraction on have
  R-hat < 1.01, as well as the rates. A linearly uniform prior is only an option (--prior linear): it pushes the bin totals far up and fails
  the fit check by hundreds of sd at 2e4 and thousands at 211k; so is the halfway prior (--prior halfway). The prior, its floor,
  edges and hyperpriors, the likelihood (the balance term and its weight among it), sigma, bounds, how the slip-rate rows
  count and the MFD bins' uncertainty level are modelling decisions: ask the user before changing them. If all-zero rows are kept (--keep-empty-rows), the fit check
  includes their constant.

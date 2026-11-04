# Posterior sampling for a nonnegative, underdetermined linear inverse problem

Paths below are relative to the repository's folder (this file is in its docs/, beside the documents it imports).
Design notes: @sampler-design.md (section 0 records what the 2e4 data showed and overrides the rest)
Findings, numbers and run commands: @REPORT.md
What to test and check before routine use: @TESTING.md
Workflow notebook: cru_workflow.ipynb (the configuration, then one call per step: it drives the scripts below end to
end and computes nothing itself; it finds them in tools/ beside it, CODE_DIR).
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
line only, needs reportlab), gibbs.py (production sampler; Gaussian
likelihood), check_convergence.py (a result set's convergence checks and PASS/CHECK verdict, with each rupture's R-hat
and effective draws in convergence.npz), make_tables.py (the posterior tables and rate vectors, pooled over the
replicates), block_fit.py (each row block's fit from a run's recorded predictions),
compare_runs.py (rate-by-rate agreement of two runs, with a figure), summarize_runs.py (log-space summaries of
replicates or priors, from the running summaries or stored draws; command line only since the notebook's summary
figure was dropped at the end of October 2026), cru_common.py (shared I/O and diagnostics), compare_priors.py (tables comparing runs under different priors),
select_draws.py (weighted representative draws for hazard), plot_rate_magnitude.py (each rupture's posterior mean
rate against its magnitude bin, from the rate vectors, with each bin's ruptures counted by decade of mean rate, and
below it each bin's total over its target with the total's posterior 16-84% range),
plot_slip_rates.py (slip-rate targets against the mean model's predictions, linear and log, from the rate vectors, with
each section mean's posterior 16-84% range), plot_paleo_rates.py (paleo-rate targets against the model's predictions,
log-log, with each prediction's posterior 16-84% range; with kept draws, the default, all of these figures and
plot_slip_sections.py's draw each kept draw as an x instead (cru_common.kept_draws, kept_predictions, draw_xs); the three ranges come from the predictions the runs next to
the rate vectors recorded, main_rep<k>.npz, cru_common.run_predictions; the slip-rate sections and paleo rows are
coloured from cru_common.CATEGORICAL (Tableau's Color Blind 10, outlined in grey as several are pale), cycled
along the targets so that neighbours differ, neighbour_colours, and the reference model is black), plot_slip_sections.py (the posterior slip-rate predictions of the rupture set's named
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
RATES[:LABEL] (cru_common.reference_rates): another model's rates, marked on every panel and tabulated; the notebook
passes the model's own OpenSHA solution (ref_args(): solution/rates.csv beside the rupture set, looked for when each
figure is drawn, so one finished during the run counts; the best fit's figures come last) when DATA_DIR holds it.
Their axes are fixed per problem by the targets and the best fit (cru_common.axis_span and best_fit_rates: all of the
targets and the central 96% of the best fit's predictions, from prelim_results.npz), so every model's figures for one
problem share them; values beyond sit on the edges as triangles. In place of the result set's label, each shows the
run's settings, SLIP_SECTIONS to FLOOR (cru_common.run_settings, from workflow_config.json next to --problem; --config),
and the rates-by-magnitude figure gives the chi2 over the bin rows.
paleo_misfit.py (the paleo rows' curves for --paleo-misfit, from the rupture set's paleo_constraint_data.json: a report of their fit with python paleo_misfit.py RUPTURE_SET). cfm_sliprates.py (preliminaries.py --cfm-sliprates: the slip-rate rows rebuilt from the NZ CFM's SR_pref/SR_min/SR_max; a report with python cfm_sliprates.py CFM_CSV RUPTURE_SET --A --d). rhmc.py (log-space HMC cross-check), evidence.py (evidence by stepping-stone sampling) and build_transfers.py (transfer
moves, single box only) are set aside in cru_retired_scripts.zip; restore them only if that work resumes.
Development tools, also in tools/ and not run by the notebook: run_nb.py (runs the notebook headless, cell by cell,
with configuration values overridden from the command line: "PRESET='quick'", CELLS=5, ...), run_quick_checks.sh
DATA_DIR RUNS_DIR (the quick-preset smoke test of docs/TESTING.md, one notebook run at a time, resumable),
static_checks.py (inventory, parsing, leftovers, defaults, options and imports of the code set: update it whenever a
default, a setting or the file list changes), compare_mfd.py (two result sets compared on fit, bins, total rate, MFD
slope, population and products, with replicate noise for scale), fig_mfd_levels.py (bin totals against targets under
two MFD uncertainty levels) and psis.py (Pareto-smoothed importance sampling, PSIS-LOO, as used for the paleo rows'
leave-one-out checks).
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
--down-the-middle adds a balance term to the misfit, for any prior: each data type's relative shortfall s = sum of
predictions / sum of targets - 1 (slip-rate sections, MFD bins, paleo rows, in units of each measurement's uncertainty)
costs A s^2. DOWN_THE_MIDDLE = 'zero-model' (--down-the-middle zero-model; the notebook's default in late October
2026) gives each type its own A, its zero-model misfit; DOWN_THE_MIDDLE = True (--down-the-middle alone) one A for
all types, the largest that keeps every type at or below one extra measurement (cru_common.balance_rows weights;
extra rows of G and d). The notebook's default is False since early November 2026, with the band. The recorded chi2 and
predictions stay on the data rows, and the term is recorded on its own; the best fit and randomized_nnls.py ignore it.
Bands by row group (since November 4th, 2026; the notebook's SLIP_BAND, SLIP_BAND_SOFT, PALEO_BAND, PALEO_BAND_SOFT):
--slip-band and --paleo-band (positive or a factor F) and --slip-log-band and --paleo-log-band (TAU, with that group's
positive) act on the subsections' slip-rate rows and the paleo rows separately (Target.set_groups; the MFD bins get no
band, which would count them twice); they replace, in the notebook, BAND and BAND_SOFT, which acted on every
positive-target row through the older flags below, still there for older runs and tests and not combinable with the new
ones. Defaults: SLIP_BAND = PALEO_BAND = 'positive', no soft band. --paleo-misfit years (PALEO_MISFIT, 'gaussian' by
default; with --rupture-set and a hard paleo band): the paleo rows scaled out of the Gaussian (by PALEO_EPS = 1e-9, as
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
preliminaries.py --cfm-sliprates CSV --rupture-set DIR (CFM_SLIPRATES, False by default; the notebook passes CFM_FILE):
each raw slip-rate row (row j = FaultID j) rebuilt from its parent fault's CFM line (by name; all 557 match at 211k) as a
split normal in mm/yr (cfm_sliprates.rebuild): mode f SR_pref with f the inputs' own factor (0.8, target x sigma /
SlipRate, checked constant; sigma from the row's coefficients against average_slips.csv), side sds --cfm-factor x f x
(SR_pref - SR_min or SR_max - SR_pref) / --cfm-sigmas: f, the assumed fraction of the slip rate the ruptures take up,
scales the whole distribution (the user, 2026-11-04: at f = 0.01 an sd of several mm/yr would be absurd), where the
inputs scale only their targets. Defaults 1 and 2: SR_min/SR_max at 1 sd and doubled, as the inputs read and double
theirs, so each side's sd is 2f x its width and sigma_ref a median 0.80 of the inputs' (5-95% 0.75-0.90; the rest from
the inputs' rounding of SD to 0.01 mm/yr and the 12 zero-width sides). Tried the same day and dropped: 2 and 2 (sigma
0.40 of the inputs'; slip chi2 698 against 507 in a short run) and f on the mode only. The user is checking the
reading with colleagues. --slip-rate-factor F and --slip-sd-multiplier K (SLIP_RATE_FACTOR 0.8 and SLIP_SD_MULTIPLIER
2 by default; the notebook passes both, with --rupture-set, unless SLIP_AS_GIVEN: CFM off, 0.8 and 2.5; --slip-sd-
multiplier replaced --cfm-factor): every slip-rate distribution with F on the whole of it, target F x rate and sd
K x F x its uncertainty: the CFM's sides (rebuild's srf and factor) or, without --cfm-sliprates, the inputs' Gaussians
(cfm_sliprates.rescale_inputs: sd K x F x SD; their own convention, checked, is target 0.8 x SlipRate, sd 2 x SD, the
factor on the targets only, an implementation error upstream in the NZSHM22 code as the user reads it, 2026-11-05; the
8 subsections with an SD of 0 keep sd = the target). So since 2026-11-05 the default runs' slip sds are 0.8 of the
inputs' (coefficients and targets x 1.25 in sd units); K = 2.5 with F = 0.8 reproduces the inputs bit for bit (every run
before then; run_quick_checks.sh pins it, for TESTING.md's values). K = 1 switches the doubling off, for slip-rate
uncertainties estimated from the data. Recorded as the problem's slip_rate_factor and slip_sd_multiplier. Without either
flag preliminaries.py leaves the inputs as given. A zero-width side takes the other's and
min = pref = max gives sd = the mode (as the inputs: sigma = 0.8 x SlipRate there);
the row rescaled to sigma_ref = the mean of the two sides, and the weights split_lo2 = (sigma_ref/sigma_lo)^2 and
split_hi2 = (sigma_ref/sigma_hi)^2 saved per cleaned row (load_problem; None without it; cru_common.split_weights).
plate_rate.cross_sections reads each subsection's target and sigma from its row, so --slip-sum sums the modes with
sigma_ref unchanged in code. gibbs.py reads the weights from the problem (T.split; spike-and-slab only, no transfers):
the misfit sum w(e) e^2, w = split_lo2 below the target and split_hi2 above, concave in m, so the on-chain slice uses
the exact likelihood (lbctx's sp fields, sp_lik; lb_floor's endpoint bound still holds), off chains draw their slab
value first and get their exact change in one pass over the column (col_split, with the soft band's bound sums), the
on/off decisions use sp_change, the swaps sp_swap; recorded chi2 is the split misfit; NumPy twins; selftest_split (both
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
dot products, settle nearly all decisions, most without logs, the same draws), T.lw = dict(W, LD, DY, rows, tau); selftest_logband in
the full set; NULL arrays without it, so runs are bit-identical. --positive (BAND = 'positive'), an alternative,
spike-and-slab only: every such row only kept above zero (Target.set_positive: a lower limit of half the smallest
one-rupture amount, no upper limit, through the same rlo/rhi; positive_start switches on, at twice the slab floor, a
rupture in each row the start predicts at 0; its exact test is in selftest_band). --mfd-monotonic (command line only since November 4th, 2026; before, the notebook's
MFD_MONOTONIC, on by default from early November 2026): no bin's total above its lower neighbour's where the target is not (and,
since November 3rd, 2026, where it rises, the rise capped at rho = (R_low + (R_up - R_low) + --mfd-rise-sd sd) / R_low
times the lower total; --mfd-rise-sd, 0.89 by default, command line only), through constraint rows after
the likelihood's (T.n_fit; eps 1e-9 coefficients, Target.add_limits), with or without the band; exact test in
selftest_band. --cross-sections all|between (CROSS_SECTIONS, 'all' by default): the strips that --slip-sum and
--plate-rate share (plate_rate.py's cross_sections: cross-sections are great circles through the Beavan et al. 2002
pole; strips the mean subsection trace length wide, 8.5 km, 72 between END_SW, where the Alpine Fault goes offshore,
and END_NE, the Needles / Wairarapa: Needles step-over in Cook Strait; 'all': 166 over all of New Zealand).
--plate-rate HI or LO,HI --rupture-set DIR (command line only since November 4th, 2026; before, the notebook's
PLATE_RATE, off by default; 0.8 if imposed, an upper limit only: the
whole plate rate at the inputs' 0.8 of the fault model's slip rates; a lower limit of 0 is not implied beyond the
edges, where some faults slip against the plate motion): each strip's share of the Pacific-Australia plate rate
within the limits, one constraint row per strip holding a rupture after the monotonic ones (plate_rate.py's
plate_rows: each crustal subsection weighted by the fraction of the strip's cross-sections crossing it, its slip
vector projected on the local plate motion over the local rate; checked against the slip-rate rows; a start above HI
scaled down to it, each chain by one factor); exact test in selftest_band (until November 3rd, 2026: 8 strips of
76 km between the edges, --plate-strips, LO,HI only). --slip-sum --rupture-set DIR (SLIP_SUM, True by default from November 3rd, 2026, False since November 5th: the user wants
to see whether every target can be well fitted without it, from the new constraints and positivity alone): a soft term
in the likelihood, rows after the balance term's (T.n_bal) and before n_fit, recorded apart as tr['slip_sum']
(plate_rate.py's slip_sum_rows: per strip, (width / L0) ((sum_i w_ki c_i (model_i - target_i)) / sigma_k)^2, w_ki the
fraction of subsection i's extent along the plate motion in strip k, so each subsection's weights add to 1; c_i its
slip vector per unit slip projected on the local plate motion, signed (since November 3rd, 2026; before, c_i = 1); sigma_k
with each parent section's subsections fully correlated; on the --cross-sections strips; L0 =
--slip-sum-l0 (SLIP_SUM_L0, 8.5 km by default, the strips' width: one extra observation per strip; 120 km would be the
unprojected target sums' decorrelation length between the edges, about the span over its 5 Alpine-Wairau sections, 122 km); no whitening between strips, so the
strip count only sets the integral's accuracy; the swap pairs' inner products include its rows, checked on a sample at
start). Its bookkeeping test is in selftest_balance. plate_rate.py reports both (plate_rate.txt, a notebook cell for
every run with a rupture set).
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
- Slip-rate units: the input rows are divided by their uncertainties, which the inputs do not give, so plot_slip_rates.py
  shows targets and predictions in units of each section's measurement uncertainty (undoing the sqrt(k)), section means
  against their targets (the 1:1 line and the one-uncertainty band are exact; mm/yr
  would need the uncertainties). plot_paleo_rates.py shows the paleo rows (every row in neither the slip-rate nor the
  bin group whose target is positive, unless --paleo-block names them) the same way, log-log: there the units only move a point along the 1:1
  line, so its distance from the line is the ratio of prediction to target.
- Code, data and outputs may be in separate folders. Scripts take input paths as arguments, find prelim_results.npz
  next to --problem (or via --prelim), and write where --out points; the notebook uses DATA_DIR (its files only read) and RUN_DIR,
  relative to its own folder (by default ~/inversions/NZSHM22_Matrices-211k/ and a new subfolder of it named after the
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
  (z-scores ~ N(0,1), |z| > 3 near 0.27%). The notebook's RUN_SELFTESTS runs only `--selftest current`, the cases
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
  one: the chi2 cap, --target cutoff, was removed in October 2026), with every slip-rate and paleo prediction above zero by default (SLIP_BAND = PALEO_BAND = 'positive', --slip-band and --paleo-band positive)
  and without the balance term (DOWN_THE_MIDDLE = False); no other prior is run as a cross-check (the square-root slab, --slab sqrt, was
  dropped at the end of October 2026). Tests and smoke runs use the default prior only (the smoke test,
  tools/run_quick_checks.sh, pins FLOOR = -10, DOWN_THE_MIDDLE = True, no band on either group, PALEO_MISFIT = 'gaussian' and CFM_SLIPRATES = False, under which TESTING.md's quick-preset
  reference values were measured); the notebook runs it alone,
  always with swap moves (--swap-cos 0.99), from the best fit and without stored draws. Sharing the fraction on by magnitude
  bin and tectonic domain or slip-rate section, pooled or not, was tried in October 2026 and changed no fit (REPORT.md).
  A spike-and-slab run counts as converged only when the number of ruptures on and every bin's fraction on have
  R-hat < 1.01, as well as the rates. A linearly uniform prior is only an option (--prior linear): it pushes the bin totals far up and fails
  the fit check by hundreds of sd at 2e4 and thousands at 211k; so is the halfway prior (--prior halfway). The prior, its floor,
  edges and hyperpriors, the likelihood (the balance term and its weight among it), the band and its factor, sigma, bounds, how the slip-rate rows
  count and the MFD bins' uncertainty level are modelling decisions: ask the user before changing them. If all-zero rows are kept (--keep-empty-rows), the fit check
  includes their constant.

Conventions that every new option follows (static_checks.py enforces the first): a model setting in the notebook is
validated in the configuration cell, recorded in CONFIG, listed in cru_common.FIGURE_SETTINGS, part of RUN_NAME where
it changes the results, passed to its script, given a default check in static_checks.py, shown in make_summary_pdf.py's
settings table, described in the notebook's markdown and in REPORT.md and CLAUDE.md, and pinned in run_quick_checks.sh
where the quick checks' reference values would otherwise move. Every term outside chi2 that a run records (the balance
term, the summed slip rates, the soft band) is recorded as its chi2-equivalent, the sum of its squared standardized
deviations as chi2 is (twice its share of the negative log posterior, as chi2 is twice the likelihood's), apart from
chi2, so the fit check stays over the data alone.

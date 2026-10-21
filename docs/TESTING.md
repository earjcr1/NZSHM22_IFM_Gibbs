# CRU workflow: testing guide

This guide covers the current code set: spike-and-slab (the default prior) and θ = 1 (its alternative and cross-check), one `FLOOR` setting (1e-10) serving as spike-and-slab's slab floor and the diagnostic cutoff, one measurement per slip-rate section (`SLIP_SECTIONS = 'auto'`), the MFD bins' uncertainties as the inputs give them (`MFD_UNCERTAINTY = 'as-given'`; `'two-measurements'` counts the bins as the two measurements they are projected from), and running summaries in place of stored draws. The per-bin boxes (`PRIOR_KIND = 'binbox'`), the earlier default, still run but are no longer part of these tests. Figures marked as estimates have not been measured. Figures marked "counting each subsection" were measured before one measurement per section became the default, with every slip-rate subsection its own measurement (now `SLIP_SECTIONS = 'none'`). The findings behind these checks are in `REPORT.md`, and the sampler design in `sampler-design.md`.

## Before any run

Run the self-tests once per machine and after any change to `gibbs.py`, and check the cleaning output against the reference values below whenever the inputs change.

- **Self-tests:** `python gibbs.py --selftest` (the notebook's `RUN_SELFTESTS`, about 6 minutes) prints five verdicts, all `PASSED`: the main suite, the retired hierarchical prior, per-coordinate lower edges, per-coordinate upper edges, spike-and-slab. The main suite includes the transfer moves; their helper now lives in `cru_common.py`, so the tests run with only the current scripts present. The retired scripts' self-tests are no longer part of the routine.
- **Row groups:** with `BLOCKS = None`, the slip-rate and magnitude-bin rows are detected automatically; `check_bin_rows` and `check_slip_rows` must pass, and any group you name is checked against detection.
- **Slip-rate sections:** cleaning prints the sections it found (runs of copies of one slip rate), with notes on targets shared between sections and on neighbours with identical targets that share no rupture, and the fit check. On new inputs, compare the sections with the fault model; if they disagree, give a mapping file (`SLIP_SECTIONS = 'file.txt'`, lines of `original_row section`).
- **Collective scale:** the preliminaries' line must say the floor (1e-10) is below the collective negligibility scale.
- **Row-block fit and MFD helpers:** `python block_fit.py --selftest` (seconds) must pass after any change to it or to the problem helpers in `cru_common.py`. It checks the block measurements and the misfit split, and the MFD uncertainty: the bins counted in magnitude order (a removed bin leaving a gap), the M7.0–7.1 bin found, each setting's multiplier, the inputs' power recovered, what the bins say about the MFD as a whole, the bin rows divided and every other row unchanged, and bad settings refused.
- **MFD uncertainty:** cleaning prints the inputs' bin uncertainties (10.0% at M7.0–7.1, from 9.8% to 15.4%, growing as R_j^−0.250, exactly) and what the bins say about the MFD as a whole (±2.8% on the total rate and ±0.024 in b; with `'two-measurements'`, ×3.000, then ±8.3% and ±0.071). On new inputs, check the power and the M7.0–7.1 bin; a warning that rupture sizes do not rise with the bins' order means the magnitudes, and the bin a number refers to, may be wrong.

| Check | Corrected 2e4 inputs |
|---|---|
| Raw shape | 867 × 18,648 |
| Removed by cleaning | 14 penalty rows with the 405 ruptures they zero; 407 all-zero rows |
| Cleaned problem | 446 × 18,243: 388 slip-rate, 40 paleo, 18 magnitude-bin rows |
| Slip-rate sections | 68, of 2–21 subsections (median 5); 32 share their target with another section; 6 identical-target neighbour pairs share no rupture |
| Bin targets, total | 0.0214 per year |
| MFD bins' uncertainty | 10.0% at M7.0–7.1 (9.8–15.4%, ∝ R_j^−0.25), as given; ×3 with `'two-measurements'` |
| Best nonnegative fit | χ² 31.25, 111 nonzero rates, bins 1.0–4.9% above targets (counting each subsection: 70.22, 132, 2–8%; with the bins counted as two measurements: 27.19, 130, 6.9–18.3%) |
| Fit check for posteriors | χ² near 126 ± 16 (68 sections and 58 other rows; counting each subsection, 446 ± 80) |
| Collective negligibility scale (0.1σ) | 1.8e-9, set by original row 156 (counting each subsection, 6.9e-10) |
| Block fit (quick spike-and-slab) | χ² per independent measurement 1.04 (68 slip-rate sections), 1.00 (18 bins), 1.29 (40 paleo rows); with the bins counted as two measurements 1.03, 1.00, 1.22 |

On new inputs the numbers change; what matters is that the groups and sections are detected, the best fit is sane and the collective-scale line says "below".

## Smoke test

After any change to the code or the inputs, run the quick preset once with spike-and-slab and once with θ: every cell should run, and the results should land near these values.

The quick preset usually says `CHECK`; it is too short for these priors' targets, and that is expected. The values below come from the quick preset as the notebook runs it (two replicates, without stored draws), on the corrected inputs with one measurement per slip-rate section.

| Prior | χ² (fit check 126 ± 16) | Total rate (/yr) | Bins vs targets: median (range) | Ruptures > 1e-5 / > 1e-4 per draw | Ruptures carrying 90% |
|---|---|---|---|---|---|
| Spike-and-slab (slab floor 1e-10) | 140 ± 12 | 0.0218 | +1.9% (+0.4% to +7.3%) | 99 / 45 | 47 |
| θ = 1 | 142 ± 12 | 0.0218 | +2.0% (0% to +10%) | 107 / 44 | 50 |

With `MFD_UNCERTAINTY = 'two-measurements'` (in its own `RUN_DIR`): spike-and-slab χ² 137 ± 12, total rate 0.0230, bins +7.9% (+2.4% to +16%), 97 / 46 ruptures above 1e-5 / 1e-4, 46 carrying 90%; θ 137 ± 12, 0.0228, +6.0% (−18% to +35%), 108 / 45, 50. θ's top four bins (8.2–8.6) then range from −18% to +35% against spike-and-slab's +5% to +16%: with the bins' bands three times wider, how the rate splits among the largest bins depends on the prior, while their sum and the products agree.

Spike-and-slab switched on about 380 ruptures per draw. With `SLIP_SECTIONS = 'none'` (each subsection its own measurement, in its own `RUN_DIR`) the earlier values return: in short tests (4 chains × 2,400 sweeps), χ² 227 ± 16 for spike-and-slab and 236 ± 15 for θ against 446 ± 80, bins 3–4% over target, 93 / 44 and 101 / 43 ruptures above 1e-5 / 1e-4, and 44 and 45 carrying 90%. The products did not depend on spike-and-slab's slab floor anywhere tested, from 1e-12 to 1e-5.

Also check that cleaning printed the slip-rate sections (68), the MFD bins' uncertainties (10.0% at M7.0–7.1, as given) and the fit check (126 ± 16), that each run's prior folder holds the three tables, `rate_vectors.csv` (18,648 rows, penalised ruptures 0), the summary figure and its text (with or without draws), the rates-by-magnitude figure and its table (`fig_rate_magnitude.png` and `.txt`), the slip-rate figure and its table (`fig_slip_rates.png` and `.txt`), the paleo-rate figure and its table (`fig_paleo_rates.png` and `.txt`), and the session log; with `REPRESENTATIVES` above 0, `rate_vectors.csv` also has columns `rep_01` to `rep_20` and the folder holds `representatives.csv` and `select_report.txt`; and that `PRIOR_KIND = 'logbox'` stops with a clear message.

## Run lengths, timing and storage

The very conservative preset is the production setting: three replicates of 16 chains.

| Preset | Spike-and-slab and θ |
|---|---|
| Quick | 4 chains × 2,400 sweeps, 2 replicates |
| Conservative | 8 × 20,000, 2 replicates |
| Very conservative | 16 × 100,000 (burn 20,000, thin 16), 3 replicates |
| Very conservative on the M2 | θ: 4.1 h; spike-and-slab: 1 h 20 min (both measured) |

The notebook's `PRESETS` gives each preset's burn-in and thinning (and keeps shorter settings for `'binbox'`). Spike-and-slab's time was measured with `MFD_UNCERTAINTY = 'two-measurements'`, stored draws and 50 representatives: 60.5 min of sampling for the three replicates (18.5 min each, 0.011 s per sweep for 16 chains on 4 threads), then 20 min of checks and tables, 19.5 of them in the convergence and table cells, which read the stored draws (17.5 GB) when they exist. Its sweeps cost about a fifth of θ's in the quick runs.

- **Draws (`OUTPUT_DRAWS = True`):** 5.9 GB per run at 2e4; 68 GB at 211,000 ruptures.
- **Without draws:** every notebook check, table, rate vector and the summary figure come from the running summaries; only `compare_runs.py`, `compare_priors.py` and the top-contact check need stored draws.
- **Kept draws (`REPRESENTATIVES` above 0):** 1,000 whole draws per run whatever `OUTPUT_DRAWS` says, 73 MB per run at 2e4 and about 2 GB at 5e5 (float32), plus their pooled copy in `rate_vectors.npz`.
- **Memory:** the running log-rate histogram takes 1,651 bins × 4 bytes per rupture, about 1.4 GB at 211,000 ruptures.

The runs are long because short tests suggested the slowest ruptures mixed poorly under both priors. In the very conservative θ run the 5th-percentile effective draws were four times the target, so χ²'s R̂, not the per-rupture target, sets the length.

## Convergence

The notebook's verdict says `PASS` when the targets below are met, reports the slow-mixing minority separately, and compares the replicates rupture by rupture.

| Target | Quick | Conservative | Very conservative |
|---|---|---|---|
| Effective draws, 95% of ruptures' log rates | 100 | 1,000 | 10,000 |
| Effective draws, total rate | 200 | 2,000 | 20,000 |
| R̂ of χ² and the total rate, below | 1.01 | 1.01 | 1.005 |

For spike-and-slab the verdict also needs R̂ below 1.01 for the number of ruptures on and every bin's fraction on, and at least a fifth of the total rate's effective draws for each fraction. A `CHECK` names the targets missed: run longer, and if χ²'s or the total's R̂ stays high, try dispersed starts.

- **Diagnostic cutoff:** per-rupture R̂ and effective draws use log rates with everything below 10^`FLOOR` (1e-10) counted as equal. Runs scored at another cutoff are not directly comparable: scored at 1e-12, the same θ draws flag about a quarter more ruptures as slow-mixing (259–288 instead of 195–243 in quick runs), while spike-and-slab's figures barely change (REPORT.md, "Floors and cutoff").
- **Noise baseline:** the worst R̂ that independent draws would give, about 1.0003 at the very conservative preset and 1.015 at the quick one. The table shows R̂ to four decimals so it is visible.
- **Slow-mixing minority:** ruptures with R̂ above 1.01, or above the baseline if higher, reported with their share of the total rate and flagged in the rupture table (`slow_mixing`). Expected under both priors: with one measurement per section, quick runs had 59–63 (0.3%) under spike-and-slab, carrying 16–18% of the total rate, and 195–243 (1.1–1.3%) under θ, carrying 23–24%. In the very conservative θ run (counting each subsection), 821–856 ruptures (4.5–4.7%) sat above the baseline and the worst R̂ was 1.35–1.38. Longer runs help them only slowly (doubling the sweeps would bring 1.36 to about 1.19).
- **Replicate agreement:** standardized differences of each rupture's mean log rate should have sd near 1, about 0.27% beyond 3, and about the expected number beyond 4 (about 1.2 for 18,000 ruptures); the very conservative θ run gave 0.988–0.996, 0.22–0.25% and none. Ruptures that never rise above the cutoff in either run (spike-and-slab's always-off ruptures) are left out and counted.

Because each rupture's difference is scaled by its own Monte Carlo error, replicate agreement also tests whether the slow ruptures' stated errors hold.

## Checklist after each run

Work through these in order; the first six come from the notebook's output, the rest from the result files.

1. **Log:** search the run's `workflow_log_<date>-<time>.txt` for "Warning", "Error" and "Traceback"; in Jupyter it also shows each cell's duration.
2. **Verdict:** `PASS` at the conservative and very conservative presets; for a `CHECK`, follow its named reasons.
3. **Slow-mixing minority:** note its count and its share of the total rate. If it carries a material share of a product you will use, check those ruptures' `slow_mixing` flags and effective draws before relying on them.
4. **Replicate agreement:** as in Convergence.
5. **Fit check:** χ² against the value cleaning prints, 126 ± 16 on the corrected inputs; in quick runs spike-and-slab and θ sat about 1 sd above it. With `SLIP_SECTIONS = 'none'` it is 446 ± 80, and they sat 2.6–2.8 sd below it.
6. **Row-block fit** (after the fit check): each block's χ² per independent measurement near 1; in quick runs 1.04 for the slip-rate sections, 1.00 for the bins and 1.29 for the paleo rows (z +0.2, 0.0 and +1.3). The bins' misfit is nearly all spread, the model's freedom, so it says little about their uncertainties. The "beyond the freedom" line compares the blocks' residuals once the freedom is set aside (slip rates 1.05 ± 0.21, paleo 1.51 ± 0.45, overall 1.20 in quick runs): a block further from the overall than its ± would point at its uncertainties. The bins' width is `MFD_UNCERTAINTY`'s, and their misfit, nearly all spread, cannot test it; the report's MFD line says so.
7. **Bin totals:** a median about 2% over their targets (at most 10%), near the best fit's 1–5%. Much larger departures would point to pressure from the prior or tension in the data.
8. **Top contact** (needs stored draws; `compare_priors.py` prints it over every draw; measured counting each subsection): under θ about 45% of draws have a rupture above its target, by up to about two of the bin's uncertainties; under spike-and-slab about 80%, by up to about three. Much more would point to tension in the data.
9. **Summary figure:** the replicates' population curves overlap; the rate-by-rate points sit on the diagonal; note the share of draws below the floor.
10. **Rates by magnitude** (`fig_rate_magnitude.png`, with its table in `.txt`): the order check should say the median number of subsections per rupture rises with magnitude (on the corrected inputs from 3 in the lowest bin to 113 in the highest, rank correlation +1.00); otherwise check the bin order before reading magnitudes off the figure. Each orange dot (the bin's total of the means) sits against its target line: in quick runs about 2% above (at most 10%). The bottom band (means below 1e-10, including exactly 0) held 32 ruptures under spike-and-slab and none under θ; shorter runs leave more there, because a rarely switched-on rupture's mean comes only from its visits.
11. **Slip rates** (`fig_slip_rates.png`, with its table in `.txt`; in units of each section's measurement uncertainty): each section's mean prediction (blue) should lie near the 1:1 line, most within the grey band of one uncertainty, with its subsections in grey behind it. In quick runs the mean model's χ² over the 388 slip-rate rows, as the likelihood weights them, was 51 under spike-and-slab and 53 under θ (72–75% of sections within one uncertainty, 97% within two); the best fit's is 17.0. This is the χ² of the posterior mean, below the mean over draws used in the fit check. 18 sections (72 subsections) are predicted below a quarter of their targets (the best fit: 6 sections), all with targets only 0.6–2.4 uncertainties above zero, so each stays within 2.3 uncertainties of its target; check them against the fault model.
12. **Paleo rates** (`fig_paleo_rates.png`, with its table in `.txt`; log-log, in units of each row's uncertainty, so a dot's distance from the 1:1 line is the ratio of prediction to target): most dots should lie within the bands, with the posterior 16–84% bars of their predictions reaching the 1:1 line. In quick runs the mean model's χ² over the 40 paleo rows was 34 under both priors (78% within one uncertainty, 92% within two; the best fit's 13.1), and most sites were predicted somewhat below their targets (median ratio 0.82–0.85), as in the best fit (0.94, 34 of 40 below). Furthest below were rows 860, 835, 853 and 861, about two uncertainties low. The best fit also leaves rows 860 and 861 at about half their targets, but nearly matches row 853, whose posterior median prediction is about 0: the ruptures that would reach it are off in most draws.
13. **Rate vectors:** 18,648 rows, penalised ruptures 0, quantiles in order. Under spike-and-slab and θ most medians and modes are 0, so sum the means, never the medians or modes.
14. **Representative draws** (`select_report.txt`): the weighted means of the total rate and the bin totals within 2% of the exact means; the subsection totals' weighted means within 10%, and their weighted 16%, 50% and 84% points within 15% of those over every recorded draw, for 90% of subsections (and of bins); every representative's χ² and total rate inside the posterior's 1–99% range; no zero weight. These tolerances are proposals to revisit with the hazard team. The quick preset usually misses them; at the conservative preset and above, if 20 representatives miss and 40 meet them, use 40. In the first conservative test (spike-and-slab), both missed the bin and subsection tolerances by similar margins (REPORT.md, "Representative draws" under open items). With `'two-measurements'` the bins' posterior spread is about 30% per bin, three times the inputs', so 20 representatives match the bins' means less closely (the worst bin 16% off at the quick preset).
15. **Keep together:** the log, `workflow_config.json` and the result files of each result set.

## Sensitivity and regression checks

The strongest evidence that a result is data-determined is that it survives a change of prior; the strongest protection against breaking the code is a fixed-seed comparison.

- **θ against spike-and-slab:** the main robustness check, worth running for every production result set. Compare their results product by product (subsection and bin totals, exceedance probabilities): two different treatments of the lower end agreeing is the main evidence that a product is data-determined. In quick runs with one measurement per section they agreed on the fit, the total rate, the bins and the population.
- **Slip-rate sections:** run the notebook again with `SLIP_SECTIONS = 'none'`, in its own `RUN_DIR`, to see what counting each subsection as its own measurement changes. It reproduces the earlier results exactly (for spike-and-slab χ² near 227 against 446 ± 80, where the default gives 140 against 126 ± 16). If the fault model's sections are available, a run with them as a mapping file checks the automatic detection.
- **MFD uncertainty:** run the notebook again with `MFD_UNCERTAINTY = 'two-measurements'`, in its own `RUN_DIR`, and, for a sense of scale, with a number (0.2 or 0.5, the M7.0–7.1 bin's relative uncertainty). In quick runs `'two-measurements'` raised the bins a median 8% above their targets (2% as given), the total rate by 5% and subsection totals by a median 4% (5–95%: −2% to +18%), and improved the paleo fit (χ² per measurement 1.22 against 1.29), with exceedance probabilities and the population within Monte Carlo noise; spike-and-slab and θ then disagree on how the four largest bins share their rate, though not on its sum. Single draws can then leave a whole bin empty: some bin fell below 1% of its target in 4.2–4.3% of the draws of a very conservative spike-and-slab run with `'two-measurements'`, and in none of the 16,000 draws of a conservative run as given, which matters wherever single draws are used, such as the representatives (REPORT.md, "MFD uncertainty").
- **Slab floor:** run the notebook again at another `FLOOR` (−8, say), in its own `RUN_DIR`. With one measurement per section, moving the slab floor from 1e-12 to 1e-10 changed every product by no more than Monte Carlo noise (subsection totals a median 0.1%, 5–95% −9% to +11%; exceedance at 1e-5 a median 0.02 in log₁₀) and only the number of ruptures switched on at undetectable rates, from about 497 to 379 per draw; counting each subsection, floors up to 1e-5 gave the same products, with about 53 fewer ruptures on per decade. With one setting, the second run's diagnostic cutoff moves too, so compare the products, not the diagnostics.
- **Dispersed starts (`SA_STARTS`):** the most important untested item. Start one replicate from simulated-annealing solutions or perturbed best fits; its χ², totals and replicate agreement should match the replicates started from the best fit. A mismatch would mean separate modes, which the current replicate check cannot see.
- **Regression baseline after code changes:** keep one quick-preset result set per prior (spike-and-slab and θ) from fixed seeds. After any change, rerun with the same seeds and compare `rate_vectors.csv` and `bins_posterior.csv`; they should match byte for byte unless the change was meant to alter sampling. Then run `--selftest`.
- **Hazard at very low probabilities:** if hazard will be used at annual exceedance probabilities of 1e-5 or below, test the hazard itself under both priors, and at two slab floors. The prior sets the population of rates below detection, which none of the products above measures.

## Before 5e5

Measure at an intermediate size first, the 211,000-rupture model or about 1e5 ruptures, at the quick and then the conservative preset, before committing to very conservative 5e5 runs.

2. **Preliminaries on the new inputs:** row groups, slip-rate sections, best fit and the collective scale against the floor (1e-10; with more ruptures per subsection the scale falls, and may come close to it). The best fit and the near-duplicate search are the steps most likely to need the sparse versions listed as open items.
3. **Time per sweep:** cost should grow roughly with the nonzeros of G; the intermediate run gives the factor to extrapolate.
4. **Memory and files:** several stored arrays grow with draws × rows or with ruptures (estimates, assuming about 2,450 rows and 2,400 subsections at 5e5, very conservative preset):

| Item | 2e4 | 5e5 (estimate) |
|---|---|---|
| Running log-rate histogram (memory) | 120 MB | 3.3 GB |
| Per-draw predictions in each run file | 285 MB | about 1.6 GB per run |
| Per-draw subsection totals in the running summaries | 250 MB | about 1.5 GB per run |
| Stored draws, if `OUTPUT_DRAWS = True` | 5.9 GB per run | about 160 GB per run |

The last row means `OUTPUT_DRAWS = False` at 5e5; the per-draw predictions and totals may need float32 storage or thinning.

- **Swap moves:** 338,000 near-duplicate pairs at 2e4 (292,000 counting each subsection: down-weighting the slip-rate rows makes more columns alike); at 5e5 there may be tens of millions (estimate), hence the filtered swap moves listed as open.
- **Per-draw 90%/99% counts:** about 0.27 s per recorded draw at 5e5, some 20 minutes per very conservative run.
- **θ's depths:** they grow with each bin's rupture count (n_j / ln 10 decades), so θ may mix more slowly at 5e5; check the effective draws at the conservative preset first. Spike-and-slab has no depths; its fractions on may need more sweeps as bins grow.

## Known weak spots and next steps

The gaps that matter most are spike-and-slab's missing long run under the defaults, the untested dispersed starts and the slow minority under both priors.

- **Spike-and-slab, very conservative:** not yet run on the corrected inputs under the defaults. With `'two-measurements'` it took 1 h 20 min on the M2 and passed in all three replicates, with no more than one rupture above R̂ 1.01 (35–42 above the noise baseline, the worst 1.013) and replicate agreement as expected. Run it under the defaults, and compare with θ product by product.
- **The existing very conservative θ run:** re-run the configuration, helper, cleaning and loading cells, put `MAIN = RUNS = [f'theta/main_rep{r}' for r in range(3)]` and `t0 = time.time()` in place of the runs cell, then run the convergence, results and figure cells. That gives the new verdict, the slow minority's share of the rate and the summary figure without re-sampling. Its 90%/99% counts will read "not available": its summaries predate them. It was run counting each subsection, so its χ² and products are for that setting.
- **Diagnostics and the cutoff:** compare slow-mixing counts only between runs scored at the same cutoff (Convergence above). The very conservative θ run kept no draws, so it can only be scored at the cutoff it was run with.
- **The slow minority:** a few percent of ruptures under θ, and fewer under spike-and-slab, switch on and off slowly at any affordable length. The remedy is moves that pass the active role between alternative ruptures, not longer runs.
- **Top contact without draws:** not recorded in the running summaries; adding a per-draw maximum of rate over target would make the check available at 5e5.
- **Slab-floor evidence:** one short run per floor on the 2e4 inputs; differences smaller than the seed noise (about ±10% on subsection totals) would go unseen.
- **Fit check:** with one measurement per section, spike-and-slab and θ pass it (about 1 sd above). It assumes each section's slip rate is uniform along it, rests on the provisional uncertainties, and counts the bins one each whatever their width; review these before treating it as a hard test.
- **MFD uncertainty level:** a modelling judgment the data cannot make, since the bins' misfit is the model's freedom. The default keeps the inputs' design value, under which the bins pin the MFD's total rate to ±2.8%, leaving its spread to the tree's N and b branches; `'two-measurements'` applies the design value to the MFD as a whole instead. A correlated covariance (the two measurements' within-branch uncertainty plus a model error per bin) would be the fuller treatment. Looser levels also let single draws leave whole bins empty (about 4% of draws with `'two-measurements'`; at `MFD_UNCERTAINTY = 1` an empty bin costs at most about 1 in χ², not yet measured).
- **Slip-rate sections:** detected from identical targets and coefficients; a mapping file from the fault model would settle the 32 sections that share a target with another and the 6 identical-target neighbour pairs that share no rupture. A geodetic slip-rate model, varying along each fault, is left unweighted; if its errors are correlated along a fault, it would need a covariance.
- **Sparse sections:** with one measurement per section, both priors leave 18 sections (72 subsections) below a quarter of their targets; check them against the fault model.
- **Representative draws' tolerances:** the proposed subsection tolerances look out of reach for 20–40 representatives under spike-and-slab, whose subsection totals vary widely from draw to draw (a third with a coefficient of variation above 0.45); revisit them with the hazard team, or calibrate the weights on the subsection and bin totals directly.
- **Retired tools:** the HMC cross-check and the evidence tests are set aside (`cru_retired_scripts.zip`); restore them if either is wanted, or compute evidence by hand with a few tempered runs.

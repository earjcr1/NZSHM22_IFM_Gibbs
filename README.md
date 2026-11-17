# Posterior sampling of NZSHM22 rupture rates

This workflow samples the Bayesian posterior of the annual rate of every rupture in an NZSHM22 rupture set, from the same constraints and a stated prior, with an exact Gibbs sampler. It is an alternative to the NZSHM22 inversion summarized below.

## Predecessors, and why this workflow

The 2022 New Zealand National Seismic Hazard Model (NZSHM22) builds its fault-based earthquake rates by an inversion (Gerstenberger et al., 2022, 2024) that follows the UCERF3 recipe (Field et al., 2014; Page et al., 2014) in OpenSHA (Milner and Field, 2023): each rupture in a rupture set (411,270 for the crustal faults) gets an annual rate such that the predicted slip rates, paleoseismic event rates and regional magnitude-frequency distribution (MFD, from a regional rate N of M ≥ 5 earthquakes and a b-value) match their targets: a linear system *G m* ≈ *d* with far more unknowns than independent data. NZSHM22 solves it by fast simulated annealing (Szu and Hartley, 1987) from all rates at zero, over about 3 billion iterations. Each logic-tree branch gets one solution, with a few hundred to about 1,700 crustal ruptures at non-zero rates.

That leaves three things this workflow sets out to supply:
- **The range of rates the data allow, carried into hazard.** The data fix sums of rates far better than single rates (here *G* has rank 2,372 for 615,700 unknowns). Each annealing run lands on a different model depending on its seed (Gerstenberger et al., 2022), but a branch keeps one. Here the whole set is sampled as a Bayesian posterior: every rate and derived quantity gets a mean and a spread, and weighted representative models (step 8) carry it into hazard.
- **Freedom from the annealing’s settings, weighting and nuances.** Combining slip rates, the MFD and paleo rates into one objective needs a weight on each, set subjectively for the subduction interfaces and, for the crustal faults, rebalanced as the inversion runs until each fits equally well relative to its uncertainties (by its own chosen misfit measure): either way a choice, to be treated as epistemic uncertainty (Gerstenberger et al., 2024). Here each datum counts by its own uncertainty in the likelihood, with no weights between data types; with MFD_GR even the MFD bins’ uncertainties come from N and b rather than a set level. The annealing’s perturbations, cooling schedule, start and iteration count also shape its solution; a posterior does not depend on how it is sampled (step 6).
- **Ability to fit the constraints within their own inherent PDFs rather than as Gaussian misfits in rate.** The inversion scores every constraint as a Gaussian in rate: each slip rate with one symmetric sd; each paleo site with a symmetric sd about its best estimate, though its recurrence-interval PDF is lopsided (its one-sigma range a median 1.9 times longer below than above) and a Gaussian caps the cost of predicting no events there; each MFD bin with a set relative uncertainty. Here each is scored in the space its uncertainty was worked out in: slip rates by split normals from the NZ Community Fault Model (Seebeck et al., 2023), corrected for their timeframes (Coffey et al., accepted); paleo sites by −2 ln of their recurrence-interval PDFs in years; MFD bins half in rate and half in log10, or, with MFD_GR, by their own PDFs from N and b. The sampler evaluates each exactly.

## The Gibbs sampler, and why it suits this problem

A Markov chain Monte Carlo (MCMC) sampler draws a sequence of models whose distribution settles on the posterior: the models that fit the data, each weighted by how well it fits and by the prior. A Gibbs sampler does it one unknown at a time: it draws each rupture’s rate from its exact distribution given all the other rates and the data, then moves to the next rupture; one sweep updates all 615,700 rates. Three things make that efficient here:
- **Each update is exact and cheap.** A rate enters the predictions linearly, so with the others fixed its misfit is a quadratic in it (or, for the terms scored in log space, a smooth one-dimensional function), and one rupture’s column of *G* holds only 47 nonzero entries on average. With each chain’s residuals kept current, an update reads one sparse column and draws from a one-dimensional distribution exactly: there is no step size to tune and no update wasted by a step that was too large.
- **Hard constraints become simple limits.** Positivity, the monotonic MFD and the plate rate each become, for one rate with the others fixed, a range of allowed values, so every draw keeps them exactly instead of paying a penalty for breaking them.
- **Switching a rupture on or off is part of its update.** Under spike-and-slab each update decides exactly whether the rupture is off or on, so the chains move between sparser and denser models as the data allow.

**Why in log rate.** The rates that matter span 6.5–7.8 decades, from the slab floor of 10<sup>−10</sup> per year up to each magnitude bin’s target. Each rupture that is on is updated by a slice sampler (Neal, 2003) in the logarithm of its rate, so its steps are proportional to its own size: a rate near 10<sup>−8</sup> per year and one near 10<sup>−3</sup> move equally readily in relative terms, where any fixed step in linear rate would be far too large for the first or far too small for the second. The prior is uniform in log rate as well, so the update works in the variable the prior is flat in. Log space also matches what matters downstream: hazard responds to rates over orders of magnitude, and the MFD and paleo misfits are themselves scored partly in log space.

**Where it is slow.** One rate at a time is slow where rates are strongly correlated. Near-duplicate ruptures (4,049,054 pairs with cosine similarity above 0.99) can trade rate between them almost freely, which single-rate updates do only in small steps; swap moves, which exchange two such ruptures’ rates in one step, address this. The slowest quantities are the fractions of ruptures on in the highest magnitude bins, which change only as individual ruptures switch (step 6).

## Other ways to explore the model space

The sampler suits this problem’s shape: hundreds of thousands of unknowns, a sparse linear forward model, exact zeros and hard limits. How other approaches would fare, judged from how they scale (none was run here):
- **Simulated annealing** (NZSHM22, UCERF3) is an optimizer: the spread of many runs reflects the search (seed, start, stopping point), not a stated probability.
- **Random-walk Metropolis** must step small enough for the most tightly fit direction; with 615,700 unknowns on scales many decades apart, nearly every proposal is rejected or barely moves.
- **Hamiltonian Monte Carlo and NUTS** (Neal, 2011; Hoffman and Gelman, 2014) move all rates along the gradient, at about a Gibbs sweep’s cost, and excel on smooth, correlated posteriors. But spike-and-slab’s exact zeros are discrete, which they cannot sample; the hard limits need reflections or transformations (log rate makes a funnel near the floor); and the stiffest direction (sums fit within a few percent) sets a step size that leaves the loosest (single rates unresolved over decades) thousands of steps from mixing. The discreteness is the single point of slowdown.
- **The Neighbourhood Algorithm** (Sambridge, 1999a, 1999b) partitions parameter space into Voronoi cells around the models tried: effective for tens of unknowns, unable to cover 615,700.
- **Ensemble samplers** (Goodman and Weare, 2010) need more walkers than twice the number of unknowns. **Parallel tempering** helps with separated modes, around an inner sampler such as this one, at several times its cost.
- **Analytic and approximate posteriors** (Gaussian, Laplace, variational) are fast, but the exact zeros, hard limits and long tails toward zero break their assumptions.
- **This Gibbs sampler**’s single point of slowdown is correlation its one-at-a-time updates cannot cross in one move: near-duplicate ruptures (handled by the swaps) and the fraction on in each bin. Its cost per sweep grows only with the nonzero entries of *G*, so it scales to the full rupture sets.

*References.* Coffey, G. L., et al. (accepted), Reassessing geologic fault slip rates and their uncertainties in Aotearoa New Zealand, *Bull. Seismol. Soc. Am.* Field, E. H., et al. (2014), *Bull. Seismol. Soc. Am.* 104, 1122–1180 (UCERF3). Gerstenberger, M. C., et al. (2022), *GNS Science Report* 2022/47, doi:10.21420/2EXG-NP48; (2024), *Bull. Seismol. Soc. Am.*, doi:10.1785/0120230165 (the NZSHM22 seismicity rate model). Goodman, J., and J. Weare (2010), *Commun. Appl. Math. Comput. Sci.* 5, 65–80. Hoffman, M. D., and A. Gelman (2014), *J. Mach. Learn. Res.* 15, 1593–1623. Milner, K. R., and E. H. Field (2023), *Bull. Seismol. Soc. Am.*, doi:10.1785/0120230122. Neal, R. M. (2003), *Ann. Stat.* 31, 705–767 (slice sampling); (2011), in *Handbook of Markov Chain Monte Carlo*, CRC Press, 113–162 (HMC). Page, M. T., et al. (2014), *Bull. Seismol. Soc. Am.* 104, 1181–1204 (the UCERF3 grand inversion). Seebeck, H., et al. (2023), *N. Z. J. Geol. Geophys.* (the NZ Community Fault Model, version 1.0). Sambridge, M. (1999a, b), *Geophys. J. Int.* 138, 479–494 and 727–746. Szu, H., and R. Hartley (1987), *Phys. Lett. A* 122, 157–162.

## This run, step by step

The sections below follow the notebook `cru_workflow.ipynb` step by step for one run; each figure is one this run wrote.

## 1. Inputs and configuration

The inputs are the inversion matrices in `NZSHM22_Matrices-630k`: `cru_A.txt` (*G*, 16,851 rows by 630,125 ruptures, 29.4 million nonzeros) and `cru_D.txt` (*d*). The OpenSHA solution for the rupture set (`NZSHM22_InversionSolution-630k`) supplies the fault geometry, the named faults and a reference model, drawn in black on every figure. One configuration cell sets everything; this run used:

| setting | value | meaning |
|---|---|---|
| SLIP_SECTIONS | 'auto' | how the slip-rate rows count (’auto’: each fault section’s slip rate as one measurement, however many subsections it is copied to) |
| MFD_UNCERTAINTY | 'as-given' | the MFD bins’ uncertainty level (’as-given’: as the inputs give them) |
| MISFIT_SLIP | 'linear' | the slip rates scored linearly, by their split normals |
| MISFIT_MFD | 'half' | half linear, half log: half their Gaussians plus half the Gaussian in log10, with the same sds |
| MISFIT_PALEO | 'years' | the paleo rows scored by each site’s recurrence-interval PDF in years |
| CFM_SLIPRATES | True | each slip rate a split normal in mm/yr from the NZ Community Fault Model corrected for its timeframe (CFM_TIMEFRAME) and scaled as a whole by 0.8, the slip sums too |
| CFM_TIMEFRAME | True | each fault’s slip rate corrected for its timeframe (Coffey et al.): a split normal through the corrected distribution’s 16%, 50% and 84% points |
| CFM_TEMPLATE_RHO | 1 | faults of quality codes 2–5 with identical SR_pref / SR_min / SR_max correlated, each weighted 1/√(1 + (n − 1) × 1) in a group of n |
| CFM_RATE_LINKS | True | also faults whose rate the CFM borrowed from another (named as the source, or an identical triple nearby or of the same name), each weighted 1/√(1 + m × ρ) for its m correlated faults |
| SLIP_RATE_FACTOR | 0.8 | the assumed fraction of each fault’s slip rate the ruptures take up, scaling each slip-rate distribution as a whole, target and sd (the inputs’ own) |
| MFD_MONOTONIC | True | no magnitude bin’s total above its lower neighbour’s where the targets do not rise (a hard constraint, gibbs.py --mfd-monotonic) |
| SLIP_ALONG_RUPTURE | 'uniform' | each rupture’s slip spread along its subsections: the same average slip on each, as the inputs have it |
| CROSS_SECTIONS | 'all' | the strips of cross-sections perpendicular to the plate motion that SLIP_SUM and PLATE_RATE use, 8.5 km wide over all of New Zealand |
| SLIP_SUM | True | the slip along the plate motion (each fault’s slip vector projected on it) summed along each cross-section perpendicular to it, against the targets’ sums (a soft term) |
| SLIP_SUM_L0 | 100 | the summed slip rates’ weight: each 100 km of the plate boundary counts as one measurement |
| PLATE_RATE | 0.8 | each strip’s share of the plate rate at most 0.8 (a hard constraint, gibbs.py --plate-rate; PLATE_RATE = True: at SLIP_RATE_FACTOR) |
| FLOOR | -10 | the lowest rate an active rupture can take, 10<sup>-10</sup> per year |
| RUN_MINUTES | 600 | the run’s length; 1 replicate × 16 chains × 178,000 sweeps (burn-in 5,000, a draw kept every 58), planned from a calibration (0.171 s per sweep) to take about 538 min in all |
| TARGETS | 'production' | the convergence verdict’s targets (check_convergence.py) |
| REPRESENTATIVES | 100 | weighted representative models selected for hazard |

## 2. Cleaning and diagnostics

Cleaning (`preliminaries.py`) turns the raw system into the problem the sampler sees. The 17 penalty rows with very large entries fix 14,418 ruptures at zero, and 7 soft zero-forcing rows fix 7 more, so 14,425 ruptures are removed (at rate 0 in every output). 14,432 rows that then hold no remaining rupture add a constant χ² of 292.3 that no model can change, and are dropped. What remains is 2,395 rows by 615,700 ruptures: 2,312 slip-rate rows, 18 MFD bins and 65 paleo rows. The slip-rate rows form 551 fault sections; each section’s rows are divided by √k for its k subsections, so a section’s slip rate counts once rather than k times. The MFD bin targets run from 0.00034 to 0.0064 per year and sum to 0.0417 per year, with relative uncertainties of 9.3% to 19.4%. If the stated uncertainties are right, a model drawn from the posterior should have χ² near 382 ± 26: this is the *fit check*, for the Gaussians; with the paleo rows scored by their sites’ recurrence-interval PDFs, each site’s own expected misfit replaces its Gaussian share in the fit check used below (section 7). The diagnostics (`prelim.png` in the run folder, not shown here) show how weakly individual rates are determined: *G* has rank 2,372, and the median diagonal of the resolution matrix is 0.002, so the data fix sums of rates far better than single rates. There are also 4,049,054 pairs of near-duplicate ruptures (cosine similarity above 0.99), which the sampler handles with swap moves.

## 3. The best fit (no prior)

Cleaning also computes the best nonnegative least-squares fit (Lawson–Hanson NNLS, 58 s). It reaches χ² = 158.3 over the 2,395 rows with only 869 ruptures above zero: 94.4 from the slip rates, 15.3 from the MFD bins and 48.7 from the paleo rows. Its total rate is 0.0458 per year, +9.8% against the bin targets, with every bin between +5.4% and +17.4%. A fit chooses among all the rates to match far fewer numbers, so its χ² lies well below the fit check; it is a reference with no prior and the starting point of the chains, not a sample of the posterior. Its fit to each data type is drawn in the run’s `nnls/` folder (not shown here); the posterior’s follows in step 9.

## 4. The model: prior, likelihood and constraints

**Prior.** Each rupture is either off, with a rate of exactly zero, or on, with a rate that is uniform in log rate from 10<sup>-10</sup> per year up to its magnitude bin’s target rate (spike-and-slab). The top is soft, a Gaussian shoulder as wide as the bin’s own uncertainty, since no single rupture can carry more than its bin’s total. The fraction of ruptures switched on is estimated separately in each bin (uniform prior on each fraction), so the data decide how many ruptures are active, and the prior puts no pressure on the bin totals. **Likelihood.** Gaussian, exp(−χ²/2) with the stated uncertainties. **Positivity.** In addition, each subsection’s slip rate, each paleo rate and each MFD bin must be above zero, so at least one rupture is on in each, with no other limit: a hard constraint. **The paleo misfit.** The paleo rows leave the Gaussian likelihood: each is scored instead by its site’s recurrence-interval PDF from the paleoseismic analysis, −2 ln of its density in years, measured from its minimum, a curve through the site’s 2.275, 15.865, 84.135 and 97.725% points with its mode at 1/bestEstimate exactly (the rupture set’s paleo_constraint_data.json); 2 site(s) fit their quantiles to worse than 0.05 decades. The ranges are lopsided, and the Gaussian caps a site’s misfit, so this penalizes paleo rates predicted well below their targets far more; it is kept exactly by Metropolis steps. **Misfit by data group** (MISFIT_SLIP 'linear', MISFIT_MFD 'half', MISFIT_PALEO 'years'). The slip rates are scored by their split normals; the MFD bins by half their Gaussians and half a Gaussian in log10 with the same sds; the paleo rows by their recurrence-interval PDFs in years. A half-and-half group’s score is the geometric mean of its two densities, so each measurement still counts once; each log part is kept exactly by Metropolis steps. **Slip rates from the CFM.** Each subsection’s slip-rate row is rebuilt from its parent fault’s line of the NZ Community Fault Model (SR_pref, SR_min, SR_max in mm/yr): a split normal in the slip rate itself, scaled as a whole by 0.8 (SLIP_RATE_FACTOR, the assumed fraction of the slip rate the ruptures take up): each fault’s numbers corrected for its slip-rate timeframe (CFM_TIMEFRAME; Coffey et al., accepted: the range widened for the few earthquakes a short timeframe captures), and the split normal fitted to that distribution’s 16%, 50% and 84% points (or, for the most right-skewed faults, in log10, whichever is closer in Kullback–Leibler divergence), times 0.8. The inputs’ Gaussian has one sd for both sides; here each side has its own, and the misfit is evaluated exactly in every update. **A monotonic MFD.** Wherever a magnitude bin’s target is not above the next lower bin’s, its total rate may not be either, in every model. **The plate rate.** In each 8.5-km strip of cross-sections perpendicular to the plate motion over all of New Zealand, the crustal faults must take up at most 0.8 of the Pacific–Australia plate rate, each fault’s slip vector projected on the local plate motion and averaged over the strip’s cross-sections (`plate_rate.txt`). **Summed slip rates.** A soft term in the misfit: along every cross-section perpendicular to the plate motion over all of New Zealand, the model’s slip along the plate motion added up (each fault’s slip vector projected on it) should match the targets’ added up, within that sum’s uncertainty, each subsection counting once in all; each 100 km of the plate boundary counts as one measurement, and it is recorded apart from χ² (`plate_rate.txt`).

## 5. Sampling

The sampler (`gibbs.py`, a C kernel run from Python) is an exact Gibbs sampler: it updates one rupture at a time, drawing its rate from its exact conditional distribution by a slice sampler in log rate, with an exact step that switches the rupture on or off. Each update is restricted to the rates that keep the hard constraints (no slip-rate or paleo or MFD bin row predicted at zero, the MFD’s monotonicity, the plate-rate limit). The summed slip rates are quadratic in the rates, so they enter the exact updates as extra rows of the misfit. Swap moves exchange the rates of near-duplicate ruptures (2,991,269 pairs; 0.2% of swaps accepted), and the fraction on in each bin is redrawn after every sweep. Every chain starts from its own draw (dispersed_starts.py): half from randomized best fits (the best fit to the data plus twice their stated noise), half from the prior (each bin’s fraction on uniform, rates log-uniform up to its target, the bin scaled around its target), each polished for feasibility in the sampler’s own model space so that it meets the hard constraints while staying over-dispersed; the starts are over-dispersed, so R̂ across the chains tests whether the run has forgotten where it started. Each of the 1 replicates ran 16 chains for 178,000 sweeps in 8.2 hours, keeping 2,982 draws per chain after burn-in: 47,712 recorded draws in all. No recorded draw broke the hard constraints. The runs keep running summaries rather than every draw, plus whole draws for the representative models and the figures.

## 6. Convergence

`check_convergence.py` judges each replicate against TARGETS 'production': verdicts main_rep0 CHECK (main_rep0: R-hat of the ruptures on or of a bin's fraction on 1.0148 (below 1.01 needed)). The run plan predicted, from an earlier run’s autocorrelation times, 74,038 effective draws for 95% of ruptures (41,096 achieved); 37,911 effective draws for the total rate (26,475 achieved); 1,313 effective draws for each bin’s fraction on (1,079 achieved). For χ² and the total rate the effective sample sizes are 20,400 and 26,480 per replicate, and for the rupture rates the 5th percentile is 41,100. About 1,476 ruptures are on in each draw. With so many rates even perfectly independent draws give a worst R̂ above 1, so R̂ is judged against that noise baseline (1.0006 here). A small minority of ruptures mixes slowly, mostly ruptures that switch on and off rarely: 71 ruptures per replicate with R̂ above the threshold (worst 1.107), carrying 11.0% of the total rate, with at least 436 effective draws each. 

| replicate | χ² (mean) | effective draws: χ² / total rate | worst R̂ (baseline) | slow-mixing ruptures (share of rate) | verdict |
|---|---|---|---|---|---|
| main_rep0 | 1032 | 20,400 / 26,480 | 1.1073 (1.0006) | 71 (11.0%) | CHECK |

## 7. Fit to the data

The posterior’s χ² as fitted is 1032.5 ± 35.6, against the fit check as fitted, 404.1 ± 27.2 (χ² as the likelihood scores the data: each group as MISFIT_SLIP, MISFIT_MFD and MISFIT_PALEO score it, with its own expected share; cleaning’s, for the Gaussians alone: 382 ± 26): above it (+23.1 standard deviations). A χ² above the fit check means some data are fitted worse than their uncertainties allow, or the uncertainties are too small. The row-block fit (`block_fit.py`) splits χ² by data type: a block’s χ² per measurement near 1 is fitted as its uncertainties say, below 1 more closely, above 1 worse. In the table, the “mean model” is the misfit of the posterior-mean prediction and the “spread” is the posterior variance of the predictions; the two add up to χ².

| row block | rows | measurements | χ² over the draws | per measurement | z | mean model | spread |
|---|---|---|---|---|---|---|---|
| slip rates (S) | 2,312 | 298.9 | 843.5 ± 35.4 | 2.82 | +24.7 | 401.0 | 442.5 |
| MFD bins (MFD) | 18 | 17.9 | 54.0 ± 10.1 | 3.02 | +6.1 | 45.4 | 8.7 |
| paleo (B8:var) | 65 | 87.3 | 135.0 ± 13.7 | 1.55 | +3.2 | 105.5 | 29.5 |
| total | 2,395 | 404.1 | 1032.5 ± 35.6 | 2.56 | +23.1 | 551.9 | 480.6 |

Each run also records the terms it adds beside χ², each on χ²’s scale and not counted in it. **The MFD misfit in log10**: 50.0 on average per replicate (5–95% of draws 36.5–64.9), in the likelihood weighted one half, beside half the bins’ Gaussian; χ² counts the whole Gaussian. **The paleo misfit**: 135.0 on average per replicate (5–95% of draws 113.2–158.2), in the likelihood in place of the paleo rows’ Gaussian, which χ² still counts. **The summed slip rates**: 29.0 on average per replicate (5–95% 22.2–37.4), against about 14.1 expected if the model matched the truth (one per 100 km of strip); `plate_rate.txt` lists the strips furthest from their target sums. **The plate rate**: every strip is inside the limits in 100.0% of the 1,000 kept draws, as a hard constraint requires; the strip closest to the limit, +59 (173.12° E, 41.81° S, mostly Hope and Awatere in the fault model), reaches 0.797 at the kept draws’ 97.5%.

## 8. Posterior tables, rate vectors and representative models

`make_tables.py` pools the replicates into per-rupture, per-subsection and per-bin tables and into rate vectors (`rate_vectors.csv`): for each of the 630,125 ruptures its posterior mean, median, mode and 2.3%, 15.9%, 84.1% and 97.7% points. Most ruptures are off in most draws, so their medians and modes are zero; sums such as fault or bin rates must be built from the means. The posterior-mean total rate is 0.0480 per year, +15.1% against the bin targets (NZSHM22_InversionSolution-630k: 0.0505); the bin totals sit a median +22.7% from their targets (+2.7% to +31.4%), against the best fit’s +13.2%. About 22% of each subsection’s posterior rate (the median) is carried by ruptures whose rates are individually below what the data can detect. `select_draws.py` then picks 100 weighted representative models from the 1,000 kept draws, clustering them by their rates per subsection and magnitude bin, for use in hazard calculations. With 100, the weighted mean total rate is off the exact mean by -0.0%, the worst bin by -1.7%, and 74% of subsection means are within 10% (the tolerance asks for 90%); the selection misses some tolerances, so the posterior-mean rate vector remains the exact input for mean hazard.

## 9. The posterior’s fit, data type by data type

The next three figures show the posterior’s fit to each data type. Each coloured dot is the posterior mean, and each faint x one of the kept draws, so where the draws gather shows darkest; open black squares are NZSHM22_InversionSolution-630k. **Rates by magnitude** (Figure 1): the posterior spreads each bin’s rate over many ruptures at low mean rates, where the best fit puts a few ruptures high and the rest at zero. The bin totals are +2.9% to +6.4% from their targets below M7.1 and +12.2% to +31.5% from M7.1 upward. **Slip rates** (Figure 2): 58% of the 551 sections’ posterior means lie within one measurement uncertainty of their targets and 87% within two (NZSHM22_InversionSolution-630k: 88% within one), with a median ratio of prediction to target of 0.88 (mean-model χ² 401.0; NZSHM22_InversionSolution-630k: 196.1). **Paleo rates** (Figure 3): the posterior underpredicts them, with a median ratio of prediction to target of 0.79; 66% of rows lie within one uncertainty and 91% within two (mean-model χ² 105.5; NZSHM22_InversionSolution-630k: 219.7).

![Figure 1](example/output_figs/fig_rate_magnitude.png)

*Figure 1.* Posterior: each rupture’s posterior mean rate against its magnitude bin (dashed: the slab floor; means below the axis, 0 included, on it), each bin’s target (black line) and total of the means (orange dot), and NZSHM22_InversionSolution-630k’s bin totals (black squares). Faint x’s: each kept draw’s bin totals. Lower panel: each bin’s total over its target. `fig_rate_magnitude.png`

![Figure 2](example/output_figs/fig_slip_rates.png)

*Figure 2.* Posterior: each fault section’s slip-rate target against its posterior-mean section mean (dots), in mm/yr, each target’s bar one sd either side. Faint x’s: each kept draw’s section means. Open squares: the reference. `fig_slip_rates.png`

![Figure 3](example/output_figs/fig_paleo_rates.png)

*Figure 3.* Posterior: each paleo row’s target against its posterior-mean prediction (dots), in events per year, each target’s bars its site’s own 15.9–84.1% and 2.3–97.7% ranges. Faint x’s: each kept draw’s predictions. Open squares: the reference. `fig_paleo_rates.png`

## 10. Faults one by one

`plot_slip_sections.py` looks at 33 named faults, ordered from the most isolated (Greendale, which shares no rupture with any other section) to the most connected (Hope, which shares ruptures with 147). **Figures 4–5** sum each one’s rupture rates by magnitude bin, its rate of events of each size; the circles are the posterior means, the x’s the kept draws’ sums, and the strip marked 0 shows by its opacity how often a draw has no event in the bin.

![Figure 4](example/output_figs/fig_slip_sections_ruptures_by_magnitude_1.png)

*Figure 4.* Rupture rates by magnitude, first part: each column’s ruptures’ rates summed by magnitude bin for each kept draw (x’s), their posterior mean (circles) and the share of draws with no event in each bin (the strip marked 0), and the reference’s sums (black outline). `fig_slip_sections_ruptures_by_magnitude_1.png`

![Figure 5](example/output_figs/fig_slip_sections_ruptures_by_magnitude_2.png)

*Figure 5.* Rupture rates by magnitude, second part. `fig_slip_sections_ruptures_by_magnitude_2.png`

## 11. Output for OpenSHA

`make_solution.py` writes the posterior-mean rates as an OpenSHA fault-system solution’s rates file (`spikeslab/rates.csv`, one row per rupture of the rupture set, removed ruptures at zero), and the best fit’s as `nnls/rates.csv`. Before writing, it checks that the rates and the rupture set describe the same ruptures: 615,700 of 615,700 kept ruptures cross exactly the fault sections the rupture set lists, and each one’s magnitude lies in its bin. With `--zip` it also writes a complete archive that OpenSHA loads. Any representative model can be written the same way.

## 12. Points to keep in mind

- **The fit check.** χ² is +23.1 standard deviations from the fit check.
- **Paleo rates.** The posterior mean underpredicts them (median ratio 0.79).
- **Slow mixing.** 71 ruptures per replicate, carrying 11.0% of the total rate, mix slowly; their individual rates are less precise than the rest, though sums over many ruptures are not affected.
- **Convergence.** Not every replicate passes: main_rep0: R-hat of the ruptures on or of a bin's fraction on 1.0148 (below 1.01 needed) (the report suggests longer runs).
- **Starts.** All 16 chains started from the same point (the best fit); runs from dispersed starts remain untested.
- **Representative models.** The 100 representatives miss some of their tolerances; the posterior-mean rate vector is the exact input for mean hazard.
- **What the data determine.** Sums of rates (fault, subsection and bin rates) and the probability that a rupture exceeds about 10<sup>−5</sup> per year are well determined; individual rates below that are largely set by the prior.

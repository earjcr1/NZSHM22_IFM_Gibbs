# Sampler design: mapping the good-fitting region of a nonnegative linear inverse problem

Hand-off from a claude.ai design discussion (2 Oct 2026). This file is the spec. It records the
decisions and the reasoning behind them so a fresh Claude Code session can act on it without the
original conversation. Sections 8 and 9 list open questions for the user and the validation gate
that must pass before anything runs on the real data.

## 0. Status after testing on the real 2e4 data (2 Oct 2026) — supersedes sections 2–6 where they conflict

The full write-up is `REPORT.md`, with code in `preliminaries.py`, `gibbs.py`,
`compare_runs.py` and `cru_common.py` (the log-space HMC script, `rhmc.py`, is set aside for now). The points below override the original design. Their numbers are for the earlier
inputs (447 × 18,356 after cleaning, 19 magnitude bins); the corrected inputs (446 × 18,243, 18 bins) are summarized at the
top of `REPORT.md`. The largest configuration is now 5e5 ruptures; the original design below says 4e5, the size
first given.

- **Cleaning.** Remove the 13 penalty rows (entries 1e20) and the 292 parameters they apply to
  (741 × 18,356), then the 294 rows left all-zero: 2 that were empty from the start (d = 1) and 292 that
  held only a 1e4 penalty on a removed parameter (d = 0). The result is 447 × 18,356, so N = 447 is the
  effective number of constraints. All entries are positive, and G is kept sparse (13% dense, not a
  dense float32 array). Keeping the empty rows (`--keep-empty-rows`) only adds a constant χ² = 2 (verified).
- **Regime.** Rates sit 100–1000× below their conditional sd, and every marginal is exponential
  (sd ≈ mean). The ρ_i analysis of section 2 assumed the opposite regime. The upper bound u = 1
  never binds, since the data cap every rate below 2.7e-3.
- **Flat prior.** Item 5 of section 2 is wrong for this problem. With σ = 1 and a uniform prior on
  [0, 1], posterior samples have χ² ≈ 10,270 (χ²_min = 67.65, N = 447), with predictions about 1.7×
  the data. Positivity makes the unconstrained directions a cone whose volume grows with the
  prediction scale. The choice of target (prior or measure) is the open modelling decision. An optional linear prior
  with each rate's top at its bin's target (`--prior linear`, October 2026) inflates the same way: bins a median 188%
  above their targets at 2e4 and 835% at 211,000 ruptures (REPORT.md, "Why not a linearly uniform prior"). A halfway
  prior, each rate from a 50/50 mix of uniform and log-uniform (`--prior halfway`), lands on its log-uniform end: the
  data give almost every value to the log-uniform part.
- **Default model (since October 2026): spike-and-slab** (`--prior spikeslab`) under the plain likelihood: per
  magnitude bin, each rupture off (rate 0) or on, log-uniform from a slab floor at 1e-10 (also the diagnostic cutoff;
  1e-12 in late October 2026) up to a soft top at the bin's target rate R_j whose width is the bin's own
  uncertainty, with the fraction on in each bin estimated. The data set the sparsity, the prior puts no pressure on
  the bin totals, and the products barely depend on the slab floor. The square-root slab (`--slab sqrt`, run from the
  command line: the same with an on rupture's rate uniform in √rate, density m^−1/2) was its cross-check until it was
  dropped at the end of October 2026; it fits as the default does, and at 211,000 ruptures it moved the subsections'
  event rates. θ = 1 (`binbox-theta:1:0`: per-bin log-uniform boxes n_j / (θ ln 10) decades deep, which reached the
  same pressure-free answer by another route) was the alternative until October 2026 and is kept for comparison only:
  from about 2e5 ruptures its boxes are too deep for its sampler to cross. Box priors with a common floor came first:
  the single box (1e-12 to 1e-2), then per-bin boxes down to 1e-12 (`binbox-floor:-12`). Their depth pushed the bin
  totals up and set the sparsity, which the data could not choose (the evidence kept rising as the floor dropped:
  −13,416, −7,408 and −5,151 nats at 1e-8, 1e-12 and 1e-16), and they gave much denser solutions, with exceedance
  probabilities at 1e-5 about 7 times higher. They are kept for comparison only. The χ² cap, once an optional model
  class, was removed in October 2026, a linearly uniform prior is not used, and the hierarchical prior is retired:
  after 36,000 sweeps its spread was still rising (R̂ 1.86). See REPORT.md.
- **Balance term (October 2026; off by default since early November 2026).** `gibbs.py --down-the-middle` adds to the Gaussian misfit, for each
  data type (slip-rate sections, MFD bins, paleo rows), A s² with s the type's relative shortfall, summed predictions
  against summed targets in units of each measurement's uncertainty. With `DOWN_THE_MIDDLE = 'zero-model'`
  (`--down-the-middle zero-model`, the default in late October 2026) each type has its own A, its zero-model misfit; with
  `DOWN_THE_MIDDLE = True` one A serves all types, the largest that keeps every type at or below one extra
  measurement. It is extra rows of G and d, so the kernel is unchanged and every prior takes it; the χ² of the fit
  check stays over the data rows. See REPORT.md, "Balance between over- and underprediction".
- **The band (early November 2026; optional).** `gibbs.py --band F` (`BAND = 10`) keeps every data row's
  prediction within a factor F of its target, from d/F to F d: a hard constraint on the rates, for any prior, with
  each update drawn within the rates the band allows. The defaults since are positivity (`--positive`, `BAND = 'positive'`:
  every prediction above zero, with no other limit), no balance term and a slab floor of 1e-10. The chains start from a penalized least-squares fit moved inside it, made once per problem. See
  REPORT.md, "The band".
- **Slip-rate sections (October 2026).** Each fault section's slip rate is one measurement, copied in the inputs to its
  k subsections; cleaning now divides each section's rows by √k (`preliminaries.py --slip-sections auto`, the default;
  `none` restores the inputs as given), so it counts once. Most numbers in this section count each subsection as its
  own measurement. With one measurement per section the fit check is 126 ± 16 on the corrected inputs: in quick runs
  spike-and-slab and θ = 1 pass it (χ² 140–142 ± 12) and the per-bin boxes fail it (547 ± 32), after which the default
  changed. See REPORT.md, "Slip-rate sections".
- **MFD uncertainty (October 2026).** The magnitude bins are a projection of two measurements (the rate of M≥5
  earthquakes and the b-value), not independent observations, and their uncertainty level in the inputs is a design
  value (10% at M7.0–7.1, growing as R_j^−0.25), not a measured one. Cleaning keeps it by default
  (`preliminaries.py --mfd-uncertainty as-given`); `two-measurements` multiplies every bin's uncertainty by
  √(bins / 2) = 3, so that together they carry two measurements' worth of information, as each slip-rate section
  counts once, and a number sets the M7.0–7.1 bin's relative uncertainty. The bin-to-bin shape is kept either way. In
  quick runs `two-measurements` let the data raise the bins a median 8% above their targets (2% as given), with the
  products within Monte Carlo noise, although single draws can then leave a whole bin empty (some bin below 1% of its
  target in about 4% of the draws of a very conservative run). This replaced block reweighting from a pilot run
  (`BLOCK_WEIGHTS`), which could not move the bins: their misfit is set by the model's freedom. See REPORT.md, "MFD
  uncertainty".
- **Production sampler: Gibbs** (C kernel via ctypes, threads across chains, about 2 ms per chain per
  sweep at 2e4, near-independent draws every 4 sweeps; at 211,000 ruptures, since October 2026's faster kernel, about
  11 ms per chain per sweep under spike-and-slab and 38 ms under θ). It also ran the acceptable-set target
  (coordinate hit-and-run on {χ² ≤ χ²_max}, the chord atlas run as an exact chain) until it was removed with the χ²
  cap in October 2026; the Gaussian likelihood is now the only target.
- **Reflective HMC (section 4) is retired for this problem.** Thousands of wall reflections per
  trajectory make acceptance collapse at any useful step size. **Log-parametrized HMC**
  (`rhmc.py --param log`, now set aside from the code set) works and was the independent cross-check: it agrees with Gibbs on all
  18,356 posterior means (z-scores N(0,1)). NUTS (section 6) adds little beyond it.
- **Log-uniform prior** (`--prior logbox:A:B:SE`, the single box, now retired; slice-sampled exactly in log m, with swap
  moves between near-duplicate ruptures, as are the current priors). With χ² ≤ 447 it fits the data, and subsection totals barely depend on the floor
  (1e-8 vs 1e-12: median 1.5%). Per-rupture medians move about 0.46 decades per decade of floor. See REPORT.md.
- **Transfer moves** (`build_transfers.py`, `gibbs.py --transfers`; set aside with the single box): exact draws along 26,925 near-null
  directions (one rupture against up to four sub-ruptures, plus near-duplicate pairs), sampled in a logit coordinate
  along each line. Validated, cheap, but no measurable gain on the real problem: log-space Gibbs already switches
  ruptures freely. The only slow mixers are the few largest ruptures (2-9 with R-hat above noise).
- **Validation gate (section 9), as now met:** self-tests against exact answers for both samplers,
  plus a rate-by-rate Gibbs vs log-HMC agreement (`compare_runs.py`) on the real data.

## 1. Problem statement

- Linear forward model `G m ≈ d`. `G` is `N × n` with `N ≈ 1e3` data (rows) and `n` between
  `2e4` and `4e5` parameters (columns), depending on configuration.
- Box constraints `0 ≤ m ≤ u`, `u ≈ 0.1`. The upper bound is a physical expectation ("not
  expected to exceed 1e-1"); it also makes the posterior proper in the null-space directions.
- Severely underdetermined (`n ≫ N`): the data constrain at most `N` linear combinations; the
  rest is constrained only by the box. Many near-collinear columns, so strong parameter
  tradeoffs, some nearly one-to-one.
- Previous practice: simulated annealing to find one good-fitting model. Not needed: the problem
  is convex, and a single model carries no information about spread or tradeoffs.
- Goal: posterior samples under the Gaussian likelihood `exp(-||Gm-d||²/2σ²)` with a flat prior
  on the box, from which to derive per-parameter PDFs, pairwise tradeoffs, and PDFs of any linear
  functional `aᵀm` needed downstream.
- Hardware: MacBook Pro M2, 8 cores, 24 GB RAM. `G` in float32 is 1.6 GB at `n = 4e5` (fits).
  The unit of cost throughout is one pass over `G` (`G @ x` or `G.T @ r`): about 1000·n flops,
  memory-bound, roughly 30 ms at `n = 4e5` and a few ms at `n = 2e4`.

Assumption to confirm with the user: the "order of 1e3 constraints" are the rows of `G` (data).
If there are also general linear inequalities `A m ≥ b`, see section 7.

## 2. Facts the design rests on

1. The posterior is a truncated Gaussian: log-concave, unimodal, convex support. No local
   minima, so no tempering or annealing. The MAP is bounded-variable least squares
   (`scipy.optimize.lsq_linear(G, d, bounds=(0, u), lsq_solver='lsmr')`, or projected gradient)
   and is highly non-unique.
2. Nothing that needs an `n × n` object is affordable: no dense covariance, whitening, polytope
   rounding, exact truncated-Gaussian HMC (Pakman–Paninski), or CHRR-type polytope samplers.
3. A sample must travel a box-width along the `~n − N` free directions and hits `O(n)` box faces
   on the way. Faces must therefore be handled per coordinate (axis-aligned walls, diagonal mass
   matrix). This rules out line-shrinkage samplers (hit-and-run, elliptical slice sampling) and
   any preconditioner that mixes coordinates.
4. Stiffness per parameter: `ρ_i = u·||G_i|| / σ`, the box width divided by the conditional
   standard deviation `s_i = σ / ||G_i||` of `m_i` given the rest. Random-walk samplers (Gibbs,
   Metropolis, coordinate hit-and-run) need `~ρ²` sweeps per independent sample; HMC needs
   `~ρ·ω_max` leapfrog steps, where `ω_max` is set by how many columns are nearly collinear.
   Both cost one pass over `G` per sweep or step, so HMC wins by roughly `ρ/ω_max`.
5. With a flat prior in 1e5 dimensions, samples sit in the typical set: misfit
   `χ² ≈ N ± sqrt(2N)`, not the minimum. The best-fit model is an atypical corner of the good
   region. This is correct behaviour, not a bug.
6. "No prior" is not on the menu. The uniform box prior says each parameter is a priori
   `Uniform(0, u)` with mean `u/2`, so the posterior favours diffuse models with many parameters
   at modest values over the sparse, vertex-like solutions a constrained optimizer returns, and
   the marginal of a poorly resolved parameter is mostly box geometry. If the user's belief is
   "usually much smaller than `u`, occasionally near it", use an exponential prior with scale `τ`:
   still log-concave; adds a constant `1/τ` to the gradient; Gibbs conditional mean shifts by
   `−s_i²/τ`.
7. `σ` (or per-datum `σ_j`) is the one unavoidable knob: it defines "good-fitting". With
   per-datum errors, whiten rows of `G` and `d` by `1/σ_j` and set `σ = 1`. A hard cutoff
   `||Gm − d|| ≤ ε` with `ε = σ·sqrt(N)` selects nearly the same region.

## 3. Deliverables, in order

1. `preliminaries.py`: load `G` as one float32 array; column norms (use `np.einsum('ij,ij->j', G, G)`,
   not `G**2`, to avoid a 1.6 GB temporary); histogram of `ρ_i`; MAP via BVLS; SVD via the
   `N × N` matrix `G Gᵀ` (singular values, `V = Gᵀ U S⁻¹`, resolution diagonal
   `R_ii = ||V[i, :]||²`); clusters of near-duplicate columns (cosine similarity, via a
   nearest-neighbour search in the `N`-dimensional column space).
2. `synth.py`: synthetic problems with planted collinear groups and known `σ` (section 9).
3. `gibbs.py`: truncated-Gaussian Gibbs sampler in numba, lockstep chains. Zero knobs.
4. `rhmc.py`: reflective HMC in numpy, lockstep chains, analytic diagonal mass, power-iteration
   step size, jittered trajectory length, trajectory-length pilot.
5. `nuts_ref.py`: numpyro NUTS reference, 2e4 configuration only.
6. `validate.py`: three-way agreement of marginals on synthetic data; R-hat and ESS.
7. `run_real.py`: production run on the real `G`; thinned samples to disk; online histograms and
   quantiles; stored functionals.

## 4. Reflective HMC (production sampler)

Target: `U(m) = ||Gm − d||² / (2σ²)` on `[0, u]^n`. Gradient `Gᵀ(Gm − d)/σ²`: one `G` pass and
one `Gᵀ` pass per leapfrog step.

Diagonal mass: `M_ii = ||G_i||²/σ² + 1/u²` (the inverse conditional variance; the `1/u²` term
keeps weak or zero columns finite). This is the right mass matrix and it is known in closed form:
no warmup adaptation.

Box handling: inside the position half-step, any coordinate that leaves `[0, u]` is folded back
(`x → −x` or `x → 2u − x`) and its momentum negated. During that sub-step each coordinate moves
in a straight line independently, so the fold is the exact billiard: reversible and
volume-preserving. The ordinary Metropolis test on `ΔH` therefore makes the sampler exact (this is
the "handling constraints by reflection" recipe in Neal's 2011 HMC review).

Step size: leapfrog is stable for `h < 2/ω_max` with
`ω_max² = λ_max(M^{-1/2} Gᵀ G M^{-1/2}) / σ²`. Compute `ω_max` once by ~30 power iterations
(`v ← M^{-1/2} Gᵀ (G (M^{-1/2} v)) / σ²`). Start at `h = 1/ω_max`; during burn-in multiply by
0.9 / 1.1 to land at 70–90% acceptance.

Trajectory time `T` (the one real knob): roughly the time for the slowest free direction to cross
its range. In these units a coordinate moves at speed `~M_ii^{-1/2}`, so the crossing time is at
most `u·sqrt(M_ii) ≈ ρ_i`. Start from `T0 = 90th percentile of u·sqrt(M_ii)`; run a pilot at
`T ∈ {0.5, 1, 2}·T0` and keep the value with the best ESS per gradient evaluation. Always jitter:
`L = int(T · Uniform(0.5, 1.5) / h)`.

Chains: run `K = 16–64` chains in lockstep as an `n × K` matrix so both passes over `G` are GEMMs
(cost grows little with `K`) and R-hat comes free. Initialise at the BVLS solution plus small
noise, clipped into the box. Burn-in until `χ²` plateaus near `N`; discard. Only the Gaussian part
is split from the kinetic term, so the acceptance rate reads directly on `h`.

```python
M = (np.einsum('ij,ij->j', G, G)/sigma**2 + 1/u**2)[:, None]; Minv, sqM = 1/M, np.sqrt(M)

def U_and_grad(X):                       # X is (n, K): K chains, one GEMM each way
    R = G @ X - d[:, None]
    return 0.5*np.sum(R**2, 0)/sigma**2, G.T @ R / sigma**2

def drift(X, P, h):                      # exact free flight + reflections off the box faces
    X = X + h*Minv*P
    while (out := (X < 0) | (X > u)).any():
        X = np.where(X < 0, -X, np.where(X > u, 2*u - X, X)); P = np.where(out, -P, P)
    return X, P

def hmc_step(X, h, L):                   # caller picks L = int(T * U(0.5, 1.5) / h)
    P = sqM*rng.standard_normal(X.shape, dtype=np.float32); U0, g = U_and_grad(X)
    H0 = U0 + 0.5*np.sum(Minv*P**2, 0); Xn, Pn = X, P - 0.5*h*g
    for i in range(L):
        Xn, Pn = drift(Xn, Pn, h); U1, g = U_and_grad(Xn)
        Pn -= (h if i < L-1 else 0.5*h)*g
    acc = np.log(rng.random(X.shape[1])) < H0 - (U1 + 0.5*np.sum(Minv*Pn**2, 0))
    return np.where(acc, Xn, X), acc
```

Diagnostics to log every run: acceptance per chain, `χ²` per chain, `Σm`, R-hat on ~200 random
parameters plus the stored functionals, ESS per gradient.

Expected cost: at `n = 4e5` a 32-chain leapfrog step is ~0.1 s. If the pilot gives `L ≈ 1e3`,
that is ~100 s per 32 near-independent samples: a few thousand samples in hours. At `n = 2e4`
the same is minutes, so prototype there.

## 5. Gibbs sampler (zero knobs; cross-check; production if ρ is small)

The conditional of `m_i` given the rest is `N(μ_i, s_i²)` truncated to `[0, u]` with
`s_i = σ/||G_i||` and `μ_i = m_i − G_i·r / ||G_i||²`, where `r = Gm − d` is a running residual
updated by `r += G_i (m_i_new − m_i_old)` after each draw. That is `2N` flops per parameter, so a
sweep is one pass over `G`; with `K` lockstep chains, `G_i` is read once per coordinate for all
chains (`G_i · R` with `R` of shape `N × K`).

Requirements:
- Use a proper truncated-normal sampler (Robert 1995, exponential rejection for the tails).
  Many conditionals sit far out in a tail; naive rejection stalls there.
- Recompute `r` from scratch once per sweep to kill float32 drift.
- numba, one process; parallelism comes from lockstep chains and from running several processes.

Mixing is `~ρ²` sweeps per independent sample: fine if the `ρ_i` are mostly below ~20–30,
hopeless at 1e3. Regardless, this sampler is the correctness reference for everything else: it
targets exactly the same distribution, so marginals of a handful of parameters must agree.

Optional upgrade if Gibbs is close to fast enough: joint "swap" moves along `e_i − e_j` for
near-duplicate column pairs (exact 1-D truncated-Gaussian conditional along the chord).

## 6. NUTS reference (2e4 configuration only)

Standard NUTS needs a smooth unconstrained target, so it uses `m = u·sigmoid(z)` with the
Jacobian. The reflective integrator cannot be dropped into NUTS: the U-turn test becomes noise
once 1e5 box-dominated coordinates flip momentum every few steps. The transform is exact and keeps
the scaling, but costs constant factors (curvature in `z` is position dependent, exponential tails
for parameters pinned near zero, loss of log-concavity), and there is no analytic mass matrix in
`z`-space, so NUTS must adapt it over a few hundred warmup iterations at full trajectory cost.
Back-of-envelope: 3–10× the gradient evaluations of tuned reflective lockstep HMC. Use it as the
reference and, if preferred, as the production tool at 2e4; not at 4e5.

```python
def model():
    m = numpyro.sample("m", dist.Uniform(0., u).expand([n]).to_event(1))   # sigmoid transform applied automatically
    numpyro.sample("d", dist.Normal(G @ m, sigma), obs=d)                  # G: jnp float32, (N, n)
mcmc = MCMC(NUTS(model, max_tree_depth=12), num_warmup=500, num_samples=1000, num_chains=8, chain_method="vectorized")
```

Raise `max_tree_depth` above the default 10 whenever `L` approaches 1e3; otherwise NUTS silently
truncates trajectories and the free directions random-walk.

## 7. Optional extensions (only if needed)

- Hard-cutoff billiard: sample uniformly on `{box} ∩ {||Gm − d|| ≤ ε}` by exact free flight with
  the same per-coordinate folds plus specular reflection off the data ellipsoid (normal `Gᵀ r`,
  crossing time a quadratic in `t`, one pass over `G` per hit). No step size, no accept/reject,
  removes the stiffness entirely; the price is event-driven code. Consider if `L` comes out at
  1e4 or more.
- Automatic trajectory length for lockstep chains: ChEES-HMC (Hoffman, Radul & Sountsov 2021;
  blackjax, TFP) tunes one shared jittered `T` from cross-chain statistics and only needs the
  endpoint velocity, so it should accept the reflective integrator.
- General linear inequalities `A m ≥ b`: crossing times along the straight-line drift are closed
  form; a hit is one `O(n)` momentum update `p ← p − 2 (aᵀv)/(aᵀ M⁻¹ a) · a`; costs one extra
  GEMM pair per step.
- Exponential prior with scale `τ` (section 2, item 6).

## 8. Open questions for the user

1. Source of `σ`: per-datum error bars (preferred; whiten rows) or a chosen scalar? Run at two
   values to assess sensitivity.
2. Prior: uniform on `[0, u]`, or exponential with a typical scale `τ`?
3. Are the ~1e3 "constraints" the rows of `G`, or are there additional inequalities?
4. Value of `u`, and whether the upper bound is expected to bind.
5. Which functionals `aᵀm` and which parameter pairs matter downstream (so they are stored
   explicitly during the production run).

## 9. Validation gate (must pass before any run on the real G)

Synthetic problem (`synth.py`): `N = 50, n = 2000` first, then `N = 1000, n = 20000`. Plant
groups of `k ∈ {2, 10, 100}` near-identical columns (small random perturbations) among random
columns; make `m_true` sparse-ish within `[0, u]`; `d = G m_true + N(0, σ²)` with known `σ`.

Checks:
- Reversibility of the reflective leapfrog: integrate `L` steps, negate momentum, integrate
  `L` steps back; recover the start to float precision. Do this with many reflections per step.
- Acceptance in 70–90% at the chosen `h`; `χ²` of samples `≈ N ± sqrt(2N)`.
- Gibbs, reflective HMC and NUTS agree on per-parameter means and 5/50/95% quantiles for a
  random subset of parameters and for all planted groups, within Monte Carlo error. Pairwise
  scatter plots for the planted pairs look identical across samplers.
- R-hat < 1.01 on the monitored set; ESS per gradient reported for each sampler and each `T`.

Only after this passes: production run on the real `G` at 2e4, then 4e5.

## 10. Outputs and bookkeeping

**As built** (supersedes the plan below; details in REPORT.md): draws are optional, stored as float32 of shape `(chains, draws, n)` (`--no-samples`, which the notebook always passes). Every run writes running summaries (`<out>_stream.npz`), including a per-rupture histogram of log rate in 0.01-decade bins, from which the notebook computes each rupture's mean, median, mode and ±1σ/±2σ-equivalent quantiles (`rate_vectors.csv`) in the original rupture numbering.

- Thinned samples as a float32 memmap of shape `(n, n_samples)`, e.g. 3000 samples at
  `n = 4e5` ≈ 5 GB on disk; plus per-chain `χ²`, `Σm`, and the user's functionals per sample.
- Per-parameter histograms (64 bins on `[0, u]`) and quantiles accumulated online.
- Any pairwise tradeoff and any `aᵀm` is then a lookup on the stored samples.

## 11. Pitfalls already identified

- Keep a single float32 copy of `G`; never create `n × n` objects or `G`-sized temporaries.
- Check that numpy uses Accelerate (`np.show_config()`); GEMMs on `G` are the whole cost.
- Recompute residuals rather than accumulating them (HMC does each step; Gibbs each sweep).
- The MAP has many exact zeros; starting on the boundary is fine for both samplers.
- Jitter `T` (periodicity); never use a fixed `L`.
- Zero columns: handled by the `1/u²` term in `M_ii`; they are simply uniform on `[0, u]`.

# Handoff: CRU rupture-rate posterior workflow

From the claude.ai cloud session of 16–21 October 2026 (claude.ai/code, session_01QV85rtK767xQyAUWQpsJGA), for
continuing the work in Claude Code on your own machine (VS Code or terminal). Read this first; then CLAUDE.md, which
Claude Code loads by itself and which pulls in the three documents in `docs/`.

## What is where

- **`cru_workflow_bundle.zip`** holds the code set, 18 files: the scripts, `cru_workflow.ipynb`, `CLAUDE.md` and
  `docs/` (`REPORT.md`: findings, numbers and run commands; `TESTING.md`: checks and expected values;
  `sampler-design.md`: the original design, with section 0 recording what the data showed). These documents are the
  authoritative record of every decision and its reasons; the conversation is not needed to reconstruct them.
- **This handoff** adds `HANDOFF.md` and `tools/`, the development tools the cloud session used that are not part of
  the code set:
  - `tools/run_nb.py`: runs the notebook headless, cell by cell, with configuration values overridden from the
    command line (`OUTPUT_DRAWS=False`, `"MFD_UNCERTAINTY='two-measurements'"`, `CELLS=5`, ...).
  - `tools/run_quick_checks.sh DATA_DIR RUNS_DIR`: the smoke test from `TESTING.md` (quick preset; spike-and-slab
    and θ under the defaults, spike-and-slab with `'two-measurements'`), one run at a time, resumable.
  - `tools/static_checks.py BUNDLE.zip UNZIPPED/cru_workflow`: the checks run on every bundle before delivery
    (inventory, parsing, leftovers, defaults, options, imports); expect "0 failures". Update it whenever a default,
    a setting or the file list changes.
  - `tools/compare_mfd.py`: two result sets compared on fit, bins, total rate, MFD slope, population and products,
    with replicate noise for scale (written for the MFD settings; works for any two result sets on the same inputs).
  - `tools/fig_mfd_levels.py`: the figure of bin totals against targets under the two MFD levels.
  - `tools/psis.py`: Pareto-smoothed importance sampling (PSIS-LOO), used for the paleo-site leave-one-out checks.
- **The inputs** `cru_A.txt` and `cru_D.txt` are yours; keep them in a data folder that nothing writes to.

## Setting up locally

1. Make a project folder, unzip the bundle into it (giving `cru_workflow/`), and unzip this handoff into
   `cru_workflow/` (adding `HANDOFF.md` and `tools/`).
2. Put the inputs in a separate data folder. In the notebook's configuration cell, set `DATA_DIR` to it and
   `RUN_DIR` to where outputs should go (relative paths are taken from the notebook's folder).
3. Track it with git: `git init && git add -A && git commit -m "CRU workflow, 21 Oct 2026"`. Leave run outputs
   out of git (add `RUN_DIR` to `.gitignore`).
4. Requirements: numpy, scipy, pandas, matplotlib, and a C compiler for the Gibbs kernel (`xcode-select --install`
   on macOS); without one `gibbs.py` falls back to NumPy, about 100 times slower.
5. Open the folder in VS Code and start Claude Code there. A good first message: "Read HANDOFF.md, run the
   self-tests and `tools/run_quick_checks.sh`, and compare the numbers with TESTING.md."

To bring the conversation itself along as well, open the Claude Code panel's **Session history**, choose the **Web**
tab and pick this session: the extension downloads a copy of the conversation (one way; nothing syncs back). It
refers to cloud paths (`/home/claude/cru_workflow`, `/mnt/user-data/uploads`), so tell the new session where the
files are now. The conversation was compacted once, so the documents remain the better record.

## State as of 21 October 2026

**Defaults:** spike-and-slab prior (`PRIOR_KIND = 'spikeslab'`, slab floor and diagnostic cutoff `FLOOR = -10`),
with θ = 1 as the alternative and cross-check; the Gaussian likelihood, the only one; one measurement per slip-rate
section (`SLIP_SECTIONS = 'auto'`); the MFD bins' uncertainties as the inputs give them (`MFD_UNCERTAINTY =
'as-given'`).

**Changes in the last few days**, all recorded in the docs:
- `MFD_UNCERTAINTY` (21 Oct) sets the magnitude bins' uncertainty level at cleaning. `'as-given'` (the default)
  keeps the inputs' design value, 10% at M7.0–7.1 growing as R_j^−0.25 (σ_j/R_j = 0.1 (R_j/R_7.0)^−0.25 exactly).
  `'two-measurements'` multiplies every bin's uncertainty by √(bins/2) = 3, so that the bins, a projection of two
  measurements (the rate of M≥5 and the b-value), count as two measurements rather than 18. A number sets the
  M7.0–7.1 bin's relative uncertainty (0.1 reproduces the inputs). The bin-to-bin shape always stays as the inputs
  give it: the 0.1 and the 0.25 power are set upstream, where cru_A/cru_D are built (the 0.1 is a holdover of
  UCERF3-style data-type weights), and the shape is to stay controlled there. In quick runs, `'two-measurements'`
  let the slip-rate and paleo data raise the bins a median 8% above their targets (2% as given) and the total rate
  5%, with the other products within Monte Carlo noise; spike-and-slab and θ then disagree on how the four largest
  bins share their rate, though not on its sum. The data cannot set the level (the bins' misfit is almost all the
  model's freedom). REPORT.md, "MFD uncertainty".
- Block reweighting from a pilot run (`BLOCK_WEIGHTS`) was built on 20 Oct and removed on 21 Oct in favour of the
  MFD setting; `block_weights.py` became `block_fit.py`, which keeps the per-block fit report (each row block's χ²,
  split into the posterior-mean misfit and the spread, the model's freedom) and self-tests the MFD helpers.
- The χ² cap (`--target cutoff`, `TARGET`) was removed on 20 Oct; the Gaussian likelihood is the only target.
- Paleo misfit (20 Oct, answered in conversation, not a code change): the paleo rows' excess χ² is concentrated in
  rows 835, 853, 860 and 861. Rows 860 and 861 are in tension with the slip-rate data (the best fit also leaves them
  at about half their targets); rows 853 and 835 are prior-driven (the best fit comes within 0.7 of an
  uncertainty of both, but the ruptures that would reach them are off in most posterior draws). The checks: PSIS-LOO per measurement, exact
  leave-one-out refits, and an NNLS conflict test (one row up-weighted tenfold).

**Quick-preset reference values** (two replicates; TESTING.md has more): spike-and-slab χ² 140 ± 12 against the fit
check's 126 ± 16, total rate 0.0218 per year against bin targets totalling 0.0214, bins a median 1.9% above their
targets; θ 142 ± 12, 0.0218, +2.0%. Self-tests: `python gibbs.py --selftest` (five verdicts, all PASSED, about 6
minutes) and `python block_fit.py --selftest` (seconds).

**Open items** are listed in REPORT.md ("Open items") and TESTING.md ("Known weak spots"). The main ones: the very
conservative spike-and-slab run on the corrected inputs is still to do; dispersed starts (`SA_STARTS`) are untested;
a slow-mixing minority of ruptures under both priors; the representative draws' tolerances; a correlated MFD
covariance (the two measurements' within-branch uncertainty plus a model error per bin) as the fuller MFD
treatment; the 5e5 configuration.

## How the work has been done (preferences to keep)

- Ask before changing a modelling decision (the list is in CLAUDE.md: the prior, its floor, edges and hyperpriors,
  the likelihood, sigma, bounds, how the slip-rate rows count, the MFD bins' uncertainty level).
- Keep draws only when needed: `OUTPUT_DRAWS = False` in the notebook, `--no-samples` for `gibbs.py`.
- Run heavy jobs one at a time.
- Delete each test folder once its numbers are recorded in the docs.
- Keep the docs in step with the code: every change updates REPORT.md, TESTING.md, CLAUDE.md and the design notes
  where they describe it, with numbers stated with the conditions they were measured under.
- Before delivering: the self-tests that apply, a quick end-to-end notebook run, and `tools/static_checks.py` on the
  bundle; then deliver the bundle as one zip of the 18 files.

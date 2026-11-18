#!/bin/bash
# Smoke test of the workflow at the old quick preset's run lengths (a fixed RUN_PLAN), as in docs/TESTING.md: one notebook run at a time, without stored draws,
# resumable (a step that finished is skipped on a rerun).
#
#   tools/run_quick_checks.sh DATA_DIR RUNS_DIR
#
# DATA_DIR holds cru_A.txt and cru_D.txt (read only). RUNS_DIR receives default/ (the defaults), two/ (with
# MFD_UNCERTAINTY = 'two-measurements') and logs/. A few minutes on one core; compare the numbers with TESTING.md, then delete RUNS_DIR. The run lengths (RUN_PLAN, in place of
# RUN_MINUTES's plan) and the default settings are set here, whatever the notebook's configuration cell holds; PYTHON names the interpreter (python3).
set -u
CODE=$(cd "$(dirname "$0")/.." && pwd)                 # the notebook's folder (the scripts are in its tools/)
DATA=$(cd "$1" && pwd)
mkdir -p "$2" && RUNS=$(cd "$2" && pwd)
L=$RUNS/logs; NB=$CODE/cru_workflow.ipynb; R=$CODE/tools/run_nb.py; PY=${PYTHON:-python3}
PIN=("RUN_PLAN=dict(chains=4, sweeps=2400, burn=400, thin=8, replicates=1)" "TARGETS='exploratory'" DISPERSED_STARTS=False "SLIP_SECTIONS='auto'" FLOOR=-10 BLOCKS=None DOWN_THE_MIDDLE=True
     "MISFIT_SLIP='linear'" "MISFIT_MFD='linear'" "MISFIT_PALEO='rate'" MFD_MONOTONIC=False MFD_GR=False MFD_GR_TAU=None DATATYPES_EQUAL=False "SLIP_ALONG_RUPTURE='uniform'" PLATE_RATE=False CFM_SLIPRATES=False CFM_TIMEFRAME=False CFM_TEMPLATE_RHO=None CFM_RATE_LINKS=True SEED="'legacy'" SLIP_RATE_FACTOR=0.8 SLIP_SUM=False)                                             # the settings TESTING.md's reference values were measured under
mkdir -p "$L"
mark() { echo "$1 $(date +%T)" >> "$L/progress.txt"; }
step() { local name=$1; shift; [ -f "$L/done_$name" ] && return; mark "$name start"; "$@" > "$L/$name.txt" 2>&1; local rc=$?
         mark "$name exit $rc"; [ $rc -eq 0 ] && touch "$L/done_$name" || exit $rc; }
step default_spikeslab "$PY" -u "$R" "$NB" "$RUNS/default" "$DATA" "${PIN[@]}" "MFD_UNCERTAINTY='as-given'" REPRESENTATIVES=20
step two_spikeslab "$PY" -u "$R" "$NB" "$RUNS/two" "$DATA" "${PIN[@]}" "MFD_UNCERTAINTY='two-measurements'" REPRESENTATIVES=0
mark "all done"

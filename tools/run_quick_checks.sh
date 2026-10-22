#!/bin/bash
# Quick-preset smoke test of the workflow, as in docs/TESTING.md: one notebook run at a time, without stored draws,
# resumable (a step that finished is skipped on a rerun).
#
#   tools/run_quick_checks.sh DATA_DIR RUNS_DIR
#
# DATA_DIR holds cru_A.txt and cru_D.txt (read only). RUNS_DIR receives default/ (spike-and-slab and theta under the
# defaults), two/ (spike-and-slab with MFD_UNCERTAINTY = 'two-measurements') and logs/. About 7 minutes on one core
# in the cloud sandbox; compare the numbers with TESTING.md, then delete RUNS_DIR. The quick preset and the default
# settings are set here, whatever the notebook's configuration cell holds; PYTHON names the interpreter (python3).
set -u
CODE=$(cd "$(dirname "$0")/.." && pwd)                 # the scripts' folder (tools/ sits inside it)
DATA=$(cd "$1" && pwd)
mkdir -p "$2" && RUNS=$(cd "$2" && pwd)
L=$RUNS/logs; NB=$CODE/cru_workflow.ipynb; R=$CODE/tools/run_nb.py; PY=${PYTHON:-python3}
PIN=("PRESET='quick'" "SLIP_SECTIONS='auto'" FLOOR=-10 "BOX_TOP='soft'" THETA=1.0 USE_SWAPS=True SA_STARTS=None BLOCKS=None
     OUTPUT_DRAWS=False BEST_FIT=True)                               # the settings TESTING.md's reference values were measured under
mkdir -p "$L"
mark() { echo "$1 $(date +%T)" >> "$L/progress.txt"; }
step() { local name=$1; shift; [ -f "$L/done_$name" ] && return; mark "$name start"; "$@" > "$L/$name.txt" 2>&1; local rc=$?
         mark "$name exit $rc"; [ $rc -eq 0 ] && touch "$L/done_$name" || exit $rc; }
step default_spikeslab "$PY" -u "$R" "$NB" spikeslab "$RUNS/default" "$DATA" 0 "${PIN[@]}" "MFD_UNCERTAINTY='as-given'" REPRESENTATIVES=20
step default_theta "$PY" -u "$R" "$NB" theta "$RUNS/default" "$DATA" 0 "${PIN[@]}" "MFD_UNCERTAINTY='as-given'" REPRESENTATIVES=0
step two_spikeslab "$PY" -u "$R" "$NB" spikeslab "$RUNS/two" "$DATA" 0 "${PIN[@]}" "MFD_UNCERTAINTY='two-measurements'" REPRESENTATIVES=0
mark "all done"

#!/usr/bin/env python3
"""Execute cru_workflow.ipynb's code cells in order in one namespace, as 'Run all' would (no Jupyter here).

    python tools/run_nb.py NOTEBOOK RUN_DIR DATA_DIR RUN_SELFTESTS(0/1) [NAME=VALUE ...] [CELLS=n]

RUN_DIR and DATA_DIR are best given as absolute paths (the notebook resolves relative ones from its own folder).
NAME=VALUE replaces a configuration value, e.g. "PRESET='quick'" or "MFD_UNCERTAINTY='two-measurements'";
CELLS=n stops after the first n code cells.

Only these configuration values are overridden, by replacing their assignments in the configuration cell; everything
else runs exactly as written in the notebook.
"""
import json
import os
import re
import sys
import time
import traceback

nb_path, run_dir, data_dir, selftests = sys.argv[1:5]
extra = dict(a.split('=', 1) for a in sys.argv[5:])
max_cells = int(extra.pop('CELLS', 10 ** 6))
nb = json.load(open(nb_path))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
src0 = ''.join(cells[0]['source'])
for name, val in (('RUN_DIR', repr(run_dir)), ('DATA_DIR', repr(data_dir)),
                  ('RUN_SELFTESTS', 'True' if selftests == '1' else 'False')) + tuple(extra.items()):
    src0, k = re.subn(rf"^{name} = [^#\n]*?(\s+#)", lambda m: f"{name} = {val}" + m.group(1), src0, count=1, flags=re.M)
    assert k == 1, name
os.chdir(os.path.dirname(os.path.abspath(nb_path)))      # the notebook's folder, as Jupyter would start it
ns = {'__name__': '__main__'}
t_all = time.time()
for i, c in enumerate(cells[:max_cells]):
    src = src0 if i == 0 else ''.join(c['source'])
    t0 = time.time()
    print(f'\n######## code cell {i + 1}/{len(cells)} ({c.get("id", "")}) started {time.strftime("%H:%M:%S")}', flush=True)
    try:
        exec(compile(src, f'<cell {c.get("id", i)}>', 'exec'), ns)
    except Exception:
        traceback.print_exc()
        print(f'######## cell {i + 1} FAILED after {time.time() - t0:.0f}s', flush=True)
        sys.exit(1)
    print(f'######## code cell {i + 1} done in {time.time() - t0:.0f}s', flush=True)
print(f'\n######## notebook finished in {(time.time() - t_all) / 60:.1f} min', flush=True)

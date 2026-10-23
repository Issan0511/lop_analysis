#!/usr/bin/env bash
# On the rented GPU: the checks, then the launcher in the background (setsid + nohup; it survives
# the shell). A check suite is (re)run unless this host already has a passing record for the current
# sources and production.json: the data-free checks (synthetic images, about 10 minutes; needed when
# production.json or the code changed after white-san ran them) and the data checks (S-data, S-stream,
# S-arm1, S-smoke, S-speed, about 12 minutes). Re-running after a stop resumes at the last task
# boundary. When everything is done the launcher writes $VIT_LONG_RAW/DONE and summary/judge.md.
# No cloud API calls.
set -euo pipefail
WORK="${WORK:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
REPO="$WORK/lop_analysis"
VENV="${VENV:-$WORK/venv}"
PY="${VIT_LONG_PY:-$VENV/bin/python}"
export VIT_LONG_RAW="${VIT_LONG_RAW:-$HOME/Projects/obsidian-research-data/vit_long_1024}"
export VIT_LONG_PY="$PY" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
# S-disk floor: 25 GiB on white-san; the rented 80 GB disk holds about 44 GB of outputs, so use 5 GiB there.
export VIT_LONG_FREE_FLOOR_GIB="${VIT_LONG_FREE_FLOOR_GIB:-5}"
mkdir -p "$VIT_LONG_RAW"
cd "$REPO"
echo "== processes and GPU before starting"
pgrep -fa python || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
valid() {   # valid <record file name>: passing, same sources, same production settings
  "$PY" - "$1" << 'PYEOF'
import json, sys
from analysis.vit_long_1024 import common as C
path = C.RESULTS / 'checks' / sys.argv[1]
record = json.loads(path.read_text()) if path.exists() else {}
ok = bool(record.get('all_pass')) and record.get('source_hashes_at_validation') == C.source_hashes() \
     and record.get('production') == C.production_layout()
sys.exit(0 if ok else 1)
PYEOF
}
if valid checks_nodata.json; then
  echo "== data-free checks: a passing record for these sources exists"
else
  echo "== data-free checks"
  "$PY" -m analysis.vit_long_1024.checks_nodata 2>&1 | grep -v -i warning | tee -a "$VIT_LONG_RAW/checks_nodata.log"
fi
if valid "checks_data_$(hostname).json"; then
  echo "== data checks: a passing record from this host for these sources exists"
  cp "$REPO/results/vit_long_1024/checks/checks_data_$(hostname).json" "$REPO/results/vit_long_1024/checks/checks_data.json"
else
  echo "== data checks"
  "$PY" -m analysis.vit_long_1024.checks_data 2>&1 | grep -v -i warning | tee -a "$VIT_LONG_RAW/checks_data.log"
fi
"$PY" - << 'PYEOF'
import json, sys
from analysis.vit_long_1024 import common as C
nodata = json.loads((C.RESULTS / 'checks/checks_nodata.json').read_text())
data = json.loads((C.RESULTS / 'checks/checks_data.json').read_text())
speed = data.get('S_speed', {})
print('data-free passes', nodata.get('passes'))
print('data passes', data.get('passes'))
print(f"S-speed ({speed.get('host')}): {speed.get('ms_per_update_training', 0):.1f} ms/update training, "
      f"{speed.get('ms_per_update_all', 0):.1f} ms/update all; {speed.get('runs')} continual runs "
      f"{speed.get('hours_continual_runs', 0):.2f} h, with fresh {speed.get('hours_total', 0):.2f} h")
sys.exit(0 if nodata.get('all_pass') and data.get('all_pass') else 1)
PYEOF
echo "== launcher in the background"
setsid nohup "$PY" -m analysis.vit_long_1024.launch >> "$VIT_LONG_RAW/launcher.log" 2>&1 < /dev/null &
echo "launcher pid $!; status: cat $VIT_LONG_RAW/status.json; stop at a task boundary: touch $VIT_LONG_RAW/STOP"

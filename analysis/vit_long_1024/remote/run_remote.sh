#!/usr/bin/env bash
# On the rented GPU: data checks (S-data, S-stream, S-arm1, S-smoke, S-speed), then the launcher in
# the background (setsid + nohup; it survives the shell). Re-running after a stop resumes at the last
# task boundary; the data checks are skipped when this host already has a passing record for these
# sources. When everything is done the launcher writes $VIT_LONG_RAW/DONE and summary/judge.md.
set -euo pipefail
WORK="${WORK:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
REPO="$WORK/lop_analysis"
VENV="${VENV:-$WORK/venv}"
PY="${VIT_LONG_PY:-$VENV/bin/python}"
export VIT_LONG_RAW="${VIT_LONG_RAW:-$HOME/Projects/obsidian-research-data/vit_long_1024}"
export VIT_LONG_PY="$PY" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
mkdir -p "$VIT_LONG_RAW"
cd "$REPO"
echo "== processes and GPU before starting"
pgrep -fa python || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
if "$PY" - << 'PYEOF'
import json, socket, sys
from analysis.vit_long_1024 import common as C
path = C.RESULTS / 'checks' / f'checks_data_{socket.gethostname()}.json'
ok = path.exists() and json.loads(path.read_text()).get('all_pass') and \
     json.loads(path.read_text()).get('source_hashes_at_validation') == C.source_hashes()
sys.exit(0 if ok else 1)
PYEOF
then
  echo "== data checks already passed on this host for these sources"
else
  echo "== data checks (about 10 minutes)"
  "$PY" -m analysis.vit_long_1024.checks_data 2>&1 | grep -v -i warning | tee -a "$VIT_LONG_RAW/checks_data.log"
fi
"$PY" - << 'PYEOF'
import json, sys
from analysis.vit_long_1024 import common as C
record = json.loads((C.RESULTS / 'checks/checks_data.json').read_text())
speed = record.get('S_speed', {})
print('passes', record.get('passes'))
print(f"S-speed: {speed.get('ms_per_update_training', 0):.1f} ms/update (training), "
      f"{speed.get('ms_per_update_all', 0):.1f} ms/update (all), 10 runs + fresh = {speed.get('hours_total', 0):.2f} h")
sys.exit(0 if record.get('all_pass') else 1)
PYEOF
echo "== launcher in the background"
setsid nohup "$PY" -m analysis.vit_long_1024.launch >> "$VIT_LONG_RAW/launcher.log" 2>&1 < /dev/null &
echo "launcher pid $!; status: cat $VIT_LONG_RAW/status.json; stop at a task boundary: touch $VIT_LONG_RAW/STOP"

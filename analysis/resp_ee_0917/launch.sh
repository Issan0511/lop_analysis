#!/usr/bin/env bash
# resp_ee_0917 -- spec section 8: seeds 0-9, one process per seed (prefix + 26 arms), N at a time, 1 thread each.
# Refuses to start unless checks.json says all_pass, PREREG_COMMIT is set in the runner, the code and the spec
# are committed and HEAD is pushed.  N = min(MAX_SLOTS, what fits in MemAvailable now with 6 GiB left for the
# desktop at 1.2x the S-cost peak RSS); N < 1 aborts.  This is a shared desktop: MAX_SLOTS defaults to 4 and
# jobs start 25 s apart so each one's RSS is visible before the next memory reading.
# A seed with provenance.json is skipped (relaunch-safe).  Log names are flat (s<seed>.log).
#   setsid nohup bash analysis/resp_ee_0917/launch.sh > results/resp_ee_0917/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/resp_ee_0917
MOD=src/resp_ee_0917.py
PY=/usr/bin/python3
MAX_SLOTS=${MAX_SLOTS:-4}
$PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
grep -q '^PREREG_COMMIT = "[0-9a-f]\{40\}"' "$MOD" || { echo "ABORT: PREREG_COMMIT not set in $MOD"; exit 1; }
[ -z "$(git status --porcelain -- src analysis/resp_ee_0917 specs/spec_resp_ee_0917.md)" ] \
  || { echo "ABORT: uncommitted code or spec"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
N=$($PY - "$MAX_SLOTS" <<'PYEOF'
import json, sys
c = json.load(open("results/resp_ee_0917/checks.json"))["S-cost"]
avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0] / 2**20
fit = int((avail - 6.0) // (1.2 * c["peak_rss_gib_max"]))
print(max(0, min(int(sys.argv[1]), fit)))
PYEOF
)
[ "$N" -ge 1 ] || { echo "ABORT: no slot fits in memory now"; exit 1; }
mkdir -p "$OUT/runs" "$OUT/logs"

run_job() {
  local seed=$1
  local name="s${seed}"
  if [ -f "$OUT/runs/$name/provenance.json" ]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  sleep $(( (seed % N_SLOTS) * 25 ))
  echo "$(date -Is) start $name  $(free -g | awk '/^Mem:/{print "avail "$7" GiB"}')"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" "$MOD" --seed "$seed" --threads 1 \
    --out "$OUT/runs/$name" > "$OUT/logs/$name.log" 2>&1 \
    && echo "$(date -Is) done  $name" || echo "$(date -Is) FAIL  $name (exit $?)"
}
export -f run_job
export OUT MOD PY
export N_SLOTS="$N"

echo "$(date -Is) launch: git $(git rev-parse HEAD)  slots $N  $(free -g | awk '/^Mem:/{print "avail "$7" GiB"}')"
printf '%s\n' 0 1 2 3 4 5 6 7 8 9 | xargs -P "$N" -I{} bash -c 'run_job "$1"' _ {}
echo "$(date -Is) all jobs returned"

#!/usr/bin/env bash
# drive_rlmnist_1007 -- spec section 8: seeds 0-9 x 10 tasks x 80 epochs (ELU->ELU ref arm, observed), N at a time,
# 1 thread each.  Refuses to start unless checks.json validates against the current sources, the code and the spec
# are committed and HEAD is pushed.  N = min(4, what fits in MemAvailable now with 6 GiB left over at 1.2x the
# checked peak RSS per process); N < 1 aborts.  Queue order 0..9 (main seeds first).  Log names are flat (sSEED).
# A seed with complete.json is skipped; an incomplete seed directory is moved aside (never overwritten).
#   nohup setsid bash analysis/drive_rlmnist_1007/launch.sh > results/drive_rlmnist_1007/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/drive_rlmnist_1007
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
MAX_SLOTS=${MAX_SLOTS:-4}
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -c "
import json, sys
from analysis.drive_rlmnist_1007.checks import validate_checks
validate_checks(json.load(open('$OUT/checks.json')))" || { echo "ABORT: checks.json does not validate"; exit 1; }
[ -z "$(git status --porcelain -- src analysis/drive_rlmnist_1007 specs/spec_drive_rlmnist_1007.md)" ] \
  || { echo "ABORT: uncommitted code or spec"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
N=$("$PY" - "$MAX_SLOTS" <<'EOF'
import json, sys
c = json.load(open("results/drive_rlmnist_1007/checks.json"))["cost"]
avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0] / 2**20
fit = int((avail - 6.0) // (1.2 * c["peak_rss_gib"]))
print(max(0, min(int(sys.argv[1]), fit)))
EOF
)
[ "$N" -ge 1 ] || { echo "ABORT: no slot fits in memory now"; exit 1; }
mkdir -p "$OUT/run" "$OUT/logs"

run_seed() {
  local seed=$1
  local dir="$OUT/run/s$seed"
  if [ -f "$dir/complete.json" ]; then echo "$(date -Is) skip s$seed (complete)"; return 0; fi
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir")" ]; then
    local aside="$OUT/run/stopped_s${seed}_$(date +%Y%m%dT%H%M%S)"
    mv "$dir" "$aside"; echo "$(date -Is) moved incomplete s$seed to $aside"
  fi
  echo "$(date -Is) start s$seed"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" -m src.drive_rlmnist_1007 --mode production --seed "$seed" \
    --out "$dir" --checks "$OUT/checks.json" --production-go > "$OUT/logs/s$seed.log" 2>&1 \
    && echo "$(date -Is) done  s$seed" || echo "$(date -Is) FAIL  s$seed (exit $?)"
}
export -f run_seed
export OUT PY

echo "$(date -Is) launch: git $(git rev-parse HEAD)  slots $N  $(free -g | awk '/^Mem:/{print "avail "$7" GiB"}')"
printf '%s\n' 0 1 2 3 4 5 6 7 8 9 | xargs -P "$N" -I{} bash -c 'run_seed {}'
echo "$(date -Is) all seeds returned"

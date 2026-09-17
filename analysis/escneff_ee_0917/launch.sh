#!/usr/bin/env bash
# escneff_ee_0917 -- spec section 8: 10 jobs (seeds 10-19, one per seed: the prefix + escape_ee_0917's 12 arms), 1 thread each, at most
# MAX_SLOTS at once.
# Refuses to start unless checks.json says all_pass, PREREG_COMMIT is set in the runner, the code and the
# spec are committed and HEAD is pushed.  This is a shared desktop: every start goes through one flock; the
# holder starts a job only if MemAvailable leaves 6 GiB for the desktop after 1.2x the S-cost peak RSS, then
# waits 25 s (so the next reading sees the new RSS) before releasing the lock; otherwise it releases and
# retries after 60 s.  A job with provenance.json is skipped (relaunch-safe).  A STOP file in $OUT keeps
# jobs that have not started from starting.  Log names are flat.
#   setsid nohup bash analysis/escneff_ee_0917/launch.sh > results/escneff_ee_0917/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/escneff_ee_0917
MOD=src/escneff_ee_0917.py
PY=/usr/bin/python3
MAX_SLOTS=${MAX_SLOTS:-4}
$PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
for f in "$MOD"; do
  grep -q '^PREREG_COMMIT = "[0-9a-f]\{40\}"' "$f" || { echo "ABORT: PREREG_COMMIT not set in $f"; exit 1; }
done
[ -z "$(git status --porcelain -- src analysis/escneff_ee_0917 specs/spec_escneff_ee_0917.md)" ] \
  || { echo "ABORT: uncommitted code or spec"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
NEED_KB=$($PY -c "
import json
c = json.load(open('$OUT/checks.json'))['S-cost']
print(int(1.2 * c['peak_rss_gib'] * 2**20))")
RESERVE_KB=$((6 * 1024 * 1024))
mkdir -p "$OUT/runs" "$OUT/logs"
LOCK="$OUT/logs/.start.lock"

run_job() {
  local kind=$1 name=$2 dir=$3; shift 3
  if [ -f "$dir/provenance.json" ]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  while true; do
    [ -f "$OUT/STOP" ] && { echo "$(date -Is) STOP: not starting $name"; return 0; }
    exec 9>"$LOCK"
    flock 9
    local avail
    avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    if [ "$avail" -ge $((RESERVE_KB + NEED_KB)) ] && [ ! -f "$OUT/STOP" ]; then
      echo "$(date -Is) start $name  avail $((avail / 1024)) MiB"
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" "$@" --threads 1 --out "$dir" \
        > "$OUT/logs/$name.log" 2>&1 9>&- &
      local pid=$!
      sleep 25
      flock -u 9
      exec 9>&-
      if wait "$pid"; then echo "$(date -Is) done  $name"; else echo "$(date -Is) FAIL  $name (exit $?)"; fi
      return 0
    fi
    flock -u 9
    exec 9>&-
    sleep 60
  done
}
export -f run_job
export OUT PY NEED_KB RESERVE_KB LOCK

JOBS=()
for s in 10 11 12 13 14 15 16 17 18 19; do JOBS+=("main s$s $OUT/runs/s$s $MOD --seed $s"); done

echo "$(date -Is) launch: git $(git rev-parse HEAD)  slots $MAX_SLOTS  need $((NEED_KB / 1024)) MiB/job  $(free -g | awk '/^Mem:/{print "avail "$7" GiB"}')"
printf '%s\n' "${JOBS[@]}" | xargs -P "$MAX_SLOTS" -I{} bash -c 'run_job $1' _ {}
echo "$(date -Is) all jobs returned"

#!/usr/bin/env bash
# doors_rlmnist_1007 -- spec section 8: 4 arms x seeds 0-9 x 100 tasks x 80 epochs (ELU->ELU, CPU, 1 thread each).
# Refuses to start unless checks.json says all_pass, the code and the spec are committed and HEAD is pushed.
# At most MAX_SLOTS (default 6) jobs at once.  Before each start: MemAvailable - 6 GiB >= 1.2 x the S-cost peak RSS,
# otherwise it waits (never aborts for memory); after each start it waits 30 s so the new RSS shows in MemAvailable.
# Queue order: seed 0 (ref, C, H, CH), seed 1, ... so an early stop leaves complete seeds.  A shard with
# provenance.json is skipped (relaunch-safe).  Touch results/doors_rlmnist_1007/STOP to stop starting new jobs.
# Shard and log names are flat (ARM_sSEED, no '/').  Start it detached:
#   nohup setsid bash analysis/doors_rlmnist_1007/launch.sh > results/doors_rlmnist_1007/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/doors_rlmnist_1007
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python      # 3.12.3, torch 2.13 (/usr/bin/python3 has no numpy)
MOD=src.doors_rlmnist_run_1007
MAX_SLOTS=${MAX_SLOTS:-6}
RESERVE_GIB=6
[ -x "$PY" ] || { echo "ABORT: $PY not found"; exit 1; }
"$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
[ -z "$(git status --porcelain -- src analysis/doors_rlmnist_1007 specs/spec_doors_rlmnist_1007.md $OUT/checks.json)" ] \
  || { echo "ABORT: uncommitted code, spec or checks"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
PEAK=$("$PY" -c "import json; print(json.load(open('$OUT/checks.json'))['S-cost']['peak_rss_gib_max'])")
mkdir -p "$OUT/runs" "$OUT/logs"

avail_gib() { awk '/^MemAvailable:/{printf "%.2f", $2/1048576}' /proc/meminfo; }
mem_ok() { awk -v a="$(avail_gib)" -v p="$PEAK" -v r="$RESERVE_GIB" 'BEGIN{exit !((a - r) >= 1.2 * p)}'; }

run_job() {
  local arm=$1 seed=$2
  local name="${arm}_s${seed}"
  echo "$(date -Is) start $name  (avail $(avail_gib) GiB)"
  if OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" -m "$MOD" --arm "$arm" --seeds "$seed" \
       --tasks 100 --epochs 80 --lr 0.001 --threads 1 --out "$OUT/runs/$name" > "$OUT/logs/$name.log" 2>&1; then
    echo "$(date -Is) done  $name"
  else
    echo "$(date -Is) FAIL  $name (exit $?)"
  fi
}

queue=()
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for arm in ref C H CH; do
    queue+=("$arm $seed")
  done
done
echo "$(date -Is) launch: git $(git rev-parse HEAD)  max_slots $MAX_SLOTS  peak ${PEAK} GiB  avail $(avail_gib) GiB"
pids=()
i=0
waits=0
while [ "$i" -lt "${#queue[@]}" ] || [ "${#pids[@]}" -gt 0 ]; do
  alive=()
  for p in "${pids[@]}"; do
    if kill -0 "$p" 2>/dev/null; then alive+=("$p"); fi
  done
  pids=("${alive[@]}")
  if [ -f "$OUT/STOP" ] && [ "${#pids[@]}" -eq 0 ]; then echo "$(date -Is) STOP file: not starting the rest"; break; fi
  if [ "$i" -lt "${#queue[@]}" ] && [ "${#pids[@]}" -lt "$MAX_SLOTS" ] && [ ! -f "$OUT/STOP" ]; then
    set -- ${queue[$i]}
    if [ -f "$OUT/runs/${1}_s${2}/provenance.json" ]; then
      echo "$(date -Is) skip ${1}_s${2} (done)"
      i=$((i + 1))
      continue
    fi
    if mem_ok; then
      run_job "$1" "$2" &
      pids+=("$!")
      i=$((i + 1))
      waits=0
      sleep 30
      continue
    fi
    if [ $((waits % 30)) -eq 0 ]; then echo "$(date -Is) wait: avail $(avail_gib) GiB < 6 + 1.2 x ${PEAK}"; fi
    waits=$((waits + 1))
  fi
  sleep 10
done
echo "$(date -Is) all jobs returned"

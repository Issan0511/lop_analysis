#!/usr/bin/env bash
# adamw_dose_1010 -- spec section 8.  Box 1: 5 arms x seeds 0-9 x 100 tasks x 80 epochs (ELU->ELU, CPU, 1 thread each).
# Box 2 (separate script launch_c51.sh) runs on the GPU and may run at the same time.
# Refuses to start unless checks.json says all_pass, the code and the spec are committed and HEAD is pushed (the CLI
# itself refuses a registered output on top of that).  At most MAX_SLOTS (default 6) of its own jobs at once.  Before
# each start: MemAvailable - 6 GiB >= 1.2 x the S-cost peak RSS, and the machine's heavy python processes (lifetime
# CPU >= 50%, any session) plus the new one <= MAX_MACHINE (12); otherwise it waits.  After each start it waits 30 s.
# Queue order: seed 0 (ref, wd1e-3, wd1e-2, wd3e-2, wd1e-1), seed 1, ...  A shard with provenance.json is skipped
# (relaunch-safe).  Touch results/adamw_dose_1010/STOP to stop starting new jobs.  Log names are flat (ARM_sSEED).
#   nohup setsid bash analysis/adamw_dose_1010/launch.sh > results/adamw_dose_1010/logs/launch_mnist.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/adamw_dose_1010
RUNS=$OUT/mnist/runs
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python      # 3.12.3, torch 2.13
MOD=src.adamw_dose_1010_run
MAX_SLOTS=${MAX_SLOTS:-6}
MAX_MACHINE=${MAX_MACHINE:-12}
RESERVE_GIB=6
[ -x "$PY" ] || { echo "ABORT: $PY not found"; exit 1; }
"$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
[ -z "$(git status --porcelain -- src analysis specs $OUT/checks.json)" ] \
  || { echo "ABORT: uncommitted code, spec or checks"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
PEAK=$("$PY" -c "import json; print(json.load(open('$OUT/checks.json'))['S-cost']['peak_rss_gib_max'])")
mkdir -p "$RUNS" "$OUT/logs"

avail_gib() { awk '/^MemAvailable:/{printf "%.2f", $2/1048576}' /proc/meminfo; }
mem_ok() { awk -v a="$(avail_gib)" -v p="$PEAK" -v r="$RESERVE_GIB" 'BEGIN{exit !((a - r) >= 1.2 * p)}'; }
heavy() { ps -eo pcpu=,args= | awk '$0 ~ /python/ && $0 !~ /awk/ && $1 >= 50 {n++} END {print n + 0}'; }

run_job() {
  local arm=$1 seed=$2
  local name="${arm}_s${seed}"
  echo "$(date -Is) start $name  (avail $(avail_gib) GiB, heavy python $(heavy))"
  if OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" -m "$MOD" mnist --arm "$arm" --seeds "$seed" \
       --tasks 100 --epochs 80 --lr 0.001 --threads 1 --out "$RUNS/$name" > "$OUT/logs/$name.log" 2>&1; then
    echo "$(date -Is) done  $name"
  else
    echo "$(date -Is) FAIL  $name (exit $?)"
  fi
}

queue=()
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for arm in ref wd1e-3 wd1e-2 wd3e-2 wd1e-1; do
    queue+=("$arm $seed")
  done
done
echo "$(date -Is) launch: git $(git rev-parse HEAD)  max_slots $MAX_SLOTS  machine cap $MAX_MACHINE  peak ${PEAK} GiB  avail $(avail_gib) GiB"
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
    if [ -f "$RUNS/${1}_s${2}/provenance.json" ]; then
      echo "$(date -Is) skip ${1}_s${2} (done)"
      i=$((i + 1))
      continue
    fi
    if mem_ok && [ "$(( $(heavy) + 1 ))" -le "$MAX_MACHINE" ]; then
      run_job "$1" "$2" &
      pids+=("$!")
      i=$((i + 1))
      waits=0
      sleep 30
      continue
    fi
    if [ $((waits % 30)) -eq 0 ]; then echo "$(date -Is) wait: avail $(avail_gib) GiB, heavy python $(heavy)"; fi
    waits=$((waits + 1))
  fi
  sleep 10
done
echo "$(date -Is) all jobs returned"

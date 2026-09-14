#!/usr/bin/env bash
# l2split_rlmnist_0914 -- spec sections 2.3 and 8.  Phase 1: the 4 verification runs (ref / l2, seed 0) first,
# then l2wt / l2rest for seeds 0-9 x R, LR (44 runs).  Then verdict --reuse-check; on REUSE_MISMATCH, phase 2
# re-runs ref / l2 for seeds 1-9 (36 runs).  Refuses to start unless checks.json says all_pass, the code is
# committed and HEAD is pushed.  Waits (never aborts) while fewer than 4 slots fit in memory, and every job
# waits before it starts until MemAvailable >= 6 GiB + 1.2 x peak RSS (other sessions share this machine).
# 2026-09-14 10:00 revision (logistics only, spec 6 deviation recorded in summary): the first launch was stopped by
# the harness for low memory when other sessions started jobs next to 11 of ours.  Now at most 6 slots, a 6 GiB
# reserve, and job starts are serialised with a 25 s pause so each new process's RSS shows in MemAvailable
# before the next memory check.
#   nohup setsid bash analysis/l2split_rlmnist_0914/launch.sh > results/l2split_rlmnist_0914/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/l2split_rlmnist_0914
MOD=src/l2split_rlmnist_0914.py
PY=/usr/bin/python3
[ -f "$MOD" ] || { echo "ABORT: $MOD not found"; exit 1; }
$PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
[ -z "$(git status --porcelain -- src analysis/l2split_rlmnist_0914)" ] || { echo "ABORT: uncommitted code"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
PEAK=$($PY -c "import json; print(json.load(open('$OUT/checks.json'))['S-cost']['peak_rss_gib_max'])")
CAP=$($PY -c "import json; print(json.load(open('$OUT/checks.json'))['S-cost']['slots'])")
MAXSLOTS=6
RESERVE=6.0
LOCK="$OUT/logs/.start.lock"
slots_now() { awk -v peak="$PEAK" -v cap="$CAP" -v mx="$MAXSLOTS" -v rs="$RESERVE" '/^MemAvailable:/{n=int(($2/1048576-rs)/(1.2*peak)); if(n>cap)n=cap; if(n>mx)n=mx; if(n<0)n=0; print n}' /proc/meminfo; }
N=$(slots_now)
while [ "$N" -lt 4 ]; do echo "$(date -Is) waiting: only $N slots fit in memory"; sleep 60; N=$(slots_now); done
mkdir -p "$OUT/runs" "$OUT/logs"

run_job() {
  local act=$1 arm=$2 seed=$3
  local name="${act}_${arm}_s${seed}"
  if [ -f "$OUT/runs/$name/provenance.json" ]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  exec 9>"$LOCK"
  flock 9
  while awk -v peak="$PEAK" -v rs="$RESERVE" '/^MemAvailable:/{exit !($2/1048576 < rs + 1.2*peak)}' /proc/meminfo; do sleep 30; done
  echo "$(date -Is) start $name"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" "$MOD" --act "$act" --arm "$arm" --seeds "$seed" \
    --tasks 50 --epochs 400 --lr 0.001 --threads 1 \
    --out "$OUT/runs/$name" > "$OUT/logs/$name.log" 2>&1 &
  local pid=$!
  sleep 25
  flock -u 9
  exec 9>&-
  wait "$pid" && echo "$(date -Is) done  $name" || echo "$(date -Is) FAIL  $name (exit $?)"
}
export -f run_job
export OUT MOD PY PEAK RESERVE LOCK

echo "$(date -Is) launch phase 1: git $(git rev-parse HEAD)  slots $N  $(free -g | awk '/^Mem:/{print "avail "$7" GiB"}')"
{
  for pair in R:ref R:l2 LR:ref LR:l2; do echo "${pair%%:*} ${pair##*:} 0"; done
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    for pair in R:l2wt R:l2rest LR:l2wt LR:l2rest; do echo "${pair%%:*} ${pair##*:} $seed"; done
  done
} | xargs -P "$N" -L 1 bash -c 'run_job "$0" "$1" "$2"'
echo "$(date -Is) phase 1 returned"

if $PY analysis/l2split_rlmnist_0914/reuse.py; then
  echo "$(date -Is) REUSE_OK: ref / l2 come from results/wcap_rlmnist_0914/runs"
else
  echo "$(date -Is) REUSE_MISMATCH: phase 2 re-runs ref / l2 for seeds 1-9"
  N=$(slots_now); while [ "$N" -lt 4 ]; do sleep 60; N=$(slots_now); done
  for seed in 1 2 3 4 5 6 7 8 9; do
    for pair in R:ref R:l2 LR:ref LR:l2; do echo "${pair%%:*} ${pair##*:} $seed"; done
  done | xargs -P "$N" -L 1 bash -c 'run_job "$0" "$1" "$2"'
  echo "$(date -Is) phase 2 returned"
fi
echo "$(date -Is) all jobs returned"

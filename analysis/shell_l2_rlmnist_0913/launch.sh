#!/usr/bin/env bash
# shell_l2_rlmnist_0913 -- spec section 6: 8 arms x seeds 0-9 x 50 tasks x 400 epochs, 16 at a time, 1 thread each.
# Refuses to start unless checks.json says all_pass, the code is committed and HEAD is pushed.
# Queue order: seed 0 (8 arms), seed 1 (8 arms), ... so an early stop leaves complete seeds.
# Shard and log names are flat and enumerated explicitly; a shard with provenance.json is skipped (relaunch-safe).
#   nohup setsid bash analysis/shell_l2_rlmnist_0913/launch.sh > results/shell_l2_rlmnist_0913/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/shell_l2_rlmnist_0913
MOD=src/shell_l2_rlmnist_0913.py
PY=.venv/bin/python
[ -f "$MOD" ] || { echo "ABORT: $MOD not found"; exit 1; }
$PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
[ -z "$(git status --porcelain -- src analysis/shell_l2_rlmnist_0913)" ] || { echo "ABORT: uncommitted code"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
avail=$(free -g | awk '/^Mem:/{print $7}'); [ "$avail" -lt 20 ] && { echo "ABORT: only ${avail} GiB available"; exit 1; }
mkdir -p "$OUT/runs" "$OUT/logs"

run_job() {
  local act=$1 reg=$2 seed=$3
  local name="${act}_${reg}_s${seed}"
  if [ -f "$OUT/runs/$name/provenance.json" ]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  local regarg=$reg; [ "$reg" != none ] && regarg="$reg:1e-3"
  echo "$(date -Is) start $name"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" "$MOD" --act "$act" --reg "$regarg" --seeds "$seed" \
    --tasks 50 --epochs 400 --lr 0.001 --c 0.6 --beta 0.01 --threads 1 \
    --out "$OUT/runs/$name" > "$OUT/logs/$name.log" 2>&1 \
    && echo "$(date -Is) done  $name" || echo "$(date -Is) FAIL  $name (exit $?)"
}
export -f run_job
export OUT MOD PY

echo "$(date -Is) launch: git $(git rev-parse HEAD)"
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for act in R SNA; do
    for reg in none l2 l2init shell; do
      echo "$act $reg $seed"
    done
  done
done | xargs -P 16 -L 1 bash -c 'run_job "$0" "$1" "$2"'
echo "$(date -Is) all jobs returned"

#!/usr/bin/env bash
# wcap_rlmnist_0914 -- spec section 8: 8 arms x seeds 0-9 x 50 tasks x 400 epochs, N at a time, 1 thread each.
# Refuses to start unless checks.json says all_pass, the code is committed and HEAD is pushed.
# N = S-cost slots, re-derived from MemAvailable right now (never more than S-cost's); N < 4 aborts.
# Queue order: seed 0 (8 arms), seed 1 (8 arms), ... so an early stop leaves complete seeds.
# Shard and log names are flat and enumerated explicitly; a shard with provenance.json is skipped (relaunch-safe).
#   nohup setsid bash analysis/wcap_rlmnist_0914/launch.sh > results/wcap_rlmnist_0914/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/wcap_rlmnist_0914
MOD=src/wcap_rlmnist_0914.py
PY=/usr/bin/python3
[ -f "$MOD" ] || { echo "ABORT: $MOD not found"; exit 1; }
$PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
[ -z "$(git status --porcelain -- src analysis/wcap_rlmnist_0914)" ] || { echo "ABORT: uncommitted code"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
N=$($PY - <<'EOF'
import json
c = json.load(open("results/wcap_rlmnist_0914/checks.json"))["S-cost"]
avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0] / 2**20
print(max(0, min(c["slots"], int((avail - 4.0) // (1.2 * c["peak_rss_gib_max"])))))
EOF
)
[ "$N" -ge 4 ] || { echo "ABORT: only $N slots fit in memory now"; exit 1; }
mkdir -p "$OUT/runs" "$OUT/logs"

run_job() {
  local act=$1 arm=$2 seed=$3
  local name="${act}_${arm}_s${seed}"
  if [ -f "$OUT/runs/$name/provenance.json" ]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  echo "$(date -Is) start $name"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" "$MOD" --act "$act" --arm "$arm" --seeds "$seed" \
    --tasks 50 --epochs 400 --lr 0.001 --threads 1 \
    --out "$OUT/runs/$name" > "$OUT/logs/$name.log" 2>&1 \
    && echo "$(date -Is) done  $name" || echo "$(date -Is) FAIL  $name (exit $?)"
}
export -f run_job
export OUT MOD PY

echo "$(date -Is) launch: git $(git rev-parse HEAD)  slots $N  $(free -g | awk '/^Mem:/{print "avail "$7" GiB"}')"
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for pair in LR:ref LR:l2 LR:l2init LR:capT1 LR:cap2 R:ref R:l2 R:cap2; do
    echo "${pair%%:*} ${pair##*:} $seed"
  done
done | xargs -P "$N" -L 1 bash -c 'run_job "$0" "$1" "$2"'
echo "$(date -Is) all jobs returned"

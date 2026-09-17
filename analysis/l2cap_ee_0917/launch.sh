#!/usr/bin/env bash
# l2cap_ee_0917 -- spec section 8: 5 arms x seeds 0-9 x 100 tasks x 80 epochs (ELU->ELU, ledger off), N at a time, 1 thread each.
# Refuses to start unless checks.json says all_pass, the code and the spec are committed and HEAD is pushed.
# N = min(MAX_SLOTS, S-cost slots, what fits in MemAvailable right now); N < 4 aborts.  MAX_SLOTS defaults
# to 6 because this is a shared desktop (spec section 8); the core count is never the budget.
# Queue order: seed 0 (5 arms), seed 1 (4 arms), ... so an early stop leaves complete seeds.
# Shard and log names are flat (arm_sSEED, no '/'); a shard with provenance.json is skipped (relaunch-safe).
#   nohup setsid bash analysis/l2cap_ee_0917/launch.sh > results/l2cap_ee_0917/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/l2cap_ee_0917
MOD=src/l2cap_ee_run_0917.py
PY=/usr/bin/python3
MAX_SLOTS=${MAX_SLOTS:-6}
[ -f "$MOD" ] || { echo "ABORT: $MOD not found"; exit 1; }
$PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
[ -z "$(git status --porcelain -- src analysis/l2cap_ee_0917 specs/spec_l2cap_ee_0917.md)" ] \
  || { echo "ABORT: uncommitted code or spec"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
N=$($PY - "$MAX_SLOTS" <<'EOF'
import json, sys
c = json.load(open("results/l2cap_ee_0917/checks.json"))["S-cost"]
avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0] / 2**20
fit = int((avail - 4.0) // (1.2 * c["peak_rss_gib_max"]))     # 4 GiB left for the desktop, 20% per job
print(max(0, min(int(sys.argv[1]), c["slots"], fit)))
EOF
)
[ "$N" -ge 4 ] || { echo "ABORT: only $N slots fit in memory now"; exit 1; }
mkdir -p "$OUT/runs" "$OUT/logs"

run_job() {
  local arm=$1 seed=$2
  local name="${arm}_s${seed}"
  if [ -f "$OUT/runs/$name/provenance.json" ]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  echo "$(date -Is) start $name"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" "$MOD" --arm "$arm" --seeds "$seed" \
    --tasks 100 --epochs 80 --lr 0.001 --threads 1 \
    --out "$OUT/runs/$name" > "$OUT/logs/$name.log" 2>&1 \
    && echo "$(date -Is) done  $name" || echo "$(date -Is) FAIL  $name (exit $?)"
}
export -f run_job
export OUT MOD PY

echo "$(date -Is) launch: git $(git rev-parse HEAD)  slots $N  $(free -g | awk '/^Mem:/{print "avail "$7" GiB"}')"
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for arm in ref cap1 cap2 cap12 cap12_bfix; do
    echo "$arm $seed"
  done
done | xargs -P "$N" -L 1 bash -c 'run_job "$0" "$1"'
echo "$(date -Is) all jobs returned"

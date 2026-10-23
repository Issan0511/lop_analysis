#!/usr/bin/env bash
# lc_elu_lr_0917 -- spec section 7: the 2x2 (4 arms x seeds 0-4; lr 1e-4 arms to 150 tasks), then the two anchors.
# Refuses to start unless checks.json says all_pass, the code and the spec are committed and HEAD is pushed.
# N = min(MAX_SLOTS, S-cost slots, what fits in MemAvailable now); N < 3 aborts.  Shared desktop: MAX_SLOTS 6.
# Each job start is serialised (flock) and waits for MemAvailable >= 4 GiB + 1.2 x peak RSS, then 25 s.
# A run with provenance.json is skipped; a run with a checkpoint resumes (the runner does it).
# touch results/lc_elu_lr_0917/STOP to stop starting new jobs.
#   setsid nohup bash analysis/lc_elu_lr_0917/launch.sh > results/lc_elu_lr_0917/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/lc_elu_lr_0917
MOD=src/lc_elu_lr_0917.py
PY=/usr/bin/python3
MAX_SLOTS=${MAX_SLOTS:-6}
$PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
[ -z "$(git status --porcelain -- src analysis/lc_elu_lr_0917 specs/spec_lc_elu_lr_0917.md)" ] \
  || { echo "ABORT: uncommitted code or spec"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
read -r N NEED_KB < <($PY - "$MAX_SLOTS" <<'EOF'
import json, sys
c = json.load(open("results/lc_elu_lr_0917/checks.json"))["checks"]["S-cost"]
avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0] / 2**20
fit = int((avail - 4.0) // (1.2 * c["peak_rss_gib"]))
need_kb = int((4.0 + 1.2 * c["peak_rss_gib"]) * 2**20)
print(max(0, min(int(sys.argv[1]), c["slots"], fit)), need_kb)
EOF
)
[ "$N" -ge 3 ] || { echo "ABORT: only $N slots fit in memory now"; exit 1; }
mkdir -p "$OUT/runs" "$OUT/logs"

run_job() {
  local arm=$1 seed=$2 tasks=$3
  local name="${arm}_s${seed}"
  if [ -f "$OUT/runs/$name/provenance.json" ]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  (
    flock 9
    if [ -f "$OUT/STOP" ]; then echo "$(date -Is) STOP: not starting $name"; exit 0; fi
    while [ "$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)" -lt "$NEED_KB" ]; do sleep 30; done
    echo "$(date -Is) start $name ($tasks tasks)"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" "$MOD" --arm "$arm" --seed "$seed" \
      --tasks "$tasks" --threads 1 --out "$OUT/runs/$name" >> "$OUT/logs/$name.log" 2>&1 9>&- &
    echo $! > "$OUT/logs/$name.pid"
    sleep 25
  ) 9> "$OUT/.launch.lock"
  [ -f "$OUT/logs/$name.pid" ] || return 0
  local pid; pid=$(cat "$OUT/logs/$name.pid")
  while kill -0 "$pid" 2>/dev/null; do sleep 20; done
  if [ -f "$OUT/runs/$name/provenance.json" ]; then echo "$(date -Is) done  $name"; else echo "$(date -Is) FAIL  $name"; fi
  rm -f "$OUT/logs/$name.pid"
}
export -f run_job
export OUT MOD PY NEED_KB

echo "$(date -Is) launch: git $(git rev-parse HEAD)  slots $N  need_kb $NEED_KB  $(free -g | awk '/^Mem:/{print "avail "$7" GiB"}')"
{
  for seed in 0 1 2 3 4; do
    echo "E1_lr1e3 $seed 50"; echo "E36_lr1e3 $seed 50"; echo "E1_lr1e4 $seed 150"; echo "E36_lr1e4 $seed 150"
  done
  for seed in 0 1 2 3 4; do echo "R_lr1e4 $seed 50"; done
  for seed in 0 1 2 3 4; do echo "LK08_lr1e3 $seed 50"; done
} | xargs -P "$N" -L 1 bash -c 'run_job "$0" "$1" "$2"'
echo "$(date -Is) all jobs returned"

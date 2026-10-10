#!/usr/bin/env bash
# adamw_dose_1010 -- spec section 8, box 2: adamw5 and adamw10 on seeds 0-9 (R = 10 stacked, 30 tasks, graph, 2 threads),
# one GPU process at a time under the shared lock /tmp/lop_analysis_gpu.lock (blocking flock).  With CSNP=1 it runs
# instead the post-run C-snp cell: adamw10 on the calibration seeds 100-109 (spec 5.6, report only).
# Before each start: MemAvailable >= 6 GiB and free VRAM >= 5 GiB (resp_cifar5p1_1007's OOM lesson), else it waits.
# A cell with provenance.json is skipped.  Refuses to start unless checks.json is all_pass, the tree is committed
# and HEAD is pushed (the CLI refuses a registered output on top of that).
#   nohup setsid bash analysis/adamw_dose_1010/launch_c51.sh > results/adamw_dose_1010/logs/launch_c51.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/adamw_dose_1010
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
LOCK=/tmp/lop_analysis_gpu.lock
"$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
[ -z "$(git status --porcelain -- src analysis specs $OUT/checks.json)" ] \
  || { echo "ABORT: uncommitted code, spec or checks"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
mkdir -p "$OUT/c51" "$OUT/logs"
avail_gib() { awk '/^MemAvailable:/{printf "%.2f", $2/1048576}' /proc/meminfo; }
vram_free_gib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%.2f", $1/1024}'; }
if [ "${CSNP:-0}" = 1 ]; then
  cells=("adamw10 100-109 csnp_adamw10")
else
  cells=("adamw5 0-9 adamw5" "adamw10 0-9 adamw10")
fi
echo "$(date -Is) launch c51: git $(git rev-parse HEAD)  cells ${cells[*]}"
for c in "${cells[@]}"; do
  set -- $c
  arm=$1; seeds=$2; name=$3
  if [ -f "$OUT/c51/$name/provenance.json" ]; then echo "$(date -Is) skip $name (done)"; continue; fi
  until awk -v a="$(avail_gib)" -v v="$(vram_free_gib)" 'BEGIN{exit !(a >= 6 && v >= 5)}'; do
    echo "$(date -Is) wait: avail $(avail_gib) GiB, vram free $(vram_free_gib) GiB"; sleep 60
  done
  echo "$(date -Is) start $name (waiting for the GPU lock)"
  if flock "$LOCK" "$PY" -m src.adamw_dose_1010_run c51 --arm "$arm" --seeds "$seeds" --out "$OUT/c51/$name" \
       > "$OUT/logs/c51_$name.log" 2>&1; then
    echo "$(date -Is) done  $name"
  else
    echo "$(date -Is) FAIL  $name (exit $?)"
  fi
done
echo "$(date -Is) c51 jobs returned"

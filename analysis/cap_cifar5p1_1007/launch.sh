#!/usr/bin/env bash
# cap_cifar5p1_1007 -- spec section 9: ref -> cap1 -> cap2 -> cap12, seeds 0-9 stacked (R = 10), one GPU
# process at a time, each under the shared GPU lock.  Refuses to start unless checks.json is all_pass on
# these very sources, src/analysis/specs are committed, HEAD is pushed and >= 6 GiB RAM is available.
# The engine itself re-checks the same in admit() before writing into results/cap_cifar5p1_1007/<arm>.
# An arm whose provenance.json exists is skipped (relaunch-safe); an arm that fails stops the queue.
#   nohup setsid bash analysis/cap_cifar5p1_1007/launch.sh > results/cap_cifar5p1_1007/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/cap_cifar5p1_1007
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
LOCK=/tmp/lop_analysis_gpu.lock
GPU_FREE_MIB=2500          # one stacked run holds ~1.5 GB (data 0.74 GB + evaluation stacks + context)
[ -f src/cap_cifar5p1_1007.py ] || { echo "ABORT: engine missing"; exit 1; }
"$PY" - <<'EOF' || { echo "ABORT: checks.json is not all_pass on these sources"; exit 1; }
import hashlib, json, sys
ck = json.load(open("results/cap_cifar5p1_1007/checks.json"))
now = {s: hashlib.sha256(open(s, "rb").read()).hexdigest() for s in ck["source_sha256"]}
need = ("S-nochange", "S-cap", "S-radius", "S-graph", "S-fresh", "S-verdict", "S-CLI")
sys.exit(0 if ck.get("all_pass") and now == ck["source_sha256"] and all(k in ck for k in need) else 1)
EOF
[ -z "$(git status --porcelain -- src analysis specs)" ] || { echo "ABORT: uncommitted code or spec"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
avail=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
[ "$avail" -ge 6 ] || { echo "ABORT: only ${avail} GiB RAM available (need 6)"; exit 1; }
mkdir -p "$OUT/logs"
echo "$(date -Is) launch: git $(git rev-parse HEAD)  RAM avail ${avail} GiB"

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

for arm in ref cap1 cap2 cap12; do
  if [ -f "$OUT/$arm/provenance.json" ]; then echo "$(date -Is) skip $arm (done)"; continue; fi
  for i in $(seq 1 30); do                       # wait (up to 30 min) for GPU memory, never squeeze in
    [ "$(gpu_free)" -ge "$GPU_FREE_MIB" ] && break
    echo "$(date -Is) waiting: GPU free $(gpu_free) MiB < $GPU_FREE_MIB"; sleep 60
  done
  [ "$(gpu_free)" -ge "$GPU_FREE_MIB" ] || { echo "ABORT: GPU memory never freed"; exit 1; }
  echo "$(date -Is) start $arm  (GPU free $(gpu_free) MiB)"
  if flock "$LOCK" "$PY" -m src.cap_cifar5p1_1007 run --arm "$arm" --threads 2 > "$OUT/logs/${arm}.log" 2>&1; then
    echo "$(date -Is) done  $arm"
  else
    echo "$(date -Is) FAIL  $arm"; exit 1
  fi
done
echo "$(date -Is) all arms returned"

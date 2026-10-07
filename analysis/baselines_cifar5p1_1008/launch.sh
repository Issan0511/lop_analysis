#!/usr/bin/env bash
# baselines_cifar5p1_1008 -- spec 9.  Two stages, each serial (one GPU process at a time):
#   launch.sh calib   none + S&P 9 + CBP 3 + ReDo 9 cells on seeds 100-109 (R = 10 per process)
#   launch.sh main    none / snp / cbp / redo x {seeds 0-9, 10-19} with the committed selected.json
# Refuses to start unless checks.json is all_pass on these very sources and src/analysis/specs are
# committed (main: and calib/selected.json is committed).  The engine re-checks the same in admit().
# Every run waits until the GPU has >= 6 GiB free on top of the run's own ~2.5 GiB and RAM has
# >= 6 GiB available, then takes the shared GPU lock.  Runs with a provenance.json are skipped
# (relaunch-safe); a failing run stops the queue.
#   setsid nohup bash analysis/baselines_cifar5p1_1008/launch.sh calib > results/baselines_cifar5p1_1008/logs/launch_calib.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE=${1:?usage: launch.sh calib|main}
RUN=baselines_cifar5p1_1008
OUT=results/$RUN
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
LOCK=/tmp/lop_analysis_gpu.lock
NEED_GPU_MIB=8704
NEED_RAM_MIB=6144
WAIT_MAX_MIN=90

[ -f src/$RUN.py ] || { echo "ABORT: engine missing"; exit 1; }
"$PY" - <<'PYEOF' || { echo "ABORT: checks.json is not all_pass on these sources"; exit 1; }
import hashlib, json, sys
sys.path.insert(0, ".")
ck = json.load(open("results/baselines_cifar5p1_1008/checks.json"))
srcs = ("src/baselines_cifar5p1_1008.py", "analysis/baselines_cifar5p1_1008/checks.py",
        "analysis/baselines_cifar5p1_1008/verdict.py")
now = {s: hashlib.sha256(open(s, "rb").read()).hexdigest() for s in srcs}
need = ("S-nochange", "S-host-repro", "S-snp", "S-cbp", "S-redo", "S-graph", "S-fresh", "S-select",
        "S-verdict", "S-CLI")
sys.exit(0 if ck.get("all_pass") and ck.get("source_sha256") == now and all(k in ck for k in need) else 1)
PYEOF
paths="src analysis specs"
if [ "$STAGE" = main ]; then
  [ -f $OUT/calib/selected.json ] || { echo "ABORT: no selected.json"; exit 1; }
  git ls-files --error-unmatch $OUT/calib/selected.json > /dev/null 2>&1 || { echo "ABORT: selected.json not committed"; exit 1; }
  paths="$paths $OUT/calib/selected.json"
fi
[ -z "$(git status --porcelain -- $paths)" ] || { echo "ABORT: uncommitted code, spec or selection"; exit 1; }
mkdir -p $OUT/logs
echo "$(date -Is) launch $STAGE: git $(git rev-parse HEAD)"

JOBS=$("$PY" - "$STAGE" <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from src import baselines_cifar5p1_1008 as E
stage = sys.argv[1]
def args(cfg, seeds):
    a = ["--method", cfg["method"]]
    for k in ("eps", "sigma", "rho", "tau", "period"):
        if k in cfg:
            a += [f"--{k}", str(cfg[k])]
    return a + ["--seeds", seeds]
jobs = []
if stage == "calib":
    jobs.append(("none", args(E.make_cfg("none"), "100-109")))
    for m in ("snp", "cbp", "redo"):
        for h in E.GRID[m]:
            cfg = E.make_cfg(m, **h)
            jobs.append((E.cfg_tag(cfg), args(cfg, "100-109")))
    root = "calib"
elif stage == "main":
    sel = json.load(open("results/baselines_cifar5p1_1008/calib/selected.json"))["selected"]
    for m in ("none", "snp", "cbp", "redo"):
        cfg = E.make_cfg("none") if m == "none" else E.make_cfg(m, **E.hyper(sel[m]))
        for g in ("0-9", "10-19"):
            jobs.append((f"{m}_s{g}", args(cfg, g)))
    root = "main"
else:
    raise SystemExit(f"unknown stage {stage}")
for name, a in jobs:
    print(f"{root}/{name}|{' '.join(a)}")
PYEOF
)

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
ram_avail() { awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo; }

while IFS='|' read -r dir argv; do
  [ -n "$dir" ] || continue
  if [ -f "$OUT/$dir/provenance.json" ]; then echo "$(date -Is) skip $dir (done)"; continue; fi
  log="$OUT/logs/${dir//\//_}.log"
  waited=0
  while true; do
    if [ "$(gpu_free)" -ge "$NEED_GPU_MIB" ] && [ "$(ram_avail)" -ge "$NEED_RAM_MIB" ]; then
      exec 9>"$LOCK"
      flock 9
      if [ "$(gpu_free)" -ge "$NEED_GPU_MIB" ] && [ "$(ram_avail)" -ge "$NEED_RAM_MIB" ]; then
        echo "$(date -Is) start $dir (GPU free $(gpu_free) MiB, RAM $(ram_avail) MiB): $argv"
        if "$PY" -m src.$RUN run $argv --threads 2 > "$log" 2>&1; then
          echo "$(date -Is) done  $dir"
          flock -u 9; exec 9>&-
          break
        else
          echo "$(date -Is) FAIL  $dir (see $log)"
          flock -u 9; exit 1
        fi
      fi
      flock -u 9; exec 9>&-
    fi
    if [ "$waited" -ge $((WAIT_MAX_MIN * 60)) ]; then
      echo "$(date -Is) ABORT: GPU memory or RAM never freed (GPU $(gpu_free) MiB, RAM $(ram_avail) MiB)"; exit 1
    fi
    [ "$waited" = 0 ] && echo "$(date -Is) waiting for $dir: GPU free $(gpu_free) MiB, RAM $(ram_avail) MiB"
    sleep 60; waited=$((waited + 60))
  done
done <<< "$JOBS"
echo "$(date -Is) all $STAGE runs returned"

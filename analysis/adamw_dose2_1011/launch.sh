#!/usr/bin/env bash
# adamw_dose2_1011 -- spec section 8.  A1: wd1e0, wd1e-1, wd3e-2 (RL-CIFAR, R = 10 stacked, tasks 2-50 from the
# cap_cifar_ee_0920 prefix, 2 threads each); A2: l2c1e-2, l2c5e-3 (5+1 CIFAR, R = 10, 30 tasks) after A1 has started.
# The GPU is shared with another agent's run (no exclusive lock is taken).  Concurrent A1 processes:
#   N = min(3, floor((16311 - used_MiB_now - 2048) / (1.2 * PEAK_MiB)))  (PEAK from checks.json S-cost),
# re-evaluated before each start; it also waits for MemAvailable - 6 GiB >= 1.2 * RSS_GiB per new process.
# Refuses to start unless checks.json is all_pass, the tree is committed and HEAD is pushed (the CLI refuses a
# registered output on top of that).  An arm with done.json is skipped; a stopped arm resumes from its checkpoint.
# Touch results/adamw_dose2_1011/STOP to stop starting new arms (running arms: touch rlcifar/<arm>/STOP).
#   nohup setsid bash analysis/adamw_dose2_1011/launch.sh > results/adamw_dose2_1011/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/adamw_dose2_1011
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
MOD=src.adamw_dose2_1011_run
"$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/checks.json')).get('all_pass') else 1)" \
  || { echo "ABORT: $OUT/checks.json is not all_pass"; exit 1; }
[ -z "$(git status --porcelain -- src analysis specs $OUT/checks.json)" ] \
  || { echo "ABORT: uncommitted code, spec or checks"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
PEAK=$("$PY" -c "import json; c=json.load(open('$OUT/checks.json'))['S-cost']['rlcifar']; print(max(c['gpu_mib_nvidia_smi'], c['peak_cuda_reserved_mib']))")
RSS=$("$PY" -c "import json; print(json.load(open('$OUT/checks.json'))['S-cost']['rlcifar']['peak_rss_gib'])")
mkdir -p "$OUT/rlcifar" "$OUT/c51" "$OUT/logs"
used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
avail_gib() { awk '/^MemAvailable:/{printf "%.2f", $2/1048576}' /proc/meminfo; }
nmem() { awk -v u="$(used_mib)" -v p="$PEAK" 'BEGIN{n=int((16311 - u - 2048) / (1.2 * p)); print (n > 3 ? 3 : n)}'; }
ram_ok() { awk -v a="$(avail_gib)" -v r="$RSS" 'BEGIN{exit !((a - 6) >= 1.2 * r)}'; }
heavy() { ps -eo args= | grep -c "^/home/issan/Projects/claude/proj_004_drift/.venv/bin/python" || true; }
echo "$(date -Is) launch: git $(git rev-parse HEAD)  peak ${PEAK} MiB/process  rss ${RSS} GiB  gpu used $(used_mib) MiB  N_mem $(nmem)  proj_004 python $(heavy)  all .venv python $(pgrep -c -f '[.]venv/bin/python' || true)"
# spec 14.9: measure the float32 lines on this GPU right before the start and keep the record
"$PY" -m analysis.adamw_dose2_1011.lines --out "$OUT/launch_record_lines.json" > /dev/null
"$PY" - "$OUT" <<'PYEOF'
import json, subprocess, sys, time
out = sys.argv[1]
lines = json.load(open(f"{out}/launch_record_lines.json"))
smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip().splitlines()
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
rec = {"at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "git_head": head, "lines": lines, "nvidia_smi": smi,
       "compute_apps": apps, "z0_matches_rule": lines["z0_last_zero"] < -103.97207708399179 <= lines["z0_first_nonzero"],
       "zf_matches_rule": lines["zf"] == -16.982107162475586}
json.dump(rec, open(f"{out}/launch_record.json", "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("at", "z0_matches_rule", "zf_matches_rule", "nvidia_smi")}))
if not (rec["z0_matches_rule"] and rec["zf_matches_rule"]):
    sys.exit("ABORT: the measured float32 lines differ from the rule's")
PYEOF
pids=()
for arm in wd1e0 wd1e-1 wd3e-2; do
  if [ -f "$OUT/rlcifar/$arm/done.json" ]; then echo "$(date -Is) skip $arm (done)"; continue; fi
  while :; do
    alive=()
    for p in "${pids[@]}"; do if kill -0 "$p" 2>/dev/null; then alive+=("$p"); fi; done
    pids=("${alive[@]}")
    if [ -f "$OUT/STOP" ]; then echo "$(date -Is) STOP file: not starting $arm"; break 2; fi
    n=$(nmem)
    if [ "${#pids[@]}" -lt "$n" ] && ram_ok; then break; fi
    echo "$(date -Is) wait for $arm: running ${#pids[@]}, N_mem $n, gpu used $(used_mib) MiB, avail $(avail_gib) GiB"
    sleep 60
  done
  echo "$(date -Is) start $arm  (gpu used $(used_mib) MiB, N_mem $(nmem), running ${#pids[@]})"
  "$PY" -m "$MOD" rlcifar --arm "$arm" --out "$OUT/rlcifar/$arm" > "$OUT/logs/rlcifar_$arm.log" 2>&1 &
  pids+=("$!")
  sleep 30
done
PEAK2=$("$PY" -c "import json; print(json.load(open('$OUT/checks.json'))['S-cost']['c51']['gpu_mib_nvidia_smi'])")
RSS2=$("$PY" -c "import json; print(json.load(open('$OUT/checks.json'))['S-cost']['c51']['peak_rss_gib'])")
gpu2_ok() { awk -v u="$(used_mib)" -v p="$PEAK2" 'BEGIN{exit !((16311 - u - 2048) >= 1.2 * p)}'; }
ram2_ok() { awk -v a="$(avail_gib)" -v r="$RSS2" 'BEGIN{exit !((a - 6) >= 1.2 * r)}'; }
for arm in l2c1e-2 l2c5e-3; do
  if [ -f "$OUT/c51/$arm/provenance.json" ]; then echo "$(date -Is) skip $arm (done)"; continue; fi
  until gpu2_ok && ram2_ok; do
    echo "$(date -Is) wait for c51 $arm: gpu used $(used_mib) MiB, avail $(avail_gib) GiB"; sleep 60
  done
  echo "$(date -Is) start c51 $arm  (gpu used $(used_mib) MiB)"
  if "$PY" -m "$MOD" c51 --arm "$arm" --seeds 0-9 --out "$OUT/c51/$arm" > "$OUT/logs/c51_$arm.log" 2>&1; then
    echo "$(date -Is) done c51 $arm"
  else
    echo "$(date -Is) FAIL c51 $arm (exit $?)"
  fi
done
for p in "${pids[@]}"; do wait "$p" || echo "$(date -Is) an A1 arm exited non-zero (pid $p)"; done
echo "$(date -Is) launch finished"

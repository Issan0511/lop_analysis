#!/usr/bin/env bash
# cnn_center_feat_1011: queue runner for the registered runs (spec §8; P181's procedure, spec §12 there).
#   nohup setsid bash analysis/cnn_center_feat_1011/launch.sh [MAX_CONC] \
#       > results/cnn_center_feat_1011/logs/launcher.log 2>&1 < /dev/null &
#   WITH_CHS=1 adds the proposed control CHs (seed 10) at the end of the queue.
# Before EVERY start the budget is recomputed from the machine as it is at that moment:
#   GPU: nvidia-smi free - 2048 MiB          >= 1.25 * M_PROC   (M_PROC = K-cost's nvidia-smi peak)
#   RAM: MemAvailable - 6 GiB - RSS_LOAD     >= 1.2  * RSS      (RSS = steady RSS of one process,
#                                                                RSS_LOAD = its extra peak while loading)
#   and fewer than MAX_CONC of these runs alive (default 2: one process already saturates the GPU,
#   so a second only keeps it busy across task ends; the rest is left to the other agents).
# A run is skipped when its provenance.json exists (done) or when a process for it is alive
# (no duplicates); a run with state.pt resumes.  Every decision goes to logs/launch_record.txt.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
OUT="$REPO/results/cnn_center_feat_1011"
MAX_CONC=${1:-2}
M_PROC=${M_PROC:-1200}      # MiB, K-cost (nvidia-smi, per process, CUDA context included)
RSS=${RSS:-2400}            # MiB, steady VmRSS of one CHf process (K-cost)
RSS_LOAD=${RSS_LOAD:-1100}  # MiB, VmHWM - steady (train-set channel mean in float64 at load)
QUEUE=("CHf 10-19" "Hf 10-19" "Hs 10")
if [ "${WITH_CHS:-0}" = 1 ]; then QUEUE+=("CHs 10"); fi
mkdir -p "$OUT/logs"
REC="$OUT/logs/launch_record${DRY:+_dry}.txt"
note() { echo "$(date '+%F %T') $*" | tee -a "$REC"; }
alive() { pgrep -f "[s]rc/cnn_center_feat_1011\.py run --arm $1 --seeds" > /dev/null; }
n_alive() { local n=0; for it in "${QUEUE[@]}"; do set -- $it; alive "$1" && n=$((n + 1)); done; echo $n; }

note "launcher start HEAD=$(git -C "$REPO" rev-parse HEAD) dirty=[$(git -C "$REPO" status --porcelain -- src analysis | tr '\n' ' ')] MAX_CONC=$MAX_CONC M_PROC=$M_PROC RSS=$RSS RSS_LOAD=$RSS_LOAD queue=[${QUEUE[*]}]"
for it in "${QUEUE[@]}"; do
  set -- $it; arm=$1; seeds=$2
  if [ -f "$OUT/$arm/provenance.json" ]; then note "skip $arm: done"; continue; fi
  if alive "$arm"; then note "skip $arm: already running"; continue; fi
  while true; do
    run=$(n_alive)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    avail=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 ))
    gpu_room=$(( free - 2048 - (M_PROC * 125 + 99) / 100 ))          # safety factors rounded up
    ram_room=$(( avail - 6144 - RSS_LOAD - (RSS * 120 + 99) / 100 ))
    if [ "$run" -lt "$MAX_CONC" ] && [ "$gpu_room" -ge 0 ] && [ "$ram_room" -ge 0 ]; then break; fi
    sleep 60
  done
  npy=$(pgrep -c -f '[p]roj_004_drift/\.venv/bin/python' || true)
  note "start $arm seeds $seeds: alive=$run free_gpu=${free}MiB avail_ram=${avail}MiB gpu_room=${gpu_room} ram_room=${ram_room} proj_004_drift_python=${npy}"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$REC"
  if [ "${DRY:-0}" = 1 ]; then note "DRY: would launch $arm"; continue; fi
  nohup setsid "$PY" "$REPO/src/cnn_center_feat_1011.py" run --arm "$arm" --seeds "$seeds" \
    --out "$OUT/$arm" > "$OUT/logs/$arm.log" 2>&1 < /dev/null &
  note "launched $arm pid $!"
  sleep 90      # past the load peak (VmHWM within ~10 s) before the next budget is read
done
while [ "$(n_alive)" -gt 0 ]; do sleep 120; done
note "launcher done: $(ls "$OUT"/*/provenance.json 2>/dev/null | wc -l) runs with provenance.json"

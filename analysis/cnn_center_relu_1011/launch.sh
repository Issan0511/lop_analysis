#!/usr/bin/env bash
# cnn_center_relu_1011: launch the registered runs (spec §8).
#   bash analysis/cnn_center_relu_1011/launch.sh <M_keep_MiB>
# M_keep = the other agent's GPU use (measured + planned) + 2048 MiB.  Concurrency
#   n = floor((M_free - M_keep) / (1.25 * M_proc)),  M_proc = K-cost's nvidia-smi peak per process.
# n >= 5: the four main arms and Cr together; otherwise the four main arms (Cr later, by hand).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
OUT="$REPO/results/cnn_center_relu_1011"
M_PROC=1170
M_KEEP=${1:?usage: launch.sh <M_keep_MiB>}
mkdir -p "$OUT/logs"
REC="$OUT/logs/launch_record.txt"
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
N=$(( (FREE - M_KEEP) * 100 / (125 * M_PROC) ))
NPY=$(pgrep -c -f '[p]roj_004_drift/\.venv/bin/python' || true)
{
  echo "$(date '+%F %T') free=${FREE}MiB keep=${M_KEEP}MiB proc=${M_PROC}MiB -> n=${N}; proj_004_drift python now=${NPY}"
  echo "HEAD $(git -C "$REPO" rev-parse HEAD)  dirty=[$(git -C "$REPO" status --porcelain -- src analysis | tr '\n' ' ')]"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
} | tee -a "$REC"
if [ "$N" -lt 4 ]; then
  echo "n=$N < 4: not launching (wait for GPU memory)" | tee -a "$REC"
  exit 1
fi
launch() {  # arm seeds
  nohup setsid "$PY" "$REPO/src/cnn_center_relu_1011.py" run --arm "$1" --seeds "$2" \
    --out "$OUT/$1" > "$OUT/logs/$1.log" 2>&1 < /dev/null &
  echo "launched $1 seeds $2 pid $!" | tee -a "$REC"
}
for arm in ref C H CH; do
  launch "$arm" 10-19
  sleep 20      # let each process reserve its evaluation peak before the next one starts
done
if [ "$N" -ge 5 ]; then
  launch Cr 10
else
  echo "n=$N < 5: Cr deferred" | tee -a "$REC"
fi

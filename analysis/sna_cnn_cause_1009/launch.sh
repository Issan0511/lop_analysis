#!/bin/bash
# usage: launch.sh <bundle-name> <arms> [seeds]
# Relaunches after a failure (e.g. CUDA OOM while ollama holds the GPU); the engine resumes
# from <out>/state.pt at the next task, so a relaunch repeats no completed task.
cd "$(dirname "$0")/../.." || exit 1
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
OUT=results/sna_cnn_cause_1009/$1
SEEDS=${3:-10-19}
mkdir -p "$OUT"
for i in $(seq 1 60); do
  "$PY" src/sna_cnn_cause_1009.py --arms "$2" --seeds "$SEEDS" --out "$OUT" >> "$OUT/log.txt" 2>&1 < /dev/null && exit 0
  echo "[$(date +%T)] exit, retry $i in 120 s" >> "$OUT/log.txt"
  sleep 120
done

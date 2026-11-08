#!/usr/bin/env bash
# pm_rl_samebox_1108 job list runner: run_jobs.sh JOBFILE PAR
# JOBFILE lines: "seq act seed delay".
set -u
WT=/home/issan/Projects/claude/wt/pm_rl_samebox_1108
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
DATA=/home/issan/Projects/obsidian-research-data/pm_rl_samebox_1108
JOBFILE=$1
PAR=${2:-8}
mkdir -p "$DATA/logs" "$DATA/runs"
cd "$WT"
export WT PY DATA
xargs -P "$PAR" -L 1 bash -c '
  q=$0; a=$1; s=$2; d=$3
  sleep "$d"
  out=$DATA/runs/${q}_${a}/seed$s
  mkdir -p "$out"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PY" "$WT/analysis/pm_rl_samebox_1108/run.py" --seq "$q" --act "$a" --seed "$s" --tasks 500 --out "$out" \
    > "$DATA/logs/${q}_${a}_seed${s}.log" 2>&1
' < "$JOBFILE"
echo "queue done: $JOBFILE at $(date)" >> "$DATA/logs/queue.log"

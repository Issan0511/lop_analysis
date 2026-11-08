#!/usr/bin/env bash
# pm_ledger_add_1108 job list runner: run_jobs.sh JOBFILE PAR
# JOBFILE lines: "arm seed delay" (arm uses + as in run.py).
set -u
WT=/home/issan/Projects/claude/wt/pm_ledger_add_1108
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
DATA=/home/issan/Projects/obsidian-research-data/pm_ledger_add_1108
JOBFILE=$1
PAR=${2:-8}
mkdir -p "$DATA/logs" "$DATA/runs"
cd "$WT"
export WT PY DATA
xargs -P "$PAR" -L 1 bash -c '
  a=$0; s=$1; d=$2
  sleep "$d"
  dn=${a//+/_}
  out=$DATA/runs/$dn/seed$s
  mkdir -p "$out"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PY" "$WT/analysis/pm_ledger_add_1108/run.py" --arm "$a" --seed "$s" --tasks 500 --out "$out" \
    > "$DATA/logs/${dn}_seed${s}.log" 2>&1
' < "$JOBFILE"
echo "queue done: $JOBFILE at $(date)" >> "$DATA/logs/queue.log"

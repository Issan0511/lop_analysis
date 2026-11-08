#!/usr/bin/env bash
# pm_ledger_1108 queue: run_queue.sh "SEEDS" PAR  (e.g. run_queue.sh "0 1 2" 6)
# Each job is one process with one CPU thread; 3 s stagger so the MNIST load peaks do not coincide.
set -u
WT=/home/issan/Projects/claude/wt/pm_ledger_1108
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
DATA=/home/issan/Projects/obsidian-research-data/pm_ledger_1108
SEEDS=${1:-"0 1 2"}
PAR=${2:-6}
mkdir -p "$DATA/logs" "$DATA/runs"
cd "$WT"
jobfile=$(mktemp)
k=0
for s in $SEEDS; do
  for a in none l2init; do
    d=$(( k < PAR ? k * 3 : 0 ))
    echo "$a $s $d" >> "$jobfile"
    k=$((k+1))
  done
done
export WT PY DATA
xargs -P "$PAR" -L 1 bash -c '
  a=$0; s=$1; d=$2
  sleep "$d"
  out=$DATA/runs/$a/seed$s
  mkdir -p "$out"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PY" "$WT/analysis/pm_ledger_1108/run.py" --arm "$a" --seed "$s" --tasks 500 --out "$out" \
    > "$DATA/logs/${a}_seed${s}.log" 2>&1
' < "$jobfile"
rm -f "$jobfile"
echo "queue done: seeds [$SEEDS] at $(date)" >> "$DATA/logs/queue.log"

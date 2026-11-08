#!/usr/bin/env bash
# pm_ledger_add_1108 queue: run_queue.sh "ARMS" "SEEDS" PAR   (e.g. run_queue.sh "CHB KKT1" "0 1 2" 6)
# Each job: one process, one CPU thread; 3 s stagger so the MNIST load peaks do not coincide.
set -u
WT=/home/issan/Projects/claude/wt/pm_ledger_add_1108
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
DATA=/home/issan/Projects/obsidian-research-data/pm_ledger_add_1108
ARMS=${1:-"CHB KKT1"}
SEEDS=${2:-"0 1 2"}
PAR=${3:-6}
mkdir -p "$DATA/logs" "$DATA/runs"
cd "$WT"
jobfile=$(mktemp)
k=0
for a in $ARMS; do
  for s in $SEEDS; do
    d=$(( k < PAR ? k * 3 : 0 ))
    echo "$a $s $d" >> "$jobfile"
    k=$((k+1))
  done
done
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
' < "$jobfile"
rm -f "$jobfile"
echo "queue done: arms [$ARMS] seeds [$SEEDS] at $(date)" >> "$DATA/logs/queue.log"

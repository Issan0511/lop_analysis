#!/usr/bin/env bash
# pm_ledger_1108 ledger queue: run_ledger.sh "SEEDS" PAR  (e.g. run_ledger.sh "0 1 2" 3)
# One process per seed, one BLAS thread each, 3 s stagger so the MNIST load peaks do not coincide.
set -u
WT=/home/issan/Projects/claude/wt/pm_ledger_1108
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
DATA=/home/issan/Projects/obsidian-research-data/pm_ledger_1108
SEEDS=${1:-"0 1 2"}
PAR=${2:-3}
mkdir -p "$DATA/logs" "$DATA/ledger"
cd "$WT"
export WT PY DATA
jobfile=$(mktemp)
k=0
for s in $SEEDS; do
  d=$(( k < PAR ? k * 3 : 0 ))
  echo "$s $d" >> "$jobfile"
  k=$((k+1))
done
xargs -P "$PAR" -L 1 bash -c '
  s=$0; d=$1
  sleep "$d"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PY" "$WT/analysis/pm_ledger_1108/ledger.py" --seed "$s" --runs "$DATA/runs" --out "$DATA/ledger" \
    > "$DATA/logs/ledger_seed${s}.log" 2>&1
' < "$jobfile"
rm -f "$jobfile"
echo "ledger done: seeds [$SEEDS] at $(date)" >> "$DATA/logs/queue.log"

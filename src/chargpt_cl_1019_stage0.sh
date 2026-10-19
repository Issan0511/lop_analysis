#!/usr/bin/env bash
# chargpt_cl_1019 段 0（spec §7）: C3 だけ、seed 100〜102、5 課題。
#   箱 M: alpha {0.01,0.03,0.1} x N_M {1000,3000,10000}
#   箱 W: N_W {3000,7500,30000}
# 使い方: bash src/chargpt_cl_1019_stage0.sh [並列数]
set -u
cd "$(dirname "$0")/.."
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
OUT=results/chargpt_cl_1019/stage0
PAR=${1:-2}
mkdir -p "$OUT/logs"
jobs_file="$OUT/jobs.txt"
: > "$jobs_file"
# 長い走から先に積む
for n in 30000 7500 3000; do
  for s in 100 101 102; do
    echo "W n$n seed$s --box W --n_per_task $n --seed $s" >> "$jobs_file"
  done
done
for n in 10000 3000 1000; do
  for a in 0.01 0.03 0.1; do
    for s in 100 101 102; do
      echo "M a${a}_n$n seed$s --box M --alpha $a --n_per_task $n --seed $s" >> "$jobs_file"
    done
  done
done
run_one() {
  box=$1; cfg=$2; sd=$3; shift 3
  d="$OUT/$box/$cfg/$sd"
  if [ -f "$d/provenance.json" ]; then echo "skip $d"; return 0; fi
  $PY src/chargpt_cl_1019.py run --arm C3 --tasks 5 --ckpt 0 --out "$d" "$@" > "$OUT/logs/${box}_${cfg}_${sd}.log" 2>&1
  echo "done $d rc=$?"
}
export -f run_one
export PY OUT
xargs -P "$PAR" -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
echo STAGE0_FINISHED > "$OUT/FINISHED"

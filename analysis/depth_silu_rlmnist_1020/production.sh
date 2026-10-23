#!/bin/bash
# depth_silu_rlmnist_1020 本走: 6 腕 x seed 0-9 x 50 課題、同時 12 本、1 走 1 スレッド、CUDA は見せない。
# 既に npz がある走は飛ばす（再開できる）。seed 0-2 を先に並べる。
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
OUT=raw/depth_silu_rlmnist_1020/runs
mkdir -p "$OUT" raw/logs
jobs=()
for s in 0 1 2 3 4 5 6 7 8 9; do
  for arm in "SILU 1 none" "SILU 2 none" "SILU 4 none" "R 1 none" "R 4 none" "SILU 4 C"; do
    set -- $arm
    n="${1}_L${2}_${3}_s${s}"
    [ -f "$OUT/$n.npz" ] || jobs+=("$1 $2 $3 $s $n")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 12 -L 1 bash -c '"$0" src/depth_silu_rlmnist_1020.py --act $1 --layers $2 --door $3 --seed $4 --tasks 50 --out '"$OUT"' --name $5 > raw/logs/run_$5.log 2>&1 < /dev/null' "$PY"
echo "production done $(date '+%F %T')" > raw/logs/production.done

#!/bin/bash
# spec §2.1 の入力: A2（C3-2）と ReLUcap-2 の seed 0〜9 の記録を、ブランチ claude/chargpt_cl_1019 から取り出す（checkout しない）。
# 使い方: bash extract_a2.sh <出力フォルダ>
set -eu
OUT=$1; REPO=$(cd "$(dirname "$0")/../../.." && pwd)
for arm in C3 ReLUcap; do for s in 0 1 2 3 4 5 6 7 8 9; do
  d=results/chargpt_cl_1019/M2/$arm/seed$s; mkdir -p $OUT/$arm/seed$s
  for f in per_task.csv online.npz fresh_online_t10.npz fresh_online_t100.npz fresh.csv unit_stats.npz layer_stats.csv provenance.json; do
    git -C $REPO show origin/claude/chargpt_cl_1019:$d/$f > $OUT/$arm/seed$s/$f
  done
done; done

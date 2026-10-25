#!/usr/bin/env bash
# 追補 C の本走: N = 250・1,000・2,500 × 6 状態を 1 件ずつ（spec 追補 C）
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_t100.json CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
R=$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_t100/capacity
mkdir -p $R
for n in 250 1000 2500; do
  for state in S R0 SL R0L fresh freshL; do
    out=$R/${state}_N$n
    echo "$(date '+%H:%M:%S') start $state N=$n"
    $PY -m analysis.vit_long_1024.capacity --state $state --n $n --out $out > $R/${state}_N$n.log 2>&1
    echo "$(date '+%H:%M:%S') end $state N=$n exit $? $(tail -1 $R/${state}_N$n.log)"
  done
done
echo "$(date '+%H:%M:%S') ALL_DONE"

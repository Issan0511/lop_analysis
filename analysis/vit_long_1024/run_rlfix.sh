#!/usr/bin/env bash
# 追補 D の本走（spec 追補 D）: 4 腕を並行で、新品 8 本を別の列で順に。seed 0。
# 走っている間は src・analysis・specs に触らない（後から起動する新品が汚れの判定で落ちる）。
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
R=$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix
COMMON="--seed 0 --steps 5000 --n-labels 10 --images-per-class 250 --augment off --dropout 0.0 --depth 2 --mlp-ratio 8.0"
mkdir -p $R/runs
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $R/run_rlfix.log; }
for arm in R0 S R0L SL; do
  (
    out=$R/runs/$arm/seed0; mkdir -p $out
    log "start $arm"
    $PY -m analysis.vit_long_1024.run --arm $arm --tasks 100 $COMMON --out $out >> $out/console.log 2>&1
    rc=$?; log "end $arm rc=$rc"
  ) &
done
(
  for arm in freshL fresh; do
    for t in 25 50 75 100; do
      out=$R/runs/$arm/seed0/task$t; mkdir -p $out
      log "start $arm task $t"
      $PY -m analysis.vit_long_1024.run --arm $arm --fresh-task $t $COMMON --out $out >> $out/console.log 2>&1
      rc=$?; log "end $arm task $t rc=$rc"
    done
  done
) &
wait
log ALL_DONE

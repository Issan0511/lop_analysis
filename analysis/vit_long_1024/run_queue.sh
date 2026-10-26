#!/usr/bin/env bash
# 追補 E の本走: 仕事の一覧（jobs_rlfix_e_*.txt）を 2 本ずつ並べて回す（奇数行と偶数行を別の列で）。
# 使い方: bash analysis/vit_long_1024/run_queue.sh analysis/vit_long_1024/jobs_rlfix_e_whitesan.txt
# 止まったら同じ命令をもう一度流す（終わった走は飛ばし、途中の走は課題の境から再開する）。
# PY（python）・VIT_LONG_RAW（出力の置き場）・VIT_LONG_DATA（Tiny ImageNet）は環境変数で変えられる。
set -uo pipefail
JOBS=$(readlink -f "$1")
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix_e.json CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
R=${VIT_LONG_RAW:-$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix}
export VIT_LONG_RAW=$R
COMMON="--steps 5000 --n-labels 10 --images-per-class 250 --augment off --dropout 0.0 --depth 2 --mlp-ratio 8.0"
mkdir -p $R/runs
LOG=$R/run_queue_$(basename "$JOBS" .txt).log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
worker() {   # worker <0|1>: 一覧の偶数番目（0）か奇数番目（1）の仕事を順に
  grep -v '^#' "$JOBS" | grep -v '^\s*$' | awk -v w=$1 'NR % 2 == (w + 1) % 2' | while read -r arm seed task; do
    if [ -n "${task:-}" ]; then
      out=$R/runs/$arm/seed$seed/task$task; extra="--fresh-task $task"
    else
      out=$R/runs/$arm/seed$seed; extra="--tasks 100"
    fi
    mkdir -p "$out"
    if [ -f "$out/done.json" ]; then log "skip $arm $seed ${task:-} (done)"; continue; fi
    log "start $arm $seed ${task:-}"
    $PY -m analysis.vit_long_1024.run --arm $arm --seed $seed $extra $COMMON --out "$out" >> "$out/console.log" 2>&1 < /dev/null
    rc=$?
    log "end $arm $seed ${task:-} rc=$rc"
  done
}
log "queue start $(basename "$JOBS") commit $(git rev-parse --short HEAD 2>/dev/null || cat SOURCE_COMMIT)"
worker 0 &
worker 1 &
wait
log ALL_DONE

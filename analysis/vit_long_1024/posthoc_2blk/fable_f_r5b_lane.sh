#!/usr/bin/env bash
# R5 の続き（1104）: SC の t30・t60 で対照と head 0（各 5 seed）、S の t30・t60 で体を凍結した head 0（5 seed）。1 本ずつ。
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix_f.json CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
R=${VIT_LONG_RAW:-$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix}
LOG=$R/posthoc_f/r5b_lane.log
run() {  # tag, args...
  local tag=$1; shift
  [ -f "$R/posthoc_f/$tag.json" ] && { echo "skip $tag" >> "$LOG"; return; }
  echo "== $(date +%T) $tag" >> "$LOG"
  nice -n 10 "$PY" -m analysis.vit_long_1024.posthoc_2blk.fable_f_head0 "$@" >> "$LOG" 2>&1 || echo "FAILED $tag" >> "$LOG"
}
for s in 10252010 10252011 10252012 10252013 10252014; do
  for t in 30 60; do
    run "r5_SC_seed${s}_t${t}_base" "$s" "$t" --arm SC --keep-head
    run "r5_SC_seed${s}_t${t}_head0" "$s" "$t" --arm SC
    run "r5_S_seed${s}_t${t}_head0_frozen" "$s" "$t" --freeze-body
  done
done
echo "R5B_ALL_DONE $(date +%T)" >> "$LOG"

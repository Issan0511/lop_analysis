#!/usr/bin/env bash
# 追補 E.3 の記録（RMSNorm の装填の押し）と G.4 の記録（cos²）を 4 腕 × 5 seed で順に回す（Fable、1111。CPU float64、1 本ずつ）。
# 1111 03:48〜04:22 に white-san で回した。出力は <raw>/posthoc_j5/j5_<腕>_seed<seed>.json（既にあれば飛ばす）と eg_j5.log。
cd "$(dirname "$0")/../../.."
RAW=$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix
LOG=$RAW/posthoc_j5/eg_j5.log
PY=$HOME/Projects/claude/proj_004_drift/.venv/bin/python
for spec in "SRMS production_2blk_rlfix_e.json" "SL production_2blk_rlfix_e.json" "SLW production_2blk_rlfix_g.json" "SLP production_2blk_rlfix_g.json"; do
  set -- $spec
  echo "$(date '+%m-%d %H:%M:%S') start $1" >> "$LOG"
  PYTHONPATH=$PWD VIT_LONG_PRODUCTION=$2 nice -n 10 "$PY" -m analysis.vit_long_1024.posthoc_2blk.fable_eg_j5 "$1" 10252010 10252011 10252012 10252013 10252014 >> "$LOG" 2>&1
  echo "$(date '+%m-%d %H:%M:%S') end $1 rc=$?" >> "$LOG"
done
echo "$(date '+%m-%d %H:%M:%S') ALL_DONE" >> "$LOG"

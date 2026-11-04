#!/usr/bin/env bash
# R5d（--reset-adam）・R5e（--center-head-input）の列（1104 事後）: S の t60 末、head を 0 に戻して 5 seed を 1 本ずつ回す。
# 使い方: systemd-run --user --unit=fable_f_r5de_<mode>_1104 --collect -p Nice=10 /usr/bin/bash $PWD/analysis/vit_long_1024/posthoc_2blk/fable_f_r5de_lane.sh <adamreset|hcenter> [task]
set -uo pipefail
mode=${1:?adamreset|hcenter}; task=${2:-60}
case $mode in adamreset) flag=--reset-adam;; hcenter) flag=--center-head-input;; *) echo "mode?" >&2; exit 2;; esac
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix_f.json CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
R=${VIT_LONG_RAW:-$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix}
OUT=$R/posthoc_f; LOG=$OUT/r5de_${mode}_lane.log
for seed in 10252010 10252011 10252012 10252013 10252014; do
  tag=r5_S_seed${seed}_t${task}_head0_${mode}
  [ -f "$OUT/$tag.json" ] && { echo "$(date +%T) skip $tag" >> "$LOG"; continue; }
  echo "$(date +%T) start $tag" >> "$LOG"
  nice -n 10 $PY -m analysis.vit_long_1024.posthoc_2blk.fable_f_head0 $seed $task $flag >> "$OUT/$tag.log" 2>&1 </dev/null
  rc=$?; echo "$(date +%T) end $tag rc=$rc" >> "$LOG"; [ $rc -ne 0 ] && echo "$(date +%T) FAILED $tag" >> "$LOG"
done
echo "$(date +%T) R5DE_${mode}_ALL_DONE" >> "$LOG"

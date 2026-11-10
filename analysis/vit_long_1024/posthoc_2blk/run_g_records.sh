#!/usr/bin/env bash
# 追補 G.4 の記録「体の速さ」: 保存した課題 100 末から head を 0 に戻して課題 101 を回した T90（Fable、1111）。
# scale_counterfactual.py の head0（eager、Adam は保存したもの、≤ 5,000 更新、0.9 で止める）。腕の hold（WD・射影）は掛けない。
# SL・SLW・SLP × 5 seed、すべて white-san。出力は <raw>/posthoc_g/body_speed_<腕>_seed<seed>.json（既にあれば飛ばす）。
set -u
cd "$(dirname "$0")/../../.."
RAW=$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix
OUT=$RAW/posthoc_g
mkdir -p "$OUT"
PY=$HOME/Projects/claude/proj_004_drift/.venv/bin/python
LOG=$OUT/run_g_records.log
for arm in SL SLW SLP; do
  if [ "$arm" = SL ]; then cfg=production_2blk_rlfix_e.json; else cfg=production_2blk_rlfix_g.json; fi
  for seed in 10252010 10252011 10252012 10252013 10252014; do
    o=$OUT/body_speed_${arm}_seed${seed}.json
    if [ -f "$o" ]; then echo "$(date '+%m-%d %H:%M:%S') skip $arm $seed" >> "$LOG"; continue; fi
    echo "$(date '+%m-%d %H:%M:%S') start $arm $seed" >> "$LOG"
    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=$cfg nice -n 5 "$PY" analysis/vit_long_1024/posthoc_2blk/scale_counterfactual.py \
      "$arm" 100 head0 "$o.tmp" --max-steps 5000 --seed "$seed" >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] && mv "$o.tmp" "$o"
    echo "$(date '+%m-%d %H:%M:%S') end $arm $seed rc=$rc" >> "$LOG"
  done
done
echo "$(date '+%m-%d %H:%M:%S') ALL_DONE" >> "$LOG"

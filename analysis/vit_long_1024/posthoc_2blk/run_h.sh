#!/usr/bin/env bash
# 追補 H.3 の 160 走 + 記録 5 走（1108 修正版）を GPU で 2 本並列に回す driver。起動は親が判断する（追補 E の列が終わってから）。
#   順: B60（--init ckpt, t_end 60, 平均置換）× 8 切除 × 5 seed → F61（--init t00, t_end 60, 零）→ B30（ckpt, 30, 平均置換）
#       → F31（t00, 30, 零）→ 記録 B60z（ckpt, 60, block1 を零の切除で）× 5 seed。
#   群の中の切除の順: none, block1, ffn1, attn1, ffn0, attn0, ffn0ffn1, all（各切除で 5 seed）。
#   出力 ~/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_h/{B60,F61,B30,F31,B60z}_{cut}_seed{seed}.json
#   status == "ok" の JSON があれば飛ばす。ログ posthoc_h/run_h.log（START / END / SKIP / FAIL / ALL_DONE）。
#   起動例: setsid nohup bash analysis/vit_long_1024/posthoc_2blk/run_h.sh > /dev/null 2>&1 &
set -u
WT=/home/issan/Projects/claude/wt/vit_long_1024
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
OUT=${H_OUT:-/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_h}
LOG=$OUT/run_h.log
PAR=${H_PARALLEL:-2}
SELF=$(readlink -f "$0")
mkdir -p "$OUT"

mode_of() {
  case "$1" in
    none) echo none ;;
    ffn1) echo cut_ffn1 ;;
    attn1) echo cut_attn1 ;;
    ffn0) echo cut_ffn0 ;;
    attn0) echo cut_attn0 ;;
    ffn0ffn1) echo cut_ffn0+cut_ffn1 ;;
    all) echo cut_attn0+cut_ffn0+cut_attn1+cut_ffn1 ;;
    block1) echo cut_attn1+cut_ffn1 ;;
    *) echo "unknown cut $1" >&2; return 1 ;;
  esac
}

if [ "${1:-}" = "--job" ]; then
  grp=$2; cut=$3; seed=$4
  case "$grp" in
    B60) init=ckpt; tend=60; kind=mean ;; F61) init=t00; tend=60; kind=zero ;;
    B30) init=ckpt; tend=30; kind=mean ;; F31) init=t00; tend=30; kind=zero ;;
    B60z) init=ckpt; tend=60; kind=zero ;;
    *) echo "unknown group $grp" >&2; exit 1 ;;
  esac
  dest=$OUT/${grp}_${cut}_seed${seed}.json
  if [ -f "$dest" ] && "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$dest')).get('status')=='ok' else 1)" 2>/dev/null; then
    echo "$(date '+%F %T') SKIP $grp $cut $seed" >> "$LOG"; exit 0
  fi
  mode=$(mode_of "$cut") || { echo "$(date '+%F %T') FAIL bad-cut $grp $cut $seed" >> "$LOG"; exit 0; }
  echo "$(date '+%F %T') START $grp $cut $seed" >> "$LOG"
  cd "$WT" && PYTHONPATH=. VIT_LONG_PRODUCTION=production_2blk_rlfix.json "$PY" \
    analysis/vit_long_1024/posthoc_2blk/cut_replay.py S "$tend" "$mode" "$dest" \
    --seed "$seed" --init "$init" --cut-kind "$kind" --h --max-steps 5000 --track-steps 100 --device cuda \
    > "$OUT/${grp}_${cut}_seed${seed}.out" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "$(date '+%F %T') END $grp $cut $seed $(tail -n 1 "$OUT/${grp}_${cut}_seed${seed}.out" | cut -c1-200)" >> "$LOG"
  else
    echo "$(date '+%F %T') FAIL rc=$rc $grp $cut $seed" >> "$LOG"
  fi
  exit 0
fi

CUTS="none block1 ffn1 attn1 ffn0 attn0 ffn0ffn1 all"
SEEDS="10252010 10252011 10252012 10252013 10252014"
jobs() {
  for grp in B60 F61 B30 F31; do
    for cut in $CUTS; do for seed in $SEEDS; do echo "$grp $cut $seed"; done; done
  done
  for seed in $SEEDS; do echo "B60z block1 $seed"; done
}
echo "$(date '+%F %T') QUEUE_START pid=$$ parallel=$PAR" >> "$LOG"
jobs | xargs -P "$PAR" -L 1 bash "$SELF" --job
echo "$(date '+%F %T') ALL_DONE" >> "$LOG"

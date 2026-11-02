#!/usr/bin/env bash
# 停滞を分ける回し直し（1102、Issa「停滞を分ける 3 本」。posthoc_scale_counterfactual.md §9）:
#   SL seed 10252010 の課題 100 末から課題 101 を、更新 25 のあとに Adam の記憶を消す／β₂ を 0.99 にする／
#   更新 25 のあとに head を 0 にして Adam も消す、で回す。どの腕も最初の 500 更新の 1 更新ごとの変位・勾配・Adam の v と
#   m̂/√v̂ を記録する。そのままと head 0 も同じ記録つきで回し直す（T90 が 1,125・575 を再現するかの検査を兼ねる）。
# 使い方: nohup nice -n 10 bash analysis/vit_long_1024/posthoc_2blk/plateau_adam_1102.sh > /dev/null 2>&1 &
#   出力は $R/posthoc_scale/scale_SL_T100_<名前>.json（既にあれば飛ばす）。log は $R/plateau_adam_1102.log。
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
R=${VIT_LONG_RAW:-$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix}
OUT=$R/posthoc_scale
LOG=$R/plateau_adam_1102.log
TRACK=${TRACK:-500}
mkdir -p "$OUT"
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
run() {   # run <mode> <名前> [追加の引数]
  local mode=$1 name=$2; shift 2
  local out=$OUT/scale_SL_T100_$name.json
  if [ -f "$out" ]; then log "skip $name (exists)"; return; fi
  log "start $name mode=$mode commit $(git rev-parse --short HEAD 2>/dev/null)"
  $PY analysis/vit_long_1024/posthoc_2blk/scale_counterfactual.py SL 100 "$mode" "$out" --track-steps "$TRACK" "$@" >> "$R/plateau_adam_1102.console.log" 2>&1 < /dev/null
  log "end $name rc=$?"
}
log "queue start commit $(git rev-parse --short HEAD 2>/dev/null) track $TRACK"
run base                 base_track           # そのまま（1,125 の再現）
run head0                head0_track          # head 0（575 の再現）
run adam0at25            adam0at25            # (a) 更新 25 のあとに Adam を全部消す
run bodyadam0at25        bodyadam0at25        # (a) 体だけ消す
run beta2_0.99           beta2_0.99           # (a) v の時定数 1,000 → 100
run head0+beta2_0.99     head0+beta2_0.99     # beta2_0.99 の床
run head0at25+adam0at25  head0at25+adam0at25  # (c) 押しで 25 更新動いた体に新しい head と新しい Adam
log ALL_DONE

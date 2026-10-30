#!/usr/bin/env bash
# 比の説明と自信を分ける回し直し（1031、Issa「F の後に足す」）:
#   SL seed 10252010 の課題 100 末（重み＋Adam）から、head の学習率だけ ×2・×4・×10 にして課題 101 を回し、T90 を測る。
#   表 2 の head ×0.5・×0.25・×0.1 と「重みに対する歩幅」が同じ組。T90 が 950・675・575 に並べば比の説明で足り、
#   1,125 の近くに留まれば softmax が尖ること（関数が変わること）の分が別にある。
# 使い方: setsid nohup bash analysis/vit_long_1024/posthoc_2blk/headlr_after_f.sh <待つ pid> [<pid> ...] &
#   渡した pid（追補 F の列の本体 run_queue.sh と run_queue_after.sh）が全部終わり、かつ analysis.vit_long_1024.run の走が
#   1 本も残っていなくなってから、1 本ずつ回す（列が別の pid で再起動されても、走が残っている間は待つ）。
#   始める前に GPU の空きが NEED_MIB 以上になるまで待つ（run_queue_after.sh と同じ既定 1540）。出力が既にあれば飛ばす。
set -uo pipefail
[ $# -ge 1 ] || { echo "使い方: $0 <待つ pid> [<pid> ...]" >&2; exit 2; }
for p in "$@"; do [[ "$p" =~ ^[0-9]+$ ]] || { echo "pid でない: $p" >&2; exit 2; }; done
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
R=${VIT_LONG_RAW:-$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix}
OUT=$R/posthoc_scale
LOG=$R/headlr_after_f.log
NEED_MIB=${NEED_MIB:-1540}
MODES=${MODES:-"headlr2 headlr4 headlr10"}
mkdir -p "$OUT"
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
alive() { [ -e "/proc/$1" ] && ! grep -q '^State:[[:space:]]*Z' "/proc/$1/status" 2>/dev/null; }
runs_active() {   # analysis.vit_long_1024.run の python が 1 本でも生きていれば真（/proc を読むだけで、何も止めない）
  local p
  for p in /proc/[0-9]*; do
    { tr '\0' ' ' < "$p/cmdline"; } 2>/dev/null | grep -q 'analysis\.vit_long_1024\.run ' && return 0   # 走の子 process も同じ cmdline で見えるが、走と一緒に消えるので差し支えない
  done
  return 1
}

log "wait start pids $* modes $MODES commit $(git rev-parse --short HEAD 2>/dev/null)${DRY_RUN:+ DRY_RUN=$DRY_RUN}"
while :; do
  busy=0
  for p in "$@"; do alive "$p" && busy=1; done
  runs_active && busy=1
  [ $busy -eq 0 ] && break
  sleep 120
done
log "F ended (pids $* gone, no run left)"
for mode in $MODES; do
  out=$OUT/scale_SL_T100_$mode.json
  if [ -f "$out" ]; then log "skip $mode (exists)"; continue; fi
  waited=0
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$free" =~ ^[0-9]+$ ]] && [ "$free" -ge "$NEED_MIB" ]; then break; fi
    [ $waited -eq 0 ] && log "wait gpu free=${free:-?} MiB < $NEED_MIB ($mode)"
    waited=1; sleep 60
  done
  log "start $mode commit $(git rev-parse --short HEAD 2>/dev/null)"
  if [ "${DRY_RUN:-0}" = 1 ]; then log "DRY $PY analysis/vit_long_1024/posthoc_2blk/scale_counterfactual.py SL 100 $mode $out"; continue; fi
  $PY analysis/vit_long_1024/posthoc_2blk/scale_counterfactual.py SL 100 "$mode" "$out" >> "$R/headlr_after_f.console.log" 2>&1 < /dev/null
  log "end $mode rc=$?"
done
log ALL_DONE

#!/usr/bin/env bash
# 追補 F の回し直し R1〜R3（spec F.4）を 1 本ずつ回す（Fable、1104）。R1 を全 seed（t10・t30 × base・fix_ffn0）、次に R2・R3
# （t30・t60 × base・center・scale）。出力 <raw>/posthoc_f/<tag>.json が既にあれば飛ばす。1 本の log は <tag>.log、列の log は queue_r123.log。
# 使い方: systemd-run --user --unit=fable_f_queue_<MMDD> --collect -p Nice=10 --working-directory=<repo> /usr/bin/bash <repo>/analysis/vit_long_1024/posthoc_2blk/fable_f_queue.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix_f.json CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
R=${VIT_LONG_RAW:-$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix}
OUT=$R/posthoc_f
LOG=$OUT/queue_r123.log
SEEDS=${SEEDS:-"10252010 10252011 10252012 10252013 10252014"}
KINDS=${KINDS:-"r1 r2r3"}
mkdir -p "$OUT"
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
run() {   # kind seed task mode
  local tag="$1_S_seed$2_t$3_$4"
  if [ -f "$OUT/$tag.json" ]; then log "skip $tag (exists)"; return; fi
  log "start $tag"
  nice -n 10 "$PY" -m analysis.vit_long_1024.posthoc_2blk.fable_f_replay "$1" "$2" "$3" "$4" >> "$OUT/$tag.log" 2>&1
  log "end $tag rc=$?"
}
log "queue start commit $(git rev-parse --short HEAD 2>/dev/null) seeds [$SEEDS] kinds [$KINDS]"
for k in $KINDS; do
  if [ "$k" = r1 ]; then
    for s in $SEEDS; do for t in 10 30; do for m in base fix_ffn0; do run r1 "$s" "$t" "$m"; done; done; done
  else
    for s in $SEEDS; do for t in 30 60; do for m in base center scale; do run r2r3 "$s" "$t" "$m"; done; done; done
  fi
done
log ALL_DONE

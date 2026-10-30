#!/usr/bin/env bash
# 走行中の列の後ろに仕事を足す（追補 F.12、Issa 1030「white-san で回す」）。
# 使い方: bash analysis/vit_long_1024/run_queue_after.sh <仕事の一覧> <pid> [<pid> ...]
#   pid 1 つが枠 1 つ。各枠は、その pid（走行中の run_queue.sh の worker）が終わるのを待ってから、
#   一覧の仕事を先頭から 1 本ずつ取って回す。枠どうしは同じ一覧を分け合うので、先に空いた枠が多く取る。
#   枠の数は pid の数で、それを超えて同時に回さない。終わっている pid を渡すと、その枠はすぐに始まる。
# 1 本の仕事の中身（環境変数・引数・出力の置き場・done.json があれば飛ばす）は run_queue.sh と同じ。
# 止まったら同じ命令をもう一度流す（終わった走は飛ばし、途中の走は課題の境から再開する。pid には終わったものを渡せばよい）。
# 始める前に GPU の空きが NEED_MIB 以上になるまで待つ（待つだけで、何も止めない）。既定 1540 = 1 本の走の実測 1284 MiB（white-san、1030）＋ 256。
# 始めるのは 1 本ずつで、始めてから RAMP_S 秒（既定 120）は次を始めない（使う量が上がりきる前に二つが同時に空きを見て入らないように）。
# DRY_RUN=1 なら走を起動せず、起動する命令を log に書くだけ。
set -uo pipefail
[ $# -ge 2 ] || { echo "使い方: $0 <仕事の一覧> <pid> [<pid> ...]" >&2; exit 2; }
JOBS=$(readlink -f "$1"); shift
for p in "$@"; do [[ "$p" =~ ^[0-9]+$ ]] || { echo "pid でない: $p" >&2; exit 2; }; done
cd "$(dirname "$0")/../.."
PROD=production_2blk_$(basename "$JOBS" .txt | sed -E 's/^jobs_(rlfix_[a-z]+)_.*$/\1/').json
[ -f "analysis/vit_long_1024/$PROD" ] || { echo "一覧の名前から設定が決まらない: $(basename "$JOBS") → $PROD" >&2; exit 1; }
export PYTHONPATH=$PWD VIT_LONG_PRODUCTION=$PROD CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
R=${VIT_LONG_RAW:-$HOME/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix}
export VIT_LONG_RAW=$R
COMMON="--steps 5000 --n-labels 10 --images-per-class 250 --augment off --dropout 0.0 --depth 2 --mlp-ratio 8.0"
NEED_MIB=${NEED_MIB:-1540}
RAMP_S=${RAMP_S:-120}
mkdir -p $R/runs
NAME=run_queue_$(basename "$JOBS" .txt)
LOG=$R/$NAME.log
LIST=$R/$NAME.list    # 起動時の一覧の写し（後で一覧を直しても、走行中のこの列には効かない）
NEXT=$R/$NAME.next    # 次に取る行（0 から）
TAKE_LOCK=$R/$NAME.take.lock
START_LOCK=$R/$NAME.start.lock
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
grep -v '^#' "$JOBS" | grep -v '^\s*$' > "$LIST"
NJOBS=$(wc -l < "$LIST")
echo 0 > "$NEXT"

take() {   # 次の仕事の 1 行を排他的に取る（無ければ何も出さない）
  (
    flock 9
    n=$(cat "$NEXT")
    if [ "$n" -lt "$NJOBS" ]; then
      sed -n "$((n + 1))p" "$LIST"
      echo $((n + 1)) > "$NEXT"
    fi
  ) 9>"$TAKE_LOCK"
}

run_job() {   # run_job <arm> <seed> [<新品の課題>]
  local arm=$1 seed=$2 task=${3:-} out extra free waited=0 jp rc
  if [ -n "$task" ]; then
    out=$R/runs/$arm/seed$seed/task$task; extra="--fresh-task $task"
  else
    out=$R/runs/$arm/seed$seed; extra="--tasks 100"
  fi
  mkdir -p "$out"
  if [ -f "$out/done.json" ]; then log "skip $arm $seed $task (done)"; return; fi
  exec 8>"$START_LOCK"; flock 8
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$free" =~ ^[0-9]+$ ]] && [ "$free" -ge "$NEED_MIB" ]; then break; fi
    [ $waited -eq 0 ] && log "wait gpu free=${free:-?} MiB < $NEED_MIB ($arm $seed $task)"
    waited=1; sleep 60
  done
  [ $waited -eq 1 ] && log "gpu free=$free MiB ($arm $seed $task)"
  log "start $arm $seed $task commit $(git rev-parse --short HEAD 2>/dev/null || cat SOURCE_COMMIT)"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    log "DRY $PY -m analysis.vit_long_1024.run --arm $arm --seed $seed $extra $COMMON --out $out"
    flock -u 8; exec 8>&-
    return
  fi
  $PY -m analysis.vit_long_1024.run --arm $arm --seed $seed $extra $COMMON --out "$out" >> "$out/console.log" 2>&1 < /dev/null 8>&- &
  jp=$!
  sleep "$RAMP_S"
  flock -u 8; exec 8>&-
  wait $jp; rc=$?
  log "end $arm $seed $task rc=$rc"
}

alive() {   # pid が生きていれば真（他人の process でも見える /proc で見る。zombie は終わった扱い）
  [ -e "/proc/$1" ] && ! grep -q '^State:[[:space:]]*Z' "/proc/$1/status" 2>/dev/null
}

slot() {   # slot <待つ pid>
  while alive "$1"; do sleep 30; done
  log "slot open (pid $1 ended)"
  local line arm seed task
  while :; do
    line=$(take)
    [ -z "$line" ] && break
    read -r arm seed task <<< "$line"
    run_job "$arm" "$seed" "${task:-}"
  done
  log "slot done (after pid $1)"
}

log "queue start $(basename "$JOBS") $PROD jobs $NJOBS after pids $* commit $(git rev-parse --short HEAD 2>/dev/null || cat SOURCE_COMMIT)${DRY_RUN:+ DRY_RUN=$DRY_RUN}"
for p in "$@"; do slot "$p" & done
wait
log ALL_DONE

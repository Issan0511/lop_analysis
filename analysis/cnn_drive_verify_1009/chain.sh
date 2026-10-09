#!/usr/bin/env bash
# Sequential GPU jobs of cnn_drive_verify_1009 (one process at a time on the shared GPU).
# Waits for the main `measure` process (PID $1) to exit, then runs each stage in order.
set -u
cd /home/issan/Projects/claude/wt/cnn_drive_verify_1009
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
SRC=src/cnn_drive_verify_1009.py
R=results/cnn_drive_verify_1009
L=$R/logs
ARMS=SNA,SNAc3,CV06FC3,CV3FC06
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 20; done; }
stage() { local name=$1; shift; echo "[$(date +%T)] start $name" >> $L/chain.log; "$@"; echo "[$(date +%T)] end $name rc=$?" >> $L/chain.log; }
# replay task t with the run's own 40-run bundle; if that does not fit the memory cap, two 20-run bundles
replay_task() {
  local t=$1 f
  f=$R/replay/replay_t$(printf %02d "$t").npy
  stage replay$t $PY $SRC replay --task "$t" --arms $ARMS --seeds 10-19 --out $R/replay > $L/replay_t$t.log 2>&1
  if [ ! -f "$f" ]; then
    stage replay${t}a $PY $SRC replay --task "$t" --arms SNA,SNAc3 --seeds 10-19 --tag _a --out $R/replay > $L/replay_t${t}a.log 2>&1
    stage replay${t}b $PY $SRC replay --task "$t" --arms CV06FC3,CV3FC06 --seeds 10-19 --tag _b --out $R/replay > $L/replay_t${t}b.log 2>&1
  fi
}

[ -n "${1:-}" ] && wait_pid "$1"
stage checks3  $PY $SRC checks3 --out $R/checks                                  > $L/checks3.log 2>&1
stage zbar     $PY $SRC zbar --arms $ARMS --seeds 10-19 --tasks 0,1,2,3,5,10,20,30 --out $R/zbar > $L/zbar.log 2>&1
for t in 1 5 10 20; do replay_task $t; done
stage measure30 $PY $SRC measure --arms $ARMS --seeds 10-19 --tasks 30 --K 65536 --out $R/measure >> $L/measure.log 2>&1
stage frozen   $PY $SRC frozen --arms $ARMS --seeds 10-19 --tasks 1,10,20 --L 32 --steps 3000 --out $R/frozen > $L/frozen.log 2>&1
stage extra    $PY $SRC extra --arms $ARMS --seeds 10-19 --tasks 1,10,20 --K 16384 --out $R/extra > $L/extra.log 2>&1
replay_task 30
stage c2exact  $PY $SRC c2exact --arms $ARMS --seeds 10-14 --tasks 1,10,20 --K 4096 --out $R/c2exact > $L/c2exact.log 2>&1
stage capacity $PY $SRC capacity --arms $ARMS --seeds 10,11 --tasks 1,20 --nsub 32 --out $R/capacity > $L/capacity.log 2>&1
echo "[$(date +%T)] chain done" >> $L/chain.log

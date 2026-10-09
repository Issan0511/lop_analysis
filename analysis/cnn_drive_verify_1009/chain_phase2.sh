#!/usr/bin/env bash
# Phase 2 (leaky ReLU / ReLU) measurements, one GPU process at a time, after the LR training run.
# $1 = PID of the LR training process to wait for.
set -u
cd /home/issan/Projects/claude/wt/cnn_drive_verify_1009
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
SRC=src/cnn_drive_verify_1009.py
R=results/cnn_drive_verify_1009
L=$R/logs
wait_pid() { while kill -0 "$1" 2>/dev/null; do sleep 20; done; }
stage() { local name=$1; shift; echo "[$(date +%T)] start $name" >> $L/phase2_chain.log; "$@"; echo "[$(date +%T)] end $name rc=$?" >> $L/phase2_chain.log; }

[ -n "${1:-}" ] && wait_pid "$1"
# checks on LR states (K1 / K8 with the pool winners and the gates held fixed)
stage checksLR   $PY $SRC checks  --states LR:10:1,LR:11:20 --out $R/LR/checks > $L/LR_checks.log 2>&1
stage checks2LR  $PY $SRC checks2 --states LR:10:1,LR:12:10,LR:11:20 --out $R/LR/checks > $L/LR_checks2.log 2>&1
stage checks3LR  $PY $SRC checks3 --states LR:10:1,LR:11:20 --out $R/LR/checks > $L/LR_checks3.log 2>&1
stage checks3R   $PY $SRC checks3 --states R:10:1,R:10:2 --out $R/R/checks > $L/R_checks3.log 2>&1
# (a) + (b1)
stage measureLR  $PY $SRC measure --arms LR --seeds 10-19 --tasks 1,5,10,20,2,30 --K 65536 --out $R/LR/measure > $L/LR_measure.log 2>&1
stage measureR   $PY $SRC measure --arms R --seeds 10 --tasks 1,2,3 --K 65536 --out $R/R/measure > $L/R_measure.log 2>&1
# self shape + Adam decomposition
stage extraLR    $PY $SRC extra --arms LR --seeds 10-19 --tasks 1,10,20 --K 16384 --out $R/LR/extra > $L/LR_extra.log 2>&1
stage extraR     $PY $SRC extra --arms R --seeds 10 --tasks 1,2,3 --K 16384 --out $R/R/extra > $L/R_extra.log 2>&1
# (c) channel means at init and every checkpoint; replays; (b2) frozen reference
stage zbarLR     $PY $SRC zbar --arms LR --seeds 10-19 --tasks 0,1,2,3,5,10,20,30 --out $R/LR/zbar > $L/LR_zbar.log 2>&1
stage zbarR      $PY $SRC zbar --arms R --seeds 10 --tasks 0,1,2,3 --out $R/R/zbar > $L/R_zbar.log 2>&1
for t in 1 10 20; do
  stage replayLR$t $PY $SRC replay --task $t --arms LR --seeds 10-19 --out $R/LR/replay > $L/LR_replay_t$t.log 2>&1
done
stage frozenLR   $PY $SRC frozen --arms LR --seeds 10-19 --tasks 1,10,20 --L 32 --steps 3000 --out $R/LR/frozen > $L/LR_frozen.log 2>&1
echo "[$(date +%T)] phase2 chain done" >> $L/phase2_chain.log

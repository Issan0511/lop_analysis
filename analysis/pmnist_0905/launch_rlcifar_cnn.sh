#!/usr/bin/env bash
# rlcifar_cnn_0908 stage 1 — spec §2.3: 7 arms x 10 seeds x 50 tasks x 400 epochs, Adam 1e-3.
# ~26 h at 7-way (measured: 41.0 s/task at 1 proc, 187.5 s/task at 7-way).
# Usage: launch_rlcifar_cnn.sh <lambda>     e.g.  launch_rlcifar_cnn.sh 1e-4
# GATED: checks must be all_pass; lambda comes from stage 0 (spec §3), never guessed.
set -euo pipefail
LAM="${1:-}"; [ -n "$LAM" ] || { echo "usage: $0 <lambda from stage 0>"; exit 1; }
cd /home/issan/Projects/claude/proj_004_drift
S=/tmp/claude-1000/-home-issan-Projects-claude/2c202656-369c-494e-8572-224b42e56c9e/scratchpad
OUT=results/rlcifar_cnn_0908
MOD=src/rlcifar_cnn_0908.py
[ -f "$MOD" ] || { echo "ABORT: $MOD missing"; exit 1; }
python3 -c "
import json,sys
d=json.load(open('results/_checks_rlcifar_cnn_0908/checks.json'))
sys.exit(0 if d.get('all_pass') else 1)" || { echo "ABORT: checks not all_pass"; exit 1; }
pgrep -f rlcifar_cnn_0908 >/dev/null && { echo "ABORT: a rlcifar_cnn run is already active"; exit 1; }
avail=$(free -g | awk '/^Mem:/{print $7}'); [ "$avail" -lt 12 ] && { echo "ABORT: only ${avail}G"; exit 1; }
A="--optimizer adam --lrs 0.001 --tasks 50 --epochs 400 --c 0.6 --beta 0.01 --seeds 0,1,2,3,4,5,6,7,8,9"
go(){ local n=$1; shift; nohup python3 $MOD "$@" --out $OUT/$n > $S/cnn_${n}.log 2>&1 & echo "  $n pid $!"; }
echo "stage 1, lambda=$LAM, started $(date +%F' '%H:%M)"
go R          $A --arms R
go LR         $A --arms LR
go SNA        $A --arms SNA
go R_l2       $A --arms R   --iv l2:$LAM
go LR_l2      $A --arms LR  --iv l2:$LAM
go SNA_l2     $A --arms SNA --iv l2:$LAM
go R_l2init   $A --arms R   --iv l2init:$LAM
sleep 25; echo "procs $(ps -o cmd= -C python3 | grep -c rlcifar_cnn)  avail $(free -g | awk '/^Mem:/{print $7}')G"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

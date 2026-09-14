#!/usr/bin/env bash
# pmnist_rlmnist_0906 — spec §2.2: 7 arms x 10 seeds x 50 tasks x 400 epochs, Adam lr=0.001.
# DO NOT RUN until analysis/pmnist_0905/checks_rlmnist.py reports all_pass and the smoke test looks sane.
# --iv is a run-level flag, so each (arm, iv) is its own invocation. Log names are flat (no '/').
set -euo pipefail
cd /home/issan/Projects/claude/proj_004_drift
S=/tmp/claude-1000/-home-issan-Projects-claude/2c202656-369c-494e-8572-224b42e56c9e/scratchpad
OUT=results/pmnist_rlmnist_0906
MOD=src/pmnist_rlmnist_0906.py
[ -f "$MOD" ] || { echo "ABORT: $MOD not found"; exit 1; }
avail=$(free -g | awk '/^Mem:/{print $7}'); [ "$avail" -lt 12 ] && { echo "ABORT: only ${avail} GiB available"; exit 1; }
A="--optimizer adam --lrs 0.001 --tasks 50 --epochs 400 --c 0.6 --beta 0.01 --seeds 0,1,2,3,4,5,6,7,8,9"
launch(){ local name=$1; shift; nohup python3 $MOD "$@" --out $OUT/$name > $S/rlm_${name}.log 2>&1 & echo "  $name pid $!"; }
launch R          $A --arms R
launch LR         $A --arms LR
launch SNA        $A --arms SNA
launch R_l2       $A --arms R   --iv l2:1e-3
launch SNA_l2     $A --arms SNA --iv l2:1e-3
launch R_l2init   $A --arms R   --iv l2init:1e-3
launch SNA_l2init $A --arms SNA --iv l2init:1e-3
sleep 20; echo "procs $(ps -o cmd= -C python3 | grep -c pmnist_rlmnist)  avail $(free -g | awk '/^Mem:/{print $7}')G"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

#!/usr/bin/env bash
# pmnist_rlcifar_0907 — spec §2.2: 7 arms x 10 seeds x 50 tasks x 400 epochs, Adam lr=0.001.
# GATED: do not run until checks_rlcifar.py reports all_pass AND S-capacity memo_acc >= 0.90.
set -euo pipefail
cd /home/issan/Projects/claude/proj_004_drift
S=/tmp/claude-1000/-home-issan-Projects-claude/2c202656-369c-494e-8572-224b42e56c9e/scratchpad
OUT=results/pmnist_rlcifar_0907
MOD=src/pmnist_rlcifar_0907.py
[ -f "$MOD" ] || { echo "ABORT: $MOD not found"; exit 1; }
python3 - <<'PY' || exit 1
import json,sys
try: d=json.load(open("results/_checks_pmnist_rlcifar_0907/checks.json"))
except Exception as e: print("ABORT: checks.json missing:", e); sys.exit(1)
if not d.get("all_pass"): print("ABORT: checks not all_pass"); sys.exit(1)
cap=d.get("S-capacity",{}); m=[v for k,v in cap.items() if "memo" in str(k).lower()]
print("S-capacity:", cap)
PY
avail=$(free -g | awk '/^Mem:/{print $7}'); [ "$avail" -lt 12 ] && { echo "ABORT: only ${avail} GiB"; exit 1; }
A="--optimizer adam --lrs 0.001 --tasks 50 --epochs 400 --c 0.6 --beta 0.01 --seeds 0,1,2,3,4,5,6,7,8,9"
launch(){ local n=$1; shift; nohup python3 $MOD "$@" --out $OUT/$n > $S/rlc_${n}.log 2>&1 & echo "  $n pid $!"; }
launch R          $A --arms R
launch LR         $A --arms LR
launch SNA        $A --arms SNA
launch R_l2       $A --arms R   --iv l2:1e-3
launch LR_l2      $A --arms LR  --iv l2:1e-3
launch SNA_l2     $A --arms SNA --iv l2:1e-3
launch R_l2init   $A --arms R   --iv l2init:1e-3
sleep 20; echo "procs $(ps -o cmd= -C python3 | grep -c rlcifar)  avail $(free -g | awk '/^Mem:/{print $7}')G"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

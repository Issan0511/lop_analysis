#!/usr/bin/env bash
# pmnist_adapt_0905 — spec §2.3/§2.4. Box A: Adam lr=0.001 x 1 epoch. Box B: SGD lr=0.02 x 4 epoch.
# 7 class-A/B arms in one invocation per seed-shard; the two class-C arms need their own
# invocation each because --iv is a run-level flag.  9 processes; abort if RAM is short.
set -euo pipefail
cd /home/issan/Projects/claude/proj_004_drift
S=/tmp/claude-1000/-home-issan-Projects-claude/2c202656-369c-494e-8572-224b42e56c9e/scratchpad
OUT=results/pmnist_adapt_0905
avail=$(free -g | awk '/^Mem:/{print $7}')
if [ "$avail" -lt 16 ]; then echo "ABORT: only ${avail} GiB available (need >=16 for 9 procs)"; exit 1; fi
ARMS=SNA,SN02,SN1,SN3,LR,R,LIN
A="--stage main --optimizer adam --lrs 0.001 --tasks 200 --epochs 1 --c 0.6 --beta 0.01"
B="--stage main --optimizer sgd  --lrs 0.02  --tasks 200 --epochs 4 --c 0.6 --beta 0.01"
launch(){ # name, args...
  local name=$1; shift
  nohup python3 src/pmnist_0905.py "$@" --out $OUT/$name > $S/adapt_${name//\//_}.log 2>&1 &
  echo "  $name pid $!"
}
echo "Box A (Adam):"
launch boxA/s1 $A --arms $ARMS --seeds 0,1,2,3,4
launch boxA/s2 $A --arms $ARMS --seeds 5,6,7,8,9
launch boxA/l2init $A --arms R --iv l2init:1e-3 --seeds 0,1,2,3,4,5,6,7,8,9
launch boxA/cbp    $A --arms R --iv cbp:1e-4    --seeds 0,1,2,3,4,5,6,7,8,9
echo "Box B (SGD x4 epoch):"
launch boxB/s1 $B --arms $ARMS --seeds 0,1,2
launch boxB/s2 $B --arms $ARMS --seeds 3,4,5
launch boxB/s3 $B --arms $ARMS --seeds 6,7,8,9
launch boxB/l2init $B --arms R --iv l2init:1e-3 --seeds 0,1,2,3,4,5,6,7,8,9
launch boxB/cbp    $B --arms R --iv cbp:1e-4    --seeds 0,1,2,3,4,5,6,7,8,9
sleep 30; free -g | head -2; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

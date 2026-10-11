#!/bin/bash
# One continued task from t10 with alpha frozen at its t10 value during the task (the sunk
# state kept, its tracking of V switched off) against the unchanged continuation (exploratory).
cd "$(dirname "$0")/../.." || exit 1
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
R=results/sna_cnn_cause_1009
OUT=$R/Z10
mkdir -p $OUT
for i in 1 2 3 4 5; do
  [ -f $OUT/provenance.json ] && break
  "$PY" src/sna_cnn_cause_1009_fork.py --src $R/A --at 10 \
    --forks SNA:SNA,SNA:SNA@frz,CV3FC06:CV3FC06,CV3FC06:CV3FC06@frz,CV06FC3:CV06FC3,CV06FC3:CV06FC3@frz \
    --tasks 1 --out $OUT >> $OUT/log.txt 2>&1 < /dev/null && break
  sleep 60
done

#!/bin/bash
# Epoch-resolved trace of one continued task from t1 and from t10, all layers vs conv frozen
# (exploratory): gates and seats of f1/f2, feature drift, per-layer move, and the fit read
# with the task-start conv.
cd "$(dirname "$0")/../.." || exit 1
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
R=results/sna_cnn_cause_1009
for AT in 10 1; do
  OUT=$R/T$AT
  mkdir -p $OUT
  for i in 1 2 3 4 5; do
    [ -f $OUT/provenance.json ] && break
    "$PY" src/sna_cnn_cause_1009_fork.py --src $R/A --at $AT \
      --forks SNA:SNA,SNA:SNA+fixconv,CV3FC06:CV3FC06,CV3FC06:CV3FC06+fixconv,CV06FC3:CV06FC3,CV06FC3:CV06FC3+fixconv \
      --tasks 1 --trace 0,1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,75,100,200,400 --out $OUT >> $OUT/log.txt 2>&1 < /dev/null && break
    sleep 60
  done
done

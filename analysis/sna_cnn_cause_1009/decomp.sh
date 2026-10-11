#!/bin/bash
# One continued task from t1 and from t10, with and without the conv layers frozen
# (exploratory, not registered): how much of the fitting speed the conv learning carries.
cd "$(dirname "$0")/../.." || exit 1
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
R=results/sna_cnn_cause_1009
for AT in 1 10; do
  OUT=$R/D$AT
  mkdir -p $OUT
  for i in 1 2 3 4 5; do
    [ -f $OUT/provenance.json ] && break
    "$PY" src/sna_cnn_cause_1009_fork.py --src $R/A --at $AT \
      --forks SNA:SNA,SNA:SNA+fixconv,CV3FC06:CV3FC06,CV3FC06:CV3FC06+fixconv,CV06FC3:CV06FC3,CV06FC3:CV06FC3+fixconv \
      --tasks 1 --out $OUT >> $OUT/log.txt 2>&1 < /dev/null && break
    sleep 60
  done
done

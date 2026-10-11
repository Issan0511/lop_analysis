#!/bin/bash
# spec §8.3 QF-mech: one traced task from t30 for the fcamp arms (bundle C) and for SNA (bundle A).
cd "$(dirname "$0")/../.." || exit 1
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
R=results/sna_cnn_cause_1009
TR=0,1,2,3,5,10,20,50,400
for spec in "C|SNA+fcamp0.7:SNA+fcamp0.7,SNA+fcamp0.9:SNA+fcamp0.9|QC30" "A|SNA:SNA,CV06FC3:CV06FC3|QA30"; do
  IFS='|' read -r SRC FORKS OUT <<< "$spec"
  # wait for the source bundle's t30 checkpoints (the last seed's file is written last)
  first=${FORKS%%:*}
  until [ -f "$R/$SRC/ckpt/${first}_seed19_t30.pt" ]; do sleep 30; done
  mkdir -p $R/$OUT
  for i in 1 2 3; do
    [ -f $R/$OUT/provenance.json ] && break
    "$PY" src/sna_cnn_cause_1009_fork.py --src $R/$SRC --at 30 --forks "$FORKS" --tasks 1 --trace $TR \
      --out $R/$OUT >> $R/$OUT/log.txt 2>&1 < /dev/null && break
    sleep 60
  done
done

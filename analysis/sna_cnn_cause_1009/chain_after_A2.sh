#!/bin/bash
# After bundle A (provenance written): B1 (SNAfrz1 and the two fixconv arms), then the t30
# forks, then B2 (F1ONLY, F2ONLY).  Same registered arms as spec §7.3, split for order only.
cd "$(dirname "$0")/../.." || exit 1
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
R=results/sna_cnn_cause_1009
until [ -f "$R/A/provenance.json" ]; do sleep 60; done
analysis/sna_cnn_cause_1009/launch.sh B1 SNAfrz1,SNA+fixconv,CV06FC3+fixconv 10-19 30
mkdir -p "$R/F30"
for i in $(seq 1 20); do
  if [ -f "$R/F30/provenance.json" ]; then break; fi
  "$PY" src/sna_cnn_cause_1009_fork.py --src "$R/A" --at 30 \
    --forks SNA:SNA,SNA:CV06FC3,SNA:CV3FC06,SNA:SNA@f3x0.5,CV3FC06:CV3FC06,CV3FC06:SNAc3 \
    --tasks 10 --out "$R/F30" >> "$R/F30/log.txt" 2>&1 < /dev/null && break
  echo "[$(date +%T)] fork exit, retry $i in 120 s" >> "$R/F30/log.txt"
  sleep 120
done
analysis/sna_cnn_cause_1009/launch.sh B2 F1ONLY,F2ONLY 10-19 30

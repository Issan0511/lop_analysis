#!/bin/bash
# 介入 8 本を同じ GPU で並列に回す（2 ブロックは小さい）
cd "$(dirname "$0")/../../../../.."
O=results/chargpt_cl_1019/quicklook/M2/intervene; PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
export CHARGPT_NLAYER=2 OMP_NUM_THREADS=2
for spec in control:40:55 lnfix:40:55 lnex1:40:55 nocommon:40:55 l0freeze:40:55 control:25:40 gamma03:25:40 gamma03f:25:40; do
  IFS=: read arm a b <<< "$spec"; d=$O/${arm}_from$a
  [ -f $d/provenance.json ] && continue
  mkdir -p $d; setsid -f $PY analysis/chargpt_cl_1019_quicklook/intervene_revival.py --arm $arm --from_task $a --to_task $b --out $d > $d/run.log 2>&1 < /dev/null
done

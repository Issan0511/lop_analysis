#!/bin/bash
# 追補 9 のジョブ 1 本: "l0resp ARM SEED T0" または "elr MODE SEED"。出力に provenance.json があれば飛ばす
cd "$(dirname "$0")/../../.."
PY=${PY:-.venv/bin/python}; [ -x $PY ] || PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
export CHARGPT_NLAYER=2 OMP_NUM_THREADS=2
R=results/chargpt_cl_1019/row5; set -- $1
if [ "$1" = l0resp ]; then
  sub=$([ "$4" = 90 ] && echo main || echo side); d=$R/$sub/$2/seed$3
  [ -f $d/provenance.json ] && exit 0; mkdir -p $d
  $PY analysis/chargpt_cl_1019_quicklook/row5_l0resp.py --arm $2 --seed $3 --t0 $4 --out $d > $d/run.log 2>&1
else
  d=$R/elr/$2/seed$3; [ -f $d/provenance.json ] && exit 0; mkdir -p $d
  $PY analysis/chargpt_cl_1019_quicklook/row5_elr.py --mode $2 --seed $3 --out $d > $d/run.log 2>&1
fi
echo "$(date '+%F %T') done $* rc=$?" >> $R/launch_$(hostname).log

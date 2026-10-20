#!/bin/bash
# 追補 7: 2 ブロックの C3-2・ReLUcap-2 を seed ごとに順に回す。引数は "腕:seed" の並び（例: C3:0 ReLUcap:0 C3:1 …）。
# 走っている間はこのファイルを書き換えない。
cd "$(dirname "$0")/.."
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
R=results/chargpt_cl_1019; mkdir -p $R/M2 $R/m2_logs
for spec in "$@"; do
  A=${spec%%:*}; S=${spec##*:}
  [ -f $R/M2/$A/seed$S/provenance.json ] && { echo "$(date '+%F %T') skip $A seed$S (done)" >> $R/m2_launch.log; continue; }
  echo "$(date '+%F %T') start $A seed$S on $(hostname) (addendum 7)" >> $R/m2_launch.log
  CHARGPT_NLAYER=2 CHARGPT_DECOMP_CKPT=1 $PY src/chargpt_cl_1019_kune.py run --box M --arm $A --seed $S --tasks 100 --n_per_task 10000 --alpha 0.01 \
    --fresh_tasks 10 25 50 100 --out $R/M2/$A/seed$S > $R/m2_logs/${A}_seed$S.log 2>&1
  echo "$(date '+%F %T') done $A seed$S rc=$?" >> $R/m2_launch.log
done

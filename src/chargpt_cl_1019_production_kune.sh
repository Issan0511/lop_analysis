#!/usr/bin/env bash
# chargpt_cl_1019 kunekune 族の腕（spec 追補 5、Issa の go のあとで回す）: 箱 M、alpha 0.01、N_M 10000、50 課題。
# 1 seed の 4 腕（KKT1 SNA GELU SNAfrz1）を続けて 1 本ずつ回す。完了した走（provenance.json あり）は飛ばす。
# 使い方: PY=<python> bash src/chargpt_cl_1019_production_kune.sh 0 1 2 3 4 5 6 7 8 9
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
OUT=results/chargpt_cl_1019/M
mkdir -p "$OUT/logs"
host=$(hostname)
for s in "$@"; do
  for arm in KKT1 SNA GELU SNAfrz1; do
    d="$OUT/$arm/seed$s"
    if [ -f "$d/provenance.json" ]; then echo "$(date '+%F %T') skip $d"; continue; fi
    rm -rf "$d"
    echo "$(date '+%F %T') start $d on $host"
    $PY src/chargpt_cl_1019_kune.py run --box M --arm "$arm" --seed "$s" --tasks 50 --n_per_task 10000 --alpha 0.01 \
        --out "$d" > "$OUT/logs/${arm}_seed${s}.log" 2>&1
    echo "$(date '+%F %T') done $d rc=$?"
  done
  echo "$(date '+%F %T') seed_done $s"
done
echo "$(date '+%F %T') KUNE_FINISHED $*"

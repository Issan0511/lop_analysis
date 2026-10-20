#!/usr/bin/env bash
# chargpt_cl_1019 追加腕（spec 追補 3、Issa の承認後に回す）: 箱 M、alpha 0.01、N_M 10000、50 課題。
# 1 seed の 3 腕（C3H B3H B3HB）を続けて 1 本ずつ回す。完了した走（provenance.json あり）は飛ばす。
# 使い方: PY=<python> bash src/chargpt_cl_1019_production_center.sh 0 1 2 3 4 5 6 7 8 9
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
OUT=results/chargpt_cl_1019/M
mkdir -p "$OUT/logs"
host=$(hostname)
for s in "$@"; do
  for arm in C3H B3H B3HB; do
    d="$OUT/$arm/seed$s"
    if [ -f "$d/provenance.json" ]; then echo "$(date '+%F %T') skip $d"; continue; fi
    rm -rf "$d"
    echo "$(date '+%F %T') start $d on $host"
    $PY src/chargpt_cl_1019_center.py run --box M --arm "$arm" --seed "$s" --tasks 50 --n_per_task 10000 --alpha 0.01 \
        --out "$d" > "$OUT/logs/${arm}_seed${s}.log" 2>&1
    echo "$(date '+%F %T') done $d rc=$?"
  done
  echo "$(date '+%F %T') seed_done $s"
done
echo "$(date '+%F %T') CENTER_FINISHED $*"

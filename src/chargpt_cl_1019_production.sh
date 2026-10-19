#!/usr/bin/env bash
# chargpt_cl_1019 本走（spec 追補 2）: 箱 M だけ、alpha 0.01、N_M 10000、50 課題、4 腕。
# 1 seed の 4 腕を続けて 1 本ずつ回す。完了した走（provenance.json あり）は飛ばす。
# 使い方: bash src/chargpt_cl_1019_production.sh 0 1 2 3 4 5 6 7 8 9
#   別マシンでは担当の seed だけを渡す（1 seed の 4 腕は同じマシンで回す）。
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
OUT=results/chargpt_cl_1019/M
mkdir -p "$OUT/logs"
host=$(hostname)
for s in "$@"; do
  for arm in C3 C4 B3 N4; do
    d="$OUT/$arm/seed$s"
    if [ -f "$d/provenance.json" ]; then echo "$(date '+%F %T') skip $d"; continue; fi
    rm -rf "$d"
    echo "$(date '+%F %T') start $d on $host"
    $PY src/chargpt_cl_1019.py run --box M --arm "$arm" --seed "$s" --tasks 50 --n_per_task 10000 --alpha 0.01 \
        --out "$d" > "$OUT/logs/${arm}_seed${s}.log" 2>&1
    echo "$(date '+%F %T') done $d rc=$?"
  done
  echo "$(date '+%F %T') seed_done $s"
done
echo "$(date '+%F %T') PRODUCTION_FINISHED $*"

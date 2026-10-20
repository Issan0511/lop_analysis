#!/bin/bash
# κ の分割を、そろっている状態について順に回す（1 状態 5 秒前後）
cd "$(dirname "$0")/../../../.."
R=results/chargpt_cl_1019/quicklook; PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python; O=$R/rho/kappa
mkdir -p $O/C3 $O/SiLU
st_a() { local arm=$1 k=$2; for d in $R/M/$arm/seed1000/ckpts $R/rho/cont/${arm}_10_15/ckpts $R/rho/cont/${arm}_15_39/ckpts $R/rho/cont/${arm}_50_99/ckpts; do [ -f $d/task$k.pt ] && { echo $d/task$k.pt; return; }; done; }
run() { local arm=$1 t=$2; local k=$((t-1)); local a=$(st_a $arm $k); [ -z "$a" ] && { echo "skip $arm t$t (no state)"; return; }
  local b=$R/rho/states/$arm/u100_task$t.pt; local B=""; [ -f $b ] && B="--state_b $b"
  [ -f $O/$arm/t${t}.json ] || $PY analysis/chargpt_cl_1019_quicklook/kappa_split.py --arm $arm --state $a $B --task $t --out $O/$arm/t${t}.json 2>&1 | grep -v Warning | tail -1; }
for t in 2 3 4 5 6 7 8 9 10 40 41 42 43 44 45 46 47 48 49 50; do run C3 $t; done
for t in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25 26 30 35 40 41 42 43 44 45 46 47 48 49 50 60 65 70 75 80 85 90 95 100; do run SiLU $t; done

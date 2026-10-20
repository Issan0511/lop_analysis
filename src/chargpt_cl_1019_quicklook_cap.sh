#!/bin/bash
# 下見（未登録）: 幅の上限 cap の腕と cap なしの対照を seed 1000 で順に回す（Issa 1020 夕方、Neokunekune 経由）。
# cap なしの 3 腕は Δμ の分解用に課題 1〜10・40〜50 の更新 100 と課題末の checkpoint を残す（CHARGPT_DECOMP_CKPT=1）。
# 走っている間はこのファイルを書き換えない（bash は読み進める）。
cd "$(dirname "$0")/.."
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
Q=results/chargpt_cl_1019/quicklook
mkdir -p $Q/logs
for spec in SiLU:1 SiLUcap:0 C3:1 ReLUcap:0 GELU:1 GELUcap:0; do
  A=${spec%%:*}; D=${spec##*:}
  echo "$(date '+%F %T') start $A seed1000 (quick look; cap study; decomp_ckpt=$D)" >> $Q/launch.log
  CHARGPT_DECOMP_CKPT=$D $PY src/chargpt_cl_1019_kune.py run --box M --arm $A --seed 1000 --tasks 50 --n_per_task 10000 --alpha 0.01 \
    --out $Q/M/$A/seed1000 > $Q/logs/${A}_seed1000.log 2>&1
  echo "$(date '+%F %T') done $A seed1000 rc=$?" >> $Q/launch.log
done

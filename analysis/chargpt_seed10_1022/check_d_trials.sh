#!/bin/bash
# D の検査の短い試走（GPU 1 本ずつ順に）。出力 = $S10/results/chargpt_seed10_1022/checks/D/trial/
#  1. SiLU 4 ブロック seed 0〜9、課題 1 だけ（データ列の sha の照合・seed 0 は 2 課題の試走との再現の照合）
#  2. SiLU 4 ブロック seed 1000、課題 1 だけ（white-san の quicklook/M/SiLU/seed1000 とのビット照合）
#  3. C3 4 ブロック seed 0、課題 1 だけ（white-san の crosscheck_white-san/M/C3/seed0 と byakko の本走 M/C3/seed0 との照合）
#  4. 保存の仕組みの試走: SiLU seed 0、n_per_task 200・16 課題・fresh 10（足した課題 11〜15 の課題末と更新 100 が出来るか）
set -u
CG=/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019
S10=/home/kubo/project/Nakatsuka/lop_seed10
PY=$CG/.venv/bin/python
W=$S10/analysis/chargpt_seed10_1022/run_silu4.py
O=$S10/results/chargpt_seed10_1022/checks/D
export CHARGPT_SRC=$CG
for k in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $O/trial/SiLU_seed${k}_t1/provenance.json ] && continue
  CHARGPT_NLAYER=4 CHARGPT_DECOMP_CKPT=1 $PY $W run --box M --arm SiLU --seed $k --tasks 1 --n_per_task 10000 --alpha 0.01 --fresh_tasks \
    --out $O/trial/SiLU_seed${k}_t1 > $O/logs/SiLU_seed${k}_t1.log 2>&1 || echo "FAIL SiLU seed $k"
done
[ -f $O/trial/SiLU_seed1000_t1/provenance.json ] || CHARGPT_NLAYER=4 CHARGPT_DECOMP_CKPT=1 $PY $W run --box M --arm SiLU --seed 1000 --tasks 1 --n_per_task 10000 --alpha 0.01 --fresh_tasks \
    --out $O/trial/SiLU_seed1000_t1 > $O/logs/SiLU_seed1000_t1.log 2>&1 || echo "FAIL SiLU seed 1000"
[ -f $O/trial/C3_seed0_t1/provenance.json ] || CHARGPT_NLAYER=4 $PY $CG/src/chargpt_cl_1019_kune.py run --box M --arm C3 --seed 0 --tasks 1 --n_per_task 10000 --alpha 0.01 --fresh_tasks \
    --out $O/trial/C3_seed0_t1 > $O/logs/C3_seed0_t1.log 2>&1 || echo "FAIL C3 seed 0"
[ -f $O/trial/SiLU_seed0_mech/provenance.json ] || CHARGPT_NLAYER=4 CHARGPT_DECOMP_CKPT=1 $PY $W run --box M --arm SiLU --seed 0 --tasks 16 --n_per_task 200 --alpha 0.01 --fresh_tasks 10 \
    --out $O/trial/SiLU_seed0_mech > $O/logs/SiLU_seed0_mech.log 2>&1 || echo "FAIL mech"
echo ALL_DONE

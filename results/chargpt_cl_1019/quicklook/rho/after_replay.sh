#!/bin/bash
cd /home/issan/Projects/claude/wt/chargpt_cl_1019
R=results/chargpt_cl_1019/quicklook; PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
while kill -0 383603 2>/dev/null; do sleep 30; done
$PY -c "
import numpy as np
a=np.load('$R/rho/cont/SiLU_50_99/online.npz')['loss']; b=np.load('$R/M/SiLU_cont/seed1000/online.npz')['loss'][:49]
print('SiLU 課題 51〜99 の再生: 一致した更新', int((a==b).sum()), '/', a.size)" >> $R/rho/after_replay.log 2>&1
$PY analysis/chargpt_cl_1019_quicklook/replay_u100.py --arm SiLU --ckpts $R/rho/cont/SiLU_50_99/ckpts --ks 59 64 69 74 79 84 89 94 99 --out $R/rho/states/SiLU_late >> $R/rho/after_replay.log 2>&1
mv $R/rho/states/SiLU_late/u100_task*.pt $R/rho/states/SiLU/ && mv $R/rho/states/SiLU_late/replay_u100_SiLU.json $R/rho/states/SiLU/replay_u100_SiLU_60_100.json
J=$R/rho/jobs_silu_60_100.txt; : > $J
for k in 59 64 69 74 79 84 89 94 99; do t=$((k+1)); D=$R/rho/cont/SiLU_50_99/ckpts
  echo "SiLU $D/task$k.pt $D/task${k}_adam.pt $t $R/rho/out/SiLU/t${t}_a_chain.json chain 0" >> $J
  echo "SiLU $R/rho/states/SiLU/u100_task$t.pt $R/rho/states/SiLU/u100_task${t}_adam.pt $t $R/rho/out/SiLU/t${t}_b_chain.json chain 0" >> $J
  echo "SiLU $D/task$k.pt $D/task${k}_adam.pt $t $R/rho/out/SiLU/t${t}_a_uniform.json uniform 1" >> $J
done
echo "$(date '+%F %T') rho: SiLU windows 60-100 queued ($(wc -l < $J) jobs)" >> $R/launch.log
cat $J | xargs -P 2 -I{} bash $R/rho/run_job_gpu.sh '{}'
echo "$(date '+%F %T') rho SiLU 60-100 jobs finished" >> $R/launch.log

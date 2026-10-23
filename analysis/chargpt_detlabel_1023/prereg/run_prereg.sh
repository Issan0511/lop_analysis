#!/bin/bash
# spec §2・§4.3・§5 の表を出したコマンド（2026-10-23、white-san、numpy 2.5.1・torch 2.13.0+cu130・python 3.12.3）。
# 学習はしない。重い連鎖（α ≤ 1e−3 で定常分布の反復が上限に当たるもの）は 1 本 80〜200 秒かかるので、並べて回した。
set -eu
cd "$(dirname "$0")"
PY=${PY:-/home/issan/Projects/claude/proj_004_drift/.venv/bin/python}
SCR=${SCR:?出力の作業フォルダ（git の外）}
R=../../../results/chargpt_detlabel_1023/prereg
mkdir -p $SCR/out $SCR/ref $R
bash extract_a2.sh $SCR/ref
$PY -I a2_table.py $SCR/ref $R/a2_table.json | tee $R/a2_table.txt
$PY -I a2_motion.py $R/a2_motion.json | tee $R/a2_motion.txt
$PY -I a2_motion_alive.py $SCR/ref $R/a2_motion_alive.json | tee $R/a2_motion_alive.txt
$PY -I a2_j4_fallback.py $R | tee $R/a2_j4_fallback.txt
OMP_NUM_THREADS=1 $PY -I argmax_all.py $R/argmax_all.json | tee $R/argmax_all.txt
for a in 0.01 0.001 0.0001; do for s in 0 1 2 3 4 5 6 7 8 9; do for t in 1 2 3; do echo "$a $s $t"; done; done; done |
  xargs -P 10 -L 1 bash -c 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '"$PY"' -I chain_stats.py '"$SCR"'/out/full_a$0_s$1_t$2.json $0 $1 $2'
$PY -I chain_table.py $SCR/out | tee $R/chain_table.md
$PY -I -c "import glob,json,sys; json.dump([json.load(open(f)) for f in sorted(glob.glob(sys.argv[1]+'/full_a*_s*_t*.json'))], open(sys.argv[2],'w'), indent=0)" $SCR/out $R/chain_stats_all.json

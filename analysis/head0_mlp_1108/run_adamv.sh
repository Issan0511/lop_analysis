#!/usr/bin/env bash
# head0_mlp_1108 追補 C（Adam の v の記憶の向き）: 5 活性化 × 2 seed の 10 走を、CPU 1 スレッドずつ 5 本同時（seed ごとの 2 波）で回す。
# 使い方: bash analysis/head0_mlp_1108/run_adamv.sh   （生データは results/head0_mlp_1108/adamv/raw/、ログは logs/。1 走 ≈ 6〜10 分）
# 集計: PYTHONPATH=. $PY -m analysis.head0_mlp_1108.judge_adamv
set -u
H=/home/issan/Projects/claude/wt/head0_cap_1108
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
OUT=$H/results/head0_mlp_1108/adamv/raw
LOG=$H/results/head0_mlp_1108/adamv/logs
mkdir -p "$OUT" "$LOG"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "start $(date '+%m-%d %H:%M:%S') commit $(git -C "$H" rev-parse --short HEAD)" >> "$LOG/run_all.log"
for s in 0 1; do
  for act in LR R ELU SILU GELU; do
    ( cd "$H" && PYTHONPATH="$H" "$PY" src/adamv_1108.py --act "$act" --seed "$s" --out "$OUT" > "$LOG/${act}_s${s}.log" 2>&1 < /dev/null
      echo "end $act s$s rc=$? $(date '+%m-%d %H:%M:%S')" >> "$LOG/run_all.log" ) &
  done
  wait
done
echo "ALL_DONE $(date '+%m-%d %H:%M:%S')" >> "$LOG/run_all.log"

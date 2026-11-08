#!/usr/bin/env bash
# head0_mlp_1108: 5 活性化 × 2 seed の 10 走を、CPU 1 スレッドずつ同時に回す。
# 使い方: bash analysis/head0_mlp_1108/run_all.sh   （worktree の中で。生データは results/head0_mlp_1108/raw/、ログは logs/）
set -u
H=/home/issan/Projects/claude/wt/head0_mlp_1108
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
OUT=$H/results/head0_mlp_1108/raw
LOG=$H/results/head0_mlp_1108/logs
mkdir -p "$OUT" "$LOG"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "start $(date '+%m-%d %H:%M:%S') commit $(git -C "$H" rev-parse --short HEAD)" >> "$LOG/run_all.log"
for act in LR R ELU SILU GELU; do
  for s in 0 1; do
    ( cd "$H" && PYTHONPATH="$H" "$PY" src/head0_mlp_1108.py --act "$act" --seed "$s" --out "$OUT" > "$LOG/${act}_s${s}.log" 2>&1 < /dev/null
      echo "end $act s$s rc=$? $(date '+%m-%d %H:%M:%S')" >> "$LOG/run_all.log" ) &
  done
done
wait
echo "ALL_DONE $(date '+%m-%d %H:%M:%S')" >> "$LOG/run_all.log"

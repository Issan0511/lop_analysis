#!/bin/bash
# cifar5p1_mlp_0920 main run: 16 arms x seeds 0-9, std x lr 1e-4 (spec section 3.4),
# plus the equal-parameter (hidden=94) control for the two width-doubling arms.
set -u
PY=/home/i_nakatsuka/Projects/lop_analysis/.venv/bin/python
cd "$(dirname "$0")"
LOG=results/cifar5p1_mlp_0920/_log
for arm in SNA KKA R LR KKT1 KKA23 SL RSL LK001 LK03 ELU SILU GELU LK07 CR DF; do
  echo "=== $arm $(date +%T) ==="
  $PY src/cifar5p1_mlp_0920.py run --arm "$arm" --seeds 0-9 --cond std --lr 1e-4 \
      > "$LOG/$arm.log" 2>&1 || echo "FAILED $arm"
  tail -2 "$LOG/$arm.log"
done
for arm in CR DF; do
  echo "=== $arm hidden=94 $(date +%T) ==="
  $PY src/cifar5p1_mlp_0920.py run --arm "$arm" --seeds 0-9 --cond std --lr 1e-4 --hidden 94 \
      > "$LOG/${arm}_h94.log" 2>&1 || echo "FAILED $arm h94"
  tail -2 "$LOG/${arm}_h94.log"
done
echo "=== done $(date +%T) ==="

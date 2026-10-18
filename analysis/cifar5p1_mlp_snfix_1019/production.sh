#!/usr/bin/env bash
# Registered main run and verdict of cifar5p1_mlp_snfix_1019 (spec §3-§5): the three new arms first, then the
# 16 arms of 0920 in its launch order, one process per arm as in 0920, then the verdict once all 19 have ended.
set -euo pipefail
run_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$run_root"
runtime=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
export PYTHONUNBUFFERED=1
for arm in SN06 SN3 SNAfrz1 SNA KKA R LR KKT1 KKA23 SL RSL LK001 LK03 ELU SILU GELU LK07 CR DF; do
  "$runtime" src/cifar5p1_mlp_snfix_1019.py run --arm "$arm"
done
"$runtime" analysis/cifar5p1_mlp_snfix_1019/verdict.py --root results/cifar5p1_mlp_snfix_1019

#!/usr/bin/env bash
# Registered main run and verdict of cifar5p1_cnn_budget_1019 (spec §3-§6): the three budgets in turn
# (780 first, so that S-reuse-780 can be read early), then the verdict once all three have ended.
set -euo pipefail
run_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$run_root"
runtime=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
export PYTHONUNBUFFERED=1
R=results/cifar5p1_cnn_budget_1019
for b in 780 2340 7800; do
  "$runtime" src/cifar5p1_cnn_budget_1019.py main --budget "$b" --out "$R/b$b"
done
"$runtime" analysis/cifar5p1_cnn_budget_1019/verdict.py --root "$R"

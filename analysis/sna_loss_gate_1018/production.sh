#!/usr/bin/env bash
set -euo pipefail
run_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$run_root"
runtime=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
"$runtime" src/sna_loss_gate_1018.py --seeds 10-19 --tasks 30 --epochs 400 --lr 0.001 \
    --out results/sna_loss_gate_1018/cnn
"$runtime" analysis/sna_loss_gate_1018/verdict.py --out results/sna_loss_gate_1018/cnn

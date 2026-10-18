#!/usr/bin/env bash
# Registered production run of neokunekune_cifar5p1_cnn_1018 (spec §2-§8), then the verdict.
set -euo pipefail
run_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$run_root"
runtime=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
export PYTHONUNBUFFERED=1
out=results/neokunekune_cifar5p1_cnn_1018/main
"$runtime" src/neokunekune_cifar5p1_cnn_1018.py run --seeds 2000-2009 --tasks 30 --out "$out"
"$runtime" analysis/neokunekune_cifar5p1_cnn_1018/verdict.py --src "$out"

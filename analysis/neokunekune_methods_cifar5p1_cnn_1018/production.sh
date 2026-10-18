#!/usr/bin/env bash
# Registered calibration -> selection -> main -> verdict of neokunekune_methods_cifar5p1_cnn_1018 (spec §4-§6).
set -euo pipefail
run_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$run_root"
runtime=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
export PYTHONUNBUFFERED=1
R=results/neokunekune_methods_cifar5p1_cnn_1018
"$runtime" src/neokunekune_methods_cifar5p1_cnn_1018.py calib --out "$R/calib"
"$runtime" src/neokunekune_methods_cifar5p1_cnn_1018.py main --out "$R/main"
"$runtime" analysis/neokunekune_methods_cifar5p1_cnn_1018/verdict.py --main "$R/main" --selected "$R/calib/selected.json"

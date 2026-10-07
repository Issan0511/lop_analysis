#!/usr/bin/env bash
# Registered main run of resp_cifar5p1_1007 (spec §8).  One GPU process; the runner itself
# takes the shared lock /tmp/lop_analysis_gpu.lock and refuses unverified or uncommitted code.
set -euo pipefail
cd "$(dirname "$0")/../.."
run=resp_cifar5p1_1007
python_bin=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
mkdir -p "results/$run/logs"
exec "$python_bin" -u -m "src.$run" --out "results/$run" > "results/$run/logs/${run}_main.log" 2>&1

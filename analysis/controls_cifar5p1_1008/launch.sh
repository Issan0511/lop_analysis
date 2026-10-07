#!/usr/bin/env bash
# Registered main run of controls_cifar5p1_1008 (spec §8).  One GPU process; the runner takes the
# shared lock /tmp/lop_analysis_gpu.lock, waits for RAM/GPU room, and refuses unverified or
# uncommitted code.  Detached with setsid so a harness restart does not take it down.
set -euo pipefail
cd "$(dirname "$0")/../.."
run=controls_cifar5p1_1008
python_bin=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
mkdir -p "results/$run/logs"
date '+launch %F %T %Z' >> "results/$run/logs/${run}_main.log"
exec "$python_bin" -u -m "src.$run" --out "results/$run" >> "results/$run/logs/${run}_main.log" 2>&1

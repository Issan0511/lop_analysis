#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run=cap_cifar_ee_0920
python_bin=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
mkdir -p "results/$run/logs"
exec "$python_bin" -u "src/$run.py" --out "results/$run" > "results/$run/logs/${run}.log" 2>&1

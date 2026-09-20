#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run=drive_cifar_c_0920
python_bin=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
mkdir -p "results/$run/logs"
exec >"results/$run/logs/${run}.log" 2>&1
trap 'code=$?; printf "%s\n" "$code" > "results/$run/logs/exit_code"' EXIT
"$python_bin" -u -m src.drive_cifar_c_0920 --mode production --production-go \
  --checks "results/$run/checks.json" --epochs 400 --tasks 5 --out "results/$run/run" "$@"

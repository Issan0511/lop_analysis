#!/usr/bin/env bash
# drive_cifar5p1_1007 production run (spec §10).  Seeds 0-9, 30 tasks + fresh control, observation on.
# Run from anywhere after the implementation commit; refuses without a passing, current checks.json.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
RES=results/drive_cifar5p1_1007
mkdir -p "$RES/raw"
"$PY" -m src.drive_cifar5p1_1007 production --out "$RES/raw/production" \
    --checks "$RES/checks.json" --production-go </dev/null >"$RES/raw/production.log" 2>&1
tail -n 3 "$RES/raw/production.log"

#!/usr/bin/env bash
# Registered main run of mlp_center_x10_1020 (spec §10): R, CH, CHB0 at 7800 x 7800 updates, seeds 0-9, std,
# lr 1e-4, one process per arm, then the verdict.  Launch detached:
#   cd <worktree> && setsid nohup bash analysis/mlp_center_x10_1020/production.sh \
#       > results/mlp_center_x10_1020/production.log 2>&1 < /dev/null &
# The first non-zero exit code (or 0) is written to results/mlp_center_x10_1020/exit_code.txt at the end.
set -u
run_root="$(cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$run_root"
runtime=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
res=results/mlp_center_x10_1020
mkdir -p "$res"
rm -f "$res/exit_code.txt"
export PYTHONUNBUFFERED=1
code=0
echo "=== start $(date '+%F %T') git $(git rev-parse HEAD) pid $$"
for arm in R CH CHB0; do
  echo "=== $arm $(date '+%F %T')"
  "$runtime" src/mlp_center_x10_1020.py run --arm "$arm"
  rc=$?
  echo "=== $arm exit $rc $(date '+%F %T')"
  if [ "$rc" -ne 0 ] && [ "$code" -eq 0 ]; then code=$rc; fi
done
if [ "$code" -eq 0 ]; then
  echo "=== verdict $(date '+%F %T')"
  "$runtime" analysis/mlp_center_x10_1020/verdict.py --root "$res"
  rc=$?
  echo "=== verdict exit $rc $(date '+%F %T')"
  if [ "$rc" -ne 0 ]; then code=$rc; fi
fi
echo "$code" > "$res/exit_code.txt"
echo "=== end $(date '+%F %T') exit $code"
exit "$code"

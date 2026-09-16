#!/bin/bash
cd /home/issan/Projects/claude/wt/l2_wall_0916
for part in chimera ext150 variant; do
  echo "START $part $(date '+%F %T')"
  python3 -m src.l2_wall_0916 --part $part > results/l2_wall_0916/logs/run_$part.log 2>&1
  echo "END $part rc=$? $(date '+%F %T')"
done
echo "ALL_DONE $(date '+%F %T')"

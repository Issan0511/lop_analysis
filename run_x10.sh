#!/bin/bash
# cifar5p1_mlp_0920 addendum 5: 10x the updates per task (spec section 17.1)
set -u
PY=/home/i_nakatsuka/Projects/lop_analysis/.venv/bin/python
cd "$(dirname "$0")"
LOG=results/cifar5p1_mlp_0920/_log
go () { local arm=$1 sh=$2 se=$3
  local tag="${arm}_std_x10_${sh}x${se}"
  echo "=== $tag $(date +%T) ==="
  $PY src/cifar5p1_mlp_0920.py run --arm "$arm" --seeds 0-9 --cond std --lr 1e-4 \
      --threads 2 --steps-hard "$sh" --steps-easy "$se" \
      --out "results/cifar5p1_mlp_0920/$tag" > "$LOG/$tag.log" 2>&1 || echo "FAILED $tag"
  tail -2 "$LOG/$tag.log"; }
go CH   7800 7800     # (a) uniform x10
go KKT1 7800 7800
go R    7800 7800
go CH   7800 780      # (b) hard only
go KKT1 7800 780
echo "=== done $(date +%T) ==="

#!/bin/bash
# cifar5p1_mlp_0920 addendum 3: the held-out seeds 10-19 (spec section 15)
set -u
PY=/home/i_nakatsuka/Projects/lop_analysis/.venv/bin/python
cd "$(dirname "$0")"
LOG=results/cifar5p1_mlp_0920/_log
go () { local arm=$1; shift
  local tag="${arm}_s10-19$(echo "$*" | tr -d ' -' | sed 's/iv//')"
  echo "=== $tag $(date +%T) ==="
  $PY src/cifar5p1_mlp_0920.py run --arm "$arm" --seeds 10-19 --cond std --lr 1e-4 \
      --threads 2 "$@" --out "results/cifar5p1_mlp_0920/$tag" > "$LOG/$tag.log" 2>&1 \
      || echo "FAILED $tag"
  tail -2 "$LOG/$tag.log"; }
go SNA
go KKA
go KKT1
go R
go R --iv l2init:1e-3
go R --iv l2init:1e-2
echo "=== done $(date +%T) ==="

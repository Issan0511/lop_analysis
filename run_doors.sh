#!/bin/bash
# cifar5p1_mlp_0920 addendum 4: the relu_doors arms (spec section 16.2)
set -u
PY=/home/i_nakatsuka/Projects/lop_analysis/.venv/bin/python
cd "$(dirname "$0")"
LOG=results/cifar5p1_mlp_0920/_log
go () { local arm=$1 cond=$2; shift 2
  local tag="${arm}_${cond}_doors"
  echo "=== $tag $(date +%T) ==="
  $PY src/cifar5p1_mlp_0920.py run --arm "$arm" --seeds 0-9 --cond "$cond" --lr 1e-4 \
      --threads 2 "$@" --out "results/cifar5p1_mlp_0920/$tag" > "$LOG/$tag.log" 2>&1 \
      || echo "FAILED $tag"
  tail -2 "$LOG/$tag.log"; }
for cond in std raw; do
  lam=0.8353; [ "$cond" = raw ] && lam=0.8168      # spec 16.1(c): derived per condition
  go H    "$cond"
  go C    "$cond"
  go CH   "$cond"
  go CHB  "$cond" --lam "$lam"
  go CHB0 "$cond"
done
go R raw                                            # the raw-side reference
echo "=== done $(date +%T) ==="

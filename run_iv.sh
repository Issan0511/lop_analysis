#!/bin/bash
# cifar5p1_mlp_0920 addendum 2: R + L2 Init (spec section 13.2)
set -u
PY=/home/i_nakatsuka/Projects/lop_analysis/.venv/bin/python
cd "$(dirname "$0")"
LOG=results/cifar5p1_mlp_0920/_log
run () { echo "=== $* $(date +%T) ==="; $PY src/cifar5p1_mlp_0920.py run --arm R --seeds 0-9 \
         --cond std --threads 2 "$@" > "$LOG/iv_$(echo "$*" | tr -d ' -' ).log" 2>&1 \
         || echo "FAILED $*"; tail -2 "$LOG/iv_$(echo "$*" | tr -d ' -').log"; }
run --lr 1e-4 --iv l2init:1e-2      # primary: Kumar's lambda, the battle's lr
run --lr 1e-4 --iv l2init:1e-4
run --lr 1e-4 --iv l2init:1e-3
run --lr 1e-4 --iv l2init:1e-1
run --lr 1e-3 --iv l2init:1e-2      # Kumar's own configuration
run --lr 1e-3                       # its matched baseline
echo "=== done $(date +%T) ==="

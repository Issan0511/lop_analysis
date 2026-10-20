#!/bin/bash
# 1 行 = arm state adam task out kind light
set -- $1
[ -f "$5" ] && exit 0
L=""; [ "$7" = "1" ] && L="--light"
/home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/chargpt_cl_1019_quicklook/rho_split.py --arm $1 --state $2 --adam $3 --task $4 --kind $6 --nseq 32 --threads 4 $L --out $5 > ${5%.json}.log 2>&1

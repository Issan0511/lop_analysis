#!/usr/bin/env python3
"""spec 2.3: compare the 4 full-length verification shards with the committed wcap shards; writes reuse.json.

    python3 analysis/l2split_rlmnist_0914/reuse.py      # exit 0 = REUSE_OK, 3 = REUSE_MISMATCH

The comparison itself lives in verdict.compare_reuse (so check S7 exercises it with its mutations).
"""
import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--reuse-check", *sys.argv[1:]]
runpy.run_path(str(Path(__file__).resolve().parent / "verdict.py"), run_name="__main__")

#!/usr/bin/env python3
"""Run unchanged host checks against independently re-executed historical code."""
import argparse
import runpy
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
args, rest = p.parse_known_args()
sys.path.insert(0, str(args.repo))
module = runpy.run_path(str(args.repo / 'src/cifar5p1_mlp_0920_checks.py'))
# Only the reference artifact location changes. Training code, comparator,
# required mutants and bit/byte equality conditions remain unmodified.
module['C'].OUT_ROOT = args.reference
sys.argv = ['cifar5p1_mlp_0920_checks.py', *rest]
module['main']()

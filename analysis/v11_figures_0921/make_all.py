#!/usr/bin/env python3
"""V11 の図を全部作り直す。committed CSV だけを読み、走は起こさない。

    python3 analysis/v11_figures_0921/make_all.py
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for f in sorted(HERE.glob("fig*.py")):
    print(f"--- {f.name}")
    runpy.run_path(str(f), run_name="__main__")

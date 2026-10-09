#!/usr/bin/env python3
"""Seed-mean tables of an epoch-resolved fork trace (results/sna_cnn_cause_1009/T<at>/trace.csv).

    python3 analysis/sna_cnn_cause_1009/trace_table.py T10 T1
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "results" / "sna_cnn_cause_1009"
COLS = ["acc", "acc_oldconv", "ce", "gate_f1", "gate_f2", "seat_f1", "seat_f2", "feat_drift",
        "move_c2", "move_f1", "move_f2", "move_f3"]


def main() -> None:
    for name in sys.argv[1:] or ["T10", "T1"]:
        f = ROOT / name / "trace.csv"
        if not f.exists():
            print(f"{name}: no trace yet")
            continue
        d = pd.read_csv(f)
        print(f"\n## {name}")
        for fork, g in d.groupby("fork", sort=False):
            m = g.groupby("epoch")[COLS].mean()
            print(f"\n### {fork}\n")
            print("| epoch | " + " | ".join(COLS) + " |")
            print("|---|" + "---|" * len(COLS))
            for e, row in m.iterrows():
                print(f"| {e} | " + " | ".join(f"{row[c]:.3f}" for c in COLS) + " |")


if __name__ == "__main__":
    main()

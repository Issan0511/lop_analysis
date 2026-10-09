#!/usr/bin/env python3
"""How the slowdown shows inside a task: epoch-wise training accuracy (pre-update, the online
measure) per arm, early tasks against late tasks.

    python3 analysis/sna_cnn_cause_1009/curves.py --bundle A

Per (arm, task group): mean accuracy at epochs 1, 2, 3, 5, 10, 20, 50; epochs to reach 0.5 /
0.9 / 0.99; and the share of the task's lost online accuracy (1 - online) that falls in the
first 10 / 50 epochs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results" / "sna_cnn_cause_1009"


def first_reach(c: np.ndarray, thr: float) -> float:
    i = np.nonzero(c >= thr)[0]
    return float(i[0] + 1) if i.size else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="A")
    ap.add_argument("--groups", default="1-3,8-12,28-32,46-50")
    args = ap.parse_args()
    d = ROOT / args.bundle
    curves = np.load(d / "curves.npy")                      # (R, tasks, epochs)
    st = json.loads((d / "provenance.json").read_text()) if (d / "provenance.json").exists() else None
    import torch
    meta = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
    slots = meta["meta"]["slots"]
    T = meta["t"]
    arms = list(dict.fromkeys(a for a, s in slots))
    groups = []
    for g in args.groups.split(","):
        a, b = (int(x) for x in g.split("-"))
        if b <= T:
            groups.append((a, b))
    E = (1, 2, 3, 5, 10, 20, 50)
    print(f"tasks completed: {T}")
    hdr = "| arm | tasks | " + " | ".join(f"ep{e}" for e in E) + " | →0.5 | →0.9 | →0.99 | loss≤10ep | loss≤50ep | 1−online |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for arm in arms:
        rs = [i for i, (a, s) in enumerate(slots) if a == arm]
        for a, b in groups:
            c = curves[rs, a - 1:b]                           # (n, k, epochs)
            m = c.mean((0, 1))
            reach = {thr: np.nanmean([first_reach(c[i, j], thr) for i in range(c.shape[0])
                                      for j in range(c.shape[1])]) for thr in (0.5, 0.9, 0.99)}
            lost = (1 - c)                                     # per epoch lost accuracy
            tot = lost.sum(-1)
            s10 = (lost[..., :10].sum(-1) / tot).mean()
            s50 = (lost[..., :50].sum(-1) / tot).mean()
            print(f"| {arm} | {a}–{b} | " + " | ".join(f"{m[e - 1]:.3f}" for e in E)
                  + f" | {reach[0.5]:.1f} | {reach[0.9]:.1f} | {reach[0.99]:.1f} | {s10:.2f} | {s50:.2f} | "
                  f"{(1 - c.mean()):.4f} |")


if __name__ == "__main__":
    main()

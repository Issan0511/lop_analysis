#!/usr/bin/env python3
"""How much of a conv channel's pooled variance is the spread of its position means (Q3).

    python3 analysis/sna_cnn_cause_1009/varsplit.py --bundle A --arms SNA,SNAc3 --tasks 1,10

For each checkpoint: per channel of c1 and c2, pooled = within + between (see posthoc_lib),
and what the within-position W would make of alpha (sqrt(pooled / within)).  Medians over
channels, then mean and range over seeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_lib as L  # noqa: E402

ROOT = L.ROOT / "results" / "sna_cnn_cause_1009"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="A")
    ap.add_argument("--arms", default="SNA")
    ap.add_argument("--tasks", default="1")
    ap.add_argument("--seeds", default="10-19")
    args = ap.parse_args()
    L.H.setup("cuda")
    seeds = L.E.parse_seeds(args.seeds)
    for arm in args.arms.split(","):
        for t in [int(x) for x in args.tasks.split(",")]:
            acc = {"c1": [], "c2": []}
            for s in seeds:
                f = ROOT / args.bundle / "ckpt" / f"{arm}_seed{s}_t{t:02d}.pt"
                if not f.exists():
                    continue
                B, _ = L.load_run(f)
                sp = L.conv_variance_split(B)
                for tag in ("c1", "c2"):
                    d = sp[tag]
                    share = (d["between"] / d["pooled"]).median().item()
                    ratio = (d["pooled"] / d["within"]).sqrt().median().item()
                    acc[tag].append((share, ratio))
                del B
                torch.cuda.empty_cache()
            for tag in ("c1", "c2"):
                if acc[tag]:
                    a = np.array(acc[tag])
                    print(f"{arm:8s} t{t:02d} {tag}: between/pooled median-over-channels = "
                          f"{a[:, 0].mean():.3f} [{a[:, 0].min():.3f}, {a[:, 0].max():.3f}]   "
                          f"sqrt(pooled/within) = {a[:, 1].mean():.3f} [{a[:, 1].min():.3f}, {a[:, 1].max():.3f}]"
                          f"   n={len(a)}")


if __name__ == "__main__":
    main()

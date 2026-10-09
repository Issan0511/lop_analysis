#!/usr/bin/env python3
"""Unit-level gates of the fc sites (and conv channels) from checkpoints.

    python3 analysis/sna_cnn_cause_1009/gates.py --bundle A --arms SNA,CV3FC06,CV06FC3 --tasks 1,5,10

Per site, over the 1200 images of the checkpoint's task:
  g_unit        mean gate phi'(z) of each unit (conv: channel, over images and positions)
  closed_units  share of units whose mean gate < 0.3
  open_per_img  per image, the share of units with gate > 0.5, averaged over images
  seat          2*alpha*mean(z) per unit (median), spread_rad = 2*alpha*sd(z) (median)
  back_gain     per image, mean over units of phi'(z)^2 (how much of a unit-space gradient
                survives the site on the way back), averaged over images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_lib as L  # noqa: E402

E, CN = L.E, L.CN
ROOT = L.ROOT / "results" / "sna_cnn_cause_1009"


@torch.no_grad()
def site_gates(B) -> dict:
    zs = {l: [] for l in range(4)}
    for i0 in range(0, CN.N_IMAGES, 100):
        o = E.forward(B.P, B.X[:, i0:i0 + 100], B.act)
        for l in range(4):
            z = o[2 * l]
            zs[l].append(z if l >= 2 else z.float())
    out = {}
    for l, tag in enumerate(CN.SITES):
        z = torch.cat(zs[l], 0 if l < 2 else 1)
        if l < 2:                                    # (N, C, H, W) -> (N*H*W, C) with per-image view
            N, C = z.shape[0], z.shape[1]
            d = B.act.dphi(z, l)
            g_unit = d.mean((0, 2, 3))
            zz = z.permute(1, 0, 2, 3).reshape(C, -1)
            mean, sd = zz.mean(1), zz.std(1)
            per_img_open = (d.mean((2, 3)) > 0.5).float().mean(1)      # (N,)
            back = (d ** 2).mean((1, 2, 3))
        else:
            z = z[0]                                 # (N, n)
            d = B.act.dphi(z[None], l)[0]
            g_unit = d.mean(0)
            mean, sd = z.mean(0), z.std(0)
            per_img_open = (d > 0.5).float().mean(1)
            back = (d ** 2).mean(1)
        a = B.act.alpha(l)[0]
        out[tag] = {"closed_units": float((g_unit < 0.3).float().mean()),
                    "g_unit_med": float(g_unit.median()),
                    "open_per_img": float(per_img_open.mean()),
                    "seat": float((2 * a * mean).median()),
                    "spread_rad": float((2 * a * sd).median()),
                    "back_gain": float(back.mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="A")
    ap.add_argument("--arms", default="SNA,CV3FC06,CV06FC3,SNAc3")
    ap.add_argument("--tasks", default="1,5,10")
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--sites", default="f1,f2")
    args = ap.parse_args()
    L.H.setup("cuda")
    sites = args.sites.split(",")
    keys = ("closed_units", "g_unit_med", "open_per_img", "seat", "spread_rad", "back_gain")
    print("| arm | t | " + " | ".join(f"{s}:{k}" for s in sites for k in keys) + " |")
    for arm in args.arms.split(","):
        for t in [int(x) for x in args.tasks.split(",")]:
            acc = []
            for s in L.E.parse_seeds(args.seeds):
                f = ROOT / args.bundle / "ckpt" / f"{arm}_seed{s}_t{t:02d}.pt"
                if not f.exists():
                    continue
                B, _ = L.load_run(f)
                acc.append(site_gates(B))
                del B
            if not acc:
                continue
            row = [np.mean([a[s][k] for a in acc]) for s in sites for k in keys]
            print(f"| {arm} | {t} | " + " | ".join(f"{v:.3f}" for v in row) + f" |  n={len(acc)}")


if __name__ == "__main__":
    main()

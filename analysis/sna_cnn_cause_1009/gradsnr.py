#!/usr/bin/env python3
"""How consistent the gradient of each layer is across minibatches at the start of a new task.

    python3 analysis/sna_cnn_cause_1009/gradsnr.py --bundle A --arms SNA,CV3FC06,CV06FC3 --tasks 1,10

At a checkpoint (end of task t), with the labels of task t+1 (the next labelling, never seen),
K = 64 minibatches of 16 drawn without replacement: per weight tensor,
  coh = |mean_k g_k|^2 / mean_k |g_k|^2      (1/K for pure noise, 1 for a fixed direction)
and the per-coordinate sign agreement of Adam's first step direction,
  sgn = mean over coordinates of |mean_k sign(g_k)|.
Read only; nothing is trained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_lib as L  # noqa: E402

E, CN = L.E, L.CN
ROOT = L.ROOT / "results" / "sna_cnn_cause_1009"
TAGS = CN.WEIGHT_TAGS


def coherence(B, y_next: torch.Tensor, K: int = 64, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(CN.N_IMAGES, generator=g)[:K * CN.BATCH].view(K, CN.BATCH)
    sums = [torch.zeros_like(B.P[2 * i][0]) for i in range(5)]
    sq = [0.0] * 5
    sg = [torch.zeros_like(B.P[2 * i][0]) for i in range(5)]
    for k in range(K):
        idx = order[k].to(B.X.device)
        out = E.forward(B.P, B.X[:, idx], B.act)
        loss = F.cross_entropy(out[8][0], y_next[idx])
        gr = torch.autograd.grad(loss, [B.P[2 * i] for i in range(5)])
        for i in range(5):
            gi = gr[i][0]
            sums[i] += gi
            sq[i] += float((gi ** 2).sum())
            sg[i] += torch.sign(gi)
    out = {}
    for i, tag in enumerate(TAGS):
        out[f"coh_{tag}"] = float((sums[i] / K).pow(2).sum() / (sq[i] / K))
        out[f"sgn_{tag}"] = float((sg[i] / K).abs().mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="A")
    ap.add_argument("--arms", default="SNA,CV3FC06,CV06FC3")
    ap.add_argument("--tasks", default="1,10")
    ap.add_argument("--seeds", default="10-19")
    args = ap.parse_args()
    L.H.setup("cuda")
    print("| arm | t | " + " | ".join(f"coh {t}" for t in TAGS) + " | " + " | ".join(f"sgn {t}" for t in TAGS) + " |")
    for arm in args.arms.split(","):
        for t in [int(x) for x in args.tasks.split(",")]:
            res = []
            for s in L.E.parse_seeds(args.seeds):
                f = ROOT / args.bundle / "ckpt" / f"{arm}_seed{s}_t{t:02d}.pt"
                if not f.exists():
                    continue
                B, _ = L.load_run(f)
                y_next = L.labels_at(s, t + 1).to(B.X.device)
                res.append(coherence(B, y_next))
                del B
            if not res:
                continue
            m = {k: np.mean([q[k] for q in res]) for k in res[0]}
            print(f"| {arm} | {t} | " + " | ".join(f"{m[f'coh_{x}']:.3f}" for x in TAGS) + " | "
                  + " | ".join(f"{m[f'sgn_{x}']:.3f}" for x in TAGS) + f" |  n={len(res)}")


if __name__ == "__main__":
    main()

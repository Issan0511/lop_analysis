#!/usr/bin/env python3
"""Checkpoint probes for sna_cnn_cause_1009: where the capacity to fit a fresh labelling lives.

    python3 analysis/sna_cnn_cause_1009/probe.py --bundle A --arms SNA,SNAc3 --tasks 1,10,30,50

Per checkpoint (one run, read only):
  rank_<site>   effective rank (exp entropy of singular values) of the site's activation over the
                1200 images, flattened per image (conv: C*H*W columns), centred
  lin_<feat>    training accuracy of a ridge-regression readout (one-hot targets, lambda = 1e-3 of
                the mean eigenvalue) on frozen features, for 3 fresh uniform labellings of the
                1200 images: h = pooled c2 output (1024), a3 = f1 output (100), a4 = f2 output (100)
  ntk_<group>   share of the empirical NTK trace (sum over 256 images of the squared norm of
                d(logit of a fresh label)/d(params of the group)) carried by conv / fc1 / fc2 / fc3
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


def eff_rank_rows(A: torch.Tensor) -> float:
    A = A.double()
    A = A - A.mean(0, keepdim=True)
    s = torch.linalg.svdvals(A)
    p = s / s.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum()))


def ridge_acc(Fm: torch.Tensor, ys: list[torch.Tensor], lam_rel: float = 1e-3) -> float:
    Fm = Fm.double()
    Fm = torch.cat([Fm - Fm.mean(0, keepdim=True), torch.ones(Fm.shape[0], 1, dtype=Fm.dtype,
                                                                device=Fm.device)], 1)
    G = Fm.T @ Fm
    lam = lam_rel * torch.diagonal(G).mean()
    Ginv = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device),
                              torch.eye(G.shape[0], dtype=G.dtype, device=G.device))
    accs = []
    for y in ys:
        Y = F.one_hot(y, 10).double()
        Wr = Ginv @ (Fm.T @ Y)
        accs.append(float(((Fm @ Wr).argmax(1) == y).double().mean()))
    return float(np.mean(accs))


@torch.no_grad()
def features(B) -> dict:
    out = {"h": [], "a1": [], "a2": [], "a3": [], "a4": []}
    for i0 in range(0, CN.N_IMAGES, 100):
        o = E.forward(B.P, B.X[:, i0:i0 + 100], B.act)
        a1, a2, a3, a4 = o[1], o[3], o[5], o[7]
        b = a1.shape[0]
        out["a1"].append(F.max_pool2d(a1, 2, 2).reshape(b, -1).cpu())
        out["a2"].append(a2.reshape(b, -1).cpu())
        out["h"].append(F.max_pool2d(a2, 2, 2).reshape(b, -1).cpu())
        out["a3"].append(a3[0].cpu())
        out["a4"].append(a4[0].cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def ntk_shares(B, n: int = 256, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(CN.N_IMAGES, generator=g)[:n]
    y = torch.randint(10, (n,), generator=g)
    groups = {"conv": [0, 1, 2, 3], "fc1": [4, 5], "fc2": [6, 7], "fc3": [8, 9]}
    tot = {k: 0.0 for k in groups}
    for i in range(0, n, 32):
        xb = B.X[:, idx[i:i + 32].to(B.X.device)]
        yb = y[i:i + 32].to(B.X.device)
        for j in range(xb.shape[1]):
            out = E.forward(B.P, xb[:, j:j + 1], B.act)
            lg = out[8][0, 0]
            # gradient of the log-probability of a fresh label: the direction the next task pulls
            lp = F.log_softmax(lg, -1)[yb[j]]
            gr = torch.autograd.grad(lp, B.P)
            for k, ids in groups.items():
                tot[k] += float(sum((gr[q] ** 2).sum() for q in ids))
    s = sum(tot.values())
    return {f"ntk_{k}": v / s for k, v in tot.items()} | {"ntk_total": s / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="A")
    ap.add_argument("--arms", default="SNA,SNAc3")
    ap.add_argument("--tasks", default="1,10,30,50")
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    L.H.setup("cuda")
    rows = []
    gy = torch.Generator().manual_seed(12345)
    ys = [torch.randint(10, (CN.N_IMAGES,), generator=gy) for _ in range(3)]
    for arm in args.arms.split(","):
        for t in [int(x) for x in args.tasks.split(",")]:
            for s in L.E.parse_seeds(args.seeds):
                f = ROOT / args.bundle / "ckpt" / f"{arm}_seed{s}_t{t:02d}.pt"
                if not f.exists():
                    continue
                B, _ = L.load_run(f)
                ft = features(B)
                r = {"arm": arm, "seed": s, "task": t}
                for k in ("a1", "a2", "h", "a3", "a4"):
                    r[f"rank_{k}"] = eff_rank_rows(ft[k].cuda())
                for k in ("h", "a3", "a4"):
                    r[f"lin_{k}"] = ridge_acc(ft[k].cuda(), [y.cuda() for y in ys])
                r.update(ntk_shares(B))
                rows.append(r)
                del B
                torch.cuda.empty_cache()
    import pandas as pd
    d = pd.DataFrame(rows)
    if args.out:
        d.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(d.groupby(["arm", "task"]).mean(numeric_only=True).drop(columns="seed").round(4).to_string())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""quiet_drift_cifar_0923 pass 4 -- does the full-batch gradient turn over within the momentum's window?

Replays a bundle exactly as pass 1/2 (the training loop is the engine's op sequence; the probes
only read) and, every 1,000 steps from 6,000 to 22,000, computes the float64 full-batch gradient
of the mean CE w.r.t. W1 at 21 consecutive states t0 ... t0+20.  Records per slot and burst:
cos(G_t0, G_t0+k) for k = 1..20, ||G_t0+k|| / ||G_t0||, cos(G_t, m_t) at each state, and the
overlap of the 50 largest-residual images between t0 and t0+k.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from src import rlcifar_mlp_battle_0918 as B          # noqa: E402
from src import pmnist_0905 as H                      # noqa: E402
from src import pmnist_rlcifar_0907 as RC             # noqa: E402
import replay_probe as RP                             # noqa: E402

BURSTS = list(range(6000, 22001, 1000))
LAGS = 20
TOPN = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=("iid", "abab"), required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--t", type=int, default=48)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = H.setup(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = RP.SRC_ROOT / RP.NET_DIR[a.net] / "ckpts" / f"t{a.t:02d}.pt"
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    SEEDS = RP.SEEDS
    R = len(SEEDS)
    lab = RP.labels_for(a.net, a.branch, a.t, st)
    cifar = RC.Cifar10()
    X = torch.stack([B.slot_inputs(cifar, s, "std", dev) for s in SEEDS])
    Y = torch.stack([lab[s] for s in SEEDS]).to(dev)
    X64 = X.double()
    act = B.make_act("LR")
    P = [torch.stack([st["P"][i][r] for r in range(R)]).to(dev).contiguous().requires_grad_(True)
         for i in range(6)]
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    with torch.no_grad():
        for dst, src in ((adam_m, st["m"]), (adam_v, st["v"])):
            for q, v in zip(dst, src):
                q.copy_(v.to(dev))
    tc = int(st["tc"])
    g_batch = {}
    for s in SEEDS:
        g_batch[s] = torch.Generator(device="cpu")
        g_batch[s].set_state(st["g_batch"][s])
    lr, b1, b2, eps = B.LR, 0.9, 0.999, 1e-8
    BATCH, N, NC = B.BATCH, B.N_IMAGES, B.N_CLASSES
    ar = torch.arange(R, device=dev)[:, None]
    static_idx = torch.zeros(R, BATCH, dtype=torch.long, device=dev)
    inv_c1 = torch.zeros((), device=dev)
    inv_c2 = torch.zeros((), device=dev)

    def full_grad():
        """float64 full-batch gradient of the mean CE w.r.t. W1, and the residuals."""
        P64 = [q.detach().double() for q in P]
        W1 = P64[0].requires_grad_(True)
        z1 = torch.baddbmm(P64[1][:, None, :], X64, W1.transpose(1, 2))
        a1 = torch.where(z1 > 0, z1, 0.1 * z1)
        z2 = torch.baddbmm(P64[3][:, None, :], a1, P64[2].transpose(1, 2))
        a2 = torch.where(z2 > 0, z2, 0.1 * z2)
        z3 = torch.baddbmm(P64[5][:, None, :], a2, P64[4].transpose(1, 2))
        ce = F.cross_entropy(z3.reshape(-1, NC), Y.reshape(-1), reduction="none").view(R, N)
        (g,) = torch.autograd.grad(ce.sum() / N, [W1])
        with torch.no_grad():
            zd = z3.detach()
            dz = zd - zd.gather(2, Y[:, :, None])
            e = torch.exp(dz)
            e.scatter_(2, Y[:, :, None], 0.0)
            se = e.sum(2)
            resid = se / (1.0 + se)
        return g.detach(), resid

    res = {k: [] for k in ("t0", "cosG", "normratio", "cosGm", "top_overlap", "L")}
    Gs, tops = [], []

    def burst_probe(ts: int, t0: int) -> None:
        g, resid = full_grad()
        mh = (adam_m[0] * inv_c1).double()
        with torch.no_grad():
            gn = g.pow(2).sum((1, 2)).sqrt()
            cgm = (g * mh).sum((1, 2)) / (gn * mh.pow(2).sum((1, 2)).sqrt() + 1e-300)
            top = torch.topk(resid, TOPN, dim=1).indices
        if ts == t0:
            Gs.clear(); tops.clear()
        Gs.append(g); tops.append(top)
        res.setdefault("_cgm_buf", []).append(cgm.cpu().numpy())
        res.setdefault("_L_buf", []).append((resid.sum(1) / N).cpu().numpy())
        if ts == t0 + LAGS:
            G0, n0 = Gs[0], Gs[0].pow(2).sum((1, 2)).sqrt()
            cos, nr, ov = [], [], []
            t0set = [set(tops[0][r].tolist()) for r in range(R)]
            for k in range(1, LAGS + 1):
                Gk = Gs[k]
                nk = Gk.pow(2).sum((1, 2)).sqrt()
                cos.append(((G0 * Gk).sum((1, 2)) / (n0 * nk + 1e-300)).cpu().numpy())
                nr.append((nk / n0).cpu().numpy())
                ov.append(np.array([len(t0set[r] & set(tops[k][r].tolist())) / TOPN for r in range(R)]))
            res["t0"].append(t0)
            res["cosG"].append(np.stack(cos))          # (LAGS, R)
            res["normratio"].append(np.stack(nr))
            res["top_overlap"].append(np.stack(ov))
            res["cosGm"].append(np.stack(res.pop("_cgm_buf")))   # (LAGS+1, R)
            res["L"].append(np.stack(res.pop("_L_buf")))

    def step() -> None:
        xb, yb = X[ar, static_idx], Y[ar, static_idx]
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, NC), yb.reshape(-1),
                                reduction="none").view(R, BATCH).mean(1)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            for p, gr, mi, vi in zip(P, grads, adam_m, adam_v):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))

    t_start = time.time()
    ts, S = 0, BURSTS[-1] + LAGS
    spe = B.STEPS_PER_EPOCH
    cur = None
    for e in range(-(-S // spe)):
        order = {s: torch.randperm(N, generator=g_batch[s]) for s in SEEDS}
        ORD = torch.stack([order[s] for s in SEEDS]).to(dev)
        for j in range(spe):
            if ts >= S:
                break
            static_idx.copy_(ORD[:, j * BATCH:(j + 1) * BATCH])
            tc += 1
            ts += 1
            inv_c1.fill_(1.0 / (1 - b1 ** tc))
            inv_c2.fill_(1.0 / (1 - b2 ** tc))
            act.begin_step(R, BATCH, dev)
            step()
            if ts in BURSTS:
                cur = ts
            if cur is not None and cur <= ts <= cur + LAGS:
                burst_probe(ts, cur)
                if ts == cur + LAGS:
                    cur = None
    np.savez_compressed(out / "lag.npz", **{k: np.stack(v) for k, v in res.items()})
    (out / "provenance_lag.json").write_text(json.dumps(
        {"experiment": "quiet_drift_cifar_0923", "pass": 4, "net": a.net, "branch": a.branch,
         "t": a.t, "bursts": BURSTS, "lags": LAGS, "ckpt": str(ckpt),
         "ckpt_sha256": RP.sha256_file(ckpt), "git": B.git_state(),
         "seconds": time.time() - t_start}, indent=1, default=str))
    print(f"done {a.net}/{a.branch} {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()

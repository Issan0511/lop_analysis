#!/usr/bin/env python3
"""quiet_drift_cifar_0923 pass 2 -- why the iid net needs bigger Adam steps after the fit.

Pass 1 (replay_probe.py) found: in the quiet phase both nets' loss decays at Adam's second-moment
memory rate (sqrt(v) e-folds every 2,000 steps, the loss every ~1,800), the active images'
margins rise at the same ~0.54 per 1,000 steps, and the iid net takes ~8x larger W1 steps to do
it.  First order, the log-loss falls per step by

    sum_l <G_l, u_l> / L  =  sum_l (||G_l|| / L) * ||u_l|| * cos(G_l, u_l)

(G_l = full-batch gradient of the mean CE w.r.t. layer l, u_l = the Adam update the engine
subtracts).  With that sum pinned, the step ||u_l|| is large when the loss gradient per unit loss
||G_l||/L is small or when Adam's direction is poorly aligned with it.  ||G_l|| itself is the
coherent sum of the per-image gradients: ||sum_n g_n|| = C_img * sum_n ||g_n||.

This replays the same bundles as pass 1 (bit for bit: the training loop is the same op sequence;
nothing here touches the training tensors) and every 100 steps computes, per slot and per layer
(W1, b1, W2, b2, W3, b3), in float64 from float64 copies of the parameters:
  L (mean CE), sum_n r_n (r_n = 1 - p_y), ||G_l||, ||u_l||, <G_l, u_l>, cos(G_l, W_l), cos(u_l, W_l),
  and for W1/W2/W3 the per-image gradient norms ||g_n|| = ||delta_n|| * ||input_n|| (outer
  products), their sum, their residual-weighted radial projections <g_n, W>, and C_img.
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
import replay_probe as RP                             # noqa: E402  labels_for, SRC_ROOT, ...

PROBE = 100
LAYERS = ("W1", "b1", "W2", "b2", "W3", "b3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=("iid", "abab"), required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--t", type=int, default=48)
    ap.add_argument("--steps", type=int, default=30000)
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
    xnorm = X64.pow(2).sum(2).sqrt()                                        # (R, N)
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
    rows: dict[str, list] = {}

    def rec(k, v):
        rows.setdefault(k, []).append(v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))

    def probe(ts: int) -> None:
        with torch.no_grad():
            correct = (B.forward(P, X, act, train=False)[4].argmax(-1) == Y).sum(1)
        P64 = [q.detach().double().requires_grad_(True) for q in P]
        W1, bb1, W2, bb2, W3, bb3 = P64
        z1 = torch.baddbmm(bb1[:, None, :], X64, W1.transpose(1, 2))
        a1 = torch.where(z1 > 0, z1, 0.1 * z1)
        z2 = torch.baddbmm(bb2[:, None, :], a1, W2.transpose(1, 2))
        a2 = torch.where(z2 > 0, z2, 0.1 * z2)
        z3 = torch.baddbmm(bb3[:, None, :], a2, W3.transpose(1, 2))
        ce = F.cross_entropy(z3.reshape(-1, NC), Y.reshape(-1), reduction="none").view(R, N)
        Lsum = ce.sum(1)                                                     # (R,)
        gz = torch.autograd.grad(Lsum.sum(), [z1, z2, z3, *P64])
        d1, d2, d3 = gz[0], gz[1], gz[2]                                     # per-image deltas
        G = [g / N for g in gz[3:]]                                          # grads of the mean CE
        with torch.no_grad():
            L = Lsum / N
            zd = z3.detach()
            dz = zd - zd.gather(2, Y[:, :, None])
            e = torch.exp(dz)
            e.scatter_(2, Y[:, :, None], 0.0)
            se = e.sum(2)
            resid = se / (1.0 + se)
            rec("step", ts); rec("correct", correct); rec("L", L); rec("resid_sum", resid.sum(1))
            rec("k_eff", resid.sum(1) ** 2 / (resid.pow(2).sum(1) + 1e-300))
            for k, nm in enumerate(LAYERS):
                p = P64[k].detach()
                g = G[k]
                vh = (adam_v[k] * inv_c2 if ts > 0 else adam_v[k]).double()
                mh = (adam_m[k] * inv_c1 if ts > 0 else adam_m[k]).double()
                u = lr * mh / (vh.sqrt() + eps)
                dims = tuple(range(1, p.dim()))
                gn = g.pow(2).sum(dims).sqrt()
                un = u.pow(2).sum(dims).sqrt()
                pn = p.pow(2).sum(dims).sqrt()
                rec(f"{nm}_G", gn); rec(f"{nm}_u", un); rec(f"{nm}_Gu", (g * u).sum(dims))
                rec(f"{nm}_cosGW", (g * p).sum(dims) / (gn * pn + 1e-300))
                rec(f"{nm}_cosuW", (u * p).sum(dims) / (un * pn + 1e-300))
                rec(f"{nm}_norm", pn)
                # pass 2b: where Adam's direction loses the gradient -- the momentum (a noisy
                # EMA of minibatch gradients) or the per-coordinate division by sqrt(v)
                mn = mh.pow(2).sum(dims).sqrt()
                gp = g / (vh.sqrt() + eps)                                   # preconditioned G
                gpn = gp.pow(2).sum(dims).sqrt()
                rec(f"{nm}_cosGm", (g * mh).sum(dims) / (gn * mn + 1e-300))
                rec(f"{nm}_cosGGp", (g * gp).sum(dims) / (gn * gpn + 1e-300))
                rec(f"{nm}_cosmu", (mh * u).sum(dims) / (mn * un + 1e-300))
                rec(f"{nm}_cosGpu", (gp * u).sum(dims) / (gpn * un + 1e-300))
                rec(f"{nm}_m", mn)
            # per-image gradient norms of the weight matrices (outer products) and radial parts
            a1d, a2d = a1.detach(), a2.detach()
            pim = {"W1": d1.pow(2).sum(2).sqrt() * xnorm,
                   "W2": d2.pow(2).sum(2).sqrt() * a1d.pow(2).sum(2).sqrt(),
                   "W3": d3.pow(2).sum(2).sqrt() * a2d.pow(2).sum(2).sqrt()}
            rad = {"W1": (d1 * (z1.detach() - bb1.detach()[:, None, :])).sum(2),   # <delta1 x^T, W1>
                   "W2": (d2 * (z2.detach() - bb2.detach()[:, None, :])).sum(2),
                   "W3": (d3 * (z3.detach() - bb3.detach()[:, None, :])).sum(2)}
            for nm in ("W1", "W2", "W3"):
                k = LAYERS.index(nm)
                ssum = pim[nm].sum(1)                                        # sum_n ||g_n||
                rec(f"{nm}_img_sum", ssum)
                rec(f"{nm}_Cimg", N * G[k].pow(2).sum((1, 2)).sqrt() / (ssum + 1e-300))
                rec(f"{nm}_img_rad", rad[nm].sum(1))                         # = N <G, W>
                rec(f"{nm}_img_radabs", rad[nm].abs().sum(1))
                rec(f"{nm}_img_top10share",
                    torch.sort(pim[nm], 1, descending=True)[0][:, :10].sum(1) / (ssum + 1e-300))

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

    t0 = time.time()
    probe(0)
    ts, S = 0, a.steps
    spe = B.STEPS_PER_EPOCH
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
            if ts % PROBE == 0:
                probe(ts)
        if (e + 1) % 80 == 0:
            print(f"[{time.strftime('%T')}] {a.net}/{a.branch} step {ts} "
                  f"({(time.time() - t0) / ts * 1e3:.2f} ms/step)", flush=True)
    arrs = {k: np.stack(v) for k, v in rows.items()}
    np.savez_compressed(out / "diag.npz", **arrs)
    (out / "provenance_diag.json").write_text(json.dumps(
        {"experiment": "quiet_drift_cifar_0923", "pass": 2, "net": a.net, "branch": a.branch,
         "t": a.t, "steps": S, "ckpt": str(ckpt), "ckpt_sha256": RP.sha256_file(ckpt),
         "git": B.git_state(), "seconds": time.time() - t0}, indent=1, default=str))
    print(f"done {a.net}/{a.branch} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

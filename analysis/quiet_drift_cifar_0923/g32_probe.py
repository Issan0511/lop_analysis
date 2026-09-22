#!/usr/bin/env python3
"""quiet_drift_cifar_0923 pass 5 -- is the gradient training sees (float32) the loss gradient (float64)?

Replays a bundle exactly (the engine's op sequence) and at steps 8,000 / 12,000 / 16,000 / 20,000
computes, read only:
  G64: the float64 full-batch gradient of the mean CE w.r.t. W1 (as in pass 2)
  G32: the same with the training's own float32 forward/backward over all 1200 images at once
       (= the mean of the training's per-image float32 gradients; an epoch's 75 minibatches
       average to this up to summation order)
and cos(G32, G64), ||G32 - G64|| / ||G64||, cos(G64, m), cos(G32, m), plus the count of images
whose float32 p_y is exactly 1.0 and the float64 residual they carry.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np, torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from src import rlcifar_mlp_battle_0918 as B          # noqa: E402
from src import pmnist_0905 as H                      # noqa: E402
from src import pmnist_rlcifar_0907 as RC             # noqa: E402
import replay_probe as RP                             # noqa: E402

AT = (8000, 12000, 16000, 20000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True); ap.add_argument("--branch", required=True)
    ap.add_argument("--t", type=int, default=48); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = H.setup("cuda")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = RP.SRC_ROOT / RP.NET_DIR[a.net] / "ckpts" / f"t{a.t:02d}.pt"
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    SEEDS = RP.SEEDS; R = len(SEEDS)
    lab = RP.labels_for(a.net, a.branch, a.t, st)
    cifar = RC.Cifar10()
    X = torch.stack([B.slot_inputs(cifar, s, "std", dev) for s in SEEDS])
    Y = torch.stack([lab[s] for s in SEEDS]).to(dev)
    X64 = X.double()
    act = B.make_act("LR")
    P = [torch.stack([st["P"][i][r] for r in range(R)]).to(dev).contiguous().requires_grad_(True) for i in range(6)]
    am = [torch.zeros_like(q) for q in P]; av = [torch.zeros_like(q) for q in P]
    with torch.no_grad():
        for dst, src in ((am, st["m"]), (av, st["v"])):
            for q, v in zip(dst, src):
                q.copy_(v.to(dev))
    tc = int(st["tc"])
    gb = {}
    for s in SEEDS:
        gb[s] = torch.Generator(device="cpu"); gb[s].set_state(st["g_batch"][s])
    lr, b1, b2, eps = B.LR, 0.9, 0.999, 1e-8
    BATCH, N, NC = B.BATCH, B.N_IMAGES, B.N_CLASSES
    ar = torch.arange(R, device=dev)[:, None]
    idx = torch.zeros(R, BATCH, dtype=torch.long, device=dev)
    ic1 = torch.zeros((), device=dev); ic2 = torch.zeros((), device=dev)
    res = {k: [] for k in ("step", "cos_32_64", "rel_diff", "cos64_m", "cos32_m", "n_py1", "resid_py1", "resid_all", "cos32_u")}

    def probe(ts):
        # float64
        P64 = [q.detach().double() for q in P]
        W1 = P64[0].requires_grad_(True)
        z1 = torch.baddbmm(P64[1][:, None, :], X64, W1.transpose(1, 2))
        a1 = torch.where(z1 > 0, z1, 0.1 * z1)
        z2 = torch.baddbmm(P64[3][:, None, :], a1, P64[2].transpose(1, 2))
        a2 = torch.where(z2 > 0, z2, 0.1 * z2)
        z3 = torch.baddbmm(P64[5][:, None, :], a2, P64[4].transpose(1, 2))
        ce = F.cross_entropy(z3.reshape(-1, NC), Y.reshape(-1), reduction="none").view(R, N)
        (g64,) = torch.autograd.grad(ce.sum() / N, [W1])
        # float32, the training's own ops, all 1200 images in one batch
        W1f = P[0].detach().clone().requires_grad_(True)
        Pf = [W1f] + [q.detach() for q in P[1:]]
        _, _, _, _, lg = B.forward(Pf, X, act, train=True)
        cef = F.cross_entropy(lg.reshape(-1, NC), Y.reshape(-1), reduction="none").view(R, N).mean(1)
        (g32,) = torch.autograd.grad(cef.sum(), [W1f])
        with torch.no_grad():
            g32 = g32.double(); m = (am[0] * ic1).double()
            u = m / ((av[0] * ic2).double().sqrt() + eps)
            nrm = lambda x: x.pow(2).sum((1, 2)).sqrt()
            c = lambda x, y: (x * y).sum((1, 2)) / (nrm(x) * nrm(y) + 1e-300)
            p32 = F.softmax(lg.detach(), -1).gather(2, Y[:, :, None]).squeeze(2)
            zd = z3.detach(); dz = zd - zd.gather(2, Y[:, :, None]); e = torch.exp(dz)
            e.scatter_(2, Y[:, :, None], 0.0); se = e.sum(2); resid = se / (1 + se)
            one = p32 == 1.0
            for k, v in (("step", torch.full((R,), ts)), ("cos_32_64", c(g32, g64)), ("rel_diff", nrm(g32 - g64) / nrm(g64)),
                         ("cos64_m", c(g64, m)), ("cos32_m", c(g32, m)), ("n_py1", one.sum(1)),
                         ("resid_py1", (resid * one).sum(1)), ("resid_all", resid.sum(1)), ("cos32_u", c(g32, u))):
                res[k].append(v.double().cpu().numpy())

    def step():
        xb, yb = X[ar, idx], Y[ar, idx]
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, NC), yb.reshape(-1), reduction="none").view(R, BATCH).mean(1)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            for p, gr, mi, vi in zip(P, grads, am, av):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p.sub_(lr * (mi * ic1) / ((vi * ic2).sqrt() + eps))

    t0 = time.time(); ts = 0; S = max(AT)
    for e in range(-(-S // B.STEPS_PER_EPOCH)):
        order = {s: torch.randperm(N, generator=gb[s]) for s in SEEDS}
        ORD = torch.stack([order[s] for s in SEEDS]).to(dev)
        for j in range(B.STEPS_PER_EPOCH):
            if ts >= S:
                break
            idx.copy_(ORD[:, j * BATCH:(j + 1) * BATCH]); tc += 1; ts += 1
            ic1.fill_(1.0 / (1 - b1 ** tc)); ic2.fill_(1.0 / (1 - b2 ** tc))
            step()
            if ts in AT:
                probe(ts)
    np.savez_compressed(out / "g32.npz", **{k: np.stack(v) for k, v in res.items()})
    (out / "provenance_g32.json").write_text(json.dumps({"pass": 5, "net": a.net, "branch": a.branch, "at": AT,
        "git": B.git_state(), "seconds": time.time() - t0}, indent=1, default=str))
    print(f"done {a.net}/{a.branch} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

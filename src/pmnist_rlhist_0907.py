"""Capture the layer-1/2 preactivation distribution every task on Random Label MNIST.

Same protocol as src/pmnist_rlmnist_0906.py (1200 images, fresh random labels per
task, 400 epochs, batch 16, Adam 1e-3, optional l2/l2init) but stores, per task,
a fixed-bin histogram of the pooled preactivations and the 100 per-unit means so
the gate can be animated against phi'.
"""
import sys, json
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pmnist_0905 as H
import pmnist_rlmnist_0906 as RL

LO, HI, NB = -16.0, 8.0, 240
EDGES = np.linspace(LO, HI, NB + 1)
BATCH = H.BATCH


def run(arm, iv, seed, lr, n_tasks, epochs, mnist, device, c=0.6, beta=0.01):
    act = H.ARMS[arm]
    if act.kind == "adaptive_snake":
        act = H.AdaptiveSnake(c, beta, device)
    p = H.init_params(seed, device)
    p0 = [q.detach().clone() for q in p] if iv.startswith("l2init") else None
    lam = float(iv.split(":")[1]) if ":" in iv else 0.0
    kind = iv.split(":")[0]
    # reuse the run module's own helpers so the subset and label stream are
    # bit-identical to pmnist_rlmnist_0906 for the same seed
    idx = RL.subset_idx(seed)
    x = mnist.train_x[idx.to(device)]
    g_lab, g_bat = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
    m = [torch.zeros_like(q) for q in p]; v = [torch.zeros_like(q) for q in p]; tc = 0
    H1, H2, M1, M2, ACC = [], [], [], [], []
    n = x.shape[0]; steps = n // BATCH
    for t in range(1, n_tasks + 1):
        y = RL.task_labels(g_lab).to(device)
        for _ in range(epochs):
            order = torch.randperm(n, generator=g_bat).to(device)
            xs, ys = x[order], y[order]
            for s in range(steps):
                out = H.forward(p, xs[s * BATCH:(s + 1) * BATCH], act)
                loss = torch.nn.functional.cross_entropy(out[4], ys[s * BATCH:(s + 1) * BATCH])
                grads = torch.autograd.grad(loss, p)
                with torch.no_grad():
                    if kind in ("l2", "l2init"):
                        grads = [gr + 2.0 * lam * (q - (p0[i] if p0 is not None else 0.0))
                                 for i, (q, gr) in enumerate(zip(p, grads))]
                    tc += 1; b1, b2, eps = 0.9, 0.999, 1e-8
                    c1, c2 = 1 - b1 ** tc, 1 - b2 ** tc
                    for q, gr, mi, vi in zip(p, grads, m, v):
                        mi.mul_(b1).add_(gr, alpha=1 - b1); vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                        q -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
                    if isinstance(act, H.AdaptiveSnake): act.update(out[0], out[2])
        with torch.no_grad():
            z1, _, z2, _, lo = H.forward(p, x, act)
            ACC.append(float((lo.argmax(1) == y).float().mean()))
            for z, Hh, Mm in ((z1, H1, M1), (z2, H2, M2)):
                Hh.append(np.histogram(z.flatten().cpu().numpy(), bins=EDGES)[0])
                Mm.append(z.mean(0).cpu().numpy())
        print(f"  {arm}{'+' + kind if kind != 'none' else ''} task {t}/{n_tasks} memo={ACC[-1]:.3f}", flush=True)
    return dict(h1=np.array(H1), h2=np.array(H2), m1=np.array(M1), m2=np.array(M2), acc=np.array(ACC))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=50); ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=0.001)
    a = ap.parse_args()
    device = H.setup("auto"); mnist = H.Mnist(device)
    out = Path(H.REPO / "results" / "_diag_rlhist_0907"); out.mkdir(parents=True, exist_ok=True)
    for arm, iv, tag in (("R", "none", "R"), ("LR", "none", "LR"), ("SNA", "none", "SNA"),
                         ("R", "l2:1e-3", "R_l2"), ("LR", "l2:1e-3", "LR_l2"),
                         ("SNA", "l2:1e-3", "SNA_l2"), ("R", "l2init:1e-3", "R_l2init")):
        r = run(arm, iv, a.seed, a.lr, a.tasks, a.epochs, mnist, device)
        np.savez_compressed(out / f"{tag}.npz", edges=EDGES, **r)
        print(f"{tag}: memo t1={r['acc'][0]:.4f} t{a.tasks}={r['acc'][-1]:.4f}", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()

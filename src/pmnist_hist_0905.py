"""Capture the layer-1/2 preactivation distribution every task, for animation.

Stores per task: a fixed-bin histogram of the pooled preactivations
z_i(x) over (test set x units), and the 100 per-unit means zbar_i.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pmnist_0905 import (ARMS, Mnist, TASK_EXAMPLES, BATCH, STEPS_PER_TASK, REPO,
                         init_params, forward, stream, stratified_draw, setup)

LO, HI, NB = -14.0, 8.0, 220
EDGES = np.linspace(LO, HI, NB + 1)


def run(arm, seed, lr, n_tasks, mnist, device):
    act = ARMS[arm]
    p = init_params(seed, device)
    gp, gd, gb = stream("perm", seed), stream("data", seed), stream("batch", seed)
    m = [torch.zeros_like(q) for q in p]; v = [torch.zeros_like(q) for q in p]; tc = 0
    H1, H2, M1, M2, ACC = [], [], [], [], []
    for t in range(1, n_tasks + 1):
        perm = torch.randperm(784, generator=gp).to(device)
        idx = stratified_draw(mnist, gd).to(device)
        order = torch.randperm(TASK_EXAMPLES, generator=gb).to(device)
        xs, ys = mnist.train_x[idx][:, perm][order], mnist.train_y[idx][order]
        for s in range(STEPS_PER_TASK):
            loss = torch.nn.functional.cross_entropy(
                forward(p, xs[s*BATCH:(s+1)*BATCH], act)[4], ys[s*BATCH:(s+1)*BATCH])
            g = torch.autograd.grad(loss, p)
            with torch.no_grad():
                tc += 1; b1, b2, eps = 0.9, 0.999, 1e-8
                c1, c2 = 1-b1**tc, 1-b2**tc
                for q, gr, mi, vi in zip(p, g, m, v):
                    mi.mul_(b1).add_(gr, alpha=1-b1)
                    vi.mul_(b2).addcmul_(gr, gr, value=1-b2)
                    q -= lr * (mi/c1) / ((vi/c2).sqrt() + eps)
        with torch.no_grad():
            z1, _, z2, _, lo = forward(p, mnist.test_x[:, perm], act)
            ACC.append(float((lo.argmax(1) == mnist.test_y).float().mean()))
            for z, H, M in ((z1, H1, M1), (z2, H2, M2)):
                zc = z.flatten().cpu().numpy()
                H.append(np.histogram(zc, bins=EDGES)[0])
                M.append(z.mean(0).cpu().numpy())
    return dict(h1=np.array(H1), h2=np.array(H2), m1=np.array(M1), m2=np.array(M2),
                acc=np.array(ACC))


def main():
    device = setup("auto"); mnist = Mnist(device)
    out = Path(REPO/"results"/"_diag_hist_0905"); out.mkdir(parents=True, exist_ok=True)
    for arm in ("R", "LR", "SN3", "SN1", "LIN"):
        r = run(arm, 0, 0.001, 200, mnist, device)
        np.savez_compressed(out/f"{arm}.npz", edges=EDGES, **r)
        print(f"{arm}: acc t1={r['acc'][0]:.4f} t200={r['acc'][-1]:.4f}", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()

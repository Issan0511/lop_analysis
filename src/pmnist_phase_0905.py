"""Phase-locking probe: does a Snake unit's mean preactivation settle onto a
zero of phi'?

phi'(z) = 1 + sin(2*alpha*z) vanishes at z_k = (-pi/2 + 2*pi*k) / (2*alpha).
For alpha=1 the nearest zeros to the origin are -pi/4 = -0.785 and
3*pi/4 = 2.356; the maxima of phi' sit halfway between.

Reports, per unit, the phase  u = frac((2*alpha*zbar + pi/2) / (2*pi))  in
[0,1), where u = 0 means sitting exactly on a zero of phi' and u = 0.5 means
sitting on a maximum. A uniform histogram means no locking.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pmnist_0905 import (ARMS, Mnist, TASK_EXAMPLES, BATCH, STEPS_PER_TASK, REPO,
                         init_params, forward, stream, stratified_draw, setup, git_hash)


def run(arm, seed, lr, n_tasks, mnist, device):
    act = ARMS[arm]
    p = init_params(seed, device)
    gp, gd, gb = stream("perm", seed), stream("data", seed), stream("batch", seed)
    snap = {}
    for t in range(1, n_tasks + 1):
        perm = torch.randperm(784, generator=gp).to(device)
        idx = stratified_draw(mnist, gd).to(device)
        order = torch.randperm(TASK_EXAMPLES, generator=gb).to(device)
        xs, ys = mnist.train_x[idx][:, perm][order], mnist.train_y[idx][order]
        for s in range(STEPS_PER_TASK):
            loss = torch.nn.functional.cross_entropy(
                forward(p, xs[s * BATCH:(s + 1) * BATCH], act)[4], ys[s * BATCH:(s + 1) * BATCH])
            g = torch.autograd.grad(loss, p)
            with torch.no_grad():
                for q, gr in zip(p, g):
                    q -= lr * gr
        if t in (1, 10, 50, 100, 200):
            with torch.no_grad():
                z1, _, z2, _, _ = forward(p, mnist.test_x[:, perm], act)
                snap[t] = {"zbar_l1": z1.mean(0).cpu().numpy().tolist(),
                           "zbar_l2": z2.mean(0).cpu().numpy().tolist(),
                           "zsd_l1": z1.std(0).cpu().numpy().tolist(),
                           "dphi_at_zbar_l1": act.dphi(z1.mean(0)).cpu().numpy().tolist(),
                           "mob_l1": act.dphi(z1).mean(0).cpu().numpy().tolist()}
    return snap


def main():
    device = setup("auto")
    mnist = Mnist(device)
    out = Path(REPO / "results" / "_diag_phase_0905"); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for arm, lr in (("SN1", 0.05), ("SN1", 0.02), ("SN3", 0.05), ("R", 0.05), ("LIN", 0.05)):
        for seed in (0, 1, 2):
            k = f"{arm}_lr{lr}_s{seed}"
            res[k] = run(arm, seed, lr, 200, mnist, device)
            print(f"done {k}", flush=True)
    (out / "phase.json").write_text(json.dumps({"git": git_hash(), "runs": res}))
    print("wrote", out / "phase.json")


if __name__ == "__main__":
    main()

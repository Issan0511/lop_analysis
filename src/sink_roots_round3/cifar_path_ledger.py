#!/usr/bin/env python3
"""sink_roots_0930 round 3, R1p_cifar_path_ledger (spec_sink_roots_0930_round3.md).

RL-CIFAR ELU/std (rlcifar_mlp_battle_0918's box: 3072-100-100-10, the seed's 1200 standardized images, random labels
per task from rlc_labels, one randperm per epoch from rlc_batch, 400 epochs = 30,000 steps per task, Adam 1e-3), one
seed per process on the CPU.  The step is the engine's eager step for R = 1 (same forward, same loss expression, same
Adam expressions with the engine's inv_c1 / inv_c2); nothing else touches the parameters.

Every `every` steps (and at each task's step 0): layer 2's W2 (100 x 100), b2, mu2 = mean over the 1200 images of the
layer-1 output a1 (float64 mean of the float32 activations), m2 = mean of z2 (float64), and k2 = open images per
layer-2 unit.  At each task's end the six parameter tensors are compared with the R7 A arm's snapshot of the same seed
and task (the natural run of the same box, sink_roots_0930 R7), which must be bit-identical.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import rlcifar_mlp_battle_0918 as B            # noqa: E402
from src import pmnist_0905 as H                        # noqa: E402
from src import pmnist_rlcifar_0907 as RC               # noqa: E402

EPOCHS = 400


def run(seed: int, n_tasks: int, every: int, out: Path, check: Path | None) -> None:
    torch.set_num_threads(1)
    t_start = time.time()
    dev = torch.device("cpu")
    act = B.make_act("ELU")
    cifar = RC.Cifar10()
    X = torch.stack([B.slot_inputs(cifar, seed, "std", dev)])                       # (1, 1200, 3072)
    init = [q.detach() for q in H.init_params(seed, dev, B.DIMS)]
    P = [torch.stack([q]).contiguous().requires_grad_(True) for q in init]
    act.init_state(1, dev, key=f"ELU|{[seed]}|{['std']}")
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    g_lab, g_batch = H.stream("rlc_labels", seed), H.stream("rlc_batch", seed)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, B.LR
    inv_c1, inv_c2 = torch.zeros(()), torch.zeros(())
    static_idx = torch.zeros(1, B.BATCH, dtype=torch.long)
    Ydev = torch.zeros(1, B.N_IMAGES, dtype=torch.long)
    ar = torch.arange(1)[:, None]
    tc = 0
    rec = {k: [] for k in ("task", "step", "W2", "b2", "mu2", "m2", "k2")}
    checks = []

    def step():
        xb, yb = X[ar, static_idx], Ydev[ar, static_idx]
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, B.N_CLASSES), yb.reshape(-1),
                                reduction="none").view(1, B.BATCH).mean(1)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            for p, gr, mi, vi in zip(P, grads, adam_m, adam_v):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
            act.update(z1.detach(), z2.detach())

    @torch.no_grad()
    def record(t: int, ts: int):
        z1, a1, z2, a2, z3 = B.forward(P, X, act, train=False)
        rec["task"].append(t); rec["step"].append(ts)
        rec["W2"].append(P[2][0].detach().numpy().copy()); rec["b2"].append(P[3][0].detach().numpy().copy())
        rec["mu2"].append(a1[0].double().mean(0).numpy()); rec["m2"].append(z2[0].double().mean(0).numpy())
        rec["k2"].append((z2[0] > 0).sum(0).numpy().astype(np.int16))

    if check is not None:                                     # the initial parameters against R7 A's t00
        d = np.load(B.snapshot_path(check, "ELU", "std", seed, 0))
        checks.append({"task": 0, "bit_identical": all(np.array_equal(d[k], P[i][0].detach().numpy())
                                                        for i, k in enumerate(("W1", "b1", "W2", "b2", "W3", "b3")))})
    for t in range(1, n_tasks + 1):
        lab = RC.task_labels(g_lab)
        Ydev.copy_(torch.stack([lab]))
        ts = 0
        record(t, 0)
        for e in range(EPOCHS):
            order = torch.randperm(B.N_IMAGES, generator=g_batch)
            ORD = torch.stack([order])
            for j in range(B.STEPS_PER_EPOCH):
                static_idx.copy_(ORD[:, j * B.BATCH:(j + 1) * B.BATCH])
                tc += 1
                ts += 1
                inv_c1.fill_(1.0 / (1 - b1 ** tc))
                inv_c2.fill_(1.0 / (1 - b2 ** tc))
                act.begin_step(1, B.BATCH, dev)
                step()
                if ts % every == 0:
                    record(t, ts)
        if check is not None:
            f = B.snapshot_path(check, "ELU", "std", seed, t)
            if f.exists():
                d = np.load(f)
                same = {k: bool(np.array_equal(d[k], P[i][0].detach().numpy()))
                        for i, k in enumerate(("W1", "b1", "W2", "b2", "W3", "b3"))}
                checks.append({"task": t, "snapshot": str(f), "bit_identical": all(same.values()), "per_tensor": same})
        print(f"R1p ELU/std seed {seed} task {t} done ({time.time() - t_start:.0f}s)"
              + (f" | bit-identical to R7 A: {checks[-1]['bit_identical']}" if checks and checks[-1]["task"] == t else ""), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"ledger_ELU_std_s{seed}.npz", task=np.array(rec["task"]), step=np.array(rec["step"]),
                        W2=np.stack(rec["W2"]), b2=np.stack(rec["b2"]), mu2=np.stack(rec["mu2"]), m2=np.stack(rec["m2"]),
                        k2=np.stack(rec["k2"]), every=every)
    prov = {"experiment": "sink_roots_0930", "study": "R1p_cifar_path_ledger", **B.git_state(), "seed": seed,
            "tasks": n_tasks, "every": every, "checks": checks, "wall_s": time.time() - t_start, "torch": torch.__version__}
    (out / f"provenance_s{seed}.json").write_text(json.dumps(prov, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default="/home/issan/Projects/obsidian-research-data/sink_roots_0930/r7/A")
    a = ap.parse_args()
    chk = Path(a.check) / f"s{a.seed}" if a.check else None
    run(a.seed, a.tasks, a.every, Path(a.out), chk)


if __name__ == "__main__":
    main()

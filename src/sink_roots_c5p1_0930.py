#!/usr/bin/env python3
"""sink_roots_0930 R11b -- the 5+1 CIFAR box of cifar5p1_mlp_0920, one seed per process on the CPU,
with probes at every task switch (spec: specs/spec_sink_roots_0930.md §6.2).

The step is cifar5p1_mlp_0920.run()'s eager step for R = 1 (same activations, same stacked forward,
same Adam expressions, same class plan and batch stream); only probes are added, and they touch no
tensor the step reads.  Grid in every task: updates {0, 10, 25, 50, 78, 150, 300, 500, 780}.

Per probe, for both hidden layers, per unit: m (mean preactivation), p+ (fraction z > 0) and the top
(max z) on the task's own training images, and m and p+ on a fixed probe set of 2,000 CIFAR-100 test
images (stream "c51_probe_0930").  At update 0 of every task after the first: the same unit stats on
the previous task's images at the same weights (the input kick is m(new) - m(old)), and the expected
push of layer 2 split into the cancel part <K phi' v^c.p> and the new-class part -<K phi' v^c_y>.
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

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import cifar5p1_mlp_0920 as C                 # noqa: E402
from src import rlcifar_mlp_battle_0918 as B           # noqa: E402
from src import pmnist_0905 as H                       # noqa: E402

GRID = (0, 10, 25, 50, 78, 150, 300, 500, 780)
N_PROBE = 2000


@torch.no_grad()
def unit_stats(P, X, act):
    z1, a1, z2, a2, z3 = B.forward(P, X[None], act, train=False)
    out = {}
    for tag, z in (("1", z1[0]), ("2", z2[0])):
        out["m" + tag] = z.mean(0)
        out["pp" + tag] = (z > 0).float().mean(0)
        out["top" + tag] = z.max(0).values
    return out, (z1[0], a1[0], z2[0], a2[0], z3[0])


def run(arm: str, seed: int, out: Path, n_tasks: int, lr: float):
    t_start = time.time()
    dev = torch.device("cpu")
    cifar = C.Cifar100()
    X_all = cifar.inputs("train", "std", dev)
    Y_all = cifar.train_y
    X_test = cifar.inputs("test", "std", dev)
    tr_rows = C.class_rows(cifar.train_y)
    plan = C.task_plan(seed)
    g_batch = H.stream("c51_batch", seed)
    probe_idx = torch.randperm(len(X_test), generator=H.stream("c51_probe_0930", seed))[:N_PROBE]
    Xp = X_test[probe_idx]
    act = C.make_act(arm, C.HIDDEN)
    init = C.init_params(arm, seed, dev, C.HIDDEN)
    P = [torch.stack([q]).contiguous().requires_grad_(True) for q in init]
    act.init_state(1, dev, key=f"{arm}|{[seed]}|std")
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    b1, b2, eps = 0.9, 0.999, 1e-8
    inv_c1 = torch.zeros(())
    inv_c2 = torch.zeros(())
    tc = 0
    U, S, rows = {}, {}, []
    prev_rows = None
    out.mkdir(parents=True, exist_ok=True)
    for t in range(1, n_tasks + 1):
        hard = t % 2 == 1
        classes = plan[t - 1][1]
        rows_t = torch.cat([tr_rows[q] for q in classes])
        batches = C.batch_indices(g_batch, rows_t, C.STEPS_PER_TASK)
        Xc, Yc = X_all[rows_t], Y_all[rows_t]
        acc_sum = 0.0
        task_U, task_S = {}, []
        for s in range(C.STEPS_PER_TASK + 1):
            if s in GRID:
                cur, (z1, a1, z2, a2, z3) = unit_stats(P, Xc, act)
                prb, _ = unit_stats(P, Xp, act)
                d = {k: v for k, v in cur.items()}
                d.update({k + "_probe": v for k, v in prb.items() if k.startswith(("m", "pp"))})
                p = z3.softmax(-1)
                sc = {"ce": float(F.cross_entropy(z3, Yc)), "acc": float((z3.argmax(-1) == Yc).float().mean()),
                      "pmax": float(p.max(-1).values.mean()),
                      "p_task": float(p[:, classes].sum(-1).mean())}
                if s == 0 and prev_rows is not None:
                    old, _ = unit_stats(P, X_all[prev_rows], act)
                    d.update({k + "_old": v for k, v in old.items()})
                    mu2 = a1.mean(0)
                    K2 = a1 @ mu2 + 1.0
                    g2 = act.dphi(z2[None], 1)[0] * K2[:, None]
                    W3 = P[4].detach()[0]                                   # (100 classes, 100 units)
                    W3c = W3 - W3.mean(0, keepdim=True)
                    d["S2_cancel"] = (g2 * (p @ W3c)).mean(0)
                    d["S2_new"] = -(g2 * W3c[Yc]).mean(0)
                    d["K2_min"] = K2.min().expand(1)
                for k, v in d.items():
                    task_U.setdefault(k, []).append(v.float().numpy())
                task_S.append(sc)
            if s == C.STEPS_PER_TASK:
                break
            idx = batches[s][None]
            xb, yb = X_all[idx], Y_all[idx]
            tc += 1
            inv_c1.fill_(1.0 / (1 - b1 ** tc))
            inv_c2.fill_(1.0 / (1 - b2 ** tc))
            act.begin_step(1, C.BATCH, dev)
            z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
            lossv = F.cross_entropy(z3.reshape(-1, C.N_CLASSES), yb.reshape(-1),
                                    reduction="none").view(1, C.BATCH).mean(1)
            acc_sum += float((z3.detach().argmax(-1) == yb).float().mean())
            grads = torch.autograd.grad(lossv.sum(), P)
            with torch.no_grad():
                for q, gr, mi, vi in zip(P, grads, adam_m, adam_v):
                    mi.mul_(b1).add_(gr, alpha=1 - b1)
                    vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                    q.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
                act.update(z1.detach(), z2.detach())
            if not torch.isfinite(lossv).all():
                rows.append({"task": t, "diverged": True, "step": s})
                break
        for k, v in task_U.items():
            U.setdefault(k, {})[t] = np.stack(v)
        S[t] = task_S
        rows.append({"task": t, "hard": int(hard), "classes": classes,
                     "online_acc": acc_sum / C.STEPS_PER_TASK, "train_acc_end": task_S[-1]["acc"],
                     "ce_end": task_S[-1]["ce"]})
        prev_rows = rows_t
        print(f"R11b {arm} s{seed} task {t:2d} {'hard' if hard else 'easy'} online {acc_sum / C.STEPS_PER_TASK:.3f} "
              f"({time.time() - t_start:.0f}s)", flush=True)
    arrays = {}
    for k, per_t in U.items():
        ts = sorted(per_t)
        arrays["u_" + k] = np.stack([per_t[t] for t in ts]) if len({per_t[t].shape for t in ts}) == 1 else \
            np.array([per_t[t] for t in ts], dtype=object)
        arrays["tasks_" + k] = np.array(ts)
    keys = sorted(set().union(*[set(d) for v in S.values() for d in v]))
    for k in keys:
        arrays["s_" + k] = np.array([[d.get(k, np.nan) for d in S[t]] for t in sorted(S)])
    arrays["grid"] = np.array(GRID)
    np.savez_compressed(out / "arrays.npz", **arrays)
    (out / "rows.json").write_text(json.dumps(rows))
    prov = {"experiment": "sink_roots_0930", "study": "R11b", **B.git_state(), "arm": arm, "seed": seed,
            "lr": lr, "n_tasks": n_tasks, "grid": GRID, "n_probe": N_PROBE, "data_sha256": cifar.sha256,
            "torch": torch.__version__, "wall_s": time.time() - t_start,
            "engine": "cifar5p1_mlp_0920 eager step for R = 1 on the CPU, probes added"}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tasks", type=int, default=C.N_TASKS)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.arm, a.seed, Path(a.out), a.tasks, a.lr)


if __name__ == "__main__":
    main()

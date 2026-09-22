#!/usr/bin/env python3
"""Per-arm, per-seed, per-task snapshot quantities for switch_push_cifar_0922 (spec §3).

    python3 analysis/switch_push_cifar_0922/tail.py     # -> results/switch_push_cifar_0922/tail.csv

Forked from le_eps_cifar_0922/tail.py; LE_ref / LL_ref are S-A's unfrozen runs (snapshots in
the obsidian-research-data archive), the others are this experiment's.  Everything is recomputed
here from the float32 weights in float64 on the cpu, the same way for every arm, so the arms are
on one footing.  `dcos_grad` uses the engine's float32 forward + autograd (train=True) at the
previous task's end state with the current task's labels (full batch).
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import layer_chimera_cifar_0921 as C          # noqa: E402
from src import pmnist_0905 as H                        # noqa: E402
from src import pmnist_rlcifar_0907 as RC               # noqa: E402

OUT = REPO / "results" / "switch_push_cifar_0922"
SA = REPO / "results" / "layer_chimera_cifar_0921"
SA_RAW = Path("/home/issan/Projects/obsidian-research-data/layer_chimera_cifar_0921/results/layer_chimera_cifar_0921")
ARMS = {
    "LE_ref": ("LE", 0, SA_RAW / "LE"), "LL_ref": ("LL", 0, SA_RAW / "LL"),
    "LE_sw750": ("LE", 750, OUT / "LE_sw750"),
    "LE_md750": ("LE", 750, OUT / "LE_md750"),
    "LE_sw7500": ("LE", 7500, OUT / "LE_sw7500"),
    "LE_md7500": ("LE", 7500, OUT / "LE_md7500"),
    "LE_sw75": ("LE", 75, OUT / "LE_sw75"),
    "LE_md75": ("LE", 75, OUT / "LE_md75"),
    "LL_sw750": ("LL", 750, OUT / "LL_sw750"),
    "LL_md750": ("LL", 750, OUT / "LL_md750"),
}
EPS_OPEN = float(np.log(1e-8))          # e^{z2} > Adam's parent eps
COLS = ["arm", "cell", "freeze_steps", "seed", "task", "open_frac", "n_img", "n_unit", "npos", "apos",
        "a1_rms", "mu2", "zbar2_med", "p_pos2", "w2_row_med", "cos2_med",
        "dcos", "dcos_pos", "drms", "dcos_grad"]


def act_np(z, kind):
    if kind == "LR":
        return np.where(z > 0, z, 0.1 * z)
    if kind == "ELU":
        return np.where(z > 0, z, np.expm1(np.minimum(z, 0)))
    raise ValueError(kind)


def labels(seed, task):
    g = H.stream("rlc_labels", seed)
    y = None
    for _ in range(task):
        y = RC.task_labels(g)
    return y


def grad_w1(P32, X32, Y, act):
    P = [p.clone().requires_grad_(True) for p in P32]
    *_, z3 = C.forward(P, X32, act, train=True)
    loss = F.cross_entropy(z3.reshape(-1, C.N_CLASSES), Y.reshape(-1))
    return torch.autograd.grad(loss, P)[0][0].detach().double().cpu().numpy()


def one_arm(arm, cell, frz, root, cifar, tasks, dev):
    a1n, a2n = C.CELLS[cell]
    act = C.make_act(cell)
    act.init_state(1, dev, key=f"tail|{arm}")
    rows = []
    for seed in range(10):
        X32 = C.slot_inputs(cifar, seed, "raw", dev)[None]
        x = X32[0].double().cpu().numpy()
        xb = x.mean(0)
        xbn = xb / np.linalg.norm(xb)
        prev = None
        for t in range(tasks + 1):
            d = np.load(C.snapshot_path(root, cell, "raw", seed, t))
            W1, b1, W2, b2 = (d[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2"))
            z1 = x @ W1.T + b1
            a1 = act_np(z1, a1n)
            z2 = a1 @ W2.T + b2
            op = z2 > EPS_OPEN
            pos = z1.mean(0) > 0
            mu = a1.mean(0)
            w2n = np.linalg.norm(W2, axis=1)
            cos2 = (W2 @ mu) / np.maximum(w2n * np.linalg.norm(mu), 1e-300)
            r = {"arm": arm, "cell": cell, "freeze_steps": frz, "seed": seed, "task": t,
                 "open_frac": float(op.mean()), "n_img": int(op.any(1).sum()), "n_unit": int(op.any(0).sum()),
                 "npos": int(pos.sum()), "apos": float(a1[:, pos].mean()) if pos.any() else float("nan"),
                 "a1_rms": float(np.sqrt((a1 ** 2).mean())), "mu2": float(np.linalg.norm(mu)),
                 "zbar2_med": float(np.median(z2.mean(0))), "p_pos2": float((z2 > 0).mean()),
                 "w2_row_med": float(np.median(w2n)), "cos2_med": float(np.median(cos2)),
                 "dcos": float("nan"), "dcos_pos": float("nan"), "drms": float("nan"), "dcos_grad": float("nan")}
            if prev is not None:
                dW = W1 - prev["W1"]
                nrm = np.maximum(np.linalg.norm(dW, axis=1), 1e-30)
                cos = (dW @ xbn) / nrm
                r["dcos"] = float(np.median(np.abs(cos)))
                r["dcos_pos"] = float(np.median(cos[pos])) if pos.any() else float("nan")
                r["drms"] = float(np.sqrt((dW ** 2).mean()))
                Y = labels(seed, t).to(dev)[None]
                g = -grad_w1(prev["P32"], X32, Y, act)
                gn, dn = np.linalg.norm(g), np.linalg.norm(dW)
                r["dcos_grad"] = float((g * dW).sum() / (gn * dn)) if gn > 0 and dn > 0 else 0.0
            rows.append(r)
            prev = {"W1": W1, "P32": [torch.from_numpy(d[k]).to(dev)[None]
                                       for k in ("W1", "b1", "W2", "b2", "W3", "b3")]}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = H.setup(a.device)
    cifar = RC.Cifar10()
    rows, t0 = [], time.time()
    for arm in a.arms.split(","):
        cell, frz, root = ARMS[arm]
        rows += one_arm(arm, cell, frz, root, cifar, a.tasks, dev)
        print(f"{arm}: {len(rows)} rows ({time.time() - t0:.0f} s)", flush=True)
    H.write_csv(OUT / "tail.csv", [{k: r[k] for k in COLS} for r in rows])
    (OUT / "tail_provenance.json").write_text(json.dumps(
        {"run_id": "switch_push_cifar_0922", "stage": "tail", **C.git_state(), "arms": a.arms,
         "tasks": a.tasks, "device": str(dev), "wall_s": time.time() - t0, "rows": len(rows),
         "eps_open": "e^{z2} > 1e-8"}, indent=2))


if __name__ == "__main__":
    main()

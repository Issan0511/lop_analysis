#!/usr/bin/env python3
"""depth_silu_rlmnist_1020 -- the 1-hidden-layer RL-MNIST box with L hidden layers (spec: specs/spec_depth_silu_rlmnist_1020.md).

    python src/depth_silu_rlmnist_1020.py --act SILU --layers 4 --seed 0 --tasks 50 --out raw/depth_silu_rlmnist_1020/runs

The box is obsidian-research-data/push_direction_proof_0930/gen_states.py (sha256 4571091d...0fec3f30) with the
hidden layer repeated: 784-(100 x L)-10, raw MNIST/255, a fixed subset of 1,200 images (stream "rl_subset"), K = 10
uniform random labels redrawn every task ("env_labels_0913"), batch 16 ("env_batch_0913"), init U(+-1/sqrt(fan_in))
for weights and biases drawn layer by layer from stream "init" (W1, b1, W2, b2, ..., readout), torch.optim.Adam
1e-3 (0.9, 0.999, 1e-8) carried across tasks, 4,000 updates per task, float32 on CPU with one thread.
L = 1 is gen_states.py op for op (check S-repro1); L = 2 is drive_recon_0930's 2-layer box (check S-repro2).

Door C (--door C): x <- x - x.mean(0) over the seed's own 1,200 images, float32, subtracted and never divided,
made once before training and used by every batch (doors_rlmnist_1007's C).

Saved per task: every parameter after 200 updates (mid) and at the task end (end), the labels, the task-end
full-data training accuracy, and per update the number of correct predictions in the batch and the batch loss,
both from the forward pass that made the update (pre-update).  Diagnostics are computed afterwards in float64
by analysis/depth_silu_rlmnist_1020/measure.py from the saved weights.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
EXPERIMENT = "depth_silu_rlmnist_1020"
N, BATCH, H = 1200, 16, 100
SQRT2 = math.sqrt(2.0)
U32, U64 = 2.0 ** -24, 2.0 ** -53
EPS64 = float(np.finfo(np.float64).eps)


def stream(role, seed):
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def read_idx(p):
    b = gzip.open(p, "rb").read()
    dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def activ(z, act):
    if act == "LR":   return torch.where(z > 0, z, z * 0.1)
    if act == "ELU":  return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))
    if act == "R":    return torch.relu(z)
    if act == "GELU": return z * 0.5 * (1.0 + torch.erf(z / SQRT2))
    if act == "SILU": return z * torch.sigmoid(z)
    raise ValueError(act)


def initial(seed, dims, order="forward"):
    """gen_states.initial: (W, b) per layer from one generator, input layer first.  order="reverse" draws the
    readout first (the S-repro2 mutant only)."""
    g = stream("init", seed); p = []
    pairs = list(zip(dims[:-1], dims[1:]))
    seq = pairs if order == "forward" else pairs[::-1]
    drawn = []
    for din, dout in seq:
        drawn.append([(torch.rand(dout, din, generator=g) * 2 - 1) * (1.0 / math.sqrt(din)),
                      (torch.rand(dout, generator=g) * 2 - 1) * (1.0 / math.sqrt(din))])
    if order != "forward":
        drawn = drawn[::-1]
    for w, b in drawn:
        p += [w, b]
    return p


def gamma(k: int, u: float) -> float:
    return k * u / (1.0 - k * u)


@torch.no_grad()
def c_bound(x_raw: torch.Tensor, xc: torch.Tensor, n_sum: int) -> torch.Tensor:
    """doors_rlmnist_1007.c_bound (copied, not imported: that module pulls in its hosts): per-pixel bound on
    |mean of the centred inputs| in float64 -- a float32 mean of n images (<= 2 extra roundings), one rounding per
    subtraction, and a float64 mean of n_sum fed values, inflated by (1 + 8 eps64)."""
    n = x_raw.shape[0]
    ax = x_raw.double().abs().mean(0)
    axc = xc.double().abs().mean(0)
    return (gamma(n + 2, U32) * ax + (U32 / (1.0 - U32) + gamma(n_sum + 1, U64)) * axc) * (1.0 + 8.0 * EPS64)


def load_inputs(seed, door):
    """(subset, x_raw, x): x is what every batch and every diagnostic sees."""
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=stream("rl_subset", seed))[:N].numpy()
    x_raw = torch.tensor(ax[subset], dtype=torch.float32)
    if door == "C":
        x = x_raw - x_raw.mean(0)
    elif door == "none":
        x = x_raw
    else:
        raise ValueError(door)
    return subset, x_raw, x


def sha(t) -> str:
    a = t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def git_info():
    try:
        h = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", f"analysis/{EXPERIMENT}"],
                           capture_output=True, text=True).stdout.strip()
        return h, d
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}", ""


def run(act, L, seed, tasks, steps, lr, door="none", eps=1e-8, init_order="forward", K=10, mid_at=200,
        progress=True, capture_adam=()):
    """One run.  capture_adam (checks only): task numbers at whose end the Adam moments are copied out."""
    torch.manual_seed(seed)
    subset, x_raw, X = load_inputs(seed, door)
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    dims = (784,) + (H,) * L + (K,)
    P = [q.to(dtype=torch.float32).requires_grad_(True) for q in initial(seed, dims, init_order)]
    optim = torch.optim.Adam(P, lr=lr, betas=(0.9, 0.999), eps=eps)
    snap = lambda: [q.detach().numpy().copy() for q in P]
    P_init = snap(); ends, mids, ys, accs = [], [], [], []
    hits = np.zeros((tasks, steps), dtype=np.uint8); losses = np.zeros((tasks, steps), dtype=np.float32)
    x_sha = sha(X)
    captured = {}
    t0 = time.monotonic()
    for task in range(1, tasks + 1):
        y_new = torch.randint(K, (N,), generator=glabel)
        Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        epochs = -(-(steps * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:steps * BATCH].reshape(steps, BATCH)
        for it in range(steps):
            idx = order[it]; xb, yb = X[idx], Y[idx]
            h = xb
            for l in range(L):
                z = h @ P[2 * l].T + P[2 * l + 1]; h = activ(z, act)
            logits = h @ P[2 * L].T + P[2 * L + 1]
            loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            with torch.no_grad():
                hits[task - 1, it] = int((logits.argmax(-1) == y_new[idx]).sum())
                losses[task - 1, it] = float(loss)
            if it + 1 == mid_at: mids.append(snap())
        ends.append(snap()); ys.append(y_new.numpy().copy())
        if task in capture_adam:
            captured[task] = {"m": [optim.state[q]["exp_avg"].numpy().copy() for q in P],
                              "v": [optim.state[q]["exp_avg_sq"].numpy().copy() for q in P]}
        with torch.no_grad():
            h = X
            for l in range(L):
                h = activ(h @ P[2 * l].T + P[2 * l + 1], act)
            lg = h @ P[2 * L].T + P[2 * L + 1]
            acc = float((lg.argmax(-1) == y_new).float().mean()); accs.append(acc)
        if progress:
            print(f"{act} L{L} {door} s{seed} task {task} acc {acc:.3f} online {hits[task - 1].mean() / BATCH:.3f} "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)
    d = dict(subset=subset, y=np.array(ys), acc=np.array(accs), act=act, K=K, seed=seed, L=L, door=door,
             steps=steps, lr=lr, eps=eps, init_order=init_order, hits=hits, loss=losses, x_sha256=x_sha,
             n_params=len(P))
    for j in range(len(P)):
        d[f"init{j}"] = P_init[j]; d[f"end{j}"] = np.stack([e[j] for e in ends])
        if mids:
            d[f"mid{j}"] = np.stack([e[j] for e in mids])
    adam = {"m": [optim.state[q]["exp_avg"].numpy().copy() for q in P],
            "v": [optim.state[q]["exp_avg_sq"].numpy().copy() for q in P], "captured": captured}
    info = {"door": door, "x_sha256": x_sha, "x_raw_sha256": sha(x_raw), "seconds": time.monotonic() - t0}
    if door == "C":
        bound = c_bound(x_raw, X, N)
        mean64 = X.double().mean(0)
        info |= {"c_mean_absmax": float(mean64.abs().max()), "c_bound_max": float(bound.max()),
                 "c_within_bound": bool((mean64.abs() <= bound).all()), "xbar_sha256": sha(x_raw.mean(0))}
    return d, adam, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True); ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--door", default="none", choices=["none", "C"]); ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=1e-8, help="Adam eps (1e-8; other values are the S-repro1 mutant)")
    ap.add_argument("--init-order", default="forward", choices=["forward", "reverse"])
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(v) != "1":
            sys.exit(f"{v} must be 1 (one BLAS thread, as gen_states)")
    name = a.name or f"{a.act}_L{a.layers}_{a.door}_s{a.seed}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gh, dirty = git_info()
    d, adam, info = run(a.act, a.layers, a.seed, a.tasks, a.steps, a.lr, a.door, a.eps, a.init_order)
    tmp = out / f"{name}.tmp.npz"
    np.savez_compressed(tmp, **d, **{f"adam_m{j}": q for j, q in enumerate(adam["m"])},
                        **{f"adam_v{j}": q for j, q in enumerate(adam["v"])})
    os.replace(tmp, out / f"{name}.npz")
    gh_end, dirty_end = git_info()
    prov = {"experiment": EXPERIMENT, "name": name, "argv": sys.argv, "git_hash": gh, "dirty_src_start": dirty,
            "git_hash_end": gh_end, "dirty_src_end": dirty_end, "torch": torch.__version__, "numpy": np.__version__,
            "python": sys.version, "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "npz_sha256": hashlib.sha256((out / f"{name}.npz").read_bytes()).hexdigest(), **info}
    (out / f"{name}.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    print(f"done {name} {info['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

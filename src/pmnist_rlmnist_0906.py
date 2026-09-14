#!/usr/bin/env python3
"""Random Label MNIST (Kumar, Marklund & Van Roy 2024 §4.2; variant of Lyle et al. 2023).

    python3 src/pmnist_rlmnist_0906.py --arms SNA --seeds 0 --tasks 50

1200 MNIST training images are drawn once per seed, uniformly and WITHOUT
stratification, and every task relabels *those same* images with fresh uniform
labels in {0..9}.  Each task is fitted to memorisation (400 epochs x 75 steps of
batch 16 = 30,000 steps); weights and Adam moments carry across tasks, never
reset.  The plasticity metric is `online_acc`, the mean pre-update batch
accuracy over the task.

The host `pmnist_0905` supplies data / rng / init / activations / forward /
metric formulas / csv and is not modified: only the task construction, the loop
and the primary metric differ.  `pmnist_lopcmp_0905` is deliberately NOT
imported -- the two interventions used here (l2, l2init) are five lines each and
copying them keeps this box free of that box's cbp/snp machinery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Determinism must be requested before the first CUDA workspace is allocated.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H          # host; not touched

EXPERIMENT = "pmnist_rlmnist_0906"
N_IMAGES = 1200                            # images per seed, shared by every task
TRAIN_N = 60_000                           # MNIST training set
BATCH = H.BATCH                            # 16
STEPS_PER_EPOCH = N_IMAGES // BATCH        # 75
N_CLASSES = H.N_CLASSES                    # 10
DEFAULT_ARMS = "SNA,SN02,SN1,SN3,LR,R,LIN"


# --------------------------------------------------------------------------
# task construction
# --------------------------------------------------------------------------

def subset_idx(seed: int) -> torch.Tensor:
    """The seed's 1200 training images: uniform, no replacement, no stratification.

    Drawn once per seed from its own stream, so the image set is a property of
    the seed and is bit-identical across arms, tasks and interventions.
    """
    g = H.stream("rl_subset", seed)
    return torch.randperm(TRAIN_N, generator=g)[:N_IMAGES]


def task_labels(g: torch.Generator) -> torch.Tensor:
    """One task's labels: iid uniform over {0..9}. Advances `g` exactly once."""
    return torch.randint(N_CLASSES, (N_IMAGES,), generator=g, dtype=torch.int64)


@dataclass(frozen=True)
class Iv:
    """Weight-space intervention, same mechanism as the host's run_one."""
    kind: str = "none"
    lam: float = 0.0

    @staticmethod
    def parse(s: str) -> "Iv":
        p = s.split(":")
        if p[0] == "none":
            return Iv("none")
        if p[0] in ("l2", "l2init"):
            return Iv(p[0], float(p[1]))
        raise ValueError(f"--iv must be none|l2:<lam>|l2init:<lam>, got {s!r}")


# --------------------------------------------------------------------------
# metrics: the host's evaluate(), verbatim, but on the 1200 images and their
# current task labels instead of the permuted test set.
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate_rl(params, x: torch.Tensor, y: torch.Tensor, act) -> dict:
    z1, a1, z2, a2, logits = H.forward(params, x, act)
    out = {"acc": float((logits.argmax(1) == y).float().mean())}
    ada = isinstance(act, H.AdaptiveSnake)
    for li, (tag, z, a) in enumerate((("l1", z1, a1), ("l2", z2, a2))):
        d = act.dphi(z, li) if ada else act.dphi(z)
        out[f"dead_frac_{tag}"] = float((d.abs().amax(0) < H.DEAD_TOL).float().mean())
        out[f"zeroout_{tag}"] = float((a.abs().amax(0) == 0).float().mean())
        out[f"zbar_{tag}"] = float(z.mean(0).median())
        out[f"zsd_{tag}"] = float(z.std(0).median())
        out[f"zbar_min_{tag}"] = float(z.mean(0).amin())
        out[f"mob_{tag}"] = float(d.mean(0).median())
        out[f"eff_rank_{tag}"] = H.eff_rank(a)
    for i, tag in enumerate(("l1", "l2", "l3")):
        if 2 * i >= len(params):
            break
        out[f"w_norm_{tag}"] = float(params[2 * i].norm(dim=1).median())
    if ada:
        out.update(act.stats())
    return out


# --------------------------------------------------------------------------
# one (arm, seed, lr) run
# --------------------------------------------------------------------------

def run_one(arm: str, seed: int, lr: float, n_tasks: int, mnist: H.Mnist,
            device: torch.device, optimizer: str = "adam", epochs: int = 400,
            c: float = 0.6, beta: float = 0.01, iv: str = "none",
            debug: dict | None = None) -> tuple[list[dict], dict]:
    act = H.ARMS[arm]
    if act.kind == "adaptive_snake":
        act = H.AdaptiveSnake(c, beta, device)        # fresh statistics per run
    params = H.init_params(seed, device)              # host init: bit-identical per seed
    if debug is not None:                             # S-init / S-online hook, unused in real runs
        debug["init"] = [q.detach().cpu().clone() for q in params]
    ivo = Iv.parse(iv)
    p0 = [q.detach().clone() for q in params] if ivo.kind == "l2init" else None
    # Adam moments live for the whole run: resetting them between tasks would break
    # the continual definition the same way resetting weights does.
    adam = ([torch.zeros_like(q) for q in params],
            [torch.zeros_like(q) for q in params], [0]) if optimizer == "adam" else None

    idx = subset_idx(seed).to(device)
    x = mnist.train_x[idx]                            # the task's inputs, fixed forever
    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
    spt = STEPS_PER_EPOCH * epochs
    rows = []
    diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "arm": arm}

    for t in range(1, n_tasks + 1):
        y = task_labels(g_lab).to(device)             # new labelling, same images
        if debug is not None:
            debug.setdefault("subset", []).append(idx.cpu().clone())
            debug.setdefault("labels", []).append(y.cpu().clone())
        acc_sum = torch.zeros((), device=device)
        bad_step = torch.full((), -1, dtype=torch.long, device=device)

        for e in range(epochs):
            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)
            xs, ys = x[order], y[order]               # reshuffled every epoch
            for j in range(STEPS_PER_EPOCH):
                s = e * STEPS_PER_EPOCH + j
                xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
                out = H.forward(params, xb, act)
                loss = torch.nn.functional.cross_entropy(out[4], yb)
                # pre-update accuracy: the argmax of the very forward pass the loss
                # came from, i.e. before this batch has been learned from.
                hit = (out[4].detach().argmax(1) == yb).float().mean()
                acc_sum += hit
                grads = torch.autograd.grad(loss, params)
                with torch.no_grad():
                    bad = ~torch.isfinite(loss)
                    bad_step = torch.where((bad_step < 0) & bad,
                                           torch.tensor(s, device=device), bad_step)
                    if ivo.kind in ("l2", "l2init"):
                        grads = [gr + 2.0 * ivo.lam * (q - (p0[i] if p0 is not None else 0.0))
                                 for i, (q, gr) in enumerate(zip(params, grads))]
                    if adam is None:
                        for p, gr in zip(params, grads):
                            p -= lr * gr
                    else:
                        m, v, tc = adam
                        tc[0] += 1
                        b1, b2, eps = 0.9, 0.999, 1e-8
                        c1 = 1 - b1 ** tc[0]
                        c2 = 1 - b2 ** tc[0]
                        for p, gr, mi, vi in zip(params, grads, m, v):
                            mi.mul_(b1).add_(gr, alpha=1 - b1)
                            vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                            p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
                    if isinstance(act, H.AdaptiveSnake):
                        act.update(out[0], out[2])    # running var of this batch's preacts
                if debug is not None:
                    debug.setdefault("online", []).append(float(hit))

        bs = int(bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t, step=(t - 1) * spt + max(bs, 0))
            rows.append({"arm": arm, "seed": seed, "lr": lr, "task": t, "iv": iv,
                         "acc": float("nan")})
            break                                     # drop, never rescue
        m = evaluate_rl(params, x, y, act)
        rows.append({"arm": arm, "seed": seed, "lr": lr, "task": t, "iv": iv,
                     "online_acc": float(acc_sum) / spt, "memo_acc": m["acc"], **m})
    return rows, diverged


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def out_tag(arms: list[str], iv: str) -> str:
    """results/<run_id>/<arm>[_<iv>].  ':' -> '-' so the path stays glob-friendly."""
    tag = "-".join(arms)
    return tag if iv == "none" else f"{tag}_{iv.replace(':', '-')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--lrs", default="0.001")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400,
                    help="passes over the 1200 examples per task (75 steps each)")
    ap.add_argument("--c", type=float, default=0.6, help="SNA: alpha_i = c / W_i")
    ap.add_argument("--beta", type=float, default=0.01, help="SNA: EMA rate of var(z_i)")
    ap.add_argument("--iv", default="none", help="none | l2:<lam> | l2init:<lam>")
    ap.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    lrs = [float(s) for s in args.lrs.split(",")]
    Iv.parse(args.iv)                                  # fail fast on a bad spec
    for a in arms:
        if a not in H.ARMS:
            raise SystemExit(f"unknown arm {a!r}; known: {','.join(H.ARMS)}")

    device = H.setup(args.device)
    t_start = time.time()
    mnist = H.Mnist(device)
    assert mnist.train_x.shape[0] == TRAIN_N, mnist.train_x.shape

    out = Path(args.out or H.REPO / "results" / EXPERIMENT / out_tag(arms, args.iv))
    out.mkdir(parents=True, exist_ok=True)

    rows, divs = [], []
    for lr in lrs:
        for arm in arms:
            for seed in seeds:
                t0 = time.time()
                r, d = run_one(arm, seed, lr, args.tasks, mnist, device,
                               optimizer=args.optimizer, epochs=args.epochs,
                               c=args.c, beta=args.beta, iv=args.iv)
                rows += r
                if d["diverged"]:
                    divs.append(d)
                print(f"[{time.time()-t_start:7.1f}s] lr={lr:<6g} {arm:<4} seed={seed} "
                      f"tasks={len([q for q in r if q.get('acc') == q.get('acc')])} "
                      f"{(time.time()-t0):.1f}s"
                      f"{'  DIVERGED@step ' + str(d['step']) if d['diverged'] else ''}",
                      flush=True)
                H.write_csv(out / "per_task.csv", rows)

    prov = {"run_id": EXPERIMENT, "git_hash": H.git_hash(),
            "arms": arms, "seeds": seeds, "lrs": lrs, "n_tasks": args.tasks,
            "epochs_per_task": args.epochs, "batch": BATCH,
            "steps_per_task": STEPS_PER_EPOCH * args.epochs,
            "n_images": N_IMAGES, "dims": list(H.DIMS), "n_classes": N_CLASSES,
            "sna_c": args.c, "sna_beta": args.beta, "intervention": args.iv,
            "optimizer": args.optimizer, "data_sha256": mnist.sha256,
            "subset_sha256": {str(s): hashlib.sha256(
                np.sort(subset_idx(s).numpy()).tobytes()).hexdigest() for s in seeds},
            "rng_roles": ["rl_subset", "rl_labels", "rl_batch", "init"],
            "device": str(device), "torch": torch.__version__,
            "wall_clock_s": time.time() - t_start, "divergences": divs}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"\nwrote {out}/per_task.csv  ({len(rows)} rows, {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()

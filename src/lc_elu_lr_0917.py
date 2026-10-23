#!/usr/bin/env python3
"""lc_elu_lr_0917 -- Lillo & Cheney's Random Label MNIST box, ELU alpha x learning rate (specs/spec_lc_elu_lr_0917.md).

    OMP_NUM_THREADS=1 python3 src/lc_elu_lr_0917.py --arm E36_lr1e4 --seed 0 --tasks 150 \\
        --out results/lc_elu_lr_0917/runs/E36_lr1e4_s0

The box follows the random_label_mnist path of github.com/lute47lillo/activations_plasticity
(branch iclr2026, commit bdce354; src/case_cl_benchmarks/bench_main.py, bench_data_handlers.py,
bench_models.py), re-implemented here because that repository carries no licence:

  * the first 1200 MNIST training images, pixels / 255, flattened; labels: for every task a fresh
    uniform draw for the 1200 training images and then one for 500 validation images, all 50 tasks
    drawn from the global torch RNG right after torch.manual_seed(seed), before the model exists
  * MLP 784-100-100-10 = nn.Linear(784,100), nn.Linear(100,100), nn.Linear(100,10) built in that
    order (PyTorch default init), two separate activation modules (nn.ELU(alpha) here)
  * torch.optim.Adam(model.parameters(), lr) with its defaults, created once (moments persist)
  * 400 epochs x 75 steps per task, batch 16, the images in their FIXED order every epoch (no shuffle)
  * online accuracy = pre-update batch accuracy, averaged over the task; their "Total Average Online
    Task Accuracy" is the mean over all 50 tasks' steps (= the mean of the 50 task averages)

Tasks beyond 50 (the extension, spec section 2.3) take their labels from a separate generator, so
tasks 1-50 are bit-identical whatever --tasks is.  After every task the run writes per_task.csv,
units.npz and a checkpoint (resume is exact: nothing random happens during training).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
RUN_ID = "lc_elu_lr_0917"
DATA = REPO / "data" / "mnist" / "train-images-idx3-ubyte.gz"
SPEC = REPO / "specs" / "spec_lc_elu_lr_0917.md"
N_IMG = 1200
N_VAL = 500
N_CLASSES = 10
BATCH = 16
EPOCHS = 400
BLOCK = 50                      # online accuracy is also kept per 50-epoch block
MAIN_TASKS = 50                 # the paper's horizon: these labels come from the global RNG
LILLO_SHA = "bdce354782cd183d63550819550b33312506d3e3"

# arm -> (activation kind, parameter, learning rate); the 2x2 is the E* arms, R/LK08 are anchors
ARMS = {
    "E1_lr1e3": ("elu", 1.0, 1e-3),
    "E36_lr1e3": ("elu", 3.6, 1e-3),
    "E1_lr1e4": ("elu", 1.0, 1e-4),
    "E36_lr1e4": ("elu", 3.6, 1e-4),     # Lillo & Cheney Table E2, Random Label MNIST
    "R_lr1e4": ("relu", 0.0, 1e-4),      # Table E2: ReLU lr 1e-4 (Table 2: 20.03 +- 2.46)
    "LK08_lr1e3": ("leaky", 0.8, 1e-3),  # Table E2: Leaky-ReLU 0.8, lr 1e-3 (Table 2: 91.53 +- 0.18)
}


# --------------------------------------------------------------------------
# box
# --------------------------------------------------------------------------

def load_images(n: int = N_IMG) -> torch.Tensor:
    """The first n training images as float32 / 255 (torchvision's MNIST.data order)."""
    with gzip.open(DATA, "rb") as fh:
        raw = fh.read(16 + n * 784)
    assert int.from_bytes(raw[:4], "big") == 2051 and int.from_bytes(raw[4:8], "big") == 60000
    arr = np.frombuffer(raw[16:16 + n * 784], dtype=np.uint8).reshape(n, 784)
    return torch.from_numpy(arr.astype(np.float32)) / 255.0


def make_act(kind: str, param: float) -> nn.Module:
    if kind == "elu":
        return nn.ELU(alpha=param)
    if kind == "relu":
        return nn.ReLU()
    if kind == "leaky":
        return nn.LeakyReLU(negative_slope=param)
    raise ValueError(kind)


class Net(nn.Module):
    """bench_models.MLPBenchmarks for a plain activation: fc1, fc2, out and two activation instances."""

    def __init__(self, kind: str, param: float):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.out = nn.Linear(100, 10)
        self.activation = nn.ModuleList([make_act(kind, param), make_act(kind, param)])

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.activation[0](z1)
        z2 = self.fc2(a1)
        a2 = self.activation[1](z2)
        return self.out(a2), (z1, a1, z2, a2)


def ext_generator(seed: int) -> torch.Generator:
    h = hashlib.sha256(f"{RUN_ID}|ext_labels|{seed}".encode()).digest()
    g = torch.Generator()
    g.manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))
    return g


def build(arm: str, seed: int, n_tasks: int):
    """Seed, draw the labels as the handler does, then the model and the optimizer (bench_main order)."""
    kind, param, lr = ARMS[arm]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    labels = []
    for _ in range(MAIN_TASKS):
        train_labels = torch.randint(0, N_CLASSES, (N_IMG,))
        torch.randint(0, N_CLASSES, (N_VAL,))          # the validation labels: drawn, not used
        labels.append(train_labels)
    model = Net(kind, param)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = ext_generator(seed)
    for _ in range(MAIN_TASKS, n_tasks):
        labels.append(torch.randint(0, N_CLASSES, (N_IMG,), generator=g))
        torch.randint(0, N_CLASSES, (N_VAL,), generator=g)
    return model, opt, labels


def train_task(model, opt, crit, x, y, epochs: int = EPOCHS):
    """One task: fixed-order mini-batches, the accuracy of each batch before its update."""
    hits_block = []
    hits = 0
    finite = True
    for epoch in range(epochs):
        for i in range(0, x.shape[0], BATCH):
            x_batch = x[i:i + BATCH]
            y_batch = y[i:i + BATCH]
            logits, _ = model(x_batch)
            with torch.no_grad():
                hits += int((logits.argmax(1) == y_batch).sum())
            loss = crit(logits, y_batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            finite = finite and bool(torch.isfinite(loss))
        if (epoch + 1) % BLOCK == 0:
            hits_block.append(hits - sum(hits_block))
    return hits, hits_block, finite


# --------------------------------------------------------------------------
# readout at the end of a task (the net that starts the next task)
# --------------------------------------------------------------------------

def log_dphi(z64: torch.Tensor, kind: str, param: float) -> torch.Tensor:
    """log phi'(z) of the function (float64, no underflow): ELU alpha e^z below 0, 1 above."""
    zero = torch.zeros_like(z64)
    if kind == "elu":
        return torch.where(z64 > 0, zero, math.log(param) + z64)
    if kind == "relu":
        return torch.where(z64 > 0, zero, torch.full_like(z64, -math.inf))
    if kind == "leaky":
        return torch.where(z64 > 0, zero, torch.full_like(z64, math.log(param)))
    raise ValueError(kind)


def train_dphi(act: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """The derivative training uses: autograd through the same module on the same float32 z."""
    zc = z.detach().clone().requires_grad_(True)
    g, = torch.autograd.grad(act(zc).sum(), zc)
    return g


def neff_of(g: torch.Tensor) -> torch.Tensor:
    """(sum g)^2 / (n sum g^2) per unit (columns), 0 for a unit whose derivative is 0 on every image."""
    s1 = g.sum(0)
    s2 = g.square().sum(0)
    return torch.where(s1 > 0, s1 * s1 / (g.shape[0] * s2).clamp(min=1e-300), torch.zeros_like(s1))


def adam_small(opt, p: torch.Tensor, eps: float = 1e-8) -> float:
    """Share of p's coordinates with sqrt(v_hat) < eps (v_hat = v / (1 - beta2^step))."""
    st = opt.state.get(p)
    if not st:
        return float("nan")
    step = float(st["step"])
    b2 = opt.param_groups[0]["betas"][1]
    vhat = st["exp_avg_sq"].double() / (1.0 - b2 ** step)
    return float((vhat.sqrt() < eps).double().mean())


@torch.no_grad()
def readout(model: Net, opt, crit, x: torch.Tensor, y: torch.Tensor, kind: str, param: float):
    logits, (z1, a1, z2, a2) = model(x)
    n = x.shape[0]
    row = {"memo_acc": float((logits.argmax(1) == y).double().mean()),
           "loss_end": float(crit(logits, y))}
    units = {}
    ins = {1: x, 2: a1}
    for li, z, lin in ((1, z1, model.fc1), (2, z2, model.fc2)):
        with torch.enable_grad():
            g = train_dphi(model.activation[li - 1], z).double()
        z64 = z.double()
        lg = log_dphi(z64, kind, param)
        lse1 = torch.logsumexp(lg, 0)
        lse2 = torch.logsumexp(2 * lg, 0)
        mu = ins[li].double().mean(0)
        W = lin.weight.double()
        b = lin.bias.double()
        u = {
            "zbar": z64.mean(0),
            "sigma": z64.std(0, unbiased=False),
            "U": z64.amax(0),
            "gtr": g.mean(0),
            "zero": (g == 0).double().mean(0),
            "neffT": neff_of(g),
            "loggmean": lse1 - math.log(n),
            "neff": torch.exp(2 * lse1 - lse2 - math.log(n)),
            "pplus": (z64 > 0).double().mean(0),
            "wnorm": W.norm(dim=1),
            "b": b,
            "wmu": W @ mu,                      # zbar = w . mu + b up to the float32 forward's rounding
        }
        for k, v in u.items():
            units[f"{k}_l{li}"] = v.numpy()
        row[f"mu_norm_l{li}"] = float(mu.norm())
        for k in ("zbar", "sigma", "U", "gtr", "zero", "neffT", "loggmean", "neff", "pplus", "wnorm", "b"):
            row[f"{k}_l{li}"] = float(u[k].mean())
        row[f"zbar_med_l{li}"] = float(u["zbar"].median())
        row[f"dead_l{li}"] = float((u["gtr"] == 0).double().mean())
        row[f"adam_small_l{li}"] = adam_small(opt, lin.weight)
        if li == 2:
            e = mu / mu.norm()
            q = W @ e
            units["q_l2"] = q.numpy()
            row["q_l2"] = float(q.mean())
            row["cos_l2"] = float((q / u["wnorm"]).mean())
    row["wnorm_l3"] = float(model.out.weight.double().norm(dim=1).mean())
    return row, units


# --------------------------------------------------------------------------
# provenance helpers
# --------------------------------------------------------------------------

def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"git_hash": run("rev-parse", "HEAD"),
            "git_dirty": run("status", "--porcelain", "--", "src", "analysis/lc_elu_lr_0917",
                             "specs/spec_lc_elu_lr_0917.md").splitlines()}


def sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def write_table(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(repr(r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------

def run(arm: str, seed: int, n_tasks: int, out: Path, epochs: int = EPOCHS, threads: int = 1,
        stop_after: int | None = None) -> dict:
    kind, param, lr = ARMS[arm]
    out.mkdir(parents=True, exist_ok=True)
    prov_path = out / "provenance.json"
    ck_path = out / "ckpts" / "last.pt"
    torch.set_num_threads(threads)
    started = dt.datetime.now().isoformat(timespec="seconds")
    gstate = git_state()
    x = load_images()
    model, opt, labels = build(arm, seed, n_tasks)
    crit = torch.nn.CrossEntropyLoss()
    init_sha = hashlib.sha256("".join(sha_tensor(p) for p in model.parameters()).encode()).hexdigest()
    label_sha = [sha_tensor(l) for l in labels]
    rows, units, blocks, t0 = [], {}, [], 1
    history = []
    if ck_path.exists():                               # exact resume
        ck = torch.load(ck_path, weights_only=False)
        assert ck["arm"] == arm and ck["seed"] == seed and ck["epochs"] == epochs
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rows, units, blocks, history = ck["rows"], ck["units"], ck["blocks"], ck["history"]
        t0 = len(rows) + 1
    history.append({"started": started, "from_task": t0, **gstate,
                    "threads": torch.get_num_threads(), "flush_denormal": False})
    steps = epochs * (N_IMG // BATCH)
    status = "COMPLETE"
    wall0 = time.time()
    for t in range(t0, n_tasks + 1):
        ts = time.time()
        y = labels[t - 1]
        hits, hb, finite = train_task(model, opt, crit, x, y, epochs)
        if not finite or not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            status = "DIVERGED"
            break
        r, u = readout(model, opt, crit, x, y, kind, param)
        counts = torch.bincount(y, minlength=N_CLASSES)
        row = {"arm": arm, "kind": kind, "param": param, "lr": lr, "seed": seed, "task": t,
               "online": hits / (steps * BATCH), "floor": float(counts.max()) / N_IMG,
               **r, "sec": time.time() - ts}
        rows.append(row)
        for k, v in u.items():
            units.setdefault(k, []).append(v)
        blocks.append([h / ((BLOCK * (N_IMG // BATCH)) * BATCH) for h in hb])
        write_table(out / "per_task.csv", rows)
        np.savez(out / "units.npz", task=np.arange(1, len(rows) + 1),
                 online_block=np.array(blocks), **{k: np.stack(v) for k, v in units.items()})
        ck_path.parent.mkdir(exist_ok=True)
        torch.save({"arm": arm, "seed": seed, "epochs": epochs, "model": model.state_dict(),
                    "opt": opt.state_dict(), "rows": rows, "units": units, "blocks": blocks,
                    "history": history}, str(ck_path) + ".tmp")
        os.replace(str(ck_path) + ".tmp", ck_path)
        print(f"{dt.datetime.now():%H:%M:%S} {arm} s{seed} t{t} online {row['online']:.4f} "
              f"zbar2 {row['zbar_l2']:+.2f} mu2 {row['mu_norm_l2']:.2f} neffT2 {row['neffT_l2']:.3f} "
              f"{row['sec']:.1f}s", flush=True)
        if stop_after is not None and t >= stop_after:
            status = "STOPPED"
            break
    done = len(rows)
    if status != "DIVERGED":
        status = "COMPLETE" if done == n_tasks else "INCOMPLETE"
    prov = {"run_id": RUN_ID, "arm": arm, "kind": kind, "param": param, "lr": lr, "seed": seed,
            "n_tasks": n_tasks, "tasks_done": done, "status": status,
            "epochs_per_task": epochs, "steps_per_task": steps, "batch": BATCH, "n_images": N_IMG,
            "dims": [784, 100, 100, 10], "optimizer": "torch.optim.Adam(defaults)", "shuffle": False,
            "main_tasks_from_global_rng": MAIN_TASKS, "lillo_cheney_commit": LILLO_SHA,
            "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
            "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest() if SPEC.exists() else None,
            "init_sha256": init_sha, "label_sha256": label_sha,
            "online_lifetime_t1_50": float(np.mean([r["online"] for r in rows[:MAIN_TASKS]])) if done >= MAIN_TASKS else None,
            "torch": torch.__version__, "python": platform.python_version(), "cpu": cpu_model(),
            "host": platform.node(), "history": history, "finished": dt.datetime.now().isoformat(timespec="seconds"),
            "wall_clock_s_this_segment": time.time() - wall0,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "argv": sys.argv}
    if prov["status"] in ("COMPLETE", "DIVERGED"):
        prov_path.write_text(json.dumps(prov, indent=1))
    else:
        (out / "progress.json").write_text(json.dumps(prov, indent=1))
    return prov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tasks", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stop-after", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = run(a.arm, a.seed, a.tasks, Path(a.out), a.epochs, a.threads, a.stop_after)
    print(json.dumps({k: p[k] for k in ("status", "tasks_done", "online_lifetime_t1_50", "peak_rss_mib")}))


if __name__ == "__main__":
    main()

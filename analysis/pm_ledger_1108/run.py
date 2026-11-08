"""pm_ledger_1108 -- Permuted MNIST run that saves the first layer task by task.

spec: specs/spec_pm_ledger_1108.md

The box is src/pmnist_0905.py.  Its run_one does not expose the per-task weights
or the online accuracy, so the update loop is copied here (same operations in the
same order) and checked against run_one bit by bit (--check-ident, spec S-ident).

Saves under --out:
  W1.npy   (tasks+1, 100, 784) float32   index 0 = initial weights
  b1.npy   (tasks+1, 100)      float32
  perm.npy (tasks+1, 784)      int16     index 0 = -1
  per_task.csv  online accuracy (mean over the 625 pre-update minibatches) and
                the box's evaluate() row at each task end
  fresh.csv     online accuracy of a net trained from the seed's initial weights
                and zero Adam state on the same task and the same minibatch order
  idx_sha.json  sha256 of each task's drawn indices (for the post-hoc rebuild)
  provenance.json

Usage:
  .venv/bin/python analysis/pm_ledger_1108/run.py --arm none --seed 0 --tasks 500 --out DIR
  .venv/bin/python analysis/pm_ledger_1108/run.py --check-ident
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""            # spec: CPU only

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import pmnist_0905 as B                            # noqa: E402

B.DATA_DIR = Path(os.environ.get("PM_MNIST_DIR",
                                 "/home/issan/Projects/claude/proj_004_drift/data/mnist"))

ARMS = {"none": None, "l2init": 1e-3}              # lam of the L2-to-theta0 penalty
ACT = B.ARMS["R"]                                  # ReLU
LR = 1e-3
FRESH_TASKS = (100, 200, 300, 400, 500)


def zero_adam(params):
    return ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])


def train_task(params, adam, lam, p0, xs, ys, lr):
    """625 updates; same operations and order as src/pmnist_0905.run_one.

    Returns the number of correctly classified examples, each read before its
    update, and the first non-finite step (or -1).
    """
    n_correct = torch.zeros((), dtype=torch.long)
    bad_step = torch.full((), -1, dtype=torch.long)
    for s in range(B.STEPS_PER_TASK):
        j = s % B.STEPS_PER_TASK
        xb = xs[j * B.BATCH:(j + 1) * B.BATCH]
        yb = ys[j * B.BATCH:(j + 1) * B.BATCH]
        out = B.forward(params, xb, ACT)
        loss = torch.nn.functional.cross_entropy(out[4], yb)
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            n_correct += (out[4].argmax(1) == yb).sum()
            bad = ~torch.isfinite(loss)
            bad_step = torch.where((bad_step < 0) & bad, torch.tensor(s), bad_step)
            if lam is not None:
                grads = [gr + 2.0 * lam * (q - p0[i])
                         for i, (q, gr) in enumerate(zip(params, grads))]
            m, v, tc = adam
            tc[0] += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            c1 = 1 - b1 ** tc[0]
            c2 = 1 - b2 ** tc[0]
            for p, gr, mi, vi in zip(params, grads, m, v):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
    return n_correct.item(), int(bad_step)


def task_batches(mnist, g_perm, g_data, g_batch):
    perm = torch.randperm(784, generator=g_perm)
    idx = B.stratified_draw(mnist, g_data)
    order = torch.randperm(B.TASK_EXAMPLES, generator=g_batch)
    xs = mnist.train_x[idx][:, perm][order]
    ys = mnist.train_y[idx][order]
    return perm, idx, xs, ys


def run_loop(arm, seed, n_tasks, mnist, out=None, on_task=None):
    device = torch.device("cpu")
    lam = ARMS[arm]
    params = B.init_params(seed, device)
    p0 = [q.detach().clone() for q in params] if lam is not None else None
    adam = zero_adam(params)
    g_perm, g_data, g_batch = B.stream("perm", seed), B.stream("data", seed), B.stream("batch", seed)

    saving = out is not None
    if saving:
        out.mkdir(parents=True, exist_ok=True)
        W1 = np.lib.format.open_memmap(out / "W1.npy", mode="w+", dtype=np.float32,
                                       shape=(n_tasks + 1, 100, 784))
        b1 = np.lib.format.open_memmap(out / "b1.npy", mode="w+", dtype=np.float32,
                                       shape=(n_tasks + 1, 100))
        perms = np.lib.format.open_memmap(out / "perm.npy", mode="w+", dtype=np.int16,
                                          shape=(n_tasks + 1, 784))
        perms[0] = -1
        W1[0] = params[0].detach().numpy()
        b1[0] = params[1].detach().numpy()
        W1.flush()
    rows, fresh_rows, idx_sha, diverged = [], [], {}, {"diverged": False}
    t0 = time.time()

    for t in range(1, n_tasks + 1):
        perm, idx, xs, ys = task_batches(mnist, g_perm, g_data, g_batch)
        n_correct, bs = train_task(params, adam, lam, p0, xs, ys, LR)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t, step=(t - 1) * B.STEPS_PER_TASK + max(bs, 0))
            rows.append({"arm": arm, "seed": seed, "task": t, "online": float("nan")})
            break
        online = n_correct / B.TASK_EXAMPLES
        row = {"arm": arm, "seed": seed, "task": t, "online": online}
        row.update(B.evaluate(params, mnist, perm, ACT))
        rows.append(row)
        if on_task is not None:
            on_task(t, params)
        if saving:
            W1[t] = params[0].detach().numpy()
            b1[t] = params[1].detach().numpy()
            perms[t] = perm.numpy().astype(np.int16)
            idx_sha[str(t)] = hashlib.sha256(idx.numpy().tobytes()).hexdigest()
            if t % 25 == 0 or t == n_tasks:
                W1.flush(); b1.flush(); perms.flush()
                B.write_csv(out / "per_task.csv", rows)
                (out / "idx_sha.json").write_text(json.dumps(idx_sha))
                if t % 100 == 0 or t == n_tasks:
                    print(f"[{time.time() - t0:7.1f}s] {arm} seed{seed} task {t} online {online:.4f}",
                          flush=True)
        if t in FRESH_TASKS:
            F = B.init_params(seed, device)
            A0 = zero_adam(F)
            p0f = [q.detach().clone() for q in F] if lam is not None else None
            n_correct_f, bs_f = train_task(F, A0, lam, p0f, xs, ys, LR)
            fresh_rows.append({"arm": arm, "seed": seed, "task": t,
                               "online_fresh": n_correct_f / B.TASK_EXAMPLES,
                               "online_cont": online,
                               "fresh_bad_step": bs_f})
            if saving:
                B.write_csv(out / "fresh.csv", fresh_rows)
    return rows, fresh_rows, idx_sha, diverged, time.time() - t0


def check_ident():
    """S-ident: same parameters, bit for bit, as src/pmnist_0905.run_one."""
    device = torch.device("cpu")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    mnist = B.Mnist(device)
    report = {}
    for arm, iv in (("none", "none"), ("l2init", "l2init:1e-3")):
        seen = {}
        real_eval = B.evaluate

        def capture(params, mnist_, perm, act):
            seen[len(seen) + 1] = [p.detach().clone() for p in params]
            return real_eval(params, mnist_, perm, act)

        B.evaluate = capture
        try:
            B.run_one("R", 0, LR, 3, mnist, device, optimizer="adam", iv=iv)
        finally:
            B.evaluate = real_eval
        mine = {}

        def snap(t, params):
            mine[t] = [p.detach().clone() for p in params]

        run_loop(arm, 0, 3, mnist, on_task=snap)
        ok = all(all(torch.equal(a, b) for a, b in zip(seen[t], mine[t])) for t in (1, 2, 3))
        report[arm] = {"bit_identical": bool(ok), "tasks_checked": 3}
        print(arm, report[arm], flush=True)
    if not all(v["bit_identical"] for v in report.values()):
        raise SystemExit("S-ident failed")
    print("S-ident PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--tasks", type=int, default=500)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check-ident", action="store_true")
    args = ap.parse_args()
    if args.check_ident:
        check_ident()
        return
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    mnist = B.Mnist(device)
    rows, fresh_rows, idx_sha, diverged, wall = run_loop(args.arm, args.seed, args.tasks, mnist, out=args.out)
    B.write_csv(args.out / "per_task.csv", rows)
    if fresh_rows:
        B.write_csv(args.out / "fresh.csv", fresh_rows)
    (args.out / "idx_sha.json").write_text(json.dumps(idx_sha))
    prov = {"run_id": "pm_ledger_1108", "arm": args.arm, "seed": args.seed, "lr": LR, "lam": ARMS[args.arm],
            "activation": "ReLU (box arm R)", "n_tasks": args.tasks, "dims": list(B.DIMS), "batch": B.BATCH,
            "steps_per_task": B.STEPS_PER_TASK, "task_examples": B.TASK_EXAMPLES,
            "git_hash": B.git_hash(), "device": "cpu", "torch": torch.__version__,
            "data_sha256": mnist.sha256, "wall_clock_s": wall,
            "max_rss_MB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "fresh_tasks": list(FRESH_TASKS), "diverged": diverged,
            "box": "src/pmnist_0905.py (loop copied; S-ident)"}
    (args.out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"done {args.arm} seed{args.seed}: {wall:.1f}s, max RSS {prov['max_rss_MB']:.0f} MB", flush=True)


if __name__ == "__main__":
    main()

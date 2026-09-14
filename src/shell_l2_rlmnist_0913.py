#!/usr/bin/env python3
"""Random Label MNIST: none / l2 / l2init / shell under R and SNA (specs/spec_shell_l2_rlmnist_0913.md).

    OMP_NUM_THREADS=1 python3 src/shell_l2_rlmnist_0913.py --act SNA --reg shell:1e-3 \
        --seeds 0 --out results/shell_l2_rlmnist_0913/runs/SNA_shell_s0

The box is pmnist_rlmnist_0906's and is imported, not copied: the 1200-image subset, the
task labels and evaluate_rl come from src/pmnist_rlmnist_0906.py; init, forward, the rng
streams and AdaptiveSnake from the host src/pmnist_0905.py.  Neither file is modified.

run_one is 0906's run_one line for line (Adam only), with three additions:
  * the shell regulariser (spec 2.3).  Its gradient is taken by autograd from the loss
    R_shell and added where 0906 adds l2 / l2init; the l2 / l2init line is 0906's, verbatim;
  * per-task logging of ||p||/||p0||, cos(p, p0) and the penalties for all six tensors;
  * sha256 of the data streams actually consumed and of the final state (spec 2.4, check S5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Determinism must be requested before the first CUDA workspace is allocated.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H              # host; not touched
from src import pmnist_rlmnist_0906 as RL     # 0906 runner; not touched

EXPERIMENT = "shell_l2_rlmnist_0913"
ACTS = ("R", "SNA")
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")
N_IMAGES = RL.N_IMAGES                      # 1200
BATCH = RL.BATCH                            # 16
STEPS_PER_EPOCH = RL.STEPS_PER_EPOCH        # 75
LAM_REF = 1e-3          # lambda of the diagnostic penalties logged for every arm (spec 3.2)
# spec 2.3: float32 S + SHELL_EPS2 rounds back to S whenever S > SHELL_EPS2 * 2**25 = 3.4e-5,
# so the norm is exact over the whole operating range; it only keeps d||p||/dp finite at p = 0.
SHELL_EPS2 = 1e-12


@dataclass(frozen=True)
class Reg:
    kind: str = "none"
    lam: float = 0.0

    @staticmethod
    def parse(s: str) -> "Reg":
        p = s.split(":")
        if p == ["none"]:
            return Reg("none")
        if p[0] in ("l2", "l2init", "shell") and len(p) == 2:
            return Reg(p[0], float(p[1]))
        raise ValueError(f"--reg must be none|l2:<lam>|l2init:<lam>|shell:<lam>, got {s!r}")


# --------------------------------------------------------------------------
# shell regulariser (spec 2.3)
# --------------------------------------------------------------------------

def shell_norm(p: torch.Tensor) -> torch.Tensor:
    """||p|| of one tensor as the shell loss sees it: sqrt(sum p^2 + eps^2)."""
    return torch.sqrt((p * p).sum() + SHELL_EPS2)


@torch.no_grad()
def shell_reference(p0: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[bool]]:
    """r0 per tensor from the same function as the live norm, and which p0 are identically 0."""
    return [shell_norm(q) for q in p0], [bool((q == 0).all()) for q in p0]


def shell_penalty(params, r0, zero0, lam: float) -> torch.Tensor:
    """R_shell = sum_p lam * (||p|| - ||p0||)^2, one norm per tensor; lam * ||p||^2 where p0 == 0."""
    total = None
    for p, r, z in zip(params, r0, zero0):
        t = lam * (p * p).sum() if z else lam * (shell_norm(p) - r) ** 2
        total = t if total is None else total + t
    return total


# --------------------------------------------------------------------------
# logging (spec 3.2): float64 recomputation from the float32 weights, exact norms
# --------------------------------------------------------------------------

@torch.no_grad()
def layer_rows(params, ref0, reg: Reg, act_name: str, seed: int, task: int) -> list[dict]:
    rows = []
    for name, p, q0 in zip(TENSORS, params, ref0):
        a, b = p.detach().double(), q0.detach().double()
        sq = float((a * a).sum())
        sq0 = float((b * b).sum())
        n, n0 = math.sqrt(sq), math.sqrt(sq0)
        dot = float((a * b).sum())
        dist2 = float(((a - b) * (a - b)).sum())
        ratio = n / n0 if n0 > 0 else float("nan")
        cos = dot / (n * n0) if n > 0 and n0 > 0 else float("nan")
        pen = {"l2": LAM_REF * sq,
               "l2init": LAM_REF * dist2,
               "shell": LAM_REF * (n - n0) ** 2 if n0 > 0 else LAM_REF * sq}
        own = 0.0 if reg.kind == "none" else pen[reg.kind] * (reg.lam / LAM_REF)
        rows.append({"act": act_name, "reg": reg.kind, "lam": reg.lam, "seed": seed, "task": task,
                     "tensor": name, "ndim": p.dim(), "numel": p.numel(),
                     "sq_norm": sq, "norm": n, "norm0": n0, "norm_ratio": ratio,
                     "cos_w0": cos, "dist_w0": math.sqrt(dist2),
                     "pen_l2": pen["l2"], "pen_l2init": pen["l2init"], "pen_shell": pen["shell"],
                     "pen_angular": 2.0 * LAM_REF * n * n0 * (1.0 - cos) if n > 0 and n0 > 0 else 0.0,
                     "reg_loss": own})
    return rows


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


def state_sha256(params, adam, act) -> str:
    h = hashlib.sha256()
    for q in params:
        h.update(_bytes(q))
    m, v, tc = adam
    for q in (*m, *v):
        h.update(_bytes(q))
    h.update(str(tc[0]).encode())
    if isinstance(act, H.AdaptiveSnake):
        for q in act.V:
            h.update(_bytes(q))
    return h.hexdigest()


# --------------------------------------------------------------------------
# one (act, reg, seed) run
# --------------------------------------------------------------------------

def run_one(act_name: str, reg_s: str, seed: int, lr: float, n_tasks: int, mnist: H.Mnist,
            device: torch.device, epochs: int = 400, c: float = 0.6, beta: float = 0.01,
            debug: dict | None = None, progress: bool = False) -> tuple[list[dict], list[dict], dict]:
    """debug (checks only): captures the init, subset, labels, batch orders, task-end weights and,
    at the (task, step) pairs in debug["capture"], the shell loss / raw gradient before the update."""
    t_start = time.time()
    act = H.ARMS[act_name]
    if act.kind == "adaptive_snake":
        act = H.AdaptiveSnake(c, beta, device)        # fresh statistics per run
    params = H.init_params(seed, device)              # host init: bit-identical per seed
    if debug is not None:
        debug["init"] = [q.detach().cpu().clone() for q in params]
    reg = Reg.parse(reg_s)
    p0 = [q.detach().clone() for q in params] if reg.kind == "l2init" else None
    ref0 = [q.detach().clone() for q in params]       # logging and the shell anchor; never read by l2/l2init
    if reg.kind == "shell":
        r0, zero0 = shell_reference(ref0)
    capture = debug.get("capture", ()) if debug is not None else ()
    # Adam moments live for the whole run: resetting them between tasks would break
    # the continual definition the same way resetting weights does.
    adam = ([torch.zeros_like(q) for q in params],
            [torch.zeros_like(q) for q in params], [0])

    idx = RL.subset_idx(seed).to(device)
    x = mnist.train_x[idx]                            # the task's inputs, fixed forever
    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
    if debug is not None:
        debug["subset"] = idx.cpu().clone()
    hashes = {"init_sha256": hashlib.sha256(b"".join(_bytes(q) for q in params)).hexdigest(),
              "subset_idx_sha256": hashlib.sha256(_bytes(idx)).hexdigest()}
    h_lab, h_batch = hashlib.sha256(), hashlib.sha256()
    spt = STEPS_PER_EPOCH * epochs
    rows, lrows = [], layer_rows(params, ref0, reg, act_name, seed, 0)
    diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "act": act_name, "reg": reg_s}

    for t in range(1, n_tasks + 1):
        y = RL.task_labels(g_lab).to(device)          # new labelling, same images
        h_lab.update(_bytes(y))
        if debug is not None:
            debug.setdefault("labels", []).append(y.cpu().clone())
        acc_sum = torch.zeros((), device=device)
        bad_step = torch.full((), -1, dtype=torch.long, device=device)

        for e in range(epochs):
            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)
            h_batch.update(_bytes(order))
            if debug is not None:
                debug.setdefault("orders", []).append(order.cpu().clone())
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
                if reg.kind == "shell":
                    # the loss R_shell, differentiated at the same pre-update weights as the CE loss
                    r_shell = shell_penalty(params, r0, zero0, reg.lam)
                    g_shell = torch.autograd.grad(r_shell, params)
                    if (t, s) in capture:
                        debug.setdefault("shell", {})[(t, s)] = {
                            "params": [q.detach().cpu().clone() for q in params],
                            "loss": r_shell.detach().cpu().clone(),
                            "grads": [q.detach().cpu().clone() for q in g_shell]}
                with torch.no_grad():
                    bad = ~torch.isfinite(loss)
                    bad_step = torch.where((bad_step < 0) & bad,
                                           torch.tensor(s, device=device), bad_step)
                    if reg.kind in ("l2", "l2init"):
                        grads = [gr + 2.0 * reg.lam * (q - (p0[i] if p0 is not None else 0.0))
                                 for i, (q, gr) in enumerate(zip(params, grads))]
                    elif reg.kind == "shell":
                        grads = [gr + gs for gr, gs in zip(grads, g_shell)]
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
            rows.append({"act": act_name, "reg": reg.kind, "lam": reg.lam, "lr": lr, "seed": seed,
                         "task": t, "online_acc": float("nan")})
            break                                     # drop, never rescue
        m = RL.evaluate_rl(params, x, y, act)
        lr_rows = layer_rows(params, ref0, reg, act_name, seed, t)
        lrows += lr_rows
        rows.append({"act": act_name, "reg": reg.kind, "lam": reg.lam, "lr": lr, "seed": seed, "task": t,
                     "online_acc": float(acc_sum) / spt, "memo_acc": m["acc"], **m,
                     "reg_loss": sum(r["reg_loss"] for r in lr_rows),
                     "pen_l2_total": sum(r["pen_l2"] for r in lr_rows),
                     "pen_l2init_total": sum(r["pen_l2init"] for r in lr_rows),
                     "pen_shell_total": sum(r["pen_shell"] for r in lr_rows)})
        if debug is not None:
            debug.setdefault("task_end_params", []).append([q.detach().cpu().clone() for q in params])
        if progress:
            print(f"[{time.time() - t_start:8.1f}s] {act_name} {reg_s} seed={seed} task {t}/{n_tasks}",
                  flush=True)

    info = {**hashes, "labels_sha256": h_lab.hexdigest(), "batch_sha256": h_batch.hexdigest(),
            "final_state_sha256": state_sha256(params, adam, act),
            "tasks_completed": len([r for r in rows if r["online_acc"] == r["online_acc"]]),
            "wall_clock_s": time.time() - t_start, "divergence": diverged}
    return rows, lrows, info


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_dirty(paths: list[str]) -> bool | None:
    try:
        r = subprocess.run(["git", "-C", str(H.REPO), "status", "--porcelain", "--", *paths],
                           capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True, choices=ACTS)
    ap.add_argument("--reg", required=True, help="none | l2:<lam> | l2init:<lam> | shell:<lam>")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400,
                    help="passes over the 1200 examples per task (75 steps each)")
    ap.add_argument("--c", type=float, default=0.6, help="SNA: alpha_i = c / W_i")
    ap.add_argument("--beta", type=float, default=0.01, help="SNA: EMA rate of var(z_i)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    seeds = [int(s) for s in args.seeds.split(",")]
    reg = Reg.parse(args.reg)                          # fail fast on a bad spec
    device = H.setup(args.device)
    t_start = time.time()
    mnist = H.Mnist(device)
    assert mnist.train_x.shape[0] == RL.TRAIN_N, mnist.train_x.shape

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").unlink(missing_ok=True)  # written last: its presence marks a finished run

    rows, lrows, runs = [], [], {}
    for seed in seeds:
        r, lr_, info = run_one(args.act, args.reg, seed, args.lr, args.tasks, mnist, device,
                               epochs=args.epochs, c=args.c, beta=args.beta, progress=True)
        rows += r
        lrows += lr_
        runs[str(seed)] = info
        print(f"[{time.time() - t_start:8.1f}s] done {args.act} {args.reg} seed={seed} "
              f"tasks={info['tasks_completed']} {info['wall_clock_s']:.1f}s"
              f"{'  DIVERGED@step ' + str(info['divergence']['step']) if info['divergence']['diverged'] else ''}",
              flush=True)
    H.write_csv(out / "per_task.csv", rows)
    H.write_csv(out / "layer_metrics.csv", lrows)

    me = Path(__file__).resolve()
    prov = {"run_id": EXPERIMENT, "git_hash": H.git_hash(),
            "git_dirty_code": git_dirty(["src", "analysis/shell_l2_rlmnist_0913"]),
            "argv": sys.argv, "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "act": args.act, "reg": reg.kind, "lam": reg.lam, "seeds": seeds, "lr": args.lr,
            "optimizer": "adam", "n_tasks": args.tasks, "epochs_per_task": args.epochs,
            "batch": BATCH, "steps_per_task": STEPS_PER_EPOCH * args.epochs,
            "n_images": N_IMAGES, "dims": list(H.DIMS), "sna_c": args.c, "sna_beta": args.beta,
            "shell_eps2": SHELL_EPS2, "lam_ref_logged_penalties": LAM_REF,
            "data_sha256": mnist.sha256,
            "subset_sha256_sorted": {str(s): hashlib.sha256(
                np.sort(RL.subset_idx(s).numpy()).tobytes()).hexdigest() for s in seeds},
            "code_sha256": {"src/shell_l2_rlmnist_0913.py": file_sha256(me),
                            "src/pmnist_0905.py": file_sha256(H.REPO / "src" / "pmnist_0905.py"),
                            "src/pmnist_rlmnist_0906.py": file_sha256(H.REPO / "src" / "pmnist_rlmnist_0906.py")},
            "device": str(device), "runs": runs, "wall_clock_s": time.time() - t_start,
            "divergences": [i["divergence"] for i in runs.values() if i["divergence"]["diverged"]]}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"wrote {out}  ({len(rows)} task rows, {len(lrows)} layer rows, {time.time() - t_start:.1f}s)",
          flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Random Label MNIST: per-unit centered-norm caps on the hidden layers (specs/spec_wcap_rlmnist_0914.md).

    OMP_NUM_THREADS=1 python3 src/wcap_rlmnist_0914.py --act LR --arm capT1 --seeds 0 \
        --out results/wcap_rlmnist_0914/runs/LR_capT1_s0

The box is pmnist_rlmnist_0906's and is imported, not copied: the 1200-image subset, the task labels
and evaluate_rl come from src/pmnist_rlmnist_0906.py; init, forward, the rng streams and the
activations from the host src/pmnist_0905.py; the l2 / l2init line, layer_rows and state_sha256 from
src/shell_l2_rlmnist_0913.py.  None of the three files is modified.

run_one is 0913's run_one (itself 0906's, Adam only) without the shell branch, with three additions:
  * the cap (spec 2.3): right after every Adam update, each row of W1 and W2 whose centered norm
    ||W_i - m_i|| exceeds r_i is rescaled about its row mean to r_i; rows at or below the cap are not
    written.  capT1: r_i = the run's own task-1-end norm, active from the first update of task 2.
    cap2: r_i = 2 ||W_i(0) - m_i(0)||, active from the first update of task 1;
  * per-task centered-norm logging (spec 3.1) and units.npz (spec 3.3), float64 from the float32 weights;
  * sha256 of the data streams actually consumed, of the task-1-end state and of the final state (spec 2.4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
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
from src import pmnist_0905 as H                 # host; not touched
from src import pmnist_rlmnist_0906 as RL        # 0906 runner; not touched
from src import shell_l2_rlmnist_0913 as SH      # 0913 runner; not touched

EXPERIMENT = "wcap_rlmnist_0914"
ACTS = ("LR", "R")
ARMS = ("ref", "l2", "l2init", "capT1", "cap2")
N_IMAGES = RL.N_IMAGES                      # 1200
BATCH = RL.BATCH                            # 16
STEPS_PER_EPOCH = RL.STEPS_PER_EPOCH        # 75
LAM = 1e-3                                  # l2 / l2init (spec 2.2)
CAP_MULT = 2.0                              # cap2: r_i = 2 ||W~_i(0)|| (spec 2.2)
CAP_PARAMS = (0, 2)                         # W1, W2 in the host's [W1, b1, W2, b2, W3, b3]
REL_TOL = 1e-4                              # spec 3.1: exceed / bind fractions (d * eps32 <= 9.3e-5)
QS = (0.1, 0.5, 0.9)                        # quantiles, linear interpolation (torch.quantile / np.quantile)


@dataclass(frozen=True)
class Arm:
    name: str

    def __post_init__(self):
        if self.name not in ARMS:
            raise ValueError(f"--arm must be one of {ARMS}, got {self.name!r}")

    @property
    def reg(self) -> SH.Reg:
        """The penalty this arm adds to the gradient; cap arms add none."""
        return SH.Reg(self.name, LAM) if self.name in ("l2", "l2init") else SH.Reg("none")

    @property
    def cap_start_task(self) -> int | None:
        return {"capT1": 2, "cap2": 1}.get(self.name)


# --------------------------------------------------------------------------
# the cap (spec 2.3)
# --------------------------------------------------------------------------

def _center(W: torch.Tensor):
    """Row mean m (rows, 1), centered rows W - m and their norms (rows, 1), all in W's dtype.
    The cap radius and the projection both go through this one function."""
    m = W.mean(dim=1, keepdim=True)
    Wt = W - m
    return m, Wt, torch.linalg.vector_norm(Wt, dim=1, keepdim=True)


@torch.no_grad()
def cap_radius(W: torch.Tensor, mult: float = 1.0) -> torch.Tensor:
    return (mult * _center(W)[2]).detach().clone()


@torch.no_grad()
def project_rows_(W: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """In place: rows with ||W_i - m_i|| > r_i become m_i + (W_i - m_i) r_i / ||W_i - m_i||.
    Rows at or below the cap keep every bit.  Returns (rows written, sum of n_i - r_i over them)."""
    m, Wt, n = _center(W)
    over = n > r
    W.copy_(torch.where(over, m + Wt * (r / n), W))
    return over.sum(), torch.where(over, n - r, torch.zeros_like(n)).sum()


# --------------------------------------------------------------------------
# logging (spec 3.1 / 3.3): float64 recomputation from the float32 weights
# --------------------------------------------------------------------------

@torch.no_grad()
def unit_arrays(params, x: torch.Tensor, act) -> dict[str, np.ndarray]:
    """Per-unit quantities of both hidden layers on the 1200 images (float64)."""
    z1, _, z2, _, _ = H.forward(params, x, act)
    out = {}
    for li, (k, z) in enumerate(((0, z1), (2, z2)), start=1):
        W = params[k].detach().double()
        m = W.mean(dim=1, keepdim=True)
        out[f"wt_l{li}"] = torch.linalg.vector_norm(W - m, dim=1).numpy()
        out[f"rowmean_l{li}"] = m[:, 0].numpy()
        out[f"b{li}"] = params[k + 1].detach().double().numpy()
        out[f"zbar_l{li}"] = z.double().mean(0).numpy()
        out[f"mob_l{li}"] = act.dphi(z).double().mean(0).numpy()
    return out


def task_fields(u: dict, u0: dict, r_t1: dict | None, r_cap: dict | None, proj: dict | None) -> dict:
    """spec 3.1 columns from one task-end unit_arrays() (u), the init one (u0), the run's own task-1-end
    norms (None before task 1 ends), the cap radii in float64 (None for non-cap arms) and the task's
    projection tally (None for non-cap arms)."""
    f = {}
    for li in (1, 2):
        w, w0 = u[f"wt_l{li}"], u0[f"wt_l{li}"]
        q = np.quantile(w, QS)
        f[f"wt_q10_l{li}"], f[f"wt_med_l{li}"], f[f"wt_q90_l{li}"] = float(q[0]), float(q[1]), float(q[2])
        f[f"wt_max_l{li}"] = float(w.max())
        f[f"wt_ratio0_med_l{li}"] = float(np.quantile(w / w0, 0.5))
        f[f"rowmean_abs_med_l{li}"] = float(np.quantile(np.abs(u[f"rowmean_l{li}"]), 0.5))
        f[f"b_med_l{li}"] = float(np.quantile(u[f"b{li}"], 0.5))
        f[f"exceed_t1_frac_l{li}"] = 0.0 if r_t1 is None else float(np.mean(w > r_t1[li] * (1 + REL_TOL)))
        f[f"exceed_2x0_frac_l{li}"] = float(np.mean(w > 2.0 * w0 * (1 + REL_TOL)))
        f[f"bind_frac_l{li}"] = float("nan") if r_cap is None else float(np.mean(w >= r_cap[li] * (1 - REL_TOL)))
        f[f"proj_rows_l{li}"] = float("nan") if proj is None else int(proj["rows"][li - 1])
        f[f"proj_removed_l{li}"] = float("nan") if proj is None else float(proj["removed"][li - 1])
    return f


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


# --------------------------------------------------------------------------
# one (act, arm, seed) run
# --------------------------------------------------------------------------

def run_one(act_name: str, arm_s: str, seed: int, lr: float, n_tasks: int, mnist: H.Mnist,
            device: torch.device, epochs: int = 400, debug: dict | None = None,
            progress: bool = False) -> tuple[list[dict], list[dict], dict, dict]:
    """debug (checks only): captures init, subset, labels, batch orders, task-end weights, per-task
    projection tallies and, at the (task, step) pairs in debug["capture"], the full state and batch
    just before the step and the state just after it (projection included)."""
    t_start = time.time()
    act = H.ARMS[act_name]
    params = H.init_params(seed, device)              # host init: bit-identical per seed
    if debug is not None:
        debug["init"] = [q.detach().cpu().clone() for q in params]
    arm = Arm(arm_s)
    reg = arm.reg
    p0 = [q.detach().clone() for q in params] if reg.kind == "l2init" else None
    ref0 = [q.detach().clone() for q in params]       # logging only; never read by the update
    cap_start = arm.cap_start_task
    cap_r = [cap_radius(params[k], CAP_MULT) for k in CAP_PARAMS] if arm.name == "cap2" else None
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
    rows, lrows = [], SH.layer_rows(params, ref0, reg, act_name, seed, 0)
    u0 = unit_arrays(params, x, act)
    units = {k: [v] for k, v in u0.items()}
    r_t1 = None
    r_cap64 = ({li: cap_r[li - 1][:, 0].double().numpy() for li in (1, 2)} if cap_r is not None else None)
    info_extra = {"init_centered": {f"wt_med_l{li}": float(np.quantile(u0[f"wt_l{li}"], 0.5)) for li in (1, 2)}}
    diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "act": act_name, "arm": arm_s}

    for t in range(1, n_tasks + 1):
        y = RL.task_labels(g_lab).to(device)          # new labelling, same images
        h_lab.update(_bytes(y))
        if debug is not None:
            debug.setdefault("labels", []).append(y.cpu().clone())
        acc_sum = torch.zeros((), device=device)
        bad_step = torch.full((), -1, dtype=torch.long, device=device)
        cap_on = cap_r is not None and t >= cap_start
        proj_rows = [torch.zeros((), dtype=torch.long) for _ in CAP_PARAMS]
        proj_removed = [torch.zeros((), dtype=torch.float64) for _ in CAP_PARAMS]

        for e in range(epochs):
            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)
            h_batch.update(_bytes(order))
            if debug is not None:
                debug.setdefault("orders", []).append(order.cpu().clone())
            xs, ys = x[order], y[order]               # reshuffled every epoch
            for j in range(STEPS_PER_EPOCH):
                s = e * STEPS_PER_EPOCH + j
                xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
                if (t, s) in capture:
                    m_, v_, tc_ = adam
                    debug.setdefault("steps", {})[(t, s)] = {
                        "before": {"params": [q.detach().clone() for q in params],
                                   "m": [q.clone() for q in m_], "v": [q.clone() for q in v_], "tc": tc_[0]},
                        "xb": xb.clone(), "yb": yb.clone(), "cap_on": cap_on,
                        "cap_r": None if cap_r is None else [r.clone() for r in cap_r]}
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
                    if reg.kind in ("l2", "l2init"):
                        grads = [gr + 2.0 * reg.lam * (q - (p0[i] if p0 is not None else 0.0))
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
                    if cap_on:
                        for li, k in enumerate(CAP_PARAMS):
                            nr, rem = project_rows_(params[k], cap_r[li])
                            proj_rows[li] += nr
                            proj_removed[li] += rem.double()
                if (t, s) in capture:
                    m_, v_, tc_ = adam
                    debug["steps"][(t, s)]["after"] = {
                        "params": [q.detach().clone() for q in params],
                        "m": [q.clone() for q in m_], "v": [q.clone() for q in v_], "tc": tc_[0]}

        bs = int(bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t, step=(t - 1) * spt + max(bs, 0))
            rows.append({"act": act_name, "arm": arm_s, "lam": reg.lam, "lr": lr, "seed": seed,
                         "task": t, "online_acc": float("nan")})
            break                                     # drop, never rescue
        if t == 1:
            info_extra["task1_end_state_sha256"] = SH.state_sha256(params, adam, act)
            if arm.name == "capT1":
                cap_r = [cap_radius(params[k]) for k in CAP_PARAMS]
                r_cap64 = {li: cap_r[li - 1][:, 0].double().numpy() for li in (1, 2)}
        mt = RL.evaluate_rl(params, x, y, act)
        u = unit_arrays(params, x, act)
        if t == 1:
            r_t1 = {li: u[f"wt_l{li}"].copy() for li in (1, 2)}
        for k, v_ in u.items():
            units[k].append(v_)
        proj = ({"rows": [int(r) for r in proj_rows], "removed": [float(r) for r in proj_removed]}
                if cap_r is not None else None)
        if debug is not None:
            debug.setdefault("proj", []).append(proj)
        lr_rows = SH.layer_rows(params, ref0, reg, act_name, seed, t)
        for r in lr_rows:
            r["arm"] = arm_s
        lrows += lr_rows
        rows.append({"act": act_name, "arm": arm_s, "lam": reg.lam, "lr": lr, "seed": seed, "task": t,
                     "online_acc": float(acc_sum) / spt, "memo_acc": mt["acc"], **mt,
                     "reg_loss": sum(r["reg_loss"] for r in lr_rows),
                     **task_fields(u, u0, r_t1, r_cap64 if cap_on else None, proj if cap_on else None)})
        if debug is not None:
            debug.setdefault("task_end_params", []).append([q.detach().cpu().clone() for q in params])
        if progress:
            print(f"[{time.time() - t_start:8.1f}s] {act_name} {arm_s} seed={seed} task {t}/{n_tasks}",
                  flush=True)

    for r in lrows[:6]:
        r["arm"] = arm_s                              # the task-0 rows
    arrays = {k: np.stack(v).astype(np.float32) for k, v in units.items()}
    for li in (1, 2):
        arrays[f"r_t1_l{li}"] = (r_t1[li] if r_t1 is not None else np.full(100, np.nan)).astype(np.float32)
        arrays[f"r_cap_l{li}"] = (r_cap64[li] if r_cap64 is not None else np.full(100, np.nan)).astype(np.float32)
    info = {**hashes, "labels_sha256": h_lab.hexdigest(), "batch_sha256": h_batch.hexdigest(),
            "final_state_sha256": SH.state_sha256(params, adam, act), **info_extra,
            "tasks_completed": len([r for r in rows if r["online_acc"] == r["online_acc"]]),
            "wall_clock_s": time.time() - t_start, "divergence": diverged}
    return rows, lrows, arrays, info


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True, choices=ACTS)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=400,
                    help="passes over the 1200 examples per task (75 steps each)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = H.setup(args.device)
    t_start = time.time()
    mnist = H.Mnist(device)
    assert mnist.train_x.shape[0] == RL.TRAIN_N, mnist.train_x.shape

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").unlink(missing_ok=True)  # written last: its presence marks a finished run

    rows, lrows, runs, arrays = [], [], {}, {}
    for seed in seeds:
        r, lr_, arr, info = run_one(args.act, args.arm, seed, args.lr, args.tasks, mnist, device,
                                    epochs=args.epochs, progress=True)
        rows += r
        lrows += lr_
        runs[str(seed)] = info
        arrays.update({f"s{seed}_{k}": v for k, v in arr.items()} if len(seeds) > 1 else arr)
        print(f"[{time.time() - t_start:8.1f}s] done {args.act} {args.arm} seed={seed} "
              f"tasks={info['tasks_completed']} {info['wall_clock_s']:.1f}s"
              f"{'  DIVERGED@step ' + str(info['divergence']['step']) if info['divergence']['diverged'] else ''}",
              flush=True)
    H.write_csv(out / "per_task.csv", rows)
    H.write_csv(out / "layer_metrics.csv", lrows)
    np.savez_compressed(out / "units.npz", **arrays)

    me = Path(__file__).resolve()
    src = H.REPO / "src"
    prov = {"run_id": EXPERIMENT, "git_hash": H.git_hash(),
            "git_dirty_code": SH.git_dirty(["src", "analysis/wcap_rlmnist_0914"]),
            "argv": sys.argv, "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "act": args.act, "arm": args.arm, "lam": Arm(args.arm).reg.lam, "cap_mult": CAP_MULT,
            "cap_start_task": Arm(args.arm).cap_start_task, "seeds": seeds, "lr": args.lr,
            "optimizer": "adam", "n_tasks": args.tasks, "epochs_per_task": args.epochs,
            "batch": BATCH, "steps_per_task": STEPS_PER_EPOCH * args.epochs,
            "n_images": N_IMAGES, "dims": list(H.DIMS), "rel_tol": REL_TOL,
            "data_sha256": mnist.sha256,
            "subset_sha256_sorted": {str(s): hashlib.sha256(
                np.sort(RL.subset_idx(s).numpy()).tobytes()).hexdigest() for s in seeds},
            "code_sha256": {"src/wcap_rlmnist_0914.py": SH.file_sha256(me),
                            "src/pmnist_0905.py": SH.file_sha256(src / "pmnist_0905.py"),
                            "src/pmnist_rlmnist_0906.py": SH.file_sha256(src / "pmnist_rlmnist_0906.py"),
                            "src/shell_l2_rlmnist_0913.py": SH.file_sha256(src / "shell_l2_rlmnist_0913.py")},
            "device": str(device), "runs": runs, "wall_clock_s": time.time() - t_start,
            "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "divergences": [i["divergence"] for i in runs.values() if i["divergence"]["diverged"]]}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"wrote {out}  ({len(rows)} task rows, {len(lrows)} layer rows, {time.time() - t_start:.1f}s)",
          flush=True)


if __name__ == "__main__":
    main()

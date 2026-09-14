#!/usr/bin/env python3
"""Random Label MNIST: L2 split into the centered-W decay and the rest (specs/spec_l2split_rlmnist_0914.md).

    OMP_NUM_THREADS=1 python3 src/l2split_rlmnist_0914.py --act R --arm l2wt --seeds 0 \
        --out results/l2split_rlmnist_0914/runs/R_l2wt_s0

The box is pmnist_rlmnist_0906's, imported through the wcap_rlmnist_0914 runner: data, labels, init, forward
and evaluate_rl from the host / 0906 runner, layer_rows and state_sha256 from the 0913 runner, the per-task
centered-norm logging (unit_arrays, task_fields) from the wcap runner.  None of the four files is modified.

run_one is wcap's run_one without the cap, with two decomposed penalties added where 0906 adds l2 (spec 1):
  l2wt   : W1, W2 get 2 lam (W - m 1)           (m = row mean);  b1, b2, W3, b3 get nothing
  l2rest : W1, W2 get 2 lam m 1;  b1, b2, W3, b3 get 2 lam p
so that l2wt + l2rest = l2 = 2 lam theta.  ref and l2 follow wcap's lines unchanged (check S4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import socket
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
from src import wcap_rlmnist_0914 as WC          # wcap runner; not touched

EXPERIMENT = "l2split_rlmnist_0914"
ACTS = ("LR", "R")
ARMS = ("ref", "l2", "l2wt", "l2rest")
N_IMAGES = RL.N_IMAGES                      # 1200
BATCH = RL.BATCH                            # 16
STEPS_PER_EPOCH = RL.STEPS_PER_EPOCH        # 75
LAM = 1e-3                                  # spec 2.1: the same lambda as l2
HIDDEN_W = (0, 2)                           # W1, W2 in the host's [W1, b1, W2, b2, W3, b3]


@dataclass(frozen=True)
class Arm:
    name: str

    def __post_init__(self):
        if self.name not in ARMS:
            raise ValueError(f"--arm must be one of {ARMS}, got {self.name!r}")

    @property
    def reg(self) -> SH.Reg:
        """The 0913 regulariser this arm runs through; the decomposed arms add theirs via extra_grad."""
        return SH.Reg("l2", LAM) if self.name == "l2" else SH.Reg("none")


# --------------------------------------------------------------------------
# the decomposed penalties (spec 1)
# --------------------------------------------------------------------------

@torch.no_grad()
def extra_grad(arm_name: str, params) -> list:
    """What l2wt / l2rest add to each tensor's gradient before Adam; None where nothing is added."""
    out = []
    for i, q in enumerate(params):
        if arm_name == "l2wt":
            out.append(2.0 * LAM * (q - q.mean(dim=1, keepdim=True)) if i in HIDDEN_W else None)
        elif arm_name == "l2rest":
            out.append(2.0 * LAM * (q.mean(dim=1, keepdim=True).expand_as(q) if i in HIDDEN_W else q))
        else:
            raise ValueError(arm_name)
    return out


@torch.no_grad()
def penalties(params) -> dict:
    """lam sum ||W~_i||^2 and lam [sum d m_i^2 + ||b1||^2 + ||b2||^2 + ||W3||^2 + ||b3||^2], float64 (spec 3)."""
    wt = rest = 0.0
    for i, q in enumerate(params):
        a = q.detach().double()
        if i in HIDDEN_W:
            m = a.mean(dim=1, keepdim=True)
            wt += float(((a - m) ** 2).sum())
            rest += float(a.shape[1] * (m * m).sum())
        else:
            rest += float((a * a).sum())
    return {"pen_wt_total": LAM * wt, "pen_rest_total": LAM * rest}


# --------------------------------------------------------------------------
# one (act, arm, seed) run
# --------------------------------------------------------------------------

def run_one(act_name: str, arm_s: str, seed: int, lr: float, n_tasks: int, mnist: H.Mnist,
            device: torch.device, epochs: int = 400, debug: dict | None = None,
            progress: bool = False) -> tuple[list[dict], list[dict], dict, dict]:
    """debug (checks only): captures init, subset, labels, batch orders, task-end weights and, at the
    (task, step) pairs in debug["capture"], the full state and batch just before the step and after it."""
    t_start = time.time()
    act = H.ARMS[act_name]
    params = H.init_params(seed, device)              # host init: bit-identical per seed
    if debug is not None:
        debug["init"] = [q.detach().cpu().clone() for q in params]
    arm = Arm(arm_s)
    reg = arm.reg
    p0 = [q.detach().clone() for q in params] if reg.kind == "l2init" else None
    ref0 = [q.detach().clone() for q in params]       # logging only; never read by the update
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
    hashes = {"init_sha256": hashlib.sha256(b"".join(WC._bytes(q) for q in params)).hexdigest(),
              "subset_idx_sha256": hashlib.sha256(WC._bytes(idx)).hexdigest()}
    h_lab, h_batch = hashlib.sha256(), hashlib.sha256()
    spt = STEPS_PER_EPOCH * epochs
    rows, lrows = [], SH.layer_rows(params, ref0, reg, act_name, seed, 0)
    u0 = WC.unit_arrays(params, x, act)
    units = {k: [v] for k, v in u0.items()}
    r_t1 = None
    info_extra = {"init_centered": {f"wt_med_l{li}": float(np.quantile(u0[f"wt_l{li}"], 0.5)) for li in (1, 2)}}
    diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "act": act_name, "arm": arm_s}

    for t in range(1, n_tasks + 1):
        y = RL.task_labels(g_lab).to(device)          # new labelling, same images
        h_lab.update(WC._bytes(y))
        if debug is not None:
            debug.setdefault("labels", []).append(y.cpu().clone())
        acc_sum = torch.zeros((), device=device)
        bad_step = torch.full((), -1, dtype=torch.long, device=device)

        for e in range(epochs):
            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)
            h_batch.update(WC._bytes(order))
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
                        "xb": xb.clone(), "yb": yb.clone()}
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
                    elif arm.name in ("l2wt", "l2rest"):
                        grads = [gr if eg is None else gr + eg for gr, eg in zip(grads, extra_grad(arm.name, params))]
                    m, v, tc = adam
                    tc[0] += 1
                    b1, b2, eps = 0.9, 0.999, 1e-8
                    c1 = 1 - b1 ** tc[0]
                    c2 = 1 - b2 ** tc[0]
                    for p, gr, mi, vi in zip(params, grads, m, v):
                        mi.mul_(b1).add_(gr, alpha=1 - b1)
                        vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                        p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
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
        mt = RL.evaluate_rl(params, x, y, act)
        u = WC.unit_arrays(params, x, act)
        if t == 1:
            r_t1 = {li: u[f"wt_l{li}"].copy() for li in (1, 2)}
        for k, v_ in u.items():
            units[k].append(v_)
        lr_rows = SH.layer_rows(params, ref0, reg, act_name, seed, t)
        for r in lr_rows:
            r["arm"] = arm_s
        lrows += lr_rows
        pen = penalties(params)
        own = {"l2wt": pen["pen_wt_total"], "l2rest": pen["pen_rest_total"]}.get(
            arm.name, sum(r["reg_loss"] for r in lr_rows))
        rows.append({"act": act_name, "arm": arm_s, "lam": reg.lam, "lr": lr, "seed": seed, "task": t,
                     "online_acc": float(acc_sum) / spt, "memo_acc": mt["acc"], **mt,
                     "reg_loss": own, **WC.task_fields(u, u0, r_t1, None, None), **pen})
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
            "git_dirty_code": SH.git_dirty(["src", "analysis/l2split_rlmnist_0914"]),
            "argv": sys.argv, "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "act": args.act, "arm": args.arm, "lam": LAM, "seeds": seeds, "lr": args.lr,
            "optimizer": "adam", "n_tasks": args.tasks, "epochs_per_task": args.epochs,
            "batch": BATCH, "steps_per_task": STEPS_PER_EPOCH * args.epochs,
            "n_images": N_IMAGES, "dims": list(H.DIMS),
            "data_sha256": mnist.sha256,
            "code_sha256": {"src/l2split_rlmnist_0914.py": SH.file_sha256(me),
                            "src/wcap_rlmnist_0914.py": SH.file_sha256(src / "wcap_rlmnist_0914.py"),
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

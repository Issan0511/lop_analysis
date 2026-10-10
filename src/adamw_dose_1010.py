#!/usr/bin/env python3
"""AdamW (decoupled weight decay on all six tensors) on the Random Label MNIST ELU->ELU box (adamw_dose_1010).

specs/spec_adamw_dose_1010.md (registered at PREREG_COMMIT, before any AdamW code existed).

The box is l2cap_ee_0917's ref arm (= doors_rlmnist_1007's ref): 1200 fixed images per seed, pixel/255, uniform
random labels per task, 784-100-100-10, ELU(1) on both hidden layers, Adam 1e-3 (moments and the update count
kept across tasks), batch 16, 6000 updates per task, flush_denormal on, first-layer ledger off.  run_adamw() is
the host's update loop written out (src/mucap_el_run_0916.py run_one, arm "ref", act2_name="ELU1",
w2cap=bias_fix=False, ledger=False; the same path as doors_rlmnist_1007's run_doors with both doors shut): the
same draws, the same arithmetic and the same diagnostics (the host's own forward2 / unit_arrays / adam_arrays are
imported, not copied), so with every lambda 0 it reproduces the recorded l2cap ref shards bit for bit (check S0).
No existing src/ file is modified.

The one addition (spec 2.1), inside adamw_step_: after the gradient, for each of the six tensors with
lambda_p != 0,

    p <- fl32(p * c),  c = fl32(1 - lr * lambda_p)          (p.mul_(1 - lr * lam): torch.optim.AdamW's op)

and then the host's Adam step, unchanged:  theta <- c theta - lr m_hat / (sqrt(v_hat) + eps).  Nothing is added to
the gradient (m and v see the loss gradient only).  lambda = 0 executes no decay op at all.  The decay is on from
the first update of task 1.

New per-task quantities go to wd_task.csv (spec 3.1); per_task.csv and units.npz carry exactly the host's columns
and arrays in every arm.  deadpix_ok is the in-run evidence that the decay ran with the right factor at every
update: the W1 columns of the seed's always-0 pixels get an exactly-0 gradient (so a 0 Adam term), and must
equal their initial values multiplied by np.float32(c) once per update so far, iterated in numpy float32.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                 # host; not touched
from src import pmnist_rlmnist_0906 as RL        # subset, labels; not touched
from src import shell_l2_rlmnist_0913 as SH      # state_sha256; not touched
from src import elu_growth_0909 as EG            # ELU1; not touched
from src import mucap_el_0916 as MU              # mu_basis, the (unused) radii; not touched
from src import l2cap_ee_0917 as W2C             # row_norms (the host records zero_r2_rows); not touched
from src import mucap_el_run_0916 as EL          # forward2, unit_arrays, adam_arrays, the box constants

EXPERIMENT = "adamw_dose_1010"
PREREG_COMMIT = "364d9cd1df7a40367a30aa16c52e9fcb0d8617dc"   # specs/spec_adamw_dose_1010.md, before any AdamW code
LR = 1e-3
N_IMAGES = EL.N_IMAGES                          # 1200
BATCH = EL.BATCH                                # 16
STEPS_PER_EPOCH = EL.STEPS_PER_EPOCH            # 75
EPOCHS = EL.EPOCHS                              # 80 -> 6000 updates per task
DENSE = EL.DENSE                                # (0, 75, 375, 1500, 3000, 6000) for tasks 2-10
TEST_TASKS = EL.TEST_TASKS                      # (1, 10, 50, 100, 150)
CAP_LAYER = EL.CAP_LAYER                        # W1: the host's (unused) first-layer radii are still recorded
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")
# the registered arms (spec 2.2): one lambda on all six tensors
ARM_LAMBDA = {"ref": 0.0, "wd1e-3": 1e-3, "wd1e-2": 1e-2, "wd3e-2": 3e-2, "wd1e-1": 1e-1}
# check-only arm (spec 6, S-ctrl): the readout's W3 and b3 only; never in the main run
CHECK_ARMS = {"ro1e-1": (0.0, 0.0, 0.0, 0.0, 1e-1, 1e-1)}


def arm_lams(arm: str) -> tuple[float, ...]:
    """The six per-tensor lambdas of an arm (W1, b1, W2, b2, W3, b3)."""
    if arm in ARM_LAMBDA:
        lam = ARM_LAMBDA[arm]
        return (lam, lam, lam, lam, lam, lam)
    if arm in CHECK_ARMS:
        return tuple(CHECK_ARMS[arm])
    raise ValueError(f"unknown arm {arm!r}; registered {tuple(ARM_LAMBDA)}, check-only {tuple(CHECK_ARMS)}")


def c32(lr: float, lam: float) -> np.float32:
    """fl32(1 - lr * lam): the factor p.mul_(1 - lr * lam) applies to a float32 tensor (spec 2.1)."""
    return np.float32(1 - lr * lam)


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


@torch.no_grad()
def adamw_step_(params, grads, adam, lr: float, lams, n_decay: list | None = None) -> None:
    """One update, in place: the update count, then per tensor the decoupled decay (only if its lambda != 0)
    and the host's Adam arithmetic, op for op (mucap_el_run_0916.run_one).  n_decay[i] counts tensor i's decays."""
    m_, v_, tc = adam
    tc[0] += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    c1, c2 = 1 - b1 ** tc[0], 1 - b2 ** tc[0]
    for i, (p, gr, mi, vi) in enumerate(zip(params, grads, m_, v_)):
        lam = lams[i]
        if lam != 0:
            p.mul_(1 - lr * lam)
            if n_decay is not None:
                n_decay[i] += 1
        mi.mul_(b1).add_(gr, alpha=1 - b1)
        vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
        p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)


def dead_pixels(x: torch.Tensor) -> tuple[np.ndarray, int]:
    """(indices of the pixels that are 0 in every one of the seed's images, number of non-constant pixels)."""
    xx = x.detach().cpu()
    dead = torch.nonzero((xx == 0).all(0)).flatten().numpy()
    d_eff = int((xx.amax(0) != xx.amin(0)).sum())
    return dead, d_eff


# --------------------------------------------------------------------------

def run_adamw(seed: int, lr: float, n_tasks: int, mnist: H.Mnist, device: torch.device,
              lams=(0.0,) * 6, arm: str = "ref", epochs: int = EPOCHS,
              debug: dict | None = None, progress: bool = False):
    """One seed of one arm.  Returns (rows, arrays, wd_rows, info): per_task rows and units arrays in the host's
    formats, the wd_task rows (spec 3.1) and the run's hashes.

    debug (checks only): init, subset, labels; at the (task, step) pairs in debug['capture'] the state before and
    after that update with the batch and the gradients; at the tasks in debug['end_capture'] the task-end params;
    with debug['grad_seq'] = [] every update's gradients are appended (S-torch)."""
    t_start = time.time()
    lams = tuple(float(v) for v in lams)
    if len(lams) != 6:
        raise ValueError("six lambdas: W1 b1 W2 b2 W3 b3")
    act1, act2 = EG.ELU(1.0), EG.ELU(1.0)
    params = H.init_params(seed, device)               # host init: bit-identical per seed
    capture = debug.get("capture", ()) if debug is not None else ()
    end_capture = debug.get("end_capture", ()) if debug is not None else ()
    grad_seq = debug.get("grad_seq") if debug is not None else None
    adam = ([torch.zeros_like(q) for q in params],
            [torch.zeros_like(q) for q in params], [0])

    idx = RL.subset_idx(seed).to(device)
    x = mnist.train_x[idx]                             # the task's inputs, fixed forever
    xt = mnist.test_x.to(device)
    e1, e1_64, mu1 = MU.mu_basis(x)                    # fixed: the images never change
    if e1 is None:
        raise RuntimeError("mu = 0 on this subset: the parallel axis is undefined")
    spt = STEPS_PER_EPOCH * epochs
    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
    dead, d_eff = dead_pixels(x)
    dead_t = torch.as_tensor(dead, dtype=torch.long)
    c_w1 = c32(lr, lams[0])
    dp_exp = params[0].detach().cpu()[:, dead_t].numpy().astype(np.float32).copy()   # independent book (numpy)
    if debug is not None:
        debug |= {"init": [q.detach().cpu().clone() for q in params], "subset": idx.cpu().clone(),
                  "e1": e1.cpu().clone(), "x": x.cpu().clone(), "dead": dead.copy()}
    h_lab, h_batch = hashlib.sha256(), hashlib.sha256()
    q_cap = v_cap = None
    rows, wd_rows, diag = [], [], {"task": [], "step": []}
    info = {"init_sha256": hashlib.sha256(b"".join(_bytes(q) for q in params)).hexdigest(),
            "subset_idx_sha256": hashlib.sha256(_bytes(idx)).hexdigest(),
            "mu1_norm": mu1, "cos_mu1_to_uniform": float(e1_64.sum() / np.sqrt(e1_64.numel())),
            "lams": list(lams), "c_hex": [float(c32(lr, v)).hex() for v in lams],
            "deadpix_n": int(len(dead)), "d_eff": d_eff}
    diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "arm": arm}
    n_updates = 0

    def snap(t: int, s: int) -> None:
        u = EL.unit_arrays(params, x, act1, act2, e1_64) | EL.adam_arrays(params, adam)
        if t in TEST_TASKS and s == spt:                # the task's end point, however it was reached
            ut = EL.unit_arrays(params, xt, act1, act2, e1_64)
            u |= {f"test_{k}": v for k, v in ut.items()}
        diag["task"].append(t)
        diag["step"].append(s)
        for k, v in u.items():
            diag.setdefault(k, []).append(v)

    for t in range(1, n_tasks + 1):
        y = RL.task_labels(g_lab).to(device)           # new labelling, same images
        h_lab.update(_bytes(y))
        if debug is not None:
            debug.setdefault("labels", []).append(y.cpu().clone())
        acc_sum = torch.zeros((), device=device)
        bad_step = torch.full((), -1, dtype=torch.long, device=device)
        proj = {"rows_par": 0, "rows_perp": 0, "rem_par": 0.0, "rem_perp": 0.0,
                "rows_w2": 0, "rem_w2": 0.0, "rows_bfix": 0}
        n_decay = [0] * 6
        dense = set(DENSE) if 2 <= t <= 10 else {0}
        if 0 in dense:
            snap(t, 0)                        # new labels, weights not yet moved

        for ep in range(epochs):
            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)
            h_batch.update(_bytes(order))
            xs, ys = x[order], y[order]                # reshuffled every epoch
            for j in range(STEPS_PER_EPOCH):
                s = ep * STEPS_PER_EPOCH + j
                if (t, s) in capture:
                    m_, v_, tc_ = adam
                    debug.setdefault("steps", {})[(t, s)] = {
                        "before": {"params": [q.detach().clone() for q in params],
                                   "m": [q.clone() for q in m_], "v": [q.clone() for q in v_], "tc": tc_[0]},
                        "xb": xs[j * BATCH:(j + 1) * BATCH].clone(),
                        "yb": ys[j * BATCH:(j + 1) * BATCH].clone()}
                xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
                out = EL.forward2(params, xb, act1, act2)
                loss = torch.nn.functional.cross_entropy(out[4], yb)
                # pre-update accuracy: the argmax of the very forward pass the loss came from
                acc_sum += (out[4].detach().argmax(1) == yb).float().mean()
                grads = torch.autograd.grad(loss, params)
                with torch.no_grad():
                    bad = ~torch.isfinite(loss)
                    bad_step = torch.where((bad_step < 0) & bad, torch.tensor(s, device=device), bad_step)
                    adamw_step_(params, grads, adam, lr, lams, n_decay)
                n_updates += 1
                if grad_seq is not None:
                    grad_seq.append([g_.detach().clone() for g_ in grads])
                if (t, s) in capture:
                    m_, v_, tc_ = adam
                    debug["steps"][(t, s)]["after"] = {
                        "params": [q.detach().clone() for q in params],
                        "m": [q.clone() for q in m_], "v": [q.clone() for q in v_], "tc": tc_[0]}
                    debug["steps"][(t, s)]["grads"] = [g_.detach().clone() for g_ in grads]
                if s + 1 in dense:
                    snap(t, s + 1)

        bs = int(bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params):
            diverged.update(diverged=True, task=t, step=(t - 1) * spt + max(bs, 0))
            rows.append({"arm": arm, "seed": seed, "task": t, "online_acc": float("nan")})
            break                                      # drop, never rescue
        if t == 1:
            info["task1_end_state_sha256"] = SH.state_sha256(params, adam, act1)
            q_cap = MU.parallel_cap(params[CAP_LAYER], e1)      # recorded by the host; no cap is applied
            v_cap = MU.perp_cap(params[CAP_LAYER], e1)
            if debug is not None:
                debug["task1_end_params"] = [q.detach().cpu().clone() for q in params]
                debug["task1_end_adam"] = ([q.clone() for q in adam[0]], [q.clone() for q in adam[1]], adam[2][0])
            info["zero_q_cap_rows"] = int((q_cap == 0).sum())
            info["zero_v_cap_rows"] = int((v_cap == 0).sum())
            info["zero_r2_rows"] = int((W2C.row_norms(params[2]) == 0).sum())
        if spt not in dense:
            snap(t, spt)
        with torch.no_grad():
            memo = float((EL.forward2(params, x, act1, act2)[4].argmax(1) == y).float().mean())
        u = EL.unit_arrays(params, x, act1, act2, e1_64)
        rows.append({"arm": arm, "seed": seed, "task": t, "online_acc": float(acc_sum) / spt,
                     "memo_acc": memo, "cap_on": 0, **proj,
                     # the best constant predictor's accuracy on this task's labels (the floor)
                     "major_frac": float(torch.bincount(y, minlength=10).max()) / N_IMAGES,
                     **{f"med_{k}": float(np.nanmedian(v)) for k, v in u.items() if v.size == 100}})
        # ---- spec 3.1 wd_task: the decay's own record (never read by the training)
        if lams[0] != 0:
            for _ in range(spt):
                dp_exp *= c_w1                         # once per update, numpy float32
        w1_dead = params[0].detach().cpu()[:, dead_t].numpy()
        with torch.no_grad():
            W3r = torch.linalg.vector_norm(params[4].detach().double(), dim=1)
            wd_rows.append({
                "arm": arm, "seed": seed, "task": t, "lam": max(lams), "c": float(c32(lr, max(lams))).hex(),
                **{f"wd_steps_{nm}": n_decay[i] for i, nm in enumerate(TENSORS)},
                "n_updates": n_updates, "deadpix_n": int(len(dead)),
                "deadpix_ok": int(np.array_equal(w1_dead, dp_exp)), "d_eff": d_eff,
                "w1_fro": float(torch.linalg.vector_norm(params[0].detach().double())),
                "w2_fro": float(torch.linalg.vector_norm(params[2].detach().double())),
                "w3_fro": float(torch.linalg.vector_norm(params[4].detach().double())),
                "w3_row_med": float(W3r.median()), "w3_row_max": float(W3r.max()),
                "b1_norm": float(torch.linalg.vector_norm(params[1].detach().double())),
                "b2_norm": float(torch.linalg.vector_norm(params[3].detach().double())),
                "b3_norm": float(torch.linalg.vector_norm(params[5].detach().double())),
                "z2_zero_pairs": float(np.mean(u["dtrain_zero_l2"]))})
        if debug is not None and t in end_capture:
            debug.setdefault("ends", {})[t] = {"params": [q.detach().clone() for q in params], "y": y.clone(),
                                               "memo": memo}
        if progress:
            print(f"[{time.time() - t_start:8.1f}s] {arm} seed={seed} task {t}/{n_tasks}", flush=True)

    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in diag.items()}
    arrays["q_cap"] = (q_cap[:, 0].double().numpy() if q_cap is not None else np.full(100, np.nan))
    arrays["v_cap"] = (v_cap[:, 0].double().numpy() if v_cap is not None else np.full(100, np.nan))
    info |= {"labels_sha256": h_lab.hexdigest(), "batch_sha256": h_batch.hexdigest(),
             "final_state_sha256": SH.state_sha256(params, adam, act1),
             "tasks_completed": len([r for r in rows if r["online_acc"] == r["online_acc"]]),
             "wall_clock_s": time.time() - t_start, "divergence": diverged}
    return rows, arrays, wd_rows, info

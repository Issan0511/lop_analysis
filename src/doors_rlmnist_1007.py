#!/usr/bin/env python3
"""Centering doors C / H / CH on the Random Label MNIST ELU->ELU box (doors_rlmnist_1007).

specs/spec_doors_rlmnist_1007.md (registered at PREREG_COMMIT, before any door code existed).

The box is l2cap_ee_0917's ref arm: 1200 fixed images per seed, pixel/255, uniform random labels per
task, 784-100-100-10, ELU(1) on both hidden layers, Adam 1e-3 (moments kept across tasks), batch 16,
6000 updates per task, flush_denormal on, first-layer ledger off.  run_doors() is the host's update loop
written out (src/mucap_el_run_0916.py run_one with arm "ref", act2_name="ELU1", w2cap=bias_fix=False,
ledger=False): same draws, same arithmetic, same diagnostics, so with both doors shut it reproduces the
recorded l2cap ref shards bit for bit (check S0).  No existing src/ file is modified.

Two doors, both on from the first update of task 1 (relu_doors_0919 / cifar5p1_mlp_0920 ReLUDoors,
ported to ELU):

    C  x <- x - x.mean(0) over the seed's own 1200 images (float32, subtracted, never divided).  The same
       x feeds every batch, memo and the units.npz diagnostics; the test images get the SAME vector.
    H  a_l = phi(z_l) - m_l on both hidden layers.  m_l is a per-unit EMA of the batch mean of the
       UNCENTRED phi(z_l):  m_l <- (1 - beta) m_l + beta * mean_batch phi(z_l),  beta = 0.01, m = 0 before
       the first update, never reset at a task boundary, updated right after each Adam step from the z of
       the forward pass that made that step's gradient (cifar5p1's update(z1, z2) order).  m is a constant:
       no gradient, not an Adam parameter, and the value in force is used by online, memo and every
       diagnostic alike.  The gate is untouched: d a_l / d z_l = phi'(z_l), which for this ELU in float32
       is expm1(min(z, 0)) + 1, exactly 0 below z = ln 2^-24 = -16.64 (resp_ee_0917).

The first-layer mean-image axis e1 of the diagnostics (q_l1 ...) is the RAW images' mean direction in
every arm (the host's value).  mu2 / a1mean_l1 is the mean of what the second layer actually receives
(a1 - m1 under H); the uncentred ||mean phi(z1)|| is logged as mu2_raw_norm.
"""

from __future__ import annotations

import hashlib
import math
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
from src import mucap_el_0916 as MU              # mu_basis, components, the (unused) radii; not touched
from src import l2cap_ee_0917 as W2C             # row_norms (the host records zero_r2_rows); not touched
from src import mucap_el_run_0916 as EL          # adam_arrays and the box constants; not touched

EXPERIMENT = "doors_rlmnist_1007"
PREREG_COMMIT = "0597501615b227f150d7cbd435c00a814c2819d0"   # specs/spec_doors_rlmnist_1007.md, before any door code
BETA = 0.01                                     # door H's EMA rate (relu_doors_0919, cifar5p1_mlp_0920)
N_IMAGES = EL.N_IMAGES                          # 1200
BATCH = EL.BATCH                                # 16
STEPS_PER_EPOCH = EL.STEPS_PER_EPOCH            # 75
EPOCHS = EL.EPOCHS                              # 80 -> 6000 updates per task
DENSE = EL.DENSE                                # (0, 75, 375, 1500, 3000, 6000) for tasks 2-10
TEST_TASKS = EL.TEST_TASKS                      # (1, 10, 50, 100, 150)
LOWGATE = EL.LOWGATE
CAP_LAYER = EL.CAP_LAYER                        # W1: the host's (unused) first-layer radii are still recorded
U32, U64 = 2.0 ** -24, 2.0 ** -53               # unit roundoffs
EPS64 = float(np.finfo(np.float64).eps)


def door_name(door_c: bool, door_h: bool) -> str:
    """The runner's own row label; the CLI relabels it with the registered arm name."""
    return f"C{int(door_c)}H{int(door_h)}"


def forward_doors(params, x, act1, act2, m=None):
    """The host's forward2 with door H: a_l = phi(z_l) - m_l when m is given.  m=None is forward2's
    arithmetic, op for op (S0 pins it through the recorded shards)."""
    W1, b1, W2, b2, W3, b3 = params
    z1 = x @ W1.T + b1
    a1 = act1.phi(z1)
    if m is not None:
        a1 = a1 - m[0]
    z2 = a1 @ W2.T + b2
    a2 = act2.phi(z2)
    if m is not None:
        a2 = a2 - m[1]
    return z1, a1, z2, a2, a2 @ W3.T + b3


@torch.no_grad()
def ema_update_(m: list, z1: torch.Tensor, z2: torch.Tensor, act1, act2, beta: float = BETA) -> None:
    """Door H, right after an Adam step: m_l <- (1 - beta) m_l + beta * mean_batch phi(z_l), the UNCENTRED
    phi of the forward pass that made the step (z2 already from the centred a1)."""
    for mi, z, act in zip(m, (z1, z2), (act1, act2)):
        mi.mul_(1.0 - beta).add_(act.phi(z).mean(0), alpha=beta)


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


def door_sha256(m) -> str | None:
    if m is None:
        return None
    h = hashlib.sha256()
    for q in m:
        h.update(_bytes(q))
    return h.hexdigest()


def gamma(k: int, u: float) -> float:
    return k * u / (1.0 - k * u)


@torch.no_grad()
def c_bound(x_raw: torch.Tensor, xc: torch.Tensor, n_sum: int) -> torch.Tensor:
    """Spec section 3: per-pixel bound on |mean of the centred inputs| (float64).  A float32 mean of n
    images (any summation order, <= 2 extra roundings), one rounding per subtraction, and a float64 mean
    of n_sum fed values; inflated by (1 + 8 eps64) for the bound's own float64 arithmetic."""
    n = x_raw.shape[0]
    ax = x_raw.double().abs().mean(0)
    axc = xc.double().abs().mean(0)
    return (gamma(n + 2, U32) * ax + (U32 / (1.0 - U32) + gamma(n_sum + 1, U64)) * axc) * (1.0 + 8.0 * EPS64)


def c_check(mean64: torch.Tensor, bound: torch.Tensor) -> tuple[bool, float, float]:
    """(all pixels within the bound, max |mean|, max |mean| / bound over pixels with a positive bound).
    A pixel with bound 0 (always 0 in every image) must have a mean of exactly 0."""
    a = mean64.abs()
    ok = bool((a <= bound).all())
    pos = bound > 0
    ratio = float((a[pos] / bound[pos]).max()) if bool(pos.any()) else 0.0
    if bool(((~pos) & (a > 0)).any()):
        ratio = float("inf")
    return ok, float(a.max()), ratio


# --------------------------------------------------------------------------
# per-unit state at one diagnostic point (float64): the host's unit_arrays through the door forward
# --------------------------------------------------------------------------

@torch.no_grad()
def unit_arrays_doors(params, x, act1, act2, e1_64, m=None) -> dict[str, np.ndarray]:
    """src/mucap_el_run_0916.py unit_arrays, line for line, with forward_doors(..., m) in place of
    forward2.  m=None gives the host's arrays bit for bit (S0)."""
    z1, a1, z2, _, _ = forward_doors(params, x, act1, act2, m)
    out = {}
    e2_32, e2_64, mu2 = MU.mu_basis(a1)
    S2 = a1.double().sum(1)
    out["s2_mean"] = np.array([float(S2.mean())])
    out["s2_sd"] = np.array([float(S2.std(unbiased=False))])
    out["mu2_norm"] = np.array([mu2])
    out["mu2_proj_sd"] = np.array([float((a1.double() @ e2_64).std(unbiased=False))
                                   if e2_64 is not None else float("nan")])
    out["a1mean_l1"] = a1.double().mean(0).numpy()
    for li, (z, act, k, e64) in enumerate(((z1, act1, 0, e1_64), (z2, act2, 2, e2_64)), start=1):
        z64 = z.double()
        g = act.dphi(z).double()
        with torch.enable_grad():
            zz = z.detach().clone().requires_grad_(True)
            gt, = torch.autograd.grad(act.phi(zz).sum(), zz)
        out[f"dtrain_mean_l{li}"] = gt.double().mean(0).numpy()
        out[f"dtrain_zero_l{li}"] = (gt == 0).double().mean(0).numpy()
        zbar, sd = z64.mean(0), z64.std(0, unbiased=False)
        U = z64.amax(0)
        safe = torch.where(sd > 0, sd, torch.full_like(sd, float("nan")))
        out[f"zbar_l{li}"] = zbar.numpy()
        out[f"sigma_l{li}"] = sd.numpy()
        out[f"U_l{li}"] = U.numpy()
        out[f"pplus_l{li}"] = (z64 > 0).double().mean(0).numpy()
        out[f"d_l{li}"] = (zbar / safe).numpy()
        out[f"K_l{li}"] = ((U - zbar) / safe).numpy()
        out[f"gate_mean_l{li}"] = g.mean(0).numpy()
        out[f"gate_rms_l{li}"] = g.square().mean(0).sqrt().numpy()
        out[f"lowgate_l{li}"] = (g < LOWGATE).double().mean(0).numpy()
        out[f"b{li}"] = params[k + 1].detach().double().numpy()
        if e64 is None:                    # an exactly-zero mean input: no axis (never seen; kept total)
            for name in ("q", "v_norm", "row_mean", "wt_norm", "row_norm"):
                out[f"{name}_l{li}"] = np.full(params[k].shape[0], np.nan)
            continue
        c = MU.components(params[k], e64)
        for name, v in c.items():
            out[f"{name}_l{li}"] = v
    return out


@torch.no_grad()
def door_arrays(params, x, act1, act2, m=None) -> dict[str, np.ndarray]:
    """Door arms only: the uncentred ||mean phi(z1)||, the mean of what the readout receives, and m."""
    z1, a1, z2, a2, _ = forward_doors(params, x, act1, act2, m)
    out = {"mu2_raw_norm": np.array([float(act1.phi(z1).double().mean(0).norm())]),
           "a2mean_l2": a2.double().mean(0).numpy()}
    if m is not None:
        out["m_l1"] = m[0].detach().double().numpy().copy()
        out["m_l2"] = m[1].detach().double().numpy().copy()
    return out


# --------------------------------------------------------------------------

def run_doors(seed: int, lr: float, n_tasks: int, mnist: H.Mnist, device: torch.device,
              door_c: bool = False, door_h: bool = False, epochs: int = EPOCHS,
              debug: dict | None = None, progress: bool = False, beta: float = BETA):
    """One seed of one arm.  Returns (rows, arrays, info) in the host's formats.

    debug (checks only): init, subset, the inputs, labels, the task-1-end state and, at the (task, step)
    pairs in debug['capture'], the state before and after that update (params, Adam moments, m) with the
    batch and the gradients; at the tasks in debug['end_capture'], the task-end params and m."""
    t_start = time.time()
    act1, act2 = EG.ELU(1.0), EG.ELU(1.0)
    params = H.init_params(seed, device)               # host init: bit-identical per seed
    capture = debug.get("capture", ()) if debug is not None else ()
    end_capture = debug.get("end_capture", ()) if debug is not None else ()
    adam = ([torch.zeros_like(q) for q in params],
            [torch.zeros_like(q) for q in params], [0])

    idx = RL.subset_idx(seed).to(device)
    x_raw = mnist.train_x[idx]                         # the task's inputs, fixed forever
    xt_raw = mnist.test_x.to(device)
    e1, e1_64, mu1 = MU.mu_basis(x_raw)                # the raw images' mean axis, in every arm
    if e1 is None:
        raise RuntimeError("mu = 0 on this subset: the parallel axis is undefined")
    spt = STEPS_PER_EPOCH * epochs
    xbar = None
    if door_c:
        xbar = x_raw.mean(0)                           # door C: the seed's own 1200-image float32 mean
        x = x_raw - xbar
        xt = xt_raw - xbar                             # the same vector, or train and test part ways
        bound_fed = c_bound(x_raw, x, spt * BATCH)
        bound_eval = c_bound(x_raw, x, N_IMAGES)
    else:
        x, xt = x_raw, xt_raw
    m = [torch.zeros(params[1].shape[0], device=device), torch.zeros(params[3].shape[0], device=device)] \
        if door_h else None
    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)
    if debug is not None:
        debug |= {"init": [q.detach().cpu().clone() for q in params], "subset": idx.cpu().clone(),
                  "e1": e1.cpu().clone(), "x": x.cpu().clone(), "x_raw": x_raw.cpu().clone(),
                  "xt": xt.cpu().clone(), "xbar": None if xbar is None else xbar.cpu().clone()}
    h_lab, h_batch = hashlib.sha256(), hashlib.sha256()
    q_cap = v_cap = None
    rows, diag = [], {"task": [], "step": []}
    arm_s = door_name(door_c, door_h)
    info = {"init_sha256": hashlib.sha256(b"".join(_bytes(q) for q in params)).hexdigest(),
            "subset_idx_sha256": hashlib.sha256(_bytes(idx)).hexdigest(),
            "mu1_norm": mu1, "cos_mu1_to_uniform": float(e1_64.sum() / np.sqrt(e1_64.numel()))}
    door_info = {"door_c": door_c, "door_h": door_h, "beta": beta if door_h else None}
    if door_c:
        ok, amax, ratio = c_check(x.double().mean(0), bound_eval)
        door_info |= {"xbar_sha256": hashlib.sha256(_bytes(xbar)).hexdigest(),
                      "c_eval_ok": ok, "c_eval_absmax": amax, "c_eval_ratio": ratio,
                      "c_bound_eval_max": float(bound_eval.max()), "c_bound_fed_max": float(bound_fed.max()),
                      "mu1_fed_norm": float(x.double().mean(0).norm())}
    diverged = {"diverged": False, "task": None, "step": None, "seed": seed, "arm": arm_s}

    def snap(t: int, s: int) -> None:
        u = unit_arrays_doors(params, x, act1, act2, e1_64, m) | EL.adam_arrays(params, adam)
        if t in TEST_TASKS and s == spt:                # the task's end point, however it was reached
            ut = unit_arrays_doors(params, xt, act1, act2, e1_64, m)
            u |= {f"test_{k}": v for k, v in ut.items()}
        if door_c or door_h:
            u |= door_arrays(params, x, act1, act2, m)
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
        fed = torch.zeros(x.shape[1], dtype=torch.float64, device=device) if door_c else None
        h_upd = [0, 0]
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
                                   "m": [q.clone() for q in m_], "v": [q.clone() for q in v_], "tc": tc_[0],
                                   "door_m": None if m is None else [q.clone() for q in m]},
                        "xb": xs[j * BATCH:(j + 1) * BATCH].clone(),
                        "yb": ys[j * BATCH:(j + 1) * BATCH].clone()}
                xb, yb = xs[j * BATCH:(j + 1) * BATCH], ys[j * BATCH:(j + 1) * BATCH]
                out = forward_doors(params, xb, act1, act2, m)
                loss = torch.nn.functional.cross_entropy(out[4], yb)
                # pre-update accuracy: the argmax of the very forward pass the loss came from
                acc_sum += (out[4].detach().argmax(1) == yb).float().mean()
                grads = torch.autograd.grad(loss, params)
                with torch.no_grad():
                    bad = ~torch.isfinite(loss)
                    bad_step = torch.where((bad_step < 0) & bad, torch.tensor(s, device=device), bad_step)
                    m_, v_, tc = adam
                    tc[0] += 1
                    b1, b2, eps = 0.9, 0.999, 1e-8
                    c1, c2 = 1 - b1 ** tc[0], 1 - b2 ** tc[0]
                    for p, gr, mi, vi in zip(params, grads, m_, v_):
                        mi.mul_(b1).add_(gr, alpha=1 - b1)
                        vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                        p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
                    if door_h:
                        m_old = [q.clone() for q in m]
                        ema_update_(m, out[0].detach(), out[2].detach(), act1, act2, beta)
                        for li in range(2):
                            h_upd[li] += int((not torch.equal(m[li], m_old[li])) and bool((m[li] != 0).any()))
                    if door_c:
                        fed += xb.double().sum(0)
                if (t, s) in capture:
                    m_, v_, tc_ = adam
                    debug["steps"][(t, s)]["after"] = {
                        "params": [q.detach().clone() for q in params],
                        "m": [q.clone() for q in m_], "v": [q.clone() for q in v_], "tc": tc_[0],
                        "door_m": None if m is None else [q.clone() for q in m]}
                    debug["steps"][(t, s)]["grads"] = [g_.detach().clone() for g_ in grads]
                if s + 1 in dense:
                    snap(t, s + 1)

        bs = int(bad_step)
        if bs >= 0 or not all(torch.isfinite(p).all() for p in params) or \
                (m is not None and not all(torch.isfinite(q).all() for q in m)):
            diverged.update(diverged=True, task=t, step=(t - 1) * spt + max(bs, 0))
            rows.append({"arm": arm_s, "seed": seed, "task": t, "online_acc": float("nan")})
            break                                      # drop, never rescue
        if t == 1:
            info["task1_end_state_sha256"] = SH.state_sha256(params, adam, act1)
            q_cap = MU.parallel_cap(params[CAP_LAYER], e1)      # recorded by the host; no cap is applied
            v_cap = MU.perp_cap(params[CAP_LAYER], e1)
            if debug is not None:
                debug["task1_end_params"] = [q.detach().cpu().clone() for q in params]
                debug["task1_end_adam"] = ([q.clone() for q in adam[0]], [q.clone() for q in adam[1]], adam[2][0])
                debug["task1_end_m"] = None if m is None else [q.clone() for q in m]
            info["zero_q_cap_rows"] = int((q_cap == 0).sum())
            info["zero_v_cap_rows"] = int((v_cap == 0).sum())
            info["zero_r2_rows"] = int((W2C.row_norms(params[2]) == 0).sum())
            door_info["task1_end_door_sha256"] = door_sha256(m)
        if spt not in dense:
            snap(t, spt)
        with torch.no_grad():
            memo = float((forward_doors(params, x, act1, act2, m)[4].argmax(1) == y).float().mean())
        u = unit_arrays_doors(params, x, act1, act2, e1_64, m)
        row = {"arm": arm_s, "seed": seed, "task": t, "online_acc": float(acc_sum) / spt,
               "memo_acc": memo, "cap_on": 0, **proj,
               # the best constant predictor's accuracy on this task's labels (the floor)
               "major_frac": float(torch.bincount(y, minlength=10).max()) / N_IMAGES,
               **{f"med_{k}": float(np.nanmedian(v)) for k, v in u.items() if v.size == 100}}
        if door_c:
            ok, amax, ratio = c_check(fed / (spt * BATCH), bound_fed)
            row |= {"c_ok": int(ok), "c_in_absmax": amax, "c_in_ratio": ratio}
        if door_h:
            row |= {"h_upd_l1": h_upd[0], "h_upd_l2": h_upd[1],
                    "m_l1_mean": float(m[0].double().mean()), "m_l2_mean": float(m[1].double().mean()),
                    "m_l1_norm": float(m[0].double().norm()), "m_l2_norm": float(m[1].double().norm())}
        rows.append(row)
        if debug is not None and t in end_capture:
            debug.setdefault("ends", {})[t] = {"params": [q.detach().clone() for q in params],
                                               "door_m": None if m is None else [q.clone() for q in m],
                                               "y": y.clone(), "memo": memo}
        if progress:
            print(f"[{time.time() - t_start:8.1f}s] {arm_s} seed={seed} task {t}/{n_tasks}", flush=True)

    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in diag.items()}
    arrays["q_cap"] = (q_cap[:, 0].double().numpy() if q_cap is not None else np.full(100, np.nan))
    arrays["v_cap"] = (v_cap[:, 0].double().numpy() if v_cap is not None else np.full(100, np.nan))
    door_info["final_door_sha256"] = door_sha256(m)
    info |= {"labels_sha256": h_lab.hexdigest(), "batch_sha256": h_batch.hexdigest(),
             "final_state_sha256": SH.state_sha256(params, adam, act1),
             "tasks_completed": len([r for r in rows if r["online_acc"] == r["online_acc"]]),
             "wall_clock_s": time.time() - t_start, "divergence": diverged, "doors": door_info}
    return rows, arrays, info

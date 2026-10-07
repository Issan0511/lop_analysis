#!/usr/bin/env python3
"""cap_cifar5p1_1007 -- row-norm caps on W1 / W2 in the 5+1 CIFAR x MLP box.

    python -m src.cap_cifar5p1_1007 run --arm cap12             # seeds 0-9 stacked, 30 tasks
    python -m src.cap_cifar5p1_1007 run --arm ref --out /tmp/x --tasks 4 --steps-hard 100

Spec: specs/spec_cap_cifar5p1_1007.md.  The box is the host's (src/cifar5p1_mlp_0920.py:
arm R, cond std, Adam lr 1e-4, 30 tasks of 780 updates, seeds stacked R = 10, one step
captured as a CUDA graph, fresh-network control) and nothing in it is modified: data, task
plan, batches, init, the stacked forward, the evaluation and the CSV writer are imported.
Only the host's `run()` is copied, because the cap has to sit inside the captured step,
right after the Adam update.  The copy keeps the host's arithmetic for R/std line for line
(the iv / door C / nan_slot branches are dropped); S-nochange compares the ref arm with the
host's own `run()` byte for byte.

The cap (spec 2.4; l2cap_ee_0917's `cap_row_norm_`, stacked and graph-safe):

    r_{l,s,i} = ||W_l[s, i, :]||              float32, right after task 1's last update
    after every Adam update from task 2's first:
        n = ||W_l[s, i, :]||;  over = n > r;  W <- where(over, W * (r / n), W)

Rows at or below their radius (equal included) keep every bit; small rows are never
inflated; nothing else -- biases, W3/b3, the Adam moments, the step counter, any RNG -- is
touched.  The radius tensors are static (the graph reads them by address): +inf during
task 1, so the projection writes nothing; set in place (copy_) between task 1's last update
and task 2's first; reset to +inf before the fresh control, which is therefore the host's
procedure with no cap (spec 2.5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H                    # streams, setup, csv writer
from src import rlcifar_mlp_battle_0918 as B        # stacked forward, git_state, parse_ints
from src import cifar5p1_mlp_0920 as C              # the box (not modified)

EXPERIMENT = "cap_cifar5p1_1007"
OUT_ROOT = H.REPO / "results" / EXPERIMENT
SPEC = "specs/spec_cap_cifar5p1_1007.md"
CHECKS_JSON = OUT_ROOT / "checks.json"
REQUIRED_CHECKS = ("S-nochange", "S-cap", "S-radius", "S-graph", "S-fresh", "S-verdict", "S-CLI")
SOURCES = ("src/cap_cifar5p1_1007.py", "analysis/cap_cifar5p1_1007/checks.py",
           "analysis/cap_cifar5p1_1007/verdict.py")

ACT_ARM, COND, LR = "R", "std", C.LR                 # the host's arm, condition and lr
CAP_ARMS = {"ref": (), "cap1": (0,), "cap2": (1,), "cap12": (0, 1)}   # 0 -> W1 (P[0]), 1 -> W2 (P[2])
ARM_ORDER = tuple(CAP_ARMS)
RADIUS_TASK = 1      # radius = each row's norm right after this task's last update
CAP_FROM = 2         # the projection acts from this task's first update on
REGISTERED = dict(seeds=list(range(10)), tasks=C.N_TASKS, lr=LR, steps_hard=C.STEPS_PER_TASK,
                  steps_easy=C.STEPS_PER_TASK, fresh=True, graph=True, threads=2)

U32 = 2.0 ** -24                                     # float32 unit roundoff
EPS32 = float(torch.finfo(torch.float32).eps)        # 2^-23
TINY32 = float(torch.finfo(torch.float32).tiny)      # smallest normal float32
U64 = 2.0 ** -53


def gamma(n: int, u: float = U64) -> float:
    """Higham's gamma_n = n u / (1 - n u)."""
    return n * u / (1.0 - n * u)


# --------------------------------------------------------------------------
# the cap (spec 2.4) -- graph-safe: no host sync, counters stay on the device
# --------------------------------------------------------------------------

@torch.no_grad()
def row_norms(W: torch.Tensor) -> torch.Tensor:
    """(R, rows, 1) float32 norms of W's rows: the radius, and the norm the projection compares."""
    return torch.linalg.vector_norm(W, dim=2, keepdim=True)


@torch.no_grad()
def project_rows_(W: torch.Tensor, r: torch.Tensor, rows: torch.Tensor,
                  rem: torch.Tensor) -> torch.Tensor:
    """In place: rows with ||w|| > r become w * r / ||w||; the others are not written.

    `rows` (R,) int64 and `rem` (R,) float64 accumulate the rows written and the norm removed,
    sum (n - r)+.  A zero row is never over (0 > r is false), so the division never meets it.
    Returns the float32 norms the decision used.
    """
    n = row_norms(W)
    over = n > r
    safe = torch.where(over, n, torch.ones_like(n))
    W.copy_(torch.where(over, W * (r / safe), W))
    rows.add_(over.sum(dim=(1, 2)))
    rem.add_((n.double() - r.double()).clamp(min=0).sum(dim=(1, 2)))
    return n


@torch.no_grad()
def monitor_(W: torch.Tensor, r: torch.Tensor, n32: torch.Tensor, n64_before: torch.Tensor,
             viol: torch.Tensor, excess: torch.Tensor) -> None:
    """After the projection: rows whose float64 norm is above the spec 7.2 bound, and the max excess.

    bound = r (1+u32)^2 / (1+theta) (1+gamma_{d+1})^2 + sqrt(d) * tiny32, with
    1/(1+theta) = n64_before / n32 the measured error of the float32 norm the decision used.
    A row that was not written has n32 <= r, hence n64 = n32/(1+theta) <= the same bound.
    `excess` is max (n64/r - 1)/eps32 over rows with a finite positive radius.
    """
    d = W.shape[2]
    n64 = torch.linalg.vector_norm(W.double(), dim=2, keepdim=True)
    r64 = r.double()
    inv = torch.where(n32 > 0, n64_before / n32.double(), torch.ones_like(n64_before))
    lim = r64 * (1.0 + U32) ** 2 * inv * (1.0 + gamma(d + 1)) ** 2 + math.sqrt(d) * TINY32
    viol.add_((n64 > lim).sum(dim=(1, 2)))
    ok = torch.isfinite(r64) & (r64 > 0)
    exc = torch.where(ok, (n64 / torch.where(ok, r64, torch.ones_like(r64)) - 1.0) / EPS32,
                      torch.full_like(n64, -math.inf))
    excess.copy_(torch.maximum(excess, exc.amax(dim=(1, 2))))


@torch.no_grad()
def cap_stats(P, Xt: torch.Tensor, act, r_saved) -> dict:
    """Task-end extras (REPORT_ONLY): ||mu_2||, mean biases, row-norm growth over task 1's end."""
    z1 = torch.baddbmm(P[1][:, None, :], Xt, P[0].transpose(1, 2))
    a1 = act.phi(z1, 0, False)
    out = {"mu2_norm": torch.linalg.vector_norm(a1.double().mean(1), dim=1),
           "b1_mean": P[1].double().mean(1), "b2_mean": P[3].double().mean(1)}
    for l in range(2):
        if r_saved[l] is None:
            continue
        ratio = torch.linalg.vector_norm(P[2 * l].double(), dim=2) / r_saved[l][..., 0].double()
        out[f"ratio_max_l{l + 1}"] = ratio.amax(1)
        out[f"ratio_med_l{l + 1}"] = ratio.cpu().median(1).values    # cpu: deterministic
    return out


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# the stacked run: the host's run() for R/std, plus the cap
# --------------------------------------------------------------------------

def run(cap_arm: str, seeds: list[int], n_tasks: int, device, out: Path, lr: float = LR,
        cifar: C.Cifar100 | None = None, fresh: bool = True, graph: bool = True,
        progress=None, debug: dict | None = None, observer=None,
        steps_hard: int = C.STEPS_PER_TASK, steps_easy: int = C.STEPS_PER_TASK) -> dict:
    """Train one cap arm's R = len(seeds) runs in lockstep over the 30-task sequence.

    `debug` (checks only) makes the steps eager; `debug["hook"]`, if given, is called as
    hook(event, **state) at "pre"/"post" (around the projection, every update),
    "task_start", "task_end", "radius" and "fresh_start".  `observer(t, state)` is called at
    every task end with P, the Adam moments and the task's training images (diagnostics; the
    graph is kept).  Neither writes to the training state.
    """
    t_start = time.time()
    progress = progress or (lambda m: print(m, flush=True))
    if cap_arm not in CAP_ARMS:
        raise SystemExit(f"unknown cap arm {cap_arm!r}; known: {','.join(ARM_ORDER)}")
    layers = CAP_ARMS[cap_arm]
    arm, cond = ACT_ARM, COND
    act = C.make_act(arm)
    R = len(seeds)
    cifar = cifar or C.Cifar100()
    X_all = cifar.inputs("train", cond, device)
    Y_all = cifar.train_y.to(device)
    X_test = cifar.inputs("test", cond, device)
    Y_test = cifar.test_y.to(device)
    tr_rows, te_rows = C.class_rows(cifar.train_y), C.class_rows(cifar.test_y)
    plans = [C.task_plan(s) for s in seeds]
    g_batch = {s: H.stream("c51_batch", s) for s in seeds}

    init = [q.detach() for s in seeds for q in C.init_params(arm, s, device)]
    P = [torch.stack(init[i::6]).contiguous() for i in range(6)]
    P = [q.requires_grad_(True) for q in P]
    P0 = [q.detach().clone() for q in P]
    act.init_state(R, device, key=f"{arm}|{seeds}|{cond}")
    act0 = act.state()
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    hook = None if debug is None else debug.get("hook")
    if debug is not None:
        debug["init"] = [q.detach().cpu().clone() for q in P]

    # ---- cap state: static tensors the captured step reads by address
    rad = [torch.full((R, P[2 * l].shape[1], 1), math.inf, device=device) for l in range(2)]
    rows_acc = torch.zeros(R, 2, dtype=torch.long, device=device)
    rem_acc = torch.zeros(R, 2, dtype=torch.float64, device=device)
    viol_acc = torch.zeros(R, 2, dtype=torch.long, device=device)
    excess_acc = torch.full((R, 2), -math.inf, dtype=torch.float64, device=device)

    # ---- one training step on static tensors (the host's, plus the projection)
    b1, b2, eps = 0.9, 0.999, 1e-8
    static_idx = torch.zeros(R, C.BATCH, dtype=torch.long, device=device)
    inv_c1 = torch.zeros((), device=device)
    inv_c2 = torch.zeros((), device=device)
    step_t = torch.zeros((), dtype=torch.long, device=device)
    acc_sum = torch.zeros(R, device=device)
    bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
    last_hit = torch.zeros(R, device=device)

    post = getattr(act, "post_update", None)

    def step():
        xb, yb = X_all[static_idx], Y_all[static_idx]            # gathers, no arithmetic
        z1, a1, z2, a2, z3 = B.forward(P, xb, act, train=True)
        lossv = F.cross_entropy(z3.reshape(-1, C.N_CLASSES), yb.reshape(-1),
                                reduction="none").view(R, C.BATCH).mean(1)
        hit = (z3.detach().argmax(-1) == yb).float().mean(1)     # pre-update: the paper's a_j
        acc_sum.add_(hit)
        last_hit.copy_(hit)
        grads = torch.autograd.grad(lossv.sum(), P)
        with torch.no_grad():
            bad = ~torch.isfinite(lossv)
            bad_step.copy_(torch.where((bad_step < 0) & bad, step_t, bad_step))
            step_t.add_(1)
            for p, gr, mi, vi in zip(P, grads, adam_m, adam_v):
                mi.mul_(b1).add_(gr, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
            act.update(z1.detach(), z2.detach())
            if post is not None:
                post(P, lr)
            if hook is not None:
                hook("pre", P=P, m=adam_m, v=adam_v, rad=rad, rows=rows_acc, rem=rem_acc)
            for l in layers:                                     # spec 2.4: W1 first, then W2
                W = P[2 * l]
                n64b = torch.linalg.vector_norm(W.double(), dim=2, keepdim=True)
                n32 = project_rows_(W, rad[l], rows_acc[:, l], rem_acc[:, l])
                monitor_(W, rad[l], n32, n64b, viol_acc[:, l], excess_acc[:, l])
            if hook is not None:
                hook("post", P=P, m=adam_m, v=adam_v, rad=rad, rows=rows_acc, rem=rem_acc)

    use_graph = graph and device.type == "cuda" and debug is None
    cg = None
    mutable = (*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit,
               rows_acc, rem_acc, viol_acc, excess_acc)
    if use_graph:
        keep = [q.detach().clone() for q in mutable]
        keep_act = act.state()
        inv_c1.fill_(1.0); inv_c2.fill_(1.0)
        static_idx.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                act.begin_step(R, C.BATCH, device)
                step()
        torch.cuda.current_stream().wait_stream(side)
        cg = torch.cuda.CUDAGraph()
        act.begin_step(R, C.BATCH, device)
        with torch.cuda.graph(cg):
            step()
        with torch.no_grad():
            for q, v in zip(mutable, keep):
                q.copy_(v)
        act.load_state(keep_act)
        del keep

    def train_task(batches, tc0: int) -> int:
        """`batches.shape[1]` steps on (R, steps, 32) global row indices.  New Adam counter."""
        tc = tc0
        acc_sum.zero_()
        bad_step.fill_(-1)
        step_t.zero_()
        rows_acc.zero_(); rem_acc.zero_(); viol_acc.zero_(); excess_acc.fill_(-math.inf)
        for j in range(batches.shape[1]):
            static_idx.copy_(batches[:, j])
            tc += 1
            inv_c1.fill_(1.0 / (1 - b1 ** tc))
            inv_c2.fill_(1.0 / (1 - b2 ** tc))
            act.begin_step(R, C.BATCH, device)
            if cg is not None:
                cg.replay()
            else:
                step()
            if debug is not None:
                debug.setdefault("online", []).append(last_hit.cpu().clone())
        if device.type == "cuda":
            torch.cuda.synchronize()
        return tc

    alive = torch.ones(R, dtype=torch.bool, device=device)
    rows, cap_rows, diverged, tc = [], [], [], 0
    last_hard = max(t for t in range(1, n_tasks + 1) if t % 2 == 1)
    saved = {}
    out.mkdir(parents=True, exist_ok=True)
    step_ms = float("nan")
    r_saved = [None, None]                 # task-1-end row norms (recorded in every arm)
    totals = {"rows_w1": 0, "rows_w2": 0, "viol_w1": 0, "viol_w2": 0}

    for t in range(1, n_tasks + 1):
        hard = t % 2 == 1
        steps = steps_hard if hard else steps_easy
        rows_t = [torch.cat([tr_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
        batches = torch.stack([C.batch_indices(g_batch[s], rows_t[r], steps)
                               for r, s in enumerate(seeds)]).to(device)
        if t == last_hard and fresh:
            saved[t] = batches.clone()
        if debug is not None:
            debug.setdefault("batches", []).append(batches.cpu().clone())
        if t == CAP_FROM:
            with torch.no_grad():
                for l in layers:
                    rad[l].copy_(r_saved[l])
        if hook is not None:
            hook("task_start", t=t, rad=rad)
        t0 = time.time()
        tc = train_task(batches, tc)
        step_ms = 1e3 * (time.time() - t0) / steps
        if hook is not None:
            hook("task_end", t=t, P=P, m=adam_m, v=adam_v, rad=rad)
        if t == RADIUS_TASK:
            with torch.no_grad():
                r_saved = [row_norms(P[2 * l]).detach().clone() for l in range(2)]
            if hook is not None:
                hook("radius", t=t, r_saved=r_saved)

        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            newly = alive & ((bad_step >= 0) | ~finite)
            for r in torch.nonzero(newly).flatten().tolist():
                diverged.append({"diverged": True, "task": t, "seed": seeds[r], "arm": arm,
                                 "cap_arm": cap_arm, "cond": cond,
                                 "step": (t - 1) * C.STEPS_PER_TASK + max(int(bad_step[r]), 0)})
                rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "slot": r, "lr": lr,
                             "task": t, "hard": int(hard), "acc": float("nan")})
            alive &= ~newly
            live = torch.nonzero(alive).flatten().tolist()
            Xt = torch.stack([X_all[rows_t[r]] for r in range(R)])
            Yt = torch.stack([Y_all[rows_t[r]] for r in range(R)])
            m = C.evaluate(P, Xt, Yt, act, live)
            extra = cap_stats(P, Xt, act, r_saved)
            if observer is not None:
                observer(t, dict(P=P, m=adam_m, v=adam_v, Xt=Xt, act=act, live=live, rad=rad))
            del Xt
            te = [torch.cat([te_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
            test_acc = C.accuracy(P, torch.stack([X_test[q] for q in te]),
                                  torch.stack([Y_test[q] for q in te]), act)
        for r in live:
            rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "slot": r, "lr": lr,
                         "task": t, "hard": int(hard), "n_classes": len(plans[r][t - 1][1]),
                         "online_acc": float(acc_sum[r]) / steps,
                         "train_acc": m[r]["acc"], "test_acc": float(test_acc[r]), **m[r]})
            cr = {"cap_arm": cap_arm, "seed": seeds[r], "slot": r, "task": t, "hard": int(hard)}
            for l in range(2):
                cr[f"rows_w{l + 1}"] = int(rows_acc[r, l])
                cr[f"rem_w{l + 1}"] = float(rem_acc[r, l])
            for l in range(2):
                cr[f"viol_w{l + 1}"] = int(viol_acc[r, l])
            for l in range(2):
                cr[f"excess_w{l + 1}"] = float(excess_acc[r, l])
            for k in ("ratio_max_l1", "ratio_med_l1", "ratio_max_l2", "ratio_med_l2",
                      "mu2_norm", "b1_mean", "b2_mean"):
                cr[k] = float(extra[k][r]) if k in extra else float("nan")
            cap_rows.append(cr)
        for l in range(2):
            totals[f"rows_w{l + 1}"] += int(rows_acc[:, l].sum())
            totals[f"viol_w{l + 1}"] += int(viol_acc[:, l].sum())
        rows.sort(key=lambda q: (q["slot"], q["task"]))
        cap_rows.sort(key=lambda q: (q["slot"], q["task"]))
        H.write_csv(out / "per_task.csv", rows)
        H.write_csv(out / "cap_task.csv", cap_rows)
        on = acc_sum[alive] / steps
        el = time.time() - t_start
        progress(f"[{time.strftime('%T')}] {cap_arm} task {t:2d}/{n_tasks} "
                 f"{'hard' if hard else 'easy'} alive {int(alive.sum())}/{R} "
                 f"online {float(on.mean()) if len(on) else float('nan'):.3f} "
                 f"rows w1 {int(rows_acc[:, 0].sum())} w2 {int(rows_acc[:, 1].sum())} "
                 f"viol {int(viol_acc.sum())}  {step_ms:.2f} ms/step  {el / 60:.1f} min")

    # ---- fresh-network control on the last hard task: the host's procedure, no cap (spec 2.5)
    fresh_rows, fresh_written = [], None
    if fresh and last_hard in saved:
        continual = {int(q["seed"]): q["online_acc"] for q in rows
                     if q["task"] == last_hard and "online_acc" in q}
        with torch.no_grad():
            for q, v in zip(P, P0):
                q.copy_(v)
            for q in (*adam_m, *adam_v):
                q.zero_()
            for l in range(2):
                rad[l].fill_(math.inf)
        act.load_state(act0)
        if hook is not None:
            hook("fresh_start", rad=rad, P=P, m=adam_m, v=adam_v)
        alive_f = torch.ones(R, dtype=torch.bool, device=device)
        train_task(saved[last_hard], 0)
        fresh_written = [int(x) for x in rows_acc.sum(0).tolist()]
        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            alive_f &= (bad_step < 0) & finite
        for r in range(R):
            if not bool(alive_f[r]) or seeds[r] not in continual:
                continue
            value = float(acc_sum[r]) / steps_hard
            fresh_rows.append({"arm": arm, "cond": cond, "seed": seeds[r], "lr": lr,
                               "task": last_hard, "fresh_online_acc": value,
                               "continual_online_acc": continual[seeds[r]],
                               "fresh_gap": value - continual[seeds[r]]})
        H.write_csv(out / "fresh_control.csv", fresh_rows)
        gaps = [q["fresh_gap"] for q in fresh_rows]
        progress(f"[{time.strftime('%T')}] {cap_arm} fresh control on task {last_hard}: "
                 f"gap median {float(np.median(gaps)) if gaps else float('nan'):+.3f} "
                 f"({sum(q > 0 for q in gaps)}/{len(gaps)} positive), rows written {fresh_written}")

    radii = {} if r_saved[0] is None else {
        f"r_l{l + 1}": r_saved[l][..., 0].cpu().numpy().astype(np.float32) for l in range(2)}
    if radii:
        np.savez(out / "radii.npz", seeds=np.asarray(seeds), **radii)
    prov = {"run_id": EXPERIMENT, **B.git_state(), "cap_arm": cap_arm,
            "cap_layers": [f"W{l + 1}" for l in layers], "radius_task": RADIUS_TASK,
            "cap_from_task": CAP_FROM,
            "cap_rule": "after every Adam update from the cap task on: n = ||w_i|| (float32); "
                        "rows with n > r_i become w_i * r_i / n; others untouched; W1 then W2; "
                        "biases, W3/b3, Adam moments, step counter, RNG untouched",
            "radii_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in radii.items()},
            "rows_written_total": {k: totals[k] for k in ("rows_w1", "rows_w2")},
            "violations_total": {k: totals[k] for k in ("viol_w1", "viol_w2")},
            "fresh_cap": "off: radius reset to +inf before the fresh control (spec 2.5)",
            "fresh_rows_written": fresh_written,
            "arm": arm, "cond": cond, "seeds": seeds, "R": R, "lr": lr, "n_tasks": n_tasks,
            "steps_per_task": C.STEPS_PER_TASK, "steps_hard": steps_hard,
            "steps_easy": steps_easy, "batch": C.BATCH, "hidden": C.HIDDEN,
            "layer_shapes": C.layer_shapes(arm), "n_params": C.n_params(arm),
            "dims": list(C.DIMS), "n_classes": C.N_CLASSES, "hard_classes": C.HARD_CLASSES,
            "per_class_train": C.PER_CLASS_TRAIN, "classes_used": C.CLASSES_USED,
            "task_parity": "task 1 hard, alternating (the paper's task 0 hard)",
            "optimizer": "adam", "weight_decay": 0.0, "data_sha256": cifar.sha256,
            "std": {"mean": C.STD_MEAN, "std": C.STD_STD, "planes": "R,G,B x 1024"},
            "class_sha256": {str(s): hashlib.sha256(
                np.asarray(C.seed_classes(s), dtype=np.int64).tobytes()).hexdigest() for s in seeds},
            "rng_roles": ["c51_classes", "c51_batch", "init"],
            "fresh_control": {"task": last_hard, "n": len(fresh_rows)} if fresh else None,
            "engine": "host run() of cifar5p1_mlp_0920 copied for R/std + row-norm projection; "
                      "stacked baddbmm, autograd, elementwise Adam"
                      + (", one step captured as a CUDA graph" if use_graph else ", eager"),
            "host_module_sha256": sha256_file(Path(C.__file__)),
            "engine_sha256": sha256_file(Path(__file__)),
            "device": str(device), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "step_ms_last_task": step_ms, "wall_clock_s": time.time() - t_start,
            "divergences": diverged, "spec": SPEC}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    progress(f"wrote {out}/per_task.csv ({len(rows)} rows, {(time.time() - t_start) / 60:.1f} min)")
    return prov


# --------------------------------------------------------------------------
# admission to the registered output (spec 9)
# --------------------------------------------------------------------------

def admit(cfg: dict, checks_json: Path = CHECKS_JSON, repo: Path = H.REPO) -> dict:
    """Refuse the registered output unless the checks passed on these very sources, the tree
    is clean and the configuration is the registered one.  Returns what was verified."""
    bad = [k for k, v in REGISTERED.items() if cfg.get(k) != v]
    if bad:
        raise SystemExit(f"ABORT: not the registered configuration: {bad}")
    if not checks_json.exists():
        raise SystemExit(f"ABORT: {checks_json} missing")
    ck = json.loads(checks_json.read_text())
    missing = [k for k in REQUIRED_CHECKS if k not in ck]
    if missing or not ck.get("all_pass"):
        raise SystemExit(f"ABORT: checks not all_pass (missing {missing})")
    now = {s: sha256_file(repo / s) for s in SOURCES}
    if ck.get("source_sha256") != now:
        raise SystemExit("ABORT: sources changed since the checks ran")
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src", "analysis",
                            "specs"], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"ABORT: uncommitted code or spec:\n{dirty}")
    return {"checks_sha256": sha256_file(checks_json), "source_sha256": now}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run"])
    ap.add_argument("--arm", required=True, choices=list(ARM_ORDER))
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--tasks", type=int, default=C.N_TASKS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--steps-hard", type=int, default=C.STEPS_PER_TASK)
    ap.add_argument("--steps-easy", type=int, default=C.STEPS_PER_TASK)
    ap.add_argument("--no-fresh", action="store_true")
    ap.add_argument("--no-graph", action="store_true", help="eager steps (checks)")
    ap.add_argument("--out", default=None, help="default results/cap_cifar5p1_1007/<arm>")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    return ap


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    torch.set_num_threads(a.threads)
    out = Path(a.out) if a.out else OUT_ROOT / a.arm
    cfg = dict(seeds=B.parse_ints(a.seeds), tasks=a.tasks, lr=a.lr, steps_hard=a.steps_hard,
               steps_easy=a.steps_easy, fresh=not a.no_fresh, graph=not a.no_graph,
               threads=torch.get_num_threads())
    admission = None
    if out.resolve().is_relative_to(OUT_ROOT.resolve()):
        admission = admit(cfg)
    device = H.setup(a.device)
    prov = run(a.arm, cfg["seeds"], cfg["tasks"], device, out, lr=cfg["lr"], fresh=cfg["fresh"],
               graph=cfg["graph"], steps_hard=cfg["steps_hard"], steps_easy=cfg["steps_easy"])
    if admission is not None:
        prov["admission"] = admission
        (out / "provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()

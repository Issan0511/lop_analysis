#!/usr/bin/env python3
"""adamw_dose2_1011, A2: coupled L2 toward 0 (2 lam theta added to the gradient, through Adam, all six tensors) on
the 5+1 CIFAR x MLP x ReLU box.

    python -m src.adamw_dose2_1011_run c51 --arm l2c1e-2 --seeds 0-9 --out results/adamw_dose2_1011/c51/l2c1e-2

Spec: specs/spec_adamw_dose2_1011.md (sections 2.4, 3.2).  The box is the host's (src/cifar5p1_mlp_0920.py: arm R,
cond std, Adam lr 1e-4, 30 tasks of 780 updates, R = 10 seeds stacked, one step captured as a CUDA graph, threads 2)
and nothing in it is modified.  The host's run() is copied for R/std as in adamw_dose_1010_c51 (P180), with the
update's form chosen per arm (ARM_SPEC):

    none        the host's R step, op for op (S-nochange)
    coupled     gr = gr.add(p.detach(), alpha=2.0 * lam) before m and v -- the host's own iv="l2" op (S-coupled-host)
    decoupled   p.mul_(wd_c), wd_c = fl32(1 - lr * lam), before m and v -- P180's AdamW (check only)
    after       the mutation control: the same 2 lam theta applied after Adam's step, outside the normalisation,
                p.sub_(p_old, alpha=lr * 2 * lam) -- algebraically decoupled with lam_W = 2 lam (check only)

A device counter per tensor (l2_steps) is incremented inside the graph with every L2 / decay op.  Fresh: the
lambda = 0 path writes the host's fresh_control.csv; every other arm writes fresh_self.csv (P = initial values,
m = v = 0, update count 0, the same L2 on), and its common fresh is R's own (the last hard task's batches' sha256 is
recorded by every arm).
"""

from __future__ import annotations

import hashlib
import json
import os
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

EXPERIMENT = "adamw_dose2_1011"
SPEC = "specs/spec_adamw_dose2_1011.md"
PREREG_COMMIT = "7525f9747a6a2872c30eb380fe3da4809bc6fa95"
OUT_ROOT = H.REPO / "results" / EXPERIMENT / "c51"
ACT_ARM, COND, LR = "R", "std", C.LR                 # the host's arm, condition and lr
THREADS = 2
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")
# the registered coupled-L2 arms (spec 2.4), the lambda = 0 path the reuse checks run, and the check-only forms
ARM_SPEC = {"nochange": ("none", 0.0), "l2c1e-2": ("coupled", 1e-2), "l2c5e-3": ("coupled", 5e-3),
            "mut_after1e-2": ("after", 1e-2), "dec2e-2": ("decoupled", 2e-2)}
ARM_LAMBDA = {k: v[1] for k, v in ARM_SPEC.items()}
REGISTERED_ARMS = ("l2c1e-2", "l2c5e-3")
REGISTERED = dict(tasks=C.N_TASKS, lr=LR, steps_hard=C.STEPS_PER_TASK, steps_easy=C.STEPS_PER_TASK,
                  fresh=True, graph=True, threads=THREADS)


def decay_factor(lr: float, lam: float) -> float:
    """1 - lr * lam as a python double; filling the float32 wd_c with it rounds once, to fl32(1 - lr * lam).
    Used only by the check-only 'decoupled' form."""
    return 1 - lr * lam


def c32(lr: float, lam: float) -> np.float32:
    return np.float32(decay_factor(lr, lam))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.no_grad()
def adam_update_(P, grads, adam_m, adam_v, inv_c1, inv_c2, lr: float, wd_c, mode: str, lam: float,
                 l2_count=None) -> None:
    """The host's elementwise Adam on the stacked tensors, with the arm's L2 form (module docstring)."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    for i, (p, gr, mi, vi) in enumerate(zip(P, grads, adam_m, adam_v)):
        if mode == "coupled":
            gr = gr.add(p.detach(), alpha=2.0 * lam)
        elif mode == "decoupled":
            p.mul_(wd_c)
        elif mode == "after":
            p_old = p.detach().clone()
        mi.mul_(b1).add_(gr, alpha=1 - b1)
        vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)
        p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))
        if mode == "after":
            p.sub_(p_old, alpha=lr * 2.0 * lam)
        if mode != "none" and l2_count is not None:
            l2_count[i].add_(1)


@torch.no_grad()
def wd_stats(P, Xt: torch.Tensor, act) -> dict:
    """Spec 3.2 l2_task: bias and weight norms, the second layer's relative position and open fraction (lists over
    slots)."""
    z1 = torch.baddbmm(P[1][:, None, :], Xt, P[0].transpose(1, 2))
    a1 = act.phi(z1, 0, False)
    z2 = torch.baddbmm(P[3][:, None, :], a1, P[2].transpose(1, 2))
    out = {}
    for li, z in ((1, z1), (2, z2)):
        zbar = z.mean(1).double().cpu()
        zsd = z.std(1).double().cpu()
        rel = []
        for r in range(z.shape[0]):
            ok = zsd[r] > 0
            v = zbar[r][ok] / zsd[r][ok]
            rel.append(float(v.median()) if len(v) else float("nan"))
        out[f"relpos_l{li}"] = rel
    out["p2"] = (z2 > 0).double().mean(dim=(1, 2)).cpu().tolist()
    for li in range(3):
        W = P[2 * li].double()
        out[f"w{li + 1}_fro"] = torch.linalg.vector_norm(W, dim=(1, 2)).cpu().tolist()
        out[f"w{li + 1}_row_med"] = W.norm(dim=2).cpu().median(dim=1).values.tolist()
        out[f"b{li + 1}_norm"] = torch.linalg.vector_norm(P[2 * li + 1].double(), dim=1).cpu().tolist()
    return out


# --------------------------------------------------------------------------
# the stacked run: the host's run() for R/std, plus the arm's L2 form
# --------------------------------------------------------------------------

def run(arm: str, seeds: list[int], n_tasks: int, device, out: Path, lr: float = LR,
        cifar: C.Cifar100 | None = None, fresh: bool = True, graph: bool = True,
        progress=None, debug: dict | None = None, observer=None,
        steps_hard: int = C.STEPS_PER_TASK, steps_easy: int = C.STEPS_PER_TASK,
        final: dict | None = None) -> dict:
    """Train one arm's R = len(seeds) runs in lockstep over the 30-task sequence.

    `debug` (checks only) makes the steps eager; debug["hook"], if given, is called as hook(event, **state) at
    "grads" (inside the step, the gradients before the update), "post_step", "task_end", "fresh_start" and
    "fresh_self_start".  `observer(t, state)` is called at every task end (diagnostics; the graph is kept).
    Neither writes to the training state."""
    t_start = time.time()
    progress = progress or (lambda m: print(m, flush=True))
    if arm not in ARM_LAMBDA:
        raise SystemExit(f"unknown arm {arm!r}; known: {','.join(ARM_LAMBDA)}")
    mode, lam = ARM_SPEC[arm]
    decay = mode != "none"                                   # every non-host form writes fresh_self, not fresh_control
    act = C.make_act(ACT_ARM)
    R = len(seeds)
    cifar = cifar or C.Cifar100()
    X_all = cifar.inputs("train", COND, device)
    Y_all = cifar.train_y.to(device)
    X_test = cifar.inputs("test", COND, device)
    Y_test = cifar.test_y.to(device)
    tr_rows, te_rows = C.class_rows(cifar.train_y), C.class_rows(cifar.test_y)
    plans = [C.task_plan(s) for s in seeds]
    g_batch = {s: H.stream("c51_batch", s) for s in seeds}

    init = [q.detach() for s in seeds for q in C.init_params(ACT_ARM, s, device)]
    P = [torch.stack(init[i::6]).contiguous() for i in range(6)]
    P = [q.requires_grad_(True) for q in P]
    P0 = [q.detach().clone() for q in P]
    act.init_state(R, device, key=f"{ACT_ARM}|{seeds}|{COND}")
    act0 = act.state()
    adam_m = [torch.zeros_like(q) for q in P]
    adam_v = [torch.zeros_like(q) for q in P]
    hook = None if debug is None else debug.get("hook")

    # ---- one training step on static tensors (the host's, plus the arm's L2 form)
    b1, b2, eps = 0.9, 0.999, 1e-8
    static_idx = torch.zeros(R, C.BATCH, dtype=torch.long, device=device)
    inv_c1 = torch.zeros((), device=device)
    inv_c2 = torch.zeros((), device=device)
    step_t = torch.zeros((), dtype=torch.long, device=device)
    acc_sum = torch.zeros(R, device=device)
    bad_step = torch.full((R,), -1, dtype=torch.long, device=device)
    last_hit = torch.zeros(R, device=device)
    wd_c = torch.ones((), device=device)                     # 1 while the graph is captured ('decoupled' only)
    l2_count = torch.zeros(6, dtype=torch.long, device=device)
    cur = {"tc": 0, "t": 0, "j": 0, "phase": ""}

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
            if hook is not None:
                hook("grads", grads=grads, P=P, m=adam_m, v=adam_v, inv_c1=inv_c1, inv_c2=inv_c2, wd_c=wd_c,
                     **cur)
            adam_update_(P, grads, adam_m, adam_v, inv_c1, inv_c2, lr, wd_c, mode, lam, l2_count)
            act.update(z1.detach(), z2.detach())
            if post is not None:
                post(P, lr)

    use_graph = graph and device.type == "cuda" and debug is None
    cg = None
    mutable = (*P, *adam_m, *adam_v, acc_sum, bad_step, step_t, last_hit, l2_count)
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
    c_lam = float(c32(lr, lam))
    if mode == "decoupled":
        wd_c.fill_(decay_factor(lr, lam))                     # in place: the captured step reads this address

    def state(**kw):
        return dict(P=P, m=adam_m, v=adam_v, wd_c=wd_c, l2_count=l2_count, **kw)

    def train_task(batches, tc0: int, phase: str, t: int) -> int:
        """`batches.shape[1]` steps on (R, steps, 32) global row indices."""
        tc = tc0
        acc_sum.zero_()
        bad_step.fill_(-1)
        step_t.zero_()
        l2_count.zero_()
        for j in range(batches.shape[1]):
            static_idx.copy_(batches[:, j])
            tc += 1
            inv_c1.fill_(1.0 / (1 - b1 ** tc))
            inv_c2.fill_(1.0 / (1 - b2 ** tc))
            if hook is not None:
                cur.update(tc=tc, t=t, j=j, phase=phase)
            act.begin_step(R, C.BATCH, device)
            if cg is not None:
                cg.replay()
            else:
                step()
            if hook is not None:
                hook("post_step", **state(t=t, j=j, tc=tc, phase=phase, idx=batches[:, j]))
        if device.type == "cuda":
            torch.cuda.synchronize()
        return tc

    alive = torch.ones(R, dtype=torch.bool, device=device)
    rows, l2_rows, diverged, tc = [], [], [], 0
    last_hard = max(t for t in range(1, n_tasks + 1) if t % 2 == 1)
    saved = {}
    out.mkdir(parents=True, exist_ok=True)
    step_ms = float("nan")

    for t in range(1, n_tasks + 1):
        hard = t % 2 == 1
        steps = steps_hard if hard else steps_easy
        rows_t = [torch.cat([tr_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
        batches = torch.stack([C.batch_indices(g_batch[s], rows_t[r], steps)
                               for r, s in enumerate(seeds)]).to(device)
        if t == last_hard and fresh:
            saved[t] = batches.clone()
        t0 = time.time()
        tc = train_task(batches, tc, "train", t)
        step_ms = 1e3 * (time.time() - t0) / steps
        l2_steps = [int(v) for v in l2_count.cpu().tolist()]

        with torch.no_grad():
            finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
            newly = alive & ((bad_step >= 0) | ~finite)
            for r in torch.nonzero(newly).flatten().tolist():
                diverged.append({"diverged": True, "task": t, "seed": seeds[r], "arm": ACT_ARM,
                                 "l2_arm": arm, "cond": COND,
                                 "step": (t - 1) * C.STEPS_PER_TASK + max(int(bad_step[r]), 0)})
                rows.append({"arm": ACT_ARM, "cond": COND, "seed": seeds[r], "slot": r, "lr": lr,
                             "task": t, "hard": int(hard), "acc": float("nan")})
            alive &= ~newly
            live = torch.nonzero(alive).flatten().tolist()
            Xt = torch.stack([X_all[rows_t[r]] for r in range(R)])
            Yt = torch.stack([Y_all[rows_t[r]] for r in range(R)])
            m = C.evaluate(P, Xt, Yt, act, live)
            ws = wd_stats(P, Xt, act)
            if observer is not None:
                observer(t, dict(P=P, m=adam_m, v=adam_v, Xt=Xt, act=act, live=live))
            te = [torch.cat([te_rows[q] for q in plans[r][t - 1][1]]) for r in range(R)]
            test_acc = C.accuracy(P, torch.stack([X_test[q] for q in te]),
                                  torch.stack([Y_test[q] for q in te]), act)
            if hook is not None:
                hook("task_end", **state(t=t, Xt=Xt, rows=m))
            del Xt
        for r in live:
            rows.append({"arm": ACT_ARM, "cond": COND, "seed": seeds[r], "slot": r, "lr": lr,
                         "task": t, "hard": int(hard), "n_classes": len(plans[r][t - 1][1]),
                         "online_acc": float(acc_sum[r]) / steps,
                         "train_acc": m[r]["acc"], "test_acc": float(test_acc[r]), **m[r]})
            wr = {"l2_arm": arm, "mode": mode, "lam": lam, "seed": seeds[r], "slot": r, "task": t,
                  "hard": int(hard), "steps": steps,
                  **{f"l2_steps_{nm}": l2_steps[i] for i, nm in enumerate(TENSORS)}}
            for k, v in ws.items():
                wr[k] = v[r]
            wr["dead_frac_l2"] = m[r]["dead_frac_l2"]
            l2_rows.append(wr)
        rows.sort(key=lambda q: (q["slot"], q["task"]))
        l2_rows.sort(key=lambda q: (q["slot"], q["task"]))
        H.write_csv(out / "per_task.csv", rows)
        H.write_csv(out / "l2_task.csv", l2_rows)
        on = acc_sum[alive] / steps
        el = time.time() - t_start
        progress(f"[{time.strftime('%T')}] {arm} task {t:2d}/{n_tasks} "
                 f"{'hard' if hard else 'easy'} alive {int(alive.sum())}/{R} "
                 f"online {float(on.mean()) if len(on) else float('nan'):.3f} "
                 f"l2_steps {l2_steps[0]}  {step_ms:.2f} ms/step  {el / 60:.1f} min")

    # ---- fresh controls on the last hard task (spec 2.3)
    fresh_rows, self_rows = [], []
    fresh_batches_sha = None
    if fresh and last_hard in saved:
        fresh_batches_sha = hashlib.sha256(saved[last_hard].cpu().contiguous().numpy().tobytes()).hexdigest()
        continual = {int(q["seed"]): q["online_acc"] for q in rows
                     if q["task"] == last_hard and "online_acc" in q}

        def reset_net() -> None:
            with torch.no_grad():
                for q, v in zip(P, P0):
                    q.copy_(v)
                for q in (*adam_m, *adam_v):
                    q.zero_()
            act.load_state(act0)

        reset_net()
        if not decay:
            # the host's procedure (lambda = 0 is the host's R; S-nochange compares this file byte for byte)
            if hook is not None:
                hook("fresh_start", **state())
            alive_f = torch.ones(R, dtype=torch.bool, device=device)
            train_task(saved[last_hard], 0, "fresh", last_hard)
            with torch.no_grad():
                finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
                alive_f &= (bad_step < 0) & finite
            for r in range(R):
                if not bool(alive_f[r]) or seeds[r] not in continual:
                    continue
                value = float(acc_sum[r]) / steps_hard
                fresh_rows.append({"arm": ACT_ARM, "cond": COND, "seed": seeds[r], "lr": lr,
                                   "task": last_hard, "fresh_online_acc": value,
                                   "continual_online_acc": continual[seeds[r]],
                                   "fresh_gap": value - continual[seeds[r]]})
            H.write_csv(out / "fresh_control.csv", fresh_rows)
            gaps = [q["fresh_gap"] for q in fresh_rows]
        else:
            # fresh_self: the same L2 form from the initial values (report only)
            if hook is not None:
                hook("fresh_self_start", **state(tc=0))
            alive_s = torch.ones(R, dtype=torch.bool, device=device)
            train_task(saved[last_hard], 0, "fresh_self", last_hard)
            l2_self = [int(v) for v in l2_count.cpu().tolist()]
            with torch.no_grad():
                finite = torch.stack([torch.isfinite(q).flatten(1).all(1) for q in P]).all(0)
                alive_s &= (bad_step < 0) & finite
            for r in range(R):
                if not bool(alive_s[r]) or seeds[r] not in continual:
                    continue
                value = float(acc_sum[r]) / steps_hard
                self_rows.append({"l2_arm": arm, "mode": mode, "lam": lam, "seed": seeds[r], "task": last_hard,
                                  "fresh_self_online_acc": value,
                                  "continual_online_acc": continual[seeds[r]],
                                  "gap_self": value - continual[seeds[r]],
                                  **{f"l2_steps_{nm}": l2_self[i] for i, nm in enumerate(TENSORS)}})
            H.write_csv(out / "fresh_self.csv", self_rows)
            gaps = [q["gap_self"] for q in self_rows]
        progress(f"[{time.strftime('%T')}] {arm} {'fresh control' if not decay else 'fresh_self'} on task "
                 f"{last_hard}: gap median {float(np.median(gaps)) if gaps else float('nan'):+.3f} "
                 f"({sum(q > 0 for q in gaps)}/{len(gaps)} positive)")

    prov = {"run_id": EXPERIMENT, **B.git_state(), "l2_arm": arm, "mode": mode, "lam": lam,
            "c_hex_decoupled_only": float(c_lam).hex() if mode == "decoupled" else None,
            "prereg_commit": PREREG_COMMIT,
            "l2_rule": {"none": "the host's R step, op for op",
                        "coupled": "gr = gr.add(p.detach(), alpha=2.0 * lam) before m and v on all six tensors "
                                   "(the host's iv='l2' op); a device counter per tensor",
                        "decoupled": "p.mul_(wd_c), wd_c = fl32(1 - lr*lam), before m and v (check only)",
                        "after": "p.sub_(p_old, alpha=lr * 2 * lam) after the Adam step (check only)"}[mode],
            "arm": ACT_ARM, "cond": COND, "seeds": seeds, "R": R, "lr": lr, "n_tasks": n_tasks,
            "steps_per_task": C.STEPS_PER_TASK, "steps_hard": steps_hard,
            "steps_easy": steps_easy, "batch": C.BATCH, "hidden": C.HIDDEN,
            "layer_shapes": C.layer_shapes(ACT_ARM), "n_params": C.n_params(ACT_ARM),
            "dims": list(C.DIMS), "n_classes": C.N_CLASSES, "hard_classes": C.HARD_CLASSES,
            "per_class_train": C.PER_CLASS_TRAIN, "classes_used": C.CLASSES_USED,
            "task_parity": "task 1 hard, alternating (the paper's task 0 hard)",
            "optimizer": {"none": "adam", "coupled": "adam with coupled L2 toward 0 (loss + lam ||theta||^2)",
                          "decoupled": "adam + decoupled weight decay (AdamW)",
                          "after": "adam, then 2 lam theta_old subtracted outside the normalisation"}[mode],
            "l2_lambda": lam, "data_sha256": cifar.sha256,
            "std": {"mean": C.STD_MEAN, "std": C.STD_STD, "planes": "R,G,B x 1024"},
            "class_sha256": {str(s): hashlib.sha256(
                np.asarray(C.seed_classes(s), dtype=np.int64).tobytes()).hexdigest() for s in seeds},
            "rng_roles": ["c51_classes", "c51_batch", "init"],
            "fresh_control": ({"task": last_hard, "n": len(fresh_rows), "rule": "host procedure (lambda = 0)"}
                              if (fresh and not decay) else None),
            "fresh_self": ({"task": last_hard, "n": len(self_rows), "rule": "the same L2 form from the initial "
                            "values, m = v = 0, update count 0"} if (fresh and decay) else None),
            "fresh_batches_sha256": fresh_batches_sha,
            "engine": "host run() of cifar5p1_mlp_0920 copied for R/std + the arm's L2 form; stacked baddbmm, "
                      "autograd, elementwise Adam"
                      + (", one step captured as a CUDA graph" if use_graph else ", eager"),
            "host_module_sha256": sha256_file(Path(C.__file__)),
            "engine_sha256": sha256_file(Path(__file__)),
            "device": str(device), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "step_ms_last_task": step_ms, "wall_clock_s": time.time() - t_start,
            "divergences": diverged, "spec": SPEC}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    if final is not None:                                       # checks: the end state
        final.update(P=[q.detach().clone() for q in P], m=[q.clone() for q in adam_m],
                     v=[q.clone() for q in adam_v])
    progress(f"wrote {out}/per_task.csv ({len(rows)} rows, {(time.time() - t_start) / 60:.1f} min)")
    return prov

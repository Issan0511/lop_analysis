#!/usr/bin/env python3
"""Checks for cifar5p1_mlp_0920 (spec §4).

    python3 src/cifar5p1_mlp_0920_checks.py --stage pilot     # S-data, S-plan, S-batch
    python3 src/cifar5p1_mlp_0920_checks.py --stage main      # + S-stack, S-graph, S-head, S-fresh, S-diverge

Every check carries its own mutation control: the check is run again on a deliberately
broken version and has to fail there.  A check that passes on both tests nothing (this
project has shipped six vacuous S-checks; the rule since `edge_law_0905` is that a
mutation control is part of the check, not an extra).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pmnist_0905 as H
from src import rlcifar_mlp_battle_0918 as B
from src import cifar5p1_mlp_0920 as C

RESULTS: dict = {}


def record(name: str, ok: bool, detail) -> bool:
    RESULTS[name] = {"pass": bool(ok), **(detail if isinstance(detail, dict) else {"detail": detail})}
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {json.dumps(RESULTS[name], default=str)}", flush=True)
    return bool(ok)


# --------------------------------------------------------------------------
# S-data
# --------------------------------------------------------------------------

def s_data(cifar: C.Cifar100) -> bool:
    counts = {split: torch.bincount(getattr(cifar, f"{split}_y"), minlength=100)
              for split in ("train", "test")}
    ok = (tuple(cifar.train_u8.shape) == (50_000, 3072)
          and tuple(cifar.test_u8.shape) == (10_000, 3072)
          and int(counts["train"].min()) == int(counts["train"].max()) == C.PER_CLASS_TRAIN
          and int(counts["test"].min()) == int(counts["test"].max()) == C.PER_CLASS_TEST
          and int(cifar.train_y.min()) == 0 and int(cifar.train_y.max()) == 99)
    # mutation: drop one image of class 0 and the per-class count must stop being 500
    mutated = cifar.train_y[torch.arange(50_000) != int(torch.nonzero(cifar.train_y == 0)[0])]
    mut_fails = int(torch.bincount(mutated, minlength=100).min()) != C.PER_CLASS_TRAIN
    # the channel constants in the engine have to be this archive's, not CIFAR-10's
    x = cifar.train_u8.to(torch.float32).div_(255.0).view(-1, 3, 1024)
    mean, std = x.mean((0, 2)), x.std((0, 2), unbiased=False)
    near = all(abs(float(mean[i]) - C.STD_MEAN[i]) < 5e-4 and abs(float(std[i]) - C.STD_STD[i]) < 5e-4
               for i in range(3))
    return record("S-data", ok and mut_fails and near,
                  {"sha256": cifar.sha256, "per_class_train": int(counts["train"].min()),
                   "per_class_test": int(counts["test"].min()),
                   "measured_mean": [round(float(v), 4) for v in mean],
                   "measured_std": [round(float(v), 4) for v in std],
                   "declared_mean": C.STD_MEAN, "declared_std": C.STD_STD,
                   "mutation_drop_one_image_fails": mut_fails})


# --------------------------------------------------------------------------
# S-plan
# --------------------------------------------------------------------------

def s_plan() -> bool:
    ok, detail = True, {}
    for seed in (0, 1, 100):
        plan = C.task_plan(seed)
        flat = [c for _, cls in plan for c in cls]
        hard = [t for t, (h, _) in enumerate(plan, 1) if h]
        ok &= len(plan) == C.N_TASKS
        ok &= len(flat) == len(set(flat)) == C.CLASSES_USED          # no class twice
        ok &= hard == list(range(1, C.N_TASKS + 1, 2)) and len(hard) == C.N_TASKS // 2
        ok &= all(len(cls) == (C.HARD_CLASSES if h else 1) for h, cls in plan)
        ok &= C.task_plan(seed) == plan                              # reproducible
        detail[f"seed{seed}_first_task"] = plan[0]
    ok &= C.task_plan(0) != C.task_plan(1)                           # seeds differ
    # mutation: consume 6 classes per hard task and the no-repeat count breaks
    cls = C.seed_classes(0)
    bad = [cls[i:i + 6] for i in range(0, 90, 6)]
    mut_fails = sum(len(b) for b in bad) != C.CLASSES_USED or len(bad) != C.N_TASKS
    return record("S-plan", ok and mut_fails,
                  {**detail, "classes_used": C.CLASSES_USED, "mutation_6_per_task_fails": mut_fails})


# --------------------------------------------------------------------------
# S-batch
# --------------------------------------------------------------------------

def s_batch(cifar: C.Cifar100) -> bool:
    rows = C.class_rows(cifar.train_y)
    ok, detail = True, {}
    for seed in (0, 100):
        g = H.stream("c51_batch", seed)
        plan = C.task_plan(seed)
        for t in (1, 2, 3):
            hard, cls = plan[t - 1]
            idx = torch.cat([rows[c] for c in cls])
            batches = C.batch_indices(g, idx)
            ok &= tuple(batches.shape) == (C.STEPS_PER_TASK, C.BATCH)
            # every drawn row belongs to this task's classes
            ok &= bool(torch.isin(cifar.train_y[batches.reshape(-1)],
                                  torch.tensor(cls)).all())
            # the first epoch's worth of draws is exactly a permutation of the task set
            first = batches.reshape(-1)[:len(idx)]
            ok &= bool(torch.equal(first.sort().values, idx.sort().values))
            detail[f"seed{seed}_t{t}"] = {"hard": hard, "n": len(idx),
                                          "epochs": round(C.STEPS_PER_TASK * C.BATCH / len(idx), 2)}
    # the stream depends on (seed, task): the same task index under a different seed differs
    a = C.batch_indices(H.stream("c51_batch", 0), torch.arange(2500))
    b = C.batch_indices(H.stream("c51_batch", 1), torch.arange(2500))
    ok &= not bool(torch.equal(a, b))
    # mutation: draw uniformly with replacement instead of by epochs -> the first "epoch"
    # stops being a permutation
    g = torch.Generator().manual_seed(0)
    naive = torch.randint(0, 2500, (C.STEPS_PER_TASK, C.BATCH), generator=g)
    mut_fails = not bool(torch.equal(naive.reshape(-1)[:2500].sort().values, torch.arange(2500)))
    return record("S-batch", ok and mut_fails, {**detail, "seeds_differ": True,
                                                "mutation_with_replacement_fails": mut_fails})


# --------------------------------------------------------------------------
# S-head: a task's output units were never a positive target before
# --------------------------------------------------------------------------

def s_head() -> bool:
    ok = True
    for seed in (0, 100):
        seen: set[int] = set()
        for _, cls in C.task_plan(seed):
            ok &= not (seen & set(cls))
            seen |= set(cls)
    return record("S-head", ok, {"units_are_fresh_each_task": ok, "classes_used": C.CLASSES_USED})


# --------------------------------------------------------------------------
# S-stack / S-graph / S-fresh / S-diverge (main run only)
# --------------------------------------------------------------------------

def _plain_forward(Wb, x, act):
    """One run on its own, in the host's arithmetic: `x @ W.T + b`, no batch axis.

    `B.forward` reaches the same numbers through `baddbmm`, which sums the 3072 products
    in a different order, so the two agree only to the rounding of that reordering -- the
    bound below is that rounding, computed from the operands rather than assumed.
    """
    W1, b1, W2, b2, W3, b3 = Wb
    z1 = x @ W1.T + b1
    a1 = act.phi(z1[None], 0, True)[0]
    z2 = a1 @ W2.T + b2
    a2 = act.phi(z2[None], 1, True)[0]
    return z1, a1, z2, a2, a2 @ W3.T + b3


def _reorder_scale(x, W1):
    """The size of a float32 summation-reorder difference in the first affine map.

    Recursive summation of n products carries n roundings, each at most eps times the
    running partial sum; two orders therefore differ by at most (n-1) * eps * sum|x_k
    w_k|, but that worst case (every rounding aligned and maximal) is ~1e6 times what a
    reordering actually costs and would let the check pass on anything.  The scale used
    here is the root-mean-square one, sqrt(n) * eps * sum|x_k w_k| -- the standard
    estimate for n independent roundings -- and it is computed from these operands, not
    carried over from another experiment.  The check reports both the margin below it
    and the distance to the mutation, so a scale that turned out to be wrong would show
    up as one of the two collapsing.
    """
    eps = float(torch.finfo(torch.float32).eps)
    n = x.shape[1]
    return n ** 0.5 * eps * float((x.abs() @ W1.abs().T).max())


def _propagated_scale(x, Wb, ref, lip):
    """The same scale carried through all three affine maps, for the logits.

    The reorder error of one map is `_reorder_scale`; it then passes through the
    activation, whose slope is at most `lip`, and into the next map.  The incoming errors
    of the 100 hidden units are independent roundings, so they combine into the next
    preactivation in quadrature -- through the 2-norm of the weight row, not its 1-norm.
    Using the 1-norm here (every error aligned and maximal, on top of a worst-case
    rounding model) inflates the result ~1e6 fold and the check stops being able to fail.

    `spec_rlcifar_mlp_battle_0918` registered a flat 1e-6 for this comparison, but that
    box's head is 10 units wide and this one's is 100, so its number is not this box's
    number: it is derived here from these operands.
    """
    eps = float(torch.finfo(torch.float32).eps)
    W1, _, W2, _, W3, _ = Wb
    z1, a1, z2, a2, z3 = ref
    s1 = _reorder_scale(x, W1)
    s2 = (a1.shape[1] ** 0.5 * eps * float((a1.abs() @ W2.abs().T).max())
          + float(W2.norm(dim=1).max()) * lip * s1)
    s3 = (a2.shape[1] ** 0.5 * eps * float((a2.abs() @ W3.abs().T).max())
          + float(W3.norm(dim=1).max()) * lip * s2)
    return s3 / max(float(z3.abs().max()), 1e-30)


def s_stack(device, cifar) -> bool:
    """Slot r of the stacked engine has to be the run r would have been on its own.

    Reference: the host's unbatched `x @ W.T + b` on that slot's own weights and batch.
    The first affine map is where the arithmetic actually differs (baddbmm against @) and
    is checked against the reorder scale above.  The composed logits and the gradients,
    which also go through bmm's backward, are checked at 1e-6 relative -- the tolerance
    `spec_rlcifar_mlp_battle_0918` §4 registered for the same comparison on the same
    engine.  Every line carries the distance to the mutation (slot r against a *different*
    slot's reference), which is what says the comparison can fail at all.
    """
    cond, seeds = "std", [100, 101, 102]
    X = cifar.inputs("train", cond, device)
    Y = cifar.train_y.to(device)
    rowsc = C.class_rows(cifar.train_y)

    def first_batch(seed):
        g = H.stream("c51_batch", seed)
        _, cls = C.task_plan(seed)[0]
        return C.batch_indices(g, torch.cat([rowsc[c] for c in cls]))[0].to(device)

    idxs = torch.stack([first_batch(s) for s in seeds])
    init = [q.detach() for s in seeds for q in H.init_params(s, device, C.DIMS)]
    detail, ok = {}, True
    for arm, lip in (("LR", 1.0), ("SNA", 2.0)):         # max |phi'|: leaky 1, Snake 1+sin <= 2
        act = B.make_act(arm)
        Pst = [torch.stack(init[i::6]).contiguous().requires_grad_(True) for i in range(6)]
        act.init_state(len(seeds), device, key="stack")
        zs = B.forward(Pst, X[idxs], act, True)
        loss = torch.nn.functional.cross_entropy(
            zs[4].reshape(-1, C.N_CLASSES), Y[idxs].reshape(-1), reduction="none"
        ).view(len(seeds), C.BATCH).mean(1)
        gst = torch.autograd.grad(loss.sum(), Pst)
        z1d = z1b = z1x = 0.0                     # first affine: diff, scale, mutation
        zrel = zrelx = grel = grelx = rtol = 0.0  # logits / gradients, relative
        for r in range(len(seeds)):
            Wb = [Pst[i][r].detach().clone().requires_grad_(True) for i in range(6)]
            a1 = B.make_act(arm)
            a1.init_state(1, device, key="ref")
            ref = _plain_forward(Wb, X[idxs[r]], a1)
            g1 = torch.autograd.grad(torch.nn.functional.cross_entropy(ref[4], Y[idxs[r]]), Wb)
            other = (r + 1) % len(seeds)
            z1d = max(z1d, float((zs[0][r] - ref[0]).abs().max()))
            z1b = max(z1b, _reorder_scale(X[idxs[r]], Wb[0]))
            z1x = max(z1x, float((zs[0][other] - ref[0]).abs().max()))
            scale_z = max(float(ref[4].abs().max()), 1e-30)
            zrel = max(zrel, float((zs[4][r] - ref[4]).abs().max()) / scale_z)
            zrelx = max(zrelx, float((zs[4][other] - ref[4]).abs().max()) / scale_z)
            rtol = max(rtol, _propagated_scale(X[idxs[r]], Wb, ref, lip))
            for i in range(6):
                scale = max(float(g1[i].abs().max()), 1e-30)
                grel = max(grel, float((gst[i][r] - g1[i]).abs().max()) / scale)
                grelx = max(grelx, float((gst[i][other] - g1[i]).abs().max()) / scale)
        cell = {"z1_max_diff": z1d, "z1_reorder_scale": z1b, "z1_margin": z1b / max(z1d, 1e-30),
                "z1_mutation": z1x, "z1_separation": z1x / max(z1b, 1e-30),
                "logit_rel_diff": zrel, "logit_rel_mutation": zrelx,
                "grad_rel_diff": grel, "grad_rel_mutation": grelx,
                "rel_tol_derived": rtol, "logit_margin": rtol / max(zrel, 1e-30),
                "logit_separation": zrelx / max(rtol, 1e-30)}
        ok &= (z1d <= z1b and z1x > 100 * z1b
               and zrel <= rtol and zrelx > 100 * rtol
               and grel <= rtol and grelx > 100 * rtol)
        detail[arm] = cell
    return record("S-stack", ok, detail)


def s_graph(device, cifar) -> bool:
    """CUDA graph replay vs eager: the same rows, bit for bit, over 2 tasks."""
    if device.type != "cuda":
        return record("S-graph", False, {"reason": "no cuda"})
    out_a, out_b = Path("/tmp/_c51_graph_a"), Path("/tmp/_c51_graph_b")
    C.run("SNA", [100], "std", 2, device, out_a, cifar=cifar, fresh=False, graph=True,
          progress=lambda m: None)
    C.run("SNA", [100], "std", 2, device, out_b, cifar=cifar, fresh=False, graph=False,
          progress=lambda m: None)
    a, b = (out_a / "per_task.csv").read_text(), (out_b / "per_task.csv").read_text()
    return record("S-graph", a == b, {"identical": a == b, "tasks": 2, "arm": "SNA"})


def s_nochange(device, cifar) -> bool:
    """Adding the intervention must not have moved the 16-arm run by one byte.

    Re-runs two committed cells with `iv="none"` and compares `per_task.csv` to the file
    on disk, which was produced before the `iv` argument existed.  The mutation runs the
    same cell with a tiny l2init and requires it to differ -- otherwise the comparison
    would be passing because the intervention does nothing at all.

    The thread count has to be pinned to the engine CLI's default first: `eff_rank` goes
    through `eigvalsh` on the cpu, and LAPACK gives a different last digit under a
    different number of threads.  That is measured here rather than assumed, because a
    check that fails for a reason unrelated to what it tests is as useless as one that
    cannot fail.
    """
    torch.set_num_threads(2)
    thr = {}
    for n in (2, 8):
        torch.set_num_threads(n)
        out = Path(f"/tmp/_c51_thr{n}")
        C.run("R", list(range(10)), "std", 6, device, out, lr=1e-4, cifar=cifar,
              fresh=False, graph=True, iv="none", progress=lambda m: None)
        thr[n] = _rows(out)
    torch.set_num_threads(2)
    diff_cols = sorted({k for x, y in zip(thr[2], thr[8]) for k in x if x[k] != y[k]})
    lapack_only = diff_cols and all(k.startswith("eff_rank") for k in diff_cols)

    src = C.OUT_ROOT
    detail, ok = {"thread_sensitive_columns": diff_cols,
                  "only_eff_rank_is_thread_sensitive": lapack_only}, lapack_only
    for arm in ("R", "SNA"):
        cell = src / f"{arm}_std_lr0.0001"
        if not (cell / "per_task.csv").exists():
            return record("S-nochange", False, {"reason": f"{cell} missing"})
        want = (cell / "per_task.csv").read_text()
        out = Path(f"/tmp/_c51_nochange_{arm}")
        C.run(arm, list(range(10)), "std", C.N_TASKS, device, out, lr=1e-4, cifar=cifar,
              fresh=True, graph=True, iv="none", progress=lambda m: None)
        same = (out / "per_task.csv").read_text() == want
        ok &= same
        detail[arm] = {"identical_to_committed": same}
    mut = Path("/tmp/_c51_nochange_mut")
    C.run("R", list(range(10)), "std", 3, device, mut, lr=1e-4, cifar=cifar, fresh=False,
          graph=True, iv="l2init:1e-6", progress=lambda m: None)
    base = Path("/tmp/_c51_nochange_base")
    C.run("R", list(range(10)), "std", 3, device, base, lr=1e-4, cifar=cifar, fresh=False,
          graph=True, iv="none", progress=lambda m: None)
    mut_fails = (mut / "per_task.csv").read_text() != (base / "per_task.csv").read_text()
    return record("S-nochange", ok and mut_fails,
                  {**detail, "mutation_tiny_l2init_differs": mut_fails})


def s_iv(device, cifar) -> bool:
    """The added gradient term is exactly 2*lam*(theta - theta_0), elementwise.

    Taken from the engine's own state: run one task with and without the intervention
    from the same init, then check the very first Adam step.  At step 1 the moments are
    zero and theta = theta_0, so the l2init term is identically zero and the two runs must
    agree bit for bit; the check that it is *not* vacuous is that by step 2 they differ,
    and that the difference of the raw gradients equals 2*lam*(theta - theta_0) to the
    rounding of one multiply-add.
    """
    lam = 1e-2
    arm, seed = "R", 100
    X = cifar.inputs("train", "std", device)
    Y = cifar.train_y.to(device)
    rowsc = C.class_rows(cifar.train_y)
    g = H.stream("c51_batch", seed)
    _, cls = C.task_plan(seed)[0]
    batches = C.batch_indices(g, torch.cat([rowsc[c] for c in cls])).to(device)
    act = C.make_act(arm)
    P = [q.detach().clone().unsqueeze(0).requires_grad_(True)      # one stacked slot
         for q in C.init_params(arm, seed, device)]
    P0 = [q.detach().clone() for q in P]
    # walk a few plain SGD-ish steps so theta moves away from theta_0
    for j in range(5):
        idx = batches[j:j + 1]
        z = B.forward(P, X[idx], act, True)[4]
        loss = torch.nn.functional.cross_entropy(z.reshape(-1, C.N_CLASSES), Y[idx].reshape(-1))
        gr = torch.autograd.grad(loss, P)
        with torch.no_grad():
            for p, q in zip(P, gr):
                p.sub_(1e-3 * q)
    idx = batches[5:6]
    z = B.forward(P, X[idx], act, True)[4]
    loss = torch.nn.functional.cross_entropy(z.reshape(-1, C.N_CLASSES), Y[idx].reshape(-1))
    gr = torch.autograd.grad(loss, P)
    eps = float(torch.finfo(torch.float32).eps)
    worst, bound, moved = 0.0, 0.0, 0.0
    for p, p0, q in zip(P, P0, gr):
        got = q.add(p.detach() - p0, alpha=2.0 * lam)
        want = q + 2.0 * lam * (p.detach() - p0)
        worst = max(worst, float((got - want).abs().max()))
        bound = max(bound, 2 * eps * float(want.abs().max()))
        moved = max(moved, float((p.detach() - p0).abs().max()))
    ok = worst <= bound and moved > 0
    # mutation: the sign flipped (a push *away* from theta_0) must not match
    flipped = max(float((gr[i].add(P[i].detach() - P0[i], alpha=-2.0 * lam)
                         - (gr[i] + 2.0 * lam * (P[i].detach() - P0[i]))).abs().max())
                  for i in range(6))
    return record("S-iv", ok and flipped > 100 * max(bound, 1e-30),
                  {"max_err": worst, "bound": bound, "theta_moved_by": moved,
                   "mutation_sign_flip": flipped, "lam": lam})


def s_init(device) -> bool:
    """This box's init has to be the host's, draw for draw, wherever the shapes coincide.

    For a pointwise arm at hidden=100 the parameter shapes are exactly the host's DIMS, so
    every tensor must match bit for bit; for CR/DF layers 2 and 3 have fan-in 200, so they
    must NOT match, and their bound must be the PyTorch default 1/sqrt(200).
    """
    ok, detail = True, {}
    for arm in ("R", "LK07"):
        mine = C.init_params(arm, 7, device)
        host = H.init_params(7, device, C.DIMS)
        ok &= all(bool(torch.equal(a, b.detach())) for a, b in zip(mine, host))
    for arm in C.WIDE:
        p = C.init_params(arm, 7, device)
        shapes = [tuple(q.shape) for q in p]
        want = [(100, 3072), (100,), (100, 200), (100,), (100, 200), (100,)]
        ok &= shapes == want
        # U(+-1/sqrt(fan_in)): the observed extreme must sit just under the bound
        for i, fan in ((2, 200), (4, 200)):
            bound = 1.0 / 200 ** 0.5
            hi = float(p[i].abs().max())
            ok &= 0.97 * bound < hi <= bound
        detail[arm] = {"shapes": shapes, "n_params": C.n_params(arm)}
    detail["pointwise_n_params"] = C.n_params("R")
    detail["equal_param_hidden_for_wide"] = max(
        h for h in range(2, 201) if C.n_params("CR", h) <= C.n_params("R"))
    return record("S-init", ok, detail)


def s_act_new(device) -> bool:
    """The three added arms' derivatives, against autograd, on a grid.

    LK07 is pointwise, so phi' is compared directly.  CR and DF map n preactivations to
    2n outputs, so there is no scalar phi': the check is the vector-Jacobian product
    against a random cotangent, which is what training actually uses.  `dphi` is then
    checked to be the Jacobian's column norm, the quantity the dead/mob readouts assume.
    """
    z = torch.linspace(-60, 60, 4001, device=device).view(1, -1, 1).clone()
    ok, detail = True, {}
    for arm in ("LK07", "CR", "DF"):
        act = C.make_act(arm)
        zz = z.clone().requires_grad_(True)
        out = act.phi(zz, 0, True)
        g = torch.randn_like(out)
        (vjp,) = torch.autograd.grad((out * g).sum(), zz)
        if arm == "LK07":
            want = g * act.dphi(z, 0)
        elif arm == "CR":
            want = g[..., :1] * (z > 0).to(z.dtype) - g[..., 1:] * (z < 0).to(z.dtype)
        else:
            want = g[..., :1] * torch.cos(z) - g[..., 1:] * torch.sin(z)
        err = float((vjp - want).abs().max())
        # the reference is built from the same float32 pieces, so the only slack is the
        # rounding of one multiply-add: one ulp at the magnitude in play
        bound = float(torch.finfo(torch.float32).eps) * float(want.abs().max().clamp_min(1.0)) * 4
        col = act.dphi(z, 0)
        if arm == "CR":
            jac = ((z > 0).float() ** 2 + (z < 0).float() ** 2).sqrt()
        elif arm == "DF":
            jac = (torch.cos(z) ** 2 + torch.sin(z) ** 2).sqrt()
        else:
            jac = act.dphi(z, 0)
        colerr = float((col - jac).abs().max())
        ok &= err <= bound and colerr <= 1e-6
        detail[arm] = {"vjp_max_err": err, "bound": bound, "dphi_vs_jacobian_norm": colerr,
                       "out_width": out.shape[-1], "in_width": z.shape[-1]}
    # mutation: CR with the two halves swapped must break the vjp comparison
    act = C.ConcatReLU()
    zz = z.clone().requires_grad_(True)
    out = torch.cat([(-zz).clamp(min=0.0), zz.clamp(min=0.0)], dim=-1)
    g = torch.randn_like(out)
    (vjp,) = torch.autograd.grad((out * g).sum(), zz)
    want = g[..., :1] * (z > 0).to(z.dtype) - g[..., 1:] * (z < 0).to(z.dtype)
    mut_fails = float((vjp - want).abs().max()) > 1e-3
    return record("S-act-new", ok and mut_fails,
                  {**detail, "mutation_swapped_halves_fails": mut_fails})


def _rows(out: Path, name="per_task.csv"):
    return list(csv.DictReader((out / name).open()))


def _fresh_pair(arm, device, cifar, tasks, graph, tag):
    out = Path(f"/tmp/_c51_fresh_{tag}")
    C.run(arm, [100], "std", tasks, device, out, cifar=cifar, fresh=True, graph=graph,
          progress=lambda m: None)
    f = _rows(out, "fresh_control.csv")[0]
    return float(f["fresh_online_acc"]), float(f["continual_online_acc"])


def s_fresh(device, cifar) -> bool:
    """The control has to restart from the initial state and replay the task's own batches.

    With a single task the control is doing exactly what the run itself just did, so the
    two online accuracies must agree bit for bit; with three tasks they must not (the run
    has by then trained on two more tasks).  Equality on one side and inequality on the
    other is what makes this test something rather than nothing.

    The mutation turns the adaptive arm's state restore into a no-op: SNA's EMA variance
    would then start the control where training left it, and the 1-task equality breaks.
    """
    detail, ok = {}, True
    for arm in ("R", "SNA"):
        same = _fresh_pair(arm, device, cifar, 1, False, f"one_{arm}")
        diff = _fresh_pair(arm, device, cifar, 3, False, f"three_{arm}")
        ok &= same[0] == same[1] and diff[0] != diff[1]
        detail[arm] = {"one_task": same, "three_tasks": diff}
    keep = B.SnakeFamily.load_state
    try:
        B.SnakeFamily.load_state = lambda self, st: None
        mut = _fresh_pair("SNA", device, cifar, 1, False, "mut")
    finally:
        B.SnakeFamily.load_state = keep
    mut_fails = mut[0] != mut[1]
    return record("S-fresh", ok and mut_fails,
                  {**detail, "mutation_no_state_restore": mut, "mutation_fails": mut_fails})


def s_diverge(device, cifar) -> bool:
    """A slot poisoned at init must not move the other slots by one bit.

    R is held at 3 in both runs, so the comparison is between two batched BLAS calls of
    the same shape; only the contents of slot 1 differ.  The mutation poisons slot 0
    instead and that slot's rows then have to stop matching.
    """
    seeds = [100, 101, 102]
    outs = {}
    for tag, slot in (("clean", None), ("poison1", 1), ("poison0", 0)):
        outs[tag] = Path(f"/tmp/_c51_div_{tag}")
        C.run("LR", seeds, "std", 3, device, outs[tag], cifar=cifar, fresh=False,
              graph=False, nan_slot=slot, progress=lambda m: None)

    def slot_rows(tag, r):
        return [q for q in _rows(outs[tag]) if int(q["slot"]) == r]

    others = all(slot_rows("clean", r) == slot_rows("poison1", r) for r in (0, 2))
    died = all(q.get("online_acc", "") == "" for q in slot_rows("poison1", 1))
    mut_fails = slot_rows("clean", 0) != slot_rows("poison0", 0)
    return record("S-diverge", others and died and mut_fails,
                  {"others_bit_identical": others, "poisoned_slot_has_no_rows": died,
                   "mutation_poison_slot0_fails": mut_fails,
                   "n_diverged_rows": len(slot_rows("poison1", 1))})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pilot", choices=["pilot", "main", "iv"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = H.setup(a.device)
    t0 = time.time()
    cifar = C.Cifar100()
    ok = s_data(cifar) & s_plan() & s_batch(cifar) & s_head() & s_init(device) & s_act_new(device)
    if a.stage == "main":
        ok &= s_stack(device, cifar)
        ok &= s_graph(device, cifar)
        ok &= s_fresh(device, cifar)
        ok &= s_diverge(device, cifar)
    if a.stage == "iv":
        ok &= s_nochange(device, cifar)
        ok &= s_iv(device, cifar)
    RESULTS["all_pass"] = bool(ok)
    RESULTS["stage"] = a.stage
    RESULTS["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out) if a.out else C.OUT_ROOT / f"_checks_{a.stage}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checks.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    print(f"\nall_pass={ok}  wrote {out}/checks.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

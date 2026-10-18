#!/usr/bin/env python3
"""Implementation checks for neokunekune_cifar5p1_cnn_1018 (spec §7).  Not performance evidence.

    python3 src/neokunekune_cifar5p1_cnn_1018_checks.py --out results/neokunekune_cifar5p1_cnn_1018/checks
    python3 src/neokunekune_cifar5p1_cnn_1018_checks.py --smoke results/neokunekune_cifar5p1_cnn_1018/smoke \
        --only smoke --out results/neokunekune_cifar5p1_cnn_1018/checks_smoke

Every check returns {"pass": bool, values..., "mutants": {name: {"detected": bool}}}.  A mutant is
a deliberately wrong variant the check must reject; all_pass = every check passes AND every
mutant is detected.  Same execution path -> bit equality; a different path (one run alone vs
the bundle) -> float64 on both sides and a tolerance derived from the arithmetic (first-order
running-error bounds, gamma_n = n u / (1 - n u) per inner product of length n, u = 2^-53;
cuDNN float64 runs implicit-GEMM / direct kernels only, so any-order summation bounds apply).
The float32 run path uses cuDNN Winograd for the 5x5 convolutions, which has no elementwise
bound of that form; its single-vs-bundle differences are reported, not gated.

  S-data      archive md5, shapes, 500/100 per class, the standardisation and plane layout
  S-plan      task plan, 90 classes without revisit, batch rows inside the task, epoch coverage,
              batch list = the host's draws
  S-init      init = the host rule (first four layers bit-equal to rlcifar_cnn_0908), heads
              100-wide, the four arms of a seed identical
  S-task1     a full task 1 in the bundle: the three Snake arms bit-identical, V = Vrec; then the
              calibration boundary and 100 updates of task 2: g from each run's own CE, mult,
              SNAfrz1's alpha / V fixed, SNA's V = Vrec
  S-gate      the gate formula, beta_eff, the floor, non-finite CE, independence (CPU)
  S-order     one update: alpha from the pre-update V, g from the pre-update CE of the same
              forward, Adam on that forward's gradient, V from that forward's variance
  S-pure      diagnostics change no state
  S-indep     changing one slot (parameters, then images) leaves the others bit-identical
  S-graph     CUDA graph = eager over task 1 + boundary + 100 updates, rule enabled after capture
  S-resume    stop after task 1 + resume = continuous, outputs identical; wrong config refused
  S-single    one run alone vs in the bundle vs an independent float64 reference: logits, CE,
              10 gradients within the derived bound
  S-stats     unit statistics vs an independent float64 reference within the derived bound
  S-eval      recorded train / test accuracy = an independent forward with the slot's own
              activation (float64, near-ties from the derived bound excluded)
  S-nonfinite a NaN run stops the run with failed_state.pt and status FAILED_NONFINITE
  S-fresh     the fresh control restarts from the initial state on the task's own batches
  S-lref      Lref recomputed on CPU from the saved CE (last 78, even median, floor)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import neokunekune_cifar5p1_cnn_1018 as NK
from src import cifar5p1_mlp_0920 as C51
from src import rlcifar_cnn_0908 as CN
from src import pmnist_0905 as H

E = NK.E
CPU = torch.device("cpu")
U64 = 2.0 ** -53
U32 = 2.0 ** -24
SEEDS2 = (9000, 9001)
SLOTS8 = [(a, s) for s in SEEDS2 for a in NK.ARMS]          # slot r = 4 * seed_index + arm_index


def gam(n: int, u: float) -> float:
    return n * u / (1.0 - n * u)


def teq(a, b) -> bool:
    if isinstance(a, dict):
        return set(a) == set(b) and all(teq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(teq(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    return a == b


def state_all(eng: NK.Engine) -> dict:
    return {"P": [p.detach().clone() for p in eng.P], "m": [q.clone() for q in eng.m],
            "v": [q.clone() for q in eng.v], "tc": eng.tc.clone(), "j": eng.j.clone(),
            "bad": eng.bad_step.clone(), "rec": {k: v.clone() for k, v in eng.rec.items()},
            "act": eng.act.state()}


def slot_view(st: dict, r: int) -> dict:
    """The part of state_all(...) that belongs to slot r."""
    return {"P": [p[r] for p in st["P"]], "m": [q[r] for q in st["m"]], "v": [q[r] for q in st["v"]],
            "rec": {k: v[r] for k, v in st["rec"].items()},
            "act": {k: [t[r] for t in vs] for k, vs in st["act"].items()}}


def slice_full_state(full: dict, r: int) -> dict:
    """Engine.full_state() of a bundle -> the full_state of a one-slot engine for slot r."""
    return {"P": [p[r:r + 1].clone() for p in full["P"]], "m": [q[r:r + 1].clone() for q in full["m"]],
            "v": [q[r:r + 1].clone() for q in full["v"]], "tc": full["tc"],
            "act": {k: [t[r:r + 1].clone() for t in vs] for k, vs in full["act"].items()},
            "slots": [full["slots"][r]]}


def cast_full_state(full: dict, dtype) -> dict:
    f = copy.deepcopy(full)
    for k in ("P", "m", "v"):
        f[k] = [t.to(dtype) for t in f[k]]
    for k, vs in f["act"].items():
        f["act"][k] = [t.to(dtype) if t.is_floating_point() else t for t in vs]
    return f


# --------------------------------------------------------------------------
# independent single-run reference (plain conv / pool / linear) and its error bounds
# --------------------------------------------------------------------------

def bc(a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return a.view(1, -1, 1, 1) if z.dim() == 4 else a.view(1, -1)


def ref_phi(z, l, arm, alpha):
    if arm == "R":
        return torch.relu(z)
    a = bc(alpha[l], z)
    return z + torch.sin(a * z) ** 2 / a


def ref_dphi(z, l, arm, alpha):
    if arm == "R":
        return (z > 0).to(z.dtype)
    return 1 + torch.sin(2 * bc(alpha[l], z) * z)


def ref_forward(Pr, x, arm, alpha) -> dict:
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = Pr
    f = {"x": x}
    f["z1"] = F.conv2d(x, W1, b1, padding=NK.PAD)
    f["a1"] = ref_phi(f["z1"], 0, arm, alpha)
    f["p1"], f["i1"] = F.max_pool2d(f["a1"], NK.POOL, NK.POOL, return_indices=True)
    f["z2"] = F.conv2d(f["p1"], W2, b2, padding=NK.PAD)
    f["a2"] = ref_phi(f["z2"], 1, arm, alpha)
    f["p2"], f["i2"] = F.max_pool2d(f["a2"], NK.POOL, NK.POOL, return_indices=True)
    f["h"] = f["p2"].flatten(1)
    f["z3"] = f["h"] @ W3.T + b3
    f["a3"] = ref_phi(f["z3"], 2, arm, alpha)
    f["z4"] = f["a3"] @ W4.T + b4
    f["a4"] = ref_phi(f["z4"], 3, arm, alpha)
    f["z5"] = f["a4"] @ W5.T + b5
    return f


def ref_loss_grads(Pr, x, y, arm, alpha):
    P = [p.detach().clone().requires_grad_(True) for p in Pr]
    f = ref_forward(P, x, arm, alpha)
    ce = F.cross_entropy(f["z5"], y, reduction="none")
    g = torch.autograd.grad(ce.mean(), P)
    return {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in f.items()}, \
        ce.detach(), [q.detach() for q in g]


def act_bounds(z, Ez, l, arm, alpha, u):
    """(E_a, phi'(z), E_phi') for a = phi(z) evaluated in floating point at a perturbed z."""
    if arm == "R":
        return Ez.clone(), (z > 0).to(z.dtype), (z.abs() <= Ez).to(z.dtype)
    a = bc(alpha[l], z)
    d = 1 + torch.sin(2 * a * z)
    L = torch.clamp(d.abs() + 2 * a * Ez, max=2.0)            # |phi'| <= 2, |phi''| <= 2 alpha
    Ea = L * Ez + 4 * u * (z.abs() + 2 / a)                  # + evaluation of z + sin^2(az)/a
    Ed = 2 * a * Ez + 2 * u * (2 * a * z).abs() + 8 * u     # + evaluation of 1 + sin(2az)
    return Ea, d, Ed


def pool_uncertain(a, Ea):
    """2x2 windows whose argmax may differ between two evaluations within Ea."""
    B, C, Hh, Ww = a.shape
    def win(t):
        return t.view(B, C, Hh // 2, 2, Ww // 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hh // 2, Ww // 2, 4)
    top2 = win(a).topk(2, dim=-1).values
    return (top2[..., 0] - top2[..., 1]) <= 2 * win(Ea).amax(-1)


def up2(t):
    return t.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


def unpool(src, idx, out_hw):
    """max_unpool2d for 2x2 / stride 2 (deterministic): position (h, w) receives src of its
    window iff it is that window's argmax."""
    H, W = out_hw
    flat = torch.arange(H * W).view(1, 1, H, W)
    return up2(src) * (up2(idx) == flat).to(src.dtype)


def bounds(Pr, x, y, f, arm, alpha, u) -> dict:
    """First-order elementwise bounds on |computed - exact| of z5 (logits), per-sample CE, the
    mean CE and the ten gradients of the mean CE, for a path with unit roundoff u whose inner
    products of length n are ordinary sums (error <= gamma_n * sum |terms|)."""
    A = [p.abs() for p in Pr]
    B = x.shape[0]
    Eb = {}
    Ez1 = gam(76, u) * F.conv2d(x.abs(), A[0], A[1], padding=NK.PAD)
    Ea1, d1, Ed1 = act_bounds(f["z1"], Ez1, 0, arm, alpha, u)
    Ep1 = F.max_pool2d(Ea1, 2, 2)
    Ez2 = gam(401, u) * F.conv2d(f["p1"].abs(), A[2], A[3], padding=NK.PAD) \
        + F.conv2d(Ep1, A[2], None, padding=NK.PAD)
    Ea2, d2, Ed2 = act_bounds(f["z2"], Ez2, 1, arm, alpha, u)
    Ep2 = F.max_pool2d(Ea2, 2, 2)
    Eh = Ep2.flatten(1)
    Ez3 = gam(1025, u) * (f["h"].abs() @ A[4].T + A[5]) + Eh @ A[4].T
    Ea3, d3, Ed3 = act_bounds(f["z3"], Ez3, 2, arm, alpha, u)
    Ez4 = gam(101, u) * (f["a3"].abs() @ A[6].T + A[7]) + Ea3 @ A[6].T
    Ea4, d4, Ed4 = act_bounds(f["z4"], Ez4, 3, arm, alpha, u)
    Ez5 = gam(101, u) * (f["a4"].abs() @ A[8].T + A[9]) + Ea4 @ A[8].T
    z5 = f["z5"]
    lse = torch.logsumexp(z5, 1)
    ce = lse - z5.gather(1, y[:, None])[:, 0]
    Einf = Ez5.amax(1)
    zmax = z5.abs().amax(1)
    Eb["z1"], Eb["z2"], Eb["z3"], Eb["z4"], Eb["z5"] = Ez1, Ez2, Ez3, Ez4, Ez5
    Eb["d"] = (Ed1, Ed2, Ed3, Ed4)
    Eb["ce"] = 2 * Einf + gam(110, u) * (2 + 2 * zmax + lse.abs() + ce.abs())
    Eb["loss"] = Eb["ce"].mean() + gam(B, u) * ce.abs().mean()
    # backward of the mean CE
    p = torch.softmax(z5, 1)
    Y = F.one_hot(y, NK.N_CLASSES).to(z5.dtype)
    d5 = (p - Y) / B
    c = 2 * Einf + gam(110, u) * (2 + 2 * zmax)
    Ed5 = p * (torch.exp(c) - 1)[:, None] / B + 2 * u * d5.abs()
    G = [None] * 10

    def fc_grads(i, dz, Edz, a_in, Ea_in):
        G[2 * i] = gam(B, u) * (dz.abs().T @ a_in.abs()) + Edz.T @ a_in.abs() + dz.abs().T @ Ea_in
        G[2 * i + 1] = gam(B, u) * dz.abs().sum(0) + Edz.sum(0)

    def back_act(da, Eda, d, Ed):
        return da * d, d.abs() * Eda + Ed * (da.abs() + Eda)

    W = Pr
    fc_grads(4, d5, Ed5, f["a4"], Ea4)
    da4, Eda4 = d5 @ W[8], gam(100, u) * (d5.abs() @ A[8]) + Ed5 @ A[8]
    dz4, Edz4 = back_act(da4, Eda4, d4, Ed4)
    fc_grads(3, dz4, Edz4, f["a3"], Ea3)
    da3, Eda3 = dz4 @ W[6], gam(100, u) * (dz4.abs() @ A[6]) + Edz4 @ A[6]
    dz3, Edz3 = back_act(da3, Eda3, d3, Ed3)
    fc_grads(2, dz3, Edz3, f["h"], Eh)
    dh, Edh = dz3 @ W[4], gam(100, u) * (dz3.abs() @ A[4]) + Edz3 @ A[4]
    dp2, Edp2 = dh.view_as(f["p2"]), Edh.view_as(f["p2"])
    unc2 = pool_uncertain(f["a2"], Ea2).to(z5.dtype)
    da2 = unpool(dp2, f["i2"], f["z2"].shape[-2:])
    Eda2 = unpool(Edp2, f["i2"], f["z2"].shape[-2:]) + up2((dp2.abs() + Edp2) * unc2)
    dz2, Edz2 = back_act(da2, Eda2, d2, Ed2)

    def cw(inp, go, shape):
        return torch.nn.grad.conv2d_weight(inp, shape, go, padding=NK.PAD)

    n2 = B * NK.CONV_HW[1] ** 2
    G[2] = gam(n2, u) * cw(f["p1"].abs(), dz2.abs(), W[2].shape) + cw(f["p1"].abs(), Edz2, W[2].shape) \
        + cw(Ep1, dz2.abs(), W[2].shape)
    G[3] = gam(n2, u) * dz2.abs().sum((0, 2, 3)) + Edz2.sum((0, 2, 3))
    dp1 = torch.nn.grad.conv2d_input(f["p1"].shape, W[2], dz2, padding=NK.PAD)
    Edp1 = gam(400, u) * torch.nn.grad.conv2d_input(f["p1"].shape, A[2], dz2.abs(), padding=NK.PAD) \
        + torch.nn.grad.conv2d_input(f["p1"].shape, A[2], Edz2, padding=NK.PAD)
    unc1 = pool_uncertain(f["a1"], Ea1).to(z5.dtype)
    da1 = unpool(dp1, f["i1"], f["z1"].shape[-2:])
    Eda1 = unpool(Edp1, f["i1"], f["z1"].shape[-2:]) + up2((dp1.abs() + Edp1) * unc1)
    dz1, Edz1 = back_act(da1, Eda1, d1, Ed1)
    n1 = B * NK.CONV_HW[0] ** 2
    G[0] = gam(n1, u) * cw(x.abs(), dz1.abs(), W[0].shape) + cw(x.abs(), Edz1, W[0].shape)
    G[1] = gam(n1, u) * dz1.abs().sum((0, 2, 3)) + Edz1.sum((0, 2, 3))
    Eb["grads"] = G
    # the manual backward is also an independent gradient: it must equal autograd's
    Eb["manual_grads"] = [d5.T @ f["a4"], d5.sum(0)]
    Eb["manual"] = {"W1": cw(x, dz1, W[0].shape), "b1": dz1.sum((0, 2, 3)),
                    "W2": cw(f["p1"], dz2, W[2].shape), "b2": dz2.sum((0, 2, 3)),
                    "W3": dz3.T @ f["h"], "b3": dz3.sum(0), "W4": dz4.T @ f["a3"], "b4": dz4.sum(0),
                    "W5": d5.T @ f["a4"], "b5": d5.sum(0)}
    return Eb


def ratio(a, b, Ebound) -> float:
    """max |a - b| / Ebound (0/0 -> 0, x/0 -> inf)."""
    d = (a.double() - b.double()).abs()
    Eb = Ebound.double()
    r = torch.where(Eb > 0, d / Eb.clamp_min(1e-300), torch.where(d > 0, torch.full_like(d, math.inf),
                                                                  torch.zeros_like(d)))
    return float(r.max())


# --------------------------------------------------------------------------
# bundle helpers
# --------------------------------------------------------------------------

class Ctx:
    def __init__(self, dev):
        self.dev = dev
        self.cifar = C51.Cifar100()
        self.D32 = NK.Data(dev, self.cifar)
        self._D64 = None
        self.prep = NK.prepare(list(SEEDS2), self.D32)

    @property
    def D64(self):
        if self._D64 is None:
            self._D64 = NK.Data(self.dev, self.cifar, dtype=torch.float64)
        return self._D64

    def batches(self, slots, t):
        return torch.stack([self.prep["batches"][s][t - 1] for a, s in slots]).to(self.dev)


def boundary(eng: NK.Engine) -> dict:
    """The task-1 boundary exactly as `run` does it, from the engine's task-1 records."""
    rec = eng.records()
    refs, control, frz, cal = {}, [], [], {}
    for r, (arm, seed) in enumerate(eng.slots):
        if arm in NK.SNAKE_ARMS:
            c = NK.reference_loss(rec["ce"][r])
            refs[r] = c["lref"]
            cal[r] = c
            if arm == "NEO":
                control.append(r)
        if arm == "SNAfrz1":
            frz.append(r)
    eng.act.set_reference(refs, control)
    eng.act.freeze_runs(frz)
    return cal


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_data(ctx: Ctx) -> dict:
    c = ctx.cifar
    path = C51.DATA_DIR / C51.ARCHIVE
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    published_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    shapes = tuple(c.train_u8.shape) == (50000, 3072) and tuple(c.test_u8.shape) == (10000, 3072)
    counts = (bool((torch.bincount(c.train_y, minlength=100) == 500).all())
              and bool((torch.bincount(c.test_y, minlength=100) == 100).all()))

    # PyTorch's CUDA `tensor / python scalar` multiplies by the float32 reciprocal; the host
    # (`Cifar100.inputs`) divides by 255 that way, so the bit-exact reference does the same.
    inv255 = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(255.0, dtype=torch.float32)

    def standardise(u8, mean, sd):
        x = u8.to(torch.float32) * inv255
        m = torch.tensor(mean).repeat_interleave(1024)
        s = torch.tensor(sd).repeat_interleave(1024)
        return ((x - m) / s).view(-1, 3, 32, 32)

    ref_tr = standardise(c.train_u8, C51.STD_MEAN, C51.STD_STD)
    ref_te = standardise(c.test_u8, C51.STD_MEAN, C51.STD_STD)
    std_eq = torch.equal(ref_tr, ctx.D32.X.cpu()) and torch.equal(ref_te, ctx.D32.Xte.cpu())
    # against the exact value: three roundings (multiply, subtract, divide) and the rounding of
    # 1/255, mean and sd to float32 (each <= u relative)
    u8 = c.train_u8.double()
    m64 = torch.tensor(C51.STD_MEAN, dtype=torch.float64).repeat_interleave(1024)
    s64 = torch.tensor(C51.STD_STD, dtype=torch.float64).repeat_interleave(1024)
    x64 = u8 / 255.0
    exact = ((x64 - m64) / s64).view(-1, 3, 32, 32)
    bound = (((3 * x64 + 2 * m64) * U32 + (x64 - m64).abs() * U32) / s64 * (1 + 2 * U32)
             + 2 * U32 * exact.abs().view(-1, 3072)).view(-1, 3, 32, 32)
    dev_exact = float(((ctx.D32.X.cpu().double() - exact).abs() / bound).max())
    # plane layout: channel k of image i is bytes [1024k, 1024(k+1)) row-major
    i = 12345
    lay = all(torch.equal(ctx.D32.X[i, k].cpu().flatten(),
                          ((c.train_u8[i, 1024 * k:1024 * (k + 1)].float() * inv255 - C51.STD_MEAN[k])
                           / C51.STD_STD[k]))
              for k in range(3))
    lab = torch.equal(ctx.D32.Y.cpu(), c.train_y) and torch.equal(ctx.D32.Yte.cpu(), c.test_y)
    mut = standardise(c.train_u8[:1000], (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    m_det = not torch.equal(mut, ctx.D32.X[:1000].cpu())
    return {"archive_md5": md5, "published_md5": published_md5, "md5_ok": md5 == published_md5,
            "sha256": c.sha256, "shapes": shapes, "per_class_500_100": counts,
            "standardisation_bit_equal": std_eq, "ratio_to_exact_bound": dev_exact,
            "plane_layout": lay, "labels": lab,
            "mutants": {"cifar10_constants": {"detected": m_det}},
            "pass": md5 == published_md5 and shapes and counts and std_eq and dev_exact <= 1.0 and lay and lab}


def plan_ok(plan, s) -> bool:
    cls = [c for _, cs in plan for c in cs]
    return (len(plan) == 30 and all(h == (t % 2 == 1) and len(cs) == (5 if h else 1)
                                   for t, (h, cs) in enumerate(plan, 1))
            and len(set(cls)) == 90 and all(0 <= c < 100 for c in cls)
            and cls == C51.seed_classes(s))


def batches_ok(b, plan, ctx) -> dict:
    y = ctx.cifar.train_y
    member, cover = True, True
    for t, (hard, cs) in enumerate(plan, 1):
        rows = torch.cat([ctx.D32.tr_rows[c] for c in cs])
        bt = b[t - 1].flatten()
        member &= bool(torch.isin(bt, rows).all()) and bool(torch.isin(y[bt], torch.tensor(cs)).all())
        cnt = torch.bincount(bt, minlength=50000)[rows]
        n = len(rows)
        full, rest = divmod(NK.STEPS * NK.BATCH, n)
        cover &= bool(((cnt == full) | (cnt == full + 1)).all()) and int((cnt == full + 1).sum()) == rest
    return {"member": member, "cover": cover}


def check_plan(ctx: Ctx) -> dict:
    seeds = list(NK.SEEDS) + list(SEEDS2)
    res, ok = {}, True
    for s in seeds:
        plan = NK.task_plan(s)
        b = NK.make_batches(s, plan, ctx.D32.tr_rows)
        # the host's own loop, literally (cifar5p1_mlp_0920.run)
        g = H.stream("c51_batch", s)
        host = torch.stack([C51.batch_indices(g, torch.cat([ctx.D32.tr_rows[q] for q in plan[t - 1][1]]),
                                              C51.STEPS_PER_TASK) for t in range(1, 31)])
        bo = batches_ok(b, plan, ctx)
        r = {"plan": plan_ok(plan, s), "shape": tuple(b.shape) == (30, 780, 32),
             "host_draws_equal": torch.equal(b, host), **bo}
        res[str(s)] = r
        ok &= all(r.values())
    s = 2000
    plan = NK.task_plan(s)
    b = NK.make_batches(s, plan, ctx.D32.tr_rows)
    bad = b.clone()
    bad[0, 5, 3] = ctx.D32.tr_rows[plan[1][1][0]][0]            # a row of task 2's class in task 1
    m1 = not batches_ok(bad, plan, ctx)["member"]
    plan2 = copy.deepcopy(plan)
    plan2[2] = (True, [plan[0][1][0]] + plan[2][1][1:])          # task 3 revisits a task-1 class
    m2 = not plan_ok(plan2, s)
    per_task = torch.stack([C51.batch_indices(H.stream("c51_batch", s),
                                              torch.cat([ctx.D32.tr_rows[q] for q in plan[t - 1][1]]),
                                              780) for t in range(1, 31)])
    m3 = not torch.equal(per_task, b)                            # a fresh stream per task
    return {"seeds": res, "mutants": {"row_outside_task": {"detected": m1},
                                      "class_revisited": {"detected": m2},
                                      "stream_restarted_per_task": {"detected": m3}},
            "pass": ok}


def check_init(ctx: Ctx) -> dict:
    res = {}
    for s in (2000, 9000):
        p, h = NK.init_params(s), CN.init_params(s, CPU)
        res[str(s)] = {
            "host_first_four_layers": all(torch.equal(a, b.detach()) for a, b in zip(p[:8], h[:8])),
            "shapes": [tuple(q.shape) for q in p] == [s_ for ws, bs in NK.SHAPES for s_ in (ws, bs)],
            "bounds": all(bool((q.abs() <= 1 / math.sqrt(f)).all())
                          for (q, f) in zip(p, [f for f in NK.FAN_IN for _ in (0, 1)]))}
    eng = NK.Engine(SLOTS8, ctx.D32, ctx.dev, graph=False)
    same = all(torch.equal(q[4 * k + i], q[4 * k]) for q in eng.P for k in range(2) for i in range(4))
    diff = all(not torch.equal(q[0], q[4]) for q in eng.P)
    m = not all(torch.equal(a, b) for a, b in zip(NK.init_params(2000), NK.init_params(2001)))
    ok = all(all(v.values()) for v in res.values()) and same and diff
    return {"per_seed": res, "four_arms_identical": same, "seeds_differ": diff,
            "mutants": {"other_seed": {"detected": m}}, "pass": ok}


def run_task1_boundary_t2(ctx: Ctx, graph: bool, n2: int = 100):
    eng = NK.Engine(SLOTS8, ctx.D32, ctx.dev, graph=graph)
    eng.train_task(ctx.batches(SLOTS8, 1))
    st1 = state_all(eng)
    st1["alpha"] = [a.clone() for a in eng.act.alpha_all()]
    cal = boundary(eng)
    st_b = state_all(eng)
    st_b["full"] = eng.full_state()
    eng.train_task(ctx.batches(SLOTS8, 2), n_steps=n2)
    return eng, st1, st_b, cal


def check_task1(ctx: Ctx, cache: dict) -> dict:
    eng, st1, st_b, cal = run_task1_boundary_t2(ctx, graph=True)
    cache["task1"] = (eng, st1, st_b, cal)
    ok_same, ok_v, ok_r = True, True, True
    for k in range(2):
        base = slot_view(st1, 4 * k + 1)                         # SNA of seed k
        for i in (2, 3):                                         # SNAfrz1, NEO
            ok_same &= teq(slot_view(st1, 4 * k + i), base)
        ok_r &= not teq(slot_view(st1, 4 * k)["P"], base["P"])
    for l in range(4):
        ok_v &= torch.equal(st1["act"]["V"][l], st1["act"]["Vrec"][l])
    gm1 = bool((st1["rec"]["gate"] == 1).all()) and bool((st1["rec"]["mult"] == 1).all())
    lref_same = all(cal[4 * k + 1]["lref"] == cal[4 * k + 2]["lref"] == cal[4 * k + 3]["lref"]
                    for k in range(2))
    # task 2 (100 updates): g from each run's own pre-update CE, in float32 as on the device
    rec = {k: v.detach().cpu().numpy() for k, v in eng.rec.items()}
    n2 = 100
    ok_g, ok_mult = True, True
    for r, (arm, seed) in enumerate(SLOTS8):
        ce = rec["ce"][r, :n2].astype(np.float32)
        if arm in NK.SNAKE_ARMS:
            lref = np.float32(cal[r]["lref"])
            g = np.minimum(np.float32(1.0), lref / np.maximum(ce, np.float32(NK.LOSS_FLOOR)))
        else:
            g = np.ones(n2, dtype=np.float32)
        ok_g &= np.array_equal(g, rec["gate"][r, :n2])
        want = {"NEO": g, "SNA": np.ones(n2, np.float32), "SNAfrz1": np.zeros(n2, np.float32),
                "R": np.ones(n2, np.float32)}[arm]
        ok_mult &= np.array_equal(want, rec["mult"][r, :n2])
    gate_active = bool((rec["mult"][[3, 7], :n2] < 1).any())
    frz_fixed = all(torch.equal(eng.act.V[l][4 * k + 2], st1["act"]["V"][l][4 * k + 2])
                    and torch.equal(eng.act.alpha(l)[4 * k + 2], st1["alpha"][l][4 * k + 2])
                    for l in range(4) for k in range(2))
    others_moved = all(not torch.equal(eng.act.alpha(l)[4 * k + i], st1["alpha"][l][4 * k + i])
                       for l in range(4) for k in range(2) for i in (1, 3))
    frz_rec_moves = all(not torch.equal(eng.act.Vrec[l][4 * k + 2], st_b["act"]["Vrec"][l][4 * k + 2])
                        for l in range(4) for k in range(2))
    sna_v = all(torch.equal(eng.act.V[l][4 * k + 1], eng.act.Vrec[l][4 * k + 1])
                for l in range(4) for k in range(2))
    neo_v = all(not torch.equal(eng.act.V[l][4 * k + 3], eng.act.Vrec[l][4 * k + 3])
                for l in range(4) for k in range(2))
    ok = (ok_same and ok_r and ok_v and gm1 and lref_same and ok_g and ok_mult and gate_active
          and frz_fixed and frz_rec_moves and sna_v and neo_v and others_moved)
    return {"snake_arms_bit_identical_task1": ok_same, "relu_differs": ok_r,
            "V_equals_Vrec_task1": ok_v, "gate_mult_one_task1": gm1,
            "lref_equal_across_snake_arms": lref_same,
            "lref": {f"{SLOTS8[r][0]}_seed{SLOTS8[r][1]}": cal[r]["lref"] for r in cal},
            "gate_from_own_ce_float32": ok_g, "mult_per_arm": ok_mult,
            "neo_gate_active_t2": gate_active, "frz_alpha_V_fixed": frz_fixed,
            "sna_neo_alpha_moved": others_moved,
            "frz_Vrec_moves": frz_rec_moves, "sna_V_equals_Vrec": sna_v, "neo_V_gated": neo_v,
            "pass": ok}


class ConstVar(NK.NeoAct):
    def batch_var(self, z, l):
        return torch.full((self.R, NK.WIDTHS[l]), 3.0)


def check_gate(ctx: Ctx) -> dict:
    arms = ["NEO", "NEO", "SNA", "SNAfrz1"]

    def make(cls=ConstVar):
        a = cls(arms, CPU)
        a.set_reference({0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}, [0, 1])
        a.freeze_runs([3])
        return a

    def zs():
        return [torch.zeros(1) for _ in range(4)]

    a = make()
    a.observe(torch.tensor([2.0, 0.25, 4.0, 1.0], requires_grad=True))
    g = a.gate[:, 0].clone()
    mult = a.mult[:, 0].clone()
    ok_g = torch.equal(g, torch.tensor([0.25, 1.0, 0.125, 0.5])) and not a.gate.requires_grad
    ok_m = torch.equal(mult, torch.tensor([0.25, 1.0, 1.0, 0.0]))
    a.update(zs())
    want = []
    for r, mu in enumerate((0.25, 1.0, 1.0, 0.0)):
        if r == 3:
            want.append(1.0)
        elif r in (0, 1):
            rate = np.float32(NK.BETA) * np.float32(mu)
            want.append(float(np.float32(1.0) * (np.float32(1.0) - rate) + rate * np.float32(3.0)))
        else:
            want.append(float(np.float32(np.float32(1.0) * np.float32(0.99)) + np.float32(0.01) * np.float32(3.0)))
    ok_v = all(torch.equal(a.V[l][:, 0], torch.tensor(want)) for l in range(4))
    exact = [1 + 0.02 * 0.25, 1.02, 1.02, 1.0]
    ok_v64 = all(abs(w - e) <= 2 * np.spacing(np.float32(e)) for w, e in zip(want, exact))
    ok_rec = all(torch.equal(a.Vrec[l][:, 0], torch.full((4,), want[2])) for l in range(4))
    # non-finite CE -> g = 0; uncalibrated -> g = 1; independence
    b = make()
    b.observe(torch.tensor([float("nan"), 0.25, float("inf"), 1.0]))
    ok_nf = float(b.gate[0, 0]) == 0.0 and float(b.gate[2, 0]) == 0.0
    c = ConstVar(arms, CPU)
    c.observe(torch.tensor([9.0, 9.0, 9.0, 9.0]))
    ok_unc = bool((c.gate == 1).all()) and bool((c.mult == 1).all())

    def indep(cls):
        x1, x2 = make(cls), make(cls)
        x1.observe(torch.tensor([2.0, 2.0, 2.0, 2.0]))
        x2.observe(torch.tensor([8.0, 2.0, 2.0, 2.0]))
        x1.update(zs()); x2.update(zs())
        own = float(x1.gate[0, 0]) != float(x2.gate[0, 0]) and not torch.equal(x1.V[0][0], x2.V[0][0])
        others = torch.equal(x1.gate[1:], x2.gate[1:]) and all(torch.equal(x1.V[l][1:], x2.V[l][1:])
                                                               for l in range(4))
        return own and others

    ok_ind = indep(ConstVar)

    class SharedGate(ConstVar):
        def observe(self, loss):
            super().observe(loss.detach().mean().expand(self.R))

    class NonfiniteOne(ConstVar):
        def observe(self, loss):
            super().observe(torch.where(torch.isfinite(loss), loss, torch.full_like(loss, 1e-9)))

    class WrongForm(ConstVar):
        @torch.no_grad()
        def update(self, zs_):
            for l, z in enumerate(zs_):
                var = self.batch_var(z, l)
                self.Vrec[l].mul_(1 - self.beta).add_(self.beta * var)
                new = self.V[l] * (1 - self.beta) + self.beta * self.mult * var
                self.V[l].copy_(torch.where(self.frozen, self.V[l], new))

    class AlphaOnlyFrozen(ConstVar):
        @torch.no_grad()
        def update(self, zs_):
            fr = self.frozen.clone()
            self.frozen.zero_()
            super().update(zs_)
            self.frozen.copy_(fr)

    def gv(cls):
        x = make(cls)
        x.observe(torch.tensor([2.0, 0.25, 4.0, 1.0]))
        x.update(zs())
        return x

    m_shared = not indep(SharedGate)
    y = make(NonfiniteOne)
    y.observe(torch.tensor([float("nan"), 0.25, 4.0, 1.0]))
    m_nf = float(y.gate[0, 0]) != 0.0
    m_form = not torch.equal(gv(WrongForm).V[0][:, 0], torch.tensor(want))
    m_frz = not torch.equal(gv(AlphaOnlyFrozen).V[0][:, 0], torch.tensor(want))
    ok = ok_g and ok_m and ok_v and ok_v64 and ok_rec and ok_nf and ok_unc and ok_ind
    return {"gate_formula": ok_g, "mult": ok_m, "V_update_float32": ok_v, "V_update_vs_exact": ok_v64,
            "Vrec_plain_ema": ok_rec, "nonfinite_gate_zero": ok_nf, "uncalibrated_gate_one": ok_unc,
            "independent_runs": ok_ind,
            "mutants": {"gate_shared_across_runs": {"detected": m_shared},
                        "nonfinite_not_zero": {"detected": m_nf},
                        "wrong_ema_form": {"detected": m_form},
                        "frozen_V_moves": {"detected": m_frz}},
            "pass": ok}


class MutantStep(NK.Engine):
    """Engine.step re-written with an optional wrong order; variant None must equal Engine.step."""
    variant = None

    def step(self):
        R, j = self.R, self.j
        idx = self.idx.index_select(1, j.view(1)).squeeze(1)
        xb, yb = self.data.X[idx], self.data.Y[idx]

        def ce_of(o):
            return F.cross_entropy(o[8].reshape(-1, NK.N_CLASSES), yb.reshape(-1),
                                   reduction="none").view(R, NK.BATCH).mean(1)
        if self.variant == "V_before_forward":
            with torch.no_grad():
                o0 = E.forward(self.P, xb, self.act)
                self.act.update([o0[0], o0[2], o0[4], o0[6]])
        out = E.forward(self.P, xb, self.act)
        lossv = ce_of(out)
        hit = (out[8].detach().argmax(-1) == yb).to(self.dtype).mean(1)
        if self.variant != "gate_from_post_update_ce":
            self.act.observe(lossv)
        grads = torch.autograd.grad(lossv.sum(), self.P)
        with torch.no_grad():
            self.tc.add_(1)
            c1 = (1 - torch.pow(self.b1, self.tc)).to(self.dtype)
            c2 = (1 - torch.pow(self.b2, self.tc)).to(self.dtype)
            for p, gr, mi, vi in zip(self.P, grads, self.m, self.v):
                mi.mul_(NK.ADAM_B1).add_(gr, alpha=1 - NK.ADAM_B1)
                vi.mul_(NK.ADAM_B2).addcmul_(gr, gr, value=1 - NK.ADAM_B2)
                p.sub_(self.lr * (mi / c1) / ((vi / c2).sqrt() + NK.ADAM_EPS))
            zs = [out[0].detach(), out[2].detach(), out[4].detach(), out[6].detach()]
            if self.variant in ("gate_from_post_update_ce", "var_from_post_update_forward"):
                o2 = E.forward(self.P, xb, self.act)
                if self.variant == "gate_from_post_update_ce":
                    self.act.observe(ce_of(o2))
                else:
                    zs = [o2[0], o2[2], o2[4], o2[6]]
            self.act.update(zs)
            pos = self.iota == j
            for k, val in (("ce", lossv.detach()), ("hit", hit),
                           ("gate", self.act.gate[:, 0]), ("mult", self.act.mult[:, 0])):
                self.rec[k].copy_(torch.where(pos, val[:, None], self.rec[k]))
            j.add_(1)


def mutant(variant):
    return type(f"Mutant_{variant}", (MutantStep,), {"variant": variant})


def order_expect(eng: NK.Engine, j: int) -> dict:
    """What one update must do, recomputed from the pre-update state without Engine.step."""
    R = eng.R
    idx = eng.idx[:, j]
    xb, yb = eng.data.X[idx], eng.data.Y[idx]
    Pc = [p.detach().clone().requires_grad_(True) for p in eng.P]
    out = E.forward(Pc, xb, eng.act)
    lossv = F.cross_entropy(out[8].reshape(-1, NK.N_CLASSES), yb.reshape(-1),
                            reduction="none").view(R, NK.BATCH).mean(1)
    grads = torch.autograd.grad(lossv.sum(), Pc)
    with torch.no_grad():
        L = lossv.detach()[:, None]
        g = torch.where(eng.act.calibrated,
                        (eng.act.loss_ref / L.clamp_min(NK.LOSS_FLOOR)).clamp(max=1.0), torch.ones_like(L))
        mult = torch.where(eng.act.frozen, torch.zeros_like(g), torch.where(eng.act.enabled, g, torch.ones_like(g)))
        tc = eng.tc + 1
        c1 = (1 - torch.pow(eng.b1, tc)).float()
        c2 = (1 - torch.pow(eng.b2, tc)).float()
        P_new = []
        for p, gr, mi, vi in zip(eng.P, grads, eng.m, eng.v):
            m1 = mi.clone().mul_(NK.ADAM_B1).add_(gr, alpha=1 - NK.ADAM_B1)
            v1 = vi.clone().mul_(NK.ADAM_B2).addcmul_(gr, gr, value=1 - NK.ADAM_B2)
            P_new.append(p.detach() - NK.LR * (m1 / c1) / ((v1 / c2).sqrt() + NK.ADAM_EPS))
        # V from this forward's variance, per run by explicit slicing, float64, and its bound
        V_exp, V_tol = [], []
        for l in range(4):
            z = out[2 * l].detach()
            Vl, Tl = [], []
            for r in range(R):
                if NK.IS_CONV[l]:
                    zr = z[:, r * NK.CH:(r + 1) * NK.CH].double().permute(1, 0, 2, 3).reshape(NK.CH, -1)
                else:
                    zr = z[r].double().T
                var = zr.var(1, unbiased=False)
                N = zr.shape[1]
                V0 = eng.act.V[l][r].double()
                mu = float(mult[r, 0])
                rate = NK.BETA * mu
                Vl.append(V0 * (1 - rate) + rate * var)
                # float32: Var of N values within gamma_{4N} (var + mean^2); 4 more roundings
                Tl.append(rate * gam(4 * N, U32) * (var + zr.mean(1) ** 2)
                          + gam(4, U32) * (V0 + rate * var))
            V_exp.append(torch.stack(Vl))
            V_tol.append(torch.stack(Tl))
    return {"ce": lossv.detach(), "gate": g[:, 0], "mult": mult[:, 0], "P": P_new, "V": V_exp,
            "Vtol": V_tol}


def check_order(ctx: Ctx, cache: dict) -> dict:
    eng0, _, st_b, _ = cache["task1"]
    full = st_b["full"]                       # right after the task-1 boundary
    J = 0                                     # first update of task 2: CE >> Lref, so g < 1

    def build(cls):
        e = cls(SLOTS8, ctx.D32, ctx.dev, graph=False)
        e.load_full_state(full)
        e.idx.copy_(ctx.batches(SLOTS8, 2))
        e.j.fill_(J)
        return e

    def judge(e):
        exp = order_expect(e, J)
        e.step()
        ce_ok = torch.equal(e.rec["ce"][:, J], exp["ce"])
        g_ok = torch.equal(e.rec["gate"][:, J], exp["gate"]) and torch.equal(e.rec["mult"][:, J], exp["mult"])
        p_ok = all(torch.equal(a.detach(), b) for a, b in zip(e.P, exp["P"]))
        v_r = max(float(((e.act.V[l].double() - exp["V"][l]).abs() / exp["Vtol"][l]).max()) for l in range(4))
        return {"ce_from_pre_update_state": ce_ok, "gate_mult_from_that_ce": g_ok,
                "adam_on_that_gradient": p_ok, "V_ratio_to_bound": v_r, "V_ok": v_r <= 1.0}

    real = judge(build(NK.Engine))
    harness = judge(build(mutant(None)))
    real["mutant_harness_faithful"] = (harness["ce_from_pre_update_state"] and harness["gate_mult_from_that_ce"]
                                       and harness["adam_on_that_gradient"] and harness["V_ok"])
    muts = {}
    for name in ("V_before_forward", "gate_from_post_update_ce", "var_from_post_update_forward"):
        r = judge(build(mutant(name)))
        muts[name] = {"detected": not (r["ce_from_pre_update_state"] and r["gate_mult_from_that_ce"]
                                       and r["adam_on_that_gradient"] and r["V_ok"]), **r}
    ok = real["ce_from_pre_update_state"] and real["gate_mult_from_that_ce"] and \
        real["adam_on_that_gradient"] and real["V_ok"] and real["mutant_harness_faithful"]
    g_lt1 = bool((eng0.rec["gate"][[3, 7], J] < 1).all())
    return {**real, "step_in_task2": J + 1, "neo_enabled": bool(eng0.act.enabled[[3, 7]].all()),
            "neo_gate_below_one": g_lt1, "mutants": muts,
            "pass": ok and bool(eng0.act.enabled[[3, 7]].all()) and g_lt1}


def check_pure(ctx: Ctx, cache: dict) -> dict:
    eng = cache["task1"][0]
    before = state_all(eng)
    probe = ctx.prep["probe"].to(ctx.dev)
    tr = NK.slot_rows(SLOTS8, ctx.prep["plans"], 2, ctx.D32.tr_rows).to(ctx.dev)
    te = NK.slot_rows(SLOTS8, ctx.prep["plans"], 2, ctx.D32.te_rows).to(ctx.dev)

    def diag(e):
        NK.unit_stats(e.P, e.act, ctx.D32.X, tr, ctx.D32.Y)
        NK.unit_stats(e.P, e.act, ctx.D32.X, probe[None].expand(e.R, -1)[:, :300])
        NK.ce_acc(e.P, e.act, ctx.D32.X, ctx.D32.Y, tr)
        NK.ce_acc(e.P, e.act, ctx.D32.Xte, ctx.D32.Yte, te)
        NK.control_arrays(e.act)
        NK.weight_row_norms(e.P)

    diag(eng)
    ok = teq(state_all(eng), before)
    orig = NK.unit_stats

    def leaky(P, act, X, rows, Y=None, chunk=50):
        out = E.forward(P, X[rows[:, :chunk]], act)
        act.update([out[0], out[2], out[4], out[6]])
        return orig(P, act, X, rows, Y, chunk)

    e2 = NK.Engine(SLOTS8, ctx.D32, ctx.dev, graph=False)
    e2.load_full_state(eng.full_state())
    b2 = state_all(e2)
    with patch.object(NK, "unit_stats", leaky):
        diag(e2)
    det = not teq(state_all(e2), b2)
    return {"state_unchanged": ok, "mutants": {"diagnostic_updates_V": {"detected": det}}, "pass": ok}


class Leak(NK.Engine):
    """mutant: every run's logits see 1e-6 x the mean logits over the runs."""
    def step(self):
        orig = E.forward

        def fwd(P, xb, act):
            out = list(orig(P, xb, act))
            out[8] = out[8] + 1e-6 * out[8].mean(0, keepdim=True)
            return tuple(out)
        with patch.object(E, "forward", fwd):
            super().step()


def check_indep(ctx: Ctx) -> dict:
    b1 = ctx.batches(SLOTS8, 1)

    def go(cls=NK.Engine, perturb=False, other_images=False, n=30):
        e = cls(SLOTS8, ctx.D32, ctx.dev, graph=False)
        bt = b1.clone()
        if perturb:
            with torch.no_grad():
                for p in e.P:
                    p[6].add_(1e-3 * torch.randn(p[6].shape, generator=torch.Generator().manual_seed(1)).to(p))
        if other_images:
            bt[6] = ctx.batches(SLOTS8, 3)[6]
        e.train_task(bt, n_steps=n)
        return state_all(e)

    base = go()
    others = [r for r in range(8) if r != 6]

    def compare(st, ref=None):
        ref = base if ref is None else ref
        same = all(teq(slot_view(st, r), slot_view(ref, r)) for r in others)
        moved = not teq(slot_view(st, 6)["P"], slot_view(ref, 6)["P"])
        return same, moved

    sp, mp = compare(go(perturb=True))
    si, mi = compare(go(other_images=True))
    sl, _ = compare(go(cls=Leak, perturb=True), go(cls=Leak))
    return {"others_bit_identical_params": sp, "slot_changed_params": mp,
            "others_bit_identical_images": si, "slot_changed_images": mi,
            "mutants": {"cross_run_leak": {"detected": not sl}},
            "pass": sp and mp and si and mi}


def check_graph(ctx: Ctx, cache: dict) -> dict:
    eg, st1g, _, _ = cache["task1"]
    ee, st1e, _, _ = run_task1_boundary_t2(ctx, graph=False)
    a = state_all(eg)
    b = state_all(ee)
    eq1 = teq(st1g, st1e)
    eq = teq(a, b)
    # the rule became active only after capture: the captured graph saw enabled = False
    active = bool((eg.rec["mult"][[3, 7], :100] < 1).any())
    return {"task1_bit_identical": eq1, "after_boundary_and_100_bit_identical": eq,
            "gate_active_after_capture": active, "graph_captured": eg.cg is not None,
            "pass": eq1 and eq and active and eg.cg is not None}


def check_resume(ctx: Ctx, work: Path) -> dict:
    slots = [(a, 9000) for a in NK.ARMS]
    a, b = work / "continuous", work / "resumed"
    for p in (a, b):
        shutil.rmtree(p, ignore_errors=True)
    quiet = lambda m: None
    NK.run(slots, a, 2, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True)
    stopped = False
    try:
        NK.run(slots, b, 2, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True, stop_after=1)
    except KeyboardInterrupt:
        stopped = True
    NK.run(slots, b, 2, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True)
    sa = torch.load(a / "state.pt", weights_only=False)
    sb = torch.load(b / "state.pt", weights_only=False)
    st_eq = teq(sa["state"], sb["state"])
    files = ["per_task.csv", "per_block.csv", "step_excerpt.csv", "site_stats.csv", "alpha_stats.csv",
             "fresh_control.csv", "calibration.json", "batch_hashes.json", "task_plan.json", "probe_ids.json"]
    txt = {f: (a / f).read_text() == (b / f).read_text() for f in files}
    npz = {}
    for f in ("per_step.npz", "unit_stats.npz"):
        with np.load(a / f) as za, np.load(b / f) as zb:
            npz[f] = set(za.files) == set(zb.files) and all(np.array_equal(za[k], zb[k]) for k in za.files)
    prov = json.loads((b / "provenance.json").read_text())
    resumed_logged = len(prov["resumed"]) == 1 and prov["resumed"][0]["at_task"] == 2
    refused = False
    try:
        NK.run(slots, b, 3, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True)
    except SystemExit:
        refused = True
    # mutant: a resume that does not restore the activation state
    c = work / "resumed_noact"
    shutil.rmtree(c, ignore_errors=True)
    try:
        NK.run(slots, c, 2, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True, stop_after=1)
    except KeyboardInterrupt:
        pass
    orig = NK.Engine.load_full_state

    def no_act(self, st):
        st = dict(st)
        st["act"] = NK.Engine(self.slots, self.data, self.device, graph=False).full_state()["act"]
        return orig(self, st)
    with patch.object(NK.Engine, "load_full_state", no_act):
        NK.run(slots, c, 2, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True)
    det = (c / "per_task.csv").read_text() != (a / "per_task.csv").read_text()
    ok = stopped and st_eq and all(txt.values()) and all(npz.values()) and resumed_logged and refused
    cache_dirs = {"continuous": str(a)}
    return {"stopped_by_hook": stopped, "state_bit_identical": st_eq, "files_identical": txt,
            "arrays_identical": npz, "resume_logged": resumed_logged, "other_config_refused": refused,
            "mutants": {"resume_without_activation_state": {"detected": det}}, "dirs": cache_dirs,
            "pass": ok}


def eng_forward_grads(e: NK.Engine, xb, yb):
    """loss (R,), logits (R, B, 100), grads (10 x (R, ...)) of the engine's own forward."""
    Pc = [p.detach().clone().requires_grad_(True) for p in e.P]
    out = E.forward(Pc, xb, e.act)
    lossv = F.cross_entropy(out[8].reshape(-1, NK.N_CLASSES), yb.reshape(-1),
                            reduction="none").view(e.R, -1)
    g = torch.autograd.grad(lossv.mean(1).sum(), Pc)
    return lossv.detach(), out[8].detach(), [q.detach() for q in g]


def check_single(ctx: Ctx, cache: dict) -> dict:
    """One run alone (R = 1) vs the same run in the bundle (R = 8), float64, from a common
    mid-run state (task 1 complete, boundary passed, 100 updates of task 2: alpha varies,
    NEO is gated, SNAfrz1 is frozen), against an independent float64 reference on the CPU."""
    eng32 = cache["task1"][0]
    full = cast_full_state(eng32.full_state(), torch.float64)
    D64 = ctx.D64
    bund = NK.Engine(SLOTS8, D64, ctx.dev, graph=False, dtype=torch.float64)
    bund.load_full_state(full)
    b2 = ctx.batches(SLOTS8, 2)
    idx = b2[:, 100]
    lb, zb, gb = eng_forward_grads(bund, D64.X[idx], D64.Y[idx])
    res, ok, worst = {}, True, 0.0
    muts = {"data_shift": True, "slot_swap": True}
    desc32 = {}
    single32_all = {}
    for r, (arm, seed) in enumerate(SLOTS8):
        one = NK.Engine([(arm, seed)], D64, ctx.dev, graph=False, dtype=torch.float64)
        one.load_full_state(slice_full_state(full, r))
        l1, z1, g1 = eng_forward_grads(one, D64.X[idx[r:r + 1]], D64.Y[idx[r:r + 1]])
        # independent reference on the CPU
        Pr = [p[r].detach().cpu() for p in bund.P]
        alpha = [a[r].detach().cpu() for a in bund.act.alpha_all()]
        x = D64.X[idx[r]].cpu()
        y = D64.Y[idx[r]].cpu()
        f, ce, gref = ref_loss_grads(Pr, x, y, arm, alpha)
        Eb = bounds(Pr, x, y, f, arm, alpha, U64)
        names = [f"{k}{i}" for i in range(1, 6) for k in ("W", "b")]
        man = Eb["manual"]
        man_ok = max(float((man[n].double() - g).abs().max() / g.abs().max().clamp_min(1e-300))
                     for n, g in zip(names, gref)) < 1e-10
        quantities = [("logits", zb[r].cpu(), z1[0].cpu(), f["z5"], Eb["z5"]),
                      ("ce", lb[r].cpu(), l1[0].cpu(), ce, Eb["ce"])]
        quantities += [(f"grad_{n}", gb[i][r].cpu(), g1[i][0].cpu(), gref[i], Eb["grads"][i])
                       for i, n in enumerate(names)]
        rr = {}
        for name, qb, q1, qref, Eq in quantities:
            r_bs = ratio(qb, q1, 2 * Eq)
            r_br = ratio(qb, qref, 2 * Eq)
            r_1r = ratio(q1, qref, 2 * Eq)
            rr[name] = {"bundle_vs_single": r_bs, "bundle_vs_ref": r_br, "single_vs_ref": r_1r,
                        "bound_rel": float(Eq.max() / qref.abs().max().clamp_min(1e-300))}
            worst = max(worst, r_bs, r_br, r_1r)
        res[f"{arm}_seed{seed}"] = {"manual_backward_equals_autograd": man_ok, "ratios": rr}
        ok &= man_ok and all(max(v["bundle_vs_single"], v["bundle_vs_ref"], v["single_vs_ref"]) <= 1.0
                             for v in rr.values())
        # mutants: the single run on the next batch; the bundle slot against another seed's reference
        ls, zs_, gs = eng_forward_grads(one, D64.X[b2[r:r + 1, 101]], D64.Y[b2[r:r + 1, 101]])
        muts["data_shift"] &= ratio(zs_[0].cpu(), f["z5"], 2 * Eb["z5"]) > 1.0 and \
            all(ratio(gs[i][0].cpu(), gref[i], 2 * Eb["grads"][i]) > 1.0 for i in range(10))
        r2 = (r + 4) % 8
        muts["slot_swap"] &= ratio(gb[0][r2].cpu(), gref[0], 2 * Eb["grads"][0]) > 1.0
        # float32: the run path (cuDNN Winograd in the bundle) -- reported, not gated
        e32 = NK.Engine([(arm, seed)], ctx.D32, ctx.dev, graph=False)
        e32.load_full_state(slice_full_state(eng32.full_state(), r))
        l32, z32, g32 = eng_forward_grads(e32, ctx.D32.X[idx[r:r + 1]], ctx.D32.Y[idx[r:r + 1]])
        single32_all[r] = (l32, z32, g32)
    lb32, zb32, gb32 = eng_forward_grads(eng32, ctx.D32.X[idx], ctx.D32.Y[idx])
    for r, (arm, seed) in enumerate(SLOTS8):
        l32, z32, g32 = single32_all[r]
        relmax = lambda a, b: float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-300))
        desc32[f"{arm}_seed{seed}"] = {
            "logits_bundle_vs_single_rel": relmax(zb32[r], z32[0]),
            "logits_bundle_vs_f64_rel": relmax(zb32[r].cpu(), zb[r].cpu()),
            "grad_max_bundle_vs_single_rel": max(relmax(gb32[i][r], g32[i][0]) for i in range(10)),
            "grad_max_bundle_vs_f64_rel": max(relmax(gb32[i][r].cpu(), gb[i][r].cpu()) for i in range(10))}
    return {"per_slot": res, "worst_ratio_to_bound": worst,
            "mutants": {"data_shift": {"detected": muts["data_shift"]},
                        "slot_swap": {"detected": muts["slot_swap"]}},
            "float32_descriptive": desc32,
            "note": "gated in float64 (cuDNN implicit-GEMM/direct kernels; bound u=2^-53); the "
                    "float32 run path uses Winograd 5x5 convolutions, reported only",
            "pass": ok}


def check_stats(ctx: Ctx, cache: dict) -> dict:
    """unit_stats (float64 bundle) vs an independent float64 reference.  For any seminorm-type
    statistic (population SD over images x positions, within-position SD, between-position SD)
    |s(z~) - s(z)| <= RMS(z~ - z) <= RMS(2 E_z); |m~ - m| <= mean 2 E_z; |dmean~ - dmean| <=
    mean 2 E_phi'; plus gamma_{4N} of the float64 sums."""
    eng32 = cache["task1"][0]
    full = cast_full_state(eng32.full_state(), torch.float64)
    D64 = ctx.D64
    bund = NK.Engine(SLOTS8, D64, ctx.dev, graph=False, dtype=torch.float64)
    bund.load_full_state(full)
    probe = ctx.prep["probe"][:256].to(ctx.dev)
    rows = probe[None].expand(8, -1)
    st, _ = NK.unit_stats(bund.P, bund.act, D64.X, rows, chunk=64)
    worst, ok = 0.0, True
    muts = {"unbiased_sd": True, "within_equals_pooled": True, "first_chunk_only": True}
    per = {}
    for r, (arm, seed) in enumerate(SLOTS8):
        Pr = [p[r].detach().cpu() for p in bund.P]
        alpha = [a[r].detach().cpu() for a in bund.act.alpha_all()]
        x = D64.X[probe].cpu()
        f = ref_forward(Pr, x, arm, alpha)
        y = torch.zeros(x.shape[0], dtype=torch.long)
        Eb = bounds(Pr, x, y, f, arm, alpha, U64)
        pr = {}
        for l, tag in enumerate(NK.SITES):
            z = f[f"z{l + 1}"]
            Ez = 2 * Eb[f"z{l + 1}"]
            d = ref_dphi(z, l, arm, alpha)
            Ed = 2 * Eb["d"][l]
            if NK.IS_CONV[l]:
                zc = z.permute(1, 0, 2, 3)                       # (C, N, H, W)
                m = zc.mean((1, 2, 3))
                s = zc.std((1, 2, 3), unbiased=False)
                pm = zc.mean(1)                                  # (C, H, W)
                s_w = ((zc - pm[:, None]) ** 2).mean((1, 2, 3)).sqrt()
                s_b = ((pm - m[:, None, None]) ** 2).mean((1, 2)).sqrt()
                rms = (Ez.permute(1, 0, 2, 3) ** 2).mean((1, 2, 3)).sqrt()
                mE = Ez.permute(1, 0, 2, 3).mean((1, 2, 3))
                dm = d.permute(1, 0, 2, 3).mean((1, 2, 3))
                dE = Ed.permute(1, 0, 2, 3).mean((1, 2, 3))
                N = z.numel() // NK.CH
                sq = (zc ** 2).mean((1, 2, 3))
                want = {"m": (m, mE), "s": (s, rms), "s_within": (s_w, rms), "s_between": (s_b, rms),
                        "dmean": (dm, dE)}
            else:
                m = z.mean(0)
                s = z.std(0, unbiased=False)
                rms = (Ez ** 2).mean(0).sqrt()
                N = z.shape[0]
                sq = (z ** 2).mean(0)
                want = {"m": (m, Ez.mean(0)), "s": (s, rms), "dmean": (d.mean(0), Ed.mean(0))}
            slack = gam(4 * N, U64) * (sq + 1.0).sqrt() * 4
            for k, (ref, tol) in want.items():
                got = torch.from_numpy(st[tag][k][r])
                rt = float(((got - ref).abs() / (tol + slack)).max())
                pr[f"{tag}_{k}"] = rt
                worst = max(worst, rt)
                ok &= rt <= 1.0
            # mutants
            got_s = torch.from_numpy(st[tag]["s"][r])
            unb = got_s * math.sqrt(N / (N - 1))
            muts["unbiased_sd"] &= float(((unb - s).abs() / (rms + slack)).max()) > 1.0
            if NK.IS_CONV[l]:
                muts["within_equals_pooled"] &= float(((got_s - s_w).abs() / (rms + slack)).max()) > 1.0
            m_first = z[:64].mean((0, 2, 3)) if NK.IS_CONV[l] else z[:64].mean(0)
            muts["first_chunk_only"] &= float(((m_first - m).abs() / (want["m"][1] + slack)).max()) > 1.0
        per[f"{arm}_seed{seed}"] = pr
    return {"per_slot_ratio_to_bound": per, "worst_ratio_to_bound": worst,
            "mutants": {k: {"detected": v} for k, v in muts.items()}, "pass": ok}


def check_eval(ctx: Ctx, run_dir: Path) -> dict:
    """The run's evaluation path (unit_stats for train accuracy, ce_acc for test accuracy) on
    the saved end state, executed in float64, = an independent float64 forward with each slot's
    own activation (samples whose top-2 logit gap is within 4 x the derived float64 logit bound
    count as near-ties).  The recorded float32 accuracies are compared descriptively."""
    import csv
    # the last saved HARD task: on an easy task (one class) a wrong activation can still score 1.0
    cks = [c for c in sorted((run_dir / "checkpoints").glob("t*.pt")) if int(c.stem[1:]) % 2 == 1]
    st = torch.load(cks[-1], map_location="cpu", weights_only=False)
    t = st["task"]
    full = cast_full_state(st["state"], torch.float64)
    slots = [tuple(x) for x in full["slots"]]
    recd = {(q["arm"], int(q["seed"])): q for q in csv.DictReader(open(run_dir / "per_task.csv"))
            if int(q["task"]) == t}
    D64 = ctx.D64
    plans = {s_: NK.task_plan(s_) for s_ in {s_ for a_, s_ in slots}}
    eng = NK.Engine(slots, D64, ctx.dev, graph=False, dtype=torch.float64)
    eng.load_full_state(full)
    tr = NK.slot_rows(slots, plans, t, D64.tr_rows).to(ctx.dev)
    te = NK.slot_rows(slots, plans, t, D64.te_rows).to(ctx.dev)
    _, tr_acc = NK.unit_stats(eng.P, eng.act, D64.X, tr, D64.Y)
    _, te_acc = NK.ce_acc(eng.P, eng.act, D64.Xte, D64.Yte, te)
    ok, out, ties = True, {}, 0
    wrong_act_differs = False
    for r, (arm, seed) in enumerate(slots):
        Pr = [p_[r].detach().cpu() for p_ in eng.P]
        alpha = [a_[r].detach().cpu() for a_ in eng.act.alpha_all()]
        res = {}
        for split, rows_t, Xs, Ys, got in (("train", tr[r].cpu(), D64.X, D64.Y, tr_acc[r]),
                                           ("test", te[r].cpu(), D64.Xte, D64.Yte, te_acc[r])):
            hits, hits_tie, n_tie, hits_wrong = 0, 0, 0, 0
            for i0 in range(0, len(rows_t), 250):
                ii = rows_t[i0:i0 + 250]
                x, y = Xs[ii.to(ctx.dev)].cpu(), Ys[ii.to(ctx.dev)].cpu()
                f = ref_forward(Pr, x, arm, alpha)
                Eb = bounds(Pr, x, y, f, arm, alpha, U64)
                top2 = f["z5"].topk(2, 1).values
                tie = (top2[:, 0] - top2[:, 1]) <= 4 * Eb["z5"].amax(1)
                right = f["z5"].argmax(1) == y
                n_tie += int(tie.sum())
                hits += int((right & ~tie).sum())
                hits_tie += int((right | tie).sum())
                if split == "train":
                    other = "SNA" if arm == "R" else "R"
                    hits_wrong += int((ref_forward(Pr, x, other, alpha)["z5"].argmax(1) == y).sum())
            n = len(rows_t)
            within = hits / n <= got <= hits_tie / n
            rec = float(recd[(arm, seed)][f"{split}_acc"])
            res[split] = {"engine_float64": float(got), "reference_float64": hits / n, "near_ties": n_tie,
                          "recorded_float32": rec, "recorded_minus_reference_samples": round((rec - hits / n) * n)}
            if split == "train":
                wrong_act_differs |= hits_wrong / n != float(got)
            ties += n_tie
            ok &= within
        out[f"{arm}_seed{seed}"] = res
    return {"task": t, "per_slot": out, "near_ties": ties,
            "mutants": {"evaluated_with_the_other_activation": {"detected": wrong_act_differs}},
            "pass": ok}


def check_nonfinite(ctx: Ctx, work: Path) -> dict:
    out = work / "nonfinite"
    shutil.rmtree(out, ignore_errors=True)
    slots = [(a, s) for s in SEEDS2 for a in NK.ARMS]
    orig = NK.init_params

    def poisoned(s):
        p = orig(s)
        if s == 9001:
            p[0] = p[0].clone()
            p[0][0, 0, 0, 0] = float("nan")
        return p
    raised = False
    with patch.object(NK, "init_params", poisoned):
        try:
            NK.run(slots, out, 1, ctx.dev, data=ctx.D32, log=lambda m: None, allow_dirty=True)
        except FloatingPointError:
            raised = True
    prov = json.loads((out / "provenance.json").read_text())
    fs = (out / "failed_state.pt").exists()
    bad = prov.get("failure", {}).get("bad_step_in_task", [])
    pattern = bad == [-1] * 4 + [0] * 4
    ok = raised and prov["status"] == "FAILED_NONFINITE" and fs and pattern
    return {"raised": raised, "status": prov["status"], "failed_state_saved": fs,
            "bad_step_in_task": bad, "only_poisoned_seed_flagged": pattern, "pass": ok}


def check_fresh(ctx: Ctx, run_dir: Path) -> dict:
    """In a 2-task run the fresh task is task 1, so a correct fresh control reproduces the
    continual task 1 bit for bit (same initial state, same batches, same kernels)."""
    import csv
    rows = list(csv.DictReader(open(run_dir / "fresh_control.csv")))
    hashes = json.loads((run_dir / "batch_hashes.json").read_text())["per_seed"]
    t = int(rows[0]["task"])
    same_hash = all(q["batch_sha256"] == hashes[q["seed"]][t - 1] for q in rows)
    twins = all(q["snake_fresh_identical_across_arms"] == "True" for q in rows if q["arm"] != "R")
    if t == 1:
        repro = all(float(q["fresh_gap_online"]) == 0.0 and q["fresh_online_acc"] == q["continual_online_acc"]
                    and q["fresh_test_acc"] == q["continual_test_acc"] for q in rows)
    else:
        repro = None
    # the initial state the fresh engine starts from = checkpoint t00
    st0 = torch.load(run_dir / "checkpoints" / "t00.pt", weights_only=False)["state"]
    slots = [tuple(s) for s in st0["slots"]]
    fe = NK.Engine(slots, ctx.D32, ctx.dev, graph=False)
    init_eq = teq(fe.full_state(), st0)
    # mutant: a fresh engine that keeps the end state's Adam moments
    det = None
    if t == 1:
        end = torch.load(sorted((run_dir / "checkpoints").glob("t*.pt"))[-1], weights_only=False)["state"]
        fe2 = NK.Engine(slots, ctx.D32, ctx.dev, graph=False)
        with torch.no_grad():
            for q, s in zip(fe2.m, end["m"]):
                q.copy_(s)
            for q, s in zip(fe2.v, end["v"]):
                q.copy_(s)
            fe2.tc.fill_(end["tc"])
        prep = NK.prepare(sorted({s for a, s in slots}), ctx.D32)
        fe2.train_task(torch.stack([prep["batches"][s][t - 1] for a, s in slots]).to(ctx.dev))
        on = fe2.records()["hit"].mean(1)
        cont = {(q["arm"], int(q["seed"])): float(q["continual_online_acc"]) for q in rows}
        det = any(abs(on[r] - cont[slots[r]]) > 0 for r in range(len(slots)))
    ok = same_hash and twins and init_eq and (repro is not False)
    return {"task": t, "batch_hash_is_the_task": same_hash, "snake_twins_identical": twins,
            "reproduces_continual_task1": repro, "initial_state_equals_t00": init_eq,
            "mutants": {"fresh_keeps_adam_moments": {"detected": det}} if det is not None else {},
            "pass": ok}


def check_lref(run_dir: Path) -> dict:
    cal = json.loads((run_dir / "calibration.json").read_text())["runs"]
    with np.load(run_dir / "per_step.npz") as z:
        ce = z["ce"]
        slots = list(z["slots"])
    ok, det = True, {"window_shifted_by_2": True, "first78": True, "mean_not_median": True,
                     "all780": True, "lower_middle": True}
    per = {}
    for name, c in cal.items():
        r = slots.index(name)
        x = ce[r, 0].astype(np.float64)
        w = np.sort(x[702:780])
        raw = 0.5 * (w[38] + w[39])
        lref = max(raw, NK.LOSS_FLOOR)
        good = (raw == c["lref_raw"] and lref == c["lref"] and np.float32(lref) == np.float32(c["lref_device"])
                and c["ce_window"] == x[702:780].tolist() and c["window_steps"] == [703, 780]
                and c["floor_hit"] == (raw < NK.LOSS_FLOOR))
        per[name] = {"recomputed": raw, "stored": c["lref_raw"], "ok": good}
        ok &= good
        det["window_shifted_by_2"] &= x[700:778].tolist() != c["ce_window"]
        det["first78"] &= float(np.median(x[:78])) != c["lref_raw"]
        det["mean_not_median"] &= float(x[702:780].mean()) != c["lref_raw"]
        det["all780"] &= float(np.median(x)) != c["lref_raw"]
        det["lower_middle"] &= float(torch.tensor(x[702:780]).median()) != c["lref_raw"]
    return {"per_run": per, "mutants": {k: {"detected": v} for k, v in det.items()}, "pass": ok}


def check_cost(run_dir: Path) -> dict:
    prov = json.loads((run_dir / "provenance.json").read_text())
    return {"timing": prov.get("timing"), "cuda_max_mem_mb": prov.get("cuda_max_mem_mb"),
            "wall_clock_s": prov.get("wall_clock_s_this_process"), "pass": True}


# --------------------------------------------------------------------------

def all_pass(res: dict) -> bool:
    ok = True
    for name, r in res.items():
        if not isinstance(r, dict):
            continue
        ok &= bool(r.get("pass"))
        for m in r.get("mutants", {}).values():
            ok &= m.get("detected") is True
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", default=None, help="a registered smoke run to verify (S-fresh/S-lref/S-eval/cost)")
    ap.add_argument("--only", default="all", choices=["all", "core", "smoke"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = NK.setup("cuda")
    t0 = time.time()
    ctx = Ctx(dev)
    work = Path(tempfile.mkdtemp(prefix="neo_checks_", dir=str(out)))
    res, cache = {}, {}
    plan = []
    if a.only in ("all", "core"):
        plan += [("S-data", lambda: check_data(ctx)), ("S-plan", lambda: check_plan(ctx)),
                 ("S-init", lambda: check_init(ctx)), ("S-gate", lambda: check_gate(ctx)),
                 ("S-task1", lambda: check_task1(ctx, cache)), ("S-order", lambda: check_order(ctx, cache)),
                 ("S-pure", lambda: check_pure(ctx, cache)), ("S-indep", lambda: check_indep(ctx)),
                 ("S-graph", lambda: check_graph(ctx, cache)),
                 ("S-single", lambda: check_single(ctx, cache)),
                 ("S-stats", lambda: check_stats(ctx, cache)),
                 ("S-resume", lambda: check_resume(ctx, work)),
                 ("S-nonfinite", lambda: check_nonfinite(ctx, work))]
        plan += [("S-fresh", lambda: check_fresh(ctx, work / "continuous")),
                 ("S-lref", lambda: check_lref(work / "continuous")),
                 ("S-eval", lambda: check_eval(ctx, work / "continuous"))]
    if a.smoke:
        sd = Path(a.smoke)
        plan += [("smoke/S-fresh", lambda: check_fresh(ctx, sd)), ("smoke/S-lref", lambda: check_lref(sd)),
                 ("smoke/S-eval", lambda: check_eval(ctx, sd)), ("smoke/S-cost", lambda: check_cost(sd))]
    for name, fn in plan:
        t1 = time.time()
        try:
            r = fn()
        except Exception as ex:                       # a crashed check is a failed check
            r = {"pass": False, "error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()}
        r["seconds"] = round(time.time() - t1, 1)
        res[name] = r
        mut = {k: v.get("detected") for k, v in r.get("mutants", {}).items()}
        print(f"[{time.strftime('%T')}] {name:14s} pass={r.get('pass')} mutants={mut} ({r['seconds']}s)"
              + (f"  ERROR {r['error']}" if "error" in r else ""), flush=True)
    summary = {"scope": "implementation checks only; no performance verdict",
               "all_pass": all_pass(res), "checks": res,
               "env": NK.env_info(dev), "git": NK.git_state(), "sources_sha256": NK.source_hashes(),
               "spec_sha256": hashlib.sha256(NK.SPEC.read_bytes()).hexdigest(),
               "seconds": round(time.time() - t0, 1)}
    NK.write_json(out / "checks.json", summary)
    shutil.rmtree(work, ignore_errors=True)
    print(f"all_pass={summary['all_pass']}  ->  {out / 'checks.json'}")


if __name__ == "__main__":
    main()

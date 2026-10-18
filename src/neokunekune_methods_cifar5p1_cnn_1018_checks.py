#!/usr/bin/env python3
"""Implementation checks of neokunekune_methods_cifar5p1_cnn_1018 (spec §7).  Not performance evidence.

    python3 src/neokunekune_methods_cifar5p1_cnn_1018_checks.py --out results/neokunekune_methods_cifar5p1_cnn_1018/checks

Each check returns {"pass": bool, ..., "mutants": {name: {"detected": bool}}}; all_pass = every check
passes and every mutant is detected.  Tolerances for different arithmetic paths are derived from
the operation count (u = 2^-53 in float64, 2^-24 in float32); same paths must agree bit for bit.

  S-nochange-smoke  the previous four arms in the previous smoke bundle = the previous smoke
  S-act             every kind's phi and the diagnostics' phi' vs independent float64 formulas
  S-rsl             noise in [l, u), per element / per update / per run, midpoint in evaluation
  S-redo            dormant sets, re-draws, zeroed bias / outgoing / Adam moments, timing, deferral
  S-l2i             the L2 Init gradient term 2 lam (theta - theta0), absent where lam = 0
  S-lr              per-run lr: the displacement scales with the run's own lr
  S-indep           one slot changed -> the others bit-identical (all kinds in the bundle)
  S-graph           CUDA graph = eager over task 1, the boundary and 100 updates (ReDo firing)
  S-resume          stop after task 1 + resume = continuous (RSL generators, ReDo streams)
  S-fresh           fresh = initial state; non-RSL arms reproduce the continual task 1 exactly
  S-eval            recorded accuracy and logits = an independent float64 forward per kind
"""
from __future__ import annotations

import argparse
import copy
import csv
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
from src import neokunekune_methods_cifar5p1_cnn_1018 as M
from src import neokunekune_cifar5p1_cnn_1018_checks as NKC     # previous helpers; not modified
from src import pmnist_0905 as H

NK, E = M.NK, M.E
CPU = torch.device("cpu")
U64, U32 = 2.0 ** -53, 2.0 ** -24
PI = math.pi
PREV_SMOKE = M.PREV_RESULTS / "smoke"
PREV_SMOKE_CKPT = Path("/home/issan/Projects/obsidian-research-data/neokunekune_cifar5p1_cnn_1018/results/"
                       "neokunekune_cifar5p1_cnn_1018/smoke/checkpoints/t02.pt")
SEEDS2 = NK.SMOKE_SEEDS
quiet = lambda m: None


def teq(a, b):
    return NKC.teq(a, b)


def state_all(e):
    st = NKC.state_all(e)
    st["n_done"] = e.n_done
    return st


# --------------------------------------------------------------------------
# independent float64 formulas of every kind
# --------------------------------------------------------------------------

def ref_phi(kind, z, a=None, p=None):
    """phi of one run: z any shape, a = alpha broadcastable to z (Snake kinds), p = shape params."""
    if kind == "relu":
        return torch.clamp(z, min=0.0)
    if kind in ("snake", "kkt1"):
        s = z + torch.sin(a * z) ** 2 / a
        if kind == "snake":
            return s
        th = 2 * a * z
        return torch.where(th < -2 * PI, z, torch.where(th > PI, z + 1 / a, s))
    if kind == "elu":
        return torch.where(z > 0, z, p[0] * (torch.exp(z) - 1))
    if kind == "leaky":
        return torch.where(z > 0, z, p[0] * z)
    if kind == "rsl":
        c, pp, l, u = p
        r = (l + u) / 2
        return r * z + (1 - r) * z / (1 + torch.exp(-c * z / pp))
    raise ValueError(kind)


def ref_dphi(kind, z, a=None, p=None):
    if kind == "relu":
        return (z > 0).to(z.dtype)
    if kind in ("snake", "kkt1"):
        d = 1 + torch.sin(2 * a * z)
        if kind == "snake":
            return d
        th = 2 * a * z
        return torch.where((th >= -2 * PI) & (th <= PI), d, torch.ones_like(z))
    if kind == "elu":
        return torch.where(z > 0, torch.ones_like(z), p[0] * torch.exp(z))
    if kind == "leaky":
        return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, p[0]))
    if kind == "rsl":
        c, pp, l, u = p
        r = (l + u) / 2
        s = 1 / (1 + torch.exp(-c * z / pp))
        return r + (1 - r) * (s + (c / pp) * z * s * (1 - s))
    raise ValueError(kind)


ACT_CFGS = ["R_lr0.0001", "SNA_lr0.0001", "SN06_lr0.0001", "SN3_lr0.0001", "KKT1_lr0.0001",
            "ELU_a3.6_lr0.001", "ELU_a1_lr0.001", "LK_a0.4_lr0.001", "LK_a0.1_lr0.001",
            "RSL_c0.5_p0.5_l0.673_u2.673_lr0.001", "RSL_c5_p3_l0.125_u0.333_lr0.001",
            "ReDo_t0_F1560_lr0.001", "L2I_lam0.01_lr0.001"]


def per_run(z, l, r, R):
    return z[:, r * NK.CH:(r + 1) * NK.CH] if NK.IS_CONV[l] else z[r]


def alpha_b(act, l, r, z):
    a = act.alpha(l)[r]
    return a.view(1, -1, 1, 1) if NK.IS_CONV[l] else a.view(1, -1)


def check_act(ctx) -> dict:
    cfgs = [M.parse_cfg(c) for c in ACT_CFGS]
    R = len(cfgs)
    act = M.MethodsAct(cfgs, CPU, dtype=torch.float64)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for l in range(4):                       # a spread of V so alpha covers the clip ends
            act.V[l].copy_(torch.exp(torch.randn(act.V[l].shape, generator=g, dtype=torch.float64) * 3))
    worst, worst_d, ok = 0.0, 0.0, True
    per = {}
    zs = {}
    for l in range(4):
        shp = (3, R * NK.CH, 4, 4) if NK.IS_CONV[l] else (R, 3, NK.WIDTHS[l])
        z = torch.randn(shp, generator=g, dtype=torch.float64) * 4
        z.view(-1)[::7] *= 10                     # some large |z| (KKT1 cuts, ELU tails)
        zs[l] = z
        zr = z.clone().requires_grad_(True)
        out = act.phi(zr, l)
        dz = torch.autograd.grad(out.sum(), zr)[0]
        dd = act.dphi(z, l)
        for r, c in enumerate(cfgs):
            zz = per_run(z, l, r, R)
            a = alpha_b(act, l, r, zz) if c["kind"] in ("snake", "kkt1") else None
            ref = ref_phi(c["kind"], zz, a, c["shape"])
            refd = ref_dphi(c["kind"], zz, a, c["shape"])
            got = per_run(out.detach(), l, r, R)
            inv = (1 / a).expand_as(zz) if a is not None else torch.zeros_like(zz)
            amax = a.expand_as(zz) if a is not None else torch.ones_like(zz)
            # <= 8 roundings, intermediates bounded by |z|, |phi|, 1, 1/alpha, alpha|z|, shape params
            scale = zz.abs() + ref.abs() + 1 + inv + amax * zz.abs() + max([abs(x) for x in c["shape"]] + [1.0])
            tol = 32 * U64 * scale
            rt = float(((got - ref).abs() / tol).max())
            rd = float(((per_run(dd, l, r, R) - refd).abs() / tol).max())
            ra = float(((per_run(dz, l, r, R) - refd).abs() / tol).max()) if c["kind"] != "relu" else \
                float(((per_run(dz, l, r, R) - refd).abs()).max())
            per[f"{c['name']}/{NK.SITES[l]}"] = {"phi": rt, "dphi": rd, "autograd": ra}
            worst = max(worst, rt)
            worst_d = max(worst_d, rd, ra)
            ok &= rt <= 1 and rd <= 1 and ra <= 1
    # mutants: wrong formulas must leave the tolerance
    def mut_ratio(cfgname, wrong):
        r = ACT_CFGS.index(cfgname)
        c = cfgs[r]
        m = 0.0
        for l in range(4):
            zz = per_run(zs[l], l, r, R)
            a = alpha_b(act, l, r, zz) if c["kind"] in ("snake", "kkt1") else None
            got = per_run(act.phi(zs[l], l), l, r, R)
            w = wrong(zz, a, c["shape"])
            m = max(m, float(((got - w).abs() / (32 * U64 * (zz.abs() + w.abs() + 1))).max()))
        return m > 1

    def kkt1_narrow(z, a, p):
        s = z + torch.sin(a * z) ** 2 / a
        th = 2 * a * z
        return torch.where(th < -PI, z, torch.where(th > PI, z + 1 / a, s))
    muts = {
        "kkt1_cut_at_minus_pi": mut_ratio("KKT1_lr0.0001", kkt1_narrow),
        "elu_without_alpha": mut_ratio("ELU_a3.6_lr0.001", lambda z, a, p: torch.where(z > 0, z, torch.exp(z) - 1)),
        "leaky_slope_1_minus_a": mut_ratio("LK_a0.4_lr0.001", lambda z, a, p: torch.where(z > 0, z, (1 - p[0]) * z)),
        "rsl_eval_at_l": mut_ratio("RSL_c0.5_p0.5_l0.673_u2.673_lr0.001",
                                   lambda z, a, p: p[2] * z + (1 - p[2]) * z / (1 + torch.exp(-p[0] * z / p[1]))),
        "rsl_c_p_swapped": mut_ratio("RSL_c5_p3_l0.125_u0.333_lr0.001",
                                     lambda z, a, p: ref_phi("rsl", z, None, (p[1], p[0], p[2], p[3]))),
    }
    return {"worst_phi_ratio": worst, "worst_derivative_ratio": worst_d, "per": per,
            "mutants": {k: {"detected": v} for k, v in muts.items()}, "pass": ok}


# --------------------------------------------------------------------------
# engine-level checks
# --------------------------------------------------------------------------

class Ctx:
    def __init__(self, dev):
        self.dev = dev
        self.cifar = NK.C51.Cifar100()
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


def check_nochange(ctx, work: Path) -> dict:
    out = work / "nochange"
    shutil.rmtree(out, ignore_errors=True)
    slots = [(f"{a}_lr0.0001", s) for a in NK.ARMS for s in SEEDS2]       # the previous smoke's order
    M.run(slots, out, 2, ctx.dev, data=ctx.D32, allow_dirty=True, log=quiet)

    def compare(new_dir):
        old = list(csv.DictReader(open(PREV_SMOKE / "per_task.csv")))
        new = list(csv.DictReader(open(new_dir / "per_task.csv")))
        cols = [c for c in old[0] if c in new[0] and c not in ("arm", "slot", "classes")]
        bad = 0
        for o in old:
            n = [q for q in new if q["cfg"] == f"{o['arm']}_lr0.0001" and q["seed"] == o["seed"]
                 and q["task"] == o["task"]][0]
            bad += sum(o[c] != n[c] for c in cols)
        old_st = torch.load(PREV_SMOKE_CKPT, weights_only=False)["state"]
        new_st = torch.load(new_dir / "checkpoints" / "t02.pt", weights_only=False)["state"]
        eq = all(torch.equal(a, b) for k in ("P", "m", "v") for a, b in zip(old_st[k], new_st[k])) and \
            all(torch.equal(a, b) for k in old_st["act"] for a, b in zip(old_st["act"][k], new_st["act"][k]))
        return {"rows": len(old), "columns": len(cols), "mismatches": bad, "end_state_bit_equal": eq}

    real = compare(out)
    # mutant: SNA's lr 1% off
    mut = work / "nochange_mut"
    shutil.rmtree(mut, ignore_errors=True)
    slots_m = [(f"{a}_lr{'0.000101' if a == 'SNA' else '0.0001'}", s) for a in NK.ARMS for s in SEEDS2]
    M.run(slots_m, mut, 2, ctx.dev, data=ctx.D32, allow_dirty=True, log=quiet)
    old = list(csv.DictReader(open(PREV_SMOKE / "per_task.csv")))
    new = list(csv.DictReader(open(mut / "per_task.csv")))
    det = any(o["online_acc"] != n["online_acc"] or o["ce_mean"] != n["ce_mean"]
              for o in old if o["arm"] == "SNA"
              for n in new if n["cfg"] == "SNA_lr0.000101" and n["seed"] == o["seed"] and n["task"] == o["task"])
    return {**real, "mutants": {"sna_lr_one_percent_off": {"detected": det}},
            "pass": real["mismatches"] == 0 and real["end_state_bit_equal"]}


def check_rsl(ctx) -> dict:
    c1, c2 = "RSL_c0.5_p0.5_l0.673_u2.673_lr0.001", "RSL_c5_p3_l0.125_u0.333_lr0.001"
    slots = [(c1, 9000), (c2, 9000), ("R_lr0.0001", 9000), (c1, 9001)]
    e = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=False)
    e.refresh_noise()
    n1 = [b.clone() for b in e.act.noise]
    e.refresh_noise()
    n2 = [b.clone() for b in e.act.noise]
    ok_range, ok_vary, ok_steps = True, True, True
    for r in (0, 1, 3):
        cfg = e.cfgs[r]
        lo, up = cfg["shape"][2], cfg["shape"][3]
        for l in range(4):
            nz = n1[l][:, r * NK.CH:(r + 1) * NK.CH] if NK.IS_CONV[l] else n1[l][r]
            rr = e.act._seg(e.act.rsl_l, l, r, r + 1) + e.act._seg(e.act.rsl_w, l, r, r + 1) * nz
            ok_range &= bool((nz >= 0).all() and (nz < 1).all()) and \
                bool((rr >= lo - 1e-6).all() and (rr <= up + 1e-6).all())
            ok_vary &= float(nz.std()) > 0.2
            nz2 = n2[l][:, r * NK.CH:(r + 1) * NK.CH] if NK.IS_CONV[l] else n2[l][r]
            ok_steps &= not torch.equal(nz, nz2)
    # per-run generator: the same (cfg, seed, stage) run in another bundle draws the same noise
    e2 = M.MethodsEngine([("R_lr0.0001", 9001), (c1, 9000)], ctx.D32, ctx.dev, graph=False)
    e2.refresh_noise()
    same = all(torch.equal(e2.act.noise[l][:, 1 * NK.CH:2 * NK.CH] if NK.IS_CONV[l] else e2.act.noise[l][1],
                           n1[l][:, 0:NK.CH] if NK.IS_CONV[l] else n1[l][0]) for l in range(4))
    other_seed_differs = not torch.equal(n1[2][0], n1[2][3])
    # evaluation uses the midpoint: phi in eval mode = the reference with r = (l+u)/2
    z = torch.randn(3, 4 * NK.CH, 4, 4, device=ctx.dev)
    mid = e.act.phi(z, 0)
    ref = ref_phi("rsl", z[:, 0:NK.CH].double(), None, e.cfgs[0]["shape"])
    mid_ok = float((mid[:, 0:NK.CH].double() - ref).abs().max()) <= 64 * U32 * float(z.abs().max() + 1)
    # mutants
    e3 = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=False)
    with patch.object(M.MethodsEngine, "refresh_noise", lambda self: None):
        e3.refresh_noise()
    det_norefresh = torch.equal(e3.act.noise[2], torch.zeros_like(e3.act.noise[2]))   # never drawn
    e4 = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=False)
    g = e4.rsl_gen[0]
    for r in e4.rsl_runs:
        e4.rsl_gen[r] = g                              # one generator shared by all runs
    e4.refresh_noise()
    shared_same_as_real = torch.equal(e4.act.noise[2][3], n1[2][3])
    ok = ok_range and ok_vary and ok_steps and same and other_seed_differs and mid_ok
    return {"range": ok_range, "varies_per_element": ok_vary, "changes_per_update": ok_steps,
            "per_run_stream": same, "seeds_differ": other_seed_differs, "eval_midpoint": mid_ok,
            "mutants": {"noise_never_refreshed": {"detected": det_norefresh},
                        "generator_shared_across_runs": {"detected": not shared_same_as_real}},
            "pass": ok}


def redo_reference(e_before: dict, scores, runs, eng: M.MethodsEngine) -> dict:
    """What redo_apply must produce, written independently from the pre-state."""
    P = [p.clone() for p in e_before["P"]]
    m = [q.clone() for q in e_before["m"]]
    v = [q.clone() for q in e_before["v"]]
    dorm = {}
    for r in runs:
        tau, F_ = eng.redo[r]
        cfg, seed = eng.slots[r]
        g = H.stream(f"neo_m_redo_reinit|{eng.stage}|{cfg}", seed)
        dl = [torch.nonzero(scores[l][r] <= tau).flatten().cpu() for l in range(4)]
        dorm[r] = dl
        for l in range(4):
            if len(dl[l]):
                fan = NK.FAN_IN[l]
                fresh = ((torch.rand((len(dl[l]), fan), generator=g, dtype=torch.float32) * 2 - 1)
                         * (1.0 / math.sqrt(fan)))
                W = P[2 * l]
                for k, i in enumerate(dl[l].tolist()):
                    W[r, i] = fresh[k].view(W.shape[2:]).to(W.device)
                    P[2 * l + 1][r, i] = 0.0
                    m[2 * l][r, i] = 0.0
                    v[2 * l][r, i] = 0.0
        for l in range(4):
            for i in dl[l].tolist():
                nxt = 2 * l + 2
                if l == 1:
                    sl = slice(i * 64, (i + 1) * 64)
                    for T in (P[nxt], m[nxt], v[nxt]):
                        T[r][:, sl] = 0.0
                else:
                    for T in (P[nxt], m[nxt], v[nxt]):
                        T[r][:, i] = 0.0
    return {"P": P, "m": m, "v": v, "dormant": dorm}


def check_redo(ctx) -> dict:
    slots = [("ReDo_t0_F1560_lr0.001", 9000), ("ReDo_t0.1_F1560_lr0.001", 9000), ("R_lr0.001", 9000),
             ("ReDo_t0.025_F1000_lr0.001", 9001)]
    e = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=False)
    b1 = ctx.batches(slots, 1)
    e.train_task(b1, n_steps=40)
    with torch.no_grad():                                # construct dormant units in runs 0, 1, 3
        for r in (0, 1, 2, 3):
            e.P[1][r, 3] = -1e3                          # c1 channel 3 dead
            e.P[3][r, 5] = -1e3                          # c2 channel 5 dead
            e.P[7][r, 7] = -1e3                          # f2 unit 7 dead
            e.P[5][r, 11] = -1e3                         # f1 unit 11 dead
    rows = b1[:, 38:40].reshape(e.R, -1)
    before = {"P": [p.detach().clone() for p in e.P], "m": [q.clone() for q in e.m],
              "v": [q.clone() for q in e.v], "draws": {r: 0 for r in range(e.R)}}
    scores = e.redo_scores(rows)
    runs = [0, 1, 3]
    ref = redo_reference(before, scores, runs, e)
    e.redo_apply(runs, rows)
    eqP = all(torch.equal(a.detach(), b) for a, b in zip(e.P, ref["P"]))
    eqm = all(torch.equal(a, b) for a, b in zip(e.m, ref["m"]))
    eqv = all(torch.equal(a, b) for a, b in zip(e.v, ref["v"]))
    constructed = all(3 in ref["dormant"][r][0].tolist() and 5 in ref["dormant"][r][1].tolist()
                      and 11 in ref["dormant"][r][2].tolist() and 7 in ref["dormant"][r][3].tolist() for r in runs)
    untouched = all(torch.equal(a.detach()[2], b[2]) for a, b in zip(e.P, before["P"]))   # the R run
    # scores from an independent float64 forward of run 0 on the same 64 images
    Pr = [p[0].detach().double().cpu() for p in before["P"]]
    f = NKC.ref_forward(Pr, ctx.D32.X[rows[0]].double().cpu(), "R", None)
    s_ref = []
    for l, a in enumerate((f["a1"], f["a2"], f["a3"], f["a4"])):
        s = a.abs().mean((0, 2, 3)) if NK.IS_CONV[l] else a.abs().mean(0)
        s_ref.append(s / (s.mean() + 1e-9))
    tau_set_ok = all(torch.equal((s_ref[l] <= 0.0), (scores[l][0].double().cpu() <= 0.0)) for l in range(4))
    rel = max(float(((scores[l][0].double().cpu() - s_ref[l]).abs() / (s_ref[l].abs() + 1e-6)).max()) for l in range(4))
    # timing: checks after updates n with n % F == 0, deferred on a task's last update
    e5 = M.MethodsEngine([("ReDo_t0_F1560_lr0.001", 9000), ("R_lr0.0001", 9000)], ctx.D32, ctx.dev, graph=False)
    e5.redo[0] = (0.0, 7)
    pend = e5.train_task(ctx.batches(e5.slots, 1), n_steps=30)
    times = [q["update"] for q in e5.redo_log]
    e6 = M.MethodsEngine([("ReDo_t0_F1560_lr0.001", 9000), ("R_lr0.0001", 9000)], ctx.D32, ctx.dev, graph=False)
    e6.redo[0] = (0.0, 780)
    pend6 = e6.train_task(ctx.batches(e6.slots, 1))
    timing_ok = times == [7, 14, 21, 28] and pend == [] and pend6 == [0] and e6.redo_log == []

    # mutants
    def mutated(fn):
        e7 = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=False)
        with torch.no_grad():
            for q, s in zip(e7.P, before["P"]):
                q.copy_(s)
            for q, s in zip(e7.m, before["m"]):
                q.copy_(s)
            for q, s in zip(e7.v, before["v"]):
                q.copy_(s)
        fn(e7)
        return not (all(torch.equal(a.detach(), b) for a, b in zip(e7.P, ref["P"]))
                    and all(torch.equal(a, b) for a, b in zip(e7.m, ref["m"]))
                    and all(torch.equal(a, b) for a, b in zip(e7.v, ref["v"])))

    def no_outgoing(e7):
        orig_P = [p.detach().clone() for p in e7.P]
        e7.redo_apply(runs, rows)
        with torch.no_grad():                              # put the outgoing weights back
            for l in range(4):
                nxt = 2 * l + 2
                for r in runs:
                    e7.P[nxt][r].copy_(torch.where(e7.P[nxt][r] == 0, orig_P[nxt][r], e7.P[nxt][r]))

    def keep_moments(e7):
        m0 = [q.clone() for q in e7.m]
        v0 = [q.clone() for q in e7.v]
        e7.redo_apply(runs, rows)
        with torch.no_grad():
            for q, s in zip(e7.m, m0):
                q.copy_(s)
            for q, s in zip(e7.v, v0):
                q.copy_(s)

    def unnormalised(e7):
        orig = M.MethodsEngine.redo_scores

        def raw(self, rows_):
            out = E.forward(self.P, self.data.X[rows_], self.act)
            return [(a.abs().mean((0, 2, 3)).view(self.R, NK.CH) if NK.IS_CONV[l] else a.abs().mean(1))
                    for l, a in enumerate((out[1], out[3], out[5], out[7]))]
        with patch.object(M.MethodsEngine, "redo_scores", raw):
            e7.redo_apply(runs, rows)

    muts = {"outgoing_not_zeroed": mutated(no_outgoing), "adam_moments_kept": mutated(keep_moments),
            "scores_not_layer_normalised": mutated(unnormalised)}
    ok = eqP and eqm and eqv and constructed and untouched and tau_set_ok and timing_ok
    return {"params_equal_reference": eqP, "m_equal": eqm, "v_equal": eqv, "constructed_dormant_found": constructed,
            "non_redo_run_untouched": untouched, "zero_tau_set_equals_float64": tau_set_ok,
            "score_rel_diff_float32_vs_float64": rel, "timing_updates": times, "deferred_on_last_update": pend6,
            "dormant_counts": {str(r): [len(x) for x in ref["dormant"][r]] for r in runs},
            "mutants": {k: {"detected": v} for k, v in muts.items()}, "pass": ok}


def check_l2i(ctx) -> dict:
    """Two runs with identical state and data, one L2I (lam 1e-2) and one plain R: after one
    update their m differ by exactly 0.1 * 2 lam (theta - theta0), up to float32 rounding."""
    slots = [("L2I_lam0.01_lr0.001", 9000), ("R_lr0.001", 9000), ("L2I_lam0.1_lr0.001", 9001)]
    e = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=False)
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for i, p in enumerate(e.P):
            d = (torch.rand(p[0].shape, generator=gen) - 0.5).to(p) * 0.02
            p[0].add_(d)
            p[1].copy_(p[0])                              # run 1 (plain R) gets the same params
    th = [p.detach().clone() for p in e.P]
    e.train_task(ctx.batches(slots, 1), n_steps=1)
    ok, worst = True, 0.0

    def ratio_of(m0, m1, theta, theta0, lam):
        want = (1 - NK.ADAM_B1) * 2 * lam * (theta - theta0)
        diff = m0 - m1
        tol = 4 * U32 * (m0.abs() + m1.abs() + want.abs()) + 1e-30
        return float(((diff - want).abs() / tol).max())

    for i in range(10):
        rt = ratio_of(e.m[i][0].double(), e.m[i][1].double(), th[i][0].double(), e.P0[i][0].double(), 1e-2)
        worst = max(worst, rt)
        ok &= rt <= 1
    # mutants: anchor at 0 (plain L2), no factor 2, sign flipped
    det = {}
    for name, f in (("anchor_at_zero", lambda t, t0: 2 * t), ("no_factor_two", lambda t, t0: (t - t0)),
                    ("sign_flipped", lambda t, t0: -2 * (t - t0))):
        bad = False
        for i in (4, 8):
            want = (1 - NK.ADAM_B1) * 1e-2 * f(th[i][0].double(), e.P0[i][0].double())
            diff = e.m[i][0].double() - e.m[i][1].double()
            tol = 4 * U32 * (e.m[i][0].double().abs() + e.m[i][1].double().abs() + want.abs()) + 1e-30
            bad |= float(((diff - want).abs() / tol).max()) > 1
        det[name] = bad
    nol2 = M.MethodsEngine([("R_lr0.001", 9000)], ctx.D32, ctx.dev, graph=False)
    absent = not nol2.l2_any and not hasattr(nol2, "P0")
    return {"worst_ratio": worst, "no_term_without_l2i": absent,
            "mutants": {k: {"detected": v} for k, v in det.items()}, "pass": ok and absent}


def check_lr(ctx) -> dict:
    slots = [("R_lr0.0001", 9000), ("R_lr0.001", 9000), ("SNA_lr0.0003", 9001), ("SNA_lr0.0001", 9001)]
    e = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=False)
    P0 = [p.detach().clone() for p in e.P]
    e.train_task(ctx.batches(slots, 1), n_steps=1)
    worst = 0.0
    for a, b, k in ((0, 1, 10.0), (3, 2, 3.0)):
        for i in range(10):
            dA = (P0[i][a] - e.P[i].detach()[a]).double()
            dB = (P0[i][b] - e.P[i].detach()[b]).double()
            # dP = fl(p - fl(lr * u)): two roundings each, u identical (same params, data, step 1)
            tol = 4 * U32 * (dB.abs() + k * dA.abs() + P0[i][b].double().abs() + k * P0[i][a].double().abs()) + 1e-30
            worst = max(worst, float(((dB - k * dA).abs() / tol).max()))
    # mutant: the lr vector reversed (slot r gets slot R-1-r's lr)
    e2 = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=False)
    e2.lr_b = [x.flip(0) for x in e2.lr_b]
    P1 = [p.detach().clone() for p in e2.P]
    e2.train_task(ctx.batches(slots, 1), n_steps=1)
    dA = (P1[4][0] - e2.P[4].detach()[0]).double()
    dB = (P1[4][1] - e2.P[4].detach()[1]).double()
    det = float((dB - 10 * dA).abs().max()) > float((4 * U32 * (dB.abs() + 10 * dA.abs() + 1)).max())
    return {"worst_ratio": worst, "mutants": {"lr_vector_misassigned": {"detected": det}}, "pass": worst <= 1}


def all_kind_slots():
    return [(M.first_cfg(a), s) for s in SEEDS2 for a in M.ARMS12]


def check_indep(ctx) -> dict:
    slots = all_kind_slots()
    k = slots.index((M.first_cfg("RSL"), 9001))

    def go(cls=M.MethodsEngine, perturb=False, n=30):
        e = cls(slots, ctx.D32, ctx.dev, graph=False)
        if perturb:
            with torch.no_grad():
                for p in e.P:
                    p[k].add_(1e-3 * torch.randn(p[k].shape, generator=torch.Generator().manual_seed(1)).to(p))
        e.train_task(ctx.batches(slots, 1), n_steps=n)
        return NKC.state_all(e)

    base = go()
    others = [r for r in range(len(slots)) if r != k]

    def compare(st, ref):
        same = all(teq(NKC.slot_view(st, r), NKC.slot_view(ref, r)) for r in others)
        moved = not teq(NKC.slot_view(st, k)["P"], NKC.slot_view(ref, k)["P"])
        return same, moved
    sp, mp = compare(go(perturb=True), base)

    class Leak(M.MethodsEngine):
        def step(self):
            orig = E.forward

            def fwd(P, xb, act):
                out = list(orig(P, xb, act))
                out[8] = out[8] + 1e-6 * out[8].mean(0, keepdim=True)
                return tuple(out)
            with patch.object(E, "forward", fwd):
                super().step()
    sl, _ = compare(go(cls=Leak, perturb=True), go(cls=Leak))
    return {"others_bit_identical": sp, "slot_changed": mp,
            "mutants": {"cross_run_leak": {"detected": not sl}}, "pass": sp and mp}


def boundary_m(e: M.MethodsEngine) -> None:
    """The task-1 boundary exactly as M.run does it."""
    rec = e.records()
    refs, control, frz = {}, [], []
    for r, c in enumerate(e.cfgs):
        if c["arm"] in NK.SNAKE_ARMS and np.isfinite(rec["ce"][r]).all():
            refs[r] = NK.reference_loss(rec["ce"][r])["lref"]
            if c["base"] == "NEO":
                control.append(r)
        if c["base"] == "SNAfrz1":
            frz.append(r)
    e.act.set_reference(refs, control)
    e.act.freeze_runs(frz)


def check_graph(ctx) -> dict:
    slots = all_kind_slots()

    def go(graph):
        e = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=graph)
        for r in e.redo:
            e.redo[r] = (0.1, 100)                      # let ReDo fire inside the window
        e.train_task(ctx.batches(slots, 1))
        rec = e.records()
        boundary_m(e)
        e.train_task(ctx.batches(slots, 2), n_steps=100)
        return NKC.state_all(e), rec, list(e.redo_log)
    sg, rg, lg = go(True)
    se, re_, le = go(False)
    eq = teq(sg, se) and all(np.array_equal(rg[k], re_[k]) for k in rg) and lg == le
    fired = len(lg) > 0 and any(sum(v for k, v in q.items() if k.startswith("n_dormant")) > 0 for q in lg)
    return {"bit_identical": eq, "redo_fired": fired, "redo_checks": len(lg), "pass": eq and fired}


def check_resume(ctx, work: Path) -> dict:
    slots = all_kind_slots()
    a, b = work / "continuous", work / "resumed"
    for p in (a, b):
        shutil.rmtree(p, ignore_errors=True)
    M.run(slots, a, 2, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True)
    stopped = False
    try:
        M.run(slots, b, 2, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True, stop_after=1)
    except KeyboardInterrupt:
        stopped = True
    M.run(slots, b, 2, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True)
    sa = torch.load(a / "state.pt", weights_only=False)
    sb = torch.load(b / "state.pt", weights_only=False)
    st_eq = teq(sa["state"], sb["state"])
    files = ["per_task.csv", "per_block.csv", "step_excerpt.csv", "site_stats.csv", "alpha_stats.csv",
             "fresh_control.csv", "calibration.json", "redo.csv"]
    txt = {f: (a / f).read_text() == (b / f).read_text() for f in files}
    redo_rows = len(list(csv.DictReader(open(a / "redo.csv"))))
    # mutant: resume without the RSL generator states
    c = work / "resumed_nogen"
    shutil.rmtree(c, ignore_errors=True)
    try:
        M.run(slots, c, 2, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True, stop_after=1)
    except KeyboardInterrupt:
        pass
    orig = M.MethodsEngine.load_full_state

    def no_gen(self, st):
        st = dict(st)
        fresh = M.MethodsEngine(self.slots, self.data, self.device, graph=False, stage=self.stage).full_state()
        st["rsl_gen"] = fresh["rsl_gen"]
        return orig(self, st)
    with patch.object(M.MethodsEngine, "load_full_state", no_gen):
        M.run(slots, c, 2, ctx.dev, data=ctx.D32, log=quiet, allow_dirty=True)
    det = (c / "per_task.csv").read_text() != (a / "per_task.csv").read_text()
    ok = stopped and st_eq and all(txt.values())
    return {"stopped": stopped, "state_bit_identical": st_eq, "files_identical": txt, "redo_rows": redo_rows,
            "mutants": {"resume_without_rsl_generators": {"detected": det}}, "dir": str(a), "pass": ok}


def check_fresh(ctx, run_dir: Path) -> dict:
    rows = list(csv.DictReader(open(run_dir / "fresh_control.csv")))
    t = int(rows[0]["task"])
    non_rsl = [q for q in rows if q["arm"] != "RSL"]
    repro = all(float(q["fresh_gap_online"]) == 0.0 and q["fresh_test_acc"] == q["continual_test_acc"]
                for q in non_rsl) if t == 1 else None
    rsl_differs = all(float(q["fresh_gap_online"]) != 0.0 for q in rows if q["arm"] == "RSL") if t == 1 else None
    st0 = torch.load(run_dir / "checkpoints" / "t00.pt", weights_only=False)["state"]
    slots = [tuple(s) for s in st0["slots"]]
    fe = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=False)
    init_eq = teq({k: fe.full_state()[k] for k in ("P", "m", "v", "tc", "act")},
                  {k: st0[k] for k in ("P", "m", "v", "tc", "act")})
    ok = init_eq and repro is not False and rsl_differs is not False
    return {"task": t, "non_rsl_reproduce_task1": repro, "rsl_fresh_uses_own_noise": rsl_differs,
            "initial_state_equals_t00": init_eq, "pass": ok}


def act_lip(cfg, alpha, l, z, Ez, u):
    k, p = cfg["kind"], cfg["shape"]
    if k == "relu":
        return Ez.clone()
    if k in ("snake", "kkt1"):
        a = NKC.bc(alpha[l], z)
        d = 1 + torch.sin(2 * a * z)
        L = torch.clamp(d.abs() + 2 * a * Ez, max=2.0)
        return L * Ez + 4 * u * (z.abs() + 2 / a)
    if k == "elu":
        return max(1.0, p[0]) * Ez + 4 * u * (z.abs() + p[0])
    if k == "leaky":
        return Ez + 2 * u * z.abs()
    if k == "rsl":
        r = (p[2] + p[3]) / 2
        return (abs(r) + abs(1 - r) * 1.25) * Ez + 8 * u * (z.abs() * (abs(r) + abs(1 - r)) + 1)
    raise ValueError(k)


def ref_forward_kind(Pr, x, cfg, alpha):
    k, p = cfg["kind"], cfg["shape"]
    def phi(z, l):
        a = NKC.bc(alpha[l], z) if k in ("snake", "kkt1") else None
        return ref_phi(k, z, a, p)
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = Pr
    f = {}
    f["z1"] = F.conv2d(x, W1, b1, padding=NK.PAD); f["a1"] = phi(f["z1"], 0)
    f["p1"] = F.max_pool2d(f["a1"], 2, 2)
    f["z2"] = F.conv2d(f["p1"], W2, b2, padding=NK.PAD); f["a2"] = phi(f["z2"], 1)
    f["p2"] = F.max_pool2d(f["a2"], 2, 2); f["h"] = f["p2"].flatten(1)
    f["z3"] = f["h"] @ W3.T + b3; f["a3"] = phi(f["z3"], 2)
    f["z4"] = f["a3"] @ W4.T + b4; f["a4"] = phi(f["z4"], 3)
    f["z5"] = f["a4"] @ W5.T + b5
    return f


def logit_bound(Pr, x, f, cfg, alpha, u):
    A = [q.abs() for q in Pr]
    Ez1 = NKC.gam(76, u) * F.conv2d(x.abs(), A[0], A[1], padding=NK.PAD)
    Ea1 = act_lip(cfg, alpha, 0, f["z1"], Ez1, u)
    Ep1 = F.max_pool2d(Ea1, 2, 2)
    Ez2 = NKC.gam(401, u) * F.conv2d(f["p1"].abs(), A[2], A[3], padding=NK.PAD) + F.conv2d(Ep1, A[2], None, padding=NK.PAD)
    Ea2 = act_lip(cfg, alpha, 1, f["z2"], Ez2, u)
    Eh = F.max_pool2d(Ea2, 2, 2).flatten(1)
    Ez3 = NKC.gam(1025, u) * (f["h"].abs() @ A[4].T + A[5]) + Eh @ A[4].T
    Ea3 = act_lip(cfg, alpha, 2, f["z3"], Ez3, u)
    Ez4 = NKC.gam(101, u) * (f["a3"].abs() @ A[6].T + A[7]) + Ea3 @ A[6].T
    Ea4 = act_lip(cfg, alpha, 3, f["z4"], Ez4, u)
    return NKC.gam(101, u) * (f["a4"].abs() @ A[8].T + A[9]) + Ea4 @ A[8].T


def check_eval(ctx, run_dir: Path) -> dict:
    """Train / test accuracy of the last hard-task checkpoint: the run's evaluation path in float64
    = an independent float64 forward of each slot's own kind, and the logits agree within the
    derived bound; the recorded float32 accuracies are compared descriptively."""
    cks = [c for c in sorted((run_dir / "checkpoints").glob("t*.pt")) if int(c.stem[1:]) % 2 == 1]
    st = torch.load(cks[-1], map_location="cpu", weights_only=False)
    t = st["task"]
    full = NKC.cast_full_state(st["state"], torch.float64)
    slots = [tuple(x) for x in full["slots"]]
    recd = {(q["cfg"], int(q["seed"])): q for q in csv.DictReader(open(run_dir / "per_task.csv"))
            if int(q["task"]) == t}
    D64 = ctx.D64
    plans = {s: NK.task_plan(s) for s in {s for a, s in slots}}
    eng = M.MethodsEngine(slots, D64, ctx.dev, graph=False, dtype=torch.float64)
    eng.load_full_state(full)
    tr = NK.slot_rows(slots, plans, t, D64.tr_rows).to(ctx.dev)
    te = NK.slot_rows(slots, plans, t, D64.te_rows).to(ctx.dev)
    _, tr_acc = NK.unit_stats(eng.P, eng.act, D64.X, tr, D64.Y)
    _, te_acc = NK.ce_acc(eng.P, eng.act, D64.Xte, D64.Yte, te)
    ok, out, worst = True, {}, 0.0
    for r, (name, seed) in enumerate(slots):
        cfg = eng.cfgs[r]
        Pr = [p[r].detach().cpu() for p in eng.P]
        alpha = [a[r].detach().cpu() for a in eng.act.alpha_all()]
        res = {}
        for split, rows_t, Xs, Ys, got in (("train", tr[r].cpu(), D64.X, D64.Y, tr_acc[r]),
                                           ("test", te[r].cpu(), D64.Xte, D64.Yte, te_acc[r])):
            hits, hits_tie, n_tie = 0, 0, 0
            for i0 in range(0, len(rows_t), 250):
                ii = rows_t[i0:i0 + 250].to(ctx.dev)
                x, y = Xs[ii].cpu(), Ys[ii].cpu()
                f = ref_forward_kind(Pr, x, cfg, alpha)
                Ez5 = logit_bound(Pr, x, f, cfg, alpha, U64)
                with torch.no_grad():
                    lg = E.forward([p[r:r + 1] for p in eng.P], Xs[ii][None], _OneAct(eng.act, r))[8][0].cpu()
                worst = max(worst, float(((lg - f["z5"]).abs() / (2 * Ez5)).max()))
                top2 = f["z5"].topk(2, 1).values
                tie = (top2[:, 0] - top2[:, 1]) <= 4 * Ez5.amax(1)
                right = f["z5"].argmax(1) == y
                n_tie += int(tie.sum())
                hits += int((right & ~tie).sum())
                hits_tie += int((right | tie).sum())
            n = len(rows_t)
            # got = k/n from a GPU division by a python scalar (a reciprocal multiply, <= 1 ulp off):
            # compare the counts, which round(got * n) recovers exactly
            k_got = int(round(float(got) * n))
            within = hits <= k_got <= hits_tie
            rec = float(recd[(name, seed)][f"{split}_acc"])
            res[split] = {"engine_float64": float(got), "reference": hits / n, "near_ties": n_tie,
                          "recorded_float32": rec, "recorded_minus_reference_samples": round((rec - hits / n) * n)}
            ok &= within
        out[f"{name}|{seed}"] = res
    ok &= worst <= 1.0
    return {"task": t, "worst_logit_ratio": worst, "per_slot": out, "pass": ok}


class _OneAct:
    """The bundle activation restricted to one run (for a single-run forward of slot r)."""
    def __init__(self, act, r):
        self.a = M.MethodsAct([act.cfgs[r]], act.V[0].device, dtype=act.dtype)
        st = {k: [t[r:r + 1].clone() for t in vs] for k, vs in act.state().items()}
        self.a.load_state(st)

    def phi(self, z, l):
        return self.a.phi(z, l)


def all_pass(res):
    ok = True
    for r in res.values():
        if not isinstance(r, dict):
            continue
        ok &= bool(r.get("pass"))
        for m in r.get("mutants", {}).values():
            ok &= m.get("detected") is True
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = NK.setup("cuda")
    t0 = time.time()
    ctx = Ctx(dev)
    work = Path(tempfile.mkdtemp(prefix="neo_m_checks_", dir=str(out)))
    res = {}
    plan = [("S-nochange-smoke", lambda: check_nochange(ctx, work)), ("S-act", lambda: check_act(ctx)),
            ("S-rsl", lambda: check_rsl(ctx)), ("S-redo", lambda: check_redo(ctx)),
            ("S-l2i", lambda: check_l2i(ctx)), ("S-lr", lambda: check_lr(ctx)),
            ("S-indep", lambda: check_indep(ctx)), ("S-graph", lambda: check_graph(ctx)),
            ("S-resume", lambda: check_resume(ctx, work)),
            ("S-fresh", lambda: check_fresh(ctx, work / "continuous")),
            ("S-eval", lambda: check_eval(ctx, work / "continuous"))]
    if a.smoke:
        sd = Path(a.smoke)
        plan += [("smoke/S-fresh", lambda: check_fresh(ctx, sd)), ("smoke/S-eval", lambda: check_eval(ctx, sd))]
    for name, fn in plan:
        t1 = time.time()
        try:
            r = fn()
        except Exception as ex:
            r = {"pass": False, "error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()}
        r["seconds"] = round(time.time() - t1, 1)
        res[name] = r
        mut = {k: v.get("detected") for k, v in r.get("mutants", {}).items()}
        print(f"[{time.strftime('%T')}] {name:17s} pass={r.get('pass')} mutants={mut} ({r['seconds']}s)"
              + (f"  ERROR {r['error']}" if "error" in r else ""), flush=True)
    summary = {"scope": "implementation checks only; no performance verdict", "all_pass": all_pass(res),
               "checks": res, "env": NK.env_info(dev), "git": NK.git_state(),
               "sources_sha256": NK.source_hashes(),
               "spec_sha256": hashlib.sha256(M.SPEC.read_bytes()).hexdigest(), "seconds": round(time.time() - t0, 1)}
    NK.write_json(out / "checks.json", summary)
    shutil.rmtree(work, ignore_errors=True)
    print(f"all_pass={summary['all_pass']} -> {out / 'checks.json'}")


if __name__ == "__main__":
    main()

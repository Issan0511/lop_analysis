#!/usr/bin/env python3
"""Implementation checks of cnn_center_budget_1020 (spec section 8).  Not performance evidence.

    python3 src/cnn_center_budget_1020_checks.py --out results/cnn_center_budget_1020/checks

  S-prefix        batch-list hashes = cifar5p1_cnn_budget_1019's batch_hashes.json (780, 7800, seeds 2000-2009)
  S-off           no centred run in the bundle: the engine = cifar5p1_cnn_budget_1019.BudgetEngine, bit for bit
  S-reuse-smoke   the production bundle (R = 40), task 1: the 20 R / ELU runs' per-update CE and hits = the
                  previous run's archived b780/steps/t01.npz, bit for bit (and the per-update time at R = 40)
  S-center        the H' EMA = cnn_center_feat_1011.FeatDoors ("feat"), bit for bit; forward with the
                  pre-update m, EMA from the pre-update z, m kept across the task boundary
  S-proj          |b| <= kappa s exactly after every projection; = Codex's project_bias except the units
                  rounded toward zero; widths vs an independent float64 reference on fixtures; in-bound
                  biases, W, the f3 bias and the Adam moments untouched
  S-graph         CUDA graph = eager for a centred bundle (task 1 + 100 updates of task 2)
  S-pure          the diagnostics change no state
  S-resume-fresh  stop after task 1 + resume = continuous; fresh = the initial state on the task's list and
                  reproduces the continual task 1
  S-stats         positive counts, the six open-fraction classes and the bias / width statistics vs
                  independent references (counts exact, float64 sums within their rounding bounds)
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import cnn_center_budget_1020 as C
from src import cnn_center_feat_1011 as CF                       # H' (P183); not modified
from src import neokunekune_cifar5p1_cnn_1018_checks as NKC       # helpers; not modified

B19, M, NK, E = C.B19, C.M, C.NK, C.E
U64 = 2.0 ** -53
SEEDS2 = NK.SMOKE_SEEDS
quiet = lambda m: None
CODEX_COMMIT = "c4e3bbd1640e1f5b05476a8b0386fde6a53d9702"
CODEX_PATH = "src/cnn_chprime_bias_1018.py"
ARCHIVE = Path.home() / "Projects" / "obsidian-research-data" / B19.EXPERIMENT / "results" / B19.EXPERIMENT


def gam(n: int, u: float = U64) -> float:
    return n * u / (1.0 - n * u)


_INT_VIEW = {torch.float64: torch.int64, torch.float32: torch.int32, torch.float16: torch.int16}


def bteq(a, b) -> bool:
    """Bitwise equality (NaN == NaN with the same bits, -0 != +0), recursing into dicts and lists."""
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(bteq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(bteq(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
        if a.dtype in _INT_VIEW:
            return torch.equal(a.view(_INT_VIEW[a.dtype]), b.view(_INT_VIEW[b.dtype]))
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    return a == b


class Ctx:
    def __init__(self, dev):
        self.dev = dev
        self.D32 = NK.Data(dev)
        self.plans = {s: NK.task_plan(s) for s in SEEDS2}
        self.batches = {b: {s: B19.make_budget_batches(s, self.plans[s], self.D32.tr_rows, b) for s in SEEDS2}
                        for b in C.BUDGETS}

    def tb(self, slots, budget, t):
        return torch.stack([self.batches[budget][s][t - 1] for a, s in slots]).to(self.dev)


def mixed_slots(seeds=SEEDS2):
    return [(C.ARM_CFG[a], s) for a in C.ARMS for s in seeds]


def state_c(e) -> dict:
    """Everything an update writes: parameters, moments, counters, records, the activation state
    (with m) and the per-task projection accumulators."""
    return {"P": [p.detach().clone() for p in e.P], "m": [q.clone() for q in e.m], "v": [q.clone() for q in e.v],
            "tc": e.tc.clone(), "j": e.j.clone(), "bad": e.bad_step.clone(),
            "rec": {k: v.clone() for k, v in e.rec.items()}, "act": e.act.state(),
            "acc": [t.clone() for t in e._acc_tensors()]}


def state_common(st: dict) -> dict:
    """The part of state_c that the previous engine also has."""
    return {"P": st["P"], "m": st["m"], "v": st["v"], "tc": st["tc"], "j": st["j"], "bad": st["bad"],
            "rec": {k: st["rec"][k] for k in NK.REC_KEYS}, "act": {k: v for k, v in st["act"].items() if k != "hm"}}


def load_codex():
    """Codex's cnn_chprime_bias_1018 module from its local-branch commit, read with git show (the
    Codex worktree and branch are not touched; nothing is written there)."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{CODEX_COMMIT}:{CODEX_PATH}"], capture_output=True,
                         check=True).stdout
    mod = types.ModuleType("codex_cnn_chprime_bias_1018")
    mod.__file__ = str(ROOT / "src" / "_codex_cnn_chprime_bias_1018_from_git.py")      # never written
    exec(compile(src, f"{CODEX_COMMIT[:8]}:{CODEX_PATH}", "exec"), mod.__dict__)
    return mod, hashlib.sha256(src).hexdigest()


# --------------------------------------------------------------------------
# S-prefix
# --------------------------------------------------------------------------

def check_prefix(ctx) -> dict:
    res, ok = {}, True
    for b in C.BUDGETS:
        ref = json.loads((C.PREV_ROOT / f"b{b}" / "batch_hashes.json").read_text())["per_seed"]
        for s in C.SEEDS:
            got = [NK.sha(x) for x in B19.make_budget_batches(s, NK.task_plan(s), ctx.D32.tr_rows, b)]
            eq = got == ref[str(s)]
            res[f"{b}|{s}"] = eq
            ok &= eq

    # mutant: the extension drawn from the c51_batch stream right after each task's 780 batches
    def sequential(seed, plan, tr_rows, budget):
        g = B19.H.stream("c51_batch", seed)
        return torch.stack([B19.C51.batch_indices(g, torch.cat([tr_rows[c] for c in cls]), budget)
                            for hard, cls in plan])
    ref = json.loads((C.PREV_ROOT / "b7800" / "batch_hashes.json").read_text())["per_seed"]["2000"]
    mut = [NK.sha(x) for x in sequential(2000, NK.task_plan(2000), ctx.D32.tr_rows, 7800)]
    return {"per_budget_seed": res, "mutants": {"extension_from_the_shared_stream": {"detected": mut != ref}},
            "pass": ok}


# --------------------------------------------------------------------------
# S-off and S-reuse-smoke
# --------------------------------------------------------------------------

def check_off(ctx) -> dict:
    slots = [(C.ARM_CFG[a], s) for a in ("R", "ELU") for s in SEEDS2]
    bt = ctx.tb(slots, 780, 1)
    ec = C.CenterEngine(slots, ctx.D32, ctx.dev, 780, graph=True)
    eb = B19.BudgetEngine(slots, ctx.D32, ctx.dev, 780, graph=True)
    ec.train_task(bt)
    eb.train_task(bt)
    eq = bteq(state_common(state_c(ec)), state_common({**NKC.state_all(eb), "acc": []}))
    no_center = (not ec.act.any_center) and (not ec.proj_on)
    eb2 = B19.BudgetEngine(slots, ctx.D32, ctx.dev, 780, graph=True)
    bt2 = bt.clone()
    bt2[:, [10, 11]] = bt2[:, [11, 10]]
    eb2.train_task(bt2)
    det = not bteq(state_common(state_c(ec)), state_common({**NKC.state_all(eb2), "acc": []}))
    return {"bit_identical": eq, "no_centre_path": no_center, "mutants": {"two_batches_swapped": {"detected": det}},
            "pass": eq and no_center}


def check_reuse_smoke(ctx) -> dict:
    """The production bundle, task 1 at budget 780, against the archived per-update records."""
    path = ARCHIVE / "b780" / "steps" / "t01.npz"
    man = json.loads((C.PREV_ROOT / "backup_manifest.json").read_text())
    ent = [m for m in man if m["backup"] == str(path)]
    sha_ok = len(ent) == 1 and hashlib.sha256(path.read_bytes()).hexdigest() == ent[0]["sha256"]
    ref = np.load(path)
    prev_slots = [(c, s) for c in B19.ARMS4 for s in C.SEEDS]
    slots = C.main_slots()
    seeds = sorted(C.SEEDS)
    prep = B19.prepare(seeds, ctx.D32, 780)
    bt = torch.stack([prep["batches"][s][0] for a, s in slots]).to(ctx.dev)
    e = C.CenterEngine(slots, ctx.D32, ctx.dev, 780, graph=True)
    torch.cuda.synchronize()
    t0 = time.time()
    e.train_task(bt)
    torch.cuda.synchronize()
    dt = time.time() - t0
    rec = e.records()
    per, ok = {}, sha_ok
    for r, (name, seed) in enumerate(slots):
        if e.cfgs[r]["center"]:
            continue
        rp = prev_slots.index((name, seed))
        eq = bool(np.array_equal(rec["ce"][r].astype(np.float32), ref["ce"][rp])
                  and np.array_equal(rec["hit"][r].astype(np.float32), ref["hit"][rp]))
        per[f"{name}|{seed}"] = eq
        ok &= eq
    viol = float(e.viol.sum())
    # mutant: the baseline runs also get the H' EMA (and therefore centring and B)
    em = C.CenterEngine(slots, ctx.D32, ctx.dev, 780, graph=True)
    em.act.center.fill_(True)
    em.train_task(bt, n_steps=50)
    rm = em.records()
    det = any(not np.array_equal(rm["ce"][r][:50].astype(np.float32), ref["ce"][prev_slots.index((n, s))][:50])
              for r, (n, s) in enumerate(slots) if not e.cfgs[r]["center"])
    return {"archive": str(path), "archive_sha256_ok": sha_ok, "per_run": per, "n_compared": len(per),
            "viol_total": viol, "seconds_780_updates_R40": dt, "ms_per_update_R40": 1000 * dt / 780,
            "mutants": {"baseline_runs_centred": {"detected": det}}, "pass": ok and len(per) == 20 and viol == 0}


# --------------------------------------------------------------------------
# S-center
# --------------------------------------------------------------------------

def check_center(ctx) -> dict:
    dev = ctx.dev
    slots = [(C.ARM_CFG["RCB"], s) for s in SEEDS2]
    e = C.CenterEngine(slots, ctx.D32, dev, 780, graph=False)
    R, B = e.R, NK.BATCH
    g = torch.Generator(device="cpu").manual_seed(1020)
    zs_seq = []
    for _ in range(5):
        zs_seq.append([(torch.randn((B, R * NK.CH, 32, 32), generator=g) * 2 + 0.3).to(dev),
                       (torch.randn((B, R * NK.CH, 16, 16), generator=g) * 2 - 0.2).to(dev),
                       (torch.randn((R, B, 100), generator=g) * 2 + 0.1).to(dev),
                       (torch.randn((R, B, 100), generator=g) * 2).to(dev)])
    act = e.act
    ref = CF.FeatDoors(R, "feat", dev)
    for zs in zs_seq:
        act.update(zs)
        ref.update(zs)
    form_eq = all(torch.equal(act.hm[l], ref.m[l]) for l in range(4))
    shapes = [tuple(t.shape) for t in act.hm] == [tuple(t.shape) for t in ref.m]

    # mutant: positions averaged too (per channel, broadcast to the positions)
    class ChanAct(C.CenterAct):
        def out_mean(self, z, l):
            m = super().out_mean(z, l)
            if NK.IS_CONV[l]:
                s = C.SIDE[l] ** 2
                return m.view(self.R, NK.CH, s).mean(2, keepdim=True).expand(-1, -1, s).reshape(self.R, -1)
            return m
    mchan = ChanAct(e.cfgs, dev)
    for zs in zs_seq:
        mchan.update(zs)
    det_chan = not all(torch.equal(mchan.hm[l], ref.m[l]) for l in range(4))

    # update order inside the engine's step: forward with the pre-update m, EMA from the pre-update z
    e2 = C.CenterEngine(slots, ctx.D32, dev, 780, graph=False)
    bt = ctx.tb(slots, 780, 1)
    e2.train_task(bt, n_steps=30)
    P_old = [p.detach().clone() for p in e2.P]
    hm_old = [h.clone() for h in e2.act.hm]
    j = int(e2.j)
    xb = ctx.D32.X[e2.idx[:, j]]
    yb = ctx.D32.Y[e2.idx[:, j]]
    with torch.no_grad():
        out_pre = C.forward_c(P_old, xb, e2.act)
        ce_pre = F.cross_entropy(out_pre[8].reshape(-1, NK.N_CLASSES), yb.reshape(-1),
                                 reduction="none").view(e2.R, -1).mean(1)
        exp_hm = [h.mul(1 - C.BETA_H).add(e2.act.out_mean(z, l), alpha=C.BETA_H)
                  for l, (h, z) in enumerate(zip(hm_old, out_pre[0:8:2]))]
    e2.step()
    hm_new = [h.clone() for h in e2.act.hm]
    ce_rec = e2.rec["ce"][:, j]
    order_ok = all(torch.equal(a, b) for a, b in zip(hm_new, exp_hm)) and torch.equal(ce_rec, ce_pre)
    with torch.no_grad():
        # mutant 1: EMA from the post-update forward
        e3 = C.CenterEngine(slots, ctx.D32, dev, 780, graph=False)
        e3.load_full_state(e2.full_state())
        for h, h0 in zip(e3.act.hm, hm_old):
            h.copy_(h0)
        P_new_pre_proj = [p.detach().clone() for p in e2.P]          # (biases already projected)
        out_post = C.forward_c(P_new_pre_proj, xb, e3.act)
        mut_post = [h.mul(1 - C.BETA_H).add(e3.act.out_mean(z, l), alpha=C.BETA_H)
                    for l, (h, z) in enumerate(zip(hm_old, out_post[0:8:2]))]
        det_post = not all(torch.equal(a, b) for a, b in zip(hm_new, mut_post))
        # mutant 2: the training forward reads the post-update m
        e3.act.hm = [h.clone() for h in hm_new]
        ce_mut = F.cross_entropy(C.forward_c(P_old, xb, e3.act)[8].reshape(-1, NK.N_CLASSES), yb.reshape(-1),
                                 reduction="none").view(e2.R, -1).mean(1)
        det_new_m = not torch.equal(ce_rec, ce_mut)
    # m kept across the task boundary (train_task does not reset it)
    e2.train_task(bt, n_steps=780 - 31)
    keep = [h.clone() for h in e2.act.hm]
    e2.train_task(ctx.tb(slots, 780, 2), n_steps=0)
    kept = all(torch.equal(a, b) for a, b in zip(keep, e2.act.hm))
    nonzero = all(float(h.abs().sum()) > 0 for h in keep)

    class ResetEngine(C.CenterEngine):
        def train_task(self, batches, n_steps=None, hook=None, hook_at=()):
            for h in self.act.hm:
                h.zero_()
            return super().train_task(batches, n_steps, hook, hook_at)
    e4 = ResetEngine(slots, ctx.D32, dev, 780, graph=False)
    e4.load_full_state(e2.full_state())
    e4.train_task(ctx.tb(slots, 780, 2), n_steps=0)
    det_reset = not all(torch.equal(a, b) for a, b in zip(keep, e4.act.hm))
    # uncentred runs of a mixed bundle keep m = 0
    em = C.CenterEngine(mixed_slots(), ctx.D32, dev, 780, graph=False)
    em.train_task(ctx.tb(mixed_slots(), 780, 1), n_steps=50)
    unc = [r for r, c in enumerate(em.cfgs) if not c["center"]]
    zero_unc = all(float(h[unc].abs().sum()) == 0 for h in em.act.hm)
    ok = form_eq and shapes and order_ok and kept and nonzero and zero_unc
    return {"ema_equals_FeatDoors_feat": form_eq, "shapes": shapes, "beta_h": C.BETA_H,
            "order_forward_pre_m_ema_pre_z": order_ok, "kept_across_boundary": kept, "uncentred_m_zero": zero_unc,
            "mutants": {"positions_averaged": {"detected": det_chan}, "ema_from_post_update_z": {"detected": det_post},
                        "forward_reads_post_update_m": {"detected": det_new_m},
                        "reset_at_task_boundary": {"detected": det_reset}},
            "pass": ok}


# --------------------------------------------------------------------------
# S-proj
# --------------------------------------------------------------------------

def ref_width(q: np.ndarray, conv: bool) -> np.ndarray:
    """Independent float64 two-pass width of one run: q (B, C, P) conv / (B, n) fc."""
    q = q.astype(np.float64)
    mu = q.mean(0, keepdims=True)
    var = ((q - mu) ** 2).mean(0)
    return np.sqrt(var.mean(-1)) if conv else np.sqrt(var)


def check_proj(ctx) -> dict:
    dev = ctx.dev
    res, ok = {}, True
    # (a) width fixtures: 100x different channels, a channel whose positions differ only in their mean
    g = torch.Generator(device="cpu").manual_seed(7)
    B, P = 32, 16
    q = torch.randn((B, 1 * NK.CH, 4, 4), generator=g)
    q[:, 1] *= 0.01                                                     # 100x narrower channel
    q[:, 2] = torch.arange(P, dtype=torch.float32).view(4, 4) * 0.5    # position means only: width 0
    q[:, 3] = q[:, 3] * 1e-3 + 50.0                                    # large mean, small width
    s = C.image_width(q.to(dev), True, 1).cpu().numpy()[0]
    sr = ref_width(q.reshape(B, NK.CH, P).numpy(), True)
    # float64 bound: var of n = 32 two-pass sums, then the mean over 16 positions
    qq = q.reshape(B, NK.CH, P).double().numpy()
    vb = 4 * gam(B + P + 8) * (((qq - qq.mean(0)) ** 2).mean(0).mean(-1) + (qq.mean(0) ** 2).mean(-1))
    tol = np.where(s + sr > 0, vb / np.maximum(s + sr, 1e-300), 0.0)
    width_ok = bool((np.abs(s - sr) <= tol).all()) and s[2] == 0.0
    fc = torch.randn((1, B, 100), generator=g) * torch.logspace(-3, 1, 100)[None, None, :]
    sf = C.image_width(fc.to(dev), False, 1).cpu().numpy()[0]
    sfr = ref_width(fc[0].numpy(), False)
    fx = fc[0].double().numpy()
    vbf = 4 * gam(B + 8) * (((fx - fx.mean(0)) ** 2).mean(0) + fx.mean(0) ** 2)
    width_ok &= bool((np.abs(sf - sfr) <= vbf / np.maximum(sf + sfr, 1e-300)).all())
    mut_total = np.sqrt(((qq - qq.mean((0, 2), keepdims=True)) ** 2).mean((0, 2)))      # images x positions
    det_total = bool((np.abs(mut_total - sr) > tol).any())
    mut_median = np.full_like(sr, np.median(sr))
    det_median = bool((np.abs(mut_median - sr) > tol).any())
    res["width_fixture"] = {"ok": bool(width_ok), "width": s.tolist(), "ref": sr.tolist(), "zero_width_channel": float(s[2])}
    ok &= width_ok
    # s = 0 -> b = 0; in-bound bits untouched
    b = torch.tensor([0.3, -1e-4, 0.7, 1e-6] + [0.0] * (NK.CH - 4))
    cap = 0.2 * torch.from_numpy(np.concatenate([sr[:4], sr[4:]]))
    bc = torch.where(b.double().abs() > cap, C.clip_toward_zero(b, cap), b)
    z_ok = float(bc[2]) == 0.0 and bool((bc.double().abs() <= cap).all()) and \
        all(bool(bc[i] == b[i]) for i in range(NK.CH) if not bool(b[i].double().abs() > cap[i]))
    res["zero_width_and_in_bound"] = z_ok
    ok &= z_ok

    # (b) against Codex's project_bias on real states (ReLU runs), plain and with 10x biases.
    # Codex rounds the float64 clip to nearest; with that rounding patched in, the engine's projection
    # must equal Codex's bit for bit in all four layers.  The registered rounding (toward zero) may
    # differ from nearest only by one ulp toward zero where the nearest value exceeds the cap (checked
    # on c1, which has no upstream correction).
    codex, codex_sha = load_codex()
    nearest = lambda b, cap: b.double().clamp(-cap, cap).to(b.dtype)
    slots = [(C.ARM_CFG["RCB"], s) for s in SEEDS2]
    e = C.CenterEngine(slots, ctx.D32, dev, 780, graph=False)
    bt = ctx.tb(slots, 780, 1)
    e.train_task(bt, n_steps=40)
    cmp = {}
    for tag, scale in (("plain", 1.0), ("bias_x10", 10.0)):
        st = e.full_state()
        e1 = C.CenterEngine(slots, ctx.D32, dev, 780, graph=False)
        e1.load_full_state(st)
        en = C.CenterEngine(slots, ctx.D32, dev, 780, graph=False)
        en.load_full_state(st)
        with torch.no_grad():
            for x in (e1, en):
                for l in range(4):
                    x.P[2 * l + 1].mul_(scale)
        xb = ctx.D32.X[bt[:, 45]]
        P_c = [p.detach().clone() for p in e1.P]
        before = state_c(e1)
        fin = e1.project(xb)
        after = state_c(e1)
        with patch.object(C, "clip_toward_zero", nearest):
            en.project(xb)
        door = CF.FeatDoors(e1.R, "feat", dev)
        for l in range(4):
            door.m[l].copy_(e1.act.hm[l])
        trace = codex.project_bias(P_c, xb, door, kappa=C.KAPPA)
        eq_codex = all(torch.equal(en.P[2 * l + 1], P_c[2 * l + 1]) for l in range(4))
        cap0 = C.KAPPA * trace[0]["s"]
        mine, near = e1.P[1].detach(), en.P[1].detach()
        diff = mine != near
        explained = bool((~diff | ((near.double().abs() > cap0) & (mine == torch.nextafter(near, torch.zeros_like(near))))).all())
        within = bool((mine.double().abs() <= cap0).all())
        n_toward0 = [int((e1.P[2 * l + 1] != en.P[2 * l + 1]).sum()) for l in range(4)]
        untouched = bteq([after["P"][i] for i in (0, 2, 4, 6, 8, 9)], [before["P"][i] for i in (0, 2, 4, 6, 8, 9)]) \
            and bteq(after["m"], before["m"]) and bteq(after["v"], before["v"]) and bteq(after["tc"], before["tc"])
        inb = all(bool((before["P"][2 * l + 1][e1.fire_cnt[l] == 0] == after["P"][2 * l + 1][e1.fire_cnt[l] == 0]).all())
                  for l in range(4))
        fired = [int(e1.fire_cnt[l].sum()) for l in range(4)]
        viol1 = float(e1.viol.sum())
        cmp[tag] = {"nearest_equals_codex_all_layers": eq_codex, "c1_difference_is_one_ulp_toward_zero": explained,
                    "c1_within_cap": within, "units_rounded_toward_zero_per_layer": n_toward0,
                    "fired_units_per_layer": fired, "viol": viol1, "W_f3bias_moments_untouched": untouched,
                    "unfired_bias_untouched": inb, "finite": bool(fin.all())}
        ok &= eq_codex and explained and within and untouched and inb and viol1 == 0 and bool(fin.all()) \
            and (sum(fired) > 0 or scale == 1.0)
    res["vs_codex_project_bias"] = {"codex_commit": CODEX_COMMIT, "codex_path": CODEX_PATH,
                                    "codex_source_sha256": codex_sha, **cmp}

    # (c) sequential order.  Mutant: the projection loop of the engine, except that the next layer is
    # fed with the uncorrected bias (every width measured before any correction)
    def all_first(eng, xb):
        act, R = eng.act, eng.R
        Bn = xb.shape[1]
        with torch.no_grad():
            value = xb.transpose(0, 1).reshape(Bn, R * 3, 32, 32)
            for l in range(4):
                W, b = eng.P[2 * l], eng.P[2 * l + 1]
                conv = NK.IS_CONV[l]
                if conv:
                    qq_ = F.conv2d(value, W.reshape(R * NK.CH, -1, NK.KER, NK.KER), None, padding=NK.PAD, groups=R)
                else:
                    qq_ = torch.bmm(value, W.transpose(1, 2))
                cap_ = C.KAPPA * C.image_width(qq_, conv, R)
                b0 = b.detach().clone()
                b.copy_(torch.where((b.double().abs() > cap_) & act.center, C.clip_toward_zero(b.detach(), cap_), b.detach()))
                if conv:
                    z = qq_ + b0.reshape(1, R * NK.CH, 1, 1)
                    value = F.max_pool2d(act.phi(z, l), NK.POOL, NK.POOL) - act.sub_map(l)
                    if l == 1:
                        value = value.reshape(Bn, R, NK.FLAT).transpose(0, 1)
                else:
                    value = act.phi(qq_ + b0[:, None, :], l) - act.hm[l][:, None, :]
    st = e.full_state()
    ea = C.CenterEngine(slots, ctx.D32, dev, 780, graph=False); ea.load_full_state(st)
    eb_ = C.CenterEngine(slots, ctx.D32, dev, 780, graph=False); eb_.load_full_state(st)
    with torch.no_grad():
        for en in (ea, eb_):
            for l in range(4):
                en.P[2 * l + 1].mul_(10.0)
    xb = ctx.D32.X[bt[:, 45]]
    ea.project(xb)
    all_first(eb_, xb)
    det_order = not all(torch.equal(ea.P[i], eb_.P[i]) for i in (1, 3, 5, 7))
    same_c1 = torch.equal(ea.P[1], eb_.P[1])          # c1 has no upstream correction: must agree

    # (d) the in-run bound counter over 300 updates of a mixed bundle; a mutant clipping at 1.5 kappa s
    em = C.CenterEngine(mixed_slots(), ctx.D32, dev, 780, graph=True)
    em.train_task(ctx.tb(mixed_slots(), 780, 1), n_steps=300)
    viol = float(em.viol.sum())
    fired = float(sum(float(t.sum()) for t in em.fire_cnt))
    unc = [r for r, c in enumerate(em.cfgs) if not c["center"]]
    no_unc_fire = all(float(t[unc].sum()) == 0 for t in em.fire_cnt)

    orig_clip = C.clip_toward_zero
    loose = lambda b, cap: orig_clip(b, 1.5 * cap)     # mutant: clips at 1.5 kappa s; the counter checks kappa s
    with patch.object(C, "clip_toward_zero", loose):
        el = C.CenterEngine(mixed_slots(), ctx.D32, dev, 780, graph=False)
        el.train_task(ctx.tb(mixed_slots(), 780, 1), n_steps=100)
    det_loose = float(el.viol.sum()) > 0
    res["in_run"] = {"viol_total_300_updates": viol, "fired_unit_updates": fired, "uncentred_never_fire": no_unc_fire,
                     "order_mutant_c1_agrees": same_c1}
    ok &= viol == 0 and fired > 0 and no_unc_fire and same_c1
    res["mutants"] = {"total_images_x_positions_variance": {"detected": det_total},
                      "layer_median_width": {"detected": det_median},
                      "widths_before_any_correction": {"detected": det_order},
                      "clip_at_1.5_kappa_vs_counter": {"detected": det_loose}}
    res["pass"] = ok
    return res


# --------------------------------------------------------------------------
# S-graph, S-pure
# --------------------------------------------------------------------------

def check_graph(ctx) -> dict:
    slots = mixed_slots()

    def go(graph):
        e = C.CenterEngine(slots, ctx.D32, ctx.dev, 780, graph=graph)
        e.train_task(ctx.tb(slots, 780, 1))
        s1 = state_c(e)
        e.train_task(ctx.tb(slots, 780, 2), n_steps=100)
        return s1, state_c(e)
    a, b = go(True), go(False)
    eq = bteq(a[0], b[0]) and bteq(a[1], b[1])
    return {"bit_identical": eq, "pass": eq}


def check_pure(ctx) -> dict:
    slots = mixed_slots()
    e = C.CenterEngine(slots, ctx.D32, ctx.dev, 780, graph=False)
    e.train_task(ctx.tb(slots, 780, 1), n_steps=60)
    before = state_c(e)
    probe = NK.probe_ids(ctx.D32.tr_rows).to(ctx.dev)
    pr = probe[None].expand(e.R, -1)
    plans = {s: NK.task_plan(s) for s in SEEDS2}
    tr = NK.slot_rows(slots, plans, 1, ctx.D32.tr_rows).to(ctx.dev)
    fwd = E.forward
    C.unit_stats_c(e.P, e.act, ctx.D32.X, pr)
    C.unit_stats_c(e.P, e.act, ctx.D32.X, tr, ctx.D32.Y)
    C.bias_stats(e.P, e.act, ctx.D32.X, pr)
    C.lag_eval(e.P, e.act, ctx.D32.X, tr)
    C.ce_acc_c(e.P, e.act, ctx.D32.X, ctx.D32.Y, tr)
    NK.control_arrays(e.act)
    restored = E.forward is fwd
    ok = bteq(state_c(e), before) and restored
    e2 = C.CenterEngine(slots, ctx.D32, ctx.dev, 780, graph=False)
    e2.load_full_state(e.full_state())
    b2 = state_c(e2)

    def leaky(P, act, X, rows, chunk=50):
        """Mutant: a diagnostic that also advances the EMA on its first chunk."""
        with torch.no_grad():
            out = C.forward_c(P, X[rows[:, :chunk]], act)
            act.update([out[0], out[2], out[4], out[6]])
        return C.lag_eval(P, act, X, rows, chunk)
    leaky(e2.P, e2.act, ctx.D32.X, tr)
    det = not bteq(state_c(e2)["act"], b2["act"])
    return {"state_unchanged": ok, "forward_restored": restored,
            "mutants": {"diagnostic_advances_ema": {"detected": det}}, "pass": ok}


# --------------------------------------------------------------------------
# S-resume and S-fresh
# --------------------------------------------------------------------------

def check_resume_fresh(ctx, work: Path) -> dict:
    slots = mixed_slots()
    a, b = work / "continuous", work / "resumed"
    for p in (a, b):
        shutil.rmtree(p, ignore_errors=True)
    kw = dict(data=ctx.D32, log=quiet, allow_dirty=True, fresh_tasks=(1,), ckpt_boundaries=(0, 1, 2))
    C.run(slots, a, 2, 780, ctx.dev, **kw)
    stopped = False
    try:
        C.run(slots, b, 2, 780, ctx.dev, stop_after=1, **kw)
    except KeyboardInterrupt:
        stopped = True
    C.run(slots, b, 2, 780, ctx.dev, **kw)
    sa = torch.load(a / "state.pt", weights_only=False)
    sb = torch.load(b / "state.pt", weights_only=False)
    files = ["per_task.csv", "probe_stats.csv", "train_stats.csv", "fresh_control.csv", "lag.csv", "batch_hashes.json"]
    txt = {f: (a / f).read_text() == (b / f).read_text() for f in files}
    resume_ok = stopped and bteq(sa["state"], sb["state"]) and all(txt.values())
    rows = list(csv.DictReader(open(a / "fresh_control.csv")))
    hashes = json.loads((a / "batch_hashes.json").read_text())["per_seed"]
    fresh_hash = all(q["batch_sha256"] == hashes[q["seed"]][0] for q in rows)
    repro = all(float(q["gap_online_acc"]) == 0.0 and float(q["gap_test_acc"]) == 0.0 for q in rows)
    st0 = torch.load(a / "checkpoints" / "b00.pt", weights_only=False)["state"]
    fe = C.CenterEngine(slots, ctx.D32, ctx.dev, 780, graph=False, stage="fresh")
    init_eq = bteq({k: fe.full_state()[k] for k in ("P", "m", "v", "tc", "act")},
                      {k: st0[k] for k in ("P", "m", "v", "tc", "act")})
    hm0 = all(float(t.abs().sum()) == 0 for t in st0["act"]["hm"])
    # mutant: resume that drops m
    d = work / "resumed_no_m"
    shutil.rmtree(d, ignore_errors=True)
    try:
        C.run(slots, d, 2, 780, ctx.dev, stop_after=1, **kw)
    except KeyboardInterrupt:
        pass
    orig = C.CenterEngine.load_full_state

    def no_m(self, st):
        st = copy.deepcopy(st)
        st["act"]["hm"] = [torch.zeros_like(t) for t in st["act"]["hm"]]
        return orig(self, st)
    with patch.object(C.CenterEngine, "load_full_state", no_m):
        C.run(slots, d, 2, 780, ctx.dev, **kw)
    det = (d / "per_task.csv").read_text() != (a / "per_task.csv").read_text()
    return {"resume": {"stopped": stopped, "files_identical": txt, "ok": resume_ok},
            "fresh": {"batch_hash_is_the_task": fresh_hash, "reproduces_continual_task1": repro,
                      "initial_state_equals_b00": init_eq, "initial_m_zero": hm0},
            "mutants": {"resume_without_m": {"detected": det}},
            "pass": resume_ok and fresh_hash and repro and init_eq and hm0}


# --------------------------------------------------------------------------
# S-stats
# --------------------------------------------------------------------------

def check_stats(ctx) -> dict:
    slots = mixed_slots()
    e = C.CenterEngine(slots, ctx.D32, ctx.dev, 780, graph=False)
    e.train_task(ctx.tb(slots, 780, 1), n_steps=300)
    probe = NK.probe_ids(ctx.D32.tr_rows)[:250].to(ctx.dev)
    pr = probe[None].expand(e.R, -1)
    chunk = 50
    st, _ = C.unit_stats_c(e.P, e.act, ctx.D32.X, pr, chunk=chunk)
    bs = C.bias_stats(e.P, e.act, ctx.D32.X, pr, chunk=chunk)
    # independent: counts of z > 0 and the whole q at once (same chunked forward, own aggregation)
    cnt = [torch.zeros(e.R, w, dtype=torch.int64, device=ctx.dev) for w in NK.WIDTHS]
    qs = [[] for _ in range(4)]
    first_cnt = None
    with torch.no_grad():
        for i0 in range(0, pr.shape[1], chunk):
            idx = pr[:, i0:i0 + chunk]
            out, ins = C.forward_c(e.P, ctx.D32.X[idx], e.act, with_inputs=True)
            for l in range(4):
                z = out[2 * l]
                if NK.IS_CONV[l]:
                    k = (z > 0).view(z.shape[0], e.R, NK.CH, -1).sum((0, 3))
                    q = F.conv2d(ins[l], e.P[2 * l].reshape(e.R * NK.CH, -1, NK.KER, NK.KER), None,
                                 padding=NK.PAD, groups=e.R)
                    qs[l].append(q.view(q.shape[0], e.R, NK.CH, -1).permute(1, 0, 2, 3).double().cpu())
                else:
                    k = (z > 0).sum(1)
                    qs[l].append(torch.bmm(ins[l], e.P[2 * l].transpose(1, 2)).double().cpu())
                cnt[l] += k
            if first_cnt is None:
                first_cnt = [c.clone() for c in cnt]
    counts_ok = all(np.array_equal(st[t]["npos"], cnt[l].cpu().numpy()) for l, t in enumerate(NK.SITES))
    det_first = not all(np.array_equal(st[t]["npos"], first_cnt[l].cpu().numpy()) for l, t in enumerate(NK.SITES))
    # the six classes at exact boundaries, against Fractions
    n = 1000
    ks = [0, 1, 49, 50, 51, 299, 300, 301, 699, 700, 701, 949, 950, 951, 1000]

    def frac_class(k):
        p = Fraction(k, n)
        return ("p0" if p == 0 else "lt05" if p < Fraction(1, 20) else "sparse" if p < Fraction(3, 10) else
                "mid" if p <= Fraction(7, 10) else "hi" if p < Fraction(19, 20) else "ge95")
    cls = C.ppos_classes(np.array(ks), n)
    cls_ok = all(sum(bool(cls[c][i]) for c in C.PPOS_BINS) == 1 and bool(cls[frac_class(k)][i])
                 for i, k in enumerate(ks))
    # bias / width statistics vs the whole-array float64 two-pass reference, within derived bounds
    worst, bs_ok, det_mix = {}, True, False
    for l, tag in enumerate(NK.SITES):
        q = torch.cat(qs[l], dim=1)                                   # (R, N, C, P) / (R, N, n)
        N = q.shape[1]
        mu = q.mean(1)
        var = ((q - mu[:, None]) ** 2).mean(1)
        absx = (q - q[:, :chunk].mean(1, keepdim=True)).abs()
        if NK.IS_CONV[l]:
            Pn = q.shape[-1]
            a_ref, sq_ref = mu.mean(-1), var.mean(-1).sqrt()
            vb = 4 * gam(N + Pn + 8) * (((q - q[:, :chunk].mean(1, keepdim=True)) ** 2).mean(1).mean(-1)
                                        + (absx.mean(1) * absx.mean(1)).mean(-1))
            ab = 4 * gam(N + Pn + 8) * (q.abs().mean(1) + absx.mean(1)).mean(-1)
            mix = (var.mean(-1) + mu.var(-1, correction=0)).sqrt()
        else:
            a_ref, sq_ref = mu, var.sqrt()
            vb = 4 * gam(N + 8) * (((q - q[:, :chunk].mean(1, keepdim=True)) ** 2).mean(1) + absx.mean(1) ** 2)
            ab = 4 * gam(N + 8) * (q.abs().mean(1) + absx.mean(1))
            mix = sq_ref
        sq = torch.from_numpy(bs[tag]["s_q"])
        a = torch.from_numpy(bs[tag]["a"])
        tol_s = torch.where(sq + sq_ref > 0, vb / (sq + sq_ref).clamp_min(1e-300), torch.zeros_like(vb))
        r_s = float(((sq - sq_ref).abs() / tol_s.clamp_min(1e-300)).max())
        r_a = float(((a - a_ref).abs() / ab.clamp_min(1e-300)).max())
        worst[tag] = {"s_q": r_s, "a": r_a}
        bs_ok &= r_s <= 1 and r_a <= 1
        b = e.P[2 * l + 1].detach().double().cpu()
        bos = torch.from_numpy(bs[tag]["b_over_s"])
        bs_ok &= bool(torch.equal(torch.isnan(bos), ~(sq > 0)))
        if NK.IS_CONV[l]:
            det_mix |= bool(((mix - sq_ref).abs() > tol_s).any())
    return {"counts_exact": counts_ok, "classes_vs_fractions": cls_ok, "bias_stats_worst_ratio_to_bound": worst,
            "mutants": {"first_chunk_only": {"detected": det_first},
                        "images_x_positions_sd_as_width": {"detected": det_mix}},
            "pass": counts_ok and cls_ok and bs_ok}


# --------------------------------------------------------------------------

def all_pass(res):
    ok = True
    for r in res.values():
        ok &= bool(r.get("pass"))
        for m in r.get("mutants", {}).values():
            ok &= m.get("detected") is True
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated check names (development only)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = NK.setup("cuda")
    t0 = time.time()
    ctx = Ctx(dev)
    work = Path(tempfile.mkdtemp(prefix="center_checks_", dir=str(out)))
    plan = [("S-prefix", lambda: check_prefix(ctx)), ("S-off", lambda: check_off(ctx)),
            ("S-reuse-smoke", lambda: check_reuse_smoke(ctx)), ("S-center", lambda: check_center(ctx)),
            ("S-proj", lambda: check_proj(ctx)), ("S-graph", lambda: check_graph(ctx)),
            ("S-pure", lambda: check_pure(ctx)), ("S-resume-fresh", lambda: check_resume_fresh(ctx, work)),
            ("S-stats", lambda: check_stats(ctx))]
    if a.only:
        keep = set(a.only.split(","))
        plan = [p for p in plan if p[0] in keep]
    res = {}
    for name, fn in plan:
        t1 = time.time()
        try:
            r = fn()
        except Exception as ex:
            r = {"pass": False, "error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()}
        r["seconds"] = round(time.time() - t1, 1)
        res[name] = r
        mut = {k: v.get("detected") for k, v in r.get("mutants", {}).items()}
        print(f"[{time.strftime('%T')}] {name:15s} pass={r.get('pass')} mutants={mut} ({r['seconds']}s)"
              + (f"  ERROR {r['error']}" if "error" in r else ""), flush=True)
    summary = {"scope": "implementation checks only; no performance verdict", "all_pass": all_pass(res),
               "complete": a.only is None, "checks": res, "env": NK.env_info(dev), "git": NK.git_state(),
               "sources_sha256": NK.source_hashes(),
               "spec_sha256": hashlib.sha256(C.SPEC.read_bytes()).hexdigest(), "seconds": round(time.time() - t0, 1)}
    NK.write_json(out / ("checks.json" if a.only is None else "checks_partial.json"), summary)
    shutil.rmtree(work, ignore_errors=True)
    print(f"all_pass={summary['all_pass']} complete={summary['complete']} -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Implementation checks of cifar5p1_cnn_budget_1019 (spec §7).  Not performance evidence.

    python3 src/cifar5p1_cnn_budget_1019_checks.py --out results/cifar5p1_cnn_budget_1019/checks

  S-prefix      780 lists = the previous lists; 780 prefix of 2340 prefix of 7800; the first 780
                batches of every task do not depend on the budget; rows inside the task; the
                extension is a concatenation of permutations
  S-engine-780  the budget engine at 780 = the 12-arm engine, bit for bit
  S-stats       ppos / dmax / amax / dead / zeroout / eff_rank vs an independent float64 reference
  S-pure        diagnostics change no state
  S-freeze      SNAfrz1 frozen after task 1 of each budget; alpha / V fixed afterwards; the frozen
                alpha differs between budgets and equals what was saved
  S-graph       CUDA graph = eager at budget 2340
  S-resume      stop after task 1 + resume = continuous (budget 2340)
  S-fresh       fresh = initial state on the task's own batch list; in a 2-task run it reproduces
                the continual task 1
"""
from __future__ import annotations

import argparse
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
from src import cifar5p1_cnn_budget_1019 as B
from src import neokunekune_methods_cifar5p1_cnn_1018_checks as MC     # previous helpers; not modified
from src import neokunekune_cifar5p1_cnn_1018_checks as NKC
from src import pmnist_0905 as H

M, NK, E, C51 = B.M, B.NK, B.E, B.C51
U64 = 2.0 ** -53
quiet = lambda m: None
SEEDS2 = NK.SMOKE_SEEDS


class Ctx:
    def __init__(self, dev):
        self.dev = dev
        self.cifar = C51.Cifar100()
        self.D32 = NK.Data(dev, self.cifar)
        self._D64 = None
        self.plans = {s: NK.task_plan(s) for s in SEEDS2}
        self.batches = {b: {s: B.make_budget_batches(s, self.plans[s], self.D32.tr_rows, b) for s in SEEDS2}
                        for b in B.BUDGETS}

    @property
    def D64(self):
        if self._D64 is None:
            self._D64 = NK.Data(self.dev, self.cifar, dtype=torch.float64)
        return self._D64

    def tb(self, slots, budget, t):
        return torch.stack([self.batches[budget][s][t - 1] for a, s in slots]).to(self.dev)


def check_prefix(ctx) -> dict:
    res, ok = {}, True
    y = ctx.cifar.train_y
    for s in list(B.SEEDS) + list(SEEDS2):
        plan = NK.task_plan(s)
        bl = {b: (ctx.batches[b][s] if s in SEEDS2 else B.make_budget_batches(s, plan, ctx.D32.tr_rows, b))
              for b in B.BUDGETS}
        prev = NK.make_batches(s, plan, ctx.D32.tr_rows)
        r = {"780_equals_previous": torch.equal(bl[780], prev),
             "780_prefix_of_2340": torch.equal(bl[2340][:, :780], bl[780]),
             "2340_prefix_of_7800": torch.equal(bl[7800][:, :2340], bl[2340]),
             "shapes": [tuple(bl[b].shape) for b in B.BUDGETS] == [(30, b, 32) for b in B.BUDGETS]}
        member, cover = True, True
        for t, (hard, cs) in enumerate(plan, 1):
            rows = torch.cat([ctx.D32.tr_rows[c] for c in cs])
            bt = bl[7800][t - 1].flatten()
            member &= bool(torch.isin(bt, rows).all()) and bool(torch.isin(y[bt], torch.tensor(cs)).all())
            ext = bl[7800][t - 1][780:].flatten()
            cnt = torch.bincount(ext, minlength=50000)[rows]
            full, rest = divmod(len(ext), len(rows))
            cover &= bool(((cnt == full) | (cnt == full + 1)).all()) and int((cnt == full + 1).sum()) == rest
        r["rows_inside_task"], r["extension_permutations"] = member, cover
        res[str(s)] = r
        ok &= all(v for v in r.values())

    # mutant: the extension drawn from the c51_batch stream right after each task's 780 batches
    def sequential(seed, plan, tr_rows, budget):
        g = H.stream("c51_batch", seed)
        out = []
        for hard, cls in plan:
            rows = torch.cat([tr_rows[c] for c in cls])
            out.append(C51.batch_indices(g, rows, budget))
        return torch.stack(out)
    s = 2000
    plan = NK.task_plan(s)
    a = sequential(s, plan, ctx.D32.tr_rows, 780)
    b = sequential(s, plan, ctx.D32.tr_rows, 2340)
    det = not torch.equal(a[1:, :780], b[1:, :780])           # later tasks shift with the budget
    return {"per_seed": res, "mutants": {"extension_from_the_shared_stream": {"detected": det}}, "pass": ok}


def check_engine780(ctx) -> dict:
    slots = B.smoke_slots()
    bt = ctx.tb(slots, 780, 1)
    eb = B.BudgetEngine(slots, ctx.D32, ctx.dev, 780, graph=True)
    em = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=True)
    eb.train_task(bt)
    em.train_task(bt)
    eq = NKC.teq(NKC.state_all(eb), NKC.state_all(em))
    em2 = M.MethodsEngine(slots, ctx.D32, ctx.dev, graph=True)
    bt2 = bt.clone()
    bt2[:, [10, 11]] = bt2[:, [11, 10]]                        # mutant: two batches swapped
    em2.train_task(bt2)
    det = not NKC.teq(NKC.state_all(eb), NKC.state_all(em2))
    return {"bit_identical": eq, "mutants": {"two_batches_swapped": {"detected": det}}, "pass": eq}


def check_stats(ctx) -> dict:
    """unit_stats2 of a float64 engine (after 300 updates of task 1) vs an independent float64
    reference forward of each run.  ppos may differ only on samples whose sign is uncertain within
    2 E_z; dmax / amax within the derived phi' / phi bounds; dead / zeroout only where the bound
    straddles the threshold; eff_rank within 1e-6 (float64 Gram, relative error ~1e-11)."""
    slots = B.smoke_slots()
    e32 = B.BudgetEngine(slots, ctx.D32, ctx.dev, 780, graph=False)
    e32.train_task(ctx.tb(slots, 780, 1), n_steps=300)
    full = NKC.cast_full_state(e32.full_state(), torch.float64)
    D64 = ctx.D64
    e = B.BudgetEngine(slots, D64, ctx.dev, 780, graph=False, dtype=torch.float64)
    e.load_full_state(full)
    probe = NK.probe_ids(D64.tr_rows)[:256].to(ctx.dev)
    st, _ = B.unit_stats2(e.P, e.act, D64.X, probe[None].expand(e.R, -1), chunk=64)
    ok, worst, per = True, {"ppos": 0.0, "dmax": 0.0, "amax": 0.0, "eff_rank": 0.0}, {}
    muts = {"ppos_first_chunk_only": False, "dead_from_mean": False, "eff_rank_on_preactivation": False}
    for r, (name, seed) in enumerate(e.slots):
        cfg = e.cfgs[r]
        Pr = [p[r].detach().cpu() for p in e.P]
        alpha = [a[r].detach().cpu() for a in e.act.alpha_all()]
        x = D64.X[probe].cpu()
        f = MC.ref_forward_kind(Pr, x, cfg, alpha)
        y = torch.zeros(x.shape[0], dtype=torch.long)
        # forward error bounds of z (float64 path) from the previous checks' bound machinery
        A = [q.abs() for q in Pr]
        Ez = {}
        Ez[1] = NKC.gam(76, U64) * F.conv2d(x.abs(), A[0], A[1], padding=NK.PAD)
        Ea1 = MC.act_lip(cfg, alpha, 0, f["z1"], Ez[1], U64)
        Ez[2] = NKC.gam(401, U64) * F.conv2d(f["p1"].abs(), A[2], A[3], padding=NK.PAD) \
            + F.conv2d(F.max_pool2d(Ea1, 2, 2), A[2], None, padding=NK.PAD)
        Ea2 = MC.act_lip(cfg, alpha, 1, f["z2"], Ez[2], U64)
        Eh = F.max_pool2d(Ea2, 2, 2).flatten(1)
        Ez[3] = NKC.gam(1025, U64) * (f["h"].abs() @ A[4].T + A[5]) + Eh @ A[4].T
        Ea3 = MC.act_lip(cfg, alpha, 2, f["z3"], Ez[3], U64)
        Ez[4] = NKC.gam(101, U64) * (f["a3"].abs() @ A[6].T + A[7]) + Ea3 @ A[6].T
        Ea = {1: Ea1, 2: Ea2, 3: Ea3, 4: MC.act_lip(cfg, alpha, 3, f["z4"], Ez[4], U64)}
        pr = {}
        for l, tag in enumerate(NK.SITES):
            z, a = f[f"z{l + 1}"], f[f"a{l + 1}"]
            E2, Ea2_ = 2 * Ez[l + 1], 2 * Ea[l + 1]
            kz = cfg["kind"]
            aa = NKC.bc(alpha[l], z) if kz in ("snake", "kkt1") else None
            d = MC.ref_dphi(kz, z, aa, cfg["shape"])
            if kz == "relu":
                Ed = (z.abs() <= E2).double()
            elif kz in ("snake", "kkt1"):
                Ed = 2 * aa * E2 + 16 * U64 * (1 + aa * z.abs())
            elif kz == "elu":
                Ed = cfg["shape"][0] * torch.exp(torch.clamp(z, max=0.0)) * (torch.exp(E2) - 1) + 16 * U64
            else:
                Ed = torch.full_like(z, 16 * U64)
            dims = (0, 2, 3) if NK.IS_CONV[l] else (0,)
            n = z.numel() // NK.WIDTHS[l]
            ppos = (z > 0).double().mean(dims)
            unc = (z.abs() <= E2).double().sum(dims) / n
            dmax, amax = d.amax(dims), a.abs().amax(dims)
            got = {k: torch.from_numpy(st[tag][k][r]) for k in ("ppos", "dmax", "amax", "dead", "zeroout")}
            dpp = (got["ppos"] - ppos).abs()
            r_pp = float(torch.where(unc > 0, dpp / unc.clamp_min(1e-300),
                                     torch.where(dpp > 0, torch.full_like(dpp, math.inf), torch.zeros_like(dpp))).max())
            r_dm = float(((got["dmax"] - dmax).abs() / (Ed.amax(dims) + 1e-300)).max())
            r_am = float(((got["amax"] - amax).abs() / (Ea2_.amax(dims) + 1e-300)).max())
            dead_ok = bool(((got["dead"] == (dmax < B.DEAD_TOL).double()) |
                            ((dmax - B.DEAD_TOL).abs() <= Ed.amax(dims))).all())
            zero_ok = bool(((got["zeroout"] == (amax == 0).double()) | (amax <= Ea2_.amax(dims))).all())
            if NK.IS_CONV[l]:
                feat = a.permute(0, 2, 3, 1).reshape(-1, NK.CH)
            else:
                feat = a
            ev = torch.linalg.eigvalsh(feat.T @ feat).clamp_min(0.0)
            sv = ev.sqrt()
            pp_ = sv / sv.sum()
            er = float(torch.exp(-(pp_ * pp_.clamp_min(1e-300).log()).sum())) if float(sv.sum()) > 0 else 0.0
            r_er = abs(float(st[tag]["eff_rank"][r]) - er) / 1e-6
            pr[tag] = {"ppos": r_pp, "dmax": r_dm, "amax": r_am, "dead_ok": dead_ok, "zeroout_ok": zero_ok,
                       "eff_rank": r_er}
            for k, v in (("ppos", r_pp), ("dmax", r_dm), ("amax", r_am), ("eff_rank", r_er)):
                worst[k] = max(worst[k], v)
            ok &= r_pp <= 1 and r_dm <= 1 and r_am <= 1 and dead_ok and zero_ok and r_er <= 1
            # mutants
            # (counting a > 0 instead of z > 0 is not a mutant here: all four kinds keep sign(a) = sign(z))
            pp64 = (z[:64] > 0).double().mean(dims)
            muts["ppos_first_chunk_only"] |= bool(((pp64 - ppos).abs() > unc + 1e-12).any())
            muts["dead_from_mean"] |= bool(((d.mean(dims) < B.DEAD_TOL) != (dmax < B.DEAD_TOL)).any()) or \
                bool(((d.mean(dims) < 0.1) != (dmax < B.DEAD_TOL)).any())
            if NK.IS_CONV[l]:
                fz = z.permute(0, 2, 3, 1).reshape(-1, NK.CH)
            else:
                fz = z
            evz = torch.linalg.eigvalsh(fz.T @ fz).clamp_min(0.0).sqrt()
            pz = evz / evz.sum()
            erz = float(torch.exp(-(pz * pz.clamp_min(1e-300).log()).sum()))
            muts["eff_rank_on_preactivation"] |= abs(erz - float(st[tag]["eff_rank"][r])) > 1e-6
        per[f"{name}|{seed}"] = pr
    return {"worst_ratio": worst, "per": per, "mutants": {k: {"detected": v} for k, v in muts.items()}, "pass": ok}


def check_pure(ctx) -> dict:
    slots = B.smoke_slots()
    e = B.BudgetEngine(slots, ctx.D32, ctx.dev, 2340, graph=False)
    e.train_task(ctx.tb(slots, 2340, 1), n_steps=50)
    before = NKC.state_all(e)
    probe = NK.probe_ids(ctx.D32.tr_rows).to(ctx.dev)
    plans = {s: NK.task_plan(s) for s in SEEDS2}
    tr = NK.slot_rows(slots, plans, 1, ctx.D32.tr_rows).to(ctx.dev)
    B.unit_stats2(e.P, e.act, ctx.D32.X, probe[None].expand(e.R, -1))
    B.unit_stats2(e.P, e.act, ctx.D32.X, tr, ctx.D32.Y)
    NK.ce_acc(e.P, e.act, ctx.D32.X, ctx.D32.Y, tr)
    NK.control_arrays(e.act)
    ok = NKC.teq(NKC.state_all(e), before)
    orig = B.unit_stats2

    def leaky(P, act, X, rows, Y=None, chunk=50):
        out = E.forward(P, X[rows[:, :chunk]], act)
        act.update([out[0], out[2], out[4], out[6]])
        return orig(P, act, X, rows, Y, chunk)
    e2 = B.BudgetEngine(slots, ctx.D32, ctx.dev, 2340, graph=False)
    e2.load_full_state(e.full_state())
    b2 = NKC.state_all(e2)
    with patch.object(B, "unit_stats2", leaky):
        B.unit_stats2(e2.P, e2.act, ctx.D32.X, probe[None].expand(e2.R, -1))
    det = not NKC.teq(NKC.state_all(e2), b2)
    return {"state_unchanged": ok, "mutants": {"diagnostic_updates_V": {"detected": det}}, "pass": ok}


def check_graph(ctx) -> dict:
    slots = B.smoke_slots()

    def go(graph):
        e = B.BudgetEngine(slots, ctx.D32, ctx.dev, 2340, graph=graph)
        e.train_task(ctx.tb(slots, 2340, 1))
        e.act.freeze_runs([r for r, c in enumerate(e.cfgs) if c["base"] == "SNAfrz1"])
        e.train_task(ctx.tb(slots, 2340, 2), n_steps=100)
        return NKC.state_all(e)
    eq = NKC.teq(go(True), go(False))
    return {"bit_identical": eq, "pass": eq}


def check_resume_freeze_fresh(ctx, work: Path) -> dict:
    slots = B.smoke_slots()
    a, b = work / "continuous", work / "resumed"
    for p in (a, b):
        shutil.rmtree(p, ignore_errors=True)
    kw = dict(data=ctx.D32, log=quiet, allow_dirty=True, fresh_tasks=(1,), ckpt_boundaries=(0, 1, 2))
    B.run(slots, a, 2, 2340, ctx.dev, **kw)
    stopped = False
    try:
        B.run(slots, b, 2, 2340, ctx.dev, stop_after=1, **kw)
    except KeyboardInterrupt:
        stopped = True
    B.run(slots, b, 2, 2340, ctx.dev, **kw)
    sa = torch.load(a / "state.pt", weights_only=False)
    sb = torch.load(b / "state.pt", weights_only=False)
    files = ["per_task.csv", "probe_stats.csv", "train_stats.csv", "fresh_control.csv", "frozen_alpha.json",
             "batch_hashes.json"]
    txt = {f: (a / f).read_text() == (b / f).read_text() for f in files}
    resume_ok = stopped and NKC.teq(sa["state"], sb["state"]) and all(txt.values())
    # freeze: alpha after task 2 == the saved frozen alpha == alpha after task 1; V unchanged
    ck1 = torch.load(a / "checkpoints" / "b01.pt", weights_only=False)["state"]
    ck2 = torch.load(a / "checkpoints" / "b02.pt", weights_only=False)["state"]
    fz = json.loads((a / "frozen_alpha.json").read_text())["runs"]
    frz = [r for r, (n, s) in enumerate(slots) if n.startswith("SNAfrz1")]
    e1 = B.BudgetEngine(slots, ctx.D32, ctx.dev, 2340, graph=False); e1.load_full_state(ck1)
    e2 = B.BudgetEngine(slots, ctx.D32, ctx.dev, 2340, graph=False); e2.load_full_state(ck2)
    al1, al2 = e1.act.alpha_all(), e2.act.alpha_all()
    freeze_ok = all(torch.equal(al1[l][r], al2[l][r]) and torch.equal(e1.act.V[l][r], e2.act.V[l][r])
                    and np.allclose(al2[l][r].cpu().numpy(), np.asarray(fz[f"{slots[r][0]}|{slots[r][1]}"][NK.SITES[l]], dtype=np.float32), rtol=0, atol=0)
                    for l in range(4) for r in frz)
    others_move = all(not torch.equal(e1.act.V[l][r], e2.act.V[l][r]) for l in range(4)
                      for r, (n, s) in enumerate(slots) if n.startswith("KKT1"))
    # the frozen alpha depends on the budget: a 780 run of task 1
    c = work / "b780"
    shutil.rmtree(c, ignore_errors=True)
    B.run(slots, c, 1, 780, ctx.dev, **{**kw, "ckpt_boundaries": (0, 1)})
    fz780 = json.loads((c / "frozen_alpha.json").read_text())["runs"]
    differs = all(fz780[k]["f2"] != fz[k]["f2"] for k in fz)
    # fresh: initial state, the task's batch list, and in a 2-task run = continual task 1
    rows = list(csv.DictReader(open(a / "fresh_control.csv")))
    hashes = json.loads((a / "batch_hashes.json").read_text())["per_seed"]
    fresh_hash = all(q["batch_sha256"] == hashes[q["seed"]][0] for q in rows)
    repro = all(float(q["gap_online_acc"]) == 0.0 and float(q["gap_test_acc"]) == 0.0 for q in rows)
    st0 = torch.load(a / "checkpoints" / "b00.pt", weights_only=False)["state"]
    fe = B.BudgetEngine(slots, ctx.D32, ctx.dev, 2340, graph=False, stage="fresh")
    init_eq = NKC.teq({k: fe.full_state()[k] for k in ("P", "m", "v", "tc", "act")},
                      {k: st0[k] for k in ("P", "m", "v", "tc", "act")})
    # mutants: resume without the activation state; a fresh engine that keeps the end Adam moments
    d = work / "resumed_noact"
    shutil.rmtree(d, ignore_errors=True)
    try:
        B.run(slots, d, 2, 2340, ctx.dev, stop_after=1, **kw)
    except KeyboardInterrupt:
        pass
    orig = B.BudgetEngine.load_full_state

    def no_act(self, st):
        st = dict(st)
        st["act"] = B.BudgetEngine(self.slots, self.data, self.device, self.budget, graph=False).full_state()["act"]
        return orig(self, st)
    with patch.object(B.BudgetEngine, "load_full_state", no_act):
        B.run(slots, d, 2, 2340, ctx.dev, **kw)
    det_resume = (d / "per_task.csv").read_text() != (a / "per_task.csv").read_text()
    return {"resume": {"stopped": stopped, "files_identical": txt, "ok": resume_ok},
            "freeze": {"alpha_V_fixed_and_saved": freeze_ok, "kkt1_V_moves": others_move,
                       "frozen_alpha_depends_on_budget": differs},
            "fresh": {"batch_hash_is_the_task": fresh_hash, "reproduces_continual_task1": repro,
                      "initial_state_equals_b00": init_eq},
            "mutants": {"resume_without_activation_state": {"detected": det_resume}},
            "pass": resume_ok and freeze_ok and others_move and differs and fresh_hash and repro and init_eq}


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
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = NK.setup("cuda")
    t0 = time.time()
    ctx = Ctx(dev)
    work = Path(tempfile.mkdtemp(prefix="budget_checks_", dir=str(out)))
    plan = [("S-prefix", lambda: check_prefix(ctx)), ("S-engine-780", lambda: check_engine780(ctx)),
            ("S-stats", lambda: check_stats(ctx)), ("S-pure", lambda: check_pure(ctx)),
            ("S-graph", lambda: check_graph(ctx)),
            ("S-resume-freeze-fresh", lambda: check_resume_freeze_fresh(ctx, work))]
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
        print(f"[{time.strftime('%T')}] {name:22s} pass={r.get('pass')} mutants={mut} ({r['seconds']}s)"
              + (f"  ERROR {r['error']}" if "error" in r else ""), flush=True)
    summary = {"scope": "implementation checks only; no performance verdict", "all_pass": all_pass(res),
               "checks": res, "env": NK.env_info(dev), "git": NK.git_state(), "sources_sha256": NK.source_hashes(),
               "spec_sha256": hashlib.sha256(B.SPEC.read_bytes()).hexdigest(), "seconds": round(time.time() - t0, 1)}
    NK.write_json(out / "checks.json", summary)
    shutil.rmtree(work, ignore_errors=True)
    print(f"all_pass={summary['all_pass']} -> {out / 'checks.json'}")


if __name__ == "__main__":
    main()

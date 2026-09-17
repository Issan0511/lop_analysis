"""Checks for neffdir_ee_0918 (specs/spec_neffdir_ee_0918.md section 6).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/neffdir_ee_0918/checks.py
    ... --only S1b           # development: writes results/_checks_neffdir_ee_0918/checks_partial.json

Each check runs on the real code and then on every mutation listed with it (an exact-once substitution in
the check's target file); the mutation MUST make the same check fail.  No check prints or stores an arm's
accuracy (the checks run before the predictions are registered).

S1a  records (seed 0, 80 epochs): the identity-carrier arms through this runner are respdyn / escape /
     resp_ee's arms bit for bit, and the probe's raw n_eff is escape's recorded value
S1b  carriers (seed 45, unregistered, 8 epochs): the q = 1 / r = 0 arms are S2dyn_10r bit for bit; branch
     logits, shadows and fields; mask shares, nesting and draws; the lifted field; the autograd factor of
     the mask and of F.elu on a real minibatch; the structural probe
S1c  derivative identities on a synthetic grid (F.elu vs dphi_felu, the host ELU vs dphi_train)
S8   the verdict on synthetic seeds 30-39
S9   the CLI's provenance and constants
S-cost  PROBE seeds at once (three arms), projected to the full seed
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_flush_denormal(True)                   # the runner's setting
from src import pmnist_0905 as H                 # noqa: E402
from src import pmnist_rlmnist_0906 as RL        # noqa: E402
from src import escape_ee_0917 as ES             # noqa: E402
from src import resp_ee_0917 as RE               # noqa: E402

RUNNER = REPO / "src" / "neffdir_ee_0918.py"
VERDICT = REPO / "analysis" / "neffdir_ee_0918" / "verdict.py"
OUT = REPO / "results" / "neffdir_ee_0918"
SCR = REPO / "results" / "_checks_neffdir_ee_0918"
EPS32 = float(np.finfo(np.float32).eps)
DEV = H.setup("cpu")
MNIST = H.Mnist(DEV)
PROBE = 3
PY = sys.executable
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
RESULTS: dict = {}
DUMP_PATH = OUT / "checks.json"
TOL_CAP = 1e-4        # a capped state is above its radius by at most 4 * eps32 * R (R <= 200): 9.5e-5
KEEP_TOL = 0.01       # > 6 binomial sd of a share of 120000 pairs at q = 0.5 (0.0087)
DEV_SEED = 45         # S1b / S9: outside the registered 30-39 and the S-cost 40-42
DEV_EPOCHS = 8        # S1b: bit identities hold at any length; 600 updates per task keeps it cheap

_n_loaded = [0]


def load(path: Path, subs=()):
    src = path.read_text()
    for old, new in subs:
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation anchor occurs {n} times in {path.name}: {old!r}")
        src = src.replace(old, new)
    _n_loaded[0] += 1
    name = f"_chk_{path.stem}_{_n_loaded[0]}"
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def brief(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "pass" and not isinstance(v, (dict, list))} | \
        {"failed_items": r.get("failed_items", [])[:8]}


def dump() -> None:
    checks = {k: v for k, v in RESULTS.items() if k.startswith("S") and isinstance(v, dict) and "pass" in v}
    RESULTS["all_pass"] = bool(checks) and all(
        v.get("pass") and v.get("all_mutations_detected", True) for v in checks.values())
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMP_PATH.write_text(json.dumps(RESULTS, indent=2, default=str))


def run_check(key, title, fn, mutations, derivation, target) -> None:
    t0 = time.time()
    base = fn(load(target))
    entry = {"title": title, "target": str(target.relative_to(REPO)), "threshold_derivation": derivation,
             **base, "mutations": []}
    for label, subs in mutations:
        mod = load(target, subs)
        try:
            r = fn(mod)
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": bool(r["pass"]),
                                       "detected": not r["pass"], "mutant_result": brief(r)})
        except Exception as e:
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
    entry["all_mutations_detected"] = (not mutations) or all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:3]) if not entry['pass'] else ''}", flush=True)


def _flush_is_on() -> bool:
    return float(torch.tensor([1e-30], dtype=torch.float32) * torch.tensor([1e-10], dtype=torch.float32)) == 0.0


def _rows(path: Path) -> dict:
    return {(r["arm"], int(r["k"])): r for r in csv.DictReader(path.open())}


_t = lambda v: v == "True"
_f = lambda v: float(v) if v not in ("", None) else float("nan")
_PRE: dict = {}


def _prefix(seed: int, epochs: int) -> dict:
    key = (seed, epochs)
    if key not in _PRE:
        _PRE[key] = ES.prefix(seed, MNIST, epochs, progress=False)
    pre = _PRE[key]
    return {**pre, "rows": [dict(r) for r in pre["rows"]]}


# --------------------------------------------------------------------------
# S1a: the identity-carrier arms reproduce the records (seed 0, 80 epochs)
# --------------------------------------------------------------------------

S1A_ARMS = ("S2dyn_10r", "S2dyn_c12", "S2u30r", "N2r")
ESCAPE_ARMS = REPO / "results" / "escape_ee_0917" / "runs" / "s0" / "arms.csv"


def s1a(M) -> dict:
    """Seed 0, 80 epochs, the escape prefix computed once, the four arms that exist in the records through
    this runner: (i) branch-point logits equal (full batch and first minibatch); (ii) state sha256 equals
    every record of the same computation in both tasks (S2dyn_10r: respdyn, escape; S2dyn_c12: escape;
    S2u30r and N2r: resp_ee, respdyn, escape); (iii) shadows are N10 and d(0) is the fixed field;
    (iv) the c12 hold stays within TOL_CAP and writes rows in both tasks; (v) the probe's raw nbar_neffT2
    equals escape's recorded value exactly (CSV repr round trip) and the structural nbar_neffS2 equals it
    for these identity carriers; (vi) provenance names this experiment and the prefix matched resp_ee on
    all 12 tasks.  Bit comparisons only."""
    out = SCR / "s1a_run"
    M.run_seed(0, out, MNIST, 80, arms=list(S1A_ARMS), progress=False, pre=_prefix(0, 80))
    rows = _rows(out / "arms.csv")
    esc = _rows(ESCAPE_ARMS)
    pv = json.loads((out / "provenance.json").read_text())
    per = {}
    per["i_logits"] = all(_t(rows[(a, 1)]["logits_equal_full"]) and _t(rows[(a, 1)]["logits_equal_mb"]) for a in S1A_ARMS)
    want = {"S2dyn_10r": ("respdyn", "escape"), "S2dyn_c12": ("escape",), "S2u30r": ("resp_ee", "respdyn", "escape"),
            "N2r": ("resp_ee", "respdyn", "escape")}
    per["ii_records"] = all(_t(rows[(a, k)][f"hash_match_{src}"]) for a, srcs in want.items() for src in srcs
                            for k in (1, 2))
    per["iii_shadows"] = all(_t(rows[(a, k)]["shadow_hash_match_natural"]) for a in ("S2dyn_10r", "S2dyn_c12")
                             for k in (1, 2)) and all(_t(rows[(a, 1)]["field0_equal_fixed"])
                                                      for a in ("S2dyn_10r", "S2dyn_c12"))
    per["iv_hold"] = all(max(_f(rows[("S2dyn_c12", k)][c]) for c in ("exc_q", "exc_v", "exc_w2")) <= TOL_CAP
                         and _f(rows[("S2dyn_c12", k)]["rows_w2"]) > 0 for k in (1, 2))
    per["v_probe"] = all(rows[(a, k)]["nbar_neffT2"] == esc[(a, k)]["nbar_neffT2"]
                         and rows[(a, k)]["nbar_neffS2"] == rows[(a, k)]["nbar_neffT2"]
                         for a in S1A_ARMS for k in (1, 2))
    per["vi_provenance"] = (pv["experiment"] == "neffdir_ee_0918" and pv["prefix_hash_match_resp_ee"] == 12
                            and pv["prefix_record_tasks_resp_ee"] == 12)
    failed = [k for k, v in per.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


A_FWD = "        a2 = a02 + (gate(phi2(z2 + e2), ob) - A2)\n"
A_HOLD = '    hold = None if arm["hold"] == "none" else ES.Hold(arm["hold"], ck["params"], e1)\n'
A_FLOOR = '    "S2u30r": _arm("uniform", delta=30.0),                      # the response floor\n'
A_PROBE_S = "    gt = car.dphi2(ze32).double()\n"
S1A_MUT = [
    ("M1a: the anchored sum re-associated", [(A_FWD, "        a2 = (a02 - A2) + gate(phi2(z2 + e2), ob)\n")]),
    ("M1b: the c12 hold becomes c1", [(A_HOLD, '    hold = None if arm["hold"] == "none" else ES.Hold("c1", ck["params"], e1)\n')]),
    ("M1c: the floor at 29", [(A_FLOOR, '    "S2u30r": _arm("uniform", delta=29.0),\n')]),
    ("M1d: the structural probe off by 1e-3", [(A_PROBE_S, "    gt = car.dphi2(ze32 - 1e-3).double()\n")]),
]


# --------------------------------------------------------------------------
# S1b: the carriers (seed 45, 8 epochs)
# --------------------------------------------------------------------------

S1B_ARMS = ("S2dyn_10r", "S2dyn_m100", "S2dyn_s100", "S2dyn_a00", "S2dyn_m50", "S2dyn_m25", "S2dyn_s50",
            "S2dyn_a02", "S2dyn_felu", "S2dyn_c12_m50", "S2dyn_c12_a02", "N2r_m50")


def _grad_factor(M, arm_name: str, pre: dict):
    """At the t2 branch with the arm's own field / carrier: d(sum(a2 * G))/dz2 on the first minibatch of
    task 3's order stream, and the carrier's own factor G * s * dphi2(z2 + e2) (s = 1/q on kept pairs)."""
    arm = {**M.ARMS, **M.CHECK_ARMS}[arm_name]
    cks, x = pre["cks"], pre["x"]
    U, V = M.tables(DEV_SEED)
    sh, mover = M.build(arm, cks, x, V)
    car = M.make_carrier(arm, DEV_SEED, U)
    fwd = M.make_forward(sh, car)
    if mover is not None:
        mover.begin_task(RE.SPE * DEV_EPOCHS)
        mover.refresh(0)
    ck = cks[2]
    params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
    order = torch.randperm(RE.N_IMG, generator=RE.gen_from(ck["g_batch"]))
    ob = order[:RE.BATCH]
    out = fwd(params, x[ob], ob)
    G = torch.linspace(-1.0, 1.0, out[3].numel()).reshape(out[3].shape)
    got = torch.autograd.grad((out[3] * G).sum(), out[2])[0]
    e2 = sh[2][ob] if sh is not None else torch.zeros_like(out[2])
    ze = out[2].detach() + e2
    s = car.s[ob] if hasattr(car, "s") else torch.ones_like(ze)
    want = (G * s) * car.dphi2(ze)
    return got, want, car


def s1b(M) -> dict:
    """Seed 45 (no registered arm there), 8 epochs, the prefix computed once:
    (i) S2dyn_m100, S2dyn_s100 and S2dyn_a00 (q = 1, r = 0) have S2dyn_10r's state sha256 in both tasks;
    (ii) every arm's branch-point logits (full batch, first minibatch) equal the natural t2 net's, every
    dyn arm's shadow is N10, d(0) is the fixed field off the lifted pairs and the constructed lift on them;
    (iii) the fixed masks keep q of the pairs (|share - q| <= KEEP_TOL), m25's kept pairs are inside m50's,
    and the step mask kept q and drew once per gradient-enabled forward (75 preflight + 600 per task);
    (iv) the lifted share is 0.02 within KEEP_TOL and each lifted pair sits at its shadow unit's top
    (|z2_branch + d(0) - top| <= 2 eps32 (|top| + |z2_branch|), float rounding of one sum and one cast);
    (v) the autograd factor d(sum a2 G)/dz2 on a real minibatch equals G s dphi2(z2 + d) exactly for the
    fixed mask (s = 1/q on kept pairs, 0 off), F.elu (dphi2 = exp) and the identity;
    (vi) the structural probe at the branch equals an independent recomputation: the masked factor for
    m50, F.elu's for felu, the raw one for the identity."""
    out = SCR / "s1b_run"
    pre = _prefix(DEV_SEED, DEV_EPOCHS)
    table = {**M.ARMS, **M.CHECK_ARMS}
    M.run_seed(DEV_SEED, out, MNIST, DEV_EPOCHS, arms=list(S1B_ARMS), progress=False, pre=pre, arm_table=table)
    rows = _rows(out / "arms.csv")
    per = {}
    ref = {k: rows[("S2dyn_10r", k)]["state_sha256"] for k in (1, 2)}
    per["i_identity"] = all(rows[(a, k)]["state_sha256"] == ref[k] for a in ("S2dyn_m100", "S2dyn_s100", "S2dyn_a00")
                            for k in (1, 2))
    dyn = [a for a in S1B_ARMS if table[a]["kind"] == "dyn"]
    per["ii_logits"] = all(_t(rows[(a, 1)]["logits_equal_full"]) and _t(rows[(a, 1)]["logits_equal_mb"]) for a in S1B_ARMS)
    per["ii_shadow_field"] = (all(_t(rows[(a, k)]["shadow_hash_match_natural"]) for a in dyn for k in (1, 2))
                              and all(_t(rows[(a, 1)]["field0_equal_fixed"]) for a in dyn)
                              and all(_t(rows[(a, 1)]["field0_top_ok"]) for a in ("S2dyn_a00", "S2dyn_a02", "S2dyn_c12_a02")))
    per["iii_keep"] = all(abs(_f(rows[(a, k)]["keep_share"]) - table[a]["q"]) <= KEEP_TOL
                          for a in ("S2dyn_m50", "S2dyn_m25", "S2dyn_s50", "S2dyn_c12_m50", "N2r_m50") for k in (1, 2))
    U, V = M.tables(DEV_SEED)
    k50, k25 = M.FixedMask(U, 0.5).keep, M.FixedMask(U, 0.25).keep
    per["iii_nested"] = bool((k25 & ~k50).sum() == 0) and bool(k25.sum() > 0)
    spt = RE.SPE * DEV_EPOCHS
    per["iii_draws"] = (int(rows[("S2dyn_s50", 1)]["step_draws"]) == RE.SPE + spt
                        and int(rows[("S2dyn_s50", 2)]["step_draws"]) == RE.SPE + 2 * spt)
    per["iv_lift_share"] = all(abs(_f(rows[(a, 1)]["lift_share"]) - 0.02) <= KEEP_TOL for a in ("S2dyn_a02", "S2dyn_c12_a02"))
    per["iv_held"] = all(max(_f(rows[(a, k)][c]) for c in ("exc_q", "exc_v", "exc_w2")) <= TOL_CAP
                         and _f(rows[(a, k)]["rows_w2"]) + _f(rows[(a, k)]["rows_perp"]) + _f(rows[(a, k)]["rows_par"]) > 0
                         for a in ("S2dyn_c12_m50", "S2dyn_c12_a02") for k in (1, 2))
    per["iv_early_cols"] = all(math.isfinite(_f(rows[(a, k)].get("early_acc"))) and
                               math.isfinite(_f(rows[(a, k)].get("nbar_neffS2_early"))) for a in S1B_ARMS for k in (1, 2))
    cks, x = pre["cks"], pre["x"]
    arm = table["S2dyn_a02"]
    sh, mover = M.build(arm, cks, x, V)
    mover.begin_task(spt)
    mover.refresh(0)
    S = mover.S
    zb = cks[2]["z2"]
    top = RE.EL.forward2(cks[10]["params"], x, RE.ACT, RE.ACT)[2].amax(0)
    ze = (zb + sh[2]).double()
    topm = top.double().unsqueeze(0).expand_as(ze)
    tol = 2 * EPS32 * (topm.abs() + zb.double().abs())
    per["iv_lift_top"] = bool(S.any()) and bool(((ze - topm).abs() <= tol)[S].all()) and \
        bool(torch.equal(sh[2][~S], RE.build_shift({**arm, "kind": "field"}, cks)[2][~S]))
    fac = {}
    for a in ("S2dyn_m50", "S2dyn_felu", "S2dyn_10r", "N2r_m50"):
        got, want, _ = _grad_factor(M, a, pre)
        fac[a] = bool(torch.equal(got, want))
    per["v_autograd"] = all(fac.values())
    # (vi) the structural probe at the branch, recomputed
    pv = {}
    for a in ("S2dyn_m50", "S2dyn_felu", "S2dyn_10r"):
        arm = table[a]
        sh, mover = M.build(arm, cks, x, V)
        car = M.make_carrier(arm, DEV_SEED, U)
        if mover is not None:
            mover.begin_task(spt)
            mover.refresh(0)
        ze32 = zb + sh[2]
        if a == "S2dyn_felu":
            g = torch.where(ze32 > 0, torch.ones_like(ze32), torch.exp(ze32.clamp(max=0.0))).double()
        else:
            g = RE.dphi_train(ze32).double()
        if a == "S2dyn_m50":
            g = g * (U < 0.5).double()
        s1 = g.sum(0)
        ne = torch.where(s1 > 0, s1 * s1 / (g.shape[0] * g.square().sum(0)).clamp(min=1e-300), torch.zeros_like(s1))
        pv[a] = float(ne.mean()) == _f(rows[(a, 1)]["neffS2_start"])
    per["vi_probe"] = all(pv.values()) and rows[("S2dyn_10r", 1)]["neffS2_start"] == rows[("S2dyn_10r", 1)]["neffT2_start"]
    failed = [k for k, v in per.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": {**per, "factor": fac, "probe": pv}}


B_GATE = "        return ud + self.s[ob] * (u - ud)\n"
B_SCALE = "        self.s = torch.where(self.keep, torch.tensor(1.0 / self.q), torch.tensor(0.0))\n"
B_DRAW = "        if not torch.is_grad_enabled():          # probes and summaries: value only, no draw\n"
B_WHERE = "            dd = torch.where(self.S, top.unsqueeze(0) - self.zbr, z2.double() - self.zbr)\n"
B_TOP = "            top = z2.amax(0).double()\n"
B_KEEP = "        self.keep = U < self.q\n"
B_PROBE = "    if car.keep is not None:\n        gt = gt * car.keep.double()\n"
B_FELU = "        return torch.autograd.grad(F.elu(zz, alpha=1.0).sum(), zz)[0]\n"
S1B_MUT = [
    ("M2a: the fixed mask leaks into the forward value", [(B_GATE, "        return ud * self.s[ob] + (u - ud)\n")]),
    ("M2b: kept pairs not rescaled by 1/q", [(B_SCALE, "        self.s = torch.where(self.keep, torch.tensor(1.0), torch.tensor(0.0))\n")]),
    ("M2c: the step mask also draws without gradients", [(B_DRAW, "        if False:\n")]),
    ("M2d: the complement is lifted", [(B_WHERE, "            dd = torch.where(~self.S, top.unsqueeze(0) - self.zbr, z2.double() - self.zbr)\n")]),
    ("M2e: the lift goes to the unit's bottom", [(B_TOP, "            top = z2.amin(0).double()\n")]),
    ("M2f: the masks are not nested", [(B_KEEP, "        self.keep = torch.rand(U.shape) < self.q\n")]),
    ("M2g: the structural probe ignores the mask", [(B_PROBE, "    if False:\n        gt = gt * car.keep.double()\n")]),
    ("M2h: F.elu's factor taken as exp everywhere", [(B_FELU, "        return torch.exp(zz.detach())\n")]),
]


# --------------------------------------------------------------------------
# S1c: derivative identities on a synthetic grid
# --------------------------------------------------------------------------

def s1c(M) -> dict:
    """float32 z on a grid over [-100, 5] (step 1/64, plus the float32 neighbours of -16.64 and -87.34):
    (i) autograd of M.FeluCarrier.phi2 equals M.dphi_felu and autograd of M.Carrier.phi2 equals
    M.Carrier.dphi2 (= RE.dphi_train) exactly; (ii) M.dphi_felu is 1 above 0 and exp(z) below it to two
    ulps (|factor / torch.exp(z) - 1| <= 2 eps32: two float32 exp kernels, each within one ulp), except where
    it is 0; (iii) the host factor is 0 exactly below ln 2^-24 and not above it, F.elu's factor is 0 only
    below the float32 normal range with flush on (ln 1.1755e-38 = -87.34); (iv) the phi2 values: the
    identity carrier's is the host ELU bit for bit, F.elu's is F.elu and within one float32 step of -1
    (2^-24) of the host ELU (they differ only where the host's expm1 kernel truncates), so the felu arm
    moves the backward and, at most, a forward value by 2^-24."""
    z = torch.arange(-6400, 321, dtype=torch.float32) / 64.0
    extra = torch.tensor([RE.ZERO_Z32, math.log(float(np.finfo(np.float32).tiny))], dtype=torch.float32)
    z = torch.cat([z, extra, torch.nextafter(extra, torch.full_like(extra, -1e9)),
                   torch.nextafter(extra, torch.full_like(extra, 1e9))])

    def factor(fn):
        zz = z.clone().requires_grad_(True)
        return torch.autograd.grad(fn(zz).sum(), zz)[0]

    g_felu, g_host = factor(M.FeluCarrier.phi2), factor(M.Carrier.phi2)
    fd = M.dphi_felu(z)
    per = {"i_felu": bool(torch.equal(g_felu, fd)), "i_host": bool(torch.equal(g_host, M.Carrier.dphi2(z)))
           and bool(torch.equal(M.Carrier.dphi2(z), RE.dphi_train(z)))}
    neg = (z <= 0) & (fd > 0)
    rel = (fd[neg].double() / torch.exp(z[neg]).double() - 1).abs()
    per["ii_felu_exp"] = bool((fd[z > 0] == 1).all()) and bool((rel <= 2 * EPS32).all())
    zero_h, zero_f = (g_host == 0), (fd == 0)
    per["iii_host_zero"] = bool(zero_h[z < RE.ZERO_Z32 - 1e-3].all()) and bool((~zero_h)[z > RE.ZERO_Z32 + 1e-3].all())
    lim = math.log(float(np.finfo(np.float32).tiny))
    per["iii_felu_zero"] = bool(zero_f[z < lim - 1e-3].all()) and bool((~zero_f)[z > lim + 1e-3].all())
    fv = M.FeluCarrier.phi2(z)
    per["iv_phi2"] = (bool(torch.equal(M.Carrier.phi2(z), RE.ACT.phi(z))) and bool(torch.equal(fv, F.elu(z)))
                      and bool(((fv.double() - RE.ACT.phi(z).double()).abs() <= 2.0 ** -24).all()))
    failed = [k for k, v in per.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S1C_MUT = [
    ("M3a: the felu carrier uses the host ELU", [("    phi2 = staticmethod(_felu)\n", "    phi2 = staticmethod(ACT.phi)\n")]),
    ("M3b: dphi_felu is the host's factor", [(B_FELU, "        return RE.dphi_train(zz.detach())\n")]),
]


# --------------------------------------------------------------------------
# S8: the verdict on synthetic seeds 30-39
# --------------------------------------------------------------------------

TOL_EST = 1e-9
T_TABLE = {(0.975, 9): 2.262157, (0.9875, 9): 2.685011, (0.95, 9): 1.833113}   # published t quantiles
SEEDS_REG = tuple(range(30, 40))
BETA, BETA_E = 11.0, 11.0
FLOOR, FLOOR_E = 0.22, 0.15
DYN_S8 = ("S2dyn_10r", "S2dyn_m50", "S2dyn_m25", "S2dyn_s50", "S2dyn_a02", "S2dyn_felu", "S2dyn_c12",
          "S2dyn_c12_m50", "S2dyn_c12_a02")
BASE_N = {"S2dyn_10r": 0.011, "S2dyn_m50": 0.0045, "S2dyn_m25": 0.0022, "S2dyn_s50": 0.0108, "S2dyn_a02": 0.024,
          "S2dyn_felu": 0.011, "S2dyn_c12": 0.0075, "S2dyn_c12_m50": 0.0033, "S2dyn_c12_a02": 0.018,
          "S2u30r": 0.0, "N2r": 0.45, "N2r_m50": 0.23}
BASE_T = {**BASE_N, "S2dyn_m50": 0.016, "S2dyn_m25": 0.020, "S2dyn_c12_m50": 0.010, "N2r_m50": 0.46}
BASE_NE = {"S2dyn_10r": 0.008, "S2dyn_m50": 0.004, "S2dyn_m25": 0.002, "S2dyn_s50": 0.008, "S2dyn_a02": 0.020,
           "S2dyn_felu": 0.008, "S2dyn_c12": 0.008, "S2dyn_c12_m50": 0.004, "S2dyn_c12_a02": 0.016,
           "S2u30r": 0.0, "N2r": 0.45, "N2r_m50": 0.23}
BASE_Z = {"S2dyn_10r": 0.95, "S2dyn_m50": 0.975, "S2dyn_m25": 0.99, "S2dyn_s50": 0.95, "S2dyn_a02": 0.93,
          "S2dyn_felu": 0.15, "S2dyn_c12": 0.96, "S2dyn_c12_m50": 0.98, "S2dyn_c12_a02": 0.94,
          "S2u30r": 1.0, "N2r": 0.0, "N2r_m50": 0.5}
Q_OF = {"S2dyn_m50": 0.5, "S2dyn_m25": 0.25, "S2dyn_s50": 0.5, "S2dyn_c12_m50": 0.5, "N2r_m50": 0.5}
TOPS = ("S2dyn_a02", "S2dyn_c12_a02")
HELDS = ("S2dyn_c12", "S2dyn_c12_m50", "S2dyn_c12_a02")


def _centered(rng, n, sd):
    x = rng.normal(0.0, 1.0, n)
    x = x - x.mean()
    return x / x.std(ddof=1) * sd if sd > 0 else np.zeros(n)


def synth(V, scen: dict) -> dict:
    """Seeds 30-39.  n (nbar_neffS2) = BASE_N (+ scen['n']), plus centred noise scen['n_noise'][arm]; raw n
    (nbar_neffT2) = BASE_T (+ scen['t']); early n = BASE_NE (+ scen['ne']).  E = FLOOR + BETA * n for the
    shadow arms (the proportional model), N2r 0.72, N2r_m50 0.70, floor FLOOR; overrides scen['E'] (absolute)
    and scen['dE'] (added).  early_acc likewise with FLOOR_E, BETA_E, BASE_NE, scen['Ee'].  Every arm gets
    m[seed] (centred, sd scen['seed_sd'], default 0.02) and its own centred noise (sd scen['noise'][arm],
    default 0.004; none for S2dyn_10r, S2dyn_c12 and the floor); the early window has its own m and noise.
    Zero shares BASE_Z (+ scen['zero']).  Records as a registered seed writes them (empty)."""
    rng = np.random.default_rng(scen.get("rng", 918))
    Nn = {**BASE_N, **scen.get("n", {})}
    Tn = {**BASE_T, **scen.get("t", {})}
    NE = {**BASE_NE, **scen.get("ne", {})}
    Zs = {**BASE_Z, **scen.get("zero", {})}
    E = {a: FLOOR + BETA * Nn[a] for a in DYN_S8} | {"S2u30r": FLOOR, "N2r": 0.72, "N2r_m50": 0.70}
    Ee = {a: FLOOR_E + BETA_E * NE[a] for a in DYN_S8} | {"S2u30r": FLOOR_E, "N2r": 0.60, "N2r_m50": 0.58}
    E |= scen.get("E", {})
    Ee |= scen.get("Ee", {})
    for a, d in scen.get("dE", {}).items():
        E[a] += d
    quiet = ("S2dyn_10r", "S2dyn_c12", "S2u30r")
    m, me = _centered(rng, 10, scen.get("seed_sd", 0.02)), _centered(rng, 10, scen.get("seed_sd", 0.02))
    nE = {a: (np.zeros(10) if a in quiet else _centered(rng, 10, scen.get("noise", {}).get(a, 0.004))) for a in V.ARMS}
    nEe = {a: (np.zeros(10) if a in quiet else _centered(rng, 10, 0.004)) for a in V.ARMS}
    nN = {a: _centered(rng, 10, scen.get("n_noise", {}).get(a, 0.0)) for a in V.ARMS}
    seeds = {}
    for i, s in enumerate(SEEDS_REG):
        arms = []
        for a in V.ARMS:
            for k in (1, 2):
                r = {"arm": a, "k": str(k), "online_acc": repr(float(E[a] + m[i] + nE[a][i])),
                     "early_acc": repr(float(Ee[a] + me[i] + nEe[a][i])),
                     "nbar_neffS2": repr(float(Nn[a] + nN[a][i])), "nbar_neffT2": repr(float(Tn[a])),
                     "nbar_neffS2_early": repr(float(NE[a])),
                     "probe_zeroS2_mean": repr(float(Zs[a])), "probe_zero2_mean": repr(float(Zs[a])),
                     "logits_equal_full": "True" if k == 1 else "", "logits_equal_mb": "True" if k == 1 else "",
                     "field0_equal_fixed": "True" if a in DYN_S8 else "",
                     "shadow_hash_match_natural": "True" if a in DYN_S8 else "",
                     "hash_match_resp_ee": "", "hash_match_respdyn": "", "hash_match_escape": "",
                     "rows_par": "3", "rows_perp": "3", "rows_w2": "3"}
                if a in Q_OF:
                    r["keep_share"] = repr(Q_OF[a] + 0.001)
                if a in TOPS:
                    r |= {"field0_top_ok": "True", "lift_share": "0.0201"}
                if a in HELDS:
                    r |= {"exc_q": "0.0", "exc_v": "0.0", "exc_w2": "0.0"}
                for (ba, bs, bk, col, val) in scen.get("break", ()):
                    if ba == a and bs == s and bk == k:
                        r[col] = val
                arms.append(r)
        prefix = [{"task": str(t), "hash_match_resp_ee": ""} for t in range(1, 13)]
        for (ps, col, val) in scen.get("prefix_break", ()):
            if ps == s:
                for p in prefix:
                    p[col] = val
        seeds[s] = {"arms": arms, "prefix": prefix, "prov": {"git_hash": "synthetic"}}
    return seeds


def _brk(arm, col, val, k=1, seeds=(37, 38, 39)):
    return tuple((arm, s, k, col, val) for s in seeds)


_HALF95 = T_TABLE[(0.975, 9)] / math.sqrt(10)                     # 95% two-sided half width per unit sd, n = 10
_HALF975 = T_TABLE[(0.9875, 9)] / math.sqrt(10)                   # 97.5% two-sided
_W95 = 0.004 * _HALF95
_MID = 0.5 * 0.004 * (_HALF95 + _HALF975)                        # significant at 95%, not at 97.5% (sd 0.004)
_EN, _EC = FLOOR + BETA * 0.011, FLOOR + BETA * 0.0075           # E of the two bases
_ENe, _ECe = FLOOR_E + BETA_E * 0.008, FLOOR_E + BETA_E * 0.008
DESIGNED = {"primary": "ADD_H_MOVES_E", "prop_add_h": "PROPORTIONAL", "add": "ADD_MOVES_E", "prop_add": "PROPORTIONAL",
            "drop_h": "DROP_H_MOVES_E", "prop_drop_h": "PROPORTIONAL", "drop": "DROP_MOVES_E",
            "prop_drop": "PROPORTIONAL", "m25": "M25_MOVES_E", "drop_early": "DROP_EARLY_MOVES_E",
            "prop_drop_early": "PROPORTIONAL", "drop_h_early": "DROP_H_EARLY_MOVES_E",
            "prop_drop_h_early": "PROPORTIONAL", "compensation": "COMPENSATES", "compensation_h": "COMPENSATES_H",
            "structure": "STRUCTURE_MATTERS", "felu": "FELU_SAME", "ladder": "NEFF_TRACKS",
            "growth": "GROWTH_REMAINDER"}
S8_SCENARIOS = [
    ("designed", {}, DESIGNED,
     {"R_none": BETA * 0.011, "R_c12": BETA * 0.0075, "dR_add_h": BETA * 0.0105, "dn_add_h": 0.0105,
      "kappa_add_h": 0.018 / 0.0075, "phi_add_h": 0.0, "dR_add": BETA * 0.013, "phi_add": 0.0,
      "dR_drop_h": BETA * 0.0042, "dn_drop_h": 0.0042, "phi_drop_h": 0.0, "dR_drop": BETA * 0.0065,
      "dn_drop": 0.0065, "kappa_drop": 0.0045 / 0.011, "phi_drop": 0.0, "dR_m25": BETA * 0.0088,
      "comp": 0.005, "comp_h": 0.0025, "struct": BETA * (0.0108 - 0.0045), "d_felu": 0.0, "dzero_felu": 0.80,
      "dn_felu": 0.0, "dR_growth": BETA * 0.0035, "I_nat": 0.02, "R_none_e": BETA_E * 0.008,
      "dR_drop_e": BETA_E * 0.004, "dn_drop_e": 0.004, "phi_drop_e": 0.0, "dR_drop_h_e": BETA_E * 0.004,
      "phi_drop_h_e": 0.0, "dR_add_h_e": BETA_E * 0.008, "n_valid": 10}),
    ("marker", {"E": {"S2dyn_c12_a02": _EC, "S2dyn_c12_m50": _EC, "S2dyn_a02": _EN, "S2dyn_m50": _EN,
                      "S2dyn_m25": _EN, "S2dyn_s50": _EN},
                "Ee": {"S2dyn_m50": _ENe, "S2dyn_c12_m50": _ECe}},
     {"primary": "ADD_H_NO_EFFECT", "prop_add_h": "SUBPROPORTIONAL", "add": "ADD_NO_EFFECT",
      "prop_add": "SUBPROPORTIONAL", "drop_h": "DROP_H_NO_EFFECT", "prop_drop_h": "SUBPROPORTIONAL",
      "drop": "DROP_NO_EFFECT", "prop_drop": "SUBPROPORTIONAL", "m25": "M25_NO_EFFECT",
      "drop_early": "DROP_EARLY_NO_EFFECT", "prop_drop_early": "SUBPROPORTIONAL", "structure": "STRUCT_EQUAL"},
     {"phi_add_h": (0.018 / 0.0075 - 1) * BETA * 0.0075, "phi_drop": (1 - 0.0045 / 0.011) * BETA * 0.011}),
    ("add_h_not_manipulated", {"n": {"S2dyn_c12_a02": 0.0075}},
     {"primary": "NOT_MANIPULATED", "prop_add_h": "NOT_MANIPULATED", "add": "ADD_MOVES_E"}, {"dn_add_h": 0.0}),
    ("not_reproduced", {"E": {"S2dyn_10r": FLOOR}}, {"primary": "NOT_REPRODUCED", "felu": "NOT_REPRODUCED",
                                                     "ladder": "NEFF_TRACKS"}, {"R_none": 0.0}),
    ("not_reproduced_held", {"E": {"S2dyn_c12": FLOOR}}, {"primary": "NOT_REPRODUCED"}, {"R_c12": 0.0}),
    ("add_h_reversed", {"E": {"S2dyn_c12_a02": _EC - 0.05}}, {"primary": "ADD_H_REVERSED", "add": "ADD_MOVES_E"}, {}),
    ("add_h_borderline_main", {"E": {"S2dyn_c12_a02": _EC + _MID}},
     {"primary": "ADD_H_NO_EFFECT"}, {"dR_add_h": _MID}),
    ("add_h_supra", {"dE": {"S2dyn_c12_a02": 0.05}}, {"prop_add_h": "SUPRAPROPORTIONAL"}, {"phi_add_h": -0.05}),
    ("prop_margin_out", {"dE": {"S2dyn_c12_a02": -(0.03 - 0.5 * _W95)}},
     {"prop_add_h": "SUBPROPORTIONAL"}, {"phi_add_h": 0.03 - 0.5 * _W95}),
    ("prop_margin_in", {"dE": {"S2dyn_c12_a02": -(0.03 - 1.5 * _W95)}}, {"prop_add_h": "PROPORTIONAL"}, {}),
    ("drop_sub", {"E": {"S2dyn_m50": _EN - 0.01}}, {"drop": "DROP_MOVES_E", "prop_drop": "SUBPROPORTIONAL"},
     {"dR_drop": 0.01}),
    ("no_compensation", {"t": {"S2dyn_m50": 0.011, "S2dyn_c12_m50": 0.0075}},
     {"compensation": "NO_COMPENSATION", "compensation_h": "NO_COMPENSATION_H"}, {"comp": 0.0}),
    ("raw_falls", {"t": {"S2dyn_m50": 0.005}}, {"compensation": "RAW_FALLS"}, {}),
    ("struct_equal", {"E": {"S2dyn_s50": FLOOR + BETA * 0.0045}}, {"structure": "STRUCT_EQUAL"}, {"struct": 0.0}),
    ("step_worse", {"E": {"S2dyn_s50": FLOOR}}, {"structure": "STEP_WORSE"}, {}),
    ("felu_higher", {"dE": {"S2dyn_felu": 0.10}}, {"felu": "FELU_HIGHER"}, {"d_felu": 0.10}),
    ("felu_lower", {"dE": {"S2dyn_felu": -0.10}}, {"felu": "FELU_LOWER"}, {}),
    ("felu_not_manipulated", {"zero": {"S2dyn_felu": 0.95}}, {"felu": "NOT_MANIPULATED"}, {"dzero_felu": 0.0}),
    ("felu_margin_out", {"dE": {"S2dyn_felu": 0.03 - 0.5 * _W95}}, {"felu": "FELU_HIGHER"}, {}),
    ("felu_margin_in", {"dE": {"S2dyn_felu": 0.03 - 1.5 * _W95}}, {"felu": "FELU_SAME"}, {}),
    ("ladder_opposes", {"n": {"S2dyn_m25": 0.03, "S2dyn_m50": 0.02, "S2dyn_a02": 0.001},
                        "E": {"S2dyn_m25": FLOOR + 0.02, "S2dyn_m50": FLOOR + 0.06, "S2dyn_a02": FLOOR + 0.30}},
     {"ladder": "NEFF_OPPOSES", "primary": "ADD_H_MOVES_E"}, {}),
    ("early_not_manipulated", {"ne": {"S2dyn_m50": 0.008}},
     {"drop_early": "NOT_MANIPULATED", "drop": "DROP_MOVES_E"}, {"dn_drop_e": 0.0}),
    ("early_no_effect", {"Ee": {"S2dyn_m50": _ENe}},
     {"drop_early": "DROP_EARLY_NO_EFFECT", "prop_drop_early": "SUBPROPORTIONAL"}, {}),
    ("growth_none", {"E": {"S2dyn_c12": _EN, "S2dyn_c12_a02": _EN + BETA * 0.0105,
                           "S2dyn_c12_m50": _EN - BETA * 0.0042}},
     {"growth": "GROWTH_NO_REMAINDER", "primary": "ADD_H_MOVES_E"}, {"dR_growth": 0.0}),
    ("shadow_broken", {"break": _brk("S2dyn_c12_a02", "shadow_hash_match_natural", "False", k=2)},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("keep_off", {"break": _brk("S2dyn_m25", "keep_share", "0.5")}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("step_keep_off", {"break": _brk("S2dyn_s50", "keep_share", "1.0", k=2)}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("top_broken", {"break": _brk("S2dyn_c12_a02", "field0_top_ok", "False")}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("lift_share_off", {"break": _brk("S2dyn_a02", "lift_share", "0.05")}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("hold_violation", {"break": _brk("S2dyn_c12_a02", "exc_w2", "0.01", k=2)}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("hold_idle", {"break": tuple(x for c in ("rows_par", "rows_perp", "rows_w2") for x in _brk("S2dyn_c12_m50", c, "0"))},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("repro_contradicts", {"break": _brk("S2dyn_c12", "hash_match_escape", "False", k=2)},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("repro_recorded_ok", {"break": _brk("S2dyn_c12", "hash_match_escape", "True", k=2)},
     {"primary": "ADD_H_MOVES_E"}, {"n_valid": 10}),
    ("logits_mb", {"break": _brk("N2r_m50", "logits_equal_mb", "False")}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("field_fixed", {"break": _brk("S2dyn_felu", "field0_equal_fixed", "False")}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("early_nonfinite", {"break": _brk("S2dyn_m50", "early_acc", "", k=2)}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("prefix_contradicts", {"prefix_break": tuple((s, "hash_match_resp_ee", "False") for s in (30, 31, 32))},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("prefix_recorded_ok", {"prefix_break": tuple((s, "hash_match_resp_ee", "True") for s in (30, 31, 32))},
     {"primary": "ADD_H_MOVES_E"}, {"n_valid": 10}),
    ("paired_matters", {"seed_sd": 1.0}, {"primary": "ADD_H_MOVES_E", "prop_add_h": "PROPORTIONAL",
                                          "prop_drop": "PROPORTIONAL"}, {"dR_add_h": BETA * 0.0105}),
    ("kappa_measured", {"n_noise": {"S2dyn_c12_a02": 0.005}},
     {"prop_add_h": "PROPORTION_UNRESOLVED", "primary": "ADD_H_MOVES_E"}, {"phi_add_h": 0.0}),
]


def _pick(res, key):
    if key == "n_valid":
        return len(res["valid_seeds"])
    return res["stats"][(key, 1, 0.95)]["mean"]


def s8(V) -> dict:
    """verdict.analyze on the synthetic scenarios over seeds 30-39: labels as designed, estimates within
    TOL_EST, the imported t quantile on the published table within 1e-6, SEEDS, MARGIN and the arm list as
    registered."""
    failed, per = [], {}
    failed += [f"t({p},{df})" for (p, df), w in T_TABLE.items() if abs(V.t_quantile(p, df) - w) > 1e-6]
    if tuple(V.SEEDS) != SEEDS_REG:
        failed.append(f"SEEDS {V.SEEDS}")
    if V.MARGIN != 0.03:
        failed.append(f"MARGIN {V.MARGIN}")
    if tuple(V.DYN) != DYN_S8:
        failed.append("DYN")
    for name, scen, want_l, want_v in S8_SCENARIOS:
        res = V.analyze(synth(V, scen))
        bad = [f"{k}: got {res['labels'].get(k)} want {w}" for k, w in want_l.items() if res["labels"].get(k) != w]
        for k, w in want_v.items():
            g = _pick(res, k)
            if not (g is not None and math.isfinite(g) and abs(g - w) <= TOL_EST):
                bad.append(f"{k}: got {g} want {w}")
        per[name] = {"labels": res["labels"], "bad": bad}
        failed += [f"{name}|{b}" for b in bad]
    return {"pass": not failed, "failed_items": failed, "detail": per}


V_DR = "        dR = sign * (R[arm] - R[base])\n"
V_PRIMARY = '        "primary": direction("add_h", MAIN_LEVEL, "ADD_H"),\n'
V_MANIP = '    manip = {tag: A_ok and lo(f"dn_{tag}") > 0 for tag in ("add_h", "add", "drop_h", "drop", "m25", "drop_e", "drop_h_e",\n'
V_KAP = "            kap = n[arm] / n[base]\n"
V_MARGIN = "MARGIN = 0.03 "
V_LEVEL = "MAIN_LEVEL = 0.975 "
V_STRUCT = '        q[("struct", k)] = e("S2dyn_s50", k) - e("S2dyn_m50", k)\n'
V_COMP = '        q[("comp", k)] = v("T", "S2dyn_m50", k) - v("T", "S2dyn_10r", k)\n'
V_FELU = '        felu = ("FELU_SAME" if (-MARGIN < x["lo"] and x["hi"] < MARGIN) else\n'
V_LADDER = '    rho = [spearman([cols["N"][(s, a, 1)] for a in LADDER], [cols["E"][(s, a, 1)] for a in LADDER]) for s in valid]\n'
V_SHADOW = '            if not _true(rows[(a, k)].get("shadow_hash_match_natural")):\n'
V_KEEPTOL = "KEEP_TOL = 0.01 "
V_TOP = '        if not (_true(r.get("field0_top_ok")) and abs(_f(r.get("lift_share")) - r_) <= LIFT_TOL):\n'
V_TOL = "TOL_CAP = 1e-4 "
V_RECORD = '                if not _ok_record(rows[(a, k)].get(f"hash_match_{src}", "")):\n'
V_SEEDS = "SEEDS = tuple(range(30, 40))\n"
V_FLOOR = '        F_ = e("S2u30r", k)\n'
V_E = '    def e(a, k=1):\n        return v("E", a, k)\n'
V_FE = '    Fe = v("Ee", "S2u30r")\n'
V_MB = '        if not (_true(rows[(a, 1)].get("logits_equal_full")) and _true(rows[(a, 1)].get("logits_equal_mb"))):\n'
V_B = '    B_ok = A_ok and st[("R_none", 1, COMP_LEVEL)]["lo"] > 0 and st[("R_c12", 1, COMP_LEVEL)]["lo"] > 0\n'
V_NE = '    ne = {a: v("Ne", a) for a in DYN}\n'
S8_MUT = [
    ("M8a: every change's sign reversed", [(V_DR, "        dR = -sign * (R[arm] - R[base])\n")]),
    ("M8b: the primary read on the free host", [(V_PRIMARY, '        "primary": direction("add", MAIN_LEVEL, "ADD_H"),\n')]),
    ("M8c: the manipulation checks dropped", [(V_MANIP, '    manip = {tag: A_ok for tag in ("add_h", "add", "drop_h", "drop", "m25", "drop_e", "drop_h_e",\n')]),
    ("M8d: kappa taken as a constant", [(V_KAP, "            kap = np.full_like(n[base], 2.0)\n")]),
    ("M8e: margin 0.3", [(V_MARGIN, "MARGIN = 0.3 ")]),
    ("M8f: main level 95%", [(V_LEVEL, "MAIN_LEVEL = 0.95 ")]),
    ("M8g: structure reversed", [(V_STRUCT, '        q[("struct", k)] = e("S2dyn_m50", k) - e("S2dyn_s50", k)\n')]),
    ("M8h: compensation read on the masked count", [(V_COMP, '        q[("comp", k)] = v("N", "S2dyn_m50", k) - v("N", "S2dyn_10r", k)\n')]),
    ("M8i: F.elu same on any interval containing 0", [(V_FELU, '        felu = ("FELU_SAME" if (x["lo"] <= 0 <= x["hi"]) else\n')]),
    ("M8j: the ladder read on E only", [(V_LADDER, '    rho = [spearman([cols["E"][(s, a, 1)] for a in LADDER], [cols["E"][(s, a, 1)] for a in LADDER]) for s in valid]\n')]),
    ("M8k: shadows not checked", [(V_SHADOW, "            if False:\n")]),
    ("M8l: keep tolerance 0.6", [(V_KEEPTOL, "KEEP_TOL = 0.6 ")]),
    ("M8m: the lift not checked", [(V_TOP, "        if False:\n")]),
    ("M8n: hold tolerance 0.1", [(V_TOL, "TOL_CAP = 1e-1 ")]),
    ("M8o: a contradicted record is valid", [(V_RECORD, "                if False:\n")]),
    ("M8p: seeds 10-19", [(V_SEEDS, "SEEDS = tuple(range(10, 20))\n")]),
    ("M8q: the floor read off N2r_m50", [(V_FLOOR, '        F_ = e("N2r_m50", k)\n')]),
    ("M8r: pairing broken", [(V_E, '    def e(a, k=1):\n        x = v("E", a, k)\n        return x[::-1] if a == "S2u30r" else x\n')]),
    ("M8s: the early floor read on the whole task", [(V_FE, '    Fe = v("E", "S2u30r")\n')]),
    ("M8t: the minibatch logits not checked", [(V_MB, '        if not _true(rows[(a, 1)].get("logits_equal_full")):\n')]),
    ("M8u: the held remainder not required", [(V_B, '    B_ok = A_ok and st[("R_none", 1, COMP_LEVEL)]["lo"] > 0\n')]),
    ("M8v: the early window's n read on the whole task", [(V_NE, '    ne = {a: v("N", a) for a in DYN}\n')]),
]


# --------------------------------------------------------------------------
# S9: the CLI
# --------------------------------------------------------------------------

E_FLUSH = "    torch.set_flush_denormal(True)\n"
E_SPEC = 'SPEC = "specs/spec_neffdir_ee_0918.md"\n'
E_EXP = 'EXPERIMENT = "neffdir_ee_0918"\n'
E_SEEDS = "SEEDS = tuple(range(30, 40))\n"
E_SALT = "MASK_SALT = 918_001 "


def s9(M) -> dict:
    """main() with 2 epochs and one arm on seed 45 writes provenance naming this experiment, its spec and
    registration field, flush on, the salts and the two tables' sha256 (recomputed here), and hashes of this
    runner and the imported runners; the registered seeds are 30-39, the arm table has the 12 registered
    arms and the 3 check arms, the salts are 918001 / 918002 and the early window is 1500 updates."""
    out = SCR / "s9_cli"
    torch.set_flush_denormal(False)
    try:
        M.main(["--seed", str(DEV_SEED), "--epochs", "2", "--arms", "N2r_m50", "--out", str(out)])
        flushed = _flush_is_on()
    finally:
        torch.set_flush_denormal(True)
    pv = json.loads((out / "provenance.json").read_text())
    g = torch.Generator(device="cpu")
    g.manual_seed(918_001 + DEV_SEED)
    U = torch.rand(RE.N_IMG, 100, generator=g)
    per = {"experiment": pv["experiment"] == "neffdir_ee_0918",
           "spec": pv["spec"] == "specs/spec_neffdir_ee_0918.md" and "prereg_commit" in pv,
           "flush": pv["flush_denormal"] is True and flushed,
           "tables": pv["tables_sha256"]["U"] == hashlib.sha256(U.numpy().tobytes()).hexdigest(),
           "hashes": {"src/neffdir_ee_0918.py", "src/escape_ee_0917.py", "src/swap_ee_0917.py",
                      "src/respdyn_ee_0917.py", "src/resp_ee_0917.py"} <= set(pv["code_sha256"]),
           "constants": (M.SEEDS == tuple(range(30, 40)) and len(M.ARMS) == 12 and len(M.CHECK_ARMS) == 3
                         and M.MASK_SALT == 918_001 and M.STEP_SALT == 918_002 and M.EARLY == 1500)}
    failed = [k for k, v in per.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S9_MUT = [
    ("M9a: flush off", [(E_FLUSH, "    torch.set_flush_denormal(False)\n")]),
    ("M9b: another spec", [(E_SPEC, 'SPEC = "specs/spec_escneff_ee_0917.md"\n')]),
    ("M9c: another experiment name", [(E_EXP, 'EXPERIMENT = "escneff_ee_0917"\n')]),
    ("M9d: seeds 10-19", [(E_SEEDS, "SEEDS = tuple(range(10, 20))\n")]),
    ("M9e: another mask salt", [(E_SALT, "MASK_SALT = 918_011 ")]),
    ("M9f: another early window", [("EARLY = 1500 ", "EARLY = 750 ")]),
]


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

COST_ARMS = ("N2r", "S2dyn_10r", "S2dyn_m50")


def s_cost() -> None:
    """PROBE processes at once on seeds 40-42 (outside the registered 30-39), each the full prefix and three
    arms (one plain, two with a shadow).  One full seed = prefix + 3 plain arms + 9 shadow arms.  Gate: all
    exit 0 and the memory budget (80% of MemAvailable) leaves >= PROBE slots."""
    t0 = time.time()
    avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0] / 2 ** 20
    SCR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(PROBE):
        d = SCR / f"cost_{i}"
        procs.append((d, subprocess.Popen([PY, str(RUNNER), "--seed", str(40 + i), "--arms", ",".join(COST_ARMS),
                                           "--out", str(d)], env=THREAD_ENV, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE)))
    rc, peak, pref, plain, dyn = [], [], [], [], []
    for d, pr in procs:
        err = pr.communicate()[1]
        rc.append(pr.returncode)
        if pr.returncode != 0:
            print(err.decode()[-400:], flush=True)
            continue
        pv = json.loads((d / "provenance.json").read_text())
        peak.append(pv["peak_rss_kb"] / 2 ** 20)
        sec = {}
        for r in csv.DictReader((d / "arms.csv").open()):
            sec[r["arm"]] = sec.get(r["arm"], 0.0) + float(r["sec"])
        pref.append(pv["seconds_total"] - sum(sec.values()))
        plain.append(sec["N2r"])
        dyn.append((sec["S2dyn_10r"] + sec["S2dyn_m50"]) / 2)
    slots = int(avail * 0.8 / max(peak)) if peak else 0
    one = (max(pref) + 3 * max(plain) + 9 * max(dyn)) if peak else float("nan")
    ok = all(r == 0 for r in rc) and slots >= PROBE
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": f"all {PROBE} exit 0 and slots >= {PROBE}", "returncodes": rc,
                         "mem_available_gib": avail, "peak_rss_gib": max(peak) if peak else None, "slots": slots,
                         "prefix_and_overhead_s": max(pref) if pref else None,
                         "plain_arm_s": max(plain) if plain else None, "dyn_arm_s": max(dyn) if dyn else None,
                         "one_seed_min": one / 60, "grid_10_wall_min_at_probe": 10 * one / 60 / PROBE,
                         "measured_at": dt.datetime.now().astimezone().isoformat(),
                         "seconds": round(time.time() - t0, 1)}
    dump()
    c = RESULTS["S-cost"]
    print(f"S-cost: pass={ok}  one seed {c['one_seed_min']:.1f} min, 10 seeds at {PROBE}: "
          f"{c['grid_10_wall_min_at_probe']:.0f} min (peak {c['peak_rss_gib']:.2f} GiB, slots {slots})", flush=True)


# --------------------------------------------------------------------------

CHECKS = {
    "S1a_S-records": ("S1a identity-carrier arms reproduce respdyn / escape / resp_ee", s1a, S1A_MUT, s1a.__doc__, RUNNER),
    "S1b_S-carriers": ("S1b the carriers do what they say", s1b, S1B_MUT, s1b.__doc__, RUNNER),
    "S1c_S-derivatives": ("S1c the derivative factors on a grid", s1c, S1C_MUT, s1c.__doc__, RUNNER),
    "S8_S-verdict": ("S8 the verdict on synthetic seeds 30-39", s8, S8_MUT, s8.__doc__, VERDICT),
    "S9_S-cli": ("S9 the CLI's provenance and constants", s9, S9_MUT, s9.__doc__, RUNNER),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    global DUMP_PATH
    keys = [k for k in CHECKS if not args.only or any(k.startswith(p) for p in args.only.split(","))]
    if args.only:
        SCR.mkdir(parents=True, exist_ok=True)
        DUMP_PATH = SCR / "checks_partial.json"
    RESULTS.update({"run_id": "neffdir_ee_0918", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "spec": "specs/spec_neffdir_ee_0918.md", "prereg_commit": load(RUNNER).PREREG_COMMIT,
                    "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (RUNNER, VERDICT, Path(__file__))},
                    "torch": torch.__version__})
    for k in keys:
        title, fn, mut, deriv, target = CHECKS[k]
        run_check(k, title, fn, mut, deriv, target)
    if not args.only or "cost" in args.only:
        s_cost()
    RESULTS["finished_at"] = dt.datetime.now().astimezone().isoformat()
    dump()
    print(f"all_pass = {RESULTS['all_pass']}  -> {DUMP_PATH}")


if __name__ == "__main__":
    main()

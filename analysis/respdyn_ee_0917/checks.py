#!/usr/bin/env python3
"""Checks for respdyn_ee_0917 (specs/spec_respdyn_ee_0917.md section 5).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/respdyn_ee_0917/checks.py
    ... --only S2,St0          # development (writes results/_checks_respdyn_ee_0917/checks_partial.json)

Each check runs on the real code and then on every listed mutation (an exact-once substitution in the
check's target file); every mutation must make the same check fail.  Tolerances come from the
arithmetic (docstrings), never from looking at values.  Writes results/respdyn_ee_0917/checks.json.

S0   machine-bound link to resp_ee_0917 (seeds 0, 1; full length): the prefix's state sha256 equals
     resp_ee's committed prefix for all 22 tasks, and N2, N2r, N10, N10r, S2_10r, R2_10, S2u30r run
     through this runner equal resp_ee's committed arms (state sha256, both tasks).  Keeps seed 0's
     full-length branch states for S3, S4, S5, S-t0.
S1b  Stepper = RE.train_task: all 26 resp_ee arms through this runner equal RE.run_arm (short tasks)
S2   continuation, all 32 arms (short tasks): natural arms and the dyn arms' shadows reproduce the prefix;
     the branch states are untouched; anchored arms keep the branch-point logits, the unanchored one
     does not; moving arms leave their fixed twins; the ramp ends tasks 1 and 2 at -20 and -30
S-t0 the moving field starts where resp_ee's fixed field is (bit for bit, full-length states); with the
     shadow frozen every dyn arm is its fixed twin, and a flat ramp is S2u10r; the field an update
     trains with is the shadow's after exactly that many updates (independent RE.train_task replay)
S3   branch point (full-length states): new anchored arms keep the logits and start on the source's
     argument; the unanchored arm's features are the source's features within the rounding bound
S4   the unanchored arm's gradient = an independent float64 chain rule through phi(z + d)
S5   fresh Adam in the new reset arms: first update = lr g / (|g| + eps)
S6   the verdict on synthetic shards with known structure; the t quantile; the arm list
S7   the CLI writes what the in-process run writes, twice the same, with this experiment's provenance
S8   the side runner: float32 + host ELU is RE.run_prefix bit for bit; the variants' dtypes; the
     variants' training derivative (F.elu: exp(z) > 0 down to the flush boundary)
S-cost  processes at once (peak RSS, seconds per update) -- recorded, not a gate
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
from src import pmnist_0905 as H                 # noqa: E402
from src import pmnist_rlmnist_0906 as RL        # noqa: E402
from src import mucap_el_run_0916 as EL          # noqa: E402
from src import resp_ee_0917 as RE               # noqa: E402

RUNNER = REPO / "src" / "respdyn_ee_0917.py"
NATRUN = REPO / "src" / "respdyn_nat_0917.py"
VERDICT = REPO / "analysis" / "respdyn_ee_0917" / "verdict.py"
OUT = REPO / "results" / "respdyn_ee_0917"
SCR = REPO / "results" / "_checks_respdyn_ee_0917"
RESP_EE_RUNS = REPO / "results" / "resp_ee_0917" / "runs"
EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)
DEV = H.setup("cpu")
MNIST = H.Mnist(DEV)
PY = sys.executable
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
PROBE = 4
RESULTS: dict = {}
DUMP_PATH = OUT / "checks.json"
CACHE: dict = {}
EP = 2                                           # short tasks (150 updates)

# --------------------------------------------------------------------------
# infrastructure (resp_ee_0917's)
# --------------------------------------------------------------------------

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
    mod.__dict__["_source_text"] = src
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def brief(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "pass" and not isinstance(v, (dict, list))} | \
        {"failed_items": r.get("failed_items", [])[:8]}


def dump() -> None:
    checks = {k: v for k, v in RESULTS.items() if k.startswith("S") and isinstance(v, dict) and "pass" in v
              and k != "S-cost"}
    RESULTS["all_pass"] = bool(checks) and all(
        v.get("pass") and v.get("all_mutations_detected", True) for v in checks.values())
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMP_PATH.write_text(json.dumps(RESULTS, indent=2, default=str))


def run_check(key: str, title: str, fn, mutations: list, derivation: str, target: Path) -> None:
    t0 = time.time()
    real = load(target)
    real._is_base = True
    base = fn(real)
    entry = {"title": title, "target": str(target.relative_to(REPO)), "threshold_derivation": derivation,
             **base, "mutations": []}
    for label, subs in mutations:
        try:
            r = fn(load(target, subs))
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": bool(r["pass"]),
                                       "detected": not r["pass"], "mutant_result": brief(r)})
        except AssertionError as e:
            if "mutation anchor" in str(e):
                raise
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
        except Exception as e:
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
    entry["all_mutations_detected"] = all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:3]) if not entry['pass'] else ''}", flush=True)


class flush_denormal:
    def __enter__(self):
        torch.set_flush_denormal(True)

    def __exit__(self, *a):
        torch.set_flush_denormal(False)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _gen(state):
    g = torch.Generator(device="cpu")
    g.set_state(state.clone())
    return g


def _state_digest(cks: dict) -> str:
    h = hashlib.sha256()
    for t in sorted(cks):
        c = cks[t]
        for q in (*c["params"], *c["m"], *c["v"], c["z1"], c["z2"], c["g_lab"], c["g_batch"]):
            h.update(q.detach().numpy().tobytes())
        h.update(str(c["tc"]).encode())
    return h.hexdigest()


def _rec(seed: int, name: str) -> pd.DataFrame:
    return pd.read_csv(RESP_EE_RUNS / f"s{seed}" / name, float_precision="round_trip")


def _short_prefix():
    """Seed 0, EP-epoch tasks, all 22 (RE's code: no mutation in this file reaches it)."""
    if "short" not in CACHE:
        with flush_denormal():
            prows, _, cks, x, _ = RE.run_prefix(0, MNIST, EP, RE.PREFIX_T, save_at=RE.BRANCH_T)
        CACHE["short"] = (prows, cks, x)
    prows, cks, x = CACHE["short"]
    return prows, cks, x


def _full_states():
    """Seed 0's full-length prefix (S0 keeps it; otherwise computed here with RE.run_prefix)."""
    if "full0" not in CACHE:
        with flush_denormal():
            prows, _, cks, x, _ = RE.run_prefix(0, MNIST, RE.EPOCHS, RE.PREFIX_T, save_at=RE.BRANCH_T)
        CACHE["full0"] = (prows, cks, x)
    return CACHE["full0"]


# --------------------------------------------------------------------------
# S0: resp_ee_0917's run, re-made on this machine
# --------------------------------------------------------------------------

S0_SEEDS = (0, 1)
S0_ARMS = ("N10r", "S2_10r", "R2_10", "S2u30r", "N2r", "N2", "N10")


def s0(R) -> dict:
    """Exact equality of the state sha256 (parameters, both Adam moments, step count) at every task end
    of the prefix and at both continuation task ends of each listed arm, against resp_ee_0917's
    committed records.  The prefix is RE.run_prefix (not reachable by a mutation of this runner), so
    it is computed once per seed and reused; the arms go through this runner.  An arm stops the seed at
    its first mismatch."""
    failed, secs, compared = [], {}, 0
    for seed in S0_SEEDS:
        key = f"full{seed}"
        t0 = time.time()
        if key not in CACHE:
            with flush_denormal():
                prows, _, cks, x, _ = RE.run_prefix(seed, MNIST, RE.EPOCHS, RE.PREFIX_T, save_at=RE.BRANCH_T)
            CACHE[key] = (prows, cks, x)
        prows, cks, x = CACHE[key]
        rp = _rec(seed, "prefix.csv").set_index("task")["state_sha256"]
        for r in prows:
            compared += 1
            if r["state_sha256"] != rp.loc[r["task"]]:
                failed.append(f"seed {seed} prefix task {r['task']}")
        ra = _rec(seed, "arms.csv")
        for a in S0_ARMS:
            with flush_denormal():
                rows = R.run_arm(a, R.ARMS[a], cks, x, prows, seed, epochs=R.EPOCHS)
            bad = False
            for row in rows:
                compared += 1
                want = ra[(ra["arm"] == a) & (ra["k"] == row["k"])]["state_sha256"].iloc[0]
                if row["state_sha256"] != want or not row["hash_match_resp_ee"]:
                    failed.append(f"seed {seed} arm {a} k{row['k']}")
                    bad = True
            if bad:
                break
        secs[seed] = time.time() - t0
    return {"pass": not failed, "compared": compared, "failed_items": failed, "seconds_per_seed": secs,
            "machine": os.uname().nodename, "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "record_sha256": {s: _sha(RESP_EE_RUNS / f"s{s}" / "arms.csv") for s in S0_SEEDS}}


# --------------------------------------------------------------------------
# S1b: the Stepper is RE.train_task
# --------------------------------------------------------------------------

S1_FIELDS = ("state_sha256", "online_acc", "online_ce", "first75_acc", "last750_acc", "memo_acc_end",
             "full_ce_end", "step_w1", "step_w2", "step_w3", "logits_equal_full", "logits_equal_mb",
             "gtr2_start", "neffT2_end", "pf_eps_w2")


def s1b(R) -> dict:
    """Bit equality, all 26 resp_ee arms, both tasks, every listed field (short tasks, seed 0)."""
    failed = []
    prows, cks, x = _short_prefix()
    with flush_denormal():
        for a in RE.ARMS:
            old = RE.run_arm(a, RE.ARMS[a], cks, x, prows, 0, epochs=EP)
            new = R.run_arm(a, R.ARMS[a], cks, x, prows, 0, epochs=EP)
            for o, n in zip(old, new):
                for f in S1_FIELDS:
                    if f in o and o[f] != n[f] and not (isinstance(o[f], float) and math.isnan(o[f]) and math.isnan(n[f])):
                        failed.append(f"{a} k{o['k']} {f}")
                        break
    return {"pass": not failed, "failed_items": failed, "n_arms": len(RE.ARMS)}


# --------------------------------------------------------------------------
# S2: continuation, all arms
# --------------------------------------------------------------------------

def s2(R) -> dict:
    """Natural arms: state sha256 after continuation task k == the prefix's at branch + k.  Dyn arms: the
    shadow's state sha256 == the prefix's at src + k.  The saved branch states are untouched by all 32
    arms.  Anchored arms: branch-point logits equal the natural net's (all images, first minibatch); the
    unanchored arm's do not (non-vacuity).  Moving arms start on their fixed field (field0_equal_fixed)
    and end task 1 in a different state from their fixed twin; the ramp's field mean is -10, -20, -30
    at the start, the end of task 1 and the end of task 2 (float32 constants, exact)."""
    failed = []
    prows, cks, x = _short_prefix()
    before = _state_digest(cks)
    rows = []
    with flush_denormal():
        for a in R.ARMS:
            rows += R.run_arm(a, R.ARMS[a], cks, x, prows, 0, epochs=EP)
    if _state_digest(cks) != before:
        failed.append("branch states modified by the arms")
    df = pd.DataFrame(rows)
    for a, arm in R.ARMS.items():
        r = df[df["arm"] == a]
        if len(r) != R.CONT:
            failed.append(f"{a}: {len(r)} rows")
            continue
        r1 = r[r["k"] == 1].iloc[0]
        anchored = arm["kind"] != "noanchor"
        eq = bool(r1["logits_equal_full"]) and bool(r1["logits_equal_mb"])
        if anchored and not eq:
            failed.append(f"{a}: branch-point logits differ")
        if not anchored and (bool(r1["logits_equal_full"]) or bool(r1["logits_equal_mb"])):
            failed.append(f"{a}: unanchored arm keeps the natural logits (vacuous)")
        if arm["kind"] is None and not arm["reset"] and not all(r["hash_match_prefix"]):
            failed.append(f"{a}: natural continuation does not reproduce the prefix")
        if arm["kind"] == "dyn" and not all(r["shadow_hash_match_prefix"]):
            failed.append(f"{a}: shadow is not N{arm['src']}")
        if arm["kind"] in ("dyn", "ramp") and not bool(r1["field0_equal_fixed"]):
            failed.append(f"{a}: field does not start at the fixed field")
        if a in R.FIXED_TWIN and arm["kind"] != "noanchor":
            tw = df[(df["arm"] == R.FIXED_TWIN[a]) & (df["k"] == 1)].iloc[0]
            if tw["state_sha256"] == r1["state_sha256"]:
                failed.append(f"{a}: identical to {R.FIXED_TWIN[a]} (the field did not move)")
        if arm["kind"] == "ramp":
            want = [(-10.0, -20.0), (-20.0, -30.0)]
            got = [(float(q["probe_dmean2_first"]), float(q["probe_dmean2_last"])) for _, q in r.sort_values("k").iterrows()]
            if got != want:
                failed.append(f"{a}: ramp field means {got} != {want}")
    return {"pass": not failed, "failed_items": failed, "n_arms": len(R.ARMS), "n_rows": len(df)}


# --------------------------------------------------------------------------
# S-t0: where the moving field starts, what freezing it gives, and its timing
# --------------------------------------------------------------------------

DYN = ("S2dyn_10r", "R2dyn_10", "S2dyn_5r", "S2dyn_20r")
T0_EPOCHS = 11                                   # 825 updates: probes at 0, 75, 750 exist


def st0(R) -> dict:
    """(a) Full-length states: after build + refresh(0), each dyn arm's field equals the fixed field
    RE.build_shift makes (torch.equal), and the shadow's full-batch z2 equals the stored source field;
    the ramp's field equals S2u10r's.
    (b) With the shadow frozen, each dyn arm reproduces its fixed twin (state sha256, both tasks); with
    RAMP = (10, 10) the ramp reproduces S2u10r (short tasks on the full-length states).
    (c) S2dyn_10r at T0_EPOCHS epochs: the field the arm's forward reads at update s (captured by the
    Stepper hook) equals float32(float64(z2) - float64(z2_branch)) with z2 from EL.forward2 on the
    source state advanced by RE.train_task for exactly s updates (s = 0, 75, 750) -- an independent
    replay; exact equality (same ops on the same bits)."""
    failed = []
    prows, cks, x = _full_states()
    for a in DYN + ("S2ramp_r",):
        arm = R.ARMS[a]
        sh, fwd, mover = R.build(arm, cks, x, R.SPE * R.EPOCHS)
        mover.begin_task(R.SPE * R.EPOCHS)
        mover.refresh(0)
        if not torch.equal(sh[2], R.fixed_field(arm, cks)):
            failed.append(f"{a}: field(0) != fixed field")
        if arm["kind"] == "dyn":
            with torch.no_grad():
                z_src = EL.forward2(cks[arm["src"]]["params"], x, R.ACT, R.ACT)[2]
            if not torch.equal(z_src, cks[arm["src"]]["z2"]):
                failed.append(f"{a}: the source's z2 is not the stored field")
            if not torch.equal(RE.build_shift({**arm, "kind": "field"}, cks)[2], sh[2]):
                failed.append(f"{a}: field(0) != resp_ee's field arm")
        else:
            if not torch.equal(sh[2], torch.full_like(cks[2]["z2"], -10.0)):
                failed.append("ramp: field(0) != S2u10r's")
    with flush_denormal():
        for a in DYN:
            fr = R.run_arm(a, R.ARMS[a], cks, x, prows, 0, epochs=EP, freeze_shadow=True)
            tw = R.run_arm(R.FIXED_TWIN[a], R.ARMS[R.FIXED_TWIN[a]], cks, x, prows, 0, epochs=EP)
            if [r["state_sha256"] for r in fr] != [r["state_sha256"] for r in tw]:
                failed.append(f"{a}: frozen shadow != {R.FIXED_TWIN[a]}")
        saved = R.RAMP
        try:
            R.RAMP = (10.0, 10.0)
            fr = R.run_arm("S2ramp_r", R.ARMS["S2ramp_r"], cks, x, prows, 0, epochs=EP)
        finally:
            R.RAMP = saved
        tw = R.run_arm("S2u10r", R.ARMS["S2u10r"], cks, x, prows, 0, epochs=EP)
        if [r["state_sha256"] for r in fr] != [r["state_sha256"] for r in tw]:
            failed.append("flat ramp != S2u10r")
        # (c) timing
        want_s = (0, 75, 750)
        got = {}

        def hook(k, s, sh):
            if k == 1 and s in want_s:
                got[s] = sh[2].clone()
        R.run_arm("S2dyn_10r", R.ARMS["S2dyn_10r"], cks, x, prows, 0, epochs=T0_EPOCHS, hook=hook)
        src, br = cks[10], cks[2]
        params = [p.detach().clone().requires_grad_(True) for p in src["params"]]
        adam = ([q.clone() for q in src["m"]], [q.clone() for q in src["v"]], [src["tc"]])
        y = RL.task_labels(_gen(src["g_lab"]))
        gb = _gen(src["g_batch"])
        done = 0
        for s in want_s:
            if s > done:
                RE.train_task(params, adam, x, y, gb, (s - done) // RE.SPE)
                done = s
            with torch.no_grad():
                z2 = EL.forward2(params, x, R.ACT, R.ACT)[2]
            want = (z2.double() - br["z2"].double()).float()
            if s not in got or not torch.equal(got[s], want):
                failed.append(f"field at update {s} is not the shadow's after {s} updates")
        moved = 0 in got and 750 in got and not torch.equal(got[0], got[750])
        if not moved:
            failed.append("the field did not move between updates 0 and 750 (vacuous)")
    return {"pass": not failed, "failed_items": failed}


# --------------------------------------------------------------------------
# S3-S5 on seed 0's full-length states
# --------------------------------------------------------------------------

def _first_minibatch(ck, x):
    y = RL.task_labels(_gen(ck["g_lab"]))
    order = torch.randperm(RL.N_IMAGES, generator=_gen(ck["g_batch"]))
    ob = order[:RL.BATCH]
    return x[ob], y[ob], ob, y


def s3(R) -> dict:
    """(a) New anchored arms (dyn, ramp) at the branch point: logits on all 1200 images and on the first
    minibatch equal the natural net's (torch.equal); the argument z_b + d(0) sits on the source field
    (dyn: within eps32 |d| + 4 eps64 (|z_b| + |d|), resp_ee's S3 bound) or on z_b - 10 (ramp).
    (b) The unanchored arm: its full-batch z2 is the stored branch field, and its float32 features
    phi(fl32(z_b + d)) equal the source's phi(z_src) within
        |fl32(z_b + d) - (z_b + d)| + |z_b + d - z_src| + rounding of the two expm1 evaluations
        <= eps32/2 |z_b + d| + bound + eps32 (|phi(u)| + |phi(z_src)|)
    (ELU is 1-Lipschitz; expm1 is accurate to one ulp), bounded by 2 bound + eps32 (|z_src| + |a| + |a_src|)
    with |z_b + d| <= |z_src| + bound; and its logits differ from the natural net's (non-vacuity)."""
    failed, worst = [], {}
    prows, cks, x = _full_states()
    for a in DYN + ("S2ramp_r", "R2_10_noanchor"):
        arm = R.ARMS[a]
        ck = cks[arm["branch"]]
        params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
        sh, fwd, mover = R.build(arm, cks, x, R.SPE * R.EPOCHS)
        if mover is not None:
            mover.begin_task(R.SPE * R.EPOCHS)
            mover.refresh(0)
        with torch.no_grad():
            nat = EL.forward2(params, x, R.ACT, R.ACT)
            out = fwd(params, x, torch.arange(x.shape[0]))
            xb, yb, ob, _ = _first_minibatch(ck, x)
            eq_full = torch.equal(nat[4], out[4])
            eq_mb = torch.equal(EL.forward2(params, xb, R.ACT, R.ACT)[4], fwd(params, xb, ob)[4])
            zb = out[2].double()
            if not torch.equal(out[2], ck["z2"]):
                failed.append(f"{a}: full-batch z2 is not the stored branch field")
            d = sh[2].double()
            bound = EPS32 * d.abs() + 4 * EPS64 * (zb.abs() + d.abs())
            if arm["kind"] == "ramp":
                target = ck["z2"].double() - 10.0
            else:
                target = cks[arm["src"]]["z2"].double()
            err = (zb + d - target).abs()
            ratio = float((err / bound.clamp(min=1e-300)).max())
            if (err > bound).any():
                failed.append(f"{a}: argument off its target (max err/bound {ratio:.3g})")
            item = {"z_err_over_bound": ratio}
            if arm["kind"] == "noanchor":
                if eq_full or eq_mb:
                    failed.append(f"{a}: logits equal the natural net's (vacuous)")
                a_src = RE.ACT.phi(cks[arm["src"]]["z2"]).double()
                ferr = (out[3].double() - a_src).abs()
                fbound = 2 * bound + EPS32 * (target.abs() + out[3].double().abs() + a_src.abs())
                item["feature_err_over_bound"] = float((ferr / fbound.clamp(min=1e-300)).max())
                if (ferr > fbound).any():
                    failed.append(f"{a}: features are not the source's (max {float(ferr.max()):.3g})")
            elif not (eq_full and eq_mb):
                failed.append(f"{a}: branch-point logits differ")
            worst[a] = item
    return {"pass": not failed, "failed_items": failed, "worst": worst}


def TRAIN_DPHI(u):
    """What autograd applies through the host ELU: expm1(u) + 1 on the negative side (resp_ee S4b)."""
    return torch.where(u > 0, torch.ones_like(u), torch.expm1(torch.clamp(u, max=0.0)) + 1.0)


def _elu(u):
    return torch.where(u > 0, u, torch.expm1(u.clamp(max=0)))


def _manual_noanchor64(p64, xb, yb, d2):
    """Independent float64 backprop of a2 = phi(z2 + d2) (value and derivative at the shifted argument),
    with its absolute-value scale for the rounding bound."""
    W1, b1, W2, b2, W3, b3 = p64
    z1 = xb @ W1.T + b1
    a1 = _elu(z1)
    z2 = a1 @ W2.T + b2
    a2 = _elu(z2 + d2)
    logits = a2 @ W3.T + b3
    g1, g2 = TRAIN_DPHI(z1), TRAIN_DPHI(z2 + d2)
    B = xb.shape[0]
    p = torch.softmax(logits, 1)
    e = (p - F.one_hot(yb, 10).double()) / B
    ea = (p + F.one_hot(yb, 10).double()) / B
    gW3, gb3 = e.T @ a2, e.sum(0)
    q2 = (e @ W3) * g2
    gW2, gb2 = q2.T @ a1, q2.sum(0)
    q1 = (q2 @ W2) * g1
    gW1, gb1 = q1.T @ xb, q1.sum(0)
    sW3, sb3 = ea.T @ a2.abs(), ea.sum(0)
    sq2 = (ea @ W3.abs()) * g2
    sW2, sb2 = sq2.T @ a1.abs(), sq2.sum(0)
    sq1 = (sq2 @ W2.abs()) * g1
    sW1, sb1 = sq1.T @ xb.abs(), sq1.sum(0)
    return [gW1, gb1, gW2, gb2, gW3, gb3], [sW1, sb1, sW2, sb2, sW3, sb3]


def s4(R) -> dict:
    """The unanchored forward is dtype-generic; run it in float64 and compare its autograd gradient with
    _manual_noanchor64.  Both are float64 evaluations of one formula, so they differ by summation
    rounding only: |diff| <= 3 * 16 * 785 * eps64 * scale (resp_ee's S4 bound).  Non-vacuity: the
    gradient with the natural value and derivative (d = 0) differs in W2."""
    failed, worst = [], {}
    prows, cks, x = _full_states()
    C = 3 * 16 * 785 * EPS64
    a = "R2_10_noanchor"
    arm = R.ARMS[a]
    ck = cks[arm["branch"]]
    sh, _, _ = R.build(arm, cks, x, R.SPE * R.EPOCHS)
    sh64 = {"p0": [p.double() for p in sh["p0"]], 2: sh[2].double()}
    fwd64 = R.make_forward_noanchor(sh64)
    p64 = [p.detach().double().requires_grad_(True) for p in ck["params"]]
    _, yb, ob, _ = _first_minibatch(ck, x)
    xb = x.double()[ob]
    out = fwd64(p64, xb, ob)
    g_run = torch.autograd.grad(F.cross_entropy(out[4], yb), p64)
    g_man, scale = _manual_noanchor64([p.detach() for p in p64], xb, yb, sh64[2][ob])
    rmax = 0.0
    for i, (gr, gm, sc) in enumerate(zip(g_run, g_man, scale)):
        diff = (gr - gm).abs()
        tol = C * sc + 1e-300
        rmax = max(rmax, float((diff / tol).max()))
        if (diff > tol).any():
            failed.append(f"{a}: tensor {i} off by {float(diff.max()):.3g}")
    g_nat, _ = _manual_noanchor64([p.detach() for p in p64], xb, yb, torch.zeros_like(sh64[2][ob]))
    rel = float((g_nat[2] - g_man[2]).abs().max() / (g_nat[2].abs().max() + g_man[2].abs().max()))
    worst[a] = {"max_diff_over_tol": rmax, "natural_vs_shifted_rel": rel}
    if rel < 1e-3:
        failed.append(f"{a}: shifted and natural gradients agree (vacuous)")
    return {"pass": not failed, "failed_items": failed, "worst": worst}


S5_ARMS = ("S2dyn_10r", "S2ramp_r", "S2dyn_20r", "S2dyn_5r")


def s5(R) -> dict:
    """A fresh Adam's first update is lr * g / (|g| + eps) up to float32 rounding (<= 8 eps32 relative
    per coordinate) and the float32 mean of N squares (<= (N - 1) eps32 relative):
    |RMS_run - RMS_expected| <= ((N - 1) + 8) eps32 RMS_expected (resp_ee's S5 bound).  g is recomputed
    with the arm's forward at the field it starts with (d(0)), on the first minibatch."""
    failed, got = [], {}
    prows, cks, x = _full_states()
    for a in S5_ARMS:
        arm = R.ARMS[a]
        ck = cks[arm["branch"]]
        units = {}
        with flush_denormal():
            R.run_arm(a, arm, cks, x, prows, 0, epochs=1, units=units)
        run_rms = units[f"{a}_k1_step"][0].astype(np.float64)
        params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
        sh, fwd, mover = R.build(arm, cks, x, R.SPE)
        xb, yb, ob, _ = _first_minibatch(ck, x)
        g = torch.autograd.grad(F.cross_entropy(fwd(params, xb, ob)[4], yb), params)
        exp_rms = np.array([float((R.LR * g[k].double() / (g[k].double().abs() + R.EPS)).square().mean().sqrt())
                            for k in (0, 2, 4)])
        numel = np.array([params[k].numel() for k in (0, 2, 4)], dtype=np.float64)
        err = np.abs(run_rms - exp_rms)
        tol = ((numel - 1) + 8) * EPS32 * exp_rms
        got[a] = {"run": run_rms.tolist(), "expected": exp_rms.tolist(),
                  "err_over_tol": (err / np.maximum(tol, 1e-300)).tolist()}
        if (err > tol).any():
            failed.append(f"{a}: first update RMS {run_rms.tolist()} vs {exp_rms.tolist()}")
    return {"pass": not failed, "failed_items": failed, "arms": got}


# --------------------------------------------------------------------------
# S6: the verdict on synthetic shards
# --------------------------------------------------------------------------

T_TABLE = {(0.975, 9): 2.262157, (0.975, 7): 2.364624, (0.975, 23): 2.068658, (0.975, 2): 4.302653}
BASE = {"N2": 0.78, "N2r": 0.72, "N10": 0.14, "N10r": 0.17, "S2_10r": 0.46, "S2u30r": 0.21, "R2_10": 0.90,
        "R2dyn_10": 0.88, "S2dyn_10r": 0.46}


def _synth(root: Path, V, eff: dict | None = None, n_seeds=10, noise=0.01, big_noise=(), invalid=(),
           prereg="SYNTH", moved=True, neff_line=(0.8, 0.25), neff_x=None, nat=None):
    """Per arm: E = base + effect + noise; nbar_neffT2 set so that E = b0 + b1 log10(nbar) holds for every
    arm (before the per-seed noise) unless neff_x gives its log10(nbar)."""
    rng = np.random.default_rng(0)
    eff = eff or {}
    if root.exists():
        shutil.rmtree(root)
    base = dict(BASE)
    for arm in V.ARM_NAMES:
        base.setdefault(arm, 0.5)
    base.update({k: v for k, v in eff.items() if k.startswith("=")})
    b0, b1 = neff_line
    for s in range(n_seeds):
        d = root / "runs" / f"s{s}"
        d.mkdir(parents=True)
        rows = []
        for arm in V.ARM_NAMES:
            e_arm = base[arm] + eff.get(arm, 0.0)
            sd = 0.5 if arm in big_noise else noise
            for k in (1, 2):
                e = e_arm + sd * rng.standard_normal()
                lx = (e_arm - b0) / b1
                if neff_x is not None and arm in neff_x:
                    lx = neff_x[arm]
                r = {"seed": s, "arm": arm, "k": k, "online_acc": e, "online_ce": 1.0, "major_frac": 0.11,
                     "memo_acc_end": e, "logits_equal_full": (arm != "R2_10_noanchor") if k == 1 else "",
                     "logits_equal_mb": (arm != "R2_10_noanchor") if k == 1 else "",
                     "hash_match_prefix": True if arm in V.N_ARMS else "",
                     "hash_match_resp_ee": True if arm not in V.NEW_ARMS else "",
                     "shadow_hash_match_prefix": (not (s in invalid)) if arm in V.DYN_ARMS else "",
                     "field0_equal_fixed": True if arm in V.DYN_ARMS + ["S2ramp_r"] else "",
                     "nbar_neffT2": 10 ** lx, "neffT2_start": 10 ** lx, "zero2_start": 0.5,
                     "probe_zero2_mean": 0.5, "zero2_end": 0.5, "probe_effbar2_first": -1.0,
                     "probe_effbar2_last": -2.0, "probe_dmean2_first": -1.0,
                     "probe_dmean2_last": (-5.0 if moved else -1.0) if arm in V.DYN_ARMS else 0.0,
                     "probe_zarm2_first": 0.0, "probe_zarm2_last": 1.0, "step_w2": 0.1}
                rows.append(r)
        pd.DataFrame(rows).to_csv(d / "arms.csv", index=False)
        pref = [{"seed": s, "task": t, "online_acc": 0.5, "memo_acc": 0.5, "g2_end": 0.5, "gtr2_end": 0.5,
                 "zero2_end": 0.0, "neffT2_end": 0.5, "zbar2_end": 0.0, "mu2_end": 1.0,
                 "hash_match_resp_ee": True} for t in range(1, 23)]
        pd.DataFrame(pref).to_csv(d / "prefix.csv", index=False)
        (d / "provenance.json").write_text(json.dumps({"prereg_commit": prereg, "flush_denormal": True,
                                                        "epochs_per_task": 80}))
    if nat is not None:
        for (v, s), curve in nat.items():
            d = root / "nat" / f"{v}_s{s}"
            d.mkdir(parents=True)
            rows = [{"variant": v, "seed": s, "task": t, "online_acc": curve(t, s), "zbar2": -1.0, "zero2": 0.0,
                     "below_2p24_2": 0.0, "log10_g2_median": -3.0, "eps_w2_end": 0.0} for t in range(1, 23)]
            pd.DataFrame(rows).to_csv(d / "nat.csv", index=False)
            dtp = "torch.float32" if v.startswith("f32") else "torch.float64"
            if v == "f64_felu" and s == 99:
                dtp = "torch.float32"
            (d / "provenance.json").write_text(json.dumps({"prereg_commit": prereg, "flush_denormal": True,
                                                            "epochs_per_task": 80, "param_dtypes": [dtp]}))


def _ref32_synth(seeds):
    """A float32 reference: healthy 0.78 until task 5, collapsed 0.12 from task 8."""
    ser = pd.Series({t: (0.78 if t <= 5 else 0.45 if t <= 7 else 0.12) for t in range(1, 23)})
    return {s: ser for s in seeds}


def s6(V) -> dict:
    """Synthetic shards with a known structure; the verdict must return the designed labels.  The
    registered interval is checked against an independent 95% t computation from the files, the n_eff
    line against numpy.polyfit on the clipped log10 n-bar with the textbook prediction interval."""
    failed, cases = [], {}
    root = SCR / "synth"

    def case(name, want, want_lit=None, **kw):
        _synth(root / name, V, **kw)
        sh, _ = V.load(root / name)
        nat, _ = V.load_nat(root / name)
        res = V.analyze(sh, nat, _ref32_synth(range(12)), prereg="SYNTH", epochs=80)
        cases[name] = {"label": res["label"], "want": want, "literal": res.get("literal_label"),
                       "sub": res.get("sub_label"), "neff": res.get("neff_label")}
        if res["label"] != want:
            failed.append(f"{name}: {res['label']} != {want}")
        if want_lit is not None and res.get("literal_label") != want_lit:
            failed.append(f"{name}: literal {res.get('literal_label')} != {want_lit}")
        return res, sh

    # dyn at the response floor (0.21), N10r below it: FULL here, PARTIAL under the literal rule
    r, sh = case("full", "FULL_MEDIATION", "PARTIAL", eff={"S2dyn_10r": -0.25})
    d = np.array([V.E(sh[s]["arms"], "S2_10r") - V.E(sh[s]["arms"], "S2dyn_10r") for s in sh])
    half = T_TABLE[(0.975, 9)] * d.std(ddof=1) / math.sqrt(len(d))
    if abs(d.mean() - r["M"]["mean"]) > 1e-12 or abs((r["M"]["hi"] - r["M"]["lo"]) / 2 - half) > 1e-6 * half:
        failed.append("full: M is not the 95% paired interval of the files")
    case("partial", "PARTIAL", "PARTIAL", eff={"S2dyn_10r": -0.12})
    case("static", "STATIC_ONLY", "STATIC_ONLY")
    case("helps", "MOVEMENT_HELPS", "MOVEMENT_HELPS", eff={"S2dyn_10r": +0.15})
    case("unresolved", "UNRESOLVED", eff={"S2dyn_10r": -0.13}, big_noise=("S2dyn_10r",))   # sd 0.5
    case("beyond", "FULL_MEDIATION", "BEYOND_NATURAL", eff={"S2dyn_10r": -0.36})
    case("literal_full", "FULL_MEDIATION", "FULL_MEDIATION", eff={"S2dyn_10r": -0.25, "N10r": +0.04})
    case("no_gap", "NOT_REPRODUCED", eff={"N10": +0.64})
    case("no_fixed_sink", "NOT_REPRODUCED", eff={"S2_10r": +0.26})
    case("floor_above_fixed", "NOT_REPRODUCED", eff={"S2u30r": +0.30})
    case("field_static", "FIELD_STATIC", eff={"S2dyn_10r": -0.25}, moved=False)
    case("shadow_invalid", "INAPPLICABLE", eff={"S2dyn_10r": -0.25}, invalid=(0, 1, 2))
    case("wrong_prereg", "INAPPLICABLE", prereg="OTHER")
    # sub-label
    r, _ = case("sub_static", "STATIC_ONLY", eff={"R2dyn_10": -0.10})
    if r["sub_label"] != "EXCESS_IS_STATIC":
        failed.append(f"sub_static: {r['sub_label']}")
    r, _ = case("sub_remains", "STATIC_ONLY")
    if r["sub_label"] != "EXCESS_REMAINS":
        failed.append(f"sub_remains: {r['sub_label']}")
    r, _ = case("sub_reversed", "STATIC_ONLY", eff={"R2dyn_10": -0.30})
    if r["sub_label"] != "EXCESS_REVERSED":
        failed.append(f"sub_reversed: {r['sub_label']}")
    r, _ = case("sub_none", "STATIC_ONLY", eff={"R2_10": -0.12})
    if r["sub_label"] != "NO_EXCESS":
        failed.append(f"sub_none: {r['sub_label']}")
    # n_eff: old arms on E = 0.8 + 0.1 log10 n; one old arm far below the clip (x must be clipped)
    r, sh = case("neff_on", "FULL_MEDIATION", eff={"S2dyn_10r": -0.25}, neff_x={"S2u30r": -6.0})
    if r["neff_label"] != "NEFF_PREDICTS":
        failed.append(f"neff_on: {r['neff_label']}")
    t = r["arm_table"].set_index("arm")
    fit_arms = [a for a in V.OLD_ARMS if a not in V.NEFF_EXCLUDE]
    xs = np.array([math.log10(max(t.loc[a, "nbar_neffT2_k1"], 1 / 1200)) for a in fit_arms])
    ys = np.array([t.loc[a, "E_k1"] for a in fit_arms])
    b1, b0 = np.polyfit(xs, ys, 1)
    if abs(b1 - r["neff"]["b1"]) > 1e-9 or abs(b0 - r["neff"]["b0"]) > 1e-9:
        failed.append(f"neff_on: line {r['neff']['b0']:.6f} {r['neff']['b1']:.6f} != polyfit {b0:.6f} {b1:.6f}")
    res = ys - (b0 + b1 * xs)
    s_ = math.sqrt(float(res @ res) / (len(xs) - 2))
    x0 = math.log10(max(t.loc["S2dyn_10r", "nbar_neffT2_k1"], 1 / 1200))
    half = T_TABLE[(0.975, 23)] * s_ * math.sqrt(1 + 1 / len(xs) + (x0 - xs.mean()) ** 2 / ((xs - xs.mean()) ** 2).sum())
    q = r["neff"]["arms"]["S2dyn_10r"]
    if abs((q["hi"] - q["lo"]) / 2 - half) > 1e-6 * half:
        failed.append("neff_on: prediction interval")
    r, _ = case("neff_off", "FULL_MEDIATION", eff={"S2dyn_10r": -0.25}, neff_x={"S2dyn_10r": -0.1})
    if r["neff_label"] != "NEFF_MISSES":
        failed.append(f"neff_off: {r['neff_label']}")
    # the side run
    col = lambda t, s: 0.78 if t <= 5 else 0.45 if t <= 7 else 0.12              # noqa: E731
    healthy = lambda t, s: 0.78                                                  # noqa: E731
    late = lambda t, s: 0.78 if t <= 14 else 0.12                                # noqa: E731
    nat_all = lambda f: {**{("f64_felu", s): f for s in range(10)},             # noqa: E731
                         **{("f32_felu", s): f for s in range(3)}, **{("f64_host", s): f for s in range(3)}}
    for name, curve, want in (("nat_repro", col, "REPRODUCED"), ("nat_not", healthy, "NOT_REPRODUCED"),
                              ("nat_delayed", late, "DELAYED")):
        r, _ = case(name, "STATIC_ONLY", nat=nat_all(curve))
        if r["nat"]["f64_felu"]["label"] != want:
            failed.append(f"{name}: {r['nat']['f64_felu']['label']} != {want}")
    few = {("f64_felu", s): col for s in range(7)}
    r, _ = case("nat_few", "STATIC_ONLY", nat=few)
    if r["nat"]["f64_felu"]["label"] != "INAPPLICABLE":
        failed.append(f"nat_few: {r['nat']['f64_felu']['label']}")
    # a variant run in the wrong dtype is invalid (seed 99 is written with float32 but is not listed:
    # use the verdict's own dtype rule on a listed seed instead)
    _synth(root / "nat_dtype", V, nat=nat_all(col))
    p = root / "nat_dtype" / "nat" / "f64_felu_s0" / "provenance.json"
    pj = json.loads(p.read_text())
    pj["param_dtypes"] = ["torch.float32"]
    p.write_text(json.dumps(pj))
    nat, _ = V.load_nat(root / "nat_dtype")
    rn = V.nat_analysis(nat, _ref32_synth(range(12)), "SYNTH", 80)
    if 0 not in rn["f64_felu"]["invalid"]:
        failed.append("nat_dtype: a float32 run counted as f64_felu")
    # prediction scoring: the impl prediction for S2dyn_10r (0.27) is inside a case centred on it
    r, _ = case("pred_hit", "PARTIAL", eff={"S2dyn_10r": -0.19})
    if not r["predictions"]["impl"]["E_S2dyn_10r"]["hit"] or r["predictions"]["design"]["E_S2dyn_10r"]["hit"]:
        failed.append("pred_hit: point-prediction scoring")
    tq = {f"{p_}/{df}": V.RV.t_quantile(p_, df) for (p_, df) in T_TABLE}
    for (p_, df), want in T_TABLE.items():
        if abs(V.RV.t_quantile(p_, df) - want) > 1e-6:
            failed.append(f"t_quantile({p_},{df})")
    Rr = CACHE["real_runner"]
    if list(V.ARM_NAMES) != list(Rr.ARMS):
        failed.append("verdict ARM_NAMES != runner ARMS")
    if V.DYN_ARMS != [a for a, v in Rr.ARMS.items() if v["kind"] == "dyn"]:
        failed.append("verdict DYN_ARMS != runner dyn arms")
    return {"pass": not failed, "failed_items": failed, "cases": cases, "t_quantiles": tq}


# --------------------------------------------------------------------------
# S7: CLI
# --------------------------------------------------------------------------

S7_ARMS = "S2dyn_10r,N2"


def s7(R) -> dict:
    """The CLI (subprocess, as the launcher runs it) and the in-process run_seed with the same short
    settings write identical arms.csv and traj.csv columns; two CLI runs agree; the provenance says flush
    on, 1 thread, this experiment, the registered commit the runner carries and a git hash read at start."""
    failed = []
    d1, d2, d3 = SCR / "s7_cli_a", SCR / "s7_cli_b", SCR / "s7_inproc"
    for d in (d1, d2, d3):
        if d.exists():
            shutil.rmtree(d)
    tmp = REPO / "src" / "_s7_respdyn_runner.py"
    for d in (d1, d2):
        cmd = [PY, str(tmp), "--seed", "0", "--epochs", "1", "--prefix-tasks", "12", "--arms", S7_ARMS,
               "--out", str(d)]
        tmp.write_text(R._source_text)
        try:
            r = subprocess.run(cmd, env=THREAD_ENV, capture_output=True, text=True, timeout=600)
        finally:
            tmp.unlink(missing_ok=True)
        if r.returncode != 0:
            failed.append(f"CLI exit {r.returncode}: {r.stderr[-300:]}")
            return {"pass": False, "failed_items": failed}
    with flush_denormal():
        R.run_seed(0, d3, MNIST, epochs=1, arms=S7_ARMS.split(","), prefix_tasks=12, save_ckpt=False,
                   progress=False)
    a1, a2, a3 = (pd.read_csv(d / "arms.csv") for d in (d1, d2, d3))
    for col in ("state_sha256", "online_acc", "online_ce", "shadow_state_sha256", "nbar_neffT2"):
        if not (a1[col].equals(a2[col]) and a1[col].equals(a3[col])):
            failed.append(f"arms.csv {col} differs")
    t1, t3 = pd.read_csv(d1 / "traj.csv"), pd.read_csv(d3 / "traj.csv")
    if not t1.equals(t3):
        failed.append("traj.csv differs")
    if not all(a1["shadow_hash_match_prefix"].dropna()):
        failed.append("CLI shadow does not reproduce the prefix")
    p1 = json.loads((d1 / "provenance.json").read_text())
    if p1.get("flush_denormal") is not True:
        failed.append("provenance flush_denormal")
    if p1.get("threads") != 1:
        failed.append(f"provenance threads {p1.get('threads')}")
    if p1.get("experiment") != "respdyn_ee_0917":
        failed.append("provenance experiment")
    if p1.get("prereg_commit") != R.PREREG_COMMIT:
        failed.append("provenance prereg_commit")
    if not p1.get("git_hash"):
        failed.append("provenance git_hash")
    for f in ("prefix.csv", "arms.csv", "traj.csv", "dstar.json", "units.npz", "provenance.json"):
        if not (d1 / f).exists():
            failed.append(f"missing {f}")
    return {"pass": not failed, "failed_items": failed, "prereg_commit": p1.get("prereg_commit"),
            "cpu_capability": p1.get("cpu_capability")}


# --------------------------------------------------------------------------
# S8: the side runner
# --------------------------------------------------------------------------

def s8(N) -> dict:
    """(a) f32_host is RE.run_prefix bit for bit (state sha256 and online accuracy, 3 short tasks).
    (b) After a short run each variant's parameters, moments and images have the variant's dtype.
    (c) Training derivative through the variant's activation, by autograd on a grid: F.elu variants
    give exp(z) > 0 on [-700, 0] (float64) and [-85, 0] (float32, above the flush boundary ln 2^-126
    = -87.3), equal to torch.exp within 4 eps relative (the same exp up to the kernel's rounding);
    the float64 host ELU gives expm1 + 1 exactly (so 0 below ln 2^-53 = -36.7)."""
    failed = []
    with flush_denormal():
        pr, _, _, _, _ = RE.run_prefix(0, MNIST, EP, 3, save_at=())
        rows, _ = N.run_nat("f32_host", 0, MNIST, EP, 3)
        if [r["state_sha256"] for r in pr] != [r["state_sha256"] for r in rows] or \
                [r["online_acc"] for r in pr] != [r["online_acc"] for r in rows]:
            failed.append("f32_host != RE.run_prefix")
        dts = {}
        for v, (dtype, act) in N.VARIANTS.items():
            _, info = N.run_nat(v, 0, MNIST, 1, 1)
            want = [str(dtype)]
            dts[v] = info
            if info["param_dtypes"] != want or info["moment_dtypes"] != want or info["x_dtype"] != str(dtype):
                failed.append(f"{v}: dtypes {info}")
    grids = {"f64_felu": (torch.float64, -700.0), "f32_felu": (torch.float32, -85.0)}
    for v, (dtype, lo) in grids.items():
        z = torch.linspace(lo, 0.0, 20001, dtype=dtype).requires_grad_(True)
        with flush_denormal():
            g, = torch.autograd.grad(N.VARIANTS[v][1].phi(z).sum(), z)
            ex = torch.exp(z.detach())
        eps = EPS64 if dtype == torch.float64 else EPS32
        if not bool((g > 0).all()):
            failed.append(f"{v}: derivative 0 somewhere on [{lo}, 0]")
        if bool(((g - ex).abs() > 4 * eps * ex).any()):
            failed.append(f"{v}: derivative is not exp(z)")
    z = torch.linspace(-45.0, 5.0, 20001, dtype=torch.float64).requires_grad_(True)
    g, = torch.autograd.grad(N.VARIANTS["f64_host"][1].phi(z).sum(), z)
    if not torch.equal(g, TRAIN_DPHI(z.detach())):
        failed.append("f64_host: derivative is not expm1 + 1")
    return {"pass": not failed, "failed_items": failed, "dtypes": dts}


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

def _mem_available_gib() -> float:
    return [int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
            if l.startswith("MemAvailable:")][0] / 2 ** 20


def s_cost() -> dict:
    """Up to PROBE processes at once: the prefix to task 12 and S2dyn_10r (a dyn arm with its shadow)
    at full length, plus one f64_felu side run of 3 tasks.  The number started is what fits now with
    6 GiB left for the desktop at 1.5 GiB per process; this is a shared machine."""
    d = SCR / "cost"
    if d.exists():
        shutil.rmtree(d)
    probe = max(1, min(PROBE, int((_mem_available_gib() - 6.0) // 1.5)))
    procs = []
    t0 = time.time()
    for i in range(max(1, probe - 1)):
        cmd = [PY, str(RUNNER), "--seed", str(i), "--prefix-tasks", "12", "--arms", "S2dyn_10r",
               "--out", str(d / f"p{i}")]
        procs.append(subprocess.Popen(cmd, env=THREAD_ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    nat_cmd = [PY, str(NATRUN), "--variant", "f64_felu", "--seed", "0", "--tasks", "3", "--out", str(d / "nat")]
    procs.append(subprocess.Popen(nat_cmd, env=THREAD_ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    codes = [p.wait() for p in procs]
    wall = time.time() - t0
    provs = [json.loads((d / f"p{i}" / "provenance.json").read_text()) for i in range(len(procs) - 1)
             if codes[i] == 0]
    arms = [pd.read_csv(d / f"p{i}" / "arms.csv") for i in range(len(procs) - 1) if codes[i] == 0]
    per_prefix = [p["seconds_prefix"] / (12 * 6000) for p in provs]
    per_dyn = [float(a["sec"].sum()) / 12000 for a in arms]
    rss = [p["peak_rss_kb"] / 2 ** 20 for p in provs]
    natp = json.loads((d / "nat" / "provenance.json").read_text()) if codes[-1] == 0 else {}
    avail = _mem_available_gib()
    return {"pass": all(c == 0 for c in codes), "codes": codes, "wall_s": wall,
            "sec_per_update_prefix": per_prefix, "sec_per_update_dyn_arm": per_dyn,
            "nat_seconds_3_tasks": natp.get("seconds_total"),
            "nat_peak_rss_gib": natp.get("peak_rss_kb", 0) / 2 ** 20,
            "peak_rss_gib": rss, "peak_rss_gib_max": max(rss) if rss else None,
            "mem_available_gib_after": avail, "probe": probe,
            "slots": int((avail - 6.0) // (1.2 * max(rss))) if rss else 0}


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    global DUMP_PATH
    if only:
        DUMP_PATH = SCR / "checks_partial.json"
    SCR.mkdir(parents=True, exist_ok=True)
    RESULTS.update({"experiment": "respdyn_ee_0917", "started": dt.datetime.now().isoformat(),
                    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                               text=True).stdout.strip(),
                    "runner_sha256": _sha(RUNNER), "natrunner_sha256": _sha(NATRUN),
                    "verdict_sha256": _sha(VERDICT), "checks_sha256": _sha(Path(__file__))})
    R = load(RUNNER)
    CACHE["real_runner"] = R
    want = lambda k: only is None or k in only             # noqa: E731

    if want("S0"):
        run_check("S0", "resp_ee_0917 re-made on this machine (seeds 0-1, full length)", s0, [
            ("Adam eps off by half in the Stepper",
             [("                upd = LR * (mi / c1) / ((vi / c2).sqrt() + EPS)\n                p -= upd\n                if k % 2 == 0:\n                    self.stp",
               "                upd = LR * (mi / c1) / ((vi / c2).sqrt() + EPS * 1.5)\n                p -= upd\n                if k % 2 == 0:\n                    self.stp")]),
        ], s0.__doc__, RUNNER)
    if want("S1b"):
        run_check("S1b", "Stepper = RE.train_task (all 26 resp_ee arms)", s1b, [
            ("bias correction of v one step late",
             [("            c1, c2 = 1 - BETA1 ** tc[0], 1 - BETA2 ** tc[0]\n            for k, (p, gr, mi, vi)",
               "            c1, c2 = 1 - BETA1 ** tc[0], 1 - BETA2 ** (tc[0] + 1)\n            for k, (p, gr, mi, vi)")]),
            ("minibatch order reversed",
             [("            self.order = torch.randperm(N_IMG, generator=self.g_batch).to(x.device)\n",
               "            self.order = torch.randperm(N_IMG, generator=self.g_batch).to(x.device).flip(0)\n")]),
            ("online accuracy miscounted",
             [("        self.acc[s] = (out[4].detach().argmax(1) == yb).float().mean()\n",
               "        self.acc[s] = (out[4].detach().argmax(1) == yb).float().mean() * 0.5\n")]),
            ("minibatch index taken from the epoch",
             [("        j = s % SPE\n", "        j = (s // SPE) % SPE\n")]),
            ("second layer leaky in the natural forward",
             [("            out = EL.forward2(params, xb, ACT, ACT)\n",
               "            out = EL.forward2(params, xb, ACT, H.ARMS['LR'])\n")]),
        ], s1b.__doc__, RUNNER)
    if want("S2"):
        run_check("S2", "continuation: naturals and shadows reproduce the prefix; moving fields move", s2, [
            ("shadow on the arm's label stream",
             [('self.g_lab, self.g_batch = gen_from(src["g_lab"]), gen_from(src["g_batch"])',
               'self.g_lab, self.g_batch = gen_from(br["g_lab"]), gen_from(src["g_batch"])')]),
            ("shadow parameters aliased to the branch state",
             [('        self.params = [p.detach().clone().requires_grad_(True) for p in src["params"]]\n',
               '        self.params = [p.detach().requires_grad_(True) for p in src["params"]]\n')]),
            ("shadow never steps",
             [("        if not self.freeze:\n            self.stepper.step(s)\n",
               "        if False:\n            self.stepper.step(s)\n")]),
            ("shadow with a fresh Adam",
             [('        self.adam = ([q.clone() for q in src["m"]], [q.clone() for q in src["v"]], [src["tc"]])\n',
               '        self.adam = ([torch.zeros_like(q) for q in src["m"]], [torch.zeros_like(q) for q in src["v"]], [0])\n')]),
            ("field refreshed from the branch (zero field)",
             [("            z2 = z2_full(self.params, self.x)\n", "            z2 = self.zbr.float()\n")]),
            ("ramp does not deepen",
             [("        self.d.fill_(-(self.lo + (self.hi - self.lo) * g / self.total))\n",
               "        self.d.fill_(-(self.lo + (self.hi - self.lo) * 0 / self.total))\n")]),
        ], s2.__doc__, RUNNER)
    if want("St0"):
        run_check("S-t0", "moving field: start = fixed field; frozen = fixed twin; timing", st0, [
            ("shadow one update ahead of the arm",
             [("            st.step(s)\n            if mover is not None:\n                mover.step(s)\n",
               "            if mover is not None:\n                mover.step(s)\n                mover.refresh(g0 + s + 1)\n            st.step(s)\n")]),
            ("field relative to the source instead of the branch",
             [('        self.zbr = br["z2"].double()\n', '        self.zbr = src["z2"].double()\n')]),
            ("freeze ignored",
             [("        if not self.freeze:\n            self.stepper.step(s)\n",
               "        if True:\n            self.stepper.step(s)\n")]),
            ("ramp starts at 11", [("RAMP = (10.0, 30.0)", "RAMP = (11.0, 30.0)")]),
        ], st0.__doc__, RUNNER)
    if want("S3"):
        run_check("S3", "branch point: anchored arms keep the logits; the unanchored arm takes the source features", s3, [
            ("unanchored arm anchored",
             [("        return sh, make_forward_noanchor(sh), None\n", "        return sh, RE.make_forward(sh), None\n")]),
            ("unanchored arm without its field",
             [("        a2 = ACT.phi(z2 + d2[ob])\n", "        a2 = ACT.phi(z2)\n")]),
            ("dyn field sign flipped",
             [("            self.d.copy_((z2.double() - self.zbr).float())\n",
               "            self.d.copy_((self.zbr - z2.double()).float())\n")]),
        ], s3.__doc__, RUNNER)
    if want("S4"):
        run_check("S4", "unanchored gradient = phi'(z + d) chain (float64, independent)", s4, [
            ("unanchored derivative bypassed (forward unchanged)",
             [("        a2 = ACT.phi(z2 + d2[ob])\n",
               "        a2 = ACT.phi(z2.detach() + d2[ob]) + (z2 - z2.detach())\n")]),
            ("unanchored field misaligned with the minibatch",
             [("        a2 = ACT.phi(z2 + d2[ob])\n", "        a2 = ACT.phi(z2 + d2[:ob.shape[0]])\n")]),
        ], s4.__doc__, RUNNER)
    if want("S5"):
        run_check("S5", "fresh Adam in the new reset arms: first update = lr g / (|g| + eps)", s5, [
            ("reset keeps the step count",
             [('        adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])\n    else:\n        adam = ([q.clone() for q in ck["m"]]',
               '        adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [ck["tc"]])\n    else:\n        adam = ([q.clone() for q in ck["m"]]')]),
            ("reset keeps v",
             [('        adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])\n    else:\n        adam = ([q.clone() for q in ck["m"]]',
               '        adam = ([torch.zeros_like(q) for q in params], [q.clone() for q in ck["v"]], [0])\n    else:\n        adam = ([q.clone() for q in ck["m"]]')]),
        ], s5.__doc__, RUNNER)
    if want("S6"):
        run_check("S6", "verdict on synthetic shards", s6, [
            ("M sign convention flipped", [("M_PAIR = (FIXED, MAIN)", "M_PAIR = (MAIN, FIXED)")]),
            ("response floor taken as N10r",
             [('MAIN, FIXED, FLOOR_ARM = "S2dyn_10r", "S2_10r", "S2u30r"',
               'MAIN, FIXED, FLOOR_ARM = "S2dyn_10r", "S2_10r", "N10r"')]),
            ("R = 0 read as PARTIAL",
             [('    return "PARTIAL" if r["sign"] == "+" else "FULL_MEDIATION"',
               '    return "PARTIAL" if r["sign"] != "-" else "FULL_MEDIATION"')]),
            ("(B) skipped", [('    if not all(B[k]["lo"] > 0 for k in B_PAIRS):', '    if False:')]),
            ("field-moved condition skipped", [('    elif not B["field_moved"]:', '    elif False:')]),
            ("shadow validity ignored",
             [('        if arm in DYN_ARMS and not all(_truth(v) for v in r["shadow_hash_match_prefix"]):',
               '        if False:')]),
            ("prereg not enforced",
             [('    if prereg is not None and prov.get("prereg_commit") != prereg:\n        return',
               '    if False:\n        return')]),
            ("n_eff not clipped at one image",
             [("    return math.log10(max(float(v), NEFF_FLOOR))", "    return math.log10(max(float(v), 1e-300))")]),
            ("side-run label ignores task 11",
             [('            if dE["hi"] < 0 and dL["hi"] < 0:', '            if dL["hi"] < 0:')]),
            ("interval level 97.5%", [("LEVEL = 0.95 ", "LEVEL = 0.975")]),
            ("excess gate removed", [('    if ex["lo"] <= 0:', '    if False:')]),
        ], s6.__doc__, VERDICT)
    if want("S7"):
        run_check("S7", "CLI = in-process run, with provenance", s7, [
            ("CLI leaves flush off",
             [("    torch.set_flush_denormal(True)\n    H.setup(\"cpu\")\n",
               "    torch.set_flush_denormal(False)\n    H.setup(\"cpu\")\n")]),
            ("CLI ignores --threads",
             [("    torch.set_num_threads(args.threads)\n", "    torch.set_num_threads(2)\n")]),
        ], s7.__doc__, RUNNER)
    if want("S8"):
        run_check("S8", "side runner: f32_host = prefix; dtypes; exp derivative", s8, [
            ("f64_felu runs the host ELU",
             [('VARIANTS = {"f64_felu": (torch.float64, FELU),', 'VARIANTS = {"f64_felu": (torch.float64, RE.ACT),')]),
            ("parameters not cast",
             [("    params = [p.detach().to(dtype).clone().requires_grad_(True)\n",
               "    params = [p.detach().clone().requires_grad_(True)\n")]),
            ("bias correction of v one step late",
             [("                c1, c2 = 1 - BETA1 ** tc[0], 1 - BETA2 ** tc[0]\n",
               "                c1, c2 = 1 - BETA1 ** tc[0], 1 - BETA2 ** (tc[0] + 1)\n")]),
            ("F.elu replaced by the expm1 form",
             [("        return F.elu(z)\n", "        return torch.where(z > 0, z, torch.expm1(z.clamp(max=0.)))\n")]),
        ], s8.__doc__, NATRUN)
    if want("Scost"):
        RESULTS["S-cost"] = s_cost()
        RESULTS["S-cost"]["pass_note"] = "recorded, not a gate"
        dump()
        print("S-cost:", {k: RESULTS["S-cost"][k] for k in ("wall_s", "sec_per_update_prefix",
                                                           "sec_per_update_dyn_arm", "peak_rss_gib_max",
                                                           "slots")}, flush=True)
    RESULTS["finished"] = dt.datetime.now().isoformat()
    dump()
    print("all_pass =", RESULTS["all_pass"])


if __name__ == "__main__":
    main()

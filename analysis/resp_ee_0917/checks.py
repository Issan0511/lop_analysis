#!/usr/bin/env python3
"""Checks for resp_ee_0917 (specs/spec_resp_ee_0917.md section 6).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/resp_ee_0917/checks.py
    ... --only S3,S4          # development

Each check runs on the real code and then on every listed mutation (an exact-once substitution in the
check's target file); every mutation must make the same check fail.  Tolerances are derived in the
docstrings from the arithmetic, never from looking at values.  Writes results/resp_ee_0917/checks.json.

S0  machine-bound link: the runner's prefix is mucap_ee_0917's ref shard (seeds 0, 1; tasks 1-22; full
    80-epoch tasks; online_acc and memo_acc equal as floats).  Also keeps seed 0's branch states for S3-S5.
S1  the runner's update loop is the host's run_one (ref, ELU->ELU) bit for bit (short tasks)
S2  continuation: every natural arm reproduces the prefix at t+1 and t+2 (state sha256); every other arm
    leaves it; the saved branch states are untouched by the arms (short tasks, all 26 arms)
S3  branch point: every intervention arm's logits equal the natural net's bit for bit (all 1200 images
    and the first minibatch); the trained-with argument z + d sits on the source field / the shifted
    value within its rounding bound (full-length states)
S4  the gradient: the runner's forward, run in float64, gives the gradient of phi'(z + d) computed by an
    independent hand-written chain rule (full-length states, first minibatch)
S5  fresh Adam: a reset arm's first update is lr * g / (|g| + eps) coordinate-wise (full-length states)
S6  the verdict on synthetic seeds with known effects; the t quantile against the published table
S7  the CLI writes what the in-process run writes, twice the same, with this experiment's provenance
S-cost  PROBE processes at once (peak RSS, seconds per update)
"""
from __future__ import annotations

import argparse
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

RUNNER = REPO / "src" / "resp_ee_0917.py"
VERDICT = REPO / "analysis" / "resp_ee_0917" / "verdict.py"
OUT = REPO / "results" / "resp_ee_0917"
SCR = REPO / "results" / "_checks_resp_ee_0917"
MUCAP_EE_RUNS = Path("/home/issan/Projects/claude/wt/mucap_ee_0917/results/mucap_ee_0917/runs")
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

# --------------------------------------------------------------------------
# infrastructure (mucap_ee_0917's)
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
    checks = {k: v for k, v in RESULTS.items() if k.startswith("S") and isinstance(v, dict) and "pass" in v}
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


def _images(seed: int) -> torch.Tensor:
    return MNIST.train_x[RL.subset_idx(seed)]


def _state_digest(cks: dict) -> str:
    h = hashlib.sha256()
    for t in sorted(cks):
        c = cks[t]
        for q in (*c["params"], *c["m"], *c["v"], c["z1"], c["z2"], c["g_lab"], c["g_batch"]):
            h.update(q.detach().numpy().tobytes())
        h.update(str(c["tc"]).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# S0: the prefix is mucap_ee_0917's ref (machine-bound)
# --------------------------------------------------------------------------

S0_SEEDS = (0, 1)
S0_TASKS = 22


def s0(R) -> dict:
    """Exact float equality of online_acc and memo_acc per task: both are exact sums/ratios of counts,
    so any trajectory difference that changes one prediction shows; the state is not recorded by
    mucap_ee, so the per-task accuracies are the link.  Stops a seed at its first mismatch."""
    failed, compared, secs = [], 0, {}
    rec_sha = {}
    for seed in S0_SEEDS:
        f = MUCAP_EE_RUNS / f"ref_s{seed}" / "per_task.csv"
        if not f.exists():
            failed.append(f"seed {seed}: no mucap_ee record at {f}")
            continue
        rec_sha[seed] = _sha(f)
        rec = pd.read_csv(f, float_precision="round_trip").set_index("task")
        bad = []

        def stop(t, row):
            nonlocal compared
            compared += 1
            ok = (float(rec.loc[t, "online_acc"]) == row["online_acc"]
                  and float(rec.loc[t, "memo_acc"]) == row["memo_acc"])
            if not ok:
                bad.append(f"seed {seed} task {t}: online {row['online_acc']!r} vs {rec.loc[t, 'online_acc']!r}, "
                           f"memo {row['memo_acc']!r} vs {rec.loc[t, 'memo_acc']!r}")
            return not ok
        t0 = time.time()
        with flush_denormal():
            prows, units, cks, x, info = R.run_prefix(seed, MNIST, R.EPOCHS, S0_TASKS, save_at=R.BRANCH_T,
                                                     stop_at=stop)
        secs[seed] = time.time() - t0
        failed += bad
        if not bad and len(prows) < S0_TASKS:
            failed.append(f"seed {seed}: only {len(prows)} tasks")
        if not bad and seed == 0 and getattr(R, "_is_base", False):
            CACHE["s0_seed0"] = {"prows": prows, "cks": cks, "x": x, "seconds": secs[seed]}
            CACHE["s0_task_seconds"] = [r["sec"] for r in prows]
    return {"pass": not failed, "compared_tasks": compared, "failed_items": failed,
            "record_sha256": rec_sha, "seconds_per_seed": secs, "machine": os.uname().nodename,
            "cpu_capability": torch.backends.cpu.get_cpu_capability()}


# --------------------------------------------------------------------------
# S1: the update loop is the host's
# --------------------------------------------------------------------------

S1_EPOCHS, S1_TASKS = 2, 3


def s1(R) -> dict:
    """Bit equality: final state sha256, per-task online_acc and memo_acc, init/subset sha256."""
    failed = []
    with flush_denormal():
        rows_h, _, _, info_h = EL.run_one("ref", 0, 1e-3, S1_TASKS, MNIST, DEV, epochs=S1_EPOCHS,
                                          ledger=False, act2_name="ELU1")
        prows, _, cks, _, info_r = R.run_prefix(0, MNIST, S1_EPOCHS, S1_TASKS, save_at=(1,))
    if prows[-1]["state_sha256"] != info_h["final_state_sha256"]:
        failed.append("final state sha256")
    if cks[1]["state_sha256"] != info_h["task1_end_state_sha256"]:
        failed.append("task-1 state sha256")
    for a, b in zip(rows_h, prows):
        if a["online_acc"] != b["online_acc"] or a["memo_acc"] != b["memo_acc"]:
            failed.append(f"task {a['task']}: online {a['online_acc']!r}/{b['online_acc']!r} "
                          f"memo {a['memo_acc']!r}/{b['memo_acc']!r}")
    for k in ("init_sha256", "subset_idx_sha256"):
        if info_h[k] != info_r[k]:
            failed.append(k)
    return {"pass": not failed, "failed_items": failed, "tasks": len(prows)}


# --------------------------------------------------------------------------
# S2: continuation identity, all arms, short tasks
# --------------------------------------------------------------------------

S2_EPOCHS = 2


def s2(R) -> dict:
    """Natural arms: state sha256 after continuation task k == the prefix's at branch + k (both k).
    Other arms: the response they start with is the one they were given -- a field arm's unit-mean
    phi' at the branch point is closer to the source task's than to the branch task's (prefix rows), a
    uniform arm's is strictly below the natural arm's (phi'(z - delta) <= phi'(z) for ELU, with equality
    only where z > delta) -- and a reset-only arm ends task 1 in a different state from the natural arm.
    Every arm's branch-point logits equal the natural net's.
    The saved branch states' digest is unchanged by running all arms."""
    failed = []
    with flush_denormal():
        prows, _, cks, x, _ = R.run_prefix(0, MNIST, S2_EPOCHS, R.PREFIX_T, save_at=R.BRANCH_T)
        before = _state_digest(cks)
        rows = []
        for a in R.ARMS:
            rows += R.run_arm(a, R.ARMS[a], cks, x, prows, 0, epochs=S2_EPOCHS)
        after = _state_digest(cks)
    if before != after:
        failed.append("branch states modified by the arms")
    df = pd.DataFrame(rows)
    nat = {}
    for a, arm in R.ARMS.items():
        r = df[df["arm"] == a]
        if len(r) != R.CONT:
            failed.append(f"{a}: {len(r)} rows")
            continue
        r1 = r[r["k"] == 1].iloc[0]
        if not (r1["logits_equal_full"] and r1["logits_equal_mb"]):
            failed.append(f"{a}: branch-point logits differ")
        if not np.isfinite(r["online_acc"]).all():
            failed.append(f"{a}: non-finite")
        if a.startswith("N") and not a.endswith("r"):
            if not all(r["hash_match_prefix"]):
                failed.append(f"{a}: does not reproduce the prefix")
            nat[arm["branch"]] = r1["state_sha256"]
    gpre = {r["task"]: r for r in prows}
    for a, arm in R.ARMS.items():
        if a.startswith("N") and not a.endswith("r"):
            continue
        r1 = df[(df["arm"] == a) & (df["k"] == 1)]
        if not len(r1):
            continue
        r1 = r1.iloc[0]
        if arm["kind"] is None:
            if r1["state_sha256"] == nat.get(arm["branch"]):
                failed.append(f"{a}: identical to the natural continuation (vacuous reset)")
            continue
        L = arm["layer"]
        g_arm = r1[f"g{L}_start"]
        g_nat = df[(df["arm"] == f"N{arm['branch']}") & (df["k"] == 1)].iloc[0][f"g{L}_start"]
        if arm["kind"] == "field":
            g_src = gpre[arm["src"]][f"g{L}_end"]
            if not abs(g_arm - g_src) < abs(g_arm - g_nat):
                failed.append(f"{a}: start response {g_arm:.6g} not nearer the source {g_src:.6g} "
                              f"than the branch {g_nat:.6g}")
        elif not g_arm < g_nat:
            failed.append(f"{a}: start response {g_arm:.6g} not below the natural {g_nat:.6g}")
    n_nat = sum(1 for a in R.ARMS if a.startswith("N") and not a.endswith("r"))
    return {"pass": not failed, "failed_items": failed, "n_arms": len(R.ARMS), "n_natural": n_nat,
            "n_rows": len(df)}


# --------------------------------------------------------------------------
# S3-S5 use seed 0's full-length branch states from S0 (the real runner's)
# --------------------------------------------------------------------------

def _states():
    if "s0_seed0" not in CACHE:
        R = CACHE["real_runner"]
        with flush_denormal():
            prows, _, cks, x, _ = R.run_prefix(0, MNIST, R.EPOCHS, R.PREFIX_T, save_at=R.BRANCH_T)
        CACHE["s0_seed0"] = {"prows": prows, "cks": cks, "x": x}
    c = CACHE["s0_seed0"]
    return c["prows"], c["cks"], c["x"]


def _first_minibatch(ck, x):
    y = RL.task_labels(_gen(ck["g_lab"]))
    order = torch.randperm(RL.N_IMAGES, generator=_gen(ck["g_batch"]))
    ob = order[:RL.BATCH]
    return x[ob], y[ob], ob, y


def _gen(state):
    g = torch.Generator(device="cpu")
    g.set_state(state.clone())
    return g


def s3(R) -> dict:
    """(a) logits at the branch point on all 1200 images and on the first minibatch equal the natural
    net's bit for bit (torch.equal).
    (b) the argument the arm trains with, on all 1200 images: the arm's full-batch z is the stored
    branch field z_b bit for bit (same op on the same state), so z_b + d with d = float32(z_src - z_b)
    differs from z_src only by the float32 cast of d (<= eps32 |d| / 2) and the float64 sum of two
    float32 values (<= eps64 (|z_b| + |d|)): bound = eps32 |d| + 2 eps64 (|z_b| + |d|), per unit and image.
    With the float64 subtraction that made d (<= eps64 (|z_src| + |z_b|) <= eps64 (2|z_b| + |d|)) the total is
    <= eps32 |d| / 2 + eps64 (3 |z_b| + 2 |d|) <= eps32 |d| + 4 eps64 (|z_b| + |d|), the bound used.
    Uniform: delta is an integer, exact in float32, so only the eps64 terms remain.  ELU's phi' is
    1-Lipschitz, so the gate differs by at most the same bound plus exp's own rounding, 2 eps64 phi'."""
    failed = []
    prows, cks, x = _states()
    worst = {}
    for a, arm in R.ARMS.items():
        if arm["kind"] is None:
            continue
        ck = cks[arm["branch"]]
        params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
        sh = R.build_shift(arm, cks)
        fwd = R.make_forward(sh)
        with torch.no_grad():
            nat = EL.forward2(params, x, R.ACT, R.ACT)
            out = fwd(params, x, torch.arange(x.shape[0]))
            if not torch.equal(nat[4], out[4]):
                failed.append(f"{a}: full-batch logits differ (max {float((nat[4] - out[4]).abs().max()):.3g})")
            xb, yb, ob, _ = _first_minibatch(ck, x)
            if not torch.equal(EL.forward2(params, xb, R.ACT, R.ACT)[4], fwd(params, xb, ob)[4]):
                failed.append(f"{a}: first-minibatch logits differ")
            L = arm["layer"]
            zb = (out[0] if L == 1 else out[2]).double()
            if not torch.equal(zb.float(), ck[f"z{L}"]):
                failed.append(f"{a}: full-batch z is not the stored branch field")
            d = sh[L].double()
            ze = zb + d
            if arm["kind"] == "field":
                target = cks[arm["src"]][f"z{L}"].double()
            else:
                target = ck[f"z{L}"].double() - arm["delta"]
            bound = EPS32 * d.abs() + 4 * EPS64 * (zb.abs() + d.abs())
            gbound = bound + 2 * EPS64 * R.ACT.dphi(target)
            err = (ze - target).abs()
            gerr = (R.ACT.dphi(ze) - R.ACT.dphi(target)).abs()
            ratio = float((err / bound.clamp(min=1e-300)).max())
            gratio = float((gerr / gbound.clamp(min=1e-300)).max())
            worst[a] = {"z_err_over_bound": ratio, "gate_err_over_bound": gratio,
                        "max_abs_d": float(d.abs().max())}
            if (err > bound).any() or (gerr > gbound).any():
                failed.append(f"{a}: argument off its target (max err/bound {ratio:.3g}, gate {gratio:.3g})")
    return {"pass": not failed, "failed_items": failed, "worst": worst,
            "n_intervention_arms": len(worst)}


S4_ARMS = ("R2_10", "R2_20", "S2_10r", "S2u10r", "S1u20r", "S2_10")


def TRAIN_DPHI(u):
    """The factor autograd applies through the host ELU: d/du expm1(u) is taken as expm1(u) + 1
    (PyTorch's formula), times 1 on the positive side.  Written here independently of the runner."""
    return torch.where(u > 0, torch.ones_like(u), torch.expm1(torch.clamp(u, max=0.0)) + 1.0)


def _manual_grads64(p64, xb, yb, ob, sh64, layer):
    """Independent float64 backprop of  a_l = phi(z0_l) + phi(z_l + d_l) - phi(z0_l + d_l)  (value at the
    branch point = phi(z0_l) = phi(z_l)) with derivative phi'(z_l + d_l); the other layer plain ELU.
    Returns the gradients of W1, b1, W2, b2, W3, b3 and their absolute-value scale (the same chain
    with every factor replaced by its absolute value)."""
    W1, b1, W2, b2, W3, b3 = p64
    z1 = xb @ W1.T + b1
    a1 = torch.where(z1 > 0, z1, torch.expm1(z1.clamp(max=0)))
    z2 = a1 @ W2.T + b2
    a2 = torch.where(z2 > 0, z2, torch.expm1(z2.clamp(max=0)))
    logits = a2 @ W3.T + b3
    dp = TRAIN_DPHI
    g1 = dp(z1 + sh64[1][ob]) if layer == 1 else dp(z1)
    g2 = dp(z2 + sh64[2][ob]) if layer == 2 else dp(z2)
    B = xb.shape[0]
    p = torch.softmax(logits, 1)
    e = (p - F.one_hot(yb, 10).double()) / B
    ea = (p + F.one_hot(yb, 10).double()) / B
    gW3, gb3 = e.T @ a2, e.sum(0)
    u2 = e @ W3
    q2 = u2 * g2
    gW2, gb2 = q2.T @ a1, q2.sum(0)
    u1 = q2 @ W2
    q1 = u1 * g1
    gW1, gb1 = q1.T @ xb, q1.sum(0)
    # absolute scale
    sW3, sb3 = ea.T @ a2.abs(), ea.sum(0)
    su2 = ea @ W3.abs()
    sq2 = su2 * g2
    sW2, sb2 = sq2.T @ a1.abs(), sq2.sum(0)
    su1 = sq2 @ W2.abs()
    sq1 = su1 * g1
    sW1, sb1 = sq1.T @ xb.abs(), sq1.sum(0)
    return [gW1, gb1, gW2, gb2, gW3, gb3], [sW1, sb1, sW2, sb2, sW3, sb3]


def s4b(R) -> dict:
    """The derivative training uses: autograd through the runner's ELU equals TRAIN_DPHI bit for bit
    in float32 and float64 on a grid over [-45, 5] (so it is 0 below the kernel's threshold, not exp(z)),
    and the runner's dphi_train is the same function.  Exact equality: same operations."""
    failed = []
    for dt_ in (torch.float32, torch.float64):
        z = torch.linspace(-45, 5, 20001, dtype=dt_).requires_grad_(True)
        g, = torch.autograd.grad(R.ACT.phi(z).sum(), z)
        ref = TRAIN_DPHI(z.detach())
        if not torch.equal(g, ref):
            failed.append(f"{dt_}: autograd != expm1 + 1 (max {float((g - ref).abs().max()):.3g})")
        if not torch.equal(R.dphi_train(z.detach()), ref):
            failed.append(f"{dt_}: runner dphi_train differs")
    z32 = torch.tensor([R.ZERO_Z32 - 1e-4, R.ZERO_Z32 + 1e-4], dtype=torch.float32)
    zz = R.dphi_train(z32)
    if not (float(zz[0]) == 0.0 and float(zz[1]) > 0.0):
        failed.append(f"float32 zero not at ZERO_Z32: {zz.tolist()}")
    return {"pass": not failed, "failed_items": failed, "zero_z32": R.ZERO_Z32}


def s4(R) -> dict:
    """The runner's forward is dtype-generic; run it in float64 (float64 state, shift field and images)
    and compare its autograd gradient with _manual_grads64, an independent chain rule whose derivative
    is TRAIN_DPHI (expm1 + 1, what autograd applies; S4b pins it).  Both are float64 evaluations of the
    same formula on bit-identical preactivations, so they differ by summation rounding only: an entry is
    a sum of <= B * max(fan_in) = 16 * 785 products, and the error enters through at most 3 layers of
    such sums, so |diff| <= 3 * 16 * 785 * eps64 * scale, scale = the absolute-value chain.  A wrong
    derivative (phi'(z) instead of phi'(z + d), a misaligned d) changes a shifted unit's gradient by
    O(1) relative, and the shifted and natural gradients must differ (non-vacuity)."""
    failed, worst = [], {}
    prows, cks, x = _states()
    x64 = x.double()
    C = 3 * 16 * 785 * EPS64
    for a in S4_ARMS:
        arm = R.ARMS[a]
        ck = cks[arm["branch"]]
        sh = R.build_shift(arm, cks)
        sh64 = {"p0": [p.double() for p in sh["p0"]], 1: None, 2: None}
        sh64[arm["layer"]] = sh[arm["layer"]].double()
        fwd64 = R.make_forward(sh64)
        p64 = [p.detach().double().requires_grad_(True) for p in ck["params"]]
        xb32, yb, ob, _ = _first_minibatch(ck, x)
        xb = x64[ob]
        out = fwd64(p64, xb, ob)
        g_run = torch.autograd.grad(F.cross_entropy(out[4], yb), p64)
        g_man, scale = _manual_grads64([p.detach() for p in p64], xb, yb, ob, sh64, arm["layer"])
        rmax = 0.0
        for i, (gr, gm, sc) in enumerate(zip(g_run, g_man, scale)):
            diff = (gr - gm).abs()
            tol = C * sc + 1e-300
            rmax = max(rmax, float((diff / tol).max()))
            if (diff > tol).any():
                failed.append(f"{a}: tensor {i} off by {float(diff.max()):.3g} (tol {float(tol.max()):.3g})")
        # non-vacuity: the natural derivative gives a different gradient for the shifted layer
        nat_g, _ = _manual_grads64([p.detach() for p in p64], xb, yb, ob,
                                   {1: torch.zeros_like(sh64[arm['layer']]), 2: torch.zeros_like(sh64[arm['layer']])},
                                   arm["layer"])
        k = 0 if arm["layer"] == 1 else 2
        rel = float((nat_g[k] - g_man[k]).abs().max() / (g_man[k].abs().max() + nat_g[k].abs().max()))
        worst[a] = {"max_diff_over_tol": rmax, "natural_vs_shifted_rel": rel}
        if rel < 1e-3:
            failed.append(f"{a}: shifted and natural gradients agree (vacuous)")
    return {"pass": not failed, "failed_items": failed, "worst": worst}


S5_ARMS = ("N10r", "N2r", "R2_10r", "S2u10r", "S2_10r")


def s5(R) -> dict:
    """A fresh Adam's first update: m = (1-b1) g, v = (1-b2) g^2, t = 1, so the update is
    lr * g / (|g| + eps) up to the float32 rounding of the 6 operations per coordinate (<= 8 eps32
    relative together), and the run's RMS is a float32 mean of N = numel squares, whose recursive sum
    is off by at most (N - 1) eps32 relative: |RMS_run - RMS_expected| <= ((N - 1) + 8) eps32 RMS_expected.
    g is recomputed here with the arm's own forward on the first minibatch (float32, same ops).
    The run's per-update RMS comes from run_arm's saved step array (1 epoch per task keeps it short)."""
    failed, got = [], {}
    prows, cks, x = _states()
    for a in S5_ARMS:
        arm = R.ARMS[a]
        ck = cks[arm["branch"]]
        units = {}
        with flush_denormal():
            R.run_arm(a, arm, cks, x, prows, 0, epochs=1, units=units)
        run_rms = units[f"{a}_k1_step"][0].astype(np.float64)
        params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]
        fwd = R.make_forward(R.build_shift(arm, cks))
        xb, yb, ob, _ = _first_minibatch(ck, x)
        out = R.forward(params, xb, ob, fwd)
        g = torch.autograd.grad(F.cross_entropy(out[4], yb), params)
        exp_rms = []
        for k in (0, 2, 4):
            u = R.LR * g[k].double() / (g[k].double().abs() + R.EPS)
            exp_rms.append(float(u.square().mean().sqrt()))
        exp_rms = np.array(exp_rms)
        numel = np.array([params[k].numel() for k in (0, 2, 4)], dtype=np.float64)
        err = np.abs(run_rms - exp_rms)
        tol = ((numel - 1) + 8) * EPS32 * exp_rms
        got[a] = {"run": run_rms.tolist(), "expected": exp_rms.tolist(),
                  "err_over_tol": (err / np.maximum(tol, 1e-300)).tolist()}
        if (err > tol).any():
            failed.append(f"{a}: first update RMS {run_rms.tolist()} vs {exp_rms.tolist()}")
    return {"pass": not failed, "failed_items": failed, "arms": got}


# --------------------------------------------------------------------------
# S6: the verdict on synthetic seeds
# --------------------------------------------------------------------------

T_TABLE = {(0.975, 9): 2.262157, (0.9875, 9): 2.685011, (0.975, 7): 2.364624, (0.9875, 7): 2.841244,
           (0.975, 4): 2.776445}


def _synth(root: Path, effects: dict, n_seeds=10, gap=0.4, invalid=(), prereg="SYNTH", dstar=12.0,
           noise=0.01, mf=0.11):
    """Arms' k=1/k=2 online accuracy = base + effect + seed noise; prefix g2 falls from t2 to t10."""
    V = CACHE["verdict_module"]
    rng = np.random.default_rng(0)
    if root.exists():
        shutil.rmtree(root)
    for s in range(n_seeds):
        d = root / "runs" / f"s{s}"
        d.mkdir(parents=True)
        base = {"N2": 0.78, "N2r": 0.78, "N10": 0.78 - gap, "N10r": 0.78 - gap}
        rows = []
        for arm in V.ARM_NAMES:
            b = base.get(arm, None)
            if b is None:
                b = 0.78 if arm.startswith("S") else 0.78 - gap
            e = b + effects.get(arm, 0.0) + noise * rng.standard_normal()
            is_nat = arm in V.N_ARMS
            for k in (1, 2):
                rows.append({"seed": s, "arm": arm, "arm_reset": arm.endswith("r"), "k": k,
                             "online_acc": e, "online_ce": 1.0, "major_frac": mf, "first75_acc": e,
                             "memo_acc_end": e, "logits_equal_full": True if k == 1 else "",
                             "logits_equal_mb": (False if (s in invalid and arm == "R2_10") else True)
                             if k == 1 else "",
                             "hash_match_prefix": True if is_nat else "",
                             "logg2_start": -1.0, "neff2_start": 0.5, "logg1_start": -1.0, "pf_eps_w2": 0.0,
                             "pf_eps_w1": 0.0, "pf_zero_w2": 0.0, "gtr2_start": 0.5, "zero2_start": 0.0,
                             "deadunits2_start": 0.0, "neffT2_start": 0.5, "gtr2_end": 0.5, "zero2_end": 0.0,
                             "step_w2": 1.0, "step_w1": 1.0, "effbar2_start": 0.0,
                             "effbar2_end": 0.0, "logg2_end": -1.0, "neff2_end": 0.5, "lowgate2_end": 0.1})
        pd.DataFrame(rows).to_csv(d / "arms.csv", index=False)
        pref = [{"seed": s, "task": t, "online_acc": 0.5, "memo_acc": 0.5,
                 "g1_end": 0.3, "g2_end": 0.7 if t <= 2 else 0.001, "logg2_end": -1.0, "neff2_end": 0.5,
                 "zbar2_end": 0.0, "sigma2_end": 1.0, "mu2_end": 1.0} for t in range(1, 23)]
        pd.DataFrame(pref).to_csv(d / "prefix.csv", index=False)
        (d / "dstar.json").write_text(json.dumps({"dstar_q": {"0.1": dstar - 1, "0.5": dstar, "0.9": dstar + 1}}))
        (d / "provenance.json").write_text(json.dumps({"prereg_commit": prereg, "flush_denormal": True,
                                                        "epochs_per_task": 80}))


def s6(V) -> dict:
    """Synthetic seeds with a known structure; the verdict must return the designed label and the
    designed means (to 1e-9: the effects are added exactly and the noise is shared by both arms of a
    contrast only through the per-arm draw, so the expected mean is recomputed here from the same draws)."""
    CACHE["verdict_module"] = V
    failed, cases = [], {}
    root = SCR / "synth"

    def case(name, effects, want, **kw):
        _synth(root / name, effects, **kw)
        sh, miss = V.load(root / name)
        res = V.analyze(sh, prereg="SYNTH", epochs=80)
        cases[name] = {"label": res["label"], "want": want,
                       "P1": res.get("P1", {}).get("mean"), "P2": res.get("P2", {}).get("mean")}
        if res["label"] != want:
            failed.append(f"{name}: {res['label']} != {want}")
        return res

    both = {"R2_10": +0.4, "S2_10r": -0.4}
    r = case("both", both, "RESPONSE_BOTH_WAYS")
    # the mean must equal the per-seed differences' mean recomputed from the files
    sh, _ = V.load(root / "both")
    d1 = np.array([V.E(sh[s]["arms"], "R2_10") - V.E(sh[s]["arms"], "N10") for s in sh])
    if abs(d1.mean() - r["P1"]["mean"]) > 1e-9:
        failed.append("both: P1 mean")
    half = T_TABLE[(0.9875, 9)] * d1.std(ddof=1) / math.sqrt(10)
    if r["P1"]["level"] != 0.975 or abs((r["P1"]["hi"] - r["P1"]["lo"]) / 2 - half) > 1e-6 * half:
        failed.append(f"both: P1 interval is not the 97.5% one (level {r['P1']['level']})")
    case("restore_only", {"R2_10": +0.4}, "RESTORE_ONLY")
    case("sink_only", {"S2_10r": -0.4}, "SINK_ONLY")
    case("none", {}, "RESPONSE_NOT_SHOWN")
    case("reversed", {"R2_10": -0.3, "S2_10r": -0.4}, "RESPONSE_REVERSED")
    case("no_gap", both, "NOT_REPRODUCED", gap=0.0)
    case("invalid", both, "INAPPLICABLE", invalid=(0, 1, 2))
    case("wrong_prereg", both, "INAPPLICABLE", prereg="OTHER")
    # ladder: F = (E - 0.1) / (E_N2r - 0.1); dstar 12 -> q0.1 = 11, q0.9 = 13: rungs 5, 10 below; 15, 20, 30 above
    lad_eff = {"S2u15r": -0.68, "S2u20r": -0.68, "S2u30r": -0.68}
    r = case("ladder_eps", {**both, **lad_eff}, "RESPONSE_BOTH_WAYS")
    if r["ladder"]["label"] != "EPS_CONSISTENT":
        failed.append(f"ladder_eps: {r['ladder']['label']}")
    r = case("ladder_early", {**both, "S2u5r": -0.68, **lad_eff}, "RESPONSE_BOTH_WAYS")
    if r["ladder"]["label"] != "DROPS_BEFORE_EPS":
        failed.append(f"ladder_early: {r['ladder']['label']}")
    r = case("ladder_late", both, "RESPONSE_BOTH_WAYS")
    if r["ladder"]["label"] != "SURVIVES_PAST_EPS":
        failed.append(f"ladder_late: {r['ladder']['label']}")
    # floor rule: mf 0.11 -> thr 0.11 + 3 sqrt(0.11 * 0.89 / 1200) = 0.137091
    thr = V.floor_thr(0.11)
    if abs(thr - (0.11 + 3 * math.sqrt(0.11 * 0.89 / 1200))) > 1e-12:
        failed.append(f"floor_thr {thr}")
    if r["ladder"]["at_floor"][20] != 0:
        failed.append("ladder_late: rung 20 counted at floor")
    # a rung between q0.1 and q0.9 (dstar 15 -> 14 .. 16) belongs to neither side
    r = case("ladder_mid", {**both, **lad_eff}, "RESPONSE_BOTH_WAYS", dstar=15.0)
    if r["ladder"]["label"] != "EPS_CONSISTENT" or r["ladder"]["below"] != [5, 10]:
        failed.append(f"ladder_mid: {r['ladder']['label']} below {r['ladder']['below']}")
    r = case("ladder_floor", {**both, **lad_eff}, "RESPONSE_BOTH_WAYS")
    if r["ladder"]["at_floor"][20] != 10 or not r["predictions"]["ladder_floor"]["hit"]:
        failed.append(f"ladder_floor: at_floor {r['ladder']['at_floor'][20]}")
    # t quantiles
    tq = {f"{p}/{df}": V.t_quantile(p, df) for (p, df) in T_TABLE}
    for (p, df), want in T_TABLE.items():
        if abs(V.t_quantile(p, df) - want) > 1e-6:
            failed.append(f"t_quantile({p},{df}) = {V.t_quantile(p, df)} != {want}")
    # the verdict's arm list is the runner's
    Rr = CACHE["real_runner"]
    if list(V.ARM_NAMES) != list(Rr.ARMS):
        failed.append("verdict ARM_NAMES != runner ARMS")
    return {"pass": not failed, "failed_items": failed, "cases": cases, "t_quantiles": tq}


# --------------------------------------------------------------------------
# S7: CLI
# --------------------------------------------------------------------------

S7_ARMS = "S2u10r,N2"


def s7(R) -> dict:
    """The CLI (subprocess, as the launcher runs it) and the in-process run_seed with the same short
    settings write identical arms.csv state hashes and online accuracies; two CLI runs agree; the
    provenance says flush on, 1 thread, this experiment and the registered commit the runner carries."""
    failed = []
    d1, d2, d3 = SCR / "s7_cli_a", SCR / "s7_cli_b", SCR / "s7_inproc"
    for d in (d1, d2, d3):
        if d.exists():
            shutil.rmtree(d)
    tmp = REPO / "src" / "_s7_runner.py"          # beside the real one, so its imports resolve the same
    for d in (d1, d2):
        cmd = [PY, str(tmp), "--seed", "0", "--epochs", "1", "--prefix-tasks", "3",
               "--arms", S7_ARMS, "--out", str(d)]
        tmp.write_text(R._source_text)
        try:
            r = subprocess.run(cmd, env=THREAD_ENV, capture_output=True, text=True, timeout=600)
        finally:
            tmp.unlink(missing_ok=True)
        if r.returncode != 0:
            failed.append(f"CLI exit {r.returncode}: {r.stderr[-300:]}")
            return {"pass": False, "failed_items": failed}
    with flush_denormal():
        R.run_seed(0, d3, MNIST, epochs=1, arms=S7_ARMS.split(","), prefix_tasks=3, save_ckpt=False,
                   progress=False)
    a1, a2, a3 = (pd.read_csv(d / "arms.csv") for d in (d1, d2, d3))
    for col in ("state_sha256", "online_acc", "online_ce"):
        if not (a1[col].equals(a2[col]) and a1[col].equals(a3[col])):
            failed.append(f"arms.csv {col} differs")
    p1 = json.loads((d1 / "provenance.json").read_text())
    if p1.get("flush_denormal") is not True:
        failed.append("provenance flush_denormal")
    if p1.get("threads") != 1:
        failed.append(f"provenance threads {p1.get('threads')}")
    if p1.get("experiment") != "resp_ee_0917":
        failed.append("provenance experiment")
    if p1.get("prereg_commit") != R.PREREG_COMMIT:
        failed.append("provenance prereg_commit")
    for f in ("prefix.csv", "arms.csv", "dstar.json", "units.npz", "provenance.json"):
        if not (d1 / f).exists():
            failed.append(f"missing {f}")
    return {"pass": not failed, "failed_items": failed, "prereg_commit": p1.get("prereg_commit"),
            "cpu_capability": p1.get("cpu_capability")}


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

def _mem_available_gib() -> float:
    return [int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()
            if l.startswith("MemAvailable:")][0] / 2 ** 20


def s_cost() -> dict:
    """Up to PROBE processes at once, each: the prefix to task 3 and two arms from t2 at full length
    (3 + 4 tasks = 42,000 updates).  Seconds per update and peak RSS per process.  The number started
    is what fits now with 6 GiB left for the desktop at 1.5 GiB per process (the S0 process's size with
    margin); this is a shared machine."""
    d = SCR / "cost"
    if d.exists():
        shutil.rmtree(d)
    probe = max(1, min(PROBE, int((_mem_available_gib() - 6.0) // 1.5)))
    procs = []
    t0 = time.time()
    for i in range(probe):
        cmd = [PY, str(RUNNER), "--seed", str(i), "--prefix-tasks", "3", "--arms", "N2,S2u20r",
               "--out", str(d / f"p{i}")]
        procs.append(subprocess.Popen(cmd, env=THREAD_ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    codes = [p.wait() for p in procs]
    wall = time.time() - t0
    provs = [json.loads((d / f"p{i}" / "provenance.json").read_text()) for i in range(probe) if codes[i] == 0]
    upd = 3 * 6000 + 2 * 2 * 6000
    per = [p["seconds_total"] / upd for p in provs]
    rss = [p["peak_rss_kb"] / 2 ** 20 for p in provs]
    avail = _mem_available_gib()
    return {"pass": all(c == 0 for c in codes), "codes": codes, "wall_s": wall,
            "sec_per_update": per, "peak_rss_gib": rss, "peak_rss_gib_max": max(rss) if rss else None,
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
    RESULTS.update({"experiment": "resp_ee_0917", "started": dt.datetime.now().isoformat(),
                    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                               text=True).stdout.strip(),
                    "runner_sha256": _sha(RUNNER), "verdict_sha256": _sha(VERDICT),
                    "checks_sha256": _sha(Path(__file__))})
    R = load(RUNNER)
    CACHE["real_runner"] = R
    want = lambda k: only is None or k in only

    if want("S0"):
        run_check("S0", "prefix = mucap_ee_0917 ref (machine-bound, seeds 0-1, tasks 1-22)", s0,
                  [("lr off by 1e-4 relative", [("LR = 1e-3\n", "LR = 1.0001e-3\n")])],
                  s0.__doc__, RUNNER)
    if want("S1"):
        run_check("S1", "update loop = host run_one (ref, ELU->ELU)", s1, [
            ("bias correction of v one step late",
             [("c1, c2 = 1 - BETA1 ** tc[0], 1 - BETA2 ** tc[0]", "c1, c2 = 1 - BETA1 ** tc[0], 1 - BETA2 ** (tc[0] + 1)")]),
            ("minibatch order reversed",
             [("order = torch.randperm(N_IMG, generator=g_batch).to(x.device)\n        xs, ys",
               "order = torch.randperm(N_IMG, generator=g_batch).to(x.device).flip(0)\n        xs, ys")]),
            ("second layer leaky in the natural forward",
             [("                out = EL.forward2(params, xb, ACT, ACT)\n",
               "                out = EL.forward2(params, xb, ACT, H.ARMS['LR'])\n")]),
            ("online accuracy miscounted",
             [("acc[s] = (out[4].detach().argmax(1) == yb).float().mean()",
               "acc[s] = (out[4].detach().argmax(1) == yb).float().mean() * 0.5")]),
        ], s1.__doc__, RUNNER)
    if want("S2"):
        run_check("S2", "natural continuations reproduce the prefix; interventions leave it", s2, [
            ("moments aliased to the branch state",
             [('adam = ([q.clone() for q in ck["m"]], [q.clone() for q in ck["v"]], [ck["tc"]])',
               'adam = (ck["m"], [q.clone() for q in ck["v"]], [ck["tc"]])')]),
            ("parameters aliased to the branch state",
             [('    params = [p.detach().clone().requires_grad_(True) for p in ck["params"]]\n    if arm["reset"]:',
               '    params = [p.detach().requires_grad_(True) for p in ck["params"]]\n    if arm["reset"]:')]),
            ("order stream restarted",
             [('g_lab, g_batch = gen_from(ck["g_lab"]), gen_from(ck["g_batch"])\n    sh = build_shift',
               'g_lab, g_batch = gen_from(ck["g_lab"]), H.stream("rl_batch", seed)\n    sh = build_shift')]),
            ("shift silently zero",
             [('    return {"p0": [p.detach().clone() for p in ck["params"]], L: d}',
               '    return {"p0": [p.detach().clone() for p in ck["params"]], L: d * 0}')]),
        ], s2.__doc__, RUNNER)
    if want("S3"):
        run_check("S3", "branch point: same logits, argument on its target", s3, [
            ("layer-2 anchor dropped",
             [("            a2 = a02 + (ACT.phi(z2 + e2) - A2)\n", "            a2 = ACT.phi(z2 + e2)\n")]),
            ("layer-1 anchor dropped",
             [("            a1 = a01 + (ACT.phi(z1 + e1) - A1)\n", "            a1 = ACT.phi(z1 + e1)\n")]),
            ("field sign flipped",
             [('d = (cks[arm["src"]][f"z{L}"].double() - ck[f"z{L}"].double()).float()',
               'd = (ck[f"z{L}"].double() - cks[arm["src"]][f"z{L}"].double()).float()')]),
            ("uniform shift sign flipped",
             [('d = torch.full_like(ck[f"z{L}"], -float(arm["delta"]))',
               'd = torch.full_like(ck[f"z{L}"], float(arm["delta"]))')]),
            ("field taken from the branch task, not the source",
             [('d = (cks[arm["src"]][f"z{L}"].double()', 'd = (cks[arm["branch"]][f"z{L}"].double()')]),
        ], s3.__doc__, RUNNER)
    if want("S4"):
        run_check("S4", "gradient = phi'(z + d) chain (float64, independent)", s4, [
            ("layer-2 derivative bypassed (forward unchanged)",
             [("            a2 = a02 + (ACT.phi(z2 + e2) - A2)\n",
               "            a2 = a02 + (ACT.phi(z2.detach() + e2) - A2) + (z2 - z2.detach())\n")]),
            ("layer-1 derivative bypassed (forward unchanged)",
             [("            a1 = a01 + (ACT.phi(z1 + e1) - A1)\n",
               "            a1 = a01 + (ACT.phi(z1.detach() + e1) - A1) + (z1 - z1.detach())\n")]),
            ("layer-2 field misaligned with the minibatch (forward unchanged at the branch point)",
             [("            e2 = d2[ob]\n", "            e2 = d2[:ob.shape[0]]\n")]),
        ], s4.__doc__, RUNNER)
    if want("S4b"):
        run_check("S4b", "training derivative = expm1 + 1 (float32 zero below -17.33)", s4b, [
            ("runner dphi_train uses exp",
             [("    return torch.where(z > 0, torch.ones_like(z), torch.expm1(z.clamp(max=0.0)) + 1)\n",
               "    return torch.where(z > 0, torch.ones_like(z), torch.exp(z.clamp(max=0.0)))\n")]),
            ("zero threshold at the round-to-nearest place", [("ZERO_Z32 = math.log(2.0 ** -24)", "ZERO_Z32 = math.log(2.0 ** -25)")]),
        ], s4b.__doc__, RUNNER)
    if want("S5"):
        run_check("S5", "fresh Adam: first update = lr g / (|g| + eps)", s5, [
            ("reset keeps the step count",
             [('        adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])\n    else:\n        adam = ([q.clone() for q in ck["m"]]',
               '        adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [ck["tc"]])\n    else:\n        adam = ([q.clone() for q in ck["m"]]')]),
            ("reset keeps v",
             [('        adam = ([torch.zeros_like(q) for q in params], [torch.zeros_like(q) for q in params], [0])\n    else:\n        adam = ([q.clone() for q in ck["m"]]',
               '        adam = ([torch.zeros_like(q) for q in params], [q.clone() for q in ck["v"]], [0])\n    else:\n        adam = ([q.clone() for q in ck["m"]]')]),
        ], s5.__doc__, RUNNER)
    if want("S6"):
        run_check("S6", "verdict on synthetic seeds", s6, [
            ("P2 sign convention flipped", [('P2 = ("N2r", "S2_10r")', 'P2 = ("S2_10r", "N2r")')]),
            ("primary level 95%", [("MAIN_LEVEL = 0.975", "MAIN_LEVEL = 0.95 ")]),
            ("floor without the binomial margin", [("FLOOR_Z = 3.0 ", "FLOOR_Z = 0.0 ")]),
            ("(B) skipped",
             [('    if not (b_gap and b_field):\n        res["label"] = "NOT_REPRODUCED"',
               '    if False:\n        res["label"] = "NOT_REPRODUCED"')]),
            ("prereg not enforced",
             [('    if prereg is not None and prov.get("prereg_commit") != prereg:',
               '    if False and prov.get("prereg_commit") != prereg:')]),
            ("ladder uses the median instead of q0.1",
             [('    below = [d for d in LADDER if d < ds["0.1"]]', '    below = [d for d in LADDER if d < ds["0.5"] + 1]')]),
            ("minibatch-logit validity ignored",
             [('        if not (_truth(r1["logits_equal_full"]) and _truth(r1["logits_equal_mb"])):',
               '        if not (_truth(r1["logits_equal_full"])):')]),
        ], s6.__doc__, VERDICT)
    if want("S7"):
        run_check("S7", "CLI = in-process run, with provenance", s7, [
            ("CLI leaves flush off",
             [("    torch.set_flush_denormal(True)\n    H.setup(\"cpu\")\n",
               "    torch.set_flush_denormal(False)\n    H.setup(\"cpu\")\n")]),
            ("CLI ignores --threads",
             [("    torch.set_num_threads(args.threads)\n", "    torch.set_num_threads(2)\n")]),
        ], s7.__doc__, RUNNER)
    if want("Scost"):
        RESULTS["S-cost"] = s_cost()
        RESULTS["S-cost"]["pass_note"] = "recorded, not a gate"
        dump()
        print("S-cost:", {k: RESULTS["S-cost"][k] for k in ("wall_s", "sec_per_update", "peak_rss_gib_max",
                                                           "slots")}, flush=True)
    RESULTS["finished"] = dt.datetime.now().isoformat()
    dump()
    print("all_pass =", RESULTS["all_pass"])


if __name__ == "__main__":
    main()

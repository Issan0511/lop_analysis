#!/usr/bin/env python3
"""Checks for neff_pred_0917 (specs/spec_neff_pred_0917.md section 7).

    python3 analysis/neff_pred_0917/checks.py
    python3 analysis/neff_pred_0917/checks.py --only S1,S4        # development

Each check runs on the real code and then on every listed mutation (an exact-once substitution in the
check's target file); every mutation must make the same check fail.  Writes
results/neff_pred_0917/checks.json.

S0a  GPU link (machine-bound): both engines, run side by side by the runner, reproduce their records for
     tasks 1-3 in every shared column (ext150 for B, layer_chimera for L), as exact floats
S0b  CPU link (machine-bound): the ee path reproduces resp_ee_0917 seed 0's prefix (state sha256, online
     accuracy, neffT) for tasks 1-3
S1   the statistics: response_stats = an independent per-unit float64 loop (dead unit 0, mixed signs,
     one-image unit 1/N, constant unit 1, subnormal float32 values); and the tensor they read is the
     factor each engine's backward multiplies by (gradients rebuilt from it are bit-equal)
S2   the branch point: in RL the start of task t+1 equals the end of task t in every statistic; in PM it
     differs in every model (the permutation is in place before the start is read)
S3   points: x and E come from row t+1, E0 from task 3, F clipped (synthetic rows with known answers)
S4   the verdict on synthetic shards: all four labels, INVALID on a missing seed / wrong prereg / dirty
S5   the calibration: rebuilt byte-identical, inputs = the committed resp_ee files, a least-squares
     optimum, LOSO coverage above the binomial 1% quantile at the nominal 95%
S6   the CLI: what the in-process path writes, with this experiment's provenance (flush on for ee,
     deterministic for gpu, git state read at start)
S-cost  seconds per task (gpu from S0a, cpu from S0b), peak RSS
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
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
sys.path.insert(0, str(REPO / "analysis" / "neff_pred_0917"))

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
from src import relu_gelu_silu_rl_0914 as B      # noqa: E402
from src import layer_chimera_rl_0914 as L       # noqa: E402

RUNNER = REPO / "src" / "neff_pred_0917.py"
VERDICT = REPO / "analysis" / "neff_pred_0917" / "verdict.py"
CALIB = REPO / "analysis" / "neff_pred_0917" / "calibrate.py"
OUT = REPO / "results" / "neff_pred_0917"
SCR = REPO / "results" / "_checks_neff_pred_0917"
RESP = REPO / "results" / "resp_ee_0917"
PY = sys.executable
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
RESULTS: dict = {}
DUMP_PATH = OUT / "checks.json"
CACHE: dict = {}
LINK_T = 3

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
    checks = {k: v for k, v in RESULTS.items() if k.startswith("S") and isinstance(v, dict) and "pass" in v
              and k != "S-cost"}
    RESULTS["all_pass"] = bool(checks) and all(
        v.get("pass") and v.get("all_mutations_detected", True) for v in checks.values())
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMP_PATH.write_text(json.dumps(RESULTS, indent=2, default=str))


def run_check(key: str, title: str, fn, mutations: list, derivation: str, target: Path) -> None:
    t0 = time.time()
    base = fn(load(target), True)
    entry = {"title": title, "target": str(target.relative_to(REPO)), "threshold_derivation": derivation,
             **base, "mutations": []}
    for label, subs in mutations:
        try:
            r = fn(load(target, subs), False)
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": bool(r["pass"]),
                                       "detected": not r["pass"], "mutant_result": brief(r)})
        except AssertionError as e:
            if "mutation anchor" in str(e):
                raise
            entry["mutations"].append({"mutation": label, "detected": True, "raised": repr(e)[:400]})
        except Exception as e:                                          # noqa: BLE001
            entry["mutations"].append({"mutation": label, "detected": True, "raised": repr(e)[:400]})
    entry["all_mutations_detected"] = all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:3]) if not entry['pass'] else ''}", flush=True)


def _git_csv(commit: str, path: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(subprocess.check_output(["git", "show", f"{commit}:{path}"], text=True)),
                       float_precision="round_trip")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# S0a / S2 / S-cost: the GPU path for LINK_T tasks
# --------------------------------------------------------------------------

def _gpu_rows(R, tag: str, tasks: int) -> tuple[pd.DataFrame, dict, float]:
    d = SCR / f"gpu_{tag}"
    if d.exists():
        shutil.rmtree(d)
    t0 = time.time()
    prov = R.run_gpu(d, tasks, "cuda", progress=False)
    wall = time.time() - t0
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return pd.read_csv(d / "rows.csv", float_precision="round_trip"), prov, wall


def s0a(R, base: bool) -> dict:
    """Exact equality (floats round-tripped through CSV) of every column the new rows share with the
    record, for all 30 (B) and 24 (L) models and tasks 1-3.  Equality is the only tolerance: the engines
    are imported unchanged and the draws are the records' own streams, so any difference is a change."""
    rows, prov, wall = _gpu_rows(R, "base" if base else f"mut{_n_loaded[0]}", LINK_T if base else 2)
    if base:
        CACHE["gpu_rows"], CACHE["gpu_wall"], CACHE["gpu_prov"] = rows, wall, prov
    T = int(rows.task.max())
    failed, detail = [], {}
    for eng, (commit, path) in (("B", ("fca473d", "results/relu_gelu_silu_rl_ext150_0914/rows.csv")),
                                ("L", ("58c1819", "results/layer_chimera_rl_0914/rows.csv"))):
        rec = _git_csv(commit, path)
        rec = rec[rec.task <= T]
        if eng == "B":
            rec = rec.rename(columns={"act": "act1"}).assign(act2=lambda d: d["act1"])
        new = rows[rows.engine == eng]
        m = new.merge(rec, on=["seed", "env", "act1", "act2", "task"], suffixes=("", "_rec"))
        cols = [c for c in rec.columns if c not in ("seed", "env", "act1", "act2", "task") and c in new.columns]
        bad = {c: int((m[c] != m[c + "_rec"]).sum()) for c in cols}
        detail[eng] = {"rows": int(len(m)), "expected": int(len(rec)), "columns": len(cols),
                       "mismatch": {k: v for k, v in bad.items() if v}}
        if len(m) != len(rec) or len(new) != len(rec) or any(bad.values()):
            failed.append(f"{eng}: {detail[eng]}")
    return {"pass": not failed, "failed_items": failed, "detail": detail, "tasks": T}


def s2(R, base: bool) -> dict:
    """Bitwise: RL start(t+1) = end(t) for every statistic and model (same images, same weights); PM
    start(t+1) != end(t) in at least one statistic for every model (a new permutation), so the RL
    equality is not a read of the same tensor twice."""
    if base and "gpu_rows" in CACHE:
        rows = CACHE["gpu_rows"]
    else:
        rows, _, _ = _gpu_rows(R, f"s2mut{_n_loaded[0]}", 2)
    keys = sorted(c[6:] for c in rows.columns if c.startswith("start_"))
    failed = []
    n_rl = n_pm = 0
    for (eng, env, a1, a2, s), d in rows.groupby(["engine", "env", "act1", "act2", "seed"]):
        d = d.set_index("task")
        for t in d.index[:-1]:
            same = [d.loc[t + 1, f"start_{k}"] == d.loc[t, f"end_{k}"] or
                    (np.isnan(d.loc[t + 1, f"start_{k}"]) and np.isnan(d.loc[t, f"end_{k}"])) for k in keys]
            if env == "RL":
                n_rl += 1
                if not all(same):
                    failed.append(f"RL {eng} {a1}/{a2} s{s} t{t}: start != end")
            else:
                n_pm += 1
                if all(same):
                    failed.append(f"PM {eng} {a1}/{a2} s{s} t{t}: start == end")
    return {"pass": not failed and n_rl > 0 and n_pm > 0, "failed_items": failed[:20], "n_rl": n_rl,
            "n_pm": n_pm, "keys": keys}


# --------------------------------------------------------------------------
# S0b: the CPU path
# --------------------------------------------------------------------------

def s0b(R, base: bool) -> dict:
    """Exact equality with resp_ee_0917 runs/s0/prefix.csv for tasks 1-3: state sha256, online accuracy
    and both layers' neffT/gtr/zero.  The ee path calls resp_ee's own run_prefix, so equality is the only
    tolerance."""
    from src import pmnist_0905 as H
    d = SCR / ("ee_base" if base else f"ee_mut{_n_loaded[0]}")
    if d.exists():
        shutil.rmtree(d)
    torch.set_flush_denormal(True)
    H.setup("cpu")
    t0 = time.time()
    try:
        prov = R.run_ee(0, d, LINK_T, progress=False)
    finally:
        torch.set_flush_denormal(False)
    wall = time.time() - t0
    new = pd.read_csv(d / "prefix.csv", float_precision="round_trip").set_index("task")
    rec = pd.read_csv(RESP / "runs" / "s0" / "prefix.csv", float_precision="round_trip").set_index("task")
    cols = ["state_sha256", "online_acc", "online_ce", "neffT1_end", "neffT2_end", "gtr1_end", "gtr2_end",
            "zero1_end", "zero2_end"]
    failed = []
    for t in range(1, LINK_T + 1):
        for c in cols:
            if t not in new.index or new.loc[t, c] != rec.loc[t, c]:
                failed.append(f"t{t} {c}")
    if base:
        CACHE["ee_wall"], CACHE["ee_prov"] = wall, prov
    return {"pass": not failed and len(new) == LINK_T, "failed_items": failed, "wall_s": wall}


# --------------------------------------------------------------------------
# S1: statistics
# --------------------------------------------------------------------------

def _loop_stats(g: torch.Tensor) -> dict:
    g = g.detach().cpu()
    M, N, U = g.shape
    out = {k: np.zeros((M, U)) for k in ("neff_abs", "neff_sgn", "gabs", "zero")}
    for m in range(M):
        for u in range(U):
            col = [float(v) for v in g[m, :, u].tolist()]
            s1a = math.fsum(abs(v) for v in col)
            s1s = math.fsum(col)
            s2 = math.fsum(v * v for v in col)
            out["neff_abs"][m, u] = s1a * s1a / (N * s2) if s2 > 0 else 0.0
            out["neff_sgn"][m, u] = s1s * s1s / (N * s2) if s2 > 0 else 0.0
            out["gabs"][m, u] = s1a / N
            out["zero"][m, u] = sum(v == 0 for v in col) / N
    return out


def s1(R, base: bool) -> dict:
    """(i) Against the per-unit loop: relative 1e-12 (float64 sums of <= 64 terms, each exact from float32;
    the loop uses fsum, the tensor path plain sums: the difference is bounded by 64 eps64 relative).
    Designed units: dead (0), constant (1), one image (1/N), mixed signs (abs != sgn), subnormal float32
    values (their squares underflow float32; a float32 path gives 0 or nan).
    (ii) Each engine's backward, rebuilt with the g the statistics read, gives bit-equal gradients."""
    failed = []
    gen = torch.Generator().manual_seed(7)
    M, N, U = 2, 64, 8
    g = torch.randn(M, N, U, generator=gen)
    g[:, :, 0] = 0.0                                   # dead
    g[:, :, 1] = 0.25                                  # constant -> 1
    g[:, :, 2] = 0.0
    g[:, 5, 2] = 0.3                                   # one image -> 1/N
    g[:, :, 3] = torch.where(torch.arange(N) % 2 == 0, 0.4, -0.1)    # mixed signs
    g[:, :, 4] = 1e-40                                 # subnormal float32, constant -> 1
    g[:, ::3, 5] = 0.0
    st = {k: v.cpu().numpy() for k, v in R.response_stats(g).items()}
    ref = _loop_stats(g)
    for k in ("neff_abs", "neff_sgn", "gabs", "zero"):
        err = np.abs(st[k] - ref[k]) / np.maximum(np.abs(ref[k]), 1e-300)
        if not np.all(err <= 1e-12):
            failed.append(f"{k} max rel err {float(err.max()):.3g}")
    expect = {(0, "neff_abs"): 0.0, (1, "neff_abs"): 1.0, (2, "neff_abs"): 1.0 / N, (4, "neff_abs"): 1.0}
    for (u, k), v in expect.items():
        if not np.allclose(st[k][:, u], v, rtol=1e-12, atol=0):
            failed.append(f"designed unit {u} {k} = {st[k][:, u]} != {v}")
    if not np.all(st["neff_sgn"][:, 3] < st["neff_abs"][:, 3]):
        failed.append("mixed-sign unit: signed n_eff not below abs")
    # (ii) the gate the statistics read is the backward's factor
    gen = torch.Generator().manual_seed(11)
    for name, mod, models in (("B", B, B.MODELS[:10]), ("L", L, L.MODELS[:8])):
        box = R.Box(name, mod, models, "cpu")
        e = box.e
        with torch.no_grad():
            for q in e.p:
                q.mul_(3.0)                                  # push some units into the negative lobe
            e.cx.copy_(torch.rand(e.cx.shape, generator=gen))
            e.cy.copy_(torch.nn.functional.one_hot(torch.randint(10, (e.M, e.cx.shape[1]), generator=gen), 10).float())
        x, y = e.cx[:, :16].clone(), e.cy[:, :16].clone()
        if name == "B":
            gr, _, _ = B.gradients(e.p, x, y, e.A)
            z1, a1, z2, a2, z3 = B.forward(e.p, x, e.A)
        else:
            gr, _, _ = L.gradients(e.p, x, y, e.elu1, e.elu2)
            z1, a1, z2, a2, z3 = L.forward(e.p, x, e.elu1, e.elu2)
        cx_full, e.cx = e.cx, x                      # the fields on exactly the tensor the backward saw
        (fz1, fg1), (fz2, fg2) = box.fields()
        e.cx = cx_full
        g3 = (z3.softmax(-1) - y) / x.shape[1]
        g2 = torch.bmm(g3, e.p[4]) * fg2
        g1 = torch.bmm(g2, e.p[2]) * fg1
        rebuilt = [torch.bmm(g1.transpose(1, 2), x), g1.sum(1), torch.bmm(g2.transpose(1, 2), a1), g2.sum(1)]
        for i, (a, b) in enumerate(zip(rebuilt, gr[:4])):
            if not torch.equal(a, b):
                failed.append(f"{name}: gradient {i} rebuilt from the read g differs")
        if name == "B" and not bool((fg2 < 0).any()):
            failed.append("B: the check never reached a negative derivative")
    return {"pass": not failed, "failed_items": failed}


# --------------------------------------------------------------------------
# S3 / S4: points and labels on synthetic shards
# --------------------------------------------------------------------------

def _cal() -> dict:
    return json.loads((OUT / "calibration.json").read_text())


def _inv(F: float, cal: dict) -> float:
    """x whose fitted F is F (for building on-curve synthetic points)."""
    F = min(max(F, 1e-6), 1 - 1e-6)
    return 10 ** (cal["c"] + cal["w"] * math.log(F / (1 - F)))


SYNTH_CLASS = ("ELU1", "R", "GELU", "SILU")


def _true_layer(a1: str, a2: str) -> int:
    return 2 if a2 in SYNTH_CLASS else (1 if a1 in SYNTH_CLASS else 2)


def _synth(root: Path, V, spec: dict, gpu_prereg="SYNTH", drop_ee_seed=False, dirty=False) -> None:
    """spec[box] = function t -> (F, x, gmean, zero) for that box's natural trajectory (all seeds alike,
    seed jitter +-0.001 in F to break ties).  Other GPU combos get an on-curve trajectory."""
    cal = _cal()["L2"]
    if root.exists():
        shutil.rmtree(root)
    (root / "gpu").mkdir(parents=True)
    e0 = 0.8

    def default(t):
        F = max(0.0, 1.0 - 0.02 * t)
        return F, _inv(F, cal) if F > 0 else 1e-9, F * 0.5, 1 - F

    rows = []
    combos = [("B", m["env"], m["act"], m["act"]) for m in B.MODELS if m["seed"] == 0] + \
             [("L", m["env"], m["act1"], m["act2"]) for m in L.MODELS if m["seed"] == 0]
    for eng, env, a1, a2 in combos:
        box = next((b for b, v in V.PRIMARY.items() if v == (eng, env, a1, a2)), None)
        fn = spec.get(box, default)
        lay = _true_layer(a1, a2)
        for s in V.GPU_SEEDS:
            for task in range(1, V.GPU_TASKS + 1):
                t = task - 1                       # this row's start is branch t-1
                F, x, gm, z = fn(max(t, 0))
                F = min(1.0, max(0.0, F + 0.001 * (s - 1) * (0 < F < 1)))
                E = 0.1 + F * (e0 - 0.1) if task != 3 else e0
                r = {"engine": eng, "seed": s, "env": env, "act1": a1, "act2": a2, "task": task,
                     "floor_acc": 0.11, "online_acc": E, "online_ce": 1.0, "finite": True}
                for li in (1, 2):
                    other = li != lay
                    for k, v in (("neff_abs", x), ("neff_sgn", x), ("gabs", gm), ("zero", z), ("tiny", z),
                                 ("neg", 0.0)):
                        val = (0.5 if k.startswith("neff") else 0.3 if k == "gabs" else 0.0) if other else v
                        r[f"start_{k}_l{li}"] = val
                        r[f"end_{k}_l{li}"] = val
                rows.append(r)
    pd.DataFrame(rows).to_csv(root / "gpu" / "rows.csv", index=False)
    (root / "gpu" / "provenance.json").write_text(json.dumps(
        {"prereg_commit": gpu_prereg, "git_dirty_code": " M src/x.py" if dirty else "", "tasks": V.GPU_TASKS}))
    fn = spec.get("RL_EE_cpu", default)
    for s in V.EE_SEEDS:
        if drop_ee_seed and s == V.EE_SEEDS[-1]:
            continue
        d = root / "ee" / f"s{s}"
        d.mkdir(parents=True)
        pr = []
        for task in range(1, V.EE_TASKS + 1):
            F, x, gm, z = fn(task)                 # prefix row t is branch t
            Fn = fn(task - 1)[0] if task > 1 else 1.0
            Fn = min(1.0, max(0.0, Fn + 0.001 * (s - 11) * (0 < Fn < 1)))
            E = 0.1 + Fn * (e0 - 0.1) if task != 3 else e0
            pr.append({"task": task, "online_acc": E, "online_ce": 1.0, "neffT1_end": 0.5, "neffT2_end": x,
                       "gtr1_end": 0.3, "gtr2_end": gm, "zero1_end": 0.0, "zero2_end": z})
        pd.DataFrame(pr).to_csv(d / "prefix.csv", index=False)
        (d / "provenance.json").write_text(json.dumps(
            {"prereg_commit": "SYNTH", "git_dirty_code": "", "tasks": V.EE_TASKS, "seed": s}))


def s3(V, base: bool) -> dict:
    """Synthetic rows whose start stats encode the task (x = 1/(10 task)), whose end stats encode
    -task, and whose online accuracy encodes the task; the points must pick x from row t+1's start,
    E from row t+1, E0 from task 3, and clip F to [0, 1].  Exact equality (constructed values)."""
    rows = []
    for task in range(1, 8):
        r = {"engine": "B", "seed": 0, "env": "PM", "act1": "GELU", "act2": "GELU", "task": task,
             "online_acc": {3: 0.6}.get(task, 0.1 + 0.05 * task if task < 7 else 0.95), "online_ce": 1.0,
             "finite": True}
        for li in (1, 2):
            for k in ("neff_abs", "neff_sgn", "gabs", "zero", "tiny", "neg"):
                r[f"start_{k}_l{li}"] = 1.0 / (10 * task) + li
                r[f"end_{k}_l{li}"] = -task
        rows.append(r)
    pts = V.gpu_points(pd.DataFrame(rows), "B", "PM", "GELU", "GELU").set_index("t")
    failed = []
    for t in range(1, 7):
        want_x = 1.0 / (10 * (t + 1)) + 2
        e = rows[t]["online_acc"]
        want_F = min(max((e - 0.1) / (0.6 - 0.1), 0.0), 1.0)
        if pts.loc[t, "neff_abs_l2"] != want_x:
            failed.append(f"t{t} x {pts.loc[t, 'neff_abs_l2']} != {want_x}")
        if pts.loc[t, "E"] != e or pts.loc[t, "E0"] != 0.6:
            failed.append(f"t{t} E/E0")
        if pts.loc[t, "F"] != want_F:
            failed.append(f"t{t} F {pts.loc[t, 'F']} != {want_F}")
    if 7 in pts.index:
        failed.append("a point at the last task")
    if not (pts["F"] <= 1).all() or not (pts["F"] >= 0).all():
        failed.append("F not clipped")
    return {"pass": not failed, "failed_items": failed}


def s4(V, base: bool) -> dict:
    """Synthetic shards with known answers (on-curve boxes, a box that falls below the band, the CPU box
    off the curve, n_eff tied with the mean derivative) -> the registered label; malformed shards ->
    INVALID.  The on-curve points sit exactly on the fitted curve, far inside any band."""
    cal_all = _cal()
    cal = cal_all["L2"]
    failed, got = [], {}

    def on_curve(t):
        F = max(0.0, 1.0 - 0.03 * t) if t < 34 else 0.0
        x = _inv(F, cal) if F > 0 else 1e-9
        return F, x, 0.2 + 0.001 * math.sin(t), 0.5 + 0.001 * math.cos(3 * t)      # g, zero uninformative

    def gelu_low(t):                                  # F falls, x stays high
        return max(0.0, 1.0 - 0.03 * t), 0.7, 0.3, 0.0

    def ee_off(t):                                    # F stays 1, x collapses
        return 1.0, 1e-6 if t >= 5 else 0.7, 0.3, 0.0

    def gelu_partial(t):                              # on the curve until t = 10, then x stays high
        F, x, g, z = on_curve(t)
        return (F, x, g, z) if t < 15 else (F, 0.7, g, z)

    def tied(t):                                      # g and zero carry the same order as x
        F, x, _, _ = on_curve(t)
        return F, x, x, -x

    cases = {
        "all_on_curve": ({b: on_curve for b in V.PRIMARY}, {}, "GENERALIZES"),
        "gelu_low": ({**{b: on_curve for b in V.PRIMARY}, "RL_GELU": gelu_low}, {}, "BOX_SPECIFIC"),
        "gelu_partial": ({**{b: on_curve for b in V.PRIMARY}, "RL_GELU": gelu_partial}, {}, "BOX_SPECIFIC"),
        "ee_off": ({**{b: on_curve for b in V.PRIMARY}, "RL_EE_cpu": ee_off}, {}, "NOT_PREDICTIVE"),
        "tied": ({**{b: on_curve for b in V.PRIMARY}, **{b: tied for b in V.OUT_OF_BOX}}, {}, "PREDICTS_NOT_BEST"),
        "missing_seed": ({b: on_curve for b in V.PRIMARY}, {"drop_ee_seed": True}, "INVALID"),
        "wrong_prereg": ({b: on_curve for b in V.PRIMARY}, {"gpu_prereg": "OTHER"}, "INVALID"),
        "dirty": ({b: on_curve for b in V.PRIMARY}, {"dirty": True}, "INVALID"),
    }
    for name, (spec, kw, want) in cases.items():
        root = SCR / f"synth_{name}"
        _synth(root, V, spec, **kw)
        sh = V.load(root, "SYNTH")
        res = V.analyze(sh, cal_all)
        got[name] = res["label"]
        if res["label"] != want:
            failed.append(f"{name}: {res['label']} != {want}")
    return {"pass": not failed, "failed_items": failed, "labels": got}


# --------------------------------------------------------------------------
# S5: calibration
# --------------------------------------------------------------------------

def _binom_q(n: int, p: float, alpha: float) -> int:
    """Largest k with P(X < k) <= alpha for X ~ Bin(n, p)."""
    cdf, k = 0.0, 0
    for j in range(n + 1):
        pj = math.comb(n, j) * p ** j * (1 - p) ** (n - j)
        if cdf + pj > alpha:
            return j
        cdf += pj
    return n


def s5(Cm, base: bool) -> dict:
    """(i) The committed calibration.json is what calibrate.py builds now (byte-equal JSON of the fitted
    entries); (ii) its input sha256 are the committed resp_ee files; (iii) the fitted (c, w) is a grid
    least-squares optimum: no step of the last grid (2e-5) in c or w lowers the SSE; (iv) the LOSO
    coverage of the primary band is at least the binomial 1% quantile of Bin(40, 0.95)."""
    failed = []
    reg = json.loads((OUT / "calibration.json").read_text())
    now = {}
    for name, key in (("L2", "neffT2_end"), ("min", "min")):
        d = Cm.natural_points(key)
        now[name] = Cm.calibrate(d)
        for k in ("c", "w", "q", "n_bin"):
            if json.dumps(now[name][k]) != json.dumps(reg[name][k]):
                failed.append(f"{name}.{k}: rebuilt {now[name][k]} != registered {reg[name][k]}")
        u, F = Cm.u_of(d["x"]), d["F"].to_numpy()
        c, w = now[name]["c"], now[name]["w"]
        sse0 = float(((Cm.logistic(u, c, w) - F) ** 2).sum())
        for dc, dw in ((2e-5, 0), (-2e-5, 0), (0, 2e-5), (0, -2e-5)):
            if float(((Cm.logistic(u, c + dc, w + dw) - F) ** 2).sum()) < sse0 - 1e-12 * sse0:
                failed.append(f"{name}: SSE lower at ({dc}, {dw})")
    a26 = Cm.arms26()
    for k in ("c", "w", "q"):
        if json.dumps(a26[k]) != json.dumps(reg["arms26"][k]):
            failed.append(f"arms26.{k} differs")
    for rel, h in reg["inputs_sha256"].items():
        if _sha(REPO / rel) != h:
            failed.append(f"input changed: {rel}")
    lo = _binom_q(40, 0.95, 0.01)
    cov = now["L2"]["loso"]["inside"]
    if now["L2"]["loso"]["n"] != 40 or cov < lo:
        failed.append(f"LOSO {cov}/{now['L2']['loso']['n']} below {lo}")
    return {"pass": not failed, "failed_items": failed, "loso_inside": cov, "loso_floor": lo}


# --------------------------------------------------------------------------
# S6: CLI
# --------------------------------------------------------------------------

def s6(R, base: bool) -> dict:
    """The CLI's shards equal the in-process ones (task 1: ee state sha256 = resp_ee s0; gpu rows = the
    S0a rows for task 1), and provenance says: flush on (ee), deterministic and no TF32 (gpu), git state
    read at start, this spec and prereg field present."""
    failed = []
    d = SCR / ("cli_base" if base else f"cli_mut{_n_loaded[0]}")
    if d.exists():
        shutil.rmtree(d)
    tmp_mod = REPO / "src" / "_chk_neff_cli.py"
    tmp_mod.write_text(R.__dict__.get("__source__") or RUNNER.read_text())
    try:
        for args in (["ee", "--seed", "0", "--tasks", "1", "--out", str(d / "ee")],
                     ["gpu", "--tasks", "1", "--out", str(d / "gpu")]):
            p = subprocess.run([PY, "-m", "src._chk_neff_cli", *args], env=THREAD_ENV, capture_output=True,
                               text=True)
            if p.returncode != 0:
                failed.append(f"{args[0]} exit {p.returncode}: {p.stderr[-300:]}")
    finally:
        tmp_mod.unlink(missing_ok=True)
    if failed:
        return {"pass": False, "failed_items": failed}
    pe = json.loads((d / "ee" / "provenance.json").read_text())
    pg = json.loads((d / "gpu" / "provenance.json").read_text())
    if pe.get("flush_denormal") is not True:
        failed.append("ee flush_denormal not on")
    if pe.get("threads") != 1:
        failed.append(f"ee threads {pe.get('threads')}")
    if pg.get("deterministic") is not True or pg.get("tf32") is not False:
        failed.append("gpu not deterministic / tf32 on")
    for p, n in ((pe, "ee"), (pg, "gpu")):
        if p.get("git_state_read") != "at start" or "prereg_commit" not in p or p.get("spec") != "specs/spec_neff_pred_0917.md":
            failed.append(f"{n} provenance fields")
    ee = pd.read_csv(d / "ee" / "prefix.csv", float_precision="round_trip")
    rec = pd.read_csv(RESP / "runs" / "s0" / "prefix.csv", float_precision="round_trip")
    if ee.loc[0, "state_sha256"] != rec.loc[0, "state_sha256"]:
        failed.append("ee task 1 state differs from resp_ee s0")
    g = pd.read_csv(d / "gpu" / "rows.csv", float_precision="round_trip")
    ref = CACHE.get("gpu_rows")
    if ref is not None:
        r1 = ref[ref.task == 1].reset_index(drop=True)
        cols = [c for c in g.columns if c in r1.columns]
        if not g[cols].equals(r1[cols]):
            failed.append("gpu CLI task 1 != in-process task 1")
    return {"pass": not failed, "failed_items": failed}


def s6_mutant_loader(subs):
    """S6 runs the CLI in a subprocess, so the mutant must exist as a file: load() gives the module, and
    the mutated source is carried along for s6 to write out."""
    src = RUNNER.read_text()
    for old, new in subs:
        assert src.count(old) == 1, f"mutation anchor occurs {src.count(old)} times: {old!r}"
        src = src.replace(old, new)
    mod = load(RUNNER, subs)
    mod.__dict__["__source__"] = src
    return mod


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

def s_cost() -> dict:
    gp = CACHE.get("gpu_prov") or {}
    ep = CACHE.get("ee_prov") or {}
    return {"gpu_sec_per_task": CACHE.get("gpu_wall", float("nan")) / LINK_T,
            "gpu_peak_rss_gib": gp.get("peak_rss_kb", 0) / 2 ** 20,
            "ee_sec_per_task_t1_3": CACHE.get("ee_wall", float("nan")) / LINK_T,
            "ee_peak_rss_gib": ep.get("peak_rss_kb", 0) / 2 ** 20,
            "resp_ee_prefix_sec_per_task": json.loads((RESP / "runs" / "s0" / "provenance.json").read_text())[
                "seconds_prefix"] / 22,
            "note": "the ee box slows as layer 2 sinks; resp_ee's 22-task prefix is the reference"}


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
    RESULTS.update({"experiment": "neff_pred_0917", "started": dt.datetime.now().isoformat(),
                    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                               text=True).stdout.strip(),
                    "runner_sha256": _sha(RUNNER), "verdict_sha256": _sha(VERDICT),
                    "calibrate_sha256": _sha(CALIB), "checks_sha256": _sha(Path(__file__)),
                    "calibration_sha256": _sha(OUT / "calibration.json")})
    want = lambda k: only is None or k in only

    if want("S0a"):
        run_check("S0a", "GPU engines = their records (tasks 1-3, all shared columns)", s0a, [
            ("perm drawn before the batch orders",
             [("        epochs_needed = -(-(STEPS * B.BATCH) // N_IMG)\n        orders = {}\n",
               "        epochs_needed = -(-(STEPS * B.BATCH) // N_IMG)\n        _ = {s: torch.randperm(784, generator=gperm[s]) for s in seeds}\n        orders = {}\n")]),
            ("PM trained on random labels",
             [('                yy = ys[s] if m["env"] == "PM" else labels[s]\n',
               '                yy = labels[s]\n')]),
            ("RL labels from the next seed's stream",
             [('    glabel = {s: B.stream("env_labels_0913", s) for s in seeds}',
               '    glabel = {s: B.stream("env_labels_0913", s + 1) for s in seeds}')]),
        ], s0a.__doc__, RUNNER)
    if want("S0b"):
        run_check("S0b", "CPU ee path = resp_ee_0917 s0 prefix (tasks 1-3)", s0b, [
            ("seed offset", [("RE.run_prefix(seed, mnist, RE.EPOCHS, tasks", "RE.run_prefix(seed + 10, mnist, RE.EPOCHS, tasks")]),
            ("one epoch short", [("RE.run_prefix(seed, mnist, RE.EPOCHS, tasks", "RE.run_prefix(seed, mnist, RE.EPOCHS - 1, tasks")]),
        ], s0b.__doc__, RUNNER)
    if want("S1"):
        run_check("S1", "statistics = per-unit loop; they read the backward's factor", s1, [
            ("abs sum uses signed g", [("s1a, s1s, s2 = ga.sum(1), g64.sum(1), g64.square().sum(1)",
                                        "s1a, s1s, s2 = g64.sum(1), g64.sum(1), g64.square().sum(1)")]),
            ("float32 arithmetic", [("    g64 = g.double()\n", "    g64 = g.float()\n")]),
            ("dead unit counted as 1", [('    zero = torch.zeros_like(s2)\n', '    zero = torch.ones_like(s2)\n')]),
            ("N missing from the denominator", [("    den = (n * s2).clamp_min(1e-300)\n", "    den = s2.clamp_min(1e-300)\n")]),
            ("B gate of layer 2 read on layer 1",
             [("            return (z1, B.gate(z1, e.A)), (z2, B.gate(z2, e.A))",
               "            return (z1, B.gate(z2, e.A)), (z2, B.gate(z2, e.A))")]),
            ("L layer masks swapped",
             [("        return (z1, L.gate(z1, e.elu1)), (z2, L.gate(z2, e.elu2))",
               "        return (z1, L.gate(z1, e.elu2)), (z2, L.gate(z2, e.elu1))")]),
            ("L layer-1 derivative floored at 0.2",
             [("        return (z1, L.gate(z1, e.elu1)), (z2, L.gate(z2, e.elu2))",
               "        return (z1, L.gate(z1, e.elu1).clamp_min(0.2)), (z2, L.gate(z2, e.elu2))")]),
        ], s1.__doc__, RUNNER)
    if want("S2"):
        run_check("S2", "RL start(t+1) = end(t); PM start is on the new permutation", s2, [
            ("statistics read after the task instead of before (weights moved)",
             [("            st0, un0 = b.stats()                   # task start: this task's inputs, weights not yet moved\n            e.acc.zero_()\n",
               "            e.acc.zero_()\n"),
              ("            met, _ = e.evaluate()\n            st1, _ = b.stats()                     # task end\n",
               "            met, _ = e.evaluate()\n            st1, _ = b.stats()                     # task end\n            st0, un0 = b.stats()\n")]),
            ("start read before the new inputs are loaded",
             [("            orderstack = torch.stack([orders[m[\"seed\"]] for m in b.models], 1).to(dev)\n",
               "            orderstack = torch.stack([orders[m[\"seed\"]] for m in b.models], 1).to(dev)\n            _pre = b.stats()\n"),
              ("            st0, un0 = b.stats()                   # task start: this task's inputs, weights not yet moved\n",
               "            st0, un0 = _pre if task > 1 else b.stats()\n")]),
        ], s2.__doc__, RUNNER)
    if want("S3"):
        run_check("S3", "points: x, E from row t+1; E0 from task 3; F clipped", s3, [
            ("x from row t", [("            r = ds.loc[t + 1]\n", "            r = ds.loc[t + 1]\n            r = r.copy(); r[[c for c in r.index if c.startswith('start_')]] = ds.loc[t, [c for c in r.index if c.startswith('start_')]]\n")]),
            ("E0 from task 2", [('        e0 = float(ds.loc[3, "online_acc"])\n        for t in ds.index[:-1]:\n            r = ds.loc[t + 1]',
                                 '        e0 = float(ds.loc[2, "online_acc"])\n        for t in ds.index[:-1]:\n            r = ds.loc[t + 1]')]),
            ("F not clipped", [("    return min(max((e - CHANCE) / (e0 - CHANCE), 0.0), 1.0)",
                                "    return (e - CHANCE) / (e0 - CHANCE)")]),
        ], s3.__doc__, VERDICT)
    if want("S4"):
        run_check("S4", "verdict on synthetic shards", s4, [
            ("pass share 0.5", [("PASS_SHARE = 0.8\n", "PASS_SHARE = 0.5\n")]),
            ("EL box read on its leaky layer", [("    return 2 if a2 in CLASS else (1 if a1 in CLASS else None)",
                                                  "    return 2 if a2 in CLASS or a2 == 'LR' else (1 if a1 in CLASS else None)")]),
            ("(b) ignored", [('        res["label"] = "GENERALIZES" if res["b"]["pass"] else "PREDICTS_NOT_BEST"',
                              '        res["label"] = "GENERALIZES"')]),
            ("CPU box not required", [('    if not boxes["RL_EE_cpu"]["pass"]:\n        res["label"] = "NOT_PREDICTIVE"',
                                        '    if False:\n        res["label"] = "NOT_PREDICTIVE"')]),
            ("prereg not enforced", [("        if prereg is not None and gprov.get(\"prereg_commit\") != prereg:",
                                      "        if False:")]),
            ("dirty start accepted", [('        if gprov.get("git_dirty_code"):\n            problems.append("gpu run started with uncommitted code")',
                                        '        if False:\n            problems.append("gpu run started with uncommitted code")')]),
        ], s4.__doc__, VERDICT)
    if want("S5"):
        run_check("S5", "calibration rebuilt, optimal, LOSO-covered", s5, [
            ("fit from t = 1", [("T_FIT = tuple(range(2, 22))", "T_FIT = tuple(range(1, 22))")]),
            ("E0 from task 2", [('        e0 = float(p.loc[3, "online_acc"])', '        e0 = float(p.loc[2, "online_acc"])')]),
            ("band at 50%", [("LEVEL = 0.95\n", "LEVEL = 0.50\n")]),
            ("coarse grid only", [("    for step_c, step_w in ((0.01, 0.01), (0.0005, 0.0005), (0.00002, 0.00002)):",
                                   "    for step_c, step_w in ((0.01, 0.01),):")]),
        ], s5.__doc__, CALIB)
    if want("S6"):
        t0 = time.time()
        base = s6(load(RUNNER), True)
        muts = []
        for label, subs in (
                ("CLI leaves flush off", [("        torch.set_flush_denormal(True)\n        H.setup(\"cpu\")\n        run_ee",
                                           "        torch.set_flush_denormal(False)\n        H.setup(\"cpu\")\n        run_ee")]),
                ("CLI leaves determinism off", [("        torch.use_deterministic_algorithms(True)\n        torch.set_num_threads(1)\n        run_gpu",
                                                 "        torch.use_deterministic_algorithms(False)\n        torch.set_num_threads(1)\n        run_gpu")]),
                ("CLI ignores --threads", [("        torch.set_num_threads(args.threads)\n", "        torch.set_num_threads(2)\n")])):
            try:
                r = s6(s6_mutant_loader(subs), False)
                muts.append({"mutation": label, "detected": not r["pass"], "mutant_result": brief(r)})
            except Exception as e:                                          # noqa: BLE001
                muts.append({"mutation": label, "detected": True, "raised": repr(e)[:300]})
        RESULTS["S6"] = {"title": "CLI = in-process, with provenance", "target": str(RUNNER.relative_to(REPO)),
                         "threshold_derivation": s6.__doc__, **base, "mutations": muts,
                         "all_mutations_detected": all(m["detected"] for m in muts),
                         "seconds": round(time.time() - t0, 1)}
        dump()
        print(f"S6: pass={base['pass']}  mutations detected {sum(m['detected'] for m in muts)}/{len(muts)}"
              f"{'  FAILED ' + str(base['failed_items'][:3]) if not base['pass'] else ''}", flush=True)
    if want("Scost"):
        RESULTS["S-cost"] = s_cost()
        dump()
        print("S-cost:", RESULTS["S-cost"], flush=True)
    RESULTS["finished"] = dt.datetime.now().isoformat()
    dump()
    print("all_pass =", RESULTS["all_pass"])


if __name__ == "__main__":
    main()

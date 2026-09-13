"""Checks for src/shell_l2_rlmnist_0913 (specs/spec_shell_l2_rlmnist_0913.md section 5).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python analysis/shell_l2_rlmnist_0913/checks.py
    ... --only S3,S6        # development: writes results/_checks_shell_l2_rlmnist_0913/checks_partial.json

S1-S10 are each run on the real code and then on every mutation listed with them.  A mutation is an
exact-once string substitution in the runner (or in the verdict script) loaded as a separate module --
or, for the CLI check S5, executed as a script -- and it must make the same check FAIL.  Tolerances are
fixed by the derivations in the docstrings (spec section 5 table), not by looking at the values.
Then the smoke test (S-smoke) and the cost measurement (S-cost, the launch gate).

Writes results/shell_l2_rlmnist_0913/checks.json after every step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import heapq
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

torch.set_num_threads(1)
from src import pmnist_0905 as H                 # noqa: E402
from src import pmnist_rlmnist_0906 as RL        # noqa: E402

RUNNER = REPO / "src" / "shell_l2_rlmnist_0913.py"
VERDICT = REPO / "analysis" / "shell_l2_rlmnist_0913" / "verdict.py"
OUT = REPO / "results" / "shell_l2_rlmnist_0913"
SCR = REPO / "results" / "_checks_shell_l2_rlmnist_0913"
SMOKE = REPO / "results" / "_smoke_shell_l2_rlmnist_0913"
PY = sys.executable
BASE_COMMIT = "33a0cab"                           # byte-identical import of the 0906 box
EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)
LAM = 1e-3
SPEC_EPS2 = 1e-12                                 # the spec's constant, independent of the module's
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")
ARMS8 = [(a, r) for a in ("R", "SNA") for r in ("none", "l2:1e-3", "l2init:1e-3", "shell:1e-3")]
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
DEADLINE_JST = dt.datetime(2026, 9, 14, 13, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))  # spec 5 S-cost
CPU = torch.device("cpu")
DEV = H.setup("cpu")
MNIST = H.Mnist(DEV)
RESULTS: dict = {}
DUMP_PATH = OUT / "checks.json"

# --------------------------------------------------------------------------
# infrastructure
# --------------------------------------------------------------------------

_n_loaded = [0]


def load(path: Path, subs=()):
    """Import `path` as a fresh module after exact-once substitutions (the real code when subs is empty)."""
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


def sha(x) -> str:
    h = hashlib.sha256()
    for t in (x if isinstance(x, (list, tuple)) else [x]):
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def dump() -> None:
    checks = {k: v for k, v in RESULTS.items() if k.startswith("S")}
    RESULTS["all_pass"] = bool(checks) and all(
        v.get("pass") and v.get("all_mutations_detected", True) for v in checks.values())
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMP_PATH.write_text(json.dumps(RESULTS, indent=2, default=str))


def brief(r: dict) -> dict:
    """What a mutant's check result looked like, without the bulky per-arm detail."""
    return {k: v for k, v in r.items() if k != "pass" and not isinstance(v, (dict, list))} | \
        {"failed_items": r.get("failed_items", [])[:8]}


def run_check(key: str, title: str, fn, target: Path, mutations: list, derivation: str) -> None:
    t0 = time.time()
    base = fn(load(target))
    entry = {"title": title, "threshold_derivation": derivation, **base, "mutations": []}
    for label, subs in mutations:
        mod = load(target, subs)          # an anchor that does not match exactly once aborts the whole run
        try:
            r = fn(mod)
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": bool(r["pass"]),
                                       "detected": not r["pass"], "mutant_result": brief(r)})
        except Exception as e:           # recorded, but a crash is not how a check should notice a defect
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
    entry["all_mutations_detected"] = bool(entry["mutations"]) and all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)",
          flush=True)


# ---- anchors in src/shell_l2_rlmnist_0913.py (each must occur exactly once) ----
A_STREAMS = '    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)\n'
A_INIT = '    params = H.init_params(seed, device)              # host init: bit-identical per seed\n'
A_SUBSET = '    idx = RL.subset_idx(seed).to(device)\n'
A_R0 = '        r0, zero0 = shell_reference(ref0)\n'
A_GSHELL = '                    g_shell = torch.autograd.grad(r_shell, params)\n'
A_NORM = '    return torch.sqrt((p * p).sum() + SHELL_EPS2)\n'
A_TERM = '        t = lam * (p * p).sum() if z else lam * (shell_norm(p) - r) ** 2\n'
A_PENALTY_BODY = ('    total = None\n'
                  '    for p, r, z in zip(params, r0, zero0):\n'
                  '        t = lam * (p * p).sum() if z else lam * (shell_norm(p) - r) ** 2\n'
                  '        total = t if total is None else total + t\n'
                  '    return total\n')
A_L2 = '                        grads = [gr + 2.0 * reg.lam * (q - (p0[i] if p0 is not None else 0.0))\n'
A_P0 = '    p0 = [q.detach().clone() for q in params] if reg.kind == "l2init" else None\n'
A_EPS = 'SHELL_EPS2 = 1e-12\n'
A_ADD = '                        grads = [gr + gs for gr, gs in zip(grads, g_shell)]\n'
A_RATIO = '        ratio = n / n0 if n0 > 0 else float("nan")\n'
A_PEN_L2INIT = '               "l2init": LAM_REF * dist2,\n'
A_LAYER_LOOP = '    for name, p, q0 in zip(TENSORS, params, ref0):\n'


# --------------------------------------------------------------------------
# S1  (spec: mandatory sanity 1)
# --------------------------------------------------------------------------

def s1(M) -> dict:
    """8 arms x seeds {0,1} x 3 tasks x 2 epochs: init / subset / labels / batch orders captured inside
    run_one are bit-identical across the 8 arms and to an independent regeneration from the host streams.
    Threshold: bit equality -- the property claimed is bit identity."""
    failed, per = [], {}
    for seed in (0, 1):
        ref = {"init": sha(H.init_params(seed, CPU)), "subset": sha(RL.subset_idx(seed))}
        g = H.stream("rl_labels", seed)
        ref["labels"] = [sha(RL.task_labels(g)) for _ in range(3)]
        g = H.stream("rl_batch", seed)
        ref["orders"] = [sha(torch.randperm(RL.N_IMAGES, generator=g)) for _ in range(3 * 2)]
        seen = set()
        for act, reg in ARMS8:
            d = {}
            M.run_one(act, reg, seed, 1e-3, 3, MNIST, DEV, epochs=2, debug=d)
            got = {"init": sha(d["init"]), "subset": sha(d["subset"]),
                   "labels": [sha(y) for y in d["labels"]], "orders": [sha(o) for o in d["orders"]]}
            same = {k: got[k] == ref[k] for k in ref}
            per[f"seed{seed}|{act}|{reg}"] = same
            failed += [f"seed{seed}|{act}|{reg}|{k}" for k, v in same.items() if not v]
            seen.add(json.dumps(got, sort_keys=True))
        per[f"seed{seed}|distinct_across_8_arms"] = len(seen)
        if len(seen) != 1:
            failed.append(f"seed{seed}: {len(seen)} distinct stream sets across the 8 arms")
    return {"pass": not failed, "failed_items": failed, "per_arm": per, "seeds": [0, 1], "tasks": 3, "epochs": 2}


S1_MUT = [
    ("M1a: shell arm draws one extra batch permutation before task 1",
     [(A_STREAMS, A_STREAMS + '    if reg.kind == "shell":\n        torch.randperm(N_IMAGES, generator=g_batch)\n')]),
    ("M1b: l2 arm's init moved by 1 ulp in one element",
     [(A_INIT, A_INIT + '    if reg_s.startswith("l2:"):\n        with torch.no_grad():\n'
                        '            params[0].view(-1)[0] = torch.nextafter(params[0].view(-1)[0], torch.tensor(1.0))\n')]),
    ("M1c: SNA arms advance the label stream by one task before task 1",
     [(A_STREAMS, A_STREAMS + '    if act_name == "SNA":\n        RL.task_labels(g_lab)\n')]),
    ("M1d: l2init arm draws its 1200 images from seed+1",
     [(A_SUBSET, '    idx = RL.subset_idx(seed + int(reg_s.startswith("l2init"))).to(device)\n')]),
]


# --------------------------------------------------------------------------
# S2  (mandatory sanity 2)
# --------------------------------------------------------------------------

def s2(M) -> dict:
    """Shell arm, seeds 0-9 x {R, SNA}: R_shell and its six gradients captured at task 1 step 0, before any
    update, are exactly 0.  Threshold: exact zero -- r0 comes from the same function applied to the same
    bits, so ||p|| - r0 is 0 without rounding."""
    failed, per = [], {}
    for act in ("R", "SNA"):
        for seed in range(10):
            d = {"capture": {(1, 0)}}
            M.run_one(act, "shell:1e-3", seed, 1e-3, 1, MNIST, DEV, epochs=1, debug=d)
            c = d["shell"][(1, 0)]
            loss = float(c["loss"])
            gmax = max(float(g.abs().max()) for g in c["grads"])
            per[f"{act}|seed{seed}"] = {"loss": loss, "max_abs_grad": gmax}
            if loss != 0.0 or gmax != 0.0:
                failed.append(f"{act}|seed{seed}: loss={loss:.3e} max|g|={gmax:.3e}")
    return {"pass": not failed, "failed_items": failed, "per_run": per}


S2_MUT = [("M2: shell anchor (r0) taken from the init of seed+1",
           [(A_R0, '        r0, zero0 = shell_reference(H.init_params(seed + 1, device))\n')])]


# --------------------------------------------------------------------------
# S3  (mandatory sanity 3 + addendum A2: per tensor, coefficient 2*lambda)
# --------------------------------------------------------------------------

def tol_par(eps_d: float, k: int = 1) -> float:
    """Parallelism: every gradient element is (upstream scalar) * p_i with k correctly rounded elementwise
    multiplies (k = 1: p*p backs off as g*p_i twice and the two copies add exactly to 2*g*p_i).  Errors in
    the scalar are common to all elements and cannot tilt g away from p.  With ratios taken in float64,
    max/min of g_i/p_i <= (1 + k*eps_d/2)/(1 - k*eps_d/2) -> spread <= k*eps_d*(1 + k*eps_d) + 4*eps64."""
    return k * eps_d * (1 + k * eps_d) + 4 * EPS64


def tol_coef(N: int, n: float, r0: float) -> float:
    """Coefficient: a float64 sum of N non-negative terms is within (N+1)/2*eps of its exact value, the norm
    within half that plus eps; the difference ||p|| - r0 amplifies absolute errors by (n + r0)/|n - r0|; the
    fsum reference and the scalar ops add a few eps."""
    return ((N + 1) / 2 + 1) * EPS64 * ((n + r0) / abs(n - r0) + 1) + 8 * EPS64


def parallel_stats(g: torch.Tensor, p: torch.Tensor):
    g64, p64 = g.detach().double().reshape(-1), p.detach().double().reshape(-1)
    nz = p64 != 0
    zero_ok = bool((g64[~nz] == 0).all())
    r = (g64[nz] / p64[nz]).numpy()
    med = float(np.median(r))
    spread = float((r.max() - r.min()) / abs(med)) if med != 0 else float("inf")
    return med, spread, zero_ok


def fsum_norm(t: torch.Tensor) -> float:
    a = t.detach().double().reshape(-1).numpy()
    return math.sqrt(math.fsum((a * a).tolist()) + SPEC_EPS2)


def s3_states(seed: int):
    """float64 states: each init tensor rescaled to radius ratio 1.25 or 0.8 (alternating so the tensors
    disagree and a single global norm cannot mimic them) and tilted to cos 0.9 by an orthogonal component."""
    p0 = [q.detach().double().clone() for q in H.init_params(seed, CPU)]
    gen = torch.Generator().manual_seed(3000 + seed)
    ps = []
    for q, t in zip(p0, (1.25, 0.8, 0.8, 1.25, 1.25, 0.8)):
        u = q.reshape(-1) / q.norm()
        w = torch.randn(q.numel(), dtype=torch.float64, generator=gen)
        w = w - (w @ u) * u
        w = w / w.norm()
        ps.append((t * q.norm() * (0.9 * u + math.sqrt(1 - 0.81) * w)).reshape(q.shape))
    return p0, ps


def s3(M) -> dict:
    """(a) formula in float64 on constructed states: g parallel to p (tol_par, k=1) and coefficient equal to
    the per-tensor analytic 2*lam*(||p|| - r0)/||p|| from math.fsum (tol_coef).  (b) the float32 raw shell
    gradient a real run adds at its last step is parallel to p in all six tensors (tol_par(eps32)) and is
    bit-identical to shell_penalty's autograd gradient at the captured weights."""
    failed, per = [], {}
    for seed in (0, 1, 2):
        p0, ps = s3_states(seed)
        r0, zero0 = M.shell_reference(p0)
        P = [q.clone().requires_grad_(True) for q in ps]
        G = torch.autograd.grad(M.shell_penalty(P, r0, zero0, LAM), P)
        for name, g, p, q0 in zip(TENSORS, G, ps, p0):
            med, spread, zero_ok = parallel_stats(g, p)
            n, rr = fsum_norm(p), fsum_norm(q0)
            c_an = 2 * LAM * (n - rr) / n
            rel = abs(med - c_an) / abs(c_an)
            tp, tc = tol_par(EPS64), tol_coef(p.numel(), n, rr)
            per[f"a|seed{seed}|{name}"] = {"spread": spread, "tol_par": tp, "coef_rel_err": rel, "tol_coef": tc,
                                           "coef": med, "coef_analytic": c_an}
            if not (spread <= tp and zero_ok):
                failed.append(f"a|seed{seed}|{name}: not parallel (spread {spread:.3e} > {tp:.3e})")
            if not rel <= tc:
                failed.append(f"a|seed{seed}|{name}: coefficient rel err {rel:.3e} > {tc:.3e}")
    for act in ("R", "SNA"):
        d = {"capture": {(2, 149)}}
        M.run_one(act, "shell:1e-3", 0, 1e-3, 2, MNIST, DEV, epochs=2, debug=d)
        c = d["shell"][(2, 149)]
        for name, g, p in zip(TENSORS, c["grads"], c["params"]):
            med, spread, zero_ok = parallel_stats(g, p)
            tp = tol_par(EPS32)
            per[f"b|{act}|{name}"] = {"spread": spread, "tol_par": tp, "coef": med}
            if not (spread <= tp and zero_ok):
                failed.append(f"b|{act}|{name}: not parallel (spread {spread:.3e} > {tp:.3e})")
        r0, zero0 = M.shell_reference([q.detach() for q in H.init_params(0, CPU)])
        P = [q.clone().requires_grad_(True) for q in c["params"]]
        G = torch.autograd.grad(M.shell_penalty(P, r0, zero0, LAM), P)
        same = all(torch.equal(a, b) for a, b in zip(G, c["grads"]))
        per[f"b|{act}|runner_grad_equals_function_grad"] = same
        if not same:
            failed.append(f"b|{act}: the runner's raw shell gradient is not shell_penalty's gradient")
    return {"pass": not failed, "failed_items": failed, "detail": per}


S3_MUT = [
    ("M3a: the loop adds the l2init gradient 2*lam*(p - p0) in place of the shell gradient",
     [(A_GSHELL, '                    g_shell = [2.0 * reg.lam * (q.detach() - q0) for q, q0 in zip(params, ref0)]\n')]),
    ("M3a': shell norm taken per row (dim=-1) instead of per tensor",
     [(A_NORM, '    return torch.sqrt((p * p).sum(-1) + SHELL_EPS2)\n'),
      (A_TERM, '        t = lam * (p * p).sum() if z else lam * ((shell_norm(p) - r) ** 2).sum()\n')]),
    ("M3b: one norm over all tensors concatenated",
     [(A_PENALTY_BODY, '    n_all = torch.sqrt(sum((p * p).sum() for p in params) + SHELL_EPS2)\n'
                       '    r_all = torch.sqrt(sum(r * r for r in r0))\n'
                       '    return lam * (n_all - r_all) ** 2\n')]),
    ("M3c: penalty in the lam/2 convention",
     [(A_TERM, '        t = lam * (p * p).sum() if z else 0.5 * lam * (shell_norm(p) - r) ** 2\n')]),
]


# --------------------------------------------------------------------------
# S4  (mandatory sanity 4)
# --------------------------------------------------------------------------

def s4(M) -> dict:
    """(R, SNA) x (none, l2, l2init) x seed 0 x 3 tasks x 3 epochs: the new runner and the untouched 0906
    RL.run_one give bit-identical weights at every task end and identical per_task metrics; the host and the
    0906 runner are byte-identical to their blobs in 33a0cab.  Threshold: bit equality."""
    failed, per = [], {}
    for f in ("src/pmnist_0905.py", "src/pmnist_rlmnist_0906.py"):
        blob = subprocess.run(["git", "show", f"{BASE_COMMIT}:{f}"], capture_output=True, check=True).stdout
        same = hashlib.sha256(blob).hexdigest() == hashlib.sha256((REPO / f).read_bytes()).hexdigest()
        per[f"blob|{f}"] = same
        if not same:
            failed.append(f"{f} differs from {BASE_COMMIT}")
    for act in ("R", "SNA"):
        for iv in ("none", "l2:1e-3", "l2init:1e-3"):
            old_hashes = []
            orig = RL.evaluate_rl

            def wrapped(params, x, y, a, _orig=orig, _acc=old_hashes):
                _acc.append(sha(params))
                return _orig(params, x, y, a)

            RL.evaluate_rl = wrapped
            try:
                rows_old, _ = RL.run_one(act, 0, 1e-3, 3, MNIST, DEV, optimizer="adam", epochs=3,
                                         c=0.6, beta=0.01, iv=iv)
            finally:
                RL.evaluate_rl = orig
            d = {}
            rows_new, _, _ = M.run_one(act, iv, 0, 1e-3, 3, MNIST, DEV, epochs=3, c=0.6, beta=0.01, debug=d)
            new_hashes = [sha(ps) for ps in d["task_end_params"]]
            cols = [k for k in rows_old[0] if k not in ("arm", "seed", "lr", "task", "iv")]
            diff_cols = sorted({k for ro, rn in zip(rows_old, rows_new) for k in cols
                                if not (ro[k] == rn.get(k) or (ro[k] != ro[k] and rn.get(k) != rn.get(k)))})
            ok = len(old_hashes) == 3 and old_hashes == new_hashes and len(rows_old) == len(rows_new) and not diff_cols
            per[f"{act}|{iv}"] = {"weights_equal_each_task": old_hashes == new_hashes, "n_task_ends": len(old_hashes),
                                  "metric_columns_compared": len(cols), "differing_columns": diff_cols}
            if not ok:
                failed.append(f"{act}|{iv}: weights_equal={old_hashes == new_hashes} differing={diff_cols[:5]}")
    return {"pass": not failed, "failed_items": failed, "detail": per}


S4_MUT = [
    ("M4a: l2/l2init gradient coefficient 2.0 -> 1.0 in the new runner",
     [(A_L2, A_L2.replace("gr + 2.0 * reg.lam", "gr + 1.0 * reg.lam"))]),
    ("M4b: l2init anchor p0 aliases the live weights (detach without clone)",
     [(A_P0, '    p0 = [q.detach() for q in params] if reg.kind == "l2init" else None\n')]),
]


# --------------------------------------------------------------------------
# S5  (mandatory sanity 5; CLI, separate processes)
# --------------------------------------------------------------------------

STUB = r'''
import json, sys
path, subs = sys.argv[1], json.loads(sys.argv[2])
src = open(path).read()
for old, new in subs:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
sys.argv = [path] + sys.argv[3:]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
'''


def cli(out: Path, act: str, reg: str, seeds: str, tasks: int, epochs: int, subs=None,
        wait: bool = True):
    args = ["--act", act, "--reg", reg, "--seeds", seeds, "--tasks", str(tasks), "--epochs", str(epochs),
            "--out", str(out)]
    cmd = [PY, str(RUNNER), *args] if subs is None else [PY, "-c", STUB, str(RUNNER), json.dumps(subs), *args]
    if not wait:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=THREAD_ENV)
    r = subprocess.run(cmd, capture_output=True, text=True, env=THREAD_ENV)
    if r.returncode != 0:
        raise RuntimeError(f"runner failed: {r.stdout[-800:]} {r.stderr[-1500:]}")


def s5_run(subs=None, tag="real") -> dict:
    """Shell arm, R and SNA, seed 0, 2 tasks x 5 epochs, twice in separate processes on this VM and .venv:
    per_task.csv and layer_metrics.csv byte-identical and the final state sha256 identical.
    Threshold: byte / bit equality."""
    failed, per = [], {}
    for act in ("R", "SNA"):
        outs = [SCR / "_runs" / f"repro_{tag}_{act}_{k}" for k in ("a", "b")]
        for o in outs:
            shutil.rmtree(o, ignore_errors=True)
            cli(o, act, "shell:1e-3", "0", 2, 5, subs)
        same = {f: (outs[0] / f).read_bytes() == (outs[1] / f).read_bytes() for f in ("per_task.csv", "layer_metrics.csv")}
        states = [json.loads((o / "provenance.json").read_text())["runs"]["0"]["final_state_sha256"] for o in outs]
        same["final_state_sha256"] = states[0] == states[1]
        per[act] = {**same, "per_task_sha256": hashlib.sha256((outs[0] / "per_task.csv").read_bytes()).hexdigest()[:16],
                    "state_sha256": states[0][:16]}
        failed += [f"{act}|{k}" for k, v in same.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


def s5() -> None:
    t0 = time.time()
    base = s5_run()
    entry = {"title": "S5 S-repro: same arm/seed twice in separate processes -> identical CSVs and state",
             "threshold_derivation": "byte/bit equality (determinism claim)", **base, "mutations": []}
    subs = [(A_STREAMS, '    g_lab, g_batch = H.stream("rl_labels", seed), '
                        'torch.Generator().manual_seed(time.time_ns() % (1 << 62))\n')]
    r = s5_run(subs, tag="mutM5")
    entry["mutations"].append({"mutation": "M5: batch stream seeded from time.time_ns()",
                               "check_pass_on_mutant": r["pass"], "detected": not r["pass"], "mutant_result": brief(r)})
    entry["all_mutations_detected"] = all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS["S5_S-repro"] = entry
    dump()
    print(f"S5_S-repro: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)", flush=True)


# --------------------------------------------------------------------------
# S6  (addendum A2: the p0 == 0 rule)
# --------------------------------------------------------------------------

def s6(M) -> dict:
    """A tensor whose anchor is identically 0: the term is lam*sum p^2 (value within (N+2)*eps64 relative:
    one fsum-vs-torch sum), its gradient is 2*lam*p (parallel within tol_par(eps64), coefficient within the
    same bound since no sum enters the gradient), and value and gradients stay finite at p = 0 -- for the
    zero-anchor tensor and for a non-zero-anchor tensor -- in float64 and float32."""
    failed, per = [], {}
    gen = torch.Generator().manual_seed(6)
    z0 = torch.zeros(1000, dtype=torch.float64)
    nz0 = torch.rand(50, dtype=torch.float64, generator=gen) - 0.5
    p_z = torch.randn(1000, dtype=torch.float64, generator=gen) * 0.1
    r0, zero0 = M.shell_reference([z0, nz0])
    per["zero_flags"] = zero0
    if zero0 != [True, False]:
        failed.append(f"zero-anchor flags {zero0} != [True, False]")
    P = p_z.clone().requires_grad_(True)
    T = M.shell_penalty([P], [r0[0]], [zero0[0]], LAM)
    (g,) = torch.autograd.grad(T, [P])
    ref = LAM * math.fsum((p_z.numpy() ** 2).tolist())
    val_rel = abs(float(T.detach()) - ref) / ref
    med, spread, zero_ok = parallel_stats(g, p_z)
    coef_rel = abs(med - 2 * LAM) / (2 * LAM)
    tv, tp = (p_z.numel() + 2) * EPS64, tol_par(EPS64)
    per.update(value_rel_err=val_rel, tol_value=tv, grad_spread=spread, grad_coef_rel_err=coef_rel, tol_par=tp)
    if not val_rel <= tv:
        failed.append(f"zero-anchor value rel err {val_rel:.3e} > {tv:.3e}")
    if not (spread <= tp and zero_ok and coef_rel <= tp):
        failed.append(f"zero-anchor gradient not 2*lam*p (spread {spread:.3e}, coef err {coef_rel:.3e})")
    for dtype in (torch.float64, torch.float32):
        r0d, zd = M.shell_reference([z0.to(dtype), nz0.to(dtype)])
        Pz = [torch.zeros(1000, dtype=dtype, requires_grad=True), torch.zeros(50, dtype=dtype, requires_grad=True)]
        T0 = M.shell_penalty(Pz, r0d, zd, LAM)
        G0 = torch.autograd.grad(T0, Pz)
        fin = math.isfinite(float(T0.detach())) and all(bool(torch.isfinite(q).all()) for q in G0)
        per[f"finite_at_p0_{str(dtype).split('.')[-1]}"] = fin
        if not fin:
            failed.append(f"non-finite value or gradient at p = 0 ({dtype})")
    return {"pass": not failed, "failed_items": failed, "detail": per}


S6_MUT = [
    ("M6a: the p0 == 0 term is dropped (T = 0)",
     [(A_TERM, '        t = 0.0 * (p * p).sum() if z else lam * (shell_norm(p) - r) ** 2\n')]),
    ("M6b: eps^2 = 0 (sqrt backward is 0/0 at p = 0)", [(A_EPS, 'SHELL_EPS2 = 0.0\n')]),
]


# --------------------------------------------------------------------------
# S7  (spec 2.3: eps is inert at init)
# --------------------------------------------------------------------------

def s7(M) -> dict:
    """float32 sqrt(S + eps^2) is bit-identical to sqrt(S) for all 6 init tensors of seeds 0-9.
    Threshold: bit equality (spec 2.3: guaranteed for S > eps^2 * 2**25; min S at init is 0.021)."""
    changed, n, min_s = [], 0, math.inf
    for seed in range(10):
        for name, q in zip(TENSORS, H.init_params(seed, CPU)):
            q = q.detach()
            S = (q * q).sum()
            min_s = min(min_s, float(S))
            n += 1
            if not torch.equal(M.shell_norm(q), torch.sqrt(S)):
                changed.append(f"seed{seed}|{name}")
    return {"pass": not changed and n == 60, "failed_items": changed, "n_tensors": n, "min_sum_sq_at_init": min_s,
            "inert_above": SPEC_EPS2 * 2 ** 25}


S7_MUT = [("M7: eps^2 = 1e-4", [(A_EPS, 'SHELL_EPS2 = 1e-4\n')])]


# --------------------------------------------------------------------------
# S8  (the shell gradient is actually wired into the update)
# --------------------------------------------------------------------------

def s8(M) -> dict:
    """Shell arm, R, seed 0, 1 task x 1 epoch (75 steps) replayed by hand -- host forward, CE, the module's
    shell_penalty, hand-written Adam with the runner's op order -- gives weights bit-identical to the
    runner's, and those differ from the none arm's (the regulariser acted).  Threshold: bit equality."""
    failed = []
    d, dn = {}, {}
    M.run_one("R", "shell:1e-3", 0, 1e-3, 1, MNIST, DEV, epochs=1, debug=d)
    M.run_one("R", "none", 0, 1e-3, 1, MNIST, DEV, epochs=1, debug=dn)
    runner, none_w = d["task_end_params"][0], dn["task_end_params"][0]
    act = H.ARMS["R"]
    P = H.init_params(0, DEV)
    r0, zero0 = M.shell_reference([q.detach().clone() for q in P])
    x = MNIST.train_x[RL.subset_idx(0).to(DEV)]
    y = RL.task_labels(H.stream("rl_labels", 0)).to(DEV)
    order = torch.randperm(RL.N_IMAGES, generator=H.stream("rl_batch", 0)).to(DEV)
    xs, ys = x[order], y[order]
    mom = [torch.zeros_like(q) for q in P]
    vel = [torch.zeros_like(q) for q in P]
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-3
    for j in range(RL.STEPS_PER_EPOCH):
        xb, yb = xs[j * RL.BATCH:(j + 1) * RL.BATCH], ys[j * RL.BATCH:(j + 1) * RL.BATCH]
        o = H.forward(P, xb, act)
        gce = torch.autograd.grad(torch.nn.functional.cross_entropy(o[4], yb), P)
        gsh = torch.autograd.grad(M.shell_penalty(P, r0, zero0, LAM), P)
        with torch.no_grad():
            c1, c2 = 1 - b1 ** (j + 1), 1 - b2 ** (j + 1)
            for p, ga, gb, mi, vi in zip(P, gce, gsh, mom, vel):
                gt = ga + gb
                mi.mul_(b1).add_(gt, alpha=1 - b1)
                vi.mul_(b2).addcmul_(gt, gt, value=1 - b2)
                p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
    replay_equal = all(torch.equal(p.detach().cpu(), q) for p, q in zip(P, runner))
    differs = not all(torch.equal(a, b) for a, b in zip(runner, none_w))
    if not replay_equal:
        failed.append("hand replay of the shell update != runner weights")
    if not differs:
        failed.append("shell arm weights == none arm weights (regulariser had no effect)")
    return {"pass": not failed, "failed_items": failed, "replay_bit_identical": replay_equal,
            "differs_from_none": differs, "steps": RL.STEPS_PER_EPOCH}


S8_MUT = [
    ("M8a: the runner does not add the shell gradient", [(A_ADD, A_ADD.replace("gr + gs", "gr"))]),
    ("M8b: the runner subtracts the shell gradient", [(A_ADD, A_ADD.replace("gr + gs", "gr - gs"))]),
]


# --------------------------------------------------------------------------
# S9  (logging)
# --------------------------------------------------------------------------

def s9_reference(p: torch.Tensor, p0: torch.Tensor, kind: str, lam_own: float) -> tuple[dict, dict]:
    """Independent numpy / math.fsum recomputation and the per-quantity rounding tolerance of two float64
    routes (torch sums vs fsum): a sum of N same-sign terms is within N*eps relative; the dot product can
    cancel, so its absolute bound uses sum|a*b|; pen_shell and pen_angular contain differences, so their
    bounds are absolute."""
    a, b = p.double().numpy().reshape(-1), p0.double().numpy().reshape(-1)
    N = a.size
    e = N * EPS64 + 8 * EPS64
    sq, sq0 = math.fsum((a * a).tolist()), math.fsum((b * b).tolist())
    n, n0 = math.sqrt(sq), math.sqrt(sq0)
    dot, absdot = math.fsum((a * b).tolist()), math.fsum(np.abs(a * b).tolist())
    dd = a - b
    dist2 = math.fsum((dd * dd).tolist())
    cos = dot / (n * n0)
    pen = {"l2": LAM * sq, "l2init": LAM * dist2, "shell": LAM * (n - n0) ** 2}
    ref = {"sq_norm": sq, "norm": n, "norm0": n0, "norm_ratio": n / n0, "cos_w0": cos,
           "dist_w0": math.sqrt(dist2), "pen_l2": pen["l2"], "pen_l2init": pen["l2init"],
           "pen_shell": pen["shell"], "pen_angular": 2 * LAM * n * n0 * (1 - cos),
           "reg_loss": 0.0 if kind == "none" else pen[kind] * (lam_own / LAM)}
    tol_cos = N * EPS64 * absdot / (n * n0) + abs(cos) * 2 * e + 8 * EPS64
    e_abs = e * (n + n0)
    tol = {"sq_norm": e * sq, "norm": e * n, "norm0": e * n0, "norm_ratio": 2 * e * (n / n0),
           "cos_w0": tol_cos, "dist_w0": e * math.sqrt(dist2), "pen_l2": e * pen["l2"],
           "pen_l2init": e * pen["l2init"],
           "pen_shell": LAM * (2 * abs(n - n0) * e_abs + e_abs ** 2) + 8 * EPS64 * pen["shell"],
           "pen_angular": 2 * LAM * n * n0 * (tol_cos + (1 - cos) * 2 * e + 8 * EPS64)}
    tol["reg_loss"] = 0.0 if kind == "none" else {"l2": tol["pen_l2"], "l2init": tol["pen_l2init"],
                                                  "shell": tol["pen_shell"]}[kind]
    return ref, tol


def s9(M) -> dict:
    """R, seed 0, 2 tasks x 2 epochs, all four regularisers: every layer_metrics value and the per_task
    penalty totals equal the independent recomputation from the weights captured at init and at each task
    end, within s9_reference's tolerances."""
    failed, worst = [], {}
    for reg in ("none", "l2:1e-3", "l2init:1e-3", "shell:1e-3"):
        kind = reg.split(":")[0]
        d = {}
        rows, lrows, _ = M.run_one("R", reg, 0, 1e-3, 2, MNIST, DEV, epochs=2, debug=d)
        states = [d["init"]] + d["task_end_params"]
        if len(lrows) != 3 * len(TENSORS):
            failed.append(f"{reg}: {len(lrows)} layer rows != 18")
            continue
        totals = {}
        for lr_ in lrows:
            i = TENSORS.index(lr_["tensor"])
            ref, tol = s9_reference(states[lr_["task"]][i], states[0][i], kind, 1e-3)
            for k in ref:
                err = abs(lr_[k] - ref[k])
                worst[f"{reg}|{k}"] = max(worst.get(f"{reg}|{k}", 0.0), err / tol[k] if tol[k] > 0 else (0.0 if err == 0 else math.inf))
                if not err <= tol[k]:
                    failed.append(f"{reg}|task{lr_['task']}|{lr_['tensor']}|{k}: |{lr_[k]:.12g} - {ref[k]:.12g}| > {tol[k]:.3e}")
            tt = totals.setdefault(lr_["task"], {"reg_loss": [0.0, 0.0], "pen_l2": [0.0, 0.0],
                                                 "pen_l2init": [0.0, 0.0], "pen_shell": [0.0, 0.0]})
            for k in tt:
                tt[k][0] += ref[k]
                tt[k][1] += tol[k]
        for row in rows:
            tt = totals[row["task"]]
            for k, col in (("reg_loss", "reg_loss"), ("pen_l2", "pen_l2_total"),
                           ("pen_l2init", "pen_l2init_total"), ("pen_shell", "pen_shell_total")):
                bound = tt[k][1] + 8 * EPS64 * abs(tt[k][0])
                if not abs(row[col] - tt[k][0]) <= bound:
                    failed.append(f"{reg}|task{row['task']}|{col}: |{row[col]:.12g} - {tt[k][0]:.12g}| > {bound:.3e}")
    return {"pass": not failed, "failed_items": failed, "worst_err_over_tol": worst}


S9_MUT = [
    ("M9a: norm_ratio logged as ||p0||/||p||", [(A_RATIO, '        ratio = n0 / n if n0 > 0 else float("nan")\n')]),
    ("M9b: pen_l2init logged with lam/2", [(A_PEN_L2INIT, '               "l2init": 0.5 * LAM_REF * dist2,\n')]),
    ("M9c: b2 is logged against the neighbouring tensor's anchor (b1's p0)",
     [(A_LAYER_LOOP, '    for name, p, q0 in zip(TENSORS, params, [ref0[0], ref0[1], ref0[2], ref0[1], ref0[4], ref0[5]]):\n')]),
]


# --------------------------------------------------------------------------
# S10  (the verdict script on synthetic shards with known answers)
# --------------------------------------------------------------------------

REGS = ("none", "l2", "l2init", "shell")
SEED_EFFECT = np.array([0.010, -0.008, 0.004, -0.002, 0.007, -0.006, 0.001, -0.009, 0.005, -0.003])
JITTER = np.array([0.0004, -0.0003, 0.0002, -0.0004, 0.0001, 0.0003, -0.0002, -0.0001, 0.0004, -0.0004])


def synth(base: dict, d2_seed=None, w3_shell=1.0, bias_l2init=1.0, drop=(), decoy30=-0.4):
    """One activation's worth of shards (the same pattern is written for R and SNA).  Window values are
    base[reg] + seed effect + a per-(reg, seed) jitter; d2_seed overrides shell - l2init per seed.
    Task 30 of shell carries a large decoy so a window shifted to 30-49 changes the answer.
    Norm ratios: weights 1.00 (shell) vs 1.02 (l2init) except W3 of shell = w3_shell; biases of l2init =
    bias_l2init."""
    pts, lms, provs, missing = [], [], {}, []
    for act in ("R", "SNA"):
        for ri, reg in enumerate(REGS):
            for seed in range(10):
                name = f"{act}_{reg}_s{seed}"
                if (reg, seed) in drop:
                    missing.append(name)
                    continue
                w = base[reg] + SEED_EFFECT[seed] + JITTER[(seed + 3 * ri) % 10]
                if reg == "shell" and d2_seed is not None:
                    w = base["l2init"] + SEED_EFFECT[seed] + JITTER[(seed + 6) % 10] + d2_seed[seed]
                for t in range(1, 51):
                    v = w + (decoy30 if (reg == "shell" and t == 30) else 0.0) + (0.2 if t <= 29 else 0.0)
                    pts.append({"act": act, "reg": reg, "seed": seed, "task": t, "online_acc": v, "memo_acc": v})
                for t in range(0, 51):
                    for tn in TENSORS:
                        nr = 1.0
                        if t > 0 and reg == "l2init":
                            nr = bias_l2init if tn.startswith("b") else 1.02
                        if t > 0 and reg == "shell" and tn == "W3":
                            nr = w3_shell
                        lms.append({"act": act, "reg": reg, "seed": seed, "task": t, "tensor": tn, "norm_ratio": nr,
                                    "cos_w0": 0.9, "sq_norm": 1.0, "pen_l2": 1e-3, "pen_l2init": 2e-4,
                                    "pen_shell": 1e-4, "reg_loss": 1e-4})
                provs[name] = {"runs": {str(seed): {"init_sha256": "i", "subset_idx_sha256": "s",
                                                    "labels_sha256": "l", "batch_sha256": "b",
                                                    "divergence": {"diverged": False}}}}
    return pd.DataFrame(pts), pd.DataFrame(lms), provs, missing


def s10_scenarios():
    A = {"none": 0.300, "l2": 0.940, "l2init": 0.960, "shell": 0.958}
    ambig = np.array([+0.0012 if s % 2 == 0 else -0.0012 for s in range(10)]) - 0.005
    return [
        ("A", synth(A, bias_l2init=1.40), ("A", "")),
        ("B", synth({**A, "shell": 0.948}), ("B", "")),
        ("C", synth({**A, "shell": 0.972}), ("C", "")),
        ("D_NORM_MATCH_FAIL", synth(A, w3_shell=1.30), ("D", "NORM_MATCH_FAIL")),
        ("D_CI_AMBIGUOUS", synth(A, d2_seed=ambig), ("D", "CI_AMBIGUOUS")),
        ("D_EQUIV_GAIN_UNRESOLVED", synth({"none": 0.3, "l2": 0.960, "l2init": 0.963, "shell": 0.9625}),
         ("D", "EQUIV_GAIN_UNRESOLVED")),
        ("D_INCOMPLETE", synth(A, drop={("shell", 9)}), ("D", "INCOMPLETE")),
    ]


def s10(V) -> dict:
    """Seven synthetic datasets, each built so that exactly one row of the spec 4.4 table applies; the
    verdict script must return that label and reason code for both activations.  Threshold: equality."""
    failed, per = [], {}
    for name, (pt, lm, provs, missing), want in s10_scenarios():
        res = V.analyze(pt, lm, provs, missing)
        for act in ("R", "SNA"):
            got = (res["acts"][act]["label"], res["acts"][act]["reason"])
            per[f"{name}|{act}"] = {"want": list(want), "got": list(got),
                                    "d2_ci95": [res["acts"][act]["contrasts"]["d2"]["ci95_lo"],
                                                res["acts"][act]["contrasts"]["d2"]["ci95_hi"]]}
            if got != want:
                failed.append(f"{name}|{act}: got {got} want {want}")
    return {"pass": not failed, "failed_items": failed, "detail": per}


S10_MUT = [
    ("M10a: window shifted to tasks 30-49", [("WIN = (31, 50)\n", "WIN = (30, 49)\n")]),
    ("M10b: d2 taken as l2init - shell",
     [('CONTRASTS = (("d1", "shell", "l2"), ("d2", "shell", "l2init"), ("d3", "l2init", "l2"))\n',
       'CONTRASTS = (("d1", "shell", "l2"), ("d2", "l2init", "shell"), ("d3", "l2init", "l2"))\n')]),
    ("M10c: norm-match guard judged on the biases", [("        g_ok, g_w = guard(lm, act, WEIGHTS)\n",
                                                     "        g_ok, g_w = guard(lm, act, BIASES)\n")]),
    ("M10d: equivalence band 0.05", [("DELTA = 0.005                      # equivalence band on d2 (spec 4.1)\n",
                                     "DELTA = 0.05                       # equivalence band on d2 (spec 4.1)\n")]),
]


# --------------------------------------------------------------------------
# smoke and cost
# --------------------------------------------------------------------------

def smoke() -> None:
    """8 arms x seeds {0,1} x 2 tasks x 3 epochs through the CLI in parallel, then the verdict script with
    an explicit --src: six output files, 32 task rows, 288 layer rows, finite accuracies and norms, shell
    penalty 0 at task 0, and both activations D / INCOMPLETE (no window exists)."""
    t0 = time.time()
    shutil.rmtree(SMOKE, ignore_errors=True)
    runs = SMOKE / "runs"
    runs.mkdir(parents=True)
    procs = []
    for act, reg in ARMS8:
        for seed in (0, 1):
            name = f"{act}_{reg.split(':')[0]}_s{seed}"
            procs.append((name, cli(runs / name, act, reg, str(seed), 2, 3, wait=False)))
    rc = {name: p.wait() for name, p in procs}
    failed = [f"{n}: exit {c}" for n, c in rc.items() if c != 0]
    r = subprocess.run([PY, str(VERDICT), "--src", str(runs), "--out", str(SMOKE), "--checks", str(DUMP_PATH)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        failed.append(f"verdict.py exit {r.returncode}: {r.stderr[-800:]}")
    files = ["per_task.csv", "layer_metrics.csv", "verdict.csv", "summary.md", "provenance.json", "checks.json"]
    present = {f: (SMOKE / f).exists() for f in files}
    failed += [f"missing {f}" for f, ok in present.items() if not ok]
    detail = {"files": present, "verdict_stdout": r.stdout[-600:]}
    if all(present.values()):
        pt, lm = pd.read_csv(SMOKE / "per_task.csv"), pd.read_csv(SMOKE / "layer_metrics.csv")
        vd = pd.read_csv(SMOKE / "verdict.csv")
        labels = {row.act: (row.label, row.reason) for row in vd[vd.kind == "label"].itertuples()}
        sh0 = lm[(lm.reg == "shell") & (lm.task == 0)].reg_loss
        detail.update(n_task_rows=len(pt), n_layer_rows=len(lm), labels={k: list(v) for k, v in labels.items()},
                      shell_task0_reg_loss_max=float(sh0.abs().max()))
        if len(pt) != 32:
            failed.append(f"{len(pt)} task rows != 32")
        if len(lm) != 288:
            failed.append(f"{len(lm)} layer rows != 288")
        if not np.isfinite(pt.online_acc).all() or not np.isfinite(lm[["norm", "norm_ratio", "cos_w0"]].to_numpy()).all():
            failed.append("non-finite accuracy or norm")
        if len(sh0) != 2 * 2 * len(TENSORS) or float(sh0.abs().max()) != 0.0:   # 2 acts x 2 seeds x 6 tensors
            failed.append(f"shell reg_loss at task 0 is not exactly 0 on all 24 rows ({len(sh0)} rows)")
        if labels != {"R": ("D", "INCOMPLETE"), "SNA": ("D", "INCOMPLETE")}:
            failed.append(f"labels {labels} != D/INCOMPLETE for both")
    RESULTS["S-smoke"] = {"pass": not failed, "failed_items": failed, "detail": detail,
                          "seconds": round(time.time() - t0, 1)}
    dump()
    print(f"S-smoke: pass={not failed}  ({RESULTS['S-smoke']['seconds']}s) {failed[:3]}", flush=True)


def cost() -> None:
    """16 concurrent runner processes (2 of each arm), 1 task x 100 epochs each: seconds per 7,500 steps
    under the planned load -> 80 runs x 50 tasks x 30,000 steps on 16 slots, in the launch order."""
    t0 = time.time()
    root = SCR / "_runs" / "cost"
    shutil.rmtree(root, ignore_errors=True)
    procs = []
    for k in range(2):
        for act, reg in ARMS8:
            name = f"{act}_{reg.split(':')[0]}_{k}"
            procs.append(((act, reg.split(":")[0]), name, cli(root / name, act, reg, "0", 1, 100, wait=False)))
    rc = [p.wait() for _, _, p in procs]
    per_arm = {}
    for (arm, name, _), c in zip(procs, rc):
        if c == 0:
            w = json.loads((root / name / "provenance.json").read_text())["runs"]["0"]["wall_clock_s"]
            per_arm.setdefault(f"{arm[0]}_{arm[1]}", []).append(w)
    ms_step = {a: 1e3 * float(np.mean(v)) / (RL.STEPS_PER_EPOCH * 100) for a, v in per_arm.items()}
    # one real run = 50 tasks x 4x the timed task (the timed wall includes its one evaluation, so this
    # over-counts evaluation 4-fold: conservative) plus ~5 s start-up
    run_s = {a: 50 * 4 * float(np.mean(v)) + 5.0 for a, v in per_arm.items()}
    queue = [f"{act}_{reg}" for seed in range(10) for act in ("R", "SNA") for reg in REGS]
    slots = [0.0] * 16
    heapq.heapify(slots)
    finish = 0.0
    for a in queue:
        start = heapq.heappop(slots)
        end = start + run_s[a]
        finish = max(finish, end)
        heapq.heappush(slots, end)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    eta = now + dt.timedelta(seconds=finish)
    ok = len(rc) == 16 and all(c == 0 for c in rc) and eta <= DEADLINE_JST
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": f"projected finish <= {DEADLINE_JST.isoformat()}",
                         "concurrent_processes": 16, "ms_per_step_under_16_load": ms_step,
                         "projected_run_minutes": {a: s / 60 for a, s in run_s.items()},
                         "projected_wall_hours_80_runs_16_slots": finish / 3600,
                         "measured_at": now.isoformat(), "projected_finish_if_launched_now": eta.isoformat(),
                         "seconds": round(time.time() - t0, 1)}
    dump()
    print(f"S-cost: pass={ok}  ms/step {ms_step}  80 runs -> {finish / 3600:.2f} h  ETA {eta:%m-%d %H:%M} JST", flush=True)


# --------------------------------------------------------------------------

def main() -> None:
    global DUMP_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of check keys (development; writes a partial file)")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    if only:
        DUMP_PATH = SCR / "checks_partial.json"
    RESULTS.update({"run_id": "shell_l2_rlmnist_0913", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                    "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (RUNNER, VERDICT, Path(__file__).resolve())},
                    "torch": torch.__version__, "threads": torch.get_num_threads(),
                    "eps32": EPS32, "eps64": EPS64})
    (SCR / "_runs").mkdir(parents=True, exist_ok=True)

    def want(k):
        return only is None or k in only

    table = [
        ("S1", "S1_S-identical", "8 arms: bit-identical init, 1200 images, labels, batch orders", s1, RUNNER, S1_MUT,
         "bit equality"),
        ("S2", "S2_S-shell-zero-init", "shell penalty and gradient exactly 0 at init", s2, RUNNER, S2_MUT,
         "exact zero (same function, same bits)"),
        ("S3", "S3_S-shell-grad-parallel", "raw shell gradient parallel to each tensor, per-tensor coefficient 2*lam",
         s3, RUNNER, S3_MUT, tol_par.__doc__ + " || " + tol_coef.__doc__),
        ("S4", "S4_S-l2-unchanged", "new runner == untouched 0906 runner for none/l2/l2init", s4, RUNNER, S4_MUT,
         "bit equality"),
        ("S6", "S6_S-zero0-rule", "p0 == 0 tensors: lam*||p||^2, finite at p = 0", s6, RUNNER, S6_MUT, s6.__doc__),
        ("S7", "S7_S-eps-inert", "eps^2 changes no bit of the init norms", s7, RUNNER, S7_MUT, s7.__doc__),
        ("S8", "S8_S-shell-wiring", "hand replay of the shell update == runner", s8, RUNNER, S8_MUT, "bit equality"),
        ("S9", "S9_S-log", "logged norms / cos / penalties == independent recomputation", s9, RUNNER, S9_MUT,
         s9_reference.__doc__),
        ("S10", "S10_S-verdict-synthetic", "verdict script returns the known label on synthetic shards", s10, VERDICT,
         S10_MUT, "equality of label and reason code"),
    ]
    for short, key, title, fn, target, muts, deriv in table:
        if want(short):
            run_check(key, f"{short} {title}", fn, target, muts, deriv)
        if short == "S4" and want("S5"):
            s5()
    if want("smoke"):
        smoke()
    if want("cost"):
        cost()
    RESULTS["finished_at"] = dt.datetime.now().astimezone().isoformat()
    dump()
    print(f"all_pass = {RESULTS['all_pass']}  -> {DUMP_PATH}")


if __name__ == "__main__":
    main()

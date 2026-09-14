"""Checks for src/l2split_rlmnist_0914 (specs/spec_l2split_rlmnist_0914.md section 6).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/l2split_rlmnist_0914/checks.py
    ... --only S2,S3        # development: writes results/_checks_l2split_rlmnist_0914/checks_partial.json

S1-S7 are each run on the real code and then on every mutation listed with them.  A mutation is an
exact-once string substitution in the runner (or in the verdict script) loaded as a separate module --
or, for the CLI check S5, executed as a script -- and it must make the same check FAIL.  Tolerances follow
the derivations in the docstrings (spec section 6), not the values.  Then S-smoke (which also runs the
reuse pipeline end to end against shards written by the untouched wcap runner) and S-cost.  Every step
that starts several processes first waits until MemAvailable leaves 4 GiB after ~1.3 GiB per process.

Writes results/l2split_rlmnist_0914/checks.json after every step.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import heapq
import json
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
from src import shell_l2_rlmnist_0913 as SH      # noqa: E402
from src import wcap_rlmnist_0914 as WC          # noqa: E402

RUNNER = REPO / "src" / "l2split_rlmnist_0914.py"
WC_RUNNER = REPO / "src" / "wcap_rlmnist_0914.py"
VERDICT = REPO / "analysis" / "l2split_rlmnist_0914" / "verdict.py"
OUT = REPO / "results" / "l2split_rlmnist_0914"
SCR = REPO / "results" / "_checks_l2split_rlmnist_0914"
SMOKE = REPO / "results" / "_smoke_l2split_rlmnist_0914"
PY = sys.executable
BLOBS = {"src/pmnist_0905.py": "33a0cab", "src/pmnist_rlmnist_0906.py": "33a0cab",
         "src/shell_l2_rlmnist_0913.py": "4fc2bba", "src/wcap_rlmnist_0914.py": "0b8ef53"}
EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)
LAM = 1e-3
ARMS8 = [(a, m) for a in ("R", "LR") for m in ("ref", "l2", "l2wt", "l2rest")]
NEW4 = [(a, m) for a in ("R", "LR") for m in ("l2wt", "l2rest")]
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
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
    return {k: v for k, v in r.items() if k != "pass" and not isinstance(v, (dict, list))} | \
        {"failed_items": r.get("failed_items", [])[:8]}


def run_check(key: str, title: str, fn, target: Path, mutations: list, derivation: str) -> None:
    t0 = time.time()
    base = fn(load(target))
    entry = {"title": title, "threshold_derivation": derivation, **base, "mutations": []}
    for label, subs in mutations:
        mod = load(target, subs)
        try:
            r = fn(mod)
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": bool(r["pass"]),
                                       "detected": not r["pass"], "mutant_result": brief(r)})
        except Exception as e:
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
    entry["all_mutations_detected"] = bool(entry["mutations"]) and all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:3]) if not entry['pass'] else ''}", flush=True)


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    raise RuntimeError("no MemAvailable")


def wait_mem(n_procs: int, per_gib: float = 1.3, reserve_gib: float = 4.0, timeout_s: float = 7200) -> float:
    """Blocks until MemAvailable >= reserve + n_procs * per_gib (other sessions share this machine)."""
    t0 = time.time()
    while True:
        a = mem_available_gib()
        if a >= reserve_gib + n_procs * per_gib:
            return a
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"memory never freed: {a:.1f} GiB available for {n_procs} processes")
        time.sleep(30)


# ---- anchors in src/l2split_rlmnist_0914.py (each must occur exactly once) ----
A_STREAMS = '    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)\n'
A_INIT = '    params = H.init_params(seed, device)              # host init: bit-identical per seed\n'
A_WT = '            out.append(2.0 * LAM * (q - q.mean(dim=1, keepdim=True)) if i in HIDDEN_W else None)\n'
A_REST = '            out.append(2.0 * LAM * (q.mean(dim=1, keepdim=True).expand_as(q) if i in HIDDEN_W else q))\n'
A_IF_WT = '        if arm_name == "l2wt":\n'
A_ELIF_REST = '        elif arm_name == "l2rest":\n'
A_EXTRA = '                        grads = [gr if eg is None else gr + eg for gr, eg in zip(grads, extra_grad(arm.name, params))]\n'
A_ELIF_ARMS = '                    elif arm.name in ("l2wt", "l2rest"):\n'
A_ADAM_STEP = '                        p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)\n'
A_L2 = '                        grads = [gr + 2.0 * reg.lam * (q - (p0[i] if p0 is not None else 0.0))\n'
A_ORDER = '            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)\n'
A_PEN_WT = '            wt += float(((a - m) ** 2).sum())\n'
A_PEN_REST = '            rest += float((a * a).sum())\n'


# --------------------------------------------------------------------------
# S1  identical streams
# --------------------------------------------------------------------------

def s1(M) -> dict:
    """8 arms x seeds {0,1} x 3 tasks x 2 epochs: init / subset / labels / batch orders captured inside
    run_one are bit-identical across the arms and to an independent regeneration from the host streams.
    Threshold: bit equality."""
    failed, per = [], {}
    for seed in (0, 1):
        ref = {"init": sha(H.init_params(seed, CPU)), "subset": sha(RL.subset_idx(seed))}
        g = H.stream("rl_labels", seed)
        ref["labels"] = [sha(RL.task_labels(g)) for _ in range(3)]
        g = H.stream("rl_batch", seed)
        ref["orders"] = [sha(torch.randperm(RL.N_IMAGES, generator=g)) for _ in range(3 * 2)]
        seen = set()
        for act, arm in ARMS8:
            d = {}
            M.run_one(act, arm, seed, 1e-3, 3, MNIST, DEV, epochs=2, debug=d)
            got = {"init": sha(d["init"]), "subset": sha(d["subset"]),
                   "labels": [sha(y) for y in d["labels"]], "orders": [sha(o) for o in d["orders"]]}
            same = {k: got[k] == ref[k] for k in ref}
            per[f"seed{seed}|{act}|{arm}"] = same
            failed += [f"seed{seed}|{act}|{arm}|{k}" for k, v in same.items() if not v]
            seen.add(json.dumps(got, sort_keys=True))
        if len(seen) != 1:
            failed.append(f"seed{seed}: {len(seen)} distinct stream sets")
    return {"pass": not failed, "failed_items": failed, "per_arm": per}


S1_MUT = [
    ("M1a: l2wt draws one extra batch permutation per epoch",
     [(A_ORDER, A_ORDER + '            if arm_s == "l2wt" and e == 1:\n                torch.randperm(N_IMAGES, generator=g_batch)\n')]),
    ("M1b: l2rest's init moved by 1 ulp in one element",
     [(A_INIT, A_INIT + '    if arm_s == "l2rest":\n        with torch.no_grad():\n'
                        '            params[0].view(-1)[0] = torch.nextafter(params[0].view(-1)[0], torch.tensor(1.0))\n')]),
]


# --------------------------------------------------------------------------
# S2  the decomposed gradients
# --------------------------------------------------------------------------

def _s2_cases(M):
    """Random float32 tensors of the host shapes with row means far from 0 (so an uncentered decay shows),
    and LR ref's weights after 1 task x 5 epochs."""
    g = torch.Generator().manual_seed(9142)
    shapes = [(100, 784), (100,), (100, 100), (100,), (10, 100), (10,)]
    rnd = []
    for sh in shapes:
        t = (torch.rand(sh, generator=g) * 2 - 1) * 0.5
        if len(sh) == 2:
            t = t + (torch.rand(sh[0], 1, generator=g) - 0.5)
        rnd.append(t.float())
    d = {}
    M.run_one("LR", "ref", 0, 1e-3, 1, MNIST, DEV, epochs=5, debug=d)
    return [("random", rnd), ("LR_ref_task1", d["task_end_params"][0])]


def s2(M) -> dict:
    """extra_grad on the _s2_cases.  tau_i = 4 eps32 2 lam max_j |W_ij| per row (rounding of subtracting /
    adding the row mean and of the coefficient touches each element at most ~3 eps32 max|W_i|).
    (i) l2wt row sums |sum_j g_ij| <= d tau_i; (ii) l2rest's W part identical within each row; (iii) elementwise
    |g_wt + g_rest - 2 lam theta| <= tau_i on W1/W2 and exact equality on b1/b2/W3/b3; (iv) l2wt adds nothing to
    b1/b2/W3/b3 (None); (v) float64 references 2 lam (W - m) and 2 lam m within tau_i elementwise."""
    failed, per = [], {}
    for name, ps in _s2_cases(M):
        gw, gr = M.extra_grad("l2wt", ps), M.extra_grad("l2rest", ps)
        det = {}
        for i, q in enumerate(ps):
            full = 2.0 * LAM * q
            if i in (0, 2):
                tau = (4 * EPS32 * 2 * LAM * q.abs().amax(dim=1, keepdim=True)).double()
                d = q.shape[1]
                rows_ok = bool((gw[i].double().sum(1, keepdim=True).abs() <= d * tau).all())
                const_ok = bool((gr[i] == gr[i][:, :1]).all())
                sum_ok = bool(((gw[i] + gr[i]).double() - full.double()).abs().le(tau).all())
                q64 = q.double()
                m64 = q64.mean(1, keepdim=True)
                ref_ok = bool(((gw[i].double() - 2 * LAM * (q64 - m64)).abs() <= tau).all()
                              and ((gr[i].double() - 2 * LAM * m64).abs() <= tau).all())
                det[i] = dict(i_rowsum=rows_ok, ii_const=const_ok, iii_sum=sum_ok, v_float64=ref_ok)
            else:
                none_ok = gw[i] is None
                exact_ok = gr[i] is not None and bool((gr[i] == full).all()) and bool((gr[i] == 2.0 * LAM * (q - 0.0)).all())
                det[i] = dict(iv_wt_none=none_ok, iii_rest_exact=exact_ok)
            failed += [f"{name}|tensor{i}|{k}" for k, v in det[i].items() if not v]
        per[name] = det
    return {"pass": not failed, "failed_items": failed, "detail": per}


S2_MUT = [
    ("M2a: l2wt without centering (2 lam W)", [(A_WT, A_WT.replace("(q - q.mean(dim=1, keepdim=True))", "q"))]),
    ("M2b: l2wt centers columns (dim=0)", [(A_WT, A_WT.replace("q.mean(dim=1, keepdim=True)", "q.mean(dim=0, keepdim=True)"))]),
    ("M2c: l2rest drops W3", [(A_REST, A_REST.replace("else q))", "else (q if i != 4 else torch.zeros_like(q))))"))]),
    ("M2d: l2wt coefficient lam instead of 2 lam", [(A_WT, A_WT.replace("2.0 * LAM", "1.0 * LAM"))]),
]


# --------------------------------------------------------------------------
# S3  wiring: hand replay of single steps
# --------------------------------------------------------------------------

def _extra_indep(arm: str, params):
    out = []
    for i, q in enumerate(params):
        if arm == "l2wt":
            out.append(2.0 * LAM * (q - q.mean(dim=1, keepdim=True)) if i in (0, 2) else None)
        else:
            out.append(2.0 * LAM * q.mean(dim=1, keepdim=True).expand_as(q) if i in (0, 2) else 2.0 * LAM * q)
    return out


def _replay(before: dict, xb, yb, act, arm: str, lr: float) -> dict:
    params = [q.detach().clone().requires_grad_(True) for q in before["params"]]
    out = H.forward(params, xb, act)
    loss = torch.nn.functional.cross_entropy(out[4], yb)
    grads = torch.autograd.grad(loss, params)
    m = [q.clone() for q in before["m"]]
    v = [q.clone() for q in before["v"]]
    tc = before["tc"] + 1
    with torch.no_grad():
        grads = [gr if eg is None else gr + eg for gr, eg in zip(grads, _extra_indep(arm, params))]
        c1, c2 = 1 - 0.9 ** tc, 1 - 0.999 ** tc
        for p, gr, mi, vi in zip(params, grads, m, v):
            mi.mul_(0.9).add_(gr, alpha=1 - 0.9)
            vi.mul_(0.999).addcmul_(gr, gr, value=1 - 0.999)
            p -= lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
    return {"params": [q.detach() for q in params], "m": m, "v": v, "tc": tc}


def s3(M) -> dict:
    """R l2wt and LR l2rest, 2 tasks x 3 epochs, captures at (1, 120) and (2, 200): the independently written
    step (forward, CE, autograd, decomposed decay, Adam) reproduces params, m, v bitwise.  Threshold: bit equality."""
    failed, per = [], {}
    for act, arm in (("R", "l2wt"), ("LR", "l2rest")):
        caps = [(1, 120), (2, 200)]
        d = {"capture": set(caps)}
        M.run_one(act, arm, 0, 1e-3, 2, MNIST, DEV, epochs=3, debug=d)
        for t, s in caps:
            st = d.get("steps", {}).get((t, s))
            if st is None or "after" not in st:
                failed.append(f"{act}|{arm}|({t},{s}) not captured")
                continue
            rep = _replay(st["before"], st["xb"], st["yb"], H.ARMS[act], arm, 1e-3)
            eq = {"params": all(bool((a == b).all()) for a, b in zip(rep["params"], st["after"]["params"])),
                  "m": all(bool((a == b).all()) for a, b in zip(rep["m"], st["after"]["m"])),
                  "v": all(bool((a == b).all()) for a, b in zip(rep["v"], st["after"]["v"])),
                  "tc": rep["tc"] == st["after"]["tc"]}
            per[f"{act}|{arm}|({t},{s})"] = eq
            failed += [f"{act}|{arm}|({t},{s})|{k}" for k, v in eq.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S3_MUT = [
    ("M3a: decay subtracted from the weights after Adam instead of added to the gradient",
     [(A_EXTRA, '                        pass\n'),
      (A_ADAM_STEP, A_ADAM_STEP + '                    if arm.name in ("l2wt", "l2rest"):\n'
                                  '                        for p_, eg_ in zip(params, extra_grad(arm.name, params)):\n'
                                  '                            if eg_ is not None:\n'
                                  '                                p_ -= lr * eg_\n')]),
    ("M3b: l2wt also decays W3", [(A_WT, A_WT.replace("if i in HIDDEN_W", "if i in (0, 2, 4)"))]),
    ("M3c: l2rest drops the hidden biases", [(A_REST, A_REST.replace("else q))", "else (torch.zeros_like(q) if i in (1, 3) else q)))"))]),
    ("M3d: l2wt and l2rest branches swapped",
     [(A_IF_WT, '        if arm_name == "l2rest":\n'), (A_ELIF_REST, '        elif arm_name == "l2wt":\n')]),
]


# --------------------------------------------------------------------------
# S4  ref / l2 are wcap's
# --------------------------------------------------------------------------

def s4(M) -> dict:
    """(R, LR) x (ref, l2) x seed 0 x 3 tasks x 3 epochs: the new runner and the untouched wcap runner give
    bit-identical weights at every task end and identical shared per_task columns; the host, 0906, 0913 and
    wcap runners are byte-identical to their blobs.  Threshold: bit equality."""
    failed, per = [], {}
    for f, commit in BLOBS.items():
        blob = subprocess.run(["git", "show", f"{commit}:{f}"], capture_output=True, check=True).stdout
        same = hashlib.sha256(blob).hexdigest() == hashlib.sha256((REPO / f).read_bytes()).hexdigest()
        per[f"blob|{f}"] = same
        if not same:
            failed.append(f"{f} differs from {commit}")
    for act in ("R", "LR"):
        for arm in ("ref", "l2"):
            d_old, d_new = {}, {}
            rows_old, _, _, _ = WC.run_one(act, arm, 0, 1e-3, 3, MNIST, DEV, epochs=3, debug=d_old)
            rows_new, _, _, _ = M.run_one(act, arm, 0, 1e-3, 3, MNIST, DEV, epochs=3, debug=d_new)
            h_old = [sha(ps) for ps in d_old["task_end_params"]]
            h_new = [sha(ps) for ps in d_new["task_end_params"]]
            cols = [k for k in rows_old[0] if k in rows_new[0] and k not in ("act", "arm", "seed", "lr", "task")]
            diff = sorted({k for ro, rn in zip(rows_old, rows_new) for k in cols
                           if not (ro[k] == rn[k] or (ro[k] != ro[k] and rn[k] != rn[k]))})
            ok = len(h_old) == 3 and h_old == h_new and not diff
            per[f"{act}|{arm}"] = {"weights_equal": h_old == h_new, "columns": len(cols), "differing": diff}
            if not ok:
                failed.append(f"{act}|{arm}: weights_equal={h_old == h_new} differing={diff[:5]}")
    return {"pass": not failed, "failed_items": failed, "detail": per}


S4_MUT = [
    ("M4a: l2 coefficient 2.0 -> 1.0", [(A_L2, A_L2.replace("gr + 2.0 * reg.lam", "gr + 1.0 * reg.lam"))]),
    ("M4b: ref also gets the l2wt decay",
     [(A_ELIF_ARMS, '                    elif arm.name in ("l2wt", "l2rest", "ref"):\n'),
      (A_IF_WT, '        if arm_name in ("l2wt", "ref"):\n')]),
]


# --------------------------------------------------------------------------
# S5  reproducibility across processes (CLI)
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


def cli(out: Path, act: str, arm: str, seeds: str, tasks: int, epochs: int, subs=None, wait: bool = True,
        runner: Path = RUNNER):
    args = ["--act", act, "--arm", arm, "--seeds", seeds, "--tasks", str(tasks), "--epochs", str(epochs), "--out", str(out)]
    cmd = [PY, str(runner), *args] if subs is None else [PY, "-c", STUB, str(runner), json.dumps(subs), *args]
    if not wait:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=THREAD_ENV)
    r = subprocess.run(cmd, capture_output=True, text=True, env=THREAD_ENV)
    if r.returncode != 0:
        raise RuntimeError(f"runner failed: {r.stdout[-800:]} {r.stderr[-1500:]}")


def _same_npz(a: Path, b: Path) -> bool:
    x, y = np.load(a), np.load(b)
    return sorted(x.files) == sorted(y.files) and all(x[k].tobytes() == y[k].tobytes() for k in x.files)


def s5_run(subs=None, tag="real") -> dict:
    """R l2wt and LR l2rest, seed 0, 3 tasks x 3 epochs, twice in separate processes: CSVs byte-identical,
    units.npz arrays bit-identical, final and task-1-end state sha256 identical.  Threshold: bit equality."""
    failed, per = [], {}
    for act, arm in (("R", "l2wt"), ("LR", "l2rest")):
        outs = [SCR / "_runs" / f"repro_{tag}_{act}_{arm}_{k}" for k in ("a", "b")]
        for o in outs:
            shutil.rmtree(o, ignore_errors=True)
            cli(o, act, arm, "0", 3, 3, subs)
        same = {f: (outs[0] / f).read_bytes() == (outs[1] / f).read_bytes() for f in ("per_task.csv", "layer_metrics.csv")}
        same["units.npz"] = _same_npz(outs[0] / "units.npz", outs[1] / "units.npz")
        pr = [json.loads((o / "provenance.json").read_text())["runs"]["0"] for o in outs]
        same["final_state_sha256"] = pr[0]["final_state_sha256"] == pr[1]["final_state_sha256"]
        same["task1_end_state_sha256"] = pr[0]["task1_end_state_sha256"] == pr[1]["task1_end_state_sha256"]
        per[f"{act}|{arm}"] = same
        failed += [f"{act}|{arm}|{k}" for k, v in same.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


def s5() -> None:
    t0 = time.time()
    base = s5_run()
    entry = {"title": "S5 S-repro: same arm/seed twice in separate processes", "threshold_derivation": "bit equality",
             **base, "mutations": []}
    subs = [(A_STREAMS, '    g_lab, g_batch = H.stream("rl_labels", seed), torch.Generator().manual_seed(time.time_ns() % (1 << 62))\n')]
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
# S6  penalty columns
# --------------------------------------------------------------------------

TOL_LOG = 1e-12     # float64 on both sides: ~64 eps64 = 1.4e-14, with margin


def s6(M) -> dict:
    """R l2wt, LR l2rest and LR ref, seed 0, 3 tasks x 3 epochs: pen_wt_total and pen_rest_total equal an
    independent numpy float64 recomputation from the captured task-end weights (relative TOL_LOG), their sum
    equals the 0913 layer rows' pen_l2 sum, and reg_loss is the arm's own penalty (ref: 0)."""
    failed, per = [], {}
    for act, arm in (("R", "l2wt"), ("LR", "l2rest"), ("LR", "ref")):
        d = {}
        rows, lrows, _, _ = M.run_one(act, arm, 0, 1e-3, 3, MNIST, DEV, epochs=3, debug=d)
        bad = []
        for t, (row, ps) in enumerate(zip(rows, d["task_end_params"]), start=1):
            wt = rest = 0.0
            for i, q in enumerate(ps):
                a = q.double().numpy()
                if i in (0, 2):
                    mm = a.mean(axis=1, keepdims=True)
                    wt += float(((a - mm) ** 2).sum())
                    rest += float(a.shape[1] * (mm ** 2).sum())
                else:
                    rest += float((a ** 2).sum())
            want = {"pen_wt_total": LAM * wt, "pen_rest_total": LAM * rest}
            for k, w in want.items():
                if abs(row[k] - w) > TOL_LOG * abs(w):
                    bad.append(f"t{t}|{k}|{row[k]} vs {w}")
            l2sum = sum(r["pen_l2"] for r in lrows if r["task"] == t)
            if abs(row["pen_wt_total"] + row["pen_rest_total"] - l2sum) > TOL_LOG * l2sum:
                bad.append(f"t{t}|identity")
            own = {"l2wt": row["pen_wt_total"], "l2rest": row["pen_rest_total"], "ref": 0.0}[arm]
            if row["reg_loss"] != own:
                bad.append(f"t{t}|reg_loss")
        per[f"{act}|{arm}"] = bad[:4]
        failed += [f"{act}|{arm}|{b}" for b in bad[:3]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S6_MUT = [
    ("M6a: pen_wt uncentered", [(A_PEN_WT, '            wt += float((a ** 2).sum())\n')]),
    ("M6b: pen_rest drops b3", [(A_PEN_REST, '            rest += float((a * a).sum()) if i != 5 else 0.0\n')]),
]


# --------------------------------------------------------------------------
# S7  verdict: labels, patterns, guards, reuse
# --------------------------------------------------------------------------

def _write_shard(d: Path, act: str, arm: str, seed: int, A: list, wt: float, dead1: float, w3: float, b1: float,
                 hashes: dict, final: str = "F", t1: str = "T", run_id: str = "l2split_rlmnist_0914") -> None:
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"act": act, "arm": arm, "seed": seed, "task": t, "online_acc": A[t - 1], "memo_acc": 1.0,
             "dead_frac_l1": dead1 if t == 1 else dead1, "wt_med_l1": wt, "mob_l1": 0.3, "mob_l2": 0.8, "zbar_l1": -0.2,
             "zbar_l2": 0.2, "rowmean_abs_med_l1": 0.001} for t in range(1, 51)]
    H.write_csv(d / "per_task.csv", rows)
    lrows = [{"act": act, "arm": arm, "seed": seed, "task": t, "tensor": tn, "norm": val}
             for t in range(0, 51) for tn, val in (("W3", w3), ("b1", b1))]
    H.write_csv(d / "layer_metrics.csv", lrows)
    (d / "provenance.json").write_text(json.dumps({
        "run_id": run_id, "git_hash": "x", "hostname": "h", "wall_clock_s": 1.0, "code_sha256": {}, "git_dirty_code": False,
        "runs": {str(seed): {**hashes, "tasks_completed": 50, "divergence": {"diverged": False},
                             "final_state_sha256": final, "task1_end_state_sha256": t1}}}))


def _traj(rho: float, G_ref: float, late: float | None = None, A1: float = 0.97) -> list:
    out = []
    for t in range(1, 51):
        g = G_ref * (1 - rho) * min(1.0, (t - 1) / 9)
        if late is not None and t >= 41:
            g = G_ref * (1 - late)
        out.append(A1 if t == 1 else A1 - g)
    return out


def _synth(root: Path, scen: dict) -> tuple[Path, Path, Path]:
    """Writes src (new arms + ref/l2 reruns), reuse_src (ref/l2) and reuse.json for one scenario."""
    shutil.rmtree(root, ignore_errors=True)
    src, rsrc = root / "runs", root / "reuse_src"
    G_ref = {"R": 0.85, "LR": 0.16}
    for act, arm in ARMS8:
        for seed in range(10):
            if seed in scen.get("drop", ()):
                continue
            rho = scen["rho"].get(f"{act}_{arm}", 0.0 if arm == "ref" else 1.0)
            A1 = 0.97 - scen.get("tax", {}).get(f"{act}_{arm}", 0.0)
            A = _traj(rho, G_ref[act], scen.get("late", {}).get(f"{act}_{arm}"), A1)
            wt = scen.get("wt", {}).get(f"{act}_{arm}", 4.0 if arm == "ref" else 1.2)
            dead1 = scen.get("dead", {}).get(f"{act}_{arm}", 0.2)
            hs = {"init_sha256": f"i{seed}", "subset_idx_sha256": f"s{seed}", "labels_sha256": f"l{seed}", "batch_sha256": f"b{seed}"}
            if (act, arm, seed) in scen.get("hash_break", ()):
                hs["batch_sha256"] = "broken"
            w3 = 40.0 if arm in ("ref", "l2wt") else 9.0
            b1 = 2.0 if arm in ("ref", "l2wt") else 0.6
            if arm in ("ref", "l2"):
                # the reuse source holds the "good" values; src holds reruns that differ when the scenario says so
                _write_shard(rsrc / f"{act}_{arm}_s{seed}", act, arm, seed, A, wt, dead1, w3, b1, hs, run_id="wcap_rlmnist_0914")
                A_src = [a - scen.get("src_shift", 0.0) for a in A]
                fin = "F" if not scen.get("mismatch_final") else "G"
                _write_shard(src / f"{act}_{arm}_s{seed}", act, arm, seed, A_src, wt, dead1, w3, b1, hs, final=fin)
            else:
                _write_shard(src / f"{act}_{arm}_s{seed}", act, arm, seed, A, wt, dead1, w3, b1, hs)
    return src, rsrc, root / "reuse.json"


S7_SCEN = [
    ("wt_suffices_R_either_LR",
     {"rho": {"R_l2wt": 0.95, "R_l2rest": 0.02, "LR_l2wt": 0.95, "LR_l2rest": 0.95}},
     {"R": "WT_SUFFICES", "LR": "EITHER_SUFFICES"}, {"R_l2wt": "SUFFICIENT", "R_l2rest": "NOT_LEVER"}, {"rho|R_l2wt": 0.95}, "REUSE_OK"),
    ("rest_suffices_R_mixed_LR",
     {"rho": {"R_l2wt": 0.02, "R_l2rest": 0.95, "LR_l2wt": 0.5, "LR_l2rest": 0.5}},
     {"R": "REST_SUFFICES", "LR": "MIXED"}, {"LR_l2wt": "PARTIAL"}, {"rho|LR_l2wt": 0.5}, "REUSE_OK"),
    ("both_needed_R_harmful_LR",
     {"rho": {"R_l2wt": 0.02, "R_l2rest": 0.0, "LR_l2wt": 0.95, "LR_l2rest": 0.02}, "tax": {"LR_l2wt": 0.02}},
     {"R": "BOTH_NEEDED", "LR": "WT_SUFFICES"}, {"LR_l2wt": "SUFFICIENT+HARMFUL", "LR_l2rest": "NOT_LEVER"}, {}, "REUSE_OK"),
    ("late_window_and_dead_guard",
     {"rho": {"R_l2wt": 1.0, "R_l2rest": 0.95}, "late": {"R_l2wt": 0.84}, "dead": {"R_l2rest": 0.35}},
     {}, {"R_l2wt": "SUFFICIENT", "R_l2rest": "SUFFICIENT+HARMFUL"}, {"rho|R_l2wt": 0.92}, "REUSE_OK"),
    ("weak_kappa", {"rho": {"R_l2wt": 0.95}, "wt": {"R_l2wt": 4.0}},
     {"R": "MIXED"}, {"R_l2wt": "WEAK_MANIPULATION"}, {}, "REUSE_OK"),
    ("incomplete", {"rho": {}, "drop": (7, 8, 9)}, {"R": "MIXED"}, {"R_l2wt": "INCOMPLETE"}, {}, "REUSE_OK"),
    ("hash_break", {"rho": {"R_l2wt": 0.95, "R_l2rest": 0.02}, "hash_break": (("LR", "l2", 9),)},
     {"R": "WT_SUFFICES"}, {}, {"n_valid": 9}, "REUSE_OK"),
    # reuse: the reruns in src differ in the final state only (task-1 hash equal) -> MISMATCH -> src is used,
    # whose ref is shifted down by 5 pt, so rho moves and the check sees which source was loaded
    ("reuse_mismatch_final_only",
     {"rho": {"R_l2wt": 0.95, "R_l2rest": 0.02}, "mismatch_final": True, "src_shift": 0.05},
     {}, {}, {}, "REUSE_MISMATCH"),
]


def s7(V) -> dict:
    """Synthetic shard trees for the scenarios above: reuse status from compare_reuse, labels (95%), patterns and
    designed estimates (1e-9).  For reuse_mismatch_final_only the loaded ref must be the shifted rerun in src."""
    failed, per = [], {}
    for name, scen, want_pat, want_lab, want_val, want_reuse in S7_SCEN:
        root = SCR / "_synth" / name
        src, rsrc, rjson = _synth(root, scen)
        reuse = V.compare_reuse(src, rsrc)
        bad = [] if reuse["status"] == want_reuse else [f"reuse {reuse['status']} != {want_reuse}"]
        pt, lm, provs, missing, origin = V.load(src, rsrc, reuse["status"])
        res = V.analyze(pt, lm, provs)
        pats = {a: v["95"] for a, v in res["patterns"].items()}
        labs = {k: v["95"][0] for k, v in res["labels"].items()}
        bad += [f"pattern {a}: {pats.get(a)} != {w}" for a, w in want_pat.items() if pats.get(a) != w]
        bad += [f"label {k}: {labs.get(k)} != {w}" for k, w in want_lab.items() if labs.get(k) != w]
        for k, w in want_val.items():
            g = res["n_valid"] if k == "n_valid" else res["stats"].get(k, {}).get("est")
            if g is None or abs(g - w) > 1e-9:
                bad.append(f"{k}: {g} != {w}")
        if name == "reuse_mismatch_final_only":
            o = origin.get("R_ref_s0", "")
            if "reuse_src" in o:
                bad.append(f"ref loaded from {o} despite mismatch")
        per[name] = {"reuse": reuse["status"], "patterns": pats, "labels": labs, "bad": bad}
        failed += [f"{name}|{b}" for b in bad]
    return {"pass": not failed, "failed_items": failed, "detail": per}


V_WIN = 'WIN = (31, 50)                 # G = A(1) - A(WIN)\n'
V_RHO = '    est = 1.0 - Gm.mean() / Gref.mean()\n'
V_HARM = '            harmful = tax < -TAX_PT / 100 or (act == "R" and dead > DEAD_TOL)\n'
V_REUSE = '    ok = all(v.get("present") and v["final_state_sha256"] and v["task1_end_state_sha256"] and v["online_acc"]\n'
S7_MUT = [
    ("M7a: window 41-50", [(V_WIN, 'WIN = (41, 50)\n')]),
    ("M7b: rho without 1 -", [(V_RHO, '    est = Gm.mean() / Gref.mean()\n')]),
    ("M7c: side-effect inequality reversed", [(V_HARM, V_HARM.replace("tax < -TAX_PT", "tax > -TAX_PT"))]),
    ("M7d: reuse judged on the task-1 hash only",
     [(V_REUSE, '    ok = all(v.get("present") and v["task1_end_state_sha256"]\n')]),
]


# --------------------------------------------------------------------------
# smoke and cost
# --------------------------------------------------------------------------

def smoke() -> None:
    """All 8 arms x seeds {0,1} x 2 tasks x 3 epochs with the new runner into runs/, and ref / l2 with the
    untouched wcap runner into reuse_src/ (8 processes at a time); then verdict --reuse-check (must say
    REUSE_OK: the two runners agree across processes) and the full verdict with explicit --src / --reuse-src.
    Five output files, reuse.json REUSE_OK, 32 task rows, finite accuracies, every label INCOMPLETE."""
    t0 = time.time()
    shutil.rmtree(SMOKE, ignore_errors=True)
    runs, rsrc = SMOKE / "runs", SMOKE / "reuse_src"
    runs.mkdir(parents=True)
    rsrc.mkdir(parents=True)
    jobs = [(runs / f"{a}_{m}_s{s}", a, m, s, RUNNER) for s in (0, 1) for a, m in ARMS8]
    jobs += [(rsrc / f"{a}_{m}_s{s}", a, m, s, WC_RUNNER) for s in (0, 1) for a, m in ARMS8 if m in ("ref", "l2")]
    failed, rc = [], {}
    for k in range(0, len(jobs), 8):
        wait_mem(8)
        batch = [(str(o), cli(o, a, m, str(s), 2, 3, wait=False, runner=r)) for o, a, m, s, r in jobs[k:k + 8]]
        rc.update({n: p.wait() for n, p in batch})
    failed += [f"{n}: exit {c}" for n, c in rc.items() if c != 0]
    # the reuse check compares *_s0 shards; the smoke runs have 2 tasks, which is all compare_reuse needs
    r1 = subprocess.run([PY, str(VERDICT), "--reuse-check", "--src", str(runs), "--reuse-src", str(rsrc),
                         "--reuse-json", str(SMOKE / "reuse.json")], capture_output=True, text=True)
    reuse = json.loads((SMOKE / "reuse.json").read_text()) if (SMOKE / "reuse.json").exists() else {}
    if reuse.get("status") != "REUSE_OK":
        failed.append(f"reuse-check: {r1.stdout[-300:]} {r1.stderr[-300:]}")
    r2 = subprocess.run([PY, str(VERDICT), "--src", str(runs), "--reuse-src", str(rsrc), "--reuse-json", str(SMOKE / "reuse.json"),
                         "--out", str(SMOKE), "--checks", str(DUMP_PATH)], capture_output=True, text=True)
    if r2.returncode != 0:
        failed.append(f"verdict exit {r2.returncode}: {r2.stderr[-600:]}")
    files = ["per_task.csv", "layer_metrics.csv", "verdict.csv", "summary.md", "provenance.json"]
    present = {f: (SMOKE / f).exists() for f in files}
    failed += [f"missing {f}" for f, ok in present.items() if not ok]
    detail = {"files": present, "reuse": reuse.get("status"), "verdict_stdout": r2.stdout[-400:]}
    if all(present.values()):
        pt, vd = pd.read_csv(SMOKE / "per_task.csv"), pd.read_csv(SMOKE / "verdict.csv")
        labels = sorted(set(vd[vd.kind == "label"].label))
        detail.update(n_task_rows=len(pt), labels=labels)
        if len(pt) != 32:
            failed.append(f"{len(pt)} task rows != 32")
        if not np.isfinite(pt.online_acc).all():
            failed.append("non-finite accuracy")
        if labels != ["INCOMPLETE"]:
            failed.append(f"labels {labels}")
    RESULTS["S-smoke"] = {"pass": not failed, "failed_items": failed, "detail": detail, "seconds": round(time.time() - t0, 1)}
    dump()
    print(f"S-smoke: pass={not failed}  reuse={reuse.get('status')}  ({RESULTS['S-smoke']['seconds']}s) {failed[:3]}", flush=True)


def cost() -> None:
    """8 concurrent processes (each of the 4 new arms -- l2wt, l2rest x R, LR -- twice), 1 task x 100 epochs: seconds per 7,500 steps and
    peak RSS.  Slots N = min(16, floor((MemAvailable - 4 GiB) / (1.2 x max peak RSS))) from MemAvailable read
    before the 8 start; the launch waits (does not stop) while N < 4.  Projection: every new arm is an L2-type
    arm, so its 200 x (100-epoch time) is multiplied by 3 (wcap: task-1 timing under-read R ref 21 -> 67 min,
    LR l2 35 -> 56 min); the 4 verification runs use wcap's measured medians."""
    t0 = time.time()
    avail = wait_mem(8)
    root = SCR / "_runs" / "cost"
    shutil.rmtree(root, ignore_errors=True)
    procs = [(f"{a}_{m}_{k}", cli(root / f"{a}_{m}_{k}", a, m, "0", 1, 100, wait=False)) for k in (0, 1) for a, m in NEW4]
    rc = [p.wait() for _, p in procs]
    per_arm, rss = {}, []
    for (name, _), c in zip(procs, rc):
        if c == 0:
            pr = json.loads((root / name / "provenance.json").read_text())
            per_arm.setdefault(name.rsplit("_", 1)[0], []).append(pr["runs"]["0"]["wall_clock_s"])
            rss.append(pr["peak_rss_kb"] / 2 ** 20)
    peak = max(rss) if rss else float("inf")
    slots = int(min(16, (avail - 4.0) // (1.2 * peak))) if rss else 0
    prov = json.loads((REPO / "results" / "wcap_rlmnist_0914" / "provenance.json").read_text())["shards"]
    verify_min = {k: prov[k]["wall_clock_s"] / 60 for k in ("R_ref_s0", "R_l2_s0", "LR_ref_s0", "LR_l2_s0")}
    run_min = {a: 3 * (200 * float(np.mean(v)) + 10.0) / 60 for a, v in per_arm.items()}
    queue = [verify_min[k] for k in ("R_ref_s0", "R_l2_s0", "LR_ref_s0", "LR_l2_s0")]
    queue += [run_min[f"{a}_{m}"] for s in range(10) for a, m in NEW4]
    finish = 0.0
    if slots > 0:
        heap = [0.0] * slots
        heapq.heapify(heap)
        for q in queue:
            st = heapq.heappop(heap)
            finish = max(finish, st + q)
            heapq.heappush(heap, st + q)
    ok = len(rc) == 8 and all(c == 0 for c in rc) and slots >= 1
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": "8 processes exit 0 (slots < 4 makes the launch wait, not fail)",
                         "mem_available_gib_before": avail, "peak_rss_gib_max": peak, "slots": slots,
                         "sec_per_100_epochs_under_8_load": {a: float(np.mean(v)) for a, v in per_arm.items()},
                         "projected_run_minutes_x3": run_min, "verification_run_minutes_from_wcap": verify_min,
                         "projected_wall_hours_44_runs": finish / 60,
                         "measured_at": dt.datetime.now().astimezone().isoformat(), "seconds": round(time.time() - t0, 1)}
    dump()
    print(f"S-cost: pass={ok}  avail {avail:.1f} GiB  peak {peak:.2f} GiB  slots {slots}  44 runs -> {finish / 60:.2f} h", flush=True)


# --------------------------------------------------------------------------

def main() -> None:
    global DUMP_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    if only:
        DUMP_PATH = SCR / "checks_partial.json"
    RESULTS.update({"run_id": "l2split_rlmnist_0914", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                    "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (RUNNER, VERDICT, Path(__file__).resolve())},
                    "torch": torch.__version__, "threads": torch.get_num_threads()})
    (SCR / "_runs").mkdir(parents=True, exist_ok=True)

    def want(k):
        return only is None or k in only

    table = [
        ("S1", "S1_S-identical", "streams identical across arms", s1, RUNNER, S1_MUT, "bit equality"),
        ("S2", "S2_S-grad", "decomposed gradients: centered / row-constant / sum to 2 lam theta", s2, RUNNER, S2_MUT, s2.__doc__),
        ("S3", "S3_S-wiring", "hand replay of l2wt / l2rest steps == runner", s3, RUNNER, S3_MUT, "bit equality"),
        ("S4", "S4_S-baseline", "ref / l2 == untouched wcap runner; blobs unchanged", s4, RUNNER, S4_MUT, "bit equality"),
        ("S6", "S6_S-log", "penalty columns == independent recomputation", s6, RUNNER, S6_MUT, s6.__doc__),
        ("S7", "S7_S-verdict", "verdict / reuse on synthetic shard trees", s7, VERDICT, S7_MUT, "label/pattern equality; estimates 1e-9"),
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

"""Checks for src/wcap_rlmnist_0914 (specs/spec_wcap_rlmnist_0914.md section 6).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/wcap_rlmnist_0914/checks.py
    ... --only S2,S3        # development: writes results/_checks_wcap_rlmnist_0914/checks_partial.json

S1-S7 are each run on the real code and then on every mutation listed with them.  A mutation is an
exact-once string substitution in the runner (or in the verdict script) loaded as a separate module --
or, for the CLI check S5, executed as a script -- and it must make the same check FAIL.  Tolerances are
fixed by the derivations in the docstrings (spec section 6 table), not by looking at the values.
Then the smoke test (S-smoke) and the cost measurement (S-cost: parallel slots from measured memory).

Writes results/wcap_rlmnist_0914/checks.json after every step.
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

RUNNER = REPO / "src" / "wcap_rlmnist_0914.py"
VERDICT = REPO / "analysis" / "wcap_rlmnist_0914" / "verdict.py"
OUT = REPO / "results" / "wcap_rlmnist_0914"
SCR = REPO / "results" / "_checks_wcap_rlmnist_0914"
SMOKE = REPO / "results" / "_smoke_wcap_rlmnist_0914"
PY = sys.executable
BLOBS = {"src/pmnist_0905.py": "33a0cab", "src/pmnist_rlmnist_0906.py": "33a0cab",
         "src/shell_l2_rlmnist_0913.py": "4fc2bba"}
EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)
ARMS8 = [("LR", "ref"), ("LR", "l2"), ("LR", "l2init"), ("LR", "capT1"), ("LR", "cap2"),
         ("R", "ref"), ("R", "l2"), ("R", "cap2")]
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
CPU = torch.device("cpu")
DEV = H.setup("cpu")
MNIST = H.Mnist(DEV)
RESULTS: dict = {}
DUMP_PATH = OUT / "checks.json"

# --------------------------------------------------------------------------
# infrastructure (0913's, unchanged in behaviour)
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
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:3]) if not entry['pass'] else ''}", flush=True)


# ---- anchors in src/wcap_rlmnist_0914.py (each must occur exactly once) ----
A_STREAMS = '    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)\n'
A_INIT = '    params = H.init_params(seed, device)              # host init: bit-identical per seed\n'
A_T1_SET = '                cap_r = [cap_radius(params[k]) for k in CAP_PARAMS]\n'
A_CAP2_SET = '    cap_r = [cap_radius(params[k], CAP_MULT) for k in CAP_PARAMS] if arm.name == "cap2" else None\n'
A_CAPSTART = '        return {"capT1": 2, "cap2": 1}.get(self.name)\n'
A_CAP_IF = '                    if cap_on:\n'
A_PROJ_CALL = '                            nr, rem = project_rows_(params[k], cap_r[li])\n'
A_PROJ_WRITE = '    W.copy_(torch.where(over, m + Wt * (r / n), W))\n'
A_CAP_PARAMS = 'CAP_PARAMS = (0, 2) '
A_CAP_BLOCK = ('                    if cap_on:\n'
               '                        for li, k in enumerate(CAP_PARAMS):\n'
               '                            nr, rem = project_rows_(params[k], cap_r[li])\n'
               '                            proj_rows[li] += nr\n'
               '                            proj_removed[li] += rem.double()\n')
A_ADAM_UNPACK = '                    m, v, tc = adam\n'
A_L2 = '                        grads = [gr + 2.0 * reg.lam * (q - (p0[i] if p0 is not None else 0.0))\n'
A_P0 = '    p0 = [q.detach().clone() for q in params] if reg.kind == "l2init" else None\n'
A_T1_BRANCH = '            if arm.name == "capT1":\n'
A_WT = '        out[f"wt_l{li}"] = torch.linalg.vector_norm(W - m, dim=1).numpy()\n'
A_RATIO0 = '        f[f"wt_ratio0_med_l{li}"] = float(np.quantile(w / w0, 0.5))\n'
A_RT1 = '            r_t1 = {li: u[f"wt_l{li}"].copy() for li in (1, 2)}\n'


# --------------------------------------------------------------------------
# S1  identical streams across the 8 arms
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
        for act, arm in ARMS8:
            d = {}
            M.run_one(act, arm, seed, 1e-3, 3, MNIST, DEV, epochs=2, debug=d)
            got = {"init": sha(d["init"]), "subset": sha(d["subset"]),
                   "labels": [sha(y) for y in d["labels"]], "orders": [sha(o) for o in d["orders"]]}
            same = {k: got[k] == ref[k] for k in ref}
            per[f"seed{seed}|{act}|{arm}"] = same
            failed += [f"seed{seed}|{act}|{arm}|{k}" for k, v in same.items() if not v]
            seen.add(json.dumps(got, sort_keys=True))
        per[f"seed{seed}|distinct_across_8_arms"] = len(seen)
        if len(seen) != 1:
            failed.append(f"seed{seed}: {len(seen)} distinct stream sets across the 8 arms")
    return {"pass": not failed, "failed_items": failed, "per_arm": per, "seeds": [0, 1], "tasks": 3, "epochs": 2}


S1_MUT = [
    ("M1a: capT1 draws one extra batch permutation after task 1",
     [(A_T1_BRANCH, A_T1_BRANCH + '                torch.randperm(N_IMAGES, generator=g_batch)\n')]),
    ("M1b: cap2 arm's init moved by 1 ulp in one element",
     [(A_INIT, A_INIT + '    if arm_s == "cap2":\n        with torch.no_grad():\n'
                        '            params[0].view(-1)[0] = torch.nextafter(params[0].view(-1)[0], torch.tensor(1.0))\n')]),
    ("M1c: R arms advance the label stream by one task before task 1",
     [(A_STREAMS, A_STREAMS + '    if act_name == "R":\n        RL.task_labels(g_lab)\n')]),
]


# --------------------------------------------------------------------------
# S2  projection geometry
# --------------------------------------------------------------------------

TOL_NORM = 1e-4       # float32 norm over d <= 784 terms: relative error <= d * eps32 = 9.3e-5
TOL_COS = 1e-6        # rounding tilts a row by ~eps32 * sqrt(d) ~ 3e-6 rad -> 1 - cos ~ 5e-12
TOL_REMOVED = 1e-3    # n - r >= 0.2 n by construction below: relative error <= 5 * 9.3e-5 + eps


def _s2_cases(M):
    """(name, W float32, r float32 (rows, 1)).  Random rows: centered norms U(2, 10), row means U(-0.1, 0.1),
    caps n * f with f drawn from {0.5..0.8} U {1.25..1.5} (never within 20% of the norm).  Real rows: W1, W2
    of LR ref after 1 task x 5 epochs, caps alternating 0.8 n and 1.25 n."""
    g = torch.Generator().manual_seed(914)
    cases = []
    for cols in (784, 100):
        Wt = torch.randn(100, cols, generator=g, dtype=torch.float64)
        Wt -= Wt.mean(1, keepdim=True)
        Wt *= (2 + 8 * torch.rand(100, 1, generator=g, dtype=torch.float64)) / Wt.norm(dim=1, keepdim=True)
        W = (Wt + 0.2 * (torch.rand(100, 1, generator=g, dtype=torch.float64) - 0.5)).float()
        f = torch.where(torch.rand(100, 1, generator=g) < 0.5, 0.5 + 0.3 * torch.rand(100, 1, generator=g),
                        1.25 + 0.25 * torch.rand(100, 1, generator=g))
        n = torch.linalg.vector_norm(W - W.mean(1, keepdim=True), dim=1, keepdim=True)
        cases.append((f"random_{cols}", W, (n * f).float()))
    d = {}
    M.run_one("LR", "ref", 0, 1e-3, 1, MNIST, DEV, epochs=5, debug=d)
    for k, name in ((0, "LR_ref_W1"), (2, "LR_ref_W2")):
        W = d["task_end_params"][0][k].clone()
        n = torch.linalg.vector_norm(W - W.mean(1, keepdim=True), dim=1, keepdim=True)
        f = torch.tensor([0.8 if i % 2 == 0 else 1.25 for i in range(W.shape[0])]).view(-1, 1)
        cases.append((name, W, (n * f).float()))
    return cases


def s2(M) -> dict:
    """project_rows_ on the _s2_cases: (i) rows with n <= r keep every bit; (ii) written rows satisfy
    |‖W~'‖/r − 1| <= 1e-4; (iii) |m'_i − m_i| <= 4·eps32·max|W_i| (three rounded operations touch the mean);
    (iv) 1 − cos(W~', W~) <= 1e-6; (v) the returned row count equals the float64 recount of n > r and the
    returned sum of n − r matches the float64 sum within relative 1e-3.  Norms / means / cosines recomputed
    in float64.  Tolerances: TOL_NORM, TOL_COS, TOL_REMOVED comments."""
    failed, per = [], {}
    for name, W0, r in _s2_cases(M):
        W = W0.clone()
        cnt, removed = M.project_rows_(W, r)
        a, b, r64 = W0.double(), W.double(), r.double()
        m_a, m_b = a.mean(1, keepdim=True), b.mean(1, keepdim=True)
        n_a = torch.linalg.vector_norm(a - m_a, dim=1, keepdim=True)
        n_b = torch.linalg.vector_norm(b - m_b, dim=1, keepdim=True)
        over = (n_a > r64)[:, 0]
        under_same = bool((W[~over] == W0[~over]).all())
        rel = ((n_b / r64 - 1).abs()[over]).max().item() if over.any() else 0.0
        dm = ((m_b - m_a).abs()[:, 0] / (a.abs().amax(1) * EPS32))[over].max().item() if over.any() else 0.0
        cos = ((b - m_b) * (a - m_a)).sum(1) / (n_a[:, 0] * n_b[:, 0])
        one_m_cos = (1 - cos[over]).abs().max().item() if over.any() else 0.0
        cnt64 = int(over.sum())
        rem64 = float((n_a - r64)[over[:, None].expand_as(n_a)].sum())
        rem_rel = abs(float(removed) - rem64) / max(rem64, 1e-300)
        ok = dict(i_under_bit_identical=under_same, ii_norm=rel <= TOL_NORM, iii_mean=dm <= 4.0,
                  iv_direction=one_m_cos <= TOL_COS, v_count=int(cnt) == cnt64, v_removed=rem_rel <= TOL_REMOVED,
                  nonvacuous=0 < cnt64 < W.shape[0])
        per[name] = {**ok, "rows_over": cnt64, "max_rel_norm": rel, "max_dmean_in_eps_units": dm,
                     "max_1mcos": one_m_cos, "removed_rel_err": rem_rel}
        failed += [f"{name}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S2_MUT = [
    ("M2a: uncentered projection W * r / ||W||",
     [(A_PROJ_WRITE, '    W.copy_(torch.where(over, W * (r / torch.linalg.vector_norm(W, dim=1, keepdim=True)), W))\n')]),
    ("M2b: every row set to norm r (rows under the cap are stretched too)",
     [(A_PROJ_WRITE, '    W.copy_(m + Wt * (r / n))\n')]),
    ("M2c: scale factor r / n^2",
     [(A_PROJ_WRITE, '    W.copy_(torch.where(over, m + Wt * (r / (n * n)), W))\n')]),
    ("M2d: written rows get a 1% tangential component",
     [(A_PROJ_WRITE, '    W.copy_(torch.where(over, m + Wt * (r / n) + 0.01 * (r / n) * torch.roll(Wt, 1, dims=1), W))\n')]),
]


# --------------------------------------------------------------------------
# S3  wiring: hand replay of single steps, and the activation schedule
# --------------------------------------------------------------------------

def _replay(before: dict, xb, yb, act, reg: SH.Reg, p0, lr: float, radii) -> dict:
    """One step written out independently of the runner: forward, CE, autograd, l2/l2init, Adam (0906's
    constants), then -- only if radii is not None -- the cap on W1 and W2 after the update."""
    params = [q.detach().clone().requires_grad_(True) for q in before["params"]]
    out = H.forward(params, xb, act)
    loss = torch.nn.functional.cross_entropy(out[4], yb)
    grads = torch.autograd.grad(loss, params)
    m = [q.clone() for q in before["m"]]
    v = [q.clone() for q in before["v"]]
    tc = before["tc"] + 1
    written = {}
    with torch.no_grad():
        if reg.kind in ("l2", "l2init"):
            grads = [gr + 2.0 * reg.lam * (q - (p0[i] if p0 is not None else 0.0)) for i, (q, gr) in enumerate(zip(params, grads))]
        c1, c2 = 1 - 0.9 ** tc, 1 - 0.999 ** tc
        for p, gr, mi, vi in zip(params, grads, m, v):
            mi.mul_(0.9).add_(gr, alpha=1 - 0.9)
            vi.mul_(0.999).addcmul_(gr, gr, value=1 - 0.999)
            p -= lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)
        if radii is not None:
            for li, k in enumerate((0, 2)):
                W = params[k]
                mm = W.mean(dim=1, keepdim=True)
                Wt = W - mm
                nn = torch.linalg.vector_norm(Wt, dim=1, keepdim=True)
                over = nn > radii[li]
                W.copy_(torch.where(over, mm + Wt * (radii[li] / nn), W))
                written[li] = int(over.sum())
    return {"params": [q.detach() for q in params], "m": m, "v": v, "tc": tc, "written": written}


def _radius(W: torch.Tensor, mult: float = 1.0) -> torch.Tensor:
    return mult * torch.linalg.vector_norm(W - W.mean(dim=1, keepdim=True), dim=1, keepdim=True)


def s3(M) -> dict:
    """Replays captured steps by hand and compares params, m, v bitwise with the runner's post-step state.
    LR capT1 (2 tasks x 3 epochs): (1, 100) must be uncapped, (2, 200) and (2, 201) capped with radii = the
    task-1-end centered norms recomputed here.  LR cap2 and R cap2 (1 task x 40 epochs): (1, 2900) and
    (1, 2901) capped with radii = 2 x the init norms recomputed here.  Every capped replay must write >= 1
    row in W1 and in W2 (else the capture is vacuous and the check fails).  Schedule: capT1's projection
    tally is 0 in task 1 and > 0 in both layers in task 2; cap2's is > 0 in both layers in task 1.
    Threshold: bit equality / integer counts."""
    failed, per = [], {}
    plan = [("LR", "capT1", 2, 3, [(1, 100, False), (2, 200, True), (2, 201, True)]),
            ("LR", "cap2", 1, 40, [(1, 2900, True), (1, 2901, True)]),
            ("R", "cap2", 1, 40, [(1, 2900, True), (1, 2901, True)])]
    for act, arm, tasks, epochs, caps in plan:
        d = {"capture": {(t, s) for t, s, _ in caps}}
        M.run_one(act, arm, 0, 1e-3, tasks, MNIST, DEV, epochs=epochs, debug=d)
        if arm == "capT1":
            radii = [_radius(d["task_end_params"][0][k]) for k in (0, 2)]
        else:
            radii = [_radius(d["init"][k], 2.0) for k in (0, 2)]
        for t, s, capped in caps:
            st = d.get("steps", {}).get((t, s))
            if st is None or "after" not in st:
                failed.append(f"{act}|{arm}|({t},{s}) not captured")
                continue
            rep = _replay(st["before"], st["xb"], st["yb"], H.ARMS[act], SH.Reg("none"), None, 1e-3,
                          radii if capped else None)
            eq = {"params": all(bool((a == b).all()) for a, b in zip(rep["params"], st["after"]["params"])),
                  "m": all(bool((a == b).all()) for a, b in zip(rep["m"], st["after"]["m"])),
                  "v": all(bool((a == b).all()) for a, b in zip(rep["v"], st["after"]["v"])),
                  "tc": rep["tc"] == st["after"]["tc"]}
            key = f"{act}|{arm}|({t},{s})|capped={capped}"
            per[key] = {**eq, "rows_written_by_replay": rep["written"]}
            failed += [f"{key}|{k}" for k, v in eq.items() if not v]
            if capped and not (rep["written"].get(0, 0) > 0 and rep["written"].get(1, 0) > 0):
                failed.append(f"{key}|vacuous: replay wrote {rep['written']}")
        rows = [p["rows"] if p is not None else None for p in d["proj"]]
        per[f"{act}|{arm}|proj_rows_per_task"] = rows
        if arm == "capT1":
            if not (rows[0] == [0, 0] and rows[1][0] > 0 and rows[1][1] > 0):
                failed.append(f"{act}|{arm}: schedule {rows}")
        elif not (rows[0] is not None and rows[0][0] > 0 and rows[0][1] > 0):
            failed.append(f"{act}|{arm}: schedule {rows}")
    return {"pass": not failed, "failed_items": failed, "detail": per}


S3_MUT = [
    ("M3a: projection placed before the Adam update",
     [(A_CAP_BLOCK, ""), (A_ADAM_UNPACK, A_CAP_BLOCK + A_ADAM_UNPACK)]),
    ("M3b: W3 is capped too", [(A_CAP_PARAMS, "CAP_PARAMS = (0, 2, 4) ")]),
    ("M3c: capT1 capped from task 1 with radii = its init norms",
     [(A_CAP2_SET, '    cap_r = [cap_radius(params[k], CAP_MULT if arm.name == "cap2" else 1.0) for k in CAP_PARAMS] '
                   'if arm.name in ("cap2", "capT1") else None\n'),
      (A_CAPSTART, A_CAPSTART.replace('"capT1": 2', '"capT1": 1'))]),
    ("M3d: cap applied only on even steps", [(A_CAP_IF, '                    if cap_on and s % 2 == 0:\n')]),
    ("M3e: W2 never capped",
     [(A_PROJ_CALL, '                            if k == 2:\n                                continue\n' + A_PROJ_CALL)]),
]


# --------------------------------------------------------------------------
# S4  the baseline loop is 0913's
# --------------------------------------------------------------------------

def s4(M) -> dict:
    """(LR, R) x (ref, l2, l2init) x seed 0 x 3 tasks x 3 epochs: the new runner and the untouched 0913
    SH.run_one give bit-identical weights at every task end and identical shared per_task columns; the host,
    the 0906 runner and the 0913 runner are byte-identical to their blobs (33a0cab, 33a0cab, 4fc2bba).
    Threshold: bit equality."""
    failed, per = [], {}
    for f, commit in BLOBS.items():
        blob = subprocess.run(["git", "show", f"{commit}:{f}"], capture_output=True, check=True).stdout
        same = hashlib.sha256(blob).hexdigest() == hashlib.sha256((REPO / f).read_bytes()).hexdigest()
        per[f"blob|{f}"] = same
        if not same:
            failed.append(f"{f} differs from {commit}")
    for act in ("LR", "R"):
        for arm, reg_s in (("ref", "none"), ("l2", "l2:1e-3"), ("l2init", "l2init:1e-3")):
            d_old, d_new = {}, {}
            rows_old, _, _ = SH.run_one(act, reg_s, 0, 1e-3, 3, MNIST, DEV, epochs=3, debug=d_old)
            rows_new, _, _, _ = M.run_one(act, arm, 0, 1e-3, 3, MNIST, DEV, epochs=3, debug=d_new)
            h_old = [sha(ps) for ps in d_old["task_end_params"]]
            h_new = [sha(ps) for ps in d_new["task_end_params"]]
            cols = [k for k in rows_old[0] if k in rows_new[0] and k not in ("act", "reg", "arm", "seed", "lr", "task")]
            diff = sorted({k for ro, rn in zip(rows_old, rows_new) for k in cols
                           if not (ro[k] == rn[k] or (ro[k] != ro[k] and rn[k] != rn[k]))})
            ok = len(h_old) == 3 and h_old == h_new and len(rows_old) == len(rows_new) and not diff
            per[f"{act}|{arm}"] = {"weights_equal_each_task": h_old == h_new, "columns_compared": len(cols),
                                   "differing_columns": diff}
            if not ok:
                failed.append(f"{act}|{arm}: weights_equal={h_old == h_new} differing={diff[:5]}")
    return {"pass": not failed, "failed_items": failed, "detail": per}


S4_MUT = [
    ("M4a: l2/l2init gradient coefficient 2.0 -> 1.0", [(A_L2, A_L2.replace("gr + 2.0 * reg.lam", "gr + 1.0 * reg.lam"))]),
    ("M4b: l2init anchor p0 aliases the live weights (no clone)",
     [(A_P0, '    p0 = [q.detach() for q in params] if reg.kind == "l2init" else None\n')]),
    ("M4c: the ref arm is capped like cap2",
     [(A_CAP2_SET, A_CAP2_SET.replace('if arm.name == "cap2"', 'if arm.name in ("cap2", "ref")')),
      (A_CAPSTART, A_CAPSTART.replace('"cap2": 1}', '"cap2": 1, "ref": 1}'))]),
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
        prefix=()):
    args = ["--act", act, "--arm", arm, "--seeds", seeds, "--tasks", str(tasks), "--epochs", str(epochs), "--out", str(out)]
    cmd = [*prefix, PY, str(RUNNER), *args] if subs is None else [*prefix, PY, "-c", STUB, str(RUNNER), json.dumps(subs), *args]
    if not wait:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=THREAD_ENV)
    r = subprocess.run(cmd, capture_output=True, text=True, env=THREAD_ENV)
    if r.returncode != 0:
        raise RuntimeError(f"runner failed: {r.stdout[-800:]} {r.stderr[-1500:]}")


def _same_npz(a: Path, b: Path) -> bool:
    x, y = np.load(a), np.load(b)
    return sorted(x.files) == sorted(y.files) and all(
        x[k].tobytes() == y[k].tobytes() for k in x.files)


def s5_run(subs=None, tag="real") -> dict:
    """LR capT1 and R cap2, seed 0, 3 tasks x 3 epochs, twice in separate processes: per_task.csv and
    layer_metrics.csv byte-identical, units.npz arrays bit-identical (the zip container carries a
    timestamp, so arrays are compared, not bytes), final and task-1-end state sha256 identical.
    Threshold: byte / bit equality."""
    failed, per = [], {}
    for act, arm in (("LR", "capT1"), ("R", "cap2")):
        outs = [SCR / "_runs" / f"repro_{tag}_{act}_{arm}_{k}" for k in ("a", "b")]
        for o in outs:
            shutil.rmtree(o, ignore_errors=True)
            cli(o, act, arm, "0", 3, 3, subs)
        same = {f: (outs[0] / f).read_bytes() == (outs[1] / f).read_bytes() for f in ("per_task.csv", "layer_metrics.csv")}
        same["units.npz"] = _same_npz(outs[0] / "units.npz", outs[1] / "units.npz")
        pr = [json.loads((o / "provenance.json").read_text())["runs"]["0"] for o in outs]
        same["final_state_sha256"] = pr[0]["final_state_sha256"] == pr[1]["final_state_sha256"]
        same["task1_end_state_sha256"] = pr[0]["task1_end_state_sha256"] == pr[1]["task1_end_state_sha256"]
        per[f"{act}|{arm}"] = {**same, "state_sha256": pr[0]["final_state_sha256"][:16]}
        failed += [f"{act}|{arm}|{k}" for k, v in same.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


def s5() -> None:
    t0 = time.time()
    base = s5_run()
    entry = {"title": "S5 S-repro: same arm/seed twice in separate processes -> identical outputs and state",
             "threshold_derivation": "byte/bit equality (determinism claim)", **base, "mutations": []}
    subs = [(A_STREAMS, '    g_lab, g_batch = H.stream("rl_labels", seed), '
                        'torch.Generator().manual_seed(time.time_ns() % (1 << 62))\n')]
    r = s5_run(subs, tag="mutM5")
    entry["mutations"].append({"mutation": "M5: batch stream seeded from time.time_ns()",
                               "check_pass_on_mutant": r["pass"], "detected": not r["pass"], "mutant_result": brief(r)})
    entry["all_mutations_detected"] = all(m["detected"] for m in entry["mutations"])
    # REPORT only (platform property, not a code check): P-core vs E-core
    pe = {}
    try:
        outs = {}
        for tag, cpu in (("pcore0", "0"), ("ecore27", "27")):
            o = SCR / "_runs" / f"repro_pin_{tag}"
            shutil.rmtree(o, ignore_errors=True)
            cli(o, "LR", "capT1", "0", 3, 3, prefix=("taskset", "-c", cpu))
            outs[tag] = json.loads((o / "provenance.json").read_text())["runs"]["0"]
        pe = {"final_state_equal": outs["pcore0"]["final_state_sha256"] == outs["ecore27"]["final_state_sha256"],
              "task1_state_equal": outs["pcore0"]["task1_end_state_sha256"] == outs["ecore27"]["task1_end_state_sha256"]}
    except Exception as e:
        pe = {"error": repr(e)[:300]}
    entry["REPORT_pcore_vs_ecore"] = pe
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS["S5_S-repro"] = entry
    dump()
    print(f"S5_S-repro: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  P/E: {pe}  ({entry['seconds']}s)",
          flush=True)


# --------------------------------------------------------------------------
# S6  logged columns and units.npz == independent recomputation
# --------------------------------------------------------------------------

TOL_LOG = 1e-12     # float64 recomputations of the same float32 weights: ~64 * eps64 = 1.4e-14, with margin
TOL_F32 = 2 * EPS32  # units.npz stores float32 casts of float64 values: at most one ulp of rounding apart


def _recompute(d: dict, act_name: str, arm: str) -> tuple[list[dict], dict]:
    act = H.ARMS[act_name]
    x = MNIST.train_x[RL.subset_idx(0)]
    states = [d["init"]] + d["task_end_params"]
    U = []
    for ps in states:
        z1, _, z2, _, _ = H.forward(ps, x, act)
        u = {}
        for li, (k, z) in enumerate(((0, z1), (2, z2)), start=1):
            W = ps[k].double().numpy()
            m = W.mean(axis=1, keepdims=True)
            u[f"wt_l{li}"] = np.sqrt(((W - m) ** 2).sum(axis=1))
            u[f"rowmean_l{li}"] = m[:, 0]
            u[f"b{li}"] = ps[k + 1].double().numpy()
            u[f"zbar_l{li}"] = z.double().mean(0).numpy()
            u[f"mob_l{li}"] = act.dphi(z).double().mean(0).numpy()
        U.append(u)
    rt1 = {li: U[1][f"wt_l{li}"] for li in (1, 2)}
    rcap = None
    if arm == "cap2":
        rcap = {li: (2.0 * torch.linalg.vector_norm(d["init"][k] - d["init"][k].mean(1, keepdim=True), dim=1)).double().numpy()
                for li, k in ((1, 0), (2, 2))}
    elif arm == "capT1":
        rcap = {li: torch.linalg.vector_norm(d["task_end_params"][0][k] - d["task_end_params"][0][k].mean(1, keepdim=True),
                                             dim=1).double().numpy() for li, k in ((1, 0), (2, 2))}
    rows = []
    for t in range(1, len(states)):
        u, u0 = U[t], U[0]
        on = rcap is not None and t >= (2 if arm == "capT1" else 1)
        f = {}
        for li in (1, 2):
            w, w0 = u[f"wt_l{li}"], u0[f"wt_l{li}"]
            f[f"wt_q10_l{li}"], f[f"wt_med_l{li}"], f[f"wt_q90_l{li}"] = (float(v) for v in np.quantile(w, [0.1, 0.5, 0.9]))
            f[f"wt_max_l{li}"] = float(w.max())
            f[f"wt_ratio0_med_l{li}"] = float(np.quantile(w / w0, 0.5))
            f[f"rowmean_abs_med_l{li}"] = float(np.quantile(np.abs(u[f"rowmean_l{li}"]), 0.5))
            f[f"b_med_l{li}"] = float(np.quantile(u[f"b{li}"], 0.5))
            f[f"exceed_t1_frac_l{li}"] = float(np.mean(w > rt1[li] * (1 + 1e-4)))
            f[f"exceed_2x0_frac_l{li}"] = float(np.mean(w > 2.0 * w0 * (1 + 1e-4)))
            f[f"bind_frac_l{li}"] = float(np.mean(w >= rcap[li] * (1 - 1e-4))) if on else float("nan")
            p = d["proj"][t - 1]
            f[f"proj_rows_l{li}"] = int(p["rows"][li - 1]) if on else float("nan")
        rows.append(f)
    arrays = {k: np.stack([u[k] for u in U]) for k in U[0]}
    for li in (1, 2):
        arrays[f"r_t1_l{li}"] = rt1[li]
        arrays[f"r_cap_l{li}"] = rcap[li] if rcap is not None else np.full(100, np.nan)
    return rows, arrays


def s6(M) -> dict:
    """LR ref / LR capT1 / LR cap2 / R cap2, seed 0, 3 tasks x 3 epochs (cap2 arms 3 tasks x 20 epochs so the
    cap binds): every new per_task column and every units.npz array equals the recomputation in this file
    from the captured init and task-end weights -- float columns within relative TOL_LOG, fractions and
    counts exactly, float32 arrays within TOL_F32 relative (NaN == NaN)."""
    failed, per = [], {}
    for act, arm, epochs in (("LR", "ref", 3), ("LR", "capT1", 3), ("LR", "cap2", 20), ("R", "cap2", 20)):
        d = {}
        rows, _, arrays, _ = M.run_one(act, arm, 0, 1e-3, 3, MNIST, DEV, epochs=epochs, debug=d)
        want_rows, want_arrays = _recompute(d, act, arm)
        bad = []
        for t, (got, want) in enumerate(zip(rows, want_rows), start=1):
            for k, w in want.items():
                g = got[k]
                if isinstance(w, float) and w != w:
                    ok = g != g
                elif "frac" in k or "proj_rows" in k:
                    ok = g == w
                else:
                    ok = abs(g - w) <= TOL_LOG * max(abs(w), 1e-300)
                if not ok:
                    bad.append(f"t{t}|{k}|got {g} want {w}")
        for k, w in want_arrays.items():
            g = arrays[k].astype(np.float64)
            w32 = np.asarray(w, dtype=np.float64)
            same_nan = np.array_equal(np.isnan(g), np.isnan(w32))
            fin = ~np.isnan(w32)
            ok = same_nan and bool(np.all(np.abs(g[fin] - w32[fin]) <= TOL_F32 * np.maximum(np.abs(w32[fin]), 1e-30)))
            if not ok:
                bad.append(f"units|{k}")
        per[f"{act}|{arm}"] = {"columns": len(want_rows[0]), "arrays": len(want_arrays), "mismatches": bad[:6],
                               "bind_frac_l2_t3": rows[-1].get("bind_frac_l2")}
        failed += [f"{act}|{arm}|{b}" for b in bad[:4]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S6_MUT = [
    ("M6a: uncentered row norms logged", [(A_WT, '        out[f"wt_l{li}"] = torch.linalg.vector_norm(W, dim=1).numpy()\n')]),
    ("M6b: ratio0 divided by the norm of the whole centered tensor",
     [(A_RATIO0, '        f[f"wt_ratio0_med_l{li}"] = float(np.quantile(w / np.linalg.norm(w0), 0.5))\n')]),
    ("M6c: exceed_t1 measured against the init norms", [(A_RT1, '            r_t1 = {li: u0[f"wt_l{li}"].copy() for li in (1, 2)}\n')]),
]


# --------------------------------------------------------------------------
# S7  verdict on synthetic shards
# --------------------------------------------------------------------------

def synth(V, scen: dict) -> tuple[pd.DataFrame, dict]:
    """8 arms x 10 seeds x 50 tasks.  online_acc(t) = A1 - G_arm(t) + seed noise, with G_arm(t) a ramp to the
    arm's plateau by task 10 (plateau split into tasks 31-40 and 41-50 when the scenario asks).  wt_med_l1
    follows init 0.577 -> t1 value -> t50 value linearly; provenance carries hashes, init norms, divergence."""
    rng = np.random.default_rng(scen.get("noise_seed", 1))
    rows, provs = [], {}
    G_ref = {"LR": 0.16, "R": 0.85}
    for act, arm in V.ARMS8:
        for seed in V.SEEDS:
            if seed in scen.get("drop_seeds", ()):
                continue
            rho = scen["rho"].get(f"{act}_{arm}", 0.0 if arm == "ref" else 1.0)
            late = scen.get("late_rho", {}).get(f"{act}_{arm}")
            noise = scen.get("noise", 0.0)
            A1 = 0.97
            e = rng.normal(0, noise)
            wt1 = scen.get("wt1", 2.0)
            wt50 = wt1 * scen.get("gamma", {}).get(act, 17.0)
            for t in range(1, 51):
                g = G_ref[act] * (1 - rho) * min(1.0, (t - 1) / 9)
                if late is not None and t >= 41:
                    g = G_ref[act] * (1 - late)
                wt = wt1 + (wt50 - wt1) * (t - 1) / 49
                rows.append({"act": act, "arm": arm, "seed": seed, "task": t,
                             "online_acc": (A1 if t == 1 else A1 - g + e), "memo_acc": 1.0,
                             "wt_med_l1": wt, "wt_med_l2": wt, "mob_l1": 0.3, "zbar_l1": -0.2, "dead_frac_l1": 0.0,
                             "bind_frac_l1": 0.5, "bind_frac_l2": 0.5, "proj_rows_l1": 10, "proj_rows_l2": 10})
            hs = {k: f"h{seed}" for k in V.STREAM_KEYS}
            if (act, arm, seed) in scen.get("hash_break", ()):
                hs["batch_sha256"] = "broken"
            init_med = scen.get("init_med", {}).get(act, 0.577)
            provs[(act, arm, seed)] = {"runs": {str(seed): {**hs, "tasks_completed": 50,
                                                             "divergence": {"diverged": False},
                                                             "task1_end_state_sha256": "x",
                                                             "init_centered": {"wt_med_l1": init_med, "wt_med_l2": init_med}}}}
    return pd.DataFrame(rows), provs


S7_SCENARIOS = [
    # name, scenario, expected labels at 95%, expected (key, value) of rho/delta estimates
    ("sufficient", {"rho": {"LR_capT1": 0.95, "LR_cap2": 1.0, "LR_l2": 1.0, "R_cap2": 0.95, "R_l2": 1.0}},
     {"Q1": "ACCUMULATION_SUFFICIENT", "Q2": "SIZE_EXPLAINS_L2", "Q2_rho_cap2": "SUFFICIENT", "Q3": "SIZE_RESCUES_RELU"},
     {"rho|LR_capT1": 0.95}),
    ("partial", {"rho": {"LR_capT1": 0.5, "LR_cap2": 0.6, "LR_l2": 1.0, "R_cap2": 0.5, "R_l2": 1.0}},
     {"Q1": "ACCUMULATION_PARTIAL", "Q2": "L2_BEYOND_SIZE", "Q2_rho_cap2": "PARTIAL", "Q3": "SIZE_PARTIAL"},
     {"rho|LR_capT1": 0.5, "delta|LR_cap2-l2": -0.4}),
    ("not_lever", {"rho": {"LR_capT1": 0.02, "LR_cap2": 1.3, "LR_l2": 1.0, "R_cap2": 0.0, "R_l2": 1.0}},
     {"Q1": "ACCUMULATION_NOT_LEVER", "Q2": "CAP_BEYOND_L2", "Q2_rho_cap2": "SUFFICIENT", "Q3": "SIZE_NOT_LEVER"},
     {"rho|LR_capT1": 0.02, "delta|LR_cap2-l2": 0.3}),
    ("near_band", {"rho": {"LR_capT1": 0.85, "LR_cap2": 0.93, "LR_l2": 1.0, "R_cap2": 0.85, "R_l2": 1.0}},
     {"Q1": "ACCUMULATION_PARTIAL", "Q2": "SIZE_EXPLAINS_L2", "Q2_rho_cap2": "SUFFICIENT", "Q3": "SIZE_PARTIAL"},
     {"rho|LR_capT1": 0.85}),
    ("straddle", {"rho": {"LR_capT1": 0.9, "LR_cap2": 0.9, "LR_l2": 1.0, "R_cap2": 0.9, "R_l2": 1.0}, "noise": 0.01},
     {"Q1": "UNRESOLVED", "Q2": "UNRESOLVED"}, {}),
    ("late_window", {"rho": {"LR_capT1": 1.0, "LR_cap2": 1.0, "LR_l2": 1.0, "R_cap2": 1.0, "R_l2": 1.0},
                     "late_rho": {"LR_capT1": 0.84}},
     {"Q1": "ACCUMULATION_SUFFICIENT"}, {"rho|LR_capT1": 0.92}),
    ("weak_gamma", {"rho": {"LR_capT1": 0.5}, "gamma": {"LR": 1.5}, "wt1": 3.5},
     {"Q1": "WEAK_MANIPULATION"}, {}),
    ("weak_kappa_R", {"rho": {"R_cap2": 0.95}, "gamma": {"R": 1.0}, "wt1": 1.0, "init_med": {"R": 0.577}},
     {"Q3": "WEAK_MANIPULATION"}, {}),
    ("incomplete", {"rho": {"LR_capT1": 0.5}, "drop_seeds": (7, 8, 9)},
     {"Q1": "INCOMPLETE", "Q2": "INCOMPLETE", "Q3": "INCOMPLETE"}, {}),
    ("hash_break", {"rho": {"LR_capT1": 0.95, "LR_cap2": 1.0, "LR_l2": 1.0, "R_cap2": 0.95, "R_l2": 1.0},
                    "hash_break": (("R", "l2", 9),)},
     {"Q1": "ACCUMULATION_SUFFICIENT"}, {"n_valid": 9}),
]


def s7(V) -> dict:
    """Runs verdict.analyze on the synthetic scenarios: labels (95%) and the designed estimates must come out
    as written (estimates within 1e-9; noise-free scenarios are exact up to float64 sums)."""
    failed, per = [], {}
    for name, scen, want_labels, want_vals in S7_SCENARIOS:
        pt, provs = synth(V, scen)
        res = V.analyze(pt, provs)
        got = {q: v["95"][0] for q, v in res["labels"].items()}
        bad = [f"{q}: got {got.get(q)} want {w}" for q, w in want_labels.items() if got.get(q) != w]
        for k, w in want_vals.items():
            g = res["n_valid"] if k == "n_valid" else res["stats"].get(k, {}).get("est")
            if g is None or abs(g - w) > 1e-9:
                bad.append(f"{k}: got {g} want {w}")
        per[name] = {"labels": got, "bad": bad}
        failed += [f"{name}|{b}" for b in bad]
    return {"pass": not failed, "failed_items": failed, "detail": per}


V_WIN = 'WIN = (31, 50)                 # G = A(1) - A(WIN)\n'
V_RHO = '    est = 1.0 - Gm.mean() / Gref.mean()\n'
V_GAMMA = '            gamma=g.wt_med_l1.loc[N_TASKS] / g.wt_med_l1.loc[1],\n'
V_BAND = 'BAND_LO, BAND_HI = 0.1, 0.9    # spec 5.1\n'
S7_MUT = [
    ("M7a: window 41-50", [(V_WIN, 'WIN = (41, 50)\n')]),
    ("M7b: rho = mean G_m / mean G_ref (the 1 - dropped)", [(V_RHO, '    est = Gm.mean() / Gref.mean()\n')]),
    ("M7c: gamma's denominator is the init norm", [(V_GAMMA, '            gamma=g.wt_med_l1.loc[N_TASKS] / init_med,\n')]),
    ("M7d: upper band 0.8", [(V_BAND, 'BAND_LO, BAND_HI = 0.1, 0.8\n')]),
]


# --------------------------------------------------------------------------
# smoke and cost
# --------------------------------------------------------------------------

def smoke() -> None:
    """8 arms x seeds {0,1} x 2 tasks x 3 epochs through the CLI, 8 processes at a time (~1 GiB RSS each), then the verdict script with an
    explicit --src: five output files, 32 task rows, 272 layer rows, finite accuracies, units.npz in every
    shard, and every label INCOMPLETE (no 50-task run exists)."""
    t0 = time.time()
    shutil.rmtree(SMOKE, ignore_errors=True)
    runs = SMOKE / "runs"
    runs.mkdir(parents=True)
    procs, rc = [], {}
    for seed in (0, 1):
        batch = []
        for act, arm in ARMS8:
            name = f"{act}_{arm}_s{seed}"
            batch.append((name, cli(runs / name, act, arm, str(seed), 2, 3, wait=False)))
        rc.update({name: p.wait() for name, p in batch})
        procs += batch
    failed = [f"{n}: exit {c}" for n, c in rc.items() if c != 0]
    r = subprocess.run([PY, str(VERDICT), "--src", str(runs), "--out", str(SMOKE), "--checks", str(DUMP_PATH)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        failed.append(f"verdict.py exit {r.returncode}: {r.stderr[-800:]}")
    files = ["per_task.csv", "layer_metrics.csv", "verdict.csv", "summary.md", "provenance.json"]
    present = {f: (SMOKE / f).exists() for f in files}
    failed += [f"missing {f}" for f, ok in present.items() if not ok]
    npz = [n for n, _ in procs if not (runs / n / "units.npz").exists()]
    failed += [f"missing units.npz in {n}" for n in npz]
    detail = {"files": present, "verdict_stdout": r.stdout[-600:]}
    if all(present.values()):
        pt, lm, vd = (pd.read_csv(SMOKE / f) for f in ("per_task.csv", "layer_metrics.csv", "verdict.csv"))
        labels = sorted(set(vd[vd.kind == "label"].label))
        detail.update(n_task_rows=len(pt), n_layer_rows=len(lm), labels=labels)
        if len(pt) != 32:
            failed.append(f"{len(pt)} task rows != 32")
        if len(lm) != 16 * 3 * 6:
            failed.append(f"{len(lm)} layer rows != {16 * 3 * 6}")
        if not np.isfinite(pt.online_acc).all():
            failed.append("non-finite accuracy")
        if labels != ["INCOMPLETE"]:
            failed.append(f"labels {labels} != INCOMPLETE only")
    RESULTS["S-smoke"] = {"pass": not failed, "failed_items": failed, "detail": detail, "seconds": round(time.time() - t0, 1)}
    dump()
    print(f"S-smoke: pass={not failed}  ({RESULTS['S-smoke']['seconds']}s) {failed[:3]}", flush=True)


def meminfo_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    raise RuntimeError("no MemAvailable")


def cost() -> None:
    """8 concurrent runner processes (one per arm; 16 at ~1 GiB RSS each would not fit next to the desktop),
    1 task x 100 epochs each: seconds per 7,500 steps under that load and each process's peak RSS.  Slots
    N = min(16, floor((MemAvailable - 4 GiB) / (1.2 x max peak RSS))) with MemAvailable read before the 8
    start; N < 4 stops the launch (spec 6).  Projected wall for 80 runs x 50 tasks x 30,000 steps on N slots
    in the launch order (seed, act, arm), at the 8-load step time (a projection, not a gate)."""
    t0 = time.time()
    root = SCR / "_runs" / "cost"
    shutil.rmtree(root, ignore_errors=True)
    avail = meminfo_available_gib()
    procs = []
    for act, arm in ARMS8:
        name = f"{act}_{arm}_0"
        procs.append((f"{act}_{arm}", name, cli(root / name, act, arm, "0", 1, 100, wait=False)))
    rc = [p.wait() for _, _, p in procs]
    per_arm, rss = {}, []
    for (arm, name, _), c in zip(procs, rc):
        if c == 0:
            pr = json.loads((root / name / "provenance.json").read_text())
            per_arm.setdefault(arm, []).append(pr["runs"]["0"]["wall_clock_s"])
            rss.append(pr["peak_rss_kb"] / 2 ** 20)
    ok_runs = len(rc) == 8 and all(c == 0 for c in rc)
    peak = max(rss) if rss else float("inf")
    slots = int(min(16, (avail - 4.0) // (1.2 * peak))) if rss else 0
    run_s = {a: 50 * 4 * float(np.mean(v)) + 10.0 for a, v in per_arm.items()}
    finish = 0.0
    if slots > 0 and len(run_s) == len(ARMS8):
        heap = [0.0] * slots
        heapq.heapify(heap)
        for seed in range(10):
            for act, arm in ARMS8:
                start = heapq.heappop(heap)
                end = start + run_s[f"{act}_{arm}"]
                finish = max(finish, end)
                heapq.heappush(heap, end)
    ok = ok_runs and slots >= 4
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": "all 8 processes exit 0 and slots >= 4",
                         "mem_available_gib_before": avail, "peak_rss_gib_max": peak, "slots": slots,
                         "sec_per_100_epochs_under_8_load": {a: float(np.mean(v)) for a, v in per_arm.items()},
                         "projected_run_minutes": {a: s / 60 for a, s in run_s.items()},
                         "projected_wall_hours_80_runs": finish / 3600,
                         "measured_at": dt.datetime.now().astimezone().isoformat(), "seconds": round(time.time() - t0, 1)}
    dump()
    print(f"S-cost: pass={ok}  avail {avail:.1f} GiB  peak RSS {peak:.2f} GiB  slots {slots}  "
          f"80 runs -> {finish / 3600:.2f} h", flush=True)


# --------------------------------------------------------------------------

def main() -> None:
    global DUMP_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of check keys (development; writes a partial file)")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    if only:
        DUMP_PATH = SCR / "checks_partial.json"
    RESULTS.update({"run_id": "wcap_rlmnist_0914", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                    "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (RUNNER, VERDICT, Path(__file__).resolve())},
                    "torch": torch.__version__, "threads": torch.get_num_threads(), "eps32": EPS32, "eps64": EPS64})
    (SCR / "_runs").mkdir(parents=True, exist_ok=True)

    def want(k):
        return only is None or k in only

    V_mod = VERDICT
    table = [
        ("S1", "S1_S-identical", "8 arms: bit-identical init, 1200 images, labels, batch orders", s1, RUNNER, S1_MUT, "bit equality"),
        ("S2", "S2_S-projection", "projection keeps under-cap rows, row means and directions; norm = r", s2, RUNNER, S2_MUT,
         s2.__doc__),
        ("S3", "S3_S-wiring", "hand replay of capped / uncapped steps == runner; activation schedule", s3, RUNNER, S3_MUT,
         "bit equality / integer counts"),
        ("S4", "S4_S-baseline", "new runner == untouched 0913 runner for ref/l2/l2init; host blobs unchanged", s4, RUNNER,
         S4_MUT, "bit equality"),
        ("S6", "S6_S-log", "logged columns and units.npz == independent recomputation", s6, RUNNER, S6_MUT, s6.__doc__),
        ("S7", "S7_S-verdict", "verdict returns the designed labels on synthetic shards", s7, V_mod, S7_MUT,
         "label equality; estimates within 1e-9"),
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

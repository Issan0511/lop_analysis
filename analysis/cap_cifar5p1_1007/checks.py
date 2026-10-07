#!/usr/bin/env python3
"""cap_cifar5p1_1007 -- the registered checks (spec section 7).

    CHECK_SCRATCH=<dir outside git> python analysis/cap_cifar5p1_1007/checks.py [--only S-cap,S-radius]

Every check runs on the real module and on each listed mutant (exactly-once substitutions in the
source, loaded as a fresh module).  It must PASS on the real module and FAIL on every mutant.
Results: results/cap_cifar5p1_1007/checks.json (all_pass = every check passes and detects every
mutant).  Run outputs go under $CHECK_SCRATCH.  Cap arms run on check seeds (100-109) only; the
registered seeds 0-9 are used for the ref arm alone (S-nochange), which is the host's network.
Tolerances are derived from the arithmetic in spec 7.1 / 7.2, never a fixed relative number.
"""
from __future__ import annotations

import argparse
import csv
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import pmnist_0905 as H                    # noqa: E402
from src import rlcifar_mlp_battle_0918 as B        # noqa: E402
from src import cifar5p1_mlp_0920 as C              # noqa: E402

RUN = "cap_cifar5p1_1007"
ENGINE = REPO / "src" / "cap_cifar5p1_1007.py"
VERDICT = REPO / "analysis" / RUN / "verdict.py"
OUT = REPO / "results" / RUN
DUMP = OUT / "checks.json"
HOST_REC = REPO / "results" / "cifar5p1_mlp_0920" / "R_std_lr0.0001"
SCR = Path(os.environ.get("CHECK_SCRATCH", "/tmp")) / f"{RUN}_checks"
SOURCES = ("src/cap_cifar5p1_1007.py", "analysis/cap_cifar5p1_1007/checks.py",
           "analysis/cap_cifar5p1_1007/verdict.py")
THREADS = 2
SEEDS = list(range(10))
CHECK3 = [100, 101, 102]
CHECK10 = list(range(100, 110))
U32, U64 = 2.0 ** -24, 2.0 ** -53
EPS32 = float(torch.finfo(torch.float32).eps)
TINY32 = float(torch.finfo(torch.float32).tiny)
PRINT_REL = 5e-10          # %.10g: half a unit in the 10th significant digit, relative


def gamma(n: int, u: float = U64) -> float:
    return n * u / (1.0 - n * u)


RESULTS: dict = {}
_n_loaded = [0]


def load(path: Path, subs=()):
    """Import `path` as a fresh module after exactly-once substitutions (the real code when empty)."""
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
    mod._is_real = not subs                # S-CLI runs the true command line only for the real module
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def same_bytes(a: Path, b: Path) -> bool:
    return Path(a).exists() and Path(b).exists() and Path(a).read_bytes() == Path(b).read_bytes()


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def brief(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "pass" and not isinstance(v, (dict, list))} | \
        {"failed_items": r.get("failed_items", [])[:30]}


def dump() -> None:
    checks = {k: v for k, v in RESULTS.items() if k.startswith("S-") and isinstance(v, dict)}
    RESULTS["all_pass"] = bool(checks) and all(v.get("pass") and v.get("all_mutations_detected")
                                               for v in checks.values())
    RESULTS["n_checks"] = len(checks)
    RESULTS["n_mutations"] = sum(len(v.get("mutations", [])) for v in checks.values())
    RESULTS["n_mutations_detected"] = sum(m["detected"] for v in checks.values()
                                          for m in v.get("mutations", []))
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    DUMP.write_text(json.dumps(RESULTS, indent=2, default=str))


def run_check(key: str, title: str, fn, mutations: list, derivation: str, target: Path) -> None:
    t0 = time.time()
    base = fn(load(target))
    entry = {"title": title, "target": str(target.relative_to(REPO)), "threshold_derivation": derivation,
             **base, "mutations": []}
    for label, subs in mutations:
        try:
            r = fn(load(target, subs))
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": bool(r["pass"]),
                                       "detected": not r["pass"], "mutant_result": brief(r)})
        except Exception as e:     # recorded; a crash is not how a check should notice a defect
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
    entry["all_mutations_detected"] = bool(entry["mutations"]) and all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:4]) if not entry['pass'] else ''}", flush=True)
    for m in entry["mutations"]:
        if not m["detected"]:
            print(f"   NOT DETECTED: {m['mutation']}", flush=True)


def result(per: dict, extra: dict | None = None) -> dict:
    failed = [k for k, v in per.items() if isinstance(v, bool) and not v]
    return {"pass": not failed, "failed_items": failed, "detail": per, **(extra or {})}


# --------------------------------------------------------------------------
# shared state: device, threads, data, cached host runs
# --------------------------------------------------------------------------

torch.set_num_threads(THREADS)
DEV = H.setup("cuda")
CIFAR = None
_host_cache: dict = {}


def cifar():
    global CIFAR
    if CIFAR is None:
        CIFAR = C.Cifar100()
    return CIFAR


def quiet(_m):
    pass


def erun(M, arm, seeds, tasks, out: Path, **kw):
    shutil.rmtree(out, ignore_errors=True)
    return M.run(arm, seeds, tasks, DEV, out, cifar=cifar(), progress=quiet, **kw)


def hrun(seeds, tasks, steps_hard=C.STEPS_PER_TASK, steps_easy=C.STEPS_PER_TASK, fresh=True) -> Path:
    """The host's unmodified run(), R/std/lr 1e-4, graph -- cached per configuration."""
    key = (tuple(seeds), tasks, steps_hard, steps_easy, fresh)
    if key not in _host_cache:
        out = SCR / f"host_{len(_host_cache)}"
        shutil.rmtree(out, ignore_errors=True)
        C.run("R", list(seeds), "std", tasks, DEV, out, lr=1e-4, cifar=cifar(), fresh=fresh,
              graph=True, iv="none", progress=quiet, steps_hard=steps_hard, steps_easy=steps_easy)
        _host_cache[key] = out
    return _host_cache[key]


def tag(M) -> str:
    return M.__name__


# --------------------------------------------------------------------------
# S-nochange (spec 7 / 7.1)
# --------------------------------------------------------------------------

def phi_slogs(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)


def eff_rank_range(lam_lo: np.ndarray, lam_hi: np.ndarray) -> tuple[float, float]:
    """Range of exp(H) over sorted eigenvalues in [lam_lo, lam_hi] (interval arithmetic, spec 7.1).

    H = log S - T/S, S = sum s, T = sum s log s, s = sqrt(max(lam, 0)).  s log s falls on
    [0, 1/e] and rises after, so each term's min/max over its interval is at an end or at 1/e."""
    s_lo, s_hi = np.sqrt(np.clip(lam_lo, 0, None)), np.sqrt(np.clip(lam_hi, 0, None))
    S_lo, S_hi = float(s_lo.sum()), float(s_hi.sum())
    if S_lo <= 0:
        return 0.0, math.inf
    pl, ph = phi_slogs(s_lo), phi_slogs(s_hi)
    inner = (s_lo < 1 / math.e) & (s_hi > 1 / math.e)
    t_lo = float(np.where(inner, -1 / math.e, np.minimum(pl, ph)).sum())
    t_hi = float(np.maximum(pl, ph).sum())
    q = [t_lo / S_lo, t_lo / S_hi, t_hi / S_lo, t_hi / S_hi]
    h_lo, h_hi = math.log(S_lo) - max(q), math.log(S_hi) - min(q)
    slack = 1e-13                              # float64 evaluation of the bounds themselves
    return math.exp(h_lo) * (1 - slack), math.exp(h_hi) * (1 + slack)


def eff_rank_band(a: torch.Tensor) -> dict:
    """Band for eff_rank(a) across two environments (spec 7.1); a is (N, n) float32 on the device."""
    A = a.double()
    N, n = A.shape
    G = A.T @ A                                                   # as H.eff_rank: on the device
    lam = torch.linalg.eigvalsh(G.cpu()).numpy()
    dG = gamma(N) * float(torch.linalg.matrix_norm(A.abs().T @ A.abs(), "fro"))
    dE = n * U64 * float(np.abs(lam).max())
    d = 2.0 * (dG + dE)
    lo, hi = eff_rank_range(lam - d, lam + d)
    return {"value": H.eff_rank(a), "lo": lo, "hi": hi, "d": d}


def s_nochange(M) -> dict:
    per, extra = {}, {}
    out_e = SCR / f"nochange_{tag(M)}"
    bands = {}

    def obs(t, st):
        z1, a1, z2, a2, _ = B.forward(st["P"], st["Xt"], st["act"], train=False)
        for li, a in ((1, a1), (2, a2)):
            for r in st["live"]:
                bands[(SEEDS[r], t, li)] = eff_rank_band(a[r])

    erun(M, "ref", SEEDS, C.N_TASKS, out_e, observer=obs)
    host = hrun(SEEDS, C.N_TASKS)
    # (a) the host's unmodified run() in this environment, byte for byte
    per["a_per_task_identical_to_host_now"] = same_bytes(out_e / "per_task.csv", host / "per_task.csv")
    per["a_fresh_identical_to_host_now"] = same_bytes(out_e / "fresh_control.csv", host / "fresh_control.csv")
    # (b) the committed record: all columns but eff_rank byte for byte, eff_rank inside the band
    mine, com = read_csv(out_e / "per_task.csv"), read_csv(HOST_REC / "per_task.csv")
    per["b_same_rows"] = len(mine) == len(com) == len(SEEDS) * C.N_TASKS and \
        all((a["seed"], a["task"]) == (b["seed"], b["task"]) for a, b in zip(mine, com))
    cols = list(com[0])
    per["b_same_columns"] = list(mine[0]) == cols
    non_er = [k for k in cols if not k.startswith("eff_rank")]
    diff_cells = [(a["seed"], a["task"], k) for a, b in zip(mine, com) for k in non_er if a[k] != b[k]]
    per["b_non_eff_rank_identical"] = not diff_cells
    extra["b_non_eff_rank_diff_cells"] = diff_cells[:10]
    per["b_fresh_identical_to_committed"] = same_bytes(out_e / "fresh_control.csv", HOST_REC / "fresh_control.csv")
    inside, printed_ok, n_equal, n_rows, worst = True, True, 0, 0, 0.0
    max_rel_dev, max_rel_band, outside = 0.0, 0.0, []
    shifted_rejected, shifted_n = 0, 0
    com_idx = {(int(b["seed"]), int(b["task"])): b for b in com}
    for a, b in zip(mine, com):
        s, t = int(a["seed"]), int(a["task"])
        for li in (1, 2):
            k = f"eff_rank_l{li}"
            bd = bands.get((s, t, li))
            if bd is None:
                inside = False
                outside.append((s, t, li, "no band"))
                continue
            n_rows += 1
            printed_ok &= f"{bd['value']:.10g}" == a[k]          # the observer saw evaluate's matrices
            c = float(b[k])
            lo, hi = bd["lo"] * (1 - 2 * PRINT_REL), bd["hi"] * (1 + 2 * PRINT_REL)
            ok = lo <= c <= hi
            inside &= ok
            if not ok:
                outside.append((s, t, li, c, bd["lo"], bd["hi"]))
            n_equal += a[k] == b[k]
            mv = float(a[k])
            if mv > 0:
                max_rel_dev = max(max_rel_dev, abs(c - mv) / mv)
                max_rel_band = max(max_rel_band, (hi - lo) / 2 / mv)
            nb = com_idx.get((s, t + 1))                          # non-vacuity: the next task's value
            if nb is not None:
                shifted_n += 1
                shifted_rejected += not (lo <= float(nb[k]) <= hi)
    per["b_eff_rank_inside_band"] = inside and n_rows == 2 * len(mine)
    per["b_observer_reproduces_printed_eff_rank"] = printed_ok
    per["b_band_rejects_a_shifted_column"] = shifted_rejected > 0
    extra.update({"eff_rank_cells": n_rows, "eff_rank_cells_byte_equal": n_equal,
                  "eff_rank_max_rel_deviation": max_rel_dev, "eff_rank_max_rel_band_halfwidth": max_rel_band,
                  "eff_rank_outside_band": outside[:10],
                  "shifted_column_rejected": f"{shifted_rejected}/{shifted_n}",
                  "ref_per_task_sha256": sha(out_e / "per_task.csv"),
                  "ref_fresh_sha256": sha(out_e / "fresh_control.csv"),
                  "committed_per_task_sha256": sha(HOST_REC / "per_task.csv"),
                  "host_now_per_task_sha256": sha(host / "per_task.csv"),
                  "torch": torch.__version__, "threads": torch.get_num_threads()})
    # the same comparison on a 4-task run (what the mutants are seen through as well)
    out_s = SCR / f"nochange4_{tag(M)}"
    erun(M, "ref", SEEDS, 4, out_s)
    host4 = hrun(SEEDS, 4)
    per["short_per_task_identical_to_host_now"] = same_bytes(out_s / "per_task.csv", host4 / "per_task.csv")
    per["short_fresh_identical_to_host_now"] = same_bytes(out_s / "fresh_control.csv", host4 / "fresh_control.csv")
    return result(per, extra)


NOCHANGE_MUT = [
    ("M-n1: the ref arm gets the W2 cap", [('CAP_ARMS = {"ref": (),', 'CAP_ARMS = {"ref": (1,),')]),
    ("M-n2: lr 1.0001e-4", [('ACT_ARM, COND, LR = "R", "std", C.LR ', 'ACT_ARM, COND, LR = "R", "std", C.LR * 1.0001 ')]),
    ("M-n3: the Adam step counter restarts every task", [("        tc = train_task(batches, tc)\n",
                                                          "        tc = train_task(batches, 0)\n")]),
]


# --------------------------------------------------------------------------
# S-cap (spec 7 / 7.2)
# --------------------------------------------------------------------------

def bound_ok(post: torch.Tensor, r: torch.Tensor, n32_pre: torch.Tensor, n64_pre: torch.Tensor) -> tuple:
    """spec 7.2: per row, (1-u)^2/(1+th)(1-g)^2 - abs <= ||w'||64 / r ... <= (1+u)^2/(1+th)(1+g)^2 + abs.
    Returns (all rows inside, max |n64/r - 1| / eps32)."""
    d = post.shape[-1]
    n64 = torch.linalg.vector_norm(post.double(), dim=-1, keepdim=True)
    r64 = r.double()
    inv = n64_pre / n32_pre.double()                    # 1/(1+theta)
    g = gamma(d + 1)
    hi = r64 * (1 + U32) ** 2 * inv * (1 + g) ** 2 + math.sqrt(d) * TINY32
    lo = r64 * (1 - U32) ** 2 * inv * (1 - g) ** 2 - math.sqrt(d) * TINY32
    ok = bool(((n64 <= hi) & (n64 >= lo)).all())
    rel = float(((n64 / r64 - 1).abs() / EPS32).max()) if n64.numel() else 0.0
    return ok, rel


def cos_ok(post: torch.Tensor, pre: torch.Tensor) -> tuple:
    d = post.shape[-1]
    a, b = post.double(), pre.double()
    cos = (a * b).sum(-1) / (torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1))
    tol = U32 ** 2 / 2 + 4 * gamma(2 * d + 2)
    worst = float((1 - cos).max()) if cos.numel() else 0.0
    return worst <= tol, worst


def synthetic_projection(M) -> dict:
    """project_rows_ on hand-made rows: over / under / equal / zero (r = 0 and r > 0) / 1-ulp both ways."""
    per = {}
    g = torch.Generator().manual_seed(1007)
    for d in (100, 3072):
        W = ((torch.rand(2, 8, d, generator=g) * 2 - 1) * 0.05).to(DEV)
        W[:, 3] = 0.0
        W[:, 4] = 0.0
        n = torch.linalg.vector_norm(W, dim=2, keepdim=True)
        r = n.clone()
        r[:, 0] = 0.8 * n[:, 0]                                   # over
        r[:, 1] = 1.3 * n[:, 1]                                   # under: must not be inflated
        # row 2: r == n exactly (equal: not over)
        r[:, 3] = 0.0                                             # zero row, zero radius
        r[:, 4] = 0.5                                             # zero row, positive radius
        r[:, 5] = 0.5 * n[:, 5]                                   # over, far
        r[:, 6] = torch.nextafter(n[:, 6], torch.zeros_like(n[:, 6]))         # 1 ulp below: over
        r[:, 7] = torch.nextafter(n[:, 7], torch.full_like(n[:, 7], 10.0))    # 1 ulp above: under
        before = W.clone()
        rows = torch.zeros(2, dtype=torch.long, device=DEV)
        rem = torch.zeros(2, dtype=torch.float64, device=DEV)
        n32 = M.project_rows_(W, r, rows, rem)
        want_over = torch.zeros(2, 8, 1, dtype=torch.bool, device=DEV)
        want_over[:, [0, 5, 6]] = True
        under = ~want_over[..., 0]
        per[f"d{d}_under_equal_zero_rows_bit_identical"] = bool(torch.equal(W[under], before[under]))
        per[f"d{d}_returned_norm_is_row_norm"] = bool(torch.equal(n32, n))
        ov = want_over[..., 0]
        n64_pre = torch.linalg.vector_norm(before.double(), dim=2, keepdim=True)
        ok, rel = bound_ok(W[ov], r[ov], n[ov], n64_pre[ov])
        per[f"d{d}_over_rows_at_radius"] = ok
        per[f"d{d}_over_rows_max_rel_eps32"] = rel
        cok, worst = cos_ok(W[ov], before[ov])
        per[f"d{d}_over_rows_direction_kept"] = cok
        per[f"d{d}_rows_tally"] = rows.tolist() == [3, 3]
        want_rem = (n.double() - r.double()).clamp(min=0).sum(dim=(1, 2))
        per[f"d{d}_rem_tally"] = bool(((rem - want_rem).abs() <= gamma(8) * want_rem + 1e-300).all())
        per[f"d{d}_finite"] = bool(torch.isfinite(W).all())
        again = W.clone()
        M.project_rows_(again, r, torch.zeros_like(rows), torch.zeros_like(rem))
        moved = (again != W).any(dim=2)
        ok2, _ = bound_ok(again[moved], r[moved], torch.linalg.vector_norm(W, dim=2, keepdim=True)[moved],
                          torch.linalg.vector_norm(W.double(), dim=2, keepdim=True)[moved]) if moved.any() else (True, 0)
        per[f"d{d}_second_application_within_bound"] = ok2
        per[f"d{d}_second_application_rows_rewritten"] = int(moved.sum())
    return per


def s_cap(M) -> dict:
    per = synthetic_projection(M)
    st = {"t": 0, "r_ind": None, "pre": None, "fail": [], "over": [0, 0], "under": [0, 0], "equal": [0, 0],
          "max_rel": [0.0, 0.0], "max_cos": [0.0, 0.0], "steps": 0, "rewritten2": [0, 0]}

    def hook(ev, **k):
        if ev == "task_start":
            st["t"] = k["t"]
        elif ev == "task_end" and k["t"] == 1:
            st["r_ind"] = [torch.linalg.vector_norm(k["P"][2 * l].detach(), dim=2, keepdim=True).clone()
                           for l in (0, 1)]
        elif ev == "pre" and st["t"] >= 2:
            st["pre"] = {"P": [q.detach().clone() for q in k["P"]], "m": [q.clone() for q in k["m"]],
                         "v": [q.clone() for q in k["v"]], "rows": k["rows"].clone(), "rem": k["rem"].clone()}
        elif ev == "post" and st["t"] >= 2:
            verify_step(st, k)

    def verify_step(st, k):
        pre, P = st["pre"], [q.detach() for q in k["P"]]
        st["steps"] += 1
        f = st["fail"]
        for i in (1, 3, 4, 5):
            if not torch.equal(P[i], pre["P"][i]):
                f.append(("bias_or_W3_changed", st["t"], i))
        for i in range(6):
            if not torch.equal(k["m"][i], pre["m"][i]) or not torch.equal(k["v"][i], pre["v"][i]):
                f.append(("moment_changed", st["t"], i))
        for l in (0, 1):
            r_used, r_ind = k["rad"][l], st["r_ind"][l]
            if not torch.equal(r_used, r_ind):
                f.append(("radius_is_not_task1_end_norm", st["t"], l))
            w0, w1 = pre["P"][2 * l], P[2 * l]
            n32 = torch.linalg.vector_norm(w0, dim=2, keepdim=True)
            over = n32 > r_ind
            under = ~over[..., 0]
            st["over"][l] += int(over.sum())
            st["under"][l] += int(under.sum())
            st["equal"][l] += int((n32 == r_ind).sum())
            if not torch.equal(w1[under], w0[under]):
                f.append(("under_row_written", st["t"], l))
            ov = over[..., 0]
            if ov.any():
                if not torch.equal(w1[ov], (w0 * (r_ind / n32))[ov]):
                    f.append(("over_row_not_w_r_over_n", st["t"], l))
                n64 = torch.linalg.vector_norm(w0.double(), dim=2, keepdim=True)
                ok, rel = bound_ok(w1[ov], r_ind[ov], n32[ov], n64[ov])
                st["max_rel"][l] = max(st["max_rel"][l], rel)
                if not ok:
                    f.append(("over_row_off_radius", st["t"], l, rel))
                cok, worst = cos_ok(w1[ov], w0[ov])
                st["max_cos"][l] = max(st["max_cos"][l], worst)
                if not cok:
                    f.append(("direction_changed", st["t"], l, worst))
            drows = (k["rows"][:, l] - pre["rows"][:, l]).tolist()
            if drows != over.sum(dim=(1, 2)).tolist():
                f.append(("rows_tally", st["t"], l, drows))
            want = (n32.double() - r_ind.double()).clamp(min=0).sum(dim=(1, 2))
            drem = k["rem"][:, l] - pre["rem"][:, l]
            if not bool(((drem - want).abs() <= gamma(w0.shape[1]) * want + 1e-300).all()):
                f.append(("rem_tally", st["t"], l))
            again = w1.clone()
            M.project_rows_(again, r_ind, torch.zeros(again.shape[0], dtype=torch.long, device=DEV),
                            torch.zeros(again.shape[0], dtype=torch.float64, device=DEV))
            moved = (again != w1).any(dim=2)
            st["rewritten2"][l] += int(moved.sum())
            if moved.any():
                ok2, _ = bound_ok(again[moved], r_ind[moved],
                                  torch.linalg.vector_norm(w1, dim=2, keepdim=True)[moved],
                                  torch.linalg.vector_norm(w1.double(), dim=2, keepdim=True)[moved])
                if not ok2:
                    f.append(("second_application", st["t"], l))

    out = SCR / f"cap_{tag(M)}"
    erun(M, "cap12", CHECK3, 3, out, debug={"hook": hook}, fresh=False, steps_hard=200, steps_easy=200)
    per["engine_steps_verified"] = st["steps"]
    per["engine_all_steps_ok"] = not st["fail"] and st["steps"] == 400
    per["engine_nonvacuous_over_and_under_rows"] = all(x > 0 for x in st["over"] + st["under"])
    extra = {"engine_failures": st["fail"][:10], "engine_n_failures": len(st["fail"]),
             "engine_over_rows": st["over"], "engine_under_rows": st["under"],
             "engine_rows_exactly_at_radius": st["equal"],
             "engine_max_rel_err_eps32": st["max_rel"], "engine_max_1_minus_cos": st["max_cos"],
             "engine_within_4eps32": all(x <= 4.0 for x in st["max_rel"]),
             "engine_second_application_rows_rewritten": st["rewritten2"]}
    return result(per, extra)


CAP_MUT = [
    ("M-c1: the radius is the initial row norm", [
        ("                r_saved = [row_norms(P[2 * l]).detach().clone() for l in range(2)]\n",
         "                r_saved = [row_norms(P0[2 * l]).detach().clone() for l in range(2)]\n")]),
    ("M-c2: the bias of a written row is scaled too", [
        ("                n32 = project_rows_(W, rad[l], rows_acc[:, l], rem_acc[:, l])\n",
         "                n32 = project_rows_(W, rad[l], rows_acc[:, l], rem_acc[:, l]); "
         "P[2 * l + 1].mul_(torch.where(n32[..., 0] > rad[l][..., 0], rad[l][..., 0] / n32[..., 0], "
         "torch.ones_like(n32[..., 0])))\n")]),
    ("M-c3: small rows are inflated to the radius", [("    over = n > r\n", "    over = n > 0\n")]),
    ("M-c4: rows equal to the radius count as over (>=)", [("    over = n > r\n", "    over = n >= r\n")]),
    ("M-c5: the removed norm is not clamped at 0", [
        ("    rem.add_((n.double() - r.double()).clamp(min=0).sum(dim=(1, 2)))\n",
         "    rem.add_((n.double() - r.double()).sum(dim=(1, 2)))\n")]),
]


# --------------------------------------------------------------------------
# S-radius
# --------------------------------------------------------------------------

def s_radius(M) -> dict:
    per = {}
    st = {"t": 0, "j": 0, "r_ind": None, "r_evt": None, "evt_task": None, "t1_inf": True,
          "t2_first_ok": None, "later_ok": True, "n_pre": {}}

    def hook(ev, **k):
        if ev == "task_start":
            st["t"], st["j"] = k["t"], 0
        elif ev == "task_end" and k["t"] == 1:
            st["r_ind"] = [torch.linalg.vector_norm(k["P"][2 * l].detach(), dim=2, keepdim=True).clone()
                           for l in (0, 1)]
        elif ev == "radius":
            st["r_evt"] = [x.clone() for x in k["r_saved"]]
            st["evt_task"] = k["t"]
        elif ev == "pre":
            st["j"] += 1
            st["n_pre"][st["t"]] = st["n_pre"].get(st["t"], 0) + 1
            rad = k["rad"]
            if st["t"] == 1:
                st["t1_inf"] &= all(bool(torch.isinf(x).all()) for x in rad)
            else:
                same = st["r_ind"] is not None and all(torch.equal(rad[l], st["r_ind"][l]) for l in (0, 1))
                if st["t"] == 2 and st["j"] == 1:
                    st["t2_first_ok"] = same
                st["later_ok"] &= same

    out = SCR / f"radius_{tag(M)}"
    erun(M, "cap12", CHECK3, 3, out, debug={"hook": hook}, fresh=False, steps_hard=200, steps_easy=200)
    out_ref = SCR / f"radius_ref_{tag(M)}"
    erun(M, "ref", CHECK3, 3, out_ref, graph=False, fresh=False, steps_hard=200, steps_easy=200)
    per["radius_event_after_task1"] = st["evt_task"] == 1
    per["radius_equals_task1_end_norms"] = (st["r_evt"] is not None and st["r_ind"] is not None and
                                           all(torch.equal(st["r_evt"][l], st["r_ind"][l]) for l in (0, 1)))
    per["task1_radius_inf"] = st["t1_inf"] and st["n_pre"].get(1) == 200
    per["task2_first_update_capped"] = bool(st["t2_first_ok"])
    per["tasks2_3_radius_fixed"] = st["later_ok"]
    cap = read_csv(out / "cap_task.csv")
    per["task1_rows_written_zero"] = all(r["rows_w1"] == "0" and r["rows_w2"] == "0" for r in cap if r["task"] == "1")
    per["task2_rows_written_positive"] = (sum(int(r["rows_w1"]) for r in cap if r["task"] == "2") > 0 and
                                         sum(int(r["rows_w2"]) for r in cap if r["task"] == "2") > 0)
    t1c = [r for r in read_csv(out / "per_task.csv") if r["task"] == "1"]
    t1r = [r for r in read_csv(out_ref / "per_task.csv") if r["task"] == "1"]
    per["task1_rows_identical_to_ref"] = len(t1c) == len(CHECK3) and t1c == t1r
    radii = np.load(out / "radii.npz")
    per["saved_radii_are_task1_end_norms"] = all(
        np.array_equal(radii[f"r_l{l + 1}"], st["r_ind"][l][..., 0].cpu().numpy()) for l in (0, 1))
    return result(per, {"rows_written_task2": {
        "W1": sum(int(r["rows_w1"]) for r in cap if r["task"] == "2"),
        "W2": sum(int(r["rows_w2"]) for r in cap if r["task"] == "2")}})


RADIUS_MUT = [
    ("M-r1: radius from task 2's end, cap from task 3", [("RADIUS_TASK = 1 ", "RADIUS_TASK = 2 "),
                                                        ("CAP_FROM = 2 ", "CAP_FROM = 3 ")]),
    ("M-r2: radius from task 1's end, cap from task 3", [("CAP_FROM = 2 ", "CAP_FROM = 3 ")]),
]


# --------------------------------------------------------------------------
# S-graph
# --------------------------------------------------------------------------

def state_hash(st: dict) -> str:
    h = hashlib.sha256()
    for q in (*st["P"], *st["m"], *st["v"]):
        h.update(q.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def s_graph(M) -> dict:
    per, hashes = {}, {"g": [], "e": []}
    outs = {}
    for mode, graph in (("g", True), ("e", False)):
        outs[mode] = SCR / f"graph_{mode}_{tag(M)}"
        erun(M, "cap12", CHECK10, 4, outs[mode], graph=graph, steps_hard=100, steps_easy=100,
             observer=lambda t, st, mode=mode: hashes[mode].append(state_hash(st)))
    for f in ("per_task.csv", "cap_task.csv", "fresh_control.csv"):
        per[f"{f}_identical"] = same_bytes(outs["g"] / f, outs["e"] / f)
    per["task_end_states_identical"] = len(hashes["g"]) == 4 and hashes["g"] == hashes["e"]
    prov = json.loads((outs["g"] / "provenance.json").read_text())
    per["graph_was_used"] = "CUDA graph" in prov["engine"]
    cap = read_csv(outs["g"] / "cap_task.csv")
    per["projection_active_in_graph_run"] = all(
        sum(int(r[f"rows_w{l}"]) for r in cap if r["task"] == str(t)) > 0 for t in (2, 3, 4) for l in (1, 2))
    return result(per)


GRAPH_MUT = [
    ("M-g1: the radius is re-bound instead of copied in place", [
        ("                    rad[l].copy_(r_saved[l])\n", "                    rad[l] = r_saved[l].clone()\n")]),
    ("M-g2: graph replay reads the next batch", [
        ("                cg.replay()\n",
         "                static_idx.copy_(batches[:, (j + 1) % batches.shape[1]]); cg.replay()\n")]),
    ("M-g3: the warm-up rewind forgets the first Adam moment", [
        ("    mutable = (*P, *adam_m, *adam_v,", "    mutable = (*P, *adam_v,")]),
]


# --------------------------------------------------------------------------
# S-fresh
# --------------------------------------------------------------------------

def s_fresh(M) -> dict:
    per = {}
    cfg = dict(steps_hard=100, steps_easy=100)
    host = hrun(CHECK3, 3, **cfg)
    host_fresh = [(r["seed"], r["fresh_online_acc"]) for r in read_csv(host / "fresh_control.csv")]
    for arm in ("ref", "cap12"):
        out = SCR / f"fresh_{arm}_{tag(M)}"
        erun(M, arm, CHECK3, 3, out, **cfg)
        if arm == "ref":       # the continual column is the host's too, so the whole file must match
            per["ref_fresh_file_identical_to_host"] = same_bytes(out / "fresh_control.csv",
                                                                 host / "fresh_control.csv")
        # the fresh network itself (continual and gap legitimately differ for a cap arm)
        mine = [(r["seed"], r["fresh_online_acc"]) for r in read_csv(out / "fresh_control.csv")]
        per[f"{arm}_fresh_online_identical_to_host"] = len(mine) == len(CHECK3) and mine == host_fresh
        prov = json.loads((out / "provenance.json").read_text())
        per[f"{arm}_fresh_rows_written_zero"] = prov["fresh_rows_written"] == [0, 0]
    st = {"seen": False}
    dbg = {}

    def hook(ev, **k):
        if ev == "fresh_start":
            st["seen"] = True
            st["rad_inf"] = all(bool(torch.isinf(x).all()) for x in k["rad"])
            st["P_is_init"] = all(torch.equal(p.detach().cpu(), q) for p, q in zip(k["P"], dbg["init"]))
            st["moments_zero"] = all(bool((q == 0).all()) for q in (*k["m"], *k["v"]))

    dbg["hook"] = hook
    erun(M, "cap12", CHECK3, 3, SCR / f"fresh_hook_{tag(M)}", debug=dbg, **cfg)
    per["hook_saw_fresh_start"] = st["seen"]
    per["fresh_radius_inf"] = bool(st.get("rad_inf"))
    per["fresh_starts_from_init"] = bool(st.get("P_is_init"))
    per["fresh_moments_zero"] = bool(st.get("moments_zero"))
    return result(per)


FRESH_MUT = [
    ("M-f1: the cap stays on for the fresh network", [("                rad[l].fill_(math.inf)\n",
                                                       "                pass\n")]),
    ("M-f2: the Adam moments are not reset for the fresh network", [
        ("            for q in (*adam_m, *adam_v):\n                q.zero_()\n",
         "            for q in ():\n                q.zero_()\n")]),
]


# --------------------------------------------------------------------------
# S-verdict (synthetic fixtures)
# --------------------------------------------------------------------------

def pattern(mean: float, sd: float) -> np.ndarray:
    """10 values with exactly this mean and sample SD (a fixed zero-mean shape, rescaled)."""
    z = np.array([-1.5, -1.1, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.1, 1.5])
    z = (z - z.mean()) / z.std(ddof=1)
    return mean + sd * z


def ep_fixture(d1: dict, d2: dict, d0: dict | None = None, ref_gap=None) -> dict:
    """Endpoint arrays: ref has a large between-seed spread; arms are ref + the given deltas."""
    base1 = np.array([.38, .47, .41, .44, .50, .36, .43, .40, .48, .42])
    base0 = base1 + .18
    gap = np.array([.25, .19, .22, .27, .20, .23, .21, .26, .18, .24]) if ref_gap is None else ref_gap
    ep = {"ref": {"E0": base0, "E1": base1, "E2": gap}}
    for arm in ("cap1", "cap2", "cap12"):
        ep[arm] = {"E0": base0 + (d0 or {}).get(arm, np.zeros(10)), "E1": base1 + d1[arm], "E2": gap + d2[arm]}
    return ep


def written(zero_task: dict | None = None) -> dict:
    out = {}
    for arm, layers in {"cap1": (1,), "cap2": (2,), "cap12": (1, 2)}.items():
        out[arm] = {}
        for l in layers:
            v = np.full(29, 7)
            if zero_task and (arm, l) in zero_task:
                v[zero_task[(arm, l)] - 2] = 0
            out[arm][l] = v
    return out


GOOD = {"complete": True, "consistent": True}
POS, NEG, ZERO = pattern(.05, .01), pattern(-.05, .01), pattern(0.0, .03)
BORDER = pattern(2.45 * .01 / math.sqrt(10), .01)          # t = 2.45: + at 95 %, 0 at 97.5 %


def s_verdict(V) -> dict:
    per = {}
    per["t_975_table"] = abs(V.t_quantile(0.975, 9) - 2.2622) < 1e-4
    per["t_9875_table"] = abs(V.t_quantile(0.9875, 9) - 2.6850) < 1e-4
    per["endpoint_zero_is_0"] = (V.sign(0.0, 0.3) == "0" and V.sign(-0.3, 0.0) == "0"
                                 and V.sign(1e-12, 1.0) == "+" and V.sign(-1.0, -1e-12) == "-")
    sd0 = [V.interval(np.full(10, c), 0.95) for c in (0.02, 0.0, -0.02)]
    per["sd0_signs"] = [q["sign"] for q in sd0] == ["+", "0", "-"]
    per["sd0_degenerate_point"] = all(q["degenerate_sd"] and q["lo"] == q["hi"] == q["mean"] for q in sd0)
    table = {("+", "-"): "RESCUED", ("+", "0"): "LEVEL_ONLY", ("+", "+"): "LEVEL_UP_GAP_UP",
             ("0", "-"): "GAP_ONLY", ("0", "0"): "NO_EFFECT", ("0", "+"): "GAP_UP",
             ("-", "-"): "WORSE", ("-", "0"): "WORSE", ("-", "+"): "WORSE"}
    per["label_table"] = all(V.label(*k) == v for k, v in table.items())
    # every label reached through judge()
    cases = {"RESCUED": (POS, NEG), "LEVEL_ONLY": (POS, ZERO), "LEVEL_UP_GAP_UP": (POS, -NEG),
             "GAP_ONLY": (ZERO, NEG), "NO_EFFECT": (ZERO, ZERO), "GAP_UP": (ZERO, -NEG), "WORSE": (NEG, NEG)}
    got = {}
    for lab, (a, b) in cases.items():
        v = V.judge(ep_fixture({k: a for k in ("cap1", "cap2", "cap12")}, {k: b for k in ("cap1", "cap2", "cap12")}),
                    written(), GOOD)
        got[lab] = [v["arms"][k]["label"] for k in ("cap1", "cap2", "cap12")]
    per["all_labels_via_judge"] = all(g == [lab] * 3 for lab, g in got.items())
    # Bonferroni: the same borderline delta1 is + for cap1 (95 %) and 0 for cap12 (97.5 %)
    v = V.judge(ep_fixture({k: BORDER for k in ("cap1", "cap2", "cap12")}, {k: NEG for k in ("cap1", "cap2", "cap12")}),
                written(), GOOD)
    per["bonferroni_cap12_only"] = (v["arms"]["cap1"]["label"] == "RESCUED" and v["arms"]["cap12"]["label"] == "GAP_ONLY"
                                    and v["arms"]["cap12"]["level"] == 0.975 and v["arms"]["cap12"]["d1_95"]["sign"] == "+")
    # pairing: a consistent paired delta under a large between-seed spread is '+'
    per["paired_not_unpaired"] = v["arms"]["cap2"]["d2"]["sign"] == "-" and \
        V.judge(ep_fixture({k: pattern(.006, .002) for k in ("cap1", "cap2", "cap12")},
                           {k: ZERO for k in ("cap1", "cap2", "cap12")}), written(), GOOD)["arms"]["cap1"]["d1"]["sign"] == "+"
    # IMPAIRED: tight negative delta0 -> flag; negative mean with a wide interval -> no flag;
    # borderline negative (t = -2.45) -> flag at 95 % (it would not be at 97.5 %)
    imp = {}
    for name, d0 in (("tight", pattern(-.02, .005)), ("wide", pattern(-.01, .05)), ("border", -BORDER)):
        v = V.judge(ep_fixture({k: ZERO for k in ("cap1", "cap2", "cap12")}, {k: ZERO for k in ("cap1", "cap2", "cap12")},
                               d0={k: d0 for k in ("cap1", "cap2", "cap12")}), written(), GOOD)
        imp[name] = v["arms"]["cap12"]["impaired"]
    per["impaired_flags"] = imp == {"tight": True, "wide": False, "border": True}
    # (B) not reproduced, (C) not applicable, (A) incomplete / inconsistent
    v = V.judge(ep_fixture({k: POS for k in ("cap1", "cap2", "cap12")}, {k: NEG for k in ("cap1", "cap2", "cap12")},
                           ref_gap=pattern(0.0, .02)), written(), GOOD)
    per["B_fails_not_reproduced"] = v["status"] == "NOT_REPRODUCED" and all(
        a["label"] == "NOT_REPRODUCED" for a in v["arms"].values())
    v = V.judge(ep_fixture({k: POS for k in ("cap1", "cap2", "cap12")}, {k: NEG for k in ("cap1", "cap2", "cap12")}),
                written({("cap2", 2): 17}), GOOD)
    per["C_fails_inapplicable"] = (v["arms"]["cap2"]["label"] == "INAPPLICABLE" and v["arms"]["cap1"]["label"] == "RESCUED"
                                   and v["applicability"]["C"]["cap2"]["zero_tasks"] == {"W2": [17]})
    per["A_incomplete"] = V.judge({}, {}, {"complete": False, "consistent": False})["status"] == "INCOMPLETE"
    per["A_inconsistent"] = V.judge({}, {}, {"complete": True, "consistent": False})["status"] == "CHECK_FAILED"
    # interaction is delta12 - delta1 - delta2: additive arms plus zero-mean scatter -> mean 0, sign 0;
    # a superadditive cap12 -> '+'
    v = V.judge(ep_fixture({"cap1": POS, "cap2": POS, "cap12": 2 * POS + pattern(0.0, .02)},
                           {k: NEG for k in ("cap1", "cap2", "cap12")}), written(), GOOD)
    w = V.judge(ep_fixture({"cap1": POS, "cap2": POS, "cap12": 3 * POS}, {k: NEG for k in ("cap1", "cap2", "cap12")}),
                written(), GOOD)
    per["interaction_definition"] = (abs(v["interaction"]["E1"]["mean"]) < 1e-12 and v["interaction"]["E1"]["sign"] == "0"
                                     and w["interaction"]["E1"]["sign"] == "+")
    # windows through endpoints(): arms differ only at t = 11 and t = 19 (outside both windows)
    per["windows"] = window_fixture(V)
    return result(per, {"labels_via_judge": got})


def window_fixture(V) -> bool:
    def rows(shift):
        pt, fr = [], []
        for s in range(10):
            for t in range(1, 31):
                on = 0.6 - 0.005 * t + 0.01 * s + (shift if t in (11, 19) else 0.0)
                pt.append({"seed": s, "task": t, "online_acc": on})
            fr.append({"seed": s, "task": 29, "fresh_gap": 0.2})
        return pt, fr
    ep_ref = V.endpoints(*rows(0.0))
    ep_arm = V.endpoints(*rows(0.1))
    return all(np.array_equal(ep_ref[k], ep_arm[k]) for k in ("E0", "E1", "E2"))


VERDICT_MUT = [
    ("M-v1: no Bonferroni (cap12 at 95 %)", [("MAIN_LEVEL = 0.975 ", "MAIN_LEVEL = 0.95 ")]),
    ("M-v2: late window hard 19-27", [("LATE = (21, 23, 25, 27, 29)", "LATE = (19, 21, 23, 25, 27)")]),
    ("M-v3: pairing broken (reference seeds shuffled)", [
        ('    d = {arm: {k: ep[arm][k] - ep["ref"][k] for k in ("E0", "E1", "E2")} for arm in CAPPED}\n',
         '    d = {arm: {k: ep[arm][k] - np.random.default_rng(0).permutation(ep["ref"][k]) '
         'for k in ("E0", "E1", "E2")} for arm in CAPPED}\n')]),
    ("M-v4: an endpoint at 0 counts as a sign (>=)", [
        ('    return "+" if lo > 0 else "-" if hi < 0 else "0"\n', '    return "+" if lo >= 0 else "-" if hi <= 0 else "0"\n')]),
    ("M-v5: IMPAIRED judged at 97.5 %", [
        ('             "d0": interval(d[arm]["E0"], COMP_LEVEL),\n', '             "d0": interval(d[arm]["E0"], MAIN_LEVEL),\n')]),
    ("M-v6: IMPAIRED judged on the mean", [
        ('        a["impaired"] = a["d0"]["hi"] < 0\n', '        a["impaired"] = a["d0"]["mean"] < 0\n')]),
]


# --------------------------------------------------------------------------
# S-CLI
# --------------------------------------------------------------------------

CLI_ARGS = ["run", "--arm", "cap1", "--seeds", "100-101", "--tasks", "3", "--steps-hard", "60",
            "--steps-easy", "40"]
_cli_ref: dict = {}


def cli_reference() -> Path:
    """The function call with the CLI's arguments, on the real module (computed once)."""
    if "out" not in _cli_ref:
        M = load(ENGINE)
        out = SCR / "cli_function"
        erun(M, "cap1", [100, 101], 3, out, steps_hard=60, steps_easy=40)
        _cli_ref["out"] = out
    return _cli_ref["out"]


def s_cli(M) -> dict:
    per, extra = {}, {}
    a = M.build_parser().parse_args(["run", "--arm", "ref"])
    per["defaults_registered"] = (a.seeds == "0-9" and a.tasks == 30 and a.lr == 1e-4 and a.threads == 2
                                  and a.steps_hard == 780 and a.steps_easy == 780 and not a.no_fresh
                                  and not a.no_graph and a.out is None)
    ref = cli_reference()
    out = SCR / f"cli_inproc_{tag(M)}"
    shutil.rmtree(out, ignore_errors=True)
    M.main(CLI_ARGS + ["--out", str(out)])
    torch.set_num_threads(THREADS)
    for f in ("per_task.csv", "cap_task.csv", "fresh_control.csv"):
        per[f"inprocess_cli_{f}_identical_to_function"] = same_bytes(out / f, ref / f)
    # the true command line, only for the real module (a mutant is not on disk)
    if getattr(M, "_is_real", False):
        outp = SCR / "cli_subprocess"
        shutil.rmtree(outp, ignore_errors=True)
        env = {**os.environ, "OMP_NUM_THREADS": str(THREADS)}
        p = subprocess.run([sys.executable, "-m", "src.cap_cifar5p1_1007", *CLI_ARGS, "--out", str(outp)],
                           cwd=REPO, capture_output=True, text=True, env=env)
        per["subprocess_exit_0"] = p.returncode == 0
        for f in ("per_task.csv", "cap_task.csv", "fresh_control.csv"):
            per[f"subprocess_{f}_identical_to_function"] = same_bytes(outp / f, ref / f)
        # the registered location refuses a non-registered configuration before writing anything
        probe = OUT / "_cli_guard_probe"
        p = subprocess.run([sys.executable, "-m", "src.cap_cifar5p1_1007", *CLI_ARGS, "--out", str(probe)],
                           cwd=REPO, capture_output=True, text=True, env=env)
        per["registered_location_refuses_unregistered_config"] = p.returncode != 0 and not probe.exists()
        shutil.rmtree(probe, ignore_errors=True)
    # admit(): wrong config, missing checks, failing checks, complete checks on a clean tree
    tmp = SCR / f"admit_{tag(M)}"
    tmp.mkdir(parents=True, exist_ok=True)
    srcs = {s: sha(REPO / s) for s in M.SOURCES}

    def refuses(cfg, path, reason):
        """True only when admit() refuses for the expected reason (not, say, a dirty tree)."""
        try:
            M.admit(cfg, checks_json=path, repo=REPO)
            return False
        except SystemExit as e:
            return reason in str(e)

    good = {k: {"pass": True} for k in M.REQUIRED_CHECKS} | {"all_pass": True, "source_sha256": srcs}
    (tmp / "good.json").write_text(json.dumps(good))
    (tmp / "failing.json").write_text(json.dumps(good | {"all_pass": False}))
    (tmp / "stale.json").write_text(json.dumps(good | {"source_sha256": {s: "0" * 64 for s in srcs}}))
    per["admit_refuses_wrong_config"] = refuses(dict(M.REGISTERED, tasks=3), tmp / "good.json",
                                                "not the registered configuration")
    per["admit_refuses_missing_checks"] = refuses(dict(M.REGISTERED), tmp / "missing.json", "missing")
    per["admit_refuses_failing_checks"] = refuses(dict(M.REGISTERED), tmp / "failing.json", "not all_pass")
    per["admit_refuses_stale_sources"] = refuses(dict(M.REGISTERED), tmp / "stale.json", "sources changed")
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "analysis", "specs"],
                           capture_output=True, text=True).stdout.strip()
    extra["tree_clean"] = not dirty
    per["admit_accepts_complete_checks_on_clean_tree"] = (not dirty) and not refuses(dict(M.REGISTERED),
                                                                                     tmp / "good.json", "")
    return result(per, extra)


CLI_MUT = [
    ("M-l1: the CLI passes --steps-easy as --steps-hard", [
        ("    cfg = dict(seeds=B.parse_ints(a.seeds), tasks=a.tasks, lr=a.lr, steps_hard=a.steps_hard,\n",
         "    cfg = dict(seeds=B.parse_ints(a.seeds), tasks=a.tasks, lr=a.lr, steps_hard=a.steps_easy,\n")]),
    ("M-l2: the default task count is 20", [('    ap.add_argument("--tasks", type=int, default=C.N_TASKS)\n',
                                            '    ap.add_argument("--tasks", type=int, default=20)\n')]),
    ("M-l3: admit() ignores all_pass", [('    if missing or not ck.get("all_pass"):\n', "    if missing:\n")]),
]


# --------------------------------------------------------------------------

CHECKS = {
    "S-verdict": ("labels, endpoints, SD 0, IMPAIRED, Bonferroni, (A)(B)(C) on synthetic fixtures",
                  s_verdict, VERDICT_MUT, VERDICT,
                  "exact expected labels; t quantiles vs the printed table to 1e-4 (4-decimal table)"),
    "S-cap": ("projection: under rows bit-unchanged, over rows at the radius with direction kept, "
              "nothing else written, tallies, second application", s_cap, CAP_MUT, ENGINE,
              "spec 7.2: per-row bound from u32 (ratio and product roundings), the measured error theta of "
              "the float32 norm, gamma64_{d+1} for the float64 checker and sqrt(d)*tiny32 for underflow; "
              "1 - cos <= u32^2/2 + 4 gamma64_{2d+2}; rem within gamma64_rows * sum"),
    "S-radius": ("radius = task-1-end row norms, +inf through task 1, active from task 2's first update",
                 s_radius, RADIUS_MUT, ENGINE, "bit equality"),
    "S-fresh": ("fresh control: the host's procedure, no cap", s_fresh, FRESH_MUT, ENGINE,
                "byte equality with the host's unmodified run() under the same configuration"),
    "S-graph": ("graph and eager runs agree bit for bit (cap12, R = 10, 4 tasks x 100 updates)",
                s_graph, GRAPH_MUT, ENGINE, "byte / bit equality"),
    "S-CLI": ("CLI = function call, registered defaults, admission guard", s_cli, CLI_MUT, ENGINE,
              "byte equality"),
    "S-nochange": ("ref = host: (a) byte-identical to the host's run() now; (b) committed record: all but "
                   "eff_rank byte-identical, eff_rank inside the spec 7.1 band", s_nochange, NOCHANGE_MUT, ENGINE,
                   "spec 7.1: |lam_hat - lam| <= gamma_N || |A|^T|A| ||_F + n u ||G||_2 per environment, Weyl, "
                   "interval arithmetic on H = log S - T/S, plus 2 x 5e-10 for the 10-digit printing"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated check ids (debugging; all_pass needs all)")
    a = ap.parse_args()
    SCR.mkdir(parents=True, exist_ok=True)
    keep = json.loads(DUMP.read_text()) if (a.only and DUMP.exists()) else {}
    RESULTS.update(keep)
    RESULTS.update({"run_id": RUN, "spec": "specs/spec_cap_cifar5p1_1007.md",
                    "source_sha256": {s: sha(REPO / s) for s in SOURCES},
                    "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                               capture_output=True, text=True).stdout.strip(),
                    "torch": torch.__version__, "cuda": torch.version.cuda, "threads": torch.get_num_threads(),
                    "scratch": str(SCR), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    only = [x for x in a.only.split(",") if x]
    for key, (title, fn, muts, target, deriv) in CHECKS.items():
        if only and key not in only:
            continue
        run_check(key, title, fn, muts, deriv, target)
    RESULTS["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump()
    print(f"all_pass={RESULTS['all_pass']}  checks {RESULTS['n_checks']}  mutations detected "
          f"{RESULTS['n_mutations_detected']}/{RESULTS['n_mutations']}", flush=True)


if __name__ == "__main__":
    main()

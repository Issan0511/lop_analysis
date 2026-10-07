#!/usr/bin/env python3
"""baselines_cifar5p1_1008 -- the registered checks (spec section 8).

    CHECK_SCRATCH=<dir outside git> python analysis/baselines_cifar5p1_1008/checks.py [--only S-cbp,S-redo]

Every check runs on the real module and on each listed mutant (exactly-once substitutions in the
source, loaded as a fresh module) and must PASS on the real one and FAIL on every mutant.
S-host-repro tests the environment, not this code, so its mutants are wrong comparisons.
Results: results/baselines_cifar5p1_1008/checks.json (all_pass = every check passes and detects
every mutant).  Run outputs go under $CHECK_SCRATCH.  Method arms run on check seeds (100-109)
and short runs only; seeds 0-19 are used for the host's own arms (S-nochange, S-host-repro).
Tolerances come from the arithmetic of the operation checked, never a fixed relative number.
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
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import pmnist_0905 as H                    # noqa: E402
from src import rlcifar_mlp_battle_0918 as B        # noqa: E402
from src import cifar5p1_mlp_0920 as C              # noqa: E402

RUN = "baselines_cifar5p1_1008"
ENGINE = REPO / "src" / f"{RUN}.py"
VERDICT = REPO / "analysis" / RUN / "verdict.py"
OUT = REPO / "results" / RUN
DUMP = OUT / "checks.json"
HOST_DIR = REPO / "results" / "cifar5p1_mlp_0920"
HOST_REC = HOST_DIR / "R_std_lr0.0001"
SCR = Path(os.environ.get("CHECK_SCRATCH", "/tmp")) / f"{RUN}_checks"
SOURCES = ("src/baselines_cifar5p1_1008.py", "analysis/baselines_cifar5p1_1008/checks.py",
           "analysis/baselines_cifar5p1_1008/verdict.py")
THREADS = 2
SEEDS = list(range(10))
CHECK3 = [100, 101, 102]
CHECK10 = list(range(100, 110))
U32, U64 = 2.0 ** -24, 2.0 ** -53
TINY32 = float(torch.finfo(torch.float32).tiny)
PRINT_REL = 5e-10          # %.10g: half a unit in the 10th significant digit, relative
FAN_IN = (C.DIMS[0], C.DIMS[0], C.DIMS[1], C.DIMS[1], C.DIMS[2], C.DIMS[2])   # W1 b1 W2 b2 W3 b3
BETA = tuple(1.0 / math.sqrt(f) for f in FAN_IN)
LR = 1e-4
AB1, AB2, AEPS = 0.9, 0.999, 1e-8


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
    mod._is_real = not subs
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


def finish(key: str, entry: dict, t0: float) -> None:
    entry["all_mutations_detected"] = bool(entry["mutations"]) and all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:6]) if not entry['pass'] else ''}", flush=True)
    for m in entry["mutations"]:
        if not m["detected"]:
            print(f"   NOT DETECTED: {m['mutation']}", flush=True)


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
        except AssertionError:
            raise                     # a broken anchor is a bug in the check, not a detection
        except Exception as e:        # recorded; a crash is not how a check should notice a defect
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
    finish(key, entry, t0)


def result(per: dict, extra: dict | None = None) -> dict:
    failed = [k for k, v in per.items() if isinstance(v, bool) and not v]
    return {"pass": not failed, "failed_items": failed, "detail": per, **(extra or {})}


# --------------------------------------------------------------------------
# shared state
# --------------------------------------------------------------------------

torch.set_num_threads(THREADS)
DEV = H.setup(os.environ.get("CHECK_DEVICE", "cuda"))      # cpu only to debug the checks themselves
CIFAR = None
_host_cache: dict = {}


def cifar():
    global CIFAR
    if CIFAR is None:
        CIFAR = C.Cifar100()
    return CIFAR


def quiet(_m):
    pass


def erun(M, cfg, seeds, tasks, out: Path, **kw):
    shutil.rmtree(out, ignore_errors=True)
    return M.run(cfg, seeds, tasks, DEV, out, cifar=cifar(), progress=quiet, **kw)


def hrun(seeds, tasks, steps_hard=C.STEPS_PER_TASK, steps_easy=C.STEPS_PER_TASK, fresh=True,
         arm="R", iv="none") -> Path:
    """The host's unmodified run(), std / lr 1e-4 / graph -- cached per configuration."""
    key = (arm, iv, tuple(seeds), tasks, steps_hard, steps_easy, fresh)
    if key not in _host_cache:
        out = SCR / f"host_{len(_host_cache)}"
        shutil.rmtree(out, ignore_errors=True)
        C.run(arm, list(seeds), "std", tasks, DEV, out, lr=1e-4, cifar=cifar(), fresh=fresh,
              graph=True, iv=iv, progress=quiet, steps_hard=steps_hard, steps_easy=steps_easy)
        _host_cache[key] = out
    return _host_cache[key]


def tag(M) -> str:
    return M.__name__


def init_stack(seeds) -> list[torch.Tensor]:
    init = [q.detach() for s in seeds for q in C.init_params("R", s, DEV)]
    return [torch.stack(init[i::6]).contiguous() for i in range(6)]


def clone_all(xs) -> list[torch.Tensor]:
    return [x.detach().clone() for x in xs]


def bits_equal(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b)) and len(a) == len(b)


def same_but_eff_rank(a: Path, b: Path) -> tuple[bool, list]:
    A = [l.split(",") for l in Path(a).read_text().splitlines()]
    Bt = [l.split(",") for l in Path(b).read_text().splitlines()]
    if len(A) != len(Bt) or A[0] != Bt[0]:
        return False, ["shape or header"]
    er = {i for i, c in enumerate(A[0]) if c.startswith("eff_rank")}
    diff = [(ra[2], ra[5], A[0][i]) for ra, rb in zip(A[1:], Bt[1:]) for i, (x, y) in enumerate(zip(ra, rb))
            if i not in er and x != y]
    return not diff, diff[:10]


# --------------------------------------------------------------------------
# S-nochange (spec 8; eff_rank band = cap_cifar5p1_1007 spec 7.1, functions copied from
# analysis/cap_cifar5p1_1007/checks.py at origin/main 49ca6ade)
# --------------------------------------------------------------------------

def phi_slogs(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)


def eff_rank_range(lam_lo: np.ndarray, lam_hi: np.ndarray) -> tuple[float, float]:
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
    slack = 1e-13
    return math.exp(h_lo) * (1 - slack), math.exp(h_hi) * (1 + slack)


def eff_rank_band(a: torch.Tensor) -> dict:
    A = a.double()
    N, n = A.shape
    G = A.T @ A
    lam = torch.linalg.eigvalsh(G.cpu()).numpy()
    dG = gamma(N) * float(torch.linalg.matrix_norm(A.abs().T @ A.abs(), "fro"))
    dE = n * U64 * float(np.abs(lam).max())
    d = 2.0 * (dG + dE)
    lo, hi = eff_rank_range(lam - d, lam + d)
    return {"value": H.eff_rank(a), "lo": lo, "hi": hi, "d": d}


def s_nochange(M) -> dict:
    per, extra = {}, {}
    none = M.make_cfg("none")
    out_e = SCR / f"nochange_{tag(M)}"
    bands = {}

    def obs(t, st):
        z1, a1, z2, a2, _ = B.forward(st["P"], st["Xt"], st["act"], train=False)
        for li, a in ((1, a1), (2, a2)):
            for r in st["live"]:
                bands[(SEEDS[r], t, li)] = eff_rank_band(a[r])

    erun(M, none, SEEDS, C.N_TASKS, out_e, observer=obs)
    host = hrun(SEEDS, C.N_TASKS)
    per["a_per_task_identical_to_host_now"] = same_bytes(out_e / "per_task.csv", host / "per_task.csv")
    per["a_fresh_identical_to_host_now"] = same_bytes(out_e / "fresh_control.csv", host / "fresh_control.csv")
    mine, com = read_csv(out_e / "per_task.csv"), read_csv(HOST_REC / "per_task.csv")
    per["b_same_rows"] = len(mine) == len(com) == len(SEEDS) * C.N_TASKS and \
        all((a["seed"], a["task"]) == (b["seed"], b["task"]) for a, b in zip(mine, com))
    cols = list(com[0])
    per["b_same_columns"] = list(mine[0]) == cols
    ok, diff = same_but_eff_rank(out_e / "per_task.csv", HOST_REC / "per_task.csv")
    per["b_non_eff_rank_identical"] = ok
    extra["b_non_eff_rank_diff_cells"] = diff
    per["b_fresh_identical_to_committed"] = same_bytes(out_e / "fresh_control.csv", HOST_REC / "fresh_control.csv")
    inside, printed_ok, n_equal, n_rows = True, True, 0, 0
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
            printed_ok &= f"{bd['value']:.10g}" == a[k]
            c = float(b[k])
            lo, hi = bd["lo"] * (1 - 2 * PRINT_REL), bd["hi"] * (1 + 2 * PRINT_REL)
            okk = lo <= c <= hi
            inside &= okk
            if not okk:
                outside.append((s, t, li, c, bd["lo"], bd["hi"]))
            n_equal += a[k] == b[k]
            mv = float(a[k])
            if mv > 0:
                max_rel_dev = max(max_rel_dev, abs(c - mv) / mv)
                max_rel_band = max(max_rel_band, (hi - lo) / 2 / mv)
            nb = com_idx.get((s, t + 1))
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
                  "none_per_task_sha256": sha(out_e / "per_task.csv"),
                  "none_fresh_sha256": sha(out_e / "fresh_control.csv"),
                  "committed_per_task_sha256": sha(HOST_REC / "per_task.csv"),
                  "host_now_per_task_sha256": sha(host / "per_task.csv"),
                  "torch": torch.__version__, "threads": torch.get_num_threads()})
    out_s = SCR / f"nochange4_{tag(M)}"
    erun(M, none, SEEDS, 4, out_s)
    host4 = hrun(SEEDS, 4)
    per["c_short_per_task_identical_to_host_now"] = same_bytes(out_s / "per_task.csv", host4 / "per_task.csv")
    per["c_short_fresh_identical_to_host_now"] = same_bytes(out_s / "fresh_control.csv", host4 / "fresh_control.csv")
    return result(per, extra)


NOCHANGE_MUT = [
    ("M-n1: a tiny shrink after every (non-CBP) Adam update",
     [("p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))",
       "p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps)).mul_(1 - 1e-7)")]),
    ("M-n2: lr 1.0001e-4", [('ACT_ARM, COND, LR = "R", "std", C.LR ', 'ACT_ARM, COND, LR = "R", "std", C.LR * 1.0001 ')]),
    ("M-n3: the Adam step counter restarts every task",
     [('        tc, pending = train_task(batches, tc, method != "none", "train", t)\n',
       '        tc, pending = train_task(batches, 0, method != "none", "train", t)\n')]),
]


# --------------------------------------------------------------------------
# S-host-repro: the comparator records still come out of the host in this environment
# --------------------------------------------------------------------------

HOST_CELLS = {("SNA", "none", 0): "SNA_std_lr0.0001", ("SNA", "none", 10): "SNA_s10-19",
              ("KKT1", "none", 0): "KKT1_std_lr0.0001", ("KKT1", "none", 10): "KKT1_s10-19",
              ("R", "l2init:1e-3", 0): "R_std_lr0.0001_l2init1e-3", ("R", "l2init:1e-3", 10): "R_s10-19l2init:1e3"}


def s_host_repro() -> None:
    t0 = time.time()
    outs = {}
    for (arm, iv, s0), cell in HOST_CELLS.items():
        outs[(arm, iv, s0)] = hrun(list(range(s0, s0 + 10)), C.N_TASKS, arm=arm, iv=iv)

    def compare(pairs) -> dict:
        per = {}
        for (key, cell) in pairs:
            ok, diff = same_but_eff_rank(outs[key] / "per_task.csv", HOST_DIR / cell / "per_task.csv")
            per[f"{key[0]}|{key[1]}|s{key[2]}_vs_{cell}_per_task"] = ok
            per[f"{key[0]}|{key[1]}|s{key[2]}_vs_{cell}_fresh"] = same_bytes(outs[key] / "fresh_control.csv",
                                                                          HOST_DIR / cell / "fresh_control.csv")
        return result(per)

    base = compare(list(HOST_CELLS.items()))
    entry = {"title": "comparators reproduce from the host in this environment", "target": "src/cifar5p1_mlp_0920.py",
             "threshold_derivation": "byte identity of every column but eff_rank (cap_cifar5p1_1007 §1.4)",
             **base, "mutations": []}
    for label, pairs in (("M-h1: l2init(1e-3) seeds 0-9 compared with the l2init(1e-2) record",
                          [(("R", "l2init:1e-3", 0), "R_std_lr0.0001_l2init1e-2")]),
                         ("M-h2: SNA seeds 10-19 compared with the seeds 0-9 record",
                          [(("SNA", "none", 10), "SNA_std_lr0.0001")])):
        r = compare(pairs)
        entry["mutations"].append({"mutation": label, "check_pass_on_mutant": r["pass"], "detected": not r["pass"],
                                   "mutant_result": brief(r)})
    finish("S-host-repro", entry, t0)


# --------------------------------------------------------------------------
# S-snp
# --------------------------------------------------------------------------

def s_snp(M) -> dict:
    eps_s, sig = "1e-3", "1e-2"
    cfg = M.make_cfg("snp", eps=eps_s, sigma=sig)
    eps_f, sig_f = float(eps_s), float(sig)
    st = {"pre": None, "adam": None, "n_post": {"train": 0, "fresh": 0, "fresh_self": 0},
          "n_steps": {"train": 0, "fresh": 0, "fresh_self": 0}, "bit_ok": True, "mv_ok": True,
          "adam_eq_step": True, "range_ok": True, "sum": 0.0, "sum2": 0.0, "n": 0, "missing_method": 0}

    def hook(ev, **s):
        if ev == "post_adam":
            st["adam"] = clone_all(s["P"])
        elif ev == "post_step":
            if st["pre"] is not None and st["pending_phase"] in ("train", "fresh_self"):
                st["missing_method"] += 1                       # the previous step got no S&P
            st["pre"] = (clone_all(s["P"]), clone_all(s["m"]), clone_all(s["v"]), s["g_snp"].get_state().clone())
            st["pending_phase"] = s["phase"]
            st["n_steps"][s["phase"]] += 1
            st["adam_eq_step"] &= bits_equal(st["adam"], st["pre"][0])
        elif ev == "post_method":
            P0_, m0, v0, gs = st["pre"]
            st["pre"] = None
            st["n_post"][s["phase"]] += 1
            g = torch.Generator(device=DEV)
            g.set_state(gs)
            for i, (p0, p1) in enumerate(zip(P0_, s["P"])):
                z = (torch.rand(p0.shape, generator=g, device=DEV) * 2 - 1) * BETA[i]
                st["bit_ok"] &= torch.equal(p0.mul(1 - eps_f).add(z, alpha=sig_f), p1.detach())
                u = (z / BETA[i]).double()
                st["range_ok"] &= bool(u.abs().max() <= 1.0)
                st["sum"] += float(u.sum()); st["sum2"] += float((u * u).sum()); st["n"] += u.numel()
            st["mv_ok"] &= bits_equal(m0, s["m"]) and bits_equal(v0, s["v"])

    st["pending_phase"] = None
    out = SCR / f"snp_{tag(M)}"
    erun(M, cfg, CHECK3, 2, out, debug={"hook": hook}, steps_hard=50, steps_easy=50)
    if st["pre"] is not None and st["pending_phase"] in ("train", "fresh_self"):
        st["missing_method"] += 1
    n = max(st["n"], 1)
    mean, m2 = st["sum"] / n, st["sum2"] / n
    band_mean = 4 * math.sqrt(1 / 3) / math.sqrt(n)              # 4 sd of the mean of U(-1, 1)
    band_m2 = 4 * math.sqrt(1 / 5 - 1 / 9) / math.sqrt(n)        # 4 sd of the mean of u^2
    per = {"every_train_step_gets_snp": st["n_post"]["train"] == st["n_steps"]["train"] > 0,
           "every_fresh_self_step_gets_snp": st["n_post"]["fresh_self"] == st["n_steps"]["fresh_self"] > 0,
           "no_snp_in_method_free_fresh": st["n_post"]["fresh"] == 0 and st["n_steps"]["fresh"] > 0,
           "no_step_without_snp": st["missing_method"] == 0,
           "after_adam_nothing_else_in_step": st["adam_eq_step"],
           "bitwise_shrink_plus_init_noise": st["bit_ok"], "zeta_in_init_range": st["range_ok"],
           "zeta_mean_in_band": abs(mean) <= band_mean, "zeta_meansq_in_band": abs(m2 - 1 / 3) <= band_m2,
           "adam_moments_untouched": st["mv_ok"]}
    return result(per, {"zeta_n": n, "zeta_mean": mean, "zeta_meansq": m2, "band_mean": band_mean,
                        "band_meansq": band_m2, "steps": st["n_steps"], "writes": st["n_post"]})


SNP_MUT = [
    ("M-s1: biases are not shrunk or perturbed",
     [("        for p, bd in zip(P, bound):\n            z = (torch.rand(p.shape, generator=g_snp",
       "        for p, bd in zip(P[0::2], bound[0::2]):\n            z = (torch.rand(p.shape, generator=g_snp")]),
    ("M-s2: zeta Gaussian instead of the init's uniform",
     [("            z = (torch.rand(p.shape, generator=g_snp, device=device) * 2 - 1) * bd",
       "            z = torch.randn(p.shape, generator=g_snp, device=device) * bd")]),
    ("M-s3: the bound from fan_out", [("    fan_in = [C.layer_shapes(arm)[i // 2][1] for i in range(6)]",
                                       "    fan_in = [C.layer_shapes(arm)[i // 2][0] for i in range(6)]")]),
    ("M-s4: S&P before the Adam update",
     [("            if is_snp:\n                snp_apply()\n                wrote = True",
       "            if is_snp:\n                wrote = False"),
      ("            act.begin_step(R, C.BATCH, device)\n            if cg is not None:\n                cg.replay()",
       "            act.begin_step(R, C.BATCH, device)\n            if is_snp and method_on:\n                snp_apply()\n"
       "            if cg is not None:\n                cg.replay()")]),
    ("M-s5: sigma forgotten", [("            p.mul_(1 - snp_eps).add_(z, alpha=snp_sigma)",
                                "            p.mul_(1 - snp_eps).add_(z)")]),
]


# --------------------------------------------------------------------------
# S-cbp
# --------------------------------------------------------------------------

def host_factor(tc: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The host's scalars exactly as its train_task fills them (python double -> float32)."""
    a = torch.zeros((), device=DEV); a.fill_(1.0 / (1 - AB1 ** tc))
    b = torch.zeros((), device=DEV); b.fill_(1.0 / (1 - AB2 ** tc))
    return a, b


def factor_tensor(L: torch.Tensor, tc: int, scal: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Registered per-element bias correction (mirror of the engine): the host scalar where L == 0,
    else float32 of the device float64 1/(1 - beta^(tc - L)).  Also returns an independent float64
    ideal computed with numpy on the cpu, for the one-rounding check."""
    s = (tc - L).double()
    dev64 = 1.0 / (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64, device=DEV), s))
    s_np = (tc - L).cpu().numpy().astype(np.float64)
    with np.errstate(divide="ignore"):
        ideal = torch.from_numpy(1.0 / (1.0 - np.power(beta, s_np))).to(DEV)
    return torch.where(L == 0, scal, dev64.float()), ideal


def s_cbp(M) -> dict:
    rho = "1e-3"
    cfg = M.make_cfg("cbp", rho=rho)
    m_mat, eta = M.CBP_MATURITY, M.CBP_DECAY
    gens = {s: H.stream("b5_cbp_reinit", s) for s in CHECK3}
    ck = {"acc": [Fraction(0), Fraction(0)], "L": None, "prev_final": None, "A": None, "S": None,
          "expect_k": None, "phase": None}
    per = {k: True for k in ("age_plus_one", "util_in_band", "P_mv_untouched_by_ingraph", "n_elig_equal_slots",
                             "n_elig_matches_engine_ledger", "write_iff_k_positive", "replaced_count_is_k",
                             "replaced_were_eligible", "replaced_have_smallest_uhat", "composed_P_bitwise",
                             "composed_mv_bitwise", "u_age_reset", "L_set_to_t", "fresh_rows_from_own_stream",
                             "fresh_rows_in_range", "adam_mirror_bitwise", "reset_factors_one_rounding")}
    stats = {"replacements": [0, 0], "max_util_err_over_band": 0.0, "reset_elems_checked": 0, "min_s_seen": 10 ** 9,
             "steps": 0}
    fails = []

    def flag(name, ok, info=None):
        if not ok:
            per[name] = False
            if len(fails) < 20:
                fails.append((name, info))

    def hook(ev, **s):
        if ev in ("fresh_start", "fresh_self_start"):
            ck["prev_final"] = clone_all(s["P"])
            ck["L"] = [torch.zeros(len(CHECK3), C.HIDDEN, dtype=torch.long, device=DEV) for _ in range(2)]
            ck["acc"] = [Fraction(0), Fraction(0)]
            ck["phase"] = "fresh" if ev == "fresh_start" else "fresh_self"
            return
        if ev == "post_adam":
            tc = s["tc"]
            if ck["L"] is None:
                ck["L"] = [torch.zeros_like(q) for q in s["L"]]
            if ck["prev_final"] is None:
                ck["prev_final"] = init_stack(CHECK3)
            # the Adam update itself (mirror with the registered per-element step count)
            s1, s2 = host_factor(tc)
            L1, L2 = ck["L"]
            Ls = [L1[:, :, None], L1, torch.maximum(L2[:, :, None], L1[:, None, :]), L2, L2[:, None, :], None]
            for i, (pp, mm, vv) in enumerate(zip(ck["prev_final"], s["m"], s["v"])):
                if Ls[i] is None:
                    f1, f2 = s1, s2
                else:
                    f1, i1 = factor_tensor(Ls[i], tc, s1, AB1)
                    f2, i2 = factor_tensor(Ls[i], tc, s2, AB2)
                    msk = Ls[i] != 0
                    if bool(msk.any()):
                        bnd = U32 + 8 * U64                    # one float32 rounding + two float64 pows
                        e1 = ((f1.double() - i1).abs() <= bnd * i1.abs())[msk.expand_as(f1)]
                        e2 = ((f2.double() - i2).abs() <= bnd * i2.abs())[msk.expand_as(f2)]
                        flag("reset_factors_one_rounding", bool(e1.all()) and bool(e2.all()))
                        stats["reset_elems_checked"] += int(msk.sum())
                        stats["min_s_seen"] = min(stats["min_s_seen"], int((tc - Ls[i][msk]).min()))
                want = pp - LR * (mm * f1) / ((vv * f2).sqrt() + AEPS)
                flag("adam_mirror_bitwise", torch.equal(want, s["P"][i].detach()), (tc, i))
            ck["A"] = (clone_all(s["P"]), clone_all(s["m"]), clone_all(s["v"]), clone_all(s["age"]),
                       clone_all(s["u"]), clone_all(s["L"]), s["a1"].clone(), s["a2"].clone())
            return
        if ev == "post_step":
            stats["steps"] += 1
            if ck.get("pending_check") is not None:
                flag("write_iff_k_positive", False, ("k > 0 but no write", ck["pending_check"]))
                ck["pending_check"] = None
            P_A, m_A, v_A, age_A, u_A, L_A, a1, a2 = ck["A"]
            flag("P_mv_untouched_by_ingraph", bits_equal(P_A, s["P"]) and bits_equal(m_A, s["m"])
                 and bits_equal(v_A, s["v"]))
            for l, (h, Wn) in enumerate(((a1, s["P"][2]), (a2, s["P"][4]))):
                flag("age_plus_one", torch.equal(s["age"][l], age_A[l] + 1))
                c64 = Wn.detach().double().abs().mean(dim=1) * h.double().abs().mean(dim=1)
                want = eta * u_A[l].double() + (1 - eta) * c64
                band = (gamma(99, U32) + gamma(31, U32) + 8 * U32) * want + 2 * TINY32
                err = (s["u"][l].double() - want).abs()
                stats["max_util_err_over_band"] = max(stats["max_util_err_over_band"], float((err / band).max()))
                flag("util_in_band", bool((err <= band).all()))
            ck["S"] = (clone_all(s["P"]), clone_all(s["m"]), clone_all(s["v"]), clone_all(s["age"]),
                       clone_all(s["u"]), clone_all(s["L"]))
            ck["prev_final"] = clone_all(s["P"])
            if s["phase"] == "fresh":
                ck["expect_k"] = [0, 0]
                return
            ks = []
            for l in range(2):
                ne = (s["age"][l] > m_mat).sum(1)
                flag("n_elig_equal_slots", bool((ne == ne[0]).all()))
                ne0 = int(ne[0])
                flag("n_elig_matches_engine_ledger", s["ledger"].n_elig(l, s["tc"]) == ne0, (s["tc"], l))
                ck["acc"][l] += Fraction(rho) * ne0
                k = math.floor(ck["acc"][l])
                ck["acc"][l] -= k
                ks.append(k)
            ck["expect_k"] = ks
            ck["wrote"] = False
            ck["pending_check"] = (s["tc"], s["phase"])
            if sum(ks) == 0:
                ck["pending_check"] = None
            return
        if ev == "post_method":
            ck["wrote"] = True
            tc = s["tc"]
            if ck.get("pending_check") is None:
                flag("write_iff_k_positive", False, ("write without k", tc))
                return
            ck["pending_check"] = None
            P_S, m_S, v_S, age_S, u_S, L_S = ck["S"]
            P_e, m_e, v_e = clone_all(P_S), clone_all(m_S), clone_all(v_S)
            u_e, age_e, L_e = clone_all(u_S), clone_all(age_S), clone_all(L_S)
            chosen_all = []
            for l in range(2):
                k = ck["expect_k"][l]
                chosen = []
                for r in range(len(CHECK3)):
                    ch = torch.nonzero((s["age"][l][r] == 0) & (age_S[l][r] > 0)).flatten()
                    flag("replaced_count_is_k", len(ch) == k, (tc, l, r, len(ch), k))
                    if len(ch):
                        el = age_S[l][r] > m_mat
                        flag("replaced_were_eligible", bool(el[ch].all()), (tc, l, r))
                        a64 = age_S[l][r].double()
                        uh = u_S[l][r].double() / (1 - torch.pow(torch.tensor(eta, dtype=torch.float64, device=DEV), a64))
                        rest = el.clone(); rest[ch] = False
                        if bool(rest.any()):
                            amax = float(a64[el].max())
                            delta = 2 * (amax + 6) * U32
                            flag("replaced_have_smallest_uhat",
                                 float(uh[ch].max()) <= float(uh[rest].min()) * (1 + delta) + TINY32, (tc, l, r))
                    chosen.append(ch)
                    stats["replacements"][l] += len(ch)
                chosen_all.append(chosen)
            for l in range(2):                                   # compose: layer 1, then layer 2
                for r, ch in enumerate(chosen_all[l]):
                    if not len(ch):
                        continue
                    rows = ((torch.rand((len(ch), FAN_IN[2 * l]), generator=gens[CHECK3[r]], dtype=torch.float32)
                             * 2 - 1) * BETA[2 * l]).to(DEV)
                    flag("fresh_rows_in_range", bool((rows.abs() <= BETA[2 * l]).all()))
                    P_e[2 * l][r, ch, :] = rows
                    P_e[2 * l + 1][r, ch] = 0.0
                    P_e[2 * l + 2][r, :, ch] = 0.0
                    for q in (m_e, v_e):
                        q[2 * l][r, ch, :] = 0.0
                        q[2 * l + 1][r, ch] = 0.0
                        q[2 * l + 2][r, :, ch] = 0.0
                    u_e[l][r, ch] = 0.0
                    age_e[l][r, ch] = 0.0
                    L_e[l][r, ch] = tc
            ok_rows = all(torch.equal(P_e[2 * l][r, ch, :], s["P"][2 * l][r, ch, :].detach())
                          for l in range(2) for r, ch in enumerate(chosen_all[l]) if len(ch))
            flag("fresh_rows_from_own_stream", ok_rows, tc)
            flag("composed_P_bitwise", bits_equal(P_e, [q.detach() for q in s["P"]]), tc)
            flag("composed_mv_bitwise", bits_equal(m_e, s["m"]) and bits_equal(v_e, s["v"]), tc)
            flag("u_age_reset", bits_equal(u_e, s["u"]) and bits_equal(age_e, s["age"]), tc)
            flag("L_set_to_t", bits_equal(L_e, s["L"]), tc)
            ck["L"] = clone_all(L_e)
            ck["prev_final"] = clone_all(s["P"])

    out = SCR / f"cbp_{tag(M)}"
    erun(M, cfg, CHECK3, 3, out, debug={"hook": hook}, steps_hard=300, steps_easy=150)

    # a step with k > 0 that never wrote is caught at the next post_step through pending_check
    # (re-checked here for the very last step)
    flag("write_iff_k_positive", ck.get("pending_check") is None, "last step")
    # neutrality before the first replacement: the cbp arm and the none arm share every bit
    hashes = {"none": [], "cbp": []}

    def hh(name):
        def f(ev, **s):
            if ev == "post_step" and s["phase"] == "train":
                h = hashlib.sha256()
                for q in s["P"]:
                    h.update(q.detach().cpu().numpy().tobytes())
                hashes[name].append((s["tc"], h.hexdigest(), int(s["n_reset"].sum())))
        return f
    erun(M, M.make_cfg("none"), CHECK3, 1, SCR / f"cbp_none_{tag(M)}", debug={"hook": hh("none")},
         steps_hard=150, steps_easy=150, fresh=False)
    erun(M, cfg, CHECK3, 1, SCR / f"cbp_cbp_{tag(M)}", debug={"hook": hh("cbp")},
         steps_hard=150, steps_easy=150, fresh=False)
    first = next((i for i, (_, _, nr) in enumerate(hashes["cbp"]) if nr > 0), None)
    per["first_replacement_happens"] = first is not None
    upto = first if first is not None else 0
    per["identical_to_none_before_first_replacement"] = upto > 100 and all(
        a[1] == b[1] for a, b in zip(hashes["none"][:upto], hashes["cbp"][:upto]))
    per["non_vacuous_replacements"] = min(stats["replacements"]) >= 10
    per["non_vacuous_small_s"] = stats["min_s_seen"] <= 3
    return result(per, {"stats": stats, "fails": fails[:20], "first_replacement_step_index": first})


CBP_MUT = [
    ("M-c1: utility from the incoming weights",
     [("                    c = Wn.abs().mean(dim=1) * h.detach().abs().mean(dim=1)",
       "                    c = P[2 * l].abs().mean(dim=2) * h.detach().abs().mean(dim=1)")]),
    ("M-c2: maturity ignored in the selection", [("            elig = (age > CBP_MATURITY).cpu().numpy()",
                                                  "            elig = (age >= 0).cpu().numpy()")]),
    ("M-c3: output weights not zeroed", [("                Wn[r, :, idx] = 0.0\n                cbp_u[l][r, idx] = 0.0",
                                          "                cbp_u[l][r, idx] = 0.0")]),
    ("M-c4: output-side moments not zeroed", [("                    q[2 * l + 2][r, :, idx] = 0.0\n", "")]),
    ("M-c5: per-element step count not reset (global bias correction)",
     [("                cbp_L[l][r, idx] = t\n", "                pass\n")]),
    ("M-c6: the fractional count is not carried", [("        self.acc[l] -= k\n", "        self.acc[l] = Fraction(0)\n")]),
    ("M-c7: the largest utility is replaced",
     [('                order = cand[np.argsort(uhat[r, cand], kind="stable")]',
       '                order = cand[np.argsort(-uhat[r, cand], kind="stable")]')]),
]


# --------------------------------------------------------------------------
# S-redo
# --------------------------------------------------------------------------

def s_redo(M) -> dict:
    per = {k: True for k in ("checks_on_schedule", "boundary_waits_for_evaluation", "evaluated_net_is_pre_recycle",
                             "mask_matches_mirror", "score_formula_float64", "composed_P_bitwise",
                             "composed_mv_bitwise", "bias_moments_untouched", "fresh_rows_in_range",
                             "adam_mirror_bitwise", "dead_post_written", "tau0_logits_invariant")}
    stats = {"recycled": [0, 0], "mid_checks": 0, "boundary_checks": 0, "tau0_recycled": 0,
             "near_threshold_skipped": 0}
    fails = []

    def flag(name, ok, info=None):
        if not ok:
            per[name] = False
            if len(fails) < 20:
                fails.append((name, info))

    def make_hook(tau: float, F_: int, gens: dict, tau0: bool):
        ck = {"prev_final": None, "S": None, "idx": [], "expect": False, "tc": 0, "steps_in_task": None,
              "t": None}

        def expected(Ppre, mpre, vpre, idx, tc):
            _, a1, _, a2, z3 = B.forward(Ppre, cifar_X()[idx], C.make_act("R"), train=False)
            masks = []
            for l, h in enumerate((a1, a2)):
                s32 = h.abs().mean(dim=1)
                s32 = s32 / (s32.mean(dim=1, keepdim=True) + 1e-9)
                h64 = h.double().abs()
                s64 = h64.mean(dim=1) / (h64.mean(dim=1).mean(dim=1, keepdim=True) + 1e-9)
                band = (gamma(64, U32) + gamma(100, U32) + 4 * U32) * (s64 + 1) + 1e-9 * U32
                near = (s64 - tau).abs() <= band
                stats["near_threshold_skipped"] += int(near.sum())
                ok = ((s32 <= tau) == (s64 <= tau)) | near
                flag("score_formula_float64", bool(ok.all()), (tc, l))
                masks.append(s32 <= tau)
            P_e, m_e, v_e = clone_all(Ppre), clone_all(mpre), clone_all(vpre)
            units = [[torch.nonzero(masks[l][r]).flatten() for r in range(len(CHECK3))] for l in range(2)]
            for l in range(2):
                for r in range(len(CHECK3)):
                    ch = units[l][r]
                    if len(ch):
                        rows = ((torch.rand((len(ch), FAN_IN[2 * l]), generator=gens[CHECK3[r]],
                                            dtype=torch.float32) * 2 - 1) * BETA[2 * l]).to(DEV)
                        flag("fresh_rows_in_range", bool((rows.abs() <= BETA[2 * l]).all()))
                        P_e[2 * l][r, ch, :] = rows
                        P_e[2 * l + 1][r, ch] = 0.0
                        m_e[2 * l][r, ch, :] = 0.0
                        v_e[2 * l][r, ch, :] = 0.0
                        stats["recycled"][l] += len(ch)
                        if tau0:
                            stats["tau0_recycled"] += len(ch)
            for l in range(2):
                for r in range(len(CHECK3)):
                    ch = units[l][r]
                    if len(ch):
                        P_e[2 * l + 2][r, :, ch] = 0.0
                        m_e[2 * l + 2][r, :, ch] = 0.0
                        v_e[2 * l + 2][r, :, ch] = 0.0
            return P_e, m_e, v_e, z3

        def verify(s, kind):
            Ppre, mpre, vpre = ck["S"]
            idx = torch.cat([ck["idx"][-2], ck["idx"][-1]], dim=1)
            P_e, m_e, v_e, z3 = expected(Ppre, mpre, vpre, idx, ck["tc"])
            flag("composed_P_bitwise", bits_equal(P_e, [q.detach() for q in s["P"]]), (kind, ck["tc"]))
            flag("composed_mv_bitwise", bits_equal(m_e, s["m"]) and bits_equal(v_e, s["v"]), (kind, ck["tc"]))
            flag("bias_moments_untouched", all(torch.equal(mpre[i], s["m"][i]) and torch.equal(vpre[i], s["v"][i])
                                               for i in (1, 3, 5)), (kind, ck["tc"]))
            if tau0:
                z3p = B.forward([q.detach() for q in s["P"]], cifar_X()[idx], C.make_act("R"), train=False)[4]
                flag("tau0_logits_invariant", torch.equal(z3, z3p), ck["tc"])
            ck["prev_final"] = clone_all(s["P"])
            ck["expect"] = False

        def hook(ev, **s):
            if ev in ("fresh_start", "fresh_self_start"):
                ck["prev_final"] = clone_all(s["P"])
                ck["idx"] = []
                ck["phase"] = ev
                return
            if ev == "post_adam":
                tc = s["tc"]
                if ck["prev_final"] is None:
                    ck["prev_final"] = init_stack(CHECK3)
                s1, s2 = host_factor(tc)
                for i, (pp, mm, vv) in enumerate(zip(ck["prev_final"], s["m"], s["v"])):
                    want = pp - LR * (mm * s1) / ((vv * s2).sqrt() + AEPS)
                    flag("adam_mirror_bitwise", torch.equal(want, s["P"][i].detach()), (tc, i))
                return
            if ev == "post_step":
                if ck["expect"]:
                    flag("checks_on_schedule", False, ("missed", ck["tc"]))
                ck["S"] = (clone_all(s["P"]), clone_all(s["m"]), clone_all(s["v"]))
                ck["prev_final"] = clone_all(s["P"])
                ck["idx"].append(s["idx"].clone())
                ck["tc"], ck["t"], ck["j"], ck["phase_now"] = s["tc"], s["t"], s["j"], s["phase"]
                ck["expect"] = s["phase"] in ("train", "fresh_self") and s["tc"] % F_ == 0
                return
            if ev == "post_method":
                if not ck["expect"]:
                    flag("checks_on_schedule", False, ("unexpected", s["tc"]))
                stats["mid_checks"] += 1
                verify(s, "mid")
                return
            if ev == "task_end":
                last = ck["S"][0]
                flag("evaluated_net_is_pre_recycle", bits_equal(last, [q.detach() for q in s["P"]]), s["t"])
                if ck["expect"]:                               # a check falls on the last update
                    flag("boundary_waits_for_evaluation", bool(s["pending"]), s["t"])
                    ck["boundary"] = True
                    ck["expect"] = False
                else:
                    flag("boundary_waits_for_evaluation", not s["pending"], s["t"])
                    ck["boundary"] = False
                return
            if ev == "boundary_post":
                stats["boundary_checks"] += 1
                verify(s, "boundary")
                return
        return hook, ck

    for tau_s, tau0, tasks in (("0.1", False, 3), ("0", True, 3)):
        cfg = M.make_cfg("redo", tau=tau_s, period=50)
        gens = {s: H.stream("b5_redo_reinit", s) for s in CHECK3}
        hook, ck = make_hook(float(tau_s), 50, gens, tau0)
        out = SCR / f"redo_{tau_s}_{tag(M)}"
        erun(M, cfg, CHECK3, tasks, out, debug={"hook": hook}, steps_hard=200, steps_easy=150)
        if ck["expect"]:
            if ck.get("phase_now") == "fresh_self" and ck["j"] == 199:
                pass                                           # the fresh run's last update: no check
            else:
                flag("checks_on_schedule", False, ("missed at end", ck["tc"]))
        ex = pd.read_csv(out / "extra_task.csv")
        flag("dead_post_written", bool(ex["dead_post_l2"].notna().sum() == 3 * len(CHECK3)), tau_s)
    per["non_vacuous_recycling_both_layers"] = min(stats["recycled"]) > 0
    per["non_vacuous_mid_and_boundary"] = stats["mid_checks"] > 0 and stats["boundary_checks"] >= 2
    per["non_vacuous_tau0"] = stats["tau0_recycled"] > 0
    return result(per, {"stats": stats, "fails": fails[:20]})


_X = {}


def cifar_X():
    if "x" not in _X:
        _X["x"] = cifar().inputs("train", "std", DEV)
    return _X["x"]


REDO_MUT = [
    ("M-r1: scores not normalised by the layer mean",
     [("            s = s / (s.mean(dim=1, keepdim=True) + REDO_SCORE_EPS)\n", "")]),
    ("M-r2: the probe is the current batch only",
     [("                    redo_apply(batches[:, j - 1:j + 1].reshape(R, -1), tc, t, j, phase)",
       "                    redo_apply(batches[:, j:j + 1].reshape(R, -1), tc, t, j, phase)")]),
    ("M-r3: output weights not zeroed", [("                    Wn[r, :, idx] = 0.0\n", "")]),
    ("M-r4: incoming moments not zeroed",
     [("                    adam_m[2 * l][r, idx, :] = 0.0\n                    adam_v[2 * l][r, idx, :] = 0.0\n", "")]),
    ("M-r5: the bias moments are zeroed too",
     [("                    b[r, idx] = 0.0\n                    adam_m[2 * l][r, idx, :] = 0.0",
       "                    b[r, idx] = 0.0\n                    adam_m[2 * l + 1][r, idx] = 0.0\n"
       "                    adam_m[2 * l][r, idx, :] = 0.0")]),
    ("M-r6: the period shifted to F + 1", [("            elif is_redo and tc % redo_F == 0:",
                                            "            elif is_redo and tc % (redo_F + 1) == 0:")]),
    ("M-r7: a task-end recycle runs before the evaluation",
     [('                if j == steps - 1:\n                    pending = phase == "train"',
       '                if False:\n                    pending = phase == "train"')]),
]


# --------------------------------------------------------------------------
# S-graph
# --------------------------------------------------------------------------

def s_graph(M) -> dict:
    per, extra = {}, {}
    cfgs = [M.make_cfg("snp", eps="1e-3", sigma="1e-2"), M.make_cfg("cbp", rho="1e-3"),
            M.make_cfg("redo", tau="0.1", period=100)]
    for cfg in cfgs:
        tg = M.cfg_tag(cfg)
        fa, fb = {}, {}
        oa, ob = SCR / f"graph_g_{tg}_{tag(M)}", SCR / f"graph_e_{tg}_{tag(M)}"
        erun(M, cfg, CHECK10, 3, oa, graph=True, steps_hard=200, steps_easy=100, final=fa)
        erun(M, cfg, CHECK10, 3, ob, graph=False, steps_hard=200, steps_easy=100, final=fb)
        for f in ("per_task.csv", "extra_task.csv", "fresh_control.csv", "fresh_self.csv", "redo_checks.csv"):
            if (oa / f).exists() or (ob / f).exists():
                per[f"{tg}_{f}_identical"] = same_bytes(oa / f, ob / f)
        per[f"{tg}_final_state_bitwise"] = all(bits_equal(fa[k], fb[k]) for k in ("P", "m", "v"))
        pa = json.loads((oa / "provenance.json").read_text())
        extra[f"{tg}_engine"] = pa["engine"]
        per[f"{tg}_graph_was_used"] = "CUDA graph" in pa["engine"]
        extra[f"{tg}_resets"] = pa["resets_total"]
    return result(per, extra)


GRAPH_MUT = [
    ("M-g1: CBP's L rebound instead of written in place",
     [("                cbp_L[l][r, idx] = t\n", "                cbp_L[l] = cbp_L[l].clone()\n                cbp_L[l][r, idx] = t\n")]),
    ("M-g2: the device step count is not written while replaying",
     [("            if is_cbp:\n                tc_dev.fill_(tc)", "            if is_cbp and cg is None:\n                tc_dev.fill_(tc)")]),
    ("M-g3: static_idx one batch behind while replaying",
     [("            static_idx.copy_(batches[:, j])",
       "            static_idx.copy_(batches[:, j] if cg is None else batches[:, max(j - 1, 0)])")]),
]


# --------------------------------------------------------------------------
# S-fresh
# --------------------------------------------------------------------------

def s_fresh(M) -> dict:
    per, extra = {}, {}
    host = hrun(CHECK3, 3, steps_hard=200, steps_easy=100)
    outs = {}
    for cfg in (M.make_cfg("none"), M.make_cfg("snp", eps="1e-3", sigma="1e-2"), M.make_cfg("cbp", rho="1e-3"),
                M.make_cfg("redo", tau="0.1", period=100)):
        o = SCR / f"fresh_{M.cfg_tag(cfg)}_{tag(M)}"
        erun(M, cfg, CHECK3, 3, o, steps_hard=200, steps_easy=100)
        outs[cfg["method"]] = o
    per["none_fresh_identical_to_host"] = same_bytes(outs["none"] / "fresh_control.csv", host / "fresh_control.csv")
    per["none_per_task_identical_to_host"] = same_bytes(outs["none"] / "per_task.csv", host / "per_task.csv")
    def fresh_col(p):
        rows = [l.split(",") for l in Path(p).read_text().splitlines()]
        i, j = rows[0].index("seed"), rows[0].index("fresh_online_acc")
        return [(r[i], r[j]) for r in rows[1:]]
    for m in ("snp", "cbp", "redo"):
        # the fresh network is shared; continual_online_acc / fresh_gap are the arm's own by construction
        per[f"{m}_fresh_network_identical_to_none"] = fresh_col(outs[m] / "fresh_control.csv") == \
            fresh_col(outs["none"] / "fresh_control.csv")
        cont = {r["seed"]: r["online_acc"] for r in read_csv(outs[m] / "per_task.csv") if r["task"] == "3"}
        per[f"{m}_continual_column_is_own_t_last"] = all(
            r["continual_online_acc"] == cont[r["seed"]] for r in read_csv(outs[m] / "fresh_control.csv"))
        per[f"{m}_fresh_self_written"] = (outs[m] / "fresh_self.csv").exists()
    P0 = init_stack(CHECK3)
    seen = {}

    def hook(ev, **s):
        if ev in ("fresh_start", "fresh_self_start"):
            ok = bits_equal(P0, [q.detach() for q in s["P"]])
            ok &= all(bool((q == 0).all()) for q in (*s["m"], *s["v"]))
            ok &= all(bool((q == 0).all()) for q in (*s["age"], *s["u"], *s["L"]))
            if s["ledger"] is not None:
                ok &= all(a == 0 for a in s["ledger"].acc) and all(not h for h in s["ledger"].hist)
            seen[ev] = seen.get(ev, True) and ok
    for cfg in (M.make_cfg("cbp", rho="1e-3"), M.make_cfg("snp", eps="1e-3", sigma="1e-2")):
        seen.clear()
        erun(M, cfg, CHECK3, 3, SCR / f"freshhook_{M.cfg_tag(cfg)}_{tag(M)}", debug={"hook": hook},
             steps_hard=200, steps_easy=100)
        per[f"{cfg['method']}_fresh_start_state_initial"] = seen.get("fresh_start", False)
        per[f"{cfg['method']}_fresh_self_start_state_initial"] = seen.get("fresh_self_start", False)
    return result(per, extra)


FRESH_MUT = [
    ("M-f1: the method stays on in the method-free fresh",
     [('        train_task(saved[last_hard], 0, False, "fresh", last_hard)',
       '        train_task(saved[last_hard], 0, True, "fresh", last_hard)')]),
    ("M-f2: CBP's age not reset before a fresh run",
     [("                    cbp_age[l].zero_()\n                    cbp_u[l].zero_()", "                    cbp_u[l].zero_()")]),
    ("M-f3: Adam moments not zeroed before a fresh run",
     [("                for q in (*adam_m, *adam_v):\n                    q.zero_()\n                for l in range(2):",
       "                for q in ():\n                    q.zero_()\n                for l in range(2):")]),
]


# --------------------------------------------------------------------------
# S-select / S-verdict (target: verdict.py)
# --------------------------------------------------------------------------

def _table(V, overrides: dict | None = None) -> list[dict]:
    """A synthetic calibration table: every registered cell, seeds 100-109."""
    rows = []
    rng = np.random.default_rng(0)
    for method in ("snp", "cbp", "redo"):
        for i, h in enumerate(V.E.GRID[method]):
            late = (0.5 + 0.001 * i + 0.0001 * rng.standard_normal(10)).tolist()
            rows.append({"method": method, "hyper": h, "seeds": list(range(100, 110)), "late": late,
                         "early": late, "diverged": False})
    for (method, i), upd in (overrides or {}).items():
        r = [q for q in rows if q["method"] == method][i]
        r.update(upd)
    return rows


def s_select(V) -> dict:
    per = {}
    E = V.E
    # (a) mean, not median, and not the early window: cell 1 has the larger mean, cell 2 the larger median
    t = _table(V, {("cbp", 0): {"late": [0.5] * 10, "early": [0.9] * 10},
                   ("cbp", 1): {"late": [0.40] * 4 + [0.62] * 6, "early": [0.1] * 10},     # mean .532, median .62
                   ("cbp", 2): {"late": [0.70] * 4 + [0.45] * 6, "early": [0.1] * 10}})    # mean .550, median .45
    sel = V.select_from_table(t)
    per["argmax_of_mean"] = sel["cbp"]["hyper"] == E.GRID["cbp"][2]
    # (b) a diverged best cell is skipped
    t = _table(V, {("snp", 5): {"late": [0.9] * 10, "diverged": True}, ("snp", 4): {"late": [0.8] * 10}})
    per["diverged_dropped"] = V.select_from_table(t)["snp"]["hyper"] == E.GRID["snp"][4]
    # (c) exact tie -> grid order
    t = _table(V, {("redo", 3): {"late": [0.9] * 10}, ("redo", 7): {"late": [0.9] * 10}})
    per["tie_grid_order"] = V.select_from_table(t)["redo"]["hyper"] == E.GRID["redo"][3]
    # (d) refusals
    for name, ov in (("refuse_unregistered", {("cbp", 0): {"hyper": {"rho": "1e-2"}}}),
                     ("refuse_wrong_seeds", {("cbp", 0): {"seeds": list(range(10))}})):
        try:
            V.select_from_table(_table(V, ov))
            per[name] = False
        except ValueError:
            per[name] = True
    t = _table(V)
    t = [r for r in t if not (r["method"] == "redo" and r["hyper"] == E.GRID["redo"][8])]
    try:
        V.select_from_table(t)
        per["refuse_missing_cell"] = False
    except ValueError:
        per["refuse_missing_cell"] = True
    return result(per)


SELECT_MUT = [
    ("M-sel1: select by the median", [('            score = float(np.mean(np.asarray(r["late"], float)))',
                                       '            score = float(np.median(np.asarray(r["late"], float)))')]),
    ("M-sel2: select by the early window", [('            score = float(np.mean(np.asarray(r["late"], float)))',
                                             '            score = float(np.mean(np.asarray(r["early"], float)))')]),
    ("M-sel3: diverged cells kept", [('            if r["diverged"]:\n                continue',
                                      '            if False:\n                continue')]),
]


def s_verdict(V) -> dict:
    per = {}
    a_win = np.linspace(0.01, 0.03, 20)
    per["A_WINS"] = V.label4(a_win)["label"] == "A_WINS"
    per["B_WINS"] = V.label4(-a_win)["label"] == "B_WINS"
    eq = np.array([0.001, -0.001] * 10)
    per["EQUIVALENT"] = V.label4(eq)["label"] == "EQUIVALENT_WITHIN_0.005"
    un = np.array([0.02, -0.02] * 10)
    per["UNRESOLVED"] = V.label4(un)["label"] == "UNRESOLVED"
    edge = np.array([-0.005] * 6 + [0.0001, -0.0001] * 4 + [0.005] * 6)       # v(6) = -.005, v(15) = .005
    per["band_closed"] = V.label4(edge)["label"] == "EQUIVALENT_WITHIN_0.005"
    just_out = np.array([-0.0051] * 6 + [0.0001, -0.0001] * 4 + [0.005] * 6)
    per["band_edge_excludes_outside"] = V.label4(just_out)["label"] == "UNRESOLVED"
    v20 = np.arange(1, 21, dtype=float)
    per["interval_n20"] = V.order_interval(v20[::-1]) == (6.0, 15.0)
    per["interval_n10"] = V.order_interval(np.arange(1, 11, dtype=float)) == (2.0, 9.0)
    per["sign_15_of_20"] = abs(V.sign_test([1] * 15 + [-1] * 5)[2] - 21700 * 2 / 2 ** 20) < 1e-15
    per["sign_14_of_20"] = abs(V.sign_test([1] * 14 + [-1] * 6)[2] - 60460 * 2 / 2 ** 20) < 1e-15
    per["sign_zeros_dropped"] = V.sign_test([0] * 5 + [1] * 15)[:2] == (15, 15)
    per["holm"] = np.allclose(V.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    per["gap_labels"] = (V.label_gap(-a_win)["label"] == "GAP_REDUCED" and V.label_gap(a_win)["label"] ==
                         "GAP_INCREASED" and V.label_gap(eq)["label"] == "GAP_NOT_SHOWN")
    # paired, not unpaired: medians point one way, the seed-by-seed differences the other
    a = pd.Series([0.50, 0.50, 0.50, 0.90, 0.90] * 4, index=range(20))
    b2 = pd.Series([0.40, 0.40, 0.40, 0.95, 0.95] * 4, index=range(20))
    per["paired_differences"] = bool(np.allclose(V.paired(a, b2), (a - b2).values))
    # windows: hard 21-29 and 1-9
    rows = [{"seed": 0, "task": t, "online_acc": float(t)} for t in range(1, 31)]
    w = V.windows(pd.DataFrame(rows))
    per["late_window_21_29"] = float(w.loc[0, "late"]) == 25.0
    per["early_window_1_9"] = float(w.loc[0, "early"]) == 5.0
    per["all_zero_is_equivalent"] = V.label4(np.zeros(20))["label"] == "EQUIVALENT_WITHIN_0.005" and \
        V.sign_test(np.zeros(20))[2] == 1.0
    per["p2a_closed_at_002"] = V.prop_p2a(0.02) and not V.prop_p2a(0.0201)
    per["p4_open_at_03"] = V.prop_p4(0.2999) and not V.prop_p4(0.3) and not V.prop_p4(-0.31)
    return result(per)


VERDICT_MUT = [
    ("M-v1: interval [v(5), v(16)]", [("        return float(v[5]), float(v[14])", "        return float(v[4]), float(v[15])")]),
    ("M-v2: band +-0.01", [("BAND = 0.005\n", "BAND = 0.01\n")]),
    ("M-v3: unpaired difference of medians", [("    return np.array([a.loc[s] - b.loc[s] for s in seeds], float)",
                                               "    return np.array([a.median() - b.median() for s in seeds], float)")]),
    ("M-v4: window hard 19-27", [("EARLY, LATE = HARD[:5], HARD[10:]", "EARLY, LATE = HARD[:5], HARD[9:14]")]),
    ("M-v5: one-sided p", [('    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else 1.0',
                            '    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0) if n else 1.0')]),
]


# --------------------------------------------------------------------------
# S-CLI
# --------------------------------------------------------------------------

def s_cli(M) -> dict:
    per, extra = {}, {}
    args = ["run", "--method", "cbp", "--rho", "1e-3", "--seeds", "100-102", "--tasks", "2",
            "--steps-hard", "120", "--steps-easy", "60", "--device", DEV.type]
    d_cli, d_fn = SCR / f"cli_{tag(M)}", SCR / f"clifn_{tag(M)}"
    shutil.rmtree(d_cli, ignore_errors=True)
    M.main(args + ["--out", str(d_cli)])
    erun(M, M.make_cfg("cbp", rho="1e-3"), CHECK3, 2, d_fn, steps_hard=120, steps_easy=60)
    files = ("per_task.csv", "extra_task.csv", "fresh_control.csv", "fresh_self.csv")
    per["cli_equals_function"] = all(same_bytes(d_cli / f, d_fn / f) for f in files)
    if getattr(M, "_is_real", False):          # the module entry point (no second GPU process: --help)
        r = subprocess.run([sys.executable, "-m", "src.baselines_cifar5p1_1008", "run", "--help"],
                           cwd=REPO, capture_output=True, text=True)
        per["module_entry_point"] = r.returncode == 0 and "--rho" in r.stdout and "--period" in r.stdout
    a = M.build_parser().parse_args(["run", "--method", "none"])
    per["defaults"] = (a.tasks == 30 and a.lr == 1e-4 and a.threads == 2 and a.steps_hard == 780
                       and a.steps_easy == 780 and a.seeds == "100-109")
    c = M.make_cfg("cbp", rho="1e-4")
    per["cbp_defaults"] = c["maturity"] == 100 and c["decay"] == 0.99
    # admission: a synthetic checks.json that passes, then the specific refusals
    fake = SCR / "fake_checks.json"
    fake.write_text(json.dumps({"all_pass": True, **{k: {"pass": True} for k in M.REQUIRED_CHECKS},
                                "source_sha256": {s: sha(REPO / s) for s in M.SOURCES}}))
    run_cfg = dict(seeds=list(range(100, 110)), tasks=30, lr=1e-4, steps_hard=780, steps_easy=780,
                   fresh=True, graph=True, threads=2)

    def refused(cfg, rc, out, cj, needle) -> bool:
        try:
            M.admit(cfg, rc, out, checks_json=cj)
            return False
        except SystemExit as e:
            return needle in str(e)
    calib = M.CALIB_ROOT / "x"
    main_ = M.MAIN_ROOT / "x"
    per["refuse_unregistered_cell"] = refused(M.make_cfg("cbp", rho="1e-2"), run_cfg, calib, fake, "registered grid")
    per["refuse_missing_checks"] = refused(M.make_cfg("cbp", rho="1e-4"), run_cfg, calib, SCR / "nope.json", "missing")
    per["refuse_calib_wrong_seeds"] = refused(M.make_cfg("cbp", rho="1e-4"), {**run_cfg, "seeds": list(range(10))},
                                              calib, fake, "seeds 100-109")
    per["refuse_main_without_selection"] = (not (M.CALIB_ROOT / "selected.json").exists()) and refused(
        M.make_cfg("cbp", rho="1e-4"), {**run_cfg, "seeds": list(range(10))}, main_, fake, "selected.json")
    per["refuse_unregistered_steps"] = refused(M.make_cfg("cbp", rho="1e-4"), {**run_cfg, "steps_hard": 100},
                                               calib, fake, "registered configuration")
    return result(per, extra)


CLI_MUT = [
    ("M-L1: the CLI ignores --steps-hard",
     [("    run_cfg = dict(seeds=seeds, tasks=a.tasks, lr=a.lr, steps_hard=a.steps_hard,",
       "    run_cfg = dict(seeds=seeds, tasks=a.tasks, lr=a.lr, steps_hard=C.STEPS_PER_TASK,")]),
    ("M-L2: the default task count is not 30", [('    ap.add_argument("--tasks", type=int, default=C.N_TASKS)',
                                                 '    ap.add_argument("--tasks", type=int, default=C.N_TASKS - 1)')]),
]


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated check keys")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    SCR.mkdir(parents=True, exist_ok=True)
    global RESULTS
    if DUMP.exists() and only:
        RESULTS.update(json.loads(DUMP.read_text()))
    RESULTS["source_sha256"] = {s: sha(REPO / s) for s in SOURCES}
    RESULTS["torch"] = torch.__version__
    RESULTS["threads"] = torch.get_num_threads()
    RESULTS["git_head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip()
    RESULTS["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    plan = [
        ("S-nochange", lambda: run_check("S-nochange", "none arm = host R, byte for byte", s_nochange,
                                         NOCHANGE_MUT, "byte identity; eff_rank band from spec 7.1 of cap_cifar5p1_1007",
                                         ENGINE)),
        ("S-host-repro", s_host_repro),
        ("S-snp", lambda: run_check("S-snp", "S&P = shrink then init-distribution noise, after Adam", s_snp, SNP_MUT,
                                    "bitwise replay of the same generator state; zeta moments within 4 sampling sd",
                                    ENGINE)),
        ("S-cbp", lambda: run_check("S-cbp", "CBP = GnT accumulate + contribution utility + AdamGnT", s_cbp, CBP_MUT,
                                    "utility band: gamma_99 + gamma_31 + 8u32 relative (float32 means and products); "
                                    "ranking tolerance 2(age_max + 6)u32 (float32 pow / division); bitwise elsewhere",
                                    ENGINE)),
        ("S-redo", lambda: run_check("S-redo", "ReDo = Sokar Alg. 1 / Dopamine NeuronRecycler", s_redo, REDO_MUT,
                                     "bitwise composition; float64 score band gamma_64 + gamma_100 + 4u32 near tau",
                                     ENGINE)),
        ("S-graph", lambda: run_check("S-graph", "CUDA graph = eager for every method", s_graph, GRAPH_MUT,
                                      "byte identity of every output and bit identity of the end state", ENGINE)),
        ("S-fresh", lambda: run_check("S-fresh", "fresh controls", s_fresh, FRESH_MUT,
                                      "byte identity with the host and between arms; initial state bitwise", ENGINE)),
        ("S-select", lambda: run_check("S-select", "calibration selection (spec 3.2)", s_select, SELECT_MUT,
                                       "synthetic tables with known answers", VERDICT)),
        ("S-verdict", lambda: run_check("S-verdict", "labels, intervals, sign test, windows, prediction rules",
                                        s_verdict, VERDICT_MUT, "exact binomial values and closed-form fixtures",
                                        VERDICT)),
        ("S-CLI", lambda: run_check("S-CLI", "CLI = function; defaults; admission", s_cli, CLI_MUT,
                                    "byte identity", ENGINE)),
    ]
    for key, fn in plan:
        if only and key not in only:
            continue
        fn()
    RESULTS["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    dump()
    print(json.dumps({k: RESULTS[k] for k in ("all_pass", "n_checks", "n_mutations", "n_mutations_detected")}))


if __name__ == "__main__":
    main()

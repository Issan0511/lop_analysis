#!/usr/bin/env python3
"""Checks for adamw_dose2_1011 (specs/spec_adamw_dose2_1011.md section 6).

    cd ~/Projects/claude/wt/adamw_dose2_1011
    /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/adamw_dose2_1011/checks.py [--only S0,S-wd]

Each check runs on the real code and then on every mutation listed with it; a mutation is an exact-once string
substitution in the check's target file, and it MUST make the same check fail.  Tolerances come from the arithmetic
in each docstring (bit comparisons have none).  Writes results/adamw_dose2_1011/checks.json after every check (with
--only: results/_checks_adamw_dose2_1011/checks_partial.json).  Check runs use the check seeds (100-109, R = 10, or
seed 100 alone), short runs and synthetic inputs; the exceptions are S0 (A1's lambda = 0 path on seeds 0-9, tasks
1-10), S-nochange and S-host-repro (A2's lambda = 0 path and the unmodified host on seeds 0-9), as registered.

A1: S0, S-wd, S-torch, S-branch, S-ctrl, S-graph, S-resume.  A2: S-nochange, S-host-repro, S-coupled-host, S-mut,
S-graph-c51, S-fresh-c51.  Both: S-verdict, S-cli, S-cost.
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
import threading
import time
import types
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch

from src import pmnist_0905 as H                       # noqa: E402
from src import rlcifar_mlp_battle_0918 as B           # noqa: E402
from src import cifar5p1_mlp_0920 as C                 # noqa: E402
from src.cifar_interventions_0920 import core_state, load_pt, tree_hash, make_inputs   # noqa: E402
from analysis.cifar_ledger_0920.replay import environment, output_artifact, setup   # noqa: E402
from analysis.cap_cifar_ee_0920 import stats as CS     # noqa: E402

RUN = "adamw_dose2_1011"
A1F = REPO / "src" / "adamw_dose2_1011.py"
A2F = REPO / "src" / "adamw_dose2_1011_c51.py"
CLI = REPO / "src" / "adamw_dose2_1011_run.py"
VERDICT = REPO / "analysis" / RUN / "verdict.py"
LINES = REPO / "analysis" / RUN / "lines.py"
RULEF = REPO / "analysis" / RUN / "rule.py"
OUT = REPO / "results" / RUN
SCR = REPO / "results" / f"_checks_{RUN}"
DUMP_PATH = OUT / "checks.json"
CAP = REPO / "results" / "cap_cifar_ee_0920"
HOST_DIR = REPO / "results" / "cifar5p1_mlp_0920"
HOST_REC = HOST_DIR / "R_std_lr0.0001"
SOURCES = ("src/adamw_dose2_1011.py", "src/adamw_dose2_1011_c51.py", "src/adamw_dose2_1011_run.py",
           "analysis/adamw_dose2_1011/checks.py", "analysis/adamw_dose2_1011/rule.py",
           "analysis/adamw_dose2_1011/lines.py")
PY = sys.executable
U32, U64 = 2.0 ** -24, 2.0 ** -53
PRINT_REL = 5e-10
# the registered values (spec 2.2, 2.4), written here, not imported
LR1, LR2 = 1e-3, 1e-4
A1_REG = {"wd3e-2": 0.03, "wd1e-1": 0.1, "wd1e0": 1.0}
A2_REG = {"l2c1e-2": ("coupled", 1e-2), "l2c5e-3": ("coupled", 5e-3)}
SEEDS = list(range(10))
CHECK10 = list(range(100, 110))
CHECK3 = [100, 101, 102]
CHECK1 = [100]
B1, B2, AEPS = 0.9, 0.999, 1e-8
RESULTS: dict = {}
HOST_KEYS_UNITS = ("zmean_l1", "zsd_l1", "gmean_l1", "q_l1", "zero_l1", "neff_l1", "neff_fraction_l1", "Wnorm_l1",
                   "bias_l1", "argmean_l1", "zmean_over_sd_l1", "zmean_l2", "zsd_l2", "gmean_l2", "q_l2", "zero_l2",
                   "neff_l2", "neff_fraction_l2", "Wnorm_l2", "bias_l2", "argmean_l2", "zmean_over_sd_l2", "mu2",
                   "cos_W2_mu2", "mu2_norm")

# --------------------------------------------------------------------------
# infrastructure (adamw_dose_1010's)
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
    mod._is_real = not subs
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def same_bytes(a: Path, b: Path) -> bool:
    return Path(a).exists() and Path(b).exists() and Path(a).read_bytes() == Path(b).read_bytes()


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def brief(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "pass" and not isinstance(v, (dict, list))} | \
        {"failed_items": r.get("failed_items", [])[:12]}


def dump() -> None:
    checks = {k: v for k, v in RESULTS.items() if k.startswith("S") and isinstance(v, dict) and "pass" in v}
    RESULTS["all_pass"] = bool(checks) and all(v.get("pass") and v.get("all_mutations_detected", True)
                                               for v in checks.values())
    RESULTS["n_checks"] = len(checks)
    RESULTS["n_mutations"] = sum(len(v.get("mutations", [])) for v in checks.values())
    RESULTS["n_mutations_detected"] = sum(m["detected"] for v in checks.values() for m in v.get("mutations", []))
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMP_PATH.write_text(json.dumps(RESULTS, indent=2, default=str))


def finish(key: str, entry: dict, t0: float) -> None:
    if entry["mutations"]:
        entry["all_mutations_detected"] = all(m["detected"] for m in entry["mutations"])
    entry["mutations_raised"] = sum("raised" in m for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}"
          f" (raised {entry['mutations_raised']})  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:4]) if not entry['pass'] else ''}", flush=True)
    for m in entry["mutations"]:
        if not m["detected"]:
            print(f"   NOT DETECTED: {m['mutation']}", flush=True)


def run_check(key: str, title: str, fn, mutations: list, derivation: str, target: Path, threads: int = 2) -> None:
    torch.set_num_threads(threads)
    t0 = time.time()
    base = fn(load(target))
    entry = {"title": title, "target": str(target.relative_to(REPO)), "threshold_derivation": derivation,
             "threads": threads, **base, "mutations": []}
    for label, subs in mutations:
        try:
            r = fn(load(target, subs))
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": bool(r["pass"]),
                                       "detected": not r["pass"], "mutant_result": brief(r)})
        except AssertionError:
            raise
        except Exception as e:
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
        torch.cuda.empty_cache()
    finish(key, entry, t0)


def result(per: dict, extra: dict | None = None) -> dict:
    failed = [k for k, v in per.items() if isinstance(v, bool) and not v]
    return {"pass": not failed, "failed_items": failed, "detail": per, **(extra or {})}


def _equal_list(a, b) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def quiet(_m):
    pass


class StopCheck(Exception):
    pass


_DEV = [None]
_CIFAR100 = [None]


def dev():
    if _DEV[0] is None:
        _DEV[0] = setup()
    return _DEV[0]


def cifar100():
    if _CIFAR100[0] is None:
        _CIFAR100[0] = C.Cifar100()
    return _CIFAR100[0]


_prefix_cache: dict = {}


def prefix_state(M_real, seeds) -> dict:
    """Task 1 on the lambda = 0 path for check seeds (the A1 checks' common prefix), computed once."""
    key = tuple(seeds)
    if key not in _prefix_cache:
        st, rows, unit = M_real.train_prefix(list(seeds))
        _prefix_cache[key] = st
        torch.cuda.empty_cache()
    return _prefix_cache[key]


_REAL_A1 = [None]


def real_a1():
    if _REAL_A1[0] is None:
        _REAL_A1[0] = load(A1F)
    return _REAL_A1[0]


def rows_equal(mine: list[dict], ref: list[dict]) -> tuple[bool, list]:
    """Every key of our rows equals the record's (the record may carry more keys)."""
    diff = []
    if len(mine) != len(ref):
        return False, [("n", len(mine), len(ref))]
    for a, b in zip(sorted(mine, key=lambda r: r["seed"]), sorted(ref, key=lambda r: r["seed"])):
        for k, v in a.items():
            if k in ("arm", "lam"):
                continue
            if k not in b or b[k] != v:
                diff.append((a["seed"], a["task"], k, v, b.get(k)))
    return not diff, diff[:6]


def units_equal(mine: dict, ref: dict) -> tuple[bool, list]:
    bad = [k for k in HOST_KEYS_UNITS if k not in mine or k not in ref or
           not np.array_equal(np.asarray(mine[k]), np.asarray(ref[k]), equal_nan=True)]
    return not bad, bad


def npz(p: Path) -> dict:
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


# --------------------------------------------------------------------------
# S0: A1's lambda = 0 path reproduces the cap_cifar_ee_0920 ref record bit for bit (spec 2.2)
# --------------------------------------------------------------------------

S0_TASKS = 10


def s0(M) -> dict:
    """(a) seeds 0-9, task 1 from the initial values on the lambda = 0 path of WDEngine (arm 'ref0', CUDA graph):
    the core state (P, m, v, tc, label and batch generator states, Y) hashes to the archived prefix/t01.pt's, every
    row equals the committed prefix/rows.json on every key the host writes, every host unit array equals the archived
    prefix/units.npz; (b) from prefix/t01.pt, tasks 2-10 on the same path: each task's rows equal the committed
    arms/ref/rows.json and its unit arrays the archived arms/ref/units_tXX.npz, and the core state after task 10
    hashes to the archived arms/ref/t10.pt.  Bit comparisons; the first task that differs ends the check (fail-fast)."""
    per, extra = {}, {}
    d = dev()
    X, _ = make_inputs(SEEDS, d)
    e = M.WDEngine(SEEDS, X, lam=M.ARM_LAMBDA["ref0"], scope=M.ARM_SCOPE["ref0"])
    rows, unit = e.train_task(1, M.EPOCHS)
    anchor = load_pt(M.CAP_PREFIX)
    per["a_t1_core_state"] = tree_hash(core_state(e.state())) == tree_hash(core_state(anchor))
    ok, diff = rows_equal(rows, json.loads((CAP / "prefix" / "rows.json").read_text()))
    per["a_t1_rows"] = ok
    extra["a_rows_diff"] = diff
    ok, bad = units_equal(unit, npz(output_artifact(CAP, Path("prefix/units.npz"))))
    per["a_t1_units"] = ok
    extra["a_units_bad"] = bad
    del e
    torch.cuda.empty_cache()
    if not all(per.values()):
        return result(per, extra)
    e = M.WDEngine(SEEDS, X, state=anchor, lam=M.ARM_LAMBDA["ref0"], scope=M.ARM_SCOPE["ref0"])
    ref_all = json.loads((CAP / "arms" / "ref" / "rows.json").read_text())
    keep_rows = []
    for t in range(2, S0_TASKS + 1):
        rr, uu = e.train_task(t, M.EPOCHS)
        keep_rows += rr
        ok_r, diff = rows_equal(rr, [r for r in ref_all if r["task"] == t])
        ok_u, bad = units_equal(uu, npz(output_artifact(CAP, Path(f"arms/ref/units_t{t:02d}.npz"))))
        per[f"b_t{t}_rows"] = ok_r
        per[f"b_t{t}_units"] = ok_u
        if not (ok_r and ok_u):
            extra["b_first_diff"] = {"task": t, "rows": diff, "units": bad}
            break
    else:
        per["b_t10_core_state"] = tree_hash(core_state(e.state())) == \
            tree_hash(core_state(load_pt(CAP / "arms" / "ref" / "t10.pt")))
    if getattr(M, "_is_real", False) and all(per.values()):
        keep = OUT / "rlcifar" / "_s0"
        keep.mkdir(parents=True, exist_ok=True)
        M.write_csv(keep / "per_task.csv", rows + keep_rows)
        (keep / "summary.json").write_text(json.dumps({"tasks": [1, S0_TASKS], "seeds": SEEDS, "pass": True,
                                                       "core_sha256_t10": tree_hash(core_state(e.state()))}, indent=2))
    del e
    torch.cuda.empty_cache()
    return result(per, extra)


S0_MUT = [
    ("Ms1: the lambda = 0 path decayed with lambda 1e-3", [('ARM_LAMBDA = {"ref0": 0.0,', 'ARM_LAMBDA = {"ref0": 1e-3,')]),
    ("Ms2: Adam eps 1e-7", [("        p.sub_(1e-3 * (mi * inv1) / ((vi * inv2).sqrt() + 1e-8))\n",
                             "        p.sub_(1e-3 * (mi * inv1) / ((vi * inv2).sqrt() + 1e-7))\n")]),
    ("Ms3: the diagnostic draws from the batch stream",
     [('        unit["Wnorm_l3"] = self.P[4].detach().double().norm(dim=2).cpu().numpy()\n',
       '        unit["Wnorm_l3"] = self.P[4].detach().double().norm(dim=2).cpu().numpy()\n'
       '        torch.randperm(2, generator=self.g_batch[self.seeds[0]])\n')]),
]

# --------------------------------------------------------------------------
# S-wd: decoupled, on synthetic tensors (A1)
# --------------------------------------------------------------------------

K_WD = 40


def _np_iter(a: np.ndarray, c: np.float32, k: int) -> np.ndarray:
    a = a.astype(np.float32).copy()
    for _ in range(k):
        a *= c
    return a


def s_wd(M) -> dict:
    """A1's update on CUDA with the stacked shapes (R = 2), the registered lambda of 'wd1e0' (1) on SCOPES['all'],
    wd_c filled with M.decay_factor(lambda) into a float32 0-dim tensor as WDEngine does, inv1/inv2 filled per step
    from python doubles as Engine.one_update does, K = 40 steps of random gradients in which a fixed third of every
    tensor's entries gets an exactly-0 gradient: (i) those entries equal np.float32 iteration p <- p * c,
    c = np.float32(1 - 1e-3 * 1), after 1 and after K steps; (ii) the whole update equals the independent
    'p * c, then Engine.step's Adam ops' bit for bit, m and v too, and the device counters read K; (iii) coupled L2
    (lambda theta added to the gradient, then Adam), computed here, moves the (i) entries by Adam's exact first step
    eta g / (|g| + eps), g = lambda theta (within 1e-3 eta), which is eta sign(theta) to 1.1% where |g| >= 100 eps,
    and so differs from (i)."""
    dv = dev()
    g0 = torch.Generator().manual_seed(1011)
    shapes = [(2, 100, 3072), (2, 100), (2, 100, 100), (2, 100), (2, 10, 100), (2, 10)]
    fan = [3072, 3072, 100, 100, 100, 100]
    P = [((torch.rand(s, generator=g0) * 2 - 1) / math.sqrt(f)) for s, f in zip(shapes, fan)]
    masks = [torch.rand(s, generator=g0) < 1 / 3 for s in shapes]
    gseq = [[torch.where(mk, torch.zeros(s), torch.randn(s, generator=g0) * 1e-2).to(dv) for s, mk in zip(shapes, masks)]
            for _ in range(K_WD)]
    lam = M.ARM_LAMBDA["wd1e0"]
    c = np.float32(1 - LR1 * 1.0)
    wd_c = torch.ones((), device=dv)
    wd_c.fill_(M.decay_factor(lam))
    Pm = [q.clone().to(dv) for q in P]
    m_, v_ = [torch.zeros_like(q) for q in Pm], [torch.zeros_like(q) for q in Pm]
    Pr = [q.clone().to(dv) for q in P]
    mr, vr = [torch.zeros_like(q) for q in Pm], [torch.zeros_like(q) for q in Pm]
    cnt = torch.zeros(6, dtype=torch.long, device=dv)
    i1, i2 = torch.zeros((), device=dv), torch.zeros((), device=dv)
    ok1 = okK = True
    for k in range(K_WD):
        tc = k + 1
        i1.fill_(1.0 / (1 - 0.9 ** tc))
        i2.fill_(1.0 / (1 - 0.999 ** tc))
        M.adamw_update_(Pm, gseq[k], m_, v_, i1, i2, wd_c, M.SCOPES["all"], cnt)
        for i in range(6):
            Pr[i].mul_(float(c))
            mr[i].mul_(0.9).add_(gseq[k][i], alpha=1 - 0.9)
            vr[i].mul_(0.999).addcmul_(gseq[k][i], gseq[k][i], value=1 - 0.999)
            Pr[i].sub_(1e-3 * (mr[i] * i1) / ((vr[i] * i2).sqrt() + 1e-8))
        if k == 0:
            for i in range(6):
                ok1 &= bool(np.array_equal(Pm[i].cpu()[masks[i]].numpy(), _np_iter(P[i][masks[i]].numpy(), c, 1)))
    for i in range(6):
        okK &= bool(np.array_equal(Pm[i].cpu()[masks[i]].numpy(), _np_iter(P[i][masks[i]].numpy(), c, K_WD)))
    i1.fill_(1.0 / (1 - 0.9))
    i2.fill_(1.0 / (1 - 0.999))
    dev_, dev_sign, differs = 0.0, 0.0, True
    for i in range(6):
        Pc = P[i].clone().to(dv)
        gc = gseq[0][i].add(Pc, alpha=1.0)
        mc = (1 - 0.9) * gc
        vc = (1 - 0.999) * gc * gc
        Pc = Pc - LR1 * (mc * i1) / ((vc * i2).sqrt() + AEPS)
        th = P[i][masks[i]].double()
        g = 1.0 * th
        dd = (Pc.cpu() - P[i])[masks[i]].double()
        dev_ = max(dev_, float((dd + LR1 * g / (g.abs() + AEPS)).abs().max()))
        big = g.abs() >= 100 * AEPS
        dev_sign = max(dev_sign, float((dd[big] + LR1 * torch.sign(th[big])).abs().max()))
        differs &= not np.array_equal(Pc.cpu()[masks[i]].numpy(), _np_iter(P[i][masks[i]].numpy(), c, 1))
    per = {"i_zero_grad_one_step_is_c": ok1, "i_zero_grad_K_steps_is_c_iterated": okK,
           "ii_update_is_decay_then_host_adam": _equal_list(Pm, Pr) and _equal_list(m_, mr) and _equal_list(v_, vr),
           "ii_counters": cnt.cpu().tolist() == [K_WD] * 6,
           "iii_coupled_is_adam_first_step": dev_ <= 1e-3 * LR1, "iii_coupled_moves_eta_sign": dev_sign <= 0.011 * LR1,
           "iii_coupled_differs": differs}
    return result(per, {"coupled_max_dev_from_first_step": dev_, "coupled_max_dev_from_eta_sign": dev_sign,
                        "c_hex": float(c).hex()})


SWD_MUT = [
    ("Mw1: coupled L2 (lambda theta into the gradient)",
     [("            p.mul_(wd_c)\n", "            gr = gr.add(p.detach() * (1 - wd_c) / 1e-3)\n")]),
    ("Mw2: the decay after the Adam step",
     [("            p.mul_(wd_c)\n            wd_count[i].add_(1)\n", "            wd_count[i].add_(1)\n"),
      ("        p.sub_(1e-3 * (mi * inv1) / ((vi * inv2).sqrt() + 1e-8))\n",
       "        p.sub_(1e-3 * (mi * inv1) / ((vi * inv2).sqrt() + 1e-8))\n        if i in decay_idx:\n            p.mul_(wd_c)\n")]),
    ("Mw3: biases not decayed", [('SCOPES = {"all": (0, 1, 2, 3, 4, 5),', 'SCOPES = {"all": (0, 2, 4),')]),
    ("Mw4: W3 only", [('SCOPES = {"all": (0, 1, 2, 3, 4, 5),', 'SCOPES = {"all": (4,),')]),
    ("Mw5: c = 1 - lambda (the lr forgotten)", [("    return 1 - LR * lam\n", "    return 1 - lam\n")]),
    ("Mw6: lambda doubled", [("    return 1 - LR * lam\n", "    return 1 - LR * lam * 2\n")]),
]

# --------------------------------------------------------------------------
# S-torch: the same formula as torch.optim.AdamW, within the rounding band (A1)
# --------------------------------------------------------------------------

N_OP_REG = 21      # adamw_dose_1010 spec 6 / 13.2: the roundings of one update's Adam term, both codes together


class RegBand:
    """|dp_K| <= sum_t [n_op u eta |s_t| + 2 u |p_t|] element by element (adamw_dose_1010 spec 6), n_op = 21,
    u = 2^-24, |s_t| = |m_hat / (sqrt(v_hat) + eps)| of our code, |p_t| = the larger of the two codes' values
    before and after the update."""

    def __init__(self, shapes, device):
        self.B = [torch.zeros(s, dtype=torch.float64, device=device) for s in shapes]
        self.step_bands = []

    def add(self, m_after, v_after, tc: int, lr: float, p_pairs) -> None:
        c1, c2 = 1 - B1 ** tc, 1 - B2 ** tc
        tot = 0.0
        for i, (mm, vv) in enumerate(zip(m_after, v_after)):
            s_ = ((mm.double() / c1) / ((vv.double() / c2).sqrt() + AEPS)).abs()
            pm = torch.stack([q.double().abs() for q in p_pairs[i]]).amax(0)
            inc = N_OP_REG * U32 * lr * s_ + 2 * U32 * pm
            self.B[i] += inc
            tot = max(tot, float(inc.median()))
        self.step_bands.append(tot)


def _band_compare(Pm, Pt, band) -> tuple[bool, float, int, int]:
    worst, n_out, n_all = 0.0, 0, 0
    for a, b, bd in zip(Pm, Pt, band.B):
        d = (a.double() - b.double()).abs()
        r = d / bd.clamp_min(1e-300)
        worst = max(worst, float(r.max()))
        n_out += int((d > bd).sum())
        n_all += d.numel()
    return n_out == 0, worst, n_out, n_all


S_TORCH_STEPS = 150


def s_torch(M) -> dict:
    """Seeds 100-101 stacked (R = 2), the arm wd1e0 (registered lambda 1), task 1 from the initial values, eager,
    the first 150 updates' gradients taken from the step hook and fed in lockstep to (a) M.adamw_update_ on a shadow
    copy -- which must equal the run's own parameters bit for bit after every update --, (b) torch.optim.AdamW(lr 1e-3,
    betas (0.9, 0.999), eps 1e-8, weight_decay 1, foreach=False) on the stacked tensors, within RegBand, with second
    moments identical at every update, and (c) torch.optim.Adam(weight_decay 1) (coupled), outside the band."""
    dv = dev()
    seeds = [100, 101]
    X, _ = make_inputs(seeds, dv)
    e = M.WDEngine(seeds, X, lam=M.ARM_LAMBDA["wd1e0"], scope=M.ARM_SCOPE["wd1e0"], graph=False)
    st = {"k": 0, "v_same": True, "shadow_ok": True}

    def hook(ev, **s):
        eng = s["engine"]
        if ev == "grads":
            if st["k"] == 0:
                init = [q.detach().clone() for q in eng.P]
                st["Pm"] = [q.clone() for q in init]
                st["m"] = [torch.zeros_like(q) for q in init]
                st["v"] = [torch.zeros_like(q) for q in init]
                st["wd_c"] = torch.ones((), device=dv)
                st["wd_c"].fill_(M.decay_factor(M.ARM_LAMBDA["wd1e0"]))
                st["cnt"] = torch.zeros(6, dtype=torch.long, device=dv)
                st["pt"] = [torch.nn.Parameter(q.clone()) for q in init]
                st["pc"] = [torch.nn.Parameter(q.clone()) for q in init]
                st["ot"] = torch.optim.AdamW(st["pt"], lr=LR1, betas=(B1, B2), eps=AEPS, weight_decay=1.0, foreach=False)
                st["oc"] = torch.optim.Adam(st["pc"], lr=LR1, betas=(B1, B2), eps=AEPS, weight_decay=1.0, foreach=False)
                st["band"] = RegBand([q.shape for q in init], dv)
            g = [x.detach() for x in s["grads"]]
            before = [q.clone() for q in st["Pm"]]
            before_t = [q.detach().clone() for q in st["pt"]]
            M.adamw_update_(st["Pm"], g, st["m"], st["v"], eng.inv1, eng.inv2, st["wd_c"], M.SCOPES["all"], st["cnt"])
            for q, gg in zip(st["pt"], g):
                q.grad = gg.clone()
            st["ot"].step()
            for q, gg in zip(st["pc"], g):
                q.grad = gg.clone()
            st["oc"].step()
            st["v_same"] &= _equal_list(st["v"], [st["ot"].state[q]["exp_avg_sq"] for q in st["pt"]])
            st["k"] += 1
            st["band"].add(st["m"], st["v"], st["k"], LR1,
                           [(before[i], before_t[i], st["Pm"][i], st["pt"][i].detach()) for i in range(6)])
        elif ev == "post":
            st["shadow_ok"] &= _equal_list(st["Pm"], [q.detach() for q in eng.P])
            if st["k"] >= S_TORCH_STEPS:
                raise StopCheck

    e.hook = hook
    try:
        e.train_task(1, M.EPOCHS)
    except StopCheck:
        pass
    ok_t, worst_t, nout_t, nall = _band_compare(st["Pm"], [q.detach() for q in st["pt"]], st["band"])
    ok_c, worst_c, nout_c, _ = _band_compare(st["Pm"], [q.detach() for q in st["pc"]], st["band"])
    per = {"a_shadow_equals_run_every_update": st["shadow_ok"], "b_adamw_within_band": ok_t,
           "b_second_moments_identical": st["v_same"], "c_coupled_adam_outside_band": not ok_c,
           "n_updates": st["k"] == S_TORCH_STEPS}
    del e
    torch.cuda.empty_cache()
    return result(per, {"worst_ratio_adamw": worst_t, "n_outside_adamw": nout_t, "worst_ratio_coupled": worst_c,
                        "n_outside_coupled": nout_c, "n_elements": nall,
                        "band_per_update_median": float(np.median(st["band"].step_bands))})


STORCH_MUT = [
    ("Mt1: the decay after the Adam step",
     [("            p.mul_(wd_c)\n            wd_count[i].add_(1)\n", "            wd_count[i].add_(1)\n"),
      ("        p.sub_(1e-3 * (mi * inv1) / ((vi * inv2).sqrt() + 1e-8))\n",
       "        p.sub_(1e-3 * (mi * inv1) / ((vi * inv2).sqrt() + 1e-8))\n        if i in decay_idx:\n            p.mul_(wd_c)\n")]),
    ("Mt2: lambda theta added to the gradient",
     [("            p.mul_(wd_c)\n", "            gr = gr.add(p.detach() * (1 - wd_c) / 1e-3)\n")]),
]


# --------------------------------------------------------------------------
# S-branch: the arms branch from one task-1 state and decay from their first update (A1)
# --------------------------------------------------------------------------

def s_branch(M) -> dict:
    """Check seeds 100-109, a task-1 prefix on the lambda = 0 path (computed once with the real module).
    (i) run_arm for ref0 and the three registered arms from that prefix (task 2 x 2 epochs): each branch.json records
    the prefix's core hash as its own starting state; task 2's label_hash and order_hash are equal across the four
    arms and the label_hash equals the hash of labels drawn here, independently, from the prefix's saved label
    generators; the task-2 end states differ between the arms; (ii) the first update of wd1e0 (eager) equals
    fl(fl(p0 * c) - a) element by element, with a = Engine.step's Adam term recomputed from the same update's m and v
    and c = np.float32(1 - 1e-3 * 1), on all six tensors; (iii) ARM_LAMBDA and ARM_SCOPE of the registered arms are
    the registered values (written here)."""
    per, extra = {}, {}
    pre = prefix_state(real_a1(), CHECK10)
    pre_core = tree_hash(core_state(pre))
    outs, cores = {}, {}
    for arm in ("ref0", "wd3e-2", "wd1e-1", "wd1e0"):
        o = SCR / f"branch_{arm}_{M.__name__}"
        shutil.rmtree(o, ignore_errors=True)
        fin = {}
        M.run_arm(arm, o, seeds=CHECK10, tasks=2, epochs=2, prefix=pre, progress=quiet, keep_units=False, final=fin)
        outs[arm] = o
        cores[arm] = tree_hash(core_state(fin["state"]))
        torch.cuda.empty_cache()
    br = {a: json.loads((o / "branch.json").read_text()) for a, o in outs.items()}
    per["i_branch_from_prefix"] = all(b["prefix_core_sha256"] == pre_core and b["branch_core_sha256"] == pre_core
                                      for b in br.values())
    rows = {a: {(r["seed"], r["task"]): r for r in read_csv(o / "per_task.csv")} for a, o in outs.items()}
    keys = sorted(rows["ref0"])
    per["i_task2_hashes_equal_across_arms"] = all(rows[a][k]["label_hash"] == rows["ref0"][k]["label_hash"] and
                                                  rows[a][k]["order_hash"] == rows["ref0"][k]["order_hash"]
                                                  for a in rows for k in keys)
    gl = {s: torch.Generator() for s in CHECK10}
    for s in CHECK10:
        gl[s].set_state(pre["g_lab"][s])
    Y = torch.stack([B.RC.task_labels(gl[s]) for s in CHECK10]).to(dev())
    per["i_task2_labels_continue_prefix_stream"] = all(rows["ref0"][(str(s), "2")]["label_hash"] == tree_hash(Y)
                                                       for s in CHECK10)
    per["i_end_states_differ"] = len(set(cores.values())) == 4
    # (ii) the first update
    X, _ = make_inputs(CHECK10, dev())
    e = M.WDEngine(CHECK10, X, state=pre, lam=M.ARM_LAMBDA["wd1e0"], scope=M.ARM_SCOPE["wd1e0"], graph=False)
    snap = {}

    def hook(ev, **s):
        eng = s["engine"]
        if ev == "grads":
            snap["p0"] = [q.detach().clone() for q in eng.P]
        else:
            snap["p1"] = [q.detach().clone() for q in eng.P]
            snap["m"] = [q.clone() for q in eng.m]
            snap["v"] = [q.clone() for q in eng.v]
            snap["i1"], snap["i2"] = eng.inv1.clone(), eng.inv2.clone()
            raise StopCheck

    e.hook = hook
    try:
        e.train_task(2, M.EPOCHS)
    except StopCheck:
        pass
    c = float(np.float32(1 - LR1 * 1.0))
    ok = True
    for p0, p1, mi, vi in zip(snap["p0"], snap["p1"], snap["m"], snap["v"]):
        a = 1e-3 * (mi * snap["i1"]) / ((vi * snap["i2"]).sqrt() + 1e-8)
        ok &= torch.equal(p1, p0 * c - a)
    per["ii_first_update_is_decay_then_adam"] = bool(ok)
    del e
    torch.cuda.empty_cache()
    per["iii_lambda_map_registered"] = {a: M.ARM_LAMBDA[a] for a in A1_REG} == A1_REG and \
        all(M.ARM_SCOPE[a] == "all" for a in A1_REG) and M.ARM_LAMBDA["ref0"] == 0.0
    extra["cores"] = cores
    return result(per, extra)


BRANCH_MUT = [
    ("Mb1: the decay factor never written (the graph and the eager step multiply by 1)",
     [("        if self.decay:\n            self.wd_c.fill_(decay_factor(self.lam))", "        if False:\n            self.wd_c.fill_(decay_factor(self.lam))")]),
    ("Mb2: the label streams re-initialised per arm",
     [("        if self.decay:\n            self.wd_c.fill_(decay_factor(self.lam))",
       "        self.g_lab = {s: E.H.stream('rlc_labels', s) for s in self.seeds}\n"
       "        if self.decay:\n            self.wd_c.fill_(decay_factor(self.lam))")]),
    ("Mb3: lambda map swapped (wd3e-2 <-> wd1e-1)",
     [('"wd3e-2": 0.03, "wd1e-1": 0.1,', '"wd3e-2": 0.1, "wd1e-1": 0.03,')]),
]

# --------------------------------------------------------------------------
# S-ctrl: readout-only WD lets the hidden widths grow (A1 mutation control)
# --------------------------------------------------------------------------

def s_ctrl(M) -> dict:
    """Seed 100 alone (R = 1): task 1 on the lambda = 0 path (computed once with the real module), then tasks 2-4
    (400 epochs each, graph) for ref0, ro1 (lambda 1 on W3 and b3 only) and wd1e0 (lambda 1 on all six).  The unit
    median of each hidden layer's width s (sd of z over the 1200 images) at the end of task 4: in both layers
    s_ro > sqrt(s_ref * s_wd) (the geometric midpoint of the two references, not a value chosen after seeing them) and
    s_wd < s_ref.  Online accuracy is not read."""
    pre = prefix_state(real_a1(), CHECK1)
    s = {}
    for arm in ("ref0", "ro1", "wd1e0"):
        o = SCR / f"ctrl_{arm}_{M.__name__}"
        shutil.rmtree(o, ignore_errors=True)
        M.run_arm(arm, o, seeds=CHECK1, tasks=4, epochs=M.EPOCHS, prefix=pre, progress=quiet, save_states=())
        u = npz(o / "units_t04.npz")
        s[arm] = {l: float(np.median(u[f"zsd_l{l}"][0])) for l in (1, 2)}
        torch.cuda.empty_cache()
    per = {}
    for l in (1, 2):
        mid = math.sqrt(s["ref0"][l] * s["wd1e0"][l])
        per[f"l{l}_readout_only_above_midpoint"] = s["ro1"][l] > mid
        per[f"l{l}_all_tensors_below_ref"] = s["wd1e0"][l] < s["ref0"][l]
    return result(per, {"widths_t4": s})


CTRL_MUT = [("Mc1: ro1 applied to all six tensors", [('"ro1": "readout"}', '"ro1": "all"}')])]

# --------------------------------------------------------------------------
# S-graph, S-resume (A1)
# --------------------------------------------------------------------------

def s_graph(M) -> dict:
    """wd1e0 on check seeds 100-109 from the prefix, task 2 x 4 epochs (300 updates): the CUDA graph and eager steps
    give byte-identical per_task.csv and wd_task.csv and bit-identical final P, m, v; the decays were counted (300)."""
    pre = prefix_state(real_a1(), CHECK10)
    fins = {}
    for g in (True, False):
        o = SCR / f"graph_{g}_{M.__name__}"
        shutil.rmtree(o, ignore_errors=True)
        fins[g] = {}
        M.run_arm("wd1e0", o, seeds=CHECK10, tasks=2, epochs=4, prefix=pre, graph=g, progress=quiet,
                  keep_units=False, final=fins[g])
        torch.cuda.empty_cache()
    oa, ob = SCR / f"graph_True_{M.__name__}", SCR / f"graph_False_{M.__name__}"
    per = {f"{f}_identical": same_bytes(oa / f, ob / f) for f in ("per_task.csv", "wd_task.csv")}
    per["final_P_m_v_bitwise"] = all(_equal_list(fins[True]["state"][k], fins[False]["state"][k]) for k in ("P", "m", "v"))
    wd = read_csv(oa / "wd_task.csv")
    per["decays_counted"] = all(int(r[f"wd_steps_{nm}"]) == 300 for r in wd for nm in ("W1", "b1", "W2", "b2", "W3", "b3"))
    per["wd_c_ok"] = all(r["wd_c_ok"] == "1" for r in wd)
    return result(per)


GRAPH_MUT = [("Mg1: the decay factor rebound outside the graph (the graph keeps reading 1)",
              [("            self.wd_c.fill_(decay_factor(self.lam))",
                "            self.wd_c = torch.full((), decay_factor(self.lam), device=X.device)")])]


def s_resume(M) -> dict:
    """wd1e0 on check seeds 100-109 from the prefix, tasks 2-3 x 4 epochs: a run stopped after epoch 2 of task 2 (its
    epoch-boundary checkpoint) and resumed gives byte-identical per_task.csv and wd_task.csv, and a bit-identical
    final state, to an uninterrupted run."""
    pre = prefix_state(real_a1(), CHECK10)
    oa, ob = SCR / f"resume_a_{M.__name__}", SCR / f"resume_b_{M.__name__}"
    for o in (oa, ob):
        shutil.rmtree(o, ignore_errors=True)
    fa, fb = {}, {}
    M.run_arm("wd1e0", oa, seeds=CHECK10, tasks=3, epochs=4, prefix=pre, progress=quiet, keep_units=False, final=fa)
    stopped = M.run_arm("wd1e0", ob, seeds=CHECK10, tasks=3, epochs=4, prefix=pre, progress=quiet, keep_units=False,
                        epoch_limit=2)
    torch.cuda.empty_cache()
    had_active = (ob / "active.pt").exists()
    done = M.run_arm("wd1e0", ob, seeds=CHECK10, tasks=3, epochs=4, prefix=pre, progress=quiet, keep_units=False,
                     final=fb)
    torch.cuda.empty_cache()
    per = {"stopped_first": stopped is False and had_active, "resumed_done": bool(done)}
    for f in ("per_task.csv", "wd_task.csv"):
        per[f"{f}_identical"] = same_bytes(oa / f, ob / f)
    per["final_state_bitwise"] = bool(fb) and tree_hash(core_state(fa["state"])) == tree_hash(core_state(fb["state"])) \
        and torch.equal(fa["state"]["wd"]["wd_count"], fb["state"]["wd"]["wd_count"])
    return result(per)


RESUME_MUT = [("Mr1: the resumed engine ignores the saved decay count",
               [('            self.wd_count.copy_(st["wd"]["wd_count"])', "            pass")])]


# --------------------------------------------------------------------------
# A2: S-nochange, S-host-repro, S-coupled-host, S-mut, S-graph-c51, S-fresh-c51
# --------------------------------------------------------------------------

_host_cache: dict = {}


def dev2():
    return H.setup("cuda")


def erun(M, arm, seeds, tasks, out: Path, **kw):
    shutil.rmtree(out, ignore_errors=True)
    return M.run(arm, seeds, tasks, dev2(), out, cifar=cifar100(), progress=quiet, **kw)


def hrun(seeds, tasks, steps_hard=C.STEPS_PER_TASK, steps_easy=C.STEPS_PER_TASK, fresh=True, iv="none") -> Path:
    """The host's unmodified run(), R / std / lr 1e-4 / graph -- cached per configuration."""
    key = (iv, tuple(seeds), tasks, steps_hard, steps_easy, fresh)
    if key not in _host_cache:
        out = SCR / f"host_{len(_host_cache)}"
        shutil.rmtree(out, ignore_errors=True)
        C.run("R", list(seeds), "std", tasks, dev2(), out, lr=1e-4, cifar=cifar100(), fresh=fresh, graph=True,
              iv=iv, progress=quiet, steps_hard=steps_hard, steps_easy=steps_easy)
        _host_cache[key] = out
    return _host_cache[key]


def same_but_eff_rank(a: Path, b: Path) -> tuple[bool, list]:
    A = [l.split(",") for l in Path(a).read_text().splitlines()]
    Bt = [l.split(",") for l in Path(b).read_text().splitlines()]
    if len(A) != len(Bt) or A[0] != Bt[0]:
        return False, ["shape or header"]
    er = {i for i, c in enumerate(A[0]) if c.startswith("eff_rank")}
    diff = [(ra[2], ra[5], A[0][i]) for ra, rb in zip(A[1:], Bt[1:]) for i, (x, y) in enumerate(zip(ra, rb))
            if i not in er and x != y]
    return not diff, diff[:10]


def gamma(n: int, u: float = U64) -> float:
    return n * u / (1.0 - n * u)


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
    """cap_cifar5p1_1007 spec 7.1 (as in adamw_dose_1010)."""
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
    """spec 2.4 / adamw_dose_1010 2.3 (a)(b): the lambda = 0 path, seeds 0-9, 30 tasks, graph, 2 threads, fresh:
    (a) per_task.csv and fresh_control.csv byte-identical to the unmodified host run() in this environment; (b)
    identical to the committed R record in every column but eff_rank, the committed eff_rank inside the
    cap_cifar5p1_1007 spec 7.1 band around the value recomputed here (the band rejects the next task's value:
    non-vacuity), fresh_control.csv byte-identical to the committed one; (c) a 4-task run is byte-identical to the
    host's 4-task run.  The real module's outputs are kept in results/adamw_dose2_1011/c51/_nochange/."""
    per, extra = {}, {}
    out_e = SCR / f"nochange_{M.__name__}"
    bands = {}

    def obs(t, st):
        z1, a1, z2, a2, _ = B.forward(st["P"], st["Xt"], st["act"], train=False)
        for li, a in ((1, a1), (2, a2)):
            for r in st["live"]:
                bands[(SEEDS[r], t, li)] = eff_rank_band(a[r])

    erun(M, "nochange", SEEDS, C.N_TASKS, out_e, observer=obs)
    host = hrun(SEEDS, C.N_TASKS)
    per["a_per_task_identical_to_host_now"] = same_bytes(out_e / "per_task.csv", host / "per_task.csv")
    per["a_fresh_identical_to_host_now"] = same_bytes(out_e / "fresh_control.csv", host / "fresh_control.csv")
    mine, com = read_csv(out_e / "per_task.csv"), read_csv(HOST_REC / "per_task.csv")
    per["b_same_rows"] = len(mine) == len(com) == len(SEEDS) * C.N_TASKS and \
        all((a["seed"], a["task"]) == (b["seed"], b["task"]) for a, b in zip(mine, com))
    per["b_same_columns"] = list(mine[0]) == list(com[0])
    ok, diff = same_but_eff_rank(out_e / "per_task.csv", HOST_REC / "per_task.csv")
    per["b_non_eff_rank_identical"] = ok
    extra["b_non_eff_rank_diff_cells"] = diff
    per["b_fresh_identical_to_committed"] = same_bytes(out_e / "fresh_control.csv", HOST_REC / "fresh_control.csv")
    inside, printed_ok, n_rows, shifted_rejected, shifted_n, outside = True, True, 0, 0, 0, []
    com_idx = {(int(b["seed"]), int(b["task"])): b for b in com}
    for a, b in zip(mine, com):
        s_, t = int(a["seed"]), int(a["task"])
        for li in (1, 2):
            k = f"eff_rank_l{li}"
            bd = bands.get((s_, t, li))
            if bd is None:
                inside = False
                outside.append((s_, t, li, "no band"))
                continue
            n_rows += 1
            printed_ok &= f"{bd['value']:.10g}" == a[k]
            cv = float(b[k])
            lo, hi = bd["lo"] * (1 - 2 * PRINT_REL), bd["hi"] * (1 + 2 * PRINT_REL)
            okk = lo <= cv <= hi
            inside &= okk
            if not okk:
                outside.append((s_, t, li, cv, bd["lo"], bd["hi"]))
            nb = com_idx.get((s_, t + 1))
            if nb is not None:
                shifted_n += 1
                shifted_rejected += not (lo <= float(nb[k]) <= hi)
    per["b_eff_rank_inside_band"] = inside and n_rows == 2 * len(mine)
    per["b_observer_reproduces_printed_eff_rank"] = printed_ok
    per["b_band_rejects_a_shifted_column"] = shifted_rejected > 0
    extra.update({"eff_rank_cells": n_rows, "eff_rank_outside_band": outside[:10],
                  "shifted_column_rejected": f"{shifted_rejected}/{shifted_n}",
                  "nochange_per_task_sha256": sha(out_e / "per_task.csv"),
                  "committed_per_task_sha256": sha(HOST_REC / "per_task.csv")})
    out_s = SCR / f"nochange4_{M.__name__}"
    erun(M, "nochange", SEEDS, 4, out_s)
    host4 = hrun(SEEDS, 4)
    per["c_short_per_task_identical_to_host_now"] = same_bytes(out_s / "per_task.csv", host4 / "per_task.csv")
    per["c_short_fresh_identical_to_host_now"] = same_bytes(out_s / "fresh_control.csv", host4 / "fresh_control.csv")
    if getattr(M, "_is_real", False):
        keep = OUT / "c51" / "_nochange"
        keep.mkdir(parents=True, exist_ok=True)
        for f in ("per_task.csv", "fresh_control.csv", "l2_task.csv", "provenance.json"):
            shutil.copy2(out_e / f, keep / f)
    return result(per, extra)


NOCHANGE_MUT = [
    ("Mn1: the lambda = 0 path gets coupled L2 1e-2", [('"nochange": ("none", 0.0)', '"nochange": ("coupled", 1e-2)')]),
    ("Mn2: lr 1.0001e-4", [('ACT_ARM, COND, LR = "R", "std", C.LR ', 'ACT_ARM, COND, LR = "R", "std", C.LR * 1.0001 ')]),
    ("Mn3: the Adam update count restarts every task",
     [('        tc = train_task(batches, tc, "train", t)\n', '        tc = train_task(batches, 0, "train", t)\n')]),
]


def s_host_repro() -> None:
    """adamw_dose_1010 2.3 (c): the unmodified host run('R', iv='l2init:1e-2', seeds 0-9) reproduces the committed
    L2 Init record in every per_task column but eff_rank, and its fresh_control.csv byte for byte.  Mutations are
    wrong comparisons: against the l2init 1e-3 record, and against the seeds 10-19 l2init 1e-2 record."""
    torch.set_num_threads(2)
    t0 = time.time()
    out = hrun(SEEDS, C.N_TASKS, iv="l2init:1e-2")

    def compare(cell: str) -> dict:
        ok, diff = same_but_eff_rank(out / "per_task.csv", HOST_DIR / cell / "per_task.csv")
        return result({"per_task_non_eff_rank_identical": ok,
                       "fresh_identical": same_bytes(out / "fresh_control.csv", HOST_DIR / cell / "fresh_control.csv")},
                      {"diff": diff})

    base = compare("R_std_lr0.0001_l2init1e-2")
    keep = OUT / "c51" / "_hostrepro"
    keep.mkdir(parents=True, exist_ok=True)
    for f in ("per_task.csv", "fresh_control.csv", "provenance.json"):
        shutil.copy2(out / f, keep / f)
    entry = {"title": "the L2 Init record reproduces from the unmodified host in this environment",
             "target": "src/cifar5p1_mlp_0920.py", "threads": 2,
             "threshold_derivation": "byte identity of every column but eff_rank (cap_cifar5p1_1007 1.4)",
             **base, "mutations": []}
    for label, cell in (("Mh1: compared with the l2init 1e-3 record", "R_std_lr0.0001_l2init1e-3"),
                        ("Mh2: compared with the seeds 10-19 record", "R_s10-19l2init:1e2")):
        r = compare(cell)
        entry["mutations"].append({"mutation": label, "check_pass_on_mutant": r["pass"], "detected": not r["pass"],
                                   "mutant_result": brief(r)})
    finish("S-host-repro", entry, t0)


def s_coupled_host(M) -> dict:
    """The coupled arms l2c1e-2 and l2c5e-3 on check seeds 100-109, 30 tasks, graph, fresh: per_task.csv byte-identical
    to the unmodified host run('R', iv='l2:<lambda>') on the same seeds; fresh_self.csv's fresh and continual values
    equal the host's fresh_control.csv (the host applies its iv to the fresh net too); the registered
    (mode, lambda) are written here and must match."""
    per = {}
    for arm, (mode, lam) in A2_REG.items():
        per[f"{arm}_registered_spec"] = M.ARM_SPEC[arm] == (mode, lam)
        o = SCR / f"coupled_{arm}_{M.__name__}"
        erun(M, arm, CHECK10, C.N_TASKS, o)
        h = hrun(CHECK10, C.N_TASKS, iv=f"l2:{lam:g}")
        per[f"{arm}_per_task_identical_to_host_l2"] = same_bytes(o / "per_task.csv", h / "per_task.csv")
        fs, fc = read_csv(o / "fresh_self.csv"), read_csv(h / "fresh_control.csv")
        per[f"{arm}_fresh_equals_host"] = len(fs) == len(fc) == 10 and all(
            a["fresh_self_online_acc"] == b["fresh_online_acc"] and a["continual_online_acc"] == b["continual_online_acc"]
            for a, b in zip(fs, fc))
        l2 = read_csv(o / "l2_task.csv")
        per[f"{arm}_l2_counted"] = all(int(r[f"l2_steps_{nm}"]) == int(r["steps"]) for r in l2
                                       for nm in ("W1", "b1", "W2", "b2", "W3", "b3"))
        if getattr(M, "_is_real", False):
            keep = OUT / "c51" / "_coupledhost" / arm
            keep.mkdir(parents=True, exist_ok=True)
            for f in ("per_task.csv", "fresh_self.csv", "l2_task.csv", "provenance.json"):
                shutil.copy2(o / f, keep / f)
    return result(per)


COUPLED_MUT = [
    ("Mc1: lambda theta instead of 2 lambda theta",
     [("            gr = gr.add(p.detach(), alpha=2.0 * lam)\n", "            gr = gr.add(p.detach(), alpha=lam)\n")]),
    ("Mc2: L2 only on the weights (biases skipped)",
     [('        if mode == "coupled":\n', '        if mode == "coupled" and p.dim() == 3:\n')]),
    ("Mc3: the L2 term in m only (v sees the raw gradient)",
     [("        vi.mul_(b2).addcmul_(gr, gr, value=1 - b2)\n",
       "        vi.mul_(b2).addcmul_(grads[i], grads[i], value=1 - b2)\n")]),
]


MUT_LAM = 1e-2
N_OP_MUT = 6


def s_mut(M) -> dict:
    """A2's mutation control (spec 6): seed 100 alone (R = 1), 2 tasks x 780 updates of 'mut_after1e-2' (eager), each
    update's gradients fed in lockstep to three shadows from the same initial values through M.adam_update_:
    'after' (lambda 1e-2: 2 lam theta_old subtracted after Adam's step), 'decoupled' (lambda_W = 2e-2) and 'coupled'
    (lambda 1e-2).  The after and decoupled shadows feed the raw gradient to m and v (so their m, v and Adam terms
    are the same) and differ only in the roundings of the decay and the two subtractions (after: 3, decoupled: 2, the
    rounding of c: 1), so |p_after - p_dec| <= sum_t 6 u (|p_t| + eta |s_t|) element by element (|p_t| the larger of
    the two shadows before and after the update).  (i) after vs decoupled within that band for every element;
    (ii) coupled vs decoupled outside it for >= 99% of the elements (the coupled pull is 2 eta lam |p| / sqrt(v_hat)
    per update, ~500x the decoupled 2 eta lam |p| at sqrt(v_hat) ~ 2e-3).  (iii) report only: 30-task runs (graph)
    of nochange, mut_after1e-2 and dec2e-2 on seed 100 -- late-window online and row norms; kept in c51/_mut/."""
    dv = dev2()
    st = {"k": 0}

    def hook(ev, **s):
        if ev != "grads" or s["phase"] != "train":
            return
        if st["k"] == 0:
            init = [q.detach().clone() for q in s["P"]]
            for nm in ("after", "dec", "cpl"):
                st[nm] = {"P": [q.clone() for q in init], "m": [torch.zeros_like(q) for q in init],
                          "v": [torch.zeros_like(q) for q in init]}
            st["wd_dec"] = torch.ones((), device=dv)
            st["wd_dec"].fill_(M.decay_factor(LR2, 2 * MUT_LAM))
            st["band"] = [torch.zeros(q.shape, dtype=torch.float64, device=dv) for q in init]
        g = [x.detach() for x in s["grads"]]
        before = {nm: [q.clone() for q in st[nm]["P"]] for nm in ("after", "dec")}
        M.adam_update_(st["after"]["P"], g, st["after"]["m"], st["after"]["v"], s["inv_c1"], s["inv_c2"], LR2,
                       st["wd_dec"], "after", MUT_LAM)
        M.adam_update_(st["dec"]["P"], g, st["dec"]["m"], st["dec"]["v"], s["inv_c1"], s["inv_c2"], LR2,
                       st["wd_dec"], "decoupled", 2 * MUT_LAM)
        M.adam_update_(st["cpl"]["P"], g, st["cpl"]["m"], st["cpl"]["v"], s["inv_c1"], s["inv_c2"], LR2,
                       st["wd_dec"], "coupled", MUT_LAM)
        tc = s["tc"]
        c1, c2 = 1 - B1 ** tc, 1 - B2 ** tc
        for i in range(6):
            s_ = ((st["dec"]["m"][i].double() / c1) / ((st["dec"]["v"][i].double() / c2).sqrt() + AEPS)).abs()
            pm = torch.stack([before["after"][i].double().abs(), before["dec"][i].double().abs(),
                              st["after"]["P"][i].double().abs(), st["dec"]["P"][i].double().abs()]).amax(0)
            st["band"][i] += N_OP_MUT * U32 * (pm + LR2 * s_)
        st["k"] += 1

    o = SCR / f"mut_replay_{M.__name__}"
    shutil.rmtree(o, ignore_errors=True)
    M.run("mut_after1e-2", CHECK1, 2, dv, o, cifar=cifar100(), progress=quiet, debug={"hook": hook}, fresh=False)
    n_in = n_all = n_out_c = 0
    worst_in, med_ratio_c = 0.0, []
    for i in range(6):
        d_ad = (st["after"]["P"][i].double() - st["dec"]["P"][i].double()).abs()
        d_cd = (st["cpl"]["P"][i].double() - st["dec"]["P"][i].double()).abs()
        bd = st["band"][i]
        n_all += bd.numel()
        n_in += int((d_ad <= bd).sum())
        worst_in = max(worst_in, float((d_ad / bd.clamp_min(1e-300)).max()))
        n_out_c += int((d_cd > bd).sum())
        med_ratio_c.append(float((d_cd / bd.clamp_min(1e-300)).median()))
    per = {"n_updates": st["k"] == 2 * C.STEPS_PER_TASK,
           "i_after_equals_decoupled_within_band": n_in == n_all,
           "ii_coupled_outside_band_99pct": n_out_c >= 0.99 * n_all,
           "after_and_dec_same_m_v": _equal_list(st["after"]["m"], st["dec"]["m"]) and
           _equal_list(st["after"]["v"], st["dec"]["v"])}
    extra = {"worst_ratio_after_vs_dec": worst_in, "frac_coupled_outside": n_out_c / n_all,
             "median_ratio_coupled_over_band_per_tensor": med_ratio_c}
    if getattr(M, "_is_real", False):
        rep = {}
        keep = OUT / "c51" / "_mut"
        for arm in ("nochange", "mut_after1e-2", "dec2e-2"):
            oo = keep / arm
            erun(M, arm, CHECK1, C.N_TASKS, oo, fresh=False)
            pt = pd.read_csv(oo / "per_task.csv")
            l2 = pd.read_csv(oo / "l2_task.csv")
            late = pt[pt.task.isin([21, 23, 25, 27, 29])]
            rep[arm] = {"late_online": float(late.online_acc.mean()), "early_online":
                        float(pt[pt.task.isin([1, 3, 5, 7, 9])].online_acc.mean()),
                        "w1_row_med_t29": float(l2[l2.task == 29].w1_row_med.iloc[0]),
                        "w2_row_med_t29": float(l2[l2.task == 29].w2_row_med.iloc[0]),
                        "dead_frac_l2_late": float(late.dead_frac_l2.mean())}
        extra["iii_seed100_runs"] = rep
    return result(per, extra)


MUT_MUT = [
    ("Mm1: after's L2 moved back before Adam (= coupled)",
     [("            p_old = p.detach().clone()\n",
       "            gr = gr.add(p.detach(), alpha=2.0 * lam)\n            p_old = torch.zeros_like(p)\n")]),
    ("Mm2: after's L2 doubled", [("            p.sub_(p_old, alpha=lr * 2.0 * lam)\n",
                                   "            p.sub_(p_old, alpha=lr * 4.0 * lam)\n")]),
    ("Mm3: after's L2 dropped", [("            p.sub_(p_old, alpha=lr * 2.0 * lam)\n",
                                  "            p.sub_(p_old, alpha=0.0)\n")]),
]


def s_graph_c51(M) -> dict:
    """l2c1e-2 on seeds 100-109, 4 tasks x 100 updates: graph and eager give byte-identical per_task.csv,
    l2_task.csv and fresh_self.csv and bit-identical final P, m, v; the graph was used; the L2 ops were counted (100)."""
    per = {}
    fa, fb = {}, {}
    oa, ob = SCR / f"graphc51_g_{M.__name__}", SCR / f"graphc51_e_{M.__name__}"
    erun(M, "l2c1e-2", CHECK10, 4, oa, graph=True, steps_hard=100, steps_easy=100, final=fa)
    erun(M, "l2c1e-2", CHECK10, 4, ob, graph=False, steps_hard=100, steps_easy=100, final=fb)
    for f in ("per_task.csv", "l2_task.csv", "fresh_self.csv"):
        per[f"{f}_identical"] = same_bytes(oa / f, ob / f)
    per["final_state_bitwise"] = all(_equal_list(fa[k], fb[k]) for k in ("P", "m", "v"))
    per["graph_was_used"] = "CUDA graph" in json.loads((oa / "provenance.json").read_text())["engine"]
    l2 = read_csv(oa / "l2_task.csv")
    per["l2_counted"] = all(int(r["l2_steps_W1"]) == 100 and int(r["l2_steps_b3"]) == 100 for r in l2)
    return result(per)


GRAPHC51_MUT = [
    ("Mg1: lambda doubled after the capture (eager reads it, the graph keeps the captured one)",
     [("    c_lam = float(c32(lr, lam))\n", "    c_lam = float(c32(lr, lam))\n    lam = lam * 2\n")]),
    ("Mg2: static_idx one batch behind while replaying",
     [("            static_idx.copy_(batches[:, j])",
       "            static_idx.copy_(batches[:, j] if cg is None else batches[:, max(j - 1, 0)])")]),
]


def s_fresh_c51(M) -> dict:
    """Short runs (seeds 100-102, 3 tasks, 200/100 updates): (i) the last hard task's batches have the same sha256 in
    l2c1e-2, l2c5e-3 and the lambda = 0 path; (ii) l2c1e-2 writes fresh_self.csv and no fresh_control.csv, its
    continual column is its own task-3 online accuracy, and the lambda = 0 path's fresh_control.csv is the host's byte
    for byte; (iii) at the start of fresh_self, P equals the initial values and m = v = 0, and the first fresh_self
    update has update count 1; (iv) every fresh_self update carries the L2 op (l2_steps = 200)."""
    per = {}
    outs = {}
    for arm in ("nochange", "l2c1e-2", "l2c5e-3"):
        o = SCR / f"fresh_{arm}_{M.__name__}"
        erun(M, arm, CHECK3, 3, o, steps_hard=200, steps_easy=100)
        outs[arm] = o
    shas = {a: json.loads((o / "provenance.json").read_text())["fresh_batches_sha256"] for a, o in outs.items()}
    per["i_same_last_hard_batches"] = len(set(shas.values())) == 1 and None not in shas.values()
    host = hrun(CHECK3, 3, steps_hard=200, steps_easy=100)
    per["ii_nochange_fresh_is_host"] = same_bytes(outs["nochange"] / "fresh_control.csv", host / "fresh_control.csv")
    per["ii_l2c_writes_fresh_self_only"] = (outs["l2c1e-2"] / "fresh_self.csv").exists() and \
        not (outs["l2c1e-2"] / "fresh_control.csv").exists()
    cont = {r["seed"]: r["online_acc"] for r in read_csv(outs["l2c1e-2"] / "per_task.csv") if r["task"] == "3"}
    fs = read_csv(outs["l2c1e-2"] / "fresh_self.csv")
    per["ii_continual_is_own_t3"] = len(fs) == 3 and all(r["continual_online_acc"] == cont[r["seed"]] for r in fs)
    per["iv_fresh_self_l2_every_update"] = all(int(r[f"l2_steps_{nm}"]) == 200 for r in fs
                                               for nm in ("W1", "b1", "W2", "b2", "W3", "b3"))
    init = [q.detach() for s in CHECK3 for q in C.init_params("R", s, dev2())]
    P0 = [torch.stack(init[i::6]).contiguous() for i in range(6)]
    seen = {}

    def hook(ev, **s):
        if ev == "fresh_self_start":
            ok = _equal_list(P0, [q.detach() for q in s["P"]])
            ok &= all(bool((q == 0).all()) for q in (*s["m"], *s["v"]))
            seen["start"] = ok
        if ev == "post_step" and s["phase"] == "fresh_self" and s["j"] == 0:
            seen["first_tc"] = s["tc"]
    erun(M, "l2c1e-2", CHECK3, 3, SCR / f"freshhook_{M.__name__}", debug={"hook": hook}, steps_hard=200, steps_easy=100)
    per["iii_start_state_initial"] = seen.get("start", False)
    per["iii_first_update_count_1"] = seen.get("first_tc") == 1
    return result(per, {"fresh_batches_sha256": shas})


FRESHC51_MUT = [
    ("Mf1: the Adam moments not zeroed before fresh_self",
     [("                for q in (*adam_m, *adam_v):\n                    q.zero_()\n",
       "                for q in ():\n                    q.zero_()\n")]),
    ("Mf2: fresh_self starts from the end state (the reset skipped)",
     [("        reset_net()\n        if not decay:\n", "        if not decay:\n")]),
]


# --------------------------------------------------------------------------
# S-verdict: the verdict on the recorded cap arms and on synthetic data
# --------------------------------------------------------------------------

def _cap_grid(df, col, s=False):
    g = np.full((10, 50), None, dtype=object) if s else np.full((10, 50), np.nan)
    for _, r in df.iterrows():
        g[int(r.seed), int(r.task) - 1] = r[col] if s else float(r[col])
    return g


def _cap_standins(V) -> dict:
    cap = pd.read_csv(CAP / "per_task.csv")

    def one(capname, lam_arm):
        df = cap[cap.arm == capname]
        meds = {k: np.full((10, 50), np.nan) for k in V.MED_KEYS}
        for t in range(1, 51):
            m = V._medians(V._cap_units(capname, t))
            for k in V.MED_KEYS:
                meds[k][:, t - 1] = m[k]
        d = {"A": _cap_grid(df, "online_acc"), "G": _cap_grid(df, "G2"), "F": _cap_grid(df, "major_frac"),
             "Z": _cap_grid(df, "zero2"), "lh": _cap_grid(df, "label_hash", True),
             "oh": _cap_grid(df, "order_hash", True), "med": meds}
        if lam_arm:
            d["wd_ok"] = np.ones((10, 49), bool)
            d["fline2"] = np.zeros((10, 50))
        return d
    return {"arms": {"ref": one("ref", False), "wd3e-2": one("cap1", True), "wd1e-1": one("cap2", True),
                     "wd1e0": one("cap12", True)}, "s0_ok": True}


def _synth_a1(V, arms: dict, ref_late=None, s0_ok=True, seed=7) -> dict:
    """A synthetic A1 data set.  arms[a] = dict(online=(10,) window level, t2=(10,) task-2 online, G=(10,) level or
    ("rel", (10,) offsets added to the ref's G), Z=(10,), eq=((10,), (10,)) row-norm multipliers of w*,
    kap=((10,), (10,)) width-slope multipliers of the ref's live slope, wd_ok=bool)."""
    rng = np.random.default_rng(seed)
    T = 50
    F = 0.114 + rng.normal(0, 0.006, (10, T))
    A_ref = np.empty((10, T))
    A_ref[:, 0] = 0.983 + rng.normal(0, 0.002, 10)
    A_ref[:, 1] = 0.92 + rng.normal(0, 0.01, 10)
    A_ref[:, 2:] = 0.10 + rng.normal(0, 0.002, (10, T - 2))
    if ref_late is not None:
        A_ref[:, 2:] = np.asarray(ref_late, float)[:, None] + rng.normal(0, 0.002, (10, T - 2))
    G_ref = np.full((10, T), 1e-5) + rng.uniform(0, 1e-6, (10, T))
    G_ref[:, 0] = 0.35 + rng.normal(0, 0.01, 10)
    Z_ref = np.full((10, T), 0.998) + rng.normal(0, 0.0005, (10, T))
    Z_ref[:, 0] = 1e-4
    tt = np.arange(1, T + 1, dtype=float)
    late1 = 190.0 + 1e-4 * (tt - 40)
    late2 = 480.0 + 1e-4 * (tt - 40)
    s1_ref = np.where(tt <= 3, 16.5 + 31.0 * (tt - 1), late1)[None, :] * (1 + rng.normal(0, 0.02, (10, 1)))
    s2_ref = np.where(tt <= 3, 12.8 + 73.0 * (tt - 1), late2)[None, :] * (1 + rng.normal(0, 0.02, (10, 1)))
    lh = np.array([[f"L{t}" for t in range(T)] for _ in range(10)], dtype=object)
    oh = np.array([[f"O{s}_{t}" for t in range(T)] for s in range(10)], dtype=object)

    def meds(s1, s2, w1, w2):
        return {"s1": s1, "s2": s2, "w1": w1, "w2": w2, "rho1": s1 / w1, "rho2": s2 / w2, "r2": np.full_like(s1, -1.9),
                "zbar2": -1.9 * s2}
    data = {"arms": {"ref": {"A": A_ref, "G": G_ref, "F": F, "Z": Z_ref, "lh": lh, "oh": oh,
                             "med": meds(s1_ref, s2_ref, np.full((10, T), 20.0), np.full((10, T), 2.2))}},
            "s0_ok": s0_ok}
    g_live = {l: np.array([np.polyfit([1, 2, 3], (s1_ref if l == 1 else s2_ref)[s, :3], 1)[0] for s in range(10)])
              for l in (1, 2)}
    for a, sp in arms.items():
        lam = V.LAMBDA[a]
        A = A_ref.copy()
        A[:, 1] = np.asarray(sp.get("t2", A_ref[:, 1]), float)
        A[:, 2:] = np.asarray(sp["online"], float)[:, None] + rng.normal(0, 0.001, (10, T - 2))
        G = G_ref.copy()
        if isinstance(sp["G"], tuple):
            G[:, 1:] = G_ref[:, 1:] + np.asarray(sp["G"][1], float)[:, None]
        else:
            G[:, 1:] = np.asarray(sp["G"], float)[:, None] + rng.uniform(0, 1e-4, (10, T - 1))
        Z = Z_ref.copy()
        Z[:, 1:] = np.clip(np.asarray(sp["Z"], float)[:, None] + rng.normal(0, 1e-4, (10, T - 1)), 0, 1)
        e1, e2 = (np.asarray(x, float) for x in sp["eq"])
        w1 = np.repeat((e1 * RULE_W(V, lam, 3072))[:, None], T, 1) * (1 + rng.normal(0, 0.001, (10, T)))
        w2 = np.repeat((e2 * RULE_W(V, lam, 100))[:, None], T, 1) * (1 + rng.normal(0, 0.001, (10, T)))
        k1, k2 = (np.asarray(x, float) for x in sp["kap"])
        s1 = 30.0 + (k1 * g_live[1])[:, None] * (tt - 40)[None, :] + rng.normal(0, 0.01, (10, T))
        s2 = 40.0 + (k2 * g_live[2])[:, None] * (tt - 40)[None, :] + rng.normal(0, 0.01, (10, T))
        data["arms"][a] = {"A": A, "G": G, "F": F.copy(), "Z": Z, "lh": lh.copy(), "oh": oh.copy(),
                           "med": meds(s1, s2, w1, w2), "wd_ok": np.full((10, T - 1), sp.get("wd_ok", True)),
                           "fline2": np.zeros((10, T))}
    return data


def _lab(res: dict, kind: str, arm: str, layer: str):
    """A layer label from the verdict, or None when the verdict did not evaluate it (no crash in the check)."""
    return res.get(kind, {}).get(arm, {}).get(layer, {}).get("label")


def RULE_W(V, lam, d):
    return V.RULE.w_star(lam, d)


def _v(x):
    return np.full(10, x, float) if np.isscalar(x) else np.asarray(x, float)


ONES, HALF_SPLIT = np.ones(10), np.array([0.3] * 5 + [0.10] * 5)


def _a2_frame(late, early, seeds=SEEDS, extra_cols=True):
    rows = []
    for i, s in enumerate(seeds):
        for t in range(1, 31):
            v = late[i] if t in (21, 23, 25, 27, 29) else early[i] if t in (1, 3, 5, 7, 9) else 0.5
            r = {"seed": s, "task": t, "online_acc": v}
            if extra_cols:
                r |= {c: 0.1 for c in ("dead_frac_l2", "zeroout_l2", "mob_l2", "eff_rank_l2", "zbar_l2", "zsd_l2",
                                       "w_norm_l1", "w_norm_l2", "w_norm_l3", "dead_frac_l1", "eff_rank_l1")}
            rows.append(r)
    return pd.DataFrame(rows)


def _synth_a2(arms: dict, gap=0.2, reuse=True, broken=None, drop=None, seed=3) -> dict:
    """arms[a] = (rho, early_shift): the late window W = R + rho (L - R) seed by seed (rho a number or (10,) array)."""
    rng = np.random.default_rng(seed)
    Rl = 0.44 + rng.normal(0, 0.02, 10)
    Re = 0.61 + rng.normal(0, 0.02, 10)
    Ll = 0.66 + rng.normal(0, 0.02, 10)
    data = {"R": {"per_task": _a2_frame(Rl, Re),
                  "fresh_control": pd.DataFrame({"seed": SEEDS, "fresh_online_acc": Rl + gap,
                                                 "fresh_gap": gap + rng.normal(0, 0.01, 10)})},
            "L": {"per_task": _a2_frame(Ll, Re)}, "adamw": {"adamw5": _a2_frame(Rl - 0.08, Re - 0.13),
                                                             "adamw10": _a2_frame(Rl - 0.09, Re - 0.15)},
            "nochange": {"prov": {"fresh_batches_sha256": "x"}},
            "reuse": {k: reuse for k in ("a", "b", "c", "coupled_host")}, "arms": {}}
    for a, (rho, early_shift) in arms.items():
        if drop == a:
            continue
        late = Rl + np.asarray(rho, float) * (Ll - Rl)
        l2 = pd.DataFrame([{"seed": s, "task": t, "steps": 780, **{f"l2_steps_{nm}": (779 if broken == a else 780)
                                                                   for nm in ("W1", "b1", "W2", "b2", "W3", "b3")},
                            **{c: 0.1 for c in ("relpos_l2", "p2", "b1_norm", "b2_norm", "b3_norm", "w1_row_med",
                                                "w2_row_med", "w3_row_med")}} for s in SEEDS for t in range(1, 31)])
        data["arms"][a] = {"per_task": _a2_frame(late, Re + early_shift),
                           "fresh_self": pd.DataFrame({"seed": SEEDS, "fresh_self_online_acc": late + 0.2}),
                           "l2": l2, "prov": {"fresh_batches_sha256": "x", "divergences": []}}
    return data


def s_verdict(V) -> dict:
    """(1) The recorded arms of cap_cifar_ee_0920 put in as stand-ins (cap1 as wd3e-2, cap2 as wd1e-1, cap12 as wd1e0)
    reproduce that run's registered labels (SPLIT, SPLIT, RESCUED), its E1/E2 means (to 1e-12) and floor counts, and
    give RULE_HOLDS; (2) synthetic A1 sets return every designed label: the five labels, IMPAIRED_EARLY, the three
    DEPTH labels, the four RULE labels, the four DOSE labels, the EQUILIBRIUM layer labels and levels, the WIDTH layer
    labels and levels, INCOMPLETE, DIVERGED, CHECK_FAILED, NOT_REPRODUCED, INAPPLICABLE; (3) synthetic A2 sets return
    RESCUED / PARTIAL / NO_RESCUE, WORSE, IMPAIRED, the three readings, the three box labels, INCOMPLETE,
    CHECK_FAILED, INAPPLICABLE, NOT_REPRODUCED; (4) t quantiles match the published table to 5e-7 and the n = 10
    sign-test p values are the exact binomial ones."""
    per, extra = {}, {}
    # (1)
    r = V.analyze_a1(_cap_standins(V))
    cv = json.loads((CAP / "verdict.json").read_text())["arms"]
    per["1_cap_labels"] = (r["labels"]["wd3e-2"], r["labels"]["wd1e-1"], r["labels"]["wd1e0"]) == ("SPLIT", "SPLIT", "RESCUED")
    per["1_cap_endpoints"] = all(abs(r["arms"][a]["E1"]["mean"] - cv[c]["E1"]["mean"]) < 1e-12 and
                                 abs(r["arms"][a]["E2"]["mean"] - cv[c]["E2"]["mean"]) < 1e-12 and
                                 r["arms"][a]["floor_count"] == cv[c]["floor_count"]
                                 for a, c in (("wd3e-2", "cap1"), ("wd1e-1", "cap2"), ("wd1e0", "cap12")))
    per["1_cap_rule_holds"] = r["labels"]["rule"] == "RULE_HOLDS"
    # (2) scenario A
    eq_ok, kap0 = (_v(1.5), _v(1.5)), (_v(0.0), _v(0.0))
    A = V.analyze_a1(_synth_a1(V, {
        "wd3e-2": dict(online=_v(0.10), G=_v(1e-5), Z=_v(0.95), eq=eq_ok, kap=kap0),
        "wd1e-1": dict(online=HALF_SPLIT, G=_v(0.02), Z=_v(0.5), eq=(_v(3.0), _v(3.0)), kap=kap0),
        "wd1e0": dict(online=_v(0.8), G=_v(0.15), Z=_v(0.001), eq=eq_ok, kap=kap0)}))
    L = A["labels"]
    per["2A_labels"] = (L["wd3e-2"], L["wd1e-1"], L["wd1e0"]) == ("COLLAPSED", "SPLIT", "RESCUED")
    per["2A_depth"] = (L["wd3e-2_depth"], L["wd1e-1_depth"], L["wd1e0_depth"]) == ("DEPTH_AT_LINE", "DEPTH_AT_LINE",
                                                                                   "DEPTH_ABOVE_LINE")
    per["2A_rule_holds"] = (L["rule"], L["rule_depth"], L["rule_stands"]) == ("RULE_HOLDS", "RULE_DEPTH_HOLDS", True)
    per["2A_dose_monotone"] = L["dose"] == "DOSE_MONOTONE"
    per["2A_width"] = (L["wd3e-2_width"], L["wd1e-1_width"], L["wd1e0_width"]) == ("WIDTH_FROZEN", "WIDTH_SPLIT",
                                                                                   "WIDTH_STOPPED")
    per["2A_equilibrium"] = (L["wd1e0_equilibrium"], _lab(A, "equilibrium", "wd1e-1", "l1")) == \
        ("EQUILIBRIUM_MATCH", "OFF_HIGH")
    per["2A_no_impairment"] = L["wd1e0_early"] == "NO_EARLY_IMPAIRMENT"
    # scenario B: RFO, AU (ref seed 9 alive, B still holds at 9/10), DEPTH_UNRESOLVED, RULE_PARTIAL, DOSE_PEAKED,
    # EQUILIBRIUM OFF_HIGH / UNRESOLVED, WIDTH GROWING / SHRINKING / UNRESOLVED
    ref_late = np.array([0.10] * 9 + [0.5])
    zspread = np.array([0.0, 0.0, 0.3, 0.0, 0.25, 0.0, 0.2, 0.0, 0.0, 0.05])
    Bs = V.analyze_a1(_synth_b(V, ref_late, zspread, eq_ok, kap0))
    L = Bs["labels"]
    per["2B_rfo_au"] = (L["wd3e-2"], L["wd1e-1"]) == ("RESCUED_FUNCTION_ONLY", "ALIVE_UNRESOLVED")
    per["2B_depth_unresolved"] = L["wd1e0_depth"] == "DEPTH_UNRESOLVED"
    per["2B_rule_main_holds"] = L["rule"] == "RULE_HOLDS"                 # functional: RFO, AU not saved; wd1e0 saved
    per["2B_rule_depth_fails"] = L["rule_depth"] == "RULE_DEPTH_FAILS"     # Z 0.001 at a FAIL call; UNRESOLVED at SAVE
    per["2B_rule_registered_partial"] = L["rule_registered"] == "RULE_PARTIAL"
    per["2B_rule_not_stands"] = L["rule_stands"] is False
    per["2B_dose_peaked"] = L["dose"] == "DOSE_PEAKED"
    per["2B_equilibrium"] = (L["wd3e-2_equilibrium"], _lab(Bs, "equilibrium", "wd3e-2", "l1"),
                             _lab(Bs, "equilibrium", "wd1e0", "l2")) == ("EQUILIBRIUM_OFF", "OFF_HIGH", "UNRESOLVED")
    per["2B_width"] = (L["wd3e-2_width"], _lab(Bs, "width", "wd3e-2", "l1"), _lab(Bs, "width", "wd1e0", "l1"),
                       _lab(Bs, "width", "wd1e0", "l2")) == ("WIDTH_NOT_STOPPED", "GROWING", "SHRINKING", "UNRESOLVED")
    # scenario C: RULE_FAILS, IMPAIRED_EARLY, OFF_LOW
    Cs = V.analyze_a1(_synth_a1(V, {
        "wd3e-2": dict(online=_v(0.8), G=_v(0.15), Z=_v(0.001), eq=eq_ok, kap=kap0),
        "wd1e-1": dict(online=_v(0.85), G=_v(0.15), Z=_v(0.001), eq=(_v(0.3), _v(0.3)), kap=kap0),
        "wd1e0": dict(online=_v(0.09), t2=_v(0.2), G=_v(0.15), Z=_v(0.6), eq=eq_ok, kap=kap0)}))
    L = Cs["labels"]
    per["2C_rule_fails"] = L["rule"] == "RULE_FAILS"
    per["2C_impaired_early"] = L["wd1e0_early"] == "IMPAIRED_EARLY"
    per["2C_off_low"] = _lab(Cs, "equilibrium", "wd1e-1", "l1") == "OFF_LOW"
    # scenario E: DOSE_OTHER (DOWN only below the UP)
    Es = V.analyze_a1(_synth_a1(V, {
        "wd3e-2": dict(online=_v(0.09), G=_v(1e-5), Z=_v(0.99), eq=eq_ok, kap=kap0),
        "wd1e-1": dict(online=_v(0.8), G=_v(0.15), Z=_v(0.001), eq=eq_ok, kap=kap0),
        "wd1e0": dict(online=_v(0.8) + np.array([1, -1] * 5) * 0.01, G=_v(0.15), Z=_v(0.001), eq=eq_ok, kap=kap0)}))
    per["2E_dose_other"] = Es["labels"]["dose"] == "DOSE_OTHER"
    # scenario G: 97.5% vs 95% (RFO), and E1 measured from the common task 1 (RESCUED)
    z = np.array([1.2, -0.4, 0.9, -1.5, 0.3, 0.8, -0.9, 0.2, -1.1, 0.5])
    z = (z - z.mean()) / z.std(ddof=1)
    up = CS.upper_floor(_synth_a1(V, {})["arms"]["ref"]["F"][:, 30:50])[0]
    Gs = V.analyze_a1(_synth_a1(V, {
        "wd3e-2": dict(online=up + 0.02, t2=_v(0.99), G=_v(0.15), Z=_v(0.001), eq=eq_ok, kap=kap0),
        "wd1e-1": dict(online=_v(0.8), G=_v(0.15), Z=_v(0.001), eq=eq_ok, kap=kap0),
        "wd1e0": dict(online=_v(0.8), G=("rel", 0.01 + 0.0129 * z), Z=_v(0.001), eq=eq_ok, kap=kap0)}))
    per["2G_task1_base_rescued"] = Gs["labels"]["wd3e-2"] == "RESCUED"
    per["2G_bonferroni_rfo"] = Gs["labels"]["wd1e0"] == "RESCUED_FUNCTION_ONLY"
    # scenario H: an IMPAIRED_EARLY RESCUED arm is not SAVED; a ratio of 1 lies below P180's band; a width slope of
    # 0.05 x the ref's live slope is above the seed-CV tolerance
    Hs = V.analyze_a1(_synth_a1(V, {
        "wd3e-2": dict(online=_v(0.10), G=_v(1e-5), Z=_v(0.95), eq=eq_ok, kap=kap0),
        "wd1e-1": dict(online=HALF_SPLIT, G=_v(0.02), Z=_v(0.5), eq=(_v(1.0), _v(1.0)), kap=kap0),
        "wd1e0": dict(online=_v(0.8), t2=_v(0.2), G=_v(0.15), Z=_v(0.001), eq=eq_ok, kap=(_v(0.05), _v(0.05)))}))
    L = Hs["labels"]
    per["2H_impaired_not_saved"] = (L["wd1e0"], L["wd1e0_early"], L["rule"], L["rule_registered"]) == \
        ("RESCUED", "IMPAIRED_EARLY", "RULE_PARTIAL", "RULE_HOLDS")
    per["2H_eq_ratio1_off_low"] = _lab(Hs, "equilibrium", "wd1e-1", "l1") == "OFF_LOW"
    per["2H_width_tolerance"] = _lab(Hs, "width", "wd1e0", "l1") == "GROWING"
    # scenario D: DOSE_FLAT
    flat = 0.10 + np.array([1, -1] * 5) * 0.004
    Ds = V.analyze_a1(_synth_a1(V, {a: dict(online=flat[::k], G=_v(1e-5), Z=_v(0.99), eq=eq_ok, kap=kap0)
                                    for a, k in (("wd3e-2", 1), ("wd1e-1", -1), ("wd1e0", 1))}))
    per["2D_dose_flat"] = Ds["labels"]["dose"] == "DOSE_FLAT"
    # applicability
    base = {a: dict(online=_v(0.8), G=_v(0.15), Z=_v(0.001), eq=eq_ok, kap=kap0) for a in V.ARMS1}
    d = _synth_a1(V, base)
    del d["arms"]["wd1e-1"]
    per["2E_incomplete"] = V.analyze_a1(d)["labels"]["wd1e0"] == "INCOMPLETE"
    d = _synth_a1(V, base)
    d["arms"]["wd1e0"]["A"][3, 20] = np.nan
    per["2E_diverged"] = V.analyze_a1(d)["labels"]["wd3e-2"] == "DIVERGED"
    d = _synth_a1(V, base)
    d["arms"]["wd1e0"]["lh"][2, 7] = "other"
    per["2E_hash_check_failed"] = V.analyze_a1(d)["labels"]["wd3e-2"] == "CHECK_FAILED"
    per["2E_s0_check_failed"] = V.analyze_a1(_synth_a1(V, base, s0_ok=False))["labels"]["wd1e0"] == "CHECK_FAILED"
    nr = V.analyze_a1(_synth_a1(V, base, ref_late=np.array([0.10] * 5 + [0.5] * 5)))["labels"]
    per["2E_not_reproduced"] = nr["wd1e0"] == "NOT_REPRODUCED" and nr["rule"] == "RULE_INAPPLICABLE"
    ina = V.analyze_a1(_synth_a1(V, {**base, "wd1e-1": {**base["wd1e-1"], "wd_ok": False}}))["labels"]
    per["2E_inapplicable"] = ina["wd1e-1"] == "INAPPLICABLE" and ina["rule"] == "RULE_INAPPLICABLE" and \
        ina["dose"] == "DOSE_INAPPLICABLE" and ina["wd1e0"] == "RESCUED"
    # (3) A2
    a = V.analyze_a2(_synth_a2({"l2c1e-2": (1.05, 0.0), "l2c5e-3": (0.5, 0.0)}))["labels"]
    per["3a_norm_both"] = (a["l2c1e-2"], a["l2c1e-2_reading"], a["l2c1e-2_reading_registered"], a["l2c5e-3"],
                           a["l2c5e-3_reading"], a["l2c5e-3_reading_registered"], a["box"]) == \
        ("RESCUED", "NORM_MATTERS", "NORM_MATTERS", "PARTIAL", "BOTH", "BOTH", "L2C_RESCUES_RELU")
    b = V.analyze_a2(_synth_a2({"l2c1e-2": (0.6, 0.0), "l2c5e-3": (-0.25, -0.1)}))["labels"]
    per["3b_both_target_worse_impaired"] = (b["l2c1e-2"], b["l2c1e-2_reading"], b["l2c1e-2_reading_registered"],
                                            b["l2c5e-3"], b["l2c5e-3_reading"], b["l2c5e-3_worse"],
                                            b["l2c5e-3_impaired"], b["l2c1e-2_impaired"]) == \
        ("RESCUED", "BOTH", "BOTH", "NO_RESCUE", "TARGET_MATTERS", True, True, False)
    tiny = np.array([1, -1] * 5) * 0.001
    c = V.analyze_a2(_synth_a2({"l2c1e-2": (tiny, 0.0), "l2c5e-3": (tiny, 0.0)}))["labels"]
    per["3c_no_rescue_target"] = (c["l2c1e-2"], c["l2c1e-2_worse"], c["l2c1e-2_reading"], c["box"]) == \
        ("NO_RESCUE", False, "TARGET_MATTERS", "L2C_NO_RESCUE_RELU")
    p_ = V.analyze_a2(_synth_a2({"l2c1e-2": (0.15, 0.0), "l2c5e-3": (0.15, 0.0)}))["labels"]
    per["3d_partial_box"] = (p_["box"], p_["l2c1e-2"], p_["l2c1e-2_reading"]) == \
        ("L2C_PARTIAL_RELU", "PARTIAL", "TARGET_MATTERS")
    u = V.analyze_a2(_synth_a2({"l2c1e-2": (np.array([0.6, 0.9] * 5), 0.0), "l2c5e-3": (0.7, 0.0)}))["labels"]
    per["3f_unresolved_and_both"] = (u["l2c1e-2_reading"], u["l2c5e-3_reading"]) == ("READING_UNRESOLVED", "BOTH")
    per["3e_incomplete"] = V.analyze_a2(_synth_a2({"l2c1e-2": (1.0, 0.0), "l2c5e-3": (1.0, 0.0)},
                                                  drop="l2c5e-3"))["labels"]["box"] == "INCOMPLETE"
    per["3e_check_failed"] = V.analyze_a2(_synth_a2({"l2c1e-2": (1.0, 0.0), "l2c5e-3": (1.0, 0.0)},
                                                    reuse=False))["labels"]["box"] == "CHECK_FAILED"
    per["3e_inapplicable"] = V.analyze_a2(_synth_a2({"l2c1e-2": (1.0, 0.0), "l2c5e-3": (1.0, 0.0)},
                                                    broken="l2c5e-3"))["labels"]["l2c5e-3"] == "INAPPLICABLE"
    per["3e_not_reproduced"] = V.analyze_a2(_synth_a2({"l2c1e-2": (1.0, 0.0), "l2c5e-3": (1.0, 0.0)},
                                                      gap=-0.05))["labels"]["box"] == "NOT_REPRODUCED"
    # (4)
    tq = {(0.975, 9): 2.262157, (1 - 0.025 / 2, 9): 2.685011, (0.95, 19): 1.729133}
    per["4_t_quantiles"] = all(abs(CS.t_quantile(p, df) - v) <= 5e-7 for (p, df), v in tq.items())
    per["4_sign_test"] = (V.sign_test([1] * 10)[2] == 2 / 1024 and V.sign_test([1] * 9 + [-1])[2] == 22 / 1024 and
                          V.sign_test([1] * 8 + [-1] * 2)[2] == 112 / 1024)
    extra["cap_standin_rule"] = r["rule"]
    return result(per, extra)


def _synth_b(V, ref_late, zspread, eq_ok, kap0):
    data = _synth_a1(V, {
        "wd3e-2": dict(online=_v(0.8), G=("rel", np.array([1, -1] * 5) * 1e-6), Z=_v(0.001), eq=(_v(3.0), _v(3.0)),
                       kap=(_v(0.5), _v(0.5))),
        "wd1e-1": dict(online=_v(0.12), G=_v(0.02), Z=_v(0.6), eq=eq_ok, kap=kap0),
        "wd1e0": dict(online=_v(0.8), G=_v(0.15), Z=zspread, eq=(_v(1.5), np.array([1.2, 4.0] * 5)),
                      kap=(_v(-0.5), np.array([0.6, -0.6] * 5)))}, ref_late=ref_late)
    # the AU arm sits just above every seed's floor (by less than the 3-sd floor would add)
    up, _, _ = CS.upper_floor(data["arms"]["ref"]["F"][:, 30:50])
    data["arms"]["wd1e-1"]["A"][:, 2:] = (up + 0.001)[:, None]
    return data


VERDICT_MUT = [
    ("Mv1: the main RULE outcome requires DEPTH too (the registration's conjunction)",
     [('        saved = lab == "RESCUED" and not impaired                    # spec 14.1-14.2: the functional outcome\n',
       '        saved = lab == "RESCUED" and not impaired and above\n')]),
    ("Mv1b: IMPAIRED_EARLY not excluded from SAVED",
     [('        saved = lab == "RESCUED" and not impaired                    # spec 14.1-14.2: the functional outcome\n',
       '        saved = lab == "RESCUED"\n')]),
    ("Mv2: the floor with 3 sd instead of the t quantile",
     [('    upper, fmean, fsd = CS.upper_floor(x["ref"]["F"][:, W0:W1])\n',
       '    fmean = CS.mean64(x["ref"]["F"][:, W0:W1], 1); fsd = x["ref"]["F"][:, W0:W1].std(1, ddof=1)\n'
       '    upper = fmean + 3 * fsd / math.sqrt(20)\n')]),
    ("Mv3: no Bonferroni (95%)", [("LEVEL = 0.975 ", "LEVEL = 0.95 ")]),
    ("Mv4: E1/E2 from each arm's task 2",
     [('        ends[a] = {k: win_mean(x[a][f]) - x[a][f][:, 0] for k, f in (("E1", "A"), ("E2", "G"))}\n',
       '        ends[a] = {k: win_mean(x[a][f]) - x[a][f][:, 1] for k, f in (("E1", "A"), ("E2", "G"))}\n')]),
    ("Mv5: WIDTH's reference slope from the ref's main window", [("LIVE = (0, 3) ", "LIVE = (30, 50) ")]),
    ("Mv6: A2's midpoint = R",
     [("        c_d[a] = np.array([Wl[s] - (Rl[s] + Ll[s]) / 2.0 for s in SEEDS2])\n",
       "        c_d[a] = np.array([Wl[s] - (Rl[s] + Rl[s]) / 2.0 for s in SEEDS2])\n")]),
    ("Mv7: the reading bands at 1/3 and 2/3", [("RHO_LO, RHO_HI = 0.25, 0.75", "RHO_LO, RHO_HI = 1 / 3, 2 / 3")]),
    ("Mv8: a straddling interval read as BOTH",
     [('        else:\n            reading = "READING_UNRESOLVED"\n', '        else:\n            reading = "BOTH"\n')]),
    ("Mv9: EQUILIBRIUM back to x1/2 .. x2", [("EQ_LO, EQ_HI = 1.61 / 1.3, 2.25 * 1.3", "EQ_LO, EQ_HI = 0.5, 2.0")]),
    ("Mv10: WIDTH tolerance fixed at 0.1",
     [("    wtol = {l: float(np.std(g_ref[l], ddof=1) / np.mean(g_ref[l])) for l in (1, 2)}",
       "    wtol = {l: 0.1 for l in (1, 2)}")]),
]


# --------------------------------------------------------------------------
# S-line: the float32 lines on this GPU (spec 1.1 (a), 14.4, 14.9)
# --------------------------------------------------------------------------

def s_line(Rm) -> dict:
    """On this GPU (analysis/adamw_dose2_1011/lines.py, bisection over float32): (i) the largest z with F.elu(z) == -1
    equals rule.ZF_CUDA bit for bit; (ii) the rule's zero line rule.Z0 lies in (the last z whose local derivative is
    exactly 0, the next float32 above it], i.e. Z0 is the derivative's float32 zero to one ulp; (iii) report only:
    the last zero of dy * e^z for dy = 1e-4 and 1e-8.  The raw values are kept in this record."""
    m = load(LINES).measure("cuda")
    per = {"i_forward_line_is_ZF_CUDA": m["zf"] == Rm.ZF_CUDA,
           "ii_Z0_is_the_local_derivative_zero": m["z0_last_zero"] < Rm.Z0 <= m["z0_first_nonzero"]}
    return result(per, {"measured": m})


LINE_MUT = [("Ml1: the CPU forward line taken for the CUDA one",
             [("ZF_CUDA = -16.982107162475586 ", "ZF_CUDA = -16.635532333438686 ")]),
            ("Ml2: the MNIST host's line taken as the zero", [("Z0 = math.log(2.0 ** -150)", "Z0 = math.log(2.0 ** -24)")])]

# --------------------------------------------------------------------------
# S-cli
# --------------------------------------------------------------------------

def s_cli(M) -> dict:
    """(i) rlcifar: the CLI (a subprocess) on check seeds 100-109 with --prefix (the task-1 state saved here), arm
    wd1e0, tasks 2 x 2 epochs, writes per_task.csv and wd_task.csv byte-identical to the in-process run_arm with the
    same arguments; (ii) c51: main(['c51', ...]) for l2c1e-2 on seeds 100-102 (3 tasks, 200/100 updates) equals the
    in-process run byte for byte; (iii) the parser's defaults are the registered ones (A1: 50 tasks x 400 epochs,
    2 threads, seeds 0-9; A2: 30 tasks, lr 1e-4, 780/780 updates, 2 threads, seeds 0-9); (iv) admission refuses a
    registered output for a missing / not-all_pass / stale checks.json, an unregistered arm, check seeds, 30 tasks,
    an unregistered A2 arm, and --prefix at the registered place, and accepts the registered configuration when the
    tree is clean."""
    per, extra = {}, {}
    A1m, A2m = real_a1(), load(A2F)
    pre = prefix_state(A1m, CHECK10)
    pf = SCR / "prefix_100_109.pt"
    torch.save(pre, pf)
    d_cli, d_fn = SCR / f"cli_rl_{M.__name__}", SCR / f"cli_rl_fn_{M.__name__}"
    for d in (d_cli, d_fn):
        shutil.rmtree(d, ignore_errors=True)
    r = subprocess.run([PY, "-m", "src.adamw_dose2_1011_run", "rlcifar", "--arm", "wd1e0", "--seeds", "100-109",
                        "--tasks", "2", "--epochs", "2", "--prefix", str(pf), "--out", str(d_cli)], cwd=REPO,
                       capture_output=True, text=True)
    extra["cli_rc"] = r.returncode
    extra["cli_err"] = r.stderr[-600:]
    A1m.run_arm("wd1e0", d_fn, seeds=CHECK10, tasks=2, epochs=2, prefix=pre, progress=quiet)
    per["i_rlcifar_cli_equals_function"] = r.returncode == 0 and all(same_bytes(d_cli / f, d_fn / f)
                                                                     for f in ("per_task.csv", "wd_task.csv"))
    torch.set_num_threads(2)
    c_cli, c_fn = SCR / f"cli_c51_{M.__name__}", SCR / f"cli_c51_fn_{M.__name__}"
    shutil.rmtree(c_cli, ignore_errors=True)
    M.main(["c51", "--arm", "l2c1e-2", "--seeds", "100-102", "--tasks", "3", "--steps-hard", "200",
            "--steps-easy", "100", "--device", "cuda", "--out", str(c_cli)])
    erun(A2m, "l2c1e-2", CHECK3, 3, c_fn, steps_hard=200, steps_easy=100)
    per["ii_c51_cli_equals_function"] = all(same_bytes(c_cli / f, c_fn / f) for f in ("per_task.csv", "l2_task.csv",
                                                                                     "fresh_self.csv"))
    a = M.build_parser().parse_args(["rlcifar", "--arm", "wd1e0", "--out", "x"])
    per["iii_defaults_rlcifar"] = (a.tasks, a.epochs, a.threads, a.seeds, a.prefix) == (50, 400, 2, "0-9", None) and \
        M.A1_REGISTERED == dict(tasks=50, epochs=400, graph=True, threads=2, seeds=SEEDS) and \
        tuple(M.A1_ARMS) == tuple(A1_REG)
    c = M.build_parser().parse_args(["c51", "--arm", "l2c1e-2", "--out", "x"])
    per["iii_defaults_c51"] = (c.tasks, c.lr, c.threads, c.steps_hard, c.steps_easy, c.seeds) == \
        (30, LR2, 2, 780, 780, "0-9") and set(M.C51_SEEDS) == set(A2_REG)
    fake = SCR / "fake_checks.json"
    fake.write_text(json.dumps({"all_pass": True, **{k: {"pass": True} for k in M.REQUIRED_CHECKS},
                                "source_sha256": {s_: sha(REPO / s_) for s_ in M.SOURCES}}))
    notpass = SCR / "fake_notpass.json"
    notpass.write_text(json.dumps({"all_pass": False, **{k: {"pass": True} for k in M.REQUIRED_CHECKS},
                                   "source_sha256": {s_: sha(REPO / s_) for s_ in M.SOURCES}}))
    stale = SCR / "fake_stale.json"
    stale.write_text(json.dumps({"all_pass": True, **{k: {"pass": True} for k in M.REQUIRED_CHECKS},
                                 "source_sha256": {s_: "0" * 64 for s_ in M.SOURCES}}))
    reg1 = dict(tasks=50, epochs=400, graph=True, threads=2, seeds=SEEDS)
    reg2 = dict(tasks=30, lr=LR2, steps_hard=780, steps_easy=780, fresh=True, graph=True, threads=2)

    def refused(box, arm, seeds, rc, registered, cj, needle) -> bool:
        try:
            M.admit(box, arm, seeds, rc, registered, checks_json=cj)
            return False
        except SystemExit as e:
            return needle in str(e)
    per["iv_refuse_missing_checks"] = refused("rlcifar", "wd1e0", SEEDS, reg1, M.A1_REGISTERED, SCR / "nope.json", "missing")
    per["iv_refuse_not_all_pass"] = refused("rlcifar", "wd1e0", SEEDS, reg1, M.A1_REGISTERED, notpass, "all_pass")
    per["iv_refuse_stale"] = refused("rlcifar", "wd1e0", SEEDS, reg1, M.A1_REGISTERED, stale, "sources changed")
    per["iv_refuse_unregistered_arm"] = refused("rlcifar", "ro1", SEEDS, reg1, M.A1_REGISTERED, fake, "registers")
    per["iv_refuse_check_seeds"] = refused("rlcifar", "wd1e0", CHECK10, {**reg1, "seeds": CHECK10},
                                           M.A1_REGISTERED, fake, "registered configuration")
    per["iv_refuse_30_tasks"] = refused("rlcifar", "wd1e0", SEEDS, {**reg1, "tasks": 30}, M.A1_REGISTERED, fake,
                                        "registered configuration")
    per["iv_refuse_c51_unregistered_arm"] = refused("c51", "dec2e-2", SEEDS, reg2, M.A2.REGISTERED, fake, "registers")
    try:                                   # in-process, so that a mutation of the CLI is what runs
        M.main(["rlcifar", "--arm", "wd1e0", "--prefix", str(pf), "--out", str(OUT / "rlcifar" / "_refused_probe")])
        per["iv_refuse_prefix_at_registered_place"] = False
    except SystemExit as e:
        per["iv_refuse_prefix_at_registered_place"] = "--prefix is for check runs" in str(e)
    shutil.rmtree(OUT / "rlcifar" / "_refused_probe", ignore_errors=True)
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "analysis", "specs"],
                           capture_output=True, text=True).stdout.strip()
    if not dirty:
        try:
            M.admit("rlcifar", "wd1e0", SEEDS, reg1, M.A1_REGISTERED, checks_json=fake)
            per["iv_accept_registered_when_clean"] = True
        except SystemExit:
            per["iv_accept_registered_when_clean"] = False
    extra["tree_clean_at_check"] = not dirty
    return result(per, extra)


CLI_MUT = [
    ("Mcli1: the default horizon is 30 tasks", [('    r.add_argument("--tasks", type=int, default=A1.TASKS)',
                                                  '    r.add_argument("--tasks", type=int, default=30)')]),
    ("Mcli2: admission does not check the A1 arm", [('        if arm not in A1_ARMS or seeds != list(A1.SEEDS):',
                                                     '        if seeds != list(A1.SEEDS):')]),
    ("Mcli3: --prefix allowed at the registered place",
     [('    if args.prefix is not None and _registered_place(out):', '    if False:')]),
]

# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

def _gpu_used_by(pid: int) -> int:
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    tot = 0
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            tot += int(parts[1])
    return tot


def _sample(proc, box: dict) -> None:
    while proc.poll() is None:
        box["vram_mib"] = max(box.get("vram_mib", 0), _gpu_used_by(proc.pid))
        try:
            for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    box["rss_kb"] = max(box.get("rss_kb", 0), int(line.split()[1]))
        except FileNotFoundError:
            pass
        time.sleep(0.5)


def s_cost() -> None:
    """A1: the CLI, wd1e0 on seeds 100-109 from the check prefix, task 2 x 20 epochs (1,500 updates), alone: ms per
    update (wall clock over the task, diagnostics included), the process's GPU memory (nvidia-smi, sampled every
    0.5 s), peak RSS.  A2: the CLI, l2c1e-2 on seeds 100-109, 30 tasks: the same.  The run-time concurrency rule
    (spec 8) uses these peaks.  Gate: both exit 0."""
    t0 = time.time()
    pf = SCR / "prefix_100_109.pt"
    if not pf.exists():
        torch.save(prefix_state(real_a1(), CHECK10), pf)
    d1 = SCR / "cost_rl"
    shutil.rmtree(d1, ignore_errors=True)
    t1 = time.time()
    pr = subprocess.Popen([PY, "-m", "src.adamw_dose2_1011_run", "rlcifar", "--arm", "wd1e0", "--seeds", "100-109",
                           "--tasks", "2", "--epochs", "20", "--prefix", str(pf), "--out", str(d1)], cwd=REPO,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    box1 = {}
    th = threading.Thread(target=_sample, args=(pr, box1))
    th.start()
    err1 = pr.communicate()[1]
    th.join()
    wall1 = time.time() - t1
    d2 = SCR / "cost_c51"
    shutil.rmtree(d2, ignore_errors=True)
    t2 = time.time()
    pr2 = subprocess.Popen([PY, "-m", "src.adamw_dose2_1011_run", "c51", "--arm", "l2c1e-2", "--seeds", "100-109",
                            "--out", str(d2)], cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    box2 = {}
    th2 = threading.Thread(target=_sample, args=(pr2, box2))
    th2.start()
    err2 = pr2.communicate()[1]
    th2.join()
    wall2 = time.time() - t2
    rl = {}
    if pr.returncode == 0:
        pe = json.loads((d1 / "provenance_end.json").read_text())
        rl = {"task_seconds": pe["elapsed_s"], "ms_per_update": 1e3 * pe["elapsed_s"] / 1500,
              "peak_cuda_reserved_mib": pe["peak_cuda_reserved"] / 2 ** 20, "gpu_mib_nvidia_smi": box1.get("vram_mib"),
              "peak_rss_gib": box1.get("rss_kb", 0) / 2 ** 20, "process_wall_s": wall1,
              "projected_arm_minutes": 49 * 30000 * (pe["elapsed_s"] / 1500) / 60}
    c51 = {}
    if pr2.returncode == 0:
        pv = json.loads((d2 / "provenance.json").read_text())
        c51 = {"ms_per_update": pv["step_ms_last_task"], "gpu_mib_nvidia_smi": box2.get("vram_mib"),
               "peak_rss_gib": box2.get("rss_kb", 0) / 2 ** 20, "wall_clock_s": pv["wall_clock_s"], "process_wall_s": wall2}
    ok = pr.returncode == 0 and pr2.returncode == 0
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": "both CLI runs exit 0", "rlcifar": rl, "c51": c51,
                         "rl_returncode": pr.returncode, "c51_returncode": pr2.returncode,
                         "rl_err": err1.decode()[-400:], "c51_err": err2.decode()[-400:],
                         "measured_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "seconds": round(time.time() - t0, 1)}
    dump()
    print(f"S-cost: pass={ok} rlcifar {rl} c51 {c51}", flush=True)


# --------------------------------------------------------------------------

CHECKS = {
    "S0": ("S0 A1's lambda = 0 path reproduces the cap_cifar_ee_0920 ref (tasks 1-10)", s0, S0_MUT, s0.__doc__, A1F, 2),
    "S-wd": ("S-wd decoupled on synthetic tensors (A1)", s_wd, SWD_MUT, s_wd.__doc__, A1F, 2),
    "S-torch": ("S-torch torch.optim.AdamW's formula within the rounding band (A1)", s_torch, STORCH_MUT,
                s_torch.__doc__ + RegBand.__doc__, A1F, 2),
    "S-branch": ("S-branch one prefix, three arms, the first update decays", s_branch, BRANCH_MUT, s_branch.__doc__, A1F, 2),
    "S-ctrl": ("S-ctrl readout-only WD lets the hidden widths grow", s_ctrl, CTRL_MUT, s_ctrl.__doc__, A1F, 2),
    "S-graph": ("S-graph CUDA graph = eager (A1)", s_graph, GRAPH_MUT, s_graph.__doc__, A1F, 2),
    "S-resume": ("S-resume an epoch-boundary resume = no interruption (A1)", s_resume, RESUME_MUT, s_resume.__doc__, A1F, 2),
    "S-nochange": ("S-nochange A2's lambda = 0 path is the host's R", s_nochange, NOCHANGE_MUT, s_nochange.__doc__, A2F, 2),
    "S-coupled-host": ("S-coupled-host the coupled arms are the host's iv='l2' byte for byte", s_coupled_host,
                       COUPLED_MUT, s_coupled_host.__doc__, A2F, 2),
    "S-mut": ("S-mut L2 after Adam = decoupled (A2 mutation control)", s_mut, MUT_MUT, s_mut.__doc__, A2F, 2),
    "S-graph-c51": ("S-graph-c51 CUDA graph = eager (A2)", s_graph_c51, GRAPHC51_MUT, s_graph_c51.__doc__, A2F, 2),
    "S-fresh-c51": ("S-fresh-c51 the fresh controls (A2)", s_fresh_c51, FRESHC51_MUT, s_fresh_c51.__doc__, A2F, 2),
    "S-verdict": ("S-verdict the verdict returns the designed labels", s_verdict, VERDICT_MUT, s_verdict.__doc__, VERDICT, 2),
    "S-cli": ("S-cli the CLI maps, defaults and admits", s_cli, CLI_MUT, s_cli.__doc__, CLI, 2),
    "S-line": ("S-line the float32 lines of F.elu on this GPU", s_line, LINE_MUT, s_line.__doc__, RULEF, 2),
}
ORDER = ("S-verdict", "S-line", "S-wd", "S-torch", "S0", "S-branch", "S-graph", "S-resume", "S-ctrl", "S-nochange",
         "S-host-repro", "S-coupled-host", "S-mut", "S-graph-c51", "S-fresh-c51", "S-cli", "S-cost")


def main() -> None:
    global DUMP_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated check keys; 'S-cost' for the cost probe")
    ap.add_argument("--merge", default="", help="comma separated partial json files of one source state -> checks.json")
    args = ap.parse_args()
    if args.merge:
        merge([Path(x).resolve() for x in args.merge.split(",") if x])
        return
    only = [k for k in args.only.split(",") if k]
    SCR.mkdir(parents=True, exist_ok=True)
    if only:
        DUMP_PATH = SCR / f"checks_partial_{'_'.join(only)}.json"
    RESULTS.update({"run_id": RUN, "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "spec": "specs/spec_adamw_dose2_1011.md", "prereg_commit": "7525f9747a6a2872c30eb380fe3da4809bc6fa95",
                    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                               text=True).stdout.strip(),
                    "source_sha256": {s: sha(REPO / s) for s in SOURCES}, "verdict_sha256": sha(VERDICT),
                    "torch": torch.__version__, "python": sys.version.split()[0]})
    dev()
    RESULTS["environment"] = environment()        # Codex 9.4: the verdict compares the arms' environment with this
    for k in ORDER:
        if only and k not in only:
            continue
        if k == "S-host-repro":
            s_host_repro()
        elif k == "S-cost":
            s_cost()
        else:
            title, fn, mut, deriv, target, threads = CHECKS[k]
            run_check(k, title, fn, mut, deriv, target, threads)
    RESULTS["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    dump()
    print(f"all_pass = {RESULTS['all_pass']}  -> {DUMP_PATH}")


def merge(parts: list[Path]) -> None:
    """Combine partial check files run in parallel on one source state into checks.json: every part must carry the
    same git head, source and verdict hashes as the current tree, no check may appear twice, and every check of
    ORDER must be present."""
    now = {s: sha(REPO / s) for s in SOURCES}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    merged = {"run_id": RUN, "spec": "specs/spec_adamw_dose2_1011.md",
              "prereg_commit": "7525f9747a6a2872c30eb380fe3da4809bc6fa95", "git_head": head, "source_sha256": now,
              "verdict_sha256": sha(VERDICT), "merged_from": [str(p.relative_to(REPO)) for p in parts]}
    envs = []
    for p in parts:
        d = json.loads(p.read_text())
        assert d["source_sha256"] == now and d["git_head"] == head and d["verdict_sha256"] == sha(VERDICT), p
        envs.append(d.get("environment"))
    assert envs and all(e == envs[0] for e in envs) and envs[0], "the parts ran in different environments"
    merged["environment"] = envs[0]
    for p in parts:
        d = json.loads(p.read_text())
        for k, v in d.items():
            if k.startswith("S") and isinstance(v, dict) and "pass" in v:
                assert k not in merged, f"{k} twice"
                merged[k] = v
    missing = [k for k in ORDER if k not in merged]
    assert not missing, f"missing checks: {missing}"
    RESULTS.clear()
    RESULTS.update(merged)
    dump()
    print(f"merged {len(parts)} parts: all_pass = {RESULTS['all_pass']}  -> {DUMP_PATH}")


if __name__ == "__main__":
    main()

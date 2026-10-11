#!/usr/bin/env python3
"""Checks for adamw_dose_1010 (specs/spec_adamw_dose_1010.md section 6).

    cd ~/Projects/claude/wt/adamw_dose_1010
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python \
        analysis/adamw_dose_1010/checks.py [--only S0,S-wd]

Each check runs on the real code and then on every mutation listed with it; a mutation is an exact-once string
substitution in the check's target file, and it MUST make the same check fail.  Tolerances come from the
arithmetic in each docstring (bit comparisons have none).  Writes results/adamw_dose_1010/checks.json after every
check (with --only: results/_checks_adamw_dose_1010/checks_partial.json).  Box 1 checks run on the CPU with one
torch thread, box 2 checks on CUDA with two (the host's setting).  Check runs use the check seeds (box 1: 100-105,
box 2: 100-109), short runs and synthetic inputs; the exceptions are S0, S-nochange and S-host-repro, which run the
lambda = 0 path and the unmodified host on registered seeds (spec 6).

Box 1: S0, S-wd, S-torch, S-deadpix, S5, S-ctrl, S7.  Box 2: S-wd-c51, S-torch-c51, S-nochange, S-host-repro,
S-graph, S-fresh.  Both: S8 (verdict), S9 (CLI), S-cost.
"""
from __future__ import annotations

import argparse
import csv
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch

from src import pmnist_0905 as H                 # noqa: E402
from src import pmnist_rlmnist_0906 as RL        # noqa: E402
from src import rlcifar_mlp_battle_0918 as B     # noqa: E402
from src import cifar5p1_mlp_0920 as C           # noqa: E402

RUN = "adamw_dose_1010"
RUNNER = REPO / "src" / "adamw_dose_1010.py"
C51 = REPO / "src" / "adamw_dose_1010_c51.py"
CLI = REPO / "src" / "adamw_dose_1010_run.py"
VERDICT = REPO / "analysis" / RUN / "verdict.py"
OUT = REPO / "results" / RUN
SCR = REPO / "results" / f"_checks_{RUN}"
DUMP_PATH = OUT / "checks.json"
L2_RECORD = REPO / "results" / "l2cap_ee_0917"
L2_MANIFEST = L2_RECORD / "backup_manifest.json"
HOST_DIR = REPO / "results" / "cifar5p1_mlp_0920"
HOST_REC = HOST_DIR / "R_std_lr0.0001"
SOURCES = ("src/adamw_dose_1010.py", "src/adamw_dose_1010_c51.py", "src/adamw_dose_1010_run.py",
           "analysis/adamw_dose_1010/checks.py")
PY = sys.executable
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
U32, U64 = 2.0 ** -24, 2.0 ** -53
EPS64 = float(np.finfo(np.float64).eps)
PRINT_REL = 5e-10
# the registered values (spec 2.2 / 2.3), written here, not imported
LR1, LR2 = 1e-3, 1e-4
ARM_MAP_REG = {"ref": 0.0, "wd1e-3": 1e-3, "wd1e-2": 1e-2, "wd3e-2": 3e-2, "wd1e-1": 1e-1}
C51_MAP_REG = {"nochange": 0.0, "adamw5": 5.0, "adamw10": 10.0}
RO_REG = (0.0, 0.0, 0.0, 0.0, 1e-1, 1e-1)
TASKS_RUN, EPOCHS_RUN, STEPS_RUN = 100, 80, 6000
EP = 2                    # 150 updates per task in the short box 1 runs
SPT_EP = 75 * EP
CHECK_SEED = 100
SEEDS2 = list(range(10))
CHECK3 = [100, 101, 102]
CHECK10 = list(range(100, 110))
B1, B2, AEPS = 0.9, 0.999, 1e-8
RESULTS: dict = {}

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


def run_check(key: str, title: str, fn, mutations: list, derivation: str, target: Path, threads: int = 1) -> None:
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
            raise                     # a broken anchor is a bug in the check, not a detection
        except Exception as e:        # recorded; a crash is not how a check should notice a defect
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
    finish(key, entry, t0)


def result(per: dict, extra: dict | None = None) -> dict:
    failed = [k for k, v in per.items() if isinstance(v, bool) and not v]
    return {"pass": not failed, "failed_items": failed, "detail": per, **(extra or {})}


class flush_denormal:
    """The box 1 CLI sets torch.set_flush_denormal(True); in-process runs here do the same."""
    def __enter__(self):
        torch.set_flush_denormal(True)

    def __exit__(self, *a):
        torch.set_flush_denormal(False)


def _flush_is_on() -> bool:
    return float(torch.tensor([1e-30], dtype=torch.float32) * torch.tensor([1e-10], dtype=torch.float32)) == 0.0


def _equal_list(a, b) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return float("nan")


DEV1 = H.setup("cpu")
MNIST = H.Mnist(DEV1)
_DEV2 = [None]
_CIFAR = [None]


def dev2():
    if _DEV2[0] is None:
        _DEV2[0] = H.setup(os.environ.get("CHECK_DEVICE", "cuda"))
    return _DEV2[0]


def cifar():
    if _CIFAR[0] is None:
        _CIFAR[0] = C.Cifar100()
    return _CIFAR[0]


def quiet(_m):
    pass


def run1(M, arm_or_lams, seed=CHECK_SEED, tasks=2, epochs=EP, debug=None):
    lams = M.arm_lams(arm_or_lams) if isinstance(arm_or_lams, str) else tuple(arm_or_lams)
    arm = arm_or_lams if isinstance(arm_or_lams, str) else "custom"
    with flush_denormal():
        return M.run_adamw(seed, LR1, tasks, MNIST, DEV1, lams=lams, arm=arm, epochs=epochs, debug=debug)


def ends_idx(arrays: dict, spt: int) -> dict:
    t, s = arrays["task"], arrays["step"]
    return {int(t[i]): int(i) for i in np.where(s == spt)[0]}


# --------------------------------------------------------------------------
# S0: the lambda = 0 path reproduces the recorded l2cap ref shards
# --------------------------------------------------------------------------

S0_SEEDS = (0, 5)
S0_TASKS = 2


def _archived_l2(shard: str, name: str) -> Path | None:
    man = json.loads(L2_MANIFEST.read_text())
    for f in man["files"]:
        if f["source_rel"] == f"results/l2cap_ee_0917/runs/{shard}/{name}":
            p = Path(f["backup"])
            return p if p.exists() and sha(p) == f["sha256"] else None
    return None


def s0(M) -> dict:
    """run_adamw with the arm 'ref' (lambda 0 on every tensor), flush on, 80 epochs, tasks 1-2, reproduces the
    recorded l2cap ref shards of seeds 0 and 5: every per_task.csv column of the record equal as written and no
    extra column, every units.npz array bit-equal on the task 1-2 rows (test arrays: the task-1 row) and no extra
    array (the archived copy only if its sha256 matches the manifest), the task-1-end state hash, and the init and
    subset hashes.  String and bit comparisons only."""
    failed, per = [], {}
    for seed in S0_SEEDS:
        shard = f"ref_s{seed}"
        rows, arrays, wd, info = run1(M, "ref", seed=seed, tasks=S0_TASKS, epochs=80)
        rec = [r for r in csv.DictReader((L2_RECORD / "runs" / shard / "per_task.csv").open())
               if int(r["task"]) <= S0_TASKS]
        prov = json.loads((L2_RECORD / "runs" / shard / "provenance.json").read_text())
        bad_cols, n_cols = [], 0
        for rr, got in zip(rec, rows):
            for k, v in rr.items():
                if k == "arm":
                    continue
                n_cols += 1
                g = got.get(k)
                if not ((str(g) == v) or (g is not None and v != "" and float(v) == float(g))):
                    bad_cols.append(f"t{rr['task']}:{k}")
        extra_cols = sorted(set(rows[0]) - set(rec[0])) if rows and rec else ["?"]
        arr_bad, n_arr, extra_arr = [], 0, []
        u_path = _archived_l2(shard, "units.npz")
        if u_path is not None:
            with np.load(u_path) as z:
                pre = f"s{seed}_"
                keep = z[pre + "task"] <= S0_TASKS
                rec_keys = {k[len(pre):] for k in z.files}
                extra_arr = sorted(set(arrays) - rec_keys)
                for k in z.files:
                    kk = k[len(pre):]
                    got = arrays.get(kk)
                    if got is None:
                        arr_bad.append(f"missing:{kk}")
                        continue
                    rec_a = z[k] if kk in ("q_cap", "v_cap") else (z[k][:got.shape[0]] if kk.startswith("test_")
                                                                    else z[k][keep])
                    n_arr += 1
                    if not np.array_equal(rec_a, got, equal_nan=True):
                        arr_bad.append(kk)
        ok = {"i_per_task_columns": not bad_cols and n_cols >= 2 * 48 and len(rows) == S0_TASKS,
              "i_no_extra_columns": not extra_cols,
              "ii_arrays": u_path is not None and not arr_bad and n_arr >= 80,
              "ii_no_extra_arrays": not extra_arr,
              "iii_task1_state": info["task1_end_state_sha256"] == prov["per_seed"][str(seed)]["task1_end_state_sha256"],
              "iv_init_subset": all(info[k] == prov["per_seed"][str(seed)][k] for k in ("init_sha256", "subset_idx_sha256")),
              "v_no_decay_recorded": all(w[f"wd_steps_{nm}"] == 0 for w in wd for nm in M.TENSORS)}
        per[shard] = {**ok, "bad_columns": bad_cols[:10], "n_columns": n_cols, "extra_columns": extra_cols[:10],
                      "bad_arrays": arr_bad[:10], "n_arrays": n_arr, "extra_arrays": extra_arr[:10],
                      "archive": str(u_path)}
        failed += [f"{shard}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S0_MUT = [
    ("M0a: ref carries wd1e-3's decay (c != 1)", [('ARM_LAMBDA = {"ref": 0.0,', 'ARM_LAMBDA = {"ref": 1e-3,')]),
    ("M0b: Adam eps 1e-7", [("b1, b2, eps = 0.9, 0.999, 1e-8", "b1, b2, eps = 0.9, 0.999, 1e-7")]),
    ("M0c: the task 2-10 diagnostic points shifted to tasks 3-10",
     [("        dense = set(DENSE) if 2 <= t <= 10 else {0}\n", "        dense = set(DENSE) if 3 <= t <= 10 else {0}\n")]),
]

# --------------------------------------------------------------------------
# S-wd: decoupled, on synthetic tensors (both boxes)
# --------------------------------------------------------------------------

K_WD = 40


def _np_iter(a: np.ndarray, c: np.float32, k: int) -> np.ndarray:
    a = a.astype(np.float32).copy()
    for _ in range(k):
        a *= c
    return a


def _host_adam_ref(p, g, m, v, tc, lr, c_mul=None):
    """The independent reference, written here: optional p.mul_(c) first, then the host's Adam op for op."""
    c1, c2 = 1 - B1 ** tc, 1 - B2 ** tc
    if c_mul is not None:
        p.mul_(c_mul)
    m.mul_(B1).add_(g, alpha=1 - B1)
    v.mul_(B2).addcmul_(g, g, value=1 - B2)
    p -= lr * (m / c1) / ((v / c2).sqrt() + AEPS)


def s_wd(M) -> dict:
    """Box 1, synthetic tensors of the box's six shapes (U(+-1/sqrt(fan_in))), lambdas from the arm 'wd1e-1'
    (registered 1e-1 on all six), lr 1e-3, K = 40 steps of random gradients in which a fixed mask of entries
    (one third of each tensor) gets an exactly-0 gradient every step.  (i) the 0-gradient, 0-moment entries equal
    np.float32 iteration p <- p * c, c = np.float32(1 - 1e-3 * 1e-1), after step 1 and after step K, in every
    tensor; (ii) the whole update equals the independent 'p.mul_(c), then the host's Adam' after K steps, bit for
    bit, and the update count is K; (iii) coupled L2 (lambda theta added to the gradient, then Adam), computed here,
    moves the (i) entries on step 1 by Adam's exact first step eta g / (|g| + eps), g = lambda theta (within 1e-3 eta:
    float32 rounding is ~1e-7 relative), which is eta sign(theta) to 1% wherever |g| >= 100 eps (within 0.011 eta),
    and so differs from (i): the check tells coupled from decoupled.  Bit comparisons except (iii)."""
    g0 = torch.Generator().manual_seed(1010)
    shapes = [(100, 784), (100,), (100, 100), (100,), (10, 100), (10,)]
    fan = [784, 784, 100, 100, 100, 100]
    P = [((torch.rand(s, generator=g0) * 2 - 1) / math.sqrt(f)) for s, f in zip(shapes, fan)]
    masks = [torch.rand(s, generator=g0) < 1 / 3 for s in shapes]
    gseq = [[torch.where(mk, torch.zeros(s), torch.randn(s, generator=g0) * 1e-2) for s, mk in zip(shapes, masks)]
            for _ in range(K_WD)]
    lams = M.arm_lams("wd1e-1")
    c = np.float32(1 - LR1 * 1e-1)
    per = {}
    Pm = [q.clone() for q in P]
    adam = ([torch.zeros_like(q) for q in P], [torch.zeros_like(q) for q in P], [0])
    Pr = [q.clone() for q in P]
    mr, vr = [torch.zeros_like(q) for q in P], [torch.zeros_like(q) for q in P]
    ok1, okK, ok_ref = True, True, True
    nd = [0] * 6
    with flush_denormal():
        for k in range(K_WD):
            M.adamw_step_(Pm, gseq[k], adam, LR1, lams, nd)
            for i in range(6):
                _host_adam_ref(Pr[i], gseq[k][i], mr[i], vr[i], k + 1, LR1, c_mul=1 - LR1 * 1e-1)
            if k == 0:
                for i in range(6):
                    exp = _np_iter(P[i][masks[i]].numpy(), c, 1)
                    ok1 &= bool(np.array_equal(Pm[i][masks[i]].numpy(), exp))
        for i in range(6):
            exp = _np_iter(P[i][masks[i]].numpy(), c, K_WD)
            okK &= bool(np.array_equal(Pm[i][masks[i]].numpy(), exp))
        ok_ref = _equal_list(Pm, Pr) and _equal_list(adam[0], mr) and _equal_list(adam[1], vr)
        # (iii) coupled L2, here
        Pc = [q.clone() for q in P]
        mc, vc = [torch.zeros_like(q) for q in P], [torch.zeros_like(q) for q in P]
        for i in range(6):
            gc = gseq[0][i].add(Pc[i], alpha=1e-1)
            _host_adam_ref(Pc[i], gc, mc[i], vc[i], 1, LR1)
        dev, dev_sign = 0.0, 0.0
        for i in range(6):
            th = P[i][masks[i]].double()
            g = 1e-1 * th
            d_ = (Pc[i] - P[i])[masks[i]].double()
            dev = max(dev, float((d_ + LR1 * g / (g.abs() + AEPS)).abs().max()))
            big = g.abs() >= 100 * AEPS
            dev_sign = max(dev_sign, float((d_[big] + LR1 * torch.sign(th[big])).abs().max()))
        differs = all(not torch.equal(Pc[i][masks[i]], torch.from_numpy(_np_iter(P[i][masks[i]].numpy(), c, 1)))
                      for i in range(6))
    per["i_zero_grad_one_step_is_c"] = ok1
    per["i_zero_grad_K_steps_is_c_iterated"] = okK
    per["ii_update_is_decay_then_host_adam"] = ok_ref
    per["ii_update_count"] = adam[2][0] == K_WD and nd == [K_WD] * 6
    per["iii_coupled_is_adam_first_step"] = dev <= 1e-3 * LR1
    per["iii_coupled_moves_eta_sign"] = dev_sign <= 0.011 * LR1
    per["iii_coupled_differs"] = differs
    per["n_decays"] = nd
    return result(per, {"coupled_max_dev_from_first_step": dev, "coupled_max_dev_from_eta_sign": dev_sign,
                        "c_hex": float(c).hex()})


SWD_MUT = [
    ("Mw1: coupled L2 (lambda theta added to the gradient)",
     [("            p.mul_(1 - lr * lam)\n", "            gr = gr.add(p, alpha=lam)\n")]),
    ("Mw2: the decay after the Adam step",
     [("            p.mul_(1 - lr * lam)\n", "            pass\n"),
      ("        p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)\n",
       "        p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)\n        if lam != 0:\n            p.mul_(1 - lr * lam)\n")]),
    ("Mw3: biases not decayed", [("        return (lam, lam, lam, lam, lam, lam)\n", "        return (lam, 0.0, lam, 0.0, lam, 0.0)\n")]),
    ("Mw4: W3 only", [("        return (lam, lam, lam, lam, lam, lam)\n", "        return (0.0, 0.0, 0.0, 0.0, lam, 0.0)\n")]),
    ("Mw5: c = 1 - lambda (the lr forgotten)", [("            p.mul_(1 - lr * lam)\n", "            p.mul_(1 - lam)\n")]),
    ("Mw6: lambda doubled", [("            p.mul_(1 - lr * lam)\n", "            p.mul_(1 - lr * lam * 2)\n")]),
]


def s_wd_c51(M) -> dict:
    """Box 2, the same on CUDA with the stacked shapes (R = 2), lambda = the arm 'adamw10' (registered 10), lr 1e-4,
    wd_c filled with M.decay_factor(lr, lambda) into a float32 0-dim tensor as run() does, the host's inv_c1/inv_c2
    filled per step from the python double: (i) the 0-gradient entries equal np.float32 iteration with
    c = np.float32(1 - 1e-4 * 10) after 1 and K steps; (ii) the update equals the independent 'p * c, then the host's
    stacked Adam' bit for bit, and the device counters read K; (iii) coupled L2 here moves the (i) entries by Adam's
    exact first step eta g / (|g| + eps), g = lambda theta (eta sign(theta) to 1% where |g| >= 100 eps) and differs
    from (i)."""
    dv = dev2()
    g0 = torch.Generator().manual_seed(1011)
    shapes = [(2, 100, 3072), (2, 100), (2, 100, 100), (2, 100), (2, 100, 100), (2, 100)]
    fan = [3072, 3072, 100, 100, 100, 100]
    P = [((torch.rand(s, generator=g0) * 2 - 1) / math.sqrt(f)) for s, f in zip(shapes, fan)]
    masks = [torch.rand(s, generator=g0) < 1 / 3 for s in shapes]
    gseq = [[torch.where(mk, torch.zeros(s), torch.randn(s, generator=g0) * 1e-2).to(dv) for s, mk in zip(shapes, masks)]
            for _ in range(K_WD)]
    lam = M.ARM_LAMBDA["adamw10"]
    c = np.float32(1 - LR2 * 10.0)
    wd_c = torch.ones((), device=dv)
    wd_c.fill_(M.decay_factor(LR2, lam))
    decay = lam != 0
    Pm = [q.clone().to(dv) for q in P]
    m_, v_ = [torch.zeros_like(q) for q in Pm], [torch.zeros_like(q) for q in Pm]
    Pr = [q.clone().to(dv) for q in P]
    mr, vr = [torch.zeros_like(q) for q in Pm], [torch.zeros_like(q) for q in Pm]
    cnt = torch.zeros(6, dtype=torch.long, device=dv)
    ic1, ic2 = torch.zeros((), device=dv), torch.zeros((), device=dv)
    ok1 = okK = True
    for k in range(K_WD):
        tc = k + 1
        ic1.fill_(1.0 / (1 - B1 ** tc))
        ic2.fill_(1.0 / (1 - B2 ** tc))
        M.adam_update_(Pm, gseq[k], m_, v_, ic1, ic2, LR2, wd_c, decay, cnt)
        for i in range(6):
            Pr[i].mul_(float(c))
            mr[i].mul_(B1).add_(gseq[k][i], alpha=1 - B1)
            vr[i].mul_(B2).addcmul_(gseq[k][i], gseq[k][i], value=1 - B2)
            Pr[i].sub_(LR2 * (mr[i] * ic1) / ((vr[i] * ic2).sqrt() + AEPS))
        if k == 0:
            for i in range(6):
                ok1 &= bool(np.array_equal(Pm[i].cpu()[masks[i]].numpy(), _np_iter(P[i][masks[i]].numpy(), c, 1)))
    for i in range(6):
        okK &= bool(np.array_equal(Pm[i].cpu()[masks[i]].numpy(), _np_iter(P[i][masks[i]].numpy(), c, K_WD)))
    Pc = [q.clone().to(dv) for q in P]
    ic1.fill_(1.0 / (1 - B1))
    ic2.fill_(1.0 / (1 - B2))
    dev_, dev_sign, differs = 0.0, 0.0, True
    for i in range(6):
        gc = gseq[0][i].add(Pc[i], alpha=10.0)
        mc = (1 - B1) * gc
        vc = (1 - B2) * gc * gc
        Pc[i] = Pc[i] - LR2 * (mc * ic1) / ((vc * ic2).sqrt() + AEPS)
        th = P[i][masks[i]].double()
        g = 10.0 * th
        d = (Pc[i].cpu() - P[i])[masks[i]].double()
        dev_ = max(dev_, float((d + LR2 * g / (g.abs() + AEPS)).abs().max()))
        big = g.abs() >= 100 * AEPS
        dev_sign = max(dev_sign, float((d[big] + LR2 * torch.sign(th[big])).abs().max()))
        differs &= not np.array_equal(Pc[i].cpu()[masks[i]].numpy(), _np_iter(P[i][masks[i]].numpy(), c, 1))
    per = {"i_zero_grad_one_step_is_c": ok1, "i_zero_grad_K_steps_is_c_iterated": okK,
           "ii_update_is_decay_then_host_adam": _equal_list(Pm, Pr) and _equal_list(m_, mr) and _equal_list(v_, vr),
           "ii_counters": cnt.cpu().tolist() == [K_WD] * 6,
           "iii_coupled_is_adam_first_step": dev_ <= 1e-3 * LR2, "iii_coupled_moves_eta_sign": dev_sign <= 0.011 * LR2,
           "iii_coupled_differs": differs}
    return result(per, {"coupled_max_dev_from_first_step": dev_, "coupled_max_dev_from_eta_sign": dev_sign,
                        "c_hex": float(c).hex()})


SWD2_MUT = [
    ("Mw1: coupled L2", [("            p.mul_(wd_c)\n", "            gr = gr.add(p.detach() * (1 - wd_c) / lr)\n")]),
    ("Mw2: the decay after the Adam step",
     [("            p.mul_(wd_c)\n", "            pass\n"),
      ("        p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))\n",
       "        p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))\n        if decay:\n            p.mul_(wd_c)\n")]),
    ("Mw3: biases not decayed", [("        if decay:\n            p.mul_(wd_c)\n", "        if decay and p.dim() == 3:\n            p.mul_(wd_c)\n")]),
    ("Mw4: W3 only", [("        if decay:\n            p.mul_(wd_c)\n", "        if decay and i == 4:\n            p.mul_(wd_c)\n")]),
    ("Mw5: c = 1 - lambda", [("    return 1 - lr * lam\n", "    return 1 - lam\n")]),
    ("Mw6: lambda doubled", [("    return 1 - lr * lam\n", "    return 1 - lr * lam * 2\n")]),
]

# --------------------------------------------------------------------------
# S-torch: the same formula as torch.optim.AdamW, within the rounding band (both boxes)
# --------------------------------------------------------------------------

N_OP_REG = 21      # spec 6: the roundings of one update's Adam term, both codes: host m 2, v 2, /c1, /c2, sqrt, +eps,
                   # /, *lr (10); torch lerp 3, v 2, sqrt, /bc2, +eps, /, *step_size (10); the float32 step_size (1)
N_OP = 16          # roundings of the step's last stage, both implementations together (host 6, torch 6), plus 4
M_ROUND = 3        # roundings per update of the first moment, per implementation (mul+add or lerp)


class RegBand:
    """The registered band (spec 6): |dp_K| <= sum_t [n_op u eta |s_t| + 2 u |p_t|], n_op = N_OP_REG, u = 2^-24, with
    the measured |s_t| = |m_hat / (sqrt(v_hat) + eps)| of our code and |p_t| = the larger of the two codes', element
    by element."""

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


class Band:
    """A wider bound, reported next to the registered one: |p_mine - p_torch| after K updates fed the same gradients

        B_K = sum_t [ eta (2 E_t / c1_t) / (sqrt(v_t / c2_t) + eps)        the first moments' rounding difference
                      + N_OP u eta S_t                                       the step's last stage, both codes
                      + 4 u max|p| ]                                         the decay product and the subtraction,
                                                                             each rounded once per code
    with S_t = (M_t / c1_t) / (sqrt(v_t / c2_t) + eps), M_t = beta1 M_{t-1} + (1 - beta1)|g_t| (>= |m_t|: the bound
    holds under cancellation), E_t = beta1 E_{t-1} + M_ROUND u (M_t + |g_t|) per code.  v is the same in both codes
    (the same ops), which the check asserts bit for bit at every update.  First order in u; float64."""

    def __init__(self, shapes, device):
        z = lambda: [torch.zeros(s, dtype=torch.float64, device=device) for s in shapes]
        self.M, self.E, self.B = z(), z(), z()
        self.step_bands = []

    def add(self, grads, v_after, tc: int, lr: float, p_pairs) -> None:
        c1, c2 = 1 - B1 ** tc, 1 - B2 ** tc
        tot = 0.0
        for i, g in enumerate(grads):
            ga = g.double().abs()
            self.M[i] = B1 * self.M[i] + (1 - B1) * ga
            self.E[i] = B1 * self.E[i] + M_ROUND * U32 * (self.M[i] + ga)
            den = (v_after[i].double() / c2).sqrt() + AEPS
            S = (self.M[i] / c1) / den
            pm = torch.stack([q.double().abs() for q in p_pairs[i]]).amax(0)
            inc = lr * (2 * self.E[i] / c1) / den + N_OP * U32 * lr * S + 4 * U32 * pm
            self.B[i] += inc
            tot = max(tot, float(inc.median()))
        self.step_bands.append(tot)


def _band_compare(Pm, Pt, band: Band) -> tuple[bool, float, int]:
    worst, n_out = 0.0, 0
    for a, b, bd in zip(Pm, Pt, band.B):
        d = (a.double() - b.double()).abs()
        r = d / bd.clamp_min(1e-300)
        worst = max(worst, float(r.max()))
        n_out += int((d > bd).sum())
    return n_out == 0, worst, n_out


def s_torch(M) -> dict:
    """Box 1: seed 100, the arm wd1e-1, 2 tasks x 2 epochs (300 updates) with every gradient captured.  (a) replaying
    M.adamw_step_ on the captured gradients from the initial values reproduces the run's final parameters bit for
    bit (the function tested is the one the run used); (b) torch.optim.AdamW(lr 1e-3, betas (0.9, 0.999), eps 1e-8,
    weight_decay = the registered 0.1, foreach=False) fed the same gradients stays within Band elementwise at the
    end, its second moments equal ours at every update; (c) torch.optim.Adam(weight_decay = 0.1) (coupled) leaves
    the band.  The arithmetic of the mutation 'decay after Adam' (eta * eta * lambda * |s| per update) against the
    band per update is reported (median over elements of one update's band, and 1e-7 * median |s|)."""
    d = {"grad_seq": [], "end_capture": (2,)}
    rows, arrays, wd, info = run1(M, "wd1e-1", debug=d)
    init = d["init"]
    lams = M.arm_lams("wd1e-1")
    Pm = [q.clone() for q in init]
    adam = ([torch.zeros_like(q) for q in init], [torch.zeros_like(q) for q in init], [0])
    Pt = [q.clone().requires_grad_(False) for q in init]
    Pc = [q.clone() for q in init]
    opt_t = torch.optim.AdamW([torch.nn.Parameter(q) for q in Pt], lr=LR1, betas=(B1, B2), eps=AEPS,
                              weight_decay=1e-1, foreach=False)
    opt_c = torch.optim.Adam([torch.nn.Parameter(q) for q in Pc], lr=LR1, betas=(B1, B2), eps=AEPS,
                             weight_decay=1e-1, foreach=False)
    pt_list = opt_t.param_groups[0]["params"]
    pc_list = opt_c.param_groups[0]["params"]
    band = Band([q.shape for q in init], "cpu")
    rband = RegBand([q.shape for q in init], "cpu")
    v_same, s_med = True, []
    s_abs = [torch.zeros(q.shape, dtype=torch.float64) for q in init]
    with flush_denormal():
        for k, g in enumerate(d["grad_seq"]):
            before = [q.clone() for q in Pm]
            before_t = [q.detach().clone() for q in pt_list]
            M.adamw_step_(Pm, g, adam, LR1, lams)
            for q, gg in zip(pt_list, g):
                q.grad = gg.clone()
            opt_t.step()
            for q, gg in zip(pc_list, g):
                q.grad = gg.clone()
            opt_c.step()
            vt = [opt_t.state[q]["exp_avg_sq"] for q in pt_list]
            v_same &= _equal_list(adam[1], vt)
            band.add(g, adam[1], k + 1, LR1,
                     [(before[i], before_t[i], Pm[i], pt_list[i].detach()) for i in range(6)])
            rband.add(adam[0], adam[1], k + 1, LR1,
                      [(before[i], before_t[i], Pm[i], pt_list[i].detach()) for i in range(6)])
            c1, c2 = 1 - B1 ** (k + 1), 1 - B2 ** (k + 1)
            for i, (mm, vv) in enumerate(zip(adam[0], adam[1])):
                s_abs[i] += ((mm.double() / c1) / ((vv.double() / c2).sqrt() + AEPS)).abs()
            if k % 50 == 0:
                s_med.append(float(torch.cat([((mm / c1) / ((vv / c2).sqrt() + AEPS)).abs().flatten()
                                              for mm, vv in zip(adam[0], adam[1])]).median()))
    per = {}
    per["a_replay_reproduces_run"] = _equal_list(Pm, [q.detach() for q in d["ends"][2]["params"]])
    ok_t, worst_t, nout_t = _band_compare(Pm, [q.detach() for q in pt_list], rband)
    ok_c, worst_c, nout_c = _band_compare(Pm, [q.detach() for q in pc_list], rband)
    okw, worst_w, _ = _band_compare(Pm, [q.detach() for q in pt_list], band)
    per["b_adamw_within_band"] = ok_t
    per["b_second_moments_identical"] = v_same
    per["c_coupled_adam_outside_band"] = not ok_c
    per["n_updates"] = len(d["grad_seq"])
    mut_per_step = LR1 * LR1 * 1e-1 * float(np.median(s_med)) if s_med else float("nan")
    # the arithmetic of 'decay after Adam': its offset is eta^2 lambda s per update; with a consistent sign it sums
    # to eta^2 lambda sum|s|, against the band B_K element by element
    reach = torch.cat([(LR1 * LR1 * 1e-1 * sa / bd).flatten() for sa, bd in zip(s_abs, rband.B)])
    return result(per, {"worst_ratio_adamw": worst_t, "n_outside_adamw": nout_t, "worst_ratio_coupled": worst_c,
                        "n_outside_coupled": nout_c, "band_per_update_median": float(np.median(rband.step_bands)),
                        "worst_ratio_adamw_wide_band": worst_w,
                        "mutation_offset_per_update_at_median_s": mut_per_step,
                        "median_abs_s": float(np.median(s_med)) if s_med else None,
                        "mutation_reach_over_band_max": float(reach.max()),
                        "mutation_reach_over_band_frac_above_10": float((reach > 10).double().mean())})


STORCH_MUT = [
    ("Mt1: the decay after the Adam step",
     [("            p.mul_(1 - lr * lam)\n", "            pass\n"),
      ("        p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)\n",
       "        p -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)\n        if lam != 0:\n            p.mul_(1 - lr * lam)\n")]),
    ("Mt2: lambda theta added to the gradient", [("            p.mul_(1 - lr * lam)\n", "            gr = gr.add(p, alpha=lam)\n")]),
]


def s_torch_c51(M) -> dict:
    """Box 2: adamw10 on seeds 100-109 stacked, 2 tasks x 100 updates, eager, every training update's gradients
    taken from the 'grads' hook and fed in lockstep to (a) M.adam_update_ on a shadow copy -- which must equal the
    run's own parameters bit for bit at the end of task 2 --, (b) torch.optim.AdamW(lr 1e-4, weight_decay = the
    registered 10, foreach=False) on the stacked tensors, within Band, second moments identical at every update,
    and (c) torch.optim.Adam(weight_decay 10) (coupled), outside the band."""
    dv = dev2()
    lam_reg = 10.0
    st = {"Pm": None}
    band_holder = {}

    def hook(ev, **s):
        if ev == "grads" and s["phase"] == "train":
            if st["Pm"] is None:
                init = [q.detach().clone() for q in s["P"]]
                st["Pm"] = [q.clone() for q in init]
                st["m"] = [torch.zeros_like(q) for q in init]
                st["v"] = [torch.zeros_like(q) for q in init]
                st["wd_c"] = torch.ones((), device=dv)
                st["wd_c"].fill_(M.decay_factor(LR2, M.ARM_LAMBDA["adamw10"]))
                st["pt"] = [torch.nn.Parameter(q.clone()) for q in init]
                st["pc"] = [torch.nn.Parameter(q.clone()) for q in init]
                st["ot"] = torch.optim.AdamW(st["pt"], lr=LR2, betas=(B1, B2), eps=AEPS, weight_decay=lam_reg,
                                             foreach=False)
                st["oc"] = torch.optim.Adam(st["pc"], lr=LR2, betas=(B1, B2), eps=AEPS, weight_decay=lam_reg,
                                            foreach=False)
                band_holder["b"] = Band([q.shape for q in init], dv)
                band_holder["r"] = RegBand([q.shape for q in init], dv)
                st["v_same"] = True
                st["k"] = 0
            g = [x.detach() for x in s["grads"]]
            before = [q.clone() for q in st["Pm"]]
            before_t = [q.detach().clone() for q in st["pt"]]
            M.adam_update_(st["Pm"], g, st["m"], st["v"], s["inv_c1"], s["inv_c2"], LR2, st["wd_c"],
                           M.ARM_LAMBDA["adamw10"] != 0)
            for q, gg in zip(st["pt"], g):
                q.grad = gg.clone()
            st["ot"].step()
            for q, gg in zip(st["pc"], g):
                q.grad = gg.clone()
            st["oc"].step()
            vt = [st["ot"].state[q]["exp_avg_sq"] for q in st["pt"]]
            st["v_same"] &= _equal_list(st["v"], vt)
            st["k"] += 1
            pp = [(before[i], before_t[i], st["Pm"][i], st["pt"][i].detach()) for i in range(6)]
            band_holder["b"].add(g, st["v"], s["tc"], LR2, pp)
            band_holder["r"].add(st["m"], st["v"], s["tc"], LR2, pp)
        if ev == "task_end" and s["t"] == 2:
            st["run_P"] = [q.detach().clone() for q in s["P"]]

    out = SCR / f"storch_{M.__name__}"
    shutil.rmtree(out, ignore_errors=True)
    M.run("adamw10", CHECK10, 2, dv, out, cifar=cifar(), progress=quiet, debug={"hook": hook},
          steps_hard=100, steps_easy=100, fresh=False)
    band = band_holder["r"]
    ok_t, worst_t, nout_t = _band_compare(st["Pm"], [q.detach() for q in st["pt"]], band)
    ok_c, worst_c, nout_c = _band_compare(st["Pm"], [q.detach() for q in st["pc"]], band)
    okw, worst_w, _ = _band_compare(st["Pm"], [q.detach() for q in st["pt"]], band_holder["b"])
    per = {"a_shadow_equals_run": _equal_list(st["Pm"], st.get("run_P", [])),
           "b_adamw_within_band": ok_t, "b_second_moments_identical": st["v_same"],
           "c_coupled_adam_outside_band": not ok_c, "n_updates": st["k"]}
    return result(per, {"worst_ratio_adamw": worst_t, "n_outside_adamw": nout_t, "worst_ratio_coupled": worst_c,
                        "n_outside_coupled": nout_c, "band_per_update_median": float(np.median(band.step_bands)),
                        "worst_ratio_adamw_wide_band": worst_w})


STORCH2_MUT = [
    ("Mt1: the decay after the Adam step",
     [("            p.mul_(wd_c)\n", "            pass\n"),
      ("        p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))\n",
       "        p.sub_(lr * (mi * inv_c1) / ((vi * inv_c2).sqrt() + eps))\n        if decay:\n            p.mul_(wd_c)\n")]),
    ("Mt2: lambda theta added to the gradient",
     [("            p.mul_(wd_c)\n", "            gr = gr.add(p.detach() * (1 - wd_c) / lr)\n")]),
]

# --------------------------------------------------------------------------
# S-deadpix: the in-run record of the decay
# --------------------------------------------------------------------------


def s_deadpix(M) -> dict:
    """Seed 100, 2 tasks x 2 epochs, the arms wd1e-1 and ref.  (i) the run's own wd_task record: deadpix_ok = 1 at
    both task ends in both arms, deadpix_n > 0, and wd_steps = 150 per tensor per task in wd1e-1, 0 in ref;
    (ii) recomputed here: the W1 columns of the seed's always-0 pixels at each task end equal the initial values
    times np.float32(1 - 1e-3 * 1e-1) iterated once per update (150 t times) in wd1e-1, and equal the initial values
    in ref; (iii) non-vacuity: those columns moved in wd1e-1, and the columns of the other pixels are not that
    iteration.  Bit comparisons."""
    per = {}
    c = np.float32(1 - LR1 * 1e-1)
    for arm in ("wd1e-1", "ref"):
        d = {"end_capture": (1, 2)}
        rows, arrays, wd, info = run1(M, arm, debug=d)
        dead = d["dead"]
        x = d["x"]
        dead_here = torch.nonzero((x == 0).all(0)).flatten().numpy()
        per[f"{arm}_dead_set_matches"] = bool(np.array_equal(dead, dead_here)) and len(dead) > 0
        exp_steps = 150 if arm == "wd1e-1" else 0
        per[f"{arm}_i_record"] = len(wd) == 2 and all(w["deadpix_ok"] == 1 and w["deadpix_n"] == len(dead_here)
                                                      for w in wd) and \
            all(w[f"wd_steps_{nm}"] == exp_steps for w in wd for nm in ("W1", "b1", "W2", "b2", "W3", "b3"))
        init_dead = d["init"][0][:, dead_here].numpy()
        init_live = d["init"][0][:, ~torch.from_numpy(np.isin(np.arange(784), dead_here))].numpy()
        ok, moved, live_not = True, True, True
        for t in (1, 2):
            W1 = d["ends"][t]["params"][0].detach()
            got = W1[:, dead_here].numpy()
            exp = _np_iter(init_dead, c, 150 * t) if arm == "wd1e-1" else init_dead
            ok &= bool(np.array_equal(got, exp))
            moved &= not np.array_equal(got, init_dead)
            live = W1[:, ~torch.from_numpy(np.isin(np.arange(784), dead_here))].numpy()
            live_not &= not np.array_equal(live, _np_iter(init_live, c, 150 * t))
        per[f"{arm}_ii_recomputed"] = ok
        if arm == "wd1e-1":
            per[f"{arm}_iii_moved"] = moved
            per[f"{arm}_iii_live_columns_are_not_the_iteration"] = live_not
    return result(per)


SDP_MUT = [
    ("Md1: coupled L2", [("            p.mul_(1 - lr * lam)\n", "            gr = gr.add(p, alpha=lam)\n")]),
    ("Md2: the decay skipped every other update",
     [("        if lam != 0:\n            p.mul_(1 - lr * lam)\n", "        if lam != 0 and tc[0] % 2 == 0:\n            p.mul_(1 - lr * lam)\n")]),
    ("Md3: c multiplied in float64 (another rounding)",
     [("            p.mul_(1 - lr * lam)\n", "            p.copy_((p.double() * (1 - lr * lam)).float())\n")]),
]

# --------------------------------------------------------------------------
# S5: the branches share their streams; the first update; all differ by task 3
# --------------------------------------------------------------------------

STREAMS = ("init_sha256", "subset_idx_sha256", "labels_sha256", "batch_sha256")


def s5(M) -> dict:
    """The five registered arms, seed 100, 3 tasks x 2 epochs: (i) init, subset, labels and batch-order hashes
    identical; (ii) every lambda arm's first update uses ref's gradients and ends with ref's moments bit for bit, and
    its parameters are fl(fl(p0 c) - a) with a = lr (m/c1) / (sqrt(v/c2) + eps) from those moments (computed here)
    and c = the registered lambda's np.float32(1 - 1e-3 lambda), while ref's are fl(p0 - a); (iii) the five final
    states are pairwise different."""
    per, infos, first, finals = {}, {}, {}, {}
    for arm in ARM_MAP_REG:
        d = {"capture": ((1, 0),)}
        rows, arrays, wd, info = run1(M, arm, tasks=3, debug=d)
        infos[arm] = tuple(info[k] for k in STREAMS)
        first[arm] = d["steps"][(1, 0)]
        finals[arm] = info["final_state_sha256"]
    per["i_streams_identical"] = len(set(infos.values())) == 1
    r = first["ref"]
    c1, c2 = 1 - B1, 1 - B2
    with flush_denormal():
        a = [LR1 * (m / c1) / ((v / c2).sqrt() + AEPS) for m, v in zip(r["after"]["m"], r["after"]["v"])]
        ref_ok = _equal_list(r["after"]["params"], [p0 - ai for p0, ai in zip(r["before"]["params"], a)])
        arm_ok = {}
        for arm, lam in ARM_MAP_REG.items():
            if arm == "ref":
                continue
            f = first[arm]
            c = np.float32(1 - LR1 * lam)
            exp = [torch.from_numpy(p0.numpy() * c) - ai for p0, ai in zip(f["before"]["params"], a)]
            arm_ok[arm] = (_equal_list(f["grads"], r["grads"]) and _equal_list(f["after"]["m"], r["after"]["m"])
                           and _equal_list(f["after"]["v"], r["after"]["v"]) and _equal_list(f["after"]["params"], exp))
    per["ii_ref_first_update"] = ref_ok
    per["ii_lambda_first_updates"] = all(arm_ok.values())
    per["iii_all_differ"] = len(set(finals.values())) == len(ARM_MAP_REG)
    return result(per, {"arm_first_update": arm_ok})


S5_MUT = [
    ("M5a: the label stream depends on the arm",
     [('    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)\n',
       '    g_lab, g_batch = H.stream("rl_labels" + arm, seed), H.stream("rl_batch", seed)\n')]),
    ("M5b: the decay starts at task 2",
     [("                    adamw_step_(params, grads, adam, lr, lams, n_decay)\n",
       "                    adamw_step_(params, grads, adam, lr, lams if t >= 2 else (0.0,) * 6, n_decay)\n")]),
    ("M5c: ref is decayed", [('ARM_LAMBDA = {"ref": 0.0,', 'ARM_LAMBDA = {"ref": 1e-1,')]),
]

# --------------------------------------------------------------------------
# S-ctrl: the readout-only decay (the mutation control)
# --------------------------------------------------------------------------

CTRL_TASKS = 10


def s_ctrl(M) -> dict:
    """Seed 100, 10 tasks x 80 epochs: ref, ro1e-1 (W3 and b3 only, lambda 0.1; M.arm_lams('ro1e-1')) and wd1e-1
    (all six).  s = the unit median of sigma at the end of task 10.  PASS iff in both layers
    s_ro > sqrt(s_ref s_wd) (the geometric midpoint) and s_wd < s_ref.  Only sigma and PASS/FAIL are read."""
    s = {}
    for arm in ("ref", "ro1e-1", "wd1e-1"):
        rows, arrays, wd, info = run1(M, arm, tasks=CTRL_TASKS, epochs=80)
        e = ends_idx(arrays, STEPS_RUN)
        s[arm] = {li: float(np.median(arrays[f"sigma_l{li}"][e[CTRL_TASKS]])) for li in (1, 2)}
    per = {}
    for li in (1, 2):
        mid = math.sqrt(s["ref"][li] * s["wd1e-1"][li])
        per[f"l{li}_ro_above_midpoint"] = s["ro1e-1"][li] > mid
        per[f"l{li}_wd_below_ref"] = s["wd1e-1"][li] < s["ref"][li]
    return result(per, {"sigma_t10": {a: {str(k): v for k, v in d_.items()} for a, d_ in s.items()},
                        "lams_ro": list(M.arm_lams("ro1e-1"))})


SCTRL_MUT = [
    ("Mc1: ro1e-1 decays every tensor",
     [('CHECK_ARMS = {"ro1e-1": (0.0, 0.0, 0.0, 0.0, 1e-1, 1e-1)}', 'CHECK_ARMS = {"ro1e-1": (1e-1,) * 6}')]),
]

# --------------------------------------------------------------------------
# S7: the same call twice
# --------------------------------------------------------------------------


def s7(M) -> dict:
    """ref and wd1e-1, seed 100, 2 tasks x 2 epochs, twice each: the same arrays, hashes, rows and wd rows."""
    per = {}
    for arm in ("ref", "wd1e-1"):
        r1_, a1, w1, i1 = run1(M, arm)
        r2_, a2, w2, i2 = run1(M, arm)
        per[f"{arm}_arrays"] = set(a1) == set(a2) and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1)
        per[f"{arm}_hashes"] = all(i1[k] == i2[k] for k in ("init_sha256", "labels_sha256", "batch_sha256",
                                                            "task1_end_state_sha256", "final_state_sha256"))
        per[f"{arm}_rows"] = r1_ == r2_ and w1 == w2 and len(a1) > 10
    return result(per)


S7_MUT = [("M7a: the batch order from the global generator",
           [("            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)\n",
             "            order = torch.randperm(N_IMAGES).to(device)\n")])]

# --------------------------------------------------------------------------
# box 2: S-nochange, S-host-repro, S-graph, S-fresh
# --------------------------------------------------------------------------

_host_cache: dict = {}


def erun(M, arm, seeds, tasks, out: Path, **kw):
    shutil.rmtree(out, ignore_errors=True)
    return M.run(arm, seeds, tasks, dev2(), out, cifar=cifar(), progress=quiet, **kw)


def hrun(seeds, tasks, steps_hard=C.STEPS_PER_TASK, steps_easy=C.STEPS_PER_TASK, fresh=True, iv="none") -> Path:
    """The host's unmodified run(), R / std / lr 1e-4 / graph -- cached per configuration."""
    key = (iv, tuple(seeds), tasks, steps_hard, steps_easy, fresh)
    if key not in _host_cache:
        out = SCR / f"host_{len(_host_cache)}"
        shutil.rmtree(out, ignore_errors=True)
        C.run("R", list(seeds), "std", tasks, dev2(), out, lr=1e-4, cifar=cifar(), fresh=fresh, graph=True,
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
    """cap_cifar5p1_1007 spec 7.1 (copied from analysis/baselines_cifar5p1_1008/checks.py)."""
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
    """spec 2.3 (a)(b): the lambda = 0 path, seeds 0-9, 30 tasks, graph, 2 threads, fresh: (a) per_task.csv and
    fresh_control.csv byte-identical to the unmodified host run() in this environment; (b) identical to the
    committed R record in every column but eff_rank (same rows, same header), the committed eff_rank inside the
    cap_cifar5p1_1007 spec 7.1 band around the value recomputed here (and the band rejects the next task's value:
    non-vacuity), fresh_control.csv byte-identical to the committed one; (c) a 4-task run is byte-identical to the
    host's 4-task run.  The real module's outputs are kept in results/adamw_dose_1010/c51/_nochange/."""
    per, extra = {}, {}
    out_e = SCR / f"nochange_{M.__name__}"
    bands = {}

    def obs(t, st):
        z1, a1, z2, a2, _ = B.forward(st["P"], st["Xt"], st["act"], train=False)
        for li, a in ((1, a1), (2, a2)):
            for r in st["live"]:
                bands[(SEEDS2[r], t, li)] = eff_rank_band(a[r])

    erun(M, "nochange", SEEDS2, C.N_TASKS, out_e, observer=obs)
    host = hrun(SEEDS2, C.N_TASKS)
    per["a_per_task_identical_to_host_now"] = same_bytes(out_e / "per_task.csv", host / "per_task.csv")
    per["a_fresh_identical_to_host_now"] = same_bytes(out_e / "fresh_control.csv", host / "fresh_control.csv")
    mine, com = read_csv(out_e / "per_task.csv"), read_csv(HOST_REC / "per_task.csv")
    per["b_same_rows"] = len(mine) == len(com) == len(SEEDS2) * C.N_TASKS and \
        all((a["seed"], a["task"]) == (b["seed"], b["task"]) for a, b in zip(mine, com))
    per["b_same_columns"] = list(mine[0]) == list(com[0])
    ok, diff = same_but_eff_rank(out_e / "per_task.csv", HOST_REC / "per_task.csv")
    per["b_non_eff_rank_identical"] = ok
    extra["b_non_eff_rank_diff_cells"] = diff
    per["b_fresh_identical_to_committed"] = same_bytes(out_e / "fresh_control.csv", HOST_REC / "fresh_control.csv")
    inside, printed_ok, n_equal, n_rows = True, True, 0, 0
    max_rel_dev, outside, shifted_rejected, shifted_n = 0.0, [], 0, 0
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
            n_equal += a[k] == b[k]
            mv = float(a[k])
            if mv > 0:
                max_rel_dev = max(max_rel_dev, abs(cv - mv) / mv)
            nb = com_idx.get((s_, t + 1))
            if nb is not None:
                shifted_n += 1
                shifted_rejected += not (lo <= float(nb[k]) <= hi)
    per["b_eff_rank_inside_band"] = inside and n_rows == 2 * len(mine)
    per["b_observer_reproduces_printed_eff_rank"] = printed_ok
    per["b_band_rejects_a_shifted_column"] = shifted_rejected > 0
    extra.update({"eff_rank_cells": n_rows, "eff_rank_cells_byte_equal": n_equal,
                  "eff_rank_max_rel_deviation": max_rel_dev, "eff_rank_outside_band": outside[:10],
                  "shifted_column_rejected": f"{shifted_rejected}/{shifted_n}",
                  "nochange_per_task_sha256": sha(out_e / "per_task.csv"),
                  "committed_per_task_sha256": sha(HOST_REC / "per_task.csv"),
                  "host_now_per_task_sha256": sha(host / "per_task.csv")})
    out_s = SCR / f"nochange4_{M.__name__}"
    erun(M, "nochange", SEEDS2, 4, out_s)
    host4 = hrun(SEEDS2, 4)
    per["c_short_per_task_identical_to_host_now"] = same_bytes(out_s / "per_task.csv", host4 / "per_task.csv")
    per["c_short_fresh_identical_to_host_now"] = same_bytes(out_s / "fresh_control.csv", host4 / "fresh_control.csv")
    if getattr(M, "_is_real", False):
        keep = OUT / "c51" / "_nochange"
        keep.mkdir(parents=True, exist_ok=True)
        for f in ("per_task.csv", "fresh_control.csv", "wd_task.csv", "provenance.json"):
            shutil.copy2(out_e / f, keep / f)
    return result(per, extra)


NOCHANGE_MUT = [
    ("Mn1: the lambda = 0 path decayed with lambda 1e-2", [('ARM_LAMBDA = {"nochange": 0.0,', 'ARM_LAMBDA = {"nochange": 1e-2,')]),
    ("Mn2: lr 1.0001e-4", [('ACT_ARM, COND, LR = "R", "std", C.LR ', 'ACT_ARM, COND, LR = "R", "std", C.LR * 1.0001 ')]),
    ("Mn3: the Adam update count restarts every task",
     [('        tc = train_task(batches, tc, "train", t)\n', '        tc = train_task(batches, 0, "train", t)\n')]),
]

HOST_CELLS = {"l2init1e-2": ("l2init:1e-2", "R_std_lr0.0001_l2init1e-2")}


def s_host_repro() -> None:
    """spec 2.3 (c): the unmodified host run('R', iv='l2init:1e-2', seeds 0-9) reproduces the committed L2 Init record
    in every per_task column but eff_rank, and its fresh_control.csv byte for byte.  Mutations are wrong comparisons:
    against the l2init 1e-3 record, and against the seeds 10-19 l2init 1e-2 record."""
    torch.set_num_threads(2)
    t0 = time.time()
    out = hrun(SEEDS2, C.N_TASKS, iv="l2init:1e-2")

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


def s_graph(M) -> dict:
    """adamw10, seeds 100-109, 4 tasks x 100 updates: graph and eager give byte-identical per_task.csv, wd_task.csv
    and fresh_self.csv and bit-identical final P, m, v; the graph was used; the decays were counted (100)."""
    per, extra = {}, {}
    fa, fb = {}, {}
    oa, ob = SCR / f"graph_g_{M.__name__}", SCR / f"graph_e_{M.__name__}"
    erun(M, "adamw10", CHECK10, 4, oa, graph=True, steps_hard=100, steps_easy=100, final=fa)
    erun(M, "adamw10", CHECK10, 4, ob, graph=False, steps_hard=100, steps_easy=100, final=fb)
    for f in ("per_task.csv", "wd_task.csv", "fresh_self.csv"):
        per[f"{f}_identical"] = same_bytes(oa / f, ob / f)
    per["final_state_bitwise"] = all(_equal_list(fa[k], fb[k]) for k in ("P", "m", "v"))
    pa = json.loads((oa / "provenance.json").read_text())
    per["graph_was_used"] = "CUDA graph" in pa["engine"]
    wd = read_csv(oa / "wd_task.csv")
    per["decays_counted"] = all(int(r["wd_steps_W1"]) == 100 and int(r["wd_steps_b3"]) == 100 for r in wd)
    extra["engine"] = pa["engine"]
    return result(per, extra)


GRAPH_MUT = [
    ("Mg1: the decay factor rebound outside the graph (the graph keeps reading 1)",
     [("        wd_c.fill_(decay_factor(lr, lam))", "        wd_c = torch.full((), decay_factor(lr, lam), device=device)")]),
    ("Mg2: static_idx one batch behind while replaying",
     [("            static_idx.copy_(batches[:, j])", "            static_idx.copy_(batches[:, j] if cg is None else batches[:, max(j - 1, 0)])")]),
]


def s_fresh(M) -> dict:
    """Short runs (seeds 100-102, 3 tasks, 200/100 updates): (i) the last hard task's batches have the same sha256
    in adamw10, adamw5 and the lambda = 0 path (the common fresh is R's own); (ii) adamw10 writes fresh_self.csv and
    no fresh_control.csv, its continual column is its own task-3 online accuracy, and the lambda = 0 path's
    fresh_control.csv is the host's byte for byte; (iii) at the start of fresh_self, P equals the initial values,
    m = v = 0, wd_c = np.float32(1 - 1e-4 * 10), and the first fresh_self update has update count 1; (iv) every
    fresh_self update decays (wd_steps = 200 in fresh_self.csv)."""
    per = {}
    outs = {}
    for arm in ("nochange", "adamw10", "adamw5"):
        o = SCR / f"fresh_{arm}_{M.__name__}"
        erun(M, arm, CHECK3, 3, o, steps_hard=200, steps_easy=100)
        outs[arm] = o
    shas = {a: json.loads((o / "provenance.json").read_text())["fresh_batches_sha256"] for a, o in outs.items()}
    per["i_same_last_hard_batches"] = len(set(shas.values())) == 1 and None not in shas.values()
    host = hrun(CHECK3, 3, steps_hard=200, steps_easy=100)
    per["ii_nochange_fresh_is_host"] = same_bytes(outs["nochange"] / "fresh_control.csv", host / "fresh_control.csv")
    per["ii_adamw10_writes_fresh_self_only"] = (outs["adamw10"] / "fresh_self.csv").exists() and \
        not (outs["adamw10"] / "fresh_control.csv").exists()
    cont = {r["seed"]: r["online_acc"] for r in read_csv(outs["adamw10"] / "per_task.csv") if r["task"] == "3"}
    fs = read_csv(outs["adamw10"] / "fresh_self.csv")
    per["ii_continual_is_own_t3"] = len(fs) == 3 and all(r["continual_online_acc"] == cont[r["seed"]] for r in fs)
    per["iv_fresh_self_decays_every_update"] = all(int(r[f"wd_steps_{nm}"]) == 200 for r in fs
                                                   for nm in ("W1", "b1", "W2", "b2", "W3", "b3"))
    init = [q.detach() for s in CHECK3 for q in C.init_params("R", s, dev2())]
    P0 = [torch.stack(init[i::6]).contiguous() for i in range(6)]
    seen = {}
    c_exp = float(np.float32(1 - LR2 * 10.0))

    def hook(ev, **s):
        if ev == "fresh_self_start":
            ok = _equal_list(P0, [q.detach() for q in s["P"]])
            ok &= all(bool((q == 0).all()) for q in (*s["m"], *s["v"]))
            ok &= float(s["wd_c"]) == c_exp
            seen["start"] = ok
        if ev == "post_step" and s["phase"] == "fresh_self" and s["j"] == 0:
            seen["first_tc"] = s["tc"]
    erun(M, "adamw10", CHECK3, 3, SCR / f"freshhook_{M.__name__}", debug={"hook": hook}, steps_hard=200,
         steps_easy=100)
    per["iii_start_state_initial_and_c"] = seen.get("start", False)
    per["iii_first_update_count_1"] = seen.get("first_tc") == 1
    return result(per, {"fresh_batches_sha256": shas})


FRESH_MUT = [
    ("Mf1: the Adam moments not zeroed before fresh_self",
     [("                for q in (*adam_m, *adam_v):\n                    q.zero_()\n",
       "                for q in ():\n                    q.zero_()\n")]),
    ("Mf2: the decay switched off for fresh_self",
     [("            # fresh_self: the same AdamW from the initial values (report only)\n",
       "            # fresh_self: the same AdamW from the initial values (report only)\n            wd_c.fill_(1.0)\n")]),
]

# --------------------------------------------------------------------------
# S8: the verdict on synthetic data
# --------------------------------------------------------------------------

T_TABLE = {(0.975, 9): 2.262157, (0.9875, 9): 2.685011, (0.95, 9): 1.833113, (0.975, 8): 2.306004}
TOL_T = 5e-7 + 1e-12
SPT1 = 6000
WIN = (51, 100)
A1 = 0.82


def _mf(t: int) -> float:
    return 0.11 + 0.004 * math.sin(t)


R_FLOOR = float(np.mean([_mf(t) - 0.01 for t in range(51, 101)]))     # ref's floored window level


F_WIN = float(np.mean([_mf(t) for t in range(WIN[0], WIN[1] + 1)]))
S_WIN = float(np.std([_mf(t) for t in range(WIN[0], WIN[1] + 1)], ddof=1))
THR_WIN = F_WIN + 3.0 * S_WIN / math.sqrt(50)


def _centered(rng, n, sd):
    x = rng.normal(0.0, 1.0, n)
    x = x - x.mean()
    return x / x.std(ddof=1) * sd if sd > 0 else np.zeros(n)


def w_star(lam: float, d: float) -> float:
    c = float(np.float32(1 - LR1 * lam))
    return LR1 * math.sqrt(d / (2.0 * (1.0 - c)))


# per arm: collapse (task or None), L (alive level), eff2 (E2 effect), eq (ratio to w*, l1, l2), kappa (l1, l2),
# zfrac (dtrain_zero_l2 pattern), A1 (task-1 online)
DEFAULT1 = {
    "ref": {"collapse": 5},
    "wd1e-3": {"collapse": 6},
    "wd1e-2": {"collapse": None, "L": 0.55, "eff2": 0.15, "eq": (1.2, 1.2), "kappa": (0.0, 0.0), "zfrac": "zero"},
    "wd3e-2": {"collapse": None, "L": 0.75, "eff2": 0.25, "eq": (1.5, 1.4), "kappa": (0.0, 0.0), "zfrac": "zero"},
    "wd1e-1": {"collapse": None, "L": 0.65, "eff2": 0.30, "eq": (1.1, 0.9), "kappa": (0.0, 0.0), "zfrac": "zero"},
}
REF_G = (0.5, 2.0)                       # ref's live sigma slopes per task (layers 1, 2)


def synth1(scen: dict) -> dict:
    rng = np.random.default_rng(scen.get("rng", 1010))
    arms = {a: {**DEFAULT1[a], **scen.get("arms", {}).get(a, {})} for a in DEFAULT1}
    NU, T = 100, 100
    r2 = -0.55
    m2 = _centered(rng, 10, 0.02)
    shards = {}
    for a, sp in arms.items():
        lam = {"ref": 0.0, "wd1e-3": 1e-3, "wd1e-2": 1e-2, "wd3e-2": 3e-2, "wd1e-1": 1e-1}[a]
        n2 = np.zeros(10) if a == "ref" else _centered(rng, 10, sp.get("noise2", 0.02))
        n1 = _centered(rng, 10, 0.0 if a == "ref" else sp.get("noise1", 0.01))
        neq = [_centered(rng, 10, sp.get("eq_noise", 0.02)) for _ in range(2)]
        nk = [_centered(rng, 10, sp.get("kappa_noise", 0.01)) for _ in range(2)]
        nz = _centered(rng, 10, sp["zfrac"][2] if isinstance(sp.get("zfrac"), tuple) else 0.0)
        for si in range(10):
            col = sp["collapse"]
            if isinstance(col, dict):
                col = col.get(si, col.get("default"))
            a1v = sp.get("A1", A1)
            online = []
            for t in range(1, T + 1):
                if t == 1:
                    online.append(a1v)
                elif col is not None and t >= col:
                    online.append(sp.get("floor_level", _mf(t) - 0.01))
                elif col is not None:
                    online.append(a1v)
                else:
                    online.append(sp["L"] + (n1[si] if WIN[0] <= t <= WIN[1] else 0.0))
            pt = pd.DataFrame({"task": np.arange(1, T + 1), "online_acc": online, "memo_acc": online,
                               "major_frac": [_mf(t) for t in range(1, T + 1)]})
            d_eff = 610 + si
            keys = {k: [] for k in ("dtrain_mean_l2", "dtrain_mean_l1", "dtrain_zero_l2", "row_norm_l1", "row_norm_l2",
                                    "sigma_l1", "sigma_l2")}
            task_, step_ = [], []
            for t in range(1, T + 1):
                for step in (0, SPT1):
                    task_.append(t)
                    step_.append(step)
                    if step == 0:
                        for k in keys:
                            keys[k].append(np.full(NU, 7.0))
                        continue
                    if t == 1:
                        lv = 0.6
                    elif WIN[0] <= t <= WIN[1]:
                        lv = 0.6 + r2 + m2[si] + (0.0 if a == "ref" else sp.get("eff2", 0.0) + n2[si])
                    else:
                        lv = 0.6 + r2 + m2[si]
                    keys["dtrain_mean_l2"].append(lv + (np.arange(NU) - 49.5) * 1e-4)
                    keys["dtrain_mean_l1"].append(np.full(NU, 0.4))
                    zf = sp.get("zfrac", "one") if (WIN[0] <= t <= WIN[1] or (col is not None and t >= col)) else "zero"
                    if a == "ref" and t >= 5:
                        zf = "one"
                    if zf == "one":
                        z = np.ones(NU)
                    elif zf == "zero":
                        z = np.zeros(NU)
                    elif zf == "dead30":            # 30 units fully dead, 70 alive: mean 0.3, median 0
                        z = np.r_[np.ones(30), np.zeros(70)]
                    elif isinstance(zf, float):
                        z = np.full(NU, zf)
                    elif isinstance(zf, tuple):             # ("seeds", centre, sd): per-seed level
                        z = np.full(NU, zf[1] + nz[si])
                    keys["dtrain_zero_l2"].append(z)
                    for li, dd in ((1, d_eff), (2, 100)):
                        if lam > 0 and "eq" in sp:
                            base = sp["eq"][li - 1] * w_star(lam, dd) * 2 ** neq[li - 1][si]
                        else:
                            base = 3.0 + 0.1 * t
                        keys[f"row_norm_l{li}"].append(base + (np.arange(NU) - 49.5) * 1e-6)
                        g = REF_G[li - 1]
                        if a == "ref" or col is not None:
                            s_val = g * min(t, 10) + 1.0
                        else:
                            k_ = sp.get("kappa", (0.0, 0.0))[li - 1] + nk[li - 1][si]
                            s_val = 2.0 + (k_ * g * (t - 75) if WIN[0] <= t <= WIN[1] else 0.0)
                        keys[f"sigma_l{li}"].append(np.full(NU, s_val) + (np.arange(NU) - 49.5) * 1e-6)
            units = {"task": np.array(task_, float), "step": np.array(step_, float),
                     **{k: np.stack(v) for k, v in keys.items()}}
            wd = pd.DataFrame({"task": np.arange(1, T + 1), "deadpix_ok": 1,
                               **{f"wd_steps_{nm}": (SPT1 if lam > 0 else 0) for nm in ("W1", "b1", "W2", "b2", "W3", "b3")}})
            for (ia, iseed, itask, icol) in scen.get("idle", ()):
                if ia == a and iseed == si:
                    wd.loc[wd.task == itask, icol] = 0
            hs = {k: f"h{si}" for k in STREAMS}
            if si in scen.get("break", ()) and a == "wd3e-2":
                hs["batch_sha256"] = "broken"
            prov = {"git_hash": "synthetic", "per_seed": {str(si): {**hs, "tasks_completed": T, "d_eff": d_eff,
                                                                    "divergence": {"diverged": False}}}}
            shards[(a, si)] = {"units": units, "per_task": pt, "wd": wd, "prov": prov}
    return shards


R_LATE = np.array([0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.44])
L_LATE = R_LATE + 0.22


def synth2(scen: dict) -> dict:
    """R and L records and two AdamW arms: late = R + a_shift (+ per-seed offsets), early = R_early + e_shift."""
    rows_R, rows_L, fc = [], [], []
    rng = np.random.default_rng(scen.get("rng", 7))
    arms = {}
    shifts = scen.get("arms", {"adamw5": {"late": -0.05, "early": -0.10}, "adamw10": {"late": -0.04, "early": -0.15}})
    R_early = 0.61
    for s in range(10):
        for t in range(1, 31):
            hard = t % 2 == 1
            on_R = (R_LATE[s] if t >= 21 else R_early) if hard else 0.95
            rows_R.append({"seed": s, "task": t, "hard": int(hard), "online_acc": on_R,
                           **{k: 0.3 + 0.01 * s for k in DIAG}})
            rows_L.append({"seed": s, "task": t, "hard": int(hard),
                           "online_acc": (L_LATE[s] if t >= 21 else R_early) if hard else 0.95})
        fc.append({"seed": s, "fresh_online_acc": R_LATE[s] + 0.2 + 0.01 * (s % 3), "fresh_gap": 0.2 + 0.01 * (s % 3)})
    for a, sh in shifts.items():
        if a in scen.get("drop", ()):
            continue
        per = sh.get("per_seed", np.zeros(10))
        rows, wdr, fs = [], [], []
        for s in range(10):
            for t in range(1, 31):
                hard = t % 2 == 1
                late = R_LATE[s] if np.isnan(per[s]) else R_LATE[s] + sh["late"] + per[s]   # nan: a tie
                on = (late if t >= 21 else R_early + sh["early"] + 0.01 * (s % 2)) if hard else 0.95
                rows.append({"seed": s, "task": t, "hard": int(hard), "online_acc": on,
                             **{k: 0.3 + 0.01 * s + sh.get("diag", 0.0) for k in DIAG}})
                wdr.append({"seed": s, "task": t, "steps": 780,
                            **{f"wd_steps_{nm}": (0 if (a, s, t) in scen.get("idle2", ()) else 780)
                               for nm in ("W1", "b1", "W2", "b2", "W3", "b3")},
                            **{k: 0.1 + sh.get("wd2", 0.0) for k in WD2}})
            fs.append({"seed": s, "fresh_self_online_acc": 0.5})
        arms[a] = {"per_task": pd.DataFrame(rows), "wd": pd.DataFrame(wdr), "fresh_self": pd.DataFrame(fs),
                   "prov": {"fresh_batches_sha256": "x", "divergences": []}}
    data = {"arms": arms, "R": {"per_task": pd.DataFrame(rows_R), "fresh_control": pd.DataFrame(fc)},
            "L": {"per_task": pd.DataFrame(rows_L)},
            "nochange": {"wd": pd.DataFrame([{"seed": s, "task": t, **{k: 0.1 for k in WD2}}
                                             for s in range(10) for t in range(1, 31)]),
                         "prov": {"fresh_batches_sha256": "x"}},
            "reuse": {"a": True, "b": True, "c": not scen.get("reuse_fail")}}
    if scen.get("fresh_gap_zero"):
        data["R"]["fresh_control"]["fresh_gap"] = [(-1) ** s * 0.01 for s in range(10)]
    return data


DIAG = ("dead_frac_l2", "mob_l2", "eff_rank_l2", "zbar_l2", "zsd_l2", "w_norm_l1", "w_norm_l2", "w_norm_l3")
WD2 = ("relpos_l2", "p2", "b1_norm", "b2_norm", "b3_norm", "w1_row_med", "w2_row_med", "w3_row_med")
ONE_OF_10 = np.array([0.0] * 9 + [-0.30])          # 9 seeds up, 1 down: p = 0.0215 (sign test, n = 10)

S8_1 = [
    ("designed", {}, {"wd1e-3": "COLLAPSED", "wd1e-2": "RESCUED", "wd3e-2": "RESCUED", "wd1e-1": "RESCUED",
                      "wd3e-2_three": "RESCUED", "wd1e-3_three": "FAIL", "box": "WD_RESCUES_BOTH",
                      "dose": "DOSE_PEAKED", "eq_wd3e-2": "EQUILIBRIUM_MATCH", "eq_wd1e-1": "EQUILIBRIUM_MATCH",
                      "width_wd3e-2": "WIDTH_STOPPED", "width_wd1e-1": "WIDTH_STOPPED",
                      "depth_wd3e-2": "DEPTH_ABOVE_FLOOR"},
     {"E1|wd3e-2": (0.75 - 0.82) - (R_FLOOR - 0.82), "eq_l1|wd1e-1": 1.1, "n_valid": 10}),
    ("monotone_and_rfo", {"arms": {"wd1e-1": {"L": 0.80, "eff2": 0.0, "noise2": 0.05}}},
     {"dose": "DOSE_MONOTONE", "wd1e-1": "RESCUED_FUNCTION_ONLY", "wd1e-1_three": "PARTIAL", "box": "WD_RESCUES_ONE"}, {}),
    ("flat", {"arms": {a: {"collapse": 5} for a in ("wd1e-3", "wd1e-2", "wd3e-2", "wd1e-1")}},
     {"dose": "DOSE_FLAT", "box": "WD_RESCUES_NONE", "wd3e-2": "COLLAPSED",
      "width_wd3e-2": "WIDTH_FROZEN"}, {}),
    ("other", {"arms": {"wd1e-3": {"collapse": 5, "floor_level": 0.05}, "wd1e-1": {"L": 0.85}}},
     {"dose": "DOSE_OTHER", "wd1e-3": "COLLAPSED"}, {}),
    ("split_alive_unresolved", {"arms": {"wd3e-2": {"collapse": {7: 20, 8: 20, 9: 20, "default": None}},
                                         "wd1e-1": {"A1": 0.95, "L": R_FLOOR + 0.13, "eff2": 0.0}}},
     {"wd3e-2": "SPLIT", "wd3e-2_three": "PARTIAL", "width_wd3e-2": "WIDTH_SPLIT", "wd1e-1": "ALIVE_UNRESOLVED",
      "wd1e-1_three": "PARTIAL"}, {}),
    ("t1_lower", {"arms": {"wd1e-1": {"A1": 0.70, "L": 0.65}}},
     {"wd1e-1": "RESCUED", "t1_lower|wd1e-1": True, "t1_lower|wd3e-2": False},
     {"E1|wd1e-1": (0.65 - 0.70) - (R_FLOOR - 0.82)}),
    ("not_reproduced", {"arms": {"ref": {"collapse": None, "L": 0.5, "eq": (1.0, 1.0), "zfrac": "zero"}}},
     {"wd3e-2": "NOT_REPRODUCED", "box": "NOT_REPRODUCED", "dose": "NOT_REPRODUCED"}, {}),
    ("inapplicable_streams", {"break": (7, 8, 9)}, {"wd3e-2": "INAPPLICABLE", "box": "INAPPLICABLE"}, {"n_valid": 7}),
    ("inapplicable_wd_idle", {"idle": (("wd1e-1", 3, 42, "deadpix_ok"), ("wd1e-2", 4, 80, "wd_steps_b2"))},
     {"wd1e-1": "INAPPLICABLE", "wd1e-2": "INAPPLICABLE", "wd3e-2": "RESCUED", "box": "WD_RESCUES_ONE",
      "dose": "INAPPLICABLE"}, {}),
    ("eq_off_and_width", {"arms": {"wd3e-2": {"eq": (3.0, 1.0), "kappa": (0.6, 0.0)},
                                   "wd1e-1": {"eq": (0.30, 1.0), "kappa": (-0.6, 0.0), "kappa_noise": 0.01}}},
     {"eq_wd3e-2": "EQUILIBRIUM_OFF", "eq_wd1e-1": "EQUILIBRIUM_OFF", "width_wd3e-2": "WIDTH_NOT_STOPPED",
      "width_wd1e-1": "WIDTH_NOT_STOPPED", "eq_l1|wd3e-2": "OFF_HIGH", "eq_l1|wd1e-1": "OFF_LOW",
      "width_l1|wd3e-2": "GROWING", "width_l1|wd1e-1": "SHRINKING"}, {}),
    ("eq_unresolved", {"arms": {"wd3e-2": {"eq": (1.95, 1.0), "eq_noise": 0.3},
                                "wd1e-1": {"kappa": (0.09, 0.0), "kappa_noise": 0.1}}},
     {"eq_l1|wd3e-2": "UNRESOLVED", "width_l1|wd1e-1": "UNRESOLVED"}, {}),
    ("eq_edge_784", {"arms": {"wd3e-2": {"eq": (0.53, 1.0), "eq_noise": 0.005}}},
     {"eq_l1|wd3e-2": "MATCH", "eq_wd3e-2": "EQUILIBRIUM_MATCH"}, {}),
    ("width_ref_frozen", {"arms": {"wd3e-2": {"kappa": (0.02, 0.0), "kappa_noise": 0.01}}},
     {"width_wd3e-2": "WIDTH_STOPPED"}, {}),
    ("depth_at_and_median", {"arms": {"wd3e-2": {"zfrac": "dead30"}, "wd1e-1": {"zfrac": ("seeds", 0.1, 0.05)}}},
     {"depth_wd3e-2": "DEPTH_AT_FLOOR", "depth_wd1e-1": "DEPTH_UNRESOLVED"}, {}),
    ("floor_margin", {"arms": {"wd1e-2": {"collapse": 20, "floor_level": THR_WIN + 0.002},
                               "wd1e-1": {"collapse": 20, "floor_level": F_WIN + 0.0005}}},
     {"wd1e-2": "RESCUED", "wd1e-1": "COLLAPSED"}, {}),
    ("holm_matters", {}, {}, {}),
]


def _dose_holm_scenario(V) -> dict:
    """A DOSE scenario whose only non-flat step has 9 of 10 seeds up (p = 0.0215): two-sided Holm over 4 steps
    keeps it FLAT (0.0215 > 0.0125); without Holm, or one-sided (0.0107), it is UP."""
    sh = synth1({"arms": {a: {"collapse": 5} for a in ("wd1e-3", "wd1e-2", "wd3e-2", "wd1e-1")}})
    for s in range(10):
        pt = sh[("wd1e-1", s)]["per_task"]
        bump = 0.002 if s < 9 else -0.002
        pt.loc[pt.task >= 51, "online_acc"] = pt.loc[pt.task >= 51, "online_acc"] + bump
    return V.analyze1(sh)


def _pick1(res: dict, key: str):
    if key == "n_valid":
        return len(res["valid_seeds"])
    if key.startswith("E1|"):
        arm = key.split("|")[1]
        return next(r["mean"] for r in res["rows"] if r["arm"] == arm and r["endpoint"] == "E1"
                    and r["role"] == "registered")
    if key.startswith("eq_l1|") or key.startswith("eq_l2|"):
        li, arm = key.split("|")[0][3:], key.split("|")[1]
        return res["equilibrium"][arm][li]["label"]
    if key.startswith("width_l1|") or key.startswith("width_l2|"):
        li, arm = key.split("|")[0][6:], key.split("|")[1]
        return res["width"][arm][li]["label"]
    if key.startswith("t1_lower|"):
        return res["t1_lower"][key.split("|")[1]]["flag"]
    return res["labels"].get(key)


S8_2 = [
    ("no_rescue_worse_impaired", {}, {"adamw5": "NO_RESCUE", "adamw10": "NO_RESCUE", "box": "WD_NO_RESCUE_RELU",
                                      "adamw5_worse": True, "adamw10_impaired": True}),
    ("partial", {"arms": {"adamw5": {"late": 0.05, "early": 0.0}, "adamw10": {"late": 0.08, "early": 0.0}}},
     {"adamw5": "PARTIAL", "adamw10": "PARTIAL", "box": "WD_PARTIAL_RELU", "adamw5_worse": False,
      "adamw5_impaired": False}),
    ("rescued", {"arms": {"adamw5": {"late": 0.15, "early": 0.0}, "adamw10": {"late": 0.30, "early": 0.0}}},
     {"adamw5": "RESCUED", "adamw10": "RESCUED", "box": "WD_RESCUES_RELU"}),
    ("holm_two", {"arms": {"adamw5": {"late": 0.05, "early": 0.0, "per_seed": np.array([0.0] * 8 + [np.nan, -0.2])},
                           "adamw10": {"late": 0.05, "early": 0.0, "per_seed": np.array([0.0] * 8 + [np.nan, -0.2])}}},
     {"adamw5": "NO_RESCUE", "adamw10": "NO_RESCUE", "adamw5_worse": False, "box": "WD_NO_RESCUE_RELU"}),
    ("incomplete", {"drop": ("adamw10",)}, {"adamw5": "INCOMPLETE", "box": "INCOMPLETE"}),
    ("check_failed", {"reuse_fail": True}, {"adamw10": "CHECK_FAILED", "box": "CHECK_FAILED"}),
    ("not_reproduced", {"fresh_gap_zero": True}, {"adamw10": "NOT_REPRODUCED", "box": "NOT_REPRODUCED"}),
    ("inapplicable", {"idle2": (("adamw10", 3, 7),)}, {"adamw10": "INAPPLICABLE", "box": "INAPPLICABLE"}),
]


def s8(V) -> dict:
    """verdict.analyze1/analyze2 on synthetic data: every designed label comes out as written (box 1: the five
    labels and their three values, the box summary, DOSE's four, EQUILIBRIUM's layer and level labels, WIDTH's layer
    and level labels, DEPTH's three; box 2: RESCUED / PARTIAL / NO_RESCUE, WORSE, IMPAIRED, the box summary; and
    INAPPLICABLE / NOT_REPRODUCED / INCOMPLETE / CHECK_FAILED), the designed estimates within (100 + 50 + 10 + 4)
    eps64 max|term|, the t quantiles within the published table's rounding (5e-7), and the n = 10 sign-test p
    values equal to the exact binomial ones.  The prediction scoring runs on the designed scenario."""
    failed, per = [], {}
    bad_t = [f"t({p},{df})" for (p, df), w in T_TABLE.items() if abs(V.t_quantile(p, df) - w) > TOL_T]
    failed += bad_t
    exact = {10: 2 / 1024, 9: 22 / 1024, 8: 112 / 1024}
    for k, p in exact.items():
        d = [1.0] * k + [-1.0] * (10 - k)
        if abs(V.sign_test(d)[2] - p) > 1e-15:
            failed.append(f"sign_test({k}/10)")
    for name, scen, want, vals in S8_1:
        if name == "holm_matters":
            res = _dose_holm_scenario(V)
            want = {"dose": "DOSE_FLAT"}
        else:
            res = V.analyze1(synth1(scen))
        bad = []
        for k, w in want.items():
            g = _pick1(res, k)
            if g != w:
                bad.append(f"{k}: got {g} want {w}")
        for k, w in vals.items():
            g = _pick1(res, k)
            if isinstance(w, float) and k.startswith("E1"):
                if g is None or abs(g - w) > (100 + 50 + 10 + 4) * 2.2e-16 * 1.0:
                    bad.append(f"{k}: got {g} want {w}")
            elif isinstance(w, float):
                gg = res["equilibrium"][k.split("|")[1]]["l1"]["ratio_geomean"]
                if abs(gg - w) > 0.02:
                    bad.append(f"{k}: got {gg} want {w}")
            elif g != w:
                bad.append(f"{k}: got {g} want {w}")
        per[name] = {"labels": res["labels"], "bad": bad}
        failed += [f"b1:{name}|{b}" for b in bad]
        if name == "designed":
            r2 = V.analyze2(synth2({}))
            sc = V.score_predictions(res, r2, None)
            ok_sc = sc["labels"]["B1-3"]["p_realized"] == 0.74 and sc["labels"]["C1"]["p_realized"] == 0.91 and \
                all(np.isfinite(v["brier"]) for v in sc["labels"].values() if "brier" in v)
            if not ok_sc:
                failed.append("scoring")
    for name, scen, want in S8_2:
        res = V.analyze2(synth2(scen))
        bad = [f"{k}: got {res['labels'].get(k)} want {w}" for k, w in want.items() if res["labels"].get(k) != w]
        per[f"c51_{name}"] = {"labels": res["labels"], "bad": bad}
        failed += [f"b2:{name}|{b}" for b in bad]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S8_MUT = [
    ("M8a: WIDTH against ref's main-window slope (the parent's original)",
     [("WIDTH_LIVE = (1, 10)  ", "WIDTH_LIVE = (51, 100)")]),
    ("M8b: DOSE strictly monotone (every step UP)",
     [('    elif "DOWN" not in dirs and "UP" in dirs:\n', '    elif all(d_ == "UP" for d_ in dirs):\n')]),
    ("M8c: Holm removed", [("        if ps[i] <= alpha / (m - rank):\n", "        if ps[i] <= alpha:\n")]),
    ("M8d: one-sided sign test",
     [("    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else 1.0\n",
       "    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0) if n else 1.0\n")]),
    ("M8e: the box 2 midpoint is R", [("        c_d[a] = np.array([Wl[s] - (Rl[s] + Ll[s]) / 2.0 for s in SEEDS2])\n",
                                       "        c_d[a] = np.array([Wl[s] - Rl[s] for s in SEEDS2])\n")]),
    ("M8f: E1 measured from ref's task 1",
     [('        return deltas(arm, ep, win) - deltas("ref", ep, win)\n',
       '        return (deltas(arm, ep, win) if ep == "E2" else np.array([window_level(shards[(arm, s)]["per_task"], win)'
       ' - float(online(shards[("ref", s)]["per_task"]).loc[1]) for s in valid])) - deltas("ref", ep, win)\n')]),
    ("M8g: EQUILIBRIUM with d = 784 on the first layer",
     [('                d = float(shards[(arm, s)]["prov"]["per_seed"][str(s)]["d_eff"]) if li == 1 else D_L2\n',
       "                d = D_FULL if li == 1 else D_L2\n")]),
    ("M8h: the floor without sqrt(50)", [("    thr = F + FLOOR_K * s / math.sqrt(n)\n", "    thr = F + FLOOR_K * s\n")]),
    ("M8i: DEPTH from the unit median",
     [('        return float(np.mean([unit_mean(u["dtrain_zero_l2"][ends[t]])[0]\n',
       '        return float(np.mean([unit_median(u["dtrain_zero_l2"][ends[t]])\n')]),
]

# --------------------------------------------------------------------------
# S9: the CLI
# --------------------------------------------------------------------------


def s9(M) -> dict:
    """Box 1: for each registered arm, main(['mnist', ...]) on seed 100, 2 tasks x 2 epochs writes per_task rows
    labelled with the arm and otherwise equal to the in-process run_adamw with the registered lambda, the same
    units.npz arrays bit for bit and the same wd rows; provenance names this experiment, spec, registration commit,
    lambda, flush on, one torch thread, the checks.json hash; main() leaves flush on.  ARM_MAP, the engines' lambda
    tables and the defaults (box 1: 100 tasks x 80 epochs; box 2: 30 tasks, lr 1e-4, 2 threads, 780 updates) are the
    registered ones.  Box 2: main(['c51', ...]) equals the in-process run byte for byte.  Admission: a registered
    output is refused for a missing checks.json, a not-all_pass one, changed sources, an unregistered configuration
    (150 tasks) and an unregistered arm / seed group, and accepted for a registered one when the tree is clean."""
    per, extra = {}, {}
    E1 = load(RUNNER)
    for arm, lam in ARM_MAP_REG.items():
        out = SCR / f"s9_{arm}_{M.__name__}"
        shutil.rmtree(out, ignore_errors=True)
        torch.set_flush_denormal(False)
        try:
            M.main(["mnist", "--arm", arm, "--seeds", str(CHECK_SEED), "--tasks", "2", "--epochs", str(EP),
                    "--out", str(out)])
            flushed = _flush_is_on()
        finally:
            torch.set_flush_denormal(False)
        rows, arrays, wd, _ = run1(E1, (lam,) * 6)
        got = list(csv.DictReader((out / "per_task.csv").open()))
        ok_rows = len(got) == 2 and all(g["arm"] == arm for g in got) and \
            set(got[0]) - {"arm"} == set(rows[0]) - {"arm"} and all(
                str(r[k]) == v or float(r[k]) == float(v) for r, g in zip(rows, got) for k, v in g.items() if k != "arm")
        gw = list(csv.DictReader((out / "wd_task.csv").open()))
        ok_wd = len(gw) == 2 and all(
            str(r[k]) == v or float(r[k]) == float(v) for r, g in zip(wd, gw) for k, v in g.items() if k != "arm")
        with np.load(out / "units.npz") as z:
            ok_units = set(z.files) == {f"s{CHECK_SEED}_{k}" for k in arrays} and all(
                np.array_equal(z[f"s{CHECK_SEED}_{k}"], v, equal_nan=True) for k, v in arrays.items())
        pv = json.loads((out / "provenance.json").read_text())
        ok_prov = (pv["experiment"] == RUN and pv["spec"] == "specs/spec_adamw_dose_1010.md"
                   and pv["prereg_commit"] == "364d9cd1df7a40367a30aa16c52e9fcb0d8617dc" and pv["lam"] == lam
                   and pv.get("flush_denormal") is True and pv.get("torch_threads") == 1 and "checks_sha256" in pv
                   and {"src/adamw_dose_1010_run.py", "src/adamw_dose_1010.py"} <= set(pv["code_sha256"]))
        per[f"{arm}_rows"], per[f"{arm}_wd"], per[f"{arm}_units"] = ok_rows, ok_wd, ok_units
        per[f"{arm}_provenance"], per[f"{arm}_flush_left_on"] = ok_prov, flushed
    per["arm_map"] = M.ARM_MAP == ARM_MAP_REG and dict(M.E1.ARM_LAMBDA) == ARM_MAP_REG and \
        dict(M.E2.ARM_LAMBDA) == C51_MAP_REG and tuple(M.E1.CHECK_ARMS["ro1e-1"]) == RO_REG
    a = M.build_parser().parse_args(["mnist", "--arm", "ref", "--out", "x"])
    per["defaults_mnist"] = a.tasks == TASKS_RUN and a.epochs == EPOCHS_RUN and a.lr == LR1 and a.threads == 1 \
        and M.TASKS == TASKS_RUN
    c = M.build_parser().parse_args(["c51", "--arm", "adamw10", "--out", "x"])
    per["defaults_c51"] = c.tasks == 30 and c.lr == LR2 and c.threads == 2 and c.steps_hard == 780 and \
        c.steps_easy == 780 and c.seeds == "0-9"
    # box 2 CLI = function
    torch.set_num_threads(2)
    d_cli, d_fn = SCR / f"s9c_cli_{M.__name__}", SCR / f"s9c_fn_{M.__name__}"
    shutil.rmtree(d_cli, ignore_errors=True)
    M.main(["c51", "--arm", "adamw10", "--seeds", "100-102", "--tasks", "2", "--steps-hard", "60",
            "--steps-easy", "30", "--device", dev2().type, "--out", str(d_cli)])
    E2 = load(C51)
    shutil.rmtree(d_fn, ignore_errors=True)
    E2.run("adamw10", CHECK3, 2, dev2(), d_fn, cifar=cifar(), progress=quiet, steps_hard=60, steps_easy=30)
    per["c51_cli_equals_function"] = all(same_bytes(d_cli / f, d_fn / f) for f in ("per_task.csv", "wd_task.csv",
                                                                                     "fresh_self.csv"))
    torch.set_num_threads(1)
    # admission
    fake = SCR / "fake_checks.json"
    fake.write_text(json.dumps({"all_pass": True, **{k: {"pass": True} for k in M.REQUIRED_CHECKS},
                                "source_sha256": {s: sha(REPO / s) for s in M.SOURCES}}))
    notpass = SCR / "fake_notpass.json"
    notpass.write_text(json.dumps({"all_pass": False, **{k: {"pass": True} for k in M.REQUIRED_CHECKS},
                                   "source_sha256": {s: sha(REPO / s) for s in M.SOURCES}}))
    stale = SCR / "fake_stale.json"
    stale.write_text(json.dumps({"all_pass": True, **{k: {"pass": True} for k in M.REQUIRED_CHECKS},
                                 "source_sha256": {s: "0" * 64 for s in M.SOURCES}}))
    reg1 = dict(tasks=100, epochs=80, lr=LR1, threads=1)

    def refused(box, arm, seeds, rc, registered, cj, needle) -> bool:
        try:
            M.admit(box, arm, seeds, rc, registered, checks_json=cj)
            return False
        except SystemExit as e:
            return needle in str(e)
    per["refuse_missing_checks"] = refused("mnist", "wd3e-2", [0], reg1, M.MNIST_REGISTERED, SCR / "nope.json", "missing")
    per["refuse_not_all_pass"] = refused("mnist", "wd3e-2", [0], reg1, M.MNIST_REGISTERED, notpass, "all_pass")
    per["refuse_changed_sources"] = refused("mnist", "wd3e-2", [0], reg1, M.MNIST_REGISTERED, stale, "sources changed")
    per["refuse_150_tasks"] = refused("mnist", "wd3e-2", [0], {**reg1, "tasks": 150}, M.MNIST_REGISTERED, fake,
                                      "registered configuration")
    per["refuse_check_seed_in_place"] = refused("mnist", "wd3e-2", [100], reg1, M.MNIST_REGISTERED, fake, "registers")
    reg2 = dict(tasks=30, lr=LR2, steps_hard=780, steps_easy=780, fresh=True, graph=True, threads=2)
    per["refuse_c51_unregistered_seeds"] = refused("c51", "adamw5", CHECK10, reg2, M.E2.REGISTERED, fake, "registers")
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "analysis", "specs"],
                           capture_output=True, text=True).stdout.strip()
    if not dirty:
        try:
            M.admit("mnist", "wd3e-2", [0], reg1, M.MNIST_REGISTERED, checks_json=fake)
            per["accept_registered_when_clean"] = True
        except SystemExit:
            per["accept_registered_when_clean"] = False
    extra["tree_clean_at_check"] = not dirty
    return result(per, extra)


S9_MUT = [
    ("M9a: the lambdas of wd3e-2 and wd1e-1 swapped", [('"wd3e-2": 3e-2, "wd1e-1": 1e-1}', '"wd3e-2": 1e-1, "wd1e-1": 3e-2}')]),
    ("M9b: the default horizon is 150 tasks", [("TASKS = 100\n", "TASKS = 150\n")]),
    ("M9c: denormals not flushed", [("    torch.set_flush_denormal(True)\n", "    torch.set_flush_denormal(False)\n")]),
]

# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

PROBE = 6


def s_cost() -> None:
    """Box 1: PROBE wd1e-1 CLI processes at once (seeds 100-105, 2 tasks x 80 epochs, one thread each).  Gate: every
    process exits 0.  Reported: peak RSS, the slots that fit now leaving 6 GiB free with 20% headroom per job, the
    per-update time under PROBE-fold load and its 100-task projection.  Box 2: adamw10 seeds 100-109, 3 tasks x 780
    updates through the CLI: ms per update, peak RSS and VRAM."""
    t0 = time.time()
    avail0 = _mem_available_gib()
    SCR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(PROBE):
        d = SCR / f"cost_{i}"
        shutil.rmtree(d, ignore_errors=True)
        procs.append((d, subprocess.Popen(
            [PY, "-m", "src.adamw_dose_1010_run", "mnist", "--arm", "wd1e-1", "--seeds", str(100 + i), "--tasks", "2",
             "--out", str(d)], cwd=REPO, env=THREAD_ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
    rc, peak, secs = [], [], []
    for d, pr in procs:
        err = pr.communicate()[1]
        rc.append(pr.returncode)
        if pr.returncode == 0:
            pv = json.loads((d / "provenance.json").read_text())
            peak.append(pv["peak_rss_kb"] / 1024 ** 2)
            secs.append(pv["wall_clock_s"])
        else:
            print(err.decode()[-400:], flush=True)
    per_update = (max(secs) / (2 * STEPS_RUN)) if secs else float("nan")
    slots = int((avail0 - 6.0) // (1.2 * max(peak))) if peak else 0
    one = TASKS_RUN * STEPS_RUN * per_update
    d2 = SCR / "cost_c51"
    shutil.rmtree(d2, ignore_errors=True)
    r2 = subprocess.run([PY, "-m", "src.adamw_dose_1010_run", "c51", "--arm", "adamw10", "--seeds", "100-109",
                         "--tasks", "3", "--out", str(d2)], cwd=REPO, capture_output=True, text=True)
    c51 = {}
    if r2.returncode == 0:
        pv = json.loads((d2 / "provenance.json").read_text())
        c51 = {"ms_per_update": pv["step_ms_last_task"], "peak_rss_gib": pv["peak_rss_kb"] / 1024 ** 2,
               "peak_vram_gib": (pv["peak_vram_bytes"] or 0) / 1024 ** 3, "wall_clock_s": pv["wall_clock_s"]}
    ok = all(r == 0 for r in rc) and len(peak) == PROBE and r2.returncode == 0
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": f"all {PROBE} box 1 processes and the box 2 run exit 0",
                         "returncodes": rc, "c51_returncode": r2.returncode, "mem_available_gib_before": avail0,
                         "peak_rss_gib_max": max(peak) if peak else None, "slots_now": slots,
                         "seconds_per_update_under_load": per_update,
                         "one_trajectory_min_projected": one / 60,
                         "grid_50_wall_hours_at_6": 50 * one / 3600 / PROBE, "c51": c51,
                         "measured_at": dt.datetime.now().astimezone().isoformat(),
                         "seconds": round(time.time() - t0, 1)}
    dump()
    c = RESULTS["S-cost"]
    print(f"S-cost: pass={ok}  {per_update * 1e3:.3f} ms/update under {PROBE} -> {c['one_trajectory_min_projected']:.1f} "
          f"min per trajectory (peak RSS {c['peak_rss_gib_max']}), c51 {c51}", flush=True)


# --------------------------------------------------------------------------

CHECKS = {
    "S0": ("S0 the lambda = 0 path reproduces the recorded l2cap ref shards", s0, S0_MUT, s0.__doc__, RUNNER, 1),
    "S-wd": ("S-wd decoupled on synthetic tensors (box 1)", s_wd, SWD_MUT, s_wd.__doc__, RUNNER, 1),
    "S-torch": ("S-torch torch.optim.AdamW's formula within the rounding band (box 1)", s_torch, STORCH_MUT,
                s_torch.__doc__ + RegBand.__doc__ + Band.__doc__, RUNNER, 1),
    "S-deadpix": ("S-deadpix the in-run record of the decay", s_deadpix, SDP_MUT, s_deadpix.__doc__, RUNNER, 1),
    "S5": ("S5 the arms share their streams; the first update; all differ", s5, S5_MUT, s5.__doc__, RUNNER, 1),
    "S-ctrl": ("S-ctrl readout-only decay leaves the hidden widths like ref's", s_ctrl, SCTRL_MUT, s_ctrl.__doc__,
               RUNNER, 1),
    "S7": ("S7 the same call twice gives the same bits", s7, S7_MUT, s7.__doc__, RUNNER, 1),
    "S-wd-c51": ("S-wd decoupled on synthetic tensors (box 2)", s_wd_c51, SWD2_MUT, s_wd_c51.__doc__, C51, 2),
    "S-torch-c51": ("S-torch torch.optim.AdamW's formula within the rounding band (box 2)", s_torch_c51, STORCH2_MUT,
                    s_torch_c51.__doc__ + RegBand.__doc__ + Band.__doc__, C51, 2),
    "S-nochange": ("S-nochange the lambda = 0 path is the host's R", s_nochange, NOCHANGE_MUT, s_nochange.__doc__, C51, 2),
    "S-graph": ("S-graph CUDA graph = eager for adamw10", s_graph, GRAPH_MUT, s_graph.__doc__, C51, 2),
    "S-fresh": ("S-fresh the fresh controls", s_fresh, FRESH_MUT, s_fresh.__doc__, C51, 2),
    "S8": ("S8 the verdict returns the designed labels on synthetic data", s8, S8_MUT, s8.__doc__, VERDICT, 1),
    "S9": ("S9 the CLI maps, records and admits", s9, S9_MUT, s9.__doc__, CLI, 1),
}
ORDER = ("S0", "S-wd", "S-torch", "S-deadpix", "S5", "S-ctrl", "S7", "S-wd-c51", "S-torch-c51", "S-nochange",
         "S-host-repro", "S-graph", "S-fresh", "S8", "S9", "S-cost")


def main() -> None:
    global DUMP_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated check keys; 'S-cost' for the cost probe")
    args = ap.parse_args()
    only = [k for k in args.only.split(",") if k]
    if only:
        SCR.mkdir(parents=True, exist_ok=True)
        DUMP_PATH = SCR / f"checks_partial_{'_'.join(only)}.json"
    SCR.mkdir(parents=True, exist_ok=True)
    RESULTS.update({"run_id": RUN, "started_at": dt.datetime.now().astimezone().isoformat(),
                    "spec": "specs/spec_adamw_dose_1010.md", "prereg_commit": "364d9cd1df7a40367a30aa16c52e9fcb0d8617dc",
                    "git_head": H.git_hash(),
                    "source_sha256": {s: sha(REPO / s) for s in SOURCES},
                    "verdict_sha256": sha(VERDICT),
                    "torch": torch.__version__, "python": sys.version.split()[0]})
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
    RESULTS["finished_at"] = dt.datetime.now().astimezone().isoformat()
    dump()
    print(f"all_pass = {RESULTS['all_pass']}  -> {DUMP_PATH}")


if __name__ == "__main__":
    main()

"""Checks for l2cap_ee_0917 (specs/spec_l2cap_ee_0917.md section 6).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/l2cap_ee_0917/checks.py
    ... --only S4            # development: writes results/_checks_l2cap_ee_0917/checks_partial.json

The box is src/mucap_el_run_0916.py's run_one with act2_name="ELU1" and the new w2cap / bias_fix
switches, driven by src/l2cap_ee_run_0917.py.  Each check runs on the real code and then on every
mutation listed with it; a mutation is an exact-once string substitution in the check's target file and
it MUST make the same check fail.  Tolerances come from the derivations in the docstrings.
Writes results/l2cap_ee_0917/checks.json after every step.

S0   the shared runner's EL path still reproduces the recorded 0916 shards (as in mucap_ee_0917)
S0b  its EE path with both new switches off reproduces the recorded mucap_ee_0917 ref and cap_both
S1-S3  the first-layer caps (src/mucap_el_0916.py unchanged): the 0916 record, re-verified by hash
S2c  the second-layer row-norm cap (src/l2cap_ee_0917.py)
S4   EE wiring: captured cap12_bfix updates are the host ELU->ELU net's update + all caps + the bias
     restore; the training-derivative diagnostics are autograd's, 0 below z = -16.64
S5   branch: the five arms share task 1; every radius and held bias is the task-1-end value
S7   the same call twice gives the same bits
S8   the verdict on synthetic shards
S9   the CLI maps each arm, relabels the rows and writes this experiment's provenance
S-cost  PROBE processes at once
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
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
from src import elu_growth_0909 as EG            # noqa: E402
from src import mucap_el_0916 as MU              # noqa: E402

EL_RUNNER = REPO / "src" / "mucap_el_run_0916.py"
L2_RUNNER = REPO / "src" / "l2cap_ee_run_0917.py"
W2CAP = REPO / "src" / "l2cap_ee_0917.py"
VERDICT = REPO / "analysis" / "l2cap_ee_0917" / "verdict.py"
EL_CHECKS_JSON = REPO / "results" / "mucap_el_0916" / "checks.json"
OUT = REPO / "results" / "l2cap_ee_0917"
SCR = REPO / "results" / "_checks_l2cap_ee_0917"
EL_RECORD = REPO / "results" / "mucap_el_0916"
EL_ARCHIVE_MANIFEST = EL_RECORD / "backup_manifest.json"
EE_RECORD = REPO / "results" / "mucap_ee_0917"
EE_ARCHIVE_MANIFEST = EE_RECORD / "backup_manifest.json"
EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)
DEV = H.setup("cpu")
MNIST = H.Mnist(DEV)
PROBE = 6
PY = sys.executable
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
TASKS_RUN, STEPS_RUN = 100, 6000
RESULTS: dict = {}
DUMP_PATH = OUT / "checks.json"
# the registered arm map (spec 2.2); S9 checks the CLI's is this
ARM_MAP_REG = {"ref": ("ref", False, False), "cap1": ("cap_both", False, False), "cap2": ("ref", True, False),
               "cap12": ("cap_both", True, False), "cap12_bfix": ("cap_both", True, True)}

# --------------------------------------------------------------------------
# infrastructure (analysis/mucap_el_0916/checks.py's, unchanged in behaviour)
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


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    return float("nan")


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
    base = fn(load(target))
    entry = {"title": title, "target": str(target.relative_to(REPO)), "threshold_derivation": derivation,
             **base, "mutations": []}
    for label, subs in mutations:
        mod = load(target, subs)
        try:
            r = fn(mod)
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": bool(r["pass"]),
                                       "detected": not r["pass"], "mutant_result": brief(r)})
        except Exception as e:          # recorded, but a crash is not how a check should notice a defect
            entry["mutations"].append({"mutation": label, "check_pass_on_mutant": False, "detected": True,
                                       "raised": repr(e)[:400]})
    entry["all_mutations_detected"] = bool(entry["mutations"]) and all(m["detected"] for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:3]) if not entry['pass'] else ''}", flush=True)


class flush_denormal:
    """The runner sets torch.set_flush_denormal(True) (spec 2.1); in-process EE runs here do the same
    and restore the previous state, so S0's EL path runs exactly as the 0916 shards did."""
    def __enter__(self):
        torch.set_flush_denormal(True)

    def __exit__(self, *a):
        torch.set_flush_denormal(False)


def _flush_is_on() -> bool:
    """torch has no getter: a float32 product that lands in the subnormal range is 0 only when flushing."""
    return float(torch.tensor([1e-30], dtype=torch.float32) * torch.tensor([1e-10], dtype=torch.float32)) == 0.0


def _images(seed: int) -> torch.Tensor:
    return MNIST.train_x[RL.subset_idx(seed).to(MNIST.train_x.device)]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# S0: the EL path is unchanged (tasks 1-2 of two recorded shards, full 80-epoch tasks)
# --------------------------------------------------------------------------

R_DEFAULT = "    if act2_name is not None:\n"
R_DEFAULT_SET = "        act2 = ACT2[act2_name]\n"
R_LOWGATE = "LOWGATE = 0.05 "
R_MEMO = "            memo = float((forward2(params, x, act1, act2)[4].argmax(1) == y).float().mean())\n"
EL_SHARDS = (("ref", 0), ("cap_both", 0))
S0_TASKS = 2


def _archived(shard: str, name: str) -> Path | None:
    """The 0916 shard's file in obsidian-research-data, used only if its sha256 matches the manifest."""
    man = json.loads(EL_ARCHIVE_MANIFEST.read_text())
    for f in man["files"]:
        if f["source_rel"] == f"results/mucap_el_0916/runs/{shard}/{name}":
            p = Path(f["backup"])
            return p if p.exists() and _sha(p) == f["sha256"] else None
    return None


def s0(M) -> dict:
    """The shared runner's default path (act2_name=None) is the 0916 EL box: for tasks 1-2 of the
    recorded shards ref_s0 and cap_both_s0 (80 epochs, ledger on), (i) every per_task.csv column the 0916
    shard has, except memo_acc, is equal as written (csv round-trips floats exactly), (ii) every
    units.npz and ledger.npz array the archived 0916 shard has is bit-equal on the task 1-2 rows (the
    archive copy is used only when its sha256 matches backup_manifest.json), (iii) the task-1-end state
    hash equals the 0916 provenance's.  (iv) memo_acc, which 0916 computed on an ELU->ELU copy of the
    net, now equals the accuracy of the trained ELU->leaky net recomputed from the task-1-end weights,
    and the recorded 0916 value equals the ELU->ELU copy's (the bug was what this says it was).
    (v) the new column major_frac is the largest label count / 1200.  Bit and string comparisons only."""
    failed, per = [], {}
    elu, lr = EG.ELU(1.0), H.ARMS["LR"]
    for arm, seed in EL_SHARDS:
        shard = f"{arm}_s{seed}"
        d = {}
        rows, arrays, led, info = M.run_one(arm, seed, 1e-3, S0_TASKS, MNIST, DEV, epochs=80, ledger=True,
                                            debug=d)
        rec = [r for r in csv.DictReader((EL_RECORD / "runs" / shard / "per_task.csv").open())
               if int(r["task"]) <= S0_TASKS]
        prov = json.loads((EL_RECORD / "runs" / shard / "provenance.json").read_text())
        bad_cols, n_cols = [], 0
        for rr, got in zip(rec, rows):
            for k, v in rr.items():
                if k in ("arm", "memo_acc"):
                    continue
                n_cols += 1
                g = got.get(k)
                same = (str(g) == v) or (g is not None and v != "" and float(v) == float(g))
                if not same:
                    bad_cols.append(f"t{rr['task']}:{k}")
        arr_bad, n_arr = [], 0
        u_path, l_path = _archived(shard, "units.npz"), _archived(shard, "ledger.npz")
        if u_path is not None:
            with np.load(u_path) as z:
                pre = f"s{seed}_"
                tk = z[pre + "task"]
                keep = tk <= S0_TASKS
                for k in z.files:
                    kk = k[len(pre):]
                    if kk in ("q_cap", "v_cap"):
                        a_rec, a_got = z[k], arrays[kk]
                    elif kk.startswith("test_"):           # the test set is read at task 1 only here
                        a_got = arrays[kk]
                        a_rec = z[k][:a_got.shape[0]]
                    else:
                        a_rec, a_got = z[k][keep], arrays[kk]
                    n_arr += 1
                    if not np.array_equal(a_rec, a_got, equal_nan=True):
                        arr_bad.append(kk)
        if l_path is not None:
            with np.load(l_path) as z:
                pre = f"s{seed}_"
                keep = z[pre + "task"] <= S0_TASKS
                for k in z.files:
                    n_arr += 1
                    if not np.array_equal(z[k][keep], led[k[len(pre):]], equal_nan=True):
                        arr_bad.append("ledger:" + k[len(pre):])
        p1 = d["task1_end_params"]
        x = _images(seed)
        y1 = d["labels"][0]
        true_acc = float((M.forward2(p1, x, elu, lr)[4].argmax(1) == y1).float().mean())
        ee_acc = float((H.forward(p1, x, elu)[4].argmax(1) == y1).float().mean())
        maj = float(torch.bincount(y1, minlength=10).max()) / 1200
        ok = {"i_per_task_columns": not bad_cols and n_cols >= 60,
              "ii_arrays": u_path is not None and l_path is not None and not arr_bad and n_arr >= 60,
              "iii_task1_state": info["task1_end_state_sha256"] == prov["per_seed"][str(seed)]["task1_end_state_sha256"],
              "iv_memo_is_the_net": rows[0]["memo_acc"] == true_acc,
              "iv_old_memo_was_ee_copy": float(rec[0]["memo_acc"]) == ee_acc and ee_acc != true_acc,
              "v_major_frac": rows[0]["major_frac"] == maj,
              "nonvacuous": arm == "ref" or rows[1]["rows_par"] > 0}
        per[shard] = {**ok, "bad_columns": bad_cols[:10], "n_columns": n_cols, "bad_arrays": arr_bad[:10],
                      "n_arrays": n_arr, "archive_used": [str(u_path), str(l_path)],
                      "memo_recorded_0916": float(rec[0]["memo_acc"]), "memo_now": rows[0]["memo_acc"],
                      "acc_elu_leaky": true_acc, "acc_elu_elu_copy": ee_acc}
        failed += [f"{shard}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S0_MUT = [
    ("M0a: the default path runs ELU on the second layer",
     [(R_DEFAULT, "    if True:\n"), (R_DEFAULT_SET, '        act2 = ACT2[act2_name or "ELU1"]\n')]),
    ("M0b: a diagnostic constant moved (low-gate mark 0.06)", [(R_LOWGATE, "LOWGATE = 0.06 ")]),
    ("M0c: memo_acc back on the ELU->ELU copy (the 0916 bug)",
     [(R_MEMO, "            memo = RL.evaluate_rl(params, x, y, act1)[\"acc\"]\n")]),
]


# --------------------------------------------------------------------------
# S0b: the EE path with the new switches off is mucap_ee_0917's
# --------------------------------------------------------------------------

R_L2ON = "        l2_on = t >= 2 and w2cap\n"
R_FIXON = "        fix_on = t >= 2 and bias_fix\n"
EE_SHARDS = (("ref", 0), ("cap_both", 0))


def _archived_ee(shard: str, name: str) -> Path | None:
    man = json.loads(EE_ARCHIVE_MANIFEST.read_text())
    for f in man["files"]:
        if f["source_rel"] == f"results/mucap_ee_0917/runs/{shard}/{name}":
            p = Path(f["backup"])
            return p if p.exists() and _sha(p) == f["sha256"] else None
    return None


def s0b(M) -> dict:
    """run_one(act2_name="ELU1") with w2cap and bias_fix off, flush on, 80 epochs, tasks 1-2, reproduces
    the recorded mucap_ee_0917 shards ref_s0 and cap_both_s0: every per_task.csv column that shard has is
    equal as written, every units.npz array it has is bit-equal on the task 1-2 rows (test arrays: the
    task-1 row; the archive copy only if its sha256 matches the manifest), the task-1-end state hash is
    the recorded one, and the new projection tallies are zero.  String and bit comparisons only."""
    failed, per = [], {}
    for arm, seed in EE_SHARDS:
        shard = f"{arm}_s{seed}"
        with flush_denormal():
            rows, arrays, _, info = M.run_one(arm, seed, 1e-3, S0_TASKS, MNIST, DEV, epochs=80, ledger=False,
                                              act2_name="ELU1")
        rec = [r for r in csv.DictReader((EE_RECORD / "runs" / shard / "per_task.csv").open())
               if int(r["task"]) <= S0_TASKS]
        prov = json.loads((EE_RECORD / "runs" / shard / "provenance.json").read_text())
        bad_cols, n_cols = [], 0
        for rr, got in zip(rec, rows):
            for k, v in rr.items():
                if k == "arm":
                    continue
                n_cols += 1
                g = got.get(k)
                if not ((str(g) == v) or (g is not None and v != "" and float(v) == float(g))):
                    bad_cols.append(f"t{rr['task']}:{k}")
        arr_bad, n_arr = [], 0
        u_path = _archived_ee(shard, "units.npz")
        if u_path is not None:
            with np.load(u_path) as z:
                pre = f"s{seed}_"
                keep = z[pre + "task"] <= S0_TASKS
                for k in z.files:
                    kk = k[len(pre):]
                    got = arrays[kk]
                    rec_a = z[k] if kk in ("q_cap", "v_cap") else (z[k][:got.shape[0]] if kk.startswith("test_")
                                                                    else z[k][keep])
                    n_arr += 1
                    if not np.array_equal(rec_a, got, equal_nan=True):
                        arr_bad.append(kk)
        ok = {"i_per_task_columns": not bad_cols and n_cols >= 70,
              "ii_arrays": u_path is not None and not arr_bad and n_arr >= 70,
              "iii_task1_state": info["task1_end_state_sha256"] == prov["per_seed"][str(seed)]["task1_end_state_sha256"],
              "iv_new_tallies_zero": all(r["rows_w2"] == 0 and r["rows_bfix"] == 0 for r in rows),
              "nonvacuous": arm == "ref" or rows[1]["rows_perp"] > 0}
        per[shard] = {**ok, "bad_columns": bad_cols[:10], "n_columns": n_cols, "bad_arrays": arr_bad[:10],
                      "n_arrays": n_arr, "archive": str(u_path)}
        failed += [f"{shard}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S0B_MUT = [
    ("M0d: the second-layer cap is on whatever w2cap says", [(R_L2ON, "        l2_on = t >= 2\n")]),
    ("M0e: the biases are held whatever bias_fix says", [(R_FIXON, "        fix_on = t >= 2\n")]),
    ("M0f: a diagnostic changed on the EE path (units read the second layer through leaky)",
     [("        u = unit_arrays(params, x, act1, act2, e1_64) | adam_arrays(params, adam)\n",
       '        u = unit_arrays(params, x, act1, H.ARMS["LR"], e1_64) | adam_arrays(params, adam)\n')]),
]


# --------------------------------------------------------------------------
# S1-S3: the caps (unchanged file) -- copied from the 0916 record, re-verified by hash
# --------------------------------------------------------------------------

def s1_3_record() -> None:
    rec = json.loads(EL_CHECKS_JSON.read_text())
    want = rec["code_sha256"]["src/mucap_el_0916.py"]
    now = hashlib.sha256((REPO / "src" / "mucap_el_0916.py").read_bytes()).hexdigest()
    for k in ("S1_S-mu-basis", "S2_S-mu-projection", "S3_S-mu-order"):
        e = rec[k]
        RESULTS[k] = {"title": e["title"] + " (0916 record; src/mucap_el_0916.py unchanged)",
                      "pass": bool(e["pass"] and e["all_mutations_detected"] and want == now),
                      "all_mutations_detected": e["all_mutations_detected"],
                      "record": "results/mucap_el_0916/checks.json", "record_sha256_of_caps": want,
                      "caps_sha256_now": now,
                      "mutations_detected": f"{sum(m['detected'] for m in e['mutations'])}/{len(e['mutations'])}"}
    dump()
    print(f"S1-S3: copied from the 0916 record, caps file unchanged = {want == now}", flush=True)


# --------------------------------------------------------------------------
# S2c: the row-norm cap
# --------------------------------------------------------------------------

C_OVER = "    over = n > r\n"
C_WRITE = "    W.copy_(torch.where(over, W * (r / safe), W))\n"
C_REM = "    return int(over.sum()), float((n - r).clamp(min=0).sum())\n"
C_NORM = "    return torch.linalg.vector_norm(W, dim=1, keepdim=True).detach().clone()\n"
TOL_RAD = 4.0        # |  ||w'|| - r | <= TOL_RAD * eps32 * r : one float32 scale and one norm (100 squares
                     # summed then sqrt) each round by ~eps32 relative; 4 covers both with margin 2
TOL_REM = 1e-5       # removed-norm tally: a float64 sum of <= 100 float32 differences, each ~eps32*|n|


def s2c(C) -> dict:
    """On real second-layer rows (task-1-end-like: init scaled by 1.5) with radii from row_norms of the
    unscaled rows, plus hand-made rows: (i) rows at or below the radius keep every bit; (ii) a row above
    it ends with norm r within TOL_RAD*eps32*r and the same direction (cosine 1 within 1e-6); (iii) the
    tally counts exactly the rows above and the removed norm matches sum(n - r)+ within TOL_REM; (iv) a
    zero row with radius 0 and a row equal to its radius stay bit-identical and no NaN appears; (v)
    row_norms is the float norm of each row; (vi) a second application writes no row by more than the
    rounding bound (projection is a fixed point up to TOL_RAD).  Non-vacuity: some rows over, some not."""
    failed, per = [], {}
    g = torch.Generator().manual_seed(917)
    W0 = H.init_params(0, DEV)[2].detach().clone()
    r = C.row_norms(W0)
    per["v_row_norms"] = bool(torch.equal(r, torch.linalg.vector_norm(W0, dim=1, keepdim=True)))
    W = W0.clone()
    scale = torch.where(torch.rand(W.shape[0], 1, generator=g) < 0.5, torch.tensor(1.5), torch.tensor(0.7))
    W = W * scale
    W[0] = 0.0
    r_used = r.clone()
    r_used[0] = 0.0
    W[1] = W0[1]                                         # exactly at its radius
    scale[1] = 1.0
    before = W.clone()
    n_before = torch.linalg.vector_norm(before, dim=1, keepdim=True)
    over_want = (n_before > r_used)[:, 0]
    rows, rem = C.cap_row_norm_(W, r_used)
    n_after = torch.linalg.vector_norm(W, dim=1)
    under = ~over_want
    per["i_under_rows_bit_identical"] = bool(torch.equal(W[under], before[under]))
    rel = ((n_after[over_want].double() - r_used[over_want, 0].double()).abs() / r_used[over_want, 0].double())
    per["ii_norm_at_radius_max_rel"] = float(rel.max())
    per["ii_norm_at_radius"] = bool(per["ii_norm_at_radius_max_rel"] <= TOL_RAD * EPS32)
    cos = (W[over_want].double() * before[over_want].double()).sum(1) / (
        n_after[over_want].double() * n_before[over_want, 0].double())
    per["ii_direction_kept"] = bool(((cos - 1).abs() <= 1e-6).all())
    want_rem = float((n_before.double() - r_used.double()).clamp(min=0).sum())
    per["iii_tally_rows"] = rows == int(over_want.sum())
    per["iii_tally_removed"] = abs(rem - want_rem) <= TOL_REM * max(1.0, want_rem)
    per["iv_zero_and_equal_rows"] = bool(torch.equal(W[0], before[0]) and torch.equal(W[1], before[1])
                                         and torch.isfinite(W).all())
    again = W.clone()
    C.cap_row_norm_(again, r_used)
    n2 = torch.linalg.vector_norm(again, dim=1)
    per["vi_second_application_bound"] = bool((((n2.double() - r_used[:, 0].double()).clamp(min=0))
                                               <= TOL_RAD * EPS32 * r_used[:, 0].double() + 1e-30).all())
    per["nonvacuous"] = bool(over_want.any() and under.any())
    keys = ("v_row_norms", "i_under_rows_bit_identical", "ii_norm_at_radius", "ii_direction_kept",
            "iii_tally_rows", "iii_tally_removed", "iv_zero_and_equal_rows", "vi_second_application_bound",
            "nonvacuous")
    failed += [k for k in keys if not per[k]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S2C_MUT = [
    ("M2a: rows at the radius are rewritten (>= instead of >)", [(C_OVER, "    over = n >= r\n")]),
    ("M2b: every row is scaled, including those below", [(C_OVER, "    over = n >= 0\n")]),
    ("M2c: rows are shrunk to half the radius", [(C_WRITE, "    W.copy_(torch.where(over, W * (0.5 * r / safe), W))\n")]),
    ("M2d: the removed norm is not clamped at zero", [(C_REM, "    return int(over.sum()), float((n - r).sum())\n")]),
    ("M2e: the radius is the squared norm", [(C_NORM, "    return (torch.linalg.vector_norm(W, dim=1, keepdim=True) ** 2).detach().clone()\n")]),
]


# --------------------------------------------------------------------------
# S4: EE wiring with every switch on, and the training derivative
# --------------------------------------------------------------------------

R_UNITS = "        u = unit_arrays(params, x, act1, act2, e1_64)\n"
R_SNAP = "        u = unit_arrays(params, x, act1, act2, e1_64) | adam_arrays(params, adam)\n"
R_W2CALL = "                        nr, rem = W2C.cap_row_norm_(params[2], r2)\n"
R_BRESTORE = "                        params[1].copy_(b_star[0])\n"
R_DTRAIN = "            gt, = torch.autograd.grad(act.phi(zz).sum(), zz)\n"
EP = 2               # 150 updates per task in the runner checks; the properties are per update
TOL_ELU = 1e-3
CAPTURE = ((1, 37), (2, 5), (2, 149), (3, 90))
ZERO_Z32 = math.log(2.0 ** -24)


def _run(M, arm, seed=0, tasks=1, debug=None, epochs=EP):
    a1, w2, bf = ARM_MAP_REG[arm]
    with flush_denormal():
        return M.run_one(a1, seed, 1e-3, tasks, MNIST, DEV, epochs=epochs, ledger=False, debug=debug,
                         act2_name="ELU1", w2cap=w2, bias_fix=bf)


def _host_update(st: dict, e1: torch.Tensor, act, w2=True, bf=True) -> list:
    """One update recomputed from the captured state with the host's single-activation forward, the
    runner's Adam arithmetic, the first-layer caps, the row-norm cap and the bias restore."""
    from src import l2cap_ee_0917 as C
    with flush_denormal():
        p = [q.clone().requires_grad_(True) for q in st["before"]["params"]]
        m = [q.clone() for q in st["before"]["m"]]
        v = [q.clone() for q in st["before"]["v"]]
        loss = torch.nn.functional.cross_entropy(H.forward(p, st["xb"], act)[4], st["yb"])
        grads = torch.autograd.grad(loss, p)
        tc = st["before"]["tc"] + 1
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-3
        c1, c2 = 1 - b1 ** tc, 1 - b2 ** tc
        with torch.no_grad():
            p = [q.detach().clone() for q in p]
            for q, g, mi, vi in zip(p, grads, m, v):
                mi.mul_(b1).add_(g, alpha=1 - b1)
                vi.mul_(b2).addcmul_(g, g, value=1 - b2)
                q -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
            if st["cap_on"]:
                MU.cap_parallel_(p[0], e1, st["q_cap"])
                MU.cap_perp_(p[0], e1, st["v_cap"])
            if st["l2_on"] and w2:
                C.cap_row_norm_(p[2], st["r2"])
            if st["fix_on"] and bf:
                p[1].copy_(st["b_star"][0])
                p[3].copy_(st["b_star"][1])
    return p + m + v


def s4(M) -> dict:
    """(i) forward2(ELU, ELU) is the host's forward.  (ii) Four captured updates of a cap12_bfix run (one
    in task 1, three with every switch on) are bit-identical to the host ELU->ELU update followed by the
    first-layer caps, the second-layer row-norm cap and the bias restore; the same recomputation without
    the row-norm cap, without the bias restore, or with leaky, gives different bits (non-vacuity).
    (iii) The task-1-end units.npz and per_task diagnostics of the second layer are the host's.  (iv) The
    training-derivative diagnostics are autograd through the host ELU on the host z (bit for bit), equal
    expm1(min(z,0))+1 on a grid over [-45, 5] in float32, are exactly 0 just below ln(2^-24) and positive
    just above, and differ from the analytic exp gate there.  (v) memo_acc is the host's accuracy.
    Bit comparisons, one tolerance: TOL_ELU (the leaky identity must be missed)."""
    failed, per = [], {}
    elu, lr = EG.ELU(1.0), H.ARMS["LR"]
    p0 = [t.detach() for t in H.init_params(0, DEV)]
    x = _images(0)
    per["i_forward2_is_host_ELU"] = all(bool((a == b).all()) for a, b in
                                        zip(M.forward2(p0, x, elu, elu), H.forward(p0, x, elu)))
    d = {"capture": CAPTURE}
    rows, arrays, _, _ = _run(M, "cap12_bfix", tasks=3, debug=d)
    e1 = d["e1"]
    upd = {}
    for key in CAPTURE:
        st = d["steps"][key]
        got = st["after"]["params"] + st["after"]["m"] + st["after"]["v"]
        same = lambda w: all(bool((a == b).all()) for a, b in zip(got, w))
        upd[str(key)] = {"equals_host": same(_host_update(st, e1, elu)),
                         "switch_on": bool(st["l2_on"] and st["fix_on"]),
                         "differs_without_w2cap": not same(_host_update(st, e1, elu, w2=False)),
                         "differs_without_bfix": not same(_host_update(st, e1, elu, bf=False)),
                         "differs_with_leaky": not same(_host_update(st, e1, lr))}
    on = [u for u in upd.values() if u["switch_on"]]
    per["ii_updates"] = upd
    per["ii_update_is_host"] = all(u["equals_host"] for u in upd.values())
    per["ii_nonvacuous"] = len(on) == 3 and all(u["differs_without_w2cap"] and u["differs_without_bfix"]
                                                 and u["differs_with_leaky"] for u in on)
    p1 = d["task1_end_params"]
    i1 = int(np.where((arrays["task"] == 1) & (arrays["step"] == M.STEPS_PER_EPOCH * EP))[0][0])
    with flush_denormal():
        z1, a1, z2, a2, lg = H.forward(p1, x, elu)
        g2 = elu.dphi(z2).double().mean(0).numpy()
        zz = z2.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            gt, = torch.autograd.grad(elu.phi(zz).sum(), zz)
    per["iii_zbar_l2_is_host"] = bool(np.array_equal(arrays["zbar_l2"][i1], z2.double().mean(0).numpy()))
    per["iii_gate_l2_is_host_elu"] = bool(np.array_equal(arrays["gate_mean_l2"][i1], g2))
    a32 = float(torch.tensor(lr.param, dtype=torch.float32))
    per["iii_not_leaky"] = float(np.abs(arrays["gate_mean_l2"][i1] - (a32 + (1 - a32) * arrays["pplus_l2"][i1])).max()) > TOL_ELU
    per["iii_row_median_is_host_elu"] = rows[0]["med_gate_mean_l2"] == float(np.nanmedian(g2))
    per["iv_dtrain_is_autograd"] = bool(np.array_equal(arrays["dtrain_mean_l2"][i1], gt.double().mean(0).numpy())
                                        and np.array_equal(arrays["dtrain_zero_l2"][i1], (gt == 0).double().mean(0).numpy()))
    grid = torch.linspace(-45, 5, 20001, dtype=torch.float32)
    with torch.enable_grad():
        gz = grid.clone().requires_grad_(True)
        gg, = torch.autograd.grad(elu.phi(gz).sum(), gz)
    formula = torch.where(grid > 0, torch.ones_like(grid), torch.expm1(grid.clamp(max=0.0)) + 1)
    edge = torch.tensor([ZERO_Z32 - 1e-4, ZERO_Z32 + 1e-4], dtype=torch.float32)
    with torch.enable_grad():
        ez = edge.clone().requires_grad_(True)
        eg, = torch.autograd.grad(elu.phi(ez).sum(), ez)
    per["iv_grid_equals_expm1_plus_1"] = bool(torch.equal(gg, formula))
    per["iv_zero_below_threshold"] = float(eg[0]) == 0.0 and float(eg[1]) > 0.0
    per["iv_differs_from_exp_gate"] = bool((gg != elu.dphi(grid)).any())
    # the runner's own diagnostic on a synthetic deep z: dtrain_zero must see the zeros exp does not
    fake = M.unit_arrays([torch.zeros(100, 784), torch.full((100,), -20.0), p1[2], p1[3], p1[4], p1[5]],
                         x[:32], elu, elu, MU.mu_basis(x)[1])
    per["iv_runner_sees_zero_derivative"] = bool((fake["dtrain_zero_l1"] == 1.0).all() and (fake["gate_mean_l1"] > 0).all())
    per["v_memo_is_host_elu"] = rows[0]["memo_acc"] == float((lg.argmax(1) == d["labels"][0]).float().mean())
    keys = ("i_forward2_is_host_ELU", "ii_update_is_host", "ii_nonvacuous", "iii_zbar_l2_is_host",
            "iii_gate_l2_is_host_elu", "iii_not_leaky", "iii_row_median_is_host_elu", "iv_dtrain_is_autograd",
            "iv_grid_equals_expm1_plus_1", "iv_zero_below_threshold", "iv_differs_from_exp_gate",
            "iv_runner_sees_zero_derivative", "v_memo_is_host_elu")
    failed += [k for k in keys if not per[k]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S4_MUT = [
    ("M4a: act2_name ignored (trains ELU->leaky)", [(R_DEFAULT, "    if False:\n")]),
    ("M4b: ACT2['ELU1'] is the leaky object",
     [('ACT2 = {"LR": H.ARMS["LR"], "ELU1": EG.ELU(1.0)}\n', 'ACT2 = {"LR": H.ARMS["LR"], "ELU1": H.ARMS["LR"]}\n')]),
    ("M4c: units.npz reads the second layer through leaky",
     [(R_SNAP, '        u = unit_arrays(params, x, act1, H.ARMS["LR"], e1_64) | adam_arrays(params, adam)\n')]),
    ("M4d: the per-task medians read the second layer through leaky",
     [(R_UNITS, '        u = unit_arrays(params, x, act1, H.ARMS["LR"], e1_64)\n')]),
    ("M4e: memo_acc through leaky", [(R_MEMO, '            memo = float((forward2(params, x, act1, H.ARMS["LR"])[4].argmax(1) == y).float().mean())\n')]),
    ("M4f: the row-norm cap lands on W3", [(R_W2CALL, "                        nr, rem = W2C.cap_row_norm_(params[4], r2[:10])\n")]),
    ("M4g: only b2 is held", [(R_BRESTORE, "                        pass\n")]),
    ("M4h: the training derivative read from the analytic exp",
     [(R_DTRAIN, "            gt = act.dphi(zz.detach())\n")]),
]


# --------------------------------------------------------------------------
# S5, S7
# --------------------------------------------------------------------------

R_CAPON = "        cap_on = t >= 2 and (do_par or do_perp)\n"
R_CAPSET = "            q_cap = MU.parallel_cap(params[CAP_LAYER], e1)\n"
R_R2SET = "            r2 = W2C.row_norms(params[2])\n"
R_BSET = "            b_star = (params[1].detach().clone(), params[3].detach().clone())\n"
R_ORDER = "            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)\n"


def s5(M) -> dict:
    """The five arms' task 1 is the same run (init, subset, labels, batch orders, task-1-end state
    sha256); every arm differs from every other by the end of task 3; the first-layer radii are the
    task-1-end rows (0916's check); the second-layer radii r2 are the norms of the task-1-end W2 rows;
    the held biases are the task-1-end biases and cap12_bfix's biases at the end of task 3 are those,
    bit for bit.  Bit comparisons only."""
    failed = []
    end1, states, extra = {}, {}, {}
    for arm in ARM_MAP_REG:
        d = {"capture": ((3, M.STEPS_PER_EPOCH * EP - 1),)}
        rows, arrays, _, info = _run(M, arm, tasks=3, debug=d)
        end1[arm] = tuple(info[k] for k in ("init_sha256", "subset_idx_sha256", "labels_sha256",
                                            "batch_sha256", "task1_end_state_sha256"))
        states[arm] = info["final_state_sha256"]
        P1 = d["task1_end_params"]
        st = d["steps"][(3, M.STEPS_PER_EPOCH * EP - 1)]
        extra[arm] = {
            "radii_l1": bool((MU.parallel_cap(P1[0], d["e1"])[:, 0].numpy() == arrays["q_cap"]).all()
                             and (MU.perp_cap(P1[0], d["e1"])[:, 0].numpy() == arrays["v_cap"]).all()),
            "r2_is_task1_end": bool(torch.equal(st["r2"], torch.linalg.vector_norm(P1[2], dim=1, keepdim=True))),
            "b_star_is_task1_end": bool(torch.equal(st["b_star"][0], P1[1]) and torch.equal(st["b_star"][1], P1[3])),
            "biases_held_at_end": bool(torch.equal(st["after"]["params"][1], P1[1])
                                       and torch.equal(st["after"]["params"][3], P1[3])),
            "biases_moved": bool(not torch.equal(st["after"]["params"][1], P1[1])),
        }
    ok = dict(i_task1_identical=len(set(end1.values())) == 1,
              ii_all_differ_by_task3=len(set(states.values())) == len(ARM_MAP_REG),
              iii_radii_l1=all(e["radii_l1"] for e in extra.values()),
              iv_r2=all(e["r2_is_task1_end"] for e in extra.values()),
              v_b_star=all(e["b_star_is_task1_end"] for e in extra.values()),
              vi_bfix_holds=extra["cap12_bfix"]["biases_held_at_end"],
              nonvacuous=all(extra[a]["biases_moved"] for a in ARM_MAP_REG if a != "cap12_bfix"))
    failed += [k for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": {**ok, "per_arm": extra}}


S5_MUT = [
    ("M5a: the first-layer cap is on from task 1", [(R_CAPON, "        cap_on = t >= 1 and (do_par or do_perp)\n")]),
    ("M5b: the first-layer radii are taken at init",
     [(R_CAPSET, "            q_cap = MU.parallel_cap(H.init_params(seed, device)[CAP_LAYER], e1)\n")]),
    ("M5c: r2 is taken at init", [(R_R2SET, "            r2 = W2C.row_norms(H.init_params(seed, device)[2])\n")]),
    ("M5d: the held biases are the init biases",
     [(R_BSET, "            b_star = tuple(q.detach().clone() for q in H.init_params(seed, device)[1:4:2])\n")]),
    ("M5e: the second-layer cap is on from task 1 with the init radii",
     [(R_L2ON, "        l2_on = t >= 1 and w2cap\n"),
      ("    r2 = b_star = None\n", "    r2 = W2C.row_norms(params[2]); b_star = None\n")]),
]


def s7(M) -> dict:
    """The same call twice gives the same bits (arrays, hashes, rows), for ref and cap12_bfix."""
    failed, per = [], {}
    for arm in ("ref", "cap12_bfix"):
        r1, a1, _, i1 = _run(M, arm, tasks=2)
        r2, a2, _, i2 = _run(M, arm, tasks=2)
        ok = dict(i_arrays=set(a1) == set(a2) and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1),
                  ii_hashes=all(i1[k] == i2[k] for k in ("init_sha256", "labels_sha256", "batch_sha256",
                                                         "task1_end_state_sha256", "final_state_sha256")),
                  iii_rows=r1 == r2, nonvacuous=len(a1) > 10)
        per[arm] = ok
        failed += [f"{arm}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S7_MUT = [("M7a: the batch order does not come from the run's own generator",
           [(R_ORDER, "            order = torch.randperm(N_IMAGES).to(device)\n")])]


# --------------------------------------------------------------------------
# S9: the CLI
# --------------------------------------------------------------------------

E_MAP = '           "cap12": ("cap_both", True, False), "cap12_bfix": ("cap_both", True, True)}\n'
E_RELABEL = '            r["arm"] = args.arm\n'
E_TASKS = "TASKS = 100\n"
E_FLUSH = "    torch.set_flush_denormal(True)\n"
E_ACT2 = 'ACT2_NAME = "ELU1"\n'


def s9(M) -> dict:
    """main() on a 2-task, 2-epoch cap12_bfix shard writes (i) per_task rows labelled cap12_bfix and
    otherwise equal to the in-process run of the registered map (cap_both, w2cap, bias_fix), and (ii) the
    same units.npz arrays, bit for bit; (iii) the CLI's ARM_MAP is the registered one; (iv) provenance
    names this experiment, act2 ELU1, flush on, the arm map, and hashes both runners and the cap module;
    (v) the defaults are 100 tasks x 80 epochs, ledger off; (vi) main() leaves flush on."""
    failed, per = [], {}
    out = SCR / "s9_cli"
    torch.set_flush_denormal(False)
    try:
        M.main(["--arm", "cap12_bfix", "--seeds", "0", "--tasks", "2", "--epochs", "2", "--out", str(out)])
        flushed = _flush_is_on()
    finally:
        torch.set_flush_denormal(False)
    EL = load(EL_RUNNER)
    rows, arrays, _, _ = _run(EL, "cap12_bfix", tasks=2)
    got = list(csv.DictReader((out / "per_task.csv").open()))
    per["i_rows"] = len(got) == 2 and all(g["arm"] == "cap12_bfix" for g in got) and all(
        str(r[k]) == v or float(r[k]) == float(v) for r, g in zip(rows, got) for k, v in g.items() if k != "arm")
    with np.load(out / "units.npz") as z:
        per["ii_units"] = set(z.files) == {f"s0_{k}" for k in arrays} and all(
            np.array_equal(z[f"s0_{k}"], v, equal_nan=True) for k, v in arrays.items())
    per["iii_arm_map"] = M.ARM_MAP == ARM_MAP_REG
    pv = json.loads((out / "provenance.json").read_text())
    per["iv_provenance"] = (pv["experiment"] == "l2cap_ee_0917" and pv["spec"] == "specs/spec_l2cap_ee_0917.md"
                            and pv["act2"] == "ELU1" and pv.get("flush_denormal") is True
                            and pv["arm_map"] == {"layer1_arm": "cap_both", "w2cap": True, "bias_fix": True}
                            and {"src/l2cap_ee_run_0917.py", "src/mucap_el_run_0916.py", "src/l2cap_ee_0917.py"}
                            <= set(pv["code_sha256"]))
    per["v_defaults"] = M.TASKS == TASKS_RUN and M.EPOCHS == 80 and pv["ledger"] is False
    per["vi_flush_left_on"] = bool(flushed)
    keys = ("i_rows", "ii_units", "iii_arm_map", "iv_provenance", "v_defaults", "vi_flush_left_on")
    failed += [k for k in keys if not per[k]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S9_MUT = [
    ("M9a: cap12_bfix does not hold the biases",
     [(E_MAP, '           "cap12": ("cap_both", True, False), "cap12_bfix": ("cap_both", True, False)}\n')]),
    ("M9b: the rows keep the shared runner's arm name", [(E_RELABEL, "            pass\n")]),
    ("M9c: the registered horizon is 150 tasks", [(E_TASKS, "TASKS = 150\n")]),
    ("M9d: denormals not flushed", [(E_FLUSH, "    torch.set_flush_denormal(False)\n")]),
    ("M9e: the CLI trains ELU->leaky", [(E_ACT2, "ACT2_NAME = None\n"), ('"act2": ACT2_NAME,', '"act2": "ELU1",')]),
]


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

def _growth() -> float:
    """mucap_ee_0917's flush pilot (the archived log, sha256-checked): mean per-task time over tasks 1-100
    / mean over tasks 2-10, tasks 41-100 extrapolated from the least-squares line through tasks 14-40."""
    import re
    man = json.loads(EE_ARCHIVE_MANIFEST.read_text())
    f = next(x for x in man["files"] if x["source_rel"] == "results/_pilot_mucap_ee_0917/flush_ref_s0.log")
    p = Path(f["backup"])
    assert _sha(p) == f["sha256"]
    ts = []
    for line in p.read_text().splitlines():
        m = re.search(r"\[\s*([\d.]+)s\] ref seed=\d task (\d+)/", line)
        if m:
            ts.append((int(m.group(2)), float(m.group(1))))
    dt_ = {t: b - a for (_, a), (t, b) in zip(ts, ts[1:])}
    base = float(np.mean([dt_[t] for t in range(2, 11)]))
    xs = np.array([t for t in dt_ if t >= 14], float)
    slope, icpt = np.polyfit(xs, np.array([dt_[int(t)] for t in xs]), 1)
    full = [dt_.get(t, base) if t <= 40 else icpt + slope * t for t in range(1, TASKS_RUN + 1)]
    return float(np.mean(full) / base)


def s_cost() -> None:
    """PROBE cap12_bfix processes of the CLI at once, 2 tasks x 80 epochs, one thread each.  Gate: every
    process exits 0 and the memory budget leaves >= PROBE slots.  Projection: measured per-update cost x
    _growth().  mucap_ee_0917's real runs (1 run 7-15 min at 6) are the check on that projection."""
    t0 = time.time()
    avail0 = _mem_available_gib()
    SCR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(PROBE):
        d = SCR / f"cost_{i}"
        procs.append((d, subprocess.Popen(
            [PY, str(L2_RUNNER), "--arm", "cap12_bfix", "--seeds", str(i), "--tasks", "2", "--out", str(d)],
            env=THREAD_ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)))
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
    growth = _growth()
    slots = int(avail0 * 0.8 / max(peak)) if peak else 0
    one = TASKS_RUN * STEPS_RUN * per_update * growth
    ok = all(r == 0 for r in rc) and slots >= PROBE
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": f"all {PROBE} processes exit 0 and slots >= {PROBE}",
                         "returncodes": rc, "mem_available_gib_before": avail0,
                         "peak_rss_gib_max": max(peak) if peak else None, "slots": slots,
                         "seconds_per_update_under_load": per_update, "growth_factor": growth,
                         "one_trajectory_min": one / 60, "grid_50_wall_hours_at_probe": 50 * one / 3600 / PROBE,
                         "measured_at": dt.datetime.now().astimezone().isoformat(),
                         "seconds": round(time.time() - t0, 1)}
    dump()
    c = RESULTS["S-cost"]
    print(f"S-cost: pass={ok}  {per_update * 1e3:.2f} ms/update under {PROBE} -> {c['one_trajectory_min']:.1f} min "
          f"per trajectory, {c['grid_50_wall_hours_at_probe']:.1f} h for 50 at {PROBE} (peak RSS "
          f"{c['peak_rss_gib_max']:.2f} GiB, slots {slots})", flush=True)


# --------------------------------------------------------------------------
# S8: the verdict on synthetic shards
# --------------------------------------------------------------------------

REG_T, REG_SPT, REG_WIN, REG_CAPWIN = 100, 6000, (51, 100), (2, 10)   # the REGISTERED constants
TOL_EST = 1e-9
T_TABLE = {(0.975, 9): 2.262157, (0.9875, 9): 2.685011, (0.95, 9): 1.833113}
TOL_T = 1e-6
A1 = 0.82


def _mf(t: int) -> float:
    return 0.11 + 0.004 * math.sin(t)


R_FLOOR = float(np.mean([_mf(t) - 0.01 for t in range(REG_WIN[0], REG_WIN[1] + 1)]))   # ref's floored level
F_WIN = float(np.mean([_mf(t) for t in range(REG_WIN[0], REG_WIN[1] + 1)]))
THR_WIN = F_WIN + 3.0 * float(np.std([_mf(t) for t in range(REG_WIN[0], REG_WIN[1] + 1)], ddof=1)) / math.sqrt(50)


def _centered(rng, n, sd):
    x = rng.normal(0.0, 1.0, n)
    x = x - x.mean()
    return x / x.std(ddof=1) * sd if sd > 0 else np.zeros(n)


DEFAULT_ARMS = {
    "ref": {"collapse": 5},
    "cap1": {"collapse": 14, "eff2": 0.0, "noise2": 0.02},
    "cap2": {"collapse": 3, "eff2": 0.0, "noise2": 0.02},
    "cap12": {"collapse": None, "L": 0.60, "eff2": 0.2, "noise2": 0.02, "tail": 5.0},
    "cap12_bfix": {"collapse": None, "L": 0.48, "eff2": 0.0, "noise2": 0.3},
}


def synth_shards(V, scen: dict) -> dict:
    """5 arms x 10 seeds x 100 tasks.  Online accuracy: A1 at task 1; an arm that collapses at T keeps A1
    before T and sits at its floor level after (major_frac(t) - 0.01 unless given); an alive arm sits at
    L (+ centred per-seed noise in the main window).  major_frac(t) is the same for every run.  Second
    layer mean phi' at task ends: 0.6 at task 1, 0.6 + r2 + m2[s] in the formation windows, plus eff2 +
    n2[a][s] (+ tail/100 on the arm with a unit-0 tail) in the main window; m2 is shared by the arms of a
    seed, so only a paired comparison sees eff2 when seeds differ; it is written to the training
    derivative's key, and the analytic exp gates of both layers are constants (a wrong E2 key reads one).
    Step-0 rows are garbage.  Cap rows stop 10 tasks after a collapse."""
    rng = np.random.default_rng(scen.get("rng", 917))
    T, NU, SEEDS = REG_T, 100, tuple(range(10))
    arms = {a: {**DEFAULT_ARMS[a], **scen.get("arms", {}).get(a, {})} for a in DEFAULT_ARMS}
    r2 = scen.get("r2", -0.55)
    m2 = _centered(rng, 10, scen.get("seed_sd", 0.02))
    shards = {}
    for a, spec in arms.items():
        n2 = np.zeros(10) if a == "ref" else _centered(rng, 10, spec.get("noise2", 0.02))
        n1 = _centered(rng, 10, 0.0 if a == "ref" else spec.get("noise1", 0.01))
        for si, s in enumerate(SEEDS):
            col = spec["collapse"]
            if isinstance(col, dict):
                col = col.get(s, col.get("default"))
            online, rows_ = [], []
            for t in range(1, T + 1):
                if t == 1:
                    online.append(A1)
                elif col is not None and t >= col:
                    online.append(spec["floor_level"] if "floor_level" in spec else _mf(t) - 0.01)
                elif col is not None:
                    online.append(A1)
                else:
                    online.append(spec["L"] + (n1[si] if REG_WIN[0] <= t <= REG_WIN[1] else 0.0))
                capped = V.CAPPED.get(a, ()) if hasattr(V, "CAPPED") else ()
                frozen = col is not None and t >= col + 10
                rows_.append({c: (5 if (a != "ref" and c in DEFAULT_CAPS[a] and t >= 2 and not frozen) else 0)
                              for c in ("rows_par", "rows_perp", "rows_w2", "rows_bfix")})
            pt = pd.DataFrame({"task": np.arange(1, T + 1), "online_acc": online,
                               "memo_acc": online, "major_frac": [_mf(t) for t in range(1, T + 1)],
                               **{c: [r[c] for r in rows_] for c in ("rows_par", "rows_perp", "rows_w2", "rows_bfix")}})
            for (ia, iseed, itask, icol) in scen.get("idle", ()):
                if ia == a and iseed == s:
                    pt.loc[pt.task == itask, icol] = 0
            task_, step_, G2 = [], [], []
            for t in range(1, T + 1):
                for step in (0, REG_SPT):
                    task_.append(t)
                    step_.append(step)
                    if step == 0:
                        G2.append(np.full(NU, 9.0))
                        continue
                    if t == 1:
                        lv = 0.6
                    elif REG_WIN[0] <= t <= REG_WIN[1]:
                        lv = 0.6 + r2 + m2[si] + (0.0 if a == "ref" else spec.get("eff2", 0.0) + n2[si])
                    else:
                        lv = 0.6 + r2 + m2[si]
                    g = lv + (np.arange(NU) - 49.5) * 1e-4
                    if spec.get("tail") and REG_WIN[0] <= t <= REG_WIN[1]:
                        g = g.copy()
                        g[0] += spec["tail"]
                    G2.append(g)
            k = len(task_)
            units = {"task": np.array(task_, float), "step": np.array(step_, float),
                     "dtrain_mean_l2": np.stack(G2), "gate_mean_l2": np.full((k, NU), 0.3),
                     "gate_mean_l1": np.full((k, NU), 0.3), "zbar_l2": np.zeros((k, NU)), "b2": np.zeros((k, NU)),
                     "U_l1": np.ones((k, NU)), "U_l2": np.ones((k, NU)),
                     "mu2_norm": np.full((k, 1), 5.0), "mu2_proj_sd": np.full((k, 1), 1.0),
                     "s2_mean": np.zeros((k, 1)), "s2_sd": np.ones((k, 1))}
            hs = {kk: f"h{s}" for kk in V.STREAM_KEYS}
            if s in scen.get("break", ()) and a == "cap12":
                hs["batch_sha256"] = "broken"
            prov = {"git_hash": "synthetic", "per_seed": {str(s): {**hs, "tasks_completed": T,
                                                                   "divergence": {"diverged": False}}}}
            # a ledger-off shard writes ledger.npz with its task column only (the real grid does)
            shards[(a, s)] = {"units": units, "per_task": pt, "prov": prov,
                              "ledger": {"task": np.arange(1, T + 1)}}
    return shards


DEFAULT_CAPS = {"cap1": ("rows_par", "rows_perp"), "cap2": ("rows_w2",), "cap12": ("rows_par", "rows_perp", "rows_w2"),
                "cap12_bfix": ("rows_par", "rows_perp", "rows_w2", "rows_bfix")}
_BORDER = 2.45 * 0.05 / math.sqrt(10)      # t = 2.45: inside (t_.975,9 = 2.262, t_.9875,9 = 2.685)
_split = {"default": 3}
S8_SCENARIOS = [
    ("designed", {},
     {"main": "RESCUED", "cap12_95": "RESCUED", "cap1": "COLLAPSED", "cap2": "COLLAPSED",
      "cap12_bfix": "RESCUED_FUNCTION_ONLY", "timing_cap12": "LATER", "timing_cap1": "LATER",
      "timing_cap2": "EARLIER", "timing_cap12_bfix": "LATER"},
     {"main_E1": 0.60 - R_FLOOR, "main_E2": 0.2 + 5.0 / 100, "n_valid": 10,
      "t_half|ref|0": 5, "t_half|cap2|0": 3, "t_half|cap1|0": 14, "t_half|cap12_bfix|0": None,
      "t_half|cap12|3": None, "state|ref": "FLOORED", "state|cap2": "FLOORED", "state|cap12": "ALIVE",
      "impaired|cap2": True, "impaired|cap1": False, "impaired|cap12": False, "impaired|cap12_bfix": False}),
    ("borderline", {"arms": {"cap12": {"eff2": _BORDER, "noise2": 0.05, "tail": 0.0}}},
     {"main": "RESCUED_FUNCTION_ONLY", "cap12_95": "RESCUED"}, {"main_E2": _BORDER}),
    ("not_reproduced", {"arms": {"ref": {"collapse": None, "L": 0.5}}},
     {"main": "NOT_REPRODUCED", "cap2": "NOT_REPRODUCED", "cap12_bfix": "NOT_REPRODUCED"}, {}),
    ("ref_8_of_10", {"arms": {"ref": {"collapse": {8: None, 9: None, "default": 5}, "L": 0.5}}},
     {"main": "RESCUED"}, {}),
    ("ref_7_of_10", {"arms": {"ref": {"collapse": {7: None, 8: None, 9: None, "default": 5}, "L": 0.5}}},
     {"main": "NOT_REPRODUCED"}, {}),
    ("split", {"arms": {"cap12": {"collapse": {7: 20, 8: 20, 9: 20, "default": None}}}},
     {"main": "SPLIT"}, {}),
    ("floor_margin", {"arms": {"cap12_bfix": {"collapse": 20, "floor_level": F_WIN + 0.0005},
                               "cap2": {"collapse": 20, "floor_level": THR_WIN + 0.002, "eff2": 0.1}}},
     {"cap12_bfix": "COLLAPSED", "cap2": "RESCUED"}, {}),
    ("cap_idle", {"idle": (("cap12_bfix", 3, 5, "rows_bfix"), ("cap12", 4, 60, "rows_w2"))},
     {"cap12_bfix": "INAPPLICABLE", "main": "RESCUED"}, {}),
    ("broken_streams", {"break": (7, 8, 9)}, {"main": "INAPPLICABLE"}, {"n_valid": 7}),
    ("timing_counts", {"arms": {"cap2": {"collapse": {8: 7, 9: 7, "default": 3}},
                                "cap12_bfix": {"collapse": {6: 5, 7: 5, 8: 5, 9: 5, "default": 3}},
                                "cap12": {"collapse": {5: 5, 6: 5, 7: 5, 8: 5, 9: 5, "default": 3}}}},
     {"timing_cap2": "NO_TIMING_DIFF", "timing_cap12_bfix": "EARLIER", "timing_cap12": "NO_TIMING_DIFF"}, {}),
    ("paired_matters", {"seed_sd": 1.0, "r2": -3.0,
                        "arms": {"cap12": {"eff2": 0.25, "noise2": 0.02, "tail": 0.0}}},
     {"main": "RESCUED"}, {"main_E2": 0.25}),
]


def _pick(res: dict, key: str):
    if key == "n_valid":
        return len(res["valid_seeds"])
    if key.startswith("main_"):
        ep = key.split("_")[1]
        return next(r["mean"] for r in res["rows"] if r["role"] == "main" and r["endpoint"] == ep)
    if key.startswith("t_half|"):
        _, arm, s = key.split("|")
        return res["t_half"][f"{arm}|{s}"]
    if key.startswith("state|"):
        return res["floor_state"][key.split("|")[1]]
    if key.startswith("impaired|"):
        return res["impaired"][key.split("|")[1]]["flag"]
    raise KeyError(key)


def s8(V) -> dict:
    """verdict.analyze on the S8 scenarios: every designed label and estimate comes out as written
    (estimates within TOL_EST, flags, states and censored times exactly), and the imported t quantile
    reproduces the published table within TOL_T.  Tolerances: TOL_EST, TOL_T."""
    failed, per = [], {}
    bad_t = [f"t({p},{df})" for (p, df), w in T_TABLE.items() if abs(V.t_quantile(p, df) - w) > TOL_T]
    failed += bad_t
    for name, scen, want_labels, want_vals in S8_SCENARIOS:
        res = V.analyze(synth_shards(V, scen))
        bad = [f"{k}: got {res['labels'].get(k)} want {w}" for k, w in want_labels.items()
               if res["labels"].get(k) != w]
        for k, w in want_vals.items():
            try:
                g = _pick(res, k)
            except (KeyError, StopIteration) as e:
                bad.append(f"{k}: missing ({e!r})")
                continue
            if w is None or isinstance(w, (bool, str, int)) and not isinstance(w, float):
                ok = g == w and (w is not None or g is None)
            else:
                ok = g is not None and np.isfinite(g) and abs(float(g) - w) <= TOL_EST
            if not ok:
                bad.append(f"{k}: got {g} want {w}")
        per[name] = {"labels": res["labels"], "bad": bad}
        failed += [f"{name}|{b}" for b in bad]
    return {"pass": not failed, "failed_items": failed, "detail": per}


V_MAIN = 'MAIN_ARM = "cap12"                          # spec 5.3: the primary comparison is cap12 - ref\n'
V_E2 = 'E2_KEY = "dtrain_mean_l2"\n'
V_LEVEL = "MAIN_LEVEL = 0.975                          # two-sided, two main endpoints (Bonferroni)\n"
V_B = "    B_ok = A_ok and ref_at >= math.ceil(REF_FLOOR_FRAC * len(valid)) and B_E2[\"hi\"] < 0\n"
V_FRAC = "REF_FLOOR_FRAC = 0.8 "
V_K = "FLOOR_K = 3.0 "
V_F = '    F, s = float(f.mean()), float(f.std(ddof=1))\n'
V_PAIR = '        return deltas(arm, ep, win) - deltas("ref", ep, win)\n'
V_UMEAN = "        m, k = unit_mean(u[key][ends[t]])\n"
V_CENS = '    if ta is None:\n        return "later"\n'
V_ALPHA = "SIGN_ALPHA = 0.05 "
V_SIGN_N = "        p = sign_test(max(e, l_), e + l_)\n"
V_HALF = "        if f.loc[t] < HALF:\n"
V_CAPWIN = "CAP_WIN = (2, 10) "
V_CAPPED = '          "cap12_bfix": ("rows_par", "rows_perp", "rows_w2", "rows_bfix")}\n'
V_STREAMS = "    if len(set(infos.values())) != 1:\n"
V_IMP = "            hits += f_arm < f_ref - (1.0 - f_ref)\n"
V_ENDS_USE = "    ends = task_ends(u)\n    base, nans = unit_mean(u[key][ends[1]])\n"
S8_MUT = [
    ("M8a: the primary arm is cap12_bfix", [(V_MAIN, 'MAIN_ARM = "cap12_bfix"\n')]),
    ("M8b: E2 reads the analytic exp gate instead of the training derivative", [(V_E2, 'E2_KEY = "gate_mean_l2"\n')]),
    ("M8c: main label at 95%", [(V_LEVEL, "MAIN_LEVEL = 0.95\n")]),
    ("M8d: ref's collapse never required", [(V_B, "    B_ok = A_ok and B_E2[\"hi\"] < 0\n")]),
    ("M8e: ref must be floored in every seed", [(V_FRAC, "REF_FLOOR_FRAC = 1.0 ")]),
    ("M8f: ref floored in half the seeds is enough", [(V_FRAC, "REF_FLOOR_FRAC = 0.5 ")]),
    ("M8g: the floor margin dropped", [(V_K, "FLOOR_K = 0.0 ")]),
    ("M8h: the floor at chance instead of the label draw", [(V_F, "    F, s = CHANCE, float(f.std(ddof=1))\n")]),
    ("M8i: the pairing broken", [(V_PAIR, '        return deltas(arm, ep, win) - deltas("ref", ep, win)[::-1]\n')]),
    ("M8j: unit median instead of the mean",
     [(V_UMEAN, "        m, k = float(np.median(u[key][ends[t]])), 0\n")]),
    ("M8k: a censored arm counted as a tie", [(V_CENS, '    if ta is None:\n        return "tie"\n')]),
    ("M8l: sign test at 0.2", [(V_ALPHA, "SIGN_ALPHA = 0.2 ")]),
    ("M8m: ties counted in the sign test's n", [(V_SIGN_N, "        p = sign_test(max(e, l_), len(valid))\n")]),
    ("M8n: T_half on raw accuracy instead of the fit fraction",
     [(V_HALF, "        if online(pt).loc[t] < HALF:\n")]),
    ("M8o: (C) read in the main window", [(V_CAPWIN, "CAP_WIN = (51, 100) ")]),
    ("M8p: cap12_bfix's (C) does not require the bias restore",
     [(V_CAPPED, '          "cap12_bfix": ("rows_par", "rows_perp", "rows_w2")}\n')]),
    ("M8q: the cross-arm stream check off", [(V_STREAMS, "    if False:\n")]),
    ("M8r: IMPAIRED against ref's early fit itself", [(V_IMP, "            hits += f_arm < f_ref\n")]),
    ("M8s: a task's first row taken as its end point (E2)",
     [(V_ENDS_USE, "    ends = {int(t_): i for i, t_ in reversed(list(enumerate(u['task'])))}\n"
                   "    base, nans = unit_mean(u[key][ends[1]])\n")]),
]

# --------------------------------------------------------------------------

CHECKS = {
    "S0_S-el-unchanged": ("S0 the shared runner's EL path reproduces the 0916 shards", s0, S0_MUT, s0.__doc__, EL_RUNNER),
    "S0b_S-ee-unchanged": ("S0b its EE path with the new switches off reproduces mucap_ee_0917", s0b, S0B_MUT,
                           s0b.__doc__, EL_RUNNER),
    "S2c_S-row-norm-cap": ("S2c the row-norm cap writes only rows above their radius, onto it", s2c, S2C_MUT,
                           s2c.__doc__, W2CAP),
    "S4_S-wiring": ("S4 cap12_bfix updates are the host ELU->ELU update + caps + bias restore; dtrain is autograd's",
                    s4, S4_MUT, s4.__doc__, EL_RUNNER),
    "S5_S-identical": ("S5 five arms share task 1; radii and held biases are task-1-end values", s5, S5_MUT,
                       s5.__doc__, EL_RUNNER),
    "S7_S-repro": ("S7 the same call twice gives the same bits", s7, S7_MUT, s7.__doc__, EL_RUNNER),
    "S8_S-verdict": ("S8 the verdict returns the designed labels and estimates on synthetic shards", s8, S8_MUT,
                     s8.__doc__, VERDICT),
    "S9_S-cli": ("S9 the CLI maps, relabels and records this experiment", s9, S9_MUT, s9.__doc__, L2_RUNNER),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated check keys or prefixes; 'cost' for S-cost")
    args = ap.parse_args()
    global DUMP_PATH
    keys = [k for k in CHECKS if not args.only or any(k.startswith(p) for p in args.only.split(","))]
    if args.only:
        SCR.mkdir(parents=True, exist_ok=True)
        DUMP_PATH = SCR / "checks_partial.json"
    RESULTS.update({"run_id": "l2cap_ee_0917", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "spec": "specs/spec_l2cap_ee_0917.md", "prereg_commit": load(L2_RUNNER).PREREG_COMMIT,
                    "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (EL_RUNNER, L2_RUNNER, W2CAP, VERDICT, Path(__file__),
                                              REPO / "src" / "mucap_el_0916.py")},
                    "torch": torch.__version__, "threads": torch.get_num_threads()})
    if not args.only:
        s1_3_record()
    for k in keys:
        title, fn, mut, deriv, target = CHECKS[k]
        run_check(k, title, fn, mut, deriv, target)
    if not args.only or "cost" in args.only:
        s_cost()
    RESULTS["finished_at"] = dt.datetime.now().astimezone().isoformat()
    dump()
    print(f"all_pass = {RESULTS['all_pass']}  -> {DUMP_PATH}")


if __name__ == "__main__":
    main()

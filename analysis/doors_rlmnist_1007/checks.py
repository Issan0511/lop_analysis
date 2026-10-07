"""Checks for doors_rlmnist_1007 (specs/spec_doors_rlmnist_1007.md section 6).

    cd ~/Projects/claude/wt/doors_rlmnist_1007
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python \
        -m analysis.doors_rlmnist_1007.checks            # or: ... analysis/doors_rlmnist_1007/checks.py
    ... --only S-H            # development: writes results/_checks_doors_rlmnist_1007/checks_partial.json

Each check runs on the real code and then on every mutation listed with it; a mutation is an exact-once
string substitution in the check's target file and it MUST make the same check fail.  Tolerances come from
the derivations in the docstrings (bit comparisons have none).  Writes results/doors_rlmnist_1007/checks.json
after every step.

S0      the doors-off path (ref) reproduces the recorded l2cap_ee_0917 ref shards (seeds 0, 5; tasks 1-2)
S-C     door C: the input is x - x.mean(0), zero mean within the derived rounding bound, not divided, and
        the same input (the same mean vector for test) is used by every evaluation
S-H     door H: m is the beta = 0.01 EMA of the uncentred phi batch mean, a constant for the gradient,
        carried across task boundaries, and the value in force is used by every evaluation
S-gate  the gate under the doors is the host ELU's autograd derivative (expm1 + 1, 0 below ln 2^-24);
        captured H / CH updates equal an independent recomputation
S5      the four arms share their streams; H's first update is ref's, CH's is C's; all differ by task 3
S7      the same call twice gives the same bits
S8      the verdict on synthetic shards
S9      the CLI maps each arm, relabels the rows and records this experiment
S-cost  PROBE CLI processes at once
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

from src import pmnist_0905 as H                 # noqa: E402
from src import pmnist_rlmnist_0906 as RL        # noqa: E402
from src import elu_growth_0909 as EG            # noqa: E402
from src import mucap_el_0916 as MU              # noqa: E402

torch.set_num_threads(1)                         # after the imports (elu_growth's chain sets it too)

RUNNER = REPO / "src" / "doors_rlmnist_1007.py"
CLI = REPO / "src" / "doors_rlmnist_run_1007.py"
VERDICT = REPO / "analysis" / "doors_rlmnist_1007" / "verdict.py"
OUT = REPO / "results" / "doors_rlmnist_1007"
SCR = REPO / "results" / "_checks_doors_rlmnist_1007"
L2_RECORD = REPO / "results" / "l2cap_ee_0917"
L2_MANIFEST = L2_RECORD / "backup_manifest.json"
EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)
U32, U64 = 2.0 ** -24, 2.0 ** -53
DEV = H.setup("cpu")
MNIST = H.Mnist(DEV)
PROBE = 6
PY = sys.executable
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
TASKS_RUN, EPOCHS_RUN, STEPS_RUN = 100, 80, 6000
RESULTS: dict = {}
DUMP_PATH = OUT / "checks.json"
ARM_MAP_REG = {"ref": (False, False), "C": (True, False), "H": (False, True), "CH": (True, True)}
BETA_REG = 0.01
ELU = EG.ELU(1.0)
LEAKY = H.ARMS["LR"]
EP = 2                    # 150 updates per task in the short runs; the properties checked are per update
SPT_EP = 75 * EP
ZERO_Z32 = math.log(2.0 ** -24)

# --------------------------------------------------------------------------
# infrastructure (analysis/l2cap_ee_0917/checks.py's, unchanged in behaviour)
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
    entry["mutations_raised"] = sum("raised" in m for m in entry["mutations"])
    entry["seconds"] = round(time.time() - t0, 1)
    RESULTS[key] = entry
    dump()
    print(f"{key}: pass={entry['pass']}  mutations detected "
          f"{sum(m['detected'] for m in entry['mutations'])}/{len(entry['mutations'])}"
          f" (raised {entry['mutations_raised']})  ({entry['seconds']}s)"
          f"{'  FAILED ' + str(entry.get('failed_items', [])[:3]) if not entry['pass'] else ''}", flush=True)


class flush_denormal:
    """The CLI sets torch.set_flush_denormal(True) (spec 2.1); in-process runs here do the same."""
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


def _run(M, door_c, door_h, seed=0, tasks=2, epochs=EP, debug=None):
    with flush_denormal():
        return M.run_doors(seed, 1e-3, tasks, MNIST, DEV, door_c=door_c, door_h=door_h, epochs=epochs,
                           debug=debug)


def _equal_list(a, b) -> bool:
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _gamma(k: int, u: float) -> float:
    return k * u / (1.0 - k * u)


def _ends(arrays: dict, spt: int) -> dict:
    t, s = arrays["task"], arrays["step"]
    return {int(t[i]): int(i) for i in np.where(s == spt)[0]}


# The independent reference: the door forward, the host's Adam and the EMA, written here, not imported.
def _fwd(p, xb, m, act2=ELU, inside=False):
    z1 = xb @ p[0].T + p[1]
    a1 = ELU.phi(z1)
    if m is not None:
        a1 = ELU.phi(z1 - m[0]) if inside else a1 - m[0]
    z2 = a1 @ p[2].T + p[3]
    a2 = act2.phi(z2)
    if m is not None:
        a2 = act2.phi(z2 - m[1]) if inside else a2 - m[1]
    return z1, a1, z2, a2, a2 @ p[4].T + p[5]


def _ref_update(st: dict, use_m=True, act2=ELU) -> dict:
    """One update recomputed from the captured before-state: forward with m a plain constant, CE,
    autograd, the host's Adam arithmetic, then the EMA on the uncentred phi of that forward."""
    with flush_denormal():
        b = st["before"]
        m = b["door_m"] if use_m else None
        p = [q.clone().requires_grad_(True) for q in b["params"]]
        mm = [q.clone() for q in b["m"]]
        vv = [q.clone() for q in b["v"]]
        out = _fwd(p, st["xb"], m, act2)
        loss = torch.nn.functional.cross_entropy(out[4], st["yb"])
        grads = torch.autograd.grad(loss, p)
        tc = b["tc"] + 1
        b1, b2, eps, lr = 0.9, 0.999, 1e-8, 1e-3
        c1, c2 = 1 - b1 ** tc, 1 - b2 ** tc
        with torch.no_grad():
            p = [q.detach().clone() for q in p]
            for q, g, mi, vi in zip(p, grads, mm, vv):
                mi.mul_(b1).add_(g, alpha=1 - b1)
                vi.mul_(b2).addcmul_(g, g, value=1 - b2)
                q -= lr * (mi / c1) / ((vi / c2).sqrt() + eps)
            dm = None
            if b["door_m"] is not None:
                dm = [q.clone() for q in b["door_m"]]
                for mi, z, act in zip(dm, (out[0].detach(), out[2].detach()), (ELU, act2)):
                    mi.mul_(1.0 - BETA_REG).add_(act.phi(z).mean(0), alpha=BETA_REG)
    return {"params": p, "m": mm, "v": vv, "door_m": dm, "grads": [g.detach() for g in grads], "out": out}


def _after_equal(st: dict, r: dict) -> bool:
    a = st["after"]
    ok = _equal_list(a["params"], r["params"]) and _equal_list(a["m"], r["m"]) and _equal_list(a["v"], r["v"])
    if a["door_m"] is not None:
        ok = ok and r["door_m"] is not None and _equal_list(a["door_m"], r["door_m"])
    return ok


# --------------------------------------------------------------------------
# S0: the doors-off path reproduces the recorded l2cap ref shards
# --------------------------------------------------------------------------

S0_SEEDS = (0, 5)
S0_TASKS = 2


def _archived_l2(shard: str, name: str) -> Path | None:
    man = json.loads(L2_MANIFEST.read_text())
    for f in man["files"]:
        if f["source_rel"] == f"results/l2cap_ee_0917/runs/{shard}/{name}":
            p = Path(f["backup"])
            return p if p.exists() and _sha(p) == f["sha256"] else None
    return None


def s0(M) -> dict:
    """run_doors with both doors shut, flush on, 80 epochs, tasks 1-2, reproduces the recorded l2cap ref
    shards of seeds 0 and 5: every per_task.csv column of the record equal as written and no extra column,
    every units.npz array bit-equal on the task 1-2 rows (test arrays: the task-1 row) and no extra array
    (the archived copy only if its sha256 matches the manifest), and the task-1-end state hash.  String
    and bit comparisons only."""
    failed, per = [], {}
    for seed in S0_SEEDS:
        shard = f"ref_s{seed}"
        rows, arrays, info = _run(M, False, False, seed=seed, tasks=S0_TASKS, epochs=80)
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
              "iv_init_subset": all(info[k] == prov["per_seed"][str(seed)][k] for k in ("init_sha256", "subset_idx_sha256"))}
        per[shard] = {**ok, "bad_columns": bad_cols[:10], "n_columns": n_cols, "extra_columns": extra_cols[:10],
                      "bad_arrays": arr_bad[:10], "n_arrays": n_arr, "extra_arrays": extra_arr[:10],
                      "archive": str(u_path)}
        failed += [f"{shard}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S0_MUT = [
    ("M0a: door H runs whatever door_h says (its m and its EMA ignore the flag)",
     [("        if door_h else None\n", "        if True else None\n"),
      ("                    if door_h:\n                        m_old = [q.clone() for q in m]\n",
       "                    if m is not None:\n                        m_old = [q.clone() for q in m]\n")]),
    ("M0b: door C runs whatever door_c says", [("    if door_c:\n        xbar = x_raw.mean(0)",
                                                "    if True:\n        xbar = x_raw.mean(0)")]),
    ("M0c: Adam eps 1e-7", [("b1, b2, eps = 0.9, 0.999, 1e-8", "b1, b2, eps = 0.9, 0.999, 1e-7")]),
    ("M0d: memo through leaky",
     [("            memo = float((forward_doors(params, x, act1, act2, m)[4].argmax(1) == y).float().mean())\n",
       '            memo = float((forward_doors(params, x, act1, H.ARMS["LR"], m)[4].argmax(1) == y).float().mean())\n')]),
    ("M0e: dense diagnostic points in task 1 too",
     [("        dense = set(DENSE) if 2 <= t <= 10 else {0}\n", "        dense = set(DENSE) if 1 <= t <= 10 else {0}\n")]),
]


# --------------------------------------------------------------------------
# S-C: door C
# --------------------------------------------------------------------------

def _c_bound(x_raw: torch.Tensor, xc: torch.Tensor, n_sum: int) -> torch.Tensor:
    """Spec section 3, written here independently of the runner."""
    n = x_raw.shape[0]
    return (_gamma(n + 2, U32) * x_raw.double().abs().mean(0)
            + (U32 / (1 - U32) + _gamma(n_sum + 1, U64)) * xc.double().abs().mean(0)) * (1 + 8 * EPS64)


def s_c(M) -> dict:
    """Door C on seed 0, 2 tasks x 2 epochs.  (i) the input the runner trains and evaluates on is
    x - x.mean(0) bit for bit (recomputed here from the host subset); (ii) its float64 mean is within
    the spec-3 bound B_j (n_sum = 1200) at every pixel, and the raw mean is not (non-vacuity);
    (iii) the per-task record of the FED batches says c_ok = 1 with c_in_ratio <= 1 in both tasks;
    (iv) not divided: per pixel |sd(xc) - sd(x)| <= u/(1-u) max|xc| + 4 gamma64_n max|x| (a constant
    shift leaves the sd; the per-element rounding perturbation has sd at most its max; plus the float64 sd
    arithmetic); (v) the captured batch at (2, 5) is rows of xc; (vi) memo at the end of task 1 is the
    accuracy of the captured task-end params on xc and not on the raw x; (vii) the task-1-end zbar_l1
    of units.npz is mean(xc W1^T + b1) and not the raw one; (viii) the task-1 test arrays use
    xt - (the SAME train mean), not xt - xt.mean(0).  Bit comparisons except (ii)/(iv)."""
    failed, per = [], {}
    d = {"capture": ((2, 5),), "end_capture": (1,)}
    rows, arrays, info = _run(M, True, False, debug=d)
    x_raw = _images(0)
    xbar = x_raw.mean(0)
    xc = x_raw - xbar
    per["i_input_is_x_minus_mean"] = bool(torch.equal(d["x"], xc))
    per["run_completed"] = (not info["divergence"]["diverged"]) and len(rows) == 2 and 1 in d.get("ends", {}) \
        and (2, 5) in d.get("steps", {})
    if not per["run_completed"]:
        return {"pass": False, "failed_items": ["run_completed"], "detail": per}
    B = _c_bound(x_raw, xc, x_raw.shape[0])
    mean64 = d["x"].double().mean(0)
    per["ii_mean_within_bound"] = bool((mean64.abs() <= B).all())
    per["ii_max_abs_mean"] = float(mean64.abs().max())
    per["ii_max_bound"] = float(B.max())
    per["ii_raw_mean_outside"] = bool((x_raw.double().mean(0).abs() > B).any())
    per["iii_fed_record"] = len(rows) == 2 and all(r.get("c_ok") == 1 and r.get("c_in_ratio", 9) <= 1 for r in rows)
    sd_tol = U32 / (1 - U32) * d["x"].abs().amax(0).double() + 4 * _gamma(x_raw.shape[0], U64) * x_raw.abs().amax(0).double()
    dsd = (d["x"].double().std(0, unbiased=False) - x_raw.double().std(0, unbiased=False)).abs()
    per["iv_not_divided"] = bool((dsd <= sd_tol).all())
    per["iv_max_sd_change"] = float(dsd.max())
    xb = d["steps"][(2, 5)]["xb"]
    rows_in = lambda X: all(bool((X == r).all(1).any()) for r in xb)
    per["v_batch_rows_from_xc"] = rows_in(xc) and not rows_in(x_raw)
    end = d["ends"][1]
    p1, y1 = end["params"], end["y"]
    with flush_denormal():
        acc_c = float((_fwd(p1, xc, None)[4].argmax(1) == y1).float().mean())
        acc_r = float((_fwd(p1, x_raw, None)[4].argmax(1) == y1).float().mean())
        zc = (xc @ p1[0].T + p1[1]).double().mean(0).numpy()
        zr = (x_raw @ p1[0].T + p1[1]).double().mean(0).numpy()
        xt = MNIST.test_x
        zt = ((xt - xbar) @ p1[0].T + p1[1]).double().mean(0).numpy()
        zt_own = ((xt - xt.mean(0)) @ p1[0].T + p1[1]).double().mean(0).numpy()
    per["vi_memo_on_xc"] = rows[0]["memo_acc"] == acc_c and acc_c != acc_r
    e1 = _ends(arrays, SPT_EP)
    per["vii_units_on_xc"] = bool(np.array_equal(arrays["zbar_l1"][e1[1]], zc)) and not np.array_equal(zc, zr)
    per["viii_test_same_mean"] = "test_zbar_l1" in arrays and bool(np.array_equal(arrays["test_zbar_l1"][0], zt)) \
        and not np.array_equal(zt, zt_own)
    keys = ("i_input_is_x_minus_mean", "ii_mean_within_bound", "ii_raw_mean_outside", "iii_fed_record",
            "iv_not_divided", "v_batch_rows_from_xc", "vi_memo_on_xc", "vii_units_on_xc", "viii_test_same_mean")
    failed += [k for k in keys if not per[k]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


SC_MUT = [
    ("M-Ca: divided by the mean (x / xbar)", [("        x = x_raw - xbar\n", "        x = x_raw / xbar\n")]),
    ("M-Cb: divided (standardised by the sd)", [("        x = x_raw - xbar\n", "        x = (x_raw - xbar) / x_raw.std()\n")]),
    ("M-Cc: another seed's mean subtracted",
     [("        xbar = x_raw.mean(0)                           # door C",
       "        xbar = mnist.train_x[RL.subset_idx(seed + 1).to(device)].mean(0)  # door C")]),
    ("M-Cd: memo on the raw input",
     [("            memo = float((forward_doors(params, x, act1, act2, m)[4].argmax(1) == y).float().mean())\n",
       "            memo = float((forward_doors(params, x_raw, act1, act2, m)[4].argmax(1) == y).float().mean())\n")]),
    ("M-Ce: the diagnostics on the raw input",
     [("        u = unit_arrays_doors(params, x, act1, act2, e1_64, m) | EL.adam_arrays(params, adam)\n",
       "        u = unit_arrays_doors(params, x_raw, act1, act2, e1_64, m) | EL.adam_arrays(params, adam)\n")]),
    ("M-Cf: the test images centred by their own mean",
     [("        xt = xt_raw - xbar ", "        xt = xt_raw - xt_raw.mean(0) ")]),
]


# --------------------------------------------------------------------------
# S-H: door H
# --------------------------------------------------------------------------

SH_CAPTURE = ((1, 0), (1, 1), (1, SPT_EP - 1), (2, 0), (2, 77))


def s_h(M) -> dict:
    """Door H on seed 0, 2 tasks x 2 epochs, five captured updates.  (i) each update's m is
    m.mul_(1 - 0.01).add_(phi(z_l).mean(0), alpha=0.01) of the UNCENTRED phi, with z from an independent
    forward of the captured before-state (z2 from a1 - m1), bit for bit; (ii) m is nonzero after the update
    and is not what tracking the centred output would give (non-vacuity); (iii) the gradients the runner
    used are the gradients of the independent forward with m a plain constant, bit for bit, and are not
    those of a forward whose m carries the batch mean's gradient (same values, non-vacuity); (iv) the
    runner's m never requires grad; (v) m before the first update of task 2 is m after the last update of
    task 1; (vi) at the end of tasks 1 and 2, memo and the units.npz zbar_l2 / a1mean_l1 are the
    independent forward's with the captured m, and the forward without m gives different values;
    (vii) the per-task record h_upd_l1, h_upd_l2 > 0.  Bit comparisons only."""
    failed, per = [], {}
    d = {"capture": SH_CAPTURE, "end_capture": (1, 2)}
    rows, arrays, info = _run(M, False, True, debug=d)
    upd = {}
    for key in SH_CAPTURE:
        st = d["steps"][key]
        b = st["before"]
        with flush_denormal():
            out = _fwd(b["params"], st["xb"], b["door_m"])
            exp_m = [q.clone() for q in b["door_m"]]
            alt_m = [q.clone() for q in b["door_m"]]
            for li, z in enumerate((out[0], out[2])):
                exp_m[li].mul_(1.0 - BETA_REG).add_(ELU.phi(z).mean(0), alpha=BETA_REG)
                alt_m[li].mul_(1.0 - BETA_REG).add_((ELU.phi(z) - b["door_m"][li]).mean(0), alpha=BETA_REG)
            # gradient with m a constant, and with m carrying the batch mean's gradient (same values)
            p = [q.clone().requires_grad_(True) for q in b["params"]]
            o = _fwd(p, st["xb"], b["door_m"])
            g_const = torch.autograd.grad(torch.nn.functional.cross_entropy(o[4], st["yb"]), p)
            p2 = [q.clone().requires_grad_(True) for q in b["params"]]
            z1 = st["xb"] @ p2[0].T + p2[1]
            a1 = ELU.phi(z1)
            a1 = a1 - (b["door_m"][0] + (a1.mean(0) - a1.mean(0).detach()))
            z2 = a1 @ p2[2].T + p2[3]
            a2 = ELU.phi(z2)
            a2 = a2 - (b["door_m"][1] + (a2.mean(0) - a2.mean(0).detach()))
            g_thru = torch.autograd.grad(torch.nn.functional.cross_entropy(a2 @ p2[4].T + p2[5], st["yb"]), p2)
        after_m = st["after"]["door_m"]
        upd[str(key)] = {
            "i_ema_uncentred": _equal_list(after_m, exp_m),
            "ii_nonzero": all(bool((q != 0).any()) for q in after_m),
            "ii_not_centred_tracking": key == (1, 0) or not _equal_list(after_m, alt_m),
            "iii_grad_is_constant_m": _equal_list(st["grads"], list(g_const)),
            "iii_grad_through_m_differs": not _equal_list(list(g_const), list(g_thru)),
            "iv_no_grad": all(not q.requires_grad for q in after_m + b["door_m"]),
        }
    per["updates"] = upd
    for k in ("i_ema_uncentred", "ii_nonzero", "ii_not_centred_tracking", "iii_grad_is_constant_m",
              "iii_grad_through_m_differs", "iv_no_grad"):
        per[k] = all(u[k] for u in upd.values())
    per["v_carried_across_boundary"] = _equal_list(d["steps"][(2, 0)]["before"]["door_m"],
                                                   d["steps"][(1, SPT_EP - 1)]["after"]["door_m"])
    e = _ends(arrays, SPT_EP)
    x = _images(0)
    ok_eval, nonvac = True, True
    for t in (1, 2):
        end = d["ends"][t]
        with flush_denormal():
            o_m = _fwd(end["params"], x, end["door_m"])
            o_n = _fwd(end["params"], x, None)
        memo_m = float((o_m[4].argmax(1) == end["y"]).float().mean())
        ok_eval &= rows[t - 1]["memo_acc"] == memo_m
        ok_eval &= bool(np.array_equal(arrays["zbar_l2"][e[t]], o_m[2].double().mean(0).numpy()))
        ok_eval &= bool(np.array_equal(arrays["a1mean_l1"][e[t]], o_m[1].double().mean(0).numpy()))
        ok_eval &= bool(np.array_equal(arrays["m_l1"][e[t]], end["door_m"][0].double().numpy()))
        nonvac &= not np.array_equal(o_m[2].double().mean(0).numpy(), o_n[2].double().mean(0).numpy())
    per["vi_eval_uses_m"] = bool(ok_eval)
    per["vi_nonvacuous"] = bool(nonvac)
    per["vii_record"] = all(r.get("h_upd_l1", 0) > 0 and r.get("h_upd_l2", 0) > 0 for r in rows) and len(rows) == 2
    per["h_upd"] = [(r.get("h_upd_l1"), r.get("h_upd_l2")) for r in rows]
    keys = ("i_ema_uncentred", "ii_nonzero", "ii_not_centred_tracking", "iii_grad_is_constant_m",
            "iii_grad_through_m_differs", "iv_no_grad", "v_carried_across_boundary", "vi_eval_uses_m",
            "vi_nonvacuous", "vii_record")
    failed += [k for k in keys if not per[k]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


SH_MUT = [
    ("M-Ha: the EMA tracks the centred output",
     [("        mi.mul_(1.0 - beta).add_(act.phi(z).mean(0), alpha=beta)\n",
       "        c_ = (act.phi(z) - mi).mean(0)\n        mi.mul_(1.0 - beta).add_(c_, alpha=beta)\n")]),
    ("M-Hb: the gradient passes through m (same values, the batch mean's gradient)",
     [("        a1 = a1 - m[0]\n", "        a1 = a1 - (m[0] + (a1.mean(0) - a1.mean(0).detach()))\n")]),
    ("M-Hc: memo without m",
     [("            memo = float((forward_doors(params, x, act1, act2, m)[4].argmax(1) == y).float().mean())\n",
       "            memo = float((forward_doors(params, x, act1, act2, None)[4].argmax(1) == y).float().mean())\n")]),
    ("M-Hd: the diagnostics without m",
     [("        u = unit_arrays_doors(params, x, act1, act2, e1_64, m) | EL.adam_arrays(params, adam)\n",
       "        u = unit_arrays_doors(params, x, act1, act2, e1_64, None) | EL.adam_arrays(params, adam)\n")]),
    ("M-He: beta = 0.1", [("BETA = 0.01  ", "BETA = 0.1   ")]),
    ("M-Hf: m reset at every task boundary",
     [("        h_upd = [0, 0]\n", "        h_upd = [0, 0]\n        if m is not None:\n"
                                  "            m = [torch.zeros_like(q) for q in m]\n")]),
    ("M-Hg: m from the post-update forward",
     [("                        ema_update_(m, out[0].detach(), out[2].detach(), act1, act2, beta)\n",
       "                        o_ = forward_doors(params, xb, act1, act2, m)\n"
       "                        ema_update_(m, o_[0].detach(), o_[2].detach(), act1, act2, beta)\n")]),
]


# --------------------------------------------------------------------------
# S-gate: the gate under the doors, and whole updates against an independent recomputation
# --------------------------------------------------------------------------

GATE_CAPTURE = ((1, 0), (1, 40), (2, 3), (2, 120))


def s_gate(M) -> dict:
    """(i) for the z of captured H / CH forwards, autograd of (phi(z) - m) and of phi(z) give the same
    derivative bit for bit (m is a constant offset); (ii) four captured updates each of H and CH are bit-
    identical to the independent recomputation (door forward + host Adam + EMA), and the recomputation
    without m, or with a leaky second layer, or with the centring inside phi (phi(z - m)), differs
    (non-vacuity); (iii) the ELU's training derivative on a float32 grid over [-45, 5] is
    expm1(min(z, 0)) + 1, exactly 0 just below ln 2^-24 and positive just above, and differs from the
    analytic exp; (iv) the runner's dtrain diagnostics at the end of task 1 (CH) are autograd through the
    host ELU on the z of the door forward, bit for bit; (v) on a synthetic deep layer (b1 = -20) the
    runner's dtrain_zero_l1 is 1 while the analytic gate is positive, with and without m.
    Bit comparisons only."""
    failed, per = [], {}
    upd = {}
    deriv_ok = True
    for arm, (c, h) in (("H", (False, True)), ("CH", (True, True))):
        d = {"capture": GATE_CAPTURE, "end_capture": (1,)}
        rows, arrays, info = _run(M, c, h, debug=d)
        for key in GATE_CAPTURE:
            st = d["steps"][key]
            r = _ref_update(st)
            with flush_denormal():
                b = st["before"]
                for li, z in ((0, r["out"][0]), (1, r["out"][2])):
                    zz = z.detach().clone().requires_grad_(True)
                    g_door, = torch.autograd.grad((ELU.phi(zz) - b["door_m"][li]).sum(), zz)
                    zz2 = z.detach().clone().requires_grad_(True)
                    g_host, = torch.autograd.grad(ELU.phi(zz2).sum(), zz2)
                    deriv_ok &= bool(torch.equal(g_door, g_host))
            r_nom = _ref_update(st, use_m=False)
            r_lk = _ref_update(st, act2=LEAKY)
            # centring inside phi: same everything else
            with flush_denormal():
                p = [q.clone().requires_grad_(True) for q in b["params"]]
                o = _fwd(p, st["xb"], b["door_m"], inside=True)
                g_in = torch.autograd.grad(torch.nn.functional.cross_entropy(o[4], st["yb"]), p)
            upd[f"{arm}{key}"] = {"equals_reference": _after_equal(st, r),
                                  "differs_without_m": not _equal_list(st["after"]["params"], r_nom["params"])
                                  if key != (1, 0) else True,   # m = 0 at the very first update
                                  "differs_with_leaky": not _equal_list(st["after"]["params"], r_lk["params"]),
                                  "differs_inside_phi": not _equal_list(st["grads"], list(g_in))
                                  if key != (1, 0) else True}
        if arm == "CH":
            e = _ends(arrays, SPT_EP)
            end = d["ends"][1]
            x = d["x"]
            with flush_denormal():
                o = _fwd(end["params"], x, end["door_m"])
                with torch.enable_grad():
                    zz = o[2].detach().clone().requires_grad_(True)
                    gt, = torch.autograd.grad(ELU.phi(zz).sum(), zz)
            per["iv_dtrain_is_autograd"] = bool(
                np.array_equal(arrays["dtrain_mean_l2"][e[1]], gt.double().mean(0).numpy())
                and np.array_equal(arrays["dtrain_zero_l2"][e[1]], (gt == 0).double().mean(0).numpy()))
            fake_p = [torch.zeros(100, 784), torch.full((100,), -20.0), end["params"][2], end["params"][3],
                      end["params"][4], end["params"][5]]
            e1_64 = MU.mu_basis(_images(0))[1]
            ok5 = True
            for mm in (None, end["door_m"]):
                fa = M.unit_arrays_doors(fake_p, x[:32], ELU, ELU, e1_64, mm)
                ok5 &= bool((fa["dtrain_zero_l1"] == 1.0).all() and (fa["gate_mean_l1"] > 0).all())
            per["v_runner_sees_zero_derivative"] = ok5
    per["updates"] = upd
    per["i_gate_unchanged_by_m"] = bool(deriv_ok)
    per["ii_update_is_reference"] = all(u["equals_reference"] for u in upd.values())
    per["ii_nonvacuous"] = all(u["differs_without_m"] and u["differs_with_leaky"] and u["differs_inside_phi"]
                               for u in upd.values())
    grid = torch.linspace(-45, 5, 20001, dtype=torch.float32)
    with torch.enable_grad():
        gz = grid.clone().requires_grad_(True)
        gg, = torch.autograd.grad(ELU.phi(gz).sum(), gz)
    formula = torch.where(grid > 0, torch.ones_like(grid), torch.expm1(grid.clamp(max=0.0)) + 1)
    edge = torch.tensor([ZERO_Z32 - 1e-4, ZERO_Z32 + 1e-4], dtype=torch.float32)
    with torch.enable_grad():
        ez = edge.clone().requires_grad_(True)
        eg, = torch.autograd.grad(ELU.phi(ez).sum(), ez)
    per["iii_grid_equals_expm1_plus_1"] = bool(torch.equal(gg, formula))
    per["iii_zero_below_threshold"] = float(eg[0]) == 0.0 and float(eg[1]) > 0.0
    per["iii_differs_from_exp_gate"] = bool((gg != ELU.dphi(grid)).any())
    keys = ("i_gate_unchanged_by_m", "ii_update_is_reference", "ii_nonvacuous", "iii_grid_equals_expm1_plus_1",
            "iii_zero_below_threshold", "iii_differs_from_exp_gate", "iv_dtrain_is_autograd",
            "v_runner_sees_zero_derivative")
    failed += [k for k in keys if not per.get(k)]
    return {"pass": not failed, "failed_items": failed, "detail": per}


SG_MUT = [
    ("M-Ga: the centring inside phi (phi(z - m): the gate moves)",
     [("        a1 = a1 - m[0]\n", "        a1 = act1.phi(z1 - m[0])\n"),
      ("        a2 = a2 - m[1]\n", "        a2 = act2.phi(z2 - m[1])\n")]),
    ("M-Gb: the training-derivative diagnostic read from the analytic exp",
     [("            gt, = torch.autograd.grad(act.phi(zz).sum(), zz)\n", "            gt = act.dphi(zz.detach())\n")]),
    ("M-Gc: the second hidden layer is leaky",
     [("    act1, act2 = EG.ELU(1.0), EG.ELU(1.0)\n", '    act1, act2 = EG.ELU(1.0), H.ARMS["LR"]\n')]),
]


# --------------------------------------------------------------------------
# S5, S7
# --------------------------------------------------------------------------

STREAMS = ("init_sha256", "subset_idx_sha256", "labels_sha256", "batch_sha256")


def s5(M) -> dict:
    """The four arms, 3 tasks x 2 epochs each: (i) init, subset, labels and batch-order hashes identical;
    (ii) H's first update (params and Adam state) is ref's, bit for bit, and H's m is nonzero right after
    it; (iii) CH's first update is C's; (iv) C and CH train on the same input tensor, ref and H on the raw
    images; (v) the four final states (params, Adam, m) are pairwise different; (vi) C's first update is
    not ref's (non-vacuity of (ii))."""
    failed = []
    infos, first, xs, finals = {}, {}, {}, {}
    for arm, (c, h) in ARM_MAP_REG.items():
        d = {"capture": ((1, 0),)}
        rows, arrays, info = _run(M, c, h, tasks=3, debug=d)
        infos[arm] = tuple(info[k] for k in STREAMS)
        first[arm] = d["steps"][(1, 0)]
        xs[arm] = d["x"]
        finals[arm] = (info["final_state_sha256"], info["doors"]["final_door_sha256"])
    same = lambda a, b: _equal_list(first[a]["after"]["params"], first[b]["after"]["params"]) and \
        _equal_list(first[a]["after"]["m"], first[b]["after"]["m"]) and _equal_list(first[a]["after"]["v"], first[b]["after"]["v"])
    ok = {"i_streams_identical": len(set(infos.values())) == 1,
          "ii_H_first_update_is_ref": same("H", "ref") and all(bool((q != 0).any()) for q in first["H"]["after"]["door_m"]),
          "iii_CH_first_update_is_C": same("CH", "C"),
          "iv_inputs": torch.equal(xs["C"], xs["CH"]) and torch.equal(xs["ref"], xs["H"])
          and torch.equal(xs["ref"], _images(0)),
          "v_all_differ": len(set(finals.values())) == 4,
          "vi_nonvacuous": not same("C", "ref")}
    failed += [k for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": ok}


S5_MUT = [
    ("M5a: the label stream depends on the arm",
     [('    g_lab, g_batch = H.stream("rl_labels", seed), H.stream("rl_batch", seed)\n',
       '    g_lab, g_batch = H.stream("rl_labels" + door_name(door_c, door_h), seed), H.stream("rl_batch", seed)\n')]),
    ("M5b: m starts nonzero",
     [("    m = [torch.zeros(params[1].shape[0], device=device), torch.zeros(params[3].shape[0], device=device)] \\\n",
       "    m = [torch.full((params[1].shape[0],), 0.01, device=device), torch.zeros(params[3].shape[0], device=device)] \\\n")]),
    ("M5c: door C also runs in the H arm", [("    if door_c:\n        xbar = x_raw.mean(0)", "    if door_c or door_h:\n        xbar = x_raw.mean(0)")]),
]


def s7(M) -> dict:
    """The same call twice gives the same bits (arrays, hashes, rows), for ref and CH."""
    failed, per = [], {}
    for arm in ("ref", "CH"):
        c, h = ARM_MAP_REG[arm]
        r1, a1, i1 = _run(M, c, h)
        r2, a2, i2 = _run(M, c, h)
        ok = dict(i_arrays=set(a1) == set(a2) and all(np.array_equal(a1[k], a2[k], equal_nan=True) for k in a1),
                  ii_hashes=all(i1[k] == i2[k] for k in ("init_sha256", "labels_sha256", "batch_sha256",
                                                         "task1_end_state_sha256", "final_state_sha256"))
                  and i1["doors"]["final_door_sha256"] == i2["doors"]["final_door_sha256"],
                  iii_rows=r1 == r2, nonvacuous=len(a1) > 10)
        per[arm] = ok
        failed += [f"{arm}|{k}" for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S7_MUT = [("M7a: the batch order does not come from the run's own generator",
           [("            order = torch.randperm(N_IMAGES, generator=g_batch).to(device)\n",
             "            order = torch.randperm(N_IMAGES).to(device)\n")])]


# --------------------------------------------------------------------------
# S9: the CLI
# --------------------------------------------------------------------------

def s9(M) -> dict:
    """For each arm, main() on a 2-task, 2-epoch shard writes (i) per_task rows labelled with the arm and
    otherwise equal to the in-process run_doors of the registered doors, and (ii) the same units.npz
    arrays, bit for bit; (iii) the CLI's ARM_MAP is the registered one; (iv) provenance names this
    experiment and spec, the doors and beta, act2 ELU1, flush on, 1 torch thread, and hashes both new
    files; (v) the defaults are 100 tasks x 80 epochs; (vi) main() leaves flush on."""
    failed, per = [], {}
    R = load(RUNNER)
    for arm, (c, h) in ARM_MAP_REG.items():
        out = SCR / f"s9_{arm}"
        torch.set_flush_denormal(False)
        try:
            M.main(["--arm", arm, "--seeds", "0", "--tasks", "2", "--epochs", str(EP), "--out", str(out)])
            flushed = _flush_is_on()
        finally:
            torch.set_flush_denormal(False)
        rows, arrays, _ = _run(R, c, h)
        got = list(csv.DictReader((out / "per_task.csv").open()))
        ok_rows = len(got) == 2 and len(rows) == 2 and all(g["arm"] == arm for g in got) \
            and set(got[0]) - {"arm"} == set(rows[0]) - {"arm"} and all(
                str(r[k]) == v or float(r[k]) == float(v) for r, g in zip(rows, got) for k, v in g.items() if k != "arm")
        with np.load(out / "units.npz") as z:
            ok_units = set(z.files) == {f"s0_{k}" for k in arrays} and all(
                np.array_equal(z[f"s0_{k}"], v, equal_nan=True) for k, v in arrays.items())
        pv = json.loads((out / "provenance.json").read_text())
        ok_prov = (pv["experiment"] == "doors_rlmnist_1007" and pv["spec"] == "specs/spec_doors_rlmnist_1007.md"
                   and pv["doors"] == {"C": c, "H": h, "beta": BETA_REG if h else None}
                   and pv["act2"] == "ELU1" and pv.get("flush_denormal") is True and pv.get("torch_threads") == 1
                   and {"src/doors_rlmnist_run_1007.py", "src/doors_rlmnist_1007.py"} <= set(pv["code_sha256"]))
        per[arm] = {"i_rows": ok_rows, "ii_units": ok_units, "iv_provenance": ok_prov, "vi_flush_left_on": flushed}
        failed += [f"{arm}|{k}" for k, v in per[arm].items() if not v]
    per["iii_arm_map"] = M.ARM_MAP == ARM_MAP_REG
    per["v_defaults"] = M.TASKS == TASKS_RUN and M.EPOCHS == EPOCHS_RUN
    failed += [k for k in ("iii_arm_map", "v_defaults") if not per[k]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S9_MUT = [
    ("M9a: H mapped to both doors", [('"H": (False, True), "CH": (True, True)}', '"H": (True, True), "CH": (True, True)}')]),
    ("M9b: the rows keep the runner's own label", [('            r["arm"] = args.arm\n', "            pass\n")]),
    ("M9c: the default horizon is 150 tasks", [("TASKS = 100\n", "TASKS = 150\n")]),
    ("M9d: denormals not flushed", [("    torch.set_flush_denormal(True)\n", "    torch.set_flush_denormal(False)\n")]),
    ("M9e: provenance names another spec", [('SPEC = "specs/spec_doors_rlmnist_1007.md"', 'SPEC = "specs/spec_l2cap_ee_0917.md"')]),
]


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

def s_cost() -> None:
    """PROBE CH processes of the CLI at once, 2 tasks x 80 epochs, one thread each.  Gate: every process
    exits 0.  Reported: peak RSS, the slots that fit now leaving 6 GiB free with 20% headroom per job, the
    per-update time under PROBE-fold load and its 100-task projection (l2cap's real ref runs, 7-11 min at
    4-fold load, are the check on that projection)."""
    t0 = time.time()
    avail0 = _mem_available_gib()
    SCR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(PROBE):
        d = SCR / f"cost_{i}"
        procs.append((d, subprocess.Popen(
            [PY, "-m", "src.doors_rlmnist_run_1007", "--arm", "CH", "--seeds", str(i), "--tasks", "2",
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
    ok = all(r == 0 for r in rc) and len(peak) == PROBE
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": f"all {PROBE} processes exit 0",
                         "returncodes": rc, "mem_available_gib_before": avail0,
                         "peak_rss_gib_max": max(peak) if peak else None, "slots_now": slots,
                         "seconds_per_update_under_load": per_update,
                         "one_trajectory_min_projected": one / 60,
                         "grid_40_wall_hours_at_6": 40 * one / 3600 / PROBE,
                         "measured_at": dt.datetime.now().astimezone().isoformat(),
                         "seconds": round(time.time() - t0, 1)}
    dump()
    c = RESULTS["S-cost"]
    print(f"S-cost: pass={ok}  {per_update * 1e3:.3f} ms/update under {PROBE} -> {c['one_trajectory_min_projected']:.1f} min "
          f"per trajectory, {c['grid_40_wall_hours_at_6']:.2f} h for 40 at {PROBE} (peak RSS "
          f"{c['peak_rss_gib_max']:.2f} GiB, slots now {slots})", flush=True)


# --------------------------------------------------------------------------
# S8: the verdict on synthetic shards
# --------------------------------------------------------------------------

REG_T, REG_SPT, REG_WIN = 100, 6000, (51, 100)   # the REGISTERED constants
T_TABLE = {(0.975, 9): 2.262157, (0.9875, 9): 2.685011, (0.95, 9): 1.833113}
TOL_T = 5e-7 + 1e-12          # the table's 6th-decimal rounding + the bisection's 1e-12
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
    "C": {"collapse": 3, "eff2": 0.0, "noise2": 0.02},
    "H": {"collapse": None, "L": 0.60, "eff2": 0.2, "noise2": 0.02, "tail": 5.0},
    "CH": {"collapse": None, "L": 0.48, "eff2": 0.0, "noise2": 0.3},
}
DOOR_COLS = {"C": ("c",), "H": ("h",), "CH": ("c", "h")}


def synth_shards(V, scen: dict) -> dict:
    """4 arms x 10 seeds x 100 tasks.  Online accuracy: the arm's A1 at task 1 (default 0.82); an arm that
    collapses at T keeps A1 before T and sits at its floor level after (major_frac(t) - 0.01 unless given);
    an alive arm sits at L (+ centred per-seed noise in the main window).  Second-layer training
    derivative at task ends: 0.6 at task 1, 0.6 + r2 + m2[s] in the formation windows, plus eff2 + n2[a][s]
    (+ tail on unit 0) in the main window; m2 is shared by a seed's arms, so only a paired comparison
    sees eff2 when seeds differ.  The analytic gates are constants (a wrong E2 key reads one).  Step-0
    rows are garbage.  Door records: c_ok = 1, h_upd_l1 = h_upd_l2 = 6000 in every task (idle cells
    from scen['idle'])."""
    rng = np.random.default_rng(scen.get("rng", 1007))
    T, NU, SEEDS = REG_T, 100, tuple(range(10))
    arms = {a: {**DEFAULT_ARMS[a], **scen.get("arms", {}).get(a, {})} for a in DEFAULT_ARMS}
    r2 = scen.get("r2", -0.55)
    m2 = _centered(rng, 10, scen.get("seed_sd", 0.02))
    shards = {}
    for a, spec in arms.items():
        n2 = np.zeros(10) if a == "ref" else _centered(rng, 10, spec.get("noise2", 0.02))
        n1 = _centered(rng, 10, 0.0 if a == "ref" else spec.get("noise1", 0.01))
        a1v = spec.get("A1", A1)
        for si, s in enumerate(SEEDS):
            col = spec["collapse"]
            if isinstance(col, dict):
                col = col.get(s, col.get("default"))
            online = []
            for t in range(1, T + 1):
                if t == 1:
                    online.append(a1v)
                elif col is not None and t >= col:
                    online.append(spec["floor_level"] if "floor_level" in spec else _mf(t) - 0.01)
                elif col is not None:
                    online.append(a1v)
                else:
                    online.append(spec["L"] + (n1[si] if REG_WIN[0] <= t <= REG_WIN[1] else 0.0))
            pt = pd.DataFrame({"task": np.arange(1, T + 1), "online_acc": online,
                               "memo_acc": online, "major_frac": [_mf(t) for t in range(1, T + 1)]})
            if "c" in DOOR_COLS.get(a, ()):
                pt["c_ok"] = 1
                pt["c_in_ratio"] = 1e-3
            if "h" in DOOR_COLS.get(a, ()):
                pt["h_upd_l1"] = 6000
                pt["h_upd_l2"] = 6000
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
                     "dtrain_mean_l2": np.stack(G2), "dtrain_mean_l1": np.full((k, NU), 0.4),
                     "gate_mean_l2": np.full((k, NU), 0.3), "gate_mean_l1": np.full((k, NU), 0.3),
                     "zbar_l2": np.zeros((k, NU)), "b2": np.zeros((k, NU)),
                     "mu2_norm": np.full((k, 1), 5.0)}
            hs = {kk: f"h{s}" for kk in V.STREAM_KEYS}
            if s in scen.get("break", ()) and a == "H":
                hs["batch_sha256"] = "broken"
            prov = {"git_hash": "synthetic", "per_seed": {str(s): {**hs, "tasks_completed": T,
                                                                   "divergence": {"diverged": False}}}}
            shards[(a, s)] = {"units": units, "per_task": pt, "prov": prov}
    return shards


_BORDER = 2.45 * 0.05 / math.sqrt(10)      # t = 2.45: inside (t_.975,9 = 2.262, t_.9875,9 = 2.685)
S8_SCENARIOS = [
    ("designed", {},
     {"main": "RESCUED", "H_95": "RESCUED", "C": "COLLAPSED", "CH": "RESCUED_FUNCTION_ONLY", "N": "NEED_H",
      "timing_C": "EARLIER", "timing_H": "LATER", "timing_CH": "LATER"},
     {"main_E1": 0.60 - R_FLOOR, "main_E2": 0.2 + 5.0 / 100, "n_valid": 10,
      "t_half|ref|0": 5, "t_half|C|0": 3, "t_half|H|0": None, "state|ref": "FLOORED", "state|C": "FLOORED",
      "state|H": "ALIVE", "impaired|C": True, "impaired|H": False, "impaired|CH": False,
      "C_ok|C": True, "C_ok|H": True, "C_ok|CH": True, "B_ok": True}),
    ("borderline", {"arms": {"H": {"eff2": _BORDER, "noise2": 0.05, "tail": 0.0}}},
     {"main": "RESCUED_FUNCTION_ONLY", "H_95": "RESCUED", "N": "NONE_RESCUED"}, {"main_E2": _BORDER}),
    ("not_reproduced", {"arms": {"ref": {"collapse": None, "L": 0.5}}},
     {"main": "NOT_REPRODUCED", "C": "NOT_REPRODUCED", "CH": "NOT_REPRODUCED", "N": "NOT_REPRODUCED"}, {}),
    ("ref_8_of_10", {"arms": {"ref": {"collapse": {8: None, 9: None, "default": 5}, "L": 0.5}}},
     {"main": "RESCUED", "N": "NEED_H"}, {}),
    ("ref_7_of_10", {"arms": {"ref": {"collapse": {7: None, 8: None, 9: None, "default": 5}, "L": 0.5}}},
     {"main": "NOT_REPRODUCED", "N": "NOT_REPRODUCED"}, {}),
    ("split", {"arms": {"H": {"collapse": {7: 20, 8: 20, 9: 20, "default": None}}}},
     {"main": "SPLIT", "N": "NONE_RESCUED"}, {"state|H": "SPLIT"}),
    ("floor_margin", {"arms": {"CH": {"collapse": 20, "floor_level": F_WIN + 0.0005},
                               "C": {"collapse": 20, "floor_level": THR_WIN + 0.002, "eff2": 0.1}}},
     {"CH": "COLLAPSED", "C": "RESCUED", "N": "NEED_C"}, {}),
    ("need_c", {"arms": {"C": {"collapse": None, "L": 0.55, "eff2": 0.2}}},
     {"C": "RESCUED", "main": "RESCUED", "N": "NEED_C"}, {}),
    ("need_ch", {"arms": {"H": {"collapse": 12}, "CH": {"eff2": 0.2, "noise2": 0.02}}},
     {"main": "COLLAPSED", "CH": "RESCUED", "N": "NEED_CH"}, {}),
    ("alive_unresolved", {"arms": {"H": {"A1": 0.95, "L": R_FLOOR + 0.13, "noise1": 0.01, "tail": 0.0}}},
     {"main": "ALIVE_UNRESOLVED", "N": "NONE_RESCUED"}, {"state|H": "ALIVE"}),
    ("door_idle", {"idle": (("C", 3, 37, "c_ok"), ("H", 4, 80, "h_upd_l2")),
                   "arms": {"C": {"collapse": None, "L": 0.55, "eff2": 0.2}}},
     {"C": "INAPPLICABLE", "main": "INAPPLICABLE", "CH": "RESCUED_FUNCTION_ONLY", "N": "NONE_RESCUED"},
     {"C_ok|C": False, "C_ok|H": False, "C_ok|CH": True}),
    ("broken_streams", {"break": (7, 8, 9)}, {"main": "INAPPLICABLE", "N": "INAPPLICABLE"}, {"n_valid": 7}),
    ("timing_counts", {"arms": {"C": {"collapse": {8: 7, 9: 7, "default": 3}},
                                "CH": {"collapse": {6: 5, 7: 5, 8: 5, 9: 5, "default": 3}},
                                "H": {"collapse": {5: 5, 6: 5, 7: 5, 8: 5, 9: 5, "default": 3}}}},
     {"timing_C": "NO_TIMING_DIFF", "timing_CH": "EARLIER", "timing_H": "NO_TIMING_DIFF"}, {}),
    ("paired_matters", {"seed_sd": 1.0, "r2": -3.0, "arms": {"H": {"eff2": 0.25, "noise2": 0.02, "tail": 0.0}}},
     {"main": "RESCUED"}, {"main_E2": 0.25}),
]


def _pick(res: dict, key: str):
    if key == "n_valid":
        return len(res["valid_seeds"])
    if key == "B_ok":
        return res["applicability"]["B_ok"]
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
    if key.startswith("C_ok|"):
        return res["applicability"]["C"][key.split("|")[1]]
    raise KeyError(key)


def _tol_est(shards: dict) -> float:
    """(100 + 50 + 10 + 4) eps64 M: the unit mean (100), window mean (50), seed mean (10) and up to 4
    differences, each a float64 sum whose error is <= k eps64 max|term|; M = the largest |term| read."""
    M = 0.0
    for sh in shards.values():
        e = {int(t): i for i, (t, st) in enumerate(zip(sh["units"]["task"], sh["units"]["step"])) if st == REG_SPT}
        M = max(M, float(np.abs(sh["units"]["dtrain_mean_l2"][list(e.values())]).max()),
                float(np.abs(sh["per_task"]["online_acc"]).max()))
    return (100 + 50 + 10 + 4) * EPS64 * M


def s8(V) -> dict:
    """verdict.analyze on the S8 scenarios: every designed label and estimate comes out as written
    (estimates within _tol_est, flags, states and censored times exactly), and the imported t quantile
    reproduces the published table within TOL_T.  Prediction scoring is exercised on the designed
    scenario (finite Brier scores, the designed realized labels)."""
    failed, per = [], {}
    bad_t = [f"t({p},{df})" for (p, df), w in T_TABLE.items() if abs(V.t_quantile(p, df) - w) > TOL_T]
    failed += bad_t
    for name, scen, want_labels, want_vals in S8_SCENARIOS:
        sh = synth_shards(V, scen)
        res = V.analyze(sh)
        tol = _tol_est(sh)
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
                ok = g is not None and np.isfinite(g) and abs(float(g) - w) <= tol
            if not ok:
                bad.append(f"{k}: got {g} want {w}")
        if name == "designed":
            sc = V.score_predictions(res, None)
            ok_sc = (sc["claude"]["H"]["realized"] == "RESCUED" and sc["claude"]["H"]["p_realized"] == 0.62
                     and sc["parent"]["N"]["p_realized"] == 0.55
                     and all(np.isfinite(v["brier"]) for who in ("claude", "parent") for v in sc[who].values()))
            if not ok_sc:
                bad.append("scoring")
        per[name] = {"labels": res["labels"], "bad": bad, "tol_est": tol}
        failed += [f"{name}|{b}" for b in bad]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S8_MUT = [
    ("M8a: the primary arm is CH", [('MAIN_ARM = "H" ', 'MAIN_ARM = "CH" ')]),
    ("M8b: E2 reads the analytic exp gate", [('E2_KEY = "dtrain_mean_l2"\n', 'E2_KEY = "gate_mean_l2"\n')]),
    ("M8c: main label at 95%", [("MAIN_LEVEL = 0.975 ", "MAIN_LEVEL = 0.95 ")]),
    ("M8d: H's registered level 95% (the ladder reads it)", [('LEVEL_OF = {"C": 0.95, "H": 0.975, "CH": 0.95}',
                                                              'LEVEL_OF = {"C": 0.95, "H": 0.95, "CH": 0.95}')]),
    ("M8e: ref's collapse never required",
     [('    B_ok = A_ok and ref_at >= math.ceil(REF_FLOOR_FRAC * len(valid)) and B_E2["hi"] < 0\n',
       '    B_ok = A_ok and B_E2["hi"] < 0\n')]),
    ("M8f: ref must be floored in every seed", [("REF_FLOOR_FRAC = 0.8 ", "REF_FLOOR_FRAC = 1.0 ")]),
    ("M8g: ref floored in half the seeds is enough", [("REF_FLOOR_FRAC = 0.8 ", "REF_FLOOR_FRAC = 0.5 ")]),
    ("M8h: the floor margin dropped", [("FLOOR_K = 3.0 ", "FLOOR_K = 0.0 ")]),
    ("M8i: the floor at chance", [("    F, s = float(f.mean()), float(f.std(ddof=1))\n",
                                   "    F, s = CHANCE, float(f.std(ddof=1))\n")]),
    ("M8j: the pairing broken", [('        return deltas(arm, ep, win) - deltas("ref", ep, win)\n',
                                  '        return deltas(arm, ep, win) - deltas("ref", ep, win)[::-1]\n')]),
    ("M8k: unit median instead of the mean",
     [("        m, k = unit_mean(u[key][ends[t]])\n", "        m, k = float(np.median(u[key][ends[t]])), 0\n")]),
    ("M8l: E1 is the window level (not the change from the arm's own task 1)",
     [("    return float(np.mean([a.loc[t] for t in range(win[0], win[1] + 1)])) - float(a.loc[1])\n",
       "    return float(np.mean([a.loc[t] for t in range(win[0], win[1] + 1)]))\n")]),
    ("M8m: (C) for H reads layer 1 only",
     [('bool(((w["h_upd_l1"] > 0) & (w["h_upd_l2"] > 0)).all())', 'bool((w["h_upd_l1"] > 0).all())')]),
    ("M8n: (C) read in the main window only", [("DOOR_WIN = (1, 100) ", "DOOR_WIN = (51, 100) ")]),
    ("M8o: the cross-arm stream check off", [("    if len(set(infos.values())) != 1:\n", "    if False:\n")]),
    ("M8p: IMPAIRED against ref's early fit itself", [("            hits += f_arm < f_ref - (1.0 - f_ref)\n",
                                                       "            hits += f_arm < f_ref\n")]),
    ("M8q: the ladder tries H before C", [('LADDER = ("C", "H", "CH") ', 'LADDER = ("H", "C", "CH") ')]),
    ("M8r: a censored arm counted as a tie", [('    if ta is None:\n        return "later"\n',
                                               '    if ta is None:\n        return "tie"\n')]),
    ("M8s: ties counted in the sign test's n", [("        p = sign_test(max(e, l_), e + l_)\n",
                                                 "        p = sign_test(max(e, l_), len(valid))\n")]),
    ("M8t: T_half on raw accuracy instead of the fit fraction",
     [("        if f.loc[t] < HALF:\n", "        if online(pt).loc[t] < HALF:\n")]),
    ("M8u: a task's first row taken as its end point (E2)",
     [("    ends = task_ends(u)\n    base, nans = unit_mean(u[key][ends[1]])\n",
       "    ends = {int(t_): i for i, t_ in reversed(list(enumerate(u['task'])))}\n"
       "    base, nans = unit_mean(u[key][ends[1]])\n")]),
]

# --------------------------------------------------------------------------

CHECKS = {
    "S0_S-ref-is-l2cap": ("S0 the doors-off path reproduces the recorded l2cap ref shards", s0, S0_MUT, s0.__doc__, RUNNER),
    "S-C_door-c": ("S-C door C: x - x.mean(0), zero mean within the bound, not divided, same input everywhere",
                   s_c, SC_MUT, s_c.__doc__, RUNNER),
    "S-H_door-h": ("S-H door H: the beta=0.01 EMA of the uncentred phi, a constant, carried, used by every evaluation",
                   s_h, SH_MUT, s_h.__doc__, RUNNER),
    "S-gate": ("S-gate the gate under the doors is the host ELU's; captured updates equal an independent recomputation",
               s_gate, SG_MUT, s_gate.__doc__, RUNNER),
    "S5_S-branch": ("S5 the arms share their streams; H's first update is ref's and CH's is C's; all differ by task 3",
                    s5, S5_MUT, s5.__doc__, RUNNER),
    "S7_S-repro": ("S7 the same call twice gives the same bits", s7, S7_MUT, s7.__doc__, RUNNER),
    "S8_S-verdict": ("S8 the verdict returns the designed labels and estimates on synthetic shards", s8, S8_MUT,
                     s8.__doc__, VERDICT),
    "S9_S-cli": ("S9 the CLI maps, relabels and records this experiment", s9, S9_MUT, s9.__doc__, CLI),
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
    RESULTS.update({"run_id": "doors_rlmnist_1007", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "spec": "specs/spec_doors_rlmnist_1007.md", "prereg_commit": load(RUNNER).PREREG_COMMIT,
                    "git_head": H.git_hash(),
                    "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (RUNNER, CLI, VERDICT, Path(__file__).resolve(),
                                              REPO / "src" / "mucap_el_run_0916.py",
                                              REPO / "analysis" / "mucap_el_0916" / "verdict.py")},
                    "torch": torch.__version__, "python": sys.version.split()[0],
                    "threads": torch.get_num_threads()})
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

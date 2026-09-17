"""Checks for escape_ee_0917 (specs/spec_escape_ee_0917.md section 6).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/escape_ee_0917/checks.py
    ... --only S1            # development: writes results/_checks_escape_ee_0917/checks_partial.json

Each check runs on the real code and then on every mutation listed with it (an exact-once substitution
in the check's target file); the mutation MUST make the same check fail.  The run itself records the
reproduction facts the verdict requires of every seed (the prefix = resp_ee_0917's, the four unheld arms
= respdyn / resp_ee's, every shadow = N10, the capped floor = the floor, the frozen arm = the floor).

S1   the arms at the real branch (seed 0, 80 epochs, the prefix computed once): logits, reproductions,
     shadows, the holds hold and bind, the frozen arm equals the floor, the capped floor is the floor
S8   the verdict on synthetic seeds
S9   the CLI's provenance and constants
S-cost  PROBE seeds at once (three arms), projected to the full seed
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
import torch

torch.set_num_threads(1)
torch.set_flush_denormal(True)                   # the runner's setting (spec 2.1)
from src import pmnist_0905 as H                 # noqa: E402

RUNNER = REPO / "src" / "escape_ee_0917.py"
VERDICT = REPO / "analysis" / "escape_ee_0917" / "verdict.py"
OUT = REPO / "results" / "escape_ee_0917"
SCR = REPO / "results" / "_checks_escape_ee_0917"
EPS32 = float(np.finfo(np.float32).eps)
DEV = H.setup("cpu")
MNIST = H.Mnist(DEV)
PROBE = 4
PY = sys.executable
THREAD_ENV = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
RESULTS: dict = {}
DUMP_PATH = OUT / "checks.json"
TOL_CAP = 1e-4        # a capped state is above its radius by at most 4 * eps32 * R (R <= 200): 9.5e-5

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
    checks = {k: v for k, v in RESULTS.items() if k.startswith("S") and isinstance(v, dict) and "pass" in v}
    RESULTS["all_pass"] = bool(checks) and all(
        v.get("pass") and v.get("all_mutations_detected", True) for v in checks.values())
    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DUMP_PATH.write_text(json.dumps(RESULTS, indent=2, default=str))


def run_check(key, title, fn, mutations, derivation, target) -> None:
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


def _flush_is_on() -> bool:
    return float(torch.tensor([1e-30], dtype=torch.float32) * torch.tensor([1e-10], dtype=torch.float32)) == 0.0


# --------------------------------------------------------------------------
# S1: the arms at the real branch
# --------------------------------------------------------------------------

S1_ARMS = ("S2dyn_10r", "S2dyn_c12", "S2dyn_frz", "S2u30r", "S2u30r_c12", "N2r_c12")
_PRE: dict = {}
A_L2 = '        self.l2 = kind in ("c2", "c12", "c12b")\n'
A_FRZ = "            for k in range(4):\n"
A_HOLD = '    hold = None if arm["hold"] == "none" else Hold(arm["hold"], ck["params"], e1)\n'
A_SRC = '            "src": SRC if kind in ("field", "dyn") else None, "delta": delta, "hold": hold, "reset": True}\n'


def s1(M) -> dict:
    """Seed 0, 80 epochs, the prefix computed once, then 6 arms through M.run_seed: (i) branch-point
    logits equal the natural t2 net's for every arm; (ii) S2dyn_10r is respdyn's arm and S2u30r is resp_ee's
    and respdyn's (state hash, both tasks); (iii) the dyn arms' shadows are N10 (t11, t12) and d(0) is the
    fixed field; (iv) S2dyn_c12 and N2r_c12 stay within TOL_CAP of every t2 radius (all three columns
    present) and the hold wrote rows in both tasks; (v) S2dyn_frz keeps W1, b1, W2, b2 at their t2 values
    exactly and its online accuracy equals the floor's in both tasks; (vi) the capped floor's state equals
    the floor's in both tasks.  Tolerance: TOL_CAP.  No accuracy is reported (checks run before the
    predictions are registered)."""
    if "pre" not in _PRE:
        _PRE["pre"] = M.prefix(0, MNIST, 80, progress=False)
    out = SCR / "s1_run"
    M.run_seed(0, out, MNIST, 80, arms=list(S1_ARMS), progress=False, pre=_PRE["pre"])
    rows = {(r["arm"], int(r["k"])): r for r in csv.DictReader((out / "arms.csv").open())}
    f = lambda v: float(v) if v not in ("", None) else float("nan")
    t = lambda v: v == "True"
    per = {}
    per["i_logits"] = all(t(rows[(a, 1)]["logits_equal_full"]) for a in S1_ARMS)
    per["ii_repro"] = all(t(rows[("S2dyn_10r", k)]["hash_match_respdyn"]) and t(rows[("S2u30r", k)]["hash_match_respdyn"])
                          and t(rows[("S2u30r", k)]["hash_match_resp_ee"]) for k in (1, 2))
    dyn = ("S2dyn_10r", "S2dyn_c12", "S2dyn_frz")
    per["iii_shadows"] = all(t(rows[(a, k)]["shadow_hash_match_natural"]) for a in dyn for k in (1, 2)) and \
        all(t(rows[(a, 1)]["field0_equal_fixed"]) for a in dyn)
    per["iv_hold"] = all(max(f(rows[(a, k)].get(c)) for c in ("exc_q", "exc_v", "exc_w2")) <= TOL_CAP
                         for a in ("S2dyn_c12", "N2r_c12") for k in (1, 2))
    per["iv_bind"] = all(f(rows[(a, k)]["rows_par"]) + f(rows[(a, k)]["rows_perp"]) + f(rows[(a, k)]["rows_w2"]) > 0
                         and f(rows[(a, k)]["rows_w2"]) > 0 for a in ("S2dyn_c12", "N2r_c12") for k in (1, 2))
    per["v_frozen"] = all(f(rows[("S2dyn_frz", k)]["exc_frz"]) == 0.0 and t(rows[("S2dyn_frz", k)]["acc_equal_floor"])
                          for k in (1, 2))
    per["vi_capped_floor"] = all(t(rows[("S2u30r_c12", k)]["hash_equal_floor"]) for k in (1, 2))
    keys = ("i_logits", "ii_repro", "iii_shadows", "iv_hold", "iv_bind", "v_frozen", "vi_capped_floor")
    failed = [k for k in keys if not per[k]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S1_MUT = [
    ("M1a: the c12 hold leaves the second layer free", [(A_L2, '        self.l2 = kind in ("c2",)\n')]),
    ("M1b: the frozen arm lets b2 move", [(A_FRZ, "            for k in (0, 1, 2):\n")]),
    ("M1c: the holds take t10's values instead of t2's", [(A_HOLD, '    hold = None if arm["hold"] == "none" else Hold(arm["hold"], cks[SRC]["params"], e1)\n')]),
    ("M1d: the moving field follows N2 instead of N10",
     [(A_SRC, '            "src": BRANCH if kind == "dyn" else (SRC if kind == "field" else None), "delta": delta, "hold": hold, "reset": True}\n')]),
]


# --------------------------------------------------------------------------
# S8: the verdict on synthetic seeds
# --------------------------------------------------------------------------

TOL_EST = 1e-9
T_TABLE = {(0.975, 9): 2.262157, (0.95, 9): 1.833113}
BASE_E = {"S2dyn_10r": 0.32, "S2dyn_c1": 0.28, "S2dyn_c2": 0.30, "S2dyn_c12": 0.25, "S2dyn_c12b": 0.23,
          "S2dyn_frz": 0.21, "S2u30r": 0.21, "S2u30r_c12": 0.21, "S2_10r": 0.46, "S2_10r_c12": 0.35,
          "N2r": 0.72, "N2r_c12": 0.74}
BASE_CLIMB = {"S2dyn_10r": 3.5, "S2dyn_c1": 2.5, "S2dyn_c2": 3.0, "S2dyn_c12": 1.5, "S2dyn_c12b": 1.0}
EXACT = ("S2u30r", "S2u30r_c12", "S2dyn_frz")          # the floors and the frozen arm are one number
_BORDER = 0.21 + 0.11 - 2.45 * 0.02 / math.sqrt(10)    # c12 with dR at t = 2.45 (noise 0.02)
HOLD_COLS = {"S2dyn_c1": ("exc_q", "exc_v"), "S2dyn_c2": ("exc_w2",), "S2dyn_c12": ("exc_q", "exc_v", "exc_w2"),
             "S2dyn_c12b": ("exc_q", "exc_v", "exc_w2", "exc_b"), "S2dyn_frz": ("exc_frz",),
             "S2u30r_c12": ("exc_q", "exc_v", "exc_w2"), "S2_10r_c12": ("exc_q", "exc_v", "exc_w2"),
             "N2r_c12": ("exc_q", "exc_v", "exc_w2")}


def _centered(rng, n, sd):
    x = rng.normal(0.0, 1.0, n)
    x = x - x.mean()
    return x / x.std(ddof=1) * sd if sd > 0 else np.zeros(n)


def synth(V, scen: dict) -> dict:
    """10 seeds.  E = BASE_E (+ overrides) + m[seed] + n[arm][seed] (centred sd 0.005, none for EXACT and
    S2dyn_10r, the references); climb = BASE_CLIMB (+ overrides, 0 elsewhere) + centred sd 0.05 noise
    (none on the unheld arm).  All reproduction flags True unless broken."""
    rng = np.random.default_rng(scen.get("rng", 917))
    E = {**BASE_E, **scen.get("E", {})}
    CL = {**BASE_CLIMB, **scen.get("climb", {})}
    m = _centered(rng, 10, scen.get("seed_sd", 0.02))
    nE = {a: (np.zeros(10) if a in (*EXACT, "S2dyn_10r") else _centered(rng, 10, scen.get("noise", {}).get(a, 0.005)))
          for a in V.ARMS}
    nC = {a: (np.zeros(10) if a == "S2dyn_10r" else _centered(rng, 10, 0.05)) for a in V.ARMS}
    seeds = {}
    for s in range(10):
        arms = []
        for a in V.ARMS:
            for k in (1, 2):
                r = {"arm": a, "k": str(k), "online_acc": repr(float(E[a] + m[s] + nE[a][s])),
                     "climb2": repr(float(CL.get(a, 0.0) + nC[a][s])),
                     "logits_equal_full": "True" if k == 1 else "", "field0_equal_fixed": "True",
                     "shadow_hash_match_natural": "True", "hash_match_resp_ee": "True", "hash_match_respdyn": "True",
                     "rows_par": "3", "rows_perp": "3", "rows_w2": "3"}
                for c in HOLD_COLS.get(a, ()):
                    r[c] = "0.0"
                for (ba, bs, bk, col, val) in scen.get("break", ()):
                    if ba == a and bs == s and bk == k:
                        r[col] = val
                arms.append(r)
        prefix = [{"task": str(t), "hash_match_resp_ee": "True"} for t in range(1, 13)]
        seeds[s] = {"arms": arms, "prefix": prefix, "prov": {"git_hash": "synthetic"}}
    return seeds


def _brk(arm, col, val, k=1, seeds=(7, 8, 9)):
    return tuple((arm, s, k, col, val) for s in seeds)


S8_SCENARIOS = [
    ("designed", {}, {"primary": "REMAINDER_REDUCED_BY_HOLD", "climb_tracks": "CLIMB_TRACKS",
                      "fixed_field": "FIX_REMAINDER_REDUCED", "impairment_flag": "NO_IMPAIRMENT_FLAG"},
     {"R_none": 0.11, "R_c12": 0.04, "dR": 0.07, "dclimb": 2.0, "I": -0.02, "dR_fix": 0.11, "R_frz": 0.0,
      "n_valid": 10}),
    ("removed", {"E": {"S2dyn_c12": 0.21, "S2dyn_c12b": 0.205}}, {"primary": "REMAINDER_REMOVED_BY_HOLD"}, {"R_c12": 0.0}),
    ("leaves", {"E": {"S2dyn_c12": 0.32}}, {"primary": "HOLD_LEAVES_REMAINDER"}, {"dR": 0.0}),
    ("raises", {"E": {"S2dyn_c12": 0.40}}, {"primary": "HOLD_RAISES_REMAINDER"}, {}),
    ("not_manipulated", {"climb": {"S2dyn_c12": 3.5}}, {"primary": "ESCAPE_NOT_MANIPULATED"}, {"dclimb": 0.0}),
    ("not_reproduced", {"E": {"S2dyn_10r": 0.21, "S2dyn_c12": 0.18}}, {"primary": "NOT_REPRODUCED"}, {"R_none": 0.0}),
    ("borderline", {"E": {"S2dyn_c12": _BORDER}, "noise": {"S2dyn_c12": 0.02}},
     {"primary": "HOLD_LEAVES_REMAINDER"}, {"dR": 2.45 * 0.02 / math.sqrt(10)}),
    ("impaired", {"E": {"N2r_c12": 0.60}}, {"impairment_flag": "HOLD_IMPAIRS"}, {"I": 0.12}),
    ("impaired_small", {"E": {"N2r_c12": 0.70}}, {"impairment_flag": "NO_IMPAIRMENT_FLAG"}, {"I": 0.02}),
    ("climb_opposes", {"climb": {"S2dyn_10r": 1.0, "S2dyn_c1": 2.0, "S2dyn_c2": 1.5, "S2dyn_c12": 3.0, "S2dyn_c12b": 3.5}},
     {"climb_tracks": "CLIMB_OPPOSES", "primary": "ESCAPE_NOT_MANIPULATED"}, {}),
    ("shadow_broken", {"break": _brk("S2dyn_c2", "shadow_hash_match_natural", "False", k=2)},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("hold_violation", {"break": _brk("S2dyn_c12", "exc_w2", "0.01")}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("hold_idle", {"break": tuple(x for c in ("rows_par", "rows_perp", "rows_w2") for x in _brk("S2dyn_c12", c, "0", k=2))},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("repro_broken", {"break": _brk("S2_10r", "hash_match_resp_ee", "False", k=2)},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("frz_unrecorded", {"break": _brk("S2dyn_frz", "exc_frz", "")}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("paired_matters", {"seed_sd": 1.0}, {"primary": "REMAINDER_REDUCED_BY_HOLD", "climb_tracks": "CLIMB_TRACKS"},
     {"dR": 0.07}),
]


def _pick(res, key):
    if key == "n_valid":
        return len(res["valid_seeds"])
    return res["stats"][(key, 1, 0.95)]["mean"]


def s8(V) -> dict:
    """verdict.analyze on 16 synthetic scenarios: labels as designed, estimates within TOL_EST, the
    imported t quantile on the published table within 1e-6."""
    failed, per = [], {}
    failed += [f"t({p},{df})" for (p, df), w in T_TABLE.items() if abs(V.t_quantile(p, df) - w) > 1e-6]
    for name, scen, want_l, want_v in S8_SCENARIOS:
        res = V.analyze(synth(V, scen))
        bad = [f"{k}: got {res['labels'].get(k)} want {w}" for k, w in want_l.items() if res["labels"].get(k) != w]
        for k, w in want_v.items():
            g = _pick(res, k)
            if not (g is not None and math.isfinite(g) and abs(g - w) <= TOL_EST):
                bad.append(f"{k}: got {g} want {w}")
        per[name] = {"labels": res["labels"], "bad": bad}
        failed += [f"{name}|{b}" for b in bad]
    return {"pass": not failed, "failed_items": failed, "detail": per}


V_DR = '        q[("dR", k)] = q[("R_none", k)] - q[("R_c12", k)]\n'
V_RC12 = '        q[("R_c12", k)] = e("S2dyn_c12", k) - e("S2u30r_c12", k)\n'
V_LEVEL = "MAIN_LEVEL = 0.975                              # dR and R_c12\n"
V_C = '    C_ok = A_ok and st[("dclimb", 1, COMP_LEVEL)]["lo"] > 0\n'
V_B = '    B_ok = A_ok and st[("R_none", 1, COMP_LEVEL)]["lo"] > 0\n'
V_SHADOW = '            if not _true(rows[(a, k)].get("shadow_hash_match_natural")):\n'
V_TOL = "TOL_CAP = 1e-4 "
V_BIND = '            if _f(r["rows_par"]) + _f(r["rows_perp"]) + _f(r["rows_w2"]) <= 0:\n'
V_E = '    def e(a, k=1):\n        return np.array([E[(s, a, k)] for s in valid])\n'
V_IMP = '    impaired = bool(A_ok and Ist["sign"] == "+" and Ist["mean"] >= dRm)\n'
V_RHO = '    rho = [spearman([C[(s, a, 1)] for a in LADDER], [E[(s, a, 1)] for a in LADDER]) for s in valid]\n'
V_K2 = "    for k in (1, 2):\n        for a, srcs in REPRO.items():\n"
V_FIX = '        q[("dR_fix", k)] = (e("S2_10r", k) - e("S2u30r", k)) - (e("S2_10r_c12", k) - e("S2u30r_c12", k))\n'
S8_MUT = [
    ("M8a: dR reversed", [(V_DR, '        q[("dR", k)] = q[("R_c12", k)] - q[("R_none", k)]\n')]),
    ("M8b: R_c12 read off the bias-held arm", [(V_RC12, '        q[("R_c12", k)] = e("S2dyn_c12b", k) - e("S2u30r_c12", k)\n')]),
    ("M8c: primary at 95%", [(V_LEVEL, "MAIN_LEVEL = 0.95\n")]),
    ("M8d: the manipulation check dropped", [(V_C, "    C_ok = A_ok\n")]),
    ("M8e: the reproduction of R dropped", [(V_B, "    B_ok = A_ok\n")]),
    ("M8f: shadows not checked", [(V_SHADOW, "            if False:\n")]),
    ("M8g: hold tolerance 0.1", [(V_TOL, "TOL_CAP = 1e-1 ")]),
    ("M8h: binding not checked", [(V_BIND, "            if False:\n")]),
    ("M8i: pairing broken", [(V_E, '    def e(a, k=1):\n        v = np.array([E[(s, a, k)] for s in valid])\n        return v[::-1] if a == "S2u30r" else v\n')]),
    ("M8j: the impairment flag on any positive I", [(V_IMP, '    impaired = bool(A_ok and Ist["sign"] == "+")\n')]),
    ("M8k: rho of E against itself", [(V_RHO, '    rho = [spearman([E[(s, a, 1)] for a in LADDER], [E[(s, a, 1)] for a in LADDER]) for s in valid]\n')]),
    ("M8l: the reproduction checks on task 1 only", [(V_K2, "    for k in (1,):\n        for a, srcs in REPRO.items():\n")]),
    ("M8m: dR_fix reversed", [(V_FIX, '        q[("dR_fix", k)] = (e("S2_10r_c12", k) - e("S2u30r_c12", k)) - (e("S2_10r", k) - e("S2u30r", k))\n')]),
]


# --------------------------------------------------------------------------
# S9: the CLI
# --------------------------------------------------------------------------

E_FLUSH = "    torch.set_flush_denormal(True)\n"
E_SPEC = 'SPEC = "specs/spec_escape_ee_0917.md"\n'
E_BR = "BRANCH, SRC = 2, 10\n"


def s9(M) -> dict:
    """main() with 2 epochs and one arm writes provenance naming this experiment and spec, flush on,
    branch 2, source 10, 12 prefix tasks, and hashes of this runner and the modules it runs; the
    registered constants are 12 arms and the six holds."""
    out = SCR / "s9_cli"
    torch.set_flush_denormal(False)
    try:
        M.main(["--seed", "0", "--epochs", "2", "--arms", "N2r_c12", "--out", str(out)])
        flushed = _flush_is_on()
    finally:
        torch.set_flush_denormal(True)
    pv = json.loads((out / "provenance.json").read_text())
    per = {"experiment": pv["experiment"] == "escape_ee_0917", "spec": pv["spec"] == "specs/spec_escape_ee_0917.md",
           "flush": pv["flush_denormal"] is True and flushed,
           "branch": pv["branch"] == 2 and pv["src"] == 10 and pv["prefix_tasks"] == 12 and pv["cont_tasks"] == 2,
           "hashes": {"src/escape_ee_0917.py", "src/swap_ee_0917.py", "src/respdyn_ee_0917.py", "src/resp_ee_0917.py",
                      "src/l2cap_ee_0917.py", "src/mucap_el_0916.py"} <= set(pv["code_sha256"]),
           "constants": len(M.ARMS) == 12 and M.HOLDS == ("none", "c1", "c2", "c12", "c12b", "frz") and M.EPOCHS == 80}
    failed = [k for k, v in per.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S9_MUT = [
    ("M9a: flush off", [(E_FLUSH, "    torch.set_flush_denormal(False)\n")]),
    ("M9b: another spec", [(E_SPEC, 'SPEC = "specs/spec_swap_ee_0917.md"\n')]),
    ("M9c: the source at 5", [(E_BR, "BRANCH, SRC = 2, 5\n")]),
]


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

def s_cost() -> None:
    """PROBE processes at once, each a full-epoch seed with 3 arms (N2r plain, S2dyn_10r and S2dyn_c12
    with a shadow).  One full seed = prefix + 6 plain arms + 6 shadow arms.  Gate: all exit 0 and the
    memory budget leaves >= PROBE slots."""
    t0 = time.time()
    avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0] / 2 ** 20
    SCR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(PROBE):
        d = SCR / f"cost_{i}"
        procs.append((d, subprocess.Popen([PY, str(RUNNER), "--seed", str(i), "--arms", "N2r,S2dyn_10r,S2dyn_c12",
                                           "--out", str(d)], env=THREAD_ENV, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE)))
    rc, peak, pref, plain, dyn = [], [], [], [], []
    for d, pr in procs:
        err = pr.communicate()[1]
        rc.append(pr.returncode)
        if pr.returncode != 0:
            print(err.decode()[-400:], flush=True)
            continue
        pv = json.loads((d / "provenance.json").read_text())
        peak.append(pv["peak_rss_kb"] / 2 ** 20)
        pref.append(pv["seconds_prefix"])
        sec = {}
        for r in csv.DictReader((d / "arms.csv").open()):
            sec[r["arm"]] = sec.get(r["arm"], 0.0) + float(r["sec"])
        plain.append(sec["N2r"])
        dyn.append((sec["S2dyn_10r"] + sec["S2dyn_c12"]) / 2)
    slots = int(avail * 0.8 / max(peak)) if peak else 0
    one = (max(pref) + 6 * max(plain) + 6 * max(dyn)) if peak else float("nan")
    ok = all(r == 0 for r in rc) and slots >= PROBE
    RESULTS["S-cost"] = {"pass": bool(ok), "gate": f"all {PROBE} exit 0 and slots >= {PROBE}", "returncodes": rc,
                         "mem_available_gib": avail, "peak_rss_gib": max(peak) if peak else None, "slots": slots,
                         "prefix_s": max(pref) if pref else None, "plain_arm_s": max(plain) if plain else None,
                         "dyn_arm_s": max(dyn) if dyn else None, "one_seed_min": one / 60,
                         "grid_10_wall_min_at_probe": 10 * one / 60 / PROBE,
                         "measured_at": dt.datetime.now().astimezone().isoformat(),
                         "seconds": round(time.time() - t0, 1)}
    dump()
    c = RESULTS["S-cost"]
    print(f"S-cost: pass={ok}  one seed {c['one_seed_min']:.1f} min, 10 seeds at {PROBE}: "
          f"{c['grid_10_wall_min_at_probe']:.0f} min (peak {c['peak_rss_gib']:.2f} GiB, slots {slots})", flush=True)


# --------------------------------------------------------------------------

CHECKS = {
    "S1_S-arms": ("S1 the escape arms at the real branch", s1, S1_MUT, s1.__doc__, RUNNER),
    "S8_S-verdict": ("S8 the verdict on synthetic seeds", s8, S8_MUT, s8.__doc__, VERDICT),
    "S9_S-cli": ("S9 the CLI's provenance and constants", s9, S9_MUT, s9.__doc__, RUNNER),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    global DUMP_PATH
    keys = [k for k in CHECKS if not args.only or any(k.startswith(p) for p in args.only.split(","))]
    if args.only:
        SCR.mkdir(parents=True, exist_ok=True)
        DUMP_PATH = SCR / "checks_partial.json"
    RESULTS.update({"run_id": "escape_ee_0917", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "spec": "specs/spec_escape_ee_0917.md", "prereg_commit": load(RUNNER).PREREG_COMMIT,
                    "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (RUNNER, VERDICT, Path(__file__))},
                    "torch": torch.__version__})
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

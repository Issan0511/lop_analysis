"""Checks for swap_ee_0917 (specs/spec_swap_ee_0917.md section 6).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/swap_ee_0917/checks.py
    ... --only S1            # development: writes results/_checks_swap_ee_0917/checks_partial.json

Each check runs on the real code and then on every mutation listed with it (an exact-once substitution
in the check's target file); the mutation MUST make the same check fail.  Tolerances are derived in the
docstrings.  The run itself records the reproduction facts the verdict requires of every seed (ref prefix
= resp_ee_0917's, cap12 prefix = l2cap_ee_0917's archived units, NR_r = resp_ee's N10r, S2dyn_10r and
S2u30r = respdyn_ee_0917's, every shadow = its network's natural continuation); these checks pin the code
that produces those facts.

S0c  the cap12 prefix (Stepper + caps) is l2cap_ee_0917's cap12 bit for bit (tasks 1-2, 80 epochs)
S1   the arms at the real branch (seed 0, 80 epochs, prefixes computed once): branch-point logits, the
     swap is a swap (the collapsed field on the cap12 host, the healthy one on the ref host), d(0) of the
     moving arms, shadows = natural continuations (with cap12's caps), caps hold and bind, free arms grow
     past cap12's radii, NR_r = resp_ee's N10r
S8   the verdict on synthetic seeds
S9   the CLI's provenance and defaults
S-cost  PROBE seeds' worth of work at once (a reduced arm list), projected to the full seed
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

RUNNER = REPO / "src" / "swap_ee_0917.py"
VERDICT = REPO / "analysis" / "swap_ee_0917" / "verdict.py"
OUT = REPO / "results" / "swap_ee_0917"
SCR = REPO / "results" / "_checks_swap_ee_0917"
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
# S0c: the cap12 prefix
# --------------------------------------------------------------------------

A_CAPS_T1 = "        st.caps = caps\n"
A_RADII = "            caps = Caps.at(params, e1)\n"
A_PERP = "        b = int(MU.cap_perp_(params[0], self.e1, self.v_cap)[0])\n"
A_W2 = "        c = int(W2C.cap_row_norm_(params[2], self.r2)[0])\n"


def s0c(M) -> dict:
    """run_cap_prefix at 80 epochs, tasks 1-2, seed 0: every EL.unit_arrays array at both task ends is
    bit-equal to l2cap_ee_0917's archived cap12_s0 units (the archive only if its sha256 matches the
    manifest), and task 2 wrote second-layer rows (non-vacuity).  Bit comparisons only."""
    arch = M._l2cap_units(0)
    rows, _, _, _ = M.run_cap_prefix(0, MNIST, epochs=80, n_tasks=2, save_at=(), archive=arch)
    per = {"archive_loaded": arch is not None,
           "match": [r["match_l2cap_units"] for r in rows], "keys": [r.get("n_l2cap_keys") for r in rows],
           "bad": [r.get("l2cap_bad_keys") for r in rows], "rows_w2_t2": rows[1]["rows_w2"]}
    ok = {"archive_loaded": arch is not None, "both_tasks_match": all(r["match_l2cap_units"] is True for r in rows),
          "keys_compared": all((r.get("n_l2cap_keys") or 0) >= 30 for r in rows),
          "nonvacuous": rows[1]["rows_w2"] > 0 and rows[1]["rows_perp"] > 0}
    failed = [k for k, v in ok.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": {**per, **ok}}


S0C_MUT = [
    ("M0a: the caps are on in task 1 with init radii", [(A_CAPS_T1, "        st.caps = caps if t > 1 else Caps.at(params, e1)\n")]),
    ("M0b: the radii are taken at init", [(A_RADII, "            caps = Caps.at(H.init_params(seed, H.setup('cpu')), e1)\n")]),
    ("M0c: the perpendicular cap is skipped", [(A_PERP, "        b = 0\n")]),
    ("M0d: the row-norm cap is skipped", [(A_W2, "        c = 0\n")]),
]


# --------------------------------------------------------------------------
# S1: the arms at the real branch
# --------------------------------------------------------------------------

S1_ARMS = ("NC_r", "NC_free_r", "SC_fix_r", "SC_dyn_r", "NR_r", "NR_cap_r", "RR_fix_r", "RR_dyn_r")
_PRE: dict = {}
A_SHADOW_CAPS = '        self.stepper = CapStepper(self.params, self.adam, x, self.g_batch, None, src.get("caps"))\n'
A_OWN = '    if arm["caps"] == "own":\n        return ck["caps"]\n'
A_RR_SRC = '    "RR_fix_r": _arm("R10", "field", src="C10"),\n'
A_ANCHOR = "        sh = RE.build_shift(arm, cks)\n        return sh, RE.make_forward(sh), None\n"


def s1(M) -> dict:
    """Seed 0, 80 epochs, the two prefixes computed once (the prefix code is S0c's and resp_ee's), then
    8 arms through M.run_seed: (i) branch-point logits equal the host's natural logits for every arm;
    (ii) the swap is real: at the start the cap12 host with ref's field has >= 0.9 of (unit, image) pairs
    with a zero training derivative and the ref host with cap12's field <= 0.01 (ref's t10 has 0.94,
    cap12's 1e-4: l2cap_ee_0917), while each host's natural arm keeps its own; (iii) d(0) of SC_dyn_r and
    RR_dyn_r is the fixed field; (iv) both shadows are their networks' natural continuations (state hash =
    the prefix's t11 and t12); (v) the capped arms stay within TOL_CAP of their radii and wrote rows, the
    free cap12 arm ends above cap12's radii (non-vacuity), the 'here' caps bind on the ref host; (vi)
    NR_r is resp_ee_0917's N10r (state hash, both tasks).  Tolerance: TOL_CAP."""
    if "pre" not in _PRE:
        _PRE["pre"] = M.prefixes(0, MNIST, 80, progress=False)
    out = SCR / "s1_run"
    M.run_seed(0, out, MNIST, 80, arms=list(S1_ARMS), progress=False, pre=_PRE["pre"])
    rows = {(r["arm"], int(r["k"])): r for r in csv.DictReader((out / "arms.csv").open())}
    f = lambda v: float(v) if v not in ("", None) else float("nan")
    t = lambda v: v == "True"
    per = {}
    per["i_logits"] = all(t(rows[(a, 1)]["logits_equal_full"]) for a in S1_ARMS)
    z = {a: f(rows[(a, 1)]["zero2_start"]) for a in S1_ARMS}
    per["zero2_start"] = z
    per["ii_swap_real"] = (z["SC_fix_r"] >= 0.9 and z["SC_dyn_r"] >= 0.9 and z["RR_fix_r"] <= 0.01
                           and z["RR_dyn_r"] <= 0.01 and z["NC_r"] <= 0.01 and z["NR_r"] >= 0.9)
    per["iii_field0"] = all(t(rows[(a, 1)]["field0_equal_fixed"]) for a in ("SC_dyn_r", "RR_dyn_r"))
    per["iv_shadows"] = all(t(rows[(a, k)]["shadow_hash_match_natural"]) for a in ("SC_dyn_r", "RR_dyn_r")
                            for k in (1, 2))
    capped = ("NC_r", "SC_fix_r", "SC_dyn_r", "NR_cap_r")
    per["v_caps_hold"] = all(max(f(rows[(a, k)]["exc_q"]), f(rows[(a, k)]["exc_v"]), f(rows[(a, k)]["exc_w2"]))
                             <= TOL_CAP for a in capped for k in (1, 2))
    per["v_caps_bind"] = all(f(rows[(a, k)]["rows_par"]) + f(rows[(a, k)]["rows_perp"]) + f(rows[(a, k)]["rows_w2"]) > 0
                             for a in capped for k in (1, 2))
    per["v_free_grows"] = max(f(rows[("NC_free_r", 2)]["exc_v_own"]), f(rows[("NC_free_r", 2)]["exc_w2_own"])) > TOL_CAP
    per["vi_NR_is_N10r"] = all(t(rows[("NR_r", k)]["hash_match_resp_ee"]) for k in (1, 2))
    # E is deliberately not reported here: the checks run before the predictions are registered
    keys = ("i_logits", "ii_swap_real", "iii_field0", "iv_shadows", "v_caps_hold", "v_caps_bind",
            "v_free_grows", "vi_NR_is_N10r")
    failed = [k for k in keys if not per[k]]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S1_MUT = [
    ("M1a: cap12's shadow runs without cap12's caps", [(A_SHADOW_CAPS, "        self.stepper = CapStepper(self.params, self.adam, x, self.g_batch, None, None)\n")]),
    ("M1b: 'own' caps are not applied", [(A_OWN, '    if arm["caps"] == "own":\n        return None\n')]),
    ("M1c: the restore arm takes ref's own field (no swap)", [(A_RR_SRC, '    "RR_fix_r": _arm("R10", "field", src="R10"),\n')]),
    ("M1d: fixed fields without the anchor (forward value moves too)",
     [(A_ANCHOR, "        sh = RE.build_shift(arm, cks)\n        return sh, (None if sh is None else RD.make_forward_noanchor(sh)), None\n")]),
]


# --------------------------------------------------------------------------
# S8: the verdict on synthetic seeds
# --------------------------------------------------------------------------

TOL_EST = 1e-9
T_TABLE = {(0.975, 9): 2.262157, (0.95, 9): 1.833113}
BASE_E = {"NC_r": 0.85, "NC_free_r": 0.80, "SC_fix_r": 0.45, "SC_dyn_r": 0.33, "SC_dyn_free_r": 0.25,
          "SC_u30_r": 0.22, "NR_r": 0.17, "NR_cap_r": 0.19, "RR_fix_r": 0.80, "RR_dyn_r": 0.70,
          "RR_dyn_cap_r": 0.78, "RR_u30_r": 0.16, "S2dyn_10r": 0.32, "S2u30r": 0.21}
_BORDER = 2.45 * 0.02 / math.sqrt(10)    # t = 2.45: inside (t_.975,9 = 2.262, t_.9875,9 = 2.685)


def _centered(rng, n, sd):
    x = rng.normal(0.0, 1.0, n)
    x = x - x.mean()
    return x / x.std(ddof=1) * sd if sd > 0 else np.zeros(n)


def synth(V, scen: dict) -> dict:
    """10 seeds.  E(arm, seed, k) = BASE_E (+ scen overrides) + m[seed] (shared by every arm) + n[arm][seed]
    (centred, sd 0.005; zero for NR_r and SC_u30_r, the references of the designed contrasts, and for arms
    listed in scen['exact']).  Every reproduction flag is True unless a scenario breaks it."""
    rng = np.random.default_rng(scen.get("rng", 917))
    E = {**BASE_E, **scen.get("E", {})}
    m = _centered(rng, 10, scen.get("seed_sd", 0.02))
    noise = {a: (np.zeros(10) if a in ("NR_r", "SC_u30_r", *scen.get("exact", ())) else
                 _centered(rng, 10, scen.get("noise", {}).get(a, 0.005))) for a in V.ARMS}
    seeds = {}
    for s in range(10):
        arms = []
        for a in V.ARMS:
            for k in (1, 2):
                r = {"arm": a, "k": str(k), "online_acc": repr(float(E[a] + m[s] + noise[a][s])),
                     "logits_equal_full": "True" if k == 1 else "", "field0_equal_fixed": "True",
                     "shadow_hash_match_natural": "True", "shadow_hash_match_prefix": "True",
                     "hash_match_resp_ee": "True", "hash_match_respdyn": "True",
                     "exc_q": "0.0", "exc_v": "0.0", "exc_w2": "0.0",
                     "rows_par": "3", "rows_perp": "3", "rows_w2": "3"}
                for (ba, bs, bk, col, val) in scen.get("break", ()):
                    if ba == a and bs == s and bk == k:
                        r[col] = val
                arms.append(r)
        prefix = [{"net": n_, "task": str(t), "hash_match_resp_ee": "True", "match_l2cap_units": "True"}
                  for n_ in ("R", "C") for t in range(1, 13)]
        for (bs, bnet, bt, col, val) in scen.get("break_prefix", ()):
            if bs == s:
                for r in prefix:
                    if r["net"] == bnet and r["task"] == str(bt):
                        r[col] = val
        seeds[s] = {"arms": arms, "prefix": prefix, "prov": {"git_hash": "synthetic"}}
    return seeds


def _brk(arm, col, val, seeds=(7, 8, 9), k=1):
    return tuple((arm, s, k, col, val) for s in seeds)


S8_SCENARIOS = [
    ("designed", {},
     {"Q1": "REMAINDER_FOLLOWS_HOST", "Q2": "BOTH_WAYS", "Q3_P_C": "GROWTH_COSTS", "Q3_P_R": "GROWTH_COSTS",
      "Q3_P_N": "GROWTH_COSTS", "Q3_P_F": "GROWTH_COSTS"},
     {"R_C": 0.11, "R_R": 0.01, "D": 0.10, "R_T2": 0.11, "TE": 0.68, "L_S": 0.52, "G_R": 0.53, "P_C": 0.08,
      "P_N": 0.02, "n_valid": 10}),
    ("in_both", {"E": {"RR_u30_r": 0.06}}, {"Q1": "REMAINDER_IN_BOTH"}, {"R_R": 0.11, "D": 0.0}),
    ("none", {"E": {"SC_u30_r": 0.33}}, {"Q1": "NO_REMAINDER_IN_NONPUSHING_HOST"}, {"R_C": 0.0}),
    ("larger_in_pushing", {"E": {"RR_u30_r": -0.05}}, {"Q1": "REMAINDER_LARGER_IN_PUSHING_HOST"}, {}),
    ("borderline", {"E": {"SC_dyn_r": 0.22 + _BORDER, "SC_fix_r": 0.30, "SC_dyn_free_r": 0.10},
                    "noise": {"SC_dyn_r": 0.02}},
     {"Q1": "NO_REMAINDER_IN_NONPUSHING_HOST"}, {"R_C": _BORDER}),
    ("not_reproduced", {"E": {"NC_r": 0.17, "NC_free_r": 0.15}}, {"Q1": "NOT_REPRODUCED", "Q2": "NOT_REPRODUCED"}, {}),
    ("r_t2_absent", {"E": {"S2dyn_10r": 0.21}, "exact": ("S2dyn_10r", "S2u30r")},
     {"Q1": "NOT_REPRODUCED"}, {"R_T2": 0.0}),
    ("shadow_broken", {"break": _brk("RR_dyn_r", "shadow_hash_match_natural", "False", k=2)},
     {"Q1": "INAPPLICABLE"}, {"n_valid": 7}),
    ("cap_violation", {"break": _brk("SC_dyn_r", "exc_w2", "0.01")}, {"Q1": "INAPPLICABLE"}, {"n_valid": 7}),
    ("cap_idle", {"break": tuple(x for c in ("rows_par", "rows_perp", "rows_w2")
                                 for x in _brk("NR_cap_r", c, "0", k=2))},
     {"Q1": "INAPPLICABLE"}, {"n_valid": 7}),
    ("sink_only", {"E": {"RR_dyn_r": 0.17, "RR_dyn_cap_r": 0.25}}, {"Q2": "SINK_ONLY"}, {"G_R": 0.0}),
    ("paired_matters", {"seed_sd": 1.0}, {"Q1": "REMAINDER_FOLLOWS_HOST", "Q2": "BOTH_WAYS"}, {"D": 0.10}),
    ("prefix_mismatch", {"break_prefix": tuple((s, "C", 11, "match_l2cap_units", "False") for s in (7, 8, 9))},
     {"Q1": "INAPPLICABLE"}, {"n_valid": 7}),
    ("respdyn_k2", {"break": _brk("S2dyn_10r", "hash_match_respdyn", "False", k=2)},
     {"Q1": "INAPPLICABLE"}, {"n_valid": 7}),
]


def _pick(res, key):
    if key == "n_valid":
        return len(res["valid_seeds"])
    return res["stats"][(key, 1, 0.95)]["mean"]


def s8(V) -> dict:
    """verdict.analyze on 14 synthetic scenarios: labels as designed, estimates within TOL_EST, the
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


V_RC = '        q[("R_C", k)] = e("SC_dyn_r", k) - e("SC_u30_r", k)\n'
V_RR = '        q[("R_R", k)] = e("NR_r", k) - e("RR_u30_r", k)\n'
V_D = '        q[("D", k)] = q[("R_C", k)] - q[("R_R", k)]\n'
V_LS = '        q[("L_S", k)] = e("NC_r", k) - e("SC_dyn_r", k)\n'
V_PN = '        q[("P_N", k)] = e("NR_cap_r", k) - e("NR_r", k)\n'
V_LEVEL = "MAIN_LEVEL = 0.975                              # Q1: two quantities (Bonferroni)\n"
V_B = '    B_ok = A_ok and all(b["lo"] > 0 for b in B.values())\n'
V_SHADOW = '            if not _true(rows[(a, k)].get("shadow_hash_match_natural")):\n'
V_TOL = "TOL_CAP = 1e-4 "
V_BIND = '            if _f(r["rows_par"]) + _f(r["rows_perp"]) + _f(r["rows_w2"]) <= 0:\n'
V_E = '    def e(a, k=1):\n        return np.array([E[(s, a, k)] for s in valid])\n'
V_PREFIX = '    if len(cap) != PREFIX_T or not all(_true(r["match_l2cap_units"]) for r in cap):\n'
V_K2 = '    for k in (1, 2):\n        if not _true(rows[("S2dyn_10r", k)].get("shadow_hash_match_prefix")):\n'
S8_MUT = [
    ("M8a: R_C against the fixed field", [(V_RC, '        q[("R_C", k)] = e("SC_fix_r", k) - e("SC_u30_r", k)\n')]),
    ("M8b: R_R against the cap12 floor", [(V_RR, '        q[("R_R", k)] = e("NR_r", k) - e("SC_u30_r", k)\n')]),
    ("M8c: D reversed", [(V_D, '        q[("D", k)] = q[("R_R", k)] - q[("R_C", k)]\n')]),
    ("M8d: Q1 at 95%", [(V_LEVEL, "MAIN_LEVEL = 0.95\n")]),
    ("M8e: (B) never required", [(V_B, "    B_ok = A_ok\n")]),
    ("M8f: shadows not checked", [(V_SHADOW, "            if False:\n")]),
    ("M8g: cap tolerance 0.1", [(V_TOL, "TOL_CAP = 1e-1 ")]),
    ("M8h: binding not checked", [(V_BIND, "            if False:\n")]),
    ("M8i: pairing broken", [(V_E, '    def e(a, k=1):\n        v = np.array([E[(s, a, k)] for s in valid])\n        return v[::-1] if a == "NR_r" else v\n')]),
    ("M8j: the sink loss with the fixed field", [(V_LS, '        q[("L_S", k)] = e("NC_r", k) - e("SC_fix_r", k)\n')]),
    ("M8k: the cap12 prefix not checked", [(V_PREFIX, "    if len(cap) != PREFIX_T:\n")]),
    ("M8l: the reproduction checks on task 1 only",
     [(V_K2, '    for k in (1,):\n        if not _true(rows[("S2dyn_10r", k)].get("shadow_hash_match_prefix")):\n')]),
    ("M8m: P_N reversed", [(V_PN, '        q[("P_N", k)] = e("NR_r", k) - e("NR_cap_r", k)\n')]),
]


# --------------------------------------------------------------------------
# S9: the CLI
# --------------------------------------------------------------------------

E_FLUSH = "    torch.set_flush_denormal(True)\n"
E_SPEC = 'SPEC = "specs/spec_swap_ee_0917.md"\n'
E_BRANCH = "BRANCH = 10\n"


def s9(M) -> dict:
    """main() with 2 epochs and one arm writes provenance naming this experiment and spec, flush on,
    the branch 10 and the 2 continuation tasks, and hashes of this runner and every module it runs;
    the registered constants are branch 10, 80 epochs, 12 prefix tasks and 14 arms."""
    out = SCR / "s9_cli"
    torch.set_flush_denormal(False)
    try:
        M.main(["--seed", "0", "--epochs", "2", "--arms", "NC_r", "--out", str(out)])
        flushed = _flush_is_on()
    finally:
        torch.set_flush_denormal(True)
    pv = json.loads((out / "provenance.json").read_text())
    per = {"experiment": pv["experiment"] == "swap_ee_0917", "spec": pv["spec"] == "specs/spec_swap_ee_0917.md",
           "flush": pv["flush_denormal"] is True and flushed,
           "branch": pv["branch"] == 10 and pv["cont_tasks"] == 2 and pv["prefix_tasks"] == 12,
           "hashes": {"src/swap_ee_0917.py", "src/respdyn_ee_0917.py", "src/resp_ee_0917.py", "src/l2cap_ee_0917.py",
                      "src/mucap_el_0916.py", "src/mucap_el_run_0916.py"} <= set(pv["code_sha256"]),
           "constants": M.BRANCH == 10 and M.EPOCHS == 80 and M.PREFIX_T == 12 and len(M.ALL_ARMS) == 14}
    failed = [k for k, v in per.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S9_MUT = [
    ("M9a: flush off", [(E_FLUSH, "    torch.set_flush_denormal(False)\n")]),
    ("M9b: another spec", [(E_SPEC, 'SPEC = "specs/spec_respdyn_ee_0917.md"\n')]),
    ("M9c: branch at 5", [(E_BRANCH, "BRANCH = 5\n")]),
]


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

def s_cost() -> None:
    """PROBE processes at once, each a full-epoch seed with 3 arms (NC_r, SC_dyn_r, RR_dyn_r: one plain
    and two with a shadow).  Projection of one full seed: the measured prefix time + the per-arm times
    scaled to 14 arms, of which 5 carry a shadow (a shadow arm costs the measured dyn-arm time).
    Gate: all exit 0 and the memory budget leaves >= PROBE slots."""
    t0 = time.time()
    avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0] / 2 ** 20
    SCR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(PROBE):
        d = SCR / f"cost_{i}"
        procs.append((d, subprocess.Popen([PY, str(RUNNER), "--seed", str(i), "--arms", "NC_r,SC_dyn_r,RR_dyn_r",
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
        plain.append(sec["NC_r"])
        dyn.append((sec["SC_dyn_r"] + sec["RR_dyn_r"]) / 2)
    slots = int(avail * 0.8 / max(peak)) if peak else 0
    one = (max(pref) + 9 * max(plain) + 5 * max(dyn)) if peak else float("nan")
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
    "S0c_S-cap-prefix": ("S0c the cap12 prefix is l2cap_ee_0917's cap12", s0c, S0C_MUT, s0c.__doc__, RUNNER),
    "S1_S-arms": ("S1 the swap arms at the real branch", s1, S1_MUT, s1.__doc__, RUNNER),
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
    RESULTS.update({"run_id": "swap_ee_0917", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "spec": "specs/spec_swap_ee_0917.md", "prereg_commit": load(RUNNER).PREREG_COMMIT,
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

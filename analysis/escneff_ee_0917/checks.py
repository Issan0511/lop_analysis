"""Checks for escneff_ee_0917 (specs/spec_escneff_ee_0917.md section 6).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analysis/escneff_ee_0917/checks.py
    ... --only S1            # development: writes results/_checks_escneff_ee_0917/checks_partial.json

The runner (src/escneff_ee_0917.py) is a thin wrapper around src/escape_ee_0917.py, which is unchanged
since escape's registration.  Each check runs on the real code and then on every mutation listed with it
(an exact-once substitution in the check's target file); the mutation MUST make the same check fail.

S1   escape's arm check, through this runner (seed 0, where resp_ee / respdyn records exist): logits,
     reproductions, shadows, holds hold and bind, frozen = floor, capped floor = floor; mutations in the
     escape runner, passed to this runner as its `es`
S1b  the prefix comparison: seed 10 matches neff_pred_0917's record on all 12 tasks and has no resp_ee
     record; seed 13 has no record at all (every comparison empty)
S8   the verdict on synthetic seeds 10-19
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

RUNNER = REPO / "src" / "escneff_ee_0917.py"
ESC_RUNNER = REPO / "src" / "escape_ee_0917.py"
VERDICT = REPO / "analysis" / "escneff_ee_0917" / "verdict.py"
OUT = REPO / "results" / "escneff_ee_0917"
SCR = REPO / "results" / "_checks_escneff_ee_0917"
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
    """Seed 0, 80 epochs, the prefix computed once, then 6 arms through this runner's run_seed with M
    (the escape runner, possibly mutated) as its engine: (i) branch-point
    logits equal the natural t2 net's for every arm; (ii) S2dyn_10r is respdyn's arm and S2u30r is resp_ee's
    and respdyn's (state hash, both tasks); (iii) the dyn arms' shadows are N10 (t11, t12) and d(0) is the
    fixed field; (iv) S2dyn_c12 and N2r_c12 stay within TOL_CAP of every t2 radius (all three columns
    present) and the hold wrote rows in both tasks; (v) S2dyn_frz keeps W1, b1, W2, b2 at their t2 values
    exactly and its online accuracy equals the floor's in both tasks; (vi) the capped floor's state equals
    the floor's in both tasks.  Tolerance: TOL_CAP.  No accuracy is reported (checks run before the
    predictions are registered).  (vii) provenance names this experiment and its engine."""
    NR = load(RUNNER)
    if "pre" not in _PRE:
        _PRE["pre"] = NR.prefix(0, MNIST, 80, progress=False, es=M)
    out = SCR / "s1_run"
    NR.run_seed(0, out, MNIST, 80, arms=list(S1_ARMS), progress=False, pre=_PRE["pre"], es=M)
    pv = json.loads((out / "provenance.json").read_text())
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
    per["vii_provenance"] = (pv["experiment"] == "escneff_ee_0917" and pv["runner_experiment"] == "escape_ee_0917"
                             and pv["prefix_hash_match_resp_ee"] == 12 and pv["prefix_record_tasks_neff_pred"] == 0)
    keys = ("i_logits", "ii_repro", "iii_shadows", "iv_hold", "iv_bind", "v_frozen", "vi_capped_floor",
            "vii_provenance")
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
# S1b: the prefix comparison with neff_pred_0917's record
# --------------------------------------------------------------------------

R_REC = '    rec = {int(r["task"]): r["state_sha256"] for r in SW._csv_rows(NEFF_PRED_EE / f"s{seed}" / "prefix.csv")}\n'
R_MATCH = '        r["hash_match_neff_pred"] = None if r["task"] not in rec else rec[r["task"]] == r["state_sha256"]\n'
_PREFIX_CACHE: dict = {}


class _CachedES:
    """The real escape runner with its prefix computed once per seed (S1b's mutations are in the wrapper)."""

    def __init__(self):
        self.mod = load(ESC_RUNNER)

    def prefix(self, seed, mnist, epochs, progress):
        if seed not in _PREFIX_CACHE:
            _PREFIX_CACHE[seed] = self.mod.prefix(seed, mnist, epochs, progress)
        pre = _PREFIX_CACHE[seed]
        return {**pre, "rows": [dict(r) for r in pre["rows"]]}


def s1b(M) -> dict:
    """M.prefix(seed, es=cached escape runner): seed 10's 12 prefix rows all match neff_pred_0917's record
    (state sha256) and have no resp_ee record (None), and the record covers 12 tasks; seed 13's rows have
    no record in either (all None, 0 tasks).  Bit comparisons only."""
    es = _CachedES()
    p10 = M.prefix(10, MNIST, 80, progress=False, es=es)
    p13 = M.prefix(13, MNIST, 80, progress=False, es=es)
    per = {"seed10_matches": len(p10["rows"]) == 12 and all(r["hash_match_neff_pred"] is True for r in p10["rows"]),
           "seed10_no_resp_ee": all(r["hash_match_resp_ee"] is None for r in p10["rows"]),
           "seed10_record_tasks": p10["record_tasks_neff_pred"] == 12,
           "seed13_unrecorded": (p13["record_tasks_neff_pred"] == 0 and
                                 all(r["hash_match_neff_pred"] is None and r["hash_match_resp_ee"] is None
                                     for r in p13["rows"]))}
    failed = [k for k, v in per.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S1B_MUT = [
    ("M1e: the record read from resp_ee's folder",
     [(R_REC, '    rec = {int(r["task"]): r["state_sha256"] for r in SW._csv_rows(ROOT / "results" / "resp_ee_0917" / "runs" / f"s{seed}" / "prefix.csv")}\n')]),
    ("M1f: the record shifted by one task",
     [(R_REC, '    rec = {int(r["task"]) + 1: r["state_sha256"] for r in SW._csv_rows(NEFF_PRED_EE / f"s{seed}" / "prefix.csv")}\n')]),
    ("M1g: the record read for the wrong seed",
     [(R_REC, '    rec = {int(r["task"]): r["state_sha256"] for r in SW._csv_rows(NEFF_PRED_EE / f"s{seed + 1}" / "prefix.csv")}\n')]),
]


# --------------------------------------------------------------------------
# S8: the verdict on synthetic seeds 10-19
# --------------------------------------------------------------------------

TOL_EST = 1e-9
T_TABLE = {(0.975, 9): 2.262157, (0.95, 9): 1.833113}
SEEDS_REG = tuple(range(10, 20))
BASE_E = {"S2dyn_10r": 0.32, "S2dyn_c1": 0.28, "S2dyn_c2": 0.30, "S2dyn_c12": 0.25, "S2dyn_c12b": 0.23,
          "S2dyn_frz": 0.21, "S2u30r": 0.21, "S2u30r_c12": 0.21, "S2_10r": 0.46, "S2_10r_c12": 0.35,
          "N2r": 0.72, "N2r_c12": 0.74}
BASE_NEFF = {"S2dyn_10r": 0.0095, "S2dyn_c1": 0.0074, "S2dyn_c2": 0.0090, "S2dyn_c12": 0.0064, "S2dyn_c12b": 0.0060,
             "S2dyn_frz": 0.0028, "S2u30r": 0.0, "S2u30r_c12": 0.0, "S2_10r": 0.024, "S2_10r_c12": 0.017,
             "N2r": 0.50, "N2r_c12": 0.50}
BASE_CLIMB = {"S2dyn_10r": 3.5, "S2dyn_c1": 2.5, "S2dyn_c2": 3.0, "S2dyn_c12": 1.5, "S2dyn_c12b": 1.0}
EXACT = ("S2u30r", "S2u30r_c12", "S2dyn_frz")
_BORDER = 0.21 + 0.11 - 2.45 * 0.02 / math.sqrt(10)
HOLD_COLS = {"S2dyn_c1": ("exc_q", "exc_v"), "S2dyn_c2": ("exc_w2",), "S2dyn_c12": ("exc_q", "exc_v", "exc_w2"),
             "S2dyn_c12b": ("exc_q", "exc_v", "exc_w2", "exc_b"), "S2dyn_frz": ("exc_frz",),
             "S2u30r_c12": ("exc_q", "exc_v", "exc_w2"), "S2_10r_c12": ("exc_q", "exc_v", "exc_w2"),
             "N2r_c12": ("exc_q", "exc_v", "exc_w2")}
NEFF_RECORDED = (10, 11, 12)


def _centered(rng, n, sd):
    x = rng.normal(0.0, 1.0, n)
    x = x - x.mean()
    return x / x.std(ddof=1) * sd if sd > 0 else np.zeros(n)


def synth(V, scen: dict) -> dict:
    """Seeds 10-19.  E = BASE_E (+ overrides) + m[seed] + nE[arm][seed] (centred sd 0.005; none for the
    floors, the frozen arm and S2dyn_10r).  n_eff = BASE_NEFF (+ overrides) + tie[arm] * nE[arm][seed] +
    centred sd 5e-5 noise (none on S2dyn_10r); the default tie on S2dyn_c12 (0.02) makes dneff and dR move
    together across seeds.  climb = BASE_CLIMB (+ overrides) + centred sd 0.05.  Records as the real run
    writes them: no resp_ee / respdyn record (empty), a neff_pred prefix record (True) on seeds 10-12."""
    rng = np.random.default_rng(scen.get("rng", 917))
    E = {**BASE_E, **scen.get("E", {})}
    NF = {**BASE_NEFF, **scen.get("neff", {})}
    CL = {**BASE_CLIMB, **scen.get("climb", {})}
    tie = scen.get("tie", {"S2dyn_c12": 0.02})
    m = _centered(rng, 10, scen.get("seed_sd", 0.02))
    nE = {a: (np.zeros(10) if a in (*EXACT, "S2dyn_10r") else _centered(rng, 10, scen.get("noise", {}).get(a, 0.005)))
          for a in V.ARMS}
    nN = {a: (np.zeros(10) if a in (*EXACT, "S2dyn_10r") else _centered(rng, 10, 5e-5)) for a in V.ARMS}
    nC = {a: (np.zeros(10) if a == "S2dyn_10r" else _centered(rng, 10, 0.05)) for a in V.ARMS}
    seeds = {}
    for i, s in enumerate(SEEDS_REG):
        arms = []
        for a in V.ARMS:
            for k in (1, 2):
                r = {"arm": a, "k": str(k), "online_acc": repr(float(E[a] + m[i] + nE[a][i])),
                     "nbar_neffT2": repr(float(NF[a] + tie.get(a, 0.0) * nE[a][i] + nN[a][i])),
                     "climb2": repr(float(CL.get(a, 0.0) + nC[a][i])),
                     "logits_equal_full": "True" if k == 1 else "", "field0_equal_fixed": "True",
                     "shadow_hash_match_natural": "True", "hash_match_resp_ee": "", "hash_match_respdyn": "",
                     "rows_par": "3", "rows_perp": "3", "rows_w2": "3"}
                for c in HOLD_COLS.get(a, ()):
                    r[c] = "0.0"
                for (ba, bs, bk, col, val) in scen.get("break", ()):
                    if ba == a and bs == s and bk == k:
                        r[col] = val
                arms.append(r)
        rec = "True" if s in NEFF_RECORDED else ""
        prefix = [{"task": str(t), "hash_match_resp_ee": "", "hash_match_neff_pred": rec} for t in range(1, 13)]
        for (ps, col, val) in scen.get("prefix_break", ()):
            if ps == s:
                for p in prefix:
                    p[col] = val
        seeds[s] = {"arms": arms, "prefix": prefix, "prov": {"git_hash": "synthetic"}}
    return seeds


def _brk(arm, col, val, k=1, seeds=(17, 18, 19)):
    return tuple((arm, s, k, col, val) for s in seeds)


S8_SCENARIOS = [
    ("designed", {}, {"primary": "REMAINDER_REDUCED_BY_HOLD", "neff_tracks": "NEFF_TRACKS", "slope": "SLOPE_POSITIVE",
                      "fixed_field": "FIX_REMAINDER_REDUCED", "impairment_flag": "NO_IMPAIRMENT_FLAG"},
     {"R_none": 0.11, "R_c12": 0.04, "dR": 0.07, "dneff": 0.0031, "I": -0.02, "dR_fix": 0.11, "R_frz": 0.0,
      "n_valid": 10}),
    ("removed", {"E": {"S2dyn_c12": 0.21, "S2dyn_c12b": 0.205}}, {"primary": "REMAINDER_REMOVED_BY_HOLD"}, {"R_c12": 0.0}),
    ("leaves", {"E": {"S2dyn_c12": 0.32}}, {"primary": "HOLD_LEAVES_REMAINDER"}, {"dR": 0.0}),
    ("raises", {"E": {"S2dyn_c12": 0.40}}, {"primary": "HOLD_RAISES_REMAINDER"}, {}),
    ("not_manipulated", {"neff": {"S2dyn_c12": 0.0095}, "tie": {}}, {"primary": "ESCAPE_NOT_MANIPULATED"}, {"dneff": 0.0}),
    ("climb_flat", {"climb": {"S2dyn_c12": 3.5, "S2dyn_c1": 3.5, "S2dyn_c2": 3.5, "S2dyn_c12b": 3.5}},
     {"primary": "REMAINDER_REDUCED_BY_HOLD", "neff_tracks": "NEFF_TRACKS"}, {}),
    ("not_reproduced", {"E": {"S2dyn_10r": 0.21, "S2dyn_c12": 0.18}}, {"primary": "NOT_REPRODUCED"}, {"R_none": 0.0}),
    ("borderline", {"E": {"S2dyn_c12": _BORDER}, "noise": {"S2dyn_c12": 0.02}},
     {"primary": "HOLD_LEAVES_REMAINDER"}, {"dR": 2.45 * 0.02 / math.sqrt(10)}),
    ("impaired", {"E": {"N2r_c12": 0.60}}, {"impairment_flag": "HOLD_IMPAIRS"}, {"I": 0.12}),
    ("impaired_small", {"E": {"N2r_c12": 0.70}}, {"impairment_flag": "NO_IMPAIRMENT_FLAG"}, {"I": 0.02}),
    ("neff_opposes", {"neff": {"S2dyn_10r": 0.0050, "S2dyn_c2": 0.0060, "S2dyn_c1": 0.0070, "S2dyn_c12": 0.0080,
                               "S2dyn_c12b": 0.0090}},
     {"neff_tracks": "NEFF_OPPOSES", "primary": "ESCAPE_NOT_MANIPULATED"}, {}),
    ("slope_negative", {"tie": {"S2dyn_c12": -0.02}}, {"slope": "SLOPE_NEGATIVE", "primary": "REMAINDER_REDUCED_BY_HOLD"}, {}),
    ("slope_flat", {"tie": {}}, {"slope": "SLOPE_UNRESOLVED"}, {}),
    ("shadow_broken", {"break": _brk("S2dyn_c2", "shadow_hash_match_natural", "False", k=2)},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("hold_violation", {"break": _brk("S2dyn_c12", "exc_w2", "0.01")}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("hold_idle", {"break": tuple(x for c in ("rows_par", "rows_perp", "rows_w2") for x in _brk("S2dyn_c12", c, "0", k=2))},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("repro_contradicts", {"break": _brk("S2u30r", "hash_match_respdyn", "False", k=2)},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("prefix_contradicts", {"prefix_break": tuple((s, "hash_match_neff_pred", "False") for s in (10, 11, 12))},
     {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("prefix_unrecorded", {"prefix_break": tuple((s, "hash_match_neff_pred", "") for s in (10, 11, 12))},
     {"primary": "REMAINDER_REDUCED_BY_HOLD"}, {"n_valid": 10}),
    ("frz_unrecorded", {"break": _brk("S2dyn_frz", "exc_frz", "")}, {"primary": "INAPPLICABLE"}, {"n_valid": 7}),
    ("paired_matters", {"seed_sd": 1.0}, {"primary": "REMAINDER_REDUCED_BY_HOLD", "neff_tracks": "NEFF_TRACKS"},
     {"dR": 0.07}),
]


def _pick(res, key):
    if key == "n_valid":
        return len(res["valid_seeds"])
    return res["stats"][(key, 1, 0.95)]["mean"]


def s8(V) -> dict:
    """verdict.analyze on 21 synthetic scenarios over seeds 10-19: labels as designed, estimates within
    TOL_EST, the imported t quantile on the published table within 1e-6."""
    failed, per = [], {}
    failed += [f"t({p},{df})" for (p, df), w in T_TABLE.items() if abs(V.t_quantile(p, df) - w) > 1e-6]
    if tuple(V.SEEDS) != SEEDS_REG:
        failed.append(f"SEEDS {V.SEEDS}")
    for name, scen, want_l, want_v in S8_SCENARIOS:
        res = V.analyze(synth(V, scen))
        bad = [f"{k}: got {res['labels'].get(k)} want {w}" for k, w in want_l.items() if res["labels"].get(k) != w]
        for k, w in want_v.items():
            g = _pick(res, k)
            if not (g is not None and math.isfinite(g) and abs(g - w) <= TOL_EST):
                bad.append(f"{k}: got {g} want {w}")
        per[name] = {"labels": res["labels"], "slope_r": res["slope_r"], "slope_p": res["slope_p"], "bad": bad}
        failed += [f"{name}|{b}" for b in bad]
    return {"pass": not failed, "failed_items": failed, "detail": per}


V_DR = '        q[("dR", k)] = q[("R_none", k)] - q[("R_c12", k)]\n'
V_RC12 = '        q[("R_c12", k)] = e("S2dyn_c12", k) - e("S2u30r_c12", k)\n'
V_LEVEL = "MAIN_LEVEL = 0.975                              # dR and R_c12\n"
V_C = '    C_ok = A_ok and st[("dneff", 1, COMP_LEVEL)]["lo"] > 0\n'
V_B = '    B_ok = A_ok and st[("R_none", 1, COMP_LEVEL)]["lo"] > 0\n'
V_SHADOW = '            if not _true(rows[(a, k)].get("shadow_hash_match_natural")):\n'
V_TOL = "TOL_CAP = 1e-4 "
V_BIND = '            if _f(r["rows_par"]) + _f(r["rows_perp"]) + _f(r["rows_w2"]) <= 0:\n'
V_E = '    def e(a, k=1):\n        return np.array([E[(s, a, k)] for s in valid])\n'
V_IMP = '    impaired = bool(A_ok and Ist["sign"] == "+" and Ist["mean"] >= dRm)\n'
V_RHO = '    rho = [spearman([N[(s, a, 1)] for a in LADDER], [E[(s, a, 1)] for a in LADDER]) for s in valid]\n'
V_SLOPE = '    xd, yd = q[("dneff", 1)], q[("dR", 1)]\n'
V_PREFIX = '            if r.get(col, "") not in ("", "None", "True", None, True):\n'
V_REPRO = '                if rows[(a, k)].get(f"hash_match_{src}", "") not in ("", "None", "True", None, True):\n'
V_FIX = '        q[("dR_fix", k)] = (e("S2_10r", k) - e("S2u30r", k)) - (e("S2_10r_c12", k) - e("S2u30r_c12", k))\n'
V_SEEDS = "SEEDS = tuple(range(10, 20))\n"
S8_MUT = [
    ("M8a: dR reversed", [(V_DR, '        q[("dR", k)] = q[("R_c12", k)] - q[("R_none", k)]\n')]),
    ("M8b: R_c12 read off the bias-held arm", [(V_RC12, '        q[("R_c12", k)] = e("S2dyn_c12b", k) - e("S2u30r_c12", k)\n')]),
    ("M8c: primary at 95%", [(V_LEVEL, "MAIN_LEVEL = 0.95\n")]),
    ("M8d: the manipulation check back on the mean climb", [(V_C, '    C_ok = A_ok and st[("dclimb", 1, COMP_LEVEL)]["lo"] > 0\n')]),
    ("M8e: the manipulation check dropped", [(V_C, "    C_ok = A_ok\n")]),
    ("M8f: the reproduction of R dropped", [(V_B, "    B_ok = A_ok\n")]),
    ("M8g: shadows not checked", [(V_SHADOW, "            if False:\n")]),
    ("M8h: hold tolerance 0.1", [(V_TOL, "TOL_CAP = 1e-1 ")]),
    ("M8i: binding not checked", [(V_BIND, "            if False:\n")]),
    ("M8j: pairing broken", [(V_E, '    def e(a, k=1):\n        v = np.array([E[(s, a, k)] for s in valid])\n        return v[::-1] if a == "S2u30r" else v\n')]),
    ("M8k: the impairment flag on any positive I", [(V_IMP, '    impaired = bool(A_ok and Ist["sign"] == "+")\n')]),
    ("M8l: the ladder read on the mean climb", [(V_RHO, '    rho = [spearman([C[(s, a, 1)] for a in LADDER], [E[(s, a, 1)] for a in LADDER]) for s in valid]\n')]),
    ("M8m: the slope on dclimb", [(V_SLOPE, '    xd, yd = q[("dclimb", 1)], q[("dR", 1)]\n')]),
    ("M8n: an unrecorded prefix is invalid", [(V_PREFIX, '            if r.get(col, "") not in ("True", True):\n')]),
    ("M8o: a contradicted prefix is valid", [(V_PREFIX, '            if False:\n')]),
    ("M8p: a contradicted reproduction is valid", [(V_REPRO, '                if False:\n')]),
    ("M8q: dR_fix reversed", [(V_FIX, '        q[("dR_fix", k)] = (e("S2_10r_c12", k) - e("S2u30r_c12", k)) - (e("S2_10r", k) - e("S2u30r", k))\n')]),
    ("M8r: seeds 0-9", [(V_SEEDS, "SEEDS = tuple(range(10))\n")]),
]


# --------------------------------------------------------------------------
# S9: the CLI
# --------------------------------------------------------------------------

E_FLUSH = "    torch.set_flush_denormal(True)\n"
E_SPEC = 'SPEC = "specs/spec_escneff_ee_0917.md"\n'
E_EXP = 'EXPERIMENT = "escneff_ee_0917"\n'
E_SEEDS = "SEEDS = tuple(range(10, 20))\n"


def s9(M) -> dict:
    """main() with 2 epochs and one arm on seed 20 (unregistered, unrecorded) writes provenance naming this experiment, its spec and
    registration field, the escape engine, flush on, and hashes of this runner and the escape runner; the
    registered seeds are 10-19 and the engine's arms and holds are escape's twelve and six."""
    out = SCR / "s9_cli"
    torch.set_flush_denormal(False)
    try:
        M.main(["--seed", "20", "--epochs", "2", "--arms", "N2r_c12", "--out", str(out)])
        flushed = _flush_is_on()
    finally:
        torch.set_flush_denormal(True)
    pv = json.loads((out / "provenance.json").read_text())
    per = {"experiment": pv["experiment"] == "escneff_ee_0917" and pv["runner_experiment"] == "escape_ee_0917",
           "spec": pv["spec"] == "specs/spec_escneff_ee_0917.md" and "prereg_commit" in pv,
           "flush": pv["flush_denormal"] is True and flushed,
           "hashes": {"src/escneff_ee_0917.py", "src/escape_ee_0917.py", "src/swap_ee_0917.py"} <= set(pv["code_sha256"]),
           "unrecorded_seed": pv["prefix_record_tasks_neff_pred"] == 0 and pv["prefix_hash_match_resp_ee"] == 0,
           "constants": M.SEEDS == tuple(range(10, 20)) and len(M.ES.ARMS) == 12 and len(M.ES.HOLDS) == 6}
    failed = [k for k, v in per.items() if not v]
    return {"pass": not failed, "failed_items": failed, "detail": per}


S9_MUT = [
    ("M9a: flush off", [(E_FLUSH, "    torch.set_flush_denormal(False)\n")]),
    ("M9b: another spec", [(E_SPEC, 'SPEC = "specs/spec_escape_ee_0917.md"\n')]),
    ("M9c: the escape experiment's name", [(E_EXP, 'EXPERIMENT = "escape_ee_0917"\n')]),
    ("M9d: seeds 0-9", [(E_SEEDS, "SEEDS = tuple(range(10))\n")]),
]


# --------------------------------------------------------------------------
# S-cost
# --------------------------------------------------------------------------

def s_cost() -> None:
    """PROBE processes at once on seeds 20-23 (outside the registered 10-19, so no registered arm value
    exists before the run), each a full-epoch seed with 3 arms (N2r plain, S2dyn_10r
    and S2dyn_c12 with a shadow).  One full seed = prefix + 6 plain arms + 6 shadow arms.  Gate: all exit 0
    and the memory budget leaves >= PROBE slots."""
    t0 = time.time()
    avail = [int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")][0] / 2 ** 20
    SCR.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(PROBE):
        d = SCR / f"cost_{i}"
        procs.append((d, subprocess.Popen([PY, str(RUNNER), "--seed", str(20 + i), "--arms", "N2r,S2dyn_10r,S2dyn_c12",
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
    "S1_S-arms": ("S1 escape's arms through this runner at the real branch", s1, S1_MUT, s1.__doc__, ESC_RUNNER),
    "S1b_S-prefix-record": ("S1b the prefix is compared with neff_pred's record where it exists", s1b, S1B_MUT,
                            s1b.__doc__, RUNNER),
    "S8_S-verdict": ("S8 the verdict on synthetic seeds 10-19", s8, S8_MUT, s8.__doc__, VERDICT),
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
    RESULTS.update({"run_id": "escneff_ee_0917", "started_at": dt.datetime.now().astimezone().isoformat(),
                    "spec": "specs/spec_escneff_ee_0917.md", "prereg_commit": load(RUNNER).PREREG_COMMIT,
                    "code_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (RUNNER, ESC_RUNNER, VERDICT, Path(__file__))},
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

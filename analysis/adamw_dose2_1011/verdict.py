#!/usr/bin/env python3
"""Verdict for adamw_dose2_1011 (specs/spec_adamw_dose2_1011.md sections 4, 5, 7.3).

    /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/adamw_dose2_1011/verdict.py [--src DIR]

A1 (RL-CIFAR, ELU->ELU, std): the three AdamW arms under results/adamw_dose2_1011/rlcifar/<arm>/ against the committed
cap_cifar_ee_0920 ref (per_task.csv in git; the ref's unit arrays from the cap_cifar_ee_0920 backup), with
cap_cifar_ee_0920's registered statistics imported unmodified (analysis/cap_cifar_ee_0920/stats.py: the floor, the
paired t intervals, the labels, T_half), then RULE_PREDICTED, DOSE, EQUILIBRIUM, WIDTH and DEPTH.  A2 (5+1 CIFAR,
ReLU): results/adamw_dose2_1011/c51/{l2c1e-2,l2c5e-3,_nochange,...} and the committed R, L2 Init and P180 AdamW
records.  analyze_a1() and analyze_a2() are pure (arrays in, results out) so checks.py S-verdict can run them on
synthetic data.  Writes results/adamw_dose2_1011/{verdict.json, verdict.csv, rule.csv, paired.csv, dose.csv,
equilibrium.csv, width.csv, depth.csv, c51_paired.csv, secondary.csv, predictions.csv, summary.md}.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from analysis.cap_cifar_ee_0920 import stats as CS          # noqa: E402  registered, unmodified
from analysis.adamw_dose2_1011 import rule as RULE          # noqa: E402  registered with the spec

RUN_ID = "adamw_dose2_1011"
OUT = REPO / "results" / RUN_ID
CAP = REPO / "results" / "cap_cifar_ee_0920"

# ---------------------------------------------------------------- A1 (spec 2.2, 4, 5.1-5.6)
ARMS1 = ("wd3e-2", "wd1e-1", "wd1e0")
ALL1 = ("ref",) + ARMS1
LAMBDA = {"ref": 0.0, "wd3e-2": 0.03, "wd1e-1": 0.1, "wd1e0": 1.0}
SEEDS = tuple(range(10))
N_TASKS = 50
STEPS = 30000
W0, W1 = 30, 50                     # 0-based task slice of the main window t31-50
LEVEL = 0.975                       # every lambda arm: 2-endpoint Bonferroni (spec 5.1)
CHANCE, HALF = 0.1, 0.5
TOL = 0.1                           # DEPTH (spec 5.6)
# EQUILIBRIUM (spec 14.7): P180's measured ratios 1.61-2.25, widened by 30% each side -> x1.238 .. x2.925
EQ_LO, EQ_HI = 1.61 / 1.3, 2.25 * 1.3
# A2 reading (spec 14.3): rho = (W - R) / (L - R); R-like <= 1/4, L2 Init-like >= 3/4 of the L2 Init benefit D
RHO_LO, RHO_HI = 0.25, 0.75
LIVE = (0, 3)                       # the ref's live tasks t1-t3 (spec 5.5)
D1, D2 = 3072, 100
DOSE_ALPHA = 0.05
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")
THREE = {"RESCUED": "RESCUED", "RESCUED_FUNCTION_ONLY": "PARTIAL", "ALIVE_UNRESOLVED": "PARTIAL",
         "SPLIT": "PARTIAL", "COLLAPSED": "FAIL"}
NONLABELS = ("INAPPLICABLE", "NOT_REPRODUCED", "INCOMPLETE", "CHECK_FAILED", "DIVERGED")
MED_KEYS = ("s1", "s2", "w1", "w2", "rho1", "rho2", "r2", "zbar2")

# ---------------------------------------------------------------- A2 (spec 2.4, 4, 5.7)
ARMS2 = ("l2c1e-2", "l2c5e-3")
LAMBDA2 = {"l2c1e-2": 1e-2, "l2c5e-3": 5e-3}
MATCHED = {"l2c1e-2": "adamw10", "l2c5e-3": "adamw5"}
N_TASKS2 = 30
HARD = list(range(1, N_TASKS2 + 1, 2))
EARLY2, LATE2 = HARD[:5], HARD[10:]
T29 = 29
STEPS2 = 780
SEEDS2 = tuple(range(10))
HOST = REPO / "results" / "cifar5p1_mlp_0920"
P180 = REPO / "results" / "adamw_dose_1010" / "c51"
DIAG2 = ("dead_frac_l2", "zeroout_l2", "mob_l2", "eff_rank_l2", "zbar_l2", "zsd_l2", "w_norm_l1", "w_norm_l2",
         "w_norm_l3", "dead_frac_l1", "eff_rank_l1")
L2T = ("relpos_l2", "p2", "b1_norm", "b2_norm", "b3_norm", "w1_row_med", "w2_row_med", "w3_row_med")
SIGN_ALPHA = 0.05


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def sign_test(d) -> tuple[int, int, float]:
    """Exact two-sided sign test with the zeros dropped: (positives, n, p).  n = 0 gives p = 1."""
    d = np.asarray(d, float)
    nz = d[d != 0]
    n = len(nz)
    pos = int((nz > 0).sum())
    k = min(pos, n - pos)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else 1.0
    return pos, n, p


def holm(ps: list[float], alpha: float = DOSE_ALPHA) -> list[bool]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej = [False] * len(ps)
    m = len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            rej[i] = True
        else:
            break
    return rej


def direction(d, rejected: bool) -> str:
    pos, n, _ = sign_test(d)
    if rejected and n and pos * 2 != n:
        return "UP" if pos * 2 > n else "DOWN"
    return "FLAT"


def order_interval(d) -> tuple[float, float]:
    v = np.sort(np.asarray(d, float))
    return (float(v[1]), float(v[8])) if len(v) == 10 else (float("nan"), float("nan"))


def slope(t, y) -> float:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    tc = t - t.mean()
    return float((tc * (y - y.mean())).sum() / (tc * tc).sum())


def win_mean(x) -> np.ndarray:
    """Per-seed mean over the main window (cap_cifar_ee's mean64)."""
    return CS.mean64(np.asarray(x, np.float64)[:, W0:W1], 1)


def label_interval(lo: float, hi: float, band: float, inside: str, high: str, low: str, closed: bool) -> str:
    if (closed and lo >= -band and hi <= band) or (not closed and lo > -band and hi < band):
        return inside
    if lo > band:
        return high
    if hi < -band:
        return low
    return "UNRESOLVED"


def band_label(lo: float, hi: float, a: float, b: float) -> str:
    """MATCH if [lo, hi] lies in [a, b]; OFF_HIGH if lo > b; OFF_LOW if hi < a; else UNRESOLVED."""
    if lo >= a and hi <= b:
        return "MATCH"
    if lo > b:
        return "OFF_HIGH"
    if hi < a:
        return "OFF_LOW"
    return "UNRESOLVED"


def returns(a: np.ndarray, run: int = 3) -> int:
    """'Return' events (P180): >= run consecutive tasks with F(t) < 1/2, then a task with F(t) >= 1/2."""
    f = (np.asarray(a, float) - CHANCE) / (a[0] - CHANCE)
    n_ev, below = 0, 0
    for t in range(1, len(f)):
        if f[t] < HALF:
            below += 1
        else:
            if below >= run:
                n_ev += 1
            below = 0
    return n_ev


# --------------------------------------------------------------------------
# A1: the analysis (pure)
# --------------------------------------------------------------------------

def analyze_a1(data: dict) -> dict:
    """data = {"arms": {arm: {"A","G","F","Z": (10, 50) float arrays, "lh","oh": (10, 50) str arrays,
    "med": {key: (10, 50)}, "fline2": (10, 50) (lambda arms), "wd_ok": (10, 49) bool (lambda arms)}},
    "s0_ok": bool}.  Task 1 is the common prefix (column 0)."""
    arms = data.get("arms", {})
    res = {"labels": {}, "arms": {}, "rule": {}, "dose": {}, "equilibrium": {}, "width": {}, "depth": {},
           "factors": {}, "per_seed": [], "applicability": {}}
    missing = [a for a in ALL1 if a not in arms]
    bad_shape = [a for a in ALL1 if a in arms and any(np.asarray(arms[a][k]).shape != (10, N_TASKS)
                                                      for k in ("A", "G", "F", "Z"))]
    nonfinite = [a for a in ALL1 if a in arms and a not in bad_shape and
                 not all(np.isfinite(np.asarray(arms[a][k], float)).all() for k in ("A", "G", "F", "Z"))]
    res["applicability"]["missing"] = missing + bad_shape
    res["applicability"]["nonfinite"] = nonfinite

    def all_labels(lab):
        for a in ARMS1:
            res["labels"][a] = lab
        for k in ("rule", "dose"):
            res["labels"][k] = lab
        return res

    if missing or bad_shape:
        return all_labels("INCOMPLETE")
    if nonfinite:
        return all_labels("DIVERGED")
    hashes_ok = all(np.array_equal(np.asarray(arms[a]["lh"]), np.asarray(arms["ref"]["lh"])) and
                    np.array_equal(np.asarray(arms[a]["oh"]), np.asarray(arms["ref"]["oh"])) and
                    np.array_equal(np.asarray(arms[a]["F"]), np.asarray(arms["ref"]["F"])) for a in ARMS1)
    prefix_ok = all(np.array_equal(np.asarray(arms[a][k])[:, 0], np.asarray(arms["ref"][k])[:, 0])
                    for a in ARMS1 for k in ("A", "G", "F"))
    res["applicability"].update(hashes_equal=bool(hashes_ok), prefix_equal=bool(prefix_ok),
                                s0_ok=bool(data.get("s0_ok", False)), env_ok=bool(data.get("env_ok", True)))
    if not (hashes_ok and prefix_ok and data.get("s0_ok", False) and data.get("env_ok", True)):
        return all_labels("CHECK_FAILED")
    x = {a: {k: np.asarray(arms[a][k], np.float64) for k in ("A", "G", "F", "Z")} for a in ALL1}
    upper, fmean, fsd = CS.upper_floor(x["ref"]["F"][:, W0:W1])
    ends, floors = {}, {}
    for a in ALL1:
        ends[a] = {k: win_mean(x[a][f]) - x[a][f][:, 0] for k, f in (("E1", "A"), ("E2", "G"))}
        floors[a] = win_mean(x[a]["A"]) <= upper
    ref_e2 = CS.interval(ends["ref"]["E2"])
    B_ok = int(floors["ref"].sum()) >= 9 and ref_e2["high"] < 0
    C = {a: bool(np.asarray(arms[a]["wd_ok"]).shape == (10, N_TASKS - 1) and np.asarray(arms[a]["wd_ok"]).all())
         for a in ARMS1}
    res["applicability"].update(B_ok=bool(B_ok), ref_floor_count=int(floors["ref"].sum()),
                                ref_E2=ref_e2, C=C, floor_upper=upper.tolist())
    ref = x["ref"]
    early_ok = bool((ref["A"][:, 1] > ref["F"][:, 1]).all())
    early_mid = (ref["A"][:, 1] + ref["F"][:, 1]) / 2
    Zref = win_mean(ref["Z"])
    for a in ARMS1:
        d = {k: ends[a][k] - ends["ref"][k] for k in ("E1", "E2")}
        ci = {k: CS.interval(d[k], LEVEL) for k in ("E1", "E2")}
        ci95 = {k: CS.interval(d[k], 0.95) for k in ("E1", "E2")}
        n = int(floors[a].sum())
        if not C[a]:
            lab = "INAPPLICABLE"
        elif not B_ok:
            lab = "NOT_REPRODUCED"
        else:
            lab = CS.classify(n, ci["E1"]["sign"], ci["E2"]["sign"])
        early_n = int((x[a]["A"][:, 1] < early_mid).sum()) if early_ok else None
        early = "EARLY_FLAG_NOT_ASSESSABLE" if not early_ok else ("IMPAIRED_EARLY" if early_n >= 6 else
                                                                  "NO_EARLY_IMPAIRMENT")
        timing = CS.timing(x[a]["A"], ref["A"])
        rets = [returns(x[a]["A"][s]) for s in range(10)]
        Zw = win_mean(x[a]["Z"])
        if (Zref > 0).all():
            zeta = Zw / Zref
            zi = CS.interval(zeta, 0.95)
            depth = ("DEPTH_ABOVE_LINE" if zi["high"] < TOL else "DEPTH_AT_LINE" if zi["low"] > TOL
                     else "DEPTH_UNRESOLVED")
        else:
            zeta, zi, depth = np.full(10, np.nan), None, "DEPTH_UNRESOLVED"
        res["labels"][a] = lab
        res["labels"][f"{a}_three"] = THREE.get(lab, lab)
        res["labels"][f"{a}_early"] = early
        res["labels"][f"{a}_depth"] = depth
        res["arms"][a] = {"label": lab, "floor_count": n, "floor_state": CS.floor_state(n), "E1": ci["E1"],
                          "E2": ci["E2"], "E1_95": ci95["E1"], "E2_95": ci95["E2"], "early": early,
                          "early_count": early_n, "timing": {k: v for k, v in timing.items() if k != "rows"},
                          "t_half": [r["arm_time"] for r in timing["rows"]], "returns": rets,
                          "online_late": float(win_mean(x[a]["A"]).mean()),
                          "online_late_seeds": win_mean(x[a]["A"]).tolist(), "G_late": float(win_mean(x[a]["G"]).mean()),
                          "Z_late": float(Zw.mean()), "Z_late_seeds": Zw.tolist(), "online_t2": float(x[a]["A"][:, 1].mean())}
        res["depth"][a] = {"label": depth, "zeta": zeta.tolist(), "interval": zi, "Z_ref": Zref.tolist()}
        for s in range(10):
            res["per_seed"].append({"arm": a, "seed": s, "U": float(upper[s]), "A_late": float(win_mean(x[a]["A"])[s]),
                                    "AT_FLOOR": bool(floors[a][s]), "E1": float(ends[a]["E1"][s]),
                                    "E2": float(ends[a]["E2"][s]), "delta_E1": float(d["E1"][s]),
                                    "delta_E2": float(d["E2"][s]), "Z_late": float(Zw[s]),
                                    "zeta": float(zeta[s]), "returns": rets[s]})
    res["arms"]["ref"] = {"online_late": float(win_mean(ref["A"]).mean()), "G_late": float(win_mean(ref["G"]).mean()),
                          "Z_late": float(Zref.mean()), "floor_count": int(floors["ref"].sum()),
                          "online_t1": float(ref["A"][:, 0].mean()), "online_t2": float(ref["A"][:, 1].mean())}
    # ---- RULE_PREDICTED (spec 14.1): functional main judgement, mechanism sub-judgement, registered conjunction
    calls = {r["arm"]: r for r in RULE.table()}
    n_main = n_depth = n_conj = n_label = 0
    rule_rows = []
    for a in ARMS1:
        lab = res["labels"][a]
        call_save = calls[a]["call"] == "SAVES"
        impaired = res["labels"][f"{a}_early"] == "IMPAIRED_EARLY"
        above = res["labels"][f"{a}_depth"] == "DEPTH_ABOVE_LINE"
        saved = lab == "RESCUED" and not impaired                    # spec 14.1-14.2: the functional outcome
        m_main = call_save == saved
        m_depth = call_save == above                                  # the depth side the rule predicted
        m_conj = call_save == (lab == "RESCUED" and above)            # the registration's definition (7525f974)
        n_main += m_main
        n_depth += m_depth
        n_conj += m_conj
        n_label += call_save == (lab == "RESCUED")
        Zl = res["arms"][a]["Z_late"]
        rule_rows.append({"arm": a, "lam": LAMBDA[a], "class": calls[a]["class"], "call": calls[a]["call"],
                          "label": lab, "impaired_early": impaired, "depth": res["labels"][f"{a}_depth"],
                          "outcome": "SAVED" if saved else "NOT_SAVED", "match": bool(m_main),
                          "depth_side_match": bool(m_depth), "registered_conjunction_match": bool(m_conj),
                          "Z_band": "Z<0.1" if Zl < 0.1 else ("0.1<=Z<0.5" if Zl < 0.5 else "Z>=0.5")})

    def count_label(n, prefix):
        return f"{prefix}_HOLDS" if n == 3 else f"{prefix}_PARTIAL" if n == 2 else f"{prefix}_FAILS"
    if any(res["labels"][a] in NONLABELS for a in ARMS1):
        rlab = dlab_ = clab = "RULE_INAPPLICABLE"
        stands = False
    else:
        rlab = count_label(n_main, "RULE")
        dlab_ = count_label(n_depth, "RULE_DEPTH")
        clab = count_label(n_conj, "RULE")
        stands = rlab == "RULE_HOLDS" and all(r["depth"] == "DEPTH_ABOVE_LINE" for r in rule_rows
                                              if r["outcome"] == "SAVED")
    res["labels"]["rule"] = rlab
    res["labels"]["rule_depth"] = dlab_
    res["labels"]["rule_registered"] = clab
    res["labels"]["rule_stands"] = bool(stands)
    res["rule"] = {"label": rlab, "n_match": int(n_main), "depth_label": dlab_, "n_depth_match": int(n_depth),
                   "registered_label": clab, "n_registered_match": int(n_conj),
                   "n_label_only_match": int(n_label), "rule_stands": bool(stands), "rows": rule_rows}
    # ---- DOSE (spec 5.3)
    levels = {a: win_mean(x[a]["A"]) for a in ALL1}
    steps = [("ref", "wd3e-2"), ("wd3e-2", "wd1e-1"), ("wd1e-1", "wd1e0")]
    dd = [levels[b] - levels[a] for a, b in steps]
    ps = [sign_test(v)[2] for v in dd]
    rej = holm(ps)
    dirs = [direction(v, r) for v, r in zip(dd, rej)]
    if any(res["labels"][a] in NONLABELS for a in ARMS1):
        dlab = "DOSE_INAPPLICABLE"
    elif "DOWN" not in dirs and "UP" in dirs:
        dlab = "DOSE_MONOTONE"
    elif any(dirs[j] == "DOWN" and "UP" in dirs[:j] for j in range(len(dirs))):
        dlab = "DOSE_PEAKED"
    elif all(v == "FLAT" for v in dirs):
        dlab = "DOSE_FLAT"
    else:
        dlab = "DOSE_OTHER"
    res["labels"]["dose"] = dlab
    res["dose"] = {"label": dlab, "steps": [{"step": f"{a}->{b}", "pos": sign_test(v)[0], "n": sign_test(v)[1],
                                             "p": p, "holm_rejected": bool(r), "dir": dr, "median": float(np.median(v))}
                                            for (a, b), v, p, r, dr in zip(steps, dd, ps, rej, dirs)]}
    # ---- EQUILIBRIUM, WIDTH, factors (spec 5.4, 5.5, 5.2 report)
    med = {a: {k: np.asarray(arms[a]["med"][k], np.float64) for k in MED_KEYS} for a in ALL1}
    g_ref = {l: np.array([slope(range(LIVE[0] + 1, LIVE[1] + 1), med["ref"][f"s{l}"][s, LIVE[0]:LIVE[1]])
                          for s in range(10)]) for l in (1, 2)}
    res["width_ref_live_slope"] = {f"s{l}": g_ref[l].tolist() for l in (1, 2)}
    # spec 14.7: the WIDTH tolerance per layer = the seed CV of the ref's live slope (record: 0.072, 0.113)
    wtol = {l: float(np.std(g_ref[l], ddof=1) / np.mean(g_ref[l])) for l in (1, 2)}
    res["width_tolerance"] = {f"l{l}": wtol[l] for l in (1, 2)}
    for a in ARMS1:
        lam = LAMBDA[a]
        eq = {}
        for l, d in ((1, D1), (2, D2)):
            ratio = win_mean(med[a][f"w{l}"]) / RULE.w_star(lam, d)
            ci = CS.interval(np.log2(ratio), 0.95)
            eq[f"l{l}"] = {"label": band_label(ci["low"], ci["high"], math.log2(EQ_LO), math.log2(EQ_HI)),
                           "ratio_geo": float(2 ** ci["mean"]), "log2_lo": ci["low"], "log2_hi": ci["high"],
                           "w_star": RULE.w_star(lam, d), "measured": float(win_mean(med[a][f"w{l}"]).mean())}
        eq["label"] = "EQUILIBRIUM_MATCH" if all(eq[f"l{l}"]["label"] == "MATCH" for l in (1, 2)) else "EQUILIBRIUM_OFF"
        res["equilibrium"][a] = eq
        res["labels"][f"{a}_equilibrium"] = eq["label"]
        state = res["arms"][a]["floor_state"]
        wd_ = {}
        if res["labels"][a] in NONLABELS:
            wlab = "WIDTH_INAPPLICABLE"
        elif state == "FLOORED":
            wlab = "WIDTH_FROZEN"
        elif state == "SPLIT":
            wlab = "WIDTH_SPLIT"
        else:
            for l in (1, 2):
                with np.errstate(divide="ignore", invalid="ignore"):
                    kap = np.array([slope(range(W0 + 1, W1 + 1), med[a][f"s{l}"][s, W0:W1]) / g_ref[l][s]
                                    for s in range(10)])
                if not np.isfinite(kap).all():                 # a reference slope of 0: kappa undefined
                    wd_[f"l{l}"] = {"label": "UNRESOLVED", "kappa_mean": float("nan"), "lo": float("nan"),
                                    "hi": float("nan"), "s_late": float(win_mean(med[a][f"s{l}"]).mean()),
                                    "note": "reference live slope 0 in some seed"}
                    continue
                ci = CS.interval(kap, 0.95)
                wd_[f"l{l}"] = {"label": label_interval(ci["low"], ci["high"], wtol[l], "STOPPED", "GROWING",
                                                        "SHRINKING", False), "tol": wtol[l],
                                "kappa_mean": ci["mean"], "lo": ci["low"],
                                "hi": ci["high"], "s_late": float(win_mean(med[a][f"s{l}"]).mean())}
            wlab = "WIDTH_STOPPED" if all(wd_[f"l{l}"]["label"] == "STOPPED" for l in (1, 2)) else "WIDTH_NOT_STOPPED"
        wd_["label"] = wlab
        res["width"][a] = wd_
        res["labels"][f"{a}_width"] = wlab
        # rule factors (report only)
        pr = RULE.predict(lam)
        f_meas = np.where(med[a]["s1"] > 0, med[a]["rho2"] / med[a]["s1"], np.nan)
        fl = np.asarray(arms[a].get("fline2", np.full((10, N_TASKS), np.nan)), np.float64)
        meas = {"kappa1": float(win_mean(med[a]["w1"]).mean() / RULE.w_star(lam, D1)),
                "kappa2": float(win_mean(med[a]["w2"]).mean() / RULE.w_star(lam, D2)),
                "rho1": float(win_mean(med[a]["rho1"]).mean()), "f": float(np.nanmean(f_meas[:, W0:W1])),
                "r": float(win_mean(med[a]["r2"]).mean()), "s2": float(win_mean(med[a]["s2"]).mean()),
                "zbar2": float(win_mean(med[a]["zbar2"]).mean()),
                "Z": float(win_mean(x[a]["Z"]).mean()), "F": float(np.nanmean(fl[:, W0:W1]))}
        meas["m"] = meas["r"] * meas["s2"]
        bands = {"kappa1": RULE.KAPPA, "kappa2": RULE.KAPPA, "rho1": RULE.RHO1, "f": RULE.F_BAND,
                 "r": (RULE.R_BAND[1], RULE.R_BAND[0]), "s2": (pr["shallow"]["s2"], pr["deep"]["s2"]),
                 "m": (pr["deep"]["m"], pr["shallow"]["m"]), "Z": (pr["shallow"]["Z"], pr["deep"]["Z"]),
                 "F": (pr["shallow"]["F"], pr["deep"]["F"])}
        res["factors"][a] = {k: {"measured": meas[k], "band_lo": float(min(bands[k])), "band_hi": float(max(bands[k])),
                                 "inside": bool(min(bands[k]) <= meas[k] <= max(bands[k]))}
                             for k in bands} | {"centre": {"s2": pr["centre"]["s2"], "m": pr["centre"]["m"],
                                                           "Z": pr["centre"]["Z"]},
                                                "zbar2_measured": meas["zbar2"]}
    return res


# --------------------------------------------------------------------------
# A2: the analysis (pure)
# --------------------------------------------------------------------------

def per_seed_window(pt: pd.DataFrame, col: str, tasks) -> dict[int, float]:
    out = {}
    for s, g in pt.groupby("seed"):
        g = g.set_index("task")
        if all(t in g.index for t in tasks):
            out[int(s)] = float(np.mean([float(g.loc[t, col]) for t in tasks]))
    return out


def ci95(x) -> dict:
    x = np.asarray(x, np.float64)
    m, sd = float(x.mean()), float(x.std(ddof=1))
    w = CS.t_quantile(0.975, len(x) - 1) * sd / math.sqrt(len(x))
    return {"mean": m, "lo": m - w, "hi": m + w}


def analyze_a2(data: dict) -> dict:
    """data: {"arms": {arm: {"per_task", "fresh_self", "l2", "prov"}}, "R": {"per_task", "fresh_control"},
    "L": {"per_task"}, "adamw": {name: per_task}, "nochange": {"prov"}, "reuse": {"a","b","c","coupled_host": bool}}."""
    res = {"labels": {}, "rows": []}
    arms = data.get("arms", {})
    incomplete = [a for a in ARMS2 if a not in arms]
    for a in ARMS2:
        if a in incomplete:
            continue
        pt = arms[a]["per_task"]
        ok = len(pt) == len(SEEDS2) * N_TASKS2 and set(pt["seed"].astype(int)) == set(SEEDS2) \
            and "online_acc" in pt and np.isfinite(pt["online_acc"].astype(float)).all()
        fs = arms[a].get("fresh_self")
        ok = ok and fs is not None and len(fs) == len(SEEDS2) and np.isfinite(fs["fresh_self_online_acc"]).all()
        ok = ok and not arms[a]["prov"].get("divergences")
        if not ok:
            incomplete.append(a)
    reuse = data.get("reuse", {})
    reuse_ok = all(reuse.get(k) is True for k in ("a", "b", "c", "coupled_host"))
    shas = {a: arms[a]["prov"].get("fresh_batches_sha256") for a in ARMS2 if a in arms}
    shas["nochange"] = data.get("nochange", {}).get("prov", {}).get("fresh_batches_sha256")
    sha_ok = len(set(shas.values())) == 1 and None not in shas.values()
    res["applicability"] = {"incomplete": incomplete, "reuse": reuse, "reuse_ok": reuse_ok,
                            "fresh_batches_sha256": shas, "fresh_batches_same": sha_ok}
    if incomplete or not (reuse_ok and sha_ok):
        lab = "INCOMPLETE" if incomplete else "CHECK_FAILED"
        for a in ARMS2:
            res["labels"][a] = lab
            res["labels"][f"{a}_reading"] = lab
        res["labels"]["box"] = lab
        return res
    fg = data["R"]["fresh_control"]["fresh_gap"].astype(float).to_numpy()
    B = ci95(fg)
    B_ok = len(fg) == len(SEEDS2) and B["lo"] > 0
    res["applicability"]["B_R_fresh_gap"] = B
    res["applicability"]["B_ok"] = bool(B_ok)
    C = {}
    for a in ARMS2:
        l2 = arms[a]["l2"]
        C[a] = bool(len(l2) == len(SEEDS2) * N_TASKS2 and all(
            (l2[f"l2_steps_{nm}"].astype(int) == l2["steps"].astype(int)).all() for nm in TENSORS)
                    and (l2["steps"].astype(int) == STEPS2).all())
    res["applicability"]["C"] = C
    Rl = per_seed_window(data["R"]["per_task"], "online_acc", LATE2)
    Ll = per_seed_window(data["L"]["per_task"], "online_acc", LATE2)
    Re = per_seed_window(data["R"]["per_task"], "online_acc", EARLY2)
    res["R_late"] = [Rl[s] for s in SEEDS2]
    res["L_late"] = [Ll[s] for s in SEEDS2]
    a_d, c_d, l_d, info = {}, {}, {}, {}
    for a in ARMS2:
        Wl = per_seed_window(arms[a]["per_task"], "online_acc", LATE2)
        We = per_seed_window(arms[a]["per_task"], "online_acc", EARLY2)
        a_d[a] = np.array([Wl[s] - Rl[s] for s in SEEDS2])
        c_d[a] = np.array([Wl[s] - (Rl[s] + Ll[s]) / 2.0 for s in SEEDS2])
        l_d[a] = np.array([Wl[s] - Ll[s] for s in SEEDS2])
        e_d = np.array([We[s] - Re[s] for s in SEEDS2])
        imp = ci95(e_d)
        info[a] = {"W_late": [Wl[s] for s in SEEDS2], "W_late_mean": float(np.mean([Wl[s] for s in SEEDS2])),
                   "early_diff": e_d.tolist(), "W_early_mean": float(np.mean([We[s] for s in SEEDS2])),
                   "impaired": {**imp, "flag": bool(imp["hi"] < 0)},
                   "rho": [float(a_d[a][i] / (Ll[s] - Rl[s])) for i, s in enumerate(SEEDS2)]}
    tests = {}
    for fam, dd in (("a", a_d), ("c", c_d)):
        ps = [sign_test(dd[a])[2] for a in ARMS2]
        rej = holm(ps, SIGN_ALPHA)
        for a, p, r in zip(ARMS2, ps, rej):
            pos, n, _ = sign_test(dd[a])
            lo, hi = order_interval(dd[a])
            tests[(fam, a)] = {"pos": pos, "n": n, "p": p, "holm_rejected": bool(r), "dir": direction(dd[a], r),
                               "median": float(np.median(dd[a])), "ord_lo": lo, "ord_hi": hi, "mean": float(np.mean(dd[a]))}
    for a in ARMS2:
        pos, n, p = sign_test(l_d[a])
        lo, hi = order_interval(l_d[a])
        below_L = bool(p < SIGN_ALPHA and pos * 2 < n)
        tests[("L", a)] = {"pos": pos, "n": n, "p": p, "dir": "DOWN" if below_L else ("UP" if p < SIGN_ALPHA else "FLAT"),
                           "median": float(np.median(l_d[a])), "ord_lo": lo, "ord_hi": hi, "mean": float(np.mean(l_d[a]))}
        ta, tc = tests[("a", a)], tests[("c", a)]
        if not C[a]:
            lab = "INAPPLICABLE"
        elif not B_ok:
            lab = "NOT_REPRODUCED"
        elif tc["dir"] == "UP":
            lab = "RESCUED"
        elif ta["dir"] == "UP":
            lab = "PARTIAL"
        else:
            lab = "NO_RESCUE"
        # spec 14.3: the reading from the share of the L2 Init benefit recovered, rho_s = (W_s - R_s) / (L_s - R_s),
        # with its paired 95% t interval: L2 Init-like if the interval is >= 3/4, R-like if <= 1/4, between if inside
        # (1/4, 3/4), unresolved if it crosses a boundary
        rho = (np.array(info[a]["W_late"]) - np.array(res["R_late"])) / (np.array(res["L_late"]) - np.array(res["R_late"]))
        rci = ci95(rho)
        if lab in ("INAPPLICABLE", "NOT_REPRODUCED"):
            reading = lab
        elif rci["lo"] >= RHO_HI:
            reading = "NORM_MATTERS"
        elif rci["hi"] <= RHO_LO:
            reading = "TARGET_MATTERS"
        elif rci["lo"] > RHO_LO and rci["hi"] < RHO_HI:
            reading = "BOTH"
        else:
            reading = "READING_UNRESOLVED"
        # the registration's reading (7525f974), kept to score the predictions made under it
        if lab in ("INAPPLICABLE", "NOT_REPRODUCED"):
            reading_reg = lab
        elif lab == "RESCUED" and not below_L:
            reading_reg = "NORM_MATTERS"
        elif lab == "NO_RESCUE":
            reading_reg = "TARGET_MATTERS"
        else:
            reading_reg = "BOTH"
        info[a]["rho_interval"] = rci
        res["labels"][f"{a}_reading_registered"] = reading_reg
        res["labels"][a] = lab
        res["labels"][f"{a}_worse"] = bool(lab == "NO_RESCUE" and ta["dir"] == "DOWN")
        res["labels"][f"{a}_impaired"] = info[a]["impaired"]["flag"]
        res["labels"][f"{a}_reading"] = reading
        for fam, dd in (("a", a_d), ("c", c_d), ("L", l_d)):
            res["rows"].append({"arm": a, "lam": LAMBDA2[a], "family": fam, **tests[(fam, a)],
                                "d": dd[a].tolist(), "label": lab})
    labs = [res["labels"][a] for a in ARMS2]
    if any(x in ("INAPPLICABLE", "NOT_REPRODUCED") for x in labs):
        res["labels"]["box"] = "NOT_REPRODUCED" if "NOT_REPRODUCED" in labs else "INAPPLICABLE"
    elif "RESCUED" in labs:
        res["labels"]["box"] = "L2C_RESCUES_RELU"
    elif "PARTIAL" in labs:
        res["labels"]["box"] = "L2C_PARTIAL_RELU"
    else:
        res["labels"]["box"] = "L2C_NO_RESCUE_RELU"
    res["info"] = info
    res["tests"] = {f"{f}|{a}": v for (f, a), v in tests.items()}
    # report only
    sec = []
    Rpt = data["R"]["per_task"]
    Rl2 = data.get("nochange", {}).get("l2")
    for a in ARMS2:
        for col in DIAG2:
            if col not in Rpt or col not in arms[a]["per_task"]:
                continue
            rv = per_seed_window(Rpt, col, LATE2)
            wv = per_seed_window(arms[a]["per_task"], col, LATE2)
            d = np.array([wv[s] - rv[s] for s in SEEDS2])
            pos, n, p = sign_test(d)
            sec.append({"arm": a, "quantity": col, "W_median": float(np.median([wv[s] for s in SEEDS2])),
                        "R_median": float(np.median([rv[s] for s in SEEDS2])), "diff_median": float(np.median(d)),
                        "pos": pos, "n": n, "p": p})
        for col in L2T:
            wv = per_seed_window(arms[a]["l2"], col, LATE2)
            row = {"arm": a, "quantity": col, "W_median": float(np.median([wv[s] for s in SEEDS2]))}
            if Rl2 is not None:
                rv = per_seed_window(Rl2, col, LATE2)
                d = np.array([wv[s] - rv[s] for s in SEEDS2])
                pos, n, p = sign_test(d)
                row |= {"R_median": float(np.median([rv[s] for s in SEEDS2])), "diff_median": float(np.median(d)),
                        "pos": pos, "n": n, "p": p}
            sec.append(row)
        aw = data.get("adamw", {}).get(MATCHED[a])
        if aw is not None:
            av = per_seed_window(aw, "online_acc", LATE2)
            Wl = per_seed_window(arms[a]["per_task"], "online_acc", LATE2)
            d = np.array([Wl[s] - av[s] for s in SEEDS2])
            pos, n, p = sign_test(d)
            info[a]["vs_matched_adamw"] = {"adamw": MATCHED[a], "adamw_late_mean": float(np.mean([av[s] for s in SEEDS2])),
                                           "median": float(np.median(d)), "pos": pos, "n": n, "p": p}
        fs = arms[a]["fresh_self"].set_index("seed")
        Rf = data["R"]["fresh_control"].set_index("seed")
        W29 = per_seed_window(arms[a]["per_task"], "online_acc", [T29])
        info[a]["fresh_gap_self_median"] = float(np.median([float(fs.loc[s, "fresh_self_online_acc"]) - W29[s]
                                                             for s in SEEDS2]))
        info[a]["fresh_gap_common_median"] = float(np.median([float(Rf.loc[s, "fresh_online_acc"]) - W29[s]
                                                               for s in SEEDS2]))
    res["secondary"] = sec
    res["R_fresh_gap_median"] = float(np.median(fg))
    return res


# --------------------------------------------------------------------------
# spec 7.3: scoring (Claude's column; Issa's and Fable's were left blank at registration)
# --------------------------------------------------------------------------

OTHER = "OTHER"
# Claude (spec 7, registered 7525f974; A1-1 and A2-3 were made under the registration's definitions and are scored
# against them: rule_registered, <arm>_reading_registered; spec 14.1/14.3)
CLAUDE_P = {
    "A1-1": {"RULE_HOLDS": .55, "RULE_PARTIAL": .38, "RULE_FAILS": .05, OTHER: .02},
    "A1-2": {"RESCUED": .28, "SPLIT": .28, "COLLAPSED": .22, "RESCUED_FUNCTION_ONLY": .12, "ALIVE_UNRESOLVED": .05,
             OTHER: .05},
    "A1-3": {"RESCUED": .55, "SPLIT": .18, "RESCUED_FUNCTION_ONLY": .12, "COLLAPSED": .08, "ALIVE_UNRESOLVED": .03,
             OTHER: .04},
    "A1-4": {"RESCUED": .85, "SPLIT": .04, "COLLAPSED": .03, "RESCUED_FUNCTION_ONLY": .03, "ALIVE_UNRESOLVED": .02,
             OTHER: .03},
    "A1-5": {"DOSE_MONOTONE": .75, "DOSE_PEAKED": .12, "DOSE_OTHER": .11, "DOSE_FLAT": .02},
    "A1-6": {"EQUILIBRIUM_MATCH": .40, "EQUILIBRIUM_OFF": .60},
    "A1-7": {"EQUILIBRIUM_MATCH": .35, "EQUILIBRIUM_OFF": .65},
    "A1-8": {"EQUILIBRIUM_MATCH": .30, "EQUILIBRIUM_OFF": .70},
    "A1-9": {"WIDTH_STOPPED": .80, "WIDTH_NOT_STOPPED": .10, "WIDTH_FROZEN|WIDTH_SPLIT": .10},
    "A1-10": {"WIDTH_STOPPED": .55, "WIDTH_NOT_STOPPED": .15, "WIDTH_FROZEN|WIDTH_SPLIT": .30},
    "A1-11.wd1e0": {"DEPTH_ABOVE_LINE": .85, "DEPTH_AT_LINE": .1125, "DEPTH_UNRESOLVED": .0375},
    "A1-11.wd1e-1": {"DEPTH_ABOVE_LINE": .25, "DEPTH_AT_LINE": .5625, "DEPTH_UNRESOLVED": .1875},
    "A1-11.wd3e-2": {"DEPTH_ABOVE_LINE": .05, "DEPTH_AT_LINE": .7125, "DEPTH_UNRESOLVED": .2375},
    "A2-1": {"NO_RESCUE": .43, "PARTIAL": .33, "RESCUED": .22, OTHER: .02},
    "A2-2": {"NO_RESCUE": .42, "PARTIAL": .36, "RESCUED": .20, OTHER: .02},
    "A2-3": {"TARGET_MATTERS": .41, "BOTH": .37, "NORM_MATTERS": .20, OTHER: .02},
    "A2-4": {"L2C_NO_RESCUE_RELU": .38, "L2C_PARTIAL_RELU": .33, "L2C_RESCUES_RELU": .27, OTHER: .02},
    # spec 14.9: registered after Codex's audit and before the run, for the definitions of spec 14.1 and 14.3
    "A1-1b": {"RULE_HOLDS": .27, "RULE_PARTIAL": .45, "RULE_FAILS": .26, OTHER: .02},
    "A1-1c": {"RULE_DEPTH_HOLDS": .58, "RULE_DEPTH_PARTIAL": .35, "RULE_DEPTH_FAILS": .05, OTHER: .02},
    "A2-3b": {"TARGET_MATTERS": .38, "BOTH": .25, "READING_UNRESOLVED": .25, "NORM_MATTERS": .12},
}
EQ_LAYER_P = {"A1-6": {"MATCH": .55, "OFF_HIGH": .35, "OFF_LOW": .03, "UNRESOLVED": .07},
              "A1-7": {"MATCH": .50, "OFF_HIGH": .35, "OFF_LOW": .05, "UNRESOLVED": .10},
              "A1-8": {"MATCH": .45, "OFF_HIGH": .20, "OFF_LOW": .20, "UNRESOLVED": .15}}
BINARY_P = {"S1": .99, "S2": .85, "S3": .97, "S4": .80, "S5": .60, "S6": .60, "S7": .55, "S8": .65, "S9": .70,
            "S10": .75, "S11": .45, "S12": .30, "C5": .30, "C6": .55, "C7": .75, "C8": .55, "C9": .60, "C10": .55,
            "C11": .85}
# Fable (spec 7, written 10/11 15:56 after the checks; reference predictions; the unnamed remainder is one bucket)
FABLE_P = {
    "A1-1": {"RULE_HOLDS": .50, "RULE_PARTIAL": .40, "RULE_FAILS": .10},
    "A1-2": {"RESCUED": .30, "SPLIT": .35, "COLLAPSED": .35},
    "A1-3": {"RESCUED": .45, "SPLIT": .30, "COLLAPSED": .25},
    "A1-4": {"RESCUED": .75, OTHER: .25},
    "A2-1": {"NO_RESCUE": .40, "PARTIAL": .35, "RESCUED": .25},
    "A2-3": {"NORM_MATTERS": .35, "BOTH": .35, "TARGET_MATTERS": .30},
}
FABLE_BIN = {"A1-11.wd1e0": .85, "A1-11.wd1e-1": .25, "A1-11.wd3e-2": .10, "IMPAIRED_EARLY.wd1e0": .15}
# Codex (specs/review_codex_adamw_dose2_1011.md, fixed blind at 10/11 16:07:48 before reading spec 7)
CODEX_P = {
    "A1-1": {"RULE_HOLDS": .38, "RULE_PARTIAL": .47, "RULE_FAILS": .15},
    "A1-2": {"RESCUED": .15, "SPLIT": .45, "COLLAPSED": .40},
    "A1-3": {"RESCUED": .55, "SPLIT": .35, "COLLAPSED": .10},
    "A1-4": {"RESCUED": .65, "SPLIT": .10, "COLLAPSED": .25},
    "A2-1": {"RESCUED": .45, "PARTIAL": .45, "NO_RESCUE": .10},
    "A2-2": {"RESCUED": .28, "PARTIAL": .52, "NO_RESCUE": .20},
    "A2-3": {"NORM_MATTERS": .28, "TARGET_MATTERS": .07, "BOTH": .65},
}
CODEX_BIN = {"A1-11.wd3e-2": .20, "A1-11.wd1e-1": .58, "A1-11.wd1e0": .97}
UNSCORED = ("WIDTH_INAPPLICABLE", "DOSE_INAPPLICABLE")


def _bucket(label: str, table: dict) -> str:
    for k in table:
        if label in k.split("|"):
            return k
    return OTHER


def score_item(table: dict, realized: str) -> dict:
    b = _bucket(realized, table)
    p = table.get(b, 0.0)
    brier = sum((q - (1.0 if k == b else 0.0)) ** 2 for k, q in table.items()) + (0.0 if b in table else 1.0)
    mode = max(table, key=table.get)
    return {"realized": realized, "bucket": b, "p": p, "mode": mode, "hit": mode == b, "brier": brier}


def _realized(r1: dict, r2: dict) -> dict:
    lab = {**r1.get("labels", {}), **{f"c51_{k}": v for k, v in r2.get("labels", {}).items()}}
    return {"A1-1": lab.get("rule_registered"), "A1-1b": lab.get("rule"), "A1-1c": lab.get("rule_depth"),
            "A1-2": lab.get("wd3e-2"), "A1-3": lab.get("wd1e-1"), "A1-4": lab.get("wd1e0"), "A1-5": lab.get("dose"),
            "A1-6": lab.get("wd1e0_equilibrium"), "A1-7": lab.get("wd1e-1_equilibrium"),
            "A1-8": lab.get("wd3e-2_equilibrium"), "A1-9": lab.get("wd1e0_width"), "A1-10": lab.get("wd1e-1_width"),
            "A1-11.wd1e0": lab.get("wd1e0_depth"), "A1-11.wd1e-1": lab.get("wd1e-1_depth"),
            "A1-11.wd3e-2": lab.get("wd3e-2_depth"),
            "A2-1": lab.get("c51_l2c1e-2"), "A2-2": lab.get("c51_l2c5e-3"),
            "A2-3": lab.get("c51_l2c1e-2_reading_registered"), "A2-3b": lab.get("c51_l2c1e-2_reading"),
            "A2-4": lab.get("c51_box")}


def _score_table(tables: dict, real: dict) -> dict:
    out = {}
    for k, table in tables.items():
        v = real.get(k)
        if v is None:
            out[k] = {"unscored": "label missing"}
        elif v in UNSCORED:
            out[k] = {"unscored": f"{v} (spec 14.8)"}
        else:
            out[k] = score_item(table, v)
    return out


def score_predictions(r1: dict, r2: dict, s0_ok: bool | None) -> dict:
    real = _realized(r1, r2)
    lab = {**r1.get("labels", {}), **{f"c51_{k}": v for k, v in r2.get("labels", {}).items()}}
    items = _score_table(CLAUDE_P, real)
    for k, arm in (("A1-6", "wd1e0"), ("A1-7", "wd1e-1"), ("A1-8", "wd3e-2")):
        eq = r1.get("equilibrium", {}).get(arm)
        if eq:
            for l in ("l1", "l2"):
                items[f"{k}.{l}"] = score_item(EQ_LAYER_P[k], eq[l]["label"])
    arms = r1.get("arms", {})
    fac = r1.get("factors", {})
    rule = r1.get("rule", {})
    a2i = r2.get("info", {})
    sec2 = {(r["arm"], r["quantity"]): r for r in r2.get("secondary", [])}
    tests2 = r2.get("tests", {})

    def sig(arm, q, sign):
        r = sec2.get((arm, q))
        if r is None or "p" not in r:
            return None
        return bool(r["p"] < 0.05 and ((r["pos"] * 2 > r["n"]) if sign > 0 else (r["pos"] * 2 < r["n"])))

    def vs(arm, sign):
        v = a2i.get(arm, {}).get("vs_matched_adamw")
        if v is None:
            return None
        return bool(v["p"] < 0.05 and ((v["pos"] * 2 > v["n"]) if sign > 0 else (v["pos"] * 2 < v["n"])))

    def below_L(arm):
        t = tests2.get(f"L|{arm}")
        return None if t is None else bool(t["p"] < 0.05 and t["pos"] * 2 < t["n"])

    outc = {
        "S1": r1.get("applicability", {}).get("B_ok"),
        "S2": s0_ok,
        "S3": None if not r1.get("applicability", {}).get("C") else all(r1["applicability"]["C"].values()),
        "S4": None if not arms or any(a not in arms for a in ARMS1) else
        all(arms[a].get("early") != "IMPAIRED_EARLY" for a in ARMS1),
        "S5": None if "wd1e0" not in arms else arms["wd1e0"]["online_late"] >= 0.70,
        "S6": None if "wd1e-1" not in arms else 0.1 <= arms["wd1e-1"]["Z_late"] <= 0.9,
        "S7": None if "wd3e-2" not in arms else any(x > 0 for x in arms["wd3e-2"]["returns"]),
        "S8": None if "wd1e0" not in fac else fac["wd1e0"]["s2"]["inside"],
        "S9": None if "wd1e-1" not in fac else fac["wd1e-1"]["s2"]["inside"],
        "S10": None if "wd1e0" not in fac else fac["wd1e0"]["r"]["measured"] > -1.85,
        "S11": None if "wd1e-1" not in fac else fac["wd1e-1"]["kappa1"]["inside"],
        "S12": None if not rule else rule.get("n_label_only_match") == 3,
        "C5": r2.get("labels", {}).get("l2c1e-2_worse"),
        "C6": r2.get("labels", {}).get("l2c1e-2_impaired"),
        "C7": sig("l2c1e-2", "dead_frac_l2", -1),
        "C8": sig("l2c1e-2", "eff_rank_l2", -1),
        "C9": vs("l2c1e-2", +1),
        "C10": vs("l2c5e-3", +1),
        "C11": below_L("l2c1e-2"),
    }
    binary = {}
    for k, p in BINARY_P.items():
        o = outc.get(k)
        binary[k] = {"p": p, "unscored": "not evaluable"} if o is None else {"p": p, "outcome": bool(o),
                                                                              "brier": (p - bool(o)) ** 2}

    def bin_items(table: dict) -> dict:
        out = {}
        for k, p in table.items():
            if k.startswith("A1-11."):
                v = real.get(k)
                o = None if v is None else v == "DEPTH_ABOVE_LINE"
            else:
                v = lab.get(k.split(".")[1] + "_early")
                o = None if v is None or v == "EARLY_FLAG_NOT_ASSESSABLE" else v == "IMPAIRED_EARLY"
            out[k] = {"p": p, "unscored": "not evaluable"} if o is None else {"p": p, "outcome": bool(o),
                                                                              "brier": (p - bool(o)) ** 2}
        return out
    return {"labels": items, "binary": binary,
            "fable": {"labels": _score_table(FABLE_P, real), "binary": bin_items(FABLE_BIN)},
            "codex": {"labels": _score_table(CODEX_P, real), "binary": bin_items(CODEX_BIN)}}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _manifest_resolve(root: Path, rel: Path) -> Path:
    """A file of this run, locally or (after the clean-up) through results/adamw_dose2_1011/backup_manifest.json."""
    p = root / rel
    if p.exists():
        return p
    mf = OUT / "backup_manifest.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        want = str((root / rel).resolve().relative_to(REPO))
        hit = [x for x in m["files"] if x["relative"] == want]
        if len(hit) == 1 and Path(hit[0]["backup"]).exists():
            return Path(hit[0]["backup"])
    raise FileNotFoundError(p)


def _cap_units(arm: str, t: int) -> dict:
    from analysis.cifar_ledger_0920.replay import output_artifact
    rel = Path("prefix/units.npz") if t == 1 else Path(f"arms/{arm}/units_t{t:02d}.npz")
    with np.load(output_artifact(CAP, rel)) as f:
        return {k: f[k] for k in f.files}


def _medians(u: dict) -> dict:
    s1 = np.median(u["zsd_l1"], axis=1)
    return {"s1": s1, "s2": np.median(u["zsd_l2"], axis=1), "w1": np.median(u["Wnorm_l1"], axis=1),
            "w2": np.median(u["Wnorm_l2"], axis=1), "rho1": np.median(u["zsd_l1"] / u["Wnorm_l1"], axis=1),
            "rho2": np.median(u["zsd_l2"] / u["Wnorm_l2"], axis=1), "r2": np.median(u["zmean_over_sd_l2"], axis=1),
            "zbar2": np.median(u["zmean_l2"], axis=1)}


def load_a1(src: Path) -> dict:
    cap_pt = pd.read_csv(CAP / "per_task.csv")
    ref = cap_pt[cap_pt.arm == "ref"].sort_values(["seed", "task"])
    t1 = ref[ref.task == 1].set_index("seed")

    def grid(df, col, dtype=float):
        g = np.full((10, N_TASKS), None, dtype=object) if dtype is str else np.full((10, N_TASKS), np.nan)
        for _, r in df.iterrows():
            g[int(r.seed), int(r.task) - 1] = r[col] if dtype is str else float(r[col])
        return g

    data = {"arms": {}}
    meds_ref = {k: np.full((10, N_TASKS), np.nan) for k in MED_KEYS}
    for t in range(1, N_TASKS + 1):
        m = _medians(_cap_units("ref", t))
        for k in MED_KEYS:
            meds_ref[k][:, t - 1] = m[k]
    data["arms"]["ref"] = {"A": grid(ref, "online_acc"), "G": grid(ref, "G2"), "F": grid(ref, "major_frac"),
                           "Z": grid(ref, "zero2"), "lh": grid(ref, "label_hash", str), "oh": grid(ref, "order_hash", str),
                           "med": meds_ref}
    for a in ARMS1:
        d = src / "rlcifar" / a
        if not (d / "done.json").exists():
            continue
        pt = pd.read_csv(d / "per_task.csv")
        expected = {(s, t) for s in SEEDS for t in range(2, N_TASKS + 1)}
        if len(pt) != len(expected) or set(zip(pt.seed.astype(int), pt.task.astype(int))) != expected:
            continue                                   # incomplete -> INCOMPLETE in analyze_a1
        full = pd.concat([ref[ref.task == 1][pt.columns.intersection(ref.columns)], pt], ignore_index=True)
        wd = pd.read_csv(d / "wd_task.csv")
        meds = {k: np.full((10, N_TASKS), np.nan) for k in MED_KEYS}
        for k in MED_KEYS:
            meds[k][:, 0] = meds_ref[k][:, 0]
        for t in range(2, N_TASKS + 1):
            with np.load(_manifest_resolve(d, Path(f"units_t{t:02d}.npz"))) as f:
                m = _medians({k: f[k] for k in f.files})
            for k in MED_KEYS:
                meds[k][:, t - 1] = m[k]
        wd_ok = np.zeros((10, N_TASKS - 1), bool)
        fline = np.full((10, N_TASKS), np.nan)
        for _, r in wd.iterrows():
            s, t = int(r.seed), int(r.task)
            wd_ok[s, t - 2] = all(int(r[f"wd_steps_{nm}"]) == STEPS for nm in TENSORS) and int(r.wd_c_ok) == 1
            fline[s, t - 1] = float(r.fline2)
        data["arms"][a] = {"A": grid(full, "online_acc"), "G": grid(full, "G2"), "F": grid(full, "major_frac"),
                           "Z": grid(full, "zero2"), "lh": grid(full, "label_hash", str),
                           "oh": grid(full, "order_hash", str), "med": meds, "wd_ok": wd_ok, "fline2": fline}
    ck = OUT / "checks.json"
    c = json.loads(ck.read_text()) if ck.exists() else {}
    data["s0_ok"] = bool(c.get("S0", {}).get("pass"))
    # Codex 9.4: the arms ran in the environment the checks (S0) ran in
    envs = {a: json.loads((src / "rlcifar" / a / "provenance_start.json").read_text())["identity"]["environment"]
            for a in ARMS1 if (src / "rlcifar" / a / "provenance_start.json").exists()}
    data["env_ok"] = bool(c.get("environment")) and all(e == c.get("environment") for e in envs.values())
    data["envs"] = envs
    return data


def load_a2(src: Path) -> dict:
    data = {"arms": {}, "adamw": {}}
    for a in ARMS2:
        d = src / "c51" / a
        if not (d / "provenance.json").exists():
            continue
        data["arms"][a] = {"per_task": pd.read_csv(d / "per_task.csv"), "fresh_self": pd.read_csv(d / "fresh_self.csv"),
                           "l2": pd.read_csv(d / "l2_task.csv"), "prov": json.loads((d / "provenance.json").read_text())}
    data["R"] = {"per_task": pd.read_csv(HOST / "R_std_lr0.0001" / "per_task.csv"),
                 "fresh_control": pd.read_csv(HOST / "R_std_lr0.0001" / "fresh_control.csv")}
    data["L"] = {"per_task": pd.read_csv(HOST / "R_std_lr0.0001_l2init1e-2" / "per_task.csv")}
    for name in ("adamw5", "adamw10"):
        data["adamw"][name] = pd.read_csv(P180 / name / "per_task.csv")
    nc = src / "c51" / "_nochange"
    if (nc / "provenance.json").exists():
        data["nochange"] = {"prov": json.loads((nc / "provenance.json").read_text()), "l2": pd.read_csv(nc / "l2_task.csv")}
    ck = OUT / "checks.json"
    c = json.loads(ck.read_text()) if ck.exists() else {}
    data["reuse"] = {"a": bool(c.get("S-nochange", {}).get("pass")), "b": bool(c.get("S-nochange", {}).get("pass")),
                     "c": bool(c.get("S-host-repro", {}).get("pass")),
                     "coupled_host": bool(c.get("S-coupled-host", {}).get("pass"))}
    return data


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def fmt(x, n=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{n}f}"


def write_outputs(r1: dict, r2: dict, sc: dict, src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    (out / "verdict.json").write_text(json.dumps({"a1": r1, "a2": r2, "scores": sc, "head": head,
                                                  "written": dt.datetime.now().isoformat()}, indent=2, default=str))
    rows = [{"box": "A1", "item": k, "label": v} for k, v in r1.get("labels", {}).items()] + \
           [{"box": "A2", "item": k, "label": v} for k, v in r2.get("labels", {}).items()]
    pd.DataFrame(rows).to_csv(out / "verdict.csv", index=False)
    pd.DataFrame(r1.get("rule", {}).get("rows", [])).to_csv(out / "rule.csv", index=False)
    pd.DataFrame(r1.get("per_seed", [])).to_csv(out / "paired.csv", index=False)
    pd.DataFrame(r1.get("dose", {}).get("steps", [])).to_csv(out / "dose.csv", index=False)
    eq_rows = [{"arm": a, "layer": l, **v[l]} for a, v in r1.get("equilibrium", {}).items() for l in ("l1", "l2")]
    pd.DataFrame(eq_rows).to_csv(out / "equilibrium.csv", index=False)
    w_rows = [{"arm": a, "level": v["label"], **({"layer": l, **v[l]} if l in v else {})}
              for a, v in r1.get("width", {}).items() for l in ("l1", "l2")]
    pd.DataFrame(w_rows).to_csv(out / "width.csv", index=False)
    d_rows = [{"arm": a, "label": v["label"], **({k: v["interval"][k] for k in ("mean", "low", "high")}
                                                   if v["interval"] else {})} for a, v in r1.get("depth", {}).items()]
    pd.DataFrame(d_rows).to_csv(out / "depth.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != "d"} for r in r2.get("rows", [])]).to_csv(out / "c51_paired.csv",
                                                                                                 index=False)
    pd.DataFrame(r2.get("secondary", [])).to_csv(out / "secondary.csv", index=False)
    pr = [{"item": k, **v} for k, v in sc["labels"].items()] + [{"item": k, **v} for k, v in sc["binary"].items()]
    pd.DataFrame(pr).to_csv(out / "predictions.csv", index=False)
    (out / "summary.md").write_text(summary_md(r1, r2, sc, head))


def summary_md(r1: dict, r2: dict, sc: dict, head: str) -> str:
    L = []
    notes = OUT / "notes.md"
    L.append("# adamw_dose2_1011 — 判定（A1: RL-CIFAR × ELU→ELU の AdamW と規則の当て／A2: 5+1 CIFAR × ReLU の coupled L2）\n")
    L.append(f"> 自動生成: `analysis/adamw_dose2_1011/verdict.py`。spec: `specs/spec_adamw_dose2_1011.md`（登録 commit "
             f"`7525f974`）。集計時の HEAD `{head[:8]}`。\n")
    if notes.exists():
        L.append("## 1. 読み（手書き。`results/adamw_dose2_1011/notes.md`）\n")
        L.append(notes.read_text() + "\n")
    lab = r1.get("labels", {})
    L.append("## 2. A1 の判定\n")
    ap = r1.get("applicability", {})
    L.append(f"- (A) 欠け {ap.get('missing')}・非有限 {ap.get('nonfinite')}・hash 一致 {ap.get('hashes_equal')}・"
             f"接頭部一致 {ap.get('prefix_equal')}・S0 {ap.get('s0_ok')}")
    L.append(f"- (B) ref の床 {ap.get('ref_floor_count')}/10・ref の E2 の上端 "
             f"{fmt(ap.get('ref_E2', {}).get('high'), 4)} → **{ap.get('B_ok')}**")
    L.append(f"- (C) {ap.get('C')}\n")
    L.append("| 腕 | λ | 規則の分類 | 二値の当て | ラベル（97.5%） | 床 | 主窓 online | δE1 [97.5%] | δE2 [97.5%] | Z（主窓） | DEPTH | 結果（RESCUED ∧ ¬IMPAIRED） | 一致（主） | DEPTH の側の一致 | IMPAIRED_EARLY | 戻り（seed ごと） |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rr = {r["arm"]: r for r in r1.get("rule", {}).get("rows", [])}
    for a in ARMS1:
        x = r1.get("arms", {}).get(a)
        if not x or a not in rr:
            L.append(f"| {a} | {LAMBDA[a]} | — | — | {lab.get(a)} | | | | | | | | | | | |")
            continue
        L.append(f"| {a} | {LAMBDA[a]} | {rr[a]['class']} | {rr[a]['call']} | **{x['label']}** | {x['floor_state']} "
                 f"({x['floor_count']}/10) | {x['online_late']:.3f} | {fmt(x['E1']['mean'])} [{fmt(x['E1']['low'])}, "
                 f"{fmt(x['E1']['high'])}] | {fmt(x['E2']['mean'], 4)} [{fmt(x['E2']['low'], 4)}, {fmt(x['E2']['high'], 4)}] "
                 f"| {x['Z_late']:.4f} | {lab.get(f'{a}_depth')} | {rr[a]['outcome']} | {'○' if rr[a]['match'] else '×'} "
                 f"| {'○' if rr[a]['depth_side_match'] else '×'} | {x['early']} ({x['early_count']}) | "
                 f"{', '.join(map(str, x['returns']))} |")
    rref = r1.get("arms", {}).get("ref", {})
    L.append(f"\nref: 主窓 online {rref.get('online_late', float('nan')):.3f}・Z {rref.get('Z_late', float('nan')):.4f}・"
             f"task 1 {rref.get('online_t1', float('nan')):.3f}・task 2 {rref.get('online_t2', float('nan')):.3f}\n")
    ru = r1.get("rule", {})
    L.append(f"**RULE_PREDICTED（主・機能）: {ru.get('label')}**（一致 {ru.get('n_match')}/3）・**RULE_DEPTH（機構）: "
             f"{ru.get('depth_label')}**（{ru.get('n_depth_match')}/3）・規則が立った（機能 3/3 かつ救った λ で "
             f"DEPTH_ABOVE_LINE）: **{ru.get('rule_stands')}**・登録時の定義（RESCUED ∧ DEPTH）での一致 "
             f"{ru.get('n_registered_match')}/3（{ru.get('registered_label')}、A1-1 の採点に使う）・ラベルだけの一致 "
             f"{ru.get('n_label_only_match')}/3\n")
    L.append("Z は活性化の局所微分 e^z が float32 で厳密に 0 になる（画像, unit）の対の割合で、上流の dy を掛けた "
             "dy·e^z の零率ではない（spec §14.4）。\n")
    L.append("### 2.1 規則の因子: 実測と帯（主窓、seed 平均）\n")
    L.append("| 腕 | κ1 | κ2 | ρ1 | f | r | s2 | m = r·s2 | 中央の z̄₂ | Z | F（z ≤ −16.98） |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS1:
        f_ = r1.get("factors", {}).get(a)
        if not f_:
            continue
        cell = lambda k: f"{f_[k]['measured']:.3g} [{f_[k]['band_lo']:.3g}, {f_[k]['band_hi']:.3g}]{'' if f_[k]['inside'] else ' 外'}"
        L.append(f"| {a} | {cell('kappa1')} | {cell('kappa2')} | {cell('rho1')} | {cell('f')} | {cell('r')} | "
                 f"{cell('s2')} | {cell('m')} | {f_['zbar2_measured']:.3g} | {cell('Z')} | {cell('F')} |")
    L.append("\n### 2.2 DOSE・EQUILIBRIUM（帯 ×1.24〜×2.93、spec §14.7）・WIDTH（許容 = ref の t1–3 の傾きの seed の変動係数 "
             f"{r1.get('width_tolerance', {})}）\n")
    L.append(f"- DOSE: **{lab.get('dose')}** " + "・".join(
        f"{s['step']} {s['pos']}/{s['n']} p {s['p']:.4g} {s['dir']} 中央値 {s['median']:+.4f}" for s in r1.get("dose", {}).get("steps", [])))
    for a in ARMS1:
        eq = r1.get("equilibrium", {}).get(a)
        wd = r1.get("width", {}).get(a)
        if eq:
            L.append(f"- {a}: EQUILIBRIUM **{eq['label']}**（第 1 層 ×{eq['l1']['ratio_geo']:.2f} {eq['l1']['label']}・"
                     f"第 2 層 ×{eq['l2']['ratio_geo']:.2f} {eq['l2']['label']}）・WIDTH **{wd['label']}**" +
                     ("" if "l1" not in wd else f"（κ 第 1 層 {wd['l1']['kappa_mean']:+.4f} {wd['l1']['label']}・第 2 層 "
                                               f"{wd['l2']['kappa_mean']:+.4f} {wd['l2']['label']}）"))
    L.append("\n## 3. A2 の判定\n")
    l2 = r2.get("labels", {})
    ap2 = r2.get("applicability", {})
    L.append(f"- (A) 欠け {ap2.get('incomplete')}・照合 {ap2.get('reuse')}・バッチ列の sha256 一致 {ap2.get('fresh_batches_same')}")
    L.append(f"- (B) R の fresh gap {ap2.get('B_R_fresh_gap')} → {ap2.get('B_ok')}・(C) {ap2.get('C')}\n")
    L.append("| 腕 | λ | ラベル | 読み（帯） | ρ = (W−R)/(L−R) [95%] | 登録時の読み | WORSE | IMPAIRED | a = W−R 中央値 · +/n · p | c = W−M 中央値 · +/n · p | W−L 中央値 · +/n · p | 後期窓 W | 名目で揃えた AdamW との差（中央値 · +/n · p） |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    T = r2.get("tests", {})
    for a in ARMS2:
        if f"a|{a}" not in T:
            L.append(f"| {a} | {LAMBDA2[a]} | {l2.get(a)} | {l2.get(a + '_reading')} | | | | | | | | | |")
            continue
        ta, tc, tl = T[f"a|{a}"], T[f"c|{a}"], T[f"L|{a}"]
        inf = r2["info"][a]
        vm = inf.get("vs_matched_adamw", {})
        ri = inf.get("rho_interval", {})
        L.append(f"| {a} | {LAMBDA2[a]} | **{l2.get(a)}** | **{l2.get(a + '_reading')}** | {ri.get('mean', float('nan')):+.3f} "
                 f"[{ri.get('lo', float('nan')):+.3f}, {ri.get('hi', float('nan')):+.3f}] | {l2.get(a + '_reading_registered')} | "
                 f"{l2.get(a + '_worse')} | "
                 f"{l2.get(a + '_impaired')} | {ta['median']:+.4f} · {ta['pos']}/{ta['n']} · {ta['p']:.4g} | "
                 f"{tc['median']:+.4f} · {tc['pos']}/{tc['n']} · {tc['p']:.4g} | {tl['median']:+.4f} · {tl['pos']}/{tl['n']} · "
                 f"{tl['p']:.4g} | {inf['W_late_mean']:.4f} | {vm.get('adamw')} {vm.get('median', float('nan')):+.4f} · "
                 f"{vm.get('pos')}/{vm.get('n')} · {vm.get('p', float('nan')):.4g} |")
    if "R_late" in r2:
        L.append(f"\nR の後期窓 {np.mean(r2['R_late']):.4f}・L2 Init {np.mean(r2['L_late']):.4f}・中点 "
                 f"{(np.mean(r2['R_late']) + np.mean(r2['L_late'])) / 2:.4f}・**A2 のまとめ: {l2.get('box')}**\n")
    if r2.get("secondary"):
        L.append("| 腕 | 量 | W の中央値 | R の中央値 | 差の中央値 | +/n | p |")
        L.append("|---|---|---|---|---|---|---|")
        for r in r2["secondary"]:
            L.append(f"| {r['arm']} | {r['quantity']} | {r['W_median']:.4g} | {r.get('R_median', float('nan')):.4g} | "
                     f"{r.get('diff_median', float('nan')):+.4g} | {r.get('pos', '')}/{r.get('n', '')} | {r.get('p', float('nan')):.3g} |")
    L.append("\n## 4. 予測の採点（spec §7.3。Claude の列）\n")
    L.append("| 項目 | 実現 | p(実現) | 最頻 | 的中 | Brier |")
    L.append("|---|---|---|---|---|---|")
    for k, v in sc["labels"].items():
        if "realized" in v:
            L.append(f"| {k} | {v['realized']}（桶 {v['bucket']}） | {v['p']:.2f} | {v['mode']} | {'○' if v['hit'] else '×'} | {v['brier']:.3f} |")
        else:
            L.append(f"| {k} | 未採点（{v.get('unscored')}） | | | | |")
    L.append("\n| 二値の項目 | p | 実現 | Brier |")
    L.append("|---|---|---|---|")
    for k, v in sc["binary"].items():
        L.append(f"| {k} | {v['p']:.2f} | {v.get('outcome', '未採点（' + str(v.get('unscored')) + '）')} | "
                 f"{'' if 'brier' not in v else format(v['brier'], '.3f')} |")
    for who, key in (("Fable（参考予測。検査と Claude の予測の要約を見た後の記入）", "fable"),
                     ("Codex（盲検、10/11 16:07:48 に §7 を見る前に確定）", "codex")):
        t = sc.get(key, {})
        L.append(f"\n**{who}**\n")
        L.append("| 項目 | 実現 | p(実現) | 最頻 | 的中 | Brier |")
        L.append("|---|---|---|---|---|---|")
        for k, v in t.get("labels", {}).items():
            if "realized" in v:
                L.append(f"| {k} | {v['realized']}（桶 {v['bucket']}） | {v['p']:.2f} | {v['mode']} | {'○' if v['hit'] else '×'} | {v['brier']:.3f} |")
            else:
                L.append(f"| {k} | 未採点（{v.get('unscored')}） | | | | |")
        for k, v in t.get("binary", {}).items():
            L.append(f"| {k}（二値） | {v.get('outcome', '未採点')} | {v['p']:.2f} | | | {'' if 'brier' not in v else format(v['brier'], '.3f')} |")
    lb = [v["brier"] for v in sc["labels"].values() if "brier" in v]
    bb = [v["brier"] for v in sc["binary"].values() if "brier" in v]
    L.append(f"\nラベルの項目の Brier 平均 {np.mean(lb) if lb else float('nan'):.3f}（{len(lb)} 項目）・二値の項目の Brier 平均 "
             f"{np.mean(bb) if bb else float('nan'):.3f}（{len(bb)} 項目）\n")
    L.append("## 5. 読みの上限（spec §10）\n")
    L.append("- A1 は RL-CIFAR × ELU→ELU・std・この 3 水準・task 2 からの減衰・50 課題、A2 は 5+1 CIFAR × ReLU・この 2 水準。"
             "相対位置 r は ref の値を使い、WD の下で導いていない。零点 −103.97 はこの箱の実装（native F.elu・CUDA）の性質。\n")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(OUT), help="the run's results directory (smoke runs: results/_smoke_*)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = Path(a.src)
    out = Path(a.out) if a.out else src
    d1, d2 = load_a1(src), load_a2(src)
    r1, r2 = analyze_a1(d1), analyze_a2(d2)
    sc = score_predictions(r1, r2, d1.get("s0_ok"))
    write_outputs(r1, r2, sc, src, out)
    print(json.dumps({"A1": {k: v for k, v in r1["labels"].items()}, "A2": r2["labels"]}, indent=1, default=str))


if __name__ == "__main__":
    main()

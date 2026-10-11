#!/usr/bin/env python3
"""Verdict for adamw_dose_1010 (specs/spec_adamw_dose_1010.md sections 4, 5, 7.3).

    /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/adamw_dose_1010/verdict.py

Box 1 (Random Label MNIST, ELU -> ELU): reads results/adamw_dose_1010/mnist/runs/<arm>_s<seed>/ (units.npz from
the run directory or, after the clean-up, from the backup the manifest names), applies doors_rlmnist_1007's
endpoints and labels to the four lambda arms, then DOSE, EQUILIBRIUM, WIDTH and DEPTH.  Box 2 (5+1 CIFAR, ReLU):
reads results/adamw_dose_1010/c51/{adamw5,adamw10,_nochange,csnp_adamw10}/ and the committed R and L2 Init records
of cifar5p1_mlp_0920.  analyze1() and analyze2() are pure (data in, results out) so that checks.py S8 can run them
on synthetic data.  The Student-t machinery is analysis/mucap_el_0916/verdict.py's, imported unmodified.

Writes results/adamw_dose_1010/{verdict.json, verdict.csv, paired.csv, dose.csv, equilibrium.csv, width.csv,
depth.csv, c51_paired.csv, secondary.csv, predictions.csv, summary.md} and figures under figs/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import subprocess
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("_mucap_el_verdict", REPO / "analysis" / "mucap_el_0916" / "verdict.py")
ELV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ELV)
paired, t_quantile, t_cdf = ELV.paired, ELV.t_quantile, ELV.t_cdf

RUN_ID = "adamw_dose_1010"
OUT = REPO / "results" / RUN_ID

# ---------------------------------------------------------------- box 1 (spec 2.2, 4.1, 5.1-5.5)
ARMS1 = ("ref", "wd1e-3", "wd1e-2", "wd3e-2", "wd1e-1")
LAM_ARMS = ARMS1[1:]
LAMBDA = {"ref": 0.0, "wd1e-3": 1e-3, "wd1e-2": 1e-2, "wd3e-2": 3e-2, "wd1e-1": 1e-1}
EFFECTIVE = ("wd3e-2", "wd1e-1")                     # tau <= run / 10: the main levels
LEVEL_OF = {"wd1e-3": 0.95, "wd1e-2": 0.95, "wd3e-2": 0.975, "wd1e-1": 0.975}
LR1 = 1e-3
SEEDS = tuple(range(10))
N_TASKS = 100
SPT = 6000                                          # updates per task: the task-end diagnostic step
MAIN_WIN = (51, 100)
FORM_WINS = ((2, 10), (11, 50))
MIN_SEEDS = 8                                       # (A)
REF_FLOOR_FRAC = 0.8                                # (B)
FLOOR_K = 3.0                                       # floor: F + 3 s / sqrt(n)
CHANCE = 0.1
HALF = 0.5
EARLY = (2, 5)                                      # IMPAIRED
WD_WIN = (1, 100)                                   # (C): every task
STREAM_KEYS = ("init_sha256", "subset_idx_sha256", "labels_sha256", "batch_sha256")
E2_KEY = "dtrain_mean_l2"
TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3")
THREE = {"RESCUED": "RESCUED", "RESCUED_FUNCTION_ONLY": "PARTIAL", "ALIVE_UNRESOLVED": "PARTIAL",
         "SPLIT": "PARTIAL", "COLLAPSED": "FAIL"}
DOSE_ALPHA = 0.05
EQ_LOG2_BAND = 1.0                                  # x1/2 .. x2
D_L2 = 100
D_FULL = 784
WIDTH_TOL = 0.1
WIDTH_LIVE = (1, 10)                                # ref's live slope (spec 5.4)
DEPTH_TOL = 0.1
SIGN_ALPHA = 0.05

# ---------------------------------------------------------------- box 2 (spec 2.3, 4.2, 5.6)
ARMS2 = ("adamw5", "adamw10")
LAMBDA2 = {"adamw5": 5.0, "adamw10": 10.0}
N_TASKS2 = 30
HARD = list(range(1, N_TASKS2 + 1, 2))
EARLY2, LATE2 = HARD[:5], HARD[10:]                 # [1,3,5,7,9], [21,23,25,27,29]
T29 = 29
STEPS2 = 780
SEEDS2 = tuple(range(10))
HOST = REPO / "results" / "cifar5p1_mlp_0920"
R_REC, L_REC = HOST / "R_std_lr0.0001", HOST / "R_std_lr0.0001_l2init1e-2"
SNP_CELL = REPO / "results" / "baselines_cifar5p1_1008" / "calib" / "snp_e1e-3_s1e-4"
DIAG2 = ("dead_frac_l2", "mob_l2", "eff_rank_l2", "zbar_l2", "zsd_l2", "w_norm_l1", "w_norm_l2", "w_norm_l3")
WD2 = ("relpos_l2", "p2", "b1_norm", "b2_norm", "b3_norm", "w1_row_med", "w2_row_med", "w3_row_med")


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
    """Holm's step-down rejections, in the input order."""
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
    """UP / DOWN when the step is rejected and its signs lean that way, else FLAT."""
    pos, n, _ = sign_test(d)
    if rejected and n and pos * 2 != n:
        return "UP" if pos * 2 > n else "DOWN"
    return "FLAT"


def order_interval(d) -> tuple[float, float]:
    """[v(2), v(9)] of n = 10 values (97.9% for the median)."""
    v = np.sort(np.asarray(d, float))
    if len(v) != 10:
        return float("nan"), float("nan")
    return float(v[1]), float(v[8])


def ci_mean(x, level: float = 0.95) -> dict:
    return paired(np.asarray(x, float), level)


def slope(t, y) -> float:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    tc = t - t.mean()
    return float((tc * (y - y.mean())).sum() / (tc * tc).sum())


def c32(lr: float, lam: float) -> float:
    return float(np.float32(1 - lr * lam))


def w_star(lr: float, lam: float, d: float) -> float:
    """The rotational equilibrium row norm eta * sqrt(d / (2 (1 - c))), c = fl32(1 - eta lambda) (spec 2.4)."""
    return lr * math.sqrt(d / (2.0 * (1.0 - c32(lr, lam))))


# --------------------------------------------------------------------------
# box 1: per shard
# --------------------------------------------------------------------------

def task_ends(u: dict) -> dict[int, int]:
    t, s = np.asarray(u["task"]), np.asarray(u["step"])
    return {int(t[i]): int(i) for i in np.where(s == SPT)[0]}


def unit_mean(v) -> tuple[float, int]:
    v = np.asarray(v, dtype=np.float64)
    bad = ~np.isfinite(v)
    return (float(v[~bad].mean()) if (~bad).any() else float("nan")), int(bad.sum())


def unit_median(v) -> float:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def online(pt: pd.DataFrame) -> pd.Series:
    return pt.set_index("task")["online_acc"].astype(float)


def e1_delta(pt: pd.DataFrame, win: tuple[int, int]) -> float:
    a = online(pt)
    return float(np.mean([a.loc[t] for t in range(win[0], win[1] + 1)])) - float(a.loc[1])


def unit_window_delta(u: dict, key: str, win: tuple[int, int]) -> float:
    ends = task_ends(u)
    base = unit_mean(u[key][ends[1]])[0]
    return float(np.mean([unit_mean(u[key][ends[t]])[0] for t in range(win[0], win[1] + 1)])) - base


def window_level(pt: pd.DataFrame, win=MAIN_WIN) -> float:
    a = online(pt)
    return float(np.mean([a.loc[t] for t in range(win[0], win[1] + 1)]))


def median_series(u: dict, key: str, tasks) -> np.ndarray:
    """The unit median of `key` at the given task ends."""
    ends = task_ends(u)
    return np.array([unit_median(u[key][ends[t]]) for t in tasks], float)


def floor_call(pt: pd.DataFrame, win: tuple[int, int] = MAIN_WIN) -> dict:
    p = pt.set_index("task")
    rows = p.loc[win[0]:win[1]]
    f = rows["major_frac"].astype(float).to_numpy()
    n = len(f)
    F, s = float(f.mean()), float(f.std(ddof=1))
    thr = F + FLOOR_K * s / math.sqrt(n)
    on = rows["online_acc"].astype(float).to_numpy()
    level = float(on.mean())
    return {"F": F, "s": s, "thr": thr, "level": level, "at_floor": level <= thr,
            "task_floor_frac": float((on <= F + FLOOR_K * s).mean())}


def fit(pt: pd.DataFrame) -> pd.Series:
    a = online(pt)
    return (a - CHANCE) / (a.loc[1] - CHANCE)


def t_half(pt: pd.DataFrame) -> int | None:
    f = fit(pt)
    for t in range(2, N_TASKS + 1):
        if f.loc[t] < HALF:
            return t
    return None


def returns(pt: pd.DataFrame, run: int = 3) -> int:
    """'Return' events (spec 4.1): a stretch of >= run consecutive tasks with F(t) < 1/2, then F(t) >= 1/2."""
    f = fit(pt)
    n_ev, below = 0, 0
    for t in range(2, N_TASKS + 1):
        if f.loc[t] < HALF:
            below += 1
        else:
            if below >= run:
                n_ev += 1
            below = 0
    return n_ev


def fit_early(pt: pd.DataFrame) -> float:
    f = fit(pt)
    return float(np.mean([f.loc[t] for t in range(EARLY[0], EARLY[1] + 1)]))


def compare_times(ta, tr) -> str:
    if ta is None and tr is None:
        return "tie"
    if ta is None:
        return "later"
    if tr is None:
        return "earlier"
    return "earlier" if ta < tr else "later" if ta > tr else "tie"


def wd_worked(wd: pd.DataFrame | None, spt: int = SPT) -> tuple[bool, list]:
    """(C) for one run: every task of WD_WIN has wd_steps = spt on all six tensors and deadpix_ok = 1."""
    if wd is None:
        return False, ["no wd_task"]
    w = wd.set_index("task")
    bad = []
    for t in range(WD_WIN[0], WD_WIN[1] + 1):
        if t not in w.index:
            bad.append(f"t{t}:missing")
            continue
        r = w.loc[t]
        for nm in TENSORS:
            if int(r[f"wd_steps_{nm}"]) != spt:
                bad.append(f"t{t}:{nm}")
        if int(r["deadpix_ok"]) != 1:
            bad.append(f"t{t}:deadpix")
    return not bad, bad


def seed_validity(shards: dict, seed: int) -> str | None:
    infos = {}
    for arm in ARMS1:
        sh = shards.get((arm, seed))
        if sh is None:
            return f"{arm} missing"
        info = sh["prov"]["per_seed"][str(seed)]
        if info.get("tasks_completed") != N_TASKS or info.get("divergence", {}).get("diverged"):
            return f"{arm} incomplete or diverged"
        pt = sh["per_task"]
        if sorted(pt["task"].astype(int)) != list(range(1, N_TASKS + 1)) or \
                not np.isfinite(pt["online_acc"].astype(float)).all():
            return f"{arm} per-task rows missing or non-finite"
        ends = task_ends(sh["units"])
        if sorted(ends) != list(range(1, N_TASKS + 1)):
            return f"{arm} task ends missing"
        g = np.stack([sh["units"][k][ends[t]] for t in ends for k in ("dtrain_mean_l1", E2_KEY)])
        if not np.isfinite(g).all():
            return f"{arm} non-finite response"
        infos[arm] = tuple(info.get(k) for k in STREAM_KEYS)
    if len(set(infos.values())) != 1:
        return "streams differ across arms"
    return None


def interval_label(lo: float, hi: float, band: float, inside: str, high: str, low: str, closed: bool) -> str:
    """inside: the interval within [-band, band] (closed) or (-band, band) (open); high: lo > band; low: hi < -band."""
    if (closed and lo >= -band and hi <= band) or (not closed and lo > -band and hi < band):
        return inside
    if lo > band:
        return high
    if hi < -band:
        return low
    return "UNRESOLVED"


# --------------------------------------------------------------------------
# box 1: the analysis
# --------------------------------------------------------------------------

def analyze1(shards: dict) -> dict:
    invalid = {s: r for s in SEEDS if (r := seed_validity(shards, s)) is not None}
    valid = [s for s in SEEDS if s not in invalid]
    res = {"valid_seeds": valid, "invalid_seeds": invalid, "rows": [], "labels": {}}
    A_ok = len(valid) >= MIN_SEEDS
    if not valid:
        res["applicability"] = {"A_valid_seeds": 0, "A_ok": False, "B_ok": False, "C": {a: False for a in LAM_ARMS}}
        for a in LAM_ARMS:
            res["labels"][a] = "INAPPLICABLE"
        res["labels"]["box"] = res["labels"]["dose"] = "INAPPLICABLE"
        return res

    D = {}
    for arm in ARMS1:
        for s in valid:
            sh = shards[(arm, s)]
            for win in (MAIN_WIN, *FORM_WINS):
                D[(arm, s, "E1", win)] = e1_delta(sh["per_task"], win)
                D[(arm, s, "E2", win)] = unit_window_delta(sh["units"], E2_KEY, win)

    def deltas(arm, ep, win=MAIN_WIN):
        return np.array([D[(arm, s, ep, win)] for s in valid])

    def diffs(arm, ep, win=MAIN_WIN):
        return deltas(arm, ep, win) - deltas("ref", ep, win)

    floors = {(arm, s): floor_call(shards[(arm, s)]["per_task"]) for arm in ARMS1 for s in valid}
    state = {}
    for arm in ARMS1:
        at = [floors[(arm, s)]["at_floor"] for s in valid]
        state[arm] = "FLOORED" if all(at) else "ALIVE" if not any(at) else "SPLIT"
    ref_at = sum(floors[("ref", s)]["at_floor"] for s in valid)
    B_E2 = paired(deltas("ref", "E2"), 0.95)
    B_ok = A_ok and ref_at >= math.ceil(REF_FLOOR_FRAC * len(valid)) and B_E2["hi"] < 0
    C, C_bad = {}, {}
    for arm in LAM_ARMS:
        bad = []
        for s in valid:
            ok, b = wd_worked(shards[(arm, s)].get("wd"))
            if not ok:
                bad += [f"s{s}:{x}" for x in b[:3]]
        C[arm] = not bad
        C_bad[arm] = bad[:10]
    res["applicability"] = {"A_valid_seeds": len(valid), "A_ok": A_ok, "B_ref_at_floor": ref_at,
                            "B_ref_E2": {k: B_E2[k] for k in ("mean", "lo", "hi")}, "B_ok": B_ok,
                            "C": C, "C_failed": C_bad}
    res["floor_state"] = state
    res["floors"] = {f"{a}|{s}": v for (a, s), v in floors.items()}

    # flags
    impaired, t1_lower = {}, {}
    for arm in LAM_ARMS:
        hits = 0
        for s in valid:
            f_ref = fit_early(shards[("ref", s)]["per_task"])
            f_arm = fit_early(shards[(arm, s)]["per_task"])
            hits += f_arm < f_ref - (1.0 - f_ref)
        impaired[arm] = {"seeds_below": hits, "flag": hits > len(valid) / 2}
        d1 = np.array([float(online(shards[(arm, s)]["per_task"]).loc[1]) -
                       float(online(shards[("ref", s)]["per_task"]).loc[1]) for s in valid])
        p1 = paired(d1, 0.95)
        t1_lower[arm] = {"mean": p1["mean"], "lo": p1["lo"], "hi": p1["hi"], "flag": bool(p1["hi"] < 0)}
    res["impaired"], res["t1_lower"] = impaired, t1_lower

    def gate(arm) -> str | None:
        if not A_ok or not C[arm]:
            return "INAPPLICABLE"
        if not B_ok:
            return "NOT_REPRODUCED"
        return None

    def label(arm, st):
        g = gate(arm)
        if g:
            return g
        if state[arm] == "FLOORED":
            return "COLLAPSED"
        if state[arm] == "SPLIT":
            return "SPLIT"
        if st["E1"]["sign"] == "+":
            return "RESCUED" if st["E2"]["sign"] == "+" else "RESCUED_FUNCTION_ONLY"
        return "ALIVE_UNRESOLVED"

    labels_at = {}
    for arm in LAM_ARMS:
        for level in sorted({0.95, LEVEL_OF[arm]}, reverse=True):
            st = {ep: paired(diffs(arm, ep), level) for ep in ("E1", "E2")}
            lab = label(arm, st)
            labels_at[(arm, level)] = lab
            for ep, x in st.items():
                res["rows"].append({"arm": arm, "lam": LAMBDA[arm], "endpoint": ep,
                                    "window": f"{MAIN_WIN[0]}-{MAIN_WIN[1]}", "level": level,
                                    **{k: x[k] for k in ("n", "mean", "sd", "lo", "hi", "sign", "degenerate")},
                                    "floor_state": state[arm], "label": lab, "three": THREE.get(lab, lab),
                                    "role": "registered" if level == LEVEL_OF[arm] else "report"})
    reg = {arm: labels_at[(arm, LEVEL_OF[arm])] for arm in LAM_ARMS}
    for arm in LAM_ARMS:
        res["labels"][arm] = reg[arm]
        res["labels"][f"{arm}_three"] = THREE.get(reg[arm], reg[arm])
        if LEVEL_OF[arm] != 0.95:
            res["labels"][f"{arm}_95"] = labels_at[(arm, 0.95)]
    if not A_ok:
        res["labels"]["box"] = "INAPPLICABLE"
    elif not B_ok:
        res["labels"]["box"] = "NOT_REPRODUCED"
    else:
        k = sum(reg[a] == "RESCUED" for a in EFFECTIVE)
        res["labels"]["box"] = {2: "WD_RESCUES_BOTH", 1: "WD_RESCUES_ONE", 0: "WD_RESCUES_NONE"}[k]
    res["per_seed"] = {arm: {ep: diffs(arm, ep).tolist() for ep in ("E1", "E2")} for arm in LAM_ARMS}
    res["arm_delta"] = {arm: {ep: deltas(arm, ep).tolist() for ep in ("E1", "E2")} for arm in ARMS1}

    # ---- DOSE (spec 5.2): the main-window online LEVEL, adjacent steps, exact sign tests, Holm
    lvl = {arm: np.array([window_level(shards[(arm, s)]["per_task"]) for s in valid]) for arm in ARMS1}
    steps = []
    for k in range(1, len(ARMS1)):
        d = lvl[ARMS1[k]] - lvl[ARMS1[k - 1]]
        pos, n, p = sign_test(d)
        steps.append({"step": f"{ARMS1[k - 1]}->{ARMS1[k]}", "pos": pos, "n": n, "p": p,
                      "median": float(np.median(d)), "d": d.tolist()})
    rej = holm([s_["p"] for s_ in steps])
    for s_, r in zip(steps, rej):
        s_["holm_rejected"] = bool(r)
        s_["dir"] = direction(s_["d"], r)
    dirs = [s_["dir"] for s_ in steps]
    if not A_ok:
        dose = "INAPPLICABLE"
    elif not B_ok:
        dose = "NOT_REPRODUCED"
    elif not all(C.values()):
        dose = "INAPPLICABLE"
    elif "DOWN" not in dirs and "UP" in dirs:
        dose = "DOSE_MONOTONE"
    elif any(dirs[i] == "UP" and "DOWN" in dirs[i + 1:] for i in range(len(dirs))):
        dose = "DOSE_PEAKED"
    elif all(d_ == "FLAT" for d_ in dirs):
        dose = "DOSE_FLAT"
    else:
        dose = "DOSE_OTHER"
    res["labels"]["dose"] = dose
    res["dose_steps"] = steps
    res["levels_online"] = {arm: lvl[arm].tolist() for arm in ARMS1}

    # ---- EQUILIBRIUM (spec 5.3): every lambda arm computed, the effective ones labelled
    eq = {}
    for arm in LAM_ARMS:
        lam = LAMBDA[arm]
        per_layer = {}
        for li in (1, 2):
            obs, rho, rho784 = [], [], []
            for s in valid:
                u = shards[(arm, s)]["units"]
                o = float(np.mean(median_series(u, f"row_norm_l{li}", range(MAIN_WIN[0], MAIN_WIN[1] + 1))))
                d = float(shards[(arm, s)]["prov"]["per_seed"][str(s)]["d_eff"]) if li == 1 else D_L2
                obs.append(o)
                rho.append(o / w_star(LR1, lam, d))
                rho784.append(o / w_star(LR1, lam, D_FULL if li == 1 else D_L2))
            l2 = np.log2(np.array(rho))
            ci = paired(l2, 0.95)
            lab = interval_label(ci["lo"], ci["hi"], EQ_LOG2_BAND, "MATCH", "OFF_HIGH", "OFF_LOW", closed=True)
            per_layer[f"l{li}"] = {"label": lab, "log2_mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
                                   "ratio_geomean": float(2 ** ci["mean"]), "obs_mean": float(np.mean(obs)),
                                   "w_star_mean": float(np.mean([o / r for o, r in zip(obs, rho)])),
                                   "ratio784_geomean": float(2 ** np.mean(np.log2(rho784))),
                                   "rho": list(map(float, rho))}
        g = gate(arm)
        lv = g if g else ("EQUILIBRIUM_MATCH" if all(v["label"] == "MATCH" for v in per_layer.values())
                          else "EQUILIBRIUM_OFF")
        eq[arm] = {"label": lv, **per_layer}
    res["equilibrium"] = eq
    for arm in EFFECTIVE:
        res["labels"][f"eq_{arm}"] = eq[arm]["label"]

    # ---- WIDTH (spec 5.4)
    g_ref = {}
    for s in valid:
        u = shards[("ref", s)]["units"]
        tl = list(range(WIDTH_LIVE[0], WIDTH_LIVE[1] + 1))
        g_ref[s] = {li: slope(tl, median_series(u, f"sigma_l{li}", tl)) for li in (1, 2)}
    width = {}
    for arm in LAM_ARMS:
        tm = list(range(MAIN_WIN[0], MAIN_WIN[1] + 1))
        layers = {}
        for li in (1, 2):
            kap = [slope(tm, median_series(shards[(arm, s)]["units"], f"sigma_l{li}", tm)) / g_ref[s][li]
                   if g_ref[s][li] != 0 else float("nan") for s in valid]
            ci = paired(np.array(kap), 0.95)
            layers[f"l{li}"] = {"label": interval_label(ci["lo"], ci["hi"], WIDTH_TOL, "STOPPED", "GROWING",
                                                        "SHRINKING", closed=False),
                                "kappa_mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "kappa": kap,
                                "s_main_mean": float(np.mean([np.mean(median_series(shards[(arm, s)]["units"],
                                                                                     f"sigma_l{li}", tm))
                                                              for s in valid]))}
        g = gate(arm)
        if g:
            lv = g
        elif state[arm] == "FLOORED":
            lv = "WIDTH_FROZEN"
        elif state[arm] == "SPLIT":
            lv = "WIDTH_SPLIT"
        else:
            lv = "WIDTH_STOPPED" if all(v["label"] == "STOPPED" for v in layers.values()) else "WIDTH_NOT_STOPPED"
        width[arm] = {"label": lv, **layers}
    res["width"] = width
    res["g_ref"] = {str(s): v for s, v in g_ref.items()}
    for arm in EFFECTIVE:
        res["labels"][f"width_{arm}"] = width[arm]["label"]

    # ---- DEPTH (spec 5.5): the RESCUED levels
    def zero_pairs(u):
        ends = task_ends(u)
        return float(np.mean([unit_mean(u["dtrain_zero_l2"][ends[t]])[0]
                              for t in range(MAIN_WIN[0], MAIN_WIN[1] + 1)]))
    depth = {}
    Zref = {s: zero_pairs(shards[("ref", s)]["units"]) for s in valid}
    for arm in LAM_ARMS:
        Z = {s: zero_pairs(shards[(arm, s)]["units"]) for s in valid}
        zeta = [Z[s] / Zref[s] for s in valid if Zref[s] > 0]
        ci = paired(np.array(zeta), 0.95)
        if reg[arm] != "RESCUED":
            lab = "NOT_EVALUATED"
        elif ci["hi"] < DEPTH_TOL:
            lab = "DEPTH_ABOVE_FLOOR"
        elif ci["lo"] > DEPTH_TOL:
            lab = "DEPTH_AT_FLOOR"
        else:
            lab = "DEPTH_UNRESOLVED"
        depth[arm] = {"label": lab, "zeta_mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "n": len(zeta),
                      "Z_mean": float(np.mean(list(Z.values())))}
        res["labels"][f"depth_{arm}"] = lab
    res["depth"] = depth
    res["Z_ref_mean"] = float(np.mean(list(Zref.values())))

    # ---- report only: timing, returns, floor-task fraction, levels and trajectories
    th = {(arm, s): t_half(shards[(arm, s)]["per_task"]) for arm in ARMS1 for s in valid}
    timing = {}
    for arm in LAM_ARMS:
        cmp_ = [compare_times(th[(arm, s)], th[("ref", s)]) for s in valid]
        e, l_ = cmp_.count("earlier"), cmp_.count("later")
        p = sign_test([1] * e + [-1] * l_)[2]
        timing[arm] = {"label": ("EARLIER" if e > l_ else "LATER") if (e + l_ > 0 and p < SIGN_ALPHA)
                       else "NO_TIMING_DIFF", "earlier": e, "later": l_, "ties": cmp_.count("tie"), "p": p}
    res["timing"] = timing
    res["t_half"] = {f"{a}|{s}": v for (a, s), v in th.items()}
    res["returns"] = {f"{a}|{s}": returns(shards[(a, s)]["per_task"]) for a in ARMS1 for s in valid}
    traj, levels = {}, {}
    for arm in ARMS1:
        for s in valid:
            u = shards[(arm, s)]["units"]
            for key in ("sigma_l1", "sigma_l2", "row_norm_l1", "row_norm_l2", "d_l2", "zbar_l2", "b2",
                        "dtrain_mean_l2", "dtrain_zero_l2", "pplus_l2"):
                if key in u:
                    for win in (MAIN_WIN, *FORM_WINS, (1, 1)):
                        v = float(np.mean(median_series(u, key, range(win[0], win[1] + 1))))
                        levels.setdefault(f"{arm}|med_{key}|{win[0]}-{win[1]}", []).append(v)
            pt = shards[(arm, s)]["per_task"].set_index("task")
            for col in ("online_acc", "memo_acc"):
                for win in (MAIN_WIN, *FORM_WINS, (1, 1)):
                    levels.setdefault(f"{arm}|{col}|{win[0]}-{win[1]}", []).append(
                        float(pt.loc[win[0]:win[1], col].astype(float).mean()))
            wd = shards[(arm, s)].get("wd")
            if wd is not None:
                w = wd.set_index("task")
                for col in ("w1_fro", "w2_fro", "w3_fro", "w3_row_med", "w3_row_max", "b1_norm", "b2_norm",
                            "b3_norm", "z2_zero_pairs"):
                    if col in w:
                        levels.setdefault(f"{arm}|{col}|{MAIN_WIN[0]}-{MAIN_WIN[1]}", []).append(
                            float(w.loc[MAIN_WIN[0]:MAIN_WIN[1], col].astype(float).mean()))
            ends = task_ends(u)
            for t in (1, 2, 3, 5, 10, 20, 50, 100):
                for key in ("sigma_l1", "sigma_l2", "row_norm_l1", "row_norm_l2", "d_l2", "dtrain_mean_l2",
                            "dtrain_zero_l2"):
                    if key in u:
                        traj.setdefault(f"{arm}|{key}|{t}", []).append(unit_median(u[key][ends[t]])
                                                                       if key != "dtrain_zero_l2"
                                                                       else unit_mean(u[key][ends[t]])[0])
                traj.setdefault(f"{arm}|online_acc|{t}", []).append(float(online(shards[(arm, s)]["per_task"]).loc[t]))
    series = {}
    for arm in ARMS1:
        acc = {}
        for s in valid:
            u = shards[(arm, s)]["units"]
            ends = task_ends(u)
            on = online(shards[(arm, s)]["per_task"])
            acc.setdefault("online_acc", []).append([float(on.loc[t]) for t in range(1, N_TASKS + 1)])
            for key in ("sigma_l1", "sigma_l2", "row_norm_l1", "row_norm_l2", "d_l2", "zbar_l2"):
                if key in u:
                    acc.setdefault(key, []).append([unit_median(u[key][ends[t]]) for t in range(1, N_TASKS + 1)])
            if "dtrain_zero_l2" in u:
                acc.setdefault("zero_pairs_l2", []).append([unit_mean(u["dtrain_zero_l2"][ends[t]])[0]
                                                            for t in range(1, N_TASKS + 1)])
        series[arm] = {k: np.mean(np.array(v, float), axis=0).tolist() for k, v in acc.items()}
    res["series"] = series
    res["levels_seed"] = levels
    res["levels"] = {k: float(np.mean(v)) for k, v in levels.items()}
    res["traj"] = {k: float(np.mean(v)) for k, v in traj.items()}
    res["task_floor_frac"] = {arm: float(np.mean([floors[(arm, s)]["task_floor_frac"] for s in valid]))
                              for arm in ARMS1}
    return res


# --------------------------------------------------------------------------
# box 2
# --------------------------------------------------------------------------

def per_seed_window(pt: pd.DataFrame, col: str, tasks) -> dict[int, float]:
    out = {}
    for s, g in pt.groupby("seed"):
        g = g.set_index("task")
        if all(t in g.index for t in tasks):
            out[int(s)] = float(np.mean([float(g.loc[t, col]) for t in tasks]))
    return out


def analyze2(data: dict) -> dict:
    """data: {"arms": {arm: {"per_task", "fresh_self", "wd", "prov"}}, "R": {"per_task", "fresh_control"},
    "L": {"per_task"}, "nochange": {"wd", "prov"}, "reuse": {"a": bool, "b": bool, "c": bool}}."""
    res = {"labels": {}, "rows": []}
    arms = data.get("arms", {})
    missing = [a for a in ARMS2 if a not in arms]
    incomplete = list(missing)
    for a in ARMS2:
        if a in missing:
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
    reuse_ok = all(reuse.get(k) is True for k in ("a", "b", "c"))
    shas = {a: arms[a]["prov"].get("fresh_batches_sha256") for a in ARMS2 if a in arms}
    shas["nochange"] = data.get("nochange", {}).get("prov", {}).get("fresh_batches_sha256")
    sha_ok = len(set(shas.values())) == 1 and None not in shas.values()
    res["applicability"] = {"incomplete": incomplete, "reuse": reuse, "reuse_ok": reuse_ok,
                            "fresh_batches_sha256": shas, "fresh_batches_same": sha_ok}
    if incomplete:
        for a in ARMS2:
            res["labels"][a] = "INCOMPLETE"
        res["labels"]["box"] = "INCOMPLETE"
        return res
    if not (reuse_ok and sha_ok):
        for a in ARMS2:
            res["labels"][a] = "CHECK_FAILED"
        res["labels"]["box"] = "CHECK_FAILED"
        return res
    # (B) R's fresh gap
    fg = data["R"]["fresh_control"]["fresh_gap"].astype(float).to_numpy()
    B = paired(fg, 0.95)
    B_ok = len(fg) == len(SEEDS2) and B["lo"] > 0
    res["applicability"]["B_R_fresh_gap"] = {k: B[k] for k in ("mean", "lo", "hi")}
    res["applicability"]["B_ok"] = bool(B_ok)
    # (C)
    C = {}
    for a in ARMS2:
        wd = arms[a]["wd"]
        C[a] = bool(len(wd) == len(SEEDS2) * N_TASKS2 and all(
            (wd[f"wd_steps_{nm}"].astype(int) == wd["steps"].astype(int)).all() for nm in TENSORS)
                    and (wd["steps"].astype(int) == STEPS2).all())
    res["applicability"]["C"] = C

    Rl = per_seed_window(data["R"]["per_task"], "online_acc", LATE2)
    Ll = per_seed_window(data["L"]["per_task"], "online_acc", LATE2)
    Re = per_seed_window(data["R"]["per_task"], "online_acc", EARLY2)
    res["R_late"] = [Rl[s] for s in SEEDS2]
    res["L_late"] = [Ll[s] for s in SEEDS2]
    a_d, c_d, info = {}, {}, {}
    for a in ARMS2:
        Wl = per_seed_window(arms[a]["per_task"], "online_acc", LATE2)
        We = per_seed_window(arms[a]["per_task"], "online_acc", EARLY2)
        a_d[a] = np.array([Wl[s] - Rl[s] for s in SEEDS2])
        c_d[a] = np.array([Wl[s] - (Rl[s] + Ll[s]) / 2.0 for s in SEEDS2])
        e_d = np.array([We[s] - Re[s] for s in SEEDS2])
        imp = paired(e_d, 0.95)
        info[a] = {"W_late": [Wl[s] for s in SEEDS2], "early_diff": e_d.tolist(),
                   "impaired": {"mean": imp["mean"], "lo": imp["lo"], "hi": imp["hi"], "flag": bool(imp["hi"] < 0)},
                   "rho": [float(a_d[a][i] / (Ll[s] - Rl[s])) for i, s in enumerate(SEEDS2)]}
    tests = {}
    for fam, dd in (("a", a_d), ("c", c_d)):
        ps = [sign_test(dd[a])[2] for a in ARMS2]
        rej = holm(ps)
        for a, p, r in zip(ARMS2, ps, rej):
            pos, n, _ = sign_test(dd[a])
            tests[(fam, a)] = {"pos": pos, "n": n, "p": p, "holm_rejected": bool(r), "dir": direction(dd[a], r),
                               "median": float(np.median(dd[a])), "ord_lo": order_interval(dd[a])[0],
                               "ord_hi": order_interval(dd[a])[1], "mean": float(np.mean(dd[a]))}
    for a in ARMS2:
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
        res["labels"][a] = lab
        res["labels"][f"{a}_worse"] = bool(lab in ("NO_RESCUE",) and ta["dir"] == "DOWN")
        res["labels"][f"{a}_impaired"] = info[a]["impaired"]["flag"]
        for fam in ("a", "c"):
            res["rows"].append({"arm": a, "lam": LAMBDA2[a], "family": fam, **tests[(fam, a)],
                                "d": (a_d if fam == "a" else c_d)[a].tolist(), "label": lab})
    labs = [res["labels"][a] for a in ARMS2]
    if any(l_ in ("INAPPLICABLE", "NOT_REPRODUCED") for l_ in labs):
        res["labels"]["box"] = "NOT_REPRODUCED" if "NOT_REPRODUCED" in labs else "INAPPLICABLE"
    elif all(l_ == "NO_RESCUE" for l_ in labs):
        res["labels"]["box"] = "WD_NO_RESCUE_RELU"
    elif "RESCUED" in labs:
        res["labels"]["box"] = "WD_RESCUES_RELU"
    else:
        res["labels"]["box"] = "WD_PARTIAL_RELU"
    res["info"] = info
    res["tests"] = {f"{f}|{a}": v for (f, a), v in tests.items()}

    # report only: diagnostics, late window, seed medians and paired differences vs R
    sec = []
    Rpt = data["R"]["per_task"]
    Rwd = data.get("nochange", {}).get("wd")
    for a in ARMS2:
        for col in DIAG2:
            rv = per_seed_window(Rpt, col, LATE2)
            wv = per_seed_window(arms[a]["per_task"], col, LATE2)
            d = np.array([wv[s] - rv[s] for s in SEEDS2])
            pos, n, p = sign_test(d)
            sec.append({"arm": a, "quantity": col, "W_median": float(np.median([wv[s] for s in SEEDS2])),
                        "R_median": float(np.median([rv[s] for s in SEEDS2])), "diff_median": float(np.median(d)),
                        "pos": pos, "n": n, "p": p})
        for col in WD2:
            wv = per_seed_window(arms[a]["wd"], col, LATE2)
            rv = per_seed_window(Rwd, col, LATE2) if Rwd is not None else {}
            row = {"arm": a, "quantity": col, "W_median": float(np.median([wv[s] for s in SEEDS2]))}
            if rv:
                d = np.array([wv[s] - rv[s] for s in SEEDS2])
                pos, n, p = sign_test(d)
                row |= {"R_median": float(np.median([rv[s] for s in SEEDS2])), "diff_median": float(np.median(d)),
                        "pos": pos, "n": n, "p": p}
            sec.append(row)
        fs = arms[a]["fresh_self"].set_index("seed")
        Rf = data["R"]["fresh_control"].set_index("seed")
        W29 = per_seed_window(arms[a]["per_task"], "online_acc", [T29])
        gap_self = [float(fs.loc[s, "fresh_self_online_acc"]) - W29[s] for s in SEEDS2]
        gap_common = [float(Rf.loc[s, "fresh_online_acc"]) - W29[s] for s in SEEDS2]
        info[a]["fresh_gap_self_median"] = float(np.median(gap_self))
        info[a]["fresh_gap_common_median"] = float(np.median(gap_common))
        info[a]["fresh_gap_self"] = gap_self
        info[a]["fresh_gap_common"] = gap_common
    res["secondary"] = sec
    res["R_fresh_gap_median"] = float(np.median(fg))
    # C-snp (report only)
    cs = data.get("csnp")
    if cs is not None:
        wv = per_seed_window(cs["adamw10"], "online_acc", LATE2)
        sv = per_seed_window(cs["snp"], "online_acc", LATE2)
        common = sorted(set(wv) & set(sv))
        d = np.array([wv[s] - sv[s] for s in common])
        lo, hi = order_interval(d)
        res["csnp"] = {"n": len(common), "median": float(np.median(d)) if len(d) else float("nan"),
                       "ord_lo": lo, "ord_hi": hi, "d": d.tolist(),
                       "adamw10_late_mean": float(np.mean([wv[s] for s in common])) if common else float("nan"),
                       "snp_late_mean": float(np.mean([sv[s] for s in common])) if common else float("nan")}
    return res


# --------------------------------------------------------------------------
# spec 7.3: scoring the registered predictions (Claude's column; Issa's and Fable's were left blank)
# --------------------------------------------------------------------------

OTHER = "OTHER"
LABELS5 = ("RESCUED", "SPLIT", "COLLAPSED", "RESCUED_FUNCTION_ONLY", "ALIVE_UNRESOLVED", OTHER)
CLAUDE_P = {
    "B1-1": {"COLLAPSED": .83, "SPLIT": .08, "RESCUED": .03, "RESCUED_FUNCTION_ONLY": .02, "ALIVE_UNRESOLVED": .02,
             OTHER: .02},
    "B1-2": {"RESCUED": .38, "SPLIT": .28, "COLLAPSED": .17, "RESCUED_FUNCTION_ONLY": .08, "ALIVE_UNRESOLVED": .06,
             OTHER: .03},
    "B1-3": {"RESCUED": .74, "SPLIT": .08, "COLLAPSED": .08, "RESCUED_FUNCTION_ONLY": .05, "ALIVE_UNRESOLVED": .03,
             OTHER: .02},
    "B1-4": {"RESCUED": .78, "RESCUED_FUNCTION_ONLY": .06, "SPLIT": .05, "COLLAPSED": .05, "ALIVE_UNRESOLVED": .04,
             OTHER: .02},
    "B1-5": {"WD_RESCUES_BOTH": .62, "WD_RESCUES_ONE": .28, "WD_RESCUES_NONE": .08, OTHER: .02},
    "B1-6": {"DOSE_MONOTONE": .45, "DOSE_PEAKED": .42, "DOSE_OTHER": .11, "DOSE_FLAT": .02},
    "B1-7": {"EQUILIBRIUM_MATCH": .42, "EQUILIBRIUM_OFF": .58},
    "B1-7.l1": {"MATCH": .52, "OFF_HIGH": .30, "OFF_LOW": .06, "UNRESOLVED": .12},
    "B1-7.l2": {"MATCH": .55, "OFF_HIGH": .25, "OFF_LOW": .10, "UNRESOLVED": .10},
    "B1-8": {"EQUILIBRIUM_MATCH": .36, "EQUILIBRIUM_OFF": .64},
    "B1-8.l1": {"MATCH": .48, "OFF_HIGH": .34, "OFF_LOW": .06, "UNRESOLVED": .12},
    "B1-8.l2": {"MATCH": .50, "OFF_HIGH": .30, "OFF_LOW": .08, "UNRESOLVED": .12},
    "B1-9": {"WIDTH_STOPPED": .74, "WIDTH_NOT_STOPPED": .12, "FROZEN_OR_SPLIT": .14},
    "B1-10": {"WIDTH_STOPPED": .78, "WIDTH_NOT_STOPPED": .12, "FROZEN_OR_SPLIT": .10},
    "C1": {"NO_RESCUE": .91, "PARTIAL": .05, "RESCUED": .02, OTHER: .02},
    "C2": {"NO_RESCUE": .90, "PARTIAL": .06, "RESCUED": .02, OTHER: .02},
    "C3": {"WD_NO_RESCUE_RELU": .86, "WD_PARTIAL_RELU": .09, "WD_RESCUES_RELU": .03, OTHER: .02},
}
DEPTH_P = {"wd3e-2": .90, "wd1e-1": .95, "wd1e-2": .45}
BINARY_P = {"S1": .98, "S2": .90, "S3": .95, "S4": .85, "S5": .60, "S6": .30, "S7": .65, "S8": .55, "S9": .85,
            "S10": .55, "S11": .40, "S12": .85, "S13": .55, "S14": .85, "C1w": .78, "C2w": .70, "C4": .75,
            "C5": .90, "C6": .75, "C7": .80, "C8": .80, "C9": .85, "C10": .50, "C11": .70}


def score_item(table: dict, realized_bucket: str, realized: str) -> dict:
    keys = set(table) | {realized_bucket}
    brier = sum((table.get(k, 0.0) - (1.0 if k == realized_bucket else 0.0)) ** 2 for k in keys)
    mode = max(table, key=table.get)
    return {"realized": realized, "bucket": realized_bucket, "p_realized": float(table.get(realized_bucket, 0.0)),
            "mode": mode, "mode_hit": mode == realized_bucket, "brier": float(brier)}


def _bucket(label: str, table: dict) -> str:
    return label if label in table and label != OTHER else OTHER


def paired_lower(d) -> bool:
    """'lower than R': exact two-sided sign test p < 0.05 with most differences negative."""
    pos, n, p = sign_test(d)
    return bool(p < 0.05 and pos * 2 < n)


def score_predictions(r1: dict, r2: dict, ref_identity: dict | None) -> dict:
    lab1, lab2 = r1["labels"], r2["labels"]
    out = {"labels": {}, "binary": {}}
    items = {"B1-1": lab1.get("wd1e-3"), "B1-2": lab1.get("wd1e-2"), "B1-3": lab1.get("wd3e-2"),
             "B1-4": lab1.get("wd1e-1"), "B1-5": lab1.get("box"), "B1-6": lab1.get("dose"),
             "C1": lab2.get("adamw5"), "C2": lab2.get("adamw10"), "C3": lab2.get("box")}
    for k, v in items.items():
        if v is None:
            continue
        out["labels"][k] = score_item(CLAUDE_P[k], _bucket(v, CLAUDE_P[k]), v)
    eq = r1.get("equilibrium", {})
    for k, arm in (("B1-7", "wd3e-2"), ("B1-8", "wd1e-1")):
        if arm in eq and eq[arm]["label"] in CLAUDE_P[k]:
            out["labels"][k] = score_item(CLAUDE_P[k], eq[arm]["label"], eq[arm]["label"])
            for li in ("l1", "l2"):
                out["labels"][f"{k}.{li}"] = score_item(CLAUDE_P[f"{k}.{li}"], eq[arm][li]["label"],
                                                        eq[arm][li]["label"])
        else:
            out["labels"][k] = {"unscored": f"{arm} {eq.get(arm, {}).get('label')}"}
    wd_ = r1.get("width", {})
    for k, arm in (("B1-9", "wd3e-2"), ("B1-10", "wd1e-1")):
        lab = wd_.get(arm, {}).get("label")
        if lab in ("WIDTH_STOPPED", "WIDTH_NOT_STOPPED"):
            out["labels"][k] = score_item(CLAUDE_P[k], lab, lab)
        elif lab in ("WIDTH_FROZEN", "WIDTH_SPLIT"):
            out["labels"][k] = score_item(CLAUDE_P[k], "FROZEN_OR_SPLIT", lab)
        else:
            out["labels"][k] = {"unscored": f"{arm} {lab}"}
    for arm, p in DEPTH_P.items():
        lab = r1.get("depth", {}).get(arm, {}).get("label")
        key = f"B1-11.{arm}"
        if lab in ("DEPTH_ABOVE_FLOOR", "DEPTH_AT_FLOOR", "DEPTH_UNRESOLVED"):
            o = lab == "DEPTH_ABOVE_FLOOR"
            out["binary"][key] = {"p": p, "true": o, "brier": (p - o) ** 2}
        else:
            out["binary"][key] = {"p": p, "true": None, "brier": None, "unscored": f"{arm} not RESCUED ({lab})"}

    lv = r1.get("levels", {})
    ap = r1.get("applicability", {})
    rets = r1.get("returns", {})
    valid = r1.get("valid_seeds", [])
    ls = r1.get("levels_seed", {})
    t_half_ = r1.get("t_half", {})
    dose_steps = r1.get("dose_steps", [])

    def last_step_down():
        return bool(dose_steps and dose_steps[-1].get("dir") == "DOWN")

    s14 = None
    if ls.get("wd1e-3|med_sigma_l2|51-100") and ls.get("ref|med_sigma_l2|51-100"):
        s14 = all(a < b for a, b in zip(ls["wd1e-3|med_sigma_l2|51-100"], ls["ref|med_sigma_l2|51-100"]))
    bin_ = {
        "S1": ap.get("B_ok"),
        "S2": None if ref_identity is None else bool(ref_identity["all_identical"]),
        "S3": all(ap.get("C", {}).values()) if ap.get("C") else None,
        "S4": not any(v["flag"] for v in r1.get("impaired", {}).values()) if r1.get("impaired") else None,
        "S5": r1.get("t1_lower", {}).get("wd1e-1", {}).get("flag"),
        "S6": r1.get("t1_lower", {}).get("wd3e-2", {}).get("flag"),
        "S7": sum(t_half_.get(f"wd1e-2|{s}") is not None for s in valid) >= 5 if valid else None,
        "S8": any(rets.get(f"wd1e-2|{s}", 0) > 0 for s in valid) if valid else None,
        "S9": all(rets.get(f"wd1e-3|{s}", 0) == 0 for s in valid) if valid else None,
        "S10": (0.70 <= lv["wd3e-2|online_acc|51-100"] <= 0.90) if "wd3e-2|online_acc|51-100" in lv else None,
        "S11": last_step_down() if dose_steps else None,
        "S12": (lv["wd3e-2|med_sigma_l2|51-100"] < 5) if "wd3e-2|med_sigma_l2|51-100" in lv else None,
        "S13": (-2 <= lv["wd3e-2|med_d_l2|51-100"] <= -0.5) if "wd3e-2|med_d_l2|51-100" in lv else None,
        "S14": s14,
    }
    sec2 = {(r["arm"], r["quantity"]): r for r in r2.get("secondary", [])}
    info2 = r2.get("info", {})

    def diff_lower(arm, q):
        r = sec2.get((arm, q))
        return None if r is None or "p" not in r else bool(r["p"] < 0.05 and r["pos"] * 2 < r["n"])

    def diff_higher(arm, q):
        r = sec2.get((arm, q))
        return None if r is None or "p" not in r else bool(r["p"] < 0.05 and r["pos"] * 2 > r["n"])

    bin_ |= {
        "C1w": r2.get("labels", {}).get("adamw5_worse") if "adamw5" in info2 else None,
        "C2w": r2.get("labels", {}).get("adamw10_worse") if "adamw10" in info2 else None,
        "C4": info2.get("adamw5", {}).get("impaired", {}).get("flag"),
        "C5": info2.get("adamw10", {}).get("impaired", {}).get("flag"),
        "C6": diff_lower("adamw10", "dead_frac_l2"),
        "C7": diff_higher("adamw10", "relpos_l2"),
        "C8": diff_lower("adamw10", "eff_rank_l2"),
        "C9": diff_lower("adamw5", "eff_rank_l2"),
        "C10": diff_higher("adamw5", "relpos_l2"),
        "C11": (r2["csnp"]["ord_lo"] <= 0 <= r2["csnp"]["ord_hi"]) if r2.get("csnp") else None,
    }
    for k, hit in bin_.items():
        p = BINARY_P[k]
        out["binary"][k] = {"p": p, "true": hit, "brier": None if hit is None else float((p - float(hit)) ** 2)}
    return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _manifest_index() -> dict:
    m = OUT / "backup_manifest.json"
    if not m.exists():
        return {}
    return {f["source"]: f for f in json.loads(m.read_text())["files"]}


def load1(src: Path) -> tuple[dict, list]:
    shards, missing = {}, []
    man = _manifest_index()
    for arm in ARMS1:
        for seed in SEEDS:
            d = src / f"{arm}_s{seed}"
            if not (d / "provenance.json").exists():
                missing.append(f"{arm}_s{seed}")
                continue
            up = d / "units.npz"
            if not up.exists():
                f = man.get(str(up.relative_to(REPO)))
                if f is None or not Path(f["backup"]).exists():
                    missing.append(f"{arm}_s{seed}:units")
                    continue
                up = Path(f["backup"])
            pre = f"s{seed}_"
            with np.load(up) as z:
                units = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
            wdp = d / "wd_task.csv"
            shards[(arm, seed)] = {"units": units, "per_task": pd.read_csv(d / "per_task.csv"),
                                   "wd": pd.read_csv(wdp) if wdp.exists() else None,
                                   "prov": json.loads((d / "provenance.json").read_text())}
    return shards, missing


def load2(root: Path) -> dict:
    data = {"arms": {}}
    for a in ARMS2:
        d = root / a
        if not (d / "provenance.json").exists():
            continue
        data["arms"][a] = {"per_task": pd.read_csv(d / "per_task.csv"),
                           "fresh_self": pd.read_csv(d / "fresh_self.csv") if (d / "fresh_self.csv").exists() else None,
                           "wd": pd.read_csv(d / "wd_task.csv"), "prov": json.loads((d / "provenance.json").read_text())}
    data["R"] = {"per_task": pd.read_csv(R_REC / "per_task.csv"), "fresh_control": pd.read_csv(R_REC / "fresh_control.csv")}
    data["L"] = {"per_task": pd.read_csv(L_REC / "per_task.csv")}
    nc = root / "_nochange"
    if (nc / "provenance.json").exists():
        data["nochange"] = {"wd": pd.read_csv(nc / "wd_task.csv"), "prov": json.loads((nc / "provenance.json").read_text())}
    ck = OUT / "checks.json"
    if ck.exists():
        c = json.loads(ck.read_text())
        a_ok = c.get("S-nochange", {}).get("detail", {})
        data["reuse"] = {"a": bool(a_ok.get("a_per_task_identical_to_host_now") and a_ok.get("a_fresh_identical_to_host_now")),
                         "b": bool(a_ok.get("b_non_eff_rank_identical") and a_ok.get("b_eff_rank_inside_band")
                                   and a_ok.get("b_fresh_identical_to_committed")),
                         "c": bool(c.get("S-host-repro", {}).get("pass"))}
    cs = root / "csnp_adamw10"
    if (cs / "provenance.json").exists() and (SNP_CELL / "per_task.csv").exists():
        data["csnp"] = {"adamw10": pd.read_csv(cs / "per_task.csv"), "snp": pd.read_csv(SNP_CELL / "per_task.csv")}
    return data


def ref_identity(shards: dict) -> dict:
    """Spec S2 (report): the rerun ref against l2cap_ee_0917's ref, all 100 tasks."""
    l2 = REPO / "results" / "l2cap_ee_0917"
    man = json.loads((l2 / "backup_manifest.json").read_text())
    bk = {f["source_rel"]: f for f in man["files"]}
    per, all_ok = {}, True
    for s in SEEDS:
        sh = shards.get(("ref", s))
        if sh is None:
            per[s] = "missing"
            all_ok = False
            continue
        d = l2 / "runs" / f"ref_s{s}"
        rec = pd.read_csv(d / "per_task.csv")
        got = sh["per_task"]
        cols_rec = [c for c in rec.columns if c != "arm"]
        same_cols = set(cols_rec) == set(c for c in got.columns if c != "arm")
        rows_ok = same_cols and rec[cols_rec].shape == got[cols_rec].shape and \
            bool(np.array_equal(rec[cols_rec].to_numpy(float), got[cols_rec].to_numpy(float), equal_nan=True))
        f = bk.get(f"results/l2cap_ee_0917/runs/ref_s{s}/units.npz")
        arr_ok, why = False, ""
        if f is not None and Path(f["backup"]).exists() and \
                hashlib.sha256(Path(f["backup"]).read_bytes()).hexdigest() == f["sha256"]:
            with np.load(f["backup"]) as z:
                pre = f"s{s}_"
                keys = {k[len(pre):] for k in z.files}
                arr_ok = keys == set(sh["units"])
                for k in keys:
                    if not np.array_equal(z[pre + k], sh["units"].get(k), equal_nan=True):
                        arr_ok, why = False, k
                        break
        else:
            why = "archive missing or sha256 mismatch"
        pv = json.loads((d / "provenance.json").read_text())["per_seed"][str(s)]
        mine = sh["prov"]["per_seed"][str(s)]
        hashes_ok = all(pv[k] == mine[k] for k in ("init_sha256", "subset_idx_sha256", "labels_sha256",
                                                   "batch_sha256", "task1_end_state_sha256", "final_state_sha256"))
        ok = bool(rows_ok and arr_ok and hashes_ok)
        all_ok &= ok
        per[s] = {"per_task": rows_ok, "units": arr_ok, "first_bad": why, "hashes": hashes_ok}
    return {"all_identical": bool(all_ok), "per_seed": per}


# --------------------------------------------------------------------------
# outputs (summary, csv, figures) live in report.py-free functions below
# --------------------------------------------------------------------------

def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    return f"{x:+.{nd}f}"


def write_outputs(r1: dict, r2: dict, scores: dict, rid: dict | None, env: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(r1.get("rows", [])).to_csv(out / "verdict.csv", index=False)
    pd.DataFrame([{"arm": a, "endpoint": ep, "seed": s, "delta_minus_ref": v}
                  for a, eps in r1.get("per_seed", {}).items() for ep, vals in eps.items()
                  for s, v in zip(r1["valid_seeds"], vals)]).to_csv(out / "paired.csv", index=False)
    pd.DataFrame([{k: v for k, v in s_.items() if k != "d"} | {"d": json.dumps(s_["d"])}
                  for s_ in r1.get("dose_steps", [])]).to_csv(out / "dose.csv", index=False)
    pd.DataFrame([{"arm": a, "layer": li, **{k: v for k, v in e[li].items() if k != "rho"}, "level_label": e["label"]}
                  for a, e in r1.get("equilibrium", {}).items() for li in ("l1", "l2")]).to_csv(
        out / "equilibrium.csv", index=False)
    pd.DataFrame([{"arm": a, "layer": li, **{k: v for k, v in e[li].items() if k != "kappa"}, "level_label": e["label"]}
                  for a, e in r1.get("width", {}).items() for li in ("l1", "l2")]).to_csv(out / "width.csv", index=False)
    pd.DataFrame([{"arm": a, **v} for a, v in r1.get("depth", {}).items()]).to_csv(out / "depth.csv", index=False)
    pd.DataFrame(r2.get("rows", [])).to_csv(out / "c51_paired.csv", index=False)
    sec = [{"box": 1, "key": k, "mean": v} for k, v in r1.get("levels", {}).items()] + \
          [{"box": 2, **r} for r in r2.get("secondary", [])]
    pd.DataFrame(sec).to_csv(out / "secondary.csv", index=False)
    pr = [{"item": k, **v} for k, v in scores["labels"].items()] + [{"item": k, **v} for k, v in scores["binary"].items()]
    pd.DataFrame(pr).to_csv(out / "predictions.csv", index=False)
    vj = {"run_id": RUN_ID, "aggregated_at": dt.datetime.now().astimezone().isoformat(), **env,
          "box1": {k: r1.get(k) for k in ("valid_seeds", "invalid_seeds", "applicability", "floor_state", "labels",
                                          "impaired", "t1_lower", "equilibrium", "width", "depth", "dose_steps",
                                          "timing", "t_half", "returns", "task_floor_frac", "levels", "traj",
                                          "g_ref", "Z_ref_mean")},
          "box2": {k: r2.get(k) for k in ("applicability", "labels", "tests", "info", "R_late", "L_late",
                                          "R_fresh_gap_median", "csnp")},
          "ref_identity": rid, "scores": scores,
          "verdict_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "verdict.json").write_text(json.dumps(vj, indent=2, default=str))


# --------------------------------------------------------------------------
# report only: step 6 of the request -- the l2split_rlmnist_0914 number the spec quotes (section 1.3)
# --------------------------------------------------------------------------

def l2split_check() -> dict:
    """A(1), A(31-50), G = A(1) - A(31-50) and rho = 1 - G_arm / G_ref from the committed per_task.csv of
    l2split_rlmnist_0914 (online accuracy in %, seed means; rho on the seed-mean G and per seed)."""
    p = REPO / "results" / "l2split_rlmnist_0914" / "per_task.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    out = {}
    for (act, arm), g in d.groupby(["act", "arm"]):
        per = {}
        for s_, gs in g.groupby("seed"):
            a = gs.set_index("task")["online_acc"].astype(float)
            per[int(s_)] = (float(a.loc[1]) * 100, float(a.loc[31:50].mean()) * 100)
        out[f"{act}|{arm}"] = {"lam": float(g["lam"].iloc[0]), "A1": float(np.mean([v[0] for v in per.values()])),
                               "A31_50": float(np.mean([v[1] for v in per.values()])), "per_seed": per}
    for act in ("R", "LR"):
        ref, l2 = out.get(f"{act}|ref"), out.get(f"{act}|l2")
        if ref and l2:
            Gr = np.mean([a - b for a, b in ref["per_seed"].values()])
            Gl = np.mean([a - b for a, b in l2["per_seed"].values()])
            l2["rho"] = float(1 - Gl / Gr)
            l2["rho_per_seed"] = [float(1 - (l2["per_seed"][s_][0] - l2["per_seed"][s_][1]) /
                                        (ref["per_seed"][s_][0] - ref["per_seed"][s_][1])) for s_ in sorted(ref["per_seed"])]
    prov = REPO / "results" / "l2split_rlmnist_0914" / "runs" / "R_l2rest_s0" / "provenance.json"
    if prov.exists():
        pv = json.loads(prov.read_text())
        out["box"] = {k: pv.get(k) for k in ("lr", "n_tasks", "epochs_per_task", "batch", "steps_per_task", "n_images")}
    return out


# --------------------------------------------------------------------------
# summary.md and figures
# --------------------------------------------------------------------------

def _g(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    return f"{x:.{nd}f}"


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def summary_md(r1: dict, r2: dict, scores: dict, rid: dict | None, env: dict, l2s: dict, notes: str | None) -> str:
    ap = r1.get("applicability", {})
    lab = r1.get("labels", {})
    n = len(r1.get("valid_seeds", []))
    lv = r1.get("levels", {})
    L = [f"# {RUN_ID} — 判定（AdamW の用量反応: 乱数ラベル MNIST × ELU→ELU と 5+1 CIFAR × ReLU）", "",
         "> 自動生成: `analysis/adamw_dose_1010/verdict.py`。spec: `specs/spec_adamw_dose_1010.md`（登録 commit `364d9cd1`）。"
         "数値は `verdict.json`・`verdict.csv`・`paired.csv`・`dose.csv`・`equilibrium.csv`・`width.csv`・`depth.csv`・"
         "`c51_paired.csv`・`secondary.csv`・`predictions.csv` から。図は `figs/`。", "",
         "## 0. 実行したもの", "",
         f"- 集計時の HEAD: `{env.get('head')}`・箱 1 の run の commit: {env.get('run_commits')}",
         f"- 箱 1: shard {env.get('n_shards')}/50・欠損 {len(env.get('missing', []))}・有効 seed {n}"
         f"（無効: {r1.get('invalid_seeds') or 'なし'}）",
         f"- 箱 2: 腕 {sorted(r2.get('info', {}).keys())}・R と L2 Init は committed の記録を再利用"
         f"（照合 {r2.get('applicability', {}).get('reuse')}・最後の難しい課題のバッチ列の sha256 が全腕で一致: "
         f"{r2.get('applicability', {}).get('fresh_batches_same')}）", ""]
    if notes:
        L += ["## 1. 読み（手書き。`results/adamw_dose_1010/notes.md`）", "", notes.strip(), ""]
    L += ["## 2. 箱 1（乱数ラベル MNIST × ELU→ELU）の判定（spec §5.1）", "",
          f"- (A) 有効 seed {ap.get('A_valid_seeds')} ≥ {MIN_SEEDS}: **{ap.get('A_ok')}**",
          f"- (B) ref が主窓で床: {ap.get('B_ref_at_floor')}/{n} seed、ref の ΔE2 "
          f"{_f((ap.get('B_ref_E2') or {}).get('mean'))} [{_f((ap.get('B_ref_E2') or {}).get('lo'))}, "
          f"{_f((ap.get('B_ref_E2') or {}).get('hi'))}] → **{ap.get('B_ok')}**",
          "- (C) 減衰が全 100 課題・6 テンソルで 6000 回掛かり、死んだ画素の W1 が c の反復と bit 一致: " + "、".join(
              f"{a} **{v}**" for a, v in (ap.get("C") or {}).items()),
          "- 主窓の床の状態: " + "、".join(f"{a} {v}" for a, v in (r1.get("floor_state") or {}).items()), "",
          "| 腕 | λ | 登録水準 | ラベル | 3 値 | 主窓 online（seed 平均） | δE1 [区間] | δE2 [区間] | T1_LOWER | IMPAIRED |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    rows = {(r["arm"], r["endpoint"], r["level"]): r for r in r1.get("rows", [])}
    for arm in LAM_ARMS:
        lvl = LEVEL_OF[arm]
        e1, e2 = rows.get((arm, "E1", lvl)), rows.get((arm, "E2", lvl))
        if e1 is None:
            continue
        t1 = r1["t1_lower"][arm]
        L.append(f"| {arm} | {LAMBDA[arm]:g} | {lvl} | **{lab.get(arm)}** | {lab.get(arm + '_three')} | "
                 f"{_g(lv.get(f'{arm}|online_acc|51-100'))} | {_f(e1['mean'], 3)} [{_f(e1['lo'], 3)}, {_f(e1['hi'], 3)}] | "
                 f"{_f(e2['mean'], 3)} [{_f(e2['lo'], 3)}, {_f(e2['hi'], 3)}] | {t1['flag']}（{_f(t1['mean'], 3)}） | "
                 f"{r1['impaired'][arm]['flag']}（{r1['impaired'][arm]['seeds_below']}/{n}） |")
    L += ["", f"- ref の主窓 online: {_g(lv.get('ref|online_acc|51-100'))}、task 1: {_g(lv.get('ref|online_acc|1-1'))}",
          "- 97.5% の腕の 95% のラベル（参考）: " + "、".join(f"{a} {lab.get(a + '_95')}" for a in EFFECTIVE),
          f"- **箱 1 のまとめ: {lab.get('box')}**", "",
          "### 2.1 用量反応 DOSE（spec §5.2・主窓 online の水準・隣り合う段の符号検定・Holm）", "",
          "| 段 | + の seed / n | p（両側） | Holm 棄却 | 向き | 差の中央値 |", "|---|---|---|---|---|---|"]
    for st in r1.get("dose_steps", []):
        L.append(f"| {st['step']} | {st['pos']}/{st['n']} | {st['p']:.4g} | {st['holm_rejected']} | {st['dir']} | "
                 f"{_f(st['median'], 4)} |")
    L += ["", f"**DOSE: {lab.get('dose')}**", "",
          "### 2.2 釣り合い EQUILIBRIUM（spec §5.3）: 回転平衡 ‖w‖* = η√(d/(2(1−c))) と実測の行ノルム", "",
          "実測は主窓の課題終端の行ノルムの unit 中央値を窓で平均し seed で平均したもの。比は seed ごとの比の幾何平均、区間は log₂ 比の 95% t 区間。",
          "", "| 腕 | 層 | d | ‖w‖*（seed 平均） | 実測 | 比 | log₂ 比 [95%] | 層のラベル | d = 784 での比 | 水準のラベル |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, e in r1.get("equilibrium", {}).items():
        for li in ("l1", "l2"):
            x = e[li]
            L.append(f"| {arm} | {li} | {'d_eff' if li == 'l1' else 100} | {_g(x['w_star_mean'])} | {_g(x['obs_mean'])} | "
                     f"{_g(x['ratio_geomean'], 2)} | {_f(x['log2_mean'], 2)} [{_f(x['lo'], 2)}, {_f(x['hi'], 2)}] | "
                     f"{x['label']} | {_g(x['ratio784_geomean'], 2)} | {e['label'] if li == 'l1' else ''} |")
    L += ["", "（wd1e-3・wd1e-2 は報告のみ。ラベルの対象は効く水準 wd3e-2・wd1e-1）", "",
          "### 2.3 幅 WIDTH（spec §5.4）: κ = 主窓の幅の傾き / 同じ seed の ref の生きている間（t1–10）の傾き", "",
          "| 腕 | 層 | κ の平均 [95%] | 層のラベル | 主窓の幅 s（seed 平均） | 水準のラベル |", "|---|---|---|---|---|---|"]
    for arm, w in r1.get("width", {}).items():
        for li in ("l1", "l2"):
            x = w[li]
            L.append(f"| {arm} | {li} | {_f(x['kappa_mean'], 3)} [{_f(x['lo'], 3)}, {_f(x['hi'], 3)}] | {x['label']} | "
                     f"{_g(x['s_main_mean'])} | {w['label'] if li == 'l1' else ''} |")
    gr = r1.get("g_ref", {})
    if gr:
        L.append(f"\nref の生きている間の傾き（seed 平均）: σ1 {_g(np.mean([v[1] for v in gr.values()]))}/課題、"
                 f"σ2 {_g(np.mean([v[2] for v in gr.values()]))}/課題")
    L += ["", "幅 s（unit 中央値・seed 平均）の時系列:", "", "| 腕 | 層 | t1 | t2 | t3 | t5 | t10 | t20 | t50 | t100 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tr = r1.get("traj", {})
    for arm in ARMS1:
        for key in ("sigma_l1", "sigma_l2", "row_norm_l1", "row_norm_l2", "d_l2", "dtrain_zero_l2", "online_acc"):
            vals = [tr.get(f"{arm}|{key}|{t}") for t in (1, 2, 3, 5, 10, 20, 50, 100)]
            if any(v is not None for v in vals):
                L.append(f"| {arm} | {key} | " + " | ".join(_g(v) for v in vals) + " |")
    L += ["", "### 2.4 深さ DEPTH（spec §5.5）と床の距離", "",
          "| 腕 | DEPTH | ζ = Z/Z_ref の平均 [95%] | Z（主窓の厳密 0 の対の割合） | 主窓 σ2 | 主窓 d2 | 16.64/σ2 + d2 | Φ(−(16.64/σ2 + d2)) |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS1:
        s2 = lv.get(f"{arm}|med_sigma_l2|51-100")
        d2 = lv.get(f"{arm}|med_d_l2|51-100")
        dist = (16.64 / s2 + d2) if (s2 and d2 is not None and s2 > 0) else None
        dp = r1.get("depth", {}).get(arm, {})
        L.append(f"| {arm} | {dp.get('label', '—')} | {_g(dp.get('zeta_mean'))} [{_g(dp.get('lo'))}, {_g(dp.get('hi'))}] | "
                 f"{_g(dp.get('Z_mean', r1.get('Z_ref_mean') if arm == 'ref' else None))} | {_g(s2)} | {_g(d2)} | "
                 f"{_g(dist)} | {_g(_phi(-dist) if dist is not None else None, 4)} |")
    L += ["", "（σ2・d2 は課題終端の unit 中央値の主窓平均を seed で平均。Φ の列は spec §1.1 (a) の正規近似の厳密 0 の割合）", "",
          "### 2.5 報告のみ: 崩壊の時刻 T½・戻り・主窓で床にいた課題の割合", "",
          "| 腕 | T½（seed） | ref と比べた時刻 | 戻りの事象（seed ごと） | 床にいた課題の割合 |", "|---|---|---|---|---|"]
    for arm in ARMS1:
        th = [r1.get("t_half", {}).get(f"{arm}|{s_}") for s_ in r1.get("valid_seeds", [])]
        rt = [r1.get("returns", {}).get(f"{arm}|{s_}") for s_ in r1.get("valid_seeds", [])]
        tm = r1.get("timing", {}).get(arm, {})
        L.append(f"| {arm} | {', '.join(str(x) if x is not None else '—' for x in th)} | "
                 f"{tm.get('label', '')} ({tm.get('earlier', '')}/{tm.get('later', '')}/{tm.get('ties', '')}) | "
                 f"{', '.join(str(x) for x in rt)} | {_g(r1.get('task_floor_frac', {}).get(arm))} |")
    L += ["", "### 2.6 ref の全 100 課題の bit 一致（l2cap_ee_0917 の ref と。報告のみ）", ""]
    L.append("- 照合しなかった" if rid is None else f"- 全 seed で一致: **{rid['all_identical']}**  "
             + "; ".join(f"s{s_}: {v}" for s_, v in rid["per_seed"].items() if not (isinstance(v, dict) and all(
                 v.get(k) for k in ("per_task", "units", "hashes")))))
    # box 2
    a2 = r2.get("applicability", {})
    lab2 = r2.get("labels", {})
    L += ["", "## 3. 箱 2（5+1 CIFAR × MLP × ReLU）の判定（spec §5.6）", "",
          f"- (A) 欠け: {a2.get('incomplete') or 'なし'}、再利用の照合 (a)(b)(c): {a2.get('reuse')}、"
          f"バッチ列の sha256 一致: {a2.get('fresh_batches_same')}",
          f"- (B) R の fresh gap {_f((a2.get('B_R_fresh_gap') or {}).get('mean'))} "
          f"[{_f((a2.get('B_R_fresh_gap') or {}).get('lo'))}, {_f((a2.get('B_R_fresh_gap') or {}).get('hi'))}] → "
          f"**{a2.get('B_ok')}**",
          f"- (C) wd_steps = 780 が全課題・全テンソル: {a2.get('C')}", "",
          "| 腕 | λ | ラベル | WORSE | IMPAIRED（早期窓 W−R の 95% 区間） | a = W−R: 中央値 [v(2), v(9)]・+/n・p | "
          "c = W−M: 中央値 [v(2), v(9)]・+/n・p | 戻りの割合 ρ の中央値 | 後期窓 W の平均 | fresh gap: self / 共通（中央値） |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tests = r2.get("tests", {})
    for a in ARMS2:
        if a not in r2.get("info", {}):
            L.append(f"| {a} | {LAMBDA2[a]:g} | **{lab2.get(a)}** | | | | | | | |")
            continue
        ta, tc = tests[f"a|{a}"], tests[f"c|{a}"]
        inf = r2["info"][a]
        im = inf["impaired"]
        L.append(f"| {a} | {LAMBDA2[a]:g} | **{lab2.get(a)}** | {lab2.get(a + '_worse')} | {im['flag']}（{_f(im['mean'], 3)} "
                 f"[{_f(im['lo'], 3)}, {_f(im['hi'], 3)}]） | {_f(ta['median'], 3)} [{_f(ta['ord_lo'], 3)}, {_f(ta['ord_hi'], 3)}]・"
                 f"{ta['pos']}/{ta['n']}・{ta['p']:.3g} | {_f(tc['median'], 3)} [{_f(tc['ord_lo'], 3)}, {_f(tc['ord_hi'], 3)}]・"
                 f"{tc['pos']}/{tc['n']}・{tc['p']:.3g} | {_g(float(np.median(inf['rho'])), 3)} | "
                 f"{_g(float(np.mean(inf['W_late'])))} | {_f(inf.get('fresh_gap_self_median'), 3)} / "
                 f"{_f(inf.get('fresh_gap_common_median'), 3)} |")
    if r2.get("R_late"):
        L.append(f"\nR の後期窓の平均 {_g(float(np.mean(r2['R_late'])))}・L2 Init {_g(float(np.mean(r2['L_late'])))}・"
                 f"中点 {_g(float(np.mean([(a_ + b_) / 2 for a_, b_ in zip(r2['R_late'], r2['L_late'])])))}・"
                 f"R の fresh gap の中央値 {_f(r2.get('R_fresh_gap_median'), 3)}")
    L += ["", f"**箱 2 のまとめ: {lab2.get('box')}**", "", "報告のみ（後期窓の課題終端の seed 中央値と、R との seed 内の差・正確な符号検定）:", "",
          "| 腕 | 量 | W の中央値 | R の中央値 | 差の中央値 | +/n | p |", "|---|---|---|---|---|---|---|"]
    for r in r2.get("secondary", []):
        L.append(f"| {r['arm']} | {r['quantity']} | {_g(r.get('W_median'), 4)} | {_g(r.get('R_median'), 4)} | "
                 f"{_f(r.get('diff_median'), 4)} | {r.get('pos', '')}/{r.get('n', '')} | "
                 f"{'' if r.get('p') is None else format(r['p'], '.3g')} |")
    cs = r2.get("csnp")
    L += ["", "C-snp（本走の後・較正 seed 100–109・adamw10 − S&P ε 1e−3 σ 1e−4 の後期窓の差）: " +
          ("走らせていない" if not cs else f"n {cs['n']}・中央値 {_f(cs['median'], 4)} [{_f(cs['ord_lo'], 4)}, "
           f"{_f(cs['ord_hi'], 4)}]・adamw10 {_g(cs['adamw10_late_mean'])}・S&P {_g(cs['snp_late_mean'])}")]
    # predictions
    L += ["", "## 4. 予測の採点（spec §7.3。Claude の列。Issa と Fable の列は登録時に空欄）", "",
          "| 項目 | 実現 | p(実現) | 最頻 | 的中 | Brier |", "|---|---|---|---|---|---|"]
    for k, v in scores["labels"].items():
        if "unscored" in v:
            L.append(f"| {k} | 未採点（{v['unscored']}） | | | | |")
        else:
            L.append(f"| {k} | {v['realized']}（桶 {v['bucket']}） | {v['p_realized']:.2f} | {v['mode']} | "
                     f"{'○' if v['mode_hit'] else '×'} | {v['brier']:.3f} |")
    L += ["", "| 二値の項目 | p | 実現 | Brier |", "|---|---|---|---|"]
    for k, v in scores["binary"].items():
        L.append(f"| {k} | {v['p']:.2f} | {v['true'] if v['true'] is not None else '未採点' + ('（' + v['unscored'] + '）' if v.get('unscored') else '')} | "
                 f"{'' if v['brier'] is None else format(v['brier'], '.3f')} |")
    bl = [v["brier"] for v in scores["labels"].values() if v.get("brier") is not None]
    bb = [v["brier"] for v in scores["binary"].values() if v.get("brier") is not None]
    L.append(f"\nラベルの項目の Brier 平均 {_g(float(np.mean(bl)) if bl else None)}（{len(bl)} 項目）・"
             f"二値の項目の Brier 平均 {_g(float(np.mean(bb)) if bb else None)}（{len(bb)} 項目）")
    # l2split
    if l2s:
        r_ref, r_l2 = l2s.get("R|ref", {}), l2s.get("R|l2", {})
        bx = l2s.get("box", {})
        L += ["", "## 5. 追加の確認: 「RL-MNIST の coupled L2 が ReLU を 98.9% 救った」（spec §1.3 が引く l2split_rlmnist_0914）", "",
              f"`results/l2split_rlmnist_0914/per_task.csv`（git の中）の online を seed 0–9 で平均して計算し直した。箱は "
              f"lr {bx.get('lr')}・{bx.get('n_tasks')} 課題・{bx.get('epochs_per_task')} epoch × batch {bx.get('batch')}"
              f"（{bx.get('steps_per_task')} 更新/課題）・{bx.get('n_images')} 枚。L2 は損失に λ‖θ‖²（勾配に 2λθ）を足して Adam に通す coupled（λ = {r_l2.get('lam')}）。", "",
              "| 腕 | A(1) | A(31–50) | G = A(1) − A(31–50) | ρ = 1 − G/G_ref |", "|---|---|---|---|---|"]
        for key in ("R|ref", "R|l2", "LR|ref", "LR|l2"):
            x = l2s.get(key)
            if x:
                G = x["A1"] - x["A31_50"]
                L.append(f"| {key} | {x['A1']:.2f} | {x['A31_50']:.2f} | {G:+.2f} | {_g(x.get('rho'), 4) if 'rho' in x else '—'} |")
        if "rho" in r_l2:
            L.append(f"\nReLU の ρ は seed ごとに {min(r_l2['rho_per_seed']):.4f}〜{max(r_l2['rho_per_seed']):.4f}。"
                     f"spec の「98.9%（A(31–50) 94.0 対 ref 12.6）」は ρ = {r_l2['rho']:.4f}・A(31–50) {r_l2['A31_50']:.2f} 対 "
                     f"{r_ref.get('A31_50', float('nan')):.2f} と一致する。98.9% は時間劣化 G の除去率で、L2 の腕は task 1 から "
                     f"{r_l2['A1']:.2f}（ref {r_ref.get('A1', float('nan')):.2f}）と低い所から始まる。")
    L += ["", "## 6. 読みの上限（spec §10）", "",
          "- 2 箱・この λ・この予算の範囲の結果。AdamW が可塑性の喪失全体を防ぐとは言わない。SiLU・GELU は測っていない。",
          "- online は学ぶ速さ（課題の中の更新前の当たり率の平均）。RESCUED は相対の規則（自分の task 1 からの低下の差）。",
          "- 箱 2 は decoupled WD・λ ∈ {5, 10} の結果で、WD 一般や coupled L2 には広げない。中心化との比較は範囲外。",
          "- 釣り合いの式は既知（Kosson 2024）。戻りの事象は報告のみの格。"]
    return "\n".join(L) + "\n"


def figures(r1: dict, r2: dict, data2: dict | None, out: Path) -> list[str]:
    """Static PNGs (report only).  Ordinal dose: ref neutral gray (dashed), the four lambdas one blue ramp light ->
    dark (validated as an ordinal ramp: monotone lightness, light end >= 2:1 on the surface)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    col = {"ref": "#8a8984", "wd1e-3": "#86b6ef", "wd1e-2": "#3987e5", "wd3e-2": "#1c5cab", "wd1e-1": "#0d366b"}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "legend.frameon": False})
    names = []
    ser = r1.get("series", {})
    if ser:
        t = np.arange(1, N_TASKS + 1)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for arm in ARMS1:
            if arm in ser:
                y = np.array(ser[arm]["online_acc"])
                ax.plot(t, y, color=col[arm], lw=2, ls="--" if arm == "ref" else "-",
                        label=f"{arm} (λ={LAMBDA[arm]:g})")
                ax.annotate(arm, (t[-1], y[-1]), xytext=(4, 0), textcoords="offset points", color=ink, fontsize=8,
                            va="center")
        ax.axvspan(MAIN_WIN[0], MAIN_WIN[1], color="#f0efec", zorder=0)
        ax.set_xlabel("task")
        ax.set_ylabel("online accuracy (seed mean)")
        ax.set_title("Box 1: Random Label MNIST, ELU→ELU, AdamW on all tensors (shaded: main window 51–100)",
                     fontsize=10, color=ink)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / "box1_online.png", dpi=150)
        plt.close(fig)
        names.append("figs/box1_online.png")
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        panels = (("sigma_l1", "width s₁ (unit median of sd z₁)"), ("sigma_l2", "width s₂ (unit median of sd z₂)"),
                  ("row_norm_l1", "‖w₁‖ row norm (unit median)"), ("row_norm_l2", "‖w₂‖ row norm (unit median)"))
        for ax, (key, title) in zip(axs.flat, panels):
            for arm in ARMS1:
                if arm in ser and key in ser[arm]:
                    ax.plot(t, ser[arm][key], color=col[arm], lw=2, ls="--" if arm == "ref" else "-", label=arm)
                    if key.startswith("row_norm") and LAMBDA[arm] > 0:
                        d = 620.0 if key.endswith("l1") else 100.0
                        ax.axhline(w_star(LR1, LAMBDA[arm], d), color=col[arm], lw=1, ls=":")
            ax.set_yscale("log")
            ax.set_title(title, fontsize=9, color=ink)
            ax.set_xlabel("task")
        axs[0, 0].legend(fontsize=8)
        fig.suptitle("Box 1: widths and row norms per task (seed mean); dotted: η√(d/2(1−c)) with d = 620 (layer 1), 100",
                     fontsize=10, color=ink)
        fig.tight_layout()
        fig.savefig(figs / "box1_width_norm.png", dpi=150)
        plt.close(fig)
        names.append("figs/box1_width_norm.png")
        fig, ax = plt.subplots(figsize=(8, 4))
        for arm in ARMS1:
            if arm in ser and "zero_pairs_l2" in ser[arm]:
                ax.plot(t, ser[arm]["zero_pairs_l2"], color=col[arm], lw=2, ls="--" if arm == "ref" else "-", label=arm)
        ax.set_xlabel("task")
        ax.set_ylabel("fraction of (image, unit) pairs, exact-0 training derivative, layer 2")
        ax.set_title("Box 1: second layer at the float32 floor (seed mean)", fontsize=10, color=ink)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / "box1_zero_pairs.png", dpi=150)
        plt.close(fig)
        names.append("figs/box1_zero_pairs.png")
    if data2 and data2.get("arms"):
        col2 = {"R": "#8a8984", "L2 Init 1e-2": "#1baf7a", "adamw5": "#3987e5", "adamw10": "#0d366b"}
        fig, ax = plt.subplots(figsize=(8, 4.2))
        series2 = {"R": data2["R"]["per_task"], "L2 Init 1e-2": data2["L"]["per_task"],
                   **{a: v["per_task"] for a, v in data2["arms"].items()}}
        for name, pt in series2.items():
            h = pt[pt["task"] % 2 == 1].groupby("task")["online_acc"].mean()
            ax.plot(h.index, h.values, color=col2[name], lw=2, ls="--" if name == "R" else "-", marker="o", ms=4,
                    label=name)
            ax.annotate(name, (h.index[-1], h.values[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                        color=ink, va="center")
        ax.axvspan(21, 29, color="#f0efec", zorder=0)
        ax.set_xlabel("task (hard tasks only)")
        ax.set_ylabel("online accuracy (seed mean)")
        ax.set_title("Box 2: 5+1 CIFAR × MLP × ReLU (shaded: late window, hard 21–29)", fontsize=10, color=ink)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(figs / "box2_online.png", dpi=150)
        plt.close(fig)
        names.append("figs/box2_online.png")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src1", default=str(OUT / "mnist" / "runs"))
    ap.add_argument("--src2", default=str(OUT / "c51"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-ref-identity", action="store_true")
    args = ap.parse_args()
    shards, missing = load1(Path(args.src1))
    r1 = analyze1(shards)
    r1["missing"] = missing
    data2 = load2(Path(args.src2))
    r2 = analyze2(data2)
    rid = None if args.no_ref_identity else ref_identity(shards)
    scores = score_predictions(r1, r2, rid)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = {"head": head, "run_commits": sorted({sh["prov"].get("git_hash") for sh in shards.values()}),
           "n_shards": len(shards), "missing": missing}
    out = Path(args.out)
    write_outputs(r1, r2, scores, rid, env, out)
    l2s = l2split_check()
    notes_p = out / "notes.md"
    notes = notes_p.read_text() if notes_p.exists() else None
    (out / "summary.md").write_text(summary_md(r1, r2, scores, rid, env, l2s, notes))
    figs = figures(r1, r2, data2, out)
    vj = json.loads((out / "verdict.json").read_text())
    vj["l2split_check"] = {k: v for k, v in l2s.items()}
    vj["figures"] = figs
    vj["box1"]["series"] = r1.get("series")
    (out / "verdict.json").write_text(json.dumps(vj, indent=2, default=str))
    print(json.dumps({"box1": r1["labels"], "box2": r2["labels"]}, indent=1, default=str))


if __name__ == "__main__":
    main()

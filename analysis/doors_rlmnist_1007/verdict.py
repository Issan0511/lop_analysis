#!/usr/bin/env python3
"""Verdict for doors_rlmnist_1007 (specs/spec_doors_rlmnist_1007.md sections 4-5, 7.3).

    /home/issan/Projects/claude/proj_004_drift/.venv/bin/python analysis/doors_rlmnist_1007/verdict.py \
        [--src results/doors_rlmnist_1007/runs] [--out results/doors_rlmnist_1007]

Reads the 40 shards, applies the registered endpoints and labels, scores the registered predictions and
writes {paired.csv, verdict.csv, secondary.csv, timing.csv, summary.md, provenance.json}.
analyze() is pure (shards in, results out) so that checks.py S8 can run it on synthetic shards.

Endpoints (spec 4.1), per seed, each minus THAT ARM'S OWN task-1 value (the doors are on from task 1,
so task 1 is not shared across arms):
  E1 = online accuracy, arithmetic mean over the window's tasks;
  E2 = the second layer's derivative as training uses it (autograd through the float32 ELU, exactly 0
       below z = -16.64): image mean -> arithmetic mean over all 100 units -> mean over the window's
       task ends.
The arm effect is the within-seed difference to ref.  Floor (spec 4.1): AT_FLOOR when the window online
accuracy is <= F + 3 s / sqrt(n), F and s (ddof 1) the mean and SD over the window's n tasks of the best
constant predictor's accuracy.  The Student-t machinery (paired, t_quantile) is mucap_el_0916's, unchanged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("_mucap_el_verdict", REPO / "analysis" / "mucap_el_0916" / "verdict.py")
ELV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ELV)
paired, t_quantile, t_cdf, task_ends, unit_mean = ELV.paired, ELV.t_quantile, ELV.t_cdf, ELV.task_ends, ELV.unit_mean

RUN_ID = "doors_rlmnist_1007"
ARMS = ("ref", "C", "H", "CH")
DOOR_ARMS = ARMS[1:]
MAIN_ARM = "H"                              # spec 5.3: the primary comparison is H - ref
DOORS = {"C": ("c",), "H": ("h",), "CH": ("c", "h")}
LEVEL_OF = {"C": 0.95, "H": 0.975, "CH": 0.95}   # spec 5.3: each arm's registered level
LADDER = ("C", "H", "CH")                   # spec 5.4: the fixed order
SEEDS = tuple(range(10))
N_TASKS = 100
SPT = 6000                                  # updates per task: the task-end diagnostic step
MAIN_WIN = (51, 100)                        # spec 4.1
FORM_WINS = ((2, 10), (11, 50))             # spec 4.1, reported only
DOOR_WIN = (1, 100)                         # spec 5.2 (C): the door must have worked in EVERY task
MIN_SEEDS = 8                               # spec 5.2 (A)
REF_FLOOR_FRAC = 0.8                        # spec 5.2 (B)
FLOOR_K = 3.0                               # spec 4.1: F + 3 s / sqrt(n)
CHANCE = 0.1                                # 10 uniform labels: the fit fraction's zero
HALF = 0.5                                  # spec 4.2-1
SIGN_ALPHA = 0.05                           # spec 4.2-1 (report only)
EARLY = (2, 5)                              # spec 5.2 IMPAIRED
MAIN_LEVEL = 0.975                          # two-sided, two main endpoints (Bonferroni)
COMP_LEVEL = 0.95                           # C, CH, and H for reference
STREAM_KEYS = ("init_sha256", "subset_idx_sha256", "labels_sha256", "batch_sha256")
E2_KEY = "dtrain_mean_l2"
ZERO_Z32 = math.log(2.0 ** -24)             # spec 4.2-4
UNIT_Q = {"g2_exp": "gate_mean_l2", "dtrain0_l2": "dtrain_zero_l2", "dtrain_l1": "dtrain_mean_l1",
          "dtrain0_l1": "dtrain_zero_l1", "pplus_l2": "pplus_l2", "zbar_l2": "zbar_l2", "zbar_l1": "zbar_l1",
          "d_l2": "d_l2", "sigma_l2": "sigma_l2", "sigma_l1": "sigma_l1", "b2": "b2", "b1": "b1",
          "row_norm_l2": "row_norm_l2", "row_norm_l1": "row_norm_l1", "q_l2": "q_l2"}
SCALARS = ("mu2_norm", "mu2_raw_norm", "mu2_proj_sd")
LABELS6 = ("RESCUED", "RESCUED_FUNCTION_ONLY", "ALIVE_UNRESOLVED", "SPLIT", "COLLAPSED", "OTHER")
N_LABELS = ("NEED_C", "NEED_H", "NEED_CH", "NONE_RESCUED")


def shard_name(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


# --------------------------------------------------------------------------
# per shard
# --------------------------------------------------------------------------

def online(pt: pd.DataFrame) -> pd.Series:
    return pt.set_index("task")["online_acc"].astype(float)


def e1_delta(pt: pd.DataFrame, win: tuple[int, int]) -> float:
    a = online(pt)
    return float(np.mean([a.loc[t] for t in range(win[0], win[1] + 1)])) - float(a.loc[1])


def unit_window_delta(u: dict, key: str, win: tuple[int, int]) -> tuple[float, int]:
    ends = task_ends(u)
    base, nans = unit_mean(u[key][ends[1]])
    vals = []
    for t in range(win[0], win[1] + 1):
        m, k = unit_mean(u[key][ends[t]])
        vals.append(m)
        nans += k
    return float(np.mean(vals)) - base, nans


def unit_window_level(u: dict, key: str, win: tuple[int, int]) -> float:
    ends = task_ends(u)
    return float(np.mean([unit_mean(u[key][ends[t]])[0] for t in range(win[0], win[1] + 1)]))


def scalar_window(u: dict, key: str, win: tuple[int, int]) -> float:
    ends = task_ends(u)
    return float(np.mean([float(np.asarray(u[key][ends[t]]).ravel()[0]) for t in range(win[0], win[1] + 1)]))


def cos_w2_window(u: dict, win: tuple[int, int]) -> float:
    """cos(w2_i, e2) = q_l2 / row_norm_l2 per unit (e2 = the current mean direction of layer 2's input),
    unit mean, then the window mean (spec 4.2-3)."""
    ends = task_ends(u)
    vals = []
    for t in range(win[0], win[1] + 1):
        q = np.asarray(u["q_l2"][ends[t]], float)
        r = np.asarray(u["row_norm_l2"][ends[t]], float)
        c = np.where(r > 0, q / np.where(r > 0, r, 1.0), np.nan)
        vals.append(unit_mean(c)[0])
    return float(np.mean(vals))


def floor_call(pt: pd.DataFrame, win: tuple[int, int]) -> dict:
    p = pt.set_index("task")
    rows = p.loc[win[0]:win[1]]
    f = rows["major_frac"].astype(float).to_numpy()
    n = len(f)
    F, s = float(f.mean()), float(f.std(ddof=1))
    thr = F + FLOOR_K * s / math.sqrt(n)
    level = float(rows["online_acc"].astype(float).mean())
    return {"F": F, "s": s, "thr": thr, "level": level, "at_floor": level <= thr}


def fit(pt: pd.DataFrame) -> pd.Series:
    a = online(pt)
    return (a - CHANCE) / (a.loc[1] - CHANCE)


def t_half(pt: pd.DataFrame) -> int | None:
    """First task >= 2 whose fit fraction (against the arm's own task 1) is below HALF; None if none."""
    f = fit(pt)
    for t in range(2, N_TASKS + 1):
        if f.loc[t] < HALF:
            return t
    return None


def t_cross(u: dict) -> int | None:
    """First task whose end-point unit-mean zbar2 is below ln 2^-24 (report only); None if never."""
    if "zbar_l2" not in u:
        return None
    ends = task_ends(u)
    for t in range(1, N_TASKS + 1):
        if unit_mean(u["zbar_l2"][ends[t]])[0] < ZERO_Z32:
            return t
    return None


def fit_early(pt: pd.DataFrame) -> float:
    f = fit(pt)
    return float(np.mean([f.loc[t] for t in range(EARLY[0], EARLY[1] + 1)]))


def sign_test(k: int, n: int) -> float:
    """Two-sided exact binomial(n, 1/2) p for the larger count k."""
    if n == 0:
        return 1.0
    k = max(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def compare_times(ta: int | None, tr: int | None) -> str:
    """The arm's T_half against ref's in the same seed; a censored time is later than any observed one."""
    if ta is None and tr is None:
        return "tie"
    if ta is None:
        return "later"
    if tr is None:
        return "earlier"
    return "earlier" if ta < tr else "later" if ta > tr else "tie"


def door_worked(pt: pd.DataFrame, door: str) -> bool:
    """Spec 5.2 (C) for one door in one run: the record says the door worked in every task of DOOR_WIN.
    c: the fed inputs' mean was within the rounding bound (c_ok == 1).  h: m changed and was nonzero in at
    least one update of the task, on both layers."""
    p = pt.set_index("task")
    w = p.loc[DOOR_WIN[0]:DOOR_WIN[1]]
    if len(w) != DOOR_WIN[1] - DOOR_WIN[0] + 1:
        return False
    if door == "c":
        return "c_ok" in w and bool((w["c_ok"] == 1).all())
    if door == "h":
        return "h_upd_l1" in w and "h_upd_l2" in w and bool(((w["h_upd_l1"] > 0) & (w["h_upd_l2"] > 0)).all())
    raise ValueError(door)


# --------------------------------------------------------------------------
# loading and validity
# --------------------------------------------------------------------------

def load(src: Path) -> tuple[dict, list]:
    shards, missing = {}, []
    for arm in ARMS:
        for seed in SEEDS:
            d = src / shard_name(arm, seed)
            if not (d / "provenance.json").exists():
                missing.append(shard_name(arm, seed))
                continue
            pre = f"s{seed}_"
            with np.load(d / "units.npz") as z:
                units = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
            shards[(arm, seed)] = {"units": units, "per_task": pd.read_csv(d / "per_task.csv"),
                                   "prov": json.loads((d / "provenance.json").read_text())}
    return shards, missing


def seed_validity(shards: dict, seed: int) -> str | None:
    """None if the seed is valid for spec 5.2 (A), else the reason."""
    infos = {}
    for arm in ARMS:
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


# --------------------------------------------------------------------------

def analyze(shards: dict) -> dict:
    invalid = {s: r for s in SEEDS if (r := seed_validity(shards, s)) is not None}
    valid = [s for s in SEEDS if s not in invalid]
    res = {"valid_seeds": valid, "invalid_seeds": invalid, "rows": [], "secondary": [], "timing": [],
           "applicability": {}}

    D = {}
    for arm in ARMS:
        for s in valid:
            sh = shards[(arm, s)]
            for win in (MAIN_WIN, *FORM_WINS):
                D[(arm, s, "E1", win)] = (e1_delta(sh["per_task"], win), 0)
                D[(arm, s, "E2", win)] = unit_window_delta(sh["units"], E2_KEY, win)

    def deltas(arm, ep, win=MAIN_WIN):
        return np.array([D[(arm, s, ep, win)][0] for s in valid])

    def diffs(arm, ep, win=MAIN_WIN):
        return deltas(arm, ep, win) - deltas("ref", ep, win)

    floors = {(arm, s): floor_call(shards[(arm, s)]["per_task"], MAIN_WIN) for arm in ARMS for s in valid}
    state = {}
    for arm in ARMS:
        at = [floors[(arm, s)]["at_floor"] for s in valid]
        state[arm] = "FLOORED" if at and all(at) else "ALIVE" if at and not any(at) else "SPLIT"

    # (A) (B) (C)
    A_ok = len(valid) >= MIN_SEEDS
    ref_at = sum(floors[("ref", s)]["at_floor"] for s in valid)
    B_E2 = paired(deltas("ref", "E2"), 0.95)
    B_ok = A_ok and ref_at >= math.ceil(REF_FLOOR_FRAC * len(valid)) and B_E2["hi"] < 0
    C = {}
    C_detail = {}
    for arm in DOOR_ARMS:
        ok = True
        bad = []
        for s in valid:
            for door in DOORS[arm]:
                w = door_worked(shards[(arm, s)]["per_task"], door)
                ok &= w
                if not w:
                    bad.append(f"s{s}:{door}")
        C[arm] = bool(ok and valid)
        C_detail[arm] = bad
    res["applicability"] = {"A_valid_seeds": len(valid), "A_ok": A_ok, "B_ref_at_floor": ref_at,
                            "B_ref_E2": {k: B_E2[k] for k in ("mean", "lo", "hi")}, "B_ok": B_ok, "C": C,
                            "C_failed": C_detail}
    res["floor_state"] = state
    res["floors"] = {f"{a}|{s}": v for (a, s), v in floors.items()}

    impaired = {}
    for arm in DOOR_ARMS:
        hits = 0
        for s in valid:
            f_ref = fit_early(shards[("ref", s)]["per_task"])
            f_arm = fit_early(shards[(arm, s)]["per_task"])
            hits += f_arm < f_ref - (1.0 - f_ref)
        impaired[arm] = {"seeds_below": hits, "flag": hits > len(valid) / 2}
    res["impaired"] = impaired

    def label(arm, st):
        if not A_ok or not C[arm]:
            return "INAPPLICABLE"
        if not B_ok:
            return "NOT_REPRODUCED"
        if state[arm] == "FLOORED":
            return "COLLAPSED"
        if state[arm] == "SPLIT":
            return "SPLIT"
        if st["E1"]["sign"] == "+":
            return "RESCUED" if st["E2"]["sign"] == "+" else "RESCUED_FUNCTION_ONLY"
        return "ALIVE_UNRESOLVED"

    labels = {}
    for arm in DOOR_ARMS:
        for level in sorted({MAIN_LEVEL, COMP_LEVEL, *LEVEL_OF.values()}, reverse=True):
            st = {ep: paired(diffs(arm, ep), level) for ep in ("E1", "E2")}
            lab = label(arm, st)
            labels[(arm, level)] = lab
            for ep, x in st.items():
                res["rows"].append({"arm": arm, "endpoint": ep, "window": f"{MAIN_WIN[0]}-{MAIN_WIN[1]}",
                                    "level": level, **{k: x[k] for k in ("n", "mean", "sd", "lo", "hi",
                                                                         "sign", "degenerate")},
                                    "floor_state": state[arm], "label": lab,
                                    "role": "main" if (arm == MAIN_ARM and level == MAIN_LEVEL) else
                                    ("registered" if level == LEVEL_OF[arm] else "report")})
    reg = {arm: labels[(arm, LEVEL_OF[arm])] for arm in DOOR_ARMS}

    # spec 5.4: the ladder, in the fixed order, on each arm's registered label
    if not A_ok:
        n_label = "INAPPLICABLE"
    elif not B_ok:
        n_label = "NOT_REPRODUCED"
    else:
        n_label = next((f"NEED_{a}" for a in LADDER if reg[a] == "RESCUED"), "NONE_RESCUED")

    # timing (spec 4.2-1, report only)
    th = {(arm, s): t_half(shards[(arm, s)]["per_task"]) for arm in ARMS for s in valid}
    timing = {}
    for arm in DOOR_ARMS:
        cmp_ = [compare_times(th[(arm, s)], th[("ref", s)]) for s in valid]
        e, l_ = cmp_.count("earlier"), cmp_.count("later")
        p = sign_test(max(e, l_), e + l_)
        lab = ("EARLIER" if e > l_ else "LATER") if (e + l_ > 0 and p < SIGN_ALPHA) else "NO_TIMING_DIFF"
        timing[arm] = {"label": lab, "earlier": e, "later": l_, "ties": cmp_.count("tie"), "p": p}
        for s, c in zip(valid, cmp_):
            res["timing"].append({"arm": arm, "seed": s, "t_half_arm": th[(arm, s)],
                                  "t_half_ref": th[("ref", s)], "compare": c})
    res["timing_labels"] = timing
    res["t_cross"] = {f"{a}|{s}": t_cross(shards[(a, s)]["units"]) for a in ARMS for s in valid}
    res["t_half"] = {f"{a}|{s}": v for (a, s), v in th.items()}

    res["labels"] = {"main": labels[(MAIN_ARM, MAIN_LEVEL)], "H_95": labels[("H", COMP_LEVEL)],
                     "C": reg["C"], "CH": reg["CH"], "N": n_label,
                     **{f"timing_{a}": timing[a]["label"] for a in DOOR_ARMS}}
    res["per_seed"] = {arm: {ep: diffs(arm, ep).tolist() for ep in ("E1", "E2")} for arm in DOOR_ARMS}
    res["ref_delta"] = {ep: deltas("ref", ep).tolist() for ep in ("E1", "E2")}
    res["arm_delta"] = {arm: {ep: deltas(arm, ep).tolist() for ep in ("E1", "E2")} for arm in ARMS}
    res["interaction"] = {ep: paired(diffs("CH", ep) - diffs("C", ep) - diffs("H", ep), COMP_LEVEL)
                          for ep in ("E1", "E2")}
    res["ch_minus_h"] = {ep: paired(diffs("CH", ep) - diffs("H", ep), COMP_LEVEL) for ep in ("E1", "E2")}
    # spec 4.2-2: the transport, arm minus ref, of the change from task 1 of zbar2 and b2
    res["transport"] = {}
    for arm in DOOR_ARMS:
        for key in ("zbar_l2", "b2"):
            d = np.array([unit_window_delta(shards[(arm, s)]["units"], key, MAIN_WIN)[0]
                          - unit_window_delta(shards[("ref", s)]["units"], key, MAIN_WIN)[0] for s in valid])
            res["transport"][f"{arm}|{key}"] = paired(d, COMP_LEVEL)
    res["arm_b2_delta"] = {arm: float(np.mean([unit_window_delta(shards[(arm, s)]["units"], "b2", MAIN_WIN)[0]
                                               for s in valid])) if valid else float("nan") for arm in ARMS}

    # secondary (never a verdict)
    wins = (MAIN_WIN, *FORM_WINS)
    levels = {}
    for arm in ARMS:
        for s_ep in ("E1", "E2"):
            for win in wins:
                v = deltas(arm, s_ep, win)
                row = {"arm": arm, "quantity": f"delta_{s_ep}", "window": f"{win[0]}-{win[1]}",
                       "mean": float(v.mean()) if v.size else float("nan")}
                if arm != "ref":
                    p = paired(diffs(arm, s_ep, win), COMP_LEVEL)
                    row |= {"diff_mean": p["mean"], "diff_lo95": p["lo"], "diff_hi95": p["hi"],
                            "diff_pos_seeds": int((diffs(arm, s_ep, win) > 0).sum())}
                res["secondary"].append(row)
        for name, key in UNIT_Q.items():
            for win in wins:
                lv = [unit_window_level(shards[(arm, s)]["units"], key, win) for s in valid
                      if key in shards[(arm, s)]["units"]]
                if lv:
                    levels[(arm, name, win)] = float(np.mean(lv))
                    res["secondary"].append({"arm": arm, "quantity": f"level_{name}",
                                             "window": f"{win[0]}-{win[1]}", "mean": float(np.mean(lv))})
        for win in wins:
            lv = [cos_w2_window(shards[(arm, s)]["units"], win) for s in valid
                  if "q_l2" in shards[(arm, s)]["units"]]
            if lv:
                levels[(arm, "cos_w2_e2", win)] = float(np.mean(lv))
                res["secondary"].append({"arm": arm, "quantity": "level_cos_w2_e2",
                                         "window": f"{win[0]}-{win[1]}", "mean": float(np.mean(lv))})
        for key in SCALARS:
            for win in wins:
                lv = [scalar_window(shards[(arm, s)]["units"], key, win) for s in valid
                      if key in shards[(arm, s)]["units"]]
                if lv:
                    levels[(arm, key, win)] = float(np.mean(lv))
                    res["secondary"].append({"arm": arm, "quantity": f"level_{key}",
                                             "window": f"{win[0]}-{win[1]}", "mean": float(np.mean(lv))})
        for col in ("memo_acc", "online_acc", "m_l1_mean", "m_l2_mean", "m_l1_norm", "m_l2_norm",
                    "c_in_absmax", "c_in_ratio"):
            for win in ((1, 1), *wins):
                lv = [float(shards[(arm, s)]["per_task"].set_index("task").loc[win[0]:win[1], col].mean())
                      for s in valid if col in shards[(arm, s)]["per_task"]]
                if lv:
                    levels[(arm, col, win)] = float(np.mean(lv))
                    res["secondary"].append({"arm": arm, "quantity": f"level_{col}",
                                             "window": f"{win[0]}-{win[1]}", "mean": float(np.mean(lv))})
        ths = [th[(arm, s)] for s in valid]
        obs = [x for x in ths if x is not None]
        res["secondary"].append({"arm": arm, "quantity": "t_half_median_observed", "window": f"2-{N_TASKS}",
                                 "mean": float(np.median(obs)) if obs else float("nan"),
                                 "n_censored": len(ths) - len(obs)})
        res["secondary"].append({"arm": arm, "quantity": "seeds_at_floor", "window": f"{MAIN_WIN[0]}-{MAIN_WIN[1]}",
                                 "mean": float(sum(floors[(arm, s)]["at_floor"] for s in valid))})
    res["levels"] = {f"{a}|{q}|{w[0]}-{w[1]}": v for (a, q, w), v in levels.items()}
    # trajectories at fixed task ends (spec 4.2-3, 6): seed means
    traj = {}
    for arm in ARMS:
        for key in ("mu2_norm", "mu2_raw_norm"):
            for t in (1, 2, 5, 10, 20, 50, 100):
                v = [float(np.asarray(shards[(arm, s)]["units"][key][task_ends(shards[(arm, s)]["units"])[t]]).ravel()[0])
                     for s in valid if key in shards[(arm, s)]["units"]]
                if v:
                    traj[f"{arm}|{key}|{t}"] = float(np.mean(v))
        for key in ("zbar_l2", "b2", "dtrain_mean_l2", "dtrain_mean_l1", "row_norm_l2", "sigma_l1"):
            for t in (1, 2, 5, 10, 20, 50, 100):
                v = [unit_mean(shards[(arm, s)]["units"][key][task_ends(shards[(arm, s)]["units"])[t]])[0]
                     for s in valid if key in shards[(arm, s)]["units"]]
                if v:
                    traj[f"{arm}|{key}|{t}"] = float(np.mean(v))
        for col in ("online_acc", "m_l1_mean", "m_l2_mean", "m_l1_norm", "m_l2_norm"):
            for t in (1, 2, 5, 10, 20, 50, 100):
                v = [float(shards[(arm, s)]["per_task"].set_index("task").loc[t, col]) for s in valid
                     if col in shards[(arm, s)]["per_task"]]
                if v:
                    traj[f"{arm}|{col}|{t}"] = float(np.mean(v))
    res["traj"] = traj
    return res


# --------------------------------------------------------------------------
# spec 7.3: scoring the registered predictions
# --------------------------------------------------------------------------

CLAUDE_P = {
    "H": {"RESCUED": 0.62, "RESCUED_FUNCTION_ONLY": 0.06, "ALIVE_UNRESOLVED": 0.07, "SPLIT": 0.10,
          "COLLAPSED": 0.13, "OTHER": 0.02},
    "CH": {"RESCUED": 0.72, "RESCUED_FUNCTION_ONLY": 0.04, "ALIVE_UNRESOLVED": 0.04, "SPLIT": 0.08,
           "COLLAPSED": 0.10, "OTHER": 0.02},
    "C": {"RESCUED": 0.10, "RESCUED_FUNCTION_ONLY": 0.03, "ALIVE_UNRESOLVED": 0.05, "SPLIT": 0.12,
          "COLLAPSED": 0.68, "OTHER": 0.02},
    "N": {"NEED_C": 0.10, "NEED_H": 0.54, "NEED_CH": 0.18, "NONE_RESCUED": 0.18},
}
# the parent's "other" is one bucket for every label it did not name (spec 7.3)
PARENT_P = {
    "H": {"RESCUED": 0.60, "RESCUED_FUNCTION_ONLY": 0.10, "SPLIT": 0.15, "COLLAPSED": 0.15},
    "CH": {"RESCUED": 0.65, "SPLIT": 0.15, "COLLAPSED": 0.15, "OTHER": 0.05},
    "C": {"COLLAPSED": 0.65, "SPLIT": 0.20, "RESCUED": 0.15},
    "N": {"NEED_H": 0.55, "NEED_CH": 0.15, "NEED_C": 0.10, "NONE_RESCUED": 0.20},
}


def score_item(table: dict, key: str, universe: tuple, realized: str) -> dict:
    """p(key), mode hit and the multi-class Brier score over universe + key; a label the table does not
    name has probability 0.  key is the realized label already mapped to the table's bucket."""
    p = {k: table.get(k, 0.0) for k in (*universe, key)}
    brier = sum((v - (1.0 if k == key else 0.0)) ** 2 for k, v in p.items())
    mode = max(table, key=table.get)
    return {"realized": realized, "bucket": key, "p_realized": float(table.get(key, 0.0)), "mode": mode,
            "mode_hit": mode == key, "brier": float(brier)}


def score_predictions(res: dict, ref_identity: dict | None) -> dict:
    lab = res["labels"]
    realized = {"H": lab["main"], "CH": lab["CH"], "C": lab["C"], "N": lab["N"]}
    out = {"claude": {}, "parent": {}, "claude_binary": {}}
    for item, r in realized.items():
        if item == "N":
            uni = N_LABELS
            rc = r if r in N_LABELS else "NONE_RESCUED"   # spec 7.1: Claude folded INAPPLICABLE/NOT_REPRODUCED in
            rp = r                                        # the parent named no such bucket: probability 0
        else:
            uni = LABELS6
            rc = r if r in LABELS6[:-1] else "OTHER"
            rp = r if r in PARENT_P[item] and r != "OTHER" else "OTHER"
        out["claude"][item] = score_item(CLAUDE_P[item], rc, uni, r)
        out["parent"][item] = score_item(PARENT_P[item], rp, uni, r)
    lv = res["levels"]
    ap = res["applicability"]
    tl = res["timing_labels"]

    def g(key):
        return lv.get(key, float("nan"))

    later_c = tl["C"]["later"] if "C" in tl else 0
    b2h = res["arm_b2_delta"].get("H", float("nan"))
    bin_ = {
        "P1_B_holds": (0.98, bool(ap["B_ok"])),
        "P2_ref_bit_identical_100_tasks": (0.90, None if ref_identity is None else bool(ref_identity["all_identical"])),
        "P3_C_holds_all_three": (0.95, all(ap["C"].values())),
        "P4_no_impaired": (0.85, not any(v["flag"] for v in res["impaired"].values())),
        "P5_C_T_half_later_in_7_of_10": (0.65, later_c >= 7),
        "P6_H_mu2_window_below_10": (0.90, g("H|mu2_norm|51-100") < 10),
        "P7_H_online_window_in_0.20_0.50": (0.50, 0.20 <= g("H|online_acc|51-100") <= 0.50),
        "P8_CH_online_window_above_0.70": (0.55, g("CH|online_acc|51-100") > 0.70),
        "P9_CH_minus_H_E1_lo95_positive": (0.75, res["ch_minus_h"]["E1"]["lo"] > 0),
        "P10_interaction_E1_lo95_positive": (0.50, res["interaction"]["E1"]["lo"] > 0),
        "P11_H_abs_delta_b2_below_3": (0.75, abs(b2h) < 3),
    }
    for k, (p, hit) in bin_.items():
        out["claude_binary"][k] = {"p": p, "true": hit,
                                   "brier": None if hit is None else float((p - (1.0 if hit else 0.0)) ** 2)}
    return out


# --------------------------------------------------------------------------
# spec 4.2-8: the ref arm against l2cap_ee_0917's ref, all 100 tasks (report only)
# --------------------------------------------------------------------------

def ref_identity(shards: dict) -> dict:
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
        a = rec[cols_rec].to_numpy(dtype=float)
        b = got[cols_rec].to_numpy(dtype=float) if same_cols else None
        rows_ok = same_cols and a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
        f = bk.get(f"results/l2cap_ee_0917/runs/ref_s{s}/units.npz")
        arr_ok, n_arr, why = None, 0, ""
        if f is not None and Path(f["backup"]).exists():
            h = hashlib.sha256(Path(f["backup"]).read_bytes()).hexdigest()
            if h == f["sha256"]:
                with np.load(f["backup"]) as z:
                    pre = f"s{s}_"
                    keys = {k[len(pre):] for k in z.files}
                    arr_ok = keys == set(sh["units"])
                    for k in keys:
                        n_arr += 1
                        if not np.array_equal(z[pre + k], sh["units"].get(k), equal_nan=True):
                            arr_ok = False
                            why = k
                            break
            else:
                why = "archive sha256 mismatch"
        else:
            why = "archive missing"
        pv = json.loads((d / "provenance.json").read_text())["per_seed"][str(s)]
        mine = sh["prov"]["per_seed"][str(s)]
        hashes_ok = all(pv[k] == mine[k] for k in ("init_sha256", "subset_idx_sha256", "labels_sha256",
                                                   "batch_sha256", "task1_end_state_sha256", "final_state_sha256"))
        ok = bool(rows_ok and arr_ok and hashes_ok)
        all_ok &= ok
        per[s] = {"per_task": rows_ok, "units": arr_ok, "n_arrays": n_arr, "first_bad": why, "hashes": hashes_ok}
    return {"all_identical": bool(all_ok), "per_seed": per}


# --------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------

def _f(x, nd=4):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{nd}f}"


def summary_md(res: dict, missing: list, env: dict, scores: dict, rid: dict | None) -> str:
    ap = res["applicability"]
    n = len(res["valid_seeds"])
    lv, tr = res["levels"], res["traj"]
    L = [f"# {RUN_ID} — 判定（中心化の扉 C / H / CH・RL-MNIST × ELU→ELU・{N_TASKS} 課題）", "",
         "> 自動生成: `analysis/doors_rlmnist_1007/verdict.py`。spec: `specs/spec_doors_rlmnist_1007.md`"
         "（登録 commit `0597501`）。数値は `verdict.csv`・`paired.csv`・`timing.csv`・`secondary.csv` から。", "",
         "## 0. 実行したもの", "",
         f"- 集計時の HEAD: `{env['head']}`・run の commit: {env['run_commits']}",
         f"- shard: {env['n_shards']}/40・欠損 {len(missing)}・有効 seed {n}（無効: {res['invalid_seeds'] or 'なし'}）", "",
         "## 1. 適用条件（spec §5.2）", "",
         f"- (A) 有効 seed {ap['A_valid_seeds']} ≥ {MIN_SEEDS}: **{ap['A_ok']}**",
         f"- (B) ref が主窓で床: {ap['B_ref_at_floor']}/{n} seed（要 ≥ {math.ceil(REF_FLOOR_FRAC * n)}）、"
         f"ref の第2層 ΔE2 {_f(ap['B_ref_E2']['mean'])} [{_f(ap['B_ref_E2']['lo'])}, {_f(ap['B_ref_E2']['hi'])}]"
         f" → **{ap['B_ok']}**",
         "- (C) 扉が全 100 課題で働いた記録: " + "、".join(
             f"{a} **{v}**" + (f"（欠け {ap['C_failed'][a][:5]}）" if ap['C_failed'][a] else "")
             for a, v in ap["C"].items()),
         "- IMPAIRED（読みのフラグ）: " + "、".join(
             f"{a} {v['seeds_below']}/{n} seed → **{v['flag']}**" for a, v in res["impaired"].items()),
         "- 主窓の床の状態: " + "、".join(f"{a} {v}" for a, v in res["floor_state"].items()),
         "", "## 2. 判定", "",
         "| 比較 | 水準 | ラベル | T_half（報告のみ） |", "|---|---|---|---|",
         f"| **主: H − ref** | 97.5% | **{res['labels']['main']}** | {res['labels']['timing_H']} |",
         f"| H − ref（参考） | 95% | {res['labels']['H_95']} | |",
         f"| C − ref | 95% | {res['labels']['C']} | {res['labels']['timing_C']} |",
         f"| CH − ref | 95% | {res['labels']['CH']} | {res['labels']['timing_CH']} |",
         f"| **梯子 N（C → H → CH）** | 各腕の登録水準 | **{res['labels']['N']}** | |", "",
         f"## 3. 腕間差（主窓 task {MAIN_WIN[0]}–{MAIN_WIN[1]}・seed 内対応差・各腕自身の task 1 から）", "",
         "| 腕 | endpoint | 水準 | 平均 | SD | 区間 | 符号 |", "|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['arm']} | {r['endpoint']} | {r['level']} | {_f(r['mean'])} | {_f(r['sd'])} | "
                 f"[{_f(r['lo'])}, {_f(r['hi'])}]{' (退化)' if r['degenerate'] else ''} | {r['sign']} |")
    L += ["", "交互作用 δ_CH − δ_C − δ_H（95%）: " + "、".join(
        f"{ep} {_f(x['mean'])} [{_f(x['lo'])}, {_f(x['hi'])}]" for ep, x in res["interaction"].items()),
          "", "CH − H（95%）: " + "、".join(
        f"{ep} {_f(x['mean'])} [{_f(x['lo'])}, {_f(x['hi'])}]" for ep, x in res["ch_minus_h"].items()), "",
          "輸送（主窓の第2層 Δz̄₂ と Δb₂ の ref との差・95%）: " + "、".join(
        f"{k} {_f(x['mean'], 3)} [{_f(x['lo'], 3)}, {_f(x['hi'], 3)}]" for k, x in res["transport"].items()), "",
          "### 3.1 seed 別の差と床", "", "| 腕 | endpoint | " + " | ".join(f"s{s}" for s in res["valid_seeds"]) + " |",
          "|---|---|" + "---|" * n]
    for arm, eps in res["per_seed"].items():
        for ep, v in eps.items():
            L.append(f"| {arm} | δ{ep} | " + " | ".join(_f(x, 3) for x in v) + " |")
    for arm in ARMS:
        L.append(f"| {arm} | 主窓 online（閾値） | " + " | ".join(
            f"{res['floors'][f'{arm}|{s}']['level']:.3f} ({res['floors'][f'{arm}|{s}']['thr']:.3f})"
            f"{' 床' if res['floors'][f'{arm}|{s}']['at_floor'] else ''}" for s in res["valid_seeds"]) + " |")
    L += ["", f"### 3.2 T_half（報告のみ。適合率が初めて 1/2 を下回る課題・— は {N_TASKS} まで未到達）", "",
          "| 腕 | " + " | ".join(f"s{s}" for s in res["valid_seeds"]) + " | 早/遅/同 | p |", "|---|" + "---|" * (n + 2)]
    for arm in ARMS:
        tl = res["timing_labels"].get(arm)
        L.append(f"| {arm} | " + " | ".join(str(res['t_half'][f'{arm}|{s}'] or '—') for s in res["valid_seeds"])
                 + (f" | {tl['earlier']}/{tl['later']}/{tl['ties']} | {tl['p']:.3g} |" if tl else " | | |"))
    L += ["", f"### 3.3 第2層のユニット平均 z̄₂ が初めて ln 2^−24 = {ZERO_Z32:.2f} を下回る課題（報告のみ・— は未到達）", "",
          "| 腕 | " + " | ".join(f"s{s}" for s in res["valid_seeds"]) + " |", "|---|" + "---|" * n]
    for arm in ARMS:
        L.append(f"| {arm} | " + " | ".join(str(res['t_cross'][f'{arm}|{s}'] or '—') for s in res["valid_seeds"]) + " |")
    L += ["", "## 4. 機構（報告のみ・課題終端の seed 平均）", "",
          "| 腕 | 量 | t1 | t2 | t5 | t10 | t20 | t50 | t100 |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for key in ("online_acc", "mu2_norm", "mu2_raw_norm", "zbar_l2", "b2", "dtrain_mean_l2", "dtrain_mean_l1",
                    "row_norm_l2", "sigma_l1", "m_l1_mean", "m_l2_mean", "m_l1_norm", "m_l2_norm"):
            vals = [tr.get(f"{arm}|{key}|{t}") for t in (1, 2, 5, 10, 20, 50, 100)]
            if any(v is not None for v in vals):
                L.append(f"| {arm} | {key} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + " |")
    L += ["", "## 5. 副 endpoint（判定に使わない）", "", "| 腕 | 量 | 窓 | 値 | 差 [95%] |", "|---|---|---|---|---|"]
    for r in res["secondary"]:
        diff = (f"{_f(r['diff_mean'])} [{_f(r['diff_lo95'])}, {_f(r['diff_hi95'])}] ({r['diff_pos_seeds']}/{n} 正)"
                if "diff_mean" in r else "")
        L.append(f"| {r['arm']} | {r['quantity']} | {r['window']} | {_f(r['mean'])} | {diff} |")
    L += ["", "## 6. ref の全 100 課題の bit 一致（spec §4.2-8・報告のみ）", ""]
    if rid is None:
        L.append("- 照合しなかった")
    else:
        L.append(f"- 全 seed で一致: **{rid['all_identical']}**")
        for s, v in rid["per_seed"].items():
            L.append(f"  - s{s}: {v}")
    L += ["", "## 7. 予測の採点（spec §7.3）", "",
          "| 項目 | 実現 | Claude: p(実現) | Claude: 最頻 | Claude: Brier | 親: p(実現) | 親: 最頻 | 親: Brier |",
          "|---|---|---|---|---|---|---|---|"]
    for item in ("H", "CH", "C", "N"):
        c, p = scores["claude"][item], scores["parent"][item]
        L.append(f"| {item} | {c['realized']} | {c['p_realized']:.2f} | {c['mode']}（{'的中' if c['mode_hit'] else '外れ'}） | "
                 f"{c['brier']:.3f} | {p['p_realized']:.2f} | {p['mode']}（{'的中' if p['mode_hit'] else '外れ'}） | {p['brier']:.3f} |")
    L += ["", "| Claude の副予測 | p | 実現 | Brier |", "|---|---|---|---|"]
    for k, v in scores["claude_binary"].items():
        b = "—" if v["brier"] is None else format(v["brier"], ".3f")
        L.append(f"| {k} | {v['p']:.2f} | {v['true']} | {b} |")
    L += ["", "## 8. 読み方（spec §5.5 のまま）", "",
          "- RESCUED は「この箱の 100 課題で床に落ちなかった」であり、無限時間の非崩壊ではない。",
          "- E1・E2 は各腕自身の task 1 からの変化で、δ は窓の水準の差ではない（扉は task 1 から入る）。",
          "- 交互作用は報告のみ。単独の腕の 0 を「両方必要」とは書かない。",
          "- 機構（‖µ₂‖・z̄₂・b₂・m）は報告のみ。腕の順位の一致は因果の証明ではない。"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "results" / RUN_ID / "runs"))
    ap.add_argument("--out", default=str(REPO / "results" / RUN_ID))
    ap.add_argument("--no-ref-identity", action="store_true")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    shards, missing = load(src)
    res = analyze(shards)
    rid = None if args.no_ref_identity else ref_identity(shards)
    scores = score_predictions(res, rid)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run_commits = sorted({sh["prov"].get("git_hash") for sh in shards.values()})
    env = {"head": head, "run_commits": run_commits, "n_shards": len(shards)}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res["rows"]).to_csv(out / "verdict.csv", index=False)
    pd.DataFrame([{"arm": a, "endpoint": ep, "seed": s, "delta_minus_ref": v}
                  for a, eps in res["per_seed"].items() for ep, vals in eps.items()
                  for s, v in zip(res["valid_seeds"], vals)]).to_csv(out / "paired.csv", index=False)
    pd.DataFrame(res["timing"]).to_csv(out / "timing.csv", index=False)
    pd.DataFrame(res["secondary"]).to_csv(out / "secondary.csv", index=False)
    (out / "summary.md").write_text(summary_md(res, missing, env, scores, rid))
    (out / "provenance.json").write_text(json.dumps({
        "run_id": RUN_ID, "aggregated_at": dt.datetime.now().astimezone().isoformat(), "head": head,
        "run_commits": run_commits, "src": str(src), "n_shards": len(shards), "missing": missing,
        "valid_seeds": res["valid_seeds"], "invalid_seeds": res["invalid_seeds"],
        "applicability": res["applicability"], "floor_state": res["floor_state"], "floors": res["floors"],
        "impaired": res["impaired"], "labels": res["labels"], "timing": res["timing_labels"],
        "t_half": res["t_half"], "t_cross": res["t_cross"], "traj": res["traj"], "levels": res["levels"],
        "interaction": {k: {kk: v[kk] for kk in ("mean", "lo", "hi", "sign")} for k, v in res["interaction"].items()},
        "ch_minus_h": {k: {kk: v[kk] for kk in ("mean", "lo", "hi", "sign")} for k, v in res["ch_minus_h"].items()},
        "transport": {k: {kk: v[kk] for kk in ("mean", "lo", "hi", "sign")} for k, v in res["transport"].items()},
        "ref_identity": rid, "scores": scores,
        "verdict_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "el_verdict_sha256": hashlib.sha256((REPO / "analysis" / "mucap_el_0916" / "verdict.py").read_bytes()).hexdigest(),
        "judgement": {"main_arm": MAIN_ARM, "main_window": MAIN_WIN, "door_window": DOOR_WIN,
                      "level_of": LEVEL_OF, "ladder": LADDER, "min_seeds": MIN_SEEDS,
                      "ref_floor_frac": REF_FLOOR_FRAC, "floor_k": FLOOR_K, "half": HALF,
                      "sign_alpha": SIGN_ALPHA, "early": EARLY, "chance": CHANCE},
    }, indent=2, default=str))
    print(f"labels: {res['labels']}  valid seeds {len(res['valid_seeds'])}  missing {len(missing)}")


if __name__ == "__main__":
    main()

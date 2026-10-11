#!/usr/bin/env python3
"""Registered verdict of cnn_center_relu_1011 (spec §5) plus the report-only tables (§5.11).

    python analysis/cnn_center_relu_1011/verdict.py            # -> results/cnn_center_relu_1011/{verdict.json,summary.md,tables/}

Labels come only from per_task.csv (git) and checks.json; the mechanism tables also read
units.npz (git-external: the run directory, or the backup in obsidian-research-data after §9).
"""

from __future__ import annotations

import json
import math
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "cnn_center_relu_1011"
BACKUP = Path.home() / "Projects" / "obsidian-research-data" / "cnn_center_relu_1011" / "results" / "cnn_center_relu_1011"
MAIN_ARMS = ("ref", "C", "H", "CH")
H_ARMS = ("H", "CH")
SEEDS = tuple(range(10, 20))
CR_SEED = 10
N_TASKS = 30
WINDOW = (21, 30)                      # main window (spec §4)
SIDE_WINDOWS = ((1, 5), (6, 10), (11, 20))
EARLY, LATE = (1, 5), (26, 30)         # drop = mean(EARLY) - mean(LATE)
CR_WINDOWS = ((1, 5), (6, 10), (11, 20), (21, 30))
KEEP_LINE = 0.5                        # relu_doors_0919's rescue line (spec §5.1)
CHANCE = 0.1
FLOOR_SIGMAS = 3.0
TF_RATIO = 0.5                         # V12 §4.0: functional loss when the fit ratio < 1/2
TD_OPEN = 0.01                         # V12 §4.0: layer loss when the unit-median open fraction < 1%
ALPHA = 0.05
LAYERS = ("c1", "c2", "f1", "f2")
DEEP = frozenset({"c2", "f1", "f2"})
CKPT = (1, 2, 3, 5, 10, 20, 30)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided p of k successes in n fair trials (k counted on the larger side)."""
    if n == 0:
        return float("nan")
    k = max(k, n - k)
    return min(1.0, 2.0 * sum(comb(n, j) for j in range(k, n + 1)) / 2.0 ** n)


def kstar(n: int) -> int | None:
    """Smallest k with two-sided binomial p(k of n) < 0.05 (spec §5.2: n = 10 -> 9)."""
    for k in range(n // 2 + 1, n + 1):
        if binom_two_sided(k, n) < ALPHA:
            return k
    return None


def sign_test(diffs) -> dict:
    """Exact two-sided sign test, ties excluded; HELPS / HURTS need p < 0.05 and <= 1 opposite."""
    d = [float(x) for x in diffs if np.isfinite(x)]
    nz = [x for x in d if x != 0.0]
    n = len(nz)
    pos = sum(1 for x in nz if x > 0)
    neg = n - pos
    p = binom_two_sided(pos, n) if n else float("nan")
    if n and p < ALPHA and neg <= 1:
        lab = "HELPS"
    elif n and p < ALPHA and pos <= 1:
        lab = "HURTS"
    else:
        lab = "TIE"
    return {"label": lab, "median": float(np.median(d)) if d else float("nan"), "pos": pos, "neg": neg,
            "n": n, "ties": len(d) - n, "p": p}


# --------------------------------------------------------------------------
# per seed
# --------------------------------------------------------------------------

def window_mean(g: pd.DataFrame, col: str, w) -> float:
    s = g[(g.task >= w[0]) & (g.task <= w[1])][col]
    return float(s.mean()) if len(s) == w[1] - w[0] + 1 else float("nan")


def t_f(online: dict) -> int:
    """First task t >= 2 with fit ratio (A(t) - 0.1)/(A(1) - 0.1) < 1/2; 31 when never."""
    a1 = online.get(1, float("nan"))
    if not np.isfinite(a1) or a1 - CHANCE <= 0:
        return 1
    for t in range(2, N_TASKS + 1):
        if (online[t] - CHANCE) / (a1 - CHANCE) < TF_RATIO:
            return t
    return N_TASKS + 1


def t_d(mob: dict) -> int:
    for t in range(1, N_TASKS + 1):
        if mob[t] < TD_OPEN:
            return t
    return N_TASKS + 1


def seed_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per seed with everything §5 needs."""
    out = []
    for seed, g in df.groupby("seed"):
        g = g.sort_values("task")
        tasks = set(int(t) for t in g.task)
        complete = tasks >= set(range(1, N_TASKS + 1))
        finite = bool(np.isfinite(g.online_acc).all() and np.isfinite(g.memo_acc).all())
        diverged = bool(g.diverged.astype(str).str.lower().eq("true").any()) if "diverged" in g else False
        valid = complete and finite and not diverged
        on = {int(t): float(v) for t, v in zip(g.task, g.online_acc)}
        row = {"seed": int(seed), "valid": valid}
        if complete:
            W = window_mean(g, "online_acc", WINDOW)
            mf = g[(g.task >= WINDOW[0]) & (g.task <= WINDOW[1])].major_frac.to_numpy(float)
            floor = float(mf.mean() + FLOOR_SIGMAS * mf.std(ddof=1) / math.sqrt(len(mf)))
            row |= {"W": W, "kept": W >= KEEP_LINE, "floor": floor, "at_floor": W <= floor,
                    "t1": on[1], "early": window_mean(g, "online_acc", EARLY),
                    "late": window_mean(g, "online_acc", LATE),
                    "T_f": t_f(on)}
            row["drop"] = row["early"] - row["late"]
            for w in SIDE_WINDOWS + CR_WINDOWS:
                row[f"W{w[0]}_{w[1]}"] = window_mean(g, "online_acc", w)
            tds = {}
            for l in LAYERS:
                mob = {int(t): float(v) for t, v in zip(g.task, g[f"mob_{l}"])}
                tds[l] = t_d(mob)
                row[f"T_d_{l}"] = tds[l]
            mn = min(tds.values())
            row["first_layers"] = ",".join(l for l in LAYERS if tds[l] == mn) if mn <= N_TASKS else ""
            last = g[g.task == N_TASKS].iloc[0]
            row["memo_t30"] = float(last.memo_acc)
            if "memo_acc_m0" in g:
                row["memo_m0_t30"] = float(last.memo_acc_m0)
            for l in LAYERS:
                row[f"mob_t30_{l}"] = float(last[f"mob_{l}"])
                row[f"dead_t30_{l}"] = float(last[f"dead_frac_{l}"])
        out.append(row)
    return pd.DataFrame(out).set_index("seed")


def door_condition_C(df: pd.DataFrame, steps_per_task: int) -> dict:
    """§5.4 (C) for the H arms: every (seed, task, layer) has h_upd = steps or a fully dead layer."""
    bad = []
    for l in LAYERS:
        col, dcol = f"h_upd_{l}", f"dead_frac_{l}"
        if col not in df:
            return {"ok": False, "reason": f"missing {col}"}
        mask = ~((df[col] == steps_per_task) | (df[dcol] == 1.0))
        for _, r in df[mask].iterrows():
            bad.append((int(r.seed), int(r.task), l, int(r[col]), float(r[dcol])))
    return {"ok": not bad, "violations": bad[:20], "n_violations": len(bad)}


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

def arm_label(st: pd.DataFrame) -> dict:
    v = st[st.valid]
    n = len(v)
    if n < 8:
        return {"label": "INAPPLICABLE", "n_valid": n}
    k = kstar(n)
    kept = int(v.kept.sum())
    at_floor = int(v.at_floor.sum())
    if kept >= k:
        lab = "RESCUED"
    elif kept <= n - k:
        lab = "FAIL"
    else:
        lab = "PARTIAL"
    sub = None
    if lab == "FAIL":
        sub = "FAIL_FLOOR" if at_floor >= k else "FAIL_ABOVE_FLOOR"
    return {"label": lab, "fail_subtype": sub, "n_valid": n, "kstar": k, "kept": kept, "at_floor": at_floor,
            "W_median": float(v.W.median()), "W_min": float(v.W.min()), "W_max": float(v.W.max()),
            "t1_median": float(v.t1.median()), "drop_median": float(v["drop"].median())}


def paired(st_a: pd.DataFrame, st_b: pd.DataFrame, col: str = "W") -> dict:
    s = sorted(set(st_a[st_a.valid].index) & set(st_b[st_b.valid].index))
    return sign_test([st_a.loc[x, col] - st_b.loc[x, col] for x in s]) | {"seeds": s}


def ladder(labels: dict, cond_ok: bool, ref_fail: bool) -> str:
    if not cond_ok:
        return "INAPPLICABLE"
    if not ref_fail:
        return "NOT_REPRODUCED"
    for arm, name in (("C", "NEED_C"), ("H", "NEED_H"), ("CH", "NEED_CH")):
        if labels[arm] == "RESCUED":
            return name
    return "NONE_RESCUED"


def doors_summary(labels: dict, d_h_given_c: str, c_learned_t1: bool, cond_ok: bool) -> str:
    if not cond_ok:
        return "INAPPLICABLE"
    if labels["ref"] != "FAIL":
        return "NOT_REPRODUCED"
    if labels["CH"] == "FAIL":
        return "DOORS_FAIL"
    if (labels["CH"] == "RESCUED" and labels["C"] != "RESCUED" and c_learned_t1
            and d_h_given_c == "HELPS"):
        return "DOORS_TRANSFER"
    return "DOORS_PARTIAL"


def layer_correspondence(st_H: pd.DataFrame, st_C: pd.DataFrame, cond_ok: bool, ref_fail: bool) -> dict:
    if not cond_ok:
        return {"label": "INAPPLICABLE"}
    if not ref_fail:
        return {"label": "NOT_REPRODUCED"}
    vH, vC = st_H[st_H.valid], st_C[st_C.valid]
    kH, kC = kstar(len(vH)), kstar(len(vC))
    h_c1 = int(sum("c1" in s.split(",") for s in vH.first_layers if s))
    c_deep = int(sum(bool(s) and set(s.split(",")) <= DEEP for s in vC.first_layers))
    c1 = h_c1 >= kH
    c2 = c_deep >= kC
    return {"label": "LAYER_CORRESPONDENCE" if c1 and c2 else "LAYER_CORRESPONDENCE_NOT_SHOWN",
            "H_first_includes_c1": h_c1, "C_first_only_deep": c_deep, "kstar_H": kH, "kstar_C": kC,
            "cond1_H_c1": c1, "cond2_C_deep": c2}


def cr_label(cr: pd.Series | None, st_ref: pd.DataFrame, st_C: pd.DataFrame) -> dict:
    if cr is None:
        return {"label": "INAPPLICABLE", "reason": "no Cr run"}
    vr, vc = st_ref[st_ref.valid], st_C[st_C.valid]
    sep, sides = [], {}
    for w in CR_WINDOWS:
        col = f"W{w[0]}_{w[1]}"
        r_lo, r_hi, c_lo, c_hi = vr[col].min(), vr[col].max(), vc[col].min(), vc[col].max()
        if r_hi < c_lo or c_hi < r_lo:
            sep.append(col)
            v = float(cr[col])
            sides[col] = "ref" if r_lo <= v <= r_hi else ("C" if c_lo <= v <= c_hi else "outside")
    if not sep:
        lab = "CR_UNINFORMATIVE"
    elif all(s == "ref" for s in sides.values()):
        lab = "CONST_SHIFT_NO_EFFECT"
    elif all(s == "C" for s in sides.values()):
        lab = "CONST_SHIFT_ACTS_LIKE_C"
    else:
        lab = "CR_UNRESOLVED"
    return {"label": lab, "separating_windows": sep, "sides": sides,
            "Cr": {f"W{w[0]}_{w[1]}": float(cr[f"W{w[0]}_{w[1]}"]) for w in CR_WINDOWS}}


def evaloff(st: pd.DataFrame) -> dict:
    v = st[st.valid]
    t = sign_test(v.memo_t30 - v.memo_m0_t30)
    return {**t, "label": "EVALOFF_DEPENDS" if t["label"] == "HELPS" else "EVALOFF_NO_DEPENDENCE", "sign_label": t["label"],
            "memo_t30_median": float(v.memo_t30.median()), "memo_m0_t30_median": float(v.memo_m0_t30.median())}


def analyze_frames(frames: dict, checks_ok: bool, steps_per_task: int, cr_frame=None,
                   c_door_ok: bool = True) -> dict:
    """frames: arm -> per_task DataFrame (main arms); returns every registered label."""
    st = {a: seed_table(frames[a]) for a in MAIN_ARMS}
    arms = {a: arm_label(st[a]) for a in MAIN_ARMS}
    condC = {a: door_condition_C(frames[a], steps_per_task) for a in H_ARMS}
    for a in H_ARMS:
        if not condC[a]["ok"]:
            arms[a] = arms[a] | {"label": "INAPPLICABLE", "condition_C": condC[a]}
    if not c_door_ok:
        for a in ("C", "CH"):
            arms[a] = arms[a] | {"label": "INAPPLICABLE", "condition_C": "C door check failed"}
    labels = {a: arms[a]["label"] for a in MAIN_ARMS}
    cond_ok = checks_ok and all(l != "INAPPLICABLE" for l in labels.values())
    ref_fail = labels["ref"] == "FAIL"
    signs = {"D_C": paired(st["C"], st["ref"]), "D_H": paired(st["H"], st["ref"]),
             "D_CH": paired(st["CH"], st["ref"]), "D_H|C": paired(st["CH"], st["C"]),
             "D_C|H": paired(st["CH"], st["H"])}
    vC = st["C"][st["C"].valid]
    kC = kstar(len(vC)) if len(vC) else None
    c_learned = bool(kC is not None and int((vC.t1 >= KEEP_LINE).sum()) >= kC)
    res = {
        "conditions": {"A_checks_all_pass": checks_ok, "n_valid": {a: arms[a].get("n_valid") for a in MAIN_ARMS},
                       "B_ref_fail": ref_fail, "C_h_door": condC, "C_c_door_check": c_door_ok,
                       "steps_per_task": steps_per_task},
        "arms": arms,
        "signs": {k: {kk: vv for kk, vv in v.items()} for k, v in signs.items()},
        "C_learned_t1": c_learned,
        "N": ladder(labels, cond_ok, ref_fail),
        "DOORS": doors_summary(labels, signs["D_H|C"]["label"], c_learned, cond_ok),
        "M_H": {"label": labels["H"],
                "vs_MLP": {"FAIL": "H_LIKE_MLP", "RESCUED": "H_DIFFERS_FROM_MLP",
                           "PARTIAL": "H_PARTIAL"}.get(labels["H"], labels["H"])},
        "LAYER": layer_correspondence(st["H"], st["C"], cond_ok, ref_fail),
        "flags": {
            "RESIDUAL_LOSS_CH": (lambda t: {"set": t["label"] == "HELPS", **t})(
                sign_test(st["CH"][st["CH"].valid]["drop"])),
            "T1_LOWER": {a: (lambda t: {"set": t["label"] == "HURTS", **t})(paired(st[a], st["ref"], "t1"))
                         for a in ("C", "H", "CH")},
        },
        "EVALOFF": {a: evaloff(st[a]) for a in H_ARMS},
        "timing_T_f": {a: {"median": float(st[a][st[a].valid].T_f.median()),
                           "vs_ref": (lambda t: {**t, "label": {"HELPS": "LATER", "HURTS": "EARLIER"}.get(
                               t["label"], "NO_TIMING_DIFF")})(paired(st[a], st["ref"], "T_f"))}
                       for a in MAIN_ARMS},
    }
    if cr_frame is not None:
        cst = seed_table(cr_frame)
        res["Cr"] = cr_label(cst.loc[CR_SEED] if CR_SEED in cst.index else None, st["ref"], st["C"])
    else:
        res["Cr"] = cr_label(None, st["ref"], st["C"])
    res["_seed_tables"] = st
    return res


# --------------------------------------------------------------------------
# report-only mechanism tables (units.npz)
# --------------------------------------------------------------------------

def find_units(arm: str) -> Path | None:
    for base in (OUT, BACKUP):
        p = base / arm / "units.npz"
        if p.exists():
            return p
    return None


def mechanism_tables(arms=MAIN_ARMS) -> dict:
    """mu path / b path (t0 -> t) per layer: unit median of dz, db, dwm and the b share
    |db| / (|db| + |dwm|), then the seed median.  Also zbar / sd / open / dead per checkpoint."""
    rows, lay = [], []
    for a in arms:
        p = find_units(a)
        if p is None:
            continue
        u = np.load(p)
        for l in LAYERS:
            zb, b, wm = u[f"zbar_{l}"], u[f"b_{l}"], u[f"wm_{l}"]           # (T+1, R, n)
            for t in CKPT:
                if t >= zb.shape[0] or not np.isfinite(zb[t]).all():
                    continue
                dz, db, dw = zb[t] - zb[0], b[t] - b[0], wm[t] - wm[0]
                share = np.abs(db) / (np.abs(db) + np.abs(dw) + 1e-30)
                rows.append({"arm": a, "layer": l, "t": t,
                             "dz_med": float(np.median(np.median(dz, 1))),
                             "db_med": float(np.median(np.median(db, 1))),
                             "dwm_med": float(np.median(np.median(dw, 1))),
                             "b_share_med": float(np.median(np.median(share, 1)))})
            for t in (0,) + CKPT:
                if t >= zb.shape[0] or not np.isfinite(zb[t]).all():
                    continue
                sd, op, dead = u[f"sd_{l}"][t], u[f"open_{l}"][t], u[f"dead_{l}"][t]
                min_ = u[f"m_in_{l}"][t]
                lay.append({"arm": a, "layer": l, "t": t,
                            "zbar_med": float(np.median(np.median(zb[t], 1))),
                            "sd_med": float(np.median(np.median(sd, 1))),
                            "rel_pos_med": float(np.median(np.median(zb[t] / np.maximum(sd, 1e-30), 1))),
                            "open_mean": float(np.median(op.mean(1))),
                            "dead_frac": float(np.median(dead.mean(1))),
                            "m_in_norm": float(np.median(np.linalg.norm(min_, axis=1)))})
    return {"path": pd.DataFrame(rows), "layers": pd.DataFrame(lay)}


def leaky_reference() -> dict:
    """Same engine, same seeds, leaky 0.1: LR (raw) and LRc (centred), report only."""
    out = {}
    for name, p in (("LR", REPO / "results" / "cnn_drive_verify_1009" / "LR" / "per_task.csv"),
                    ("LRc", REPO / "results" / "cnn_center_1010" / "LRc" / "per_task.csv")):
        if p.exists():
            d = pd.read_csv(p)
            g = []
            for seed, s in d.groupby("seed"):
                g.append({"seed": seed, "W": window_mean(s, "online_acc", WINDOW),
                          "drop": window_mean(s, "online_acc", EARLY) - window_mean(s, "online_acc", LATE)})
            g = pd.DataFrame(g)
            out[name] = {"W_median": float(g.W.median()), "drop_median": float(g["drop"].median()),
                         "drop_mean": float(g["drop"].mean())}
    return out


# --------------------------------------------------------------------------
# prediction scoring (spec §7)
# --------------------------------------------------------------------------

PRED_LABELS = {
    "L1": ("ref", {"FAIL": .93, "PARTIAL": .04, "RESCUED": .01, "OTHER": .02}),
    "L2": ("C", {"FAIL": .72, "PARTIAL": .16, "RESCUED": .10, "OTHER": .02}),
    "L3": ("H", {"FAIL": .40, "PARTIAL": .30, "RESCUED": .28, "OTHER": .02}),
    "L4": ("CH", {"RESCUED": .64, "PARTIAL": .20, "FAIL": .14, "OTHER": .02}),
    "L5": ("N", {"NEED_CH": .36, "NONE_RESCUED": .30, "NEED_H": .24, "NEED_C": .08, "OTHER": .02}),
    "L6": ("DOORS", {"DOORS_TRANSFER": .50, "DOORS_PARTIAL": .30, "DOORS_FAIL": .13, "OTHER": .07}),
    "L7": ("LAYER", {"LAYER_CORRESPONDENCE": .30, "LAYER_CORRESPONDENCE_NOT_SHOWN": .65, "NOT_REPRODUCED": .05}),
    "L8": ("Cr", {"CR_UNINFORMATIVE": .50, "CONST_SHIFT_NO_EFFECT": .30, "CR_UNRESOLVED": .10,
                  "CONST_SHIFT_ACTS_LIKE_C": .10}),
}
# Fable (parent), relayed 2026-10-11 13:57 before any door run (spec §7.2); "OTHER" is one bucket
PRED_LABELS_FABLE = {
    "L2": ("C", {"FAIL": .70, "PARTIAL": .22, "RESCUED": .08}),
    "L3": ("H", {"FAIL": .45, "PARTIAL": .30, "RESCUED": .25}),
    "L4": ("CH", {"RESCUED": .60, "PARTIAL": .25, "FAIL": .15}),
    "L5": ("N", {"NEED_CH": .40, "NONE_RESCUED": .30, "NEED_H": .20, "OTHER": .10}),
    "L6": ("DOORS", {"DOORS_TRANSFER": .45, "DOORS_PARTIAL": .35, "DOORS_FAIL": .12, "NOT_REPRODUCED": .08}),
    "L7": ("LAYER", {"LAYER_CORRESPONDENCE": .30, "OTHER": .70}),
}
PRED_BINARY = {"B1": .60, "B2": .55, "B3": .25, "B4": .50, "B5": .50, "B6": .70, "B7": .35, "B8": .12,
               "B9": .90, "B10": .75, "B11": .45, "B12": .80}
PRED_TEXT = {
    "B1": "ref's FAIL subtype is FAIL_FLOOR (scored only if ref FAIL)",
    "B2": "ref's first failing layers include c2 in >= 9 seeds",
    "B3": "C's T_f is LATER than ref's",
    "B4": "H's t30 mob_c1 (seed median) < 0.01",
    "B5": "CH's main-window online (seed median) >= 0.90",
    "B6": "RESIDUAL_LOSS set on CH",
    "B7": "CH's drop (seed median) < LRc's drop (seed median, leaky, same seeds)",
    "B8": "T1_LOWER set on CH",
    "B9": "CH is EVALOFF_DEPENDS",
    "B10": "CH's b-path share t0->t30 > 0.5 in both c2 and f2",
    "B11": "CH's b-path share t0->t30 > 0.5 in f1",
    "B12": "ref's c2 dz t0->t30 < -50",
}


def realised(res: dict) -> dict:
    lab = {"ref": res["arms"]["ref"]["label"], "C": res["arms"]["C"]["label"], "H": res["arms"]["H"]["label"],
           "CH": res["arms"]["CH"]["label"], "N": res["N"], "DOORS": res["DOORS"],
           "LAYER": res["LAYER"]["label"], "Cr": res["Cr"]["label"]}
    return lab


def score_labels(res: dict, table: dict) -> dict:
    lab = realised(res)
    out = {}
    for key, (item, probs) in table.items():
        r = lab[item]
        bucket = r if r in probs else "OTHER"
        p = probs.get(bucket, 0.0)
        brier = sum((pk - (1.0 if k == bucket else 0.0)) ** 2 for k, pk in probs.items()) + \
            (0.0 if bucket in probs else 1.0)
        top = max(probs, key=probs.get)
        out[key] = {"item": item, "realised": r, "p_realised": p, "mode": top, "mode_hit": top == bucket,
                    "brier": brier}
    return out


def score(res: dict, mech: dict | None, leaky: dict) -> dict:
    lab = realised(res)
    out = {"labels": {}, "binary": {}, "labels_fable": score_labels(res, PRED_LABELS_FABLE)}
    for key, (item, probs) in PRED_LABELS.items():
        r = lab[item]
        bucket = r if r in probs else "OTHER"
        p = probs.get(bucket, 0.0)
        brier = sum((pk - (1.0 if k == bucket else 0.0)) ** 2 for k, pk in probs.items()) + \
            (0.0 if bucket in probs else 1.0)
        top = max(probs, key=probs.get)
        out["labels"][key] = {"item": item, "realised": r, "p_realised": p, "mode": top, "mode_hit": top == bucket,
                              "brier": brier}
    st = res["_seed_tables"]
    truth: dict[str, bool | None] = {}
    truth["B1"] = (res["arms"]["ref"].get("fail_subtype") == "FAIL_FLOOR") if lab["ref"] == "FAIL" else None
    vr = st["ref"][st["ref"].valid]
    truth["B2"] = int(sum("c2" in s.split(",") for s in vr.first_layers if s)) >= (kstar(len(vr)) or 99)
    truth["B3"] = res["timing_T_f"]["C"]["vs_ref"]["label"] == "LATER"
    vh = st["H"][st["H"].valid]
    truth["B4"] = float(vh.mob_t30_c1.median()) < 0.01
    truth["B5"] = res["arms"]["CH"].get("W_median", float("nan")) >= 0.90
    truth["B6"] = res["flags"]["RESIDUAL_LOSS_CH"]["set"]
    truth["B7"] = (res["arms"]["CH"].get("drop_median", float("nan")) < leaky["LRc"]["drop_median"]) \
        if "LRc" in leaky else None
    truth["B8"] = res["flags"]["T1_LOWER"]["CH"]["set"]
    truth["B9"] = res["EVALOFF"]["CH"]["label"] == "EVALOFF_DEPENDS"
    if mech is not None and len(mech["path"]):
        pth = mech["path"]

        def share(arm, layer):
            q = pth[(pth.arm == arm) & (pth.layer == layer) & (pth.t == N_TASKS)]
            return float(q.b_share_med.iloc[0]) if len(q) else float("nan")

        truth["B10"] = share("CH", "c2") > 0.5 and share("CH", "f2") > 0.5
        truth["B11"] = share("CH", "f1") > 0.5
        q = pth[(pth.arm == "ref") & (pth.layer == "c2") & (pth.t == N_TASKS)]
        truth["B12"] = float(q.dz_med.iloc[0]) < -50 if len(q) else None
    else:
        truth["B10"] = truth["B11"] = truth["B12"] = None
    for k, p in PRED_BINARY.items():
        o = truth.get(k)
        out["binary"][k] = {"claim": PRED_TEXT[k], "p": p, "true": o,
                            "brier": None if o is None else (p - (1.0 if o else 0.0)) ** 2}
    return out


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def load_frames(base: Path = OUT) -> tuple[dict, pd.DataFrame | None]:
    frames = {a: pd.read_csv(base / a / "per_task.csv") for a in MAIN_ARMS}
    crp = base / "Cr" / "per_task.csv"
    return frames, (pd.read_csv(crp) if crp.exists() else None)


def steps_per_task(base: Path = OUT) -> int:
    prov = json.loads((base / "CH" / "provenance.json").read_text())
    return int(prov["epochs"]) * 75


def checks_state(base: Path = OUT) -> tuple[bool, bool]:
    p = base / "checks.json"
    if not p.exists():
        return False, False
    c = json.loads(p.read_text())
    return bool(c.get("all_pass")), bool(c.get("K-center", {}).get("pass"))


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (
        f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_report(res: dict, mech: dict, leaky: dict, sc: dict, base: Path = OUT) -> None:
    (base / "tables").mkdir(parents=True, exist_ok=True)
    for a, s in res["_seed_tables"].items():
        s.to_csv(base / "tables" / f"seeds_{a}.csv")
    if len(mech["path"]):
        mech["path"].to_csv(base / "tables" / "mu_b_path.csv", index=False)
        mech["layers"].to_csv(base / "tables" / "layers.csv", index=False)
    clean = {k: v for k, v in res.items() if k != "_seed_tables"}
    clean["leaky_reference"] = leaky
    clean["prediction_scoring"] = sc
    (base / "verdict.json").write_text(json.dumps(clean, indent=1, default=lambda o: o if not isinstance(
        o, (np.integer, np.floating, np.bool_)) else o.item()))
    L = ["# cnn_center_relu_1011 — ReLU の CNN に中心化の扉 C・H・CH", "",
         "Kumar らの CNN・RL-CIFAR・ReLU・seed 10–19・30 課題・主窓 t21–30（spec `specs/spec_cnn_center_relu_1011.md`）。", "",
         "## 腕（主窓 = t21–30 の online、seed ごとに平均してから seed 中央値）", "",
         "| 腕 | ラベル | 主窓 中央値 | 範囲 | KEPT（≥ 0.5） | 床以下 | t1 中央値 | 低下（t1–5 − t26–30）中央値 | T_f 中央値 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a in MAIN_ARMS:
        r = res["arms"][a]
        lab = r["label"] + (f"（{r['fail_subtype']}）" if r.get("fail_subtype") else "")
        L.append(f"| {a} | **{lab}** | {fmt(r.get('W_median'))} | {fmt(r.get('W_min'))}–{fmt(r.get('W_max'))} | "
                 f"{r.get('kept')}/{r.get('n_valid')} | {r.get('at_floor')}/{r.get('n_valid')} | {fmt(r.get('t1_median'))} | "
                 f"{fmt(r.get('drop_median'))} | {fmt(res['timing_T_f'][a]['median'], 1)} |")
    L += ["", "## 登録判定", ""]
    L.append(f"- **DOORS** = `{res['DOORS']}`（C が課題 1 を学ぶ: {res['C_learned_t1']}）")
    L.append(f"- **N** = `{res['N']}`")
    L.append(f"- **M_H** = `{res['M_H']['label']}`（{res['M_H']['vs_MLP']}）")
    lay = res["LAYER"]
    L.append(f"- **LAYER** = `{lay['label']}` — " + ", ".join(f"{k} {v}" for k, v in lay.items() if k != "label"))
    for k, v in res["signs"].items():
        L.append(f"- {k}: `{v['label']}` — 中央値 {fmt(v['median'])}, +{v['pos']}/−{v['neg']}（同値 {v['ties']}）, p {fmt(v['p'])}")
    f = res["flags"]
    L.append(f"- RESIDUAL_LOSS（CH）: {f['RESIDUAL_LOSS_CH']['set']} — 中央値 {fmt(f['RESIDUAL_LOSS_CH']['median'])}, "
             f"+{f['RESIDUAL_LOSS_CH']['pos']}/−{f['RESIDUAL_LOSS_CH']['neg']}, p {fmt(f['RESIDUAL_LOSS_CH']['p'])}")
    for a, t in f["T1_LOWER"].items():
        L.append(f"- T1_LOWER（{a}）: {t['set']} — t1 の差の中央値 {fmt(t['median'])}, +{t['pos']}/−{t['neg']}, p {fmt(t['p'])}")
    for a, t in res["EVALOFF"].items():
        L.append(f"- EVALOFF（{a}）: `{t['label']}` — t30 の memo {fmt(t['memo_t30_median'])}、m = 0 で {fmt(t['memo_m0_t30_median'])}")
    cr = res["Cr"]
    L.append(f"- Cr（seed {CR_SEED}）: `{cr['label']}` — 分ける窓 {cr.get('separating_windows')}, 側 {cr.get('sides')}, Cr の値 "
             + ", ".join(f"{k} {fmt(v)}" for k, v in cr.get("Cr", {}).items()))
    c = res["conditions"]
    L.append(f"- 適用条件: (A) 検査 all_pass {c['A_checks_all_pass']}・有効 seed {c['n_valid']}、(B) ref FAIL {c['B_ref_fail']}、"
             f"(C) H の扉 {{{', '.join(f'{a}: {v['ok']}' for a, v in c['C_h_door'].items())}}}・C の扉の検査 {c['C_c_door_check']}")
    L += ["", "## T_f（網の機能の低下、適合率 < 1/2）と ref との符号検定（報告のみ）", ""]
    for a in MAIN_ARMS:
        t = res["timing_T_f"][a]
        L.append(f"- {a}: 中央値 {fmt(t['median'], 1)}、対 ref `{t['vs_ref']['label']}`（+{t['vs_ref']['pos']}/−{t['vs_ref']['neg']}, p {fmt(t['vs_ref']['p'])}）")
    if len(mech["layers"]):
        L += ["", "## 層ごと（seed 中央値。位置・幅は単位中央値、開いた割合・死亡率は層の平均）", "",
              "| 腕 | 層 | t | z̄ | sd | z̄/sd | 開いた割合 | 死亡率 | ‖m_in‖ |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in mech["layers"].iterrows():
            if r.t in (0, 1, 3, 10, 30):
                L.append(f"| {r.arm} | {r.layer} | {r.t} | {r.zbar_med:.3f} | {r.sd_med:.3f} | {r.rel_pos_med:.2f} | "
                         f"{r.open_mean:.3f} | {r.dead_frac:.3f} | {r.m_in_norm:.3f} |")
        L += ["", "## µ 経路と b 経路（t0 → t、単位中央値の seed 中央値。b の取り分 = |Δb|/(|Δb| + |Δ⟨W, m_in⟩|)）", "",
              "| 腕 | 層 | t | Δz̄ | Δb | Δ⟨W, m_in⟩ | b の取り分 |", "|---|---|---|---|---|---|---|"]
        for _, r in mech["path"].iterrows():
            if r.t in (3, 10, 30):
                L.append(f"| {r.arm} | {r.layer} | {r.t} | {r.dz_med:.3f} | {r.db_med:.3f} | {r.dwm_med:.3f} | {r.b_share_med:.2f} |")
    if leaky:
        L += ["", "## 同じ seed・同じエンジンの leaky（報告のみ）", ""]
        for k, v in leaky.items():
            L.append(f"- {k}: 主窓 {fmt(v['W_median'])}、低下 中央値 {fmt(v['drop_median'])}（平均 {fmt(v['drop_mean'])}）")
    L += ["", "## 予測の採点（Claude、spec §7.1）", "", "| # | 項目 | 実現 | p(実現) | 最頻 | 的中 | Brier |", "|---|---|---|---|---|---|---|"]
    for k, v in sc["labels"].items():
        L.append(f"| {k} | {v['item']} | {v['realised']} | {v['p_realised']:.2f} | {v['mode']} | {'○' if v['mode_hit'] else '✗'} | {v['brier']:.3f} |")
    L += ["", "### Fable（spec §7.2、本走の前の記入）", "", "| # | 項目 | 実現 | p(実現) | 最頻 | 的中 | Brier |",
          "|---|---|---|---|---|---|---|"]
    for k, v in sc["labels_fable"].items():
        L.append(f"| {k} | {v['item']} | {v['realised']} | {v['p_realised']:.2f} | {v['mode']} | {'○' if v['mode_hit'] else '✗'} | {v['brier']:.3f} |")
    L += ["", "### 二値（Claude）", "", "| # | 主張 | p | 真偽 | Brier |", "|---|---|---|---|---|"]
    for k, v in sc["binary"].items():
        truth = "—" if v["true"] is None else ("○" if v["true"] else "✗")
        brier = "—" if v["brier"] is None else f"{v['brier']:.3f}"
        L.append(f"| {k} | {v['claim']} | {v['p']:.2f} | {truth} | {brier} |")
    (base / "summary.md").write_text("\n".join(L) + "\n")


def main() -> None:
    frames, crf = load_frames()
    checks_ok, c_ok = checks_state()
    res = analyze_frames(frames, checks_ok, steps_per_task(), cr_frame=crf, c_door_ok=c_ok)
    mech = mechanism_tables()
    leaky = leaky_reference()
    sc = score(res, mech, leaky)
    write_report(res, mech, leaky, sc)
    print(json.dumps({k: (v if isinstance(v, str) else v.get("label") if isinstance(v, dict) else v)
                      for k, v in res.items() if k in ("N", "DOORS", "M_H", "LAYER", "Cr")}, default=str))
    print({a: res["arms"][a]["label"] for a in MAIN_ARMS})


if __name__ == "__main__":
    main()

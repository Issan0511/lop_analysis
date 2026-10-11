#!/usr/bin/env python3
"""Registered verdict of cnn_center_feat_1011 (spec §5) plus the report-only tables (§5.10).

    python analysis/cnn_center_feat_1011/verdict.py       # -> results/cnn_center_feat_1011/{verdict.json,summary.md,tables/}

New arms (Hf = H', CHf = CH', controls Hs / CHs) come from results/cnn_center_feat_1011/<arm>/;
the reused P181 arms (ref, C, H, CH) from results/cnn_center_relu_1011/<arm>/per_task.csv, which must
be the blobs of commit b9ae5a26 (§5.4 (A)).  Labels come only from per_task.csv (git), checks.json and
the provenance; the mechanism tables also read units.npz / checkpoints (git-external: the run
directory, or the backups in obsidian-research-data after §9).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "cnn_center_feat_1011"
BACKUP = Path.home() / "Projects" / "obsidian-research-data" / "cnn_center_feat_1011" / "results" / "cnn_center_feat_1011"
P181 = REPO / "results" / "cnn_center_relu_1011"
P181_BACKUP = Path.home() / "Projects" / "obsidian-research-data" / "cnn_center_relu_1011" / "results" / "cnn_center_relu_1011"
P181_COMMIT = "b9ae5a26"
NEW_ARMS = ("Hf", "CHf")
REUSED = ("ref", "C", "H", "CH")
ALL_ARMS = REUSED + NEW_ARMS
H_ARMS = ("H", "CH", "Hf", "CHf")
CTRL = {"Hs": ("Hf", "H"), "CHs": ("CHf", "CH")}       # control -> (new arm, P181 arm) it separates
CTRL_NAMES = {"Hs": ("SHUFFLE_ACTS_LIKE_H", "SHUFFLE_ACTS_LIKE_HF"),
              "CHs": ("SHUFFLE_ACTS_LIKE_CH", "SHUFFLE_ACTS_LIKE_CHF")}
SEEDS = tuple(range(10, 20))
CTRL_SEED = 10
N_TASKS = 30
WINDOW = (21, 30)                      # main window (spec §4)
SIDE_WINDOWS = ((1, 5), (6, 10), (11, 20))
EARLY, LATE = (1, 5), (26, 30)         # drop = mean(EARLY) - mean(LATE)
CTRL_WINDOWS = ((1, 5), (6, 10), (11, 20), (21, 30))
KEEP_LINE = 0.5
CHANCE = 0.1
FLOOR_SIGMAS = 3.0
TF_RATIO = 0.5
TD_OPEN = 0.01
ALPHA = 0.05
LAYERS = ("c1", "c2", "f1", "f2")
CKPT = (1, 2, 3, 5, 10, 20, 30)
SIDE = {"c2": 16, "f1": 8}             # pool-output side of the input of c2, f1
CH_ = 16
STEPS_PER_TASK = 400 * 75


# --------------------------------------------------------------------------
# statistics (P181's)
# --------------------------------------------------------------------------

def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    k = max(k, n - k)
    return min(1.0, 2.0 * sum(comb(n, j) for j in range(k, n + 1)) / 2.0 ** n)


def kstar(n: int) -> int | None:
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


def t_f(online: dict) -> float:
    """First task t >= 2 with fit ratio (A(t) - 0.1)/(A(1) - 0.1) < 1/2; 31 when never; NaN when A(1) <= 0.1."""
    a1 = online.get(1, float("nan"))
    if not np.isfinite(a1) or a1 - CHANCE <= 0:
        return float("nan")
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
    out = []
    for seed, g in df.groupby("seed"):
        g = g.sort_values("task")
        tasks = set(int(t) for t in g.task)
        complete = tasks >= set(range(1, N_TASKS + 1))
        finite = bool(np.isfinite(g.online_acc).all() and np.isfinite(g.memo_acc).all())
        diverged = bool(g.diverged.astype(str).str.lower().eq("true").any()) if "diverged" in g else False
        valid = complete and finite and not diverged
        on = {int(t): float(v) for t, v in zip(g.task, g.online_acc)}
        why = ",".join(w for w, bad in (("incomplete", not complete), ("non-finite", not finite),
                                        ("diverged", diverged)) if bad)
        row = {"seed": int(seed), "valid": valid, "invalid_reason": why}
        if complete:
            W = window_mean(g, "online_acc", WINDOW)
            mf = g[(g.task >= WINDOW[0]) & (g.task <= WINDOW[1])].major_frac.to_numpy(float)
            floor = float(mf.mean() + FLOOR_SIGMAS * mf.std(ddof=1) / math.sqrt(len(mf)))
            row |= {"W": W, "kept": W >= KEEP_LINE, "floor": floor, "at_floor": W <= floor,
                    "t1": on[1], "early": window_mean(g, "online_acc", EARLY),
                    "late": window_mean(g, "online_acc", LATE), "T_f": t_f(on)}
            row["drop"] = row["early"] - row["late"]
            for w in SIDE_WINDOWS + CTRL_WINDOWS:
                row[f"W{w[0]}_{w[1]}"] = window_mean(g, "online_acc", w)
            tds = {}
            for l in LAYERS:
                mob = {int(t): float(v) for t, v in zip(g.task, g[f"mob_{l}"])}
                tds[l] = t_d(mob)
                row[f"T_d_{l}"] = tds[l]
            mn = min(tds.values())
            row["first_layers"] = ",".join(l for l in LAYERS if tds[l] == mn) if mn <= N_TASKS else ""
            first = g[g.task == 1].iloc[0]
            last = g[g.task == N_TASKS].iloc[0]
            row["memo_t30"] = float(last.memo_acc)
            if "memo_acc_m0" in g:
                row["memo_m0_t30"] = float(last.memo_acc_m0)
            if "logit_sd" in g:
                row["logit_sd_t1"] = float(first.logit_sd)
            for c in ("pdc_pre_f1", "pdc_post_f1"):
                if c in g:
                    row[f"{c}_t1"] = float(first[c])
            for l in LAYERS:
                row[f"mob_t30_{l}"] = float(last[f"mob_{l}"])
                row[f"dead_t30_{l}"] = float(last[f"dead_frac_{l}"])
        out.append(row)
    return pd.DataFrame(out).set_index("seed")


def door_record(df: pd.DataFrame) -> dict:
    """(C') (spec §5.4): every seed and layer has h_upd >= 1 in task 1 (the EMA moved during task 1).
    Later tasks are not part of the condition; their shortfalls are listed separately (report only)."""
    bad = []
    for l in LAYERS:
        col = f"h_upd_{l}"
        if col not in df:
            return {"ok": False, "reason": f"missing {col}", "violations": [], "n_violations": 0}
        t1 = df[df.task == 1]
        for _, r in t1[~(t1[col] >= 1)].iterrows():
            bad.append((int(r.seed), 1, l, r[col]))
        seeds_t1 = set(int(x) for x in t1.seed)
        if seeds_t1 != set(int(x) for x in df.seed):
            bad.append(("missing task 1", sorted(set(int(x) for x in df.seed) - seeds_t1)))
    return {"ok": not bad, "violations": bad[:20], "n_violations": len(bad)}


def record_shortfalls(df: pd.DataFrame, arm: str) -> list[dict]:
    """Report only: every (seed, task, layer) with h_upd short of the full count and a layer not fully
    dead (P181's (C) record), with the dead fraction."""
    out = []
    for l in LAYERS:
        col, dcol = f"h_upd_{l}", f"dead_frac_{l}"
        if col not in df:
            continue
        for _, r in df[(df[col] != STEPS_PER_TASK) & (df[dcol] != 1.0)].iterrows():
            out.append({"arm": arm, "layer": l, "seed": int(r.seed), "task": int(r.task),
                        "h_upd": int(r[col]), "dead_frac": float(r[dcol])})
    return out


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


def timing(st_a: pd.DataFrame, st_b: pd.DataFrame) -> dict:
    """T_f of a against b (spec §5.7): LATER = a's T_f larger (HELPS direction), EARLIER, NO_TIMING_DIFF."""
    t = paired(st_a, st_b, "T_f")
    return {**t, "label": {"HELPS": "LATER", "HURTS": "EARLIER"}.get(t["label"], "NO_TIMING_DIFF")}


def feature_door(labels: dict, cond_ok: bool, contrast_ok: bool) -> str:
    """Spec §5.5, evaluated in order."""
    if not cond_ok:
        return "INAPPLICABLE"
    if not contrast_ok:
        return "NOT_REPRODUCED"
    if labels["CHf"] == "RESCUED":
        return "FEATURE_CENTERING_SUFFICES"
    if labels["CHf"] == "FAIL" and labels["Hf"] == "FAIL":
        return "CENTERING_NOT_A_DOOR_IN_CNN"
    return "FEATURE_CENTERING_PARTIAL"


def ctrl_label(name: str, row: pd.Series | None, st_new: pd.DataFrame, st_old: pd.DataFrame) -> dict:
    """Spec §5.9: separating windows = the windows where the 10-seed ranges of the new arm (Hf / CHf)
    and the P181 arm (H / CH) do not overlap; the control's seed must fall on the P181 side in all of
    them (ACTS_LIKE_H / CH) or on the new side in all (ACTS_LIKE_HF / CHF)."""
    like_old, like_new = CTRL_NAMES[name]
    if row is None:
        return {"label": "INAPPLICABLE", "reason": "control seed missing, incomplete or not finite"}
    vn, vo = st_new[st_new.valid], st_old[st_old.valid]
    sep, sides = [], {}
    for w in CTRL_WINDOWS:
        col = f"W{w[0]}_{w[1]}"
        n_lo, n_hi, o_lo, o_hi = vn[col].min(), vn[col].max(), vo[col].min(), vo[col].max()
        if n_hi < o_lo or o_hi < n_lo:
            sep.append(col)
            v = float(row[col])
            sides[col] = "old" if o_lo <= v <= o_hi else ("new" if n_lo <= v <= n_hi else "outside")
    if not sep:
        lab = "SHUF_UNINFORMATIVE"
    elif all(s == "old" for s in sides.values()):
        lab = like_old
    elif all(s == "new" for s in sides.values()):
        lab = like_new
    else:
        lab = "SHUF_UNRESOLVED"
    return {"label": lab, "separating_windows": sep, "sides": sides,
            "value": {f"W{w[0]}_{w[1]}": float(row[f"W{w[0]}_{w[1]}"]) for w in CTRL_WINDOWS}}


def evaloff(st: pd.DataFrame) -> dict:
    v = st[st.valid]
    t = sign_test(v.memo_t30 - v.memo_m0_t30)
    return {**t, "label": "EVALOFF_DEPENDS" if t["label"] == "HELPS" else "EVALOFF_NO_DETECTED_DEPENDENCE",
            "sign_label": t["label"], "memo_t30_median": float(v.memo_t30.median()),
            "memo_m0_t30_median": float(v.memo_m0_t30.median())}


def analyze_frames(frames: dict, checks_ok: bool, ctrl_frames: dict | None = None, c_door_ok: bool = True) -> dict:
    """frames: arm -> per_task DataFrame for ref, C, H, CH (P181) and Hf, CHf (new);
    ctrl_frames: Hs / CHs -> DataFrame (seed 10) or None (not run).  Returns every registered label."""
    st = {a: seed_table(frames[a]) for a in ALL_ARMS}
    arms = {a: arm_label(st[a]) for a in ALL_ARMS}
    rec = {a: door_record(frames[a]) for a in H_ARMS}
    for a in H_ARMS:
        if not rec[a]["ok"]:
            arms[a] = arms[a] | {"label": "INAPPLICABLE", "condition_C": rec[a]}
    if not c_door_ok:
        arms["CHf"] = arms["CHf"] | {"label": "INAPPLICABLE", "condition_C": "C door check failed"}
    labels = {a: arms[a]["label"] for a in ALL_ARMS}
    cond_ok = checks_ok and all(labels[a] != "INAPPLICABLE" for a in NEW_ARMS)
    contrast_ok = labels["ref"] == "FAIL" and labels["CH"] == "FAIL"
    signs = {"D_CHf|CH": paired(st["CHf"], st["CH"]), "D_Hf|H": paired(st["Hf"], st["H"]),
             "D_C|Hf": paired(st["CHf"], st["Hf"]), "D_CHf": paired(st["CHf"], st["ref"]),
             "D_Hf": paired(st["Hf"], st["ref"])}
    res = {
        "conditions": {"A_checks_and_gates": checks_ok, "n_valid": {a: arms[a].get("n_valid") for a in ALL_ARMS},
                       "B_contrast_ref_and_CH_FAIL": contrast_ok, "C_door_record": rec, "C_c_door": c_door_ok},
        "arms": arms,
        "FEATURE_DOOR": feature_door(labels, cond_ok, contrast_ok),
        "M_Hf": {"label": labels["Hf"],
                 "vs_MLP": {"FAIL": "HF_LIKE_MLP", "RESCUED": "HF_DIFFERS_FROM_MLP",
                            "PARTIAL": "HF_PARTIAL"}.get(labels["Hf"], labels["Hf"])},
        "signs": signs,
        "timing": {"medians": {a: float(st[a][st[a].valid].T_f.median()) if len(st[a][st[a].valid]) else float("nan")
                               for a in ALL_ARMS},
                   "CHf_vs_CH": timing(st["CHf"], st["CH"]), "CHf_vs_ref": timing(st["CHf"], st["ref"]),
                   "Hf_vs_H": timing(st["Hf"], st["H"]), "Hf_vs_ref": timing(st["Hf"], st["ref"])},
        "flags": {
            "RESIDUAL_LOSS": {a: (lambda t: {"set": t["label"] == "HELPS", **t})(sign_test(st[a][st[a].valid]["drop"]))
                              for a in NEW_ARMS},
            "T1_LOWER": {a: (lambda t: {"set": t["label"] == "HURTS", **t})(paired(st[a], st["ref"], "t1"))
                         for a in NEW_ARMS},
        },
        "EVALOFF": {a: evaloff(st[a]) for a in NEW_ARMS},
    }
    res["ctrl"] = {}
    for name, (new, old) in CTRL.items():
        f = None if ctrl_frames is None else ctrl_frames.get(name)
        if f is None:
            res["ctrl"][name] = {"label": "NOT_RUN"}
            continue
        cst = seed_table(f)
        ok = CTRL_SEED in cst.index and bool(cst.loc[CTRL_SEED, "valid"]) and \
            all(np.isfinite(cst.loc[CTRL_SEED, f"W{w[0]}_{w[1]}"]) for w in CTRL_WINDOWS)
        res["ctrl"][name] = ctrl_label(name, cst.loc[CTRL_SEED] if ok else None, st[new], st[old])
        res["ctrl"][name]["seed_row"] = cst.reset_index().to_dict("records")
    res["_seed_tables"] = st
    return res


# --------------------------------------------------------------------------
# report-only tables (units.npz, checkpoints)
# --------------------------------------------------------------------------

def base_of(arm: str) -> tuple[Path, Path]:
    return (P181, P181_BACKUP) if arm in REUSED else (OUT, BACKUP)


def find_units(arm: str) -> Path | None:
    for base in base_of(arm):
        p = base / arm / "units.npz"
        if p.exists():
            return p
    return None


def find_ckpt(arm: str) -> Path | None:
    for base in base_of(arm):
        p = base / arm / "ckpt"
        if p.exists() and any(p.glob("*.pt")):
            return p
    return None


def mechanism_tables(arms=ALL_ARMS) -> dict:
    """mu path / b path (t0 -> t and t1 -> t) per layer: unit median of dz, db, dwm and the b share
    |db| / (|db| + |dwm|), then the seed median.  Also zbar / sd / open / dead per checkpoint."""
    rows, lay = [], []
    for a in arms:
        p = find_units(a)
        if p is None:
            continue
        u = dict(np.load(p))
        for l in LAYERS:
            zb, b, wm = u[f"zbar_{l}"], u[f"b_{l}"], u[f"wm_{l}"]           # (T+1, R, n)
            for t_from in (0, 1):
                for t in CKPT:
                    if t <= t_from or t >= zb.shape[0] or not np.isfinite(zb[t]).all():
                        continue
                    dz, db, dw = zb[t] - zb[t_from], b[t] - b[t_from], wm[t] - wm[t_from]
                    share = np.abs(db) / (np.abs(db) + np.abs(dw) + 1e-30)
                    rows.append({"arm": a, "layer": l, "from": t_from, "t": t,
                                 "dz_med": float(np.median(np.median(dz, 1))),
                                 "db_med": float(np.median(np.median(db, 1))),
                                 "dwm_med": float(np.median(np.median(dw, 1))),
                                 "b_share_med": float(np.median(np.median(share, 1)))})
            for t in (0,) + CKPT:
                if t >= zb.shape[0] or not np.isfinite(zb[t]).all():
                    continue
                sd, op, dead = u[f"sd_{l}"][t], u[f"open_{l}"][t], u[f"dead_{l}"][t]
                lay.append({"arm": a, "layer": l, "t": t,
                            "zbar_med": float(np.median(np.median(zb[t], 1))),
                            "sd_med": float(np.median(np.median(sd, 1))),
                            "rel_pos_med": float(np.median(np.median(zb[t] / np.maximum(sd, 1e-30), 1))),
                            "open_mean": float(np.median(op.mean(1))),
                            "dead_frac": float(np.median(dead.mean(1))),
                            "m_in_norm": float(np.median(np.linalg.norm(u[f"m_in_{l}"][t], axis=1)))})
    return {"path": pd.DataFrame(rows), "layers": pd.DataFrame(lay)}


def three_way_table(arms=ALL_ARMS, tasks=(1, 3, 10, 30)) -> pd.DataFrame:
    """Report only: the mu path <W, m_in> from t0 to t split into own <dW, m0>, upstream <W0, dm>
    and cross <dW, dm> (an exact identity), unit median per seed, then seed median."""
    import torch
    sys.path.insert(0, str(REPO))
    from src import rlcifar_cnn_0908 as CN
    rows = []
    for a in arms:
        up, ck = find_units(a), find_ckpt(a)
        if up is None or ck is None:
            continue
        u = dict(np.load(up))
        R = u["m_in_c1"].shape[1]
        seeds = SEEDS if R == len(SEEDS) else (CTRL_SEED,) if R == 1 else tuple(range(10, 10 + R))
        for si, seed in enumerate(seeds):
            P0 = [q.detach().double() for q in CN.init_params(seed, torch.device("cpu"))]
            for t in tasks:
                f = ck / f"{a}_seed{seed}_t{t:02d}.pt"
                if not f.exists():
                    continue
                Pt = [q.double() for q in torch.load(f, map_location="cpu", weights_only=False)["P"]]
                for li, l in enumerate(LAYERS):
                    W0 = P0[2 * li].reshape(P0[2 * li].shape[0], -1)
                    Wt = Pt[2 * li].reshape(Pt[2 * li].shape[0], -1)
                    m0 = torch.from_numpy(u[f"m_in_{l}"][0, si]).double()
                    mt = torch.from_numpy(u[f"m_in_{l}"][t, si]).double()
                    dW, dm = Wt - W0, mt - m0
                    own, ups, cross = dW @ m0, W0 @ dm, dW @ dm
                    total = Wt @ mt - W0 @ m0
                    rows.append({"arm": a, "seed": seed, "t": t, "layer": l,
                                 "own_med": float(np.median(own.numpy())), "up_med": float(np.median(ups.numpy())),
                                 "cross_med": float(np.median(cross.numpy())), "dwm_med": float(np.median(total.numpy())),
                                 "identity_res": float((own + ups + cross - total).abs().max())})
    d = pd.DataFrame(rows)
    if not len(d):
        return d
    return d.groupby(["arm", "layer", "t"])[["own_med", "up_med", "cross_med", "dwm_med", "identity_res"]].agg(
        {"own_med": "median", "up_med": "median", "cross_med": "median", "dwm_med": "median",
         "identity_res": "max"}).reset_index()


def dc_parts_np(mu: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mu (R, C*side*side), (c, y, x) order -> |per-position part|, |channel part over positions|, |mu|."""
    R = mu.shape[0]
    mc = mu.reshape(R, CH_, side * side).astype(np.float64)
    cbar = mc.mean(2, keepdims=True)
    delta = (mc - cbar).reshape(R, -1)
    return np.linalg.norm(delta, axis=1), np.linalg.norm(cbar[:, :, 0], axis=1) * side, \
        np.linalg.norm(mc.reshape(R, -1), axis=1)


def pdc_table(frames: dict, ctrl_frames: dict | None = None) -> pd.DataFrame:
    """Report only (spec §5.10): the per-position mean of the inputs of c2 and f1 before / after the
    door, over the rms deviation.  New runs: their per_task / t0 columns.  P181 arms: f1 only, from
    units.npz m_in_f1 (= mu_post) and r_in_f1 (sigma = |m_in| / r_in); pre = post by construction
    (no door, or a per-channel mean that leaves the per-position part alone)."""
    rows = []
    allf = dict(frames)
    for k, v in (ctrl_frames or {}).items():
        if v is not None:
            allf[k] = v
    for a, df in allf.items():
        t0p = base_of(a)[0] / a / "t0.csv"
        d = pd.concat([pd.read_csv(t0p), df], ignore_index=True) if t0p.exists() else df
        if a in REUSED:
            up = find_units(a)
            if up is None:
                continue
            m_in = np.load(up)["m_in_f1"]                                     # (T+1, R, 1024)
            for t in (0,) + CKPT:
                g = d[d.task == t].sort_values("seed")
                if t >= m_in.shape[0] or len(g) != m_in.shape[1]:
                    continue
                pd_, cd_, dc_ = dc_parts_np(m_in[t], SIDE["f1"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    sig = dc_ / g.r_in_f1.to_numpy(float)
                for si, seed in enumerate(g.seed):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rows.append({"arm": a, "seed": int(seed), "task": t, "layer": "f1",
                                     "pdc_pre": pd_[si] / sig[si], "pdc_post": pd_[si] / sig[si],
                                     "cdc_post": cd_[si] / sig[si], "dc_post": dc_[si] / sig[si],
                                     "source": "P181 units.npz (pre = post)"})
        else:
            for l in ("c2", "f1"):
                if f"pdc_post_{l}" not in d:
                    continue
                for _, r in d[d.task.isin((0,) + CKPT)].iterrows():
                    rows.append({"arm": a, "seed": int(r.seed), "task": int(r.task), "layer": l,
                                 "pdc_pre": float(r[f"pdc_pre_{l}"]), "pdc_post": float(r[f"pdc_post_{l}"]),
                                 "cdc_post": float(r[f"cdc_post_{l}"]), "dc_post": float(r[f"dc_post_{l}"]),
                                 "source": "per_task"})
    return pd.DataFrame(rows)


def pdc_summary(tab: pd.DataFrame) -> pd.DataFrame:
    if not len(tab):
        return tab
    t = tab.copy()
    t["ratio"] = t.pdc_post / t.pdc_pre
    return t.groupby(["arm", "layer", "task"])[["pdc_pre", "pdc_post", "ratio", "cdc_post", "dc_post"]].median().reset_index()


# --------------------------------------------------------------------------
# prediction scoring (spec §7)
# --------------------------------------------------------------------------

PRED_LABELS = {
    "L1": ("CHf", {"RESCUED": .30, "PARTIAL": .30, "FAIL": .37, "OTHER": .03}),
    "L2": ("Hf", {"RESCUED": .10, "PARTIAL": .20, "FAIL": .67, "OTHER": .03}),
    "L3": ("FEATURE_DOOR", {"FEATURE_CENTERING_SUFFICES": .30, "FEATURE_CENTERING_PARTIAL": .33,
                            "CENTERING_NOT_A_DOOR_IN_CNN": .34, "OTHER": .03}),
    "L4": ("T_CHf_CH", {"LATER": .62, "NO_TIMING_DIFF": .35, "EARLIER": .03}),
    "L5": ("T_CHf_ref", {"LATER": .85, "NO_TIMING_DIFF": .14, "EARLIER": .01}),
    "L6": ("T_Hf_H", {"LATER": .62, "NO_TIMING_DIFF": .38, "EARLIER": .00}),
    "L7": ("Hs", {"SHUFFLE_ACTS_LIKE_H": .45, "SHUF_UNINFORMATIVE": .30, "SHUF_UNRESOLVED": .20,
                  "SHUFFLE_ACTS_LIKE_HF": .05}),
    "L8": ("CHs", {"SHUFFLE_ACTS_LIKE_CH": .40, "SHUF_UNINFORMATIVE": .30, "SHUF_UNRESOLVED": .25,
                   "SHUFFLE_ACTS_LIKE_CHF": .05}),
}
# Fable (parent), from the request, before any H' run (spec §7.1); NOT_A_DOOR scored with §5.5's rule 4
PRED_LABELS_FABLE = {
    "L1": ("CHf", {"RESCUED": .50, "PARTIAL": .25, "FAIL": .25}),
    "L2": ("Hf", {"RESCUED": .30, "PARTIAL": .30, "FAIL": .40}),
    "L3": ("FEATURE_DOOR", {"FEATURE_CENTERING_SUFFICES": .45, "FEATURE_CENTERING_PARTIAL": .25,
                            "CENTERING_NOT_A_DOOR_IN_CNN": .30}),
}
# Codex: transcribed after the audit (spec §7.2)
PRED_LABELS_CODEX = {
    "L1": ("CHf", {"RESCUED": .45, "PARTIAL": .28, "FAIL": .24, "OTHER": .03}),
    "L2": ("Hf", {"RESCUED": .15, "PARTIAL": .25, "FAIL": .57, "OTHER": .03}),
    "L3": ("FEATURE_DOOR", {"FEATURE_CENTERING_SUFFICES": .45, "FEATURE_CENTERING_PARTIAL": .32,
                            "CENTERING_NOT_A_DOOR_IN_CNN": .20, "OTHER": .03}),
    "L4": ("T_CHf_CH", {"LATER": .70, "NO_TIMING_DIFF": .23, "EARLIER": .07}),
}
PRED_BINARY = {"B1": .45, "B2": .35, "B3": .50, "B4": .85, "B5": .15, "B6": .85, "B7": .70, "B8": .25,
               "B9": .85, "B10": .10}
PRED_TEXT = {
    "B1": "CHf's t30 f1 dead fraction (seed median) < 0.5",
    "B2": "CHf's first failing layers include c1 in >= 5 seeds (10 valid seeds only)",
    "B3": "CHf's t1 logit SD (seed median) > 10 x ref's (P181) t1 logit SD",
    "B4": "CHf's t1 pdc_post_f1 (seed median) < 0.05",
    "B5": "CHf's main-window online (seed median) >= 0.90",
    "B6": "RESIDUAL_LOSS set on CHf",
    "B7": "CHf's b-path share t0->t30 > 0.5 in all four layers",
    "B8": "Hf's first failing layers include c1 in >= 9 seeds (10 valid seeds only)",
    "B9": "CHf is EVALOFF_DEPENDS",
    "B10": "T1_LOWER set on CHf",
}


def realised(res: dict) -> dict:
    return {"CHf": res["arms"]["CHf"]["label"], "Hf": res["arms"]["Hf"]["label"],
            "FEATURE_DOOR": res["FEATURE_DOOR"], "T_CHf_CH": res["timing"]["CHf_vs_CH"]["label"],
            "T_CHf_ref": res["timing"]["CHf_vs_ref"]["label"], "T_Hf_H": res["timing"]["Hf_vs_H"]["label"],
            "Hs": res["ctrl"]["Hs"]["label"], "CHs": res["ctrl"]["CHs"]["label"]}


def score_labels(res: dict, table: dict) -> dict:
    lab = realised(res)
    out = {}
    for key, (item, probs) in table.items():
        r = lab[item]
        if item == "CHs" and r == "NOT_RUN":
            out[key] = {"item": item, "realised": r, "unscored": "CHs not run"}
            continue
        bucket = r if r in probs else "OTHER"
        p = probs.get(bucket, 0.0)
        brier = sum((pk - (1.0 if k == bucket else 0.0)) ** 2 for k, pk in probs.items()) + \
            (0.0 if bucket in probs else 1.0)
        top = max(probs, key=probs.get)
        out[key] = {"item": item, "realised": r, "p_realised": p, "mode": top, "mode_hit": top == bucket,
                    "brier": brier}
    return out


def score(res: dict, mech: dict | None, ref_logit_t1: float) -> dict:
    out = {"labels": score_labels(res, PRED_LABELS), "labels_fable": score_labels(res, PRED_LABELS_FABLE),
           "labels_codex": score_labels(res, PRED_LABELS_CODEX), "binary": {}}
    st = res["_seed_tables"]
    v = st["CHf"][st["CHf"].valid]
    vh = st["Hf"][st["Hf"].valid]
    truth: dict = {}
    truth["B1"] = float(v.dead_t30_f1.median()) < 0.5 if len(v) else None
    truth["B2"] = (int(sum("c1" in s.split(",") for s in v.first_layers if s)) >= 5) if len(v) == 10 else None
    truth["B3"] = (float(v.logit_sd_t1.median()) > 10 * ref_logit_t1) if len(v) and "logit_sd_t1" in v else None
    truth["B4"] = (float(v.pdc_post_f1_t1.median()) < 0.05) if len(v) and "pdc_post_f1_t1" in v else None
    truth["B5"] = res["arms"]["CHf"].get("W_median", float("nan")) >= 0.90
    truth["B6"] = res["flags"]["RESIDUAL_LOSS"]["CHf"]["set"]
    if mech is not None and len(mech["path"]):
        pth = mech["path"]
        q = pth[(pth.arm == "CHf") & (pth["from"] == 0) & (pth.t == N_TASKS)]
        truth["B7"] = bool(len(q) == 4 and (q.b_share_med > 0.5).all()) if len(q) else None
    else:
        truth["B7"] = None
    truth["B8"] = (int(sum("c1" in s.split(",") for s in vh.first_layers if s)) >= 9) if len(vh) == 10 else None
    truth["B9"] = res["EVALOFF"]["CHf"]["label"] == "EVALOFF_DEPENDS"
    truth["B10"] = res["flags"]["T1_LOWER"]["CHf"]["set"]
    for k, p in PRED_BINARY.items():
        o = truth.get(k)
        out["binary"][k] = {"claim": PRED_TEXT[k], "p": p, "true": o,
                            "brier": None if o is None else (p - (1.0 if o else 0.0)) ** 2}
    return out


# --------------------------------------------------------------------------
# io, input and provenance gates
# --------------------------------------------------------------------------

REQUIRED = ("online_acc", "memo_acc", "major_frac") + tuple(f"{q}_{l}" for q in ("mob", "dead_frac") for l in LAYERS)
REQUIRED_H = tuple(f"h_upd_{l}" for l in LAYERS) + ("memo_acc_m0",)


def validate_frame(df: pd.DataFrame, arm: str, seeds) -> list[str]:
    """Problems that make the table not the registered box (empty list = valid)."""
    bad = []
    if "arm" not in df or set(df.arm.astype(str)) != {arm}:
        bad.append(f"{arm}: arm column {sorted(set(df.arm.astype(str))) if 'arm' in df else 'missing'}")
    if set(int(x) for x in df.seed) != set(seeds):
        bad.append(f"{arm}: seeds {sorted(set(int(x) for x in df.seed))} != {list(seeds)}")
    for seed, g in df.groupby("seed"):
        t = sorted(int(x) for x in g.task)
        if t != list(range(1, N_TASKS + 1)):
            bad.append(f"{arm}: seed {seed} tasks not exactly 1..{N_TASKS} once (n={len(t)})")
    need = REQUIRED + (REQUIRED_H if arm in H_ARMS + tuple(CTRL) else ())
    miss = [c for c in need if c not in df]
    if miss:
        bad.append(f"{arm}: missing columns {miss}")
    else:
        nonfinite = [c for c in need if not np.isfinite(df[c].to_numpy(float)).all()]
        if nonfinite:
            bad.append(f"{arm}: non-finite values in {nonfinite}")
    return bad


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)


def reused_blobs(base: Path | None = None) -> dict:
    """§5.4 (A): the working-tree per_task.csv of each reused arm is the blob of P181's commit, and
    P181's checks.json (also that blob) is all_pass."""
    base = P181 if base is None else base
    bad = []
    for a in REUSED:
        rel = f"results/cnn_center_relu_1011/{a}/per_task.csv"
        want = git("rev-parse", f"{P181_COMMIT}:{rel}").stdout.decode().strip()
        have = git("hash-object", str(base / a / "per_task.csv")).stdout.decode().strip()
        if not want or want != have:
            bad.append(f"{a}: per_task.csv blob {have[:12]} != {P181_COMMIT}:{want[:12]}")
    chk = git("show", f"{P181_COMMIT}:results/cnn_center_relu_1011/checks.json")
    if chk.returncode != 0 or not json.loads(chk.stdout).get("all_pass"):
        bad.append("P181 checks.json missing or not all_pass")
    return {"ok": not bad, "problems": bad}


def engine_shas_at(commit: str) -> tuple | None:
    out = []
    for f in ("src/cnn_center_feat_1011.py", "src/cnn_center_relu_1011.py"):
        r = git("show", f"{commit}:{f}")
        if r.returncode != 0:
            return None
        out.append(hashlib.sha256(r.stdout).hexdigest())
    return tuple(out)


EXPECTED = {"n_tasks": N_TASKS, "epochs": 400, "lr": 0.001, "beta_h": 0.01}
H_MODE = {"Hf": "feat", "CHf": "feat", "Hs": "shuf", "CHs": "shuf"}


def provenance_gate(base: Path | None = None, arms=NEW_ARMS + tuple(CTRL)) -> dict:
    """Every new arm's provenance.json is the registered configuration, from a clean tree, with the
    engine (this experiment's file and P181's file it imports) that the checks ran on.  The two
    controls were adopted by the parent (spec §12.1): a missing control is a problem too (audit 2)."""
    base = OUT if base is None else base
    bad = []
    c = json.loads((base / "checks.json").read_text()) if (base / "checks.json").exists() else {}
    if not c.get("all_pass"):
        bad.append("checks.json missing or not all_pass")
    chk_commit = c.get("_meta", {}).get("git_hash")
    chk_sha = engine_shas_at(chk_commit) if chk_commit else None
    if c.get("_meta", {}).get("git_dirty_code"):
        bad.append("checks ran on a dirty tree")
    shas, data = set(), set()
    for a in arms:
        pp = base / a / "provenance.json"
        if not pp.exists():
            if a in NEW_ARMS + tuple(CTRL):
                bad.append(f"{a}: provenance.json missing")
            continue
        pv = json.loads(pp.read_text())
        seeds = [CTRL_SEED] if a in CTRL else list(SEEDS)
        if pv.get("arm") != a or pv.get("slots") != [[a, s] for s in seeds]:
            bad.append(f"{a}: provenance arm/slots {pv.get('arm')} {pv.get('slots')}")
        for k, v in EXPECTED.items():
            if pv.get(k) != v:
                bad.append(f"{a}: provenance {k} = {pv.get(k)} != {v}")
        if pv.get("h_mode") != H_MODE[a]:
            bad.append(f"{a}: door mode {pv.get('h_mode')} != {H_MODE[a]}")
        for g in pv.get("gits", []):
            if g.get("git_dirty_code"):
                bad.append(f"{a}: run from a dirty tree ({g.get('git_hash')})")
            shas.add(engine_shas_at(g.get("git_hash")))
        data.add(json.dumps(pv.get("data_sha256"), sort_keys=True))
    if chk_sha is None or shas != {chk_sha}:
        bad.append(f"engine sha256 of runs {sorted(str(x)[:40] for x in shas)} != checks {str(chk_sha)[:40]}")
    if len(data) != 1:
        bad.append("data sha256 differs between arms")
    return {"ok": not bad, "problems": bad, "engine_sha256": chk_sha}


def c_door_from_runs() -> dict:
    """(C) for CHf from the runs: the t0 mean input patch of c1 is < 0.1 x Hf's (raw input), per seed."""
    pr, pc = find_units("Hf"), find_units("CHf")
    if pr is None or pc is None:
        return {"ok": None, "reason": "units.npz of Hf or CHf not found"}
    raw = np.linalg.norm(np.load(pr)["m_in_c1"][0], axis=1)
    cen = np.linalg.norm(np.load(pc)["m_in_c1"][0], axis=1)
    return {"ok": bool((cen < 0.1 * raw).all()), "ratio_max": float((cen / raw).max())}


def control_problems(ctrl: dict) -> list[str]:
    """Both controls were adopted by the parent (spec §12.1): each must be there and valid (audit 2)."""
    out = []
    for a in CTRL:
        if ctrl.get(a) is None:
            out.append(f"{a}: per_task.csv missing (control adopted by the parent, spec §12.1)")
        else:
            out += validate_frame(ctrl[a], a, (CTRL_SEED,))
    return out


def c_door_ok_from(cdoor: dict) -> bool:
    """Fail closed (audit 1): the C door counts as checked only when the runs' units.npz say so."""
    return cdoor.get("ok") is True


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def required_mechanism_inputs() -> list[str]:
    """Audit 3: every artifact the report-only tables need (empty list = complete).  New arms and the
    controls: units.npz with tasks 0..30 and every checkpoint (CKPT tasks x seeds), in OUT or in the
    backup (whose sha256 must match this run's backup_manifest.json when the files were moved).
    Reused P181 arms: the backup units.npz and checkpoints, sha256 as P181's backup_manifest.json says."""
    bad = []
    own = OUT / "backup_manifest.json"
    own_man = {r["backup"]: r["sha256"] for r in json.loads(own.read_text())["files"]} if own.exists() else {}

    def check_own(f: Path) -> None:
        if str(f).startswith(str(BACKUP)) and own_man.get(str(f)) not in (None, sha256_file(f)):
            bad.append(f"{f}: sha256 differs from backup_manifest.json")
        if str(f).startswith(str(BACKUP)) and str(f) not in own_man:
            bad.append(f"{f}: not in backup_manifest.json")

    for a in NEW_ARMS + tuple(CTRL):
        seeds = (CTRL_SEED,) if a in CTRL else SEEDS
        up, ck = find_units(a), find_ckpt(a)
        if up is None:
            bad.append(f"{a}: units.npz missing")
        else:
            td = [int(x) for x in np.load(up)["tasks_done"]]
            if td != list(range(0, N_TASKS + 1)):
                bad.append(f"{a}: units.npz tasks_done {td[:2]}..{td[-2:]}")
            check_own(up)
        if ck is None:
            bad.append(f"{a}: ckpt missing")
            continue
        for sd in seeds:
            for t in CKPT:
                f = ck / f"{a}_seed{sd}_t{t:02d}.pt"
                if not f.exists():
                    bad.append(f"{a}: {f.name} missing")
                else:
                    check_own(f)
    man_p = P181 / "backup_manifest.json"
    man = {r["source"]: r for r in json.loads(man_p.read_text())["files"]} if man_p.exists() else {}
    for a in REUSED:
        need = [f"results/cnn_center_relu_1011/{a}/units.npz"] + [
            f"results/cnn_center_relu_1011/{a}/ckpt/{a}_seed{sd}_t{t:02d}.pt" for sd in SEEDS for t in CKPT]
        for rel in need:
            r = man.get(rel)
            if r is None:
                bad.append(f"{a}: {rel} not in P181's backup_manifest.json")
            elif not Path(r["backup"]).exists():
                bad.append(f"{a}: {r['backup']} missing")
            elif sha256_file(Path(r["backup"])) != r["sha256"]:
                bad.append(f"{a}: {Path(r['backup']).name} sha256 differs from P181's manifest")
    return bad


def commit_time(h: str | None) -> str:
    if not h:
        return "—"
    r = git("show", "-s", "--format=%h %ci", h)
    return r.stdout.decode().strip() if r.returncode == 0 else f"{h[:8]} (not found)"


def checks_ok_state(base: Path | None = None) -> bool:
    p = (OUT if base is None else base) / "checks.json"
    return bool(json.loads(p.read_text()).get("all_pass")) if p.exists() else False


def load_frames() -> tuple[dict, dict]:
    frames = {a: pd.read_csv(P181 / a / "per_task.csv") for a in REUSED}
    frames |= {a: pd.read_csv(OUT / a / "per_task.csv") for a in NEW_ARMS}
    ctrl = {}
    for a in CTRL:
        p = OUT / a / "per_task.csv"
        ctrl[a] = pd.read_csv(p) if p.exists() else None
    return frames, ctrl


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (
        f"{x:.{nd}f}" if isinstance(x, float) else str(x))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def write_report(res: dict, mech: dict, pdc: pd.DataFrame, short: list, sc: dict, base: Path | None = None) -> None:
    base = OUT if base is None else base
    (base / "tables").mkdir(parents=True, exist_ok=True)
    for a, s in res["_seed_tables"].items():
        s.to_csv(base / "tables" / f"seeds_{a}.csv")
    if len(mech["path"]):
        mech["path"].to_csv(base / "tables" / "mu_b_path.csv", index=False)
        mech["layers"].to_csv(base / "tables" / "layers.csv", index=False)
    if len(pdc):
        pdc.to_csv(base / "tables" / "pos_dc.csv", index=False)
    pds = pdc_summary(pdc)
    sh = pd.DataFrame(short)
    if len(sh):
        sh.to_csv(base / "tables" / "record_shortfalls.csv", index=False)
    clean = {k: v for k, v in res.items() if k != "_seed_tables"}
    clean["prediction_scoring"] = sc
    clean["record_shortfalls_n"] = {a: int((sh.arm == a).sum()) if len(sh) else 0 for a in H_ARMS}
    (base / "verdict.json").write_text(json.dumps(clean, indent=1, default=lambda o: o if not isinstance(
        o, (np.integer, np.floating, np.bool_)) else o.item()))
    L = ["# cnn_center_feat_1011 — ReLU の CNN で全特徴（チャネル × 位置）の中心化 H′", "",
         "Kumar らの CNN・RL-CIFAR・ReLU・seed 10–19・30 課題・主窓 t21–30（spec `specs/spec_cnn_center_feat_1011.md`）。"
         "ref・C・H・CH は P181（`cnn_center_relu_1011`、`b9ae5a26`）の走の再利用（bit 照合）。", "",
         "## 腕（主窓 = t21–30 の online、seed ごとに平均してから seed 中央値）", "",
         "| 腕 | ラベル | 主窓 中央値 | 範囲 | KEPT（≥ 0.5） | 床以下 | t1 中央値 | 低下（t1–5 − t26–30）中央値 | T_f 中央値 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a in NEW_ARMS + REUSED:
        r = res["arms"][a]
        lab = r["label"] + (f"（{r['fail_subtype']}）" if r.get("fail_subtype") and r["label"] == "FAIL" else "")
        src = "" if a in NEW_ARMS else "（P181）"
        L.append(f"| {a}{src} | **{lab}** | {fmt(r.get('W_median'))} | {fmt(r.get('W_min'))}–{fmt(r.get('W_max'))} | "
                 f"{r.get('kept')}/{r.get('n_valid')} | {r.get('at_floor')}/{r.get('n_valid')} | {fmt(r.get('t1_median'))} | "
                 f"{fmt(r.get('drop_median'))} | {fmt(res['timing']['medians'][a], 1)} |")
    L += ["", "## 登録判定", ""]
    L.append(f"- **FEATURE_DOOR** = `{res['FEATURE_DOOR']}`")
    L.append(f"- **M_Hf** = `{res['M_Hf']['label']}`（{res['M_Hf']['vs_MLP']}）")
    for k, v in res["signs"].items():
        L.append(f"- {k}: `{v['label']}` — 中央値 {fmt(v['median'])}, +{v['pos']}/−{v['neg']}（同値 {v['ties']}）, p {fmt(v['p'])}")
    for k in ("CHf_vs_CH", "CHf_vs_ref", "Hf_vs_H", "Hf_vs_ref"):
        v = res["timing"][k]
        L.append(f"- T_f {k}: `{v['label']}` — 差の中央値 {fmt(v['median'], 1)}, +{v['pos']}/−{v['neg']}（同値 {v['ties']}）, p {fmt(v['p'])}")
    for a in NEW_ARMS:
        f = res["flags"]
        L.append(f"- RESIDUAL_LOSS（{a}）: {f['RESIDUAL_LOSS'][a]['set']} — 中央値 {fmt(f['RESIDUAL_LOSS'][a]['median'])}, "
                 f"+{f['RESIDUAL_LOSS'][a]['pos']}/−{f['RESIDUAL_LOSS'][a]['neg']}, p {fmt(f['RESIDUAL_LOSS'][a]['p'])}")
        L.append(f"- T1_LOWER（{a}）: {f['T1_LOWER'][a]['set']} — t1 の差の中央値 {fmt(f['T1_LOWER'][a]['median'])}, "
                 f"+{f['T1_LOWER'][a]['pos']}/−{f['T1_LOWER'][a]['neg']}, p {fmt(f['T1_LOWER'][a]['p'])}")
        t = res["EVALOFF"][a]
        L.append(f"- EVALOFF（{a}）: `{t['label']}` — t30 の memo {fmt(t['memo_t30_median'])}、m = 0 で {fmt(t['memo_m0_t30_median'])}")
    for name, c in res["ctrl"].items():
        L.append(f"- 対照 {name}（seed {CTRL_SEED}）: `{c['label']}` — 分ける窓 {c.get('separating_windows')}, 側 {c.get('sides')}, "
                 f"値 " + ", ".join(f"{k} {fmt(v)}" for k, v in c.get("value", {}).items()))
    c = res["conditions"]
    L.append(f"- 適用条件: (A) 検査・関門 {c['A_checks_and_gates']}（表の問題 {c.get('validation_problems') or 'なし'}、"
             f"provenance {('OK' if c.get('provenance_gate', {}).get('ok') else c.get('provenance_gate', {}).get('problems'))}、"
             f"再利用の blob {('OK' if c.get('reused_blobs', {}).get('ok') else c.get('reused_blobs', {}).get('problems'))}）・"
             f"有効 seed {c['n_valid']}、(B) 対比 ref・CH が FAIL {c['B_contrast_ref_and_CH_FAIL']}、"
             f"(C′) 課題 1 の EMA {{{', '.join(f'{a}: {v['ok']}' for a, v in c['C_door_record'].items())}}}・C の扉 {c['C_c_door']} {c.get('C_door_from_runs', '')}")
    cm = c.get("commits")
    if cm:
        L.append(f"- commit（hash と commit 時刻。先後はこちらで読む）: 登録 {cm['registration']}、検査 {cm['checks']}、"
                 f"走 {'・'.join(cm['runs']) or '—'}、この集計 {cm['verdict_head']}、P181 {cm['p181']}")
    for a in ALL_ARMS:
        v = res["_seed_tables"][a]
        if "invalid_reason" in v and (~v.valid).any():
            L.append(f"- 有効でない seed（{a}）: " + "、".join(f"{s_} {r_}" for s_, r_ in v[~v.valid].invalid_reason.items()))
    if len(sh):
        L.append(f"- 報告のみ: 課題 2 以降を含む h_upd の満点（30,000）割れで層が全滅していない（seed, 課題, 層）: "
                 + "、".join(f"{a} {int((sh.arm == a).sum())}" for a in H_ARMS))
    if len(pds):
        L += ["", "## 装置の読み出し: pool 出力の位置ごとの平均（扉の前・後、rms 偏差で割る。seed 中央値。P181 の腕は f1 だけ、pre = post）", "",
              "| 腕 | 層の入力 | t | 位置ごと（前） | 位置ごと（後） | 後/前 | チャネル（後） | 全体（後） |", "|---|---|---|---|---|---|---|---|"]
        for _, r in pds.iterrows():
            if r.task in (0, 1, 3, 10, 30):
                L.append(f"| {r.arm} | {r.layer} | {r.task} | {r.pdc_pre:.4f} | {r.pdc_post:.4f} | {r.ratio:.3f} | {r.cdc_post:.4f} | {r.dc_post:.4f} |")
    if len(mech["layers"]):
        L += ["", "## 層ごと（seed 中央値。位置・幅は単位中央値、開いた割合・死亡率は層の平均）", "",
              "| 腕 | 層 | t | z̄ | sd | z̄/sd | 開いた割合 | 死亡率 | ‖m_in‖ |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in mech["layers"].iterrows():
            if r.t in (0, 1, 3, 10, 30) and r.arm in NEW_ARMS + tuple(CTRL):
                L.append(f"| {r.arm} | {r.layer} | {r.t} | {r.zbar_med:.3f} | {r.sd_med:.3f} | {r.rel_pos_med:.2f} | "
                         f"{r.open_mean:.3f} | {r.dead_frac:.3f} | {r.m_in_norm:.3f} |")
        L += ["", "## µ 経路と b 経路（起点 → t、単位中央値の seed 中央値。b の取り分 = |Δb|/(|Δb| + |Δ⟨W, m_in⟩|)）", "",
              "| 腕 | 層 | 起点 | t | Δz̄ | Δb | Δ⟨W, m_in⟩ | b の取り分 |", "|---|---|---|---|---|---|---|---|"]
        for _, r in mech["path"].iterrows():
            if r.t in (3, 10, 30) and r.arm in NEW_ARMS + tuple(CTRL):
                L.append(f"| {r.arm} | {r.layer} | {r['from']} | {r.t} | {r.dz_med:.3f} | {r.db_med:.3f} | {r.dwm_med:.3f} | {r.b_share_med:.2f} |")
    tw = mech.get("three_way")
    if tw is not None and len(tw):
        tw.to_csv(base / "tables" / "mu_path_split.csv", index=False)
        L += ["", "## µ 経路の内訳（t0 → t、自分 ⟨ΔW, m₀⟩・上流 ⟨W₀, Δm⟩・交差 ⟨ΔW, Δm⟩。単位中央値の seed 中央値、報告のみ）", "",
              "| 腕 | 層 | t | 自分 | 上流 | 交差 | Δ⟨W, m_in⟩ |", "|---|---|---|---|---|---|---|"]
        for _, r in tw.iterrows():
            if r.layer != "c1" and r.arm in NEW_ARMS + tuple(CTRL):
                L.append(f"| {r.arm} | {r.layer} | {r.t} | {r.own_med:.3f} | {r.up_med:.3f} | {r.cross_med:.3f} | {r.dwm_med:.3f} |")
    L += ["", "## 最初に落ちる層（開いた割合の単位中央値が 1% を割る最初の課題。報告のみ）", ""]
    for a in NEW_ARMS + REUSED:
        v = res["_seed_tables"][a]
        v = v[v.valid]
        cnt = {}
        for s_ in v.first_layers:
            cnt[s_ or "なし"] = cnt.get(s_ or "なし", 0) + 1
        L.append(f"- {a}: " + "、".join(f"{k} {n}" for k, n in sorted(cnt.items())) +
                 f"（t30 の死亡率の seed 中央値 " + ", ".join(f"{l} {float(v[f'dead_t30_{l}'].median()):.2f}" for l in LAYERS) + "）")
    L += ["", "## 予測の採点（Claude、spec §7.1）", "", "| # | 項目 | 実現 | p(実現) | 最頻 | 的中 | Brier |", "|---|---|---|---|---|---|---|"]
    for who, key in (("Claude", "labels"), ("Fable", "labels_fable"), ("Codex", "labels_codex")):
        if who != "Claude":
            if not sc[key]:
                continue
            L += ["", f"### {who}", "", "| # | 項目 | 実現 | p(実現) | 最頻 | 的中 | Brier |", "|---|---|---|---|---|---|---|"]
        for k, v in sc[key].items():
            if "unscored" in v:
                L.append(f"| {k} | {v['item']} | {v['realised']} | — | — | — | 未採点（{v['unscored']}） |")
            else:
                L.append(f"| {k} | {v['item']} | {v['realised']} | {v['p_realised']:.2f} | {v['mode']} | "
                         f"{'○' if v['mode_hit'] else '✗'} | {v['brier']:.3f} |")
    L += ["", "### 二値（Claude）", "", "| # | 主張 | p | 真偽 | Brier |", "|---|---|---|---|---|"]
    for k, v in sc["binary"].items():
        truth = "—" if v["true"] is None else ("○" if v["true"] else "✗")
        brier = "—" if v["brier"] is None else f"{v['brier']:.3f}"
        L.append(f"| {k} | {v['claim']} | {v['p']:.2f} | {truth} | {brier} |")
    (base / "summary.md").write_text("\n".join(L) + "\n")


def main() -> None:
    frames, ctrl = load_frames()
    problems = [x for a in REUSED + NEW_ARMS for x in validate_frame(frames[a], a, SEEDS)]
    cbad = control_problems(ctrl)          # a completion gate (audit 2)
    problems += cbad
    for a in CTRL:
        if any(x.startswith(f"{a}:") for x in cbad):
            ctrl[a] = None
    mech_bad = required_mechanism_inputs()
    if mech_bad and "--allow-missing-mechanism" not in sys.argv:
        raise SystemExit("mechanism inputs incomplete (audit 3), no report written: " + "; ".join(mech_bad[:12]))
    gate = provenance_gate()
    blobs = reused_blobs()
    chk = checks_ok_state()
    cdoor = c_door_from_runs()
    ok_a = chk and gate["ok"] and blobs["ok"] and not problems
    res = analyze_frames(frames, ok_a, ctrl, c_door_ok=c_door_ok_from(cdoor))
    chk_meta = json.loads((OUT / "checks.json").read_text()).get("_meta", {}) if (OUT / "checks.json").exists() else {}
    run_commits = sorted({g.get("git_hash") for a in NEW_ARMS + tuple(CTRL) if (OUT / a / "provenance.json").exists()
                          for g in json.loads((OUT / a / "provenance.json").read_text()).get("gits", [])})
    res["conditions"] |= {"validation_problems": problems, "provenance_gate": gate, "reused_blobs": blobs,
                          "checks_all_pass": chk, "C_door_from_runs": cdoor, "mechanism_inputs_missing": mech_bad,
                          "commits": {"registration": commit_time("b1b28854"), "checks": commit_time(chk_meta.get("git_hash")),
                                      "runs": [commit_time(h) for h in run_commits],
                                      "verdict_head": commit_time(git("rev-parse", "HEAD").stdout.decode().strip()),
                                      "p181": commit_time(P181_COMMIT)}}
    short = [x for a in H_ARMS for x in record_shortfalls(frames[a], a)]
    mech = mechanism_tables(ALL_ARMS + tuple(a for a in CTRL if ctrl[a] is not None))
    mech["three_way"] = three_way_table(NEW_ARMS + tuple(a for a in CTRL if ctrl[a] is not None))
    pdc = pdc_table(frames, ctrl)
    st_ref = res["_seed_tables"]["ref"]
    ref_logit = float(st_ref[st_ref.valid].logit_sd_t1.median())
    sc = score(res, mech, ref_logit)
    write_report(res, mech, pdc, short, sc)
    print(json.dumps({"FEATURE_DOOR": res["FEATURE_DOOR"], "M_Hf": res["M_Hf"]["label"],
                      "arms": {a: res["arms"][a]["label"] for a in ALL_ARMS},
                      "ctrl": {k: v["label"] for k, v in res["ctrl"].items()}}))


if __name__ == "__main__":
    main()

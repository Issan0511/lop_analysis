#!/usr/bin/env python3
"""Registered verdict for lc_elu_lr_0917 (specs/spec_lc_elu_lr_0917.md section 4).

    python3 analysis/lc_elu_lr_0917/verdict.py --stage main   # Q1-Q3 + reports: all 20 core runs have >= 50 tasks
    python3 analysis/lc_elu_lr_0917/verdict.py --stage ext    # adds Q4: the 10 lr 1e-4 runs have 150 tasks
    ... --src results/_checks_lc_elu_lr_0917/synthetic/runs --out <dir>    # checks only

Reads runs/<arm>_s<seed>/per_task.csv (+ provenance.json when present).  Refuses to judge while a
required run is short, unless the missing tasks follow a DIVERGED status (they then count as floor).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "lc_elu_lr_0917" / "runs"
OUT = REPO / "results" / "lc_elu_lr_0917"

CORE = ("E1_lr1e3", "E36_lr1e3", "E1_lr1e4", "E36_lr1e4")
LO = ("E1_lr1e4", "E36_lr1e4")
ANCHORS = {"R_lr1e4": (20.03, 2.46), "LK08_lr1e3": (91.53, 0.18)}   # Lillo & Cheney Table 2 (mean, SD; n = 5)
LC_E36 = (84.23, 0.70)
SEEDS = (0, 1, 2, 3, 4)
N_LC = 5
WIN = (31, 50)
PRE = (11, 30)
EXT_WIN = (131, 150)
HORIZON = {"E1_lr1e3": 50, "E36_lr1e3": 50, "E1_lr1e4": 150, "E36_lr1e4": 150, "R_lr1e4": 50, "LK08_lr1e3": 50}
S_C = 0.001967                      # section 4.1: act_chimera_0913's collapsed series, t31-50, O_t - F_t
Z = NormalDist().inv_cdf(1 - 0.05 / 20)
M_W = Z * S_C / math.sqrt(20)
M_1 = Z * S_C
N_MAJ = 4                           # of 5 seeds
Q2_K = 3.0


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def load(src: Path, arm: str, seed: int) -> tuple[pd.DataFrame | None, dict]:
    d = src / f"{arm}_s{seed}"
    p = d / "per_task.csv"
    if not p.exists():
        return None, {}
    df = pd.read_csv(p, float_precision="round_trip").sort_values("task").reset_index(drop=True)
    prov = {}
    for name in ("provenance.json", "progress.json"):
        if (d / name).exists():
            prov = json.loads((d / name).read_text())
            break
    return df, prov


def usable(df: pd.DataFrame | None, prov: dict, upto: int) -> bool:
    if df is None:
        return False
    return len(df) >= upto or prov.get("status") == "DIVERGED"


# --------------------------------------------------------------------------
# section 4.1 floor
# --------------------------------------------------------------------------

def window(df: pd.DataFrame, win: tuple[int, int], col: str) -> float:
    s = df[(df.task >= win[0]) & (df.task <= win[1])][col]
    return float(s.mean())


def seed_floor(df: pd.DataFrame, prov: dict, win: tuple[int, int]) -> dict:
    """FLOOR iff O_W <= F_W + M_W; a run that diverged before the window's end counts as FLOOR."""
    if len(df) < win[1]:
        if prov.get("status") == "DIVERGED":
            return {"floor": True, "O_W": float("nan"), "F_W": float("nan"), "diverged": True}
        raise ValueError(f"run has {len(df)} tasks < {win[1]}")
    o, f = window(df, win, "online"), window(df, win, "floor")
    return {"floor": bool(o <= f + M_W), "O_W": o, "F_W": f, "diverged": False}


def arm_label(floors: list[bool], yes: str = "COLLAPSED", no: str = "ABOVE") -> str:
    k = sum(bool(v) for v in floors)
    if k >= N_MAJ:
        return yes
    if len(floors) - k >= N_MAJ:
        return no
    return "SPLIT"


def q1_label(cells: dict[str, str]) -> str:
    c = cells
    if all(c[a] == "COLLAPSED" for a in CORE):
        return "ALL_COLLAPSE"
    if all(c[a] == "ABOVE" for a in CORE):
        return "NONE_COLLAPSE"
    if (c["E1_lr1e3"] == c["E36_lr1e3"] == "COLLAPSED") and (c["E1_lr1e4"] == c["E36_lr1e4"] == "ABOVE"):
        return "LR_DOMINATES"
    if (c["E1_lr1e3"] == c["E1_lr1e4"] == "COLLAPSED") and (c["E36_lr1e3"] == c["E36_lr1e4"] == "ABOVE"):
        return "ALPHA_DOMINATES"
    if c["E36_lr1e4"] == "ABOVE" and all(c[a] == "COLLAPSED" for a in CORE if a != "E36_lr1e4"):
        return "JOINT_ONLY"
    return "OTHER(" + ", ".join(f"{a}={c[a]}" for a in CORE) + ")"


# --------------------------------------------------------------------------
# section 4.3 Q2
# --------------------------------------------------------------------------

def lifetime(df: pd.DataFrame, upto: int = 50) -> float:
    return float(df[df.task <= upto]["online"].mean())


def q2(life_pts: list[float], ref: tuple[float, float] = LC_E36) -> dict:
    x = np.asarray(life_pts, dtype=float)
    s = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    se = math.sqrt(s * s / len(x) + ref[1] ** 2 / N_LC)
    delta = float(x.mean() - ref[0])
    if abs(delta) <= Q2_K * se:
        lab = "REPRODUCED"
    else:
        lab = "HIGHER" if delta > 0 else "LOWER"
    return {"label": lab, "mean_pt": float(x.mean()), "sd_pt": s, "delta_pt": delta, "se_pt": se,
            "band_pt": Q2_K * se, "ref": ref}


# --------------------------------------------------------------------------
# section 4.4 Q3
# --------------------------------------------------------------------------

def q3_seed(df: pd.DataFrame) -> dict:
    dmu = window(df, WIN, "mu_norm_l2") - window(df, PRE, "mu_norm_l2")
    dz = window(df, WIN, "zbar_l2") - window(df, PRE, "zbar_l2")
    return {"running": bool(dmu > 0 and dz < 0), "dmu": dmu, "dz": dz}


def q3_arm(running: list[bool]) -> str:
    k = sum(bool(v) for v in running)
    if k >= N_MAJ:
        return "RUNNING"
    if k <= len(running) - N_MAJ:
        return "STOPPED"
    return "MIXED"


# --------------------------------------------------------------------------
# section 4.6 reports
# --------------------------------------------------------------------------

def t_half(df: pd.DataFrame) -> float:
    o1 = float(df.online.iloc[0])
    hit = df[df.online <= (o1 + df.floor) / 2]
    return float(hit.task.iloc[0]) if len(hit) else math.inf


def t_floor(df: pd.DataFrame) -> float:
    hit = df[df.online <= df.floor + M_1]
    return float(hit.task.iloc[0]) if len(hit) else math.inf


def fmt_t(v: float, horizon: int) -> str:
    return f">{horizon}" if math.isinf(v) else f"{int(v)}"


RESCALE_COLS = ("mu_norm_l2", "zbar_l2", "gtr_l2", "neffT_l2", "wnorm_l2", "wnorm_l1")


def rescale_rows(runs: dict, ks=(1, 2, 3, 4, 5, 10, 15)) -> list[dict]:
    """lr 1e-4 at task 10k against lr 1e-3 at task k, same alpha, seed means (section 4.6)."""
    out = []
    for alpha, hi, lo in (("1", "E1_lr1e3", "E1_lr1e4"), ("3.6", "E36_lr1e3", "E36_lr1e4")):
        for k in ks:
            row = {"alpha": alpha, "k": k, "t_lr1e3": k, "t_lr1e4": 10 * k}
            for col in RESCALE_COLS:
                a = [float(d[d.task == k][col].iloc[0]) for (arm, s), (d, _) in runs.items()
                     if arm == hi and d is not None and (d.task == k).any()]
                b = [float(d[d.task == 10 * k][col].iloc[0]) for (arm, s), (d, _) in runs.items()
                     if arm == lo and d is not None and (d.task == 10 * k).any()]
                row[f"{col}_lr1e3"] = float(np.mean(a)) if a else float("nan")
                row[f"{col}_lr1e4"] = float(np.mean(b)) if b else float("nan")
            out.append(row)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def judge(src: Path, stage: str) -> dict:
    runs = {}
    need = {a: (50 if stage == "main" else HORIZON[a]) for a in CORE}
    for a in list(CORE) + list(ANCHORS):
        for s in SEEDS:
            runs[(a, s)] = load(src, a, s)
    for a in CORE:
        for s in SEEDS:
            df, prov = runs[(a, s)]
            if not usable(df, prov, need[a]):
                raise SystemExit(f"REFUSE: {a}_s{s} has {0 if df is None else len(df)} tasks < {need[a]}")
    res = {"stage": stage, "M_W": M_W, "M_1": M_1, "Z": Z, "S_C": S_C, "seeds": {}, "cells": {}}
    for a in CORE:
        fl = []
        for s in SEEDS:
            df, prov = runs[(a, s)]
            r = seed_floor(df, prov, WIN)
            r.update(life=lifetime(df) if len(df) >= 50 else float("nan"),
                     t_half=t_half(df.head(HORIZON[a] if stage == "ext" else 50)),
                     t_floor=t_floor(df.head(HORIZON[a] if stage == "ext" else 50)))
            res["seeds"][f"{a}_s{s}"] = r
            fl.append(r["floor"])
        res["cells"][a] = arm_label(fl)
    res["Q1"] = q1_label(res["cells"])
    res["Q2"] = q2([100 * res["seeds"][f"E36_lr1e4_s{s}"]["life"] for s in SEEDS])
    q3 = {}
    for a in LO:
        per = [q3_seed(runs[(a, s)][0]) for s in SEEDS]
        for s, p in zip(SEEDS, per):
            res["seeds"][f"{a}_s{s}"].update(q3_dmu=p["dmu"], q3_dz=p["dz"], q3_running=p["running"])
        q3[a] = q3_arm([p["running"] for p in per])
    res["Q3"] = f"E1={q3['E1_lr1e4']}, E36={q3['E36_lr1e4']}"
    res["Q3_cells"] = q3
    if stage == "ext":
        q4 = {}
        for a in LO:
            fl = []
            for s in SEEDS:
                df, prov = runs[(a, s)]
                r = seed_floor(df, prov, EXT_WIN)
                res["seeds"][f"{a}_s{s}"].update(ext_floor=r["floor"], ext_O_W=r["O_W"], ext_F_W=r["F_W"])
                fl.append(r["floor"])
            q4[a] = arm_label(fl, "COLLAPSED_BY_150", "ABOVE_AT_150")
        res["Q4"] = f"E1={q4['E1_lr1e4']}, E36={q4['E36_lr1e4']}"
        res["Q4_cells"] = q4
    anchors = {}
    for a, ref in ANCHORS.items():
        got = [runs[(a, s)] for s in SEEDS]
        if all(d is not None and len(d) >= 50 for d, _ in got):
            anchors[a] = q2([100 * lifetime(d) for d, _ in got], ref)
            anchors[a]["floor_W"] = [seed_floor(d, p, WIN)["floor"] for d, p in got]
    res["anchors"] = anchors
    res["rescale"] = rescale_rows(runs)
    return res, runs


def write(res: dict, runs: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"question": "Q1", "label": res["Q1"]}]
    rows += [{"question": f"Q1:{a}", "label": res["cells"][a]} for a in CORE]
    rows += [{"question": "Q2", "label": res["Q2"]["label"]}, {"question": "Q3", "label": res["Q3"]}]
    if "Q4" in res:
        rows.append({"question": "Q4", "label": res["Q4"]})
    pd.DataFrame(rows).to_csv(out / f"verdict_{res['stage']}.csv", index=False)
    seeds = pd.DataFrame([{"run": k, **v} for k, v in res["seeds"].items()])
    seeds.to_csv(out / f"seeds_{res['stage']}.csv", index=False)
    pd.DataFrame(res["rescale"]).to_csv(out / f"rescale_{res['stage']}.csv", index=False)
    (out / f"verdict_{res['stage']}.json").write_text(json.dumps(res, indent=1, default=float))

    L = [f"# lc_elu_lr_0917 — 判定（stage {res['stage']}）", "",
         f"床の幅 m_W = {M_W:.5f}（z = {Z:.3f}・s_c = {S_C}）、m_1 = {M_1:.4f}。", "",
         f"**Q1 = `{res['Q1']}`**", "", "| 腕 | 判定 | seed の O_W − F_W | 生涯 online（t1–50, %） | T_half | T_floor |",
         "|---|---|---|---|---|---|"]
    for a in CORE:
        ss = [res["seeds"][f"{a}_s{s}"] for s in SEEDS]
        hz = HORIZON[a] if res["stage"] == "ext" else 50
        L.append(f"| {a} | {res['cells'][a]} | " + " / ".join(f"{r['O_W'] - r['F_W']:+.4f}" for r in ss)
                 + " | " + " / ".join(f"{100 * r['life']:.2f}" for r in ss)
                 + " | " + " / ".join(fmt_t(r["t_half"], hz) for r in ss)
                 + " | " + " / ".join(fmt_t(r["t_floor"], hz) for r in ss) + " |")
    q = res["Q2"]
    L += ["", f"**Q2 = `{q['label']}`**: E36_lr1e4 の生涯平均 {q['mean_pt']:.2f} ± {q['sd_pt']:.2f}（5 seed）、"
          f"Lillo & Cheney 84.23 ± 0.70。Δ = {q['delta_pt']:+.2f} pt、帯 ±{q['band_pt']:.2f}（3 SE）。", "",
          f"**Q3 = `{res['Q3']}`**", "", "| run | Δ‖µ₂‖（31–50 − 11–30） | Δz̄₂ | RUNNING |", "|---|---|---|---|"]
    for a in LO:
        for s in SEEDS:
            r = res["seeds"][f"{a}_s{s}"]
            L.append(f"| {a}_s{s} | {r['q3_dmu']:+.3f} | {r['q3_dz']:+.3f} | {r['q3_running']} |")
    if "Q4" in res:
        L += ["", f"**Q4 = `{res['Q4']}`**", "", "| run | O_W − F_W（t131–150） | FLOOR |", "|---|---|---|"]
        for a in LO:
            for s in SEEDS:
                r = res["seeds"][f"{a}_s{s}"]
                L.append(f"| {a}_s{s} | {r['ext_O_W'] - r['ext_F_W']:+.4f} | {r['ext_floor']} |")
    if res["anchors"]:
        L += ["", "錨（報告のみ）:", "", "| 腕 | 生涯平均 | 表 2 | Δ | 帯（3 SE） | 窓 t31–50 の FLOOR |", "|---|---|---|---|---|---|"]
        for a, q in res["anchors"].items():
            L.append(f"| {a} | {q['mean_pt']:.2f} ± {q['sd_pt']:.2f} | {q['ref'][0]} ± {q['ref'][1]} | "
                     f"{q['delta_pt']:+.2f} | ±{q['band_pt']:.2f} | {sum(q['floor_W'])}/5 |")
    L += ["", "時間の尺度合わせ（報告のみ・lr 1e−4 の t = 10k と lr 1e−3 の t = k、seed 平均）:", "",
          "| α | k | ‖µ₂‖ 1e−3 / 1e−4 | z̄₂ 1e−3 / 1e−4 | gtr₂ 1e−3 / 1e−4 | neffT₂ 1e−3 / 1e−4 |", "|---|---|---|---|---|---|"]
    for r in res["rescale"]:
        L.append(f"| {r['alpha']} | {r['k']} | {r['mu_norm_l2_lr1e3']:.2f} / {r['mu_norm_l2_lr1e4']:.2f} | "
                 f"{r['zbar_l2_lr1e3']:+.2f} / {r['zbar_l2_lr1e4']:+.2f} | {r['gtr_l2_lr1e3']:.3g} / {r['gtr_l2_lr1e4']:.3g} | "
                 f"{r['neffT_l2_lr1e3']:.3f} / {r['neffT_l2_lr1e4']:.3f} |")
    (out / f"summary_{res['stage']}.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("main", "ext"), required=True)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    res, runs = judge(Path(a.src), a.stage)
    write(res, runs, Path(a.out))
    print(json.dumps({k: res[k] for k in ("Q1", "Q3") if k in res} | {"Q2": res["Q2"]["label"]}
                     | ({"Q4": res["Q4"]} if "Q4" in res else {})))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Registered verdicts of spec addenda 1 and 2 (§7.4, §8.3) of sna_cnn_cause_1009.

    python3 analysis/sna_cnn_cause_1009/verdict2.py

References from bundle A (same engine, same seeds), spec addendum 4: primary window t11-20 with
CV06FC3 as the flat arm, s_X = (W11(CV06FC3) - W11(X)) / (W11(CV06FC3) - W11(SNA)); secondary window
t21-30 with SNAc3 as the flat arm (CV06FC3 runs away from t25), s'_X analogous.  A label is read from
the primary window; when the two windows straddle a band the PARTIAL side is taken.
Writes results/sna_cnn_cause_1009/verdict2.json and summary2.md.  A verdict is issued only when
every run it needs has reached the tasks it reads.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "results" / "sna_cnn_cause_1009"


def sign_p(k: int, n: int) -> float:
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(name: str) -> pd.DataFrame | None:
    f = ROOT / name / "per_task.csv"
    return pd.read_csv(f) if f.exists() else None


def window(d: pd.DataFrame, arm: str, lo: int, hi: int, col="arm") -> pd.Series | None:
    g = d[(d[col] == arm) & (d.task >= lo) & (d.task <= hi)]
    if g.empty or g.groupby("seed")["task"].nunique().min() < hi - lo + 1 or g.seed.nunique() < 10:
        return None
    return g.groupby("seed")["online_acc"].mean()


def drop5(d: pd.DataFrame, arm: str) -> pd.Series | None:
    """spec §10.2: drop_20 = mean(t1-5) - mean(t16-20)"""
    e, l = window(d, arm, 1, 5), window(d, arm, 16, 20)
    return None if e is None or l is None else e - l


def main() -> None:
    A = load("A")
    V: dict = {}
    L = ["# sna_cnn_cause_1009 — 追補 1・2 の判定（spec §7.4・§8.3）", ""]
    wS, wC = window(A, "SNA", 11, 20), window(A, "CV06FC3", 11, 20)
    wS2, wC2 = window(A, "SNA", 21, 30), window(A, "SNAc3", 21, 30)
    if wS is None or wC is None or wS2 is None or wC2 is None:
        print("bundle A has not reached t30 for SNA / CV06FC3 / SNAc3")
        return
    gap = float((wC - wS).mean())
    gap2 = float((wC2 - wS2).mean())
    L += [f"基準（束 A）: 主窓 t11–20 W11(CV06FC3) = {wC.mean():.4f}, W11(SNA) = {wS.mean():.4f}, "
          f"gap = {gap:+.4f}（{int((wC - wS > 0).sum())}/10 seed）; "
          f"副窓 t21–30 W21(SNAc3) = {wC2.mean():.4f}, W21(SNA) = {wS2.mean():.4f}, gap′ = {gap2:+.4f}"
          f"（{int((wC2 - wS2 > 0).sum())}/10）", ""]

    def share(d: pd.DataFrame, arm: str) -> dict | None:
        w = window(d, arm, 11, 20)
        w2 = window(d, arm, 21, 30)
        if w is None:
            return None
        s = float((wC.mean() - w.mean()) / gap)
        dS = w - wS
        r = {"W11": float(w.mean()), "s": s, "vs_SNA": float(dS.mean()), "wins_over_SNA": int((dS > 0).sum()),
             "p_vs_SNA": sign_p(int((dS > 0).sum()), len(dS)),
             "vs_CV06FC3": float((w - wC).mean()), "wins_over_CV06FC3": int((w - wC > 0).sum())}
        if w2 is not None:
            r["W21"] = float(w2.mean()); r["s2"] = float((wC2.mean() - w2.mean()) / gap2)
            r["vs_SNA_21"] = float((w2 - wS2).mean()); r["wins_over_SNA_21"] = int((w2 - wS2 > 0).sum())
        return r

    def band(s: float, s2: float | None, lo_lab: str, hi_lab: str, mid_lab: str) -> str:
        """label from the primary s; a secondary s2 straddling the band pulls to PARTIAL"""
        lab = lo_lab if s <= 0.25 else hi_lab if s >= 0.75 else mid_lab
        if s2 is not None:
            lab2 = lo_lab if s2 <= 0.25 else hi_lab if s2 >= 0.75 else mid_lab
            if lab2 != lab:
                lab = mid_lab
        return lab

    L += ["| 腕 | 束 | W11 | s | 対 SNA | 対 CV06FC3 | W21 | s′ | 対 SNA(21) |", "|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for b, arms in (("C", ["SNA+fcamp0.7", "SNA+fcamp0.9"]),
                    ("D", ["S:a3-c0.6-c3-c3", "S:a3-c0.6-c0.6-c0.6+fcamp0.7"]),
                    ("B1", ["SNAfrz1", "SNA+fixconv", "CV06FC3+fixconv"]),
                    ("B2", ["F1ONLY", "F2ONLY"])):
        d = load(b)
        if d is None:
            continue
        for a in arms:
            r = share(d, a)
            if r is None:
                continue
            rows[a] = r
            L.append(f"| {a} | {b} | {r['W11']:.4f} | {r['s']:+.2f} | {r['vs_SNA']:+.4f}（{r['wins_over_SNA']}/10, p {r['p_vs_SNA']:.3f}）"
                     f" | {r['vs_CV06FC3']:+.4f}（{r['wins_over_CV06FC3']}/10） | "
                     + (f"{r['W21']:.4f} | {r['s2']:+.2f} | {r['vs_SNA_21']:+.4f}（{r['wins_over_SNA_21']}/10）" if "W21" in r else "— | — | —") + " |")
    L += [""]
    if "SNA+fcamp0.7" in rows:
        r = rows["SNA+fcamp0.7"]
        V["QF (post hoc: windows chosen after bundle C was read, spec §10)"] = {
            "label": band(r["s"], r.get("s2"), "FLAT_POINT_CAUSE", "FLAT_POINT_NOT", "FLAT_POINT_PARTIAL"),
            "s_fcamp0.7": r["s"], "s2_fcamp0.7": r.get("s2"),
            "s_fcamp0.9": rows.get("SNA+fcamp0.9", {}).get("s"), "s2_fcamp0.9": rows.get("SNA+fcamp0.9", {}).get("s2")}
    if "F1ONLY" in rows and "F2ONLY" in rows:
        s1, s2 = rows["F1ONLY"]["s"], rows["F2ONLY"]["s"]
        lab = ("F2_CARRIES" if s2 >= 0.75 and s1 <= 0.25 else "F1_CARRIES" if s1 >= 0.75 and s2 <= 0.25 else
               "EITHER_SUFFICES" if s1 >= 0.75 and s2 >= 0.75 else "NEEDS_BOTH" if s1 <= 0.25 and s2 <= 0.25 else "SPLIT")
        V["Q1b"] = {"label": lab, "s_F1ONLY": s1, "s_F2ONLY": s2,
                    "s2_F1ONLY": rows["F1ONLY"].get("s2"), "s2_F2ONLY": rows["F2ONLY"].get("s2")}
    if "SNAfrz1" in rows:
        r = rows["SNAfrz1"]
        V["Q2'"] = {"label": band(r["s"], r.get("s2"), "FROZEN_RESCUES", "FROZEN_NO", "FROZEN_PARTIAL"),
                    "s": r["s"], "s2": r.get("s2")}
    D_ = load("D")
    if D_ is not None:
        a1, a2 = "S:a3-c0.6-c3-c3", "S:a3-c0.6-c0.6-c0.6+fcamp0.7"
        ref30 = D_[(D_.task == 30)]
        snaA = A[A.task == 30].groupby("arm")[["zsd_f2", "logit_sd"]].mean()
        if a1 in rows and not ref30[ref30.arm == a1].empty:
            z = ref30[ref30.arm == a1][["zsd_f2", "logit_sd"]].mean()
            runaway = bool(z["zsd_f2"] >= 2 * snaA.loc["SNA", "zsd_f2"] or z["logit_sd"] >= 2 * snaA.loc["SNA", "logit_sd"])
            w = rows[a1]
            lab = "ANCHOR_NOT" if runaway else ("ANCHOR_HOLDS" if w["vs_CV06FC3"] >= -0.002 else "ANCHOR_COSTS")
            V["QD1"] = {"label": lab, "runaway": runaway, "zsd_f2_t30": float(z["zsd_f2"]), "logit_sd_t30": float(z["logit_sd"]),
                        "SNA_t30": {"zsd_f2": float(snaA.loc["SNA", "zsd_f2"]), "logit_sd": float(snaA.loc["SNA", "logit_sd"])}, **w}
        if a2 in rows and not ref30[ref30.arm == a2].empty:
            z = ref30[ref30.arm == a2][["zsd_f2", "logit_sd"]].mean()
            runaway = bool(z["zsd_f2"] >= 2 * snaA.loc["SNA", "zsd_f2"] or z["logit_sd"] >= 2 * snaA.loc["SNA", "logit_sd"])
            w = rows[a2]
            lab = band(w["s"], w.get("s2"), "BOTH_EXPLAIN", "MORE_TO_IT", "PARTIAL")
            if runaway and lab == "BOTH_EXPLAIN":
                lab = "PARTIAL"
            V["QD2"] = {"label": lab, "runaway": runaway, "zsd_f2_t30": float(z["zsd_f2"]), "logit_sd_t30": float(z["logit_sd"]), **w}
    B1 = load("B1")
    if B1 is not None:
        f06, f3 = drop5(B1, "SNA+fixconv"), drop5(B1, "CV06FC3+fixconv")
        rS, rC = drop5(A, "SNA"), drop5(A, "CV06FC3")
        if f06 is not None and f3 is not None:
            dfix, dref = float((f06 - f3).mean()), float((rS - rC).mean())
            r = dfix / dref
            V["Q4"] = {"label": "FC_ALONE" if r >= 0.75 else "NEEDS_MOVING_CONV" if r <= 0.25 else "FIX_PARTIAL",
                       "r": r, "delta_fix": dfix, "delta_ref": dref,
                       "drop5": {"SNA+fixconv": float(f06.mean()), "CV06FC3+fixconv": float(f3.mean()),
                                 "SNA": float(rS.mean()), "CV06FC3": float(rC.mean())}}
    F = load("F30")
    if F is not None and F.groupby("fork")["task"].max().min() >= 40:
        wSA, wCA = window(A, "SNA", 31, 40), window(A, "SNAc3", 31, 40)      # spec §10.2: SNAc3 is the flat reference
        if wSA is not None and wCA is not None:
            gF = float((wCA - wSA).mean())
            fw = lambda fk: window(F, fk, 31, 40, col="fork")
            base, toFC3 = fw("SNA>SNA"), fw("SNA>CV06FC3")
            rec = float((toFC3.mean() - base.mean()) / gF)
            # spec §10.2: if the fc->3 fork runs away (scale), rec is not a measure of damage; the
            # conv->3 fork is reported alongside and F is capped at MIXED
            z40 = F[(F.fork == "SNA>CV06FC3") & (F.task == 40)][["zsd_f2", "logit_sd"]].mean()
            zA = A[(A.arm == "SNA") & (A.task == 40)][["zsd_f2", "logit_sd"]].mean()
            runaway = bool(z40["zsd_f2"] >= 2 * zA["zsd_f2"] or z40["logit_sd"] >= 2 * zA["logit_sd"])
            lab = "REGIME" if rec >= 0.75 else "DAMAGE" if rec <= 0.25 else "MIXED"
            if runaway and lab == "DAMAGE":
                lab = "MIXED"
            rec_conv = float((fw("SNA>CV3FC06").mean() - base.mean()) / gF)
            V["F"] = {"label": lab, "rec": rec, "rec_conv(SNA>CV3FC06)": rec_conv, "fc3_fork_runaway": runaway,
                      "fc3_fork_t40": {"zsd_f2": float(z40["zsd_f2"]), "logit_sd": float(z40["logit_sd"])}, "g_F": gF,
                      "SNA>SNA": float(base.mean()), "SNA>CV06FC3": float(toFC3.mean()),
                      "SNA>CV3FC06": float(fw("SNA>CV3FC06").mean()),
                      "CV3FC06>CV3FC06": float(fw("CV3FC06>CV3FC06").mean()),
                      "CV3FC06>SNAc3": float(fw("CV3FC06>SNAc3").mean()),
                      "calib_SNA>SNA_minus_A": float(base.mean() - wSA.mean())}
            w3S, w3C = window(A, "SNA", 31, 33), window(A, "SNAc3", 31, 33)      # spec §10.2
            lg, b3 = window(F, "SNA>SNA@f3x0.5", 31, 33, col="fork"), window(F, "SNA>SNA", 31, 33, col="fork")
            ratio = float((lg.mean() - b3.mean()) / (w3C.mean() - w3S.mean()))
            V["F-logit"] = {"label": "LOGIT_SCALE_MATTERS" if ratio >= 0.5 else "LOGIT_SCALE_NOT", "ratio": ratio}
    L += ["## 判定", ""]
    for k, v in V.items():
        L.append(f"- **{k}** = `{v['label']}` — " + json.dumps({kk: vv for kk, vv in v.items() if kk != 'label'},
                                                                ensure_ascii=False, default=float))
    (ROOT / "verdict2.json").write_text(json.dumps(V, indent=2, ensure_ascii=False, default=float))
    (ROOT / "summary2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

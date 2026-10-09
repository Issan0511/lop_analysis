#!/usr/bin/env python3
"""Registered verdicts of spec addenda 1 and 2 (§7.4, §8.3) of sna_cnn_cause_1009.

    python3 analysis/sna_cnn_cause_1009/verdict2.py

References from bundle A (same engine, same seeds): SNA (degrades) and CV06FC3 (flat).
W21 = seed mean of online_acc over t21-30; s_X = (W21(CV06FC3) - W21(X)) / (W21(CV06FC3) - W21(SNA)).
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
    e, l = window(d, arm, 1, 5), window(d, arm, 26, 30)
    return None if e is None or l is None else e - l


def main() -> None:
    A = load("A")
    V: dict = {}
    L = ["# sna_cnn_cause_1009 — 追補 1・2 の判定（spec §7.4・§8.3）", ""]
    wS, wC = window(A, "SNA", 21, 30), window(A, "CV06FC3", 21, 30)
    if wS is None or wC is None:
        print("bundle A has not reached t30 for SNA / CV06FC3")
        return
    gap = float((wC - wS).mean())
    L += [f"基準（束 A、t21–30）: W21(CV06FC3) = {wC.mean():.4f}, W21(SNA) = {wS.mean():.4f}, "
          f"gap_B = {gap:+.4f}（CV06FC3 > SNA が {int((wC - wS > 0).sum())}/10 seed）", ""]

    def share(d: pd.DataFrame, arm: str) -> dict | None:
        w = window(d, arm, 21, 30)
        if w is None:
            return None
        s = float((wC.mean() - w.mean()) / gap)
        dS = w - wS
        return {"W21": float(w.mean()), "s": s, "vs_SNA": float(dS.mean()), "wins_over_SNA": int((dS > 0).sum()),
                "p_vs_SNA": sign_p(int((dS > 0).sum()), len(dS)),
                "vs_CV06FC3": float((w - wC).mean()), "wins_over_CV06FC3": int((w - wC > 0).sum())}

    L += ["| 腕 | 束 | W21 | s | 対 SNA | 対 CV06FC3 |", "|---|---|---|---|---|---|"]
    rows = {}
    for b, arms in (("C", ["SNA+fcamp0.7", "SNA+fcamp0.9"]),
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
            L.append(f"| {a} | {b} | {r['W21']:.4f} | {r['s']:+.2f} | {r['vs_SNA']:+.4f}（{r['wins_over_SNA']}/10, p {r['p_vs_SNA']:.3f}）"
                     f" | {r['vs_CV06FC3']:+.4f}（{r['wins_over_CV06FC3']}/10） |")
    L += [""]
    if "SNA+fcamp0.7" in rows:
        s = rows["SNA+fcamp0.7"]["s"]
        V["QF"] = {"label": "FLAT_POINT_CAUSE" if s <= 0.25 else "FLAT_POINT_NOT" if s >= 0.75 else "FLAT_POINT_PARTIAL",
                   "s_fcamp0.7": s, "s_fcamp0.9": rows.get("SNA+fcamp0.9", {}).get("s")}
    if "F1ONLY" in rows and "F2ONLY" in rows:
        s1, s2 = rows["F1ONLY"]["s"], rows["F2ONLY"]["s"]
        lab = ("F2_CARRIES" if s2 >= 0.75 and s1 <= 0.25 else "F1_CARRIES" if s1 >= 0.75 and s2 <= 0.25 else
               "EITHER_SUFFICES" if s1 >= 0.75 and s2 >= 0.75 else "NEEDS_BOTH" if s1 <= 0.25 and s2 <= 0.25 else "SPLIT")
        V["Q1b"] = {"label": lab, "s_F1ONLY": s1, "s_F2ONLY": s2}
    if "SNAfrz1" in rows:
        s = rows["SNAfrz1"]["s"]
        V["Q2'"] = {"label": "FROZEN_RESCUES" if s <= 0.25 else "FROZEN_NO" if s >= 0.75 else "FROZEN_PARTIAL", "s": s}
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
        wSA, wCA = window(A, "SNA", 31, 40), window(A, "CV06FC3", 31, 40)
        if wSA is not None and wCA is not None:
            gF = float((wCA - wSA).mean())
            fw = lambda fk: window(F, fk, 31, 40, col="fork")
            base, toFC3 = fw("SNA>SNA"), fw("SNA>CV06FC3")
            rec = float((toFC3.mean() - base.mean()) / gF)
            V["F"] = {"label": "REGIME" if rec >= 0.75 else "DAMAGE" if rec <= 0.25 else "MIXED", "rec": rec, "g_F": gF,
                      "SNA>SNA": float(base.mean()), "SNA>CV06FC3": float(toFC3.mean()),
                      "SNA>CV3FC06": float(fw("SNA>CV3FC06").mean()),
                      "CV3FC06>CV3FC06": float(fw("CV3FC06>CV3FC06").mean()),
                      "CV3FC06>SNAc3": float(fw("CV3FC06>SNAc3").mean()),
                      "calib_SNA>SNA_minus_A": float(base.mean() - wSA.mean())}
            w3S, w3C = window(A, "SNA", 31, 33), window(A, "CV06FC3", 31, 33)
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

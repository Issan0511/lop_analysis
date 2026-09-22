#!/usr/bin/env python3
"""Registered verdicts for switch_push_cifar_0922 (spec §4).

    python3 analysis/switch_push_cifar_0922/report.py   # -> verdict.json, per_seed.csv, summary.md

Reads each arm's per_task.csv (LE_ref / LL_ref are S-A's) and tail.csv.  Every threshold is the
spec's: 0.5 (the switch window carried 79% of a task's ||mu2|| growth, so blocking it must remove
more than half), 0.9 for the memorization gate, 0.5 / 0.2 for the parent's survival rule, and the
reference band is S-A LE's own seed range.
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src import layer_chimera_cifar_0921 as C          # noqa: E402

OUT = REPO / "results" / "switch_push_cifar_0922"
SA = REPO / "results" / "layer_chimera_cifar_0921"
ARMS = {"LE_ref": SA / "LE", "LL_ref": SA / "LL"}
for a in ("LE_sw75", "LE_md75", "LE_sw750", "LE_md750", "LE_sw7500", "LE_md7500", "LL_sw750", "LL_md750"):
    ARMS[a] = OUT / a
WIN, T5, HALF, MEMO_MIN = (31, 50), 5, 0.5, 0.9
PRED = {"P1": ("Q1 = SWITCH_MAKES_THE_PUSH", 0.85), "P2": ("Q2 = MONOTONE", 0.8),
        "P3": ("G2 holds (md ~ reference)", 0.75), "P4": ("Q3 = ANTI_ALIGN_REDUCED", 0.7),
        "P5": ("Q4: LE_sw7500 window >= 0.5", 0.3), "P6": ("Q4: LE_sw7500 T_A > 3", 0.85),
        "P7": ("Q5 = LL_SAME_MECHANISM", 0.75), "P8": ("G1 holds for every arm", 0.9)}


def sign(diff) -> dict:
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    n, pos = len(nz), int((nz > 0).sum())
    k = min(pos, n - pos)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else float("nan")
    return {"median": float(np.median(d)) if len(d) else float("nan"), "pos": pos, "n": n, "p": p,
            "call": "A_WINS" if (n - pos) <= 1 and p < 0.05 and n else
                    "B_WINS" if pos <= 1 and p < 0.05 and n else "TIE"}


def per_seed() -> pd.DataFrame:
    rows = []
    for arm, root in ARMS.items():
        d = pd.read_csv(root / "per_task.csv", float_precision="round_trip")
        d = d[d.cond == "raw"]
        for seed, g in d.groupby("seed"):
            ok = g[g.online_acc.notna()]
            w = ok[(ok.task >= WIN[0]) & (ok.task <= WIN[1])].online_acc
            below = ok[ok.online_acc < 0.5].task
            m1 = ok[ok.task == 1]
            rows.append({"arm": arm, "seed": int(seed), "window": float(w.mean()),
                         "t1_online": float(m1.online_acc.iloc[0]),
                         "t1_memo": float(m1.memo_acc.iloc[0]),
                         "T_A": int(below.min()) if len(below) else 0,
                         "tasks": int(ok.task.max()), "diverged": bool(g.online_acc.isna().any())})
    return pd.DataFrame(rows)


def col(tail, arm, t, c):
    return tail[(tail.arm == arm) & (tail.task == t)].set_index("seed")[c].reindex(range(10))


def pair(tail, a, b, c, t=T5):
    """A_WINS means a < b for every seed but at most one (the diff is b - a)."""
    return {"pair": f"{b} - {a}", **sign(col(tail, b, t, c) - col(tail, a, t, c)),
            f"median_{a}": float(col(tail, a, t, c).median()),
            f"median_{b}": float(col(tail, b, t, c).median())}


def main():
    tail = pd.read_csv(OUT / "tail.csv", float_precision="round_trip")
    ps = per_seed()
    ps.to_csv(OUT / "per_seed.csv", index=False)
    v = {}

    g1 = {a: float(ps[ps.arm == a].t1_memo.median()) for a in ARMS}
    v["G1"] = {"pass": all(x >= MEMO_MIN for x in g1.values()), "t1_memo_median": g1, "rule": f">= {MEMO_MIN}"}
    ref = col(tail, "LE_ref", T5, "mu2")
    band = (float(ref.min()), float(ref.max()))
    g2 = {a: float(col(tail, a, T5, "mu2").median()) for a in ("LE_md75", "LE_md750", "LE_md7500")}
    v["G2"] = {"pass": all(band[0] <= x <= band[1] for x in g2.values()),
               "reference_band_LE_ref_t5": band, "mu2_t5_median": g2}

    q1 = pair(tail, "LE_sw750", "LE_md750", "mu2")
    halved = q1["median_LE_sw750"] < HALF * q1["median_LE_md750"]
    v["Q1"] = {"label": "SWITCH_MAKES_THE_PUSH" if q1["call"] == "A_WINS" and halved else
                        "REDUCED" if q1["call"] == "A_WINS" else
                        "MID_MAKES_THE_PUSH" if q1["call"] == "B_WINS" else "NO_TIMING",
               "halved": bool(halved), **q1}
    sw = {k: float(col(tail, f"LE_sw{k}", T5, "mu2").median()) for k in (75, 750, 7500)}
    v["Q2"] = {"label": "MONOTONE" if sw[75] > sw[750] > sw[7500] else "NOT_MONOTONE",
               "mu2_t5_median_switch": sw,
               "mu2_t5_median_mid": {k: float(col(tail, f"LE_md{k}", T5, "mu2").median()) for k in (75, 750, 7500)},
               "mu2_t5_median_ref": float(ref.median())}
    q3 = pair(tail, "LE_md750", "LE_sw750", "cos2_med")        # A_WINS = md more negative than sw
    v["Q3"] = {"label": "ANTI_ALIGN_REDUCED" if q3["call"] == "A_WINS" else "ANTI_ALIGN_UNCHANGED", **q3}
    w = lambda a: float(ps[ps.arm == a].window.median())
    lab = lambda x: "RESCUED" if x >= 0.5 else "PARTIAL_RESCUE" if x >= 0.2 else "NOT_RESCUED"
    v["Q4"] = {a: {"label": lab(w(a)), "window_median": w(a),
                   "T_A_median": float(ps[ps.arm == a].T_A.median()),
                   "T_A_per_seed": ps[ps.arm == a].sort_values("seed").T_A.tolist()}
               for a in ("LE_sw75", "LE_sw750", "LE_sw7500", "LE_md750", "LE_ref")}
    q5 = pair(tail, "LL_sw750", "LL_md750", "mu2")
    v["Q5"] = {"label": "LL_SAME_MECHANISM" if q5["call"] == "A_WINS" else
                        "LL_NO_TIMING" if q5["call"] == "TIE" else "LL_REVERSED", **q5}

    hit = {"P1": v["Q1"]["label"] == "SWITCH_MAKES_THE_PUSH", "P2": v["Q2"]["label"] == "MONOTONE",
           "P3": v["G2"]["pass"], "P4": v["Q3"]["label"] == "ANTI_ALIGN_REDUCED",
           "P5": v["Q4"]["LE_sw7500"]["label"] == "RESCUED",
           "P6": v["Q4"]["LE_sw7500"]["T_A_median"] > 3, "P7": v["Q5"]["label"] == "LL_SAME_MECHANISM",
           "P8": v["G1"]["pass"]}
    v["predictions"] = {k: {"predicate": PRED[k][0], "claude": PRED[k][1], "hit": hit[k]} for k in PRED}
    v["claude_score"] = sum(1 for k in PRED if (PRED[k][1] >= 0.5) == hit[k])
    v["_git"] = C.git_state()
    (OUT / "verdict.json").write_text(json.dumps(v, indent=2, default=float))

    med = lambda a, t, c: float(col(tail, a, t, c).median())
    L = ["# switch_push_cifar_0922 — 凍結窓の位置（spec §4）", "",
         f"**G1**（t1 memo ≥ 0.9）: {'成立' if v['G1']['pass'] else '不成立'}。"
         f"**G2**（md ≈ 参照 {band[0]:.0f}–{band[1]:.0f}）: {'成立' if v['G2']['pass'] else '不成立'} {v['G2']['mu2_t5_median']}", "",
         f"**Q1 主**: **{v['Q1']['label']}** — ‖µ₂‖(t5) 切替 {q1['median_LE_sw750']:.0f} 対 課題内 {q1['median_LE_md750']:.0f}"
         f"（参照 {float(ref.median()):.0f}）、{q1['call']} {q1['pos']}/{q1['n']} p={q1['p']:.4f}、半減 {halved}", "",
         f"**Q2 用量**: **{v['Q2']['label']}** — 切替 K=75/750/7500 で {sw[75]:.0f} / {sw[750]:.0f} / {sw[7500]:.0f}", "",
         f"**Q3 反平行**: **{v['Q3']['label']}** — cos(w₂,µ̂₂)(t5) 切替 {q3['median_LE_sw750']:.3f} 対 課題内 {q3['median_LE_md750']:.3f}", "",
         "**Q4 救命**: " + "、".join(f"{a} {v['Q4'][a]['label']} 窓 {v['Q4'][a]['window_median']:.3f} T_A {v['Q4'][a]['T_A_median']:.0f}"
                                   for a in ("LE_sw75", "LE_sw750", "LE_sw7500", "LE_md750", "LE_ref")), "",
         f"**Q5 LL**: **{v['Q5']['label']}** — ‖µ₂‖(t5) 切替 {q5['median_LL_sw750']:.1f} 対 課題内 {q5['median_LL_md750']:.1f}", "",
         "## 時間の形（seed 中央値）", "",
         "| arm | t | ‖µ₂‖ | z̄₂ | a₁ rms | cos(w₂,µ̂₂) | open | 正 unit | ΔW₁ rms |",
         "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for t in (1, 2, 3, 5, 10, 50):
            L.append(f"| {arm} | {t} | {med(arm,t,'mu2'):.0f} | {med(arm,t,'zbar2_med'):.0f} | {med(arm,t,'a1_rms'):.1f} | "
                     f"{med(arm,t,'cos2_med'):+.3f} | {med(arm,t,'open_frac'):.3f} | {med(arm,t,'npos'):.0f} | {med(arm,t,'drms'):.3f} |")
    L += ["", "## 予測の採点（Claude の列）", "", "| # | 述語 | Claude | 結果 |", "|---|---|---|---|"]
    for k, q in v["predictions"].items():
        L.append(f"| {k} | {q['predicate']} | {q['claude']} | {'○' if q['hit'] else '×'} |")
    L.append(f"\nClaude {v['claude_score']}/8（p ≥ 0.5 の向きで採点）")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()

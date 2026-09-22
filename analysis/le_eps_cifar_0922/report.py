#!/usr/bin/env python3
"""Registered verdicts for le_eps_cifar_0922 (spec §4).

    python3 analysis/le_eps_cifar_0922/report.py     # -> verdict.json, per_seed.csv, summary.md

Reads per_task.csv of the six arms (LE_e8/LL_e8 = S-A) and tail.csv.  Thresholds are the spec's:
2.2 = sqrt(0.87 * 5.64) (geometric mean of the two S-A bands' nearest edges), 5.64 = S-A LE's
minimum ratio, 0.15 = sqrt(0.06 * 0.38), 0.5 = the parent battle's collapse rule, 0.2 / 0.05 /
0.15 as written in §4.
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

OUT = REPO / "results" / "le_eps_cifar_0922"
SA = REPO / "results" / "layer_chimera_cifar_0921"
ARMS = {"LE_e8": SA / "LE", "LE_e4": OUT / "LE_e4", "LE_e3": OUT / "LE_e3",
        "LL_e8": SA / "LL", "LL_e4": OUT / "LL_e4", "LL_e3": OUT / "LL_e3"}
WIN = (31, 50)
R_STOP, R_SA_MIN, DCOS_THR = 2.2, 5.64, 0.15
PRED = {"P1": ("Q1 = AMPLIFIER_IS_ADAM (e3)", 0.85), "P2": ("Q1_e4 in {AMPLIFIER_IS_ADAM, REDUCED}", 0.8),
        "P3": ("MONOTONE", 0.85), "P4": ("Q2 = BOTH_STOPPED (e3)", 0.75), "P5": ("Q3 = INCOHERENT (e3)", 0.7),
        "P6": ("Q4 = NOT_RESCUED (e3)", 0.7), "P7": ("Q4 in {PARTIAL_RESCUE, RESCUED} (e3)", 0.3),
        "P8": ("Q5(e3) in {NO_TAX, SMALL_TAX}", 0.7), "P9": ("Q5(e4) = NO_TAX", 0.85)}


def sign(diff):
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    n, pos = len(nz), int((nz > 0).sum())
    k = min(pos, n - pos)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else float("nan")
    return {"median": float(np.median(d)) if len(d) else float("nan"), "pos": pos, "n": n, "p": p}


def per_seed():
    rows = []
    for arm, root in ARMS.items():
        d = pd.read_csv(root / "per_task.csv", float_precision="round_trip")
        d = d[d.cond == "raw"]
        for seed, g in d.groupby("seed"):
            ok = g[g.online_acc.notna()]
            w = ok[(ok.task >= WIN[0]) & (ok.task <= WIN[1])].online_acc
            below = ok[ok.online_acc < 0.5].task
            rows.append({"arm": arm, "seed": int(seed), "window": float(w.mean()),
                         "t1_online": float(ok[ok.task == 1].online_acc.iloc[0]),
                         "T_A": int(below.min()) if len(below) else 0, "tasks": int(ok.task.max()),
                         "diverged": bool(g.online_acc.isna().any())})
    return pd.DataFrame(rows)


def col(tail, arm, t, c):
    return tail[(tail.arm == arm) & (tail.task == t)].set_index("seed")[c].reindex(range(10))


def q1(tail, arm):
    r = col(tail, arm, 5, "a1_rms") / col(tail, arm, 2, "a1_rms")
    med = float(r.median())
    lab = ("AMPLIFIER_IS_ADAM" if med < R_STOP and (r < R_SA_MIN).all()
           else "REDUCED" if med < R_SA_MIN else "NOT_ADAM")
    return {"label": lab, "ratio_median": med, "ratio_per_seed": [round(float(v), 2) for v in r],
            "ratio_max": float(r.max())}


def q2(tail):
    mu3, mu8 = col(tail, "LE_e3", 5, "mu2"), col(tail, "LE_e8", 5, "mu2")
    z3, z8 = col(tail, "LE_e3", 5, "zbar2_med"), col(tail, "LE_e8", 5, "zbar2_med")
    mu_ref, z_ref = float(col(tail, "LE_e8", 2, "mu2").median()), float(col(tail, "LE_e8", 2, "zbar2_med").median())
    s = sign(mu8 - mu3)
    mu_ok = s["pos"] == 10 and float(mu3.median()) < 2 * mu_ref
    z_ok = float(z3.median()) > 2 * z_ref
    return {"label": "BOTH_STOPPED" if mu_ok and z_ok else "PARTIAL" if (mu_ok or z_ok) else "SINK_CONTINUES",
            "mu2_stopped": bool(mu_ok), "sink_stopped": bool(z_ok), "mu2_t5_median_e3": float(mu3.median()),
            "mu2_t5_median_e8": float(mu8.median()), "mu2_t2_median_e8(ref)": mu_ref, "mu2_sign": s,
            "zbar2_t5_median_e3": float(z3.median()), "zbar2_t5_median_e8": float(z8.median()),
            "zbar2_t2_median_e8(ref)": z_ref}


def q3(tail):
    d = col(tail, "LE_e3", 3, "dcos")
    return {"label": "INCOHERENT" if float(d.median()) < DCOS_THR else "STILL_ALIGNED",
            "dcos_t3_median_e3": float(d.median()), "dcos_t3_median_e8": float(col(tail, "LE_e8", 3, "dcos").median()),
            "dcos_t3_median_e4": float(col(tail, "LE_e4", 3, "dcos").median())}


def q4(ps, tail):
    w = ps[ps.arm == "LE_e3"].set_index("seed").window.reindex(range(10))
    med = float(w.median())
    return {"label": "RESCUED" if med >= 0.5 else "PARTIAL_RESCUE" if med >= 0.2 else "NOT_RESCUED",
            "window_median_e3": med, "window_per_seed": [round(float(v), 3) for v in w],
            "window_median_e4": float(ps[ps.arm == "LE_e4"].window.median()),
            "window_median_e8": float(ps[ps.arm == "LE_e8"].window.median()),
            "open_frac_t50_median": {a: float(col(tail, a, 50, "open_frac").median()) for a in ("LE_e8", "LE_e4", "LE_e3")}}


def q5(ps, arm):
    x = ps[ps.arm == arm].set_index("seed").window.reindex(range(10))
    y = ps[ps.arm == "LL_e8"].set_index("seed").window.reindex(range(10))
    d = x - y
    lab = "NO_TAX" if (d.abs() < 0.05).all() else "SMALL_TAX" if float(d.median()) > -0.15 else "LARGE_TAX"
    return {"label": lab, **sign(d), "window_median": float(x.median()), "window_median_e8": float(y.median())}


def main():
    tail = pd.read_csv(OUT / "tail.csv", float_precision="round_trip")
    ps = per_seed()
    ps.to_csv(OUT / "per_seed.csv", index=False)
    v = {"Q1": q1(tail, "LE_e3"), "Q1_e4": q1(tail, "LE_e4"), "Q1_e8": q1(tail, "LE_e8"),
         "Q2": q2(tail), "Q3": q3(tail), "Q4": q4(ps, tail), "Q5_e3": q5(ps, "LL_e3"), "Q5_e4": q5(ps, "LL_e4")}
    v["MONOTONE"] = v["Q1_e8"]["ratio_median"] > v["Q1_e4"]["ratio_median"] > v["Q1"]["ratio_median"]
    hit = {"P1": v["Q1"]["label"] == "AMPLIFIER_IS_ADAM",
           "P2": v["Q1_e4"]["label"] in ("AMPLIFIER_IS_ADAM", "REDUCED"), "P3": bool(v["MONOTONE"]),
           "P4": v["Q2"]["label"] == "BOTH_STOPPED", "P5": v["Q3"]["label"] == "INCOHERENT",
           "P6": v["Q4"]["label"] == "NOT_RESCUED", "P7": v["Q4"]["label"] in ("PARTIAL_RESCUE", "RESCUED"),
           "P8": v["Q5_e3"]["label"] in ("NO_TAX", "SMALL_TAX"), "P9": v["Q5_e4"]["label"] == "NO_TAX"}
    v["predictions"] = {k: {"predicate": PRED[k][0], "claude": PRED[k][1], "hit": hit[k]} for k in PRED}
    v["claude_score"] = sum(1 for k in PRED if (PRED[k][1] >= 0.5) == hit[k])
    v["_git"] = C.git_state()
    (OUT / "verdict.json").write_text(json.dumps(v, indent=2, default=float))

    def med(arm, t, c):
        return float(col(tail, arm, t, c).median())
    L = ["# le_eps_cifar_0922 — Adam ε 介入の登録判定（spec §4）", "",
         f"Q1 (LE_e3, r = a₁rms(t5)/a₁rms(t2)): **{v['Q1']['label']}** 中央値 {v['Q1']['ratio_median']:.2f}, 最大 {v['Q1']['ratio_max']:.2f}"
         f"（e4: {v['Q1_e4']['label']} {v['Q1_e4']['ratio_median']:.2f}; e8: {v['Q1_e8']['ratio_median']:.2f}）; MONOTONE = {v['MONOTONE']}", "",
         f"Q2 (t5): **{v['Q2']['label']}** ‖µ₂‖ e3 {v['Q2']['mu2_t5_median_e3']:.0f} / e8 {v['Q2']['mu2_t5_median_e8']:.0f}（参照 2×{v['Q2']['mu2_t2_median_e8(ref)']:.0f}）, "
         f"z̄₂ e3 {v['Q2']['zbar2_t5_median_e3']:.0f} / e8 {v['Q2']['zbar2_t5_median_e8']:.0f}（参照 2×{v['Q2']['zbar2_t2_median_e8(ref)']:.0f}）", "",
         f"Q3 (ΔW₁ t2→t3 の |cos x̄|): **{v['Q3']['label']}** e3 {v['Q3']['dcos_t3_median_e3']:.3f} / e4 {v['Q3']['dcos_t3_median_e4']:.3f} / e8 {v['Q3']['dcos_t3_median_e8']:.3f}", "",
         f"Q4 (窓 t31–50): **{v['Q4']['label']}** e3 {v['Q4']['window_median_e3']:.3f} / e4 {v['Q4']['window_median_e4']:.3f} / e8 {v['Q4']['window_median_e8']:.3f}; open_frac t50 {v['Q4']['open_frac_t50_median']}", "",
         f"Q5 (LL の税): e3 **{v['Q5_e3']['label']}**（窓 {v['Q5_e3']['window_median']:.3f} 対 {v['Q5_e3']['window_median_e8']:.3f}, 差の中央値 {v['Q5_e3']['median']:+.3f}）; "
         f"e4 **{v['Q5_e4']['label']}**（{v['Q5_e4']['window_median']:.3f}, {v['Q5_e4']['median']:+.3f}）", "",
         "## 時間の形（seed 中央値）", "",
         "| arm | t | open_frac | n_img | 正 unit | a⁺ | a₁ rms | ‖µ₂‖ | z̄₂ | ΔW₁ \\|cos x̄\\| | ΔW₁ rms | cos(−∇, ΔW₁) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("LE_e8", "LE_e4", "LE_e3", "LL_e8", "LL_e4", "LL_e3"):
        for t in (1, 2, 3, 4, 5, 10, 50):
            L.append(f"| {arm} | {t} | {med(arm,t,'open_frac'):.3f} | {med(arm,t,'n_img'):.0f} | {med(arm,t,'npos'):.0f} | {med(arm,t,'apos'):.1f} | "
                     f"{med(arm,t,'a1_rms'):.1f} | {med(arm,t,'mu2'):.0f} | {med(arm,t,'zbar2_med'):.0f} | {med(arm,t,'dcos'):.2f} | {med(arm,t,'drms'):.3f} | {med(arm,t,'dcos_grad'):+.2f} |")
    L += ["", "## 予測の採点（Claude の列）", "", "| # | 述語 | Claude | 結果 |", "|---|---|---|---|"]
    for k, p in v["predictions"].items():
        L.append(f"| {k} | {p['predicate']} | {p['claude']} | {'○' if p['hit'] else '×'} |")
    L.append(f"\nClaude {v['claude_score']}/9（p ≥ 0.5 の向きで採点）")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summary and registered verdicts of sna_cnn_cause_1009 (spec §3-§4).

    python3 analysis/sna_cnn_cause_1009/verdict.py [--bundles A,B]

Reads results/sna_cnn_cause_1009/<bundle>/per_task.csv, writes summary.md and verdict.json
next to the bundles.  Partial runs are summarised over the tasks they have, and the verdicts
are only issued when every run of the arms involved reached task 50.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "results" / "sna_cnn_cause_1009"
HOST = {"SNA": 0.9458, "SNAc3": 0.9618}
SITES = ("c1", "c2", "f1", "f2")


def sign_p(k: int, n: int) -> float:
    """two-sided exact sign test"""
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(bundles) -> pd.DataFrame:
    ds = []
    for b in bundles:
        f = ROOT / b / "per_task.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["bundle"] = b
            ds.append(d)
    return pd.concat(ds, ignore_index=True)


def per_seed(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, seed), g in d.groupby(["arm", "seed"]):
        g = g.set_index("task")
        T = g.index.max()
        r = {"arm": arm, "seed": seed, "T": T,
             "early": g.loc[g.index <= 10, "online_acc"].mean(),
             "window": g.loc[(g.index >= 31) & (g.index <= 50), "online_acc"].mean() if T >= 31 else np.nan,
             "late": g.loc[(g.index >= 41) & (g.index <= 50), "online_acc"].mean() if T >= 41 else np.nan}
        r["drop"] = r["early"] - r["late"]
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default="A,B")
    args = ap.parse_args()
    d = load(args.bundles.split(","))
    ps = per_seed(d)
    arms = list(dict.fromkeys(d["arm"]))
    lines = ["# sna_cnn_cause_1009 — summary", "",
             "窓 = t31–50 の online_acc、早期 = t1–10、低下 = 早期 − t41–50（seed 平均 ± seed 間 SD）。", ""]
    lines += ["| 腕 | n | 完了課題 | 窓 | 早期 | 低下 |", "|---|---|---|---|---|---|"]
    for a in arms:
        g = ps[ps.arm == a]
        lines.append(f"| {a} | {len(g)} | {int(g['T'].min())}–{int(g['T'].max())} | "
                     f"{g['window'].mean():.4f} ± {g['window'].std():.4f} | {g['early'].mean():.4f} | "
                     f"{g['drop'].mean():+.4f} |")
    # trajectory table
    lines += ["", "online_acc（seed 平均）:", "", "| t | " + " | ".join(arms) + " |",
              "|---|" + "---|" * len(arms)]
    m = d.groupby(["arm", "task"])["online_acc"].mean()
    for t in (1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50):
        if t > d["task"].max():
            break
        lines.append(f"| {t} | " + " | ".join(f"{m.get((a, t), np.nan):.4f}" for a in arms) + " |")
    # mechanism columns at selected tasks
    cols = ["switch_ce", "ep_first99", "w_fro_f3", "conf"] + [f"two_alpha_sd_med_{s}" for s in SITES] \
        + [f"zsd_{s}" for s in SITES] + [f"mob_{s}" for s in SITES] + [f"seat_{s}" for s in SITES] \
        + [f"eff_rank_{s}" for s in SITES] + [f"alpha_med_{s}" for s in SITES]
    for t in (1, 10, 30, 50):
        if t > d["task"].max():
            break
        lines += ["", f"t = {t}（seed 平均）:", "", "| 量 | " + " | ".join(arms) + " |",
                  "|---|" + "---|" * len(arms)]
        g = d[d.task == t].groupby("arm")[cols].mean()
        for c in cols:
            lines.append(f"| {c} | " + " | ".join(f"{g.loc[a, c]:.3f}" if a in g.index else "—" for a in arms) + " |")

    verdict = {}
    full = {a: bool((ps[ps.arm == a]["T"] >= 50).all()) and len(ps[ps.arm == a]) == 10 for a in arms}
    W = {a: ps[ps.arm == a].set_index("seed")["window"] for a in arms}
    if full.get("SNA") and full.get("SNAc3"):
        gap = float((W["SNAc3"] - W["SNA"]).mean())
        gap_d = W["SNAc3"] - W["SNA"]
        e0 = (abs(W["SNA"].mean() - HOST["SNA"]) <= 0.004 and abs(W["SNAc3"].mean() - HOST["SNAc3"]) <= 0.0025
              and gap >= 0.008)
        verdict["E0"] = {"label": "ENGINE_OK" if e0 else "ENGINE_DIFFERS",
                         "SNA": float(W["SNA"].mean()), "SNAc3": float(W["SNAc3"].mean()), "gap": gap,
                         "gap_wins": int((gap_d > 0).sum()), "gap_p": sign_p(int((gap_d > 0).sum()), len(gap_d))}

        def q(a):
            diff = W[a] - W["SNA"]
            return {"q": float(diff.mean() / gap), "diff": float(diff.mean()),
                    "wins_over_SNA": int((diff > 0).sum()), "p_vs_SNA": sign_p(int((diff > 0).sum()), len(diff)),
                    "vs_SNAc3": float((W[a] - W["SNAc3"]).mean()),
                    "wins_over_SNAc3": int((W[a] - W["SNAc3"] > 0).sum())}
        if full.get("CV3FC06") and full.get("CV06FC3"):
            qc, qf = q("CV3FC06"), q("CV06FC3")
            a_, b_ = qc["q"], qf["q"]
            if a_ <= 0.25 and b_ >= 0.75:
                lab = "CONV_LOCUS"
            elif b_ <= 0.25 and a_ >= 0.75:
                lab = "FC_LOCUS"
            elif a_ <= 0.25 and b_ <= 0.25:
                lab = "SUFFICIENT_EACH"
            elif a_ >= 0.75 and b_ >= 0.75:
                lab = "JOINT"
            else:
                lab = "PARTIAL"
            verdict["Q1"] = {"label": lab, "q_conv(CV3FC06)": qc, "q_fc(CV06FC3)": qf}
        if full.get("SNAfrz1"):
            qq = q("SNAfrz1")
            verdict["Q2"] = {"label": "FROZEN_RESCUES" if qq["q"] >= 0.75 else
                             "FROZEN_NO" if qq["q"] <= 0.25 else "FROZEN_PARTIAL", **qq}
        if full.get("SNApp"):
            qq = q("SNApp")
            dd = d[(d.task >= 31) & (d.task <= 50)]
            pp_2aw = float(dd[dd.arm == "SNApp"]["two_alpha_W_med_c2"].mean())
            manip = abs(pp_2aw - 1.2) / 1.2 >= 0.10
            lab = ("PP_NOT_MANIPULATED" if not manip else "POOLING_CAUSE" if qq["q"] >= 0.75 else
                   "POOLING_NOT" if qq["q"] <= 0.25 else "POOLING_PARTIAL")
            verdict["Q3"] = {"label": lab, "pooled_2aW_c2_window": pp_2aw, **qq}
    lines += ["", "## 判定（spec §4）", ""]
    for k, v in verdict.items():
        lines.append(f"- **{k}** = `{v['label']}` — " + json.dumps({kk: vv for kk, vv in v.items() if kk != 'label'},
                                                                   ensure_ascii=False))
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    (ROOT / "verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

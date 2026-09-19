#!/usr/bin/env python3
"""relu_doors_0919 -- the registered calls of spec §5 and the two prediction scores of §6."""
from __future__ import annotations

import json
import math
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "relu_doors_0919"
REF = REPO / "results" / "rlcifar_mlp_battle_0918" / "R" / "per_task.csv"
LADDER = ("ref", "C", "CH", "CHB", "CHB0")
EXTRA = ("CS", "LN")          # 追補 1 (spec §10)
SEEDS = list(range(10))
WIN, EARLY, LATE = (31, 50), (1, 10), (41, 50)

PRED = {                                   # spec §6, Claude (before implementation)
    "P1":  ("M == RESCUED", 0.65),
    "P2":  ("C alone collapses", 0.85),
    "P5":  ("CH collapses (b2 takes the door over)", 0.60),
    "P6":  ("B_ROUTE == BIAS_TAKES_OVER", 0.60),
    "P7":  ("R1 == L1_SAVED", 0.80),
    "P8":  ("R2 == L2_HELD", 0.45),
    "P9":  ("SKEW_ROUTE == SKEW_TAKES_OVER", 0.20),
    "P10": ("D4 == TIE (removal is no better than the derived WD)", 0.55),
    "P11": ("N == NEED_CHB", 0.50),
    "P13": ("if CHB is rescued, its window beats leaky 0.1 raw (0.726)", 0.35),
    "P14": ("no run recovers after gate_zero_frac_l2 reaches 1.00", 0.90),
}
ADD = {                                    # spec §10.4, Claude (2026-09-19 09:50, before the 2 arms)
    "Q1": ("CS_M == RESCUED", 0.70),
    "Q2": ("LN_M == RESCUED", 0.75),
    "Q3": ("E3 == ALL_THREE", 0.60),
    "Q4": ("E1 == SCALE_ENOUGH", 0.65),
    "Q5": ("LN's window beats Kumar's RL-MNIST LayerNorm (0.54) by >= 0.3", 0.70),
}
ADD_ISSA = {                               # spec §10.4, Issa (2026-09-19 10:25, before the 2 arms)
    "J1": "CS_M == COLLAPSED",
    "J2": "LN_M == RESCUED and E2 == LN_WORSE",
    "J3": "E1 == CENTER_NEEDED",
}
ISSA = {                                   # spec §6, Issa (2026-09-19 01:11, before any run)
    "I1": "M == RESCUED",
    "I2": "N in (NEED_CH, NEED_CHB)",
    "I3": "B_ROUTE == BIAS_TAKES_OVER",
    "I4": "SKEW_ROUTE == NO_SKEW_ROUTE",
}


def sign(diff) -> tuple[int, int, float]:
    d = np.asarray(diff, float)
    nz = d[d != 0]
    n = len(nz)
    pos = int((nz > 0).sum())
    k = min(pos, n - pos)
    p = min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0) if n else float("nan")
    return pos, n, p


def verdict(a, b, lo: str, hi: str, tie: str = "TIE") -> dict:
    """`a` - `b`, seed-paired: `hi` if a wins, `lo` if b wins (spec §4's rule)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    pos, n, p = sign(d)
    lab = tie
    if n and p < 0.05:
        if n - pos <= 1:
            lab = hi
        elif pos <= 1:
            lab = lo
    return {"label": lab, "median": float(np.median(d)), "pos": pos, "n": n, "p": float(p)}


def load() -> dict:
    d = {}
    for arm in LADDER + EXTRA:
        f = REF if arm == "ref" else OUT / arm / "per_task.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f, float_precision="round_trip")
        if arm == "ref":
            t = t[t.cond == "raw"]
        d[arm] = t
    return d


def per_seed(t: pd.DataFrame, col: str, lo: int, hi: int) -> dict:
    g = t[(t.task >= lo) & (t.task <= hi)].groupby("seed")[col].mean()
    return {int(s): float(v) for s, v in g.items()}


def at_task(t: pd.DataFrame, col: str, task: int) -> dict:
    """`ref` is the parent run and predates the new columns, so those read as NaN."""
    if col not in t.columns:
        return {s: float("nan") for s in SEEDS}
    g = t[t.task == task].groupby("seed")[col].mean()
    return {int(s): float(v) for s, v in g.items()}


def main() -> None:
    d = load()
    ALL = [a for a in LADDER + EXTRA if a in d]
    win = {a: per_seed(d[a], "online_acc", *WIN) for a in ALL}
    early = {a: per_seed(d[a], "online_acc", *EARLY) for a in ALL}
    late = {a: per_seed(d[a], "online_acc", *LATE) for a in ALL}
    vec = lambda m: [m[s] for s in SEEDS]
    resc = {a: sum(1 for s in SEEDS if win[a][s] >= 0.5) for a in ALL}

    L = {}
    L["M"] = {"rescued_seeds": resc["CHB"], "median_window": float(np.median(vec(win["CHB"]))),
              "label": "RESCUED" if resc["CHB"] >= 9 else
                       ("COLLAPSED" if resc["CHB"] <= 1 else "SPLIT")}
    first = next((a for a in ("C", "CH", "CHB", "CHB0") if resc[a] >= 9), None)
    L["N"] = {"label": f"NEED_{first}" if first else "NONE_RESCUED",
              "rescued_seeds": {a: resc[a] for a in LADDER}}
    L["D1"] = verdict(vec(win["C"]), vec(win["ref"]), "C_HURTS", "C_HELPS")
    L["D2"] = verdict(vec(win["CH"]), vec(win["C"]), "H_HURTS", "H_HELPS")
    L["D3"] = verdict(vec(win["CHB"]), vec(win["CH"]), "B_HURTS", "B_HELPS")
    L["D4"] = verdict(vec(win["CHB0"]), vec(win["CHB"]), "REMOVAL_HURTS", "REMOVAL_HELPS")

    r1 = at_task(d["C"], "dead_frac_l1", 1)
    L["R1"] = {"label": "L1_SAVED" if float(np.median(vec(r1))) < 0.10 else "L1_DIES",
               "median_dead_frac_l1_t1": float(np.median(vec(r1))),
               "ref_t1": float(np.median(vec(at_task(d["ref"], "dead_frac_l1", 1))))}
    r2 = at_task(d["CH"], "sink_ratio_l2", 50)
    L["R2"] = {"label": "L2_HELD" if float(np.median(vec(r2))) > -1.6 else "L2_SUNK",
               "median_sink_ratio_l2_t50": float(np.median(vec(r2)))}

    bc = at_task(d["C"], "bias_over_sd_l2", 50)
    bh = at_task(d["CH"], "bias_over_sd_l2", 50)
    v = verdict(vec(bh), vec(bc), "SMALLER", "LARGER")
    L["B_ROUTE"] = {"label": "BIAS_TAKES_OVER" if v["label"] == "LARGER" else "BIAS_IRRELEVANT",
                    "C_median": float(np.median(vec(bc))), "CH_median": float(np.median(vec(bh))),
                    **{k: v[k] for k in ("median", "pos", "n", "p")}}

    sk = {a: at_task(d[a], "onesided_frac_l1", 50) for a in ("CHB", "CHB0")}
    mx = max(float(np.median(vec(sk[a]))) for a in ("CHB", "CHB0"))
    L["SKEW_ROUTE"] = {"label": "SKEW_TAKES_OVER" if mx > 0.10 else "NO_SKEW_ROUTE",
                       "median_onesided_frac_l1_t50": {a: float(np.median(vec(sk[a])))
                                                       for a in ("CHB", "CHB0")}}
    zb = {a: max(float(np.max(vec(at_task(d[a], "abs_zbar_l1", t)))) for t in (1, 25, 50))
          for a in ("CHB", "CHB0")}
    L["ZBAR1"] = {"label": "PINNED" if max(zb.values()) < 1e-4 else "NOT_PINNED",
                  "max_abs_zbar_l1": zb}

    gz = {}
    for a in ALL:
        t = d[a]
        bad = 0
        for s in SEEDS:
            r = t[t.seed == s].sort_values("task")
            hit = r.index[r.gate_zero_frac_l2 >= 1.0] if "gate_zero_frac_l2" in r.columns else []
            if len(hit):
                after = r.loc[hit[0]:, "online_acc"]
                bad += int((after > 0.5).any())
        gz[a] = bad
    L["P14_recoveries"] = gz

    if "CS" in d and "LN" in d:
        lab = lambda a: ("RESCUED" if resc[a] >= 9 else
                         ("COLLAPSED" if resc[a] <= 1 else "SPLIT"))
        L["CS_M"] = {"label": lab("CS"), "rescued_seeds": resc["CS"],
                     "median_window": float(np.median(vec(win["CS"])))}
        L["LN_M"] = {"label": lab("LN"), "rescued_seeds": resc["LN"],
                     "median_window": float(np.median(vec(win["LN"])))}
        e2 = verdict(vec(win["LN"]), vec(win["CH"]), "LN_WORSE", "LN_BETTER")
        L["E2"] = e2
        v1 = verdict(vec(win["CS"]), vec(win["CH"]), "CENTER_BETTER", "SCALE_BETTER")
        L["E1"] = {"label": ("CENTER_NEEDED" if L["CS_M"]["label"] == "COLLAPSED" else
                             ("SCALE_ENOUGH" if v1["label"] in ("TIE", "SCALE_BETTER")
                              else "CENTER_BETTER")),
                   **{k: v1[k] for k in ("median", "pos", "n", "p")}}
        got_set = tuple(sorted(a for a in ("CH", "CS", "LN") if resc[a] >= 9))
        L["E3"] = {"label": {("CH", "CS", "LN"): "ALL_THREE", ("CH",): "ONLY_CENTER",
                             ("CS",): "ONLY_SCALE", ("CH", "LN"): "CENTER_AND_LN",
                             (): "NONE"}.get(got_set, "OTHER_" + "_".join(got_set)),
                   "rescued": list(got_set)}
    got = {"P1": L["M"]["label"] == "RESCUED",
           "P2": resc["C"] <= 1,
           "P5": resc["CH"] <= 1,
           "P6": L["B_ROUTE"]["label"] == "BIAS_TAKES_OVER",
           "P7": L["R1"]["label"] == "L1_SAVED",
           "P8": L["R2"]["label"] == "L2_HELD",
           "P9": L["SKEW_ROUTE"]["label"] == "SKEW_TAKES_OVER",
           "P10": L["D4"]["label"] == "TIE",
           "P11": L["N"]["label"] == "NEED_CHB",
           "P13": L["M"]["label"] == "RESCUED" and float(np.median(vec(win["CHB"]))) > 0.726,
           "P14": sum(gz.values()) == 0}
    S, br = {}, []
    for k, (claim, p) in PRED.items():
        S[k] = {"claim": claim, "p": p, "hit": bool(got[k])}
        br.append((p - (1 if got[k] else 0)) ** 2)
    S["_summary"] = {"n": len(br), "hits": sum(v["hit"] for k, v in S.items() if k != "_summary"),
                     "brier": float(np.mean(br))}
    gi = {"I1": L["M"]["label"] == "RESCUED",
          "I2": L["N"]["label"] in ("NEED_CH", "NEED_CHB"),
          "I3": L["B_ROUTE"]["label"] == "BIAS_TAKES_OVER",
          "I4": L["SKEW_ROUTE"]["label"] == "NO_SKEW_ROUTE"}
    SI = {k: {"claim": ISSA[k], "hit": bool(gi[k])} for k in ISSA}
    SI["_summary"] = {"n": len(ISSA), "hits": sum(v["hit"] for v in SI.values() if "hit" in v)}
    SA = SJ = None
    if "CS_M" in L:
        ga = {"Q1": L["CS_M"]["label"] == "RESCUED",
              "Q2": L["LN_M"]["label"] == "RESCUED",
              "Q3": L["E3"]["label"] == "ALL_THREE",
              "Q4": L["E1"]["label"] == "SCALE_ENOUGH",
              "Q5": L["LN_M"]["median_window"] - 0.54 >= 0.3}
        SA, b2 = {}, []
        for k, (claim, p) in ADD.items():
            SA[k] = {"claim": claim, "p": p, "hit": bool(ga[k])}
            b2.append((p - (1 if ga[k] else 0)) ** 2)
        SA["_summary"] = {"n": len(b2), "hits": sum(v["hit"] for k, v in SA.items() if k != "_summary"),
                          "brier": float(np.mean(b2))}
        gj = {"J1": L["CS_M"]["label"] == "COLLAPSED",
              "J2": L["LN_M"]["label"] == "RESCUED" and L["E2"]["label"] == "LN_WORSE",
              "J3": L["E1"]["label"] == "CENTER_NEEDED"}
        SJ = {k: {"claim": ADD_ISSA[k], "hit": bool(gj[k])} for k in ADD_ISSA}
        SJ["_summary"] = {"n": len(ADD_ISSA), "hits": sum(v["hit"] for v in SJ.values() if "hit" in v)}

    f = lambda v, sp=".4f": "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, sp)
    lines = ["# relu_doors_0919 -- ReLU が沈む道を 1 本ずつ塞ぐ", "",
             "入力は全腕 raw、活性化は全腕 ReLU、seed 0-9、50 タスク。`ref` は "
             "`rlcifar_mlp_battle_0918` の `R` raw の再利用。", "",
             "## 窓（t31-50 の online、seed 中央値）", "",
             "| 腕 | 窓 | seed の範囲 | 窓 >= 0.5 の seed | 低下 | t50 の第2層ゲート厳密0 |",
             "|---|---|---|---|---|---|"]
    for a in ALL:
        w = vec(win[a])
        gz2 = float(np.median(vec(at_task(d[a], "gate_zero_frac_l2", 50))))
        drop = float(np.median([early[a][s] - late[a][s] for s in SEEDS]))
        lines.append(f"| `{a}` | {f(float(np.median(w)))} | {f(min(w))}-{f(max(w))} | "
                     f"**{resc[a]}/10** | {f(drop, '+.4f')} | {f(gz2, '.3f')} |")
    lines += ["", "## 登録判定", ""]
    for k in ("M", "N", "D1", "D2", "D3", "D4", "R1", "R2", "B_ROUTE", "SKEW_ROUTE", "ZBAR1",
              "CS_M", "LN_M", "E1", "E2", "E3"):
        if k not in L:
            continue
        v = L[k]
        det = ", ".join(f"{kk} {f(vv, '.4f') if isinstance(vv, float) else vv}"
                        for kk, vv in v.items() if kk != "label")
        lines.append(f"- **{k}**: `{v['label']}` — {det}")
    lines += ["", f"- P14 の反例（ゲート 1.00 到達後に窓 > 0.5 に戻った走）: {gz}", "",
              "## 予測の採点", "",
              f"**Claude {S['_summary']['hits']}/{S['_summary']['n']}, Brier "
              f"{S['_summary']['brier']:.3f}**", "", "| key | 主張 | p | 的中 |", "|---|---|---|---|"]
    for k, v in S.items():
        if k != "_summary":
            lines.append(f"| {k} | {v['claim']} | {v['p']:.2f} | {'yes' if v['hit'] else 'no'} |")
    lines += ["", f"**Issa {SI['_summary']['hits']}/{SI['_summary']['n']}**", "",
              "| key | 主張 | 的中 |", "|---|---|---|"]
    for k, v in SI.items():
        if k != "_summary":
            lines.append(f"| {k} | {v['claim']} | {'yes' if v['hit'] else 'no'} |")
    if SA:
        lines += ["", "## 追補 1 の採点（spec §10.4）", "",
                  f"**Claude {SA['_summary']['hits']}/{SA['_summary']['n']}, Brier "
                  f"{SA['_summary']['brier']:.3f}**", "", "| key | 主張 | p | 的中 |", "|---|---|---|---|"]
        for k, v in SA.items():
            if k != "_summary":
                lines.append(f"| {k} | {v['claim']} | {v['p']:.2f} | {'yes' if v['hit'] else 'no'} |")
        lines += ["", f"**Issa {SJ['_summary']['hits']}/{SJ['_summary']['n']}**", "",
                  "| key | 主張 | 的中 |", "|---|---|---|"]
        for k, v in SJ.items():
            if k != "_summary":
                lines.append(f"| {k} | {v['claim']} | {'yes' if v['hit'] else 'no'} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "verdict.json").write_text(json.dumps(
        {"labels": L, "score_claude": S, "score_issa": SI,
         "score_add_claude": SA, "score_add_issa": SJ,
         "window": {a: win[a] for a in ALL}}, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

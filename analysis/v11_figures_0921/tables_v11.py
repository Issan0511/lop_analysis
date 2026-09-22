#!/usr/bin/env python3
"""表 2〜4 の数値部分を committed CSV から作り直し、vault の転記と照合できる形で書き出す。

    python3 analysis/v11_figures_0921/tables_v11.py  ->  results/v11_figures_0921/tables_2-4_from_csv.md

表 1（三つの箱）と表 5（証拠表）は文の表なので vault の原稿（論文作成/V11原稿_表1-5_0922）が正本。
走は起こさない。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import data as D
import style as S

OUT = S.OUT / "tables_2-4_from_csv.md"


def window(d: pd.DataFrame, lo=31, hi=50) -> pd.Series:
    return d[(d.task >= lo) & (d.task <= hi)].groupby("seed").online_acc.mean()


def table2() -> list[str]:
    L = ["## 表 2  中心化の扉（relu_doors_0919・S2・S3）— 数値は per_task.csv から", "",
         "| 腕 | 窓 t31–50（seed 中央値） | t1 online | t1 末 第 1 層死亡率 | t50 第 2 層 z̄₂ | t50 sd₂ | t50 第 2 層死亡率 |",
         "|---|---|---|---|---|---|---|"]
    doors = D.doors()
    for a in D.DOOR_ORDER:
        d = doors[a]
        w = window(d).median()
        t1 = d[d.task == 1].online_acc.median()
        dead1 = d[d.task == 1].dead_frac_l1.median()
        e = d[d.task == 50]
        L.append(f"| {a} | {w:.4f} | {t1:.3f} | {dead1:.3f} | {e.zbar_l2.median():.3g} | {e.zsd_l2.median():.3g} | {e.dead_frac_l2.median():.3f} |")
    v = json.loads((D.RES / "relu_doors_0919" / "verdict.json").read_text())["labels"]
    vh = json.loads((D.RES / "relu_doors_h_ref_0920" / "verdict.json").read_text())["labels"]
    L += ["", "登録ラベル（verdict.json）:", ""]
    for k in ("M", "N", "D1", "D2", "D3", "D4", "R1", "R2", "B_ROUTE", "E1", "E2", "E3", "CS_M", "LN_M"):
        r = v[k]
        extra = ""
        if isinstance(r, dict):
            if "median" in r and "pos" in r:
                extra = f"（中央値 {r['median']:+.4f}・{r['pos']}/{r['n']}・p {r['p']:.4g}）"
            elif "rescued_seeds" in r and isinstance(r["rescued_seeds"], int):
                extra = f"（救済 {r['rescued_seeds']}/10・窓中央値 {r.get('median_window', float('nan')):.4f}）"
        L.append(f"- {k}: `{r['label']}`{extra}")
    L.append("- S2（relu_doors_h_ref_0920）: " + "・".join(f"{k} `{val}`" for k, val in vh.items()))
    ch = D.ch_chb_200()
    L += ["", "S3（ch_chb_200_0919・t181–200）:", ""]
    for a, d in ch.items():
        w = window(d, 181, 200)
        L.append(f"- {a}: 窓 t181–200 の seed 別 {', '.join(f'{x:.3f}' for x in w.sort_values())}（< 0.51 が {(w < 0.51).sum()}/10）")
    return L


def table3() -> list[str]:
    L = ["", "## 表 3  13 活性化のバトル（rlcifar_mlp_battle_0918）— 窓 t31–50 の seed 中央値", "",
         "| 腕 | raw | std | std − raw | 崩壊 raw/std（窓 < 0.5 の seed 数） |", "|---|---|---|---|---|"]
    arms = ["KKT1", "SNA", "KKA", "KKA23", "RSL", "LK03", "LR", "SL", "LK001", "SILU", "GELU", "R", "ELU"]
    for a in arms:
        d = D.per_task(f"rlcifar_mlp_battle_0918/{a}/per_task.csv")
        rows = {}
        for cond in ("raw", "std"):
            sub = d[d.cond == cond]
            rows[cond] = window(sub)
        L.append(f"| {a} | {rows['raw'].median():.4f} | {rows['std'].median():.4f} | {rows['std'].median() - rows['raw'].median():+.3f} | {(rows['raw'] < 0.5).sum()}/{(rows['std'] < 0.5).sum()} |")
    return L


def table4() -> list[str]:
    L = ["", "## 表 4  5+1 CIFAR の 16 腕（cifar5p1_mlp_0920・std・lr 1e−4）— 後期窓 hard 21–29 の seed 中央値", "",
         "| # | 腕 | 後期窓 | fresh gap | mob l2 | eff_rank l2 | train 末 | test 末 |", "|---|---|---|---|---|---|---|---|"]
    rows = []
    for a in D.ARMS_16:
        d = D.arm5p1(f"{a}_std_lr0.0001")
        w = D.window5p1(d, S.LATE_5P1)
        g = D.fresh5p1(f"{a}_std_lr0.0001").set_index("seed").fresh_gap
        e = d[d.task == 29]
        rows.append(dict(arm=a, window=w.median(), gap=g.median(), mob=e.mob_l2.median(), rank=e.eff_rank_l2.median(),
                         train=e.train_acc.median(), test=e.test_acc.median()))
    t = pd.DataFrame(rows).sort_values("window", ascending=False).reset_index(drop=True)
    for i, r in t.iterrows():
        L.append(f"| {i+1} | {r.arm} | {r.window:.4f} | {r.gap:+.3f} | {r.mob:.3f} | {r['rank']:.1f} | {r.train:.3f} | {r.test:.3f} |")
    p = pd.read_csv(D.RES / "cifar5p1_mlp_0920" / "paired_tests.csv")
    L += ["", "対応符号検定（paired_tests.csv・Holm）:", ""]
    for _, r in p.iterrows():
        L.append(f"- {r.a} − {r.b}（{r.col}）: 中央値 {r.median_diff:+.4f}・{r.a_wins}/{r.n}・p_holm {r.p_holm:.4g} → `{r.verdict}`")
    return L


if __name__ == "__main__":
    lines = ["# 表 2〜4 の数値（committed CSV から再計算・vault の転記との照合用）", ""]
    lines += table2() + table3() + table4()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)

#!/usr/bin/env python3
"""付録 C  初期配置の残り — γ 梯子・第 2 層・bias 0 の副診断（図表一覧 §4 C）。

本文の図 1(b) は第 1 層の raw・std・C だけ。ここに γ 梯子（入力平均を γ 倍）と第 2 層、
bias を 0 にした副診断を置く。
元データ: results/initgeom_cifar_0920/report/{display_rows.csv,verdict.json}
格: 条件ごとのラベルと家族ラベル（L1_MODEL_MISS・L2_CONDITIONAL_MODEL_MISS）は登録。
    bias 0 は副診断（記述・主判定を置き換えない）。seed 20–39・訓練なし。
"""
from __future__ import annotations

import json
import sys
from math import erf, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D
import style as S

ORDER = ["std", "C", "gamma025", "gamma050", "gamma075", "raw", "gamma150", "gamma200"]
NAME = {"std": "std", "C": "C", "gamma025": "γ .25", "gamma050": "γ .5", "gamma075": "γ .75",
        "raw": "raw（γ 1）", "gamma150": "γ 1.5", "gamma200": "γ 2"}
LABCOL = {"PREDICTED": "#2f6b8f", "OFF_LOW": "#c05a3c", "OFF_HIGH": "#c05a3c"}


def load():
    g = pd.read_csv(D.RES / "initgeom_cifar_0920" / "report" / "display_rows.csv")
    v = json.loads((D.RES / "initgeom_cifar_0920" / "report" / "verdict.json").read_text())
    conds = {(c["layer"], c["condition"]): c for c in v["conditions"]}
    return g, v, conds


def panel_layer(ax, g, v, conds, layer):
    m = g[(g.layer == layer) & (g.variant == "main")]
    rr = np.linspace(0.02, 3.2, 500)
    Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))
    if layer == 1:
        ax.plot(rr, 2 * Phi(-2.33 / rr), color="#999999", lw=1.2, ls="--", label="$2\\Phi(-2.33/r)$（ガウス近似）", zorder=1)
    n_pred = 0
    for cond in ORDER:
        sub = m[m.condition == cond]
        c = conds.get((layer, cond))
        if c is None or len(sub) == 0:
            continue
        col = LABCOL[c["label"]]
        r = sub.r.median()
        ax.vlines(r, c["band"]["low"], c["band"]["high"], color=col, lw=7, alpha=0.28, zorder=2)
        ax.scatter(sub.r, sub.f, s=18, color=col, alpha=0.65, zorder=3)
        ax.annotate(f"{NAME[cond]}\n{c['label']}", xy=(r, max(sub.f.max(), c["band"]["high"])),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.3, color=col)
        n_pred += c["label"] == "PREDICTED"
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("$r$ = 層入力の平均 / 偏差")
    ax.set_ylabel(f"第 {layer} 層の初期片側率 $f$")
    ax.set_ylim(-0.03, 0.75)
    fam = v["families"][str(layer)]
    ax.set_title(f"({'ab'[layer-1]}) 第 {layer} 層: {n_pred}/{len(ORDER)} 条件が帯内")
    if layer == 1:
        ax.legend(loc="upper left", fontsize=7)
    S.grade(ax, "registered", fam, loc="lower right")


def panel_bias0(ax, g, conds):
    """bias を 0 にした副診断（第 2 層）。main と bias0 を seed で対応させる。"""
    m = g[(g.layer == 2)].pivot_table(index=["condition", "seed"], columns="variant", values="f").reset_index()
    for i, cond in enumerate(ORDER):
        sub = m[m.condition == cond]
        if len(sub) == 0:
            continue
        ax.scatter(np.full(len(sub), i - 0.16), sub["main"], s=16, color="#2f6b8f", alpha=0.6, zorder=3,
                   label="bias あり（主）" if i == 0 else None)
        ax.scatter(np.full(len(sub), i + 0.16), sub["bias0"], s=16, color="#c05a3c", alpha=0.6, zorder=3,
                   label="bias = 0（副診断）" if i == 0 else None)
        ax.hlines(sub["main"].median(), i - 0.30, i - 0.02, color="#2f6b8f", lw=2.2)
        ax.hlines(sub["bias0"].median(), i + 0.02, i + 0.30, color="#c05a3c", lw=2.2)
        c = conds.get((2, cond))
        if c is not None:
            ax.hlines([c["band"]["low"], c["band"]["high"]], i - 0.36, i + 0.36, color="#888888", lw=0.8, ls=":")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([NAME[c] for c in ORDER], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("第 2 層の初期片側率 $f$")
    ax.set_ylim(-0.03, 0.75)
    ax.set_title("(c) 第 2 層の bias を 0 にすると（点線は登録の予測帯）")
    ax.legend(loc="upper left", fontsize=7)
    S.grade(ax, "posthoc", "副診断（主判定を置き換えない）", loc="lower right")


def build():
    g, v, conds = load()
    fig, axes = S.grid(1, 3, 13.0, 4.5)
    panel_layer(axes[0], g, v, conds, 1)
    panel_layer(axes[1], g, v, conds, 2)
    panel_bias0(axes[2], g, conds)
    fig.suptitle("付録 C  初期配置の残り — RL-CIFAR / MLP・seed 20–39・訓練なし", fontsize=12)
    return fig


NOTE = """付録 C. 初期配置の残り（A5 initgeom_cifar_0920・未使用 seed 20–39・訓練なし）。
点は seed 1 つずつ（20 seed）。太い縦の帯は登録した予測帯（independent seed Binomial の二つの中央順序統計量に
union bound・coverage >= 0.99375）。青は PREDICTED、赤は帯外（OFF_LOW / OFF_HIGH）。
(a) 第 1 層。raw・std・C と γ .25・.5・2 は帯内、γ .75 は帯より低く γ 1.5 は帯より高い。家族ラベル
    L1_MODEL_MISS（6/8 帯内・γ 梯子の標本中央値は単調）。
(b) 第 2 層。8/8 が帯より高い（L2_CONDITIONAL_MODEL_MISS）。予測は測った a1 に条件付けたもので、
    bias を省略した分の限界。
(c) 第 2 層の bias を 0 にした副診断。C の第 2 層が .12 → 0 に落ちる。主判定を置き換えない。
元データ: results/initgeom_cifar_0920/report/{display_rows.csv,verdict.json}。
"""

if __name__ == "__main__":
    print(S.save(build(), "figC_initgeom_rest", NOTE))

#!/usr/bin/env python3
"""図 1 幾何と初期配置（図表一覧 §2）。

(a) 共通位置と揺らぎ・mu 方向成分・bias のレバーの模式（測定値ではない）
(b) 第 1 層の初期片側率 f 対 r（raw・std・C の観測と予測帯）
元データ: results/initgeom_cifar_0920/report/{display_rows.csv,verdict.json}
格: (b) は登録（3 条件）。gamma 梯子と第 2 層は付録 C
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import erf, sqrt

import data as D
import style as S

COND = {"raw": ("raw（生の画素）", "#7a7a7a"), "std": ("std（標準化）", "#2f6b8f"),
        "C": ("C（中心化）", "#4c9f70")}


def panel_a(ax):
    rng = np.random.default_rng(3)
    mu = np.array([2.6, 1.1])
    X = rng.normal(size=(400, 2)) @ np.array([[0.62, 0.20], [0.20, 0.42]]) + mu
    ax.scatter(X[:, 0], X[:, 1], s=6, color="#b9c4cc", alpha=0.8, zorder=1, label="入力 $x$")
    w = np.array([0.80, 0.60])
    t = np.linspace(-2.2, 2.2, 2)
    # z = 0 の境界（bias を動かすと平行に動く）
    for b, ls, lab in ((-np.dot(w, mu), "-", "$z=0$（$b = -w\\cdot\\mu$）"),
                       (-np.dot(w, mu) + 1.6, "--", "bias を動かすと平行移動")):
        c = -b / np.dot(w, w)
        ax.plot(c * w[0] + t * (-w[1]), c * w[1] + t * w[0], color="#c05a3c", lw=1.5, ls=ls, label=lab)
    ax.annotate("", xy=mu, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#1f4e6b", lw=2))
    ax.text(*(mu * 0.55 + np.array([-0.30, 0.34])), "$\\mu$（共通位置）", color="#1f4e6b", fontsize=9)
    ax.annotate("", xy=mu + w * 1.25, xytext=mu, arrowprops=dict(arrowstyle="-|>", color="#8a5aa8", lw=2))
    ax.text(*(mu + w * 1.32), "$w$", color="#8a5aa8", fontsize=10)
    ax.scatter([0], [0], s=28, color="#333333", zorder=4)
    ax.text(0.08, -0.36, "原点", fontsize=8, color="#333333")
    ax.text(3.6, 0.15, r"$z = w\cdot\mu + b$" "\n" r"$\quad +\, w\cdot(x-\mu)$", fontsize=9, color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f4f1ea", ec="#ccc4b4", lw=0.7))
    ax.set_xlim(-1.2, 6.0); ax.set_ylim(-1.5, 3.6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title("(a) 共通位置・揺らぎ・bias のレバー")
    ax.legend(loc="lower right", fontsize=7.5)
    S.grade(ax, "column", "模式（測定値ではない）", loc="upper left")


def panel_b(ax):
    g = pd.read_csv(D.RES / "initgeom_cifar_0920" / "report" / "display_rows.csv")
    v = json.loads((D.RES / "initgeom_cifar_0920" / "report" / "verdict.json").read_text())
    m = g[(g.layer == 1) & (g.variant == "main")]
    rr = np.linspace(0.02, 3.2, 500)
    Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))
    ax.plot(rr, 2 * Phi(-2.33 / rr), color="#999999", lw=1.2, ls="--",
            label="$2\\Phi(-2.33/r)$（ガウス近似）", zorder=1)
    for cond, (lab, col) in COND.items():
        sub = m[m.condition == cond]
        band = next(c for c in v["conditions"] if c["condition"] == cond and c["layer"] == 1)
        r = sub.r.median()
        ax.vlines(r, band["band"]["low"], band["band"]["high"], color=col, lw=6, alpha=0.30, zorder=2)
        ax.scatter(sub.r, sub.f, s=26, color=col, alpha=0.75, zorder=3, label=f"{lab}（{band['label']}）")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("$r$ = 入力の平均 / 偏差")
    ax.set_ylabel("第 1 層の初期片側率 $f$")
    ax.set_ylim(-0.02, 0.45)
    ax.set_title("(b) 初期の片側ユニットの割合")
    ax.legend(loc="upper left")
    S.grade(ax, "registered", "3 条件とも PREDICTED", loc="lower right")


def build():
    fig, axes = S.grid(1, 2, 11.0, 4.3)
    panel_a(axes[0])
    panel_b(axes[1])
    fig.suptitle("図 1  幾何と初期配置 — RL-CIFAR / MLP・20 seed", fontsize=12)
    return fig


NOTE = """図 1. 幾何と初期配置。
(a) は模式図で、測定値ではない。前活性 z は共通位置 w・mu + b と揺らぎ w・(x − mu) に分かれ、
    bias は境界を平行に動かすレバーとして働く。
(b) 点は seed 1 つずつ（20 seed）。太い縦の帯は登録した予測帯（independent seed Binomial の
    二つの中央順序統計量に union bound をかけたもの・coverage >= 0.99375）。破線はガウス近似
    2Phi(-2.33/r) で、予測帯そのものではない。3 条件とも PREDICTED。
    gamma 梯子（L1_MODEL_MISS）と第 2 層（L2_CONDITIONAL_MODEL_MISS）は付録 C。
元データ: results/initgeom_cifar_0920/report/{display_rows.csv,verdict.json}。
"""

if __name__ == "__main__":
    print(S.save(build(), "fig01_geometry", NOTE))

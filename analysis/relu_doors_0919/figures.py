#!/usr/bin/env python3
"""relu_doors_0919 -- the run's figures: the ladder, and the mechanism behind it."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "relu_doors_0919"
REF = REPO / "results" / "rlcifar_mlp_battle_0918" / "R" / "per_task.csv"
ARMS = ("ref", "C", "CH", "CHB", "CHB0", "CS", "LN")
COL = {"ref": "#7f7f7f", "C": "#9467bd", "CH": "#1f77b4", "CHB": "#2ca02c",
       "CHB0": "#17becf", "CS": "#d62728", "LN": "#ff7f0e"}
LABEL = {"ref": "ref（素の ReLU）", "C": "C（入力の中心化）", "CH": "CH（＋中間層の中心化）",
         "CHB": "CHB（＋bias）", "CHB0": "CHB0（bias 除去）",
         "CS": "CS（分散だけ）", "LN": "LN（素の LayerNorm）"}
for f in ("Noto Sans CJK JP", "Noto Sans CJK TC", "IPAGothic", "DejaVu Sans"):
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False


def load() -> dict:
    d = {}
    for a in ARMS:
        f = REF if a == "ref" else OUT / a / "per_task.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f, float_precision="round_trip")
        d[a] = t[t.cond == "raw"] if a == "ref" else t
    return d


def med(t: pd.DataFrame, col: str):
    if col not in t.columns:
        return None
    g = t.groupby("task")[col].median()
    return g.index.values, g.values


def main() -> None:
    d = load()
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.6), dpi=110)
    panels = [("online_acc", "online 正解率", "タスクごとの成績", (0, 1.02), False),
              ("r_a1", "第 2 層の入力の 平均/偏差  r", "扉 H だけが第 2 層の直流成分を消す", None, True),
              ("sink_ratio_l2", "第 2 層の  z̄ / sd", "沈み（−1.6 を割るとゲートが閉じる）", None, False),
              ("gate_zero_frac_l2", "第 2 層で φ′ が厳密に 0 の割合", "ゲートが閉じた量", (-0.03, 1.03), False)]
    for k, (col, ylab, title, ylim, logy) in enumerate(panels):
        a = ax[k // 2, k % 2]
        for arm in ARMS:
            if arm not in d:
                continue
            r = med(d[arm], col)
            if r is None:
                continue
            lw = 2.2 if arm in ("CH", "CS", "LN") else 1.5
            a.plot(r[0], r[1], lw=lw, color=COL[arm], label=LABEL[arm],
                   ls="-" if arm in ("ref", "CH", "CS", "LN") else "--")
        if col == "sink_ratio_l2":
            a.axhline(-1.6, color="0.4", lw=1.1, ls=":", zorder=0)
            a.text(50, -1.6, " 崩壊の閾値", va="center", fontsize=8, color="0.35")
            a.set_ylim(-3.0, 0.6)
        if logy:
            a.set_yscale("log")
        if ylim:
            a.set_ylim(*ylim)
        a.set_xlabel("タスク")
        a.set_ylabel(ylab)
        a.set_title(title, fontsize=10)
        a.grid(alpha=.25)
    ax[0, 0].legend(fontsize=7.5, ncol=2, loc="center right")
    fig.suptitle("relu_doors_0919 — RL-CIFAR × MLP で ReLU が沈む道を塞ぐ（seed 中央値、10 seed）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / "figure.png")
    print(f"wrote {OUT}/figure.png")


if __name__ == "__main__":
    main()

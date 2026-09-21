#!/usr/bin/env python3
"""図 8 応答 -> 再学習 — S4 の 12 腕（図表一覧 §2）。

分岐 t1 末 / t10 末から、第 2 層の応答の場を入れ替えたときの次課題 online。
元データ: results/resp_cifar_ee_0920/{arm_table.csv,paired.csv,verdict.json}
格: 登録（RESPONSE_BOTH_WAYS）。集約は平均 + sd の区間なので平均（§2.5-1）
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

# (表示名, 群)  群: 自然 / 復元（t1 の場を t10 へ） / 沈降（t10 の場を t1 へ） / 一様
ARMS = [
    ("N1r",       "t1 末・自然", "自然"),
    ("N10r",      "t10 末・自然", "自然"),
    ("N1",        "t1 末・自然（head 保持）", "自然"),
    ("N10",       "t10 末・自然（head 保持）", "自然"),
    ("R1_10r",    "t10 に t1 の場（復元）", "復元"),
    ("R1_10",     "t10 に t1 の場（head 保持）", "復元"),
    ("S1_10r",    "t1 に t10 の場（沈降）", "沈降"),
    ("S1_10",     "t1 に t10 の場（head 保持）", "沈降"),
    ("S1_L1_10r", "t1 に t10 の第 1 層の場", "沈降"),
    ("S1u5r",     "t1 に一様 u=5", "一様"),
    ("S1u10r",    "t1 に一様 u=10", "一様"),
    ("S1u20r",    "t1 に一様 u=20", "一様"),
]
GCOL = {"自然": "#7a7a7a", "復元": "#2f6b8f", "沈降": "#c05a3c", "一様": "#8a5aa8"}


def build():
    a = pd.read_csv(D.RES / "resp_cifar_ee_0920" / "arm_table.csv").set_index("arm")
    p = pd.read_csv(D.RES / "resp_cifar_ee_0920" / "paired.csv")
    v = json.loads((D.RES / "resp_cifar_ee_0920" / "verdict.json").read_text())

    fig, axes = S.grid(1, 2, 12.0, 4.6, gridspec_kw=dict(width_ratios=[1.45, 1.0]))
    ax, axp = axes

    y = np.arange(len(ARMS))[::-1]
    ax.set_xlim(0, 1.02)
    for yi, (key, lab, grp) in zip(y, ARMS):
        r = a.loc[key]
        ax.hlines(yi, r.online_min, r.online_max, color=GCOL[grp], lw=4, alpha=0.32)
        ax.scatter(r.online_mean, yi, color=GCOL[grp], s=34, zorder=3)
    # 6 腕が 0.92-0.97 に、2 腕が 0.10 に詰まるので、軸は 0-1 のまま数値を添える
    S.value_labels(ax, y, [a.loc[k].online_mean for k, _, _ in ARMS], "{:.3f}", flip_at=0.55)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l, _ in ARMS])
    S.acc_axis(ax, label=False)
    ax.set_ylim(-0.7, len(ARMS) - 0.3)
    ax.set_xlabel("次課題のオンライン精度")
    ax.grid(axis="y", alpha=0)
    ax.set_title("(a) 12 腕の次課題の学習")
    handles = [plt_line(c, g) for g, c in GCOL.items()]
    ax.legend(handles=handles, loc="lower left")
    S.grade(ax, "registered", v["label"], loc="upper left")

    # (b) 登録の対応差
    metrics = [("P1", "復元 R1_10r − N10r"), ("P2", "沈降 N1r − S1_10r"),
               ("restore_reset", "復元（head 保持との差）"), ("sink_keep", "沈降（head 保持との差）"),
               ("sink_L1", "第 1 層の場での沈降"), ("layer_difference", "層の差")]
    for i, (m, lab) in enumerate(metrics):
        d = p[p.metric == m].difference.to_numpy()
        axp.scatter(np.full(d.shape, i), d, s=20, color="#2f4858", alpha=0.7, zorder=3)
        axp.hlines(np.mean(d), i - 0.18, i + 0.18, color="#c46a2f", lw=2.4, zorder=4)
    axp.axhline(0, color="#999999", lw=0.9)
    axp.set_xticks(range(len(metrics)))
    axp.set_xticklabels([l for _, l in metrics], rotation=28, ha="right", fontsize=7.5)
    axp.set_ylabel("対応差（次課題の online）")
    axp.set_title("(b) 登録した対応差（10 seed）")
    S.grade(axp, "registered", "両方向", loc="upper left")
    S.breathe(axp)

    fig.suptitle("図 8  応答 → 再学習能力 — RL-CIFAR / MLP・S4・10 seed", fontsize=12)
    return fig


def plt_line(color, label):
    import matplotlib.pyplot as plt
    return plt.Line2D([], [], marker="o", ls="", color=color, label=label)


NOTE = """図 8. 応答から再学習能力へ。
(a) 点は seed 平均、線は seed の全範囲（信頼区間ではない・resp_cifar_ee の登録が平均と sd の
    区間なので平均を描く）。r のついた腕は出力 head を初期化してある。
(b) 登録した対応差。P1 = 復元（t10 の網に t1 の応答の場を移すと次課題が学べるように戻る）、
    P2 = 沈降（t1 の網に t10 の場を移すと学べなくなる）。登録判定は RESPONSE_BOTH_WAYS。
元データ: results/resp_cifar_ee_0920/{arm_table.csv,paired.csv,verdict.json}。
"""

if __name__ == "__main__":
    print(S.save(build(), "fig08_response", NOTE))

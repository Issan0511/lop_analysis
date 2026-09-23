#!/usr/bin/env python3
"""付録 B  5+1 CIFAR の予算 ×10 — 遅さか到達点か（図表一覧 §4 B）。

原典の 780 更新/課題に対し、hard と easy の両方を 10 倍（7800/7800）、hard だけ 10 倍（7800/780）。
登録ラベル: CH − KKT1 は LEVEL（差は縮まず広がる）。CH − R は 780 で +.116（10/10）、7800 で −.086（0/10）
と符号が反転する（扉の利得は予算に依存する）。train 末は −.150 → −.016 まで縮む（CH は遅い）。
元データ: results/cifar5p1_mlp_0920/{R_std_lr0.0001,CH_std_doors,KKT1_std_lr0.0001,
          R_std_x10_7800x7800,CH_std_x10_7800x7800,KKT1_std_x10_7800x7800,
          CH_std_x10_7800x780,KKT1_std_x10_7800x780}/per_task.csv
格: 登録（LEVEL・符号反転）。集約は seed 中央値（cifar5p1 は中央値＋符号検定・§2.5-1）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D
import style as S

FOLDER = {
    ("R", "780"): "R_std_lr0.0001", ("CH", "780"): "CH_std_doors", ("KKT1", "780"): "KKT1_std_lr0.0001",
    ("R", "7800/7800"): "R_std_x10_7800x7800", ("CH", "7800/7800"): "CH_std_x10_7800x7800",
    ("KKT1", "7800/7800"): "KKT1_std_x10_7800x7800",
    ("CH", "7800/780"): "CH_std_x10_7800x780", ("KKT1", "7800/780"): "KKT1_std_x10_7800x780",
}
ARMS = ["R", "CH", "KKT1"]
BUDGETS = ["780", "7800/780", "7800/7800"]
LS = {"780": "-", "7800/780": ":", "7800/7800": "--"}
SHORT = {"780": "×1", "7800/780": "hard ×10", "7800/7800": "×10"}


def load():
    return {k: D.arm5p1(v) for k, v in FOLDER.items()}


def build():
    dd = load()
    fig, axes = S.grid(1, 3, 13.0, 4.5, gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    ax, axw, axt = axes

    # (a) hard 課題の online（seed 中央値＋全範囲）。780 は実線・7800/7800 は破線
    for arm in ARMS:
        for b in ("780", "7800/7800"):
            d = dd[(arm, b)]
            t, Y = D.matrix(d[d.hard == 1], "online_acc")
            S.band(ax, t, Y, S.COLOR[arm], f"{S.LABEL.get(arm, arm)}・{SHORT[b]}", agg="median", ls=LS[b], alpha=0.10)
    S.window_span(ax, S.LATE_5P1[0] - 0.5, S.LATE_5P1[-1] + 0.5)
    ax.annotate("後期窓 hard 21–29", xy=(25, 0.03), ha="center", fontsize=7, color="#555555")
    S.acc_axis(ax)
    ax.set_xlabel(S.AXIS["task"] + "（hard だけ）")
    ax.set_title("(a) 課題別 online: ×1（実線）対 ×10（破線）")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    S.grade(ax, "registered", "窓は登録・課題別は登録列", loc="lower right")

    # (b) 後期窓・(c) train 末（t29）: 腕 × 予算
    def stat(arm, b, col):
        d = dd[(arm, b)]
        if col == "window":
            return D.window5p1(d, S.LATE_5P1)
        return d[d.task == 29].set_index("seed").train_acc

    for axx, col, ylab, ttl in ((axw, "window", S.AXIS["window"], "(b) 後期窓"),
                                (axt, "train", "t29 の train 精度", "(c) 到達点（train 末）")):
        x = 0
        ticks, labels = [], []
        for arm in ARMS:
            for b in BUDGETS:
                if (arm, b) not in dd:
                    continue
                s = stat(arm, b, col)
                axx.vlines(x, s.min(), s.max(), color=S.COLOR[arm], lw=4, alpha=0.30)
                axx.scatter(x, s.median(), color=S.COLOR[arm], s=34, zorder=3,
                            marker={"780": "o", "7800/780": "^", "7800/7800": "s"}[b])
                axx.annotate(f"{s.median():.3f}", xy=(x, s.max()), xytext=(0, 4), textcoords="offset points",
                             ha="center", fontsize=6.5, color="#555555")
                ticks.append(x); labels.append(f"{arm}\n{SHORT[b]}")
                x += 1
            x += 0.6
        axx.set_xticks(ticks); axx.set_xticklabels(labels, fontsize=7)
        axx.set_ylabel(ylab)
        axx.set_title(ttl)
        S.acc_axis(axx, label=False)
        axx.set_ylim(0.3, 1.06)

    # 登録の対応差を計算してラベルに（数値はキャプションへ・§2.5-6 だが符号反転は図の主張なので短縮形で）
    w = {k: D.window5p1(v, S.LATE_5P1) for k, v in dd.items()}
    d_ch_kkt = {b: (w[("CH", b)] - w[("KKT1", b)]).median() for b in BUDGETS}
    d_ch_r = {b: (w[("CH", b)] - w[("R", b)]).median() for b in ("780", "7800/7800")}
    S.grade(axw, "registered", "CH − KKT1 = LEVEL（%.3f / %.3f / %.3f）" % tuple(d_ch_kkt[b] for b in BUDGETS), loc="upper left")
    S.grade(axt, "registered", "CH − R: %+.3f → %+.3f（符号反転）" % (d_ch_r["780"], d_ch_r["7800/7800"]), loc="upper left")

    fig.suptitle("付録 B  5+1 CIFAR の予算 ×10 — MLP・std・10 seed", fontsize=12)
    return fig


NOTE = """付録 B. 5+1 CIFAR の予算 ×10（原典の 780 更新/課題に対し 7800/7800 と hard だけ 7800/780）。
(a) hard 課題の online（seed 中央値、帯は seed の全範囲・信頼区間ではない）。780 は実線、7800/7800 は破線。
登録窓は hard の課題 21–29。
(b) 後期窓の seed 中央値と全範囲。○ ×1（780）・△ hard ×10（7800/780）・□ ×10（7800/7800）。
登録ラベル: CH − KKT1 は 3 予算とも負で、差は縮まず広がる（LEVEL）。
(c) t29 の train 精度（到達点）。CH − KKT1 の差は −.150 から −.016 / −.010 まで縮む。CH は「届かない」のでは
なく「遅い」（登録した 3 ラベルは到達点と online を書き分けられていなかった・設計の記録）。
CH − R は 780 で +.116（10/10）、7800/7800 で −.086（0/10）と符号が反転する。7800 条件の R は後期 online .874 だが fresh gap +.094（10/10）が残る。
R の死 .095・eff_rank 37。この予算では CH の後期成績が R より低い。KKT1 − R は両予算で正（+.226 / +.063・10/10）。
16 腕の順位は原典の 780 更新に固有で、「この活性化は可塑性喪失を防ぐ」を予算をまたいで言わない。
元データ: results/cifar5p1_mlp_0920/{R_std_lr0.0001,CH_std_doors,KKT1_std_lr0.0001,*_x10_*}/per_task.csv。
"""

if __name__ == "__main__":
    print(S.save(build(), "figB_5p1_budget", NOTE))

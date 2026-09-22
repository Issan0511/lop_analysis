#!/usr/bin/env python3
"""図 10 実ラベルの箱 — 5+1 CIFAR の 16 腕・L2 Init・課題別（図表一覧 §2）。

元データ: results/cifar5p1_mlp_0920/
格: 登録（窓は hard 課題だけの平均・対応差の符号検定・Holm）
集約: seed 中央値（§2.5-1）
hard/easy: 分けて描く（§2.5-7）。窓は hard の課題 21-29。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D
import style as S

ARMS = D.ARMS_16
# 0922 Issa: Snake・KK 族は KKT1 だけ載せる（E1 で KKT1 = SNA が登録されており代表腕にできる）。
# (a) は登録の 16 腕の順位そのものなので 16 腕のまま。
PANEL_C = ["R", "l2init", "KKT1", "ELU", "DF", "CR"]
C_FOLDER = {"R": "R_std_lr0.0001", "l2init": "R_std_lr0.0001_l2init1e-3",
            "KKT1": "KKT1_std_lr0.0001", "SNA": "SNA_std_lr0.0001",
            "ELU": "ELU_std_lr0.0001", "DF": "DF_std_lr0.0001", "CR": "CR_std_lr0.0001"}
C_LABEL = {"R": "R（ReLU）", "l2init": "R ＋ L2 Init（1e−3）", "KKT1": "KKT1", "SNA": "SNA（Snake）",
           "ELU": "ELU", "DF": "DF", "CR": "CR"}


def windows() -> pd.DataFrame:
    rows = []
    for a in ARMS:
        d = D.arm5p1(f"{a}_std_lr0.0001")
        w = D.window5p1(d, S.LATE_5P1)
        g = D.fresh5p1(f"{a}_std_lr0.0001").set_index("seed").fresh_gap
        for s in w.index:
            rows.append({"arm": a, "seed": s, "window": w[s], "gap": float(g.get(s, np.nan))})
    return pd.DataFrame(rows)


def l2init_pairs() -> dict:
    """登録 E2・E3 の対応差: Snake 族 − L2 Init を λ ごとに、seed 0-9 と追試 10-19 で。"""
    folders = {
        ("seed 0–9", "KKT1"): "KKT1_std_lr0.0001",
        ("seed 0–9", "1e-3"): "R_std_lr0.0001_l2init1e-3", ("seed 0–9", "1e-2"): "R_std_lr0.0001_l2init1e-2",
        ("seed 10–19", "KKT1"): "KKT1_s10-19",
        ("seed 10–19", "1e-3"): "R_s10-19l2init:1e3", ("seed 10–19", "1e-2"): "R_s10-19l2init:1e2",
    }
    return {k: D.window5p1(D.arm5p1(v), S.LATE_5P1) for k, v in folders.items()}


def panel_a(ax, ax2, w: pd.DataFrame, variant: str):
    order = w.groupby("arm").window.median().sort_values().index.tolist()
    y = np.arange(len(order))
    med = w.groupby("arm").window.median()
    lo = w.groupby("arm").window.min()
    hi = w.groupby("arm").window.max()
    gmed = w.groupby("arm").gap.median()
    glo = w.groupby("arm").gap.min()
    ghi = w.groupby("arm").gap.max()
    cols = [S.COLOR[a] for a in order]

    ax.set_xlim(0, 1.02)
    ax2.set_xlim(-0.02, 0.72)
    if variant == "points":
        ax.hlines(y, lo[order], hi[order], color=cols, lw=4, alpha=0.30)
        ax.scatter(med[order], y, color=cols, s=30, zorder=3)
        ax2.hlines(y, glo[order], ghi[order], color=cols, lw=4, alpha=0.30)
        ax2.scatter(gmed[order], y, color=cols, s=30, zorder=3)
    else:                                            # "bars"
        ax.barh(y, med[order], color=cols, height=0.62)
        ax.hlines(y, lo[order], hi[order], color="#222222", lw=1.0)
        ax2.barh(y, gmed[order], color=cols, height=0.62)
        ax2.hlines(y, glo[order], ghi[order], color="#222222", lw=1.0)
    ax.set_xlabel(S.AXIS["window"])
    ax2.set_xlabel(S.AXIS["gap"])
    ax2.axvline(0, color="#999999", lw=0.8)
    # 13 腕が 0.58-0.67 に、fresh gap は 13 腕が 0 の近くに詰まる。軸は動かさず数値を添える。
    # 棒の版は棒の上に字を置くと濃い色で読めないので、必ず棒の外（右）に出す。
    if variant == "bars":
        # 棒の上は濃い色で字が読めず、棒の端は seed 範囲の線と重なるので、範囲の右端の外に出す
        S.value_labels(ax, y, med[order].to_numpy(), "{:.3f}", flip_at=99, at=hi[order].to_numpy())
        S.value_labels(ax2, y, gmed[order].to_numpy(), "{:.3f}", flip_at=99, at=ghi[order].to_numpy())
    else:
        S.value_labels(ax, y, med[order].to_numpy(), "{:.3f}", flip_at=0.62)
        S.value_labels(ax2, y, gmed[order].to_numpy(), "{:.3f}", flip_at=0.42)

    for a in (ax, ax2):
        a.set_yticks(y)
        a.set_ylim(-0.8, len(order) - 0.2)
        a.grid(axis="y", alpha=0)
    ax.set_yticklabels(order)
    ax2.set_yticklabels([])
    ax.set_title("(a) 16 腕の後期窓")
    ax2.set_title("(a) fresh gap")
    S.grade(ax, "registered", "窓 = hard 21–29", loc="lower right")
    S.grade(ax2, "registered", "全 16 腕が喪失", loc="upper right")


def panel_b(ax, w5):
    """(b) 登録 E2・E3: Snake 族 − L2 Init の対応差（±0.005 の同等性帯つき）。"""
    groups = [("KKT1", "1e-3"), ("KKT1", "1e-2")]
    marks = {"seed 0–9": ("o", "#2f4858"), "seed 10–19": ("s", "#c46a2f")}
    ax.axhspan(-0.005, 0.005, color="#7fa06f", alpha=0.13, lw=0, zorder=0)
    for i, (arm, lam) in enumerate(groups):
        for j, (tag, (mk, col)) in enumerate(marks.items()):
            a, b = w5[(tag, arm)], w5[(tag, lam)]
            common = a.index.intersection(b.index)
            diff = (a[common] - b[common]).to_numpy()
            x = i + (j - 0.5) * 0.26
            ax.scatter(np.full(diff.shape, x), diff, s=16, color=col, marker=mk,
                       alpha=0.7, zorder=3, label=tag if i == 0 else None)
            ax.hlines(np.median(diff), x - 0.09, x + 0.09, color=col, lw=2.2, zorder=4)
    ax.axhline(0, color="#999999", lw=0.9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{a} −\nL2Init λ={l.replace('e-', 'e−')}" for a, l in groups], fontsize=8.5)
    ax.set_ylabel("後期窓の対応差")
    ax.set_title("(b) L2 Init との対決（帯は ±0.005 の同等性）")
    ax.legend(loc="upper left", ncol=2)
    S.grade(ax, "registered", "E2 同等 / E3 A_WINS 19/20", loc="lower right")


def panel_c(ax, hard="dotted"):
    # 7 腕ぶんの帯を重ねると濁って線が読めなくなる。seed の散らばりは (a) が出しているので
    # ここは中央値の線だけにする（§2.5-2 の帯は (a) が担う）。
    for key in PANEL_C:
        d = D.arm5p1(C_FOLDER[key])
        th, Yh = D.matrix(d[d.hard == 1], "online_acc")
        ax.plot(th, np.median(Yh, 0), color=S.COLOR[key], lw=1.8, label=C_LABEL[key], zorder=2)
        if hard == "dotted":
            te, Ye = D.matrix(d[d.hard == 0], "online_acc")
            ax.plot(te, np.median(Ye, 0), color=S.COLOR[key], lw=0.9, ls=":", zorder=1)
    S.acc_axis(ax)
    ax.set_xlim(1, 30)
    ax.set_xlabel(S.AXIS["task"])
    ax.set_title("(c) 課題をまたぐ学習")
    S.window_span(ax, S.LATE_5P1[0], S.LATE_5P1[-1])
    ax.legend(loc="lower left", ncol=2)
    S.grade(ax, "registered", loc="lower right")


def build(variant="points", hard="dotted"):
    w = windows()
    w5 = l2init_pairs()
    fig = plt.figure(figsize=(13.0, 7.2), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 0.75, 1.5], height_ratios=[1, 1])
    axa = fig.add_subplot(gs[:, 0])
    axg = fig.add_subplot(gs[:, 1])
    axb = fig.add_subplot(gs[0, 2])
    axc = fig.add_subplot(gs[1, 2])
    panel_a(axa, axg, w, variant)
    panel_b(axb, w5)
    panel_c(axc, hard)
    fig.suptitle("図 10  実ラベルの箱 — 5+1 CIFAR / MLP・30 課題・10 seed", fontsize=12)
    return fig


NOTE = """図 10. 実ラベルの箱。点（棒）は seed 中央値、線（帯）は seed の全範囲（信頼区間ではない）。
窓は登録どおり hard 課題 21・23・25・27・29 の online の平均。(c) の実線は hard、点線は easy。
(a) 16 腕すべてが fresh gap > 0、すなわち全腕が可塑性を失っている。順位は窓の中央値。
    13 腕が 0.58-0.67 に、fresh gap は 13 腕が 0 の近くに詰まって位置から順位が読めないので、
    軸は動かさず（規約 §2.5-3）数値を点のそばに添えた。
(c) は中央値の線だけ。帯を重ねると濁るので、seed の散らばりは (a) で見る。Snake・KK 族は
    KKT1 だけ。(a) は登録の 16 腕の順位そのものなので 16 腕を出す。
(b) KKT1 − L2 Init の対応差。Snake・KK 族は E1 で KKT1 = SNA が登録されているので KKT1 だけを
    代表に載せる。λ=1e−3（箱ごとに調整した値）とは ±0.005 の同等性の範囲、λ=1e−2（原典の値）
    には 19/20 seed で勝つ。seed 10–19 は未使用 seed での追試。
検定は対応差の符号検定（Holm 補正）: SNA − R は中央値 +0.2258・10/10・p_holm 0.0137。
元データ: results/cifar5p1_mlp_0920/<arm>/per_task.csv・fresh_control.csv・paired_tests.csv。
"""

VARIANT = "bars"        # 0922 Issa 決定（順位と水準が読みやすい）

if __name__ == "__main__":
    print(S.save(build(VARIANT), "fig09_5p1", NOTE))

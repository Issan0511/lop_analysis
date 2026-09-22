#!/usr/bin/env python3
"""付録 D  駆動の全体 — 5 seed × 4 課題の境界／後続の表、M1x・M2x・M3x（図表一覧 §4 D）。

本文の図 5 は境界／後続の下向き条件率の 1 パネルだけ。ここに M1x（課題間を常時押す条件は優勢でない）、
課題別の自己 S・上流 U の帳簿（§1.8c の表）を置く。conf/label の分解は載せない（共通分母の相殺）。
元データ: results/drive_cifar_c_0920/report/{transport_by_window.csv,verdict.json}
格: M1x・M3x は登録、M2x は登録（副）、課題別は記述。
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

C_B, C_L, C_F = "#c05a3c", "#2f6b8f", "#444444"
TASKS = ["2", "3", "4", "5"]


def load():
    d = pd.read_csv(D.RES / "drive_cifar_c_0920" / "report" / "transport_by_window.csv")
    d["task"] = d["task"].astype(str)
    v = json.loads((D.RES / "drive_cifar_c_0920" / "report" / "verdict.json").read_text())
    return d, v


def panel_m1x(ax, d, v):
    """(a) 全窓の下向き条件率と上向き条件率（課題 2–5 まとめと課題別）。"""
    cats = ["all_tasks2to5"] + TASKS
    for i, task in enumerate(cats):
        sub = d[(d.window == "full") & (d.task == task)]
        ax.scatter(np.full(len(sub), i - 0.15), sub.cert_down_frequency, s=22, color=C_B, alpha=0.75, zorder=3,
                   label="下向きの条件" if i == 0 else None)
        ax.scatter(np.full(len(sub), i + 0.15), sub.cert_up_frequency, s=22, color=C_L, alpha=0.75, zorder=3,
                   marker="s", label="上向きの条件" if i == 0 else None)
        ax.hlines(sub.cert_down_frequency.median(), i - 0.26, i - 0.04, color=C_B, lw=2.2)
        ax.hlines(sub.cert_up_frequency.median(), i + 0.04, i + 0.26, color=C_L, lw=2.2)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(["課題 2–5"] + [f"課題 {t}" for t in TASKS])
    ax.set_ylabel("条件が立った (unit, 更新) の割合")
    ax.set_ylim(0, 0.30)
    ax.set_title("(a) 課題の全期間: 下向き < 上向き")
    ax.legend(loc="upper right")
    m1 = v["M1x"]
    S.grade(ax, "registered", f"M1x = {m1['label']}（{m1['positive']}/{m1['nonzero']}）", loc="upper left")


def panel_ledger(ax, d, which):
    """(b)(c) 課題別の自己 S・上流 U（窓の総和・seed ごとの線）。"""
    win, term, col, ttl = which
    for s in sorted(d.seed.unique()):
        sub = d[(d.window == win) & (d.seed == s) & (d.task.isin(TASKS))].sort_values("task")
        ax.plot([int(t) for t in sub.task], sub[f"{term}_sum"], marker="o", ms=3.5, lw=1.1, color=col, alpha=0.75,
                label="seed ごと" if s == 0 else None)
    ax.axhline(0, color="#999999", lw=0.9)
    ax.set_xticks([int(t) for t in TASKS])
    ax.set_xlabel(S.AXIS["task"])
    ax.set_ylabel(f"{'自己 S' if term == 'S' else '上流 U'} の総和（窓内）")
    ax.set_title(ttl)
    S.grade(ax, "posthoc", "課題別は記述", loc="lower left" if term == "U" else "upper right")


def panel_m3x(ax, d, v):
    """(d) 境界と後続の下向き条件率の差（M3x）と、後続の自己 S の符号（課題別）。"""
    m3 = v["M3x"]
    diffs = np.asarray(m3["differences"])
    ax.scatter(np.zeros(len(diffs)), diffs, s=26, color=C_B, zorder=3)
    ax.hlines(np.median(diffs), -0.2, 0.2, color=C_B, lw=2.4)
    m1 = np.asarray(v["M1x"]["differences"])
    ax.scatter(np.ones(len(m1)), m1, s=26, color=C_L, zorder=3)
    ax.hlines(np.median(m1), 0.8, 1.2, color=C_L, lw=2.4)
    ax.axhline(0, color="#999999", lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["M3x\n境界 − 後続\n（下向き条件率）", "M1x\n下向き − 上向き\n（全期間）"], fontsize=7.5)
    ax.set_ylabel("seed ごとの差")
    ax.set_title("(d) 登録した 2 つの差")
    S.breathe(ax)
    S.grade(ax, "registered", f"M3x {m3['label']}・M2x {v['M2x']}", loc="lower right")


def build():
    d, v = load()
    fig, axes = S.grid(2, 2, 11.0, 8.2)
    panel_m1x(axes[0], d, v)
    panel_m3x(axes[1], d, v)
    panel_ledger(axes[2], d, ("boundary", "S", C_B, "(b) 境界（切替直後 75 更新）の自己 S"))
    panel_ledger(axes[3], d, ("later", "U", C_L, "(c) 後続（残り 29,925 更新）の上流 U"))
    fig.suptitle("付録 D  駆動の全体 — RL-CIFAR / MLP・C 条件・第 2 層・課題 2–5・5 seed", fontsize=12)
    return fig


NOTE = """付録 D. 駆動の全体（A2 drive_cifar_c_0920・relu_doors の C 腕・ReLU・第 2 層・課題 2–5・seed 0–4）。
点は seed 1 つずつ、横線は中央値。
(a) 課題の全期間で、自己更新が共通位置を下げる十分条件が立つ割合（6.6–9.1%）は上げる条件（17.2–22.3%）より
    小さい。5/5 で逆向き。登録判定 M1x = NOT_SUPPORTED（「誤差修正が課題間を常時下向きに押す」は書けない）。
(b) 境界（各課題の最初の 75 更新）の自己移動 S の総和。t2 が最大で、4/5 seed で t5 までに 3–4 割縮む。
    「W の伸びが切替で顕現して衝撃が大きくなる」はこの窓では出ていない（記述）。
(c) 後続の上流 U の総和。課題を追うごとに大きくなり（seed 4: −446 → −3,442）、t5 では S より U が大きい
    seed が 4/5。課題の中の連続した輸送で、切替の衝撃ではない（記述）。
(d) 登録した 2 つの差。M3x = 境界 − 後続の下向き条件率（+0.43〜+0.49・5/5・片側 p 1/32）→ BOUNDARY_ENRICHED。
    M1x = 全期間の下向き − 上向き（−0.09〜−0.14・0/5）→ NOT_SUPPORTED。M2x = CERTIFICATE_CONSISTENT
    （条件が立った更新で符号違反 0）。conf/label の分解は共通分母で 10 桁の相殺があるため載せない。
元データ: results/drive_cifar_c_0920/report/{transport_by_window.csv,verdict.json}。
"""

if __name__ == "__main__":
    print(S.save(build(), "figD_drive_full", NOTE))

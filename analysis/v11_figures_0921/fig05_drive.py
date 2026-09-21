#!/usr/bin/env python3
"""図 5 駆動 — 境界と後続の下向き条件率、自己 S と上流 U（図表一覧 §2）。

本文に入れるのは (a) の 1 パネルだけ。残りは付録 D。
元データ: results/drive_cifar_c_0920/report/{transport_by_window.csv,verdict.json}
格: M3x は登録・課題別は記述
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

WCOL = {"boundary": ("境界（切替直後 75 更新）", "#c05a3c"),
        "later": ("後続（それ以降）", "#2f6b8f")}
TASKS = ["2", "3", "4", "5"]


def build():
    d = pd.read_csv(D.RES / "drive_cifar_c_0920" / "report" / "transport_by_window.csv")
    d["task"] = d["task"].astype(str)
    v = json.loads((D.RES / "drive_cifar_c_0920" / "report" / "verdict.json").read_text())

    fig, axes = S.grid(1, 2, 11.0, 4.2)
    ax, ax2 = axes

    # (a) 課題別の下向き条件率
    for j, (win, (lab, col)) in enumerate(WCOL.items()):
        for i, task in enumerate(TASKS):
            sub = d[(d.window == win) & (d.task == task)]
            x = i + (j - 0.5) * 0.30
            ax.scatter(np.full(len(sub), x), sub.cert_down_frequency, s=22, color=col,
                       alpha=0.75, zorder=3, label=lab if i == 0 else None)
            ax.hlines(sub.cert_down_frequency.median(), x - 0.11, x + 0.11, color=col, lw=2.4, zorder=4)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([f"課題 {t}" for t in TASKS])
    ax.set_ylabel("下向きの条件が立つ割合")
    ax.set_ylim(0, 0.88)
    ax.set_title("(a) 誤差修正が自分を押し下げる割合")
    ax.legend(loc="upper right")
    m3 = v["M3x"]
    S.grade(ax, "registered", f"M3x = {m3['label']}（{m3['positive']}/{m3['nonzero']}）", loc="upper left")

    # (b) 自己 S と上流 U（1 更新あたり）
    lab_pos = {("boundary", "S"): 0, ("boundary", "U"): 1, ("later", "S"): 2, ("later", "U"): 3}
    for (win, term), i in lab_pos.items():
        sub = d[(d.window == win) & (d.task == "all_tasks2to5")]
        vals = sub[f"{term}_sum_per_model_update"]
        col = WCOL[win][1]
        ax2.scatter(np.full(len(vals), i), vals, s=24, color=col, alpha=0.75, zorder=3)
        ax2.hlines(vals.median(), i - 0.16, i + 0.16, color=col, lw=2.4, zorder=4)
    ax2.axhline(0, color="#999999", lw=0.9)
    ax2.set_yscale("symlog", linthresh=0.01)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(["境界\n自己 S", "境界\n上流 U", "後続\n自己 S", "後続\n上流 U"], fontsize=8)
    ax2.set_ylabel("1 更新あたりの寄与")
    ax2.set_title("(b) 自己 S と上流 U（付録 D）")
    S.grade(ax2, "posthoc", "課題 2–5 をまとめた記述", loc="lower left")

    fig.suptitle("図 5  駆動 — RL-CIFAR / MLP・C 条件・5 seed", fontsize=12)
    return fig


NOTE = """図 5. 駆動。点は seed 1 つずつ（5 seed）、横線は中央値。
(a) 誤差修正が自分のユニットを押し下げる十分条件が立った更新の割合。課題 2-5 をまとめると
    境界（切替直後 75 更新）で 50-57%、後続では 6.5-9%。課題別に見ると境界の値は課題 2 の 0.76 から
    課題 5 の 0.35 へ下がる（記述・登録ではない）。登録判定は M3x = BOUNDARY_ENRICHED（5/5・片側 p = 0.031）。
    M1x（後続でも下向きが優勢）は NOT_SUPPORTED。
(b) は付録 D に回す 1 枚。境界では自己 S が大きく負、後続では正味がほぼ 0 になる。事後の記述で、
    因果でも正味の沈下の分解でもない（verdict の limitations のとおり）。
元データ: results/drive_cifar_c_0920/report/transport_by_window.csv。
"""

if __name__ == "__main__":
    print(S.save(build(), "fig05_drive", NOTE))

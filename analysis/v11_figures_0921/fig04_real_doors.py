#!/usr/bin/env python3
"""図 4 実ラベルでの扉 — R・H・CH・KKT1・L2 Init の 4 小パネル（図表一覧 §2）。

H は沈下を止めるが階数を戻さない、が読みどころ。
元データ: results/cifar5p1_mlp_0920/<arm>/per_task.csv
格: 登録（H 対 R）・階数は登録列の読み
集約: seed 中央値（cifar5p1 の登録は median + 符号検定・§2.5-1）
hard/easy: 5+1 の箱なので分けて描く（§2.5-7）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D
import style as S

ARMS = [("R", "R_std_lr0.0001", "R（ReLU・素）"),
        ("H", "H_std_doors", "H（中間層の中心化）"),
        ("CH", "CH_std_doors", "CH（入力＋中間層）"),
        ("KKT1", "KKT1_std_lr0.0001", "KKT1（kunekune）"),
        ("l2init", "R_std_lr0.0001_l2init1e-3", "R ＋ L2 Init（λ=1e−3）")]
PANELS = [("online_acc", S.AXIS["online"], "(a) 課題をまたぐ学習"),
          ("zbar_l2", S.AXIS["zbar2"], "(b) 第 2 層の $\\bar z_2$"),
          ("dead_frac_l2", S.AXIS["dead2"], "(c) 第 2 層の死亡率"),
          ("eff_rank_l2", S.AXIS["effrank"], "(d) 第 2 層の実効階数")]


def build(hard="dotted"):
    fig, axes = S.grid(1, 4, 13.0, 3.7)
    for ax, (col, ylab, title) in zip(axes, PANELS):
        for key, folder, lab in ARMS:
            d = D.arm5p1(folder)
            th, Yh = D.matrix(d[d.hard == 1], col)
            S.band(ax, th, Yh, S.COLOR[key], lab if ax is axes[0] else None, agg="median")
            if hard == "dotted":
                te, Ye = D.matrix(d[d.hard == 0], col)
                import numpy as _np
                ax.plot(te, _np.median(Ye, 0), color=S.COLOR[key], lw=0.9, ls=":", zorder=1)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel(S.AXIS["task"])
        ax.set_xlim(1, 30)
        S.window_span(ax, S.LATE_5P1[0], S.LATE_5P1[-1])
    S.acc_axis(axes[0])
    axes[0].legend(loc="lower left")
    axes[2].set_ylim(bottom=0)
    S.breathe(axes[2])   # 4 腕が 0 に張り付くので軸線から離す
    axes[2].annotate("H・CH・KKT1・L2 Init は 0 のまま", xy=(0.5, 0.055),
                     xycoords="axes fraction", ha="center", fontsize=7.5, color="#555555")
    axes[3].set_ylim(bottom=0)
    S.breathe(axes[3])
    S.grade(axes[0], "registered", "H 対 R", loc="lower right")
    for ax in axes[1:3]:
        S.grade(ax, "column", loc="upper left")
    S.grade(axes[3], "column", "階数は戻らない")
    sub = "実線 = hard（5 クラス CIFAR）" + ("・点線 = easy（1 クラス）" if hard == "dotted" else "のみ")
    fig.suptitle(f"図 4  実ラベルでの扉 — 5+1 CIFAR / MLP・30 課題・10 seed（{sub}）", fontsize=12)
    return fig


NOTE = """図 4. 実ラベルでの扉。線は seed 中央値、帯は seed の全範囲（信頼区間ではない）。
5+1 の箱は hard（5 クラス CIFAR）と easy（1 クラス）が交互に走るので分けて描く。実線が hard、
点線が easy。薄い帯は登録窓 = hard の課題 21・23・25・27・29。
(a) の格は登録（H 対 R）、(b)(c)(d) は登録列の読み。H は第 2 層の沈下と死亡を止めるが、
実効階数は R と同じところまで落ちる。
元データ: results/cifar5p1_mlp_0920/{R_std_lr0.0001,H_std_doors,CH_std_doors,KKT1_std_lr0.0001,
R_std_lr0.0001_l2init1e-3}/per_task.csv。
"""

if __name__ == "__main__":
    for hard in ("dotted", "hard_only"):
        print(S.save(build(hard), f"fig04_real_doors_{hard}", NOTE))

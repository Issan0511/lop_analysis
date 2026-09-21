#!/usr/bin/env python3
"""図 3 bias 経路 — CH と CHB の t1-200（図表一覧 §2）。

元データ: results/ch_chb_200_0919/{CH,CHB}/per_task.csv
格: 腕別は登録（CH = SPLIT・CHB = HOLDS）、総合は INCONCLUSIVE を図中に明記
bias は生の平均絶対値で出す（|b_2|/sd_2 は分母の sd_2 が動く・引用禁止 C）
集約: seed ごとに転移の時刻が違うので個別線（§2.5-2 の例外）。
      variant="median" は中央値＋全範囲にしたときの見え方（比較用）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as D
import style as S

# 0922 Issa: |b_2|/sd_2 は分母が動くので生の平均絶対値を出す（引用禁止 C）。
COLS = [("online_acc", S.AXIS["online"], "(a) 課題をまたぐ学習"),
        ("bias_absmean_l2", "第 2 層の $|b_2|$（生・平均絶対値）", "(b) 第 2 層の bias"),
        ("gate_zero_frac_l2", S.AXIS["gate0"], "(c) gate が厳密に 0")]


def build(variant="seeds"):
    dd = D.ch_chb_200()
    v = json.loads((D.RES / "ch_chb_200_0919" / "verdict.json").read_text())
    fig, axes = S.grid(1, 3, 11.5, 3.8)
    for ax, (col, ylab, title) in zip(axes, COLS):
        for arm in ("CH", "CHB"):
            t, Y = D.matrix(dd[arm], col)
            if variant == "seeds":
                S.seeds_lines(ax, t, Y, S.COLOR[arm], S.LABEL[arm])
            else:
                S.band(ax, t, Y, S.COLOR[arm], S.LABEL[arm], agg="median")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel(S.AXIS["task"])
        ax.set_xlim(1, 200)
        S.window_span(ax, 181, 200)
    S.acc_axis(axes[0])
    axes[0].legend(loc="lower left")
    axes[1].set_yscale("log")
    axes[2].set_ylim(-0.02, 1.02)
    S.grade(axes[0], "registered", f"CH = {v['arms']['CH']} / CHB = {v['arms']['CHB']}", loc="upper left")
    S.grade(axes[1], "registered", f"総合 {v['main']}")
    S.grade(axes[2], "registered", f"B_ROUTE_DELAY {v['support']['B_ROUTE_DELAY']}/10")
    fig.suptitle("図 3  bias 経路 — CH と CHB を 200 課題まで・10 seed", fontsize=12)
    return fig


NOTE = """図 3. bias 経路。細い線は seed 1 本ずつ（seed ごとに転移の時刻が違い、中央値が形を壊すため・
規約 §2.5-2 の唯一の例外）。薄い帯は登録窓 t181-200。
格は腕別が登録（CH = SPLIT・CHB = HOLDS）、総合は INCONCLUSIVE。seed 別の読みは
B_ROUTE_DELAY 7/10・INCONCLUSIVE 3/10。
元データ: results/ch_chb_200_0919/{CH,CHB}/per_task.csv。
"""

if __name__ == "__main__":
    for variant in ("seeds", "median"):
        print(S.save(build(variant), f"fig03_bias_route_{variant}", NOTE))

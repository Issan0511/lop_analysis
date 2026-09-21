#!/usr/bin/env python3
"""図 2 中心化の扉 — 7 腕の課題別 online と第 2 層の層別量（図表一覧 §2）。

本文に出すのは主張の梯子 ref -> C -> CH -> CHB の 4 腕だけ（0921・Issa）。
H（中間層だけ）・CS（分散だけ）・LN（素の LayerNorm）は「その扉だけでは足りない」を示す対照で、
付録 E に回す（fig02e_doors_controls）。

元データ: results/relu_doors_h_ref_0920/{ref,H}/per_task.csv（S2 の R=10 走）と
          results/relu_doors_0919/{C,CH,CHB,CS,LN}/per_task.csv
格: 窓は登録（M/N/D1-D4/R1/R2/B_ROUTE）・層別は登録列の読み
集約: seed 中央値（verdict.json が median + 符号検定・§2.5-1）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import data as D
import style as S

WIN = (31, 50)
# 0922 Issa: 正規化量（z_bar/sd・|b|/sd）は分母がゼロに落ちるだけで発散する。ref は t50 で
# sd_2 = 0・z_bar_2 = -0.037 なので、沈下比 -3.7e10 も |b|/sd 1.5e11 も分母の写しでしかない
# （引用禁止 C「正規化量で力学を書かない。生の x で測る」）。本文は生の z_bar_2 と sd_2 を出す。
MAIN_ARMS = ("ref", "C", "CH", "CHB")            # 本文: 主張の梯子
CTRL_ARMS = ("ref", "H", "CS", "LN")             # 付録 E: 開ける扉を変えた対照
PANELS = [("zbar_l2", "zbar2"), ("zsd_l2", "zsd2"), ("dead_frac_l2", "dead2")]
AX = {"zbar2": "第 2 層の $\\bar z_2$（生）", "zsd2": "第 2 層の $\\mathrm{sd}_2$（生）",
      "dead2": S.AXIS["dead2"]}


def build(layout="1x4", arms=MAIN_ARMS, title=None, tag="本文"):
    dd = D.doors()
    verdict = json.loads((D.RES / "relu_doors_0919" / "verdict.json").read_text())
    if layout == "1x4":
        fig, axes = S.grid(1, 4, 13.0, 3.6)
    else:
        fig, axes = S.grid(2, 2, 9.4, 6.6)

    for arm in arms:
        t, Y = D.matrix(dd[arm], "online_acc")
        S.band(axes[0], t, Y, S.COLOR[arm], S.LABEL[arm], agg="median", ls=S.LS.get(arm, "-"))
    S.acc_axis(axes[0])
    axes[0].set_title("(a) 課題をまたぐ学習")
    axes[0].legend(ncol=1, loc="center left")
    lab = (f"M = {verdict['labels']['M']['label']} / N = {verdict['labels']['N']['label']}"
           if tag == "本文" else "対照: どの扉も単独では足りない")
    S.grade(axes[0], "registered", lab, loc="upper left")

    for ax, (col, key) in zip(axes[1:], PANELS):
        for arm in arms:
            t, Y = D.matrix(dd[arm], col)
            # ref は他の腕とほぼ同じ高さに重なるので太めに敷いて、上の腕の隙間から見えるようにする
            S.band(ax, t, Y, S.COLOR[arm], None, agg="median", ls=S.LS.get(arm, "-"),
                   lw=2.6 if arm == "ref" else 1.7, zorder=1 if arm == "ref" else 2)
        ax.set_ylabel(AX[key])
        S.grade(ax, "column", loc={"zbar2": "upper right", "zsd2": "upper left",
                                   "dead2": "center right"}[key])
    axes[1].set_title("(b) 第 2 層の沈下")
    axes[2].set_title("(c) 第 2 層の目盛")
    axes[3].set_title("(d) 第 2 層の死亡率")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].axhline(0, color="#999999", lw=0.7, zorder=0)
    axes[1].set_yticks([1e1, 1e0, 0, -1e0, -1e1, -1e2])
    axes[2].set_ylim(bottom=0)
    S.breathe(axes[2])
    axes[3].set_ylim(0, 1)
    S.breathe(axes[3])   # 0 と 1 に張り付く腕が軸線に隠れないように

    for ax in axes:
        ax.set_xlim(1, 50)
        ax.set_xlabel(S.AXIS["task"])
        S.window_span(ax, *WIN)
    fig.suptitle(title or "図 2  中心化の扉 — RL-CIFAR / MLP・50 課題・10 seed", fontsize=12)
    return fig


NOTE = f"""図 2. 中心化の扉。本文に出すのは主張の梯子 ref -> C -> CH -> CHB の 4 腕。
線は seed 中央値、{S.SEED_BAND_NOTE}。薄い帯は登録窓 t31-50。
(b)(c) は生の量。正規化した沈下比 z_bar_2/sd_2 と |b_2|/sd_2 は本文に出さない: ref は t50 で
sd_2 = 0・z_bar_2 = -0.037 なので、比が -3.7e10 や 1.5e11 になるのは分母がゼロに落ちたためで、
生の量は動いていない（引用禁止 C）。ref が死ぬのは沈んだからではなく目盛が潰れたため、が (b)(c)
の読み。登録列 R2（沈下比の t50 中央値 -0.191、CH）はキャプションの数値として引く。
(a) の格は登録（M = RESCUED・N = NEED_CH）、(b) の 3 枚は登録列の読み。
符号検定は D2 = H_HELPS（中央値 +0.874・10/10・p = 0.00195）、D1/D3/D4 = TIE。
B_ROUTE = BIAS_TAKES_OVER（中央値 +0.179・10/10・p = 0.00195）。
H（中間層だけ）・CS（分散だけ）・LN（素の LayerNorm）はどれも床のままで、付録 E に回した。
bias の経路は図 3 で生の |b_2| を見る。
元データ: results/relu_doors_h_ref_0920/{{ref,H}}/per_task.csv・results/relu_doors_0919/<arm>/per_task.csv。
"""

NOTE_E = f"""付録 E 図. 開ける扉を変えた対照。線は seed 中央値、{S.SEED_BAND_NOTE}。薄い帯は登録窓 t31-50。
H は中間層の中心化だけ、CS は同じ統計量の分散の側だけ（RMSNorm 的・中心化なし）、LN は文献どおりの
LayerNorm。いずれも後期窓の中央値は ref と同じ床（H 0.113・CS 0.113・LN 0.206）で、
本文の CH 0.987 に届かない。登録判定 N = NEED_CH（入力と中間層の両方が要る）の対照側。
H は ref と同じく sd_2 が潰れる型（t50 で 0.079）で、沈んで死ぬのではない。
LN が正側で線形化して実効階数が潰れる読みは relu_doors 結果ノート §3.2。
元データ: results/relu_doors_h_ref_0920/H/per_task.csv・results/relu_doors_0919/{{CS,LN}}/per_task.csv。
"""

if __name__ == "__main__":
    for layout in ("1x4", "2x2"):
        print(S.save(build(layout), f"fig02_doors_{layout}", NOTE))
    print(S.save(build("1x4", CTRL_ARMS,
                       "付録 E 図  開ける扉を変えた対照 — H・CS・LN（RL-CIFAR / MLP・10 seed）", "付録"),
                 "fig02e_doors_controls", NOTE_E))

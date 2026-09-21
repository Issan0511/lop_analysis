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
MAIN_ARMS = ("ref", "C", "CH", "CHB")            # 本文: 主張の梯子
CTRL_ARMS = ("ref", "H", "CS", "LN")             # 付録 E: 開ける扉を変えた対照
PANELS = [("sink_ratio_l2", "zbar2r"), ("dead_frac_l2", "dead2"), ("bias_over_sd_l2", "bias2")]
AX = {"zbar2r": "第 2 層の沈下 $\\bar z_2/\\mathrm{sd}_2$", "dead2": S.AXIS["dead2"],
      "bias2": S.AXIS["bias2"]}


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
            S.band(ax, t, Y, S.COLOR[arm], None, agg="median", ls=S.LS.get(arm, "-"))
        ax.set_ylabel(AX[key])
        S.grade(ax, "column")
    axes[1].set_title("(b) 第 2 層の沈下")
    axes[2].set_title("(c) 第 2 層の死亡率")
    axes[3].set_title("(d) 第 2 層の bias")
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].axhline(0, color="#999999", lw=0.7, zorder=0)
    axes[1].set_yticks([1e1, 0, -1e0, -1e3, -1e6, -1e9, -1e12])
    axes[2].set_ylim(-0.02, 1.02)
    axes[3].set_yscale("log")

    for ax in axes:
        ax.set_xlim(1, 50)
        ax.set_xlabel(S.AXIS["task"])
        S.window_span(ax, *WIN)
    fig.suptitle(title or "図 2  中心化の扉 — RL-CIFAR / MLP・50 課題・10 seed", fontsize=12)
    return fig


NOTE = f"""図 2. 中心化の扉。本文に出すのは主張の梯子 ref -> C -> CH -> CHB の 4 腕。
線は seed 中央値、{S.SEED_BAND_NOTE}。薄い帯は登録窓 t31-50。
(a) の格は登録（M = RESCUED・N = NEED_CH）、(b) の 3 枚は登録列の読み。
符号検定は D2 = H_HELPS（中央値 +0.874・10/10・p = 0.00195）、D1/D3/D4 = TIE。
B_ROUTE = BIAS_TAKES_OVER（中央値 +0.179・10/10・p = 0.00195）。
H（中間層だけ）・CS（分散だけ）・LN（素の LayerNorm）はどれも床のままで、付録 E に回した。
元データ: results/relu_doors_h_ref_0920/{{ref,H}}/per_task.csv・results/relu_doors_0919/<arm>/per_task.csv。
"""

NOTE_E = f"""付録 E 図. 開ける扉を変えた対照。線は seed 中央値、{S.SEED_BAND_NOTE}。薄い帯は登録窓 t31-50。
H は中間層の中心化だけ、CS は同じ統計量の分散の側だけ（RMSNorm 的・中心化なし）、LN は文献どおりの
LayerNorm。いずれも後期窓の中央値は ref と同じ床（H 0.113・CS 0.113・LN 0.206）で、
本文の CH 0.987 に届かない。登録判定 N = NEED_CH（入力と中間層の両方が要る）の対照側。
LN が正側で線形化して実効階数が潰れる読みは relu_doors 結果ノート §3.2。
元データ: results/relu_doors_h_ref_0920/H/per_task.csv・results/relu_doors_0919/{{CS,LN}}/per_task.csv。
"""

if __name__ == "__main__":
    for layout in ("1x4", "2x2"):
        print(S.save(build(layout), f"fig02_doors_{layout}", NOTE))
    print(S.save(build("1x4", CTRL_ARMS,
                       "付録 E 図  開ける扉を変えた対照 — H・CS・LN（RL-CIFAR / MLP・10 seed）", "付録"),
                 "fig02e_doors_controls", NOTE_E))

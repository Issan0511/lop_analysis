#!/usr/bin/env python3
"""付録 J  層別キメラ（S-A）— 7 セルの課題別 online と層別の p⁺・Q・‖µ₂‖（図表一覧 §4 J）。

第 1 層 × 第 2 層 = {leaky .1, ELU, GELU} × {leaky .1, ELU} と GG。raw・R=10・seed 0–9・50 課題。
決定 11 により付録へ配置。両層 leaky と片層 ELU の比較で、微分の床だけの必要性は未同定。
元データ: results/layer_chimera_cifar_0921/{<cell>/per_task.csv,measure.csv,per_seed.csv,verdict.json}
格: Q1 = MIXED（登録）。C1・C2 の対応差は登録。層別の量は登録列の読み。
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

CELLS = ["LL", "EL", "GL", "LE", "EE", "GE", "GG"]
ACT = {"L": "leaky .1", "E": "ELU", "G": "GELU"}
COL1 = {"L": S.COLOR["LR"], "E": S.COLOR["ELU"], "G": S.COLOR["GELU"]}
LS2 = {"L": "-", "E": "--", "G": ":"}
RUN = "layer_chimera_cifar_0921"


def label(cell):
    return f"{cell}（{ACT[cell[0]]} → {ACT[cell[1]]}）"


def load():
    pt = {c: D.per_task(f"{RUN}/{c}/per_task.csv") for c in CELLS}
    m = pd.read_csv(D.RES / RUN / "measure.csv")
    ps = pd.read_csv(D.RES / RUN / "per_seed.csv")
    v = json.loads((D.RES / RUN / "verdict.json").read_text())
    return pt, m, ps, v


def panel_online(ax, pt, ps):
    for c in CELLS:
        t, Y = D.matrix(pt[c], "online_acc")
        S.band(ax, t, Y, COL1[c[0]], label(c), agg="median", ls=LS2[c[1]], alpha=0.10)
    S.window_span(ax, 31, 50, "登録窓 t31–50")
    ax.axhline(0.5, color="#999999", lw=0.9, ls=":")
    S.acc_axis(ax)
    ax.set_xlabel(S.AXIS["task"])
    ax.set_title("(a) 課題別 online（色 = 第 1 層・線種 = 第 2 層）")
    ax.legend(loc="center right", fontsize=6.6)
    S.grade(ax, "registered", "Q1 = MIXED（生存は LL だけ）", loc="upper left")


def grouped(ax, m, col, task, ylab, ttl, logy=False):
    sub = m[m.task == task]
    w = 0.38
    for j, layer in enumerate((1, 2)):
        vals, lo, hi = [], [], []
        for c in CELLS:
            s = sub[(sub.cell == c) & (sub.layer == layer)][col]
            vals.append(s.median()); lo.append(s.min()); hi.append(s.max())
        x = np.arange(len(CELLS)) + (j - 0.5) * w
        ax.bar(x, vals, width=w * 0.92, color=[COL1[c[0]] for c in CELLS], alpha=0.95 if layer == 1 else 0.45,
               label=f"第 {layer} 層", edgecolor="none")
        ax.vlines(x, lo, hi, color="#222222", lw=0.9)
    ax.set_xticks(range(len(CELLS))); ax.set_xticklabels(CELLS)
    ax.set_ylabel(ylab)
    if logy:
        ax.set_yscale("log")
    ax.set_title(ttl)
    ax.legend(loc="upper right", fontsize=7)


def build():
    pt, m, ps, v = load()
    fig, axes = S.grid(2, 2, 11.5, 8.4)
    panel_online(axes[0], pt, ps)
    grouped(axes[1], m, "p_pos", 50, "正側にいる (unit, 画像) の割合 $p^+$", "(b) t50 の $p^+$（濃 = 第 1 層・淡 = 第 2 層）")
    S.grade(axes[1], "column", "登録列の読み", loc="upper left")
    grouped(axes[2], m, "Q", 50, "低応答の割合 Q（$|\\varphi'|<10^{-6}$）", "(c) t50 の Q（leaky の Q は恒真 0）")
    S.grade(axes[2], "column", "登録列の読み", loc="upper left")
    # (d) ‖µ₂‖ の推移（第 2 層の入力平均のノルム・seed 中央値）
    ax = axes[3]
    for c in CELLS:
        sub = m[(m.cell == c) & (m.layer == 2)]
        t, Y = D.matrix(sub, "mu_norm")
        S.band(ax, t, Y, COL1[c[0]], label(c), agg="median", ls=LS2[c[1]], alpha=0.08)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 3e3)      # GELU 列は seed 最小値が 0 近くまで落ちる。帯の下端は 1e-3 で切る（キャプションに明記）
    ax.set_xlabel(S.AXIS["task"])
    ax.set_ylabel(S.AXIS["mu2"])
    ax.set_title("(d) $\\|\\mu_2\\|$ の推移: 育てるのは第 2 層の活性化（Q3）")
    S.grade(ax, "registered", "Q3: EE − EL @t10 +450（10/10）", loc="lower right")
    fig.suptitle("付録 J  層別キメラ S-A — RL-CIFAR / MLP・raw・10 seed・50 課題", fontsize=12)
    return fig


NOTE = """付録 J. 層別キメラ（S-A layer_chimera_cifar_0921・raw・R=10・seed 0–9・50 課題・事前登録）。
第 1 層 × 第 2 層 = {leaky .1, ELU, GELU} × {leaky .1, ELU} と参照 GG。色は第 1 層、線種は第 2 層。
(a) 課題別 online（seed 中央値・帯は seed の全範囲・信頼区間ではない）。登録窓 t31–50。生き残るのは
LL（0.728）だけで、どちらか一方の層を ELU にするだけで落ちる（EL 0.140・LE 0.100・EE 0.101、GELU 列は t1 から床）。
主ラベル Q1 = MIXED（起案の BOTH_LAYERS_THREE_TYPES は外れ）。
登録した対応差: LL − LE +0.627・LL − EL +0.589（ともに 10/10）。比較した raw の 7 セルでは両層 leaky の腕だけが生存基準を満たした。
(b) t50 の p⁺（正側にいる (unit, 画像) の割合・seed 中央値と全範囲）。LL の第 1 層も p⁺ 0.021 まで沈むが窓 0.73 を保つ。
前向き値と微分は未分離であり、微分の床だけを必要条件とはしない。MNIST の FLOOR_IN_DEEP_LAYER はこの CIFAR の比較へは移らなかった。
(c) t50 の Q（|φ′| < 1e−6 の割合）。leaky 層の Q は恒真 0 なので層の先後には使えない（登録列の読み）。
(d) ‖µ₂‖（第 2 層の入力平均のノルム・seed 中央値と全範囲）。GELU 列の seed 最小値は 0 近くまで落ちるので、帯の下端は 1e−3 で切ってある。育てるのは第 2 層の活性化（Q3: EE − EL @t10 +450・10/10）。LE は
第 1 層が全セル中いちばん健康なまま第 2 層だけ死に、‖µ₂‖ 847 まで伸びる（raw で「第 2 層の死」を単独で作る）。
限定: raw のみ・1 箱。LL・EE・GG の水準はバトルの R=20 の実現値と bit 一致で新しい標本ではない。
ε 介入（le_eps_cifar_0922）: LE の第 1 層の伸びは Adam の ε を 1e−8 → 1e−3 にしても 4–5 分の 1 に縮むだけで
消えない（Adam は速さを決めるが存在は決めない・REDUCED/MONOTONE）。
元データ: results/layer_chimera_cifar_0921/{<cell>/per_task.csv,measure.csv,per_seed.csv,verdict.json}。
"""

if __name__ == "__main__":
    print(S.save(build(), "figJ_layer_chimera", NOTE))

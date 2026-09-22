#!/usr/bin/env python3
"""付録 F  kunekune の裾 — 帯の外に出た質量と K1・K3（図表一覧 §4 F・B12）。

KKT1（kunekune）は帯 [−3π/2, +π/2] の外を傾き 1 の直線にしてある。裾が働いたかは同定していない。
ここに置くのは、帯の外に出た (unit, 画像) の割合（バトルの登録列の読み）と、K1（周期）・K3（裾）の登録。
元データ: results/rlcifar_mlp_battle_0918/summary.md の phase の表（seed 中央値）と K ラベル
格: 帯外の割合は登録列の読み、K1・K3 は登録。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D
import style as S

ARMS = ["SNA", "KKA", "KKA23", "KKT1"]
LS = {"raw": "-", "std": "--"}
TASKS = [1, 10, 25, 50]


def phase_table() -> pd.DataFrame:
    txt = (D.RES / "rlcifar_mlp_battle_0918" / "summary.md").read_text(encoding="utf-8")
    sec = txt.split("## phase:")[1].split("\n## ")[0]
    rows = []
    for line in sec.splitlines():
        m = re.match(r"\|\s*(\w+)_(raw|std)\s*\|\s*t(\d+)_l(\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([-\d.]+)\s*\|", line)
        if m:
            arm, cond, t, l, above, below, theta = m.groups()
            rows.append(dict(arm=arm, cond=cond, task=int(t), layer=int(l), above=float(above),
                             below=float(below), theta=float(theta)))
    return pd.DataFrame(rows)


def k_labels() -> dict:
    txt = (D.RES / "rlcifar_mlp_battle_0918" / "summary.md").read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"\*\*K([13]) \((raw|std)\)\*\*: `(\w+)` — (.+?) median ([-+\d.]+), (\d+)/10 seeds, p=([\d.]+)", txt):
        k, cond, lab, what, med, pos, p = m.groups()
        out[(int(k), cond)] = dict(label=lab, what=what.strip(), median=float(med), pos=int(pos), p=float(p))
    return out


def panel_layer(ax, ph, layer):
    for arm in ARMS:
        for cond in ("raw", "std"):
            sub = ph[(ph.arm == arm) & (ph.cond == cond) & (ph.layer == layer)].sort_values("task")
            ax.plot(sub.task, sub.above + sub.below, marker="o", ms=3.5, lw=1.4, ls=LS[cond],
                    color=S.COLOR[arm], label=f"{arm}・{cond}")
    ax.set_xscale("log")
    ax.set_xticks(TASKS); ax.set_xticklabels([str(t) for t in TASKS])
    ax.set_xlabel(S.AXIS["task"])
    ax.set_ylabel("帯の外の (unit, 画像) の割合")
    ax.set_ylim(0, 0.26)
    ax.set_title(f"({'ab'[layer-1]}) 第 {layer} 層: 帯 [−3π/2, +π/2] の外")
    if layer == 1:
        ax.legend(loc="upper right", fontsize=6.8, ncol=2)
    S.grade(ax, "column", "seed 中央値", loc="upper left" if layer == 2 else "lower left")


def panel_k(ax, ph, K):
    """(c) t50 の上側／下側の内訳と、K1・K3 の登録。"""
    sub = ph[(ph.task == 50)]
    x = 0
    ticks, labels = [], []
    for cond in ("raw", "std"):
        for arm in ARMS:
            for layer in (1, 2):
                r = sub[(sub.arm == arm) & (sub.cond == cond) & (sub.layer == layer)].iloc[0]
                ax.bar(x, r.above, color=S.COLOR[arm], width=0.8, alpha=0.9 if layer == 1 else 0.5)
                ax.bar(x, r.below, bottom=r.above, color="#333333", width=0.8, alpha=0.35 if layer == 1 else 0.2)
                ticks.append(x); labels.append(f"{arm}\n{cond} l{layer}")
                x += 1
            x += 0.4
        x += 0.8
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=5.6, rotation=90)
    ax.set_ylabel("t50 の帯外の割合（色 = +π/2 の上・黒 = −3π/2 の下）")
    ax.set_title("(c) t50 の内訳と K1・K3")
    lines = []
    for (k, cond), r in sorted(K.items()):
        lines.append(f"K{k} {cond}: {r['label']}（{r['what']} 中央値 {r['median']:+.4f}・{r['pos']}/10・p {r['p']:.3f}）")
    ax.text(0.01, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=6.4,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f4", ec="#cccccc", lw=0.6))
    ax.set_ylim(0, 0.12)
    S.grade(ax, "registered", "K1・K3", loc="lower right")


def build():
    ph = phase_table()
    K = k_labels()
    fig, axes = S.grid(1, 3, 13.0, 4.6)
    panel_layer(axes[0], ph, 1)
    panel_layer(axes[1], ph, 2)
    panel_k(axes[2], ph, K)
    fig.suptitle("付録 F  kunekune の裾 — RL-CIFAR / MLP バトル・10 seed", fontsize=12)
    return fig


NOTE = """付録 F. kunekune の裾（B12）。帯 [−3π/2, +π/2] は KKA（Snake の 1 周期を切り出した帯）と同じ定義で、
KKT1（kunekune）は帯の外を傾き 1 の直線にする。
(a)(b) 帯の外に出た (unit, 画像) の割合（2α_i z の位相で測る・seed 中央値・登録列の読み）。t1 の 10–19% から
    t10 以降は第 1 層 3–8%・第 2 層 1.5–3% に落ち、以後ほぼ一定。共通位置は帯の中に居続ける。
(c) t50 の内訳（上側 +π/2 の外が色、下側 −3π/2 の外が黒）。登録した K1（周期の有無: KKA − SNA）は raw で
    PERIOD_FREE、std で PERIOD_HURTS（+0.0002・10/10）。K3（裾の傾き: KKT1 − KKA）は raw・std とも TAIL_FREE。
    K3 は「裾を使う入力が居ない」のもとでの結果で、裾単独の寄与は同定していない。「裾が救った」も
    「裾は無関係」も結論しない（草案 §4・§7）。
元データ: results/rlcifar_mlp_battle_0918/summary.md（phase の表と K ラベル）。
"""

if __name__ == "__main__":
    print(S.save(build(), "figF_kunekune_tail", NOTE))

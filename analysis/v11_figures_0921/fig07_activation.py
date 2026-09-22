#!/usr/bin/env python3
"""図 7 活性化 — 定義・13 腕のバトル・先行層（図表一覧 §2）。

(a) Snake・KKA・KKT1 の phi と phi'（定義図・alpha=1 に固定して形だけを見せる）
(b) 13 腕 × raw/std の窓（点と seed 範囲）
(c) 先行層: ELU・GELU・SILU の T* 分布（どちらの層が先に落ちるか）
元データ: results/rlcifar_mlp_battle_0918/<arm>/per_task.csv・results/cifar_ledger_0920/per_seed.csv
格: 登録（窓は tasks 31-50・符号検定）。集約は seed 中央値（§2.5-1）
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

PI = np.pi
BATTLE = ("SNA", "KKA", "KKA23", "KKT1", "RSL", "LK03", "SL", "LR", "LK001", "GELU", "SILU", "ELU", "R")
WIN = (31, 50)


def phi(z, kind, a=1.0):
    snake = z + np.sin(a * z) ** 2 / a
    th = 2 * a * z
    if kind == "snake":
        return snake
    if kind == "kk":
        left = 2 * z + (0.75 * PI) / a + 0.5 / a
        right = 2 * z - (0.25 * PI) / a + 0.5 / a
        return np.where(th < -1.5 * PI, left, np.where(th > 0.5 * PI, right, snake))
    return np.where(th < -2 * PI, z, np.where(th > PI, z + 1 / a, snake))   # kkt1


def dphi(z, kind, a=1.0):
    th = 2 * a * z
    s = 1 + np.sin(th)
    if kind == "snake":
        return s
    if kind == "kk":
        return np.where((th >= -1.5 * PI) & (th <= 0.5 * PI), s, 2.0)
    return np.where((th >= -2 * PI) & (th <= PI), s, 1.0)


def battle_windows() -> pd.DataFrame:
    rows = []
    for arm in BATTLE:
        d = D.per_task(f"rlcifar_mlp_battle_0918/{arm}/per_task.csv")
        w = d[d.task.between(*WIN)]
        for (cond, seed), g in w.groupby(["cond", "seed"]):
            rows.append({"arm": arm, "cond": cond, "seed": seed, "window": g.online_acc.mean()})
    return pd.DataFrame(rows)


def build():
    fig = plt.figure(figsize=(13.0, 7.0), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.55, 1.0])
    ax_phi = fig.add_subplot(gs[0, 0])
    ax_dphi = fig.add_subplot(gs[1, 0])
    ax_win = fig.add_subplot(gs[:, 1])
    ax_raw = fig.add_subplot(gs[0, 2])
    ax_std = fig.add_subplot(gs[1, 2], sharex=ax_raw, sharey=ax_raw)

    # (a) 定義図
    z = np.linspace(-6, 4, 1600)
    for kind, key, lab in (("snake", "SNA", "Snake"), ("kk", "KKA", "KKA"), ("kkt1", "KKT1", "KKT1")):
        ax_phi.plot(z, phi(z, kind), color=S.COLOR[key], lw=1.8, label=lab)
        ax_dphi.plot(z, dphi(z, kind), color=S.COLOR[key], lw=1.8)
    ax_phi.plot(z, np.maximum(z, 0), color=S.GREY, lw=1.0, ls="--", label="ReLU")
    ax_dphi.plot(z, (z > 0).astype(float), color=S.GREY, lw=1.0, ls="--")
    ax_phi.set_title("(a) $\\varphi$（$\\alpha=1$ に固定）")
    ax_dphi.set_title("(a) $\\varphi'$")
    ax_phi.legend(loc="upper left")
    for ax in (ax_phi, ax_dphi):
        ax.set_xlabel("$z$")
        ax.axhline(0, color="#bbbbbb", lw=0.7)
        ax.axvline(0, color="#bbbbbb", lw=0.7)
    S.grade(ax_phi, "column", "定義", loc="lower right")

    # (b) 13 腕 × raw/std
    w = battle_windows()
    order = w[w.cond == "std"].groupby("arm").window.median().sort_values().index.tolist()
    y = np.arange(len(order))
    for cond, dx, mk in (("raw", -0.17, "o"), ("std", 0.17, "s")):
        sub = w[w.cond == cond]
        med = sub.groupby("arm").window.median()
        lo, hi = sub.groupby("arm").window.min(), sub.groupby("arm").window.max()
        cols = [S.COLOR[a] for a in order]
        ax_win.hlines(y + dx, lo[order], hi[order], color=cols, lw=3, alpha=0.30)
        ax_win.scatter(med[order], y + dx, color=cols, s=34, marker=mk, zorder=3,
                       edgecolor="white", linewidth=0.5)
        # 8 腕が 0.85-0.99 に詰まって順位が読めないので、数値を添える（軸は 0-1 のまま）
        S.value_labels(ax_win, y + dx, med[order].to_numpy(), "{:.3f}", flip_at=0.55)
    ax_win.axvline(0.5, color="#999999", lw=0.9, ls=":")
    ax_win.annotate("崩壊の線 0.5", xy=(0.5, len(order) - 0.6), fontsize=7, color="#666666", ha="left")
    ax_win.set_yticks(y); ax_win.set_yticklabels(order)
    ax_win.set_xlim(0, 1.02); ax_win.set_xlabel("後期窓のオンライン精度（t31–50）")
    S.breathe(ax_win, "y", 0.03)
    ax_win.set_title("(b) 13 腕 × raw / std の窓")
    ax_win.grid(axis="y", alpha=0)
    ax_win.legend(handles=[plt.Line2D([], [], marker=m, ls="", color="#555555", label=c)
                           for c, m in (("raw", "o"), ("std", "s"))], loc="lower right")
    S.grade(ax_win, "registered", "A = SNA_BEATEN（両 cond）", loc="upper right")

    # (c)(d) 先行層 — 低応答（|phi'| < 1e-6）のユニットの割合が 0.5 を越える時刻。
    # 登録の T* はこの線を最初に越えた課題で、どちらの層が先かが A6 の判定。
    pt = pd.read_csv(D.RES / "cifar_ledger_0920" / "per_task.csv")
    ps = pd.read_csv(D.RES / "cifar_ledger_0920" / "per_seed.csv")
    arms = ["ELU", "GELU", "SILU"]
    for ax, cond, tag in ((ax_raw, "raw", "(c)"), (ax_std, "std", "(d)")):
        for arm in arms:
            for layer, ls in ((1, "-"), (2, "--")):
                s = pt[(pt.arm == arm) & (pt.cond == cond) & (pt.layer == layer) & (pt.task <= 10)]
                m = s.pivot_table(index="task", columns="seed", values="train_low_mean")
                ax.plot(m.index, np.median(m.to_numpy(), 1), color=S.COLOR[arm], ls=ls, lw=1.8,
                        label=arm if layer == 1 else None)
        ax.axhline(0.5, color="#999999", lw=0.9, ls=":")
        row = ps[(ps.cond == cond) & (ps.arm.isin(arms))]
        first = row.layer.mode().iloc[0]
        n = int((row.layer == first).sum())
        name = {"L1_FIRST": "第 1 層が先", "L2_FIRST": "第 2 層が先"}[first]
        ax.set_title(f"{tag} 低応答ユニットの割合 — {cond}")
        ax.set_ylabel("低応答の割合")
        ax.set_ylim(0, 1)
        S.breathe(ax)
        ax.set_xlim(0, 10)
        S.grade(ax, "registered", f"{name}　{n}/{len(row)}",
                loc="center right" if cond == "raw" else "lower right")
    handles, labels = ax_raw.get_legend_handles_labels()
    leg = ax_std.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
                        title="実線 = 第 1 層・破線 = 第 2 層", columnspacing=1.2, handlelength=1.4)
    leg.get_title().set_fontsize(7.5)
    ax_std.set_xlabel(S.AXIS["task"])
    ax_raw.tick_params(labelbottom=False)

    fig.suptitle("図 7  活性化 — RL-CIFAR / MLP・50 課題・10 seed", fontsize=12)
    return fig


NOTE = """図 7. 活性化。(a) は定義図で、実装の alpha は unit ごとに c/W で動くが、ここでは形を見せるため
alpha=1 に固定している（測定値ではない）。
(b) 点は seed 中央値、線は seed の全範囲（信頼区間ではない）。窓は登録どおり t31-50、丸が raw、
四角が std。0.5 は登録の崩壊の線。登録判定は両 cond とも A = SNA_BEATEN。
腕が床（0.10-0.12）と上（0.85-0.99）の二つに分かれて位置から順位が読めないので、軸は 0-1 に
固定したまま（規約 §2.5-3）数値を点のそばに添えた。
(c)(d) 低応答 = 訓練時の微分の絶対値が 1e-6 未満のユニットの割合（層ごと・seed 中央値）。
登録の T* は、この割合が 0.5 を越えた最初の課題（t1-t10 の範囲で探す）。raw では第 1 層が先に
越え（3 腕 30/30）、std では第 2 層が先に越える（30/30）。std の第 1 層は 0.5 の線の少し下で
止まり、越えないことが多い。R は初めから過半が低応答で、この規則では先後が決まらない。
元データ: results/rlcifar_mlp_battle_0918/<arm>/per_task.csv・results/cifar_ledger_0920/per_seed.csv。
"""

if __name__ == "__main__":
    print(S.save(build(), "fig07_activation", NOTE))

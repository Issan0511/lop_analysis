#!/usr/bin/env python3
"""図 7 W 増大 -> 進行 — A6 の帳簿と S5 の層別上限（図表一覧 §2）。

(a) ELU/std 第 2 層の 4 項（主窓 = 0->T* と補助窓 = 10-50 の積み上げ）
(b) S5 の 5 腕の課題別 online と ||mu_2||
元データ: results/cifar_ledger_0920/{windows.csv,window_summary.csv}・
          results/cap_cifar_ee_0920/{per_task.csv,verdict.json}
格: 登録。集約は cap 側が平均 + sd の区間なので平均（§2.5-1）
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

TERMS = [("upstream", "上流", "#2f6b8f"), ("self", "自己", "#c97b3c"),
         ("cross", "交差", "#7b9e6b"), ("bias", "bias", "#9a6ba8")]
CAP_ARMS = ("ref", "cap1", "cap2", "cap12", "cap12_bfix")


def ledger_shares() -> pd.DataFrame:
    w = pd.read_csv(D.RES / "cifar_ledger_0920" / "windows.csv")
    w = w[(w.arm == "ELU") & (w.cond == "std") & (w.layer == 2)]
    rows = []
    prim = w[w.role == "PRIMARY"]
    rows.append({"window": "主窓 0→T*", **{k: prim[f"{k}_share"].median() for k, *_ in TERMS}})
    for a, b, lab in ((0, 10, "補助窓 0–10"), (10, 50, "補助窓 10–50")):
        aux = w[(w.role.str.contains("REPORT_ONLY")) & (w.start == a) & (w.end == b)]
        rows.append({"window": lab, **{k: aux[f"{k}_share"].median() for k, *_ in TERMS}})
    return pd.DataFrame(rows).set_index("window")


def build():
    fig = plt.figure(figsize=(13.0, 4.2), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.1, 1.1])
    ax_l = fig.add_subplot(gs[0, 0])
    ax_o = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[0, 2])

    # (a) 帳簿
    sh = ledger_shares()
    x = np.arange(len(sh))
    bottom = np.zeros(len(sh))
    for key, lab, col in TERMS:
        v = sh[key].to_numpy()
        ax_l.bar(x, v, 0.55, bottom=bottom, color=col, label=lab)
        for xi, (b, vi) in enumerate(zip(bottom, v)):
            if abs(vi) > 0.06:
                ax_l.text(xi, b + vi / 2, f"{vi:.0%}", ha="center", va="center",
                          fontsize=8, color="white")
        bottom += v
    ax_l.set_xticks(x); ax_l.set_xticklabels(sh.index, fontsize=8)
    ax_l.set_ylabel("沈下に占める割合")
    ax_l.axhline(0, color="#999999", lw=0.8)
    ax_l.set_title("(a) 帳簿（ELU / std・第 2 層）")
    ax_l.set_ylim(0, 1.0)
    ax_l.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                columnspacing=1.1, handlelength=1.2)
    ax_l.grid(axis="x", alpha=0)
    S.grade(ax_l, "registered", "A6 · MIXED", loc="lower right")

    # (b) S5
    d = D.per_task("cap_cifar_ee_0920/per_task.csv")
    v = json.loads((D.RES / "cap_cifar_ee_0920" / "verdict.json").read_text())
    for arm in CAP_ARMS:
        sub = d[d.arm == arm]
        t, Y = D.matrix(sub, "online_acc")
        S.band(ax_o, t, Y, S.COLOR[arm], S.LABEL.get(arm, arm), agg="mean", ls=S.LS.get(arm, "-"))
        t, M = D.matrix(sub, "mu2_norm")
        S.band(ax_m, t, M, S.COLOR[arm], None, agg="mean", ls=S.LS.get(arm, "-"))
    S.acc_axis(ax_o)
    ax_o.set_title("(b) 課題をまたぐ学習")
    ax_o.legend(loc="center left")
    ax_m.set_title("(c) 第 2 層の入力の平均")
    ax_m.set_ylabel(S.AXIS["mu2"])
    ax_m.set_yscale("log")
    for ax in (ax_o, ax_m):
        ax.set_xlim(1, 50); ax.set_xlabel(S.AXIS["task"])
        S.window_span(ax, 31, 50)
    S.grade(ax_o, "registered", v["label"], loc="center right")
    S.grade(ax_m, "column", loc="center right")
    fig.suptitle("図 7  W 増大 → 進行 — RL-CIFAR / MLP・ELU / std・10 seed", fontsize=12)
    return fig


NOTE = """図 7. W 増大から進行へ。
(a) A6 の帳簿。ELU/std の第 2 層の沈下を 4 項に割ったときの各項の割合（seed 中央値）。
    主窓は登録の 0→T*（T* 中央値 t2）、補助窓は固定窓 0-10 と 10-50（REPORT_ONLY で主判定を
    置き換えない）。登録判定は MIXED: 主窓では交差項が 60% で最大、自己 23%、上流 16.5%。
    上流が大半になるのは低応答に達したあとの補助窓（0-10 で 73%・10-50 で 99%）。
(b) S5。線は seed 平均、帯は seed の全範囲（信頼区間ではない・cap_cifar_ee の登録が平均と sd の
    区間なので平均を描く）。薄い帯は登録窓 t31-50。登録判定は RESCUED（cap12 が 10/10）。
元データ: results/cifar_ledger_0920/windows.csv・results/cap_cifar_ee_0920/per_task.csv。
"""

if __name__ == "__main__":
    print(S.save(build(), "fig06_growth", NOTE))

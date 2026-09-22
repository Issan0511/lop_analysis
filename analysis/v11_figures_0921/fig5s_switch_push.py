#!/usr/bin/env python3
"""正式な図 6。凍結窓の位置を比較する（raw LE・登録済みデータのみ）。"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import data as D
import style as S


def build():
    root = D.RES / "switch_push_cifar_0922"
    verdict = json.loads((root / "verdict.json").read_text())
    tail = pd.read_csv(root / "tail.csv")
    paired = tail[(tail.task == 5) & tail.arm.isin(["LE_sw750", "LE_md750"])].pivot(
        index="seed", columns="arm", values="mu2").sort_index()
    assert paired.shape == (10, 2) and not paired.isna().any().any()
    assert (paired.LE_sw750 < paired.LE_md750).sum() == verdict["Q1"]["pos"]
    for arm in paired:
        assert np.isclose(paired[arm].median(), verdict["Q1"]["median_" + arm])
    fig, (a, b) = S.grid(1, 2, 12.8, 4.6)
    colors = [S.OKABE["blue"], S.OKABE["vermillion"]]
    for _, row in paired.iterrows():
        a.plot([0, 1], row[["LE_sw750", "LE_md750"]], color="#bbbbbb", lw=.8, zorder=1)
    for x, arm in enumerate(["LE_sw750", "LE_md750"]):
        a.scatter(np.full(10, x), paired[arm], color=colors[x], s=25, zorder=3)
        med = paired[arm].median()
        a.hlines(med, x-.12, x+.12, color="black", lw=2.2, zorder=4)
        a.annotate(f"中央値 {med:.0f}", (x, med), (12, 4), textcoords="offset points")
    a.set(xlim=(-.4, 1.65), ylim=(0, 1050), ylabel="第 5 課題末の $\\|\\mu_2\\|$",
          title="(a) 同量の凍結、異なる位置（K=750）")
    a.set_xticks([0, 1], ["切替直後", "課題中央"])
    S.grade(a, "registered", "Q1: SWITCH_MAKES_THE_PUSH", loc="upper left")

    arms = [
        ("LE_ref", "無凍結参照", "#222222", "-"),
        ("LE_sw750", "切替直後 K=750", colors[0], "-"),
        ("LE_md750", "課題中央 K=750", colors[1], "--"),
        ("LE_sw7500", "切替直後 K=7500", S.OKABE["green"], "-"),
        ("LE_md7500", "課題中央 K=7500", S.OKABE["purple"], ":"),
    ]
    for arm, label, color, ls in arms:
        path = D.RES / "layer_chimera_cifar_0921/LE/per_task.csv" if arm == "LE_ref" else root / arm / "per_task.csv"
        d = pd.read_csv(path)
        x, ys = D.matrix(d, "online_acc", tasks=range(1, 51))
        assert ys.shape == (10, 50) and np.isfinite(ys).all()
        if arm in verdict["Q4"]:
            assert np.isclose(np.median(ys[:, 30:50].mean(axis=1)), verdict["Q4"][arm]["window_median"])
        b.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=color, alpha=.075)
        b.plot(x, np.median(ys, axis=0), color=color, ls=ls, lw=1.7, label=label)
    b.set(xlim=(1, 50), ylim=(0, 1.02), xlabel="課題", ylabel="オンライン精度",
          title="(b) 初期の抑制の後も喪失は続く")
    S.window_span(b, 31, 50)
    b.legend(loc="upper right")
    S.grade(b, "column", "Q4: K=7500 でも PARTIAL_RESCUE", loc="lower left")
    fig.suptitle("図 6  凍結窓の位置 — RL-CIFAR / raw・LE（leaky 0.1 → ELU）・10 seed")
    return fig


NOTE = """図 6. 凍結窓の位置。乱数ラベル CIFAR × 二隠れ層 MLP、raw、LE = leaky 0.1→ELU。
(a) 両隠れ層の重み・bias と Adam の m, v を各課題の同じ 750 更新だけ止め、切替直後と課題中央を比較。
点は seed、接続線は同じ seed（10 対）、黒線は中央値。第 5 課題末の ‖µ₂‖ は 141 対 672、無凍結参照は 697。
課題中央−切替直後の対応差の中央値は 539、10/10、両側符号検定 p=0.001953125。
主比較は事前登録の Q1（単一比較、Holm 補正なし）で、半減基準を満たす SWITCH_MAKES_THE_PUSH。
(b) 線は seed 中央値、帯は seed の全範囲（信頼区間ではない）。薄い背景は登録窓 t31–50。
曲線は登録列の記述で、主判定 Q1 と救命判定 Q4 を分ける。切替直後の K=750 は NOT_RESCUED
（窓 0.178）、K=7500 は PARTIAL_RESCUE（0.272）。初期の成長を抑えても完全救済にはならない。
実験全体は LE 6 腕＋LL 2 腕の追加 8 腕、各 10 seed。出力層は常に訓練。
無凍結参照は S-A の既存軌道を再利用しており、新しい標本には数えない。
LL の Q5 は LL_REVERSED（‖µ₂‖(t5) は切替 58.0 対課題中央 30.3）で、この向きは一般化しない。
std・MNIST は未検証。W₁・W₂・読み出し適応の寄与を単独には分離していない。
元データ: results/switch_push_cifar_0922/tail.csv（task=5, mu2）、各腕の per_task.csv（task=1–50, online_acc）、
results/layer_chimera_cifar_0921/LE/per_task.csv。判定: switch_push_cifar_0922/verdict.json の Q1・Q4・Q5。
"""

if __name__ == "__main__":
    print(S.save(build(), "fig5s_switch_push", NOTE))

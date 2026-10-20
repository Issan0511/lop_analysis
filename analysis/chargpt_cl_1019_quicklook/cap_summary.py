#!/usr/bin/env python3
"""cap の下見（seed 1000）の集計。Fable の予測の尺度 3 つ（深い層の φ′・課題の頭 100 更新・序盤→終盤）と水準・fresh gap、絶対の深さ。

  python analysis/chargpt_cl_1019_quicklook/cap_summary.py [--arms C3 ReLUcap SiLU SiLUcap GELU GELUcap]

- φ′: 課題末の probe でのトークン × unit の平均（chain = その課題の系列、uniform = 全課題で同じ一様乱数の系列 = 固定 probe）。
  φ′ < 0.1 の unit の割合も。課題 1 末に対する課題 50 末の比。
- 頭 100 更新: オンライン超過損失の更新 0〜99 の平均を、課題 2〜10 と 41〜50 で平均。
- 悪化: 課題 41〜50 − 課題 1〜10 のオンライン超過損失（verdict と同じ）。水準 = 課題 41〜50。fresh gap = 本体 − 新品（課題 10・25・50）。
- 絶対の深さ: μ の unit 中央値（z の単位）と幅 σ の中央値、μ/σ の中央値。
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_verdict as V  # noqa: E402

Q = ROOT / "results/chargpt_cl_1019/quicklook/M"


def rows_end(d):
    pt = V.rd(d / "per_task.csv")
    n = int(pt[0]["n"])
    E = {}
    for r in V.rd(d / "layer_stats.csv"):
        if int(r["u"]) == n:
            E[(r["probe"], int(r["task"]), int(r["layer"]))] = r
    return pt, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["C3", "ReLUcap", "SiLU", "SiLUcap", "GELU", "GELUcap"])
    a = ap.parse_args()
    arms = [x for x in a.arms if (Q / x / "seed1000" / "per_task.csv").exists()]
    print("腕:", arms, "（seed 1000）")
    print("\n== 学ぶ力（オンライン超過損失）")
    print("腕       | 序盤 1-10 | 終盤 41-50（水準）| 悪化 | 頭100更新 2-10 / 41-50 | fresh gap 10 / 25 / 50")
    for arm in arms:
        d = Q / arm / "seed1000"
        s = V.seed_stats(d, "M")
        pt = V.rd(d / "per_task.csv")
        H = np.array([float(r["H"]) for r in pt])
        on = np.load(d / "online.npz")["loss"].astype(np.float64)
        head = (on[:, :100] - H[:, None]).mean(1)
        fr = {int(r["task"]): float(r["online_all"]) for r in V.rd(d / "fresh.csv")}
        ona = {int(r["task"]): float(r["online_all"]) for r in pt}
        print(f"{arm:8s} | {s['on_early']:.4f} | {s['on_late']:.4f} | {s['on_worse']:+.4f} | {head[1:10].mean():.3f} / {head[40:50].mean():.3f} | "
              + " / ".join(f"{ona[t] - fr[t]:+.3f}" for t in (10, 25, 50)))
    for probe in ("chain", "uniform"):
        print(f"\n== φ′ の平均（{probe} probe）と φ′ < 0.1 の unit の割合。層 0〜3。課題 1 末 → 25 → 50、比 = 50/1")
        for arm in arms:
            _, E = rows_end(Q / arm / "seed1000")
            ph = lambda t, l: float(E[(probe, t, l)]["phi1_mean"])
            lt = lambda t, l: float(E[(probe, t, l)]["units_phi1_lt01"])
            print(f"{arm:8s} " + " | ".join(f"L{l} {ph(1, l):.2f}→{ph(25, l):.2f}→{ph(50, l):.2f} (×{ph(50, l) / ph(1, l):.2f}) <.1: {lt(1, l):.2f}→{lt(50, l):.2f}" for l in range(4)))
    print("\n== 絶対の深さ μ の中央値（z の単位）・幅 σ の中央値・μ/σ の中央値（chain probe、課題 1 / 10 / 25 / 50 の末）")
    for arm in arms:
        _, E = rows_end(Q / arm / "seed1000")
        U = np.load(Q / arm / "seed1000" / "unit_stats.npz")                 # 課題末・chain probe の unit ごとの μ・σ（古い走にもある）
        for l in range(4):
            g = lambda t, k: float(E[("chain", t, l)][k])
            mu = lambda t: float(np.median(U["chain_mu"][t - 1, l])); sd = lambda t: float(np.median(U["chain_sd"][t - 1, l]))
            print(f"{arm:8s} L{l}: μ " + " ".join(f"{mu(t):+8.2f}" for t in (1, 10, 25, 50)) + " | σ " + " ".join(f"{sd(t):7.2f}" for t in (1, 10, 25, 50))
                  + " | μ/σ " + " ".join(f"{g(t, 'r_med'):+.2f}" for t in (1, 10, 25, 50))
                  + (" | cap に当たる unit " + " ".join(f"{g(t, 'cap_frac'):.2f}" for t in (10, 25, 50)) if "cap_frac" in E[("chain", 50, l)] else ""))


if __name__ == "__main__":
    main()

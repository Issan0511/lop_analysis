#!/usr/bin/env python3
"""3 相（課題ごと）: excess = オンライン損失 − 鎖のエントロピー。
  T015  = 50 更新の後ろ向き移動平均が 0.15 以下に初めて届く更新番号（1 始まり。届かなければ 10,000）
  E1000 = 更新 951〜1050 の excess の平均
  ELast = 最後の 1,000 更新（9,001〜10,000）の excess の平均
spec 追補 7 の定義。モジュールとしても使う（phases(loss_row, H)）。"""
import numpy as np


def phases(loss, H, n=10000):
    ex = np.asarray(loss, dtype=np.float64) - H
    ma = np.convolve(ex, np.ones(50) / 50, mode="valid")          # ma[k] = 更新 k+1〜k+50 の平均 → 更新 k+50 での後ろ向き移動平均
    hit = np.nonzero(ma <= 0.15)[0]
    t015 = float(hit[0] + 50) if len(hit) else float(n)
    return t015, float(ex[950:1050].mean()), float(ex[-1000:].mean())


if __name__ == "__main__":
    import csv, sys
    from pathlib import Path
    R = Path(__file__).resolve().parents[2] / "results/chargpt_cl_1019/M/C3"
    rows = {}
    for s in range(10):
        d = R / f"seed{s}"; on = np.load(d / "online.npz")["loss"]; H = [float(r["H"]) for r in csv.DictReader(open(d / "per_task.csv"))]
        P = np.array([phases(on[t], H[t]) for t in range(on.shape[0])])          # [課題, 3]
        fr = {}
        for t in (10, 25, 50):
            f = d / f"fresh_online_t{t}.npz"
            if f.exists(): fr[t] = phases(np.load(f)["loss"], H[t - 1])
        rows[s] = dict(early=P[1:10].mean(0), late=P[40:50].mean(0), fresh=np.mean([fr[t] for t in fr], 0))
    for k in ("early", "late", "fresh"):
        M = np.array([rows[s][k] for s in range(10)])
        print(f"4 ブロック C3（本走 seed 0〜9）{k:5s}: T015 中央 {np.median(M[:,0]):.0f}（SD {M[:,0].std(ddof=1):.0f}）| E1000 中央 {np.median(M[:,1]):.3f}（SD {M[:,1].std(ddof=1):.3f}）| ELast 中央 {np.median(M[:,2]):.3f}（SD {M[:,2].std(ddof=1):.3f}）")

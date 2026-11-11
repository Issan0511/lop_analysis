#!/usr/bin/env python3
"""T（毎回の介入）の記述（登録外・事後）: 課題 30 の末の体の状態（閉じた割合・|φ'| の平均・前活性の標準偏差・標準化した平均 r の中央値）と、
課題ごとの不足・読み出しのノルムの推移。出力: results/stage1_ctrl_1111/describe_T.md"""
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[2]; RAW = W / "results/stage1_ctrl_1111/raw"
ACTS = ["LR", "R", "ELU", "SILU", "GELU"]; NAME = {"LR": "leaky ReLU", "R": "ReLU", "ELU": "ELU", "SILU": "SiLU", "GELU": "GELU"}
ARMS = ["S", "H0", "Q", "R", "P", "A0", "F200"]
L = ["| 活性化 | 軌道 | 閉じた割合（課題 30 の末） | |φ'| の平均 | 前活性の標準偏差の中央値 | r = m/σ の中央値 | 読み出しのノルム（課題 30 の頭） | 不足（課題 21〜30） |", "|---" * 8 + "|"]
for act in ACTS:
    for arm in ARMS:
        cl, dp, sd, rr, vn, df = [], [], [], [], [], []
        for s in range(10):
            b1 = np.load(RAW / f"branch_adam_lr0.001_{act}_s{s}.npz")
            if arm == "S":
                d, key = b1, "S"
            else:
                d, key = np.load(RAW / f"persist_{arm}_adam_lr0.001_{act}_s{s}.npz"), f"{arm}all"
            u = d[f"{key}|t30|units"][-1]          # 課題 30 の末: m, σ, p+, |φ'| の平均, |φ'| の最大
            cl.append((u[2] == 0).mean()); dp.append(u[3].mean()); sd.append(np.median(u[1]))
            rr.append(np.median(u[0] / np.maximum(u[1], 1e-12)))
            vn.append(float(d[f"{key}|t30|vnorm"][0]))
            df.append(np.mean([b1[f"fa|t{t:02d}|eval"][-1, 0] - d[f"{key}|t{t:02d}|eval"][-1, 0] for t in range(21, 31)]))
        f = lambda x, p=2: f"{np.mean(x):.{p}f}"
        L.append(f"| {NAME[act]} | {'S（本線）' if arm == 'S' else arm + 'all'} | {f(cl)} | {f(dp)} | {f(sd)} | {f(rr)} | {f(vn, 1)} | {f(df)} |")
out = W / "results/stage1_ctrl_1111/describe_T.md"
out.write_text("（登録外・事後の記述。10 seed の平均）\n\n" + "\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))

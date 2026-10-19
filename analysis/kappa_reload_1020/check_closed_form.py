#!/usr/bin/env python3
"""measure.py の閉じた式を autograd と照合する（変異対照つき）。
(1) 実ラベルの全バッチ GD 一歩の位置の変位 dm_gd_real
(2) 新ラベルについて平均した一歩 −(η/N)S_exact（勾配はラベルに線形なので、一様な柔らかい的 1/K の勾配と一致する）
(3) 変異対照: K_n の +1 を落とすと（bias の経路を外すと）一致が壊れること
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import measure as M

torch.set_num_threads(1)
worst = 0.0
for act in M.ACTS:
    d = np.load(M.STATES / f"{act}_K10_s0.npz")
    X = M.load_X(d["subset"]); N = X.shape[0]; xbar = X.mean(0); Kn = X @ xbar + 1.0
    for k in (1, 5, 12):
        P = [d[f"end{q}"][k - 1].astype(np.float64) for q in range(4)]
        e = M.evaluate(P, act, X, xbar)
        y_new = d["y"][k].astype(int); Yn = M.onehot(y_new, 10)
        delta = e["g"] * ((e["pi"] - Yn) @ e["W2"])
        dm_real = -(M.ETA / N) * (Kn[:, None] * delta).sum(0)
        S_exact = (Kn[:, None] * e["g"] * (e["pi"] @ e["W2c"])).sum(0)
        dm_exp = -(M.ETA / N) * S_exact
        mut = -(M.ETA / N) * ((Kn - 1.0)[:, None] * delta).sum(0)          # 変異: +1 を落とす

        Xt = torch.tensor(X); tp = [torch.tensor(p, requires_grad=True) for p in P]
        def act_t(z):
            if act == "LR": return torch.where(z > 0, z, 0.1 * z)
            if act == "ELU": return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))
            if act == "R": return torch.relu(z)
            if act == "GELU": return z * 0.5 * (1.0 + torch.erf(z / M.SQRT2))
            if act == "SILU": return z * torch.sigmoid(z)
        def dm_from_target(T):
            for p in tp: p.grad = None
            z = Xt @ tp[0].T + tp[1]; lg = act_t(z) @ tp[2].T + tp[3]
            loss = (lg.logsumexp(-1) - (lg * T).sum(-1)).mean(); loss.backward()
            return -M.ETA * (tp[0].grad.numpy() @ xbar + tp[1].grad.numpy())
        ag_real = dm_from_target(torch.tensor(Yn))
        ag_exp = dm_from_target(torch.full((N, 10), 0.1, dtype=torch.float64))
        e1 = np.abs(ag_real - dm_real).max() / np.abs(ag_real).max()
        e2 = np.abs(ag_exp - dm_exp).max() / np.abs(ag_exp).max()
        e3 = np.abs(ag_real - mut).max() / np.abs(ag_real).max()
        worst = max(worst, e1, e2)
        print(f"{act:4s} k={k:2d}  real rel.err {e1:.2e}  expected rel.err {e2:.2e}  mutant(+1 落とす) rel.err {e3:.2e}")
        assert e1 < 1e-9 and e2 < 1e-9, "閉じた式が autograd と合わない"
        assert e3 > 1e-3, "変異対照が検出されない（検査が空虚）"
print(f"PASS  worst rel.err {worst:.2e}")

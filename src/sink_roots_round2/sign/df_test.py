"""自由度（self-influence）の検算: 保存した切替状態から、独立な一様乱数ラベル R 組で 1 課題（4,000 更新）を学習し直し、
課題末の押し S と旧ラベル整列 S^oh の、ラベルの抽選についての分布を unit ごとに測る。
腕（どのパラメタを学習するか・Adam の状態は保存したものから継続）:
  full : 全部（本来の訓練）
  vc   : 読み出し v と出力 bias c だけ（特徴固定 → 読み出しの自由度だけ = v 経路）
  wb   : 第 1 層 w, b だけ（読み出し固定 → 門の自由度だけ = z 経路）
予言（走の前に derivation.md §P1.5 に固定）: E[S^oh] = 読み出しの自由度 + 門の自由度。
  vc: 正（FWL と同じ向き）、wb: 凸な phi（ELU・leaky）で正、GELU・SiLU は谷の分だけ弱い／負がありうる。full ≈ vc + wb（一次）。
"""
import sys, math
from pathlib import Path
import numpy as np, torch
from common import load_state, activ, gate
torch.set_num_threads(1)
N, BATCH = 1200, 16

def pushes(P, X, act, y, K):
    W1, b1, W2, b2 = [q.detach() for q in P]
    z = X @ W1.T + b1; a = activ(z, act); g = gate(z, act); lg = a @ W2.T + b2; p = lg.softmax(-1)
    mu = X.mean(0); Kn = X @ mu + 1.0; V = W2.T; Vc = V - V.mean(1, keepdim=True)
    B = Kn[:, None] * g
    S = (B * (p @ Vc.T)).sum(0); Soh = (B * Vc[:, y].T).sum(0)
    op = (z > 0).to(z.dtype)                                   # sink_roots_0930 round 2: open-input parts
    S_open = (B * (p @ Vc.T) * op).sum(0); Soh_open = (B * Vc[:, y].T * op).sum(0)
    return S.numpy(), Soh.numpy(), float((lg.argmax(-1) == y).double().mean()), ((z > 0).sum(0) > 0).numpy(), S_open.numpy(), Soh_open.numpy()

def run(path, arm, R, steps=4000, seed=7):
    st = load_state(path); X = st["X"].float(); act = st["act"]; K = st["K"]
    gen = torch.Generator().manual_seed(seed)
    res = []
    for r in range(R):
        P = [q.float().clone().requires_grad_(True) for q in st["P"]]
        train = {"full": [0, 1, 2, 3], "vc": [2, 3], "wb": [0, 1]}[arm]
        opt = torch.optim.Adam([P[j] for j in train], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        for j in train:                                          # Adam の状態を継続
            opt.state[P[j]] = {"step": torch.tensor(st["step"]), "exp_avg": st["mA"][j].float().clone(), "exp_avg_sq": st["vA"][j].float().clone()}
        y = torch.randint(K, (N,), generator=gen); Y = torch.nn.functional.one_hot(y, K).float()
        order = torch.stack([torch.randperm(N, generator=gen) for _ in range(-(-(steps * BATCH) // N))]).reshape(-1)[:steps * BATCH].reshape(steps, BATCH)
        for it in range(steps):
            idx = order[it]
            lg = activ(X[idx] @ P[0].T + P[1], act) @ P[2].T + P[3]
            loss = (lg.logsumexp(-1) - (lg * Y[idx]).sum(-1)).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        res.append(pushes([q.double() for q in P], st["X"], act, y, K))
    return res

if __name__ == "__main__":
    path, arm, R = sys.argv[1], sys.argv[2], int(sys.argv[3])
    res = run(path, arm, R)
    S = np.array([r[0] for r in res]); Soh = np.array([r[1] for r in res]); acc = np.array([r[2] for r in res]); alive = np.array([r[3] for r in res])
    So = np.array([r[4] for r in res]); Soho = np.array([r[5] for r in res])
    al = alive.all(0)
    import os
    np.savez_compressed(Path(os.environ.get("DF_OUT", "out")) / f"df_{Path(path).stem}_{arm}.npz", S=S, Soh=Soh, acc=acc, alive=alive, S_open=So, Soh_open=Soho)
    clop = np.median(np.abs(S[:, al] - So[:, al]) / (np.abs(So[:, al]) + 1e-30))
    clop_oh = np.median(np.abs(Soh[:, al] - Soho[:, al]) / (np.abs(Soho[:, al]) + 1e-30))
    print(f"{Path(path).stem:24s} {arm:4s} |S_closed|/|S_open| median (unit-draws) {clop:.3f}  S^oh {clop_oh:.3f}", flush=True)
    pos_unit = (S[:, al] > 0).mean(0)                          # unit ごとの、抽選について S>0 の確率
    print(f"{Path(path).stem:24s} {arm:4s} R {R} acc {acc.mean():.2f} alive(all draws) {al.sum():3d} | frac S>0 (all unit-draws) {np.mean(S[:, al] > 0):.3f} "
          f"Soh>0 {np.mean(Soh[:, al] > 0):.3f} | units with E[S]>0 {np.mean(S[:, al].mean(0) > 0):.2f} | P_draw(S>0) per unit: min {pos_unit.min():.2f} q10 {np.quantile(pos_unit, 0.1):.2f} med {np.median(pos_unit):.2f} "
          f"| median E[S] {np.median(S[:, al].mean(0)):+.3f} sd_draw/|mean| med {np.median(S[:, al].std(0) / np.abs(S[:, al].mean(0))):.2f}", flush=True)

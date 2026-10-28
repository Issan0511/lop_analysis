"""block0 の注意の q・k の課題をまたぐ増分の整合性（拡散か足し算か）を保存状態だけで測る（CPU float64）。

A_h = Γ W_k,hᵀ W_q,h Γ / √d_head（head ごと）、増分 ΔA = A_{t+10} − A_t（t = 10..90）。
整合性指数 C = ‖A_100 − A_10‖² / Σ‖ΔA‖²（拡散なら 1、完全に整合なら増分の数 9）。
Adam の m（課題末 10・30・60・100）の q・k 行の cos も出す。
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'
SEED = 10252010
D, H, DH = 384, 6, 64
TASKS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load(arm, t):
    return torch.load(RUNS / arm / f'seed{SEED}' / f'ckpt/model_t{t:02d}.pt', map_location='cpu', weights_only=False)


def parts(sd, block=0):
    p = f'layers.block_{block}.layers.'
    W = sd[p + 'attn.layers.qkv.weight'].double()
    b = sd[p + 'attn.layers.qkv.bias'].double()
    g = sd[p + 'norm1.weight'].double()
    beta = sd[p + 'norm1.bias'].double()
    Wq, Wk = W[:D], W[D:2 * D]
    bq = b[:D]
    G = torch.diag(g)
    A = torch.stack([G @ Wk[h * DH:(h + 1) * DH].T @ Wq[h * DH:(h + 1) * DH] @ G / DH ** .5 for h in range(H)])  # (H, D, D)
    bvec = torch.stack([G @ Wk[h * DH:(h + 1) * DH].T @ (Wq[h * DH:(h + 1) * DH] @ beta + bq[h * DH:(h + 1) * DH]) / DH ** .5 for h in range(H)])
    return dict(A=A, b=bvec, Wq=Wq, Wk=Wk, gamma=g,
                fc1_0=sd['layers.block_0.layers.mlp.layers.fc1.weight'].double(),
                fc1_1=sd['layers.block_1.layers.mlp.layers.fc1.weight'].double(),
                v=W[2 * D:], proj=sd[p + 'attn.layers.proj.weight'].double())


def coherence(mats):
    """mats: list of tensors at TASKS. Returns C, increments' norms, pairwise cos of consecutive increments."""
    inc = [(mats[i + 1] - mats[i]).flatten() for i in range(len(mats) - 1)]
    norms = torch.tensor([float(x.norm()) for x in inc])
    total = (mats[-1] - mats[0]).flatten()
    C = float(total.norm() ** 2 / (norms ** 2).sum())
    cos = torch.zeros(len(inc), len(inc))
    for i in range(len(inc)):
        for j in range(len(inc)):
            cos[i, j] = float(inc[i] @ inc[j] / (inc[i].norm() * inc[j].norm()))
    off = cos[~torch.eye(len(inc), dtype=bool)]
    adj = torch.tensor([float(cos[i, i + 1]) for i in range(len(inc) - 1)])
    return C, norms, float(off.mean()), adj


def main():
    for arm in ('SL', 'R0L'):
        sds = {t: load(arm, t) for t in TASKS}
        P = {t: parts(sds[t]) for t in TASKS}
        print(f'=== {arm}: ‖A‖_F at t10/t100 = {float(P[10]["A"].norm()):.3f} / {float(P[100]["A"].norm()):.3f}, '
              f'‖b‖ = {float(P[10]["b"].norm()):.3f} / {float(P[100]["b"].norm()):.3f}, '
              f'‖Wq‖ {float(P[10]["Wq"].norm()):.2f}/{float(P[100]["Wq"].norm()):.2f}, ‖Wk‖ {float(P[10]["Wk"].norm()):.2f}/{float(P[100]["Wk"].norm()):.2f}, '
              f'γ mean {float(P[10]["gamma"].mean()):.3f}/{float(P[100]["gamma"].mean()):.3f}')
        for key in ('A', 'b', 'Wq', 'Wk', 'v', 'proj', 'fc1_0', 'fc1_1'):
            C, norms, mcos, adj = coherence([P[t][key] for t in TASKS])
            print(f'  {key:6s} C = {C:5.2f}  mean offdiag cos {mcos:+.3f}  adjacent cos {" ".join(f"{a:+.2f}" for a in adj)}  '
                  f'‖Δ‖ per decade {" ".join(f"{n:.3g}" for n in norms)}')
        # 早期（課題 0〜10 の 1 課題ごと）
        early = [load(arm, t) for t in range(0, 11)]
        Pe = [parts(sd) for sd in early]
        for key in ('A', 'Wq', 'fc1_0'):
            C, norms, mcos, adj = coherence([p[key] for p in Pe])
            print(f'  early t0..10 {key:6s} C = {C:5.2f} (max 10)  mean offdiag cos {mcos:+.3f}  ‖Δ‖ per task {" ".join(f"{n:.3g}" for n in norms)}')
        del sds, P, early, Pe

    # Adam の一次モーメント m の q・k 行の cos（課題末 10・30・60・100）
    os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix.json')
    from analysis.vit_long_1024 import common as C_
    model = C_.build_arm_model('SL', SEED, 'cpu', 0.0, 2, 8.0)
    names = [n for n, _ in model.named_parameters()]
    idx = names.index('layers.block_0.layers.attn.layers.qkv.weight')
    idx_fc1 = names.index('layers.block_0.layers.mlp.layers.fc1.weight')
    print(f'param index qkv0 = {idx}, fc1_0 = {idx_fc1} (of {len(names)})')
    for arm in ('SL', 'R0L'):
        ms, cohs, ms_fc1 = {}, {}, {}
        for t in (10, 30, 60, 100):
            st = torch.load(RUNS / arm / f'seed{SEED}' / f'ckpt/adam_t{t:02d}.pt', map_location='cpu', weights_only=False)['optimizer']['state']
            m = st[idx]['exp_avg'].double()[:2 * D]
            v = st[idx]['exp_avg_sq'].double()[:2 * D]
            ms[t] = m.flatten()
            cohs[t] = float((m.abs() / (v.sqrt() + 1e-12)).median())
            ms_fc1[t] = st[idx_fc1]['exp_avg'].double().flatten()
        ts = list(ms)
        pairs = [(a, b) for i, a in enumerate(ts) for b in ts[i + 1:]]
        cs = {f'{a}-{b}': float(ms[a] @ ms[b] / (ms[a].norm() * ms[b].norm())) for a, b in pairs}
        cs_fc1 = {f'{a}-{b}': float(ms_fc1[a] @ ms_fc1[b] / (ms_fc1[a].norm() * ms_fc1[b].norm())) for a, b in pairs}
        print(f'=== {arm} Adam m (q,k rows of block0): cos across task ends {cs}')
        print(f'    median |m|/√v at task ends {cohs}')
        print(f'    fc1_0 m cos across task ends {cs_fc1}')


if __name__ == '__main__':
    main()

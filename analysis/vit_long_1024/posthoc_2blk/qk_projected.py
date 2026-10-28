"""token に射影した q・k の増分の整合性（F5〜F8）。CPU float64。

ξ: 課題 10 末の SL（と R0L それぞれ自分の）状態で、固定 32 枚の block0 の正規化 token（LN1 の affine 前）。
a_t(ξ) = A_t ξ（head ごと）。増分の整合性指数 C_a、半径成分 cos(Δa, a)、ΔW の特異値の参加比、ΔW_q と W_k の整合。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix.json')
import numpy as np
import torch

from analysis.vit_long_1024 import common as C

RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'
SEED = 10252010
D, H, DH = 384, 6, 64
TASKS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load(arm, t):
    return torch.load(RUNS / arm / f'seed{SEED}' / f'ckpt/model_t{t:02d}.pt', map_location='cpu', weights_only=False)


def tokens(model, x):
    """block0 の norm1 の affine 前の正規化 token ξ（B, N, D）。"""
    L = model.layers
    hh = L['patch_embed'](x)
    hh = torch.cat((model.cls_token.expand(hh.shape[0], -1, -1), hh), 1) + model.pos_embed
    mu = hh.mean(-1, keepdim=True)
    var = hh.var(-1, unbiased=False, keepdim=True)
    return (hh - mu) / (var + 1e-5).sqrt()


def operator(sd, block=0):
    p = f'layers.block_{block}.layers.'
    W = sd[p + 'attn.layers.qkv.weight'].double()
    b = sd[p + 'attn.layers.qkv.bias'].double()
    g = sd[p + 'norm1.weight'].double()
    beta = sd[p + 'norm1.bias'].double()
    Wq, Wk = W[:D], W[D:2 * D]
    G = torch.diag(g)
    A = torch.stack([G @ Wk[h * DH:(h + 1) * DH].T @ Wq[h * DH:(h + 1) * DH] @ G / DH ** .5 for h in range(H)])
    bvec = torch.stack([G @ Wk[h * DH:(h + 1) * DH].T @ (Wq[h * DH:(h + 1) * DH] @ beta + b[h * DH:(h + 1) * DH]) / DH ** .5 for h in range(H)])
    return A, bvec, Wq, Wk


def pr_singular(M):
    s = torch.linalg.svdvals(M)
    return float(s.sum() ** 2 / (s ** 2).sum())


def main():
    torch.set_num_threads(4)
    source = C.image_source('real')
    data, _ = C.task_dataset(source, SEED, 11, 10, 250, augment=False)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(2500, generator=g)[:32]
    x = C.eval_tensor(data.images, 'cpu')[idx]
    for arm in ('SL', 'R0L'):
        model = C.build_arm_model(arm, SEED, 'cpu', 0.0, 2, 8.0)
        model.load_state_dict(load(arm, 10))
        model.eval()
        with torch.no_grad():
            xi = tokens(model, x).double().reshape(-1, D)          # (32*65, D) 固定
        ops = {t: operator(load(arm, t)) for t in TASKS}
        # a_t(ξ) = A_t ξ: (H, D, N) → 平らに
        a = {t: torch.einsum('hde,ne->hdn', ops[t][0], xi) for t in TASKS}
        inc = [(a[TASKS[i + 1]] - a[TASKS[i]]).flatten() for i in range(len(TASKS) - 1)]
        total = (a[100] - a[10]).flatten()
        Ca = float(total.norm() ** 2 / sum(float(v.norm() ** 2) for v in inc))
        cos_adj = [float(inc[i] @ inc[i + 1] / (inc[i].norm() * inc[i + 1].norm())) for i in range(len(inc) - 1)]
        radial = [float((a[TASKS[i + 1]] - a[TASKS[i]]).flatten() @ a[TASKS[i]].flatten() / ((a[TASKS[i + 1]] - a[TASKS[i]]).norm() * a[TASKS[i]].norm())) for i in range(len(TASKS) - 1)]
        rms = {t: float(a[t].norm() / np.sqrt(a[t].numel() / D)) for t in (10, 40, 70, 100)}
        # bias 項 b·ξ の寄与（query に依らない key の好み）: token 間の std
        bscore = {t: float(torch.einsum('hd,nd->hn', ops[t][1], xi).std(1).mean()) for t in (10, 100)}
        print(f'=== {arm}: rms‖a(ξ)‖ {rms} | token-std of b·ξ {bscore}')
        print(f'  C_a = {Ca:.2f} (拡散 1・整合 9)  adjacent cos {" ".join(f"{c:+.2f}" for c in cos_adj)}')
        print(f'  radial cos(Δa, a) per decade {" ".join(f"{c:+.2f}" for c in radial)}  median {np.median(radial):+.3f}')
        # ΔW の階数と整合
        prs_q, prs_k, align = [], [], []
        for i in range(len(TASKS) - 1):
            _, _, Wq0, Wk0 = ops[TASKS[i]]
            _, _, Wq1, Wk1 = ops[TASKS[i + 1]]
            dq, dk = Wq1 - Wq0, Wk1 - Wk0
            prs_q.append(pr_singular(dq)); prs_k.append(pr_singular(dk))
            # head ごと: ΔW_q,h (64×384) の行空間と W_k,h の行空間の整合（入力側）、および列（head 次元）側の整合
            al = []
            for h in range(H):
                Dq = dq[h * DH:(h + 1) * DH]; Kh = Wk0[h * DH:(h + 1) * DH]
                al.append(float((Kh @ Dq.T).norm() / (Kh.norm() * Dq.norm())))
            align.append(float(np.mean(al)))
        print(f'  PR(ΔW_q) per decade {" ".join(f"{p:.0f}" for p in prs_q)} | PR(ΔW_k) {" ".join(f"{p:.0f}" for p in prs_k)} | PR(W_q at t10/t100) {pr_singular(ops[10][2]):.0f}/{pr_singular(ops[100][2]):.0f}')
        print(f'  align ‖W_k ΔW_qᵀ‖/(‖W_k‖‖ΔW_q‖) per decade {" ".join(f"{v:.3f}" for v in align)}')
        # token 入力の集中度（参考）: ξ の特異値の参加比
        print(f'  PR(ξ) at t10 state = {pr_singular(xi):.1f} (max {D})')


if __name__ == '__main__':
    main()

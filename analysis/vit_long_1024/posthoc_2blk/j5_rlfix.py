"""追補 D の J5（spec §7.4、装填と切替の押しの向き）を 2 ブロック版の保存状態で計算する（Claude、1026）。

最終ブロック（block_1）の FFN の CLS 位置。課題 t の末の重みで、
  装填: F_t = Σ_m (e_{y_m} − 1/K) ⊗ h_m（課題 t の 2,500 枚・そのラベル）、κ_t = ⟨W^c, F_t⟩/‖F_t‖²、cos²、Q
  押し: S_j = mean_n K_n φ′(z_nj) [(D_n + R_n)·U_j]（probe は課題 t+1 の画像 = この箱では同じ 2,500 枚、e_n = softmax − 1/K）
  s_load（装填成分 vL = κF を通す）・s_full（head 全体を通す）・s_r（J5r、g_n = −h_n）
さらに Adam の状態がある課題末（1・10・30・60・100）では、課題 t+1 の最初の batch で実際に 1 歩進め、
unit ごとの自分の重みを通る変化 Δ_own_j = Δw_j·ȳ + Δb_j と、その batch の勾配の射影 G_j = ∇w_j·ȳ + ∇b_j の符号を S_j と比べる。
CPU float64（押し）、GPU float32 eager（実際の 1 歩）。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json \
        python analysis/vit_long_1024/posthoc_2blk/j5_rlfix.py SL <出力.json>
"""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import train_loader
from analysis.joudaki_vit_battle_0919.model import prepare_noise, reset_head, update_adaptive
from analysis.joudaki_vit_battle_0919.run import set_rng
from analysis.vit_long_1024 import common as C

SEED, K, CHUNK = 10252010, 10, 250
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'
ADAM_TASKS = (1, 10, 30, 60, 100)


def load_model(arm, t, device='cpu', dtype=torch.float64):
    m = C.build_model(SEED, device, 0.0, arm in C.LN_ARMS, 2, 8.0)
    m.load_state_dict(torch.load(RUNS / arm / f'seed{SEED}' / f'ckpt/model_t{t:02d}.pt', map_location=device,
                                 weights_only=False))
    return m.to(dtype).eval()


@torch.no_grad()
def cls_quantities(model, x):
    """CLS 位置の r_n（注意 2 の後の残差）・y_n（FFN 入力）・z_n・t_n・q_n・σ_n・h_n を chunk ごとに集める。"""
    L = model.layers
    out = dict(r=[], y=[], z=[], t=[], q=[], sigma=[], h=[])
    for s in range(0, len(x), CHUNK):
        xb = x[s:s + CHUNK]
        hh = L['patch_embed'](xb)
        hh = torch.cat((model.cls_token.expand(hh.shape[0], -1, -1), hh), 1) + model.pos_embed
        b0 = L['block_0'].layers
        hh = hh + b0['attn'](b0['norm1'](hh))
        hh = hh + b0['mlp'](b0['norm2'](hh))
        b1 = L['block_1'].layers
        r = (hh + b1['attn'](b1['norm1'](hh)))[:, 0]                # CLS
        y = b1['norm2'](r)
        fc1, fc2 = b1['mlp'].layers['fc1'], b1['mlp'].layers['fc2']
        z = fc1(y)
        t = r + fc2(torch.relu(z))
        ln = L['norm']
        if isinstance(ln, torch.nn.LayerNorm):
            mu = t.mean(1, keepdim=True)
            var = ((t - mu) ** 2).mean(1, keepdim=True)
            sigma = torch.sqrt(var + ln.eps)
            q = (t - mu) / sigma
            h = ln.weight * q + ln.bias
        else:                                   # LN なし: h_n = t_n（spec §7.4）
            sigma = torch.ones(t.shape[0], 1, dtype=t.dtype)
            q = t
            h = t
        for k, v in zip(('r', 'y', 'z', 't', 'q', 'sigma', 'h'), (r, y, z, t, q, sigma.squeeze(1), h)):
            out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def ln_backprop(model, g, q, sigma):
    """最終 LN を通す逆伝播。直通 D と分散微分 R（spec §7.4）。LN なしは D = g、R = 0。"""
    if not isinstance(model.layers['norm'], torch.nn.LayerNorm):
        return g, torch.zeros_like(g)
    gamma = model.layers['norm'].weight
    b = gamma * g
    D = (b - b.mean(1, keepdim=True)) / sigma[:, None]
    R = -q * (q * b).mean(1, keepdim=True) / sigma[:, None]
    return D, R


def push(model, g, Q, alive_mask):
    """S_j = mean_n K_n φ′(z_nj) [(D_n + R_n)·U_j]。返すのは S（3072）と、直通だけ・分散微分だけの S。"""
    D, R = ln_backprop(model, g, Q['q'], Q['sigma'])
    U = model.layers['block_1'].layers['mlp'].layers['fc2'].weight          # (384, 3072), 列 j = U_j
    phi = (Q['z'] > 0).to(Q['z'].dtype)
    Kn = (Q['y'] @ Q['ybar'] + 1)[:, None]
    S = (Kn * phi * ((D + R) @ U)).mean(0)
    S_D = (Kn * phi * (D @ U)).mean(0)
    S_R = (Kn * phi * (R @ U)).mean(0)
    frac = lambda v: float((v[alive_mask] > 0).double().mean())
    return S, dict(s=frac(S), s_direct=frac(S_D), s_var=frac(S_R))


def j5_state(arm, t, x, labels_t):
    model = load_model(arm, t)
    rows = torch.tensor(C.loss_classes(arm, SEED, t, 10, 250))
    head = model.layers['out']
    W, b = head.weight[rows], head.bias[rows]
    Q = cls_quantities(model, x)
    Q['ybar'] = Q['y'].mean(0)
    # 装填 F_t（課題 t のラベル）
    E = F.one_hot(labels_t, K).double() - 1.0 / K
    Fm = E.T @ Q['h']                                                  # (K, 384)
    Wc = W - W.mean(0, keepdim=True)
    inner = float((Wc * Fm).sum())
    kappa = inner / float((Fm * Fm).sum())
    cos2 = inner ** 2 / (float((Wc * Wc).sum()) * float((Fm * Fm).sum()))
    # probe = 課題 t+1 の画像（同じ 2,500 枚）、期待の誤差
    p = torch.softmax(Q['h'] @ W.T + b, 1)
    e = p - 1.0 / K
    alive = (Q['z'] > 0).any(0)
    S_load, f_load = push(model, e @ (kappa * Fm), Q, alive)
    S_full, f_full = push(model, e @ W, Q, alive)
    S_r, f_r = push(model, -Q['h'], Q, alive)
    # 補題: ⟨W^c, F_t⟩ ≥ N(log K − L_K) − Σ_c (n_c − N/K) b^c_c
    N = len(labels_t)
    L_K = float(F.cross_entropy(Q['h'] @ W.T + b, labels_t))
    n_c = torch.bincount(labels_t, minlength=K).double()
    bc = b - b.mean()
    bound = N * (math.log(K) - L_K) - float(((n_c - N / K) * bc).sum())
    res = dict(task=t, kappa=kappa, cos2=cos2, Q=kappa * math.sqrt(float((Fm * Fm).sum())), inner=inner, lemma_bound=bound,
               lemma_holds=inner >= bound - 1e-6, loss_on_task=L_K, alive=int(alive.sum()), n_units=int(alive.numel()),
               K_pos=float((Q['y'] @ Q['ybar'] + 1 > 0).double().mean()),
               s_load=f_load['s'], s_load_direct=f_load['s_direct'], s_load_var=f_load['s_var'],
               s_full=f_full['s'], s_full_direct=f_full['s_direct'], s_full_var=f_full['s_var'],
               s_r=f_r['s'], s_r_direct=f_r['s_direct'], s_r_var=f_r['s_var'],
               cls_mean_z=float(Q['z'].mean()), cls_open_frac=float((Q['z'] > 0).double().mean()))
    return res, dict(S_load=S_load, S_full=S_full, S_r=S_r, alive=alive, ybar=Q['ybar'], model=model)


def actual_first_step(arm, t, aux, source):
    """課題 t の末（重み＋Adam）から課題 t+1 の最初の batch で 1 歩。unit ごとの Δ_own と勾配の射影。"""
    device = 'cuda'
    model = C.build_model(SEED, device, 0.0, arm in C.LN_ARMS, 2, 8.0)
    model.load_state_dict(torch.load(RUNS / arm / f'seed{SEED}' / f'ckpt/model_t{t:02d}.pt', map_location=device,
                                     weights_only=False))
    optimizer = C.build_optimizer(model)
    adam = torch.load(RUNS / arm / f'seed{SEED}' / f'ckpt/adam_t{t:02d}.pt', map_location=device, weights_only=False)
    assert adam['task'] == t
    optimizer.load_state_dict(adam['optimizer'])
    set_rng(adam['rng'])
    task = t + 1
    data, _ = C.task_dataset(source, SEED, task, 10, 250, augment=False)
    rows_t = torch.tensor(C.loss_classes(arm, SEED, task, 10, 250), device=device)
    if C.zero_head_at(arm, task, 1):
        reset_head(model)
    model.train()
    x, y = next(iter(train_loader(data, SEED, task, 1, 0)))
    x, y = x.to(device), y.to(device)
    fc1 = model.layers['block_1'].layers['mlp'].layers['fc1']
    w0, b0 = fc1.weight.detach().double().cpu().clone(), fc1.bias.detach().double().cpu().clone()
    optimizer.zero_grad(set_to_none=True)
    prepare_noise(model, len(y))
    loss = F.cross_entropy(model(x).index_select(1, rows_t), y)
    loss.backward()
    gw, gb = fc1.weight.grad.detach().double().cpu(), fc1.bias.grad.detach().double().cpu()
    optimizer.step()
    update_adaptive(model)
    w1, b1 = fc1.weight.detach().double().cpu(), fc1.bias.detach().double().cpu()
    ybar = aux['ybar']
    G = gw @ ybar + gb                        # 勾配の射影。G_j > 0 なら勾配降下は w_j·ȳ + b_j を下げる
    d_own = (w1 - w0) @ ybar + (b1 - b0)      # Adam の 1 歩での自分の重みを通る変化
    alive = aux['alive']
    S = aux['S_load']
    agree = lambda a, b_: float(((a[alive] > 0) == (b_[alive] > 0)).double().mean())
    return dict(first_batch_loss=float(loss), batch=int(len(y)),
                frac_grad_sinks=float((G[alive] > 0).double().mean()),
                frac_adam_sinks=float((d_own[alive] < 0).double().mean()),
                agree_Sload_grad=agree(S, G), agree_Sload_adam=agree(S, -d_own), agree_Sfull_grad=agree(aux['S_full'], G),
                agree_grad_adam=agree(G, -d_own),
                median_d_own=float(d_own[alive].median()), mean_d_own=float(d_own[alive].mean()))


def main(arm, out):
    torch.set_num_threads(16)
    C.set_determinism(16)
    source = C.image_source('real')
    data1, _ = C.task_dataset(source, SEED, 1, 10, 250, augment=False)
    x = C.eval_tensor(data1.images, 'cpu').double()        # 全課題で同じ 2,500 枚
    tasks = sorted(int(p.stem.split('_t')[1]) for p in (RUNS / arm / f'seed{SEED}' / 'ckpt').glob('model_t*.pt'))
    tasks = [t for t in tasks if t >= 1]
    results, t0 = [], time.monotonic()
    for t in tasks:
        labels_t = C.task_labels(2500, SEED, t, 10)
        res, aux = j5_state(arm, t, x, labels_t)
        if t in ADAM_TASKS and t < 100 or (t == 100):
            try:
                res['first_step'] = actual_first_step(arm, t, aux, source)
            except Exception as exc:                        # 課題 101 の Adam は無い場合など
                res['first_step'] = dict(error=repr(exc))
        results.append(res)
        fs = res.get('first_step', {})
        print(f"{arm} t{t:3d} κ {res['kappa']:+.4f} cos² {res['cos2']:.3f} 補題 {'○' if res['lemma_holds'] else '×'} 生 {res['alive']:4d}/{res['n_units']}"
              f" | 沈める側 装填 {res['s_load']:.3f} 全体 {res['s_full']:.3f} J5r {res['s_r']:.3f}"
              + (f" | 実際の 1 歩: 勾配 {fs['frac_grad_sinks']:.3f} Adam {fs['frac_adam_sinks']:.3f} 一致(装填・勾配) {fs['agree_Sload_grad']:.3f} 一致(装填・Adam) {fs['agree_Sload_adam']:.3f}" if 'frac_grad_sinks' in fs else '')
              + f" | {time.monotonic() - t0:.0f}s", flush=True)
        Path(out).write_text(json.dumps(dict(arm=arm, seed=SEED, results=results), indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

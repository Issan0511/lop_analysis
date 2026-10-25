"""S の 2 つの残りを確かめる（Claude、1025）。
(a) 入力 u が変わったことによる FFN1 の ā の変化を、u の token 平均 m1 の差し替えと、平均を引いた残り（token ごとの部分）の差し替えに分ける。
(b) 2 層目の死の反実仮想: 状態 t の重みのまま、2 層目の FFN 入力の共通成分 m2 だけを k 倍して死の割合を数え直す。"""
import json, sys
from pathlib import Path
import torch
from analysis.vit_long_1024 import common as C
torch.set_grad_enabled(False)
arm, seed = 'S', 0
root = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_t100/runs' / arm / f'seed{seed}' / 'ckpt'
src = C.image_source('real')
x = C.eval_tensor(src.val(C.probe_names(src, seed)), 'cpu').double()
def state(t):
    m = C.build_model(seed, 'cpu', dropout=0.0, ln=False, depth=2, mlp_ratio=8.0)
    m.load_state_dict(torch.load(root / f'model_t{t:02d}.pt', map_location='cpu', weights_only=False))
    m = m.double().eval(); L = m.layers
    h = torch.cat((m.cls_token.expand(x.shape[0], -1, -1), L['patch_embed'](x)), 1) + m.pos_embed
    b0, b1 = L['block_0'].layers, L['block_1'].layers
    u1 = h + b0['attn'](b0['norm1'](h))
    h1 = u1 + b0['mlp'](u1)
    u2 = h1 + b1['attn'](b1['norm1'](h1))
    f1, f2 = b0['mlp'].layers, b1['mlp'].layers
    u1 = u1.reshape(-1, u1.shape[-1]); u2 = u2.reshape(-1, u2.shape[-1])
    return dict(u1=u1, m1=u1.mean(0), W1=f1['fc1'].weight, b1=f1['fc1'].bias, W2=f1['fc2'].weight, b2=f1['fc2'].bias,
                u2=u2, m2=u2.mean(0), V1=f2['fc1'].weight, c1=f2['fc1'].bias)
cache = {}
get = lambda t: cache.setdefault(t, state(t))
abar = lambda u, S: torch.relu(u @ S['W1'].T + S['b1']).mean(0)
print('(a) 入力 u による ā の変化（重みは前の状態）を m1 の差し替えと token ごとの部分の差し替えに分ける。値は ‖f̄‖ の変化への寄与')
print('区間 | 入力 u 全部 | m1 だけ後 | token ごとの部分だけ後 | 交差 | ‖m1‖ 前→後 | token ごとの部分の大きさ（rms）前→後')
for ta, tb in [(50, 60), (70, 80), (70, 100), (85, 100), (40, 100)]:
    A, B = get(ta), get(tb)
    fa = A['W2'] @ abar(A['u1'], A) + A['b2']; e = fa / fa.norm()
    base = abar(A['u1'], A)
    full = abar(B['u1'], A)
    only_m = abar(A['u1'] - A['m1'] + B['m1'], A)
    only_c = abar(B['u1'] - B['m1'] + A['m1'], A)
    pr = lambda v: float((A['W2'] @ (v - base)) @ e)
    rms = lambda S: float((S['u1'] - S['m1']).pow(2).sum(1).mean().sqrt())
    pf, pm, pc = pr(full), pr(only_m), pr(only_c)
    print(f"{ta}→{tb} | {pf:+.1f} | {pm:+.1f} | {pc:+.1f} | {pf-pm-pc:+.1f} | {float(A['m1'].norm()):.2f}→{float(B['m1'].norm()):.2f} | {rms(A):.1f}→{rms(B):.1f}")
print()
print('(b) 2 層目の死の割合: m2 を k 倍したとき（重みと token ごとの部分はその状態のまま）')
ks = [0.5, 0.67, 1.0, 1.5, 2.0]
print('課題 | ‖m2‖ | ' + ' | '.join(f'k={k}' for k in ks))
for t in [40, 70, 85, 90, 100]:
    S_ = get(t); res = []
    for k in ks:
        u = S_['u2'] - S_['m2'] + k * S_['m2']
        p = ((u @ S_['V1'].T + S_['c1']) > 0).double().mean(0)
        res.append(float((p < .005).double().mean()))
    print(f"{t} | {float(S_['m2'].norm()):.0f} | " + ' | '.join(f'{r:.2f}' for r in res))
# 2 層目の行と m2 の向き・大きさ・bias の内訳（t70 と t100）
print()
print('(c) 2 層目の unit の平均 z̄ = v·m2 + c の内訳（unit の中央値）と、token ごとの広がり σ')
for t in [70, 85, 100]:
    S_ = get(t)
    vm = S_['V1'] @ S_['m2']; sig = ((S_['u2'] - S_['m2']) @ S_['V1'].T).std(0)
    cos = vm / (S_['V1'].norm(dim=1) * S_['m2'].norm())
    r = (vm + S_['c1']) / sig
    q = lambda v: float(v.median())
    print(f"t{t}: ‖m2‖ {float(S_['m2'].norm()):.0f} | cos(v,m2) 中央値 {q(cos):+.3f} | ‖v‖ 中央値 {q(S_['V1'].norm(dim=1)):.3f} | v·m2 中央値 {q(vm):+.1f} | c 中央値 {q(S_['c1']):+.2f} | σ 中央値 {q(sig):.1f} | r 中央値 {q(r):+.2f}")

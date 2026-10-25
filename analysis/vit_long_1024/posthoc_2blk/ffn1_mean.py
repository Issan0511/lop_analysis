"""1 層目の FFN の出力の token 平均 f̄ = W2 ā + b2 の変化を、重み W2 の変化と活性の平均 ā の変化に分ける（Claude、1025）。
Δf̄ = ΔW2 ā_a + W2_a Δā + ΔW2 Δā（厳密）。各項を f̄_a の向きへ射影して ‖f̄‖ の変化への寄与にする。
ā の変化はさらに、fc1 の重みの変化（入力を固定）と入力 u の変化（重みを固定）に分ける（ReLU なので近似、残りを交差に入れる）。"""
import json, sys
from pathlib import Path
import torch
from analysis.vit_long_1024 import common as C
torch.set_grad_enabled(False)
arm, seed = sys.argv[1], int(sys.argv[2])
root = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_t100/runs' / arm / f'seed{seed}' / 'ckpt'
src = C.image_source('real')
x = C.eval_tensor(src.val(C.probe_names(src, seed)), 'cpu').double()
def state(t):
    m = C.build_model(seed, 'cpu', dropout=0.0, ln=False, depth=2, mlp_ratio=8.0)
    m.load_state_dict(torch.load(root / f'model_t{t:02d}.pt', map_location='cpu', weights_only=False))
    m = m.double().eval(); L = m.layers
    h = torch.cat((m.cls_token.expand(x.shape[0], -1, -1), L['patch_embed'](x)), 1) + m.pos_embed
    blk = L['block_0'].layers
    u = h + blk['attn'](blk['norm1'](h))
    fc1, fc2 = blk['mlp'].layers['fc1'], blk['mlp'].layers['fc2']
    a = torch.relu(fc1(u)).reshape(-1, fc1.out_features)
    head = L['out'].weight
    return dict(u=u.reshape(-1, u.shape[-1]), W1=fc1.weight, b1=fc1.bias, W2=fc2.weight, b2=fc2.bias,
                abar=a.mean(0), alive=(a > 0).double().mean(0), head=head)
pairs = [(int(a), int(b)) for a, b in (p.split('-') for p in sys.argv[3].split(','))]
cache = {}
get = lambda t: cache.setdefault(t, state(t))
print('区間 | ‖f̄‖ 前→後 | W2 の変化 | ā の変化 | 交差 | ā の変化の内訳: fc1 重み / 入力 u / 残り | 生きた 1 層目 unit の平均の割合 前→後')
for ta, tb in pairs:
    A, B = get(ta), get(tb)
    fa = A['W2'] @ A['abar'] + A['b2']; fb = B['W2'] @ B['abar'] + B['b2']
    e = fa / fa.norm()
    dW, da = B['W2'] - A['W2'], B['abar'] - A['abar']
    t1, t2, t3 = float((dW @ A['abar']) @ e), float((A['W2'] @ da) @ e), float((dW @ da) @ e)
    tb2 = float((B['b2'] - A['b2']) @ e)
    # ā の変化の内訳: 重みだけ後のもの（入力は前）、入力だけ後のもの（重みは前）
    a_w = torch.relu(A['u'] @ B['W1'].T + B['b1']).mean(0)
    a_u = torch.relu(B['u'] @ A['W1'].T + A['b1']).mean(0)
    pw, pu = float((A['W2'] @ (a_w - A['abar'])) @ e), float((A['W2'] @ (a_u - A['abar'])) @ e)
    print(f"{ta}→{tb} | {fa.norm():.1f}→{fb.norm():.1f} | {t1:+.1f} | {t2:+.1f} | {t3:+.1f} | {pw:+.1f} / {pu:+.1f} / {t2-pw-pu:+.1f} | {float(A['alive'].mean()):.3f}→{float(B['alive'].mean()):.3f}  (b2 {tb2:+.2f})")

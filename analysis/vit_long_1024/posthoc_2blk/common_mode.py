"""LN なしの 2 ブロック ViT で、FFN に入る残差の token 平均（共通成分）m を部分ごとに分ける（Claude、1025）。
u（block i の FFN 入力）= e + Σ_{j<i}(a_j + f_j) + a_i。各部分の出力を bias（全 token に同じ）とそれ以外に分け、
⟨m_c, m⟩/‖m‖（m への符号つきの寄与。和が ‖m‖）を課題末ごとに出す。float64、CPU、固定 probe。"""
import json, sys
from pathlib import Path
import torch
from analysis.vit_long_1024 import common as C

torch.set_grad_enabled(False)
arm, seed = sys.argv[1], int(sys.argv[2])
root = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_t100/runs' / arm / f'seed{seed}' / 'ckpt'
ln = arm in ('R0L', 'SL')
src = C.image_source('real')
x = C.eval_tensor(src.val(C.probe_names(src, seed)), 'cpu').double()
tasks = sorted(int(p.stem.split('_t')[1]) for p in root.glob('model_t*.pt'))
out = []
for t in tasks:
    model = C.build_model(seed, 'cpu', dropout=0.0, ln=ln, depth=2, mlp_ratio=8.0)
    model.load_state_dict(torch.load(root / f'model_t{t:02d}.pt', map_location='cpu', weights_only=False))
    model = model.double().eval()
    L = model.layers
    pe = L['patch_embed'](x)                                    # (B, 64, D)
    b_pe = L['patch_embed'].layers['proj'].bias                 # (D,)
    B, Np, D = pe.shape
    N = Np + 1
    comp = {}
    comp['埋め込み・重み'] = (pe - b_pe).mean((0, 1)) * Np / N
    comp['埋め込み・bias'] = b_pe * Np / N
    comp['位置'] = model.pos_embed[0].mean(0)
    comp['CLS'] = model.cls_token[0, 0] / N
    h = torch.cat((model.cls_token.expand(B, -1, -1), pe), 1) + model.pos_embed
    rows_t = dict(task=t)
    for i in range(model.depth):
        blk = L[f'block_{i}'].layers
        a = blk['attn'](blk['norm1'](h))
        ba = blk['attn'].layers['proj'].bias
        comp[f'注意{i+1}・重み'] = (a - ba).mean((0, 1)); comp[f'注意{i+1}・bias'] = ba
        h = h + a
        u = blk['norm2'](h)
        m = u.mean((0, 1))
        if not ln:   # 分解の検算は LN なしでだけ成り立つ（u = h）
            s = sum(comp.values())
            rows_t[f'L{i+1}_check'] = float((s - m).norm() / m.norm())
        z = blk['mlp'].layers['fc1'](u)
        p = (z.reshape(-1, z.shape[-1]) > 0).double().mean(0)
        rows_t[f'L{i+1}_norm_m'] = float(m.norm())
        rows_t[f'L{i+1}_dead'] = float((p < .005).double().mean())
        rows_t[f'L{i+1}_contrib'] = {k: float(v @ m / m.norm()) for k, v in comp.items()}
        rows_t[f'L{i+1}_norms'] = {k: float(v.norm()) for k, v in comp.items()}
        f = blk['mlp'](u)
        bf = blk['mlp'].layers['fc2'].bias
        comp[f'FFN{i+1}・重み'] = (f - bf).mean((0, 1)); comp[f'FFN{i+1}・bias'] = bf
        h = h + f
    out.append(rows_t)
    print(t, f"L2 ‖m‖ {rows_t['L2_norm_m']:.1f} 死 {rows_t['L2_dead']:.2f} 検算 {rows_t.get('L2_check', float('nan')):.1e}", flush=True)
Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False, indent=1))

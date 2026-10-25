"""S の FFN1 の入力 u1 の token ごとの広がりと、FFN1 が書く共通成分 ‖f̄‖ を課題末ごとに並べる（Claude、1025）。
広がり = u1 から token 平均を引いた残りの rms。埋め込み e と 1 層目の注意の出力 a に分け、rms² = ‖e_c‖² + ‖a_c‖² + 2⟨e_c,a_c⟩ の各項を出す。
CLS とパッチ 64 個も分ける。"""
import json, sys
from pathlib import Path
import torch
from analysis.vit_long_1024 import common as C
torch.set_grad_enabled(False)
arm, seed = 'S', 0
root = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_t100/runs' / arm / f'seed{seed}' / 'ckpt'
src = C.image_source('real')
x = C.eval_tensor(src.val(C.probe_names(src, seed)), 'cpu').double()
tasks = sorted(int(p.stem.split('_t')[1]) for p in root.glob('model_t*.pt'))
out = []
for t in tasks:
    m = C.build_model(seed, 'cpu', dropout=0.0, ln=False, depth=2, mlp_ratio=8.0)
    m.load_state_dict(torch.load(root / f'model_t{t:02d}.pt', map_location='cpu', weights_only=False))
    m = m.double().eval(); L = m.layers
    e = torch.cat((m.cls_token.expand(x.shape[0], -1, -1), L['patch_embed'](x)), 1) + m.pos_embed   # (B, 65, D)
    b0 = L['block_0'].layers
    a = b0['attn'](b0['norm1'](e))
    u = e + a
    f = b0['mlp'](u)
    D = u.shape[-1]
    mu, me, ma = u.mean((0, 1)), e.mean((0, 1)), a.mean((0, 1))
    uc, ec, ac = u - mu, e - me, a - ma
    ms = lambda v: float(v.pow(2).sum(-1).mean())
    fc1 = b0['mlp'].layers['fc1']
    z = fc1(u)
    out.append(dict(task=t, f_norm=float(f.mean((0, 1)).norm()), rms_u=ms(uc) ** .5,
                    e2=ms(ec), a2=ms(ac), cross=2 * float((ec * ac).sum(-1).mean()),
                    rms_u_cls=ms(uc[:, :1]) ** .5, rms_u_patch=ms(uc[:, 1:]) ** .5,
                    alive1=float((z > 0).double().mean()), b1_med=float(fc1.bias.median()),
                    zbar_over_sd_med=float((z.reshape(-1, z.shape[-1]).mean(0) / z.reshape(-1, z.shape[-1]).std(0)).median())))
    r = out[-1]
    print(f"t{t:3d} ‖f̄‖ {r['f_norm']:6.1f} | 入力の広がり {r['rms_u']:5.1f}（CLS {r['rms_u_cls']:6.1f}・パッチ {r['rms_u_patch']:5.1f}）| 比 {r['f_norm']/r['rms_u']:5.2f} | 内訳 埋め込み² {r['e2']:7.1f} 注意1² {r['a2']:7.1f} 交差 {r['cross']:+7.1f} | 1 層目の r 中央値 {r['zbar_over_sd_med']:+.2f}", flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1))

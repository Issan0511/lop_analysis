"""LN なしの腕（R0・S）の rlfix 保存状態で、最終 FFN の沈み・死・尺度と、その入力の共通成分の書き手を課題末ごとに並べる（Claude、1028）。

u2（最終 FFN の入力）= e + a0 + f0 + a1（埋め込み・注意 0・FFN0・注意 1）。token 平均 m2 への各部分の符号つきの寄与
⟨m_c, m2⟩/‖m2‖（和が ‖m2‖）、m2 のエネルギーの割合、最終 FFN の前活性（全 token と CLS）の平均・幅・r・死、
行 w と m2 の cos の分布、FFN0 の平均出力 ‖W2 ā + b2‖ と開いている割合、head に入る特徴の尺度 lr×Σ|h|。float64、CPU、固定 probe。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json python analysis/vit_long_1024/posthoc_2blk/r0_common_mode.py R0 <出力.json>
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix.json')
import torch

from analysis.vit_long_1024 import common as C

torch.set_grad_enabled(False)
SEED = 10252010
LR = 1e-4
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'


def stats(zz):
    mu, sd = zz.mean(0), zz.std(0)
    open_frac = (zz > 0).double().mean(0)
    return dict(mean=float(mu.mean()), width=float(sd.median()), r=float((mu / sd.clamp_min(1e-12)).median()),
                dead=float((open_frac <= 0.005).double().mean()), dead0=float((open_frac == 0).double().mean()),
                neg=float((zz < 0).double().mean()))


def main(arm, out):
    torch.set_num_threads(4)
    src = C.image_source('real')
    x = C.eval_tensor(src.val(C.probe_names(src, SEED)), 'cpu').double()
    root = RUNS / arm / f'seed{SEED}' / 'ckpt'
    tasks = sorted(int(p.stem.split('_t')[1]) for p in root.glob('model_t*.pt'))
    rows = []
    for t in tasks:
        m = C.build_arm_model(arm, SEED, 'cpu', 0.0, 2, 8.0)
        m.load_state_dict(torch.load(root / f'model_t{t:02d}.pt', map_location='cpu', weights_only=False))
        m = m.double().eval()
        L = m.layers
        e = torch.cat((m.cls_token.expand(x.shape[0], -1, -1), L['patch_embed'](x)), 1) + m.pos_embed
        b0, b1 = L['block_0'].layers, L['block_1'].layers
        a0 = b0['attn'](b0['norm1'](e)); u1 = e + a0
        f0 = b0['mlp'](b0['norm2'](u1)); h1 = u1 + f0
        a1 = b1['attn'](b1['norm1'](h1)); u2 = h1 + a1
        fc1, fc2 = b1['mlp'].layers['fc1'], b1['mlp'].layers['fc2']
        y2 = b1['norm2'](u2)                                  # LN なしでは恒等
        z = fc1(y2)
        f1 = b1['mlp'](y2)
        hcls = L['norm'](u2 + f1)[:, 0]                       # head に入る特徴
        D = u2.shape[-1]
        m2 = y2.mean((0, 1))
        contrib = {k: float(v.mean((0, 1)) @ m2 / m2.norm()) for k, v in (('e', e), ('attn0', a0), ('ffn0', f0), ('attn1', a1))}
        energy = float(m2.norm() ** 2 / y2.reshape(-1, D).pow(2).mean(0).sum())
        W = fc1.weight
        cosw = (W @ m2) / (W.norm(dim=1) * m2.norm())
        wm = W @ m2 + fc1.bias                                 # 共通成分（と bias）が書く前活性
        g0 = b0['mlp'].layers
        A0 = torch.relu(g0['fc1'](b0['norm2'](u1))).reshape(-1, g0['fc1'].out_features)
        abar = A0.mean(0)
        fbar = g0['fc2'].weight @ abar + g0['fc2'].bias
        rows.append(dict(task=t, m2_norm=float(m2.norm()), m2_energy=energy, contrib=contrib,
                         z_all=stats(z.reshape(-1, z.shape[-1])), z_cls=stats(z[:, 0]),
                         cos_w_m2_median=float(cosw.median()), cos_w_m2_neg_frac=float((cosw < 0).double().mean()),
                         wm_mean=float(wm.mean()), wm_median=float(wm.median()), wm_neg_frac=float((wm < 0).double().mean()),
                         ffn0_mean_out_norm=float(fbar.norm()), ffn0_abar_norm=float(abar.norm()), ffn0_alive=float((A0 > 0).double().mean()),
                         ffn0_fc1_norm=float(g0['fc1'].weight.norm()), ffn0_fc2_norm=float(g0['fc2'].weight.norm()),
                         fc1_norm=float(W.norm()), fc2_norm=float(fc2.weight.norm()), head_norm=float(L['out'].weight.norm()),
                         h_cls_norm_med=float(hcls.norm(dim=1).median()), lr_sum_abs_h=float(LR * hcls.abs().sum(1).median()),
                         u2_rms=float(y2.reshape(-1, D).pow(2).mean().sqrt()), u2_spread=float((y2 - m2).reshape(-1, D).pow(2).mean().sqrt())))
        r = rows[-1]
        print(f"{arm} t{t:3d} |m2| {r['m2_norm']:8.2f} E {energy:.2f} 寄与 e {contrib['e']:7.2f} a0 {contrib['attn0']:7.2f} f0 {contrib['ffn0']:8.2f} a1 {contrib['attn1']:8.2f} | "
              f"z 全 mean {r['z_all']['mean']:9.2f} 幅 {r['z_all']['width']:8.2f} r {r['z_all']['r']:6.2f} 死 {r['z_all']['dead']:.2f} | CLS mean {r['z_cls']['mean']:9.2f} 死 {r['z_cls']['dead']:.2f} | "
              f"cos(w,m2) 中央値 {r['cos_w_m2_median']:+.3f} 負 {r['cos_w_m2_neg_frac']:.2f} w·m2 中央値 {r['wm_median']:8.2f} | f̄0 {r['ffn0_mean_out_norm']:8.2f} 開 {r['ffn0_alive']:.2f} fc2_0 {r['ffn0_fc2_norm']:.1f} | "
              f"lrΣ|h| {r['lr_sum_abs_h']:.2f} head {r['head_norm']:.2f} fc1 {r['fc1_norm']:.1f}", flush=True)
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

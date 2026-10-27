"""凍結した体の線形 probe（Claude、1027）: 保存状態の体を動かさず、head が読む特徴 h（2,500 枚、CLS、最終 LN の後）に
新しい線形 head（10 行、0 から）を Adam 1e−4 で 3,000 更新当てはめ、課題 T+1 の乱数ラベルへの到達（暗記率の最大・最終）を SL と R0L で比べる。
batch 128・reshuffle は run.py と同じ流れでなくてよい（特徴の分けやすさだけを見る）。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json python analysis/vit_long_1024/posthoc_2blk/linear_probe.py
"""
import torch
from torch.nn import functional as F
from pathlib import Path
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk.j5_rlfix import cls_quantities
torch.set_num_threads(12)
R = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'
SEED = 10252010
src = C.image_source('real')
data, _ = C.task_dataset(src, SEED, 1, 10, 250, augment=False)
x = C.eval_tensor(data.images, 'cpu').double()
print('腕 課題末 | 有効次元 | 線形 probe（3,000 更新）: 暗記率の最大 / 最終、損失の最終 | 1,000 更新時の暗記率')
for arm in ['SL', 'R0L']:
    for t in [10, 60, 100]:
        with torch.no_grad():
            m = C.build_arm_model(arm, SEED, 'cpu', 0.0, 2, 8.0)
            m.load_state_dict(torch.load(R / arm / f'seed{SEED}/ckpt/model_t{t:02d}.pt', map_location='cpu', weights_only=False))
            h = cls_quantities(m.double().eval(), x)['h'].float()
        y = C.task_labels(2500, SEED, t + 1, 10)
        Hc = h - h.mean(0); e = torch.linalg.svdvals(Hc.double()) ** 2; pr = float(e.sum() ** 2 / (e ** 2).sum())
        g = torch.Generator().manual_seed(0)
        W = torch.zeros(10, h.shape[1], requires_grad=True); b = torch.zeros(10, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-4, betas=(.9, .999), eps=1e-8)
        best, acc1000, acc = 0., None, 0.
        for step in range(1, 3001):
            idx = torch.randint(0, 2500, (128,), generator=g)
            loss = F.cross_entropy(h[idx] @ W.T + b, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 0:
                with torch.no_grad():
                    logits = h @ W.T + b
                    acc = float((logits.argmax(1) == y).float().mean()); best = max(best, acc)
                    full = float(F.cross_entropy(logits, y))
                if step == 1000: acc1000 = acc
        print(f'{arm} t{t:3d} | {pr:5.1f} | {best:.3f} / {acc:.3f}, {full:.3f} | {acc1000:.3f}')

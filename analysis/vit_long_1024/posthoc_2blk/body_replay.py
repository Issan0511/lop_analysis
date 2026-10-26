"""課題の胴の回し直し（Claude、1026）: 保存した課題 T の末（重み＋Adam）から課題 T+1 を run.py と同じ手順で
5,000 更新回し、更新ごとに fc1（2 ブロック）の勾配の向きの揃いと w への射影を測る。

更新ごとに記録するもの（fc1 の重み W（3072 × 384）、unit j の行 w_j）:
  coh     Adam の比 r = m̂/(√v̂ + ε) の二乗平均平方根（1 なら勾配の向きが揃い、雑音だけなら小さい）。1 歩 ≈ lr·r
  step    mean_j ‖Δw_j‖² ÷ (lr² d)（= unit ごとの r² の平均。拡散の目安は Adam の歩幅が一定なら ≈ coh²）
  rad_g   勾配の w 方向の成分: Σ_j g_j·ŵ_j ÷ Σ_j ‖g_j‖（符号つき。正なら勾配降下は ‖w‖ を縮める向き）
  cos_g   unit ごとの cos(g_j, w_j) の平均
  rad_d   1 歩の相対的なノルムの変化: mean_j Δw_j·w_j ÷ ‖w_j‖²
  frac_d  ‖Δw‖² のうち w 方向の分: Σ_j (Δw_j·ŵ_j)² ÷ Σ_j ‖Δw_j‖²
  loss    その batch の損失
  gnorm   mean_j ‖g_j‖

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json \
        python analysis/vit_long_1024/posthoc_2blk/body_replay.py SL 10 <出力.npz>
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import train_loader
from analysis.joudaki_vit_battle_0919.model import prepare_noise, reset_head, train_forward, update_adaptive
from analysis.joudaki_vit_battle_0919.run import set_rng
from analysis.vit_long_1024 import common as C

SEED, STEPS = 10252010, 5000
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'
KEYS = ('coh', 'step', 'rad_g', 'cos_g', 'rad_d', 'frac_d', 'gnorm')


def main(arm, t_end, out):
    C.set_determinism(4)
    device = 'cuda'
    d = RUNS / arm / f'seed{SEED}'
    model = C.build_arm_model(arm, SEED, device, 0.0, 2, 8.0)
    model.load_state_dict(torch.load(d / f'ckpt/model_t{t_end:02d}.pt', map_location=device, weights_only=False))
    optimizer = C.build_optimizer(model)
    adam = torch.load(d / f'ckpt/adam_t{t_end:02d}.pt', map_location=device, weights_only=False)
    assert adam['task'] == t_end
    optimizer.load_state_dict(adam['optimizer'])
    set_rng(adam['rng'])
    forward = train_forward(model, 'compile')
    source = C.image_source('real')
    task = t_end + 1
    data, _ = C.task_dataset(source, SEED, task, 10, 250, augment=False)
    rows_t = torch.tensor(C.loss_classes(arm, SEED, task, 10, 250), device=device)
    if C.zero_head_at(arm, task, 1):
        reset_head(model)
    fc1s = [model.layers[f'block_{i}'].layers['mlp'].layers['fc1'].weight for i in range(2)]
    beta1, beta2, eps, lr = .9, .999, 1e-8, C.LR
    dim = fc1s[0].shape[1]
    rec = {k: np.zeros((STEPS, 2)) for k in KEYS}
    loss_rec = np.zeros(STEPS)
    model.train()
    t0 = time.monotonic()
    for step, (x, y) in enumerate(train_loader(data, SEED, task, STEPS, 2), 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prepare_noise(model, len(y))
        loss = F.cross_entropy(forward(x).index_select(1, rows_t), y)
        loss.backward()
        with torch.no_grad():
            w_old = [w.detach().clone() for w in fc1s]
            g_old = [w.grad.detach().clone() for w in fc1s]
        optimizer.step()
        update_adaptive(model)
        with torch.no_grad():
            for i, w in enumerate(fc1s):
                st = optimizer.state[w]
                k = float(st['step'])
                m_hat = st['exp_avg'] / (1 - beta1 ** k)
                v_hat = st['exp_avg_sq'] / (1 - beta2 ** k)
                r = m_hat / (v_hat.sqrt() + eps)
                dw = w.detach() - w_old[i]
                wn = w_old[i].norm(dim=1)
                gn = g_old[i].norm(dim=1)
                g_dot = (g_old[i] * w_old[i]).sum(1)
                d_dot = (dw * w_old[i]).sum(1)
                rec['coh'][step - 1, i] = float(r.pow(2).mean().sqrt())
                rec['step'][step - 1, i] = float(dw.pow(2).sum(1).mean() / (lr ** 2 * dim))
                rec['rad_g'][step - 1, i] = float((g_dot / wn).sum() / gn.sum())
                rec['cos_g'][step - 1, i] = float((g_dot / (gn * wn + 1e-30)).mean())
                rec['rad_d'][step - 1, i] = float((d_dot / wn.pow(2)).mean())
                rec['frac_d'][step - 1, i] = float((d_dot / wn).pow(2).sum() / dw.pow(2).sum())
                rec['gnorm'][step - 1, i] = float(gn.mean())
        loss_rec[step - 1] = float(loss)
    torch.cuda.synchronize()
    original = {int(r['task']): r for r in csv.DictReader(open(d / 'per_task.csv'))}.get(task)
    np.savez_compressed(out, loss=loss_rec, **rec, arm=arm, t_end=t_end, task=task, lr=lr, dim=dim,
                        original_T90=int(original['T90']) if original else -1, seconds=time.monotonic() - t0)
    W = 100
    print(f'{arm} T{t_end}→{task} T90 元 {original["T90"] if original else "—"}  （{time.monotonic() - t0:.0f} s）')
    print('  窓（更新） | 損失 | 2 層目: coh | step/lr²d | rad_g | rad_d | frac_d  | 1 層目: coh | step | rad_g | rad_d')
    for a in (0, 100, 500, 1000, 2000, 3000, 4000, 4900):
        s = slice(a, a + W)
        f = lambda k, i: rec[k][s, i].mean()
        print(f"  {a + 1:5d}〜{a + W:5d} | {loss_rec[s].mean():.2e} | {f('coh', 1):.3f} | {f('step', 1):.3f} | {f('rad_g', 1):+.4f} | {f('rad_d', 1):+.2e} | {f('frac_d', 1):.4f}"
              f"  | {f('coh', 0):.3f} | {f('step', 0):.3f} | {f('rad_g', 0):+.4f} | {f('rad_d', 0):+.2e}")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])

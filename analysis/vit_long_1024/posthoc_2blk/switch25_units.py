"""切替の直後 25 更新で、最終ブロックの FFN の unit ごとに CLS 位置の平均前活性がどう動いたかを測る（Claude、1026）。
課題 t の末（重み＋Adam）から課題 t+1 を 25 更新回し、unit j ごとに
  Δz̄_j = 後の mean_n z_nj − 前の mean_n z_nj（上流の変化を含む全体）
  Δ_own_j = Δw_j·ȳ + Δb_j（自分の重みを通る分、ȳ は前の probe の FFN 入力の平均）
を出し、J5 の S_j（j5_rlfix.py の出力）の符号と比べる。probe は課題 t+1 の画像（同じ 2,500 枚）の各クラス先頭 25 枚。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json \
        python analysis/vit_long_1024/posthoc_2blk/switch25_units.py SL "1,10,30,60" <出力.json>
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import train_loader
from analysis.joudaki_vit_battle_0919.model import prepare_noise, reset_head, train_forward, update_adaptive
from analysis.joudaki_vit_battle_0919.run import set_rng
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk.j5_rlfix import cls_quantities, j5_state

SEED, STEPS = 10252010, 25
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'


def run_one(arm, t, x_probe, j5):
    device = 'cuda'
    d = RUNS / arm / f'seed{SEED}'
    model = C.build_model(SEED, device, 0.0, arm in C.LN_ARMS, 2, 8.0)
    model.load_state_dict(torch.load(d / f'ckpt/model_t{t:02d}.pt', map_location=device, weights_only=False))
    optimizer = C.build_optimizer(model)
    adam = torch.load(d / f'ckpt/adam_t{t:02d}.pt', map_location=device, weights_only=False)
    optimizer.load_state_dict(adam['optimizer'])
    set_rng(adam['rng'])
    task = t + 1
    source = C.image_source('real')
    data, _ = C.task_dataset(source, SEED, task, 10, 250, augment=False)
    rows_t = torch.tensor(C.loss_classes(arm, SEED, task, 10, 250), device=device)
    if C.zero_head_at(arm, task, 1):
        reset_head(model)
    fc1 = model.layers['block_1'].layers['mlp'].layers['fc1']

    def snapshot():
        model.eval()
        m64 = C.build_model(SEED, 'cpu', 0.0, arm in C.LN_ARMS, 2, 8.0)
        m64.load_state_dict({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in model.state_dict().items()})
        Q = cls_quantities(m64.double().eval(), x_probe)
        model.train()
        return Q['z'].mean(0), Q['y'].mean(0), (Q['z'] > 0).any(0), fc1.weight.detach().double().cpu().clone(), fc1.bias.detach().double().cpu().clone()

    z0, ybar0, alive0, w0, b0 = snapshot()
    forward = train_forward(model, 'compile')
    model.train()
    for step, (x, y) in enumerate(train_loader(data, SEED, task, STEPS, 2), 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prepare_noise(model, len(y))
        loss = F.cross_entropy(forward(x).index_select(1, rows_t), y)
        loss.backward()
        optimizer.step()
        update_adaptive(model)
    z1, ybar1, alive1, w1, b1 = snapshot()
    dz = z1 - z0
    d_own = (w1 - w0) @ ybar0 + (b1 - b0)
    d_input = (w0 @ (ybar1 - ybar0))                      # 重みを固定して入力の平均が動いた分
    S = torch.tensor(j5['S_load']) if 'S_load' in j5 else None
    a = alive0
    res = dict(task=t, alive=int(a.sum()), frac_total_sinks=float((dz[a] < 0).double().mean()),
               frac_own_sinks=float((d_own[a] < 0).double().mean()), frac_input_sinks=float((d_input[a] < 0).double().mean()),
               median_dz=float(dz[a].median()), median_d_own=float(d_own[a].median()), median_d_input=float(d_input[a].median()),
               rms_dz=float(dz[a].pow(2).mean().sqrt()), rms_d_own=float(d_own[a].pow(2).mean().sqrt()),
               rms_d_input=float(d_input[a].pow(2).mean().sqrt()),
               alive_after=int(alive1.sum()), ybar_shift_norm=float((ybar1 - ybar0).norm()), ybar_norm=float(ybar0.norm()),
               corr_total_own=float(np.corrcoef(dz[a].numpy(), d_own[a].numpy())[0, 1]),
               corr_total_input=float(np.corrcoef(dz[a].numpy(), d_input[a].numpy())[0, 1]))
    if S is not None:
        res.update(agree_S_own=float(((S[a] > 0) == (d_own[a] < 0)).double().mean()),
                   agree_S_total=float(((S[a] > 0) == (dz[a] < 0)).double().mean()),
                   frac_S_sinks=float((S[a] > 0).double().mean()))
    return res


def main(arm, tasks, out):
    C.set_determinism(8)
    source = C.image_source('real')
    data, _ = C.task_dataset(source, SEED, 1, 10, 250, augment=False)
    idx = [c * 250 + i for c in range(10) for i in range(25)]
    x_probe = C.eval_tensor(data.images[idx], 'cpu').double()
    x_full = C.eval_tensor(data.images, 'cpu').double()
    results = []
    for t in tasks:
        _, aux = j5_state(arm, t, x_full, C.task_labels(2500, SEED, t, 10))     # S_j（J5 の装填成分の押し）
        res = run_one(arm, t, x_probe, dict(S_load=aux['S_load'].tolist()))
        results.append(res)
        print(f"{arm} t{t:3d}→{t + 1}: 25 更新で沈んだ unit の割合 全体 {res['frac_total_sinks']:.3f} 自分の重み {res['frac_own_sinks']:.3f} 入力 {res['frac_input_sinks']:.3f}"
              f" | 中央値 Δz̄ {res['median_dz']:+.3f}（自分 {res['median_d_own']:+.4f}・入力 {res['median_d_input']:+.3f}）"
              f" | rms 全体 {res['rms_dz']:.3f} 自分 {res['rms_d_own']:.4f} 入力 {res['rms_d_input']:.3f} | ȳ の移動 {res['ybar_shift_norm']:.2f}/{res['ybar_norm']:.2f}"
              + (f" | S_load との一致: 自分 {res['agree_S_own']:.3f} 全体 {res['agree_S_total']:.3f}" if 'agree_S_own' in res else ''), flush=True)
        Path(out).write_text(json.dumps(dict(arm=arm, results=results), indent=1))


if __name__ == '__main__':
    main(sys.argv[1], [int(v) for v in sys.argv[2].split(",")], sys.argv[3])

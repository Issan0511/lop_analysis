"""「沈んでも幅を伸ばして応急処置する」説の unit ごとの帳簿（Claude、1027）。

replay: 保存した課題 T の末（重み＋Adam）から課題 T+1 を run.py と同じ手順で 5,000 更新回し直し、
        SNAP の更新で 2 つの FFN の unit ごとの前活性の平均 μ_j・幅 σ_j（固定 probe の全トークン）、
        fc1 の行ノルム ‖w_j‖・bias b_j・開いている割合 p_j を記録する。
tasks:  保存した課題末の重み（model_t00〜t10、t20〜t100）から同じ量を記録する（訓練しない）。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json \
        python analysis/vit_long_1024/posthoc_2blk/width_aid.py replay SL 100 <出力.npz>
    ... width_aid.py tasks SL <出力.npz>
"""
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

SEED, STEPS = 10252010, 5000
SNAP = (0, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 3500, 5000)
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'
SAVED = tuple(range(0, 11)) + tuple(range(20, 101, 10))


@torch.no_grad()
def units(model, probe):
    """FFN ごとの (μ, σ, ‖w‖, b, p)。形は (層, 量, unit)。"""
    was = model.training
    model.eval()
    _, blocks = C.trace(model, probe)
    out = []
    for i, (_, z) in enumerate(blocks):
        Z = z.reshape(-1, z.shape[-1]).double()
        layer = C.fc1(model, i)
        out.append(torch.stack([Z.mean(0), Z.std(0, unbiased=True), layer.weight.double().norm(dim=1),
                                layer.bias.double(), (Z > 0).double().mean(0)]).cpu().numpy())
    model.train(was)
    return np.stack(out)


def replay(arm, t_end, out):
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
    probe = C.eval_tensor(source.val(C.probe_names(source, SEED)), device)
    task = t_end + 1
    data, _ = C.task_dataset(source, SEED, task, 10, 250, augment=False)
    rows_t = torch.tensor(C.loss_classes(arm, SEED, task, 10, 250), device=device)
    if C.zero_head_at(arm, task, 1):
        reset_head(model)
    snaps, losses = [units(model, probe)], []
    model.train()
    for step, (x, y) in enumerate(train_loader(data, SEED, task, STEPS, 2), 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prepare_noise(model, len(y))
        loss = F.cross_entropy(forward(x).index_select(1, rows_t), y)
        loss.backward()
        optimizer.step()
        update_adaptive(model)
        losses.append(float(loss))
        if step in SNAP:
            snaps.append(units(model, probe))
    np.savez(out, snap=np.array(SNAP), units=np.stack(snaps), loss=np.array(losses), arm=arm, t_end=t_end)
    print(out)


def tasks(arm, out):
    device = 'cuda'
    d = RUNS / arm / f'seed{SEED}'
    model = C.build_arm_model(arm, SEED, device, 0.0, 2, 8.0)
    source = C.image_source('real')
    probe = C.eval_tensor(source.val(C.probe_names(source, SEED)), device)
    snaps = []
    for t in SAVED:
        model.load_state_dict(torch.load(d / f'ckpt/model_t{t:02d}.pt', map_location=device, weights_only=False))
        snaps.append(units(model, probe))
    np.savez(out, task=np.array(SAVED), units=np.stack(snaps), arm=arm)
    print(out)


if __name__ == '__main__':
    if sys.argv[1] == 'replay':
        replay(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        tasks(sys.argv[2], sys.argv[3])

"""追補 D の切替の押しを測る（Claude、1026）: 保存した課題 T の末（重みと Adam の状態）から課題 T+1 を
run.py と同じ手順で 5,000 更新回し直し、途中の更新で層ごとの前活性の平均・幅・相対位置を固定 probe で取る。
元の走は借りた機械なので丸めは違う（課題の始めの暗記損失で元の記録と照合する）。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json \
        python analysis/vit_long_1024/posthoc_2blk/switch_replay.py SL 30 <出力.json>
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import train_loader
from analysis.joudaki_vit_battle_0919.model import prepare_noise, reset_head, train_forward, update_adaptive
from analysis.joudaki_vit_battle_0919.run import set_rng
from analysis.vit_long_1024 import common as C

SEED, STEPS = 10252010, 5000
SNAP = (0, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 3500, 5000)
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'


def main(arm, t_end, out):
    C.set_determinism(4)
    device = 'cuda'
    d = RUNS / arm / f'seed{SEED}'
    model = C.build_model(SEED, device, 0.0, arm in C.LN_ARMS, 2, 8.0)
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
    x_eval, y_eval = C.eval_tensor(data.images, device), data.labels.to(device)

    @torch.no_grad()
    def memo():
        model.eval()
        correct, loss = 0, 0.
        for s in range(0, len(y_eval), C.BATCH):
            logits = model(x_eval[s:s + C.BATCH]).index_select(1, rows_t)
            correct += int((logits.argmax(1) == y_eval[s:s + C.BATCH]).sum())
            loss += float(F.cross_entropy(logits, y_eval[s:s + C.BATCH], reduction='sum'))
        model.train()
        return correct / len(y_eval), loss / len(y_eval)

    snaps = []

    def snap(step):
        acc, loss = memo()
        layers = C.layer_summary(model, probe)
        snaps.append(dict(step=step, mem_acc=acc, mem_loss=loss,
                          layers=[{k: r[k] for k in ('layer', 'mu_med', 'sigma_med', 'r_med', 'up_norm_med', 'bias_med',
                                                     'neg_mass', 'dead005', 'cls_r_med', 'in_mean_frac')} for r in layers]))

    model.train()
    snap(0)
    loader = train_loader(data, SEED, task, STEPS, 2)
    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prepare_noise(model, len(y))
        loss = F.cross_entropy(forward(x).index_select(1, rows_t), y)
        loss.backward()
        optimizer.step()
        update_adaptive(model)
        if step in SNAP:
            snap(step)
    original = {int(r['task']): r for r in csv.DictReader(open(d / 'per_task.csv'))}.get(task)
    result = dict(arm=arm, t_end=t_end, task=task, snapshots=snaps,
                  original_mem_loss_start=float(original['mem_loss_start']) if original else None,
                  original_T90=int(original['T90']) if original else None, finished=time.time())
    Path(out).write_text(json.dumps(result, indent=1))
    s0, s100, s_end = snaps[0], [s for s in snaps if s['step'] == 100][0], snaps[-1]
    print(arm, 'T', t_end, '→', task, 'mem_loss start', round(s0['mem_loss'], 4), 'original', result['original_mem_loss_start'])
    for L in (0, 1):
        a, b, c = s0['layers'][L], s100['layers'][L], s_end['layers'][L]
        print(f"  L{L + 1} 平均 {a['mu_med']:+.3f} → {b['mu_med']:+.3f}（100 更新）→ {c['mu_med']:+.3f}（5,000）"
              f" | 幅 {a['sigma_med']:.3f} → {b['sigma_med']:.3f} → {c['sigma_med']:.3f}"
              f" | r {a['r_med']:+.3f} → {b['r_med']:+.3f} → {c['r_med']:+.3f}")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])

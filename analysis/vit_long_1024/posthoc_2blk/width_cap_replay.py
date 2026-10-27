"""幅の伸びを止めて課題を続けて回し直す（Claude、1027）: 「沈んでも幅を伸ばして応急処置するが、副作用で可塑性を失う」説の介入。
保存した課題 T の末（重み＋Adam）から課題 T+1 … T_last を run.py と同じ手順（各課題 5,000 更新）で回す。

cap:  毎更新の後、2 つの FFN の fc1 の行ノルムを課題 T の末の値で上から抑える（w_j ← w_j · min(1, n_j/‖w_j‖)）。
      bias と Adam の状態には触れない。関数は変わる（行が伸びた分だけ縮める）。
none: 抑えない（同じ機械での対照）。

説が正しければ、cap では (i) 沈んだ unit が戻れず死が増え、(ii) 可塑性は保たれる（T90 が延びない）。

課題ごとに T50/T75/T90（2,500 枚で 25 更新ごと、0.9 に届いたら以後は課題末だけ評価）、始めの暗記損失、
fc1 の mean ‖w‖²、課題末の層ごとの要約（固定 probe）を書く。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json \
        python analysis/vit_long_1024/posthoc_2blk/width_cap_replay.py SL 30 60 cap <出力.json>
"""
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
from analysis.vit_long_1024.run import first_reach

SEED, STEPS, N = 10252010, 5000, 2500
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'


@torch.no_grad()
def memorized(model, x_eval, y_eval, rows_t):
    model.eval()
    correct, loss = 0, 0.
    for s in range(0, N, C.BATCH):
        logits = model(x_eval[s:s + C.BATCH]).index_select(1, rows_t)
        correct += int((logits.argmax(1) == y_eval[s:s + C.BATCH]).sum())
        loss += float(F.cross_entropy(logits, y_eval[s:s + C.BATCH], reduction='sum'))
    model.train()
    return correct, loss / N


def main(arm, t_start, t_last, mode, out):
    C.set_determinism(4)
    device = 'cuda'
    d = RUNS / arm / f'seed{SEED}'
    model = C.build_arm_model(arm, SEED, device, 0.0, 2, 8.0)
    model.load_state_dict(torch.load(d / f'ckpt/model_t{t_start:02d}.pt', map_location=device, weights_only=False))
    optimizer = C.build_optimizer(model)
    adam = torch.load(d / f'ckpt/adam_t{t_start:02d}.pt', map_location=device, weights_only=False)
    assert adam['task'] == t_start
    optimizer.load_state_dict(adam['optimizer'])
    set_rng(adam['rng'])
    fc1s = [C.fc1(model, i).weight for i in range(model.depth)]
    caps = [w.detach().norm(dim=1, keepdim=True).clone() for w in fc1s]
    forward = train_forward(model, 'compile')
    source = C.image_source('real')
    probe = C.eval_tensor(source.val(C.probe_names(source, SEED)), device)
    rows = []
    for task in range(t_start + 1, t_last + 1):
        t0 = time.monotonic()
        data, _ = C.task_dataset(source, SEED, task, 10, 250, augment=False)
        rows_t = torch.tensor(C.loss_classes(arm, SEED, task, 10, 250), device=device)
        if C.zero_head_at(arm, task, 1):
            reset_head(model)
        x_eval, y_eval = C.eval_tensor(data.images, device), data.labels.to(device)
        c0, l0 = memorized(model, x_eval, y_eval, rows_t)
        correct, reached = [c0], c0 * 100 >= 90 * N
        model.train()
        for step, (x, y) in enumerate(train_loader(data, SEED, task, STEPS, 2), 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prepare_noise(model, len(y))
            loss = F.cross_entropy(forward(x).index_select(1, rows_t), y)
            loss.backward()
            optimizer.step()
            update_adaptive(model)
            if mode == 'cap':
                with torch.no_grad():
                    for w, n in zip(fc1s, caps):
                        w.mul_(torch.clamp(n / w.norm(dim=1, keepdim=True), max=1.0))
            if step % C.EVAL_EVERY == 0 and not reached:
                c, _ = memorized(model, x_eval, y_eval, rows_t)
                correct.append(c)
                reached = c * 100 >= 90 * N
        c_end, l_end = memorized(model, x_eval, y_eval, rows_t)
        T = {f'T{p}': first_reach(correct, N, p) if any(c * 100 >= p * N for c in correct) else STEPS + C.EVAL_EVERY
             for p in (50, 75, 90)}
        layers = C.layer_summary(model, probe)
        rows.append(dict(task=task, **T, mem_loss_start=l0, mem_acc_end=c_end / N, mem_loss_end=l_end,
                         w2=[float(w.detach().double().pow(2).sum(1).mean()) for w in fc1s],
                         layers=[{k: r[k] for k in ('layer', 'mu_med', 'sigma_med', 'r_med', 'dead005', 'dead0',
                                                    'neg_mass', 'up_norm_med', 'in_mean_frac')} for r in layers],
                         seconds=round(time.monotonic() - t0, 1)))
        Path(out).write_text(json.dumps(dict(arm=arm, t_start=t_start, mode=mode, rows=rows), indent=1))
        r = rows[-1]
        print(f"{arm} {mode} 課題 {task}: T90 {r['T90']} 始めの損失 {l0:.2f} 末の暗記率 {r['mem_acc_end']:.3f} "
              f"mean‖w‖² {r['w2'][0]:.3f}/{r['w2'][1]:.3f} 死 {layers[0]['dead005']:.3f}/{layers[1]['dead005']:.3f} "
              f"r {layers[0]['r_med']:+.2f}/{layers[1]['r_med']:+.2f} ({r['seconds']} s)", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])

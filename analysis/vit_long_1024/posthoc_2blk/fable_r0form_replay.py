"""R0・S の 1 課題の回し直しで、patch 側の尺度の課題内の時系列と力の符号を記録する（Fable エージェント A、依頼 4 の A 形成則、1028）。

保存状態（model_tK + adam_tK）から次の課題 K+1 を eager で ≤ 5,000 更新回す。batch 列・ラベル・損失の 10 クラスは本走と同じ。
R0 は head を 0 に戻す（head の Adam のモーメントは本走どおり残す）。
- 25 更新ごと（と切替直後の 1,2,3,5,10,15,20 更新。R0 t10・S t10 の最初の走には無い）: 固定 probe（val 64 枚）で各段の patch rms・CLS rms（rms = token ベクトルのノルムの二乗平均の平方根）、
  m2（最終 FFN の入力の token 平均）、FFN0・注意 0・注意 1 の平均 patch 出力のノルム、最終 FFN の死、FFN0 の開口、head のノルム、
  訓練 2,500 枚の暗記率と損失。
- 50 更新ごと: FFN0 の patch 出力に仮想の倍率 α を入れた ∂L/∂α（訓練 2,500 枚、画像ごと、forward-mode の JVP）と、
  head 特徴の倍率 α_h の ∂L/∂α_h = (p−y)ᵀWh。当たっている画像・外れている画像に分けて平均。平均ベクトル（m2 など）を npz に残す。
- 最初の 200 更新: 毎更新、体の 1 歩のノルム・体と head の勾配ノルム・head のノルム・batch 損失。
--adam keep|zero_m|zero_mv: 体の Adam のモーメントの扱い（A-3）。zero_m は m だけ 0、zero_mv は m・v・step を 0（体だけ新品の Adam）。
--freeze_body_until N: 最初の N 更新は体の lr を 0 にして head だけ動かす（切替相の体の過渡を除く反実仮想。A-2・A-7）。

    PYTHONPATH=$PWD nice -n 10 python analysis/vit_long_1024/posthoc_2blk/fable_r0form_replay.py R0 30 --adam keep
"""
import argparse
import json
import resource
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.joudaki_vit_battle_0919.model import reset_head
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import C, ROOT, SEED, ckpt, images, load_model, sha

OUT = ROOT / 'posthoc_r0_form'
LR = 1e-4
NODES = ['e', 'attn0', 'u1', 'ffn0', 'h1', 'attn1', 'u2', 'ffn1', 'h2']
FINE = {1, 2, 3, 5, 10, 15, 20}          # 切替直後の過渡を細かく記録する更新（25 更新ごとに加えて）


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')


def progress(message):
    with (OUT / 'PROGRESS.md').open('a') as f:
        f.write(f'- {now()} {message}\n')


def resources():
    r = dict(rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    if torch.cuda.is_initialized():
        r.update(cuda_alloc_mib=torch.cuda.max_memory_allocated() / 2**20,
                 cuda_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    if r['rss_mib'] > 2560:
        raise RuntimeError(f'RSS budget exceeded: {r}')
    return r


def run_nodes(model, x, alpha=None):
    """残差流の各段を返す。alpha は FFN0 の patch token の出力だけに掛ける仮想の倍率（CLS は掛けない）。"""
    L = model.layers
    e = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), L['patch_embed'](x)), 1) + model.pos_embed
    b0, b1 = L['block_0'].layers, L['block_1'].layers
    a0 = b0['attn'](b0['norm1'](e))
    u1 = e + a0
    m0 = b0['mlp'].layers
    z0 = m0['fc1'](b0['norm2'](u1))
    f0 = m0['fc2'](m0['act'](z0))
    if alpha is not None:
        f0 = torch.cat((f0[:, :1], f0[:, 1:] * alpha), 1)
    h1 = u1 + f0
    a1 = b1['attn'](b1['norm1'](h1))
    u2 = h1 + a1
    m1 = b1['mlp'].layers
    z1 = m1['fc1'](b1['norm2'](u2))
    f1 = m1['fc2'](m1['act'](z1))
    h2 = u2 + f1
    h = L['norm'](h2)[:, 0]
    return dict(e=e, attn0=a0, u1=u1, z0=z0, ffn0=f0, h1=h1, attn1=a1, u2=u2, z1=z1, ffn1=f1, h2=h2), h


def rms_norm(z):
    return float(z.double().square().sum(-1).mean().sqrt())


@torch.no_grad()
def geometry(model, xp, head):
    nd, h = run_nodes(model, xp)
    g = {}
    for k in NODES:
        g[f'{k}_patch_rms'] = rms_norm(nd[k][:, 1:])
        g[f'{k}_cls_rms'] = rms_norm(nd[k][:, 0])
    y2 = nd['u2'].double()                                   # LN なしでは norm2 は恒等
    m2 = y2.mean((0, 1))
    vecs = dict(m2=m2, m2_patch=y2[:, 1:].mean((0, 1)), m2_cls=y2[:, 0].mean(0),
                ffn0_patch=nd['ffn0'].double()[:, 1:].mean((0, 1)), ffn0_cls=nd['ffn0'].double()[:, 0].mean(0),
                attn0_patch=nd['attn0'].double()[:, 1:].mean((0, 1)),
                attn1_patch=nd['attn1'].double()[:, 1:].mean((0, 1)), attn1_cls=nd['attn1'].double()[:, 0].mean(0),
                e_patch=nd['e'].double()[:, 1:].mean((0, 1)))
    for k, v in vecs.items():
        g[f'{k}_mean_norm'] = float(v.norm())
    z1 = nd['z1'].double()
    zf = z1.flatten(0, 1)
    g['z1_mean'] = float(zf.mean())
    g['z1_cls_mean'] = float(z1[:, 0].mean())
    g['z1_dead'] = float(((zf > 0).double().mean(0) <= .005).double().mean())
    g['z1_cls_dead'] = float(((z1[:, 0] > 0).double().mean(0) <= .005).double().mean())
    g['ffn0_open_patch'] = float((nd['z0'][:, 1:] > 0).double().mean())
    g['ffn0_open_cls'] = float((nd['z0'][:, 0] > 0).double().mean())
    g['head_norm'] = float(head.weight.detach().norm())
    g['h_cls_norm_median'] = float(h.double().norm(dim=1).median())
    return g, {k: v.cpu().numpy().astype(np.float32) for k, v in vecs.items()}


def scale_derivative(model, x, y, head, rows):
    """∂L/∂α（FFN0 の patch 出力の倍率）と ∂L/∂α_h（head 特徴の倍率）を画像ごとに出し、当たり・外れで分けて平均する。"""
    W = head.weight.detach()[rows].double()
    b = head.bias.detach()[rows].double()
    one = torch.ones((), device=x.device)
    acc = dict(n=0, n_correct=0)
    sums = {k: 0. for k in ['sp', 'sp_c', 'sp_w', 'sp_neg', 'sh', 'sh_c', 'sh_w', 'sh_neg', 'loss']}
    e_sum = torch.zeros(W.shape[0], dtype=torch.float64, device=x.device)      # Σ_i (p_i − y_i)（K ベクトル）
    push_sum = torch.zeros(W.shape[1], dtype=torch.float64, device=x.device)   # Σ_i Wᵀ(p_i − y_i)（head 特徴への共通の勾配）
    h_sum = torch.zeros(W.shape[1], dtype=torch.float64, device=x.device)
    pred_hist = torch.zeros(W.shape[0], dtype=torch.float64, device=x.device)  # 予測クラスの度数
    for s in range(0, len(x), 125):
        xb, yb = x[s:s + 125], y[s:s + 125]
        h, dh = torch.func.jvp(lambda a: run_nodes(model, xb, alpha=a)[1], (one,), (one,))
        h, dh = h.detach().double(), dh.detach().double()
        logits = h @ W.T + b
        p = logits.softmax(1)
        e = p.clone()
        e[torch.arange(len(yb), device=x.device), yb] -= 1          # p − y
        sp = (e * (dh @ W.T)).sum(1)                                  # (p−y)ᵀ W ∂h/∂α
        sh = (e * (h @ W.T)).sum(1)                                   # (p−y)ᵀ W h
        correct = logits.argmax(1) == yb
        e_sum += e.sum(0)
        push_sum += (e @ W).sum(0)
        h_sum += h.sum(0)
        pred_hist += torch.bincount(logits.argmax(1), minlength=W.shape[0]).double()
        acc['n'] += len(yb)
        acc['n_correct'] += int(correct.sum())
        sums['loss'] += float(F.cross_entropy(logits, yb, reduction='sum'))
        for key, v in (('sp', sp), ('sh', sh)):
            sums[key] += float(v.sum())
            sums[key + '_c'] += float(v[correct].sum())
            sums[key + '_w'] += float(v[~correct].sum())
            sums[key + '_neg'] += int((v < 0).sum())
    n, nc = acc['n'], acc['n_correct']
    nw = n - nc
    out = dict(n_correct=nc, acc=nc / n, loss=sums['loss'] / n)
    for key, name in (('sp', 'dL_dalpha_ffn0'), ('sh', 'dL_dalpha_h')):
        out[name] = sums[key] / n
        out[name + '_correct'] = sums[key + '_c'] / nc if nc else None
        out[name + '_wrong'] = sums[key + '_w'] / nw if nw else None
        out[name + '_frac_negative'] = sums[key + '_neg'] / n
    # 共通の押し: 画像に依らない head 特徴への勾配の和 Σ Wᵀ(p−y)。ラベルが釣り合い予測も偏らなければ 0 に近い。
    # 崩れ（予測が少数のクラスに集中）では Σ(p−y) が大きくなり、体に画像に依らない押しが掛かる。
    push = -push_sum / n                                              # 降下の向き（−勾配）
    mean_h = h_sum / n
    out.update(pbar_minus_ybar=(e_sum / n).tolist(), pbar_minus_ybar_norm=float((e_sum / n).norm()),
               pred_max_frac=float(pred_hist.max() / n), common_push_norm=float(push.norm()),
               cos_push_mean_h=float(push @ mean_h / (push.norm() * mean_h.norm() + 1e-30)))
    return out, {'common_push': push.cpu().numpy().astype(np.float32), 'mean_h': mean_h.cpu().numpy().astype(np.float32)}


def main(args):
    assert 0 < args.steps <= 5000
    C.set_determinism(4)
    torch.cuda.set_per_process_memory_fraction(.3)
    tag = f'fable_replay_{args.arm}_t{args.task}_{args.adam}' + (f'_frz{args.freeze_body_until}' if args.freeze_body_until > 0 else '') + args.suffix
    dest = OUT / f'{tag}.json'
    if dest.exists():
        raise FileExistsError(dest)
    model = load_model(args.arm, args.task)
    opt = C.build_optimizer(model)
    adamfile = ckpt(args.arm, args.task, 'adam')
    loaded = False
    if adamfile.exists():
        st = torch.load(adamfile, map_location='cpu', weights_only=False)
        opt.load_state_dict(st['optimizer'])
        del st
        loaded = True
    if args.arm == 'R0':
        reset_head(model)                                      # head の Adam のモーメントは本走どおり残す
    head = model.layers['out']
    headids = {id(p) for p in head.parameters()}
    body = [p for p in model.parameters() if id(p) not in headids]
    if args.adam != 'keep':
        assert loaded
        for p in body:
            s = opt.state[p]
            s['exp_avg'].zero_()
            if args.adam == 'zero_mv':
                s['exp_avg_sq'].zero_()
                s['step'].zero_()
    body_group = None
    if args.freeze_body_until > 0:
        # 体と head を別の param group に分ける（lr は同じ）。体だけ最初の N 更新は lr=0 で止める。Adam のモーメントは勾配を受け続ける。
        assert len(opt.param_groups) == 1
        base = dict(opt.param_groups[0])
        allparams = base.pop('params')
        opt.param_groups = [dict(base, params=[p for p in allparams if id(p) not in headids]),
                            dict(base, params=[p for p in allparams if id(p) in headids])]
        body_group = opt.param_groups[0]
    x = images()
    xp = images(train=False)
    y = C.task_labels(len(x), SEED, args.task + 1, 10).cuda()
    rows = torch.tensor(C.loss_classes(args.arm, SEED, args.task + 1, 10, 250), device='cuda')

    def forward(xb):
        return F.linear(run_nodes(model, xb)[1], head.weight, head.bias).index_select(1, rows)

    @torch.no_grad()
    def evaluate():
        H = torch.cat([run_nodes(model, x[s:s + 125])[1] for s in range(0, len(x), 125)])
        logits = F.linear(H, head.weight, head.bias).index_select(1, rows)
        correct = int((logits.argmax(1) == y).sum())
        return dict(acc=correct / len(x), loss=float(F.cross_entropy(logits, y)),
                    h_train_norm_median=float(H.norm(dim=1).median()),
                    lr_sum_abs_h_median=float(LR * H.abs().sum(1).median()))

    records, early, derivs, vec_steps, vec_store = [], [], [], [], {}
    first90 = None
    start = time.monotonic()
    win = dict(body_step_sum=0., body_step_max=0., head_step_sum=0., batch_loss_sum=0., n=0)   # 前の記録からの窓の歩幅の和など

    def snapshot(step):
        nonlocal first90
        row = dict(step=step, **evaluate())
        row.update({f'win_{k}': v for k, v in win.items()})
        for k in win:
            win[k] = 0 if k == 'n' else 0.
        g, vecs = geometry(model, xp, head)
        row.update(g)
        records.append(row)
        if row['acc'] >= .9 and first90 is None:
            first90 = step
        if step % 50 == 0:
            sd, extra = scale_derivative(model, x, y, head, rows)
            derivs.append(dict(step=step, **sd))
            vec_steps.append(step)
            for k, v in list(vecs.items()) + list(extra.items()):
                vec_store.setdefault(k, []).append(v)
        if step % 250 == 0:
            print(f'{tag} step{step} acc={row["acc"]:.4f} loss={row["loss"]:.4g} ffn0_patch={row["ffn0_patch_rms"]:.4g} '
                  f'u2_cls={row["u2_cls_rms"]:.4g} head={row["head_norm"]:.3g} sec={time.monotonic() - start:.0f}', flush=True)

    def save(step):
        data = dict(arm=args.arm, task_end=args.task, next_task=args.task + 1, adam=args.adam, loaded_adam=loaded,
                    freeze_body_until=args.freeze_body_until,
                    reset_head=args.arm == 'R0', steps=step, T90=first90, records=records, early=early,
                    scale_derivative=derivs, seconds=time.monotonic() - start, resources=resources(),
                    inputs={str(ckpt(args.arm, args.task)): sha(ckpt(args.arm, args.task)),
                            str(adamfile): sha(adamfile) if loaded else None})
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        np.savez(OUT / f'{tag}_vectors.npz', steps=np.array(vec_steps), **{k: np.stack(v) for k, v in vec_store.items()})

    snapshot(0)
    last = 0
    for step, batch in enumerate(PairedBatches(len(x), 128, args.steps, SEED, args.task + 1), 1):
        ix = torch.tensor([i for i, _ in batch], device='cuda')
        opt.zero_grad(set_to_none=True)
        before = torch.cat([p.detach().flatten() for p in body])
        headbefore = head.weight.detach().clone()
        loss = F.cross_entropy(forward(x[ix]), y[ix])
        loss.backward()
        if step <= 200:
            gb = float(torch.stack([p.grad.square().sum() for p in body if p.grad is not None]).sum().sqrt())
            gh = float(torch.stack([p.grad.square().sum() for p in head.parameters() if p.grad is not None]).sum().sqrt())
        if body_group is not None:
            body_group['lr'] = 0. if step <= args.freeze_body_until else LR
        opt.step()
        last = step
        with torch.no_grad():
            after = torch.cat([p.detach().flatten() for p in body])
            bstep = float((after - before).norm())
            hstep = float((head.weight - headbefore).norm())
            win['body_step_sum'] += bstep
            win['body_step_max'] = max(win['body_step_max'], bstep)
            win['head_step_sum'] += hstep
            win['batch_loss_sum'] += float(loss)
            win['n'] += 1
            if step <= 200:
                early.append(dict(step=step, batch_loss=float(loss), body_grad_norm=gb, head_grad_norm=gh,
                                  body_step_norm=bstep, head_norm=float(head.weight.norm())))
        del before, after, headbefore
        if step % 25 == 0 or step == args.steps or step in FINE:
            snapshot(step)
        if step % 250 == 0:
            save(step)
    save(last)
    progress(f'{tag} 完了。更新 {last}、T90 {first90}、末の暗記率 {records[-1]["acc"]:.4f}、FFN0 patch rms {records[0]["ffn0_patch_rms"]:.4g}→{records[-1]["ffn0_patch_rms"]:.4g}、'
             f'保存 Adam {loaded}、資源 {resources()}、{time.monotonic() - start:.0f} 秒。')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=['R0', 'S'])
    p.add_argument('task', type=int)
    p.add_argument('--adam', choices=['keep', 'zero_m', 'zero_mv'], default='keep')
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--freeze_body_until', type=int, default=0, help='最初の N 更新は体を止める（lr=0、head だけ動く）。切替相の体の動きを除く反実仮想')
    p.add_argument('--suffix', default='')
    main(p.parse_args())

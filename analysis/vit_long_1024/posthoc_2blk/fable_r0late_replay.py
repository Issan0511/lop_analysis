"""R0 の後期（課題 60→61）の持続崩れ: 届いた時点で分岐する回し直し（Fable エージェント B、1028）。

保存状態（model_tK + adam_tK、R0 は head を 0 に戻す）から本走と同じ決定的な batch 列で 1 課題 ≤ 5,000 更新を eager で回す。
  --save_t90        暗記率が初めて 0.9 に届いた評価点で model と Adam を一度だけ保存し、そのまま --steps まで続ける
  --branch <state>  保存した状態から再開する（batch 列はその更新数ぶん飛ばす）。--freeze / --new_head をここで掛ける
  --freeze {none,body,head,ffn1,attn1,block0}  その群の requires_grad を切り lr を 0 にする（分岐の頭、または --save_t90 の走では使わない）
  --new_head        head を 0 に戻し head の Adam 状態を捨て、体を凍結して head だけ 5,000 更新で学び直す（新しい batch 列の頭から）
  --head_factor/--body_factor  lr 1e-4 に掛ける係数
一歩ごとの logit の変化 Δℓ = (ΔW)h + W(Δh) + ΔWΔh を固定 512 枚で、通常は --diag_every 更新ごと、T90 の前後の窓では毎更新、
体の項は群（block0 = 埋め込み+ブロック 0 / attn1 / ffn1）ごとに分けて記録する。
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.joudaki_vit_battle_0919.model import reset_head
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import (C, ROOT, SEED, ckpt, dump, features, images,
                                                                 load_model, now, parts)

OUT = ROOT / 'posthoc_r0_late'
GROUPS = ['block0', 'attn1', 'ffn1', 'head']
BODY = ['block0', 'attn1', 'ffn1']
LR = 1e-4
RSS_LIMIT_MIB = 2560


def progress(message):
    with (OUT / 'PROGRESS.md').open('a') as f:
        f.write(f'\n## {now()}\n\n{message}\n')


def resources():
    import resource
    out = {'rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if torch.cuda.is_initialized():
        out.update(cuda_alloc_mib=torch.cuda.max_memory_allocated() / 2**20,
                   cuda_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    if out['rss_mib'] > RSS_LIMIT_MIB:
        raise RuntimeError(f'RSS budget exceeded: {out}')
    return out


def setup():
    C.set_determinism(4)
    torch.cuda.set_per_process_memory_fraction(.30)


def group_of(name):
    if name.startswith('layers.out.'):
        return 'head'
    if 'block_1' in name:
        return 'attn1' if ('attn' in name or 'norm1' in name) else 'ffn1'
    if name.startswith('layers.norm.'):
        return 'ffn1'
    return 'block0'


def regroup(opt, model, head_factor, body_factor):
    base = dict(opt.param_groups[0])
    base.pop('params')
    named = list(model.named_parameters())
    groups = {g: [p for n, p in named if group_of(n) == g] for g in GROUPS}
    assert sum(len(v) for v in groups.values()) == len(named)
    opt.param_groups = [dict(base, params=groups[g], lr=LR * (head_factor if g == 'head' else body_factor), name=g)
                        for g in GROUPS]
    return groups


def set_factors(opt, head_factor, body_factor, frozen):
    for pg in opt.param_groups:
        g = pg['name']
        pg['lr'] = 0. if g in frozen else LR * (head_factor if g == 'head' else body_factor)


def freeze_groups(groups, frozen):
    for g in frozen:
        for p in groups[g]:
            p.requires_grad_(False)


def main(args):
    assert 0 < args.steps <= 5000
    setup()
    OUT.mkdir(exist_ok=True)
    dest = OUT / f'{args.tag}.json'
    if dest.exists():
        raise FileExistsError(dest)
    model = load_model(args.arm, args.task)
    opt = C.build_optimizer(model)
    frozen = set()
    if args.freeze == 'body':
        frozen = set(BODY)
    elif args.freeze != 'none':
        frozen = {args.freeze}
    if args.new_head:
        frozen = set(BODY)
    start_step = 0
    if args.branch:
        st = torch.load(args.branch, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model'])
        groups = regroup(opt, model, args.head_factor, args.body_factor)
        opt.load_state_dict(st['opt'])
        start_step = int(st['step'])
        assert st['arm'] == args.arm and st['task'] == args.task
        del st
        if args.new_head:
            reset_head(model)
            for p in groups['head']:
                opt.state.pop(p, None)
            start_step = 0                      # 新しい 5,000 更新の予算、batch 列の頭から
    else:
        adamfile = ckpt(args.arm, args.task, 'adam')
        assert adamfile.exists(), adamfile
        st = torch.load(adamfile, map_location='cpu', weights_only=False)
        opt.load_state_dict(st['optimizer'])
        del st
        if args.arm == 'R0':
            reset_head(model)
        groups = regroup(opt, model, args.head_factor, args.body_factor)
    set_factors(opt, args.head_factor, args.body_factor, frozen)
    freeze_groups(groups, frozen)
    body_frozen = frozen >= set(BODY)
    head = model.layers['out']
    x = images()
    y = C.task_labels(len(x), SEED, args.task + 1, 10).cuda()
    rows = torch.tensor(C.loss_classes(args.arm, SEED, args.task + 1, 10, 250), device='cuda')
    sub = torch.arange(0, len(x), len(x) // 512)[:512].cuda()      # 一歩の分解に使う固定 512 枚
    xs, ys = x[sub], y[sub]
    Hfix = None
    if body_frozen:
        with torch.no_grad():
            Hfix = torch.cat([features(model, x[s:s + 128]) for s in range(0, len(x), 128)])

    def feats(idx):
        if Hfix is not None:
            return Hfix[idx]
        return features(model, x[idx])

    @torch.no_grad()
    def all_logits():
        H = Hfix if Hfix is not None else torch.cat([features(model, x[s:s + 128]) for s in range(0, len(x), 128)])
        return H, F.linear(H, head.weight, head.bias).index_select(1, rows)

    @torch.no_grad()
    def margins(logits, labels):
        top2 = logits.topk(2, dim=1).values
        correct = logits.gather(1, labels[:, None])[:, 0]
        other = torch.where(logits.argmax(1) == labels, top2[:, 1], top2[:, 0])
        return correct - other

    records, diags, saved_state, first90 = [], [], None, None
    t0 = time.monotonic()

    @torch.no_grad()
    def evaluate(step):
        H, logits = all_logits()
        acc = float((logits.argmax(1) == y).double().mean())
        m = margins(logits, y)
        row = dict(step=step, acc=acc, loss=float(F.cross_entropy(logits, y)),
                   margin_median=float(m.median()), margin_q10=float(m.quantile(.1)),
                   h_norm_median=float(H.norm(dim=1).median()), eta_l1_median=float(LR * args.head_factor * H.abs().sum(1).median()),
                   head_rows_norm=float(head.weight[rows].norm()))
        if step % 100 == 0 and not body_frozen:
            nd = parts(model, xs[:256])
            zc = nd['z1'][:, 0]
            row.update(cls_open_frac=float((zc > 0).double().mean()), cls_alive=float((zc > 0).any(0).double().mean()))
        records.append(row)
        if step % 250 == 0 or step in (1, 25, 100):
            print(f'{args.tag} step{step} acc={acc:.4f} loss={row["loss"]:.4g} margin={row["margin_median"]:.3g} '
                  f'h={row["h_norm_median"]:.4g} sec={time.monotonic() - t0:.1f}', flush=True)
        return acc

    def snapshot():
        return {g: [p.detach().clone() for p in groups[g]] for g in BODY}

    def load_group(g, src):
        for p, q in zip(groups[g], src):
            p.data.copy_(q)

    def decompose(step, h0, W0, b0, l0, old, full):
        with torch.no_grad():
            W1, b1 = head.weight[rows].clone(), head.bias[rows].clone()
            h1 = feats(sub)
            l1 = F.linear(h1, W1, b1)
            dW, db = W1 - W0, b1 - b0
            headterm = F.linear(h0, dW, db)
            bodyterm = F.linear(h1 - h0, W0, None)
            cross = F.linear(h1 - h0, dW, None)

            def rms(v):
                return float((v - v.mean(1, keepdim=True)).square().mean().sqrt())
            m0, m1 = margins(l0, ys), margins(l1, ys)
            r = l0.argmax(1)
            d = dict(step=step, total=rms(l1 - l0), head=rms(headterm), body=rms(bodyterm), cross=rms(cross),
                     error=float((l1 - l0 - headterm - bodyterm - cross).abs().max()),
                     margin_median_before=float(m0.median()), dmargin_rms=float((m1 - m0).square().mean().sqrt()),
                     dmargin_abs_median=float((m1 - m0).abs().median()),
                     flip_frac=float((l1.argmax(1) != r).double().mean()),
                     correct_before=float((r == ys).double().mean()), correct_after=float((l1.argmax(1) == ys).double().mean()),
                     head_coord_max=float(dW.abs().max()))
            if full and not body_frozen:
                new = snapshot()
                parts_rms = {}
                for g in BODY:
                    for q in BODY:
                        load_group(q, new[q] if q == g else old[q])
                    hg = feats(sub)
                    parts_rms[g] = rms(F.linear(hg - h0, W0, None))
                for q in BODY:
                    load_group(q, new[q])
                d['body_groups'] = parts_rms
            diags.append(d)

    evaluate(start_step)
    last_acc = records[-1]['acc']
    if last_acc >= .9:
        first90 = start_step
    last = start_step
    stream = PairedBatches(len(x), 128, args.steps, SEED, args.task + 1)
    for step, batch in enumerate(stream, 1):
        if step <= start_step:
            continue
        ix = torch.tensor([i for i, _ in batch], device='cuda')
        full = (first90 is None and last_acc >= .8) or (first90 is not None and step <= first90 + args.window) \
            or (args.branch is not None and step <= start_step + args.window)
        diag = full or step % args.diag_every == 0
        if diag:
            with torch.no_grad():
                h0 = feats(sub)
                W0, b0 = head.weight[rows].clone(), head.bias[rows].clone()
                l0 = F.linear(h0, W0, b0)
                old = snapshot() if (full and not body_frozen) else None
        opt.zero_grad(set_to_none=True)
        logits = F.linear(feats(ix), head.weight, head.bias).index_select(1, rows)
        loss = F.cross_entropy(logits, y[ix])
        loss.backward()
        opt.step()
        last = step
        if diag:
            decompose(step, h0, W0, b0, l0, old, full)
        if step % 25 == 0 or step == args.steps:
            last_acc = evaluate(step)
            if last_acc >= .9 and first90 is None:
                first90 = step
                window_until = step + args.window
                if args.save_t90 and saved_state is None:
                    saved_state = OUT / f'{args.tag}_t90state.pt'
                    if not saved_state.exists():
                        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, arm=args.arm, task=args.task,
                                        head_factor=args.head_factor, body_factor=args.body_factor, acc=last_acc), saved_state)
                    progress(f'{args.tag}: 更新 {step} で暗記率 {last_acc:.4f}。状態を {saved_state.name} に保存。')
            if step % 500 == 0 or step == args.steps:
                dump(dest, dict(tag=args.tag, arm=args.arm, task_end=args.task, next_task=args.task + 1,
                                head_factor=args.head_factor, body_factor=args.body_factor, freeze=sorted(frozen),
                                new_head=args.new_head, branch=str(args.branch) if args.branch else None,
                                start_step=start_step, steps=step, T90=first90, records=records, step_diagnostics=diags,
                                saved_state=str(saved_state) if saved_state else None,
                                seconds=time.monotonic() - t0, resources=resources()))
    tail = [r['acc'] for r in records if r['step'] > args.steps - 1000]
    after = [r['acc'] for r in records if first90 is not None and r['step'] >= first90]
    summary = dict(T90=first90, final_acc=records[-1]['acc'], tail_mean=sum(tail) / len(tail), tail_min=min(tail),
                   min_after_t90=min(after) if after else None)
    out = json.loads(dest.read_text()) if dest.exists() else {}
    out.update(summary=summary, steps=last, records=records, step_diagnostics=diags, seconds=time.monotonic() - t0,
               resources=resources())
    dump(dest, out)
    progress(f'{args.tag} 完了。更新 {start_step}→{last}。T90={first90}。末 {summary["final_acc"]:.4f}、最後 1,000 更新の平均 '
             f'{summary["tail_mean"]:.4f}、最小 {summary["tail_min"]:.4f}。凍結={sorted(frozen)}。資源={resources()}。')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=['R0', 'S'])
    p.add_argument('task', type=int)
    p.add_argument('tag')
    p.add_argument('--head_factor', type=float, default=.01)
    p.add_argument('--body_factor', type=float, default=1.)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--save_t90', action='store_true')
    p.add_argument('--branch', type=Path, default=None)
    p.add_argument('--freeze', choices=['none', 'body', 'head', 'ffn1', 'attn1', 'block0'], default='none')
    p.add_argument('--new_head', action='store_true')
    p.add_argument('--diag_every', type=int, default=5)
    p.add_argument('--window', type=int, default=400)
    main(p.parse_args())

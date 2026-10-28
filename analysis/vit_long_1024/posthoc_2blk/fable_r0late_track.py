"""届いた時点の保存状態から介入なしで続け、毎更新の Adam の歩幅・batch の損失・logit の変化を記録する（Fable エージェント B、1028）。

分岐の元（head lr 1/100 の base）と同じ決定的な batch 列なので、軌跡は base と一致する。急落が起きる更新で何が特別かを見る:
  群ごとの ‖Δθ‖₂・‖Δθ‖∞（Adam の歩幅）、群ごとの勾配ノルム、batch の損失（平均・最大・最大 1 枚の取り分・誤り数）、
  固定 512 枚での一歩の logit 変化（全体 / head / 体）、余裕の変化、予測の入替。

    PYTHONPATH=$PWD python analysis/vit_long_1024/posthoc_2blk/fable_r0late_track.py <state.pt> <tag> [--steps 5000]
"""
import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import C, ROOT, SEED, dump, features, images, load_model
from analysis.vit_long_1024.posthoc_2blk.fable_r0late_replay import BODY, GROUPS, LR, progress, regroup, resources, set_factors, setup

OUT = ROOT / 'posthoc_r0_late'


def main(args):
    setup()
    dest = OUT / f'{args.tag}.json'
    if dest.exists():
        raise FileExistsError(dest)
    st = torch.load(args.state, map_location='cpu', weights_only=False)
    arm, task = st['arm'], st['task']
    model = load_model(arm, task)
    model.load_state_dict(st['model'])
    opt = C.build_optimizer(model)
    groups = regroup(opt, model, st['head_factor'], st['body_factor'])
    opt.load_state_dict(st['opt'])
    set_factors(opt, st['head_factor'], st['body_factor'], set())
    start = int(st['step'])
    del st
    head = model.layers['out']
    x = images()
    y = C.task_labels(len(x), SEED, task + 1, 10).cuda()
    rows = torch.tensor(C.loss_classes(arm, SEED, task + 1, 10, 250), device='cuda')
    sub = torch.arange(0, len(x), len(x) // 512)[:512].cuda()
    xs, ys = x[sub], y[sub]
    ngroup = {g: sum(p.numel() for p in groups[g]) for g in GROUPS}

    @torch.no_grad()
    def margins(logits, labels):
        top2 = logits.topk(2, dim=1).values
        correct = logits.gather(1, labels[:, None])[:, 0]
        other = torch.where(logits.argmax(1) == labels, top2[:, 1], top2[:, 0])
        return correct - other

    @torch.no_grad()
    def evaluate(step):
        H = torch.cat([features(model, x[s:s + 128]) for s in range(0, len(x), 128)])
        logits = F.linear(H, head.weight, head.bias).index_select(1, rows)
        m = margins(logits, y)
        return dict(step=step, acc=float((logits.argmax(1) == y).double().mean()), loss=float(F.cross_entropy(logits, y)),
                    margin_median=float(m.median()), margin_q10=float(m.quantile(.1)), h_norm_median=float(H.norm(dim=1).median()))

    records, steps_log = [evaluate(start)], []
    acc_sum, acc_norm, acc_n, acc_margin = {}, {}, {}, {}
    t0 = time.monotonic()
    for step, batch in enumerate(PairedBatches(len(x), 128, args.steps, SEED, task + 1), 1):
        if step <= start:
            continue
        ix = torch.tensor([i for i, _ in batch], device='cuda')
        with torch.no_grad():
            old = {g: [p.detach().clone() for p in groups[g]] for g in GROUPS}
            h0 = features(model, xs)
            W0, b0 = head.weight[rows].clone(), head.bias[rows].clone()
            l0 = F.linear(h0, W0, b0)
        opt.zero_grad(set_to_none=True)
        logits = F.linear(features(model, x[ix]), head.weight, head.bias).index_select(1, rows)
        per = F.cross_entropy(logits, y[ix], reduction='none')
        loss = per.mean()
        loss.backward()
        with torch.no_grad():
            gn = {g: float(torch.stack([p.grad.square().sum() for p in groups[g] if p.grad is not None]).sum().sqrt()) for g in GROUPS}
            g1 = {g: float(torch.stack([p.grad.abs().sum() for p in groups[g] if p.grad is not None]).sum()) for g in GROUPS}
        opt.step()
        with torch.no_grad():
            d2 = {g: float(torch.stack([(p - q).square().sum() for p, q in zip(groups[g], old[g])]).sum().sqrt()) for g in GROUPS}
            dinf = {g: float(torch.stack([(p - q).abs().max() for p, q in zip(groups[g], old[g])]).max()) for g in GROUPS}
            h1 = features(model, xs)
            W1, b1 = head.weight[rows].clone(), head.bias[rows].clone()
            l1 = F.linear(h1, W1, b1)
            headterm = F.linear(h0, W1 - W0, b1 - b0)
            bodyterm = F.linear(h1 - h0, W0, None)

            def rms(v):
                return float((v - v.mean(1, keepdim=True)).square().mean().sqrt())

            def cen(v):
                return v - v.mean(1, keepdim=True)
            # 体の項を群ごとに分ける（その群だけ新しい値、他は古い値で順伝播）
            new = {g: [p.detach().clone() for p in groups[g]] for g in BODY}
            gterm = {}
            for g in BODY:
                for q in BODY:
                    for p, src in zip(groups[q], new[q] if q == g else old[q]):
                        p.data.copy_(src)
                gterm[g] = cen(F.linear(features(model, xs) - h0, W0, None))
            for q in BODY:
                for p, src in zip(groups[q], new[q]):
                    p.data.copy_(src)
            cos = {}
            for a in range(len(BODY)):
                for b in range(a + 1, len(BODY)):
                    u, v = gterm[BODY[a]], gterm[BODY[b]]
                    cos[f'{BODY[a]}_{BODY[b]}'] = float((u * v).sum() / (u.norm() * v.norm() + 1e-30))
            upstream = gterm['block0'] + gterm['attn1']
            m0, m1 = margins(l0, ys), margins(l1, ys)
            r0 = l0.argmax(1)
            masked = l0.clone()
            masked[torch.arange(len(ys), device=ys.device), ys] = -float('inf')
            rr = masked.argmax(1)                                   # 正解以外の最大（余裕の相手）
            # 一貫性: 項の和のノルム / ノルムの和（1 なら毎歩同じ向き、1/√n なら乱歩）
            for key, term in (('head', cen(headterm)), ('body', cen(bodyterm)), ('ffn1', gterm['ffn1']), ('upstream', upstream)):
                acc_sum[key] = acc_sum.get(key, 0) + term
                acc_norm[key] = acc_norm.get(key, 0.) + float(term.norm())
                acc_n[key] = acc_n.get(key, 0) + 1
                contrib = term.gather(1, ys[:, None])[:, 0] - term.gather(1, rr[:, None])[:, 0]
                acc_margin[key] = acc_margin.get(key, 0) + contrib.double()
            steps_log.append(dict(
                coherence={k: float(acc_sum[k].norm()) / (acc_norm[k] + 1e-30) for k in acc_sum},
                cum_margin_median={k: float(acc_margin[k].median()) for k in acc_margin},
                cum_n=acc_n['head'],
                groups_rms={g: float(gterm[g].square().mean().sqrt()) for g in BODY}, group_cos=cos,
                upstream_rms=float(upstream.square().mean().sqrt()),
                upstream_plus_ffn1_rms=float((upstream + gterm['ffn1']).square().mean().sqrt()),
                cos_ffn1_upstream=float((gterm['ffn1'] * upstream).sum() / (gterm['ffn1'].norm() * upstream.norm() + 1e-30)),
                group_sum_error=float((cen(bodyterm) - upstream - gterm['ffn1']).square().mean().sqrt()),
                step=step, batch_loss=float(loss), batch_loss_max=float(per.max()), batch_top1_share=float(per.max() / per.sum()),
                batch_wrong=int((logits.argmax(1) != y[ix]).sum()),
                grad_l2={g: gn[g] for g in GROUPS}, grad_l1={g: g1[g] for g in GROUPS},
                dtheta_l2={g: d2[g] for g in GROUPS}, dtheta_inf={g: dinf[g] for g in GROUPS},
                dtheta_l2_over_eta_sqrt_n={g: d2[g] / (LR * (.01 if g == 'head' else 1.) * ngroup[g] ** .5) for g in GROUPS},
                total=rms(l1 - l0), head=rms(headterm), body=rms(bodyterm),
                margin_median_before=float(m0.median()), margin_q10_before=float(m0.quantile(.1)),
                dmargin_abs_median=float((m1 - m0).abs().median()), dmargin_abs_max=float((m1 - m0).abs().max()),
                flip_frac=float((l1.argmax(1) != r0).double().mean()),
                correct_before=float((r0 == ys).double().mean()), correct_after=float((l1.argmax(1) == ys).double().mean()),
                h_norm_median=float(h1.norm(dim=1).median())))
        if step % 25 == 0 or step == args.steps:
            records.append(evaluate(step))
            if step % 250 == 0:
                print(f'{args.tag} step{step} acc={records[-1]["acc"]:.4f} sec={time.monotonic() - t0:.0f}', flush=True)
        if step % 500 == 0 or step == args.steps:
            dump(dest, dict(tag=args.tag, arm=arm, task_end=task, start_step=start, steps=step, n_params=ngroup,
                            records=records, per_step=steps_log, seconds=time.monotonic() - t0, resources=resources()))
    progress(f'{args.tag} 完了。更新 {start}→{args.steps}。末の暗記率 {records[-1]["acc"]:.4f}。資源={resources()}。')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('state', type=Path)
    p.add_argument('tag')
    p.add_argument('--steps', type=int, default=5000)
    main(p.parse_args())

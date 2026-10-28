"""届いた時点の状態で、一歩の logit の余裕の変化の上界を群ごとに出す（Fable エージェント B、1028）。

Adam の一歩は座標ごとに |Δθ_i| = η |m̂_i|/(√v̂_i + ε) で、β1 = 0.9・β2 = 0.999 では |m̂_i|/√v̂_i ≤ (1−β1)/√(1−β2) = 3.16
（Kingma & Ba 2015 §2.1 の実効歩幅の上界。(1−β1) > √(1−β2) の場合）。一次近似で
  |Δ(ℓ_y − ℓ_r)| ≤ Σ_i |∂(ℓ_y − ℓ_r)/∂θ_i| |Δθ_i| ≤ 3.16 η_g Σ_{i∈g} |∂(ℓ_y − ℓ_r)/∂θ_i|   （群 g ごとに足す）
を、固定の訓練画像 256 枚で画像ごとに測り、余裕 m = ℓ_y − ℓ_r（r は正解以外の最大）と比べる。
JSON の bound_* は係数 3.16 を掛ける前の η_g‖∂m/∂θ_g‖₁ を保存する。
Adam の状態（exp_avg, exp_avg_sq）から出す「いまの歩幅」での和 Σ|g_i| η|m̂_i|/(√v̂_i+ε) も併記する。

    PYTHONPATH=$PWD python analysis/vit_long_1024/posthoc_2blk/fable_r0late_bound.py <state.pt> <tag>
"""
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import C, ROOT, SEED, dump, features, images, load_model
from analysis.vit_long_1024.posthoc_2blk.fable_r0late_replay import GROUPS, LR, group_of, progress, regroup, resources, setup

OUT = ROOT / 'posthoc_r0_late'


def main(state_path, tag):
    setup()
    st = torch.load(state_path, map_location='cpu', weights_only=False)
    arm, task = st['arm'], st['task']
    model = load_model(arm, task)
    model.load_state_dict(st['model'])
    opt = C.build_optimizer(model)
    groups = regroup(opt, model, st['head_factor'], st['body_factor'])
    opt.load_state_dict(st['opt'])
    factors = {g: (st['head_factor'] if g == 'head' else st['body_factor']) for g in GROUPS}
    names = {id(p): n for n, p in model.named_parameters()}
    x = images()
    y = C.task_labels(len(x), SEED, task + 1, 10).cuda()
    rows = torch.tensor(C.loss_classes(arm, SEED, task + 1, 10, 250), device='cuda')
    sub = torch.arange(0, len(x), len(x) // 512)[:256].cuda()
    head = model.layers['out']
    params = [p for g in GROUPS for p in groups[g]]
    pos = {id(p): k for k, p in enumerate(params)}
    # いまの Adam の歩幅（座標ごと）
    stepsize = {}
    for g in GROUPS:
        for p in groups[g]:
            s = opt.state.get(p)
            if s is None or 'exp_avg' not in s:
                stepsize[id(p)] = None
                continue
            t = float(s['step']) if torch.is_tensor(s['step']) else float(s['step'])
            m = s['exp_avg'] / (1 - .9 ** t)
            v = s['exp_avg_sq'] / (1 - .999 ** t)
            stepsize[id(p)] = (LR * factors[g] * m.abs() / (v.sqrt() + 1e-8)).cuda()
    per_image = []
    for i in sub.tolist():
        xi = x[i:i + 1]
        h = features(model, xi)
        logits = F.linear(h, head.weight, head.bias).index_select(1, rows)[0]
        yi = int(y[i])
        other = logits.clone()
        other[yi] = -float('inf')
        r = int(other.argmax())
        margin = logits[yi] - logits[r]
        grads = torch.autograd.grad(margin, params, allow_unused=True)
        row = dict(index=i, margin=float(margin.detach()), correct=bool(logits.argmax() == yi))
        for g in GROUPS:
            l1 = l2 = state_sum = 0.
            for p in groups[g]:
                gi = grads[pos[id(p)]]
                if gi is None:
                    continue
                l1 += float(gi.abs().sum())
                l2 += float(gi.square().sum())
                ss = stepsize[id(p)]
                if ss is not None:
                    state_sum += float((gi.abs() * ss).sum())
            row[f'l1_{g}'] = l1
            row[f'l2_{g}'] = math.sqrt(l2)
            row[f'bound_{g}'] = LR * factors[g] * l1
            row[f'statebound_{g}'] = state_sum
        row['bound_body'] = sum(row[f'bound_{g}'] for g in GROUPS if g != 'head')
        row['statebound_body'] = sum(row[f'statebound_{g}'] for g in GROUPS if g != 'head')
        per_image.append(row)
    keys = [k for k in per_image[0] if k not in ('index', 'correct')]
    tens = {k: torch.tensor([r[k] for r in per_image], dtype=torch.float64) for k in keys}
    margin = tens['margin']
    summary = dict(tag=tag, arm=arm, task=task, step=int(st['step']), acc_at_state=float(st.get('acc', float('nan'))),
                   n_images=len(per_image), correct_frac=float(sum(r['correct'] for r in per_image) / len(per_image)),
                   margin=dict(median=float(margin.median()), q10=float(margin.quantile(.1)), q90=float(margin.quantile(.9))))
    for g in GROUPS + ['body']:
        for kind in ('bound', 'statebound'):
            b = tens[f'{kind}_{g}']
            ratio = b / margin.clamp_min(1e-9)
            summary[f'{kind}_{g}'] = dict(median=float(b.median()), q90=float(b.quantile(.9)),
                                          ratio_median=float(ratio.median()),
                                          frac_ratio_gt1=float((ratio > 1).double().mean()))
    stepmed = {}
    for g in GROUPS:
        vals = [stepsize[id(p)].flatten() for p in groups[g] if stepsize[id(p)] is not None]
        if vals:
            v = torch.cat(vals)
            stepmed[g] = dict(median=float(v.median()), q90=float(v.quantile(.9)), max=float(v.max()), eta=LR * factors[g])
    summary['adam_step_abs'] = stepmed
    summary['resources'] = resources()
    dump(OUT / f'{tag}.json', dict(summary=summary, per_image=per_image))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    progress(f'{tag}: 上界の計算完了（{len(per_image)} 枚）。余裕の中央値 {summary["margin"]["median"]:.3g}、'
             f'体の上界の中央値 {summary["bound_body"]["median"]:.3g}、head の上界 {summary["bound_head"]["median"]:.3g}。')


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2])

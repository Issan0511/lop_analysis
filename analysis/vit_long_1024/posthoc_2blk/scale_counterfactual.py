"""重みの大きさだけを変えて次の課題を回し直す（Claude、1027）: 「w が大きいと実効的な学習率が下がる」のどの w かを決める。
保存した課題 T の末（重み＋Adam）から課題 T+1 を回し、暗記率が 0.9 に届くまでの更新数 T90 を測る（25 更新ごとに 2,500 枚で評価）。

介入（mode）:
  base        そのまま
  head0       head の 200 行を 0 に戻す（R0L の手順）
  fc1x<c>     2 ブロックの fc1 の重みと bias を c 倍、fc2 の重みを 1/c 倍（ReLU の正斉次性で関数は不変）。
              Adam の moment も同じ比で換算（m を c 倍・v を c² 倍、fc2 は逆）。変わるのは lr の相対的な大きさだけ。
  headx<c>    head の損失の行の重みと bias を c 倍（向きは保ち、logit の大きさ＝確信だけを変える）。Adam の moment も換算。
  proj<k>     最終 LN の後（head が読む特徴）に固定の射影を挟む: h → μ + P(h − μ) + ε(I − P)(h − μ)。
              μ と P（上位 k 主成分への射影）は課題の始めの 2,500 枚の CLS 特徴から 1 回だけ決め、以後動かさない。
              ε は既定 0。proj<k>e<ε> で ε を指定、proj<k>m<n> で射影後の有効次元（参加比）が n になる ε を選ぶ。
              最終 LN の出力は head にしか行かないので、これは新しい head の行を上位 k 方向（＋ε 倍の残り）に縛るのと同じ。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json \
        python analysis/vit_long_1024/posthoc_2blk/scale_counterfactual.py SL 100 fc1x0.66 <出力.json> [--max-steps 3000]
"""
import argparse
import re
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import train_loader
from analysis.joudaki_vit_battle_0919.model import prepare_noise, reset_head, train_forward, update_adaptive
from analysis.joudaki_vit_battle_0919.run import set_rng
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.run import Recorder, first_reach

SEED = 10252010
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'


class LowRank(torch.nn.Module):
    """h → μ + P(h − μ) + ε(I − P)(h − μ)、P = U Uᵀ（固定の buffer）。"""

    def __init__(self, mu, U, eps):
        super().__init__()
        self.register_buffer('mu', mu)
        self.register_buffer('U', U)
        self.eps = eps

    def forward(self, x):
        c = x - self.mu
        p = (c @ self.U) @ self.U.T
        return self.mu + p + self.eps * (c - p)


def participation(e):
    return float(e.sum() ** 2 / (e ** 2).sum())


def projected_pr(e, k, eps):
    return participation(torch.cat([e[:k], eps ** 2 * e[k:]]))


@torch.no_grad()
def rescale(param, optimizer, c):
    param.mul_(c)
    st = optimizer.state.get(param)
    if st:
        st['exp_avg'].mul_(c)
        st['exp_avg_sq'].mul_(c * c)


@torch.no_grad()
def cls_features(model, x, raw=False):
    """head が読む CLS 特徴（最終の norm の後）。raw なら射影を挟む前。"""
    L = model.layers
    hh = L['patch_embed'](x)
    hh = torch.cat((model.cls_token.expand(hh.shape[0], -1, -1), hh), 1) + model.pos_embed
    for i in range(model.depth):
        hh = L[f'block_{i}'](hh)
    norm = L['norm']
    if raw and isinstance(norm, torch.nn.Sequential):
        norm = norm[0]
    return norm(hh)[:, 0]


def main(args):
    C.set_determinism(4)
    device = 'cuda'
    d = RUNS / args.arm / f'seed{SEED}'
    model = C.build_arm_model(args.arm, SEED, device, 0.0, 2, 8.0)
    model.load_state_dict(torch.load(d / f'ckpt/model_t{args.t_end:02d}.pt', map_location=device, weights_only=False))
    optimizer = C.build_optimizer(model)
    adam = torch.load(d / f'ckpt/adam_t{args.t_end:02d}.pt', map_location=device, weights_only=False)
    assert adam['task'] == args.t_end
    optimizer.load_state_dict(adam['optimizer'])
    set_rng(adam['rng'])
    source = C.image_source('real')
    task = args.t_end + 1
    data, _ = C.task_dataset(source, SEED, task, 10, 250, augment=False)
    rows_t = torch.tensor(C.loss_classes(args.arm, SEED, task, 10, 250), device=device)
    x_eval, y_eval = C.eval_tensor(data.images, device), data.labels.to(device)
    model.eval()
    with torch.no_grad():
        logits_before = torch.cat([model(x_eval[s:s + 128]) for s in range(0, 2500, 128)])
    # 介入
    head = model.layers['out']
    note = {}
    for mode in args.mode.split('+'):            # 'head0+fc1x0.3' のように重ねられる
        if mode == 'head0':
            reset_head(model)
        elif mode == 'headrandn':                 # 損失の行を、同じ行ノルムの乱数の向きに置き換える（大きさだけ残す）
            with torch.no_grad():
                g = torch.Generator(device='cpu').manual_seed(2027)
                W = head.weight[rows_t]
                R = torch.randn(W.shape, generator=g).to(W.device)
                R = R / R.norm(dim=1, keepdim=True) * W.norm(dim=1, keepdim=True)
                head.weight[rows_t] = R
        elif mode == 'headperm':                  # 損失の 10 行（と bias）の割り当てを入れ替える（確信は保ち、どの行がどのクラスかだけ変える）
            with torch.no_grad():
                g = torch.Generator(device='cpu').manual_seed(2027)
                perm = torch.randperm(len(rows_t), generator=g).to(rows_t.device)
                while bool((perm == torch.arange(len(rows_t), device=rows_t.device)).any()):
                    perm = torch.randperm(len(rows_t), generator=g).to(rows_t.device)
                head.weight[rows_t] = head.weight[rows_t][perm]
                head.bias[rows_t] = head.bias[rows_t][perm]
                note['perm'] = perm.tolist()
        elif mode in ('ffn1init', 'ffn2init', 'attn1init', 'attn2init'):   # その部分だけ初期値（model_t00.pt）に戻し、Adam の記憶も消す
            init = torch.load(d / 'ckpt/model_t00.pt', map_location=device, weights_only=False)
            i = int(re.search(r'\d', mode).group()) - 1
            part = 'mlp' if mode.startswith('ffn') else 'attn'
            prefix = f'layers.block_{i}.layers.{part}.'
            with torch.no_grad():
                sd = model.state_dict()
                for k in sd:
                    if k.startswith(prefix) and k in init and torch.is_tensor(sd[k]):
                        sd[k].copy_(init[k])
            for name, prm in model.named_parameters():
                if name.startswith(prefix):
                    optimizer.state.pop(prm, None)
            note.setdefault('reset', []).append(mode)
        elif mode == 'adam0':                     # 体と head の Adam の記憶を全部消す（重みはそのまま）
            optimizer.state.clear()
        elif mode.startswith('fc1x'):
            c = float(mode[4:])
            for i in range(2):
                mlp = model.layers[f'block_{i}'].layers['mlp'].layers
                rescale(mlp['fc1'].weight, optimizer, c)
                rescale(mlp['fc1'].bias, optimizer, c)
                rescale(mlp['fc2'].weight, optimizer, 1 / c)
            note['c_fc1'] = c
        elif mode.startswith('headx'):
            c = float(mode[5:])
            rescale(head.weight, optimizer, c)
            rescale(head.bias, optimizer, c)
            note['c_head'] = c
        elif mode.startswith('proj'):
            m = re.fullmatch(r'proj(\d+)(?:e([\d.]+)|m([\d.]+))?', mode)
            k = int(m.group(1))
            model.eval()
            with torch.no_grad():
                H = torch.cat([cls_features(model, x_eval[s_:s_ + 250]) for s_ in range(0, 2500, 250)]).double()
                mu = H.mean(0)
                _, sv, Vh = torch.linalg.svd(H - mu, full_matrices=False)
            e = sv ** 2
            if m.group(2):
                eps = float(m.group(2))
            elif m.group(3):
                target, lo, hi = float(m.group(3)), 0.0, 1.0
                if target >= projected_pr(e, k, 1.0):
                    lo = hi = 1.0
                elif target <= projected_pr(e, k, 0.0):
                    hi = 0.0
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if projected_pr(e, k, mid) < target else (lo, mid)
                eps = (lo + hi) / 2
            else:
                eps = 0.0
            dtype = model.cls_token.dtype
            proj = LowRank(mu.to(dtype), Vh[:k].T.contiguous().to(dtype), eps)
            model.layers['norm'] = torch.nn.Sequential(model.layers['norm'], proj)
            note['proj'] = dict(k=k, eps=eps, pr_before=participation(e), pr_after=projected_pr(e, k, eps),
                                var_kept=float(torch.cat([e[:k], eps ** 2 * e[k:]]).sum() / e.sum()))
        elif mode != 'base':
            raise ValueError(mode)
    if C.zero_head_at(args.arm, task, 1) and args.mode != 'head0':
        reset_head(model)                       # 戻す腕なら腕の規則どおり
    with torch.no_grad():
        logits_after = torch.cat([model(x_eval[s:s + 128]) for s in range(0, 2500, 128)])
    note['max_logit_change'] = float((logits_after - logits_before).abs().max())
    note['fc1_row_norm_med'] = [float(model.layers[f'block_{i}'].layers['mlp'].layers['fc1'].weight.norm(dim=1).median()) for i in range(2)]
    note['head_norm'] = float(head.weight[rows_t].norm())
    @torch.no_grad()
    def eff_dim():
        """head が読む特徴の参加比と上位 10 の割合。射影を挟んだときは、射影の前（体が作る特徴）の 2 つも足す。"""
        was = model.training; model.eval()
        hs = [cls_features(model, x_eval[s_:s_ + 250], raw=True) for s_ in range(0, 2500, 250)]
        out = []
        for H in ([torch.cat([model.layers['norm'][1](h) for h in hs])] if 'proj' in note else []) + [torch.cat(hs)]:
            H = H.double(); H = H - H.mean(0)
            e = torch.linalg.svdvals(H) ** 2
            out += [float(e.sum() ** 2 / (e ** 2).sum()), float(e[:10].sum() / e.sum())]
        model.train(was)
        return tuple(out)
    dims = [(0, *eff_dim())] if args.track_dim else []
    forward = train_forward(model, 'compile')
    model.train()
    points = args.max_steps // C.EVAL_EVERY + 1
    rec = Recorder(model, x_eval, y_eval, rows_t, points, device)
    rec.take(0)
    t0 = time.monotonic()
    losses = []
    last = 0
    status = 'budget'
    for step, (x, y) in enumerate(train_loader(data, SEED, task, args.max_steps, 2), 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prepare_noise(model, len(y))
        loss = F.cross_entropy(forward(x).index_select(1, rows_t), y)
        loss.backward()
        optimizer.step()
        update_adaptive(model)
        losses.append(float(loss))
        if args.track_dim and step % 50 == 0:
            dims.append((step, *eff_dim()))
        if step % C.EVAL_EVERY == 0:
            last = step // C.EVAL_EVERY
            rec.take(last)
            if int(rec.correct[last]) * 100 >= 90 * 2500 and not args.full:
                status = 'reached90'
                break
    torch.cuda.synchronize()
    r = rec.numpy()
    n = 2500
    out = dict(arm=args.arm, t_end=args.t_end, task=task, mode=args.mode, note=note, status=status, steps_run=len(losses),
               T50=first_reach(r['correct'][:last + 1], n, 50), T75=first_reach(r['correct'][:last + 1], n, 75),
               T90=first_reach(r['correct'][:last + 1], n, 90),
               mem_acc=(r['correct'][:last + 1] / n).tolist(), mem_loss=(r['loss'][:last + 1] / n).tolist(),
               loss_100=[sum(losses[i:i + 100]) / len(losses[i:i + 100]) for i in range(0, len(losses), 100)],
               seconds=round(time.monotonic() - t0, 1), eff_dim=dims)
    Path(args.out).write_text(json.dumps(out, indent=1))
    if dims:
        print('  有効次元 (更新, 参加比/上位10の割合[/射影前の参加比/上位10]):', ' '.join(f'{u}:' + '/'.join(f'{v:.2f}' for v in rest) for u, *rest in dims[::2]))
    if 'proj' in note:
        print('  射影:', {k_: round(v, 4) if isinstance(v, float) else v for k_, v in note['proj'].items()})
    print(f"{args.arm} T{args.t_end}→{task} {args.mode}: T50 {out['T50']} T75 {out['T75']} T90 {out['T90']} ({status}, {out['steps_run']} 更新, {out['seconds']} s)"
          f" | 始めの暗記損失 {out['mem_loss'][0]:.2f} | 関数の変化 {note['max_logit_change']:.1e} | fc1 行ノルム中央値 {note['fc1_row_norm_med']} | head ノルム {note['head_norm']:.2f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('arm'); p.add_argument('t_end', type=int); p.add_argument('mode'); p.add_argument('out')
    p.add_argument('--max-steps', type=int, default=3000)
    p.add_argument('--full', action='store_true', help='0.9 に届いても止めない')
    p.add_argument('--track-dim', action='store_true', help='50 更新ごとに head が読む特徴の有効次元を記録')
    main(p.parse_args())

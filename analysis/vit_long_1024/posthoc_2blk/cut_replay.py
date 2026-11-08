"""追補 H（1108）: 枝の切除の回し直し。scale_counterfactual.py の写し（元は触らない）に、seed の引数化・枝の切除・初期値から始める選択肢を足した。

保存状態（--init ckpt なら model_t{t_end}.pt、--init t00 なら同じ seed の初期値 model_t00.pt）から課題 t_end+1 を
engine のラベルとミニバッチ順（C.task_dataset・train_loader、seed と課題 t_end+1）で U 更新回し、25 更新ごとに 2,500 枚で評価する。
0.9 に届いても止めない（A_end に最後の 500 更新が要る）。

mode（'+' で重ねる。適用の順は文字列の順によらず: 切除 → head0 → adam0）:
  none / base       何もしない
  cut_ffn0 / cut_attn0 / cut_ffn1 / cut_attn1
                    block_i の mlp / attn の出力を forward hook で置き換える（残差 x = x + 出力 は残す）。その枝のパラメタは requires_grad False。
                    --cut-kind mean: 課題 t_end の 2,500 枚での出力の平均（token 位置ごとの定数ベクトル。切除なしの網で全枝分を先に取る）。
                    --cut-kind zero: 0 * output。既定は --init ckpt → mean、--init t00 → zero（追補 H.3、1108 修正）。
  head0             head の 200 行と bias を 0 に戻す。
  adam0             Adam を新しく作る。対象は requires_grad が残るパラメタだけ（切除した枝は param group に入らない）。
  --h               mode の前に head0+adam0+ を足す（追補 H.3: どの腕も head 0・Adam 新品）。

    cd <worktree> && PYTHONPATH=. VIT_LONG_PRODUCTION=production_2blk_rlfix.json \
        python analysis/vit_long_1024/posthoc_2blk/cut_replay.py S 60 cut_ffn1 <出力.json> --seed 10252011 --init ckpt --h \
        [--max-steps 5000] [--track-steps 100] [--device cuda] [--engine compile] [--self-check]
"""
import argparse
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

RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'
CUTS = {'cut_ffn0': (0, 'mlp'), 'cut_attn0': (0, 'attn'), 'cut_ffn1': (1, 'mlp'), 'cut_attn1': (1, 'attn')}
A_END_POINTS = 20                                  # 最後の 500 更新 = 評価 20 点（EVAL_EVERY 25）


def zero_output(module, inputs, output):
    return 0 * output


def branch(model, i, part):
    return model.layers[f'block_{i}'].layers[part]


def const_output(value):
    """平均置換の hook: 枝の出力を token 位置ごとの定数ベクトル value（[tokens, dim]）を batch に広げたものに置き換える。"""
    def hook(module, inputs, output):
        return value.to(output.dtype).expand(output.shape[0], -1, -1)
    return hook


@torch.no_grad()
def branch_means(model, cuts, x, batch=250):
    """切る枝ごとに、出力の全画像平均（token 位置ごとの定数ベクトル [tokens, dim]）。
    どの枝の平均も、切除を 1 つも入れていない読み込んだままの網で取る（同じ mode の先の切除は入れない）。hook は全部の平均を取った後で付ける。"""
    was = model.training
    model.eval()
    sums, n = {c: None for c in cuts}, 0
    hs = []
    for c in cuts:
        def rec(mod, inp, out, c=c):              # None を返す（出力は変えない）
            v = out.double().sum(0)
            sums[c] = v if sums[c] is None else sums[c] + v
        hs.append(branch(model, *CUTS[c]).register_forward_hook(rec))
    for s_ in range(0, len(x), batch):
        model(x[s_:s_ + batch])
        n += len(x[s_:s_ + batch])
    for h in hs:
        h.remove()
    model.train(was)
    dtype = model.cls_token.dtype
    return {c: (sums[c] / n).to(dtype) for c in cuts}


def apply_cuts(model, cuts, means=None):
    """切除: hook を登録し、枝のパラメタを凍結する。means（branch_means の返り値）があれば平均置換、無ければ 0 を掛ける。
    返り値は hook の handle と凍結したパラメタの名前。"""
    handles, frozen = [], []
    for c in cuts:
        i, part = CUTS[c]
        sub = branch(model, i, part)
        handles.append(sub.register_forward_hook(const_output(means[c]) if means is not None else zero_output))
        for name, prm in sub.named_parameters():
            prm.requires_grad_(False)
            frozen.append(f'layers.block_{i}.layers.{part}.{name}')
    return handles, frozen


def new_adam(model):
    """C.build_optimizer と同じ設定（lr・betas・eps・wd 0・fused）で、requires_grad が残るパラメタだけ。"""
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=C.LR, betas=(.9, .999), eps=1e-8,
                            weight_decay=0., fused=True)


def parse_mode(mode):
    parts = [m for m in mode.split('+') if m not in ('', 'none', 'base')]
    cuts = sorted({m for m in parts if m in CUTS}, key=list(CUTS).index)
    rest = [m for m in parts if m not in CUTS]
    for m in rest:
        if m not in ('head0', 'adam0'):
            raise ValueError(f'追補 H の mode ではない: {m}')
    return cuts, 'head0' in rest, 'adam0' in rest


def main(args):
    C.set_determinism(args.threads)
    device = args.device
    seed = args.seed
    mode = ('head0+adam0+' + args.mode) if args.h else args.mode
    cuts, do_head0, do_adam0 = parse_mode(mode)
    d = RUNS / args.arm / f'seed{seed}'
    model = C.build_arm_model(args.arm, seed, device, 0.0, 2, 8.0)
    src = d / ('ckpt/model_t00.pt' if args.init == 't00' else f'ckpt/model_t{args.t_end:02d}.pt')
    model.load_state_dict(torch.load(src, map_location=device, weights_only=False))
    for i in range(model.depth):                   # LN なしの箱（S）: 枝の前の norm は Identity のはず
        for nm in ('norm1', 'norm2'):
            assert isinstance(model.layers[f'block_{i}'].layers[nm], torch.nn.Identity), (i, nm)
    adam_path = d / f'ckpt/adam_t{args.t_end:02d}.pt'
    adam = None
    if args.init == 'ckpt' and adam_path.exists():
        adam = torch.load(adam_path, map_location=device, weights_only=False)
        assert adam['task'] == args.t_end
        if device == 'cuda':
            set_rng(adam['rng'])
        else:
            torch.set_rng_state(adam['rng']['cpu'].cpu())
    else:
        torch.manual_seed(seed + args.t_end)
    source = C.image_source('real')
    task = args.t_end + 1
    data, _ = C.task_dataset(source, seed, task, 10, 250, augment=False)
    rows_t = torch.tensor(C.loss_classes(args.arm, seed, task, 10, 250), device=device)
    x_eval, y_eval = C.eval_tensor(data.images, device), data.labels.to(device)
    model.eval()
    with torch.no_grad():
        logits_before = torch.cat([model(x_eval[s:s + 128]) for s in range(0, 2500, 128)])
    head = model.layers['out']
    head_ids = {id(head.weight), id(head.bias)}
    note = dict(src=str(src))
    # 介入: 切除 → head0 → Adam
    # 切除の中身（追補 H.3、1108 修正）: --cut-kind mean は枝の出力を、学んだ課題 t_end の 2,500 枚（no grad、250 枚ずつ）での
    # 平均（token 位置ごとの定数ベクトル）に置き換える。平均は読み込んだままの網（切除なし）で全部の枝について先に取り、その後で hook を付ける。
    # zero は 0 を掛ける。既定は --init に従う（ckpt → mean、t00 → zero）。
    cut_kind = args.cut_kind or ('mean' if args.init == 'ckpt' else 'zero')
    means = None
    if cuts and cut_kind == 'mean':
        data_t, _ = C.task_dataset(source, seed, args.t_end, 10, 250, augment=False)
        x_t = C.eval_tensor(data_t.images, device)
        means = branch_means(model, cuts, x_t)
        del x_t
        note['mean_from_task'] = args.t_end
        note['mean_rms'] = {c: dict(overall=float(m.double().pow(2).mean().sqrt()),
                                    per_token=m.double().pow(2).mean(1).sqrt().tolist()) for c, m in means.items()}
    handles, frozen = apply_cuts(model, cuts, means)
    note['frozen_params'] = frozen
    if do_head0:
        reset_head(model)
    if do_adam0:
        optimizer = new_adam(model)
        note['adam'] = 'new'
    else:
        optimizer = C.build_optimizer(model)
        if adam is not None:
            optimizer.load_state_dict(adam['optimizer'])
            note['adam'] = 'loaded'
        else:
            note['adam'] = 'new (no saved state)'
    if C.zero_head_at(args.arm, task, 1) and not do_head0:
        reset_head(model)
    in_opt = {id(p) for g in optimizer.param_groups for p in g['params']}
    named = dict(model.named_parameters())
    note['n_opt_params'] = len(in_opt)
    note['n_trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checks = {}
    if args.self_check:
        checks['frozen_requires_grad_false'] = all(not named[n].requires_grad for n in frozen)
        checks['frozen_absent_from_adam'] = (not do_adam0) or all(id(named[n]) not in in_opt for n in frozen)
        checks['trainable_all_in_adam'] = (not do_adam0) or all(id(p) in in_opt for p in model.parameters() if p.requires_grad)
        checks['head_zero_before_update1'] = bool(head.weight.abs().max() == 0 and head.bias.abs().max() == 0) if do_head0 else None
        seen = {}

        def check_hook(name):
            def h(module, inputs, output):
                if means is None:
                    seen[name] = bool((output == 0).all())
                else:                             # 違う 2 枚（batch の 0 番と 7 番）の出力が両方とも保存した定数と一致し、定数が 0 でない
                    m = means[name]
                    seen[name] = dict(img0_equals_const=bool(torch.equal(output[0], m)), img7_equals_const=bool(torch.equal(output[7], m)),
                                      const_nonzero=bool(m.abs().max() > 0), const_rms=float(m.double().pow(2).mean().sqrt()))
            return h
        extra = [branch(model, *CUTS[c]).register_forward_hook(check_hook(c)) for c in cuts]   # 切除の hook の後に走る
        with torch.no_grad():
            model(x_eval[:16])
        for h_ in extra:
            h_.remove()
        checks['cut_outputs_exactly_zero' if means is None else 'cut_outputs_equal_mean_const'] = {c: seen.get(c) for c in cuts}
        frozen_before = {n: named[n].detach().clone() for n in frozen}
    with torch.no_grad():
        logits_after = torch.cat([model(x_eval[s:s + 128]) for s in range(0, 2500, 128)])
    note['max_logit_change'] = float((logits_after - logits_before).abs().max())
    note['head_norm'] = float(head.weight[rows_t].detach().norm())
    forward = train_forward(model, args.engine)
    if args.self_check and args.engine == 'compile':
        # head が 0 だと logit は全部 0 で比べても空なので、一時的に乱数の head を載せて compile と eager と hook 無しを比べる
        model.eval()
        hw, hb = head.weight.detach().clone(), head.bias.detach().clone()
        with torch.no_grad():
            g = torch.Generator(device='cpu').manual_seed(2027)
            head.weight.copy_(0.02 * torch.randn(head.weight.shape, generator=g).to(device))
            lc, le = forward(x_eval[:128]), model(x_eval[:128])
            for h_ in handles:
                h_.remove()
            lu = model(x_eval[:128])
            handles, _ = apply_cuts(model, cuts, means)
            lc2 = forward(x_eval[:128])          # hook を付け直した後も compile が hook を通すか（作り直し）
            head.weight.copy_(hw); head.bias.copy_(hb)
        checks['compiled_vs_eager_max_abs'] = float((lc - le).abs().max())
        checks['compiled_after_rehook_vs_eager_max_abs'] = float((lc2 - le).abs().max())
        checks['eager_without_hooks_vs_with_max_abs'] = float((lu - le).abs().max())   # 切除があれば 0 でないはず
        model.train()
    model.train()
    points = args.max_steps // C.EVAL_EVERY + 1
    rec = Recorder(model, x_eval, y_eval, rows_t, points, device)
    rec.take(0)
    track = args.track_steps
    trk = None
    if track:
        body_params = [p for p in model.parameters() if id(p) not in head_ids]
        prev_body = [p.detach().clone() for p in body_params]
        prev_hw, prev_hb = head.weight.detach().clone(), head.bias.detach().clone()
        trk = dict(step=[], loss=[], d_body=[], d_head=[], g_body=[], g_head=[], v_body=[], s_body=[], v_head=[], s_head=[])

        def group_of(p):
            return next(g for g in optimizer.param_groups if any(p is q for q in g['params']))

        @torch.no_grad()
        def adam_stats(params, sel):
            vs, ss, n = [], [], 0
            for p in params:
                st = optimizer.state.get(p)
                if not st or 'exp_avg_sq' not in st:
                    continue
                b1, b2 = group_of(p)['betas']
                t = st['step'].float() if torch.is_tensor(st['step']) else torch.tensor(float(st['step']))
                m, v = st['exp_avg'], st['exp_avg_sq']
                if sel is not None:
                    m, v = sel(m), sel(v)
                mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
                vs.append(v.sum()); ss.append((mh / (vh.sqrt() + 1e-8)).pow(2).sum()); n += v.numel()
            nan = torch.tensor(float('nan'), device=device)
            if n == 0:
                return nan, nan
            return (torch.stack(vs).sum() / n).sqrt(), (torch.stack(ss).sum() / n).sqrt()

        @torch.no_grad()
        def trk_take(step, loss_value):
            db = torch.stack([(p - q).pow(2).sum() for p, q in zip(body_params, prev_body)]).sum().sqrt()
            for p, q in zip(body_params, prev_body):
                q.copy_(p)
            dh = ((head.weight[rows_t] - prev_hw[rows_t]).pow(2).sum() + (head.bias[rows_t] - prev_hb[rows_t]).pow(2).sum()).sqrt()
            prev_hw.copy_(head.weight); prev_hb.copy_(head.bias)
            gb = torch.stack([p.grad.pow(2).sum() for p in body_params if p.grad is not None]).sum().sqrt()
            gh = (head.weight.grad[rows_t].pow(2).sum() + head.bias.grad[rows_t].pow(2).sum()).sqrt()
            vb, sb = adam_stats(body_params, None)
            vh, sh = adam_stats([head.weight, head.bias], lambda z: z[rows_t])
            vals = torch.stack([db, dh, gb, gh, vb, sb, vh, sh]).tolist()
            trk['step'].append(step); trk['loss'].append(loss_value)
            for k_, v_ in zip(('d_body', 'd_head', 'g_body', 'g_head', 'v_body', 's_body', 'v_head', 's_head'), vals):
                trk[k_].append(v_)
    t0 = time.monotonic()
    losses = []
    last = 0
    for step, (x, y) in enumerate(train_loader(data, seed, task, args.max_steps, 2), 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prepare_noise(model, len(y))
        loss = F.cross_entropy(forward(x).index_select(1, rows_t), y)
        loss.backward()
        if args.self_check and step == 1:
            checks['frozen_grad_none_step1'] = all(named[n].grad is None for n in frozen)
        optimizer.step()
        update_adaptive(model)
        losses.append(float(loss))
        if track and step <= track:
            trk_take(step, losses[-1])
        if step % C.EVAL_EVERY == 0:
            last = step // C.EVAL_EVERY
            rec.take(last)
    if device == 'cuda':
        torch.cuda.synchronize()
    if args.self_check:
        checks['frozen_unchanged_after_run'] = all(torch.equal(frozen_before[n], named[n].detach()) for n in frozen)
        note['self_check'] = checks
    r = rec.numpy()
    n = 2500
    corr = r['correct'][:last + 1]
    acc = (corr / n).tolist()
    steps_run = len(losses)
    status = 'ok' if steps_run == args.max_steps and last == points - 1 else 'short'
    k1000 = 1000 // C.EVAL_EVERY
    out = dict(arm=args.arm, t_end=args.t_end, task=task, mode=mode, init=args.init, seed=seed, cuts=cuts, cut_kind=cut_kind if cuts else None,
               engine=args.engine, device=device, note=note, status=status, steps_run=steps_run, max_steps=args.max_steps,
               eval_every=C.EVAL_EVERY, eval_steps=[k * C.EVAL_EVERY for k in range(last + 1)],
               T50=first_reach(corr, n, 50), T75=first_reach(corr, n, 75), T90=first_reach(corr, n, 90),
               A1000=acc[k1000] if len(acc) > k1000 else None,
               A_end=(sum(acc[-A_END_POINTS:]) / A_END_POINTS) if len(acc) > A_END_POINTS else None,
               spike=(sum(losses[:5]) / 5) if len(losses) >= 5 else None,
               mem_acc=acc, mem_loss=(r['loss'][:last + 1] / n).tolist(),
               loss_100=[sum(losses[i:i + 100]) / len(losses[i:i + 100]) for i in range(0, len(losses), 100)],
               seconds=round(time.monotonic() - t0, 1), track=trk)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(dest)
    print(f"{args.arm} seed{seed} {args.init} T{args.t_end}→{task} {mode} [{cut_kind if cuts else '-'}]: T90 {out['T90']} A1000 {out['A1000']} A_end {out['A_end']}"
          f" spike {out['spike']} ({status}, {steps_run} 更新, {out['seconds']} s) | 始めの暗記損失 {out['mem_loss'][0]:.3f}")
    if args.self_check:
        print('  self_check:', json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('arm'); p.add_argument('t_end', type=int); p.add_argument('mode'); p.add_argument('out')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--init', choices=('ckpt', 't00'), required=True)
    p.add_argument('--max-steps', type=int, default=5000)
    p.add_argument('--track-steps', type=int, default=100)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--engine', choices=('compile', 'eager'), default='compile')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--cut-kind', choices=('mean', 'zero'), default=None, help='切除の中身。既定は --init に従う（ckpt → mean、t00 → zero）')
    p.add_argument('--h', action='store_true', help='mode の前に head0+adam0+ を足す（追補 H.3）')
    p.add_argument('--self-check', action='store_true', help='切除・凍結・head 0・Adam の群・compile と eager の一致を確かめて note に書く')
    main(p.parse_args())

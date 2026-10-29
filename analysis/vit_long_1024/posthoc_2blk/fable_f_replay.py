"""追補 F の回し直し R1〜R3（spec 追補 F.4、走る前に書いた道具、1029）。

S の保存状態（model_tK + adam_tK、head は戻さない）から課題 K+1 を、本走と同じ決定的な batch 列で eager に回す。
  r1    最初の 100 更新を、対照（base）と FFN0 の共通成分の固定（fix_ffn0）で。fix_ffn0 は FFN0 の出力 f を各 forward で
        f - mean(f) + c に置き換える（文字 GPT の追補 8 = X2 と同じ形）。mean はその batch の全 token 平均、c はその入力の
        切替時の値: 既定（--c0-from batch）では、切替時の網の写しが同じ batch に出す FFN0 の token 平均。だから更新 1 の
        forward は対照と同じで、違いは学習の間 FFN0 の共通成分が動けないことだけになる。測定の forward（固定 probe）にも
        同じ置き換えを入れる。X2 の字義どおり probe の切替時の平均を全部に使うのは --c0-from probe、入力の組ごとの定数
        （学習と暗記率は 2,500 枚の平均）は --c0-from own。固定 probe（val 64 枚、200 クラス）と 2,500 枚（10 クラス）では
        FFN0 の平均の大きさが 12〜18% 違う（seed 10252010、向きの cos 0.9998）ので、この 2 つは更新 1 から対照と違う。
        量は最終 FFN の生きた unit（切替時に probe の token の 0.5% 以上で開く）の Δz̄（固定 probe、全 token）を、更新 25 と
        100 で自分の重み・入力・交差・bias の分に割った帳簿。減り = 1 - 固定 / 対照（J F3）。対照の更新 25 は J F2 の量でもある。
  r2r3  5,000 更新を、対照（base）、最終 FFN の入力の中心化（center、SC と同じ演算: S の状態を SC の網に載せる）、
        head の入力の定数倍（scale）で。scale の倍率は、2,500 枚での ‖h‖（CLS の head の入力）の中央値を、同 seed の fresh の
        課題末の中央値に合わせる。fresh は既定で 4 本（課題 25・50・75・100）の中央値の中央値、--ref-task で 1 本に。seed
        10252010 では fresh 154〜219 に対して S は t10 50・t30 64・t60 82・t100 827 なので、t10〜t60 では拡大になる。
        量は T90 と最終 FFN の死（固定 probe）。--full が無ければ T90 の 250 更新後で止める。
回し直しは eager で本走（compile）と軌跡が一致しない。比べるのは同じ機械の対照とだけ（spec 追補 F.4）。

    python -m analysis.vit_long_1024.posthoc_2blk.fable_f_replay r1 10252011 10 fix_ffn0
    python -m analysis.vit_long_1024.posthoc_2blk.fable_f_replay r2r3 10252011 30 center

出力は <raw>/posthoc_f/<kind>_S_seed<seed>_t<K>_<mode>.json（既にあれば断る。既定でない --c0-from・--ref-task・--steps は
名前の末尾に付く）。
"""
import argparse
import copy
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix_f.json')
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.vit_long_1024 import common as C

OUT = C.RAW / 'posthoc_f'
STEPS = {'r1': 100, 'r2r3': 5000}
MODES = {'r1': ('base', 'fix_ffn0'), 'r2r3': ('base', 'center', 'scale')}
PROBE_STEPS = {'r1': (25, 100), 'r2r3': tuple(range(250, 5001, 250))}
RSS_LIMIT_MIB = 3072


def ckpt(seed, task, kind='model', arm='S'):
    return C.RAW / 'runs' / arm / f'seed{seed}' / 'ckpt' / f'{kind}_t{task:02d}.pt'


def fresh_end(seed, task):
    return C.RAW / 'runs' / 'fresh' / f'seed{seed}' / f'task{task}' / 'ckpt' / f'model_t{task:02d}.pt'


def tensors(images, device):
    return torch.cat([C.eval_tensor(images[s:s + 32], device) for s in range(0, len(images), 32)])


def median(v):
    return float(torch.quantile(v.double(), .5)) if v.numel() else float('nan')


def resources():
    out = dict(rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               cuda_alloc_mib=torch.cuda.max_memory_allocated() / 2**20)
    if out['rss_mib'] > RSS_LIMIT_MIB:
        raise RuntimeError(f'RSS budget exceeded: {out}')
    return out


def run(model, x, c0=None):
    """VisionTransformer.forward step by step (as C.trace). Returns the head input, the last FFN's input u and
    pre-activation z, and FFN0's output as added to the residual; with c0 (a tensor, or a function of x), FFN0's
    output f becomes f - mean(f) + c0, the mean over the batch and tokens (r1 fix_ffn0)."""
    L = model.layers
    h = L['patch_embed'](x)
    h = L['pos_drop'](torch.cat((model.cls_token.expand(h.shape[0], -1, -1), h), dim=1) + model.pos_embed)
    for i in range(model.depth):
        b = L[f'block_{i}'].layers
        h = h + b['attn'](b['norm1'](h))
        u = b['norm2'](h)
        mlp = b['mlp'].layers
        z = mlp['fc1'](u)
        f = mlp['drop2'](mlp['fc2'](mlp['drop1'](mlp['act'](z))))
        if i == 0:
            if c0 is not None:
                f = f - f.mean(dim=(0, 1), keepdim=True) + (c0(x) if callable(c0) else c0)
            f0 = f
        h = h + f
    return dict(head=L['norm'](h)[:, 0], u=u, z=z, f0=f0)


@torch.no_grad()
def probe_state(model, probe, c0):
    """The last FFN on the fixed probe: input mean m over all tokens, fc1's rows w and bias b (the z-bar ledger),
    the open fraction p per unit, and the deaths as in C.layer_summary."""
    out = run(model, probe, c0)
    z = out['z'].double()
    p = (z > 0).double().mean((0, 1))
    pc = (z[:, 0] > 0).double().mean(0)
    fc = model.layers[f'block_{model.depth - 1}'].layers['mlp'].layers['fc1']
    return dict(m=out['u'].double().mean((0, 1)), w=fc.weight.detach().double().clone(),
                b=fc.bias.detach().double().clone(), p=p, dead005=float((p < .005).double().mean()),
                dead0=float((p == 0).double().mean()), cls_dead0=float((pc == 0).double().mean()),
                zbar_median=median(z.mean((0, 1))), f0_mean=out['f0'].double().mean((0, 1)))


def ledger(s0, s1):
    """z-bar of the units alive at s0 (open on >= 0.5% of probe tokens), split exactly into the own-weight, input,
    cross and bias parts: dz = dw.m0 + w0.dm + dw.dm + db."""
    dw, dm = s1['w'] - s0['w'], s1['m'] - s0['m']
    parts = dict(weight=dw @ s0['m'], input=s0['w'] @ dm, cross=dw @ dm, bias=s1['b'] - s0['b'])
    total = s1['w'] @ s1['m'] + s1['b'] - s0['w'] @ s0['m'] - s0['b']
    alive = s0['p'] >= .005
    out = dict(alive=int(alive.sum()), total_median=median(total[alive]),
               sink_frac=float((total[alive] < 0).double().mean()),
               identity_error=float((total - sum(parts.values())).abs().max()))
    out.update({f'{k}_median': median(v[alive]) for k, v in parts.items()})
    out['input_share'] = out['input_median'] / out['total_median'] if out['total_median'] else float('nan')
    return out


@torch.no_grad()
def head_norm_median(model, x):
    return median(torch.cat([run(model, x[s:s + C.BATCH])['head'] for s in range(0, len(x), C.BATCH)]).norm(dim=1))


@torch.no_grad()
def ffn0_mean(model, x):
    """FFN0's output mean over the images and tokens of x (plain forward), shaped to replace mean(f)."""
    total = sum(run(model, x[s:s + C.BATCH])['f0'].double().sum((0, 1)) for s in range(0, len(x), C.BATCH))
    return (total / (len(x) * model.pos_embed.shape[1])).float()[None, None]


def main(args):
    steps = args.steps or STEPS[args.kind]
    assert args.mode in MODES[args.kind] and 0 < steps <= STEPS[args.kind] and steps % C.EVAL_EVERY == 0
    C.set_determinism(4)
    torch.cuda.set_per_process_memory_fraction(.45)
    out_dir = Path(args.out) if args.out else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f'{args.kind}_S_seed{args.seed}_t{args.task}_{args.mode}'
           + (f'_c0{args.c0_from}' if args.mode == 'fix_ffn0' and args.c0_from != 'batch' else '')
           + (f'_ref{args.ref_task}' if args.mode == 'scale' and args.ref_task else '')
           + (f'_u{steps}' if steps < STEPS[args.kind] else ''))
    dest = out_dir / f'{tag}.json'
    if dest.exists():
        raise FileExistsError(dest)
    device, seed, task = 'cuda', args.seed, args.task
    model = C.build_arm_model('SC' if args.mode == 'center' else 'S', seed, device, 0.0, 2, 8.0).eval()
    model.load_state_dict(torch.load(ckpt(seed, task), map_location=device, weights_only=True))
    opt = C.build_optimizer(model)
    opt.load_state_dict(torch.load(ckpt(seed, task, 'adam'), map_location='cpu', weights_only=False)['optimizer'])
    head = model.layers['out']
    source = C.image_source('real')
    data, _ = C.task_dataset(source, seed, 1, 10, 250, augment=False)       # the fixed 2,500 images (追補 D.1)
    x = tensors(data.images, device)
    probe = tensors(source.val(C.probe_names(source, seed)), device)
    y = C.task_labels(len(x), seed, task + 1, 10).to(device)
    rows = torch.tensor(C.loss_classes('S', seed, task + 1, 10, 250), device=device)
    with torch.no_grad():
        plain_equal = bool(torch.equal(F.linear(run(model, x[:C.BATCH])['head'], head.weight, head.bias),
                                       model(x[:C.BATCH])))
    c0 = c0_train = c0_info = None              # c0 for the probe measurements, c0_train for training and evaluation
    if args.mode == 'fix_ffn0':
        c0 = ffn0_mean(model, probe)
        c0_train = c0 if args.c0_from == 'probe' else ffn0_mean(model, x)
        if args.c0_from == 'batch':
            frozen = copy.deepcopy(model).requires_grad_(False)
            c0_train = lambda xb: run(frozen, xb)['f0'].detach().mean(dim=(0, 1), keepdim=True)
        own = ffn0_mean(model, x).flatten().double()
        c0_info = dict(c0_from=args.c0_from, probe_norm=float(c0.norm()), train_norm=float(own.norm()),
                       cos=float(own @ c0.flatten().double() / own.norm() / c0.norm()))
    scale, scale_ref = 1., None
    if args.mode == 'scale':
        ref = C.build_arm_model('fresh', seed, device, 0.0, 2, 8.0).eval()
        fresh_h = {}
        for t in ([args.ref_task] if args.ref_task else C.production()['fresh_tasks']):
            ref.load_state_dict(torch.load(fresh_end(seed, t), map_location=device, weights_only=True))
            fresh_h[t] = head_norm_median(ref, x)
        scale_ref = dict(fresh_h_median=fresh_h, target=median(torch.tensor(list(fresh_h.values()))),
                         s_h_median=head_norm_median(model, x))
        scale = scale_ref['target'] / scale_ref['s_h_median']
        del ref

    def logits_of(xb):
        return F.linear(run(model, xb, c0_train)['head'] * scale, head.weight, head.bias).index_select(1, rows)

    @torch.no_grad()
    def evaluate(step):
        correct, loss = 0, 0.
        for s in range(0, len(x), C.BATCH):                    # the recorder's chunks of 128 (run.Recorder)
            lg = logits_of(x[s:s + C.BATCH])
            correct += int((lg.argmax(1) == y[s:s + C.BATCH]).sum())
            loss += float(F.cross_entropy(lg, y[s:s + C.BATCH], reduction='sum'))
        records.append(dict(step=step, correct=correct, acc=correct / len(x), loss=loss / len(x)))
        return correct

    def take_probe(step):
        s = probe_state(model, probe, c0)
        row = dict(step=step, dead005=s['dead005'], dead0=s['dead0'], cls_dead0=s['cls_dead0'],
                   zbar_median=s['zbar_median'], m_norm=float(s['m'].norm()))
        if c0 is not None:
            row['f0_mean_dev'] = float((s['f0_mean'] - c0.flatten().double()).abs().max())
        if step:
            row['ledger'] = ledger(s_start, s)
        probes.append(row)

    records, probes, losses, first90, t0 = [], [], [], None, time.monotonic()
    s_start = probe_state(model, probe, c0)
    take_probe(0)
    evaluate(0)
    last = 0
    for step, batch in enumerate(PairedBatches(len(x), C.BATCH, steps, seed, task + 1), 1):
        ix = torch.tensor([i for i, _ in batch], device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits_of(x[ix]), y[ix])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'{tag}: nonfinite loss at update {step}')
        loss.backward()
        opt.step()
        losses.append(float(loss))
        last = step
        if step in PROBE_STEPS[args.kind]:
            take_probe(step)
        if step % C.EVAL_EVERY == 0:
            if evaluate(step) * 100 >= 90 * len(x) and first90 is None:
                first90 = step
                if args.kind == 'r2r3' and step not in PROBE_STEPS[args.kind]:
                    take_probe(step)
            if step % 250 == 0 or step == steps:
                print(f'{tag} step{step} acc={records[-1]["acc"]:.4f} loss={records[-1]["loss"]:.4g} '
                      f'dead={probes[-1]["dead005"]:.3f} sec={time.monotonic() - t0:.1f}', flush=True)
            if args.kind == 'r2r3' and first90 is not None and step >= first90 + 250 and not args.full:
                break
    if probes[-1]['step'] != last:
        take_probe(last)
    head_state, dirty = C.git_state()
    result = dict(tag=tag, kind=args.kind, mode=args.mode, arm='S', seed=seed, task_end=task, next_task=task + 1,
                  steps=last, budget=steps, full=args.full, T90=first90 if first90 is not None else steps + C.EVAL_EVERY,
                  reached90=first90 is not None, feature_scale=scale, scale_ref=scale_ref, c0=c0_info,
                  plain_forward_equals_model=plain_equal,
                  start=dict(model=str(ckpt(seed, task)), adam=str(ckpt(seed, task, 'adam'))),
                  records=records, probes=probes, train_loss=losses, git_hash=head_state, dirty=dirty,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), seconds=time.monotonic() - t0,
                  resources=resources())
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f'{tag} done: updates {last}, T90 {result["T90"]}, dead {probes[0]["dead005"]:.3f} -> '
          f'{probes[-1]["dead005"]:.3f}, {dest}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=sorted(STEPS))
    p.add_argument('seed', type=int)
    p.add_argument('task', type=int, help='the saved task end K (S: model_tK and adam_tK); replays task K+1')
    p.add_argument('mode', choices=sorted({m for v in MODES.values() for m in v}))
    p.add_argument('--steps', type=int, default=0, help='fewer updates than the budget (smoke tests)')
    p.add_argument('--full', action='store_true', help='r2r3: run all 5,000 updates instead of stopping 250 after T90')
    p.add_argument('--ref-task', type=int, default=0,
                   help="scale: one fresh run's task-end ‖h‖ as the target (default 0: the median of the seed's fresh runs)")
    p.add_argument('--c0-from', choices=('batch', 'own', 'probe'), default='batch',
                   help="fix_ffn0: every batch keeps its own switch-time mean from a frozen copy of the switch-time net "
                        "(batch), each input set one constant (own), or the probe's constant for all (probe, X2's form)")
    p.add_argument('--out', default='', help='output folder (default <raw>/posthoc_f)')
    main(p.parse_args())

"""追補 F の R5（事後に足した回し直し、Fable、1104）: S の保存状態から head だけ 0 に戻して次の課題を回し、遅れを head の側と体の側に割る。

J F5 の副（R2 中心化・R3 尺度）では S の遅れの担い手が決まらなかった（seed 10 の t30: 中心化で 4,250 → 5,025、尺度 ×3.1 で 4,175）。
LN あり（SL）の追補 D の事後解析では、同じ課題末の状態で head を 0 に戻すと T90 が新品に並び、遅れは全部 head の装填費用だった。
同じ分解を S で行う。fable_f_replay の r2r3 の対照（base）と同じ保存状態（model_tK + adam_tK）・同じ batch 列・同じ eager の学習で、
head の 200 行と bias を 0 に戻し（Adam の状態は本走の R0 と同じく残す。scale_counterfactual.py の head0 と同じ約束）、T90 と最終 FFN の死を測る。
比べるのは同じ機械の r2r3 の対照とだけ。走る前の予測（Fable）: t30 で head 0 の T90 が対照の半分以下になる seed が 4/5（0.65）、t60 で 4/5（0.5）。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix_f.json \
        python -m analysis.vit_long_1024.posthoc_2blk.fable_f_head0 10252010 30 [--steps N] [--full] [--out DIR]

出力 <raw>/posthoc_f/r5_S_seed<seed>_t<K>_head0.json（既にあれば断る）。

追加（1104、R5 の結果を見てから）: --arm SC で SC の保存状態（SC の網、対照は --keep-head の同じ道具）、--freeze-body で体を凍結して
head だけ学ぶ（R4 の新 head の分岐と同じ形。t60 の S で head 0 が届かないのが体の表現の喪失か、動く体との結合した更新の不安定かを分ける）。
走る前の予測（Fable）: SC は t30・t60 とも head 0 の T90 が同 seed の freshC の課題 50 の 1.3 倍以内（4/5、0.7）。S の t60 で体を凍結した
新 head は 500 更新以内に届く（4/5、0.7）。
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix_f.json')
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.joudaki_vit_battle_0919.model import reset_head
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk import fable_f_replay as FR

STEPS = 5000
PROBE_STEPS = tuple(range(250, 5001, 250))


def main(args):
    steps = args.steps or STEPS
    assert 0 < steps <= STEPS and steps % C.EVAL_EVERY == 0
    C.set_determinism(4)
    torch.cuda.set_per_process_memory_fraction(.45)
    out_dir = Path(args.out) if args.out else FR.OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = (('base' if args.keep_head else 'head0') + ('_frozen' if args.freeze_body else '')
            + ('_adamreset' if args.reset_adam else '') + ('_hcenter' if args.center_head_input else ''))
    tag = f'r5_{args.arm}_seed{args.seed}_t{args.task}_{mode}' + (f'_u{steps}' if steps < STEPS else '')
    dest = out_dir / f'{tag}.json'
    if dest.exists():
        raise FileExistsError(dest)
    device, seed, task = 'cuda', args.seed, args.task
    model = C.build_arm_model(args.arm, seed, device, 0.0, 2, 8.0).eval()
    model.load_state_dict(torch.load(FR.ckpt(seed, task, arm=args.arm), map_location=device, weights_only=True))
    opt = C.build_optimizer(model)
    if not args.reset_adam:                             # R5d（1104 事後）: --reset-adam は保存した Adam の状態を読まず新品の Adam で回す
        opt.load_state_dict(torch.load(FR.ckpt(seed, task, 'adam', arm=args.arm), map_location='cpu', weights_only=False)['optimizer'])
    head = model.layers['out']
    head_norm_before = float(head.weight.detach().norm())
    if not args.keep_head:
        reset_head(model)                               # head の重みと bias を 0 に。Adam の状態は残す（R0 の手順）
    if args.freeze_body:                                # 体の requires_grad を切る（grad が None の param は Adam が飛ばす）
        for name, prm in model.named_parameters():
            if not name.startswith('layers.out.'):
                prm.requires_grad_(False)
    source = C.image_source('real')
    data, _ = C.task_dataset(source, seed, 1, 10, 250, augment=False)
    x = FR.tensors(data.images, device)
    probe = FR.tensors(source.val(C.probe_names(source, seed)), device)
    y = C.task_labels(len(x), seed, task + 1, 10).to(device)
    rows = torch.tensor(C.loss_classes(args.arm, seed, task + 1, 10, 250), device=device)

    h_shift = None                                      # R5e（1104 事後）: --center-head-input は head の入力から、切替時の全画像平均 m̂ を固定で引く（R1 の固定と同じ約束）
    head_input_stats = None
    if args.center_head_input:
        with torch.no_grad():
            hs = torch.cat([FR.run(model, x[s:s + C.BATCH])['head'] for s in range(0, len(x), C.BATCH)]).double()
            h_shift = hs.mean(0).float()
            head_input_stats = dict(mean_norm=float(hs.mean(0).norm()), median_norm=float(FR.median(hs.norm(dim=1))),
                                    median_centered_norm=float(FR.median((hs - hs.mean(0)).norm(dim=1))))
            del hs

    def logits_of(xb):
        hb = FR.run(model, xb)['head']
        if h_shift is not None:
            hb = hb - h_shift
        return F.linear(hb, head.weight, head.bias).index_select(1, rows)

    @torch.no_grad()
    def evaluate(step):
        correct, loss = 0, 0.
        for s in range(0, len(x), C.BATCH):
            lg = logits_of(x[s:s + C.BATCH])
            correct += int((lg.argmax(1) == y[s:s + C.BATCH]).sum())
            loss += float(F.cross_entropy(lg, y[s:s + C.BATCH], reduction='sum'))
        records.append(dict(step=step, correct=correct, acc=correct / len(x), loss=loss / len(x)))
        return correct

    def take_probe(step):
        s = FR.probe_state(model, probe, None)
        row = dict(step=step, dead005=s['dead005'], dead0=s['dead0'], cls_dead0=s['cls_dead0'], zbar_median=s['zbar_median'],
                   m_norm=float(s['m'].norm()), head_rows_norm=float(head.weight[rows].norm()))
        if step:
            row['ledger'] = FR.ledger(s_start, s)
        probes.append(row)

    records, probes, losses, first90, t0 = [], [], [], None, time.monotonic()
    s_start = FR.probe_state(model, probe, None)
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
        losses.append(float(loss.detach()))
        last = step
        if step in PROBE_STEPS:
            take_probe(step)
        if step % C.EVAL_EVERY == 0:
            if evaluate(step) * 100 >= 90 * len(x) and first90 is None:
                first90 = step
                if step not in PROBE_STEPS:
                    take_probe(step)
            if step % 250 == 0 or step == steps:
                print(f'{tag} step{step} acc={records[-1]["acc"]:.4f} loss={records[-1]["loss"]:.4g} dead={probes[-1]["dead005"]:.3f} '
                      f'sec={time.monotonic() - t0:.1f}', flush=True)
            if first90 is not None and step >= first90 + 250 and not args.full:
                break
    if probes[-1]['step'] != last:
        take_probe(last)
    head_state, dirty = C.git_state()
    result = dict(tag=tag, kind='r5', mode=mode, arm=args.arm, seed=seed, task_end=task, next_task=task + 1, steps=last, budget=steps,
                  full=args.full, T90=first90 if first90 is not None else steps + C.EVAL_EVERY, reached90=first90 is not None,
                  head_norm_before=head_norm_before, head_adam_state_kept=not args.reset_adam, keep_head=args.keep_head, freeze_body=args.freeze_body,
                  reset_adam=args.reset_adam, center_head_input=args.center_head_input, head_input_stats=head_input_stats,
                  start=dict(model=str(FR.ckpt(seed, task)), adam=str(FR.ckpt(seed, task, 'adam'))),
                  records=records, probes=probes, train_loss=losses, git_hash=head_state, dirty=dirty, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), seconds=time.monotonic() - t0, resources=FR.resources())
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f'{tag} done: updates {last}, T90 {result["T90"]}, dead {probes[0]["dead005"]:.3f} -> {probes[-1]["dead005"]:.3f}, {dest}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('seed', type=int)
    p.add_argument('task', type=int)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--full', action='store_true')
    p.add_argument('--out', default='')
    p.add_argument('--arm', default='S', choices=('S', 'SC'))
    p.add_argument('--keep-head', action='store_true', help='head を戻さない（その腕の対照）')
    p.add_argument('--freeze-body', action='store_true', help='体を凍結して head だけ学ぶ')
    p.add_argument('--reset-adam', action='store_true', help='R5d: 保存した Adam の状態を読まず新品の Adam で回す')
    p.add_argument('--center-head-input', action='store_true', help='R5e: head の入力から切替時の全画像平均を固定で引く')
    main(p.parse_args())

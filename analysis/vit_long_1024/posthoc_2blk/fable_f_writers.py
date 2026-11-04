"""追補 F の R1 の補い（事後の測定、判定には使わない。Fable、1104）: 切替の直後に最終 FFN の入力の共通成分 m を動かす書き手を分ける。

spec F.6 は「PF3a が低いのは、S の切替の沈みのうち FFN0 の分と注意 1 の分の割合を切替窓で測っていないため」と記していた。R1 の固定
（fix_ffn0）は損失を下げられず沈みを増やしたので、介入の代わりに対照の軌道で書き手を分ける。fable_f_replay の r1 と同じ保存状態・同じ
batch 列・同じ eager の学習で 100 更新を回し（既定は対照。fix_ffn0 も選べる）、更新 0・1・2・5・10・25・50・100 で固定 probe（全 token）の
  u₂ = e + a₀ + f₀ + a₁（埋め込み＋cls＋位置、注意 0、FFN0、注意 1。S は norm が Identity なので u₂ が最終 FFN の入力）
の各部分の token 平均 m_part を取り、切替時からの変化 Δm_part と、生きた unit の自分の行 w₀ を通した入力の分 w₀·Δm_part の中央値を出す
（Σ_part w₀·Δm_part = w₀·Δm は厳密）。あわせて m の各部分への分解（⟨m_part, m⟩/‖m‖、和が ‖m‖）と損失を記す。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix_f.json \
        python -m analysis.vit_long_1024.posthoc_2blk.fable_f_writers 10252010 10 [base|fix_ffn0] [--out DIR]

出力 <raw>/posthoc_f/writers_S_seed<seed>_t<K>_<mode>.json（既にあれば断る）。
"""
import argparse
import copy
import json
import os
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix_f.json')
import torch
from torch.nn import functional as F

from analysis.joudaki_vit_battle_0919.data import PairedBatches
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk import fable_f_replay as FR

STEPS = 100
PROBE_STEPS = (1, 2, 5, 10, 25, 50, 100)
PARTS = ('e', 'a0', 'f0', 'a1')


@torch.no_grad()
def writers(model, x, c0=None):
    """The last FFN's input u₂ split into its writers on x (eval mode). Returns the token means (over images and tokens)
    of e, a₀, f₀, a₁ and u₂, and the last FFN's pre-activations z (all tokens)."""
    L = model.layers
    h = L['patch_embed'](x)
    e = L['pos_drop'](torch.cat((model.cls_token.expand(h.shape[0], -1, -1), h), dim=1) + model.pos_embed)
    b0 = L['block_0'].layers
    a0 = b0['attn'](b0['norm1'](e))
    h1 = e + a0
    m0 = b0['mlp'].layers
    f0 = m0['drop2'](m0['fc2'](m0['drop1'](m0['act'](m0['fc1'](b0['norm2'](h1))))))
    if c0 is not None:
        f0 = f0 - f0.mean(dim=(0, 1), keepdim=True) + (c0(x) if callable(c0) else c0)
    h1 = h1 + f0
    b1 = L['block_1'].layers
    a1 = b1['attn'](b1['norm1'](h1))
    u = b1['norm2'](h1 + a1)
    z = b1['mlp'].layers['fc1'](u)
    means = {k: v.double().mean((0, 1)) for k, v in dict(e=e, a0=a0, f0=f0, a1=a1).items()}
    return means, u.double().mean((0, 1)), z.double()


def main(args):
    C.set_determinism(4)
    torch.cuda.set_per_process_memory_fraction(.45)
    out_dir = FR.OUT if not args.out else __import__('pathlib').Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'writers_S_seed{args.seed}_t{args.task}_{args.mode}'
    dest = out_dir / f'{tag}.json'
    if dest.exists():
        raise FileExistsError(dest)
    device, seed, task = 'cuda', args.seed, args.task
    model = C.build_arm_model('S', seed, device, 0.0, 2, 8.0).eval()
    model.load_state_dict(torch.load(FR.ckpt(seed, task), map_location=device, weights_only=True))
    opt = C.build_optimizer(model)
    opt.load_state_dict(torch.load(FR.ckpt(seed, task, 'adam'), map_location='cpu', weights_only=False)['optimizer'])
    head = model.layers['out']
    source = C.image_source('real')
    data, _ = C.task_dataset(source, seed, 1, 10, 250, augment=False)
    x = FR.tensors(data.images, device)
    probe = FR.tensors(source.val(C.probe_names(source, seed)), device)
    y = C.task_labels(len(x), seed, task + 1, 10).to(device)
    rows = torch.tensor(C.loss_classes('S', seed, task + 1, 10, 250), device=device)
    c0 = c0_train = None
    if args.mode == 'fix_ffn0':                       # fable_f_replay の --c0-from batch と同じ
        c0 = FR.ffn0_mean(model, probe)
        frozen = copy.deepcopy(model).requires_grad_(False)
        c0_train = lambda xb: FR.run(frozen, xb)['f0'].detach().mean(dim=(0, 1), keepdim=True)
    fc = model.layers[f'block_{model.depth - 1}'].layers['mlp'].layers['fc1']

    def logits_of(xb):
        return F.linear(FR.run(model, xb, c0_train)['head'], head.weight, head.bias).index_select(1, rows)

    @torch.no_grad()
    def evaluate():
        loss, correct = 0., 0
        for s in range(0, len(x), C.BATCH):
            lg = logits_of(x[s:s + C.BATCH])
            loss += float(F.cross_entropy(lg, y[s:s + C.BATCH], reduction='sum'))
            correct += int((lg.argmax(1) == y[s:s + C.BATCH]).sum())
        return loss / len(x), correct / len(x)

    def take(step):
        means, m, z = writers(model, probe, c0)
        u_check = FR.run(model, probe, c0)['u'].double().mean((0, 1))
        row = dict(step=step, m_norm=float(m.norm()), m_check=float((m - u_check).abs().max()),
                   sum_check=float((sum(means.values()) - m).abs().max()),
                   parts_norm={k: float(v.norm()) for k, v in means.items()},
                   parts_proj={k: float(v @ m / m.norm()) for k, v in means.items()},
                   dead005=float(((z > 0).double().mean((0, 1)) < .005).double().mean()),
                   zbar_median=FR.median(z.mean((0, 1))))
        if step:
            dm = {k: means[k] - start['means'][k] for k in PARTS}
            dm_total = m - start['m']
            w0 = start['w']
            alive = start['alive']
            contrib = {k: w0 @ dm[k] for k in PARTS}
            total = w0 @ dm_total
            row.update(dm_norm={k: float(v.norm()) for k, v in dm.items()}, dm_total_norm=float(dm_total.norm()),
                       dm_cos={k: float(dm[k] @ dm_total / (dm[k].norm() * dm_total.norm() + 1e-30)) for k in PARTS},
                       input_median={k: FR.median(contrib[k][alive]) for k in PARTS},
                       input_total_median=FR.median(total[alive]),
                       input_identity_error=float((sum(contrib.values()) - total).abs().max()),
                       sink_frac_input=float((total[alive] < 0).double().mean()),
                       share_of_input={k: (FR.median(contrib[k][alive]) / FR.median(total[alive]) if FR.median(total[alive]) else float('nan')) for k in PARTS})
        loss, acc = evaluate()
        row.update(loss=loss, acc=acc)
        probes.append(row)
        print(f"{tag} step{step} loss={loss:.3g} ‖m‖={row['m_norm']:.1f} dead={row['dead005']:.3f} zbar={row['zbar_median']:.2f}"
              + (f" | Δinput {row['input_total_median']:+.3f} 内訳 " + ' '.join(f"{k} {row['input_median'][k]:+.3f}" for k in PARTS)
                 + ' | ‖Δm‖ ' + ' '.join(f"{k} {row['dm_norm'][k]:.1f}" for k in PARTS) if step else
                 ' | m の分解 ' + ' '.join(f"{k} {row['parts_proj'][k]:+.1f}" for k in PARTS)), flush=True)

    probes, t0 = [], time.monotonic()
    with torch.no_grad():
        means0, m0, z0 = writers(model, probe, c0)
        start = dict(means=means0, m=m0, w=fc.weight.detach().double().clone(), alive=(z0 > 0).double().mean((0, 1)) >= .005)
    take(0)
    losses = []
    for step, batch in enumerate(PairedBatches(len(x), C.BATCH, STEPS, seed, task + 1), 1):
        ix = torch.tensor([i for i, _ in batch], device=device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits_of(x[ix]), y[ix])
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step in PROBE_STEPS:
            take(step)
    head_state, dirty = C.git_state()
    dest.write_text(json.dumps(dict(tag=tag, seed=seed, task_end=task, next_task=task + 1, mode=args.mode, steps=STEPS,
                                    alive=int(start['alive'].sum()), probes=probes, train_loss=losses, git_hash=head_state, dirty=dirty,
                                    seconds=time.monotonic() - t0, resources=FR.resources()), ensure_ascii=False, indent=1))
    print(f'{tag} done {time.monotonic() - t0:.0f}s {dest}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('seed', type=int)
    p.add_argument('task', type=int)
    p.add_argument('mode', nargs='?', default='base', choices=('base', 'fix_ffn0'))
    p.add_argument('--out', default='')
    main(p.parse_args())

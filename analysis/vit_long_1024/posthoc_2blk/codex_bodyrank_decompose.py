"""Read-only checkpoint diagnostics. No training, full-gradient storage, or git calls.

Only this task's new output directory is written. Each image's gradients are freed.
Run with nice -n 10, PYTHONDONTWRITEBYTECODE=1, and production_2blk_rlfix.json.
"""
import argparse
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk.tangent_rank import cls_features

SEED = 10252010
ROOT = Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix')
OUT = ROOT / 'posthoc_codex_bodyrank'


def group(name):
    if name in ('cls_token', 'pos_embed'):
        return name
    if name.startswith('layers.patch_embed'):
        return 'patch'
    if name.startswith('layers.norm'):
        return 'final_norm'
    z = name.split('.')
    return z[1] + '.' + z[3] + ('.' + z[5] if z[3] in ('attn', 'mlp') else '')


def forward_nodes(model, x):
    L = model.layers
    z = L['patch_embed'](x)
    z = torch.cat((model.cls_token.expand(len(x), -1, -1), z), 1) + model.pos_embed
    nodes = {'input': z}
    aux = {}
    for i in range(model.depth):
        b = L[f'block_{i}'].layers
        pre = f'b{i}'
        nodes[pre + '.n1x'] = z
        a = b['norm1'](z)
        nodes[pre + '.n1y'] = a
        att = b['attn']
        B, N, D = a.shape
        qkv = att.layers['qkv'](a).reshape(B, N, 3, att.n_heads, D // att.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        score = (q @ k.transpose(-2, -1)) * att.scale
        A = score.softmax(-1)
        av = (A @ v).transpose(1, 2).reshape(B, N, D)
        ao = att.layers['proj'](av)
        nodes.update({pre + '.score': score, pre + '.A': A, pre + '.av': av, pre + '.ao': ao})
        z = z + ao
        nodes[pre + '.n2x'] = z
        a = b['norm2'](z)
        nodes[pre + '.n2y'] = a
        mlp = b['mlp'].layers
        f = mlp['fc1'](a)
        act = mlp['act'](f)
        mo = mlp['fc2'](act)
        nodes.update({pre + '.f': f, pre + '.act': act, pre + '.mo': mo})
        z = z + mo
        nodes[pre + '.out'] = z
    nodes['final.x'] = z
    h = L['norm'](z)[:, 0]
    nodes['final.h'] = h
    return h, nodes, aux


def sq(x):
    return x.square().sum()


def norms_metrics(model, nodes, grads, u):
    """All outputs are tensors until one batched host transfer per direction."""
    vals = {}
    for i in range(model.depth):
        pre = f'b{i}'
        b = model.layers[f'block_{i}'].layers
        for j in (1, 2):
            key = pre + f'.n{j}'
            x, r = nodes[key + 'x'], grads[key + 'y']
            norm = b[f'norm{j}']
            xc = x - x.mean(-1, keepdim=True)
            sigma = (xc.square().mean(-1, keepdim=True) + norm.eps).sqrt()
            q = xc / sigma
            gr = r * norm.weight
            direct = gr / sigma
            back = (gr - gr.mean(-1, keepdim=True) - q * (gr * q).mean(-1, keepdim=True)) / sigma
            vals[key + '.sigma_cls'] = sigma[0, 0, 0]
            vals[key + '.sigma_token_med'] = sigma[0, 1:, 0].median()
            vals[key + '.sigma_token_min'] = sigma[0, 1:, 0].min()
            vals[key + '.inv_sigma_weighted'] = (sq(direct) / sq(gr).clamp_min(1e-30)).sqrt()
            vals[key + '.projection_survival'] = (sq(back) / sq(direct).clamp_min(1e-30)).sqrt()
            vals[key + '.back_norm'] = sq(back).sqrt()
            residual = grads[pre + ('.n2x' if j == 1 else '.out')]
            actual = grads[key + 'x']
            vals[key + '.residual_gain'] = (sq(actual) / sq(residual).clamp_min(1e-30)).sqrt()
            vals[key + '.branch_gain'] = (sq(back) / sq(residual).clamp_min(1e-30)).sqrt()
            vals[key + '.branch_cos'] = (back * residual).sum() / (sq(back) * sq(residual)).sqrt().clamp_min(1e-30)
            vals[key + '.identity_error'] = (sq(actual - residual - back) / sq(actual).clamp_min(1e-30)).sqrt()
        A = nodes[pre + '.A'][0]
        ac = A[:, 0]
        vals[pre + '.attn.cls_entropy'] = -(ac * ac.clamp_min(1e-30).log()).sum(-1).mean()
        vals[pre + '.attn.cls_max'] = ac.max(-1).values.mean()
        vals[pre + '.attn.cls_self'] = ac[:, 0].mean()
        # Frobenius norm squared of diag(a)-aa^T, averaged across CLS heads.
        s2, s3 = ac.square().sum(-1), ac.pow(3).sum(-1)
        vals[pre + '.attn.softmax_frob'] = (s2 - 2 * s3 + s2.square()).mean().sqrt()
        vals[pre + '.attn.softmax_backward_gain'] = (sq(grads[pre + '.score']) / sq(grads[pre + '.A']).clamp_min(1e-30)).sqrt()
        f, act, r = nodes[pre + '.f'], nodes[pre + '.act'], grads[pre + '.mo']
        bpre = grads[pre + '.act']
        gate = (f > 0).float()
        vals[pre + '.mlp.open_cls'] = gate[0, 0].mean()
        vals[pre + '.mlp.open_tokens'] = gate[0, 1:].mean()
        vals[pre + '.mlp.weighted_gate'] = (bpre.square() * gate).sum() / sq(bpre).clamp_min(1e-30)
        vals[pre + '.mlp.weighted_gate_cls'] = (bpre[0, 0].square() * gate[0, 0]).sum() / sq(bpre[0, 0]).clamp_min(1e-30)
        vals[pre + '.mlp.act_cls_norm'] = act[0, 0].norm()
        vals[pre + '.mlp.fc2_transpose_gain'] = (sq(bpre) / sq(r).clamp_min(1e-30)).sqrt()
        vals[pre + '.mlp.r_norm'] = sq(r).sqrt()
    x = nodes['final.x'][0, 0]
    norm = model.layers['norm']
    sigma = ((x - x.mean()).square().mean() + norm.eps).sqrt()
    q = (x - x.mean()) / sigma
    v = norm.weight * u
    pv = v - v.mean() - q * (v * q).mean()
    vals['final.sigma'] = sigma
    vals['final.angular'] = pv.norm()
    vals['final.projection_survival'] = pv.norm() / v.norm()
    vals['final.delta_norm'] = grads['final.x'].norm()
    vals['final.identity_error'] = (grads['final.x'][0, 0] - pv / sigma).norm() / grads['final.x'].norm().clamp_min(1e-30)
    vals['feature.proj'] = (nodes['final.h'][0] * u).sum()
    return vals


def run(arm, t, n, prefix, checkpoint=None, old_task=None):
    C.set_determinism(2)
    torch.cuda.set_per_process_memory_fraction(0.15)
    model = C.build_arm_model(arm, SEED, 'cuda', 0., 2, 8.)
    path = checkpoint or ROOT / 'runs' / arm / f'seed{SEED}/ckpt/model_t{t:02d}.pt'
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    model.eval()
    data, _ = C.task_dataset(C.image_source('real'), SEED, old_task or max(1, t), 10, 250, augment=False)
    # Convert only one small image batch at a time; no DataLoader workers.
    with torch.no_grad():
        H = torch.cat([cls_features(model, C.eval_tensor(data.images[s:s+32], 'cuda')) for s in range(0, 2500, 32)])
        Hc = H - H.mean(0)
        _, sv, vh = torch.linalg.svd(Hc, full_matrices=False)
        rng = torch.Generator().manual_seed(0)
        idx = torch.randperm(2500, generator=rng)[:256][:n]
        rand = torch.randn(4, 384, generator=rng).to('cuda')
        rand = rand / rand.norm(dim=1, keepdim=True)
        U = torch.cat((rand, vh[:4]))
        old = data.labels.to('cuda')
        centers = torch.stack([H[old == c].mean(0) for c in range(10)])
        nc_resid = (H - centers[old]).norm(dim=1)
        nc_angle = ((H * centers[old]).sum(1) / H.norm(dim=1) / centers[old].norm(dim=1))
    named = [(name, p) for name, p in model.named_parameters() if not name.startswith('layers.out')]
    groups = sorted(set(group(name) for name, _ in named))
    gid = torch.tensor([groups.index(group(name)) for name, _ in named], device='cuda')
    records, field_names = [], None
    start = time.monotonic()
    for row, m in enumerate(idx.tolist()):
        h, nodes, _ = forward_nodes(model, C.eval_tensor(data.images[m:m+1], 'cuda'))
        nodes_list = list(nodes.items())
        per_dir = []
        for k, u in enumerate(U):
            allg = torch.autograd.grad((h[0] * u).sum(), [p for _, p in named] + [v for _, v in nodes_list], retain_graph=k < 7)
            paramg = allg[:len(named)]
            dg = dict(zip(nodes, allg[len(named):]))
            pnorm = torch.stack([sq(g) for g in paramg])
            gn = torch.zeros(len(groups), device='cuda').scatter_add_(0, gid, pnorm)
            d = {f'param.{key}': gn[i] for i, key in enumerate(groups)}
            d['total_sq'] = pnorm.sum()
            d.update(norms_metrics(model, nodes, dg, u))
            d['upstream_gain'] = ((d['total_sq'] - d['param.final_norm']).clamp_min(0) / d['final.delta_norm'].square().clamp_min(1e-30)).sqrt()
            d['nc_resid'] = nc_resid[m]
            d['nc_angle'] = nc_angle[m]
            if field_names is None:
                field_names = list(d)
            per_dir.append(torch.stack([d[key] for key in field_names]).detach())
            del allg, paramg, dg
        records.append(torch.stack(per_dir).cpu().numpy())
        del h, nodes, nodes_list, per_dir
        if (row + 1) % 64 == 0:
            print(f'{arm} {t}: {row+1}/{n}; {time.monotonic()-start:.1f}s', flush=True)
    arr = np.stack(records)
    output = OUT / f'{prefix}_{arm}_t{t}_n{n}.npz'
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, values=arr, fields=np.array(field_names), idx=idx.numpy(), U=U.detach().cpu().numpy(), H=H.detach().cpu().numpy(), old=data.labels.numpy(), feature_singular=sv.cpu().numpy())
    info = dict(arm=arm, task=t, n=n, groups=groups, checkpoint=str(path), seconds=time.monotonic()-start,
                rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                cuda_peak_mb=torch.cuda.max_memory_allocated()/2**20, cuda_reserved_mb=torch.cuda.max_memory_reserved()/2**20,
                file=str(output))
    (output.with_suffix('.json')).write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('arm')
    p.add_argument('task', type=int)
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--prefix', default='decompose')
    p.add_argument('--checkpoint')
    p.add_argument('--old-task',type=int)
    a = p.parse_args()
    run(a.arm, a.task, a.n, a.prefix, a.checkpoint, a.old_task)

"""Read-only checkpoint helpers for the authorized R0 posthoc analysis."""
import hashlib
import json
import os
import resource
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix.json')
import numpy as np
import torch
from torch.nn import functional as F
from analysis.vit_long_1024 import common as C

ROOT = Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix')
OUT = ROOT / 'posthoc_r0'
REPO = Path('/home/issan/Projects/claude/wt/vit_long_1024')
RESULT = REPO / 'results/vit_long_1024_2blk_rlfix'
SEED = 10252010


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')


def progress(message):
    with (OUT / 'PROGRESS.md').open('a') as f:
        f.write(f'\n## {now()}\n\n{message}\n')


def dump(path, data):
    path = Path(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def setup(device='cuda'):
    C.set_determinism(2)
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.38)


def resources():
    result = {'rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if torch.cuda.is_initialized():
        result.update(cuda_alloc_mib=torch.cuda.max_memory_allocated() / 2**20,
                      cuda_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    if result['rss_mib'] > 3072:
        raise RuntimeError(f'RSS budget exceeded: {result}')
    return result


def ckpt(arm, task, kind='model'):
    return ROOT / 'runs' / arm / f'seed{SEED}/ckpt/{kind}_t{task:02d}.pt'


def load_model(arm, task, device='cuda', dtype=torch.float32):
    m = C.build_arm_model(arm, SEED, device, 0., 2, 8.)
    m.load_state_dict(torch.load(ckpt(arm, task), map_location='cpu', weights_only=False))
    return m.to(dtype=dtype).eval()


def images(device='cuda', train=True):
    source = C.image_source('real')
    if train:
        data, _ = C.task_dataset(source, SEED, 1, 10, 250, augment=False)
        return torch.cat([C.eval_tensor(data.images[s:s+32], device)
                          for s in range(0, len(data), 32)])
    return C.eval_tensor(source.val(C.probe_names(source, SEED)), device)


def attention(att, x):
    b, n, d = x.shape
    q, k, v = att.layers['qkv'](x).reshape(b, n, 3, att.n_heads, d//att.n_heads).permute(2, 0, 3, 1, 4).unbind(0)
    score = (q @ k.transpose(-2, -1)) * att.scale
    a = score.softmax(-1)
    mixed = (a @ v).transpose(1, 2).reshape(b, n, d)
    output = att.layers['proj'](mixed)
    return output, a, v, score


def parts(model, x, att_detail=False):
    L = model.layers
    e = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), L['patch_embed'](x)), 1) + model.pos_embed
    nodes = {'e': e}
    h = e
    for i in range(2):
        block = L[f'block_{i}'].layers
        nx = block['norm1'](h)
        if att_detail:
            a, aw, v, score = attention(block['attn'], nx)
            nodes.update({f'A{i}': aw, f'V{i}': v, f'score{i}': score})
        else:
            a = block['attn'](nx)
        u = h + a
        y = block['norm2'](u)
        mlp = block['mlp'].layers
        z = mlp['fc1'](y)
        act = mlp['act'](z)
        f = mlp['fc2'](act)
        nodes.update({f'attn{i}': a, f'u{i+1}': u, f'y{i+1}': y,
                      f'z{i}': z, f'act{i}': act, f'ffn{i}': f})
        h = u + f
        nodes[f'h{i+1}'] = h
    nodes['h'] = L['norm'](h)[:, 0]
    return nodes


def features(model, x):
    L = model.layers
    h = torch.cat((model.cls_token.expand(x.shape[0], -1, -1), L['patch_embed'](x)), 1) + model.pos_embed
    for i in range(2):
        h = L[f'block_{i}'](h)
    return L['norm'](h)[:, 0]


def summary(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(mean=float(v.mean()), median=float(np.median(v)),
                negative=float((v < 0).mean()), abs_median=float(np.median(abs(v))))


def logit_stats(h, head, rows, labels=None):
    h = h.double()
    w = head.weight.detach()[rows].double()
    b = head.bias.detach()[rows].double()
    wc, bc = w-w.mean(0), b-b.mean()
    linear = h @ wc.T
    logits = linear + bc
    hrms = float(h.square().sum(1).mean().sqrt())
    wn = float(wc.norm())
    lrms = float(linear.square().sum(1).mean().sqrt())
    out = dict(h_norm_rms=hrms, h_norm_median=float(h.norm(dim=1).median()),
               lr_l1_median=float(1e-4*h.abs().sum(1).median()),
               head_norm=float(w.norm()), centered_head_norm=wn,
               centered_linear_logit_rms=lrms,
               alignment_factor=lrms/(wn*hrms) if wn*hrms else 0.,
               centered_logit_rms=float(logits.square().sum(1).mean().sqrt()),
               centered_bias_norm=float(bc.norm()),
               h_common_energy=float(h.mean(0).square().sum()/h.square().sum(1).mean()))
    if labels is not None:
        out.update(acc=float((logits.argmax(1)==labels).double().mean()),
                   loss=float(F.cross_entropy(logits, labels)))
    return out

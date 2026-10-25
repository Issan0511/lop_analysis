"""Frozen original ViT/data imports plus depth-two measurement helpers."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/issan/Projects/obsidian-research-data')
SOURCE = DATA/'vit_ffn_ablation_1025/source'
ARCHIVE = DATA/'vit_two_block_1025'
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from torch.nn import functional as F
from analysis.vit_long_1024 import common as C
from analysis.joudaki_vit_battle_0919.data import PairedBatches, train_loader
from analysis.joudaki_vit_battle_0919.model import make_model, reset_head, train_forward
from analysis.joudaki_vit_battle_0919.run import atomic_json, atomic_save, get_rng, set_rng, sha256, write_rows

CONFIG = dict(C.model_config(0., True), depth=2)
STEPS = 500
SEEDS = (0, 1, 2, 3, 4)


def snapshot_steps(task):
    if task in (1, 10, 60): return (0, 1, 5, 20, 50, 100, 250, 500)
    if task in (2, 11): return (0, 20, 100, 500)
    if task == 30: return (500,)
    return ()


def init():
    C.set_determinism(2)
    torch.cuda.set_per_process_memory_fraction(.45)


def original_reference():
    archive = DATA/'vit_long_1024/_gigafile_dl/vit_long_1024_vn_raw_seed0.tar'
    with tarfile.open(archive, 'r:') as tar:
        p = json.loads(tar.extractfile('vit_long_1024/runs/SL/seed0/provenance.json').read())
    for name, digest in p['source_hashes'].items():
        assert sha256(SOURCE/name) == digest, name
    assert dict(CONFIG, depth=6) == p['config']['model_config']
    return dict(archive=str(archive), git_hash=p['git_hash'], source_hashes=p['source_hashes'],
                config=p['config'])


def own_hashes():
    paths = [ROOT/'analysis/vit_two_block_1025'/f for f in ('common.py', 'train.py')]
    paths += [ROOT/'specs/spec_vit_two_block_1025.md']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def git_state():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--',
                                    'analysis/vit_two_block_1025', 'specs/spec_vit_two_block_1025.md'],
                                   cwd=ROOT, text=True)
    return head, dirty


def build(seed):
    return make_model('R', seed, 'cuda', CONFIG)


def cpu_tree(obj):
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().clone()
    if isinstance(obj, dict): return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list): return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple): return tuple(cpu_tree(v) for v in obj)
    return copy.deepcopy(obj)


def digest_tree(obj):
    h = hashlib.sha256()
    def visit(x):
        h.update(type(x).__name__.encode())
        if isinstance(x, torch.Tensor):
            t=x.detach().cpu().contiguous()
            h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k, v in x.items(): visit(k); visit(v)
        elif isinstance(x, (tuple, list)):
            for v in x: visit(v)
        else: h.update(repr(x).encode())
    visit(obj)
    return h.hexdigest()


def warm_compile(model, forward, classes):
    rng=get_rng(); model.train()
    for n in (128, 122):
        x=torch.zeros(n, 3, 64, 64, device='cuda')
        y=torch.zeros(n, dtype=torch.long, device='cuda')
        F.cross_entropy(forward(x).index_select(1, classes), y).backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); set_rng(rng)


@torch.no_grad()
def observe(model, x, classes, y=None):
    was_training=model.training; model.eval()
    outputs=[]; zs=[[], []]
    for xx in x.split(32):
        logits, blocks=C.trace(model, xx)
        outputs.append(logits.index_select(1, classes))
        for i, (_, z) in enumerate(blocks): zs[i].append(z[:, 0].double())
    logits=torch.cat(outputs); prob=logits.softmax(1)
    fit=dict(pmax=float(prob.max(1).values.mean()))
    if y is not None:
        fit.update(ce=float(F.cross_entropy(logits, y)), accuracy=float((logits.argmax(1)==y).double().mean()))
    rows=[]; arrays={}
    for i, parts in enumerate(zs, 1):
        z=torch.cat(parts); mu=z.mean(0); sd=z.std(0, unbiased=True)
        valid=sd>1e-8; r=mu/(sd+1e-12); opened=(z>0).double().mean(0)
        assert torch.isfinite(r).all()
        values=dict(mu=mu, sd=sd, r=r, valid=valid, opened=opened)
        arrays.update({f'L{i}_{k}':v.cpu().numpy() for k,v in values.items()})
        rv=r[valid].cpu().numpy()
        rows.append(dict(layer=i, n_valid=int(valid.sum()), n_invalid=int((~valid).sum()),
                         mu_mean=float(mu.mean()), sd_mean=float(sd.mean()),
                         r_mean=float(rv.mean()) if len(rv) else None,
                         r_median=float(np.median(rv)) if len(rv) else None,
                         open_mean=float(opened.mean()), closed_unit_fraction=float((opened==0).double().mean()),
                         **fit))
    model.train(was_training)
    return rows, arrays


def array_sha(x):
    a=np.ascontiguousarray(x)
    return hashlib.sha256(a.tobytes()+str(a.shape).encode()+str(a.dtype).encode()).hexdigest()

"""Shared bookkeeping for the preregistered ELU specificity experiment."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'specs/spec_elu_specificity_1112.md'
ARMS = ('S1', 'S01', 'P1', 'P01', 'Z_keep', 'Z_clear', 'F0')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous()
            h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=lambda k: str(k)):
                add(key); add(x[key])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode()); h.update(str(len(x)).encode())
            for item in x: add(item)
        elif isinstance(x, Path):
            add(str(x))
        else:
            h.update(type(x).__name__.encode()); h.update(repr(x).encode()); h.update(b'\0')
    add(value)
    return h.hexdigest()


def finite_tree(value, path='root'):
    if isinstance(value, dict):
        for key, item in value.items(): finite_tree(item, path + '.' + str(key))
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value): finite_tree(item, path + '.' + str(i))
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise FloatingPointError(path)


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    finite_tree(obj)
    data = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def points(limit, every):
    return sorted({0,1,5,25,50,100,199,200,201,limit,*range(0,limit+1,every)} & set(range(limit+1)))


def execution_files():
    rels = [
        'analysis/elu_specificity_1112/common.py',
        'analysis/elu_specificity_1112/mlp.py',
        'analysis/elu_specificity_1112/vit.py',
        'analysis/loading_bridge_1111/dynamic.py',
        'analysis/loading_bridge_1111/static_probe.py',
        'analysis/p_residual_loading_1111/run.py',
        'analysis/readout_separation_1111/ops.py',
        'analysis/readout_separation_1111/run.py',
        'src/stage1_ctrl_1111.py',
        'src/rlcifar_mlp_battle_0918.py',
        'src/pmnist_0905.py',
        'src/pmnist_rlcifar_0907.py',
        'src/pmnist_rlmnist_0906.py',
        'analysis/vit_long_1024/common.py',
        'analysis/vit_long_1024/production_2blk_rlfix.json',
        'analysis/vit_long_1024/posthoc_2blk/scale_counterfactual.py',
    ]
    rels += [str(p.relative_to(ROOT)) for p in (ROOT/'analysis/joudaki_vit_battle_0919').rglob('*.py')]
    return sorted(set(rels))


def provenance(sources=()):
    code = {rel:sha(ROOT/rel) for rel in execution_files() if (ROOT/rel).is_file()}
    return dict(
        execution_git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        spec_sha256=sha(SPEC),code_hashes=code,code_sha256=digest(code),
        sources={str(Path(p).resolve()):sha(p) for p in sources},
        python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))


@torch.no_grad()
def apply_head(box, bank, arm):
    """F0 needs the model-specific fresh-source restoration and is not handled here."""
    if arm not in ARMS or arm == 'F0': raise ValueError(arm)
    W, b, rows = box.head()
    if arm.startswith('Z_'):
        W.zero_(); b.zero_()
    else:
        entry = bank['entries'][arm]
        W[rows] = entry['W']; b[rows] = entry['b']
    box.opt.state.pop(W, None); box.opt.state.pop(b, None)
    if arm == 'Z_clear':
        for p in box.body(): box.opt.state.pop(p, None)

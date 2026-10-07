#!/usr/bin/env python3
"""Pre-registration design statistics for control 1 (spec §1.2(b)).  No training, no new arm.

Reads the resp_cifar5p1_1007 branch states t02/t28 from their backup (sha256 checked against
the committed backup manifest), evaluates layer-2 preactivations on each seed's task-29 images
ON THE CPU (float32; not bit-identical to the GPU run, used for design only) and summarises
the open fraction (z >= 0) of the candidate fields.  Output: results/controls_cifar5p1_1008/
pre_registration/design_stats.json
"""
import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
torch.set_num_threads(2)
from src import cifar5p1_mlp_0920 as C
B, H = C.B, C.H
man = json.loads((ROOT / 'results/resp_cifar5p1_1007/backup_manifest.json').read_text())
states = {}
for t in (2, 28):
    rec = [f for f in man['files'] if f['relative'] == f'results/resp_cifar5p1_1007/prefix/t{t:02d}.pt']
    assert len(rec) == 1
    p = Path(rec[0]['backup'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == rec[0]['sha256']
    states[t] = torch.load(p, map_location='cpu', weights_only=False)
seeds = list(states[28]['seeds'])
cif = C.Cifar100(Path('/home/issan/Projects/claude/proj_004_drift/data/cifar100/cifar-100-python.tar.gz'))
mean = torch.tensor(C.STD_MEAN).repeat_interleave(1024)
std = torch.tensor(C.STD_STD).repeat_interleave(1024)
tr = C.class_rows(cif.train_y)
rows = [torch.cat([tr[q] for q in C.task_plan(s)[28][1]]) for s in seeds]      # task 29
X = torch.stack([((cif.train_u8[r].float() / 255.0) - mean) / std for r in rows])


def z2(P):
    with torch.no_grad():
        return B.forward(P, X, B.ReLU(), train=False)[2]


def stack(seed_of):
    init = [q.detach() for s in seeds for q in C.init_params('R', seed_of(s), torch.device('cpu'), C.HIDDEN)]
    return [torch.stack(init[i::6]).contiguous() for i in range(6)]


zb, zs, zf = z2(states[28]['P']), z2(states[2]['P']), z2(stack(lambda s: 1000 + s))


def summary(z):
    g = (z >= 0).double()
    pu = g.mean(0)
    return dict(open=float(g.mean()), units_never_open=int((pu == 0).sum()), units_always_open=int((pu == 1).sum()),
                unit_open_q10_q50_q90=[float(x) for x in np.quantile(pu.numpy(), [.1, .5, .9])],
                z_sd_unit_median=float(z.std(0).median()), z_mean_unit_median=float(z.mean(0).median()))


out = dict(note='CPU float32 design statistics on task-29 images (2500 per seed); not the registered GPU values',
           seeds=seeds, per_seed={})
for r, s in enumerate(seeds):
    K = int((zs[r] >= 0).sum())
    v = zb[r].flatten().double().sort(descending=True).values
    c = float(-(v[K - 1] + v[K]) / 2)
    out['per_seed'][str(s)] = dict(receiver_t28=summary(zb[r]), donor_t02=summary(zs[r]), fresh_1000_plus_s=summary(zf[r]),
                                   match_shift=dict(K=K, c=c, boundary_gap=float(v[K - 1] - v[K]),
                                                    **summary(zb[r] + c)))
agg = {}
for key in ('receiver_t28', 'donor_t02', 'fresh_1000_plus_s', 'match_shift'):
    agg[key] = {k: float(np.mean([out['per_seed'][str(s)][key][k] for s in seeds]))
                for k in ('open', 'units_never_open', 'units_always_open', 'z_sd_unit_median', 'z_mean_unit_median')}
agg['match_shift']['c_range'] = [min(out['per_seed'][str(s)]['match_shift']['c'] for s in seeds),
                                 max(out['per_seed'][str(s)]['match_shift']['c'] for s in seeds)]
out['seed_mean'] = agg
dst = ROOT / 'results/controls_cifar5p1_1008/pre_registration/design_stats.json'
dst.write_text(json.dumps(out, indent=1))
print(json.dumps(agg, indent=1))

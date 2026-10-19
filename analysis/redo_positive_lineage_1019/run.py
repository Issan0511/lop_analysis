#!/usr/bin/env python3
"""Read-only instrumentation of the original graph engine; see preregistered spec."""
from __future__ import annotations
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import baselines_cifar5p1_1008 as S
from src import cifar5p1_mlp_0920 as C
from src import rlcifar_mlp_battle_0918 as B
from src import pmnist_0905 as H

RUN = 'redo_positive_lineage_1019'
OUT = ROOT / 'results' / RUN
DATA = Path('/home/issan/Projects/claude/proj_004_drift/data/cifar100/cifar-100-python.tar.gz')
POINTS = {1, 2, 5, 10, 25, 50, 100, 200, 400, 780}
EXPECTED_SHA = 'c66760ba7c55af952adc76a5432edf3a6b765b86887e0fea3732fab8ae693049'


def instrumented(callback):
    original = Path(S.__file__).read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED_SHA:
        raise RuntimeError('Original engine hash changed; review the instrumentation first')
    src = inspect.getsource(S.run)
    def insert(old, new):
        nonlocal src
        if src.count(old) != 1:
            raise RuntimeError(f'instrumentation anchor count {src.count(old)}: {old[:100]}')
        src = src.replace(old, new)
    insert('        grads = torch.autograd.grad(lossv.sum(), P)\n',
           '        grads = torch.autograd.grad(lossv.sum(), P)\n        _lineage_grads[:] = grads\n')
    insert('        units = [[torch.nonzero(masks[l][r]).flatten() for r in range(R)] for l in range(2)]\n',
           '        units = [[torch.nonzero(masks[l][r]).flatten() for r in range(R)] for l in range(2)]\n'
           '        if diagnostic is not None:\n'
           '            diagnostic("reset_pre", P=P, m=adam_m, v=adam_v, masks=masks, tc=n, t=t, act=act)\n')
    insert('                    adam_v[2 * l][r, idx, :] = 0.0\n        for l in range(2):',
           '                    adam_v[2 * l][r, idx, :] = 0.0\n'
           '            if diagnostic is not None:\n'
           '                diagnostic(f"reset_incoming_{l+1}", P=P, m=adam_m, v=adam_v, masks=masks, tc=n, t=t, act=act)\n'
           '        for l in range(2):')
    insert('                    adam_v[2 * l + 2][r, :, idx] = 0.0\n        for r in range(R):',
           '                    adam_v[2 * l + 2][r, :, idx] = 0.0\n'
           '            if diagnostic is not None:\n'
           '                diagnostic(f"reset_outgoing_{l+1}", P=P, m=adam_m, v=adam_v, masks=masks, tc=n, t=t, act=act)\n'
           '        for r in range(R):')
    insert('            act.begin_step(R, C.BATCH, device)\n            if cg is not None:',
           '            if diagnostic is not None and phase == "train" and j + 1 in _lineage_points:\n'
           '                diagnostic("before_step", P=P, m=adam_m, v=adam_v, tc=tc, t=t, j=j, idx=batches[:, j], act=act)\n'
           '            act.begin_step(R, C.BATCH, device)\n            if cg is not None:')
    insert('            if hook is not None:\n                hook("post_step",',
           '            if diagnostic is not None and phase == "train" and j + 1 in _lineage_points:\n'
           '                diagnostic("after_step", P=P, m=adam_m, v=adam_v, tc=tc, t=t, j=j, grads=_lineage_grads, inv_c1=inv_c1, inv_c2=inv_c2, act=act)\n'
           '            if hook is not None:\n                hook("post_step",')
    insert('    for t in range(1, n_tasks + 1):\n',
           '    if diagnostic is not None:\n'
           '        diagnostic("init", P=P, m=adam_m, v=adam_v, tc=0, t=1, act=act)\n'
           '    for t in range(1, n_tasks + 1):\n')
    insert('            if observer is not None:\n',
           '            if diagnostic is not None:\n'
           '                diagnostic("task_end", P=P, m=adam_m, v=adam_v, tc=tc, t=t, Xt=Xt, act=act)\n'
           '            if observer is not None:\n')
    env = dict(S.__dict__, diagnostic=callback, _lineage_grads=[], _lineage_points=POINTS)
    exec(compile(src, '<read-only-lineage-instrumentation>', 'exec'), env)
    return env['run'], src


def cpu(x):
    return x.detach().cpu().numpy().copy()


def summary(z):
    z = z.detach()
    mean, sd = z.mean(1), z.std(1)
    return {
        'p': cpu((z > 0).float().mean(1)), 'min': cpu(z.amin(1)),
        'max': cpu(z.amax(1)), 'mean': cpu(mean), 'sd': cpu(sd),
        'r': cpu(mean / sd.clamp_min(1e-20)),
        'negative_amplitude': cpu((-z).clamp_min(0).mean(1)),
        'activation_mean': cpu(z.clamp_min(0).mean(1)),
    }


class Trace:
    def __init__(self, cifar, seeds, device, dest, check=False):
        self.dest, self.seeds, self.device = dest, seeds, device
        dest.mkdir(parents=True, exist_ok=True)
        self.check = check
        self.X = cifar.inputs('train', 'std', device)
        self.Y = cifar.train_y.to(device)
        self.rows = C.class_rows(cifar.train_y)
        gi = torch.cat([r[:10] for r in self.rows]).to(device)
        hi = torch.cat([r[10:20] for r in self.rows]).to(device)
        self.gX, self.hX = self.X[gi], self.X[hi]
        self.gi, self.hi = cpu(gi), cpu(hi)
        self.plans = [C.task_plan(s) for s in seeds]
        idx3 = [torch.cat([self.rows[c] for c in p[2][1]]) for p in self.plans]
        self.X3 = torch.stack([self.X[r.to(device)] for r in idx3])
        self.events, self.values, self.counts, self.deltas = [], [], [], []
        self.generation = np.zeros((len(seeds), 2, C.HIDDEN), dtype=np.int32)
        self.last_reset = np.zeros_like(self.generation)
        self._before = None
        self.grad_checks = []

    def forward_global(self, P, act):
        return B.forward(P, self.gX.unsqueeze(0).expand(len(self.seeds), -1, -1), act, train=False)

    @torch.no_grad()
    def __call__(self, event, **s):
        P, t, tc, act = s['P'], s['t'], s['tc'], s['act']
        if event == 'before_step':
            z1, h1, z2, _, _ = self.forward_global(P, act)
            self._before = dict(W=P[2].clone(), b=P[3].clone(), mu=h1.mean(1), mean=z2.mean(1), p=(z2>0).float().mean(1),
                                momentW=s['m'][2].clone(), momentb=s['m'][3].clone())
            if self.check and len(self.grad_checks) == 0:
                self._before['params'] = [p.detach().clone().requires_grad_(True) for p in P]
                self._before['xb'] = self.X[s['idx']]
                self._before['yb'] = self.Y[s['idx']]
            return
        if event == 'reset_pre':
            self.pending = np.stack([cpu(x) for x in s['masks']], axis=1)
        if event == 'reset_incoming_1':
            self.generation[:, 0] += self.pending[:, 0]
            self.last_reset[:, 0] = np.where(self.pending[:, 0], tc, self.last_reset[:, 0])
        if event == 'reset_incoming_2':
            self.generation[:, 1] += self.pending[:, 1]
            self.last_reset[:, 1] = np.where(self.pending[:, 1], tc, self.last_reset[:, 1])
        z1, h1, z2, a2, _ = self.forward_global(P, act)
        data = {}
        for l, z in ((1, z1), (2, z2)):
            data.update({f'g_l{l}_{k}': v for k, v in summary(z).items()})
        if t <= 3:
            f3 = B.forward(P, self.X3, act, train=False)
            for l, z in ((1, f3[0]), (2, f3[2])):
                data.update({f't3_l{l}_{k}': v for k, v in summary(z).items()})
        if event == 'task_end':
            ft = B.forward(P, s['Xt'], act, train=False)
            for l, z in ((1, ft[0]), (2, ft[2])):
                data.update({f'current_l{l}_{k}': v for k, v in summary(z).items()})
        data['generation'] = self.generation.copy()
        data['last_reset'] = self.last_reset.copy()
        data['outnorm_l1'] = cpu(torch.linalg.vector_norm(P[2], dim=1))
        data['outnorm_l2'] = cpu(torch.linalg.vector_norm(P[4], dim=1))
        if event.startswith('reset_'):
            data['reset_mask'] = self.pending.copy()
        if event == 'after_step':
            b = self._before
            mu = h1.mean(1)
            own_w = torch.einsum('rij,rj->ri', P[2] - b['W'], b['mu'])
            own_b = P[3] - b['b']
            upstream = torch.einsum('rij,rj->ri', P[2], mu - b['mu'])
            actual = z2.mean(1) - b['mean']
            gW, gb = s['grads'][2], s['grads'][3]
            ic1, ic2 = s['inv_c1'], s['inv_c2']
            denW = (s['v'][2] * ic2).sqrt() + 1e-8
            denb = (s['v'][3] * ic2).sqrt() + 1e-8
            currentW = -C.LR * ((1 - .9) * gW * ic1) / denW
            currentb = -C.LR * ((1 - .9) * gb * ic1) / denb
            oldW = -C.LR * (.9 * b['momentW'] * ic1) / denW
            oldb = -C.LR * (.9 * b['momentb'] * ic1) / denb
            current = torch.einsum('rij,rj->ri', currentW, b['mu']) + currentb
            memory = torch.einsum('rij,rj->ri', oldW, b['mu']) + oldb
            d = dict(event=len(self.events), task=t, tc=tc, step=s['j']+1,
                     own_w=cpu(own_w), own_b=cpu(own_b), upstream=cpu(upstream), actual=cpu(actual),
                     adam_current=cpu(current), adam_memory=cpu(memory),
                     sgd_direction=cpu(-(torch.einsum('rij,rj->ri', gW, b['mu']) + gb)),
                     bias_gradient=cpu(gb),
                     closure=cpu(actual - own_w - own_b - upstream),
                     adam_closure=cpu(own_w + own_b - current - memory),
                     pre_p=cpu(b['p']))
            self.deltas.append(d)
            if 'params' in b:
                with torch.enable_grad():
                    zz = B.forward(b['params'], b['xb'], act, train=False)[-1]
                    lv = F.cross_entropy(zz.reshape(-1,C.N_CLASSES), b['yb'].reshape(-1), reduction='none').view(len(self.seeds),-1).mean(1).sum()
                    gr = torch.autograd.grad(lv, b['params'])
                self.grad_checks.append(max(float((x-y).abs().max()) for x,y in zip(gr,s['grads'])))
            self._before = None
        meta = dict(index=len(self.events), event=event, task=t, tc=tc,
                    step=s.get('j', 779)+1 if event == 'after_step' else (0 if event == 'init' else 780))
        self.events.append(meta)
        self.values.append(data)
        for r, seed in enumerate(self.seeds):
            row = dict(**meta, seed=seed)
            for scope in ('g', 't3', 'current'):
                for l in (1,2):
                    key = f'{scope}_l{l}_p'
                    if key not in data:
                        continue
                    p = data[key][r]
                    row.update({f'{scope}_l{l}_{k}': v for k,v in {
                        'all_on': int((p==1).sum()), 'all_off': int((p==0).sum()),
                        'near95': int((p>=.95).sum()), 'near99': int((p>=.99).sum()),
                        'p_median': float(np.sort(p)[49]), 'p_mean': float(p.mean()),
                        'mean_median': float(np.sort(data[f'{scope}_l{l}_mean'][r])[49]),
                    }.items()})
            self.counts.append(row)
        deep = t <= 3 or (event == 'task_end' and t in {10,20,29})
        if deep:
            _, hh, zh, _, _ = B.forward(P, self.hX.unsqueeze(0).expand(len(self.seeds),-1,-1), act, train=False)
            curs = [torch.cat([self.rows[c] for c in p[t-1][1]]) for p in self.plans]
            cur_idx = torch.stack([v[torch.linspace(0,len(v)-1,256).long()] for v in curs]).to(self.device)
            _, hc, _, _, _ = B.forward(P, self.X[cur_idx], act, train=False)
            np.savez_compressed(self.dest/f'deep_{meta["index"]:04d}.npz',
                                h1=cpu(h1), z2=cpu(z2), held_h1=cpu(hh), held_z2=cpu(zh),
                                current_h1=cpu(hc), current_y=cpu(self.Y[cur_idx]),
                                W2=cpu(P[2]), b2=cpu(P[3]), W3=cpu(P[4]), b3=cpu(P[5]))
        if event == 'task_end' and (self.check or t % 5 == 0):
            self.save()

    def save(self):
        (self.dest/'events.json').write_text(json.dumps(dict(seeds=self.seeds, events=self.events,
                                                          gradient_check_max=self.grad_checks),indent=2))
        keys = sorted(set().union(*(x.keys() for x in self.values)))
        arrays = {}
        for k in keys:
            ref = next(x[k] for x in self.values if k in x)
            arrays[k] = np.stack([x.get(k, np.full_like(ref, np.nan, dtype=np.float32)) for x in self.values])
        np.savez_compressed(self.dest/'units.npz', **arrays)
        if self.deltas:
            np.savez_compressed(self.dest/'deltas.npz', **{k:np.stack([d[k] for d in self.deltas]) for k in self.deltas[0]})
        H.write_csv(self.dest/'counts.csv', self.counts)


def compare_csv(a,b):
    aa = list(csv.DictReader(Path(a).open()))
    bb = list(csv.DictReader(Path(b).open()))
    bb = {(q['seed'], q['task']):q for q in bb}
    diffs=[]
    for q in aa:
        ref=bb[(q['seed'],q['task'])]
        for k,v in q.items():
            if k in ref and v != ref[k]:
                try:
                    x,y=float(v),float(ref[k])
                    if np.isnan(x) and np.isnan(y): continue
                    diffs.append(dict(seed=q['seed'],task=q['task'],metric=k,actual=x,reference=y,absdiff=abs(x-y)))
                except ValueError:
                    diffs.append(dict(seed=q['seed'],task=q['task'],metric=k,actual=v,reference=ref[k]))
    return dict(rows=len(aa),exact=not diffs,n_different=len(diffs),differences=diffs)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['check','run'],required=True)
    ap.add_argument('--method',choices=['redo','none'],default='redo')
    ap.add_argument('--group',type=int,choices=[0,10],default=0)
    a=ap.parse_args()
    torch.set_num_threads(2)
    device=H.setup('cuda')
    cifar=C.Cifar100(DATA)
    cfg=S.make_cfg(a.method,tau='0.1',period=1560) if a.method=='redo' else S.make_cfg('none')
    seeds=list(range(a.group,a.group+10))
    out=OUT/('checks' if a.mode=='check' else 'raw')/f'{a.method}_s{a.group}-{a.group+9}'
    trace=Trace(cifar,seeds,device,out/'trace',check=a.mode=='check')
    run,source=instrumented(trace)
    (out/'instrumented_source.py').write_text(source)
    n=3 if a.mode=='check' else 30
    final={}
    prov=run(cfg,seeds,n,device,out/'replay',cifar=cifar,fresh=False,graph=True,final=final)
    trace.save()
    if a.mode=='check':
        other={}
        S.run(cfg,seeds,n,device,out/'plain',cifar=cifar,fresh=False,graph=True,final=other)
        tensors={k:all(torch.equal(x,y) for x,y in zip(final[k],other[k])) for k in ('P','m','v')}
        compare=compare_csv(out/'replay/per_task.csv',out/'plain/per_task.csv')
        de=np.load(out/'trace/deltas.npz')
        result=dict(tensors=tensors,per_task=compare,gradient_check_max=trace.grad_checks,
                    max_position_closure=float(np.abs(de['closure']).max()),
                    max_adam_closure=float(np.abs(de['adam_closure']).max()))
        (out/'check.json').write_text(json.dumps(result,indent=2))
        if not all(tensors.values()) or not compare['exact'] or max(trace.grad_checks)>1e-7:
            raise RuntimeError(result)
    else:
        original=ROOT/'results/baselines_cifar5p1_1008/main'/f'{a.method}_s{a.group}-{a.group+9}'/'per_task.csv'
        result=compare_csv(out/'replay/per_task.csv',original)
        (out/'reproduction.json').write_text(json.dumps(result,indent=2))
        print('REPRODUCTION',json.dumps({k:v for k,v in result.items() if k!='differences'}),flush=True)
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,
                     tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                     tf32_cudnn=torch.backends.cudnn.allow_tf32,
                     float32_matmul_precision=torch.get_float32_matmul_precision(),
                     git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                     engine_sha=hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest(),
                     instrumentation_sha=hashlib.sha256(source.encode()).hexdigest(),
                     data_sha=cifar.sha256, original=prov)
    (out/'environment.json').write_text(json.dumps(environment,indent=2))


if __name__=='__main__': main()

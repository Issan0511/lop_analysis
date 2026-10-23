#!/usr/bin/env python3
"""Exact final-FFN loading-gradient diagnostics, with no parameter updates."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import csv
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'results/chargpt_loadsign_1023'
DATA = Path('/home/issan/Projects/obsidian-research-data')
BUNDLE = DATA/'chargpt_seed10_1022/transfer/chargpt_seed10_1022_bundle.tar'
BUNDLE_SHA = '5729ec0279f25d884cfec0aef8c4234d2f9788d57208fed99a5c7eff4ad23e61'
sp = importlib.util.spec_from_file_location('latepush', ROOT/'analysis/chargpt_latepush_1023/measure.py')
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
A, C, torch, np = P.A, P.C, P.torch, P.np
dump, write_csv, relmax = P.dump, P.write_csv, P.relative_max


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4<<20), b''): h.update(b)
    return h.hexdigest()


class Inputs:
    def __init__(self):
        self.tar = tarfile.open(BUNDLE, 'r:')
        self.members = {m.name: m for m in self.tar.getmembers()}
        self.old = {}
        self.ref = {(r['arm'], int(r['seed']), int(r['task'])): r for r in
                    csv.DictReader((ROOT/'results/chargpt_latepush_1023/states.csv').open()) if r['layer']=='1'}
        pv = json.loads((ROOT/'results/chargpt_latepush_1023/provenance.json').read_text())
        self.rawhash = {Path(r['path']).name:r['sha256'] for r in pv['raw_files']}

    def load(self, arm, seed, task):
        name = f'results/chargpt_cl_1019/M2/{arm}/seed{seed}/ckpts/task{task}.pt'
        sd = torch.load(io.BytesIO(self.tar.extractfile(self.members[name]).read()), map_location='cpu', weights_only=True)
        C.L = 2
        m = C.GPT(arm, 'M'); m.load_state_dict(sd); m.eval().double()
        for b in m.h: b.mlp.track_v = False
        assert m.wte.weight is m.head.weight
        return m

    def previous(self, arm, seed):
        key = (arm, seed)
        if key not in self.old:
            p = DATA/f'chargpt_latepush_1023/results/chargpt_latepush_1023/raw/{arm}_seed{seed}.npz'
            assert sha(p) == self.rawhash[p.name]
            self.old[key] = np.load(p)
        return self.old[key]


def median(x):
    return float(np.median(x.detach().numpy())) if x.numel() else float('nan')


def cosine(a, b):
    return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-300)


def cent(x): return x-x.mean(-1, keepdim=True)


@torch.no_grad()
def task_fields(m, x, labels):
    hs, ys, zs = [], [], []
    for _, xx in A.chunks(x):
        cap = []; h = m.body(xx, cap)
        hs.append(h.reshape(-1,C.D))
        ys.append(m.h[1].n2(cap[2]).reshape(-1,C.D))
        zs.append(cap[3].reshape(-1,C.FF))
    h, y, z = map(torch.cat, [hs,ys,zs])
    vc = m.head.weight-m.head.weight.mean(0)
    fm = (torch.nn.functional.one_hot(labels.reshape(-1),C.V).to(A.DT)-1/C.V).T@h
    kappa = (vc*fm).sum()/fm.square().sum()
    return y.mean(0), (z>0).any(0), y, vc, kappa*fm, float(kappa)


@torch.no_grad()
def calculate(m, xu, ybar, alive, vl):
    """Algebraic pullback through final LN, preserving original forward state."""
    mlp = m.h[1].mlp
    U = mlp.down.weight; Uc = U-U.mean(0,keepdim=True)
    pushes = {f'{e}_{l}_{k}':torch.zeros(C.FF,dtype=A.DT)
              for e in ['original','loaderror'] for l in ['exact','direct'] for k in ['K','absK']}
    extra = {n:torch.zeros(C.FF,dtype=A.DT) for n in ['radial','constantK','projection_cos2']}
    active = torch.zeros(C.FF,dtype=torch.bool)
    stats = {}
    def add(name, vals): stats[name] = stats.get(name, 0.)+float(vals.sum())/N
    N = xu.numel()
    for _, xx in A.chunks(xu):
        cap=[]; h=m.body(xx,cap).reshape(-1,C.D)
        r=cap[2].reshape(-1,C.D); z=cap[3].reshape(-1,C.FF)
        y=m.h[1].n2(r); gate=mlp.dphi(z); active |= (gate!=0).any(0)
        ffn=mlp.down(mlp.phi(z)); t=r+ffn
        c=cent(t); sigma=(c.square().mean(-1,keepdim=True)+m.nf.eps).sqrt(); q=c/sigma
        assert relmax(h-m.nf.weight*q,h)<=1e-12
        k=y@ybar
        logits=m.head(h); plogits=h@vl.T
        prob=logits.softmax(-1); pload=plogits.softmax(-1)
        err=prob-1/C.V; errload=pload-1/C.V
        qU=q@U
        for ename, e in [('original',err),('loaderror',errload)]:
            b=m.nf.weight*(e@vl)
            d=cent(b)/sigma
            direct=d@U
            radial=-qU*(q*b).mean(-1,keepdim=True)/sigma
            exact=direct+radial
            for lname, pull in [('exact',exact),('direct',direct)]:
                for kname, weight in [('K',k),('absK',k.abs())]:
                    pushes[f'{ename}_{lname}_{kname}']+=(weight[:,None]*gate*pull).sum(0)/N
            if ename=='original':
                extra['radial']+=(k[:,None]*gate*radial).sum(0)/N
                extra['constantK']+=(gate*exact).sum(0)/N
        extra['projection_cos2']+=(qU.square()/(q.square().sum(-1,keepdim=True)*Uc.square().sum(0)).clamp_min(1e-300)).sum(0)/N
        add('K_uniform_positive', (k>0).double()); add('K_uniform_negative',(k<0).double())
        add('sigma_mean',sigma.flatten())
        tc,rc,fc=cent(t),cent(r),cent(ffn)
        add('t_norm_mean',tc.norm(dim=-1)); add('residual_norm_mean',rc.norm(dim=-1)); add('ffn_norm_mean',fc.norm(dim=-1))
        add('ffn_over_residual_mean',fc.norm(dim=-1)/rc.norm(dim=-1).clamp_min(1e-300))
        add('ffn_residual_cos_mean',cosine(fc,rc)); add('error_cos_mean',cosine(err,errload))
        add('error_loadlogit_dot_mean',(err*plogits).sum(-1))
        add('error_loadlogit_dot_positive',((err*plogits).sum(-1)>0).double())
        add('entropy_mean',-(prob*prob.clamp_min(1e-300).log()).sum(-1))
        add('loaderror_entropy_mean',-(pload*pload.clamp_min(1e-300).log()).sum(-1))
    s=pushes['original_exact_K']; direct=pushes['original_direct_K']; radial=extra['radial']
    add_error=relmax(s-direct-radial,s,direct,radial)
    assert add_error<=1e-10
    rows={f's_{n}':A.sink(v,alive) for n,v in pushes.items()}
    rows.update({f'abs_{n}':median(v[alive].abs()) for n,v in pushes.items()})
    denom=direct[alive].abs()+radial[alive].abs(); nz=denom>0
    rows.update(stats)
    rows.update(s_radial=A.sink(radial,alive), s_constantK=A.sink(extra['constantK'],alive),
                n_gradient_active=int(active.sum()), n_alive_gradient_active=int((active&alive).sum()),
                s_nonzero=A.sink(s,alive&(s!=0)), n_zero=int((alive&(s==0)).sum()),
                cancellation=median(s[alive][nz].abs()/denom[nz]), cancellation_missing=int((~nz).sum()),
                direct_positive_to_nonpositive=float(((direct[alive]>0)&(s[alive]<=0)).double().mean()),
                direct_nonpositive_to_positive=float(((direct[alive]<=0)&(s[alive]>0)).double().mean()),
                projection_cos2_median=median(extra['projection_cos2'][alive]),additivity_error=add_error)
    return rows, {**pushes, **extra, 'alive':alive, 'gradient_active':active}


def state(inputs,arm,seed,task,autograd=False):
    m=inputs.load(arm,seed,task)
    x,labels,_,_=A.data(seed,task)
    ybar,alive,y,vc,vl,kappa=task_fields(m,x,labels)
    row, raw=calculate(m,A.uniform(seed),ybar,alive,vl)
    s=raw['original_exact_K']; old=inputs.previous(arm,seed)
    old_s=torch.from_numpy(old[f't{task}_L1_load'])
    old_alive=torch.from_numpy(old[f't{task}_L1_alive'])
    ref=inputs.ref[(arm,seed,task)]
    err=relmax(s-old_s,old_s)
    assert err<=1e-10,(arm,seed,task,err)
    assert torch.equal(alive,old_alive)
    assert A.sink(s,alive)==float(ref['s_load']),(arm,seed,task,'sign mismatch')
    assert A.rel(kappa,ref['kappa'])<=1e-10
    agerr=None
    if autograd:
        g=P.fixed_error_gradients(m,A.uniform(seed),[vl])[0][1]
        ags=(g*ybar).sum(1)
        agerr=relmax(s-ags,ags)
        assert agerr<=1e-10,(arm,seed,task,'autograd',agerr)
    key=dict(arm=arm,seed=seed,task=task,layer=1,n_alive=int(alive.sum()),kappa=kappa,
             K_task_positive=float(((y@ybar)>0).double().mean()),
             v_norm=float(vc.norm()),vL_norm=float(vl.norm()),W_norm=float(m.h[1].mlp.up.weight.norm()),
             v_over_W=float(vc.norm()/m.h[1].mlp.up.weight.norm()),
             previous_error=err,autograd_error=agerr)
    return {**key,**row},raw


def gate(inputs):
    rows=[]
    for arm in A.ARMS:
        for t in [1,9,50,100]:
            r,_=state(inputs,arm,0,t,autograd=True); rows.append(r)
    dump(OUT/'gate.json',dict(passed=True,states=8,rows=rows,
        max_previous_error=max(r['previous_error'] for r in rows),
        max_autograd_error=max(r['autograd_error'] for r in rows),code_sha256=sha(__file__)))
    print('gate passed',flush=True)


def run(inputs):
    g=json.loads((OUT/'gate.json').read_text()); assert g['passed'] and g['code_sha256']==sha(__file__)
    rows=[]; start=time.monotonic()
    for seed in range(10):
        for arm in A.ARMS:
            part=OUT/'parts'/f'{arm}_seed{seed}.csv'; rawpath=OUT/'raw'/f'{arm}_seed{seed}.npz'
            done=part.with_suffix('.json')
            if done.exists():
                d=json.loads(done.read_text()); assert d['code_sha256']==sha(__file__) and d['sha256']==sha(part) and d['raw_sha256']==sha(rawpath)
                rows.extend(list(csv.DictReader(part.open()))); continue
            rr=[]; arrays={}; ts=time.monotonic()
            for t in A.TASKS:
                row,raw=state(inputs,arm,seed,t); rr.append(row)
                arrays.update({f't{t}_{k}':v.numpy() for k,v in raw.items()})
            write_csv(part,rr); rawpath.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(rawpath,**arrays)
            dump(done,dict(sha256=sha(part),raw_sha256=sha(rawpath),code_sha256=sha(__file__),seconds=time.monotonic()-ts))
            rows.extend(rr)
            print(f'{arm} seed{seed}: {time.monotonic()-ts:.1f}s',flush=True)
        A.CACHE.clear(); A.UCACHE.clear()
    write_csv(OUT/'states.csv',rows)
    dump(OUT/'provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         script_sha256=sha(__file__),spec_sha256=sha(ROOT/'specs/spec_chargpt_loadsign_1023.md'),
         previous_script_sha256=sha(ROOT/'analysis/chargpt_latepush_1023/measure.py'),
         bundle_sha256=BUNDLE_SHA,training_steps=0,optimizer_created=False,checkpoint_extracted=False,
         host=platform.node(),torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),device='cpu',dtype='float64',
         finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=len(rows),seconds=time.monotonic()-start,
         previous_max_error=max(float(r['previous_error']) for r in rows)))


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['gate','run']); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='gate':
        got=sha(BUNDLE); assert got==BUNDLE_SHA
        dump(OUT/'input_verification.json',dict(path=str(BUNDLE),sha256=got,bytes=BUNDLE.stat().st_size))
    inputs=Inputs()
    if args.mode=='gate': gate(inputs)
    else: run(inputs)


if __name__=='__main__': main()

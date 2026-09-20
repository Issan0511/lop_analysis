#!/usr/bin/env python3
"""Registered S3 continuation. No training equations are implemented here."""
from __future__ import annotations
import csv, hashlib, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import relu_doors_0919 as E
ROOT=E.H.REPO
RUN='ch_chb_200_0919'
OUT=ROOT/'results'/RUN
PARENT=ROOT/'results/relu_doors_0919'
SPEC=ROOT/'specs/spec_ch_chb_200_0919.md'
APPROVAL='a26475002cd08cf4859f317737cd0d83c51a4282'
LAM={'CH':0.,'CHB':.1713}
EXPECTED={'CH':'5135c3b634c292177f8d26fb6243ba8a6540b2aa22aaef9f2c9972e7ade9ec97', 'CHB':'8e43503921212d4239818271a9153c3fc4dc64ff0f8e88ccdf4677d52f8f2866'}
STATE_KEYS=('t','P','m','v','tc','act_state','g_lab','g_batch','alive')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def put(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=True)+'\n');os.replace(tmp,path)

def readcsv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def writecsv(path, rows):
    assert rows
    cols=list(rows[0]); assert all(set(r)==set(cols) for r in rows)
    tmp=Path(str(path)+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def digest(obj):
    h=hashlib.sha256()
    def feed(o):
        if isinstance(o,torch.Tensor):
            a=o.detach().cpu().contiguous().numpy();feed(str(a.dtype));feed(list(a.shape));h.update(a.tobytes())
        elif isinstance(o,dict):
            h.update(b'{')
            for k in sorted(o,key=lambda v:(type(v).__name__,str(v))):feed(k);feed(o[k])
            h.update(b'}')
        elif isinstance(o,(list,tuple)):
            h.update(b'[')
            for v in o:feed(v)
            h.update(b']')
        else:h.update((type(o).__name__+':'+repr(o)+';').encode())
    feed(obj);return h.hexdigest()

def state(st):return {k:st[k] for k in STATE_KEYS}

def live_state(c):
    return dict(t=c['t_first']-1,P=c['P'],m=c['adam_m'],v=c['adam_v'],tc=c['tc'],
        act_state=c['act'].state(),g_lab={s:g.get_state() for s,g in c['g_lab'].items()},
        g_batch={s:g.get_state() for s,g in c['g_batch'].items()},alive=c['alive'])

def validate_checkpoint(st,arm,seeds,epochs,t):
    assert st['meta']==dict(arm=arm,seeds=seeds,conds=['raw'],epochs=epochs,lr=.001,
        lam=LAM[arm],beta=.01,doors=E.DOORS[arm],perturb=None,nan_slot=None),'meta'
    assert st['t']==t and st['tc']==t*epochs*75,'task/Adam clock'
    assert st['alive'].tolist()==[True]*len(seeds),'alive'
    assert len(st['rows'])==len(seeds)*t,'rows'
    assert {(r['seed'],r['task']) for r in st['rows']}=={(s,i) for s in seeds for i in range(1,t+1)},'row keys'
    for k in ('P','m','v'):
        assert len(st[k])==6 and all(torch.isfinite(v).all() for v in st[k]),k

def validate_history(out,arm,seeds,t):
    rows=readcsv(out/'per_task.csv')
    assert {(int(r['seed']),int(r['task'])) for r in rows}=={(s,i) for s in seeds for i in range(1,t+1)}
    assert len(rows)==t*len(seeds)
    for s in seeds:
        rr=sorted([r for r in rows if int(r['seed'])==s],key=lambda r:int(r['task']))
        with np.load(out/'hist'/f'{arm}_raw_seed{s}.npz') as d:
            for k in d.files:
                if k not in ('edges','th_edges'):assert len(d[k])==t,(k,s,t)
            assert [f'{x:.10g}' for x in d['acc']]==[r['memo_acc'] for r in rr]
        for i in range(t+1):assert E.snapshot_path(out,arm,'raw',s,i).is_file()

def prepare():
    manifest=PARENT/'backup_manifest.json'
    assert sha(manifest)=='27df5aa8da591e98ff9abd1c211eeaba9a034c8ad0ac24d04136e07d139cf790'
    records=[]
    for arm in LAM:
        for rec in json.loads(manifest.read_text())['files']:
            prefix=f'results/relu_doors_0919/{arm}/'
            if not rec['source'].startswith(prefix):continue
            rel=Path(rec['source'].removeprefix(prefix));src=Path(rec['backup'])
            assert not src.is_symlink() and src.stat().st_size==rec['bytes'] and sha(src)==rec['sha256']
            imm=OUT/'inputs'/arm/('t50_ckpt.pt' if str(rel)=='ckpt.pt' else rel)
            dst=OUT/arm/rel
            for p in (imm,dst):
                p.parent.mkdir(parents=True,exist_ok=True)
                if not p.exists():shutil.copy2(src,p)
                assert sha(p)==rec['sha256']
            records.append(dict(**rec,immutable=str(imm.relative_to(ROOT)),destination=str(dst.relative_to(ROOT))))
        for name in ('per_task.csv','provenance.json'):
            src=PARENT/arm/name
            for p in (OUT/'inputs'/arm/name,OUT/arm/name):
                p.parent.mkdir(parents=True,exist_ok=True)
                if not p.exists():shutil.copy2(src,p)
                assert sha(p)==sha(src)
            records.append(dict(source=str(src.relative_to(ROOT)),backup=None,bytes=src.stat().st_size,sha256=sha(src)))
        assert sha(OUT/arm/'ckpt.pt')==EXPECTED[arm]
        st=torch.load(OUT/arm/'ckpt.pt',map_location='cpu',weights_only=False)
        validate_checkpoint(st,arm,list(range(10)),400,50)
        validate_history(OUT/arm,arm,list(range(10)),50)
        prov=json.loads((PARENT/arm/'provenance.json').read_text())
        assert prov['door_lam']==LAM[arm] and prov['torch']==torch.__version__
    put(OUT/'source_manifest.json',dict(parent_manifest_sha256=sha(manifest),files=records))
    put(OUT/'registration.json',registration())

def registration():
    v=PARENT/'verdict.json';r=ROOT/'results/rlcifar_mlp_battle_0918/R/per_task.csv'
    assert sha(v)=='3f062ba3398782c20e1747de9e12d13ff8a38ace7032328ef3dc53840c9c2edc'
    assert sha(r)=='f6d1d0a2a5c9d96dad1a3aa58bbba4c41f0711fe38cd14e1dcd2f429ca0d8328'
    rr=readcsv(PARENT/'CH/per_task.csv')
    b=np.array([np.mean([float(q['online_acc']) for q in rr if int(q['seed'])==s and 31<=int(q['task'])<=50]) for s in range(10)])
    h=1.8331129326536335*np.std(b,ddof=1)*math.sqrt(1.1)
    # Reference verdict stores these same windows; retain full derived precision.
    ref=readcsv(r); before=[];after=[]
    for s in range(10):
        rs=sorted([q for q in ref if q['cond']=='std' and int(q['seed'])==s],key=lambda q:int(q['task']))
        j=next(i for i in range(1,len(rs)) if float(rs[i]['online_acc'])<.5<=float(rs[i-1]['online_acc']))
        assert int(rs[j]['task'])==3
        before.append(float(rs[j-1]['zbar_l2'])/float(rs[j-1]['zsd_l2']))
        after.append(float(rs[j]['zbar_l2'])/float(rs[j]['zsd_l2']))
    dc=float((np.median(before)+np.median(after))/2)
    assert dc==-1.6023907100833572
    return dict(run_id=RUN,approval_commit=APPROVAL,spec_sha256=sha(SPEC),baseline=[31,50],endpoint=[181,200],
        h=float(h),L_H=float(np.mean(b)-h),d_c=dc,beta_omega=.0105,seeds=list(range(10)),lam=LAM,
        source_sha256={str(v.relative_to(ROOT)):sha(v),str(r.relative_to(ROOT)):sha(r)},independent_audit=False)

def ratio(a,b):return a/b if b else (-math.inf if a<0 else math.inf if a>0 else math.nan)

def weight_metrics(w,lr=.001):
    # Saved layout is (output unit,input coordinate), explicitly checked against DIMS.
    w=np.asarray(w,dtype=np.float64); n=np.linalg.norm(w-w.mean(axis=1,keepdims=True),axis=1)
    N=float(np.exp(np.mean(np.log(n)))) if np.all(n>0) else math.nan
    return n,N,lr/N

def gate_counts(z):
    z=np.asarray(z);return int(np.all(z<=0,axis=0).sum()),int((z<=0).sum()),int((z==0).sum())

def gamma(n,u):return n*u/(1-n*u)

def pin_bounds(w,x,z):
    w=np.asarray(w,np.float64);x=np.asarray(x,np.float64);z=np.asarray(z,np.float64)
    residual=np.abs(w@x.mean(axis=0))
    bound=residual+gamma(x.shape[1],2**-24)*(np.abs(w)@np.mean(np.abs(x),axis=0))+gamma(x.shape[0],2**-24)*np.mean(np.abs(z),axis=0)
    bound=bound/(1-gamma(x.size+w.size,2**-53))
    measured=np.abs(np.mean(z,axis=0))
    return measured,bound

@torch.no_grad()
def diagnostics(P,X,act,oldrows,expected_X=None):
    z1,a1,z2,a2,_=E.forward(P,X,act,train=False)
    if expected_X is not None:assert torch.equal(X,expected_X),'C input differs from independently reconstructed subset'
    zs=[z1.cpu().numpy(),z2.cpu().numpy()];xx=(expected_X if expected_X is not None else X).cpu().numpy()
    pp=[p.detach().cpu().numpy() for p in P]
    out=[];units={}
    for r,old in enumerate(oldrows):
        d={}
        for l in range(2):
            tag=f'l{l+1}';z=zs[l][r];b=pp[2*l+1][r].astype(np.float64)
            n,N,omega=weight_metrics(pp[2*l][r]);nd,ng,nzero=gate_counts(z)
            d.update({f'depth_ratio_{tag}':ratio(float(old[f'zbar_{tag}']),float(old[f'zsd_{tag}'])),f'wtilde_gmean_{tag}':N,f'omega_{tag}':omega,f'n_dead_{tag}':nd,f'n_gate0_{tag}':ng,f'n_exact_zero_{tag}':nzero,f'bias_mean_{tag}':float(b.mean()),f'bias_absmean_{tag}':float(np.abs(b).mean())})
            assert nd==round(float(old[f'dead_frac_{tag}'])*100)
            assert ng==round(float(old[f'gate_zero_frac_{tag}'])*120000)
            units[f'norm_{r}_{tag}']=n
        if act.arm=='CHB':
            assert np.all(pp[1][r]==0),'b1 not zero'
            measured,bound=pin_bounds(pp[0][r],xx[r],zs[0][r])
            d['pin_max_ratio']=float(np.max(measured/bound))
            units[f'pin_measured_{r}']=measured;units[f'pin_bound_{r}']=bound
            assert np.all(measured<=bound),'pin violated'
        else:d['pin_max_ratio']=''
        out.append(d)
    return out,units

class Observer:
    def __init__(self,out,arm,expected,seeds,epochs,stop=None,diagnose=True):
        self.out,self.arm,self.expected,self.seeds,self.epochs=out,arm,expected,seeds,epochs
        self.stop,self.diagnose=stop,diagnose
        self.extra={};self.cont={};self.started=time.time()
        if (out/'per_task.csv').exists():
            self.old=readcsv(out/'per_task.csv')
            for row in self.old:
                self.extra[(int(row['seed']),int(row['task']))]={k:v for k,v in row.items() if k.startswith(('depth_ratio','wtilde_','omega_','n_dead','n_gate0','n_exact_zero','bias_mean','bias_absmean','pin_max'))}
        else:self.old=[]
    def __call__(self,event,c):
        if event=='ready':
            actual=live_state(c);want=state(self.expected) if self.expected else actual
            self.cont=dict(expected_state=digest(want),before_first_draw=digest(actual),start_task=c['t_first'],pid=os.getpid())
            put(self.out/'continuity.json',self.cont)
            assert self.cont['expected_state']==self.cont['before_first_draw'],'continuity'
            self.nextlabs=[];self.nextorders=[]
            for s in self.seeds:
                gl=torch.Generator();gl.set_state(want['g_lab'][s]);self.nextlabs.append(E.RC.task_labels(gl))
                gb=torch.Generator();gb.set_state(want['g_batch'][s]);self.nextorders.append(torch.randperm(1200,generator=gb))
            self.cont['data_sha256']=c['cifar'].sha256
            self.cont['subset_sha256']={str(s):sha_tensor(E.RC.subset_idx(s)) for s in self.seeds}
            self.expected_X=expected_inputs(c['cifar'],self.seeds,c['device'])
            assert torch.equal(c['X'],self.expected_X),'C input changed'
            self.cont['X_sha256']=sha_tensor(c['X'])
            put(self.out/'heartbeat.json',dict(status='ready',task=c['t_first']-1,pid=os.getpid(),time=time.time()))
        elif event in ('labels','order'):
            actual=c['Y'] if event=='labels' else c['ORD']
            expected=torch.stack(self.nextlabs if event=='labels' else self.nextorders)
            self.cont[event+'_expected']=sha_tensor(expected);self.cont[event+'_actual']=sha_tensor(actual)
            put(self.out/'continuity.json',self.cont)
            assert torch.equal(actual.cpu(),expected),event
        elif event=='task_end':
            if c['diverged']:
                put(self.out/'failure.json',dict(status='NUMERICAL_FAILURE',diverged=c['diverged']))
                raise RuntimeError('NUMERICAL_FAILURE; checkpoint saved')
            if self.diagnose:
                old=readcsv(self.out/'per_task.csv'); t=c['t']
                current=sorted([r for r in old if int(r['task'])==t],key=lambda r:int(r['slot']))
                extras,units=diagnostics(c['P'],c['X'],c['act'],current,self.expected_X)
                raw=self.out/'diagnostic_units';raw.mkdir(exist_ok=True)
                np.savez(raw/f't{t:03d}.npz',**units)
                for r,d in zip(current,extras):self.extra[(int(r['seed']),t)]=d
                combined=[dict(r,**self.extra[(int(r['seed']),int(r['task']))]) for r in old]
                writecsv(self.out/'per_task.csv',combined)
                # Compare every inherited cell; new columns are deliberately excluded.
                bykey={(int(r['seed']),int(r['task'])):r for r in combined}
                for r in self.old:
                    now=bykey[(int(r['seed']),int(r['task']))]
                    assert all(now[k]==v for k,v in r.items()),'prefix cell changed'
            stop=self.stop is not None and self.stop.exists()
            put(self.out/'heartbeat.json',dict(status='stopped' if stop else 'running',task=c['t'],pid=os.getpid(),time=time.time(),elapsed_s=time.time()-self.started))
            return stop
        return False

def sha_tensor(v):return digest(v)

@torch.no_grad()
def expected_inputs(cifar,seeds,device):
    xs=[cifar.images(E.RC.subset_idx(s),device) for s in seeds]
    return torch.stack([x-x.mean(0) for x in xs])

@torch.no_grad()
def enrich_prefix(arm,cifar,device):
    out=OUT/arm;rows=readcsv(out/'per_task.csv');bykey={(int(r['seed']),int(r['task'])):r for r in rows}
    slots=[(s,'raw') for s in range(10)]
    X=expected_inputs(cifar,list(range(10)),device)
    act=E.make_act(arm,LAM[arm]);act.init_state(10,device,'prefix')
    for t in range(1,51):
        ds=[np.load(E.snapshot_path(out,arm,'raw',s,t)) for s in range(10)]
        P=[torch.stack([torch.from_numpy(d[k]) for d in ds]).to(device) for k in ('W1','b1','W2','b2','W3','b3')]
        act.load_state({'m':[torch.stack([torch.from_numpy(d[k]) for d in ds]).to(device) for k in ('m1','m2')]})
        old=[bykey[(s,t)] for s in range(10)];extra,units=diagnostics(P,X,act,old)
        for row,d in zip(old,extra):row.update(d)
        raw=out/'diagnostic_units';raw.mkdir(exist_ok=True);np.savez(raw/f't{t:03d}.npz',**units)
        for d in ds:d.close()
    writecsv(out/'per_task.csv',rows)
    prefix=readcsv(OUT/'inputs'/arm/'per_task.csv')
    assert all(all(new[k]==v for k,v in old.items()) for old,new in zip(prefix,rows))
    put(OUT/'checks'/f'prefix_{arm}.json',dict(rows=len(rows),old_cells=sum(len(r) for r in prefix),bit_equal=True,snapshot_replay=True))

if __name__=='__main__':
    if sys.argv[1:] == ['prepare']:prepare()
    else:raise SystemExit('use analysis/ch_chb_200_0919/launch.py')

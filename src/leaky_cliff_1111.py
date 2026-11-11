#!/usr/bin/env python3
"""Registered leaky_cliff_1111 CPU experiment; see specs/spec_leaky_cliff_1111.md."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, csv, gzip, hashlib, json, math, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import pmnist_0905 as H
from src import pmnist_rlmnist_0906 as RL
torch.set_num_threads(1)
torch.set_flush_denormal(True)
torch.use_deterministic_algorithms(True)
RUN='leaky_cliff_1111'
BASE=ROOT/'results'/RUN
DATA=Path('/home/issan/Projects/claude/proj_004_drift/data/mnist')
SPEC=ROOT/'specs/spec_leaky_cliff_1111.md'
PREREG='73338cd8'
LR=.001
EPS=1e-8

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvwrite(p,rows):
    if not rows:return
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def ident(args):
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',PREREG,git],cwd=ROOT,check=True)
    return dict(args=vars(args),git_hash=git,spec_sha256=sha(SPEC),code_sha256=sha(__file__),
                prereg=PREREG,torch=torch.__version__,python=platform.python_version(),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                data_sha256=sha(DATA/'train-images-idx3-ubyte.gz'))
def get_x(seed):
    arr=H._read_idx(DATA/'train-images-idx3-ubyte.gz')
    idx=RL.subset_idx(seed).numpy()
    return torch.from_numpy(arr[idx].reshape(-1,784).astype(np.float32)/255.)
def clone_ps(ps):return [p.detach().clone().requires_grad_(True) for p in ps]
def rng(state):
    g=torch.Generator();g.set_state(state);return g
def phi(z,a):return torch.where(z>0,z,a*z)
def gate(z,a):return torch.where(z>0,torch.ones_like(z),torch.full_like(z,a))

class Bwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z,fa,ba):
        ctx.save_for_backward(z);ctx.ba=ba;return phi(z,fa)
    @staticmethod
    def backward(ctx,gr):return gr*gate(ctx.saved_tensors[0],ctx.ba),None,None

class Net:
    def __init__(self,ps,a,b=None,anchor=None,old_a=None):
        self.ps=ps;self.a=[a,a] if isinstance(a,(float,int)) else list(a)
        self.b=self.a if b is None else ([b,b] if isinstance(b,(float,int)) else list(b))
        self.anchor=anchor;self.old_a=old_a
    def forward(self,x):
        zs=[];hs=[];h=x
        if self.anchor is not None:
            with torch.no_grad():
                h0=x;ref=[]
                for l in range(2):
                    z0=h0@self.anchor[2*l].T+self.anchor[2*l+1]
                    h0=phi(z0,self.old_a);ref.append((z0,h0))
        for l in range(2):
            z=h@self.ps[2*l].T+self.ps[2*l+1]
            if self.anchor is not None:
                z0,h0=ref[l];h=h0+(phi(z,self.a[l])-phi(z0,self.a[l]))
            elif self.a[l]!=self.b[l]:h=Bwd.apply(z,self.a[l],self.b[l])
            else:h=phi(z,self.a[l])
            zs.append(z);hs.append(h)
        return zs,hs,h@self.ps[4].T+self.ps[5]

def init_adam(ps):return ([torch.zeros_like(p) for p in ps],[torch.zeros_like(p) for p in ps],0)
@torch.no_grad()
def adam_step(ps,gr,adm):
    m,v,tc=adm;tc+=1;c1=1-.9**tc;c2=1-.999**tc
    for p,g,mi,vi in zip(ps,gr,m,v):
        mi.mul_(.9).add_(g,alpha=1-.9);vi.mul_(.999).addcmul_(g,g,value=1-.999)
        p-=LR*(mi/c1)/((vi/c2).sqrt()+EPS)
    return m,v,tc

def norm(ps):return float(torch.stack([p.double().square().sum() for p in ps]).sum().sqrt())
def qadd(d,key,v):
    v=v.detach().double().flatten();v=v[torch.isfinite(v)]
    if not v.numel():
        d[key+'_mean']=None;return
    d[key+'_mean']=float(v.mean())
    qs=torch.quantile(v,torch.tensor([0.,.1,.5,.9,1.],dtype=torch.double)).tolist()
    for k,val in zip(('min','q10','q50','q90','max'),qs):d[key+'_'+k]=val

@torch.no_grad()
def geometry(net,out):
    zs,hs,f=out;d={}
    for l,z in enumerate(zs):
        z=z.double();u=f'L{l+1}_';p=(z>0).double().mean(0);s=z.std(0,unbiased=False);m=z.mean(0)
        qadd(d,u+'m',m);qadd(d,u+'s',s);qadd(d,u+'r',torch.where(s>0,m/s,torch.full_like(s,float('nan'))))
        d[u+'zero_s']=int((s==0).sum());qadd(d,u+'pplus_unit',p)
        qadd(d,u+'positive_width_image',(z>0).double().sum(1))
        d[u+'positive_width_unit']=int((p>.005).sum())
        d[u+'strict_negative_frac']=float((p==0).double().mean())
        d[u+'strict_positive_frac']=float((p==1).double().mean())
        d[u+'one_sided_frac']=float((torch.minimum(p,1-p)<.01).double().mean())
        for label,a in [('true',net.a[l]),('backward',net.b[l])]:
            g=gate(z,a);g1=g.mean(0);g2=g.square().mean(0)
            d[u+label+'_gmean']=float(g1.mean());d[u+label+'_g2mean']=float(g2.mean())
            d[u+label+'_within_gvar']=float((g2-g1.square()).mean())
            d[u+label+'_zero_frac']=float((g==0).double().mean())
            ne=torch.where(g2>0,g1.square()/g2,torch.zeros_like(g1))
            qadd(d,u+label+'_neff',ne);qadd(d,u+label+'_g_unit',g1)
            qadd(d,u+label+'_g_all',g)
        d[u+'weight_norm']=norm(net.ps[l*2:l*2+1])
    return d

@torch.no_grad()
def jvp(net,x,out,dp,hidden=False):
    zs,hs,f=out
    dz1=x@dp[0].T+dp[1];dh1=gate(zs[0],net.a[0])*dz1
    dz2=dh1@net.ps[2].T+hs[0]@dp[2].T+dp[3];dh2=gate(zs[1],net.a[1])*dz2
    df=dh2@net.ps[4].T
    if not hidden:df=df+hs[1]@dp[4].T+dp[5]
    return df

@torch.no_grad()
def functional(net,x,y,before,after,dp,gr,adm,ps0):
    # J is evaluated at pre-update params. These are the TRUE forward derivatives even in B arms.
    before_net=Net(ps0,net.a,net.b,net.anchor,net.old_a)
    lin=jvp(before_net,x,before,dp);lh=jvp(before_net,x,before,dp,hidden=True)
    f0=before[2];f1=after[2];df=f1-f0
    r=torch.softmax(f0,1)-F.one_hot(y,10).to(f0.dtype)
    d={};cen=lambda f:f-f.mean(1,keepdim=True)
    for label,v in [('actual',df),('jvp',lin),('hidden_jvp',lh)]:
        cv=cen(v).double();rr=r.double();vnorm=cv.norm();rnorm=rr.norm()
        d[label+'_logit_rms']=float(cv.square().mean().sqrt())
        d[label+'_descent']=float(-(rr*cv).sum()/len(x))
        d[label+'_error_cos']=float(-(rr*cv).sum()/(rnorm*vnorm).clamp_min(1e-300))
    d['jvp_relative_error']=float((cen(df-lin).double().norm()/cen(df).double().norm().clamp_min(1e-300)))
    d['full_ce_drop']=float(F.cross_entropy(f0,y)-F.cross_entropy(f1,y))
    m,v,tc=adm;c2=1-.999**tc
    for l in range(3):
        sl=slice(2*l,2*l+2);key=f'L{l+1}_'
        d[key+'raw_grad_norm']=norm(gr[sl]);d[key+'actual_delta_norm']=norm(dp[sl])
        intended=[-LR*(mi/(1-.9**tc))/((vi/c2).sqrt()+EPS) for mi,vi in zip(m[sl],v[sl])]
        d[key+'intended_delta_norm']=norm(intended)
        d[key+'eps_frac']=float(torch.cat([(vi/c2).sqrt().flatten() for vi in v[sl]]).lt(EPS).double().mean())
    return d

@torch.no_grad()
def scale_shadow(gr,adm):
    m,v,tc=adm;newt=tc+1;res=[]
    for c in (1.,.1,.01):
        for mode in ('stale','scaled_history','fresh'):
            ups=[]
            for g,mi,vi in zip(gr,m,v):
                mt=mi if mode=='stale' else (mi*c if mode=='scaled_history' else torch.zeros_like(mi))
                vt=vi if mode=='stale' else (vi*c*c if mode=='scaled_history' else torch.zeros_like(vi))
                t=1 if mode=='fresh' else newt
                mn=.9*mt+.1*c*g;vn=.999*vt+.001*(c*g).square()
                ups.append(-LR*(mn/(1-.9**t))/((vn/(1-.999**t)).sqrt()+EPS))
            res.append(dict(c=c,mode=mode,**{f'L{l+1}_delta_norm':norm(ups[2*l:2*l+2]) for l in range(3)}))
    return res

def train_task(net,adm,x,y,g_batch,steps,task,curves,diags,shadows):
    assert steps%75==0
    dense={1,75}|set(range(750,steps+1,750))
    ce=[];ac=[];t90=None;t80=None;tce=None
    with torch.no_grad():
        oo=net.forward(x);diags.append(dict(task=task,step=0,**geometry(net,oo)))
        curves.append(dict(task=task,step=0,acc=float((oo[2].argmax(1)==y).float().mean()),ce=float(F.cross_entropy(oo[2],y))))
    for ep in range(steps//75):
        order=torch.randperm(len(x),generator=g_batch);xs=x[order];ys=y[order]
        for j in range(75):
            s=ep*75+j+1
            if s in dense:
                with torch.no_grad():before=net.forward(x);ps0=[p.detach().clone() for p in net.ps]
            out=net.forward(xs[j*16:(j+1)*16]);loss=F.cross_entropy(out[2],ys[j*16:(j+1)*16])
            ce.append(float(loss.detach()));ac.append(float((out[2].detach().argmax(1)==ys[j*16:(j+1)*16]).float().mean()))
            grads=torch.autograd.grad(loss,net.ps)
            if s==1:
                shadows.extend([dict(task=task,step=s,**r) for r in scale_shadow(grads,adm)])
            adm=adam_step(net.ps,grads,adm)
            if s in dense:
                with torch.no_grad():
                    after=net.forward(x);dp=[p.detach()-p0 for p,p0 in zip(net.ps,ps0)]
                    if net.anchor is None and net.a!=net.b:
                        assert torch.equal(after[2],Net(net.ps,net.a).forward(x)[2]),'B forward mismatch'
                    diags.append(dict(task=task,step=s,**geometry(net,after),**functional(net,x,y,before,after,dp,grads,adm,ps0)))
        with torch.no_grad():
            oo=net.forward(x);acc=float((oo[2].argmax(1)==y).float().mean());cee=float(F.cross_entropy(oo[2],y))
        assert math.isfinite(cee),('NONFINITE',task,ep)
        step=(ep+1)*75
        curves.append(dict(task=task,step=step,acc=acc,ce=cee,online_acc=float(np.mean(ac[-75:])),online_ce=float(np.mean(ce[-75:]))))
        if t90 is None and acc>=.9:t90=step
        if t80 is None and acc>=.8:t80=step
        if tce is None and cee<=.5:tce=step
    summary=dict(task=task,steps=steps,online_acc=float(np.mean(ac)),online_ce=float(np.mean(ce)),endpoint_acc=acc,endpoint_ce=cee,
                 T90=t90,T80=t80,Tce05=tce,T90_censored=int(t90 is None))
    for b in (6000,24000):
        if b<=steps:
            rr=next(r for r in curves if r['task']==task and r['step']==b)
            summary.update({f'acc_{b}':rr['acc'],f'ce_{b}':rr['ce'],f'online_acc_first{b}':float(np.mean(ac[:b]))})
    return adm,summary

def save_state(path,net,adm,gl,gb,x,task,seed,a):
    with torch.no_grad():out=net.forward(x)
    torch.save(dict(params=[p.detach().clone() for p in net.ps],m=adm[0],v=adm[1],tc=adm[2],
                    g_lab=gl.get_state(),g_batch=gb.get_state(),task=task,seed=seed,a=a,z=out[0]),path)
    np.savez_compressed(path.with_suffix('.npz'),z1=out[0][0].numpy(),z2=out[0][1].numpy(),
                        g1=gate(out[0][0],a).numpy(),g2=gate(out[0][1],a).numpy())

def series(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    identity=ident(args);put(out/'provenance.json',identity)
    x=get_x(args.seed);ps=H.init_params(args.seed,torch.device('cpu'));net=Net(ps,args.a)
    adm=init_adam(ps);gl=H.stream('rl_labels',args.seed);gb=H.stream('rl_batch',args.seed)
    rows=[];curves=[];diags=[];shadows=[];t0=time.time()
    for t in range(1,args.tasks+1):
        y=RL.task_labels(gl);adm,row=train_task(net,adm,x,y,gb,args.steps,t,curves,diags,shadows);rows.append(row)
        if t in (2,5,10,15,20):save_state(out/f'state_t{t}.pt',net,adm,gl,gb,x,t,args.seed,args.a)
        csvwrite(out/'tasks.csv',rows);csvwrite(out/'curves.csv',curves);csvwrite(out/'diagnostics.csv',diags)
        csvwrite(out/'scale_shadow.csv',shadows)
        print(f'seed={args.seed} a={args.a} task={t} online={row["online_acc"]:.3f} end={row["endpoint_acc"]:.3f} elapsed={time.time()-t0:.1f}',flush=True)
    put(out/'done.json',dict(seconds=time.time()-t0,tasks=len(rows),identity=identity))

def branch(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);identity=ident(args);put(out/'provenance.json',identity)
    ck=torch.load(args.source,map_location='cpu',weights_only=False);old=ck['a'];target=args.a
    x=get_x(args.seed);ps=clone_ps(ck['params']);adm=([v.clone() for v in ck['m']],[v.clone() for v in ck['v']],ck['tc'])
    gl=rng(ck['g_lab']);gb=rng(ck['g_batch']);y=RL.task_labels(gl)
    mode=args.arm
    if mode in ('Fbad','Fgood'):ps=H.init_params(args.seed,torch.device('cpu'));adm=init_adam(ps)
    if mode in ('O','H','BO'):adm=init_adam(ps)
    if mode=='H':
        with torch.no_grad():ps[4].zero_();ps[5].zero_()
    if mode in ('B','BO','Breverse'):net=Net(ps,old,target)
    elif mode=='B1':net=Net(ps,old,[target,old])
    elif mode=='B2':net=Net(ps,old,[old,target])
    elif mode in ('X','Xreverse'):net=Net(ps,target,anchor=[p.detach().clone() for p in ck['params']],old_a=old)
    elif mode in ('A','Fgood'):net=Net(ps,target)
    else:net=Net(ps,old)
    with torch.no_grad():
        natural=Net(ps,old).forward(x)[2];initial=net.forward(x)[2]
        diff=float((initial-natural).abs().max());same=torch.equal(natural,initial)
    if mode in ('B','BO','B1','B2','Breverse','X','Xreverse'):
        assert same,('branch forward mismatch',mode,diff)
    put(out/'branch_check.json',dict(initial_natural_logit_equal=same,initial_maxabs=diff,source_sha256=sha(args.source),old_a=old,target_a=target))
    curves=[];diags=[];shadows=[];t0=time.time()
    adm,row=train_task(net,adm,x,y,gb,args.steps,21,curves,diags,shadows)
    csvwrite(out/'tasks.csv',[row]);csvwrite(out/'curves.csv',curves);csvwrite(out/'diagnostics.csv',diags);csvwrite(out/'scale_shadow.csv',shadows)
    torch.save(dict(params=[p.detach() for p in ps],m=adm[0],v=adm[1],tc=adm[2]),out/'final.pt')
    put(out/'done.json',dict(seconds=time.time()-t0,identity=identity))
    print(mode,args.seed,row,flush=True)

def checks(args):
    torch.manual_seed(9921);p=[p.double().requires_grad_(True) for p in H.init_params(100,torch.device('cpu'))]
    x=torch.randn(7,784,dtype=torch.double);a=.1;net=Net(p,a);out=net.forward(x)
    dp=[torch.randn_like(v)*.001 for v in p];lin=jvp(net,x,out,dp)
    h=1e-4
    plus=Net([v+h*d for v,d in zip(p,dp)],a).forward(x)[2]
    minus=Net([v-h*d for v,d in zip(p,dp)],a).forward(x)[2]
    jerr=float(((plus-minus)/(2*h)-lin).abs().max());assert jerr<1e-8
    z=torch.randn(50,dtype=torch.double,requires_grad=True)
    g=torch.autograd.grad(phi(z,a).sum(),z)[0];gn=torch.autograd.grad(F.leaky_relu(z,a).sum(),z)[0]
    assert torch.equal(g,gn)
    bn=Net(p,a,.3);bo=bn.forward(x);assert torch.equal(out[2],bo[2])
    gb=torch.autograd.grad(bo[2].square().sum(),p);g0=torch.autograd.grad(out[2].square().sum(),p)
    assert not torch.equal(gb[0],g0[0])
    xn=Net(p,.3,anchor=[q.detach().clone() for q in p],old_a=a);xo=xn.forward(x)
    assert torch.equal(xo[2],out[2])
    gx=torch.autograd.grad(xo[2].square().sum(),p)
    assert all(torch.equal(u,v) for u,v in zip(gb,gx))
    pp=clone_ps(p);qq=clone_ps(p);ad=init_adam(pp);ma,va,tc=init_adam(qq)
    for i in range(4):
        gg=[torch.randn_like(v) for v in pp];ad=adam_step(pp,gg,ad);tc+=1
        with torch.no_grad():
            for v,gr,mi,vi in zip(qq,gg,ma,va):
                mi.mul_(.9).add_(gr,alpha=1-.9);vi.mul_(.999).addcmul_(gr,gr,value=1-.999)
                v-=LR*(mi/(1-.9**tc))/((vi/(1-.999**tc)).sqrt()+EPS)
    # float64 1-.9 vs .1 differs in last bits; the experiment uses the host constants below.
    aderr=max(float((u-v).abs().max()) for u,v in zip(pp,qq));assert aderr<1e-12
    # Save/reload all optimizer and RNG state, next actual update identical.
    tmp=Path(args.out);tmp.mkdir(parents=True,exist_ok=True)
    torch.save(dict(ps=pp,adm=ad,gen=H.stream('rl_batch',100).get_state()),tmp/'restart.pt')
    loaded=torch.load(tmp/'restart.pt',weights_only=False);p2=clone_ps(loaded['ps']);ad2=loaded['adm']
    gg=[torch.randn_like(v) for v in pp];ad=adam_step(pp,gg,ad);ad2=adam_step(p2,gg,ad2)
    assert all(torch.equal(u,v) for u,v in zip(pp,p2))
    assert torch.equal(torch.randperm(1200,generator=rng(loaded['gen'])),torch.randperm(1200,generator=H.stream('rl_batch',100)))
    put(tmp/'checks.json',dict(all_pass=True,jvp_maxabs=jerr,adam_host_maxabs=aderr,backward_equals_anchored_gradient=True,source_sha256=sha(__file__)))
    print('checks pass',jerr,aderr)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['series','branch','checks']);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--a',type=float,default=.1);ap.add_argument('--tasks',type=int,default=20);ap.add_argument('--steps',type=int,default=6000)
    ap.add_argument('--arm',default='N');ap.add_argument('--source');ap.add_argument('--out',required=True);args=ap.parse_args()
    {'series':series,'branch':branch,'checks':checks}[args.mode](args)
if __name__=='__main__':main()

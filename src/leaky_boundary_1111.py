#!/usr/bin/env python3
"""Boundary clocks and fixed-gate/fixed-Jacobian controls; frozen spec in specs/."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
import argparse,csv,hashlib,json,math,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import leaky_cliff_1111 as C
H=C.H;RL=C.RL
torch.set_num_threads(1);torch.set_flush_denormal(True);torch.use_deterministic_algorithms(True)
RUN='leaky_boundary_1111';BASE=ROOT/'results'/RUN;SPEC=ROOT/'specs/spec_leaky_boundary_1111.md'
A=.003
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,obj):C.put(p,obj)
def write(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r));p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def ident(args):
    return dict(args=vars(args),git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                code_sha256=sha(__file__),spec_sha256=sha(SPEC),dependency_sha256=sha(ROOT/'src/leaky_cliff_1111.py'),
                data_sha256=sha(C.DATA/'train-images-idx3-ubyte.gz'),torch=torch.__version__,python=platform.python_version())
def copy_adm(ck):return ([p.clone() for p in ck['m']],[p.clone() for p in ck['v']],ck['tc'])
def state(ps,adm,age,seed,gl=None,gb=None):
    return dict(params=[p.detach().clone() for p in ps],m=adm[0],v=adm[1],tc=adm[2],age=age,seed=seed,a=A,
                g_lab=gl.get_state() if gl is not None else None,g_batch=gb.get_state() if gb is not None else None)
def probe_labels(seed):return RL.task_labels(H.stream('boundary_probe_labels',seed))
def tag(age):return format(age,'.2f').replace('.','p')

class Model:
    def __init__(self,ps,x,arm='N',a=A):
        self.arm=arm;self.a=a;self.p0=[p.detach().clone() for p in ps]
        self.ps=C.clone_ps(ps)
        with torch.no_grad():
            self.z0,self.h0,self.f0=C.Net(self.p0,a).forward(x)
            self.g0=[C.gate(z,a) for z in self.z0]
    def forward(self,x,ids=None):
        if self.arm=='N':return C.Net(self.ps,self.a).forward(x)
        sl=slice(None) if ids is None else ids
        if ids is None:
            zref,href,fref=self.z0,self.h0,self.f0
        else:
            with torch.no_grad():zref,href,fref=C.Net(self.p0,self.a).forward(x)
        gg=[C.gate(z,self.a) for z in zref]
        if self.arm=='G':
            h=x;zs=[];hs=[]
            for l in range(2):
                z=h@self.ps[2*l].T+self.ps[2*l+1];h=z*gg[l];zs.append(z);hs.append(h)
            return zs,hs,h@self.ps[4].T+self.ps[5]
        dp=[p-q for p,q in zip(self.ps,self.p0)]
        dz1=x@dp[0].T+dp[1];dh1=gg[0]*dz1
        dz2=dh1@self.p0[2].T+href[0]@dp[2].T+dp[3];dh2=gg[1]*dz2
        df=dh2@self.p0[4].T+href[1]@dp[4].T+dp[5]
        return [zref[0]+dz1,zref[1]+dz2],[href[0]+dh1,href[1]+dh2],fref+df
    def gates(self,zs):return [C.gate(z,self.a) for z in zs] if self.arm=='N' else self.g0

@torch.no_grad()
def part_cov(x):
    x=x.double();x=x-x.mean(0);cov=x.T@x/len(x);ev=torch.linalg.eigvalsh(cov).clamp_min(0)
    tr=ev.sum();return dict(participation=float(tr.square()/ev.square().sum().clamp_min(1e-300)),trace=float(tr),
                           numerical_rank=int((ev>ev.max()*1e-6).sum()) if float(ev.max())>0 else 0),ev.cpu().numpy()

@torch.no_grad()
def measure(net,out,y,previous=None,expensive=False):
    zs,hs,f=out;used=net.gates(zs);d={};masks=[]
    d['acc']=float((f.argmax(1)==y).float().mean());d['ce']=float(F.cross_entropy(f,y))
    other=f.clone();other.scatter_(1,y[:,None],-float('inf'));margin=f.gather(1,y[:,None]).flatten()-other.max(1).values
    d['margin_mean']=float(margin.mean());d['margin_near0_frac']=float((margin.abs()<.1).float().mean())
    for q in [.1,.5,.9]:d[f'margin_q{int(q*100)}']=float(torch.quantile(margin,q))
    for l,(z,h,g) in enumerate(zip(zs,hs,used)):
        key=f'L{l+1}_';mask=(g>.5);masks.append(mask);p=mask.double().mean(0);sd=z.double().std(0,unbiased=False);m=z.double().mean(0)
        r=torch.where(sd>0,m/sd,torch.full_like(sd,float('nan')))
        d.update({key+'pplus':float(p.mean()),key+'gmean':float(g.double().mean()),key+'gvar':float(g.double().var(0,unbiased=False).mean()),
                  key+'one_sided':float((torch.minimum(p,1-p)<.01).double().mean()),key+'zero_positive_input':float((mask.sum(1)==0).double().mean()),
                  key+'strict_negative_unit':float((p==0).double().mean()),key+'m':float(m.mean()),key+'s':float(sd.mean()),
                  key+'r_median':float(r[torch.isfinite(r)].median()) if torch.isfinite(r).any() else None,
                  key+'z_positive':float((z>0).double().mean()),key+'mask_turnover':float((mask!=previous[l]).double().mean()) if previous is not None else None})
        for q in [.1,.5,.9]:d[key+f'pplus_q{int(100*q)}']=float(torch.quantile(p,q))
        if expensive:
            cov,_=part_cov(mask);hc,_=part_cov(h)
            d.update({key+'gate_'+k:v for k,v in cov.items()});d.update({key+'feature_'+k:v for k,v in hc.items()})
            d[key+'unique_gate_patterns']=len(np.unique(mask.cpu().numpy(),axis=0))
    assert math.isfinite(d['ce'])
    return d,masks

def fit(net,adm,x,y,gb,steps,dr,task=0,callback=None):
    rows=[];ac=[];losses=[];previous=None;t0=time.time();ts={};streak=0
    with torch.no_grad():
        out=net.forward(x);row,previous=measure(net,out,y,expensive=True);rows.append(dict(task=task,step=0,**row))
        np.savez_compressed(dr/f'logits_t{task}_s0.npz',logits=out[2].numpy(),labels=y.numpy())
    for ep in range(steps//75):
        order=torch.randperm(len(x),generator=gb)
        for j in range(75):
            step=ep*75+j+1;ids=order[j*16:(j+1)*16];dense=step==1 or step%750==0
            if dense:
                with torch.no_grad():before=net.forward(x);ps0=[p.detach().clone() for p in net.ps]
            out=net.forward(x[ids],ids);loss=F.cross_entropy(out[2],y[ids]);hit=float((out[2].detach().argmax(1)==y[ids]).float().mean())
            ac.append(hit);losses.append(float(loss.detach()));gr=torch.autograd.grad(loss,net.ps);adm=C.adam_step(net.ps,gr,adm)
            if step==1 or step%75==0:
                with torch.no_grad():
                    after=net.forward(x);row,previous=measure(net,after,y,previous,expensive=dense)
                    row.update(task=task,step=step,online_acc=float(np.mean(ac[-75:])),online_ce=float(np.mean(losses[-75:])))
                    if dense:
                        delta=[p.detach()-q for p,q in zip(net.ps,ps0)];df=after[2]-before[2];df-=df.mean(1,keepdim=True)
                        err=torch.softmax(before[2],1)-F.one_hot(y,10).to(x.dtype)
                        row['actual_logit_rms']=float(df.double().square().mean().sqrt());row['actual_descent']=float(-(err*df).sum()/len(x))
                        row['actual_error_cos']=float(-(err*df).sum()/(err.norm()*df.norm()).clamp_min(1e-30))
                        row['full_ce_drop']=float(F.cross_entropy(before[2],y)-F.cross_entropy(after[2],y))
                        for l in range(3):
                            sl=slice(2*l,2*l+2);row[f'L{l+1}_grad_norm']=C.norm(gr[sl]);row[f'L{l+1}_delta_norm']=C.norm(delta[sl])
                            mm,vv,tc=adm;intended=[-C.LR*(m/(1-.9**tc))/((v/(1-.999**tc)).sqrt()+C.EPS) for m,v in zip(mm[sl],vv[sl])]
                            row[f'L{l+1}_intended_norm']=C.norm(intended)
                    rows.append(row)
                    if step%75==0:
                        for threshold in [2.,1.5,1.,.5,.2,.1]:
                            if row['ce']<=threshold:ts.setdefault('Tce'+str(threshold),step)
                        if row['acc']>=.9:ts.setdefault('T90',step);streak+=1
                        else:streak=0
                        if streak>=4:ts.setdefault('T90_sustained',step-3*75)
                if step in [6000,12000,24000,48000,96000]:
                    np.savez_compressed(dr/f'logits_t{task}_s{step}.npz',logits=after[2].numpy(),labels=y.numpy())
            if callback is not None and step%1500==0:callback(step,net,adm)
        if (ep+1)%80==0:write(dr/f'curves_t{task}.csv',rows)
    write(dr/f'curves_t{task}.csv',rows)
    summary=dict(task=task,steps=steps,online_acc=float(np.mean(ac)),online_ce=float(np.mean(losses)),endpoint_acc=rows[-1]['acc'],endpoint_ce=rows[-1]['ce'],seconds=time.time()-t0,
                 **{k:ts.get(k) for k in ['T90','T90_sustained','Tce2.0','Tce1.5','Tce1.0','Tce0.5','Tce0.2','Tce0.1']})
    for b in [6000,12000,24000,48000,96000]:
        if b<=steps:
            r=next(r for r in rows if r['step']==b);summary.update({f'acc_{b}':r['acc'],f'ce_{b}':r['ce']})
    return adm,summary

def prefix(args):
    dr=Path(args.out);dr.mkdir(parents=True,exist_ok=True);identity=ident(args);put(dr/'provenance.json',identity)
    x=C.get_x(args.seed);ps=H.init_params(args.seed,torch.device('cpu'));net=Model(ps,x);adm=C.init_adam(net.ps)
    gl=H.stream('rl_labels',args.seed);gb=H.stream('rl_batch',args.seed);rows=[]
    torch.save(state(net.ps,adm,0,args.seed,gl,gb),dr/'state_0p00.pt')
    for task in range(1,args.tasks+1):
        y=RL.task_labels(gl)
        def callback(step,nn,ad):
            age=task-1+step/6000;torch.save(state(nn.ps,ad,age,args.seed,gl,gb),dr/f'state_{tag(age)}.pt')
        adm,row=fit(net,adm,x,y,gb,6000,dr,task,callback);rows.append(row);write(dr/'tasks.csv',rows)
        print('prefix',args.seed,task,row['online_acc'],row['endpoint_acc'],flush=True)
    drop=[rows[j-1]['online_acc']-rows[j]['online_acc'] for j in range(1,len(rows))]
    if drop:
        j=int(np.argmax(drop))+1;put(dr/'selection.json',dict(seed=args.seed,task_before=j,task_after=j+1,pre_age=j-1,post_age=j,drop=drop[j-1],drop_ge_15=drop[j-1]>=.15))
    else:put(dr/'selection.json',dict(seed=args.seed,short_smoke_no_boundary=True))
    put(dr/'done.json',dict(identity=identity))

def probe(args):
    dr=Path(args.out);dr.mkdir(parents=True,exist_ok=True);identity=ident(args);put(dr/'provenance.json',identity)
    ck=torch.load(args.source,map_location='cpu',weights_only=False);assert ck['seed']==args.seed
    x=C.get_x(args.seed);net=Model(ck['params'],x,args.arm);adm=copy_adm(ck);y=probe_labels(args.seed);gb=H.stream('boundary_probe_batch',args.seed)
    with torch.no_grad():
        nat=C.Net(ck['params'],A).forward(x)[2];actual=net.forward(x)[2];assert torch.equal(nat,actual)
    put(dr/'branch_check.json',dict(initial_forward_equal=True,source_sha256=sha(args.source),age=ck['age'],probe_labels_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
    adm,row=fit(net,adm,x,y,gb,args.steps,dr);row.update(arm=args.arm,seed=args.seed,age=ck['age']);write(dr/'summary.csv',[row])
    torch.save(state(net.ps,adm,ck['age'],args.seed),dr/'final.pt');put(dr/'done.json',dict(identity=identity))
    print('probe',args.arm,args.seed,ck['age'],row,flush=True)

def checks(args):
    dr=Path(args.out);dr.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(991);x=torch.randn(17,7);ps=[torch.randn(5,7)*.1,torch.randn(5)*.1,torch.randn(4,5)*.1,torch.randn(4)*.1,torch.randn(10,4)*.1,torch.randn(10)*.1]
    y=torch.randint(10,(17,));models={a:Model(ps,x,a) for a in ['N','G','L']};outs={a:n.forward(x)[2] for a,n in models.items()}
    assert all(torch.equal(outs['N'],o) for o in outs.values())
    grads={a:torch.autograd.grad(F.cross_entropy(outs[a],y),n.ps) for a,n in models.items()}
    err=max(float((u-v).abs().max()) for a in ['G','L'] for u,v in zip(grads['N'],grads[a]));assert err<2e-8
    base_m=[torch.randn_like(p)*.01 for p in ps];base_v=[torch.rand_like(p)*.01 for p in ps]
    deltas={}
    for a,n in models.items():
        old=[p.detach().clone() for p in n.ps];C.adam_step(n.ps,grads[a],([p.clone() for p in base_m],[p.clone() for p in base_v],100))
        deltas[a]=[p.detach()-q for p,q in zip(n.ps,old)]
    # All models store theta coordinates so the first realized update can be compared directly.
    derr=max(float((u-v).abs().max()) for a in ['G','L'] for u,v in zip(deltas['N'],deltas[a]));assert derr<2e-8
    lin=models['L'];before=lin.forward(x)[2].detach();old=[p.detach().clone() for p in lin.ps]
    dd=[torch.randn_like(p)*.01 for p in lin.ps]
    with torch.no_grad():
        for p,d in zip(lin.ps,dd):p.add_(d)
    ref=C.Net(ps,A);expected=C.jvp(ref,x,ref.forward(x),[p.detach()-q for p,q in zip(lin.ps,old)])
    lerr=float((lin.forward(x)[2].detach()-before-expected).abs().max());assert lerr<1e-7
    # Exact native update and RNG restart, compared with previous runner.
    xx=C.get_x(100);p=H.init_params(100,torch.device('cpu'));n=Model(p,xx);oldnet=C.Net(C.clone_ps(p),A)
    ad=C.init_adam(n.ps);ad0=C.init_adam(oldnet.ps);gb=H.stream('rl_batch',100);yy=RL.task_labels(H.stream('rl_labels',100));order=torch.randperm(1200,generator=gb)
    for j in range(4):
        ids=order[j*16:(j+1)*16]
        g=torch.autograd.grad(F.cross_entropy(n.forward(xx[ids],ids)[2],yy[ids]),n.ps)
        g0=torch.autograd.grad(F.cross_entropy(oldnet.forward(xx[ids])[2],yy[ids]),oldnet.ps)
        ad=C.adam_step(n.ps,g,ad);ad0=C.adam_step(oldnet.ps,g0,ad0)
    assert all(torch.equal(p,q) for p,q in zip(n.ps,oldnet.ps))
    # Real minibatch shape: recomputed frozen references must match native GEMM arithmetic.
    models={arm:Model(n.ps,xx,arm) for arm in ['N','G','L']};ids=order[64:80];grs={};changes={}
    for arm,model in models.items():
        f=model.forward(xx[ids],ids)[2]
        if arm=='N':ref=f.detach().clone()
        else:assert torch.equal(f,ref)
        grs[arm]=torch.autograd.grad(F.cross_entropy(f,yy[ids]),model.ps)
        before=[p.detach().clone() for p in model.ps]
        C.adam_step(model.ps,grs[arm],([v.clone() for v in ad[0]],[v.clone() for v in ad[1]],ad[2]))
        changes[arm]=[p.detach()-q for p,q in zip(model.ps,before)]
    realg=max(float((u-v).abs().max()) for arm in ['G','L'] for u,v in zip(grs['N'],grs[arm]))
    reald=max(float((u-v).abs().max()) for arm in ['G','L'] for u,v in zip(changes['N'],changes[arm]));assert realg==0 and reald==0,(realg,reald)
    ck=state(n.ps,ad,0,100,gb,gb);torch.save(ck,dr/'restart.pt');ck=torch.load(dr/'restart.pt',weights_only=False)
    nn=Model(ck['params'],xx);ad2=copy_adm(ck);ids=order[64:80]
    for model,adm in [(n,ad),(nn,ad2)]:
        g=torch.autograd.grad(F.cross_entropy(model.forward(xx[ids],ids)[2],yy[ids]),model.ps);C.adam_step(model.ps,g,adm)
    assert all(torch.equal(p,q) for p,q in zip(n.ps,nn.ps))
    assert torch.equal(C.rng(ck['g_batch']).get_state(),gb.get_state())
    put(dr/'checks.json',dict(all_pass=True,initial_gradient_maxabs=err,initial_actual_delta_maxabs=derr,linearized_finite_delta_maxabs=lerr,native_host_exact=True,restart_exact=True,real_minibatch_gradient_maxabs=realg,real_minibatch_delta_maxabs=reald,code_sha256=sha(__file__)))
    print('training checks pass',err,derr,lerr)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prefix','probe','checks']);ap.add_argument('--seed',type=int,default=100)
    ap.add_argument('--tasks',type=int,default=6);ap.add_argument('--steps',type=int,default=24000);ap.add_argument('--source');ap.add_argument('--arm',choices=['N','G','L'],default='N');ap.add_argument('--out',required=True)
    a=ap.parse_args();{'prefix':prefix,'probe':probe,'checks':checks}[a.mode](a)
if __name__=='__main__':main()

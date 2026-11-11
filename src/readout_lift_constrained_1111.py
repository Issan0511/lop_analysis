#!/usr/bin/env python3
"""Follow-up: exact gradient through prescribed negative-body means."""
import argparse,concurrent.futures,copy,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
import stage1_ctrl_1111 as E
import demand_allocation_1111 as D
import readout_lift_1111 as R

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
ARMS=('C1','C01','C01U','C01R')

def beta(P,bx,reference):return (reference-(P[0].double()*bx).sum(1)).float()

def set_beta(P,bx,reference):
    with torch.no_grad():P[1].copy_(beta(P,bx,reference))

def step(P,opt,tr,xb,Y,bx,ref,nextref,capture=False):
    z=xb@P[0].T+beta(P,bx,ref);a=E.activ(z,'ELU');ell=a@P[2].T+P[3]
    loss=(ell.logsumexp(-1)-(ell*Y).sum(-1)).mean();opt.zero_grad(set_to_none=True);loss.backward()
    with torch.no_grad():
        G=torch.where(z>0,torch.ones_like(z),a+1);dy=((.1-Y)@P[2])*G/len(xb)
        gy=dy.T@xb-(dy.sum(0).double()[:,None]*bx).float()
        tr['label'].mul_(.9).add_(gy,alpha=.1)
        tr['uniform'].mul_(.9).add_(P[0].grad-gy,alpha=.1)
        if capture:before=P[0].detach().double().clone()
    opt.step();set_beta(P,bx,nextref)
    if not capture:return
    with torch.no_grad():
        st=opt.state[P[0]];t=float(st['step']);denom=(st['exp_avg_sq'].double()/(1-.999**t)).sqrt()+1e-8
        coef=-.001/(1-.9**t)/denom
        parts={k:coef*v.double() for k,v in tr.items()}
        upd=P[0].detach().double()-before;parts['rounding']=upd-sum(parts.values())
        me=float((sum(tr.values())-st['exp_avg']).abs().max());re=float(parts['rounding'].abs().max())
        assert me<3e-6 and re<4e-6*max(1,float(before.abs().max())),(me,re)
        return parts,me,re

def ybatch(arm,Y,idx,fresh,t):
    if arm=='C01U':return torch.full((len(idx),10),.1)
    if arm=='C01R':return torch.nn.functional.one_hot(fresh[t],10).float()
    return Y[idx]

def run(seed,task,rawstr,refsstr,steps,git_hash):
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    start=time.monotonic();out=Path(rawstr)/f's{seed}_t{task}';out.mkdir(parents=True,exist_ok=False)
    source=R.ARCH/f'ELU_s{seed}'/f'baseline_before_t{task}.pt';base=torch.load(source,weights_only=False,map_location='cpu')
    refpath=Path(refsstr)/f's{seed}_t{task}'/'V1_trace.npz';ref=torch.from_numpy(np.load(refpath)['body'])[:steps+1]
    X,subset=E.load_X(seed);assert np.array_equal(subset,base['subset']);y=base['y'];Y=torch.nn.functional.one_hot(y,10).float()
    order=base['order'][:steps];fresh=torch.randint(10,(steps,16),generator=R.private(seed,task))
    F0=D.fields(base['P'],X,y,'ELU');F0['upper']=torch.zeros_like(F0['z'],dtype=torch.bool)
    F0['upper'].scatter_(0,torch.argsort(F0['z'],dim=0,stable=True)[-60:],True)
    bx,eligible=R.body_setup(X,F0);assert bool(eligible.all());F0['body']=R.body_mean(base['P'],bx);F0['h']=F0['z'].max(0).values.double()
    rows={'state':[],'demand':[],'source':[]};checks=[];ctx=E.Ctx(X,'ELU')
    for arm in ARMS:
        P,opt=E.restore((base['P'],base['opt']),'adam',.001)
        if arm!='C1':
            with torch.no_grad():
                old=P[2].clone();abar=E.mean_act(P,ctx);P[2].copy_(old.mean(0)+.1*(old-old.mean(0)));E.fix_bias(P,old,abar)
        opt.state.pop(P[0],None);P[1].requires_grad_(False);P[2].requires_grad_(False)
        V0=P[2].detach().clone();set_beta(P,bx,ref[0]);body_error=0.;unit={};bm=[]
        zinit=X@P[0].detach().T+P[1].detach();initerr=float((zinit-F0['z']).abs().max());assert initerr<2e-5
        tr={k:torch.zeros_like(P[0]) for k in ('label','uniform')}
        for t in range(steps+1):
            body=R.body_mean(P,bx);bm.append(body.numpy());er=float((body-ref[t]).abs().max());body_error=max(body_error,er)
            assert er<2e-5*max(1,float(ref[t].abs().max())),er
            if t in R.TIMES or t==steps:
                meta=dict(seed=seed,task=task,arm=arm,step=t)
                F=D.fields(P,X,y,'ELU');R.stats(P,F,F0,bx,eligible,X,y,meta,rows,unit)
                # R.stats raw-gradient diagnostics use the unconstrained coordinates; do not publish them here.
                if t in (0,200,500,1000,2000) and t<steps:
                    Pc,oc=E.restore(E.snapshot(P,opt),'adam',.001);Pc[1].requires_grad_(False);Pc[2].requires_grad_(False)
                    tc=copy.deepcopy(tr);idx=order[t];yb=ybatch(arm,Y,idx,fresh,t)
                    parts,me,re=step(Pc,oc,tc,X[idx],yb,bx,ref[t],ref[t+1],True)
                    for group,m in R.groups(F,F0,y,P[2].detach()).items():
                        n=m.sum(0);xm=m.T.double()@X.double()/n.clamp_min(1)[:,None]-bx
                        values={k:((xm*v).sum(1)*n).sum()/n.sum() for k,v in parts.items()}
                        values['reference']=((ref[t+1]-ref[t])*n).sum()/n.sum()
                        values['total']=sum(values.values())
                        for k,v in values.items():rows['source'].append(dict(**meta,group=group,source=k,dz=float(v),moment_error=me,step_rounding=re))
            if t<steps:
                idx=order[t];step(P,opt,tr,X[idx],ybatch(arm,Y,idx,fresh,t),bx,ref[t],ref[t+1])
        assert torch.equal(P[2],V0)
        sp,so=E.snapshot(P,opt);torch.save(dict(P=sp,opt=so,trace=tr,y=y),out/f'{arm}_end.pt')
        np.savez_compressed(out/f'{arm}_trace.npz',body=np.array(bm),**{f'{t}_{k}':v.numpy() for t,d in unit.items() for k,v in d.items()})
        checks.append(dict(arm=arm,initial_z_error=initerr,body_error=body_error,fixed_V_error=0.))
        for k in ('state','source'):R.csvwrite(out/f'{k}.csv',rows[k])
        print(f'followup seed={seed} task={task} arm={arm} done seconds={time.monotonic()-start:.1f}',flush=True)
    meta=dict(seed=seed,task=task,steps=steps,git_hash=git_hash,code_sha256=R.sha(__file__),helper_sha256=R.sha(R.__file__),engine_sha256=R.sha(E.__file__),
        spec_sha256=R.sha(ROOT/'specs/spec_readout_lift_1111_addendum.md'),source_sha256=R.sha(source),reference_sha256=R.sha(refpath),checks=checks,seconds=time.monotonic()-start)
    R.js(out/'complete.json',meta);return meta

def synthetic(out):
    torch.manual_seed(12011);X=torch.randn(13,7,dtype=torch.float64);bx=torch.randn(5,7,dtype=torch.float64)
    W=torch.randn(5,7,dtype=torch.float64,requires_grad=True);V=torch.randn(10,5,dtype=torch.float64);b=torch.randn(10,dtype=torch.float64);ref=torch.randn(5,dtype=torch.float64)
    Y=torch.nn.functional.one_hot(torch.randint(10,(13,)),10).double()
    z=X@W.T-(W*bx).sum(1)+ref;z.retain_grad();a=E.activ(z,'ELU');ell=a@V.T+b;loss=(ell.logsumexp(-1)-(ell*Y).sum(-1)).mean();loss.backward()
    manual=z.grad.T@X-z.grad.sum(0)[:,None]*bx;er=float((manual-W.grad).abs().max());assert er<1e-10
    v=torch.randn_like(W);v/=v.norm();eps=1e-5
    def value(w):
        q=E.activ(X@w.T-(w*bx).sum(1)+ref,'ELU')@V.T+b
        return (q.logsumexp(-1)-(q*Y).sum(-1)).mean()
    fd=abs(float((value(W.detach()+eps*v)-value(W.detach()-eps*v))/(2*eps)-(W.grad*v).sum()));assert fd<1e-7
    R.js(out,dict(gradient_error=er,finite_difference_error=fd));print(json.dumps(dict(gradient_error=er,finite_difference_error=fd)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--refs');ap.add_argument('--workers',type=int,default=2);ap.add_argument('--smoke',action='store_true');ap.add_argument('--synthetic',action='store_true');a=ap.parse_args()
    if a.synthetic:synthetic(a.out);return
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    R.js(out/'launch.json',dict(git_hash=git,start_unix=start,code_sha256=R.sha(__file__),smoke=a.smoke))
    jobs=[(0,10)] if a.smoke else [(s,t) for t in (10,25) for s in (0,1,2)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(run,s,t,str(out),a.refs,12 if a.smoke else 4000,git) for s,t in jobs];results=[f.result() for f in fs]
    R.js(out/'complete.json',dict(jobs=results,seconds=time.time()-start))

if __name__=='__main__':main()

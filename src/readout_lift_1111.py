#!/usr/bin/env python3
"""Readout-lift causal diagnostic; frozen design in specs/spec_readout_lift_1111.md."""
import argparse, concurrent.futures, copy, csv, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
import stage1_ctrl_1111 as E
import demand_allocation_1111 as D

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
ARCH=Path('/home/issan/Projects/obsidian-research-data/demand_allocation_1111/results/demand_allocation_1111/raw')
TIMES=(0,1,10,50,200,500,1000,2000,4000)
ARMS=('V1','S','Q','QV','V01','V01U','V01R','V01M','V01UM')
LR=.001

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v): Path(p).write_text(json.dumps(v,indent=2))
def mean(x): return float(x.double().mean()) if x.numel() else float('nan')
def rms(x): return float(x.double().square().mean().sqrt()) if x.numel() else float('nan')
def csvwrite(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(p,'w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def private(seed,task):
    h=hashlib.sha256(f'readout_lift_1111|fresh_labels|{seed}|{task}'.encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8],'little')&((1<<63)-1))

def groups(F,F0,y,V):
    z=F['z'];z0=F0['z'];top=torch.zeros_like(z,dtype=torch.bool)
    top.scatter_(0,torch.argsort(z,dim=0,stable=True)[-60:],True)
    return dict(all=torch.ones_like(z,dtype=torch.bool),body=z0<=-3,upper=F0['upper'],
                current_top=top,open=z>0,matching_label=y[:,None]==V.argmax(0)[None,:])

def body_setup(X,F0):
    B=F0['z']<=-3;n=B.sum(0);eligible=n>=30
    bx=B.T.double()@X.double()/n.clamp_min(1)[:,None]
    bx[n==0]=X.double().mean(0)
    return bx,eligible

def body_mean(P,bx):return (P[0].detach().double()*bx).sum(1)+P[1].detach().double()

def trace_init(P,opt):
    return dict(t=0,old=[opt.state[p]['exp_avg'].detach().clone() for p in P[:2]],
                uniform=[torch.zeros_like(p) for p in P[:2]],label=[torch.zeros_like(p) for p in P[:2]])

def one_step(P,opt,trace,xb,Y,frozen,projection=None,bx=None,eligible=None,capture=False):
    z=xb@P[0].T+P[1];a=E.activ(z,'ELU');ell=a@P[2].T+P[3]
    loss=(ell.logsumexp(-1)-(ell*Y).sum(-1)).mean()
    opt.zero_grad(set_to_none=True);loss.backward()
    with torch.no_grad():
        # This matches the derivative of the native float32 expm1 ELU, including rounded zeros.
        G=torch.where(z>0,torch.ones_like(z),a+1)
        dy=((.1-Y)@P[2])*G/len(xb)
        gy=[dy.T@xb,dy.sum(0)]
        for j,p in enumerate(P[:2]):
            trace['old'][j].mul_(.9)
            trace['label'][j].mul_(.9).add_(gy[j],alpha=.1)
            trace['uniform'][j].mul_(.9).add_(p.grad-gy[j],alpha=.1)
        trace['t']+=1
    if capture: before=[p.detach().double().clone() for p in P[:2]]
    if frozen:P[2].grad=None
    opt.step()
    if capture: ordinary=[p.detach().double().clone() for p in P[:2]]
    correction=torch.zeros_like(P[1])
    if projection is not None:
        with torch.no_grad():
            correction[eligible]=(projection-body_mean(P,bx))[eligible].float()
            P[1].add_(correction)
    if not capture:return correction
    with torch.no_grad():
        parts={k:[] for k in ('old','uniform','label')};merr=0.
        for j,p in enumerate(P[:2]):
            st=opt.state[p];t=float(st['step'])
            denom=(st['exp_avg_sq'].double()/(1-.999**t)).sqrt()+1e-8
            coef=-LR/(1-.9**t)/denom
            msum=sum(trace[k][j] for k in parts)
            merr=max(merr,float((msum-st['exp_avg']).abs().max()))
            for k in parts:parts[k].append(coef*trace[k][j].double())
        parts['rounding']=[ordinary[j]-before[j]-sum(parts[k][j] for k in ('old','uniform','label')) for j in range(2)]
        parts['projection']=[p.detach().double()-ordinary[j] for j,p in enumerate(P[:2])]
        residual=max(float(v.abs().max()) for v in parts['rounding'])
        assert merr<3e-6,(merr,'moment source error')
        assert residual<4e-6*max(1,max(float(v.abs().max()) for v in before)),(residual,'Adam source error')
    return dict(parts=parts,moment_error=merr,step_rounding=residual,before=before)

def stats(P,F,F0,bx,eligible,X,y,meta,rows,unit):
    z,a=F['z'],F['a'];b=body_mean(P,bx);b0=F0['body'];h=z.max(0).values.double();h0=F0['h']
    B=F0['z']<=-3;n=B.sum(0).clamp_min(1)
    bsd=((z.double()-b[None,:]).square()*B).sum(0).div(n).sqrt()
    gap=h-b;g0=h0-b0;Vc=P[2].detach()-P[2].detach().mean(0)
    logit=F['ell'];true=logit.gather(1,y[:,None]).squeeze(1)
    other=logit.clone();other.scatter_(1,y[:,None],-float('inf'))
    data=dict(**meta,ce=F['ce'],acc=F['acc'],eligible=int(eligible.sum()),
        G=mean(gap[eligible]),G_change=mean((gap-g0)[eligible]),h_median=float(h.median()),h_change=float(h.median()-h0.median()),
        body=mean(b[eligible]),body_change=mean((b-b0)[eligible]),body_sd=mean(bsd[eligible]),
        zmean=mean(z),open_per_input=mean((z>0).sum(1)),activity=mean(a),positive_activity=mean(a.clamp_min(0)),
        fixed_upper_z=mean(z[F0['upper']]),fixed_upper_a=mean(a[F0['upper']]),
        fixed_upper_da=mean((a-F0['a'])[F0['upper']]),
        cooccur_0=float((((b-b0)<-.1)&((h-h0)>.1))[eligible].double().mean()),
        Vnorm=float(Vc.norm()),confidence=mean(F['p'].max(1).values),
        entropy=mean(-(F['p']*F['p'].clamp_min(1e-30).log()).sum(1)),
        margin=mean(true-other.max(1).values),mean_class_margin=mean(true-logit.mean(1)),
        uniform_loss=mean(logit.logsumexp(-1)-logit.mean(1)))
    if 200 in unit:
        u=unit[200];data['cooccur_200']=mean((((b-u['body'])<-.1)&((h-u['h'])>.1))[eligible])
    rows['state'].append(data)
    unit[meta['step']]=dict(body=b.clone(),body_sd=bsd.clone(),h=h.clone(),gap=gap.clone(),
        zmean=z.double().mean(0),activity=a.double().mean(0),open=(z>0).double().mean(0),
        fixed_upper=(z.double()*F0['upper']).sum(0)/60,
        fixed_upper_a=(a.double()*F0['upper']).sum(0)/60)
    DU=(F['p']-.1)@P[2].detach();DY=(.1-torch.nn.functional.one_hot(y,10))@P[2].detach()
    err=float((DU+DY-F['D']).abs().max());assert err<5e-5*max(1,float(F['D'].abs().max()))
    for name,m in groups(F,F0,y,P[2].detach()).items():
        d=F['D'][m];du=DU[m];dy=DY[m]
        dg=F['delta']*m/len(X)
        gw=dg.T@X;gb=dg.sum(0)
        rows['demand'].append(dict(**meta,group=name,label_channel='fixed_evaluation',n=int(m.sum()),
            D=mean(d),DU=mean(du),DY=mean(dy),up_fraction=mean(d<0),up_magnitude=mean((-d).clamp_min(0)),
            DU_rms=rms(du),DY_rms=rms(dy),gradient_rms=rms(torch.cat([gw.flatten(),gb])),
            alignment=mean(((a-a.mean(0))*(-DY))[m]),split_error=err))

def counterfactual(P,F,F0,y,meta,rows):
    V=P[2].detach();b=P[3].detach();a=F['a'];abar=a.mean(0)
    masks=groups(F,F0,y,V);Y=torch.nn.functional.one_hot(y,10)
    for c in (.1,1.,10.):
        Vc=V.mean(0)+c*(V-V.mean(0));bc=b+(V-Vc)@abar
        pc=(a@Vc.T+bc).softmax(-1)
        combos=((1.,F['p'],1.,V),) if c==1 else ((1.,F['p'],c,Vc),(c,pc,1.,V),(c,pc,c,Vc))
        for ps,p,vs,v in combos:
            DU=(p-.1)@v;DY=(.1-Y)@v;dm=DU+DY
            for name in ('all','body','upper','current_top','open','matching_label'):
                m=masks[name]
                rows['counterfactual'].append(dict(**meta,group=name,p_scale=ps,V_scale=vs,
                    D=mean(dm[m]),DU=mean(DU[m]),DY=mean(DY[m]),up_fraction=mean(dm[m]<0),
                    up_magnitude=mean((-dm[m]).clamp_min(0)),sign_changed=mean((dm[m]*F['D'][m])<0)))

def probe(P,opt,trace,X,Y,order,R,arm,t,F,F0,y,bx,eligible,ref,meta,rows):
    Pc,oc=E.restore(E.snapshot(P,opt),'adam',LR);tc=copy.deepcopy(trace)
    idx=order[t%len(order)];yb=targets(arm,Y,idx,R,t%len(order))
    proj=ref[min(t+1,len(ref)-1)] if arm.endswith('M') else None
    cap=one_step(Pc,oc,tc,X[idx],yb,arm not in ('S','Q'),proj,bx,eligible,True)
    total=[p.detach().double()-q.detach().double() for p,q in zip(Pc[:2],P[:2])]
    dz=X.double()@total[0].T+total[1]
    zn=X@Pc[0].detach().T+Pc[1].detach();err=float((dz-(zn-F['z']).double()).abs().max())
    assert err<2e-5*max(1,float(F['z'].abs().max())),err
    for name,m in groups(F,F0,y,P[2].detach()).items():
        n=m.sum(0);xm=m.T.double()@X.double()/n.clamp_min(1)[:,None]
        for part,up in {**cap['parts'],'total':total}.items():
            per=(xm*up[0]).sum(1)+up[1]
            val=float((per*n).sum()/n.sum()) if n.sum()>0 else float('nan')
            rows['source'].append(dict(**meta,group=name,source=part,dz=val,
                moment_error=cap['moment_error'],step_rounding=cap['step_rounding'],z_rounding=err,
                post_horizon=(t>=4000),projection_reference_step=min(t+1,len(ref)-1) if proj is not None else -1))

def targets(arm,Y,idx,R,t):
    if arm in ('V01U','V01UM'):return torch.full((len(idx),10),.1)
    if arm=='V01R':return torch.nn.functional.one_hot(R[t],10).float()
    return Y[idx]

def run_job(seed,task,outstr,steps=4000,git_hash=''):
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    torch.set_num_threads(1);start=time.monotonic()
    out=Path(outstr)/f's{seed}_t{task}';out.mkdir(parents=True,exist_ok=False)
    source=ARCH/f'ELU_s{seed}'/f'baseline_before_t{task}.pt'
    base=torch.load(source,weights_only=False,map_location='cpu');X,subset=E.load_X(seed)
    assert np.array_equal(subset,base['subset']);y=base['y'];order=base['order'][:steps]
    Y=torch.nn.functional.one_hot(y,10).float();R=torch.randint(10,(steps,16),generator=private(seed,task))
    initial=E.initial(seed,(784,100,10));ctx=E.Ctx(X,'ELU')
    F0=D.fields(base['P'],X,y,'ELU');F0['upper']=torch.zeros_like(F0['z'],dtype=torch.bool)
    F0['upper'].scatter_(0,torch.argsort(F0['z'],dim=0,stable=True)[-60:],True)
    bx,eligible=body_setup(X,F0);F0['body']=body_mean(base['P'],bx);F0['h']=F0['z'].max(0).values.double()
    rows={k:[] for k in ('state','demand','source','counterfactual')};checks=[];references={};units={}
    probe_times=sorted(set(t for t in TIMES if t<=steps)|{steps})
    for arm in ARMS:
        P,opt=E.restore((base['P'],base['opt']),'adam',LR)
        if arm in ('Q','QV'):E.intervene('Q',P,opt,ctx,E.own('unused',seed),initial[2],'adam')
        if arm.startswith('V01'):
            with torch.no_grad():
                old=P[2].clone();abar=E.mean_act(P,ctx)
                P[2].copy_(old.mean(0)+.1*(old-old.mean(0)))
                E.fix_bias(P,old,abar)
        Vstart=P[2].detach().clone();Fstart=D.fields(P,X,y,'ELU')
        if arm!='S':
            mlerr=float((Fstart['ell'].mean(0)-F0['ell'].mean(0)).abs().max())
            assert mlerr<1e-4,mlerr
        else:mlerr=0.
        trace=trace_init(P,opt);unit={};bm=[];corrections=[];max_match=0.
        for t in range(steps+1):
            now=body_mean(P,bx);bm.append(now.numpy())
            if arm.endswith('M'):
                er=float((now-references['V1'][t])[eligible].abs().max());max_match=max(max_match,er)
                assert er<2e-5*max(1,float(references['V1'][t].abs().max())),er
            if t in probe_times:
                meta=dict(seed=seed,task=task,arm=arm,step=t)
                F=D.fields(P,X,y,'ELU');stats(P,F,F0,bx,eligible,X,y,meta,rows,unit)
                counterfactual(P,F,F0,y,meta,rows)
                ref=references.get('V1',torch.stack([now]*(steps+1)))
                probe(P,opt,trace,X,Y,order,R,arm,t,F,F0,y,bx,eligible,ref,meta,rows)
            if t<steps:
                idx=order[t];yb=targets(arm,Y,idx,R,t)
                proj=references['V1'][t+1] if arm.endswith('M') else None
                corr=one_step(P,opt,trace,X[idx],yb,arm not in ('S','Q'),proj,bx,eligible)
                corrections.append(corr.detach().numpy().copy())
        references[arm]=torch.from_numpy(np.array(bm))
        frozen_err=float((P[2].detach()-Vstart).abs().max()) if arm not in ('S','Q') else 0.
        assert frozen_err==0
        repro=None
        if steps==4000 and arm in ('S','Q','QV'):
            old=torch.load(ARCH/f'ELU_s{seed}'/f'{arm}_t{task}_fit_end.pt',weights_only=False,map_location='cpu')
            repro=max(float((p.detach()-q).abs().max()) for p,q in zip(P,old['P']))
            assert repro==0,('reproduction',seed,task,arm,repro)
        checks.append(dict(arm=arm,mean_logit_error=mlerr,frozen_V_error=frozen_err,repro_maxabs=repro,max_body_match=max_match))
        sp,so=E.snapshot(P,opt);torch.save(dict(P=sp,opt=so,trace=trace,y=y),out/f'{arm}_end.pt')
        np.savez_compressed(out/f'{arm}_trace.npz',body=np.array(bm),corrections=np.array(corrections),eligible=eligible.numpy(),
            **{f'{t}_{k}':v.numpy() for t,d in unit.items() for k,v in d.items()})
        for k,v in rows.items():csvwrite(out/f'{k}.csv',v)
        print(f'seed={seed} task={task} arm={arm} done seconds={time.monotonic()-start:.1f}',flush=True)
    result=dict(seed=seed,task=task,steps=steps,git_hash=git_hash,source=str(source),source_sha256=sha(source),
        code_sha256=sha(__file__),engine_sha256=sha(E.__file__),fields_sha256=sha(D.__file__),
        spec_sha256=sha(ROOT/'specs/spec_readout_lift_1111.md'),input_sha256=sha(E.DATA/'train-images-idx3-ubyte.gz'),
        labels_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest(),order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
        fresh_labels_sha256=hashlib.sha256(R.numpy().tobytes()).hexdigest(),checks=checks,
        eligible_units=int(eligible.sum()),rows={k:len(v) for k,v in rows.items()},seconds=time.monotonic()-start,torch=torch.__version__)
    js(out/'complete.json',result);return result

def synthetic(out):
    torch.manual_seed(1111);checks=[]
    X=torch.randn(11,7,dtype=torch.float64);P=[torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in ((5,7),(5,),(10,5),(10,))]
    Y=torch.nn.functional.one_hot(torch.randint(10,(11,)),10).double();z=X@P[0].T+P[1];a=E.activ(z,'ELU');ell=a@P[2].T+P[3]
    L=(ell.logsumexp(-1)-(ell*Y).sum(-1)).mean();LU=(ell.logsumexp(-1)-ell.mean(-1)).mean();LY=((.1-Y)*ell).sum(-1).mean()
    g=torch.autograd.grad(L,P,retain_graph=True);gu=torch.autograd.grad(LU,P,retain_graph=True);gy=torch.autograd.grad(LY,P)
    err=max(float((q-u-v).abs().max()) for q,u,v in zip(g,gu,gy));assert err<1e-12
    checks.append(dict(test='float64_gradient_split',error=err))
    z=torch.tensor([-24.,-17.,-16.,-1.,0.,1.],requires_grad=True);a=E.activ(z,'ELU');G=torch.autograd.grad(a.sum(),z)[0]
    manual=torch.where(z>0,torch.ones_like(z),a.detach()+1);err=float((G-manual).abs().max());assert err==0
    checks.append(dict(test='native_elu_derivative',error=err,values=G.tolist()))
    # Float64 independent Adam source/step identity with nonzero old moments.
    q=torch.randn(13,dtype=torch.float64,requires_grad=True);opt=torch.optim.Adam([q],lr=LR)
    for _ in range(4):q.grad=torch.randn_like(q);opt.step()
    old=opt.state[q]['exp_avg'].clone();u=torch.randn_like(q);y=torch.randn_like(q);before=q.detach().clone();q.grad=u+y;opt.step()
    st=opt.state[q];t=float(st['step']);coef=-LR/(1-.9**t)/((st['exp_avg_sq']/(1-.999**t)).sqrt()+1e-8)
    predicted=coef*(.9*old+.1*u+.1*y);err=float(((q.detach()-before)-predicted).abs().max());assert err<1e-14
    checks.append(dict(test='float64_adam_sources',error=err))
    # Fixed-body projection uses rowwise feature means and leaves all within-unit differences unchanged.
    X=torch.randn(23,7);W=torch.randn(5,7);b=torch.randn(5);B=torch.rand(23,5)>.4;bx=B.T.double()@X.double()/B.sum(0)[:,None]
    base=(W.double()*bx).sum(1)+b.double();target=base-.7;z=X@W.T+b
    bn=b+(target-base).float();zn=X@W.T+bn
    err=float((((W.double()*bx).sum(1)+bn.double())-target).abs().max());assert err<2e-6
    differr=float(((zn-zn.mean(0))-(z-z.mean(0))).abs().max());assert differr<3e-6
    checks.append(dict(test='body_projection',error=err,difference_error=differr))
    js(out,checks);print(json.dumps(checks),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--synthetic',action='store_true');args=ap.parse_args()
    if args.synthetic:synthetic(args.out);return
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    launch=dict(git_hash=git_hash,source_sha256=sha(__file__),unix_start=time.time(),smoke=args.smoke,workers=args.workers)
    js(out/'launch.json',launch)
    jobs=[(0,10)] if args.smoke else [(seed,task) for task in (10,25) for seed in (0,1,2)]
    steps=12 if args.smoke else 4000
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(run_job,s,t,str(out),steps,git_hash) for s,t in jobs]
        results=[f.result() for f in fs]
    js(out/'complete.json',dict(jobs=results,seconds=time.time()-launch['unix_start']))

if __name__=='__main__':main()

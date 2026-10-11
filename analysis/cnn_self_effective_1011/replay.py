#!/usr/bin/env python3
"""Exact native-engine short replay, additive CE self/Adam accounting for c1."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src import pmnist_0905 as H, pmnist_rlcifar_0907 as RC
from src import rlcifar_cnn_0908 as CN, sna_cnn_cause_1009 as E
from src.cnn_drive_verify_1009_lr import BundleLR
from ce_self import decompose_batch
SOURCE=Path('/home/issan/Projects/obsidian-research-data/cnn_drive_verify_1009/results/cnn_drive_verify_1009/LR/state.pt')
RC.DATA_DIR=Path('/home/issan/Projects/claude/proj_004_drift/data/cifar10')
COMP=('self','cross_bias','label','numerical_residual','history','rounding')

def stats(X, cache):
    if cache.exists():
        d=np.load(cache); return torch.from_numpy(d['mu']),torch.from_numpy(d['cov'])
    ms,cs=[],[]
    for r in range(len(X)):
        sm=torch.zeros(75,dtype=torch.float64);ss=torch.zeros(75,75,dtype=torch.float64);n=0
        for x in X[r].cpu().split(32):
            p=F.unfold(x.double(),5,padding=2).transpose(1,2).reshape(-1,75)
            sm+=p.sum(0);ss+=p.T@p;n+=len(p)
        m=sm/n;ms.append(m);cs.append(ss/n-m[:,None]*m[None,:])
    mu,cov=torch.stack(ms),torch.stack(cs)
    np.savez(cache,mu=mu.numpy(),cov=cov.numpy())
    return mu,cov

def channel_stats(P,mu,cov):
    w=P[0].detach().double().reshape(len(mu),16,75);b=P[1].detach().double()
    return ((w*mu[:,None]).sum(-1)+b,torch.einsum('rci,rij,rcj->rc',w,cov,w))

def csv_write(path,rows):
    with path.open('w') as f:
        dw=csv.DictWriter(f,fieldnames=list(rows[0]));dw.writeheader();dw.writerows(rows)

def serial(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [serial(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def run(args):
    torch.set_num_threads(4)
    device=H.setup(args.device)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    st=torch.load(SOURCE,map_location='cpu',weights_only=False)
    slots=st['meta']['slots']
    B=BundleLR(slots,RC.Cifar10(),device,graph=False)
    B.load_full_state(st['state'])
    assert int(B.tc)==900000 and st['t']==30
    mu,cov=stats(B.X,out/'input_stats.npz');mu=mu.to(device);cov=cov.to(device)
    B.new_labels()
    if args.start_step%75:raise ValueError('start-step must be an epoch boundary')
    burn_start=time.time()
    if args.start_step:
        B.graph=True
        for epoch in range(args.start_step//75):B.run_epoch()
        B.graph=False
    print(json.dumps({'burn_steps':args.start_step,'burn_seconds':time.time()-burn_start}),flush=True)
    B.new_order()
    initial=[p.detach().clone() for p in B.P]
    initial_state=B.full_state()
    initial_means,initial_vars=channel_stats(B.P,mu,cov)
    # The native reference runs from the identical complete state and exact same batch order.
    reference=BundleLR(slots,RC.Cifar10(),device,graph=False)
    reference.load_full_state(initial_state)
    reference.Y.copy_(B.Y);reference.order.copy_(B.order)
    moment={k:[torch.zeros_like(B.m[i],dtype=torch.float64) for i in (0,1)] for k in COMP[:-1]}
    moment['history']=[B.m[i].double().clone() for i in (0,1)]
    integrated=torch.zeros(len(COMP),len(slots),16,dtype=torch.float64,device=device)
    integrated_var=torch.zeros_like(integrated)
    integrated_bias=torch.zeros_like(integrated)
    integrated_theta=torch.zeros(len(COMP),len(slots),16,76,dtype=torch.float64,device=device)
    ds=[];checks=[];step_rows=[];t0=time.time()
    for step in range(args.steps):
        if step and step%75==0:
            B.new_order();reference.order.copy_(B.order)
        j=step%75;idx=B.order[:,j*16:(j+1)*16]
        xb=B.X[B.ar,idx];yb=B.Y[B.ar,idx]
        result=decompose_batch(B,xb,yb)
        gs=result['grads'];comps=result['components']
        before=[p.detach().clone() for p in B.P[:2]]
        premean,prevar=channel_stats(B.P,mu,cov)
        with torch.no_grad():
            B.tc.add_(1);bc1=(1-torch.pow(B.b1,B.tc)).float();bc2=(1-torch.pow(B.b2,B.tc)).float()
            for p,g,m,v in zip(B.P,gs,B.m,B.v):
                m.mul_(0.9).add_(g,alpha=0.1);v.mul_(0.999).addcmul_(g,g,value=0.001)
                p.sub_(B.lr*(m/bc1)/((v/bc2).sqrt()+1e-8))
            B.act.update(result['zs'])
            for k in COMP[:-1]:
                for i in (0,1):
                    moment[k][i].mul_(0.9)
                    if k!='history':moment[k][i].add_(comps[k][i].double(),alpha=0.1)
            # Actual full denominator shared by every component; first-moment rounding explicit.
            for i in (0,1):
                closure=B.m[i].double()-sum(moment[k][i] for k in COMP[:-1])
                moment['numerical_residual'][i].add_(closure)
            updates={}
            for k in COMP[:-1]:
                updates[k]=[-B.lr*(moment[k][i]/bc1.double())/((B.v[i]/bc2).sqrt()+1e-8).double() for i in (0,1)]
            updates['rounding']=[B.P[i].double()-before[i].double()-sum(updates[k][i] for k in COMP[:-1]) for i in (0,1)]
            dw=(B.P[0].double()-before[0].double()).reshape(len(slots),16,75)
            midpoint=2*before[0].double().reshape(len(slots),16,75)+dw
            dms=[]
            for ki,k in enumerate(COMP):
                u=updates[k][0].reshape(len(slots),16,75)
                dm=(u*mu[:,None]).sum(-1)+updates[k][1]
                # Symmetric finite-step quadratic allocation, sums exactly to total delta variance.
                dv=torch.einsum('rci,rij,rcj->rc',u,cov,midpoint)
                integrated[ki]+=dm;integrated_var[ki]+=dv;dms.append(dm)
                integrated_bias[ki]+=updates[k][1]
                integrated_theta[ki,:,:,:75]+=u
                integrated_theta[ki,:,:,75]+=updates[k][1]
            postmean,postvar=channel_stats(B.P,mu,cov)
            closure_mean=float((sum(dms)-(postmean-premean)).abs().max())
        reference.step(j)
        fidelity=max(float((p.detach()-q.detach()).abs().max()) for p,q in zip(B.P,reference.P))
        diagnostics=serial(result['diagnostics'])
        checks.append(dict(step=step+1,mean_closure=closure_mean,native_param_absmax=fidelity,**diagnostics))
        ds.append(torch.stack(dms).cpu().numpy())
        for r,(arm,seed) in enumerate(slots):
            for c in range(16):
                row=dict(seed=seed,channel=c+1,step=step+1,mean=float(postmean[r,c]),variance=float(postvar[r,c]),dm=float(postmean[r,c]-premean[r,c]),dv=float(postvar[r,c]-prevar[r,c]))
                row.update({k:float(dms[i][r,c]) for i,k in enumerate(COMP)})
                step_rows.append(row)
        if step in [0,4,14,29,49,74] or step+1==args.steps:
            print(json.dumps({'step':step+1,'elapsed_s':time.time()-t0,'native_absmax':fidelity,'mean_closure':closure_mean}),flush=True)
    final_means,final_vars=channel_stats(B.P,mu,cov)
    rows=[]
    for r,(arm,seed) in enumerate(slots):
        for c in range(16):
            dm=float(final_means[r,c]-initial_means[r,c]);dv=float(final_vars[r,c]-initial_vars[r,c])
            row=dict(seed=seed,channel=c+1,mean_before=float(initial_means[r,c]),mean_after=float(final_means[r,c]),variance_before=float(initial_vars[r,c]),variance_after=float(final_vars[r,c]),delta_mean=dm,delta_variance=dv,upward=dm>0,shrinking=dv<0)
            row.update({k:float(integrated[i,r,c]) for i,k in enumerate(COMP)})
            row.update({k+'_dvar':float(integrated_var[i,r,c]) for i,k in enumerate(COMP)})
            row.update({k+'_bias':float(integrated_bias[i,r,c]) for i,k in enumerate(COMP)})
            row.update({k+'_mu':float(integrated[i,r,c]-integrated_bias[i,r,c]) for i,k in enumerate(COMP)})
            rows.append(row)
    csv_write(out/'channels.csv',rows);csv_write(out/'steps.csv',step_rows)
    np.savez_compressed(out/'raw.npz',mean_contributions=np.stack(ds),integrated_theta=integrated_theta.cpu().numpy(),components=np.array(COMP),W_before=initial[0].cpu().numpy(),b_before=initial[1].cpu().numpy(),W_after=B.P[0].detach().cpu().numpy(),b_after=B.P[1].detach().cpu().numpy(),mean_before=initial_means.cpu().numpy(),mean_after=final_means.cpu().numpy(),variance_before=initial_vars.cpu().numpy(),variance_after=final_vars.cpu().numpy())
    torch.save(B.full_state(),out/'end_state.pt')
    validation=dict(native_param_absmax=max(x['native_param_absmax'] for x in checks),mean_step_closure=max(x['mean_closure'] for x in checks),mean_integral_closure=float((integrated.sum(0)-(final_means-initial_means)).abs().max()),variance_integral_closure=float((integrated_var.sum(0)-(final_vars-initial_vars)).abs().max()))
    # Scale-aware float64 summation bounds. Late task CE terms can cancel by
    # many orders of magnitude; passing algebra does not make their shares identifiable.
    eps64=torch.finfo(torch.float64).eps
    validation['mean_closure_tolerance']=max(1e-8,256*eps64*float(integrated.abs().sum(0).max()))
    validation['variance_closure_tolerance']=max(1e-8,256*eps64*float(integrated_var.abs().sum(0).max()))
    cancellation=integrated.abs().sum(0)/(final_means-initial_means).abs().clamp_min(1e-12)
    validation['mean_cancellation_factor_median']=float(cancellation.median())
    validation['mean_cancellation_factor_max']=float(cancellation.max())
    validation['quantitative_shares_ill_conditioned']=bool(cancellation.median()>1e6)
    (out/'checks.json').write_text(json.dumps(dict(validation=validation,steps=checks),indent=2))
    summary=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),source_task=30,task=31,start_step=args.start_step,steps=args.steps,slots=slots,components=COMP,elapsed_s=time.time()-t0,torch_version=torch.__version__,deterministic=torch.are_deterministic_algorithms_enabled(),cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,validation=validation)
    (out/'provenance.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    assert validation['native_param_absmax']==0,validation
    assert validation['mean_integral_closure']<=validation['mean_closure_tolerance'],validation
    assert validation['variance_integral_closure']<=validation['variance_closure_tolerance'],validation

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=75);p.add_argument('--start-step',type=int,default=0);p.add_argument('--device',default='cuda');p.add_argument('--out',default=str(ROOT/'results/cnn_self_effective_1011/replay'))
    run(p.parse_args())

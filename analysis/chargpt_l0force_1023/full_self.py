#!/usr/bin/env python3
"""Unit-self source through every downstream Jacobian, including time mixing."""
import argparse,csv,json,subprocess,time
from pathlib import Path
import measure as F
M,L,A,C,torch,np=F.M,F.L,F.A,F.C,F.torch,F.np
OUT=F.OUT/'full_self'
STATES=[39,100]

def secant(f):
    x=f-(f.logsumexp(-1,keepdim=True)-np.log(C.V));small=x.abs()<1e-7
    return torch.where(small,(1+x/2+x.square()/6+x.pow(3)/24)/C.V,torch.expm1(x)/(C.V*torch.where(small,torch.ones_like(x),x)))

def calculate(inp,arm,seed,task,gate=False):
    m=inp.model(arm,seed,task,10000);_,raw=L.fields(m,A.uniform(seed));ad=inp.adam(m,arm,seed,task,10000)
    alive=np.flatnonzero(raw['alive'].numpy());rng=np.random.default_rng(10230000+seed*1000+task)
    ids=np.sort(rng.choice(alive,size=min(32,len(alive)),replace=False));ii=torch.from_numpy(ids)
    W=raw['W'];wh=raw['wh'];mu=raw['mu'];sd=raw['s'];mean=raw['m']
    xf_u,_=A.moments(m,A.uniform(seed))[0];xc=xf_u-xf_u.mean(0);cov=xc.T@xc/(len(xc)-1)
    kr=mean/sd[:,None]-mu[:,None]*(W@cov)/sd[:,None].pow(3)
    for p in m.parameters():p.requires_grad_(False)
    gs={k:torch.zeros(len(ids),C.D,dtype=A.DT) for k in ['self','all0','switch','target']}
    checks={k:0. for k in ['direct_self_independent','secant','H_one','jvp_fd','bilinear','source','reference']};energy=torch.zeros(len(ids),dtype=A.DT)
    x,y,_,_=A.data(seed,task);qtab=torch.from_numpy(C.ChainM(seed,task,.01).P);N=len(x)*(x.shape[1]-1)
    for ci,(start,xx) in enumerate(A.chunks(x)):
        with torch.no_grad():
            cap=[];m.h[0](m.wte(xx),cap);r0=cap[0];xf=m.h[0].n2(r0);z=cap[1];a0=m.h[0].mlp.phi(z);gate0=m.h[0].mlp.dphi(z)
        def post(a):
            h0=r0+m.h[0].mlp.down(a);h1=m.h[1](h0);return m.head(m.nf(h1))
        a=a0.detach().requires_grad_();f=post(a)
        with torch.no_grad():
            d=secant(f.detach());p=f.softmax(-1);e=p-1/C.V;truth=torch.zeros_like(f);truth[:,1:]=qtab[xx[:,:-1],xx[:,1:]]-1/C.V
            mask=torch.ones_like(f[...,:1]);mask[:,0]=0
            checks['secant']=max(checks['secant'],F.rel(F.apply_h(d,f.detach())-e,e))
            _,fall0=torch.func.jvp(post,(a0,),(a0,))
            if gate and ci==0:
                # Explicit class-space H independently verifies the optimized
                # direct-path self coefficient; subtraction closure cannot do so.
                ds=d.reshape(-1,C.V)[1:5];hh=torch.diag_embed(ds)-ds[:,:,None]*ds[:,None,:]/ds.sum(-1)[:,None,None]
                U=m.h[0].mlp.down.weight;V=m.head.weight-m.head.weight.mean(0);B=V@(m.nf.weight[:,None]*(U-U.mean(0,keepdim=True)))
                bf=B[:,ii[:8]];explicit=torch.einsum('ci,ncd,di->ni',bf,hh,bf)
                fast=ds@(bf*bf)-(ds@bf).square()/ds.sum(-1,keepdim=True)
                checks['direct_self_independent']=F.rel(explicit-fast,explicit)
                checks['H_one']=float(hh.sum(-1).abs().max())
                assert float(torch.linalg.eigvalsh(hh).min())>=-1e-12
        def back(err):
            return torch.autograd.grad(f,a,grad_outputs=err*mask/N,retain_graph=True)[0].detach()
        for name,err in [('switch',e),('target',truth),('all0',F.apply_h(d,fall0))]:
            ga=back(err);gs[name]+=(ga[...,ii]*gate0[...,ii]).reshape(-1,len(ids)).T@xf.reshape(-1,C.D)
        for j,unit in enumerate(ids):
            tangent=torch.zeros_like(a0);tangent[...,unit]=a0[...,unit]
            with torch.no_grad():_,fi=torch.func.jvp(post,(a0,),(tangent,))
            ei=F.apply_h(d,fi);ga=back(ei)[...,unit]
            gs['self'][j]+=(ga*gate0[...,unit]).reshape(-1)@xf.reshape(-1,C.D)
            energy[j]+=(ga*a0[...,unit]).sum()
            if gate and ci==0 and j==0:
                with torch.no_grad():
                    direction=torch.zeros_like(a0);direction[...,unit]=gate0[...,unit]*(xf@kr[unit])
                    _,fr=torch.func.jvp(post,(a0,),(direction,))
                    bilinear=(ei*fr*mask).sum()/N
                    predicted=((ga*gate0[...,unit]).reshape(-1)@xf.reshape(-1,C.D))@kr[unit]
                    checks['bilinear']=abs(float(bilinear-predicted))/max(abs(float(bilinear)),abs(float(predicted)),1e-300)
                    eps=1e-6
                    fd=(post(a0+eps*tangent)-post(a0-eps*tangent))/(2*eps)
                    checks['jvp_fd']=F.rel(fd-fi,fi)
        del a,f
    gs['other0']=gs['all0']-gs['self'];gs['residual']=gs['switch']-gs['all0'];gs['fit']=gs['switch']-gs['target'];gs['negative_target']=-gs['target']
    checks['source']=F.rel(gs['self']+gs['other0']+gs['residual']-gs['switch'],gs['switch'],gs['self'],gs['other0'],gs['residual'])
    assert float(energy.min())>=-1e-10,(arm,seed,task,float(energy.min()))
    assert max(v for k,v in checks.items() if k!='jvp_fd')<1e-9,checks
    assert checks['jvp_fd']<1e-4,checks
    prior=np.load(F.OUT/'raw'/f'{arm}_seed{seed}.npz');rows=[];arrays={'ids':ids,'energy':energy.numpy()}
    for metric,den in [('SGD',None),('Adam_fixed',ad[0][ii])]:
        vals={};row=dict(arm=arm,seed=seed,task=task,metric=metric,n_alive=len(alive),n_selected=len(ids),min_self_energy=float(energy.min()),**checks)
        for name,g in gs.items():
            g=g if den is None else g/den
            vals[name+'_mu']=(g*mean).sum(1);vals[name+'_r']=(g*kr[ii]).sum(1)
            vals[name+'_mu_tangent']=(g*(mean-wh[ii]*(wh[ii]@mean)[:,None])).sum(1)
        for name,v in vals.items():
            row[name+'_med']=F.med(v);row[name+'_abs']=F.med(v.abs());row[name+'_s']=float((v>0).double().mean())
            arrays[metric+'_'+name]=v.numpy()
        for error in ['fit','switch']:
            # Compare independently recomputed full gradients with stage 1.
            expected=torch.from_numpy(prior[f't{task}_u10000_{metric}_{error}_r'][ids]);got=vals[error+'_r']
            row['reference']=max(row['reference'],F.rel(got-expected,expected,got))
            assert row['reference']<1e-9,row
            mask=got>0;n=int(mask.sum());parts=['self','other0','residual']+(['negative_target'] if error=='fit' else [])
            deno=sum(vals[k+'_r'].abs() for k in parts);nz=deno>0
            row[error+'_sink_n']=n;row[error+'_self_positive']=float((vals['self_r'][mask]>0).double().mean()) if n else None
            row[error+'_self_removal']=float(((got-vals['self_r'])[mask]<=0).double().mean()) if n else None
            row[error+'_self_share']=F.med(vals['self_r'][nz].abs()/deno[nz])
            row[error+'_self_carries']=bool(n>=8 and row[error+'_self_positive']>=.8 and row[error+'_self_removal']>=.5)
        rows.append(row)
    return rows,arrays

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['gate','run']);args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    inp=F.Inputs();ts=time.monotonic();code=M.sha(__file__)
    if args.mode=='gate':
        rows=[]
        for arm in A.ARMS:
            for task in STATES:
                rr,ar=calculate(inp,arm,0,task,True);rows+=rr
                print('full gate',arm,task,round(time.monotonic()-ts,1),flush=True)
        F.dump(OUT/'gate.json',dict(passed=True,code_sha256=code,rows=rows,seconds=time.monotonic()-ts));return
    gate=json.loads((OUT/'gate.json').read_text());assert gate['passed'] and gate['code_sha256']==code
    rows=[]
    for seed in range(10):
        for arm in A.ARMS:
            for task in STATES:
                start=time.monotonic();rr,ar=calculate(inp,arm,seed,task);rows+=rr
                p=OUT/'raw'/f'{arm}_seed{seed}_t{task}.npz';p.parent.mkdir(exist_ok=True);np.savez_compressed(p,**ar)
                F.csvwrite(OUT/'parts'/f'{arm}_seed{seed}_t{task}.csv',rr)
                print(arm,seed,task,round(time.monotonic()-start,1),flush=True)
        A.CACHE.clear();A.UCACHE.clear()
    F.csvwrite(OUT/'states.csv',rows)
    F.dump(OUT/'provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),code_sha256=code,spec_sha256=M.sha(F.ROOT/'specs/spec_chargpt_l0force_1023.md'),training_steps=0,optimizer_created=False,states=40,rows=len(rows),seconds=time.monotonic()-ts))

if __name__=='__main__':main()

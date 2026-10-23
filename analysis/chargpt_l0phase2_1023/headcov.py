#!/usr/bin/env python3
"""Phase2 through the unrestricted final head, and actual covariance / CE."""
import time,subprocess,argparse
import measure as Q
torch,np,F,L,A,C,M,OUT=Q.torch,Q.np,Q.F,Q.L,Q.A,Q.C,Q.M,Q.OUT

def one(inp,arm,seed,task,step,check=False,tol=1e-10):
    m=inp.model(arm,seed,task,step);sraw=np.load(Q.file_path('scalar',arm,seed,task,step))
    raw={k:torch.from_numpy(sraw[k]) for k in ['W','mu','s','m','cov','alive']}
    kr=raw['m']/raw['s'][:,None]-raw['mu'][:,None]*(raw['W']@raw['cov'])/raw['s'][:,None].pow(3)
    x=A.data(seed,task)[0];N=len(x)*(C.T-1)
    with torch.no_grad():h=torch.cat([m.body(xx)[:,1:].reshape(-1,C.D) for _,xx in A.chunks(x)])
    g=h.T@h;val,vec=torch.linalg.eigh(g);keep=val>val[-1]*tol;v=vec[:,keep];gi=(v/val[keep])@v.T
    ctx=(x[:,:-1]*C.V+x[:,1:]).reshape(-1);uq,ix=torch.unique(ctx,return_inverse=True)
    # Direct dual factor avoids G+ D G+ cancellation.
    dual=(h@v/val[keep])@v.T;group=torch.zeros(len(uq),C.D,dtype=A.DT);group.index_add_(0,ix,dual)
    vc=m.head.weight.detach()-m.head.weight.detach().mean(0)
    sigmas={'iid':gi,'ctx':group.T@group,'empirical':vc.T@vc}
    signals={k:h@s for k,s in sigmas.items()}
    w=m.h[0].mlp.up.weight;grad={k:torch.zeros_like(w) for k in sigmas}
    direct=torch.zeros_like(w);headnorm={e:torch.zeros(C.V,C.D,dtype=A.DT) for e in ['fit','switch']}
    qt=torch.from_numpy(C.ChainM(seed,task,.01).P)
    for start,xx in A.chunks(x):
        hh=m.body(xx);hs=hh[:,1:].reshape(-1,C.D);lo=start*(C.T-1);hi=lo+len(xx)*(C.T-1)
        for k in sigmas:
            grad[k]+=torch.autograd.grad(hs,w,grad_outputs=signals[k][lo:hi],retain_graph=True)[0].detach()
        if check:
            ff=m.head(hh)[:,1:];fc=ff-ff.mean(-1,keepdim=True)
            direct+=torch.autograd.grad(fc.square().sum()/2,w)[0].detach()
        ff=m.head(hh).detach();p=ff.softmax(-1);qq=qt[xx[:,:-1],xx[:,1:]]
        headnorm['switch']+=(p[:,1:]-1/C.V).reshape(-1,C.V).T@hs.detach()
        headnorm['fit']+=(p[:,1:]-qq).reshape(-1,C.V).T@hs.detach()
    checks={}
    if check:
        checks['squared_autograd']=F.rel(direct-grad['empirical'],direct);assert checks['squared_autograd']<1e-10
        direction=grad['iid']/grad['iid'].norm();eps=1e-4;ww=w.detach().clone();vals=[]
        # Finite differences on a temporary in-memory parameter; restore exactly.
        with torch.no_grad():
            for sign in [-1,1]:
                w.copy_(ww+sign*eps*direction)
                hp=torch.cat([m.body(xx)[:,1:].reshape(-1,C.D) for _,xx in A.chunks(x)])
                eig=torch.linalg.eigvalsh(hp.T@hp);vals.append(eig[-int(keep.sum()):].log().sum()/2)
            w.copy_(ww)
        fd=(vals[1]-vals[0])/(2*eps);pred=(grad['iid']*direction).sum()
        checks['capacity_fd']=float((fd-pred).abs()/pred.abs());assert checks['capacity_fd']<1e-4,checks
    ad=inp.adam(m,arm,seed,task,step);alive=raw['alive'];rows=[];arrays={'alive':Q.arr(alive)}
    for metric,den in [('SGD',None)]+([('Adam_fixed',ad[0])] if ad else []):
        values={k:(gg*(kr if den is None else kr/den)).sum(1) for k,gg in grad.items()}
        for k,sv in values.items():
            arrays[metric+'_'+k]=Q.arr(sv)
            row=dict(arm=arm,seed=seed,task=task,step=step,metric=metric,covariance=k,tolerance=tol,rank=int(keep.sum()),
                 condition=float(val[-1]/val[keep].min()),n_alive=int(alive.sum()),positive=float((sv[alive]>0).double().mean()),
                 mean=float(sv[alive].mean()),median=Q.med(sv[alive]),head_fit_ratio=float(headnorm['fit'].norm()/headnorm['switch'].norm()),**checks)
            for er in ['fit','switch']:
                actual=sraw[metric+'_'+er+'_r'];row.update({er+'_'+kk:vv for kk,vv in Q.compare(sv,actual,alive).items()})
            for target in ['iid','ctx','empirical']:
                row['rho_'+target]=Q.rho(Q.arr(sv)[Q.arr(alive)],Q.arr(values[target])[Q.arr(alive)])
                row['cov_cos_'+target]=float((sigmas[k]*sigmas[target]).sum()/(sigmas[k].norm()*sigmas[target].norm()))
            rows.append(row)
    for alt_tol in [1e-8,1e-12]:
        kk=val>val[-1]*alt_tol;vv=vec[:,kk];gin=(vv/val[kk])@vv.T
        arrays[f'head_inverse_tol{alt_tol:g}']=Q.arr(gin)
        for row in rows:row[f'rank_tol{alt_tol:g}']=int(kk.sum())
    tag='headcov' if tol==1e-10 else f'headcov_tol{tol:g}'
    np.savez_compressed(OUT/'raw'/f'{tag}_{arm}_seed{seed}_t{task}_u{step}.npz',**arrays)
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sensitivity',action='store_true');args=ap.parse_args()
    inp=F.Inputs();start=time.monotonic();rows=[]
    if args.sensitivity:
        for seed,task,step in [(0,100,100),(1,100,100),(1,100,10000),(7,100,100)]:
            for tol in [1e-8,1e-12]:rows.extend(one(inp,'C3',seed,task,step,tol=tol))
        Q.csvwrite(OUT/'headcov_sensitivity.csv',rows)
        Q.dump(OUT/'headcov_sensitivity_provenance.json',dict(code_sha256=M.sha(__file__),seconds=time.monotonic()-start,training_steps=0))
        print('sensitivity complete',round(time.monotonic()-start,1),flush=True);raise SystemExit(0)
    rows.extend(one(inp,'C3',0,39,10000,True));print('gate passed',round(time.monotonic()-start,1),flush=True)
    for seed in range(10):
        for arm in A.ARMS:
            for task,step in Q.STATES:
                if (arm,seed,task,step)==('C3',0,39,10000):continue
                rows.extend(one(inp,arm,seed,task,step))
        A.CACHE.clear();A.UCACHE.clear();print('seed',seed,round(time.monotonic()-start,1),flush=True)
    Q.csvwrite(OUT/'headcov_states.csv',rows)
    Q.dump(OUT/'headcov_provenance.json',dict(code_sha256=M.sha(__file__),source_sha256=M.sha(Q.__file__),spec_sha256=M.sha(Q.ROOT/'specs/spec_chargpt_l0phase2_1023.md'),
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Q.ROOT,text=True).strip(),seconds=time.monotonic()-start,training_steps=0))

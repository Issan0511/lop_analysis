#!/usr/bin/env python3
"""Exploratory phase-2 prediction with covariance shared by Markov context."""
import json,time,subprocess
import measure as Q
torch,np,F,L,A,C,M,OUT=Q.torch,Q.np,Q.F,Q.L,Q.A,Q.C,Q.M,Q.OUT

def covariance_score(gn,cross,dnorm,norm,ids,valid):
    val,vec=torch.linalg.eigh(gn);keep=val>val[-1]*1e-10
    v=vec[:,keep];inv=(v/val[keep])@v.T;cov=inv@dnorm@inv
    score=(cross@cov[:,ids]).diagonal()/norm[ids].clamp_min(1e-300)
    return torch.where(valid,score,torch.nan)

def one(inp,arm,seed,task):
    m=inp.model(arm,seed,task,10000);raw,_,_=Q.uniform_fields(m,seed)
    old=np.load(Q.file_path('full',arm,seed,task));ids=old['ids'];ii=torch.from_numpy(ids)
    kr=raw['m']/raw['s'][:,None]-raw['mu'][:,None]*(raw['W']@raw['cov'])/raw['s'][:,None].pow(3)
    for p in m.parameters():p.requires_grad_(False)
    xx=A.data(seed,task)[0][:8];N=len(xx)*(C.T-1)
    with torch.no_grad():
        cap=[];m.h[0](m.wte(xx),cap);r0=cap[0];xf=m.h[0].n2(r0);a=m.h[0].mlp.phi(cap[1]);gate=m.h[0].mlp.dphi(cap[1])
        def post(act):return m.head(m.nf(m.h[1](r0+m.h[0].mlp.down(act))))
        def js(t):return torch.func.jvp(post,(a,),(t,))[1]
        def flat(t):return (t-t.mean(-1,keepdim=True))[:,:,1:,:].reshape(t.shape[0],-1).T
        f=post(a);features=torch.zeros(N*C.V,C.FF+1,dtype=A.DT);resp=torch.zeros(N*C.V,len(ids),dtype=A.DT)
        active=torch.nonzero(a.abs().sum((0,1))>0).flatten()
        for start in range(0,len(active),8):
            jj=active[start:start+8];t=torch.zeros(len(jj),*a.shape,dtype=A.DT)
            t[torch.arange(len(jj)),:,:,jj]=a[...,jj].permute(2,0,1)
            features[:,jj]=flat(torch.func.vmap(js)(t))
        for start in range(0,len(ii),8):
            jj=ii[start:start+8];t=torch.zeros(len(jj),*a.shape,dtype=A.DT)
            t[torch.arange(len(jj)),:,:,jj]=(gate[...,jj]*(xf@kr[jj].T)).permute(2,0,1)
            resp[:,start:start+len(jj)]=flat(torch.func.vmap(js)(t))
        features[:,-1]=(f-f.mean(-1,keepdim=True))[:,1:].reshape(-1)-features[:,:C.FF].sum(1)
        norm=features.norm(dim=0).clamp_min(1e-300);fn=features/norm;gn=fn.T@fn;cross=resp.T@fn
        ctx=(xx[:,:-1]*C.V+xx[:,1:]).reshape(-1)
        unique,inv,counts=torch.unique(ctx,return_inverse=True,return_counts=True)
        grouped=torch.zeros(len(unique),C.V,C.FF+1,dtype=A.DT)
        grouped.index_add_(0,inv,fn.reshape(N,C.V,C.FF+1))
        gf=grouped.reshape(-1,C.FF+1);dn=gf.T@gf
        valid=torch.from_numpy(old['valid_1e-10'])
        qiid=covariance_score(gn,cross,gn,norm,ii,valid)
        prior=torch.from_numpy(old['q_1e-10']);ok=torch.isfinite(qiid)&torch.isfinite(prior)
        err=F.rel(qiid[ok]-prior[ok],prior[ok]);assert err<1e-7,err
        qctx=covariance_score(gn,cross,dn,norm,ii,valid);beta=1/(1+C.V*.01)
        qmix=(1-beta)*qiid+beta*qctx
        # Independently factor the covariance as B B': sum label-basis forces.
        if arm=='C3' and seed==0 and task==39:
            val,vec=torch.linalg.eigh(gn);keep=val>val[-1]*1e-10;vv=vec[:,keep];gi=(vv/val[keep])@vv.T
            coeff=gi@gf.T;independent=((cross@coeff)*coeff[ii]).sum(1)/norm[ii]
            factor_err=F.rel(independent[ok]-qctx[ok],qctx[ok]);assert factor_err<1e-7,factor_err
        else:factor_err=None
    arrays=dict(ids=ids,q_iid=Q.arr(qiid),q_ctx=Q.arr(qctx),q_mix=Q.arr(qmix),context_counts=Q.arr(counts),context_gram=Q.arr(dn))
    np.savez_compressed(OUT/'raw'/f'context_{arm}_seed{seed}_t{task}.npz',**arrays)
    rows=[]
    for name,pred in [('iid',qiid),('ctx',qctx),('mix',qmix)]:
        good=torch.isfinite(pred)
        row=dict(arm=arm,seed=seed,task=task,covariance=name,n_valid=int(good.sum()),positive=float((pred[good]>0).double().mean()),q_median=Q.med(pred),
                 n_context=len(unique),n_positions=N,max_context_count=int(counts.max()),repeat_fraction=float((counts[inv]>1).double().mean()),
                 iid_reference=err,factor_check=factor_err,rho_iid=Q.rho(pred,qiid))
        for e in ['fit','switch','canonical_fit','canonical_switch','allparam_fit','allparam_switch']:
            row.update({e+'_'+k:v for k,v in Q.compare(pred,old[e]).items()})
        rows.append(row)
    return rows

if __name__=='__main__':
    inp=F.Inputs();start=time.monotonic();rows=[]
    for seed in range(10):
        for arm in A.ARMS:
            for task in [39,100]:rows.extend(one(inp,arm,seed,task))
        A.CACHE.clear();A.UCACHE.clear();print('seed',seed,'seconds',round(time.monotonic()-start,1),flush=True)
    Q.csvwrite(OUT/'contextual_states.csv',rows)
    Q.dump(OUT/'contextual_provenance.json',dict(code_sha256=M.sha(__file__),source_sha256=M.sha(Q.__file__),spec_sha256=M.sha(Q.ROOT/'specs/spec_chargpt_l0phase2_1023.md'),
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Q.ROOT,text=True).strip(),seconds=time.monotonic()-start,training_steps=0,context_covariance_beta=1/(1+C.V*.01)))

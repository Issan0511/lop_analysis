#!/usr/bin/env python3
"""Accumulate the full-probe Gram, without storing 2 GB of feature columns."""
import time
import measure as Q
torch,np,F,L,A,C,M,OUT=Q.torch,Q.np,Q.F,Q.L,Q.A,Q.C,Q.M,Q.OUT

def one(inp,task):
    arm,seed='C3',0;m=inp.model(arm,seed,task,10000);raw,_,_=Q.uniform_fields(m,seed)
    old=np.load(Q.file_path('full',arm,seed,task));ids=old['ids'];ii=torch.from_numpy(ids)
    kr=raw['m']/raw['s'][:,None]-raw['mu'][:,None]*(raw['W']@raw['cov'])/raw['s'][:,None].pow(3)
    for p in m.parameters():p.requires_grad_(False)
    gram=torch.zeros(C.FF+1,C.FF+1,dtype=A.DT);cross=torch.zeros(len(ids),C.FF+1,dtype=A.DT)
    uenergy=torch.zeros(len(ids),dtype=A.DT);actual={e:torch.zeros(len(ids),dtype=A.DT) for e in ['fit','switch']}
    qt=torch.from_numpy(C.ChainM(seed,task,.01).P);x=A.data(seed,task)[0];N=len(x)*(C.T-1)
    for _,xx in A.chunks(x):
        with torch.no_grad():
            cap=[];m.h[0](m.wte(xx),cap);r0=cap[0];xf=m.h[0].n2(r0);a=m.h[0].mlp.phi(cap[1]);gate=m.h[0].mlp.dphi(cap[1])
            def post(act):return m.head(m.nf(m.h[1](r0+m.h[0].mlp.down(act))))
            def js(t):return torch.func.jvp(post,(a,),(t,))[1]
            def flat(t):return (t-t.mean(-1,keepdim=True))[:,:,1:,:].reshape(t.shape[0],-1).T
            f=post(a);p=f.softmax(-1);q=torch.full_like(f,1/C.V);q[:,1:]=qt[xx[:,:-1],xx[:,1:]]
            features=torch.zeros(len(xx)*(C.T-1)*C.V,C.FF+1,dtype=A.DT)
            resp=torch.zeros(features.shape[0],len(ids),dtype=A.DT)
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
            gram+=features.T@features;cross+=resp.T@features;uenergy+=resp.square().sum(0)
            for e,err in [('fit',p-q),('switch',p-1/C.V)]:actual[e]+=resp.T@err[:,1:].reshape(-1)/N
    norm=gram.diagonal().clamp_min(0).sqrt();gn=gram/norm[:,None].clamp_min(1e-300)/norm[None,:].clamp_min(1e-300)
    val,vec=torch.linalg.eigh(gn);rows=[];arrays={'ids':ids,'gram':Q.arr(gram),'cross':Q.arr(cross)}
    for tol in Q.TOLS:
        keep=val>val[-1]*tol;v=vec[:,keep];inv=(v/val[keep])@v.T
        valid=(norm[ii]>0)&((1-v.square().sum(1))[ii]<1e-7)
        energy=1/inv.diagonal()[ii].clamp_min(1e-300)
        pred=((cross/norm.clamp_min(1e-300))@inv[:,ii]).diagonal()/norm[ii].clamp_min(1e-300)
        bound=uenergy.sqrt()/(norm[ii]*energy.sqrt()).clamp_min(1e-300)
        pred=torch.where(pred.abs()<=1e-10*bound,torch.zeros_like(pred),pred)
        pred=torch.where(valid,pred,torch.nan)
        row=dict(arm=arm,seed=seed,task=task,tol=tol,n_valid=int(valid.sum()),rank=int(keep.sum()),**{'probe_'+k:v for k,v in Q.compare(old[f'q_{tol:g}'],pred).items()})
        arrays[f'q_{tol:g}']=Q.arr(pred)
        for e in actual:
            err=F.rel(actual[e]-torch.from_numpy(old['canonical_'+e]),actual[e]);assert err<1e-9,err
            row[e+'_reference']=err;row.update({e+'_'+k:v for k,v in Q.compare(pred,actual[e]).items()})
        rows.append(row)
    np.savez_compressed(OUT/'raw'/f'validate64_C3_seed0_t{task}.npz',**arrays)
    return rows

if __name__=='__main__':
    inp=F.Inputs();start=time.monotonic();rows=[]
    for task in [39,100]:
        rows.extend(one(inp,task));print(task,'seconds',round(time.monotonic()-start,1),flush=True)
    Q.csvwrite(OUT/'probe64_validation.csv',rows)
    Q.dump(OUT/'probe64_provenance.json',dict(code_sha256=M.sha(__file__),source_sha256=M.sha(Q.__file__),spec_sha256=M.sha(Q.ROOT/'specs/spec_chargpt_l0phase2_1023.md'),seconds=time.monotonic()-start,training_steps=0))

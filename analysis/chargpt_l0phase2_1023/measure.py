#!/usr/bin/env python3
"""Frozen-state phase-2 geometry versus actual complete forces. No updates."""
import argparse,csv,hashlib,importlib.util,json,subprocess,time,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('oldforce',ROOT/'analysis/chargpt_l0force_1023/measure.py')
F=importlib.util.module_from_spec(sp);sp.loader.exec_module(F)
M,L,A,C,torch,np=F.M,F.L,F.A,F.C,F.torch,F.np
OUT=ROOT/'results/chargpt_l0phase2_1023'
OLD=M.DATA/'chargpt_l0force_1023/results/chargpt_l0force_1023'
STATES=F.STATES
TOLS=[1e-10,1e-8,1e-12]
dump,csvwrite=F.dump,F.csvwrite

def arr(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
def med(x):
    x=arr(x);x=x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None
def rho(x,y):
    x,y=arr(x),arr(y);ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.ptp(x[ok])==0 or np.ptp(y[ok])==0:return None
    def rank(a):
        _,inv,counts=np.unique(a,return_inverse=True,return_counts=True)
        return (np.cumsum(counts)-(counts-1)/2)[inv]
    return float(np.corrcoef(rank(x[ok]),rank(y[ok]))[0,1])
def compare(x,y,mask=None):
    x,y=arr(x),arr(y);ok=np.isfinite(x)&np.isfinite(y)
    if mask is not None:ok&=arr(mask).astype(bool)
    x,y=x[ok],y[ok]
    yp=y>0;xp=x>0
    ba=float(((xp[yp]).mean()+(~xp[~yp]).mean())/2) if yp.any() and (~yp).any() else None
    return dict(n=len(x),rho=rho(x,y),agreement=float((xp==yp).mean()) if len(x) else None,
                balanced_accuracy=ba,pred_positive=float(xp.mean()) if len(x) else None,
                actual_positive=float(yp.mean()) if len(x) else None)

def projected(features,responses,ids,tols=TOLS):
    """Exact leave-one-column-out scores using a normalized Gram inverse.

    features may already have common nuisance columns projected out. Extra
    columns (e.g. background) remain in features but not in ids/responses.
    Nullspace participation marks non-identifiable columns, never as Q=0.
    """
    norms=features.norm(dim=0);nz=norms>0
    scales=norms.clamp_min(1e-300);fn=features/scales
    gram=fn.T@fn;cross=responses.T@fn
    val,vec=torch.linalg.eigh(gram);top=val[-1].clamp_min(1e-300)
    ids=torch.as_tensor(ids,dtype=torch.long);out={}
    for tol in tols:
        keep=val>top*tol;v=vec[:,keep];inv=(v/val[keep])@v.T
        null=(1-v.square().sum(1)).clamp_min(0)
        valid=nz[ids]&(null[ids]<1e-7)
        q=(cross@inv[:,ids]).diagonal()/scales[ids]
        energy=1/inv.diagonal()[ids].clamp_min(1e-300)
        bound=responses.norm(dim=0)/(scales[ids]*energy.sqrt()).clamp_min(1e-300)
        q=torch.where(q.abs()<=1e-10*bound,torch.zeros_like(q),q)
        q=torch.where(valid,q,torch.full_like(q,float('nan')))
        out[tol]=dict(q=q,valid=valid,energy=energy,null=null[ids],rank=int(keep.sum()),
                      eig_min=float(val.min()),condition=float(top/val[keep].min()) if keep.any() else None)
    return out,gram,cross

def geometry(x,z,raw,den):
    # The first position has no loss. Uniform r itself uses all positions.
    a=z.clamp_min(0);gate=(z>0).double();ac=a-a.mean(0);zc=z-z.mean(0)
    mean,W,mu,s,cov=[raw[k] for k in ['m','W','mu','s','cov']]
    kr=mean/s[:,None]-mu[:,None]*(W@cov)/s[:,None].pow(3)
    dirs={'r':kr,'mu':mean[None,:].expand_as(W)}
    if den is not None:dirs['adam_r']=kr/den
    result={};meta={}
    for name,k in dirs.items():
        u=gate*(x@k.T)
        scores,_,_=projected(ac,u,np.arange(C.FF))
        for tol,o in scores.items():
            tag=f'{name}_tol{tol:g}'
            result[tag]=arr(o['q']);result[tag+'_energy']=arr(o['energy']);result[tag+'_valid']=arr(o['valid'])
            meta[tag+'_rank']=o['rank']
        d=ac.square().sum(0);valid=d>0
        qc=(u*ac).sum(0)/d
        qc=torch.where(qc.abs()<=1e-10*u.norm(dim=0)/d.sqrt(),torch.zeros_like(qc),qc)
        result[name+'_constant']=arr(torch.where(valid,qc,torch.nan))
        az=(ac*zc).sum(0)/zc.square().sum(0).clamp_min(1e-300)
        azres=ac-zc*az;dz=azres.square().sum(0)
        qz=(u*azres).sum(0)/dz
        qz=torch.where(qz.abs()<=1e-10*u.norm(dim=0)/dz.sqrt(),torch.zeros_like(qz),qz)
        result[name+'_constant_z']=arr(torch.where(dz>d*1e-10,qz,torch.nan))
    # Distribution-free constant+z formula with constant K=1; not main r.
    p=gate.mean(0);nplus=gate.sum(0);nminus=len(z)-nplus
    mp=(z*gate).sum(0)/nplus.clamp_min(1);mn=(z*(1-gate)).sum(0)/nminus.clamp_min(1)
    vp=((z-mp).square()*gate).sum(0)/nplus.clamp_min(1)
    vm=((z-mn).square()*(1-gate)).sum(0)/nminus.clamp_min(1)
    balance=(1-p)*mp*vm-p*mn.abs()*vp
    result.update(p=arr(p),constant_z_sign_balance=arr(balance))
    return result,meta

def uniform_fields(m,seed):
    _,raw=L.fields(m,A.uniform(seed));x,z=A.moments(m,A.uniform(seed))[0]
    xc=x-x.mean(0);raw['cov']=xc.T@xc/(len(x)-1)
    return raw,x,z

def verify_inputs():
    assert M.sha(M.BUNDLE)==M.BUNDLE_SHA
    man=json.loads((ROOT/'results/chargpt_l0force_1023/backup_manifest.json').read_text())
    records=man if isinstance(man,list) else next(v for v in man.values() if isinstance(v,list))
    checked=[]
    for r in records:
        p=Path(r.get('backup',r.get('path','')))
        if p.suffix=='.npz' and p.parent.name=='raw' and 'full_self' not in str(p):
            assert M.sha(p)==r['sha256'];checked.append(dict(path=str(p),sha256=r['sha256']))
    assert len(checked)==20,len(checked)
    return checked

def scalar_state(inp,arm,seed,task,step):
    m=inp.model(arm,seed,task,step);raw,xu,zu=uniform_fields(m,seed)
    inp.reference(arm,seed,task,step,raw)
    ad=inp.adam(m,arm,seed,task,step);den=None if ad is None else ad[0]
    old=np.load(OLD/'raw'/f'{arm}_seed{seed}.npz')
    prefix=f't{task}_u{step}_';arrays={k:arr(raw[k]) for k in ['alive','r','mu','s','m','cov','W']}
    rows=[]
    for probe in ['uniform','chain']:
        if probe=='uniform':x,z=xu,zu
        else:x,z=A.moments(m,A.data(seed,task)[0])[0]
        mask=torch.arange(len(z))%C.T!=0;x,z=x[mask],z[mask]
        scores,meta=geometry(x,z,raw,den)
        arrays.update({probe+'_'+k:v for k,v in scores.items()})
        base=dict(arm=arm,seed=seed,task=task,step=step,probe=probe,n_alive=int(raw['alive'].sum()),**meta)
        for metric in ['r','mu']+(['adam_r'] if ad else []):
            for proj in ['constant','constant_z']+[f'tol{t:g}' for t in TOLS]:
                key=metric+'_'+proj;q=scores[key];ok=arr(raw['alive'])&np.isfinite(q)
                row=dict(base,metric=metric,projection=proj,n_valid=int(ok.sum()),q_positive=float((q[ok]>0).mean()) if ok.any() else None,q_median=med(q[ok]))
                for er in ['fit','switch']:
                    metric0='Adam_fixed' if metric=='adam_r' else 'SGD'
                    target=old[prefix+metric0+'_'+er+'_'+('mu' if metric=='mu' else 'r')]
                    arrays[metric0+'_'+er+'_'+('mu' if metric=='mu' else 'r')]=target
                    row.update({er+'_'+k:v for k,v in compare(q,target,ok).items()})
                rows.append(row)
    return rows,arrays

def all_gradients(m,seed,task):
    for p in m.parameters():p.requires_grad_(True)
    params=dict(m.named_parameters());g={e:{k:torch.zeros_like(v) for k,v in params.items()} for e in ['fit','switch']}
    x=A.data(seed,task)[0];qt=torch.from_numpy(C.ChainM(seed,task,.01).P);N=len(x)*(C.T-1)
    for _,xx in A.chunks(x):
        f=m(xx);p=f.detach().softmax(-1);q=torch.full_like(p,1/C.V);q[:,1:]=qt[xx[:,:-1],xx[:,1:]]
        for ei,(e,err) in enumerate([('fit',p-q),('switch',p-1/C.V)]):
            err[:,0]=0
            vals=torch.autograd.grad(f,tuple(params.values()),grad_outputs=err/N,retain_graph=ei==0)
            for k,v in zip(params,vals):g[e][k]+=v.detach()
    return g

class Z(torch.nn.Module):
    def __init__(self,m):super().__init__();self.model=m
    def forward(self,x):
        cap=[];self.model.h[0](self.model.wte(x),cap);return cap[1]

def all_r_push(m,seed,g,raw):
    wr=Z(m);params={f'model.{k}':v.detach() for k,v in m.named_parameters()}
    tang={f'model.{k}':v for k,v in g.items()};ds=torch.zeros(C.FF,dtype=A.DT);zs=ds.clone()
    x=A.uniform(seed);N=x.numel()
    for _,xx in A.chunks(x):
        def fn(pp):return torch.func.functional_call(wr,pp,(xx,),tie_weights=True)
        z,dz=torch.func.jvp(fn,(params,),(tang,));ds+=dz.sum((0,1));zs+=(z*dz).sum((0,1))
    dmean=ds/N;dvar=2*(zs-raw['mu']*ds)/(N-1)
    return dmean/raw['s']-raw['mu']*dvar/(2*raw['s'].pow(3))

def full_state(inp,arm,seed,task,gate=False):
    m=inp.model(arm,seed,task,10000);raw,_,_=uniform_fields(m,seed)
    alive=np.flatnonzero(arr(raw['alive']));rng=np.random.default_rng(10230000+seed*1000+task)
    ids=np.sort(rng.choice(alive,min(32,len(alive)),replace=False));ii=torch.tensor(ids)
    kr=raw['m']/raw['s'][:,None]-raw['mu'][:,None]*(raw['W']@raw['cov'])/raw['s'][:,None].pow(3)
    gd=all_gradients(m,seed,task);arrays={};checks={}
    old=np.load(OLD/'raw'/f'{arm}_seed{seed}.npz')
    for e,g in gd.items():
        own=(g['h.0.mlp.up.weight']*kr).sum(1)
        expected=torch.from_numpy(old[f't{task}_u10000_SGD_{e}_r'])
        checks[e+'_reference']=F.rel(own-expected,expected)
        assert checks[e+'_reference']<1e-9,checks
        arrays['allparam_'+e]=arr(all_r_push(m,seed,g,raw)[ii])
        arrays['canonical_'+e]=arr(own[ii])
    for p in m.parameters():p.requires_grad_(False)
    xx=A.data(seed,task)[0][:8]
    with torch.no_grad():
        cap=[];m.h[0](m.wte(xx),cap);r0=cap[0];xf=m.h[0].n2(r0);z=cap[1];a=m.h[0].mlp.phi(z);gt=m.h[0].mlp.dphi(z)
    def post(act):return m.head(m.nf(m.h[1](r0+m.h[0].mlp.down(act))))
    f=post(a);p=f.softmax(-1);q=torch.full_like(f,1/C.V)
    qt=torch.from_numpy(C.ChainM(seed,task,.01).P);q[:,1:]=qt[xx[:,:-1],xx[:,1:]]
    N=len(xx)*(C.T-1);D=N*C.V
    features=torch.zeros(D,C.FF+1,dtype=A.DT);responses=torch.zeros(D,len(ids),dtype=A.DT)
    active=torch.nonzero(a.abs().sum((0,1))>0).flatten()
    def js(tang):return torch.func.jvp(post,(a,),(tang,))[1]
    def flat(t):return (t-t.mean(-1,keepdim=True))[:,:,1:,:].reshape(t.shape[0],-1).T
    with torch.no_grad():
        for start in range(0,len(active),8):
            jj=active[start:start+8];tang=torch.zeros(len(jj),*a.shape,dtype=A.DT)
            tang[torch.arange(len(jj)),:,:,jj]=a[...,jj].permute(2,0,1)
            ft=torch.func.vmap(js)(tang);features[:,jj]=flat(ft)
        for start in range(0,len(ids),8):
            jj=ii[start:start+8];tang=torch.zeros(len(jj),*a.shape,dtype=A.DT)
            tang[torch.arange(len(jj)),:,:,jj]=(gt[...,jj]*(xf@kr[jj].T)).permute(2,0,1)
            responses[:,start:start+len(jj)]=flat(torch.func.vmap(js)(tang))
        fc=(f-f.mean(-1,keepdim=True))[:,1:].reshape(-1)
        features[:,-1]=fc-features[:,:C.FF].sum(1)
        scores,gram,cross=projected(features,responses,ids)
        others,_,_=projected(features[:,:C.FF],responses,ids,[1e-10])
        actual={e:responses.T@err[:,1:].reshape(-1)/N for e,err in [('fit',p-q),('switch',p-1/C.V)]}
        normal={e:features.T@err[:,1:].reshape(-1)/N for e,err in [('fit',p-q),('switch',p-1/C.V)]}
        joint=float(normal['fit'].norm()/normal['switch'].norm())
        with torch.enable_grad():
            ac=a.detach().requires_grad_();fa=post(ac)
            for ei,(e,err) in enumerate([('fit',p-q),('switch',p-1/C.V)]):
                er=err.clone();er[:,0]=0;ga=torch.autograd.grad(fa,ac,grad_outputs=er/N,retain_graph=ei==0)[0]
                direct=((ga[...,ii]*gt[...,ii]).reshape(-1,len(ids)).T@xf.reshape(-1,C.D)*kr[ii]).sum(1)
                checks[e+'_bilinear']=F.rel(direct-actual[e],direct);assert checks[e+'_bilinear']<1e-9,checks
        if gate:
            unit=ids[0];tang=torch.zeros_like(a);tang[...,unit]=a[...,unit];h=1e-6
            fd=(post(a+h*tang)-post(a-h*tang))/(2*h);j=js(tang)
            checks['jvp_fd']=F.rel(fd-j,j);assert checks['jvp_fd']<1e-4,checks
    arrays.update(ids=ids,gram=arr(gram),cross=arr(cross),others_q=arr(others[1e-10]['q']),normal_fit=arr(normal['fit']),normal_switch=arr(normal['switch']))
    rows=[]
    for tol,o in scores.items():
        arrays[f'q_{tol:g}']=arr(o['q']);arrays[f'energy_{tol:g}']=arr(o['energy']);arrays[f'valid_{tol:g}']=arr(o['valid'])
        row=dict(arm=arm,seed=seed,task=task,tol=tol,n_selected=len(ids),n_valid=int(o['valid'].sum()),rank=o['rank'],joint_fit_ratio=joint,
                 energy_median=med(o['energy'][o['valid']]),q_positive=med([]),**checks)
        qv=arr(o['q']);ok=np.isfinite(qv);row['q_positive']=float((qv[ok]>0).mean()) if ok.any() else None
        for e in actual:
            arrays[e]=arr(actual[e])
            row.update({e+'_'+k:v for k,v in compare(qv,actual[e]).items()})
            row.update({'probe_'+e+'_'+k:v for k,v in compare(actual[e],arrays['canonical_'+e]).items()})
            row.update({'allparam_'+e+'_'+k:v for k,v in compare(qv,arrays['allparam_'+e]).items()})
        rows.append(row)
    return rows,arrays

def algebra_checks():
    gen=torch.Generator().manual_seed(102323)
    a=torch.randn(41,7,generator=gen,dtype=A.DT);a-=a.mean(0)
    u=torch.randn(41,7,generator=gen,dtype=A.DT);out,_,_=projected(a,u,range(7))
    expected=[]
    for i in range(7):
        other=a[:,torch.arange(7)!=i];r=a[:,i]-other@torch.linalg.lstsq(other,a[:,i],driver='gelsd').solution
        expected.append((u[:,i]@r)/(r@r))
    expected=torch.stack(expected);err=F.rel(out[1e-10]['q']-expected,expected);assert err<1e-10
    dup=torch.cat([a,a[:,0,None]],1);o,_,_=projected(dup,u,range(7));assert not bool(o[1e-10]['valid'][0]) and bool(o[1e-10]['valid'][1:].all())
    z=torch.tensor([1.,2.,-1.,-1.,-1.],dtype=A.DT);aa=z.clamp_min(0);base=torch.stack([torch.ones_like(z),z],1)
    r=aa-base@torch.linalg.lstsq(base,aa,driver='gelsd').solution
    counter=float((z.gt(0).double()@r)/(r@r));assert counter<0
    return dict(leave_one_out_relative=err,duplicate_nonidentifiable=True,nonGaussian_counterexample=counter)

def file_path(kind,arm,seed,task,step=10000):return OUT/'raw'/f'{kind}_{arm}_seed{seed}_t{task}_u{step}.npz'
def save(kind,arm,seed,task,step,rows,arrays):
    p=file_path(kind,arm,seed,task,step);p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,**arrays)
    meta=dict(code_sha256=M.sha(__file__),sha256=M.sha(p),rows=rows)
    dump(p.with_suffix('.json'),meta)
def recover(kind,arm,seed,task,step):
    p=file_path(kind,arm,seed,task,step)
    if not p.exists():return None
    meta=json.loads(p.with_suffix('.json').read_text());assert meta['code_sha256']==M.sha(__file__) and meta['sha256']==M.sha(p)
    return meta['rows']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['gate','run']);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);inp=F.Inputs();start=time.monotonic()
    if args.mode=='gate':
        checks=algebra_checks();files=verify_inputs()
        rows,raw=full_state(inp,'C3',0,39,True)
        save('full','C3',0,39,10000,rows,raw)
        dump(OUT/'gate.json',dict(passed=True,algebra=checks,source_files=files,rows=rows,code_sha256=M.sha(__file__),seconds=time.monotonic()-start))
        print('PASS',json.dumps(checks),'seconds',time.monotonic()-start,flush=True);return
    gate=json.loads((OUT/'gate.json').read_text());assert gate['passed'] and gate['code_sha256']==M.sha(__file__)
    scalar=[];full=[]
    for seed in range(10):
        for arm in A.ARMS:
            ts=time.monotonic()
            for task,step in STATES:
                rows=recover('scalar',arm,seed,task,step)
                if rows is None:
                    rows,raw=scalar_state(inp,arm,seed,task,step);save('scalar',arm,seed,task,step,rows,raw)
                scalar.extend(rows)
            print('scalar',arm,seed,round(time.monotonic()-ts,1),flush=True)
            for task in [39,100]:
                rows=recover('full',arm,seed,task,10000)
                if rows is None:
                    rows,raw=full_state(inp,arm,seed,task);save('full',arm,seed,task,10000,rows,raw)
                full.extend(rows)
            print('full',arm,seed,round(time.monotonic()-ts,1),flush=True)
        A.CACHE.clear();A.UCACHE.clear()
    csvwrite(OUT/'states.csv',scalar);csvwrite(OUT/'full_states.csv',full)
    dump(OUT/'provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_sha256=M.sha(__file__),
        spec_sha256=M.sha(ROOT/'specs/spec_chargpt_l0phase2_1023.md'),training_steps=0,optimizer_created=False,device='CPU',threads=torch.get_num_threads(),
        scalar_states=100,full_states=40,full_probe_sequences=8,canonical_probe_sequences=64,seconds=time.monotonic()-start,
        bundle=dict(path=str(M.BUNDLE),sha256=M.BUNDLE_SHA),borrowed=list(inp.borrowed.values())))

if __name__=='__main__':main()

#!/usr/bin/env python3
"""SiLU saved states: complete downstream gradients and full-parameter r JVPs."""
import argparse,csv,hashlib,importlib.util,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('early',ROOT/'analysis/chargpt_earlyphase_1024/measure.py')
E=importlib.util.module_from_spec(sp);sp.loader.exec_module(E)
torch,np,A,C,Q,M=E.torch,E.np,E.A,E.C,E.Q,E.M
OUT=ROOT/'results/chargpt_siluphase2_1024'
SOURCE=Path('/home/issan/Projects/claude/wt/chargpt_cl_1019/results/chargpt_cl_1019/quicklook')
LAYERS=[2,3];SEED=1000
STATES=[('SiLU',t,10000) for t in [1,5,10,25,50,100]]+[('SiLU',100,100)]+[('SiLUcap',t,10000) for t in [10,25,50,100]]
dump,arr,rel=E.dump,Q.arr,E.rel
USED={}

def remember(p):
    USED[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=M.sha(p))
    return p

def load(arm,task,step):
    folder=SOURCE/'M'/(arm+'_cont' if task==100 else arm)/'seed1000'
    p=(folder/'ckpts'/f'task{task}.pt') if step==10000 else SOURCE/'rho/states'/arm/f'u100_task{task}.pt'
    C.L=4;m=C.GPT(arm,'M');m.load_state_dict(torch.load(remember(p),map_location='cpu',weights_only=True))
    m.eval().double()
    for b in m.h:b.mlp.track_v=False
    assert m.wte.weight is m.head.weight
    adam=torch.load(remember(p.with_name(p.stem+'_adam.pt')),map_location='cpu',weights_only=True)
    group=adam['param_groups'][0];params=dict(m.named_parameters());den={}
    assert len(group['params'])==len(params)
    for i,(name,w) in zip(group['params'],params.items()):
        st=adam['state'][i];assert st['exp_avg_sq'].shape==w.shape
        den[name]=(st['exp_avg_sq'].double()/(1-group['betas'][1]**float(st['step']))).sqrt()+group['eps']
    return m,den,folder

@torch.no_grad()
def fields(m):
    out={}
    for layer,(x,z) in enumerate(A.moments(m,A.uniform(SEED))):
        if layer not in LAYERS:continue
        w=m.h[layer].mlp.up.weight.detach();mean=x.mean(0);mu=z.mean(0);s=z.std(0)
        cov=(x-mean).T@(x-mean)/(len(x)-1)
        assert rel(mu,w@mean)<1e-9
        gate=m.h[layer].mlp.dphi(z)
        out[layer]=dict(mu=mu,s=s,r=mu/s,m=mean,kr=mean/s[:,None]-mu[:,None]*(w@cov)/s[:,None]**3,
            responsive=gate.abs().max(0).values>1e-6,p=(z>0).double().mean(0),abs_gate=gate.abs().mean(0),N=len(x))
    return out

def reference(folder,task,step,raw):
    p=remember(folder/'layer_stats.csv')
    refs={int(r['layer']):r for r in csv.DictReader(p.open()) if int(r['task'])==task and int(r['u'])==step and r['probe']=='uniform'}
    if not refs and step==100:
        # Continued trajectories only recorded endpoints. State remains usable, no invented reference.
        return {'reference_available':False}
    assert set(refs)>=set(LAYERS),(p,task,step)
    checks={}
    for l in LAYERS:
        got={'r_med':float(torch.quantile(raw[l]['r'],.5)),'sd_med':float(raw[l]['s'].median())}
        for key,v in got.items():
            ex=float(refs[l][key]);diff=abs(v-ex);checks[f'L{l}_{key}_abs']=diff
            assert diff<=max(1e-5,abs(ex)*1e-3),(task,l,key,v,ex,diff)
    return checks

def gradients(m,task,gate):
    x=A.data(SEED,task)[0];n=len(x)*(C.T-1);params=dict(m.named_parameters())
    with torch.no_grad():
        h=torch.cat([m.body(xx)[:,1:].reshape(-1,C.D) for _,xx in A.chunks(x)])
        eig,vec=torch.linalg.eigh(h.T@h);ranks={f'{tol:g}':int((eig>eig[-1]*tol).sum()) for tol in [1e-8,1e-10,1e-12]}
        signals={}
        for tol in [1e-10,1e-8,1e-12]:
            if tol!=1e-10 and ranks[f'{tol:g}']==ranks['1e-10']:continue
            keep=eig>eig[-1]*tol;v=vec[:,keep];dual=(h@v/eig[keep])@v.T
            name='P2' if tol==1e-10 else f'P2tol{tol:g}';signals[name]=dual
        ctx=(x[:,:-1]*C.V+x[:,1:]).reshape(-1);uq,ix=torch.unique(ctx,return_inverse=True)
        summed=torch.zeros(len(uq),C.D,dtype=A.DT);summed.index_add_(0,ix,signals['P2'])
        signals['P2ctx']=h@(summed.T@summed)
        logits=m.head(h);p=logits.softmax(-1);qt=torch.from_numpy(C.ChainM(SEED,task,.01).P)
        q=qt[x[:,:-1],x[:,1:]].reshape(-1,C.V);vc=m.head.weight.detach()-m.head.weight.detach().mean(0)
        signals['switch_body']=(p-1/C.V)@vc/n;signals['fit_body']=(p-q)@vc/n
        direct={'switch':(p-1/C.V).T@h/n,'fit':(p-q).T@h/n}
        stats=dict(rank=ranks['1e-10'],rank_1e8=ranks['1e-08'],rank_1e12=ranks['1e-12'],
            eig_max=float(eig[-1]),eig_min_kept=float(eig[eig>eig[-1]*1e-10].min()),
            pmax=float(p.max(-1).values.mean()),qmax=float(q.max(-1).values.mean()),
            KL=float((q*(q.clamp_min(1e-300).log()-logits.log_softmax(-1))).sum(-1).mean()))
    names=list(signals)+['switch_full','fit_full'];gs={k:{name:torch.zeros_like(w) for name,w in params.items()} for k in names}
    for start,xx in A.chunks(x):
        hh=m.body(xx)[:,1:].reshape(-1,C.D);lo=start*(C.T-1);hi=lo+len(hh)
        losses={}
        for name,signal in signals.items():
            gg=torch.autograd.grad(hh,list(params.values()),grad_outputs=signal[lo:hi],retain_graph=True,allow_unused=True)
            for (key,_),v in zip(params.items(),gg):
                if v is not None:gs[name][key]+=v.detach()
        f=m.head(hh)
        losses['switch_full']=(f.logsumexp(-1)-f.mean(-1)).sum()/n
        losses['fit_full']=-(q[lo:hi]*f.log_softmax(-1)).sum()/n
        for name,loss in losses.items():
            gg=torch.autograd.grad(loss,list(params.values()),retain_graph=True,allow_unused=True)
            for (key,_),v in zip(params.items(),gg):
                if v is not None:gs[name][key]+=v.detach()
    checks={}
    for error in ['switch','fit']:
        expected={k:v.clone() for k,v in gs[error+'_body'].items()};expected['wte.weight']+=direct[error]
        delta=max(float((gs[error+'_full'][k]-expected[k]).abs().max()) for k in params)
        scale=max(float(v.abs().max()) for v in expected.values());checks[error+'_body_head_closure']=delta/scale
    assert max(checks.values())<1e-9,checks
    return gs,stats,checks

class Moments(torch.nn.Module):
    def __init__(self,m,raw):
        super().__init__();self.model=m;self.raw=raw
    def forward(self,x):
        cap=[];self.model.body(x,cap);out=[]
        for l in LAYERS:
            z=cap[2*l+1];out.extend([z.mean((0,1)),(z-self.raw[l]['mu']).square().mean((0,1))])
        return torch.stack(out)

def full(m,raw,g):
    wrapper=Moments(m,raw);params={f'model.{k}':v.detach() for k,v in m.named_parameters()}
    tangent={f'model.{k}':v for k,v in g.items()};x=A.uniform(SEED)
    total=torch.zeros(4,C.FF,dtype=A.DT)
    for _,xx in A.chunks(x):
        def fun(pp):return torch.func.functional_call(wrapper,pp,(xx,),tie_weights=True)
        with torch.no_grad():_,j=torch.func.jvp(fun,(params,),(tangent,))
        total+=j.detach()*len(xx)/len(x)
    out={}
    for i,l in enumerate(LAYERS):
        r=raw[l];dmu=total[2*i];dvar=total[2*i+1]*r['N']/(r['N']-1)
        out[l]={'mu':dmu,'r':dmu/r['s']-r['mu']*dvar/(2*r['s']**3)}
    return out

def finite_gate(m,task,raw,gs,rank):
    base={k:v.detach().clone() for k,v in m.named_parameters()};params=dict(m.named_parameters());rows=[]
    x=A.data(SEED,task)[0]
    for name in ['P2','switch_full']:
        norm=sum(v.square().sum() for v in gs[name].values()).sqrt()
        direction={k:v/norm for k,v in gs[name].items()};pred=full(m,raw,direction)
        for eps in [1e-4,1e-5,1e-6]:
            vals=[];capacities=[]
            for sign in [-1,1]:
                with torch.no_grad():
                    for k,w in params.items():w.copy_(base[k]+sign*eps*direction[k])
                vv=fields(m);vals.append(vv)
                if name=='P2':
                    with torch.no_grad():
                        h=torch.cat([m.body(xx)[:,1:].reshape(-1,C.D) for _,xx in A.chunks(x)])
                        eig=torch.linalg.eigvalsh(h.T@h);assert int((eig>eig[-1]*1e-10).sum())==rank
                        capacities.append(float(eig[-rank:].log().sum()/2))
            with torch.no_grad():
                for k,w in params.items():w.copy_(base[k])
            for l in LAYERS:
                fd=(vals[1][l]['r']-vals[0][l]['r'])/(2*eps)
                error=rel(fd,pred[l]['r']);rows.append(dict(name=name,epsilon=eps,layer=l,error=error))
                if eps==1e-6:assert error<1e-4,rows
            if name=='P2':
                fd=(capacities[1]-capacities[0])/(2*eps);error=abs(fd-float(norm))/max(abs(fd),float(norm))
                rows.append(dict(name='P2_capacity',epsilon=eps,layer=-1,error=error))
                if eps==1e-6:assert error<1e-4,rows
    assert all(torch.equal(w,base[k]) for k,w in params.items())
    return rows

def numeric_zero(a,tol):
    a=np.array(a,copy=True);a[np.abs(a)<=tol*np.max(np.abs(a))]=0
    return a

def state(arm,task,step,gate=False):
    m,den,folder=load(arm,task,step);raw=fields(m)
    checks=reference(folder,task,step,raw)
    gs,stats,closure=gradients(m,task,gate);checks.update(closure)
    if gate:checks['finite_difference']=finite_gate(m,task,raw,gs,stats['rank'])
    arrays={};rows=[]
    for l,r in raw.items():
        for k in ['mu','s','r','responsive','p','abs_gate']:arrays[f'L{l}_{k}']=arr(r[k])
    for metric in ['SGD','Adam_fixed']:
        for name,g0 in gs.items():
            g=g0 if metric=='SGD' else {k:v/den[k] for k,v in g0.items()}
            projected=full(m,raw,g)
            for l in LAYERS:
                for target in ['r','mu']:
                    own=(g[f'h.{l}.mlp.up.weight']*(raw[l]['kr'] if target=='r' else raw[l]['m'])).sum(1)
                    value=projected[l][target]
                    prefix=f'L{l}_{metric}_{target}'
                    arrays[f'{prefix}_full_{name}']=arr(value)
                    arrays[f'{prefix}_own_{name}']=arr(own)
                    arrays[f'{prefix}_upstream_{name}']=arr(value-own)
    for l in LAYERS:
        for metric in ['SGD','Adam_fixed']:
            for target in ['r','mu']:
                for route in ['full','own','upstream']:
                    prefix=f'L{l}_{metric}_{target}_{route}'
                    for pred in [k for k in gs if k.startswith('P2')]:
                        for error in ['switch_full','fit_full','switch_body','fit_body']:
                            for tol in [1e-10,1e-8,1e-12]:
                                a=numeric_zero(arrays[f'{prefix}_{pred}'],tol);b=numeric_zero(arrays[f'{prefix}_{error}'],tol)
                                responsive=arrays[f'L{l}_responsive'];mask=responsive&(a!=0)&(b!=0)
                                cc=E.compare(a,b,mask)
                                rows.append(dict(arm=arm,seed=SEED,task=task,step=step,layer=l,metric=metric,target=target,route=route,
                                    predictor=pred,error=error,zero_tol=tol,**cc,n_responsive=int(responsive.sum()),
                                    pred_zero=int((a==0).sum()),actual_zero=int((b==0).sum()),
                                    pred_mean_all=float(a.mean()),actual_mean_all=float(b.mean()),pred_median_all=float(np.median(a)),actual_median_all=float(np.median(b)),
                                    r_median=float(np.median(arrays[f'L{l}_r'])),s_median=float(raw[l]['s'].median()),
                                    positive_token_fraction=float(raw[l]['p'].mean()),abs_gate_mean=float(raw[l]['abs_gate'].mean()),**stats))
    tag=f'{arm}_t{task}_u{step}';np.savez_compressed(OUT/'raw'/f'{tag}.npz',**arrays)
    dump(OUT/'raw'/f'{tag}.json',dict(checks=checks,stats=stats))
    return rows,checks

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gate',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'raw').mkdir(exist_ok=True)
    selected=[('SiLU',10,10000),('SiLU',100,10000)] if args.gate else STATES
    start=time.time();rows=[];checks=[]
    for arm,t,u in selected:
        rr,cc=state(arm,t,u,args.gate);rows+=rr;checks.append(dict(arm=arm,task=t,step=u,checks=cc))
        print(arm,t,u,'complete',round(time.time()-start,1),'seconds',flush=True)
        dump(OUT/('gate.json' if args.gate else 'checks.json'),checks)
    E.csvwrite(OUT/('gate_states.csv' if args.gate else 'states.csv'),rows)
    if not args.gate:
        dump(OUT/'provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            spec_sha256=M.sha(ROOT/'specs/spec_chargpt_siluphase2_1024.md'),code_sha256=M.sha(__file__),inputs=list(USED.values()),
            source_sha256={p.name:M.sha(p) for p in (ROOT/'analysis/chargpt_latepush_1023/source/src').glob('*.py')},
            elapsed_seconds=time.time()-start,training_steps=0,optimizer_created=False,seed_count=1,states=len(selected),
            limitation='Free head conditional LS predictor versus actual tied-head network; direct shared-head input updates separately recorded.'))

if __name__=='__main__':main()

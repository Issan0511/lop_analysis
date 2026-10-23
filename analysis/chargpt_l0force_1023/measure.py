#!/usr/bin/env python3
"""Exact frozen-state error/source/path ledger; no optimizer or training."""
import argparse,csv,hashlib,importlib.util,io,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('carrier',ROOT/'analysis/chargpt_l0carrier_1023/measure.py')
L=importlib.util.module_from_spec(sp);sp.loader.exec_module(L)
M,A,C,torch,np=L.M,L.A,L.C,L.torch,L.np
OUT=ROOT/'results/chargpt_l0force_1023'
DATA=M.DATA
BORROW=DATA/'chargpt_l0carrier_1023/inputs/M2'
STATES=[(9,10000),(39,10000),(99,10000),(100,100),(100,10000)]
PARTS=['self','other0','residual','LNrad','upper']

def dump(p,o):M.dump(p,o)
def med(t):return float(torch.quantile(t,.5)) if t.numel() else None
def rel(d,*v):return M.relmax(d,*v)
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));M.write_csv(p,[{k:r.get(k) for k in keys} for r in rows])

class Inputs(M.Inputs):
    def __init__(self):
        super().__init__();self.borrowed={}
        self.refs={}
        self.man=json.loads((ROOT/'results/chargpt_l0carrier_1023/backup_manifest.json').read_text())
    def model(self,arm,seed,task,step):
        if step==10000:return self.load(arm,seed,task)
        p=BORROW/arm/f'seed{seed}/ckpts/u100_task{task}.pt'
        self.borrowed[str(p)]=dict(path=str(p),sha256=M.sha(p),bytes=p.stat().st_size)
        C.L=2;m=C.GPT(arm,'M');m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));m.eval().double()
        for b in m.h:b.mlp.track_v=False
        return m
    def adam(self,m,arm,seed,task,step):
        if step!=10000:return None
        n=f'results/chargpt_cl_1019/M2/{arm}/seed{seed}/ckpts/task{task}_adam.pt'
        o=torch.load(io.BytesIO(self.tar.extractfile(n).read()),map_location='cpu',weights_only=True)
        group=o['param_groups'][0];params=list(m.named_parameters())
        assert len(group['params'])==len(params)
        for index,(name,p) in zip(group['params'],params):
            st=o['state'][index];assert st['exp_avg'].shape==p.shape
            if name=='h.0.mlp.up.weight':
                k=float(st['step']);b1,b2=group['betas'];den=(st['exp_avg_sq'].double()/(1-b2**k)).sqrt()+group['eps']
                return den,st['exp_avg'].double()/(1-b1**k)/den
        raise AssertionError('up weight missing')
    def reference(self,arm,seed,task,step,raw):
        key=arm,seed
        if key not in self.refs:
            p=BORROW/arm/f'seed{seed}/layer_stats.csv'
            self.borrowed[str(p)]=dict(path=str(p),sha256=M.sha(p),bytes=p.stat().st_size)
            self.refs[key]={(int(r['task']),int(r['u'])):r for r in csv.DictReader(p.open()) if r['layer']=='0' and r['probe']=='uniform'}
        r=self.refs[key][task,step];checks={}
        got={'r_med':med(raw['r']),'sd_med':float(raw['s'].median()),'dead':float((~raw['alive']).double().mean())}
        for k,a in got.items():
            e=float(r[k]);err=abs(a-e);tol=max(1e-6,1e-3*abs(e));assert err<=tol,(arm,seed,task,step,k,a,e)
            checks[k+'_abs']=err
        return checks

def apply_h(d,z):return d*(z-(d*z).sum(-1,keepdim=True)/d.sum(-1,keepdim=True))

def forward(m,xx):
    cap=[];h0=m.h[0](m.wte(xx),cap)
    xf=m.h[0].n2(cap[0]);z0=cap[1]
    b=m.h[1];h1=h0+b.sc*b.attn(b.n1(h0));z1=b.mlp.up(b.n2(h1));h2=h1+b.mlp.down(b.mlp.phi(z1))
    f=m.head(m.nf(h2))
    return f,xf,z0,h0,h1,h2

def gradients(m,seed,task,gate=False):
    x,y,_,_=A.data(seed,task);chain=C.ChainM(seed,task,.01);qt=torch.from_numpy(chain.P)
    W=m.h[0].mlp.up.weight;U=m.h[0].mlp.down.weight.detach();gamma=m.nf.weight.detach();vc=m.head.weight.detach()-m.head.weight.detach().mean(0)
    B=vc@(gamma[:,None]*(U-U.mean(0,keepdim=True)))
    keys=PARTS+['switch','target','sample','upper_ffn','upper_attn']
    gs={k:torch.zeros_like(W) for k in keys};checks={k:0. for k in ['secant','source','gradient','path','forward','fit']}
    stat={k:0. for k in ['Hq','KL','Hp','pmax','qmax','alpha','alpha_negative','alpha_under1']}
    N=len(x)*(x.shape[1]-1)
    for start,xx in A.chunks(x):
        f,xf,z,h0,h1,h2=forward(m,xx)
        with torch.no_grad():
            if gate:checks['forward']=max(checks['forward'],rel(f-m(xx),f))
            p=f.softmax(-1);q=torch.full_like(p,1/C.V);q[:,1:]=qt[xx[:,:-1],xx[:,1:]]
            target=torch.nn.functional.one_hot(y[start:start+len(xx)],C.V).to(A.DT)
            e=p-1/C.V;truth=q-1/C.V;sample=p-target
            mask=torch.ones_like(f[...,:1]);mask[:,0]=0
            for k,t in [('Hq',-(q*q.clamp_min(1e-300).log()).sum(-1)),('KL',(q*(q.clamp_min(1e-300).log()-f.log_softmax(-1))).sum(-1)),('Hp',-(p*p.clamp_min(1e-300).log()).sum(-1)),('pmax',p.max(-1).values),('qmax',q.max(-1).values)]:stat[k]+=float(t[:,1:].sum())/N
            alpha=(e*truth).sum(-1)/truth.square().sum(-1).clamp_min(1e-300)
            for k,t in [('alpha',alpha),('alpha_negative',(alpha<0).double()),('alpha_under1',(alpha<1).double())]:stat[k]+=float(t[:,1:].sum())/N
        gZ,g0,g1,g2,gW=torch.autograd.grad(f,[z,h0,h1,h2,W],grad_outputs=e*mask/N,retain_graph=True)
        with torch.no_grad():
            xxflat=xf.detach().reshape(-1,C.D);zz=z.detach().reshape(-1,C.FF);a=m.h[0].mlp.phi(zz);gt=m.h[0].mlp.dphi(zz)
            ff=f.detach().reshape(-1,C.V);ee=e.reshape(-1,C.V);msk=mask.reshape(-1,1)
            final=M.cent(h2.detach()).reshape(-1,C.D);sig=(final.square().mean(-1,keepdim=True)+m.nf.eps).sqrt();qn=final/sig
            delta=ff-(ff.logsumexp(-1,keepdim=True)-np.log(C.V));small=delta.abs()<1e-7;safe=torch.where(small,torch.ones_like(delta),delta)
            d=torch.where(small,(1+delta/2+delta.square()/6+delta.pow(3)/24)/C.V,torch.expm1(delta)/(C.V*safe))
            assert bool((d>0).all())
            checks['secant']=max(checks['secant'],rel(apply_h(d,ff)-ee,ee))
            coeff=d@(B*B)-(d@B).square()/d.sum(-1,keepdim=True)
            own=a*coeff/sig.square()
            ff0=a@B.T/sig
            direct=ee@B/sig
            all0=(apply_h(d,ff0)@B)/sig
            residual=(apply_h(d,ff-ff0)@B)/sig
            other=all0-own
            b=(ee@vc)*gamma
            lnrad=-(qn@U)*(qn*b).mean(-1,keepdim=True)/sig
            checks['source']=max(checks['source'],rel(own+other+residual-direct,direct,own,other,residual))
            vals={'self':own,'other0':other,'residual':residual,'LNrad':lnrad}
            for name,val in vals.items():gs[name]+=(gt*val*msk).T@xxflat/N
            ub=(g0-g2).reshape(-1,C.D)@U*gt
            uff=(g1-g2).reshape(-1,C.D)@U*gt;uat=(g0-g1).reshape(-1,C.D)@U*gt
            for name,val in [('upper',ub),('upper_ffn',uff),('upper_attn',uat)]:gs[name]+=val.T@xxflat
            direct_g=((gt*(direct+lnrad)*msk)/N+ub).T@xxflat
            checks['gradient']=max(checks['gradient'],rel(direct_g-gW,gW,direct_g))
            checks['path']=max(checks['path'],rel(gZ-(gt*(direct+lnrad)*msk/N+ub).reshape_as(gZ),gZ))
            gs['switch']+=gW.detach()
        gs['target']+=torch.autograd.grad(f,W,grad_outputs=truth*mask/N,retain_graph=True)[0].detach()
        gs['sample']+=torch.autograd.grad(f,W,grad_outputs=sample*mask/N)[0].detach()
    gs['fit']=gs['switch']-gs['target'];gs['negative_target']=-gs['target']
    checks['gradient']=max(checks['gradient'],rel(gs['switch']-sum(gs[k] for k in PARTS),gs['switch'],*[gs[k] for k in PARTS]))
    if gate:
        directfit=torch.zeros_like(W)
        for start,xx in A.chunks(x):
            f=m(xx);q=torch.full_like(f,1/C.V);q[:,1:]=qt[xx[:,:-1],xx[:,1:]]
            loss=-(q[:,1:]*f[:,1:].log_softmax(-1)).sum()/N
            directfit+=torch.autograd.grad(loss,W)[0].detach()
        checks['fit']=rel(directfit-gs['fit'],directfit,gs['fit'])
    assert max(checks.values())<=1e-9,checks
    return gs,stat,checks

def project(gs,raw,m,metric,den=None,momentum=None):
    W=raw['W'];wh=raw['wh'];mean=raw['m'];mu=raw['mu'];s=raw['s']
    # Covariance is reconstructed from the original fixed uniform probe.
    xf,_=A.moments(m,A.uniform(raw['seed']))[0];xc=xf-xf.mean(0);cov=xc.T@xc/(len(xf)-1)
    kr=mean[None,:]/s[:,None]-mu[:,None]*(W@cov)/s[:,None].pow(3)
    assert float((kr*W).sum(1).abs().max())<1e-10
    data={};rows={}
    gg=dict(gs)
    if momentum is not None:gg['momentum']=momentum
    for name,g in gg.items():
        if den is not None and name!='momentum':g=g/den
        smu=(g*mean).sum(1);sr=(g*kr).sum(1)
        radial=(g*wh).sum(1)*(wh*mean).sum(1)
        data[name+'_mu']=smu;data[name+'_r']=sr;data[name+'_mu_radial']=radial;data[name+'_mu_tangent']=smu-radial
    for group,mask in [('alive',raw['alive']),('all',torch.ones_like(raw['alive']))]:
        for name,v in data.items():
            rows[f'{group}_{name}_med']=med(v[mask]);rows[f'{group}_{name}_abs']=med(v[mask].abs());rows[f'{group}_{name}_s']=float((v[mask]>0).double().mean()) if mask.any() else None
        for error in ['fit','switch']:
            sinking=mask&(data[error+'_r']>0);cnt=int(sinking.sum())
            rows[f'{group}_{error}_sink_n']=cnt
            rows[f'{group}_{error}_self_positive']=float((data['self_r'][sinking]>0).double().mean()) if cnt else None
            rows[f'{group}_{error}_self_removal']=float(((data[error+'_r']-data['self_r'])[sinking]<=0).double().mean()) if cnt else None
            parts=PARTS+(['negative_target'] if error=='fit' else [])
            denom=sum(data[k+'_r'].abs() for k in parts);ok=mask&(denom>0)
            rows[f'{group}_{error}_self_share']=med(data['self_r'][ok].abs()/denom[ok])
            rows[f'{group}_{error}_self_carries']=bool(cnt>=32 and rows[f'{group}_{error}_self_positive']>=.8 and rows[f'{group}_{error}_self_removal']>=.5)
    return rows,data

def state(inp,arm,seed,task,step,gate=False):
    m=inp.model(arm,seed,task,step);stats,raw=L.fields(m,A.uniform(seed));raw['seed']=seed
    refs=inp.reference(arm,seed,task,step,raw);gs,conf,checks=gradients(m,seed,task,gate)
    ad=inp.adam(m,arm,seed,task,step)
    rows=[];arrays={'alive':raw['alive'].numpy(),'mu':raw['mu'].numpy(),'s':raw['s'].numpy(),'r':raw['r'].numpy()}
    for metric,den,mom in [('SGD',None,None)]+([('Adam_fixed',ad[0],ad[1])] if ad else []):
        rr,data=project(gs,raw,m,metric,den,mom)
        rows.append(dict(arm=arm,seed=seed,task=task,step=step,metric=metric,n_alive=int(raw['alive'].sum()),r_med=med(raw['r']),**conf,**refs,**{f'check_{k}':v for k,v in checks.items()},**rr))
        arrays.update({metric+'_'+k:v.numpy() for k,v in data.items()})
    if gate:
        xf,z=A.moments(m,A.uniform(seed))[0];W=raw['W'];g=gs['switch'];s=raw['s'];mu=raw['mu'];xc=xf-xf.mean(0);cov=xc.T@xc/(len(xf)-1)
        kr=raw['m']/s[:,None]-mu[:,None]*(W@cov)/s[:,None].pow(3)
        direction=g/g.norm();eps=1e-4
        zp=z+eps*(xf@direction.T);zm=z-eps*(xf@direction.T)
        fd=(zp.mean(0)/zp.std(0)-zm.mean(0)/zm.std(0))/(2*eps)
        pred=(kr*direction).sum(1);err=rel(fd-pred,pred);assert err<1e-6,err
        for r in rows:r['finite_difference_rel']=err
    return rows,arrays

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['gate','run']);args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    inp=Inputs();start=time.monotonic();code_sha=M.sha(__file__)
    if args.mode=='gate':
        assert M.sha(M.BUNDLE)==M.BUNDLE_SHA
        rows=[]
        for arm in A.ARMS:
            for task,step in [(9,10000),(100,100),(100,10000)]:
                r,_=state(inp,arm,0,task,step,True);rows.extend(r);print('gate',arm,task,step,round(time.monotonic()-start,1),flush=True)
        dump(OUT/'gate.json',dict(passed=True,code_sha256=code_sha,rows=rows,seconds=time.monotonic()-start));return
    gate=json.loads((OUT/'gate.json').read_text());assert gate['passed'] and gate['code_sha256']==code_sha
    rows=[]
    for seed in range(10):
        for arm in A.ARMS:
            p=OUT/'parts'/f'{arm}_seed{seed}.csv';ar=OUT/'raw'/f'{arm}_seed{seed}.npz'
            meta=p.with_suffix('.json')
            if p.exists() and ar.exists() and meta.exists():
                saved=json.loads(meta.read_text())
                assert saved['code_sha256']==code_sha,'Restart in a clean result directory after code changes'
                assert saved['csv_sha256']==M.sha(p) and saved['raw_sha256']==M.sha(ar)
                inp.borrowed.update({r['path']:r for r in saved['borrowed']})
                rows.extend(list(csv.DictReader(p.open())));continue
            local=[];raw={};ts=time.monotonic()
            for task,step in STATES:
                r,arr=state(inp,arm,seed,task,step);local.extend(r);raw.update({f't{task}_u{step}_{k}':v for k,v in arr.items()})
            csvwrite(p,local);ar.parent.mkdir(exist_ok=True);np.savez_compressed(ar,**raw);rows.extend(local)
            dump(meta,dict(code_sha256=code_sha,csv_sha256=M.sha(p),raw_sha256=M.sha(ar),borrowed=list(inp.borrowed.values())))
            print(arm,seed,round(time.monotonic()-ts,1),flush=True)
        A.CACHE.clear();A.UCACHE.clear()
    csvwrite(OUT/'states.csv',rows)
    dump(OUT/'provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_sha256=code_sha,spec_sha256=M.sha(ROOT/'specs/spec_chargpt_l0force_1023.md'),training_steps=0,optimizer_created=False,device='CPU',threads=torch.get_num_threads(),states=100,rows=len(rows),seconds=time.monotonic()-start,bundle=dict(path=str(M.BUNDLE),sha256=M.BUNDLE_SHA),borrowed=list(inp.borrowed.values())))

if __name__=='__main__':main()

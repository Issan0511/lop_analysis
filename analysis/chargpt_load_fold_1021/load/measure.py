#!/usr/bin/env python3
"""CPU-only fixed-state load ledger. Checks gate all production measurements."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, csv, math, argparse, hashlib, subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
SOURCE = Path('/home/issan/Projects/claude/wt/chargpt_cl_1019')
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/chargpt_load_fold_1021/load'
sys.path.insert(0, str(SOURCE / 'src'))
import chargpt_cl_1019_kune as K
C = K.C
torch.set_num_threads(8)
DT = torch.float64
SEED = 1000
CHUNK = 8
ARMS = [('M', a) for a in ['C3','SiLU','GELU','ReLUcap','SiLUcap','GELUcap']] + [('M2', a) for a in ['C3','SiLU']]
CACHE = {}

def data(t):
    if t not in CACHE:
        chain = C.ChainM(SEED,t,0.01)
        seq = chain.sample(C.M_PROBE, C.cpu_gen(SEED,'probe',t,'chain'))
        ev = chain.sample(C.M_EVAL, C.cpu_gen(SEED,'probe',t,'eval'))
        CACHE[t] = (seq[:,:C.T], seq[:,1:], ev, chain.H)
    return CACHE[t]

def directory(box,arm):
    return SOURCE / 'results/chargpt_cl_1019/quicklook' / box / arm / 'seed1000'

def load(box,arm,t,stage='end'):
    C.L = 4 if box == 'M' else 2
    # Construct the actual source model, then restore every checkpoint tensor.
    m = C.GPT(arm,'M')
    p = directory(box,arm)/'ckpts'/('init.pt' if t == 0 else f'{"u100_" if stage=="u100" else ""}task{t}.pt')
    m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
    m.eval().double()
    for b in m.h:
        # Source updates V even in eval() when grad is enabled. These arms use V
        # only for the training cap; disable tracking for pure fixed-state AD.
        b.mlp.track_v = False
    assert m.wte.weight is m.head.weight
    return m

def chunks(x):
    for i in range(0,len(x),CHUNK):
        yield i,x[i:i+CHUNK]

def ce(logits,y=None):
    if y is None:
        return (logits.logsumexp(-1)-logits.mean(-1)).mean()
    return F.cross_entropy(logits.reshape(-1,C.V),y.reshape(-1))

@torch.no_grad()
def moments(m,x):
    ys=[[] for _ in m.h]; zs=[[] for _ in m.h]
    for _,xx in chunks(x):
        cap=[]; m.body(xx,cap)
        for l,b in enumerate(m.h):
            ys[l].append(b.n2(cap[2*l]).reshape(-1,C.D))
            zs[l].append(cap[2*l+1].reshape(-1,C.FF))
    return [(torch.cat(y),torch.cat(z)) for y,z in zip(ys,zs)]

@torch.no_grad()
def head(m,x,y,nperm=1000):
    hs=[]; fs=[]
    for _,xx in chunks(x):
        h=m.body(xx); hs.append(h.reshape(-1,C.D)); fs.append(m.head(h).reshape(-1,C.V))
    h=torch.cat(hs); f=torch.cat(fs); yy=y.reshape(-1)
    vc=m.head.weight-m.head.weight.mean(0)
    fm=(F.one_hot(yy,C.V).to(DT)-1/C.V).T@h
    ff=float(fm.square().sum()); A=float((vc*fm).sum())
    aid=float((f.gather(1,yy[:,None]).squeeze(1)-f.mean(1)).sum())
    loss=float(ce(f,yy)); D=len(yy)*(math.log(C.V)-loss)
    J=float((f.logsumexp(1)-f.mean(1)-math.log(C.V)).sum())
    gen=torch.Generator().manual_seed(1021000)
    mat=vc@fm.T
    null=torch.stack([mat[torch.randperm(C.V,generator=gen),torch.arange(C.V)].sum()/ff for _ in range(nperm)]).numpy()
    kap=A/ff
    r=dict(N=len(yy),feature_dim=C.D,kappa=kap,cos2=A*A/(ff*float(vc.square().sum())),Q=A/math.sqrt(ff),F_norm=math.sqrt(ff),v_norm=float(vc.norm()),A=A,A_id=aid,identity_rel=abs(A-aid)/max(abs(A),abs(aid),1e-300),L=loss,D=D,J=J,kappa_lb=D/ff,kappa_over_lb=A/D if D>0 else float('nan'),lemma_violation=int(A-D < -1e-10*max(1,abs(A),abs(D))),null_quantile=float((null<=kap).mean()),null_sd=float(null.std(ddof=1)))
    r.update({f'null_q{p:02d}':float(np.percentile(null,p)) for p in [1,5,50,95,99]})
    return r

def gradient(m,x,y=None):
    m.zero_grad(set_to_none=True)
    for i,xx in chunks(x):
        (ce(m(xx),None if y is None else y[i:i+len(xx)])*len(xx)/len(x)).backward()
    return {n:p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for n,p in m.named_parameters()}

class Position(nn.Module):
    def __init__(self,m):
        super().__init__(); self.model=m
    def forward(self,x):
        cap=[]; self.model.body(x,cap)
        return cap[-1].mean((0,1))

def full(m,x,g):
    wrapper=Position(m)
    p={f'model.{n}':v.detach() for n,v in m.named_parameters()}
    tangent={f'model.{n}':g[n] for n in g}
    result=torch.zeros(C.FF,dtype=DT)
    for _,xx in chunks(x):
        def fn(pp):
            return torch.func.functional_call(wrapper,pp,(xx,),tie_weights=True)
        _,j=torch.func.jvp(fn,(p,),(tangent,))
        result+=j.detach()*len(xx)/len(x)
    return result

def agreement(a,b,mask):
    mk=mask&(a!=0)&(b!=0)
    return (float((a[mk].sign()==b[mk].sign()).double().mean()) if mk.any() else float('nan'),int(mk.sum()))

def units(m,x,g,gf=None):
    rows=[]; raw=[]
    for l,(y,z) in enumerate(moments(m,x)):
        b=m.h[l]; ybar=y.mean(0); mu=z.mean(0); alive=(z>0).any(0)
        own=(g[f'h.{l}.mlp.up.weight']*ybar).sum(1)
        kn=y@ybar
        d=(kn[:,None]*b.mlp.phi(z)*b.mlp.dphi(z)).T@y
        scalar=d@ybar
        agree,n=agreement(own,scalar,alive)
        r=dict(layer=l,n_units=C.FF,n_alive=int(alive.sum()),m_median=float(mu.median()),closure_max=float((mu-b.mlp.up.weight.detach()@ybar).abs().max()),own_sink=float((own[alive]>0).double().mean()) if alive.any() else float('nan'),own_abs_median=float(own.abs().median()),own_abs_median_alive=float(own[alive].abs().median()) if alive.any() else float('nan'),self_agreement=agree,self_n=n)
        if gf is not None and l==len(m.h)-1:
            a,n=agreement(own,gf,alive)
            r.update(full_sink=float((gf[alive]>0).double().mean()) if alive.any() else float('nan'),full_own_agreement=a,full_own_n=n,full_abs_median=float(gf.abs().median()))
        rows.append(r); raw.append(dict(m=mu,alive=alive,own=own,self_scalar=scalar,ybar=ybar))
    return rows,raw

@torch.no_grad()
def eval_metric(m,ev,H):
    val=0.
    for _,s in chunks(ev):
        val+=float(ce(m(s[:,:C.T])[:,1:],s[:,2:]))*len(s)/len(ev)
    return val-H

@torch.no_grad()
def loss_value(m,x,y=None):
    return sum(float(ce(m(xx),None if y is None else y[i:i+len(xx)]))*len(xx)/len(x) for i,xx in chunks(x))

def rounding_bound(text):
    # CSV writer uses .9g. Half of last decimal place, determined from value.
    v=abs(float(text))
    return 0.5*10**(math.floor(math.log10(v))-8) if v else 0.

def checks():
    rec=[]
    def add(name,ok,**kw):
        r=dict(check=name,pass_=bool(ok),**kw); rec.append(r)
        print(json.dumps(r),flush=True)
    x,y,ev,H=data(1); xu=C.uniform_probe_M(SEED,'cpu')
    # Representative full-size probes, three activation families and both depths.
    for box,arm in [('M','C3'),('M','SiLU'),('M','GELU'),('M','ReLUcap'),('M','SiLUcap'),('M','GELUcap'),('M2','C3'),('M2','SiLU')]:
        m=load(box,arm,1); tag=f'{box}/{arm}'
        add('tied_weight',m.wte.weight is m.head.weight and m.wte.weight.data_ptr()==m.head.weight.data_ptr(),arm=tag)
        h=head(m,x,y)
        add('lemma_identity',h['identity_rel']<=1e-10,arm=tag,relative_error=h['identity_rel'])
        g=gradient(m,xu); gf=full(m,x,g); ur,raw=units(m,x,g,gf)
        add('position_closure',max(r['closure_max'] for r in ur)<=1e-10,arm=tag,max_abs=max(r['closure_max'] for r in ur))
        # Frozen-feature head mutation: preserve embedding and replace only head.
        old=m.head.weight
        m.head.weight=nn.Parameter(-old.detach().clone())
        hm=head(m,x,y)
        add('negate_head',h['kappa']>0 and hm['kappa']<0 and abs(hm['kappa']+h['kappa'])<1e-12,arm=tag,kappa=h['kappa'],mutant_kappa=hm['kappa'],note='head-only mutant intentionally untied; embedding unchanged')
        m.head.weight=old
        gc=gradient(m,xu,torch.zeros_like(xu))
        uclass,rclass=units(m,x,gc)
        alive=raw[-1]['alive']; go=raw[-1]['own']; go1=rclass[-1]['own']
        flips=int(((go.sign()!=go1.sign())&alive).sum())
        add('one_class_mutant',flips>0 and ur[-1]['own_sink']!=uclass[-1]['own_sink'],arm=tag,sign_flips=flips,uniform_sink=ur[-1]['own_sink'],one_class_sink=uclass[-1]['own_sink'])
        # FD follows the same chain position and uniform loss as the AD definition.
        ids=torch.where(alive & (go.abs()>go.abs().max()*1e-3) & (gf.abs()>gf.abs().max()*1e-3))[0]
        ids=ids[torch.linspace(0,len(ids)-1,min(3,len(ids))).long()].tolist()
        own_fd=[]; full_fd=[]
        base={n:p.detach().clone() for n,p in m.named_parameters()}
        eps=1e-6
        for j in ids:
            w=m.h[-1].mlp.up.weight; yb=raw[-1]['ybar']
            with torch.no_grad(): w[j].add_(eps*yb)
            lp=loss_value(m,xu)
            with torch.no_grad(): w[j].sub_(2*eps*yb)
            lm=loss_value(m,xu)
            with torch.no_grad(): w.copy_(base[f'h.{len(m.h)-1}.mlp.up.weight'])
            fd=(lp-lm)/(2*eps); exact=float(go[j])
            own_fd.append(dict(unit=j,ad=exact,fd=fd,relative_error=abs(fd-exact)/max(abs(fd),abs(exact),1e-300)))
        # Use smaller steps for ReLU boundary crossings; report every attempted step.
        for eps_full in [1e-6,1e-7,1e-8]:
            vals=[]
            for sign in [1,-1]:
                with torch.no_grad():
                    for n,p in m.named_parameters(): p.copy_(base[n]+sign*eps_full*g[n])
                vals.append(moments(m,x)[-1][1].mean(0))
            fd=(vals[0]-vals[1])/(2*eps_full)
            trial=[dict(unit=j,ad=float(gf[j]),fd=float(fd[j]),relative_error=float(abs(fd[j]-gf[j])/torch.maximum(fd[j].abs(),gf[j].abs()).clamp_min(1e-300))) for j in ids]
            full_fd.append(dict(epsilon=eps_full,units=trial))
            if all(r['relative_error']<=1e-4 for r in trial): break
        with torch.no_grad():
            for n,p in m.named_parameters(): p.copy_(base[n])
        add('own_finite_difference',bool(ids) and all(r['relative_error']<=1e-4 for r in own_fd),arm=tag,epsilon=eps,units=own_fd)
        add('full_finite_difference',bool(ids) and all(r['relative_error']<=1e-4 for r in full_fd[-1]['units']),arm=tag,trials=full_fd)
        record=next(r for r in csv.DictReader(open(directory(box,arm)/'per_task.csv')) if int(r['task'])==1)
        got=eval_metric(m,ev,H); expected=float(record['end_metric'])
        csv_tol=rounding_bound(record['end_metric'])
        tol=2e-6*abs(expected)+csv_tol
        add('recorded_eval_rounding',abs(got-expected)<=tol,arm=tag,computed_float64=got,recorded=expected,abs_error=abs(got-expected),relative_error=abs(got-expected)/abs(expected),tolerance=tol,rtol=2e-6,csv_rounding_bound=csv_tol,H_recomputed=H,H_recorded=float(record['H']),definition='mean CE over eval_seq[:,2:] (256x127 targets) minus H',dtype_evidence='chargpt_cl_1019.py:152-165 creates default-float32 Embedding/Linear; :195-198 build_model only .to(device), no .double(); :468-471 eval_M computes CE in model dtype before float(ce)-H; :697 writes .9g. float64 recomputation differs from float32 evaluation and CSV rounding. Only this check uses the user-authorized rtol=2e-6.')
    result=dict(all_pass=all(r['pass_'] for r in rec),checks=rec,provenance=provenance())
    (OUT/'checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return result

def provenance():
    paths=[ROOT/'specs/spec_chargpt_load_fold_1021.md',
           SOURCE/'src/chargpt_cl_1019.py', SOURCE/'src/chargpt_cl_1019_kune.py']
    return dict(source_root=str(SOURCE),source_head=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),required_source_head='f71be184',spec_commit='8122b8b1',seed=SEED,device='cpu',dtype='float64',threads=8,torch=torch.__version__,numpy=np.__version__,python=sys.version,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},probe=dict(sequences=C.M_PROBE,tokens_per_sequence=C.T,eval_sequences=C.M_EVAL,chain_alpha=0.01),null=dict(permutations=1000,seed=1021000),rounding_exception='recorded_eval_rounding only: rtol 2e-6 + .9g rounding bound, explicitly authorized by user')

def writecsv(name,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def measure():
    check=json.loads((OUT/'checks.json').read_text())
    if not check['all_pass']: raise RuntimeError('Required checks failed; production calculation is prohibited.')
    if check['provenance']['script_sha256'] != provenance()['script_sha256']:
        raise RuntimeError('Script changed after checks; rerun checks.')
    heads=[]; unitrows=[]; xu=C.uniform_probe_M(SEED,'cpu')
    for box,arm in ARMS:
        ts=([1,10,25,50] if 'cap' in arm else list(range(1,11))+[25]+list(range(39,51))+(list(range(89,101)) if box=='M2' else []))
        k1=None
        states=[(t,'init' if t==0 else 'end') for t in [0]+ts]
        states += sorted((int(p.stem.split('task')[1]),'u100') for p in (directory(box,arm)/'ckpts').glob('u100_task*.pt') if not p.stem.endswith('_adam'))
        for t,stage in states:
            m=load(box,arm,t,stage); x,y,_,_=data(max(1,t))
            h=head(m,x,y); h.update(box=box,arm=arm,task=t,stage=stage)
            if t==1 and stage=='end': k1=h['kappa']
            h['kappa_ratio']=h['kappa']/k1 if k1 is not None else float('nan'); heads.append(h)
            g=gradient(m,xu); gf=full(m,x,g); rows,raw=units(m,x,g,gf)
            xn,yn,_,_=data(t+1); gn=gradient(m,xn,yn); rn,raw_next=units(m,xn,gn)
            for r in rows:
                if l_should_include(box,arm,r['layer'],len(m.h)):
                    r.update(box=box,arm=arm,task=t,stage=stage)
                    if r['layer']==len(m.h)-1:
                        r.update(next_sink=rn[-1]['own_sink'],next_minus_uniform=rn[-1]['own_sink']-r['own_sink'])
                    unitrows.append(r)
            # Auditable per-unit values, all 512 entries; tables retain state summaries.
            payload={f'L{l}_{n}':v.numpy() for l,q in enumerate(raw) for n,v in q.items() if n!='ybar'}
            payload['G_full_last']=gf.numpy()
            payload['next_own_last']=raw_next[-1]['own'].numpy()
            payload['next_alive_last']=raw_next[-1]['alive'].numpy()
            suffix='_u100' if stage=='u100' else ''
            np.savez_compressed(OUT/f'units_{box}_{arm}_t{t}{suffix}.npz',**payload)
            writecsv('table_head.csv',heads); writecsv('table_units.csv',unitrows)
            print(f'measured {box}/{arm} t{t} {stage}',flush=True)
    (OUT/'provenance.json').write_text(json.dumps(provenance(),indent=2)+'\n')

def l_should_include(box,arm,l,n):
    return l==n-1 or (box=='M' and arm in ['C3','SiLU','GELU'])

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    mode=argparse.ArgumentParser(); mode.add_argument('mode',choices=['checks','measure']); args=mode.parse_args()
    if args.mode=='checks': checks()
    else: measure()

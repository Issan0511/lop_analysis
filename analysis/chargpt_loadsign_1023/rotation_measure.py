#!/usr/bin/env python3
"""Spec section 10: readout alignment and endpoint rotation; saved states only."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import measure as M
A, C, torch, np = M.A, M.C, M.torch, M.np
ROOT = M.ROOT
OUT = M.OUT/'rotation'
SOURCE = Path('/home/issan/Projects/claude/wt/chargpt_cl_1019')
ELR = SOURCE/'results/chargpt_cl_1019/row5/elr/elr'
ARCHIVE = M.DATA/'chargpt_loadsign_1023/results/chargpt_loadsign_1023/raw_loading'


def clean(x):
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def dump(path, value): M.dump(path, clean(value))


def cosine(a, b):
    den = a.norm(dim=-1)*b.norm(dim=-1)
    return torch.where(den>0, (a*b).sum(-1)/den, float('nan')).clamp(-1, 1)


def midrank(x):
    x=np.asarray(x); order=np.argsort(x, kind='stable'); ranked=np.empty(len(x), dtype=float)
    starts=np.r_[0, np.flatnonzero(np.diff(x[order]))+1]; ends=np.r_[starts[1:],len(x)]
    for lo,hi in zip(starts,ends): ranked[order[lo:hi]]=(lo+hi-1)/2+1
    return ranked


def rho(a, b):
    a,b=np.asarray(a),np.asarray(b); keep=np.isfinite(a)&np.isfinite(b); a,b=a[keep],b[keep]
    if len(a)<3 or len(np.unique(a))<2 or len(np.unique(b))<2: return float('nan')
    return float(np.corrcoef(midrank(a),midrank(b))[0,1])


def tiny_checks():
    assert np.array_equal(midrank([7,2,2,9]),[3,1.5,1.5,4])
    assert abs(rho([1,2,2,4],[4,3,3,1])+1)<1e-14
    a=torch.tensor([[1.,0.],[1.,0.],[0.,0.]],dtype=A.DT)
    b=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],dtype=A.DT)
    got=1-cosine(a,b)
    assert got[0]==0 and got[1]==1 and torch.isnan(got[2])
    return dict(ties=True,rotation=True)


@torch.no_grad()
def calculate(m, seed, task, explicit=False):
    x,labels,_,_=A.data(seed,task)
    ybar,alive,_,vc,vl,kappa=M.task_fields(m,x,labels)
    assert kappa>0, (seed,task,kappa)
    mlp=m.h[1].mlp; U=mlp.down.weight; T=m.nf.weight[:,None]*(U-U.mean(0,keepdim=True)); G=T.T@T
    B=torch.zeros(C.V,C.FF,dtype=A.DT); Fr=torch.zeros(C.V,C.D,dtype=A.DT)
    for i,xx in A.chunks(x):
        cap=[]; m.body(xx,cap); r=cap[2].reshape(-1,C.D); a=mlp.phi(cap[3].reshape(-1,C.FF))
        sig=(M.cent(r+mlp.down(a)).square().mean(-1,keepdim=True)+m.nf.eps).sqrt()
        xi=torch.nn.functional.one_hot(labels[i:i+len(xx)].reshape(-1),C.V).to(A.DT)-1/C.V
        B+=xi.T@(a/sig); Fr+=xi.T@(M.cent(r)*m.nf.weight/sig)
    ferr=M.relmax(kappa*(B@T.T+Fr)-vl,vl); assert ferr<1e-10
    E=torch.zeros(C.V,C.FF,dtype=A.DT)
    radial={s:torch.zeros(C.FF,dtype=A.DT) for s in ['self','other','residual','load']}
    independent=torch.zeros(C.FF,dtype=A.DT)
    xu=A.uniform(seed); N=xu.numel()
    for _,xx in A.chunks(xu):
        cap=[]; h=m.body(xx,cap).reshape(-1,C.D); r=cap[2].reshape(-1,C.D); z=cap[3].reshape(-1,C.FF)
        y=m.h[1].n2(r); k=y@ybar; gate=mlp.dphi(z); w=k[:,None]*gate
        sig=(M.cent(r+mlp.down(mlp.phi(z))).square().mean(-1,keepdim=True)+m.nf.eps).sqrt()
        q=M.cent(r+mlp.down(mlp.phi(z)))/sig; e=m.head(h).softmax(-1)-1/C.V
        E+=e.T@(w/sig)/N
        qU=q@U; gamma=m.nf.weight
        rl=-qU*(q*(gamma*(e@vl))).mean(-1,keepdim=True)/sig
        rr=-qU*(q*(gamma*(e@(kappa*Fr)))).mean(-1,keepdim=True)/sig
        rs=-qU*(q@(gamma[:,None]*T))*(e@(kappa*B))/(C.D*sig)
        for s,v in [('load',rl),('residual',rr),('self',rs),('other',rl-rr-rs)]: radial[s]+=(w*v).sum(0)/N
        direct=M.cent(gamma*(e@vl))@U/sig
        independent+=(w*(direct+rl)).sum(0)/N
    cross=B.T@E
    R=T@cross-T*cross.diag()[None,:]
    inner=(T*R).sum(0); alpha=cosine(T.T,R.T)
    direct=dict(self=kappa*G.diag()*(B*E).sum(0),other=kappa*inner,residual=kappa*((Fr@T)*E).sum(0))
    direct['load']=sum(direct.values())
    raw={f'{s}_{l}':v for s in direct for l,v in [('direct',direct[s]),('exact',direct[s]+radial[s])]}
    raw['radial']=radial['load']; raw['cos']=alpha; raw['alive']=alive
    raw['T_norm']=T.norm(dim=0); raw['R_norm']=R.norm(dim=0)
    add=M.relmax(raw['load_exact']-independent,independent); assert add<1e-10
    assert torch.equal(alpha>0,raw['other_direct']>0)
    explicit_err=0.
    if explicit:
        for i in [0,1,127,255,511]:
            rsum=sum(T[:,j]*torch.dot(B[:,j],E[:,i]) for j in range(C.FF) if j!=i)
            explicit_err=max(explicit_err,M.relmax(R[:,i]-rsum,rsum))
        assert explicit_err<1e-10
    finite=alive&torch.isfinite(alpha)
    row=dict(seed=seed,task=task,n_alive=int(alive.sum()),n_cos_finite=int(finite.sum()),n_cos_missing=int((alive&~finite).sum()),
             kappa=kappa,cos_positive=float((alpha[alive]>0).double().mean()),
             cos_positive_finite=float((alpha[finite]>0).double().mean()) if finite.any() else float('nan'),
             F_identity_error=ferr,additivity_error=add,explicit_sum_error=explicit_err)
    for q in [10,25,50,75,90]: row[f'cos_q{q}']=float(torch.quantile(alpha[finite],q/100)) if finite.any() else float('nan')
    for key,v in raw.items():
        if key.endswith(('direct','exact')) or key=='radial':
            row[f's_{key}']=A.sink(v,alive); row[f'abs_{key}']=M.median(v[alive].abs())
    dirs={k:v.detach().clone() for k,v in [('T',T.T),('U',U.T),('W',mlp.up.weight)]}
    return row,raw,dirs,ybar,vl


class Inputs(M.Inputs):
    def __init__(self):
        super().__init__(); self.loading={}
        mani=json.loads((M.OUT/'backup_manifest.json').read_text())
        self.hashes={Path(r['backup']).name:r['sha256'] for r in mani['files'] if '/raw_loading/' in r['backup']}

    def loading_ref(self,arm,seed):
        if (arm,seed) not in self.loading:
            path=ARCHIVE/f'{arm}_seed{seed}.npz'; assert M.sha(path)==self.hashes[path.name]
            self.loading[arm,seed]=np.load(path)
        return self.loading[arm,seed]


def state(inp,arm,seed,task,explicit=False):
    m=inp.load(arm,seed,task); row,raw,dirs,_,_=calculate(m,seed,task,explicit)
    old=inp.loading_ref(arm,seed); err=0.
    assert np.array_equal(raw['alive'].numpy(),old[f't{task}_alive'])
    for s in ['self','other','residual']:
        for l in ['direct','exact']:
            ref=torch.from_numpy(old[f't{task}_{s}_{l}_K']); v=raw[f'{s}_{l}']
            err=max(err,M.relmax(v-ref,ref)); assert torch.equal(v>0,ref>0),(arm,seed,task,s,l)
    assert err<1e-10,err
    row.update(arm=arm,previous_max_error=err)
    return row,raw,dirs


def run(inp,gate=False):
    rows=[]; start=time.monotonic()
    for seed in ([0] if gate else range(10)):
        for arm in A.ARMS:
            rr=[]; arrays={}; prev=None; ts=time.monotonic()
            for task in A.TASKS:
                row,raw,dirs=state(inp,arm,seed,task,gate and task in [1,9,50,100])
                pair=prev is not None and task==prev[0]+1
                raw['prev_alive']=prev[1]['alive'] if pair else torch.zeros(C.FF,dtype=torch.bool)
                for key in ['T','U','W']:
                    rot=1-cosine(prev[2][key],dirs[key]) if pair else torch.full((C.FF,),float('nan'),dtype=A.DT)
                    raw[f'rot{key}']=rot
                    vals=rot[raw['alive']&torch.isfinite(rot)]
                    row[f'rot{key}_median']=M.median(vals)
                row['rotation_pair']=int(pair); rr.append(row)
                arrays.update({f't{task}_{k}':v.numpy() for k,v in raw.items()}); prev=(task,raw,dirs)
            rows.extend(rr)
            if not gate:
                M.write_csv(OUT/'parts'/f'{arm}_seed{seed}.csv',[clean(r) for r in rr])
                path=OUT/'raw'/f'{arm}_seed{seed}.npz'; path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,**arrays)
            print(f'{"gate" if gate else "run"} {arm} seed{seed}: {time.monotonic()-ts:.1f}s',flush=True)
        A.CACHE.clear(); A.UCACHE.clear()
    if gate:
        dump(OUT/'gate.json',dict(passed=True,code_sha256=M.sha(__file__),rows=len(rows),tiny_checks=tiny_checks(),
            max_previous_error=max(r['previous_max_error'] for r in rows),max_explicit_error=max(r['explicit_sum_error'] for r in rows),
            signs_exact=True,seconds=time.monotonic()-start));return
    M.write_csv(OUT/'states.csv',[clean(r) for r in rows])
    dump(OUT/'provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256=M.sha(__file__),spec_sha256=M.sha(ROOT/'specs/spec_chargpt_loadsign_1023.md'),bundle_sha256=M.BUNDLE_SHA,
        states=len(rows),training_steps=0,optimizer_created=False,checkpoint_extracted=False,threads=torch.get_num_threads(),
        device='cpu',dtype='float64',seconds=time.monotonic()-start,max_previous_error=max(r['previous_max_error'] for r in rows)))


def inventory():
    records=[]
    for seed in range(10):
        for task in [50,100]:
            p=ELR/f'seed{seed}/state_task{task}.pt'
            records.append(dict(seed=seed,task=task,path=str(p),exists=p.exists(),bytes=p.stat().st_size if p.exists() else None,
                                sha256=M.sha(p) if p.exists() else None))
    # Search paths are recorded, including the absence of ELR entries in the original bundle.
    import tarfile
    with tarfile.open(M.BUNDLE,'r:') as tar:
        elr_members=[m.name for m in tar.getmembers() if '/elr/' in m.name.lower()]
    manifests=[]
    for base in [ROOT/'results',M.DATA,SOURCE/'results/chargpt_cl_1019']:
        for p in base.rglob('*manifest*.json'):
            try: txt=p.read_text()
            except (OSError,UnicodeError): continue
            if 'row5/elr' in txt or 'state_task100.pt' in txt:
                manifests.append(dict(path=str(p),sha256=M.sha(p)))
    src=SOURCE/'analysis/chargpt_cl_1019_quicklook/row5_elr.py'
    dump(OUT/'elr_inventory.json',dict(checkpoints=records,bundle_ELR_members=elr_members,matching_backup_manifests=manifests,
        source_path=str(src),source_sha256=M.sha(src),note='Local saved files only; no remote training machines contacted, no new training.'))


def load_elr(seed,task):
    p=ELR/f'seed{seed}/state_task{task}.pt'; sd=torch.load(p,map_location='cpu',weights_only=True)
    C.L=2; m=C.GPT('C3','M'); m.load_state_dict(sd); m.eval().double()
    for b in m.h: b.mlp.track_v=False
    rnpath=SOURCE/f'results/chargpt_cl_1019/M2/C3/seed{seed}/unit_stats.npz'
    rn=torch.from_numpy(np.load(rnpath)['rownorm'][:,0,:]).float(); k=(rn[task-1]/rn[0]).double()
    m.h[0].mlp.phi=lambda z,k=k:torch.nn.functional.relu(k*z)
    m.h[0].mlp.dphi=lambda z,k=k:k*(k*z>0).to(z.dtype)
    return m,k,rnpath


@torch.no_grad()
def validate_elr(m,k,seed,task):
    x,_,_,_=A.data(seed,task); ys,zs=[],[]
    for _,xx in A.chunks(x):
        cap=[];m.body(xx,cap);zs.append(cap[3].reshape(-1,C.FF))
    z=torch.cat(zs); mu=z.mean(0); sd=z.std(0); p=(z>=0).double().mean(0)
    got=dict(mu_med=float(mu.median()),sd_med=float(sd.median()),mu_absmean=float(mu.abs().mean()),
             neg_frac=float((z<0).double().mean()),dead=float(((z>0).sum(0)==0).double().mean()),
             frac_p_lt005=float((p<.005).double().mean()),p_mean=float(p.mean()))
    ref=next(r for r in csv.DictReader((ELR/f'seed{seed}/layer_stats.csv').open()) if int(r['task'])==task and r['probe']=='chain' and r['layer']=='1')
    per=next(r for r in csv.DictReader((ELR/f'seed{seed}/per_task.csv').open()) if int(r['task'])==task)
    vals=[]
    for key,value in got.items():
        expected=float(ref[key]); tol=(1/512+1e-12) if key in ['dead','frac_p_lt005'] else (1e-4*abs(expected)+1e-6)
        vals.append(dict(metric=key,actual=value,expected=expected,tolerance=tol,passed=abs(value-expected)<=tol))
    for key,value in [('k_med',float(k.median())),('w0_norm_med',float(m.h[0].mlp.up.weight.norm(dim=1).median()))]:
        expected=float(per[key]);tol=1e-4*abs(expected)+1e-6
        vals.append(dict(metric=key,actual=value,expected=expected,tolerance=tol,passed=abs(value-expected)<=tol))
    return dict(seed=seed,task=task,passed=all(v['passed'] for v in vals),checks=vals)


def elr():
    inventory(); inp=Inputs(); rows=[]; checks=[]; inputs=[]; arrays={}
    for seed in range(10):
        for task in [50,100]:
            if not (ELR/f'seed{seed}/state_task{task}.pt').exists(): continue
            m,k,rnp=load_elr(seed,task); check=validate_elr(m,k,seed,task);checks.append(check)
            if not check['passed']:
                dump(OUT/'elr_gate.json',dict(passed=False,checks=checks)); raise RuntimeError('ELR forward restoration mismatch; B stopped')
            row,raw,_,ybar,vl=calculate(m,seed,task,True)
            g=M.P.fixed_error_gradients(m,A.uniform(seed),[vl])[0][1]
            err=M.relmax(raw['load_exact']-(g*ybar).sum(1),raw['load_exact']); assert err<1e-10
            check['autograd_error']=err
            rows.append(dict(arm='ELR',**row))
            arrays.update({f's{seed}_t{task}_{kk}':vv.numpy() for kk,vv in raw.items()})
            for arm in A.ARMS:
                rr,_,_=state(inp,arm,seed,task);rr.pop('previous_max_error')
                rows.append(rr)
            inputs.append(dict(seed=seed,task=task,rownorm_path=str(rnp),rownorm_sha256=M.sha(rnp),
                checkpoint_sha256=M.sha(ELR/f'seed{seed}/state_task{task}.pt')))
        A.CACHE.clear(); A.UCACHE.clear()
    M.write_csv(OUT/'elr_states.csv',[clean(r) for r in rows])
    (OUT/'raw').mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'raw/ELR.npz',**arrays)
    dump(OUT/'elr_gate.json',dict(passed=True,checks=checks,max_autograd_error=max(c['autograd_error'] for c in checks)))
    dump(OUT/'elr_provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256=M.sha(__file__),input_files=inputs,n_ELR_states=len(checks),training_steps=0,optimizer_created=False))
    print(f'ELR {len(checks)} states restored, decomposed, autograd checked',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['gate','run','elr']);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='elr': elr();return
    inp=Inputs()
    if args.mode=='gate':
        assert M.sha(M.BUNDLE)==M.BUNDLE_SHA;run(inp,True)
    else:
        gate=json.loads((OUT/'gate.json').read_text());assert gate['passed'] and gate['code_sha256']==M.sha(__file__)
        run(inp)


if __name__=='__main__': main()

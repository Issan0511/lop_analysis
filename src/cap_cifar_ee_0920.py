#!/usr/bin/env python3
"""S5: rowwise growth caps and post-Adam hidden-bias projection."""
from __future__ import annotations
import argparse,csv,gc,json,os,resource,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cifar_interventions_0920 import *
from analysis.cifar_ledger_0920 import ledger as L
RUN='cap_cifar_ee_0920'
ARMS={'ref':(), 'cap1':(1,), 'cap2':(2,), 'cap12':(1,2), 'cap12_bfix':(1,2)}
COUNTERS=('hits','removed','max_ratio','max_excess','violations','bfix_removed','bfix_hits','bfix_errors')


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)


def norm_bound(radius,d):
    u=2**-24;gam=(2*d+2)*u/(1-(2*d+2)*u)
    # Covers float32 norm, ratio and multiply, plus float64 checker arithmetic.
    check=(2*d+8)*2**-53/(1-(2*d+8)*2**-53)
    return (radius.double()*(1+u)**2/(1-gam)+math_sqrt(d)*torch.finfo(torch.float32).tiny)*(1+4*check)


def math_sqrt(x):return x**.5


def project_rows(w,radius):
    norm=w.norm(dim=2);hit=(norm>radius)|((radius==0)&(w!=0).any(dim=2))
    denom=torch.where(norm>0,norm,torch.ones_like(norm))
    scale=radius/denom
    return torch.where(hit[:,:,None],w*scale[:,:,None],w),hit


class CapEngine(Engine):
    def __init__(self,seeds,X,state=None,arm='ref',anchor=None,graph=True,scale=1.):
        assert arm in ARMS
        self.cap_arm=arm;self.scale=float(scale)
        meta=None if state is None else state.get('cap')
        if meta is not None:
            assert meta['arm']==arm and meta['scale']==scale,'cap configuration'
            self.radii=[q.to(X.device).clone() for q in meta['radii']]
            self.b_fixed=[q.to(X.device).clone() for q in meta['b_fixed']]
            self.anchor_hash=meta['anchor_hash']
        elif anchor is not None:
            self.radii=[anchor['P'][i].to(X.device).norm(dim=2)*scale for i in (0,2)]
            self.b_fixed=[anchor['P'][i].to(X.device).clone() for i in (1,3)]
            self.anchor_hash=tree_hash(core_state(anchor))
        else:
            assert arm=='ref'
            self.radii=[torch.zeros(len(seeds),100,device=X.device) for _ in range(2)]
            self.b_fixed=[torch.zeros_like(q) for q in self.radii];self.anchor_hash=None
        self.counts={k:torch.zeros(len(seeds),2,100,device=X.device,dtype=torch.int64 if k in ('hits','violations','bfix_hits','bfix_errors') else torch.float64) for k in COUNTERS}
        super().__init__(seeds,X,state=state,graph=graph)

    def mutable_tensors(self):return super().mutable_tensors()+list(self.counts.values())

    def state(self):
        st=super().state();st['cap']=cpu(dict(arm=self.cap_arm,scale=self.scale,radii=self.radii,b_fixed=self.b_fixed,anchor_hash=self.anchor_hash,counts=self.counts));return st

    def restore(self,st):
        super().restore(st)
        if 'cap' in st:
            meta=st['cap'];assert meta['arm']==self.cap_arm and meta['anchor_hash']==self.anchor_hash
            for k,q in self.counts.items():q.copy_(meta['counts'][k])
            for dst,src in zip(self.radii,meta['radii']):dst.copy_(src)
            for dst,src in zip(self.b_fixed,meta['b_fixed']):dst.copy_(src)

    def step(self):
        super().step()
        with torch.no_grad():
            for layer in ARMS[self.cap_arm]:
                i=layer-1;w=self.P[2*i];radius=self.radii[i]
                before=w.clone();projected,hit=project_rows(w,radius)
                w.copy_(projected)
                after=w.double().norm(dim=2);removed=(before.double()-w.double()).norm(dim=2)
                bound=norm_bound(radius,w.shape[2])
                ratio=torch.where(radius>0,after/torch.where(radius>0,radius.double(),1),torch.where(after==0,torch.ones_like(after),torch.full_like(after,float('inf'))))
                self.counts['hits'][:,i].add_(hit)
                self.counts['removed'][:,i].add_(removed)
                self.counts['max_ratio'][:,i].copy_(torch.maximum(self.counts['max_ratio'][:,i],ratio))
                self.counts['max_excess'][:,i].copy_(torch.maximum(self.counts['max_excess'][:,i],after-radius.double()))
                self.counts['violations'][:,i].add_(after>bound)
            if self.cap_arm=='cap12_bfix':
                for i in range(2):
                    b=self.P[2*i+1];delta=(b.double()-self.b_fixed[i].double()).abs()
                    self.counts['bfix_removed'][:,i].add_(delta);self.counts['bfix_hits'][:,i].add_(delta>0)
                    b.copy_(self.b_fixed[i]);self.counts['bfix_errors'][:,i].add_(b!=self.b_fixed[i])

    def train_task(self,task,epochs=400,**kwargs):
        if not self.in_task:
            for q in self.counts.values():q.zero_()
        return super().train_task(task,epochs,**kwargs)

    def diagnostic(self):
        rows,unit=super().diagnostic()
        for r,row in enumerate(rows):
            for i in range(2):
                for k,v in self.counts.items():row[f'{k}_l{i+1}']=float(v[r,i].max()) if k.startswith('max_') else float(v[r,i].sum())
        for k,v in self.counts.items():unit['cap_'+k]=v.detach().cpu().numpy()
        for i in range(2):unit[f'radius_l{i+1}']=self.radii[i].cpu().numpy()
        return rows,unit


def ledger_state(engine):
    with torch.no_grad():z1,a1,z2,a2,logits=E.forward(engine.P,engine.X,E.ELU())
    a=a1.detach().cpu().numpy().astype(np.float64);mean=L.mean_ball(a,axis=1)
    w=engine.P[2].detach().cpu().numpy().astype(np.float64);b=engine.P[3].detach().cpu().numpy().astype(np.float64)
    algebra=L.dot(L.Ball(w),L.expand(mean,1))+L.Ball(b)
    direct=L.mean_ball(z2.detach().cpu().numpy().astype(np.float64),axis=1)
    # Native B=1200 dot-product error is kept separate from the endpoint ledger.
    u=2**-24;native=L.gamma(2*w.shape[2]+2,u)*(np.einsum('rhd,rd->rh',abs(w),abs(a).mean(1))+abs(b))
    native+=np.sqrt(w.shape[2])*np.finfo(np.float32).tiny+direct.e+algebra.e
    residual=abs(direct.v-algebra.v);assert np.all(residual<=native),('native mean closure',float((residual-native).max()))
    mu_sd=a.std(1);ratio=np.divide(mean.v,mu_sd,out=np.full_like(mean.v,np.nan),where=mu_sd>0)
    return dict(w=w,b=b,mu=mean.v,mu_error=mean.e,m_alg=algebra.v,m_alg_error=algebra.e,m_direct=direct.v,m_native_bound=native,m_native_residual=residual,mu_sd=mu_sd,mu_mean_over_sd=ratio)


def ledger_interval(old,new):
    out={}
    for s in range(len(old['w'])):
        oo={k:old[k][s] for k in ('w','b','mu','mu_error')};nn={k:new[k][s] for k in oo}
        terms=L.components(oo,nn)
        for name,value in terms.items():
            out.setdefault(name,[]).append(value.v);out.setdefault(name+'_error',[]).append(value.e)
        checks=[terms['closure'],terms['upstream_growth']+terms['upstream_rotation']-terms['upstream'],terms['self_growth']+terms['self_rotation']-terms['self']]
        for q in checks:assert np.all(abs(q.v)<=q.e),('ledger closure',s,float((abs(q.v)-q.e).max()))
    return {k:np.asarray(v) for k,v in out.items()}


def cap_hashes():
    hashes=source_hashes(RUN)
    for f in ('analysis/resp_cifar_ee_0920/stats.py',):hashes[f]=sha(ROOT/f)
    return hashes


def registered_identity(seeds,X,cifar,epochs,tasks):
    d=identity(RUN,seeds,X,cifar,epochs);d['source_sha256']=cap_hashes();d['tasks']=tasks;return d


def verify_anchor(engine,anchor):
    assert engine.anchor_hash==tree_hash(core_state(anchor))
    for i in range(2):
        assert tree_hash(engine.radii[i])==tree_hash(anchor['P'][2*i].to(engine.device).norm(dim=2))
        assert tree_hash(engine.b_fixed[i])==tree_hash(anchor['P'][2*i+1])


def run(out,seeds,epochs=400,tasks=50,check_mode=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dev=setup();X,cifar=make_inputs(seeds,dev);ident=registered_identity(seeds,X,cifar,epochs,tasks)
    if not check_mode:
        assert seeds==list(range(10)) and epochs==400 and tasks==50
        chk=json.loads((ROOT/f'results/_checks_{RUN}/checks.json').read_text());assert chk['all_pass'] and chk['source_sha256']==cap_hashes()
        assert not subprocess.check_output(['git','status','--porcelain','--','src','analysis','specs'],cwd=ROOT,text=True)
        admission=out/'admission_checks.json'
        if admission.exists():assert json.loads(admission.read_text())==chk
        else:put(admission,chk)
    start=out/'provenance_start.json'
    if start.exists():require_identity(json.loads(start.read_text())['identity'],ident)
    else:put(start,dict(identity=ident,pid=os.getpid(),argv=sys.argv,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),started_jst=__import__('datetime').datetime.now(__import__('datetime').timezone(__import__('datetime').timedelta(hours=9))).isoformat(),dirty_diff=subprocess.check_output(['git','diff','HEAD','--','src','analysis','specs'],cwd=ROOT,text=True),independent_audit=False))
    put(out/'input_manifest.json',{k:ident[k] for k in ('data_sha256','input_hash','subset_order')})
    stop=out/'STOP';prefix=out/'prefix';prefix.mkdir(exist_ok=True);tt=time.time()
    if (prefix/'done.json').exists():verify_done(prefix/'done.json',ident)
    else:
        active=prefix/'active.pt'
        if active.exists():
            saved=load_pt(active);require_identity(saved['identity'],ident);e=Engine(seeds,X,saved['state'])
        else:e=Engine(seeds,X)
        if stop.exists():return False
        result=e.train_task(1,epochs,checkpoint=lambda st:save_pt(active,dict(identity=ident,state=st)),stop=stop)
        if result is None:return False
        rows,unit=result;assert all(r['finite'] for r in rows)
        put(prefix/'rows.json',rows);save_npz(prefix/'units.npz',unit);save_pt(prefix/'t01.pt',e.state());save_npz(prefix/'ledger_state.npz',ledger_state(e))
        mark_done(prefix/'done.json',ident,[prefix/f for f in ('rows.json','units.npz','t01.pt','ledger_state.npz')])
        del e;gc.collect();torch.cuda.empty_cache();print('Fresh S5 task 1 saved',flush=True)
    anchor=load_pt(prefix/'t01.pt');initial=json.loads((prefix/'rows.json').read_text());history_hashes={}
    with np.load(prefix/'ledger_state.npz') as f:ledger_first={k:f[k] for k in f.files}
    for name in ARMS:
        arm=out/'arms'/name;arm.mkdir(parents=True,exist_ok=True)
        if (arm/'done.json').exists():verify_done(arm/'done.json',ident);continue
        active=arm/'active.pt'
        if active.exists():
            saved=load_pt(active);require_identity(saved['identity'],ident);e=CapEngine(seeds,X,saved['state'],name);rows=saved['rows'];old=saved['ledger']
        else:
            e=CapEngine(seeds,X,anchor,name,anchor=anchor);rows=[{**r,'arm':name} for r in initial];old=ledger_first
            save_npz(arm/'units_t01.npz',{k:v for k,v in np.load(prefix/'units.npz').items()});save_npz(arm/'ledger_state_t01.npz',old)
            save_pt(arm/'cap_anchor.pt',e.state()['cap']);put(arm/'branch.json',dict(core_sha256=tree_hash(core_state(e.state())),anchor_sha256=tree_hash(core_state(anchor))))
        verify_anchor(e,anchor)
        first=e.task if e.in_task else e.task+1
        for task in range(first,tasks+1):
            if stop.exists():return False
            result=e.train_task(task,epochs,checkpoint=lambda st:save_pt(active,dict(identity=ident,state=st,rows=rows,ledger=old)),stop=stop)
            if result is None:return False
            rr,unit=result
            assert all(r['finite'] for r in rr),'DIVERGED'
            assert not bool(e.counts['violations'].any()) and not bool(e.counts['bfix_errors'].any()),'projection check failed'
            for r in rr:r['arm']=name
            rows.extend(rr);new=ledger_state(e);parts=ledger_interval(old,new)
            put(arm/'rows.json',rows);save_npz(arm/f'units_t{task:02d}.npz',unit);save_npz(arm/f'ledger_state_t{task:02d}.npz',new);save_npz(arm/f'ledger_t{task:02d}.npz',parts)
            if task in (10,30,50):save_pt(arm/f't{task:02d}.pt',e.state())
            old=new;save_pt(active,dict(identity=ident,state=e.state(),rows=rows,ledger=old))
            put(out/'status.json',dict(stage='running',arm=name,completed_task=task,wall_s=time.time()-tt))
            message=f'{name} task {task}/{tasks} saved; elapsed {time.time()-tt:.1f}s'
            print(message,flush=True)
            (out/'logs').mkdir(exist_ok=True)
            with (out/'logs'/f'{RUN}_{name}.log').open('a') as log:log.write(message+'\n')
        mark_done(arm/'done.json',ident,sorted(p for p in arm.iterdir() if p.name not in ('done.json','active.pt')))
        del e;gc.collect();torch.cuda.empty_cache()
    # Check label and order pairing in explicit manifests after every arm completes.
    ref=json.loads((out/'arms/ref/rows.json').read_text());lookup={(r['seed'],r['task']):r for r in ref}
    assert len(ref)==len(seeds)*tasks
    for name in ARMS:
        rows=json.loads((out/'arms'/name/'rows.json').read_text());assert len(rows)==len(seeds)*tasks
        assert {(r['seed'],r['task']) for r in rows}==set(lookup)
        for r in rows:
            for k in ('label_hash','order_hash','major_frac'):assert r[k]==lookup[r['seed'],r['task']][k]
    put(out/'pairing_checks.json',dict(all_pass=True,arms=list(ARMS),tasks=tasks,seeds=seeds))
    put(out/'status.json',dict(stage='completed',arms=5,seeds=len(seeds),tasks=tasks,wall_s=time.time()-tt))
    put(out/'provenance_end.json',dict(identity=ident,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed_this_invocation_s=time.time()-tt,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'results'/RUN));p.add_argument('--seeds',default='0-9');p.add_argument('--epochs',type=int,default=400);p.add_argument('--tasks',type=int,default=50);p.add_argument('--check-mode',action='store_true')
    a=p.parse_args()
    if a.check_mode:assert Path(a.out).resolve()!=ROOT/'results'/RUN
    with exclusive():success=run(Path(a.out),E.parse_ints(a.seeds),a.epochs,a.tasks,a.check_mode)
    if not success:print('STOP acknowledged; checkpoint saved.',flush=True)


if __name__=='__main__':main()

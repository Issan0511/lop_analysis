"""Direct readout parameter x head-Adam factorial; preregistered p_residual_loading_1111."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
import torch
from analysis.loading_bridge_1111 import dynamic as D
from src import stage1_ctrl_1111 as M

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT/'specs/spec_p_residual_loading_1111.md'
ARMS = ('KK','KC','ZK','ZC')

def snapshot(box):
    if isinstance(box,D.MLP):return M.snapshot(box.P,box.opt)
    from analysis.joudaki_vit_battle_0919.run import get_rng
    return copy.deepcopy((box.model.state_dict(),box.opt.state_dict(),get_rng()))

def restore(box,sp):
    if isinstance(box,D.MLP):box.P,box.opt=M.restore(sp,'adam',.001)
    else:
        from analysis.joudaki_vit_battle_0919.run import set_rng
        box.model.load_state_dict(sp[0]);box.opt=box.C.build_optimizer(box.model)
        box.opt.load_state_dict(copy.deepcopy(sp[1]));set_rng(sp[2]);box.model.train()

def digest(items):
    h=hashlib.sha256()
    def feed(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous().numpy();h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x):h.update(str(k).encode());feed(x[k])
        elif isinstance(x,(tuple,list)):
            for y in x:feed(y)
        else:h.update(repr(x).encode())
    feed(items);return h.hexdigest()

def fingerprints(box):
    body=list(box.body());head=list(box.head()[:2])
    return dict(body=digest(body),body_adam=digest([box.opt.state.get(p,{}) for p in body]),
                head=digest(head),head_adam=digest([box.opt.state.get(p,{}) for p in head]))

@torch.no_grad()
def factorial(box,arm):
    assert arm in ARMS
    W,b,_=box.head()
    if arm[0]=='Z':W.zero_();b.zero_()
    if arm[1]=='C':box.opt.state.pop(W,None);box.opt.state.pop(b,None)

def permute(box,t,history):
    before=fingerprints(box);old=box.head()[0].detach().clone()
    if isinstance(box,D.MLP):
        label=f'persist|P|adam|{t+1}' if history=='Pall' else f'arm|P|adam|{t+1}'
        M.intervene('P',box.P,box.opt,box.ctx,M.own(label,box.seed),M.initial(box.seed,(784,100,10))[2],'adam')
    else:D.intervene('P',box,box.seed,t)
    after=fingerprints(box);assert before['body']==after['body'] and before['body_adam']==after['body_adam']
    W,_,rows=box.head();assert torch.equal(old[rows].sort(0).values,W[rows].detach().sort(0).values)

@torch.no_grad()
def measure(box,step,units):
    out=box.measure('S',units=units)
    if isinstance(box,D.MLP):out['acc'],out['ce']=M.evaluate(box.P,box.ctx,box.new)
    return dict(step=int(step),**out)

def curve_run(box,tau,limit,capture25=False):
    points=sorted({tau,1,25,26,limit,*range(0,limit+1,box.eval_every)} & set(range(tau,limit+1)))
    rec=[measure(box,tau,True)];cap=None
    for step,(x,y) in enumerate(box.batches(limit,tau),tau+1):
        loss=D.loss_for(box,box.forward(x),y);box.opt.zero_grad(set_to_none=True);loss.backward();box.opt.step()
        if step in points:rec.append(measure(box,step,step in (25,200,limit)))
        if step==25 and capture25:cap=snapshot(box)
    assert [r['step'] for r in rec]==points
    return rec,cap

def name(model,act,history,seed,t,source,tau,arm):
    return f'{model}_{act}_{history}_s{seed}_t{t}_{source}_tau{tau}_{arm}.json'

def source_files(box,args):
    sources=list(getattr(box,'sources',[getattr(box,'source',None)]))
    if isinstance(box,D.MLP):
        sources += [D.RAW_M/f'branch_adam_lr0.001_{box.act}_s{box.seed}.npz'] if args.history=='S' else []
    return [p for p in sources if p is not None]

def provenance(box,args):
    helpers=[Path(D.__file__),Path(M.__file__),ROOT/'analysis/loading_bridge_1111/static_probe.py']
    if args.history=='Pall':helpers.append(ROOT/'analysis/p_residual_loading_1111/pall.py')
    return dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                script_sha256=D.sha(__file__),spec_sha256=D.sha(SPEC),helpers={str(p.relative_to(ROOT)):D.sha(p) for p in helpers},
                sources={str(p):D.sha(p) for p in source_files(box,args)},torch=torch.__version__,numpy=np.__version__,
                args=vars(args),replay_checks=getattr(box,'replay_checks',[]),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))

def existing_check(box,t,source,arm,curve,history):
    if not isinstance(box,D.MLP) or arm not in ('KK','ZC'):return None
    # H0 is the canonical head+head-memory reset, not the former all-Adam reset.
    if history=='Pall':
        if arm!='KK':return None
        path=box.source;key=f'Pall|t{t+1:02d}|eval'
    else:
        path=D.RAW_M/f'branch_adam_lr0.001_{box.act}_s{box.seed}.npz'
        key=f'{source if arm=="KK" else "H0"}|t{t+1:02d}|eval'
    old=np.load(path)[key]
    rows=[r for r in curve if r['step']%100==0]
    current=np.array([[r['acc'],r['ce']] for r in rows],np.float32)
    match=np.array_equal(current,old[:len(current)])
    assert match,(str(path),key,float(np.max(np.abs(current-old[:len(current)]))))
    return dict(path=str(path),key=key,full_eval_exact=True,points=len(current))

def run_cell(box,t,args,prov):
    act=getattr(box,'act','SL');outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    limit=args.limit or box.recovery_limit
    # Fully reconstruct each cell on resume if any logical output is absent. Existing
    # complete outputs are checked for compatible implementation before replacement.
    sources=('P',) if args.history=='Pall' else ('S','P')
    keys=[name(args.model,act,args.history,box.seed,t,s,0,a) for s in sources for a in ARMS]
    if args.history=='S':keys += [name(args.model,act,args.history,box.seed,t,'P',25,a) for a in ARMS]
    present=0
    for key in keys:
        p=outdir/key
        if p.exists():
            d=json.loads(p.read_text());q=d['provenance']
            assert d['complete'] and q['script_sha256']==prov['script_sha256'] and q['spec_sha256']==prov['spec_sha256'] and q['helpers']==prov['helpers']
            assert q['sources']==prov['sources'], 'source archive changed'
            assert d['limit']==limit and q['args']['engine']==args.engine
            present+=1
    if present==len(keys):print('RESUME complete',keys[0],flush=True);return
    shared=None;shared_post=None;captured=None;pk=None
    for source in sources:
        box.reset(t)
        if source=='P':permute(box,t,args.history)
        sp=snapshot(box);fork=fingerprints(box)
        for arm in ARMS:
            restore(box,sp);assert fingerprints(box)==fork
            factorial(box,arm);post=fingerprints(box)
            assert all(post[k]==fork[k] for k in ('body','body_adam'))
            started=time.monotonic();key=name(args.model,act,args.history,box.seed,t,source,0,arm)
            d=dict(model=args.model,activation=act,seed=box.seed,t_end=t,history=args.history,source=source,tau=0,arm=arm,limit=limit,
                   fork_hashes=fork,post_hashes=post,provenance=prov,reused=False)
            if source=='P' and arm=='ZC' and shared is not None:
                assert post==shared_post
                d.update(curve=copy.deepcopy(shared['curve']),reused=True,reuse_of=shared['file'],identity='S/P zero head+head Adam, body unchanged')
            else:
                curve,cap=curve_run(box,0,limit,capture25=(source=='P' and arm=='KK' and args.history=='S'))
                d['curve']=curve
                if cap is not None:captured=cap
            d['legacy_eval_check']=existing_check(box,t,source,arm,d['curve'],args.history)
            d.update(complete=True,elapsed_s=time.monotonic()-started,file=key)
            D.save(outdir/key,d)
            print(f'DONE {key} acc={d["curve"][-1]["acc"]:.4f} {d["elapsed_s"]:.1f}s reused={d["reused"]}',flush=True)
            if source=='S' and arm=='ZC':shared=d;shared_post=post
            if source=='P' and arm=='KK':pk=d
    if args.history=='S':
        assert captured is not None and limit>=26
        restore(box,captured);fork=fingerprints(box)
        for arm in ARMS:
            restore(box,captured);assert fingerprints(box)==fork
            factorial(box,arm);post=fingerprints(box)
            assert all(post[k]==fork[k] for k in ('body','body_adam'))
            started=time.monotonic();key=name(args.model,act,args.history,box.seed,t,'P',25,arm)
            d=dict(model=args.model,activation=act,seed=box.seed,t_end=t,history=args.history,source='P',tau=25,arm=arm,limit=limit,
                   fork_hashes=fork,post_hashes=post,provenance=prov,reused=arm=='KK')
            if arm=='KK':
                d.update(curve=[copy.deepcopy(r) for r in pk['curve'] if r['step']>=25],reuse_of=pk['file'],identity='unchanged KK continuation suffix')
                assert d['curve'][0]==measure(box,25,True)
            else:d['curve'],_=curve_run(box,25,limit)
            d.update(complete=True,elapsed_s=time.monotonic()-started,file=key)
            D.save(outdir/key,d)
            print(f'DONE {key} acc={d["curve"][-1]["acc"]:.4f} {d["elapsed_s"]:.1f}s reused={d["reused"]}',flush=True)

def synthetic():
    class Toy:
        def __init__(self):
            self.ps=[torch.randn(2,3,requires_grad=True),torch.randn(2,requires_grad=True),torch.randn(3,2,requires_grad=True),torch.randn(3,requires_grad=True)]
            self.opt=torch.optim.Adam(self.ps)
            sum(p.square().sum() for p in self.ps).backward();self.opt.step()
        def body(self):return self.ps[:2]
        def head(self):return self.ps[2],self.ps[3],torch.arange(3)
    toy=Toy();start=copy.deepcopy(toy);f=fingerprints(toy)
    for a in ARMS:
        toy=copy.deepcopy(start);factorial(toy,a);g=fingerprints(toy)
        assert g['body']==f['body'] and g['body_adam']==f['body_adam']
        assert (g['head']==f['head'])==(a[0]=='K')
        assert (g['head_adam']==f['head_adam'])==(a[1]=='K')
        if a[0]=='Z':assert all(torch.count_nonzero(p)==0 for p in toy.head()[:2])
    p=copy.deepcopy(start)
    with torch.no_grad():
        W,b,_=p.head();idx=torch.tensor([[1,2],[2,0],[0,1]]);W.copy_(W.gather(0,idx));b.add_(1.)
        for k in ('exp_avg','exp_avg_sq'):p.opt.state[W][k]=p.opt.state[W][k].gather(0,idx)
    factorial(p,'ZC');factorial(start,'ZC');assert fingerprints(p)==fingerprints(start)
    print('PASS synthetic factorial, body invariance, S/P ZC identity')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['mlp','vit'],default='mlp');ap.add_argument('--history',choices=['S','Pall'],default='S')
    ap.add_argument('--act',default='ELU');ap.add_argument('--seeds',default='0');ap.add_argument('--tasks',default='1,10')
    ap.add_argument('--limit',type=int,default=0);ap.add_argument('--engine',choices=['compile','eager'],default='compile')
    ap.add_argument('--out',required=True);ap.add_argument('--synthetic',action='store_true');args=ap.parse_args()
    if args.synthetic:synthetic();return
    assert args.limit==0 or args.limit>=26
    for seed in map(int,args.seeds.split(',')):
        start=time.monotonic()
        if args.model=='mlp':
            tasks=list(map(int,args.tasks.split(',')))
            if args.history=='Pall':
                from analysis.p_residual_loading_1111.pall import Pall
                box=Pall(seed,args.act,tasks)
            else:box=D.MLP(seed,args.act,tasks)
        else:
            assert args.history=='S';tasks=[100];box=D.ViT(seed,args)
        prov=provenance(box,args);prov['setup_elapsed_s']=time.monotonic()-start
        print('SETUP',args.model,args.act,args.history,seed,round(prov['setup_elapsed_s'],1),flush=True)
        for t in tasks:run_cell(box,t,args,prov)
        del box
        if args.model=='vit':torch.cuda.empty_cache()

if __name__=='__main__':main()

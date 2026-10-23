"""Preregistered prospective freezing experiment; spec_elu_reserve_0913.
Only this module is new. Imported frozen training modules remain unchanged.
"""
from pathlib import Path
import argparse, concurrent.futures, csv, hashlib, itertools, json, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from src import width_sink_clamp_0909 as C
from src import transport_common_0910 as T
from src import gate_shape_0911 as GS
from src import unit_triage_0911 as U
H=C.H
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/elu_reserve_0913'
SPEC=ROOT/'specs/spec_elu_reserve_0913.md'
PREREG='b5a64757db2a790502b1947c76c4a304e28d1d8e'
STEPS=(0,20,100,300,625)
BRANCHES=('NONE','TARGET','MATCHED','RANDOM','SHALLOW','SHALLOW_MATCHED')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def thash(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def ranks(x):
    _,inv,count=np.unique(np.asarray(x),return_inverse=True,return_counts=True)
    starts=np.r_[0,np.cumsum(count)[:-1]]
    return ((starts+(count-1)/2)/(len(x)-1))[inv]

def assignment(cost):
    """Rectangular Hungarian minimum-cost assignment, deterministic column order."""
    a=np.asarray(cost,float); n,m=a.shape
    assert n<=m
    if n==0:return np.array([],dtype=int)
    u=np.zeros(n+1);v=np.zeros(m+1);p=np.zeros(m+1,dtype=int);way=np.zeros(m+1,dtype=int)
    for i in range(1,n+1):
        p[0]=i;j0=0;minv=np.full(m+1,np.inf);used=np.zeros(m+1,dtype=bool)
        while True:
            used[j0]=True;i0=p[j0];delta=np.inf;j1=0
            for j in range(1,m+1):
                if not used[j]:
                    cur=a[i0-1,j-1]-u[i0]-v[j]
                    if cur<minv[j]:minv[j]=cur;way[j]=j0
                    if minv[j]<delta:delta=minv[j];j1=j
            for j in range(m+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minv[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    out=np.empty(n,dtype=int)
    for j in range(1,m+1):
        if p[j]:out[p[j]-1]=j-1
    return out

def match(ids,features):
    other=np.setdiff1d(np.arange(100),ids)
    cost=((features[ids,None,:]-features[None,other,:])**2).sum(-1)
    chosen=other[assignment(cost)]
    balance=np.abs(features[ids]-features[chosen])
    return chosen,dict(total_cost=float((balance**2).sum()),
                       mean_abs_percentile_difference=balance.mean(0).tolist() if len(ids) else [0.,0.],
                       max_abs_percentile_difference=balance.max(0).tolist() if len(ids) else [0.,0.],
                       pairs=list(zip(ids.tolist(),chosen.tolist())))

def initial(seed,arm):
    p=H.init_params(seed,torch.device('cpu'));act=C.make_act(arm)
    adam=([torch.zeros_like(q) for q in p],[torch.zeros_like(q) for q in p],[0])
    gens=[H.stream('perm',seed),H.stream('data',seed),H.stream('batch',seed)]
    return p,act,adam,gens

def data(seed):
    T.data_dir();mnist=H.Mnist(torch.device('cpu'))
    ix=torch.randperm(len(mnist.test_x),generator=H.stream('boundary_probe',seed))
    sel,ev,held=ix[:512],ix[512:1024],ix[512:]
    assert len(set(sel.tolist())&set(ev.tolist()))==0
    probe=C.Probe(mnist.test_x[sel],mnist.test_y[sel],float(mnist.train_x.mean()))
    return mnist,probe,ev,held,sel

def adam_step(p,gr,adam,mask=None):
    m,v,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
    with torch.no_grad():
        frozen=[p[0][mask].clone(),p[1][mask].clone()] if mask is not None and len(mask) else None
        for q,g,mi,vi in zip(p,gr,m,v):
            mi.mul_(.9).add_(g,alpha=1-.9)
            vi.mul_(.999).addcmul_(g,g,value=1-.999)
            q-=.001*(mi/c1)/((vi/c2).sqrt()+1e-8)
        if frozen is not None:
            p[0][mask]=frozen[0];p[1][mask]=frozen[1]

def evaluate(p,act,x,y):
    with torch.no_grad():
        logits=H.forward(p,x,act)[4]
        return float(F.cross_entropy(logits,y)),float((logits.argmax(1)==y).float().mean())

def task(p,act,adam,gens,mnist,taskid,mask=None,evalix=None,heldix=None,check=None):
    perm=torch.randperm(784,generator=gens[0])
    ix=H.stratified_draw(mnist,gens[1]);order=torch.randperm(H.TASK_EXAMPLES,generator=gens[2])
    xs=mnist.train_x[ix][:,perm][order];ys=mnist.train_y[ix][order]
    hashes=dict(perm=thash(perm),data=thash(ix),order=thash(order))
    curve=[]
    if evalix is not None:
        xe=mnist.test_x[evalix][:,perm];ye=mnist.test_y[evalix]
        ce,acc=evaluate(p,act,xe,ye);curve.append(dict(task=taskid,step=0,ce=ce,acc=acc))
    frozen=[p[0][mask].detach().clone(),p[1][mask].detach().clone()] if mask is not None and len(mask) else None
    for step in range(1,626):
        out=H.forward(p,xs[(step-1)*16:step*16],act)
        gr=torch.autograd.grad(F.cross_entropy(out[4],ys[(step-1)*16:step*16]),p)
        adam_step(p,gr,adam,mask)
        if frozen is not None:
            with torch.no_grad():
                assert torch.equal(p[0][mask],frozen[0]) and torch.equal(p[1][mask],frozen[1]),'frozen parameter drift'
        if evalix is not None and step in STEPS:
            ce,acc=evaluate(p,act,xe,ye);curve.append(dict(task=taskid,step=step,ce=ce,acc=acc))
    if heldix is not None:
        ce,acc=evaluate(p,act,mnist.test_x[heldix][:,perm],mnist.test_y[heldix])
        curve[-1].update(heldout_ce=ce,heldout_acc=acc)
    return perm,curve,hashes

def checkpoint_diagnostics(p,act,probe,perm,seed,t):
    with torch.no_grad():
        zs=torch.stack([probe.px[:,rp]@p[0].detach().T+p[1].detach() for rp in C.refperms()])
        occ={str(e):(zs.abs()<=e).double().mean((0,1)).numpy() for e in (.5,1.,2.)}
        W,m,Wt=C.rows_of(p[0]);cn=Wt.norm(dim=1).numpy()
        z=probe.px[:,perm]@p[0].detach().T+p[1].detach()
        w2=p[2].detach().double().norm(dim=0).numpy()
        abl=U.ablation(p,act,probe,perm)
        zero=U.ablation(p,act,probe,perm,zero_col=0)
        assert zero['dce'][0]==0.,'zero W2 ablation must be exact0'
        small=np.argsort(cn,kind='stable')[:25]
        shallow=np.argsort(-occ['1.0'],kind='stable')[:25]
        target=np.intersect1d(small,shallow)
        feat=np.stack([ranks(np.abs(abl['dce'])),ranks(w2)],axis=1)
        matched,mbal=match(target,feat);shmatched,sbal=match(shallow,feat)
        rng=np.random.default_rng(20260913+1000*seed+t)
        random=rng.choice(np.setdiff1d(np.arange(100),target),size=len(target),replace=False)
        masks={'NONE':np.array([],dtype=int),'TARGET':target,'MATCHED':matched,'RANDOM':random,
               'SHALLOW':shallow,'SHALLOW_MATCHED':shmatched}
        assert np.array_equal(np.intersect1d(small,shallow),target)
        for name in ('MATCHED','RANDOM'):
            assert len(masks[name])==len(target) and not len(np.intersect1d(target,masks[name]))
        assert len(shmatched)==25 and not len(np.intersect1d(shallow,shmatched))
        units=[]
        for i in range(100):
            r=dict(unit=i,cnorm=float(cn[i]),raw_norm=float(W[i].norm()),row_mean=float(m[i]),
                   bias=float(p[1][i]),current_z_mean=float(z[:,i].double().mean()),
                   current_z_sd=float(z[:,i].double().std(unbiased=False)),
                   ref_z_mean=float(zs[:,:,i].double().mean()),
                   ref_z_sd=float(zs[:,:,i].double().std(unbiased=False)),
                   h05=float(occ['0.5'][i]),h1=float(occ['1.0'][i]),h2=float(occ['2.0'][i]),
                   dce=float(abl['dce'][i]),w2col=float(w2[i]),
                   dce_rank=float(feat[i,0]),w2_rank=float(feat[i,1]))
            r.update({k.lower():int(i in ids) for k,ids in masks.items() if k!='NONE'});units.append(r)
        diag=dict(seed=seed,checkpoint=t,n_target=len(target),candidate_status='EXISTS' if len(target)>=3 else 'COHORT_TOO_SMALL',
                  masks={k:v.tolist() for k,v in masks.items()},target_matching=mbal,shallow_matching=sbal,
                  mean_h1_shallow=float(occ['1.0'][shallow].mean()),
                  mean_h1_rest=float(np.delete(occ['1.0'],shallow).mean()),
                  selection='top25 P(abs(z)<=1) over8 fixed permutations intersect bottom25 centered norms')
    return diag,units,masks

def check_anchor(p,act,probe,perm,arm,seed,t):
    ref=np.load(ROOT/'results/unit_triage_0911'/f'{arm}_none_s{seed}_units.npz')
    _,u=C.measure(p,act,probe,perm,False,None,{})
    _,gu,_=GS.gate_block(p,act,probe,perm)
    fields=('cnorm_i','m_i','bias_i','zcur_i','sdcur_i')
    merged={**u,**gu};checks={}
    for k in fields:
        expected=ref[f'ref_{k}_t{t}'];got=merged[k]
        checks[k]=float(np.max(np.abs(got-expected)))
        assert np.array_equal(got,expected),(arm,seed,t,k,checks[k])
    return checks

def smoke():
    torch.set_num_threads(1);H.setup('cpu')
    rng=np.random.default_rng(23)
    for n,m in ((1,3),(2,3),(3,4)):
        for rep in range(10):
            cost=rng.uniform(size=(n,m));chosen=assignment(cost)
            exact=min(sum(cost[i,j] for i,j in enumerate(js)) for js in itertools.permutations(range(m),n))
            assert abs(cost[np.arange(n),chosen].sum()-exact)<1e-12
    assert np.allclose(ranks([0,0,1,2]),[1/6,1/6,2/3,1])
    mnist,probe,ev,held,sel=data(0)
    p,act,adam,gens=initial(0,'ELU1');snap=C.snapshot(p,act,adam,gens)
    p1,a1,ad1,g1=C.restore(snap,'ELU1');C.loop(p1,a1,ad1,g1,mnist,probe,1,1,'ref',None,None,None,{})
    p2,a2,ad2,g2=C.restore(snap,'ELU1');perm,_,_=task(p2,a2,ad2,g2,mnist,1,torch.tensor([],dtype=torch.long))
    assert C.exact(C.snapshot(p1,a1,ad1,g1),C.snapshot(p2,a2,ad2,g2)),'original loop mismatch'
    frozen=torch.tensor([0,2,5]);pp,aa,ad,gg=C.restore(C.snapshot(p2,a2,ad2,g2),'ELU1')
    original=[q[frozen].detach().clone() for q in pp[:2]]
    xp=mnist.train_x[:16];yp=mnist.train_y[:16]
    grads=torch.autograd.grad(F.cross_entropy(H.forward(pp,xp,aa)[4],yp),pp)
    adam_step(pp,grads,ad,frozen)
    assert all(torch.equal(q[frozen],old) for q,old in zip(pp[:2],original))
    pm,am,adm,gm=C.restore(C.snapshot(p2,a2,ad2,g2),'ELU1')
    gmuts=list(torch.autograd.grad(F.cross_entropy(H.forward(pm,xp,am)[4],yp),pm))
    for g in gmuts[:2]:g[frozen]=0
    before=pm[0][frozen].detach().clone();adam_step(pm,gmuts,adm)
    mutation=float((pm[0][frozen].detach()-before).abs().max())
    assert mutation>0,'gradient-only mutation did not drift'
    diag,units,masks=checkpoint_diagnostics(p2,a2,probe,perm,0,1)
    report=dict(status='PASS',hungarian_bruteforce_cases=30,original_loop_exact=True,
                empty_mask_exact=True,actual_mask_exact=True,gradient_only_mutation_drift=mutation,
                zero_column_ablation_exact=True,data_sha=T.DATA_SHA,torch=torch.__version__,
                smoke_scope='one original task only; registered t20/t100 endpoints not generated')
    OUT.mkdir(parents=True,exist_ok=True);write_json(OUT/'smoke.json',report)
    print(json.dumps(report),flush=True)

def run(arm,seed):
    torch.set_num_threads(1);H.setup('cpu');start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    assert subprocess.check_output(['git','merge-base','--is-ancestor',PREREG,'HEAD'],cwd=ROOT).decode()==''
    launch_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    mnist,probe,ev,held,sel=data(seed)
    p,act,adam,gens=initial(seed,arm);snaps={};anchor={}
    for t in range(1,101):
        perm,_,_=task(p,act,adam,gens,mnist,t)
        if t in (20,100):
            anchor[str(t)]=check_anchor(p,act,probe,perm,arm,seed,t)
            snaps[t]=(C.snapshot(p,act,adam,gens),perm.clone())
        if t%20==0:print('PREFIX',arm,seed,t,round(time.monotonic()-start,1),flush=True)
    allrows=[];summaries=[];allhashes={}
    for checkpoint,(snap,lastperm) in snaps.items():
        prefix=f'{arm}_s{seed}_t{checkpoint}'
        torch.save(dict(snapshot=snap,perm=lastperm),OUT/f'{prefix}_checkpoint.pt')
        pp,aa,ad,gg=C.restore(snap,arm)
        diag,units,masks=checkpoint_diagnostics(pp,aa,probe,lastperm,seed,checkpoint)
        write_json(OUT/f'{prefix}_cohorts.json',diag);write_csv(OUT/f'{prefix}_units.csv',units)
        starting_logits=H.forward(pp,mnist.test_x[ev][:,lastperm],aa)[4].detach()
        branch_hashes={};empty_outcome=None
        for name in BRANCHES:
            pp,aa,ad,gg=C.restore(snap,arm)
            assert C.exact(C.snapshot(pp,aa,ad,gg),snap),'branch initial state mismatch'
            assert torch.equal(H.forward(pp,mnist.test_x[ev][:,lastperm],aa)[4].detach(),starting_logits),'step0 logits mismatch'
            mask=torch.tensor(masks[name],dtype=torch.long)
            outcomes=[];hashes=[];initial_frozen=[q[mask].detach().clone() for q in pp[:2]]
            for t in range(checkpoint+1,checkpoint+11):
                _,curve,hsh=task(pp,aa,ad,gg,mnist,t,mask,ev,held)
                hashes.append(hsh)
                for r in curve:r.update(arm=arm,seed=seed,checkpoint=checkpoint,branch=name,n_frozen=len(mask))
                outcomes.extend(curve)
            assert all(torch.equal(q[mask],old) for q,old in zip(pp[:2],initial_frozen)),'longitudinal frozen drift'
            branch_hashes[name]=hashes
            if not len(mask):
                if empty_outcome is None:empty_outcome=outcomes
                else:
                    a=[(r['ce'],r['acc']) for r in outcomes];b=[(r['ce'],r['acc']) for r in empty_outcome]
                    assert a==b,'empty branch outcome differs'
            areas=[]
            for t in range(checkpoint+1,checkpoint+11):
                rr=[r for r in outcomes if r['task']==t]
                areas.append(float(np.trapezoid([r['ce'] for r in rr],[r['step'] for r in rr])/625))
            end=[r for r in outcomes if r['step']==625]
            summ=dict(arm=arm,seed=seed,checkpoint=checkpoint,branch=name,n_frozen=len(mask),
                      cohort_status=diag['candidate_status'],auc_ce=float(np.mean(areas)),
                      task1_auc_ce=areas[0],tasks2to10_auc_ce=float(np.mean(areas[1:])),
                      heldout_ce=float(np.mean([r['heldout_ce'] for r in end])),
                      heldout_acc=float(np.mean([r['heldout_acc'] for r in end])))
            assert all(np.isfinite(r['ce']) for r in outcomes)
            summaries.append(summ);allrows.extend(outcomes)
            write_csv(OUT/f'{prefix}_{name}_curve.csv',outcomes)
            print('BRANCH',prefix,name,len(mask),round(time.monotonic()-start,1),flush=True)
        assert all(x==branch_hashes['NONE'] for x in branch_hashes.values()),'task RNG mismatch'
        allhashes[str(checkpoint)]=branch_hashes['NONE']
    write_csv(OUT/f'{arm}_s{seed}_summary.csv',summaries)
    prov=dict(prereg_commit=PREREG,implementation_commit=launch_head,
              spec_sha=sha(SPEC),code_sha=sha(__file__),data_sha=T.DATA_SHA,torch=torch.__version__,
              python=sys.version,selection_indices=sel.tolist(),evaluation_indices=ev.tolist(),
              module_sha={str(Path(m.__file__).relative_to(ROOT)):sha(m.__file__) for m in (C,T,GS,U,H)},
              exact_anchor_maxabs=anchor,task_hashes=allhashes,worker_threads=torch.get_num_threads(),
              scope='CPU; original t20/t100 checkpoints; six same-state branches; fixed625 updates;10 fresh tasks',
              wall_seconds=time.monotonic()-start)
    write_json(OUT/f'{arm}_s{seed}_provenance.json',prov)
    print('DONE',arm,seed,round(time.monotonic()-start,1),flush=True)

def report():
    rows=[]
    for arm in ('ELU1','LR'):
        for seed in (0,1,2):
            path=OUT/f'{arm}_s{seed}_summary.csv';assert path.exists(),path
            with open(path) as f:rows.extend(csv.DictReader(f))
    contrasts=[]
    for arm in ('ELU1','LR'):
        for seed in (0,1,2):
            for checkpoint in (20,100):
                d={r['branch']:r for r in rows if r['arm']==arm and int(r['seed'])==seed and int(r['checkpoint'])==checkpoint}
                ce=lambda k:float(d[k]['auc_ce'])
                contrasts.append(dict(arm=arm,seed=seed,checkpoint=checkpoint,n_target=int(d['TARGET']['n_frozen']),
                    cohort_status=d['TARGET']['cohort_status'],none_auc=ce('NONE'),
                    target_harm=ce('TARGET')-ce('NONE'),matched_harm=ce('MATCHED')-ce('NONE'),
                    target_minus_matched=ce('TARGET')-ce('MATCHED'),target_minus_random=ce('TARGET')-ce('RANDOM'),
                    shallow_harm=ce('SHALLOW')-ce('NONE'),shallow_minus_matched=ce('SHALLOW')-ce('SHALLOW_MATCHED')))
    primary=[r for r in contrasts if r['arm']=='ELU1' and r['checkpoint']==100]
    if any(r['n_target']<3 for r in primary):verdict='NOT_IDENTIFIABLE'
    elif all(r['target_minus_matched']>1e-5 for r in primary):verdict='DIRECTIONAL_PILOT_SUPPORT'
    elif all(r['target_minus_matched']<=1e-5 for r in primary):verdict='NO_SELECTIVE_RESERVE_EVIDENCE'
    else:verdict='MIXED'
    write_csv(OUT/'all_seed_summary.csv',rows);write_csv(OUT/'contrasts.csv',contrasts)
    write_csv(OUT/'verdict.csv',[dict(primary_label=verdict,**r) for r in contrasts])
    write_json(OUT/'verdict.json',dict(primary=verdict,primary_scope='ELU1 task100;3seed directional pilot, not confirmatory inference',rows=contrasts))
    lines=['# ELU reserve freezing 0913 results','',f'Primary preregistered pilot verdict: **{verdict}**.',
           'Positive excess harm means freezing the designated group impaired prospective learning more than its current-contribution/W2-norm matched control.',
           'The primary candidate is the intersection of top25 actual near-zero occupancy and bottom25 centered pattern norm. Broad shallow-only results remain separate.',
           '', '| Activation | Checkpoint | Seed | n target | Target−matched AUC CE | Shallow−matched AUC CE |',
           '|---|---:|---:|---:|---:|---:|']
    for r in contrasts:lines.append(f"| {r['arm']} | {r['checkpoint']} | {r['seed']} | {r['n_target']} | {r['target_minus_matched']:+.6f} | {r['shallow_minus_matched']:+.6f} |")
    lines+=['','AUC is measured on512 test images disjoint from the512 used for cohort selection and current contribution matching. Final metrics use the9488 non-selection test images.',
            'All freeze interventions begin with identical functions/Adam/task RNG states. Frozen IDs never change; Adam moments continue but actual W1-row and bias updates are blocked.',
            'Limitations: small3-seed directional pilot; finite10-task horizon; relative-quantile cohort; matching quality is reported in cohort JSON, not assumed; cohort sizes may differ across activations; null effects do not imply saturation is harmless.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    print('VERDICT',verdict,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--arm',choices=['ELU1','LR'])
    ap.add_argument('--seed',type=int);ap.add_argument('--all',action='store_true');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--report',action='store_true');args=ap.parse_args()
    if args.smoke:smoke()
    elif args.report:report()
    elif args.all:
        assert 1<=args.workers<=4
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut=[ex.submit(run,a,s) for a in ('ELU1','LR') for s in (0,1,2)]
            for f in concurrent.futures.as_completed(fut):f.result()
        report()
    else:
        assert args.arm is not None and args.seed in (0,1,2)
        run(args.arm,args.seed)
if __name__=='__main__':main()


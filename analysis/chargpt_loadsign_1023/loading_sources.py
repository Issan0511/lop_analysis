#!/usr/bin/env python3
"""Loading-side source decomposition; distinct from softmax prediction sources."""
import argparse,csv,json,subprocess,time
import measure as M
from report import WINDOWS
A,C,torch,np=M.A,M.C,M.torch,M.np
OUT=M.OUT
ARCHIVE=M.DATA/'chargpt_loadsign_1023/results/chargpt_loadsign_1023'


@torch.no_grad()
def state(inp,arm,seed,task):
    model=inp.load(arm,seed,task);x,labels,_,_=A.data(seed,task)
    ybar,alive,_,vc,vl,kappa=M.task_fields(model,x,labels)
    mlp=model.h[1].mlp;U=mlp.down.weight;Uc=U-U.mean(0,keepdim=True)
    gamma=model.nf.weight;T=gamma[:,None]*Uc;G=T.T@T
    B=torch.zeros(C.V,C.FF,dtype=A.DT);Fr=torch.zeros(C.V,C.D,dtype=A.DT)
    for i,xx in A.chunks(x):
        cap=[];model.body(xx,cap)
        r=cap[2].reshape(-1,C.D);a=mlp.phi(cap[3].reshape(-1,C.FF))
        t=r+mlp.down(a);sigma=(M.cent(t).square().mean(-1,keepdim=True)+model.nf.eps).sqrt()
        xi=torch.nn.functional.one_hot(labels[i:i+len(xx)].reshape(-1),C.V).to(A.DT)-1/C.V
        B+=xi.T@(a/sigma);Fr+=xi.T@(M.cent(r)*gamma/sigma)
    ferr=M.relmax(kappa*(B@T.T+Fr)-vl,vl);assert ferr<=1e-10
    Qself=kappa*B*G.diag();Qr=kappa*Fr@T;Qother=kappa*B@G-Qself
    acc={f'{s}_{l}_{k}':torch.zeros(C.FF,dtype=A.DT) for s in ['self','other','residual'] for l in ['direct','exact'] for k in ['K','absK']}
    xu=A.uniform(seed);N=xu.numel()
    for _,xx in A.chunks(xu):
        cap=[];h=model.body(xx,cap).reshape(-1,C.D)
        r=cap[2].reshape(-1,C.D);z=cap[3].reshape(-1,C.FF)
        gate=mlp.dphi(z);y=model.h[1].n2(r);k=y@ybar
        t=r+mlp.down(mlp.phi(z));sigma=(M.cent(t).square().mean(-1,keepdim=True)+model.nf.eps).sqrt();q=M.cent(t)/sigma
        e=model.head(h).softmax(-1)-1/C.V
        qU=q@U
        radial=-qU*(q*(gamma*(e@vl))).mean(-1,keepdim=True)/sigma
        rr=-qU*(q*(gamma*(e@(kappa*Fr)))).mean(-1,keepdim=True)/sigma
        rs=-qU*(q@(gamma[:,None]*T))*(e@(kappa*B))/(C.D*sigma)
        ro=radial-rr-rs
        for name,Q,rad in [('self',Qself,rs),('other',Qother,ro),('residual',Qr,rr)]:
            direct=(e@Q)/sigma
            for lname,v in [('direct',direct),('exact',direct+rad)]:
                for kn,kv in [('K',k),('absK',k.abs())]:acc[f'{name}_{lname}_{kn}']+=(kv[:,None]*gate*v).sum(0)/N
    old=np.load(ARCHIVE/'raw'/f'{arm}_seed{seed}.npz')
    err=0.
    for l in ['direct','exact']:
        for k in ['K','absK']:
            total=sum(acc[f'{s}_{l}_{k}'] for s in ['self','other','residual'])
            ref=torch.from_numpy(old[f't{task}_original_{l}_{k}'])
            err=max(err,M.relmax(total-ref,ref))
            assert A.sink(total,alive)==A.sink(ref,alive)
    assert err<=1e-10,(arm,seed,task,err)
    row=dict(arm=arm,seed=seed,task=task,n_alive=int(alive.sum()),F_identity_error=ferr,push_identity_error=err)
    for key,v in acc.items():
        row['s_'+key]=A.sink(v,alive);row['abs_'+key]=M.median(v[alive].abs())
    for l in ['direct','exact']:
        den=sum(acc[f'{s}_{l}_K'][alive].abs() for s in ['self','other','residual']);nz=den>0
        for s in ['self','other','residual']:row[f'share_{s}_{l}']=M.median(acc[f'{s}_{l}_K'][alive][nz].abs()/den[nz])
    acc['alive']=alive
    return row,acc


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['gate','run']);args=ap.parse_args();inp=M.Inputs()
    if args.mode=='gate':
        rows=[state(inp,a,0,t)[0] for a in A.ARMS for t in [1,9,50,100]]
        M.dump(OUT/'loading_sources_gate.json',dict(passed=True,rows=rows,code_sha256=M.sha(__file__)));print('loading gate passed');return
    g=json.loads((OUT/'loading_sources_gate.json').read_text());assert g['passed'] and g['code_sha256']==M.sha(__file__)
    rows=[];start=time.monotonic()
    for seed in range(10):
        for arm in A.ARMS:
            rr=[];raw={};ts=time.monotonic()
            for t in A.TASKS:
                r,v=state(inp,arm,seed,t);rr.append(r);raw.update({f't{t}_{k}':x.numpy() for k,x in v.items()})
            p=OUT/'parts_loading'/f'{arm}_seed{seed}.csv';M.write_csv(p,rr)
            q=OUT/'raw_loading'/f'{arm}_seed{seed}.npz';q.parent.mkdir(exist_ok=True);np.savez_compressed(q,**raw)
            M.dump(p.with_suffix('.json'),dict(sha256=M.sha(p),raw_sha256=M.sha(q),code_sha256=M.sha(__file__),seconds=time.monotonic()-ts))
            rows.extend(rr);print(f'loading {arm} seed{seed}: {time.monotonic()-ts:.1f}s',flush=True)
        A.CACHE.clear();A.UCACHE.clear()
    M.write_csv(OUT/'loading_sources_states.csv',rows)
    fields=[k for k in rows[0] if k not in ['arm','seed','task']];wins=[]
    for arm in A.ARMS:
        for seed in range(10):
            for w,tasks in WINDOWS.items():
                r=[x for x in rows if x['arm']==arm and x['seed']==seed and x['task'] in tasks]
                wins.append(dict(arm=arm,seed=seed,window=w,**{k:float(np.median([x[k] for x in r])) for k in fields}))
    M.write_csv(OUT/'loading_sources_windows.csv',wins)
    M.dump(OUT/'loading_sources_provenance.json',dict(code_sha256=M.sha(__file__),git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M.ROOT,text=True).strip(),
        spec_sha256=M.sha(M.ROOT/'specs/spec_chargpt_loadsign_1023.md'),rows=len(rows),training_steps=0,seconds=time.monotonic()-start,
        max_F_error=max(r['F_identity_error'] for r in rows),max_push_error=max(r['push_identity_error'] for r in rows)))


if __name__=='__main__':main()

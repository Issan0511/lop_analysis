#!/usr/bin/env python3
"""Registered leak sweep of adaptation times; uses unchanged native host arithmetic."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
import argparse,hashlib,json,math,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import leaky_boundary_1111 as B
C=B.C;H=C.H;RL=C.RL;torch.set_num_threads(1)
RUN='leaky_order_1111';OUT=ROOT/'results'/RUN;SPEC=ROOT/'specs/spec_leaky_order_1111.md'
MAX=1536000;BUDGETS=[6000,24000,96000,384000,1536000]
def tag(a):return format(a,'.3f').replace('.','p')
def ident(args):
    return dict(args=vars(args),git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256=C.sha(__file__),spec_sha256=C.sha(SPEC),dependencies={n:C.sha(ROOT/n) for n in ['src/leaky_cliff_1111.py','src/leaky_boundary_1111.py']},
        data_sha256=C.sha(C.DATA/'train-images-idx3-ubyte.gz'),torch=torch.__version__,python=platform.python_version())
def checkpoint(net,adm,seed,age,a,gb,gl=None,step=0,times=None,streak=0):
    return dict(params=[p.detach().clone() for p in net.ps],m=[p.clone() for p in adm[0]],v=[p.clone() for p in adm[1]],tc=adm[2],
        seed=seed,age=age,a=a,g_batch=gb.get_state(),g_lab=gl.get_state() if gl is not None else None,probe_step=step,times=times or {},streak=streak)
def update_times(row,times,streak):
    if row['step']==0 or row['step']%75:return streak
    hits=row['correct'];n=row['n'];step=row['step']
    for pct in [80,90,95,99]:
        if 100*hits>=pct*n:times.setdefault('T'+str(pct),step)
    for threshold in [2.,1.5,1.,.5,.2,.1]:
        if row['ce']<=threshold:times.setdefault('Tce'+str(threshold),step)
    streak=streak+1 if 100*hits>=90*n else 0
    if streak>=4:times.setdefault('T90_sustained',step-225)
    return streak
@torch.no_grad()
def evaluation(net,x,y,step,expensive=False):
    out=net.forward(x);hits=int((out[2].argmax(1)==y).sum());row=dict(step=step,correct=hits,n=len(y),acc=hits/len(y),ce=float(F.cross_entropy(out[2],y)))
    assert math.isfinite(row['ce']),(step,row)
    if step in [0,1] or step%750==0:
        geom,_=B.measure(net,out,y,expensive=expensive);geom.pop('acc');geom.pop('ce');row.update(geom)
    return row,out
def fit(net,adm,x,y,gb,limit,dr,adaptive=False,seed=100,age=0,a=.001,task=0,start=0,rows=None,times=None,streak=0,save_raw=True):
    assert start%75==0 and limit%75==0 and limit>start
    rows=[] if rows is None else rows;times={} if times is None else times;t0=time.time();steps=start;online=[];losses=[]
    if start==0:
        row,out=evaluation(net,x,y,0,True);row['task']=task;rows.append(row)
        if save_raw:np.savez_compressed(dr/f'logits_t{task}_s0.npz',logits=out[2].numpy(),labels=y.numpy())
    for ep in range(start//75,limit//75):
        order=torch.randperm(len(x),generator=gb);ac=[];ce=[]
        for j in range(75):
            step=ep*75+j+1;ids=order[j*16:(j+1)*16];dense=step==1 or step%750==0
            if dense:
                with torch.no_grad():before=net.forward(x);ps0=[p.detach().clone() for p in net.ps]
            f=net.forward(x[ids],ids)[2];loss=F.cross_entropy(f,y[ids]);ac.append(float((f.detach().argmax(1)==y[ids]).float().mean()));ce.append(float(loss.detach()))
            grads=torch.autograd.grad(loss,net.ps);adm=C.adam_step(net.ps,grads,adm)
            if step==1 or step%75==0:
                row,out=evaluation(net,x,y,step,step in BUDGETS);row.update(task=task,online_acc=float(np.mean(ac)),online_ce=float(np.mean(ce)))
                if dense:
                    with torch.no_grad():
                        delta=[p.detach()-v for p,v in zip(net.ps,ps0)];df=out[2]-before[2];df-=df.mean(1,keepdim=True);err=torch.softmax(before[2],1)-F.one_hot(y,10).to(x.dtype)
                        row.update(actual_logit_rms=float(df.double().square().mean().sqrt()),actual_descent=float(-(err*df).sum()/len(x)),full_ce_drop=float(F.cross_entropy(before[2],y)-F.cross_entropy(out[2],y)))
                        for l in range(3):row[f'L{l+1}_grad_norm']=C.norm(grads[2*l:2*l+2]);row[f'L{l+1}_delta_norm']=C.norm(delta[2*l:2*l+2])
                rows.append(row);streak=update_times(row,times,streak)
        online.append(np.mean(ac));losses.append(np.mean(ce));steps=(ep+1)*75
        if steps in BUDGETS and save_raw:
            torch.save(checkpoint(net,adm,seed,age,a,gb,step=steps,times=times,streak=streak),dr/f'state_t{task}_s{steps}.pt')
            np.savez_compressed(dr/f'logits_t{task}_s{steps}.npz',logits=out[2].numpy(),labels=y.numpy())
        if steps%6000==0:
            B.write(dr/f'curves_t{task}.csv',rows)
            if steps in BUDGETS:print('budget',seed,a,age,task,steps,row['acc'],row['ce'],times.get('T90_sustained'),flush=True)
        if adaptive and steps>=24000 and 'T90_sustained' in times:break
    B.write(dr/f'curves_t{task}.csv',rows)
    summary=dict(seed=seed,a=a,age=age,task=task,steps=steps,endpoint_acc=rows[-1]['acc'],endpoint_ce=rows[-1]['ce'],seconds=time.time()-t0,
        online_acc=float(np.mean(online)) if online else None,online_ce=float(np.mean(losses)) if losses else None,
        stop_reason='sustained90_after_minimum24k' if adaptive and 'T90_sustained' in times else ('maximum_budget' if adaptive else 'fixed_prefix'),
        **{k:times.get(k) for k in ['T80','T90','T90_sustained','T95','T99','Tce2.0','Tce1.5','Tce1.0','Tce0.5','Tce0.2','Tce0.1']})
    for b in BUDGETS:
        if b<=steps:
            r=next(r for r in rows if r['step']==b);summary[f'acc_{b}']=r['acc'];summary[f'ce_{b}']=r['ce']
    return adm,summary,rows,times,streak
def prefix(args):
    dr=Path(args.out);dr.mkdir(parents=True,exist_ok=True);C.put(dr/'provenance.json',ident(args));x=C.get_x(args.seed)
    net=B.Model(H.init_params(args.seed,torch.device('cpu')),x,a=args.a);adm=C.init_adam(net.ps);gl=H.stream('rl_labels',args.seed);gb=H.stream('rl_batch',args.seed);records=[]
    torch.save(checkpoint(net,adm,args.seed,0,args.a,gb,gl),dr/'source_age0.pt')
    for task in range(1,args.tasks+1):
        y=RL.task_labels(gl);adm,row,_,_,_=fit(net,adm,x,y,gb,6000,dr,seed=args.seed,age=task-1,a=args.a,task=task);records.append(row);B.write(dr/'tasks.csv',records)
        if task in [3,6,args.tasks]:torch.save(checkpoint(net,adm,args.seed,task,args.a,gb,gl),dr/f'source_age{task}.pt')
    C.put(dr/'done.json',dict(tasks=args.tasks))
def check_reference(dr,seed,age):
    base=ROOT/'results/leaky_boundary_1111/probes/N'/f's{seed}'/f'age_{B.tag(age)}';old=np.genfromtxt(base/'curves_t0.csv',names=True,delimiter=',');new=np.genfromtxt(dr/'curves_t0.csv',names=True,delimiter=',')
    end=min(int(old['step'][-1]),int(new['step'][-1]));aa=old[(old['step']%75==0)&(old['step']<=end)];bb=new[(new['step']%75==0)&(new['step']<=end)]
    assert np.array_equal(aa['step'],bb['step']);ce=float(np.max(abs(aa['ce']-bb['ce'])));correct=bool(np.array_equal(np.rint(aa['acc']*1200),bb['correct']))
    assert ce==0 and correct,(seed,age,ce,correct)
    C.put(dr/'reference_check.json',dict(all_pass=True,old_run='leaky_boundary_1111',steps_compared=end,ce_maxabs=ce,integer_correct_counts_equal=correct,note='Thresholds use exact correct counts, including exactly90%, rather than comparing rounded float32 mean to decimal .9.'))
def probe(args):
    dr=Path(args.out);dr.mkdir(parents=True,exist_ok=True);C.put(dr/'provenance.json',ident(args));ck=torch.load(args.source,weights_only=False,map_location='cpu')
    assert ck['seed']==args.seed and ck['a']==args.a;age=ck['age'];x=C.get_x(args.seed);y=B.probe_labels(args.seed);gb=H.stream('boundary_probe_batch',args.seed)
    net=B.Model(ck['params'],x,a=args.a);adm=B.copy_adm(ck)
    C.put(dr/'branch_check.json',dict(source_sha256=C.sha(args.source),a=args.a,age=age,probe_labels_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
    adm,row,rows,times,streak=fit(net,adm,x,y,gb,args.max_steps,dr,adaptive=True,seed=args.seed,age=age,a=args.a)
    B.write(dr/'summary.csv',[row]);torch.save(checkpoint(net,adm,args.seed,age,args.a,gb,step=row['steps'],times=times,streak=streak),dr/'final.pt')
    if args.a==.003:check_reference(dr,args.seed,age)
    C.put(dr/'done.json',dict(summary=row))
def checks(args):
    dr=Path(args.out);dr.mkdir(parents=True,exist_ok=True);x=C.get_x(100);y=B.probe_labels(100);ps=H.init_params(100,torch.device('cpu'))
    n=B.Model(ps,x,a=.001);m=B.Model(ps,x,a=.001);an=C.init_adam(n.ps);am=C.init_adam(m.ps)
    g1=H.stream('boundary_probe_batch',100);g2=H.stream('boundary_probe_batch',100)
    for p in [dr/'whole',dr/'split']:p.mkdir(exist_ok=True)
    an,r,rows,times,streak=fit(n,an,x,y,g1,300,dr/'whole',save_raw=False)
    am,rr,rs,tt,st=fit(m,am,x,y,g2,150,dr/'split',save_raw=False)
    ck=checkpoint(m,am,100,0,.001,g2,step=150,times=tt,streak=st);torch.save(ck,dr/'restart.pt');ck=torch.load(dr/'restart.pt',weights_only=False)
    m=B.Model(ck['params'],x,a=ck['a']);am=B.copy_adm(ck);g2=C.rng(ck['g_batch'])
    am,rr,rs,tt,st=fit(m,am,x,y,g2,300,dr/'split',start=150,rows=rs,times=tt,streak=st,save_raw=False)
    assert all(torch.equal(p,q) for p,q in zip(n.ps,m.ps));assert all(torch.equal(p,q) for grp1,grp2 in zip(an[:2],am[:2]) for p,q in zip(grp1,grp2));assert torch.equal(g1.get_state(),g2.get_state());assert times==tt
    timer={};streak=0
    for step,hits in [(75,1079),(150,1080),(225,1080),(300,1080),(375,1080)]:streak=update_times(dict(step=step,correct=hits,n=1200,ce=1),timer,streak)
    assert timer['T90']==150 and timer['T90_sustained']==150
    timer={};streak=0
    for step,hits in [(75,1080),(150,1080),(225,1079),(300,1080),(375,1080),(450,1080)]:streak=update_times(dict(step=step,correct=hits,n=1200,ce=1),timer,streak)
    assert timer['T90']==75 and 'T90_sustained' not in timer
    assert torch.equal(B.probe_labels(100),B.probe_labels(100))
    C.put(dr/'checks.json',dict(all_pass=True,restart_params_and_moments_and_rng_exact=True,exact90_percent_qualifies=True,failed_streak_is_censored=True,code_sha256=C.sha(__file__)));print('checks pass')
def hostcheck(args):
    new=torch.load(Path(args.out)/'source_age2.pt',weights_only=False);old=torch.load(Path(args.source)/'state_t2.pt',weights_only=False);r={}
    for key in ['params','m','v']:r[key]=all(torch.equal(p,q) for p,q in zip(new[key],old[key]))
    for key in ['g_lab','g_batch']:r[key]=torch.equal(new[key],old[key])
    r['tc']=new['tc']==old['tc'];assert all(r.values());C.put(Path(args.out)/'host_check.json',dict(all_pass=True,seed=100,a=.001,tasks=2,code_sha256=C.sha(__file__),exact=r));print('full host match',r)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prefix','probe','checks','hostcheck']);p.add_argument('--seed',type=int,default=100);p.add_argument('--a',type=float,default=.001);p.add_argument('--tasks',type=int,default=6);p.add_argument('--max-steps',type=int,default=MAX);p.add_argument('--source');p.add_argument('--out',required=True)
    a=p.parse_args();globals()[a.mode](a)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Held-out background fits, local logit interventions, and individual lifetimes."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/redo_positive_lineage_1019'


def write(path, rows):
    rows=list(rows)
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,keys); w.writeheader(); w.writerows(rows)


def finite_median(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def r2(y,pred):
    mse=np.mean((y-pred)**2,axis=0)
    var=np.var(y,axis=0)
    return np.where(var>1e-20,1-mse/np.maximum(var,1e-20),np.nan)


def fit_leave_self(A,Z,AH,ZH,columns=None,relative=1e-6):
    """Constrained ridge for every target: coefficient of its own activation is exactly 0."""
    if columns is None: columns=np.arange(A.shape[1])
    columns=np.asarray(columns,dtype=int)
    am=A[:,columns].mean(0); zm=Z.mean(0)
    ac=A[:,columns]-am; zc=Z-zm
    if not len(columns):
        return np.broadcast_to(zm,ZH.shape).copy(), np.broadcast_to(zm,Z.shape).copy()
    gram=ac.T@ac/len(A)
    lam=max(float(np.trace(gram)/len(columns))*relative,1e-15)
    inv=np.linalg.solve(gram+lam*np.eye(len(columns)),np.eye(len(columns)))
    coef=inv@(ac.T@zc/len(A))
    for j,i in enumerate(columns):
        coef[:,i]-=inv[:,j]*(coef[j,i]/inv[j,j])
        coef[j,i]=0
    return (AH[:,columns]-am)@coef+zm, ac@coef+zm


def local_gradients(h,z,W,b,V,c,y,full_open,provider):
    """CE gradients at fixed features; zero selected readout contributions without refitting."""
    ac=np.maximum(h@W.T+b,0); gate=(h@W.T+b)>0
    logits=ac@V.T+c
    def softmax(L):
        L=L-L.max(-1,keepdims=True); p=np.exp(L); return p/p.sum(-1,keepdims=True)
    err=softmax(logits); err[np.arange(len(y)),y]-=1
    delta=(err@V)*gate
    # N x input is used to return full weight gradient direction on a caller-specified mean.
    full_b=delta.mean(0); full_w=delta.T@h/len(h)
    # target x samples x classes; retain each target's own contribution and output bias.
    own=ac.T[:,:,None]*V.T[:,None,:]+c
    ep=softmax(own)
    ep[:,np.arange(len(y)),y]-=1
    dd=np.einsum('inc,ic->in',ep,V.T)*gate.T
    self_b=dd.mean(1); self_w=dd@h/len(h)
    # Remove other full-open units, while retaining target if it is itself full-open.
    base=logits-ac[:,full_open]@V[:,full_open].T
    L=np.broadcast_to(base,(len(W),*base.shape)).copy()
    for i in full_open:
        L[i]+=ac[:,i,None]*V[:,i][None,:]
    ee=softmax(L); ee[:,np.arange(len(y)),y]-=1
    dd=np.einsum('inc,ic->in',ee,V.T)*gate.T
    open_b=dd.mean(1); open_w=dd@h/len(h)
    L=logits[None,:,:]-ac[:,provider].T[:,:,None]*V[:,provider].T[:,None,:]
    ee=softmax(L); ee[:,np.arange(len(y)),y]-=1
    dd=np.einsum('inc,ic->in',ee,V.T)*gate.T
    return dict(full_b=full_b,full_w=full_w,self_b=self_b,self_w=self_w,
                without_open_b=open_b,without_open_w=open_w,
                without_provider_b=dd.mean(1),without_provider_w=dd@h/len(h))


def deep_one(d,r):
    h,hv,z,zv=[d[k][r].astype(np.float64) for k in ('h1','held_h1','z2','held_z2')]
    a,av=np.maximum(z,0),np.maximum(zv,0)
    g=(z>0).astype(float); gv=(zv>0).astype(float)
    n,m=a.shape
    pp, _=fit_leave_self(a,z,av,zv)
    ra=r2(zv,pp)
    a_pred,a_train=fit_leave_self(a,a,av,av)
    _,g_train=fit_leave_self(a,g,av,gv)
    qA=a-a_train; qG=g-g_train
    allproxy=np.mean(qA*qG,axis=0)
    allopen=np.flatnonzero((z.min(0)>0)&(zv.min(0)>0))
    op,_=fit_leave_self(a,z,av,zv,allopen)
    ro=r2(zv,op)
    ac=a-a.mean(0); zc=z-z.mean(0); gc=g-g.mean(0)
    va=np.mean(ac*ac,0); vz=np.mean(zc*zc,0)
    cov=ac.T@zc/n
    scores=cov**2/np.maximum(va[:,None]*vz[None,:],1e-30)
    np.fill_diagonal(scores,-np.inf)
    scores[va<1e-20,:]=-np.inf
    provider=np.argmax(scores,axis=0)
    for i in range(m):
        if provider[i]==i: provider[i]=(i+1)%m
    slope=cov[provider,np.arange(m)]/np.maximum(va[provider],1e-30)
    pred=z.mean(0)+(av[:,provider]-a.mean(0)[provider])*slope
    rs=r2(zv,pred)
    W=d['W2'][r].astype(np.float64)
    cosine=np.sum(W*W[provider],1)/np.maximum(np.linalg.norm(W,axis=1)*np.linalg.norm(W[provider],axis=1),1e-30)
    ba=np.mean(zc*ac,0)/np.maximum(vz,1e-30)
    bg=np.mean(zc*gc,0)/np.maximum(vz,1e-30)
    qa=ac-zc*ba; qg=gc-zc*bg
    affine=np.mean(qa*qg,0)
    affine_ratio=np.mean(qa*qa,0)/np.maximum(va,1e-30)
    grad=local_gradients(d['current_h1'][r].astype(float),z,W,d['b2'][r].astype(float),
                         d['W3'][r].astype(float),d['b3'][r].astype(float),
                         d['current_y'][r],allopen,provider)
    mu=h.mean(0)
    out=dict(p=g.mean(0),held_p=gv.mean(0),z_mean=z.mean(0),z_sd=z.std(0),
             background_r2=ra,open_background_r2=ro,single_r2=rs,single_provider=provider,
             single_cos=cosine,provider_p=g.mean(0)[provider],provider_held_p=gv.mean(0)[provider],
             n_open_others=np.full(m,len(allopen))-np.isin(np.arange(m),allopen),
             affine_proxy=affine,actual_background_proxy=allproxy,
             affine_residual_ratio=affine_ratio,actual_residual_ratio=np.mean(qA*qA,0)/np.maximum(va,1e-30),
             raw_proxy=np.mean(g*a,0),center_proxy=np.mean(gc*ac,0),
             outnorm=np.linalg.norm(d['W3'][r],axis=0))
    for name in ('full','self','without_open','without_provider'):
        out[name+'_bias_direction']=-grad[name+'_b']
        out[name+'_mean_direction']=-(grad[name+'_w']@mu+grad[name+'_b'])
    if True: # preregistered sensitivity at selected major points is selected by caller.
        out['_arrays']=(a,z,av,zv)
    return out


def summarize_group(path, method, births, firsts, steps, resetrows, summaries):
    meta=json.loads((path/'trace/events.json').read_text()); seeds=meta['seeds']; ev=meta['events']
    u=np.load(path/'trace/units.npz'); dd=np.load(path/'trace/deltas.npz')
    valid=[j for j,e in enumerate(ev) if e['event'] in ('init','after_step','reset_outgoing_2')]
    for r,seed in enumerate(seeds):
        for scope in ('g','t3'):
            p=u[f'{scope}_l2_p'][:,r]
            seen=set(); initial=int((p[0]==1).sum()); first=None
            for vi,j in enumerate(valid):
                if not np.isfinite(p[j]).all(): continue
                gen=u['generation'][j,r,1].astype(int)
                for i in np.flatnonzero(p[j]==1):
                    key=(int(i),int(gen[i]))
                    if key in seen: continue
                    seen.add(key)
                    e=ev[j]; previous=valid[vi-1] if vi else j
                    samegen=int(u['generation'][previous,r,1,i])==gen[i]
                    nexts=[k for k in valid[vi+1:] if int(u['generation'][k,r,1,i])==gen[i] and np.isfinite(p[k,i])]
                    drops=[k for k in nexts if p[k,i]<1]
                    drops99=[k for k in nexts if p[k,i]<.99]
                    drops50=[k for k in nexts if p[k,i]<.5]
                    last=nexts[-1] if nexts else j
                    drop=drops[0] if drops else None
                    b=dict(method=method,seed=seed,scope=scope,unit=int(i),generation=int(gen[i]),
                           event_index=j,event=e['event'],task=e['task'],step=e['step'],tc=e['tc'],
                           onset='initial' if j==0 else ('reset' if e['event']=='reset_outgoing_2' else 'learning'),
                           age=e['tc']-int(u['last_reset'][j,r,1,i]),
                           previous_index=previous,previous_p=float(p[previous,i]) if samegen else None,
                           zmean=float(u[f'{scope}_l2_mean'][j,r,i]),
                           r=float(u[f'{scope}_l2_r'][j,r,i]),
                           observed_allon_duration=(ev[drop]['tc'] if drop is not None else ev[last]['tc'])-e['tc'],
                           dropped=int(drop is not None),drop_task=ev[drop]['task'] if drop is not None else None,
                           dropped99=int(bool(drops99)),dropped50=int(bool(drops50)),
                           last_p=float(p[last,i]),last_task=ev[last]['task'],
                           outnorm=float(u['outnorm_l2'][j,r,i]))
                    births.append(b)
                    if first is None: first=b
            firsts.append(dict(method=method,seed=seed,scope=scope,initial_allon=initial,
                               n_allon_lifetimes=len(seen),first_tc=first['tc'] if first else None,
                               first_event=first['event'] if first else None,
                               first_task=first['task'] if first else None))
        for k in range(len(dd['task'])):
            pre=dd['pre_p'][k,r]
            for cohort,mask in [('all',np.ones(100,dtype=bool)),('all_on',pre==1),('near99',pre>=.99),
                                ('half_positive',pre>.5),('half_negative',pre<.5)]:
                if not mask.any(): continue
                row=dict(method=method,seed=seed,task=int(dd['task'][k]),tc=int(dd['tc'][k]),
                         step=int(dd['step'][k]),cohort=cohort,n=int(mask.sum()))
                for name in ('actual','own_w','own_b','upstream','adam_current','adam_memory','sgd_direction'):
                    vv=dd[name][k,r,mask]
                    row[name+'_mean']=float(vv.mean()); row[name+'_positive']=int((vv>0).sum())
                    row[name+'_negative']=int((vv<0).sum())
                own=(dd['own_w'][k,r]+dd['own_b'][k,r])[mask]
                row['own_total_negative']=int((own<0).sum())
                row['own_total_positive']=int((own>0).sum())
                row['max_closure']=float(np.abs(dd['closure'][k,r]).max())
                steps.append(row)
        for j,e in enumerate(ev):
            if not e['event'].startswith('reset_'): continue
            p=u['g_l2_p'][j,r]; mask=u['reset_mask'][j,r,1].astype(bool)
            row=dict(method=method,seed=seed,**e,n_selected=int(mask.sum()),
                     all_on=int((p==1).sum()),near99=int((p>=.99).sum()),
                     selected_mean=float(u['g_l2_mean'][j,r,mask].mean()) if mask.any() else None,
                     survivor_mean=float(u['g_l2_mean'][j,r,~mask].mean()) if (~mask).any() else None)
            resetrows.append(row)
    maxclos=float(np.abs(dd['closure']).max())
    summaries.append(dict(method=method,seeds=seeds,max_position_closure=maxclos,
                          max_adam_closure=float(np.abs(dd['adam_closure']).max()),
                          original_reproduction=json.loads((path/'reproduction.json').read_text())['exact']))


def analyze_deep(path, method, outrows, eventrows, lookup):
    meta=json.loads((path/'trace/events.json').read_text()); seeds=meta['seeds']; events=meta['events']
    for f in sorted((path/'trace').glob('deep_*.npz')):
        index=int(f.stem.split('_')[-1]); event=events[index]; d=np.load(f)
        print(f'DEEP {method} seeds={seeds[0]}-{seeds[-1]} index={index} t={event["task"]} {event["event"]}',flush=True)
        for r,seed in enumerate(seeds):
            met=deep_one(d,r); arrays=met.pop('_arrays')
            if event['event'] in ('init','task_end'):
                for rel,label in ((1e-8,'small'),(1e-4,'large')):
                    pred,_=fit_leave_self(*arrays,relative=rel)
                    met['background_r2_'+label]=r2(arrays[3],pred)
            for i in range(100):
                row=dict(method=method,seed=seed,index=index,task=event['task'],event=event['event'],
                         tc=event['tc'],step=event['step'],unit=i)
                row.update({k:float(v[i]) for k,v in met.items()})
                lookup[(method,seed,index,i)]=row
                outrows.append(row)
            for cohort,mask in [('all',np.ones(100,dtype=bool)),('all_on',met['p']==1),
                                ('near99',met['p']>=.99),('half_positive',met['p']>.5)]:
                if not mask.any(): continue
                row=dict(method=method,seed=seed,index=index,task=event['task'],event=event['event'],
                         tc=event['tc'],cohort=cohort,n=int(mask.sum()))
                for key in ('background_r2','open_background_r2','single_r2','single_cos','affine_residual_ratio',
                            'actual_residual_ratio','outnorm'):
                    row[key+'_median']=finite_median(met[key][mask])
                for key in ('background_r2','open_background_r2','single_r2'):
                    row[key+'_ge95']=int((met[key][mask]>=.95).sum())
                    row[key+'_ge99']=int((met[key][mask]>=.99).sum())
                for key in ('full_mean_direction','self_mean_direction','without_open_mean_direction',
                            'without_provider_mean_direction','affine_proxy','actual_background_proxy'):
                    row[key+'_positive']=int((met[key][mask]>0).sum())
                    row[key+'_negative']=int((met[key][mask]<0).sum())
                row['open_ablation_sign_flip']=int(((met['full_mean_direction'][mask]>0)!=(met['without_open_mean_direction'][mask]>0)).sum())
                row['provider_ablation_sign_flip']=int(((met['full_mean_direction'][mask]>0)!=(met['without_provider_mean_direction'][mask]>0)).sum())
                eventrows.append(row)


def math_checks():
    rng=np.random.default_rng(1019)
    A=rng.normal(size=(64,5)); Z=rng.normal(size=(64,5)); AH=rng.normal(size=(31,5)); ZH=rng.normal(size=(31,5))
    pred,_=fit_leave_self(A,Z,AH,ZH)
    ac=A-A.mean(0); ah=AH-A.mean(0); zc=Z-Z.mean(0)
    lam=np.trace(ac.T@ac/64)/5*1e-6
    err=0
    for i in range(5):
        cols=[j for j in range(5) if j!=i]
        coef=np.linalg.solve(ac[:,cols].T@ac[:,cols]/64+lam*np.eye(4),ac[:,cols].T@zc[:,i]/64)
        direct=ah[:,cols]@coef+Z[:,i].mean()
        err=max(err,float(np.abs(pred[:,i]-direct).max()))
    if err>1e-10: raise RuntimeError(f'constrained fit mismatch {err}')
    h=rng.normal(size=(9,3)); W=rng.normal(size=(5,3)); b=rng.normal(size=5)
    V=rng.normal(size=(4,5)); c=rng.normal(size=4); y=rng.integers(4,size=9)
    gg=local_gradients(h,h@W.T+b,W,b,V,c,y,np.array([1,3]),np.array([1,2,3,4,0]))
    def loss(ww,bb,kind,i):
        a=np.maximum(h@ww.T+bb,0)
        if kind=='self':
            L=a[:,i,None]*V[:,i][None,:]+c
        elif kind=='without_open':
            keep=np.ones(5,dtype=bool); keep[[1,3]]=False; keep[i]=True
            L=a[:,keep]@V[:,keep].T+c
        else: L=a@V.T+c
        lmax=L.max(1)
        return float(np.mean(lmax+np.log(np.exp(L-lmax[:,None]).sum(1))-L[np.arange(9),y]))
    gerr=0
    eps=1e-6; mu=h.mean(0)
    for kind in ('full','self','without_open'):
        for i in range(5):
            wp,wm=W.copy(),W.copy(); bp,bm=b.copy(),b.copy()
            wp[i]+=eps*mu; wm[i]-=eps*mu; bp[i]+=eps; bm[i]-=eps
            numerical=(loss(wp,bp,kind,i)-loss(wm,bm,kind,i))/(2*eps)
            exact=gg[kind+'_w'][i]@mu+gg[kind+'_b'][i]
            gerr=max(gerr,abs(numerical-exact))
    if gerr>1e-7: raise RuntimeError(f'local CE gradient mismatch {gerr}')
    return dict(constrained_ridge_max_error=err,local_ce_gradient_max_error=gerr)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks=math_checks()
    births=[]; firsts=[]; steps=[]; resets=[]; summaries=[]; unitdeep=[]; eventdeep=[]; lookup={}
    for method in ('redo','none'):
        for group in (0,10):
            path=OUT/'raw'/f'{method}_s{group}-{group+9}'
            summarize_group(path,method,births,firsts,steps,resets,summaries)
            analyze_deep(path,method,unitdeep,eventdeep,lookup)
    for b in births:
        for when,j in (('birth',b['event_index']),('before',b['previous_index'])):
            rec=lookup.get((b['method'],b['seed'],j,b['unit']))
            if rec:
                for k in ('background_r2','open_background_r2','single_r2','single_provider','single_cos',
                          'n_open_others','full_mean_direction','without_open_mean_direction','self_mean_direction',
                          'affine_proxy','actual_background_proxy'):
                    b[when+'_'+k]=rec[k]
    write(OUT/'births.csv',births)
    write(OUT/'first_allon.csv',firsts)
    write(OUT/'step_cohorts.csv',steps)
    write(OUT/'reset_stages.csv',resets)
    write(OUT/'deep_cohorts.csv',eventdeep)
    write(OUT/'raw/unit_deep_metrics.csv',unitdeep)
    checks['groups']=summaries
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2))
    result={}
    for method in ('redo','none'):
        ff=[r for r in firsts if r['method']==method and r['scope']=='g']
        bb=[r for r in births if r['method']==method and r['scope']=='g']
        ss=[r for r in steps if r['method']==method and r['cohort']=='all_on']
        d={
            'initial_global_allon_seeds':sum(r['initial_allon']>0 for r in ff),
            'initial_global_allon_units':sum(r['initial_allon'] for r in ff),
            'ever_global_allon_seeds':sum(r['first_tc'] is not None for r in ff),
            'global_allon_lifetimes':len(bb),
            'births_by_onset':{k:sum(r['onset']==k for r in bb) for k in ('initial','reset','learning')},
            'ever_drop_from_full':sum(r['dropped'] for r in bb),
            'ever_drop_below_99':sum(r['dropped99'] for r in bb),
            'ever_drop_below_half':sum(r['dropped50'] for r in bb),
            'sampled_allon_unit_steps':sum(r['n'] for r in ss),
            'actual_downward_unit_steps':sum(r['actual_negative'] for r in ss),
            'own_downward_unit_steps':sum(r['own_total_negative'] for r in ss),
            'sgd_downward_unit_steps':sum(r['sgd_direction_negative'] for r in ss),
        }
        result[method]=d
    (OUT/'verdict.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

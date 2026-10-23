#!/usr/bin/env python3
"""Posthoc join of saved opening rates and phase-2 force arrays; no model calls."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/issan/Projects/obsidian-research-data')
OLD = DATA/'chargpt_earlyphase_1024/results/chargpt_earlyphase_1024/raw'
NEW = DATA/'chargpt_phase2conditions_1024/results/chargpt_phase2conditions_1024/raw'
OUT = ROOT/'results/chargpt_popen_1024'
WINDOWS = {'early': [1,2,3,4], 'late': [6,7,8,9]}
BINS = [0,.1,.2,.3,.4,.5,.6,.8,1.000001]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def write_csv(name, rows):
    with (OUT/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def med(x):
    return float(np.median(x)) if len(x) else None

def rho(x, y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return None
    def rank(v):
        _, ix, counts = np.unique(v, return_inverse=True, return_counts=True)
        mids = np.cumsum(counts) - (counts-1)/2
        return mids[ix]
    return float(np.corrcoef(rank(x),rank(y))[0,1])

def cosine(x,y):
    den = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den else None

def stats(p,m,d,a):
    nz = (m!=0)&(d!=0)
    prod = m*d
    return dict(n=len(p), p_med=med(p), p_mean=float(np.mean(p)), p_gt_half=float(np.mean(p>.5)),
                p_lt_half=float(np.mean(p<.5)), p_min=float(p.min()), p_max=float(p.max()),
                reinforce=float(np.mean(prod[nz]>0)), M_sink=float(np.mean(m>0)),
                D_sink=float(np.mean(d>0)), actual_sink=float(np.mean(a>0)),
                rho_p_M=rho(p,m), rho_p_D=rho(p,d), rho_p_product=rho(p,prod),
                cos_M_D=cosine(m,d))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, sources, checks, cache = [], [], [], {}
    for arm in ['C3','ReLUcap']:
        for seed in range(10):
            for task in range(1,10):
                po = OLD/f'{arm}_seed{seed}_task{task}.npz'
                pn = NEW/f'{arm}_seed{seed}_task{task}_u10000.npz'
                with np.load(po) as old, np.load(pn) as new:
                    cc = dict(arm=arm, seed=seed, task=task)
                    for layer in [0,1]:
                        assert np.array_equal(old[f'L{layer}_alive'],new[f'L{layer}_mask'])
                        pairs = [(f'L{layer}_r',f'L{layer}_r')]
                        for metric in ['SGD','Adam_fixed']:
                            pairs += [(f'L{layer}_{metric}_r_P2',f'L{layer}_{metric}_P2'),
                                      (f'L{layer}_{metric}_r_switch',f'L{layer}_{metric}_actual')]
                        for ok,nk in pairs:
                            aa,bb=old[ok],new[nk]
                            err=float(np.max(np.abs(aa-bb))/max(np.max(np.abs(aa)),np.max(np.abs(bb)),1e-300))
                            assert err<1e-8,(po,nk,err)
                            cc[nk]=err
                    for metric in ['SGD','Adam_fixed']:
                        p=old['L0_p'].copy()
                        mask=old['L0_alive'].astype(bool)
                        m=new[f'L0_{metric}_iid_mean'].copy()
                        d=new[f'L0_{metric}_task_departure'].copy()
                        a=new[f'L0_{metric}_actual'].copy()
                        cache[arm,seed,task,metric]=dict(p=p,m=m,d=d,a=a,mask=mask)
                        records.append(dict(arm=arm, seed=seed,task=task,metric=metric,
                                            **stats(p[mask],m[mask],d[mask],a[mask])))
                    checks.append(cc)
                sources.extend([dict(path=str(q),bytes=q.stat().st_size,sha256=sha(q)) for q in [po,pn]])
    write_csv('states.csv',records)
    win, bins, trans, temporal = [], [], [], []
    for arm in ['C3','ReLUcap']:
        for seed in range(10):
            for metric in ['SGD','Adam_fixed']:
                rr=[r for r in records if (r['arm'],r['seed'],r['metric'])==(arm,seed,metric)]
                temporal.append(dict(arm=arm,seed=seed,metric=metric,n_states=len(rr),
                    rho_state_p_median_cos=rho([r['p_med'] for r in rr],[r['cos_M_D'] for r in rr]),
                    rho_state_p_mean_cos=rho([r['p_mean'] for r in rr],[r['cos_M_D'] for r in rr])))
                bywindow={}
                for window,tasks in WINDOWS.items():
                    rows=[r for r in records if (r['arm'],r['seed'],r['metric'])==(arm,seed,metric) and r['task'] in tasks]
                    win.append(dict(arm=arm,seed=seed,metric=metric,window=window,
                                    **{k:med([r[k] for r in rows if r[k] is not None]) for k in rows[0] if k not in ['arm','seed','metric','task']}))
                    vv=[cache[arm,seed,t,metric] for t in tasks]
                    bywindow[window]={k:np.median(np.stack([v[k] for v in vv]),axis=0) for k in ['p','m','d','a']}
                    bywindow[window]['mask']=np.logical_and.reduce([v['mask'] for v in vv])
                    pooled={k:np.concatenate([v[k][v['mask']] for v in vv]) for k in ['p','m','d','a']}
                    for lo,hi in zip(BINS[:-1],BINS[1:]):
                        keep=(pooled['p']>=lo)&(pooled['p']<hi)
                        if not keep.any(): continue
                        ss=stats(*[pooled[k][keep] for k in ['p','m','d','a']])
                        bins.append(dict(arm=arm,seed=seed,metric=metric,window=window,lo=lo,hi=min(hi,1.),**ss))
                e,l=bywindow['early'],bywindow['late']
                keep=e['mask']&l['mask']
                flip=keep&(e['m']*e['d']<0)&(l['m']*l['d']>0)
                cross=(e['p']-.5)*(l['p']-.5)<0
                mf=np.sign(e['m'])!=np.sign(l['m'])
                df=np.sign(e['d'])!=np.sign(l['d'])
                trans.append(dict(arm=arm,seed=seed,metric=metric,n=int(keep.sum()),
                    p_delta_med=med((l['p']-e['p'])[keep]), n_cross_half=int((cross&keep).sum()),
                    n_oppose_to_reinforce=int(flip.sum()), n_flip_cross_half=int((flip&cross).sum()),
                    n_flip_M_only=int((flip&mf&~df).sum()), n_flip_D_only=int((flip&~mf&df).sum()),
                    n_flip_M_sink_to_float=int((flip&mf&~df&(e['m']>0)&(l['m']<0)).sum()),
                    n_flip_D_float_to_sink=int((flip&~mf&df&(e['d']<0)&(l['d']>0)).sum()),
                    n_flip_both=int((flip&mf&df).sum()),
                    n_p_decrease=int(((l['p']<e['p'])&keep).sum()),
                    rho_delta_p_delta_product=rho((l['p']-e['p'])[keep],(l['m']*l['d']-e['m']*e['d'])[keep])))
    write_csv('windows.csv',win)
    write_csv('bins.csv',bins)
    write_csv('transitions.csv',trans)
    write_csv('temporal.csv',temporal)
    dump(OUT/'checks.json',checks)
    summary=[]
    for arm in ['C3','ReLUcap']:
        for metric in ['SGD','Adam_fixed']:
            for window in WINDOWS:
                vv=[w for w in win if (w['arm'],w['metric'],w['window'])==(arm,metric,window)]
                summary.append(dict(arm=arm,metric=metric,window=window,
                                    **{k:[min(w[k] for w in vv),max(w[k] for w in vv)] for k in vv[0] if k not in ['arm','metric','window','seed']}))
    dump(OUT/'summary.json',summary)
    dump(OUT/'provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         code_sha256=sha(Path(__file__)),spec_sha256=sha(ROOT/'specs/spec_chargpt_popen_1024.md'),
         state_count=180,training_steps=0,model_calls=0,scope='L0 own up row',sources=sources))
    print(json.dumps([s for s in summary if s['metric']=='SGD'],indent=2))
    print('TRANSITIONS',json.dumps([r for r in trans if r['metric']=='SGD'],indent=2))
    print('maximum join relative error', max(v for c in checks for k,v in c.items() if k not in ['arm','seed','task']))

if __name__=='__main__': main()

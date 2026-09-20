#!/usr/bin/env python3
"""Post-run replay mutations and actual-state resume validation. No training."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='2'
import json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analysis.cifar_ledger_0920 import replay as R


def reject(fn):
    try:fn()
    except AssertionError as exc:return dict(detected=True,message=str(exc)[:400])
    return dict(detected=False)


def run():
    start=time.time();dev=R.setup();cifar=R.E.RC.Cifar10()
    X=torch.stack([R.E.slot_inputs(cifar,s,c,dev) for s,c in R.SLOTS]);xc=X.cpu().numpy()
    result=dict(independent_audit=False,scope='post-run actual-state mutation and resume checks',mutations={})
    for arm in ('ELU','KKT1'):
        ds,P,act=R.load_state(arm,1,dev)
        hist=[R.read_npz(R.RAW/arm/'hist'/f'{arm}_{c}_seed{s}.npz') for s,c in R.SLOTS]
        rec={(int(x['seed']),x['cond'],int(x['task'])):x for x in R.records(R.PARENT/arm/'per_task.csv')}
        Y=torch.stack([R.E.RC.task_labels(R.E.H.stream('rlc_labels',s)) for s,c in R.SLOTS]).to(dev)
        with torch.no_grad():z1,a1,z2,a2,logits=R.E.forward(P,X,act)
        def check(zs=(z1,z2),lo=logits,hist_=hist):return R.snap_check(arm,1,zs,lo,Y,act,ds,hist_,rec)
        check()
        if arm=='ELU':
            for name,dest,src in [('std_as_raw',1,0),('wrong_seed_images',0,2)]:
                bad=X.clone();bad[dest]=X[src]
                with torch.no_grad():v=R.E.forward(P,bad,act)
                r=reject(lambda:check((v[0],v[2]),v[4]));r['changed_preactivation_elements']=int((v[0]!=z1).sum())
                result['mutations'][name]=r
            hm=[dict(x) for x in hist];hm[0]['m1']=np.roll(hm[0]['m1'],1,axis=0)
            result['mutations']['hist_task_shift']=reject(lambda:check(hist_=hm))
            result['mutations']['t01_as_t00']=reject(lambda:R.snap_check(arm,0,(z1,z2),logits,None,act,ds,hist,rec))
        else:
            saved=[v.clone() for v in act.V]
            for name,mut in [('V_ones',[torch.ones_like(v) for v in saved]),('V_layers_swapped',list(reversed(saved)))]:
                act.V=mut
                with torch.no_grad():v=R.E.forward(P,X,act)
                r=reject(lambda:check((v[0],v[2]),v[4]));r['changed_layer2_elements']=int((v[2]!=z2).sum())
                result['mutations'][name]=r
            act.V=saved
    # Rebuild an interrupted ELU t1 state, then evaluate t2. Compare every numeric
    # column with the completed uninterrupted run, including the transition ledger.
    arm='ELU';hist=[R.read_npz(R.RAW/arm/'hist'/f'{arm}_{c}_seed{s}.npz') for s,c in R.SLOTS]
    rec={(int(x['seed']),x['cond'],int(x['task'])):x for x in R.records(R.PARENT/arm/'per_task.csv')}
    gens=[R.E.H.stream('rlc_labels',s) for s,c in R.SLOTS]
    y1=torch.stack([R.E.RC.task_labels(g) for g in gens]).to(dev)
    _,state,_=R.compute(arm,1,X,xc,dev,hist,rec,y1,None)
    y2=torch.stack([R.E.RC.task_labels(g) for g in gens]).to(dev)
    resumed,_,ev=R.compute(arm,2,X,xc,dev,hist,rec,y2,state)
    uninterrupted=R.read_npz(R.output_artifact(R.OUT,'shards/ELU_t02.npz'))
    equal={k:np.array_equal(v,uninterrupted[k],equal_nan=True) for k,v in resumed.items()}
    result['actual_resume']=dict(all_equal=all(equal.values()),keys=len(equal),checks=ev)
    result['all_pass']=all(x['detected'] for x in result['mutations'].values()) and result['actual_resume']['all_equal']
    result['elapsed_s']=time.time()-start
    result['source_sha256']={str(p.relative_to(R.ROOT)):R.sha(p) for p in Path(__file__).parent.glob('*.py')}
    R.put(R.OUT/'actual_state_checks.json',result)
    print(json.dumps(result,indent=2))
    assert result['all_pass']


if __name__=='__main__':
    with R.exclusive():run()

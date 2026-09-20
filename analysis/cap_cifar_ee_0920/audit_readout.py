"""Post-run CSV audit, prepared before reading outcomes. Implementer cross-check only."""
import csv,hashlib,json,math,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from src import pmnist_0905 as H
P=Path('results/cap_cifar_ee_0920')
assert json.loads((P/'status.json').read_text())['stage']=='completed'
rows=list(csv.DictReader((P/'per_task.csv').open()));arms=['ref','cap1','cap2','cap12','cap12_bfix']
lookup={(r['arm'],int(r['seed']),int(r['task'])):r for r in rows}
assert len(rows)==len(lookup)==2500 and set(lookup)=={(a,s,t) for a in arms for s in range(10) for t in range(1,51)}
v=json.loads((P/'verdict.json').read_text())
cache={}
def quantile(p,df):
    if (p,df) in cache:return cache[p,df]
    coefficient=math.gamma((df+1)/2)/(math.sqrt(df*math.pi)*math.gamma(df/2))
    low,high=0.,20.
    for _ in range(45):
        mid=(low+high)/2;xx=np.linspace(0,mid,20001);density=coefficient*(1+xx*xx/df)**(-(df+1)/2)
        integral=(xx[1]-xx[0])/3*(density[0]+density[-1]+4*density[1:-1:2].sum()+2*density[2:-1:2].sum())
        if .5+integral<p:low=mid
        else:high=mid
    cache[p,df]=(low+high)/2;return cache[p,df]
def value(a,s,t,k):return float(lookup[a,s,t][k])
def mean(values):return math.fsum(values)/len(values)
def ci(values,p):
    m=mean(values);sd=statistics.stdev(values) if len(set(values))>1 else 0.;w=quantile(p,9)*sd/math.sqrt(10)
    return m,m-w,m+w
upper=[];floors={};endpoint={};max_error=0
for s in range(10):
    gen=H.stream('rlc_labels',s);ff=[]
    for t in range(1,51):
        labels=torch.randint(10,(1200,),generator=gen,dtype=torch.int64);f=int(torch.bincount(labels,minlength=10).max())/1200
        assert all(value(a,s,t,'major_frac')==f for a in arms)
        if t>=31:ff.append(f)
    upper.append(mean(ff)+quantile(.95,19)*statistics.stdev(ff)/math.sqrt(20))
for a in arms:
    floors[a]=[mean([value(a,s,t,'online_acc') for t in range(31,51)])<=upper[s] for s in range(10)]
    endpoint[a]={key:[mean([value(a,s,t,field) for t in range(31,51)])-value(a,s,1,field) for s in range(10)] for key,field in [('E1','online_acc'),('E2','G2')]}
for a in arms:
    n=sum(floors[a]);assert n==v['arms'][a]['floor_count']
    for key in ['E1','E2']:
        differences=[x-y for x,y in zip(endpoint[a][key],endpoint['ref'][key])]
        qq=ci(differences,.9875 if a=='cap12' else .975)
        for field,want in zip(['mean','low','high'],qq):
            err=abs(v['arms'][a][key][field]-want);max_error=max(max_error,err);assert err<2e-10,(a,key,field,err)
    if a=='ref':continue
    if not v['applicable']:expected='NOT_REPRODUCED'
    elif n==10:expected='COLLAPSED'
    elif n:expected='SPLIT'
    elif v['arms'][a]['E1']['low']>0:expected='RESCUED' if v['arms'][a]['E2']['low']>0 else 'RESCUED_FUNCTION_ONLY'
    else:expected='ALIVE_UNRESOLVED'
    assert v['arms'][a]['label']==expected
applicable=sum(floors['ref'])>=9 and ci(endpoint['ref']['E2'],.975)[2]<0
assert applicable==v['applicable']
for a in arms:
    assess=all(value('ref',s,2,'online_acc')>value('ref',s,2,'major_frac') for s in range(10))
    n=sum(value(a,s,2,'online_acc')<(value('ref',s,2,'online_acc')+value('ref',s,2,'major_frac'))/2 for s in range(10))
    label='EARLY_FLAG_NOT_ASSESSABLE' if not assess else 'IMPAIRED_EARLY' if n>=6 else 'NO_EARLY_IMPAIRMENT'
    assert label==v['arms'][a]['early']
obj=dict(all_pass=True,rows=2500,paired_CI_max_error=max_error,independent_t_quantiles={str(k):q for k,q in cache.items()},label_generator_recreated=True,all_labels_agree=True,independent_auditor=False,
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['per_task.csv','verdict.json']})
(P/'readout_audit.json').write_text(json.dumps(obj,indent=2)+'\n');print(json.dumps(obj))

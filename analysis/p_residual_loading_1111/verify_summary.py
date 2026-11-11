import importlib.util
import copy
import numpy as np
import pandas as pd
from analysis.p_residual_loading_1111 import summarize as m
rows=[]
for source in ['S','P']:
 for seed in range(10):
  z=.2+seed*.001
  values={'ZC':z,'KC':z+.12+seed*.005,'KK':z+.15+seed*.005,'ZK':z+.01}
  for arm,u in values.items():
   r=dict(model='mlp',activation='ELU',t_end=10,history='S',source=source,tau=0,seed=seed,arm=arm,limit=100,complete=True,_synthetic=True,
          evaluation_grid=[0,25,100],curve=[dict(step=s,acc=1-u,ce=2*u) for s in [0,25,100]])
   rows.append(m.trajectory_metrics(r))
df=pd.DataFrame(rows);ct=m.paired_contrasts(df)
s=ct[(ct.source=='S')&(ct.contrast=='parameter_clear')&(ct.metric=='U')].iloc[0]
p=ct[(ct.source=='P')&(ct.contrast=='parameter_clear')&(ct.metric=='U')].iloc[0]
assert np.array_equal(s[['estimate','ci_low','ci_high']].to_numpy(),p[['estimate','ci_low','ci_high']].to_numpy())
r=m.guarded_ratios(df,ct).iloc[0]
assert r.estimable and r.estimate==r.ci_low==r.ci_high==1.
print('PASS paired source-identical resamples and ratio-of-means=1 on nonconstant seed effects')
bad=df[~((df.source=='P')&(df.seed==9))]
try:
 result=m.guarded_ratios(bad,m.paired_contrasts(bad))
except ValueError as e:
 print('PASS unmatched S/P seeds rejected:',e)
else:
 raise AssertionError('Unmatched source seed sets accepted')
# An endpoint-only hit must remain observable as a single-evaluation sustained event.
r=dict(model='mlp',activation='ELU',t_end=10,history='S',source='P',tau=25,seed=0,arm='KK',limit=100,complete=True,_synthetic=True,
       evaluation_grid=[25,26,100],curve=[dict(step=s,acc=a,ce=1.) for s,a in [(25,.95),(26,.89),(100,.9)]])
v=m.trajectory_metrics(r)
assert v['T90_first_elapsed']==0 and v['T90_sustained_elapsed']==75 and v['sustained_n_evaluations']==1 and v['sustained_observation_duration']==0
r['curve'][-1]['acc']=.899
v=m.trajectory_metrics(r)
assert not v['T90_first_censored'] and v['T90_sustained_censored'] and v['T90_sustained_elapsed'] is None
print('PASS separate first/sustained censor, tau elapsed offsets, endpoint-only support count')
from pathlib import Path
source=Path('/tmp/p_residual_validator_synthetic_source.txt');source.write_text('synthetic provenance fixture only\n')
prov={'script_sha256':m.digest(m.ROOT/'analysis/p_residual_loading_1111/run.py'),
      'spec_sha256':m.digest(m.ROOT/'specs/spec_p_residual_loading_1111.md'),
      'git_hash':m.MAIN_RUNNER_COMMIT,'helpers':{},'sources':{str(source):m.digest(source)},'args':{'engine':'compile'}}
records=[];by={}
for key in sorted(m.expected_design()):
 r=dict(zip(m.IDENT,key));r.update(complete=True,limit=4000 if r['model']=='mlp' else 1500,
       provenance=copy.deepcopy(prov),file='fixture_'+str(len(records))+'.json',reused=False)
 f={'body':str((key[:4],key[6],key[5],'body')),'body_adam':str((key[:4],key[6],key[5],'body_adam')),
    'head':str((key[:7],'head')),'head_adam':str((key[:7],'head_adam'))}
 p=dict(f)
 if r['arm'][0]=='Z':p['head']='zero-'+r['model']
 if r['arm'][1]=='C':p['head_adam']='empty-adam'
 r['fork_hashes'],r['post_hashes']=f,p
 r['curve']=[{'step':s,'acc':.5,'ce':1.} for s in m.required_grid(r)]
 records.append(r);by[key]=r
for r in records:
 key=m.record_key(r)
 if r['history']=='S' and r['source']=='P' and r['tau']==0 and r['arm']=='ZC':
  target=list(key);target[4]='S';ref=by[tuple(target)]
  r.update(reused=True,reuse_of=ref['file'])
 if r['history']=='S' and r['source']=='P' and r['tau']==25 and r['arm']=='KK':
  target=list(key);target[5]=0;ref=by[tuple(target)]
  r.update(reused=True,reuse_of=ref['file'])
assert len(records)==820 and sum(not r['reused'] for r in records)==690
m.validate_runner_records(records,False)
print('PASS: complete 820-logical/690-direct synthetic main fixture')
cases=[]
cases.append(('missing logical record',records[:-1]))
bad=copy.deepcopy(records);bad[0]['post_hashes']['body_adam']='changed';cases.append(('changed body Adam',bad))
bad=copy.deepcopy(records);r=next(r for r in bad if r['source']=='P' and r['tau']==0 and r['arm']=='ZC' and r['history']=='S');r['curve'][0]['acc']=.6;cases.append(('broken ZC alias',bad))
bad=copy.deepcopy(records);next(r for r in bad if r['model']=='vit')['provenance']['args']['engine']='eager';cases.append(('mixed main ViT engine',bad))
bad=copy.deepcopy(records);bad[0]['provenance']['sources'][str(source)]='mismatching source hash';cases.append(('mismatching source hash',bad))
for label,bad in cases:
 try:m.validate_runner_records(bad,False)
 except ValueError as e:print('PASS rejects',label,':',str(e)[:100])
 else:raise AssertionError('Validator accepted '+label)
vit=df[df.seed<5].copy();vit['model']='vit';vit['activation']='SL';vit['t_end']=100;vit['seed']+=10252010
both=pd.concat([df,vit],ignore_index=True)
v=m.verdict(m.paired_contrasts(both),True)
assert v['operational_commonality_verdict']=='supported' and v['both_strata_practical_above_0_01']
for effect,label in [(-.05,'rejected_for_primary_conditions'),(0.,'unresolved')]:
 a=both.copy();v_mask=(a.model=='vit')&(a.arm=='ZC')
 lookup=a[v_mask].set_index(['source','seed'])['U'].to_dict()
 for ix,r in a[(a.model=='vit')&(a.arm=='KC')].iterrows():a.loc[ix,'U']=lookup[(r.source,r.seed)]+effect
 result=m.verdict(m.paired_contrasts(a),True)
 assert result['operational_commonality_verdict']==label
assert m.verdict(m.paired_contrasts(both),False)['operational_commonality_verdict']=='not_evaluated'
print('PASS operational verdict: paired positive, negative primary, unresolved zero, incomplete not evaluated')

"""Pre-registered verdicts for width_depth_intervention_0909."""
import json,itertools
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_gradient_0908 as G
OUT=G.ROOT/'results/width_depth_intervention_0909'
ARMS=[(a,i) for a in ['LR','SNA'] for i in ['none','l2']]
def lab3(vals,lo,hi,low_label,high_label,mid_label):
 if all(abs(v)<lo for v in vals):return low_label
 if all(abs(v)>hi for v in vals):return high_label
 return mid_label
def main():
 d=pd.concat([pd.read_csv(OUT/f'{a}_{i}_s{s}_rows.csv') for a,i in ARMS for s in range(3)],ignore_index=True)
 ends=d[(d.step==625)|((d.task==20)&(d.phase=='post'))].copy();ends['t']=ends.task
 seedrows=[];armrows=[]
 for a,i in ARMS:
  per={}
  for s in range(3):
   e=ends[(ends.arm==a)&(ends.iv==i)&(ends.seed==s)]
   ref=e[e.intervention=='k1.0'].set_index('t').sort_index()
   tr=ref.index[ref.index>=21]
   k_inv=float(np.polyfit(ref.loc[tr,'sigma_inv'],ref.loc[tr,'zbar_inv'],1)[0])
   row=dict(arm=a,iv=i,seed=s,k_inv=k_inv,
    ref_sigma_inv_20=float(ref.loc[20,'sigma_inv']),ref_sigma_inv_40=float(ref.loc[40,'sigma_inv']),
    ref_zbar_inv_20=float(ref.loc[20,'zbar_inv']),ref_zbar_inv_40=float(ref.loc[40,'zbar_inv']))
   for name in ['k0.7','k1.4','bplus','bminus']:
    x=e[e.intervention==name].set_index('t').sort_index()
    dsig=(x.sigma_inv-ref.sigma_inv);dz=(x.zbar_inv-x.loc[20,'zbar_inv'])-(ref.zbar_inv-ref.loc[20,'zbar_inv']);dzraw=x.zbar_inv-ref.zbar_inv
    ds=dsig.loc[tr].to_numpy();dzz=dz.loc[tr].to_numpy();dzr=dzraw.loc[tr].to_numpy()
    if name.startswith('k'):
     P=float(dsig.loc[40]/dsig.loc[20]);C=float((dzz*ds).sum()/(ds*ds).sum())
     below=[int(t) for t in tr if abs(dsig.loc[t])<.25*abs(dsig.loc[20])]
     row.update({f'{name}_dsig20':float(dsig.loc[20]),f'{name}_dsig40':float(dsig.loc[40]),f'{name}_P':P,f'{name}_C':C,
                 f'{name}_C_over_k':C/k_inv,f'{name}_dz40':float(dz.loc[40]),f'{name}_first_restored':below[0] if below else None,
                 f'{name}_acc_gap40':float(x.loc[40,'acc']-ref.loc[40,'acc'])})
    else:
     db=float(dzraw.loc[20]);Q=float(dzraw.loc[40]/db);S=float((ds*dzr).sum()/(dzr*dzr).sum())
     row.update({f'{name}_db':db,f'{name}_Q':Q,f'{name}_S':S,f'{name}_S_times_k':S*k_inv,f'{name}_dsig40':float(dsig.loc[40]),
                 f'{name}_acc_gap40':float(x.loc[40,'acc']-ref.loc[40,'acc'])})
   per[s]=row;seedrows.append(row)
  V=dict(arm=a,iv=i,k_inv_seeds=';'.join(f'{per[s]["k_inv"]:+.3f}' for s in range(3)))
  wl={};cl={}
  for name in ['k0.7','k1.4']:
   Ps=[per[s][f'{name}_P'] for s in range(3)];Cs=[per[s][f'{name}_C_over_k'] for s in range(3)]
   wl[name]=lab3(Ps,.25,.75,'WIDTH_RESTORED','WIDTH_PERSISTS','WIDTH_PARTIAL')
   if wl[name]=='WIDTH_RESTORED':cl[name]='NOT_TESTABLE_WIDTH_RESTORED'
   elif all(c>=.5 for c in Cs):cl[name]='DEPTH_FOLLOWS_WIDTH'
   elif all(abs(c)<=.25 for c in Cs):cl[name]='DEPTH_IGNORES_WIDTH'
   else:cl[name]='DEPTH_PARTIAL'
   V[f'{name}_P_seeds']=';'.join(f'{v:+.2f}' for v in Ps);V[f'{name}_C_over_k_seeds']=';'.join(f'{v:+.2f}' for v in Cs)
   V[f'{name}_width']=wl[name];V[f'{name}_coupling']=cl[name]
   V[f'{name}_acc_flag']=int(any(per[s][f'{name}_acc_gap40']<-.02 for s in range(3)))
  V['width_verdict']=wl['k0.7'] if wl['k0.7']==wl['k1.4'] else 'ASYMMETRIC:'+wl['k0.7']+'/'+wl['k1.4']
  V['coupling_verdict']=cl['k0.7'] if cl['k0.7']==cl['k1.4'] else 'ASYMMETRIC:'+cl['k0.7']+'/'+cl['k1.4']
  ql={}
  for name in ['bplus','bminus']:
   Qs=[per[s][f'{name}_Q'] for s in range(3)];ql[name]=lab3(Qs,.25,.75,'DEPTH_RESTORED','DEPTH_FREE','DEPTH_PARTIAL')
   V[f'{name}_Q_seeds']=';'.join(f'{v:+.2f}' for v in Qs);V[f'{name}_S_times_k_seeds']=';'.join(f'{per[s][f"{name}_S_times_k"]:+.2f}' for s in range(3))
  V['depth_verdict']=ql['bplus'] if ql['bplus']==ql['bminus'] else 'ASYMMETRIC:'+ql['bplus']+'/'+ql['bminus']
  armrows.append(V)
 pd.DataFrame(seedrows).to_csv(OUT/'seed_verdict.csv',index=False)
 ver=pd.DataFrame(armrows);ver.to_csv(OUT/'verdict.csv',index=False)
 # validation table
 vr=[]
 for a,i in ARMS:
  for s in range(3):
   m=json.loads((OUT/f'{a}_{i}_s{s}_provenance.json').read_text());c=m['checks']
   vr.append(dict(arm=a,iv=i,seed=s,exact_replay=c['k1.0']['exact_replay'],exact_replay_mutctl=c['k1.0'].get('exact_replay_mutctl'),
    measure_mutctl=c['k1.0'].get('measure_mutctl'),sigma2_identity=max(c['k0.7']['sigma2_identity'],c['k1.4']['sigma2_identity']),
    sigma2_mutctl=min(c['k0.7']['sigma2_identity_mutctl'],c['k1.4']['sigma2_identity_mutctl']),
    rowsum_rel=max(c['k0.7']['rowsum_rel'],c['k1.4']['rowsum_rel']),bias_shift_z=max(c['bplus']['bias_shift_z'],c['bminus']['bias_shift_z']),
    bias_shift_mutctl=min(c['bplus']['bias_shift_mutctl'],c['bminus']['bias_shift_mutctl']),sigma_unchanged=max(c['bplus']['sigma_unchanged'],c['bminus']['sigma_unchanged']),
    wall=round(m['wall_seconds'],1)))
 pd.DataFrame(vr).to_csv(OUT/'validation.csv',index=False)
 # figures: time course of dsigma and dz per arm, seed median
 fig,ax=plt.subplots(2,4,figsize=(16,7),sharex=True)
 for k,(a,i) in enumerate(ARMS):
  for name,col in [('k0.7','C0'),('k1.4','C3'),('bplus','C2'),('bminus','C4')]:
   ds_=[];dz_=[]
   for s in range(3):
    e=ends[(ends.arm==a)&(ends.iv==i)&(ends.seed==s)];ref=e[e.intervention=='k1.0'].set_index('t').sort_index();x=e[e.intervention==name].set_index('t').sort_index()
    ds_.append((x.sigma_inv-ref.sigma_inv).to_numpy());dz_.append((x.zbar_inv-ref.zbar_inv).to_numpy())
   t=np.array(sorted(e.t.unique()))
   ax[0][k].plot(t,np.median(ds_,0),color=col,label=name);ax[1][k].plot(t,np.median(dz_,0),color=col,label=name)
  ax[0][k].set_title(f'{a} {i}');ax[0][k].axhline(0,color='k',lw=.5);ax[1][k].axhline(0,color='k',lw=.5)
  ax[1][k].set_xlabel('task');ax[0][0].set_ylabel('sigma_inv - ref');ax[1][0].set_ylabel('zbar_inv - ref (raw)')
  if k==0:ax[0][k].legend(fontsize=7)
 fig.suptitle('width_depth_intervention_0909: intervention minus reference, task ends (seed median)');fig.tight_layout()
 for e in ('png','pdf'):fig.savefig(OUT/f'course.{e}',dpi=140)
 print(ver[['arm','iv','width_verdict','coupling_verdict','depth_verdict','k_inv_seeds']].to_string(index=False))
if __name__=='__main__':main()

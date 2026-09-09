"""Pre-registered verdicts for perm_cycle_0909."""
import json
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_gradient_0908 as G
OUT=G.ROOT/'results/perm_cycle_0909'
ARMS=[('LR','none'),('SNA','none'),('LR','l2')];CONDS={'FRESH':0,'CYCLE1':1,'CYCLE2':2,'CYCLE5':5}
def main():
 d=pd.concat([pd.read_csv(OUT/f'{a}_{i}_s{s}_rows.csv') for a,i in ARMS for s in range(3)],ignore_index=True)
 seedrows=[];armrows=[]
 for a,i in ARMS:
  per={}
  for s in range(3):
   e=d[(d.arm==a)&(d.iv==i)&(d.seed==s)];row=dict(arm=a,iv=i,seed=s)
   g={}
   for c,K in CONDS.items():
    x=e[e.cond==c].set_index('task').sort_index()
    lo=21+K;tr=x.index[x.index>=lo]
    g[c]=float((x.loc[tr,'dnorm2']+x.loc[tr,'align']).mean())   # = mean Δ‖W̃‖² per task over tasks K+1..40
    row[f'{c}_g']=g[c];row[f'{c}_cnorm2_40']=float(x.loc[40,'cnorm2']);row[f'{c}_cos']=float(x.loc[tr,'cos'].mean())
    row[f'{c}_dz_26_40']=float(x.loc[40,'zbar_inv']-x.loc[25,'zbar_inv']);row[f'{c}_dsig_26_40']=float(x.loc[40,'sigma_inv']-x.loc[25,'sigma_inv'])
   for c in ['CYCLE1','CYCLE2','CYCLE5']:row[f'{c}_g_ratio']=g[c]/g['FRESH']
   m=np.load(OUT/f'{a}_{i}_s{s}_cos.npz')
   for c in ['CYCLE2','CYCLE5']:
    C=m[c+'_cos'];pt=m[c+'_permtask'];n=len(pt);same=[];diff=[]
    for p_ in range(n):
     for q_ in range(p_+1,n):(same if pt[p_]==pt[q_] else diff).append(C[p_,q_])
    row[f'{c}_a_same']=float(np.mean(same));row[f'{c}_a_diff']=float(np.mean(diff))
   C=m['FRESH_cos'];n=C.shape[0];row['FRESH_a_diff']=float(np.mean([C[p_,q_] for p_ in range(n) for q_ in range(p_+1,n)]))
   row['CYCLE5_dz_ratio']=row['CYCLE5_dz_26_40']/row['FRESH_dz_26_40'] if row['FRESH_dz_26_40']!=0 else np.nan
   per[s]=row;seedrows.append(row)
  V=dict(arm=a,iv=i)
  ratios={c:[per[s][f'{c}_g_ratio'] for s in range(3)] for c in ['CYCLE1','CYCLE2','CYCLE5']}
  V['g_FRESH_seeds']=';'.join(f'{per[s]["FRESH_g"]:+.3f}' for s in range(3))
  for c in ratios:V[f'{c}_g_ratio_seeds']=';'.join(f'{v:+.2f}' for v in ratios[c])
  if all(v<.25 for c in ratios for v in ratios[c]):V['growth']='GROWTH_IS_PERMUTATION_DRIVEN'
  elif any(v>.75 for c in ratios for v in ratios[c]):V['growth']='GROWTH_PERSISTS_UNDER_CYCLING'
  else:V['growth']='PARTIAL'
  same=[per[s][f'{c}_a_same'] for s in range(3) for c in ['CYCLE2','CYCLE5']];diff=[per[s][f'{c}_a_diff'] for s in range(3) for c in ['CYCLE2','CYCLE5']]
  V['a_same_seeds']=';'.join(f'{per[s]["CYCLE5_a_same"]:+.3f}' for s in range(3));V['a_diff_seeds']=';'.join(f'{per[s]["CYCLE5_a_diff"]:+.3f}' for s in range(3))
  V['FRESH_a_diff_seeds']=';'.join(f'{per[s]["FRESH_a_diff"]:+.3f}' for s in range(3))
  V['alignment']='SAME_PERM_ALIGNED' if all(v>.2 for v in same) and all(abs(v)<.05 for v in diff) else ('NO_ALIGNMENT' if all(v<.05 for v in same) else 'PARTIAL')
  dz=[per[s]['CYCLE5_dz_ratio'] for s in range(3)];V['dz_ratio_seeds']=';'.join(f'{v:+.2f}' for v in dz)
  V['depth']='DEPTH_STOPS_WITH_WIDTH' if all(abs(v)<.25 for v in dz) else ('DEPTH_KEEPS_SINKING' if all(v>.75 for v in dz) else 'PARTIAL')
  if i=='l2':V['wd']='WD_PINS_REGARDLESS' if all(abs(per[s][f'{c}_g'])<.05 for s in range(3) for c in CONDS) else 'WD_DOES_NOT_PIN'
  armrows.append(V)
 pd.DataFrame(seedrows).to_csv(OUT/'seed_verdict.csv',index=False);ver=pd.DataFrame(armrows);ver.to_csv(OUT/'verdict.csv',index=False)
 vr=[]
 for a,i in ARMS:
  for s in range(3):
   m=json.loads((OUT/f'{a}_{i}_s{s}_provenance.json').read_text());c=m['checks']
   vr.append(dict(arm=a,iv=i,seed=s,exact_replay=c['FRESH']['exact_replay'],exact_replay_mutctl=c['FRESH'].get('exact_replay_mutctl'),measure_mutctl=c['FRESH'].get('measure_mutctl'),
    identity=max(c[k]['identity'] for k in c),identity_mutctl=min(c[k]['identity_mutctl'] for k in c),wall=round(m['wall_seconds'],1)))
 pd.DataFrame(vr).to_csv(OUT/'validation.csv',index=False)
 fig,ax=plt.subplots(2,3,figsize=(14,7),sharex=True)
 for k,(a,i) in enumerate(ARMS):
  for c,col in [('FRESH','k'),('CYCLE1','C0'),('CYCLE2','C2'),('CYCLE5','C3')]:
   xs=[];zs=[]
   for s in range(3):
    x=d[(d.arm==a)&(d.iv==i)&(d.seed==s)&(d.cond==c)].sort_values('task');xs.append(x.cnorm2.to_numpy());zs.append(x.zbar_inv.to_numpy())
   t=x.task.to_numpy();ax[0][k].plot(t,np.median(xs,0),color=col,label=c);ax[1][k].plot(t,np.median(zs,0),color=col,label=c)
  ax[0][k].set_title(f'{a} {i}');ax[0][k].set_ylabel('||W~||^2 (unit mean)');ax[1][k].set_ylabel('zbar_inv');ax[1][k].set_xlabel('task')
  if k==0:ax[0][k].legend(fontsize=8)
 fig.suptitle('perm_cycle_0909: centered W1 norm and permutation-invariant zbar under cycled permutations (seed median)');fig.tight_layout()
 for e in ('png','pdf'):fig.savefig(OUT/f'course.{e}',dpi=140)
 print(ver.to_string(index=False))
if __name__=='__main__':main()

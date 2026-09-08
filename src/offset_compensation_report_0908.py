
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'results/offset_compensation_0908'
rs=list(csv.DictReader((P/'seed_metrics.csv').open()))
ors=list(csv.DictReader((P/'oracle_seed_metrics.csv').open()))
def write(n,rows):
 with (P/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
seedrows=[];summary=[]
metrics=['residual_ratio','mean_residual_ratio','net_force_ratio','negative_force_rms_ratio','force_rms_ratio','down_fraction','force_mean','self_mean','rest_mean']
for c in ['all','K','Kc','KU','KUc']:
 for seed in range(10):
  a=[r for r in rs if r['condition']==c and int(r['seed'])==seed and int(r['update'])>=9500]
  valid=len(a)==6 and all(r['valid']=='True' for r in a)
  comp=valid and all(float(r['residual_ratio'])<=.1 for r in a)
  sr={'condition':c,'seed':seed,'valid':valid,'compensated':comp}
  for m in metrics:sr[m]=float(np.mean([float(r[m]) for r in a])) if valid else None
  seedrows.append(sr)
 a=[r for r in seedrows if r['condition']==c and r['valid']]
 su={'condition':c,'n_valid':len(a),'n_compensated':sum(r['compensated'] for r in a)}
 for m in metrics:
  vals=[r[m] for r in a]
  for name,v in zip(['q25','median','q75'],np.percentile(vals,[25,50,75])):su[m+'_'+name]=float(v)
 summary.append(su)
write('verdict.csv',summary);write('seed_endpoints.csv',seedrows)
a=[r for r in seedrows if r['condition']=='K' and r['compensated']]
verdict={'n_compensated':len(a),'n_total':10,'primary':'K','window':[9500,10000],'verdict':'NOT_ENOUGH_COMPENSATED_SEEDS'}
if len(a)>=6:
 vals=np.array([r['net_force_ratio'] for r in a]);rng=np.random.default_rng(202609081020)
 lo,hi=np.percentile(np.median(rng.choice(vals,(5000,len(vals))),axis=1),[2.5,97.5])
 label='INCONCLUSIVE'
 if hi<-.1:label='DOWNWARD_DRIVE_REMAINS_AFTER_COMPENSATION'
 elif lo>=-.1 and hi<=.1:label='NO_MATERIAL_NET_DRIFT_AFTER_COMPENSATION'
 elif lo>.1:label='UPWARD_DRIVE_AFTER_COMPENSATION'
 verdict.update(verdict=label,median=float(np.median(vals)),ci95=[float(lo),float(hi)])
(P/'verdict.json').write_text(json.dumps(verdict,indent=2))
os=[]
for c in ['mean_only','oracle_K','oracle_Kc','oracle_KUc']:
 a=[r for r in ors if r['condition']==c]
 su={'condition':c,'n':len(a)}
 for m in ['rank','coefficient_change_norm','residual_ratio','mean_residual_ratio','net_force_ratio','negative_force_rms_ratio','force_rms_ratio','self_force_mean','rest_force_mean']:
  vals=[float(r[m]) for r in a if r[m]!='']
  su[m+'_median']=float(np.median(vals))
 os.append(su)
write('oracle_verdict.csv',os)
with np.load(P/'trajectories.npz') as d:
 # Group contributions remain separately auditable; summarize changes per seed.
 bookkeeping=[]
 for ci,c in enumerate(d['conditions']):
  for si,seed in enumerate(d['seeds']):
   su={'condition':str(c),'seed':int(seed)}
   for group in ['L','K','U']:
    diff=d['output_'+group][-1,ci,si]-d['output_'+group][0,ci,si]
    su['output_change_'+group+'_rms']=float(np.sqrt(np.mean(diff**2)))
    mask=d['group_'+group][si]
    su['position_change_'+group+'_rms']=float(np.sqrt(np.mean((d['zmean'][-1,ci,si,mask]-d['zmean'][0,ci,si,mask])**2)))
   su['c_change']=float(d['c'][-1,ci,si]-d['c'][0,ci,si])
   bookkeeping.append(su)
 write('output_bookkeeping.csv',bookkeeping)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
for c,col in zip(['all','K','Kc','KU','KUc'],['#30343b','#cc6333','#e2ac42','#2266aa','#6a9d68']):
 for ax,metric,title in zip(axs,['residual_ratio','force_rms_ratio','negative_force_rms_ratio'],
                          ['Prediction error remaining','Leak: full mean-position force RMS','Leak: downward force RMS']):
  steps=sorted(set(int(r['update']) for r in rs));vv=[]
  for st in steps:
   vals=[float(r[metric]) for r in rs if r['condition']==c and int(r['update'])==st and r['valid']=='True']
   vv.append(np.percentile(vals,[25,50,75]))
  vv=np.array(vv)
  ax.plot(steps,vv[:,1],label=c,color=col);ax.fill_between(steps,vv[:,0],vv[:,2],color=col,alpha=.08)
  ax.set_xscale('symlog',linthresh=1);ax.set_yscale('log');ax.set_xlabel('Updates');ax.set_title(title)
  ax.grid(alpha=.15);ax.set_ylabel('Ratio to initial RMS')
axs[0].axhline(.1,color='#888',ls=':');axs[0].legend(fontsize=9)
fig.suptitle('c = -2, saved 40M state: fixed-task compensation and virtual leak gradients\nMedian and IQR across 10 seeds; L stays frozen except in all.',fontsize=12)
fig.savefig(P/'compensation.png',dpi=150);fig.savefig(P/'compensation.pdf');plt.close(fig)
print('PRIMARY',verdict)
for row in summary:print({k:row[k] for k in ['condition','n_compensated','residual_ratio_median','force_rms_ratio_median','negative_force_rms_ratio_median','net_force_ratio_median','self_mean_median','rest_mean_median']})
for row in os:print('ORACLE',row)

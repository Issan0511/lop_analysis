"""Aggregate by boundary then seed. Three seeds: descriptive medians/ranges, no CI."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_gradient_0908 as G
OUT=G.OUT
ARMS=[('SNA','none'),('SNA','l2'),('LR','none'),('LR','l2')]
ADDITIVE=['actual','W','b','star','alignment','history','data','l2','positive','negative',
'history_W','history_b','data_W','data_b','l2_W','l2_b','sgd_data','sgd_l2','sgd_mean','sgd_cov',
'pre_history','pre_history_W','pre_history_b','within_history','rounding']
LEVEL=['gate_mean','gate_rms','upstream_rms','local_delta_rms','positive_fraction','gW_norm','gb']
def meanmasked(a,masks):
 # a=(boundary,step,unit). Equal boundary weights after per-group unit mean.
 n=masks.sum(1);return np.nanmean(np.where(n[:,None]>0,(a*masks[:,None,:]).sum(2)/np.maximum(n[:,None],1),np.nan),axis=0)
def main():
 rows=[];eps=[];fits=[];signals=[];validation=[]
 for arm,iv in ARMS:
  for seed in range(3):
   prefix=f'{arm}_{iv}_s{seed}'
   data=dict(np.load(OUT/'raw'/(prefix+'.npz')))
   memory=dict(np.load(OUT/'raw'/(prefix+'_memory.npz')))
   data.update(memory);data['pre_history']=data['pre_history_W']+data['pre_history_b']
   data['within_history']=data['history']-data['pre_history']
   f=pd.read_csv(OUT/(prefix+'_credits.csv'))
   for group in ['ALL']+G.B.GROUPS:
    masks=np.ones((19,100),bool) if group=='ALL' else data['masks'][:,G.B.GROUPS.index(group)]
    count=masks.sum(1)
    series={k:meanmasked(np.cumsum(data[k],axis=1),masks) for k in ADDITIVE}
    series.update({k:meanmasked(data[k],masks) for k in LEVEL})
    valid=(abs(data['sgd_data'])>1e-8)&(abs(data['actual'])>1e-8)&masks[:,None,:]
    opposed=(data['sgd_data']*data['actual']<0)&valid
    for t in range(100):
     row=dict(arm=arm,iv=iv,seed=seed,group=group,step=t+1,n_units=float(count.mean()))
     row.update({k:float(v[t]) for k,v in series.items()})
     rows.append(row)
    for t in [20,50,100]:
     row=dict(arm=arm,iv=iv,seed=seed,group=group,step=t)
     row.update({k:float(v[t-1]) for k,v in series.items()})
     row['opposed_fraction']=float(opposed[:,:t].sum()/valid[:,:t].sum()) if valid[:,:t].sum() else np.nan
     for comp in ['W','b','data','history','l2','pre_history','within_history']:
      cum=data[comp][:,:t].sum(1)
      row[comp+'_down_mass']=float((-np.minimum(cum,0)*masks).sum(1).mean()/100)
      row[comp+'_up_mass']=float((np.maximum(cum,0)*masks).sum(1).mean()/100)
     denom=row['W_down_mass']+row['b_down_mass']
     row['bias_down_share']=row['b_down_mass']/denom if denom>1e-10 else np.nan
     eps.append(row)
    if group!='ALL':
     for step in G.CHECK_STEPS:
      ff=f[f.step==step]
      credit=ff['credit_'+group].to_numpy()
      fits.append(dict(arm=arm,iv=iv,seed=seed,group=group,step=step,
       credit=float(credit.mean()),credit_per_unit=float(np.nanmean(np.where(count>0,credit/np.maximum(count,1),np.nan))),
       CE_improvement=float(ff.CE_improvement.mean()),CE_after=float(ff.CE_after.mean())))
    # Pooled sample/unit summaries within each boundary, then equal boundary averaging.
    for lo,hi in [(1,5),(6,20),(21,50)]:
     exz=data['example_z'][:,lo-1:hi]
     up=data['example_dL_da'][:,lo-1:hi]*16
     delta=data['example_dL_dz'][:,lo-1:hi]*16
     gate=data['example_gate'][:,lo-1:hi]
     for side in ['positive','negative']:
      mask=(exz>0 if side=='positive' else exz<=0)&masks[:,None,None,:]
      dims=(1,2,3);den=mask.sum(dims)
      def avg(v):return np.nanmean(np.where(den>0,(v*mask).sum(dims)/np.maximum(den,1),np.nan))
      signals.append(dict(arm=arm,iv=iv,seed=seed,group=group,window=f'{lo}-{hi}',side=side,
       gate_mean=float(avg(gate)),upstream_mean_abs=float(avg(abs(up))),
       delta_mean_abs=float(avg(abs(delta))),delta_signed_mean=float(avg(delta)),
       samples_mean=float(den.mean())))
   for step in G.CHECK_STEPS:
    ff=f[f.step==step]
    fits.append(dict(arm=arm,iv=iv,seed=seed,group='downstream',step=step,
     credit=float(ff.credit_downstream.mean()),credit_per_unit=np.nan,
     CE_improvement=float(ff.CE_improvement.mean()),CE_after=float(ff.CE_after.mean())))
   meta=json.loads((OUT/(prefix+'_provenance.json')).read_text())
   for check in meta['checks']:validation.append(dict(arm=arm,iv=iv,seed=seed,**check))
 df=pd.DataFrame(rows);ep=pd.DataFrame(eps);fit=pd.DataFrame(fits);sig=pd.DataFrame(signals)
 for name,frame in [('seed_timeseries',df),('seed_verdict',ep),('seed_credits',fit),('seed_signals',sig),('validation',pd.DataFrame(validation))]:
  frame.to_csv(OUT/(name+'.csv'),index=False,lineterminator='\n')
 idcols=['arm','iv','group','step']
 vv=ep.groupby(idcols).median(numeric_only=True).drop(columns=['seed']).reset_index()
 vv.to_csv(OUT/'verdict.csv',index=False,lineterminator='\n')
 for name,frame,ids in [('credit_verdict',fit,idcols),('signal_verdict',sig,['arm','iv','group','window','side'])]:
  frame.groupby(ids).median(numeric_only=True).drop(columns=['seed']).reset_index().to_csv(OUT/(name+'.csv'),index=False,lineterminator='\n')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 colors={'actual':'black','W':'#0072B2','b':'#E69F00','data':'#009E73','l2':'#CC79A7',
 'pre_history':'#D55E00','within_history':'#0072B2','D_only':'#D55E00','N_only':'#0072B2','downstream':'#777777',
 'positive':'#E69F00','negative':'#0072B2'}
 def line(ax,frame,key,label,color):
  piv=frame.pivot(index='seed',columns='step',values=key)
  x=piv.columns.to_numpy();a=piv.to_numpy()
  ax.plot(x,np.median(a,axis=0),label=label,color=color,lw=1.8)
  ax.fill_between(x,np.min(a,axis=0),np.max(a,axis=0),color=color,alpha=.10)
 def axesframe(ax,title,ylabel):
  ax.axhline(0,color='#aaa',lw=.7);ax.axvline(20,color='#aaa',lw=.7,ls=':')
  ax.set_title(title);ax.set_xlabel('Updates after switch');ax.set_ylabel(ylabel)
 fig,axs=plt.subplots(2,4,figsize=(17,7),layout='constrained')
 for col,(arm,iv) in enumerate(ARMS):
  f=df[(df.arm==arm)&(df.iv==iv)&(df.group=='ALL')&(df.step<=50)]
  title=arm+(' + L2' if iv=='l2' else '')
  for key in ['actual','W','b']:line(axs[0,col],f,key,{'actual':'Actual','W':'Weight path','b':'Bias path'}[key],colors[key])
  for key in ['actual','data','l2','pre_history','within_history']:
   line(axs[1,col],f,key,{'actual':'Actual','data':'Current CE','l2':'Current L2','pre_history':'Pre-switch m','within_history':'Post-switch past m'}[key],colors[key])
  axesframe(axs[0,col],title,'Cumulative mean z change')
  axesframe(axs[1,col],title,'Adam numerator contributions')
 axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8)
 fig.suptitle('Early descent: parameter paths and Adam contributions\nCPU replay; 19 boundaries per seed; median and range of 3 seeds; step 0 = after permutation',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(OUT/('gradient_accounting.'+ext),dpi=170)
 plt.close(fig)
 fig,axs=plt.subplots(2,4,figsize=(17,7),layout='constrained')
 for col,(arm,iv) in enumerate(ARMS):
  title=arm+(' + L2' if iv=='l2' else '')
  for group in ['D_only','N_only']:
   f=df[(df.arm==arm)&(df.iv==iv)&(df.group==group)&(df.step<=50)]
   line(axs[0,col],f,'actual',group,colors[group])
  for group in ['D_only','N_only','downstream']:
   f=fit[(fit.arm==arm)&(fit.iv==iv)&(fit.group==group)&(fit.step<=50)]
   line(axs[1,col],f,'credit',group,colors[group])
  axesframe(axs[0,col],title,'Cumulative group mean z change')
  axesframe(axs[1,col],title,'CE improvement credit (cumulative)')
 axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8)
 fig.suptitle('Who moves, and who repairs the output?\nGroups fixed at switch; N = relative nearest 25, not an absolute zero band; credits are Shapley bookkeeping',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(OUT/('groups_and_fit.'+ext),dpi=170)
 plt.close(fig)
 fig,axs=plt.subplots(2,4,figsize=(17,7),layout='constrained')
 for col,(arm,iv) in enumerate(ARMS):
  for row,group in enumerate(['ALL','N_only']):
   f=df[(df.arm==arm)&(df.iv==iv)&(df.group==group)&(df.step<=50)]
   for key in ['data','positive','negative']:line(axs[row,col],f,key,key,colors.get(key,'black'))
   axesframe(axs[row,col],arm+(' + L2' if iv=='l2' else '')+' / '+group,'Current CE update contribution')
 axs[0,0].legend(fontsize=8)
 fig.suptitle('Which input region generates the current CE gradient?\nPositive/negative = per-example first-layer z sign; shared actual Adam denominator, no causal removal',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(OUT/('input_regions.'+ext),dpi=170)
 plt.close(fig)
 # Summary is the canonical transcription source; supplementary tables retained as CSV.
 chunks=['# Early boundary gradient audit\n','CPU exact continuation; tasks101..119, 3 seeds. Median of seed means. Posthoc investigation with definitions frozen before measurement.',
 '## All units, cumulative first20\n',vv[(vv.group=='ALL')&(vv.step==20)][['arm','iv','actual','W','b','star','alignment','data','l2','pre_history','within_history','opposed_fraction','bias_down_share']].to_string(index=False),
 '## Group cumulative first20\n',vv[(vv.group.isin(['D_only','N_only']))&(vv.step==20)][['arm','iv','group','actual','W','b','data','l2','pre_history','within_history','opposed_fraction']].to_string(index=False),
 '## Output credits first20\n',fit[fit.step==20].groupby(['arm','iv','group'])[['credit','credit_per_unit','CE_improvement']].median().to_string(),
 '## Validation maxima\n',pd.DataFrame(validation).drop(columns=['arm','iv','seed','task']).max().to_string(),
 '## Limits\n','Numerator decomposition uses the actual denominator: L2 removal and Adam reset counterfactuals are not tested. Pre-switch m split is a posthoc supplement. History in the main spec includes earlier updates within the new task. CE multi-layer training has no scalar-MSE self/rest attribution here. Relative N is not a physical leak/positive classification; per-input signs are recorded separately. Separate seed medians need not add exactly. No claim of original CUDA gradient identity or long-term causal ratchet.']
 (OUT/'summary.md').write_text('\n\n'.join(chunks))
 print('\n\n'.join(chunks),flush=True)

 # Posthoc diagnostic: five-step average rates. Kept distinct from main prerecorded endpoints.
 rates=[]
 for (arm,iv,seed),frame in df[df.group=='ALL'].groupby(['arm','iv','seed']):
  frame=frame.sort_values('step')
  for lo,hi in [(i,i+4) for i in range(1,47,5)]:
   row=dict(arm=arm,iv=iv,seed=seed,window=f'{lo}-{hi}',step=(lo+hi)/2)
   for k in ['actual','data','history','l2','pre_history','within_history','sgd_data','positive','negative']:
    row[k]=(frame.iloc[hi-1][k]-(frame.iloc[lo-2][k] if lo>1 else 0))/(hi-lo+1)
   rates.append(row)
 rates=pd.DataFrame(rates)
 rates.to_csv(OUT/'posthoc_window_rates.csv',index=False,lineterminator='\n')
 rv=rates.groupby(['arm','iv','window']).median(numeric_only=True).drop(columns='seed')
 fig,axs=plt.subplots(1,4,figsize=(17,4),layout='constrained')
 for ax,(arm,iv) in zip(axs,ARMS):
  f=rates[(rates.arm==arm)&(rates.iv==iv)]
  for key in ['actual','data','within_history']:
   line(ax,f,key,{'actual':'Actual','data':'Current CE','within_history':'Post-switch past m'}[key],colors[key])
  axesframe(ax,arm+(' + L2' if iv=='l2' else ''),'Mean z change per update')
 axs[0].legend(fontsize=8)
 fig.suptitle('The gradient can turn upward before motion does\nPosthoc five-step averages; same actual Adam denominator for all numerator contributions',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(OUT/('gradient_direction.'+ext),dpi=170)
 plt.close(fig)
 with (OUT/'summary.md').open('a') as file:
  file.write('\n\n## Posthoc five-step rates (per update)\n\n'+rv.to_string()
   +'\n\n## Signals first1-5 (seed median)\n\n'+sig[(sig.group=='ALL')&(sig.window=='1-5')].groupby(['arm','iv','side']).median(numeric_only=True).drop(columns='seed').to_string())

 summary_path=OUT/'summary.md'
 summary_path.write_text('\n'.join(line.rstrip() for line in summary_path.read_text().splitlines())+'\n')

if __name__=='__main__':main()

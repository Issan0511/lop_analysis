"""Descriptive report: task21..40, first update and fixed early-update windows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_tasks20_40_transport_0909 as T
OUT=T.ROOT/'results/boundary_tasks20_40_0909';RAW=OUT/'transport/raw'
WINDOWS=[(1,1),(2,5),(6,10),(11,20),(1,20)]
COMP=['pos','neg','l2','pre','current','current_l2','history','current_pos','current_neg','sgd_CE','sgd_L2']
def average_mask(a,m):
 n=m.sum(axis=(1,2,3))
 return np.divide((a*m).sum(axis=(1,2,3)),n,out=np.full(n.shape,np.nan),where=n>0)
def md(f):
 def fmt(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
 return '\n'.join(['| '+' | '.join(f.columns)+' |','|'+'|'.join(['---']*len(f.columns))+'|']+['| '+' | '.join(fmt(x) for x in r)+' |' for _,r in f.iterrows()])
def compute(raw,lo,hi):
 sl=slice(lo-1,hi);a=raw['actual_mean'][:,sl];down=a<-1e-8;mass=np.maximum(-a,0).sum((1,2))
 d={}
 for key in ['actual']+COMP:
  for suffix in ['mean','W','b','star']:
   k=key+'_'+suffix;vv=raw[k][:,sl]
   d[k]=vv.sum(1).mean(1)
   if suffix=='mean':
    d[key+'_down_share']=np.divide((-vv*down).sum((1,2)),mass,out=np.full(mass.shape,np.nan),where=mass>1e-12)
 d['down_mass']=mass/a.shape[2]
 d['alignment']=d['actual_mean']-d['actual_star']
 d['actual_down_W_share']=np.divide((-raw['actual_W'][:,sl]*down).sum((1,2)),mass,out=np.full(mass.shape,np.nan),where=mass>1e-12)
 d['actual_down_b_share']=np.divide((-raw['actual_b'][:,sl]*down).sum((1,2)),mass,out=np.full(mass.shape,np.nan),where=mass>1e-12)
 z=raw['pair_z'][:,sl];local=raw['pair_local_direction'][:,sl];actual=raw['pair_actual'][:,sl]
 for label,mask in [('positive',z>0),('negative',z<=0)]:
  d[label+'_local']=average_mask(local,mask)
  d[label+'_upstream']=average_mask(raw['pair_upstream_direction'][:,sl],mask)
  d[label+'_gate']=average_mask(raw['pair_gate'][:,sl],mask)
  d[label+'_actual']=average_mask(actual,mask)
  d[label+'_current']=average_mask(raw['pair_current'][:,sl],mask)
  d[label+'_own']=average_mask(raw['pair_own'][:,sl],mask)
  d[label+'_others']=average_mask(raw['pair_others'][:,sl],mask)
  d[label+'_localup_fraction']=average_mask((local>1e-8).astype(float),mask)
  d[label+'_actualdown_fraction']=average_mask((actual<-1e-8).astype(float),mask)
  d[label+'_up_actualdown_fraction']=average_mask((actual<-1e-8).astype(float),mask&(local>1e-8))
 d['positive_fraction']=(z>0).mean((1,2,3))
 for side,si in [('fixed_positive',0),('fixed_negative',1)]:
  n=raw['receiver_counts'][:,si]
  for key in ['actual','pos','neg','l2','pre','rounding']:
   cum=np.nansum(raw['receiver_'+key+'_fixed'][:,sl,si],axis=1)
   d[side+'_'+key]=(cum*n).sum(1)/n.sum(1)
 return d
def main():
 allrows=[];steps=[];checks=[];position=[]
 for arm in ['LR','SNA']:
  for iv in ['none','l2']:
   for seed in range(3):
    prefix=f'{arm}_{iv}_s{seed}';raw=dict(np.load(RAW/(prefix+'.npz')))
    for lo,hi in WINDOWS:
     d=compute(raw,lo,hi)
     for j,task in enumerate(raw['task']):
      allrows.append(dict(arm=arm,iv=iv,seed=seed,task=task,window=f'{lo}-{hi}',**{k:v[j] for k,v in d.items()}))
    for step in range(1,21):
     d=compute(raw,step,step)
     steps.append(dict(arm=arm,iv=iv,seed=seed,step=step,**{k:float(np.nanmean(v)) for k,v in d.items()}))
    meta=json.loads((OUT/'transport'/(prefix+'_provenance.json')).read_text())
    checks.extend([dict(arm=arm,iv=iv,seed=seed,**x) for x in meta['checks']])
    p=pd.read_csv(OUT/(prefix+'_positions.csv'))
    for task,g in p.groupby('task'):
     g=g.set_index('step')
     for step in [20,300,625]:
      position.append(dict(arm=arm,iv=iv,seed=seed,task=task,step=step,
       jump=g.loc[0,'mean']-g.loc[0,'pre_switch_mean'],
       mean=g.loc[step,'mean']-g.loc[0,'mean'],star=g.loc[step,'star']-g.loc[0,'star'],
       since20_mean=g.loc[step,'mean']-g.loc[20,'mean'],since20_star=g.loc[step,'star']-g.loc[20,'star'],
       within=g.loc[step,'within']/g.loc[0,'within'],between=g.loc[step,'between']/g.loc[0,'between']))
 b=pd.DataFrame(allrows);s=b.groupby(['arm','iv','seed','window']).mean(numeric_only=True).drop(columns='task').reset_index()
 st=pd.DataFrame(steps)
 for name,f in [('boundary_windows',b),('seed_windows',s),('seed_steps',st),('validation',pd.DataFrame(checks)),('boundary_positions',pd.DataFrame(position))]:
  f.to_csv(OUT/(name+'.csv'),index=False,lineterminator='\n')
 v=s.groupby(['arm','iv','window']).median(numeric_only=True).drop(columns='seed').reset_index();v.to_csv(OUT/'verdict.csv',index=False,lineterminator='\n')
 ranges=s.groupby(['arm','iv','window']).agg({c:['min','max'] for c in s.columns if c not in ['arm','iv','window','seed']})
 ranges.columns=['_'.join(c) for c in ranges.columns];ranges.reset_index().to_csv(OUT/'seed_ranges.csv',index=False,lineterminator='\n')
 sp=pd.DataFrame(position).groupby(['arm','iv','seed','step']).mean(numeric_only=True).drop(columns='task').reset_index()
 sp.to_csv(OUT/'seed_positions.csv',index=False,lineterminator='\n')
 vp=sp.groupby(['arm','iv','step']).median(numeric_only=True).drop(columns='seed').reset_index();vp.to_csv(OUT/'position_verdict.csv',index=False,lineterminator='\n')
 chunks=['# task21..40 切替直後の勾配と下降寄与（追加記述解析）',
 '20境界/seed、4条件3seed。符号付き寄与は境界内累積→seed平均→中央値。独立な機構確証ではない。局所方向は−B*dL/dz（正が上向き）。',
 '## 各窓の実移動・現在CE・履歴',md(v[['arm','iv','window','actual_mean','actual_W','actual_b','actual_star','current_mean','current_l2_mean','history_mean']]),
 '## step1: 入力符号別の局所方向と同じ標本の実移動',md(v[v.window=='1-1'][['arm','iv','positive_local','negative_local','positive_actual','negative_actual','positive_fraction','negative_up_actualdown_fraction']]),
 '## 正負入力源の蓄積による分解',md(v[v.window.isin(['1-1','1-20'])][['arm','iv','window','actual_mean','pos_mean','neg_mean','l2_mean','pre_mean','fixed_negative_actual','fixed_positive_actual']]),
 '## 実際に平均が下がったユニット更新での符号付き寄与率',md(v[v.window.isin(['1-1','1-20'])][['arm','iv','window','actual_down_W_share','actual_down_b_share','pos_down_share','neg_down_share','l2_down_share','pre_down_share']]),
 '## タスク内の各時点',md(vp),
 '## 検算',md(pd.DataFrame(checks).drop(columns=['arm','iv','seed','task']).max().rename('max').reset_index().rename(columns={'index':'check'})),
 '## 限界','符号付き成分の中央値は一般に加算しない。下降寄与率は実Δzbar<−1e−8のユニット更新に条件付け、各境界内で割合→seed平均→中央値。上向き成分は負の割合となる。自己標本の寄与とscalar MSE自己項を同一視しない。共有更新は同じユニットの別入力による寄与であり他ユニットの因果効果ではない。Adam分母固定の寄与分解で、成分を除去した介入効果ではない。']
 (OUT/'summary.md').write_text('\n\n'.join(chunks)+'\n')
 plots(st)
 print('\n\n'.join(chunks),flush=True)
def plots(st):
 arms=[('LR','none'),('LR','l2'),('SNA','none'),('SNA','l2')]
 fig,axs=plt.subplots(3,4,figsize=(16,10),layout='constrained')
 for col,(arm,iv) in enumerate(arms):
  f=st[(st.arm==arm)&(st.iv==iv)]
  for row,series in enumerate([
   [('actual_mean','Actual','black'),('current_mean','Current CE','#0072B2'),('history_mean','Earlier moments','#D55E00')],
   [('actual_mean','Actual','black'),('pos_mean','Positive-source CE','#D55E00'),('neg_mean','Negative-source CE','#0072B2'),('l2_mean','Post-switch L2','#CC79A7'),('pre_mean','Pre-switch moments','gray')],
   [('positive_local','Local: z > 0','#D55E00'),('negative_local','Local: z <= 0','#0072B2')]]):
   ax=axs[row,col]
   for key,label,color in series:
    pp=f.pivot(index='seed',columns='step',values=key);a=pp.values
    ax.plot(pp.columns,np.median(a,0),color=color,label=label,lw=1.5)
    ax.fill_between(pp.columns,a.min(0),a.max(0),color=color,alpha=.1)
   ax.axhline(0,color='gray',lw=.7);ax.set_title(arm+(' + L2' if iv=='l2' else ''))
   ax.set_xlabel('Update after switch');ax.set_ylabel('Local direction' if row==2 else 'Mean z change per update')
 for row in range(3):axs[row,0].legend(fontsize=7)
 fig.suptitle('Task21..40: first update and evolving gradient contributions\nMedian and range of 3 seed means; local derivative and actual shared update are different quantities')
 for ext in ['png','pdf']:fig.savefig(OUT/('first_updates.'+ext),dpi=160)
 plt.close(fig)
 fig,axs=plt.subplots(1,4,figsize=(16,4),layout='constrained')
 for ax,(arm,iv) in zip(axs,arms):
  f=st[(st.arm==arm)&(st.iv==iv)]
  for key,label,color in [('actual_mean','All inputs','black'),('fixed_negative_actual','Initially negative','#0072B2'),('fixed_positive_actual','Initially positive','#D55E00')]:
   pp=f.pivot(index='seed',columns='step',values=key).cumsum(axis=1);a=pp.values
   ax.plot(pp.columns,np.median(a,0),label=label,color=color)
   ax.fill_between(pp.columns,a.min(0),a.max(0),color=color,alpha=.1)
  ax.axhline(0,color='gray',lw=.7);ax.set_title(arm+(' + L2' if iv=='l2' else ''));ax.set_xlabel('Update after switch');ax.set_ylabel('Cumulative mean z change')
 axs[0].legend(fontsize=8)
 for ext in ['png','pdf']:fig.savefig(OUT/('fixed_receivers.'+ext),dpi=160)
 plt.close(fig)
if __name__=='__main__':main()

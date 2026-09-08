"""Report paired-gradient reanalysis. All results descriptive, boundary -> seed aggregation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_transport_0908 as T
OUT=T.OUT
ARMS=[('SNA','none'),('SNA','l2'),('LR','none'),('LR','l2')]
def md(frame):
 if isinstance(frame,pd.Series):frame=frame.to_frame('value')
 if not isinstance(frame.index,pd.RangeIndex):frame=frame.reset_index()
 def val(x):return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
 lines=['| '+' | '.join(map(str,frame.columns))+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
 lines+=['| '+' | '.join(val(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
 return '\n'.join(lines)
def main():
 seeds=[];variance=[];receiver=[];paired=[];checks=[];examples={}
 for arm,iv in ARMS:
  for seed in range(3):
   prefix=f'{arm}_{iv}_s{seed}'
   raw=dict(np.load(OUT/'raw'/(prefix+'.npz')))
   counts=pd.read_csv(OUT/(prefix+'_paired_counts.csv'))
   vv=pd.read_csv(OUT/(prefix+'_variance.csv'))
   for lo,hi in [(1,5),(6,10),(11,15),(16,20),(1,20)]:
    c=counts[(counts.step>=lo)&(counts.step<=hi)].groupby('task').sum(numeric_only=True)
    denom=c.n_neg_local_up.replace(0,np.nan)
    paired.append(dict(arm=arm,iv=iv,seed=seed,window=f'{lo}-{hi}',
     neg_localup_actualdown=float((c.n_neg_local_up_actual_down/denom).mean()),
     neg_localup_currentdown=float((c.n_neg_local_up_current_down/denom).mean()),
     local_actual_opposed=float((c.n_opposed/c.n_valid.replace(0,np.nan)).mean()),
     n_negative=int(c.n_negative.sum()),n_neg_local_up=int(c.n_neg_local_up.sum()),
     n_up_actualdown=int(c.n_neg_local_up_actual_down.sum()),
     n_up_currentdown=int(c.n_neg_local_up_current_down.sum()),n_boundaries=int(denom.notna().sum())))
   before=vv[vv.step==0].set_index('task')
   for step in range(21):
    after=vv[vv.step==step].set_index('task')
    row=dict(arm=arm,iv=iv,seed=seed,step=step)
    for k in ['mean','within','between','total']:
     row[k+'_before']=float(before[k].mean());row[k]=float(after[k].mean())
     row[k+'_delta']=float((after[k]-before[k]).mean())
     if k!='mean':row[k+'_ratio']=float((after[k]/before[k].where(before[k]>1e-12)).mean())
    row['within_unit_ratio']=float(np.nanmean(raw['within_var'][:,step]/np.where(raw['within_var'][:,0]>1e-12,raw['within_var'][:,0],np.nan)))
    row['within_flow']=float(after.within_flow.mean());row['between_flow']=float(after.between_flow.mean())
    variance.append(row)
   for side,si in [('positive',0),('negative',1)]:
    n=raw['receiver_counts'][:,si,:]
    for step in range(1,21):
     row=dict(arm=arm,iv=iv,seed=seed,receiver=side,step=step,n_per_boundary=float(n.sum(1).mean()))
     for key in T.COMP+['actual','rounding']:
      cum=np.nansum(raw['receiver_'+key+'_fixed'][:,:step,si,:],axis=1)
      per_boundary=(cum*n).sum(1)/n.sum(1)
      row[key]=float(per_boundary.mean())
      row[key+'_equalunit']=float(np.nanmean(np.where(n>0,cum,np.nan),axis=1).mean())
     z0=raw['probe_z0'];z20=raw['probe_z20']
     mask=z0>0 if si==0 else z0<=0
     row['fraction_down_at20']=float(np.mean((((z20-z0)<-1e-8)&mask).sum(axis=(1,2))/mask.sum(axis=(1,2))))
     receiver.append(row)
   # Global numerator balance, no receiving-side conditioning.
   for step in [1,5,10,20]:
    row=dict(arm=arm,iv=iv,seed=seed,step=step)
    for key in T.COMP+['actual','rounding']:
     row[key]=float(raw[key][:,:step].sum(1).mean())
    seeds.append(row)
   if seed==0:examples[(arm,iv)]=raw
   meta=json.loads((OUT/(prefix+'_provenance.json')).read_text())
   checks.extend([dict(arm=arm,iv=iv,seed=seed,**c) for c in meta['checks']])
 frames={'seed_global':pd.DataFrame(seeds),'seed_variance':pd.DataFrame(variance),
 'seed_receiver':pd.DataFrame(receiver),'seed_paired':pd.DataFrame(paired),'validation':pd.DataFrame(checks)}
 for name,f in frames.items():f.to_csv(OUT/(name+'.csv'),index=False,lineterminator='\n')
 vv=frames['seed_variance'].groupby(['arm','iv','step']).median(numeric_only=True).drop(columns='seed').reset_index()
 pp=frames['seed_paired'].groupby(['arm','iv','window']).median(numeric_only=True).drop(columns='seed').reset_index()
 rr=frames['seed_receiver'].groupby(['arm','iv','receiver','step']).median(numeric_only=True).drop(columns='seed').reset_index()
 gg=frames['seed_global'].groupby(['arm','iv','step']).median(numeric_only=True).drop(columns='seed').reset_index()
 for name,f in [('verdict',gg),('variance_verdict',vv),('paired_verdict',pp),('receiver_verdict',rr)]:
  f.to_csv(OUT/(name+'.csv'),index=False,lineterminator='\n')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 colors={'pos':'#D55E00','neg':'#0072B2','l2':'#CC79A7','pre':'#777777','actual':'black'}
 def line(ax,f,key,label,color,baseline=False):
  p=f.pivot(index='seed',columns='step',values=key);x=p.columns.to_numpy();a=p.to_numpy()
  ax.plot(x,np.median(a,axis=0),color=color,label=label,lw=1.8)
  ax.fill_between(x,a.min(0),a.max(0),color=color,alpha=.12)
 def title(arm,iv):return arm+(' + L2' if iv=='l2' else '')
 fig,axs=plt.subplots(2,4,figsize=(17,7),layout='constrained')
 for col,(arm,iv) in enumerate(ARMS):
  for row,side in enumerate(['negative','positive']):
   f=frames['seed_receiver'];f=f[(f.arm==arm)&(f.iv==iv)&(f.receiver==side)]
   for key in ['actual','pos','neg','l2','pre']:
    line(axs[row,col],f,key,{'actual':'Actual','pos':'Positive-source CE','neg':'Negative-source CE','l2':'Post-switch L2','pre':'Pre-switch m'}[key],colors[key])
   axs[row,col].axhline(0,color='#aaa',lw=.7)
   axs[row,col].set_title(title(arm,iv)+' / '+side+' receivers')
   axs[row,col].set_xlabel('Updates after switch');axs[row,col].set_ylabel('Cumulative mean z change')
 axs[0,0].legend(fontsize=8)
 fig.suptitle('Which gradients move the same held-out inputs?\nReceiver signs fixed at step 0; source signs at gradient generation; current and accumulated post-switch moments included',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(OUT/('source_to_receiver.'+ext),dpi=170)
 plt.close(fig)
 fig,axs=plt.subplots(2,4,figsize=(17,7),layout='constrained')
 for col,(arm,iv) in enumerate(ARMS):
  f=frames['seed_variance'];f=f[(f.arm==arm)&(f.iv==iv)]
  line(axs[0,col],f,'mean_delta','Mean change','black')
  for key,label,color in [('within_ratio','Within-unit input variance','#0072B2'),('between_ratio','Variance of unit means','#D55E00')]:
   line(axs[1,col],f,key,label,color)
  axs[0,col].axhline(0,color='#aaa',lw=.7);axs[1,col].axhline(1,color='#aaa',lw=.7)
  axs[0,col].set_title(title(arm,iv));axs[1,col].set_title(title(arm,iv))
  for row in range(2):axs[row,col].set_xlabel('Updates after switch')
  axs[0,col].set_ylabel('Mean z change');axs[1,col].set_ylabel('Variance / step-0 variance')
 axs[1,0].legend(fontsize=8)
 fig.suptitle('Mean descent and two different widths\nSame fixed 512 probe images; 19 boundaries/seed; median and range of 3 seeds; ratios calculated within each boundary',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(OUT/('mean_and_variances.'+ext),dpi=170)
 plt.close(fig)
 fig,axs=plt.subplots(1,4,figsize=(17,4),layout='constrained')
 windows=['1-5','6-10','11-15','16-20']
 for ax,(arm,iv) in zip(axs,ARMS):
  f=frames['seed_paired'];f=f[(f.arm==arm)&(f.iv==iv)&(f.window!='1-20')]
  for key,label,color in [('neg_localup_actualdown','Actual update','#D55E00'),('neg_localup_currentdown','Current CE only','#0072B2')]:
   a=np.array([[f[(f.seed==seed)&(f.window==w)].iloc[0][key] for w in windows] for seed in range(3)])
   ax.plot(range(4),100*np.median(a,axis=0),label=label,color=color,marker='o')
   ax.fill_between(range(4),100*a.min(0),100*a.max(0),color=color,alpha=.12)
  ax.set_xticks(range(4),windows);ax.set_ylim(0,100);ax.set_title(title(arm,iv))
  ax.set_xlabel('Update window');ax.set_ylabel('Moves down despite local-up (%)')
 axs[0].legend(fontsize=8)
 fig.suptitle('Same training sample: negative z, local gradient says up, but the update moves it down\nCurrent CE only includes shared-parameter coupling; Actual also includes accumulated moments and L2',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(OUT/('local_vs_actual.'+ext),dpi=170)
 plt.close(fig)
 chunks=['# 初期20更新の局所勾配と実移動：事後再解析',
 'CPU保存軌道の全4腕3seed19境界を20更新再生。seed中央値、境界平均。元CUDA軌道とは異なる。比と成分の中央値は一般に加算しない。',
 '## 局所上向きの負側標本のうち下向きへ動く割合（1..20）',
 md(pp[pp.window=='1-20'][['arm','iv','neg_localup_actualdown','neg_localup_currentdown','local_actual_opposed','n_boundaries']]),
 '## 最初20更新の全体の寄与',md(gg[gg.step==20]),
 '## step0負側受け手の最初20更新',md(rr[(rr.receiver=='negative')&(rr.step==20)][['arm','iv','actual','pos','neg','l2','pre','fraction_down_at20']]),
 '## 分散の初期比（step20）',md(vv[vv.step==20][['arm','iv','mean_delta','within_ratio','between_ratio','total_ratio','within_unit_ratio']]),
 '## 検算最大誤差',md(frames['validation'].drop(columns=['arm','iv','seed','task']).max()),
 '## 読みの制限',
 '正側/負側の源は同じユニットの標本で、正側ユニット/負側ユニットの分業ではない。初期に負だった受け手を固定して追跡する。局所符号と実更新の比較は同じ学習標本。同じsampleの寄与は局所符号と一致し、他標本との共有更新によって現在CE全体が逆向きになり得る。Adam由来分解は実分母を固定した帳簿であり、正側勾配の除去やAdamリセットの介入効果ではない。自己項帰属・長期ラチェットの原因は未確定。分散は同じ固定入力集合/ユニット集合で対応した有限変位から直接算出。']
 (OUT/'summary.md').write_text('\n\n'.join(chunks)+'\n')
 print('\n\n'.join(chunks))
if __name__=='__main__':main()

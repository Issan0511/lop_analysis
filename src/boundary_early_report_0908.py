"""Early formation vs late boundary windows; same-seed descriptive comparison."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_early_source_0908 as E
from src import boundary_transport_0908 as T
from src import boundary_transport_report_0908 as R
ROOT=E.ROOT;OUT=ROOT/'results/boundary_early_0908'
def metrics(st,x,mu):
 p=st['params'];w=p[0].double();b=p[1].double()
 z=x.double()@w.T+b;mean=z.mean(0)
 return dict(mean=float(mean.mean()),star=float((w.sum(1)*mu+b).mean()),
             star_W=float((w.sum(1)*mu).mean()),bias=float(b.mean()),
             within=float(z.var(0,unbiased=False).mean()),
             between=float(mean.var(unbiased=False)),
             W_norm=float(w.norm()),b_norm=float(b.norm()))
def restore20(raw,arm,iv,mnist):
 p,act,adam=T.G.clone_state(raw,arm)
 gd=torch.Generator();gd.set_state(raw['rng_after_perm']['data'])
 gb=torch.Generator();gb.set_state(raw['rng_after_perm']['batch'])
 idx=E.H.stratified_draw(mnist,gd);order=torch.randperm(E.H.TASK_EXAMPLES,generator=gb)
 xs=mnist.train_x[idx][:,raw['perm']][order];ys=mnist.train_y[idx][order]
 for step in range(20):
  out=E.H.forward(p,xs[16*step:16*(step+1)],act)
  g=torch.autograd.grad(torch.nn.functional.cross_entropy(out[4],ys[16*step:16*(step+1)]),p)
  with torch.no_grad():
   if iv=='l2':g=[gg+2*.001*q for gg,q in zip(g,p)]
   m,v,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
   for q,gg,mi,vi in zip(p,g,m,v):
    mi.mul_(.9).add_(gg,alpha=1-.9);vi.mul_(.999).addcmul_(gg,gg,value=1-.999)
    q-=.001*(mi/c1)/((vi/c2).sqrt()+1e-8)
   if arm=='SNA':act.update(out[0],out[2])
 return E.state(p,act)
def main():
 torch.set_num_threads(1);E.H.setup('cpu');mnist=E.H.Mnist(torch.device('cpu'))
 R.OUT=OUT/'transport';R.main()
 trajectories=[];boundaryrows=[];initial=[];sanity=[]
 for period,source,transport in [('early',OUT/'source',OUT/'transport'),
                                ('late',ROOT/'results/boundary_groups_0908',ROOT/'results/boundary_transport_0908')]:
  for arm in ['SNA','LR']:
   for iv in ['none','l2']:
    for seed in range(3):
     prefix=f'{arm}_{iv}_s{seed}'
     saved=torch.load(source/(prefix+'_states.pt'),weights_only=False,map_location='cpu')
     raw=dict(np.load(transport/'raw'/(prefix+'.npz')))
     cc=pd.read_csv(transport/(prefix+'_paired_counts.csv')).groupby('task').sum(numeric_only=True)
     px=mnist.test_x[saved['probe_indices']];mu=saved['mu_mean']
     if period=='early':
      ft=saved['first_task'];x=px[:,ft['perm']]
      for step,key in [(0,'before'),(20,'after_20'),(625,'after_625')]:
       initial.append(dict(arm=arm,iv=iv,seed=seed,task=1,step=step,**metrics(ft[key]['state'],x,mu)))
     for j,b in enumerate(saved['boundaries'][:19]):
      task=b['task'];x=px[:,b['perm']]
      states=[b['before']['state'],
              b['after_20']['state'] if period=='early' else restore20(b,arm,iv,mnist),
              b['after_625']['state']]
      mm=[metrics(st,x,mu) for st in states]
      with torch.no_grad():
       measured=T.evaluate(states[1]['params'][:2],x).mean(0).numpy()
      error=float(abs(measured-raw['mean'][j,20]).max())
      assert error<=2e-5,(period,prefix,task,error)
      sanity.append(dict(period=period,arm=arm,iv=iv,seed=seed,task=task,step20_mean_replay_maxabs=error))
      for step,m in zip([0,20,625],mm):
       trajectories.append(dict(period=period,arm=arm,iv=iv,seed=seed,task=task,step=step,**m))
      row=dict(period=period,arm=arm,iv=iv,seed=seed,task=task,index=j+1,
               bin='first5' if j<5 else 'next5' if j<10 else 'last9')
      for key in ['actual','pos','neg','l2','pre']:
       row[key]=float(raw[key][j].sum(0).mean())
      for si,side in [(0,'positive'),(1,'negative')]:
       n=raw['receiver_counts'][j,si]
       for key in ['actual','pos','neg','l2','pre']:
        values=np.nansum(raw['receiver_'+key+'_fixed'][j,:,si,:],axis=0)
        row[side+'_'+key]=float((values*n).sum()/n.sum())
      c=cc.loc[task]
      row['localup_actualdown']=float(c.n_neg_local_up_actual_down/c.n_neg_local_up)
      row['localup_currentdown']=float(c.n_neg_local_up_current_down/c.n_neg_local_up)
      row['within_ratio']=mm[1]['within']/mm[0]['within']
      row['between_ratio']=mm[1]['between']/mm[0]['between']
      for step,k in [(20,1),(625,2)]:
       for key in ['star','star_W','bias','mean','within']:
        row[key+'_delta'+str(step)]=mm[k][key]-mm[0][key]
      boundaryrows.append(row)
     print('METRICS',period,prefix,flush=True)
 traj=pd.DataFrame(trajectories);br=pd.DataFrame(boundaryrows);ini=pd.DataFrame(initial)
 for name,f in [('task_states',traj),('per_boundary',br),('first_task',ini),('report_validation',pd.DataFrame(sanity))]:
  f.to_csv(OUT/(name+'.csv'),index=False,lineterminator='\n')
 numeric=[c for c in br.columns if c not in ['period','arm','iv','seed','task','index','bin']]
 sd=br.groupby(['period','arm','iv','seed'])[numeric].mean().reset_index()
 sb=br.groupby(['period','arm','iv','seed','bin'])[numeric].mean().reset_index()
 sd.to_csv(OUT/'seed_comparison.csv',index=False,lineterminator='\n')
 sb.to_csv(OUT/'seed_bins.csv',index=False,lineterminator='\n')
 verdict=sd.groupby(['period','arm','iv'])[numeric].median().reset_index()
 verdict.to_csv(OUT/'verdict.csv',index=False,lineterminator='\n')
 paired=sd[sd.period=='early'].set_index(['arm','iv','seed'])[numeric]-sd[sd.period=='late'].set_index(['arm','iv','seed'])[numeric]
 paired.reset_index().to_csv(OUT/'paired_early_minus_late.csv',index=False,lineterminator='\n')
 drift=[]
 for (period,arm,iv,seed),f in traj.groupby(['period','arm','iv','seed']):
  start=f[f.step==0].sort_values('task').iloc[0];end=f[f.step==625].sort_values('task').iloc[-1]
  terms=br[(br.period==period)&(br.arm==arm)&(br.iv==iv)&(br.seed==seed)]
  assert abs(terms.star_delta625.sum()-(end.star-start.star))<1e-10
  prefix=f'{arm}_{iv}_s{seed}'
  if period=='early':
   prior=ini[(ini.arm==arm)&(ini.iv==iv)&(ini.seed==seed)&(ini.step==625)].iloc[0]
   prior_within=float(prior.within);prior_mean=float(prior['mean'])
  else:
   cp=torch.load(ROOT/'results/boundary_groups_0908'/(prefix+'_task100.pt'),weights_only=False,map_location='cpu')
   ids=torch.randperm(len(mnist.test_x),generator=E.H.stream('boundary_probe',seed))[:512]
   oldmetrics=metrics(cp['state'],mnist.test_x[ids][:,cp['perm']],float(mnist.train_x.mean()))
   prior_within=oldmetrics['within'];prior_mean=oldmetrics['mean']
  row=dict(period=period,arm=arm,iv=iv,seed=seed,start_star=start.star,end_star=end.star,
           total_star=end.star-start.star,total_star_W=end.star_W-start.star_W,
           total_bias=end.bias-start.bias,start_within=prior_within,end_within=end.within,
           start_postswitch_within=start.within,prior_end_mean=prior_mean,
           start_postswitch_mean=start['mean'],end_mean=end['mean'],
           start_Wnorm=start.W_norm,end_Wnorm=end.W_norm)
  drift.append(row)
 drift=pd.DataFrame(drift);drift.to_csv(OUT/'seed_drift.csv',index=False,lineterminator='\n')
 drift.groupby(['period','arm','iv']).median(numeric_only=True).drop(columns='seed').reset_index().to_csv(OUT/'drift_verdict.csv',index=False,lineterminator='\n')
 makeplots(traj,br,ini,sd)
 keycols=['period','arm','iv','actual','negative_actual','pos','neg','within_ratio','star_delta20','star_delta625']
 chunks=['# 初期task2..20と後半task101..119：形成過程の比較',
 '初期/後半は時期の名称。定常性判定ではない。4腕3seed、各19境界、同じCPU実装。表示値はseed平均の中央値。',
 '## 最初20更新とタスク終端',R.md(verdict[keycols]),
 '## 設定点の19境界の累積変位（幅のstart/endは前タスク終端と最後のタスク終端で位相を一致）',R.md(drift.groupby(['period','arm','iv'])[['start_star','end_star','total_star','total_star_W','total_bias','start_within','end_within']].median()),
 '## 初期の固定ビン',R.md(sb[sb.period=='early'].groupby(['arm','iv','bin'])[['actual','negative_actual','pos','neg','within_ratio','star_delta625']].median()),
 '## 最初のタスク1（切替なし）',R.md(ini.groupby(['arm','iv','step'])[['mean','star','within']].median()),
 '## 限界',
 'タスク2..20を自動的に非平衡、101..119を自動的に平衡と呼ばない。位置/幅/設定点/源別相殺は異なる量。Adam分母固定の寄与帳簿で自己項やユニット間因果を帰属しない。初期と後半は同一seedの学習経路の異なる時点であり、独立した介入比較ではない。元CUDAとは別。']
 (OUT/'summary.md').write_text('\n\n'.join(chunks)+'\n')
 print('\n\n'.join(chunks))
def makeplots(traj,br,ini,sd):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 arms=[('SNA','none'),('SNA','l2'),('LR','none'),('LR','l2')]
 def title(a,i):return a+(' + L2' if i=='l2' else '')
 def band(ax,f,x,key,label,color):
  pp=f.pivot(index='seed',columns=x,values=key);xx=pp.columns.to_numpy();a=pp.to_numpy()
  ax.plot(xx,np.median(a,0),label=label,color=color,lw=1.8)
  ax.fill_between(xx,a.min(0),a.max(0),color=color,alpha=.12)
 fig,axs=plt.subplots(3,4,figsize=(17,10),layout='constrained')
 for col,(arm,iv) in enumerate(arms):
  f=traj[(traj.period=='early')&(traj.arm==arm)&(traj.iv==iv)&(traj.step==625)]
  f=pd.concat([f,ini[(ini.arm==arm)&(ini.iv==iv)&(ini.step==625)]])
  start=ini[(ini.arm==arm)&(ini.iv==iv)&(ini.step==0)].copy();start['task']=0
  f=pd.concat([start,f])
  for row,(key,label) in enumerate([('mean','Mean preactivation'),('star','Setpoint star'),('within','Mean within-unit variance')]):
   for seed,g in f.groupby('seed'):axs[row,col].plot(g.sort_values('task').task,g.sort_values('task')[key],alpha=.45,lw=1,label='seed '+str(seed))
   pp=f.groupby('task')[key].median();axs[row,col].plot(pp.index,pp.values,color='black',lw=2,label='median')
   axs[row,col].set_title(title(arm,iv));axs[row,col].set_xlabel('Task (end of task)');axs[row,col].set_ylabel(label)
 axs[0,0].legend(fontsize=8)
 fig.suptitle('Formation from initialization through task 20\nTask 0 = initialized parameters on task-1 inputs; tasks 1..20 = task end; same 3 seeds as late-window audit',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(OUT/('formation.'+ext),dpi=170)
 plt.close(fig)
 fig,axs=plt.subplots(2,4,figsize=(17,7),layout='constrained')
 for col,(arm,iv) in enumerate(arms):
  f=br[(br.period=='early')&(br.arm==arm)&(br.iv==iv)]
  for row,pref in enumerate(['','negative_']):
   for key,color in [('actual','black'),('pos','#D55E00'),('neg','#0072B2')]:
    band(axs[row,col],f,'task',pref+key,{'actual':'Actual','pos':'Positive-source CE','neg':'Negative-source CE'}[key],color)
   axs[row,col].axhline(0,color='#aaa',lw=.7);axs[row,col].set_title(title(arm,iv)+(' / all inputs' if row==0 else ' / initially negative'))
   axs[row,col].set_xlabel('Task after switch');axs[row,col].set_ylabel('Change during first 20 updates')
 axs[0,0].legend(fontsize=8)
 fig.suptitle('How the source balance develops across the first 19 switches\nSource signs at gradient generation; receivers fixed at each switch; median and range of 3 seeds',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(OUT/('early_source_balance.'+ext),dpi=170)
 plt.close(fig)
 fig,axs=plt.subplots(2,4,figsize=(17,7),layout='constrained')
 for col,(arm,iv) in enumerate(arms):
  f=br[(br.period=='early')&(br.arm==arm)&(br.iv==iv)]
  for key,color,label in [('star_delta20','#D55E00','First 20 updates'),('star_delta625','#0072B2','Whole task (625)')]:
   band(axs[0,col],f,'task',key,label,color)
  axs[0,col].axhline(0,color='#aaa',lw=.7)
  for key,color,label in [('within_ratio','#0072B2','Within-unit variance'),('between_ratio','#D55E00','Variance of unit means')]:
   band(axs[1,col],f,'task',key,label,color)
  axs[1,col].axhline(1,color='#aaa',lw=.7)
  for row in [0,1]:
   axs[row,col].set_title(title(arm,iv));axs[row,col].set_xlabel('Task after switch')
  axs[0,col].set_ylabel('Setpoint change');axs[1,col].set_ylabel('Variance at 20 / at switch')
 axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8)
 fig.suptitle('What remains after the early dip?\nPer-boundary changes: first20 versus whole-task setpoint change; no stationarity assumption',fontsize=13)
 for ext in ['png','pdf']:fig.savefig(OUT/('early_residual.'+ext),dpi=170)
 plt.close(fig)
if __name__=='__main__':main()

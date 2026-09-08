"""Task20..100 continuation of the existing CPU trajectory; no new intervention."""
from pathlib import Path
import argparse,json,hashlib,time,subprocess,sys,concurrent.futures
import torch
import numpy as np
import pandas as pd
from src import boundary_early_source_0908 as E
from src import boundary_transport_0908 as T
ROOT=E.ROOT;OUT=ROOT/'results/task20to100_0908'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def exact(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
 return a==b
@torch.no_grad()
def measure(p,x,mu):
 w,b=p[0].double(),p[1].double()
 z=x.double()@w.T+b
 return {'zmean':z.mean(0).numpy(),'within':z.var(0,unbiased=False).numpy(),
         'star':(w.sum(1)*mu+b).numpy(),'star_W':(w.sum(1)*mu).numpy(),'bias':b.numpy().copy()}
def run(arm,iv,seed):
 torch.set_num_threads(1);E.H.setup('cpu');mnist=E.H.Mnist(torch.device('cpu'))
 OUT.mkdir(parents=True,exist_ok=True);prefix=f'{arm}_{iv}_s{seed}'
 sp=ROOT/'results/boundary_early_0908/source'/(prefix+'_states.pt')
 cp=ROOT/'results/boundary_groups_0908'/(prefix+'_task100.pt')
 saved=torch.load(sp,weights_only=False,map_location='cpu');b=saved['boundaries'][-1];assert b['task']==20
 restore=dict(b,before=b['after_625'],adam_before=b['adam_after'])
 p,act,adam=T.G.clone_state(restore,arm)
 gp=torch.Generator();gp.set_state(b['rng_after_perm']['perm'])
 gd=torch.Generator();gd.set_state(b['rng_after_perm']['data'])
 gb=torch.Generator();gb.set_state(b['rng_after_perm']['batch'])
 E.H.stratified_draw(mnist,gd);torch.randperm(E.H.TASK_EXAMPLES,generator=gb)
 assert exact(E.state(p,act),b['after_625']['state'])
 px=mnist.test_x[saved['probe_indices']];mu=saved['mu_mean'];perm=b['perm']
 rec=[measure(p,px[:,perm],mu)];rows=[];tasks=[20]
 started=time.monotonic()
 for task in range(21,101):
  assert time.monotonic()-started<900
  perm=torch.randperm(784,generator=gp)
  idx=E.H.stratified_draw(mnist,gd);order=torch.randperm(E.H.TASK_EXAMPLES,generator=gb)
  xs=mnist.train_x[idx][:,perm][order];ys=mnist.train_y[idx][order]
  for step in range(625):
   out=E.H.forward(p,xs[16*step:16*(step+1)],act)
   gr=torch.autograd.grad(torch.nn.functional.cross_entropy(out[4],ys[16*step:16*(step+1)]),p)
   with torch.no_grad():
    if iv=='l2':gr=[g+2*.001*q for g,q in zip(gr,p)]
    m,v,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
    for q,g,mi,vi in zip(p,gr,m,v):
     mi.mul_(.9).add_(g,alpha=1-.9);vi.mul_(.999).addcmul_(g,g,value=1-.999)
     q-=.001*(mi/c1)/((vi/c2).sqrt()+1e-8)
    if arm=='SNA':act.update(out[0],out[2])
  assert all(torch.isfinite(q).all() for q in p)
  tasks.append(task);rec.append(measure(p,px[:,perm],mu))
  if task%20==0:print(prefix,'task',task,'seconds',round(time.monotonic()-started,1),flush=True)
 ref=torch.load(cp,weights_only=False,map_location='cpu')
 validation={'state':exact(E.state(p,act),ref['state']),'adam':exact(adam,ref['adam']),
             'perm':exact(perm,ref['perm']),'streams':exact({'perm':gp.get_state(),'data':gd.get_state(),'batch':gb.get_state()},ref['streams'])}
 meta={'arm':arm,'iv':iv,'seed':seed,'task100_exact':validation,'source_sha256':sha(sp),'reference100_sha256':sha(cp),
       'code_sha256':sha(Path(__file__)),'spec_sha256':sha(ROOT/'specs/spec_task20to100_0908.md'),'data_sha256':mnist.sha256,'seconds':time.monotonic()-started}
 (OUT/(prefix+'_provenance.json')).write_text(json.dumps(meta,indent=2))
 assert all(validation.values()),validation
 np.savez_compressed(OUT/(prefix+'.npz'),task=np.array(tasks),**{k:np.stack([r[k] for r in rec]) for k in rec[0]})
 for t,r in zip(tasks,rec):
  z=r['zmean']
  rows.append(dict(arm=arm,iv=iv,seed=seed,task=t,mean=float(z.mean()),q10=float(np.quantile(z,.1)),
                   q50=float(np.quantile(z,.5)),q90=float(np.quantile(z,.9)),within=float(r['within'].mean()),
                   between=float(z.var()),star=float(r['star'].mean()),star_W=float(r['star_W'].mean()),bias=float(r['bias'].mean())))
 pd.DataFrame(rows).to_csv(OUT/(prefix+'.csv'),index=False,lineterminator='\n')
 print('FINISHED',prefix,validation,flush=True)
def report():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fs=[pd.read_csv(OUT/f'{arm}_{iv}_s{s}.csv') for arm in ['LR','SNA'] for iv in ['none','l2'] for s in range(3)]
 df=pd.concat(fs,ignore_index=True);df.to_csv(OUT/'trajectory.csv',index=False,lineterminator='\n')
 cols=['mean','q10','q50','q90','within','between','star','star_W','bias'];rows=[]
 for (arm,iv,seed),g in df.groupby(['arm','iv','seed']):
  g=g.set_index('task')
  for lo,hi in [(20,100),(20,40),(40,60),(60,80),(80,100)]:
   rows.append(dict(arm=arm,iv=iv,seed=seed,window=f'{lo}-{hi}',**{c:g.loc[hi,c]-g.loc[lo,c] for c in cols}))
 sd=pd.DataFrame(rows);sd.to_csv(OUT/'seed_delta.csv',index=False,lineterminator='\n')
 vv=[]
 for (arm,iv,window),g in sd.groupby(['arm','iv','window']):
  r=dict(arm=arm,iv=iv,window=window)
  for c in cols:r.update({c+'_median':g[c].median(),c+'_min':g[c].min(),c+'_max':g[c].max(),c+'_n_down':int((g[c]<-1e-6).sum())})
  vv.append(r)
 vv=pd.DataFrame(vv);vv.to_csv(OUT/'verdict.csv',index=False,lineterminator='\n')
 levels=df[df.task.isin([20,40,60,80,100])].groupby(['arm','iv','task'])[cols].median().reset_index()
 levels.to_csv(OUT/'levels.csv',index=False,lineterminator='\n')
 def md(f):
  return '\n'.join(['| '+' | '.join(f.columns)+' |','|'+'|'.join(['---']*len(f.columns))+'|']+['| '+' | '.join(f'{x:.6g}' if isinstance(x,float) else str(x) for x in r)+' |' for _,r in f.iterrows()])
 summary='# タスク20〜100の前活性分布（タスク終端を比較）\n\n既存CPU軌道の追加記録。全12走のtask100 checkpointが全状態・Adam・RNGまで完全一致。seed内差→3seed中央値。分位点はユニット平均前活性の分布。\n\n## 20→100の差\n\n'+md(vv[vv.window=='20-100'][['arm','iv','mean_median','mean_min','mean_max','mean_n_down','q10_median','q50_median','q90_median','star_median','within_median']])+'\n\n## 終端水準のseed中央値\n\n'+md(levels)+'\n\n## 区間別平均位置の差\n\n'+md(vv[['arm','iv','window','mean_median','mean_min','mean_max','mean_n_down','star_median']])+'\n'
 (OUT/'summary.md').write_text(summary);print(summary,flush=True)
 fig,axs=plt.subplots(3,4,figsize=(15,10),layout='constrained')
 for col,(arm,iv) in enumerate([('LR','none'),('LR','l2'),('SNA','none'),('SNA','l2')]):
  f=df[(df.arm==arm)&(df.iv==iv)]
  for row,c in enumerate(['mean','star','within']):
   ax=axs[row,col]
   for seed,g in f.groupby('seed'):ax.plot(g.task,g[c],lw=.8,alpha=.5,label=f'seed {seed}')
   med=f.groupby('task')[c].median();ax.plot(med.index,med.values,color='black',lw=1.7,label='median')
   ax.set_title(arm+(' + L2' if iv=='l2' else ''));ax.set_xlabel('Task (end of task)')
   ax.set_ylabel({'mean':'Mean preactivation','star':'Setpoint star','within':'Mean within-unit variance'}[c])
 axs[0,0].legend(fontsize=8)
 fig.suptitle('Task 20 to 100: position and width at the same task-end phase\nSame CPU trajectories as early/late audits; every task contains 625 updates')
 for ext in ['png','pdf']:fig.savefig(OUT/('formation_20_100.'+ext),dpi=160)
 plt.close(fig)
 fig,axs=plt.subplots(1,4,figsize=(15,4),layout='constrained')
 for ax,(arm,iv) in zip(axs,[('LR','none'),('LR','l2'),('SNA','none'),('SNA','l2')]):
  f=df[(df.arm==arm)&(df.iv==iv)]
  for c,color in [('q10','#0072B2'),('q50','black'),('q90','#D55E00')]:
   pp=f.pivot(index='seed',columns='task',values=c);a=pp.values
   ax.plot(pp.columns,np.median(a,0),color=color,label=c)
   ax.fill_between(pp.columns,a.min(0),a.max(0),color=color,alpha=.1)
  ax.set_title(arm+(' + L2' if iv=='l2' else ''));ax.set_xlabel('Task (end of task)');ax.set_ylabel('Quantile of unit mean preactivations')
 axs[0].legend()
 for ext in ['png','pdf']:fig.savefig(OUT/('unit_distribution.'+ext),dpi=160)
 plt.close(fig)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--iv');ap.add_argument('--seed',type=int);ap.add_argument('--all',action='store_true');a=ap.parse_args()
 if a.all:
  jobs=[(arm,iv,s) for arm in ['LR','SNA'] for iv in ['none','l2'] for s in range(3)]
  def job(j):subprocess.run([sys.executable,'-m','src.task20to100_0908','--arm',j[0],'--iv',j[1],'--seed',str(j[2])],check=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for _ in pool.map(job,jobs):pass
  report()
 else:run(a.arm,a.iv,a.seed)

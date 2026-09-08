"""Registered descriptive replay of steps20..100; no intervention."""
from pathlib import Path
import time,json,hashlib
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_early_source_0908 as E
from src import boundary_transport_0908 as T
ROOT=E.ROOT;OUT=ROOT/'results/boundary_20to100_0908'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def values(p,xmu,mu):
 w=p[0].double();b=p[1].double()
 sw=w.sum(1)*mu
 return {'mean':float((w@xmu+b).mean()),'star':float((sw+b).mean()),'star_W':float(sw.mean()),'bias':float(b.mean())}
def main():
 started=time.monotonic();torch.set_num_threads(1);E.H.setup('cpu')
 OUT.mkdir(parents=True,exist_ok=True);mnist=E.H.Mnist(torch.device('cpu'))
 rows=[];checks=[];hashes={}
 for arm in ['LR','SNA']:
  for iv in ['none','l2']:
   for seed in range(3):
    name=f'{arm}_{iv}_s{seed}';source=ROOT/'results/boundary_early_0908/source'
    sp=source/(name+'_states.pt');npz=source/(name+'.npz')
    hashes[str(sp.relative_to(ROOT))]=sha(sp);hashes[str(npz.relative_to(ROOT))]=sha(npz)
    saved=torch.load(sp,weights_only=False,map_location='cpu');dense=np.load(npz)
    for j,raw in enumerate(saved['boundaries']):
     assert time.monotonic()-started<900
     p,act,adam=T.G.clone_state(raw,arm)
     px=mnist.test_x[saved['probe_indices']][:,raw['perm']]
     xmu=px.double().mean(0);mu=saved['mu_mean']
     ids=dict(arm=arm,iv=iv,seed=seed,task=raw['task'],bin='2-6' if j<5 else '7-11' if j<10 else '12-20')
     rows.append(dict(**ids,step=0,**values(p,xmu,mu)))
     gd=torch.Generator();gd.set_state(raw['rng_after_perm']['data'])
     gb=torch.Generator();gb.set_state(raw['rng_after_perm']['batch'])
     idx=E.H.stratified_draw(mnist,gd);order=torch.randperm(E.H.TASK_EXAMPLES,generator=gb)
     xs=mnist.train_x[idx][:,raw['perm']][order];ys=mnist.train_y[idx][order]
     for step in range(100):
      out=E.H.forward(p,xs[16*step:16*(step+1)],act)
      gr=torch.autograd.grad(torch.nn.functional.cross_entropy(out[4],ys[16*step:16*(step+1)]),p)
      with torch.no_grad():
       if iv=='l2':gr=[g+2*.001*q for g,q in zip(gr,p)]
       m,v,tc=adam;tc[0]+=1;c1=1-.9**tc[0];c2=1-.999**tc[0]
       for q,g,mi,vi in zip(p,gr,m,v):
        mi.mul_(.9).add_(g,alpha=1-.9);vi.mul_(.999).addcmul_(g,g,value=1-.999)
        q-=.001*(mi/c1)/((vi/c2).sqrt()+1e-8)
       if arm=='SNA':act.update(out[0],out[2])
       n=step+1
       if n>=20 and n%5==0:rows.append(dict(**ids,step=n,**values(p,xmu,mu)))
       if n in [20,100]:
        z=E.H.forward(p,px,act)[0].mean(0).numpy()
        error=float(abs(z-dense['dense1'][j,n-1]).max())
        assert error<=2e-5,(name,raw['task'],n,error)
        exact=None
        if n==20:
         old=raw['after_20']['state']
         exact=all(torch.equal(a,b) for a,b in zip(p,old['params']))
         if arm=='SNA':exact=exact and all(torch.equal(a,b) for a,b in zip(act.V,old['V']))
         assert exact
        checks.append(dict(**ids,step=n,z_replay_maxabs=error,checkpoint_exact=exact))
     rows.append(dict(**ids,step=625,**values(raw['after_625']['state']['params'],xmu,mu)))
    print('DONE',name,round(time.monotonic()-started,1),flush=True)
 df=pd.DataFrame(rows);df.to_csv(OUT/'trajectory.csv',index=False,lineterminator='\n')
 pd.DataFrame(checks).to_csv(OUT/'validation.csv',index=False,lineterminator='\n')
 delta=[]
 for key,g in df.groupby(['arm','iv','seed','task','bin']):
  g=g.set_index('step')
  for lo,hi in [(0,20),(20,100),(0,100),(100,625),(0,625)]:
   delta.append(dict(zip(['arm','iv','seed','task','bin'],key),window=f'{lo}-{hi}',**{k:g.loc[hi,k]-g.loc[lo,k] for k in ['mean','star','star_W','bias']}))
 d=pd.DataFrame(delta);d.to_csv(OUT/'boundary_delta.csv',index=False,lineterminator='\n')
 keys=['arm','iv','bin','window'];cols=['mean','star','star_W','bias']
 sd=d.groupby(keys+['seed'])[cols].mean().reset_index()
 sd.to_csv(OUT/'seed_delta.csv',index=False,lineterminator='\n')
 verdict=[]
 for key,g in sd.groupby(keys):
  row=dict(zip(keys,key))
  for c in cols:row.update({c+'_median':g[c].median(),c+'_min':g[c].min(),c+'_max':g[c].max(),c+'_n_down':int((g[c]<-1e-6).sum())})
  verdict.append(row)
 vv=pd.DataFrame(verdict);vv.to_csv(OUT/'verdict.csv',index=False,lineterminator='\n')
 fig,axs=plt.subplots(2,4,figsize=(15,7),layout='constrained')
 for col,(arm,iv) in enumerate([('LR','none'),('LR','l2'),('SNA','none'),('SNA','l2')]):
  for row,c in enumerate(['mean','star']):
   ax=axs[row,col]
   for bn,color in [('2-6','#D55E00'),('7-11','#009E73'),('12-20','#0072B2')]:
    g=df[(df.arm==arm)&(df.iv==iv)&(df.bin==bn)&(df.step<=100)]
    pp=g.groupby(['seed','step'])[c].mean().unstack()
    pp=pp.sub(pp[20],axis=0);a=pp.values
    ax.plot(pp.columns,np.median(a,axis=0),color=color,label='Tasks '+bn)
    ax.fill_between(pp.columns,a.min(0),a.max(0),color=color,alpha=.12)
   ax.axhline(0,color='gray',lw=.7);ax.axvline(20,color='gray',ls=':',lw=.8)
   ax.set_title(arm+(' + L2' if iv=='l2' else ''))
   ax.set_xlabel('Updates after switch');ax.set_ylabel(c+' - value at update 20')
 axs[0,0].legend(fontsize=8)
 fig.suptitle('Does descent occur between updates 20 and 100?\nSame fixed inputs within each task; each line: seed median of task-bin means, shade: seed range')
 for ext in ['png','pdf']:fig.savefig(OUT/('trajectory.'+ext),dpi=160)
 plt.close(fig)
 show=vv[(vv.window=='20-100')][keys+['mean_median','mean_min','mean_max','mean_n_down','star_median','star_n_down']]
 def md(f):
  lines=['| '+' | '.join(f.columns)+' |','|'+'|'.join(['---']*len(f.columns))+'|']
  for _,r in f.iterrows():lines.append('| '+' | '.join(f'{x:.6g}' if isinstance(x,float) else str(x) for x in r)+' |')
  return '\n'.join(lines)
 focus=vv[(vv.arm=='LR')&(vv.iv=='none')&(vv.bin=='12-20')]
 summary='# 切替後20〜100更新の下降確認（既存CPU軌道の追加再生）\n\n境界差→seed内ビン平均→3seed中央値。下向きseed数は差 < -1e-6。独立確証走・因果帰属ではない。\n\n'+md(show)+'\n\n## LR none / task12..20 全窓\n\n'+md(focus[keys+['mean_median','mean_min','mean_max','mean_n_down','star_median','star_min','star_max','star_n_down']])+'\n'
 (OUT/'summary.md').write_text(summary)
 (OUT/'provenance.json').write_text(json.dumps({'source_hashes':hashes,'code_sha256':sha(Path(__file__)),'spec_sha256':sha(ROOT/'specs/spec_boundary_20to100_0908.md'),'data_sha256':mnist.sha256,'seconds':time.monotonic()-started,'max_replay_error':max(c['z_replay_maxabs'] for c in checks),'exact20_count':sum(c['checkpoint_exact'] is True for c in checks)},indent=2))
 print(summary,flush=True)
if __name__=='__main__':main()

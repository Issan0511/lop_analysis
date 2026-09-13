"""Posthoc fixed 20-task bin comparison of existing results; no training."""
from pathlib import Path
import pandas as pd
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/task20to100_0908/bins20_posthoc_0909';OUT.mkdir(parents=True,exist_ok=True)
early=ROOT/'results/boundary_early_0908/first_task.csv'
later=ROOT/'results/task20to100_0908/trajectory.csv'
a=pd.read_csv(early);a=a[a.step==0].copy();a['task']=0
d=pd.concat([a,pd.read_csv(later)],ignore_index=True)
rows=[]
for (arm,iv,seed),g in d.groupby(['arm','iv','seed']):
 g=g.set_index('task')
 for lo,hi in [(0,20),(20,40),(40,60),(60,80),(80,100)]:
  rows.append(dict(arm=arm,iv=iv,seed=seed,start=lo,end=hi,window=f'{lo}-{hi}',
                   mean=g.loc[hi,'mean']-g.loc[lo,'mean'],star=g.loc[hi,'star']-g.loc[lo,'star']))
s=pd.DataFrame(rows);s.to_csv(OUT/'seed_delta.csv',index=False,lineterminator='\n')
v=s.groupby(['arm','iv','start','end','window']).agg(mean=('mean','median'),mean_min=('mean','min'),mean_max=('mean','max'),star=('star','median')).reset_index()
v.to_csv(OUT/'verdict.csv',index=False,lineterminator='\n')
best=[]
for scope,ss in [('0-100',s),('20-100',s[s.start>=20])]:
 for (arm,iv,seed),g in ss.groupby(['arm','iv','seed']):
  for metric in ['mean','star']:
   r=g.loc[g[metric].idxmin()]
   best.append(dict(scope=scope,arm=arm,iv=iv,seed=seed,metric=metric,window=r.window,delta=r[metric]))
b=pd.DataFrame(best);b.to_csv(OUT/'seed_steepest.csv',index=False,lineterminator='\n')
def md(f):
 return '\n'.join(['| '+' | '.join(f.columns)+' |','|'+'|'.join(['---']*len(f.columns))+'|']+['| '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in r)+' |' for _,r in f.iterrows()])
summary='# 20タスク刻みの下降量比較（事後集計・0909）\n\n新しい走はなし。seed内の同じ量の差→3seed中央値。0はtask1入力上の初期化時点、20以降は各task終端。0→20だけは最初の学習を含み、切替は19回。切替を20回含む後続ビンと機構的に同一視しない。最大下降区間の順位は記述で、有意差や活性化固有の時期の証明ではない。\n\n'+md(v[['arm','iv','window','mean','mean_min','mean_max','star']])+'\n\n## seed別の最も下がる区間\n\n'+md(b)+'\n'
(OUT/'summary.md').write_text(summary)
(OUT/'provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [early,later,Path(__file__)]},indent=2))
print(summary)

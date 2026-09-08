"""Aggregation and pre-registered verdicts for ce_factor_swap_0909."""
from pathlib import Path
import json,itertools
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_gradient_0908 as G
OUT=G.ROOT/'results/ce_factor_swap_0909'
ARMS=[('LR','none'),('SNA','none'),('LR','l2'),('SNA','l2')]
FACTORS=['r','J','g','D']
WINDOWS=[('w1',1,1),('w2_5',2,5),('w6_10',6,10),('w11_20',11,20),('w21_30',21,30),('w31_40',31,40)]
def load(kind):
 fs=[OUT/f'{a}_{i}_s{s}_{kind}.csv' for a,i in ARMS for s in range(3)]
 return pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
def sign(x,tol=0.):return 0 if abs(x)<=tol else (1 if x>0 else -1)
def main():
 steps=load('steps');swaps=load('swaps')
 # ---- per-step course: boundary median within seed, then the 3 seeds ----
 cols=['diag_net','diag_U','diag_D','diag_net_pos','diag_net_neg','diag_frozenD','diag_sgd',
       'train_current','train_history','train_actual','diag_block_negfrac']
 seedstep=steps.groupby(['arm','iv','seed','step'])[cols].median().reset_index()
 seedstep.to_csv(OUT/'seed_steps.csv',index=False)
 course=seedstep.groupby(['arm','iv','step'])[cols].median().reset_index()
 course.to_csv(OUT/'course.csv',index=False)
 # ---- windows: cumulate within window per boundary, then aggregate ----
 rows=[]
 for (a,i,s,t),grp in steps.groupby(['arm','iv','seed','task']):
  for name,lo,hi in WINDOWS:
   w=grp[(grp.step>=lo)&(grp.step<=hi)]
   rows.append(dict(arm=a,iv=i,seed=s,task=t,window=name,
     **{c:float(w[c].sum()) for c in ['diag_net','diag_U','diag_D','diag_frozenD','diag_sgd',
                                      'train_current','train_history','train_actual']}))
 bw=pd.DataFrame(rows);bw.to_csv(OUT/'boundary_windows.csv',index=False)
 sw=bw.groupby(['arm','iv','seed','window']).agg(['median','mean']).reset_index()
 sw.columns=['_'.join(c).rstrip('_') for c in sw.columns]
 sw=sw.drop(columns=[c for c in sw.columns if c.startswith('task_')])
 sw.to_csv(OUT/'seed_windows.csv',index=False)
 # ---- pre-registered verdicts ----
 cellmed=swaps.groupby(['arm','iv','seed','pair','cell'])['net'].median().reset_index()
 cellmed.to_csv(OUT/'seed_cells.csv',index=False)
 out=[]
 for (a,i),pair in itertools.product(ARMS,['1_15','1_40']):
  sub=cellmed[(cellmed.arm==a)&(cellmed.iv==i)&(cellmed.pair==pair)]
  get=lambda c:[float(sub[(sub.seed==s)&(sub.cell==c)]['net'].iloc[0]) for s in range(3)]
  A=get('AAAA');Bc=get('BBBB')
  row=dict(arm=a,iv=i,pair=pair,
   N_A_med=float(np.median(A)),N_B_med=float(np.median(Bc)),
   N_A_seeds=';'.join(f'{v:.6g}' for v in A),N_B_seeds=';'.join(f'{v:.6g}' for v in Bc),
   corner_sign_flip=int(all(sign(x)!=sign(y) for x,y in zip(A,Bc))))
  suff=[];nec=[]
  for k,f in enumerate(FACTORS):
   one=''.join('B' if j==k else 'A' for j in range(4))
   rest=''.join('A' if j==k else 'B' for j in range(4))
   v1=get(one);v2=get(rest)
   row[f'suff_{f}_med']=float(np.median(v1));row[f'nec_{f}_med']=float(np.median(v2))
   s1=all(sign(x)!=sign(y) and sign(x)!=0 for x,y in zip(v1,A))
   s2=all(sign(x)!=sign(y) and sign(x)!=0 for x,y in zip(v2,Bc))
   row[f'SUFFICIENT_{f}']=int(s1);row[f'NECESSARY_{f}']=int(s2)
   if s1:suff.append(f)
   if s2:nec.append(f)
  both=[f for f in FACTORS if f in suff and f in nec]
  row['verdict']=('SINGLE_FACTOR_'+both[0] if len(both)==1 else
                  'MULTI_FACTOR:'+'+'.join(both) if len(both)>1 else 'INTERACTION_ONLY')
  row['sufficient']='+'.join(suff) or '-';row['necessary']='+'.join(nec) or '-'
  for extra in ['J:gate2_only','J:weights_only','alpha_only']:
   if extra in set(sub.cell):row[extra.replace(':','_')]=float(np.median(get(extra)))
  # U/D attribution between the two states of the pair
  ta,tb=(int(x) for x in pair.split('_'))
  ss=seedstep[(seedstep.arm==a)&(seedstep.iv==i)]
  dU=[float(ss[(ss.seed==s)&(ss.step==tb)]['diag_U'].iloc[0]-ss[(ss.seed==s)&(ss.step==ta)]['diag_U'].iloc[0]) for s in range(3)]
  dD=[float(ss[(ss.seed==s)&(ss.step==tb)]['diag_D'].iloc[0]-ss[(ss.seed==s)&(ss.step==ta)]['diag_D'].iloc[0]) for s in range(3)]
  mU,mD=float(np.median(dU)),float(np.median(dD))
  row.update(dU_med=mU,dD_med=mD,dnet_med=mU-mD,
   ud_label='U_DOMINANT' if abs(mU)>2*abs(mD) else ('D_DOMINANT' if abs(mD)>2*abs(mU) else 'BOTH'))
  out.append(row)
 verdict=pd.DataFrame(out);verdict.to_csv(OUT/'verdict.csv',index=False)
 # ---- gate G1 on the primary arm ----
 g1=[]
 for a,i in ARMS:
  for st in (15,40):
   v=[float(seedstep[(seedstep.arm==a)&(seedstep.iv==i)&(seedstep.seed==s)&(seedstep.step==st)]['diag_net'].iloc[0]) for s in range(3)]
   t=[float(seedstep[(seedstep.arm==a)&(seedstep.iv==i)&(seedstep.seed==s)&(seedstep.step==st)]['train_current'].iloc[0]) for s in range(3)]
   g1.append(dict(arm=a,iv=i,step=st,diag_seeds=';'.join(f'{x:.6g}' for x in v),
    diag_all_negative=int(all(x<0 for x in v)),train_seeds=';'.join(f'{x:.6g}' for x in t),
    train_all_negative=int(all(x<0 for x in t))))
 gate=pd.DataFrame(g1);gate.to_csv(OUT/'gate_g1.csv',index=False)
 g1pass=bool(gate[(gate.arm=='LR')&(gate.iv=='none')&(gate.step==15)]['diag_all_negative'].iloc[0])
 # ---- validation ----
 vr=[]
 for a,i in ARMS:
  for s in range(3):
   m=json.loads((OUT/f'{a}_{i}_s{s}_provenance.json').read_text())
   c=m['checks'][0];agg={k:max(x[k] for x in m['checks']) for k in ['delta','additive','replay','ud','corner']}
   agg.update({k:min(x[k] for x in m['checks']) for k in ['delta_mutctl','additive_mutctl','ud_mutctl','corner_mutctl']})
   vr.append(dict(arm=a,iv=i,seed=s,wall_seconds=round(m['wall_seconds'],1),**agg,
    diag_noninvasive=c.get('diag_noninvasive'),diag_noninvasive_mutctl=c.get('diag_noninvasive_mutctl'),
    replay_mutctl=c.get('replay_mutctl')))
 pd.DataFrame(vr).to_csv(OUT/'validation.csv',index=False)
 # ---- figures ----
 fig,ax=plt.subplots(2,2,figsize=(11,7),sharex=True)
 for k,(a,i) in enumerate(ARMS):
  x=ax[k//2][k%2];c=course[(course.arm==a)&(course.iv==i)]
  x.axhline(0,color='k',lw=.6)
  x.plot(c.step,c.diag_net,label='diagnostic current CE',color='C0')
  x.plot(c.step,c.train_current,label='training-batch current CE',color='C1',ls='--')
  x.plot(c.step,c.diag_U,label='U (up)',color='C2',lw=.8)
  x.plot(c.step,-c.diag_D,label='-D (down)',color='C3',lw=.8)
  x.set_title(f'{a} {i}');x.set_xlabel('update after switch')
  if k==0:x.legend(fontsize=7)
 fig.suptitle('ce_factor_swap_0909: per-update projected current-CE displacement (task21..40, boundary median, seed median)')
 fig.tight_layout()
 for e in ('png','pdf'):fig.savefig(OUT/f'course.{e}',dpi=140)
 plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(12,4.2))
 for k,pair in enumerate(['1_15','1_40']):
  sub=cellmed[(cellmed.arm=='LR')&(cellmed.iv=='none')&(cellmed.pair==pair)]
  cells=[''.join(c) for c in itertools.product('AB',repeat=4)]
  vals=[float(np.median([sub[(sub.seed==s)&(sub.cell==c)]['net'].iloc[0] for s in range(3)])) for c in cells]
  o=np.argsort([c.count('B') for c in cells],kind='stable')
  ax[k].bar(range(16),[vals[j] for j in o],color=['C0' if vals[j]>0 else 'C3' for j in o])
  ax[k].set_xticks(range(16));ax[k].set_xticklabels([cells[j] for j in o],rotation=90,fontsize=7)
  ax[k].axhline(0,color='k',lw=.6);ax[k].set_title(f'LR none, factors (r,J,g,D) from updates {pair.replace("_"," / ")}')
  ax[k].set_ylabel('projected current-CE dzbar')
 fig.tight_layout()
 for e in ('png','pdf'):fig.savefig(OUT/f'swap_table.{e}',dpi=140)
 plt.close(fig)
 print('G1_PASS' if g1pass else 'G1_FAIL')
 print(verdict[['arm','iv','pair','N_A_med','N_B_med','sufficient','necessary','verdict','ud_label']].to_string(index=False))
if __name__=='__main__':main()

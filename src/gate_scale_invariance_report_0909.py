"""Pre-registered verdicts for gate_scale_invariance_0909."""
import json
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_gradient_0908 as G
OUT=G.ROOT/'results/gate_scale_invariance_0909'
ARMS=['R','LR03','LR','LR001','SN02','SN06','SN15','ELU1']
G1=['R','LR03','LR001'];G2=['SN02','SN06','SN15']
def sp(x,y):
 x=pd.Series(np.asarray(x,float)).rank();y=pd.Series(np.asarray(y,float)).rank();return float(np.corrcoef(x,y)[0,1])
def main():
 d=pd.concat([pd.read_csv(OUT/f'{a}_s{s}_rows.csv') for a in ARMS for s in range(3)],ignore_index=True)
 rows=[]
 for a in ARMS:
  for s in range(3):
   x=d[(d.arm==a)&(d.seed==s)].sort_values('task');t=x.task.to_numpy().astype(float)
   m=t>=60;p_late=float(np.polyfit(np.log(t[m]),np.log(x.N.to_numpy()[m]),1)[0])
   w1=x[(x.task>=21)&(x.task<=40)];w2=x[(x.task>=101)&(x.task<=120)]
   beta=float(np.log(w2.D2.mean()/w1.D2.mean())/np.log(w2.N.mean()/w1.N.mean()))
   tr=x[x.kappa_g.notna()]
   late=tr[tr.task>=101]
   rows.append(dict(arm=a,seed=s,p_late=p_late,beta=beta,kappa_g=float(late.kappa_g.mean()),
    persist=float(late.persist.mean()),D2_early=float(w1.D2.mean()),D2_late=float(w2.D2.mean()),
    N_early=float(w1.N.mean()),N_late=float(w2.N.mean()),cos_late=float(w2.cos.mean()),
    dN2_early=float(w1.D2.mean()+w1.erode.mean()),dN2_late=float(w2.D2.mean()+w2.erode.mean()),
    N120=float(x[x.task==120].N.iloc[0]),acc120=float(x[x.task==120].acc.iloc[0]),
    E_late=float(x[(x.task>=100)&x.E.notna()].E.median()) if x.E.notna().any() else np.nan))
 sv=pd.DataFrame(rows);sv.to_csv(OUT/'seed_verdict.csv',index=False)
 g=lambda a,f:[float(sv[(sv.arm==a)&(sv.seed==s)][f].iloc[0]) for s in range(3)]
 V={}
 # V1
 ok=all(.44<=v<=.52 for a in G1 for v in g(a,'p_late')) and all(.45<=v<=.65 for a in G1 for v in g(a,'beta'))
 bad=any(all(v<.42 for v in g(a,'p_late')) or all(v<.40 for v in g(a,'beta')) for a in G1)
 V['V1']='SCALE_INVARIANCE_DECIDES' if ok else ('SCALE_INVARIANCE_INSUFFICIENT' if bad else 'V1_PARTIAL')
 # V2
 kmono=all(g('SN02','kappa_g')[s]<g('SN06','kappa_g')[s]<g('SN15','kappa_g')[s] for s in range(3))
 if not kmono:V['V2']='V2_VOID'
 else:
  mono=all(g('SN02',f)[s]>g('SN06',f)[s]>g('SN15',f)[s] for f in ['beta','p_late'] for s in range(3))
  V['V2']='CURVATURE_MONOTONE' if mono else 'V2_PARTIAL'
 # V3, V4
 r3=[sp([g(a,'beta')[s] for a in ARMS],[g(a,'p_late')[s] for a in ARMS]) for s in range(3)]
 r4=[sp([g(a,'persist')[s] for a in ARMS],[g(a,'p_late')[s] for a in ARMS]) for s in range(3)]
 V['V3']='BETA_PREDICTS_EXPONENT' if all(v>=.85 for v in r3) else ('BETA_UNRELATED' if all(v<=.4 for v in r3) else 'V3_PARTIAL')
 V['V4']='PERSISTENCE_NOT_THE_MECHANISM' if all(r4[s]<=r3[s]-.2 for s in range(3)) else 'V4_NOT_SHOWN'
 V['spearman_beta_p']=';'.join(f'{v:+.2f}' for v in r3);V['spearman_persist_p']=';'.join(f'{v:+.2f}' for v in r4)
 pd.DataFrame([V]).to_csv(OUT/'verdict.csv',index=False)
 vr=[]
 for a in ARMS:
  for s in range(3):
   m=json.loads((OUT/f'{a}_s{s}_provenance.json').read_text());c=m['checks']
   vr.append(dict(arm=a,seed=s,**{k:c.get(k) for k in ['g1_maxabs','g1_n','g1_mutctl','ident','ident_mut','hzero','decomp','decomp_mut','dphi','d2phi','h','dphi_mutctl','d2phi_mutctl','h_mutctl','meas_noninv','meas_mutctl']},wall=round(m['wall_seconds'],1)))
 pd.DataFrame(vr).to_csv(OUT/'validation.csv',index=False)
 fig,ax=plt.subplots(1,3,figsize=(15,4.5))
 col={'R':'k','LR03':'C0','LR':'C1','LR001':'C2','SN02':'C3','SN06':'C4','SN15':'C5','ELU1':'C6'}
 for a in ARMS:
  x=d[d.arm==a].groupby('task').N.median();ax[0].loglog(x.index,x.values,color=col[a],label=a)
  ax[1].scatter(np.median(g(a,'kappa_g')),np.median(g(a,'beta')),color=col[a],s=60,label=a)
  ax[2].scatter(np.median(g(a,'beta')),np.median(g(a,'p_late')),color=col[a],s=60,label=a)
 ax[0].set_xlabel('task');ax[0].set_ylabel('||W~||');ax[0].set_title('width growth');ax[0].legend(fontsize=7)
 ax[1].set_xscale('symlog',linthresh=1e-3);ax[1].set_xlabel('kappa_g = E|z phi\'\'|');ax[1].set_ylabel('beta');ax[1].set_title('curvature vs injection scaling')
 ax[2].set_xlabel('beta');ax[2].set_ylabel('p_late');ax[2].set_title('injection scaling vs exponent');ax[2].legend(fontsize=7)
 fig.suptitle('gate_scale_invariance_0909 (seed median)');fig.tight_layout()
 for e in ('png','pdf'):fig.savefig(OUT/f'course.{e}',dpi=140)
 print(V)
 pd.set_option('display.width',250)
 agg=sv.groupby('arm')[['p_late','beta','kappa_g','persist','cos_late','D2_early','D2_late','N120','acc120']].median().reindex(ARMS)
 print(agg.to_string(float_format=lambda v:f'{v:.4f}'))
if __name__=='__main__':main()

"""Pre-registered verdicts for elu_growth_0909."""
import json
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_gradient_0908 as G
OUT=G.ROOT/'results/elu_growth_0909'
ARMS=[('LR','none'),('SNA','none'),('ELU1','none'),('ELU03','none'),('ELU1','l2')]
def expo(t,y):
 m=(t>=1)&(y>0);return float(np.polyfit(np.log(t[m]),np.log(y[m]),1)[0])
def main():
 d=pd.concat([pd.read_csv(OUT/f'{a}_{i}_s{s}_rows.csv') for a,i in ARMS for s in range(3)],ignore_index=True)
 e=d[d.phase=='end'].copy()
 seedrows=[];armrows=[]
 for a,i in ARMS:
  per={}
  for s in range(3):
   x=e[(e.arm==a)&(e.iv==i)&(e.seed==s)].sort_values('task')
   t=x.task.to_numpy().astype(float);cn=x.cnorm.to_numpy();zb=x.zbar_inv.to_numpy();sg=x.sigma_inv.to_numpy()
   u=np.load(OUT/f'{a}_{i}_s{s}_units.npz')['cnorm_i']          # (120,100)
   m20=t>=20
   pe=[expo(t[(t>=5)&(t<=30)],cn[(t>=5)&(t<=30)]),expo(t[(t>=60)],cn[(t>=60)])]
   ux=[expo(t[m20],u[m20,k]) for k in range(u.shape[1])]
   sat20=float(x[x.task==20]['sat'].iloc[0]);sat120=float(x[x.task==120]['sat'].iloc[0])
   sl=lambda lo,hi:float(np.polyfit(t[(t>=lo)&(t<=hi)],zb[(t>=lo)&(t<=hi)],1)[0])
   A,B=np.polyfit(np.sqrt(t[m20]),zb[m20],1);r2=1-((zb[m20]-(A*np.sqrt(t[m20])+B))**2).mean()/zb[m20].var()
   k,_=np.polyfit(sg[m20],zb[m20],1);kr2=1-((zb[m20]-np.polyval(np.polyfit(sg[m20],zb[m20],1),sg[m20]))**2).mean()/zb[m20].var()
   Ev=x[x.E.notna()][['task','E','r_W2','r_scale']]
   row=dict(arm=a,iv=i,seed=s,p_early=pe[0],p_late=pe[1],
    unit_iqr=float(np.percentile(ux,75)-np.percentile(ux,25)),unit_med=float(np.median(ux)),
    frozen=int(sum(v<0.10 for v in ux)),sat20=sat20,sat120=sat120,
    slope_early=sl(20,60),slope_late=sl(80,120),sqrt_r2=float(r2),k=float(k),k_r2=float(kr2),
    E_neg=int((Ev.E<0).sum()),E_n=len(Ev),E_med=float(Ev.E.median()),
    E_rel_med=float((Ev.E.abs()/Ev.r_W2.abs().clip(lower=1e-9)).median()),
    cnorm120=float(cn[-1]),zbar120=float(zb[-1]),sigma120=float(sg[-1]),
    acc_early=float(x[(x.task>=2)&(x.task<=6)]['acc'].mean()),acc_late=float(x[x.task>=116]['acc'].mean()))
   row['r_slope']=row['slope_late']/row['slope_early'] if row['slope_early']!=0 else np.nan
   row['acc_drop']=row['acc_late']-row['acc_early'];per[s]=row;seedrows.append(row)
  g=lambda f:[per[s][f] for s in range(3)]
  V=dict(arm=a,iv=i,p_late=';'.join(f'{v:.3f}' for v in g('p_late')),p_early=';'.join(f'{v:.3f}' for v in g('p_early')))
  pl=g('p_late')
  V['V1']=('EXPONENT_HALF' if all(.44<=v<=.52 for v in pl) else 'EXPONENT_SUB' if all(v<.40 for v in pl)
           else 'EXPONENT_SUPER' if all(v>.56 for v in pl) else 'EXPONENT_MIXED')
  V['decelerates']=int(all(per[s]['p_late']<per[s]['p_early']-.10 for s in range(3)))
  iq=g('unit_iqr');V['unit_iqr']=';'.join(f'{v:.3f}' for v in iq);V['frozen']=';'.join(str(v) for v in g('frozen'))
  V['V2']='UNIT_UNIMODAL' if all(v<.10 for v in iq) else 'UNIT_SPLIT' if all(v>.20 for v in iq) else 'UNIT_PARTIAL'
  V['sat20']=';'.join(f'{v:.3f}' for v in g('sat20'));V['sat120']=';'.join(f'{v:.3f}' for v in g('sat120'))
  V['V3']=('SATURATION_GROWS' if all(per[s]['sat120']>=.20 and per[s]['sat120']>2*per[s]['sat20'] for s in range(3))
           else 'SATURATION_FLAT' if all(abs(per[s]['sat120']-per[s]['sat20'])<.05 for s in range(3)) else 'SATURATION_PARTIAL')
  rs=g('r_slope');V['r_slope']=';'.join(f'{v:+.2f}' for v in rs);V['sqrt_r2']=';'.join(f'{v:.3f}' for v in g('sqrt_r2'))
  V['V4']=('SINK_SQRT_LIKE' if all(per[s]['sqrt_r2']>.90 and .4<=per[s]['r_slope']<=.9 for s in range(3))
           else 'SINK_SATURATES' if all(v<.25 for v in rs) else 'SINK_PARTIAL')
  V['k']=';'.join(f'{v:+.2f}' for v in g('k'));V['k_r2']=';'.join(f'{v:.2f}' for v in g('k_r2'))
  neg=sum(g('E_neg'));tot=sum(g('E_n'))
  V['E_neg']=f'{neg}/{tot}';V['E_med']=';'.join(f'{v:+.4f}' for v in g('E_med'));V['E_rel']=';'.join(f'{v:.3f}' for v in g('E_rel_med'))
  V['V5']='DEFECT_OUTWARD' if neg>=tot*15/18 else 'DEFECT_RESTORING' if (tot-neg)>=tot*15/18 else 'DEFECT_MIXED'
  if i=='l2':V['V6']='WD_PINS_ELU' if all(-.05<=v<=.05 for v in pl) else 'WD_DOES_NOT_PIN'
  V['cnorm120']=';'.join(f'{v:.2f}' for v in g('cnorm120'));V['zbar120']=';'.join(f'{v:+.2f}' for v in g('zbar120'))
  V['acc_drop_pt']=';'.join(f'{v*100:+.2f}' for v in g('acc_drop'))
  armrows.append(V)
 pd.DataFrame(seedrows).to_csv(OUT/'seed_verdict.csv',index=False)
 ver=pd.DataFrame(armrows);ver.to_csv(OUT/'verdict.csv',index=False)
 vr=[]
 for a,i in ARMS:
  for s in range(3):
   m=json.loads((OUT/f'{a}_{i}_s{s}_provenance.json').read_text());c=m['checks']
   vr.append(dict(arm=a,iv=i,seed=s,wall=round(m['wall_seconds'],1),**{k:c.get(k) for k in
     ['g1_maxabs','g1_n','g1_mutctl','identity_rel','identity_mutctl','leaky_zero','measure_noninvasive','measure_mutctl','elu_dphi','elu_dphi_mutctl']}))
 pd.DataFrame(vr).to_csv(OUT/'validation.csv',index=False)
 fig,ax=plt.subplots(2,2,figsize=(12,7.5))
 col={'LR none':'k','SNA none':'C2','ELU1 none':'C3','ELU03 none':'C1','ELU1 l2':'C0'}
 for a,i in ARMS:
  lab=f'{a} {i}';cs=[];zs=[];ss=[];sat=[]
  for s in range(3):
   x=e[(e.arm==a)&(e.iv==i)&(e.seed==s)].sort_values('task')
   cs.append(x.cnorm.to_numpy());zs.append(x.zbar_inv.to_numpy());ss.append(x.sigma_inv.to_numpy());sat.append(x.sat.to_numpy())
  t=x.task.to_numpy()
  ax[0][0].loglog(t,np.median(cs,0),color=col[lab],label=lab)
  ax[0][1].plot(t,np.median(zs,0),color=col[lab],label=lab)
  ax[1][0].plot(t,np.median(sat,0),color=col[lab],label=lab)
  ax[1][1].plot(np.median(ss,0),np.median(zs,0),color=col[lab],label=lab)
 ax[0][0].set_xlabel('task');ax[0][0].set_ylabel('||W~|| (unit mean)');ax[0][0].set_title('width growth (log-log)');ax[0][0].legend(fontsize=7)
 for x_,y_ in [(30,None)]:pass
 ax[0][1].set_xlabel('task');ax[0][1].set_ylabel('zbar_inv');ax[0][1].set_title('sinking')
 ax[1][0].set_xlabel('task');ax[1][0].set_ylabel("frac phi' < 0.05");ax[1][0].set_title('saturation')
 ax[1][1].set_xlabel('sigma_inv');ax[1][1].set_ylabel('zbar_inv');ax[1][1].set_title('depth vs width')
 fig.suptitle('elu_growth_0909: task ends, seed median');fig.tight_layout()
 for ext in ('png','pdf'):fig.savefig(OUT/f'course.{ext}',dpi=140)
 print(ver[['arm','iv','V1','p_late','V2','V3','V4','V5','E_neg']].to_string(index=False))
if __name__=='__main__':main()

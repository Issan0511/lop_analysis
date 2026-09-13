"""Pre-registered verdicts for linear_growth_0910 (spec §3)."""
import json
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import boundary_gradient_0908 as G
OUT=G.ROOT/'results/linear_growth_0910'
ARMS=['LIN','LIN0','LR']
def main():
 d=pd.concat([pd.read_csv(OUT/f'{a}_s{s}_rows.csv') for a in ARMS for s in range(3)],ignore_index=True)
 rows=[]
 for a in ARMS:
  for s in range(3):
   x=d[(d.arm==a)&(d.seed==s)].sort_values('task');t=x.task.to_numpy().astype(float)
   m=t>=60;p_late=float(np.polyfit(np.log(t[m]),np.log(x.N.to_numpy()[m]),1)[0])
   tr=x[x.zbar_inv.notna()]
   z20=float(tr[tr.task==20].zbar_inv.iloc[0]);z120=float(tr[tr.task==120].zbar_inv.iloc[0])
   tt=tr[tr.task>=20];sl,ic=np.polyfit(np.log(tt.N),np.log(tt.sigma_inv),1)
   r2=1-((np.log(tt.sigma_inv)-(sl*np.log(tt.N)+ic))**2).mean()/np.log(tt.sigma_inv).var()
   ratio20=float(tr[tr.task==20].sigma_inv.iloc[0]**2/tr[tr.task==20].N.iloc[0]**2)
   ratio120=float(tr[tr.task==120].sigma_inv.iloc[0]**2/tr[tr.task==120].N.iloc[0]**2)
   w=x[(x.task>=60)&(x.task<=120)]
   rows.append(dict(arm=a,seed=s,p_late=p_late,z20=z20,z120=z120,dz=z120-z20,sig_slope=float(sl),sig_r2=float(r2),
    s2n2_20=ratio20,s2n2_120=ratio120,cos_late=float(w.cos.mean()),N120=float(x[x.task==120].N.iloc[0]),
    D2_late=float(w.D2.mean()),erode_late=float(w.erode.mean()),
    acc120=float(tr[tr.task==120].acc.iloc[0]),acc20=float(tr[tr.task==20].acc.iloc[0]),
    persist_late=float(tr[tr.task>=100].persist.mean()),
    E_rel_max=float((x.E.abs()/x.r_scale).max()) if x.E.notna().any() else np.nan))
 sv=pd.DataFrame(rows);sv.to_csv(OUT/'seed_verdict.csv',index=False)
 g=lambda a,f:[float(sv[(sv.arm==a)&(sv.seed==s)][f].iloc[0]) for s in range(3)]
 V={}
 p=g('LIN','p_late')
 V['V1']='LIN_GROWS_LIKE_LEAKY' if all(.44<=v<=.52 for v in p) else 'LIN_SLOWER' if all(v<.40 for v in p) else 'LIN_FASTER' if all(v>.56 for v in p) else 'V1_PARTIAL'
 z=g('LIN','z120');dz=g('LIN','dz')
 V['V2']='NO_SINK' if all(abs(z[s])<.5 and abs(dz[s])<.3 for s in range(3)) else 'SINKS' if all(v<-1. for v in z) else 'V2_PARTIAL'
 sl=g('LIN','sig_slope');r2=g('LIN','sig_r2')
 V['V3']='SIGMA_TRACKS_WIDTH' if all(.9<=sl[s]<=1.1 and r2[s]>.95 for s in range(3)) else 'V3_PARTIAL'
 c=g('LIN0','cos_late')
 V['V4']='LIN0_OUTWARD' if all(v>0 for v in c) else 'LIN0_INWARD' if all(v<-.02 for v in c) else 'V4_PARTIAL'
 # V5: LIN の |E|/スケール（E_pred=0 なので ident と同値）と LIN0 の恒等式誤差 ident は provenance から
 idv=[json.loads((OUT/f'{a}_s{s}_provenance.json').read_text())['checks']['ident'] for a in ARMS for s in range(3)]
 V['V5_identity_ok']=int(all(v<1e-6 for v in idv));V['V5_ident_max']=f'{max(idv):.1e}'
 for a in ARMS:
  for f in ['p_late','z120','cos_late','N120','sig_slope']:V[f'{a}_{f}']=';'.join(f'{v:+.3f}' for v in g(a,f))
 pd.DataFrame([V]).to_csv(OUT/'verdict.csv',index=False)
 vr=[]
 for a in ARMS:
  for s in range(3):
   m=json.loads((OUT/f'{a}_s{s}_provenance.json').read_text());c_=m['checks']
   vr.append(dict(arm=a,seed=s,wall=round(m['wall_seconds'],1),**{k:c_.get(k) for k in ['g1_maxabs','g1_n','g1_mutctl','ident','ident_mutctl','hzero','decomp','decomp_mutctl','lin0_fwd','lin0_fwd_mutctl','meas_noninv','meas_mutctl']}))
 pd.DataFrame(vr).to_csv(OUT/'validation.csv',index=False)
 fig,ax=plt.subplots(1,3,figsize=(15,4.5));col={'LIN':'C0','LIN0':'C3','LR':'k'}
 for a in ARMS:
  x=d[d.arm==a].groupby('task');ax[0].loglog(x.N.median().index,x.N.median().values,color=col[a],label=a)
  tr=d[(d.arm==a)&d.zbar_inv.notna()].groupby('task')
  ax[1].plot(tr.zbar_inv.median().index,tr.zbar_inv.median().values,color=col[a],label=a)
  ax[2].loglog(tr.N.median().values,tr.sigma_inv.median().values,'o-',color=col[a],label=a,ms=4)
 ax[0].set_xlabel('task');ax[0].set_ylabel('||W~|| (unit mean)');ax[0].set_title('width');ax[0].legend()
 ax[1].set_xlabel('task');ax[1].set_ylabel('zbar_inv');ax[1].set_title('depth');ax[1].axhline(0,color='gray',lw=.5)
 ax[2].set_xlabel('||W~||');ax[2].set_ylabel('sigma_inv');ax[2].set_title('sigma vs width (log-log)');ax[2].legend()
 fig.suptitle('linear_growth_0910 (seed median)');fig.tight_layout()
 for e in ('png','pdf'):fig.savefig(OUT/f'course.{e}',dpi=140)
 print(V)
 pd.set_option('display.width',250)
 print(sv.groupby('arm')[['p_late','z20','z120','sig_slope','sig_r2','s2n2_20','s2n2_120','cos_late','N120','D2_late','erode_late','persist_late','acc20','acc120']].median().reindex(ARMS).to_string(float_format=lambda v:f'{v:+.4f}'))
if __name__=='__main__':main()

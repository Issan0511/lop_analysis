"""Registered analysis: existing CUDA records and CPU records are separate sources."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from src.boundary_groups_0908 import partition,GROUPS,csvwrite
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'results/boundary_groups_0908'
OLD=Path('/home/issan/Projects/claude/proj_004_drift/results/pmnist_boundary_0908')
def oldrows(path,arm,iv,seed):
 out=[]
 with np.load(path) as d:
  for t in range(20):
   en=d['end1'][t];j=d['jump1'][t];drop=j-en;masks=partition(en,j);D=masks[0]|masks[2];den=float((drop[D]**2).sum())
   for step in [300,625]:
    if step==625 and t==19:continue
    aft=d['dense1'][t,-1] if step==300 else d['end1'][t+1]
    row={'arm':arm,'iv':iv,'seed':seed,'task':int(d['task'][t]),'step':step,
         'n_D':int(D.sum()),'recovery':float(((aft-j)[D]*-drop[D]).sum()/den),
         'leftover':float(np.sqrt(((aft-en)[D]**2).sum()/den)),
         'returned_fraction':float((abs(aft[D]-en[D])<=.1*abs(drop[D])).mean())}
    if step==625:
     ds=d['star1'][t+1]-d['star1'][t]
     for label,mask in zip(GROUPS,masks):
      row[label+'_star_sum']=float(ds[mask].sum())
      row[label+'_star_negative_sum']=float(np.minimum(ds[mask],0).sum())
      row[label+'_star_mean']=float(ds[mask].mean()) if mask.any() else None
     row['star_all_mean']=float(ds.mean())
     assert abs(sum(row[g+'_star_sum'] for g in GROUPS)/100-row['star_all_mean'])<1e-6
    out.append(row)
 return out
def main():
 positions=[];credit=[];manifest=[]
 for arm in ['SNA','LR']:
  for iv in ['none','l2']:
   for seed in range(3):
    name=arm+'_'+iv+'_s'+str(seed)
    cp=P/(name+'_provenance.json');assert cp.exists(),cp
    old=OLD/(arm+'_'+('none' if iv=='none' else 'l2-1e-3')+'_s'+str(seed)+'.npz')
    manifest.append({'path':str(old),'sha256':hashlib.sha256(old.read_bytes()).hexdigest()})
    for row in oldrows(old,arm,iv,seed):row['source']='original_CUDA';positions.append(row)
    for row in csv.DictReader((P/(name+'_positions.csv')).open()):
     r={k:float(v) if v and k not in ['arm','iv'] else v for k,v in row.items()}
     r['source']='rerun_CPU';positions.append(r)
    for row in csv.DictReader((P/(name+'_credits.csv')).open()):
     r={k:float(v) if k not in ['arm','iv'] else v for k,v in row.items()}
     credit.append(r)
 (P/'original_manifest.json').write_text(json.dumps(manifest,indent=2))
 seeds=[]
 for source in ['original_CUDA','rerun_CPU']:
  for arm in ['SNA','LR']:
   for iv in ['none','l2']:
    for seed in range(3):
     for step in [300,625]:
      # Common task set 101..119 for all summaries, so differences are not windows.
      a=[r for r in positions if r['source']==source and r['arm']==arm and r['iv']==iv and r['seed']==seed and r['step']==step and r['task']<=119]
      s={'source':source,'arm':arm,'iv':iv,'seed':seed,'step':step,'n_boundaries':len(a)}
      for key in ['recovery','leftover','returned_fraction']:
       s[key]=float(np.mean([r[key] for r in a]))
      for g in GROUPS:
       s[g+'_star_per_boundary_per_network_unit']=float(np.mean([r[g+'_star_sum']/100 for r in a])) if step==625 or source=='rerun_CPU' else None
       s[g+'_star_negative_per_boundary_per_network_unit']=float(np.mean([r[g+'_star_negative_sum']/100 for r in a])) if step==625 or source=='rerun_CPU' else None
      s['star_all_per_boundary']=float(np.mean([r['star_all_mean'] for r in a])) if step==625 or source=='rerun_CPU' else None
      if source=='rerun_CPU':
       cc=[r for r in credit if r['arm']==arm and r['iv']==iv and r['seed']==seed and r['step']==step and r['task']<=119]
       for k in ['CE_before','CE_after','CE_improvement','acc_before','acc_after']+['credit_'+g for g in GROUPS+['downstream']]:
        s[k]=float(np.mean([r[k] for r in cc]))
       s['credit_D_total']=s['credit_D_only']+s['credit_both']
       signatures=[]
       for r in cc:
        pos=next(v for v in a if v['task']==r['task'])
        signatures.append(pos['leftover']>.5 and r['CE_improvement']>1e-6 and r['credit_N_only']>max(0,r['credit_D_only']+r['credit_both']))
       s['signature_count']=sum(signatures);s['signature_fraction']=float(np.mean(signatures))
       assert abs(sum(s['credit_'+g] for g in GROUPS+['downstream'])-s['CE_improvement'])<1e-8
      seeds.append(s)
 keys=list(dict.fromkeys(k for r in seeds for k in r))
 csvwrite(P/'seed_verdict.csv',[{k:r.get(k) for k in keys} for r in seeds])
 summary=[]
 for source in ['original_CUDA','rerun_CPU']:
  for arm in ['SNA','LR']:
   for iv in ['none','l2']:
    for step in [300,625]:
     a=[r for r in seeds if r['source']==source and r['arm']==arm and r['iv']==iv and r['step']==step]
     s={'source':source,'arm':arm,'iv':iv,'step':step,'n_seeds':3,'tasks':'101..119'}
     for k in keys:
      if k in ['source','arm','iv','seed','step','n_boundaries']:continue
      vals=[r[k] for r in a if r.get(k) is not None]
      for tag,val in [('median',np.median(vals)),('min',min(vals)),('max',max(vals))] if vals else []:
       s[k+'_'+tag]=float(val)
     summary.append(s)
 keys=list(dict.fromkeys(k for r in summary for k in r))
 csvwrite(P/'verdict.csv',[{k:r.get(k) for k in keys} for r in summary])
 reproduction=[]
 for f in sorted(P.glob('*_provenance.json')):
  d=json.loads(f.read_text());reproduction.append({'arm':d['arm'],'iv':d['iv'],'seed':d['seed'],
   'original_exact':d['original_exact'],'max_z_error':d['original_max_abs_errors']['dense1'],
   'max_star_error':d['original_max_abs_errors']['star1'],'wall_seconds':d['wall_seconds']})
 csvwrite(P/'reproduction_verdict.csv',reproduction)
 # Seed-wise plotting prevents aggregate bars from hiding n=3.
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
 arms=[('SNA','none'),('SNA','l2'),('LR','none'),('LR','l2')]
 for idx,(arm,iv) in enumerate(arms):
  for source,offset,col in [('original_CUDA',-.12,'#a5a5a5'),('rerun_CPU',.12,'#2266aa')]:
   a=[r for r in seeds if r['source']==source and r['arm']==arm and r['iv']==iv and r['step']==625]
   axs[0].scatter(np.full(3,idx+offset),[r['recovery'] for r in a],color=col,s=35,label=source if idx==0 else None)
 axs[0].axhline(1,color='#aaa',ls=':');axs[0].axhline(0,color='#aaa',lw=.7)
 axs[0].set_title('Drop-group recovery at task end');axs[0].set_ylabel('Projection onto reversal of initial drop')
 axs[0].legend(fontsize=8)
 for j,g in enumerate(['D_total','N_only','downstream']):
  for idx,(arm,iv) in enumerate(arms):
   a=[r for r in seeds if r['source']=='rerun_CPU' and r['arm']==arm and r['iv']==iv and r['step']==625]
   vals=[r['credit_'+g] for r in a]
   axs[1].scatter(np.full(3,idx+(j-1)*.20),vals,color=['#cc6333','#2266aa','#777'][j],s=30,label=g if idx==0 else None)
 axs[1].set_title('CPU: contribution to CE improvement');axs[1].set_ylabel('Exact block Shapley credit');axs[1].legend(fontsize=8);axs[1].axhline(0,color='#aaa',lw=.7)
 for j,g in enumerate(GROUPS):
  for idx,(arm,iv) in enumerate(arms):
   a=[r for r in seeds if r['source']=='original_CUDA' and r['arm']==arm and r['iv']==iv and r['step']==625]
   vals=[r[g+'_star_per_boundary_per_network_unit'] for r in a]
   axs[2].scatter(np.full(3,idx+(j-1.5)*.16),vals,color=['#cc6333','#2266aa','#b092cc','#888'][j],s=25,label=g if idx==0 else None)
 axs[2].set_title('Original: group contributions to setpoint drift');axs[2].set_ylabel('Per task / 100 network units');axs[2].axhline(0,color='#aaa',lw=.7);axs[2].legend(fontsize=8)
 for ax in axs:
  ax.set_xticks(range(4),['SNA','SNA + L2','LR','LR + L2']);ax.grid(alpha=.15)
 fig.suptitle('Who recovers, who repairs output, and whose setpoint drifts?\nEach dot = one seed averaged over tasks 101–119. Attribution is endpoint accounting, not a freezing intervention.')
 fig.savefig(P/'boundary_groups.png',dpi=150);fig.savefig(P/'boundary_groups.pdf');plt.close(fig)
 for r in summary:
  if r['step']==625:print({k:v for k,v in r.items() if k in ['source','arm','iv'] or k in ['recovery_median','leftover_median','credit_D_total_median','credit_N_only_median','credit_downstream_median','signature_count_median','star_all_per_boundary_median','N_only_star_per_boundary_per_network_unit_median','D_only_star_per_boundary_per_network_unit_median']})
if __name__=='__main__':main()

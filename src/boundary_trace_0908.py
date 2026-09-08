
"""Posthoc time-resolved plots of the original CUDA boundary records."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.boundary_groups_0908 import partition,csvwrite
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'results/boundary_trace_0908'
OLD=Path('/home/issan/Projects/claude/proj_004_drift/results/pmnist_boundary_0908')
ARMS=[('SNA','none'),('SNA','l2-1e-3'),('LR','none'),('LR','l2-1e-3')]
GROUPS=['all','D_only','N_only','both','rest']
TIME=np.r_[0,np.arange(1,301),625]
def main():
 P.mkdir(parents=True,exist_ok=True)
 rows=[];summ=[];manifest=[];examples={};checks=[]
 prev=list(csv.DictReader((ROOT/'results/boundary_groups_0908/seed_verdict.csv').open()))
 for arm,iv in ARMS:
  for seed in range(3):
   f=OLD/(arm+'_'+iv+'_s'+str(seed)+'.npz')
   manifest.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
   with np.load(f) as x:d={k:x[k].astype(float) for k in x.files}
   assert np.array_equal(d['task'],np.arange(101,121))
   for layer in [1,2]:
    end=d['end'+str(layer)][:19];jump=d['jump'+str(layer)][:19]
    z=np.concatenate([jump[:,None],d['dense'+str(layer)][:19],d['end'+str(layer)][1:20,None]],axis=1)
    delta=z-jump[:,None];strip=jump-end
    assert np.allclose(z-end[:,None],strip[:,None]+delta,rtol=1e-12,atol=1e-12)
    group_traces={g:[] for g in GROUPS};group_strip={g:[] for g in GROUPS};rec=[]
    for ti in range(19):
     masks=partition(end[ti],jump[ti]) if layer==1 else None
     if layer==1:
      den=(strip[ti,masks[0]|masks[2]]**2).sum()
      rec.append(float((delta[ti,-1,masks[0]|masks[2]]*-strip[ti,masks[0]|masks[2]]).sum()/den))
      reconstructed=np.zeros(len(TIME))
     for g in GROUPS if layer==1 else ['all']:
      mask=np.ones(100,dtype=bool) if g=='all' else masks[GROUPS.index(g)-1]
      if not mask.any():
       group_traces[g].append(np.full(len(TIME),np.nan));group_strip[g].append(np.nan);continue
      curve=delta[ti][:,mask].mean(-1)
      group_traces[g].append(curve);group_strip[g].append(strip[ti,mask].mean())
      if g!='all':reconstructed+=curve*mask.sum()/100
     if layer==1:assert np.allclose(reconstructed,delta[ti].mean(-1),atol=1e-12)
    if layer==1:
     ref=next(r for r in prev if r['source']=='original_CUDA' and r['arm']==arm and r['iv']==('none' if iv=='none' else 'l2') and int(r['seed'])==seed and r['step']=='625')
     err=abs(np.mean(rec)-float(ref['recovery']));assert err<2e-6
     checks.append({'arm':arm,'iv':iv,'seed':seed,'previous_D_recovery_error':err})
    for g in GROUPS if layer==1 else ['all']:
     curve=np.nanmean(group_traces[g],axis=0)
     for step,val in zip(TIME,curve):
      rows.append({'arm':arm,'iv':iv,'seed':seed,'layer':layer,'group':g,'step':int(step),
       'learning_delta':float(val),'strip':float(np.nanmean(group_strip[g]))})
     low=int(np.argmin(curve[:301]))
     summ.append({'arm':arm,'iv':iv,'seed':seed,'layer':layer,'group':g,
      'strip':float(np.nanmean(group_strip[g])),'step1_delta':float(curve[1]),
      'early_1_to_20_mean_delta':float(curve[1:21].mean()),'minimum_0_to_300_delta':float(curve[low]),'minimum_step':low,
      'step20_delta':float(curve[20]),'step100_delta':float(curve[100]),'step300_delta':float(curve[300]),'step625_delta':float(curve[-1])})
    if seed==0 and layer==1:examples[(arm,iv)]={'end':end[0],'jump':jump[0],'z':z[0],'masks':partition(end[0],jump[0])}
 csvwrite(P/'seed_timeseries.csv',rows);csvwrite(P/'seed_verdict.csv',summ)
 metrics=['strip','step1_delta','early_1_to_20_mean_delta','minimum_0_to_300_delta','minimum_step','step20_delta','step100_delta','step300_delta','step625_delta']
 summary=[]
 for arm,iv in ARMS:
  for layer in [1,2]:
   for g in GROUPS if layer==1 else ['all']:
    a=[r for r in summ if r['arm']==arm and r['iv']==iv and r['layer']==layer and r['group']==g]
    s={'arm':arm,'iv':iv,'layer':layer,'group':g}
    for k in metrics:
     vals=[r[k] for r in a]
     s[k+'_median']=float(np.median(vals));s[k+'_min']=float(min(vals));s[k+'_max']=float(max(vals))
    summary.append(s)
 csvwrite(P/'verdict.csv',summary)
 # Per-unit absolute trajectories: show every unit; break 301..624 explicitly.
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
 colors=['#cc6333','#2266aa','#aa77aa','#999999']
 for ax,(arm,iv) in zip(axs.flat,ARMS):
  ex=examples[(arm,iv)]
  xx=np.r_[-1,TIME[:301]]
  for mask,col,g in zip(ex['masks'],colors,GROUPS[1:]):
   vals=np.concatenate([ex['end'][None],ex['z'][:301]],axis=0)
   for i in np.flatnonzero(mask):
    ax.plot(xx,vals[:,i],lw=.6,alpha=.45,color=col)
    ax.scatter([625],[ex['z'][-1,i]],s=7,alpha=.4,color=col)
   if mask.any():ax.plot([],[],color=col,label=g)
  ax.axvspan(301,624,color='#ddd',alpha=.25);ax.axvline(0,color='#555',lw=.7);ax.axhline(0,color='#777',ls=':',lw=.8)
  ax.set_xscale('symlog',linthresh=1);ax.set_xticks([-1,0,1,10,100,300,625],['pre','0','1','10','100','300','625'])
  ax.set_title(arm+(' + L2' if iv!='none' else ''));ax.set_ylabel('Mean preactivation of each unit');ax.set_xlabel('Training updates after permutation switch')
  ax.grid(alpha=.12)
 axs[0,0].legend(fontsize=8,ncol=2)
 fig.suptitle('Follow the SAME 100 units through one task switch\nOriginal CUDA / seed 0 / task 101. pre -> 0 changes only the permutation; 301–624 are unobserved.',fontsize=12)
 fig.savefig(P/'unit_trajectories.png',dpi=160);fig.savefig(P/'unit_trajectories.pdf');plt.close(fig)
 # Aggregate paired learning changes, not unpaired positions.
 fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
 for ax,(arm,iv) in zip(axs.flat,ARMS):
  for g,col in [('all','#30343b'),('D_only','#cc6333'),('N_only','#2266aa'),('rest','#999999')]:
   vv=[]
   for seed in range(3):
    a=[r for r in rows if r['arm']==arm and r['iv']==iv and r['seed']==seed and r['layer']==1 and r['group']==g]
    vv.append([r['learning_delta'] for r in a])
   vv=np.array(vv);med=np.median(vv,axis=0)
   ax.plot(TIME[:301],med[:301],color=col,label=g,lw=1.6 if g!='all' else 2.4)
   ax.fill_between(TIME[:301],vv.min(0)[:301],vv.max(0)[:301],color=col,alpha=.10)
   ax.scatter([625],[med[-1]],color=col,s=28)
   ax.vlines(625,vv[:,-1].min(),vv[:,-1].max(),color=col,lw=1.5)
  ax.axvspan(301,624,color='#ddd',alpha=.25);ax.axhline(0,color='#777',lw=.7)
  ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,10,100,300,625])
  ax.set_title(arm+(' + L2' if iv!='none' else ''));ax.set_ylabel('Change from immediately AFTER switch');ax.set_xlabel('Training updates')
  ax.grid(alpha=.12)
 axs[0,0].legend(fontsize=8,ncol=2)
 fig.suptitle('What LEARNING does after the instantaneous jump\nOriginal CUDA; per-seed average over tasks 101–119, then median and range of 3 seeds. Groups fixed at each boundary.',fontsize=12)
 fig.savefig(P/'learning_trajectories.png',dpi=160);fig.savefig(P/'learning_trajectories.pdf');plt.close(fig)
 (P/'provenance.json').write_text(json.dumps({'source_manifest':manifest,'checks':checks,'new_training':False,'spec_sha256':hashlib.sha256((ROOT/'specs/spec_boundary_trace_0908.md').read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
 for s in summary:
  if s['layer']==1 and s['group'] in ['all','D_only','N_only']:
   print({k:v for k,v in s.items() if k in ['arm','iv','group'] or k in [m+'_median' for m in metrics]})
if __name__=='__main__':main()

"""Registered analysis of target-only versus all-free recovery."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'results/collective_target_only_0908'
CFG=json.loads((ROOT/'configs/collective_target_only_0908.json').read_text())
rows=[];checks=[];fails=[]
for model in CFG['models']:
 label=model['label']
 with np.load(P/(label+'.npz')) as f:d={k:f[k] for k in f.files}
 names=d['conditions'].tolist();T=d['target'];v=d['v'];M=d['M']
 ST=(v*v*M*T[None,None]).sum(-1);S=(v*v*M).sum(-1)
 A=(v*v*(d['m_positive']+d['m_negative_abs'])).sum(-1)
 with np.load(ROOT/'results/collective_kick_0908'/(label+'.npz')) as old:
  oldnames=old['conditions'].tolist()
  for con in ['free_sham','pulse_plus','pulse_minus']:
   ni=names.index(con);oi=oldnames.index(con);errs={}
   assert np.array_equal(d['failure_step'][ni],old['failure_step'][oi])
   for key in ['W','b','v','c','zmean','M','prediction','loss']:
    lhs=d[key][:,ni];rhs=old[key][:,oi];err=float(np.max(abs(lhs-rhs)));errs[key]=err
    assert np.allclose(lhs,rhs,rtol=1e-8,atol=1e-8),(label,con,key,err)
   checks.append({'model':label,'condition':con,'max_abs_errors':errs,'all_exact':all(e==0 for e in errs.values())})
 for ci,con in enumerate(names):
  if con.endswith('sham'):continue
  sham='free_sham' if con.startswith('pulse') else con.rsplit('_',1)[0]+'_sham'
  sh=names.index(sham);dt=ST[0,ci]-ST[0,sh]
  op=abs(dt)>np.maximum(1e-10,1e-6*A[0,sh])
  zinit=np.sqrt((((d['zmean'][0,ci]-d['zmean'][0,sh])**2)*T).sum(-1)/T.sum(-1))
  outinit=np.sqrt(np.mean((d['prediction'][0,ci]-d['prediction'][0,sh])**2,-1))
  for ri,seed in enumerate(d['seeds']):
   if d['failure_step'][ci,ri]>=0:
    fails.append({'model':label,'condition':con,'seed':int(seed),'update':int(d['failure_step'][ci,ri])})
   for ti,step in enumerate(d['steps']):
    valid=bool(d['active'][ti,ci,ri] and d['active'][ti,sh,ri] and op[ri])
    dz=d['zmean'][ti,ci,ri,T[ri]]-d['zmean'][ti,sh,ri,T[ri]]
    dout=d['prediction'][ti,ci,ri]-d['prediction'][ti,sh,ri]
    vref=np.sqrt(np.sum(v[ti,sh,ri,T[ri]]**2))
    mask=T[ri]
    geometry=float(np.sum(v[ti,sh,ri,mask]**2*(M[ti,ci,ri,mask]-M[ti,sh,ri,mask])))
    readout=float(np.sum((v[ti,ci,ri,mask]**2-v[ti,sh,ri,mask]**2)*M[ti,ci,ri,mask]))
    assert np.isclose(geometry+readout,ST[ti,ci,ri]-ST[ti,sh,ri],rtol=1e-8,atol=1e-8)
    rows.append({'target_geometry_residual_signed':geometry/float(dt[ri]),
      'target_readout_residual_signed':readout/float(dt[ri]),'model':label,'condition':con,'seed':int(seed),'update':int(step),'valid':valid,
      'target_self_residual':float(abs(ST[ti,ci,ri]-ST[ti,sh,ri])/abs(dt[ri])),
      'total_self_residual':float(abs(S[ti,ci,ri]-S[ti,sh,ri])/abs(dt[ri])),
      'target_z_residual':float(np.sqrt(np.mean(dz**2))/zinit[ri]),
      'prediction_residual':float(np.sqrt(np.mean(dout**2))/outinit[ri]) if outinit[ri]>1e-10 else float('nan'),
      'target_v_norm_ratio':float(np.sqrt(np.sum(v[ti,ci,ri,T[ri]]**2))/vref) if vref>1e-12 else float('nan'),
      'loss':float(d['loss'][ti,ci,ri]),'sham_loss':float(d['loss'][ti,sh,ri]),
      'c_difference':float(d['c'][ti,ci,ri]-d['c'][ti,sh,ri]),
      'rho':float(abs(S[ti,ci,ri])/A[ti,ci,ri])})
def write(name,data):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
write('seed_metrics.csv',rows)
paired=[];summ=[]
metrics=['target_self_residual','total_self_residual','target_z_residual','prediction_residual','target_v_norm_ratio']
for label in [m['label'] for m in CFG['models']]:
 for direction in ['plus','minus']:
  groups={pre:{r['seed']:r for r in rows if r['model']==label and r['condition']==pre+'_'+direction and r['update']==2000}
          for pre in ['pulse','alonec','alone']}
  valid=sorted(set.intersection(*[{s for s,r in g.items() if r['valid']} for g in groups.values()]))
  for pre,g in groups.items():
   sr={'model':label,'direction':direction,'condition':pre,'n_total':len(g),'n_valid_all3':len(valid),
       'n_failed':sum(not r['valid'] for r in g.values())}
   for metric in metrics:
    q=np.percentile([g[s][metric] for s in valid],[25,50,75])
    for suf,val in zip(['q25','median','q75'],q):sr[metric+'_'+suf]=float(val)
   summ.append(sr)
  for pre in ['alonec','alone']:
   for s in sorted(set(groups['pulse'])&set(groups[pre])):
    a=groups[pre][s];b=groups['pulse'][s]
    paired.append({'model':label,'direction':direction,'comparison':pre+'-pulse','seed':s,
                   'valid':a['valid'] and b['valid'],'target_self_residual_difference':a['target_self_residual']-b['target_self_residual']})
write('verdict.csv',summ);write('paired_comparison.csv',paired)
primary=[r for r in paired if r['model']=='leaky100' and r['direction']=='plus' and r['comparison']=='alonec-pulse' and r['valid']]
vals=np.array([r['target_self_residual_difference'] for r in primary]);res={'n':len(vals),'seed_values':primary}
if len(vals)==10:
 rng=np.random.default_rng(CFG['bootstrap_seed'])
 boot=np.median(rng.choice(vals,(5000,10),replace=True),axis=1)
 lo,hi=np.percentile(boot,[2.5,97.5])
 if lo>.1:verdict='PEER_LEARNING_BENEFITS_TARGET_RECOVERY'
 elif hi<-.1:verdict='PEER_LEARNING_HINDERS_TARGET_RECOVERY'
 elif lo>=-.1 and hi<=.1:verdict='NO_MATERIAL_PEER_DEPENDENCE_IN_WINDOW'
 else:verdict='INCONCLUSIVE'
 res.update(median=float(np.median(vals)),ci95=[float(lo),float(hi)],verdict=verdict)
else:res['verdict']='INCOMPLETE'
res['failures']=fails
(P/'verdict.json').write_text(json.dumps(res,indent=2))
(P/'reproduction_checks.json').write_text(json.dumps(checks,indent=2))
print('PRIMARY',json.dumps(res),flush=True)
for s in summ:
 print(json.dumps({k:s[k] for k in ['model','direction','condition','n_valid_all3','target_self_residual_median','target_z_residual_median','prediction_residual_median']}))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
for j,label in enumerate([m['label'] for m in CFG['models']]):
 for i,direction in enumerate(['plus','minus']):
  ax=axes[i,j]
  for pre,color in [('pulse','#2266aa'),('alonec','#cc6333'),('alone','#629548')]:
   ts=[];med=[];lo=[];hi=[]
   for step in CFG['record_steps']:
    rr=[r for r in rows if r['model']==label and r['condition']==pre+'_'+direction and r['update']==step and r['valid']]
    # Matched three-arm survivors at the final endpoint, fixed throughout display.
    survivors=[r['seed'] for r in rows if r['model']==label and r['condition']=='pulse_'+direction and r['update']==2000 and r['valid']]
    for pp in ['alonec','alone']:
     survivors=list(set(survivors)&{r['seed'] for r in rows if r['model']==label and r['condition']==pp+'_'+direction and r['update']==2000 and r['valid']})
    vv=[r['target_self_residual'] for r in rr if r['seed'] in survivors]
    if vv:
     q=np.percentile(vv,[25,50,75]);ts.append(step);lo.append(q[0]);med.append(q[1]);hi.append(q[2])
   ax.plot(ts,med,color=color,label={'pulse':'All learn','alonec':'Target + output bias','alone':'Target only'}[pre])
   ax.fill_between(ts,lo,hi,color=color,alpha=.12)
  ax.set_xscale('symlog',linthresh=1);ax.set_title(label+' / '+direction)
  ax.set_xlabel('Updates');ax.set_ylabel('Target self-moment residual')
  ax.axhline(0,color='#777',lw=.7);ax.axhline(1,color='#777',ls=':',lw=.7);ax.grid(alpha=.15)
  if i==0 and j==0:ax.legend(fontsize=8)
fig.suptitle('Does target recovery need learning in OTHER units?\nAbsolute residual relative to each matched sham; median and IQR across matched surviving seeds.',fontsize=12)
fig.savefig(P/'target_recovery.png',dpi=150);fig.savefig(P/'target_recovery.pdf');plt.close(fig)

"""Report registered endpoints; do not choose arms/endpoints based on outcome."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'results/collective_kick_0908'
CFG=json.loads((ROOT/'configs/collective_kick_0908.json').read_text())
def write(name,rows):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def div(num,den):return num/den if np.isfinite(den) and abs(den)>1e-300 else float('nan')
rows=[];failures=[];max_split_error=0.;max_hold_constant_error=0.
for model in CFG['models']:
 label=model['label'];path=P/(label+'.npz')
 if not path.exists():continue
 with np.load(path) as f:d={k:f[k] for k in f.files}
 names=d['conditions'].tolist();target=d['target'];sigma=d['sigma'];steps=d['steps'];seeds=d['seeds']
 mom=d['M'];v=d['v'];sv=v*v*mom
 ss=sv.sum(-1);aa=(v*v*(d['m_positive']+d['m_negative_abs'])).sum(-1)
 st=(sv*target[None,None]).sum(-1);so=(sv*(~target)[None,None]).sum(-1)
 for ci,c in enumerate(names):
  sham='hold_sham' if c.startswith('hold') else 'readout_sham' if c.startswith('readout') else 'free_sham'
  sh=names.index(sham)
  islocal=c.startswith(('pulse','hold','readout')) and not c.endswith('sham')
  for ri,seed in enumerate(seeds):
   fstep=int(d['failure_step'][ci,ri])
   if fstep>=0:failures.append({'model':label,'condition':c,'seed':int(seed),'failure_step':fstep})
   oth=~target[ri]
   ds0=ss[0,ci,ri]-ss[0,sh,ri]
   dt0=st[0,ci,ri]-st[0,sh,ri]
   do0=d['prediction'][0,ci,ri]-d['prediction'][0,sh,ri]
   norm0=float(np.mean(do0**2))
   opvalid=abs(dt0)>max(1e-10,1e-6*aa[0,sh,ri])
   for ti,step in enumerate(steps):
    valid=bool(d['active'][ti,ci,ri] and d['active'][ti,sh,ri])
    if c.startswith('hold') and valid:
     max_hold_constant_error=max(max_hold_constant_error,abs(st[ti,ci,ri]-st[0,ci,ri]))
    dso=so[ti,ci,ri]-so[ti,sh,ri]
    geom=float(np.sum(v[ti,sh,ri,oth]**2*(mom[ti,ci,ri,oth]-mom[ti,sh,ri,oth])))
    read=float(np.sum((v[ti,ci,ri,oth]**2-v[ti,sh,ri,oth]**2)*mom[ti,ci,ri,oth]))
    if valid:
     max_split_error=max(max_split_error,abs(geom+read-dso))
     assert np.isclose(geom+read,dso,rtol=1e-9,atol=1e-9)
    dop=d['output_other'][ti,ci,ri]-d['output_other'][ti,sh,ri]
    dp=d['prediction'][ti,ci,ri]-d['prediction'][ti,sh,ri]
    dz=d['zmean'][ti,ci,ri,oth]-d['zmean'][ti,sh,ri,oth]
    dv=v[ti,ci,ri,oth]-v[ti,sh,ri,oth]
    dc=float(d['c'][ti,ci,ri]-d['c'][ti,sh,ri])
    localvalid=valid and islocal and opvalid
    outvalid=valid and islocal and norm0>1e-20
    row={'model':label,'condition':c,'seed':int(seed),'update':int(step),'valid':valid,
         'self_operation_valid':bool(opvalid),'initial_S_target_jump':float(dt0),'initial_prediction_rms':math.sqrt(norm0),
         'S':float(ss[ti,ci,ri]),'A':float(aa[ti,ci,ri]),'rho':div(abs(ss[ti,ci,ri]),aa[ti,ci,ri]),
         'sham_S':float(ss[ti,sh,ri]),'sham_rho':div(abs(ss[ti,sh,ri]),aa[ti,sh,ri]),
         'S_difference':float(ss[ti,ci,ri]-ss[ti,sh,ri]),
         'S_difference_ratio':div(ss[ti,ci,ri]-ss[ti,sh,ri],ds0) if valid else float('nan'),
         'C_S':div(-dso,dt0) if localvalid else float('nan'),
         'C_S_geometry':div(-geom,dt0) if localvalid else float('nan'),
         'C_S_readout':div(-read,dt0) if localvalid else float('nan'),
         'C_output':div(-float(np.mean(dop*do0)),norm0) if outvalid else float('nan'),
         'C_output_c_only':div(-dc*float(np.mean(do0)),norm0) if outvalid else float('nan'),
         'prediction_difference_ratio':div(float(np.sqrt(np.mean(dp**2))),math.sqrt(norm0)) if norm0>1e-20 and valid else float('nan'),
         'z_other_rms':float(np.sqrt(np.mean(dz**2))),'z_other_rms_over_sigma':float(np.sqrt(np.mean(dz**2))/sigma[ri]),
         'z_other_mean':float(np.mean(dz)),'v_other_rms':float(np.sqrt(np.mean(dv**2))),
         'z_population_difference':float(np.mean(d['zmean'][ti,ci,ri]-d['zmean'][ti,sh,ri])),
         'force_z_population_difference':float(np.mean(d['force_zmean'][ti,ci,ri]-d['force_zmean'][ti,sh,ri])),
         'force_b_population_difference':float(np.mean(d['force_b'][ti,ci,ri]-d['force_b'][ti,sh,ri])),
         'loss':float(d['loss'][ti,ci,ri]),'sham_loss':float(d['loss'][ti,sh,ri])}
    rows.append(row)
assert max_hold_constant_error<1e-8,max_hold_constant_error
write('paired_metrics.csv',rows)
# Primary: symmetric signed-kick average within each leaky seed, then median across seeds.
primary=[]
for seed in range(10):
 pair=[r for r in rows if r['model']=='leaky100' and r['seed']==seed and r['update']==CFG['steps'] and r['condition'] in ['hold_plus','hold_minus']]
 if len(pair)==2 and all(r['valid'] and np.isfinite(r['C_S']) for r in pair):
  primary.append({'seed':seed,'C_S_symmetric':float(np.mean([r['C_S'] for r in pair]))})
values=np.array([r['C_S_symmetric'] for r in primary])
res={'registered_primary':'LR_1216 hold ± symmetric compensation at 2000 updates','n_valid':len(values),'seed_values':primary}
if len(values)==10:
 rng=np.random.default_rng(CFG['bootstrap_seed'])
 boot=np.median(rng.choice(values,(CFG['bootstrap_n'],len(values)),replace=True),axis=1)
 lo,hi=np.percentile(boot,[2.5,97.5])
 if lo>=.8 and hi<=1.2:verdict='NEAR_COMPLETE_COMPENSATION'
 elif lo>0:verdict='PARTIAL_COMPENSATION'
 elif hi<=0:verdict='NO_POSITIVE_COMPENSATION'
 else:verdict='INCONCLUSIVE'
 res.update(median=float(np.median(values)),ci95=[float(lo),float(hi)],verdict=verdict)
else:res['verdict']='INCOMPLETE'
res['failures']=failures
res['max_geometry_readout_split_error']=max_split_error
res['max_pinned_target_S_change']=max_hold_constant_error
(P/'verdict.json').write_text(json.dumps(res,indent=2))
# Final-state registered secondary descriptions, per model-condition. Never merge seeds across models.
summ=[]
for label in [m['label'] for m in CFG['models']]:
 for c in ['global_plus','global_minus','pulse_plus','pulse_minus','hold_plus','hold_minus','readout_plus','readout_minus']:
  group=[r for r in rows if r['model']==label and r['condition']==c and r['update']==CFG['steps']]
  if not group:continue
  sr={'model':label,'condition':c,'n_total':len(group),'n_valid':sum(r['valid'] for r in group)}
  for key in ['C_S','C_S_geometry','C_S_readout','C_output','C_output_c_only',
              'prediction_difference_ratio','z_other_rms_over_sigma','S_difference_ratio','rho','sham_rho']:
   vs=[r[key] for r in group if r['valid'] and np.isfinite(r[key])]
   qs=np.percentile(vs,[25,50,75]) if vs else [float('nan')]*3
   for q,num in zip(['q25','median','q75'],qs):sr[key+'_'+q]=float(num)
  summ.append(sr)
write('verdict.csv',summ)
print('PRIMARY',json.dumps(res),flush=True)
for s in summ:
 if s['condition'].startswith(('hold','readout')):
  print(json.dumps({k:s[k] for k in ['model','condition','n_valid','C_S_median','C_S_geometry_median','C_S_readout_median','C_output_median','C_output_c_only_median','z_other_rms_over_sigma_median']}),flush=True)

# Global endpoints are per-seed ratios before aggregation.
global_rows=[]
for label in [m['label'] for m in CFG['models']]:
 for condition in ['global_plus','global_minus']:
  init={r['seed']:r for r in rows if r['model']==label and r['condition']==condition and r['update']==0}
  end={r['seed']:r for r in rows if r['model']==label and r['condition']==condition and r['update']==CFG['steps']}
  ratios=[div(end[k]['z_population_difference'],init[k]['z_population_difference']) for k in init if k in end and end[k]['valid']]
  sign=1 if condition.endswith('plus') else -1
  global_rows.append({'model':label,'condition':condition,
   'z_remaining_ratio_median':float(np.nanmedian(ratios)),
   'initial_full_z_restoring_count':sum(sign*r['force_z_population_difference']<0 for r in init.values()),'n':len(init)})
(P/'global_endpoint.json').write_text(json.dumps(global_rows,indent=2))

# Figures: median and IQR across seed, not confidence bands.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
for col,label in enumerate([m['label'] for m in CFG['models']]):
 for rowno,key in enumerate(['C_S','C_output']):
  ax=axes[rowno,col]
  for c,color,ls in [('hold_plus','#c45c30','-'),('hold_minus','#2467aa','-'),
                     ('readout_plus','#c45c30','--'),('readout_minus','#2467aa','--')]:
   gr=[r for r in rows if r['model']==label and r['condition']==c and r['valid']]
   tt=[];med=[];lo=[];hi=[]
   for t in CFG['record_steps']:
    vals=[r[key] for r in gr if r['update']==t and np.isfinite(r[key])]
    if vals:
     q=np.percentile(vals,[25,50,75]);tt.append(t);lo.append(q[0]);med.append(q[1]);hi.append(q[2])
   ax.plot(tt,med,color=color,ls=ls,lw=1.6,label=c)
   if c.startswith('hold'):ax.fill_between(tt,lo,hi,color=color,alpha=.12)
  ax.set_xscale('symlog',linthresh=1);ax.set_xlabel('Updates after kick')
  ax.axhline(0,color='#777',lw=.6);ax.axhline(1,color='#555',ls=':',lw=.8)
  ax.set_title(label);ax.set_ylabel('Self-moment compensation' if rowno==0 else 'Output compensation')
  ax.grid(alpha=.15)
  if col==0 and rowno==0:ax.legend(fontsize=8)
fig.suptitle('Pinned minority: do OTHER units compensate? Median and IQR across seeds\nSolid: other hidden units and readout learn. Dashed: readout only. Frozen task / full-batch GD.',fontsize=12)
fig.savefig(P/'minority_compensation.png',dpi=150);fig.savefig(P/'minority_compensation.pdf')
plt.close(fig)
# Global-perturbation trajectories relative to sham.
fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
for col,label in enumerate([m['label'] for m in CFG['models']]):
 for rowno,key in enumerate(['z_population_difference','S_difference_ratio']):
  ax=axes[rowno,col]
  for c,color in [('global_plus','#c45c30'),('global_minus','#2467aa')]:
   gr=[r for r in rows if r['model']==label and r['condition']==c and r['valid']]
   tt=[];med=[];lo=[];hi=[]
   for t in CFG['record_steps']:
    vs=[r[key] for r in gr if r['update']==t and np.isfinite(r[key])]
    if vs:
     q=np.percentile(vs,[25,50,75]);tt.append(t);lo.append(q[0]);med.append(q[1]);hi.append(q[2])
   ax.plot(tt,med,color=color,lw=1.6,label=c);ax.fill_between(tt,lo,hi,color=color,alpha=.12)
  ax.set_xscale('symlog',linthresh=1);ax.set_xlabel('Updates after kick')
  ax.axhline(0,color='#555',ls=':',lw=.8);ax.set_title(label)
  ax.set_ylabel('Mean z difference from sham' if rowno==0 else 'S difference / initial S difference')
  ax.grid(alpha=.15)
fig.suptitle('Whole-population kick: return toward the matched sham trajectory\nMedian and IQR across seeds. Returning to sham is distinct from absolute S = 0.',fontsize=12)
fig.savefig(P/'global_response.png',dpi=150);plt.close(fig)

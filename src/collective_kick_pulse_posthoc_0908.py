"""Post-hoc pulse-versus-hold audit requested after registered pilot. No training."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/collective_kick_0908'
OUT=ROOT/'results/collective_kick_pulse_posthoc_0908'
OUT.mkdir(parents=True,exist_ok=True)
seedrows=[];summary=[];sources=[]
for label in ['leaky100','elu100','snake075','snake1']:
 path=SRC/(label+'.npz')
 sources.append({'model':label,'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 with np.load(path) as d:
  names=d['conditions'].tolist();T=d['target'];v=d['v'];m=d['M']
  ST=(v*v*m*T[None,None]).sum(-1);SO=(v*v*m*(~T)[None,None]).sum(-1)
  for direction in ['plus','minus']:
   pp=names.index('pulse_'+direction);hh=names.index('hold_'+direction)
   ps=names.index('free_sham');hs=names.index('hold_sham')
   dt=ST[0,pp]-ST[0,ps]
   assert np.allclose(dt,ST[0,hh]-ST[0,hs],rtol=1e-12,atol=1e-12)
   A=(v[0,ps]**2*(d['m_positive'][0,ps]+d['m_negative_abs'][0,ps])).sum(-1)
   valid=(d['active'][-1,pp]&d['active'][-1,hh]&d['active'][-1,ps]&d['active'][-1,hs]
          &(abs(dt)>np.maximum(1e-10,1e-6*A)))
   remainp=(ST[-1,pp]+SO[-1,pp]-ST[-1,ps]-SO[-1,ps])/dt
   remainh=(ST[-1,hh]+SO[-1,hh]-ST[-1,hs]-SO[-1,hs])/dt
   ct=1-(ST[-1,pp]-ST[-1,ps])/dt
   co=-(SO[-1,pp]-SO[-1,ps])/dt
   assert np.allclose((ct+co+remainp)[valid],1,atol=1e-10)
   zrem=((d['zmean'][-1,pp]-d['zmean'][-1,ps])*T).sum(-1)/((d['zmean'][0,pp]-d['zmean'][0,ps])*T).sum(-1)
   vn=np.sqrt((v[-1,pp]**2*T).sum(-1)/(v[-1,ps]**2*T).sum(-1))
   geom=(v[-1,ps]**2*(m[-1,pp]-m[-1,ps])*T).sum(-1)
   read=((v[-1,pp]**2-v[-1,ps]**2)*m[-1,pp]*T).sum(-1)
   assert np.allclose((1-geom/dt-read/dt)[valid],ct[valid],atol=1e-10)
   group=[]
   for i,seed in enumerate(d['seeds']):
    row={'model':label,'direction':direction,'seed':int(seed),'valid_paired':bool(valid[i]),
     'pulse_abs_remaining':float(abs(remainp[i])),'hold_abs_remaining':float(abs(remainh[i])),
     'pulse_signed_remaining':float(remainp[i]),'hold_signed_remaining':float(remainh[i]),
     'pulse_target_self_recovery':float(ct[i]),'pulse_others_self_compensation':float(co[i]),
     'pulse_target_z_remaining':float(zrem[i]),'pulse_target_v_norm_ratio':float(vn[i]),
     'pulse_target_geometry_recovery':float(1-geom[i]/dt[i]),'pulse_target_readout_recovery':float(-read[i]/dt[i])}
    seedrows.append(row)
    if valid[i]:group.append(row)
   sr={'model':label,'direction':direction,'n_total':len(valid),'n_valid_paired':int(valid.sum()),
       'pulse_smaller_abs_error_count':int(np.sum(abs(remainp[valid])<abs(remainh[valid])))}
   for key in list(group[0])[4:]:
    q=np.percentile([r[key] for r in group],[25,50,75])
    for name,value in zip(['q25','median','q75'],q):sr[key+'_'+name]=float(value)
   summary.append(sr)
for file,data in [('verdict.csv',summary),('seed_metrics.csv',seedrows)]:
 with (OUT/file).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
(OUT/'provenance.json').write_text(json.dumps({'tier':'post-hoc descriptive; no new training','checkpoint_update':2000,
 'sources':sources,'aggregation':'absolute residual per seed before median; pulse/hold matched valid seeds',
 'warning':'target/other terms are state accounting, not independent causal effects; medians do not add'},indent=2))
print('Saved post-hoc matched comparisons for',len(summary),'model-direction groups')

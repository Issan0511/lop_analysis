"""Aggregate preregistered one-shot rescue contrasts from raw curves."""
from pathlib import Path
import csv,json,math,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/elu_sunk_rescue_0913'
BR=['A','B20','C20','D20','C5','D5','C10','D10','RB20','RC20','RD20']
STEPS=[0,75,375,1500,3000,6000]
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def interval(v):
 v=np.asarray(v,float);mu=float(v.mean());sd=float(v.std(ddof=1));h=4.302652729911275*sd/math.sqrt(3)
 return mu,sd,mu-h,mu+h
def main():
 summaries=[];contrasts=[];verdict=[];raw={};total=0;pools={};qa=[]
 for env in ['PM','RL']:
  for T in [10,20,50]:
   for layer in [1,2]:
    job=OUT/f'{env}_t{T}_l{layer}';rows=read(job/'rows.csv');assert len(rows)==990,(job,len(rows));total+=len(rows)
    sel=json.loads((job/'selections.json').read_text())
    for q in sel['seeds']:pools[env,T,layer,int(q['seed'])]=len(q['pool'])
    prov=json.loads((job/'provenance.json').read_text());qa.append({'job':job.name,'provenance':str((job/'provenance.json').relative_to(ROOT))})
    for s in range(3):
     for b in BR:
      rr=[r for r in rows if int(r['seed'])==s and r['branch']==b]
      assert len(rr)==30
      aa=[];ec=[];ea=[]
      for t in range(T+1,T+6):
       r=sorted([r for r in rr if int(r['task'])==t],key=lambda x:int(x['step']))
       assert [int(x['step']) for x in r]==STEPS
       yy=np.array([float(x['ce']) for x in r]);acc=np.array([float(x['acc']) for x in r])
       assert np.isfinite(yy).all() and np.isfinite(acc).all()
       assert (acc>=0).all() and (acc<=1).all()
       aa.append(float(np.trapezoid(yy,STEPS)/6000));ec.append(float(yy[-1]));ea.append(float(acc[-1]))
       raw[env,T,layer,s,b,t]=r
      summaries.append(dict(env=env,checkpoint=T,layer=layer,seed=s,branch=b,pool=pools[env,T,layer,s],
         first_auc=aa[0],all_auc=float(np.mean(aa)),first_end_ce=ec[0],all_end_ce=float(np.mean(ec)),
         first_end_acc=ea[0],all_end_acc=float(np.mean(ea))))
    for s in range(3):
     arms={r['branch']:r for r in summaries if r['env']==env and r['checkpoint']==T and r['layer']==layer and r['seed']==s}
     for metric in ['first_auc','all_auc','first_end_ce','all_end_ce','first_end_acc','all_end_acc']:
      # Convert every primitive to cost; positive benefit has the same interpretation.
      sign=-100 if metric.endswith('acc') else 1
      v={b:sign*r[metric] for b,r in arms.items()}
      deep=v['B20']-v['A'];lift=v['D20']-v['C20']
      out=dict(env=env,checkpoint=T,layer=layer,seed=s,pool=pools[env,T,layer,s],metric=metric,
          G20=v['A']-v['C20'],deep_update_benefit=deep,lift_update_benefit=lift,R20=lift-deep,
          Grandom=v['A']-v['RC20'],Rrandom=(v['RD20']-v['RC20'])-(v['RB20']-v['A']),
          G5=v['A']-v['C5'],G10=v['A']-v['C10'],
          update_benefit5=v['D5']-v['C5'],update_benefit10=v['D10']-v['C10'])
      out['R_target_minus_random']=out['R20']-out['Rrandom'];out['G_target_minus_random']=out['G20']-out['Grandom'];contrasts.append(out)
    eligible=all(pools[env,T,layer,s]>=5 for s in range(3))
    c=[r for r in contrasts if r['env']==env and r['checkpoint']==T and r['layer']==layer and r['metric']=='first_auc']
    if not eligible:label='NOT_IDENTIFIABLE'
    elif all(r['R20']>.01 and r['G20']>.01 for r in c):label='DIRECTIONAL_RESCUE_AND_UPDATE_SUPPORT'
    elif all(r['G20']>.01 for r in c):label='RESCUE_WITHOUT_SELECTIVE_UPDATE_SUPPORT'
    else:label='INCONCLUSIVE'
    for metric in ['first_auc','all_auc','first_end_ce','all_end_ce','first_end_acc','all_end_acc']:
     cc=[r for r in contrasts if r['env']==env and r['checkpoint']==T and r['layer']==layer and r['metric']==metric]
     for name in ['R20','G20','deep_update_benefit','lift_update_benefit','Rrandom','Grandom','R_target_minus_random','G_target_minus_random','G5','G10','update_benefit5','update_benefit10']:
      vv=[next(r[name] for r in cc if r['seed']==s) for s in range(3)]
      mu,sd,lo,hi=interval(vv)
      verdict.append(dict(env=env,checkpoint=T,layer=layer,metric=metric,contrast=name,seed0=vv[0],seed1=vv[1],seed2=vv[2],
            mean=mu,sd=sd,ci95_low=lo,ci95_high=hi,
            scope='PRIMARY' if (env,T,layer,metric)==('RL',20,2,'first_auc') else 'SECONDARY',
            cohort_status='ELIGIBLE' if eligible else 'COHORT_TOO_SMALL',pilot_label=label if metric=='first_auc' else 'SECONDARY'))
 write(OUT/'seed_endpoints.csv',summaries);write(OUT/'contrasts.csv',contrasts);write(OUT/'verdict.csv',verdict)
 audit=dict(status='PASS',jobs=len(qa),branches=len(summaries),curve_rows=total,
    first_and_all_task_auc_recomputed=True,seed_replication=3,accuracy_contrasts_in_percentage_points=True,
    primary='RL checkpoint20 layer2 first future task full dose',jobs_provenance=qa,
    verdict_sha256=hashlib.sha256((OUT/'verdict.csv').read_bytes()).hexdigest())
 (OUT/'aggregation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 # Primary curves: log CE retains the full immediate shock.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 styles={'A':('#3273a8','-','Deep / train'),'B20':('#3273a8','--','Deep / frozen'),
         'C20':('#d66535','-','Lift once / train'),'D20':('#d66535','--','Lift once / frozen')}
 fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
 for b,(color,ls,label) in styles.items():
  for ax,key,mul in [(axs[0],'ce',1),(axs[1],'acc',100)]:
   vv=np.array([[float(r[key])*mul for r in raw['RL',20,2,s,b,21]] for s in range(3)])
   for rr in vv:ax.plot(STEPS,rr,color=color,linestyle=ls,alpha=.2,lw=.7)
   ax.plot(STEPS,vv.mean(0),color=color,linestyle=ls,lw=2,label=label)
 for ax in axs:
  ax.set_xscale('symlog',linthresh=75);ax.set_xticks(STEPS,[str(x) for x in STEPS],fontsize=8);ax.set_xlabel('Updates on fresh task21')
  ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
 axs[0].set_yscale('log');axs[0].set_ylabel('CE (log scale)');axs[1].set_ylabel('Training accuracy (%)');axs[1].set_ylim(0,102)
 fig.suptitle('One-shot rescue: RL, layer2, checkpoint20, up to20 units\nMean of3 seeds; faint lines are individual seeds')
 fig.savefig(OUT/'primary_curves.png',dpi=160);plt.close(fig)
 lines=['# One-shot ELU sunk-unit rescue results','',
 'Preregistered 3-seed pilot; same1200 images and6000updates per new task. All outcomes below concern fresh-task learning, not just early-minus-late decline.',
 'Primary: RL, layer2, checkpoint20, up to20 persistent units, first subsequent task.',
 'R=(D-C)-(B-A); G=A-C, with A deep/train, B deep/frozen, C lifted/train, D lifted/frozen. Positive is restoration/benefit for cost metrics.',
 'CE area is divided by6000; accuracy contrast units are percentage points. Every timepoint is retained including the immediate lift shock.',
 '',
 '|Environment|Checkpoint|Layer|Pool sizes|R mean [95% CI]|G mean [95% CI]|Pilot label|',
 '|---|---:|---:|---|---|---|---|']
 for env in ['PM','RL']:
  for T in [10,20,50]:
   for l in [1,2]:
    rr=next(r for r in verdict if (r['env'],r['checkpoint'],r['layer'],r['metric'],r['contrast'])==(env,T,l,'first_auc','R20'))
    gg=next(r for r in verdict if (r['env'],r['checkpoint'],r['layer'],r['metric'],r['contrast'])==(env,T,l,'first_auc','G20'))
    ns=','.join(str(pools[env,T,l,s]) for s in range(3))
    lines.append(f"|{env}|{T}|{l}|{ns}|{rr['mean']:.6f} [{rr['ci95_low']:.6f},{rr['ci95_high']:.6f}]|{gg['mean']:.6f} [{gg['ci95_low']:.6f},{gg['ci95_high']:.6f}]|{rr['pilot_label']}|")
 lines+=['','## Primary per-seed detail','','|Seed|Pool|A CE area|B CE area|C CE area|D CE area|A end accuracy %|C end accuracy %|D end accuracy %|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for s in range(3):
  a={r['branch']:r for r in summaries if (r['env'],r['checkpoint'],r['layer'],r['seed'])==('RL',20,2,s)}
  lines.append(f"|{s}|{pools['RL',20,2,s]}|{a['A']['first_auc']:.6f}|{a['B20']['first_auc']:.6f}|{a['C20']['first_auc']:.6f}|{a['D20']['first_auc']:.6f}|{a['A']['first_end_acc']*100:.3f}|{a['C20']['first_end_acc']*100:.3f}|{a['D20']['first_end_acc']*100:.3f}|")
 lines+=['','Limits: retrospective choice of environment/checkpoints from the previous result; prospective intervention comparisons. Finite5-task observation does not prove permanent non-recovery. Bias lifting changes initial predictions across depth conditions; paired train/frozen functions match. Incident-parameter freezing includes incoming rows, bias and outgoing columns, while other parameters still learn. Restored fixed features can help even without target updates, so lack of R is not evidence that capacity was never lost. A failed one-shot rescue is not proof of harmless sinking. W norms may diverge after intervention; this is not a W-independent mediated fraction. Random controls may contain other persistent units. PM ceiling and RL floor remain visible.','',
 'See per-job selections.json for exact IDs/deltas and provenance.json for all technical checks, initial shocks and hashes. Raw curves and per-input transport diagnostics are retained.','']
 (OUT/'summary.md').write_text('\n'.join(lines))
 print(json.dumps(audit,indent=2));print('\n'.join(lines[:22]))
if __name__=='__main__':main()


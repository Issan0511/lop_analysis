import csv, json, hashlib
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[1]
root=repo/'results'/'elu_sunk_rescue_0913'/'RL_t20_l2'
steps=[0,75,375,1500,3000,6000]; tasks=range(21,26); branches=['A','B20','C20','D20']
sel=json.loads((root/'selections.json').read_text())['seeds']
selected={int(x['seed']):[int(u) for u in x['target20']] for x in sel}
assert all(len(v)==20 and len(set(v))==20 for v in selected.values())
rows=list(csv.DictReader((root/'rows.csv').open()))
row_lookup={(int(r['seed']),r['branch'],int(r['task']),int(r['step'])):r for r in rows}
with np.load(root/'units.npz') as z:
 out=[]; agg={}
 for seed in range(3):
  ids=selected[seed]
  for branch in branches:
   task_stats=[]
   for task in tasks:
    vals=[]
    for step in steps:
     prefix=f's{seed}_{branch}_t{task}_u{step}_'; a={m:z[prefix+m+'_l2'][ids] for m in ('q','zmean','gate_mean')}
     if step: a.update({m:z[prefix+m+'_l2'][ids] for m in ('dWin','db','dWout')})
     metric=row_lookup[(seed,branch,task,step)]
     upstream={m:float(z[prefix+m+'_l1'].mean()) for m in ('dWin','db','dWout')} if step else {}
     coarse=(float(row_lookup[(seed,branch,task,0)]['ce'])+float(metric['ce']))*.5*75/6000 if step==75 else ''
     for rank,u in enumerate(ids):
      out.append(dict(seed=seed,branch=branch,task=task,step=step,selected_rank=rank+1,unit_id=u,q=float(a['q'][rank]),zmean=float(a['zmean'][rank]),gate_mean=float(a['gate_mean'][rank]),sink_q_ge_0_95=int(a['q'][rank]>=.95),dWin_interval_norm=float(a['dWin'][rank]) if step else '',db_interval_abs=float(a['db'][rank]) if step else '',dWout_interval_norm=float(a['dWout'][rank]) if step else '',upstream_l1_dWin_mean_interval=upstream.get('dWin',''),upstream_l1_db_mean_interval=upstream.get('db',''),upstream_l1_dWout_mean_interval=upstream.get('dWout',''),probe_ce=float(metric['ce']),probe_acc=float(metric['acc']),online_ce=float(metric['online_ce']) if metric['online_ce'] else '',online_acc=float(metric['online_acc']) if metric['online_acc'] else '',probe_auc_0_75_term=coarse))
     vals.append(a)
    task_stats.append(dict(q0=float(vals[0]['q'].mean()),qend=float(vals[-1]['q'].mean()),z0=float(vals[0]['zmean'].mean()),zend=float(vals[-1]['zmean'].mean()),gate0=float(vals[0]['gate_mean'].mean()),gateend=float(vals[-1]['gate_mean'].mean()),resunk=int((vals[-1]['q']>=.95).sum()),strong_open=int((vals[-1]['q']<.5).sum()),dWin=float(sum(v['dWin'].mean() for v in vals[1:])),db=float(sum(v['db'].mean() for v in vals[1:])),dWout=float(sum(v['dWout'].mean() for v in vals[1:]))))
   for scope,ts in [('first_task',[task_stats[0]]),('all5_task_mean',task_stats)]: agg[seed,branch,scope]={k:float(np.mean([x[k] for x in ts])) for k in task_stats[0]}
 with (root/'primary_unit_recovery.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
doses=['C5','D5','C10','D10','C20','D20']; learning=[]
for seed in range(3):
 for b in doses:
  rr=sorted([r for r in rows if int(r['seed'])==seed and r['branch']==b and int(r['step'])==6000],key=lambda r:int(r['task'])); assert [int(r['task']) for r in rr]==list(tasks)
  learning.append((seed,b,float(rr[0]['ce']),float(rr[0]['acc']),float(np.mean([float(r['ce']) for r in rr])),float(np.mean([float(r['acc']) for r in rr]))))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
lines=['# Primary selected-unit recovery (descriptive registered secondary readout)','', '**DESCRIPTIVE ONLY.** This reads the completed primary job; it adds no intervention, inferential test, causal claim, or hypothesis-significance analysis. Co-occurrence of gate opening and learning differences does not establish that one caused the other.','', 'Exact scope: RL, checkpoint T=20, layer 2; the same 20 `target20` IDs per seed from `selections.json`; branches A/B20/C20/D20; future tasks 21-25; registered steps 0, 75, 375, 1500, 3000, 6000. Sink means q>=.95.','', '`dWin_interval_norm`, `db_interval_abs`, and `dWout_interval_norm` in the CSV are actual changes between adjacent measurement points. Table movement values sum those five interval norms within each task, then average over selected IDs; all-five values additionally average tasks 21-25. They are path lengths, not signed or net displacement.','', '## Selected target20 unit state and movement','', 'Each state value is the mean over the 20 selected IDs. `resunk` is the count out of 20 at the task endpoint. Values are shown as first future task / mean across all five future tasks.','', '| seed | branch | q step0 | q endpoint | resunk | zmean step0 | zmean endpoint | gate step0 | gate endpoint | dWin path | db path | dWout path |','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for seed in range(3):
 for b in branches:
  x=agg[seed,b,'first_task']; y=agg[seed,b,'all5_task_mean']; fmt=lambda k,n=4:f'{x[k]:.{n}g} / {y[k]:.{n}g}'
  lines.append(f"| {seed} | {b} | {fmt('q0')} | {fmt('qend')} | {x['resunk']:.0f} / {y['resunk']:.1f} | {fmt('z0')} | {fmt('zend')} | {fmt('gate0')} | {fmt('gateend')} | {fmt('dWin')} | {fmt('db')} | {fmt('dWout')} |")
lines += ['','## Dose endpoint learning','','Endpoint is step 6000. Accuracy remains a fraction. First is task 21; all-five is the arithmetic mean across tasks 21-25.','', '| seed | branch | CE first | accuracy first | CE all-five mean | accuracy all-five mean |','|---:|---|---:|---:|---:|---:|']
for s,b,cf,af,ca,aa in learning: lines.append(f'| {s} | {b} | {cf:.6g} | {af:.6f} | {ca:.6g} | {aa:.6f} |')
lines += ['','## Post-hoc first-task estimator check','','This post-hoc check does not change the registered primary verdict. `online_ce` is cumulative minibatch CE through update 6000; the probe AUC uses full-1200 probe CE at six points. They are distinct estimators. `0-75 probe area` is only the first trapezoid contribution after division by 6000.','', '| seed | branch | online CE through 6000 | 0-75 probe area contribution |','|---:|---|---:|---:|']
for s in range(3):
 for b in branches:
  end=row_lookup[(s,b,21,6000)]; r0=row_lookup[(s,b,21,0)]; r75=row_lookup[(s,b,21,75)]; term=(float(r0['ce'])+float(r75['ce']))*.5*75/6000
  lines.append(f"| {s} | {b} | {float(end['online_ce']):.6f} | {term:.6f} |")
lines += ['','Mean first-task online CE across seeds is '+f"{np.mean([float(row_lookup[(s,'A',21,6000)]['online_ce']) for s in range(3)]):.6f}"+' for A and '+f"{np.mean([float(row_lookup[(s,'C20',21,6000)]['online_ce']) for s in range(3)]):.6f}"+' for C20: the online estimator improves descriptively even though the registered six-point probe AUC is worsened by the large initial-shock contribution.','', 'Across the registered task endpoints, A/B20 remain almost entirely sunk. The lifted C20/D20 units are opened at step 0, but most or all are again at q>=.95 by step 6000. On task 21 and in the five-task endpoint averages, C20 and D20 each have lower CE and higher accuracy than A and B20 in all three seeds. Thus improved fresh-task learning accompanies the transient gate opening descriptively. C20 versus D20 learning is mixed across seeds, while only C20 permits selected layer-2 incident-parameter movement, so the improvement does not consistently track that updating.','', 'D20 does not hold the selected units preactivation-fixed: its selected layer-2 incoming rows, biases, and outgoing columns have zero direct movement, but layer-1 parameters and activations still evolve. The CSV therefore includes mean layer-1 interval movement alongside each selected-unit record. These co-occurrences are not a mediation or causal decomposition. Gate opening also creates a large immediate forward shock, so endpoint or trajectory improvement cannot be attributed solely to gate state or parameter updating. A unit may open and re-sink between recorded endpoints.','', '## Sources','', '- `selections.json` SHA256: `'+sha(root/'selections.json')+'`','- `units.npz` SHA256: `'+sha(root/'units.npz')+'`','- `rows.csv` SHA256: `'+sha(root/'rows.csv')+'`','- Source directory: `/home/issan/Projects/claude/elu_sunk_rescue_0913/results/elu_sunk_rescue_0913/RL_t20_l2`','']
(root/'primary_unit_recovery.md').write_text('\n'.join(lines))
print('rows',len(out)); print('\n'.join(lines[:30]))

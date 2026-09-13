#!/usr/bin/env python3
"""Aggregate the preregistered ELU response-anchor experiment.

This script never trains.  It requires one complete 5-task x 18-model result
and keeps seeds as the only replicate unit.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'elu_response_anchor_0913'
BRANCHES=('A','B','C','D','AF','BF'); SEEDS=range(3); TASKS=range(21,26)
STEPS=(0,1,2,5,10,20,25,50,75,150,375,750,1500,3000,6000)
STYLE={'A':('#3273a8','-'),'B':('#3273a8','--'),'C':('#d66535','-'),'D':('#d66535','--'),'AF':('#777777','-'),'BF':('#777777','--')}

def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ci3(values):
    x=np.asarray(values,float); mean=float(x.mean()); sd=float(x.std(ddof=1)); h=4.302652729911275*sd/math.sqrt(3)
    return mean,sd,mean-h,mean+h
def classify(gains):
    if all(x>.01 for x in gains):return 'DIRECTIONAL_RESPONSE_SUPPORT'
    if all(x<-.01 for x in gains):return 'DIRECTIONAL_RESPONSE_HARM'
    return 'INCONCLUSIVE'
def auc(y,steps,denom):return float(np.trapezoid(np.asarray(y,float),np.asarray(steps,float))/denom)
def require_unique(rows,fields):
    keys=[tuple(r[f] for f in fields) for r in rows]
    if len(keys)!=len(set(keys)):raise RuntimeError('duplicate keys for '+','.join(fields))

def self_test():
    assert classify([.02,.03,.011])=='DIRECTIONAL_RESPONSE_SUPPORT'
    assert classify([-.02,-.03,-.011])=='DIRECTIONAL_RESPONSE_HARM'
    assert classify([.02,0,-.02])=='INCONCLUSIVE'
    assert abs(auc([2,1],[0,75],75)-1.5)<1e-12
    require_unique([{'metric':'x','seed':0},{'metric':'x','seed':1}],('metric','seed'))
    # Positive cost benefit and interaction signs.
    A,B,C,D=2.0,1.8,3.0,2.7
    assert A-B>0 and C-D>0 and abs(((C-D)-(A-B))-.1)<1e-12
    print('SELF_TEST PASS')

def load():
    required=[OUT/'rows.csv',OUT/'learning.npz',OUT/'provenance.json',OUT/'validation.json']
    if any(not p.exists() for p in required):raise FileNotFoundError('missing required result: '+', '.join(str(p) for p in required if not p.exists()))
    provenance=json.loads(required[2].read_text()); validation=json.loads(required[3].read_text())
    if provenance.get('status')!='COMPLETE':raise RuntimeError('provenance status is not COMPLETE')
    if validation.get('status')!='PASS':raise RuntimeError('validation status is not PASS')
    rows=read_csv(required[0]); expected={(s,b,t,u) for s in SEEDS for b in BRANCHES for t in TASKS for u in STEPS}
    actual={(int(r['seed']),r['branch'],int(r['task']),int(r['step'])) for r in rows}
    if len(rows)!=len(expected) or actual!=expected:raise RuntimeError(f'rows completeness mismatch: rows={len(rows)}, unique={len(actual)}, expected={len(expected)}')
    if not all(np.isfinite(float(r[k])) for r in rows for k in ('ce','acc')):raise RuntimeError('nonfinite full-probe result')
    if not all(0<=float(r['acc'])<=1 for r in rows):raise RuntimeError('full-probe accuracy out of bounds')
    lookup={(int(r['seed']),r['branch'],int(r['task']),int(r['step'])):r for r in rows}
    with np.load(required[1],allow_pickle=False) as z:
        needed={'online_ce','online_acc','tasks','model_seed','model_branch'}
        if not needed.issubset(z.files):raise RuntimeError(f'learning.npz keys {z.files}; need {sorted(needed)}')
        ce=np.asarray(z['online_ce']); acc=np.asarray(z['online_acc']); tasks=np.asarray(z['tasks']).astype(int)
        seeds=np.asarray(z['model_seed']).astype(int); branches=np.asarray(z['model_branch']).astype(str)
    if ce.shape!=(5,18,6000) or acc.shape!=ce.shape:raise RuntimeError(f'online shape mismatch: CE {ce.shape}, accuracy {acc.shape}')
    if list(tasks)!=list(TASKS):raise RuntimeError(f'online task axis mismatch: {tasks}')
    keys=list(zip(seeds.tolist(),branches.tolist())); expected_models={(s,b) for s in SEEDS for b in BRANCHES}
    if len(keys)!=18 or set(keys)!=expected_models:raise RuntimeError(f'online model metadata mismatch: {keys}')
    if not np.isfinite(ce).all() or not np.isfinite(acc).all():raise RuntimeError('nonfinite online result')
    if not ((acc>=0)&(acc<=1)).all():raise RuntimeError('online accuracy out of bounds')
    model_index={k:i for i,k in enumerate(keys)}
    return required,provenance,validation,rows,lookup,ce,acc,model_index

def main():
    required,provenance,validation,rows,probe,online_ce,online_acc,mi=load()
    values={}
    for s in SEEDS:
      for b in BRANCHES:
       j=mi[s,b]
       for ti,t in enumerate(TASKS):
        rr=[probe[s,b,t,u] for u in STEPS]; y=[float(r['ce']) for r in rr]
        values[s,b,t]=dict(online_ce=float(np.mean(online_ce[ti,j],dtype=np.float64)),online_acc=float(np.mean(online_acc[ti,j],dtype=np.float64)),
          endpoint_ce=float(rr[-1]['ce']),endpoint_acc=float(rr[-1]['acc']),probe_auc=auc(y,STEPS,6000),early75_auc=auc(y[:9],STEPS[:9],75))
    metrics=[]
    definitions={
      'online_ce_first75_task21':lambda s,b:float(np.mean(online_ce[0,mi[s,b],:75],dtype=np.float64)),
      'online_acc_first75_task21_pp_cost':lambda s,b:-100*float(np.mean(online_acc[0,mi[s,b],:75],dtype=np.float64)),
      'probe_auc_task21':lambda s,b:values[s,b,21]['probe_auc'],
      'early75_probe_auc_task21':lambda s,b:values[s,b,21]['early75_auc'],
      'endpoint_ce_task21':lambda s,b:values[s,b,21]['endpoint_ce'],
      'endpoint_acc_task21_pp_cost':lambda s,b:-100*values[s,b,21]['endpoint_acc'],
      'online_ce_all5_mean':lambda s,b:float(np.mean([values[s,b,t]['online_ce'] for t in TASKS])),
      'online_acc_all5_mean_pp_cost':lambda s,b:-100*float(np.mean([values[s,b,t]['online_acc'] for t in TASKS])),
      'endpoint_ce_all5_mean':lambda s,b:float(np.mean([values[s,b,t]['endpoint_ce'] for t in TASKS])),
      'endpoint_acc_all5_mean_pp_cost':lambda s,b:-100*float(np.mean([values[s,b,t]['endpoint_acc'] for t in TASKS])),
    }
    for name,get in definitions.items():
      for s in SEEDS:
       v={b:get(s,b) for b in BRANCHES}
       metrics.append(dict(metric=name,seed=s,A_minus_B=v['A']-v['B'],C_minus_D=v['C']-v['D'],response_interaction=(v['C']-v['D'])-(v['A']-v['B']),AF_minus_BF=v['AF']-v['BF'],A_minus_C=v['A']-v['C'],B_minus_D=v['B']-v['D'],A_minus_D=v['A']-v['D']))
    # Taskwise durability retains task identity; no tasks-as-replicates averaging.
    for t in TASKS:
      for s in SEEDS:
       v={b:values[s,b,t]['online_ce'] for b in BRANCHES}
       metrics.append(dict(metric=f'online_ce_task{t}',seed=s,A_minus_B=v['A']-v['B'],C_minus_D=v['C']-v['D'],response_interaction=(v['C']-v['D'])-(v['A']-v['B']),AF_minus_BF=v['AF']-v['BF'],A_minus_C=v['A']-v['C'],B_minus_D=v['B']-v['D'],A_minus_D=v['A']-v['D']))
      for s in SEEDS:
       v={b:-100*values[s,b,t]['online_acc'] for b in BRANCHES}
       metrics.append(dict(metric=f'online_acc_task{t}_pp_cost',seed=s,A_minus_B=v['A']-v['B'],C_minus_D=v['C']-v['D'],response_interaction=(v['C']-v['D'])-(v['A']-v['B']),AF_minus_BF=v['AF']-v['BF'],A_minus_C=v['A']-v['C'],B_minus_D=v['B']-v['D'],A_minus_D=v['A']-v['D']))
    require_unique(metrics,('metric','seed'))
    write_csv(OUT/'contrasts.csv',metrics)
    verdict=[]
    for metric in dict.fromkeys(r['metric'] for r in metrics):
      mm=[r for r in metrics if r['metric']==metric]
      for contrast in ('A_minus_B','C_minus_D','response_interaction','AF_minus_BF','A_minus_C','B_minus_D','A_minus_D'):
       vv=[next(float(r[contrast]) for r in mm if int(r['seed'])==s) for s in SEEDS]; mean,sd,lo,hi=ci3(vv)
       primary=metric=='online_ce_task21' and contrast=='A_minus_B'
       verdict.append(dict(metric=metric,contrast=contrast,seed0=vv[0],seed1=vv[1],seed2=vv[2],mean=mean,sd=sd,ci95_low=lo,ci95_high=hi,scope='PRIMARY' if primary else 'SECONDARY',pilot_label=classify(vv) if primary else 'SECONDARY_NOT_CLASSIFIED'))
    write_csv(OUT/'verdict.csv',verdict)
    shock=[]
    for s in SEEDS:
      for shallow,feature,k in (('A','C','K0'),('B','D','K1')):
       x=probe[s,shallow,21,0];y=probe[s,feature,21,0]
       shock.append(dict(seed=s,pair=f'{shallow}_to_{feature}',K=k,F0_branch=shallow,F1_branch=feature,F1_minus_F0_ce=float(y['ce'])-float(x['ce']),F1_minus_F0_acc_pp=100*(float(y['acc'])-float(x['acc']))))
    write_csv(OUT/'initial_shock.csv',shock)
    endpoints=[]
    for s in SEEDS:
      for b in BRANCHES:
       for t in TASKS:
        ti=t-21;j=mi[s,b]
        endpoints.append(dict(seed=s,branch=b,task=t,online_ce=values[s,b,t]['online_ce'],online_acc=values[s,b,t]['online_acc'],online_ce_first75=float(np.mean(online_ce[ti,j,:75],dtype=np.float64)),online_acc_first75=float(np.mean(online_acc[ti,j,:75],dtype=np.float64)),endpoint_ce=values[s,b,t]['endpoint_ce'],endpoint_acc=values[s,b,t]['endpoint_acc'],probe_auc=values[s,b,t]['probe_auc'],early75_probe_auc=values[s,b,t]['early75_auc']))
    require_unique(endpoints,('seed','branch','task'));write_csv(OUT/'seed_endpoints.csv',endpoints)

    fig,axs=plt.subplots(1,2,figsize=(13,5.2));fig.subplots_adjust(top=.77,bottom=.16,wspace=.25)
    for b in ('A','B','C','D'):
      color,ls=STYLE[b]; curves=np.asarray([[100*values[s,b,t]['endpoint_acc'] for t in TASKS] for s in SEEDS])
      for x in curves:axs[0].plot(list(TASKS),x,color=color,ls=ls,lw=.7,alpha=.2)
      axs[0].plot(list(TASKS),curves.mean(0),color=color,ls=ls,lw=2.3,marker='o',ms=4,label=b)
      curves=np.asarray([[float(probe[s,b,21,u]['ce']) for u in STEPS] for s in SEEDS])
      for x in curves:axs[1].plot(STEPS,x,color=color,ls=ls,lw=.7,alpha=.2)
      axs[1].plot(STEPS,curves.mean(0),color=color,ls=ls,lw=2.3,label=b)
    axs[0].set(title='Fresh-task endpoint learning',xlabel='Task',ylabel='Train accuracy at update 6000 (%)',xticks=list(TASKS));axs[0].grid(alpha=.18)
    axs[1].set_xscale('symlog',linthresh=1);axs[1].set_xticks(STEPS,[str(x) for x in STEPS],rotation=45,ha='right',fontsize=7)
    axs[1].set(title='Task 21 full-probe CE trajectory',xlabel='Updates (symlog; step 0 retained)',ylabel='Full-1200 probe CE');axs[1].grid(alpha=.18)
    inset=axs[1].inset_axes([.52,.50,.45,.43])
    ab=[]
    for b in ('A','B'):
      color,ls=STYLE[b];curve=np.mean([[float(probe[s,b,21,u]['ce']) for u in STEPS] for s in SEEDS],axis=0);ab.extend(curve)
      inset.plot(STEPS,curve,color=color,ls=ls,lw=1.6,label=b)
    inset.set_xscale('symlog',linthresh=1);inset.set_xticks((0,20,75,375,6000),('0','20','75','375','6000'),fontsize=6)
    pad=max((max(ab)-min(ab))*.12,.01);inset.set_ylim(min(ab)-pad,max(ab)+pad);inset.set_title('A/B zoom',fontsize=8);inset.tick_params(axis='y',labelsize=6);inset.grid(alpha=.15)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False)
    fig.suptitle('Response-anchor intervention: registered descriptive outcomes',y=.88,fontsize=14)
    fig.text(.5,.025,'A=F0K0, B=F0K1, C=F1K0, D=F1K1. F1 changes initial function; solid/dashed compare K within F.',ha='center',fontsize=9)
    fig.savefig(OUT/'response_anchor_outcomes.png',dpi=180);plt.close(fig)

    p=next(r for r in verdict if r['scope']=='PRIMARY')
    lines=['# ELU response-anchor results','', '**Preregistered 3-seed pilot.** Seeds are the replicate unit; tasks and units are not replicates. The primary outcome is mean online minibatch CE over all 6,000 updates of task 21.','', 'A=F0K0, B=F0K1, C=F1K0, D=F1K1; AF/BF freeze layer 1. Positive A-B means the K1 response architecture lowers online CE within F0.','', '## Primary per seed','', '| seed | A online CE | B online CE | A-B benefit |','|---:|---:|---:|---:|']
    for s in SEEDS:lines.append(f"| {s} | {values[s,'A',21]['online_ce']:.6f} | {values[s,'B',21]['online_ce']:.6f} | {float(next(r for r in metrics if r['metric']=='online_ce_task21' and int(r['seed'])==s)['A_minus_B']):.6f} |")
    lines += ['',f"Mean A-B = {float(p['mean']):.6f}; t95% CI with df=2 [{float(p['ci95_low']):.6f}, {float(p['ci95_high']):.6f}]. Pilot label: **{p['pilot_label']}**. This is a directional pilot threshold, not a population-significance claim.",'','## Task 21 response contrasts','', '| seed | C-D | (C-D)-(A-B) | AF-BF |','|---:|---:|---:|---:|']
    primary_rows=[r for r in metrics if r['metric']=='online_ce_task21']
    for s in SEEDS:
      r=next(x for x in primary_rows if int(x['seed'])==s);lines.append(f"| {s} | {r['C_minus_D']:.6f} | {r['response_interaction']:.6f} | {r['AF_minus_BF']:.6f} |")
    lines.append('| mean | '+' | '.join(f"{np.mean([float(r[k]) for r in primary_rows]):.6f}" for k in ('C_minus_D','response_interaction','AF_minus_BF'))+' |')
    lines += ['','All three are registered secondary descriptions and receive no primary support label. AF/BF freeze layer 1, while layer 2 and output parameters remain trainable.','','## Endpoint accuracy and durability','', '| task | A acc % | B acc % | C acc % | D acc % | AF acc % | BF acc % |','|---:|---:|---:|---:|---:|---:|---:|']
    for t in TASKS:lines.append('| '+str(t)+' | '+' | '.join(f"{100*np.mean([values[s,b,t]['endpoint_acc'] for s in SEEDS]):.3f}" for b in BRANCHES)+' |')
    lines += ['','Full-probe AUC, early-75 probe AUC, first-75 online means, endpoint CE/accuracy, taskwise durability, C-D, (C-D)-(A-B), AF-BF, total ordinary lift A-D, and feature effects A-C/B-D are registered secondary descriptions in `contrasts.csv` and `verdict.csv`.','', 'C-D and A-B compare K within a fixed F state. A-C and B-D change F and therefore start from different functions; they are feature effects with an initial-difference label, not pure update effects. `initial_shock.csv` reports those step-0 differences separately.','', 'The intervention identifies an architecture response under matched inputs. Within each F state, corresponding branches have the same full initial function; initial W/Adam state is identical as registered. Output equivalence is initial only. Do not interpret any contrast as a natural mediated percentage.','', '## Source and validation','',f"- rows.csv SHA256 `{sha(required[0])}`",f"- learning.npz SHA256 `{sha(required[1])}`",f"- provenance status `{provenance['status']}`; validation status `{validation['status']}`",'- Exact required grid: 3 seeds x 6 branches x 5 tasks x 15 full-probe steps; dense online shape 5 x 18 x 6000.','']
    (OUT/'summary.md').write_text('\n'.join(lines))
    audit=dict(status='PASS',source_sha256={p.name:sha(p) for p in required},row_count=len(rows),row_grid_complete=True,online_shape=list(online_ce.shape),online_model_metadata_complete=True,contrast_metric_seed_unique=True,seed_endpoint_rows=len(endpoints),finite=True,seed_replication=3,primary='task21 mean online minibatch CE over updates1..6000; A-B positive benefit',primary_label=p['pilot_label'],no_task_or_unit_pseudoreplication=True)
    (OUT/'aggregation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    self_test() if args.self_test else main()

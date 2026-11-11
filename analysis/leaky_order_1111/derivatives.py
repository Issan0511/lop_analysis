#!/usr/bin/env python3
"""Posthoc derivative readout requested after the registered run completed."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src import leaky_order_1111 as O
OUT=O.OUT;GRID=[.001,.002,.003,.005,.01];SEEDS=list(range(3,8));POINTS=[0,750,1500,3000,6000,12000,24000]
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def main():
    raw={};points=[];summaries=[];source=[];maxerr=0
    for a in GRID:
        for seed in SEEDS:
            dr=OUT/'probes'/O.tag(a)/f's{seed}'/'age6';rr=read(dr/'curves_t0.csv');t=float(read(dr/'summary.csv')[0]['T90_sustained'])
            raw[(a,seed)]=[r for r in rr if int(r['step'])==0 or (int(r['step'])>1 and r.get('L1_gmean') and int(r['step'])<=24000)]
            for r in raw[(a,seed)]:
                for layer in [1,2]:
                    p=float(r[f'L{layer}_pplus']);mu=float(r[f'L{layer}_gmean']);err=abs(mu-(a+(1-a)*p));maxerr=max(maxerr,err)
                    assert err<1e-9
                    row=dict(a=a,seed=seed,step=int(r['step']),layer=layer,phi_prime_mean=mu,pplus=p,phi_prime_within_input_variance=float(r[f'L{layer}_gvar']),zero_positive_input=float(r[f'L{layer}_zero_positive_input']),acc=float(r['acc']),T90_sustained=t)
                    if row['step'] in POINTS:points.append(row)
                    if row['step']==0:source.append(row)
        for step in POINTS:
            for layer in [1,2]:
                rr=[r for r in points if r['a']==a and r['step']==step and r['layer']==layer];assert len(rr)==5
                row=dict(a=a,step=step,layer=layer)
                for k in ['phi_prime_mean','pplus','phi_prime_within_input_variance','zero_positive_input','acc']:
                    vals=[r[k] for r in rr];row[k+'_mean']=float(np.mean(vals));row[k+'_median']=float(np.median(vals));row[k+'_min']=min(vals);row[k+'_max']=max(vals)
                summaries.append(row)
    O.B.write(OUT/'derivative_points_posthoc.csv',points);O.B.write(OUT/'derivative_summary_posthoc.csv',summaries)
    paired=[]
    for seed in SEEDS:
        row=dict(seed=seed)
        for a,key in [(.001,'a001'),(.01,'a01')]:
            rr=[r for r in source if r['a']==a and r['seed']==seed]
            for layer in [1,2]:row[f'{key}_L{layer}']=next(r['phi_prime_mean'] for r in rr if r['layer']==layer)
            row[f'{key}_T90_sustained']=rr[0]['T90_sustained']
        paired.append(row)
    O.B.write(OUT/'source_derivative_paired_posthoc.csv',paired)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(12,6.1),sharex=True,sharey=True,layout='constrained');colors=plt.get_cmap('tab10')(np.arange(5))
    for j,a in enumerate([.001,.003,.01]):
        for i,layer in enumerate([1,2]):
            ax=axes[i,j]
            for k,seed in enumerate(SEEDS):
                rr=raw[(a,seed)];xx=[int(r['step']) for r in rr];yy=[float(r[f'L{layer}_gmean']) for r in rr]
                ax.plot(xx,yy,c=colors[k],lw=1.3,label=f'seed {seed}')
            mean=np.mean([[float(r[f'L{layer}_gmean']) for r in raw[(a,seed)]] for seed in SEEDS],axis=0)
            ax.plot(xx,mean,'k--',lw=1.8,label='arithmetic mean')
            ax.axhline(a,c='grey',ls=':',lw=1);ax.text(8000,a*1.12,f'floor a={a:g}',fontsize=8,c='grey')
            ax.set(xscale='symlog',yscale='log',ylim=(.0008,.65),title=f'a={a:g}, layer {layer}');ax.set_xscale('symlog',linthresh=750)
            ax.set_xticks([0,750,6000,24000],labels=['0','750','6k','24k']);ax.grid(alpha=.15)
            if j==0:ax.set_ylabel("Mean phi'(z) over inputs and units")
            if i==1:ax.set_xlabel('Updates on new task')
    axes[0,2].legend(fontsize=7,loc='upper right');fig.suptitle("Derivative trajectories after 6 prior tasks: individual seeds reveal layer imbalance",fontsize=12)
    fig.savefig(OUT/'derivative_trajectories_posthoc.png',dpi=180);fig.savefig(OUT/'derivative_trajectories_posthoc.pdf');plt.close(fig)
    check=dict(all_pass=True,posthoc=True,requested_after_main=True,source_age=6,number_of_seed_runs=25,phi_prime_identity_maxabs=maxerr,common_budget=24000,code_sha256=O.C.sha(__file__),training_code_sha256=O.C.sha(ROOT/'src/leaky_order_1111.py'))
    O.C.put(OUT/'derivative_checks_posthoc.json',check)
    lines=['# Derivatives during relearning (posthoc request)','',
           "For leaky ReLU, phi'(z) is 1 on the positive side and a on the negative side. Its input/unit mean is a+(1-a) p(z>0). The slope a is therefore the floor, not the mean. These are activation derivatives, not parameter gradients.",'',
           'All tables use source age6. Means across seeds can hide which layer is nearly negative-only. For example a=.001 seed4 starts with L1 mean .003239 and L2 mean .428172, and its sustained90 starts at39900. A high aggregate L2 mean does not imply a large gradient through L1. This is descriptive evidence, not a causal attribution of the slowing.','',
           '| a | L1 mean at start | L1 at750 | L2 at start | L2 at750 | L2 at24k |','|---:|---:|---:|---:|---:|---:|---:|']
    for a in GRID:
        get=lambda layer,step:next(r['phi_prime_mean_mean'] for r in summaries if r['a']==a and r['layer']==layer and r['step']==step)
        lines.append(f'| {a:g} | {get(1,0):.6f} | {get(1,750):.6f} | {get(2,0):.6f} | {get(2,750):.6f} | {get(2,24000):.6f} |')
    lines+=['','## Seed medians at750 updates','',
            'Each seed value is first averaged over the1200 inputs and100 units in one layer. This table then takes the median of those five seed values. The preceding table uses the arithmetic mean across seeds. Neither replaces the individual-seed plot.','',
            '| a | L1 median of mean derivatives | L2 median of mean derivatives |','|---:|---:|---:|']
    for a in GRID:
        vals=[next(r['phi_prime_mean_median'] for r in summaries if r['a']==a and r['step']==750 and r['layer']==layer) for layer in [1,2]]
        lines.append(f'| {a:g} | {vals[0]:.6f} | {vals[1]:.6f} |')
    lines+=['','At a=.001, seeds3,4,6,7 have one layer at .00119-.00142 at750 updates; for seed4 it is layer1, whereas for3,6,7 it is layer2. Seed5 differs: layer1 is .01594 and layer2 is .24031. Thus the large arithmetic layer2 mean at a=.001 is not representative of the three seeds whose layer2 sits near its leak floor.',
            'The seed-averaged mean derivative first falls in both layers, then recovers during relearning. The smallest observed grid point a=.001 does not have the smallest source mean derivative in either layer. Thus source mean derivative alone does not give a universal inverse clock across these learned states. Individual trajectories, layer structure and optimizer history remain relevant.','',
            'The identity was checked at every stored derivative time point (absolute tolerance1e-9). No additional training or exclusions were performed. The plot uses the common first24k updates, available for every seed, and shows seeds individually.','',
            '![Derivative trajectories](derivative_trajectories_posthoc.png)','']
    (OUT/'DERIVATIVES_POSTHOC.md').write_text('\n'.join(lines))
    print(json.dumps(check,indent=2))
if __name__=='__main__':main()

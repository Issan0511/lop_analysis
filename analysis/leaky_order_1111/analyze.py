#!/usr/bin/env python3
"""Descriptive orders of time, with all right-censored runs retained."""
import csv,json,math,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src import leaky_order_1111 as O
OUT=O.OUT;GRID=[.001,.002,.003,.005,.01];AGES=[0,3,6];SEEDS=list(range(3,8))
def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def number(r,k):return float(r[k]) if r.get(k) else None
def show_time(v,c=False):return ('>' if c else '')+f'{v:,.0f}'
def main():
    complete=json.loads((OUT/'COMPLETE.json').read_text());assert complete['probes']==75 and complete['prefixes']==20
    allrows=[];curves={};refs=[];codes=set();labels={s:set() for s in SEEDS};batch_checks=json.loads((OUT/'paired_stream_checks.json').read_text());assert batch_checks['all_pass']
    for age in AGES:
        for a in GRID:
            for seed in SEEDS:
                dr=OUT/'probes'/O.tag(a)/f's{seed}'/f'age{age}'
                s=rows(dr/'summary.csv')[0];raw=rows(dr/'curves_t0.csv');ident=json.loads((dr/'provenance.json').read_text());codes.add(ident['code_sha256'])
                br=json.loads((dr/'branch_check.json').read_text());labels[seed].add(br['probe_labels_sha256'])
                assert br['a']==a and br['age']==age and (dr/'done.json').exists()
                steps=int(s['steps']);assert steps>=24000 and int(raw[-1]['step'])==steps
                clocks={};streak=0
                for r in raw:
                    z=dict(step=int(r['step']),correct=int(r['correct']),n=int(r['n']),ce=float(r['ce']))
                    assert z['n']==1200 and 0<=z['correct']<=z['n']
                    streak=O.update_times(z,clocks,streak)
                for key in ['T80','T90','T90_sustained','T95','T99','Tce2.0','Tce1.5','Tce1.0','Tce0.5','Tce0.2','Tce0.1']:
                    assert number(s,key)==clocks.get(key),(a,seed,age,key)
                t=number(s,'T90_sustained');censored=t is None
                if censored:assert steps==O.MAX
                else:assert steps==max(24000,int(t)+225)
                # A start time needs three later confirmations. Without one by C,
                # its rigorous lower bound is C-225 (first T90 uses C).
                lower=steps-225 if censored else t
                r=dict(a=a,seed=seed,age=age,steps=steps,T90=number(s,'T90'),T90_sustained=t,T90_censored=number(s,'T90') is None,
                       T90_sustained_censored=censored,T90_sustained_lower=lower,
                       acc_6000=number(s,'acc_6000'),acc_24000=number(s,'acc_24000'),endpoint_acc=number(s,'endpoint_acc'),endpoint_ce=number(s,'endpoint_ce'),seconds=float(s['seconds']))
                source=raw[0]
                for layer in [1,2]:
                    for k in ['gmean','pplus','gvar','zero_positive_input','one_sided','gate_participation','unique_gate_patterns']:
                        r[f'source_L{layer}_{k}']=number(source,f'L{layer}_{k}')
                for budget in O.BUDGETS:
                    r[f'sustained_start_by_{budget}']=t is not None and t<=budget
                    r[f'sustained_confirmed_by_{budget}']=t is not None and t+225<=budget
                    r[f'first90_by_{budget}']=number(s,'T90') is not None and float(s['T90'])<=budget
                allrows.append(r);curves[(age,a,seed)]=raw
                if a==.003:
                    ref=json.loads((dr/'reference_check.json').read_text());assert ref['all_pass'];refs.append(ref)
    assert len(codes)==1 and next(iter(codes))==O.C.sha(ROOT/'src/leaky_order_1111.py')
    assert all(len(x)==1 for x in labels.values()) and len(refs)==15
    O.B.write(OUT/'all_probes.csv',allrows);groups=[]
    for age in AGES:
        for a in GRID:
            rr=[r for r in allrows if r['age']==age and r['a']==a];obs=sorted(r['T90_sustained'] for r in rr if not r['T90_sustained_censored']);lower=sorted(r['T90_sustained_lower'] for r in rr)
            med=obs[2] if len(obs)>=3 else None
            first=sorted(r['T90'] for r in rr if r['T90'] is not None)
            g=dict(age=age,a=a,n=5,recovered=len(obs),median_T90_sustained=med,median_lower=lower[2],median_censored=med is None,median_epochs_lower=lower[2]/75,median_first_T90=first[2] if len(first)>=3 else None,
                   minimum_lower=lower[0],maximum_lower=lower[-1],any_censored=len(obs)<5,mean_time_lower=float(np.mean(lower)),
                   mean_acc_6000=float(np.mean([r['acc_6000'] for r in rr])),mean_acc_24000=float(np.mean([r['acc_24000'] for r in rr])),
                   per_seed_times='; '.join(f's{r["seed"]}='+show_time(r['T90_sustained_lower'],r['T90_sustained_censored']) for r in rr))
            for b in O.BUDGETS:
                for name in ['sustained_start','sustained_confirmed','first90']:g[f'{name}_by_{b}']=sum(r[f'{name}_by_{b}'] for r in rr)
            groups.append(g)
    O.B.write(OUT/'group_summary.csv',groups)
    ratios=[];slopes=[]
    for age in AGES:
        for seed in SEEDS:
            lookup={r['a']:r for r in allrows if r['age']==age and r['seed']==seed}
            for lo,hi in list(zip(GRID[:-1],GRID[1:]))+[(.001,.01)]:
                l,h=lookup[lo],lookup[hi];lc,hc=l['T90_sustained_censored'],h['T90_sustained_censored']
                val=l['T90_sustained_lower']/h['T90_sustained_lower']
                qualifier='unidentified' if lc and hc else ('lower_bound' if lc else ('upper_bound' if hc else 'observed'))
                rec=dict(age=age,seed=seed,a_low=lo,a_high=hi,time_low_over_high=None if lc and hc else val,qualifier=qualifier)
                ratios.append(rec)
                if qualifier=='observed':slopes.append(dict(**rec,exploratory_exponent=math.log(val)/math.log(hi/lo)))
    O.B.write(OUT/'paired_ratios.csv',ratios);O.B.write(OUT/'exploratory_local_slopes.csv',slopes)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=plt.get_cmap('tab10')(np.arange(5));fig,axes=plt.subplots(1,3,figsize=(12.2,4.1),sharey=True,layout='constrained')
    for ax,age in zip(axes,AGES):
        for i,seed in enumerate(SEEDS):
            rr=sorted([r for r in allrows if r['age']==age and r['seed']==seed],key=lambda r:r['a'])
            ax.plot(GRID,[r['T90_sustained_lower'] for r in rr],'-o',c=colors[i],alpha=.62,lw=1,ms=4,label=f'seed {seed}')
            for r in rr:
                if r['T90_sustained_censored']:ax.scatter(r['a'],r['T90_sustained_lower'],marker='^',c=[colors[i]],s=65)
        gg=[r for r in groups if r['age']==age];ax.plot(GRID,[r['median_lower'] for r in gg],'k--',lw=2,label='median / lower bound')
        ax.axhline(6000,color='grey',ls=':',lw=1);ax.text(.00102,6100,'usual 6,000',fontsize=8,color='grey',va='bottom')
        ax.set(xscale='log',yscale='log',xlabel='Negative slope a',title=f'After {age} prior tasks')
        ax.set_xticks(GRID,labels=['.001','.002','.003','.005','.01']);ax.minorticks_off();ax.grid(True,which='major',alpha=.15)
    axes[0].set_ylabel('Updates to sustained 90% training accuracy')
    max_t=max(r['T90_sustained_lower'] for r in allrows);axes[0].set_ylim(700,max(35000,max_t*1.5))
    axes[-1].legend(fontsize=7,loc='best');fig.suptitle('Leak .001-.01: measured adaptation times (5 paired seeds)',fontsize=13)
    fig.savefig(OUT/'time_orders.png',dpi=180);fig.savefig(OUT/'time_orders.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.2,3.7),sharey=True,layout='constrained')
    for ax,age in zip(axes,AGES):
        for ai,a in enumerate(GRID):
            traces=[];xs=None
            for seed in SEEDS:
                rr=[r for r in curves[(age,a,seed)] if 0<int(r['step'])<=24000 and int(r['step'])%75==0]
                x=np.array([int(r['step']) for r in rr]);y=np.array([float(r['acc']) for r in rr]);assert len(x)==320
                if xs is None:xs=x
                else:assert np.array_equal(xs,x)
                traces.append(y)
            data=np.array(traces);color=plt.get_cmap('viridis')(ai/4)
            ax.plot(xs,data.mean(0),color=color,label=f'a={a:g}',lw=1.5);ax.fill_between(xs,data.min(0),data.max(0),color=color,alpha=.08)
        ax.axhline(.9,color='grey',ls=':',lw=1);ax.set(xscale='log',xlabel='Updates on unchanged new task',title=f'After {age} prior tasks',ylim=(0,1.02));ax.grid(alpha=.12)
    axes[0].set_ylabel('Training accuracy (mean; band = seed range)');axes[-1].legend(fontsize=8)
    fig.savefig(OUT/'common_budget_curves.png',dpi=180);fig.savefig(OUT/'common_budget_curves.pdf');plt.close(fig)
    primary={g['a']:g for g in groups if g['age']==6};paired=[r['time_low_over_high'] for r in ratios if r['age']==6 and r['a_low']==.001 and r['a_high']==.01 and r['qualifier']=='observed']
    lines=['# Adaptation-time orders across a=.001-.01','',
           f'All {sum(not r["T90_sustained_censored"] for r in allrows)}/75 probes reached sustained90. The latest start was {max(r["T90_sustained_lower"] for r in allrows):,.0f} updates. At source age6, the median was {primary[.001]["median_lower"]:,.0f} for a=.001 and {primary[.01]["median_lower"]:,.0f} for a=.01. The five paired ratios were {min(paired):.2f}-{max(paired):.2f} (median {np.median(paired):.2f}). These are tens of thousands of updates, not hundreds of thousands or millions in this registered setting.',
           'The intermediate grid is not strictly monotone: a=.002 has a larger sample median than .001, and .005 a smaller sample median than .01. Five seeds and differing accumulated states do not identify a sharp threshold or a universal inverse-a law. Fresh networks stay near 1,400-1,500 updates across the grid. The substantial slowdown appears after repeated tasks.','',
           'Registered design: five paired seeds, 1200 random-label MNIST samples, two hidden layers of 100 units, Adam .001. Source ages 0, 3, 6; each old task used 6000 updates. Every probe has a single unchanged new task. This is a new host experiment, not a reconstruction of CondA.','',
           'Primary time is the start of four consecutive evaluations >=90% (75-update spacing, confirmation 225 updates later). Exactly 1080/1200 qualifies. The common minimum budget is 24k; adaptive maximum is 1,536,000. The negative leak is not the mean derivative. Cross-leak comparisons include different accumulated learned states.','',
           '| Prior tasks | a | Recovered | Median sustained T90 | Per-seed sustained T90 (updates) | Mean accuracy at 6k | At 24k |','|---:|---:|---:|---:|---|---:|---:|']
    for g in groups:lines.append(f'| {g["age"]} | {g["a"]:g} | {g["recovered"]}/5 | {show_time(g["median_lower"],g["median_censored"])} | {g["per_seed_times"]} | {100*g["mean_acc_6000"]:.2f}% | {100*g["mean_acc_24000"]:.2f}% |')
    lines+=['','## Verification and interpretation','',
            f'All 75 time records recomputed from integer correct counts. All 15 reused-a=.003 comparisons match prior CE exactly and match integer accuracy at common evaluation times. Source initial parameters/moments and prefix label/batch RNG states match across leaks. All seed-specific probe labels match. Training code SHA256: `{next(iter(codes))}`.',
            'The older runner compared a float32 accuracy mean with decimal .9, so exactly 90% could be excluded. This runner uses integer correct counts; CE and all training arithmetic are unchanged. Do not treat .003 reruns as new independent seeds.',
            'A right-censored first T90 means T90 > last evaluated update. For sustained start time, the rigorous lower bound is C-225 because three later evaluations are required to confirm the start. No censored case is omitted from summaries. Secondary T95/T99 can be censored by stopping at sustained90. Event counts distinguish start by budget from confirmation by budget.',
            'The minimum24k curve includes every run. Later accuracies from surviving slow runs must not be averaged as though they represent the original cohort. Finite-budget failures do not demonstrate impossibility or a mathematical phase transition. Local log-log slopes in exploratory_local_slopes.csv are descriptive, not a universal law.','',
            '![Time orders](time_orders.png)','![Common-budget curves](common_budget_curves.png)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    O.C.put(OUT/'analysis_checks.json',dict(all_pass=True,probes=75,time_records_recomputed=75,old_reference_checks=15,labels_paired=True,source_rng_paired=True,training_code_sha256=next(iter(codes))))
    print(json.dumps(groups,indent=2))
if __name__=='__main__':main()

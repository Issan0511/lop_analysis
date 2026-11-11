#!/usr/bin/env python3
"""Posthoc inspection of the two groups seen at update750; no new training."""
import csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'results/leaky_order_1111';OUT=ROOT/'results/leaky_two_groups_1111'
GRID=[.001,.002,.003,.005,.01];STEPS=[0,750,1500,3000,6000];COL={'L1_smaller':'#c44e52','L2_smaller':'#287db5'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def fit(rows):
    x=np.log([r['product'] for r in rows]);y=np.log([r['T90_sustained'] for r in rows]);b,c=np.polyfit(x,y,1)
    return dict(loglog_slope=float(b),loglog_intercept=float(c),loglog_R2=float(np.corrcoef(x,y)[0,1]**2))
def main():
    OUT.mkdir(parents=True,exist_ok=True);allrows=[];manifest=[];changes=[];checks=0
    for a in GRID:
        for seed in range(3,8):
            dr=SRC/'probes'/format(a,'.3f').replace('.','p')/f's{seed}'/'age6';rr=read(dr/'curves_t0.csv');s=read(dr/'summary.csv')[0];t=float(s['T90_sustained'])
            for name in ['curves_t0.csv','summary.csv']:manifest.append(dict(path=str((dr/name).relative_to(ROOT)),sha256=sha(dr/name)))
            at750=next(r for r in rr if int(r['step'])==750);group='L1_smaller' if float(at750['L1_gmean'])<float(at750['L2_gmean']) else 'L2_smaller'
            for r in rr:
                if int(r['step'])<=6000 and r.get('L1_gmean'):
                    actual='L1_smaller' if float(r['L1_gmean'])<float(r['L2_gmean']) else 'L2_smaller';checks+=1
                    if actual!=group:changes.append(dict(a=a,seed=seed,step=r['step'],group=actual))
                if int(r['step']) not in STEPS:continue
                mu1=float(r['L1_gmean']);mu2=float(r['L2_gmean']);step=int(r['step']);assert step<t
                row=dict(a=a,seed=seed,step=step,group=group,mu1=mu1,mu2=mu2,ratio_mu2_mu1=mu2/mu1,product=mu1*mu2,T90_sustained=t,acc=float(r['acc']))
                for layer in [1,2]:
                    for key in ['gvar','one_sided','zero_positive_input']:row[f'L{layer}_{key}']=float(r[f'L{layer}_{key}'])
                allrows.append(row)
    at750=[r for r in allrows if r['step']==750];assert len(at750)==25 and not changes
    write(OUT/'per_seed_groups.csv',allrows);summaries=[];fits=[];paired=[]
    for step in STEPS:
        rr=[r for r in allrows if r['step']==step]
        for group in ['pooled','L1_smaller','L2_smaller']:
            gg=rr if group=='pooled' else [r for r in rr if r['group']==group]
            f=dict(step=step,group=group,n=len(gg),**fit(gg));fits.append(f)
            row=dict(step=step,group=group,n=len(gg))
            for k in ['mu1','mu2','ratio_mu2_mu1','product','T90_sustained','acc','L1_gvar','L2_gvar','L1_one_sided','L2_one_sided','L1_zero_positive_input','L2_zero_positive_input']:
                values=[r[k] for r in gg];row[k+'_median']=float(np.median(values));row[k+'_min']=min(values);row[k+'_max']=max(values)
            summaries.append(row)
    for a in GRID:
        aa=[r for r in at750 if r['a']==a];row=dict(a=a)
        for group in ['L1_smaller','L2_smaller']:
            tt=[r['T90_sustained'] for r in aa if r['group']==group];row[group+'_n']=len(tt);row[group+'_T90_median']=float(np.median(tt));row[group+'_T90_min']=min(tt);row[group+'_T90_max']=max(tt)
        row['all_L1_group_slower']=row['L1_smaller_T90_min']>row['L2_smaller_T90_max'];paired.append(row)
    write(OUT/'group_summary.csv',summaries);write(OUT/'within_group_fits.csv',fits);write(OUT/'within_leak_times.csv',paired)
    blue=[r for r in at750 if r['group']=='L2_smaller'];red=[r for r in at750 if r['group']=='L1_smaller'];assert len(blue)==17 and len(red)==8
    assert max(r['mu1'] for r in red)<min(r['mu1'] for r in blue)
    assert max(r['mu2'] for r in blue)<min(r['mu2'] for r in red)
    prod=sorted(at750,key=lambda r:r['product']);gaps=np.array([r['product'] for r in prod[1:]])/np.array([r['product'] for r in prod[:-1]]);j=int(np.argmax(gaps));low=prod[:j+1];high=prod[j+1:]
    product_gap=dict(selected_posthoc=True,lower_count=len(low),upper_count=len(high),lower_max=low[-1]['product'],upper_min=high[0]['product'],multiplicative_gap=float(gaps[j]),low_product_L1_group=[dict(a=r['a'],seed=r['seed']) for r in low if r['group']=='L1_smaller'])
    (OUT/'product_gap.json').write_text(json.dumps(product_gap,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for r in at750:
        c=COL[r['group']];old=next(s for s in allrows if s['a']==r['a'] and s['seed']==r['seed'] and s['step']==0)
        axes[0].plot(old['mu1'],old['mu2'],'o',ms=4,mfc='none',mec=c,alpha=.35)
        axes[0].annotate('',xy=(r['mu1'],r['mu2']),xytext=(old['mu1'],old['mu2']),arrowprops=dict(arrowstyle='->',color=c,alpha=.3,lw=.8))
        axes[0].scatter(r['mu1'],r['mu2'],c=c,s=38,edgecolors='white',linewidths=.3)
        axes[1].scatter(r['product'],r['T90_sustained'],c=c,s=38,edgecolors='white',linewidths=.3)
    lim=np.geomspace(.0008,.5,100);axes[0].plot(lim,lim,':',c='grey',lw=1)
    axes[0].set(xscale='log',yscale='log',xlabel="Layer 1 mean phi'",ylabel="Layer 2 mean phi'",xlim=(.0009,.25),ylim=(.0008,.6),title='Same membership at start and at 750\nOpen = start; filled = 750; arrows show movement')
    legend=[]
    for group,gg in [('L2_smaller',blue),('L1_smaller',red)]:
        f=next(r for r in fits if r['step']==750 and r['group']==group);x=np.geomspace(min(r['product'] for r in gg)*.9,max(r['product'] for r in gg)*1.1,100)
        axes[1].plot(x,np.exp(f['loglog_intercept'])*x**f['loglog_slope'],c=COL[group],lw=1.7)
        label=f'{group.replace("_"," ")} (n={len(gg)})';legend.append(Line2D([0],[0],marker='o',color=COL[group],lw=0,label=label))
    axes[0].legend(handles=legend,fontsize=9,loc='lower left')
    special=next(r for r in red if r['a']==.001 and r['seed']==4)
    axes[1].annotate('a=.001, seed 4\nL1 smaller despite low product',xy=(special['product'],special['T90_sustained']),xytext=(.000115,51000),fontsize=8,arrowprops=dict(arrowstyle='->',color=COL['L1_smaller']))
    axes[1].set(xscale='log',yscale='log',xlabel="Product of layer-mean phi' at 750",ylabel='Updates to sustained 90%',ylim=(7000,65000),title='Within L2-smaller group: slope=-0.51, R2=0.88\nWithin L1-smaller group: slope=-0.22, R2=0.32')
    for ax in axes:ax.grid(alpha=.14)
    fig.suptitle('Two layer configurations in the same 25 runs after 6 prior tasks (posthoc)',fontsize=12)
    fig.savefig(OUT/'two_groups_posthoc.png',dpi=180);fig.savefig(OUT/'two_groups_posthoc.pdf');plt.close(fig)
    lines=['# Two groups at update750 (posthoc)','',
           'Issa observed apparent polarization in the previous750-update plot. Inspect the existing25 source-age6 runs, retaining all five leaks and five paired seeds. This is a descriptive posthoc analysis, not a registered two-component test and not new training.',
           'Define group solely by which layer has the smaller input/unit mean derivative at750 (mu1<mu2 or mu2<mu1); do not use adaptation time in this assignment. There are8 L1-smaller runs and17 L2-smaller runs. This inequality always defines two sides; the evidence for separation is the unoccupied intervals and separated coordinate ranges, not the mere ability to label two groups.','',
           '| Configuration at750 | Count | L1 mean derivative range | L2 range | Median sustained90 time | Time range |','|---|---:|---:|---:|---:|---:|']
    for group in ['L2_smaller','L1_smaller']:
        r=next(r for r in summaries if r['step']==750 and r['group']==group)
        lines.append(f'| {group} | {r["n"]} | {r["mu1_min"]:.5f}-{r["mu1_max"]:.5f} | {r["mu2_min"]:.5f}-{r["mu2_max"]:.5f} | {r["T90_sustained_median"]:,.0f} | {r["T90_sustained_min"]:,.0f}-{r["T90_sustained_max"]:,.0f} |')
    lines+=['',
            'The ratio mu2/mu1 spans .0103-.2135 for L2-smaller runs and1.9774-323.4504 for L1-smaller runs: no observation between .2135 and1.9774. Both layer coordinates have non-overlapping ranges between groups. The two-dimensional derivative configurations are clearly separated in this finite sample.',
            'The raw product alone has its largest adjacent gap between .000674918 and .001410622, producing18 low-product and7 high-product observations. This differs from17/8: a=.001 seed4 is an L1-smaller run with a low product (.000546379) and very late time39900. Do not silently discard or relabel it.',
            'All25 group labels at750 already match their labels at update0 and stay the same at every stored derivative evaluation through6000 (including update1). The layer-ratio gap increases from source ranges .0997-.6472 versus1.8898-132.1754 to the750 ranges above. This is evidence for amplification of a pre-existing configuration difference; it does not locate a newly occurring transition at750.',
            'Pooled750 regression has slope+.1799 and log R2=.1389. Within L2-smaller17, slope=-.5075 and R2=.8751; within L1-smaller8, slope=-.2221 and R2=.3210. Thus an inverse trend survives within the main17-run group even though pooling reverses its sign. Its exponent is near-.5, not the source-state pooled exponent near-1. These are in-sample descriptive fits; no universal time law or causal mediation follows.','',
            '## Compare within each leak','',
            '| a | L2-smaller n | Median time | L1-smaller n | Median time | All L1-smaller times exceed all L2-smaller times? |','|---:|---:|---:|---:|---:|---|']
    for r in paired:lines.append(f'| {r["a"]:g} | {r["L2_smaller_n"]} | {r["L2_smaller_T90_median"]:,.1f} | {r["L1_smaller_n"]} | {r["L1_smaller_T90_median"]:,.1f} | {r["all_L1_group_slower"]} |')
    lines+=['',
            'The between-group time difference appears within every leak, rather than coming only from different leak proportions. Counts per cell are small and these are the same five paired seeds. Naming a configuration by its lower derivative does not prove that layer is the causal rate-limiting mechanism. Finite separation is not a thermodynamic or mathematical phase-transition demonstration.',
            'At750, the median input variance of derivatives in layer1 is .001412 in the L1-smaller group and .038461 in the L2-smaller group. Median fraction of inputs with no positive layer2 unit is0 in L1-smaller versus .97417 in L2-smaller. These differences are descriptions of different configurations, not a new causal intervention. All members and observer summaries are in the CSV files.','',
            '![Two groups](two_groups_posthoc.png)','']
    (OUT/'summary.md').write_text('\n'.join(lines))
    (OUT/'checks.json').write_text(json.dumps(dict(all_pass=True,all25_retained=True,groups={'L1_smaller':8,'L2_smaller':17},classification_checks=checks,changes_through6000=changes,all_five_leaks_L1_group_slower=all(r['all_L1_group_slower'] for r in paired)),indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(posthoc=True,new_training=False,source_run='leaky_order_1111',source_product_result_git='765f2ef722bfd8837811a274c4667a75fdfe0b10',analysis_code_sha256=sha(__file__),base_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),inputs=manifest),indent=2)+'\n')
    (OUT/'backup_manifest.json').write_text(json.dumps(dict(run='leaky_two_groups_1111',files=[],reason='No new raw training data; all derived files and input hashes are versioned. Original raw remains in leaky_order_1111 archive.'),indent=2)+'\n')
    print(json.dumps(dict(checks=checks,groups=[r for r in summaries if r['step']==750],product_gap=product_gap),indent=2))
if __name__=='__main__':main()
